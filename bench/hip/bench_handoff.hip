// EXP-0081b: can the CPU/GPU handoff be made cheaper than 33 us?
//
// A hybrid pays one handoff per layer: GPU finishes the fixed-weight part,
// CPU needs 2048 floats to run the router, GPU needs the expert sum back.
// 38 layers x 2 pageable copies was 47 us each = 1.8 ms of an 8.6 ms token.
//
// Three candidate fixes, each measured rather than assumed:
//   1. pinned staging buffers (already 33 us)
//   2. zero-copy: GPU reads/writes the host's pinned memory directly through a
//      device pointer, so there is NO explicit copy at all -- only the
//      implicit one the kernel does over PCIe
//   3. the same but transferring only what the CPU actually needs: the router
//      input is one 2048-float row; a fused kernel could write just that
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

__global__ void touch(float * p) { p[threadIdx.x] = p[threadIdx.x] * 1.0000001f; }
// reads the row from wherever it lives and writes back -- the per-layer traffic
__global__ void roundtrip(float * p, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) p[i] = p[i] * 1.0000001f + 1.0f;
}

int main() {
    const int n = 2048;
    const size_t bytes = n * sizeof(float);
    std::vector<float> src(n, 1.0f);

    float * d = nullptr;  CK(hipMalloc(&d, bytes));
    float * h_pin = nullptr; CK(hipHostMalloc((void **)&h_pin, bytes, hipHostMallocDefault));
    float * h_map = nullptr; CK(hipHostMalloc((void **)&h_map, bytes, hipHostMallocMapped | hipHostMallocCoherent));
    float * d_map = nullptr; CK(hipHostGetDevicePointer((void **)&d_map, h_map, 0));
    CK(hipMemcpy(d, src.data(), bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_map, src.data(), bytes, hipMemcpyHostToDevice));

    const int reps = 1000;
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto timed = [&](const char * name, auto body) {
        body(20); CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0)); body(reps); CK(hipEventRecord(e1)); CK(hipDeviceSynchronize());
        float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
        std::printf("%-46s %8.2f us/switch\n", name, ms * 1000.0 / reps);
    };

    timed("pinned: D2H + kernel + H2D", [&](int r) {
        for (int i = 0; i < r; ++i) {
            CK(hipMemcpy(h_pin, d, bytes, hipMemcpyDeviceToHost));
            CK(hipMemcpy(d, h_pin, bytes, hipMemcpyHostToDevice));
            roundtrip<<<1, 256>>>(d, n);
        }
    });
    timed("zero-copy: kernel on mapped host mem, no copy", [&](int r) {
        for (int i = 0; i < r; ++i) roundtrip<<<1, 256>>>(d_map, n);
    });
    timed("zero-copy: kernel + sync (what CPU pays)", [&](int r) {
        for (int i = 0; i < r; ++i) { roundtrip<<<1, 256>>>(d_map, n); CK(hipStreamSynchronize(0)); }
    });
    timed("pinned: single async H2D + kernel + sync", [&](int r) {
        for (int i = 0; i < r; ++i) {
            CK(hipMemcpyAsync(d, h_pin, bytes, hipMemcpyHostToDevice, 0));
            roundtrip<<<1, 256>>>(d, n);
            CK(hipMemcpyAsync(h_pin, d, bytes, hipMemcpyDeviceToHost, 0));
            CK(hipStreamSynchronize(0));
        }
    });
    // 40-layer dependent chain: the real per-token bill for each strategy
    timed("40-layer chain, pinned async/sync", [&](int r) {
        for (int i = 0; i < r; ++i)
            for (int L = 0; L < 40; ++L) {
                CK(hipMemcpyAsync(d, h_pin, bytes, hipMemcpyHostToDevice, 0));
                roundtrip<<<1, 256>>>(d, n);
                CK(hipMemcpyAsync(h_pin, d, bytes, hipMemcpyDeviceToHost, 0));
                CK(hipStreamSynchronize(0));
                h_pin[0] += 1.0f;
            }
    });
    timed("40-layer chain, zero-copy + sync", [&](int r) {
        for (int i = 0; i < r; ++i)
            for (int L = 0; L < 40; ++L) {
                roundtrip<<<1, 256>>>(d_map, n);
                CK(hipStreamSynchronize(0));
                h_map[0] += 1.0f;
            }
    });
    return 0;
}

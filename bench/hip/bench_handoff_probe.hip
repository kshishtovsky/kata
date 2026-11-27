// EXP-0081: cost of handing a decode token between GPU and CPU, per MoE layer.
//
// The hybrid this prices needs one handoff per layer: the GPU finishes the
// attention/fixed-weight part, the CPU needs its output to run the router and
// the experts, then the GPU needs the expert sum back. 38 layers x 2 handoffs
// per token -- if the round trip is 20 us that is 1.5 ms of an 8.6 ms token.
//
// Measures, for the two payload sizes that actually matter (2048 floats of
// residual stream and 4 KB control), pageable vs pinned pinned-in-advance:
//   * D2H then H2D with no kernel in between  (pure copy latency)
//   * a 1-thread kernel, then hipStreamSynchronize (wake-up latency)
//   * the same handoff inside a 40-step dependent chain (what a token pays)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

__global__ void touch(float * p) { p[threadIdx.x] = p[threadIdx.x] * 1.0000001f; }
__global__ void tiny(float * p) { if (threadIdx.x == 0) p[0] += 1.0f; }

static double now_us(hipEvent_t a, hipEvent_t b) {
    float ms = 0; CK(hipEventElapsedTime(&ms, a, b)); return ms * 1000.0;
}

int main() {
    int dev = 0; CK(hipGetDevice(&dev));
    hipDeviceProp_t p {}; CK(hipGetDeviceProperties(&p, dev));
    std::printf("%s, %d CUs\n\n", p.name, p.multiProcessorCount);

    const size_t n = 2048;                 // residual stream, f32
    const size_t bytes = n * sizeof(float);
    float * d = nullptr; CK(hipMalloc(&d, bytes));
    float * h_page = (float *)std::malloc(bytes);
    float * h_pin = nullptr; CK(hipHostMalloc((void **)&h_pin, bytes));
    std::vector<float> src(n, 1.0f);
    if (hipMemcpy(d, src.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) std::exit(1);

    const int reps = 2000;
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));

    auto bench = [&](const char * name, void (*body)(float *, float *, size_t, int)) {
        body(d, h_page, bytes, 20);                       // warm
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0));
        body(d, h_page, bytes, reps);
        CK(hipEventRecord(e1));
        CK(hipDeviceSynchronize());
        std::printf("%-42s %8.2f us/handoff\n", name, now_us(e0, e1) / reps);
    };

    // 1. pageable D2H + H2D, nothing else
    bench("pageable D2H+H2D (no kernel)", [](float * d, float * h, size_t b, int r) {
        for (int i = 0; i < r; ++i) {
            hipMemcpy(h, d, b, hipMemcpyDeviceToHost);
            hipMemcpy(d, h, b, hipMemcpyHostToDevice);
        }
    });
    // 2. pinned D2H + H2D
    {
        auto body = [](float * d, float * h, size_t b, int r) {
            for (int i = 0; i < r; ++i) {
                hipMemcpy(h, d, b, hipMemcpyDeviceToHost);
                hipMemcpy(d, h, b, hipMemcpyHostToDevice);
            }
        };
        // pinned arm needs its own host pointer; do it inline so the lambda stays simple
        body(d, h_pin, bytes, 20); CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0)); body(d, h_pin, bytes, reps); CK(hipEventRecord(e1));
        CK(hipDeviceSynchronize());
        std::printf("%-42s %8.2f us/handoff\n", "pinned D2H+H2D (no kernel)", now_us(e0, e1) / reps);
    }
    // 3. kernel + synchronize, no copy: pure wake-up
    bench("tiny kernel + stream sync", [](float * d, float *, size_t, int r) {
        for (int i = 0; i < r; ++i) { tiny<<<1, 32>>>(d); hipStreamSynchronize(0); }
    });
    // 4. full handoff: kernel, D2H, H2D -- what one layer switch costs
    bench("kernel + D2H + H2D (one switch)", [](float * d, float * h, size_t b, int r) {
        for (int i = 0; i < r; ++i) {
            touch<<<1, 256>>>(d);
            hipMemcpy(h, d, b, hipMemcpyDeviceToHost);
            hipMemcpy(d, h, b, hipMemcpyHostToDevice);
        }
    });
    // 5. a 40-step dependent chain: what a token pays
    bench("40-step switch chain (one token)", [](float * d, float * h, size_t b, int r) {
        hipMemcpy(h, d, b, hipMemcpyDeviceToHost);   // host works on h between switches
        for (int i = 0; i < r; ++i) {
            for (int L = 0; L < 40; ++L) {
                hipMemcpy(d, h, b, hipMemcpyHostToDevice);
                touch<<<1, 256>>>(d);
                hipMemcpy(h, d, b, hipMemcpyDeviceToHost);
                h[0] += 1.0f;                        // stand-in for the CPU's share
            }
        }
    });
    return 0;
}

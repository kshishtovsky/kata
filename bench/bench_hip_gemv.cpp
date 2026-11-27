// EXP-0072: q4_K GEMV on the GPU, checked against the CPU kernel.
//
// The point is not speed. It is to answer one question before any GPU backend
// work: does a HIP kernel reading the SAME blocks produce the SAME numbers as
// the CPU path? If it does not, everything downstream (attention, MoE, KV on
// VRAM) is built on sand. Speed is reported so the 764 GB/s read-only ceiling
// measured earlier can be compared against a real kernel rather than a probe.
//
// Layout reminder (include/kata/types.h): 144 bytes per 256 values.
//   16 bits d, 16 bits dmin, 12 bytes of 6-bit scale/min pairs, 128 bytes qs.
// Value v[j] for j in [0,256) lives in qs[32*(j/64) + (j%32)] as nibble
// (j%64) >= 32 ? high : low, scaled by d*sc[j/32] and offset by dmin*m[j/32].
//
// Build: hipcc -O3 bench/bench_hip_gemv.cpp -o bench_hip_gemv \
//          -I include build/libkata_core.a -lpthread
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "kata/gguf.h"
#include "kata/quant.h"
#include "kata/types.h"

namespace {

// Device copies of the two CPU helpers the kernel needs. They are duplicated on
// purpose: kata's CPU versions are plain host functions, and a shared header
// would need __host__ __device__ annotations throughout. When a real GPU backend
// lands these move into a kernel-side header with one implementation.
//
// f16 conversion uses the hardware path, which is IEEE-correct including
// subnormals -- the same property the CPU path needed a bugfix for (EXP-0006).
// __half is only defined when a header is pulled in, so convert by bit pattern
// rather than by type punning through a header-only type.
__device__ inline float f16_dev(unsigned short h) {
    const unsigned int s = (h & 0x8000u) << 16;
    const unsigned int e = (h >> 10) & 0x1fu;
    unsigned int m = h & 0x03ffu;
    unsigned int bits;
    if (e == 0) {
        // subnormal or zero: (mantissa / 1024) * 2^-14
        if (m == 0) { bits = s; }
        else {
            int shift = 0;
            while ((m & 0x0400u) == 0) { m <<= 1; ++shift; }
            m &= 0x03ffu;
            bits = s | ((unsigned int)(113 - shift) << 23) | (m << 13);
        }
    } else if (e == 31) {
        bits = s | 0x7f800000u | (m << 13);
    } else {
        bits = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    return __uint_as_float(bits);
}

__device__ inline void scale_min_dev(int j, const unsigned char * q, unsigned char * d,
                                     unsigned char * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (unsigned char)((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4));
        *m = (unsigned char)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

// 4 threads per super-block. Each thread owns ONE 32-byte group of qs, and
// that group holds two sub-blocks at once: qs[32i+l] carries value 32i+l in the
// low nibble and value 32i+32+l in the high nibble. The earlier 8-thread
// version had two lanes read the same 32 bytes to reach the two nibbles, which
// doubled the load traffic for no benefit.
//
// A 256-thread block covers 64 super-blocks: whole rows when nb <= 64 (the real
// case -- cols = 2048 gives nb = 8), or a slice of one row otherwise. Both
// reduce inside the block, so no atomics and no cross-block partials.
__global__ void gemv_q4_K_kernel(const kata::block_q4_K * __restrict__ w,
                                 const kata::block_q8_K * __restrict__ x,
                                 float * __restrict__ out,
                                 int rows, int nb) {
    const int lane = threadIdx.x & 3;      // which 32-byte group of qs
    const int group = threadIdx.x >> 2;    // 0..63
    const int nsb = nb * 8 / 2;            // super-blocks are one per 8 groups... see below
    (void)nsb;

    int row, b;
    if (nb <= 64) {
        row = group / nb;
        b = group % nb;
    } else {
        row = blockIdx.x;
        b = group;
    }
    const int rpb = nb <= 64 ? (64 / nb) : 1;
    if (nb <= 64) row += blockIdx.x * rpb;

    float acc = 0.0f;
    bool active = nb <= 64 ? (group < rpb * nb && row < rows) : (b < nb);

    if (active) {
        for (int bb = b; bb < nb; bb += (nb <= 64 ? nb : 64)) {
            const kata::block_q4_K & blk = w[(size_t)row * nb + bb];
            const kata::block_q8_K & ab = x[bb];

            const float d = f16_dev(blk.d) * ab.d;
            const float dmin = f16_dev(blk.dmin) * ab.d;

            // Two independent scale/min lookups per lane keep the integer pipes
            // busy while the nibble extraction runs; folding them costs nothing.
            unsigned char sc0, m0, sc1, m1;
            scale_min_dev(2 * lane + 0, blk.scales, &sc0, &m0);
            scale_min_dev(2 * lane + 1, blk.scales, &sc1, &m1);

            // 32 contiguous bytes, vector-loaded so the compiler emits 2x128-bit.
            uint4 v0 = *reinterpret_cast<const uint4 *>(blk.qs + 32 * lane);
            uint4 v1 = *reinterpret_cast<const uint4 *>(blk.qs + 32 * lane + 16);
            const unsigned int * qw = reinterpret_cast<const unsigned int *>(&v0);

            const int j = 64 * lane;
            int acc0 = 0, acc1 = 0, sy0 = 0, sy1 = 0;
            // 4 words per uint4 -> 8 words over the 32 bytes; each word is 4 bytes.
            for (int w_i = 0; w_i < 4; ++w_i) {
                const unsigned int word = qw[w_i];
                for (int byte = 0; byte < 4; ++byte) {
                    const int shift = 8 * byte;
                    const int lo = (word >> shift) & 0x0f;
                    const int hi = (word >> (shift + 4)) & 0x0f;
                    const int l = 4 * w_i + byte;
                    const int y0 = ab.qs[j + l];
                    const int y1 = ab.qs[j + 32 + l];
                    acc0 += lo * y0;
                    acc1 += hi * y1;
                    sy0 += y0;
                    sy1 += y1;
                }
            }
            const unsigned int * qw2 = reinterpret_cast<const unsigned int *>(&v1);
            for (int w_i = 0; w_i < 4; ++w_i) {
                const unsigned int word = qw2[w_i];
                for (int byte = 0; byte < 4; ++byte) {
                    const int shift = 8 * byte;
                    const int lo = (word >> shift) & 0x0f;
                    const int hi = (word >> (shift + 4)) & 0x0f;
                    const int l = 16 + 4 * w_i + byte;
                    const int y0 = ab.qs[j + l];
                    const int y1 = ab.qs[j + 32 + l];
                    acc0 += lo * y0;
                    acc1 += hi * y1;
                    sy0 += y0;
                    sy1 += y1;
                }
            }
            acc += d * (float)sc0 * (float)acc0 - dmin * (float)m0 * (float)sy0;
            acc += d * (float)sc1 * (float)acc1 - dmin * (float)m1 * (float)sy1;
            if (nb <= 64) break;
        }
    }

    // 4-lane tree reduce; lane 0 of each group holds the full super-block sum.
    acc += __shfl_down_sync(0xffffffffull, acc, 2);
    acc += __shfl_down_sync(0xffffffffull, acc, 1);

    __shared__ float gsum[64];
    if (lane == 0) gsum[group] = acc;
    __syncthreads();

    const int per_row = nb <= 64 ? rpb : 1;
    if (threadIdx.x < per_row) {
        const int r = threadIdx.x;
        float s = 0.0f;
        if (nb <= 64) {
            const int gr = blockIdx.x * rpb + r;
            if (gr >= rows) return;
            for (int i = 0; i < nb; ++i) s += gsum[r * nb + i];
            out[gr] = s;
        } else {
            for (int i = 0; i < 64; ++i) s += gsum[i];
            out[blockIdx.x] = s;
        }
    }
}

struct Event {
    hipEvent_t h{};
    Event() { hipEventCreate(&h); }
    ~Event() { hipEventDestroy(h); }
};

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [tensor_name]\n", argv[0]);
        return 2;
    }

    kata::GgufFile g;
    try {
        g.open(argv[1]);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "open: %s\n", e.what());
        return 2;
    }

    // Default: the biggest 2-D q4_K tensor in the file, which is at least a
    // realistic shape. 1-D tensors are norms; 3-D are expert stacks.
    // "--all": stream every q4_K tensor once, which is what one decode token
    // costs in weight traffic. One launch per tensor, summed. This is the number
    // that decides whether a GPU backend is worth building, so it is measured
    // rather than extrapolated from a single, L2-resident tensor.
    // "--all": stream the weights one decode token actually reads, in one timed
    // sweep. Two things this must not do, both of which produced absurd numbers
    // on the first attempt:
    //   * best-of-N per tensor -- a 13.6 MB tensor fits in the 80 MB Infinity
    //     Cache, so reps 2..N are cache hits and it reported 2485 GB/s against a
    //     764 GB/s ceiling;
    //   * all 64 experts -- a token activates 4, so charging all of them
    //     overstates the traffic by 16x.
    if (argc >= 3 && std::strcmp(argv[2], "--all") == 0) {
        const bool moe = argc >= 4 && std::strcmp(argv[3], "--moe") == 0;
        struct Item { const kata::GgufTensor * t; size_t cols, rows, nb; };
        std::vector<Item> items;
        size_t total_bytes = 0;
        for (const auto & c : g.tensors()) {
            if (c.type != 12 || c.dims.size() < 2) continue;
            size_t cc = c.dims[0], rr = 1;
            for (size_t d = 1; d < c.dims.size(); ++d) rr *= c.dims[d];
            if (cc % kata::QK_K != 0) continue;
            // Expert stacks are [cols][ffn][n_expert]: a token activates 4 of 64,
            // so only 4 experts' rows are read. Charging all 64 overstates the
            // traffic by 16x; charging one understates it by 4x.
            size_t experts = 1;
            if (c.dims.size() == 3) {
                experts = c.dims[2];
                if (!moe) rr = c.dims[1] * std::min<size_t>(4, experts);
                if (!moe) { /* rows below are recomputed from rr */ }
            }
            items.push_back({&c, cc, rr, cc / kata::QK_K});
            total_bytes += c.n_bytes / experts * (moe ? experts : std::min<size_t>(4, experts));
        }

        // One allocation per tensor, all resident at once, so the sweep below
        // measures VRAM reads and nothing else (the model is 14.5 GB).
        struct Res { kata::block_q4_K * w; kata::block_q8_K * x; float * out; dim3 gr; dim3 bl; size_t rows, nb; };
        std::vector<Res> res(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            const Item & it = items[i];
            const size_t n_bytes = it.rows * it.nb * sizeof(kata::block_q4_K);
            std::vector<kata::block_q8_K> hx(it.nb);
            std::vector<float> ha(it.cols);
            for (size_t k = 0; k < it.cols; ++k) ha[k] = std::sin(0.001f * (float)k) * 1.7f;
            kata::quantize_row_q8_K(ha.data(), hx.data(), it.cols);
            hipMalloc(&res[i].w, n_bytes);
            hipMalloc(&res[i].x, it.nb * sizeof(kata::block_q8_K));
            hipMalloc(&res[i].out, it.rows * sizeof(float));
            hipMemcpy(res[i].w, g.data(*it.t), n_bytes, hipMemcpyHostToDevice);
            hipMemcpy(res[i].x, hx.data(), it.nb * sizeof(kata::block_q8_K), hipMemcpyHostToDevice);
            const int rp = it.nb <= 64 ? (64 / (int)it.nb) : 1;
            res[i].gr = dim3(it.nb <= 64 ? (it.rows + rp - 1) / rp : it.rows);
            res[i].bl = dim3(256);
            res[i].rows = it.rows;
            res[i].nb = it.nb;
        }

        // One launch per tensor is a decode-shaped sweep; the launch overhead is
        // part of the honest number since the real engine issues them serially too.
        Event e0, e1;
        double best = 1e9;
        for (int rep = 0; rep < 6; ++rep) {
            // Rotate the start so no single rep is systematically warm on the tail.
            const size_t off = (size_t)rep * 37 % (items.size() ? items.size() : 1);
            hipEventRecord(e0.h);
            for (size_t k = 0; k < items.size(); ++k) {
                const size_t i = (k + off) % items.size();
                gemv_q4_K_kernel<<<res[i].gr, res[i].bl>>>(res[i].w, res[i].x, res[i].out,
                                                           (int)res[i].rows, (int)res[i].nb);
            }
            hipEventRecord(e1.h);
            hipEventSynchronize(e1.h);
            float ms = 0;
            hipEventElapsedTime(&ms, e0.h, e1.h);
            best = std::min(best, (double)ms);
        }
        const double gbs = total_bytes / (best * 1e6);
        std::printf("sweep%s: %zu tensors, %.1f MB, %.2f ms -> %.1f GB/s (ceiling 764.3 -> %.0f%%)\n",
                    moe ? " (all 64 experts)" : " (4 of 64 experts)", items.size(),
                    total_bytes / 1e6, best, gbs, 100.0 * gbs / 764.3);
        std::printf("  launches: %zu -> %.1f us each of the %.2f ms\n", items.size(),
                    1e3 * best / items.size(), best);
        for (auto & r : res) { hipFree(r.w); hipFree(r.x); hipFree(r.out); }
        return 0;
    }

    const kata::GgufTensor * t = nullptr;
    if (argc >= 3) {
        t = g.find(argv[2]);
        if (!t) { std::fprintf(stderr, "no tensor %s\n", argv[2]); return 2; }
    } else {
        size_t best = 0;
        for (const auto & c : g.tensors())
            if (c.dims.size() == 2 && c.type == 12 && c.n_bytes > best) { best = c.n_bytes; t = &c; }
        if (!t) { std::fprintf(stderr, "no 2-D q4_K tensor found\n"); return 2; }
    }

    const size_t cols = t->dims[0], rows = t->dims[1];
    std::printf("tensor %s: %zux%zu q4_K = %.1f MB\n", t->name.c_str(), rows, cols,
                t->n_bytes / 1e6);
    if (cols % kata::QK_K != 0) { std::fprintf(stderr, "cols not a multiple of 256\n"); return 2; }

    const auto * host_w = static_cast<const kata::block_q4_K *>(g.data(*t));
    const size_t nb = cols / kata::QK_K;

    // A deterministic activation row; any values work, correctness is what we
    // are checking, but a fixed pattern makes a rerun comparable.
    std::vector<float> act(cols);
    for (size_t i = 0; i < cols; ++i)
        act[i] = std::sin(0.001f * (float)i) * 1.7f + 0.3f * std::cos(0.01f * (float)i);
    std::vector<kata::block_q8_K> host_x(nb);
    kata::quantize_row_q8_K(act.data(), host_x.data(), cols);

    // CPU reference: the scalar path, so the comparison is kernel vs kernel.
    std::vector<float> cpu(rows);
    for (size_t r = 0; r < rows; ++r)
        cpu[r] = kata::vec_dot_q4_K_q8_K_ref(cols, host_w + r * nb, host_x.data());

    kata::block_q4_K * dev_w = nullptr;
    kata::block_q8_K * dev_x = nullptr;
    float * dev_out = nullptr;
    hipMalloc(&dev_w, t->n_bytes);
    hipMalloc(&dev_x, nb * sizeof(kata::block_q8_K));
    hipMalloc(&dev_out, rows * sizeof(float));
    hipMemcpy(dev_w, host_w, t->n_bytes, hipMemcpyHostToDevice);
    hipMemcpy(dev_x, host_x.data(), nb * sizeof(kata::block_q8_K), hipMemcpyHostToDevice);

    std::vector<float> gpu(rows);
    const int rpb = nb <= 64 ? (64 / (int)nb) : 1;
    const dim3 grid(nb <= 64 ? (rows + rpb - 1) / rpb : rows), block(256);
    gemv_q4_K_kernel<<<grid, block>>>(dev_w, dev_x, dev_out, (int)rows, (int)nb);
    const auto err = hipDeviceSynchronize();
    if (err != hipSuccess) { std::fprintf(stderr, "kernel: %s\n", hipGetErrorString(err)); return 2; }
    hipMemcpy(gpu.data(), dev_out, rows * sizeof(float), hipMemcpyDeviceToHost);

    // Correctness. A float sum over 256 values per block in a different order is
    // not bit-exact, so compare relative to the row's magnitude.
    // Per-row relative error is meaningless for rows whose dot product is near
    // zero, so the gate is absolute, scaled by the largest |row| in the output.
    double max_abs = 0, max_row = 0;
    for (size_t r = 0; r < rows; ++r) {
        max_abs = std::max(max_abs, std::fabs((double)gpu[r] - (double)cpu[r]));
        max_row = std::max(max_row, std::fabs((double)cpu[r]));
    }
    const double tol = 1e-4 * max_row;
    std::printf("accuracy: max|abs| %.3e  largest |row| %.3f  tol %.3e -> %s\n",
                max_abs, max_row, tol, max_abs <= tol ? "OK" : "FAIL");
    const size_t bad = max_abs <= tol ? 0 : 1;

    // Timing, best of several runs; the weight array stays in VRAM so this is
    // the resident-weights case a real decode would run in.
    Event e0, e1;
    double best = 1e9;
    for (int r = 0; r < 20; ++r) {
        hipEventRecord(e0.h);
        gemv_q4_K_kernel<<<grid, block>>>(dev_w, dev_x, dev_out, (int)rows, (int)nb);
        hipEventRecord(e1.h);
        hipEventSynchronize(e1.h);
        float ms = 0;
        hipEventElapsedTime(&ms, e0.h, e1.h);
        best = std::min(best, (double)ms);
    }
    const double gbs = t->n_bytes / (best * 1e6);
    std::printf("gpu: %.3f ms  %.1f GB/s  (read-only ceiling 764.3 GB/s -> %.0f%%)\n",
                best, gbs, 100.0 * gbs / 764.3);

    // CPU for the same shape, single thread, so the ratio is per-core.
    std::vector<float> tmp(rows);
    const auto c0 = std::chrono::steady_clock::now();
    for (size_t r = 0; r < rows; ++r) tmp[r] = kata::vec_dot_q4_K_q8_K_ref(cols, host_w + r * nb, host_x.data());
    const auto c1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    std::printf("cpu (1 thread, scalar ref): %.1f ms -> %.1fx slower than gpu\n",
                cpu_ms, cpu_ms / best);

    hipFree(dev_w); hipFree(dev_x); hipFree(dev_out);
    return bad == 0 ? 0 : 1;
}
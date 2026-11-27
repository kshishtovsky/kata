// EXP-0084: does the resident weight stream amortise over a batch on the GPU?
//
// EXP-0083 measured one token's whole weight stream at 3.241 ms / 468 GB/s with
// 5 launches. At B=1 that is a memory-bound GEMV. With B activation rows the same
// weight bytes serve B tokens, so the memory cost per token should fall by up to
// Bx -- but only if the kernel is still memory-bound, and this kernel is scalar
// (no VNNI), so compute may take over first. That crossover is the question the
// whole GPU plan rests on: 182 tok/s for one chat is a latency win, but only
// amortisation makes the card useful for several agents.
//
// Method: take the bulk q4_K bucket (every q4_K row a token reads, 808448 rows,
// nb=8, 931 MB), pin it in VRAM, and run the same dot product for B activation
// rows inside one block so each weight block is loaded exactly once and used B
// times. Report achieved weight bandwidth and MAC/s for B = 1,2,4,8,16.
//
// Build: hipcc -O3 bench/hip/bench_gpu_batch.hip -o /tmp/gpb -I include \
//              -L build -lkata_core -lpthread
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kata/gguf.h"
#include "kata/quant.h"
#include "kata/types.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {
static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

__device__ inline float f16_dev(unsigned short h) {
    const unsigned int s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu;
    unsigned int m = h & 0x03ffu, bits;
    if (e == 0) { if (m == 0) bits = s; else { int sh = 0; while ((m & 0x0400u) == 0) { m <<= 1; ++sh; } m &= 0x03ffu; bits = s | ((unsigned)(113 - sh) << 23) | (m << 13); } }
    else if (e == 31) bits = s | 0x7f800000u | (m << 13);
    else bits = s | ((e - 15 + 127) << 23) | (m << 13);
    return __uint_as_float(bits);
}
__device__ inline void scale_min_dev(int j, const unsigned char * q, unsigned char * d, unsigned char * m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (unsigned char)((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4)); *m = (unsigned char)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4)); }
}

// B activation rows per weight block. `B` is a template parameter so the
// accumulators stay in registers -- a runtime bound over float acc[16] would
// spill to local memory and measure the L1, not the card.
template <int B>
__global__ void gemv_q4_K_batch_b(const kata::block_q4_K * __restrict__ w,
                                  const kata::block_q8_K * __restrict__ x,   // [B][nb]
                                  float * __restrict__ out,                  // [B][rows]
                                  int rows, int nb) {
    const int rpb = nb <= 64 ? (64 / nb) : 1;
    const int grid_row = blockIdx.x * rpb;
    const int lane = threadIdx.x & 3;
    const int group = threadIdx.x >> 2;
    const int r = group / nb;
    const int b = group % nb;
    float acc[B];
#pragma unroll
    for (int i = 0; i < B; ++i) acc[i] = 0.0f;
    if (r < rpb && grid_row + r < rows) {
        const kata::block_q4_K * base = w + (size_t)(grid_row + r) * nb;
        for (int bb = b; bb < nb; bb += (nb <= 64 ? nb : 64)) {
            const kata::block_q4_K & blk = base[bb];
            const float d = f16_dev(blk.d);
            const float dmin = f16_dev(blk.dmin);
            unsigned char sc0, m0, sc1, m1;
            scale_min_dev(2 * lane + 0, blk.scales, &sc0, &m0);
            scale_min_dev(2 * lane + 1, blk.scales, &sc1, &m1);
            uint4 v0 = *reinterpret_cast<const uint4 *>(blk.qs + 32 * lane);
            uint4 v1 = *reinterpret_cast<const uint4 *>(blk.qs + 32 * lane + 16);
            const unsigned int * qw = reinterpret_cast<const unsigned int *>(&v0);
            const int j = 64 * lane;
            // The min term needs sum(y) over the group, which does not depend on
            // the weight block at all -- block_q8_K already carries those sums.
            // Accumulating sy inside the loop cost 2 registers per activation row
            // and pushed B=16 into local memory.
#pragma unroll
            for (int t = 0; t < B; ++t) {
                const kata::block_q8_K & ab = x[(size_t)t * nb + bb];
                const float da = d * ab.d;
                const float dmina = dmin * ab.d;
                const int sy0 = ab.bsums[4 * lane] + ab.bsums[4 * lane + 1];
                const int sy1 = ab.bsums[4 * lane + 2] + ab.bsums[4 * lane + 3];
                int a0 = 0, a1 = 0;
                // Unpack four nibbles, then spend them on every activation row
                // before touching the next word: only 8 integers stay live, so the
                // accumulators keep their registers at B up to 16.
                for (int wi = 0; wi < 8; ++wi) {
                    const unsigned int word = (wi < 4 ? qw[wi] : reinterpret_cast<const unsigned int *>(&v1)[wi - 4]);
                    const int base_idx = 4 * wi;
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        const int shift = 8 * byte;
                        const int l = base_idx + byte;
                        const int lo = (word >> shift) & 0x0f;
                        const int hi = (word >> (shift + 4)) & 0x0f;
                        a0 += lo * ab.qs[j + l];
                        a1 += hi * ab.qs[j + 32 + l];
                    }
                }
                acc[t] += da * (float)sc0 * (float)a0 - dmina * (float)m0 * (float)sy0;
                acc[t] += da * (float)sc1 * (float)a1 - dmina * (float)m1 * (float)sy1;
            }
        }
    }
    __shared__ float gsum[64][B];
    __shared__ float red[256];
#pragma unroll
    for (int t = 0; t < B; ++t) {
        float v = acc[t];
        v += __shfl_down_sync(0xffffffffull, v, 2);
        v += __shfl_down_sync(0xffffffffull, v, 1);
        if (lane == 0) gsum[group][t] = v;
    }
    __syncthreads();
    // reduction is serial per thread here: it is the same 8-64 adds the B=1
    // kernel does, so it does not distort the batch comparison.
    if (threadIdx.x < rpb && grid_row + threadIdx.x < rows) {
#pragma unroll
        for (int t = 0; t < B; ++t) {
            float s = 0.0f;
            for (int i = 0; i < (nb < 64 ? nb : 64); ++i) s += gsum[threadIdx.x * nb + i][t];
            out[(size_t)t * rows + grid_row + threadIdx.x] = s;
        }
        (void)red;
    }
}
// v_dot4_i32_i8 variant: gfx1100 computes four signed-byte MACs per instruction,
// so the 3270 GMAC/s scalar plateau (EXP-0084) should move ~4x. q4_K's nibbles are
// 0..15 and fit i8, so `lo = w & 0x0f0f0f0f` and `hi = (w >> 4) & 0x0f0f0f0f` turn
// one word into two packed int8 groups of four.
__device__ inline int sdot4(unsigned int a, unsigned int b, int acc) {
    int r;
    __asm__ volatile("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(r) : "v"((int)a), "v"((int)b), "v"(acc));
    return r;
}

template <int B>
__global__ void gemv_q4_K_batch_sdot(const kata::block_q4_K * __restrict__ w,
                                     const kata::block_q8_K * __restrict__ x,
                                     float * __restrict__ out,
                                     int rows, int nb) {
    const int rpb = nb <= 64 ? (64 / nb) : 1;
    const int grid_row = blockIdx.x * rpb;
    const int lane = threadIdx.x & 3;
    const int group = threadIdx.x >> 2;
    const int r = group / nb;
    const int b = group % nb;
    float acc[B];
#pragma unroll
    for (int i = 0; i < B; ++i) acc[i] = 0.0f;
    if (r < rpb && grid_row + r < rows) {
        const kata::block_q4_K * base = w + (size_t)(grid_row + r) * nb;
        for (int bb = b; bb < nb; bb += (nb <= 64 ? nb : 64)) {
            const kata::block_q4_K & blk = base[bb];
            const float d = f16_dev(blk.d);
            const float dmin = f16_dev(blk.dmin);
            unsigned char sc0, m0, sc1, m1;
            scale_min_dev(2 * lane + 0, blk.scales, &sc0, &m0);
            scale_min_dev(2 * lane + 1, blk.scales, &sc1, &m1);
            const unsigned int * qw = reinterpret_cast<const unsigned int *>(blk.qs + 32 * lane);
            const int j = 64 * lane;
            int ilo[B], ihi[B];
#pragma unroll
            for (int t = 0; t < B; ++t) { ilo[t] = 0; ihi[t] = 0; }
            // Two 128-bit loads per group per row. Eight separate 4-byte reads per
            // group cost 2.3x at B=1 (491 vs 722 GB/s): the byte loop it replaced
            // was auto-vectorised, 32 scalar loads are not. The instruction itself
            // is fine -- 12.1 T MAC/s isolated against imad's 3.3.
            unsigned int y0w[B][8], y1w[B][8];
#pragma unroll
            for (int t = 0; t < B; ++t) {
                const kata::block_q8_K & ab = x[(size_t)t * nb + bb];
                const uint4 a0 = *reinterpret_cast<const uint4 *>(ab.qs + j);
                const uint4 a1 = *reinterpret_cast<const uint4 *>(ab.qs + j + 16);
                const uint4 b0 = *reinterpret_cast<const uint4 *>(ab.qs + j + 32);
                const uint4 b1 = *reinterpret_cast<const uint4 *>(ab.qs + j + 48);
                const unsigned int * p0 = reinterpret_cast<const unsigned int *>(&a0);
                const unsigned int * p1 = reinterpret_cast<const unsigned int *>(&a1);
                const unsigned int * q0 = reinterpret_cast<const unsigned int *>(&b0);
                const unsigned int * q1 = reinterpret_cast<const unsigned int *>(&b1);
#pragma unroll
                for (int w4 = 0; w4 < 4; ++w4) {
                    y0w[t][w4] = p0[w4]; y0w[t][w4 + 4] = p1[w4];
                    y1w[t][w4] = q0[w4]; y1w[t][w4 + 4] = q1[w4];
                }
            }
#pragma unroll
            for (int wi = 0; wi < 8; ++wi) {
                const unsigned int word = qw[wi];
                const unsigned int lo = word & 0x0f0f0f0fu;
                const unsigned int hi = (word >> 4) & 0x0f0f0f0fu;
#pragma unroll
                for (int t = 0; t < B; ++t) {
                    ilo[t] = sdot4(lo, y0w[t][wi], ilo[t]);
                    ihi[t] = sdot4(hi, y1w[t][wi], ihi[t]);
                }
            }
#pragma unroll
            for (int t = 0; t < B; ++t) {
                const kata::block_q8_K & ab = x[(size_t)t * nb + bb];
                const float da = d * ab.d;
                const float dmina = dmin * ab.d;
                const int sy0 = ab.bsums[4 * lane] + ab.bsums[4 * lane + 1];
                const int sy1 = ab.bsums[4 * lane + 2] + ab.bsums[4 * lane + 3];
                acc[t] += da * (float)sc0 * (float)ilo[t] - dmina * (float)m0 * (float)sy0;
                acc[t] += da * (float)sc1 * (float)ihi[t] - dmina * (float)m1 * (float)sy1;
            }
        }
    }
    __shared__ float gsum[64][B];
#pragma unroll
    for (int t = 0; t < B; ++t) {
        float v = acc[t];
        v += __shfl_down_sync(0xffffffffull, v, 2);
        v += __shfl_down_sync(0xffffffffull, v, 1);
        if (lane == 0) gsum[group][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < rpb && grid_row + threadIdx.x < rows) {
#pragma unroll
        for (int t = 0; t < B; ++t) {
            float s = 0.0f;
            for (int i = 0; i < (nb < 64 ? nb : 64); ++i) s += gsum[threadIdx.x * nb + i][t];
            out[(size_t)t * rows + grid_row + threadIdx.x] = s;
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("usage: %s model.gguf [n_used_experts]\n", argv[0]); return 2; }
    const int n_used = argc > 2 ? std::atoi(argv[2]) : 4;
    kata::GgufFile g; g.open(argv[1]);
    int n_expert = 0;
    if (const auto * e = g.find("blk.2.ffn_gate_exps.weight")) n_expert = (int)e->dims[2];

    // The bulk bucket: every q4_K row one token reads, nb == 8, so the whole
    // bucket is a single shape and the grid arithmetic stays trivial.
    std::vector<const kata::GgufTensor *> picked;
    size_t rows = 0, bytes = 0;
    for (const auto & t : g.tensors()) {
        if (t.type != 12) continue;
        if ((int)(t.dims[0] / kata::QK_K) != 8) continue;
        const bool is_exp = t.name.find("_exps.") != std::string::npos;
        const size_t r_total = t.n_elements / t.dims[0];
        const size_t r = r_total / (size_t)(is_exp ? n_expert : 1);
        picked.push_back(&t);
        rows += r * (size_t)(is_exp ? n_used : 1);
        bytes += r * (t.n_bytes / r_total) * (size_t)(is_exp ? n_used : 1);
    }
    if (picked.empty()) { std::printf("no q4_K nb=8 tensors\n"); return 1; }
    const int nb = 8;
    std::printf("bucket: %zu tensors, %zu rows, %.1f MB\n", picked.size(), rows, bytes / 1e6);

    // Lay every picked slab's first `n_used` expert blocks out contiguously.
    std::vector<unsigned char> blob;
    blob.reserve(bytes);
    for (const auto * t : picked) {
        const bool is_exp = t->name.find("_exps.") != std::string::npos;
        const size_t r_total = t->n_elements / t->dims[0];
        const size_t r = r_total / (size_t)(is_exp ? n_expert : 1);
        const size_t row_b = t->n_bytes / r_total;
        const size_t slabs = (size_t)(is_exp ? n_used : 1);
        const auto * src = (const unsigned char *)g.data(*t);
        for (size_t s = 0; s < slabs; ++s) blob.insert(blob.end(), src + s * r * row_b, src + (s + 1) * r * row_b);
    }
    bytes = blob.size();

    void * dw = nullptr;
    CK(hipMalloc(&dw, bytes));
    CK(hipMemcpy(dw, blob.data(), bytes, hipMemcpyHostToDevice));
    std::vector<unsigned char>().swap(blob);

    const int BMAX = 16;
    std::vector<kata::block_q8_K> hact((size_t)BMAX * nb);
    std::vector<float> hv((size_t)nb * kata::QK_K);
    for (int b = 0; b < BMAX; ++b) {
        for (size_t i = 0; i < hv.size(); ++i)
            hv[i] = std::sin(0.001f * (float)(i + 37 * b)) * 1.7f;
        kata::quantize_row_q8_K(hv.data(), hact.data() + (size_t)b * nb, hv.size());
    }
    kata::block_q8_K * dact = nullptr;
    CK(hipMalloc(&dact, hact.size() * sizeof(kata::block_q8_K)));
    CK(hipMemcpy(dact, hact.data(), hact.size() * sizeof(kata::block_q8_K), hipMemcpyHostToDevice));
    float * dout = nullptr;
    CK(hipMalloc(&dout, (size_t)BMAX * rows * sizeof(float)));

    // Verify before timing: a bandwidth figure above the card's own ceiling means
    // the kernel is not reading what it claims, and a wrong row mapping runs fast.
    {
        const dim3 bl(256), gr((unsigned)((rows + 7) / 8));
        gemv_q4_K_batch_b<1><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb);
        CK(hipDeviceSynchronize());
        std::vector<float> got(rows);
        CK(hipMemcpy(got.data(), dout, rows * sizeof(float), hipMemcpyDeviceToHost));
        // host copy of the first bucket rows for the reference
        double mx = 0, sc = 0;
        std::vector<kata::block_q4_K> hrows(rows * nb);
        CK(hipMemcpy(hrows.data(), dw, rows * nb * sizeof(kata::block_q4_K), hipMemcpyDeviceToHost));
        for (int r = 0; r < 2000; ++r) {
            const double ref = kata::vec_dot_q4_K_q8_K_ref(nb * kata::QK_K, hrows.data() + (size_t)r * nb, hact.data());
            mx = std::max(mx, std::fabs(ref - (double)got[(size_t)r]));
            sc = std::max(sc, std::fabs(ref));
        }
        std::printf("verify B=1 first 2000 rows: max|diff| %.3e on %.3e (%.1e rel) %s\n",
                    mx, sc, mx / sc, mx / sc < 1e-4 ? "PASS" : "FAIL");
        if (!(mx / sc < 1e-4)) return 1;
    }

    // The sdot4 arm needs its own gate: a wrong nibble mapping still runs fast.
    {
        const dim3 bl(256), gr((unsigned)((rows + 7) / 8));
        gemv_q4_K_batch_sdot<1><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb);
        CK(hipDeviceSynchronize());
        std::vector<float> got(rows);
        CK(hipMemcpy(got.data(), dout, rows * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<kata::block_q4_K> hrows(rows * nb);
        CK(hipMemcpy(hrows.data(), dw, rows * nb * sizeof(kata::block_q4_K), hipMemcpyDeviceToHost));
        double mx = 0, sc = 0;
        for (int r = 0; r < 2000; ++r) {
            const double ref = kata::vec_dot_q4_K_q8_K_ref(nb * kata::QK_K, hrows.data() + (size_t)r * nb, hact.data());
            mx = std::max(mx, std::fabs(ref - (double)got[(size_t)r]));
            sc = std::max(sc, std::fabs(ref));
        }
        std::printf("verify sdot4 B=1 first 2000 rows: max|diff| %.3e on %.3e (%.1e rel) %s\n",
                    mx, sc, mx / sc, mx / sc < 1e-4 ? "PASS" : "FAIL");
        if (!(mx / sc < 1e-4)) return 1;
    }

    const double macs = (double)rows * nb * kata::QK_K;   // MACs per activation row
    std::printf("\n%3s %10s %9s %10s %10s %9s\n", "B", "ms", "GB/s w", "GMAC/s", "ns/token", "vs B=1");
    double base_ns = 0;
    for (int B : {1, 2, 4, 8, 16}) {
        const dim3 bl(256);
        const dim3 gr((unsigned)((rows + 7) / 8));
        auto run = [&] {
            switch (B) {
                case 1: gemv_q4_K_batch_b<1><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb); break;
                case 2: gemv_q4_K_batch_b<2><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb); break;
                case 4: gemv_q4_K_batch_b<4><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb); break;
                case 8: gemv_q4_K_batch_b<8><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb); break;
                default: gemv_q4_K_batch_b<16><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb); break;
            }
            CK(hipDeviceSynchronize());
        };
        run();
        const int reps = 3;
        const double a = now_ms();
        for (int i = 0; i < reps; ++i) run();
        const double ms = (now_ms() - a) / reps;
        const double ns_per_token = ms * 1e6 / B;
        if (B == 1) base_ns = ns_per_token;
        std::printf("%3d %10.3f %9.1f %10.1f %10.1f %9.2fx\n", B, ms, bytes / 1e6 / ms,
                    macs * B / ms / 1e6, ns_per_token, base_ns / ns_per_token);
        // same B with v_dot4_i32_i8
        auto run_sd = [&] {
            const dim3 bl(256), gr((unsigned)((rows + 7) / 8));
            switch (B) {
                case 1: gemv_q4_K_batch_sdot<1><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb); break;
                case 2: gemv_q4_K_batch_sdot<2><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb); break;
                case 4: gemv_q4_K_batch_sdot<4><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb); break;
                case 8: gemv_q4_K_batch_sdot<8><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb); break;
                default: gemv_q4_K_batch_sdot<16><<<gr, bl>>>((const kata::block_q4_K *)dw, dact, dout, (int)rows, nb); break;
            }
            CK(hipDeviceSynchronize());
        };
        run_sd();
        const double a2 = now_ms();
        for (int i = 0; i < reps; ++i) run_sd();
        const double ms2 = (now_ms() - a2) / reps;
        const double ns2 = ms2 * 1e6 / B;
        std::printf("%3s %10.3f %9.1f %10.1f %10.1f %9.2fx\n", "sdot", ms2, bytes / 1e6 / ms2,
                    macs * B / ms2 / 1e6, ns2, base_ns / ns2);
    }

    // Compute ceiling of this scalar kernel: same shape with the weight block
    // reused B times, so the MAC rate above IS the ceiling at large B. Print the
    // implied per-token weight time floor for the full token (1517.6 MB, EXP-0083).
    std::printf("\nfull-token projection (1517.6 MB weights, EXP-0083):\n");
    std::printf("  B=1 3.241 ms measured -> 309 tok/s; batch amortises only while\n"
                "  the kernel stays memory-bound, i.e. while GB/s above still climbs\n"
                "  with B. Where GMAC/s flattens, compute owns the token.\n");
    return 0;
}
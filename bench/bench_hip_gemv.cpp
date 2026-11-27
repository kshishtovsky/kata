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

    // Two layouts, both reduced inside the block:
    //   nb <= 64  a block covers rpb whole rows, group g owns (g/nb, g%nb)
    //   nb >  64  a block covers ONE row, group g owns super-blocks g, g+64, ...
    const int rpb = nb <= 64 ? (64 / nb) : 1;
    int row, b;
    if (nb <= 64) {
        row = blockIdx.x * rpb + group / nb;
        b = group % nb;
    } else {
        row = blockIdx.x;
        b = group;
    }

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


// q6_K: 210 bytes per 256 values -- ql[128], qh[64], scales[16] (int8), d (f16).
// Layout per the reference (src/quant.cpp:38): two halves of 128 values; within
// a half, byte l holds sub-block scales that alternate every 16 values, so
// lane*4/16 picks a single is per lane and the four q-slots read sc[is+0],
// sc[is+2], sc[is+4], sc[is+6].
//
// 8 threads per super-block, each owning 4 consecutive l values in both halves
// (16 values per half, 32 per super-block). 32 super-blocks per 256-thread
// block, so nb <= 32 packs whole rows -- cols = 2048 gives nb = 8, i.e. 4 rows.
//
// Kept in integers per scale group: q in [-32,31] and y in [-127,127] make a
// 4-term accumulation fit in int comfortably, and the scale multiply happens
// once per group instead of once per value. That is the same trick that made
// the CPU VNNI version work (EXP-0029) after the naive form failed.
__global__ void gemv_q6_K_kernel(const kata::block_q6_K * __restrict__ w,
                                 const kata::block_q8_K * __restrict__ x,
                                 float * __restrict__ out,
                                 int rows, int nb) {
    const int lane = threadIdx.x & 7;      // 8 threads per super-block
    const int group = threadIdx.x >> 3;    // 0..31
    const int rpb = nb <= 32 ? (32 / nb) : 1;

    int row, b;
    if (nb <= 32) {
        row = blockIdx.x * rpb + group / nb;
        b = group % nb;
    } else {
        row = blockIdx.x;
        b = group;
    }
    const bool active = nb <= 32 ? (group < rpb * nb && row < rows) : (b < nb);

    float acc = 0.0f;
    if (active) {
        for (int bb = b; bb < nb; bb += (nb <= 32 ? nb : 32)) {
            const kata::block_q6_K & blk = w[(size_t)row * nb + bb];
            const kata::block_q8_K & ab = x[bb];
            const float d = f16_dev(blk.d) * ab.d;

            const int is = lane >> 2;          // 0 or 1, the 16-value sub-block pair
            const int l0 = lane * 4;
            float blk_sum = 0.0f;

            for (int half = 0; half < 2; ++half) {
                const unsigned char * ql = blk.ql + half * 64;
                const unsigned char * qh = blk.qh + half * 32;
                const signed char * q8 = ab.qs + half * 128;
                int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                for (int k = 0; k < 4; ++k) {
                    const int l = l0 + k;
                    const unsigned char qlb = ql[l];
                    const unsigned char qhb = qh[l];
                    const int q1 = (int)((qlb & 0x0f) | (((qhb >> 0) & 3) << 4)) - 32;
                    const int q2 = (int)((ql[l + 32] & 0x0f) | (((qhb >> 2) & 3) << 4)) - 32;
                    const int q3 = (int)((qlb >> 4) | (((qhb >> 4) & 3) << 4)) - 32;
                    const int q4 = (int)((ql[l + 32] >> 4) | (((qhb >> 6) & 3) << 4)) - 32;
                    a0 += q1 * (int)q8[l + 0];
                    a1 += q2 * (int)q8[l + 32];
                    a2 += q3 * (int)q8[l + 64];
                    a3 += q4 * (int)q8[l + 96];
                }
                const signed char * sc = blk.scales + half * 8;
                blk_sum += (float)sc[is + 0] * (float)a0 + (float)sc[is + 2] * (float)a1
                         + (float)sc[is + 4] * (float)a2 + (float)sc[is + 6] * (float)a3;
            }
            acc += d * blk_sum;
            if (nb <= 32) break;
        }
    }

    for (int off = 4; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffull, acc, off);

    __shared__ float gsum6[32];
    if (lane == 0) gsum6[group] = acc;
    __syncthreads();

    if (threadIdx.x < (nb <= 32 ? rpb : 1)) {
        const int r = threadIdx.x;
        float s = 0.0f;
        if (nb <= 32) {
            const int gr = blockIdx.x * rpb + r;
            if (gr >= rows) return;
            for (int i = 0; i < nb; ++i) s += gsum6[r * nb + i];
            out[gr] = s;
        } else {
            for (int i = 0; i < 32; ++i) s += gsum6[i];
            out[blockIdx.x] = s;
        }
    }
}

// Ground truth in double. Comparing two float implementations cannot say which
// is right -- an early gate at 1e-4 relative flagged 11 rows and was wrong: the
// GPU was CLOSER to this reference than the CPU arm on those rows. Rows whose
// dot product nearly cancels are where float order shows up, and only a higher
// precision reference can separate "different rounding" from "wrong".
double vec_dot_q4_K_q8_K_double(size_t n, const kata::block_q4_K * x, const kata::block_q8_K * y) {
    double sum = 0.0;
    for (size_t b = 0; b < n / kata::QK_K; ++b) {
        const double d = (double)kata::f16_to_f32(x[b].d) * y[b].d;
        const double dmin = (double)kata::f16_to_f32(x[b].dmin) * y[b].d;
        int is = 0;
        const uint8_t * q = x[b].qs;
        for (int j = 0; j < kata::QK_K; j += 64) {
            uint8_t sc, m;
            long long a1 = 0, a2 = 0, s1 = 0, s2 = 0;
            kata::q4_K_scale_min(is + 0, x[b].scales, &sc, &m);
            for (int l = 0; l < 32; ++l) {
                a1 += (long long)(q[l] & 0x0f) * y[b].qs[j + l];
                a2 += (long long)(q[l] >> 4) * y[b].qs[j + 32 + l];
                s1 += y[b].qs[j + l];
                s2 += y[b].qs[j + 32 + l];
            }
            sum += d * (double)sc * (double)a1 - dmin * (double)m * (double)s1;
            kata::q4_K_scale_min(is + 1, x[b].scales, &sc, &m);
            sum += d * (double)sc * (double)a2 - dmin * (double)m * (double)s2;
            q += 32;
            is += 2;
        }
    }
    return sum;
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
    // "--all": stream the weights one decode token actually reads, in one timed
    // sweep, both formats. This is the number that decides whether a GPU backend
    // is worth building, so it is measured rather than extrapolated.
    //
    // Two traps, both of which produced absurd numbers on the first attempt:
    //   * best-of-N per tensor -- a 13.6 MB tensor fits in the 80 MB Infinity
    //     Cache, so reps 2..N are cache hits and it reported 2485 GB/s against a
    //     764 GB/s ceiling. The sweep times every tensor once per rep and rotates
    //     the start offset.
    //   * all 64 experts -- a token activates 4, so charging all of them
    //     overstates the traffic by 16x, and charging 1 understates it by 4x.
    if (argc >= 3 && std::strcmp(argv[2], "--all") == 0) {
        const bool moe = argc >= 4 && std::strcmp(argv[3], "--moe") == 0;
        struct Item { const kata::GgufTensor * t; size_t cols, rows, nb; uint32_t type; };
        std::vector<Item> items;
        double bytes_q4 = 0, bytes_q6 = 0, total_bytes = 0;
        for (const auto & c : g.tensors()) {
            if ((c.type != 12 && c.type != 14) || c.dims.size() < 2) continue;
            const size_t cc = c.dims[0];
            if (cc % kata::QK_K != 0) continue;
            size_t rr = 1;
            for (size_t d = 1; d < c.dims.size(); ++d) rr *= c.dims[d];
            size_t experts = 1;
            if (c.dims.size() == 3) {
                experts = c.dims[2];
                rr = c.dims[1] * (moe ? experts : std::min<size_t>(4, experts));
            }
            items.push_back({&c, cc, rr, cc / kata::QK_K, c.type});
            const double bytes = (double)c.n_bytes / (double)experts * (double)(moe ? experts : std::min<size_t>(4, experts));
            total_bytes += bytes;
            if (c.type == 12) bytes_q4 += bytes; else bytes_q6 += bytes;
        }

        struct Res { void * w; kata::block_q8_K * x; float * out; dim3 gr, bl; size_t rows, nb; uint32_t type; };
        std::vector<Res> res(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            const Item & it = items[i];
            const size_t block = it.type == 12 ? sizeof(kata::block_q4_K) : sizeof(kata::block_q6_K);
            const size_t n_bytes = it.rows * it.nb * block;
            std::vector<kata::block_q8_K> hx(it.nb);
            std::vector<float> ha(it.cols);
            for (size_t k = 0; k < it.cols; ++k) ha[k] = std::sin(0.001f * (float)k) * 1.7f;
            kata::quantize_row_q8_K(ha.data(), hx.data(), it.cols);
            hipMalloc(&res[i].w, n_bytes);
            hipMalloc(&res[i].x, it.nb * sizeof(kata::block_q8_K));
            hipMalloc(&res[i].out, it.rows * sizeof(float));
            hipMemcpy(res[i].w, g.data(*it.t), n_bytes, hipMemcpyHostToDevice);
            hipMemcpy(res[i].x, hx.data(), it.nb * sizeof(kata::block_q8_K), hipMemcpyHostToDevice);
            const int per = it.type == 12 ? 64 : 32;
            const int rp = it.nb <= (size_t)per ? per / (int)it.nb : 1;
            res[i].gr = dim3(it.nb <= (size_t)per ? (it.rows + rp - 1) / rp : it.rows);
            res[i].bl = dim3(256);
            res[i].rows = it.rows;
            res[i].nb = it.nb;
            res[i].type = it.type;
        }

        Event e0, e1;
        double best = 1e9;
        for (int rep = 0; rep < 6; ++rep) {
            const size_t off = (size_t)rep * 37 % (items.size() ? items.size() : 1);
            hipEventRecord(e0.h);
            for (size_t k = 0; k < items.size(); ++k) {
                const size_t i = (k + off) % items.size();
                if (res[i].type == 12)
                    gemv_q4_K_kernel<<<res[i].gr, res[i].bl>>>((const kata::block_q4_K *)res[i].w,
                        res[i].x, res[i].out, (int)res[i].rows, (int)res[i].nb);
                else
                    gemv_q6_K_kernel<<<res[i].gr, res[i].bl>>>((const kata::block_q6_K *)res[i].w,
                        res[i].x, res[i].out, (int)res[i].rows, (int)res[i].nb);
            }
            hipEventRecord(e1.h);
            hipEventSynchronize(e1.h);
            float ms = 0;
            hipEventElapsedTime(&ms, e0.h, e1.h);
            best = std::min(best, (double)ms);
        }
        const double gbs = total_bytes / (best * 1e6);
        std::printf("DECODE SWEEP%s: %zu tensors, %.1f MB (q4_K %.0f + q6_K %.0f), %.2f ms\n",
                    moe ? " (all 64 experts)" : " (4 of 64 experts)", items.size(),
                    total_bytes / 1e6, bytes_q4 / 1e6, bytes_q6 / 1e6, best);
        std::printf("  %.1f GB/s (ceiling 764.3 -> %.0f%%);  %.1f us per launch\n",
                    gbs, 100.0 * gbs / 764.3, 1e3 * best / items.size());
        std::printf("  a decode token's weights alone: %.2f ms -> %.0f tok/s if everything else were free\n",
                    best, 1e3 / best);
        for (auto & r : res) { hipFree(r.w); hipFree(r.x); hipFree(r.out); }
        return 0;
    }

    // q6_K single tensor or sweep. Separate entry points rather than a type
    // switch inside one function, because the two formats need different block
    // sizes and the comparison logic is identical only in shape.
    const bool want_q6 = argc >= 3 && std::strcmp(argv[2], "--q6") == 0;
    if (want_q6) {
        const char * name = argc >= 4 ? argv[3] : "blk.0.attn_v.weight";
        const auto * q6 = g.find(name);
        if (!q6 || q6->type != 14) { std::fprintf(stderr, "no q6_K tensor %s\n", name); return 2; }
        const size_t cc = q6->dims[0];
        size_t rr = 1;
        for (size_t d = 1; d < q6->dims.size(); ++d) rr *= q6->dims[d];
        const size_t nbb = cc / kata::QK_K;
        std::printf("tensor %s: %zux%zu q6_K = %.1f MB\n", q6->name.c_str(), rr, cc, q6->n_bytes / 1e6);

        std::vector<float> ha(cc);
        for (size_t i = 0; i < cc; ++i) ha[i] = std::sin(0.001f * (float)i) * 1.7f + 0.3f * std::cos(0.01f * (float)i);
        std::vector<kata::block_q8_K> hxx(nbb);
        kata::quantize_row_q8_K(ha.data(), hxx.data(), cc);

        const auto * hw = static_cast<const kata::block_q6_K *>(g.data(*q6));
        std::vector<float> ref(rr);
        for (size_t r = 0; r < rr; ++r) ref[r] = kata::vec_dot_q6_K_q8_K_ref(cc, hw + r * nbb, hxx.data());

        kata::block_q6_K * dw; kata::block_q8_K * dx; float * dout;
        hipMalloc(&dw, q6->n_bytes); hipMalloc(&dx, nbb * sizeof(kata::block_q8_K)); hipMalloc(&dout, rr * 4);
        hipMemcpy(dw, hw, q6->n_bytes, hipMemcpyHostToDevice);
        hipMemcpy(dx, hxx.data(), nbb * sizeof(kata::block_q8_K), hipMemcpyHostToDevice);
        const int rp6 = nbb <= 32 ? (32 / (int)nbb) : 1;
        const dim3 gr6(nbb <= 32 ? (rr + rp6 - 1) / rp6 : rr), bl6(256);
        gemv_q6_K_kernel<<<gr6, bl6>>>(dw, dx, dout, (int)rr, (int)nbb);
        const auto e6 = hipDeviceSynchronize();
        if (e6 != hipSuccess) { std::fprintf(stderr, "kernel: %s\n", hipGetErrorString(e6)); return 2; }
        std::vector<float> gpu6(rr);
        hipMemcpy(gpu6.data(), dout, rr * 4, hipMemcpyDeviceToHost);

        double ma6 = 0, mr6 = 0;
        for (size_t r = 0; r < rr; ++r) {
            ma6 = std::max(ma6, std::fabs((double)gpu6[r] - (double)ref[r]));
            mr6 = std::max(mr6, std::fabs((double)ref[r]));
        }
        const double tol6 = 1e-4 * mr6;
        std::printf("accuracy vs cpu ref: max|abs| %.3e  largest |row| %.3f  tol %.3e -> %s\n",
                    ma6, mr6, tol6, ma6 <= tol6 ? "OK" : "FAIL");

        Event a, b2;
        double best6 = 1e9;
        for (int r = 0; r < 20; ++r) {
            hipEventRecord(a.h);
            gemv_q6_K_kernel<<<gr6, bl6>>>(dw, dx, dout, (int)rr, (int)nbb);
            hipEventRecord(b2.h);
            hipEventSynchronize(b2.h);
            float ms = 0;
            hipEventElapsedTime(&ms, a.h, b2.h);
            best6 = std::min(best6, (double)ms);
        }
        std::printf("gpu: %.3f ms  %.1f GB/s (ceiling 764.3)\n", best6, q6->n_bytes / (best6 * 1e6));
        hipFree(dw); hipFree(dx); hipFree(dout);
        return ma6 <= tol6 ? 0 : 1;
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
    // Gate against double truth, not against the other float arm. The two float
    // arms differ by up to 1.8e-4 relative on rows that nearly cancel, and on
    // those rows the GPU is the closer one -- so "agrees with the CPU" is the
    // wrong question and a CPU-relative gate would fail a correct kernel.
    double max_abs = 0, max_row = 0, err_cpu = 0;
    for (size_t r = 0; r < rows; ++r) {
        const double tru = vec_dot_q4_K_q8_K_double(cols, host_w + r * nb, host_x.data());
        max_abs = std::max(max_abs, std::fabs((double)gpu[r] - tru));
        err_cpu = std::max(err_cpu, std::fabs((double)cpu[r] - tru));
        max_row = std::max(max_row, std::fabs(tru));
    }
    // f32 accumulation over 256 values per block cannot be expected below a few
    // ULP of the magnitude, hence 1e-5 relative rather than exact.
    const double tol = 1e-5 * max_row;
    std::printf("accuracy: max|abs| vs double %.3e  (cpu arm %.3e)  largest |row| %.3f  tol %.3e -> %s\n",
                max_abs, err_cpu, max_row, tol, max_abs <= tol ? "OK" : "FAIL");
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
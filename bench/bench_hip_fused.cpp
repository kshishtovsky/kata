// EXP-0077: one launch per (format, width) bucket instead of one per tensor.
//
// Build: hipcc -O3 bench/bench_hip_fused.cpp -o /tmp/fuse -I include -L build -lkata_core -lpthread
//
// EXP-0073 streams a token's weights in 221 launches at 408 GB/s, of which
// 2.77 us x 221 = 0.61 ms is the launch itself. Removing launches entirely would
// give 488 GB/s. The question is whether a fused launch ALSO recovers the gap to
// the 603 GB/s the full-occupancy cold sweep reached -- i.e. whether the small
// tensors were launch-bound or occupancy-bound.
//
// The mechanism: a bucket table maps a global row index to (tensor, row within
// tensor), so one grid covers every matvec of a layer with each block free to
// work on any of them. This is the same idea as ggml's mul_mat_id, which groups
// the selected experts' rows into one launch.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include "kata/gguf.h"
#include "kata/quant.h"
#include "kata/types.h"

namespace {
constexpr int kG = 32;   // rows per block: 8 groups... see below

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

// A block computes `rpb` consecutive ROWS of ONE bucket. Group g maps to
// (row = g / nb, super-block = g % nb), so the group count must be a multiple of
// nb; rpb = unit / nb keeps that exact (unit = 64 groups for q4_K with 4 lanes
// each, 32 groups for q6_K with 8 lanes each). Blocks are numbered by ROW GROUP,
// not by row: block gr covers rows [gr*rpb, gr*rpb + rpb). Getting that mapping
// wrong overran `out` and faulted the GPU.
__global__ void gemv_q4_K_fused(const kata::block_q4_K * __restrict__ w,
                                const kata::block_q8_K * __restrict__ x,
                                float * __restrict__ out,
                                const int * __restrict__ row_of,
                                int nb, int nrows) {
    const int rpb = nb <= 64 ? (64 / nb) : 1;
    const int r0 = blockIdx.x * rpb;
    if (r0 >= nrows) return;
    const int lane = threadIdx.x & 3;
    const int group = threadIdx.x >> 2;
    const int r = group / nb;              // row within this block
    const int b = group % nb;              // super-block within the row
    float acc = 0.0f;
    if (r < rpb && r0 + r < nrows) {
        const int row = row_of[r0 + r];
        for (int bb = b; bb < nb; bb += (nb <= 64 ? nb : 64)) {
            const kata::block_q4_K & blk = w[(size_t)row * nb + bb];
            const kata::block_q8_K & ab = x[bb];
            const float d = f16_dev(blk.d) * ab.d;
            const float dmin = f16_dev(blk.dmin) * ab.d;

            unsigned char sc0, m0, sc1, m1;
            scale_min_dev(2 * lane + 0, blk.scales, &sc0, &m0);
            scale_min_dev(2 * lane + 1, blk.scales, &sc1, &m1);

            uint4 v0 = *reinterpret_cast<const uint4 *>(blk.qs + 32 * lane);
            uint4 v1 = *reinterpret_cast<const uint4 *>(blk.qs + 32 * lane + 16);
            const unsigned int * qw = reinterpret_cast<const unsigned int *>(&v0);
            const unsigned int * qw2 = reinterpret_cast<const unsigned int *>(&v1);

            const int j = 64 * lane;
            int acc0 = 0, acc1 = 0, sy0 = 0, sy1 = 0;
            for (int w_i = 0; w_i < 4; ++w_i) {
                const unsigned int word = qw[w_i];
                for (int byte = 0; byte < 4; ++byte) {
                    const int shift = 8 * byte;
                    const int lo = (word >> shift) & 0x0f;
                    const int hi = (word >> (shift + 4)) & 0x0f;
                    const int l = 4 * w_i + byte;
                    const int y0 = ab.qs[j + l];
                    const int y1 = ab.qs[j + 32 + l];
                    acc0 += lo * y0; acc1 += hi * y1; sy0 += y0; sy1 += y1;
                }
            }
            for (int w_i = 0; w_i < 4; ++w_i) {
                const unsigned int word = qw2[w_i];
                for (int byte = 0; byte < 4; ++byte) {
                    const int shift = 8 * byte;
                    const int lo = (word >> shift) & 0x0f;
                    const int hi = (word >> (shift + 4)) & 0x0f;
                    const int l = 16 + 4 * w_i + byte;
                    const int y0 = ab.qs[j + l];
                    const int y1 = ab.qs[j + 32 + l];
                    acc0 += lo * y0; acc1 += hi * y1; sy0 += y0; sy1 += y1;
                }
            }
            // int32 accumulation, scaled by each sub-block's own scale; the
            // per-super-block factor d*ab.d is applied after (EXP-0029).
            acc += d * (float)sc0 * (float)acc0 - dmin * (float)m0 * (float)sy0;
            acc += d * (float)sc1 * (float)acc1 - dmin * (float)m1 * (float)sy1;
        }
    }
    acc += __shfl_down_sync(0xffffffffull, acc, 2);
    acc += __shfl_down_sync(0xffffffffull, acc, 1);
    __shared__ float gsum[64];
    if (lane == 0) gsum[group] = acc;
    __syncthreads();
    if (threadIdx.x < rpb && r0 + threadIdx.x < nrows) {
        const int rr = threadIdx.x;
        float s = 0.0f;
        for (int i = 0; i < (nb < 64 ? nb : 64); ++i) s += gsum[rr * nb + i];
        out[r0 + rr] = s;
    }
}

__global__ void gemv_q6_K_fused(const kata::block_q6_K * __restrict__ w,
                                const kata::block_q8_K * __restrict__ x,
                                float * __restrict__ out,
                                const int * __restrict__ row_of,
                                int nb, int nrows) {
    const int rpb = nb <= 32 ? (32 / nb) : 1;
    const int r0 = blockIdx.x * rpb;
    if (r0 >= nrows) return;
    const int lane = threadIdx.x & 7;
    const int group = threadIdx.x >> 3;
    const int r = group / nb;
    const int b = group % nb;
    float acc = 0.0f;
    if (r < rpb && r0 + r < nrows) {
        const int row = row_of[r0 + r];
        for (int bb = b; bb < nb; bb += (nb <= 32 ? nb : 32)) {
            const kata::block_q6_K & blk = w[(size_t)row * nb + bb];
            const kata::block_q8_K & ab = x[bb];
            const float d = f16_dev(blk.d) * ab.d;
            const int is = lane >> 2, l0 = lane * 4;   // 4 consecutive l, one scale group
            float bs = 0.0f;
            for (int half = 0; half < 2; ++half) {
                const unsigned char * ql = blk.ql + half * 64;
                const unsigned char * qh = blk.qh + half * 32;
                const signed char * q8 = ab.qs + half * 128;
                int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                for (int k = 0; k < 4; ++k) {
                    const int l = l0 + k;
                    const unsigned char qlb = ql[l], qhb = qh[l];
                    const int q1 = (int)((qlb & 0x0f) | (((qhb >> 0) & 3) << 4)) - 32;
                    const int q2 = (int)((ql[l + 32] & 0x0f) | (((qhb >> 2) & 3) << 4)) - 32;
                    const int q3 = (int)((qlb >> 4) | (((qhb >> 4) & 3) << 4)) - 32;
                    const int q4 = (int)((ql[l + 32] >> 4) | (((qhb >> 6) & 3) << 4)) - 32;
                    a0 += q1 * (int)q8[l]; a1 += q2 * (int)q8[l + 32];
                    a2 += q3 * (int)q8[l + 64]; a3 += q4 * (int)q8[l + 96];
                }
                const signed char * sc = blk.scales + half * 8;
                bs += (float)sc[is + 0] * (float)a0 + (float)sc[is + 2] * (float)a1
                    + (float)sc[is + 4] * (float)a2 + (float)sc[is + 6] * (float)a3;
            }
            acc += d * bs;
        }
    }
    for (int off = 4; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffull, acc, off);
    __shared__ float gsum6[32];
    if (lane == 0) gsum6[group] = acc;
    __syncthreads();
    if (threadIdx.x < rpb && r0 + threadIdx.x < nrows) {
        const int rr = threadIdx.x;
        float s = 0.0f;
        for (int i = 0; i < (nb < 32 ? nb : 32); ++i) s += gsum6[rr * nb + i];
        out[r0 + rr] = s;
    }
}
} // namespace

namespace {
template <class T>
const void * w_host_t(const std::vector<T> & v, size_t row, size_t nb) {
    return (const void *)&v[row * nb];
}
} // namespace

int main(int argc, char ** argv) {
    kata::GgufFile g; g.open(argv[1]);
    // The kernel derives rows-per-block from nb (64/nb for q4_K, 32/nb for
    // q6_K), so the grid MUST be sized from the same rule. A bucket therefore
    // holds one (type, nb) pair: mixing widths in one bucket overran `out`,
    // which is what faulted (grid sized with rpb=8 for an nb=46 tensor whose
    // real rpb is 1 -> 8x too many blocks).
    struct Bucket {
        int type = 0;          // 12 = q4_K, 14 = q6_K
        int nb = 0;
        std::vector<kata::block_q4_K> q4;
        std::vector<kata::block_q6_K> q6;
        std::vector<int> row_of;
        size_t rows = 0;
        double bytes = 0;
        int rpb() const {
            const int unit = type == 12 ? 64 : 32;
            return nb <= unit ? unit / nb : 1;
        }
        size_t nblocks() const { return (rows + rpb() - 1) / rpb(); }
    };
    std::vector<Bucket> buckets;
    auto bucket_for = [&](int type, int nb) -> Bucket & {
        for (auto & b : buckets)
            if (b.type == type && b.nb == nb) return b;
        buckets.push_back({});
        buckets.back().type = type;
        buckets.back().nb = nb;
        return buckets.back();
    };

    for (const auto & c : g.tensors()) {
        if ((c.type != 12 && c.type != 14) || c.dims.size() < 2) continue;
        const size_t cols = c.dims[0];
        if (cols % kata::QK_K != 0) continue;
        const size_t rows = c.dims[1];
        const size_t experts = c.dims.size() == 3 ? c.dims[2] : 1;
        const size_t take = experts > 1 ? std::min<size_t>(4, experts) : 1;
        const size_t nb = cols / kata::QK_K;
        const size_t per_expert = rows * nb;
        const size_t bsz = c.type == 12 ? sizeof(kata::block_q4_K) : sizeof(kata::block_q6_K);
        Bucket & bk = bucket_for(c.type, (int)nb);
        for (size_t e = 0; e < take; ++e) {
            const size_t slab = (experts > 1 ? e : 0) * per_expert;
            const size_t row_base = bk.rows;
            if (c.type == 12) {
                const auto * src = (const kata::block_q4_K *)g.data(c) + slab;
                bk.q4.insert(bk.q4.end(), src, src + per_expert);
            } else {
                const auto * src = (const kata::block_q6_K *)g.data(c) + slab;
                bk.q6.insert(bk.q6.end(), src, src + per_expert);
            }
            for (size_t r = 0; r < rows; ++r) bk.row_of.push_back((int)(row_base + r));
            bk.rows += rows;
            bk.bytes += (double)per_expert * bsz;
        }
    }
    double total_bytes = 0;
    size_t total_rows = 0;
    for (const auto & b : buckets) { total_bytes += b.bytes; total_rows += b.rows; }
    std::printf("%zu buckets, %zu rows, %.1f MB\n", buckets.size(), total_rows, total_bytes / 1e6);

    // The widest activation row needed by any bucket.
    size_t max_nb = 1;
    for (const auto & b : buckets) max_nb = std::max<size_t>(max_nb, (size_t)b.nb);
    std::vector<kata::block_q8_K> hx(max_nb);
    std::vector<float> act(max_nb * kata::QK_K);
    for (size_t i = 0; i < act.size(); ++i) act[i] = std::sin(0.001f * (float)i) * 1.7f;
    kata::quantize_row_q8_K(act.data(), hx.data(), act.size());

    kata::block_q8_K * dx;
    hipMalloc(&dx, hx.size() * sizeof(kata::block_q8_K));
    hipMemcpy(dx, hx.data(), hx.size() * sizeof(kata::block_q8_K), hipMemcpyHostToDevice);

    struct Dev { kata::block_q4_K * w4 = nullptr; kata::block_q6_K * w6 = nullptr;
                 float * o = nullptr; int * row = nullptr; const Bucket * b = nullptr; };
    std::vector<Dev> devs(buckets.size());
    for (size_t i = 0; i < buckets.size(); ++i) {
        const Bucket & b = buckets[i];
        Dev & d = devs[i];
        d.b = &b;
        if (b.type == 12) {
            hipMalloc(&d.w4, b.q4.size() * sizeof(kata::block_q4_K));
            hipMemcpy(d.w4, b.q4.data(), b.q4.size() * sizeof(kata::block_q4_K), hipMemcpyHostToDevice);
        } else {
            hipMalloc(&d.w6, b.q6.size() * sizeof(kata::block_q6_K));
            hipMemcpy(d.w6, b.q6.data(), b.q6.size() * sizeof(kata::block_q6_K), hipMemcpyHostToDevice);
        }
        hipMalloc(&d.o, b.rows * sizeof(float));
        hipMalloc(&d.row, b.row_of.size() * sizeof(int));
        hipMemcpy(d.row, b.row_of.data(), b.row_of.size() * sizeof(int), hipMemcpyHostToDevice);
    }

    auto launches = [&] {
        for (Dev & d : devs) {
            const Bucket & b = *d.b;
            const unsigned gx = (unsigned)b.nblocks();
            if (b.type == 12)
                gemv_q4_K_fused<<<dim3(gx), 256>>>(d.w4, dx, d.o, d.row, b.nb, (int)b.rows);
            else
                gemv_q6_K_fused<<<dim3(gx), 256>>>(d.w6, dx, d.o, d.row, b.nb, (int)b.rows);
        }
    };
    launches();
    if (hipDeviceSynchronize() != hipSuccess) { std::printf("kernel error\n"); return 2; }

    // Correctness: every fused row against the per-tensor C++ reference, on the
    // same activation. A speed number from an unverified kernel is worthless.
    {
        double worst = 0, worst_rel = 0;
        int checked = 0, bad = 0;
        for (size_t i = 0; i < buckets.size(); ++i) {
            const Bucket & b = buckets[i];
            std::vector<float> got(b.rows);
            hipMemcpy(got.data(), devs[i].o, b.rows * sizeof(float), hipMemcpyDeviceToHost);
            for (size_t r = 0; r < b.rows; ++r) {
                const size_t row = (size_t)b.row_of[r];
                const double ref = b.type == 12
                    ? kata::vec_dot_q4_K_q8_K_ref(b.nb * kata::QK_K,
                          (const kata::block_q4_K *)w_host_t(b.q4, row, b.nb), hx.data())
                    : kata::vec_dot_q6_K_q8_K_ref(b.nb * kata::QK_K,
                          (const kata::block_q6_K *)w_host_t(b.q6, row, b.nb), hx.data());
                worst = std::max(worst, std::fabs((double)got[r] - ref));
                worst_rel = std::max(worst_rel, std::fabs((double)got[r] - ref) / std::max(1e-9, std::fabs(ref)));
                ++checked;
                if (std::fabs((double)got[r] - ref) > 1e-4 * std::max(1.0, std::fabs(ref))) ++bad;
            }
        }
        for (size_t i = 0; i < buckets.size(); ++i) {
            const Bucket & b = buckets[i];
            std::vector<float> got(b.rows);
            hipMemcpy(got.data(), devs[i].o, b.rows * sizeof(float), hipMemcpyDeviceToHost);
            int nb_bad = 0; double w = 0; size_t first = 0;
            for (size_t r = 0; r < b.rows; ++r) {
                const size_t row = (size_t)b.row_of[r];
                const double ref = b.type == 12
                    ? kata::vec_dot_q4_K_q8_K_ref(b.nb * kata::QK_K,
                          (const kata::block_q4_K *)w_host_t(b.q4, row, b.nb), hx.data())
                    : kata::vec_dot_q6_K_q8_K_ref(b.nb * kata::QK_K,
                          (const kata::block_q6_K *)w_host_t(b.q6, row, b.nb), hx.data());
                const double d = std::fabs((double)got[r] - ref);
                if (d > 1e-4 * std::max(1.0, std::fabs(ref))) { if (!nb_bad) first = r; ++nb_bad; }
                w = std::max(w, d);
            }
            std::printf("  bucket type=%d nb=%d rows=%zu rpb=%d: %d bad, worst %.3e, first row %zu\n",
                        b.type, b.nb, b.rows, b.rpb(), nb_bad, w, first);
        }
        std::printf("accuracy: %d rows, max|abs| %.3e, max rel %.3e, %d outside 1e-4 -> %s\n",
                    checked, worst, worst_rel, bad, bad == 0 ? "OK" : "FAIL");
        if (bad) return 3;
    }

    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    double best = 1e9;
    for (int r = 0; r < 20; ++r) {
        hipEventRecord(e0); launches(); hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1);
        best = std::min(best, (double)ms);
    }
    const double gbs = total_bytes / (best * 1e6);
    std::printf("fused %zu launches for %.1f MB: %.3f ms -> %.1f GB/s\n",
                buckets.size(), total_bytes / 1e6, best, gbs);
    std::printf("  per-tensor for the same bytes: 3.72 ms, 408 GB/s, 221 launches\n");
    std::printf("token: weights %.2f + attn 1.34 + glue 0.90 = %.2f ms -> %.0f tok/s\n",
                best, best + 2.24, 1e3 / (best + 2.24));
    return 0;
}

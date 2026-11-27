// EXP-0083: the whole token's weight stream on the GPU, with every tensor resident.
//
// EXP-0082 established that a token's MoE stage costs 1.69 ms on the GPU when its
// 930 MB of experts are resident, against 14.20 ms on the CPU. But that is one
// stage of nine, and this project has a history of projections shrinking as each
// remaining stage gets measured (8.5x -> 6.3x -> 5.3x in EXP-0076).
//
// The model is 14.52 GB and 13.84 GB of that is experts, so ALL of it fits in the
// card's 20 GB. That removes the hybrid entirely: with everything resident there
// are no PCIe transfers and no per-layer handoffs, and this probe asks the only
// question left -- what does one token's full weight stream cost when nothing has
// to move and nothing has to bounce to the host?
//
// It streams every q4_K and q6_K tensor a token touches (all 40 layers' attention,
// shortconv, dense FFN, lm_head and 4 of 64 experts per MoE layer) through the two
// bucket kernels, with the activation rows also resident.
//
// Build: hipcc -O3 bench/hip/bench_token_gpu.hip -o /tmp/tokg -I include \
//              -L build -lkata_core -lpthread
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "kata/gguf.h"
#include "kata/pool.h"
#include "kata/quant.h"
#include "kata/types.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {
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

// EXP-0077's bucket kernel, specialised to one shape: `nslab` expert slabs of
// `rows` rows each, all sharing one q4_K width `nb`. One launch covers every
// slab, so the grid is big enough to fill the device -- a per-slab launch was
// occupancy-bound at 22-43 GB/s.
__global__ void gemv_q4_K_batch(const kata::block_q4_K * __restrict__ w,
                                const kata::block_q8_K * __restrict__ x,
                                float * __restrict__ out,
                                int rows, int nb, int nslab) {
    const int rpb = nb <= 64 ? (64 / nb) : 1;
    // Flat 1-D grid: a (x=row-groups, y=slabs) grid made every slab a separate
    // wavefront and measured 60 GB/s against the 764 the same card reaches on a
    // plain stream. One flat index lets the scheduler interleave slabs freely.
    const int nb_row_blocks = (rows + rpb - 1) / rpb;
    const int slab = blockIdx.x / nb_row_blocks;
    const int grid_row = (blockIdx.x % nb_row_blocks) * rpb;   // row within a slab
    if (slab >= nslab || grid_row >= rows) return;
    const int lane = threadIdx.x & 3;
    const int group = threadIdx.x >> 2;
    const int r = group / nb;
    const int b = group % nb;
    float acc = 0.0f;
    if (r < rpb && grid_row + r < rows) {
        const int row = grid_row + r;
        const kata::block_q4_K * base = w + (size_t)slab * rows * nb;
        for (int bb = b; bb < nb; bb += (nb <= 64 ? nb : 64)) {
            const kata::block_q4_K & blk = base[(size_t)row * nb + bb];
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
            for (int wi = 0; wi < 4; ++wi) {
                const unsigned int word = qw[wi];
                for (int byte = 0; byte < 4; ++byte) {
                    const int shift = 8 * byte;
                    const int lo = (word >> shift) & 0x0f, hi = (word >> (shift + 4)) & 0x0f;
                    const int l = 4 * wi + byte;
                    const int y0 = ab.qs[j + l], y1 = ab.qs[j + 32 + l];
                    acc0 += lo * y0; acc1 += hi * y1; sy0 += y0; sy1 += y1;
                }
            }
            for (int wi = 0; wi < 4; ++wi) {
                const unsigned int word = qw2[wi];
                for (int byte = 0; byte < 4; ++byte) {
                    const int shift = 8 * byte;
                    const int lo = (word >> shift) & 0x0f, hi = (word >> (shift + 4)) & 0x0f;
                    const int l = 16 + 4 * wi + byte;
                    const int y0 = ab.qs[j + l], y1 = ab.qs[j + 32 + l];
                    acc0 += lo * y0; acc1 += hi * y1; sy0 += y0; sy1 += y1;
                }
            }
            acc += d * (float)sc0 * (float)acc0 - dmin * (float)m0 * (float)sy0;
            acc += d * (float)sc1 * (float)acc1 - dmin * (float)m1 * (float)sy1;
        }
    }
    acc += __shfl_down_sync(0xffffffffull, acc, 2);
    acc += __shfl_down_sync(0xffffffffull, acc, 1);
    __shared__ float gsum[64];
    if (lane == 0) gsum[group] = acc;
    __syncthreads();
    if (threadIdx.x < rpb && grid_row + threadIdx.x < rows) {
        float s = 0.0f;
        for (int i = 0; i < (nb < 64 ? nb : 64); ++i) s += gsum[threadIdx.x * nb + i];
        out[(size_t)slab * rows + grid_row + threadIdx.x] = s;
    }
}

// q6_K bucket kernel. The value mapping is the one verified in EXP-0073
// (bench/bench_hip_gemv.cpp) -- an `off >> 1` shortcut I first wrote here
// silently picked the wrong nibble and read 4.0 relative error, while still
// running at full speed.
__global__ void gemv_q6_K_batch(const kata::block_q6_K * __restrict__ w,
                                const kata::block_q8_K * __restrict__ x,
                                float * __restrict__ out,
                                int rows, int nb, int nslab) {
    const int rpb = nb <= 32 ? (32 / nb) : 1;
    const int lane = threadIdx.x & 7;      // 8 threads per super-block
    const int group = threadIdx.x >> 3;    // 0..31
    const int nb_row_blocks = (rows + rpb - 1) / rpb;
    const int slab = blockIdx.x / nb_row_blocks;
    const int row0 = (blockIdx.x % nb_row_blocks) * rpb;
    if (slab >= nslab) return;

    int row, b;
    if (nb <= 32) {
        row = row0 + group / nb;
        b = group % nb;
    } else {
        row = row0;
        b = group;
    }
    const bool active = nb <= 32 ? (group < rpb * nb && row < rows) : (b < nb);

    float acc = 0.0f;
    if (active) {
        const kata::block_q6_K * base = w + (size_t)slab * rows * nb;
        for (int bb = b; bb < nb; bb += (nb <= 32 ? nb : 32)) {
            const kata::block_q6_K & blk = base[(size_t)row * nb + bb];
            const kata::block_q8_K & ab = x[bb];
            const float d = f16_dev(blk.d) * ab.d;

            const int is = lane >> 2;
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
            const int gr = row0 + r;
            if (gr >= rows) return;
            for (int i = 0; i < nb; ++i) s += gsum6[r * nb + i];
            out[(size_t)slab * rows + gr] = s;
        } else {
            for (int i = 0; i < 32; ++i) s += gsum6[i];
            out[(size_t)slab * rows + row0] = s;
        }
    }
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

// One tensor's worth of work on the device: where its rows start in the blob and
// how the kernel must be shaped for it.
struct Stream {
    const void * rows;
    int rows_n, nb, nslab;
    bool q6;
};

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("usage: %s <model.gguf> [n_expert_used]\n", argv[0]); return 1; }
    std::setbuf(stdout, nullptr);
    const int n_used = argc > 2 ? std::atoi(argv[2]) : 4;
    kata::GgufFile g; g.open(argv[1]);

    // Every quantised tensor a token reads, minus the 60 experts per layer it does
    // not route to. Sizes come from the file, never from assumed bit widths.
    struct Item { const kata::GgufTensor * t; int nslab; };
    std::vector<Item> items;
    size_t total_b = 0, fixed_b = 0, expert_b = 0;
    int n_expert = 0;
    if (const auto * e = g.find("blk.2.ffn_gate_exps.weight")) n_expert = (int)e->dims[2];
    for (const auto & t : g.tensors()) {
        if (t.type != 12 && t.type != 14) continue;              // q4_K, q6_K only
        const bool is_exp = t.name.find("_exps.") != std::string::npos;
        const int slabs = is_exp ? n_used : 1;
        // token_embd is read once by the lm_head; everything else once per layer.
        const size_t b = (t.n_bytes / (size_t)(is_exp ? (size_t)n_expert : 1)) * (size_t)slabs;
        items.push_back({&t, slabs});
        total_b += b;
        (is_exp ? expert_b : fixed_b) += b;
    }
    std::printf("token stream: %zu tensors, %d of %d experts\n", items.size(), n_used, n_expert);
    std::printf("  fixed %.1f MB + experts %.1f MB = %.1f MB\n",
                fixed_b / 1e6, expert_b / 1e6, total_b / 1e6);

    // Group the picked slabs into BUCKETS keyed on (format, nb) and lay each bucket
    // out contiguously. Rows within a bucket are independent and share `nb`, so one
    // kernel launch covers the whole bucket. Launching per tensor instead made 221
    // launches whose grids were too small to fill 84 CUs: 410 GB/s of payload
    // against the 677 EXP-0077 reached with exactly this bucketing.
    // Two passes: collect every picked slab per bucket, then lay each bucket out
    // contiguously. One pass would interleave buckets, and a bucket's rows must be
    // adjacent for a single launch to cover them -- that is the whole point.
    struct Slab { const unsigned char * src; size_t rows, row_b; };
    std::map<std::pair<int, int>, std::vector<Slab>> buckets;   // key = (is_q6, nb)
    uint32_t pick_seed = 12345;
    auto next_pick = [&](int n) {
        pick_seed = pick_seed * 1103515245u + 12345u;
        return (int)((pick_seed >> 16) % (unsigned)n);
    };
    for (const auto & it : items) {
        const auto & t = *it.t;
        const size_t vals_per_row = t.dims[0];
        const int nb = (int)(vals_per_row / kata::QK_K);
        const size_t rows_total = t.n_elements / vals_per_row;
        const bool is_exp = t.name.find("_exps.") != std::string::npos;
        const size_t rows = rows_total / (size_t)(is_exp ? n_expert : 1);
        const size_t n_slab_total = rows_total / rows;
        const size_t row_b = t.n_bytes / rows_total;
        auto & bk = buckets[{t.type == 14 ? 1 : 0, nb}];
        for (int s = 0; s < it.nslab; ++s) {
            const int which = n_slab_total > 1 ? next_pick((int)n_slab_total) : 0;
            bk.push_back({(const unsigned char *)g.data(t) + (size_t)which * rows * row_b, rows, row_b});
        }
    }
    std::vector<unsigned char> blob;
    std::vector<Stream> streams;                       // offsets + row counts per bucket
    std::vector<std::pair<int, int>> bucket_keys;
    size_t off = 0;
    for (const auto & [key, slabs] : buckets) {
        size_t rows_sum = 0, bytes_sum = 0;
        for (const auto & s : slabs) {
            blob.resize(off + s.rows * s.row_b);
            std::memcpy(blob.data() + off, s.src, s.rows * s.row_b);
            off += s.rows * s.row_b;
            rows_sum += s.rows;
            bytes_sum += s.rows * s.row_b;
        }
        streams.push_back({(const void *)(off - bytes_sum), (int)rows_sum, key.second, 1, key.first != 0});
        bucket_keys.push_back(key);
    }
    const size_t stream_b = blob.size();
    const std::vector<unsigned char> & host_blob = blob;
    void * dblob = nullptr;
    const double u0 = now_ms();
    CK(hipMalloc(&dblob, stream_b));
    CK(hipMemcpy(dblob, blob.data(), stream_b, hipMemcpyHostToDevice));
    std::printf("  upload %.1f MB in %.0f ms = %.1f GB/s (load time, once)\n",
                stream_b / 1e6, now_ms() - u0, stream_b / 1e6 / (now_ms() - u0));
    std::printf("  %zu buckets:", streams.size());
    for (size_t i = 0; i < bucket_keys.size(); ++i)
        std::printf(" %s/nb=%d x%d rows", bucket_keys[i].first ? "q6" : "q4",
                    bucket_keys[i].second, streams[i].rows_n);
    std::printf("\n");

    // Offsets stay in `streams` for the host-side verification; the device copy
    // rebases them. Overwriting them with device pointers made the verification
    // read `host_blob + <device address>` and segfault inside the CPU reference.
    std::vector<Stream> dev = streams;
    for (auto & s : dev) s.rows = (const unsigned char *)dblob + (size_t)s.rows;

    // Activation rows, one per distinct nb the token needs, all in VRAM.
    std::map<int, kata::block_q8_K *> dact;
    std::map<int, kata::block_q8_K *> hact;
    for (const auto & s : streams) {
        if (dact.count(s.nb)) continue;
        std::vector<float> v((size_t)s.nb * kata::QK_K);
        for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(0.001f * (float)i) * 1.7f;
        auto * h = (kata::block_q8_K *)std::malloc((size_t)s.nb * sizeof(kata::block_q8_K));
        kata::quantize_row_q8_K(v.data(), h, v.size());
        kata::block_q8_K * d = nullptr;
        CK(hipMalloc(&d, (size_t)s.nb * sizeof(kata::block_q8_K)));
        CK(hipMemcpy(d, h, (size_t)s.nb * sizeof(kata::block_q8_K), hipMemcpyHostToDevice));
        dact[s.nb] = d; hact[s.nb] = h;
    }
    float * dout = nullptr;
    size_t out_n = 0;
    for (const auto & s : streams) out_n = std::max(out_n, (size_t)s.rows_n);
    CK(hipMalloc(&dout, out_n * sizeof(float)));

    const dim3 bl(256);
    auto token_pass = [&] {
        for (const auto & s : dev) {
            kata::block_q8_K * x = dact[s.nb];
            if (s.q6) {
                const int rpb = s.nb <= 32 ? 32 / s.nb : 1;
                const dim3 gr((unsigned)(((s.rows_n + rpb - 1) / rpb) * s.nslab));
                gemv_q6_K_batch<<<gr, bl>>>((const kata::block_q6_K *)s.rows, x, dout,
                                            s.rows_n, s.nb, s.nslab);
            } else {
                const int rpb = s.nb <= 64 ? 64 / s.nb : 1;
                const dim3 gr((unsigned)(((s.rows_n + rpb - 1) / rpb) * s.nslab));
                gemv_q4_K_batch<<<gr, bl>>>((const kata::block_q4_K *)s.rows, x, dout,
                                            s.rows_n, s.nb, s.nslab);
            }
        }
        CK(hipDeviceSynchronize());
    };

    // One tensor of each format verified against the CPU reference before any
    // timing: a bucket kernel with a wrong row mapping still runs at full speed.
    for (int want = 0; want < 2; ++want) {
        // Launch from `dev` (device pointers) but verify against `streams` (host
        // offsets): using `streams` for the launch passed the byte OFFSET as a
        // device address, so the GPU faulted at address 163840 == the first
        // bucket's offset.
        for (size_t bi = 0; bi < streams.size(); ++bi) {
            const Stream & s = streams[bi];
            const Stream & sd = dev[bi];
            if ((int)s.q6 != want) continue;
            std::vector<float> gpu_v((size_t)s.rows_n), cpu_v((size_t)s.rows_n);
            const int rpb = !s.q6 ? (s.nb <= 64 ? 64 / s.nb : 1) : (s.nb <= 32 ? 32 / s.nb : 1);
            const dim3 gr((unsigned)((s.rows_n + rpb - 1) / rpb));
            if (s.q6) gemv_q6_K_batch<<<gr, bl>>>((const kata::block_q6_K *)sd.rows, dact[s.nb],
                                                  dout, s.rows_n, s.nb, 1);
            else gemv_q4_K_batch<<<gr, bl>>>((const kata::block_q4_K *)sd.rows, dact[s.nb],
                                             dout, s.rows_n, s.nb, 1);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(gpu_v.data(), dout, gpu_v.size() * sizeof(float), hipMemcpyDeviceToHost));
            const auto * hw_dev = (const unsigned char *)host_blob.data() + (size_t)s.rows;
            const size_t nv = (size_t)s.nb * kata::QK_K;
            for (int r = 0; r < s.rows_n; ++r) {
                if (s.q6)
                    cpu_v[r] = kata::vec_dot_q6_K_q8_K_ref(nv, (const kata::block_q6_K *)hw_dev + (size_t)r * s.nb, hact[s.nb]);
                else
                    cpu_v[r] = kata::vec_dot_q4_K_q8_K_ref(nv, (const kata::block_q4_K *)hw_dev + (size_t)r * s.nb, hact[s.nb]);
            }
            double mx = 0, scale = 0;
            for (int r = 0; r < s.rows_n; ++r) {
                const double dv = std::fabs((double)gpu_v[r] - (double)cpu_v[r]);
                const double cv = std::fabs((double)cpu_v[r]);
                if (dv > mx) mx = dv;
                if (cv > scale) scale = cv;
            }
            std::printf("\nverify %s nb=%d rows=%d off=%zu: max|diff| %.3e on %.3e (%.1e rel) %s\n",
                        want ? "q6_K" : "q4_K", s.nb, s.rows_n, (size_t)s.rows, mx, scale,
                        mx / scale, mx / scale < 1e-4 ? "PASS" : "FAIL");
            if (!(mx / scale < 1e-4)) return 1;
            goto next_format;
        }
    next_format:;
    }

    // Per-bucket timing: an aggregate hides a slow small bucket behind the bulk one.
    {
        std::printf("\nper-bucket:\n");
        for (const auto & sd : dev) {
            auto one = [&] {
                kata::block_q8_K * x = dact[sd.nb];
                const int rpb = sd.q6 ? (sd.nb <= 32 ? 32 / sd.nb : 1) : (sd.nb <= 64 ? 64 / sd.nb : 1);
                const dim3 gr((unsigned)((sd.rows_n + rpb - 1) / rpb));
                if (sd.q6) gemv_q6_K_batch<<<gr, bl>>>((const kata::block_q6_K *)sd.rows, x, dout, sd.rows_n, sd.nb, 1);
                else gemv_q4_K_batch<<<gr, bl>>>((const kata::block_q4_K *)sd.rows, x, dout, sd.rows_n, sd.nb, 1);
                CK(hipDeviceSynchronize());
            };
            one();
            const double a = now_ms();
            for (int i = 0; i < 5; ++i) one();
            const double ms = (now_ms() - a) / 5;
            size_t b = 0;
            for (size_t k = 0; k < bucket_keys.size(); ++k)
                if (&dev[k] == &sd)
                    for (const auto & sl : buckets[bucket_keys[k]]) b += sl.rows * sl.row_b;
            std::printf("  %s nb=%2d rows=%7d %7.1f MB %7.3f ms %6.1f GB/s\n",
                        sd.q6 ? "q6" : "q4", sd.nb, sd.rows_n, b / 1e6, ms, b / 1e6 / ms);
        }
    }

    token_pass();                                        // warm
    // Per-rep print: the card's clocks ramp with sustained load, and a single
    // 5-rep average taken cold reads ~29% slower than one taken warm, which is
    // exactly the gap between two runs of this same binary.
    for (int i = 0; i < 10; ++i) {
        const double a = now_ms();
        token_pass();
        std::printf("  rep %2d: %.3f ms  %.1f GB/s\n", i, now_ms() - a, stream_b / 1e6 / (now_ms() - a));
    }
    const int reps = 20;
    const double t = [&] {
        const double a = now_ms();
        for (int i = 0; i < reps; ++i) token_pass();
        return (now_ms() - a) / reps;
    }();
    const double launch_ms = dev.size() * 2.77e-3;
    std::printf("\nFULL TOKEN, everything resident: %.3f ms in %zu launches\n",
                t, dev.size());
    std::printf("  %.1f GB/s | launches cost %.3f ms | payload %.3f ms = %.1f GB/s\n",
                stream_b / 1e6 / t, launch_ms, t - launch_ms,
                stream_b / 1e6 / (t - launch_ms));
    std::printf("  -> %.0f tok/s of weight streaming alone (no attention, no lm_head compute)\n",
                1000.0 / t);
    std::printf("  CPU decode today: 22.5 ms = 44 tok/s\n");
    std::printf("  speedup on the memory-bound part: %.1fx\n", 22.5 / t);
    return 0;
}

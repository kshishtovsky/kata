// EXP-0082b: the whole MoE stage of one decode token, on the GPU.
//
// EXP-0082 measured that the CPU and GPU stream weights concurrently without
// stealing each other's bandwidth, and that a single layer's 4 experts (7-24 MB)
// is too small to split -- one GPU launch plus handoff costs more than the CPU
// spends on the whole slab. This probe asks the question that decides the hybrid:
// if a token's ENTIRE expert traffic (38 layers x 4 experts = ~931 MB) stays
// resident in VRAM, what does the MoE stage cost per token versus the 14.20 ms
// the CPU engine profiles for `ffn`?
//
// Build: hipcc -O3 bench/hip/bench_moe_gpu.hip -o /tmp/moeg -I include \
//              -L build -lkata_core -lpthread
#include <hip/hip_runtime.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Same bucket idea for q6_K: 8 lanes per super-block, each owning 4 values in
// both halves, 32 super-blocks per 256 threads so `nb <= 32` packs whole rows.
__global__ void gemv_q6_K_batch(const kata::block_q6_K * __restrict__ w,
                                const kata::block_q8_K * __restrict__ x,
                                float * __restrict__ out,
                                int rows, int nb, int nslab) {
    const int rpb = nb <= 32 ? (32 / nb) : 1;
    const int nb_row_blocks = (rows + rpb - 1) / rpb;
    const int slab = blockIdx.x / nb_row_blocks;
    const int grid_row = (blockIdx.x % nb_row_blocks) * rpb;
    if (slab >= nslab || grid_row >= rows) return;
    const int lane = threadIdx.x & 7;
    const int group = threadIdx.x >> 3;          // 32 groups per block
    const int r = group / nb;
    const int b = group % nb;
    float acc = 0.0f;
    if (r < rpb && grid_row + r < rows) {
        const int row = grid_row + r;
        const kata::block_q6_K * base = w + (size_t)slab * rows * nb;
        for (int bb = b; bb < nb; bb += (nb <= 32 ? nb : 32)) {
            const kata::block_q6_K & blk = base[(size_t)row * nb + bb];
            const kata::block_q8_K & ab = x[bb];
            const int j = 4 * lane;               // within a 32-value half
            int isum = 0;
            for (int half = 0; half < 2; ++half) {
                const int off = j + 32 * half;
                const unsigned char q4 = blk.ql[off >> 1];
                const int q = ((off & 1) ? (q4 >> 4) : (q4 & 0x0f)) |
                              (((blk.qh[off >> 2] >> (2 * (off & 3))) & 3) << 4);
                const int sc = blk.scales[(off) >> 4];
                int s = 0;
                for (int k = 0; k < 4; ++k) s += q * ab.qs[j + 32 * half + k];
                isum += sc * s;
            }
            acc += f16_dev(blk.d) * ab.d * (float)isum;
        }
    }
    for (int off = 4; off; off >>= 1) acc += __shfl_down_sync(0xffffffffull, acc, off);
    __shared__ float gsum[32];
    if (lane == 0) gsum[group] = acc;
    __syncthreads();
    if (threadIdx.x < rpb && grid_row + threadIdx.x < rows) {
        float s = 0.0f;
        for (int i = 0; i < (nb < 32 ? nb : 32); ++i) s += gsum[threadIdx.x * nb + i];
        out[(size_t)slab * rows + grid_row + threadIdx.x] = s;
    }
}

// The four experts a token picks sit 6.1 MB apart inside a stacked tensor, so a
// launch cannot cover them by pointer arithmetic alone and EXP-0086 measured 456
// launches/token = 1.089 ms. These two kernels take a DEVICE array of slab
// pointers instead, so one launch covers the whole layer without staging a copy
// of the weights: 3 launches per layer, no extra VRAM.
__global__ void gemv_q4_K_ptrs(const kata::block_q4_K * const * __restrict__ wptrs,
                               const kata::block_q8_K * __restrict__ x,
                               float * __restrict__ out,
                               int rows, int nb, int nslab) {
    const int rpb = nb <= 64 ? (64 / nb) : 1;
    const int nb_row_blocks = (rows + rpb - 1) / rpb;
    const int slab = blockIdx.x / nb_row_blocks;
    const int grid_row = (blockIdx.x % nb_row_blocks) * rpb;
    if (slab >= nslab || grid_row >= rows) return;
    const int lane = threadIdx.x & 3;
    const int group = threadIdx.x >> 2;
    const int r = group / nb;
    const int b = group % nb;
    float acc = 0.0f;
    if (r < rpb && grid_row + r < rows) {
        const int row = grid_row + r;
        const kata::block_q4_K * base = wptrs[slab];
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

__global__ void gemv_q6_K_ptrs(const kata::block_q6_K * const * __restrict__ wptrs,
                               const kata::block_q8_K * __restrict__ x,
                               float * __restrict__ out,
                               int rows, int nb, int nslab) {
    const int rpb = nb <= 32 ? (32 / nb) : 1;
    const int nb_row_blocks = (rows + rpb - 1) / rpb;
    const int slab = blockIdx.x / nb_row_blocks;
    const int grid_row = (blockIdx.x % nb_row_blocks) * rpb;
    if (slab >= nslab || grid_row >= rows) return;
    const int lane = threadIdx.x & 7;
    const int group = threadIdx.x >> 3;
    const int r = group / nb;
    const int b = group % nb;
    float acc = 0.0f;
    if (r < rpb && grid_row + r < rows) {
        const int row = grid_row + r;
        const kata::block_q6_K * base = wptrs[slab];
        for (int bb = b; bb < nb; bb += (nb <= 32 ? nb : 32)) {
            const kata::block_q6_K & blk = base[(size_t)row * nb + bb];
            const kata::block_q8_K & ab = x[bb];
            const int j = 4 * lane;
            int isum = 0;
            for (int half = 0; half < 2; ++half) {
                const int off = j + 32 * half;
                const unsigned char q4 = blk.ql[off >> 1];
                const int q = ((off & 1) ? (q4 >> 4) : (q4 & 0x0f)) |
                              (((blk.qh[off >> 2] >> (2 * (off & 3))) & 3) << 4);
                const int sc = blk.scales[(off) >> 4];
                int s = 0;
                for (int k = 0; k < 4; ++k) s += q * ab.qs[j + 32 * half + k];
                isum += sc * s;
            }
            acc += f16_dev(blk.d) * ab.d * (float)isum;
        }
    }
    for (int off = 4; off; off >>= 1) acc += __shfl_down_sync(0xffffffffull, acc, off);
    __shared__ float gsum[32];
    if (lane == 0) gsum[group] = acc;
    __syncthreads();
    if (threadIdx.x < rpb && grid_row + threadIdx.x < rows) {
        float s = 0.0f;
        for (int i = 0; i < (nb < 32 ? nb : 32); ++i) s += gsum[threadIdx.x * nb + i];
        out[(size_t)slab * rows + grid_row + threadIdx.x] = s;
    }
}

// The last thing the CPU needs from the GPU per layer is ONE 2048-float row,
// not four: this sums the experts with their router weights on the device. It
// cuts the D2H payload 4x and costs nothing (2048 floats x 4 adds).
__global__ void weighted_sum(const float * __restrict__ src, int nslab, int n,
                             const float * __restrict__ w, float * __restrict__ dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.0f;
    for (int s = 0; s < nslab; ++s) a += w[s] * src[(size_t)s * n + i];
    dst[i] = a;
}

double now_ms() {    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

// One layer's expert weights: gate/up share the q4_K kernel, down is q6_K.
struct LayerSlabs {
    const kata::block_q4_K * gate;
    const kata::block_q4_K * up;
    const kata::block_q6_K * down;
    int gate_nb, up_nb, down_nb, gate_rows, up_rows, down_rows;
};

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("usage: %s <model.gguf>\n", argv[0]); return 1; }
    std::setbuf(stdout, nullptr);
    kata::GgufFile g; g.open(argv[1]);
    const kata::GgufTensor * tg = g.find("blk.2.ffn_gate_exps.weight");
    const kata::GgufTensor * tu = g.find("blk.2.ffn_up_exps.weight");
    const kata::GgufTensor * td = g.find("blk.2.ffn_down_exps.weight");
    if (!tg || !tu || !td) { std::printf("missing blk.2 expert tensors\n"); return 1; }
    const int cols = (int)tg->dims[0], rows = (int)tg->dims[1], n_exp = (int)tg->dims[2];
    const int dn_rows = (int)td->dims[1], dn_cols = (int)td->dims[0];
    const int gate_nb = cols / kata::QK_K, down_nb = dn_cols / kata::QK_K;
    const size_t g_slab = (size_t)rows * gate_nb * sizeof(kata::block_q4_K);
    const size_t d_slab = (size_t)dn_rows * down_nb * sizeof(kata::block_q6_K);
    const size_t layer_b = 2 * g_slab + d_slab;
    const int n_moe = 38, n_used = 4;                 // LFM2.5-24B-A2B: 38 MoE layers, top-4
    const size_t total_b = (size_t)n_moe * n_used * layer_b;
    std::printf("one token's experts: %d layers x %d experts x %.1f MB = %.0f MB\n",
                n_moe, n_used, layer_b / 1e6, total_b / 1e6);

    // Deterministic distinct experts per layer -- the routing ids affect nothing
    // but they must differ, else all layers read one slab and the footprint the
    // kernel walks is 38x smaller than claimed (the same trap as EXP-0071).
    std::vector<uint32_t> pick((size_t)n_moe * n_used);
    for (int l = 0; l < n_moe; ++l)
        for (int i = 0; i < n_used; ++i) pick[(size_t)l * n_used + i] = (uint32_t)((l * 7 + i * 13) % n_exp);

    // Concatenate the picked slabs into one host blob, then one upload.
    std::vector<unsigned char> blob(total_b);
    std::vector<LayerSlabs> lay(n_moe);
    size_t off = 0;
    const auto * gbase = (const kata::block_q4_K *)g.data(*tg);
    const auto * dbase = (const kata::block_q6_K *)g.data(*td);
    for (int l = 0; l < n_moe; ++l) {
        for (int i = 0; i < n_used; ++i) {
            const uint32_t e = pick[(size_t)l * n_used + i];
            std::memcpy(blob.data() + off, gbase + (size_t)e * rows * gate_nb, g_slab);
            off += g_slab;
        }
        for (int i = 0; i < n_used; ++i) {
            const uint32_t e = pick[(size_t)l * n_used + i];
            std::memcpy(blob.data() + off, gbase + (size_t)e * rows * gate_nb, g_slab);
            off += g_slab;
        }
        for (int i = 0; i < n_used; ++i) {
            const uint32_t e = pick[(size_t)l * n_used + i];
            std::memcpy(blob.data() + off, dbase + (size_t)e * dn_rows * down_nb, d_slab);
            off += d_slab;
        }
    }
    void * dblob = nullptr;
    const double up_ms = [] { return 0.0; }();
    (void)up_ms;
    const double u0 = now_ms();
    CK(hipMalloc(&dblob, total_b));
    CK(hipMemcpy(dblob, blob.data(), total_b, hipMemcpyHostToDevice));
    const double upload_ms = now_ms() - u0;
    std::printf("upload %.0f MB in %.0f ms = %.1f GB/s (one-off)\n",
                total_b / 1e6, upload_ms, total_b / 1e6 / upload_ms);

    size_t cur = 0;
    for (int l = 0; l < n_moe; ++l) {
        auto * p = (unsigned char *)dblob + cur;
        lay[l].gate = (const kata::block_q4_K *)p; cur += n_used * g_slab;
        lay[l].up = (const kata::block_q4_K *)p;   cur += n_used * g_slab;
        lay[l].down = (const kata::block_q6_K *)p; cur += n_used * d_slab;
        lay[l].gate_nb = lay[l].up_nb = gate_nb; lay[l].down_nb = down_nb;
        lay[l].gate_rows = lay[l].up_rows = rows; lay[l].down_rows = dn_rows;
    }

    std::vector<float> act(cols);
    for (int i = 0; i < cols; ++i) act[i] = std::sin(0.001f * (float)i) * 1.7f;
    std::vector<kata::block_q8_K> hx(gate_nb);
    kata::quantize_row_q8_K(act.data(), hx.data(), cols);
    kata::block_q8_K * dx = nullptr;
    CK(hipMalloc(&dx, gate_nb * sizeof(kata::block_q8_K)));
    CK(hipMemcpy(dx, hx.data(), gate_nb * sizeof(kata::block_q8_K), hipMemcpyHostToDevice));
    float * dout = nullptr;
    CK(hipMalloc(&dout, (size_t)rows * n_used * sizeof(float)));

    const int rpb4 = gate_nb <= 64 ? 64 / gate_nb : 1;
    const int rpb6 = down_nb <= 32 ? 32 / down_nb : 1;
    const dim3 bl(256);
    auto token_pass = [&] {
        for (int l = 0; l < n_moe; ++l) {
            const dim3 gg4((unsigned)(((rows + rpb4 - 1) / rpb4) * n_used));
            gemv_q4_K_batch<<<gg4, bl>>>(lay[l].gate, dx, dout, rows, gate_nb, n_used);
            gemv_q4_K_batch<<<gg4, bl>>>(lay[l].up, dx, dout, rows, gate_nb, n_used);
            const dim3 gg6((unsigned)(((dn_rows + rpb6 - 1) / rpb6) * n_used));
            gemv_q6_K_batch<<<gg6, bl>>>(lay[l].down, dx, dout, dn_rows, down_nb, n_used);
        }
        CK(hipDeviceSynchronize());
    };

    // Verify before timing: the bucket kernel derives its own row mapping, and a
    // wrong mapping still runs at full speed (EXP-0077 shipped a 46-row bucket
    // that summed uninitialised slots and produced `inf`). Compare against the
    // CPU reference on the same weights and activations.
    {
        const size_t nb_all = (size_t)n_used * rows * gate_nb;
        std::vector<kata::block_q4_K> host_gate(nb_all);
        CK(hipMemcpy(host_gate.data(), dblob, nb_all * sizeof(kata::block_q4_K),
                     hipMemcpyDeviceToHost));
        std::vector<float> gpu_out((size_t)n_used * rows), cpu_out((size_t)n_used * rows);
        const int rpb = gate_nb <= 64 ? 64 / gate_nb : 1;
        const dim3 gg((unsigned)(((rows + rpb - 1) / rpb) * n_used));
        gemv_q4_K_batch<<<gg, dim3(256)>>>(lay[0].gate, dx, dout, rows, gate_nb, n_used);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(gpu_out.data(), dout, gpu_out.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
        for (int e = 0; e < n_used; ++e)
            for (int r = 0; r < rows; ++r)
                cpu_out[(size_t)e * rows + r] = kata::vec_dot_q4_K_q8_K_ref(
                    (size_t)cols, host_gate.data() + ((size_t)e * rows + r) * gate_nb, hx.data());
        double mx = 0, scale = 0;
        for (size_t i = 0; i < gpu_out.size(); ++i) {
            const double dv = std::fabs((double)gpu_out[i] - (double)cpu_out[i]);
            const double cv = std::fabs((double)cpu_out[i]);
            if (dv > mx) mx = dv;
            if (cv > scale) scale = cv;
        }
        std::printf("\nverify gate bucket: max|diff| %.3e on scale %.3e (%.1e rel)\n",
                    mx, scale, mx / scale);
        std::printf("  gpu[0..3] %.4f %.4f %.4f %.4f | cpu[0..3] %.4f %.4f %.4f %.4f\n",
                    gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3],
                    cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3]);
        if (!(mx / scale < 1e-4)) { std::printf("  FAIL: kernel disagrees with the CPU reference\n"); return 1; }
        std::printf("  PASS\n");
    }

    token_pass();                                     // warm
    const int reps = 5;
    const double t = [&] {
        const double a = now_ms();
        for (int i = 0; i < reps; ++i) token_pass();
        return (now_ms() - a) / reps;
    }();
    // `t` is already per-rep, so the launch count must not be multiplied by reps
    // (it was, which made the payload rate read 11503 GB/s -- above the card's
    // measured 764 ceiling, the check that caught it).
    const double launch_ms = n_moe * 3 * 2.77e-3;
    const double ev_ms = t - launch_ms;
    std::printf("\nMoE per token, all experts resident: %.3f ms (%d launches)\n", t, n_moe * 3);
    std::printf("  per layer %.3f ms | %.1f GB/s | launches cost %.2f ms\n",
                t / n_moe, total_b / 1e6 / t, launch_ms);
    std::printf("  payload-only %.3f ms = %.1f GB/s\n", ev_ms, total_b / 1e6 / ev_ms);
    std::printf("\nCPU engine profiles ffn = 14.20 ms/token (EXP-0082 probe), 22.5 ms total\n");
    std::printf("  -> MoE stage on GPU is %.2fx the CPU's\n", 14.20 / t);
    std::printf("  -> whole token would be %.2f ms = %.0f tok/s (CPU total minus ffn plus GPU MoE)\n",
                22.5 - 14.20 + t, 1000.0 / (22.5 - 14.20 + t));

    // ---- EXP-0086: what the same arithmetic costs as a real pipeline --------
    //
    // `t` above is the best case: 3 launches per layer covering all four slabs,
    // one activation row shared by all of them, no transfer. A real decode has
    // none of that -- the four experts of a layer sit 6 MB apart inside a
    // stacked tensor, so the honest grid is one launch per expert matrix, and
    // the residual stream crosses PCIe twice per layer.
    //
    // Arm B adds the launches (12 per layer instead of 3, slab pointers kept).
    // Arm C adds one H2D of the q8_K row and one D2H of 4x2048 floats per layer,
    // synchronous, through pinned buffers -- the shape EXP-0081 priced at ~20 us.
    // The down projection still shares one activation row: with per-expert
    // activations it would read 4 rows instead of 1, which is 1.2 KB of a
    // 6.1 MB slab and cannot move the number.
    {
        kata::block_q8_K * hx_pin = nullptr;
        float * hh_out = nullptr;
        CK(hipHostMalloc((void **)&hx_pin, gate_nb * sizeof(kata::block_q8_K), hipHostMallocDefault));
        CK(hipHostMalloc((void **)&hh_out, (size_t)rows * n_used * sizeof(float), hipHostMallocDefault));
        std::memcpy(hx_pin, hx.data(), gate_nb * sizeof(kata::block_q8_K));
        const int blocks_in = gate_nb * (int)sizeof(kata::block_q8_K);
        const int blocks_out = (int)((size_t)rows * n_used * sizeof(float));

        auto pass_per_expert = [&](bool xfer) {
            for (int l = 0; l < n_moe; ++l) {
                if (xfer) CK(hipMemcpyAsync(dx, hx_pin, blocks_in, hipMemcpyHostToDevice));
                for (int e = 0; e < n_used; ++e) {
                    const dim3 g4((unsigned)((rows + rpb4 - 1) / rpb4));
                    gemv_q4_K_batch<<<g4, bl>>>(lay[l].gate + (size_t)e * rows * gate_nb, dx, dout,
                                                rows, gate_nb, 1);
                    gemv_q4_K_batch<<<g4, bl>>>(lay[l].up + (size_t)e * rows * gate_nb, dx, dout,
                                                rows, gate_nb, 1);
                    const dim3 g6((unsigned)((dn_rows + rpb6 - 1) / rpb6));
                    gemv_q6_K_batch<<<g6, bl>>>(lay[l].down + (size_t)e * dn_rows * down_nb, dx,
                                                dout, dn_rows, down_nb, 1);
                }
                if (xfer) CK(hipMemcpyAsync(hh_out, dout, blocks_out, hipMemcpyDeviceToHost));
                if (xfer) CK(hipDeviceSynchronize());
            }
            if (!xfer) CK(hipDeviceSynchronize());
        };

        pass_per_expert(false);                                  // warm
        auto bench = [&](auto body) {
            const double a = now_ms();
            for (int i = 0; i < 5; ++i) body();
            return (now_ms() - a) / 5.0;
        };
        const double tb = bench([&] { pass_per_expert(false); });
        const double tc = bench([&] { pass_per_expert(true); });
        const int launches = n_moe * n_used * 3;
        std::printf("\nEXP-0086 pipeline, %d launches/token, %.2f GB resident:\n", launches, total_b / 1e9);
        std::printf("  A grouped, no transfer      %.3f ms\n", t);
        std::printf("  B per expert, no transfer   %.3f ms (+%.3f = %.1f us x %d launches)\n",
                    tb, tb - t, 1e3 * (tb - t) / launches, launches);
        std::printf("  C + H2D/D2H per layer       %.3f ms (+%.3f = %.1f us x %d layers)\n",
                    tc, tc - tb, 1e3 * (tc - tb) / n_moe, n_moe);
        std::printf("  -> token: 22.5 - 14.20 + %.3f = %.2f ms = %.1f tok/s (CPU MoE 14.20)\n",
                    tc, 22.5 - 14.20 + tc, 1e3 / (22.5 - 14.20 + tc));
        if (!(tc > t)) std::printf("  NOTE: pipeline arm is not slower -- check the transfer path\n");
        (void)hipHostFree(hx_pin);
        (void)hipHostFree(hh_out);
    }

    // ---- EXP-0087: kill the launches and the round trip ---------------------
    //
    // Arm C's two costs have separate fixes and neither needs a second copy of
    // the weights:
    //
    //   launches  -> a device array of slab pointers lets ONE launch cover all
    //                four experts (3 launches per layer instead of 12).
    //   handoff   -> sum the four expert outputs with their router weights ON
    //                the device, so the D2H payload is one row of 2048 floats
    //                instead of four (4x less to copy, same one wait).
    //
    // All arms share one activation row, so the down outputs here are four
    // responses to the same input; that makes the weighted sum a legitimate
    // stand-in for the combine (it measures launch and transfer cost, which is
    // what this arm is for) but NOT a correctness statement about real routing.
    {
        kata::block_q4_K const ** dgate = nullptr, ** dup = nullptr;
        kata::block_q6_K const ** ddown = nullptr;
        float * dw = nullptr, * dsout = nullptr, * dcomb = nullptr;
        float * hcomb = nullptr;
        const size_t nptr = (size_t)n_moe * (size_t)n_used;
        CK(hipMalloc((void **)&dgate, nptr * sizeof(void *)));
        CK(hipMalloc((void **)&dup, nptr * sizeof(void *)));
        CK(hipMalloc((void **)&ddown, nptr * sizeof(void *)));
        std::vector<kata::block_q4_K const *> hgate(nptr), hup(nptr);
        std::vector<kata::block_q6_K const *> hdown(nptr);
        // Layer l owns slabs [l*n_used, (l+1)*n_used): the kernel indexes
        // wptrs[slab] with slab in [0, n_used), so the base pointer handed to
        // launch l must already be offset by l*n_used.
        for (int l = 0; l < n_moe; ++l)
            for (int e = 0; e < n_used; ++e) {
                const size_t i = (size_t)l * n_used + e;
                hgate[i] = lay[l].gate + (size_t)e * rows * gate_nb;
                hup[i] = lay[l].up + (size_t)e * rows * gate_nb;
                hdown[i] = lay[l].down + (size_t)e * dn_rows * down_nb;
            }
        CK(hipMemcpy(dgate, hgate.data(), nptr * sizeof(void *), hipMemcpyHostToDevice));
        CK(hipMemcpy(dup, hup.data(), nptr * sizeof(void *), hipMemcpyHostToDevice));
        CK(hipMemcpy(ddown, hdown.data(), nptr * sizeof(void *), hipMemcpyHostToDevice));
        CK(hipMalloc(&dw, (size_t)n_used * sizeof(float)));
        CK(hipMalloc(&dsout, (size_t)n_used * dn_rows * sizeof(float)));
        CK(hipMalloc(&dcomb, (size_t)dn_rows * sizeof(float)));
        CK(hipHostMalloc((void **)&hcomb, (size_t)dn_rows * sizeof(float), hipHostMallocDefault));
        const std::vector<float> hw(n_used, 0.25f);
        CK(hipMemcpy(dw, hw.data(), hw.size() * sizeof(float), hipMemcpyHostToDevice));

        const int g4r = (rows + rpb4 - 1) / rpb4;
        const int g6r = (dn_rows + rpb6 - 1) / rpb6;
        auto pass_ptrs = [&] {
            for (int l = 0; l < n_moe; ++l) {
                gemv_q4_K_ptrs<<<g4r * n_used, bl>>>(dgate + (size_t)l * n_used, dx, dout, rows, gate_nb, n_used);
                gemv_q4_K_ptrs<<<g4r * n_used, bl>>>(dup + (size_t)l * n_used, dx, dout, rows, gate_nb, n_used);
                gemv_q6_K_ptrs<<<g6r * n_used, bl>>>(ddown + (size_t)l * n_used, dx, dsout, dn_rows, down_nb, n_used);
                weighted_sum<<<(dn_rows + 255) / 256, bl>>>(dsout, n_used, dn_rows, dw, dcomb);
            }
            CK(hipDeviceSynchronize());
        };

        // correctness of the pointer kernel against the slab kernel it replaces
        {
            std::vector<float> a1((size_t)n_used * dn_rows), a2((size_t)n_used * dn_rows);
            const dim3 g((unsigned)(g6r * n_used));
            gemv_q6_K_batch<<<g, bl>>>(lay[0].down, dx, dsout, dn_rows, down_nb, n_used);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(a1.data(), dsout, a1.size() * sizeof(float), hipMemcpyDeviceToHost));
            gemv_q6_K_ptrs<<<g, bl>>>(ddown, dx, dsout, dn_rows, down_nb, n_used);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(a2.data(), dsout, a2.size() * sizeof(float), hipMemcpyDeviceToHost));
            double mx = 0, scale = 0;
            for (size_t i = 0; i < a1.size(); ++i) {
                mx = std::max(mx, std::fabs((double)a1[i] - (double)a2[i]));
                scale = std::max(scale, std::fabs((double)a1[i]));
            }
            std::printf("\nverify ptr kernel vs slab kernel: max|diff| %.3e on %.3e (%.1e rel)\n",
                        mx, scale, mx / scale);
            // bit-identical is the only acceptable answer: same arithmetic, only
            // the base pointer changed.
            if (!(mx == 0.0)) { std::printf("  FAIL: pointer kernel differs from the slab kernel\n"); return 1; }
            std::printf("  PASS (bit-identical)\n");
        }

        pass_ptrs();                                   // warm
        auto bench = [&](auto body) {
            const double a = now_ms();
            for (int i = 0; i < 5; ++i) body();
            return (now_ms() - a) / 5.0;
        };
        const double td = bench([&] {
            for (int l = 0; l < n_moe; ++l) {
                gemv_q4_K_ptrs<<<g4r * n_used, bl>>>(dgate + (size_t)l * n_used, dx, dout, rows, gate_nb, n_used);
                gemv_q4_K_ptrs<<<g4r * n_used, bl>>>(dup + (size_t)l * n_used, dx, dout, rows, gate_nb, n_used);
                gemv_q6_K_ptrs<<<g6r * n_used, bl>>>(ddown + (size_t)l * n_used, dx, dsout, dn_rows, down_nb, n_used);
                weighted_sum<<<(dn_rows + 255) / 256, bl>>>(dsout, n_used, dn_rows, dw, dcomb);
                CK(hipMemcpyAsync(hcomb, dcomb, (size_t)dn_rows * sizeof(float), hipMemcpyDeviceToHost));
                CK(hipDeviceSynchronize());
            }
        });
        const int launches_d = n_moe * 4;
        std::printf("\nEXP-0087 pointer launches + device combine:\n");
        std::printf("  D %d launches (%d/layer), %.1f KB D2H/layer  %.3f ms\n",
                    launches_d, launches_d / n_moe, (double)dn_rows * 4 / 1024.0, td);
        std::printf("  -> token: 22.5 - 14.20 + %.3f = %.2f ms = %.1f tok/s\n",
                    td, 22.5 - 14.20 + td, 1e3 / (22.5 - 14.20 + td));
        std::printf("  vs arm C 3.663 ms -> %.2fx\n", 3.663 / td);
        if (!(td < 3.663)) std::printf("  NOTE: arm D did not beat C\n");

        // ---- EXP-0088: what a device router would actually buy ------------
        //
        // Arm D's remaining cost is 38 waits: the host router sits inside the
        // layer loop, so each layer needs its experts decided before its
        // launches. A device router removes the wait entirely -- so measure the
        // wait by removing it, keeping the launches identical. One sync for the
        // whole token instead of 38.
        //
        // This is a FLOOR, not a router: the activation row is constant across
        // layers here, which is exactly the dependency a real router would have
        // to satisfy (layer i's output feeding layer i+1's router). A real
        // device router must pay one 2048-wide dot per expert per layer, which
        // is 0.5 MB of weights against this arm's 24 MB/layer -- 2% of the
        // stream, so the floor is close to the achievable number.
        auto pass_no_sync = [&] {
            for (int l = 0; l < n_moe; ++l) {
                gemv_q4_K_ptrs<<<g4r * n_used, bl>>>(dgate + (size_t)l * n_used, dx, dout, rows, gate_nb, n_used);
                gemv_q4_K_ptrs<<<g4r * n_used, bl>>>(dup + (size_t)l * n_used, dx, dout, rows, gate_nb, n_used);
                gemv_q6_K_ptrs<<<g6r * n_used, bl>>>(ddown + (size_t)l * n_used, dx, dsout, dn_rows, down_nb, n_used);
                weighted_sum<<<(dn_rows + 255) / 256, bl>>>(dsout, n_used, dn_rows, dw, dcomb);
            }
            CK(hipDeviceSynchronize());
        };
        pass_no_sync();                                // warm
        const double te = bench(pass_no_sync);
        std::printf("\nEXP-0088 no per-layer wait (1 sync/token, %d launches):\n", launches_d);
        std::printf("  E %.3f ms  (D minus per-layer waits: %.3f ms = %.1f us x %d layers)\n",
                    te, td - te, 1e3 * (td - te) / n_moe, n_moe);
        std::printf("  -> token: 22.5 - 14.20 + %.3f = %.2f ms = %.1f tok/s\n",
                    te, 22.5 - 14.20 + te, 1e3 / (22.5 - 14.20 + te));
        if (!(te < td)) std::printf("  NOTE: removing the waits did not help -- they were overlapped\n");

        (void)hipHostFree(hcomb);
        CK(hipFree(dgate));
        CK(hipFree(dup));
        CK(hipFree(ddown));
        CK(hipFree(dw));
        CK(hipFree(dsout));
        CK(hipFree(dcomb));
    }
    return 0;
}

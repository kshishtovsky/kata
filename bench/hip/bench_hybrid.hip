// EXP-0082: can the CPU and the GPU stream the expert weights AT THE SAME TIME,
// and does their bandwidth add up?
//
// EXP-0080/0081 modelled a hybrid decode as max(cpu_i, gpu_i) per layer, which
// is only true if the two devices' bandwidths are independent. They are not
// obviously so: the GPU reads its weights over PCIe through the CPU's memory
// controller, so it could steal the very bandwidth the CPU shares depend on.
//
// This measures the one number that decides it: aggregate MB/ms when all the
// expert traffic of one layer is split across both devices and run
// concurrently, against the same traffic run on the CPU alone.
//
// Build: hipcc -O3 bench/hip/bench_hybrid.hip -o /tmp/hyb -I include \
//              -L build -lkata_core -lpthread
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "kata/gguf.h"
#include "kata/gemv.h"
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

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("usage: %s <model.gguf>\n", argv[0]); return 1; }
    std::setbuf(stdout, nullptr);
    kata::GgufFile g; g.open(argv[1]);
    const kata::GgufTensor * t = g.find("blk.2.ffn_gate_exps.weight");
    if (!t) { std::printf("no blk.2.ffn_gate_exps.weight\n"); return 1; }
    const size_t cols = t->dims[0], rows = t->dims[1], n_exp = t->dims[2];
    const size_t nb = cols / kata::QK_K;
    const size_t slab_b = rows * nb * sizeof(kata::block_q4_K);
    const auto * base = (const kata::block_q4_K *)g.data(*t);
    std::printf("%zu experts, %zu x %zu each, %.2f MB/slab, %.1f MB total\n",
                n_exp, rows, cols, slab_b / 1e6, n_exp * slab_b / 1e6);

    std::vector<float> act(cols);
    for (size_t i = 0; i < cols; ++i) act[i] = std::sin(0.001f * (float)i) * 1.7f;
    std::vector<kata::block_q8_K> hx(nb);
    kata::quantize_row_q8_K(act.data(), hx.data(), cols);

    // Whole layer staged in VRAM: the fused launch then reads all of it.
    kata::block_q4_K * dw = nullptr;
    CK(hipMalloc(&dw, n_exp * slab_b));
    CK(hipMemcpy(dw, base, n_exp * slab_b, hipMemcpyHostToDevice));
    // Activation row mapped, so the kernel reads it straight from host memory.
    kata::block_q8_K * hx_map = nullptr;
    CK(hipHostMalloc((void **)&hx_map, nb * sizeof(kata::block_q8_K) + 4096,
                     hipHostMallocMapped | hipHostMallocCoherent));
    std::memcpy(hx_map, hx.data(), nb * sizeof(kata::block_q8_K));
    kata::block_q8_K * dx = nullptr;
    CK(hipHostGetDevicePointer((void **)&dx, hx_map, 0));
    // Second activation copy held in VRAM. Which one is faster is measured, not
    // assumed: the kernel reads the row once PER BLOCK (12288 blocks), so a
    // mapped host pointer turns 2.3 KB of setup into megabytes of PCIe traffic.
    kata::block_q8_K * dx_vram = nullptr;
    CK(hipMalloc(&dx_vram, nb * sizeof(kata::block_q8_K)));
    CK(hipMemcpy(dx_vram, hx.data(), nb * sizeof(kata::block_q8_K), hipMemcpyHostToDevice));
    float * dout = nullptr;
    CK(hipMalloc(&dout, n_exp * rows * sizeof(float)));

    kata::ThreadPool pool(kata::ThreadPool::default_threads());
    std::vector<float> hout(n_exp * rows);

    const int rpb = nb <= 64 ? (int)(64 / nb) : 1;
    const dim3 bl(256);
    auto gpu_run_x = [&](size_t n, kata::block_q8_K * act_ptr) {
        const dim3 gr((unsigned)(((rows + rpb - 1) / rpb) * n));
        gemv_q4_K_batch<<<gr, bl>>>(dw, act_ptr, dout, (int)rows, (int)nb, (int)n);
        CK(hipDeviceSynchronize());
    };
    auto cpu_run = [&](size_t from, size_t n) {
        for (size_t e = from; e < from + n; ++e)
            kata::gemv_q4_K_q8_K_mt(rows, cols, hout.data() + e * rows,
                                    base + e * rows * nb, hx.data(), pool);
    };

    auto gpu_run = [&](size_t n) { gpu_run_x(n, dx_vram); };

    gpu_run(n_exp); gpu_run_x(n_exp, dx); cpu_run(0, n_exp);   // warm both

    auto bench = [&](auto body) {
        const double a = now_ms();
        body();
        return now_ms() - a;
    };
    const int reps = 5;

    const double cpu_all = bench([&] { for (int i = 0; i < reps; ++i) cpu_run(0, n_exp); }) / reps;
    const double gpu_all = bench([&] { for (int i = 0; i < reps; ++i) gpu_run(n_exp); }) / reps;
    std::printf("\nwhole layer, %zu experts (%zu KB):\n", n_exp, n_exp * slab_b / 1024);
    std::printf("  CPU only   %7.3f ms = %6.1f GB/s\n", cpu_all, n_exp * slab_b / 1e6 / cpu_all);
    std::printf("  GPU only   %7.3f ms = %6.1f GB/s   (activation in VRAM)\n",
                gpu_all, n_exp * slab_b / 1e6 / gpu_all);
    const double gpu_map = bench([&] {
        for (int i = 0; i < reps; ++i) gpu_run_x(n_exp, dx); }) / reps;
    std::printf("  GPU only   %7.3f ms = %6.1f GB/s   (activation zero-copy from host)\n",
                gpu_map, n_exp * slab_b / 1e6 / gpu_map);

    // Split: GPU takes the tail, CPU the head, both at once.
    std::printf("\n  split, concurrent:\n");
    std::printf("  %-6s %-9s %-9s %-9s %-9s %s\n",
                "gpu", "cpu ms", "gpu ms", "serial", "concurrent", "gain");
    for (size_t ngpu : {8.0, 16.0, 32.0, 44.0, 54.0}) {
        const size_t n_cpu = n_exp - (size_t)ngpu;
        if (n_cpu < 2) continue;
        const double c = bench([&] { for (int i = 0; i < reps; ++i) cpu_run(0, n_cpu); }) / reps;
        const double gg = bench([&] { for (int i = 0; i < reps; ++i) gpu_run((size_t)ngpu); }) / reps;
        double both = 0;
        for (int i = 0; i < reps; ++i) {
            const double a = now_ms();
            std::thread gth([&] { gpu_run((size_t)ngpu); });
            cpu_run(0, n_cpu);
            gth.join();
            both += now_ms() - a;
        }
        both /= reps;
        std::printf("  %-6zu %-9.3f %-9.3f %-9.3f %-9.3f %.2fx\n",
                    (size_t)ngpu, c, gg, c + gg, both, (c + gg) / both);
    }
    // The shape that decode actually runs: n_expert_used = 4 slabs per layer,
    // 24.5 MB, comfortably inside the 96 MB L3. This is the arm that decides
    // whether a hybrid speeds up a real token -- the cold whole-layer sweep
    // above is a stress test, not a forecast.
    std::printf("\nreal decode shape: 4 experts/layer (%.1f MB), split\n", 4 * slab_b / 1e6);
    std::printf("  %-10s %-9s %-9s %-9s %s\n", "cpu:gpu", "cpu ms", "gpu ms", "concurrent", "vs cpu-only");
    const double cpu4 = bench([&] { for (int i = 0; i < reps * 8; ++i) cpu_run(0, 4); }) / (reps * 8);
    std::printf("  cpu only   %-9.3f %-9s %-9s %s\n", cpu4, "-", "-", "1.00x");
    for (int ngpu = 1; ngpu <= 3; ++ngpu) {
        const size_t n_cpu = 4 - (size_t)ngpu;
        const double c = bench([&] { for (int i = 0; i < reps * 8; ++i) cpu_run(0, n_cpu); }) / (reps * 8);
        const double gg = bench([&] { for (int i = 0; i < reps * 8; ++i) gpu_run((size_t)ngpu); }) / (reps * 8);
        double both = 0;
        for (int i = 0; i < reps * 8; ++i) {
            const double a = now_ms();
            std::thread gth([&] { gpu_run((size_t)ngpu); });
            cpu_run(0, n_cpu);
            gth.join();
            both += now_ms() - a;
        }
        both /= reps * 8;
        char label[16]; std::snprintf(label, sizeof label, "%zu:%d", n_cpu, ngpu);
        std::printf("  %-10s %-9.3f %-9.3f %-9.3f %.2fx\n",
                    label, c, gg, both, cpu4 / both);
    }
    // The arm that decides the hybrid: one layer's MoE work at the REAL per-layer
    // footprint (4 experts x gate+up+down = 24.5 MB) with the layers cycled so
    // nothing stays in L3. 24.5 MB fits L3; 38 of them (931 MB) do not, which is
    // why a real decode runs at 93% of the DRAM ceiling and a single-layer bench
    // would flatter the CPU.
    {
        const kata::GgufTensor * up = g.find("blk.2.ffn_up_exps.weight");
        const kata::GgufTensor * dn = g.find("blk.2.ffn_down_exps.weight");
        const size_t up_rows = up->dims[1], up_cols = up->dims[0];
        const size_t dn_rows = dn->dims[1], dn_cols = dn->dims[0];
        const size_t up_slab = up_rows * (up_cols / kata::QK_K) * sizeof(kata::block_q4_K);
        const size_t dn_slab = dn_rows * (dn_cols / kata::QK_K) * sizeof(kata::block_q6_K);
        const double layer_mb = (2 * slab_b + up_slab + dn_slab) / 1e6;
        std::printf("\nreal layer footprint: gate %.2f + up %.2f + down %.2f MB = %.1f MB for 4 experts\n",
                    slab_b / 1e6, up_slab / 1e6, dn_slab / 1e6, 3 * layer_mb);
        // Cycle every layer's 4 experts: 38 x 24.5 MB = 931 MB, exactly the
        // traffic EXP-0071 measured for one decode token.
        const size_t n_layers = 38;
        double cpu_t = 0, gpu_t = 0, both_t = 0;
        for (size_t L = 0; L < n_layers; ++L) {
            const kata::block_q4_K * ga = base + ((L * 4) % n_exp) * rows * nb;
            const kata::block_q4_K * ua = (const kata::block_q4_K *)g.data(*up) +
                                          ((L * 4) % n_exp) * up_rows * (up_cols / kata::QK_K);
            const kata::block_q6_K * da = (const kata::block_q6_K *)g.data(*dn) +
                                          ((L * 4) % n_exp) * dn_rows * (dn_cols / kata::QK_K);
            const double a = now_ms();
            for (size_t e = 0; e < 4; ++e) {
                kata::gemv_q4_K_q8_K_mt(rows, cols, hout.data(), ga + e * rows * nb, hx.data(), pool);
                kata::gemv_q4_K_q8_K_mt(up_rows, up_cols, hout.data(),
                                        (const kata::block_q4_K *)ua + e * up_rows * (up_cols / kata::QK_K),
                                        hx.data(), pool);
                kata::gemv_q6_K_q8_K_mt(dn_rows, dn_cols, hout.data(),
                                        (const kata::block_q6_K *)da + e * dn_rows * (dn_cols / kata::QK_K),
                                        hx.data(), pool);
            }
            cpu_t += now_ms() - a;
        }
        cpu_t /= n_layers;
        std::printf("  CPU only, 38 layers cycled: %.3f ms/layer = %.2f ms/token of MoE\n",
                    cpu_t, cpu_t * n_layers);
        std::printf("  (that against a 30.0 ms CPU token = MoE is %.0f%%)\n",
                    100.0 * cpu_t * n_layers / 30.0);
        std::printf("  GPU would need %.3f ms/layer at 268 GB/s, plus one 28 us handoff\n",
                    layer_mb * 1e3 / 268.0);
    }

    std::printf("\nCPUs used by the pool: %u\n", pool.size());
    return 0;
}

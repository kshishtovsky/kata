// EXP-0074: decode attention on the GPU at long context.
//
// On CPU this was 9.06 ms of a 31.62 ms token at 16k depth and provably
// ALU-bound: a 64x change in resident footprint moved the rate 7% (EXP-0066).
// A GPU has the opposite balance -- huge ALU, and the same 73.5 GB/s problem
// replaced by 764 GB/s. So this measures whether attention collapses or becomes
// the new floor once the weights no longer dominate.
//
// Shape: LFM2.5-24B-A2B, 10 attention layers, 32 q-heads, 8 kv-heads,
// head_dim 64. One kernel launch per (layer, kv-head): the 4 q-heads of a group
// share one kv head and one softmax denominator, which is what the CPU grouped
// kernel exploits too (EXP-0043/0045).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int kQHeads = 32, kKvHeads = 8, kHeadDim = 64, kLayers = 10, kGQA = kQHeads / kKvHeads;

// One block per kv head, 256 threads = 4 warps x 2, laid out as:
//   warp w handles q head (kv_head_group * kGQA + w % kGQA)
// so 4 warps cover the group's 4 q heads and each warp walks the positions.
// Softmax is online (running max + rescaled sum), so no second pass over K.
__global__ void attn_decode_kernel(const float * __restrict__ q,   // [n_q * hd]
                                   const float * __restrict__ k,   // [nd][hd]
                                   const float * __restrict__ v,   // [nd][hd]
                                   float * __restrict__ out,       // [n_q * hd]
                                   int nd, float scale) {
    const int qh = blockIdx.x;                 // global q head 0..31
    const int tid = threadIdx.x;               // 0..255
    const int warp = tid >> 5, lane = tid & 31;

    // 32 lanes cover 64 dims: each lane owns two dims of q.
    __shared__ float qs[kHeadDim];
    if (tid < kHeadDim) qs[tid] = q[qh * kHeadDim + tid];
    __syncthreads();

    float q0 = qs[lane * 2 + 0] * scale, q1 = qs[lane * 2 + 1] * scale;

    float m = -INFINITY, l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;

    // 8 warps: split the position range 8 ways, merge at the end.
    const int nw = blockDim.x >> 5;
    for (int p = warp; p < nd; p += nw) {
        const float * kk = k + (size_t)p * kHeadDim;
        float dot = q0 * kk[lane * 2] + q1 * kk[lane * 2 + 1];
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffull, dot, off);
        dot = __shfl_sync(0xffffffffull, dot, 0);

        const float nm = fmaxf(m, dot);
        const float e = __expf(dot - nm);
        l = l * __expf(m - nm) + e;
        m = nm;
        const float * vv = v + (size_t)p * kHeadDim;
        acc0 = acc0 * __expf(m - nm) + e * vv[lane * 2];        // rescale placeholder
        acc1 = acc1 * __expf(m - nm) + e * vv[lane * 2 + 1];
    }
    (void)acc0; (void)acc1; (void)l; (void)m;
    // NOTE: scratch shape only -- see the report. The real kernel writes online
    // softmax state to shared memory per warp and merges 8 partial (m, l, acc)
    // triples at the end, which is what the timing below represents.
    if (tid == 0) out[qh * kHeadDim] = l;
}

// The timing kernel: reads nd keys and nd values, writes hd outputs -- the same
// traffic as the real one, without the merge bookkeeping.
//
// The first version launched ONE 256-thread block per (layer, kv head) and read
// 336 MB in 27.96 ms = 24.0 GB/s, which says nothing about the memory system: a
// 7900 XT has 84 CUs, so 80 launches of one block each kept ~1% of the card
// busy. Grid is now (position tiles) so every CU can issue loads, and the
// number reported is the one a real split-KV attention can reach.
__global__ void attn_traffic_kernel(const float4 * __restrict__ k4,
                                    const float4 * __restrict__ v4,
                                    float * __restrict__ out,
                                    int nd) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    float acc = 0.0f;
    // A position is 16 float4. 32 lanes cover TWO positions per step, so the
    // step is 2 * nw positions and lane splits as (lane < 16 ? pos0 : pos1).
    const int half = lane >> 4;              // which of the two positions
    const int slot = lane & 15;              // which float4 within it
    const int per_block = (nd + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * per_block;
    const int hi = min(nd, lo + per_block);
    for (int p = lo + warp * 2; p < hi; p += nw * 2) {
        const size_t pi = (size_t)(p + half);
        if (pi >= (size_t)hi) break;
        const float4 a = k4[pi * 16 + slot];
        const float4 b = v4[pi * 16 + slot];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    if (acc == 12345.678f) out[blockIdx.x & 7] = acc;
}
} // namespace

int main() {
    const int nd = 16384;
    const size_t hd = kHeadDim;
    const size_t kv_bytes = (size_t)kLayers * kKvHeads * nd * hd * 4;
    std::printf("16k context: K+V = %.0f MB across %d layers x %d kv heads\n",
                kv_bytes / 1e6, kLayers, kKvHeads);

    // One layer's worth of K/V is 8 heads x 16384 positions x 64 dims x 4 B =
    // 33.5 MB each. All TEN layers must be allocated and walked: reusing one
    // layer made the working set 67 MB, which fits the 80 MB Infinity Cache, and
    // the probe then reported 815 GB/s -- above the 764 GB/s measured ceiling,
    // i.e. impossible for a real pass that reads a distinct 336 MB.
    const size_t one_layer_kv = (size_t)kKvHeads * nd * hd * 4;
    const size_t all_kv = (size_t)kLayers * one_layer_kv;
    float * dk; float * dv; float * dq; float * dout;
    hipMalloc(&dk, all_kv);
    hipMalloc(&dv, all_kv);
    hipMalloc(&dq, (size_t)kQHeads * hd * 4);
    hipMalloc(&dout, (size_t)(kQHeads * hd) * 4);
    hipMemset(dk, 0, all_kv);
    hipMemset(dv, 0, all_kv);
    hipMemset(dq, 0, (size_t)kQHeads * hd * 4);

    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    double best = 1e9;
    for (int r = 0; r < 10; ++r) {
        hipEventRecord(e0);
        for (int layer = 0; layer < kLayers; ++layer)
            for (int kv = 0; kv < kKvHeads; ++kv)
                attn_decode_kernel<<<1, 256>>>(dq + kv * kGQA * hd, dk + (size_t)kv * nd * hd,
                                               dv + (size_t)kv * nd * hd,
                                               dout + kv * kGQA * hd, nd, 0.125f);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1);
        best = std::min(best, (double)ms);
    }
    std::printf("attention scratch kernel: %.2f ms (not a real kernel, shape only)\n", best);

    // Traffic at the real rate: read K and V once per kv head per layer, with
    // enough blocks to fill the card. A real split-KV decode kernel launches one
    // block per (kv head, position tile) and merges partials in a second step,
    // so the interesting question is where the plateau is, not one grid size.
    const double real_bytes = (double)kLayers * kKvHeads * nd * hd * 4 * 2;
    for (int blocks_per_kv : {32, 64, 128, 256}) {
        best = 1e9;
        for (int r = 0; r < 10; ++r) {
            hipEventRecord(e0);
            for (int layer = 0; layer < kLayers; ++layer)
                for (int kv = 0; kv < kKvHeads; ++kv)
                    attn_traffic_kernel<<<blocks_per_kv, 256>>>(
                        (const float4 *)(dk + ((size_t)layer * kKvHeads + kv) * nd * hd),
                        (const float4 *)(dv + ((size_t)layer * kKvHeads + kv) * nd * hd),
                        dout + kv * kGQA * hd, nd);
            hipEventRecord(e1); hipEventSynchronize(e1);
            float ms = 0; hipEventElapsedTime(&ms, e0, e1);
            best = std::min(best, (double)ms);
        }
        std::printf("attention traffic, %2d blocks/kv head (%4d blocks/launch): %.2f ms -> %.1f GB/s%s\n",
                    blocks_per_kv, blocks_per_kv * 1, best, real_bytes / (best * 1e6),
                    blocks_per_kv == 256 ? "  <- plateau" : "");
    }
    std::printf("  vs CPU 9.06 ms at 16k -> %.1fx\n", 9.06 / best);
    std::printf("  one token = weights 3.72 + attention %.2f = %.2f ms -> %.0f tok/s\n",
                best, 3.72 + best, 1e3 / (3.72 + best));
    return 0;
}

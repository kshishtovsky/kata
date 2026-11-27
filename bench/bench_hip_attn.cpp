// EXP-0074/0075: decode attention on the GPU at long context.
//
// On CPU this was 9.06 ms of a 31.62 ms token at 16k depth and provably
// ALU-bound: a 64x change in resident footprint moved the rate 7% (EXP-0066).
// A GPU has the opposite balance -- huge ALU, and the same 73.5 GB/s problem
// replaced by 764 GB/s. So this measures whether attention collapses or becomes
// the new floor once the weights no longer dominate.
//
// Three parts:
//   1. the traffic floor -- read K and V once per (layer, kv head) with enough
//      blocks to fill the card. No real kernel can beat it.
//   2. a REAL kernel at that floor's block count, checked against kata's CPU
//      grouped decode.
//   3. a one-lane-per-position variant, kept because it was bit-identical to
//      the CPU and still 4x slower -- the reason coalescing beats arithmetic
//      here, recorded rather than remembered.
//
// Shape: LFM2.5-24B-A2B, 10 attention layers, 32 q-heads, 8 kv-heads,
// head_dim 64. The 4 q-heads of a group share one kv head and one softmax
// denominator, as the CPU grouped kernel exploits too (EXP-0043/0045).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "kata/attn.h"

namespace {
constexpr int kQHeads = 32, kKvHeads = 8, kHeadDim = 64, kLayers = 10;
constexpr int kGQA = kQHeads / kKvHeads;
constexpr int kThreads = 128;              // 4 warps, one per q head of a group
constexpr int kWarps = kThreads / 32;

struct Partial {
    float m[kGQA];
    float l[kGQA];
    float acc[kGQA][kHeadDim];
};

// ---------------------------------------------------------------- traffic floor
// Reads nd keys and nd values, writes nothing useful. The grid is
// (position tiles), because a 7900 XT has 84 CUs and one block per kv head
// keeps ~1% of the card busy -- that mistake read 24.0 GB/s and looked like
// "the GPU is bad at attention".
__global__ void attn_traffic_kernel(const float4 * __restrict__ k4,
                                    const float4 * __restrict__ v4,
                                    float * __restrict__ out,
                                    int nd) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    float acc = 0.0f;
    // A position is 16 float4. 32 lanes cover TWO positions per step, so the
    // step is 2 * nw positions.
    const int half = lane >> 4;
    const int slot = lane & 15;
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

// ------------------------------------------------------------- the real kernel
// Warps split POSITIONS, not q heads. Giving each warp one q head of the group
// was correct (3.4e-08) but read K and V four times over, because all four
// warps walked the same tile: 1.95 ms against a 1.14 ms floor. Here one warp
// carries all kGQA q heads of the group, so a loaded key serves four dots and
// the traffic matches the floor. The price is 4x the shuffle reductions per
// position, which the GPU has spare.
//
// Lane owns two dims of every q head in the group, so a score is a 2-FMA dot
// plus a 5-step butterfly -- the same decomposition as the CPU grouped kernel.
__global__ void attn_decode_tile(const float * __restrict__ q,   // [kQHeads][kHeadDim]
                                 const float * __restrict__ k,   // [kKvHeads][nd][kHeadDim]
                                 const float * __restrict__ v,   // [kKvHeads][nd][kHeadDim]
                                 Partial * __restrict__ part,    // [nblk*kWarps][kKvHeads]
                                 int nd, int nblk, float scale) {
    const int kv = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;

    const float * kk = k + (size_t)kv * nd * kHeadDim;
    const float * vv = v + (size_t)kv * nd * kHeadDim;

    float q0[kGQA], q1[kGQA];
    #pragma unroll
    for (int g = 0; g < kGQA; ++g) {
        q0[g] = q[(kv * kGQA + g) * kHeadDim + lane * 2 + 0] * scale;
        q1[g] = q[(kv * kGQA + g) * kHeadDim + lane * 2 + 1] * scale;
    }

    const int per = (nd + nblk - 1) / nblk;
    const int lo = blockIdx.x * per;
    const int hi = min(nd, lo + per);

    float m[kGQA], l[kGQA], a0[kGQA], a1[kGQA];
    #pragma unroll
    for (int g = 0; g < kGQA; ++g) {
        m[g] = -INFINITY; l[g] = 0.0f; a0[g] = 0.0f; a1[g] = 0.0f;
    }

    for (int p = lo + warp; p < hi; p += nw) {
        const float2 k2 = *(const float2 *)(kk + (size_t)p * kHeadDim + lane * 2);
        const float2 v2 = *(const float2 *)(vv + (size_t)p * kHeadDim + lane * 2);
        #pragma unroll
        for (int g = 0; g < kGQA; ++g) {
            float dot = q0[g] * k2.x + q1[g] * k2.y;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffull, dot, off);
            const float nm = fmaxf(m[g], dot);
            const float e = __expf(dot - nm);
            const float corr = __expf(m[g] - nm);
            l[g] = l[g] * corr + e;
            a0[g] = a0[g] * corr + e * v2.x;
            a1[g] = a1[g] * corr + e * v2.y;
            m[g] = nm;
        }
    }

    Partial & out = part[(size_t)(blockIdx.x * nw + warp) * kKvHeads + kv];
    #pragma unroll
    for (int g = 0; g < kGQA; ++g) {
        const float iv = 1.0f / l[g];
        if (lane == 0) { out.m[g] = m[g]; out.l[g] = l[g]; }
        out.acc[g][lane * 2 + 0] = a0[g] * iv;
        out.acc[g][lane * 2 + 1] = a1[g] * iv;
    }
}

// Merge npart partials per (kv head, q head). npart * kGQA * kHeadDim floats,
// so this is a rounding error next to the K/V traffic it summarises.
__global__ void attn_merge(const Partial * __restrict__ part, int npart,
                           float * __restrict__ out) {   // [kQHeads][kHeadDim]
    const int qh = blockIdx.x, lane = threadIdx.x;
    const int kv = qh / kGQA, w = qh % kGQA;
    float m = -INFINITY, l = 0.0f, a = 0.0f;
    for (int i = 0; i < npart; ++i) {
        const Partial & p = part[(size_t)i * kKvHeads + kv];
        const float pm = p.m[w], pl = p.l[w];
        const float nm = fmaxf(m, pm);
        const float c0 = __expf(m - nm), c1 = __expf(pm - nm);
        l = l * c0 + pl * c1;
        a = a * c0 + p.acc[w][lane] * pl * c1;
        m = nm;
    }
    out[qh * kHeadDim + lane] = a / l;
}
} // namespace

int main(int argc, char ** argv) {
    const int nd = argc > 1 ? std::atoi(argv[1]) : 16384;
    const size_t hd = kHeadDim;
    const size_t kv_bytes = (size_t)kLayers * kKvHeads * nd * hd * 4;
    std::printf("context %d: K+V = %.0f MB across %d layers x %d kv heads\n",
                nd, kv_bytes / 1e6, kLayers, kKvHeads);

    // All TEN layers must be allocated and walked. Reusing one layer made the
    // working set 67 MB, which fits the 80 MB Infinity Cache, and the probe then
    // reported 815 GB/s -- above the 764 GB/s measured ceiling, i.e. impossible
    // for a real pass that reads distinct bytes.
    const size_t one_layer = (size_t)kKvHeads * nd * hd * 4;
    const size_t all_kv = (size_t)kLayers * one_layer;
    float * dk; float * dv; float * dq; float * dout;
    hipMalloc(&dk, all_kv);
    hipMalloc(&dv, all_kv);
    hipMalloc(&dq, (size_t)kQHeads * hd * 4);
    hipMalloc(&dout, (size_t)kQHeads * hd * 4);
    hipMemset(dk, 0, all_kv);
    hipMemset(dv, 0, all_kv);
    hipMemset(dq, 0, (size_t)kQHeads * hd * 4);

    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    auto time_floor = [&](int blocks_per_kv) {
        double best = 1e9;
        for (int r = 0; r < 10; ++r) {
            hipEventRecord(e0);
            for (int layer = 0; layer < kLayers; ++layer)
                for (int kv = 0; kv < kKvHeads; ++kv)
                    attn_traffic_kernel<<<blocks_per_kv, 256>>>(
                        (const float4 *)(dk + ((size_t)layer * kKvHeads + kv) * nd * hd),
                        (const float4 *)(dv + ((size_t)layer * kKvHeads + kv) * nd * hd),
                        dout, nd);
            hipEventRecord(e1);
            hipEventSynchronize(e1);
            float ms = 0;
            hipEventElapsedTime(&ms, e0, e1);
            best = std::min(best, (double)ms);
        }
        return best;
    };

    double floor_ms = 0;
    for (int b : {1, 8, 32, 64, 128, 256}) {
        const double ms = time_floor(b);
        if (b == 32) floor_ms = ms;
        std::printf("floor, %3d blocks/kv head: %6.2f ms -> %6.1f GB/s%s\n",
                    b, ms, 2.0 * all_kv / (ms * 1e6), b == 32 ? "   <- floor" : "");
    }

    // Correctness against the CPU grouped decode, head-major cache.
    {
        std::vector<float> hq(kQHeads * hd), hk(one_layer / 4), hv(one_layer / 4);
        unsigned s = 7;
        auto rnd = [&s] { s = s * 1664525u + 1013904223u; return (float)((s >> 8) * (1.0 / 16777216.0)) - 0.5f; };
        for (auto & x : hq) x = rnd();
        for (auto & x : hk) x = rnd();
        for (auto & x : hv) x = rnd();

        std::vector<float> cpu(kQHeads * hd), scratch(8 * (size_t)nd * 8);
        kata::AttnArgs a;
        a.q = hq.data(); a.k_cache = hk.data(); a.v_cache = hv.data();
        a.n_kv = nd; a.n_heads = kQHeads; a.n_kv_heads = kKvHeads; a.head_dim = hd;
        a.scale = 1.0f / std::sqrt((float)hd); a.out = cpu.data();
        a.stride_pos = hd; a.stride_head = (size_t)nd * hd;
        kata::attention_decode_grouped(a, scratch.data());

        const int nblk = 32, npart = nblk * kWarps;
        Partial * dpart;
        hipMalloc(&dpart, (size_t)npart * kKvHeads * sizeof(Partial));
        hipMemcpy(dk, hk.data(), one_layer, hipMemcpyHostToDevice);
        hipMemcpy(dv, hv.data(), one_layer, hipMemcpyHostToDevice);
        hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice);
        attn_decode_tile<<<dim3(nblk, kKvHeads), kThreads>>>(dq, dk, dv, dpart, nd, nblk, a.scale);
        attn_merge<<<kQHeads, kHeadDim>>>(dpart, npart, dout);
        const auto err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::fprintf(stderr, "kernel: %s\n", hipGetErrorString(err));
            return 2;
        }
        std::vector<float> gpu(cpu.size());
        hipMemcpy(gpu.data(), dout, gpu.size() * 4, hipMemcpyDeviceToHost);
        double ma = 0, mr = 0;
        for (size_t i = 0; i < cpu.size(); ++i) {
            ma = std::max(ma, std::fabs((double)gpu[i] - (double)cpu[i]));
            mr = std::max(mr, std::fabs((double)cpu[i]));
        }
        // __expf versus the CPU polynomial: a few ULP of the magnitude, not exact
        // by design. The bit-identical variant needed one lane per position and
        // was 4x slower, so this is the trade taken.
        const double tol = 1e-3 * mr;
        std::printf("accuracy vs cpu grouped: max|abs| %.3e  largest %.4f -> %s\n",
                    ma, mr, ma <= tol ? "OK" : "FAIL");
        hipFree(dpart);
        if (ma > tol) return 1;
    }

    // The real kernel at several block counts: nblk=32 is the measured optimum
    // for this shape; past it, partial merges and re-reads cost more than the
    // extra occupancy buys (nblk=512 is 3.5x worse).
    for (int nblk : {8, 16, 24, 32, 64, 128}) {
        const int npart = nblk * kWarps;
        Partial * dpart;
        hipMalloc(&dpart, (size_t)npart * kKvHeads * sizeof(Partial));
        double best = 1e9;
        for (int r = 0; r < 10; ++r) {
            hipEventRecord(e0);
            for (int layer = 0; layer < kLayers; ++layer) {
                const size_t off = (size_t)layer * kKvHeads * nd * hd;
                attn_decode_tile<<<dim3(nblk, kKvHeads), kThreads>>>(dq, dk + off, dv + off, dpart, nd, nblk,
                                                                    1.0f / std::sqrt((float)hd));
                attn_merge<<<kQHeads, kHeadDim>>>(dpart, npart, dout);
            }
            hipEventRecord(e1);
            hipEventSynchronize(e1);
            float ms = 0;
            hipEventElapsedTime(&ms, e0, e1);
            best = std::min(best, (double)ms);
        }
        std::printf("real kernel, %3d blocks/kv head: %6.2f ms -> %6.1f GB/s ; vs cpu 9.06 ms -> %.1fx\n",
                    nblk, best, 2.0 * all_kv / (best * 1e6), 9.06 / best);
        hipFree(dpart);
    }

    std::printf("\nfloor %.2f ms; a token at this depth = weights 3.72 + attention "
                "-> see EXP-0075 for the total\n", floor_ms);
    return 0;
}
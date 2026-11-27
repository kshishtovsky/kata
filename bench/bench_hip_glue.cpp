// EXP-0076: the glue a decode token needs besides weights and attention.
//
// Build: hipcc -O3 bench/bench_hip_glue.cpp -o /tmp/glue
//
// At 2048 width every one of these is a rounding error in BYTES (a token's
// activations total ~10 MB) but each is a separate kernel, and 2.77 us of launch
// overhead per kernel is not a rounding error at all. So the question is not
// "how fast is silu" but "how many launches does a token cost, and does batching
// them into persistent kernels change the answer".
//
// Sequence per token, derived from src/model.cpp:
//   40 x (rms_norm, moe_router(top4 of 64))
//   38 x (moe_gather q8, moe_scatter)       + 2 x dense down/up path
//   10 x (attn q/k/v rope kvwrite + o_proj)
//   30 x (shortconv in_proj, 3-tap conv, out_proj)
//    1 x (final norm, lm_head over 128000 vocab, sampler)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int kN = 2048;       // model width
constexpr int kLayers = 40, kAttn = 10, kConv = 30, kMoE = 38;
constexpr int kExperts = 64, kTopK = 4, kVocab = 128000;

__global__ void rms_norm_k(const float * __restrict__ x, float * __restrict__ y, int n) {
    __shared__ float s[256];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffull, acc, off);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int w = 0; w < 8; ++w) t += s[w];
        s[0] = rsqrtf(t / n + 1e-5f);
    }
    __syncthreads();
    const float sc = s[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * sc;
}

// Router: sigmoid over 64 logits, top-4. Small on purpose -- it is all launch.
__global__ void router_k(const float * __restrict__ logits, const float * __restrict__ bias,
                         int * __restrict__ idx, float * __restrict__ w, int n_exp, int topk) {
    if (threadIdx.x == 0) {
        float best[8], bi[8];
        for (int t = 0; t < topk; ++t) { best[t] = -1e30f; bi[t] = 0; }
        float sum = 0.0f;
        for (int e = 0; e < n_exp; ++e) {
            const float p = 1.0f / (1.0f + __expf(-logits[e]));
            for (int t = 0; t < topk; ++t) {
                if (p + bias[e] > best[t]) {
                    for (int k = topk - 1; k > t; --k) { best[k] = best[k - 1]; bi[k] = bi[k - 1]; }
                    best[t] = p + bias[e]; bi[t] = (float)e;
                    break;
                }
            }
        }
        for (int t = 0; t < topk; ++t) { idx[t] = (int)bi[t]; sum += w[t]; }
        for (int t = 0; t < topk; ++t) w[t] = 1.0f / (1.0f + __expf(-logits[idx[t]]));
        float s = 0.0f;
        for (int t = 0; t < topk; ++t) s += w[t];
        for (int t = 0; t < topk; ++t) w[t] /= s;
    }
}

// The 3-tap shortconv core: y = c * (k0*bx + k1*bx[-1] + k2*bx[-2]).
__global__ void shortconv_k(const float * __restrict__ bcx, float * __restrict__ y,
                            int n, int taps) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const float b = bcx[e], c = bcx[n + e];
    float acc = 0.0f;
    for (int t = 0; t < taps; ++t) acc += bcx[2 * n + e] * (1.0f / (t + 1));
    y[e] = c * acc + b * 0.0f;
}

// MoE gather: copy the activation into kTopK expert slots (row 0..kTopK of `n`).
__global__ void moe_gather_k(const float * __restrict__ x, float * __restrict__ xk, int n, int topk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int t = 0; t < topk; ++t) xk[t * n + i] = x[i];
}

// lm_head over 128000 rows: only the matvec is big, but the argmax is a scan.
__global__ void argmax_k(const float * __restrict__ logits, int n, int * __restrict__ best) {
    __shared__ float sv[256];
    __shared__ int si[256];
    float bv = -1e30f; int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (logits[i] > bv) { bv = logits[i]; bi = i; }
    }
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s && sv[threadIdx.x + s] > sv[threadIdx.x]) {
            sv[threadIdx.x] = sv[threadIdx.x + s];
            si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *best = si[0];
}

// One launch for all of a layer's glue. The pieces are 2048-wide vectors, so
// each is a few microseconds of work; the launch is what costs. This is what a
// real implementation would do, and it sets the upper bound on fusion.
__global__ void layer_glue_k(const float * __restrict__ x, float * __restrict__ y,
                             float * __restrict__ xk, const float * __restrict__ bias,
                             int * __restrict__ idx, float * __restrict__ w,
                             int n, int n_exp, int topk, bool conv, int taps) {
    __shared__ float s[256];
    __shared__ float red[8];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffull, acc, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int wq = 0; wq < 8; ++wq) t += red[wq];
        red[0] = rsqrtf(t / n + 1e-5f);
    }
    __syncthreads();
    const float sc = red[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * sc;

    if (threadIdx.x == 0) {
        float best[8], bi[8];
        for (int t = 0; t < topk; ++t) { best[t] = -1e30f; bi[t] = 0; }
        for (int e = 0; e < n_exp; ++e) {
            const float p = 1.0f / (1.0f + __expf(-y[e])) + bias[e];
            for (int t = 0; t < topk; ++t)
                if (p > best[t]) {
                    for (int k = topk - 1; k > t; --k) { best[k] = best[k - 1]; bi[k] = bi[k - 1]; }
                    best[t] = p; bi[t] = (float)e;
                    break;
                }
        }
        float sum = 0.0f;
        for (int t = 0; t < topk; ++t) { idx[t] = (int)bi[t]; w[t] = best[t] > -1e29f ? 1.0f : 0.0f; sum += 1.0f; }
        for (int t = 0; t < topk; ++t) w[t] = 1.0f / sum;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        for (int t = 0; t < topk; ++t) xk[t * n + i] = y[i];

    if (conv) {
        const int e = threadIdx.x;
        for (int i = e; i < n; i += blockDim.x) {
            float a = 0.0f;
            for (int tp = 0; tp < taps; ++tp) a += y[i] * (1.0f / (tp + 1));
            y[i] = y[n + i] * a;
        }
    }
}
} // namespace

int main() {
    float * dx; float * dy; float * dg; int * di; float * dw; int * dbest;
    hipMalloc(&dx, kN * 4); hipMalloc(&dy, 4 * kN * 4); hipMalloc(&dg, 4 * kN * 4);
    hipMalloc(&di, 8 * 4); hipMalloc(&dw, 8 * 4); hipMalloc(&dbest, 4);
    hipMemset(dx, 0, kN * 4);

    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    const int nthr = 256, nblk = (kN + nthr - 1) / nthr;

    int counts[6] = {0};
    auto run = [&](int kind) {
        rms_norm_k<<<1, nthr>>>(dx, dy, kN);                       counts[kind] += 1;
        router_k<<<1, 32>>>(dx, dy, di, dw, kExperts, kTopK);      counts[kind] += 1;
        if (kind == 2) { moe_gather_k<<<nblk, nthr>>>(dx, dg, kN, kTopK); counts[kind] += 1; }
        if (kind == 3) { shortconv_k<<<nblk, nthr>>>(dx, dy, kN, 3); counts[kind] += 1; }
    };

    // Count one token's launches without timing them, so the count cannot be
    // inflated by the repetition loop (the first version divided by 10 reps of
    // accumulated counts and reported 2840 launches for a 148-launch token).
    { for (int l = 0; l < kLayers; ++l) { run(0); if (l < kMoE) run(2); if (l < kConv) run(3); } }
    const int total = counts[0] + counts[2] + counts[3];
    for (int i = 0; i < 6; ++i) counts[i] = 0;

    double best = 1e9;
    for (int r = 0; r < 10; ++r) {
        hipEventRecord(e0);
        for (int l = 0; l < kLayers; ++l) {
            run(0);                                  // norm + router
            if (l < kMoE) run(2);                    // moe gather
            if (l < kConv) run(3);                   // shortconv core
        }
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1);
        best = std::min(best, (double)ms);
    }
    std::printf("glue per token: %d launches, %.3f ms -> %.2f us per launch\n",
                total, best, 1e3 * best / total);
    std::printf("  (%.0f%% is the launch itself at 2.77 us)\n", 100.0 * 2.77 * total / (best * 1e3));

    // Same work, one launch per layer: a persistent kernel per stage. This is
    // the upper bound on what fusing the glue can win.
    best = 1e9;
    for (int r = 0; r < 10; ++r) {
        hipEventRecord(e0);
        for (int l = 0; l < kLayers; ++l) rms_norm_k<<<1, nthr>>>(dx, dy, kN);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1);
        best = std::min(best, (double)ms);
    }
    std::printf("40 norm launches alone: %.3f ms (%.2f us each)\n", best, best * 1e3 / kLayers);

    // Fused: one launch per layer for ALL of that layer's glue.
    best = 1e9;
    for (int r = 0; r < 10; ++r) {
        hipEventRecord(e0);
        for (int l = 0; l < kLayers; ++l)
            layer_glue_k<<<1, nthr>>>(dx, dy, dg, dw, di, dw, kN, kExperts, kTopK, l < kConv, 3);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1);
        best = std::min(best, (double)ms);
    }
    std::printf("fused, 40 launches (one per layer): %.3f ms -> %.2f us each\n", best, best * 1e3 / kLayers);
    std::printf("  token total if fused: weights 3.72 + attn 1.34 + glue %.2f = %.2f ms -> %.0f tok/s\n",
                best, 5.06 + best, 1e3 / (5.06 + best));
    std::printf("  token total if not:   weights 3.72 + attn 1.34 + glue 2.76 = %.2f ms -> %.0f tok/s\n",
                5.06 + 2.76, 1e3 / (5.06 + 2.76));
    return 0;
}

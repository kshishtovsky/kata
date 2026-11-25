#include "kata/attn.h"

#include "kata/cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#ifndef SCORE_N
#define SCORE_N 4
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace kata {
namespace {

inline size_t head_of(size_t h, size_t n_heads, size_t n_kv_heads) {
    // GQA: every group of (n_heads / n_kv_heads) query heads shares one kv head.
    const size_t group = n_heads / n_kv_heads;
    return h / group;
}

} // namespace

void attention_decode_ref(const AttnArgs & a, float * scratch) {
    const size_t hd = a.head_dim;
    for (size_t h = 0; h < a.n_heads; ++h) {
        const float * q = a.q + h * hd;
        const size_t base = head_of(h, a.n_heads, a.n_kv_heads) * hd;

        // Pass 1: scores, with a running max so a long context cannot overflow.
        float mx = -INFINITY;
        for (size_t t = 0; t < a.n_kv; ++t) {
            const float * k = a.k_cache + t * a.n_kv_heads * hd + base;
            float s = 0.0f;
            for (size_t i = 0; i < hd; ++i) s += q[i] * k[i];
            s *= a.scale;
            scratch[t] = s;
            mx = std::max(mx, s);
        }
        // Pass 2: exp and sum.
        float sum = 0.0f;
        for (size_t t = 0; t < a.n_kv; ++t) {
            scratch[t] = std::exp(scratch[t] - mx);
            sum += scratch[t];
        }
        // Pass 3: weighted sum of v.
        float * o = a.out + h * hd;
        for (size_t i = 0; i < hd; ++i) o[i] = 0.0f;
        const float inv = 1.0f / sum;
        for (size_t t = 0; t < a.n_kv; ++t) {
            const float p = scratch[t] * inv;
            const float * v = a.v_cache + t * a.n_kv_heads * hd + base;
            for (size_t i = 0; i < hd; ++i) o[i] += p * v[i];
        }
    }
}

#if defined(__x86_64__) || defined(__i386__)

namespace {

__attribute__((target("avx2,fma")))
float dot_avx2(const float * x, const float * y, size_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8), a1);
    }
    for (; i + 8 <= n; i += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), a0);
    a0 = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(a0), hi = _mm256_extractf128_ps(a0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < n; ++i) s += x[i] * y[i];
    return s;
}

} // namespace

namespace {

// exp(x) = exp2(x * log2(e)). There is no exp2 instruction on AVX2, so this is a
// polynomial with the integer exponent split out: 2^(k + f) = 2^k * 2^f, where
// 2^f comes from a degree-5 minimax fit on f in [-0.5, 0.5] and 2^k is an
// exponent-field add. Range is clamped to keep the shift in range.
// 4 independent accumulator chains: one chain would measure the add latency.
__attribute__((target("avx2,fma")))
__m256 exp256_ps(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    x = _mm256_mul_ps(x, log2e);
    // round to nearest int (ties to even)
    __m256 k = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 f = _mm256_sub_ps(x, k);
    // 2^f on f in [-0.5, 0.5]
    __m256 poly = _mm256_set1_ps(1.3333558146e-03f);
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(9.6181291076e-03f));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(5.5504108665e-02f));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(2.4022650696e-01f));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(6.9314718056e-01f));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(1.0f));
    // 2^k via exponent field; k is in [-126, 127] after the clamp above.
    const __m256i ki = _mm256_cvtps_epi32(k);
    const __m256 pow2k = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23));
    return _mm256_mul_ps(poly, pow2k);
}

__attribute__((target("avx2,fma")))
float softmax_exp_inplace(float * p, size_t n, float mx) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 upper = _mm256_set1_ps(88.0f);
    const __m256 lower = _mm256_set1_ps(-87.0f);
    const __m256 damax = _mm256_set1_ps(mx);
    __m256 s0 = zero, s1 = zero, s2 = zero, s3 = zero;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 v0 = _mm256_sub_ps(_mm256_loadu_ps(p + i), damax);
        __m256 v1 = _mm256_sub_ps(_mm256_loadu_ps(p + i + 8), damax);
        __m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(p + i + 16), damax);
        __m256 v3 = _mm256_sub_ps(_mm256_loadu_ps(p + i + 24), damax);
        v0 = _mm256_max_ps(_mm256_min_ps(v0, upper), lower);
        v1 = _mm256_max_ps(_mm256_min_ps(v1, upper), lower);
        v2 = _mm256_max_ps(_mm256_min_ps(v2, upper), lower);
        v3 = _mm256_max_ps(_mm256_min_ps(v3, upper), lower);
        v0 = exp256_ps(v0);
        v1 = exp256_ps(v1);
        v2 = exp256_ps(v2);
        v3 = exp256_ps(v3);
        _mm256_storeu_ps(p + i, v0);
        _mm256_storeu_ps(p + i + 8, v1);
        _mm256_storeu_ps(p + i + 16, v2);
        _mm256_storeu_ps(p + i + 24, v3);
        s0 = _mm256_add_ps(s0, v0);
        s1 = _mm256_add_ps(s1, v1);
        s2 = _mm256_add_ps(s2, v2);
        s3 = _mm256_add_ps(s3, v3);
    }
    // Tail accumulates into a scalar: _mm256_set1_ps would add the value to all
    // eight lanes and inflate the sum eightfold (caught at n_kv = 1).
    float tail = 0.0f;
    for (; i < n; ++i) {
        p[i] = std::exp(p[i] - mx);
        tail += p[i];
    }
    (void)one;
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 lo = _mm256_castps256_ps128(s0), hi = _mm256_extractf128_ps(s0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo) + tail;
}

// NV ymm registers of accumulator, NV*8 floats of head_dim handled in registers.
template <int NV>
__attribute__((target("avx2,fma")))
void v_accum(const AttnArgs & a, float * o, const float * p, size_t hd, size_t base, float inv) {
    __m256 acc[NV];
    for (int j = 0; j < NV; ++j) acc[j] = _mm256_setzero_ps();
    for (size_t t = 0; t < a.n_kv; ++t) {
        const __m256 pv = _mm256_set1_ps(p[t]);
        const float * v = a.v_cache + t * a.n_kv_heads * hd + base;
        for (int j = 0; j < NV; ++j) acc[j] = _mm256_fmadd_ps(pv, _mm256_loadu_ps(v + 8 * j), acc[j]);
    }
    const __m256 iv = _mm256_set1_ps(inv);
    for (int j = 0; j < NV; ++j) _mm256_storeu_ps(o + 8 * j, _mm256_mul_ps(acc[j], iv));
    for (size_t i = size_t(NV) * 8; i < hd; ++i) {
        float s = 0.0f;
        for (size_t t = 0; t < a.n_kv; ++t) s += p[t] * a.v_cache[t * a.n_kv_heads * hd + base + i];
        o[i] = s * inv;
    }
}

// Fallback for head dims whose ymm count would spill the registers anyway.
__attribute__((target("avx2,fma")))
void v_accum_generic(const AttnArgs & a, float * o, const float * p, size_t hd, size_t base, float inv) {
    const size_t nv = (hd / 8) * 8;
    for (size_t i = 0; i < nv; i += 8) _mm256_storeu_ps(o + i, _mm256_setzero_ps());
    for (size_t i = nv; i < hd; ++i) o[i] = 0.0f;
    for (size_t t = 0; t < a.n_kv; ++t) {
        const __m256 pv = _mm256_set1_ps(p[t]);
        const float * v = a.v_cache + t * a.n_kv_heads * hd + base;
        for (size_t i = 0; i < nv; i += 8)
            _mm256_storeu_ps(o + i, _mm256_fmadd_ps(pv, _mm256_loadu_ps(v + i), _mm256_loadu_ps(o + i)));
        for (size_t i = nv; i < hd; ++i) o[i] += p[t] * v[i];
    }
    const __m256 iv = _mm256_set1_ps(inv);
    for (size_t i = 0; i < nv; i += 8)
        _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_loadu_ps(o + i), iv));
    for (size_t i = nv; i < hd; ++i) o[i] *= inv;
}

} // namespace

namespace {

// N positions at once (N even). Two wins: q is loaded once per N k rows, and
// the per-position horizontal reduce is shared by pairs. meas: N=4 was 1.10x
// over per-position scoring at 4-16k, N=8 another 1.33x.
//
// After hadd(hadd(a,b), hadd(a,b)) the lanes hold
//   [a0+a1+a2+a3, b0+b1+b2+b3, ... , a4+a5+a6+a7, b4+b5+b6+b7]
// so sum(a) = lane0 + lane4 and sum(b) = lane1 + lane5.
template <int N>
__attribute__((target("avx2,fma")))
void scores_n(const AttnArgs & a, const float * q, size_t base, size_t t0, size_t hd, float scale,
              float * out) {
    static_assert(N % 2 == 0, "N must be even");
    const size_t stride = a.n_kv_heads * hd;
    const float * k0 = a.k_cache + t0 * stride + base;
    __m256 r[N];
    for (int j = 0; j < N; ++j) r[j] = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= hd; i += 8) {
        const __m256 qv = _mm256_loadu_ps(q + i);
        for (int j = 0; j < N; ++j)
            r[j] = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + j * stride + i), r[j]);
    }
    alignas(32) float lane[8];
    for (int p = 0; p < N; p += 2) {
        const __m256 h = _mm256_hadd_ps(r[p], r[p + 1]);
        _mm256_store_ps(lane, _mm256_hadd_ps(h, h));
        out[p] = (lane[0] + lane[4]) * scale;
        out[p + 1] = (lane[1] + lane[5]) * scale;
    }
    // Head dims not divisible by 8: finish the tail scalar per position.
    if (i < hd) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t z = i; z < hd; ++z) acc += q[z] * k0[j * stride + z];
            out[j] += acc * scale;
        }
    }
}

// One query head: score every cached position, return the max for softmax.
__attribute__((target("avx2,fma")))
float attn_head_scores(const AttnArgs & a, float * scratch) {
    const float * q = a.q;
    const size_t hd = a.head_dim;
    const size_t base = a.base;
    float mx = -INFINITY;
    size_t t = 0;
    for (; t + SCORE_N <= a.n_kv; t += SCORE_N) {
        scores_n<SCORE_N>(a, q, base, t, hd, a.scale, scratch + t);
        for (int j = 0; j < SCORE_N; ++j) mx = std::max(mx, scratch[t + j]);
    }
    for (; t < a.n_kv; ++t) {
        const float * k = a.k_cache + t * a.n_kv_heads * hd + base;
        const float s = dot_avx2(q, k, hd) * a.scale;
        scratch[t] = s;
        mx = std::max(mx, s);
    }
    return mx;
}

// One query head: weighted sum of v, normalised by 1/sum.
__attribute__((target("avx2,fma")))
void attn_head_v(const AttnArgs & a, const float * p, float inv) {
    float * o = a.out;
    const size_t hd = a.head_dim;
    const size_t base = a.base;
    const size_t nv = hd / 8;
    switch (nv) {
    case 1:  v_accum<1>(a, o, p, hd, base, inv); break;
    case 2:  v_accum<2>(a, o, p, hd, base, inv); break;
    case 4:  v_accum<4>(a, o, p, hd, base, inv); break;
    case 8:  v_accum<8>(a, o, p, hd, base, inv); break;
    case 16: v_accum<16>(a, o, p, hd, base, inv); break;
    default: v_accum_generic(a, o, p, hd, base, inv); break;
    }
}

// 8x8 transpose of ymm lanes. After it, a[i] holds lane i of all eight
// accumulators, so one vector add across a[0..7] yields the eight horizontal
// sums. The per-pair hadd chain this replaces was 32% of the kernel.
__attribute__((target("avx2,fma")))
void transpose8(__m256 * a) {
    const __m256 t0 = _mm256_unpacklo_ps(a[0], a[1]);
    const __m256 t1 = _mm256_unpackhi_ps(a[0], a[1]);
    const __m256 t2 = _mm256_unpacklo_ps(a[2], a[3]);
    const __m256 t3 = _mm256_unpackhi_ps(a[2], a[3]);
    const __m256 t4 = _mm256_unpacklo_ps(a[4], a[5]);
    const __m256 t5 = _mm256_unpackhi_ps(a[4], a[5]);
    const __m256 t6 = _mm256_unpacklo_ps(a[6], a[7]);
    const __m256 t7 = _mm256_unpackhi_ps(a[6], a[7]);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    a[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    a[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    a[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    a[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    a[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    a[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    a[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    a[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// 8 (position, head) pairs: 2 positions x 4 query heads of one kv group. Each
// k vector is loaded once and used by all four heads, which is the whole point
// of grouping; the transpose at the end replaces eight hadd chains with one.
// acc index is j*4 + h, matching the out[j][h] layout the caller wants.
__attribute__((target("avx2,fma")))
void scores_8x(const AttnArgs & a, const float * q, size_t base, size_t t0, size_t hd,
               float scale, float * out) {
    const size_t stride = a.n_kv_heads * hd;
    const float * k0 = a.k_cache + t0 * stride + base;
    const float * k1 = k0 + stride;
    __m256 r[8];
    for (int i = 0; i < 8; ++i) r[i] = _mm256_setzero_ps();

    size_t z = 0;
    for (; z + 8 <= hd; z += 8) {
        const __m256 kk0 = _mm256_loadu_ps(k0 + z);
        const __m256 kk1 = _mm256_loadu_ps(k1 + z);
        for (int h = 0; h < 4; ++h) {
            const __m256 qh = _mm256_loadu_ps(q + h * hd + z);
            r[h] = _mm256_fmadd_ps(qh, kk0, r[h]);
            r[4 + h] = _mm256_fmadd_ps(qh, kk1, r[4 + h]);
        }
    }
    transpose8(r);
    __m256 w = r[0];
    for (int i = 1; i < 8; ++i) w = _mm256_add_ps(w, r[i]);
    alignas(32) float lane[8];
    _mm256_store_ps(lane, w);
    for (int i = 0; i < 8; ++i) out[i] = lane[i];
    // Head dims with a remainder: finish scalar, per (position, head).
    if (z < hd) {
        for (int h = 0; h < 4; ++h) {
            float a0 = 0.0f, a1 = 0.0f;
            for (size_t i = z; i < hd; ++i) {
                a0 += q[h * hd + i] * k0[i];
                a1 += q[h * hd + i] * k1[i];
            }
            out[h] += a0;
            out[4 + h] += a1;
        }
    }
    for (int i = 0; i < 8; ++i) out[i] *= scale;
}

// G query heads share one kv head. Walking the kv head once and scoring all G
// against the same k block removes the G-fold re-read of K, which the phase
// split showed was 256 MiB per token at 32k for 64 MiB of actual data.
// Accumulators: NT positions x G heads; NT=2, G=4 -> 8 ymm, fits with room.
template <int G, int NT>
__attribute__((target("avx2,fma")))
void scores_group(const AttnArgs & a, const float * q, size_t base, size_t t0, size_t hd,
                  const float * scale, float * out, float * mx) {
    const size_t stride = a.n_kv_heads * hd;
    const float * k0 = a.k_cache + t0 * stride + base;
    __m256 acc[NT][G];
    for (int j = 0; j < NT; ++j)
        for (int h = 0; h < G; ++h) acc[j][h] = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= hd; i += 8) {
        __m256 kv[NT];
        for (int j = 0; j < NT; ++j) kv[j] = _mm256_loadu_ps(k0 + j * stride + i);
        for (int h = 0; h < G; ++h) {
            const __m256 qv = _mm256_loadu_ps(q + h * hd + i);
            for (int j = 0; j < NT; ++j) acc[j][h] = _mm256_fmadd_ps(qv, kv[j], acc[j][h]);
        }
    }
    for (int j = 0; j < NT; ++j) {
        for (int h = 0; h < G; ++h) {
            __m128 t = _mm_add_ps(_mm256_castps256_ps128(acc[j][h]), _mm256_extractf128_ps(acc[j][h], 1));
            t = _mm_hadd_ps(t, t);
            t = _mm_hadd_ps(t, t);
            float sc = _mm_cvtss_f32(t);
            for (size_t z = i; z < hd; ++z) sc += q[h * hd + z] * k0[j * stride + z];
            sc *= scale[h];
            out[h * a.n_kv + t0 + j] = sc;
            mx[h] = std::max(mx[h], sc);
        }
    }
}

// G heads sharing one kv head: weighted sum of v. NZ is a compile-time zmm
// count on purpose: with a runtime bound the accumulator array lives on the
// stack and GCC emits one kmovw per spill reload, which profiling showed as 35%
// of the kernel.
template <int G, int NZ>
__attribute__((target("avx512f,avx512vl,avx512bw,fma")))
void v_group(const AttnArgs & a, const float * p, size_t hd, size_t base, const float * inv,
             float * out) {
    const size_t stride = a.n_kv_heads * hd;
    __m512 acc[G][NZ];
    for (int h = 0; h < G; ++h)
        for (int j = 0; j < NZ; ++j) acc[h][j] = _mm512_setzero_ps();

    for (size_t t = 0; t < a.n_kv; ++t) {
        const float * v = a.v_cache + t * stride + base;
        __m512 pv[G];
        for (int h = 0; h < G; ++h) pv[h] = _mm512_set1_ps(p[h * a.n_kv + t]);
        for (int j = 0; j < NZ; ++j) {
            const __m512 vv = _mm512_loadu_ps(v + j * 16);
            for (int h = 0; h < G; ++h) acc[h][j] = _mm512_fmadd_ps(pv[h], vv, acc[h][j]);
        }
    }
    const size_t done = size_t(NZ) * 16;
    for (int h = 0; h < G; ++h) {
        const __m512 iv = _mm512_set1_ps(inv[h]);
        for (int j = 0; j < NZ; ++j)
            _mm512_storeu_ps(out + h * hd + j * 16, _mm512_mul_ps(acc[h][j], iv));
        for (size_t i = done; i < hd; ++i) {
            float s = 0.0f;
            for (size_t t = 0; t < a.n_kv; ++t) s += p[h * a.n_kv + t] * a.v_cache[t * stride + base + i];
            out[h * hd + i] = s * inv[h];
        }
    }
}


// Score pass over the transposed k cache: the vector lanes are positions, the
// accumulators are the four query heads of one kv group. No reduction step and
// no transpose at the end -- a lane already holds the finished dot product.
// Downsides recorded in EXP-0018: no q reuse across head_dim (a broadcast per
// dim per head) and the k write in decode is head_dim strided stores.
template <int NP>
__attribute__((target("avx512f,avx512vl,fma")))
void scores_kt(const float * q, const float * kt_row0, size_t hd, size_t n_kv, size_t stride_d,
               float scale, float * out) {
    __m512 acc[4];
    for (int h = 0; h < 4; ++h) acc[h] = _mm512_setzero_ps();
    const float * col = kt_row0;
    for (size_t d = 0; d < hd; ++d, col += stride_d) {
        const __m512 kv = _mm512_loadu_ps(col);
        for (int h = 0; h < 4; ++h)
            acc[h] = _mm512_fmadd_ps(_mm512_set1_ps(q[h * hd + d]), kv, acc[h]);
    }
    for (int h = 0; h < 4; ++h) {
        alignas(64) float lane[16];
        _mm512_store_ps(lane, _mm512_mul_ps(acc[h], _mm512_set1_ps(scale)));
        for (int j = 0; j < NP; ++j) out[j * 4 + h] = lane[j];
    }
}

} // namespace

__attribute__((target("avx2,fma")))
void attention_decode_avx2(const AttnArgs & a, float * scratch) {
    const size_t hd = a.head_dim;
    for (size_t h = 0; h < a.n_heads; ++h) {
        AttnArgs one = a;
        one.n_heads = 1;
        one.q = a.q + h * hd;
        one.out = a.out + h * hd;
        one.base = head_of(h, a.n_heads, a.n_kv_heads) * hd;

        const float mx = attn_head_scores(one, scratch);
        const float sum = softmax_exp_inplace(scratch, a.n_kv, mx);
        attn_head_v(one, scratch, 1.0f / sum);
    }
}

// Grouped variant: one kv head serves G query heads, so both passes walk the
// kv cache once per group instead of once per head.
size_t attention_scratch_floats_grouped(const AttnArgs & a) {
    const size_t g = a.n_heads / a.n_kv_heads;
    return a.n_kv * (g ? g : 1) + 8;
}

__attribute__((target("avx2,fma")))
void attention_decode_avx2_grouped(const AttnArgs & a, float * scratch) {
    const size_t hd = a.head_dim;
    const size_t g = a.n_heads / a.n_kv_heads;

    // Only the G=4, head_dim%16==0 shape has a register-resident v pass.
    if (g != 4 || hd % 16 != 0 || hd > 64) {
        attention_decode_avx2(a, scratch);
        return;
    }

    alignas(64) float scale[4], mxv[4], inv[4];
    for (size_t h = 0; h < 4; ++h) {
        scale[h] = a.scale;
        mxv[h] = -INFINITY;
    }

    for (size_t gb = 0; gb < a.n_kv_heads; ++gb) {
        const size_t base = gb * hd;
        const float * q = a.q + gb * 4 * hd;

        float * p = scratch;  // 4 heads x n_kv probabilities, head-major
        size_t t = 0;
        alignas(32) float tile[8];
        // hd % 8 == 0 is guaranteed by the hd % 16 check above, so the tiled
        // path covers whole tiles and only the odd position is left over.
        for (; t + 2 <= a.n_kv; t += 2) {
            scores_8x(a, q, base, t, hd, a.scale, tile);
            for (int j = 0; j < 2; ++j) {
                for (int h = 0; h < 4; ++h) {
                    const float sc = tile[j * 4 + h];
                    p[h * a.n_kv + t + j] = sc;
                    mxv[h] = std::max(mxv[h], sc);
                }
            }
        }
        // Odd tail position: reuse the NT=2 kernel on a copy would read past the
        // cache, so score it per head.
        if (t < a.n_kv) {
            for (size_t h = 0; h < 4; ++h) {
                const float * k = a.k_cache + t * a.n_kv_heads * hd + base;
                const size_t hi = gb * 4 + h;
                const float sc = dot_avx2(a.q + hi * hd, k, hd) * a.scale;
                p[h * a.n_kv + t] = sc;
                mxv[h] = std::max(mxv[h], sc);
            }
        }

        for (size_t h = 0; h < 4; ++h) {
            float * ph = p + h * a.n_kv;
            const float sum = softmax_exp_inplace(ph, a.n_kv, mxv[h]);
            inv[h] = 1.0f / sum;
        }

        // head_dim / 16 is compile-time per case so the accumulators stay in zmm.
        switch (hd / 16) {
        case 1: v_group<4, 1>(a, p, hd, base, inv, a.out + gb * 4 * hd); break;
        case 2: v_group<4, 2>(a, p, hd, base, inv, a.out + gb * 4 * hd); break;
        case 3: v_group<4, 3>(a, p, hd, base, inv, a.out + gb * 4 * hd); break;
        default: v_group<4, 4>(a, p, hd, base, inv, a.out + gb * 4 * hd); break;
        }
    }
}


__attribute__((target("avx512f,avx512vl,avx2,fma")))
void attention_decode_avx2_grouped_kt(const AttnArgs & a, float * scratch, const float * kt) {
    const size_t hd = a.head_dim;
    const size_t g = a.n_heads / a.n_kv_heads;
    if (g != 4 || hd != 64) { attention_decode_avx2_grouped(a, scratch); return; }

    alignas(64) float mxv[4], inv[4];
    for (size_t h = 0; h < 4; ++h) mxv[h] = -INFINITY;
    const size_t blocks = (a.n_kv + KT_BLK - 1) / KT_BLK;

    for (size_t gb = 0; gb < a.n_kv_heads; ++gb) {
        const size_t base = gb * hd;
        const float * q = a.q + gb * 4 * hd;
        float * p = scratch;

        for (size_t blk = 0; blk < blocks; ++blk) {
            const size_t t0 = blk * KT_BLK;
            const size_t n = std::min(KT_BLK, a.n_kv - t0);
            const float * row = kt + ((gb * blocks + blk) * hd) * KT_BLK;
            if (n == KT_BLK) {
                alignas(64) float tile[4 * 16];
                for (size_t off = 0; off < KT_BLK; off += 16) {
                    scores_kt<16>(q, row + off, hd, a.n_kv, KT_BLK, a.scale, tile);
                    for (size_t j = 0; j < 16; ++j)
                        for (int h = 0; h < 4; ++h) {
                            const float sc = tile[j * 4 + h];
                            p[h * a.n_kv + t0 + off + j] = sc;
                            mxv[h] = std::max(mxv[h], sc);
                        }
                }
            } else {
                for (size_t j = 0; j < n; ++j)
                    for (int h = 0; h < 4; ++h) {
                        const size_t hi = gb * 4 + h;
                        const float * k = a.k_cache + (t0 + j) * a.n_kv_heads * hd + base;
                        const float sc = dot_avx2(a.q + hi * hd, k, hd) * a.scale;
                        p[h * a.n_kv + t0 + j] = sc;
                        mxv[h] = std::max(mxv[h], sc);
                    }
            }
        }

        for (size_t h = 0; h < 4; ++h) inv[h] = 1.0f / softmax_exp_inplace(p + h * a.n_kv, a.n_kv, mxv[h]);
        switch (hd / 16) {
        case 1: v_group<4, 1>(a, p, hd, base, inv, a.out + gb * 4 * hd); break;
        case 2: v_group<4, 2>(a, p, hd, base, inv, a.out + gb * 4 * hd); break;
        case 3: v_group<4, 3>(a, p, hd, base, inv, a.out + gb * 4 * hd); break;
        default: v_group<4, 4>(a, p, hd, base, inv, a.out + gb * 4 * hd); break;
        }
    }
}

void attention_decode_grouped_kt(const AttnArgs & a, float * scratch, const float * kt) {
#if defined(__x86_64__) || defined(__i386__)
    if (cpu_isa() == Isa::Avx512Vnni) { attention_decode_avx2_grouped_kt(a, scratch, kt); return; }
#else
    (void)kt;
#endif
    attention_decode_avx2_grouped(a, scratch);
}

#else
void attention_decode_avx2(const AttnArgs & a, float * scratch) {
    attention_decode_ref(a, scratch);
}

#endif

size_t attention_scratch_floats(const AttnArgs & a) {
    // EXP-0012 showed the grouped loop loses at depth, so scoring stays per
    // query head and scratch only needs one head's scores.
    (void)a;
    return a.n_kv + 8;
}

void attention_decode_phases(const AttnArgs & a, float * scratch, double out_us[3]) {
    // Mirrors attention_decode pass by pass; keep the two in step.
    double us[3] = {0, 0, 0};
    const size_t hd = a.head_dim;
    if (cpu_isa() == Isa::Scalar) {
        attention_decode(a, scratch);
        out_us[0] = out_us[1] = out_us[2] = 0.0;
        return;
    }
#if defined(__x86_64__) || defined(__i386__)
    for (size_t h = 0; h < a.n_heads; ++h) {
        AttnArgs one = a;
        one.n_heads = 1;
        one.q = a.q + h * hd;
        one.out = a.out + h * hd;
        one.base = head_of(h, a.n_heads, a.n_kv_heads) * hd;

        const auto t0 = std::chrono::steady_clock::now();
        const float mx = attn_head_scores(one, scratch);
        const auto t1 = std::chrono::steady_clock::now();
        const float sum = softmax_exp_inplace(scratch, a.n_kv, mx);
        const auto t2 = std::chrono::steady_clock::now();
        attn_head_v(one, scratch, 1.0f / sum);
        const auto t3 = std::chrono::steady_clock::now();
        us[0] += std::chrono::duration<double, std::micro>(t1 - t0).count();
        us[1] += std::chrono::duration<double, std::micro>(t2 - t1).count();
        us[2] += std::chrono::duration<double, std::micro>(t3 - t2).count();
    }
#endif
    for (int i = 0; i < 3; ++i) out_us[i] = us[i];
}

void attention_decode_grouped(const AttnArgs & a, float * scratch) {
    if (cpu_isa() == Isa::Scalar) {
        attention_decode_ref(a, scratch);
        return;
    }
    attention_decode_avx2_grouped(a, scratch);
}

void attention_decode(const AttnArgs & a, float * scratch) {
    if (cpu_isa() == Isa::Scalar) {
        attention_decode_ref(a, scratch);
        return;
    }
    attention_decode_avx2(a, scratch);
}

} // namespace kata
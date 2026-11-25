#include "kata/attn.h"

#include "kata/cpu.h"

#include <algorithm>
#include <cmath>

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

__attribute__((target("avx2,fma")))
void attention_decode_avx2(const AttnArgs & a, float * scratch) {
    const size_t hd = a.head_dim;
    for (size_t h = 0; h < a.n_heads; ++h) {
        const float * q = a.q + h * hd;
        const size_t base = head_of(h, a.n_heads, a.n_kv_heads) * hd;

        float mx = -INFINITY;
        for (size_t t = 0; t < a.n_kv; ++t) {
            const float * k = a.k_cache + t * a.n_kv_heads * hd + base;
            const float s = dot_avx2(q, k, hd) * a.scale;
            scratch[t] = s;
            mx = std::max(mx, s);
        }

        // exp() has no vector form here, but it is one call per position rather
        // than one per element, so this pass stays cheap relative to the dots.
        float sum = 0.0f;
        for (size_t t = 0; t < a.n_kv; ++t) {
            scratch[t] = std::exp(scratch[t] - mx);
            sum += scratch[t];
        }

        float * o = a.out + h * hd;
        const float inv = 1.0f / sum;
        const size_t nv = (hd / 8) * 8;

        for (size_t i = 0; i < nv; i += 8) _mm256_storeu_ps(o + i, _mm256_setzero_ps());
        for (size_t i = nv; i < hd; ++i) o[i] = 0.0f;

        // Position-major v means walking t is sequential; the output row stays
        // in L1 across positions, so re-reading it each step costs nothing.
        for (size_t t = 0; t < a.n_kv; ++t) {
            const __m256 pv = _mm256_set1_ps(scratch[t]);
            const float * v = a.v_cache + t * a.n_kv_heads * hd + base;
            for (size_t i = 0; i < nv; i += 8)
                _mm256_storeu_ps(o + i, _mm256_fmadd_ps(pv, _mm256_loadu_ps(v + i), _mm256_loadu_ps(o + i)));
            for (size_t i = nv; i < hd; ++i) o[i] += scratch[t] * v[i];
        }
        const __m256 iv = _mm256_set1_ps(inv);
        for (size_t i = 0; i < nv; i += 8)
            _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_loadu_ps(o + i), iv));
        for (size_t i = nv; i < hd; ++i) o[i] *= inv;
    }
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

void attention_decode(const AttnArgs & a, float * scratch) {
    if (cpu_isa() == Isa::Scalar) {
        attention_decode_ref(a, scratch);
        return;
    }
    attention_decode_avx2(a, scratch);
}

} // namespace kata
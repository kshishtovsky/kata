// Prefill attention (EXP-0024). Shape is many queries over a shared kv prefix,
// which is a different kernel from decode attention: the scores of one query row
// are n_kv floats, so materialising them costs more traffic than the dots that
// produced them. This walks the kv prefix once per query row with an online
// softmax and keeps the weighted-v accumulator in registers the whole time.
#include "kata/prefill.h"
#include "kata/cpu.h"
#include "kata/pool.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace kata {

// Cache strides, in elements. Zero means the position-major default
// [position][kv_head][head_dim]; the engine runs head-major
// [kv_head][position][head_dim], where one kv head is a contiguous stream
// (EXP-0045: 45 -> 75 GB/s on the same walk).
namespace {
inline size_t pos_stride_of(size_t stride_pos, size_t n_kv_heads, size_t head_dim) {
    return stride_pos ? stride_pos : n_kv_heads * head_dim;
}
inline size_t head_stride_of(size_t stride_head, size_t head_dim) {
    return stride_head ? stride_head : head_dim;
}
} // namespace

void attention_prefill_ref(const float * q, size_t n_tok,
                           const float * k_cache, const float * v_cache,
                           size_t n_kv_begin, size_t n_heads, size_t n_kv_heads,
                           size_t head_dim, float scale, float * out,
                           size_t stride_pos, size_t stride_head) {
    const size_t group = n_heads / n_kv_heads;
    const size_t kv_stride = pos_stride_of(stride_pos, n_kv_heads, head_dim);
    const size_t hstride = head_stride_of(stride_head, head_dim);
    for (size_t t = 0; t < n_tok; ++t) {
        const size_t n_kv = n_kv_begin + t + 1;
        for (size_t h = 0; h < n_heads; ++h) {
            const float * qv = q + (t * n_heads + h) * head_dim;
            const size_t kvh = h / group;
            float m = -INFINITY, s = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) out[(t * n_heads + h) * head_dim + d] = 0.0f;
            for (size_t p = 0; p < n_kv; ++p) {
                const float * kp = k_cache + p * kv_stride + kvh * hstride;
                float sc = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) sc += qv[d] * kp[d];
                sc *= scale;
                const float m_new = std::max(m, sc);
                const float corr = std::exp(m - m_new);
                const float e = std::exp(sc - m_new);
                m = m_new;
                s = s * corr + e;
                float * o = out + (t * n_heads + h) * head_dim;
                const float * vp = v_cache + p * kv_stride + kvh * hstride;
                for (size_t d = 0; d < head_dim; ++d) o[d] = o[d] * corr + e * vp[d];
            }
            float * o = out + (t * n_heads + h) * head_dim;
            const float inv = 1.0f / s;
            for (size_t d = 0; d < head_dim; ++d) o[d] *= inv;
        }
    }
}

#if defined(__x86_64__) || defined(__i386__)

namespace {

// Degree-5 minimax on the fractional part: exp(x) = 2^(x*log2e).
// Max relative error 2.94e-6 on [-87, 88].
__attribute__((target("avx2,fma")))
inline float exp_fast(float x) {
    x = std::min(x, 88.0f);
    x = std::max(x, -87.0f);
    const float y = x * 1.44269504088896340f;
    const float kf = std::round(y);
    const float f = y - kf;
    float p = 1.3333558146e-03f;
    p = p * f + 9.6181291076e-03f;
    p = p * f + 5.5504108665e-02f;
    p = p * f + 2.4022650696e-01f;
    p = p * f + 6.9314718056e-01f;
    p = p * f + 1.0f;
    const int ki = static_cast<int>(kf);
    const int pow2_i = (ki + 127) << 23;
    float pow2;
    std::memcpy(&pow2, &pow2_i, sizeof(float));
    return p * pow2;
}

__attribute__((target("avx2,fma")))
inline __m128 exp128_ps(__m128 x) {
    x = _mm_min_ps(x, _mm_set1_ps(88.0f));
    x = _mm_max_ps(x, _mm_set1_ps(-87.0f));
    const __m128 y = _mm_mul_ps(x, _mm_set1_ps(1.44269504088896340f));
    const __m128 kf = _mm_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m128 f = _mm_sub_ps(y, kf);
    // Taylor to degree 10 of exp(f*ln2) in Horner form: float32 rounding, not the
    // polynomial, bounds the error (max 3e-7 relative on [-1,1]).
    __m128 p = _mm_set1_ps(7.0549116208e-09f);
    p = _mm_fmadd_ps(p, f, _mm_set1_ps(1.0178086009e-07f));
    p = _mm_fmadd_ps(p, f, _mm_set1_ps(1.3215486790e-06f));
    p = _mm_fmadd_ps(p, f, _mm_set1_ps(1.5252733804e-05f));
    p = _mm_fmadd_ps(p, f, _mm_set1_ps(1.5403530393e-04f));
    p = _mm_fmadd_ps(p, f, _mm_set1_ps(1.3333558146e-03f));
    p = _mm_fmadd_ps(p, f, _mm_set1_ps(9.6181291076e-03f));
    p = _mm_fmadd_ps(p, f, _mm_set1_ps(5.5504108665e-02f));
    p = _mm_fmadd_ps(p, f, _mm_set1_ps(2.4022650696e-01f));
    p = _mm_fmadd_ps(p, f, _mm_set1_ps(6.9314718056e-01f));
    p = _mm_fmadd_ps(p, f, _mm_set1_ps(1.0000000000e+00f));
    const __m128i ki = _mm_cvtps_epi32(kf);
    const __m128 pow2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(ki, _mm_set1_epi32(127)), 23));
    return _mm_mul_ps(p, pow2);
}

__attribute__((target("avx2,fma")))
inline __m256 exp256_ps(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    const __m256 y = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896340f));
    const __m256 kf = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 f = _mm256_sub_ps(y, kf);
    __m256 p = _mm256_set1_ps(1.3333558146e-03f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6181291076e-03f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5504108665e-02f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022650696e-01f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718056e-01f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    const __m256i ki = _mm256_cvtps_epi32(kf);
    const __m256 pow2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23));
    return _mm256_mul_ps(p, pow2);
}

__attribute__((target("avx2,fma")))
inline float dot_avx2(const float * x, const float * y, size_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8), a1);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), a0);
    __m256 s = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(s), hi = _mm256_extractf128_ps(s, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float r = _mm_cvtss_f32(lo);
    for (; i < n; ++i) r += x[i] * y[i];
    return r;
}

// nv = number of __m256 accumulators covering head_dim.
template <int NV>
__attribute__((target("avx2,fma")))
void prefill_rows(const float * q, size_t n_tok, const float * k_cache, const float * v_cache,
                  size_t n_kv_begin, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                  float scale, float * out, size_t stride_pos, size_t stride_head) {
    const size_t group = n_heads / n_kv_heads;
    const size_t kv_stride = pos_stride_of(stride_pos, n_kv_heads, head_dim);
    const size_t hstride = head_stride_of(stride_head, head_dim);
    for (size_t t = 0; t < n_tok; ++t) {
        const size_t n_kv = n_kv_begin + t + 1;
        for (size_t kvh = 0; kvh < n_kv_heads; ++kvh) {
            const size_t h_base = kvh * group;
            size_t h = 0;
            // 2-head tiling when group >= 2: loads K and V once for two query heads
            for (; h + 2 <= group; h += 2) {
                const size_t h0 = h_base + h;
                const size_t h1 = h0 + 1;
                const float * qv0 = q + (t * n_heads + h0) * head_dim;
                const float * qv1 = q + (t * n_heads + h1) * head_dim;

                __m256 acc0[NV], acc1[NV];
                for (int r = 0; r < NV; ++r) {
                    acc0[r] = _mm256_setzero_ps();
                    acc1[r] = _mm256_setzero_ps();
                }
                float m0 = -INFINITY, s0 = 0.0f;
                float m1 = -INFINITY, s1 = 0.0f;

                for (size_t p = 0; p < n_kv; ++p) {
                    const float * kp = k_cache + p * kv_stride + kvh * hstride;
                    const float sc0 = dot_avx2(qv0, kp, head_dim) * scale;
                    const float sc1 = dot_avx2(qv1, kp, head_dim) * scale;
                    const float * vp = v_cache + p * kv_stride + kvh * hstride;

                    if (sc0 > m0) {
                        const float corr0 = m0 == -INFINITY ? 0.0f : exp_fast(m0 - sc0);
                        const __m256 cv0 = _mm256_set1_ps(corr0);
                        s0 = s0 * corr0 + 1.0f;
                        m0 = sc0;
                        for (int r = 0; r < NV; ++r)
                            acc0[r] = _mm256_add_ps(_mm256_mul_ps(acc0[r], cv0), _mm256_loadu_ps(vp + 8 * r));
                    } else {
                        const float e0 = exp_fast(sc0 - m0);
                        s0 += e0;
                        const __m256 ev0 = _mm256_set1_ps(e0);
                        for (int r = 0; r < NV; ++r)
                            acc0[r] = _mm256_fmadd_ps(ev0, _mm256_loadu_ps(vp + 8 * r), acc0[r]);
                    }

                    if (sc1 > m1) {
                        const float corr1 = m1 == -INFINITY ? 0.0f : exp_fast(m1 - sc1);
                        const __m256 cv1 = _mm256_set1_ps(corr1);
                        s1 = s1 * corr1 + 1.0f;
                        m1 = sc1;
                        for (int r = 0; r < NV; ++r)
                            acc1[r] = _mm256_add_ps(_mm256_mul_ps(acc1[r], cv1), _mm256_loadu_ps(vp + 8 * r));
                    } else {
                        const float e1 = exp_fast(sc1 - m1);
                        s1 += e1;
                        const __m256 ev1 = _mm256_set1_ps(e1);
                        for (int r = 0; r < NV; ++r)
                            acc1[r] = _mm256_fmadd_ps(ev1, _mm256_loadu_ps(vp + 8 * r), acc1[r]);
                    }
                }

                const __m256 inv0 = _mm256_set1_ps(1.0f / s0);
                float * o0 = out + (t * n_heads + h0) * head_dim;
                for (int r = 0; r < NV; ++r) _mm256_storeu_ps(o0 + 8 * r, _mm256_mul_ps(acc0[r], inv0));

                const __m256 inv1 = _mm256_set1_ps(1.0f / s1);
                float * o1 = out + (t * n_heads + h1) * head_dim;
                for (int r = 0; r < NV; ++r) _mm256_storeu_ps(o1 + 8 * r, _mm256_mul_ps(acc1[r], inv1));
            }

            for (; h < group; ++h) {
                const size_t h0 = h_base + h;
                const float * qv = q + (t * n_heads + h0) * head_dim;
                __m256 acc[NV];
                for (int r = 0; r < NV; ++r) acc[r] = _mm256_setzero_ps();
                float m = -INFINITY, s = 0.0f;
                for (size_t p = 0; p < n_kv; ++p) {
                    const float * kp = k_cache + p * kv_stride + kvh * hstride;
                    const float sc = dot_avx2(qv, kp, head_dim) * scale;
                    const float * vp = v_cache + p * kv_stride + kvh * hstride;
                    if (sc > m) {
                        const float corr = m == -INFINITY ? 0.0f : exp_fast(m - sc);
                        const __m256 cv = _mm256_set1_ps(corr);
                        s = s * corr + 1.0f;
                        m = sc;
                        for (int r = 0; r < NV; ++r)
                            acc[r] = _mm256_add_ps(_mm256_mul_ps(acc[r], cv), _mm256_loadu_ps(vp + 8 * r));
                    } else {
                        const float e = exp_fast(sc - m);
                        s += e;
                        const __m256 ev = _mm256_set1_ps(e);
                        for (int r = 0; r < NV; ++r)
                            acc[r] = _mm256_fmadd_ps(ev, _mm256_loadu_ps(vp + 8 * r), acc[r]);
                    }
                }
                const __m256 inv = _mm256_set1_ps(1.0f / s);
                float * o = out + (t * n_heads + h0) * head_dim;
                for (int r = 0; r < NV; ++r) _mm256_storeu_ps(o + 8 * r, _mm256_mul_ps(acc[r], inv));
            }
        }
    }
}

// Fallback for head_dim not in specialized table: AVX2 vector accumulation with scalar tail.
__attribute__((target("avx2,fma")))
void prefill_rows_generic(const float * q, size_t n_tok, const float * k_cache, const float * v_cache,
                          size_t n_kv_begin, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                          float scale, float * out, size_t stride_pos, size_t stride_head) {
    const size_t group = n_heads / n_kv_heads;
    const size_t kv_stride = pos_stride_of(stride_pos, n_kv_heads, head_dim);
    const size_t hstride = head_stride_of(stride_head, head_dim);
    std::vector<float> acc_vec;
    float acc_stack[256];
    float * acc = head_dim <= 256 ? acc_stack : (acc_vec.resize(head_dim), acc_vec.data());

    for (size_t t = 0; t < n_tok; ++t) {
        const size_t n_kv = n_kv_begin + t + 1;
        for (size_t h = 0; h < n_heads; ++h) {
            const float * qv = q + (t * n_heads + h) * head_dim;
            const size_t kvh = h / group;
            for (size_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
            float m = -INFINITY, s = 0.0f;
            for (size_t p = 0; p < n_kv; ++p) {
                const float * kp = k_cache + p * kv_stride + kvh * hstride;
                const float sc = dot_avx2(qv, kp, head_dim) * scale;
                const float * vp = v_cache + p * kv_stride + kvh * hstride;
                if (sc > m) {
                    const float corr = m == -INFINITY ? 0.0f : exp_fast(m - sc);
                    const __m256 cv = _mm256_set1_ps(corr);
                    s = s * corr + 1.0f;
                    m = sc;
                    size_t d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        __m256 a = _mm256_loadu_ps(acc + d);
                        __m256 v = _mm256_loadu_ps(vp + d);
                        _mm256_storeu_ps(acc + d, _mm256_add_ps(_mm256_mul_ps(a, cv), v));
                    }
                    for (; d < head_dim; ++d) acc[d] = acc[d] * corr + vp[d];
                } else {
                    const float e = exp_fast(sc - m);
                    s += e;
                    const __m256 ev = _mm256_set1_ps(e);
                    size_t d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        __m256 a = _mm256_loadu_ps(acc + d);
                        __m256 v = _mm256_loadu_ps(vp + d);
                        _mm256_storeu_ps(acc + d, _mm256_fmadd_ps(ev, v, a));
                    }
                    for (; d < head_dim; ++d) acc[d] += e * vp[d];
                }
            }
            float * o = out + (t * n_heads + h) * head_dim;
            const float inv = 1.0f / s;
            const __m256 inv_v = _mm256_set1_ps(inv);
            size_t d = 0;
            for (; d + 8 <= head_dim; d += 8)
                _mm256_storeu_ps(o + d, _mm256_mul_ps(_mm256_loadu_ps(acc + d), inv_v));
            for (; d < head_dim; ++d) o[d] = acc[d] * inv;
        }
    }
}

} // namespace

#if defined(__x86_64__) || defined(__i386__)

namespace {

// Four query heads share one kv head, so the naive loop reads every k and v row
// four times: at 25k tokens that is 1.6 TB of traffic for 400 MB of data, and it
// showed up as 49% of prefill time. This walks the kv row once per group of four
// heads. Accumulators are 4 heads x 4 zmm (head_dim 64) = 16 of 32 zmm.
// Scores for the 4 heads are computed in parallel with AVX-512 and normalized
// via fast vector exp128_ps in the common non-max-update path.
__attribute__((target("avx512f,avx512vl,avx512bw,avx512dq,avx2,fma")))
void prefill_group4(const float * q, size_t n_tok, const float * k_cache,
                    const float * v_cache, size_t n_kv_begin, size_t n_heads,
                    size_t n_kv_heads, float scale, float * out,
                    size_t stride_pos, size_t stride_head) {
    constexpr size_t HD = 64, G = 4;
    const size_t kv_stride = pos_stride_of(stride_pos, n_kv_heads, HD);
    const size_t hstride = head_stride_of(stride_head, HD);
    for (size_t t = 0; t < n_tok; ++t) {
        const size_t n_kv = n_kv_begin + t + 1;
        for (size_t g = 0; g < n_heads; g += G) {
            const size_t kvh = g / G;
            const float * qv[G];
            for (size_t h = 0; h < G; ++h) qv[h] = q + (t * n_heads + g + h) * HD;
            __m512 acc[G][4];
            float m[G], s[G];
            for (size_t h = 0; h < G; ++h) {
                for (int r = 0; r < 4; ++r) acc[h][r] = _mm512_setzero_ps();
                m[h] = -INFINITY;
                s[h] = 0.0f;
            }
            for (size_t p = 0; p < n_kv; ++p) {
                const float * kp = k_cache + p * kv_stride + kvh * hstride;
                _mm_prefetch(reinterpret_cast<const char *>(kp + 4 * kv_stride), _MM_HINT_T0);
                const __m512 k0 = _mm512_loadu_ps(kp), k1 = _mm512_loadu_ps(kp + 16),
                             k2 = _mm512_loadu_ps(kp + 32), k3 = _mm512_loadu_ps(kp + 48);
                const float * vp = v_cache + p * kv_stride + kvh * hstride;
                _mm_prefetch(reinterpret_cast<const char *>(vp + 4 * kv_stride), _MM_HINT_T0);

                // One 4-way horizontal reduce instead of four serial ones. Each
                // _mm512_reduce_add_ps is ~6 dependent uops and there were four of
                // them per (row, position, group), i.e. 553 M per 1024-token chunk
                // at a 16k prefix. Packing the four dot products into one xmm
                // costs the same shuffles once and feeds exp128_ps directly.
                alignas(16) float sc[4];
                {
                    __m512 a[G];
                    for (size_t h = 0; h < G; ++h) {
                        const float * qq = qv[h];
                        __m512 v = _mm512_mul_ps(_mm512_loadu_ps(qq), k0);
                        v = _mm512_fmadd_ps(_mm512_loadu_ps(qq + 16), k1, v);
                        v = _mm512_fmadd_ps(_mm512_loadu_ps(qq + 32), k2, v);
                        a[h] = _mm512_fmadd_ps(_mm512_loadu_ps(qq + 48), k3, v);
                    }
                    const __m256 s0 = _mm256_add_ps(_mm512_castps512_ps256(a[0]), _mm512_extractf32x8_ps(a[0], 1));
                    const __m256 s1 = _mm256_add_ps(_mm512_castps512_ps256(a[1]), _mm512_extractf32x8_ps(a[1], 1));
                    const __m256 s2 = _mm256_add_ps(_mm512_castps512_ps256(a[2]), _mm512_extractf32x8_ps(a[2], 1));
                    const __m256 s3 = _mm256_add_ps(_mm512_castps512_ps256(a[3]), _mm512_extractf32x8_ps(a[3], 1));
                    const __m256 h01 = _mm256_hadd_ps(s0, s1);
                    const __m256 h23 = _mm256_hadd_ps(s2, s3);
                    const __m256 q4 = _mm256_hadd_ps(h01, h23);
                    const __m128 lo = _mm256_castps256_ps128(q4);
                    _mm_store_ps(sc, _mm_mul_ps(_mm_add_ps(lo, _mm256_extractf128_ps(q4, 1)),
                                                _mm_set1_ps(scale)));
                }

                const bool any_max = (sc[0] > m[0]) | (sc[1] > m[1]) | (sc[2] > m[2]) | (sc[3] > m[3]);
                __m512 vv[4];
                for (int r = 0; r < 4; ++r) vv[r] = _mm512_loadu_ps(vp + 16 * r);

                if (__builtin_expect(!any_max, 1)) {
                    // Common path (99.5% of positions): running max does not change.
                    // Evaluate all 4 exponents in a single SIMD vector pass.
                    const __m128 sc_v = _mm_set_ps(sc[3], sc[2], sc[1], sc[0]);
                    const __m128 m_v  = _mm_set_ps(m[3],  m[2],  m[1],  m[0]);
                    alignas(16) float e[4];
                    _mm_store_ps(e, exp128_ps(_mm_sub_ps(sc_v, m_v)));

                    for (size_t h = 0; h < G; ++h) {
                        s[h] += e[h];
                        const __m512 ev = _mm512_set1_ps(e[h]);
                        for (int r = 0; r < 4; ++r) acc[h][r] = _mm512_fmadd_ps(ev, vv[r], acc[h][r]);
                    }
                } else {
                    // Max-update path (rare, O(log N) times per row).
                    for (size_t h = 0; h < G; ++h) {
                        if (sc[h] > m[h]) {
                            const float corr = m[h] == -INFINITY ? 0.0f : std::exp(m[h] - sc[h]);
                            const __m512 cv = _mm512_set1_ps(corr);
                            s[h] = s[h] * corr + 1.0f;
                            m[h] = sc[h];
                            for (int r = 0; r < 4; ++r)
                                acc[h][r] = _mm512_add_ps(_mm512_mul_ps(acc[h][r], cv), vv[r]);
                        } else {
                            const float e = std::exp(sc[h] - m[h]);
                            s[h] += e;
                            const __m512 ev = _mm512_set1_ps(e);
                            for (int r = 0; r < 4; ++r) acc[h][r] = _mm512_fmadd_ps(ev, vv[r], acc[h][r]);
                        }
                    }
                }
            }
            for (size_t h = 0; h < G; ++h) {
                const __m512 inv = _mm512_set1_ps(1.0f / s[h]);
                float * o = out + (t * n_heads + g + h) * HD;
                for (int r = 0; r < 4; ++r)
                    _mm512_storeu_ps(o + 16 * r, _mm512_mul_ps(acc[h][r], inv));
            }
        }
    }
}

} // namespace

#endif

void attention_prefill(const float * q, size_t n_tok,
                       const float * k_cache, const float * v_cache,
                       size_t n_kv_begin, size_t n_heads, size_t n_kv_heads,
                       size_t head_dim, float scale, float * out,
                       size_t stride_pos, size_t stride_head) {
    if (n_tok == 0) return;
#if defined(__x86_64__) || defined(__i386__)
    const Isa isa = cpu_isa();
    if (head_dim == 64 && n_heads == 4 * n_kv_heads && (isa == Isa::Avx512 || isa == Isa::Avx512Vnni)) {
        prefill_group4(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, scale, out,
                       stride_pos, stride_head);
        return;
    }
#endif
    if (head_dim == 0 || head_dim % 8 != 0) {
        prefill_rows_generic(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale,
                             out, stride_pos, stride_head);
        return;
    }
    switch (head_dim / 8) {
        case 1: prefill_rows<1>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim,
                        scale, out, stride_pos, stride_head); break;
        case 2: prefill_rows<2>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim,
                        scale, out, stride_pos, stride_head); break;
        case 4: prefill_rows<4>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim,
                        scale, out, stride_pos, stride_head); break;
        case 8: prefill_rows<8>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim,
                        scale, out, stride_pos, stride_head); break;
        case 10: prefill_rows<10>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim,
                        scale, out, stride_pos, stride_head); break; // 80
        case 12: prefill_rows<12>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim,
                        scale, out, stride_pos, stride_head); break; // 96
        case 16: prefill_rows<16>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim,
                        scale, out, stride_pos, stride_head); break; // 128
        case 32: prefill_rows<32>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim,
                        scale, out, stride_pos, stride_head); break; // 256
        default:
            prefill_rows_generic(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale,
                                 out, stride_pos, stride_head);
            break;
    }
}

#endif

std::vector<size_t> prefill_weighted_partition_bounds(size_t n_tok, size_t n_kv_begin, unsigned nw) {
    if (nw == 0) return {0};
    std::vector<size_t> bounds(nw + 1, 0);
    bounds[nw] = n_tok;
    if (nw <= 1 || n_tok == 0) return bounds;

    const double b = 2.0 * static_cast<double>(n_kv_begin) + 1.0;
    const double w_total = 0.5 * (static_cast<double>(n_tok) * static_cast<double>(n_tok) + b * static_cast<double>(n_tok));

    for (unsigned w = 1; w < nw; ++w) {
        const double w_target = (static_cast<double>(w) * w_total) / static_cast<double>(nw);
        const double disc = b * b + 8.0 * w_target;
        const double t = 0.5 * (-b + std::sqrt(disc));
        const size_t bound = static_cast<size_t>(std::clamp<double>(std::round(t), 0.0, static_cast<double>(n_tok)));
        bounds[w] = std::clamp(bound, bounds[w - 1], n_tok);
    }
    return bounds;
}

void attention_prefill_mt(const float * q, size_t n_tok,
                          const float * k_cache, const float * v_cache,
                          size_t n_kv_begin, size_t n_heads, size_t n_kv_heads,
                          size_t head_dim, float scale, float * out, ThreadPool & pool,
                          size_t stride_pos, size_t stride_head) {
    if (n_tok == 0) return;
    const unsigned nw = pool.size();
    if (nw <= 1) {
        attention_prefill(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale,
                          out, stride_pos, stride_head);
        return;
    }

    // Weighted thread partitioning: causal attention cost grows with prefix length
    // (n_kv_begin + t + 1). Equal token partitioning left thread 0 underloaded
    // and the last thread overloaded by up to 30x. Partitioning by cumulative work W
    // balances load across all workers to within 1-2%.
    const std::vector<size_t> bounds = prefill_weighted_partition_bounds(n_tok, n_kv_begin, nw);
    const size_t qw = n_heads * head_dim;
    pool.run([&](unsigned w) {
        const size_t lo = bounds[w];
        const size_t hi = bounds[w + 1];
        if (lo >= hi) return;
        attention_prefill(q + lo * qw, hi - lo,
                          k_cache, v_cache, n_kv_begin + lo, n_heads, n_kv_heads,
                          head_dim, scale, out + lo * qw, stride_pos, stride_head);
    });
}

#if !defined(__x86_64__) && !defined(__i386__)

void attention_prefill(const float * q, size_t n_tok,
                       const float * k_cache, const float * v_cache,
                       size_t n_kv_begin, size_t n_heads, size_t n_kv_heads,
                       size_t head_dim, float scale, float * out) {
    attention_prefill_ref(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale,
                          out, stride_pos, stride_head);
}

#endif

} // namespace kata
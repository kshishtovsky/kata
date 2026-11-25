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

void attention_prefill_ref(const float * q, size_t n_tok,
                           const float * k_cache, const float * v_cache,
                           size_t n_kv_begin, size_t n_heads, size_t n_kv_heads,
                           size_t head_dim, float scale, float * out) {
    const size_t group = n_heads / n_kv_heads;
    const size_t kv_stride = n_kv_heads * head_dim;
    for (size_t t = 0; t < n_tok; ++t) {
        const size_t n_kv = n_kv_begin + t + 1;
        for (size_t h = 0; h < n_heads; ++h) {
            const float * qv = q + (t * n_heads + h) * head_dim;
            const size_t kvh = h / group;
            float m = -INFINITY, s = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) out[(t * n_heads + h) * head_dim + d] = 0.0f;
            for (size_t p = 0; p < n_kv; ++p) {
                const float * kp = k_cache + p * kv_stride + kvh * head_dim;
                float sc = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) sc += qv[d] * kp[d];
                sc *= scale;
                const float m_new = std::max(m, sc);
                const float corr = std::exp(m - m_new);
                const float e = std::exp(sc - m_new);
                m = m_new;
                s = s * corr + e;
                float * o = out + (t * n_heads + h) * head_dim;
                const float * vp = v_cache + p * kv_stride + kvh * head_dim;
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

// Same polynomial as the decode kernel: exp(x) = 2^(x*log2e) with a degree-5
// minimax on the fractional part. Max relative error 6.8e-6 on [-80, 80].
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
                  float scale, float * out) {
    const size_t group = n_heads / n_kv_heads;
    const size_t kv_stride = n_kv_heads * head_dim;
    for (size_t t = 0; t < n_tok; ++t) {
        const size_t n_kv = n_kv_begin + t + 1;
        for (size_t h = 0; h < n_heads; ++h) {
            const float * qv = q + (t * n_heads + h) * head_dim;
            const size_t kvh = h / group;
            __m256 acc[NV];
            for (int r = 0; r < NV; ++r) acc[r] = _mm256_setzero_ps();
            float m = -INFINITY, s = 0.0f;
            for (size_t p = 0; p < n_kv; ++p) {
                const float * kp = k_cache + p * kv_stride + kvh * head_dim;
                const float sc = dot_avx2(qv, kp, head_dim) * scale;
                // The running max rises O(log n) times over a whole row, so most
                // positions take the pure-FMA path: acc += e*v with no rescale of
                // the 64-float accumulator. Rescaling every position was 2 mul + 1
                // add per element instead of one fused multiply-add.
                const float * vp = v_cache + p * kv_stride + kvh * head_dim;
                if (sc > m) {
                    const float corr = m == -INFINITY ? 0.0f : std::exp(m - sc);
                    const __m256 cv = _mm256_set1_ps(corr);
                    s = s * corr + 1.0f;
                    m = sc;
                    for (int r = 0; r < NV; ++r)
                        acc[r] = _mm256_add_ps(_mm256_mul_ps(acc[r], cv), _mm256_loadu_ps(vp + 8 * r));
                } else {
                    const float e = std::exp(sc - m);
                    s += e;
                    const __m256 ev = _mm256_set1_ps(e);
                    for (int r = 0; r < NV; ++r)
                        acc[r] = _mm256_fmadd_ps(ev, _mm256_loadu_ps(vp + 8 * r), acc[r]);
                }
            }
            const __m256 inv = _mm256_set1_ps(1.0f / s);
            float * o = out + (t * n_heads + h) * head_dim;
            for (int r = 0; r < NV; ++r) _mm256_storeu_ps(o + 8 * r, _mm256_mul_ps(acc[r], inv));
        }
    }
}

// Fallback for head_dim not divisible by 8: same loop, scalar accumulation.
__attribute__((target("avx2,fma")))
void prefill_rows_generic(const float * q, size_t n_tok, const float * k_cache, const float * v_cache,
                          size_t n_kv_begin, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                          float scale, float * out) {
    const size_t group = n_heads / n_kv_heads;
    const size_t kv_stride = n_kv_heads * head_dim;
    std::vector<float> acc(head_dim);
    for (size_t t = 0; t < n_tok; ++t) {
        const size_t n_kv = n_kv_begin + t + 1;
        for (size_t h = 0; h < n_heads; ++h) {
            const float * qv = q + (t * n_heads + h) * head_dim;
            const size_t kvh = h / group;
            std::fill(acc.begin(), acc.end(), 0.0f);
            float m = -INFINITY, s = 0.0f;
            for (size_t p = 0; p < n_kv; ++p) {
                const float * kp = k_cache + p * kv_stride + kvh * head_dim;
                const float sc = dot_avx2(qv, kp, head_dim) * scale;
                const float m_new = sc > m ? sc : m;
                const float corr = std::exp(m - m_new);
                const float e = std::exp(sc - m_new);
                m = m_new;
                s = s * corr + e;
                const float * vp = v_cache + p * kv_stride + kvh * head_dim;
                for (size_t d = 0; d < head_dim; ++d) acc[d] = acc[d] * corr + e * vp[d];
            }
            float * o = out + (t * n_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) o[d] = acc[d] / s;
        }
    }
}

} // namespace

#if defined(__x86_64__) || defined(__i386__)

namespace {

// Four query heads share one kv head, so the naive loop reads every k and v row
// four times: at 25k tokens that is 1.6 TB of traffic for 400 MB of data, and it
// showed up as 49% of prefill time. This walks the kv row once per group of four
// heads. Accumulators are 4 heads x 4 zmm (head_dim 64) = 16 of 32 zmm; q is
// re-read from L1 each position instead, because holding it would need the other
// 16 zmm and leave no room for accumulators.
__attribute__((target("avx512f,avx512vl,avx512bw,avx2,fma")))
void prefill_group4(const float * q, size_t n_tok, const float * k_cache,
                    const float * v_cache, size_t n_kv_begin, size_t n_heads,
                    size_t n_kv_heads, float scale, float * out) {
    constexpr size_t HD = 64, G = 4;
    const size_t kv_stride = n_kv_heads * HD;
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
                const float * kp = k_cache + p * kv_stride + kvh * HD;
                const __m512 k0 = _mm512_loadu_ps(kp), k1 = _mm512_loadu_ps(kp + 16),
                             k2 = _mm512_loadu_ps(kp + 32), k3 = _mm512_loadu_ps(kp + 48);
                const float * vp = v_cache + p * kv_stride + kvh * HD;
                __m512 vv[4];
                bool v_loaded = false;
                for (size_t h = 0; h < G; ++h) {
                    const float * qq = qv[h];
                    __m512 a = _mm512_mul_ps(_mm512_loadu_ps(qq), k0);
                    a = _mm512_fmadd_ps(_mm512_loadu_ps(qq + 16), k1, a);
                    a = _mm512_fmadd_ps(_mm512_loadu_ps(qq + 32), k2, a);
                    a = _mm512_fmadd_ps(_mm512_loadu_ps(qq + 48), k3, a);
                    const float sc = _mm512_reduce_add_ps(a) * scale;
                    if (sc > m[h]) {
                        if (!v_loaded) {
                            for (int r = 0; r < 4; ++r) vv[r] = _mm512_loadu_ps(vp + 16 * r);
                            v_loaded = true;
                        }
                        const float corr = m[h] == -INFINITY ? 0.0f : std::exp(m[h] - sc);
                        const __m512 cv = _mm512_set1_ps(corr);
                        s[h] = s[h] * corr + 1.0f;
                        m[h] = sc;
                        for (int r = 0; r < 4; ++r)
                            acc[h][r] = _mm512_add_ps(_mm512_mul_ps(acc[h][r], cv), vv[r]);
                    } else {
                        if (!v_loaded) {
                            for (int r = 0; r < 4; ++r) vv[r] = _mm512_loadu_ps(vp + 16 * r);
                            v_loaded = true;
                        }
                        const float e = std::exp(sc - m[h]);
                        s[h] += e;
                        const __m512 ev = _mm512_set1_ps(e);
                        for (int r = 0; r < 4; ++r) acc[h][r] = _mm512_fmadd_ps(ev, vv[r], acc[h][r]);
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
                       size_t head_dim, float scale, float * out) {
    if (n_tok == 0) return;
#if defined(__x86_64__) || defined(__i386__)
    if (head_dim == 64 && n_heads == 4 * n_kv_heads && cpu_isa() == Isa::Avx512Vnni) {
        prefill_group4(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, scale, out);
        return;
    }
#endif
    if (head_dim == 0 || head_dim % 8 != 0) {
        prefill_rows_generic(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale, out);
        return;
    }
    switch (head_dim / 8) {
        case 1: prefill_rows<1>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale, out); break;
        case 2: prefill_rows<2>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale, out); break;
        case 4: prefill_rows<4>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale, out); break;
        case 8: prefill_rows<8>(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale, out); break;
        default:
            prefill_rows_generic(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale, out);
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
                          size_t head_dim, float scale, float * out, ThreadPool & pool) {
    if (n_tok == 0) return;
    const unsigned nw = pool.size();
    if (nw <= 1) {
        attention_prefill(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale, out);
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
                          head_dim, scale, out + lo * qw);
    });
}

#if !defined(__x86_64__) && !defined(__i386__)

void attention_prefill(const float * q, size_t n_tok,
                       const float * k_cache, const float * v_cache,
                       size_t n_kv_begin, size_t n_heads, size_t n_kv_heads,
                       size_t head_dim, float scale, float * out) {
    attention_prefill_ref(q, n_tok, k_cache, v_cache, n_kv_begin, n_heads, n_kv_heads, head_dim, scale, out);
}

#endif

} // namespace kata
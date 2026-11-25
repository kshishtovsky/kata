#include "kata/rope.h"

#include "kata/cpu.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace kata {
namespace {

// freq_i = base^(-2i/head_dim) does not depend on the position, and pow() costs
// more than the rotation itself, so the table is built once per (head_dim, base).
const std::vector<double> & freq_table(size_t head_dim, float theta_base) {
    static thread_local size_t cached_dim = 0;
    static thread_local float cached_base = 0.0f;
    static thread_local std::vector<double> freqs;
    if (cached_dim != head_dim || cached_base != theta_base) {
        const size_t half = head_dim / 2;
        freqs.resize(half);
        const double base = (double)theta_base;
        for (size_t i = 0; i < half; ++i) {
            // Denominator is head_dim, not half*2: they differ for an odd head_dim.
            freqs[i] = std::pow(base, -2.0 * (double)i / (double)head_dim);
        }
        cached_dim = head_dim;
        cached_base = theta_base;
    }
    return freqs;
}

// theta_i = pos * base^(-2i/head_dim). Computed in double: at pos = 128k and
// base 1e7 the exponent makes float cos drift far enough to be visible.
inline void angles(size_t head_dim, size_t pos, float theta_base, std::vector<float> & cos_t,
                   std::vector<float> & sin_t) {
    const size_t half = head_dim / 2;
    cos_t.resize(half);
    sin_t.resize(half);
    const std::vector<double> & freq = freq_table(head_dim, theta_base);
    for (size_t i = 0; i < half; ++i) {
        const double a = (double)pos * freq[i];
        cos_t[i] = (float)std::cos(a);
        sin_t[i] = (float)std::sin(a);
    }
}

} // namespace

void rope_tables(size_t head_dim, size_t pos, float theta_base, float * cos_out, float * sin_out) {
    const size_t half = head_dim / 2;
    std::vector<float> c, s;
    angles(head_dim, pos, theta_base, c, s);
    for (size_t i = 0; i < half; ++i) { cos_out[i] = c[i]; sin_out[i] = s[i]; }
}

void rope_ref(size_t n_heads, size_t head_dim, float * x, size_t pos, float theta_base, bool interleaved) {
    const size_t half = head_dim / 2;
    std::vector<float> c, s;
    angles(head_dim, pos, theta_base, c, s);
    for (size_t h = 0; h < n_heads; ++h) {
        float * v = x + h * head_dim;
        for (size_t i = 0; i < half; ++i) {
            const size_t a = interleaved ? 2 * i : i;
            const size_t b = interleaved ? 2 * i + 1 : i + half;
            const float xa = v[a], xb = v[b];
            v[a] = xa * c[i] - xb * s[i];
            v[b] = xa * s[i] + xb * c[i];
        }
    }
}

#if defined(__x86_64__) || defined(__i386__)

__attribute__((target("avx2,fma")))
void rope_avx2(size_t n_heads, size_t head_dim, float * x, size_t pos, float theta_base, bool interleaved) {
    const size_t half = head_dim / 2;
    std::vector<float> c, s;
    angles(head_dim, pos, theta_base, c, s);

    for (size_t h = 0; h < n_heads; ++h) {
        float * v = x + h * head_dim;
        if (interleaved) {
            // Pairs are adjacent, so they can be handled as 8 packed (lo,hi)
            // pairs at a time with two deinterleaving shuffles per register.
            size_t i = 0;
            for (; i + 8 <= half; i += 8) {
                const __m256 cv = _mm256_loadu_ps(&c[i]);
                const __m256 sv = _mm256_loadu_ps(&s[i]);
                // xa = v[2i], xb = v[2i+1]
                __m256 a = _mm256_setr_ps(v[2 * i + 0], v[2 * i + 2], v[2 * i + 4], v[2 * i + 6],
                                          v[2 * i + 8], v[2 * i + 10], v[2 * i + 12], v[2 * i + 14]);
                __m256 b = _mm256_setr_ps(v[2 * i + 1], v[2 * i + 3], v[2 * i + 5], v[2 * i + 7],
                                          v[2 * i + 9], v[2 * i + 11], v[2 * i + 13], v[2 * i + 15]);
                const __m256 ra = _mm256_fmsub_ps(a, cv, _mm256_mul_ps(b, sv));
                const __m256 rb = _mm256_fmadd_ps(a, sv, _mm256_mul_ps(b, cv));
                const __m256 lo = _mm256_unpacklo_ps(ra, rb);
                const __m256 hi = _mm256_unpackhi_ps(ra, rb);
                _mm256_storeu_ps(v + 2 * i, _mm256_permute2f128_ps(lo, hi, 0x20));
                _mm256_storeu_ps(v + 2 * i + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
            }
            for (; i < half; ++i) {
                const float xa = v[2 * i], xb = v[2 * i + 1];
                v[2 * i] = xa * c[i] - xb * s[i];
                v[2 * i + 1] = xa * s[i] + xb * c[i];
            }
        } else {
            // Pairs are head_dim/2 apart: two independent contiguous halves.
            size_t i = 0;
            for (; i + 8 <= half; i += 8) {
                const __m256 a = _mm256_loadu_ps(v + i);
                const __m256 b = _mm256_loadu_ps(v + i + half);
                const __m256 cv = _mm256_loadu_ps(&c[i]);
                const __m256 sv = _mm256_loadu_ps(&s[i]);
                _mm256_storeu_ps(v + i, _mm256_fmsub_ps(a, cv, _mm256_mul_ps(b, sv)));
                _mm256_storeu_ps(v + i + half, _mm256_fmadd_ps(a, sv, _mm256_mul_ps(b, cv)));
            }
            for (; i < half; ++i) {
                const float xa = v[i], xb = v[i + half];
                v[i] = xa * c[i] - xb * s[i];
                v[i + half] = xa * s[i] + xb * c[i];
            }
        }
    }
}

#else

void rope_avx2(size_t n_heads, size_t head_dim, float * x, size_t pos, float theta_base, bool interleaved) {
    rope_ref(n_heads, head_dim, x, pos, theta_base, interleaved);
}

#endif

void rope(size_t n_heads, size_t head_dim, float * x, size_t pos, float theta_base, bool interleaved) {
    if (cpu_isa() == Isa::Scalar || head_dim < 2) {
        rope_ref(n_heads, head_dim, x, pos, theta_base, interleaved);
        return;
    }
    rope_avx2(n_heads, head_dim, x, pos, theta_base, interleaved);
}

} // namespace kata
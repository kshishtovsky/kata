#include "kata/norm.h"

#include "kata/cpu.h"

#include <cmath>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace kata {

void rms_norm_ref(size_t rows, size_t cols, float * out, const float * x, const float * w, float eps) {
    for (size_t r = 0; r < rows; ++r) {
        const float * xr = x + r * cols;
        float * orow = out + r * cols;
        // Double accumulator: summing 4096 squared floats in float loses enough
        // precision to change the scale in the last digits of every element.
        double sum = 0.0;
        for (size_t c = 0; c < cols; ++c) sum += (double)xr[c] * (double)xr[c];
        const float scale = 1.0f / std::sqrt((float)(sum / (double)cols) + eps);
        for (size_t c = 0; c < cols; ++c) orow[c] = xr[c] * scale * (w ? w[c] : 1.0f);
    }
}

#if defined(__x86_64__) || defined(__i386__)

__attribute__((target("avx2,fma")))
void rms_norm_avx2(size_t rows, size_t cols, float * out, const float * x, const float * w, float eps) {
    const size_t nv = cols & ~size_t(7);

    for (size_t r = 0; r < rows; ++r) {
        const float * xr = x + r * cols;
        float * orow = out + r * cols;

        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        size_t c = 0;
        for (; c + 16 <= nv; c += 16) {
            const __m256 a = _mm256_loadu_ps(xr + c);
            const __m256 b = _mm256_loadu_ps(xr + c + 8);
            s0 = _mm256_fmadd_ps(a, a, s0);
            s1 = _mm256_fmadd_ps(b, b, s1);
        }
        for (; c < nv; c += 8) {
            const __m256 a = _mm256_loadu_ps(xr + c);
            s0 = _mm256_fmadd_ps(a, a, s0);
        }
        __m256 s = _mm256_add_ps(s0, s1);
        __m128 lo = _mm256_castps256_ps128(s);
        __m128 hi = _mm256_extractf128_ps(s, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 sh = _mm_movehdup_ps(lo);
        lo = _mm_add_ps(lo, sh);
        sh = _mm_movehl_ps(sh, lo);
        float sum = _mm_cvtss_f32(_mm_add_ss(lo, sh));
        for (; c < cols; ++c) sum += xr[c] * xr[c];  // tail, only for odd widths

        const float scale = 1.0f / std::sqrt(sum / (float)cols + eps);
        const __m256 vs = _mm256_set1_ps(scale);

        c = 0;
        if (w) {
            for (; c < nv; c += 8) {
                _mm256_storeu_ps(orow + c, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + c), vs),
                                                         _mm256_loadu_ps(w + c)));
            }
            for (; c < cols; ++c) orow[c] = xr[c] * scale * w[c];
        } else {
            for (; c < nv; c += 8) _mm256_storeu_ps(orow + c, _mm256_mul_ps(_mm256_loadu_ps(xr + c), vs));
            for (; c < cols; ++c) orow[c] = xr[c] * scale;
        }
    }
}

#else

void rms_norm_avx2(size_t rows, size_t cols, float * out, const float * x, const float * w, float eps) {
    rms_norm_ref(rows, cols, out, x, w, eps);
}

#endif

void rms_norm(size_t rows, size_t cols, float * out, const float * x, const float * w, float eps) {
    if (cpu_isa() == Isa::Scalar) rms_norm_ref(rows, cols, out, x, w, eps);
    else rms_norm_avx2(rows, cols, out, x, w, eps);
}

} // namespace kata
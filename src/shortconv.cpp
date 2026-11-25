#include "kata/shortconv.h"

#include <immintrin.h>

namespace kata {

__attribute__((target("avx2,fma")))
void shortconv_bx(const float * bcx, float * bx, size_t m, size_t nc) {
    const size_t nc3 = 3 * nc;
    for (size_t i = 0; i < m; ++i) {
        const float * b = bcx + i * nc3;
        const float * x = b + 2 * nc;
        float * o = bx + i * nc;
        size_t e = 0;
        for (; e + 8 <= nc; e += 8)
            _mm256_storeu_ps(o + e, _mm256_mul_ps(_mm256_loadu_ps(b + e), _mm256_loadu_ps(x + e)));
        for (; e < nc; ++e) o[e] = b[e] * x[e];
    }
}

// kern is [e][tap] with tap fastest: tap t of channel e is kern[e*ntaps + t].
__attribute__((target("avx2,fma")))
void shortconv_conv(const float * bx, const float * bcx, size_t m, size_t nc,
                    size_t ntaps, const float * kern, float * out) {
    for (size_t i = 0; i < m; ++i) {
        const float * c = bcx + i * 3 * nc + nc;
        const float * cur = bx + i * nc;                       // bx[i]
        float * o = out + i * nc;
        size_t e = 0;
        if (ntaps == 3) {
            // The only width LFM2 uses; written out so the taps stay in registers.
            for (; e + 8 <= nc; e += 8) {
                const __m256 k0 = _mm256_setr_ps(kern[(e+0)*3], kern[(e+1)*3], kern[(e+2)*3], kern[(e+3)*3],
                                                 kern[(e+4)*3], kern[(e+5)*3], kern[(e+6)*3], kern[(e+7)*3]);
                const __m256 k1 = _mm256_setr_ps(kern[(e+0)*3+1], kern[(e+1)*3+1], kern[(e+2)*3+1], kern[(e+3)*3+1],
                                                 kern[(e+4)*3+1], kern[(e+5)*3+1], kern[(e+6)*3+1], kern[(e+7)*3+1]);
                const __m256 k2 = _mm256_setr_ps(kern[(e+0)*3+2], kern[(e+1)*3+2], kern[(e+2)*3+2], kern[(e+3)*3+2],
                                                 kern[(e+4)*3+2], kern[(e+5)*3+2], kern[(e+6)*3+2], kern[(e+7)*3+2]);
                __m256 acc = _mm256_mul_ps(k2, _mm256_loadu_ps(cur + e));
                acc = _mm256_fmadd_ps(k1, _mm256_loadu_ps(cur - nc + e), acc);
                acc = _mm256_fmadd_ps(k0, _mm256_loadu_ps(cur - 2 * nc + e), acc);
                _mm256_storeu_ps(o + e, _mm256_mul_ps(_mm256_loadu_ps(c + e), acc));
            }
        }
        for (; e < nc; ++e) {
            const float * k = kern + e * ntaps;
            float acc = 0.0f;
            for (size_t t = 0; t < ntaps; ++t)
                acc += k[t] * cur[(t + 1 - ntaps) * nc + e];   // bx[i-ntaps+1+t]
            o[e] = c[e] * acc;
        }
    }
}

} // namespace kata

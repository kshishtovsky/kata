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

__attribute__((target("avx2,fma")))
void shortconv_step(const float * b, const float * c, const float * x,
                    float * hist, const float * kern, size_t d_conv,
                    size_t nc, float * out) {
    if (d_conv == 2) {
        float * h0 = hist;
        float * h1 = hist + nc;
        size_t e = 0;
        for (; e + 8 <= nc; e += 8) {
            const __m256 bv = _mm256_loadu_ps(b + e);
            const __m256 xv = _mm256_loadu_ps(x + e);
            const __m256 bxe = _mm256_mul_ps(bv, xv);
            const __m256 cv = _mm256_loadu_ps(c + e);
            const __m256 h0v = _mm256_loadu_ps(h0 + e);
            const __m256 h1v = _mm256_loadu_ps(h1 + e);

            const __m256 k0 = _mm256_setr_ps(kern[(e+0)*3],   kern[(e+1)*3],   kern[(e+2)*3],   kern[(e+3)*3],
                                             kern[(e+4)*3],   kern[(e+5)*3],   kern[(e+6)*3],   kern[(e+7)*3]);
            const __m256 k1 = _mm256_setr_ps(kern[(e+0)*3+1], kern[(e+1)*3+1], kern[(e+2)*3+1], kern[(e+3)*3+1],
                                             kern[(e+4)*3+1], kern[(e+5)*3+1], kern[(e+6)*3+1], kern[(e+7)*3+1]);
            const __m256 k2 = _mm256_setr_ps(kern[(e+0)*3+2], kern[(e+1)*3+2], kern[(e+2)*3+2], kern[(e+3)*3+2],
                                             kern[(e+4)*3+2], kern[(e+5)*3+2], kern[(e+6)*3+2], kern[(e+7)*3+2]);

            __m256 acc = _mm256_mul_ps(k2, bxe);
            acc = _mm256_fmadd_ps(k1, h1v, acc);
            acc = _mm256_fmadd_ps(k0, h0v, acc);
            _mm256_storeu_ps(out + e, _mm256_mul_ps(cv, acc));
            _mm256_storeu_ps(h0 + e, h1v);
            _mm256_storeu_ps(h1 + e, bxe);
        }
        for (; e < nc; ++e) {
            const float bxe = b[e] * x[e];
            const float s = kern[e * 3 + 0] * h0[e] + kern[e * 3 + 1] * h1[e] + kern[e * 3 + 2] * bxe;
            out[e] = c[e] * s;
            h0[e] = h1[e];
            h1[e] = bxe;
        }
        return;
    }
    for (size_t e = 0; e < nc; ++e) {
        const float bxe = b[e] * x[e];
        float sum = kern[e * (d_conv + 1) + d_conv] * bxe;
        for (size_t j = 0; j < d_conv; ++j) {
            sum += kern[e * (d_conv + 1) + j] * hist[j * nc + e];
        }
        out[e] = c[e] * sum;
        for (size_t j = 0; j + 1 < d_conv; ++j) hist[j * nc + e] = hist[(j + 1) * nc + e];
        hist[(d_conv - 1) * nc + e] = bxe;
    }
}

} // namespace kata

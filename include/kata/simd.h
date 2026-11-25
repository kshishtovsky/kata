#pragma once
// Small SIMD helpers shared by the kernel files. Each is tagged with the same
// target string as the kernels that call it; without a matching target GCC
// refuses to inline ("target specific option mismatch").
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

namespace kata {

// Hardware half->float. The generic f16_to_f32 branches (and loops on the
// subnormal path), and real Q4_K weights hit that path often enough for it to
// cost 20% of the row time.
__attribute__((target("avx2,f16c,fma")))
inline float f16_fast(uint16_t h) { return _cvtsh_ss(h); }

// Byte mask selecting scale s into the low 64 bits and scale s+1 into the high
// 64 bits: the two halves of a maddubs result span two 16-value sub-blocks.
__attribute__((target("avx2")))
inline __m128i scale_pair_bytes(int s) {
    return _mm_set_epi64x((long long)(0x0101010101010101ull * (unsigned)(s + 1)),
                          (long long)(0x0101010101010101ull * (unsigned)s));
}

__attribute__((target("avx2,f16c,fma")))
inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, lo);
    return _mm_cvtss_f32(_mm_add_ss(lo, sh));
}

// Vectorised exp(x) via degree-5 minimax polynomial for 2^f on [-0.5, 0.5]
// and 2^k via IEEE-754 exponent field addition. Clamped to [-87, 88].
__attribute__((target("avx2,fma")))
inline __m256 exp256_ps(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0f)), _mm256_set1_ps(-87.0f));
    x = _mm256_mul_ps(x, log2e);
    const __m256 k = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 f = _mm256_sub_ps(x, k);
    __m256 poly = _mm256_set1_ps(1.3143408886e-04f);
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(1.3534167919e-03f));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(9.6181291076e-03f));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(5.5504108665e-02f));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(2.4022650696e-01f));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(6.9314718056e-01f));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(1.0f));
    const __m256i ki = _mm256_cvtps_epi32(k);
    const __m256 pow2k = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23));
    return _mm256_mul_ps(poly, pow2k);
}

// Vectorised exp(x) via Cody-Waite range reduction (1.45 ULP) for AVX-512 (16 floats/vector).
__attribute__((target("avx512f,avx512vl,avx512dq,fma")))
inline __m512 exp512_ps(__m512 x) {
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b =
        _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                         _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __mmask16 d =
        _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                        _mm512_set1_ps(0x1.573e2ep-5f)),
                        u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                        _mm512_set1_ps(0x1.fffdb6p-2f))),
        u,
        _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
    const __m512 res = _mm512_scalef_ps(j, n);
    if (_mm512_kortestz(d, d))
        return res;
    const __m512 zero = _mm512_setzero_ps();
    const __m512 alt = _mm512_mask_blend_ps(
        _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(__builtin_inff()), zero);
    return _mm512_mask_blend_ps(d, res, alt);
}

// Unpack the 12 packed q4_K scale bytes into utmp[0..3]: scales in the low byte
// of each word, mins in the high byte. Same bit shuffling as ggml's reference.
__attribute__((target("avx2")))
inline void unpack_k_scales(const uint8_t * scales, uint32_t * utmp) {
    const uint32_t kmask1 = 0x3f3f3f3fu, kmask2 = 0x0f0f0f0fu, kmask3 = 0x03030303u;
    std::memcpy(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

} // namespace kata
#endif
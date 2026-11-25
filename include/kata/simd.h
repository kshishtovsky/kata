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
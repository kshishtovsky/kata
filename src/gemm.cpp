// Batched matmul for prefill (EXP-0023, phase A).
//
// Why this shape: the decode kernels walk one activation row per pass, so a
// weight block is unpacked once per token. Here the weight block is unpacked
// once and reused across MR activation rows, which is the whole point of
// batching. Loop order is weight-row outer and activation-tile inner because the
// activation matrix (~1 MB for 512 tokens) stays in L2 while the weights do not:
// the weight matrix is then read exactly once per gemm.
#include "kata/gemm.h"
#include "kata/gemv.h"
#include "kata/quant.h"
#include "kata/simd.h"

#include <algorithm>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace kata {

void gemm_q4_K_q8_K_ref(size_t m, size_t n, size_t k, float * out, size_t ldc,
                        const block_q4_K * w, const block_q8_K * a) {
    const size_t nb = k / QK_K;
    for (size_t i = 0; i < m; ++i)
        for (size_t r = 0; r < n; ++r)
            out[i * ldc + r] = vec_dot_q4_K_q8_K_ref(k, w + r * nb, a + i * nb);
}

void gemm_q6_K_q8_K_ref(size_t m, size_t n, size_t k, float * out, size_t ldc,
                        const block_q6_K * w, const block_q8_K * a) {
    const size_t nb = k / QK_K;
    for (size_t i = 0; i < m; ++i)
        for (size_t r = 0; r < n; ++r)
            out[i * ldc + r] = vec_dot_q6_K_q8_K_ref(k, w + r * nb, a + i * nb);
}

#if defined(__x86_64__) || defined(__i386__)

namespace {

// Activation rows sharing one unpacked weight block. This is the single number
// that sets the cost: every weight byte is re-read m/MR times, so at m=512 a
// tile of 4 costs 128 passes over the weights (302 MB for one 2048x2048 layer)
// while a tile of 16 costs 32. The limit is the vector register file: MR float
// accumulators plus the unpacked weight block and temporaries must fit, which
// is why 16 needs AVX-512 (32 ymm available) and 4 is all AVX2 can do.
constexpr size_t MR = 8;

// Accumulator budget: MR*2 ymm for q4_K (value + min), MR for q6_K, plus the
// unpacked weight block. MR=4 keeps the whole set under 16 ymm.
#if defined(__AVX512F__) || defined(KATA_AVX512)
#define KATA_GEMM_TARGET "avx512f,avx512vl,avx512bw,avx2,f16c,fma"
#else
#define KATA_GEMM_TARGET "avx2,f16c,fma"
#endif

__attribute__((target(KATA_GEMM_TARGET)))
// `ap` is the activation tile packed block-major: row i of block b lives at
// ap[b*MR + i]. The unpacked layout (a[i*nb + b]) puts each activation row
// nb*292 bytes from the next, so the inner loop touches MR different pages per
// block and the hardware prefetcher cannot follow any of them. Packing costs
// one pass over the chunk and makes every inner-loop load contiguous.
void gemm_q4_K_row_mr(const block_q4_K * wr, const block_q8_K * ap, size_t nb,
                      size_t avail, float * out, size_t ldc) {
    // Two accumulator sets, not three: the per-block integer sum is folded into
    // the float accumulator before the next block starts, so the min term rides
    // on the same register. Three arrays (value, min, integer) is 3*MR ymm and
    // spills the whole file past MR=8.
    __m256 accf[MR];
    for (size_t i = 0; i < MR; ++i) accf[i] = _mm256_setzero_ps();

    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m128i m2 = _mm_set1_epi16(2);

    for (size_t b = 0; b < nb; ++b) {
        const block_q4_K & wb = wr[b];
        uint32_t utmp[4];
        unpack_k_scales(wb.scales, utmp);
        const __m128i utmps = _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]);
        const __m128i sc128 = _mm_cvtepu8_epi16(utmps);
        const __m256i sclab = _mm256_set_m128i(sc128, sc128);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        float dscale[MR], dmin_term[MR];
        for (size_t i = 0; i < avail; ++i) {
            const block_q8_K & ab = ap[b * MR + i];
            const float xd = ab.d;
            dscale[i] = xd * f16_fast(wb.d);
            dmin_term[i] = -xd * f16_fast(wb.dmin);
        }

        __m256i sumi[MR];
        for (size_t i = 0; i < MR; ++i) sumi[i] = _mm256_setzero_si256();

        __m128i shuffle = _mm_set1_epi16(0x0100);
        const __m256i * q4 = (const __m256i *)wb.qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            // Weight side, once per j: reused by every activation row.
            const __m256i q4bits = _mm256_loadu_si256(q4 + j);
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            const __m256i scale_l = _mm256_shuffle_epi8(sclab, _mm256_set_m128i(shuffle, shuffle));
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m256i scale_h = _mm256_shuffle_epi8(sclab, _mm256_set_m128i(shuffle, shuffle));
            shuffle = _mm_add_epi16(shuffle, m2);

            for (size_t i = 0; i < avail; ++i) {
                const __m256i * q8 = (const __m256i *)(ap[b * MR + i].qs);
                __m256i p = _mm256_maddubs_epi16(q4l, _mm256_loadu_si256(q8 + 2 * j));
                sumi[i] = _mm256_add_epi32(sumi[i], _mm256_madd_epi16(scale_l, p));
                p = _mm256_maddubs_epi16(q4h, _mm256_loadu_si256(q8 + 2 * j + 1));
                sumi[i] = _mm256_add_epi32(sumi[i], _mm256_madd_epi16(scale_h, p));
            }
        }

        for (size_t i = 0; i < avail; ++i) {
            const block_q8_K & ab = ap[b * MR + i];
            // bsums are per-32-value sub-block sums, so the min term is a
            // 4-lane dot. It rides in the same ymm accumulator with its upper
            // half zeroed: one register per row instead of a second array.
            const __m128i q8s = _mm_hadd_epi16(_mm_loadu_si128((const __m128i *)&ab.bsums[0]),
                                               _mm_loadu_si128((const __m128i *)&ab.bsums[8]));
            const __m128 acc_m = _mm_mul_ps(_mm_set1_ps(dmin_term[i]),
                                            _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s)));
            accf[i] = _mm256_fmadd_ps(_mm256_set1_ps(dscale[i]), _mm256_cvtepi32_ps(sumi[i]), accf[i]);
            accf[i] = _mm256_add_ps(accf[i], _mm256_set_m128(_mm_setzero_ps(), acc_m));
        }
    }

    for (size_t i = 0; i < avail; ++i) out[i * ldc] = hsum_ps(accf[i]);
}

__attribute__((target(KATA_GEMM_TARGET)))
void gemm_q6_K_row_mr(const block_q6_K * wr, const block_q8_K * ap, size_t nb,
                      size_t avail, float * out, size_t ldc) {
    __m256 accf[MR];
    for (size_t i = 0; i < MR; ++i) accf[i] = _mm256_setzero_ps();

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i mc0 = _mm256_set1_epi8((char)0xc0);

    for (size_t b = 0; b < nb; ++b) {
        const block_q6_K & wb = wr[b];
        const __m128i scales128 = _mm_loadu_si128((const __m128i *)wb.scales);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales128);

        float dv[MR];
        for (size_t i = 0; i < avail; ++i) dv[i] = ap[b * MR + i].d * f16_fast(wb.d);

        __m256i sumi[MR];
        for (size_t i = 0; i < MR; ++i) sumi[i] = _mm256_setzero_si256();

        const uint8_t * ql = wb.ql;
        const uint8_t * qh = wb.qh;
        int is = 0;
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)ql);
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            const __m256i qhbits = _mm256_loadu_si256((const __m256i *)qh);

            const __m256i h0 = _mm256_slli_epi16(_mm256_and_si256(qhbits, m3), 4);
            const __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(qhbits, m12), 2);
            const __m256i h2 = _mm256_and_si256(qhbits, m48);
            const __m256i h3 = _mm256_srli_epi16(_mm256_and_si256(qhbits, mc0), 2);

            const __m256i q0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), h0);
            const __m256i q1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), h1);
            const __m256i q2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), h2);
            const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), h3);

            // One maddubs spans two 16-value sub-blocks, so its halves need
            // scales s and s+1; a duplicated scale misweights half the block.
            const __m256i sc0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales128, scale_pair_bytes(is + 0)));
            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales128, scale_pair_bytes(is + 2)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales128, scale_pair_bytes(is + 4)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales128, scale_pair_bytes(is + 6)));
            is += 8;

            for (size_t i = 0; i < avail; ++i) {
                const __m256i * q8 = (const __m256i *)(ap[b * MR + i].qs);
                __m256i p0 = _mm256_madd_epi16(sc0, _mm256_maddubs_epi16(q0, _mm256_loadu_si256(q8 + 4 * j + 0)));
                __m256i p1 = _mm256_madd_epi16(sc1, _mm256_maddubs_epi16(q1, _mm256_loadu_si256(q8 + 4 * j + 1)));
                __m256i p2 = _mm256_madd_epi16(sc2, _mm256_maddubs_epi16(q2, _mm256_loadu_si256(q8 + 4 * j + 2)));
                __m256i p3 = _mm256_madd_epi16(sc3, _mm256_maddubs_epi16(q3, _mm256_loadu_si256(q8 + 4 * j + 3)));
                sumi[i] = _mm256_add_epi32(sumi[i], _mm256_add_epi32(_mm256_add_epi32(p0, p1),
                                                                    _mm256_add_epi32(p2, p3)));
            }

            ql += 64;
            qh += 32;
        }

        for (size_t i = 0; i < avail; ++i) {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)ap[b * MR + i].bsums);
            const __m256i s = _mm256_sub_epi32(sumi[i],
                                               _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), 5));
            accf[i] = _mm256_fmadd_ps(_mm256_set1_ps(dv[i]), _mm256_cvtepi32_ps(s), accf[i]);
        }
    }

    for (size_t i = 0; i < avail; ++i) out[i * ldc] = hsum_ps(accf[i]);
}

__attribute__((target(KATA_GEMM_TARGET)))
void gemm_q4_K_q8_K_avx2(size_t m, size_t n, size_t k, float * out, size_t ldc,
                         const block_q4_K * w, const block_q8_K * a) {
    const size_t nb = k / QK_K;
    std::vector<block_q8_K> packed(MR * nb);
    for (size_t i = 0; i < m; i += MR) {
        const size_t avail = std::min(MR, m - i);
        for (size_t b = 0; b < nb; ++b)
            for (size_t j = 0; j < avail; ++j)
                packed[b * MR + j] = a[(i + j) * nb + b];
        for (size_t r = 0; r < n; ++r)
            gemm_q4_K_row_mr(w + r * nb, packed.data(), nb, avail, out + i * ldc + r, ldc);
    }
}

__attribute__((target(KATA_GEMM_TARGET)))
void gemm_q6_K_q8_K_avx2(size_t m, size_t n, size_t k, float * out, size_t ldc,
                         const block_q6_K * w, const block_q8_K * a) {
    const size_t nb = k / QK_K;
    std::vector<block_q8_K> packed(MR * nb);
    for (size_t i = 0; i < m; i += MR) {
        const size_t avail = std::min(MR, m - i);
        for (size_t b = 0; b < nb; ++b)
            for (size_t j = 0; j < avail; ++j)
                packed[b * MR + j] = a[(i + j) * nb + b];
        for (size_t r = 0; r < n; ++r)
            gemm_q6_K_row_mr(w + r * nb, packed.data(), nb, avail, out + i * ldc + r, ldc);
    }
}

} // namespace

void gemm_q4_K_q8_K(size_t m, size_t n, size_t k, float * out, size_t ldc,
                    const block_q4_K * w, const block_q8_K * a) {
    if (m == 0 || n == 0) return;
    gemm_q4_K_q8_K_avx2(m, n, k, out, ldc, w, a);
}

void gemm_q6_K_q8_K(size_t m, size_t n, size_t k, float * out, size_t ldc,
                    const block_q6_K * w, const block_q8_K * a) {
    if (m == 0 || n == 0) return;
    gemm_q6_K_q8_K_avx2(m, n, k, out, ldc, w, a);
}

#else

void gemm_q4_K_q8_K(size_t m, size_t n, size_t k, float * out, size_t ldc,
                    const block_q4_K * w, const block_q8_K * a) {
    gemm_q4_K_q8_K_ref(m, n, k, out, ldc, w, a);
}

void gemm_q6_K_q8_K(size_t m, size_t n, size_t k, float * out, size_t ldc,
                    const block_q6_K * w, const block_q8_K * a) {
    gemm_q6_K_q8_K_ref(m, n, k, out, ldc, w, a);
}

#endif

// Rows of `n` are split into one contiguous block per worker, so each worker
// streams a single region of the weight matrix and its output columns are
// contiguous per row — `out + lo` with the same ldc is exactly that window.
void gemm_q4_K_q8_K_mt(size_t m, size_t n, size_t k, float * out, size_t ldc,
                       const block_q4_K * w, const block_q8_K * a, ThreadPool & pool) {
    const unsigned nt = pool.size();
    if (nt <= 1 || n < nt) { gemm_q4_K_q8_K(m, n, k, out, ldc, w, a); return; }
    const size_t nb = k / QK_K;
    const size_t chunk = (n + nt - 1) / nt;
    pool.run([&](unsigned tid) {
        const size_t lo = std::min(n, (size_t)tid * chunk);
        const size_t hi = std::min(n, lo + chunk);
        if (lo < hi) gemm_q4_K_q8_K(m, hi - lo, k, out + lo, ldc, w + lo * nb, a);
    });
}

void gemm_q6_K_q8_K_mt(size_t m, size_t n, size_t k, float * out, size_t ldc,
                       const block_q6_K * w, const block_q8_K * a, ThreadPool & pool) {
    const unsigned nt = pool.size();
    if (nt <= 1 || n < nt) { gemm_q6_K_q8_K(m, n, k, out, ldc, w, a); return; }
    const size_t nb = k / QK_K;
    const size_t chunk = (n + nt - 1) / nt;
    pool.run([&](unsigned tid) {
        const size_t lo = std::min(n, (size_t)tid * chunk);
        const size_t hi = std::min(n, lo + chunk);
        if (lo < hi) gemm_q6_K_q8_K(m, hi - lo, k, out + lo, ldc, w + lo * nb, a);
    });
}

} // namespace kata
// Batched matmul for prefill (EXP-0023, phase A).
//
// Why this shape: the decode kernels walk one activation row per pass, so a
// weight block is unpacked once per token. Here the weight block is unpacked
// once and reused across MR activation rows, which is the whole point of
// batching. Loop order is weight-row outer and activation-tile inner because the
// activation matrix (~1 MB for 512 tokens) stays in L2 while the weights do not:
// the weight matrix is then read exactly once per gemm.
#include "kata/cpu.h"
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
constexpr size_t MR = 16;

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

// ---------------------------------------------------------------------------
// VNNI microkernel, 8 weight rows x 16 tokens (EXP-0029).
//
// The AVX2 path above spends 2 int ops per (weight block, token) on the value
// term and re-shuffles scales per token. This one packs the activation tile so
// 16 tokens' q8 codes sit in one zmm, which lets vpdpbusd do 64 MACs per
// instruction for all 8 rows at once: measured 160 vs 57 GMAC/s single thread.
//
// Two details decide whether it stays in registers, and both are load bearing:
//  - every loop over the 8 rows carries `#pragma GCC unroll 8`. With a runtime
//    row index GCC keeps the accumulators in memory and emits
//    load/vpdpbusd/store, which halves throughput.
//  - the min term is summed in its own pass over sub-blocks. Folding it into
//    the dpbusd loop costs 8 more live zmm and spills.
constexpr size_t VTN = 16; // tokens per tile
constexpr size_t VRN = 8;  // weight rows per band

struct alignas(64) VaTile {
    int32_t codes[QK_K / 4][VTN]; // 4 int8 codes per u32, one u32 per 4 k values
    int32_t bsum[QK_K / 32][VTN];
    float d[VTN];
};

struct alignas(64) VwBand {
    uint32_t g[VRN][QK_K / 4]; // 4 nibbles widened to bytes, one k value per byte
    int32_t sc[VRN][QK_K / 32];
    int32_t mn[VRN][QK_K / 32];
    float d[VRN], dmin[VRN];
};

void pack_a_tiles(const block_q8_K * a, size_t m, size_t nb, std::vector<VaTile> & tiles) {
    const size_t ntile = (m + VTN - 1) / VTN;
    tiles.assign(ntile * nb, VaTile{});
    for (size_t jt = 0; jt < ntile; ++jt) {
        const size_t nt = std::min(VTN, m - jt * VTN);
        for (size_t b = 0; b < nb; ++b) {
            VaTile & t = tiles[jt * nb + b];
            for (size_t j = 0; j < nt; ++j) {
                const block_q8_K & ab = a[(jt * VTN + j) * nb + b];
                t.d[j] = ab.d;
                for (int s = 0; s < QK_K / 32; ++s)
                    t.bsum[s][j] = ab.bsums[2 * s] + ab.bsums[2 * s + 1];
                for (int g = 0; g < QK_K / 4; ++g) {
                    int32_t v;
                    std::memcpy(&v, &ab.qs[g * 4], 4);
                    t.codes[g][j] = v;
                }
            }
        }
    }
}

__attribute__((target("avx2")))
void unpack_w_band(const block_q4_K * w, size_t nb, size_t nr, VwBand & o) {
    for (size_t r = 0; r < VRN; ++r) {
        if (r >= nr) {
            std::memset(o.g[r], 0, sizeof(o.g[r]));
            std::memset(o.sc[r], 0, sizeof(o.sc[r]));
            std::memset(o.mn[r], 0, sizeof(o.mn[r]));
            o.d[r] = o.dmin[r] = 0.0f;
            continue;
        }
        const block_q4_K & wr = w[r * nb];
        o.d[r] = f16_fast(wr.d);
        o.dmin[r] = f16_fast(wr.dmin);
        for (int s = 0; s < QK_K / 32; ++s) {
            uint8_t sc8 = 0, mn8 = 0;
            q4_K_scale_min(s, wr.scales, &sc8, &mn8);
            o.sc[r][s] = sc8;
            o.mn[r][s] = mn8;
        }
        // Sub-block pair (s, s+1) shares qs[(s/2)*32 .. +32]: the low nibbles are
        // sub-block s's 32 values, the high nibbles sub-block s+1's. Both become
        // one byte per k value, which is the form vpdpbusd wants. Done a whole
        // sub-block at a time: byte-at-a-time was 512 scalar ops per block and
        // showed up as 43% of prefill time in the profile.
        for (int s = 0; s < QK_K / 32; ++s) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(wr.qs + (s / 2) * 32));
            if (s % 2) v = _mm256_srli_epi16(v, 4);
            v = _mm256_and_si256(v, _mm256_set1_epi8(0x0f));
            _mm256_store_si256((__m256i *)&o.g[r][s * 8], v);
        }
    }
}

__attribute__((target("avx512f,avx512vl,avx512bw,avx512vnni,avx512dq,avx2,f16c,fma")))
void q4_K_vnni_tile(const VwBand * uw, const VaTile * tiles, size_t nb, float * out,
                    size_t ldc, size_t nr, size_t nt) {
    __m512 acc[VRN];
#pragma GCC unroll 8
    for (size_t r = 0; r < VRN; ++r) acc[r] = _mm512_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        const VaTile & tb = tiles[b];
        const VwBand & ub = uw[b];
        __m512i s1[VRN];
#pragma GCC unroll 8
        for (size_t r = 0; r < VRN; ++r) s1[r] = _mm512_setzero_si512();

        for (int s = 0; s < QK_K / 32; ++s) {
            __m512i a16[VRN];
#pragma GCC unroll 8
            for (size_t r = 0; r < VRN; ++r) a16[r] = _mm512_setzero_si512();
#pragma GCC unroll 8
            for (int gg = 0; gg < 8; ++gg) {
                const __m512i codes = _mm512_load_si512((const __m512i *)tb.codes[s * 8 + gg]);
#pragma GCC unroll 8
                for (size_t r = 0; r < VRN; ++r)
                    a16[r] = _mm512_dpbusd_epi32(a16[r], _mm512_set1_epi32((int)ub.g[r][s * 8 + gg]), codes);
            }
#pragma GCC unroll 8
            for (size_t r = 0; r < VRN; ++r)
                s1[r] = _mm512_add_epi32(s1[r], _mm512_mullo_epi32(a16[r], _mm512_set1_epi32(ub.sc[r][s])));
        }

        __m512i s2[VRN];
#pragma GCC unroll 8
        for (size_t r = 0; r < VRN; ++r) s2[r] = _mm512_setzero_si512();
        for (int s = 0; s < QK_K / 32; ++s) {
            const __m512i bsv = _mm512_load_si512((const __m512i *)tb.bsum[s]);
#pragma GCC unroll 8
            for (size_t r = 0; r < VRN; ++r)
                s2[r] = _mm512_add_epi32(s2[r], _mm512_mullo_epi32(bsv, _mm512_set1_epi32(ub.mn[r][s])));
        }

        const __m512 dact = _mm512_load_ps(tb.d);
#pragma GCC unroll 8
        for (size_t r = 0; r < VRN; ++r) {
            __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(s1[r]), _mm512_set1_ps(ub.d[r]));
            f = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(s2[r]), _mm512_set1_ps(ub.dmin[r]), f);
            acc[r] = _mm512_fmadd_ps(f, dact, acc[r]);
        }
    }

    for (size_t r = 0; r < nr; ++r) {
        alignas(64) float tmp[VTN];
        _mm512_store_ps(tmp, acc[r]);
        for (size_t j = 0; j < nt; ++j) out[j * ldc + r] = tmp[j];
    }
}

__attribute__((target("avx512f,avx512vl,avx512bw,avx512vnni,avx512dq,avx2,f16c,fma")))
void q4_K_vnni_rows(const VaTile * tiles, size_t ntile, size_t m, size_t nb, size_t n,
                    const block_q4_K * w, float * out, size_t ldc) {
    std::vector<VwBand> uw(nb);
    for (size_t rw = 0; rw < n; rw += VRN) {
        const size_t nr = std::min(VRN, n - rw);
        for (size_t b = 0; b < nb; ++b) unpack_w_band(w + rw * nb + b, nb, nr, uw[b]);
        // Band outer, token tile inner: the band is unpacked once and reused by
        // every tile. With the loops the other way round it was re-unpacked per
        // tile, i.e. m/VTN times, which showed up as 10% of prefill.
        for (size_t jt = 0; jt < ntile; ++jt) {
            const size_t nt = std::min(VTN, m - jt * VTN);
            q4_K_vnni_tile(uw.data(), tiles + jt * nb, nb, out + (jt * VTN) * ldc + rw, ldc, nr, nt);
        }
    }
}

} // namespace

void gemm_q4_K_q8_K(size_t m, size_t n, size_t k, float * out, size_t ldc,
                    const block_q4_K * w, const block_q8_K * a) {
    if (m == 0 || n == 0) return;
    if (cpu_isa() == Isa::Avx512Vnni && k % QK_K == 0) {
        const size_t nb = k / QK_K;
        std::vector<VaTile> tiles;
        pack_a_tiles(a, m, nb, tiles);
        q4_K_vnni_rows(tiles.data(), (m + VTN - 1) / VTN, m, nb, n, w, out, ldc);
        return;
    }
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


void gemm_f32(size_t m, size_t n, size_t k, float * out, size_t ldc,
              const float * w, const float * a) {
    for (size_t i = 0; i < m; ++i) {
        const float * ai = a + i * k;
        for (size_t r = 0; r < n; ++r) {
            const float * wr = w + r * k;
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            size_t c = 0;
            for (; c + 4 <= k; c += 4) {
                s0 += wr[c] * ai[c];
                s1 += wr[c + 1] * ai[c + 1];
                s2 += wr[c + 2] * ai[c + 2];
                s3 += wr[c + 3] * ai[c + 3];
            }
            for (; c < k; ++c) s0 += wr[c] * ai[c];
            out[i * ldc + r] = (s0 + s1) + (s2 + s3);
        }
    }
}

void gemm_f32_mt(size_t m, size_t n, size_t k, float * out, size_t ldc,
                 const float * w, const float * a, ThreadPool & pool) {
    const unsigned nt = pool.size();
    if (nt <= 1 || n < nt) { gemm_f32(m, n, k, out, ldc, w, a); return; }
    const size_t chunk = (n + nt - 1) / nt;
    pool.run([&](unsigned tid) {
        const size_t lo = std::min(n, (size_t)tid * chunk);
        const size_t hi = std::min(n, lo + chunk);
        if (lo < hi) gemm_f32(m, hi - lo, k, out + lo, ldc, w + lo * k, a);
    });
}

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
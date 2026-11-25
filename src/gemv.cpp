#include "kata/gemv.h"
#include "kata/simd.h"

#include <map>
#include <memory>
#include <mutex>
#include "kata/cpu.h"
#include "kata/quant.h"

#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace kata {

void gemv_q4_0_q8_0_ref(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x) {
    const size_t nb = cols / QK;
    for (size_t r = 0; r < rows; ++r) {
        out[r] = vec_dot_q4_0_q8_0_ref(cols, w + r * nb, x);
    }
}

#if defined(__x86_64__) || defined(__i386__)

namespace {

// 32 nibbles in block_q4_0 order, still unsigned 0..15.
__attribute__((target("avx2")))
inline __m256i nibbles_q4_avx2(const uint8_t * qs) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)qs);
    const __m256i both = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    return _mm256_and_si256(_mm256_set1_epi8(0x0f), both);
}

// Signed -8..7, the value the format actually encodes.
__attribute__((target("avx2")))
inline __m256i expand_q4_avx2(const uint8_t * qs) {
    return _mm256_sub_epi8(nibbles_q4_avx2(qs), _mm256_set1_epi8(8));
}

__attribute__((target("avx2")))
inline __m256 dot_i8_i8_avx2(__m256i qx, __m256i qy) {
    const __m256i ax = _mm256_sign_epi8(qx, qx);
    const __m256i sy = _mm256_sign_epi8(qy, qx);
    const __m256i pairs = _mm256_maddubs_epi16(ax, sy);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), pairs));
}

// Hardware half->float. The generic f16_to_f32 branches (and loops on the
// subnormal path), and real Q4_K weights hit that path often enough for it to
// cost 20% of the row time.
__attribute__((target("avx2,f16c,fma")))
inline float f16_fast(uint16_t h) {
    return _cvtsh_ss(h);
}

// Byte mask selecting scale s into the low 64 bits and scale s+1 into the high
// 64 bits: the two halves of a maddubs result, which span two sub-blocks.
__attribute__((target("avx2")))
inline __m128i scale_pair(int s) {
    return _mm_set_epi64x((long long)(0x0101010101010101ull * (unsigned)(s + 1)),
                          (long long)(0x0101010101010101ull * (unsigned)s));
}

__attribute__((target("avx2,f16c,fma")))
inline float hsum_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, lo);
    return _mm_cvtss_f32(_mm_add_ss(lo, sh));
}

__attribute__((target("avx2,f16c,fma")))
inline float scale_pair(uint16_t dx, uint16_t dy) {
    return _cvtsh_ss(dx) * _cvtsh_ss(dy);
}

} // namespace

__attribute__((target("avx2,f16c,fma")))
void gemv_q4_0_q8_0_avx2(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x) {
    const size_t nb = cols / QK;
    size_t r = 0;

    // Two rows at a time: the activation block is loaded once and feeds two
    // independent accumulator chains, which keeps the FP ports busy while the
    // weights stream in.
    for (; r + 1 < rows; r += 2) {
        const block_q4_0 * w0 = w + r * nb;
        const block_q4_0 * w1 = w0 + nb;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        for (size_t b = 0; b < nb; ++b) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *)x[b].qs);
            const float d0 = scale_pair(w0[b].d, x[b].d);
            const float d1 = scale_pair(w1[b].d, x[b].d);
            const __m256 q0 = dot_i8_i8_avx2(expand_q4_avx2(w0[b].qs), qy);
            const __m256 q1 = dot_i8_i8_avx2(expand_q4_avx2(w1[b].qs), qy);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), q0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), q1, acc1);
        }
        out[r]     = hsum_avx2(acc0);
        out[r + 1] = hsum_avx2(acc1);
    }

    if (r < rows) {
        const block_q4_0 * w0 = w + r * nb;
        __m256 acc = _mm256_setzero_ps();
        for (size_t b = 0; b < nb; ++b) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *)x[b].qs);
            const float d = scale_pair(w0[b].d, x[b].d);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), dot_i8_i8_avx2(expand_q4_avx2(w0[b].qs), qy), acc);
        }
        out[r] = hsum_avx2(acc);
    }
}

__attribute__((target("avx512f,avx512vl,avx512bw,avx512vnni,f16c,fma")))
void gemv_q4_0_q8_0_avx512(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x) {
    const size_t nb = cols / QK;
    size_t r = 0;

    // Four rows at a time so one activation load feeds four dot products.
    for (; r + 3 < rows; r += 4) {
        const block_q4_0 * w0 = w + r * nb;
        const block_q4_0 * w1 = w0 + nb;
        const block_q4_0 * w2 = w1 + nb;
        const block_q4_0 * w3 = w2 + nb;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const __m256i zero = _mm256_setzero_si256();

        for (size_t b = 0; b < nb; ++b) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *)x[b].qs);
            const float dy = _cvtsh_ss(x[b].d);

            const __m256i q0 = nibbles_q4_avx2(w0[b].qs);
            const __m256i q1 = nibbles_q4_avx2(w1[b].qs);
            const __m256i q2 = nibbles_q4_avx2(w2[b].qs);
            const __m256i q3 = nibbles_q4_avx2(w3[b].qs);

            // VNNI: unsigned 0..15 weight nibbles times signed activations.
            const __m256i p0 = _mm256_dpbusd_epi32(zero, q0, qy);
            const __m256i p1 = _mm256_dpbusd_epi32(zero, q1, qy);
            const __m256i p2 = _mm256_dpbusd_epi32(zero, q2, qy);
            const __m256i p3 = _mm256_dpbusd_epi32(zero, q3, qy);
            // The stored nibble is value+8, so sum((u-8)*y) = sum(u*y) - 8*sum(y).
            const __m256i sy = _mm256_madd_epi16(_mm256_set1_epi16(1),
                                                 _mm256_maddubs_epi16(_mm256_set1_epi8(1), qy));
            const __m256i corr = _mm256_slli_epi32(sy, 3);

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w0[b].d) * dy), _mm256_cvtepi32_ps(_mm256_sub_epi32(p0, corr)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w1[b].d) * dy), _mm256_cvtepi32_ps(_mm256_sub_epi32(p1, corr)), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w2[b].d) * dy), _mm256_cvtepi32_ps(_mm256_sub_epi32(p2, corr)), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w3[b].d) * dy), _mm256_cvtepi32_ps(_mm256_sub_epi32(p3, corr)), acc3);
        }
        out[r]     = hsum_avx2(acc0);
        out[r + 1] = hsum_avx2(acc1);
        out[r + 2] = hsum_avx2(acc2);
        out[r + 3] = hsum_avx2(acc3);
    }

    for (; r < rows; ++r) {
        const block_q4_0 * w0 = w + r * nb;
        __m256 acc = _mm256_setzero_ps();
        const __m256i zero = _mm256_setzero_si256();
        for (size_t b = 0; b < nb; ++b) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *)x[b].qs);
            const __m256i q = nibbles_q4_avx2(w0[b].qs);
            const __m256i p = _mm256_dpbusd_epi32(zero, q, qy);
            const __m256i sy = _mm256_madd_epi16(_mm256_set1_epi16(1),
                                                 _mm256_maddubs_epi16(_mm256_set1_epi8(1), qy));
            const __m256i corr = _mm256_slli_epi32(sy, 3);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w0[b].d) * _cvtsh_ss(x[b].d)),
                                  _mm256_cvtepi32_ps(_mm256_sub_epi32(p, corr)), acc);
        }
        out[r] = hsum_avx2(acc);
    }
}

__attribute__((target("avx512f,avx512vl,avx512bw,f16c,fma")))
void gemv_q4_0_q8_0_avx512_r1(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x) {
    const size_t nb = cols / QK;
    const __m256i off = _mm256_set1_epi8(8);
    for (size_t r = 0; r < rows; ++r) {
        const block_q4_0 * wr = w + r * nb;
        // One accumulator per 4 blocks: the chain is the dependency bottleneck
        // when a single thread has to keep multiple DRAM requests in flight.
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        size_t b = 0;
        for (; b + 3 < nb; b += 4) {
            const __m256i qy0 = _mm256_loadu_si256((const __m256i *)x[b + 0].qs);
            const __m256i qy1 = _mm256_loadu_si256((const __m256i *)x[b + 1].qs);
            const __m256i qy2 = _mm256_loadu_si256((const __m256i *)x[b + 2].qs);
            const __m256i qy3 = _mm256_loadu_si256((const __m256i *)x[b + 3].qs);
            const __m256i sx0 = _mm256_sub_epi8(nibbles_q4_avx2(wr[b + 0].qs), off);
            const __m256i sx1 = _mm256_sub_epi8(nibbles_q4_avx2(wr[b + 1].qs), off);
            const __m256i sx2 = _mm256_sub_epi8(nibbles_q4_avx2(wr[b + 2].qs), off);
            const __m256i sx3 = _mm256_sub_epi8(nibbles_q4_avx2(wr[b + 3].qs), off);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(wr[b + 0].d) * _cvtsh_ss(x[b + 0].d)),
                                   dot_i8_i8_avx2(sx0, qy0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(wr[b + 1].d) * _cvtsh_ss(x[b + 1].d)),
                                   dot_i8_i8_avx2(sx1, qy1), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(wr[b + 2].d) * _cvtsh_ss(x[b + 2].d)),
                                   dot_i8_i8_avx2(sx2, qy2), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(wr[b + 3].d) * _cvtsh_ss(x[b + 3].d)),
                                   dot_i8_i8_avx2(sx3, qy3), acc3);
        }
        for (; b < nb; ++b) {
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(wr[b].d) * _cvtsh_ss(x[b].d)),
                                   dot_i8_i8_avx2(_mm256_sub_epi8(nibbles_q4_avx2(wr[b].qs), off),
                                                  _mm256_loadu_si256((const __m256i *)x[b].qs)), acc0);
        }
        out[r] = hsum_avx2(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    }
}

#else
void gemv_q4_0_q8_0_avx512_r1(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x) {
    gemv_q4_0_q8_0_ref(rows, cols, out, w, x);
}
void gemv_q4_0_q8_0_avx2(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x) {
    gemv_q4_0_q8_0_ref(rows, cols, out, w, x);
}
void gemv_q4_0_q8_0_avx512(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x) {
    gemv_q4_0_q8_0_ref(rows, cols, out, w, x);
}
#endif

GemvFn gemv_q4_0_q8_0_best() {
    switch (cpu_isa()) {
        case Isa::Avx512Vnni: return gemv_q4_0_q8_0_avx512;
        case Isa::Avx2:       return gemv_q4_0_q8_0_avx2;
        default:              return gemv_q4_0_q8_0_ref;
    }
}

const char * gemv_best_name() {
    switch (cpu_isa()) {
        case Isa::Avx512Vnni: return "q4_0_q8_0/avx512-vnni";
        case Isa::Avx2:       return "q4_0_q8_0/avx2";
        default:              return "q4_0_q8_0/ref";
    }
}


void gemv_q4_0_q8_0_mt(size_t rows, size_t cols, float * out, const block_q4_0 * w,
                       const block_q8_0 * x, ThreadPool & pool) {
    const unsigned nt = pool.size();
    if (nt <= 1 || rows < nt * 2) { gemv_q4_0_q8_0_best()(rows, cols, out, w, x); return; }

    const size_t nb = cols / QK;
    const size_t chunk = (rows + nt - 1) / nt;
    pool.run([&](unsigned t) {
        const size_t lo = (size_t)t * chunk;
        const size_t hi = std::min(lo + chunk, rows);
        if (lo >= hi) return;
        gemv_q4_0_q8_0_best()(hi - lo, cols, out + lo, w + lo * nb, x);
    });
}

// Pools are cached per thread count so repeated calls do not respawn workers.
static ThreadPool & pool_for(unsigned n_threads) {
    static std::mutex mu;
    static std::map<unsigned, std::unique_ptr<ThreadPool>> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(n_threads);
    if (it == cache.end())
        it = cache.emplace(n_threads, std::make_unique<ThreadPool>(n_threads)).first;
    return *it->second;
}

void gemv_q4_0_q8_0_auto(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x, unsigned n_threads) {
    gemv_q4_0_q8_0_mt(rows, cols, out, w, x, pool_for(n_threads));
}

void gemv_q4_0_q8_0_auto(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x) {
    static ThreadPool pool(ThreadPool::default_threads());
    gemv_q4_0_q8_0_mt(rows, cols, out, w, x, pool);
}

// ------------------------------------------------------------- q4_K kernels

void gemv_q4_K_q8_K_mt_batch(size_t n_batch, size_t rows, size_t cols, float * out,
                             const block_q4_K * const * w, const block_q8_K * const * x,
                             ThreadPool & pool) {
    if (n_batch == 0 || rows == 0) return;
    const GemvKNfn fn = gemv_q4_K_q8_K_best();
    const unsigned nt = pool.size();
    const size_t total = n_batch * rows;
    const size_t chunk = (total + nt - 1) / nt;
    const size_t nb = cols / QK_K;
    pool.run([&](unsigned tid) {
        size_t r = (size_t)tid * chunk;
        const size_t rend = std::min(total, r + chunk);
        while (r < rend) {
            const size_t b = r / rows;
            const size_t lo = r - b * rows;
            const size_t hi = std::min(rows, lo + (rend - r));
            fn(hi - lo, cols, out + b * rows + lo, w[b] + lo * nb, x[b]);
            r += hi - lo;
        }
    });
}

void gemv_q4_K_q8_K_ref(size_t rows, size_t cols, float * out, const block_q4_K * w, const block_q8_K * x) {
    const size_t nb = cols / QK_K;
    for (size_t r = 0; r < rows; ++r) out[r] = vec_dot_q4_K_q8_K_ref(cols, w + r * nb, x);
}

#if defined(__x86_64__) || defined(__i386__)

namespace {

// Unpack the 12 packed scale bytes into utmp[0..3]: scales in the low byte of
// each word, mins in the high byte. Same bit shuffling as ggml's reference.
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

// One row per pass, sequential weight stream. The q4_K loop has no bias
// correction to fold, so there is nothing for extra rows to hide.
__attribute__((target("avx2,f16c,fma")))
void gemv_q4_K_q8_K_avx2_impl(size_t rows, size_t cols, float * out, const block_q4_K * w, const block_q8_K * x) {
    const size_t nb = cols / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m128i m2 = _mm_set1_epi16(2);

    for (size_t r = 0; r < rows; ++r) {
        const block_q4_K * wr = w + r * nb;
        __m256 acc = _mm256_setzero_ps();
        __m128 acc_m = _mm_setzero_ps();

        for (size_t b = 0; b < nb; ++b) {
            const float d = x[b].d * f16_fast(wr[b].d);
            const float dmin = -x[b].d * f16_fast(wr[b].dmin);

            uint32_t utmp[4];
            unpack_k_scales(wr[b].scales, utmp);
            const __m128i utmps = _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]);
            const __m128i sc128 = _mm_cvtepu8_epi16(utmps);
            const __m256i scales = _mm256_set_m128i(sc128, sc128);
            const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

            // min term: sum over sub-blocks of min[i] * sum(y in sub-block i),
            // both already 16-bit, so one madd handles all eight.
            const __m128i q8sums = _mm_loadu_si128((const __m128i *)x[b].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm_loadu_si128((const __m128i *)&x[b].bsums[0]),
                                              _mm_loadu_si128((const __m128i *)&x[b].bsums[8]));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            (void)q8sums;
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            const __m256i * q4 = (const __m256i *)wr[b].qs;
            const __m256i * q8 = (const __m256i *)x[b].qs;
            __m256i sumi = _mm256_setzero_si256();
            __m128i shuffle = _mm_set1_epi16(0x0100);

            for (int j = 0; j < QK_K / 64; ++j) {
                // value j in the low nibble, value j+32 in the high nibble
                const __m256i q4bits = _mm256_loadu_si256(q4 + j);
                const __m256i q4l = _mm256_and_si256(q4bits, m4);
                const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                const __m256i scale_l = _mm256_shuffle_epi8(scales, _mm256_set_m128i(shuffle, shuffle));
                shuffle = _mm_add_epi16(shuffle, m2);
                const __m256i scale_h = _mm256_shuffle_epi8(scales, _mm256_set_m128i(shuffle, shuffle));
                shuffle = _mm_add_epi16(shuffle, m2);

                __m256i p = _mm256_maddubs_epi16(q4l, _mm256_loadu_si256(q8 + 2 * j));
                p = _mm256_madd_epi16(scale_l, p);
                sumi = _mm256_add_epi32(sumi, p);

                p = _mm256_maddubs_epi16(q4h, _mm256_loadu_si256(q8 + 2 * j + 1));
                p = _mm256_madd_epi16(scale_h, p);
                sumi = _mm256_add_epi32(sumi, p);
            }

            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }

        acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
        out[r] = hsum_avx2(acc) + _mm_cvtss_f32(acc_m);
    }
}

// q6_K: values are (low4 | high2<<4) - 32, i.e. 0..63 biased by 32. The bias is
// folded out through bsums once per block (32 = 1<<5), so the hot loop stays all
// integer. Scales are signed bytes, one per 16 values.
__attribute__((target("avx2,f16c,fma")))
void gemv_q6_K_q8_K_avx2_impl(size_t rows, size_t cols, float * out, const block_q6_K * w, const block_q8_K * x) {
    const size_t nb = cols / QK_K;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i mc0 = _mm256_set1_epi8((char)0xc0);

    for (size_t r = 0; r < rows; ++r) {
        const block_q6_K * wr = w + r * nb;
        __m256 acc = _mm256_setzero_ps();

        for (size_t b = 0; b < nb; ++b) {
            const block_q6_K * wb = wr + b;
            const float d = x[b].d * f16_fast(wb->d);

            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)x[b].bsums);
            const __m128i scales128 = _mm_loadu_si128((const __m128i *)wb->scales);
            const __m256i scales16 = _mm256_cvtepi8_epi16(scales128);
            // 32 * sum(scales[i] * sum(q8 in sub-block i))
            const __m256i bias = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), 5);

            const uint8_t * ql = wb->ql;
            const uint8_t * qh = wb->qh;
            const __m256i * q8 = (const __m256i *)x[b].qs;
            __m256i sumi = _mm256_setzero_si256();
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

                __m256i p0 = _mm256_maddubs_epi16(q0, _mm256_loadu_si256(q8 + 4 * j + 0));
                __m256i p1 = _mm256_maddubs_epi16(q1, _mm256_loadu_si256(q8 + 4 * j + 1));
                __m256i p2 = _mm256_maddubs_epi16(q2, _mm256_loadu_si256(q8 + 4 * j + 2));
                __m256i p3 = _mm256_maddubs_epi16(q3, _mm256_loadu_si256(q8 + 4 * j + 3));

                // One maddubs spans 32 values, i.e. TWO 16-value sub-blocks: its
                // low 128-bit half needs scale s and its high half scale s+1.
                // A single duplicated scale here silently misweights half of
                // every block.
                p0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales128, scale_pair(is + 0))), p0);
                p1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales128, scale_pair(is + 2))), p1);
                p2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales128, scale_pair(is + 4))), p2);
                p3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales128, scale_pair(is + 6))), p3);
                is += 8;

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p2, p3));

                ql += 64;
                qh += 32;
            }

            sumi = _mm256_sub_epi32(sumi, bias);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }

        out[r] = hsum_avx2(acc);
    }
}

} // namespace

void gemv_q4_K_q8_K_avx2(size_t rows, size_t cols, float * out, const block_q4_K * w, const block_q8_K * x) {
    gemv_q4_K_q8_K_avx2_impl(rows, cols, out, w, x);
}

void gemv_q6_K_q8_K_avx2(size_t rows, size_t cols, float * out, const block_q6_K * w, const block_q8_K * x) {
    gemv_q6_K_q8_K_avx2_impl(rows, cols, out, w, x);
}

#else

void gemv_q4_K_q8_K_avx2(size_t rows, size_t cols, float * out, const block_q4_K * w, const block_q8_K * x) {
    gemv_q4_K_q8_K_ref(rows, cols, out, w, x);
}

void gemv_q6_K_q8_K_avx2(size_t rows, size_t cols, float * out, const block_q6_K * w, const block_q8_K * x) {
    const size_t nb = cols / QK_K;
    for (size_t r = 0; r < rows; ++r) out[r] = vec_dot_q6_K_q8_K_ref(cols, w + r * nb, x);
}

#endif

GemvK6fn gemv_q6_K_q8_K_best() {
    switch (cpu_isa()) {
        case Isa::Avx512Vnni:
        case Isa::Avx2:       return gemv_q6_K_q8_K_avx2;
        default:              return [](size_t rows, size_t cols, float * out, const block_q6_K * w, const block_q8_K * x) {
            const size_t nb = cols / QK_K;
            for (size_t r = 0; r < rows; ++r) out[r] = vec_dot_q6_K_q8_K_ref(cols, w + r * nb, x);
        };
    }
}

void gemv_q6_K_q8_K_mt(size_t rows, size_t cols, float * out, const block_q6_K * w, const block_q8_K * x, ThreadPool & pool) {
    const GemvK6fn fn = gemv_q6_K_q8_K_best();
    const unsigned nt = pool.size();
    const size_t chunk = (rows + nt - 1) / nt;
    pool.run([&](unsigned tid) {
        const size_t r0 = (size_t)tid * chunk;
        const size_t r1 = std::min(rows, r0 + chunk);
        if (r0 < r1) fn(r1 - r0, cols, out + r0, w + r0 * (cols / QK_K), x);
    });
}

void gemv_q6_K_q8_K_mt_batch(size_t n_batch, size_t rows, size_t cols, float * out,
                             const block_q6_K * const * w, const block_q8_K * const * x,
                             ThreadPool & pool) {
    if (n_batch == 0 || rows == 0) return;
    const GemvK6fn fn = gemv_q6_K_q8_K_best();
    const unsigned nt = pool.size();
    const size_t total = n_batch * rows;
    const size_t chunk = (total + nt - 1) / nt;
    const size_t nb = cols / QK_K;
    pool.run([&](unsigned tid) {
        size_t r = (size_t)tid * chunk;
        const size_t rend = std::min(total, r + chunk);
        while (r < rend) {
            const size_t b = r / rows;
            const size_t lo = r - b * rows;
            const size_t hi = std::min(rows, lo + (rend - r));
            fn(hi - lo, cols, out + b * rows + lo, w[b] + lo * nb, x[b]);
            r += hi - lo;
        }
    });
}

void gemv_q6_K_q8_K_ref(size_t rows, size_t cols, float * out, const block_q6_K * w, const block_q8_K * x) {
    const size_t nb = cols / QK_K;
    for (size_t r = 0; r < rows; ++r) out[r] = vec_dot_q6_K_q8_K_ref(cols, w + r * nb, x);
}

GemvKNfn gemv_q4_K_q8_K_best() {
    switch (cpu_isa()) {
        // AVX-512 VNNI q4_K measured 0.83x of this AVX2 loop: Zen4 splits every
        // 512-bit op in two, so dropping the byte shuffle did not pay for the
        // extra zmm inserts and 32-bit mullo. Revisit only on a real 512-bit core.
        case Isa::Avx512Vnni:
        case Isa::Avx2:       return gemv_q4_K_q8_K_avx2;
        default:              return gemv_q4_K_q8_K_ref;
    }
}

const char * gemv_K_best_name() {
    return cpu_isa() == Isa::Scalar ? "ref" : "avx2";
}

void gemv_q4_K_q8_K_mt(size_t rows, size_t cols, float * out, const block_q4_K * w, const block_q8_K * x, ThreadPool & pool) {
    const GemvKNfn fn = gemv_q4_K_q8_K_best();
    const unsigned nt = pool.size();
    const size_t chunk = (rows + nt - 1) / nt;
    pool.run([&](unsigned tid) {
        const size_t r0 = (size_t)tid * chunk;
        const size_t r1 = std::min(rows, r0 + chunk);
        if (r0 < r1) fn(r1 - r0, cols, out + r0, w + r0 * (cols / QK_K), x);
    });
}

void gemv_f32_ref(size_t rows, size_t cols, float * out, const float * w, const float * x) {
    for (size_t r = 0; r < rows; ++r) {
        const float * wr = w + r * cols;
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) sum += wr[c] * x[c];
        out[r] = sum;
    }
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2,fma")))
void gemv_f32_avx2(size_t rows, size_t cols, float * out, const float * w, const float * x) {
    size_t r = 0;
    for (; r + 4 <= rows; r += 4) {
        const float * w0 = w + (r + 0) * cols;
        const float * w1 = w + (r + 1) * cols;
        const float * w2 = w + (r + 2) * cols;
        const float * w3 = w + (r + 3) * cols;
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        size_t c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256 xv = _mm256_loadu_ps(x + c);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + c), xv, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + c), xv, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + c), xv, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + c), xv, a3);
        }
        out[r + 0] = hsum_ps(a0);
        out[r + 1] = hsum_ps(a1);
        out[r + 2] = hsum_ps(a2);
        out[r + 3] = hsum_ps(a3);
        for (; c < cols; ++c) {
            out[r + 0] += w0[c] * x[c];
            out[r + 1] += w1[c] * x[c];
            out[r + 2] += w2[c] * x[c];
            out[r + 3] += w3[c] * x[c];
        }
    }
    for (; r < rows; ++r) {
        const float * wr = w + r * cols;
        __m256 a0 = _mm256_setzero_ps();
        size_t c = 0;
        for (; c + 8 <= cols; c += 8) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(wr + c), _mm256_loadu_ps(x + c), a0);
        }
        float sum = hsum_ps(a0);
        for (; c < cols; ++c) sum += wr[c] * x[c];
        out[r] = sum;
    }
}
#endif

void gemv_f32(size_t rows, size_t cols, float * out, const float * w, const float * x) {
#if defined(__x86_64__) || defined(__i386__)
    if (cpu_isa() >= Isa::Avx2) {
        gemv_f32_avx2(rows, cols, out, w, x);
        return;
    }
#endif
    gemv_f32_ref(rows, cols, out, w, x);
}

} // namespace kata

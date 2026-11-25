#include "kata/gemv.h"

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

} // namespace kata

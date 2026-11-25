#pragma once
#include <cstddef>
#include "kata/types.h"
#include "kata/pool.h"

namespace kata {

// out[i] = dot(w + i*cols, x) for i in [0, rows), cols must be a multiple of QK.
using GemvFn = void (*)(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x);

void gemv_q4_0_q8_0_ref   (size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x);
void gemv_q4_0_q8_0_avx2  (size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x);
void gemv_q4_0_q8_0_avx512(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x);

// Best kernel the current CPU supports.
GemvFn gemv_q4_0_q8_0_best();
// Same math, one row per pass: used to test whether interleaving four weight
// streams hurts once the weights no longer fit in cache.
void gemv_q4_0_q8_0_avx512_r1(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x);
const char * gemv_best_name();

// ---- q4_K x q8_K -------------------------------------------------------
// The format the shipped models actually use. 144 bytes per 256 values: same
// density as q4_0 but scales are amortised over 8 sub-blocks, and the min term
// rides on bsums instead of a per-block bias correction in the hot loop.
using GemvKNfn = void (*)(size_t rows, size_t cols, float * out, const block_q4_K * w, const block_q8_K * x);

void gemv_q4_K_q8_K_ref   (size_t rows, size_t cols, float * out, const block_q4_K * w, const block_q8_K * x);
void gemv_q4_K_q8_K_avx2  (size_t rows, size_t cols, float * out, const block_q4_K * w, const block_q8_K * x);
GemvKNfn gemv_q4_K_q8_K_best();
const char * gemv_K_best_name();
void gemv_q4_K_q8_K_mt(size_t rows, size_t cols, float * out, const block_q4_K * w, const block_q8_K * x, ThreadPool & pool);

// ---- q6_K x q8_K -------------------------------------------------------
// 22.3% of LFM2.5-24B-A2B by bytes and every attn_v / output tensor in both
// shipped models, so no forward pass is possible without it.
using GemvK6fn = void (*)(size_t rows, size_t cols, float * out, const block_q6_K * w, const block_q8_K * x);

void gemv_q6_K_q8_K_ref   (size_t rows, size_t cols, float * out, const block_q6_K * w, const block_q8_K * x);
void gemv_q6_K_q8_K_avx2  (size_t rows, size_t cols, float * out, const block_q6_K * w, const block_q8_K * x);
GemvK6fn gemv_q6_K_q8_K_best();
void gemv_q6_K_q8_K_mt(size_t rows, size_t cols, float * out, const block_q6_K * w, const block_q8_K * x, ThreadPool & pool);

// Batched driver: `n_batch` independent matrices of `rows` each, each with its
// own weight base and its own activation row. The whole batch is split across
// workers by global row index, so one pool wake-up covers `n_batch` matvecs.
// Workers are the scarce resource here: a 1536-row call moves ~1.5 MB and a
// worker is still starting up when the slice ends.
void gemv_q4_K_q8_K_mt_batch(size_t n_batch, size_t rows, size_t cols, float * out,
                             const block_q4_K * const * w, const block_q8_K * const * x,
                             ThreadPool & pool);
void gemv_q6_K_q8_K_mt_batch(size_t n_batch, size_t rows, size_t cols, float * out,
                             const block_q6_K * const * w, const block_q8_K * const * x,
                             ThreadPool & pool);

// Multi-threaded driver. Rows are split into one contiguous block per worker so
// each worker streams a single region of the weight matrix (TLB friendly).
void gemv_q4_0_q8_0_mt(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x, ThreadPool & pool);

// Convenience: lazily creates one pool of default_threads().
void gemv_q4_0_q8_0_auto(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x);
void gemv_q4_0_q8_0_auto(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x, unsigned n_threads);

} // namespace kata

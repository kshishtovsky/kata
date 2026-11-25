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
const char * gemv_best_name();

// Multi-threaded driver. Rows are split into one contiguous block per worker so
// each worker streams a single region of the weight matrix (TLB friendly).
void gemv_q4_0_q8_0_mt(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x, ThreadPool & pool);

// Convenience: lazily creates one pool of default_threads().
void gemv_q4_0_q8_0_auto(size_t rows, size_t cols, float * out, const block_q4_0 * w, const block_q8_0 * x);

} // namespace kata

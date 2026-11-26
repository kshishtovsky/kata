#pragma once
#include <cstddef>
#include "kata/types.h"
#include "kata/pool.h"

namespace kata {

// Batched matmul for prefill, i.e. the same dot products as the GEMV kernels but
// with `m` activation rows sharing one pass over the weights.
//
//   w   : [n][k] row-major, quantised rows
//   a   : [m][k] row-major, one q8_K row per token
//   out : [m][ldc] row-major, ldc >= n
//
// The decode kernels re-read a weight row for every token, which is why they are
// bandwidth bound at m=1. Here each weight block is unpacked once and reused
// across MR activation rows, so the arithmetic is amortised instead.
void gemm_q4_K_q8_K(size_t m, size_t n, size_t k, float * out, size_t ldc,
                    const block_q4_K * w, const block_q8_K * a);
void gemm_q6_K_q8_K(size_t m, size_t n, size_t k, float * out, size_t ldc,
                    const block_q6_K * w, const block_q8_K * a);

// Scalar reference: one GEMV per activation row. The tests compare against this.
void gemm_q4_K_q8_K_ref(size_t m, size_t n, size_t k, float * out, size_t ldc,
                        const block_q4_K * w, const block_q8_K * a);
void gemm_q6_K_q8_K_ref(size_t m, size_t n, size_t k, float * out, size_t ldc,
                        const block_q6_K * w, const block_q8_K * a);

// Plain float weights, batched: the MoE router is stored as F32, so the `m`
// tokens of one prefill chunk need their own small matmul.
void gemm_f32(size_t m, size_t n, size_t k, float * out, size_t ldc,
              const float * w, const float * a);
void gemm_f32_mt(size_t m, size_t n, size_t k, float * out, size_t ldc,
                 const float * w, const float * a, ThreadPool & pool);

// Multi-threaded wrappers: rows of `n` are split into one contiguous block per
// worker, so each worker streams a single region of the weight matrix.
void gemm_q4_K_q8_K_mt(size_t m, size_t n, size_t k, float * out, size_t ldc,
                       const block_q4_K * w, const block_q8_K * a, ThreadPool & pool);
void gemm_q6_K_q8_K_mt(size_t m, size_t n, size_t k, float * out, size_t ldc,
                       const block_q6_K * w, const block_q8_K * a, ThreadPool & pool);

// Two q4_K matrices that share the SAME activation tile, in one fan-out and one
// pack. The MoE gate and up projections of one expert are exactly this: same
// aq_g, two weight matrices, two outputs. Two separate _mt calls packed the same
// tiles twice and woke the pool twice.
void gemm_q4_K_q8_K_mt2(size_t m, size_t n_a, size_t n_b, size_t k,
                        float * out_a, size_t lda, const block_q4_K * w_a,
                        float * out_b, size_t ldb, const block_q4_K * w_b,
                        const block_q8_K * a, ThreadPool & pool);

} // namespace kata
#pragma once

#include <cstddef>

namespace kata {

// Prefill attention: n_tok query rows at once. Row t has global position
// n_kv_begin + t and attends to every cached position 0 .. n_kv_begin + t, so
// the KV cache must already hold all n_tok new rows. One pass over the kv prefix
// per query row, online softmax, no score materialisation: at prefill depth the
// scores of one row are n_kv floats and writing them out costs more than the
// dot products that produced them.
void attention_prefill_ref(const float * q, size_t n_tok,
                           const float * k_cache, const float * v_cache,
                           size_t n_kv_begin, size_t n_heads, size_t n_kv_heads,
                           size_t head_dim, float scale, float * out);

void attention_prefill(const float * q, size_t n_tok,
                       const float * k_cache, const float * v_cache,
                       size_t n_kv_begin, size_t n_heads, size_t n_kv_heads,
                       size_t head_dim, float scale, float * out);

// Same, with the query rows split across threads. Every row is independent, so
// the split is exact: row t always reads the same kv prefix regardless of who
// computes it. This is the dominant prefill cost and it was single-threaded.
struct ThreadPool;
void attention_prefill_mt(const float * q, size_t n_tok,
                          const float * k_cache, const float * v_cache,
                          size_t n_kv_begin, size_t n_heads, size_t n_kv_heads,
                          size_t head_dim, float scale, float * out, ThreadPool & pool);

} // namespace kata

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kata {

// Decode attention: one query token attends over a KV cache of n_kv positions.
// Prefill (many tokens at once) is a different shape and gets its own kernel.
struct AttnArgs {
    const float * q;        // [n_heads * head_dim]
    const float * k_cache;  // [n_kv * n_kv_heads * head_dim], position-major
    const float * v_cache;  // same layout
    size_t n_kv;
    size_t n_heads;
    size_t n_kv_heads;      // n_heads / group_size; GQA shares each kv head across a group
    size_t head_dim;
    float scale;            // 1/sqrt(head_dim), passed in because some models differ
    float * out;            // [n_heads * head_dim]
    // Offset of this head's kv slice inside one position's row. Set by the
    // decode loop; callers of attention_decode leave it at 0.
    size_t base = 0;
    // Element stride between consecutive positions and between consecutive kv
    // heads. The default pair describes the position-major cache
    // [position][kv_head][head_dim]; the engine runs head-major
    // [kv_head][position][head_dim], which is what a decode worker streams
    // contiguously (EXP-0045: 45 -> 75 GB/s).
    size_t stride_pos = 0;   // 0 means n_kv_heads * head_dim
    size_t stride_head = 0;  // 0 means head_dim
    // Optionally int8 v cache (EXP-0047). Null keeps the float32 path. When set,
    // v_q holds the same layout as v_cache but as int8 with one f16 scale per
    // VQ_BLK values, so the v pass reads a quarter of the bytes -- it is a pure
    // weighted sum, so this is the one pass where less bytes is directly faster.
    const int8_t * v_q = nullptr;
    const uint16_t * v_scale = nullptr;
};
inline constexpr size_t VQ_BLK = 32;  // values per int8 scale

void attention_decode_ref(const AttnArgs & a, float * scratch);
void attention_decode_avx2(const AttnArgs & a, float * scratch);
// scratch must hold at least attention_scratch_floats(a) floats: the scores of
// one kv group, which is (n_heads / n_kv_heads) * n_kv.
size_t attention_scratch_floats(const AttnArgs & a);

// Diagnostic only: same decode as attention_decode, but reports the elapsed
// microseconds of the score, softmax and weighted-v passes separately.
// Requires grouped scratch (attention_scratch_floats_grouped), because it
// follows whichever kernel attention_decode_grouped would pick.
void attention_decode_phases(const AttnArgs & a, float * scratch, double out_us[3]);

// Variant that walks the kv cache once per group of query heads instead of once
// per head. Needs its own, larger scratch: see attention_scratch_floats_grouped.
size_t attention_scratch_floats_grouped(const AttnArgs & a);
void attention_decode_grouped(const AttnArgs & a, float * scratch);

// Kv heads are independent: each owns its query heads' scores and its own
// output slice. So the pooled path splits them across workers and produces a
// bit-identical result, no online-softmax merge and no extra precision loss.
// scratch must hold attention_scratch_floats_grouped_mt(a, n_threads).
struct ThreadPool;
size_t attention_scratch_floats_grouped_mt(const AttnArgs & a, unsigned n_threads);
void attention_decode_grouped_mt(const AttnArgs & a, float * scratch, ThreadPool & pool);
// Same decode, but the score pass reads k from the transposed cache built by
// kt_write. Needs at least kt_floats() and the grouped scratch.
void attention_decode_grouped_kt(const AttnArgs & a, float * scratch, const float * kt);
void attention_decode(const AttnArgs & a, float * scratch);

// Writes k and v for one token into the cache slot at a->n_kv - 1, applying
// nothing else. Separated so the caller can RoPE k first.
inline float * kv_slot(float * cache, size_t pos, size_t n_kv_heads, size_t head_dim) {
    return cache + pos * n_kv_heads * head_dim;
}

// Writes one token's k and v rows into a head-major cache
// [kv_head][position][head_dim]. src_k/src_v are laid out [kv_head][head_dim].
inline void write_kv_head_major(float * k_cache, float * v_cache, size_t pos, size_t n_ctx,
                               size_t n_kv_heads, size_t head_dim, const float * src_k,
                               const float * src_v) {
    for (size_t h = 0; h < n_kv_heads; ++h) {
        float * kd = k_cache + (h * n_ctx + pos) * head_dim;
        float * vd = v_cache + (h * n_ctx + pos) * head_dim;
        for (size_t d = 0; d < head_dim; ++d) {
            kd[d] = src_k[h * head_dim + d];
            vd[d] = src_v[h * head_dim + d];
        }
    }
}

// Transposed K cache, used only by the score pass: it keeps positions in the
// vector lanes instead of head_dim, so the accumulator count stops depending on
// how many positions a tile covers and K is read contiguously by position.
// Layout [kv_head][pos_block][dim][pos_in_block], KT_BLK positions per block.
inline constexpr size_t KT_BLK = 64;

inline size_t kt_floats(size_t n_kv, size_t n_kv_heads, size_t head_dim) {
    const size_t blocks = (n_kv + KT_BLK - 1) / KT_BLK;
    return blocks * n_kv_heads * head_dim * KT_BLK;
}

inline float * kt_slot(float * kt, size_t pos, size_t kv_head, size_t n_kv, size_t head_dim) {
    const size_t blocks = (n_kv + KT_BLK - 1) / KT_BLK;
    return kt + ((kv_head * blocks + pos / KT_BLK) * head_dim) * KT_BLK + (pos % KT_BLK);
}

// Writes one position's k row (head_dim floats, this kv head only) transposed.
inline void kt_write(float * kt, size_t pos, size_t kv_head, size_t n_kv, size_t head_dim,
                     const float * k_row) {
    const size_t blocks = (n_kv + KT_BLK - 1) / KT_BLK;
    float * base = kt + ((kv_head * blocks + pos / KT_BLK) * head_dim) * KT_BLK + (pos % KT_BLK);
    for (size_t d = 0; d < head_dim; ++d) base[d * KT_BLK] = k_row[d];
}

} // namespace kata
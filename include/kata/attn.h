#pragma once

#include <cstddef>
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
};

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
void attention_decode(const AttnArgs & a, float * scratch);

// Writes k and v for one token into the cache slot at a->n_kv - 1, applying
// nothing else. Separated so the caller can RoPE k first.
inline float * kv_slot(float * cache, size_t pos, size_t n_kv_heads, size_t head_dim) {
    return cache + pos * n_kv_heads * head_dim;
}

} // namespace kata
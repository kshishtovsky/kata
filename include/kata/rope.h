#pragma once

#include <cstddef>

namespace kata {

// Rotary position embedding, applied in place to one token's vectors.
//
// Two pairings appear in real models and they are NOT interchangeable:
//   interleaved=true  pairs (2i, 2i+1)          — llama.cpp ROPE_TYPE_NORM, most older models
//   interleaved=false pairs (i, i + head_dim/2) — ROPE_TYPE_NEOX, GPT-NeoX / Qwen / LFM2
// Getting this wrong rotates real numbers into the wrong partners: the output
// norm stays correct, so nothing looks broken, but generation quality collapses.
void rope_ref(size_t n_heads, size_t head_dim, float * x, size_t pos, float theta_base, bool interleaved);
void rope_avx2(size_t n_heads, size_t head_dim, float * x, size_t pos, float theta_base, bool interleaved);
// Fills cos/sin for the given position, so a caller looping over positions pays
// the pow() table once instead of per token.
void rope_tables(size_t head_dim, size_t pos, float theta_base, float * cos_out, float * sin_out);

// Same rotation, but cos/sin come from a caller-owned table instead of being
// rebuilt from (pos, theta_base). Prefill re-derives identical tables for the
// same positions in every attention layer, which showed up as 614 ms of a 16k
// run; this overload lets the caller build them once per chunk.
void rope_tab(size_t n_heads, size_t head_dim, float * x, const float * cos_t, const float * sin_t,
              bool interleaved);

// Fills cos_out/sin_out for n_pos consecutive positions from pos0, laid out as
// [position][head_dim/2] — the layout rope_tab expects.
void rope_table_range(size_t head_dim, size_t pos0, size_t n_pos, float theta_base, float * cos_out,
                      float * sin_out);
void rope(size_t n_heads, size_t head_dim, float * x, size_t pos, float theta_base, bool interleaved);

} // namespace kata
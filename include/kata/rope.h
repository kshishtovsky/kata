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
void rope(size_t n_heads, size_t head_dim, float * x, size_t pos, float theta_base, bool interleaved);

} // namespace kata
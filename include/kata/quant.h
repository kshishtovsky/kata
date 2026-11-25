#pragma once
#include <cstddef>
#include <cstdint>
#include "kata/types.h"

namespace kata {

float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);

// x: n floats, n must be a multiple of QK. y holds n/QK blocks.
void quantize_row_q8_0(const float * x, block_q8_0 * y, size_t n);
void quantize_row_q4_0(const float * x, block_q4_0 * y, size_t n);
void dequantize_row_q4_0(const block_q4_0 * x, float * y, size_t n);

// Independent scalar reference, used to validate every SIMD path.
// ggml-compatible: iscale = -127/max, so the extreme value lands on -127.
void quantize_row_q8_K(const float * x, block_q8_K * y, size_t n);
void dequantize_row_q4_K(const block_q4_K * x, float * y, size_t n);
float vec_dot_q4_K_q8_K_ref(size_t n, const block_q4_K * x, const block_q8_K * y);

// Extract the 8 6-bit scales and 8 6-bit mins of one super-block.
void q4_K_scale_min(int j, const uint8_t * q, uint8_t * d, uint8_t * m);

float vec_dot_q4_0_q8_0_ref(size_t n, const block_q4_0 * x, const block_q8_0 * y);

// q6_K: 6-bit values in [-32, 31], 16 sub-blocks of 16 with signed 8-bit scales.
void   dequantize_row_q6_K(const block_q6_K * x, float * y, size_t n);
float  vec_dot_q6_K_q8_K_ref(size_t n, const block_q6_K * x, const block_q8_K * y);

} // namespace kata

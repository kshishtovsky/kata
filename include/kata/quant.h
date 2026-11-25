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
float vec_dot_q4_0_q8_0_ref(size_t n, const block_q4_0 * x, const block_q8_0 * y);

} // namespace kata

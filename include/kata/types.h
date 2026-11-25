#pragma once
#include <cstdint>
#include <cstddef>

namespace kata {

// Elements per block for the q4_0/q8_0 families.
inline constexpr int QK = 32;

// Byte layout is identical to ggml/gguf, so existing .gguf weights can be read
// without a conversion pass.
struct block_q4_0 {
    uint16_t d;            // f16 scale
    uint8_t  qs[QK / 2];   // 32 signed 4-bit values, value j in low nibble j, value j+16 in high nibble j
};
struct block_q8_0 {
    uint16_t d;            // f16 scale
    int8_t   qs[QK];
};

static_assert(sizeof(block_q4_0) == 18, "q4_0 block must stay 18 bytes");
static_assert(sizeof(block_q8_0) == 34, "q8_0 block must stay 34 bytes");
static_assert(offsetof(block_q4_0, qs) == 2, "q4_0 qs offset");
static_assert(offsetof(block_q8_0, qs) == 2, "q8_0 qs offset");

} // namespace kata

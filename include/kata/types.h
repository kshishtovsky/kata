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

// K-quant family: 256 elements per super-block, 8 sub-blocks of 32 sharing
// 6-bit scales packed into 12 bytes. This is the format the shipped models
// actually use (77.6% of LFM2.5-24B-A2B-Q4_K_M is Q4_K).
inline constexpr int QK_K = 256;
inline constexpr int K_SCALE_SIZE = 12;

// weight = d * scale[i] * q - dmin * min[i], q in [0,15]
struct block_q4_K {
    uint16_t d;                    // f16 super-block scale
    uint16_t dmin;                 // f16 super-block min scale
    uint8_t  scales[K_SCALE_SIZE]; // 8 x 6-bit scales and 8 x 6-bit mins
    uint8_t  qs[QK_K / 2];         // value j in the low nibble, value j+32 in the high nibble
};

// Activation side of the K-quant dot: 256 signed bytes plus per-16 sums, which
// is what lets the per-sub-block min be applied with integer math only.
struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(offsetof(block_q4_0, qs) == 2, "q4_0 qs offset");
static_assert(offsetof(block_q8_0, qs) == 2, "q8_0 qs offset");
static_assert(sizeof(block_q4_K) == 144, "q4_K block must stay 144 bytes");
static_assert(sizeof(block_q8_K) == 4 + 256 + 32, "q8_K block must stay 292 bytes");

} // namespace kata

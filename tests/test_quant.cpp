#include <gtest/gtest.h>
#include "kata/quant.h"
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace kata;

TEST(F16, RoundTripRepresentative) {
    const float vals[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, 6.1035156e-05f, 1e-8f, 123.456f, -0.001234f};
    for (float v : vals) {
        const float back = f16_to_f32(f32_to_f16(v));
        ASSERT_NEAR(back, v, std::fabs(v) * 1e-3f + 1e-7f) << "v=" << v;
    }
}

TEST(F16, SubnormalsAreNotHalved) {
    // Every f16 subnormal: value must be mantissa * 2^-24 exactly. This is the
    // regression that corrupted q4_K blocks with small super-block scales.
    for (uint32_t m = 1; m < 0x400; ++m) {
        const float expect = (float)m * 5.9604644775390625e-08f;  // 2^-24
        const float got = kata::f16_to_f32((uint16_t)m);
        ASSERT_EQ(got, expect) << "m=" << m;
        ASSERT_EQ(kata::f16_to_f32((uint16_t)(m | 0x8000u)), -expect);
    }
    ASSERT_EQ(kata::f16_to_f32(0x0001u), std::ldexp(1.0f, -24));
    ASSERT_EQ(kata::f16_to_f32(0x03ffu), 1023.0f * std::ldexp(1.0f, -24));
    // Round trip through the encoder must also survive.
    for (uint32_t m = 1; m < 0x400; ++m) {
        ASSERT_EQ(kata::f32_to_f16((float)m * 5.9604644775390625e-08f), (uint16_t)m) << "m=" << m;
    }
}

TEST(F16, SpecialValues) {
    EXPECT_TRUE(std::isinf(f16_to_f32(f32_to_f16(1e30f))));
    EXPECT_TRUE(std::isnan(f16_to_f32(f32_to_f16(NAN))));
    EXPECT_EQ(f16_to_f32(0x0000), 0.0f);
    EXPECT_TRUE(std::signbit(f16_to_f32(0x8000)));
    EXPECT_EQ(f16_to_f32(0x3c00), 1.0f);
}

TEST(Quant, Q8RoundTripErrorBounded) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 3.0f);
    std::vector<float> src(QK * 8);
    for (auto & v : src) v = dist(rng);

    std::vector<block_q8_0> q(src.size() / QK);
    quantize_row_q8_0(src.data(), q.data(), src.size());

    for (size_t b = 0; b < q.size(); ++b) {
        const float d = f16_to_f32(q[b].d);
        for (int j = 0; j < QK; ++j) {
            const float back = d * (float)q[b].qs[j];
            // One quant step is d; allow 1.5 steps for the f16 scale rounding too.
            ASSERT_LE(std::fabs(back - src[b * QK + j]), 1.5f * d + 1e-6f);
        }
    }
}

TEST(Quant, Q8AllZeroBlock) {
    std::vector<float> src(QK, 0.0f);
    block_q8_0 q;
    quantize_row_q8_0(src.data(), &q, src.size());
    EXPECT_EQ(f16_to_f32(q.d), 0.0f);
    for (int j = 0; j < QK; ++j) EXPECT_EQ(q.qs[j], 0);
}

TEST(Quant, Q4DequantLayout) {
    // value j in low nibble of byte j, value j+16 in high nibble of byte j
    block_q4_0 b;
    b.d = f32_to_f16(0.5f);
    for (int j = 0; j < QK / 2; ++j) b.qs[j] = (uint8_t)(((j % 8) << 4) | ((15 - j) & 0x0f));

    float out[QK];
    dequantize_row_q4_0(&b, out, QK);

    for (int j = 0; j < QK / 2; ++j) {
        EXPECT_FLOAT_EQ(out[j], 0.5f * (float)((15 - j) - 8)) << "low j=" << j;
        EXPECT_FLOAT_EQ(out[j + QK / 2], 0.5f * (float)((j % 8) - 8)) << "high j=" << j;
    }
}

TEST(Quant, RefDotMatchesDequant) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> q4(0, 255);
    std::uniform_int_distribution<int> q8(-127, 127);
    std::uniform_real_distribution<float> sc(0.001f, 0.05f);

    const size_t nb = 16;
    std::vector<block_q4_0> w(nb);
    std::vector<block_q8_0> x(nb);
    for (size_t b = 0; b < nb; ++b) {
        w[b].d = f32_to_f16(sc(rng));
        x[b].d = f32_to_f16(sc(rng));
        for (int j = 0; j < QK / 2; ++j) w[b].qs[j] = (uint8_t)q4(rng);
        for (int j = 0; j < QK; ++j) x[b].qs[j] = (int8_t)q8(rng);
    }

    double expect = 0.0;
    for (size_t b = 0; b < nb; ++b) {
        float wf[QK], xf[QK];
        dequantize_row_q4_0(&w[b], wf, QK);
        const float dx = f16_to_f32(x[b].d);
        for (int j = 0; j < QK; ++j) {
            xf[j] = dx * (float)x[b].qs[j];
            expect += (double)wf[j] * (double)xf[j];
        }
    }
    const float got = vec_dot_q4_0_q8_0_ref(QK * nb, w.data(), x.data());
    ASSERT_NEAR(got, (float)expect, std::fabs((float)expect) * 1e-5f + 1e-3f);
}

// ggml rounds to nearest-even when packing scales; a half-up rule changed 1.6% of
// blocks on blk.0.ffn_gate of LFM2.5.
static uint16_t f32_to_f16_rne(float f) {
    const uint32_t x = [&] { uint32_t t; std::memcpy(&t, &f, 4); return t; }();
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    int32_t e = (int32_t)exp - 127 + 15;
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t sh = (uint32_t)(14 - e), r0 = mant >> sh, rem = mant & ((1u << sh) - 1);
        const uint32_t half = 1u << (sh - 1);
        return (uint16_t)(sign | (r0 + (rem > half || (rem == half && (r0 & 1)))));
    }
    uint32_t r = mant >> 13;
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1)))
        if (++r == 0x400u) { r = 0; if (++e >= 0x1f) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)e << 10) | r);
}

TEST(F16, MatchesRoundToNearestEven) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-70000.0f, 70000.0f);
    for (int i = 0; i < 20000; ++i) {
        const float v = dist(rng) * std::pow(10.0f, -(float)(rng() % 12));
        ASSERT_EQ(f32_to_f16(v), f32_to_f16_rne(v)) << "v=" << v;
    }
    // Cases exactly on a tie: halfway between 1.0 and the next f16 value rounds to
    // 1.0 (even), halfway between the next two rounds up (even) instead of down.
    const float kU = 0.0009765625f;                       // f16 ulp in [1,2)
    EXPECT_EQ(f16_to_f32(f32_to_f16(1.0f + 0.5f * kU)), 1.0f);
    EXPECT_NEAR(f16_to_f32(f32_to_f16(1.0f + 1.5f * kU)), 1.0f + 2 * kU, 1e-7f);
}

// The signed scale must map the extreme weight onto a representable code, else
// the whole block is offset and every dot product drifts.
TEST(Quant, Q4ScaleAnchorsExtremeWeight) {
    std::vector<float> x(64);
    for (int i = 0; i < 64; ++i) x[i] = 0.001f * (float)i;
    x[37] = 0.75f;                                        // dominant positive in block 1
    std::vector<block_q4_0> q(2);
    quantize_row_q4_0(x.data(), q.data(), 64);
    float dq[64];
    dequantize_row_q4_0(q.data(), dq, 64);
    EXPECT_NEAR(f16_to_f32(q[1].d), -0.75f / 8.0f, 1e-6f);  // d = maxv / -8, signed
    EXPECT_NEAR(dq[37], 0.75f, 2e-2f);                    // round trip within one step
}

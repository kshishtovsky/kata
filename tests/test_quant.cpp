#include <gtest/gtest.h>
#include "kata/quant.h"
#include <cmath>
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

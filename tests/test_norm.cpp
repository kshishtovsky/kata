#include <gtest/gtest.h>

#include "kata/backend.h"
#include "kata/cpu.h"
#include "kata/norm.h"
#include "kata/gemv.h"
#include "kata/quant.h"

#include <cmath>
#include <random>
#include <vector>

namespace {
float rms_scale(const std::vector<float> & x, size_t off, size_t cols, float eps) {
    double sum = 0.0;
    for (size_t c = 0; c < cols; ++c) sum += (double)x[off + c] * (double)x[off + c];
    return 1.0f / std::sqrt((float)(sum / (double)cols) + eps);
}
} // namespace

TEST(RmsNorm, MatchesDefinition) {
    std::mt19937 g(5);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    const size_t rows = 3, cols = 40;
    std::vector<float> x(rows * cols), w(cols), out(rows * cols);
    for (auto & v : x) v = nd(g);
    for (auto & v : w) v = 0.5f + 0.01f * (float)(&v - w.data());
    const float eps = 1e-5f;
    kata::rms_norm_ref(rows, cols, out.data(), x.data(), w.data(), eps);
    for (size_t r = 0; r < rows; ++r) {
        const float s = rms_scale(x, r * cols, cols, eps);
        for (size_t c = 0; c < cols; ++c) {
            EXPECT_NEAR(out[r * cols + c], x[r * cols + c] * s * w[c], 1e-6f) << r << "," << c;
        }
    }
}

TEST(RmsNorm, NoWeightScalesOnly) {
    std::mt19937 g(6);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t cols = 16;
    std::vector<float> x(cols), out(cols), out_ref(cols);
    for (auto & v : x) v = nd(g);
    kata::rms_norm_ref(1, cols, out_ref.data(), x.data(), nullptr, 1e-6f);
    kata::rms_norm(1, cols, out.data(), x.data(), nullptr, 1e-6f);
    const float s = rms_scale(x, 0, cols, 1e-6f);
    for (size_t c = 0; c < cols; ++c) {
        EXPECT_NEAR(out[c], x[c] * s, 1e-6f);
        EXPECT_NEAR(out[c], out_ref[c], 1e-6f);
    }
}

TEST(RmsNorm, VectorPathsMatchRefForUnalignedWidths) {
    if (kata::cpu_isa() == kata::Isa::Scalar) GTEST_SKIP() << "no AVX2 on this host";
    std::mt19937 g(7);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    const float eps = 1e-6f;
    for (size_t cols : {size_t(1), size_t(7), size_t(8), size_t(15), size_t(16), size_t(23), size_t(64), size_t(257)}) {
        for (const bool with_w : {false, true}) {
            const size_t rows = 3;
            std::vector<float> x(rows * cols), w(cols), a(rows * cols), b(rows * cols);
            for (auto & v : x) v = nd(g);
            for (auto & v : w) v = 0.25f + 0.001f * (float)(&v - w.data());
            const float * wp = with_w ? w.data() : nullptr;
            kata::rms_norm_ref(rows, cols, a.data(), x.data(), wp, eps);
            kata::rms_norm(rows, cols, b.data(), x.data(), wp, eps);
            for (size_t i = 0; i < rows * cols; ++i) {
                EXPECT_NEAR(b[i], a[i], 2e-5f * std::max(1.0f, std::fabs(a[i])))
                    << "cols=" << cols << " with_w=" << with_w << " i=" << i;
            }
        }
    }
}

TEST(RmsNorm, ZeroRowGivesZeroNotNan) {
    std::vector<float> x(8, 0.0f), out(8, 123.0f);
    kata::rms_norm(1, 8, out.data(), x.data(), nullptr, 1e-5f);
    for (float v : out) { EXPECT_FALSE(std::isnan(v)); EXPECT_EQ(v, 0.0f); }
}

TEST(Backend, CpuIsAvailableAndRoutesNorm) {
    const kata::Backend & b = kata::backend();
    ASSERT_NE(b.name(), nullptr);
    EXPECT_TRUE(b.available());
    std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f}, out(4), ref(4);
    kata::RmsNormArgs a{};
    a.x = x.data(); a.w = nullptr; a.out = out.data(); a.rows = 1; a.cols = 4; a.eps = 1e-5f;
    b.rms_norm(a);
    kata::rms_norm_ref(1, 4, ref.data(), x.data(), nullptr, 1e-5f);
    for (size_t i = 0; i < 4; ++i) EXPECT_NEAR(out[i], ref[i], 1e-6f);
}

TEST(Backend, RoutesBothQuantisedGemvFormats) {
    // The backend boundary is the only path forward code will take, so a wrong
    // cast or swapped argument here would corrupt every later result.
    const kata::Backend & b = kata::backend();
    std::vector<kata::block_q4_K> w4(2);
    std::vector<kata::block_q6_K> w6(2);
    for (auto & x : w4) {
        x.d = kata::f32_to_f16(0.02f); x.dmin = kata::f32_to_f16(0.01f);
        for (auto & v : x.qs) v = 0x21;
        for (auto & v : x.scales) v = 0x24;
    }
    for (auto & x : w6) {
        x.d = kata::f32_to_f16(0.02f);
        for (auto & v : x.ql) v = 0x21;
        for (auto & v : x.qh) v = 0x10;
        for (auto & v : x.scales) v = 3;
    }
    std::vector<float> act(kata::QK_K, 0.5f);
    std::vector<kata::block_q8_K> xq(1);
    kata::quantize_row_q8_K(act.data(), xq.data(), kata::QK_K);

    std::vector<float> got4(2), want4(2), got6(2), want6(2);
    kata::GemvArgs a{}; a.rows = 2; a.cols = kata::QK_K; a.x = xq.data();
    a.out = got4.data(); a.w = w4.data();
    b.gemv_q4_K_q8_K(a);
    kata::gemv_q4_K_q8_K_ref(2, kata::QK_K, want4.data(), w4.data(), xq.data());
    a.out = got6.data(); a.w = w6.data();
    b.gemv_q6_K_q8_K(a);
    kata::gemv_q6_K_q8_K_ref(2, kata::QK_K, want6.data(), w6.data(), xq.data());
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(got4[i], want4[i], 1e-5f * std::max(1.0f, std::fabs(want4[i])));
        EXPECT_NEAR(got6[i], want6[i], 1e-5f * std::max(1.0f, std::fabs(want6[i])));
    }
}

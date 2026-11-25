#include <gtest/gtest.h>
#include "kata/gemv.h"
#include "kata/quant.h"
#include "kata/cpu.h"
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace kata;

namespace {

struct Data {
    size_t rows, cols;
    std::vector<block_q4_0> w;
    std::vector<block_q8_0> x;
    std::vector<float> ref;

    Data(size_t r, size_t c, unsigned seed) : rows(r), cols(c) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> q4(0, 255);
        std::uniform_real_distribution<float> sc(0.001f, 0.02f);
        std::normal_distribution<float> act(0.0f, 1.0f);

        const size_t nb = c / QK;
        w.resize(r * nb);
        x.resize(nb);
        for (auto & blk : w) {
            blk.d = f32_to_f16(sc(rng));
            for (int j = 0; j < QK / 2; ++j) blk.qs[j] = (uint8_t)q4(rng);
        }
        std::vector<float> xf(c);
        for (auto & v : xf) v = act(rng);
        quantize_row_q8_0(xf.data(), x.data(), c);

        ref.resize(r);
        gemv_q4_0_q8_0_ref(r, c, ref.data(), w.data(), x.data());
    }
};

void expect_same(const Data & d, const std::vector<float> & got, float rel) {
    ASSERT_EQ(got.size(), d.ref.size());
    for (size_t i = 0; i < got.size(); ++i) {
        const float tol = std::fabs(d.ref[i]) * rel + 1e-3f;
        ASSERT_NEAR(got[i], d.ref[i], tol) << "row " << i;
    }
}

} // namespace

TEST(Isa, DetectionIsDenseWithKernelSelection) {
    const Isa isa = cpu_isa();
    const GemvFn fn = gemv_q4_0_q8_0_best();
    ASSERT_NE(fn, nullptr);
    if (isa == Isa::Scalar) EXPECT_EQ(fn, gemv_q4_0_q8_0_ref);
    if (isa == Isa::Avx2)     EXPECT_EQ(fn, gemv_q4_0_q8_0_avx2);
    if (isa == Isa::Avx512Vnni) EXPECT_EQ(fn, gemv_q4_0_q8_0_avx512);
    EXPECT_STRNE(isa_name(isa), "");
    EXPECT_STRNE(gemv_best_name(), "");
}

TEST(Gemv, RefMatchesItself) {
    Data d(7, 128, 11);
    std::vector<float> got(d.rows);
    gemv_q4_0_q8_0_ref(d.rows, d.cols, got.data(), d.w.data(), d.x.data());
    expect_same(d, got, 1e-6f);
}

TEST(Gemv, Avx2MatchesRef) {
    Data d(9, 256, 12);   // odd row count exercises the tail
    std::vector<float> got(d.rows);
    gemv_q4_0_q8_0_avx2(d.rows, d.cols, got.data(), d.w.data(), d.x.data());
    expect_same(d, got, 2e-4f);
}

TEST(Gemv, Avx512MatchesRef) {
    Data d(11, 256, 13);  // tail of 3 after four-row groups
    std::vector<float> got(d.rows);
    gemv_q4_0_q8_0_avx512(d.rows, d.cols, got.data(), d.w.data(), d.x.data());
    expect_same(d, got, 2e-4f);
}

TEST(Gemv, SingleRowAllKernels) {
    Data d(1, 64, 14);
    for (GemvFn fn : {gemv_q4_0_q8_0_ref, gemv_q4_0_q8_0_avx2, gemv_q4_0_q8_0_avx512}) {
        std::vector<float> got(1);
        fn(d.rows, d.cols, got.data(), d.w.data(), d.x.data());
        expect_same(d, got, 2e-4f);
    }
}

TEST(Gemv, FourRowsExactGroup) {
    Data d(4, 96, 15);
    std::vector<float> got(4);
    gemv_q4_0_q8_0_avx512(4, 96, got.data(), d.w.data(), d.x.data());
    expect_same(d, got, 2e-4f);
}

TEST(Gemv, KnownDotValue) {
    // Hand-computed: all nibbles encode 0 except a single 1 at position 0 in
    // row 0, activations all 1 -> dot is d_w * d_x * 1.
    const size_t cols = QK;
    std::vector<block_q4_0> w(1);
    std::vector<block_q8_0> x(1);
    w[0].d = f32_to_f16(0.25f);
    x[0].d = f32_to_f16(0.5f);
    for (int j = 0; j < QK / 2; ++j) w[0].qs[j] = 0x88;  // all values are 8 -> decoded 0
    for (int j = 0; j < QK; ++j) x[0].qs[j] = 1;
    w[0].qs[0] = 0x98;                                   // low nibble 9 -> decoded +1

    float got = 0.0f;
    gemv_q4_0_q8_0_ref(1, cols, &got, w.data(), x.data());
    EXPECT_FLOAT_EQ(got, 0.25f * 0.5f * 1.0f * 1.0f);

    for (GemvFn fn : {gemv_q4_0_q8_0_avx2, gemv_q4_0_q8_0_avx512}) {
        float v = 0.0f;
        fn(1, cols, &v, w.data(), x.data());
        EXPECT_NEAR(v, got, 1e-6f);
    }
}

TEST(Gemv, MultithreadedMatchesReference) {
    Data d(37, 512, 21);   // prime-ish row count to hit uneven chunks
    for (unsigned nt : {1u, 2u, 3u, 8u}) {
        ThreadPool pool(nt);
        std::vector<float> got(d.rows, 0.0f);
        gemv_q4_0_q8_0_mt(d.rows, d.cols, got.data(), d.w.data(), d.x.data(), pool);
        expect_same(d, got, 2e-4f);
    }
}

TEST(Gemv, MultithreadedTinyMatrices) {
    for (size_t rows : {1u, 2u, 3u}) {
        Data d(rows, 64, 22 + (unsigned)rows);
        ThreadPool pool(8);
        std::vector<float> got(rows, 0.0f);
        gemv_q4_0_q8_0_mt(rows, 64, got.data(), d.w.data(), d.x.data(), pool);
        expect_same(d, got, 2e-4f);
    }
}

namespace {
std::vector<kata::block_q4_K> random_q4_K(size_t nblocks, std::mt19937 & g) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<kata::block_q4_K> out(nblocks);
    for (auto & b : out) {
        b.d = kata::f32_to_f16(0.001f + 0.01f * (byte(g) / 255.0f));
        b.dmin = kata::f32_to_f16(0.001f + 0.01f * (byte(g) / 255.0f));
        for (int i = 0; i < 12; ++i) b.scales[i] = (uint8_t)byte(g);
        for (int i = 0; i < 128; ++i) b.qs[i] = (uint8_t)byte(g);
    }
    return out;
}
std::vector<kata::block_q8_K> random_q8_K(size_t nblocks, size_t cols, std::mt19937 & g) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f(cols);
    for (auto & v : f) v = nd(g);
    std::vector<kata::block_q8_K> out(nblocks);
    kata::quantize_row_q8_K(f.data(), out.data(), cols);
    return out;
}
} // namespace

TEST(GemvK, RefMatchesDequantisedDot) {
    std::mt19937 g(7);
    const size_t cols = 3 * kata::QK_K;
    auto w = random_q4_K(4 * 3, g);
    auto x = random_q8_K(3, cols, g);
    std::vector<float> deq(cols), got(4);
    kata::gemv_q4_K_q8_K_ref(4, cols, got.data(), w.data(), x.data());
    for (size_t r = 0; r < 4; ++r) {
        kata::dequantize_row_q4_K(&w[r * 3], deq.data(), cols);
        double want = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            want += (double)deq[c] * (double)x[c / kata::QK_K].qs[c % kata::QK_K] * (double)x[c / kata::QK_K].d;
        }
        EXPECT_NEAR(got[r], want, 1e-3 * std::max(1.0, double(std::fabs(want)))) << "row " << r;
    }
}

TEST(GemvK, Avx2MatchesRef) {
    if (kata::cpu_isa() == kata::Isa::Scalar) GTEST_SKIP() << "no AVX2 on this host";
    std::mt19937 g(11);
    const size_t cols = 5 * kata::QK_K, rows = 9;
    auto w = random_q4_K(rows * 5, g);
    auto x = random_q8_K(5, cols, g);
    std::vector<float> want(rows), got(rows);
    kata::gemv_q4_K_q8_K_ref(rows, cols, want.data(), w.data(), x.data());
    kata::gemv_q4_K_q8_K_avx2(rows, cols, got.data(), w.data(), x.data());
    for (size_t r = 0; r < rows; ++r) EXPECT_NEAR(got[r], want[r], 1e-4 * std::max(1.0, double(std::fabs(want[r])))) << "row " << r;
}

TEST(GemvK, SubnormalSuperblockScales) {
    // Regression for the f16 exponent bug: rows whose super-block scales are
    // f16 subnormals used to come out at half magnitude.
    std::mt19937 g(3);
    auto w = random_q4_K(2, g);
    w[0].d = 0x0350;      // subnormal
    w[1].d = 0x0398;      // subnormal
    auto x = random_q8_K(1, kata::QK_K, g);
    std::vector<float> want(2), got(2);
    kata::gemv_q4_K_q8_K_ref(2, kata::QK_K, want.data(), w.data(), x.data());
    kata::gemv_q4_K_q8_K_avx2(2, kata::QK_K, got.data(), w.data(), x.data());
    EXPECT_GT(std::fabs(want[0]), 0.0f);
    EXPECT_NEAR(got[0], want[0], 1e-5 * std::max(1.0, double(std::fabs(want[0]))));
    EXPECT_NEAR(got[1], want[1], 1e-5 * std::max(1.0, double(std::fabs(want[1]))));
}

namespace {
std::vector<kata::block_q6_K> random_q6_K(size_t nblocks, std::mt19937 & g) {
    std::uniform_int_distribution<int> byte(0, 255), sc(-128, 127);
    std::vector<kata::block_q6_K> out(nblocks);
    for (auto & b : out) {
        b.d = kata::f32_to_f16(0.001f + 0.01f * (byte(g) / 255.0f));
        for (auto & v : b.ql) v = (uint8_t)byte(g);
        for (auto & v : b.qh) v = (uint8_t)byte(g);
        for (auto & v : b.scales) v = (int8_t)sc(g);
    }
    return out;
}
} // namespace

TEST(GemvK6, RefMatchesDequantisedDot) {
    std::mt19937 g(17);
    const size_t cols = 2 * kata::QK_K;
    auto w = random_q6_K(3 * 2, g);
    auto x = random_q8_K(2, cols, g);
    std::vector<float> deq(cols), got(3);
    kata::gemv_q6_K_q8_K_ref(3, cols, got.data(), w.data(), x.data());
    for (size_t r = 0; r < 3; ++r) {
        kata::dequantize_row_q6_K(&w[r * 2], deq.data(), cols);
        double want = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            want += (double)deq[c] * (double)x[c / kata::QK_K].qs[c % kata::QK_K] * (double)x[c / kata::QK_K].d;
        }
        EXPECT_NEAR(got[r], want, 1e-3 * std::max(1.0, double(std::fabs(want)))) << "row " << r;
    }
}

TEST(GemvK6, Avx2MatchesRef) {
    if (kata::cpu_isa() == kata::Isa::Scalar) GTEST_SKIP() << "no AVX2 on this host";
    std::mt19937 g(19);
    const size_t cols = 5 * kata::QK_K, rows = 7;
    auto w = random_q6_K(rows * 5, g);
    auto x = random_q8_K(5, cols, g);
    std::vector<float> want(rows), got(rows);
    kata::gemv_q6_K_q8_K_ref(rows, cols, want.data(), w.data(), x.data());
    kata::gemv_q6_K_q8_K_avx2(rows, cols, got.data(), w.data(), x.data());
    for (size_t r = 0; r < rows; ++r) {
        EXPECT_NEAR(got[r], want[r], 1e-4 * std::max(1.0, double(std::fabs(want[r])))) << "row " << r;
    }
}

TEST(GemvK6, PerSubBlockScalesAreNotDuplicated) {
    // Regression: one maddubs covers 32 values = two 16-value sub-blocks, so its
    // low and high halves need different scales. Using one duplicated scale made
    // half of every block wrong (error ~3.0 relative on a real model).
    std::mt19937 g(23);
    auto w = random_q6_K(1, g);
    for (int i = 0; i < 16; ++i) w[0].scales[i] = (int8_t)((i % 2) ? 1 : 100);  // alternate wildly
    auto x = random_q8_K(1, kata::QK_K, g);
    float want = 0.0f, got = 0.0f;
    kata::gemv_q6_K_q8_K_ref(1, kata::QK_K, &want, w.data(), x.data());
    kata::gemv_q6_K_q8_K_avx2(1, kata::QK_K, &got, w.data(), x.data());
    EXPECT_GT(std::fabs(want), 1e-6f);
    EXPECT_NEAR(got, want, 1e-4 * std::max(1.0, double(std::fabs(want))));
}

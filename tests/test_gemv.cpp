#include <gtest/gtest.h>
#include "kata/gemv.h"
#include "kata/quant.h"
#include "kata/cpu.h"
#include <cmath>
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

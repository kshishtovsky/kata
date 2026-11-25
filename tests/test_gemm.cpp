#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "kata/gemm.h"
#include "kata/gemv.h"
#include "kata/pool.h"
#include "kata/quant.h"
#include "kata/types.h"

namespace {

// kata has no q4_K/q6_K quantiser yet (only the dequantisers and the q8_K
// quantiser), so weight blocks are filled with random bytes and plausible
// scales. That is enough for a kernel-vs-kernel comparison, which is what these
// tests are: both sides read the identical bytes.
void fill_q4_K_weights(std::vector<kata::block_q4_K> & w, size_t rows, size_t cols, uint32_t seed) {
    (void)rows; (void)cols;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto & b : w) {
        b.d = kata::f32_to_f16(0.001f + 0.01f * (byte(rng) / 255.0f));
        b.dmin = kata::f32_to_f16(0.001f + 0.01f * (byte(rng) / 255.0f));
        for (int i = 0; i < 12; ++i) b.scales[i] = (uint8_t)byte(rng);
        for (int i = 0; i < 128; ++i) b.qs[i] = (uint8_t)byte(rng);
    }
}

void fill_q6_K_weights(std::vector<kata::block_q6_K> & w, size_t rows, size_t cols, uint32_t seed) {
    (void)rows; (void)cols;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto & b : w) {
        b.d = kata::f32_to_f16(0.001f + 0.01f * (byte(rng) / 255.0f));
        for (int i = 0; i < 128; ++i) b.ql[i] = (uint8_t)byte(rng);
        for (int i = 0; i < 64; ++i) b.qh[i] = (uint8_t)byte(rng);
        for (int i = 0; i < 16; ++i) b.scales[i] = (int8_t)byte(rng);
    }
}

void fill_activations(std::vector<kata::block_q8_K> & a, size_t m, size_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f(cols);
    for (size_t i = 0; i < m; ++i) {
        for (size_t c = 0; c < cols; ++c) f[c] = nd(rng);
        kata::quantize_row_q8_K(f.data(), a.data() + i * (cols / kata::QK_K), cols);
    }
}

// The GEMM must agree with one GEMV per activation row on the same data. The
// tolerance is relative to the row magnitude: rows whose dot cancels to ~0 make
// a relative check meaningless, so normalise by sum|w|*max|x| instead.
double max_rel_err(const std::vector<float> & got, const std::vector<float> & ref, size_t m, size_t n) {
    double scale = 0.0;
    for (float v : ref) scale = std::max(scale, (double)std::fabs(v));
    if (scale == 0.0) scale = 1.0;
    double worst = 0.0;
    for (size_t i = 0; i < m * n; ++i) worst = std::max(worst, std::fabs((double)got[i] - ref[i]) / scale);
    return worst;
}

} // namespace

TEST(GemmQ4K, MatchesPerRowGemv) {
    struct Shape { size_t m, n, k; };
    const Shape shapes[] = {{1, 16, 512}, {3, 16, 512}, {4, 16, 512}, {5, 16, 512},
                            {7, 32, 1024}, {16, 8, 2048}, {64, 4, 2048}};
    for (const Shape & s : shapes) {
        const size_t nb = s.k / kata::QK_K;
        std::vector<kata::block_q4_K> w(s.n * nb);
        std::vector<kata::block_q8_K> a(s.m * nb);
        fill_q4_K_weights(w, s.n, s.k, 11 + (uint32_t)s.m);
        fill_activations(a, s.m, s.k, 22 + (uint32_t)s.n);

        std::vector<float> got(s.m * s.n, 0.0f), ref(s.m * s.n, 0.0f);
        kata::gemm_q4_K_q8_K_ref(s.m, s.n, s.k, ref.data(), s.n, w.data(), a.data());
        kata::gemm_q4_K_q8_K(s.m, s.n, s.k, got.data(), s.n, w.data(), a.data());

        EXPECT_LT(max_rel_err(got, ref, s.m, s.n), 1e-5)
            << "m=" << s.m << " n=" << s.n << " k=" << s.k;
    }
}

TEST(GemmQ6K, MatchesPerRowGemv) {
    struct Shape { size_t m, n, k; };
    const Shape shapes[] = {{1, 16, 512}, {4, 16, 512}, {5, 16, 512}, {7, 32, 1024}, {16, 8, 2048}};
    for (const Shape & s : shapes) {
        const size_t nb = s.k / kata::QK_K;
        std::vector<kata::block_q6_K> w(s.n * nb);
        std::vector<kata::block_q8_K> a(s.m * nb);
        fill_q6_K_weights(w, s.n, s.k, 31 + (uint32_t)s.m);
        fill_activations(a, s.m, s.k, 41 + (uint32_t)s.n);

        std::vector<float> got(s.m * s.n, 0.0f), ref(s.m * s.n, 0.0f);
        kata::gemm_q6_K_q8_K_ref(s.m, s.n, s.k, ref.data(), s.n, w.data(), a.data());
        kata::gemm_q6_K_q8_K(s.m, s.n, s.k, got.data(), s.n, w.data(), a.data());

        EXPECT_LT(max_rel_err(got, ref, s.m, s.n), 1e-5)
            << "m=" << s.m << " n=" << s.n << " k=" << s.k;
    }
}

TEST(Gemm, BatchOfOneEqualsGemv) {
    const size_t n = 24, k = 1024, nb = k / kata::QK_K;
    std::vector<kata::block_q4_K> w(n * nb);
    std::vector<kata::block_q8_K> a(nb);
    fill_q4_K_weights(w, n, k, 5);
    fill_activations(a, 1, k, 6);

    std::vector<float> got(n), ref(n);
    kata::gemv_q4_K_q8_K_ref(n, k, ref.data(), w.data(), a.data());
    kata::gemm_q4_K_q8_K(1, n, k, got.data(), n, w.data(), a.data());

    EXPECT_LT(max_rel_err(got, ref, 1, n), 1e-6);
}

TEST(Gemm, MtDriverMatchesSingleThreaded) {
    const size_t m = 12, n = 64, k = 1024, nb = k / kata::QK_K;
    std::vector<kata::block_q4_K> w(n * nb);
    std::vector<kata::block_q8_K> a(m * nb);
    fill_q4_K_weights(w, n, k, 7);
    fill_activations(a, m, k, 8);

    std::vector<float> ref(m * n, 0.0f), got(m * n, 0.0f);
    kata::gemm_q4_K_q8_K(m, n, k, ref.data(), n, w.data(), a.data());

    kata::ThreadPool pool(4);
    kata::gemm_q4_K_q8_K_mt(m, n, k, got.data(), n, w.data(), a.data(), pool);

    EXPECT_LT(max_rel_err(got, ref, m, n), 1e-6);
}

TEST(Gemm, HandlesEmptyDimensions) {
    std::vector<float> out(4, 0.0f);
    std::vector<kata::block_q4_K> w(1);
    std::vector<kata::block_q8_K> a(1);
    kata::gemm_q4_K_q8_K(0, 4, 256, out.data(), 4, w.data(), a.data());
    kata::gemm_q4_K_q8_K(4, 0, 256, out.data(), 4, w.data(), a.data());
    for (float v : out) EXPECT_EQ(v, 0.0f);
}
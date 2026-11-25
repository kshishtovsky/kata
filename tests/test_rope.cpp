#include <gtest/gtest.h>

#include "kata/backend.h"
#include "kata/cpu.h"
#include "kata/rope.h"

#include <cmath>
#include <random>
#include <vector>

namespace {
// Independent, obviously-correct definition of RoPE to compare against.
void rope_by_definition(std::vector<float> & v, size_t n_heads, size_t head_dim, size_t pos,
                        float theta_base, bool interleaved) {
    const size_t half = head_dim / 2;
    for (size_t h = 0; h < n_heads; ++h) {
        for (size_t i = 0; i < half; ++i) {
            const double freq = std::pow((double)theta_base, -2.0 * (double)i / (double)head_dim);
            const double a = (double)pos * freq;
            const float ca = (float)std::cos(a), sa = (float)std::sin(a);
            float * base = v.data() + h * head_dim;
            const size_t ia = interleaved ? 2 * i : i;
            const size_t ib = interleaved ? 2 * i + 1 : i + half;
            const float xa = base[ia], xb = base[ib];
            base[ia] = xa * ca - xb * sa;
            base[ib] = xa * sa + xb * ca;
        }
    }
}
std::vector<float> random_vec(size_t n, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto & x : v) x = nd(g);
    return v;
}
} // namespace

TEST(Rope, MatchesDefinitionBothLayouts) {
    for (const bool interleaved : {true, false}) {
        for (size_t head_dim : {size_t(2), size_t(8), size_t(64), size_t(128), size_t(6), size_t(7)}) {
            for (size_t pos : {size_t(0), size_t(1), size_t(37), size_t(4096)}) {
                const size_t n_heads = 3;
                auto want = random_vec(n_heads * head_dim, 11 + (unsigned)head_dim);
                auto got = want;
                rope_by_definition(want, n_heads, head_dim, pos, 10000.0f, interleaved);
                kata::rope_ref(n_heads, head_dim, got.data(), pos, 10000.0f, interleaved);
                for (size_t i = 0; i < got.size(); ++i) {
                    EXPECT_NEAR(got[i], want[i], 2e-5f * std::max(1.0f, std::fabs(want[i])))
                        << "interleaved=" << interleaved << " head_dim=" << head_dim
                        << " pos=" << pos << " i=" << i;
                }
            }
        }
    }
}

TEST(Rope, VectorPathMatchesRef) {
    if (kata::cpu_isa() == kata::Isa::Scalar) GTEST_SKIP() << "no AVX2 on this host";
    for (const bool interleaved : {true, false}) {
        for (size_t head_dim : {size_t(8), size_t(16), size_t(64), size_t(128), size_t(10)}) {
            const size_t n_heads = 5, pos = 1234;
            auto base = random_vec(n_heads * head_dim, 5);
            auto a = base, b = base;
            kata::rope_ref(n_heads, head_dim, a.data(), pos, 500000.0f, interleaved);
            kata::rope(n_heads, head_dim, b.data(), pos, 500000.0f, interleaved);
            for (size_t i = 0; i < a.size(); ++i) {
                EXPECT_NEAR(b[i], a[i], 1e-5f * std::max(1.0f, std::fabs(a[i])))
                    << "interleaved=" << interleaved << " head_dim=" << head_dim << " i=" << i;
            }
        }
    }
}

TEST(Rope, PositionZeroIsIdentity) {
    const size_t n_heads = 2, head_dim = 16;
    auto x = random_vec(n_heads * head_dim, 3);
    const auto orig = x;
    kata::rope(n_heads, head_dim, x.data(), 0, 10000.0f, false);
    for (size_t i = 0; i < x.size(); ++i) EXPECT_FLOAT_EQ(x[i], orig[i]);
}

TEST(Rope, PreservesNormOfEachPair) {
    // A rotation must not change the magnitude of the (a,b) pair it acts on.
    const size_t n_heads = 2, head_dim = 32, half = head_dim / 2;
    auto x = random_vec(n_heads * head_dim, 9);
    auto before = x;
    kata::rope(n_heads, head_dim, x.data(), 777, 10000.0f, false);
    for (size_t h = 0; h < n_heads; ++h) {
        for (size_t i = 0; i < half; ++i) {
            const float b0 = before[h * head_dim + i], b1 = before[h * head_dim + i + half];
            const float a0 = x[h * head_dim + i], a1 = x[h * head_dim + i + half];
            EXPECT_NEAR(std::sqrt(a0 * a0 + a1 * a1), std::sqrt(b0 * b0 + b1 * b1), 1e-4f);
        }
    }
}

TEST(Rope, TablesAgreeWithFullApply) {
    const size_t head_dim = 64, half = head_dim / 2, pos = 900;
    std::vector<float> c(half), s(half);
    kata::rope_tables(head_dim, pos, 10000.0f, c.data(), s.data());
    auto x = random_vec(head_dim, 4);
    auto got = x;
    kata::rope_ref(1, head_dim, got.data(), pos, 10000.0f, false);
    for (size_t i = 0; i < half; ++i) {
        const float xa = x[i], xb = x[i + half];
        EXPECT_NEAR(got[i], xa * c[i] - xb * s[i], 1e-5f);
        EXPECT_NEAR(got[i + half], xa * s[i] + xb * c[i], 1e-5f);
    }
}

TEST(Rope, BackendRoutes) {
    const size_t n_heads = 2, head_dim = 8;
    auto x = random_vec(n_heads * head_dim, 13), ref = x;
    kata::rope_ref(n_heads, head_dim, ref.data(), 42, 10000.0f, true);
    kata::RopeArgs a{};
    a.x = x.data(); a.n_heads = n_heads; a.head_dim = head_dim;
    a.pos = 42; a.theta_base = 10000.0f; a.interleaved = true;
    kata::backend().rope(a);
    for (size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], ref[i], 1e-5f);
}
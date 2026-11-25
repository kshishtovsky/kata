#include <gtest/gtest.h>

#include "kata/attn.h"
#include "kata/cpu.h"

#include <cmath>
#include <random>
#include <vector>

namespace {
std::vector<float> rnd(size_t n, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 0.8f);
    std::vector<float> v(n);
    for (auto & x : v) x = nd(g);
    return v;
}
struct Case {
    size_t n_heads, n_kv_heads, head_dim, n_kv;
};
} // namespace

TEST(Attn, VectorMatchesRefAcrossShapes) {
    if (kata::cpu_isa() == kata::Isa::Scalar) GTEST_SKIP() << "no AVX2 on this host";
    const Case cases[] = {
        {32, 8, 64, 1},    // first token
        {32, 8, 64, 37},   // gqa, aligned
        {8, 8, 64, 129},   // mha
        {8, 2, 96, 200},   // head_dim divisible by 8 but not 16
        {4, 1, 80, 63},    // mqa
        {4, 2, 60, 41},    // head_dim not divisible by 8 -> scalar tail
    };
    for (const auto & c : cases) {
        auto q = rnd(c.n_heads * c.head_dim, 1);
        auto k = rnd(c.n_kv * c.n_kv_heads * c.head_dim, 2);
        auto v = rnd(c.n_kv * c.n_kv_heads * c.head_dim, 3);
        std::vector<float> want(c.n_heads * c.head_dim), got(c.n_heads * c.head_dim);
        kata::AttnArgs a{};
        a.q = q.data(); a.k_cache = k.data(); a.v_cache = v.data();
        a.n_kv = c.n_kv; a.n_heads = c.n_heads; a.n_kv_heads = c.n_kv_heads;
        a.head_dim = c.head_dim; a.scale = 1.0f / std::sqrt((float)c.head_dim);
        a.out = want.data();
        std::vector<float> s1(kata::attention_scratch_floats(a)), s2(s1.size());
        kata::attention_decode_ref(a, s1.data());
        a.out = got.data();
        kata::attention_decode(a, s2.data());
        for (size_t i = 0; i < want.size(); ++i) {
            EXPECT_NEAR(got[i], want[i], 2e-5f * std::max(1.0f, std::fabs(want[i])))
                << "heads=" << c.n_heads << "/" << c.n_kv_heads << " hd=" << c.head_dim
                << " n_kv=" << c.n_kv << " i=" << i;
        }
    }
}

TEST(Attn, FirstTokenReturnsExactlyV) {
    // With a single cached position softmax is 1, so out == v of that head.
    const size_t n_heads = 4, n_kv_heads = 2, hd = 8, n_kv = 1;
    auto q = rnd(n_heads * hd, 4);
    auto k = rnd(n_kv * n_kv_heads * hd, 5);
    auto v = rnd(n_kv * n_kv_heads * hd, 6);
    std::vector<float> out(n_heads * hd);
    std::vector<float> scratch(n_heads / n_kv_heads * n_kv + 8);
    kata::AttnArgs a{};
    a.q = q.data(); a.k_cache = k.data(); a.v_cache = v.data();
    a.n_kv = n_kv; a.n_heads = n_heads; a.n_kv_heads = n_kv_heads;
    a.head_dim = hd; a.scale = 0.5f; a.out = out.data();
    kata::attention_decode(a, scratch.data());
    for (size_t h = 0; h < n_heads; ++h)
        for (size_t i = 0; i < hd; ++i)
            EXPECT_NEAR(out[h * hd + i], v[(h / (n_heads / n_kv_heads)) * hd + i], 1e-6f);
}

TEST(Attn, EqualScoresGiveMeanOfV) {
    // Identical k for every position => uniform softmax => out is the mean of v.
    // This pins the normalisation; a missing 1/sum shows up as a factor of n_kv.
    const size_t n_heads = 2, n_kv_heads = 2, hd = 16, n_kv = 32;
    auto q = rnd(n_heads * hd, 7);
    auto k = rnd(n_kv_heads * hd, 8);
    std::vector<float> kk(n_kv * n_kv_heads * hd);
    for (size_t t = 0; t < n_kv; ++t)
        for (size_t i = 0; i < n_kv_heads * hd; ++i) kk[t * n_kv_heads * hd + i] = k[i];
    auto v = rnd(n_kv * n_kv_heads * hd, 9);
    std::vector<float> out(n_heads * hd), s(n_heads / n_kv_heads * n_kv + 8);
    kata::AttnArgs a{};
    a.q = q.data(); a.k_cache = kk.data(); a.v_cache = v.data();
    a.n_kv = n_kv; a.n_heads = n_heads; a.n_kv_heads = n_kv_heads; a.head_dim = hd;
    a.scale = 1.0f / std::sqrt((float)hd); a.out = out.data();
    kata::attention_decode(a, s.data());
    for (size_t h = 0; h < n_heads; ++h) {
        const size_t kh = h / (n_heads / n_kv_heads);
        for (size_t i = 0; i < hd; ++i) {
            float mean = 0.0f;
            for (size_t t = 0; t < n_kv; ++t) mean += v[t * n_kv_heads * hd + kh * hd + i];
            mean /= (float)n_kv;
            EXPECT_NEAR(out[h * hd + i], mean, 2e-5f * std::max(1.0f, std::fabs(mean)));
        }
    }
}

TEST(Attn, KvSlotIsDisjointAndPositionMajor) {
    const size_t n_kv_heads = 4, hd = 8, n_slots = 5;
    std::vector<float> cache(n_slots * n_kv_heads * hd, 0.0f);
    for (size_t p = 0; p < n_slots; ++p)
        for (size_t i = 0; i < n_kv_heads * hd; ++i)
            kata::kv_slot(cache.data(), p, n_kv_heads, hd)[i] = (float)(p * 100 + i);
    for (size_t p = 0; p < n_slots; ++p)
        for (size_t i = 0; i < n_kv_heads * hd; ++i)
            EXPECT_EQ(kata::kv_slot(cache.data(), p, n_kv_heads, hd)[i], (float)(p * 100 + i));
}
TEST(Attn, GroupedMatchesPlainDecode) {
    // The grouped kernel is a separate code path with its own scratch layout, so
    // it has to agree with the plain one on every shape it accepts, and fall back
    // unchanged on every shape it does not.
    if (kata::cpu_isa() == kata::Isa::Scalar) GTEST_SKIP() << "no AVX2 on this host";
    struct C { size_t nh, nkv, hd, n_kv; };
    const C cases[] = {
        {32, 8, 64, 1}, {32, 8, 64, 37}, {32, 8, 64, 256}, {32, 8, 64, 4097},
        {8, 8, 64, 129},   // G=1: must fall back
        {8, 2, 96, 200},   // G=4 but hd=96: must fall back
        {8, 2, 32, 63},    // G=4, hd=32, odd n_kv
        {16, 4, 16, 511},
    };
    for (const auto & c : cases) {
        auto q = rnd(c.nh * c.hd, 11);
        auto k = rnd(c.n_kv * c.nkv * c.hd, 12);
        auto v = rnd(c.n_kv * c.nkv * c.hd, 13);
        kata::AttnArgs a{};
        a.q = q.data(); a.k_cache = k.data(); a.v_cache = v.data();
        a.n_kv = c.n_kv; a.n_heads = c.nh; a.n_kv_heads = c.nkv;
        a.head_dim = c.hd; a.scale = 1.0f / std::sqrt((float)c.hd);
        std::vector<float> want(c.nh * c.hd), got(c.nh * c.hd);
        a.out = want.data();
        std::vector<float> s1(kata::attention_scratch_floats(a));
        kata::attention_decode(a, s1.data());
        a.out = got.data();
        std::vector<float> s2(kata::attention_scratch_floats_grouped(a));
        kata::attention_decode_grouped(a, s2.data());
        for (size_t i = 0; i < want.size(); ++i) {
            ASSERT_FALSE(std::isnan(got[i])) << "heads=" << c.nh << "/" << c.nkv
                                             << " hd=" << c.hd << " n_kv=" << c.n_kv << " i=" << i;
            EXPECT_NEAR(got[i], want[i], 5e-4f * std::max(1.0f, std::fabs(want[i])))
                << "heads=" << c.nh << "/" << c.nkv << " hd=" << c.hd
                << " n_kv=" << c.n_kv << " i=" << i;
        }
    }
}

TEST(Attn, ExtremeScoresMatchRefAfterClamp) {
    // The vector path replaces exp with a polynomial and clamps the exponent
    // range. Wild score spans are normal with untrained or saturated attention,
    // and a clamp that is too narrow shows up here as NaN or a wrong winner.
    if (kata::cpu_isa() == kata::Isa::Scalar) GTEST_SKIP() << "no AVX2 on this host";
    const size_t n_heads = 4, n_kv_heads = 2, hd = 32;
    std::mt19937 g(99);
    std::normal_distribution<float> nd(0.0f, 40.0f);
    for (size_t n_kv : {size_t(8), size_t(257)}) {
        std::vector<float> k(n_kv * n_kv_heads * hd), v(n_kv * n_kv_heads * hd), q(n_heads * hd);
        for (auto & x : k) x = nd(g);
        for (auto & x : v) x = nd(g);
        for (auto & x : q) x = nd(g);
        kata::AttnArgs a{};
        a.q = q.data(); a.k_cache = k.data(); a.v_cache = v.data();
        a.n_kv = n_kv; a.n_heads = n_heads; a.n_kv_heads = n_kv_heads; a.head_dim = hd;
        a.scale = 1.0f / std::sqrt((float)hd);
        std::vector<float> want(n_heads * hd), got(n_heads * hd);
        std::vector<float> s(kata::attention_scratch_floats(a));
        a.out = want.data();
        kata::attention_decode_ref(a, s.data());
        a.out = got.data();
        kata::attention_decode(a, s.data());
        for (size_t i = 0; i < want.size(); ++i) {
            ASSERT_FALSE(std::isnan(got[i])) << "i=" << i << " n_kv=" << n_kv;
            EXPECT_NEAR(got[i], want[i], 5e-4f * std::max(1.0f, std::fabs(want[i])))
                << "n_kv=" << n_kv << " i=" << i;
        }
    }
}

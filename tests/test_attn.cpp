#include <gtest/gtest.h>

#include "kata/attn.h"
#include "kata/cpu.h"
#include "kata/pool.h"
#include <chrono>
#include "kata/simd.h"

#include <cmath>
#include <random>
#include <vector>

namespace {
// Position-major [position][kv_head][head_dim] -> head-major
// [kv_head][position][head_dim], the layout the engine runs.
std::vector<float> to_head_major(const std::vector<float> & pm, size_t n_kv, size_t n_kv_heads,
                                 size_t hd) {
    std::vector<float> hm(pm.size());
    for (size_t t = 0; t < n_kv; ++t)
        for (size_t h = 0; h < n_kv_heads; ++h)
            for (size_t d = 0; d < hd; ++d)
                hm[(h * n_kv + t) * hd + d] = pm[(t * n_kv_heads + h) * hd + d];
    return hm;
}
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

TEST(Attn, KtWriteRoundTripsThroughTheTransposedLayout) {
    const size_t n_kv = 200, n_kv_heads = 8, hd = 64;
    std::vector<float> k(n_kv * n_kv_heads * hd);
    for (size_t i = 0; i < k.size(); ++i) k[i] = (float)(i % 97) - 40.0f;
    std::vector<float> kt(kata::kt_floats(n_kv, n_kv_heads, hd));
    for (size_t t = 0; t < n_kv; ++t)
        for (size_t h = 0; h < n_kv_heads; ++h)
            kata::kt_write(kt.data(), t, h, n_kv, hd, &k[t * n_kv_heads * hd + h * hd]);
    // Every element must be reachable from the transposed cache at its own slot.
    for (size_t t = 0; t < n_kv; ++t)
        for (size_t h = 0; h < n_kv_heads; ++h) {
            const float * slot = kata::kt_slot(kt.data(), t, h, n_kv, hd);
            for (size_t d = 0; d < hd; ++d)
                ASSERT_EQ(slot[d * kata::KT_BLK], k[t * n_kv_heads * hd + h * hd + d])
                    << "t=" << t << " h=" << h << " d=" << d;
        }
}

TEST(Attn, KtDecodeMatchesPositionMajorDecode) {
    for (size_t n_kv : {1u, 63u, 64u, 65u, 129u, 200u}) {
        const size_t n_heads = 32, n_kv_heads = 8, hd = 64;
        std::mt19937 g(7);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        std::vector<float> q(n_heads * hd), k(n_kv * n_kv_heads * hd), v(n_kv * n_kv_heads * hd);
        for (auto & x : q) x = nd(g);
        for (auto & x : k) x = nd(g);
        for (auto & x : v) x = nd(g);
        std::vector<float> kt(kata::kt_floats(n_kv, n_kv_heads, hd));
        for (size_t t = 0; t < n_kv; ++t)
            for (size_t h = 0; h < n_kv_heads; ++h)
                kata::kt_write(kt.data(), t, h, n_kv, hd, &k[t * n_kv_heads * hd + h * hd]);

        kata::AttnArgs a{};
        a.q = q.data(); a.k_cache = k.data(); a.v_cache = v.data();
        a.n_kv = n_kv; a.n_heads = n_heads; a.n_kv_heads = n_kv_heads; a.head_dim = hd;
        a.scale = 1.0f / std::sqrt((float)hd);
        std::vector<float> want(n_heads * hd), got(n_heads * hd);
        std::vector<float> s(kata::attention_scratch_floats_grouped(a));
        a.out = want.data();
        kata::attention_decode_grouped(a, s.data());
        a.out = got.data();
        kata::attention_decode_grouped_kt(a, s.data(), kt.data());
        for (size_t i = 0; i < want.size(); ++i) {
            ASSERT_FALSE(std::isnan(got[i])) << "i=" << i << " n_kv=" << n_kv;
            EXPECT_NEAR(got[i], want[i], 5e-4f * std::max(1.0f, std::fabs(want[i])))
                << "n_kv=" << n_kv << " i=" << i;
        }
    }
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2,fma")))
static void check_exp256(float x, float * out) {
    __m256 vx = _mm256_set1_ps(x);
    __m256 ve = kata::exp256_ps(vx);
    _mm256_storeu_ps(out, ve);
}

__attribute__((target("avx512f,avx512vl,avx512dq,fma")))
static void check_exp512(float x, float * out) {
    __m512 vx = _mm512_set1_ps(x);
    __m512 ve = kata::exp512_ps(vx);
    _mm512_storeu_ps(out, ve);
}

TEST(SimdExp, Exp256MatchesStdExp) {
    if (kata::cpu_isa() == kata::Isa::Scalar) GTEST_SKIP() << "no AVX2 on this host";
    alignas(32) float out[8];
    for (float x = -20.0f; x <= 20.0f; x += 0.125f) {
        check_exp256(x, out);
        float expected = std::exp(x);
        for (int i = 0; i < 8; ++i) {
            float err = std::abs(out[i] - expected);
            float rel = err / std::max(expected, 1e-12f);
            EXPECT_LT(rel, 2e-5f) << "x=" << x;
        }
    }
}

TEST(SimdExp, Exp512MatchesStdExp) {
    auto isa = kata::cpu_isa();
    if (isa != kata::Isa::Avx512 && isa != kata::Isa::Avx512Vnni) GTEST_SKIP() << "no AVX-512 on this host";
    alignas(64) float out[16];
    for (float x = -20.0f; x <= 20.0f; x += 0.125f) {
        check_exp512(x, out);
        float expected = std::exp(x);
        for (int i = 0; i < 16; ++i) {
            float err = std::abs(out[i] - expected);
            float rel = err / std::max(expected, 1e-12f);
            EXPECT_LT(rel, 2e-5f) << "x=" << x;
        }
    }
}
#endif

TEST(Attn, MtSplitMatchesSingleThreaded) {
    // Kv heads own disjoint query heads and output slices, so splitting them
    // across workers must be bit-identical, not merely close.
    if (kata::cpu_isa() == kata::Isa::Scalar) GTEST_SKIP() << "no AVX2 on this host";
    struct C { size_t nh, nkv, hd, n_kv; };
    const C cases[] = {
        {32, 8, 64, 1}, {32, 8, 64, 37}, {32, 8, 64, 256}, {32, 8, 64, 4097},
        {32, 8, 32, 63}, {16, 4, 16, 511}, {8, 2, 96, 200}, {8, 8, 64, 129},
    };
    for (const auto & c : cases) {
        auto q = rnd(c.nh * c.hd, 31);
        auto k = rnd(c.n_kv * c.nkv * c.hd, 32);
        auto v = rnd(c.n_kv * c.nkv * c.hd, 33);
        kata::AttnArgs a{};
        a.q = q.data(); a.k_cache = k.data(); a.v_cache = v.data();
        a.n_kv = c.n_kv; a.n_heads = c.nh; a.n_kv_heads = c.nkv;
        a.head_dim = c.hd; a.scale = 1.0f / std::sqrt((float)c.hd);
        std::vector<float> want(c.nh * c.hd), got(c.nh * c.hd);

        a.out = want.data();
        std::vector<float> s0(kata::attention_scratch_floats_grouped(a));
        kata::attention_decode_grouped(a, s0.data());

        for (unsigned nt : {2u, 3u, 8u}) {
            kata::ThreadPool pool(nt);
            a.out = got.data();
            std::vector<float> s1(kata::attention_scratch_floats_grouped_mt(a, nt));
            kata::attention_decode_grouped_mt(a, s1.data(), pool);
            for (size_t i = 0; i < want.size(); ++i)
                ASSERT_EQ(got[i], want[i])
                    << "nt=" << nt << " heads=" << c.nh << "/" << c.nkv << " hd=" << c.hd
                    << " n_kv=" << c.n_kv << " i=" << i;
        }
    }
}

TEST(Attn, PhaseProfilerCoversTheWholeKernel) {
    // The profiler used to re-implement the passes per head, so it measured code
    // the grouped kernel no longer ran. It must now account for the real work:
    // phases sum to within a few percent of the whole call.
    if (kata::cpu_isa() == kata::Isa::Scalar || kata::cpu_isa() == kata::Isa::Avx2)
        GTEST_SKIP() << "grouped path needs AVX-512";
    const size_t nh = 32, nkv = 8, hd = 64, n_kv = 2048;
    auto q = rnd(nh * hd, 61);
    auto k = rnd(n_kv * nkv * hd, 62);
    auto v = rnd(n_kv * nkv * hd, 63);
    kata::AttnArgs a{};
    a.q = q.data(); a.k_cache = k.data(); a.v_cache = v.data();
    a.n_kv = n_kv; a.n_heads = nh; a.n_kv_heads = nkv;
    a.head_dim = hd; a.scale = 1.0f / std::sqrt((float)hd);
    std::vector<float> out(nh * hd), sc(kata::attention_scratch_floats_grouped(a));
    a.out = out.data();

    double phases[3] = {0, 0, 0};
    kata::attention_decode_phases(a, sc.data(), phases);
    double sum = phases[0] + phases[1] + phases[2];
    ASSERT_GT(sum, 0.0) << "profiler reported no work at all";
    ASSERT_GT(phases[0], 0.0) << "score phase never measured";
    ASSERT_GT(phases[2], 0.0) << "v phase never measured";

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i) kata::attention_decode_grouped(a, sc.data());
    const double whole =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 5;
    // Phases exclude only the loop overhead, so they should be close to the call.
    EXPECT_LT(sum, whole * 1.5);
    EXPECT_GT(sum, whole * 0.5);
}

TEST(Attn, HeadMajorCacheMatchesPositionMajor) {
    // The engine keeps kv as [kv_head][position][head_dim] so one worker streams
    // one contiguous history. Strides must not change a single number: this is a
    // layout change, not a maths change.
    if (kata::cpu_isa() == kata::Isa::Scalar) GTEST_SKIP() << "no AVX2 on this host";
    const Case cases[] = {
        {32, 8, 64, 1}, {32, 8, 64, 37}, {32, 8, 64, 256}, {32, 8, 64, 4097},
        {8, 2, 96, 200}, {8, 2, 32, 63},
    };
    for (const auto & c : cases) {
        auto q = rnd(c.n_heads * c.head_dim, 21);
        auto k_pm = rnd(c.n_kv * c.n_kv_heads * c.head_dim, 22);
        auto v_pm = rnd(c.n_kv * c.n_kv_heads * c.head_dim, 23);
        auto k_hm = to_head_major(k_pm, c.n_kv, c.n_kv_heads, c.head_dim);
        auto v_hm = to_head_major(v_pm, c.n_kv, c.n_kv_heads, c.head_dim);

        kata::AttnArgs a{};
        a.q = q.data();
        a.n_kv = c.n_kv; a.n_heads = c.n_heads; a.n_kv_heads = c.n_kv_heads;
        a.head_dim = c.head_dim; a.scale = 1.0f / std::sqrt((float)c.head_dim);
        std::vector<float> want(c.n_heads * c.head_dim), got(c.n_heads * c.head_dim);

        a.k_cache = k_pm.data(); a.v_cache = v_pm.data(); a.out = want.data();
        std::vector<float> s1(kata::attention_scratch_floats_grouped(a));
        kata::attention_decode_grouped(a, s1.data());

        a.k_cache = k_hm.data(); a.v_cache = v_hm.data(); a.out = got.data();
        a.stride_pos = c.head_dim;
        a.stride_head = c.n_kv * c.head_dim;
        std::vector<float> s2(kata::attention_scratch_floats_grouped(a));
        kata::attention_decode_grouped(a, s2.data());

        for (size_t i = 0; i < want.size(); ++i)
            ASSERT_EQ(got[i], want[i]) << "heads=" << c.n_heads << "/" << c.n_kv_heads
                                      << " hd=" << c.head_dim << " n_kv=" << c.n_kv << " i=" << i;
    }
}

TEST(Attn, WriteKvHeadMajorRoundTrips) {
    const size_t n_kv_heads = 4, hd = 8, n_ctx = 6;
    for (size_t pos : {size_t(0), size_t(1), size_t(5)}) {
        std::vector<float> k(n_kv_heads * hd), v(n_kv_heads * hd);
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = (float)(pos * 100 + i + 1);   // never 0, so "untouched" is testable
            v[i] = -(float)(pos * 100 + i + 1);
        }
        std::vector<float> km(n_kv_heads * n_ctx * hd, 0.0f), vm(n_kv_heads * n_ctx * hd, 0.0f);
        kata::write_kv_head_major(km.data(), vm.data(), pos, n_ctx, n_kv_heads, hd, k.data(), v.data());
        for (size_t h = 0; h < n_kv_heads; ++h)
            for (size_t d = 0; d < hd; ++d) {
                const size_t at = (h * n_ctx + pos) * hd + d;
                EXPECT_EQ(km[at], k[h * hd + d]);
                EXPECT_EQ(vm[at], v[h * hd + d]);
            }
        // Every other slot must be untouched, or decode would read stale tokens.
        size_t touched = 0;
        for (float x : km) touched += x != 0.0f ? 1 : 0;
        EXPECT_EQ(touched, n_kv_heads * hd);
    }
}

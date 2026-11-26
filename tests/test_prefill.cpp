#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "kata/attn.h"
#include "kata/prefill.h"
#include "kata/pool.h"

using namespace kata;

namespace {

// Builds a kv cache whose rows are random, then checks prefill against the
// already-verified decode kernel, which is the oracle: decode at position
// n_kv_begin + t must equal prefill row t when both read the same cache.
struct Case {
    size_t n_tok, n_kv_begin, n_heads, n_kv_heads, head_dim;
};

void check(const Case & c) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t kv = c.n_kv_heads * c.head_dim;
    const size_t n_kv_total = c.n_kv_begin + c.n_tok;
    std::vector<float> k(n_kv_total * kv), v(n_kv_total * kv);
    for (auto & x : k) x = nd(rng);
    for (auto & x : v) x = nd(rng);
    std::vector<float> q(c.n_tok * c.n_heads * c.head_dim);
    for (auto & x : q) x = nd(rng);
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));

    std::vector<float> fast(c.n_tok * c.n_heads * c.head_dim);
    std::vector<float> ref(fast.size());
    attention_prefill(q.data(), c.n_tok, k.data(), v.data(), c.n_kv_begin,
                      c.n_heads, c.n_kv_heads, c.head_dim, scale, fast.data());
    attention_prefill_ref(q.data(), c.n_tok, k.data(), v.data(), c.n_kv_begin,
                          c.n_heads, c.n_kv_heads, c.head_dim, scale, ref.data());

    float maxdiff = 0.0f;
    for (size_t i = 0; i < fast.size(); ++i) maxdiff = std::max(maxdiff, std::fabs(fast[i] - ref[i]));
    EXPECT_LT(maxdiff, 2e-5f) << "tok " << c.n_tok << " begin " << c.n_kv_begin
                              << " heads " << c.n_heads << "/" << c.n_kv_heads
                              << " hd " << c.head_dim;

    // And against decode, one row at a time.
    std::vector<float> dec(c.n_heads * c.head_dim);
    AttnArgs a{};
    a.k_cache = k.data();
    a.v_cache = v.data();
    a.n_heads = c.n_heads;
    a.n_kv_heads = c.n_kv_heads;
    a.head_dim = c.head_dim;
    a.scale = scale;
    a.out = dec.data();
    // Scratch is sized by n_kv, so ask for the largest cache this case reaches.
    a.n_kv = n_kv_total;
    std::vector<float> scratch(attention_scratch_floats_grouped(a));
    for (size_t t = 0; t < c.n_tok; ++t) {
        a.q = q.data() + t * c.n_heads * c.head_dim;
        a.n_kv = c.n_kv_begin + t + 1;
        attention_decode_grouped(a, scratch.data());
        for (size_t i = 0; i < dec.size(); ++i) {
            const float want = ref[t * c.n_heads * c.head_dim + i];
            ASSERT_LT(std::fabs(dec[i] - want), 2e-5f) << "token " << t << " element " << i;
        }
    }
}

} // namespace

TEST(Prefill, MatchesRefAcrossShapes) {
    check({1, 0, 32, 8, 64});
    check({5, 0, 32, 8, 64});
    check({5, 37, 32, 8, 64});
    check({17, 3, 8, 8, 64});
    check({3, 11, 8, 2, 96});
    check({4, 5, 4, 1, 80});
    check({6, 2, 4, 2, 60});
    check({2, 9, 2, 1, 8});
    check({2, 7, 8, 2, 128});
    check({2, 3, 4, 1, 256});
    check({3, 5, 6, 2, 64});
    check({3, 5, 3, 1, 96});
    check({2, 4, 6, 2, 128});
    check({8, 2048, 32, 8, 64});
}

TEST(Prefill, FirstRowOfEmptyCacheIsJustV) {
    const size_t heads = 4, kvh = 2, hd = 16;
    std::vector<float> k(kvh * hd), v(kvh * hd), q(heads * hd);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    for (auto & x : k) x = nd(rng);
    for (auto & x : v) x = nd(rng);
    for (auto & x : q) x = nd(rng);
    std::vector<float> out(heads * hd);
    attention_prefill(q.data(), 1, k.data(), v.data(), 0, heads, kvh, hd, 0.5f, out.data());
    for (size_t h = 0; h < heads; ++h)
        for (size_t d = 0; d < hd; ++d)
            ASSERT_NEAR(out[h * hd + d], v[(h / (heads / kvh)) * hd + d], 1e-6f);
}

TEST(Prefill, AttentionRowsDoNotSeeTheFuture) {
    // Row 0 of a 4-token tile must ignore the last three kv rows entirely.
    const size_t heads = 8, kvh = 2, hd = 64, n_tok = 4;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd;
    const size_t kv = kvh * hd;
    std::vector<float> k(n_tok * kv), v(n_tok * kv);
    for (auto & x : k) x = nd(rng);
    for (auto & x : v) x = nd(rng);
    std::vector<float> q(n_tok * heads * hd);
    for (auto & x : q) x = nd(rng);
    std::vector<float> a(n_tok * heads * hd), b(a.size());
    attention_prefill(q.data(), n_tok, k.data(), v.data(), 0, heads, kvh, hd, 0.125f, a.data());
    for (size_t i = kv; i < k.size(); ++i) { k[i] = 1e6f; v[i] = -1e6f; }
    attention_prefill(q.data(), n_tok, k.data(), v.data(), 0, heads, kvh, hd, 0.125f, b.data());
    for (size_t i = 0; i < heads * hd; ++i) ASSERT_FLOAT_EQ(a[i], b[i]);
}
TEST(Prefill, MtSplitMatchesSingleThreaded) {
    // The MT split is by query row and every row reads the same kv prefix, so
    // it must be bit-identical, not merely close. n_tok deliberately does not
    // divide evenly by the worker count.
    const size_t heads = 8, kvh = 2, hd = 64, n_kv_begin = 7;
    for (size_t n_tok : {1u, 3u, 7u, 16u, 37u}) {
        std::mt19937 rng(21);
        std::normal_distribution<float> nd;
        const size_t kv = kvh * hd, total_kv = n_kv_begin + n_tok;
        std::vector<float> k(total_kv * kv), v(total_kv * kv);
        for (auto & x : k) x = nd(rng);
        for (auto & x : v) x = nd(rng);
        std::vector<float> q(n_tok * heads * hd);
        for (auto & x : q) x = nd(rng);
        std::vector<float> a(n_tok * heads * hd), b(a.size());
        attention_prefill(q.data(), n_tok, k.data(), v.data(), n_kv_begin, heads, kvh, hd, 0.125f, a.data());
        for (unsigned nt : {2u, 3u, 8u}) {
            kata::ThreadPool pool(nt);
            attention_prefill_mt(q.data(), n_tok, k.data(), v.data(), n_kv_begin, heads, kvh, hd,
                                 0.125f, b.data(), pool);
            for (size_t i = 0; i < a.size(); ++i) ASSERT_FLOAT_EQ(a[i], b[i]) << "n_tok=" << n_tok << " workers=" << nt;
        }
    }
}

TEST(Prefill, WeightedPartitionBalancing) {
    const size_t n_tok = 544, n_kv_begin = 0;
    const unsigned nw = 16;
    const auto bounds = kata::prefill_weighted_partition_bounds(n_tok, n_kv_begin, nw);

    ASSERT_EQ(bounds.size(), nw + 1);
    EXPECT_EQ(bounds[0], 0u);
    EXPECT_EQ(bounds[nw], n_tok);

    std::vector<double> work(nw);
    double total_work = 0.0;
    for (unsigned w = 0; w < nw; ++w) {
        ASSERT_LE(bounds[w], bounds[w + 1]);
        double w_w = 0.0;
        for (size_t t = bounds[w]; t < bounds[w + 1]; ++t) {
            w_w += static_cast<double>(n_kv_begin + t + 1);
        }
        work[w] = w_w;
        total_work += w_w;
    }

    const double expected_work = total_work / nw;
    double max_work = work[0], min_work = work[0];
    for (unsigned w = 0; w < nw; ++w) {
        max_work = std::max(max_work, work[w]);
        min_work = std::min(min_work, work[w]);
        // Every thread should be within 10% of the target work
        EXPECT_NEAR(work[w], expected_work, expected_work * 0.10) << "thread " << w;
    }
    // Balance ratio max/min should be < 1.15 (vs 30x unweighted!)
    EXPECT_LT(max_work / min_work, 1.15);
}

TEST(Prefill, WeightedPartitionMonotonicityAndBounds) {
    for (size_t n_tok : {0u, 1u, 2u, 5u, 16u, 37u, 128u, 544u, 2048u}) {
        for (size_t n_kv_begin : {0u, 1u, 10u, 512u, 2048u, 25000u}) {
            for (unsigned nw : {1u, 2u, 4u, 8u, 16u, 32u}) {
                const auto b = kata::prefill_weighted_partition_bounds(n_tok, n_kv_begin, nw);
                ASSERT_EQ(b.size(), nw + 1);
                EXPECT_EQ(b[0], 0u);
                EXPECT_EQ(b[nw], n_tok);
                for (unsigned w = 0; w < nw; ++w) {
                    EXPECT_LE(b[w], b[w + 1]);
                    EXPECT_LE(b[w + 1], n_tok);
                }
            }
        }
    }
}

TEST(Prefill, HeadMajorCacheMatchesPositionMajor) {
    // Same layout contract as the decode test: head-major is a layout, not maths.
    struct C { size_t n_tok, n_kv_begin, n_heads, n_kv_heads, head_dim; };
    const C cases[] = {
        {8, 0, 32, 8, 64}, {7, 5, 32, 8, 64}, {3, 40, 8, 2, 96}, {5, 3, 8, 2, 32},
    };
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.0f, 0.7f);
    for (const auto & c : cases) {
        const size_t kv = c.n_kv_heads * c.head_dim;
        const size_t n_kv = c.n_kv_begin + c.n_tok;
        std::vector<float> k(n_kv * kv), v(n_kv * kv), q(c.n_tok * c.n_heads * c.head_dim);
        for (auto & x : k) x = nd(rng);
        for (auto & x : v) x = nd(rng);
        for (auto & x : q) x = nd(rng);
        std::vector<float> kh(n_kv * kv), vh(n_kv * kv);
        for (size_t t = 0; t < n_kv; ++t)
            for (size_t h = 0; h < c.n_kv_heads; ++h)
                for (size_t d = 0; d < c.head_dim; ++d) {
                    kh[(h * n_kv + t) * c.head_dim + d] = k[(t * c.n_kv_heads + h) * c.head_dim + d];
                    vh[(h * n_kv + t) * c.head_dim + d] = v[(t * c.n_kv_heads + h) * c.head_dim + d];
                }
        const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
        std::vector<float> want(q.size()), got(q.size());
        attention_prefill(q.data(), c.n_tok, k.data(), v.data(), c.n_kv_begin, c.n_heads,
                          c.n_kv_heads, c.head_dim, scale, want.data());
        attention_prefill(q.data(), c.n_tok, kh.data(), vh.data(), c.n_kv_begin, c.n_heads,
                          c.n_kv_heads, c.head_dim, scale, got.data(), c.head_dim,
                          n_kv * c.head_dim);
        for (size_t i = 0; i < want.size(); ++i)
            ASSERT_EQ(got[i], want[i]) << "n_tok=" << c.n_tok << " n_kv_begin=" << c.n_kv_begin
                                      << " hd=" << c.head_dim << " i=" << i;
    }
}

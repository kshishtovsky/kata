#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kata/sampler.h"

namespace {

// A logit array where `best` dominates, so greedy and top_k=1 must agree.
std::vector<float> peaked(size_t n, size_t best, float margin = 10.0f) {
    std::vector<float> l(n, 0.0f);
    l[best] = margin;
    return l;
}

}  // namespace

TEST(Sampler, GreedyTakesTheArgmax) {
    kata::SamplerConfig c;
    c.temp = 0.0f;
    kata::Sampler s(c);
    auto l = peaked(8, 5);
    for (int i = 0; i < 50; ++i) EXPECT_EQ(s.sample(l.data(), l.size(), {}), 5u);
}

TEST(Sampler, NegativeTemperatureIsGreedy) {
    kata::SamplerConfig c;
    c.temp = -1.0f;
    kata::Sampler s(c);
    auto l = peaked(4, 2);
    EXPECT_EQ(s.sample(l.data(), l.size(), {}), 2u);
}

TEST(Sampler, TopKOneMatchesGreedy) {
    kata::SamplerConfig c;
    c.temp = 1.0f;
    c.top_k = 1;
    c.top_p = 1.0f;
    kata::Sampler s(c);
    auto l = peaked(16, 11);
    for (int i = 0; i < 50; ++i) EXPECT_EQ(s.sample(l.data(), l.size(), {}), 11u);
}

TEST(Sampler, ZeroTopKIsDisabled) {
    // With top_k disabled and a high temperature, a peak of only 3 over a flat
    // 64-wide field must sometimes lose.
    kata::SamplerConfig c;
    c.temp = 1.0f;
    c.top_k = 0;
    c.top_p = 1.0f;
    c.seed = 7;
    kata::Sampler s(c);
    auto l = peaked(64, 0, 3.0f);
    bool saw_other = false;
    for (int i = 0; i < 2000 && !saw_other; ++i) saw_other = s.sample(l.data(), l.size(), {}) != 0;
    EXPECT_TRUE(saw_other);
}

TEST(Sampler, SamplingIsDeterministicForASeed) {
    kata::SamplerConfig c;
    c.temp = 1.0f;
    c.top_k = 0;
    c.top_p = 1.0f;
    c.seed = 123;
    auto l = peaked(32, 3, 1.0f);
    kata::Sampler a(c), b(c);
    for (int i = 0; i < 100; ++i) EXPECT_EQ(a.sample(l.data(), l.size(), {}), b.sample(l.data(), l.size(), {}));
}

TEST(Sampler, TopPKeepsTheNucleus) {
    // Two candidates far apart in logit space: with top_p 0.9 only the first
    // survives, so the second can never be drawn.
    kata::SamplerConfig c;
    c.temp = 1.0f;
    c.top_k = 0;
    c.top_p = 0.9f;
    c.seed = 5;
    kata::Sampler s(c);
    std::vector<float> l(8, -100.0f);
    l[0] = 0.0f;
    l[1] = -20.0f;
    for (int i = 0; i < 200; ++i) EXPECT_EQ(s.sample(l.data(), l.size(), {}), 0u);
}

TEST(Sampler, TopPBelowOneShrinksTheSupport) {
    // Four equal logits, top_p 0.5: only the first two of the nucleus can appear.
    kata::SamplerConfig c;
    c.temp = 1.0f;
    c.top_k = 0;
    c.top_p = 0.5f;
    c.seed = 99;
    kata::Sampler s(c);
    std::vector<float> l(4, 0.0f);
    int seen[4] = {0, 0, 0, 0};
    for (int i = 0; i < 400; ++i) ++seen[s.sample(l.data(), l.size(), {})];
    EXPECT_EQ(seen[2], 0);
    EXPECT_EQ(seen[3], 0);
    EXPECT_GT(seen[0] + seen[1], 0);
}

TEST(Sampler, RepeatPenaltyLowersASeenToken) {
    kata::SamplerConfig c;
    c.repeat_penalty = 4.0f;
    c.temp = 0.0f;
    c.repeat_last_n = 8;
    kata::Sampler s(c);
    // Token 0 leads by 1.0; a 4x penalty on it must hand the argmax to token 1.
    std::vector<float> l = { 5.0f, 4.0f, 0.0f, 0.0f };
    EXPECT_EQ(s.sample(l.data(), l.size(), {}), 0u);
    EXPECT_EQ(s.sample(l.data(), l.size(), { 0 }), 1u) << "greedy path applies the repeat penalty";
    const std::vector<uint32_t> hist = { 0 };
    std::vector<float> scratch = l;
    s.apply_penalty(scratch.data(), scratch.size(), hist);
    EXPECT_FLOAT_EQ(scratch[0], 1.25f);
    EXPECT_FLOAT_EQ(scratch[1], 4.0f);
}

TEST(Sampler, RepeatPenaltyPushesNegativeLogitsUp) {
    kata::SamplerConfig c;
    c.repeat_penalty = 2.0f;
    kata::Sampler s(c);
    std::vector<float> l = { -3.0f, 0.0f };
    s.apply_penalty(l.data(), l.size(), { 0 });
    EXPECT_FLOAT_EQ(l[0], -6.0f);
}

TEST(Sampler, RepeatPenaltyWindowIsRespected) {
    kata::SamplerConfig c;
    c.repeat_penalty = 2.0f;
    c.repeat_last_n = 2;
    kata::Sampler s(c);
    // Token 9 appears only outside the 2-token window, so it must be untouched.
    std::vector<float> l(12, 1.0f);
    s.apply_penalty(l.data(), l.size(), { 9, 1, 2 });
    EXPECT_FLOAT_EQ(l[9], 1.0f);
    EXPECT_FLOAT_EQ(l[1], 0.5f);
    EXPECT_FLOAT_EQ(l[2], 0.5f);
}

TEST(Sampler, OutOfRangeHistoryIsIgnored) {
    kata::SamplerConfig c;
    c.repeat_penalty = 2.0f;
    kata::Sampler s(c);
    std::vector<float> l(4, 1.0f);
    s.apply_penalty(l.data(), l.size(), { 999 });
    for (float x : l) EXPECT_FLOAT_EQ(x, 1.0f);
}

TEST(Sampler, EmptyVocabularyThrows) {
    kata::Sampler s;
    std::vector<float> none;
    EXPECT_THROW(s.sample(none.data(), 0, {}), std::runtime_error);
}

TEST(Sampler, DistributionFollowsTheWeights) {
    // With two candidates at logits 0 and ln(3) the ratio must be about 3:1.
    kata::SamplerConfig c;
    c.temp = 1.0f;
    c.top_k = 0;
    c.top_p = 1.0f;
    c.seed = 2024;
    kata::Sampler s(c);
    std::vector<float> l = { 0.0f, std::log(3.0f) };
    int hits[2] = { 0, 0 };
    const int n = 20000;
    for (int i = 0; i < n; ++i) ++hits[s.sample(l.data(), l.size(), {})];
    const double ratio = double(hits[1]) / double(hits[0]);
    EXPECT_NEAR(ratio, 3.0, 0.25);
}

TEST(Sampler, RepeatPenaltyDoesNotMultiplyForDuplicates) {
    kata::SamplerConfig c;
    c.repeat_penalty = 2.0f;
    c.repeat_last_n = 8;
    kata::Sampler s(c);
    std::vector<float> l = { 4.0f, 1.0f };
    s.apply_penalty(l.data(), l.size(), { 0, 0, 0, 0 });
    // Must be penalized exactly once (4.0 / 2.0 = 2.0), not 4.0 / (2^4) = 0.25
    EXPECT_FLOAT_EQ(l[0], 2.0f);
}

TEST(Sampler, ValidationRejectsNullLogits) {
    kata::Sampler s;
    EXPECT_THROW(s.sample(nullptr, 10, {}), std::invalid_argument);
    EXPECT_THROW(s.apply_penalty(nullptr, 10, {}), std::invalid_argument);
}

TEST(Sampler, ValidationRejectsNaNLogits) {
    kata::Sampler s;
    std::vector<float> l = { 1.0f, NAN, 2.0f };
    EXPECT_THROW(s.sample(l.data(), l.size(), {}), std::runtime_error);
}

TEST(Sampler, ValidationRejectsInvalidParameters) {
    {
        kata::SamplerConfig c;
        c.top_p = 0.0f;
        EXPECT_THROW(kata::Sampler s(c), std::invalid_argument);
    }
    {
        kata::SamplerConfig c;
        c.top_p = 1.5f;
        EXPECT_THROW(kata::Sampler s(c), std::invalid_argument);
    }
    {
        kata::SamplerConfig c;
        c.repeat_penalty = -1.0f;
        EXPECT_THROW(kata::Sampler s(c), std::invalid_argument);
    }
    {
        kata::SamplerConfig c;
        c.temp = NAN;
        EXPECT_THROW(kata::Sampler s(c), std::invalid_argument);
    }
}

TEST(Sampler, ZeroTopKWithTopPMatchesNucleus) {
    kata::SamplerConfig c;
    c.temp = 0.8f;
    c.top_k = 0;
    c.top_p = 0.9f;
    c.seed = 42;
    kata::Sampler s(c);
    std::vector<float> l(1000, 0.0f);
    l[123] = 10.0f;
    l[456] = 9.5f;
    for (int i = 0; i < 50; ++i) {
        uint32_t t = s.sample(l.data(), l.size(), {});
        EXPECT_TRUE(t == 123u || t == 456u);
    }
}


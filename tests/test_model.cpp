// Model-level tests. src/model.cpp is the largest file in the project (904
// lines) and sat at 1.44% coverage, because exercising it needs a GGUF with a
// real architecture. These tests use the smallest compatible model on hand
// (LFM2.5-8B-A1B, ~5 GB) and SKIP cleanly when it is absent, so the suite still
// passes on a machine without it -- silently passing on a machine WITH it is
// what we want; silently skipping while claiming coverage is what we must not
// do, hence the explicit GTEST_SKIP message.
//
// Override the path with KATA_TEST_MODEL=<path.gguf>.
#include "kata/model.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr const char * kDefaultModel = "/home/medovarov/models/gguf/LFM2.5-8B-A1B-Q4_K_M.gguf";

std::string model_path() {
    if (const char * p = std::getenv("KATA_TEST_MODEL")) return p;
    return kDefaultModel;
}

// One shared model: loading 5 GB per test would dominate the suite's runtime,
// and every test below starts from reset() anyway.
kata::Model & shared_model() {
    // Model is non-copyable, so it is held behind a pointer rather than returned
    // by value from the initialiser.
    static auto m = [] {
        auto x = std::make_unique<kata::Model>();
        x->load(model_path(), 2048);
        return x;
    }();
    return *m;
}

bool model_available() { return std::filesystem::exists(model_path()); }

#define REQUIRE_MODEL()                                        \
    if (!model_available())                                     \
    GTEST_SKIP() << "no model at " << model_path()               \
                 << " (set KATA_TEST_MODEL to run these)"

} // namespace

TEST(Model, LoadReportsConsistentDimensions) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    EXPECT_GT(m.n_vocab(), 0u);
    EXPECT_GT(m.n_embd(), 0u);
    EXPECT_GT(m.n_layer(), 0u);
    // A model either has MoE layers or does not; the count must never exceed
    // the layer count, which is the bug that made the routing record empty
    // (EXP-0078).
    EXPECT_LE(m.n_moe_layer_count(), m.n_layer());
}

TEST(Model, RejectsMissingFile) {
    kata::Model m;
    EXPECT_THROW(m.load("/tmp/kata_no_such_model.gguf", 64), std::runtime_error);
}

TEST(Model, ForwardReturnsFiniteLogitsOfVocabularyLength) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    shared_model().reset();
    kata::ModelState st;
    const float * lg = m.forward(1, st);
    ASSERT_NE(lg, nullptr);
    EXPECT_EQ(st.pos, 1u);
    // Every logit finite: a NaN here would silently poison sampling later.
    for (size_t v = 0; v < m.n_vocab(); ++v) {
        ASSERT_TRUE(std::isfinite(lg[v])) << "logit " << v << " is not finite";
    }
}

TEST(Model, ForwardIsDeterministicAcrossReset) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    std::vector<float> first;
    {
        shared_model().reset();
        kata::ModelState st;
        const float * lg = m.forward(1, st);
        first.assign(lg, lg + m.n_vocab());
    }
    shared_model().reset();
    kata::ModelState st;
    const float * lg = m.forward(1, st);
    // reset() must clear the KV cache and the shortconv history; if it does not,
    // the same token in a fresh state produces different logits.
    for (size_t v = 0; v < m.n_vocab(); ++v)
        ASSERT_FLOAT_EQ(lg[v], first[v]) << "logit " << v << " changed after reset";
}

TEST(Model, ForwardRejectsTokenOutsideVocabulary) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    shared_model().reset();
    kata::ModelState st;
    EXPECT_THROW(m.forward(static_cast<uint32_t>(m.n_vocab()), st), std::runtime_error);
}

TEST(Model, ForwardRespectsContextLimit) {
    REQUIRE_MODEL();
    kata::Model m;
    // A context this small makes the boundary reachable in a few hundred
    // forward calls instead of thousands.
    m.load(model_path(), 8);
    shared_model().reset();
    kata::ModelState st;
    for (size_t i = 0; i < 8; ++i) m.forward(1, st);
    EXPECT_EQ(st.pos, 8u);
    EXPECT_THROW(m.forward(1, st), std::runtime_error);
}

TEST(Model, PrefillIsDeterministicAcrossRepeatedRuns) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    const std::vector<uint32_t> toks = { 1, 2, 3, 4, 5 };
    std::vector<float> first(m.n_vocab());
    for (int rep = 0; rep < 2; ++rep) {
        shared_model().reset();
        kata::ModelState st;
        const float * lg = m.prefill(toks.data(), toks.size(), st);
        if (rep == 0) {
            std::copy(lg, lg + m.n_vocab(), first.begin());
        } else {
            // Same code path, so this must be bit-exact; a drift here means
            // state leaked between runs (KV, shortconv history, a pool race).
            for (size_t v = 0; v < m.n_vocab(); ++v)
                ASSERT_FLOAT_EQ(lg[v], first[v]) << "logit " << v << " drifted";
        }
    }
}

TEST(Model, PrefillAndDecodeAgreeToWithinAGrossTolerance) {
    REQUIRE_MODEL();
    // Decode runs GEMV, prefill runs a chunked GEMM; the two sum floats in
    // different orders, so they are NOT bit-identical and their argmax can
    // differ on a near-tie (measured max|diff| 0.85-2.0 over 40 layers). This
    // test therefore only guards against gross corruption -- a wrong weight,
    // a bad norm, a broken rope -- which moves logits by an order of magnitude
    // more than float order can -- measured worst case on this pair of
    // prompts is 5.7 over a 17.4 peak, so 20 catches corruption and nothing else.
    kata::Model & m = shared_model();
    const std::vector<uint32_t> toks = { 1, 2, 3, 4, 5 };

    std::vector<float> looped;
    {
        shared_model().reset();
        kata::ModelState st;
        const float * lg = nullptr;
        for (uint32_t t : toks) lg = m.forward(t, st);
        looped.assign(lg, lg + m.n_vocab());
    }
    std::vector<float> rows(toks.size() * m.n_vocab());
    {
        shared_model().reset();
        kata::ModelState st;
        m.prefill_logits(toks.data(), toks.size(), st, rows.data());
    }
    const float * batched = rows.data() + (toks.size() - 1) * m.n_vocab();
    for (size_t v = 0; v < m.n_vocab(); ++v) {
        ASSERT_TRUE(std::isfinite(batched[v])) << "prefill logit " << v << " not finite";
        ASSERT_LT(std::abs(looped[v] - batched[v]), 20.0f) << "logit " << v;
    }
}

TEST(Model, PrefillAdvancesPositionByTokenCount) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    shared_model().reset();
    kata::ModelState st;
    const std::vector<uint32_t> toks(7, 5);
    m.prefill(toks.data(), toks.size(), st);
    EXPECT_EQ(st.pos, toks.size());
}

TEST(Model, PrefillRejectsEmptyAndOverflow) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    kata::ModelState st;
    EXPECT_THROW(m.prefill(nullptr, 0, st), std::runtime_error);

    kata::Model small;
    small.load(model_path(), 4);
    shared_model().reset();
    kata::ModelState st2;
    const std::vector<uint32_t> too_many(5, 1);
    EXPECT_THROW(small.prefill(too_many.data(), too_many.size(), st2), std::runtime_error);
}

TEST(Model, PrefillRejectsTokenOutsideVocabulary) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    shared_model().reset();
    kata::ModelState st;
    const std::vector<uint32_t> bad = { 1, static_cast<uint32_t>(m.n_vocab()) };
    EXPECT_THROW(m.prefill(bad.data(), bad.size(), st), std::runtime_error);
}

TEST(Model, PrefillLogitsFillsOneRowPerToken) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    const std::vector<uint32_t> toks = { 10, 11, 12 };
    std::vector<float> out(toks.size() * m.n_vocab(), 0.0f);
    shared_model().reset();
    kata::ModelState st;
    m.prefill_logits(toks.data(), toks.size(), st, out.data());
    EXPECT_EQ(st.pos, toks.size());
    // The last row must be a real distribution, not the zero fill.
    bool any = false;
    for (size_t v = 0; v < m.n_vocab(); ++v) any |= out[(toks.size() - 1) * m.n_vocab() + v] != 0.0f;
    EXPECT_TRUE(any);
}

TEST(Model, PrefillLogitsRejectsEmpty) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    std::vector<float> out(m.n_vocab());
    kata::ModelState st;
    EXPECT_THROW(m.prefill_logits(nullptr, 0, st, out.data()), std::runtime_error);
}

TEST(Model, TruncateToCurrentPositionIsANoop) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    shared_model().reset();
    kata::ModelState st;
    m.prefill(std::vector<uint32_t>(3, 1).data(), 3, st);
    const size_t before = st.pos;
    m.truncate(st, before);
    EXPECT_EQ(st.pos, before);
}

TEST(Model, TruncateRejectsForwardPosition) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    shared_model().reset();
    kata::ModelState st;
    m.prefill(std::vector<uint32_t>(3, 1).data(), 3, st);
    EXPECT_THROW(m.truncate(st, st.pos + 1), std::runtime_error);
}

TEST(Model, TruncateThenReplayReproducesTheSameLogits) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    const std::vector<uint32_t> toks = { 1, 2, 3, 4 };

    std::vector<float> straight;
    {
        shared_model().reset();
        kata::ModelState st;
        m.prefill(toks.data(), toks.size(), st);
        const float * lg = m.forward(9, st);
        straight.assign(lg, lg + m.n_vocab());
    }
    // Same sequence, but the last token is rolled back and replayed. This is
    // exactly what a rejected speculative candidate does, and the shortconv
    // history is the only state that makes it non-trivial.
    std::vector<float> replayed;
    {
        shared_model().reset();
        kata::ModelState st;
        m.prefill(toks.data(), toks.size(), st);
        const size_t mark = st.pos;
        (void)m.forward(9, st);
        m.truncate(st, mark);
        const float * lg = m.forward(9, st);
        replayed.assign(lg, lg + m.n_vocab());
    }
    ASSERT_EQ(straight.size(), replayed.size());
    for (size_t v = 0; v < straight.size(); ++v)
        ASSERT_FLOAT_EQ(straight[v], replayed[v]) << "logit " << v;
}

TEST(Model, TruncateOutsideTheChunkFailsLoudly) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    shared_model().reset();
    kata::ModelState st;
    // Two chunks: position 0 is no longer inside the snapshot window, so the
    // shortconv history for it cannot be restored. Continuing on a stale state
    // would produce plausible garbage; the engine refuses instead.
    m.prefill(std::vector<uint32_t>(1100, 1).data(), 1100, st);
    EXPECT_THROW(m.truncate(st, 0), std::runtime_error);
}

TEST(Model, PerplexityIsFiniteAndRepeatable) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    const std::vector<uint32_t> toks = { 1, 2, 3, 4, 5, 6, 7, 8 };
    size_t counted = 0;
    const double p1 = m.perplexity(toks.data(), toks.size(), 512, &counted);
    EXPECT_EQ(counted, toks.size() - 1);
    EXPECT_TRUE(std::isfinite(p1));
    EXPECT_GT(p1, 1.0);
    const double p2 = m.perplexity(toks.data(), toks.size(), 512);
    EXPECT_DOUBLE_EQ(p1, p2);
}

TEST(Model, PerplexityOfTooShortInputIsZeroAndSmallChunksRejected) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    const std::vector<uint32_t> one = { 1 };
    EXPECT_DOUBLE_EQ(m.perplexity(one.data(), 1, 512), 0.0);
    const std::vector<uint32_t> toks = { 1, 2, 3 };
    EXPECT_THROW(m.perplexity(toks.data(), toks.size(), 1), std::runtime_error);
}

TEST(Model, HeatmapCountsEverySelectionOnce) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    if (m.n_moe_layer_count() == 0) GTEST_SKIP() << "dense model, no routing to count";
    shared_model().reset();
    kata::ModelState st;
    m.heatmap_start();
    const std::vector<uint32_t> toks(4, 3);
    m.prefill(toks.data(), toks.size(), st);
    const std::vector<uint64_t> heat = m.heatmap_take();
    ASSERT_EQ(heat.size(), m.n_layer() * m.n_expert());
    // take() must disarm: a second take on an un-armed map is empty.
    EXPECT_TRUE(m.heatmap_take().empty());
    uint64_t total = 0;
    for (uint64_t v : heat) total += v;
    EXPECT_GT(total, 0u);
}

TEST(Model, RoutingRecordHasOneSlotPerMoEExpertSelection) {
    REQUIRE_MODEL();
    kata::Model & m = shared_model();
    if (m.n_moe_layer_count() == 0) GTEST_SKIP() << "dense model, no routing to record";
    shared_model().reset();
    kata::ModelState st;
    m.routing_record_start();
    m.forward(3, st);
    m.forward(4, st);
    const std::vector<kata::Model::TokenRouting> rec = m.routing_record_take();
    ASSERT_EQ(rec.size(), 2u);
    for (const auto & r : rec) {
        EXPECT_EQ(r.n_moe_layer, m.n_moe_layer_count());
        EXPECT_GT(r.n_expert_used, 0u);
        EXPECT_EQ(r.expert.size(), r.n_moe_layer * r.n_expert_used);
        for (uint32_t e : r.expert) EXPECT_LT(e, m.n_expert());
    }
    // Take disarms, so a later forward records nothing.
    m.forward(5, st);
    EXPECT_TRUE(m.routing_record_take().empty());
}
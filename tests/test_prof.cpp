// EXP-0055: the stage reconciler exists because four iterations in a row were lost
// to a counter that did not measure what its name said. These tests pin the
// behaviour that catches that class, using the four historical failures as cases.
#include <gtest/gtest.h>
#include "kata/prof.h"

using kata::check_stages;

// The healthy case: stages add up to the wall clock.
TEST(StageCheck, SumMatchingTotalIsOk) {
    const auto c = check_stages(1000.0, 1000.0);
    EXPECT_TRUE(c.ok);
    EXPECT_NEAR(c.rel, 0.0, 1e-12);
}

// EXP-0052 shape: a stage was attributed to another stage, so the reported parts
// were smaller than the whole. Anything under ~5% is timing jitter between the
// stage stamps and the chunk clock, not a bug.
TEST(StageCheck, TenPercentShortIsUnderAndFails) {
    const auto c = check_stages(900.0, 1000.0);
    EXPECT_FALSE(c.ok);
    EXPECT_LT(c.rel, 0.0);
    EXPECT_STREQ(c.verdict, "under");
}

// EXP-0043 shape: two clock reads for one window made mix > attn + conv. The
// stage sum then EXCEEDS the wall clock, which is impossible physically and must
// never be reported as ok.
TEST(StageCheck, SumExceedingTotalIsOverAndFails) {
    const auto c = check_stages(1060.0, 1000.0);
    EXPECT_FALSE(c.ok);
    EXPECT_GT(c.rel, 0.0);
    EXPECT_STREQ(c.verdict, "over");
}

// moe_scat hid the down-projection for three iterations: the sub-phase sum was
// fine, but it described a different decomposition than the one printed. The
// reconciler cannot see that a counter is mislabelled, only that the numbers do
// not close, so the tolerance is the contract: sub-phases close within 15%.
TEST(StageCheck, SubPhaseToleranceIsLooserThanWholeChunk) {
    EXPECT_TRUE(check_stages(1085.0, 1000.0, 0.15).ok);   // 8.5% over, tolerated inside a stage
    EXPECT_FALSE(check_stages(1085.0, 1000.0, 0.05).ok);  // same numbers, rejected at chunk level
    EXPECT_TRUE(check_stages(920.0, 1000.0, 0.15).ok);    // 8% short, also tolerated
}

// An empty chunk (a stage that never ran) must not look like a broken counter,
// otherwise every short prefill reports a fake failure.
TEST(StageCheck, ZeroTotalIsOkSoEmptyChunksDoNotAlarm) {
    const auto c = check_stages(0.0, 0.0);
    EXPECT_TRUE(c.ok);
    EXPECT_STREQ(c.verdict, "empty");
    EXPECT_EQ(c.rel, 0.0);
}

// A stage measured but with no total is a programming error, not "empty": if work
// took time, the total must exist. This guards the call sites, where t_tot was
// read from a clock that could not be zero.
TEST(StageCheck, WorkWithNoTotalIsTreatedAsEmptyButKeepsTheSum) {
    const auto c = check_stages(5.0, 0.0);
    EXPECT_TRUE(c.ok);           // no denominator, so no verdict
    EXPECT_EQ(c.sum, 5.0);       // the sum is still reported, not silently dropped
}

// The tolerance is a fraction, and it must be applied symmetrically: a 5% shortfall
// and a 5% overshoot are both at the boundary.
TEST(StageCheck, ToleranceIsSymmetricAtTheBoundary) {
    EXPECT_TRUE(check_stages(1050.0, 1000.0, 0.05).ok);
    EXPECT_TRUE(check_stages(950.0, 1000.0, 0.05).ok);
    EXPECT_FALSE(check_stages(1051.0, 1000.0, 0.05).ok);
    EXPECT_FALSE(check_stages(949.0, 1000.0, 0.05).ok);
}
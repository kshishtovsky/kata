// The routing record is the input to every residency decision (EXP-0078), so a
// silent zero there produces a plausible, wrong placement curve. These checks
// pin the shape and the disarm behaviour without needing a real model file.
#include "kata/model.h"

#include <gtest/gtest.h>

#include <vector>

TEST(Routing, StartsDisarmedAndReturnsNothing) {
    kata::Model m;   // never loaded: no layers, no experts
    EXPECT_TRUE(m.routing_record_take().empty());
}

TEST(Routing, ArmDisarmIsIdempotent) {
    kata::Model m;
    m.routing_record_start();
    m.routing_record_start();
    EXPECT_TRUE(m.routing_record_take().empty());
    // Recording after take() must stay off rather than resume mid-stream.
    EXPECT_TRUE(m.routing_record_take().empty());
}

TEST(Routing, ExpertCountIsReportedBeforeAnyRun) {
    kata::Model m;
    EXPECT_EQ(m.n_expert(), 0u);
    EXPECT_EQ(m.n_moe_layer_count(), 0u);
}

TEST(Routing, TokenRoutingLayoutMatchesTheDeclaredShape) {
    kata::Model::TokenRouting r;
    r.n_moe_layer = 3;
    r.n_expert_used = 4;
    r.expert.assign(r.n_moe_layer * r.n_expert_used, 7u);
    EXPECT_EQ(r.expert.size(), 12u);
    EXPECT_EQ(r.expert[11], 7u);
}

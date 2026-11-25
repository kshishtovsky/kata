#include <gtest/gtest.h>
#include "kata/pool.h"
#include <atomic>
#include <vector>

using namespace kata;

TEST(ThreadPool, SingleThreadRunsInline) {
    ThreadPool p(1);
    EXPECT_EQ(p.size(), 1u);
    int count = 0;
    p.run([&](unsigned id) { EXPECT_EQ(id, 0u); ++count; });
    EXPECT_EQ(count, 1);
}

TEST(ThreadPool, EveryWorkerRunsExactlyOnce) {
    ThreadPool p(4);
    std::atomic<int> hits{0};
    for (int iter = 0; iter < 100; ++iter) {
        hits = 0;
        p.run([&](unsigned) { hits.fetch_add(1, std::memory_order_relaxed); });
        ASSERT_EQ(hits.load(), 4) << "iter " << iter;
    }
}

TEST(ThreadPool, WriteDisjointRangesNoLoss) {
    ThreadPool p(8);
    std::vector<int> v(8000, 0);
    p.run([&](unsigned id) {
        for (int i = id * 1000; i < (int)(id + 1) * 1000; ++i) v[i] = (int)id + 1;
    });
    for (int i = 0; i < 8000; ++i) ASSERT_EQ(v[i], i / 1000 + 1) << "i=" << i;
}

TEST(ThreadPool, DefaultThreadsIsSane) {
    const unsigned n = ThreadPool::default_threads();
    EXPECT_GE(n, 1u);
    EXPECT_LE(n, 64u);
}

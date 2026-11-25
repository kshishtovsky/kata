#include <gtest/gtest.h>
#include "kata/pool.h"
#include "kata/throttle.h"
#include <atomic>
#include <chrono>
#include <thread>
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

// The load guard exists because sustained all-core saturation hard-resets this
// machine; a broken cap would silently turn a "light" benchmark into a hazard.
TEST(Throttle, DefaultsLeaveHeadroom) {
    EXPECT_GE(kata::throttle::max_threads(), 1u);
    EXPECT_LE(kata::throttle::max_threads(), std::max(1u, std::thread::hardware_concurrency()));
    EXPECT_GT(kata::throttle::duty(), 0.0);
    EXPECT_LE(kata::throttle::duty(), 1.0);
}

TEST(Throttle, PauseIsProportionalToWorkDone) {
    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    kata::throttle::pause_since(t0);
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    // 20 ms of work at ~90% duty adds ~2 ms of idle, never a runaway sleep.
    EXPECT_GT(elapsed, std::chrono::milliseconds(20));
    EXPECT_LT(elapsed, std::chrono::milliseconds(500));
}

TEST(Throttle, PoolDefaultRespectsCap) {
    EXPECT_LE(kata::ThreadPool::default_threads(), kata::throttle::max_threads());
}

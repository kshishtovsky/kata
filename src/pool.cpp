#include "kata/pool.h"
#include "kata/throttle.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdio>
#include <chrono>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define KATA_RELAX() _mm_pause()
#else
#define KATA_RELAX() std::this_thread::yield()
#endif

namespace kata {

namespace {
// A prefill step issues thousands of back-to-back fan-outs of a few hundred
// microseconds each (10112 calls for 544 tokens, measured). Parking 15 threads
// on a futex for every one of them left them idle 502 ms of a 2060 ms prefill,
// which is the "wave" visible in a CPU monitor. Spinning for a budget first
// keeps the cores hot across the gap; the budget is bounded so an idle engine
// still sleeps.
// Microseconds a worker burns before parking. Self-tuning: the caller publishes
// 4x the gap it just observed between fan-outs, so a busy pipeline (gaps of tens
// of microseconds) keeps threads hot, while an engine that is thinking or idle
// parks them on the first try. KATA_SPIN pins it for experiments.
unsigned spin_cap() {
    static const unsigned v = [] {
        const char * e = std::getenv("KATA_SPIN");
        return e ? (unsigned)std::strtoul(e, nullptr, 10) : 20000u;
    }();
    return v;
}
} // namespace

struct ThreadPool::Impl {
    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    const std::function<void(unsigned)> * job = nullptr;
    // Atomic, not mutex-protected: at 16 threads the completion path was 15.75%
    // of prefill time in native_queued_spin_lock_slowpath, because every worker
    // grabbed this mutex to decrement a counter only it and 14 peers touched.
    // Now only the last worker takes the lock, and only to wake the caller.
    std::atomic<unsigned> waiting{0};
    // Published under `m` and read without it by spinning workers, so the store
    // is release and the load acquire; `job` rides on that ordering.
    std::atomic<unsigned> generation{0};
    std::atomic<unsigned> sleepers{0};
    std::atomic<unsigned> spin_us{200};
    std::atomic<bool> caller_waiting{false};
    std::chrono::steady_clock::time_point last_call{};
    bool stop = false;

    void worker_loop(unsigned id) {
        unsigned seen = 0;
        for (;;) {
            const std::function<void(unsigned)> * my_job = nullptr;
            const auto spin_end = std::chrono::steady_clock::now() +
                                  std::chrono::microseconds(spin_us.load(std::memory_order_relaxed));
            for (unsigned n = 0;;) {
                const unsigned g = generation.load(std::memory_order_acquire);
                if (g != seen) { my_job = job; seen = g; break; }
                if (++n % 32 != 0) { KATA_RELAX(); continue; }
                for (unsigned i = 0; i < 32; ++i) KATA_RELAX();
                if (std::chrono::steady_clock::now() < spin_end) continue;
                std::unique_lock<std::mutex> lk(m);
                sleepers.fetch_add(1, std::memory_order_relaxed);
                cv_start.wait(lk, [&] {
                    return stop || generation.load(std::memory_order_acquire) != seen;
                });
                sleepers.fetch_sub(1, std::memory_order_relaxed);
                if (stop) return;
                my_job = job;
                seen = generation.load(std::memory_order_acquire);
                break;
            }
            (*my_job)(id);
            if (waiting.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                if (caller_waiting.load(std::memory_order_acquire)) {
                    std::lock_guard<std::mutex> lk(m);
                    cv_done.notify_one();
                }
            }
        }
    }
};

ThreadPool::ThreadPool(unsigned n_threads) : n_(n_threads ? n_threads : 1), impl_(new Impl) {
    impl_->workers.reserve(n_ - 1);
    for (unsigned i = 1; i < n_; ++i) {
        impl_->workers.emplace_back([this, i] { impl_->worker_loop(i); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(impl_->m);
        impl_->stop = true;
    }
    impl_->cv_start.notify_all();
    for (auto & t : impl_->workers) t.join();
    delete impl_;
}

void ThreadPool::run(const std::function<void(unsigned)> & fn) {
    if (n_ == 1) { fn(0); return; }
    {
        const auto now = std::chrono::steady_clock::now();
        const auto gap = std::chrono::duration_cast<std::chrono::microseconds>(
                             now - impl_->last_call).count();
        impl_->last_call = now;
        unsigned want = gap > 0 ? (unsigned)std::min<long long>(gap * 4, spin_cap()) : 200u;
        if (want < 8000) want = 8000;
        const char * e = std::getenv("KATA_SPIN");
        impl_->spin_us.store(e ? (unsigned)std::strtoul(e, nullptr, 10) : want,
                             std::memory_order_relaxed);
    }
    if (impl_->sleepers.load(std::memory_order_relaxed) != 0) {
        std::lock_guard<std::mutex> lk(impl_->m);
        impl_->job = &fn;
        impl_->waiting.store(n_ - 1, std::memory_order_release);
        impl_->generation.fetch_add(1, std::memory_order_release);
        impl_->cv_start.notify_all();
    } else {
        impl_->job = &fn;
        impl_->waiting.store(n_ - 1, std::memory_order_release);
        impl_->generation.fetch_add(1, std::memory_order_release);
    }

    fn(0);   // the caller participates; it is the only thread guaranteed free

    const auto end = std::chrono::steady_clock::now() +
                     std::chrono::microseconds(impl_->spin_us.load(std::memory_order_relaxed));
    for (unsigned n = 0; impl_->waiting.load(std::memory_order_acquire) != 0; ++n) {
        if (n % 32 == 0 && std::chrono::steady_clock::now() >= end) {
            std::unique_lock<std::mutex> lk(impl_->m);
            impl_->caller_waiting.store(true, std::memory_order_release);
            impl_->cv_done.wait(lk, [&] { return impl_->waiting.load(std::memory_order_acquire) == 0; });
            impl_->caller_waiting.store(false, std::memory_order_relaxed);
            return;
        }
        KATA_RELAX();
    }
}

unsigned ThreadPool::default_threads() {
    // physical core count: SMT gives latency hiding, not throughput, for GEMV
    const unsigned want = std::min(std::max(1u, std::thread::hardware_concurrency() / 2),
                                   throttle::max_threads());
    return want;
}

} // namespace kata

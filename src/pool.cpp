#include "kata/pool.h"
#include "kata/throttle.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace kata {

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
    unsigned generation = 0;
    bool stop = false;

    void worker_loop(unsigned id) {
        unsigned seen = 0;
        for (;;) {
            const std::function<void(unsigned)> * my_job = nullptr;
            {
                std::unique_lock<std::mutex> lk(m);
                cv_start.wait(lk, [&] { return stop || generation != seen; });
                if (stop) return;
                seen = generation;
                my_job = job;
            }
            (*my_job)(id);
            if (waiting.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(m);
                cv_done.notify_one();
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
        std::lock_guard<std::mutex> lk(impl_->m);
        impl_->job = &fn;
        impl_->waiting.store(n_ - 1, std::memory_order_relaxed);
        ++impl_->generation;
    }
    impl_->cv_start.notify_all();

    fn(0);   // the caller participates; it is the only thread guaranteed free

    std::unique_lock<std::mutex> lk(impl_->m);
    impl_->cv_done.wait(lk, [&] { return impl_->waiting.load(std::memory_order_acquire) == 0; });
}

unsigned ThreadPool::default_threads() {
    // physical core count: SMT gives latency hiding, not throughput, for GEMV
    const unsigned want = std::min(std::max(1u, std::thread::hardware_concurrency() / 2),
                                   throttle::max_threads());
    return want;
}

} // namespace kata

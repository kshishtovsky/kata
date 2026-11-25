#include "kata/pool.h"

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
    unsigned waiting = 0;       // workers not yet finished with the current job
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
            {
                std::lock_guard<std::mutex> lk(m);
                if (--waiting == 0) cv_done.notify_one();
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
        impl_->waiting = n_ - 1;
        ++impl_->generation;
    }
    impl_->cv_start.notify_all();

    fn(0);   // the caller participates; it is the only thread guaranteed free

    std::unique_lock<std::mutex> lk(impl_->m);
    impl_->cv_done.wait(lk, [&] { return impl_->waiting == 0; });
}

unsigned ThreadPool::default_threads() {
    const unsigned hw = std::thread::hardware_concurrency();
    // 7950X3D-style parts lose more to scheduling than they gain past the
    // physical core count, and this leaves headroom for the rest of the engine.
    return hw > 2 ? hw - 1 : 1;
}

} // namespace kata

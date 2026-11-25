#pragma once
#include <cstddef>
#include <functional>

namespace kata {

// Fixed-size worker pool for row-parallel kernels. GEMV at decode time is
// memory-bandwidth bound, so throughput comes from having many outstanding
// streams, not from wider vectors.
class ThreadPool {
public:
    explicit ThreadPool(unsigned n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool & operator=(const ThreadPool &) = delete;

    unsigned size() const { return n_; }

    // Runs fn(worker_index) on every worker, including the calling thread as
    // worker 0, and returns when all are done.
    void run(const std::function<void(unsigned)> & fn);

    static unsigned default_threads();

private:
    unsigned n_;
    struct Impl;
    Impl * impl_;
};

} // namespace kata

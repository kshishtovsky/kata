#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>

// Load guard for benchmarks. Sustained all-core memory saturation has hard-reset
// this machine repeatedly, so every harness must cap threads and idle a slice of
// each iteration. Env: KATA_MAX_THREADS, KATA_DUTY (0..1, default 0.90).
namespace kata::throttle {

inline unsigned max_threads() {
    static const unsigned v = [] {
        if (const char * e = std::getenv("KATA_MAX_THREADS")) {
            const int n = std::atoi(e);
            if (n > 0) return (unsigned)n;
        }
        // Half of hardware_concurrency is the physical core count on SMT parts.
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        return std::max(1u, hw / 2);
    }();
    return v;
}

inline double duty() {
    static const double v = [] {
        if (const char * e = std::getenv("KATA_DUTY")) {
            const double d = std::atof(e);
            if (d > 0.0 && d <= 1.0) return d;
        }
        return 0.90;
    }();
    return v;
}

// Sleeps so that the work just done occupies `duty` of the wall clock.
inline void pause_since(std::chrono::steady_clock::time_point t0) {
    const double d = duty();
    if (d >= 1.0) return;
    const auto worked = std::chrono::steady_clock::now() - t0;
    const auto idle = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        worked * ((1.0 - d) / d));
    if (idle > std::chrono::microseconds(50)) std::this_thread::sleep_for(idle);
}

} // namespace kata::throttle
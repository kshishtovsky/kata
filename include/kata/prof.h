#pragma once
#include <cstdio>
#include <cstddef>

namespace kata {

// Per-stage accounting check (EXP-0055).
//
// Four iterations in a row were lost to a denominator that did not measure what
// its name said: "78 GB/s" (one accumulator instead of a stream), "453 tok/s"
// (a different bench), "21.5% of peak" (a stage folded into another stage), and
// "scatter 365 ms" that was actually the down-projection GEMM. Each was found by
// hand, late. This closes the class: the sum of the stages a run reports must
// match the wall clock of that run, or the run says so.
//
// The rule is deliberately about *the reported split*, not about the total: an
// unaccounted stage shows up as sum < total, a double-counted one as sum > total.
struct StageCheck {
    double sum = 0;      // sum of the per-stage times
    double total = 0;    // wall clock the stages are supposed to fill
    double rel = 0;      // (sum - total) / total
    bool ok = false;
    const char * verdict = "";
};

// tol is a fraction (0.05 = 5%). A total of 0 means nothing ran; report ok so an
// empty chunk does not look like a broken counter.
inline StageCheck check_stages(double sum, double total, double tol = 0.05) {
    StageCheck c;
    c.sum = sum;
    c.total = total;
    if (!(total > 0.0)) {
        c.rel = 0.0;
        c.ok = true;
        c.verdict = "empty";
        return c;
    }
    c.rel = (sum - total) / total;
    const double a = c.rel < 0 ? -c.rel : c.rel;
    c.ok = a <= tol;
    c.verdict = c.ok ? "ok" : (c.rel < 0 ? "under" : "over");
    return c;
}

// One line, always the same shape, so a log grep finds every occurrence.
inline void print_stage_check(const char * what, const StageCheck & c) {
    std::fprintf(stderr, "[reconcile] %s sum %.1f vs total %.1f = %+.2f%% %s\n", what,
                 c.sum, c.total, 100.0 * c.rel, c.verdict);
}

} // namespace kata
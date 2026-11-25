#include "kata/gemv.h"
#include "kata/quant.h"
#include "kata/cpu.h"
#include "kata/pool.h"
#include "kata/throttle.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace kata;
using Clock = std::chrono::steady_clock;

static double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

static size_t read_peak_rss_kb() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) { sscanf(line + 6, "%zu", &kb); break; }
    }
    fclose(f);
    return kb;
}

static size_t l3_bytes() {
    FILE * f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
    if (!f) return 0;
    char buf[64] = {0};
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
    fclose(f);
    size_t v = strtoul(buf, nullptr, 10);
    if (strchr(buf, 'K')) v *= 1024;
    else if (strchr(buf, 'M')) v *= 1024 * 1024;
    return v;
}

int main(int argc, char ** argv) {
    // Default geometry is a real decode-shaped GEMV: 4096 rows by 4096 cols.
    // Weight buffer is 4096*4096/32*18 = 9.4 MB, so set --mb to exceed L3 when
    // measuring DRAM-bound behaviour.
    size_t rows = 4096, cols = 4096, mb = 0;
    int iters = 0, threads = 1, reps = 1;
    std::string kernel = "best";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : "0"; };
        if (a == "--rows") rows = strtoul(next(), nullptr, 10);
        else if (a == "--cols") cols = strtoul(next(), nullptr, 10);
        else if (a == "--mb") mb = strtoul(next(), nullptr, 10);
        else if (a == "--iters") iters = atoi(next());
        else if (a == "--threads") threads = atoi(next());
        else if (a == "--allow-full-load") {}   // documented no-op; cap is env-driven
        else if (a == "--reps") reps = std::max(1, atoi(next()));
        else if (a == "--kernel") kernel = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (cols % QK) { fprintf(stderr, "cols must be a multiple of %d\n", QK); return 2; }

    size_t nb = cols / QK;
    // --mb inflates the row count so the weight buffer outgrows L3.
    if (mb) {
        const size_t bytes_per_row = nb * sizeof(block_q4_0);
        rows = std::max<size_t>(mb * 1024 * 1024 / bytes_per_row, 1);
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> q4(0, 255);
    std::uniform_real_distribution<float> sc(0.001f, 0.02f);
    std::normal_distribution<float> act(0.0f, 1.0f);

    std::vector<block_q4_0> w(rows * nb);
    for (auto & b : w) {
        b.d = f32_to_f16(sc(rng));
        for (int j = 0; j < QK / 2; ++j) b.qs[j] = (uint8_t)q4(rng);
    }
    std::vector<float> xf(cols);
    for (auto & v : xf) v = act(rng);
    std::vector<block_q8_0> x(nb);
    quantize_row_q8_0(xf.data(), x.data(), cols);
    std::vector<float> out(rows);

    GemvFn fn = gemv_q4_0_q8_0_best();
    std::string kname = gemv_best_name();
    if (kernel == "ref") { fn = gemv_q4_0_q8_0_ref; kname = "q4_0_q8_0/ref"; }
    else if (kernel == "avx2") { fn = gemv_q4_0_q8_0_avx2; kname = "q4_0_q8_0/avx2"; }
    else if (kernel == "avx512") { fn = gemv_q4_0_q8_0_avx512; kname = "q4_0_q8_0/avx512-vnni"; }

    const double bytes = (double)w.size() * sizeof(block_q4_0) + (double)x.size() * sizeof(block_q8_0);
    const double flops = 2.0 * (double)rows * (double)cols;

    if (!iters) iters = (int)std::clamp(2.0e9 / std::max(bytes, 1.0), 5.0, 2000.0);

    // Threading multiplies effective bandwidth, so the iteration budget shrinks.
    // Hard cap: sustained all-core memory saturation has hard-reset this box.
    const unsigned tcap = throttle::max_threads();
    if (threads > (int)tcap) {
        printf("threads %d capped to %u by KATA_MAX_THREADS (env to override)\n", threads, tcap);
        threads = (int)tcap;
    }
    if (threads > 1) iters = std::max(3, iters * 2 / threads);

    ThreadPool pool((unsigned)threads);
    auto call = [&] {
        if (threads > 1) gemv_q4_0_q8_0_mt(rows, cols, out.data(), w.data(), x.data(), pool);
        else fn(rows, cols, out.data(), w.data(), x.data());
    };
    if (threads > 1) kname += "/mt" + std::to_string(threads);

    call();   // warm up page cache + code path

    // Whole-run repetitions matter: per-iteration medians hide the run-to-run
    // swings this chip shows at high thread counts.
    std::vector<double> secs, run_best;
    for (int r = 0; r < reps; ++r) {
        std::vector<double> cur;
        for (int i = 0; i < iters; ++i) {
            const auto w0 = std::chrono::steady_clock::now();
            const double t0 = now_s();
            call();
            cur.push_back(now_s() - t0);
            throttle::pause_since(w0);
        }
        std::sort(cur.begin(), cur.end());
        run_best.push_back(cur[cur.size() / 2]);
        secs.insert(secs.end(), cur.begin(), cur.end());
    }
    std::sort(run_best.begin(), run_best.end());
    const double rep_med = run_best[run_best.size() / 2];
    const double rep_best = run_best.front();
    std::sort(secs.begin(), secs.end());
    const double med = secs[secs.size() / 2];
    const double p10 = secs[secs.size() / 10];
    const double p90 = secs[secs.size() * 9 / 10];

    printf("kernel=%s isa=%s rows=%zu cols=%zu iters=%d reps=%d\n", kname.c_str(), isa_name(cpu_isa()), rows, cols, iters, reps);
    if (reps > 1) {
        printf("reps: median_GB_per_s=%.3f best_GB_per_s=%.3f  run_spread=%.1f%%\n",
               bytes / rep_med / 1e9, bytes / rep_best / 1e9, (reps > 1 ? (run_best.back() - rep_best) / rep_med * 100.0 : 0.0));
    }
    printf("weights_mb=%.2f l3_mb=%.2f workout_of_l3=%s\n",
           (double)w.size() * sizeof(block_q4_0) / 1048576.0, l3_bytes() / 1048576.0,
           (double)w.size() * sizeof(block_q4_0) > (double)l3_bytes() ? "yes" : "no");
    printf("median_ms=%.4f  p10_ms=%.4f  p90_ms=%.4f  spread=%.1f%%\n",
           med * 1e3, p10 * 1e3, p90 * 1e3, (p90 - p10) / med * 100.0);
    printf("GB_per_s=%.3f  GFLOP_per_s=%.2f  peak_rss_mb=%.1f\n",
           bytes / med / 1e9, flops / med / 1e9, read_peak_rss_kb() / 1024.0);

    // One-line machine-readable record for the benchmark history.
    printf("CSV,%s,%zu,%zu,%.4f,%.3f,%.2f\n", kname.c_str(), rows, cols, med * 1e3,
           bytes / med / 1e9, flops / med / 1e9);
    return 0;
}

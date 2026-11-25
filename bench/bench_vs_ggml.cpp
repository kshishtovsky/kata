// Kernel-vs-kernel comparison against the llama.cpp/ggml build on this box.
// ggml's GEMV is threaded and multi-row internally, so the fair kernel-level
// comparison is: one row of dot products per call, same data, same loop shape.
#include "kata/gemv.h"
#include "kata/quant.h"
#include "kata/cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <random>
#include <string>
#include <vector>

using namespace kata;
using Clock = std::chrono::steady_clock;
static double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

using VecDotFn = void (*)(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

int main(int argc, char ** argv) {
    size_t rows = 4096, cols = 4096, mb = 0;
    int iters = 0;
    std::string lib = "/home/medovarov/lcpp-base/build/bin/libggml-cpu.so";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : "0"; };
        if (a == "--rows") rows = strtoull(next(), nullptr, 10);
        else if (a == "--cols") cols = strtoull(next(), nullptr, 10);
        else if (a == "--mb") mb = strtoull(next(), nullptr, 10);
        else if (a == "--iters") iters = atoi(next());
        else if (a == "--lib") lib = next();
    }

    void * h = dlopen(lib.c_str(), RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
    auto ggml_dot = (VecDotFn)dlsym(h, "ggml_vec_dot_q4_0_q8_0");
    if (!ggml_dot) { fprintf(stderr, "symbol missing: %s\n", dlerror()); return 1; }
    // ggml keeps a lazily filled f16->f32 table; without this it is all zeros.
    auto ggml_init = (void (*)())dlsym(h, "ggml_cpu_init");
    if (!ggml_init) { fprintf(stderr, "ggml_cpu_init missing\n"); return 1; }
    ggml_init();

    const size_t nb = cols / QK;
    if (mb) {
        const size_t per_row = nb * sizeof(block_q4_0);
        rows = std::max<size_t>(mb * 1024 * 1024 / per_row, 1);
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

    std::vector<float> ref(rows), mine(rows), theirs(rows);
    GemvFn fn = gemv_q4_0_q8_0_best();
    fn(rows, cols, mine.data(), w.data(), x.data());
    for (size_t r = 0; r < rows; ++r) {
        ggml_dot((int)cols, &theirs[r], 0, w.data() + r * nb, 0, x.data(), 0, 1);
    }
    gemv_q4_0_q8_0_ref(rows, cols, ref.data(), w.data(), x.data());

    // Relative to the row magnitude, not the result: rows whose dot cancels to
    // near zero would otherwise show meaningless relative error.
    double max_my_err = 0, max_gg_err = 0;
    for (size_t r = 0; r < rows; ++r) {
        double mag = 0.0;
        for (size_t b = 0; b < nb; ++b) mag += (double)f16_to_f32(w[r * nb + b].d) * 127.0;
        mag = std::max(1e-9, mag);
        max_my_err = std::max(max_my_err, std::fabs(mine[r] - ref[r]) / mag);
        max_gg_err = std::max(max_gg_err, std::fabs(theirs[r] - ref[r]) / mag);
    }

    const double bytes = (double)w.size() * sizeof(block_q4_0) + (double)x.size() * sizeof(block_q8_0);
    if (!iters) iters = (int)std::clamp(2.0e9 / std::max(bytes, 1.0), 3.0, 2000.0);

    auto time_it = [&](auto && body) {
        body();
        std::vector<double> s;
        for (int i = 0; i < iters; ++i) { const double t = now_s(); body(); s.push_back(now_s() - t); }
        std::sort(s.begin(), s.end());
        return s[s.size() / 2];
    };

    const double t_mine = time_it([&] { fn(rows, cols, mine.data(), w.data(), x.data()); });
    const double t_gg = time_it([&] {
        for (size_t r = 0; r < rows; ++r) ggml_dot((int)cols, &theirs[r], 0, w.data() + r * nb, 0, x.data(), 0, 1);
    });

    printf("rows=%zu cols=%zu iters=%d weights_mb=%.1f\n", rows, cols, iters,
           (double)w.size() * sizeof(block_q4_0) / 1048576.0);
    printf("max_rel_err_vs_ref: kata=%.3e ggml=%.3e\n", max_my_err, max_gg_err);
    printf("kata  %-22s %8.4f ms  %7.3f GB/s\n", gemv_best_name(), t_mine * 1e3, bytes / t_mine / 1e9);
    printf("ggml  %-22s %8.4f ms  %7.3f GB/s\n", "q4_0_q8_0 (same loop)", t_gg * 1e3, bytes / t_gg / 1e9);
    printf("kata/ggml speedup = %.3fx\n", t_gg / t_mine);
    return 0;
}

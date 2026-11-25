// kata RMSNorm cost: scalar reference vs the vector path.
//
// There is no bare `ggml_rms_norm` entry point to race against — in ggml it is a
// graph node — so this reports absolute cost per call plus the implied bandwidth
// (x + w read, out written) against the machine's measured ~78 GB/s DRAM
// ceiling. That is enough to tell whether the kernel is the bottleneck.
#include "kata/norm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace kata;

int main(int argc, char ** argv) {
    // Defaults match LFM2.5-24B-A2B: hidden 2048, one token per call in decode.
    size_t rows = 1, cols = 2048;
    int iters = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc)       rows = std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--cols" && i + 1 < argc)  cols = std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }

    const float eps = 1e-6f;
    std::vector<float> x(rows * cols), w(cols), out(rows * cols);
    std::mt19937 g(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : x) v = nd(g);
    for (auto & v : w) v = 1.0f + 0.01f * nd(g);

    // The vector path must agree with the scalar definition before being timed.
    std::vector<float> ref(rows * cols);
    kata::rms_norm_ref(rows, cols, ref.data(), x.data(), w.data(), eps);
    kata::rms_norm(rows, cols, out.data(), x.data(), w.data(), eps);
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < rows * cols; ++i) {
        worst = std::max(worst, std::fabs((double)out[i] - (double)ref[i]));
        scale = std::max(scale, std::fabs((double)ref[i]));
    }
    std::printf("rows=%zu cols=%zu  vector vs scalar ref: max|diff|/max|val| = %.3e\n\n",
                rows, cols, worst / scale);

    const double bytes = (double)(rows * cols) * 4.0 * 3.0;  // read x, read w, write out
    auto timeit = [&](auto call, const char * name) {
        call();
        std::vector<double> v;
        v.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            call();
            v.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6);
        }
        std::sort(v.begin(), v.end());
        const double med = v[v.size() / 2];
        std::printf("  %-10s median %8.2f us   %6.2f GB/s\n", name, med, bytes / (med * 1e-6) / 1e9);
        return med;
    };

    const double t_ref = timeit([&] { kata::rms_norm_ref(rows, cols, out.data(), x.data(), w.data(), eps); }, "scalar");
    const double t_vec = timeit([&] { kata::rms_norm(rows, cols, out.data(), x.data(), w.data(), eps); }, "vector");
    std::printf("\nspeedup vector/scalar = %.2fx\n", t_ref / t_vec);
    return 0;
}
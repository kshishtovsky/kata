// RoPE is a per-token in-place transform: 2 passes over 2 * n_heads * head_dim
// floats. At decode (one token) that is a few microseconds at most, so the only
// question worth asking is whether the vector path is worth the branch.
#include "kata/rope.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
double bench(void (*fn)(size_t, size_t, float *, size_t, float, bool), size_t n_heads, size_t head_dim,
             size_t pos, float theta, bool interleaved, int iters) {
    std::mt19937 g(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(n_heads * head_dim);
    std::vector<double> t;
    for (int i = 0; i < iters; ++i) {
        for (auto & v : x) v = nd(g);
        const auto t0 = Clock::now();
        fn(n_heads, head_dim, x.data(), pos + i, theta, interleaved);
        const auto t1 = Clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    return median(t);
}
} // namespace

int main(int argc, char ** argv) {
    size_t n_heads = 32, head_dim = 64, pos = 1000;
    int iters = 2000;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--heads")) n_heads = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--head-dim")) head_dim = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--pos")) pos = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(argv[++i]);
    }

    std::printf("rope: %zu heads x %zu dim, pos %zu\n", n_heads, head_dim, pos);
    for (const bool interleaved : {false, true}) {
        const double t_ref = bench(kata::rope_ref, n_heads, head_dim, pos, 1e7f, interleaved, iters);
        const double t_vec = bench(kata::rope, n_heads, head_dim, pos, 1e7f, interleaved, iters);
        const size_t bytes = n_heads * head_dim * sizeof(float) * 2;
        std::printf("%-11s scalar %8.2f us (%6.1f GB/s)  vector %8.2f us (%6.1f GB/s)  %.2fx\n",
                    interleaved ? "interleaved" : "neox", t_ref, bytes / t_ref / 1e3,
                    t_vec, bytes / t_vec / 1e3, t_ref / t_vec);
    }
    return 0;
}
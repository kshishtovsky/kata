// Decode attention is the only operation whose cost grows with context, so the
// question is not "is it fast" but "at what depth does it stop being noise".
#include "kata/attn.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc, char ** argv) {
    size_t n_heads = 32, n_kv_heads = 8, head_dim = 64;
    int iters = 200;
    bool vector_only = false;
    bool phases = false;
    bool grouped = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--heads")) n_heads = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--kv-heads")) n_kv_heads = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--head-dim")) head_dim = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--vector-only")) vector_only = true;
        else if (!std::strcmp(argv[i], "--phases")) phases = true;
        else if (!std::strcmp(argv[i], "--grouped")) grouped = true;
    }

    std::mt19937 g(42);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    auto fill = [&](std::vector<float> & v) { for (auto & x : v) x = nd(g); };

    std::vector<float> q(n_heads * head_dim);
    fill(q);
    std::vector<float> out(n_heads * head_dim);

    std::printf("attention decode: %zu heads / %zu kv heads x %zu dim\n", n_heads, n_kv_heads, head_dim);
    std::printf("%8s %10s %10s %10s %10s\n", "n_kv", "scalar_us", "vector_us", "gain", "kv_MiB");

    for (size_t n_kv : {1, 128, 1024, 4096, 16384, 32768}) {
        std::vector<float> k(n_kv * n_kv_heads * head_dim), v(n_kv * n_kv_heads * head_dim);
        fill(k);
        fill(v);
        kata::AttnArgs a{};
        a.q = q.data(); a.k_cache = k.data(); a.v_cache = v.data();
        a.n_kv = n_kv; a.n_heads = n_heads; a.n_kv_heads = n_kv_heads; a.head_dim = head_dim;
        a.scale = 1.0f / std::sqrt((float)head_dim); a.out = out.data();
        std::vector<float> scratch(kata::attention_scratch_floats(a));

        auto time_it = [&](bool use_scalar) {
            std::vector<double> t;
            for (int i = 0; i < iters; ++i) {
                const auto t0 = Clock::now();
                if (use_scalar) kata::attention_decode_ref(a, scratch.data());
                else kata::attention_decode(a, scratch.data());
                const auto t1 = Clock::now();
                t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(t.begin(), t.end());
            return t[t.size() / 2];
        };
        const double ts = vector_only ? 0.0 : time_it(true);
        const double tv = time_it(false);
        const double kv_mib = 2.0 * n_kv * n_kv_heads * head_dim * sizeof(float) / (1024.0 * 1024.0);
        if (grouped) {
            std::vector<float> sg(kata::attention_scratch_floats_grouped(a));
            auto time_grouped = [&] {
                std::vector<double> t;
                for (int i = 0; i < iters; ++i) {
                    const auto t0 = Clock::now();
                    kata::attention_decode_grouped(a, sg.data());
                    const auto t1 = Clock::now();
                    t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }
                std::sort(t.begin(), t.end());
                return t[t.size() / 2];
            };
            const double tg = time_grouped();
            std::printf("%8zu %10.1f %10.1f %9.2fx %10.1f\n", n_kv, tv, tg, tv / tg, kv_mib);
            std::printf("CSV,attn_grouped,%zu,%.3f,%.3f\n", n_kv, tv, tg);
            continue;
        }
        if (phases) {
            // Same three passes, timed separately, so the remaining time can be
            // attributed before choosing what to optimise next.
            std::vector<float> sc(kata::attention_scratch_floats(a));
            double best[3] = {1e30, 1e30, 1e30};
            for (int r = 0; r < 5; ++r) {
                double t[3];
                kata::attention_decode_phases(a, sc.data(), t);
                for (int k = 0; k < 3; ++k) best[k] = std::min(best[k], t[k]);
            }
            std::printf("%8zu %10.1f %10.1f %10.1f %9.1f\n", n_kv, best[0], best[1], best[2], kv_mib);
        } else if (vector_only) std::printf("%8zu %10.2f %10.1f\n", n_kv, tv, kv_mib);
        else std::printf("%8zu %10.2f %10.2f %9.2fx %10.1f\n", n_kv, ts, tv, ts / tv, kv_mib);
        std::printf("CSV,attn,%zu,%.3f,%.3f\n", n_kv, tv, ts);
    }
    return 0;
}
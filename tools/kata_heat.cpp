// Expert routing heat: how skewed is the router really?
//
// Residency decisions (pin hot experts in L3, stream cold ones) only pay if the
// per-expert selection counts are far from uniform. This tool answers that with
// the engine's own counters instead of trusting another engine's telemetry.
// Usage: kata_heat <model.gguf> <tokens.bin> <n_tokens> [--gen N]
#include "kata/model.h"
#include "kata/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> read_tokens(const char * path, size_t limit) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<uint32_t> t;
    uint32_t v = 0;
    while (f.read(reinterpret_cast<char *>(&v), sizeof(v))) {
        t.push_back(v);
        if (limit && t.size() >= limit) break;
    }
    return t;
}

void report(const char * label, const std::vector<uint64_t> & heat, size_t n_layer, size_t n_expert) {
    // One row per MoE layer, but decided on the pooled total across layers:
    // residency is a per-expert decision over the whole model only when the
    // same expert is hot everywhere, which is worth checking separately.
    std::vector<uint64_t> pooled(n_expert, 0);
    uint64_t total = 0;
    for (size_t il = 0; il < n_layer; ++il)
        for (size_t e = 0; e < n_expert; ++e) {
            pooled[e] += heat[il * n_expert + e];
            total += heat[il * n_expert + e];
        }
    if (total == 0) { std::printf("%s: no routing observed\n", label); return; }

    std::vector<size_t> order(n_expert);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return pooled[a] > pooled[b]; });

    const double uniform = 1.0 / static_cast<double>(n_expert);
    const double expect = static_cast<double>(total) / static_cast<double>(n_expert);
    double share = 0.0, top_share = 0.0;
    for (size_t e = 0; e < n_expert; ++e) {
        const double d = static_cast<double>(pooled[e]) - expect;
        share += d * d / expect;
    }
    const size_t topn = std::max<size_t>(1, n_expert / 8);   // 8 hot experts of 64
    for (size_t i = 0; i < topn; ++i) top_share += static_cast<double>(pooled[order[i]]) / static_cast<double>(total);

    size_t live = 0;
    for (size_t il = 0; il < n_layer; ++il) {
        uint64_t row = 0;
        for (size_t e = 0; e < n_expert; ++e) row += heat[il * n_expert + e];
        if (row) ++live;
    }
    std::printf("%s: %llu selections over %zu MoE layers x %zu experts\n", label,
                static_cast<unsigned long long>(total), live, n_expert);
    std::printf("  hottest %zu experts hold %.1f%% of all selections (uniform would be %.1f%%)\n",
                topn, top_share * 100.0, 100.0 * topn / n_expert);
    // Echoing the degrees of freedom next to the statistic keeps a reader from
    // comparing it against the wrong table: df = 63 here, so the 99.9% point is
    // about 105.
    std::printf("  chi-square vs uniform: %.0f (df=%zu; 99.9%% point ~105, so >105 means skew)\n",
                share, n_expert - 1);
    std::printf("  top 8:");
    for (size_t i = 0; i < std::min<size_t>(8, n_expert); ++i)
        std::printf(" e%zu=%.2f%%", order[i],
                    100.0 * static_cast<double>(pooled[order[i]]) / static_cast<double>(total));
    std::printf("\n  coldest 4:");
    for (size_t i = n_expert; i-- > n_expert - std::min<size_t>(4, n_expert);)
        std::printf(" e%zu=%.3f%%", order[i],
                    100.0 * static_cast<double>(pooled[order[i]]) / static_cast<double>(total));
    std::printf("\n  a uniform router would give every expert %.2f%%\n", uniform * 100.0);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tokens.bin> <n_tokens> [--gen N]\n", argv[0]);
        return 1;
    }
    size_t n = std::strtoul(argv[3], nullptr, 10);
    size_t gen = 0;
    for (int i = 4; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--gen") == 0) gen = std::strtoul(argv[i + 1], nullptr, 10);

    try {
        kata::Model m;
        m.load(argv[1], std::max<size_t>(n + gen + 8, 64));
        const std::vector<uint32_t> toks = read_tokens(argv[2], n);
        if (toks.empty()) throw std::runtime_error("kata: empty token file");
        kata::ModelState st;

        // Prefill routing is the interesting half: prompt tokens are what an
        // agent session actually feeds in.
        m.heatmap_start();
        m.prefill(toks.data(), toks.size(), st);
        report("prefill", m.heatmap_take(), m.n_layer(), m.n_expert());

        if (gen > 0) {
            kata::SamplerConfig cfg;
            cfg.temp = 0.1f;
            cfg.top_k = 50;
            cfg.top_p = 0.9f;
            cfg.repeat_penalty = 1.05f;
            cfg.seed = 1;
            kata::Sampler smp(cfg);
            std::vector<uint32_t> hist(toks.begin(), toks.end());
            m.heatmap_start();
            const float * lg = nullptr;
            uint32_t next = toks.back();
            for (size_t i = 0; i < gen; ++i) {
                lg = m.forward(next, st);
                hist.push_back(next);
                next = smp.sample(lg, m.n_vocab(), hist);
            }
            report("decode", m.heatmap_take(), m.n_layer(), m.n_expert());
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "kata_heat: %s\n", e.what());
        return 1;
    }
    return 0;
}
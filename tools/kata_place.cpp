// EXP-0078: what a VRAM budget actually buys on a hybrid CPU/GPU decode.
//
// The engine cannot stream weights across PCIe per token: at ~16 GB/s a single
// 6.1 MB expert costs 0.4 ms and a token activates 152 of them. So the only
// viable policy is that an expert lives permanently on ONE device and the token
// is routed to it. This tool prices that policy from real routing data.
//
// The output is a curve: for every residency budget, how many of a token's
// expert bytes still have to cross the CPU's memory bus. Timing comes from the
// measured 73.5 GB/s CPU ceiling; the GPU rate is a parameter, because a
// consumer card is 280-400 GB/s and this machine's 764 GB/s is not
// representative. Reporting the CPU-only column is honest for any card.
//
// Usage: kata_place <model.gguf> <tokens.bin> <n_tokens> [--gen N] [--gpu-gbs X]
#include "kata/gguf.h"
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

struct Budget {
    double bytes;
    const char * label;
};

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <tokens.bin> <n_tokens> [--gen N] [--gpu-gbs X]\n",
                     argv[0]);
        return 1;
    }
    size_t n = std::strtoul(argv[3], nullptr, 10);
    size_t gen = 32;
    double gpu_gbs = 280.0;
    for (int i = 4; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--gen") == 0) gen = std::strtoul(argv[i + 1], nullptr, 10);
        if (std::strcmp(argv[i], "--gpu-gbs") == 0) gpu_gbs = std::strtod(argv[i + 1], nullptr);
    }

    try {
        kata::Model m;
        m.load(argv[1], std::max<size_t>(n + gen + 8, 64));
        const std::vector<uint32_t> toks = read_tokens(argv[2], n);
        if (toks.empty()) throw std::runtime_error("kata: empty token file");
        kata::ModelState st;

        // Bytes per (layer, expert) come from the model's own dimensions, not a
        // guess: gate + up + down are q4_K at 0.5625 B/value except down, which
        // is q6_K at 0.82 B/value. Charging q4_K to all three (the EXP-0071
        // mistake) understates an expert by 24%.
        const size_t n_expert = m.n_expert();
        size_t n_moe = 0;
        for (size_t il = 0; il < m.n_layer(); ++il) (void)il;
        m.heatmap_start();
        m.prefill(toks.data(), toks.size(), st);
        const std::vector<uint64_t> heat = m.heatmap_take();
        for (size_t il = 0; il < m.n_layer(); ++il) {
            uint64_t row = 0;
            for (size_t e = 0; e < n_expert; ++e) row += heat[il * n_expert + e];
            if (row) ++n_moe;
        }
        if (n_moe == 0) throw std::runtime_error("kata: model has no MoE layers");

        // Collect per-token routing for the decode steps.
        kata::SamplerConfig cfg;
        cfg.temp = 0.1f;
        cfg.top_k = 50;
        cfg.top_p = 0.9f;
        cfg.repeat_penalty = 1.05f;
        cfg.seed = 1;
        kata::Sampler smp(cfg);
        std::vector<uint32_t> hist(toks.begin(), toks.end());
        m.routing_record_start();
        uint32_t next = toks.back();
        for (size_t i = 0; i < gen; ++i) {
            const float * lg = m.forward(next, st);
            hist.push_back(next);
            next = smp.sample(lg, m.n_vocab(), hist);
        }
        const std::vector<kata::Model::TokenRouting> rec = m.routing_record_take();
        if (rec.empty()) throw std::runtime_error("kata: no routing recorded");
        const size_t nu = rec.front().n_expert_used;
        const size_t nmoe = rec.front().n_moe_layer;

        // Expert byte cost measured from the GGUF itself: the three expert
        // matrices of layer 0 divided by the expert count. Hard-coding
        // 0.5625/0.82 B-per-value would repeat the EXP-0071 error of charging
        // q4_K to ffn_down_exps, which is q6_K.
        double expert_bytes = 0;
        {
            kata::GgufFile g;
            g.open(argv[1]);
            const char * names[] = { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" };
            double sum = 0;
            int found = 0;
            for (const char * nm : names) {
                const std::string full = "blk.2." + std::string(nm);
                if (const kata::GgufTensor * tn = g.find(full)) { sum += (double)tn->n_bytes; ++found; }
            }
            if (found != 3) throw std::runtime_error("kata: cannot find the three expert matrices of blk.2");
            expert_bytes = sum / static_cast<double>(g.u32("lfm2moe.expert_count"));
        }

        // Rank experts pooled across layers -- the residency unit is one expert
        // of one layer, but pooling tells us whether heat is consistent.
        std::vector<uint64_t> pooled(n_expert, 0);
        for (size_t il = 0; il < m.n_layer(); ++il)
            for (size_t e = 0; e < n_expert; ++e) pooled[e] += heat[il * n_expert + e];
        std::vector<size_t> order(n_expert);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return pooled[a] > pooled[b]; });

        // Per-token byte cost for a residency set, using the REAL recorded
        // routing of each decode step rather than the pooled average.
        auto cpu_bytes = [&](const std::vector<char> & resident) {
            std::vector<double> per_token;
            per_token.reserve(rec.size());
            for (const auto & r : rec) {
                size_t on_cpu = 0;
                // resident[] is indexed [layer][expert]: expert 7 of layer 3
                // and expert 7 of layer 4 are different bytes, so indexing by
                // the bare expert id made every expert look resident and the
                // curve flat at the all-CPU row.
                for (size_t il = 0; il < nmoe; ++il)
                    for (size_t i = 0; i < nu; ++i)
                        if (!resident[il * n_expert + r.expert[il * nu + i]]) ++on_cpu;
                per_token.push_back(static_cast<double>(on_cpu) * expert_bytes);
            }
            return per_token;
        };

        std::printf("model: %zu layers, %zu MoE, %zu experts, %zu active, %.2f MB per expert\n",
                    m.n_layer(), nmoe, n_expert, nu, expert_bytes / 1e6);
        std::printf("decode steps captured: %zu, selections per token: %zu\n\n",
                    rec.size(), nmoe * nu);
        std::printf("%-14s %-10s %-9s %-9s %-9s %-9s\n",
                    "VRAM budget", "experts", "CPU MB", "CPU ms", "tok/s", "spread");

        const Budget budgets[] = {
            { 0, "0 (now)" }, { 1e9, "1 GB" }, { 2e9, "2 GB" }, { 4e9, "4 GB" },
            { 6e9, "6 GB" }, { 8e9, "8 GB" }, { 12e9, "12 GB" }, { 16e9, "16 GB" },
        };
        // Per-token non-expert traffic at 16k decode (EXP-0071, from the GGUF):
        // 591 MB fixed weights + 687 MB KV. Both scale with nothing but context
        // length and are the first thing a small card should take.
        const double fixed_mb = 591.0;
        for (const Budget & b : budgets) {
            // Fill the budget: fixed weights + KV first (they are read EVERY
            // token, so they are worth more than any expert), then experts by
            // heat.
            const size_t slots = static_cast<size_t>(m.n_layer() * n_expert);
            std::vector<char> resident(slots, 0);
            const double fixed_bytes = (fixed_mb + 687.0) * 1e6;
            double left = b.bytes - fixed_bytes;
            size_t nres = 0;
            if (left > 0) {
                // Placement is per (layer, expert): rank by that layer's own
                // counts, since expert 7 of layer 3 and of layer 4 are
                // unrelated bytes.
                std::vector<std::pair<uint64_t, size_t>> ranked;
                ranked.reserve(slots);
                for (size_t il = 0; il < m.n_layer(); ++il)
                    for (size_t e = 0; e < n_expert; ++e) {
                        const uint64_t c = heat[il * n_expert + e];
                        if (c) ranked.push_back({ c, il * n_expert + e });
                    }
                std::sort(ranked.begin(), ranked.end(),
                          [](const auto & x, const auto & y) { return x.first > y.first; });
                for (const auto & r : ranked) {
                    if (left < expert_bytes) break;
                    resident[r.second] = 1;
                    left -= expert_bytes;
                    ++nres;
                }
            }
            const std::vector<double> per = cpu_bytes(resident);
            double sum = 0, mx = 0;
            for (double v : per) { sum += v; mx = std::max(mx, v); }
            const double mean = sum / per.size();
            // CPU time for the remaining expert bytes; the fixed weights and KV
            // are on the GPU in every row with a budget > 0, and stay on the CPU
            // in the baseline row.
            const bool gpu_has_fixed = b.bytes > fixed_bytes;
            const double cpu_mb = mean / 1e6 + (gpu_has_fixed ? 0.0 : fixed_mb + 687.0);
            const double cpu_ms = cpu_mb / 73.5;
            const double gpu_ms = gpu_has_fixed
                ? (fixed_mb + 687.0) / gpu_gbs
                : 0.0;
            // Both devices run concurrently and a token waits for the slower
            // one. Expert work is split by placement, so CPU and GPU each pay
            // their own share; the token is max(), not sum().
            const double ms = std::max(cpu_ms, gpu_ms);
            std::printf("%-14s %-9zu %-9.0f %-9.2f %-9.1f %.0f%%\n",
                        b.label, nres, cpu_mb, cpu_ms, 1000.0 / std::max(ms, 1e-6),
                        100.0 * (mx - mean) / std::max(mean, 1e-9));
        }
        std::printf("\ngpu-gbs %.0f assumed; CPU ceiling 73.5 GB/s measured\n", gpu_gbs);
        std::printf("spread = max per-token CPU bytes vs mean, over the %zu captured steps\n",
                    rec.size());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "kata_place: %s\n", e.what());
        return 1;
    }
    return 0;
}
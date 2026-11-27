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

        // Fixed per-token bytes: all non-expert weights, plus one KV row per
        // attention layer per position. Both are computed here rather than
        // hard-coded at 16k, because the run's own prompt length is what the
        // KV term depends on and a 16k figure in a 1k-token run silently
        // overstates the buffer by 7x (the mistake EXP-0071 shipped).
        size_t n_attn_layer = 0;
        double fixed_weight_bytes = 0;
        double kv_bytes_per_pos = 0;
        {
            kata::GgufFile g;
            g.open(argv[1]);
            // Everything that is not an expert matrix is read once per token.
            for (const kata::GgufTensor & tn : g.tensors()) {
                if (tn.name.find("_exps.") != std::string::npos) continue;
                fixed_weight_bytes += (double)tn.n_bytes;
            }
            // The kv head count is not a metadata key in this architecture; it
            // only shows up as attn_k's output width (src/model.cpp does the
            // same). Guessing the key name is how this threw
            // "missing key lfm2moe.attention.head_count_kv" the first time.
            const double n_embd = (double)g.u32("lfm2moe.embedding_length");
            const double n_head = (double)g.u32("lfm2moe.attention.head_count");
            const double head_dim = n_embd / n_head;
            double head_kv = 0;
            for (const kata::GgufTensor & tn : g.tensors())
                if (tn.name.find("attn_k.weight") != std::string::npos) {
                    head_kv = (double)tn.dims[1] / head_dim;
                    ++n_attn_layer;
                }
            if (head_kv == 0) throw std::runtime_error("kata: no attn_k tensor found");
            // K and V, f32, one row per position per kv head, per attention layer.
            kv_bytes_per_pos = 2.0 * head_kv * head_dim * 4.0 * (double)n_attn_layer;
        }
        const size_t ctx_positions = st.pos;
        const double kv_bytes = kv_bytes_per_pos * (double)ctx_positions;

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
        //
        // `per_layer_cpu_bytes` is what the dependent chain needs: a token's
        // layer i cannot start its router until every expert of layer i-1 has
        // finished on BOTH devices, so the token costs sum(max(cpu_i, gpu_i))
        // over layers, not max(sum(cpu), sum(gpu)). The second form is what a
        // free-running traffic model computes, and it is an upper bound on
        // speed that no scheduler can reach.
        struct StepCost {
            double all_bytes = 0;
            std::vector<double> cpu_layer_bytes;   // per MoE layer, for this step
        };
        auto step_costs = [&](const std::vector<char> & resident) {
            std::vector<StepCost> out;
            out.reserve(rec.size());
            for (const auto & r : rec) {
                StepCost sc;
                sc.cpu_layer_bytes.assign(nmoe, 0.0);
                for (size_t il = 0; il < nmoe; ++il) {
                    size_t on_cpu = 0;
                    // resident[] is indexed [layer][expert]: expert 7 of layer 3
                    // and expert 7 of layer 4 are different bytes, so indexing by
                    // the bare expert id made every expert look resident and the
                    // curve flat at the all-CPU row.
                    for (size_t i = 0; i < nu; ++i)
                        if (!resident[il * n_expert + r.expert[il * nu + i]]) ++on_cpu;
                    sc.cpu_layer_bytes[il] = static_cast<double>(on_cpu) * expert_bytes;
                    sc.all_bytes += sc.cpu_layer_bytes[il];
                }
                out.push_back(std::move(sc));
            }
            return out;
        };

        std::printf("model: %zu layers, %zu MoE, %zu experts, %zu active, %.2f MB per expert\n",
                    m.n_layer(), nmoe, n_expert, nu, expert_bytes / 1e6);
        std::printf("decode steps captured: %zu, selections per token: %zu\n\n",
                    rec.size(), nmoe * nu);
        const Budget budgets[] = {
            { 0, "0 (now)" }, { 1e9, "1 GB" }, { 2e9, "2 GB" }, { 4e9, "4 GB" },
            { 6e9, "6 GB" }, { 8e9, "8 GB" }, { 12e9, "12 GB" }, { 16e9, "16 GB" },
        };
        // 68.3 GB/s is what a real decode achieves on this machine (EXP-0069),
        // not the 73.5 peak: using the peak makes every plan look better than
        // the machine can deliver.
        const double cpu_rate = 68.3e9;
        const double fixed_bytes = fixed_weight_bytes + kv_bytes;
        std::printf("fixed per token: %.0f MB weights + %.0f MB KV at %zu positions (%.2f MB/pos)\n\n",
                    fixed_weight_bytes / 1e6, kv_bytes / 1e6, ctx_positions, kv_bytes_per_pos / 1e6);
        std::printf("%-14s %-8s %-9s %-9s %-9s %-7s\n",
                    "VRAM budget", "experts", "today", "ideal", "chain", "loss");
        // Every layer's work depends on the previous layer on BOTH devices, so a
        // token's cost is the sum over layers of max(cpu_layer, gpu_layer), not
        // max(total cpu, total gpu). The gap between the two is the price of the
        // dependency and it is what decides whether a hybrid is worth building.
        for (const Budget & b : budgets) {
            const size_t slots = static_cast<size_t>(m.n_layer() * n_expert);
            std::vector<char> resident(slots, 0);
            double left = b.bytes - fixed_bytes;
            size_t nres = 0;
            if (left > 0) {
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
            const bool gpu_has_fixed = b.bytes > fixed_bytes;
            const std::vector<StepCost> steps = step_costs(resident);

            // Fixed bytes are split evenly per layer: an approximation, stated
            // as one. It matters only for the chain column, where a layer's
            // fixed share is the floor under its max().
            const double fixed_per_layer = fixed_bytes / (double)m.n_layer();
            double today_sum = 0, ideal_sum = 0, chain_sum = 0;
            for (const StepCost & sc : steps) {
                today_sum += (sc.all_bytes + fixed_bytes) / cpu_rate;
                // No dependency: the token waits for the slower device once.
                const double cpu_tot = (sc.all_bytes + (gpu_has_fixed ? 0.0 : fixed_bytes)) / cpu_rate;
                const double gpu_tot = gpu_has_fixed ? fixed_bytes / (gpu_gbs * 1e9) : 0.0;
                ideal_sum += std::max(cpu_tot, gpu_tot);
                double t = 0;
                size_t il_moe = 0;
                for (size_t il = 0; il < m.n_layer(); ++il) {
                    const double ex = (il_moe < nmoe) ? sc.cpu_layer_bytes[il_moe] : 0.0;
                    const bool is_moe = (il_moe < nmoe);
                    // A weight lives on exactly ONE device. Leaving the fixed
                    // share on the CPU while also charging it to the GPU made
                    // chain == today for every budget (the CPU term was always
                    // the larger, so the GPU never won a layer).
                    const double cpu_b = (gpu_has_fixed ? 0.0 : fixed_per_layer) + ex;
                    const double gpu_b = (gpu_has_fixed ? fixed_per_layer : 0.0) +
                                         ((is_moe ? 1.0 : 0.0) * expert_bytes * nu - ex);
                    t += std::max(cpu_b / cpu_rate, gpu_b / (gpu_gbs * 1e9));
                    if (is_moe) ++il_moe;
                }
                chain_sum += t;
            }
            const double today_ms = 1000.0 * today_sum / (double)steps.size();
            const double ideal_ms = 1000.0 * ideal_sum / (double)steps.size();
            const double chain_ms = 1000.0 * chain_sum / (double)steps.size();
            std::printf("%-14s %-8zu %-9.1f %-9.1f %-9.1f %.2fx\n",
                        b.label, nres, today_ms, ideal_ms, chain_ms,
                        chain_ms / std::max(ideal_ms, 1e-9));
        }
        std::printf("\ngpu-gbs %.0f assumed; CPU %.1f GB/s (achieved in a real decode)\n", gpu_gbs, cpu_rate / 1e9);
        std::printf("spread = max per-token CPU bytes vs mean, over the %zu captured steps\n",
                    rec.size());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "kata_place: %s\n", e.what());
        return 1;
    }
    return 0;
}
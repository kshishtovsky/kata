// EXP-0071: the continuous-batching curve, plus the ceiling that decides whether
// it is worth having a scheduler at all.
//
// The thesis is that one forward pass over B rows reads every weight once, so B
// rows must cost far less than B single-row passes. That is true and it is not
// the binding constraint: MoE weight traffic grows with the UNION of selected
// experts, and 64 experts x 4 picks fill that union within 32 rows.
//
// KNOWN LIMIT, and it must be read before trusting the speedup column: the
// batched arm is a B-token causal chunk in ONE shared KV cache, because the
// cache lives in Model::Impl and ModelState holds only `pos`. Row 0 is a true
// decode step; row i also attends to rows 0..i-1, so its hidden state is mixed
// with its neighbours' and its expert choices are more correlated than
// independent sequences would be -- which flatters the union column. The work
// per row (bytes read, FMAs) still matches a real batch within B/prefix, so the
// curve bounds what a scheduler can deliver rather than predicting it. Verified
// in /tmp/bprobe: row 0 changes by 0.0 when row 1's token changes, row 1 changes
// by 53.7% of max logit when row 0's token changes. Making it a real batch needs
// the KV cache owned per sequence first (671 MB per sequence at 16k).
// shortcut: B rows share one cache, revisit when per-sequence KV lands
//
// Build: cmake --build build --target bench_batch
// Run:   KATA_MAX_THREADS=16 ./build/bench_batch <model.gguf> <tokens.bin>
//
// Self-check: B=1 batched must not be more than 1.6x slower than one decode
// pass (it goes through the chunked GEMM path, which is not what decode uses),
// and B=64 batched must be faster per row than B=1. Both are structural
// assertions on the shape of the curve, not on absolute speed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "kata/model.h"

namespace {

std::vector<uint32_t> read_tokens(const char * path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const auto nb = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> v(nb / 4);
    f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(nb));
    return v;
}

// Shape constants of LFM2.5-24B-A2B, read from the model in a real deployment.
// Kept explicit here so the traffic model can be checked against the profile by
// hand; a wrong constant shows up as a GB/s figure no kernel could reach.
struct Shape {
    double bytes_per_param = 0.5625;   // q4_K
    size_t n_moe = 30, n_attn = 10, n_conv = 30, n_dense = 2;
    size_t n_embd = 2048, n_ff_exp = 1536, n_expert = 64, n_used = 4;
    size_t n_vocab = 128000, n_ff_dense = 11776;
    size_t n_head_kv = 8, head_dim = 64;
};

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tokens.bin>\n", argv[0]);
        return 2;
    }
    const std::vector<uint32_t> prompt = read_tokens(argv[2]);
    const size_t pos = prompt.size();
    const Shape S;

    const double exp_layer_mb =
        3.0 * S.n_ff_exp * S.n_embd * S.n_expert * S.bytes_per_param / 1e6;
    const double fixed_mb =
        (S.n_vocab * S.n_embd * S.bytes_per_param                                       // lm_head
         + S.n_conv * 4.0 * S.n_embd * S.n_embd * S.bytes_per_param                     // shortconv
         + S.n_attn * (3072.0 * S.n_embd + S.n_embd * S.n_embd) * S.bytes_per_param     // qkv + wo
         + S.n_dense * 3.0 * S.n_ff_dense * S.n_embd * S.bytes_per_param                // dense ffn
         + S.n_moe * S.n_expert * S.n_embd * 4.0) / 1e6;                                // routers
    const double kv_mb =
        S.n_attn * S.n_head_kv * S.head_dim * pos * 4 * 2 / 1e6;

    std::printf("traffic model: fixed %.0f MB + KV %.1f MB + %.0f MB per expert layer (all 64)\n",
                fixed_mb, kv_mb, exp_layer_mb);
    std::printf("%4s %11s %11s %9s %8s %9s %8s\n",
                "B", "Bx decode", "batched", "speedup", "union", "MB/pass", "GB/s");

    double per_row_b1 = 0, per_row_b64 = 0;
    double b1_batched = 0, decode1 = 0;

    for (size_t B : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
        double dec = 0;
        {
            kata::Model a;
            a.load(argv[1], 4096);
            a.reset();
            kata::ModelState st;
            a.prefill(prompt.data(), prompt.size(), st);
            std::vector<kata::ModelState> sts(B, st);
            const uint32_t tok = prompt[0];
            for (auto & s : sts) a.forward(tok, s);            // warm
            const auto t0 = std::chrono::steady_clock::now();
            for (auto & s : sts) a.forward(tok, s);
            dec = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();
        }
        double bat = 1e18;
        for (int rep = 0; rep < 3; ++rep) {
            kata::Model a;
            a.load(argv[1], 4096);
            a.reset();
            kata::ModelState st;
            a.prefill(prompt.data(), prompt.size(), st);
            std::vector<float> out(B * a.n_vocab());
            const auto t0 = std::chrono::steady_clock::now();
            a.prefill_logits(prompt.data(), B, st, out.data());
            bat = std::min(bat, std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0).count());
        }
        // Expected distinct experts among B draws of top-4-of-64.
        const double union_e =
            S.n_expert * (1.0 - std::pow(1.0 - static_cast<double>(S.n_used) / S.n_expert,
                                         static_cast<double>(B)));
        const double mb = fixed_mb + kv_mb + union_e / S.n_expert * exp_layer_mb * S.n_moe;
        std::printf("%4zu %11.1f %11.1f %9.2fx %8.1f %9.0f %8.1f\n",
                    B, dec, bat, dec / bat, union_e, mb, mb / bat);
        if (B == 1) { b1_batched = bat; decode1 = dec; per_row_b1 = bat; }
        if (B == 64) per_row_b64 = bat / 64.0;
    }

    // Structural self-checks, as promised in the header.
    const double ratio_m1 = b1_batched / decode1;
    std::printf("\nself-check: batched(B=1) / decode = %.2fx (must be < 1.6x)\n", ratio_m1);
    std::printf("self-check: per-row B=64 / per-row B=1 = %.2fx (must be < 1.0x)\n",
                per_row_b64 / per_row_b1);
    if (ratio_m1 >= 1.6) { std::printf("FAIL: batched path is not decode-comparable\n"); return 1; }
    if (per_row_b64 >= per_row_b1) { std::printf("FAIL: batching did not amortise\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
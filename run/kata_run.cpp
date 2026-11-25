// Minimal driver: tokenise nothing yet (the CLI takes token ids), run N tokens
// through the model and dump logits so the forward pass can be diffed against a
// reference implementation.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "kata/gguf.h"
#include "kata/model.h"
#include "kata/sampler.h"

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: kata_run <model.gguf> <tok,tok,...> [--n-ctx N] [--dump F]\n");
        return 2;
    }
    std::string model = argv[1];
    std::vector<uint32_t> toks;
    for (const char * p = argv[2]; *p;) {
        char * end = nullptr;
        const unsigned long v = std::strtoul(p, &end, 10);
        if (end == p) break;
        toks.push_back(static_cast<uint32_t>(v));
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    size_t n_ctx = 64;
    size_t greedy = 0;   // argmax decode in the old sense: is set by --greedy N
    size_t gen = 0;
    const char * dump = nullptr;

    // The GGUF itself recommends sampling parameters; use them unless overridden.
    kata::SamplerConfig cfg;
    uint32_t eos = UINT32_MAX;
    bool have_tk = false, have_tp = false, have_pp = false;
    {
        kata::GgufFile f;
        f.open(model);
        if (f.has_key("general.sampling.top_k")) { cfg.top_k = static_cast<int>(f.u32("general.sampling.top_k")); have_tk = true; }
        if (f.has_key("general.sampling.temp")) { cfg.temp = f.f32("general.sampling.temp"); }
        if (f.has_key("general.sampling.penalty_repeat")) { cfg.repeat_penalty = f.f32("general.sampling.penalty_repeat"); have_pp = true; }
        // No standard key for top_p; leave the default.
        if (f.has_key("general.sampling.top_p")) { cfg.top_p = f.f32("general.sampling.top_p"); have_tp = true; }
        // End of generation: the GGUF names one eos; the model also has a second
        // EOG id (124895) that the vocab flags as end-of-generation.
        if (f.has_key("tokenizer.ggml.eos_token_id")) eos = f.u32("tokenizer.ggml.eos_token_id");
    }
    for (int i = 3; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--n-ctx")) n_ctx = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--greedy")) greedy = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--gen")) gen = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--temp")) cfg.temp = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--top-k")) { cfg.top_k = std::atoi(argv[++i]); have_tk = true; }
        else if (!std::strcmp(argv[i], "--top-p")) { cfg.top_p = std::strtof(argv[++i], nullptr); have_tp = true; }
        else if (!std::strcmp(argv[i], "--repeat-penalty")) { cfg.repeat_penalty = std::strtof(argv[++i], nullptr); have_pp = true; }
        else if (!std::strcmp(argv[i], "--seed")) cfg.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--repeat-last-n")) cfg.repeat_last_n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--dump")) dump = argv[++i];
    }
    (void)have_tp; (void)have_pp; (void)have_tk;
    std::fprintf(stderr, "sampling: temp=%.3f top_k=%d top_p=%.3f repeat=%.3f\n",
                 cfg.temp, cfg.top_k, cfg.top_p, cfg.repeat_penalty);

    kata::Model m;
    m.load(model, n_ctx);
    std::fprintf(stderr, "loaded: n_vocab=%zu n_embd=%zu n_layer=%zu\n", m.n_vocab(), m.n_embd(), m.n_layer());

    kata::ModelState st;
    kata::Sampler smp(cfg);
    std::vector<uint32_t> history(toks);
    const float * logits = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t t : toks) logits = m.forward(t, st);
    double gen_ms = 0.0;
    if (gen) {
        const auto g0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < gen; ++i) {
            const uint32_t t = smp.sample(logits, m.n_vocab(), history);
            if (t == eos || t == 124895) { std::printf("<eog:%u> ", t); std::fflush(stdout); break; }
            std::printf("%u ", t);
            std::fflush(stdout);
            history.push_back(t);
            logits = m.forward(t, st);
        }
        gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
        std::printf("\n");
    }
    if (greedy) {
        const auto g0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < greedy; ++i) {
            size_t best = 0;
            for (size_t v = 1; v < m.n_vocab(); ++v) if (logits[v] > logits[best]) best = v;
            std::printf("%zu ", best);
            std::fflush(stdout);
            logits = m.forward(static_cast<uint32_t>(best), st);
        }
        gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
        std::printf("\n");
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (dump) {
        FILE * f = std::fopen(dump, "wb");
        if (!f) { std::perror("dump"); return 1; }
        std::fwrite(logits, sizeof(float), m.n_vocab(), f);
        std::fclose(f);
    }
    size_t best = 0;
    for (size_t i = 1; i < m.n_vocab(); ++i) if (logits[i] > logits[best]) best = i;
    std::printf("argmax=%zu logit=%.5f  prefilled %zu tokens\n", best, logits[best], toks.size());
    if (gen || greedy) std::printf("generated %zu tokens in %.2f ms = %.2f ms/token, %.2f tok/s\n",
                         gen ? gen : greedy, gen_ms, gen_ms / (gen ? gen : greedy), (gen ? gen : greedy) * 1000.0 / gen_ms);
    return 0;
}

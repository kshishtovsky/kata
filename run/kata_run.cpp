// Minimal driver: tokenise nothing yet (the CLI takes token ids), run N tokens
// through the model and dump logits so the forward pass can be diffed against a
// reference implementation.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kata/model.h"

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
    size_t greedy = 0;
    const char * dump = nullptr;
    for (int i = 3; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--n-ctx")) n_ctx = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--greedy")) greedy = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--dump")) dump = argv[++i];
    }

    kata::Model m;
    m.load(model, n_ctx);
    std::fprintf(stderr, "loaded: n_vocab=%zu n_embd=%zu n_layer=%zu\n", m.n_vocab(), m.n_embd(), m.n_layer());

    kata::ModelState st;
    const float * logits = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t t : toks) logits = m.forward(t, st);
    if (greedy) {
        for (size_t i = 0; i < greedy; ++i) {
            size_t best = 0;
            for (size_t v = 1; v < m.n_vocab(); ++v) if (logits[v] > logits[best]) best = v;
            std::printf("%zu ", best);
            std::fflush(stdout);
            logits = m.forward(static_cast<uint32_t>(best), st);
        }
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
    std::printf("argmax=%zu logit=%.5f  %.2f ms/token (%zu tokens)\n", best, logits[best], ms / toks.size(), toks.size());
    return 0;
}

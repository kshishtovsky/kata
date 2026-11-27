// Greedy speculative decoding: a small draft model proposes K tokens, the target
// verifies them in one batched pass. Acceptance is the only number that decides
// whether the mechanism is worth product code, so this measures it directly on
// real generation instead of on synthetic text.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "kata/model.h"

namespace {
std::vector<uint32_t> read_tokens(const char * path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const auto n = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> t(n / sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(t.data()), (std::streamsize)(t.size() * sizeof(uint32_t)));
    return t;
}
uint32_t argmax(const float * v, size_t n) {
    uint32_t best = 0;
    for (size_t i = 1; i < n; ++i) if (v[i] > v[best]) best = (uint32_t)i;
    return best;
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <target.gguf> <draft.gguf> <tokens.bin> <K> [n_gen]\n", argv[0]);
        return 2;
    }
    const int K = std::atoi(argv[4]);
    const size_t n_gen = argc > 5 ? (size_t)std::atoi(argv[5]) : 64;
    const size_t n_ctx = 2048;
    std::vector<uint32_t> prompt = read_tokens(argv[3]);
    if (prompt.empty()) { std::fprintf(stderr, "empty prompt\n"); return 2; }

    kata::Model target, draft;
    target.load(argv[1], n_ctx);
    draft.load(argv[2], n_ctx);
    if (target.n_vocab() != draft.n_vocab()) {
        std::fprintf(stderr, "vocab mismatch %zu vs %zu\n", target.n_vocab(), draft.n_vocab());
        return 2;
    }
    target.reset();
    draft.reset();
    kata::ModelState st, sd;

    // Warm both models so page faults do not land inside the measured loop.
    const float * tlog = target.prefill(prompt.data(), prompt.size(), st);
    (void)draft.prefill(prompt.data(), prompt.size(), sd);
    uint32_t t0 = argmax(tlog, target.n_vocab());

    // Reference: plain greedy decode with the target alone.
    std::vector<uint32_t> ref{t0};
    {
        kata::ModelState s; kata::Model m;
        m.load(argv[1], n_ctx); m.reset();
        m.prefill(prompt.data(), prompt.size(), s);
        uint32_t cur = t0;
        for (size_t i = 1; i < n_gen; ++i) {
            cur = argmax(m.forward(cur, s), m.n_vocab());
            ref.push_back(cur);
        }
    }

    std::vector<uint32_t> out{t0};
    std::vector<uint32_t> draft_tok(K);
    std::vector<uint32_t> accept_hist(K + 1, 0);
    double t_draft = 0, t_verify = 0;
    size_t cycles = 0, accepted_total = 0;
    while (out.size() < n_gen) {
        // --- draft: K greedy tokens from the draft model
        const double a = now_ms();
        uint32_t dcur = out.back();
        for (int i = 0; i < K; ++i) {
            dcur = argmax(draft.forward(dcur, sd), draft.n_vocab());
            draft_tok[i] = dcur;
        }
        t_draft += now_ms() - a;
        // --- verify: one batched pass of the target over the K candidates
        const double b = now_ms();
        const float * vlog = target.prefill(draft_tok.data(), (size_t)K, st);
        // logits of candidate i predict the token AFTER draft_tok[i]; candidate i
        // itself is checked against what the target would have produced at that
        // position, which the pass returns only for the last row, so we verify by
        // regenerating: the target's argmax of row i must equal draft_tok[i].
        // prefill returns the last row only, so we run it row by row for exactness.
        (void)vlog;
        t_verify += now_ms() - b;
        ++cycles;
        break;   // K-token verification needs per-row logits; handled below
    }
    (void)t_draft; (void)t_verify; (void)accepted_total; (void)accept_hist; (void)ref;
    std::printf("harness needs per-row logits from prefill; see next revision\n");
    return 0;
}

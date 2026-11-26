// EXP-0046: does a prompt-lookup draft accept enough to be worth building?
//
// Two numbers decide it, and neither needs a speculative decoder to exist yet:
//   1. how many tokens a lookup draft (2..4-token suffix match) predicts
//      correctly in the greedy continuation of a real prompt, per block size;
//   2. what one verification pass of n candidates actually costs, measured as a
//      real prefill call at a real context depth.
// The reported figure is accepted tokens per cycle divided by the cycle's
// measured cost -- throughput, not acceptance rate.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kata/model.h"

namespace {

constexpr size_t kMaxBlock = 8;

uint32_t argmax(const float * v, size_t n) {
    uint32_t best = 0;
    for (size_t i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

// Greedy continuation: the ground truth a speculative decoder has to reproduce.
std::vector<uint32_t> greedy(kata::Model & m, const std::vector<uint32_t> & prompt, size_t n) {
    kata::ModelState st;
    const float * logits = m.prefill(prompt.data(), prompt.size(), st);
    std::vector<uint32_t> out(prompt);
    out.reserve(prompt.size() + n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t t = argmax(logits, m.n_vocab());
        out.push_back(t);
        logits = m.forward(t, st);
    }
    return out;
}

// Prompt lookup: find the most recent earlier occurrence of the last `order`
// tokens of the prefix and copy what followed it. Longer order first, because a
// 4-token match predicts better than a 2-token one when it exists.
size_t draft_lookup(const std::vector<uint32_t> & seq, size_t pos, size_t n, size_t min_order, uint32_t * out) {
    for (size_t order = 4; order + 1 > min_order; --order) {
        if (pos < order) break;
        const size_t start = pos - order;
        for (size_t j = start; j-- > 0;) {
            if (std::memcmp(seq.data() + j, seq.data() + start, order * sizeof(uint32_t)) != 0) continue;
            const size_t from = j + order;
            const size_t avail = std::min(n, pos - from);
            if (avail == 0) continue;
            std::memcpy(out, seq.data() + from, avail * sizeof(uint32_t));
            return avail;
        }
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: bench_spec <model.gguf> <tok.bin> [--gen N] [--depth N] [--min-order N]\n");
        return 2;
    }
    const std::string model = argv[1];
    size_t gen = 256, depth = 0, min_order = 2;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--gen")) gen = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--depth")) depth = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--min-order")) min_order = std::strtoul(argv[++i], nullptr, 10);
    }
    std::vector<uint32_t> prompt;
    {
        FILE * f = std::fopen(argv[2], "rb");
        if (!f) { std::perror("tok.bin"); return 2; }
        uint32_t v;
        while (std::fread(&v, sizeof v, 1, f) == 1) prompt.push_back(v);
        std::fclose(f);
    }
    if (prompt.empty()) { std::fprintf(stderr, "empty prompt\n"); return 2; }
    const size_t n_ctx = prompt.size() + gen + 64;

    kata::Model m;
    m.load(model, n_ctx);
    std::fprintf(stderr, "prompt %zu tok, generating %zu greedy tokens (ctx %zu)\n", prompt.size(), gen, n_ctx);

    const auto g0 = std::chrono::steady_clock::now();
    const std::vector<uint32_t> seq = greedy(m, prompt, gen);
    const double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
    std::fprintf(stderr, "greedy %zu tok in %.1f ms = %.2f ms/token\n", gen, gen_ms, gen_ms / (double)gen);

    // Verification cost first: a speculative cycle is one prefill of `block`
    // tokens at the context depth the cycle runs at.
    const size_t at = depth ? std::min(depth, n_ctx - kMaxBlock - 2)
                            : std::min(seq.size() / 2, n_ctx - kMaxBlock - 2);
    double verify_ms[kMaxBlock + 1] = {};
    std::printf("\nverify cost at context depth %zu (prefill of n tokens, 5 reps, median)\n", at);
    std::printf("block  ms/pass  ms/token\n");
    for (size_t block = 1; block <= kMaxBlock; ++block) {
        std::vector<double> samples;
        for (int rep = 0; rep < 5; ++rep) {
            kata::ModelState st;
            m.prefill(seq.data(), at, st);
            const auto t0 = std::chrono::steady_clock::now();
            m.prefill(seq.data() + at, block, st);
            samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(samples.begin(), samples.end());
        verify_ms[block] = samples[samples.size() / 2];
        std::printf("%5zu  %7.2f  %8.2f\n", block, verify_ms[block], verify_ms[block] / (double)block);
    }
    std::printf("baseline (no speculation): %.2f ms/token\n", verify_ms[1]);

    // Walk the generated tail exactly as the speculative loop would: draft a
    // block, keep the matching prefix, take one token the target itself gives.
    std::printf("\nblock  cycles  acc/cycle  hit%%  tok/pass  ms/pass  tok/s  vs baseline\n");
    for (size_t block = 1; block <= kMaxBlock; ++block) {
        size_t pos = prompt.size(), cycles = 0, accepted = 0, hits = 0;
        std::vector<uint32_t> buf(kMaxBlock);
        while (pos < seq.size()) {
            const size_t want = std::min(block, seq.size() - pos);
            const size_t got = draft_lookup(seq, pos, want, min_order, buf.data());
            if (got) ++hits;
            size_t k = 0;
            while (k < got && seq[pos + k] == buf[k]) ++k;
            accepted += k;
            ++cycles;
            pos += k + 1; // the token the target itself produces after the prefix
        }
        const size_t produced = cycles + accepted;
        const double per_cycle = (double)produced / (double)cycles;
        const double tok_s = 1000.0 * per_cycle / verify_ms[block];
        const double base_s = 1000.0 / verify_ms[1];
        std::printf("%5zu  %6zu  %9.2f  %4.0f  %8.2f  %7.2f  %5.1f  %11.2fx\n", block, cycles,
                    (double)accepted / (double)cycles, 100.0 * (double)hits / (double)cycles,
                    per_cycle, verify_ms[block], tok_s, tok_s / base_s);
    }
    return 0;
}
#include "kata/sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kata {

Sampler::Sampler(SamplerConfig cfg) : cfg_(cfg), rng_(cfg.seed) {}

template <class Set>
void Sampler::penalize(Set set, size_t n_vocab, const std::vector<uint32_t> & history) const {
    if (cfg_.repeat_penalty == 1.0f || history.empty()) return;
    const size_t n = std::min(history.size(), static_cast<size_t>(std::max(0, cfg_.repeat_last_n)));
    for (size_t i = history.size() - n; i < history.size(); ++i) {
        const uint32_t t = history[i];
        if (t >= n_vocab) continue;
        // Same asymmetry as llama.cpp: a positive logit is pushed down, a
        // negative one up, so the ordering around zero stays sane.
        const float x = set.get(t);
        set.set(t, x > 0.0f ? x / cfg_.repeat_penalty : x * cfg_.repeat_penalty);
    }
}

void Sampler::apply_penalty(float * logits, size_t n_vocab, const std::vector<uint32_t> & history) const {
    penalize(PtrSet{ logits }, n_vocab, history);
}

uint32_t Sampler::sample(const float * logits, size_t n_vocab, const std::vector<uint32_t> & history) {
    if (n_vocab == 0) throw std::runtime_error("kata: sampler needs a vocabulary");
    if (cfg_.temp <= 0.0f) {
        uint32_t best = 0;
        for (size_t v = 1; v < n_vocab; ++v) if (logits[v] > logits[best]) best = static_cast<uint32_t>(v);
        return best;
    }

    // Penalty first, on a copy indexed by vocab id, then rank.
    cand_.resize(n_vocab);
    for (size_t v = 0; v < n_vocab; ++v) cand_[v] = { logits[v], static_cast<uint32_t>(v) };
    struct CandSet {
        std::vector<std::pair<float, uint32_t>> * c;
        float get(size_t i) const { return (*c)[i].first; }
        void set(size_t i, float x) const { (*c)[i].first = x; }
    };
    penalize(CandSet{ &cand_ }, n_vocab, history);

    size_t keep = n_vocab;
    if (cfg_.top_k > 0 && static_cast<size_t>(cfg_.top_k) < keep) keep = static_cast<size_t>(cfg_.top_k);
    // Tie-break on id so equal logits sample the same nucleus every run.
    std::partial_sort(cand_.begin(), cand_.begin() + keep, cand_.end(), [](const auto & a, const auto & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });

    const float maxv = cand_[0].first / cfg_.temp;
    double sum = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        const float p = std::exp(static_cast<float>(cand_[i].first / cfg_.temp - maxv));
        cand_[i].first = p;
        sum += p;
    }
    if (sum <= 0.0) return cand_[0].second;

    // top_p: keep the smallest prefix whose cumulative mass reaches the threshold.
    if (cfg_.top_p < 1.0f) {
        double acc = 0.0;
        size_t cut = keep;
        for (size_t i = 0; i < keep; ++i) {
            acc += cand_[i].first;
            if (acc / sum >= cfg_.top_p) { cut = i + 1; break; }
        }
        keep = cut;
        sum = acc;
    }

    std::uniform_real_distribution<double> u(0.0, sum);
    const double r = u(rng_);
    double acc = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        acc += cand_[i].first;
        if (r <= acc) return cand_[i].second;
    }
    return cand_[keep - 1].second;
}

}  // namespace kata

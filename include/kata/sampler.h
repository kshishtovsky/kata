// Token sampling. Without this the engine is pure argmax, which self-reinforces
// through the KV cache and degenerates into repeats.
#pragma once
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace kata {

struct SamplerConfig {
    float temp = 0.8f;            // <= 0 means greedy
    int top_k = 40;               // <= 0 disables
    float top_p = 0.9f;           // >= 1 disables
    float repeat_penalty = 1.0f;  // 1.0 disables
    int repeat_last_n = 64;       // history window considered by the penalty
    uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig cfg = {});
    // `history` is the full token sequence so far; the last `repeat_last_n` are used.
    uint32_t sample(const float * logits, size_t n_vocab, const std::vector<uint32_t> & history);
    // Exposed for tests: applies the repeat penalty in place.
    void apply_penalty(float * logits, size_t n_vocab, const std::vector<uint32_t> & history) const;
    const SamplerConfig & config() const { return cfg_; }

private:
    SamplerConfig cfg_;
    std::mt19937_64 rng_;
    // A logit array addressed by vocab id, or the same thing inside cand_.
    struct PtrSet {
        float * v;
        float get(size_t i) const { return v[i]; }
        void set(size_t i, float x) const { v[i] = x; }
    };
    template <class Set> void penalize(Set set, size_t n_vocab, const std::vector<uint32_t> & history) const;
    std::vector<std::pair<float, uint32_t>> cand_;
};

}  // namespace kata

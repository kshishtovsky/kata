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
    struct Cand {
        float logit;
        uint32_t id;
    };

    SamplerConfig cfg_;
    std::mt19937_64 rng_;
    mutable std::uniform_real_distribution<double> dist_01_{0.0, 1.0};

    std::vector<Cand> heap_;
    std::vector<std::pair<float, uint32_t>> cand_;
    std::vector<uint8_t> pen_mask_;
    std::vector<float> pen_val_;

    void validate_config() const;
    void ensure_capacity(size_t n_vocab);
    std::vector<uint32_t> get_unique_history(size_t n_vocab, const std::vector<uint32_t> & history) const;
};

}  // namespace kata

#include "kata/sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kata {

Sampler::Sampler(SamplerConfig cfg) : cfg_(cfg), rng_(cfg.seed) {
    validate_config();
}

void Sampler::validate_config() const {
    if (std::isnan(cfg_.temp) || std::isinf(cfg_.temp)) {
        throw std::invalid_argument("kata: sampler temperature must be a finite number");
    }
    if (std::isnan(cfg_.top_p) || cfg_.top_p <= 0.0f || cfg_.top_p > 1.0f) {
        throw std::invalid_argument("kata: sampler top_p must be in (0, 1]");
    }
    if (std::isnan(cfg_.repeat_penalty) || cfg_.repeat_penalty <= 0.0f) {
        throw std::invalid_argument("kata: sampler repeat_penalty must be positive");
    }
}

void Sampler::ensure_capacity(size_t n_vocab) {
    if (pen_mask_.size() < n_vocab) {
        pen_mask_.assign(n_vocab, 0);
        pen_val_.resize(n_vocab);
    }
}

std::vector<uint32_t> Sampler::get_unique_history(size_t n_vocab, const std::vector<uint32_t> & history) const {
    if (cfg_.repeat_penalty == 1.0f || history.empty()) return {};
    const size_t n = std::min(history.size(), static_cast<size_t>(std::max(0, cfg_.repeat_last_n)));
    std::vector<uint32_t> tokens;
    tokens.reserve(n);
    for (size_t i = history.size() - n; i < history.size(); ++i) {
        const uint32_t t = history[i];
        if (t < n_vocab) tokens.push_back(t);
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
}

void Sampler::apply_penalty(float * logits, size_t n_vocab, const std::vector<uint32_t> & history) const {
    if (!logits) throw std::invalid_argument("kata: sampler logits pointer is null");
    validate_config();
    const auto tokens = get_unique_history(n_vocab, history);
    for (uint32_t t : tokens) {
        const float x = logits[t];
        logits[t] = x > 0.0f ? x / cfg_.repeat_penalty : x * cfg_.repeat_penalty;
    }
}

namespace {
struct MaskGuard {
    std::vector<uint8_t> & mask;
    const std::vector<uint32_t> & tokens;
    ~MaskGuard() {
        for (uint32_t t : tokens) mask[t] = 0;
    }
};
}  // namespace

uint32_t Sampler::sample(const float * logits, size_t n_vocab, const std::vector<uint32_t> & history) {
    if (n_vocab == 0) throw std::runtime_error("kata: sampler needs a vocabulary");
    if (!logits) throw std::invalid_argument("kata: sampler logits pointer is null");
    validate_config();

    auto min_cmp = [](const Cand & a, const Cand & b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    };

    // 1. Greedy path (temp <= 0.0f)
    if (cfg_.temp <= 0.0f) {
        const auto pen_tokens = get_unique_history(n_vocab, history);
        if (pen_tokens.empty()) {
            float max_val = logits[0];
            if (std::isnan(max_val)) throw std::runtime_error("kata: NaN in logits");
            uint32_t best = 0;
            for (size_t v = 1; v < n_vocab; ++v) {
                const float val = logits[v];
                if (std::isnan(val)) throw std::runtime_error("kata: NaN in logits");
                if (val > max_val) {
                    max_val = val;
                    best = static_cast<uint32_t>(v);
                }
            }
            return best;
        }

        ensure_capacity(n_vocab);
        for (uint32_t t : pen_tokens) {
            pen_mask_[t] = 1;
            const float x = logits[t];
            pen_val_[t] = x > 0.0f ? x / cfg_.repeat_penalty : x * cfg_.repeat_penalty;
        }
        MaskGuard guard{ pen_mask_, pen_tokens };

        float max_val = pen_mask_[0] ? pen_val_[0] : logits[0];
        if (std::isnan(max_val)) throw std::runtime_error("kata: NaN in logits");
        uint32_t best = 0;
        for (size_t v = 1; v < n_vocab; ++v) {
            float val = logits[v];
            if (std::isnan(val)) throw std::runtime_error("kata: NaN in logits");
            if (__builtin_expect(pen_mask_[v], 0)) val = pen_val_[v];
            if (val > max_val) {
                max_val = val;
                best = static_cast<uint32_t>(v);
            }
        }
        return best;
    }

    // 2. Non-greedy path (temp > 0.0f)
    const auto pen_tokens = get_unique_history(n_vocab, history);
    const bool has_pen = !pen_tokens.empty();
    if (has_pen) {
        ensure_capacity(n_vocab);
        for (uint32_t t : pen_tokens) {
            pen_mask_[t] = 1;
            const float x = logits[t];
            pen_val_[t] = x > 0.0f ? x / cfg_.repeat_penalty : x * cfg_.repeat_penalty;
        }
    }
    MaskGuard guard{ pen_mask_, pen_tokens };

    auto get_logit = [&](size_t v) -> float {
        float l = logits[v];
        if (std::isnan(l)) throw std::runtime_error("kata: NaN in logits");
        if (__builtin_expect(has_pen && pen_mask_[v], 0)) l = pen_val_[v];
        return l;
    };

    // Case A: Top-K enabled
    if (cfg_.top_k > 0) {
        const size_t K = std::min(n_vocab, static_cast<size_t>(cfg_.top_k));
        heap_.clear();
        heap_.reserve(K);
        for (size_t v = 0; v < K; ++v) {
            heap_.push_back({ get_logit(v), static_cast<uint32_t>(v) });
        }
        std::make_heap(heap_.begin(), heap_.end(), min_cmp);

        for (size_t v = K; v < n_vocab; ++v) {
            const float l = get_logit(v);
            const auto & root = heap_.front();
            if (l > root.logit || (l == root.logit && static_cast<uint32_t>(v) < root.id)) {
                std::pop_heap(heap_.begin(), heap_.end(), min_cmp);
                heap_.back() = { l, static_cast<uint32_t>(v) };
                std::push_heap(heap_.begin(), heap_.end(), min_cmp);
            }
        }
        std::sort_heap(heap_.begin(), heap_.end(), min_cmp);

        const float maxv = heap_[0].logit / cfg_.temp;
        double sum = 0.0;
        for (size_t i = 0; i < K; ++i) {
            const float p = std::exp(heap_[i].logit / cfg_.temp - maxv);
            heap_[i].logit = p;
            sum += p;
        }
        if (sum <= 0.0) return heap_[0].id;

        size_t keep = K;
        if (cfg_.top_p < 1.0f) {
            double acc = 0.0;
            for (size_t i = 0; i < K; ++i) {
                acc += heap_[i].logit;
                if (acc / sum >= cfg_.top_p) {
                    keep = i + 1;
                    sum = acc;
                    break;
                }
            }
        }

        const double r = dist_01_(rng_) * sum;
        double acc = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            acc += heap_[i].logit;
            if (r <= acc) return heap_[i].id;
        }
        return heap_[keep - 1].id;
    }

    // Case B: Top-K disabled (top_k <= 0)
    float max_l = get_logit(0);
    for (size_t v = 1; v < n_vocab; ++v) {
        const float l = get_logit(v);
        if (l > max_l) max_l = l;
    }
    const float maxv = max_l / cfg_.temp;

    // Subcase B1: Pure temperature sampling (top_p >= 1.0f)
    if (cfg_.top_p >= 1.0f) {
        double total_sum = 0.0;
        for (size_t v = 0; v < n_vocab; ++v) {
            const float diff = get_logit(v) / cfg_.temp - maxv;
            if (diff >= -16.0f) total_sum += std::exp(diff);
        }
        if (total_sum <= 0.0) return 0;

        const double r = dist_01_(rng_) * total_sum;
        double acc = 0.0;
        uint32_t chosen = 0;
        for (size_t v = 0; v < n_vocab; ++v) {
            const float diff = get_logit(v) / cfg_.temp - maxv;
            if (diff >= -16.0f) {
                acc += std::exp(diff);
                if (r <= acc) { chosen = static_cast<uint32_t>(v); break; }
            }
        }
        return chosen;
    }

    // Subcase B2: Top-P with Top-K disabled
    constexpr size_t K_NUC = 512;
    const size_t init_k = std::min(n_vocab, K_NUC);
    heap_.clear();
    heap_.reserve(init_k);
    double total_sum = 0.0;

    for (size_t v = 0; v < init_k; ++v) {
        const float l = get_logit(v);
        heap_.push_back({ l, static_cast<uint32_t>(v) });
        const float diff = l / cfg_.temp - maxv;
        if (diff >= -16.0f) total_sum += std::exp(diff);
    }
    std::make_heap(heap_.begin(), heap_.end(), min_cmp);

    for (size_t v = init_k; v < n_vocab; ++v) {
        const float l = get_logit(v);
        const float diff = l / cfg_.temp - maxv;
        if (diff >= -16.0f) total_sum += std::exp(diff);

        const auto & root = heap_.front();
        if (l > root.logit || (l == root.logit && static_cast<uint32_t>(v) < root.id)) {
            std::pop_heap(heap_.begin(), heap_.end(), min_cmp);
            heap_.back() = { l, static_cast<uint32_t>(v) };
            std::push_heap(heap_.begin(), heap_.end(), min_cmp);
        }
    }
    std::sort_heap(heap_.begin(), heap_.end(), min_cmp);

    if (total_sum <= 0.0) return heap_[0].id;

    double acc = 0.0;
    size_t keep = 0;
    for (size_t i = 0; i < init_k; ++i) {
        const float p = std::exp(heap_[i].logit / cfg_.temp - maxv);
        heap_[i].logit = p;
        acc += p;
        if (acc / total_sum >= cfg_.top_p) {
            keep = i + 1;
            break;
        }
    }

    if (keep > 0) {
        const double r = dist_01_(rng_) * acc;
        double cur = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            cur += heap_[i].logit;
            if (r <= cur) return heap_[i].id;
        }
        return heap_[keep - 1].id;
    }

    // Rare fallback when nucleus requires > 512 tokens
    cand_.clear();
    for (size_t v = 0; v < n_vocab; ++v) {
        const float l = get_logit(v);
        const float diff = l / cfg_.temp - maxv;
        if (diff >= -16.0f) {
            cand_.push_back({ std::exp(diff), static_cast<uint32_t>(v) });
        }
    }
    std::sort(cand_.begin(), cand_.end(), [](const auto & a, const auto & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    double c_acc = 0.0;
    size_t c_keep = cand_.size();
    for (size_t i = 0; i < cand_.size(); ++i) {
        c_acc += cand_[i].first;
        if (c_acc / total_sum >= cfg_.top_p) {
            c_keep = i + 1;
            break;
        }
    }
    const double r = dist_01_(rng_) * c_acc;
    double cur = 0.0;
    for (size_t i = 0; i < c_keep; ++i) {
        cur += cand_[i].first;
        if (r <= cur) return cand_[i].second;
    }
    return cand_[c_keep - 1].second;
}

}  // namespace kata

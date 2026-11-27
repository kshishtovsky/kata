#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kata {

// Everything that survives between tokens. One sequence only for now: the
// engine is at the "produce a correct token" stage, batching comes later.
struct ModelState {
    size_t pos = 0;
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    void load(const std::string & path, size_t n_ctx);
    void reset();

    // One token in, logits for the next token out (n_vocab floats, owned by the
    // model and valid until the next call).
    const float * forward(uint32_t token, ModelState & st);

    // A chunk of tokens in one pass. Only the logits of the LAST token are
    // returned: a prefill of a prompt does not need the other 511 rows of a
    // 128k-wide lm_head, and computing them would cost more than the model.
    const float * prefill(const uint32_t * tokens, size_t n, ModelState & st);

    // Same pass, but keeps the logits of EVERY row so a verifier can compare a
    // drafted block against the target's greedy choice at each position. Costs
    // n * n_vocab floats (a 8 x 128k block is 4 MB) and one extra lm_head per
    // row, so it is only for verification, never for a plain prefill.
    // `out` must hold n * n_vocab() floats; out[i*n_vocab + v] is row i.
    void prefill_logits(const uint32_t * tokens, size_t n, ModelState & st, float * out);

    // Rewinds the sequential state to `pos`. The KV cache needs no work (slots
    // past `pos` are simply overwritten), but the shortconv ring buffers and the
    // RNG-free position counter must be put back, or a rejected speculative tail
    // silently corrupts the continuation.
    void truncate(ModelState & st, size_t pos);

    // Teacher-forced negative log-likelihood over tokens[1 .. n-1], each chunk
    // of n_ctx tokens scored in a fresh context (the llama.cpp perplexity
    // convention, so the two are comparable). Returns the mean of
    // exp(-log p) over counted tokens and reports the token count.
    double perplexity(const uint32_t * tokens, size_t n, size_t n_ctx_chunk, size_t * counted = nullptr);

    size_t n_vocab() const { return n_vocab_; }
    size_t n_embd() const { return n_embd_; }
    size_t n_layer() const { return n_layer_; }

    // Expert routing telemetry (EXP-0053). Call heatmap_start(), run whatever
    // mix of forward/prefill you care about, then heatmap_take() to collect
    // per-[layer][expert] selection counts and disarm the collection.
    void heatmap_start();
    std::vector<uint64_t> heatmap_take();
    size_t n_expert() const { return n_expert_; }

    // Per-token routing record (EXP-0078): one entry per decode step, listing
    // the experts that ONE token chose in each MoE layer. The pooled heatmap
    // above averages over a prompt and cannot answer whether the 152 expert
    // selections a token makes are independently spread (low variance) or
    // clumped (a single unlucky token could stall a hybrid CPU/GPU schedule).
    struct TokenRouting {
        std::vector<uint32_t> expert;   // [n_moe_layer * n_expert_used]
        size_t n_moe_layer = 0;
        size_t n_expert_used = 0;
    };
    void routing_record_start();
    std::vector<TokenRouting> routing_record_take();
    size_t n_moe_layer_count() const { return n_moe_layer_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    size_t n_vocab_ = 0, n_embd_ = 0, n_layer_ = 0, n_expert_ = 0, n_moe_layer_ = 0;
};

} // namespace kata
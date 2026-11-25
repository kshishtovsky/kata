#pragma once
#include <cstdint>
#include <memory>
#include <string>

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

    size_t n_vocab() const { return n_vocab_; }
    size_t n_embd() const { return n_embd_; }
    size_t n_layer() const { return n_layer_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    size_t n_vocab_ = 0, n_embd_ = 0, n_layer_ = 0;
};

} // namespace kata
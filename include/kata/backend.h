// Compute backend boundary.
//
// Every kernel family lives behind this interface so a GPU path can be added
// without touching call sites. CPU is the only implementation today; the names
// here are the surface the CPU path already needs for a forward pass, not a
// guess at what a future GPU would want.
#pragma once

#include <cstddef>

#include "kata/types.h"

namespace kata {

// Normalisation along a contiguous row.
struct RmsNormArgs {
    const float * x;    // [rows * cols]
    const float * w;    // [cols], optional (nullptr = no weight)
    float * out;        // [rows * cols]
    size_t rows;
    size_t cols;
    float eps;
};

// Rotary position embedding applied in place to one token's head vectors.
struct RopeArgs {
    float * x;          // [n_heads * head_dim]
    size_t n_heads;
    size_t head_dim;
    size_t pos;
    float theta_base;
    // Interleaved (pairs 2i,2i+1) vs half-split (pairs i,i+head_dim/2).
    // Not interchangeable: the wrong choice keeps norms valid and quality broken.
    bool interleaved;
};

// Quantised matrix-vector products. `out[r] = dot(W[r], x)`.
struct GemvArgs {
    size_t rows;
    size_t cols;
    float * out;
    const void * w;
    const void * x;
};

// One backend = one complete set of kernels. A backend may fall back to another
// for kernels it does not implement, but it must never silently return garbage:
// `available()` gates selection.
class Backend {
public:
    virtual ~Backend() = default;
    virtual const char * name() const = 0;
    virtual bool available() const = 0;

    virtual void rms_norm(const RmsNormArgs &) const = 0;
    virtual void rope(const RopeArgs &) const = 0;

    virtual void gemv_q4_K_q8_K(const GemvArgs &) const = 0;
    virtual void gemv_q6_K_q8_K(const GemvArgs &) const = 0;
};

// Process-wide backend selection. Picks the fastest available on first call.
const Backend & backend();

} // namespace kata
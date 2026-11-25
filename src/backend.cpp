#include "kata/backend.h"

#include "kata/cpu.h"
#include "kata/gemv.h"
#include "kata/norm.h"
#include "kata/rope.h"

namespace kata {
namespace {

// The CPU backend is the reference: it is selected whenever nothing faster is
// available, so it must be correct on every host, not just x86.
class CpuBackend final : public Backend {
public:
    const char * name() const override { return "cpu"; }
    bool available() const override { return true; }

    void rms_norm(const RmsNormArgs & a) const override {
        kata::rms_norm(a.rows, a.cols, a.out, a.x, a.w, a.eps);
    }

    void rope(const RopeArgs & a) const override {
        kata::rope(a.n_heads, a.head_dim, a.x, a.pos, a.theta_base, a.interleaved);
    }

    void gemv_q4_K_q8_K(const GemvArgs & a) const override {
        gemv_q4_K_q8_K_best()(a.rows, a.cols, a.out, static_cast<const block_q4_K *>(a.w),
                              static_cast<const block_q8_K *>(a.x));
    }

    void gemv_q6_K_q8_K(const GemvArgs & a) const override {
        gemv_q6_K_q8_K_best()(a.rows, a.cols, a.out, static_cast<const block_q6_K *>(a.w),
                              static_cast<const block_q8_K *>(a.x));
    }
};

} // namespace

const Backend & backend() {
    static const CpuBackend cpu;
    return cpu;
}

} // namespace kata
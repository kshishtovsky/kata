#include "kata/model.h"
#include "kata/shortconv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "kata/attn.h"
#include "kata/gguf.h"
#include "kata/gemm.h"
#include "kata/gemv.h"
#include "kata/norm.h"
#include "kata/pool.h"
#include "kata/prefill.h"
#include "kata/quant.h"
#include "kata/rope.h"
#include "kata/simd.h"
#include "kata/types.h"

namespace kata {

namespace {

// LFM2 attention layers all share one kv-head count; shortconv layers have none.
struct Layer {
    bool attn = false;
    bool moe = false;
    const void * attn_norm = nullptr;
    const void * ffn_norm = nullptr;
    // attention
    const void * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
    const void * q_norm = nullptr, * k_norm = nullptr;
    uint32_t tq = 0, tk = 0, tv = 0, to = 0;
    // shortconv
    const void * conv = nullptr, * in_proj = nullptr, * out_proj = nullptr;
    uint32_t t_in = 0, t_out = 0;
    // dense ffn
    const void * gate = nullptr, * up = nullptr, * down = nullptr;
    uint32_t tg = 0, tu = 0, td = 0;
    // moe
    const void * gate_inp = nullptr, * exp_probs_b = nullptr;
    const void * gate_exps = nullptr, * up_exps = nullptr, * down_exps = nullptr;
    uint32_t t_ge = 0, t_ue = 0, t_de = 0;
};

// Dispatches on the gguf type, so the forward pass never assumes a format.
void gemv_quant(uint32_t type, size_t rows, size_t cols, float * out,
                const void * w, const block_q8_K * aq, ThreadPool & pool) {
    switch (type) {
        case 12: gemv_q4_K_q8_K_mt(rows, cols, out, static_cast<const block_q4_K *>(w), aq, pool); break;
        case 14: gemv_q6_K_q8_K_mt(rows, cols, out, static_cast<const block_q6_K *>(w), aq, pool); break;
        default: throw std::runtime_error("kata: unsupported weight type in forward pass: " + std::to_string(type));
    }
}

const void * require(const GgufFile & f, const std::string & name) {
    const GgufTensor * t = f.find(name);
    if (!t) throw std::runtime_error("kata: missing tensor " + name);
    return f.data(*t);
}

uint32_t type_of(const GgufFile & f, const std::string & name) {
    const GgufTensor * t = f.find(name);
    if (!t) throw std::runtime_error("kata: missing tensor " + name);
    return t->type;
}

inline void silu_inplace(float * x, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

// SwiGLU product: g = silu(g) * u. The same pass that folds the two expert
// projections together; keeping it in one function lets the MT split cover it.
inline void silu_mul(float * g, const float * u, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float x = g[i];
        g[i] = (x / (1.0f + std::exp(-x))) * u[i];
    }
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2,fma")))
static inline void vec_add_inplace(float * dst, const float * src, size_t n) {
    size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        _mm256_storeu_ps(dst + j, _mm256_add_ps(_mm256_loadu_ps(dst + j), _mm256_loadu_ps(src + j)));
    }
    for (; j < n; ++j) dst[j] += src[j];
}

__attribute__((target("avx2,fma")))
static inline void moe_accumulate(float * out, const float * down, size_t nu, size_t n_embd, const float * w, float wsum) {
    for (size_t i = 0; i < nu; ++i) {
        const float wj = w[i] / wsum;
        const float * d = down + i * n_embd;
        const __m256 wv = _mm256_set1_ps(wj);
        if (i == 0) {
            size_t j = 0;
            for (; j + 8 <= n_embd; j += 8) {
                _mm256_storeu_ps(out + j, _mm256_mul_ps(wv, _mm256_loadu_ps(d + j)));
            }
            for (; j < n_embd; ++j) out[j] = wj * d[j];
        } else {
            size_t j = 0;
            for (; j + 8 <= n_embd; j += 8) {
                _mm256_storeu_ps(out + j, _mm256_fmadd_ps(wv, _mm256_loadu_ps(d + j), _mm256_loadu_ps(out + j)));
            }
            for (; j < n_embd; ++j) out[j] += wj * d[j];
        }
    }
}
#endif

} // namespace

struct Model::Impl {
    GgufFile file;
    size_t n_ctx = 0;
    size_t n_embd = 0, n_vocab = 0, n_layer = 0;
    size_t n_head = 0, n_head_kv = 0, head_dim = 0;
    size_t n_ff = 0, n_ff_exp = 0;
    size_t n_expert = 0, n_expert_used = 0;
    float eps = 1e-5f, rope_base = 1e4f;
    size_t l_cache = 3;  // kernel width of the shortconv
    bool sigmoid_gate = true;

    std::vector<Layer> layers;
    const void * tok_embd = nullptr;
    uint32_t t_tok_embd = 0;
    const void * out_norm = nullptr;

    // Per-layer attention slabs, then everything else, all float32.
    std::vector<float> k_cache, v_cache;   // [n_layer][n_ctx][n_head_kv*head_dim]
    std::vector<float> conv_state;         // [n_layer][l_cache-1][n_embd]
    std::vector<float> x, prev, norm_out, q, k, v, attn_out, bcx;
    std::vector<float> ffn_gate, ffn_up, ffn_down, moe_acc;
    std::vector<float> logits;
    std::vector<float> attn_scratch;   // decode attention scratch, sized once
    std::vector<float> d_logits, d_probs, d_w, d_gu, d_down;
    std::vector<std::pair<float, uint32_t>> d_sel;
    std::vector<block_q8_K> d_aq_down;
    std::vector<const block_q8_K *> d_xd, d_xgu;
    std::vector<const void *> d_wd, d_wgu;
    std::vector<block_q8_K> aq;
    // Decode-path scratch that used to be allocated per call. Steady-state
    // decode was paying a malloc/free for every one of the 30 shortconv layers.
    std::vector<block_q8_K> spr_3, spr_ao;
    std::vector<float> spr_out;
    float * logits_ptr = nullptr;

    // Prefill works on a chunk of tokens at a time; every buffer below is
    // [ubatch][width] and reused across chunks.
    size_t ubatch = 1;
    std::vector<float> px, pprev, pnorm, pffn, pffn2, pffn_out;
    std::vector<float> pq, pk, pv, pattn, pbcx, pexpacc;
    std::vector<float> pconv_out, pbx, gact, ggate, gup;
    std::vector<block_q8_K> paq, paq2;
    // MoE scratch, sized once. These were locals inside moe_ffn_rows, so every
    // one of the 38 MoE layers re-allocated ~100 MB at ubatch 1024: 151 ms of a
    // 1734 ms prefill, all of it serial inside the pool's idle window.
    struct ExpertPair { uint32_t expert, token; float weight; };
    std::vector<float> mprobs;
    std::vector<ExpertPair> mpairs;
    std::vector<std::pair<float, uint32_t>> msel;
    std::vector<block_q8_K> maq_g, maq_d;
    std::vector<float> mgout, muout, mdout;
    std::vector<uint32_t> mptok;
    std::vector<float> mpw;
    ThreadPool pool{ThreadPool::default_threads()};

    // Set KATA_DUMP=<dir> to write intermediates for diffing against a reference.
    std::string dump_dir;
    // Per-phase wall clock, printed only when KATA_PROF is set.
    bool prof = std::getenv("KATA_PROF") != nullptr;
    size_t dump_pos = 0;
    bool router_log = std::getenv("KATA_ROUTER") != nullptr;   // KATA_DUMP_POS: which decode position the per-layer dumps capture
    double t_emb = 0, t_mix = 0, t_ffn = 0, t_lm = 0, t_norm = 0, t_mm = 0, t_q = 0;
    size_t n_mm = 0;
    struct Tick { double * acc; std::chrono::steady_clock::time_point t0; bool on;
        Tick(double * a, bool o) : acc(a), on(o), t0(std::chrono::steady_clock::now()) {}
        ~Tick() { if (on) *acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } };
    void dump(const char * name, const float * v, size_t n) const {
        if (dump_dir.empty()) return;
        const std::string path = dump_dir + "/k_" + name + ".bin";
        FILE * f = std::fopen(path.c_str(), "wb");
        if (!f) return;
        std::fwrite(v, sizeof(float), n, f);
        std::fclose(f);
    }

    void alloc() {
        const size_t kv = n_head_kv * head_dim;
        k_cache.assign(n_layer * n_ctx * kv, 0.0f);
        v_cache.assign(n_layer * n_ctx * kv, 0.0f);
        conv_state.assign(n_layer * (l_cache - 1) * n_embd, 0.0f);
        x.assign(n_embd, 0.0f);
        prev.assign(n_embd, 0.0f);
        norm_out.assign(n_embd, 0.0f);
        q.assign(n_head * head_dim, 0.0f);
        k.assign(n_head_kv * head_dim, 0.0f);
        v.assign(n_head_kv * head_dim, 0.0f);
        attn_out.assign(n_head * head_dim, 0.0f);
        bcx.assign(3 * n_embd, 0.0f);
        ffn_gate.assign(std::max(n_ff, n_ff_exp), 0.0f);
        ffn_up.assign(std::max(n_ff, n_ff_exp), 0.0f);
        ffn_down.assign(n_embd, 0.0f);
        moe_acc.assign(n_embd, 0.0f);
        logits.assign(n_vocab, 0.0f);
        logits_ptr = logits.data();
        aq.assign(std::max({n_embd, n_ff, n_ff_exp}) / QK_K + 1, {});
        // Decode-path scratch. moe_ffn used to allocate eight vectors per layer,
        // 38 layers per token; the same reuse that EXP-0031 brought to prefill.
        const size_t nu = n_expert_used;
        d_logits.assign(n_expert, 0.0f);
        d_probs.assign(n_expert, 0.0f);
        d_sel.resize(n_expert);
        d_w.assign(nu, 0.0f);
        d_gu.assign(2 * nu * n_ff_exp, 0.0f);
        d_aq_down.assign(nu * (n_ff_exp / QK_K), {});
        d_down.assign(nu * n_embd, 0.0f);
        d_xd.resize(nu);
        d_wd.resize(nu);
        d_wgu.resize(2 * nu);
        d_xgu.resize(2 * nu);
        // grouped decode attention wants n_heads/n_kv_heads floats per cached
        // position plus 8 for the tail.
        attn_scratch.assign(n_ctx * (n_head / n_head_kv) + 8, 0.0f);
        spr_3.assign(3 * n_embd / QK_K, {});
        spr_ao.assign(n_embd / QK_K, {});
        spr_out.assign(n_embd, 0.0f);

        // 1024 measured 253.6 tok/s vs 231.8 at 512 on a 544-token prompt: the
        // per-layer expert buckets get twice as many rows, so the small-m GEMM
        // penalty is paid on half as many calls.
        ubatch = std::min<size_t>(n_ctx, 1024);
        if (const char * d = std::getenv("KATA_DUMP_POS")) dump_pos = std::strtoul(d, nullptr, 10);
        if (const char * u = std::getenv("KATA_UBATCH")) ubatch = std::max<size_t>(1, std::strtoul(u, nullptr, 10));
        const size_t pkv = n_head_kv * head_dim;
        const size_t wide = std::max(n_ff, n_ff_exp);
        px.assign(ubatch * n_embd, 0.0f);
        pprev.assign(ubatch * n_embd, 0.0f);
        pnorm.assign(ubatch * n_embd, 0.0f);
        pffn.assign(ubatch * wide, 0.0f);
        pffn2.assign(ubatch * wide, 0.0f);
        pffn_out.assign(ubatch * n_embd, 0.0f);
        pq.assign(ubatch * n_head * head_dim, 0.0f);
        pk.assign(ubatch * pkv, 0.0f);
        pv.assign(ubatch * pkv, 0.0f);
        pattn.assign(ubatch * n_head * head_dim, 0.0f);
        pbcx.assign(ubatch * 3 * n_embd, 0.0f);
        pexpacc.assign(ubatch * n_embd, 0.0f);
        pconv_out.assign(ubatch * n_embd, 0.0f);
        pbx.assign((ubatch + 4) * n_embd, 0.0f);
        gact.assign(ubatch * n_embd, 0.0f);
        ggate.assign(ubatch * n_ff_exp, 0.0f);
        gup.assign(ubatch * n_ff_exp, 0.0f);
        paq.assign(ubatch * std::max({n_embd, n_ff, n_ff_exp}) / QK_K, {});
        paq2.assign(ubatch * std::max({n_embd, n_ff, n_ff_exp}) / QK_K, {});
        const size_t np_max = ubatch * n_expert_used;
        mprobs.assign(ubatch * n_expert, 0.0f);
        mpairs.resize(np_max);
        msel.resize(n_expert);
        maq_g.assign(np_max * (n_embd / QK_K), {});
        maq_d.assign(np_max * (n_ff_exp / QK_K), {});
        mgout.assign(np_max * n_ff_exp, 0.0f);
        muout.assign(np_max * n_ff_exp, 0.0f);
        mdout.assign(np_max * n_embd, 0.0f);
        mptok.resize(np_max);
        mpw.resize(np_max);
    }

    // Quantises the activation block and runs one quantised matvec.
    void mm(uint32_t type, size_t rows, size_t cols, float * out, const void * w, const float * in) {
        const auto q0 = std::chrono::steady_clock::now();
        quantize_row_q8_K(in, aq.data(), cols);
        const auto q1 = std::chrono::steady_clock::now();
        gemv_quant(type, rows, cols, out, w, aq.data(), pool);
        const auto q2 = std::chrono::steady_clock::now();
        if (prof) {
            t_q += std::chrono::duration<double, std::milli>(q1 - q0).count();
            t_mm += std::chrono::duration<double, std::milli>(q2 - q1).count();
            ++n_mm;
        }
    }

    // `n` matvecs sharing one already-quantised activation, issued as a single
    // pool wake-up. This is the fix for EXP-0020: 1536-row calls are too small
    // to amortise a fan-out.
    void mm_batch(uint32_t type, size_t n, size_t rows, size_t cols, float * out,
                  const void * const * w, const block_q8_K * const * x) {
        const auto q0 = std::chrono::steady_clock::now();
        if (type == 12) {
            const block_q4_K * wp[8];
            for (size_t i = 0; i < n; ++i) wp[i] = static_cast<const block_q4_K *>(w[i]);
            gemv_q4_K_q8_K_mt_batch(n, rows, cols, out, wp, x, pool);
        } else {
            const block_q6_K * wp[8];
            for (size_t i = 0; i < n; ++i) wp[i] = static_cast<const block_q6_K *>(w[i]);
            gemv_q6_K_q8_K_mt_batch(n, rows, cols, out, wp, x, pool);
        }
        const auto q1 = std::chrono::steady_clock::now();
        if (prof) {
            t_mm += std::chrono::duration<double, std::milli>(q1 - q0).count();
            n_mm += n;
        }
    }

    void embed(uint32_t token) {
        if (t_tok_embd != 14) throw std::runtime_error("kata: token_embd must be q6_K");
        const size_t nb = n_embd / QK_K;
        const uint8_t * base = static_cast<const uint8_t *>(tok_embd) + token * nb * sizeof(block_q6_K);
        dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(base), x.data(), n_embd);
    }

    void shortconv(size_t il, const Layer & L, const std::vector<float> & in, std::vector<float> & out) {
        // in_proj emits b, c, x as three contiguous chunks of n_embd.
        quantize_row_q8_K(in.data(), spr_3.data(), n_embd);
        gemv_quant(L.t_in, 3 * n_embd, n_embd, bcx.data(), L.in_proj, spr_3.data(), pool);

        const size_t d_conv = l_cache - 1;
        float * hist = conv_state.data() + il * d_conv * n_embd;
        const float * kern = static_cast<const float *>(L.conv);
        const float * b = bcx.data();
        const float * c = bcx.data() + n_embd;
        const float * xx = bcx.data() + 2 * n_embd;
        shortconv_step(b, c, xx, hist, kern, d_conv, n_embd, spr_out.data());
        quantize_row_q8_K(spr_out.data(), spr_ao.data(), n_embd);
        gemv_quant(L.t_out, n_embd, n_embd, out.data(), L.out_proj, spr_ao.data(), pool);
    }

    void attention(size_t il, const Layer & L, const std::vector<float> & in, size_t pos) {
        const size_t qw = n_head * head_dim;
        const size_t kw = n_head_kv * head_dim;
        const size_t vw = n_head_kv * head_dim;
        if (L.tq == 12 && L.tk == 12 && L.tv == 14) {
            const auto q0 = std::chrono::steady_clock::now();
            quantize_row_q8_K(in.data(), aq.data(), n_embd);
            const auto q1 = std::chrono::steady_clock::now();
            gemv_qkv_mt(qw, kw, vw, n_embd, q.data(), k.data(), v.data(),
                        static_cast<const block_q4_K *>(L.wq),
                        static_cast<const block_q4_K *>(L.wk),
                        static_cast<const block_q6_K *>(L.wv),
                        aq.data(), pool);
            const auto q2 = std::chrono::steady_clock::now();
            if (prof) {
                t_q += std::chrono::duration<double, std::milli>(q1 - q0).count();
                t_mm += std::chrono::duration<double, std::milli>(q2 - q1).count();
                n_mm += 3;
            }
        } else {
            mm(L.tq, qw, n_embd, q.data(), L.wq, in.data());
            mm(L.tk, kw, n_embd, k.data(), L.wk, in.data());
            mm(L.tv, vw, n_embd, v.data(), L.wv, in.data());
        }

        rms_norm(n_head, head_dim, q.data(), q.data(), static_cast<const float *>(L.q_norm), eps);
        rms_norm(n_head_kv, head_dim, k.data(), k.data(), static_cast<const float *>(L.k_norm), eps);

        // LFM2 is a NeoX-rope model: pairs are (i, i + head_dim/2).
        rope(n_head, head_dim, q.data(), pos, rope_base, false);
        rope(n_head_kv, head_dim, k.data(), pos, rope_base, false);

        const size_t kv = n_head_kv * head_dim;
        float * kc = k_cache.data() + (il * n_ctx + pos) * kv;
        float * vc = v_cache.data() + (il * n_ctx + pos) * kv;
        std::memcpy(kc, k.data(), kv * sizeof(float));
        std::memcpy(vc, v.data(), kv * sizeof(float));

        AttnArgs a{};
        a.q = q.data();
        a.k_cache = k_cache.data() + il * n_ctx * kv;
        a.v_cache = v_cache.data() + il * n_ctx * kv;
        a.n_kv = pos + 1;
        a.n_heads = n_head;
        a.n_kv_heads = n_head_kv;
        a.head_dim = head_dim;
        a.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        a.out = attn_out.data();
        attention_decode_grouped(a, attn_scratch.data());

        mm(L.to, n_embd, n_head * head_dim, attn_out.data(), L.wo, attn_out.data());
    }

    void dense_ffn(size_t il, const Layer & L, const std::vector<float> & in, std::vector<float> & out) {
        mm(L.tg, n_ff, n_embd, ffn_gate.data(), L.gate, in.data());
        if (!dump_dir.empty()) dump(("dgate" + std::to_string(il)).c_str(), ffn_gate.data(), n_ff);
        mm(L.tu, n_ff, n_embd, ffn_up.data(), L.up, in.data());
        if (!dump_dir.empty()) dump(("dup" + std::to_string(il)).c_str(), ffn_up.data(), n_ff);
        silu_mul(ffn_gate.data(), ffn_up.data(), n_ff);
        mm(L.td, n_embd, n_ff, out.data(), L.down, ffn_gate.data());
    }

    void moe_ffn(size_t il, const Layer & L, const std::vector<float> & in, std::vector<float> & out) {
        float * logits_e = d_logits.data();
        gemv_f32(n_expert, n_embd, logits_e, static_cast<const float *>(L.gate_inp), in.data());

        float * probs = d_probs.data();
        for (size_t e = 0; e < n_expert; ++e)
            probs[e] = sigmoid_gate ? 1.0f / (1.0f + std::exp(-logits_e[e])) : logits_e[e];

        // Selection uses probs + bias; the weight comes from the unbiased prob.
        const float * bias = static_cast<const float *>(L.exp_probs_b);
        auto * sel = d_sel.data();
        for (size_t e = 0; e < n_expert; ++e) sel[e] = { probs[e] + bias[e], static_cast<uint32_t>(e) };
        std::partial_sort(sel, sel + n_expert_used, sel + n_expert,
                          [](const auto & a, const auto & b) { return a.first > b.first; });

        const size_t nu = n_expert_used;
        float * w = d_w.data();
        float wsum = 0.0f;
        for (size_t i = 0; i < nu; ++i) { w[i] = probs[sel[i].second]; wsum += w[i]; }
        if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
        if (!dump_dir.empty()) {
            dump(("moelogits" + std::to_string(il)).c_str(), logits_e, n_expert);
            dump(("moeprobs" + std::to_string(il)).c_str(), probs, n_expert);
            std::vector<float> biased(n_expert), wsel(n_expert_used), wn(n_expert_used);
            for (size_t e = 0; e < n_expert; ++e) biased[e] = probs[e] + bias[e];
            for (size_t i = 0; i < n_expert_used; ++i) { wsel[i] = w[i]; wn[i] = w[i] / wsum; }
            dump(("moebiased" + std::to_string(il)).c_str(), biased.data(), n_expert);
            dump(("moeweights" + std::to_string(il)).c_str(), wsel.data(), n_expert_used);
            dump(("moewnorm" + std::to_string(il)).c_str(), wn.data(), n_expert_used);
        }

        // One expert's matrix is a slice of the stacked tensor. The four experts
        // share one activation row, so gate and up go out as one batched call of
        // 8 matvecs instead of 8 separate pool wake-ups (EXP-0021).
        const size_t sb_g = n_ff_exp * n_embd / QK_K * ggml_type_size(L.t_ge);
        const size_t sb_u = n_ff_exp * n_embd / QK_K * ggml_type_size(L.t_ue);
        const size_t sb_d = n_embd * n_ff_exp / QK_K * ggml_type_size(L.t_de);
        const uint8_t * bg = static_cast<const uint8_t *>(L.gate_exps);
        const uint8_t * bu = static_cast<const uint8_t *>(L.up_exps);
        const uint8_t * bd = static_cast<const uint8_t *>(L.down_exps);

        // The batched path bypasses mm(), so the shared activation row has to be
        // quantised here.
        const auto q0 = std::chrono::steady_clock::now();
        quantize_row_q8_K(in.data(), aq.data(), n_embd);
        if (prof) t_q += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - q0).count();
        const void ** wgu = d_wgu.data();
        const block_q8_K ** xgu = d_xgu.data();
        for (size_t i = 0; i < nu; ++i) {
            const uint32_t e = sel[i].second;
            wgu[2 * i]     = bg + e * sb_g;
            wgu[2 * i + 1] = bu + e * sb_u;
            xgu[2 * i] = xgu[2 * i + 1] = aq.data();
        }
        // ffn_gate holds the gate results of all experts packed [i*n_ff_exp + j].
        float * gu = d_gu.data();
        if (L.t_ge == L.t_ue) {
            mm_batch(L.t_ge, 2 * nu, n_ff_exp, n_embd, gu, wgu, xgu);
        } else {
            for (size_t i = 0; i < nu; ++i) {
                mm_batch(L.t_ge, 1, n_ff_exp, n_embd, gu + (2 * i) * n_ff_exp, wgu + 2 * i, xgu + 2 * i);
                mm_batch(L.t_ue, 1, n_ff_exp, n_embd, gu + (2 * i + 1) * n_ff_exp, wgu + 2 * i + 1, xgu + 2 * i + 1);
            }
        }

        // Down projection: each expert now has a different activation, so those
        // four still share one call, with per-slot quantised copies.
        block_q8_K * aq_down = d_aq_down.data();
        const block_q8_K ** xd = d_xd.data();
        const void ** wd = d_wd.data();
        for (size_t i = 0; i < nu; ++i) {
            float * g = gu + (2 * i) * n_ff_exp;
            float * u = gu + (2 * i + 1) * n_ff_exp;
            silu_mul(g, u, n_ff_exp);
            block_q8_K * dst = aq_down + i * (n_ff_exp / QK_K);
            quantize_row_q8_K(g, dst, n_ff_exp);
            xd[i] = dst;
            wd[i] = bd + sel[i].second * sb_d;
            if (!dump_dir.empty() && il == 2) {
                dump(("xgate" + std::to_string(i)).c_str(), g, n_ff_exp);
            }
        }
        float * down = d_down.data();
        mm_batch(L.t_de, nu, n_embd, n_ff_exp, down, wd, xd);

        if (!dump_dir.empty() && il == 2) {
            for (size_t i = 0; i < nu; ++i) {
                dump(("xdown" + std::to_string(i)).c_str(), down + i * n_embd, n_embd);
            }
        }
#if defined(__x86_64__) || defined(__i386__)
        moe_accumulate(out.data(), down, nu, n_embd, w, wsum);
#else
        std::fill(out.begin(), out.end(), 0.0f);
        for (size_t i = 0; i < nu; ++i) {
            const float wj = w[i] / wsum;
            const float * d = down + i * n_embd;
            for (size_t j = 0; j < n_embd; ++j) out[j] += wj * d[j];
        }
#endif
    }


    // ---- prefill helpers -------------------------------------------------

    // One quantised matmul over m token rows. mi is the block index of a row.
    size_t nmac = 0;
    void mmn(uint32_t type, size_t rows, size_t cols, float * out, size_t ldc,
             const void * w, const block_q8_K * a, size_t m) {
        switch (type) {
            case 12: gemm_q4_K_q8_K_mt(m, rows, cols, out, ldc,
                                       static_cast<const block_q4_K *>(w), a, pool); break;
            case 14: gemm_q6_K_q8_K_mt(m, rows, cols, out, ldc,
                                       static_cast<const block_q6_K *>(w), a, pool); break;
            default: throw std::runtime_error("kata: unsupported weight type in prefill: " + std::to_string(type));
        }
    }

    // Quantises m contiguous activation rows of `cols` floats each.
    void q8_rows(const float * a, size_t m, size_t cols, block_q8_K * dst) {
        const size_t nb = cols / QK_K;
        for (size_t i = 0; i < m; ++i) quantize_row_q8_K(a + i * cols, dst + i * nb, cols);
    }

    // Same, but the source rows are gathered from a separate token buffer. Fusing
    // the gather into the quantiser removes one full copy of every expert input:
    // at 544 tokens x 38 layers x 4 experts that was 660 MB of memcpy.
    void q8_rows_gather(const float * a, const uint32_t * rows, size_t m, size_t cols,
                        block_q8_K * dst) {
        const size_t nb = cols / QK_K;
        for (size_t i = 0; i < m; ++i) quantize_row_q8_K(a + rows[i] * cols, dst + i * nb, cols);
    }

    // The expert bucket's glue (gather, SiLU, re-quantise) was single-threaded
    // and cost 1.03 s of a 3.18 s prefill: 64 buckets per layer, 38 layers, and
    // ~425 us of serial work each against ~0 GEMM time saved by shrinking the
    // batch. These three walk whole rows, so the split is by row and exact.
    unsigned band(size_t n, size_t per) const { return (unsigned)((n + per - 1) / per); }

    void q8_rows_gather_mt(const float * a, const uint32_t * rows, size_t m, size_t cols,
                           block_q8_K * dst) {
        const size_t nb = cols / QK_K;
        const unsigned nw = std::min<unsigned>(pool.size(), band(m, 4));
        if (nw <= 1) { q8_rows_gather(a, rows, m, cols, dst); return; }
        const size_t per = (m + nw - 1) / nw;
        pool.run([&](unsigned w) {
            const size_t lo = std::min(m, w * per), hi = std::min(m, lo + per);
            for (size_t i = lo; i < hi; ++i)
                quantize_row_q8_K(a + rows[i] * cols, dst + i * nb, cols);
        });
    }

    void q8_rows_mt(const float * a, size_t m, size_t cols, block_q8_K * dst) {
        const size_t nb = cols / QK_K;
        const unsigned nw = std::min<unsigned>(pool.size(), band(m, 4));
        if (nw <= 1) { q8_rows(a, m, cols, dst); return; }
        const size_t per = (m + nw - 1) / nw;
        pool.run([&](unsigned w) {
            const size_t lo = std::min(m, w * per), hi = std::min(m, lo + per);
            for (size_t i = lo; i < hi; ++i)
                quantize_row_q8_K(a + i * cols, dst + i * nb, cols);
        });
    }

    void silu_mul_mt(float * g, const float * u, size_t n) {
        const unsigned nw = std::min<unsigned>(pool.size(), band(n, 2048));
        if (nw <= 1) { silu_mul(g, u, n); return; }
        const size_t per = (n + nw - 1) / nw;
        pool.run([&](unsigned w) {
            const size_t lo = std::min(n, w * per), hi = std::min(n, lo + per);
            silu_mul(g + lo, u + lo, hi - lo);
        });
    }

    // dst[rows[i]] += w[i] * src[i]. Scalar before: 2048 float ops per expert
    // pair, 82k pairs on a 544-token prompt, entirely single-threaded.
#if defined(__x86_64__) || defined(__i386__)
    __attribute__((target("avx2,fma")))
#endif
    void scatter_add(const float * src, const uint32_t * rows, const float * w, size_t m,
                     size_t cols, float * dst) {
        for (size_t i = 0; i < m; ++i) {
            const float * s = src + i * cols;
            float * d = dst + rows[i] * cols;
#if defined(__x86_64__) || defined(__i386__)
            const __m256 wv = _mm256_set1_ps(w[i]);
            size_t j = 0;
            for (; j + 8 <= cols; j += 8)
                _mm256_storeu_ps(d + j, _mm256_fmadd_ps(wv, _mm256_loadu_ps(s + j),
                                                        _mm256_loadu_ps(d + j)));
#else
            size_t j = 0;
#endif
            for (; j < cols; ++j) d[j] += w[i] * s[j];
        }
    }

    void embed_rows(const uint32_t * tokens, size_t m) {
        if (t_tok_embd != 14) throw std::runtime_error("kata: token_embd must be q6_K");
        const size_t nb = n_embd / QK_K;
        const uint8_t * base = static_cast<const uint8_t *>(tok_embd);
        for (size_t i = 0; i < m; ++i)
            dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(base + tokens[i] * nb * sizeof(block_q6_K)),
                                px.data() + i * n_embd, n_embd);
    }

    // shortconv over a whole chunk. One row was the encode path; here every row
    // has its own 2-token history, and the recurrence within the chunk is the
    // bx lives in one buffer in front of the history: [d_conv history rows][m chunk rows].
    // Row i of the chunk is then row d_conv+i, its two inputs are the rows before it,
    // and the carry-over is a memcpy of the tail. The old form shifted the history
    // by hand inside the row loop, which serialised the layer and read out of
    // bounds whenever a chunk was shorter than d_conv.

    void shortconv_rows(size_t il, const Layer & L, size_t m) {
        const size_t nc3 = 3 * n_embd;
        const size_t nc = n_embd;
        const size_t d_conv = l_cache - 1;

        q8_rows_mt(pnorm.data(), m, n_embd, paq.data());
        mmn(L.t_in, nc3, n_embd, pbcx.data(), nc3, L.in_proj, paq.data(), m);

        float * seq = pbx.data();
        float * hist = conv_state.data() + il * d_conv * nc;
        std::memcpy(seq, hist, d_conv * nc * sizeof(float));
        float * bx = seq + d_conv * nc;
        shortconv_bx(pbcx.data(), bx, m, nc);
        shortconv_conv(bx, pbcx.data(), m, nc, l_cache,
                       static_cast<const float *>(L.conv), pconv_out.data());
        // Tail of this chunk becomes the history for the next one.
        std::memcpy(hist, seq + m * nc, d_conv * nc * sizeof(float));

        q8_rows_mt(pconv_out.data(), m, n_embd, paq.data());
        mmn(L.t_out, n_embd, n_embd, pattn.data(), n_embd, L.out_proj, paq.data(), m);
    }

    void attention_rows(size_t il, const Layer & L, size_t m, size_t pos0) {
        const size_t qw = n_head * head_dim;
        const size_t kvw = n_head_kv * head_dim;
        q8_rows_mt(pnorm.data(), m, n_embd, paq.data());
        mmn(L.tq, qw, n_embd, pq.data(), qw, L.wq, paq.data(), m);
        mmn(L.tk, kvw, n_embd, pk.data(), kvw, L.wk, paq.data(), m);
        mmn(L.tv, kvw, n_embd, pv.data(), kvw, L.wv, paq.data(), m);

        // Per-head norm happens before RoPE, at row granularity, so it is one
        // call over m*n_head rows.
        rms_norm(m * n_head, head_dim, pq.data(), pq.data(), static_cast<const float *>(L.q_norm), eps);
        rms_norm(m * n_head_kv, head_dim, pk.data(), pk.data(), static_cast<const float *>(L.k_norm), eps);
        for (size_t i = 0; i < m; ++i) {
            const size_t pos = pos0 + i;
            rope(n_head, head_dim, pq.data() + i * qw, pos, rope_base, false);
            rope(n_head_kv, head_dim, pk.data() + i * kvw, pos, rope_base, false);
            std::memcpy(k_cache.data() + (il * n_ctx + pos) * kvw, pk.data() + i * kvw, kvw * sizeof(float));
            std::memcpy(v_cache.data() + (il * n_ctx + pos) * kvw, pv.data() + i * kvw, kvw * sizeof(float));
        }
        attention_prefill_mt(pq.data(), m, k_cache.data() + il * n_ctx * kvw,
                          v_cache.data() + il * n_ctx * kvw, pos0, n_head, n_head_kv,
                          head_dim, 1.0f / std::sqrt(static_cast<float>(head_dim)),
                          pattn.data(), pool);
        q8_rows_mt(pattn.data(), m, qw, paq.data());
        mmn(L.to, n_embd, qw, pattn.data(), n_embd, L.wo, paq.data(), m);
    }

    void dense_ffn_rows(size_t il, const Layer & L, size_t m) {
        q8_rows_mt(pnorm.data(), m, n_embd, paq.data());
        mmn(L.tg, n_ff, n_embd, pffn.data(), n_ff, L.gate, paq.data(), m);
        mmn(L.tu, n_ff, n_embd, pffn2.data(), n_ff, L.up, paq.data(), m);
        silu_mul(pffn.data(), pffn2.data(), m * n_ff);
        q8_rows_mt(pffn.data(), m, n_ff, paq.data());
        mmn(L.td, n_embd, n_ff, pffn_out.data(), n_embd, L.down, paq.data(), m);
    }

    // MoE over m rows. Every (token, expert) pair becomes one row of a gathered
    // matrix, pairs are grouped by expert so each expert's three matrices are
    // touched once per chunk, and the results scatter back with the router
    // weight. This is the shape that makes prefill worth doing: at m=1 the same
    // expert is loaded 4 times per token, here it is loaded once for all the
    // tokens that picked it.
    void moe_ffn_rows(size_t il, const Layer & L, size_t m) {
        const float * bias = static_cast<const float *>(L.exp_probs_b);
        float * probs = mprobs.data();
        gemm_f32_mt(m, n_expert, n_embd, probs, n_expert,
                    static_cast<const float *>(L.gate_inp), pnorm.data(), pool);
        for (size_t i = 0; i < m * n_expert; ++i)
            probs[i] = sigmoid_gate ? 1.0f / (1.0f + std::exp(-probs[i])) : probs[i];

        ExpertPair * pairs = mpairs.data();
        size_t np = 0;
        for (size_t t = 0; t < m; ++t) {
            const float * p = probs + t * n_expert;
            auto * sel = msel.data();
            for (size_t e = 0; e < n_expert; ++e) sel[e] = { p[e] + bias[e], static_cast<uint32_t>(e) };
            std::partial_sort(sel, sel + n_expert_used, sel + n_expert,
                              [](const auto & a, const auto & b) { return a.first > b.first; });
            float wsum = 0.0f;
            for (size_t i = 0; i < n_expert_used; ++i) wsum += p[sel[i].second];
            if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
            for (size_t i = 0; i < n_expert_used; ++i)
                pairs[np++] = {sel[i].second, static_cast<uint32_t>(t), p[sel[i].second] / wsum};
            if (router_log) {
                std::fprintf(stderr, "router serial il=%zu [", il);
                for (size_t i = 0; i < n_expert_used; ++i) std::fprintf(stderr, " %u", sel[i].second);
                std::fprintf(stderr, " ]\n");
            }
        }
        std::sort(pairs, pairs + np,
                  [](const ExpertPair & a, const ExpertPair & b) { return a.expert < b.expert; });

        block_q8_K * aq_g = maq_g.data();
        block_q8_K * aq_d = maq_d.data();
        float * gout = mgout.data();
        float * uout = muout.data();
        float * dout = mdout.data();
        uint32_t * ptok = mptok.data();
        float * pw = mpw.data();
        std::fill(pexpacc.begin(), pexpacc.begin() + m * n_embd, 0.0f);

        const size_t sb_g = n_ff_exp * n_embd / QK_K * ggml_type_size(L.t_ge);
        const size_t sb_u = n_ff_exp * n_embd / QK_K * ggml_type_size(L.t_ue);
        const size_t sb_d = n_embd * n_ff_exp / QK_K * ggml_type_size(L.t_de);
        const uint8_t * bg = static_cast<const uint8_t *>(L.gate_exps);
        const uint8_t * bu = static_cast<const uint8_t *>(L.up_exps);
        const uint8_t * bd = static_cast<const uint8_t *>(L.down_exps);

        for (size_t first = 0; first < np;) {
            size_t last = first;
            while (last < np && pairs[last].expert == pairs[first].expert) ++last;
            const size_t c = last - first;
            const uint32_t e = pairs[first].expert;

            for (size_t i = 0; i < c; ++i) ptok[i] = pairs[first + i].token;
            q8_rows_gather_mt(pnorm.data(), ptok, c, n_embd, aq_g);
            mmn(L.t_ge, n_ff_exp, n_embd, gout, n_ff_exp, bg + e * sb_g, aq_g, c);
            mmn(L.t_ue, n_ff_exp, n_embd, uout, n_ff_exp, bu + e * sb_u, aq_g, c);
            silu_mul_mt(gout, uout, c * n_ff_exp);
            q8_rows_mt(gout, c, n_ff_exp, aq_d);
            mmn(L.t_de, n_embd, n_ff_exp, dout, n_embd, bd + e * sb_d, aq_d, c);
            for (size_t i = 0; i < c; ++i) pw[i] = pairs[first + i].weight;
            scatter_add(dout, ptok, pw, c, n_embd, pexpacc.data());
            first = last;
        }
        // pexpacc is the accumulator, so the copy into pffn_out is the only
        // hand-off the caller needs; it is 4.5 MB per layer at ubatch 544.
        std::memcpy(pffn_out.data(), pexpacc.data(), m * n_embd * sizeof(float));
    }

    const float * prefill(const uint32_t * tokens, size_t n, ModelState & st) {
        if (n == 0) throw std::runtime_error("kata: empty prefill");
        if (st.pos + n > n_ctx) throw std::runtime_error("kata: context full");
        for (size_t done = 0; done < n; done += ubatch) {
            const size_t m = std::min(ubatch, n - done);
            const size_t pos0 = st.pos + done;
            const auto t_start = std::chrono::steady_clock::now();
            double t_emb_s=0,t_norm_s=0,t_mix_s=0,t_res_s=0,t_ffn_s=0,t_attn_s=0,t_conv_s=0;
            auto NS=[&]{return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();};
            double a=NS();
            embed_rows(tokens + done, m); t_emb_s+=NS()-a;
            for (size_t il = 0; il < n_layer; ++il) {
                const Layer & L = layers[il];
                a=NS(); std::memcpy(pprev.data(), px.data(), m * n_embd * sizeof(float));
                rms_norm(m, n_embd, pnorm.data(), px.data(), static_cast<const float *>(L.attn_norm), eps); t_norm_s+=NS()-a;
                a=NS(); if (L.attn) { attention_rows(il, L, m, pos0); t_attn_s+=NS()-a; }
                else { shortconv_rows(il, L, m); t_conv_s+=NS()-a; } t_mix_s+=NS()-a;
                a=NS(); for (size_t i = 0; i < m * n_embd; ++i) px[i] = pprev[i] + pattn[i]; t_res_s+=NS()-a;

                a=NS(); rms_norm(m, n_embd, pnorm.data(), px.data(), static_cast<const float *>(L.ffn_norm), eps); t_norm_s+=NS()-a;
                a=NS(); if (L.moe) moe_ffn_rows(il, L, m);
                else dense_ffn_rows(il, L, m); t_ffn_s+=NS()-a;
                a=NS(); for (size_t i = 0; i < m * n_embd; ++i) px[i] += pffn_out[i]; t_res_s+=NS()-a;
                if (!dump_dir.empty() && done + m == n)
                    dump(("pout" + std::to_string(il)).c_str(), px.data() + (m - 1) * n_embd, n_embd);
            }
            // Only the last row of the last chunk needs the vocabulary projection.
            if (done + m == n) {
                const float * last = px.data() + (m - 1) * n_embd;
                rms_norm(1, n_embd, pnorm.data(), last, static_cast<const float *>(out_norm), eps);
                mm(t_tok_embd, n_vocab, n_embd, logits.data(), tok_embd, pnorm.data());
            }
            if (prof) {
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_start).count();
                std::fprintf(stderr, "[prefill] chunk %zu tok in %.2f ms = %.1f tok/s\n", m, ms, 1000.0 * m / ms);
                std::fprintf(stderr, "[phase] emb %.1f norm %.1f mix %.1f (attn %.1f conv %.1f) res %.1f ffn %.1f\n", t_emb_s,t_norm_s,t_mix_s,t_attn_s,t_conv_s,t_res_s,t_ffn_s);
            }
        }
        st.pos += n;
        return logits.data();
    }

    const float * forward(uint32_t token, ModelState & st) {
        if (st.pos >= n_ctx) throw std::runtime_error("kata: context full");
        const size_t pos = st.pos;
        { Tick tk(&t_emb, prof); embed(token); }
        for (size_t il = 0; il < n_layer; ++il) {
            const Layer & L = layers[il];
            if (pos == dump_pos && il == 0) dump(("emb"), x.data(), n_embd);
            { Tick tk(&t_norm, prof); rms_norm(1, n_embd, norm_out.data(), x.data(), static_cast<const float *>(L.attn_norm), eps); }
            if (pos == dump_pos) dump(("norm" + std::to_string(il)).c_str(), norm_out.data(), n_embd);
            { Tick tk(&t_mix, prof);
              if (L.attn) attention(il, L, norm_out, pos);
              else shortconv(il, L, norm_out, attn_out); }
            if (pos == dump_pos) dump(("mixer" + std::to_string(il)).c_str(), attn_out.data(), n_embd);
#if defined(__x86_64__) || defined(__i386__)
            vec_add_inplace(x.data(), attn_out.data(), n_embd);
#else
            for (size_t j = 0; j < n_embd; ++j) x[j] += attn_out[j];
#endif

            rms_norm(1, n_embd, norm_out.data(), x.data(), static_cast<const float *>(L.ffn_norm), eps);
            if (pos == dump_pos) dump(("ffnn" + std::to_string(il)).c_str(), norm_out.data(), n_embd);
            { Tick tk(&t_ffn, prof);
              if (L.moe) moe_ffn(il, L, norm_out, ffn_down);
              else dense_ffn(il, L, norm_out, ffn_down); }
            if (pos == dump_pos) dump(("ffn" + std::to_string(il)).c_str(), ffn_down.data(), n_embd);
#if defined(__x86_64__) || defined(__i386__)
            vec_add_inplace(x.data(), ffn_down.data(), n_embd);
#else
            for (size_t j = 0; j < n_embd; ++j) x[j] += ffn_down[j];
#endif
            if (pos == dump_pos) dump(("out" + std::to_string(il)).c_str(), x.data(), n_embd);
        }
        rms_norm(1, n_embd, norm_out.data(), x.data(), static_cast<const float *>(out_norm), eps);
        { Tick tk(&t_lm, prof); mm(t_tok_embd, n_vocab, n_embd, logits.data(), tok_embd, norm_out.data()); }
        st.pos = pos + 1;
        if (prof) {
            const double tot = t_emb + t_mix + t_ffn + t_lm + t_norm;
            std::fprintf(stderr, "[prof] mm %7.2f ms in %zu calls (%.1f us/call), quantize %5.2f ms\n", t_mm, n_mm, n_mm ? 1000.0*t_mm/n_mm : 0.0, t_q);
            std::fprintf(stderr, "[prof] total %7.2f ms | mix %7.2f (%4.1f%%) | ffn %7.2f (%4.1f%%) | lm %6.2f (%4.1f%%) | norm %5.2f (%4.1f%%) | emb %5.2f\n",
                         tot, t_mix, 100*t_mix/tot, t_ffn, 100*t_ffn/tot, t_lm, 100*t_lm/tot, t_norm, 100*t_norm/tot, t_emb);
            t_emb = t_mix = t_ffn = t_lm = t_norm = t_mm = t_q = 0; n_mm = 0;
        }
        return logits.data();
    }
};

Model::Model() : p_(new Impl) {}
Model::~Model() = default;

void Model::load(const std::string & path, size_t n_ctx) {
    Impl & m = *p_;
    m.file.open(path);
    m.n_ctx = n_ctx;
    const GgufFile & f = m.file;
    m.n_embd = f.u32("lfm2moe.embedding_length");
    m.n_layer = f.u32("lfm2moe.block_count");
    m.n_head = f.u32("lfm2moe.attention.head_count");
    m.n_ff = f.u32("lfm2moe.feed_forward_length");
    m.n_expert = f.u32("lfm2moe.expert_count");
    m.n_expert_used = f.u32("lfm2moe.expert_used_count");
    m.n_ff_exp = f.u32("lfm2moe.expert_feed_forward_length");
    m.eps = f.f32("lfm2moe.attention.layer_norm_rms_epsilon");
    m.rope_base = f.f32("lfm2moe.rope.freq_base");
    if (f.has_key("lfm2moe.shortconv.l_cache")) m.l_cache = f.u32("lfm2moe.shortconv.l_cache");
    if (f.has_key("lfm2moe.expert_gating_func")) m.sigmoid_gate = f.u32("lfm2moe.expert_gating_func") == 2;

    const GgufTensor * te = f.find("token_embd.weight");
    if (!te) throw std::runtime_error("kata: missing token_embd.weight");
    m.tok_embd = f.data(*te);
    m.t_tok_embd = te->type;
    m.n_vocab = te->dims[1];
    m.out_norm = require(f, "token_embd_norm.weight");
    // k/v projections are the only place the kv-head count shows up as a shape.
    for (size_t il = 0; il < m.n_layer; ++il) {
        if (const GgufTensor * t = f.find("blk." + std::to_string(il) + ".attn_k.weight")) {
            m.head_dim = m.n_embd / m.n_head;
            m.n_head_kv = t->dims[1] / m.head_dim;
            break;
        }
    }
    if (m.n_head_kv == 0) throw std::runtime_error("kata: no attention layer found");

    m.layers.resize(m.n_layer);
    for (size_t il = 0; il < m.n_layer; ++il) {
        Layer & L = m.layers[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        L.attn_norm = require(f, p + "attn_norm.weight");
        L.ffn_norm = require(f, p + "ffn_norm.weight");
        L.attn = f.find(p + "attn_q.weight") != nullptr;
        L.moe = f.find(p + "ffn_gate_inp.weight") != nullptr;
        if (L.attn) {
            L.wq = require(f, p + "attn_q.weight"); L.tq = type_of(f, p + "attn_q.weight");
            L.wk = require(f, p + "attn_k.weight"); L.tk = type_of(f, p + "attn_k.weight");
            L.wv = require(f, p + "attn_v.weight"); L.tv = type_of(f, p + "attn_v.weight");
            L.wo = require(f, p + "attn_output.weight"); L.to = type_of(f, p + "attn_output.weight");
            L.q_norm = require(f, p + "attn_q_norm.weight");
            L.k_norm = require(f, p + "attn_k_norm.weight");
        } else {
            L.conv = require(f, p + "shortconv.conv.weight");
            L.in_proj = require(f, p + "shortconv.in_proj.weight"); L.t_in = type_of(f, p + "shortconv.in_proj.weight");
            L.out_proj = require(f, p + "shortconv.out_proj.weight"); L.t_out = type_of(f, p + "shortconv.out_proj.weight");
        }
        if (L.moe) {
            L.gate_inp = require(f, p + "ffn_gate_inp.weight");
            L.exp_probs_b = require(f, p + "exp_probs_b.bias");
            L.gate_exps = require(f, p + "ffn_gate_exps.weight"); L.t_ge = type_of(f, p + "ffn_gate_exps.weight");
            L.up_exps = require(f, p + "ffn_up_exps.weight"); L.t_ue = type_of(f, p + "ffn_up_exps.weight");
            L.down_exps = require(f, p + "ffn_down_exps.weight"); L.t_de = type_of(f, p + "ffn_down_exps.weight");
        } else {
            L.gate = require(f, p + "ffn_gate.weight"); L.tg = type_of(f, p + "ffn_gate.weight");
            L.up = require(f, p + "ffn_up.weight"); L.tu = type_of(f, p + "ffn_up.weight");
            L.down = require(f, p + "ffn_down.weight"); L.td = type_of(f, p + "ffn_down.weight");
        }
    }
    if (const char * d = std::getenv("KATA_DUMP")) m.dump_dir = d;
    m.alloc();
    n_vocab_ = m.n_vocab;
    n_embd_ = m.n_embd;
    n_layer_ = m.n_layer;
}

void Model::reset() {
    Impl & m = *p_;
    std::fill(m.k_cache.begin(), m.k_cache.end(), 0.0f);
    std::fill(m.v_cache.begin(), m.v_cache.end(), 0.0f);
    std::fill(m.conv_state.begin(), m.conv_state.end(), 0.0f);
}

const float * Model::forward(uint32_t token, ModelState & st) { return p_->forward(token, st); }

const float * Model::prefill(const uint32_t * tokens, size_t n, ModelState & st) {
    return p_->prefill(tokens, n, st);
}

} // namespace kata
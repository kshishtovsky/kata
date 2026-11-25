#include "kata/model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "kata/attn.h"
#include "kata/gguf.h"
#include "kata/gemv.h"
#include "kata/norm.h"
#include "kata/pool.h"
#include "kata/quant.h"
#include "kata/rope.h"
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

void gemv_f32(size_t rows, size_t cols, float * out, const float * w, const float * x) {
    for (size_t r = 0; r < rows; ++r) {
        const float * wr = w + r * cols;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        size_t c = 0;
        for (; c + 4 <= cols; c += 4) {
            s0 += wr[c] * x[c];
            s1 += wr[c + 1] * x[c + 1];
            s2 += wr[c + 2] * x[c + 2];
            s3 += wr[c + 3] * x[c + 3];
        }
        for (; c < cols; ++c) s0 += wr[c] * x[c];
        out[r] = (s0 + s1) + (s2 + s3);
    }
}

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
    std::vector<block_q8_K> aq;
    float * logits_ptr = nullptr;
    ThreadPool pool{ThreadPool::default_threads()};

    // Set KATA_DUMP=<dir> to write intermediates for diffing against a reference.
    std::string dump_dir;
    // Per-phase wall clock, printed only when KATA_PROF is set.
    bool prof = std::getenv("KATA_PROF") != nullptr;
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

    void embed(uint32_t token) {
        if (t_tok_embd != 14) throw std::runtime_error("kata: token_embd must be q6_K");
        const size_t nb = n_embd / QK_K;
        const uint8_t * base = static_cast<const uint8_t *>(tok_embd) + token * nb * sizeof(block_q6_K);
        dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(base), x.data(), n_embd);
    }

    void shortconv(size_t il, const Layer & L, const std::vector<float> & in, std::vector<float> & out) {
        // in_proj emits b, c, x as three contiguous chunks of n_embd.
        std::vector<block_q8_K> a(3 * n_embd / QK_K);
        quantize_row_q8_K(in.data(), a.data(), 3 * n_embd);
        gemv_quant(L.t_in, 3 * n_embd, n_embd, bcx.data(), L.in_proj, a.data(), pool);

        const size_t d_conv = l_cache - 1;
        float * hist = conv_state.data() + il * d_conv * n_embd;
        // conv.weight is ggml [3, n_embd]: ne[0]=3 varies fastest, so channel e
        // owns the three consecutive floats kern[e*3 .. e*3+2].
        const float * kern = static_cast<const float *>(L.conv);
        std::vector<float> conv_out(n_embd);
        const float * b = bcx.data();
        const float * c = bcx.data() + n_embd;
        const float * xx = bcx.data() + 2 * n_embd;
        for (size_t e = 0; e < n_embd; ++e) {
            const float bxe = b[e] * xx[e];
            const float * k3 = kern + e * 3;
            float s = k3[0] * hist[0 * n_embd + e]
                    + k3[1] * hist[1 * n_embd + e]
                    + k3[2] * bxe;
            conv_out[e] = c[e] * s;
        }
        // Shift the history window: newest bx becomes hist[d_conv-1].
        for (size_t e = 0; e < n_embd; ++e) {
            for (size_t j = 0; j + 1 < d_conv; ++j) hist[j * n_embd + e] = hist[(j + 1) * n_embd + e];
            hist[(d_conv - 1) * n_embd + e] = b[e] * xx[e];
        }
        std::vector<block_q8_K> ao(n_embd / QK_K);
        quantize_row_q8_K(conv_out.data(), ao.data(), n_embd);
        gemv_quant(L.t_out, n_embd, n_embd, out.data(), L.out_proj, ao.data(), pool);
    }

    void attention(size_t il, const Layer & L, const std::vector<float> & in, size_t pos) {
        mm(L.tq, n_head * head_dim, n_embd, q.data(), L.wq, in.data());
        mm(L.tk, n_head_kv * head_dim, n_embd, k.data(), L.wk, in.data());
        mm(L.tv, n_head_kv * head_dim, n_embd, v.data(), L.wv, in.data());

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
        std::vector<float> scratch(attention_scratch_floats_grouped(a));
        attention_decode_grouped(a, scratch.data());

        mm(L.to, n_embd, n_head * head_dim, attn_out.data(), L.wo, attn_out.data());
    }

    void dense_ffn(size_t il, const Layer & L, const std::vector<float> & in, std::vector<float> & out) {
        mm(L.tg, n_ff, n_embd, ffn_gate.data(), L.gate, in.data());
        if (!dump_dir.empty()) dump(("dgate" + std::to_string(il)).c_str(), ffn_gate.data(), n_ff);
        mm(L.tu, n_ff, n_embd, ffn_up.data(), L.up, in.data());
        if (!dump_dir.empty()) dump(("dup" + std::to_string(il)).c_str(), ffn_up.data(), n_ff);
        silu_inplace(ffn_gate.data(), n_ff);
        for (size_t i = 0; i < n_ff; ++i) ffn_gate[i] *= ffn_up[i];
        mm(L.td, n_embd, n_ff, out.data(), L.down, ffn_gate.data());
    }

    void moe_ffn(size_t il, const Layer & L, const std::vector<float> & in, std::vector<float> & out) {
        std::vector<float> logits_e(n_expert);
        gemv_f32(n_expert, n_embd, logits_e.data(), static_cast<const float *>(L.gate_inp), in.data());

        std::vector<float> probs(n_expert);
        for (size_t e = 0; e < n_expert; ++e)
            probs[e] = sigmoid_gate ? 1.0f / (1.0f + std::exp(-logits_e[e])) : logits_e[e];

        // Selection uses probs + bias; the weight comes from the unbiased prob.
        const float * bias = static_cast<const float *>(L.exp_probs_b);
        std::vector<std::pair<float, uint32_t>> sel(n_expert);
        for (size_t e = 0; e < n_expert; ++e) sel[e] = { probs[e] + bias[e], static_cast<uint32_t>(e) };
        std::partial_sort(sel.begin(), sel.begin() + n_expert_used, sel.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });

        std::vector<float> w(n_expert_used);
        float wsum = 0.0f;
        for (size_t i = 0; i < n_expert_used; ++i) { w[i] = probs[sel[i].second]; wsum += w[i]; }
        if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
        if (!dump_dir.empty()) {
            dump(("moelogits" + std::to_string(il)).c_str(), logits_e.data(), n_expert);
            dump(("moeprobs" + std::to_string(il)).c_str(), probs.data(), n_expert);
            std::vector<float> biased(n_expert), wsel(n_expert_used), wn(n_expert_used);
            for (size_t e = 0; e < n_expert; ++e) biased[e] = probs[e] + bias[e];
            for (size_t i = 0; i < n_expert_used; ++i) { wsel[i] = w[i]; wn[i] = w[i] / wsum; }
            dump(("moebiased" + std::to_string(il)).c_str(), biased.data(), n_expert);
            dump(("moeweights" + std::to_string(il)).c_str(), wsel.data(), n_expert_used);
            dump(("moewnorm" + std::to_string(il)).c_str(), wn.data(), n_expert_used);
        }

        std::fill(out.begin(), out.end(), 0.0f);
        // One expert's matrix is a slice of the stacked tensor.
        const size_t sb_g = n_ff_exp * n_embd / QK_K * ggml_type_size(L.t_ge);
        const size_t sb_u = n_ff_exp * n_embd / QK_K * ggml_type_size(L.t_ue);
        const size_t sb_d = n_embd * n_ff_exp / QK_K * ggml_type_size(L.t_de);
        for (size_t i = 0; i < n_expert_used; ++i) {
            const uint32_t e = sel[i].second;
            const void * wg = static_cast<const uint8_t *>(L.gate_exps) + e * sb_g;
            const void * wu = static_cast<const uint8_t *>(L.up_exps) + e * sb_u;
            const void * wd = static_cast<const uint8_t *>(L.down_exps) + e * sb_d;
            mm(L.t_ge, n_ff_exp, n_embd, ffn_gate.data(), wg, in.data());
            mm(L.t_ue, n_ff_exp, n_embd, ffn_up.data(), wu, in.data());
            dump(("rg" + std::to_string(i)).c_str(), ffn_gate.data(), n_ff_exp);
            dump(("ru" + std::to_string(i)).c_str(), ffn_up.data(), n_ff_exp);
            silu_inplace(ffn_gate.data(), n_ff_exp);
            for (size_t j = 0; j < n_ff_exp; ++j) ffn_gate[j] *= ffn_up[j];
            mm(L.t_de, n_embd, n_ff_exp, moe_acc.data(), wd, ffn_gate.data());
            if (!dump_dir.empty() && il == 2) {
                dump(("xgate" + std::to_string(i)).c_str(), ffn_gate.data(), n_ff_exp);
                dump(("xdown" + std::to_string(i)).c_str(), moe_acc.data(), n_embd);
            }
            const float wj = w[i] / wsum;
            for (size_t j = 0; j < n_embd; ++j) out[j] += wj * moe_acc[j];
        }
    }

    const float * forward(uint32_t token, ModelState & st) {
        if (st.pos >= n_ctx) throw std::runtime_error("kata: context full");
        const size_t pos = st.pos;
        { Tick tk(&t_emb, prof); embed(token); }
        for (size_t il = 0; il < n_layer; ++il) {
            const Layer & L = layers[il];
            if (pos == 0 && il == 0) dump(("emb"), x.data(), n_embd);
            prev = x;
            { Tick tk(&t_norm, prof); rms_norm(1, n_embd, norm_out.data(), x.data(), static_cast<const float *>(L.attn_norm), eps); }
            if (pos == 0) dump(("norm" + std::to_string(il)).c_str(), norm_out.data(), n_embd);
            { Tick tk(&t_mix, prof);
              if (L.attn) attention(il, L, norm_out, pos);
              else shortconv(il, L, norm_out, attn_out); }
            if (pos == 0) dump(("mixer" + std::to_string(il)).c_str(), attn_out.data(), n_embd);
            for (size_t j = 0; j < n_embd; ++j) x[j] = prev[j] + attn_out[j];

            rms_norm(1, n_embd, norm_out.data(), x.data(), static_cast<const float *>(L.ffn_norm), eps);
            if (pos == 0) dump(("ffnn" + std::to_string(il)).c_str(), norm_out.data(), n_embd);
            { Tick tk(&t_ffn, prof);
              if (L.moe) moe_ffn(il, L, norm_out, ffn_down);
              else dense_ffn(il, L, norm_out, ffn_down); }
            if (pos == 0) dump(("ffn" + std::to_string(il)).c_str(), ffn_down.data(), n_embd);
            for (size_t j = 0; j < n_embd; ++j) x[j] += ffn_down[j];
            if (pos == 0) dump(("out" + std::to_string(il)).c_str(), x.data(), n_embd);
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

} // namespace kata
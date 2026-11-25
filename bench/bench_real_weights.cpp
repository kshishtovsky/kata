// Runs kata's kernels against ggml on a real matrix taken out of a .gguf file.
//
// The shipped models are K-quant (Q4_K/Q6_K/IQ3_XXS), not q4_0, so the tensor is
// dequantised with ggml's own reference and then requantised to q4_0 by kata.
// That way the comparison uses real weight distributions rather than noise.
#include "kata/cpu.h"
#include "kata/gemv.h"
#include "kata/gguf.h"
#include "kata/pool.h"
#include "kata/quant.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

using namespace kata;
using Clock = std::chrono::steady_clock;
static double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

// Mirror of ggml_type_traits from ggml.h:3064.
struct TypeTraits {
    const char * type_name;
    int64_t blck_size;
    int64_t blck_size_interleave;
    size_t type_size;
    bool is_quantized;
    void (*to_float)(const void *, float *, int64_t);
    void (*from_float_ref)(const float *, void *, int64_t);
};
using GetTraitsFn = const TypeTraits *(*)(int);
// ggml_vec_dot_t, ggml-cpu.h:114: the trailing args are strides and nrc.
using VecDotFn = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

template <class Fn>
static Fn sym(void * h, const char * name) {
    auto f = (Fn)dlsym(h, name);
    if (!f) { fprintf(stderr, "dlsym %s: %s\n", name, dlerror()); exit(1); }
    return f;
}

int main(int argc, char ** argv) {
    std::string path = "/home/medovarov/models/gguf/LFM2.5-24B-A2B-Q4_K_M.gguf";
    std::string lib_base = "/home/medovarov/lcpp-base/build/bin/libggml-base.so";
    std::string lib_cpu = "/home/medovarov/lcpp-base/build/bin/libggml-cpu.so";
    std::string want = "blk.0.ffn_gate.weight";
    int iters = 20;
    int copies = 1;   // replicate the matrix so the working set leaves L3
    std::vector<unsigned> thread_list{1, 8, 16, 32};

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") path = next();
        else if (a == "--tensor") want = next();
        else if (a == "--iters") iters = std::max(1, atoi(next().c_str()));
        else if (a == "--copies") copies = std::max(1, atoi(next().c_str()));
        else if (a == "--list") want.clear();
        else if (a == "--threads") { thread_list.clear(); for (auto s : {next()}) thread_list.push_back((unsigned)atoi(s.c_str())); }
        else if (a == "--threads-list") { thread_list.clear(); std::string s = next(); size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); thread_list.push_back((unsigned)atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
    }

    GgufFile f;
    f.open(path);
    printf("model=%s arch=%s file_tensors=%zu\n", path.c_str(), f.architecture().c_str(), f.tensors().size());

    if (want.empty()) {
        std::vector<const GgufTensor *> rows;
        for (const auto & t : f.tensors())
            if (t.dims.size() == 2 && t.dims[0] % 32 == 0) rows.push_back(&t);
        std::sort(rows.begin(), rows.end(), [](auto * a, auto * b) { return a->n_bytes > b->n_bytes; });
        for (size_t i = 0; i < rows.size() && i < 25; ++i) {
            const auto & t = *rows[i];
            printf("  %-46s %-8s [%zu x %zu] %8.2f MiB\n", t.name.c_str(), ggml_type_name(t.type),
                   (size_t)t.dims[0], (size_t)t.dims[1], t.n_bytes / 1048576.0);
        }
        return 0;
    }

    const GgufTensor * gt = f.find(want);
    if (!gt) { fprintf(stderr, "tensor '%s' not in %s\n", want.c_str(), path.c_str()); return 1; }
    if (gt->dims.size() != 2 || gt->dims[0] % 32 != 0) {
        fprintf(stderr, "tensor '%s' has dims [%zu] or cols=%zu not divisible by 32\n",
                want.c_str(), gt->dims.size(), gt->dims.size() ? (size_t)gt->dims[0] : 0);
        return 1;
    }
    const size_t cols = (size_t)gt->dims[0];
    const size_t rows = (size_t)gt->dims[1];
    printf("tensor=%s type=%s dims=[%zu x %zu] %.2f MiB\n", gt->name.c_str(), ggml_type_name(gt->type),
           cols, rows, gt->n_bytes / 1048576.0);

    void * hb = dlopen(lib_base.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!hb) { fprintf(stderr, "dlopen base: %s\n", dlerror()); return 1; }
    void * hc = dlopen(lib_cpu.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!hc) { fprintf(stderr, "dlopen cpu: %s\n", dlerror()); return 1; }
    auto ggml_cpu_init = sym<void (*)()>(hc, "ggml_cpu_init");
    ggml_cpu_init();  // without this the f16 table is all zeros and every dot returns 0
    auto get_traits = sym<GetTraitsFn>(hb, "ggml_get_type_traits");
    auto ggml_dot = sym<VecDotFn>(hc, "ggml_vec_dot_q4_0_q8_0");

    const TypeTraits * tr = get_traits((int)gt->type);
    if (!tr || !tr->to_float) { fprintf(stderr, "no to_float for type %s\n", ggml_type_name(gt->type)); return 1; }
    printf("dequant: ggml type %s blck=%lld size=%zu quantized=%d\n", tr->type_name,
           (long long)tr->blck_size, tr->type_size, (int)tr->is_quantized);

    std::vector<float> wf(rows * cols);
    tr->to_float(f.data(*gt), wf.data(), (int64_t)(rows * cols));

    double wsum = 0.0, wmin = 1e30, wmax = -1e30;
    size_t zeros = 0;
    for (float v : wf) { wsum += (double)v * v; wmin = std::min(wmin, (double)v); wmax = std::max(wmax, (double)v); if (v == 0.0f) ++zeros; }
    printf("weights: rms=%.5f min=%.5f max=%.5f zero_frac=%.3f\n",
           std::sqrt(wsum / (double)wf.size()), wmin, wmax, (double)zeros / (double)wf.size());

    std::vector<block_q4_0> wq(rows * cols / QK);
    quantize_row_q4_0(wf.data(), wq.data(), rows * cols);

    // Our quantiser must produce the same bytes as ggml's, otherwise the speed
    // comparison is between two different value sets.
    auto ggml_q4_ref = sym<void (*)(const float *, void *, int64_t)>(hb, "quantize_row_q4_0_ref");
    std::vector<block_q4_0> wq_ggml(wq.size());
    ggml_q4_ref(wf.data(), wq_ggml.data(), (int64_t)(rows * cols));
    size_t q_diff = 0, d_diff = 0;
    for (size_t i = 0; i < wq.size(); ++i) {
        if (wq[i].d != wq_ggml[i].d) ++d_diff;
        if (std::memcmp(wq[i].qs, wq_ggml[i].qs, sizeof wq[i].qs) != 0) ++q_diff;
    }
    printf("quantiser vs ggml: %zu/%zu blocks differ in scale, %zu/%zu in nibbles\n",
           d_diff, wq.size(), q_diff, wq.size());

    std::vector<float> dq(rows * cols);
    dequantize_row_q4_0(wq.data(), dq.data(), rows * cols);
    double qerr = 0.0, qref = 0.0;
    for (size_t i = 0; i < wf.size(); ++i) { qerr += (double)(dq[i] - wf[i]) * (dq[i] - wf[i]); qref += (double)wf[i] * wf[i]; }
    printf("quant: relative RMS error %.4f%%\n", 100.0 * std::sqrt(qerr / qref));

    // Deterministic activations with the same statistics as a real post-RMSNorm row.
    std::vector<float> ours(rows), theirs(rows), ref(rows);
    if (copies > 1) {
        std::vector<block_q4_0> big;
        big.reserve(wq.size() * copies);
        for (int c = 0; c < copies; ++c) big.insert(big.end(), wq.begin(), wq.end());
        wq.swap(big);
        ours.resize(rows * copies);
        theirs.resize(rows * copies);
        ref.resize(rows * copies);
        printf("replicated %dx: working set %.1f MiB\n", copies,
               (double)wq.size() * sizeof(block_q4_0) / 1048576.0);
    }
    const size_t total_rows = rows * copies;

    std::vector<float> act(cols);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (auto & v : act) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        v = (float)((int32_t)(s >> 32)) / 2147483648.0f * 2.5f;
    }
    std::vector<block_q8_0> aq(cols / QK);
    quantize_row_q8_0(act.data(), aq.data(), cols);

    // Correctness: both kernels against the scalar reference on real weights.
    gemv_q4_0_q8_0_best()(total_rows, cols, ours.data(), wq.data(), aq.data());
    for (size_t r = 0; r < rows; ++r) ggml_dot((int)cols, &theirs[r], 0, &wq[r * cols / QK], 0, aq.data(), 0, 1);

    double max_den = 0.0, max_rel = 0.0, big = 0.0;
    gemv_q4_0_q8_0_ref(rows, cols, ref.data(), wq.data(), aq.data());
    for (size_t r = 0; r < rows; ++r) max_den = std::max(max_den, (double)std::fabs(ref[r]));
    for (size_t r = 0; r < rows; ++r) {
        const double d_ours = std::fabs((double)ours[r] - ref[r]);
        const double d_ggml = std::fabs((double)theirs[r] - ref[r]);
        max_rel = std::max(max_rel, d_ours / max_den);
        big = std::max(big, d_ggml / max_den);
    }
    printf("correctness vs scalar ref (normalised by max|ref|): kata=%.3e ggml=%.3e\n", max_rel, big);

    const double bytes = (double)wq.size() * sizeof(block_q4_0) + (double)aq.size() * sizeof(block_q8_0);
    // ggml exposes no public thread-pool entry point at this level, so its rows
    // are spread over kata's pool by hand. Equal threading, equal work: the only
    // difference left is the inner kernel.
    const size_t nb = cols / QK;
    auto r1_mt = [&](ThreadPool & pool, unsigned nt) {
        const size_t chunk = (total_rows + nt - 1) / nt;
        pool.run([&](unsigned t) {
            const size_t lo = (size_t)t * chunk, hi = std::min(lo + chunk, total_rows);
            if (lo < hi) gemv_q4_0_q8_0_avx512_r1(hi - lo, cols, ours.data() + lo, wq.data() + lo * nb, aq.data());
        });
    };
    auto ggml_gemv_mt = [&](ThreadPool & pool, unsigned nt) {
        if (nt <= 1) {
            for (size_t r = 0; r < total_rows; ++r) ggml_dot((int)cols, &theirs[r], 0, &wq[r * nb], 0, aq.data(), 0, 1);
            return;
        }
        const size_t chunk = (total_rows + nt - 1) / nt;
        pool.run([&](unsigned t) {
            const size_t lo = (size_t)t * chunk, hi = std::min(lo + chunk, total_rows);
            for (size_t r = lo; r < hi; ++r) ggml_dot((int)cols, &theirs[r], 0, &wq[r * nb], 0, aq.data(), 0, 1);
        });
    };

    // Second kata variant: identical math, one weight stream instead of four.
    printf("\n%-6s %10s %10s %10s %10s %8s %10s\n", "thr", "kata_GB/s", "ggml_GB/s", "kata_ms", "ggml_ms",
           "speedup", "r1_GB/s");
    for (unsigned nt : thread_list) {
        ThreadPool gpool(nt);
        std::vector<double> tk, tg, tr;
        // Order alternates: whoever runs second inherits a warmer cache, which is
        // worth several percent on a buffer that half-fits in L3.
        for (int it = 0; it < iters; ++it) {
            const bool kata_first = (it & 1) == 0;
            if (kata_first) {
                const double t0 = now_s();
                if (nt == 1) gemv_q4_0_q8_0_best()(total_rows, cols, ours.data(), wq.data(), aq.data());
                else gemv_q4_0_q8_0_auto(total_rows, cols, ours.data(), wq.data(), aq.data(), nt);
                tk.push_back(now_s() - t0);

                const double tr0 = now_s();
                if (nt == 1) gemv_q4_0_q8_0_avx512_r1(total_rows, cols, ours.data(), wq.data(), aq.data());
                else r1_mt(gpool, nt);
                tr.push_back(now_s() - tr0);

                const double t1 = now_s();
                ggml_gemv_mt(gpool, nt);
                tg.push_back(now_s() - t1);
            } else {
                const double t1 = now_s();
                ggml_gemv_mt(gpool, nt);
                tg.push_back(now_s() - t1);

                const double tr0 = now_s();
                if (nt == 1) gemv_q4_0_q8_0_avx512_r1(total_rows, cols, ours.data(), wq.data(), aq.data());
                else r1_mt(gpool, nt);
                tr.push_back(now_s() - tr0);

                const double t0 = now_s();
                if (nt == 1) gemv_q4_0_q8_0_best()(total_rows, cols, ours.data(), wq.data(), aq.data());
                else gemv_q4_0_q8_0_auto(total_rows, cols, ours.data(), wq.data(), aq.data(), nt);
                tk.push_back(now_s() - t0);
            }
        }
        const double mk = tk[tk.size() / 2], mg = tg[tg.size() / 2], mr = tr[tr.size() / 2];
        // Quartile spread so a 5% difference is not read out of a 30% noise floor.
        auto spread = [](const std::vector<double> & v) {
            return v.empty() ? 0.0 : (v[v.size() * 3 / 4] - v[v.size() / 4]) / v[v.size() / 2] * 100.0;
        };
        printf("%-6u %8.2f(%4.0f%%) %8.2f(%4.0f%%) %8.2f %8.2f %7.2fx %8.2f(%4.0f%%)\n", nt,
               bytes / mk / 1e9, spread(tk), bytes / mg / 1e9, spread(tg), mk * 1e3, mg * 1e3, mg / mk,
               bytes / mr / 1e9, spread(tr));
    }
    return 0;
}

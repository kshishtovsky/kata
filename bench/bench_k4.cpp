// kata q4_K GEMV vs ggml, on real weights taken verbatim from a .gguf.
// Nothing is dequantised or requantised: the bytes compared are the bytes on disk.
#include "kata/gemv.h"
#include "kata/gguf.h"
#include "kata/quant.h"
#include "kata/pool.h"
#include "kata/throttle.h"

#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace kata;

namespace {

using DotFn = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
using InitFn = void (*)();

} // namespace

int main(int argc, char ** argv) {
    std::string model = "/home/medovarov/models/gguf/LFM2.5-24B-A2B-Q4_K_M.gguf";
    std::string tensor = "blk.0.ffn_gate.weight";
    std::string libcpu = "/home/medovarov/lcpp-base/build/bin/libggml-cpu.so";
    size_t target_mb = 768;
    unsigned threads = 1;
    int iters = 5;
    uint32_t tensor_type = 12;  // 12 = Q4_K, 14 = Q6_K
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc)        model = argv[++i];
        else if (a == "--tensor" && i + 1 < argc)  tensor = argv[++i];
        else if (a == "--lib" && i + 1 < argc)     libcpu = argv[++i];
        else if (a == "--mb" && i + 1 < argc)      target_mb = std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--iters" && i + 1 < argc)   iters = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = (unsigned)std::atoi(argv[++i]);
        else if (a == "--q6")                      tensor_type = 14;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }

    void * hcpu = dlopen(libcpu.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!hcpu) { std::fprintf(stderr, "dlopen %s: %s\n", libcpu.c_str(), dlerror()); return 1; }
    auto * init = reinterpret_cast<InitFn>(dlsym(hcpu, "ggml_cpu_init"));
    if (!init) { std::fprintf(stderr, "ggml_cpu_init not found\n"); return 1; }
    init();  // without this ggml's f16 table is all zeros and every dot is 0
    const char * gdot_name = tensor_type == 14 ? "ggml_vec_dot_q6_K_q8_K" : "ggml_vec_dot_q4_K_q8_K";
    auto * ggml_dot = reinterpret_cast<DotFn>(dlsym(hcpu, gdot_name));
    if (!ggml_dot) { std::fprintf(stderr, "%s not found\n", gdot_name); return 1; }

    GgufFile f;
    f.open(model);
    const GgufTensor * t = f.find(tensor);
    if (!t) { std::fprintf(stderr, "tensor %s not found\n", tensor.c_str()); return 1; }
    if (t->type != tensor_type) {
        std::fprintf(stderr, "tensor %s is %s, expected %s\n", tensor.c_str(),
                     ggml_type_name(t->type), ggml_type_name(tensor_type));
        return 1;
    }

    const size_t cols = t->dims[0];
    const size_t rows = t->n_elements / cols;
    const size_t nb = cols / QK_K;
    if (cols % QK_K) { std::fprintf(stderr, "cols %zu not a multiple of %d\n", cols, QK_K); return 1; }

    std::vector<float> xf(cols);
    std::mt19937 g(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : xf) v = nd(g);
    std::vector<block_q8_K> xq(nb);
    quantize_row_q8_K(xf.data(), xq.data(), cols);

    threads = std::min(threads, throttle::max_threads());
    std::printf("threads=%u\n", threads);
    ThreadPool pool(threads);

    // Identical driver for both formats; only the block type and kernel differ.
    auto run_format = [&]<typename BlockT>(auto kata_fn, const char * fmt) {
        size_t reps = 1;
        while (rows * nb * sizeof(BlockT) * reps < target_mb * size_t(1024 * 1024) && reps < 1024) ++reps;
        const size_t total_rows = rows * reps;

        std::vector<BlockT> w(total_rows * nb);
        const BlockT * src = static_cast<const BlockT *>(f.data(*t));
        for (size_t i = 0; i < reps; ++i) std::memcpy(&w[i * rows * nb], src, rows * nb * sizeof(BlockT));

        const double bytes = static_cast<double>(w.size()) * sizeof(BlockT);
        std::vector<float> a(total_rows), b(total_rows);

        kata_fn(total_rows, cols, a.data(), w.data(), xq.data());
        double scale = 0.0, worst = 0.0;
        const size_t probe = std::min<size_t>(rows, 64);
        for (size_t r = 0; r < probe; ++r) {
            ggml_dot(static_cast<int>(cols), &b[r], 0, w.data() + r * nb, 0, xq.data(), 0, 1);
            scale = std::max(scale, std::fabs(static_cast<double>(b[r])));
        }
        for (size_t r = 0; r < probe; ++r) {
            worst = std::max(worst, std::fabs(a[r] - b[r]) / std::max(1e-3, scale));
        }

        std::printf("tensor %s  cols=%zu rows=%zu nb=%zu  reps=%zu  workset=%.0f MB\n",
                    tensor.c_str(), cols, rows, nb, reps, bytes / 1048576.0);
        std::printf("accuracy: max|kata-ggml| / max|ggml| = %.3e\n\n", worst);

        const size_t chunk = (total_rows + threads - 1) / threads;
        auto run = [&](auto call, const char * name, std::vector<double> & out) {
            call();
            for (int i = 0; i < iters; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                call();
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                out.push_back(bytes / secs / 1e9);
                throttle::pause_since(t0);
            }
            std::sort(out.begin(), out.end());
            std::printf("  %-8s median %6.2f GB/s   p10 %6.2f   p90 %6.2f\n",
                        name, out[out.size() / 2], out[out.size() / 10], out[out.size() * 9 / 10]);
        };

        std::vector<double> k, gg;
        run([&] {
            if (threads == 1) { kata_fn(total_rows, cols, a.data(), w.data(), xq.data()); return; }
            pool.run([&](unsigned tid) {
                const size_t r0 = (size_t)tid * chunk;
                const size_t r1 = std::min(total_rows, r0 + chunk);
                if (r0 < r1) kata_fn(r1 - r0, cols, a.data() + r0, w.data() + r0 * nb, xq.data());
            });
        }, "kata", k);
        run([&] {
            auto body = [&](size_t r0, size_t r1) {
                for (size_t r = r0; r < r1; ++r) {
                    ggml_dot(static_cast<int>(cols), &b[r], 0, w.data() + r * nb, 0, xq.data(), 0, 1);
                }
            };
            if (threads == 1) { body(0, total_rows); return; }
            pool.run([&](unsigned tid) {
                const size_t r0 = (size_t)tid * chunk;
                const size_t r1 = std::min(total_rows, r0 + chunk);
                if (r0 < r1) body(r0, r1);
            });
        }, "ggml", gg);

        const double ratio = k[k.size() / 2] / gg[gg.size() / 2];
        std::printf("\nratio kata/ggml = %.3fx  (%s)\n", ratio, ratio >= 1.0 ? "faster" : "slower");
        std::printf("CSV,%s,%u,%zu,%zu,%.2f,%.2f,%.3f\n",
                    fmt, threads, total_rows, cols, k[k.size() / 2], gg[gg.size() / 2], ratio);
    };

    if (tensor_type == 14) run_format.template operator()<block_q6_K>(gemv_q6_K_q8_K_best(), "q6_K");
    else                   run_format.template operator()<block_q4_K>(gemv_q4_K_q8_K_best(), "q4_K");
    return 0;
}

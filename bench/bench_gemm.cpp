// Prefill matmul benchmark: batched GEMM vs the per-row GEMV it replaces.
//
// Both sides read the same bytes and produce the same numbers; the only
// difference is whether the weight block is unpacked once (GEMM) or once per
// activation row (GEMV). Bytes moved = weights + activations, which is what the
// GB/s column means.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "kata/cpu.h"
#include "kata/gemm.h"
#include "kata/gemv.h"
#include "kata/pool.h"
#include "kata/quant.h"
#include "kata/throttle.h"
#include "kata/types.h"

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

struct Case { size_t m, n, k; const char * label; };

int main(int argc, char ** argv) {
    unsigned threads = 8;
    bool q6 = false;
    size_t repeats = 5;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = (unsigned)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--q6")) q6 = true;
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) repeats = (size_t)std::atoi(argv[++i]);
    }
    threads = std::min(threads, kata::throttle::max_threads());

    // Shapes taken from LFM2.5-24B-A2B: q/k/v/o, dense FFN, expert FFN, lm_head.
    std::vector<Case> cases;
    if (q6) {
        cases = {{128, 2048, 2048, "attn_o(2048x2048)"}, {512, 512, 2048, "attn_v(512x2048)"},
                 {512, 2048, 2048, "attn_q(2048x2048)"}, {512, 2048, 1536, "expert_down(2048x1536)"}};
    } else {
        cases = {{128, 2048, 2048, "attn_q(2048x2048)"}, {512, 512, 2048, "attn_k(512x2048)"},
                 {512, 2048, 2048, "attn_o(2048x2048)"}, {512, 2048, 11776, "dense_up(11776x2048)"},
                 {256, 2048, 1536, "expert_gate(2048x1536)"}};
    }

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::printf("kata prefill matmul bench: isa=%s threads=%u duty=%.2f kind=%s\n",
                kata::isa_name(kata::cpu_isa()), threads, kata::throttle::duty(), q6 ? "q6_K" : "q4_K");

    for (const Case & c : cases) {
        const size_t nb = c.k / kata::QK_K;
        std::vector<float> act(c.m * c.k);
        for (float & v : act) v = nd(rng);

        std::vector<kata::block_q8_K> aq(c.m * nb);
        for (size_t i = 0; i < c.m; ++i)
            kata::quantize_row_q8_K(act.data() + i * c.k, aq.data() + i * nb, c.k);

        std::vector<float> out_gemm(c.m * c.n, 0.0f), out_gemv(c.m * c.n, 0.0f);
        double t_gemm = 0.0, t_gemv = 0.0;
        size_t bytes = 0;

        if (q6) {
            std::vector<kata::block_q6_K> w(c.n * nb);
            for (auto & b : w) {
                b.d = kata::f32_to_f16(0.001f + 0.01f * (byte(rng) / 255.0f));
                for (auto & x : b.ql) x = (uint8_t)byte(rng);
                for (auto & x : b.qh) x = (uint8_t)byte(rng);
                for (auto & x : b.scales) x = (int8_t)byte(rng);
            }
            bytes = (size_t)(w.size() * sizeof(kata::block_q6_K)) + aq.size() * sizeof(kata::block_q8_K);
            kata::ThreadPool pool(threads);
            for (size_t r = 0; r < repeats; ++r) {
                double t0 = now_ms();
                kata::gemm_q6_K_q8_K_mt(c.m, c.n, c.k, out_gemm.data(), c.n, w.data(), aq.data(), pool);
                t_gemm += now_ms() - t0;
            }
            std::fill(out_gemv.begin(), out_gemv.end(), 0.0f);
            for (size_t r = 0; r < repeats; ++r) {
                double t0 = now_ms();
                for (size_t i = 0; i < c.m; ++i)
                    kata::gemv_q6_K_q8_K_mt(c.n, c.k, out_gemv.data() + i * c.n, w.data(), aq.data() + i * nb, pool);
                t_gemv += now_ms() - t0;
            }
        } else {
            std::vector<kata::block_q4_K> w(c.n * nb);
            for (auto & b : w) {
                b.d = kata::f32_to_f16(0.001f + 0.01f * (byte(rng) / 255.0f));
                b.dmin = kata::f32_to_f16(0.001f + 0.01f * (byte(rng) / 255.0f));
                for (auto & x : b.scales) x = (uint8_t)byte(rng);
                for (auto & x : b.qs) x = (uint8_t)byte(rng);
            }
            bytes = (size_t)(w.size() * sizeof(kata::block_q4_K)) + aq.size() * sizeof(kata::block_q8_K);
            kata::ThreadPool pool(threads);
            for (size_t r = 0; r < repeats; ++r) {
                double t0 = now_ms();
                kata::gemm_q4_K_q8_K_mt(c.m, c.n, c.k, out_gemm.data(), c.n, w.data(), aq.data(), pool);
                t_gemm += now_ms() - t0;
            }
            std::fill(out_gemv.begin(), out_gemv.end(), 0.0f);
            for (size_t r = 0; r < repeats; ++r) {
                double t0 = now_ms();
                for (size_t i = 0; i < c.m; ++i)
                    kata::gemv_q4_K_q8_K_mt(c.n, c.k, out_gemv.data() + i * c.n, w.data(), aq.data() + i * nb, pool);
                t_gemv += now_ms() - t0;
            }
        }

        double scale = 1e-9;
        double worst = 0.0;
        for (size_t i = 0; i < out_gemm.size(); ++i) scale = std::max(scale, (double)std::fabs(out_gemv[i]));
        for (size_t i = 0; i < out_gemm.size(); ++i)
            worst = std::max(worst, std::fabs((double)out_gemm[i] - out_gemv[i]) / scale);

        t_gemm /= (double)repeats;
        t_gemv /= (double)repeats;
        const double gb = bytes / 1e9;
        std::printf("%-22s m=%-4zu n=%-5zu gemm %8.2f ms (%6.1f GB/s)  gemv %8.2f ms (%6.1f GB/s)  x%.2f  relerr %.1e\n",
                    c.label, c.m, c.n, t_gemm, gb / (t_gemm / 1e3), t_gemv, gb / (t_gemv / 1e3),
                    t_gemv / t_gemm, worst);
        std::printf("CSV,%s,%zu,%zu,%zu,%.3f,%.3f,%.2f,%.1e\n", q6 ? "q6_K" : "q4_K", c.m, c.n, c.k,
                    t_gemm, t_gemv, t_gemv / t_gemm, worst);
    }
    return 0;
}
// Diagnostic: how does the MoE GEMM scale with m? The prefill chunk is 1024 rows,
// but an expert bucket is only m = chunk * top_k / n_expert, so the kernel mostly
// runs at small m. This table decides whether the 21.5%-of-peak figure is a
// structural tile problem (bad at every m) or a small-m problem (bad only low).
#include "kata/gemm.h"
#include "kata/pool.h"
#include "kata/quant.h"
#include "kata/types.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using kata::QK;

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Filled with plausible q4_K bytes: the kernel reads them, so densities matter
// less than the byte pattern, but a zero scale would let GCC fold work away.
void fill_q4k(std::vector<kata::block_q4_K> & w, std::mt19937 & rng) {
    for (auto & b : w) {
        for (auto & x : b.qs) x = static_cast<uint8_t>(rng());
        b.d = kata::f32_to_f16(0.01f);
        b.dmin = kata::f32_to_f16(0.002f);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const unsigned threads = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 16;
    const size_t k = 2048;
    const size_t nb = k / QK;
    const double vnni_peak = 4975.0;   // GMAC/s, measured by tools/peak_bench

    std::mt19937 rng(7);
    std::vector<kata::block_q4_K> w(11776 * nb);
    fill_q4k(w, rng);
    std::vector<float> in(8192 * k), out(8192 * 11776);
    for (auto & x : in) x = static_cast<float>(static_cast<int>(rng() % 200) - 100) / 100.0f;
    std::vector<kata::block_q8_K> a(8192 * nb);
    for (size_t i = 0; i < 8192; ++i)
        kata::quantize_row_q8_K(in.data() + i * k, a.data() + i * nb, k);

    kata::ThreadPool pool(threads);
    std::printf("threads=%u, VNNI peak %.0f GMAC/s, k=%zu\n", threads, vnni_peak, k);
    std::printf("%-14s %6s %8s %9s %8s %7s\n", "shape", "m", "ms", "GMAC/s", "peak%", "MB");
    for (const auto & sh : std::vector<std::pair<const char *, size_t>>{
             {"expert_gate", 1536}, {"attn_o", 2048}, {"dense_up", 11776}}) {
        const size_t n = sh.second;
        for (size_t m : {34, 128, 512, 2048, 8192}) {
            if (m > 8192) continue;
            const int reps = static_cast<int>(std::max<size_t>(1, 40000000 / (m * n)));
            double best = 1e9;
            for (int r = 0; r < 3; ++r) {
                const double t0 = now_ms();
                for (int i = 0; i < reps; ++i)
                    kata::gemm_q4_K_q8_K_mt(m, n, k, out.data(), n, w.data(), a.data(), pool);
                best = std::min(best, (now_ms() - t0) / reps);
            }
            const double mac = static_cast<double>(m) * n * k;
            const double gmac = mac / best / 1e6;
            std::printf("%-14s %6zu %8.2f %9.1f %7.1f %7.1f\n", sh.first, m, best, gmac,
                        100.0 * gmac / vnni_peak, static_cast<double>(n) * k * 0.5625 / (1u << 20));
        }
    }
    return 0;
}

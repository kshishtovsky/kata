// Peaks with COMPILE-TIME unroll. The previous version indexed a[] with a runtime
// loop bound, so GCC kept the accumulators in memory and every "peak" was really
// a load-store benchmark. Same bug as the vt5/vt7 prototypes.
// Freeze the machine's peak in CI: if this number moves between commits, the
// benchmark changed, not the hardware. Every "we are at X% of peak" claim in the
// journal divides by these two constants.
//
// The compile-time unroll below is not stylistic. With a runtime loop bound the
// accumulators live on the stack and the measurement collapses to a load-store
// benchmark (it reported 57 GFLOP/s for every thread count). Do not "simplify" it.
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>
#include <algorithm>

template <int N> struct FloatPeak {
    __attribute__((target("avx512f,avx512bw,avx512vl,fma")))
    static double run(long long iters) {
        __m512 a[N];                                    // compile-time bound
        for (int i = 0; i < N; ++i) a[i] = _mm512_set1_ps((float)i + 1.0f);
        const __m512 b = _mm512_set1_ps(1.0000001f), c = _mm512_set1_ps(1e-7f);
        const auto t0 = std::chrono::steady_clock::now();
        for (long long it = 0; it < iters; ++it) {
            for (int i = 0; i < N; ++i) a[i] = _mm512_fmadd_ps(a[i], b, c);
        }
        const auto t1 = std::chrono::steady_clock::now();
        float s = 0;
        for (int i = 0; i < N; ++i) s += _mm512_reduce_add_ps(a[i]);
        if (s == 12345.0f) std::printf("impossible\n");
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        return static_cast<double>(iters) * N * 16.0 * 2.0 / sec / 1e9;
    }
};

template <int N> struct VnniPeak {
    __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
    static double run(long long iters) {
        __m512i a[N];
        for (int i = 0; i < N; ++i) a[i] = _mm512_setzero_si512();
        const __m512i b = _mm512_set1_epi32(0x01010101), c = _mm512_set1_epi32(1);
        const auto t0 = std::chrono::steady_clock::now();
        for (long long it = 0; it < iters; ++it) {
            for (int i = 0; i < N; ++i) a[i] = _mm512_dpbusd_epi32(a[i], b, c);
        }
        const auto t1 = std::chrono::steady_clock::now();
        long long s = 0;
        for (int i = 0; i < N; ++i) { alignas(64) int32_t v[16]; _mm512_store_si512(v, a[i]); for (int j = 0; j < 16; ++j) s += v[j]; }
        if (s == 12345) std::printf("impossible\n");
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        return static_cast<double>(iters) * N * 16.0 * 4.0 / sec / 1e9;   // 4 MAC per lane
    }
};

template <typename F> double multi(int nt, F f) {
    double best = 0;
    for (int r = 0; r < 3; ++r) {
        std::vector<double> res(nt, 0.0);
        std::vector<std::thread> th;
        for (int t = 0; t < nt; ++t) th.emplace_back([&res, t, &f] { res[t] = f(); });
        for (auto & x : th) x.join();
        double sum = 0; for (double v : res) sum += v;
        best = std::max(best, sum);
    }
    return best;
}

static double g_fp32_peak = 0;
static double g_vnni_peak = 0;

int main(int argc, char ** argv) {
    const int nt = argc > 1 ? std::atoi(argv[1]) : 1;
    const long long iters = 4000000;
    std::printf("threads=%d\n", nt);
    std::printf("  fp32 FMA: ");
    for (int n : {1, 4, 8, 12}) {
        double v = n == 1 ? multi(nt, [] { return FloatPeak<1>::run(40000000); })
                 : n == 4 ? multi(nt, [] { return FloatPeak<4>::run(20000000); })
                 : n == 8 ? multi(nt, [] { return FloatPeak<8>::run(10000000); })
                          : multi(nt, [] { return FloatPeak<12>::run(8000000); });
        std::printf("N=%d %6.1f GFLOP/s  ", n, v);
        if (n == 12) g_fp32_peak = v;
    }
    std::printf("\n  int8 VNNI: ");
    for (int n : {1, 4, 8, 12}) {
        double v = n == 1 ? multi(nt, [] { return VnniPeak<1>::run(40000000); })
                 : n == 4 ? multi(nt, [] { return VnniPeak<4>::run(20000000); })
                 : n == 8 ? multi(nt, [] { return VnniPeak<8>::run(10000000); })
                          : multi(nt, [] { return VnniPeak<12>::run(8000000); });
        std::printf("N=%d %6.1f GMAC/s  ", n, v);
        if (n == 12) g_vnni_peak = v;
    }
    std::printf("\n");
    // Exit code 2 when the measured peaks drift from the frozen values by more
    // than 10%: that is a benchmark bug, not a faster machine.
    static constexpr double kFrozenFp32 = 2642.0, kFrozenVnni = 4975.0;
    const bool drift = std::fabs(g_fp32_peak - kFrozenFp32) > 0.10 * kFrozenFp32 ||
                       std::fabs(g_vnni_peak - kFrozenVnni) > 0.10 * kFrozenVnni;
    if (drift && nt >= 16) {
        std::fprintf(stderr, "peak drift: fp32 %.0f (frozen %.0f), vnni %.0f (frozen %.0f)\n",
                     g_fp32_peak, kFrozenFp32, g_vnni_peak, kFrozenVnni);
        return 2;
    }
    return 0;
}

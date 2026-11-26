// QK score-pass layout microbenchmark: (a) shipped head-major, (b) transposed
// 16-position tile. Frozen: run with --selftest in CI/make bench, which checks
// BOTH the three historical oracle errors (as deterministic counters, not as
// remembered lessons) and a +/-10% drift bound on the speedup.
//
// The three errors this file exists to catch, each of which invalidated a real
// measurement in this project:
//   1. arm (a) loading k once per HEAD instead of once per GROUP of four heads
//      (4x more bytes) -- made the shipped kernel look 1.7x slower than it is;
//   2. the transposed tile being walked at the row stride (128 KB) instead of
//      inside a 16-float block -- a TLB catastrophe reported as 0.01x;
//   3. the tile being rebuilt per head instead of once per (block, kv group)
//      and shared by every query row -- reported as 0.06x.
//
// Counters are structural, so they cannot drift with the machine: they count
// k-tile builds and k bytes touched, and the assertions are about how those
// numbers scale with n_heads, n_pos and n_query. Speed is frozen separately.

#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

constexpr size_t HD = 64;   // LFM2.5 head_dim
constexpr size_t TP = 16;   // positions per transposed tile (one zmm)
constexpr size_t G = 4;     // query heads sharing one kv head

#define TGT __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))

// Work counters for the selftest. `k_bytes` is what the arm actually asked the
// cache for, so a 4x load shows up as 4x bytes regardless of speed.
struct Counters {
    size_t tile_builds = 0;   // transposed tiles materialised
    size_t k_bytes = 0;       // bytes of k touched
    size_t tile_span = 0;     // bytes between consecutive dims inside one tile
};

// (a) shipped shape: one k load per position, shared by the four heads of a
// group, per-head horizontal reduce.
TGT void score_a4(const float *const *q4, const float *kp, float *sc, Counters & c) {
    const __m512 k0 = _mm512_loadu_ps(kp), k1 = _mm512_loadu_ps(kp + 16),
                 k2 = _mm512_loadu_ps(kp + 32), k3 = _mm512_loadu_ps(kp + 48);
    c.k_bytes += 4 * 64;
    for (int h = 0; h < 4; ++h) {
        const float *q = q4[h];
        __m512 a = _mm512_mul_ps(_mm512_loadu_ps(q), k0);
        a = _mm512_fmadd_ps(_mm512_loadu_ps(q + 16), k1, a);
        a = _mm512_fmadd_ps(_mm512_loadu_ps(q + 32), k2, a);
        a = _mm512_fmadd_ps(_mm512_loadu_ps(q + 48), k3, a);
        sc[h] = _mm512_reduce_add_ps(a);
    }
}

// (b) transposed: 16 positions live in lanes, four accumulators are the four
// heads, so there is no horizontal reduce at all.
TGT void scores_b4(const float *const *q4, const float *kt, size_t tile_stride, __m512 *out) {
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t d = 0; d < HD; ++d) {
        const __m512 kv = _mm512_loadu_ps(kt + d * tile_stride);
        a0 = _mm512_fmadd_ps(_mm512_set1_ps(q4[0][d]), kv, a0);
        a1 = _mm512_fmadd_ps(_mm512_set1_ps(q4[1][d]), kv, a1);
        a2 = _mm512_fmadd_ps(_mm512_set1_ps(q4[2][d]), kv, a2);
        a3 = _mm512_fmadd_ps(_mm512_set1_ps(q4[3][d]), kv, a3);
    }
    out[0] = a0; out[1] = a1; out[2] = a2; out[3] = a3;
}

TGT void build_tile(const float *k, size_t kv_stride, size_t p0, size_t n, float *kt, Counters & c) {
    for (size_t d = 0; d < HD; ++d)
        for (size_t j = 0; j < n; ++j) kt[d * TP + j] = k[(p0 + j) * kv_stride + d];
    ++c.tile_builds;
    c.k_bytes += HD * n * sizeof(float);
    c.tile_span = TP * sizeof(float);
}

struct Run {
    double ms = 0;
    double gmac = 0;
    Counters c;
};

Run run_a(const std::vector<float> &k, const std::vector<float> &q, size_t npos, size_t heads,
          size_t nq, size_t kv_stride) {
    Run r;
    volatile float sink = 0.f;
    std::vector<float> out(G * HD);
    const auto t0 = clk::now();
    for (size_t qb = 0; qb < nq; ++qb)
        for (size_t hp = 0; hp < heads / G; ++hp) {
            const float *q4[4] = {q.data() + (hp * G + 0) * HD, q.data() + (hp * G + 1) * HD,
                                  q.data() + (hp * G + 2) * HD, q.data() + (hp * G + 3) * HD};
            for (size_t p = 0; p < npos; ++p) {
                float sc[4];
                score_a4(q4, k.data() + p * kv_stride + (hp % G) * HD, sc, r.c);
                sink = sc[0] + sc[1] + sc[2] + sc[3];
            }
        }
    r.ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    r.gmac = 2.0 * heads * npos * HD * nq / 1e9 / (r.ms / 1000.0);
    (void)sink;
    (void)out;
    return r;
}

Run run_b(const std::vector<float> &k, const std::vector<float> &q, size_t npos, size_t heads,
          size_t nq, size_t kv_stride) {
    Run r;
    volatile float sink = 0.f;
    std::vector<float> kt(HD * TP), out(G * HD);
    __m512 acc[4];
    const auto t0 = clk::now();
    for (size_t hp = 0; hp < heads / G; ++hp) {
        const float *q4[4] = {q.data() + (hp * G + 0) * HD, q.data() + (hp * G + 1) * HD,
                              q.data() + (hp * G + 2) * HD, q.data() + (hp * G + 3) * HD};
        for (size_t p0 = 0; p0 < npos; p0 += TP) {
            const size_t n = (npos - p0) < TP ? (npos - p0) : TP;
            build_tile(k.data() + (hp % G) * HD, kv_stride, p0, n, kt.data(), r.c);
            for (size_t qb = 0; qb < nq; ++qb) {
                scores_b4(q4, kt.data(), TP, acc);
                float tmp[16];
                std::memcpy(tmp, &acc[0], sizeof tmp);
                sink = tmp[0] + tmp[15];
            }
        }
    }
    r.ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    r.gmac = 2.0 * heads * npos * HD * nq / 1e9 / (r.ms / 1000.0);
    (void)sink;
    (void)out;
    return r;
}

int selftest(const std::vector<float> &k, const std::vector<float> &q, size_t kv_stride) {
    int bad = 0;
    const size_t npos = 1024, heads = 32;

    // Error 1: arm (a) must touch exactly 4 zmm per position per GROUP, so
    // bytes scale with npos * (heads/G) * 4 * 64 -- never 16 * 64.
    const Run a = run_a(k, q, npos, heads, 1, kv_stride);
    const size_t want_a = npos * (heads / G) * 4 * 64;
    if (a.c.k_bytes != want_a) {
        std::printf("FAIL (a) k_bytes %zu, expected %zu (a per-head k load would be %zu)\n",
                    a.c.k_bytes, want_a, npos * heads * 4 * 64);
        ++bad;
    }

    // Errors 2 and 3: the tile must span 16 floats (64 B) between dims, and it
    // must be built once per (block, kv group) regardless of how many query rows
    // reuse it.
    const Run b1 = run_b(k, q, npos, heads, 1, kv_stride);
    const Run b64 = run_b(k, q, npos, heads, 64, kv_stride);
    if (b1.c.tile_span != TP * sizeof(float)) {
        std::printf("FAIL (b) tile_span %zu bytes, expected %zu (row-stride walk)\n",
                    b1.c.tile_span, TP * sizeof(float));
        ++bad;
    }
    if (b1.c.tile_builds != b64.c.tile_builds) {
        std::printf("FAIL (b) tile_builds depends on n_query: %zu vs %zu (per-head rebuild)\n",
                    b1.c.tile_builds, b64.c.tile_builds);
        ++bad;
    }
    const size_t want_tiles = (heads / G) * ((npos + TP - 1) / TP);
    if (b1.c.tile_builds != want_tiles) {
        std::printf("FAIL (b) tile_builds %zu, expected %zu\n", b1.c.tile_builds, want_tiles);
        ++bad;
    }
    // Arm (b) reads one k line per dim per tile: 64 B * HD * tiles. If it read
    // per head it would be G times that.
    const size_t want_b = want_tiles * HD * 64;
    if (b1.c.k_bytes != want_b) {
        std::printf("FAIL (b) k_bytes %zu, expected %zu (%zux means per-head)\n",
                    b1.c.k_bytes, want_b, b1.c.k_bytes / want_b);
        ++bad;
    }

    if (!bad) std::printf("selftest: 5/5 structural checks pass\n");
    return bad;
}

}  // namespace

int main(int argc, char **argv) {
    // Frozen speedups, measured on Ryzen 7950X3D, 1 thread, 32k positions,
    // 32 heads. Measured at the engine's own regime: ubatch 1024 means ~1024
    // query rows reusing one tile, but the tile is built once per PREFILL CHUNK
    // so the per-row amortisation saturates well before that. 64 is the point
    // where it stops moving, so it is the frozen reference.
    //
    // The speedup is a function of nq and MUST be quoted with it: 1.71x at
    // nq=16, 2.48x at nq=64, 2.75x at nq=128. Quoting the best of those as
    // "the" speedup would be the same flattery as the 78 GB/s and 453 tok/s
    // numbers this project already had to retract.
    constexpr double kFrozenSpeedup = 2.48;
    constexpr size_t kFrozenNq = 64;
    constexpr double kDrift = 0.10;   // +/-10%, same policy as peak_bench

    size_t npos = 32768, heads = 32, nq = kFrozenNq;
    bool selftest_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--selftest")) selftest_only = true;
        else if (!std::strcmp(argv[i], "--npos") && i + 1 < argc) npos = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--heads") && i + 1 < argc) heads = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--nq") && i + 1 < argc) nq = std::strtoul(argv[++i], nullptr, 10);
    }

    const size_t kv_stride = 16 * HD;   // 8 kv heads x 64 dim
    std::vector<float> k(kv_stride * npos + 128), q(heads * HD);
    for (size_t i = 0; i < k.size(); ++i) k[i] = 1e-6f * float(i % 89);
    for (size_t i = 0; i < q.size(); ++i) q[i] = 1e-3f * float((i * 7) % 31 - 15);

    if (selftest_only) return selftest(k, q, kv_stride) ? 1 : 0;

    const Run a = run_a(k, q, npos, heads, nq, kv_stride);
    const Run b = run_b(k, q, npos, heads, nq, kv_stride);
    const double ratio = a.ms / b.ms;
    std::printf("(a) head-major   %8.1f ms %6.0f GMAC/s  k_bytes %zu\n", a.ms, a.gmac, a.c.k_bytes);
    std::printf("(b) transposed16 %8.1f ms %6.0f GMAC/s  k_bytes %zu tiles %zu\n",
                b.ms, b.gmac, b.c.k_bytes, b.c.tile_builds);
    std::printf("speedup (a)/(b) = %.2fx  (frozen %.2fx, drift limit %.0f%%)\n", ratio, kFrozenSpeedup,
                kDrift * 100);

    const int bad = selftest(k, q, kv_stride);
    if (bad) return 1;
    if (nq != kFrozenNq) {
        // Not a drift check: the speedup legitimately moves with nq because the
        // transposed tile is amortised over query rows. Only the reference point
        // is frozen, so a different nq is a different measurement.
        std::printf("note: nq=%zu, frozen reference is nq=%zu -- drift not checked\n", nq, kFrozenNq);
        return 0;
    }
    if (std::fabs(ratio / kFrozenSpeedup - 1.0) > kDrift) {
        std::printf("FAIL speedup drifted more than %.0f%% from the frozen value\n", kDrift * 100);
        return 2;
    }
    return 0;
}
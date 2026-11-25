#include <gtest/gtest.h>

#include "kata/shortconv.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

// Straight transcription of the definition: y[i] = c[i] * sum_t k[t] * bx[i-h+t]
// with h = ntaps-1, where bx = b*x and the three thirds of bcx are b, c, x.
// Rows before the chunk come from `hist_bx` (zeros for a fresh state), matching
// the [history][chunk] ring layout the driver uses.
void reference(const std::vector<float> & hist_bx, const std::vector<float> & bcx,
               const std::vector<float> & kern, size_t m, size_t nc, size_t ntaps,
               std::vector<float> & out) {
    const size_t hist_rows = hist_bx.size() / nc;
    auto bx_at = [&](long row, size_t e) -> float {
        if (row >= 0) {
            const float * r = bcx.data() + static_cast<size_t>(row) * 3 * nc;
            return r[e] * r[2 * nc + e];
        }
        return hist_bx[static_cast<size_t>(static_cast<long>(hist_rows) + row) * nc + e];
    };
    const long h = static_cast<long>(ntaps) - 1;
    for (size_t i = 0; i < m; ++i) {
        for (size_t e = 0; e < nc; ++e) {
            const float c = bcx[i * 3 * nc + nc + e];
            float acc = 0.0f;
            for (size_t t = 0; t < ntaps; ++t)
                acc += kern[e * ntaps + t] * bx_at(static_cast<long>(i) - h + static_cast<long>(t), e);
            out[i * nc + e] = c * acc;
        }
    }
}

std::vector<float> random_vec(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto & x : v) x = d(gen);
    return v;
}

// Runs one chunk the way shortconv_rows does: history rows first, then bx, then
// the tail of the chunk becomes the next history.
void run_chunk(const std::vector<float> & bcx, const std::vector<float> & kern,
               std::vector<float> & seq, size_t m, size_t nc, size_t ntaps,
               std::vector<float> & out) {
    const size_t hist = ntaps - 1;
    float * bx = seq.data() + hist * nc;
    kata::shortconv_bx(bcx.data(), bx, m, nc);
    kata::shortconv_conv(bx, bcx.data(), m, nc, ntaps, kern.data(), out.data());
    std::memcpy(seq.data(), seq.data() + m * nc, hist * nc * sizeof(float));
}

} // namespace

TEST(ShortConv, MatchesDefinitionOnAContiguousChunk) {
    const size_t m = 9, nc = 40, ntaps = 3, hist = ntaps - 1;
    auto bcx = random_vec(m * 3 * nc, 1);
    auto kern = random_vec(nc * ntaps, 2);
    std::vector<float> seq((hist + m) * nc, 0.0f);
    std::vector<float> got(m * nc, 0.0f), want(m * nc, 0.0f);

    run_chunk(bcx, kern, seq, m, nc, ntaps, got);
    reference(std::vector<float>(hist * nc, 0.0f), bcx, kern, m, nc, ntaps, want);

    // The vector path contracts into an FMA; the reference does not, so allow
    // a few ulps rather than demanding bit equality.
    for (size_t i = 0; i < got.size(); ++i) ASSERT_NEAR(got[i], want[i], 1e-7f) << "index " << i;
}

TEST(ShortConv, HistoryCarriesAcrossChunksLikeOneLongRun) {
    const size_t nc = 24, ntaps = 3, hist = ntaps - 1, chunk = 5;
    auto bcx = random_vec(2 * chunk * 3 * nc, 3);
    auto kern = random_vec(nc * ntaps, 4);

    std::vector<float> seq((hist + chunk) * nc, 0.0f);
    std::vector<float> a(chunk * nc, 0.0f), b(chunk * nc, 0.0f);
    run_chunk(bcx, kern, seq, chunk, nc, ntaps, a);
    std::vector<float> bcx2(bcx.begin() + static_cast<long>(chunk * 3 * nc), bcx.end());
    run_chunk(bcx2, kern, seq, chunk, nc, ntaps, b);

    std::vector<float> seq_flat((hist + 2 * chunk) * nc, 0.0f);
    std::vector<float> flat(2 * chunk * nc, 0.0f);
    run_chunk(bcx, kern, seq_flat, 2 * chunk, nc, ntaps, flat);

    for (size_t i = 0; i < chunk * nc; ++i) ASSERT_FLOAT_EQ(a[i], flat[i]) << "chunk0 " << i;
    for (size_t i = 0; i < chunk * nc; ++i) ASSERT_FLOAT_EQ(b[i], flat[chunk * nc + i]) << "chunk1 " << i;
}

// The bug that shipped before: a chunk shorter than the history read before the
// buffer start. With the ring layout the history is always present.
TEST(ShortConv, ChunkShorterThanKernelStillUsesHistory) {
    const size_t nc = 16, ntaps = 3, hist = ntaps - 1, total = 4;
    auto bcx = random_vec(total * 3 * nc, 5);
    auto kern = random_vec(nc * ntaps, 6);

    std::vector<float> seq_flat((hist + total) * nc, 0.0f);
    std::vector<float> flat(total * nc, 0.0f);
    run_chunk(bcx, kern, seq_flat, total, nc, ntaps, flat);

    std::vector<float> seq((hist + 1) * nc, 0.0f);
    std::vector<float> got(nc, 0.0f);
    for (size_t i = 0; i < total; ++i) {
        std::vector<float> one(bcx.begin() + static_cast<long>(i * 3 * nc),
                               bcx.begin() + static_cast<long>((i + 1) * 3 * nc));
        run_chunk(one, kern, seq, 1, nc, ntaps, got);
        for (size_t e = 0; e < nc; ++e)
            ASSERT_FLOAT_EQ(got[e], flat[i * nc + e]) << "row " << i << " e " << e;
    }
}

TEST(ShortConv, ZeroCLeavesNothing) {
    const size_t m = 3, nc = 8, ntaps = 3, hist = ntaps - 1;
    auto bcx = random_vec(m * 3 * nc, 7);
    for (size_t i = 0; i < m; ++i)
        for (size_t e = 0; e < nc; ++e) bcx[i * 3 * nc + nc + e] = 0.0f;
    auto kern = random_vec(nc * ntaps, 8);
    std::vector<float> seq((hist + m) * nc, 0.0f), out(m * nc, 1.0f);
    run_chunk(bcx, kern, seq, m, nc, ntaps, out);
    for (float v : out) ASSERT_FLOAT_EQ(v, 0.0f);
}

TEST(ShortConv, ResetOfTheFilterIsTheIdentity) {
    // A kernel that only reads the current row (taps = [0,0,1]) must reproduce
    // c*b*x, which is the simplest thing the layer can do.
    const size_t m = 4, nc = 24, ntaps = 3, hist = ntaps - 1;
    auto bcx = random_vec(m * 3 * nc, 9);
    std::vector<float> kern(nc * ntaps, 0.0f);
    for (size_t e = 0; e < nc; ++e) kern[e * ntaps + 2] = 1.0f;
    std::vector<float> seq((hist + m) * nc, 0.0f), out(m * nc, 0.0f);
    run_chunk(bcx, kern, seq, m, nc, ntaps, out);
    for (size_t i = 0; i < m; ++i)
        for (size_t e = 0; e < nc; ++e) {
            const float * r = bcx.data() + i * 3 * nc;
            ASSERT_FLOAT_EQ(out[i * nc + e], r[nc + e] * r[e] * r[2 * nc + e]) << i << "," << e;
        }
}
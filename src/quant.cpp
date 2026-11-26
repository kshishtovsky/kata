#include "kata/quant.h"
#include "kata/attn.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace kata {

void dequantize_row_q6_K(const block_q6_K * x, float * y, size_t n) {
    const size_t nb = n / QK_K;
    for (size_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t * ql = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t * sc = x[i].scales;
        float * out = y + i * QK_K;
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l + 0] = d * (float)sc[is + 0] * (float)q1;
                out[l + 32] = d * (float)sc[is + 2] * (float)q2;
                out[l + 64] = d * (float)sc[is + 4] * (float)q3;
                out[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

float vec_dot_q6_K_q8_K_ref(size_t n, const block_q6_K * x, const block_q8_K * y) {
    const size_t nb = n / QK_K;
    float sum = 0.0f;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t * ql = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t * sc = x[i].scales;
        const int8_t * q8 = y[i].qs;
        const float d = f16_to_f32(x[i].d) * y[i].d;
        float block = 0.0f;
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                block += (float)sc[is + 0] * (float)q1 * (float)q8[l + 0];
                block += (float)sc[is + 2] * (float)q2 * (float)q8[l + 32];
                block += (float)sc[is + 4] * (float)q3 * (float)q8[l + 64];
                block += (float)sc[is + 6] * (float)q4 * (float)q8[l + 96];
            }
            ql += 64; qh += 32; sc += 8; q8 += 128;
        }
        sum += d * block;
    }
    return sum;
}

float f16_to_f32(uint16_t h) {
    // Ported from the classic half->float bit trick: faster and branch-lighter
    // than a table, and exact for subnormals/inf/nan.
    const uint32_t s = (h >> 15) & 0x00000001u;
    const uint32_t e = (h >> 10) & 0x0000001fu;
    const uint32_t m = h & 0x000003ffu;
    uint32_t out;

    if (e == 0) {
        if (m == 0) {
            out = s << 31;
        } else {
            // subnormal: normalize
            // f16 subnormal value is m * 2^-24; after `shift` steps bit 10 is
            // set, so the exponent is 113 - shift in f32 biased form. Using
            // 112 - shift here made every subnormal half its true size, which
            // is what broke q4_K blocks 3/4/6 of a real model.
            uint32_t mm = m;
            int shift = 0;
            while ((mm & 0x00000400u) == 0) { mm <<= 1; ++shift; }
            mm &= 0x000003ffu;
            out = (s << 31) | ((uint32_t)(113 - shift) << 23) | (mm << 13);
        }
    } else if (e == 0x1f) {
        out = (s << 31) | 0x7f800000u | (m << 13);
    } else {
        out = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
    }

    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

uint16_t f32_to_f16(float f) {
    // Round-to-nearest-even: half-up rounding here produced different scales than
    // ggml on 1.6% of blocks. Kept scalar because this runs on the load path only.
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x007fffffu;

    if (exp == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));  // inf / nan

    int32_t e = (int32_t)exp - 127 + 15;
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u);                            // overflow
    if (e <= 0) {                                                                // subnormal
        if (e < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        const uint32_t sh = (uint32_t)(14 - e);
        uint32_t r = mant >> sh;
        const uint32_t rem = mant & ((1u << sh) - 1);
        const uint32_t halfway = 1u << (sh - 1);
        if (rem > halfway || (rem == halfway && (r & 1))) ++r;
        return (uint16_t)(sign | r);
    }
    uint32_t r = mant >> 13;
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) {
        if (++r == 0x400u) { r = 0; if (++e >= 0x1f) return (uint16_t)(sign | 0x7c00u); }
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | r);
}

void quantize_row_q8_0(const float * x, block_q8_0 * y, size_t n) {
    const size_t nb = n / QK;
    for (size_t i = 0; i < nb; ++i) {
        const float * xi = x + i * QK;
        float amax = 0.0f;
        for (int j = 0; j < QK; ++j) {
            const float a = std::fabs(xi[j]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = f32_to_f16(d);
        for (int j = 0; j < QK; ++j) {
            const float v = std::nearbyint(xi[j] * id);
            int q = (int)v;
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            y[i].qs[j] = (int8_t)q;
        }
    }
}

void quantize_row_q4_0(const float * x, block_q4_0 * y, size_t n) {
    const size_t nb = n / QK;
    for (size_t i = 0; i < nb; ++i) {
        const float * xi = x + i * QK;
        // Same rule as ggml: keep the signed extreme with the largest magnitude.
        float amax = 0.0f, maxv = 0.0f;
        for (int j = 0; j < QK; ++j) {
            const float v = xi[j];
            if (std::fabs(v) > amax) { amax = std::fabs(v); maxv = v; }
        }
        (void)amax;
        // ggml maps the extreme value to -8 exactly, so the scale is -maxv/8.
        const float d = maxv / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = f32_to_f16(d);
        for (int j = 0; j < QK / 2; ++j) {
            const int lo = std::min(15, (int)(xi[j] * id + 8.5f));
            const int hi = std::min(15, (int)(xi[j + QK / 2] * id + 8.5f));
            y[i].qs[j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

void dequantize_row_q4_0(const block_q4_0 * x, float * y, size_t n) {
    const size_t nb = n / QK;
    for (size_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(x[i].d);
        for (int j = 0; j < QK / 2; ++j) {
            const int lo = (x[i].qs[j] & 0x0f) - 8;
            const int hi = (x[i].qs[j] >> 4) - 8;
            y[i * QK + j] = d * (float)lo;
            y[i * QK + j + QK / 2] = d * (float)hi;
        }
    }
}

float vec_dot_q4_0_q8_0_ref(size_t n, const block_q4_0 * x, const block_q8_0 * y) {
    float sum = 0.0f;
    for (size_t b = 0; b < n / QK; ++b) {
        const float d = f16_to_f32(x[b].d) * f16_to_f32(y[b].d);
        int isum = 0;
        for (int j = 0; j < QK / 2; ++j) {
            const int lo = (x[b].qs[j] & 0x0f) - 8;
            const int hi = (x[b].qs[j] >> 4) - 8;
            isum += lo * (int)y[b].qs[j] + hi * (int)y[b].qs[j + QK / 2];
        }
        sum += d * (float)isum;
    }
    return sum;
}

// ---------------------------------------------------------------- K-quants

void q4_K_scale_min(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        // 6 bits do not fit in 8, so the top 2 bits of each scale live in the
        // high bits of the earlier bytes.
        *d = (uint8_t)((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

void quantize_row_q8_K(const float * x, block_q8_K * y, size_t n) {
    const size_t nb = n / QK_K;
    for (size_t i = 0; i < nb; ++i) {
        const float * xi = x + i * QK_K;
        float maxv = 0.0f, amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            const float ax = std::fabs(xi[j]);
            if (ax > amax) { amax = ax; maxv = xi[j]; }
        }
        if (amax == 0.0f) {
            y[i].d = 0.0f;
            std::memset(y[i].qs, 0, QK_K);
            std::memset(y[i].bsums, 0, sizeof y[i].bsums);
            continue;
        }
        // ggml uses -127/max rather than -128 so the AVX path stays in int8.
        const float iscale = -127.0f / maxv;
        for (int j = 0; j < QK_K; ++j) {
            y[i].qs[j] = (int8_t)std::min(127, (int)std::nearbyint(iscale * xi[j]));
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1.0f / iscale;
    }
}

void dequantize_row_q4_K(const block_q4_K * x, float * y, size_t n) {
    const size_t nb = n / QK_K;
    for (size_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(x[i].d);
        const float dmin = f16_to_f32(x[i].dmin);
        const uint8_t * q = x[i].qs;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            q4_K_scale_min(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * (float)sc, m1 = dmin * (float)m;
            q4_K_scale_min(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * (float)sc, m2 = dmin * (float)m;
            for (int l = 0; l < 32; ++l) y[i * QK_K + j + l] = d1 * (float)(q[l] & 0x0f) - m1;
            for (int l = 0; l < 32; ++l) y[i * QK_K + j + 32 + l] = d2 * (float)(q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

float vec_dot_q4_K_q8_K_ref(size_t n, const block_q4_K * x, const block_q8_K * y) {
    float sum = 0.0f;
    for (size_t b = 0; b < n / QK_K; ++b) {
        const float d = f16_to_f32(x[b].d) * y[b].d;
        const float dmin = f16_to_f32(x[b].dmin) * y[b].d;
        int is = 0;
        const uint8_t * q = x[b].qs;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            q4_K_scale_min(is + 0, x[b].scales, &sc, &m);
            int acc1 = 0, acc2 = 0, sy1 = 0, sy2 = 0;
            for (int l = 0; l < 32; ++l) {
                acc1 += (q[l] & 0x0f) * (int)y[b].qs[j + l];
                acc2 += (q[l] >> 4) * (int)y[b].qs[j + 32 + l];
                sy1 += (int)y[b].qs[j + l];
                sy2 += (int)y[b].qs[j + 32 + l];
            }
            sum += d * (float)sc * (float)acc1 - dmin * (float)m * (float)sy1;
            q4_K_scale_min(is + 1, x[b].scales, &sc, &m);
            sum += d * (float)sc * (float)acc2 - dmin * (float)m * (float)sy2;
            q += 32;
            is += 2;
        }
    }
    return sum;
}

void quantize_row_i8(const float * x, int8_t * q, uint16_t * scales, size_t n) {
    // Symmetric, one scale per 32 values: the v pass multiplies by a probability,
    // so a per-block scale costs one broadcast per 32 elements and nothing else.
    for (size_t b = 0; b < n / VQ_BLK; ++b) {
        const float * xb = x + b * VQ_BLK;
        float amax = 0.0f;
        for (size_t i = 0; i < VQ_BLK; ++i) amax = std::max(amax, std::fabs(xb[i]));
        const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float id = 1.0f / d;
        scales[b] = f32_to_f16(d);
        for (size_t i = 0; i < VQ_BLK; ++i) {
            const float v = xb[i] * id;
            q[b * VQ_BLK + i] = static_cast<int8_t>(v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : std::lrint(v)));
        }
    }
}

} // namespace kata

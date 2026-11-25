#include "kata/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace kata {

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
            uint32_t mm = m;
            int shift = 0;
            while ((mm & 0x00000400u) == 0) { mm <<= 1; ++shift; }
            mm &= 0x000003ffu;
            out = (s << 31) | ((127 - 15 - shift) << 23) | (mm << 13);
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

} // namespace kata

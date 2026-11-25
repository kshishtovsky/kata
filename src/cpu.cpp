#include <cpuid.h>
#include <cstdint>
#include "kata/cpu.h"

namespace kata {

#if defined(__x86_64__) || defined(__i386__)
// __builtin_cpu_supports needs literal strings, so probe the CPUID leaves by
// hand instead of wrapping it in a helper.
static bool cpuid_leaf(uint32_t leaf, uint32_t sub, uint32_t regs[4]) {
    __cpuid_count(leaf, sub, regs[0], regs[1], regs[2], regs[3]);
    return regs[0] || regs[1] || regs[2] || regs[3];
}

static Isa detect_isa() {
    uint32_t r[4];
    if (!cpuid_leaf(0, 0, r)) return Isa::Scalar;
    const uint32_t max_leaf = r[0];
    if (max_leaf < 7) return Isa::Scalar;

    cpuid_leaf(1, 0, r);
    const bool f16c = (r[2] >> 29) & 1u;   // ECX bit 29
    const bool fma  = (r[2] >> 12) & 1u;   // ECX bit 12
    const bool avx  = (r[2] >> 28) & 1u;   // ECX bit 28

    cpuid_leaf(7, 0, r);
    const bool avx2        = (r[1] >> 5) & 1u;   // EBX bit 5
    const bool avx512f     = (r[1] >> 16) & 1u;  // EBX bit 16
    const bool avx512bw    = (r[1] >> 30) & 1u;  // EBX bit 30
    const bool avx512vl    = (r[1] >> 31) & 1u;  // EBX bit 31
    const bool avx512vnni  = (r[2] >> 11) & 1u;  // ECX bit 11

    if (avx && f16c && fma && avx512f && avx512bw && avx512vl && avx512vnni) return Isa::Avx512Vnni;
    if (avx && f16c && fma && avx2) return Isa::Avx2;
    return Isa::Scalar;
}
#endif

Isa cpu_isa() {
#if defined(__x86_64__) || defined(__i386__)
    static const Isa cached = detect_isa();
    return cached;
#else
    return Isa::Scalar;
#endif
}

const char * isa_name(Isa isa) {
    switch (isa) {
        case Isa::Avx512Vnni: return "avx512-vnni";
        case Isa::Avx2:       return "avx2";
        default:              return "scalar";
    }
}

} // namespace kata

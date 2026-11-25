#pragma once

namespace kata {

enum class Isa { Scalar, Avx2, Avx512Vnni };

// Detected once from the running CPU, never from compile flags alone.
Isa cpu_isa();
const char * isa_name(Isa isa);

} // namespace kata

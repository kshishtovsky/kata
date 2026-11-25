#pragma once
#include <cstddef>

namespace kata {

// y[i,e] = c[i,e] * sum_t kern[e*ntaps+t] * bx[i-ntaps+1+t, e], where bx = b*x
// and b, c, x are the three contiguous thirds of the in_proj output.
// `bx` must sit at its final offset: the ntaps-1 preceding rows (the history of
// the previous chunk) live right before it in the same buffer. `bcx` is the
// in_proj output the b and x thirds are read from.
void shortconv_bx(const float * bcx, float * bx, size_t m, size_t nc);

void shortconv_conv(const float * bx, const float * bcx, size_t m, size_t nc,
                    size_t ntaps, const float * kern, float * out);

} // namespace kata

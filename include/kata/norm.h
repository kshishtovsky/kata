#pragma once

#include <cstddef>

namespace kata {

// y = w * x / sqrt(mean(x^2) + eps), along each contiguous row.
void rms_norm_ref(size_t rows, size_t cols, float * out, const float * x, const float * w, float eps);
// nullptr weight is allowed and means "no scaling".
void rms_norm_avx2(size_t rows, size_t cols, float * out, const float * x, const float * w, float eps);
void rms_norm(size_t rows, size_t cols, float * out, const float * x, const float * w, float eps);

} // namespace kata
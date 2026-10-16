#include "../common/dsl.h"

// Materialize the eager graph's rounding boundaries. The regular fused
// attention kernels remain the default for all existing architectures.
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(AttentionEagerScores)(
    const device T* queries, const device T* keys, device T* scores,
    const constant uint& length, const constant uint& heads,
    const constant uint& head_dim, const constant uint& window, const constant float& scale,
    const uint k AXIS(length, 16), const uint q AXIS(length, 4), const uint h AXIS(heads, 1)
) {
  const bool allowed = window == 0 || (q > k ? q - k : k - q) <= window / 2;
  float dot = 0.0f;
  if (allowed) {
    for (uint d = 0; d < head_dim; ++d) {
      dot += float(queries[(h * length + q) * head_dim + d]) * float(keys[(k * heads + h) * head_dim + d]);
    }
  }
  scores[(h * length + q) * length + k] = allowed ? T(float(T(dot)) * scale) : T(-INFINITY);
}

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(AttentionEagerValues)(
    const device T* probabilities, const device T* values, device T* output,
    const constant uint& length, const constant uint& heads, const constant uint& head_dim,
    const uint d AXIS(head_dim, 32), const uint q AXIS(length, 4), const uint h AXIS(heads, 1)
) {
  float sum = 0.0f;
  for (uint k = 0; k < length; ++k) {
    sum += float(probabilities[(h * length + q) * length + k]) * float(values[(k * heads + h) * head_dim + d]);
  }
  output[(q * heads + h) * head_dim + d] = T(sum);
}

// Trust-weight composition (A601.trust-weight-composition).
//
// The hybrid blend weight is not the clipped correlation alone: it composes
// four bounded, measured factors. Calibration: under a calibrated
// heteroscedastic reference the expected surprise (err^2 * exp(-log_unc))
// is 1, so the distance of the measured mean from 1 is miscalibration.
// Held-out: the fraction of probe tuples whose surprise exceeds the
// over-bound threshold (probe tuples are never trained on). Diversity: the
// ensemble variance relative to a half-saturation constant, so vacuous
// agreement (all predictors identical) drives the predictor's weight down.
// Correlation: the existing clipped Pearson r. The product is bounded to
// [0, 1]; any zero factor vetoes.

#ifndef COEVO_PREDICTOR_TRUST_WEIGHT_CUH
#define COEVO_PREDICTOR_TRUST_WEIGHT_CUH

#include "../config/constants.cuh"

namespace slime::predictor {

__host__ __device__ inline float clamp01(float x) {
    if (x < 0.f) return 0.f;
    if (x > 1.f) return 1.f;
    return x;
}

__host__ __device__ inline float calibration_factor(float mean_surprise) {
    float d = fabsf(mean_surprise - 1.f);
    return clamp01(1.f - d / TRUST_CAL_TOL);
}

__host__ __device__ inline float held_factor(float over_bound_fraction) {
    return clamp01(1.f - over_bound_fraction / TRUST_HELD_TOL);
}

__host__ __device__ inline float diversity_factor(float variance) {
    return variance / (variance + TRUST_DIVERSITY_VAR0);
}

__host__ __device__ inline float trust_weight(float r, float cal, float held,
                                              float div) {
    return clamp01(clamp01(r) * clamp01(cal) * clamp01(held)
                   * clamp01(div));
}

}  // namespace slime::predictor

#endif  // COEVO_PREDICTOR_TRUST_WEIGHT_CUH

// Sheet A-501: CAME scalar step (shared by the device kernel and host tests)
//
// The production CAME update per weight, per cuda_engineering.md section 4.3:
//   g  = mean_grad
//   m  = beta1*m  + (1-beta1)*g
//   v  = beta2*v  + (1-beta2)*g^2
//   u  = m / (sqrt(v) + eps)
//   instability = (u - prev_u)^2
//   c  = beta3*c  + (1-beta3)*instability
//   confidence  = 1 / (1 + c)
//   w -= lr * confidence * u + weight_decay * w
//   prev_u = u
//
// Kept as one host/device inline so the GPU kernel and the host unit test can
// never drift apart.

#ifndef COEVO_OPTIMIZER_CAME_MATH_CUH
#define COEVO_OPTIMIZER_CAME_MATH_CUH

namespace slime::optimizer {

// CAME hyperparameters (pinned by spec, cuda_engineering.md section 8).
struct CameHyperparams {
    float lr;
    float beta1;
    float beta2;
    float beta3;
    float epsilon;
    float weight_decay;
};

constexpr CameHyperparams CAME_DEFAULTS = {
    1e-3f,   // lr
    0.9f,    // beta1
    0.999f,  // beta2
    0.999f,  // beta3
    1e-8f,   // epsilon
    0.01f    // weight_decay
};

struct CameScalarState {
    float m;       // 1st moment
    float v;       // 2nd moment
    float c;       // instability accumulator
    float prev_u;  // previous normalized update direction
};

struct CameScalarParams {
    float lr;
    float beta1;
    float beta2;
    float beta3;
    float epsilon;
    float weight_decay;
};

// One scalar CAME step. Writes the new weight into *w, updates the state,
// and returns the applied update magnitude |lr * confidence * u| plus the
// decay contribution is folded into *w. Returns confidence for telemetry.
__host__ __device__ inline float came_step_scalar(
    CameScalarState* s,
    float* w,
    float g,
    const CameScalarParams& p)
{
    float mi = p.beta1 * s->m + (1.f - p.beta1) * g;
    s->m = mi;

    float vi = p.beta2 * s->v + (1.f - p.beta2) * g * g;
    s->v = vi;

    float u = mi / (sqrtf(vi) + p.epsilon);

    float diff = u - s->prev_u;
    float instability = diff * diff;
    float ci = p.beta3 * s->c + (1.f - p.beta3) * instability;
    s->c = ci;

    float confidence = 1.f / (1.f + ci);
    float update = p.lr * confidence * u;
    *w -= update + p.weight_decay * (*w);

    s->prev_u = u;
    return confidence;
}

}  // namespace slime::optimizer

#endif  // COEVO_OPTIMIZER_CAME_MATH_CUH

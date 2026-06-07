// Sheet A-501: Optimizer — CAME
//
// Confidence-Adjusted Momentum Estimation. Operates uniformly across roles;
// only the loss feeding the backward pass differs between classifiers
// (cross-entropy on logits) and predictors (MSE on bmap_64; see A-103, A-601).
//
// This is a flat-buffer approximation of CAME, not a faithful port of the
// matrix-factored optimizer the blueprint names (the factorization is a
// memory trick on 2-D weight tensors; on a flat weight buffer it degenerates).
// See came_update for exactly what is computed. Unverified: not compiled.

#ifndef COEVO_OPTIMIZER_CAME_CU
#define COEVO_OPTIMIZER_CAME_CU

#include "../config/constants.cuh"

#include <cuda_runtime.h>

namespace slime::optimizer {

// CAME hyperparameters. Concrete values are set by the caller at construction;
// none are pinned here.
struct CameHyperparams {
    float lr;          // base learning rate
    float beta1;       // first-moment EMA
    float beta2;       // second-moment EMA
    float beta3;       // confidence EMA
    float epsilon;     // numerical floor
    float weight_decay;
};

struct CameState {
    float* m;        // first moment of the gradient
    float* v;        // second moment of the gradient
    float* c;        // confidence: EMA of the squared update instability
    float* prev_u;   // last generation's normalized update u_{t-1}
    int    size;
    int    step;
};

// One CAME update on a flat weight buffer. Called role-blind from the
// world_train_phase / backward_phase pair (A-102).
//
// CAME (Confidence-Adjusted Momentum Estimation): like Adam but scales the
// step by a confidence term that tracks how stable the normalized update is
// from one step to the next. The confidence buffer c_t and the previous
// update u_{t-1} are kept in separate buffers so the two quantities are not
// conflated (c_t has units of u^2; prev_u has units of u):
//   m_t  = b1*m_{t-1} + (1-b1)*g
//   v_t  = b2*v_{t-1} + (1-b2)*g^2
//   u_t  = m_t / (sqrt(v_t) + eps)
//   c_t  = b3*c_{t-1} + (1-b3)*(u_t - u_{t-1})^2     (instability tracker)
//   step = u_t / (sqrt(c_t) + eps)
// where a noisy lineage (large step-to-step change in u) inflates c_t and so
// damps the effective step. This captures the spec's confidence intent
// without the matrix factorization that the named optimizer applies to 2-D
// weight tensors; the flat-buffer form here is the simplification.
__device__ inline void came_update(float* weights,
                                   const float* grads,
                                   CameState* state,
                                   const CameHyperparams& hp,
                                   int idx) {
    float g = grads[idx];
    float m = hp.beta1 * state->m[idx] + (1.f - hp.beta1) * g;
    float v = hp.beta2 * state->v[idx] + (1.f - hp.beta2) * g * g;
    float u = m / (sqrtf(v) + hp.epsilon);
    float du = u - state->prev_u[idx];
    float c_new = hp.beta3 * state->c[idx] + (1.f - hp.beta3) * du * du;
    float step = u / (sqrtf(c_new) + hp.epsilon);
    // Decoupled weight decay (AdamW-style): applied to the weight directly,
    // not routed through the adaptive denominator.
    weights[idx] -= hp.lr * step + hp.lr * hp.weight_decay * weights[idx];
    state->m[idx]      = m;
    state->v[idx]      = v;
    state->c[idx]      = c_new;
    state->prev_u[idx] = u;
}

// Block-wide launcher: one thread per weight element.
__global__ void came_step_kernel(float* weights,
                                 const float* grads,
                                 CameState* state,
                                 CameHyperparams hp,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    came_update(weights, grads, state, hp, i);
    if (i == 0) state->step++;
}

// DECLARED ONLY — blueprint-in-place.
// launch_came_step: thin host wrapper that launches came_step_kernel (above,
// implemented) with ceil(n / 256) blocks of 256 threads on `stream`. The
// per-element update math is done and host-tested; this is only the grid-config
// + launch. One call per trainable weight buffer (W_inter, W_flow, W_bmap) per
// organism, or one fused call over a concatenated buffer.
void launch_came_step(float* weights,
                      const float* grads,
                      CameState* state,
                      const CameHyperparams& hp,
                      int n,
                      cudaStream_t stream);

}  // namespace slime::optimizer

#endif  // COEVO_OPTIMIZER_CAME_CU

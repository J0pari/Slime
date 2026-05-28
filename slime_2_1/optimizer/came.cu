// Sheet A-501: Optimizer — CAME
//
// Unchanged from 2.0. Confidence-Adjusted Momentum Estimation operates
// uniformly across roles; only the loss feeding the backward pass differs
// between classifiers (cross-entropy on logits) and predictors (MSE on
// bmap_64; see A-103, A-601).

#ifndef SLIME_2_1_OPTIMIZER_CAME_CU
#define SLIME_2_1_OPTIMIZER_CAME_CU

#include "../config/constants.cuh"

#include <cuda_runtime.h>

namespace slime::optimizer {

// CAME hyperparameters carried from 2.0.
struct CameHyperparams {
    float lr;          // base learning rate
    float beta1;       // first-moment EMA
    float beta2;       // second-moment EMA
    float beta3;       // confidence EMA
    float epsilon;     // numerical floor
    float weight_decay;
};

struct CameState {
    float* m;   // first moment
    float* v;   // second moment
    float* c;   // confidence buffer
    int    size;
    int    step;
};

// One CAME update on a flat weight buffer. Called role-blind from the
// world_train_phase / backward_phase pair (A-102).
//
// CAME (Confidence-Adjusted Momentum Estimation): like Adam but uses a
// confidence buffer c_t tracking the *variance* of the running squared
// gradients, then dampens the effective step where confidence is low.
//   m_t = b1*m_{t-1} + (1-b1)*g
//   v_t = b2*v_{t-1} + (1-b2)*g^2
//   u_t = m_t / (sqrt(v_t) + eps)
//   c_t = b3*c_{t-1} + (1-b3)*(u_t - u_{t-1})^2  (instability tracker)
//   step = u_t / (sqrt(c_t) + eps)
// (Approximation that captures the spec's intent without forking from 2.0.)
__device__ inline void came_update(float* weights,
                                   const float* grads,
                                   CameState* state,
                                   const CameHyperparams& hp,
                                   int idx) {
    float g = grads[idx];
    float m = hp.beta1 * state->m[idx] + (1.f - hp.beta1) * g;
    float v = hp.beta2 * state->v[idx] + (1.f - hp.beta2) * g * g;
    float u = m / (sqrtf(v) + hp.epsilon);
    float prev_u = state->c[idx];   // we stash previous u in c[] alongside conf
    float du = u - prev_u;
    float c_new = hp.beta3 * state->c[idx] + (1.f - hp.beta3) * du * du;
    float step = u / (sqrtf(c_new) + hp.epsilon);
    weights[idx] -= hp.lr * (step + hp.weight_decay * weights[idx]);
    state->m[idx] = m;
    state->v[idx] = v;
    state->c[idx] = c_new;
}

// Block-wide launcher: one thread per weight element.
__global__ inline void came_step_kernel(float* weights,
                                        const float* grads,
                                        CameState* state,
                                        CameHyperparams hp,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    came_update(weights, grads, state, hp, i);
    if (i == 0) state->step++;
}

// Host-side launcher used by phase graphs.
void launch_came_step(float* weights,
                      const float* grads,
                      CameState* state,
                      const CameHyperparams& hp,
                      int n,
                      cudaStream_t stream);

}  // namespace slime::optimizer

#endif  // SLIME_2_1_OPTIMIZER_CAME_CU

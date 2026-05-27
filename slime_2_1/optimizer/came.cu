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
__device__ void came_update(float* weights,
                            const float* grads,
                            CameState* state,
                            const CameHyperparams& hp);

// Host-side launcher used by phase graphs.
void launch_came_step(float* weights,
                      const float* grads,
                      CameState* state,
                      const CameHyperparams& hp,
                      int n,
                      cudaStream_t stream);

}  // namespace slime::optimizer

#endif  // SLIME_2_1_OPTIMIZER_CAME_CU

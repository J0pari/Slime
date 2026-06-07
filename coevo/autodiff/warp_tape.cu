// Sheet A-103: Autodiff — Checkpointed Warp Tape (Trajectory-Aware)
//
// A checkpointed reverse-mode tape over the CA forward pass. A predictor
// organism's loss is MSE between bmap_64 of its forward pass and a target
// bmap_64 (ground-truth label drawn from the Intent Registry). The Warp tape
// records operations producing bmap_64 via the W_bmap projection, so no custom
// adjoints beyond the standard matmul/activation set are required.
//
// Classifier loss is cross-entropy on classification logits. Both losses flow
// through the same checkpointed backward.
//
// Only the two loss routers below have bodies; the tape recording and the
// backward sweep (launch_backward) are declared-only — see their comments.

#ifndef COEVO_AUTODIFF_WARP_TAPE_CU
#define COEVO_AUTODIFF_WARP_TAPE_CU

#include "../config/constants.cuh"

#include <cuda_runtime.h>

namespace slime::autodiff {

enum class OpType : uint8_t {
    MatMul,
    Activation,
    BmapProject,
    Loss,
    Checkpoint,
};

struct TapeOp {
    OpType   type;
    uint32_t a;
    uint32_t b;
    uint32_t out;
    uint16_t shape[3];
};

struct WarpTape {
    TapeOp*  ops;       // device buffer
    int*     head;      // atomic counter
    int      capacity;
};

// Loss routers. Both write into the gradient buffers feeding W_bmap, W_flow,
// and W_inter through the standard backward sweep (perception is a fixed
// stencil, so there is no W_perc gradient).
//
// Classifier loss: numerically-stable softmax cross-entropy.
//   p_i        = exp(z_i - max_z) / sum_j exp(z_j - max_z)
//   loss       = -log p_target
//   dloss/dz_i = p_i - 1_{i == target}
//
// n_classes is set by the dataset wrapper (A-701); a small upper bound is
// enforced so the temporary p[] buffer stays on the stack/register.
constexpr int MAX_CLASSES = 64;

__host__ __device__ inline void classifier_loss(const float* logits,
                                                int target_class,
                                                int n_classes,
                                                float* dlogits,
                                                float* loss_out) {
    if (n_classes <= 0 || n_classes > MAX_CLASSES) {
        if (loss_out) *loss_out = 0.f;
        return;
    }
    float max_z = logits[0];
    for (int i = 1; i < n_classes; ++i) {
        if (logits[i] > max_z) max_z = logits[i];
    }
    float p[MAX_CLASSES];
    float Z = 0.f;
    for (int i = 0; i < n_classes; ++i) {
        p[i] = expf(logits[i] - max_z);
        Z += p[i];
    }
    float invZ = (Z > 1e-30f) ? (1.0f / Z) : 0.f;
    for (int i = 0; i < n_classes; ++i) p[i] *= invZ;
    float logp_t = logf(p[target_class] + 1e-30f);
    if (loss_out) *loss_out = -logp_t;
    for (int i = 0; i < n_classes; ++i) {
        dlogits[i] = p[i] - ((i == target_class) ? 1.0f : 0.0f);
    }
}

// Predictor MSE loss against the target bmap_64. dloss/d_pred = 2*(p - t)/N.
__host__ __device__ inline void predictor_mse_loss(const float* bmap_pred,
                                                   const float* bmap_target,
                                                   float* dbmap_pred,
                                                   float* loss_out) {
    float acc = 0.f;
    const float scale = 2.0f / static_cast<float>(BMAP_DIM);
    for (int i = 0; i < BMAP_DIM; ++i) {
        float diff = bmap_pred[i] - bmap_target[i];
        acc += diff * diff;
        dbmap_pred[i] = scale * diff;
    }
    if (loss_out) *loss_out = acc / static_cast<float>(BMAP_DIM);
}

// DECLARED ONLY — blueprint-in-place.
// The tape itself is also not yet recorded: the forward pass (A-201) does not
// push TapeOp entries, so there is nothing to walk backward over. Both halves
// are open:
//   Recording: during forward, each matmul / activation / bmap-projection
//   appends a TapeOp (atomicAdd on tape->head, bounds-checked against capacity;
//   on overflow, fall back to gradient checkpointing — recompute from the last
//   Checkpoint op). The 64-step CA rollout is the main source of tape length,
//   which is exactly why checkpointing is in the design.
//   launch_backward: seed the output gradient (dlogits for classifiers,
//   dbmap_pred for predictors — both implemented above), then walk the tape in
//   reverse, applying the adjoint of each op into the FP32 gradient buffers for
//   W_bmap, W_flow, W_inter. Matmul adjoint: dA = dC.B^T, dB = A^T.dC.
//   Activation adjoint: elementwise GELU'/identity. Re-run forward from the
//   nearest checkpoint when the tape segment was not retained.
void launch_backward(WarpTape* tape,
                     float* seed_grad,
                     cudaStream_t stream);

}  // namespace slime::autodiff

#endif  // COEVO_AUTODIFF_WARP_TAPE_CU

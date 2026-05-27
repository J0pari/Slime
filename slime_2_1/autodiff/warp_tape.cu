// Sheet A-103: Autodiff — Checkpointed Warp Tape (Trajectory-Aware)
//
// Unchanged from 2.0 in essentials. The only modification: a predictor
// organism's loss is MSE between bmap_64 of its forward pass and a target
// bmap_64 (ground truth label drawn from the Intent Registry). The Warp tape
// already records operations producing bmap_64 via the W_bmap projection;
// no new custom adjoints are required.
//
// Classifier loss continues to be cross-entropy on classification logits.
// Both losses flow through the same checkpointed backward.

#ifndef SLIME_2_1_AUTODIFF_WARP_TAPE_CU
#define SLIME_2_1_AUTODIFF_WARP_TAPE_CU

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
// W_inter, W_perc through the standard backward sweep.
__device__ void classifier_loss(const float* logits,
                                int target_class,
                                float* dlogits,
                                float* loss_out);

__device__ void predictor_mse_loss(const float* bmap_pred,
                                   const float* bmap_target,
                                   float* dbmap_pred,
                                   float* loss_out);

// Checkpoint-aware backward sweep. Identical control flow for both roles;
// only the seed gradient (dlogits vs dbmap_pred) and the upstream loss
// computation differ.
void launch_backward(WarpTape* tape,
                     float* seed_grad,
                     cudaStream_t stream);

}  // namespace slime::autodiff

#endif  // SLIME_2_1_AUTODIFF_WARP_TAPE_CU

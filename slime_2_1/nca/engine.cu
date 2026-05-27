// Sheet A-201: NCA Engine, Role-Switched Input, Behavioral Trajectory
//
// 16-channel 64x64 grid; 64 CA steps. The role tag in the genome selects how
// the initial grid is populated; W_perc, W_inter, W_flow, W_bmap, and the
// reaction-diffusion machinery are role-blind.
//
// BTRAJ samples (bmap_16, bmap_32, bmap_48, bmap_64) are written to the
// Intent Registry. bmap_64 retains its 2.0 roles (archive descriptor, audit
// input, placeholder regressor input).

#ifndef SLIME_2_1_NCA_ENGINE_CU
#define SLIME_2_1_NCA_ENGINE_CU

#include "../config/constants.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace slime::nca {

// One organism's CA state in managed memory. Channels are the innermost layout
// to match WMMA tile alignment along the channel axis.
struct OrganismState {
    __half grid[GRID_SIZE * GRID_SIZE * CA_CHANNELS];  // current step
    __half scratch[GRID_SIZE * GRID_SIZE * CA_CHANNELS]; // double buffer
    float  bmap_traj[BTRAJ_SAMPLES * BMAP_DIM];        // BTRAJ output
    Role   role;
};

// Inputs to a forward pass. Predictor inputs read from the Intent Registry
// (target organism's bmap_32); classifier inputs read from the task batch.
struct ForwardInputs {
    Role        role;
    const float* task_embedding;   // [TASK_EMBED_DIM]
    const __half* image_rgb;       // [GRID_SIZE * GRID_SIZE * 3] or null
    const float* target_bmap_32;   // [BMAP_DIM] for predictors, else null
};

// ---- Role-switched grid initialization -----------------------------------
// A-201: classifier seeds channels 11-13 (image), 6-10 (task), zero elsewhere.
// Predictor seeds channels 14-15 in a centered 4x4 region with bmap_32 (16
// cells * 2 channels = 32 slots), 6-10 with task embedding, zero elsewhere.
__device__ void seed_classifier_grid(__half* grid,
                                     const __half* image_rgb,
                                     const float* task_embedding);

__device__ void seed_predictor_grid(__half* grid,
                                    const float* target_bmap_32,
                                    const float* task_embedding);

// Single CA step. role-blind: same W_perc / W_inter / W_flow / W_bmap path
// as 2.0. Mass conservation via reintegration (see reaction_diffusion.cu).
__device__ void ca_step(__half* state_curr,
                        __half* state_next,
                        const float* W_perc,
                        const float* W_inter,
                        const float* W_flow);

// 64-step forward, sampling bmap at BTRAJ_STEPS into bmap_traj.
__global__ void forward_kernel(OrganismState* organisms,
                               const ForwardInputs* inputs,
                               const float* W_bmap,
                               int n_organisms);

// Global average pool + W_bmap projection. Produces bmap_t at the requested
// step. Called inside forward_kernel after each BTRAJ step.
__device__ void project_bmap(const __half* state,
                             const float* W_bmap,
                             float* bmap_out_32);

// ---- Public host launchers -----------------------------------------------
void launch_forward(OrganismState* organisms,
                    const ForwardInputs* inputs,
                    const float* W_bmap,
                    int n_organisms,
                    cudaStream_t stream);

// Copies bmap_64 (last BTRAJ slot) into the archive descriptor buffer for the
// caller. bmap_16, bmap_32 stay in the Intent Registry for predictor consumers
// (A-401, A-601).
void extract_descriptor(const OrganismState* organisms,
                        float* descriptors_out,
                        int n_organisms,
                        cudaStream_t stream);

}  // namespace slime::nca

#endif  // SLIME_2_1_NCA_ENGINE_CU

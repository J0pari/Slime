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

// ---- Grid index helper ---------------------------------------------------
__host__ __device__ inline int grid_idx(int y, int x, int c) {
    return (y * GRID_SIZE + x) * CA_CHANNELS + c;
}

// ---- Role-switched grid initialization -----------------------------------
// A-201: classifier seeds channels 11-13 (image), 6-10 (task), zero elsewhere.
// Predictor seeds channels 14-15 in a centered 4x4 region with bmap_32
// (16 cells * 2 channels = 32 slots), 6-10 with task embedding, zero elsewhere.
__device__ inline void seed_classifier_grid(__half* grid,
                                            const __half* image_rgb,
                                            const float* task_embedding) {
    // image_rgb layout: [GRID_SIZE * GRID_SIZE * 3] interleaved RGB.
    for (int y = threadIdx.y; y < GRID_SIZE; y += blockDim.y) {
        for (int x = threadIdx.x; x < GRID_SIZE; x += blockDim.x) {
            // Chemicals + auxiliary zero.
            for (int c = CH_CHEM_FIRST; c <= CH_CHEM_LAST; ++c)
                grid[grid_idx(y, x, c)] = __float2half(0.f);
            for (int c = CH_AUX_FIRST; c <= CH_AUX_LAST; ++c)
                grid[grid_idx(y, x, c)] = __float2half(0.f);
            // Task embedding broadcast spatially. TASK_EMBED_DIM = 16,
            // channels 6..10 carry the first 5 dims; remaining task dims fold
            // into a small dot-product mixing applied during ca_step.
            #pragma unroll
            for (int c = CH_TASK_FIRST; c <= CH_TASK_LAST; ++c) {
                int t_idx = c - CH_TASK_FIRST;
                grid[grid_idx(y, x, c)] = __float2half(task_embedding[t_idx]);
            }
            // Image into channels 11..13.
            int pix = (y * GRID_SIZE + x) * 3;
            grid[grid_idx(y, x, CH_IMG_FIRST + 0)] = image_rgb[pix + 0];
            grid[grid_idx(y, x, CH_IMG_FIRST + 1)] = image_rgb[pix + 1];
            grid[grid_idx(y, x, CH_IMG_FIRST + 2)] = image_rgb[pix + 2];
        }
    }
}

__device__ inline void seed_predictor_grid(__half* grid,
                                           const float* target_bmap_32,
                                           const float* task_embedding) {
    // Zero everything first.
    for (int y = threadIdx.y; y < GRID_SIZE; y += blockDim.y) {
        for (int x = threadIdx.x; x < GRID_SIZE; x += blockDim.x) {
            for (int c = 0; c < CA_CHANNELS; ++c) {
                grid[grid_idx(y, x, c)] = __float2half(0.f);
            }
            // Task embedding broadcast on chans 6..10 (same as classifier).
            #pragma unroll
            for (int c = CH_TASK_FIRST; c <= CH_TASK_LAST; ++c) {
                int t_idx = c - CH_TASK_FIRST;
                grid[grid_idx(y, x, c)] = __float2half(task_embedding[t_idx]);
            }
        }
    }
    // Centered 4x4 region carries bmap_32 across the two aux channels:
    // (16 cells * 2 channels = 32 slots).
    __syncthreads();
    constexpr int CENTER = GRID_SIZE / 2;
    constexpr int LO = CENTER - 2;       // 30
    constexpr int HI = CENTER + 2;       // 34 (exclusive)
    if (threadIdx.y < 4 && threadIdx.x < 4) {
        int local = threadIdx.y * 4 + threadIdx.x;   // 0..15
        int y = LO + threadIdx.y;
        int x = LO + threadIdx.x;
        grid[grid_idx(y, x, CH_AUX_FIRST + 0)] = __float2half(target_bmap_32[local * 2 + 0]);
        grid[grid_idx(y, x, CH_AUX_FIRST + 1)] = __float2half(target_bmap_32[local * 2 + 1]);
    }
    (void)HI;
}

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
//
// Spatial average over GRID_SIZE*GRID_SIZE cells produces a 16-d summary s_t
// (one value per channel); W_bmap is [CA_CHANNELS x BMAP_DIM].
__device__ inline void project_bmap(const __half* state,
                                    const float* W_bmap,
                                    float* bmap_out_32) {
    __shared__ float summary[CA_CHANNELS];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    // Reset summary.
    for (int c = tid; c < CA_CHANNELS; c += nthreads) summary[c] = 0.f;
    __syncthreads();
    // Spatial sum into per-channel accumulators.
    constexpr float scale = 1.0f / static_cast<float>(GRID_SIZE * GRID_SIZE);
    for (int idx = tid; idx < GRID_SIZE * GRID_SIZE; idx += nthreads) {
        int y = idx / GRID_SIZE;
        int x = idx % GRID_SIZE;
        for (int c = 0; c < CA_CHANNELS; ++c) {
            float v = __half2float(state[grid_idx(y, x, c)]) * scale;
            atomicAdd(&summary[c], v);
        }
    }
    __syncthreads();
    // Project: bmap_out_32[d] = sum_c W_bmap[c, d] * summary[c].
    for (int d = tid; d < BMAP_DIM; d += nthreads) {
        float acc = 0.f;
        for (int c = 0; c < CA_CHANNELS; ++c) {
            acc += W_bmap[c * BMAP_DIM + d] * summary[c];
        }
        bmap_out_32[d] = acc;
    }
    __syncthreads();
}

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

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

// The 2-bit role tag has four codes but only two defined roles; 10/11 are
// reserved (A-301). Until additional roles exist, canonicalize a reserved
// code to its low bit so role-switched logic stays total: 10 -> Classifier,
// 11 -> Predictor. read_role in the codec returns the raw 2-bit value; the
// substrate uses this canonical view to pick an input pathway.
__host__ __device__ inline Role canonical_role(Role raw) {
    return (static_cast<uint8_t>(raw) & 0x1u) ? Role::Predictor
                                              : Role::Classifier;
}

// Forward declaration: forward_kernel calls project_bmap, which is defined
// after it (the definition needs no forward refs of its own).
__device__ inline void project_bmap(const __half* state,
                                    const float* W_bmap,
                                    float* bmap_out_32);

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

// ---- Weight shapes (role-blind, A-201) -----------------------------------
//
// Perception: identity + Sobel_x + Sobel_y over the 16-channel state, giving
// a 48-d perception vector per cell. W_inter mixes the perception vector
// into a 32-d hidden representation (GELU). W_flow projects the hidden to a
// 16-d delta. The new state is state + delta (clamped to FP16 range). RD
// machinery for the chemical channels runs in parallel via rd_step.
constexpr int PERC_DIM    = CA_CHANNELS * 3;   // 48
constexpr int HIDDEN_DIM  = 32;

// 3x3 stencils used by perception (identity, Sobel_x, Sobel_y).
__device__ inline void sample_neighborhood(const __half* state,
                                           int y, int x,
                                           float* perc_out) {  // [PERC_DIM]
    // Toroidal wrap on the grid edge.
    auto at = [&](int dy, int dx, int c) -> float {
        int yy = (y + dy + GRID_SIZE) % GRID_SIZE;
        int xx = (x + dx + GRID_SIZE) % GRID_SIZE;
        return __half2float(state[grid_idx(yy, xx, c)]);
    };
    for (int c = 0; c < CA_CHANNELS; ++c) {
        // Identity.
        perc_out[c] = at(0, 0, c);
        // Sobel_x.
        float sx = -at(-1, -1, c) - 2.f * at(0, -1, c) - at(1, -1, c)
                 + at(-1,  1, c) + 2.f * at(0,  1, c) + at(1,  1, c);
        perc_out[CA_CHANNELS + c] = sx * 0.125f;
        // Sobel_y.
        float sy = -at(-1, -1, c) - 2.f * at(-1, 0, c) - at(-1, 1, c)
                 + at( 1, -1, c) + 2.f * at( 1, 0, c) + at( 1, 1, c);
        perc_out[2 * CA_CHANNELS + c] = sy * 0.125f;
    }
}

__device__ inline float gelu_approx(float x) {
    // Hendrycks-Gimpel approximation.
    const float k = 0.7978845608f;          // sqrt(2/pi)
    return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// Single CA step. role-blind: same W_perc / W_inter / W_flow / W_bmap path
// as 2.0. Mass conservation hooks live in rd_step (A-202).
//   W_perc  : [PERC_DIM x CA_CHANNELS]  (currently unused - perception via
//             fixed Sobel stencils; kept in the API to mirror 2.0 weights
//             and to leave room for learned perception kernels)
//   W_inter : [PERC_DIM x HIDDEN_DIM]
//   W_flow  : [HIDDEN_DIM x CA_CHANNELS]
//
// One thread = one cell. Block layout is (16, 16); each block covers a 16x16
// tile of the 64x64 grid via four iterations.
__device__ inline void ca_step(const __half* state_curr,
                               __half* state_next,
                               const float* /*W_perc*/,
                               const float* W_inter,
                               const float* W_flow) {
    for (int by = 0; by < GRID_SIZE; by += blockDim.y) {
        for (int bx = 0; bx < GRID_SIZE; bx += blockDim.x) {
            int y = by + threadIdx.y;
            int x = bx + threadIdx.x;
            if (y >= GRID_SIZE || x >= GRID_SIZE) continue;

            float perc[PERC_DIM];
            sample_neighborhood(state_curr, y, x, perc);

            float hidden[HIDDEN_DIM];
            #pragma unroll
            for (int h = 0; h < HIDDEN_DIM; ++h) {
                float acc = 0.f;
                #pragma unroll
                for (int p = 0; p < PERC_DIM; ++p) {
                    acc += W_inter[p * HIDDEN_DIM + h] * perc[p];
                }
                hidden[h] = gelu_approx(acc);
            }

            #pragma unroll
            for (int c = 0; c < CA_CHANNELS; ++c) {
                float acc = 0.f;
                #pragma unroll
                for (int h = 0; h < HIDDEN_DIM; ++h) {
                    acc += W_flow[h * CA_CHANNELS + c] * hidden[h];
                }
                float prev = __half2float(state_curr[grid_idx(y, x, c)]);
                float next = prev + acc;
                // FP16 clamp; mirrors 2.0 numerical hygiene.
                if (next > 65504.f)  next = 65504.f;
                if (next < -65504.f) next = -65504.f;
                state_next[grid_idx(y, x, c)] = __float2half(next);
            }
        }
    }
    __syncthreads();
}

// 64-step forward, sampling bmap at BTRAJ_STEPS into bmap_traj. One block per
// organism. The kernel itself iterates the substrate; outer host code maps
// blocks to organisms.
__global__ void forward_kernel(OrganismState* organisms,
                               const ForwardInputs* inputs,
                               const float* W_inter,    // [PERC_DIM x HIDDEN_DIM]
                               const float* W_flow,     // [HIDDEN_DIM x CA_CHANNELS]
                               const float* W_bmap,     // [CA_CHANNELS x BMAP_DIM]
                               int n_organisms) {
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    OrganismState* o     = &organisms[org];
    const ForwardInputs& in = inputs[org];

    // Role-switched seeding (A-201). Reserved role codes canonicalize to a
    // defined role so the pathway choice is total.
    if (canonical_role(in.role) == Role::Classifier) {
        seed_classifier_grid(o->grid, in.image_rgb, in.task_embedding);
    } else {
        seed_predictor_grid(o->grid, in.target_bmap_32, in.task_embedding);
    }
    __syncthreads();

    // Copy the BTRAJ sample schedule into registers so the inner loop never
    // indexes the namespace-scope constexpr array with a runtime subscript
    // (which is not guaranteed to be addressable from device code).
    int steps[BTRAJ_SAMPLES];
    #pragma unroll
    for (int i = 0; i < BTRAJ_SAMPLES; ++i) steps[i] = BTRAJ_STEPS[i];

    __half* curr = o->grid;
    __half* next = o->scratch;
    int sample_idx = 0;

    for (int step = 1; step <= CA_STEPS; ++step) {
        ca_step(curr, next, /*W_perc*/ nullptr, W_inter, W_flow);
        __half* tmp = curr; curr = next; next = tmp;

        if (sample_idx < BTRAJ_SAMPLES && step == steps[sample_idx]) {
            project_bmap(curr,
                         W_bmap,
                         &o->bmap_traj[sample_idx * BMAP_DIM]);
            sample_idx++;
        }
    }
    // Leave the final state in o->grid so downstream phases (descriptor
    // extraction, audit) can read directly without tracking the double buffer.
    if (curr != o->grid) {
        for (int idx = threadIdx.y * blockDim.x + threadIdx.x;
             idx < GRID_SIZE * GRID_SIZE * CA_CHANNELS;
             idx += blockDim.x * blockDim.y) {
            o->grid[idx] = curr[idx];
        }
    }
}

// Global average pool + W_bmap projection. Produces bmap_t at the requested
// step. Called inside forward_kernel after each BTRAJ step (forward-declared
// above).
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
// Mirrors forward_kernel's parameter list: the substrate weights (W_inter,
// W_flow, W_bmap) are role-blind and shared across the launched organisms.
void launch_forward(OrganismState* organisms,
                    const ForwardInputs* inputs,
                    const float* W_inter,
                    const float* W_flow,
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

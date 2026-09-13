// Sheet A-201: NCA Engine, Role-Switched Input, Behavioral Trajectory
//
// 16-channel 64x64 grid; 64 CA steps. The role tag in the genome selects how
// the initial grid is populated. Perception is a learned bank of depthwise 3x3
// filters (W_perc); the learned weights W_perc, W_inter, W_flow, W_bmap and the
// reaction-diffusion machinery are role-blind.
//
// BTRAJ samples (bmap_16, bmap_32, bmap_48, bmap_64) are written to the
// Intent Registry. bmap_64 is the archive descriptor, the audit input, and the
// reference regressor input.

#ifndef COEVO_NCA_ENGINE_CU
#define COEVO_NCA_ENGINE_CU

#include "../config/constants.cuh"
#include "../genome/codec.cu"
#include "reaction_diffusion.cu"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "context_adjoint.cuh"
#include "rd_adjoint.cuh"

namespace slime::nca {

// One organism's CA state in managed memory. Channels are the innermost layout
// to match WMMA tile alignment along the channel axis.
struct OrganismState {
    __half grid[GRID_SIZE * GRID_SIZE * CA_CHANNELS];  // current step
    __half scratch[GRID_SIZE * GRID_SIZE * CA_CHANNELS]; // double buffer
    float  bmap_traj[BTRAJ_SAMPLES * BMAP_DIM];        // BTRAJ output
    float  sample_summary[BTRAJ_SAMPLES * CA_CHANNELS]; // pre-broadcast means (A-203)
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

// canonical_role is defined once in config/constants.cuh (A-301) and shared
// between host and device code.

// Forward declaration: forward_kernel calls project_bmap, which is defined
// after it (the definition needs no forward refs of its own).
__device__ inline void project_bmap(__half* state,
                                    const float* W_bmap,
                                    float* bmap_out_32,
                                    float* summary_out);

// Fixed task-embedding projection (A-201): channel `ch` (0..4) of the task
// field carries the DCT row `ch` of the 16-d embedding. Every embedding
// dimension contributes to every task channel, so no advertised task
// dimension is inert in the forward.
__host__ __device__ inline float task_channel_value(const float* task_embedding,
                                                    int ch) {
    float acc = 0.f;
#ifdef __CUDA_ARCH__
    const float* row = d_TASK_PROJ[ch];
#else
    const float* row = TASK_PROJ[ch];
#endif
    #pragma unroll
    for (int d = 0; d < TASK_EMBED_DIM; ++d) {
        acc += row[d] * task_embedding[d];
    }
    return acc;
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
            // Task embedding broadcast spatially through the fixed 16->5
            // projection (A-201): channels 6..10 carry TASK_PROJ * e.
            #pragma unroll
            for (int c = CH_TASK_FIRST; c <= CH_TASK_LAST; ++c) {
                int t_idx = c - CH_TASK_FIRST;
                grid[grid_idx(y, x, c)] =
                    __float2half(task_channel_value(task_embedding, t_idx));
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
            // Task embedding broadcast on chans 6..10 (same fixed projection).
            #pragma unroll
            for (int c = CH_TASK_FIRST; c <= CH_TASK_LAST; ++c) {
                int t_idx = c - CH_TASK_FIRST;
                grid[grid_idx(y, x, c)] =
                    __float2half(task_channel_value(task_embedding, t_idx));
            }
        }
    }
    // Centered 4x4 region carries bmap_32 across the two aux channels:
    // (16 cells * 2 channels = 32 slots).
    __syncthreads();
    constexpr int CENTER = GRID_SIZE / 2;
    constexpr int LO = CENTER - 2;       // 30
    constexpr int HI = CENTER + 2;       // 34 (exclusive)
    if (threadIdx.y < PREDICTOR_SEED_REGION &&
        threadIdx.x < PREDICTOR_SEED_REGION) {
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
// Perception: a learned bank of N_PERC_FILTERS depthwise 3x3 filters (W_perc,
// shared across channels) convolved over each channel's neighborhood, giving a
// PERC_DIM = N_PERC_FILTERS * CA_CHANNELS perception vector. W_inter mixes that
// into a HIDDEN_DIM hidden representation (GELU). W_flow projects the hidden to
// a 16-channel delta added to the state. Reaction-diffusion (A-202) then adds
// spatial diffusion + decay to the chemical channels 0-5 on top of that delta.
constexpr int PERC_DIM    = N_PERC_FILTERS * CA_CHANNELS;   // 48

// Learned depthwise perception: perc_out[f*CA_CHANNELS + c] is filter f convolved
// over channel c's 3x3 toroidal neighborhood. W_perc layout: [N_PERC_FILTERS][3][3]
// row-major, i.e. W_perc[f*9 + (ky+1)*3 + (kx+1)] for offsets ky,kx in {-1,0,1}.
__device__ inline void sample_neighborhood(const __half* state,
                                           const float* W_perc,
                                           int y, int x,
                                           float* perc_out) {  // [PERC_DIM]
    // Toroidal wrap on the grid edge.
    auto at = [&](int dy, int dx, int c) -> float {
        int yy = (y + dy + GRID_SIZE) % GRID_SIZE;
        int xx = (x + dx + GRID_SIZE) % GRID_SIZE;
        return __half2float(state[grid_idx(yy, xx, c)]);
    };
    for (int f = 0; f < N_PERC_FILTERS; ++f) {
        const float* k = &W_perc[f * 9];
        for (int c = 0; c < CA_CHANNELS; ++c) {
            float acc = 0.f;
            #pragma unroll
            for (int ky = -1; ky <= 1; ++ky) {
                #pragma unroll
                for (int kx = -1; kx <= 1; ++kx) {
                    acc += k[(ky + 1) * 3 + (kx + 1)] * at(ky, kx, c);
                }
            }
            perc_out[f * CA_CHANNELS + c] = acc;
        }
    }
}

__device__ inline float gelu_approx(float x) {
    // Hendrycks-Gimpel approximation.
    const float k = GELU_K;                  // sqrt(2/pi)
    return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// Single CA step. Role-blind. Learned perception (W_perc) feeds the learned
// W_inter / W_flow path; W_flow produces a delta for all 16 channels, added to
// the state with residual timestep alpha: next = prev + alpha * F(x) (A-201).
// The chemical channels 0-5 are updated here like any other channel
// (cells produce/consume morphogens); reaction-diffusion (A-202) then adds
// spatial diffusion + decay to those channels in a following rd_step.
//   W_perc  : [N_PERC_FILTERS x 3 x 3]
//   W_inter : [PERC_DIM x HIDDEN_DIM]
//   W_flow  : [HIDDEN_DIM x CA_CHANNELS]
//
// One thread = one cell. Block layout is (16, 16); each block covers the
// 64x64 grid via a grid-stride loop.
__device__ inline void ca_step(const __half* state_curr,
                               __half* state_next,
                               const float* W_perc,
                               const float* W_inter,
                               const float* W_flow,
                               float alpha) {
    for (int by = 0; by < GRID_SIZE; by += blockDim.y) {
        for (int bx = 0; bx < GRID_SIZE; bx += blockDim.x) {
            int y = by + threadIdx.y;
            int x = bx + threadIdx.x;
            if (y >= GRID_SIZE || x >= GRID_SIZE) continue;

            float perc[PERC_DIM];
            sample_neighborhood(state_curr, W_perc, y, x, perc);

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

            // W_flow drives all 16 channels (chemicals included). rd_step then
            // adds spatial diffusion + decay to channels 0-5 on top of this.
            // Residual timestep: next = prev + RESIDUAL_ALPHA * F(x) (A-201).
            #pragma unroll
            for (int c = 0; c < CA_CHANNELS; ++c) {
                float acc = 0.f;
                #pragma unroll
                for (int h = 0; h < HIDDEN_DIM; ++h) {
                    acc += W_flow[h * CA_CHANNELS + c] * hidden[h];
                }
                float prev = __half2float(state_curr[grid_idx(y, x, c)]);
                float next = prev + alpha * acc;
                // Clamp to the FP16 representable range before narrowing.
                if (next >  FP16_MAX_VALUE) next =  FP16_MAX_VALUE;
                if (next < -FP16_MAX_VALUE) next = -FP16_MAX_VALUE;
                state_next[grid_idx(y, x, c)] = __float2half(next);
            }
        }
    }
    __syncthreads();
}

// 64-step forward, sampling bmap at BTRAJ_STEPS into bmap_traj. One block per
// organism. The kernel itself iterates the substrate; outer host code maps
// blocks to organisms.
//
// coeffs is the per-organism reaction-diffusion coefficient array (A-202). It
// may be null, in which case the chemical channels still evolve cellwise (the
// CA writes all 16 channels) but get no spatial diffusion. When non-null,
// rd_step adds diffusion + decay to channels 0-5 after each ca_step.
//
// forward_one is the shared body used by both the shared-weight kernel and the
// per-organism effective-weight kernel; both must produce identical results
// for identical weight banks (see evolution_regression.cu determinism check).
__device__ inline void forward_one(OrganismState* o,
                                   const ForwardInputs& in,
                                   const rd::Coefficients* coeffs,  // this organism's RD coeffs or null
                                   const float* W_perc,     // [N_PERC_FILTERS x 3 x 3]
                                   const float* W_inter,    // [PERC_DIM x HIDDEN_DIM]
                                   const float* W_flow,     // [HIDDEN_DIM x CA_CHANNELS]
                                   const float* W_bmap,     // [CA_CHANNELS x BMAP_DIM]
                                   float alpha) {           // residual timestep (A-201)
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
    for (int i = 0; i < BTRAJ_SAMPLES; ++i) steps[i] = d_BTRAJ_STEPS[i];

    __half* curr = o->grid;
    __half* next = o->scratch;
    int sample_idx = 0;

    for (int step = 1; step <= CA_STEPS; ++step) {
        // ca_step writes all 16 channels of next (chemicals included);
        // rd_step (when enabled) adds spatial diffusion + decay to next's
        // chemical channels 0-5, using curr's chemical field for the Laplacian.
        // Both read curr; ca_step then rd_step write next before the swap.
        ca_step(curr, next, W_perc, W_inter, W_flow, alpha);
        if (coeffs != nullptr) {
            rd::rd_step(curr, next, *coeffs);
        }
        __half* tmp = curr; curr = next; next = tmp;

        if (sample_idx < BTRAJ_SAMPLES && step == steps[sample_idx]) {
            project_bmap(curr,
                         W_bmap,
                         &o->bmap_traj[sample_idx * BMAP_DIM],
                         GLOBAL_CONTEXT_ENABLED
                             ? &o->sample_summary[sample_idx * CA_CHANNELS]
                             : nullptr);
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

// Shared-weight forward: every organism uses the same four weight banks.
__global__ void forward_kernel(OrganismState* organisms,
                               const ForwardInputs* inputs,
                               const rd::Coefficients* coeffs,
                               const float* W_perc,     // [N_PERC_FILTERS x 3 x 3]
                               const float* W_inter,    // [PERC_DIM x HIDDEN_DIM]
                               const float* W_flow,     // [HIDDEN_DIM x CA_CHANNELS]
                               const float* W_bmap,     // [CA_CHANNELS x BMAP_DIM]
                               float alpha,             // residual timestep
                               int n_organisms) {
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    const rd::Coefficients* org_coeffs =
        (coeffs != nullptr) ? &coeffs[org] : nullptr;
    forward_one(&organisms[org], inputs[org], org_coeffs,
                W_perc, W_inter, W_flow, W_bmap, alpha);
}

// Effective-weight forward: organism `org` uses the flat weight bank
// `eff_weights[bank_of[org] * weight_stride .. +weight_stride)`. The bank
// layout matches the concatenated shared layout (W_perc | W_inter | W_flow |
// W_bmap) so an identity bank_of with banks materialized as W_shared + delta
// reproduces exactly the shared-weight forward when every delta is empty.
// Used for SOT reference passes where each pool organism is compared against
// a reference roll-out computed with ITS OWN effective weights.
__global__ void forward_effective_kernel(OrganismState* organisms,
                                         const ForwardInputs* inputs,
                                         const rd::Coefficients* coeffs,
                                         const float* eff_weights,  // [n_banks * weight_stride]
                                         const int* bank_of,        // [n_organisms] or null (identity)
                                         int weight_stride,
                                         float alpha,               // residual timestep
                                         int n_organisms) {
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    int bank = (bank_of != nullptr) ? bank_of[org] : org;
    const float* wbase = &eff_weights[static_cast<size_t>(bank) * weight_stride];
    const rd::Coefficients* org_coeffs =
        (coeffs != nullptr) ? &coeffs[org] : nullptr;
    forward_one(&organisms[org], inputs[org], org_coeffs,
                wbase + genome::DELTA_OFF_PERC, wbase + genome::DELTA_OFF_INTER,
                wbase + genome::DELTA_OFF_FLOW, wbase + genome::DELTA_OFF_BMAP,
                alpha);
}

// Global average pool + W_bmap projection. Produces bmap_t at the requested
// step. Called inside forward_kernel after each BTRAJ step (forward-declared
// above).
//
// Spatial average over GRID_SIZE*GRID_SIZE cells produces a 16-d summary s_t
// (one value per channel); W_bmap is [CA_CHANNELS x BMAP_DIM].
//
// Deterministic reduction: each thread accumulates its cells into thread-local
// registers, then a tree reduction in shared memory produces the final sum.
// No atomicAdd — guarantees identical bit patterns for the same thread layout
// across kernel launches. Required because BTRAJ feeds scoring, archive, and
// loss (see cuda_engineering.md section 4.1).
__device__ inline void project_bmap(__half* state,
                                    const float* W_bmap,
                                    float* bmap_out_32,
                                    float* summary_out) {
    // 256 threads (16x16 block), CA_CHANNELS = 16.
    // Shared memory layout: [nthreads * CA_CHANNELS] for the tree reduction.
    // The reduction hard-codes 256 threads; if the forward block dims ever
    // change, this must be updated.  The static_assert guards against that.
    constexpr int NTHREADS = 256; // blockDim.x * blockDim.y
    static_assert(NTHREADS == 16 * 16, "project_bmap assumes a 16x16 block (256 threads)");
    __shared__ float reduce_buf[NTHREADS * CA_CHANNELS];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;

    // Phase 1: Each thread accumulates its cells into thread-local registers.
    float local_sum[CA_CHANNELS];
    #pragma unroll
    for (int c = 0; c < CA_CHANNELS; ++c) local_sum[c] = 0.f;

    for (int idx = tid; idx < GRID_SIZE * GRID_SIZE; idx += nthreads) {
        int y = idx / GRID_SIZE;
        int x = idx % GRID_SIZE;
        #pragma unroll
        for (int c = 0; c < CA_CHANNELS; ++c) {
            local_sum[c] += __half2float(state[grid_idx(y, x, c)]);
        }
    }

    // Phase 2: Tree reduction in shared memory across all threads.
    // Store thread-local sums.
    #pragma unroll
    for (int c = 0; c < CA_CHANNELS; ++c) {
        reduce_buf[tid * CA_CHANNELS + c] = local_sum[c];
    }
    __syncthreads();

    // Binary tree reduction. At each step, thread tid adds the value from
    // tid + stride. Deterministic because the reduction tree is fixed.
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int c = 0; c < CA_CHANNELS; ++c) {
                reduce_buf[tid * CA_CHANNELS + c] += reduce_buf[(tid + stride) * CA_CHANNELS + c];
            }
        }
        __syncthreads();
    }

    // Thread 0 now has the total sum in reduce_buf[0..CA_CHANNELS-1].
    // Compute mean into shared memory for all threads to read.
    __shared__ float summary[CA_CHANNELS];
    constexpr float scale = 1.0f / static_cast<float>(GRID_SIZE * GRID_SIZE);
    if (tid < CA_CHANNELS) {
        summary[tid] = reduce_buf[tid] * scale;
    }
    __syncthreads();

    // The pre-broadcast summary is the backward's reference for the context
    // adjoint (the aux channels are overwritten below, so it cannot be
    // recovered from the post-broadcast state).
    if (summary_out != nullptr && tid < CA_CHANNELS) {
        summary_out[tid] = summary[tid];
    }

    // Phase 3: Project: bmap_out_32[d] = sum_c W_bmap[c, d] * summary[c].
    for (int d = tid; d < BMAP_DIM; d += nthreads) {
        float acc = 0.f;
        for (int c = 0; c < CA_CHANNELS; ++c) {
            acc += W_bmap[c * BMAP_DIM + d] * summary[c];
        }
        bmap_out_32[d] = acc;
    }
    __syncthreads();

    // Phase 3b (A-203, I5): broadcast the global context into channels 14-15.
    // W_ctx follows W_bmap in the flat layout, so it is derived from the same
    // bank pointer and needs no extra kernel argument.
    if (GLOBAL_CONTEXT_ENABLED) {
        const float* W_ctx = W_bmap + CA_CHANNELS * BMAP_DIM;
        __shared__ float ctx[CTX_K];
        if (tid < CTX_K) {
            float tmp[CTX_K];
            context_map(summary, W_ctx, tmp);
            ctx[tid] = tmp[tid];
        }
        __syncthreads();
        for (int idx = tid; idx < GRID_SIZE * GRID_SIZE; idx += nthreads) {
            int y = idx / GRID_SIZE;
            int x = idx % GRID_SIZE;
            #pragma unroll
            for (int k = 0; k < CTX_K; ++k) {
                state[grid_idx(y, x, CH_AUX_FIRST + k)] =
                    __float2half(ctx[k]);
            }
        }
    }
    __syncthreads();
}

// ---- Public host launchers -----------------------------------------------
// launch_forward: one block of (16, 16) threads per organism on `stream`.
// Weights (W_perc, W_inter, W_flow, W_bmap) are role-blind and shared across
// organisms. coeffs may be null to skip reaction-diffusion. Block dim is fixed
// at 16x16 because seed_predictor_grid's centered-4x4 write and project_bmap's
// reductions assume it.
inline void launch_forward(OrganismState* organisms,
                           const ForwardInputs* inputs,
                           const rd::Coefficients* coeffs,
                           const float* W_perc,
                           const float* W_inter,
                           const float* W_flow,
                           const float* W_bmap,
                           float alpha,
                           int n_organisms,
                           cudaStream_t stream) {
    if (n_organisms <= 0) return;
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>(n_organisms));
    forward_kernel<<<grid, block, 0, stream>>>(
        organisms, inputs, coeffs, W_perc, W_inter, W_flow, W_bmap,
        alpha, n_organisms);
}

// launch_forward_effective: per-organism flat weight banks (W_shared + delta)
// selected through bank_of (identity when bank_of is null). Same block layout
// as launch_forward.
inline void launch_forward_effective(OrganismState* organisms,
                                     const ForwardInputs* inputs,
                                     const rd::Coefficients* coeffs,
                                     const float* eff_weights,
                                     const int* bank_of,
                                     int weight_stride,
                                     float alpha,
                                     int n_organisms,
                                     cudaStream_t stream) {
    if (n_organisms <= 0) return;
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>(n_organisms));
    forward_effective_kernel<<<grid, block, 0, stream>>>(
        organisms, inputs, coeffs, eff_weights, bank_of,
        weight_stride, alpha, n_organisms);
}

// Extract bmap_64 (the last BTRAJ slot) into a contiguous output array.
// One thread per float, one organism per block-row.
__global__ void extract_descriptor_kernel(const OrganismState* organisms,
                                          float* descriptors_out,
                                          int n_organisms) {
    int org = blockIdx.x;
    int d   = threadIdx.x;
    if (org >= n_organisms || d >= BMAP_DIM) return;
    // bmap_64 is the last BTRAJ sample: index (BTRAJ_SAMPLES - 1) * BMAP_DIM + d.
    descriptors_out[org * BMAP_DIM + d] =
        organisms[org].bmap_traj[(BTRAJ_SAMPLES - 1) * BMAP_DIM + d];
}

inline void extract_descriptor(const OrganismState* organisms,
                                float* descriptors_out,
                                int n_organisms,
                                cudaStream_t stream) {
    if (n_organisms <= 0) return;
    dim3 block(BMAP_DIM);  // 32 threads
    dim3 grid(static_cast<unsigned>(n_organisms));
    extract_descriptor_kernel<<<grid, block, 0, stream>>>(
        organisms, descriptors_out, n_organisms);
}

}  // namespace slime::nca

#endif  // COEVO_NCA_ENGINE_CU

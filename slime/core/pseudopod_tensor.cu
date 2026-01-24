
#ifndef PSEUDOPOD_TENSOR_CU
#define PSEUDOPOD_TENSOR_CU

#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
#include "../utils/genome_params.cuh"
#include "pseudopod.cu"
#include "../metrics/hardware_geometry.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

__global__ void tensor_core_conv3x3_kernel(
    half* __restrict__ input,
    half* __restrict__ kernel,
    float* __restrict__ output,
    int grid_size,
    int channels
) {

    const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int tile_x = (blockIdx.x * WMMA_TILE_DIM) % grid_size;
    const int tile_y = (blockIdx.y * WMMA_TILE_DIM) % grid_size;

    if (tile_x >= grid_size || tile_y >= grid_size) return;

    float mass_before = 0.0f;
    float mass_after = 0.0f;

    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    __shared__ half tile_shared[WMMA_TILE_DIM][WMMA_TILE_DIM + BANK_PAD];
    __shared__ half kernel_shared[WMMA_TILE_DIM][WMMA_TILE_DIM + BANK_PAD];

    if (threadIdx.x < BLOCK_SIZE) {
        int ti = threadIdx.x / WMMA_TILE_DIM;
        int tj = threadIdx.x % WMMA_TILE_DIM;
        int y = tile_y + ti;
        int x = tile_x + tj;
        if (y < grid_size && x < grid_size && tj < channels) {
            tile_shared[ti][tj] = input[y * grid_size * channels + x * channels + tj];
            mass_before += __half2float(tile_shared[ti][tj]);
        } else {
            tile_shared[ti][tj] = __float2half(0.0f);
        }
    }

    // Warp-level reduction for mass_before using tile ops
    mass_before = WarpReduce<WARP_SIZE>::sum(mass_before);

    __syncthreads();

    load_matrix_sync(a_frag, (half*)tile_shared, WMMA_TILE_DIM);

    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            if (threadIdx.x < BLOCK_SIZE) {
                int ki = threadIdx.x / WMMA_TILE_DIM;
                int kj = threadIdx.x % WMMA_TILE_DIM;
                if (ki < channels && kj < channels) {
                    kernel_shared[ki][kj] = kernel[ky * 3 * channels * channels + kx * channels * channels + ki * channels + kj];
                } else {
                    kernel_shared[ki][kj] = __float2half(0.0f);
                }
            }
            __syncthreads();

            load_matrix_sync(b_frag, (half*)kernel_shared, WMMA_TILE_DIM);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    __shared__ float result_shared[WMMA_TILE_DIM][WMMA_TILE_DIM + BANK_PAD];
    store_matrix_sync(result_shared[0], c_frag, 16, mem_row_major);
    __syncthreads();

    if (threadIdx.x < BLOCK_SIZE) {
        int ri = threadIdx.x / WMMA_TILE_DIM;
        int rj = threadIdx.x % WMMA_TILE_DIM;
        int y = tile_y + ri;
        int x = tile_x + rj;
        if (y < grid_size && x < grid_size && rj < channels) {
            int idx = y * grid_size * channels + x * channels + rj;
            output[idx] = result_shared[ri][rj];
            mass_after += result_shared[ri][rj];
        }
    }

    // Warp-level reduction for mass_after using tile ops
    mass_after = WarpReduce<WARP_SIZE>::sum(mass_after);

    // Use warpId to coordinate multi-warp convergence check via ballot primitives
    int mass_conserved = approx_equal(mass_after, mass_before);
    unsigned int ballot = WarpReduce<WARP_SIZE>::ballot(mass_conserved);

    // Store per-warp ballot results in shared memory indexed by warpId
    __shared__ unsigned int warp_ballots[256 / WARP_SIZE];
    if (laneId == 0) {
        warp_ballots[warpId] = ballot;
    }
    __syncthreads();

    // Check if ALL warps achieved mass conservation (all bits set)
    int all_converged = WarpReduce<WARP_SIZE>::all(
        (threadIdx.x < (blockDim.x / WARP_SIZE)) ?
        (warp_ballots[threadIdx.x] == 0xFFFFFFFF) : 1
    );

    // Only apply normalization if mass conservation converged globally
    if (all_converged && laneId == 0 && is_meaningful(mass_after, mass_before)) {
        float scale = mass_before / mass_after;

        // Broadcast scale to all lanes in warp using cooperative groups
        auto tile = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
        scale = tile.shfl(scale, 0);

        // Apply mass-conserving normalization
        if (laneId < channels) {
            int idx = tile_y * grid_size * channels + tile_x * channels + laneId;
            output[idx] *= scale;
        }
    }
}



__global__ void compute_effective_rank_from_latent_tensor_kernel(
    float* __restrict__ latent_genome,
    float* __restrict__ effective_rank,
    int latent_dim
) {
    float mean = 0.0f;
    for (int i = 0; i < latent_dim; i++) {
        mean += latent_genome[i];
    }
    mean /= latent_dim;

    float variance = 0.0f;
    for (int i = 0; i < latent_dim; i++) {
        float diff = latent_genome[i] - mean;
        variance += diff * diff;
    }
    variance /= latent_dim;

    if (variance < 0.0f) {
        printf("FATAL [effective_rank_tensor]: variance=%f\n", variance);
        *effective_rank = 0.0f;
        return;
    }
    *effective_rank = sqrtf(variance) * latent_dim;
}

__global__ void compute_coherence_tensor_kernel(
    float* __restrict__ loss_history,
    float* __restrict__ coherence,
    int history_length
) {
    int tid = threadIdx.x;

    float local_improvement = 0.0f;

    if (tid < history_length - 1) {
        float current_loss = loss_history[tid];
        float next_loss = loss_history[tid + 1];

        if (current_loss <= 0.0f) {
            printf("FATAL [compute_coherence_tensor]: current_loss[%d]=%f\n", tid, current_loss);
            return;
        }
        local_improvement = fmaxf(0.0f, (current_loss - next_loss) / current_loss);
    }

    float total = BlockReduce<BLOCK_SIZE>::sum(local_improvement);

    if (tid == 0) {
        *coherence = total / (history_length - 1);
    }
}

__global__ void init_multihead_ca_tensor_kernel(
    MultiHeadCAState* state,
    unsigned int seed,
    int num_heads,
    int channels,
    int head_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState_t rand_state;
    curand_init(seed, tid, 0, &rand_state);

    int perception_size = num_heads * channels * head_dim;
    if (tid < perception_size) {
        float fan_in = (float)channels;
        float fan_out = (float)head_dim;
        float scale = sqrtf(2.0f / (fan_in + fan_out));
        float val = validated_curand_normal(&rand_state, "init_tensor_ca_perception", tid) * scale;
        state->perception_weights[tid] = __float2half(val);
    }

    int interaction_size = num_heads * head_dim * head_dim;
    if (tid < interaction_size) {
        float fan_in = (float)head_dim;
        float fan_out = (float)head_dim;
        float scale = sqrtf(2.0f / (fan_in + fan_out));
        float val = validated_curand_normal(&rand_state, "init_tensor_ca_interaction", tid) * scale;
        state->interaction_weights[tid] = __float2half(val);
    }

    int value_size = num_heads * head_dim * channels;
    if (tid < value_size) {
        float fan_in = (float)head_dim;
        float fan_out = (float)channels;
        float scale = sqrtf(2.0f / (fan_in + fan_out));
        float val = validated_curand_normal(&rand_state, "init_tensor_ca_value", tid) * scale;
        state->value_weights[tid] = __float2half(val);
    }
}

__global__ void pipelined_ca_kernel(
    float* __restrict__ ca_state_in,
    float* __restrict__ ca_state_out,
    half* __restrict__ perception_weights,
    half* __restrict__ interaction_weights,
    half* __restrict__ value_weights,
    int grid_size,
    int batch_size,
    ArchitectureParams arch
) {
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;

    if (cell_idx >= grid_size * grid_size || batch_id >= batch_size) return;

    __shared__ float input_tile[BLOCK_SIZE];
    __shared__ float perception_out[BLOCK_SIZE];
    __shared__ float interaction_out[BLOCK_SIZE];

    AsyncCopy<WARP_SIZE>::memcpy_async_tile(&input_tile[threadIdx.x],
                                 &ca_state_in[batch_id * grid_size * grid_size * arch.channels + cell_idx],
                                 sizeof(float));
    AsyncCopy<WARP_SIZE>::commit_group();

    int head_id = blockIdx.z;
    half* perc_w = perception_weights + head_id * arch.channels * arch.head_dim;
    half* inter_w = interaction_weights + head_id * arch.head_dim * arch.head_dim;
    half* val_w = value_weights + head_id * arch.head_dim * arch.channels;

    AsyncCopy<WARP_SIZE>::wait_group();

    float perc_accum = 0.0f;
    for (int i = 0; i < arch.channels; i++) {
        perc_accum += input_tile[threadIdx.x] * __half2float(perc_w[i]);
    }
    perception_out[threadIdx.x] = fmaxf(0.0f, perc_accum);
    __syncthreads();

    float inter_accum = 0.0f;
    for (int i = 0; i < arch.head_dim; i++) {
        inter_accum += perception_out[threadIdx.x] * __half2float(inter_w[i]);
    }
    interaction_out[threadIdx.x] = fmaxf(0.0f, inter_accum);
    __syncthreads();

    float value_accum = 0.0f;
    for (int i = 0; i < arch.channels; i++) {
        value_accum += interaction_out[threadIdx.x] * __half2float(val_w[i]);
    }

    AsyncCopy<WARP_SIZE>::memcpy_async_tile(&ca_state_out[batch_id * grid_size * grid_size * arch.channels + cell_idx],
                                 &value_accum,
                                 sizeof(float));
    AsyncCopy<WARP_SIZE>::commit_group();
    AsyncCopy<WARP_SIZE>::wait_group();
}

#endif

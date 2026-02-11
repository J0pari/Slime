
#ifndef PSEUDOPOD_CU
#define PSEUDOPOD_CU

#include "../config/config.cu"
#include "organism.cu"
#include "../utils/genome_params.cuh"
#include "../memory/pool.cu"
#include "../metrics/hardware_geometry.cu"
#include "flow_lenia_ops.cuh"
#include "ca_state.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

__global__ void compute_coherence_kernel(
    float* __restrict__ loss_history,
    float* __restrict__ coherence,
    int history_length
) {
    int tid = threadIdx.x;

    float local_improvement = 0.0f;

    if (tid < history_length - 1) {
        float current_loss = loss_history[tid];
        float next_loss = loss_history[tid + 1];
        DEVICE_FATAL_IF(current_loss <= 0.0f, "coherence_kernel: loss history contains non-positive value");
        local_improvement = fmaxf(0.0f, (current_loss - next_loss) / current_loss);
    }

    float total = BlockReduce<BLOCK_SIZE>::sum(local_improvement);

    if (tid == 0) {
        *coherence = total / (history_length - 1);
    }
}

__device__ float get_ca_xavier_scale(
    int flat_idx,
    int num_heads,
    int channels,
    int head_dim,
    int* out_matrix,        
    int* out_local_idx      
) {
    int perception_size = num_heads * channels * head_dim;
    int interaction_size = num_heads * head_dim * head_dim;
    int value_size = num_heads * head_dim * channels;

    if (flat_idx < perception_size) {
        *out_matrix = 0;
        *out_local_idx = flat_idx;
        float fan_in = (float)(CA_KERNEL_CELL_COUNT * channels);
        float fan_out = (float)head_dim;
        return sqrtf(2.0f / (fan_in + fan_out));
    }
    flat_idx -= perception_size;

    if (flat_idx < interaction_size) {
        *out_matrix = 1;
        *out_local_idx = flat_idx;
        float fan_in = (float)head_dim;
        float fan_out = (float)head_dim;
        return sqrtf(2.0f / (fan_in + fan_out));
    }
    flat_idx -= interaction_size;

    if (flat_idx < value_size) {
        *out_matrix = 2;
        *out_local_idx = flat_idx;
        float fan_in = (float)head_dim;
        float fan_out = (float)channels;
        return sqrtf(2.0f / (fan_in + fan_out));
    }

    *out_matrix = -1;
    *out_local_idx = -1;
    return 0.0f;
}

__device__ void init_ca_weights_xavier(
    half* perception_weights,
    half* interaction_weights,
    half* value_weights,
    int weight_idx,
    int num_heads,
    int channels,
    int head_dim,
    curandState_t* rand_state
) {
    int perception_size = num_heads * channels * head_dim;
    int interaction_size = num_heads * head_dim * head_dim;
    int value_size = num_heads * head_dim * channels;

    int matrix, local_idx;
    float scale;

    if (weight_idx < perception_size) {
        scale = get_ca_xavier_scale(weight_idx, num_heads, channels, head_dim, &matrix, &local_idx);
        perception_weights[weight_idx] = __float2half(curand_normal(rand_state) * scale);
    }
    if (weight_idx < interaction_size) {
        scale = get_ca_xavier_scale(perception_size + weight_idx, num_heads, channels, head_dim, &matrix, &local_idx);
        interaction_weights[weight_idx] = __float2half(curand_normal(rand_state) * scale);
    }
    if (weight_idx < value_size) {
        scale = get_ca_xavier_scale(perception_size + interaction_size + weight_idx, num_heads, channels, head_dim, &matrix, &local_idx);
        value_weights[weight_idx] = __float2half(curand_normal(rand_state) * scale);
    }
}

__global__ void init_organism_ca_weights_kernel(
    ComponentPool* __restrict__ pool,
    ArchitectureParams arch
) {
    int compact_idx = blockIdx.y;
    if (compact_idx >= pool->alive_indices_count) return;

    int entry_idx = pool->alive_indices[compact_idx];
    PoolEntry* entry = &pool->entries[entry_idx];
    DEVICE_FATAL_IF(!pool->alive_flags[entry_idx], "init_organism_ca_weights_kernel: dead entry in alive_indices");

    MultiHeadCAState* ca_state = entry->ca_state;
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t organism_seed = entry->genome_hash ^ (entry->id * 0x9E3779B97F4A7C15ULL);
    curandState_t rand_state;
    curand_init(organism_seed, weight_idx, 0, &rand_state);

    init_ca_weights_xavier(
        ca_state->perception_weights,
        ca_state->interaction_weights,
        ca_state->value_weights,
        weight_idx, arch.num_heads, arch.channels, arch.head_dim,
        &rand_state
    );
}

__global__ void init_ca_weights_kernel(
    half* perception_weights,
    half* interaction_weights,
    half* value_weights,
    int num_heads,
    int channels,
    int head_dim,
    unsigned long seed
) {
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t rand_state;
    curand_init(seed, weight_idx, 0, &rand_state);

    init_ca_weights_xavier(
        perception_weights, interaction_weights, value_weights,
        weight_idx, num_heads, channels, head_dim,
        &rand_state
    );
}

#endif

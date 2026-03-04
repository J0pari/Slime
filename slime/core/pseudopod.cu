
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

__device__ void compute_coherence_device(Organism* organism) {
    float* loss_history = organism->loss_history;
    float* coherence = organism->coherence_output;
    int history_length = organism->loss_history_length;

    int tid = threadIdx.x;

    float local_improvement = 0.0f;

    if (tid < history_length - 1) {
        float current_loss = loss_history[tid];
        float next_loss = loss_history[tid + 1];
        DEVICE_FATAL_IF(current_loss <= 0.0f, "compute_coherence_device: loss history contains non-positive value");
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
    int flow_projection_size = num_heads * 2 * head_dim;

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

    if (flat_idx < flow_projection_size) {
        *out_matrix = 2;
        *out_local_idx = flat_idx;
        float fan_in = (float)head_dim;
        float fan_out = 2.0f;
        return sqrtf(2.0f / (fan_in + fan_out));
    }

    *out_matrix = -1;
    *out_local_idx = -1;
    return 0.0f;
}

__device__ void init_ca_weights_xavier(
    half* perception_weights,
    half* interaction_weights,
    half* flow_projection_weights,
    int weight_idx,
    int num_heads,
    int channels,
    int head_dim,
    curandState_t* rand_state
) {
    int perception_size = num_heads * channels * head_dim;
    int interaction_size = num_heads * head_dim * head_dim;
    int flow_projection_size = num_heads * 2 * head_dim;

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
    if (weight_idx < flow_projection_size) {
        float fp_scale = sqrtf(2.0f / ((float)head_dim + 2.0f));
        flow_projection_weights[weight_idx] = __float2half(curand_normal(rand_state) * fp_scale);
    }
}

__device__ void init_organism_ca_weights_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    Architecture arch = Architecture::maxBounds();

    int perception_size = arch.num_heads * arch.channels * arch.head_dim;
    int interaction_size = arch.num_heads * arch.head_dim * arch.head_dim;
    int flow_projection_size = arch.num_heads * 2 * arch.head_dim;
    int max_weight_size = max(perception_size, max(interaction_size, flow_projection_size));

    int total_work = pool->alive_indices_count * max_weight_size;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int global_idx = thread_id; global_idx < total_work; global_idx += total_threads) {
        int compact_idx = global_idx / max_weight_size;
        int weight_idx = global_idx % max_weight_size;

        int entry_idx = pool->alive_indices[compact_idx];
        PoolEntry* entry = &pool->entries[entry_idx];
        DEVICE_FATAL_IF(!pool->alive_flags[entry_idx], "init_organism_ca_weights_device: dead entry in alive_indices");

        MultiHeadCAState* ca_state = entry->ca_state;

        uint64_t organism_seed = entry->genome_hash ^ (entry->id * 0x9E3779B97F4A7C15ULL);
        curandState_t rand_state;
        curand_init(organism_seed, weight_idx, 0, &rand_state);

        init_ca_weights_xavier(
            ca_state->perception_weights,
            ca_state->interaction_weights,
            ca_state->flow_projection_weights,
            weight_idx, arch.num_heads, arch.channels, arch.head_dim,
            &rand_state
        );
    }
}


__global__ void init_ca_weights_kernel(
    half* perception_weights,
    half* interaction_weights,
    half* flow_projection_weights,
    int num_heads, int channels, int head_dim,
    unsigned long seed
) {
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t rand_state;
    curand_init(seed, weight_idx, 0, &rand_state);

    init_ca_weights_xavier(
        perception_weights, interaction_weights, flow_projection_weights,
        weight_idx, num_heads, channels, head_dim,
        &rand_state
    );
}

#endif

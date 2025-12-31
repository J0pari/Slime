
#ifndef PSEUDOPOD_CU
#define PSEUDOPOD_CU

#include "../config/config.cu"
#include "../utils/genome_params.cuh"
#include "../memory/pool.cu"
#include "flow_lenia_ops.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

struct MultiHeadCAState {
    half* perception_weights;
    half* interaction_weights;
    half* value_weights;
    float* ca_concentration;
    float* ca_output;
    float* affinity_reduced;
    float* flow_field;
    float* reintegration_buffer;
    half* fp16_workspace;
    float* fp32_workspace;
};

__global__ void multi_head_ca_kernel(
    float* __restrict__ ca_state,
    half* __restrict__ perception_weights,
    half* __restrict__ interaction_weights,
    half* __restrict__ value_weights,
    float* __restrict__ ca_output,
    int batch_size,
    int grid_size,
    ArchitectureParams arch
) {

    int head_id = blockIdx.y;
    int batch_id = blockIdx.z;
    int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_y = blockIdx.x * blockDim.y + threadIdx.y;

    if (cell_x >= grid_size || cell_y >= grid_size) return;

    __shared__ float neighborhood[3][3][MAX_HEAD_DIM + BANK_PAD];

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = clamp(cell_x + dx, 0, grid_size - 1);
            int ny = clamp(cell_y + dy, 0, grid_size - 1);

            int idx = batch_id * grid_size * grid_size * arch.head_dim +
                     ny * grid_size * arch.head_dim +
                     nx * arch.head_dim;

            if (threadIdx.z < arch.head_dim) {

                neighborhood[dy + 1][dx + 1][threadIdx.z] = ca_state[idx + threadIdx.z];
            }
        }
    }
    __syncthreads();

    float perception[MAX_HEAD_DIM];

    for (int i = 0; i < arch.head_dim; i++) {
        float accum = 0.0f;

        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
                for (int c = 0; c < arch.channels; c++) {
                    int weight_idx = head_id * arch.channels * arch.head_dim +
                                    c * arch.head_dim + i;

                    float neighbor_val = neighborhood[dy][dx][c];
                    float weight_val = __half2float(perception_weights[weight_idx]);
                    accum += neighbor_val * weight_val;
                }
            }
        }

        perception[i] = fmaxf(0.0f, accum);
    }

    float interaction[MAX_HEAD_DIM];

    for (int i = 0; i < arch.head_dim; i++) {
        float accum = 0.0f;

        for (int j = 0; j < arch.head_dim; j++) {
            int weight_idx = head_id * arch.head_dim * arch.head_dim +
                           j * arch.head_dim + i;

            float weight_val = __half2float(interaction_weights[weight_idx]);
            accum += perception[j] * weight_val;
        }

        float x = accum;
        interaction[i] = GELU_SCALE * x * (GELU_OFFSET + tanhf(GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC_COEFFICIENT * x * x * x)));
    }

    float output[MAX_CHANNELS];

    for (int i = 0; i < arch.channels; i++) {
        float accum = 0.0f;

        for (int j = 0; j < arch.head_dim; j++) {
            int weight_idx = head_id * arch.head_dim * arch.channels +
                           j * arch.channels + i;

            float weight_val = __half2float(value_weights[weight_idx]);
            accum += interaction[j] * weight_val;
        }

        output[i] = accum;
    }

    int out_idx = batch_id * arch.num_heads * grid_size * grid_size * arch.channels +
                  head_id * grid_size * grid_size * arch.channels +
                  cell_y * grid_size * arch.channels +
                  cell_x * arch.channels;

    for (int i = 0; i < arch.channels; i++) {
        ca_output[out_idx + i] = output[i];
    }
}

__global__ void reduce_affinity_kernel(
    ComponentPool* __restrict__ pool,
    int max_grid_size,
    int entry_idx
) {
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    int grid_size = entry->grid_size;
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = grid_size * grid_size;

    if (cell_idx >= total_cells) return;

    MultiHeadCAState* ca_state = entry->ca_state;
    ArchitectureParams arch;
    arch.num_heads = entry->num_heads;
    arch.head_dim = entry->head_dim;

    float U = FlowLeniaOps::reduce_affinity_warp(
        ca_state->ca_output, cell_idx, grid_size, arch.num_heads, arch.head_dim
    );

    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        ca_state->affinity_reduced[cell_idx] = U;
    }
}

__global__ void compute_flow_field_kernel(
    ComponentPool* __restrict__ pool,
    int max_grid_size,
    int entry_idx
) {
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    int grid_size = entry->grid_size;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    MultiHeadCAState* ca_state = entry->ca_state;
    float beta_A = entry->flow_beta_A;
    float n = entry->flow_n;
    int channels = entry->channels;

    int cell_idx = y * grid_size + x;

    float U_center = ca_state->affinity_reduced[cell_idx];
    int x_E = min(x + 1, grid_size - 1);
    int y_N = min(y + 1, grid_size - 1);
    float U_E = ca_state->affinity_reduced[y * grid_size + x_E];
    float U_N = ca_state->affinity_reduced[y_N * grid_size + x];

    float A_sum_center = 0.0f;
    float A_sum_E = 0.0f;
    float A_sum_N = 0.0f;

    for (int c = 0; c < channels; c++) {
        A_sum_center += ca_state->ca_concentration[cell_idx * channels + c];
        A_sum_E += ca_state->ca_concentration[(y * grid_size + x_E) * channels + c];
        A_sum_N += ca_state->ca_concentration[(y_N * grid_size + x) * channels + c];
    }

    float2 F = FlowLeniaOps::compute_flow_at(
        U_center, U_E, U_N,
        A_sum_center, A_sum_E, A_sum_N,
        beta_A, n
    );

    ca_state->flow_field[cell_idx * 2 + 0] = F.x;
    ca_state->flow_field[cell_idx * 2 + 1] = F.y;
}

__global__ void reintegration_redistribute_kernel(
    ComponentPool* __restrict__ pool,
    int max_grid_size,
    int entry_idx
) {
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    int grid_size = entry->grid_size;
    int source_x = blockIdx.x;
    int source_y = blockIdx.y;

    if (source_x >= grid_size || source_y >= grid_size) return;

    MultiHeadCAState* ca_state = entry->ca_state;
    float dt = entry->flow_resource_dt;
    float s = entry->flow_s;
    int channels = entry->channels;

    int source_idx = source_y * grid_size + source_x;

    float Fx = ca_state->flow_field[source_idx * 2 + 0];
    float Fy = ca_state->flow_field[source_idx * 2 + 1];

    float displaced_x = (float)source_x + dt * Fx;
    float displaced_y = (float)source_y + dt * Fy;

    int center_x = (int)floorf(displaced_x);
    int center_y = (int)floorf(displaced_y);

    for (int ty = center_y - 1; ty <= center_y + 1; ty++) {
        for (int tx = center_x - 1; tx <= center_x + 1; tx++) {
            if (tx < 0 || tx >= grid_size || ty < 0 || ty >= grid_size) continue;

            float I_val = FlowLeniaOps::gaussian_overlap_integral(
                displaced_x, displaced_y, s, tx, ty
            );

            int target_idx = ty * grid_size + tx;

            for (int c = threadIdx.x; c < channels; c += blockDim.x) {
                float mass = ca_state->ca_concentration[source_idx * channels + c] * I_val;
                atomicAdd(&ca_state->reintegration_buffer[target_idx * channels + c], mass);
            }
        }
    }
}

__global__ void clear_reintegration_buffer_kernel(ComponentPool* pool, int max_buffer_size, int entry_idx) {
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    int buffer_size = entry->grid_size * entry->grid_size * entry->channels;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < buffer_size) {
        entry->ca_state->reintegration_buffer[idx] = 0.0f;
    }
}

__global__ void copy_reintegration_to_concentration_kernel(ComponentPool* pool, int max_buffer_size, int entry_idx) {
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    int buffer_size = entry->grid_size * entry->grid_size * entry->channels;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < buffer_size) {
        entry->ca_state->ca_concentration[idx] = entry->ca_state->reintegration_buffer[idx];
    }
}

__global__ void compute_effective_rank_from_latent_kernel(
    ComponentPool* pool,
    float* effective_rank_history,
    float* workspace_genomes,
    int latent_dim,
    int entry_idx
) {
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    float* latent_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];

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
        return;
    }
    effective_rank_history[entry_idx] = sqrtf(variance) * latent_dim;
}

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

        if (current_loss <= 0.0f) {
            printf("FATAL [compute_coherence]: current_loss[%d]=%f\n", tid, current_loss);
            return;
        }
        local_improvement = fmaxf(0.0f, (current_loss - next_loss) / current_loss);
    }

    float total = BlockReduce<BLOCK_SIZE>::sum(local_improvement);

    if (tid == 0) {
        *coherence = total / (history_length - 1);
    }
}

__global__ void init_organism_ca_weights_kernel(
    ComponentPool* __restrict__ pool,
    ArchitectureParams arch
) {
    int entry_idx = blockIdx.y;
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    MultiHeadCAState* ca_state = entry->ca_state;
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int perception_size = arch.num_heads * arch.channels * arch.head_dim;
    int interaction_size = arch.num_heads * arch.head_dim * arch.head_dim;
    int value_size = arch.num_heads * arch.head_dim * arch.channels;

    uint64_t organism_seed = entry->genome_hash ^ (entry->id * 0x9E3779B97F4A7C15ULL);

    curandState_t rand_state;
    curand_init(organism_seed, weight_idx, 0, &rand_state);

    if (weight_idx < perception_size) {
        float fan_in = (float)arch.channels;
        float fan_out = (float)arch.head_dim;
        float scale = sqrtf(2.0f / (fan_in + fan_out));
        float val = curand_normal(&rand_state) * scale;
        ca_state->perception_weights[weight_idx] = __float2half(val);
    }

    if (weight_idx < interaction_size) {
        float scale = sqrtf(2.0f / (float)arch.head_dim);
        float val = curand_normal(&rand_state) * scale;
        ca_state->interaction_weights[weight_idx] = __float2half(val);
    }

    if (weight_idx < value_size) {
        float scale = sqrtf(2.0f / (float)arch.head_dim);
        float val = curand_normal(&rand_state) * scale;
        ca_state->value_weights[weight_idx] = __float2half(val);
    }
}

#endif

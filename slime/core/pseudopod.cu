
#ifndef PSEUDOPOD_CU
#define PSEUDOPOD_CU

#include "../config/config.cu"
#include "../utils/genome_params.cuh"
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
                for (int c = 0; c < arch.head_dim; c++) {
                    int weight_idx = head_id * arch.channels * arch.hidden_dim +
                                    c * arch.hidden_dim + i;

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
            int weight_idx = head_id * arch.channels * arch.hidden_dim +
                           j * arch.hidden_dim + i;

            float weight_val = __half2float(interaction_weights[weight_idx]);
            accum += perception[j] * weight_val;
        }

        float x = accum;
        interaction[i] = GELU_SCALE * x * (GELU_OFFSET + tanhf(GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC_COEFFICIENT * x * x * x)));
    }

    float output[MAX_HEAD_DIM];

    for (int i = 0; i < arch.head_dim; i++) {
        float accum = 0.0f;

        for (int j = 0; j < arch.hidden_dim; j++) {
            int weight_idx = head_id * arch.hidden_dim * arch.channels +
                           j * arch.channels + i;

            float weight_val = __half2float(value_weights[weight_idx]);
            accum += interaction[j % arch.head_dim] * weight_val;
        }

        output[i] = accum;
    }

    int out_idx = batch_id * arch.num_heads * grid_size * grid_size * arch.head_dim +
                  head_id * grid_size * grid_size * arch.head_dim +
                  cell_y * grid_size * arch.head_dim +
                  cell_x * arch.head_dim;

    for (int i = 0; i < arch.head_dim; i++) {
        ca_output[out_idx + i] = output[i];
    }
}

__global__ void reduce_affinity_kernel(
    const float* __restrict__ ca_output,
    float* __restrict__ affinity_reduced,
    int grid_size,
    ArchitectureParams arch
) {
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = grid_size * grid_size;

    if (cell_idx >= total_cells) return;

    float U = FlowLeniaOps::reduce_affinity_warp(
        ca_output, cell_idx, grid_size, arch.num_heads, arch.head_dim
    );

    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        affinity_reduced[cell_idx] = U;
    }
}

__global__ void compute_flow_field_kernel(
    const float* __restrict__ affinity_reduced,
    const float* __restrict__ ca_concentration,
    float* __restrict__ flow_field,
    int grid_size,
    float beta_A,
    float n,
    ArchitectureParams arch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int cell_idx = y * grid_size + x;

    float U_center = affinity_reduced[cell_idx];
    int x_E = min(x + 1, grid_size - 1);
    int y_N = min(y + 1, grid_size - 1);
    float U_E = affinity_reduced[y * grid_size + x_E];
    float U_N = affinity_reduced[y_N * grid_size + x];

    float A_sum_center = 0.0f;
    float A_sum_E = 0.0f;
    float A_sum_N = 0.0f;

    for (int c = 0; c < arch.channels; c++) {
        A_sum_center += ca_concentration[cell_idx * arch.channels + c];
        A_sum_E += ca_concentration[(y * grid_size + x_E) * arch.channels + c];
        A_sum_N += ca_concentration[(y_N * grid_size + x) * arch.channels + c];
    }

    float2 F = FlowLeniaOps::compute_flow_at(
        U_center, U_E, U_N,
        A_sum_center, A_sum_E, A_sum_N,
        beta_A, n
    );

    flow_field[cell_idx * 2 + 0] = F.x;
    flow_field[cell_idx * 2 + 1] = F.y;
}

__global__ void reintegration_redistribute_kernel(
    const float* __restrict__ ca_concentration,
    const float* __restrict__ flow_field,
    float* __restrict__ reintegration_buffer,
    int grid_size,
    float dt,
    float s,
    ArchitectureParams arch
) {
    int source_x = blockIdx.x;
    int source_y = blockIdx.y;

    if (source_x >= grid_size || source_y >= grid_size) return;

    int source_idx = source_y * grid_size + source_x;

    float Fx = flow_field[source_idx * 2 + 0];
    float Fy = flow_field[source_idx * 2 + 1];

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

            for (int c = threadIdx.x; c < arch.channels; c += blockDim.x) {
                float mass = ca_concentration[source_idx * arch.channels + c] * I_val;
                atomicAdd(&reintegration_buffer[target_idx * arch.channels + c], mass);
            }
        }
    }
}


__global__ void compute_effective_rank_from_latent_kernel(
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

    *effective_rank = sqrtf(variance + EPSILON) * latent_dim;
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

        if (current_loss > EPSILON) {
            local_improvement = fmaxf(0.0f, (current_loss - next_loss) / current_loss);
        }
    }

    float total = BlockReduce<BLOCK_SIZE>::sum(local_improvement);

    if (tid == 0) {
        *coherence = total / (history_length - 1);
    }
}

__global__ void init_multihead_ca_kernel(
    MultiHeadCAState* state,
    unsigned int seed,
    ArchitectureParams arch
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState_t rand_state;
    curand_init(seed, tid, 0, &rand_state);

    int perception_size = arch.num_heads * arch.channels * arch.hidden_dim;
    if (tid < perception_size) {
        float fan_in = (float)arch.channels;
        float fan_out = (float)arch.hidden_dim;
        float scale = sqrtf(2.0f / (fan_in + fan_out));
        float val = curand_normal(&rand_state) * scale;
        state->perception_weights[tid] = __float2half(val);
    }

    int interaction_size = arch.num_heads * arch.channels * arch.hidden_dim;
    if (tid < interaction_size) {
        float scale = sqrtf(2.0f / (float)arch.channels);
        float val = curand_normal(&rand_state) * scale;
        state->interaction_weights[tid] = __float2half(val);
    }

    int value_size = arch.num_heads * arch.hidden_dim * arch.channels;
    if (tid < value_size) {
        float scale = sqrtf(2.0f / (float)arch.hidden_dim);
        float val = curand_normal(&rand_state) * scale;
        state->value_weights[tid] = __float2half(val);
    }
}

#endif

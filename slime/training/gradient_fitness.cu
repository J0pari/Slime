
#ifndef GRADIENT_FITNESS_CU
#define GRADIENT_FITNESS_CU

#include "../config/config.cu"
#include "../learning/autodiff.cu"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void extract_head_gradient_magnitudes_kernel(
    ADTape* __restrict__ tape,
    int* __restrict__ param_start_indices,
    int* __restrict__ param_counts,
    float* __restrict__ gradient_magnitudes,
    int num_heads
) {
    int head_id = blockIdx.x;

    if (head_id >= num_heads) return;

    int start_idx = param_start_indices[head_id];
    int count = param_counts[head_id];

    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int param_idx = start_idx + i;
        if (param_idx < tape->current_value_idx) {
            float grad = tape->grad_buffer[param_idx];
            local_sum += grad * grad;
        }
    }

    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = WMMA_TILE_DIM; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    if (threadIdx.x == 0) {
        gradient_magnitudes[head_id] = sqrtf(local_sum);
    }
}

__global__ void compute_gradient_fitness_kernel(
    float* __restrict__ gradient_magnitudes,
    float* __restrict__ coherence_values,
    float* __restrict__ fitness_out,
    int num_heads,
    float gradient_weight,
    float coherence_weight
) {
    int head_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (head_id >= num_heads) return;

    float grad_mag = gradient_magnitudes[head_id];
    float coherence = coherence_values[head_id];

    __shared__ float mean_grad;
    __shared__ float std_grad;

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < num_heads; i++) {
            sum += gradient_magnitudes[i];
        }
        mean_grad = sum / num_heads;

        float var = 0.0f;
        for (int i = 0; i < num_heads; i++) {
            float diff = gradient_magnitudes[i] - mean_grad;
            var += diff * diff;
        }
        std_grad = sqrtf(var / num_heads + EPSILON);
    }
    __syncthreads();

    float grad_fitness = (grad_mag - mean_grad) / std_grad;

    fitness_out[head_id] = gradient_weight * fmaxf(0.0f, grad_fitness) +
                          coherence_weight * coherence;
}

__global__ void update_fitness_ema_kernel(
    float* __restrict__ current_fitness,
    float* __restrict__ fitness_ema,
    int num_heads,
    float alpha
) {
    int head_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (head_id >= num_heads) return;

    float curr = current_fitness[head_id];
    float prev_ema = fitness_ema[head_id];

    fitness_ema[head_id] = alpha * curr + (1.0f - alpha) * prev_ema;
}

__global__ void compute_relative_fitness_kernel(
    float* __restrict__ absolute_fitness,
    float* __restrict__ behavioral_coords,
    float* __restrict__ relative_fitness,
    int num_components,
    int behavioral_dim,
    int k_neighbors
) {
    int comp_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (comp_id >= num_components) return;

    float distances[5];
    int neighbor_ids[5];

    for (int i = 0; i < k_neighbors; i++) {
        distances[i] = 1e9f;
        neighbor_ids[i] = -1;
    }

    for (int other = 0; other < num_components; other++) {
        if (other == comp_id) continue;

        float dist_sq = 0.0f;
        for (int d = 0; d < behavioral_dim; d++) {
            float diff = behavioral_coords[comp_id * behavioral_dim + d] -
                        behavioral_coords[other * behavioral_dim + d];
            dist_sq += diff * diff;
        }

        float dist = sqrtf(dist_sq);
        for (int i = 0; i < k_neighbors; i++) {
            if (dist < distances[i]) {

                for (int j = k_neighbors - 1; j > i; j--) {
                    distances[j] = distances[j-1];
                    neighbor_ids[j] = neighbor_ids[j-1];
                }
                distances[i] = dist;
                neighbor_ids[i] = other;
                break;
            }
        }
    }

    float neighbor_mean = 0.0f;
    for (int i = 0; i < k_neighbors; i++) {
        if (neighbor_ids[i] >= 0) {
            neighbor_mean += absolute_fitness[neighbor_ids[i]];
        }
    }
    neighbor_mean /= k_neighbors;

    float neighbor_std = 0.0f;
    for (int i = 0; i < k_neighbors; i++) {
        if (neighbor_ids[i] >= 0) {
            float diff = absolute_fitness[neighbor_ids[i]] - neighbor_mean;
            neighbor_std += diff * diff;
        }
    }
    neighbor_std = sqrtf(neighbor_std / k_neighbors + EPSILON);

    float my_fitness = absolute_fitness[comp_id];
    relative_fitness[comp_id] = (my_fitness - neighbor_mean) / neighbor_std;
}

#endif

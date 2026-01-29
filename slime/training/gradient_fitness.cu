
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
    DEVICE_FATAL_IF(tape == nullptr, "extract_head_gradient_magnitudes: tape is null");
    DEVICE_FATAL_IF(tape->grad_buffer == nullptr, "extract_head_gradient_magnitudes: grad_buffer is null");
    DEVICE_FATAL_IF(param_start_indices == nullptr, "extract_head_gradient_magnitudes: param_start_indices is null");
    DEVICE_FATAL_IF(param_counts == nullptr, "extract_head_gradient_magnitudes: param_counts is null");
    DEVICE_FATAL_IF(gradient_magnitudes == nullptr, "extract_head_gradient_magnitudes: gradient_magnitudes is null");
    DEVICE_FATAL_IF(num_heads <= 0, "extract_head_gradient_magnitudes: num_heads must be positive");

    int head_id = blockIdx.x;

    if (head_id >= num_heads) return;

    int start_idx = param_start_indices[head_id];
    int count = param_counts[head_id];

    DEVICE_FATAL_IF(start_idx < 0, "extract_head_gradient_magnitudes: negative start_idx");
    DEVICE_FATAL_IF(count < 0, "extract_head_gradient_magnitudes: negative count");
    DEVICE_FATAL_IF(start_idx + count > tape->value_capacity, "extract_head_gradient_magnitudes: param range exceeds tape capacity");

    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int param_idx = start_idx + i;
        float grad = tape->grad_buffer[param_idx];
        DEVICE_FATAL_IF(isnan(grad), "extract_head_gradient_magnitudes: gradient is NaN");
        local_sum += grad * grad;
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
    DEVICE_FATAL_IF(gradient_magnitudes == nullptr, "compute_gradient_fitness: gradient_magnitudes is null");
    DEVICE_FATAL_IF(coherence_values == nullptr, "compute_gradient_fitness: coherence_values is null");
    DEVICE_FATAL_IF(fitness_out == nullptr, "compute_gradient_fitness: fitness_out is null");
    DEVICE_FATAL_IF(num_heads <= 0, "compute_gradient_fitness: num_heads must be positive");

    int head_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (head_id >= num_heads) return;

    float grad_mag = gradient_magnitudes[head_id];
    float coherence = coherence_values[head_id];
    DEVICE_FATAL_IF(isnan(grad_mag), "compute_gradient_fitness: gradient_magnitude is NaN");
    DEVICE_FATAL_IF(isnan(coherence), "compute_gradient_fitness: coherence is NaN");

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
        std_grad = sqrtf(var / num_heads);
    }
    __syncthreads();

    // z-score: when std=0 all values are identical, z-score is 0 by definition
    float grad_fitness = is_meaningful(std_grad, 1.0f) ? (grad_mag - mean_grad) / std_grad : 0.0f;

    fitness_out[head_id] = gradient_weight * fmaxf(0.0f, grad_fitness) +
                          coherence_weight * coherence;
}

__global__ void update_fitness_ema_kernel(
    float* __restrict__ current_fitness,
    float* __restrict__ fitness_ema,
    int num_heads,
    float alpha
) {
    DEVICE_FATAL_IF(current_fitness == nullptr, "update_fitness_ema: current_fitness is null");
    DEVICE_FATAL_IF(fitness_ema == nullptr, "update_fitness_ema: fitness_ema is null");
    DEVICE_FATAL_IF(num_heads <= 0, "update_fitness_ema: num_heads must be positive");
    DEVICE_FATAL_IF(alpha < 0.0f || alpha > 1.0f, "update_fitness_ema: alpha must be in [0,1]");

    int head_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (head_id >= num_heads) return;

    float curr = current_fitness[head_id];
    float prev_ema = fitness_ema[head_id];

    DEVICE_FATAL_IF(isnan(curr), "update_fitness_ema: current_fitness is NaN");
    DEVICE_FATAL_IF(isnan(prev_ema), "update_fitness_ema: prev_ema is NaN");

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
    DEVICE_FATAL_IF(absolute_fitness == nullptr, "compute_relative_fitness: absolute_fitness is null");
    DEVICE_FATAL_IF(behavioral_coords == nullptr, "compute_relative_fitness: behavioral_coords is null");
    DEVICE_FATAL_IF(relative_fitness == nullptr, "compute_relative_fitness: relative_fitness is null");
    DEVICE_FATAL_IF(num_components <= 0, "compute_relative_fitness: num_components must be positive");
    DEVICE_FATAL_IF(behavioral_dim <= 0, "compute_relative_fitness: behavioral_dim must be positive");
    DEVICE_FATAL_IF(k_neighbors <= 0 || k_neighbors > 5, "compute_relative_fitness: k_neighbors must be in [1,5]");

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

    float neighbor_var = 0.0f;
    for (int i = 0; i < k_neighbors; i++) {
        if (neighbor_ids[i] >= 0) {
            float diff = absolute_fitness[neighbor_ids[i]] - neighbor_mean;
            neighbor_var += diff * diff;
        }
    }
    float neighbor_std = sqrtf(neighbor_var / k_neighbors);

    float my_fitness = absolute_fitness[comp_id];
    // z-score: when std=0 all neighbors have identical fitness, z-score is 0
    relative_fitness[comp_id] = is_meaningful(neighbor_std, 1.0f) ? (my_fitness - neighbor_mean) / neighbor_std : 0.0f;
}

#endif

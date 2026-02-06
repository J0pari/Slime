
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
    DEVICE_FATAL_IF(count <= 0, "extract_head_gradient_magnitudes: non-positive count");
    DEVICE_FATAL_IF(start_idx + count > tape->value_capacity, "extract_head_gradient_magnitudes: param range exceeds tape capacity");

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        float grad = tape->grad_buffer[start_idx + i];
        DEVICE_FATAL_IF(isnan(grad), "extract_head_gradient_magnitudes: gradient is NaN");
        local_sum += grad * grad;
    }

    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = WMMA_TILE_DIM; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    if (threadIdx.x == 0) {
        gradient_magnitudes[head_id] = sqrtf(local_sum / (float)count);
    }
}

__global__ void compute_effective_rank_from_gradients_kernel(
    float* __restrict__ gradient_magnitudes,  // [num_heads] per-head RMSE
    float* __restrict__ effective_rank_out,   // [1] scalar output
    int num_heads,
    float renyi_order_q                       // genome-derived, typically 1.0 for Shannon
) {
    DEVICE_FATAL_IF(gradient_magnitudes == nullptr, "compute_effective_rank: gradient_magnitudes is null");
    DEVICE_FATAL_IF(effective_rank_out == nullptr, "compute_effective_rank: effective_rank_out is null");
    DEVICE_FATAL_IF(num_heads <= 0, "compute_effective_rank: num_heads must be positive");

    int tid = threadIdx.x;

    __shared__ float s_total_sq;
    float local_sq = 0.0f;
    for (int h = tid; h < num_heads; h += blockDim.x) {
        float g = gradient_magnitudes[h];
        local_sq += g * g;
    }

    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = WMMA_TILE_DIM; offset > 0; offset /= 2) {
        local_sq += __shfl_down_sync(mask, local_sq, offset);
    }
    if (tid == 0) s_total_sq = local_sq;
    __syncthreads();

    float total_sq = s_total_sq;

    DEVICE_FATAL_IF(total_sq < 1e-12f, "compute_effective_rank: zero gradient magnitude - no learning signal");

    __shared__ float s_entropy_sum;
    float local_entropy = 0.0f;

    bool use_shannon = fabsf(renyi_order_q - 1.0f) < 0.01f;

    for (int h = tid; h < num_heads; h += blockDim.x) {
        float g = gradient_magnitudes[h];
        float p = (g * g) / total_sq;  // Probability (normalized squared magnitude)

        if (p > 1e-12f) {  // Avoid log(0)
            if (use_shannon) {
                local_entropy -= p * logf(p);  // Shannon: -p log(p)
            } else {
                local_entropy += powf(p, renyi_order_q);  // Rényi: p^q
            }
        }
    }

    #pragma unroll
    for (int offset = WMMA_TILE_DIM; offset > 0; offset /= 2) {
        local_entropy += __shfl_down_sync(mask, local_entropy, offset);
    }
    if (tid == 0) s_entropy_sum = local_entropy;
    __syncthreads();

    if (tid == 0) {
        float entropy;
        if (use_shannon) {
            entropy = s_entropy_sum;  // Already computed as -Σ p log(p)
        } else {
            entropy = logf(s_entropy_sum) / (1.0f - renyi_order_q);
        }

        float eff_rank = expf(entropy);

        eff_rank = fmaxf(1.0f, fminf((float)num_heads, eff_rank));

        DEVICE_FATAL_IF(isnan(eff_rank), "compute_effective_rank: result is NaN");
        DEVICE_FATAL_IF(isinf(eff_rank), "compute_effective_rank: result is Inf");

        effective_rank_out[0] = eff_rank;
    }
}

__global__ void compute_multiplicative_fitness_kernel(
    float task_accuracy,          // From classification head output
    float generalization_gap,     // |train_accuracy - test_accuracy|
    float effective_rank,         // From compute_effective_rank_from_gradients_kernel
    float hardware_efficiency,    // From trace buffer aggregation
    float alpha,                  // task_exponent (genome-derived)
    float beta,                   // generalization_exponent (genome-derived)
    float gamma,                  // rank_exponent (genome-derived)
    float delta,                  // efficiency_exponent (genome-derived)
    float* __restrict__ fitness_out
) {
    if (threadIdx.x != 0) return;

    DEVICE_FATAL_IF(fitness_out == nullptr, "compute_multiplicative_fitness: fitness_out is null");
    DEVICE_FATAL_IF(task_accuracy < 0.0f || task_accuracy > 1.0f, "compute_multiplicative_fitness: task_accuracy out of range");
    DEVICE_FATAL_IF(generalization_gap < 0.0f || generalization_gap > 1.0f, "compute_multiplicative_fitness: gen_gap out of range");
    DEVICE_FATAL_IF(effective_rank < 1.0f, "compute_multiplicative_fitness: effective_rank below 1");
    DEVICE_FATAL_IF(hardware_efficiency <= 0.0f, "compute_multiplicative_fitness: hw_efficiency non-positive");

    float fitness = powf(task_accuracy + 1e-6f, alpha) *
                    powf(1.0f - generalization_gap + 1e-6f, beta) *
                    powf(effective_rank, gamma) *
                    powf(hardware_efficiency + 1e-6f, delta);

    DEVICE_FATAL_IF(isnan(fitness), "compute_multiplicative_fitness: fitness is NaN");
    DEVICE_FATAL_IF(isinf(fitness), "compute_multiplicative_fitness: fitness is Inf");

    fitness_out[0] = fitness;
}

__global__ void update_fitness_ema_kernel(
    float* __restrict__ current_fitness,
    float* __restrict__ fitness_ema,
    int num_entries,
    float alpha
) {
    DEVICE_FATAL_IF(current_fitness == nullptr, "update_fitness_ema: current_fitness is null");
    DEVICE_FATAL_IF(fitness_ema == nullptr, "update_fitness_ema: fitness_ema is null");
    DEVICE_FATAL_IF(num_entries <= 0, "update_fitness_ema: num_entries must be positive");
    DEVICE_FATAL_IF(alpha < 0.0f || alpha > 1.0f, "update_fitness_ema: alpha must be in [0,1]");

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_entries) return;

    float curr = current_fitness[idx];
    float prev_ema = fitness_ema[idx];

    DEVICE_FATAL_IF(isnan(curr), "update_fitness_ema: current_fitness is NaN");
    DEVICE_FATAL_IF(isnan(prev_ema), "update_fitness_ema: prev_ema is NaN");

    fitness_ema[idx] = alpha * curr + (1.0f - alpha) * prev_ema;
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
    DEVICE_FATAL_IF(num_components < 2, "compute_relative_fitness: need at least 2 components for relative fitness");
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
    int valid_neighbors = 0;
    for (int i = 0; i < k_neighbors; i++) {
        if (neighbor_ids[i] >= 0) {
            neighbor_mean += absolute_fitness[neighbor_ids[i]];
            valid_neighbors++;
        }
    }
    DEVICE_FATAL_IF(valid_neighbors <= 0, "compute_relative_fitness: no valid neighbors found");
    neighbor_mean /= valid_neighbors;

    float neighbor_var = 0.0f;
    for (int i = 0; i < k_neighbors; i++) {
        if (neighbor_ids[i] >= 0) {
            float diff = absolute_fitness[neighbor_ids[i]] - neighbor_mean;
            neighbor_var += diff * diff;
        }
    }
    float neighbor_std = sqrtf(neighbor_var / valid_neighbors);

    float my_fitness = absolute_fitness[comp_id];
    relative_fitness[comp_id] = is_meaningful(neighbor_std, 1.0f) ?
        (my_fitness - neighbor_mean) / neighbor_std : NAN;
}

#endif

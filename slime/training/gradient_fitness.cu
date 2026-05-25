
#ifndef GRADIENT_FITNESS_CU
#define GRADIENT_FITNESS_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include "../learning/autodiff.cu"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void extract_head_gradient_magnitudes_device(ADTape* tape, CAParameterMap* param_map,
                                                        float* gradient_magnitudes, int num_heads) {
    DEVICE_FATAL_IF(tape == nullptr, "extract_head_gradient_magnitudes: tape is null");
    DEVICE_FATAL_IF(tape->grad_buffer == nullptr, "extract_head_gradient_magnitudes: grad_buffer is null");
    DEVICE_FATAL_IF(param_map == nullptr, "extract_head_gradient_magnitudes: param_map is null");
    DEVICE_FATAL_IF(gradient_magnitudes == nullptr, "extract_head_gradient_magnitudes: gradient_magnitudes is null");
    DEVICE_FATAL_IF(num_heads <= 0, "extract_head_gradient_magnitudes: num_heads must be positive");

    int head_id = blockIdx.x;
    if (head_id < num_heads) {
        int start_idx = param_map->head_param_offsets[head_id];
        int count = param_map->head_param_counts[head_id];

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
}

__device__ void compute_effective_rank_from_gradients_device(Organism* organism) {
    float* gradient_magnitudes = organism->gf_gradient_magnitudes;
    float* effective_rank_out = organism->gf_effective_rank_out;
    int num_heads = organism->gf_num_heads;
    float renyi_order_q = organism->gf_renyi_order_q;
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
    cg::this_grid().sync();

    float total_sq = s_total_sq;

    DEVICE_FATAL_IF(total_sq < 1e-12f, "compute_effective_rank: total_sq < 1e-12");

    __shared__ float s_entropy_sum;
    float local_entropy = 0.0f;

    bool use_shannon = fabsf(renyi_order_q - 1.0f) < 0.01f;

    for (int h = tid; h < num_heads; h += blockDim.x) {
        float g = gradient_magnitudes[h];
        float p = (g * g) / total_sq;

        if (p > 1e-12f) {
            if (use_shannon) {
                local_entropy -= p * logf(p);
            } else {
                local_entropy += powf(p, renyi_order_q);
            }
        }
    }

    #pragma unroll
    for (int offset = WMMA_TILE_DIM; offset > 0; offset /= 2) {
        local_entropy += __shfl_down_sync(mask, local_entropy, offset);
    }
    if (tid == 0) s_entropy_sum = local_entropy;
    cg::this_grid().sync();

    if (tid == 0) {
        float entropy;
        if (use_shannon) {
            entropy = s_entropy_sum;
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

__device__ void compute_multiplicative_fitness_device(Organism* organism) {
    float task_accuracy = organism->gf_task_accuracy;
    float generalization_gap = organism->gf_generalization_gap;
    float effective_rank = organism->gf_effective_rank;
    float hardware_efficiency = organism->gf_hardware_efficiency;
    float alpha = organism->gf_alpha;
    float beta = organism->gf_beta;
    float gamma = organism->gf_gamma;
    float delta = organism->gf_delta;
    float* fitness_out = organism->gf_fitness_out;
    if (threadIdx.x == 0) {
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
}

__device__ void update_fitness_ema_device(Organism* organism) {
    float* current_fitness = organism->gf_current_fitness;
    float* fitness_ema = organism->gf_fitness_ema;
    int num_entries = organism->gf_num_entries;
    float alpha = organism->gf_ema_alpha;
    DEVICE_FATAL_IF(current_fitness == nullptr, "update_fitness_ema: current_fitness is null");
    DEVICE_FATAL_IF(fitness_ema == nullptr, "update_fitness_ema: fitness_ema is null");
    DEVICE_FATAL_IF(num_entries <= 0, "update_fitness_ema: num_entries must be positive");
    DEVICE_FATAL_IF(alpha < 0.0f || alpha > 1.0f, "update_fitness_ema: alpha must be in [0,1]");

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries) {
        float curr = current_fitness[idx];
        float prev_ema = fitness_ema[idx];

        DEVICE_FATAL_IF(isnan(curr), "update_fitness_ema: current_fitness is NaN");
        DEVICE_FATAL_IF(isnan(prev_ema), "update_fitness_ema: prev_ema is NaN");

        fitness_ema[idx] = alpha * curr + (1.0f - alpha) * prev_ema;
    }
}

__device__ void compute_relative_fitness_device(Organism* organism) {
    float* absolute_fitness = organism->gf_absolute_fitness;
    float* behavioral_coords = organism->gf_behavioral_coords;
    float* relative_fitness = organism->gf_relative_fitness;
    int num_components = organism->gf_num_components;
    int behavioral_dim = organism->gf_behavioral_dim;
    int k_neighbors = organism->gf_k_neighbors;
    DEVICE_FATAL_IF(absolute_fitness == nullptr, "compute_relative_fitness: absolute_fitness is null");
    DEVICE_FATAL_IF(behavioral_coords == nullptr, "compute_relative_fitness: behavioral_coords is null");
    DEVICE_FATAL_IF(relative_fitness == nullptr, "compute_relative_fitness: relative_fitness is null");
    DEVICE_FATAL_IF(num_components < 2, "compute_relative_fitness: need at least 2 components for relative fitness");
    DEVICE_FATAL_IF(behavioral_dim <= 0, "compute_relative_fitness: behavioral_dim must be positive");
    DEVICE_FATAL_IF(k_neighbors <= 0 || k_neighbors > 5, "compute_relative_fitness: k_neighbors must be in [1,5]");

    int comp_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (comp_id < num_components) {
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
}

__device__ __forceinline__ uint64_t compute_fitness_input_hash(
    float task_accuracy, float gen_gap_term, float effective_rank, float hardware_efficiency,
    float task_exp, float gen_exp, float rank_exp, float eff_exp
) {
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    hash ^= __float_as_uint(task_accuracy) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= __float_as_uint(gen_gap_term) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= __float_as_uint(effective_rank) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= __float_as_uint(hardware_efficiency) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= __float_as_uint(task_exp) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= __float_as_uint(gen_exp) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= __float_as_uint(rank_exp) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= __float_as_uint(eff_exp) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
}

__device__ void compute_fitness(PoolEntry* entry, int generation) {
    float task_accuracy = entry->task_accuracy.value;
    float gen_gap_term = 1.0f - entry->generalization_gap.value;
    float effective_rank = entry->effective_rank.value;
    float hardware_efficiency = entry->hardware_efficiency.value;

    DEVICE_FATAL_IF(isnan(task_accuracy), "compute_fitness: task_accuracy is NaN");
    DEVICE_FATAL_IF(isnan(entry->generalization_gap.value), "compute_fitness: generalization_gap is NaN");
    DEVICE_FATAL_IF(isnan(hardware_efficiency), "compute_fitness: hardware_efficiency is NaN");
    DEVICE_FATAL_IF(isnan(effective_rank), "compute_fitness: effective_rank is NaN");
    DEVICE_FATAL_IF(gen_gap_term <= 0.0f, "compute_fitness: gen_gap_term non-positive");
    DEVICE_FATAL_IF(task_accuracy <= 0.0f, "compute_fitness: task_accuracy non-positive");
    DEVICE_FATAL_IF(effective_rank <= 0.0f, "compute_fitness: effective_rank non-positive");
    DEVICE_FATAL_IF(hardware_efficiency <= 0.0f, "compute_fitness: hardware_efficiency non-positive");

    float fitness = powf(task_accuracy, entry->fitness_task_exponent)
                  * powf(gen_gap_term, entry->fitness_gen_exponent)
                  * powf(effective_rank, entry->fitness_rank_exponent)
                  * powf(hardware_efficiency, entry->fitness_efficiency_exponent);

    uint64_t input_hash = compute_fitness_input_hash(
        task_accuracy, gen_gap_term, effective_rank, hardware_efficiency,
        entry->fitness_task_exponent, entry->fitness_gen_exponent,
        entry->fitness_rank_exponent, entry->fitness_efficiency_exponent
    );

    measured_value_set_computed(&entry->fitness, fitness, generation, input_hash);
}

__device__ void compute_fitness_from_diresa_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    int generation = organism->generation;

    int entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry_idx < pool->capacity && pool->alive_flags[entry_idx]) {
        PoolEntry* entry = &pool->entries[entry_idx];

        DEVICE_FATAL_IF(entry->task_accuracy.state != ComputeState::COMPUTED, "compute_fitness: task_accuracy not COMPUTED for alive entry");
        DEVICE_FATAL_IF(entry->hardware_efficiency.state != ComputeState::COMPUTED, "compute_fitness: hardware_efficiency not COMPUTED for alive entry");
        DEVICE_FATAL_IF(entry->effective_rank.state != ComputeState::COMPUTED, "compute_fitness: effective_rank not COMPUTED for alive entry");
        DEVICE_FATAL_IF(entry->generalization_gap.state != ComputeState::COMPUTED, "compute_fitness: generalization_gap not COMPUTED for alive entry");

        compute_fitness(entry, generation);
        pool->fitness_values[entry_idx] = entry->fitness.value;
    }
}

#endif

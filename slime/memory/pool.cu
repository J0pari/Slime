
#ifndef POOL_CU
#define POOL_CU
#include "../config/config.cu"
#include "../core/organism.cu"
#include "../utils/genome_params.cuh"
#include "../utils/cuda_primitives.cuh"
#include "../memory/genome_ops.cuh"
#include "../memory/archive.cu"
#include "../compression/delta.cu"
#include "../memory/parallel_compaction.cu"
#include "../core/ca_state.cuh"
#include "pool_types.cuh"
#include <cuda_runtime.h>

__device__ __forceinline__ Architecture get_arch_from_pool(Organism* organism, int idx) {
    ComponentPool* pool = organism->pool;
    Architecture arch;
    arch.num_heads = pool->entries[idx].num_heads;
    arch.channels = pool->entries[idx].channels;
    arch.hidden_dim = pool->entries[idx].hidden_dim;
    arch.head_dim = pool->entries[idx].head_dim;
    arch.grid_size = pool->entries[idx].grid_size;
    return arch;
}


#include "../learning/diresa.cu"

__device__ void init_rng_states_device(Organism* organism) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = organism->pool->capacity;
    curandState* states = organism->rng_states;
    unsigned long seed = organism->init_seed;
    if (idx < count) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}


__device__ void mark_alive_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    int* flags = organism->pool_compaction_flags;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pool->capacity) {
        flags[idx] = pool->alive_flags[idx] ? 1 : 0;
    }
}




__device__ void scatter_alive_indices_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    int* flags = organism->pool_compaction_flags;
    int* scan_results = organism->pool_compaction_scan;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pool->capacity && flags[idx]) {
        pool->alive_indices[scan_results[idx]] = idx;
    }
}


__device__ void finalize_alive_count_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    int* flags = organism->pool_compaction_flags;
    int* scan_results = organism->pool_compaction_scan;
    int capacity = pool->capacity;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // All threads can compute the final value (it's a simple read), one writes
    int last_idx = capacity - 1;
    int final_count = scan_results[last_idx] + flags[last_idx];
    if (global_tid == 0) {
        pool->alive_indices_count = final_count;
    }
}

__device__ __forceinline__ void derive_architecture(const float* genome, PoolEntry* entry) {
    // Architecture is environmental - all entries use shared structure
    entry->num_heads = NUM_HEADS;
    entry->head_dim = HEAD_DIM;
    entry->channels = CHANNELS;
    entry->grid_size = GRID_SIZE;
    entry->hidden_dim = HIDDEN_DIM;
}

__device__ __forceinline__ void derive_diresa(const float* genome, PoolEntry* entry) {
    float replicas_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::diresa_num_replicas, NORMALIZED_MIN, NORMALIZED_MAX);
    float hidden1_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::diresa_hidden1, NORMALIZED_MIN, NORMALIZED_MAX);
    float hidden2_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::diresa_hidden2, NORMALIZED_MIN, NORMALIZED_MAX);
    float batch_size_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::diresa_batch_size, NORMALIZED_MIN, NORMALIZED_MAX);
    float anneal_step_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::diresa_anneal_step, NORMALIZED_MIN, NORMALIZED_MAX);
    float cov_target_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::diresa_cov_target, NORMALIZED_MIN, NORMALIZED_MAX);
    float dist_weight_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::dist_weight, NORMALIZED_MIN, NORMALIZED_MAX);
    float recon_weight_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::recon_weight, NORMALIZED_MIN, NORMALIZED_MAX);
    float distance_exponent_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::distance_exponent, NORMALIZED_MIN, NORMALIZED_MAX);
    float quality_weight_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::quality_weight, NORMALIZED_MIN, NORMALIZED_MAX);

    entry->num_tempering_replicas = (int)fmaxf((float)NUM_TEMPERING_REPLICAS_MIN, fminf((float)NUM_TEMPERING_REPLICAS_MAX, NUM_TEMPERING_REPLICAS_MIN + replicas_norm * (NUM_TEMPERING_REPLICAS_MAX - NUM_TEMPERING_REPLICAS_MIN)));
    entry->diresa_hidden1 = (int)fmaxf((float)DIRESA_HIDDEN1_MIN, fminf((float)DIRESA_HIDDEN1_MAX, DIRESA_HIDDEN1_MIN + hidden1_norm * (DIRESA_HIDDEN1_MAX - DIRESA_HIDDEN1_MIN)));
    entry->diresa_hidden2 = (int)fmaxf((float)DIRESA_HIDDEN2_MIN, fminf((float)DIRESA_HIDDEN2_MAX, DIRESA_HIDDEN2_MIN + hidden2_norm * (DIRESA_HIDDEN2_MAX - DIRESA_HIDDEN2_MIN)));
    entry->diresa_batch_size = (int)fmaxf((float)DIRESA_BATCH_SIZE_MIN, fminf((float)DIRESA_BATCH_SIZE_MAX, DIRESA_BATCH_SIZE_MIN + batch_size_norm * (DIRESA_BATCH_SIZE_MAX - DIRESA_BATCH_SIZE_MIN)));
    entry->anneal_step = fmaxf(ANNEAL_STEP_MIN, fminf(ANNEAL_STEP_MAX, ANNEAL_STEP_MIN + anneal_step_norm * (ANNEAL_STEP_MAX - ANNEAL_STEP_MIN)));
    entry->cov_target = fmaxf(COV_TARGET_MIN, fminf(COV_TARGET_MAX, COV_TARGET_MIN + cov_target_norm * (COV_TARGET_MAX - COV_TARGET_MIN)));
    entry->dist_weight = fmaxf(DIST_WEIGHT_MIN, fminf(DIST_WEIGHT_MAX, DIST_WEIGHT_MIN + dist_weight_norm * (DIST_WEIGHT_MAX - DIST_WEIGHT_MIN)));
    entry->recon_weight = fmaxf(RECON_WEIGHT_MIN, fminf(RECON_WEIGHT_MAX, RECON_WEIGHT_MIN + recon_weight_norm * (RECON_WEIGHT_MAX - RECON_WEIGHT_MIN)));
    entry->distance_exponent = fmaxf(DIRESA_DISTANCE_EXPONENT_MIN, fminf(DIRESA_DISTANCE_EXPONENT_MAX, DIRESA_DISTANCE_EXPONENT_MIN + distance_exponent_norm * (DIRESA_DISTANCE_EXPONENT_MAX - DIRESA_DISTANCE_EXPONENT_MIN)));
    entry->quality_weight = fmaxf(DIRESA_QUALITY_WEIGHT_MIN, fminf(DIRESA_QUALITY_WEIGHT_MAX, DIRESA_QUALITY_WEIGHT_MIN + quality_weight_norm * (DIRESA_QUALITY_WEIGHT_MAX - DIRESA_QUALITY_WEIGHT_MIN)));
}

__device__ __forceinline__ void derive_fitness_exponents(const float* genome, PoolEntry* entry) {
    float rank_exp_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::fitness_rank_exponent, NORMALIZED_MIN, NORMALIZED_MAX);
    float coh_exp_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::fitness_coherence_exponent, NORMALIZED_MIN, NORMALIZED_MAX);
    float coupling_exp_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::fitness_coupling_exponent, NORMALIZED_MIN, NORMALIZED_MAX);
    float task_exp_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::fitness_task_exponent, NORMALIZED_MIN, NORMALIZED_MAX);
    float gen_exp_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::fitness_gen_exponent, NORMALIZED_MIN, NORMALIZED_MAX);
    float eff_exp_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::fitness_efficiency_exponent, NORMALIZED_MIN, NORMALIZED_MAX);
    float baldwin_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::baldwin_sensitivity, NORMALIZED_MIN, NORMALIZED_MAX);
    float window_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::coherence_window_size, NORMALIZED_MIN, NORMALIZED_MAX);
    float renyi_q_norm = genome_to_bootstrap_param(genome, entry->gradients, GenomeParamTable::renyi_q, NORMALIZED_MIN, NORMALIZED_MAX);

    entry->fitness_rank_exponent = FITNESS_RANK_EXPONENT_MIN + rank_exp_norm * (FITNESS_RANK_EXPONENT_MAX - FITNESS_RANK_EXPONENT_MIN);
    entry->fitness_coherence_exponent = FITNESS_COHERENCE_EXPONENT_MIN + coh_exp_norm * (FITNESS_COHERENCE_EXPONENT_MAX - FITNESS_COHERENCE_EXPONENT_MIN);
    entry->fitness_coupling_exponent = FITNESS_COUPLING_EXPONENT_MIN + coupling_exp_norm * (FITNESS_COUPLING_EXPONENT_MAX - FITNESS_COUPLING_EXPONENT_MIN);
    entry->fitness_task_exponent = FITNESS_TASK_EXPONENT_MIN + task_exp_norm * (FITNESS_TASK_EXPONENT_MAX - FITNESS_TASK_EXPONENT_MIN);
    entry->fitness_gen_exponent = FITNESS_GEN_EXPONENT_MIN + gen_exp_norm * (FITNESS_GEN_EXPONENT_MAX - FITNESS_GEN_EXPONENT_MIN);
    entry->fitness_efficiency_exponent = FITNESS_EFFICIENCY_EXPONENT_MIN + eff_exp_norm * (FITNESS_EFFICIENCY_EXPONENT_MAX - FITNESS_EFFICIENCY_EXPONENT_MIN);
    entry->baldwin_sensitivity = BALDWIN_SENSITIVITY_MIN + baldwin_norm * (BALDWIN_SENSITIVITY_MAX - BALDWIN_SENSITIVITY_MIN);
    entry->coherence_window_size = COHERENCE_WINDOW_SIZE_MIN + (int)(window_norm * (COHERENCE_WINDOW_SIZE_MAX - COHERENCE_WINDOW_SIZE_MIN));
    entry->renyi_q = RANK_RENYI_ORDER_MIN + renyi_q_norm * (RANK_RENYI_ORDER_MAX - RANK_RENYI_ORDER_MIN);
}

__device__ float prng_next(PRNGState* state) {
    uint64_t x = state->s0;
    uint64_t y = state->s1;
    uint64_t result = x + y;
    y ^= x;
    state->s0 = ((x << XORSHIFT128_ROTL_A) | (x >> (XORSHIFT_STATE_BITS - XORSHIFT128_ROTL_A))) ^ y ^ (y << XORSHIFT128_SHIFT_C);
    state->s1 = (y << XORSHIFT128_ROTL_B) | (y >> (XORSHIFT_STATE_BITS - XORSHIFT128_ROTL_B));
    return (double)result / XORSHIFT_NORMALIZATION_SCALE;
}

__device__ float prng_levy_stable(PRNGState* state, float alpha, float scale) {
    float U = prng_next(state);
    float V = prng_next(state);
    float W = prng_next(state);

    float phi = (TAU * 0.25f) * (2.0f * V - 1.0f);
    float alpha_phi = alpha * phi;
    float one_minus_alpha_phi = (1.0f - alpha) * phi;

    float levy_num = sinf(alpha_phi);
    float cos_phi = cosf(phi);
    DEVICE_FATAL_IF(cos_phi <= 0.0f, "levy_stable: cos_phi <= 0 indicates PRNG corruption");
    float levy_denom = powf(cos_phi, (1.0f / alpha));

    DEVICE_FATAL_IF(W <= 0.0f || W >= 1.0f, "levy_stable: W out of (0,1) indicates PRNG corruption");
    float log_w = -logf(W);
    DEVICE_FATAL_IF(log_w <= 0.0f, "levy_stable: log_w <= 0 indicates PRNG corruption");
    float cos_one_minus_alpha_phi = cosf(one_minus_alpha_phi);
    DEVICE_FATAL_IF(cos_one_minus_alpha_phi <= 0.0f, "levy_stable: cos_one_minus_alpha_phi <= 0 indicates PRNG corruption");
    float levy_factor = powf(cos_one_minus_alpha_phi / log_w, ((1.0f - alpha) / alpha));

    return (levy_num / levy_denom * levy_factor) * scale;
}

__device__ void spawn_component_device(
    Organism* organism,
    int parent_id,
    float mutation_rate,
    float* workspace_parent_genome,
    float* workspace_child_genome,
    float* workspace_parent_temp
) {
    ComponentPool* pool = organism->pool;
    GPUElite* archive = organism->archive;
    int archive_size = organism->archive_size;
    int generation = organism->generation;
    DIRESAWeights* diresa_genome_weights = organism->diresa_genome_weights;
    int slot_idx = -1;
    int new_id;
    int parent_archive_idx;
    float inheritance_factor;
    PRNGState rng;
    PoolInitParams params;
    int use_latent;
    bool should_spawn = true;

    // Check capacity
    int old_count = atomicAdd((int*)&pool->active_count, 1);
    if (old_count >= pool->capacity) {
        atomicSub((int*)&pool->active_count, 1);
        should_spawn = false;
    }

    // Find empty slot
    if (should_spawn) {
        for (int i = 0; i < pool->capacity; i++) {
            if (pool->entries[i].id == INT_MAX) {
                int old_id = atomicCAS(&pool->entries[i].id, INT_MAX, -2);
                if (old_id == INT_MAX) {
                    slot_idx = i;
                    break;
                }
            }
        }

        if (slot_idx < 0) {
            atomicSub((int*)&pool->active_count, 1);
            should_spawn = false;
        }
    }

    if (should_spawn) {
        new_id = atomicAdd((int*)&pool->total_spawned, 1);
        int i = slot_idx;

        pool->entries[i].id = new_id;
        pool->entries[i].age = 0;
        pool->alive_flags[i] = true;

        DEVICE_FATAL_IF(parent_id < 0 || parent_id >= pool->capacity, "spawn_component_device: invalid parent_id");
        pool->entries[i].parent_idx = parent_id;

        float* parent_genome = workspace_parent_genome;
        float* child_genome = workspace_child_genome;
        float* reference_genome = workspace_parent_temp;
        PoolEntry* parent = &pool->entries[parent_id];

        int parent_self_archived = hash_table_lookup(
            archive->hash_table_keys,
            archive->hash_table_values,
            parent->genome_hash
        );

        uint64_t reference_hash;
        int reference_archive_idx;
        if (parent_self_archived >= 0) {
            reference_hash = parent->genome_hash;
            reference_archive_idx = parent_self_archived;
        } else {
            reference_hash = parent->parent_hash;
            reference_archive_idx = hash_table_lookup(
                archive->hash_table_keys,
                archive->hash_table_values,
                parent->parent_hash
            );
        }

        parent_archive_idx = reference_archive_idx;
        use_latent = (reference_archive_idx >= 0 && archive->latent_genome != nullptr) ? 1 : 0;
        rng.s0 = new_id * XORSHIFT_GOLDEN_RATIO_A;
        rng.s1 = parent_id * XORSHIFT_GOLDEN_RATIO_B;

        DEVICE_FATAL_IF(reference_archive_idx < 0, "spawn: reference not in archive");
        diresa_decode(&archive->latent_genome[reference_archive_idx * GENOME_LATENT_DIM_MAX], reference_genome, diresa_genome_weights);

        reconstruct_genome_from_archive(parent->parent_hash, archive, archive_size,
            parent->delta_indices, parent->delta_values, parent->num_deltas,
            parent->max_deltas, parent_genome, GENOME_SIZE, workspace_parent_temp, diresa_genome_weights);

        params.derive_from_genome(parent_genome, parent->gradients);

        if (use_latent) {
            float* ref_latent = &archive->latent_genome[reference_archive_idx * GENOME_LATENT_DIM_MAX];
            float* mutated_latent_temp = workspace_parent_temp;
            for (int j = 0; j < GENOME_LATENT_DIM_MAX; j++) {
                float mutated_latent = ref_latent[j];
                PRNGState local_rng = rng;
                local_rng.s0 ^= j;
                if (prng_next(&local_rng) < mutation_rate) {
                    mutated_latent += prng_levy_stable(&local_rng, params.mutation_levy_alpha, params.mutation_scale);
                    mutated_latent = tanhf(mutated_latent);
                }
                mutated_latent_temp[j] = mutated_latent;
            }
            diresa_decode(mutated_latent_temp, child_genome, diresa_genome_weights);
        } else {
            for (int j = 0; j < GENOME_SIZE; j++) {
                float val = reference_genome[j];
                PRNGState local_rng = rng;
                local_rng.s0 ^= j;
                if (prng_next(&local_rng) < mutation_rate) {
                    val += prng_levy_stable(&local_rng, params.mutation_levy_alpha, params.mutation_scale);
                    val = tanhf(val);
                }
                child_genome[j] = val;
            }
        }

        pool->entries[i].parent_hash = reference_hash;

        float inherit_center = genome_to_bootstrap_param(parent_genome, parent->gradients, GenomeParamTable::fitness_inherit_center,
            LIFECYCLE_FITNESS_INHERIT_CENTER_MIN, LIFECYCLE_FITNESS_INHERIT_CENTER_MAX);
        float inherit_steepness = genome_to_bootstrap_param(parent_genome, parent->gradients, GenomeParamTable::fitness_inherit_steepness,
            LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MIN, LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MAX);
        inheritance_factor = activation_sigmoid(inherit_steepness * (parent->fitness.value - inherit_center));

        for (int j = 0; j < GENOME_SIZE; j++) {
            pool->entries[i].gradients[j] = parent->gradients[j] * inheritance_factor;
        }

        pool->entries[i].genome_hash = gpu_sha256(child_genome, GENOME_SIZE);
        pool->entries[i].num_deltas = 0;

        compute_genome_deltas(
            child_genome,
            reference_genome,
            pool->entries[i].delta_indices,
            pool->entries[i].delta_values,
            &pool->entries[i].num_deltas,
            pool->entries[i].max_deltas,
            pool->entries[i].genome_hash
        );

        PoolInitParams init_params;
        init_params.derive_from_genome(child_genome, pool->entries[i].gradients);
        measured_value_set_computed(&pool->entries[i].hunger, init_params.initial_hunger, generation, pool->entries[i].genome_hash);
        derive_architecture(child_genome, &pool->entries[i]);
        derive_diresa(child_genome, &pool->entries[i]);
        derive_fitness_exponents(child_genome, &pool->entries[i]);
        measured_value_set_uncomputed(&pool->entries[i].fitness);
        pool->fitness_values[i] = NAN;
        measured_value_set_uncomputed(&pool->entries[i].coherence);
        measured_value_set_uncomputed(&pool->entries[i].task_accuracy);
        measured_value_set_uncomputed(&pool->entries[i].generalization_gap);
        measured_value_set_uncomputed(&pool->entries[i].hardware_efficiency);
        measured_value_set_uncomputed(&pool->entries[i].effective_rank);
        pool->entries[i].active_warps = 0;
        pool->entries[i].divergent_branches = 0;
        pool->entries[i].total_branches = 0;
        pool->entries[i].global_loads = 0;
        pool->entries[i].global_stores = 0;
        pool->entries[i].l2_transactions = 0;
        pool->entries[i].dram_transactions = 0;
        pool->entries[i].inst_executed = 0;
        pool->entries[i].inst_issued = 0;
        pool->entries[i].cycles_elapsed = 0;
        pool->entries[i].tensor_core_cycles = 0;
    }
}

__device__ void cull_weak_device(Organism* organism, float threshold) {
    ComponentPool* pool = organism->pool;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity) {
        if (pool->alive_flags[idx] && pool->fitness_values[idx] < threshold) {
            pool->entries[idx].phase = LifecyclePhase::DEAD;
            pool->alive_flags[idx] = false;
            Atomics::decrement_int(pool->active_count);
            Atomics::increment_int(pool->total_culled);
        }
    }
}

__device__ void cull_hungry_device(Organism* organism, float hunger_threshold) {
    ComponentPool* pool = organism->pool;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity) {
        if (pool->alive_flags[idx] && pool->entries[idx].hunger.value > hunger_threshold) {
            pool->entries[idx].phase = LifecyclePhase::DEAD;
            pool->alive_flags[idx] = false;
            Atomics::decrement_int(pool->active_count);
            Atomics::increment_int(pool->total_culled);
        }
    }
}

__device__ void age_components_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity && pool->alive_flags[idx]) {
        pool->entries[idx].age++;
    }
}

__device__ void update_hunger_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    int generation = organism->generation;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity && pool->alive_flags[idx]) {
        float hunger_val = fmaxf(0.01f, 1.0f - pool->entries[idx].coherence.value);
        measured_value_set_computed(&pool->entries[idx].hunger, hunger_val, generation, pool->entries[idx].genome_hash);
    }
}

__device__ void sort_by_fitness_device(Organism* organism, int stage, int step) {
    ComponentPool* pool = organism->pool;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_distance = 1 << step;
    int block_width = pair_distance << 1;
    int block_id = tid / pair_distance;

    int left_id = block_id * block_width + (tid % pair_distance);
    int right_id = left_id + pair_distance;

    if (right_id < pool->capacity) {
        PoolEntry left = pool->entries[left_id];
        PoolEntry right = pool->entries[right_id];

        bool ascending = ((block_id >> stage) & 1) == 0;
        bool swap = ascending ? (left.fitness.value > right.fitness.value) :
                               (left.fitness.value < right.fitness.value);

        if (swap) {
            pool->entries[left_id] = right;
            pool->entries[right_id] = left;
        }
    }
}

__device__ void select_top_k_device(Organism* organism, int* selected_indices, int k) {
    ComponentPool* pool = organism->pool;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < k && tid < pool->capacity) {
        if (pool->alive_flags[tid]) {
            selected_indices[tid] = tid;
        } else {
            selected_indices[tid] = -1;
        }
    }
}

__device__ __noinline__ float compute_genome_distance(Organism* organism, PoolEntry* entry1, PoolEntry* entry2) {
    GPUElite* archive = organism->archive;

    int idx1 = hash_table_lookup(
        archive->hash_table_keys,
        archive->hash_table_values,
        entry1->parent_hash
    );
    int idx2 = hash_table_lookup(
        archive->hash_table_keys,
        archive->hash_table_values,
        entry2->parent_hash
    );

    DEVICE_FATAL_IF(idx1 < 0, "compute_genome_distance: entry1->parent_hash not found in archive");
    DEVICE_FATAL_IF(idx2 < 0, "compute_genome_distance: entry2->parent_hash not found in archive");

    const float* latent1 = &archive->latent_genome[idx1 * GENOME_LATENT_DIM_MAX];
    const float* latent2 = &archive->latent_genome[idx2 * GENOME_LATENT_DIM_MAX];
    float distance_squared = DIRESAOps::compute_latent_distance_sq(latent1, latent2, GENOME_LATENT_DIM_MAX);

    return sqrtf(distance_squared / GENOME_LATENT_DIM_MAX);
}

__device__ void diversity_selection_device(Organism* organism, int* selected_indices, int num_select) {
    ComponentPool* pool = organism->pool;
    __shared__ float distances[BLOCK_SIZE];
    __shared__ int indices[BLOCK_SIZE];
    __shared__ float warp_max[BLOCK_SIZE / 32];
    __shared__ int warp_max_idx[BLOCK_SIZE / 32];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int capacity = pool->capacity;

    float my_distance = -1.0f;
    
    if (idx < capacity && pool->alive_flags[idx]) {
        float sum_dist = 0.0f;
        int count = 0;
        for (int i = tid; i < capacity; i += blockDim.x) {
            if (i != idx && pool->alive_flags[i]) {
                float dist = compute_genome_distance(organism, &pool->entries[idx], &pool->entries[i]);
                sum_dist += dist;
                count++;
            }
        }
        sum_dist = warp_reduce_sum(sum_dist);
        count = (int)warp_reduce_sum((float)count);
        if (lane == 0 && count > 0) {
            my_distance = sum_dist / count;
        }
        my_distance = __shfl_sync(0xffffffff, my_distance, 0);
    }
    distances[tid] = my_distance;
    indices[tid] = idx;
    cg::this_grid().sync();

    for (int sel = 0; sel < num_select; sel++) {
        float val = distances[tid];
        int my_idx = indices[tid];
        float max_val = val;
        int max_idx = my_idx;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }

        if (lane == 0) {
            warp_max[warp_id] = max_val;
            warp_max_idx[warp_id] = max_idx;
        }
        cg::this_grid().sync();

        if (tid < (BLOCK_SIZE / 32)) {
            max_val = warp_max[tid];
            max_idx = warp_max_idx[tid];
            unsigned active = __activemask();
            for (int offset = (BLOCK_SIZE / 64); offset > 0; offset >>= 1) {
                float other_val = __shfl_down_sync(active, max_val, offset);
                int other_idx = __shfl_down_sync(active, max_idx, offset);
                if (other_val > max_val) {
                    max_val = other_val;
                    max_idx = other_idx;
                }
            }
        }

        if (tid == 0 && max_idx >= 0 && max_val >= 0.0f) {
            selected_indices[sel] = max_idx;
        }
        cg::this_grid().sync();

        if (tid == 0) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                if (indices[j] == max_idx) {
                    distances[j] = -1.0f;
                    break;
                }
            }
        }
        cg::this_grid().sync();
    }
}

__device__ void init_pool_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    int capacity = pool->capacity;
    uint16_t* delta_indices_buffer = organism->delta_indices_buffer;
    float* delta_values_buffer = organism->delta_values_buffer;
    float* gradients_buffer = organism->gradients_buffer;
    int generation = organism->generation;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Count threads working on pool init
    __shared__ int active_inits;
    __shared__ int alive_inits;
    if (threadIdx.x == 0) { active_inits = 0; alive_inits = 0; }
    cg::this_grid().sync();

    if (idx < capacity) {
        atomicAdd(&active_inits, 1);

        PRNGState rng;
        rng.s0 = idx * XORSHIFT_GOLDEN_RATIO_A;
        rng.s1 = idx * XORSHIFT_GOLDEN_RATIO_B;

        pool->entries[idx].max_deltas = GENOME_SIZE;
        pool->entries[idx].delta_indices = &delta_indices_buffer[idx * GENOME_SIZE];
        pool->entries[idx].delta_values = &delta_values_buffer[idx * GENOME_SIZE];
        pool->entries[idx].gradients = &gradients_buffer[idx * GENOME_SIZE];

        if (idx < POOL_CAPACITY_MIN) {
            atomicAdd(&alive_inits, 1);
            pool->entries[idx].id = idx;
            pool->entries[idx].phase = LifecyclePhase::ACTIVE;
            pool->alive_flags[idx] = true;
            pool->entries[idx].age = 0;
            pool->entries[idx].parent_hash = UINT64_MAX;
            pool->entries[idx].parent_idx = INT_MAX;
            pool->entries[idx].num_deltas = GENOME_SIZE;

            float* temp_genome = pool->entries[idx].delta_values;
            for (int i = 0; i < GENOME_SIZE; i++) {
                temp_genome[i] = prng_next(&rng) * GENOME_RANGE_SCALE + GENOME_VALUE_MIN;
                pool->entries[idx].delta_indices[i] = i;
                pool->entries[idx].gradients[i] = 0.0f;
            }

            pool->entries[idx].genome_hash = gpu_sha256(temp_genome, GENOME_SIZE);

            PoolInitParams init_params;
            init_params.derive_from_genome(temp_genome, pool->entries[idx].gradients);
            measured_value_set_computed(&pool->entries[idx].hunger, init_params.initial_hunger, generation, pool->entries[idx].genome_hash);

            derive_architecture(temp_genome, &pool->entries[idx]);
            derive_diresa(temp_genome, &pool->entries[idx]);
            derive_fitness_exponents(temp_genome, &pool->entries[idx]);
            measured_value_set_uncomputed(&pool->entries[idx].fitness);
            pool->fitness_values[idx] = NAN;
            measured_value_set_uncomputed(&pool->entries[idx].coherence);
            measured_value_set_uncomputed(&pool->entries[idx].task_accuracy);
            measured_value_set_uncomputed(&pool->entries[idx].generalization_gap);
            measured_value_set_uncomputed(&pool->entries[idx].hardware_efficiency);
            measured_value_set_uncomputed(&pool->entries[idx].effective_rank);
        } else {
            pool->entries[idx].id = INT_MAX;
            pool->entries[idx].phase = LifecyclePhase::DEAD;
            pool->alive_flags[idx] = false;
            measured_value_set_uncomputed(&pool->entries[idx].fitness);
            pool->fitness_values[idx] = NAN;
            measured_value_set_uncomputed(&pool->entries[idx].coherence);
            measured_value_set_uncomputed(&pool->entries[idx].task_accuracy);
            measured_value_set_uncomputed(&pool->entries[idx].generalization_gap);
            measured_value_set_uncomputed(&pool->entries[idx].hardware_efficiency);
            measured_value_set_uncomputed(&pool->entries[idx].effective_rank);
            measured_value_set_uncomputed(&pool->entries[idx].hunger);
            pool->entries[idx].age = INT_MAX;
            pool->entries[idx].parent_hash = UINT64_MAX;
            pool->entries[idx].parent_idx = INT_MAX;
            pool->entries[idx].num_deltas = UINT16_MAX;

            for (int i = 0; i < GENOME_SIZE; i++) {
                pool->entries[idx].gradients[i] = NAN;
            }
        }
    }

    if (idx < POOL_CAPACITY_MIN) {
        pool->alive_indices[idx] = idx;
    }

    cg::this_grid().sync();

    // Report aggregate stats
    if (threadIdx.x == 0) {
        // Use atomicAdd to aggregate across blocks
        int block_active = active_inits;
        int block_alive = alive_inits;
        atomicAdd((int*)&pool->total_spawned, block_alive);  // temp use for counting
    }
    cg::this_grid().sync();

    if (idx == 0) {
        int total_initialized = pool->total_spawned;
        pool->active_count = POOL_CAPACITY_MIN;
        pool->total_spawned = POOL_CAPACITY_MIN;
        pool->total_culled = 0;
        pool->alive_indices_count = POOL_CAPACITY_MIN;
        printf("V:init_pool_done cap=%d alive=%d threads_worked=%d\n", capacity, POOL_CAPACITY_MIN, total_initialized);
    }
}

__device__ void compute_pool_stats_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    GPUElite* archive = organism->archive;
    int archive_size = organism->archive_size;
    float* avg_fitness = organism->stats_avg_fitness;
    float* avg_coherence = organism->stats_avg_coherence;
    float* avg_age = organism->stats_avg_age;
    float* genetic_diversity = organism->stats_genetic_diversity;
    float* workspace_genomes = organism->workspace_genomes;
    DIRESAWeights* diresa_genome_weights = organism->diresa_genome_weights;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_fitness = 0.0f;
    float local_coherence = 0.0f;
    float local_age = 0.0f;

    if (idx < pool->capacity && pool->alive_flags[idx]) {
        local_fitness = pool->fitness_values[idx];
        local_coherence = pool->entries[idx].coherence.value;
        local_age = (float)pool->entries[idx].age;
    }

    float total_fitness = BlockReduce<BLOCK_SIZE>::sum(local_fitness);
    float total_coherence = BlockReduce<BLOCK_SIZE>::sum(local_coherence);
    float total_age = BlockReduce<BLOCK_SIZE>::sum(local_age);

    if (tid == 0) {
        atomicAdd(avg_fitness, total_fitness);
        atomicAdd(avg_coherence, total_coherence);
        atomicAdd(avg_age, total_age);
    }

    if (idx < pool->capacity && pool->alive_flags[idx]) {
        float diversity = 0.0f;

        PRNGState rng;
        rng.s0 = idx * XORSHIFT_GOLDEN_RATIO_A;
        rng.s1 = pool->total_spawned * XORSHIFT_GOLDEN_RATIO_B;

        PoolInitParams params;
        uint64_t genome_hash = pool->entries[idx].genome_hash;

        int hunger_slot = GenomeParamTable::initial_hunger;
        int mutation_slot = GenomeParamTable::genome_mutation_scale;
        int levy_alpha_slot = GenomeParamTable::mutation_levy_alpha;
        int diversity_slot = GenomeParamTable::diversity_normalization;
        int diversity_samples_slot = GenomeParamTable::diversity_sample_count;

        int hunger_min_slot = GenomeParamTable::initial_hunger_min;
        int hunger_max_slot = GenomeParamTable::initial_hunger_max;
        int mutation_min_slot = GenomeParamTable::genome_mutation_scale_min;
        int mutation_max_slot = GenomeParamTable::genome_mutation_scale_max;
        int diversity_min_slot = GenomeParamTable::diversity_normalization_min;
        int diversity_max_slot = GenomeParamTable::diversity_normalization_max;
        int samples_min_slot = GenomeParamTable::diversity_sample_count_min;
        int samples_max_slot = GenomeParamTable::diversity_sample_count_max;
        int samples_base_slot = GenomeParamTable::diversity_sample_count_base;
        int samples_range_slot = GenomeParamTable::diversity_sample_count_range;

        float* temp_genome = &workspace_genomes[tid * GENOME_SIZE * 2];
        float* temp_parent = &workspace_genomes[tid * GENOME_SIZE * 2 + GENOME_SIZE];
        reconstruct_genome_from_archive(
            pool->entries[idx].parent_hash,
            archive,
            archive_size,
            pool->entries[idx].delta_indices,
            pool->entries[idx].delta_values,
            pool->entries[idx].num_deltas,
            pool->entries[idx].max_deltas,
            temp_genome,
            GENOME_SIZE,
            temp_parent,
            diresa_genome_weights
        );

        float* entry_gradients = pool->entries[idx].gradients;

        float samples_min = genome_slot_to_unit(temp_genome, samples_min_slot);
        float samples_max = samples_min + genome_slot_to_unit(temp_genome, samples_max_slot) * (NORMALIZED_MAX - samples_min);
        float samples_base = genome_slot_to_unit(temp_genome, samples_base_slot);
        float samples_range = genome_slot_to_unit(temp_genome, samples_range_slot);
        int base_samples = (int)(samples_base * POOL_CAPACITY_MAX);
        int range_samples = (int)(samples_range * POOL_CAPACITY_MAX);
        int diversity_sample_count = base_samples + (int)(genome_to_bootstrap_param(temp_genome, entry_gradients, diversity_samples_slot, samples_min, samples_max) * range_samples);

        for (int i = 0; i < diversity_sample_count; i++) {
            int other_idx = (int)(prng_next(&rng) * pool->capacity) % pool->capacity;

            if (other_idx != idx && pool->alive_flags[other_idx]) {
                diversity += compute_genome_distance(organism, &pool->entries[idx], &pool->entries[other_idx]);
            }
        }

        float hunger_min = genome_slot_to_unit(temp_genome, hunger_min_slot);
        float hunger_max = hunger_min + genome_slot_to_unit(temp_genome, hunger_max_slot) * (NORMALIZED_MAX - hunger_min);
        float mutation_min = genome_slot_to_unit(temp_genome, mutation_min_slot);
        float mutation_max = mutation_min + genome_slot_to_unit(temp_genome, mutation_max_slot) * (NORMALIZED_MAX - mutation_min);
        float diversity_min = genome_slot_to_unit(temp_genome, diversity_min_slot);
        float diversity_max = diversity_min + genome_slot_to_unit(temp_genome, diversity_max_slot) * (NORMALIZED_MAX - diversity_min);

        params.initial_hunger = genome_to_bootstrap_param(temp_genome, entry_gradients, hunger_slot, hunger_min, hunger_max);
        params.mutation_scale = genome_to_bootstrap_param(temp_genome, entry_gradients, mutation_slot, mutation_min, mutation_max);
        params.mutation_levy_alpha = genome_to_bootstrap_param(temp_genome, entry_gradients, levy_alpha_slot, CHEMOTAXIS_LEVY_ALPHA_MIN, CHEMOTAXIS_LEVY_ALPHA_MAX);
        params.diversity_normalization = genome_to_bootstrap_param(temp_genome, entry_gradients, diversity_slot, diversity_min, diversity_max);

        atomicAdd(genetic_diversity, diversity / params.diversity_normalization);
    }
}


__device__ void compact_pool_alive_indices_device(Organism* organism, cg::grid_group& grid) {
    ComponentPool* pool = organism->pool;
    int* flags = organism->pool_compaction_flags;
    int* scan_workspace = organism->pool_compaction_scan;
    int* scan_recursive_workspace = organism->pool_compaction_scan_recursive;
    int capacity = pool->capacity;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    __shared__ int warp_sums[WARP_SIZE];
    __shared__ int alive_count;

    int is_alive = 0;
    if (tid < capacity) {
        is_alive = pool->alive_flags[tid] ? 1 : 0;
        flags[tid] = is_alive;
    }

    grid.sync();

    int val = (tid < capacity) ? flags[tid] : 0;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        int n = warp.shfl_up(val, offset);
        if (lane >= offset) val += n;
    }

    if (lane == WARP_SIZE - 1) {
        warp_sums[warp_id] = val;
    }
    cg::this_grid().sync();

    if (warp_id == 0) {
        int warp_sum = (lane < (blockDim.x / WARP_SIZE)) ? warp_sums[lane] : 0;
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            int n = warp.shfl_up(warp_sum, offset);
            if (lane >= offset) warp_sum += n;
        }
        warp_sums[lane] = warp_sum;
    }
    cg::this_grid().sync();

    int warp_offset = (warp_id > 0) ? warp_sums[warp_id - 1] : 0;
    int inclusive_val = warp_offset + val;
    int exclusive_val = inclusive_val - ((tid < capacity) ? flags[tid] : 0);

    if (tid < capacity) {
        scan_workspace[tid] = exclusive_val;
    }

    if (threadIdx.x == blockDim.x - 1) {
        scan_recursive_workspace[blockIdx.x] = inclusive_val;
    }

    grid.sync();

    if (blockIdx.x == 0) {
        __shared__ int bsum_shared[WARP_SIZE];
        int num_blocks = gridDim.x;

        int bval = (threadIdx.x < num_blocks) ? scan_recursive_workspace[threadIdx.x] : 0;

        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            int n = warp.shfl_up(bval, offset);
            if (lane >= offset) bval += n;
        }

        if (lane == WARP_SIZE - 1) {
            bsum_shared[warp_id] = bval;
        }
        cg::this_grid().sync();

        if (warp_id == 0) {
            int ws = (lane < (blockDim.x / WARP_SIZE)) ? bsum_shared[lane] : 0;
            #pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                int n = warp.shfl_up(ws, offset);
                if (lane >= offset) ws += n;
            }
            bsum_shared[lane] = ws;
        }
        cg::this_grid().sync();

        int bprefix = (warp_id > 0) ? bsum_shared[warp_id - 1] : 0;
        int b_inclusive = bprefix + bval;
        int b_exclusive = b_inclusive - ((threadIdx.x < num_blocks) ? scan_recursive_workspace[threadIdx.x] : 0);

        if (threadIdx.x < num_blocks) {
            scan_recursive_workspace[threadIdx.x] = b_exclusive;
        }
    }

    grid.sync();

    if (tid < capacity && blockIdx.x > 0) {
        scan_workspace[tid] += scan_recursive_workspace[blockIdx.x];
    }

    grid.sync();

    if (tid < capacity && flags[tid]) {
        int write_pos = scan_workspace[tid];
        pool->alive_indices[write_pos] = tid;
    }

    if (tid == 0) {
        if (capacity > 0) {
            alive_count = scan_workspace[capacity - 1] + flags[capacity - 1];
        } else {
            alive_count = 0;
        }
    }

    grid.sync();

    if (tid == 0) {
        pool->alive_indices_count = alive_count;
    }
}

__device__ void collect_pool_task_accuracies_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    float* pool_task_accuracies = organism->pool_task_accuracies;

    int compact_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (compact_idx < pool->alive_indices_count) {
        int tid = pool->alive_indices[compact_idx];
        DEVICE_FATAL_IF(!pool->alive_flags[tid], "collect_pool_task_accuracies_device: dead entry in alive_indices");

        pool_task_accuracies[tid] = pool->entries[tid].task_accuracy.value;
    }
}

__device__ void inherit_ca_weights_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    int num_pending_inherits = *organism->num_pending_inherits;
    int* child_indices = organism->inherit_child_indices;
    int* parent_indices = organism->inherit_parent_indices;

    int inherit_idx = blockIdx.y;

    if (inherit_idx < num_pending_inherits) {
        int child_idx = child_indices[inherit_idx];
        int parent_idx = parent_indices[inherit_idx];

        PoolEntry* child = &pool->entries[child_idx];
        PoolEntry* parent = &pool->entries[parent_idx];

        DEVICE_FATAL_IF(!pool->alive_flags[child_idx], "inherit_ca_weights_device: child became dead between find and inherit");
        DEVICE_FATAL_IF(!pool->alive_flags[parent_idx], "inherit_ca_weights_device: parent became dead between find and inherit");
        DEVICE_FATAL_IF(!child->ca_state, "inherit_ca_weights_device: child ca_state is null");
        DEVICE_FATAL_IF(!parent->ca_state, "inherit_ca_weights_device: parent ca_state is null");

        MultiHeadCAState* child_ca = child->ca_state;
        MultiHeadCAState* parent_ca = parent->ca_state;

        bool dims_match = (parent->num_heads == child->num_heads) &&
                         (parent->head_dim == child->head_dim) &&
                         (parent->channels == child->channels);

        if (!dims_match) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                printf("WARN: inherit_ca_weights dimension mismatch: parent(%d,%d,%d) != child(%d,%d,%d) - skipping inherit\n",
                       parent->num_heads, parent->head_dim, parent->channels,
                       child->num_heads, child->head_dim, child->channels);
            }
            return;
        }

        int perception_size = child->num_heads * child->channels * child->head_dim;
        int interaction_size = child->num_heads * child->head_dim * child->head_dim;
        int value_size = child->num_heads * child->head_dim * child->channels;

        int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (weight_idx < perception_size) {
            child_ca->perception_weights[weight_idx] = parent_ca->perception_weights[weight_idx];
        }
        if (weight_idx < interaction_size) {
            child_ca->interaction_weights[weight_idx] = parent_ca->interaction_weights[weight_idx];
        }
        if (weight_idx < value_size) {
            child_ca->value_weights[weight_idx] = parent_ca->value_weights[weight_idx];
        }
    }
}

__device__ void find_pending_weight_inherits_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    int* child_indices = organism->inherit_child_indices;
    int* parent_indices = organism->inherit_parent_indices;
    int* num_pending = organism->num_pending_inherits;

    int compact_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (compact_idx < pool->alive_indices_count) {
        int idx = pool->alive_indices[compact_idx];
        DEVICE_FATAL_IF(!pool->alive_flags[idx], "find_pending_weight_inherits_device: dead entry in alive_indices");

        PoolEntry* entry = &pool->entries[idx];

        // Only process new entries (age == 0) that have valid parents
        if (entry->age == 0) {
            int p = entry->parent_idx;

            if (p != INT_MAX && p >= 0 && p < pool->capacity && pool->alive_flags[p]) {
                int slot = atomicAdd(num_pending, 1);
                child_indices[slot] = idx;
                parent_indices[slot] = p;
            }
        }
    }
}

__device__ void seed_archive_from_pool_device(Organism* organism, int num_to_seed, int num_classes) {
    GPUElite* archive = organism->archive;
    int* archive_size = &organism->archive_size;
    ComponentPool* pool = organism->pool;
    VoronoiCell* voronoi_cells = organism->voronoi_cells;
    int num_voronoi_cells = organism->num_voronoi_cells;
    DIRESAWeights* diresa_genome_weights = organism->diresa_genome_weights;
    float* workspace_genomes = organism->workspace_genomes;
    int hw_dim = organism->archive->hw_dim;
    int task_dim = organism->archive->task_dim;
    int gen_dim = organism->archive->gen_dim;

    int compact_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_num_to_seed = min(num_to_seed, pool->alive_indices_count);

    if (compact_idx < actual_num_to_seed) {
        int idx = pool->alive_indices[compact_idx];
        DEVICE_FATAL_IF(!pool->alive_flags[idx], "seed_archive_from_pool_device: dead entry in alive_indices");

        PoolEntry* entry = &pool->entries[idx];

        float* temp_genome = &workspace_genomes[idx * GENOME_SIZE * 2];
        float* temp_parent = &workspace_genomes[idx * GENOME_SIZE * 2 + GENOME_SIZE];

        reconstruct_genome_from_archive(entry->parent_hash, archive, *archive_size,
            entry->delta_indices, entry->delta_values, entry->num_deltas,
            entry->max_deltas, temp_genome, GENOME_SIZE, temp_parent, diresa_genome_weights);

        float latent[GENOME_LATENT_DIM_MAX];
        diresa_encode(temp_genome, latent, diresa_genome_weights);

        float hw_coords[BEHAVIORAL_DIM_HW];
        float task_coords[BEHAVIORAL_DIM_TASK];
        float gen_coords[BEHAVIORAL_DIM_GEN];
        float per_class_accuracy[NUM_CLASSES_MAX];

        for (int i = 0; i < hw_dim; i++) hw_coords[i] = 0.0f;
        for (int i = 0; i < task_dim; i++) task_coords[i] = 0.0f;
        for (int i = 0; i < gen_dim; i++) gen_coords[i] = 0.0f;
        for (int i = 0; i < num_classes; i++) per_class_accuracy[i] = 0.0f;

        insert_elite_device(
            archive,
            archive_size,
            entry->fitness.value,
            entry->coherence.value,
            entry->effective_rank.value,
            entry->genome_hash,
            0,
            0,
            0,
            hw_coords,
            task_coords,
            gen_coords,
            0.0f,
            per_class_accuracy,
            num_classes,
            voronoi_cells,
            num_voronoi_cells,
            latent,
            entry->fitness.input_hash,
            entry->fitness.computed_at_generation
        );
    }
}

__device__ void spawn_wave_device(Organism* organism) {
    float spawn_probability = organism->spawn_probability;
    ComponentPool* pool = organism->pool;
    float* workspace_genomes = organism->spawn_workspace;

    __shared__ int qualifying_parents[BLOCK_SIZE];
    __shared__ int qualifying_count;

    int tid = threadIdx.x;
    int alive_count = pool->alive_indices_count;

    if (tid == 0) qualifying_count = 0;
    cg::this_grid().sync();

    for (int compact = tid; compact < alive_count; compact += blockDim.x) {
        int i = pool->alive_indices[compact];
        DEVICE_FATAL_IF(!pool->alive_flags[i], "spawn_wave_kernel: dead entry in alive_indices");

        float* temp_genome = &workspace_genomes[tid * GENOME_SIZE * SPAWN_WS_COUNT + GENOME_SIZE * SPAWN_WS_TEMP_GENOME];
        float* temp_parent = &workspace_genomes[tid * GENOME_SIZE * SPAWN_WS_COUNT + GENOME_SIZE * SPAWN_WS_TEMP_PARENT];

        PoolEntry* spawn_entry = &pool->entries[i];
        reconstruct_genome_from_archive(spawn_entry->parent_hash, (GPUElite*)organism->archive, organism->archive_size,
            spawn_entry->delta_indices, spawn_entry->delta_values, spawn_entry->num_deltas,
            spawn_entry->max_deltas, temp_genome, GENOME_SIZE, temp_parent, organism->diresa_genome_weights);

        uint64_t temp_hash = pool->entries[i].genome_hash;
        int fitness_threshold_slot = GenomeParamTable::spawn_fitness_threshold;
        float local_morphogen = sample_neighborhood(
            organism->chemical_field->concentration, i, pool->entries[i].grid_size);

        float entry_fitness = pool->fitness_values[i];
        float entry_hunger = pool->entries[i].hunger.value;

        int ctx_metabolic_slot = GenomeParamTable::spawn_ctx_metabolic;
        float ctx_metabolic = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_metabolic_slot,
            entry_fitness, entry_hunger, local_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            NORMALIZED_MIN, NORMALIZED_MAX);

        int ctx_stress_slot = GenomeParamTable::spawn_ctx_stress;
        float ctx_stress = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_stress_slot,
            entry_fitness, entry_hunger, local_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            NORMALIZED_MIN, NORMALIZED_MAX);

        int ctx_morphogen_slot = GenomeParamTable::spawn_ctx_morphogen;
        float ctx_morphogen = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_morphogen_slot,
            entry_fitness, entry_hunger, local_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            NORMALIZED_MIN, NORMALIZED_MAX);

        float fitness_threshold = genome_to_param(
            temp_genome, pool->entries[i].gradients, fitness_threshold_slot,
            ctx_metabolic, ctx_stress, ctx_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            SPAWN_PROBABILITY_MIN_MIN, SPAWN_PROBABILITY_MIN_MAX
        );

        if (entry_fitness > fitness_threshold) {
            int slot = atomicAdd(&qualifying_count, 1);
            if (slot < BLOCK_SIZE) {
                qualifying_parents[slot] = i;
            }
        }
    }
    cg::this_grid().sync();

    // Only spawn if there are qualifying parents
    if (qualifying_count > 0) {
        unsigned int seed = tid * organism->generation * RNG_SEED_MULTIPLIER;
        seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
        float rand = (seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE;

        // Only spawn with given probability
        if (rand < spawn_probability) {
            int parent_slot = tid % qualifying_count;
            int parent_idx = qualifying_parents[parent_slot];

            float* workspace_parent_genome = &workspace_genomes[tid * GENOME_SIZE * SPAWN_WS_COUNT + GENOME_SIZE * SPAWN_WS_PARENT_GENOME];
            float* workspace_child_genome = &workspace_genomes[tid * GENOME_SIZE * SPAWN_WS_COUNT + GENOME_SIZE * SPAWN_WS_CHILD_GENOME];
            float* workspace_parent_parent_temp = &workspace_genomes[tid * GENOME_SIZE * SPAWN_WS_COUNT + GENOME_SIZE * SPAWN_WS_PARENT_PARENT_TEMP];

            PoolEntry* parent_entry = &pool->entries[parent_idx];
            reconstruct_genome_from_archive(parent_entry->parent_hash, (GPUElite*)organism->archive, organism->archive_size,
                parent_entry->delta_indices, parent_entry->delta_values, parent_entry->num_deltas,
                parent_entry->max_deltas, workspace_parent_genome, GENOME_SIZE, workspace_parent_parent_temp, organism->diresa_genome_weights);

            uint64_t parent_hash = pool->entries[parent_idx].genome_hash;
            float parent_morphogen = sample_neighborhood(
                organism->chemical_field->concentration, parent_idx, pool->entries[parent_idx].grid_size);

            float parent_fitness = pool->fitness_values[parent_idx];
            float parent_hunger = pool->entries[parent_idx].hunger.value;

            int ctx_metabolic_slot = GenomeParamTable::mutation_ctx_metabolic;
            float ctx_metabolic = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_metabolic_slot,
                parent_fitness, parent_hunger, parent_morphogen,
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                NORMALIZED_MIN, NORMALIZED_MAX);

            int ctx_stress_slot = GenomeParamTable::mutation_ctx_stress;
            float ctx_stress = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_stress_slot,
                parent_fitness, parent_hunger, parent_morphogen,
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                NORMALIZED_MIN, NORMALIZED_MAX);

            int ctx_morphogen_slot = GenomeParamTable::mutation_ctx_morphogen;
            float ctx_morphogen = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_morphogen_slot,
                parent_fitness, parent_hunger, parent_morphogen,
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                NORMALIZED_MIN, NORMALIZED_MAX);

            int mutation_rate_slot = GenomeParamTable::metalearning_mutation_rate;
            float mutation_rate = genome_to_param(
                workspace_parent_genome,
                pool->entries[parent_idx].gradients,
                mutation_rate_slot,
                ctx_metabolic, ctx_stress, ctx_morphogen,
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                SPAWN_RATE_MIN, SPAWN_RATE_MAX
            );

            spawn_component_device(
                organism,
                parent_idx,
                mutation_rate,
                workspace_parent_genome,
                workspace_child_genome,
                workspace_parent_parent_temp
            );
        }
    }
}

__device__ void culling_device(Organism* organism, float fitness_threshold, float hunger_threshold) {
    ComponentPool* pool = organism->pool;
    GPUElite* archive = organism->archive;
    int archive_size = organism->archive_size;
    ChemicalField* chemical_field = organism->chemical_field;
    float* workspace_genomes = organism->workspace_genomes;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity) {
        PoolEntry* entry = &pool->entries[idx];

        if (pool->alive_flags[idx]) {
            float* entry_genome = &workspace_genomes[idx * GENOME_SIZE * 2];
            float* entry_parent_temp = &workspace_genomes[idx * GENOME_SIZE * 2 + GENOME_SIZE];
            reconstruct_genome_from_archive(entry->parent_hash, archive, archive_size,
                entry->delta_indices, entry->delta_values, entry->num_deltas,
                entry->max_deltas, entry_genome, GENOME_SIZE, entry_parent_temp, organism->diresa_genome_weights);

            uint64_t entry_genome_hash = entry->genome_hash;
            float ctx_metabolic = entry->fitness.value;
            float ctx_stress = entry->hunger.value;

            float ctx_morphogen = 0.0f;
            for (int c = 0; c < chemical_field->channels; c++) {
                ctx_morphogen += chemical_field->cached_mean[c];
            }
            ctx_morphogen /= chemical_field->channels;

            int fitness_cull_mult_slot = GenomeParamTable::lifecycle_fitness_culling_mult;
            float fitness_cull_mult = genome_to_param(entry_genome, entry->gradients, fitness_cull_mult_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, FITNESS_CULLING_MULT_MIN, FITNESS_CULLING_MULT_MAX);

            if (entry->fitness.value < fitness_threshold * fitness_cull_mult) {
                pool->alive_flags[idx] = false;
                Atomics::increment_int(pool->total_culled);
                Atomics::decrement_int(pool->active_count);
            }

            else if (entry->hunger.value > hunger_threshold) {
                pool->alive_flags[idx] = false;
                Atomics::increment_int(pool->total_culled);
                Atomics::decrement_int(pool->active_count);
            }
        }
    }
}

#endif

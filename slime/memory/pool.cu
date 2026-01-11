
#ifndef POOL_CU
#define POOL_CU
#include "../config/config.cu"
#include "../utils/genome_params.cuh"
#include "../utils/cuda_primitives.cuh"
#include "../memory/archive.cu"
#include "../compression/delta.cu"
#include "../memory/genome_ops.cuh"
#include "../memory/parallel_compaction.cu"
#include "../core/ca_state.cuh"
#include <cuda_runtime.h>
#include <cuda/atomic>

struct PoolEntry {
    int id;
    float fitness;
    float coherence;
    float task_accuracy;
    float generalization_gap;
    float hardware_efficiency;
    float hunger;
    int age;
    bool alive;
    uint64_t genome_hash;
    uint64_t parent_hash;
    int parent_idx;  // Pool index of parent, or INT_MAX if spawned from archive/random
    int generation;
    uint16_t num_deltas;
    uint16_t max_deltas;
    uint16_t* delta_indices;
    float* delta_values;
    float* gradients;
    int num_heads;
    int channels;
    int hidden_dim;
    int head_dim;
    int grid_size;
    int num_tempering_replicas;
    int diresa_hidden1;
    int diresa_hidden2;
    int diresa_batch_size;
    float anneal_step;
    float cov_target;
    float dist_weight;
    float recon_weight;
    float distance_exponent;
    float quality_weight;
    float fitness_rank_exponent;
    float fitness_coherence_exponent;
    float fitness_coupling_exponent;
    float fitness_task_exponent;
    float fitness_gen_exponent;
    float fitness_efficiency_exponent;
    float baldwin_sensitivity;
    int coherence_window_size;
    float renyi_q;

    // Per-organism Flow Lenia parameters (derived from genome)
    float flow_beta_A;
    float flow_n;
    float flow_s;
    float flow_alpha_min;
    float flow_alpha_max;
    float flow_sharpness;
    float flow_resource_dt;

    MultiHeadCAState* ca_state;
};

struct ComponentPool {
    PoolEntry* entries;
    cuda::atomic<int> active_count;
    cuda::atomic<int> total_spawned;
    cuda::atomic<int> total_culled;
    int capacity;

    // Compact index array - maps [0, alive_count) to actual entry indices
    // Built by compact_alive_indices_kernel before kernels that need dense iteration
    int* alive_indices;
    int alive_indices_count;  // Snapshot of active_count at compaction time

    // SoA hot fields for coalesced access in iteration kernels
    bool* alive_flags;      // [capacity] - mirrors entries[i].alive
    float* fitness_values;  // [capacity] - mirrors entries[i].fitness
};

// DIRESA autoencoder - needs PoolEntry definition above
#include "../learning/diresa.cu"

__global__ void init_rng_states_kernel(curandState* states, int count, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Step 1: Mark alive entries into flags array
__global__ void mark_alive_kernel(ComponentPool* pool, int* flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pool->capacity) {
        // Use SoA for coalesced alive read
        flags[idx] = pool->alive_flags[idx] ? 1 : 0;
    }
}

// Step 2: Use existing exclusive_scan_kernel from parallel_compaction.cu

// Step 3: Scatter indices based on scan results
__global__ void scatter_alive_indices_kernel(
    ComponentPool* pool, int* flags, int* scan_results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pool->capacity && flags[idx]) {
        pool->alive_indices[scan_results[idx]] = idx;
    }
}

// Step 4: Finalize count
__global__ void finalize_alive_count_kernel(
    ComponentPool* pool, int* flags, int* scan_results, int capacity
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int last_idx = capacity - 1;
        pool->alive_indices_count = scan_results[last_idx] + flags[last_idx];
    }
}

__device__ __forceinline__ void derive_architecture(uint64_t genome_hash, const float* genome, PoolEntry* entry) {
    int num_heads_slot = derive_param_slot(genome_hash, "arch_num_heads");
    int channels_slot = derive_param_slot(genome_hash, "arch_channels");
    int head_dim_slot = derive_param_slot(genome_hash, "arch_head_dim");
    int grid_size_slot = derive_param_slot(genome_hash, "arch_grid_size");

    float num_heads_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[num_heads_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float channels_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[channels_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float head_dim_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[head_dim_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float grid_size_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[grid_size_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));

    entry->num_heads = (int)fmaxf((float)NUM_HEADS_MIN, fminf((float)NUM_HEADS_MAX, NUM_HEADS_MIN + num_heads_norm * (NUM_HEADS_MAX - NUM_HEADS_MIN)));

    int head_dim_tiles = min(HEAD_DIM_TILES_MAX, HEAD_DIM_TILES_MIN + (int)(head_dim_norm * (HEAD_DIM_TILES_MAX - HEAD_DIM_TILES_MIN + 1)));
    int channels_octets = min(CHANNELS_OCTETS_MAX, CHANNELS_OCTETS_MIN + (int)(channels_norm * (CHANNELS_OCTETS_MAX - CHANNELS_OCTETS_MIN + 1)));

    entry->head_dim = head_dim_tiles * WMMA_TILE_DIM;
    entry->channels = channels_octets * WMMA_ALIGNMENT;
    entry->grid_size = (int)fmaxf((float)GRID_SIZE_MIN, fminf((float)GRID_SIZE_MAX, GRID_SIZE_MIN + grid_size_norm * (GRID_SIZE_MAX - GRID_SIZE_MIN)));
    entry->hidden_dim = entry->num_heads * entry->head_dim;
}

__device__ __forceinline__ void derive_diresa(uint64_t genome_hash, const float* genome, PoolEntry* entry) {
    int replicas_slot = derive_param_slot(genome_hash, "diresa_num_replicas");
    int hidden1_slot = derive_param_slot(genome_hash, "diresa_hidden1");
    int hidden2_slot = derive_param_slot(genome_hash, "diresa_hidden2");
    int batch_size_slot = derive_param_slot(genome_hash, "diresa_batch_size");
    int anneal_step_slot = derive_param_slot(genome_hash, "diresa_anneal_step");
    int cov_target_slot = derive_param_slot(genome_hash, "diresa_cov_target");
    int dist_weight_slot = derive_param_slot(genome_hash, "diresa_dist_weight");
    int recon_weight_slot = derive_param_slot(genome_hash, "diresa_recon_weight");
    int distance_exponent_slot = derive_param_slot(genome_hash, "diresa_distance_exponent");
    int quality_weight_slot = derive_param_slot(genome_hash, "diresa_quality_weight");

    float replicas_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[replicas_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float hidden1_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[hidden1_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float hidden2_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[hidden2_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float batch_size_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[batch_size_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float anneal_step_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[anneal_step_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float cov_target_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[cov_target_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float dist_weight_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[dist_weight_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float recon_weight_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[recon_weight_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float distance_exponent_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[distance_exponent_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    float quality_weight_norm = fmaxf(NORMALIZED_MIN, fminf(NORMALIZED_MAX, (genome[quality_weight_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));

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

__device__ __forceinline__ void derive_fitness_exponents(uint64_t genome_hash, const float* genome, PoolEntry* entry) {
    int rank_exp_slot = derive_param_slot(genome_hash, "fitness_rank_exponent");
    int coh_exp_slot = derive_param_slot(genome_hash, "fitness_coherence_exponent");
    int coupling_exp_slot = derive_param_slot(genome_hash, "fitness_coupling_exponent");
    int task_exp_slot = derive_param_slot(genome_hash, "fitness_task_exponent");
    int gen_exp_slot = derive_param_slot(genome_hash, "fitness_gen_exponent");
    int eff_exp_slot = derive_param_slot(genome_hash, "fitness_efficiency_exponent");
    int baldwin_slot = derive_param_slot(genome_hash, "baldwin_sensitivity");
    int window_slot = derive_param_slot(genome_hash, "coherence_window_size");
    int renyi_q_slot = derive_param_slot(genome_hash, "rank_renyi_order");

    float rank_exp_norm = (genome[rank_exp_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float coh_exp_norm = (genome[coh_exp_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float coupling_exp_norm = (genome[coupling_exp_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float task_exp_norm = (genome[task_exp_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float gen_exp_norm = (genome[gen_exp_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float eff_exp_norm = (genome[eff_exp_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float baldwin_norm = (genome[baldwin_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float window_norm = (genome[window_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float renyi_q_norm = (genome[renyi_q_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;

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

struct PRNGState {
    uint64_t s0;
    uint64_t s1;

    __device__ float next() {
        uint64_t x = s0;
        uint64_t y = s1;
        uint64_t result = x + y;
        y ^= x;
        s0 = ((x << XORSHIFT128_ROTL_A) | (x >> (XORSHIFT_STATE_BITS - XORSHIFT128_ROTL_A))) ^ y ^ (y << XORSHIFT128_SHIFT_C);
        s1 = (y << XORSHIFT128_ROTL_B) | (y >> (XORSHIFT_STATE_BITS - XORSHIFT128_ROTL_B));
        return (double)result / XORSHIFT_NORMALIZATION_SCALE;
    }

    __device__ float levy_stable(float alpha, float scale) {
        float U = next();
        float V = next();
        float W = next();

        float phi = (TAU * 0.25f) * (2.0f * V - 1.0f);
        float alpha_phi = alpha * phi;
        float one_minus_alpha_phi = (1.0f - alpha) * phi;

        float levy_num = sinf(alpha_phi);
        float cos_phi = cosf(phi);
        if (cos_phi <= 0.0f) {
            return NAN;
        }
        float levy_denom = powf(cos_phi, (1.0f / alpha));

        if (W <= 0.0f || W >= 1.0f) {
            return NAN;
        }
        float log_w = -logf(W);
        if (log_w <= 0.0f) {
            return NAN;
        }
        float cos_one_minus_alpha_phi = cosf(one_minus_alpha_phi);
        if (cos_one_minus_alpha_phi <= 0.0f) {
            return NAN;
        }
        float levy_factor = powf(cos_one_minus_alpha_phi / log_w, ((1.0f - alpha) / alpha));

        return (levy_num / levy_denom * levy_factor) * scale;
    }
};

__device__ void spawn_component_device(
    ComponentPool* pool,
    GPUElite* archive,
    int archive_size,
    int parent_id,
    float mutation_rate,
    float* workspace_parent_genome,
    float* workspace_child_genome,
    float* workspace_parent_temp,
    DIRESAWeights* diresa_genome_weights
) {
    int slot_idx = -1;
    int new_id;
    int parent_archive_idx;
    uint64_t parent_genome_hash;
    float inheritance_factor;
    PRNGState rng;
    PoolInitParams params;
    float delta_threshold;
    int has_parent;
    int use_latent;

    int count = Atomics::load_int(pool->active_count);
    if (count < pool->capacity) {
        new_id = atomicAdd((int*)&pool->total_spawned, 1);
    } else {
        return;
    }

    for (int i = 0; i < pool->capacity; i++) {
        // Use SoA for coalesced alive read
        if (!pool->alive_flags[i]) {
            int claimed = atomicCAS((int*)&slot_idx, -1, i);
            if (claimed == -1) break;
        }
    }

    if (slot_idx < 0) return;
    int i = slot_idx;

    pool->entries[i].id = new_id;
    pool->entries[i].age = 0;
    pool->entries[i].alive = true;
    pool->alive_flags[i] = true;  // SoA sync
    has_parent = (parent_id >= 0 && parent_id < pool->capacity) ? 1 : 0;
    pool->entries[i].parent_idx = has_parent ? parent_id : INT_MAX;
    if (has_parent) {
        parent_genome_hash = pool->entries[parent_id].genome_hash;
    }

    float* parent_genome = workspace_parent_genome;
    float* child_genome = workspace_child_genome;

    if (has_parent) {
        PoolEntry* parent = &pool->entries[parent_id];

        // O(1) lookup via hash table
        parent_archive_idx = hash_table_lookup(
            archive->hash_table_keys,
            archive->hash_table_values,
            parent->parent_hash
        );

        use_latent = (parent_archive_idx >= 0 && archive->latent_genome != nullptr) ? 1 : 0;
        rng.s0 = new_id * 0x9e3779b97f4a7c15ULL;
        rng.s1 = parent_id * 0xbf58476d1ce4e5b9ULL;

        reconstruct_genome_from_archive(parent->parent_hash, archive, archive_size,
            parent->delta_indices, parent->delta_values, parent->num_deltas,
            parent->max_deltas, parent_genome, GENOME_SIZE, workspace_parent_temp, diresa_genome_weights);

        params.derive_from_genome(parent->genome_hash, parent_genome);

        if (use_latent) {
            float* parent_latent = &archive->latent_genome[parent_archive_idx * GENOME_LATENT_DIM_MAX];
            for (int j = 0; j < GENOME_LATENT_DIM_MAX; j++) {
                float mutated_latent = parent_latent[j];
                PRNGState local_rng = rng;
                local_rng.s0 ^= j;
                if (local_rng.next() < mutation_rate) {
                    mutated_latent += local_rng.levy_stable(params.mutation_levy_alpha, params.mutation_scale);
                    mutated_latent = tanhf(mutated_latent);
                }
                workspace_parent_temp[j] = mutated_latent;
            }
            diresa_decode(workspace_parent_temp, child_genome, diresa_genome_weights);
        } else {
            for (int j = 0; j < GENOME_SIZE; j++) {
                float val = parent_genome[j];
                PRNGState local_rng = rng;
                local_rng.s0 ^= j;
                if (local_rng.next() < mutation_rate) {
                    val += local_rng.levy_stable(params.mutation_levy_alpha, params.mutation_scale);
                    val = tanhf(val);
                }
                child_genome[j] = val;
            }
        }

        pool->entries[i].parent_hash = parent->genome_hash;
        int inherit_center_slot = derive_param_slot(parent->genome_hash, "fitness_inherit_center");
        int inherit_steep_slot = derive_param_slot(parent->genome_hash, "fitness_inherit_steepness");
        float inherit_center = LIFECYCLE_FITNESS_INHERIT_CENTER_MIN +
            ((parent_genome[inherit_center_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE) *
            (LIFECYCLE_FITNESS_INHERIT_CENTER_MAX - LIFECYCLE_FITNESS_INHERIT_CENTER_MIN);
        float inherit_steepness = LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MIN +
            ((parent_genome[inherit_steep_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE) *
            (LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MAX - LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MIN);
        inheritance_factor = 1.0f / (1.0f + expf(-inherit_steepness * (parent->fitness - inherit_center)));

        for (int j = 0; j < GENOME_SIZE; j++) {
            pool->entries[i].gradients[j] = parent->gradients[j] * inheritance_factor;
        }

        // Weight copy deferred to inherit_ca_weights_kernel
        pool->entries[i].parent_hash = parent->genome_hash;
    } else {
        rng.s0 = new_id * 0x9e3779b97f4a7c15ULL;
        rng.s1 = new_id * 0xbf58476d1ce4e5b9ULL;
        pool->entries[i].parent_hash = UINT64_MAX;  // No parent

        for (int j = 0; j < GENOME_SIZE; j++) {
            PRNGState local_rng = rng;
            local_rng.s0 ^= j;
            child_genome[j] = local_rng.next() * GENOME_RANGE_SCALE + GENOME_VALUE_MIN;
            pool->entries[i].gradients[j] = 0.0f;
        }
    }

    pool->entries[i].genome_hash = gpu_sha256(child_genome, GENOME_SIZE);
    pool->entries[i].num_deltas = 0;
    int delta_threshold_slot = derive_param_slot(pool->entries[i].genome_hash, "delta_threshold");
    delta_threshold = (child_genome[delta_threshold_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * 0.01f;

    for (int j = 0; j < GENOME_SIZE; j++) {
        float diff = child_genome[j] - parent_genome[j];
        if (fabsf(diff) > delta_threshold) {
            int delta_idx = atomicAdd((int*)&pool->entries[i].num_deltas, 1);
            if (delta_idx < pool->entries[i].max_deltas) {
                pool->entries[i].delta_indices[delta_idx] = j;
                pool->entries[i].delta_values[delta_idx] = diff;
            }
        }
    }

    PoolInitParams init_params;
    init_params.derive_from_genome(pool->entries[i].genome_hash, child_genome);
    pool->entries[i].hunger = init_params.initial_hunger;
    derive_architecture(pool->entries[i].genome_hash, child_genome, &pool->entries[i]);
    derive_diresa(pool->entries[i].genome_hash, child_genome, &pool->entries[i]);
    derive_fitness_exponents(pool->entries[i].genome_hash, child_genome, &pool->entries[i]);
    int init_fitness_slot = derive_param_slot(pool->entries[i].genome_hash, "initial_fitness_prior");
    float init_fitness_prior = fmaxf(0.01f, (child_genome[init_fitness_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE);
    pool->entries[i].fitness = init_fitness_prior;
    pool->fitness_values[i] = init_fitness_prior;  // SoA sync
    pool->entries[i].coherence = init_fitness_prior;
    pool->entries[i].task_accuracy = NAN;  // Set by classification evaluation
    pool->entries[i].generalization_gap = NAN;  // Set by held-out evaluation
    pool->entries[i].hardware_efficiency = NAN;  // Set by hardware profiling
    Atomics::increment_int(pool->active_count);
}

__global__ void cull_weak_kernel(
    ComponentPool* pool,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity) {
        // Use SoA for coalesced alive/fitness reads
        if (pool->alive_flags[idx] && pool->fitness_values[idx] < threshold) {
            pool->entries[idx].alive = false;
            pool->alive_flags[idx] = false;  // SoA sync
            Atomics::decrement_int(pool->active_count);
            Atomics::increment_int(pool->total_culled);
        }
    }
}

__global__ void cull_hungry_kernel(
    ComponentPool* pool,
    float hunger_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity) {
        // Use SoA for coalesced alive read
        if (pool->alive_flags[idx] && pool->entries[idx].hunger > hunger_threshold) {
            pool->entries[idx].alive = false;
            pool->alive_flags[idx] = false;  // SoA sync
            Atomics::decrement_int(pool->active_count);
            Atomics::increment_int(pool->total_culled);
        }
    }
}

__global__ void age_components_kernel(ComponentPool* pool) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use SoA for coalesced alive read
    if (idx < pool->capacity && pool->alive_flags[idx]) {
        pool->entries[idx].age++;
    }
}

__global__ void update_hunger_kernel(ComponentPool* pool) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use SoA for coalesced alive read
    if (idx < pool->capacity && pool->alive_flags[idx]) {
        pool->entries[idx].hunger = fmaxf(0.01f, 1.0f - pool->entries[idx].coherence);
    }
}

__global__ void sort_by_fitness_kernel(
    ComponentPool* pool,
    int stage,
    int step
) {
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
        bool swap = ascending ? (left.fitness > right.fitness) :
                               (left.fitness < right.fitness);

        if (swap) {
            pool->entries[left_id] = right;
            pool->entries[right_id] = left;
        }
    }
}

__global__ void select_top_k_kernel(
    ComponentPool* pool,
    int* selected_indices,
    int k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < k && tid < pool->capacity) {
        // Use SoA for coalesced alive read
        if (pool->alive_flags[tid]) {
            selected_indices[tid] = tid;
        } else {
            selected_indices[tid] = -1;
        }
    }
}

__device__ __noinline__ float compute_genome_distance(
    PoolEntry* entry1,
    PoolEntry* entry2,
    GPUElite* archive,
    int archive_size
) {
    // O(1) lookups via hash table
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

    if (idx1 < 0 || idx2 < 0) return 1e10f;

    const float* latent1 = &archive->latent_genome[idx1 * GENOME_LATENT_DIM_MAX];
    const float* latent2 = &archive->latent_genome[idx2 * GENOME_LATENT_DIM_MAX];
    float distance_squared = DIRESAOps::compute_latent_distance_sq(latent1, latent2, GENOME_LATENT_DIM_MAX);

    return sqrtf(distance_squared / GENOME_LATENT_DIM_MAX);
}

__global__ void diversity_selection_kernel(
    ComponentPool* pool,
    GPUElite* archive,
    int archive_size,
    int* selected_indices,
    int num_select
) {
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
    // Use SoA for coalesced alive reads
    if (idx < capacity && pool->alive_flags[idx]) {
        float sum_dist = 0.0f;
        int count = 0;
        for (int i = tid; i < capacity; i += blockDim.x) {
            if (i != idx && pool->alive_flags[i]) {
                float dist = compute_genome_distance(&pool->entries[idx], &pool->entries[i], archive, archive_size);
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
    __syncthreads();

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
        __syncthreads();

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
        __syncthreads();

        if (tid == 0) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                if (indices[j] == max_idx) {
                    distances[j] = -1.0f;
                    break;
                }
            }
        }
        __syncthreads();
    }
}

__global__ void init_pool_kernel(
    ComponentPool* pool,
    int capacity,
    uint16_t* delta_indices_buffer,
    float* delta_values_buffer,
    float* gradients_buffer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < capacity) {

        PRNGState rng;
        rng.s0 = idx * 0x9e3779b97f4a7c15ULL;
        rng.s1 = idx * 0xbf58476d1ce4e5b9ULL;

        pool->entries[idx].max_deltas = GENOME_SIZE;
        pool->entries[idx].delta_indices = &delta_indices_buffer[idx * GENOME_SIZE];
        pool->entries[idx].delta_values = &delta_values_buffer[idx * GENOME_SIZE];
        pool->entries[idx].gradients = &gradients_buffer[idx * GENOME_SIZE];

        if (idx < POOL_CAPACITY_MIN) {
            pool->entries[idx].id = idx;
            pool->entries[idx].alive = true;
            pool->alive_flags[idx] = true;  // SoA sync
            pool->entries[idx].age = 0;
            pool->entries[idx].parent_hash = 0;
            pool->entries[idx].parent_idx = INT_MAX;  // No parent - randomly generated
            pool->entries[idx].num_deltas = GENOME_SIZE;

            float temp_genome[GENOME_SIZE];
            for (int i = 0; i < GENOME_SIZE; i++) {
                temp_genome[i] = rng.next() * GENOME_RANGE_SCALE + GENOME_VALUE_MIN;
                pool->entries[idx].delta_indices[i] = i;
                pool->entries[idx].delta_values[i] = temp_genome[i];
                pool->entries[idx].gradients[i] = 0.0f;
            }

            pool->entries[idx].genome_hash = gpu_sha256(temp_genome, GENOME_SIZE);

            PoolInitParams init_params;
            init_params.derive_from_genome(pool->entries[idx].genome_hash, temp_genome);
            pool->entries[idx].hunger = init_params.initial_hunger;

            derive_architecture(pool->entries[idx].genome_hash, temp_genome, &pool->entries[idx]);
            derive_diresa(pool->entries[idx].genome_hash, temp_genome, &pool->entries[idx]);
            derive_fitness_exponents(pool->entries[idx].genome_hash, temp_genome, &pool->entries[idx]);

            int init_fitness_slot = derive_param_slot(pool->entries[idx].genome_hash, "initial_fitness_prior");
            float init_fitness_prior = fmaxf(0.01f, (temp_genome[init_fitness_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE);
            pool->entries[idx].fitness = init_fitness_prior;
            pool->fitness_values[idx] = init_fitness_prior;  // SoA sync
            pool->entries[idx].coherence = init_fitness_prior;
            pool->entries[idx].task_accuracy = NAN;  // Set by classification evaluation
            pool->entries[idx].generalization_gap = NAN;  // Set by held-out evaluation
            pool->entries[idx].hardware_efficiency = NAN;  // Set by hardware profiling
        } else {
            pool->entries[idx].id = INT_MAX;  // Poison value
            pool->entries[idx].alive = false;
            pool->alive_flags[idx] = false;  // SoA sync
            pool->entries[idx].fitness = NAN;
            pool->fitness_values[idx] = NAN;  // SoA sync
            pool->entries[idx].coherence = NAN;
            pool->entries[idx].task_accuracy = NAN;
            pool->entries[idx].generalization_gap = NAN;
            pool->entries[idx].hardware_efficiency = NAN;
            pool->entries[idx].hunger = NAN;
            pool->entries[idx].age = INT_MAX;  // Poison value
            pool->entries[idx].parent_hash = UINT64_MAX;  // Poison value
            pool->entries[idx].parent_idx = INT_MAX;  // Poison value
            pool->entries[idx].num_deltas = UINT16_MAX;  // Poison value

            for (int i = 0; i < GENOME_SIZE; i++) {
                pool->entries[idx].gradients[i] = NAN;
            }
        }
    }

    if (idx == 0) {
        pool->capacity = capacity;
        pool->active_count = POOL_CAPACITY_MIN;
        pool->total_spawned = POOL_CAPACITY_MIN;
        pool->total_culled = 0;
    }
}

__global__ void compute_pool_stats_kernel(
    ComponentPool* pool,
    GPUElite* archive,
    int archive_size,
    float* avg_fitness,
    float* avg_coherence,
    float* avg_age,
    float* genetic_diversity,
    float* workspace_genomes,
    DIRESAWeights* diresa_genome_weights
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_fitness = 0.0f;
    float local_coherence = 0.0f;
    float local_age = 0.0f;

    // Use SoA for coalesced alive/fitness reads
    if (idx < pool->capacity && pool->alive_flags[idx]) {
        local_fitness = pool->fitness_values[idx];
        local_coherence = pool->entries[idx].coherence;
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

    // Use SoA for coalesced alive read
    if (idx < pool->capacity && pool->alive_flags[idx]) {
        float diversity = 0.0f;

        PRNGState rng;
        rng.s0 = idx * 0x9e3779b97f4a7c15ULL;
        rng.s1 = pool->total_spawned * 0xbf58476d1ce4e5b9ULL;

        PoolInitParams params;
        uint64_t genome_hash = pool->entries[idx].genome_hash;

        int hunger_slot = derive_param_slot(genome_hash, "initial_hunger");
        int mutation_slot = derive_param_slot(genome_hash, "genome_mutation_scale");
        int levy_alpha_slot = derive_param_slot(genome_hash, "mutation_levy_alpha");
        int diversity_slot = derive_param_slot(genome_hash, "diversity_normalization");
        int diversity_samples_slot = derive_param_slot(genome_hash, "diversity_sample_count");

        // Reconstruct full genome to extract parameters
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

        float genome_diversity_samples = temp_genome[diversity_samples_slot];
        int diversity_sample_count = 4 + (int)((genome_diversity_samples + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * 28.0f);

        for (int i = 0; i < diversity_sample_count; i++) {
            int other_idx = (int)(rng.next() * pool->capacity) % pool->capacity;
            // Use SoA for coalesced alive read
            if (other_idx != idx && pool->alive_flags[other_idx]) {
                diversity += compute_genome_distance(
                    &pool->entries[idx],
                    &pool->entries[other_idx],
                    archive,
                    archive_size
                );
            }
        }

        float genome_hunger = temp_genome[hunger_slot];
        float genome_mutation = temp_genome[mutation_slot];
        float genome_levy = temp_genome[levy_alpha_slot];
        float genome_diversity = temp_genome[diversity_slot];

        params.initial_hunger = fmaxf(0.01f, (genome_hunger + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE);
        params.mutation_scale = (genome_mutation + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * 0.1f;
        params.mutation_levy_alpha = CHEMOTAXIS_LEVY_ALPHA_MIN + (genome_levy + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * (CHEMOTAXIS_LEVY_ALPHA_MAX - CHEMOTAXIS_LEVY_ALPHA_MIN);
        params.diversity_normalization = 1.0f + (genome_diversity + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * 1000.0f;

        atomicAdd(genetic_diversity, diversity / params.diversity_normalization);
    }
}

// Device kernel to compact pool alive indices (CDP)
// Must be called before kernels that iterate only over alive entries
// Since POOL_CAPACITY_MAX = BLOCK_SIZE, single-block scan is sufficient
__global__ void compact_pool_alive_indices_kernel(
    ComponentPool* pool,
    int* flags,
    int* scan_workspace,
    int* scan_recursive_workspace,
    int capacity
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int blocks = (capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Step 1: Mark alive entries
    mark_alive_kernel<<<blocks, BLOCK_SIZE>>>(pool, flags);
    cudaDeviceSynchronize();

    // Step 2: Exclusive scan (device-side recursive for correctness)
    exclusive_scan_recursive_kernel<<<1, 1>>>(flags, scan_workspace, scan_recursive_workspace, capacity, BLOCK_SIZE);
    cudaDeviceSynchronize();

    // Step 3: Scatter indices
    scatter_alive_indices_kernel<<<blocks, BLOCK_SIZE>>>(pool, flags, scan_workspace);
    cudaDeviceSynchronize();

    // Step 4: Finalize count
    finalize_alive_count_kernel<<<1, 1>>>(pool, flags, scan_workspace, capacity);
    cudaDeviceSynchronize();
}

__global__ void collect_pool_task_accuracies_kernel(
    ComponentPool* pool,
    float* pool_task_accuracies
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < pool->capacity) {
        // Use SoA for coalesced alive read
        if (pool->alive_flags[tid]) {
            pool_task_accuracies[tid] = pool->entries[tid].task_accuracy;
        } else {
            pool_task_accuracies[tid] = 0.0f;
        }
    }
}

__global__ void inherit_ca_weights_kernel(
    ComponentPool* pool,
    int num_pending_inherits,
    int* child_indices,
    int* parent_indices
) {
    int inherit_idx = blockIdx.y;
    if (inherit_idx >= num_pending_inherits) return;

    int child_idx = child_indices[inherit_idx];
    int parent_idx = parent_indices[inherit_idx];

    PoolEntry* child = &pool->entries[child_idx];
    PoolEntry* parent = &pool->entries[parent_idx];

    // Use SoA for coalesced alive reads
    if (!pool->alive_flags[child_idx] || !pool->alive_flags[parent_idx]) return;
    if (!child->ca_state || !parent->ca_state) return;

    MultiHeadCAState* child_ca = child->ca_state;
    MultiHeadCAState* parent_ca = parent->ca_state;

    int perception_size = parent->num_heads * parent->channels * parent->head_dim;
    int interaction_size = parent->num_heads * parent->head_dim * parent->head_dim;
    int value_size = parent->num_heads * parent->head_dim * parent->channels;
    int total_size = perception_size + interaction_size + value_size;

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

__global__ void find_pending_weight_inherits_kernel(
    ComponentPool* pool,
    int* child_indices,
    int* parent_indices,
    int* num_pending
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pool->capacity) return;

    // Use SoA for coalesced alive read
    if (!pool->alive_flags[idx]) return;

    PoolEntry* entry = &pool->entries[idx];
    if (entry->age != 0) return;

    int p = entry->parent_idx;
    // Use SoA for coalesced alive read
    if (p != INT_MAX && p >= 0 && p < pool->capacity && pool->alive_flags[p]) {
        int slot = atomicAdd(num_pending, 1);
        child_indices[slot] = idx;
        parent_indices[slot] = p;
    }
}

#endif

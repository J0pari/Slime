
#ifndef POOL_CU
#define POOL_CU
#include "../config/config.cu"
#include "../utils/genome_params.cuh"
#include "../utils/tile_ops.cuh"
#include "../compression/delta.cu"
#include "../memory/archive.cu"
#include "../memory/genome_ops.cuh"
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
};

struct ComponentPool {
    PoolEntry* entries;
    cuda::atomic<int> active_count;
    cuda::atomic<int> total_spawned;
    cuda::atomic<int> total_culled;
    int capacity;
};

__device__ __forceinline__ void derive_architecture(uint64_t genome_hash, const float* genome, PoolEntry* entry) {
    int num_heads_slot = derive_param_slot(genome_hash, "arch_num_heads");
    int channels_slot = derive_param_slot(genome_hash, "arch_channels");
    int head_dim_slot = derive_param_slot(genome_hash, "arch_head_dim");
    int grid_size_slot = derive_param_slot(genome_hash, "arch_grid_size");

    float num_heads_norm = (genome[num_heads_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float channels_norm = (genome[channels_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float head_dim_norm = (genome[head_dim_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float grid_size_norm = (genome[grid_size_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;

    entry->num_heads = NUM_HEADS_MIN + (int)(num_heads_norm * (NUM_HEADS_MAX - NUM_HEADS_MIN));
    entry->channels = CHANNELS_MIN + (int)(channels_norm * (CHANNELS_MAX - CHANNELS_MIN));
    entry->head_dim = HEAD_DIM_MIN + (int)(head_dim_norm * (HEAD_DIM_MAX - HEAD_DIM_MIN));
    entry->grid_size = GRID_SIZE_MIN + (int)(grid_size_norm * (GRID_SIZE_MAX - GRID_SIZE_MIN));
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

    float replicas_norm = (genome[replicas_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float hidden1_norm = (genome[hidden1_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float hidden2_norm = (genome[hidden2_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float batch_size_norm = (genome[batch_size_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float anneal_step_norm = (genome[anneal_step_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float cov_target_norm = (genome[cov_target_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float dist_weight_norm = (genome[dist_weight_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float recon_weight_norm = (genome[recon_weight_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float distance_exponent_norm = (genome[distance_exponent_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float quality_weight_norm = (genome[quality_weight_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;

    entry->num_tempering_replicas = NUM_TEMPERING_REPLICAS_MIN + (int)(replicas_norm * (NUM_TEMPERING_REPLICAS_MAX - NUM_TEMPERING_REPLICAS_MIN));
    entry->diresa_hidden1 = DIRESA_HIDDEN1_MIN + (int)(hidden1_norm * (DIRESA_HIDDEN1_MAX - DIRESA_HIDDEN1_MIN));
    entry->diresa_hidden2 = DIRESA_HIDDEN2_MIN + (int)(hidden2_norm * (DIRESA_HIDDEN2_MAX - DIRESA_HIDDEN2_MIN));
    entry->diresa_batch_size = DIRESA_BATCH_SIZE_MIN + (int)(batch_size_norm * (DIRESA_BATCH_SIZE_MAX - DIRESA_BATCH_SIZE_MIN));
    entry->anneal_step = ANNEAL_STEP_MIN + anneal_step_norm * (ANNEAL_STEP_MAX - ANNEAL_STEP_MIN);
    entry->cov_target = COV_TARGET_MIN + cov_target_norm * (COV_TARGET_MAX - COV_TARGET_MIN);
    entry->dist_weight = DIST_WEIGHT_MIN + dist_weight_norm * (DIST_WEIGHT_MAX - DIST_WEIGHT_MIN);
    entry->recon_weight = RECON_WEIGHT_MIN + recon_weight_norm * (RECON_WEIGHT_MAX - RECON_WEIGHT_MIN);
    entry->distance_exponent = DIRESA_DISTANCE_EXPONENT_MIN + distance_exponent_norm * (DIRESA_DISTANCE_EXPONENT_MAX - DIRESA_DISTANCE_EXPONENT_MIN);
    entry->quality_weight = DIRESA_QUALITY_WEIGHT_MIN + quality_weight_norm * (DIRESA_QUALITY_WEIGHT_MAX - DIRESA_QUALITY_WEIGHT_MIN);
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
        uint64_t const y = s1;
        s0 = y;
        x ^= x << XORSHIFT_A;
        s1 = x ^ y ^ (x >> XORSHIFT_B) ^ (y >> XORSHIFT_C);
        return (s1 + y) / XORSHIFT_NORMALIZATION_SCALE;
    }

    __device__ float levy_stable(float alpha, float scale) {
        float U = next();
        float V = next();
        float W = next();

        float phi = TAU * (V - CENTERED_DIFFERENCE_SCALE);
        float quarter_tau_alpha = (TAU * QUARTER_SCALE) * alpha;
        float xi = (TAU * QUARTER_SCALE) - quarter_tau_alpha + phi;
        float alpha_phi = alpha * phi;

        float levy_num = sinf(alpha_phi);
        float levy_denom = powf(cosf(phi) + EPSILON, (NORMALIZED_MAX / alpha));
        float levy_factor = powf(cosf(xi) / (-logf(W + EPSILON) + EPSILON), ((NORMALIZED_MAX - alpha) / alpha));

        return (levy_num / levy_denom * levy_factor) * scale;
    }
};

__global__ void spawn_component_kernel(
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
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int count = Atomics::load_int(pool->active_count);

        if (count < pool->capacity) {
            int new_id = atomicAdd((int*)&pool->total_spawned, 1);

            for (int i = 0; i < pool->capacity; i++) {
                if (!pool->entries[i].alive) {
                    pool->entries[i].id = new_id;
                    pool->entries[i].fitness = DEFAULT_FITNESS;
                    pool->entries[i].coherence = DEFAULT_FITNESS;
                    pool->entries[i].task_accuracy = DEFAULT_FITNESS;
                    pool->entries[i].generalization_gap = DEFAULT_FITNESS;
                    pool->entries[i].hardware_efficiency = DEFAULT_FITNESS;
                    pool->entries[i].age = 0;
                    pool->entries[i].alive = true;

                    float* parent_genome = workspace_parent_genome;
                    float* child_genome = workspace_child_genome;

                    if (parent_id >= 0 && parent_id < pool->capacity) {
                        PoolEntry* parent = &pool->entries[parent_id];

                        int parent_archive_idx = -1;
                        for (int k = 0; k < archive_size; k++) {
                            if (archive->genome_hash[k] == parent->parent_hash) {
                                parent_archive_idx = k;
                                break;
                            }
                        }

                        if (parent_archive_idx >= 0 && archive->latent_genome) {
                            float* parent_latent = &archive->latent_genome[parent_archive_idx * GENOME_LATENT_DIM_MAX];

                            PRNGState rng;
                            rng.s0 = new_id * 0x9e3779b97f4a7c15ULL;
                            rng.s1 = parent_id * 0xbf58476d1ce4e5b9ULL;

                            reconstruct_genome_from_archive(parent->parent_hash, archive, archive_size, parent->delta_indices, parent->delta_values, parent->num_deltas, parent->max_deltas, parent_genome, GENOME_SIZE, workspace_parent_temp, diresa_genome_weights);

                            PoolInitParams parent_params;
                            parent_params.derive_from_genome(parent->genome_hash, parent_genome);

                            for (int j = 0; j < GENOME_LATENT_DIM_MAX; j++) {
                                float mutated_latent = parent_latent[j];

                                if (rng.next() < mutation_rate) {
                                    mutated_latent += rng.levy_stable(parent_params.mutation_levy_alpha, parent_params.mutation_scale);
                                    mutated_latent = tanhf(mutated_latent);
                                }

                                workspace_parent_temp[j] = mutated_latent;
                            }

                            diresa_decode(workspace_parent_temp, child_genome, diresa_genome_weights);
                        } else {
                            reconstruct_genome_from_archive(parent->parent_hash, archive, archive_size, parent->delta_indices, parent->delta_values, parent->num_deltas, parent->max_deltas, parent_genome, GENOME_SIZE, workspace_parent_temp, diresa_genome_weights);

                            PRNGState rng;
                            rng.s0 = new_id * 0x9e3779b97f4a7c15ULL;
                            rng.s1 = parent_id * 0xbf58476d1ce4e5b9ULL;

                            PoolInitParams parent_params;
                            parent_params.derive_from_genome(parent->genome_hash, parent_genome);

                            for (int j = 0; j < GENOME_SIZE; j++) {
                                child_genome[j] = parent_genome[j];

                                if (rng.next() < mutation_rate) {
                                    child_genome[j] += rng.levy_stable(parent_params.mutation_levy_alpha, parent_params.mutation_scale);
                                    child_genome[j] = tanhf(child_genome[j]);
                                }
                            }
                        }

                        pool->entries[i].parent_hash = parent->genome_hash;

                        // Fitness-weighted Lamarckian inheritance using genome-derived sigmoid
                        int inherit_center_slot = derive_param_slot(parent->genome_hash, "fitness_inherit_center");
                        int inherit_steep_slot = derive_param_slot(parent->genome_hash, "fitness_inherit_steepness");

                        float inherit_center = LIFECYCLE_FITNESS_INHERIT_CENTER_MIN +
                            ((parent_genome[inherit_center_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE) *
                            (LIFECYCLE_FITNESS_INHERIT_CENTER_MAX - LIFECYCLE_FITNESS_INHERIT_CENTER_MIN);

                        float inherit_steepness = LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MIN +
                            ((parent_genome[inherit_steep_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE) *
                            (LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MAX - LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MIN);

                        // Smooth sigmoid: inheritance_factor = 1 / (1 + exp(-steepness*(fitness - center)))
                        float inheritance_factor = 1.0f / (1.0f + expf(-inherit_steepness * (parent->fitness - inherit_center)));

                        for (int j = 0; j < GENOME_SIZE; j++) {
                            pool->entries[i].gradients[j] = parent->gradients[j] * inheritance_factor;
                        }
                    } else {
                        PRNGState rng;
                        rng.s0 = new_id * 0x9e3779b97f4a7c15ULL;
                        rng.s1 = new_id * 0xbf58476d1ce4e5b9ULL;

                        for (int j = 0; j < GENOME_SIZE; j++) {
                            child_genome[j] = rng.next() * GENOME_RANGE_SCALE + GENOME_VALUE_MIN;
                            pool->entries[i].gradients[j] = 0.0f;
                        }

                        pool->entries[i].parent_hash = 0;
                    }

                    pool->entries[i].genome_hash = gpu_sha256(child_genome, GENOME_SIZE);

                    pool->entries[i].num_deltas = 0;
                    int delta_threshold_slot = derive_param_slot(pool->entries[i].genome_hash, "delta_threshold");
                    float delta_threshold = (child_genome[delta_threshold_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * 0.01f;

                    for (int j = 0; j < GENOME_SIZE; j++) {
                        float diff = child_genome[j] - parent_genome[j];
                        if (fabsf(diff) > delta_threshold && pool->entries[i].num_deltas < pool->entries[i].max_deltas) {
                            int delta_idx = pool->entries[i].num_deltas++;
                            pool->entries[i].delta_indices[delta_idx] = j;
                            pool->entries[i].delta_values[delta_idx] = diff;
                        }
                    }

                    PoolInitParams init_params;
                    init_params.derive_from_genome(pool->entries[i].genome_hash, child_genome);
                    pool->entries[i].hunger = init_params.initial_hunger;

                    derive_architecture(pool->entries[i].genome_hash, child_genome, &pool->entries[i]);
                    derive_diresa(pool->entries[i].genome_hash, child_genome, &pool->entries[i]);
                    derive_fitness_exponents(pool->entries[i].genome_hash, child_genome, &pool->entries[i]);

                    Atomics::increment_int(pool->active_count);
                    break;
                }
            }
        }
    }
}

__global__ void cull_weak_kernel(
    ComponentPool* pool,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity) {
        PoolEntry* entry = &pool->entries[idx];

        if (entry->alive && entry->fitness < threshold) {
            entry->alive = false;
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
        PoolEntry* entry = &pool->entries[idx];

        if (entry->alive && entry->hunger > hunger_threshold) {
            entry->alive = false;
            Atomics::decrement_int(pool->active_count);
            Atomics::increment_int(pool->total_culled);
        }
    }
}

__global__ void age_components_kernel(ComponentPool* pool) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity && pool->entries[idx].alive) {
        pool->entries[idx].age++;
    }
}

__global__ void update_hunger_kernel(ComponentPool* pool) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity && pool->entries[idx].alive) {

        pool->entries[idx].hunger = 1.0f - pool->entries[idx].coherence;
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

        if (pool->entries[tid].alive) {
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
    // Compute distance in DIRESA latent space (much faster than full genome)
    int idx1 = -1, idx2 = -1;
    for (int i = 0; i < archive_size; i++) {
        if (archive->genome_hash[i] == entry1->parent_hash) idx1 = i;
        if (archive->genome_hash[i] == entry2->parent_hash) idx2 = i;
    }

    if (idx1 < 0 || idx2 < 0) return 1e10f;  // Not in archive

    // Use vectorized DIRESAOps warp-reduce distance (128D -> 32 float4 loads)
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

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < pool->capacity && pool->entries[idx].alive) {

        float avg_distance = 0.0f;
        int count = 0;

        for (int i = 0; i < pool->capacity; i++) {
            if (i != idx && pool->entries[i].alive) {

                float dist = compute_genome_distance(
                    &pool->entries[idx],
                    &pool->entries[i],
                    archive,
                    archive_size
                );
                avg_distance += dist;
                count++;
            }
        }

        distances[tid] = count > 0 ? avg_distance / count : 0.0f;
    } else {
        distances[tid] = -1.0f;
    }

    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < num_select && i < blockDim.x; i++) {
            float max_dist = -1.0f;
            int max_idx = -1;

            for (int j = 0; j < blockDim.x; j++) {
                if (distances[j] > max_dist) {
                    max_dist = distances[j];
                    max_idx = j;
                }
            }

            if (max_idx >= 0) {
                selected_indices[i] = blockIdx.x * blockDim.x + max_idx;
                distances[max_idx] = -1.0f;
            }
        }
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

    if (idx == 0) printf("[KERNEL] Thread 0 started\n");

    if (idx < capacity) {

        PRNGState rng;
        rng.s0 = idx * 0x9e3779b97f4a7c15ULL;
        rng.s1 = idx * 0xbf58476d1ce4e5b9ULL;

        pool->entries[idx].max_deltas = GENOME_SIZE;
        pool->entries[idx].delta_indices = &delta_indices_buffer[idx * GENOME_SIZE];
        pool->entries[idx].delta_values = &delta_values_buffer[idx * GENOME_SIZE];
        pool->entries[idx].gradients = &gradients_buffer[idx * GENOME_SIZE];

        if (idx == 0) printf("[KERNEL] Buffers assigned\n");

        if (idx < MIN_POOL_SIZE) {
            pool->entries[idx].id = idx;
            pool->entries[idx].alive = true;
            pool->entries[idx].fitness = DEFAULT_FITNESS;
            pool->entries[idx].coherence = DEFAULT_FITNESS;
            pool->entries[idx].task_accuracy = DEFAULT_FITNESS;
            pool->entries[idx].generalization_gap = DEFAULT_FITNESS;
            pool->entries[idx].hardware_efficiency = DEFAULT_FITNESS;
            pool->entries[idx].age = 0;
            pool->entries[idx].parent_hash = 0;
            pool->entries[idx].num_deltas = GENOME_SIZE;

            if (idx == 0) printf("[KERNEL] Metadata initialized\n");

            float temp_genome[GENOME_SIZE];
            for (int i = 0; i < GENOME_SIZE; i++) {
                temp_genome[i] = rng.next() * GENOME_RANGE_SCALE + GENOME_VALUE_MIN;
                pool->entries[idx].delta_indices[i] = i;
                pool->entries[idx].delta_values[i] = temp_genome[i];
                pool->entries[idx].gradients[i] = 0.0f;
            }

            if (idx == 0) printf("[KERNEL] temp_genome filled, calling gpu_sha256\n");

            pool->entries[idx].genome_hash = gpu_sha256(temp_genome, GENOME_SIZE);

            if (idx == 0) printf("[KERNEL] SHA256 done, calling derive_from_genome\n");

            PoolInitParams init_params;
            init_params.derive_from_genome(pool->entries[idx].genome_hash, temp_genome);
            pool->entries[idx].hunger = init_params.initial_hunger;

            if (idx == 0) printf("[KERNEL] derive_from_genome done, calling derive_architecture\n");

            derive_architecture(pool->entries[idx].genome_hash, temp_genome, &pool->entries[idx]);

            if (idx == 0) printf("[KERNEL] derive_architecture done, calling derive_diresa\n");

            derive_diresa(pool->entries[idx].genome_hash, temp_genome, &pool->entries[idx]);

            if (idx == 0) printf("[KERNEL] derive_diresa done, calling derive_fitness_exponents\n");

            derive_fitness_exponents(pool->entries[idx].genome_hash, temp_genome, &pool->entries[idx]);

            if (idx == 0) printf("[KERNEL] derive_fitness_exponents done\n");
        } else {
            pool->entries[idx].id = -1;
            pool->entries[idx].alive = false;
            pool->entries[idx].fitness = DEFAULT_FITNESS;
            pool->entries[idx].coherence = DEFAULT_FITNESS;
            pool->entries[idx].task_accuracy = DEFAULT_FITNESS;
            pool->entries[idx].generalization_gap = DEFAULT_FITNESS;
            pool->entries[idx].hardware_efficiency = DEFAULT_FITNESS;
            pool->entries[idx].hunger = DEFAULT_FITNESS;
            pool->entries[idx].age = 0;
            pool->entries[idx].parent_hash = 0;
            pool->entries[idx].num_deltas = 0;

            for (int i = 0; i < GENOME_SIZE; i++) {
                pool->entries[idx].gradients[i] = 0.0f;
            }
        }
    }

    if (idx == 0) {
        printf("[KERNEL] About to set atomic fields\n");
        pool->capacity = capacity;
        pool->active_count = MIN_POOL_SIZE;
        pool->total_spawned = MIN_POOL_SIZE;
        pool->total_culled = 0;
        printf("[KERNEL] Atomic fields set, kernel complete\n");
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

    if (idx < pool->capacity && pool->entries[idx].alive) {
        local_fitness = pool->entries[idx].fitness;
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

    if (idx < pool->capacity && pool->entries[idx].alive) {
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
            if (other_idx != idx && pool->entries[other_idx].alive) {
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

        params.initial_hunger = (genome_hunger + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        params.mutation_scale = (genome_mutation + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * 0.1f;
        params.mutation_levy_alpha = CHEMOTAXIS_LEVY_ALPHA_MIN + (genome_levy + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * (CHEMOTAXIS_LEVY_ALPHA_MAX - CHEMOTAXIS_LEVY_ALPHA_MIN);
        params.diversity_normalization = 1.0f + (genome_diversity + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * 1000.0f;

        atomicAdd(genetic_diversity, diversity / params.diversity_normalization);
    }
}

#endif

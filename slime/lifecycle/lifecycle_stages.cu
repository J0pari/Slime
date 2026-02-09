#ifndef LIFECYCLE_STAGES_CU
#define LIFECYCLE_STAGES_CU

#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
#include "../memory/pool.cu"
#include "../memory/archive.cu"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;



template<int SECTION_SIZE>
struct LocalOrganismState {
    
    
    int organism_indices[SECTION_SIZE];  
    float local_fitness[SECTION_SIZE];
    float local_coherence[SECTION_SIZE];
    float gradient_history[SECTION_SIZE][8];
    LifecyclePhase phases[SECTION_SIZE];
    float stress_accum[SECTION_SIZE];

    __device__ void observe(int idx, const ComponentPool* pool, int generation) {
        if (idx >= SECTION_SIZE) return;

        organism_indices[idx] = idx;
        local_fitness[idx] = pool->fitness_values[idx];
        local_coherence[idx] = pool->entries[idx].coherence.value;

        for (int i = 7; i > 0; i--) {
            gradient_history[idx][i] = gradient_history[idx][i-1];
        }

        float grad_mag = 0.0f;
        for (int g = 0; g < GENOME_SIZE && g < 100; g++) {
            float grad = pool->entries[idx].gradients[g];
            grad_mag += grad * grad;
        }
        gradient_history[idx][0] = sqrtf(grad_mag / 100.0f);
    }

    __device__ LifecyclePhase decide_transition(int idx, bool alive, uint64_t genome_hash, const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        if (idx >= SECTION_SIZE || !alive) return LifecyclePhase::DORMANT;

        LifecyclePhase current = phases[idx];
        float fitness = local_fitness[idx];
        float coherence = local_coherence[idx];

        
        int coherence_stressed_slot = GenomeParamTable::lifecycle_coherence_stressed;
        int coherence_recover_slot = GenomeParamTable::lifecycle_coherence_recover;
        int stress_accum_rate_slot = GenomeParamTable::lifecycle_stress_accum_rate;
        int stress_decay_rate_slot = GenomeParamTable::lifecycle_stress_decay_rate;
        int stress_threshold_slot = GenomeParamTable::lifecycle_stress_threshold;
        int fitness_multiplier_slot = GenomeParamTable::lifecycle_fitness_multiplier;
        int gradient_stagnation_slot = GenomeParamTable::lifecycle_gradient_stagnation;
        int dormant_stress_mult_slot = GenomeParamTable::lifecycle_dormant_stress_mult;
        int fitness_threshold_center_slot = GenomeParamTable::lifecycle_fitness_threshold_center;
        int fitness_threshold_steepness_slot = GenomeParamTable::lifecycle_fitness_threshold_steepness;
        int sigmoid_threshold_slot = GenomeParamTable::lifecycle_sigmoid_threshold;


        float coherence_stressed = genome_to_param(genome, gradients, coherence_stressed_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_COHERENCE_STRESSED_MIN, LIFECYCLE_COHERENCE_STRESSED_MAX);
        float coherence_recover = genome_to_param(genome, gradients, coherence_recover_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_COHERENCE_RECOVER_MIN, LIFECYCLE_COHERENCE_RECOVER_MAX);
        float stress_accum_rate = genome_to_param(genome, gradients, stress_accum_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_STRESS_ACCUM_RATE_MIN, LIFECYCLE_STRESS_ACCUM_RATE_MAX);
        float stress_decay_rate = genome_to_param(genome, gradients, stress_decay_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_STRESS_DECAY_RATE_MIN, LIFECYCLE_STRESS_DECAY_RATE_MAX);
        float stress_threshold = genome_to_param(genome, gradients, stress_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_STRESS_THRESHOLD_MIN, LIFECYCLE_STRESS_THRESHOLD_MAX);
        float fitness_multiplier = genome_to_param(genome, gradients, fitness_multiplier_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_FITNESS_MULTIPLIER_MIN, LIFECYCLE_FITNESS_MULTIPLIER_MAX);
        float gradient_stagnation_thresh = genome_to_param(genome, gradients, gradient_stagnation_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_GRADIENT_STAGNATION_MIN, LIFECYCLE_GRADIENT_STAGNATION_MAX);
        float dormant_stress_mult = genome_to_param(genome, gradients, dormant_stress_mult_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_DORMANT_STRESS_MULT_MIN, LIFECYCLE_DORMANT_STRESS_MULT_MAX);
        float fitness_threshold_center = genome_to_param(genome, gradients, fitness_threshold_center_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_FITNESS_THRESHOLD_CENTER_MIN, LIFECYCLE_FITNESS_THRESHOLD_CENTER_MAX);
        float fitness_threshold_steepness = genome_to_param(genome, gradients, fitness_threshold_steepness_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_FITNESS_THRESHOLD_STEEPNESS_MIN, LIFECYCLE_FITNESS_THRESHOLD_STEEPNESS_MAX);
        float sigmoid_threshold = genome_to_param(genome, gradients, sigmoid_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_SIGMOID_THRESHOLD_MIN, LIFECYCLE_SIGMOID_THRESHOLD_MAX);

        float recent_grad_avg = 0.0f;
        for (int i = 0; i < 4; i++) {
            recent_grad_avg += gradient_history[idx][i];
        }
        recent_grad_avg /= 4.0f;

        float older_grad_avg = 0.0f;
        for (int i = 4; i < 8; i++) {
            older_grad_avg += gradient_history[idx][i];
        }
        older_grad_avg /= 4.0f;

        bool gradient_stagnant = (recent_grad_avg < older_grad_avg * gradient_stagnation_thresh);
        float fitness_sigmoid = activation_sigmoid(fitness_threshold_steepness * (fitness - fitness_threshold_center * fitness_multiplier));
        bool fitness_plateaued = (fitness_sigmoid < sigmoid_threshold);
        bool learning_stopped = (coherence < coherence_stressed);

        switch (current) {
            case LifecyclePhase::ACTIVE:
                if (gradient_stagnant || fitness_plateaued) {
                    stress_accum[idx] += stress_accum_rate;
                    if (stress_accum[idx] > stress_threshold) {
                        return LifecyclePhase::STRESSED;
                    }
                } else {
                    stress_accum[idx] = fmaxf(0.0f, stress_accum[idx] - stress_decay_rate);
                }
                break;

            case LifecyclePhase::STRESSED:
                if (learning_stopped || stress_accum[idx] > stress_threshold * dormant_stress_mult) {
                    return LifecyclePhase::DORMANT;
                } else if (coherence > coherence_recover && !gradient_stagnant) {
                    stress_accum[idx] = 0.0f;
                    return LifecyclePhase::ACTIVE;
                } else {
                    stress_accum[idx] += stress_accum_rate;
                }
                break;

            case LifecyclePhase::DORMANT:

                break;

            case LifecyclePhase::REACTIVATING:

                stress_accum[idx] = 0.0f;
                return LifecyclePhase::ACTIVE;
        }

        return current;
    }
};

__device__ int sample_from_niche_aware(
    GPUElite* archive,
    int archive_size,
    VoronoiCell* voronoi_cells,
    int num_cells,
    const float* hw_coords,
    const float* task_coords,
    const float* gen_coords,
    curandState* rand_state,
    float density_multiplier
) {
    DEVICE_FATAL_IF(archive_size <= 0, "sample_from_niche_aware: empty archive");
    DEVICE_FATAL_IF(num_cells <= 0, "sample_from_niche_aware: no voronoi cells");

    float min_dist = 1e9f;
    int closest_cell = 0;

    for (int c = 0; c < num_cells; c++) {
        float dist = compute_three_axis_distance_sq(
            hw_coords,
            task_coords,
            gen_coords,
            voronoi_cells[c].hw_centroid,
            voronoi_cells[c].task_centroid,
            voronoi_cells[c].gen_centroid,
            archive->hw_dim,
            archive->task_dim,
            archive->gen_dim
        );

        if (dist < min_dist) {
            min_dist = dist;
            closest_cell = c;
        }
    }

    int sparse_neighbors[MAX_SPARSE_NEIGHBORS];
    int num_sparse = 0;
    int target_cell_density = voronoi_cells[closest_cell].density;

    for (int c = 0; c < num_cells && num_sparse < MAX_SPARSE_NEIGHBORS; c++) {
        if (voronoi_cells[c].density < target_cell_density * density_multiplier) {
            sparse_neighbors[num_sparse++] = c;
        }
    }

    int selected_cell;
    if (num_sparse == 0) {
        selected_cell = closest_cell;
    } else {
        selected_cell = sparse_neighbors[curand(rand_state) % num_sparse];
    }

    int elite_idx = voronoi_cells[selected_cell].best_elite_idx;
    DEVICE_FATAL_IF(elite_idx < 0 || elite_idx >= archive_size,
        "sample_from_niche_aware: cell %d has invalid best_elite_idx %d",
        selected_cell, elite_idx);

    return elite_idx;
}

extern "C" __global__ void lifecycle_transition_kernel(
    ComponentPool* pool,
    GPUElite* archive,
    int archive_size,
    VoronoiCell* voronoi_cells,
    int num_cells,
    BehavioralState* behavioral_agents,
    LocalOrganismState<BLOCK_SIZE>* sections,
    int generation,
    ChemicalField* chemical_field,
    int grid_size,
    TelemetryBuffer* telemetry,
    float* workspace_genomes,
    DIRESAWeights* diresa_genome_weights
) {
    int compact_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int alive_count = pool->alive_indices_count;
    if (compact_idx >= alive_count) return;

    int idx = pool->alive_indices[compact_idx];
    PoolEntry* entry = &pool->entries[idx];
    DEVICE_FATAL_IF(!pool->alive_flags[idx], "lifecycle_transition_kernel: dead entry in alive_indices");

    float ctx_complexity = telemetry->genome_complexity.hash_entropy;
    float ctx_niche = telemetry->archive_topology.novelty_gradient;
    float ctx_learning = telemetry->diresa_evolution.behavioral_drift_rate;
    float ctx_performance = telemetry->task_performance.accuracy;

    __shared__ LocalOrganismState<BLOCK_SIZE> local_state;

    int local_idx = threadIdx.x;
    int threads_in_block = min((int)blockDim.x, alive_count - blockIdx.x * blockDim.x);
    if (local_idx < threads_in_block) {
        local_state.organism_indices[local_idx] = idx;
        local_state.local_fitness[local_idx] = pool->fitness_values[idx];
        local_state.local_coherence[local_idx] = entry->coherence.value;
    }
    __syncthreads();

    float* entry_genome = &workspace_genomes[compact_idx * GENOME_SIZE * 2];
    float* parent_genome_temp = &workspace_genomes[compact_idx * GENOME_SIZE * 2 + GENOME_SIZE];

    reconstruct_genome_from_archive(entry->parent_hash, archive, archive_size,
        entry->delta_indices, entry->delta_values, entry->num_deltas,
        entry->max_deltas, entry_genome, GENOME_SIZE, parent_genome_temp, diresa_genome_weights);

    float ctx_metabolic = entry->fitness.value;
    float ctx_stress = entry->hunger.value;
    float ctx_morphogen = sample_neighborhood(chemical_field->concentration, idx, grid_size);

    LifecyclePhase new_phase = local_state.decide_transition(local_idx, true, entry->genome_hash, entry_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);

    if (new_phase == LifecyclePhase::DORMANT) {
        int archive_threshold_center_slot = GenomeParamTable::lifecycle_archive_threshold_center;
        int archive_threshold_steepness_slot = GenomeParamTable::lifecycle_archive_threshold_steepness;
        int sigmoid_threshold_slot = GenomeParamTable::lifecycle_sigmoid_threshold;
        float archive_threshold_center = genome_to_param(entry_genome, entry->gradients, archive_threshold_center_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_FITNESS_THRESHOLD_CENTER_MIN, LIFECYCLE_FITNESS_THRESHOLD_CENTER_MAX);
        float archive_threshold_steepness = genome_to_param(entry_genome, entry->gradients, archive_threshold_steepness_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_FITNESS_THRESHOLD_STEEPNESS_MIN, LIFECYCLE_FITNESS_THRESHOLD_STEEPNESS_MAX);
        float sigmoid_threshold = genome_to_param(entry_genome, entry->gradients, sigmoid_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_SIGMOID_THRESHOLD_MIN, LIFECYCLE_SIGMOID_THRESHOLD_MAX);
        float archive_prob = activation_sigmoid(archive_threshold_steepness * (entry->fitness.value - archive_threshold_center));

        if (archive_prob > sigmoid_threshold && archive_size < MAX_ARCHIVE_SIZE) {
            pool->alive_flags[idx] = false;
            Atomics::increment_int(pool->total_culled);
            Atomics::decrement_int(pool->active_count);
        }
    }

    if (new_phase == LifecyclePhase::REACTIVATING) {
        DEVICE_FATAL_IF(archive_size <= 0, "lifecycle_transition_kernel: reactivating with empty archive");

        curandState rand_state;
        curand_init(generation * pool->capacity + idx, 0, 0, &rand_state);

        int density_mult_slot = GenomeParamTable::lifecycle_density_multiplier;
        float density_multiplier = genome_to_param(entry_genome, entry->gradients, density_mult_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_DENSITY_MULTIPLIER_MIN, LIFECYCLE_DENSITY_MULTIPLIER_MAX);

        int elite_idx = sample_from_niche_aware(
            archive, archive_size, voronoi_cells, num_cells,
            behavioral_agents[idx].hw_coords,
            behavioral_agents[idx].task_coords,
            behavioral_agents[idx].gen_coords,
            &rand_state,
            density_multiplier
        );

        int hw_dim = archive->hw_dim;
        int task_dim = archive->task_dim;
        int gen_dim = archive->gen_dim;

        restore_fitness_from_archive(entry, archive, elite_idx);
        entry->coherence.value = archive->coherence[elite_idx];
        entry->coherence.state = ComputeState::COMPUTED;
        entry->coherence.computed_at_generation = archive->fitness_computed_at_generation[elite_idx];
        entry->hunger.value = NORMALIZED_MAX - archive->coherence[elite_idx];
        entry->hunger.state = ComputeState::COMPUTED;
        entry->hunger.computed_at_generation = generation;
        entry->generation = generation;
        entry->age = 0;
        entry->genome_hash = archive->genome_hash[elite_idx];

        for (int d = 0; d < hw_dim; d++) {
            behavioral_agents[idx].hw_coords[d] = archive->hw_coords[elite_idx * hw_dim + d];
        }
        for (int d = 0; d < task_dim; d++) {
            behavioral_agents[idx].task_coords[d] = archive->task_coords[elite_idx * task_dim + d];
        }
        for (int d = 0; d < gen_dim; d++) {
            behavioral_agents[idx].gen_coords[d] = archive->gen_coords[elite_idx * gen_dim + d];
        }

        local_state.phases[local_idx] = LifecyclePhase::REACTIVATING;
    } else {
        local_state.phases[local_idx] = new_phase;
    }

    if (new_phase == LifecyclePhase::STRESSED) {
        int stress_penalty_slot = GenomeParamTable::lifecycle_stress_fitness_penalty;
        float stress_penalty = genome_to_param(entry_genome, entry->gradients, stress_penalty_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_STRESS_FITNESS_PENALTY_MIN, LIFECYCLE_STRESS_FITNESS_PENALTY_MAX);
        float new_fitness = entry->fitness.value * stress_penalty;
        uint64_t mod_hash = entry->fitness.input_hash;
        mod_hash ^= __float_as_uint(stress_penalty) + 0x9e3779b9 + (mod_hash << 6) + (mod_hash >> 2);
        entry->fitness.set_computed(new_fitness, generation, mod_hash);
        pool->fitness_values[idx] = new_fitness;
    }
}

extern "C" __global__ void hierarchical_lifecycle_kernel(
    ComponentPool* pool,
    LocalOrganismState<BLOCK_SIZE>* thread_sections,
    GPUElite* archive,
    int* archive_size,
    VoronoiCell* voronoi_cells,
    int num_cells,
    int generation,
    ChemicalField* chemical_field,
    int grid_size,
    TelemetryBuffer* telemetry,
    float* workspace_genomes,
    DIRESAWeights* diresa_genome_weights
) {
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int compact_idx = block_id * blockDim.x + tid;
    int alive_count = pool->alive_indices_count;

    
    
    int threads_in_block = min((int)blockDim.x, alive_count - block_id * blockDim.x);
    DEVICE_FATAL_IF(threads_in_block <= 0, "hierarchical_lifecycle_kernel: block launched with no valid threads");

    bool has_organism = tid < threads_in_block;
    DEVICE_FATAL_IF(!has_organism && compact_idx < alive_count,
        "hierarchical_lifecycle_kernel: thread validity invariant violated");

    float ctx_complexity = telemetry->genome_complexity.hash_entropy;
    float ctx_niche = telemetry->archive_topology.novelty_gradient;
    float ctx_learning = telemetry->diresa_evolution.behavioral_drift_rate;
    float ctx_performance = telemetry->task_performance.accuracy;

    LocalOrganismState<BLOCK_SIZE>& local_state = thread_sections[block_id];

    __shared__ int shared_actual_idx[BLOCK_SIZE];

    if (has_organism) {
        int actual_idx = pool->alive_indices[compact_idx];
        shared_actual_idx[tid] = actual_idx;
        PoolEntry* entry = &pool->entries[actual_idx];

        local_state.organism_indices[tid] = actual_idx;
        local_state.local_fitness[tid] = entry->fitness.value;
        local_state.local_coherence[tid] = entry->coherence.value;
        for (int i = 7; i > 0; i--) {
            local_state.gradient_history[tid][i] = local_state.gradient_history[tid][i-1];
        }
        float grad_mag = 0.0f;
        for (int g = 0; g < GENOME_SIZE && g < 100; g++) {
            float grad = entry->gradients[g];
            grad_mag += grad * grad;
        }
        local_state.gradient_history[tid][0] = sqrtf(grad_mag / 100.0f);
    }
    __syncthreads();

    if (has_organism) {
        int actual_idx = shared_actual_idx[tid];
        PoolEntry* entry = &pool->entries[actual_idx];

        float* entry_genome = &workspace_genomes[compact_idx * GENOME_SIZE * 2];
        float* parent_genome_temp = &workspace_genomes[compact_idx * GENOME_SIZE * 2 + GENOME_SIZE];

        reconstruct_genome_from_archive(entry->parent_hash, archive, *archive_size,
            entry->delta_indices, entry->delta_values, entry->num_deltas,
            entry->max_deltas, entry_genome, GENOME_SIZE, parent_genome_temp, diresa_genome_weights);

        float ctx_metabolic = entry->fitness.value;
        float ctx_stress = entry->hunger.value;
        float ctx_morphogen = sample_neighborhood(chemical_field->concentration, actual_idx, grid_size);

        uint64_t entry_hash = entry->genome_hash;
        const float* entry_gradients = entry->gradients;

        LifecyclePhase new_phase = local_state.decide_transition(tid, true, entry_hash, entry_genome, entry_gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
        (void)new_phase;
    }

    __shared__ float shared_fitness[BLOCK_SIZE];
    __shared__ float shared_coherence[BLOCK_SIZE];
    __shared__ float shared_stress[BLOCK_SIZE];
    __shared__ float shared_morphogen[BLOCK_SIZE];
    __shared__ float block_avg_fitness;
    __shared__ float block_avg_coherence;
    __shared__ float block_avg_stress;
    __shared__ float block_avg_morphogen;
    __shared__ int block_active_count;

    if (has_organism) {
        int actual_idx = shared_actual_idx[tid];
        PoolEntry* entry = &pool->entries[actual_idx];
        shared_fitness[tid] = local_state.local_fitness[tid];
        shared_coherence[tid] = local_state.local_coherence[tid];
        shared_stress[tid] = entry->hunger.value;
        shared_morphogen[tid] = sample_neighborhood(chemical_field->concentration, actual_idx, grid_size);
    }
    __syncthreads();

    if (tid == 0) {
        float sum_fitness = 0.0f;
        float sum_coherence = 0.0f;
        float sum_stress = 0.0f;
        float sum_morphogen = 0.0f;
        for (int i = 0; i < threads_in_block; i++) {
            sum_fitness += shared_fitness[i];
            sum_coherence += shared_coherence[i];
            sum_stress += shared_stress[i];
            sum_morphogen += shared_morphogen[i];
        }
        block_avg_fitness = sum_fitness / threads_in_block;
        block_avg_coherence = sum_coherence / threads_in_block;
        block_avg_stress = sum_stress / threads_in_block;
        block_avg_morphogen = sum_morphogen / threads_in_block;
        block_active_count = threads_in_block;
    }
    __syncthreads();

    int block_leader_compact = block_id * blockDim.x;
    int block_leader_actual = pool->alive_indices[block_leader_compact];
    uint64_t block_genome_hash = pool->entries[block_leader_actual].genome_hash;

    float* block_genome = &workspace_genomes[block_leader_compact * GENOME_SIZE * 2];
    const float* block_gradients = pool->entries[block_leader_actual].gradients;

    float block_ctx_metabolic = block_avg_fitness;
    float block_ctx_stress = block_avg_stress;
    float block_ctx_morphogen = block_avg_morphogen;

    int boost_threshold_center_slot = GenomeParamTable::lifecycle_boost_threshold_center;
    int boost_threshold_steepness_slot = GenomeParamTable::lifecycle_boost_threshold_steepness;
    int crisis_fitness_mult_slot = GenomeParamTable::lifecycle_crisis_fitness_mult;
    int crisis_coherence_slot = GenomeParamTable::lifecycle_crisis_coherence;
    int crisis_threshold_center_slot = GenomeParamTable::lifecycle_crisis_threshold_center;
    int crisis_threshold_steepness_slot = GenomeParamTable::lifecycle_crisis_threshold_steepness;
    int elite_fitness_inherit_slot = GenomeParamTable::lifecycle_elite_fitness_inherit;
    int elite_coherence_reset_slot = GenomeParamTable::lifecycle_elite_coherence_reset;
    int sigmoid_threshold_slot = GenomeParamTable::lifecycle_sigmoid_threshold;
    int boost_fitness_ratio_slot = GenomeParamTable::lifecycle_boost_fitness_ratio;
    int coherence_boost_slot = GenomeParamTable::lifecycle_coherence_boost;
    int crisis_active_ratio_slot = GenomeParamTable::lifecycle_crisis_active_ratio;

    float boost_threshold_center = genome_to_param(block_genome, block_gradients, boost_threshold_center_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_BOOST_THRESHOLD_CENTER_MIN, LIFECYCLE_BOOST_THRESHOLD_CENTER_MAX);
    float boost_threshold_steepness = genome_to_param(block_genome, block_gradients, boost_threshold_steepness_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_BOOST_THRESHOLD_STEEPNESS_MIN, LIFECYCLE_BOOST_THRESHOLD_STEEPNESS_MAX);
    float crisis_fitness_mult = genome_to_param(block_genome, block_gradients, crisis_fitness_mult_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_CRISIS_FITNESS_MULT_MIN, LIFECYCLE_CRISIS_FITNESS_MULT_MAX);
    float crisis_coherence = genome_to_param(block_genome, block_gradients, crisis_coherence_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_CRISIS_COHERENCE_MIN, LIFECYCLE_CRISIS_COHERENCE_MAX);
    float crisis_threshold_center = genome_to_param(block_genome, block_gradients, crisis_threshold_center_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_CRISIS_THRESHOLD_CENTER_MIN, LIFECYCLE_CRISIS_THRESHOLD_CENTER_MAX);
    float crisis_threshold_steepness = genome_to_param(block_genome, block_gradients, crisis_threshold_steepness_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_CRISIS_THRESHOLD_STEEPNESS_MIN, LIFECYCLE_CRISIS_THRESHOLD_STEEPNESS_MAX);
    float elite_fitness_inherit = genome_to_param(block_genome, block_gradients, elite_fitness_inherit_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_ELITE_FITNESS_INHERIT_MIN, LIFECYCLE_ELITE_FITNESS_INHERIT_MAX);
    float elite_coherence_reset = genome_to_param(block_genome, block_gradients, elite_coherence_reset_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_ELITE_COHERENCE_RESET_MIN, LIFECYCLE_ELITE_COHERENCE_RESET_MAX);
    float sigmoid_threshold = genome_to_param(block_genome, block_gradients, sigmoid_threshold_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_SIGMOID_THRESHOLD_MIN, LIFECYCLE_SIGMOID_THRESHOLD_MAX);
    float boost_fitness_ratio = genome_to_param(block_genome, block_gradients, boost_fitness_ratio_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_BOOST_FITNESS_RATIO_MIN, LIFECYCLE_BOOST_FITNESS_RATIO_MAX);
    float coherence_boost = genome_to_param(block_genome, block_gradients, coherence_boost_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_COHERENCE_BOOST_MIN, LIFECYCLE_COHERENCE_BOOST_MAX);
    float crisis_active_ratio = genome_to_param(block_genome, block_gradients, crisis_active_ratio_slot, block_ctx_metabolic, block_ctx_stress, block_ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_CRISIS_ACTIVE_RATIO_MIN, LIFECYCLE_CRISIS_ACTIVE_RATIO_MAX);

    float boost_prob = activation_sigmoid(boost_threshold_steepness * (block_avg_fitness - boost_threshold_center));
    if (has_organism && boost_prob > sigmoid_threshold) {
        int actual_idx = shared_actual_idx[tid];
        PoolEntry* entry = &pool->entries[actual_idx];
        float my_fitness = shared_fitness[tid];
        float my_coherence = shared_coherence[tid];
        if (my_fitness < block_avg_fitness * boost_fitness_ratio) {
            entry->coherence.value = fminf(NORMALIZED_MAX, my_coherence + coherence_boost);
        }
    }

    float crisis_fitness_prob = activation_sigmoid(crisis_threshold_steepness * (block_avg_fitness - crisis_threshold_center * crisis_fitness_mult));
    __shared__ bool block_in_crisis;
    if (tid == 0) {
        int min_active_count = (int)(BLOCK_SIZE * crisis_active_ratio);
        block_in_crisis = (crisis_fitness_prob < sigmoid_threshold) ||
                          (block_avg_coherence < crisis_coherence) ||
                          (block_active_count < min_active_count);
    }
    __syncthreads();

    if (block_in_crisis && tid == 0) {
        DEVICE_FATAL_IF(*archive_size <= 0, "hierarchical_lifecycle_kernel: crisis with empty archive");

        atomicAdd((int*)&pool->total_culled, 1);

        unsigned int seed = block_id * POOL_CAPACITY_MAX + generation;
        curandState_t rand_state;
        curand_init(seed, 0, 0, &rand_state);

        int sample_idx = (int)(curand_uniform(&rand_state) * (*archive_size)) % (*archive_size);
        DEVICE_FATAL_IF(sample_idx < 0 || sample_idx >= *archive_size,
            "hierarchical_lifecycle_kernel: sample_idx out of bounds");

        int worst_tid = 0;
        float worst_fitness = shared_fitness[0];
        for (int i = 1; i < threads_in_block; i++) {
            if (shared_fitness[i] < worst_fitness) {
                worst_fitness = shared_fitness[i];
                worst_tid = i;
            }
        }

        int worst_actual = shared_actual_idx[worst_tid];
        local_state.phases[worst_tid] = LifecyclePhase::REACTIVATING;
        float new_fitness = archive->fitness[sample_idx] * elite_fitness_inherit;
        
        pool->entries[worst_actual].fitness.value = new_fitness;
        pool->entries[worst_actual].fitness.state = ComputeState::COMPUTED;
        pool->entries[worst_actual].fitness.computed_at_generation = generation;
        pool->entries[worst_actual].fitness.input_hash = archive->fitness_input_hash[sample_idx];
        pool->fitness_values[worst_actual] = new_fitness;
        pool->entries[worst_actual].coherence.value = elite_coherence_reset;
        pool->entries[worst_actual].coherence.state = ComputeState::COMPUTED;
        pool->entries[worst_actual].coherence.computed_at_generation = generation;
    }

    __syncthreads();
    if (tid == 0) {
        int total_active = Atomics::load_int(pool->active_count);
        DEVICE_FATAL_IF(total_active <= 0, "hierarchical_lifecycle_kernel: population extinct");
    }
}

#endif

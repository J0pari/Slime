
#ifndef ARCHIVE_SAMPLING_CU
#define ARCHIVE_SAMPLING_CU

#include "../config/config.cu"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../memory/archive.cu"
#include "../memory/pool.cu"
#include "../core/chemotaxis.cu"

__device__ int sample_from_archive_novel(GPUElite* archive, int archive_size, VoronoiCell* voronoi_cells, int num_cells, curandState* rand_state, int* sparse_cells_buffer) {
    if (archive_size == 0) return -1;

    int* sparse_cells = sparse_cells_buffer;

    int min_density = MIN_DENSITY_INIT;
    int num_sparse = 0;

    for (int i = 0; i < num_cells; i++) {
        if (voronoi_cells[i].density < min_density + ARCHIVE_DENSITY_MARGIN) {
            if (voronoi_cells[i].density < min_density) {
                min_density = voronoi_cells[i].density;
                num_sparse = 0;
            }
            sparse_cells[num_sparse++] = i;
        }
    }

    int result;
    if (num_sparse == 0) {
        result = curand(rand_state) % archive_size;
    } else {
        int cell_idx = sparse_cells[curand(rand_state) % num_sparse];
        int elite_idx = voronoi_cells[cell_idx].best_elite_idx;
        result = (elite_idx >= 0 && elite_idx < archive_size) ? elite_idx : 0;
    }

    return result;
}

__device__ void replace_from_archive_device(ComponentPool* pool, GPUElite* archive, int archive_size, VoronoiCell* voronoi_cells, int num_cells, BehavioralState* behavioral_agents, int pool_idx, unsigned int seed, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance, float* workspace_genome, DIRESAWeights* diresa_genome_weights) {
    if (archive_size == 0) return;

    curandState rand_state;
    curand_init(seed, 0, 0, &rand_state);

    int* sparse_cells_buffer = (int*)(workspace_genome + 2 * GENOME_SIZE);
    int elite_idx = sample_from_archive_novel(archive, archive_size, voronoi_cells, num_cells, &rand_state, sparse_cells_buffer);
    if (elite_idx < 0 || elite_idx >= archive_size) return;

    PoolEntry* entry = &pool->entries[pool_idx];

    entry->id = atomicAdd((int*)&pool->total_spawned, 1);
    entry->age = 0;
    entry->genome_hash = archive->genome_hash[elite_idx];
    entry->alive = true;
    entry->parent_idx = -1;
    entry->parent_hash = archive->genome_hash[elite_idx];
    entry->num_deltas = 0;

    float* elite_genome = workspace_genome;
    diresa_decode(&archive->latent_genome[elite_idx * GENOME_LATENT_DIM_MAX], elite_genome, diresa_genome_weights);

    int fitness_inherit_center_slot = derive_param_slot(entry->genome_hash, "lifecycle_fitness_inherit_center");
    int fitness_inherit_steepness_slot = derive_param_slot(entry->genome_hash, "lifecycle_fitness_inherit_steepness");
    float fitness_inherit_center = genome_to_param(elite_genome, entry->gradients, fitness_inherit_center_slot, 0.5f, 0.5f, 0.5f, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_FITNESS_INHERIT_CENTER_MIN, LIFECYCLE_FITNESS_INHERIT_CENTER_MAX);
    float fitness_inherit_steepness = genome_to_param(elite_genome, entry->gradients, fitness_inherit_steepness_slot, 0.5f, 0.5f, 0.5f, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MIN, LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MAX);
    float fitness_modulation = NORMALIZED_MAX / (NORMALIZED_MAX + expf(-fitness_inherit_steepness * (archive->fitness[elite_idx] - fitness_inherit_center)));

    entry->fitness = archive->fitness[elite_idx] * fitness_modulation;
    entry->coherence = archive->coherence[elite_idx];
    entry->task_accuracy = NAN;  
    entry->generalization_gap = NAN;  
    entry->hardware_efficiency = NAN;  
    entry->hunger = NORMALIZED_MAX - archive->coherence[elite_idx];
    entry->generation = archive->generation[elite_idx];

    for (int g = 0; g < GENOME_SIZE; g++) {
        entry->gradients[g] = 0.0f;
    }

    derive_architecture(entry->genome_hash, elite_genome, entry);
    derive_diresa(entry->genome_hash, elite_genome, entry);
    derive_fitness_exponents(entry->genome_hash, elite_genome, entry);

    int hw_dim = archive->hw_dim;
    int task_dim = archive->task_dim;
    int gen_dim = archive->gen_dim;

    if (pool_idx < POOL_CAPACITY_MAX) {
        BehavioralState* agent = &behavioral_agents[pool_idx];
        for (int i = 0; i < hw_dim; i++) {
            agent->hw_coords[i] = archive->hw_coords[elite_idx * hw_dim + i];
        }
        for (int i = 0; i < task_dim; i++) {
            agent->task_coords[i] = archive->task_coords[elite_idx * task_dim + i];
        }
        for (int i = 0; i < gen_dim; i++) {
            agent->gen_coords[i] = archive->gen_coords[elite_idx * gen_dim + i];
        }
    }

    entry->ca_state->tape.needs_weight_restore = 1;
    entry->ca_state->tape.restore_elite_idx = elite_idx;
}

#endif

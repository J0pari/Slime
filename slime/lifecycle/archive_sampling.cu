
#ifndef ARCHIVE_SAMPLING_CU
#define ARCHIVE_SAMPLING_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../memory/archive.cu"
#include "../memory/pool.cu"
#include "../core/chemotaxis.cu"
#include "genealogy.cu"

__device__ int sample_from_archive_novel(GPUElite* archive, int archive_size, VoronoiCell* voronoi_cells, int num_cells, curandState* rand_state, int* sparse_cells_buffer) {
    DEVICE_FATAL_IF(archive_size <= 0, "sample_from_archive_novel: empty archive");
    DEVICE_FATAL_IF(num_cells <= 0, "sample_from_archive_novel: no voronoi cells");

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
        DEVICE_FATAL_IF(elite_idx < 0 || elite_idx >= archive_size,
            "sample_from_archive_novel: cell %d has invalid best_elite_idx %d", cell_idx, elite_idx);
        result = elite_idx;
    }

    return result;
}

__device__ void replace_from_archive_device(ComponentPool* pool, GPUElite* archive, int archive_size, VoronoiCell* voronoi_cells, int num_cells, BehavioralState* behavioral_agents, int pool_idx, unsigned int seed, int generation, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance, float* workspace_genome, DIRESAWeights* diresa_genome_weights) {
    DEVICE_FATAL_IF(archive_size <= 0, "replace_from_archive_device: empty archive");
    DEVICE_FATAL_IF(num_cells <= 0, "replace_from_archive_device: no voronoi cells");

    curandState rand_state;
    curand_init(seed, 0, 0, &rand_state);

    int* sparse_cells_buffer = (int*)(workspace_genome + 2 * GENOME_SIZE);
    int elite_idx = sample_from_archive_novel(archive, archive_size, voronoi_cells, num_cells, &rand_state, sparse_cells_buffer);

    PoolEntry* entry = &pool->entries[pool_idx];

    entry->id = atomicAdd((int*)&pool->total_spawned, 1);
    entry->age = 0;
    entry->genome_hash = archive->genome_hash[elite_idx];
    entry->phase = LifecyclePhase::ACTIVE;
    pool->alive_flags[pool_idx] = true;
    entry->parent_idx = -1;
    entry->parent_hash = archive->genome_hash[elite_idx];
    entry->num_deltas = 0;

    float* elite_genome = workspace_genome;
    diresa_decode(&archive->latent_genome[elite_idx * GENOME_LATENT_DIM_MAX], elite_genome, diresa_genome_weights);

    InitContext ctx;
    ctx.derive_from_genome(elite_genome, entry->gradients);

    int fitness_inherit_center_slot = GenomeParamTable::lifecycle_fitness_inherit_center;
    int fitness_inherit_steepness_slot = GenomeParamTable::lifecycle_fitness_inherit_steepness;
    float fitness_inherit_center = genome_to_param(elite_genome, entry->gradients, fitness_inherit_center_slot, ctx.metabolic, ctx.stress, ctx.morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_FITNESS_INHERIT_CENTER_MIN, LIFECYCLE_FITNESS_INHERIT_CENTER_MAX);
    float fitness_inherit_steepness = genome_to_param(elite_genome, entry->gradients, fitness_inherit_steepness_slot, ctx.metabolic, ctx.stress, ctx.morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MIN, LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MAX);
    float fitness_modulation = NORMALIZED_MAX / (NORMALIZED_MAX + expf(-fitness_inherit_steepness * (archive->fitness[elite_idx] - fitness_inherit_center)));

    float modulated_fitness = archive->fitness[elite_idx] * fitness_modulation;
    uint64_t mod_hash = archive->fitness_input_hash[elite_idx];
    mod_hash ^= __float_as_uint(fitness_modulation) + 0x9e3779b9 + (mod_hash << 6) + (mod_hash >> 2);
    measured_value_set_computed(&entry->fitness, modulated_fitness, generation, mod_hash);
    measured_value_set_computed(&entry->coherence, archive->coherence[elite_idx], generation, entry->genome_hash);
    measured_value_set_uncomputed(&entry->task_accuracy);
    measured_value_set_uncomputed(&entry->generalization_gap);
    measured_value_set_uncomputed(&entry->hardware_efficiency);
    float hunger_val = NORMALIZED_MAX - archive->coherence[elite_idx];
    measured_value_set_computed(&entry->hunger, hunger_val, generation, entry->genome_hash);
    entry->generation = generation;

    for (int g = 0; g < GENOME_SIZE; g++) {
        entry->gradients[g] = 0.0f;
    }

    derive_architecture(elite_genome, entry);
    derive_diresa(elite_genome, entry);
    derive_fitness_exponents(elite_genome, entry);

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

__device__ void selection_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    GPUElite* archive = organism->archive;
    VoronoiCell* voronoi_cells = organism->voronoi_cells;
    int num_cells = organism->num_voronoi_cells;
    int* archive_size = &organism->archive_size;
    float* workspace_genomes = organism->workspace_genomes;

    int compact_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_idx < pool->alive_indices_count) {
        int entry_idx = pool->alive_indices[compact_idx];
        PoolEntry* entry = &pool->entries[entry_idx];
        DEVICE_FATAL_IF(!pool->alive_flags[entry_idx], "selection_device: dead entry in alive_indices");

        // Skip entries whose coherence hasn't been computed yet (e.g. gen=0, no previous accuracy)
        if (entry->coherence.state != ComputeState::COMPUTED) return;

        float* organism_genome = &workspace_genomes[entry_idx * 2 * GENOME_SIZE];
        float* temp_parent = &workspace_genomes[entry_idx * 2 * GENOME_SIZE + GENOME_SIZE];

        reconstruct_genome_from_archive(entry->parent_hash, archive, *archive_size,
            entry->delta_indices, entry->delta_values, entry->num_deltas,
            entry->max_deltas, organism_genome, GENOME_SIZE, temp_parent, organism->diresa_genome_weights);

        float* latent_genome = organism->latent_genome_pool + entry_idx * GENOME_LATENT_DIM_MAX;
        diresa_encode(organism_genome, latent_genome, &organism->diresa_genome_weights[0]);

        int hw_dim = archive->hw_dim;
        int task_dim = archive->task_dim;
        int gen_dim = archive->gen_dim;

        float hw_features[1] = {entry->hardware_efficiency.value};
        float* hw_coords_component = &organism->hw_coords_pool[entry_idx * hw_dim];
        diresa_encode(hw_features, hw_coords_component, entry->diresa_hw_weights);

        float gen_features[1] = {entry->generalization_gap.value};
        float* gen_coords_component = &organism->gen_coords_pool[entry_idx * gen_dim];
        diresa_encode(gen_features, gen_coords_component, entry->diresa_gen_weights);

        float* entry_genome = organism_genome;

        uint32_t parent_id_0;
        uint32_t parent_id_1 = 0;

        if (entry->parent_hash == UINT64_MAX) {
            parent_id_0 = 0;
        } else {
            int parent_idx = find_parent_by_hash(archive, *archive_size, entry->parent_hash);
            DEVICE_FATAL_IF(parent_idx < 0, "organism: parent not found in archive");
            parent_id_0 = parent_idx;
        }

        DEVICE_FATAL_IF(isnan(entry->coherence.value), "organism: entry coherence is NaN");
        DEVICE_FATAL_IF(isinf(entry->coherence.value), "organism: entry coherence is Inf");

        insert_elite_device(
            archive,
            archive_size,
            entry->fitness.value,
            entry->coherence.value,
            entry->fitness.value / entry->coherence.value,
            entry->genome_hash,
            parent_id_0,
            parent_id_1,
            organism->generation,
            &organism->hw_coords_pool[entry_idx * hw_dim],
            &organism->task_coords_pool[entry_idx * task_dim],
            &organism->gen_coords_pool[entry_idx * gen_dim],
            entry->task_accuracy.value,
            &archive->per_class_accuracy[entry_idx * NUM_CLASSES_MAX],
            NUM_CLASSES_MAX,
            voronoi_cells,
            num_cells,
            latent_genome,
            entry->fitness.input_hash,
            entry->fitness.computed_at_generation
        );

        if (entry->parent_hash == UINT64_MAX) {
            entry->parent_hash = entry->genome_hash;
        }
    }
}

#endif

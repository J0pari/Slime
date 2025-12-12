
#ifndef ORGANISM_CU
#define ORGANISM_CU
#include "../config/config.cu"
#include "../utils/tile_ops.cuh"
#include "../debug/auto_trace.cuh"
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include "../memory/archive.cu"
#include "../memory/pool.cu"
#include "../memory/tubes.cu"
#include "../learning/diresa.cu"
#include "../learning/autodiff.cu"
#include "../diagnostics/telemetry_probes.cu"
#include "../compression/delta.cu"
#include "../lifecycle/genealogy.cu"
#include "../lifecycle/archive_sampling.cu"
#include "../metrics/hardware_geometry.cu"
#include "../memory/parallel_compaction.cu"
#include "../kernels/tensor_core_ca.cu"
#include "../kernels/warp_ca.cu"
#include "correlation_matrix.cu"
#include "pseudopod.cu"
#include "pseudopod_tensor.cu"
#include "chemotaxis.cu"
#include "../utils/genome_params.cuh"

namespace cg = cooperative_groups;

struct CAParameterMap;
struct HybridTrainingMode;

struct Dataset;
struct ClassificationHead;
struct AdaptiveCurriculum;

struct Organism {

    ComponentPool* pool;
    GPUElite* archive;
    int archive_size;
    VoronoiCell* voronoi_cells;
    int num_voronoi_cells;
    TemporalTube* memory_tubes;
    MultiHeadCAState* ca_state;
    BehavioralState* behavioral_agents;

    ChemicalField* chemical_field;

    float* fitness_history;
    float* effective_rank_history;
    float* coherence_history;
    int generation;
    int active_components;

    float* behavioral_field_pool;
    float* behavioral_gradient_pool;
    float* behavioral_coords_pool;
    float* coherence_workspace_pool;
    float* memory_data_pool;
    float* fitness_rank_pool;
    float* fitness_coherence_pool;
    float* correlation_matrix_pool;
    float* prediction_error_history;
    float* fitness_workspace_pool;

    // Three-axis DIRESA compression: hardware/task/generalization
    DIRESAWeights* diresa_hw_weights;      // Hardware traces → hw_coords
    DIRESAWeights* diresa_task_weights;    // Task performance → task_coords
    DIRESAWeights* diresa_gen_weights;     // Generalization gap → gen_coords
    DIRESAWeights* diresa_genome_weights;  // Genome → latent genome coords

    // Three-axis behavioral coordinates (concatenated for Voronoi archive)
    float* hw_coords_pool;     // [pool_capacity * DIM_HW_MAX]
    float* task_coords_pool;   // [pool_capacity * DIM_TASK_MAX]
    float* gen_coords_pool;    // [pool_capacity * DIM_GEN_MAX]

    uint16_t* delta_indices_pool;
    float* delta_values_pool;
    uint16_t* delta_counts_pool;

    float* latent_genome_pool;  // DIRESA-compressed genomes [MAX_ARCHIVE_SIZE * GENOME_LATENT_DIM_MAX]

    TraceBuffer* trace_buffer;
    HardwareGeometry* hardware_geom;

    ADTape* ad_tape;
    CAParameterMap* param_map;

    HybridTrainingMode* training_mode;
    Dataset** dataset_array;
    Dataset* current_dataset;
    ClassificationHead* classifier;
    AdaptiveCurriculum* curriculum;
    float* voronoi_occupancy_histogram;
    float* pool_task_accuracies;

    void* lifecycle_states;

    TelemetryBuffer* telemetry;

    int* memory_compaction_valid_flags;
    int* memory_compaction_scan;
    MemoryEntry* memory_compaction_buffer;

    half* fp16_ca_workspace;
    float* fp32_ca_workspace;

    float* rd_u_field;
    float* rd_v_field;
    float* rd_u_next;
    float* rd_v_next;

    float* resource_density;
    float* resource_next;
    float* fitness_landscape;

    int* lifecycle_phase_counts;

    float* gradient_features_pool;
    float* gradient_logits_pool;
    float* gradient_loss_pool;
    float* gradient_logit_grads_pool;
    float* gradient_magnitudes_pool;

    float* pooling_weights_grad;
    float* fc_weights_grad;
    float* fc_bias_grad;
    float* features_grad;

    float* adam_m_pooling;
    float* adam_v_pooling;
    float* adam_m_fc_weights;
    float* adam_v_fc_weights;
    float* adam_m_fc_bias;
    float* adam_v_fc_bias;

    uint8_t* elite_compressed_pool;
    uint32_t* elite_size_pool;

    TapeEntry* ad_tape_entries_pool;
    float* ad_tape_values_pool;
    float* ad_tape_grads_pool;

    float* adam_m_perception_pool;
    float* adam_v_perception_pool;
    float* adam_m_interaction_pool;
    float* adam_v_interaction_pool;
    float* adam_m_value_pool;
    float* adam_v_value_pool;
    float* adam_m_classifier_pool;
    float* adam_v_classifier_pool;

    float* batch_ca_states_pool;
    int* batch_labels_pool;

    float* task_loss_pool;
    float* reg_loss_pool;
    float* rank_loss_pool;
    float* coherence_loss_pool;
    float* diversity_loss_pool;
    float* total_loss_pool;
};

#include "../training/hybrid_lifecycle.cu"
#include "../lifecycle/lifecycle_stages.cu"
__global__ void memory_update_kernel(TemporalTube* tubes, float* fitness_history, int* valid_flags_workspace, int* scan_workspace, MemoryEntry* temp_buffer, int generation, float* genome, float* gradients, uint64_t genome_hash, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance);
__global__ void selection_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, VoronoiCell* voronoi_cells, int num_cells, int* archive_size, BehavioralState* behavioral_agents, int generation, float* workspace_genomes);
__global__ void archive_driven_lifecycle_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, int archive_size, VoronoiCell* voronoi_cells, int num_cells, BehavioralState* behavioral_agents, float hunger_threshold, int generation, float* workspace_genomes, DIRESAWeights* diresa_genome_weights);
__global__ void spawn_wave_kernel(Organism* organism, ComponentPool* pool, float spawn_probability, int generation, float* workspace_genomes);
__global__ void culling_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, int archive_size, ChemicalField* chemical_field, ArchitectureParams arch, float fitness_threshold, float hunger_threshold);
__global__ void compute_fitness_from_diresa_kernel(float* latent_genome, int latent_dim, float* fitness_out, float task_exponent, float gen_exponent, float rank_exponent, float efficiency_exponent, float task_accuracy, float generalization_gap, float hardware_efficiency, float renyi_q);
__global__ void coherence_kernel(float* prediction_errors, float* coherence_out, int history_length);
__global__ void initialize_ca_from_field_kernel(float* ca_state, float* chemical_concentration, int grid_size);
__global__ void update_field_from_ca_kernel(float* chemical_concentration, float* ca_state, int grid_size);
__global__ void store_navigation_history_kernel(Organism* organism, BehavioralState* agents, TemporalTube* tubes, int generation, int hw_dim, int task_dim, int gen_dim);

extern "C" __global__ void organism_lifecycle_kernel(
    Organism* organism,
    int generation,
    float* workspace_genomes
) {
    printf("[DEBUG-ORGANISM-LIFECYCLE-CALLED] gen=%d tid=%d bid=%d\n", generation, threadIdx.x, blockIdx.x);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[DEBUG-ORGANISM-LIFECYCLE-ENTER] gen=%d tid=%d bid=%d\n", generation, threadIdx.x, blockIdx.x);
        cudaError_t err;

        if (!organism) {
            printf("FATAL: organism is nullptr!\n");
            return;
        }

        if (!organism->pool) {
            printf("FATAL: organism->pool is nullptr!\n");
            return;
        }

        if (!organism->pool->entries) {
            printf("FATAL: organism->pool->entries is nullptr!\n");
            return;
        }

        float* primary_genome = &workspace_genomes[0];
        float* primary_parent_temp = &workspace_genomes[GENOME_SIZE];
        reconstruct_genome_from_archive(
            organism->pool->entries[0].parent_hash,
            (GPUElite*)organism->archive,
            organism->archive_size,
            organism->pool->entries[0].delta_indices,
            organism->pool->entries[0].delta_values,
            organism->pool->entries[0].num_deltas,
            organism->pool->entries[0].max_deltas,
            primary_genome,
            GENOME_SIZE,
            primary_parent_temp,
            organism->diresa_genome_weights
        );

        if (generation % 5 == 0 || generation < 3) {
            printf("[GEN %3d] Pool: %3d active, %3d spawned, %3d culled | Archive: %2d elites\n",
                   generation,
                   Atomics::load_int(organism->pool->active_count),
                   Atomics::load_int(organism->pool->total_spawned),
                   Atomics::load_int(organism->pool->total_culled),
                   organism->archive_size);
        }

        ArchitectureParams arch;
        arch.num_heads = organism->pool->entries[0].num_heads;
        arch.channels = organism->pool->entries[0].channels;
        arch.hidden_dim = organism->pool->entries[0].hidden_dim;
        arch.head_dim = organism->pool->entries[0].head_dim;
        arch.grid_size = organism->pool->entries[0].grid_size;

        dim3 component_grid((MAX_COMPONENTS + (BLOCK_SIZE - 1)) / BLOCK_SIZE);
        dim3 component_block(BLOCK_SIZE);

        float* component_workspace_genomes = &workspace_genomes[0];

        printf("[DEBUG-LIFECYCLE] gen=%d BEFORE component_evolution_kernel\n", generation);
        component_evolution_kernel<<<component_grid, component_block>>>(
            organism,
            organism->pool,
            organism->archive,
            organism->voronoi_cells,
            organism->num_voronoi_cells,
            &organism->archive_size,
            organism->chemical_field,
            organism->behavioral_agents,
            organism->fitness_history,
            organism->coherence_history,
            generation,
            arch,
            component_workspace_genomes
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: component_evolution launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[DEBUG-LIFECYCLE] gen=%d AFTER component_evolution_kernel launch err=%d\n", generation, (int)err);

        float* selection_workspace_genomes = &workspace_genomes[organism->pool->capacity * 2 * GENOME_SIZE];

        atomicAdd(&organism->lifecycle_phase_counts[0], 1);

        selection_kernel<<<1, WARP_SIZE>>>(
            organism,
            organism->pool,
            organism->archive,
            organism->voronoi_cells,
            organism->num_voronoi_cells,
            &organism->archive_size,
            organism->behavioral_agents,
            generation,
            selection_workspace_genomes
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: selection launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        uint64_t pool_genome_hash = organism->pool->entries[0].genome_hash;
        float ctx_metabolic = organism->pool->entries[0].fitness;
        float ctx_stress = organism->pool->entries[0].hunger;

        float sum_morphogen = 0.0f;
        int total_cells = arch.grid_size * arch.grid_size * arch.channels;
        for (int i = 0; i < total_cells; i++) {
            sum_morphogen += organism->chemical_field->concentration[i];
        }
        float ctx_morphogen = sum_morphogen / (float)total_cells;

        int spawn_rate_slot = derive_param_slot(pool_genome_hash, "lifecycle_spawn_rate");
        int spawn_min_slot = derive_param_slot(pool_genome_hash, "lifecycle_spawn_min");
        float spawn_rate = genome_to_param(primary_genome, organism->pool->entries[0].gradients, spawn_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, SPAWN_RATE_MIN, SPAWN_RATE_MAX);
        float spawn_min = genome_to_param(primary_genome, organism->pool->entries[0].gradients, spawn_min_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, SPAWN_PROBABILITY_MIN_MIN, SPAWN_PROBABILITY_MIN_MAX);

        printf("[DEBUG-TELEMETRY] gen=%d mod_10=%d mod_100=%d DETAILED=%d COMPREHENSIVE=%d\n",
               generation, generation % 10, generation % 100, TELEMETRY_DETAILED, TELEMETRY_COMPREHENSIVE);

        if (generation % TELEMETRY_DETAILED == 0) {
            printf("[DEBUG-ENTERING-DETAILED] gen=%d\n", generation);
            genome_complexity_probe_kernel<<<1, 1>>>(organism->pool, &organism->telemetry->genome_complexity);

            int active = Atomics::load_int(organism->pool->active_count);
            float spawn_prob = spawn_rate * expf(-active / (float)MAX_COMPONENTS);
            printf("[GEN %3d] active=%d unique=%d entropy=%.2f archive=%d spawn_prob=%.3f\n",
                   generation, active, organism->telemetry->genome_complexity.unique_hashes,
                   organism->telemetry->genome_complexity.hash_entropy, organism->archive_size, spawn_prob);
        }

        if (generation % TELEMETRY_COMPREHENSIVE == 0) {
            printf("[DEBUG-ENTERING-COMPREHENSIVE] gen=%d\n", generation);
            archive_topology_probe_kernel<<<1, 1>>>((GPUElite*)organism->archive, organism->archive_size, organism->voronoi_cells, organism->num_voronoi_cells, &organism->telemetry->archive_topology);
            diresa_evolution_probe_kernel<<<1, 1>>>((GPUElite*)organism->archive, organism->archive_size, &organism->telemetry->diresa_evolution);

            printf("[ARCHIVE] occupied_cells=%d/%d density_var=%.3f novelty_grad=%.3f quality_range=%.6f\n",
                   organism->telemetry->archive_topology.occupied_cells, organism->num_voronoi_cells,
                   organism->telemetry->archive_topology.density_variance, organism->telemetry->archive_topology.novelty_gradient,
                   organism->telemetry->archive_topology.quality_range);
            printf("[DIRESA] drift_rate=%.3f hw_corr=%.3f grad_mag=%.3f injections=%d\n",
                   organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->diresa_evolution.hardware_feature_correlation,
                   organism->telemetry->diresa_evolution.gradient_magnitude_avg, organism->telemetry->diresa_evolution.archive_injections);
            printf("[LIFECYCLE] selection=%d spawn=%d archive=%d memory=%d\n",
                   organism->lifecycle_phase_counts[0], organism->lifecycle_phase_counts[1],
                   organism->lifecycle_phase_counts[2], organism->lifecycle_phase_counts[3]);
        }

        float spawn_prob = spawn_rate * expf(-Atomics::load_int(organism->pool->active_count) / (float)MAX_COMPONENTS);
        if (spawn_prob > spawn_min) {
            atomicAdd(&organism->lifecycle_phase_counts[1], 1);

            float* spawn_wave_workspace_genomes = &workspace_genomes[4 * GENOME_SIZE + 2 * MAX_POOL_SIZE * GENOME_SIZE];

            spawn_wave_kernel<<<1, WARP_SIZE>>>(
                organism,
                organism->pool,
                spawn_prob,
                generation,
                spawn_wave_workspace_genomes
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [organism_lifecycle]: spawn_wave launch failed: %s\n", cudaGetErrorString(err));
                return;
            }
        }

        int hunger_threshold_slot = derive_param_slot(pool_genome_hash, "lifecycle_hunger_threshold");
        float hunger_threshold = genome_to_param(primary_genome, organism->pool->entries[0].gradients, hunger_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, HUNGER_THRESHOLD_MIN, HUNGER_THRESHOLD_MAX);

        atomicAdd(&organism->lifecycle_phase_counts[2], 1);

        archive_driven_lifecycle_kernel<<<(organism->pool->capacity + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
            organism,
            organism->pool,
            organism->archive,
            organism->archive_size,
            organism->voronoi_cells,
            organism->num_voronoi_cells,
            organism->behavioral_agents,
            hunger_threshold,
            generation,
            workspace_genomes,
            organism->diresa_genome_weights
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: archive_driven_lifecycle launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        atomicAdd(&organism->lifecycle_phase_counts[3], 1);

        memory_update_kernel<<<1, BLOCK_SIZE>>>(
            organism->memory_tubes,
            organism->fitness_history,
            organism->memory_compaction_valid_flags,
            organism->memory_compaction_scan,
            organism->memory_compaction_buffer,
            generation,
            primary_genome,
            organism->pool->entries[0].gradients,
            pool_genome_hash,
            ctx_metabolic,
            ctx_stress,
            ctx_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: memory_update launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        float* workspace_genome_buffer = &workspace_genomes[GENOME_SIZE * 2];

        dim3 ca_grid(arch.grid_size / WMMA_TILE_DIM, arch.num_heads, 1);
        dim3 ca_block(WMMA_TILE_DIM, WMMA_TILE_DIM, 1);
        neural_ca_update_kernel<<<ca_grid, ca_block>>>(
            organism,
            organism->ca_state,
            organism->chemical_field,
            organism->effective_rank_history,
            organism->fp32_ca_workspace,
            generation,
            organism->pool,
            organism->fitness_history,
            organism->coherence_history,
            workspace_genome_buffer
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: neural_ca_update launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        if (generation % CHECKPOINT_INTERVAL == 0 && generation > 0) {
            int active = Atomics::load_int(organism->pool->active_count);
            float effective_rank = organism->effective_rank_history[generation];

            printf("[GEN %3d] Topology: active=%d, eff_rank=%.2f, archives=%d\n",
                   generation, active, effective_rank, organism->archive_size);
        }

        behavioral_update_kernel<<<component_grid, component_block>>>(
            organism,
            organism->behavioral_agents,
            organism->chemical_field,
            organism->memory_tubes,
            generation,
            arch,
            workspace_genomes
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: behavioral_update launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        dim3 field_grid((arch.grid_size + 15) / 16, (arch.grid_size + 15) / 16);
        dim3 field_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

        update_fitness_landscape_kernel<<<field_grid, field_block>>>(
            organism->pool,
            organism->fitness_landscape,
            arch.grid_size
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: update_fitness_landscape launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        uint64_t genome_hash = organism->pool->entries[0].genome_hash;
        float* genome = primary_genome;
        float* gradients = organism->pool->entries[0].gradients;

        int diffusivity_slot = derive_param_slot(genome_hash, "resource_flow_diffusivity");
        int advection_slot = derive_param_slot(genome_hash, "resource_flow_advection");
        int dt_slot = derive_param_slot(genome_hash, "resource_flow_dt");

        int diffusivity_min_slot = derive_param_slot(genome_hash, "resource_flow_diffusivity_min");
        int diffusivity_max_slot = derive_param_slot(genome_hash, "resource_flow_diffusivity_max");
        int advection_min_slot = derive_param_slot(genome_hash, "resource_flow_advection_min");
        int advection_max_slot = derive_param_slot(genome_hash, "resource_flow_advection_max");
        int dt_min_slot = derive_param_slot(genome_hash, "resource_flow_dt_min");
        int dt_max_slot = derive_param_slot(genome_hash, "resource_flow_dt_max");

        float diffusivity_min = (genome[diffusivity_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float diffusivity_max = (genome[diffusivity_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float advection_min = (genome[advection_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float advection_max = (genome[advection_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float dt_min = (genome[dt_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float dt_max = (genome[dt_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;

        float rf_diffusivity = genome_to_param(genome, gradients, diffusivity_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, diffusivity_min, diffusivity_max);
        float rf_advection = genome_to_param(genome, gradients, advection_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, advection_min, advection_max);
        float rf_dt = genome_to_param(genome, gradients, dt_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, dt_min, dt_max);

        resource_flow_kernel<<<field_grid, field_block>>>(
            organism->resource_density,
            organism->resource_next,
            organism->fitness_landscape,
            arch.grid_size,
            rf_dt,
            rf_diffusivity,
            rf_advection
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: resource_flow launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        float* tmp = organism->resource_density;
        organism->resource_density = organism->resource_next;
        organism->resource_next = tmp;

        int num_blocks = (organism->pool->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;

        float* hierarchical_workspace = &workspace_genomes[4 * GENOME_SIZE];

        hierarchical_lifecycle_kernel<<<num_blocks, BLOCK_SIZE>>>(
            organism->pool,
            (LocalOrganismState<BLOCK_SIZE>*)organism->lifecycle_states,
            (GPUElite*)organism->archive,
            &organism->archive_size,
            organism->voronoi_cells,
            organism->num_voronoi_cells,
            generation,
            organism->chemical_field,
            arch.grid_size,
            organism->telemetry,
            hierarchical_workspace,
            organism->diresa_genome_weights
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: hierarchical_lifecycle launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

    }
}

__global__ void component_evolution_kernel(
    Organism* organism,
    ComponentPool* pool,
    GPUElite* archive,
    VoronoiCell* voronoi_cells,
    int num_cells,
    int* archive_size,
    ChemicalField* chemical_field,
    BehavioralState* behavioral_agents,
    float* fitness_history,
    float* coherence_history,
    int generation,
    ArchitectureParams arch,
    float* workspace_genomes
) {
    int tid = GridStride::thread_id();
    if (tid >= pool->capacity || !pool->entries[tid].alive) return;

    float* primary_genome = &workspace_genomes[tid * 2 * GENOME_SIZE];
    float* primary_parent_temp = &workspace_genomes[tid * 2 * GENOME_SIZE + GENOME_SIZE];

    reconstruct_genome_from_archive(
        pool->entries[tid].parent_hash,
        archive,
        *archive_size,
        pool->entries[tid].delta_indices,
        pool->entries[tid].delta_values,
        pool->entries[tid].num_deltas,
        pool->entries[tid].max_deltas,
        primary_genome,
        GENOME_SIZE,
        primary_parent_temp,
        organism->diresa_genome_weights
    );

    cudaError_t err;

    float current_task_accuracy = organism->telemetry->task_performance.accuracy;
    pool->entries[tid].task_accuracy = current_task_accuracy;
    pool->entries[tid].generalization_gap = fabsf(organism->telemetry->task_performance.train_accuracy - organism->telemetry->task_performance.test_accuracy);

    if (tid == 0) {
        aggregate_hardware_geometry_kernel<<<1, BLOCK_SIZE>>>(organism->trace_buffer, organism->hardware_geom);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [component_evolution]: aggregate_hardware_geometry launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
    }
    __syncthreads();

    {
        float hardware_features_temp[HARDWARE_FEATURES_DIM];
        extract_hardware_features(organism->hardware_geom, hardware_features_temp);

        float hw_efficiency_sum = EPSILON;
        for (int i = 0; i < HARDWARE_FEATURES_DIM; i++) {
            int hw_weight_slot = derive_param_slot(pool->entries[tid].genome_hash, "hw_efficiency_weight");
            float hw_weight = genome_to_param(
                primary_genome,
                pool->entries[tid].gradients,
                hw_weight_slot,
                pool->entries[tid].fitness,
                pool->entries[tid].hunger,
                organism->chemical_field->concentration[0],
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                FITNESS_EFFICIENCY_EXPONENT_MIN, FITNESS_EFFICIENCY_EXPONENT_MAX
            );
            hw_efficiency_sum += hw_weight * hardware_features_temp[i];
        }
        pool->entries[tid].hardware_efficiency = hw_efficiency_sum;

        float* latent_genome = organism->latent_genome_pool + tid * GENOME_LATENT_DIM_MAX;

        printf("[DIRESA-INPUT] tid=%d latent[0]=%.6f task_acc=%.6f gen_gap=%.6f hw_eff=%.6f\n",
               tid, latent_genome[0], pool->entries[tid].task_accuracy,
               pool->entries[tid].generalization_gap, pool->entries[tid].hardware_efficiency);
        printf("[DIRESA-EXPONENTS] tid=%d α=%.3f β=%.3f γ=%.3f δ=%.3f renyi_q=%.3f\n",
               tid, pool->entries[tid].fitness_task_exponent, pool->entries[tid].fitness_gen_exponent,
               pool->entries[tid].fitness_rank_exponent, pool->entries[tid].fitness_efficiency_exponent,
               pool->entries[tid].renyi_q);

        compute_fitness_from_diresa_kernel<<<1, 1>>>(
            latent_genome,
            GENOME_LATENT_DIM_MAX,
            &pool->entries[tid].fitness,
            pool->entries[tid].fitness_task_exponent,
            pool->entries[tid].fitness_gen_exponent,
            pool->entries[tid].fitness_rank_exponent,
            pool->entries[tid].fitness_efficiency_exponent,
            pool->entries[tid].task_accuracy,
            pool->entries[tid].generalization_gap,
            pool->entries[tid].hardware_efficiency,
            pool->entries[tid].renyi_q
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [component_evolution]: compute_fitness_from_diresa launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        printf("[DIRESA-OUTPUT] tid=%d fitness=%.9f\n", tid, pool->entries[tid].fitness);
    }

    if (generation > 0) {
        float prev_task_accuracy = organism->fitness_history[(generation - 1) * MAX_COMPONENTS + tid];
        float learning_success = pool->entries[tid].task_accuracy - prev_task_accuracy;

        if (learning_success > EPSILON) {
            float baldwin_sensitivity = pool->entries[tid].baldwin_sensitivity;

            for (int g = 0; g < GENOME_SIZE; g++) {
                pool->entries[tid].gradients[g] += learning_success * baldwin_sensitivity * primary_genome[g];
                pool->entries[tid].gradients[g] = tanhf(pool->entries[tid].gradients[g]);
            }
        }
    }
}

__global__ void compute_fitness_from_diresa_kernel(
    float* latent_genome,
    int latent_dim,
    float* fitness_out,
    float task_exponent,
    float gen_exponent,
    float rank_exponent,
    float efficiency_exponent,
    float task_accuracy,
    float generalization_gap,
    float hardware_efficiency,
    float renyi_q
) {
    // Effective rank from DIRESA latent space variance (higher variance = higher effective dimensionality)
    float mean = 0.0f;
    for (int i = 0; i < latent_dim; i++) {
        mean += latent_genome[i];
    }
    mean /= latent_dim;

    float variance = 0.0f;
    for (int i = 0; i < latent_dim; i++) {
        float diff = latent_genome[i] - mean;
        variance += diff * diff;
    }
    variance /= latent_dim;

    // Effective rank ~ sqrt(variance) * latent_dim (normalized manifold spread)
    float effective_rank = sqrtf(variance + EPSILON) * latent_dim;

    float gen_gap_term = (1.0f - generalization_gap) + EPSILON;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[FITNESS-COMPONENTS] task_acc=%.6f gen_gap=%.6f (1-gen_gap=%.6f) eff_rank=%.6f hw_eff=%.6f\n",
               task_accuracy, generalization_gap, gen_gap_term, effective_rank, hardware_efficiency);
        printf("[FITNESS-EXPONENTS] α=%.3f β=%.3f γ=%.3f δ=%.3f\n",
               task_exponent, gen_exponent, rank_exponent, efficiency_exponent);
    }

    // Task-grounded power-law fitness: task^α × (1-gen_gap)^β × rank^γ × efficiency^δ
    *fitness_out = powf(task_accuracy + EPSILON, task_exponent)
                 * powf(gen_gap_term, gen_exponent)
                 * powf(effective_rank + EPSILON, rank_exponent)
                 * powf(hardware_efficiency + EPSILON, efficiency_exponent);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[FITNESS-RESULT] fitness=%.9f\n", *fitness_out);
    }
}

__global__ void selection_kernel(
    Organism* organism,
    ComponentPool* pool,
    GPUElite* archive,
    VoronoiCell* voronoi_cells,
    int num_cells,
    int* archive_size,
    BehavioralState* behavioral_agents,
    int generation,
    float* workspace_genomes
) {
    int tid = GridStride::thread_id();

    for (int i = tid; i < pool->capacity; i += WARP_SIZE) {
        if (pool->entries[i].alive) {
            float* organism_genome = &workspace_genomes[i * 2 * GENOME_SIZE];
            float* temp_parent = &workspace_genomes[i * 2 * GENOME_SIZE + GENOME_SIZE];

            reconstruct_genome_from_archive(
                pool->entries[i].parent_hash,
                archive,
                *archive_size,
                pool->entries[i].delta_indices,
                pool->entries[i].delta_values,
                pool->entries[i].num_deltas,
                pool->entries[i].max_deltas,
                organism_genome,
                GENOME_SIZE,
                temp_parent,
                organism->diresa_genome_weights
            );

            float* latent_genome = organism->latent_genome_pool + i * GENOME_LATENT_DIM_MAX;
            diresa_encode(organism_genome, latent_genome, &organism->diresa_genome_weights[0]);
        }
    }

    __syncthreads();

    // Encode behavioral coordinates per component
    if (tid == 0) {
        aggregate_hardware_geometry_kernel<<<1, BLOCK_SIZE>>>(organism->trace_buffer, organism->hardware_geom);
    }
    __syncthreads();

    for (int i = tid; i < pool->capacity; i += WARP_SIZE) {
        if (pool->entries[i].alive) {
            int hw_dim = archive->hw_dim;
            int task_dim = archive->task_dim;
            int gen_dim = archive->gen_dim;

            float hardware_features_temp[HARDWARE_FEATURES_DIM];
            extract_hardware_features(organism->hardware_geom, hardware_features_temp);

            float* hw_coords_component = &organism->hw_coords_pool[i * hw_dim];
            diresa_encode(hardware_features_temp, hw_coords_component, &organism->diresa_hw_weights[0]);

            float task_features[1] = {pool->entries[i].task_accuracy};
            float* task_coords_component = &organism->task_coords_pool[i * task_dim];
            diresa_encode(task_features, task_coords_component, &organism->diresa_task_weights[0]);

            float gen_features[1] = {pool->entries[i].generalization_gap};
            float* gen_coords_component = &organism->gen_coords_pool[i * gen_dim];
            diresa_encode(gen_features, gen_coords_component, &organism->diresa_gen_weights[0]);
        }
    }

    __syncthreads();

    // Archive insertion per component
    for (int i = tid; i < pool->capacity; i += WARP_SIZE) {
        if (pool->entries[i].alive) {
            PoolEntry* entry = &pool->entries[i];
            float* entry_genome = &workspace_genomes[i * 2 * GENOME_SIZE];

            int hw_dim = archive->hw_dim;
            int task_dim = archive->task_dim;
            int gen_dim = archive->gen_dim;

            int parent_idx = find_parent_by_hash(archive, *archive_size, entry->genome_hash);
            uint32_t parent_id_0 = (parent_idx >= 0) ? parent_idx : 0;
            uint32_t parent_id_1 = 0;

            insert_elite_kernel<<<1, 1>>>(
                archive,
                archive_size,
                entry->fitness,
                entry->coherence,
                entry->fitness / (entry->coherence + EPSILON),
                gpu_sha256(entry_genome, GENOME_SIZE),
                parent_id_0,
                parent_id_1,
                generation,
                &organism->hw_coords_pool[i * hw_dim],
                &organism->task_coords_pool[i * task_dim],
                &organism->gen_coords_pool[i * gen_dim],
                entry->task_accuracy,
                &archive->per_class_accuracy[i * NUM_CLASSES_MAX],
                NUM_CLASSES_MAX,
                voronoi_cells,
                num_cells
            );
        }
    }

    __syncthreads();

    if (tid == 0) {
        int hw_dim = archive->hw_dim;
        int task_dim = archive->task_dim;
        int gen_dim = archive->gen_dim;

        float* genome_sample = &workspace_genomes[0];
        PoolEntry* entry_sample = &pool->entries[0];

        update_voronoi_density_kernel<<<(num_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            voronoi_cells,
            archive,
            *archive_size,
            num_cells,
            hw_dim + task_dim + gen_dim,
            genome_sample,
            entry_sample->gradients,
            entry_sample->genome_hash,
            entry_sample->fitness,
            entry_sample->hunger,
            organism->chemical_field->concentration[0],
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy
        );
    }
}

__global__ void spawn_wave_kernel(
    Organism* organism,
    ComponentPool* pool,
    float spawn_probability,
    int generation,
    float* workspace_genomes
) {
    int tid = threadIdx.x;

    unsigned int seed = tid * generation * RNG_SEED_MULTIPLIER;
    seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
    float rand = (seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE;

    if (rand < spawn_probability) {

        float* temp_genome = &workspace_genomes[tid * GENOME_SIZE * 5];
        float* temp_parent = &workspace_genomes[tid * GENOME_SIZE * 5 + GENOME_SIZE];

        int parent_idx = -1;
        for (int i = 0; i < pool->capacity; i++) {
            if (pool->entries[i].alive) {
                reconstruct_genome_from_archive(
                    pool->entries[i].parent_hash,
                    (GPUElite*)organism->archive,
                    organism->archive_size,
                    pool->entries[i].delta_indices,
                    pool->entries[i].delta_values,
                    pool->entries[i].num_deltas,
                    pool->entries[i].max_deltas,
                    temp_genome,
                    GENOME_SIZE,
                    temp_parent,
                    organism->diresa_genome_weights
                );

                uint64_t temp_hash = pool->entries[i].genome_hash;
                int fitness_threshold_slot = derive_param_slot(temp_hash, "spawn_fitness_threshold");

                int ctx_metabolic_slot = derive_param_slot(temp_hash, "spawn_ctx_metabolic");
                int ctx_stress_slot = derive_param_slot(temp_hash, "spawn_ctx_stress");
                int ctx_morphogen_slot = derive_param_slot(temp_hash, "spawn_ctx_morphogen");

                float ctx_metabolic = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_metabolic_slot,
                    pool->entries[i].fitness, pool->entries[i].hunger, EPSILON,
                    organism->telemetry->genome_complexity.hash_entropy,
                    organism->telemetry->archive_topology.novelty_gradient,
                    organism->telemetry->diresa_evolution.behavioral_drift_rate,
                    organism->telemetry->task_performance.accuracy,
                    NORMALIZED_MIN, NORMALIZED_MAX);

                float ctx_stress = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_stress_slot,
                    pool->entries[i].fitness, pool->entries[i].hunger, EPSILON,
                    organism->telemetry->genome_complexity.hash_entropy,
                    organism->telemetry->archive_topology.novelty_gradient,
                    organism->telemetry->diresa_evolution.behavioral_drift_rate,
                    organism->telemetry->task_performance.accuracy,
                    NORMALIZED_MIN, NORMALIZED_MAX);

                float ctx_morphogen = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_morphogen_slot,
                    pool->entries[i].fitness, pool->entries[i].hunger, EPSILON,
                    organism->telemetry->genome_complexity.hash_entropy,
                    organism->telemetry->archive_topology.novelty_gradient,
                    organism->telemetry->diresa_evolution.behavioral_drift_rate,
                    organism->telemetry->task_performance.accuracy,
                    NORMALIZED_MIN, NORMALIZED_MAX);

                float fitness_threshold = genome_to_param(
                    temp_genome,
                    pool->entries[i].gradients,
                    fitness_threshold_slot,
                    ctx_metabolic, ctx_stress, ctx_morphogen,
                    organism->telemetry->genome_complexity.hash_entropy,
                    organism->telemetry->archive_topology.novelty_gradient,
                    organism->telemetry->diresa_evolution.behavioral_drift_rate,
                    organism->telemetry->task_performance.accuracy,
                    SPAWN_PROBABILITY_MIN_MIN, SPAWN_PROBABILITY_MIN_MAX
                );

                if (pool->entries[i].fitness > fitness_threshold) {
                    parent_idx = i;
                    break;
                }
            }
        }

        if (parent_idx >= 0) {
            float* workspace_parent_genome = &workspace_genomes[tid * GENOME_SIZE * 5 + GENOME_SIZE * 2];
            float* workspace_child_genome = &workspace_genomes[tid * GENOME_SIZE * 5 + GENOME_SIZE * 3];
            float* workspace_parent_parent_temp = &workspace_genomes[tid * GENOME_SIZE * 5 + GENOME_SIZE * 4];

            reconstruct_genome_from_archive(
                pool->entries[parent_idx].parent_hash,
                (GPUElite*)organism->archive,
                organism->archive_size,
                pool->entries[parent_idx].delta_indices,
                pool->entries[parent_idx].delta_values,
                pool->entries[parent_idx].num_deltas,
                pool->entries[parent_idx].max_deltas,
                workspace_parent_genome,
                GENOME_SIZE,
                workspace_parent_parent_temp,
                organism->diresa_genome_weights
            );

            uint64_t parent_hash = pool->entries[parent_idx].genome_hash;

            int ctx_metabolic_slot = derive_param_slot(parent_hash, "mutation_ctx_metabolic");
            int ctx_stress_slot = derive_param_slot(parent_hash, "mutation_ctx_stress");
            int ctx_morphogen_slot = derive_param_slot(parent_hash, "mutation_ctx_morphogen");

            float ctx_metabolic = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_metabolic_slot,
                pool->entries[parent_idx].fitness, pool->entries[parent_idx].hunger, EPSILON,
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                NORMALIZED_MIN, NORMALIZED_MAX);

            float ctx_stress = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_stress_slot,
                pool->entries[parent_idx].fitness, pool->entries[parent_idx].hunger, EPSILON,
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                NORMALIZED_MIN, NORMALIZED_MAX);

            float ctx_morphogen = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_morphogen_slot,
                pool->entries[parent_idx].fitness, pool->entries[parent_idx].hunger, EPSILON,
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                NORMALIZED_MIN, NORMALIZED_MAX);

            int mutation_rate_slot = derive_param_slot(parent_hash, "metalearning_mutation_rate");
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

            spawn_component_kernel<<<1, 1>>>(
                pool,
                (GPUElite*)organism->archive,
                organism->archive_size,
                parent_idx,
                mutation_rate,
                workspace_parent_genome,
                workspace_child_genome,
                workspace_parent_parent_temp,
                organism->diresa_genome_weights
            );
        }
    }
}

__global__ void culling_kernel(
    Organism* organism,
    ComponentPool* pool,
    GPUElite* archive,
    int archive_size,
    ChemicalField* chemical_field,
    ArchitectureParams arch,
    float fitness_threshold,
    float hunger_threshold,
    float* workspace_genomes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[idx];

    if (entry->alive) {
        float* entry_genome = &workspace_genomes[idx * GENOME_SIZE * 2];
        float* entry_parent_temp = &workspace_genomes[idx * GENOME_SIZE * 2 + GENOME_SIZE];
        reconstruct_genome_from_archive(
            entry->parent_hash,
            archive,
            archive_size,
            entry->delta_indices,
            entry->delta_values,
            entry->num_deltas,
            entry->max_deltas,
            entry_genome,
            GENOME_SIZE,
            entry_parent_temp,
            organism->diresa_genome_weights
        );

        uint64_t entry_genome_hash = entry->genome_hash;
        float ctx_metabolic = entry->fitness;
        float ctx_stress = entry->hunger;

        float sum_morphogen = 0.0f;
        int total_cells = arch.grid_size * arch.grid_size * arch.channels;
        for (int i = 0; i < total_cells; i++) {
            sum_morphogen += chemical_field->concentration[i];
        }
        float ctx_morphogen = sum_morphogen / (float)total_cells;

        int fitness_cull_mult_slot = derive_param_slot(entry_genome_hash, "lifecycle_fitness_culling_mult");
        float fitness_cull_mult = genome_to_param(entry_genome, entry->gradients, fitness_cull_mult_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, FITNESS_CULLING_MULT_MIN, FITNESS_CULLING_MULT_MAX);

        if (entry->fitness < fitness_threshold * fitness_cull_mult) {
            entry->alive = false;
            Atomics::increment_int(pool->total_culled);
            Atomics::decrement_int(pool->active_count);
        }

        else if (entry->hunger > hunger_threshold) {
            entry->alive = false;
            Atomics::increment_int(pool->total_culled);
            Atomics::decrement_int(pool->active_count);
        }
    }
}

__global__ void archive_driven_lifecycle_kernel(
    Organism* organism,
    ComponentPool* pool,
    GPUElite* archive,
    int archive_size,
    VoronoiCell* voronoi_cells,
    int num_cells,
    BehavioralState* behavioral_agents,
    float hunger_threshold,
    int generation,
    float* workspace_genomes,
    DIRESAWeights* diresa_genome_weights
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[idx];

    bool should_cull = false;
    if (entry->alive) {
        if (entry->hunger > hunger_threshold) {
            should_cull = true;
        }
    }

    if (should_cull || !entry->alive) {
        if (archive_size > 0) {
            if (should_cull) {
                Atomics::increment_int(pool->total_culled);
                Atomics::decrement_int(pool->active_count);
            }

            replace_from_archive_kernel<<<1, 1>>>(
                pool,
                archive,
                archive_size,
                voronoi_cells,
                num_cells,
                behavioral_agents,
                idx,
                generation * pool->capacity + idx,
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                workspace_genomes,
                diresa_genome_weights
            );

            Atomics::increment_int(pool->active_count);
        }
    }
}

__global__ void neural_ca_update_kernel(
    Organism* organism,
    MultiHeadCAState* ca_state,
    ChemicalField* chemical_field,
    float* effective_rank_history,
    float* fp32_workspace,
    int generation,
    ComponentPool* pool,
    float* fitness_history,
    float* coherence_history,
    float* workspace_genome
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {



        int best_idx = 0;
        float best_fitness = -1e9f;

        for (int i = 0; i < pool->capacity; i++) {
            if (pool->entries[i].alive && pool->entries[i].fitness > best_fitness) {
                best_fitness = pool->entries[i].fitness;
                best_idx = i;
            }
        }

        float* best_genome = workspace_genome;
        float* best_parent_temp = &workspace_genome[GENOME_SIZE];
        reconstruct_genome_from_archive(
            pool->entries[best_idx].parent_hash,
            (GPUElite*)organism->archive,
            organism->archive_size,
            pool->entries[best_idx].delta_indices,
            pool->entries[best_idx].delta_values,
            pool->entries[best_idx].num_deltas,
            pool->entries[best_idx].max_deltas,
            best_genome,
            GENOME_SIZE,
            best_parent_temp,
            organism->diresa_genome_weights
        );

        float* genome = best_genome;
        float* epigenetic = pool->entries[best_idx].gradients;

        uint64_t best_hash = pool->entries[best_idx].genome_hash;

        float context_metabolic = pool->entries[best_idx].fitness;

        int stress_numerator_slot = derive_param_slot(best_hash, "context_stress_numerator");
        float stress_numerator = genome_to_param(
            genome, epigenetic, stress_numerator_slot,
            pool->entries[best_idx].fitness,
            pool->entries[best_idx].hunger,
            EPSILON,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            NORMALIZED_MIN, NORMALIZED_MAX
        );
        float context_stress = stress_numerator / (pool->entries[best_idx].hunger + EPSILON);  

        
        float* ca_concentration = ca_state->ca_concentration;
        float* ca_output = ca_state->ca_output;

        ArchitectureParams arch;
        arch.num_heads = pool->entries[best_idx].num_heads;
        arch.channels = pool->entries[best_idx].channels;
        arch.hidden_dim = pool->entries[best_idx].hidden_dim;
        arch.head_dim = pool->entries[best_idx].head_dim;
        arch.grid_size = pool->entries[best_idx].grid_size;

        float sum_morphogen = 0.0f;
        int morph_total_cells = arch.grid_size * arch.grid_size * arch.channels;
        for (int i = 0; i < morph_total_cells; i++) {
            sum_morphogen += ca_concentration[i];
        }
        float context_morphogen = sum_morphogen / (float)morph_total_cells;

        dim3 init_grid((arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, (arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM);
        dim3 init_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

        initialize_ca_from_field_kernel<<<init_grid, init_block>>>(
            ca_concentration,
            chemical_field->concentration,
            arch.grid_size
        );

        dim3 ca_grid(arch.grid_size / WMMA_TILE_DIM, arch.num_heads, 1);
        dim3 ca_block(WMMA_TILE_DIM, WMMA_TILE_DIM, 1);

        multi_head_ca_tensor_kernel<<<ca_grid, ca_block>>>(
            ca_concentration,
            ca_state->perception_weights,
            ca_state->interaction_weights,
            ca_state->value_weights,
            ca_output,
            1,
            arch.grid_size,
            arch.num_heads,
            arch,
            nullptr
        );

        int total_cells = arch.grid_size * arch.grid_size;
        dim3 affinity_grid((total_cells + WARP_SIZE - 1) / WARP_SIZE);
        dim3 affinity_block(32);
        reduce_affinity_kernel<<<affinity_grid, affinity_block>>>(
            ca_output,
            ca_state->affinity_reduced,
            arch.grid_size,
            arch
        );

        int s_param_slot = derive_param_slot(pool->entries[best_idx].genome_hash, "flow_lenia_s");
        float s_param = genome_to_param(
            genome, epigenetic, s_param_slot,
            context_metabolic, context_stress, context_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            FLOW_LENIA_S_MIN, FLOW_LENIA_S_MAX
        );
        int beta_A_slot = derive_param_slot(pool->entries[best_idx].genome_hash, "flow_lenia_beta_a");
        float beta_A_param = genome_to_param(
            genome, epigenetic, beta_A_slot,
            context_metabolic, context_stress, context_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            FLOW_LENIA_BETA_A_MIN, FLOW_LENIA_BETA_A_MAX
        );
        int n_param_slot = derive_param_slot(pool->entries[best_idx].genome_hash, "flow_lenia_n");
        float n_param = genome_to_param(
            genome, epigenetic, n_param_slot,
            context_metabolic, context_stress, context_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            FLOW_LENIA_N_MIN, FLOW_LENIA_N_MAX
        );

        int resource_flow_dt_slot = derive_param_slot(pool->entries[best_idx].genome_hash, "flow_lenia_resource_dt");
        float resource_flow_dt = genome_to_param(genome, epigenetic, resource_flow_dt_slot, context_metabolic, context_stress, context_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, RESOURCE_FLOW_DT_MIN, RESOURCE_FLOW_DT_MAX);

        dim3 flow_grid((arch.grid_size + 15) / 16, (arch.grid_size + 15) / 16);
        dim3 flow_block(WMMA_TILE_DIM, WMMA_TILE_DIM);
        compute_flow_field_kernel<<<flow_grid, flow_block>>>(
            ca_state->affinity_reduced,
            ca_state->ca_concentration,
            ca_state->flow_field,
            arch.grid_size,
            beta_A_param,
            n_param,
            arch
        );

        int buffer_size = arch.grid_size * arch.grid_size * arch.channels;
        dim3 clear_grid((buffer_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 clear_block(256);
        clear_buffer_kernel<<<clear_grid, clear_block>>>(
            ca_state->reintegration_buffer,
            buffer_size
        );

        dim3 reint_grid(arch.grid_size, arch.grid_size);
        dim3 reint_block(256);
        reintegration_redistribute_kernel<<<reint_grid, reint_block>>>(
            ca_state->ca_concentration,
            ca_state->flow_field,
            ca_state->reintegration_buffer,
            arch.grid_size,
            resource_flow_dt,
            s_param,
            arch
        );

        dim3 copy_grid((buffer_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 copy_block(256);
        copy_buffer_kernel<<<copy_grid, copy_block>>>(
            ca_state->reintegration_buffer,
            ca_state->ca_concentration,
            buffer_size
        );

        update_field_from_ca_kernel<<<init_grid, init_block>>>(
            chemical_field->concentration,
            ca_state->ca_concentration,
            arch.grid_size
        );

        int weight_count = arch.num_heads * arch.channels * arch.hidden_dim;

        int convert_threads = BLOCK_SIZE;
        int convert_blocks = (weight_count + convert_threads - 1) / convert_threads;
        convert_weights_to_fp32<<<convert_blocks, convert_threads>>>(
            ca_state->perception_weights,
            fp32_workspace,
            weight_count
        );

        // Encode best_genome to latent first, then compute effective rank from latent variance
        float* temp_latent = workspace_genome;  // Reuse workspace for temp latent (128 floats)
        diresa_encode(best_genome, temp_latent, &organism->diresa_genome_weights[0]);

        compute_effective_rank_from_latent_kernel<<<1, BLOCK_SIZE>>>(
            temp_latent,
            &effective_rank_history[generation],
            GENOME_LATENT_DIM_MAX
        );
    }
}

__global__ void behavioral_update_kernel(
    Organism* organism,
    BehavioralState* agents,
    ChemicalField* chemical_field,
    TemporalTube* memory_tubes,
    int generation,
    ArchitectureParams arch,
    float* workspace_genomes
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int num_agents = MAX_COMPONENTS;

        float* primary_genome = &workspace_genomes[0];
        float* primary_parent_temp = &workspace_genomes[GENOME_SIZE];

        reconstruct_genome_from_archive(
            organism->pool->entries[0].parent_hash,
            (GPUElite*)organism->archive,
            organism->archive_size,
            organism->pool->entries[0].delta_indices,
            organism->pool->entries[0].delta_values,
            organism->pool->entries[0].num_deltas,
            organism->pool->entries[0].max_deltas,
            primary_genome,
            GENOME_SIZE,
            primary_parent_temp,
            organism->diresa_genome_weights
        );

        uint64_t genome_hash = organism->pool->entries[0].genome_hash;
        float* genome = primary_genome;
        float* gradients = organism->pool->entries[0].gradients;
        float ctx_metabolic = organism->pool->entries[0].fitness;
        float ctx_stress = organism->pool->entries[0].hunger;

        float sum_morphogen = 0.0f;
        int total_cells = arch.grid_size * arch.grid_size * arch.channels;
        for (int i = 0; i < total_cells; i++) {
            sum_morphogen += chemical_field->concentration[i];
        }
        float ctx_morphogen = sum_morphogen / (float)total_cells;

        BehavioralDimensions dims;
        dims.derive_from_genome(genome_hash, primary_genome);

        int chem_diff_dt_slot = derive_param_slot(genome_hash, "chemical_diffusion_dt");
        float chem_diff_dt = genome_to_param(genome, gradients, chem_diff_dt_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, CHEMICAL_DIFFUSION_DT_MIN, CHEMICAL_DIFFUSION_DT_MAX);

        dim3 field_grid((arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, (arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM);
        dim3 field_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

        diffusion_reaction_kernel<<<field_grid, field_block>>>(
            chemical_field->concentration,
            chemical_field->gradient_x,
            chemical_field->gradient_y,
            chemical_field->laplacian,
            chemical_field->sources,
            arch.grid_size,
            chem_diff_dt,
            genome,
            gradients,
            genome_hash,
            ctx_metabolic,
            ctx_stress,
            ctx_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy
        );

        int field_size = arch.grid_size * arch.grid_size;
        float global_time = (float)generation;
        store_chemical_snapshot_kernel<<<field_grid, field_block>>>(chemical_field, field_size, global_time, genome_hash, genome);

        float* behavioral_field = organism->behavioral_field_pool;
        float* behavioral_gradients = organism->behavioral_gradient_pool;

        compute_behavioral_field_kernel<<<field_grid, field_block>>>(behavioral_field, agents, num_agents, arch.grid_size, primary_genome, organism->pool->entries[0].gradients,
            organism->pool->entries[0].genome_hash,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy, dims.hw_dim, dims.task_dim, dims.gen_dim);

        int total_warps = (arch.grid_size * arch.grid_size + WARP_SIZE - 1) / WARP_SIZE;
        dim3 warp_grid((total_warps + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE));
        dim3 warp_block(BLOCK_SIZE);
        warp_ca_kernel<<<warp_grid, warp_block>>>(
            behavioral_field,
            organism->behavioral_gradient_pool,
            arch.grid_size,
            arch.grid_size,
            genome,
            gradients,
            genome_hash,
            ctx_metabolic,
            ctx_stress,
            ctx_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy
        );

        dim3 grad_grid((arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, (arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, dims.total());
        behavioral_gradient_kernel<<<grad_grid, field_block>>>(organism->behavioral_gradient_pool,
            behavioral_gradients,
            arch.grid_size, dims.hw_dim, dims.task_dim, dims.gen_dim);

        int chemotaxis_dt_slot = derive_param_slot(genome_hash, "chemotaxis_dt");
        float chemotaxis_dt = genome_to_param(genome, gradients, chemotaxis_dt_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, CHEMOTAXIS_DT_MIN, CHEMOTAXIS_DT_MAX);

        chemotactic_navigation_kernel<<<(num_agents + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(agents, chemical_field->concentration,
            chemical_field->gradient_x,
            chemical_field->gradient_y,
            behavioral_gradients,
            num_agents,
            arch.grid_size,
            chemotaxis_dt,
            primary_genome,
            organism->pool->entries[0].gradients,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy, dims.hw_dim, dims.task_dim, dims.gen_dim);

        store_navigation_history_kernel<<<1, BLOCK_SIZE>>>(
            organism,
            agents,
            memory_tubes,
            generation,
            dims.hw_dim,
            dims.task_dim,
            dims.gen_dim
        );
    }
}

__global__ void store_navigation_history_kernel(
    Organism* organism,
    BehavioralState* agents,
    TemporalTube* tubes,
    int generation,
    int hw_dim,
    int task_dim,
    int gen_dim
) {
    int tid = threadIdx.x;
    if (tid >= MAX_COMPONENTS) return;

    int behavioral_dim = hw_dim + task_dim + gen_dim;
    int memory_entry_size = behavioral_dim + AGENT_SPATIAL_DIMS;
    float* d_memory_data = organism->memory_data_pool + tid * memory_entry_size;

    d_memory_data[0] = agents[tid].position[0];
    d_memory_data[1] = agents[tid].position[1];
    d_memory_data[2] = agents[tid].velocity[0];
    d_memory_data[3] = agents[tid].velocity[1];

    int offset = AGENT_SPATIAL_DIMS;
    for (int i = 0; i < hw_dim; i++) {
        d_memory_data[offset++] = agents[tid].hw_coords[i];
    }
    for (int i = 0; i < task_dim; i++) {
        d_memory_data[offset++] = agents[tid].task_coords[i];
    }
    for (int i = 0; i < gen_dim; i++) {
        d_memory_data[offset++] = agents[tid].gen_coords[i];
    }

    float importance = agents[tid].exploration_noise;

    if (tid == 0) {
        store_memory_kernel<<<1, 1>>>(
            tubes,
            d_memory_data,
            memory_entry_size,
            importance
        );
    }
}

__global__ void memory_update_kernel(
    TemporalTube* tubes,
    float* fitness_history,
    int* valid_flags_workspace,
    int* scan_workspace,
    MemoryEntry* temp_buffer,
    int generation,
    float* genome,
    float* gradients,
    uint64_t genome_hash,
    float ctx_metabolic,
    float ctx_stress,
    float ctx_morphogen,
    float ctx_complexity,
    float ctx_niche,
    float ctx_learning,
    float ctx_performance
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int old_count = tubes->count;

        int consolidation_slot = derive_param_slot(genome_hash, "memory_consolidation_threshold");
        float consolidation_threshold = genome_to_param(genome, gradients, consolidation_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CONSOLIDATION_THRESHOLD_MIN, CONSOLIDATION_THRESHOLD_MAX);

        int flow_lenia_dt_slot = derive_param_slot(genome_hash, "flow_lenia_dt");
        float flow_lenia_dt = genome_to_param(genome, gradients, flow_lenia_dt_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, FLOW_LENIA_DT_MIN, FLOW_LENIA_DT_MAX);

        apply_decay_kernel<<<(tubes->count + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
            tubes,
            flow_lenia_dt
        );

        int decay_threshold_slot = derive_param_slot(genome_hash, "memory_decay_threshold");
        float decay_threshold = genome_to_param(genome, gradients, decay_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, DECAY_THRESHOLD_MIN, DECAY_THRESHOLD_MAX);

        prune_memories_kernel<<<(tubes->count + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
            tubes,
            decay_threshold
        );

        int post_prune_count = tubes->count;

        consolidate_memories_kernel<<<1, min(BLOCK_SIZE, tubes->count)>>>(
            tubes,
            consolidation_threshold
        );

        int post_consolidate_count = tubes->count;

        int pruning_threshold_slot = derive_param_slot(genome_hash, "memory_pruning_threshold");
        float pruning_threshold = genome_to_param(genome, gradients, pruning_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, PRUNING_THRESHOLD_MIN, PRUNING_THRESHOLD_MAX);

        int archive_pruning_interval_slot = derive_param_slot(genome_hash, "archive_pruning_interval");
        float interval_norm = (genome[archive_pruning_interval_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        int archive_pruning_interval = ARCHIVE_PRUNING_INTERVAL_MIN + (int)(interval_norm * (ARCHIVE_PRUNING_INTERVAL_MAX - ARCHIVE_PRUNING_INTERVAL_MIN));

        if (post_consolidate_count < old_count * pruning_threshold && generation % archive_pruning_interval == 0) {
            printf("[GEN %3d] Memory pruning: %d->%d->%d entries (pruned %d, consolidated %d)\n",
                   generation, old_count, post_prune_count, post_consolidate_count,
                   old_count - post_prune_count, post_prune_count - post_consolidate_count);
        }

        if (generation % TELEMETRY_DETAILED == 0 && tubes->count > 0) {
            int pre_compact_count = tubes->count;

            prune_and_compact_memories_kernel<<<(tubes->count + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
                tubes,
                valid_flags_workspace,
                scan_workspace,
                temp_buffer,
                decay_threshold
            );

            printf("[GEN %3d] Memory compaction: %d->%d entries (removed %d low-value memories)\n",
                   generation, pre_compact_count, tubes->count, pre_compact_count - tubes->count);
        }

    }
}

__global__ void init_behavioral_dimensions_kernel(
    Organism* organism,
    float* workspace_genomes
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float* primary_genome = &workspace_genomes[GENOME_SIZE * 2];
        float* primary_parent_temp = &workspace_genomes[GENOME_SIZE * 3];
        reconstruct_genome_from_archive(
            organism->pool->entries[0].parent_hash,
            (GPUElite*)organism->archive,
            organism->archive_size,
            organism->pool->entries[0].delta_indices,
            organism->pool->entries[0].delta_values,
            organism->pool->entries[0].num_deltas,
            organism->pool->entries[0].max_deltas,
            primary_genome,
            GENOME_SIZE,
            primary_parent_temp,
            organism->diresa_genome_weights
        );

        BehavioralDimensions dims;
        dims.derive_from_genome(organism->pool->entries[0].genome_hash, primary_genome);

        organism->archive->hw_dim = dims.hw_dim;
        organism->archive->task_dim = dims.task_dim;
        organism->archive->gen_dim = dims.gen_dim;

        float* behavioral_hw_coords_buffer;
        float* behavioral_task_coords_buffer;
        float* behavioral_gen_coords_buffer;
        cudaMalloc(&behavioral_hw_coords_buffer, sizeof(float) * MAX_COMPONENTS * dims.hw_dim);
        cudaMalloc(&behavioral_task_coords_buffer, sizeof(float) * MAX_COMPONENTS * dims.task_dim);
        cudaMalloc(&behavioral_gen_coords_buffer, sizeof(float) * MAX_COMPONENTS * dims.gen_dim);

        for (int i = 0; i < MAX_COMPONENTS; i++) {
            organism->behavioral_agents[i].hw_coords = &behavioral_hw_coords_buffer[i * dims.hw_dim];
            organism->behavioral_agents[i].task_coords = &behavioral_task_coords_buffer[i * dims.task_dim];
            organism->behavioral_agents[i].gen_coords = &behavioral_gen_coords_buffer[i * dims.gen_dim];
        }

        // Continue with remaining allocations...
    }
}

__global__ void init_organism_kernel(
    Organism* organism,
    Dataset** dataset_array,
    unsigned int seed,
    float* workspace_genomes
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        organism->generation = 0;
        organism->active_components = MIN_POOL_SIZE;
        cudaMalloc(&organism->pool, sizeof(ComponentPool));
        cudaMalloc(&organism->pool->entries, sizeof(PoolEntry) * MAX_POOL_SIZE);
        organism->pool->capacity = MAX_POOL_SIZE;
        *((int*)&organism->pool->active_count) = MIN_POOL_SIZE;
        *((int*)&organism->pool->total_spawned) = 0;
        *((int*)&organism->pool->total_culled) = 0;

        cudaMalloc(&organism->archive, sizeof(GPUElite));
        cudaMalloc(&organism->voronoi_cells, sizeof(VoronoiCell) * MAX_COMPONENTS);

        cudaMalloc(&organism->memory_tubes, sizeof(TemporalTube));
        cudaMalloc(&organism->memory_tubes->entries, sizeof(MemoryEntry) * MAX_HISTORY_LENGTH);

        cudaMalloc(&organism->behavioral_agents, sizeof(BehavioralState) * MAX_COMPONENTS);

        organism->archive_size = 0;
        organism->num_voronoi_cells = MAX_COMPONENTS;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ALLOC] FAILED basic structures: %s\n", cudaGetErrorString(err));
            return;
        }

        PoolBuffers* pool_buffers_device;
        err = cudaMalloc(&pool_buffers_device, sizeof(PoolBuffers));
        if (err != cudaSuccess) {
            printf("[organism] pool_buffers struct alloc_err=%s size=%llu\n", cudaGetErrorString(err), (unsigned long long)sizeof(PoolBuffers));
            return;
        }

        init_pool_buffers_kernel<<<1, 1>>>(organism->pool, MAX_POOL_SIZE, pool_buffers_device);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[organism] init_pool_buffers launch_err=%s\n", cudaGetErrorString(err));
            return;
        }
    }
}

__global__ void init_organism_phase2_kernel(
    Organism* organism,
    Dataset** dataset_array,
    unsigned int seed,
    float* workspace_genomes
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cudaError_t err;

        float* primary_genome = &workspace_genomes[GENOME_SIZE * 2];
        float* primary_parent_temp = &workspace_genomes[GENOME_SIZE * 3];
        reconstruct_genome_from_archive(
            organism->pool->entries[0].parent_hash,
            (GPUElite*)organism->archive,
            organism->archive_size,
            organism->pool->entries[0].delta_indices,
            organism->pool->entries[0].delta_values,
            organism->pool->entries[0].num_deltas,
            organism->pool->entries[0].max_deltas,
            primary_genome,
            GENOME_SIZE,
            primary_parent_temp,
            organism->diresa_genome_weights
        );

        BehavioralDimensions dims;
        dims.derive_from_genome(organism->pool->entries[0].genome_hash, primary_genome);
        printf("[organism] behavioral_dims: hw=%d task=%d gen=%d total=%d\n", dims.hw_dim, dims.task_dim, dims.gen_dim, dims.total());

        organism->archive->hw_dim = dims.hw_dim;
        organism->archive->task_dim = dims.task_dim;
        organism->archive->gen_dim = dims.gen_dim;

        // Allocate contiguous buffers for all behavioral agents (avoid 768 individual mallocs)
        float* behavioral_hw_coords_buffer;
        float* behavioral_task_coords_buffer;
        float* behavioral_gen_coords_buffer;
        cudaMalloc(&behavioral_hw_coords_buffer, sizeof(float) * MAX_COMPONENTS * dims.hw_dim);
        cudaMalloc(&behavioral_task_coords_buffer, sizeof(float) * MAX_COMPONENTS * dims.task_dim);
        cudaMalloc(&behavioral_gen_coords_buffer, sizeof(float) * MAX_COMPONENTS * dims.gen_dim);

        // Set pointers into contiguous buffers
        for (int i = 0; i < MAX_COMPONENTS; i++) {
            organism->behavioral_agents[i].hw_coords = &behavioral_hw_coords_buffer[i * dims.hw_dim];
            organism->behavioral_agents[i].task_coords = &behavioral_task_coords_buffer[i * dims.task_dim];
            organism->behavioral_agents[i].gen_coords = &behavioral_gen_coords_buffer[i * dims.gen_dim];
        }

        cudaMalloc(&organism->archive->fitness, sizeof(float) * MAX_ARCHIVE_SIZE);
        cudaMalloc(&organism->archive->coherence, sizeof(float) * MAX_ARCHIVE_SIZE);
        cudaMalloc(&organism->archive->effective_rank, sizeof(float) * MAX_ARCHIVE_SIZE);
        cudaMalloc(&organism->archive->genome_hash, sizeof(uint64_t) * MAX_ARCHIVE_SIZE);
        cudaMalloc(&organism->archive->parent_ids, sizeof(uint32_t) * MAX_ARCHIVE_SIZE * PARENT_COUNT);
        cudaMalloc(&organism->archive->generation, sizeof(uint16_t) * MAX_ARCHIVE_SIZE);
        cudaMalloc(&organism->archive->hw_coords, sizeof(float) * MAX_ARCHIVE_SIZE * dims.hw_dim);
        cudaMalloc(&organism->archive->task_coords, sizeof(float) * MAX_ARCHIVE_SIZE * dims.task_dim);
        cudaMalloc(&organism->archive->gen_coords, sizeof(float) * MAX_ARCHIVE_SIZE * dims.gen_dim);
        cudaMalloc(&organism->archive->latent_genome, sizeof(float) * MAX_ARCHIVE_SIZE * GENOME_LATENT_DIM_MAX);
        cudaMalloc(&organism->archive->hardware_features, sizeof(float) * MAX_ARCHIVE_SIZE * (WMMA_TILE_DIM - 1));
        cudaMalloc(&organism->archive->task_performance, sizeof(float) * MAX_ARCHIVE_SIZE);
        cudaMalloc(&organism->archive->per_class_accuracy, sizeof(float) * MAX_ARCHIVE_SIZE * NUM_CLASSES_MAX);

        // Allocate contiguous buffers for all voronoi cell centroids (avoid 768 individual mallocs)
        float* voronoi_hw_centroid_buffer;
        float* voronoi_task_centroid_buffer;
        float* voronoi_gen_centroid_buffer;
        cudaMalloc(&voronoi_hw_centroid_buffer, sizeof(float) * MAX_COMPONENTS * dims.hw_dim);
        cudaMalloc(&voronoi_task_centroid_buffer, sizeof(float) * MAX_COMPONENTS * dims.task_dim);
        cudaMalloc(&voronoi_gen_centroid_buffer, sizeof(float) * MAX_COMPONENTS * dims.gen_dim);

        // Set pointers into contiguous buffers
        for (int i = 0; i < MAX_COMPONENTS; i++) {
            organism->voronoi_cells[i].hw_centroid = &voronoi_hw_centroid_buffer[i * dims.hw_dim];
            organism->voronoi_cells[i].task_centroid = &voronoi_task_centroid_buffer[i * dims.task_dim];
            organism->voronoi_cells[i].gen_centroid = &voronoi_gen_centroid_buffer[i * dims.gen_dim];
        }

        int default_decay_rate_slot = derive_param_slot(organism->pool->entries[0].genome_hash, "memory_default_decay_rate");
        float default_decay_rate_norm = (primary_genome[default_decay_rate_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float default_decay_rate = DEFAULT_DECAY_RATE_MIN + default_decay_rate_norm * (DEFAULT_DECAY_RATE_MAX - DEFAULT_DECAY_RATE_MIN);

        init_tube_kernel<<<(MAX_HISTORY_LENGTH + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
            organism->memory_tubes,
            MAX_HISTORY_LENGTH,
            default_decay_rate
        );
        
        if (err != cudaSuccess) {
            printf("[organism] init_tube launch_err=%s\n", cudaGetErrorString(err));
            return;
        }

        ArchitectureParams arch;
        arch.num_heads = organism->pool->entries[0].num_heads;
        arch.channels = organism->pool->entries[0].channels;
        arch.hidden_dim = organism->pool->entries[0].hidden_dim;
        arch.head_dim = organism->pool->entries[0].head_dim;
        arch.grid_size = organism->pool->entries[0].grid_size;
        int field_size = arch.grid_size * arch.grid_size;
        printf("[organism] arch from pool[0]: heads=%d channels=%d hidden=%d head_dim=%d grid=%d field_size=%d\n", arch.num_heads, arch.channels, arch.hidden_dim, arch.head_dim, arch.grid_size, field_size);

        cudaMalloc(&organism->ca_state, sizeof(MultiHeadCAState));
        cudaMalloc(&organism->chemical_field, sizeof(ChemicalField));

        cudaMalloc(&organism->chemical_field->history, sizeof(TemporalTube));
        cudaMalloc(&organism->chemical_field->history->entries, sizeof(MemoryEntry) * MAX_HISTORY_LENGTH);

        init_tube_kernel<<<(MAX_HISTORY_LENGTH + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
            organism->chemical_field->history,
            MAX_HISTORY_LENGTH,
            default_decay_rate
        );

        if (err != cudaSuccess) {
            printf("[ERROR] init_tube_kernel for chemical_field->history failed: %s\n", cudaGetErrorString(err));
            return;
        }

        // Allocate contiguous buffer for all history entry data
        float* history_data_buffer;
        cudaMalloc(&history_data_buffer, field_size * sizeof(float) * MAX_HISTORY_LENGTH);
        for (int i = 0; i < MAX_HISTORY_LENGTH; i++) {
            organism->chemical_field->history->entries[i].data = &history_data_buffer[i * field_size];
        }

        int perception_size = arch.num_heads * arch.channels * arch.hidden_dim;
        int interaction_size = arch.num_heads * arch.hidden_dim * arch.hidden_dim;
        int value_size = arch.num_heads * arch.hidden_dim * arch.head_dim;
        int total_weights_size = perception_size + interaction_size + value_size;

        half* all_ca_weights;
        cudaMalloc(&all_ca_weights, sizeof(half) * total_weights_size);
        organism->ca_state->perception_weights = all_ca_weights;
        organism->ca_state->interaction_weights = all_ca_weights + perception_size;
        organism->ca_state->value_weights = all_ca_weights + perception_size + interaction_size;

        int ca_conc_size = field_size * arch.channels;
        int ca_output_size = arch.num_heads * field_size * arch.head_dim;
        int ca_state_total = ca_conc_size + ca_output_size + field_size + field_size * 2 + ca_conc_size;

        float* all_ca_state;
        cudaMalloc(&all_ca_state, sizeof(float) * ca_state_total);
        organism->ca_state->ca_concentration = all_ca_state;
        organism->ca_state->ca_output = all_ca_state + ca_conc_size;
        organism->ca_state->affinity_reduced = all_ca_state + ca_conc_size + ca_output_size;
        organism->ca_state->flow_field = all_ca_state + ca_conc_size + ca_output_size + field_size;
        organism->ca_state->reintegration_buffer = all_ca_state + ca_conc_size + ca_output_size + field_size + field_size * 2;

        int chem_total = field_size * 6;
        float* all_chem_fields;
        cudaMalloc(&all_chem_fields, sizeof(float) * chem_total);
        organism->chemical_field->concentration = all_chem_fields;
        organism->chemical_field->gradient_x = all_chem_fields + field_size;
        organism->chemical_field->gradient_y = all_chem_fields + field_size * 2;
        organism->chemical_field->laplacian = all_chem_fields + field_size * 3;
        organism->chemical_field->sources = all_chem_fields + field_size * 4;
        organism->chemical_field->decay_factors = all_chem_fields + field_size * 5;

        cudaMalloc(&organism->fitness_history, sizeof(float) * 100 * MAX_COMPONENTS);
        cudaMalloc(&organism->effective_rank_history, sizeof(float) * 100);
        cudaMalloc(&organism->coherence_history, sizeof(float) * 100 * MAX_COMPONENTS);

        int rd_fields_total = field_size * 7;
        float* all_rd_fields;
        cudaMalloc(&all_rd_fields, sizeof(float) * rd_fields_total);
        organism->resource_density = all_rd_fields;
        organism->resource_next = all_rd_fields + field_size;
        organism->fitness_landscape = all_rd_fields + field_size * 2;
        organism->rd_u_field = all_rd_fields + field_size * 3;
        organism->rd_v_field = all_rd_fields + field_size * 4;
        organism->rd_u_next = all_rd_fields + field_size * 5;
        organism->rd_v_next = all_rd_fields + field_size * 6;

        int max_workspace_size = MAX_COMPONENTS * MAX_COMPONENTS;
        float* shared_workspace;
        cudaMalloc(&shared_workspace, sizeof(float) * max_workspace_size);
        organism->coherence_workspace_pool = shared_workspace;
        organism->correlation_matrix_pool = shared_workspace;
        organism->fitness_workspace_pool = shared_workspace;

        cudaMalloc(&organism->lifecycle_states, sizeof(LocalOrganismState<BLOCK_SIZE>) * ((MAX_POOL_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE));

        cudaMalloc(&organism->telemetry, sizeof(TelemetryBuffer));
        organism->telemetry->valid = false;
        organism->telemetry->generation = 0;

        // Allocate DIRESA weights using genome-derived parameters from pool entry 0
        PoolEntry* first_entry = &organism->pool->entries[0];

        // Genome-derive classifier output dimensions and dataset sample dimensions
        int num_classes_slot = derive_param_slot(first_entry->genome_hash, "num_classes");
        int num_classes = NUM_CLASSES_MIN +
            ((primary_genome[num_classes_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE) *
            (NUM_CLASSES_MAX - NUM_CLASSES_MIN);

        organism->telemetry->memory_allocation.total_gpu_allocated = 0;
        organism->telemetry->memory_allocation.archive_pools_size = 0;
        organism->telemetry->memory_allocation.training_pools_size = 0;
        organism->telemetry->memory_allocation.ca_state_size = 0;
        organism->telemetry->memory_allocation.behavioral_pools_size = 0;
        organism->telemetry->memory_allocation.diresa_weights_size = 0;
        organism->telemetry->memory_allocation.autodiff_tape_size = 0;

        size_t diresa_struct_size = sizeof(DIRESAWeights);
        int num_replicas = first_entry->num_tempering_replicas;
        size_t diresa_size = diresa_struct_size * num_replicas;
        size_t diresa_mb = diresa_size / BYTES_PER_MB;

        err = cudaMalloc(&organism->diresa_hw_weights, diresa_size);
        track_allocation("diresa_hw_weights", organism->diresa_hw_weights, diresa_size, err, &organism->telemetry->memory_allocation);
        if (err != cudaSuccess) {
            printf("[organism] diresa_hw alloc_err=%s size=%llu replicas=%d\n", cudaGetErrorString(err), (unsigned long long)diresa_size, num_replicas);
            return;
        }

        err = cudaMalloc(&organism->diresa_task_weights, diresa_size);
        track_allocation("diresa_task_weights", organism->diresa_task_weights, diresa_size, err, &organism->telemetry->memory_allocation);
        if (err != cudaSuccess) {
            printf("[organism] diresa_task alloc_err=%s size=%llu replicas=%d\n", cudaGetErrorString(err), (unsigned long long)diresa_size, num_replicas);
            return;
        }

        err = cudaMalloc(&organism->diresa_gen_weights, diresa_size);
        track_allocation("diresa_gen_weights", organism->diresa_gen_weights, diresa_size, err, &organism->telemetry->memory_allocation);
        if (err != cudaSuccess) {
            printf("[organism] diresa_gen alloc_err=%s size=%llu replicas=%d\n", cudaGetErrorString(err), (unsigned long long)diresa_size, num_replicas);
            return;
        }

        err = cudaMalloc(&organism->diresa_genome_weights, diresa_size);
        track_allocation("diresa_genome_weights", organism->diresa_genome_weights, diresa_size, err, &organism->telemetry->memory_allocation);
        if (err != cudaSuccess) {
            printf("[organism] diresa_genome alloc_err=%s size=%llu replicas=%d\n", cudaGetErrorString(err), (unsigned long long)diresa_size, num_replicas);
            return;
        }

        organism->telemetry->memory_allocation.diresa_weights_size = diresa_size * 4;

        init_diresa_kernel<<<num_replicas, 1024>>>(
            organism->diresa_hw_weights, HARDWARE_FEATURES_DIM, BEHAVIORAL_DIM_HW_MAX, first_entry, seed + 999999);
        init_diresa_kernel<<<num_replicas, 1024>>>(
            organism->diresa_task_weights, BEHAVIORAL_DIM_TASK_MAX, BEHAVIORAL_DIM_TASK_MAX, first_entry, seed + 888888);
        init_diresa_kernel<<<num_replicas, 1024>>>(
            organism->diresa_gen_weights, BEHAVIORAL_DIM_GEN_MAX, BEHAVIORAL_DIM_GEN_MAX, first_entry, seed + 777777);
        init_diresa_kernel<<<num_replicas, 1024>>>(
            organism->diresa_genome_weights, GENOME_SIZE, GENOME_LATENT_DIM_MAX, first_entry, seed + 666666);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[organism] init_diresa launch_err=%s blocks=%d\n", cudaGetErrorString(err), num_replicas);
            return;
        }

        // Allocate archive compression pools on device
        // DIRESA latent genome storage (replaces SVD compressed_genome_pool)
        TRACKED_ALLOC(organism->latent_genome_pool, sizeof(float) * MAX_ARCHIVE_SIZE * GENOME_LATENT_DIM_MAX, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.archive_pools_size);

        // Allocate behavioral/memory pools with complete tracking
        int behavioral_dim = dims.total();
        TRACKED_ALLOC(organism->behavioral_field_pool, sizeof(float) * field_size * behavioral_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->behavioral_gradient_pool, sizeof(float) * field_size * behavioral_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->memory_data_pool, sizeof(float) * MAX_COMPONENTS * (behavioral_dim + AGENT_SPATIAL_DIMS), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->prediction_error_history, sizeof(float) * TELEMETRY_DETAILED, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->trace_buffer, sizeof(TraceBuffer), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->hardware_geom, sizeof(HardwareGeometry), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->delta_indices_pool, sizeof(uint16_t) * MAX_POOL_SIZE * MAX_DELTAS_PER_ENTRY, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->delta_values_pool, sizeof(float) * MAX_POOL_SIZE * MAX_DELTAS_PER_ENTRY, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->delta_counts_pool, sizeof(uint16_t) * MAX_POOL_SIZE, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->memory_compaction_valid_flags, sizeof(int) * MAX_HISTORY_LENGTH, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->memory_compaction_scan, sizeof(int) * MAX_HISTORY_LENGTH, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->memory_compaction_buffer, sizeof(MemoryEntry) * MAX_HISTORY_LENGTH, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);

        // Allocate CA workspace
        int weight_count = arch.num_heads * arch.channels * arch.hidden_dim;
        TRACKED_ALLOC(organism->fp32_ca_workspace, sizeof(float) * weight_count, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.ca_state_size);
        TRACKED_ALLOC(organism->fp16_ca_workspace, sizeof(half) * weight_count, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.ca_state_size);

        // Allocate SVD/fitness pools
        TRACKED_ALLOC(organism->fitness_rank_pool, sizeof(float) * MAX_COMPONENTS, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);
        TRACKED_ALLOC(organism->fitness_coherence_pool, sizeof(float) * MAX_COMPONENTS, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);

        // Allocate autodiff tape system
        TRACKED_ALLOC(organism->ad_tape, sizeof(ADTape), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.autodiff_tape_size);
        TRACKED_ALLOC(organism->ad_tape_entries_pool, sizeof(TapeEntry) * MAX_TAPE_SIZE, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.autodiff_tape_size);
        TRACKED_ALLOC(organism->ad_tape_values_pool, sizeof(float) * MAX_TAPE_VALUES, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.autodiff_tape_size);
        TRACKED_ALLOC(organism->ad_tape_grads_pool, sizeof(float) * MAX_TAPE_VALUES, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.autodiff_tape_size);
        init_ad_tape_kernel<<<1, 1>>>(organism->ad_tape, organism->ad_tape_entries_pool, organism->ad_tape_values_pool, organism->ad_tape_grads_pool, MAX_TAPE_SIZE, MAX_TAPE_VALUES);

        // Allocate CA parameter map
        TRACKED_ALLOC(organism->param_map, sizeof(CAParameterMap), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.ca_state_size);
        init_ca_param_map_kernel<<<1, 1>>>(organism->param_map, arch);

        // Allocate lifecycle phase tracking
        TRACKED_ALLOC(organism->lifecycle_phase_counts, sizeof(int) * 8, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.behavioral_pools_size);

        // Allocate gradient pools
        int batch_size = 64;
        TRACKED_ALLOC(organism->gradient_features_pool, sizeof(float) * batch_size * HARDWARE_FEATURES_DIM, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->gradient_logits_pool, sizeof(float) * batch_size * num_classes, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->gradient_loss_pool, sizeof(float) * batch_size, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->gradient_logit_grads_pool, sizeof(float) * batch_size * num_classes, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->gradient_magnitudes_pool, sizeof(float) * MAX_COMPONENTS, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);

        // Allocate classifier gradient buffers
        TRACKED_ALLOC(organism->pooling_weights_grad, sizeof(float) * arch.channels, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->fc_weights_grad, sizeof(float) * num_classes * dims.hw_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->fc_bias_grad, sizeof(float) * num_classes, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->features_grad, sizeof(float) * batch_size * dims.hw_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);

        // Allocate Adam optimizer state for CA weights
        TRACKED_ALLOC(organism->adam_m_perception_pool, sizeof(float) * arch.num_heads * arch.channels * arch.hidden_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_v_perception_pool, sizeof(float) * arch.num_heads * arch.channels * arch.hidden_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_m_interaction_pool, sizeof(float) * arch.num_heads * arch.hidden_dim * arch.hidden_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_v_interaction_pool, sizeof(float) * arch.num_heads * arch.hidden_dim * arch.hidden_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_m_value_pool, sizeof(float) * arch.num_heads * arch.hidden_dim * arch.head_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_v_value_pool, sizeof(float) * arch.num_heads * arch.hidden_dim * arch.head_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_m_classifier_pool, sizeof(float) * num_classes * dims.hw_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_v_classifier_pool, sizeof(float) * num_classes * dims.hw_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);

        // Allocate Adam optimizer state for classifier weights
        TRACKED_ALLOC(organism->adam_m_pooling, sizeof(float) * arch.channels, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_v_pooling, sizeof(float) * arch.channels, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_m_fc_weights, sizeof(float) * num_classes * dims.hw_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_v_fc_weights, sizeof(float) * num_classes * dims.hw_dim, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_m_fc_bias, sizeof(float) * num_classes, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->adam_v_fc_bias, sizeof(float) * num_classes, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);

        // Allocate batch training pools
        TRACKED_ALLOC(organism->batch_ca_states_pool, sizeof(float) * batch_size * field_size * arch.channels, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->batch_labels_pool, sizeof(int) * batch_size, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->task_loss_pool, sizeof(float), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->reg_loss_pool, sizeof(float), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->rank_loss_pool, sizeof(float), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->coherence_loss_pool, sizeof(float), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->diversity_loss_pool, sizeof(float), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        TRACKED_ALLOC(organism->total_loss_pool, sizeof(float), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);

        // Allocate training mode controller
        TRACKED_ALLOC(organism->training_mode, sizeof(HybridTrainingMode), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);
        init_training_mode_kernel<<<1, 1>>>(organism->training_mode, organism->pool->entries[0].grid_size);

        // Allocate classification head
        TRACKED_ALLOC(organism->classifier, sizeof(ClassificationHead), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);

        // Allocate classifier workspace: pooling_weights + fc_weights + fc_bias
        size_t classifier_workspace_size = sizeof(float) * (dims.hw_dim + (dims.hw_dim * num_classes) + num_classes);
        float* classifier_workspace;
        TRACKED_ALLOC(classifier_workspace, classifier_workspace_size, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);

        init_classifier_kernel<<<1, 1>>>(organism->classifier, dims.hw_dim, num_classes, seed + 777777, classifier_workspace);

        // Allocate adaptive curriculum
        TRACKED_ALLOC(organism->curriculum, sizeof(AdaptiveCurriculum), &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);

        // Allocate voronoi occupancy histogram for curriculum tracking
        TRACKED_ALLOC(organism->voronoi_occupancy_histogram, sizeof(float) * organism->num_voronoi_cells, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);

        // Allocate pool task accuracies buffer for curriculum tracking
        TRACKED_ALLOC(organism->pool_task_accuracies, sizeof(float) * organism->pool->capacity, &organism->telemetry->memory_allocation, organism->telemetry->memory_allocation.training_pools_size);

        // Print memory allocation summary
        print_size("[MEM SUMMARY] Total GPU allocated", "", organism->telemetry->memory_allocation.total_gpu_allocated, "");
        print_size("[MEM SUMMARY]   ", "Archive pools", organism->telemetry->memory_allocation.archive_pools_size, "");
        print_size("[MEM SUMMARY]   ", "Behavioral pools", organism->telemetry->memory_allocation.behavioral_pools_size, "");
        print_size("[MEM SUMMARY]   ", "CA state", organism->telemetry->memory_allocation.ca_state_size, "");
        print_size("[MEM SUMMARY]   ", "Training pools", organism->telemetry->memory_allocation.training_pools_size, "");
        print_size("[MEM SUMMARY]   ", "DIRESA weights", organism->telemetry->memory_allocation.diresa_weights_size, "");
        print_size("[MEM SUMMARY]   ", "Autodiff tape", organism->telemetry->memory_allocation.autodiff_tape_size, "");

        // Store pre-loaded dataset array and set initial dataset
        organism->dataset_array = dataset_array;
        organism->current_dataset = dataset_array[0];

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] Full system allocation failed: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ALLOC] FAILED: %s\n", cudaGetErrorString(err));
            return;
        }

        init_multihead_ca_kernel<<<(arch.num_heads * arch.channels * arch.hidden_dim + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
            organism->ca_state,
            seed,
            arch
        );

        if (err != cudaSuccess) {
            printf("[ERROR] init_multihead_ca_kernel failed: %s\n", cudaGetErrorString(err));
            return;
        }

        BehavioralInitSlots behavioral_slots;
        behavioral_slots.agent_embedding_scale = derive_param_slot(organism->pool->entries[0].genome_hash, "chemotaxis_agent_embedding_scale");
        behavioral_slots.init_exploration = derive_param_slot(organism->pool->entries[0].genome_hash, "chemotaxis_init_exploration");
        behavioral_slots.init_sensitivity = derive_param_slot(organism->pool->entries[0].genome_hash, "chemotaxis_init_sensitivity");
        behavioral_slots.ctx_metabolic = derive_param_slot(organism->pool->entries[0].genome_hash, "init_context_metabolic");
        behavioral_slots.ctx_stress = derive_param_slot(organism->pool->entries[0].genome_hash, "init_context_stress");
        behavioral_slots.ctx_morphogen = derive_param_slot(organism->pool->entries[0].genome_hash, "init_context_morphogen");

        init_behavioral_state_kernel<<<(MAX_COMPONENTS + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(organism->behavioral_agents,
            MAX_COMPONENTS,
            seed,
            primary_genome,
            organism->pool->entries[0].gradients,
            organism->pool->entries[0].genome_hash,
            0,
            behavioral_slots,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy, dims.hw_dim, dims.task_dim, dims.gen_dim);
        printf("[DEVICE-DBG] init_behavioral_state_kernel launch returned, calling sync...\n");
        
        printf("[DEVICE-DBG] cudaDeviceSynchronize returned: %d (%s)\n", (int)err, cudaGetErrorString(err));
        if (err != cudaSuccess) {
            printf("[ERROR] init_behavioral_state_kernel failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[DEVICE] init_behavioral_state_kernel completed\n");

        dim3 chem_grid((arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, (arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM);
        dim3 chem_block(WMMA_TILE_DIM, WMMA_TILE_DIM);
        printf("[DEVICE] Launching init_chemical_field_kernel with grid=(%d,%d) block=(%d,%d)...\n", chem_grid.x, chem_grid.y, chem_block.x, chem_block.y);
        init_chemical_field_kernel<<<chem_grid, chem_block>>>(
            organism->chemical_field,
            arch.grid_size,
            primary_genome,
            organism->pool->entries[0].gradients,
            organism->pool->entries[0].genome_hash,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy
        );

        if (err != cudaSuccess) {
            printf("[ERROR] init_chemical_field_kernel failed: %s\n", cudaGetErrorString(err));
            return;
        }

        set_chemical_sources_from_agents_kernel<<<1, MAX_COMPONENTS>>>(
            organism->chemical_field->sources,
            organism->behavioral_agents,
            MAX_COMPONENTS,
            arch.grid_size,
            primary_genome,
            organism->pool->entries[0].gradients,
            organism->chemical_field->concentration,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy
        );

        if (err != cudaSuccess) {
            printf("[ERROR] set_chemical_sources_from_agents_kernel failed: %s\n", cudaGetErrorString(err));
            return;
        }

        int voronoi_init_dt_slot = derive_param_slot(organism->pool->entries[0].genome_hash, "voronoi_init_dt");
        float voronoi_init_dt_norm = (primary_genome[voronoi_init_dt_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float voronoi_init_dt = VORONOI_INIT_DT_MIN + voronoi_init_dt_norm * (VORONOI_INIT_DT_MAX - VORONOI_INIT_DT_MIN);

        uint64_t init_genome_hash = organism->pool->entries[0].genome_hash;
        int ctx_metabolic_slot = derive_param_slot(init_genome_hash, "init_ctx_metabolic");
        int ctx_stress_slot = derive_param_slot(init_genome_hash, "init_ctx_stress");
        int ctx_morphogen_slot = derive_param_slot(init_genome_hash, "init_ctx_morphogen");

        float ctx_metabolic = (primary_genome[ctx_metabolic_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float ctx_stress = (primary_genome[ctx_stress_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float ctx_morphogen = (primary_genome[ctx_morphogen_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;

        diffusion_reaction_kernel<<<chem_grid, chem_block>>>(
            organism->chemical_field->concentration,
            organism->chemical_field->gradient_x,
            organism->chemical_field->gradient_y,
            organism->chemical_field->laplacian,
            organism->chemical_field->sources,
            arch.grid_size,
            voronoi_init_dt,
            primary_genome,
            organism->pool->entries[0].gradients,
            organism->pool->entries[0].genome_hash,
            ctx_metabolic,
            ctx_stress,
            ctx_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy
        );

        if (err != cudaSuccess) {
            printf("[ERROR] diffusion_reaction_kernel failed: %s\n", cudaGetErrorString(err));
            return;
        }

        store_chemical_snapshot_kernel<<<chem_grid, chem_block>>>(organism->chemical_field, field_size, (float)organism->generation, organism->pool->entries[0].genome_hash, primary_genome);

        if (err != cudaSuccess) {
            printf("[ERROR] store_chemical_snapshot failed: %s\n", cudaGetErrorString(err));
            return;
        }

        init_curriculum_kernel<<<1, 1>>>(
            organism->curriculum,
            primary_genome,
            organism->pool->entries[0].gradients,
            organism->pool->entries[0].genome_hash,
            0.5f,  // ctx_metabolic - initial neutral value
            0.1f,  // ctx_stress - initial low stress
            0.3f,  // ctx_morphogen - initial low concentration
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy
        );
        
        if (err != cudaSuccess) {
            printf("[ERROR] init_curriculum_kernel failed: %s\n", cudaGetErrorString(err));
            return;
        }

    }
}

__global__ void check_convergence_kernel(
    Organism* organism,
    int generation,
    bool* converged,
    float* workspace_genomes
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float fitness = organism->fitness_history[generation * MAX_COMPONENTS];
        float coherence = organism->coherence_history[generation * MAX_COMPONENTS];

        float* convergence_genome = &workspace_genomes[0];
        float* convergence_parent_temp = &workspace_genomes[GENOME_SIZE];
        reconstruct_genome_from_archive(
            organism->pool->entries[0].parent_hash,
            (GPUElite*)organism->archive,
            organism->archive_size,
            organism->pool->entries[0].delta_indices,
            organism->pool->entries[0].delta_values,
            organism->pool->entries[0].num_deltas,
            organism->pool->entries[0].max_deltas,
            convergence_genome,
            GENOME_SIZE,
            convergence_parent_temp,
            organism->diresa_genome_weights
        );

        uint64_t genome_hash = organism->pool->entries[0].genome_hash;
        float* genome = convergence_genome;

        int fitness_conv_slot = derive_param_slot(genome_hash, "convergence_fitness_threshold");
        int coherence_conv_slot = derive_param_slot(genome_hash, "convergence_coherence_threshold");
        int fitness_min_slot = derive_param_slot(genome_hash, "convergence_fitness_min");
        int fitness_max_slot = derive_param_slot(genome_hash, "convergence_fitness_max");
        int coherence_min_slot = derive_param_slot(genome_hash, "convergence_coherence_min");
        int coherence_max_slot = derive_param_slot(genome_hash, "convergence_coherence_max");

        float fitness_min = (genome[fitness_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float fitness_max = (genome[fitness_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float coherence_min = (genome[coherence_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float coherence_max = (genome[coherence_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;

        int conv_ctx_metabolic_slot = derive_param_slot(genome_hash, "convergence_ctx_metabolic");
        int conv_ctx_stress_slot = derive_param_slot(genome_hash, "convergence_ctx_stress");
        int conv_ctx_morphogen_slot = derive_param_slot(genome_hash, "convergence_ctx_morphogen");

        float conv_ctx_metabolic = (genome[conv_ctx_metabolic_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float conv_ctx_stress = (genome[conv_ctx_stress_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float conv_ctx_morphogen = (genome[conv_ctx_morphogen_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;

        float fitness_threshold = genome_to_param(
            genome,
            organism->pool->entries[0].gradients,
            fitness_conv_slot,
            conv_ctx_metabolic, conv_ctx_stress, conv_ctx_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            fitness_min, fitness_max
        );

        float coherence_threshold = genome_to_param(
            genome,
            organism->pool->entries[0].gradients,
            coherence_conv_slot,
            conv_ctx_metabolic, conv_ctx_stress, conv_ctx_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            coherence_min, coherence_max
        );

        if (fitness > fitness_threshold && coherence > coherence_threshold) {
            printf("[CONVERGED] Gen %d: fitness=%.3f (>%.3f), coherence=%.3f (>%.3f)\n",
                   generation, fitness, fitness_threshold, coherence, coherence_threshold);
            *converged = true;
        }
    }
}

__global__ void persistent_evolution_kernel(
    unsigned int seed,
    Dataset** dataset_array
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    printf("[DEBUG-PERSISTENT-START] Entering persistent_evolution_kernel\n");

    Organism* organism;
    cudaError_t err = cudaMalloc(&organism, sizeof(Organism));
    if (err != cudaSuccess) {
        printf("[PERSISTENT-ERROR] cudaMalloc failed for Organism: %d\n", (int)err);
        return;
    }
    printf("[DEBUG-PERSISTENT] cudaMalloc organism done\n");

    float* organism_workspace_genomes;
    cudaMalloc(&organism_workspace_genomes, sizeof(float) * (2 * MAX_POOL_SIZE * GENOME_SIZE + 10 * GENOME_SIZE));
    printf("[DEBUG-PERSISTENT] cudaMalloc workspace done\n");

    printf("[DEBUG-PERSISTENT] Launching init_organism_kernel\n");
    init_organism_kernel<<<1, 1>>>(organism, dataset_array, seed, organism_workspace_genomes);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[PERSISTENT-ERROR] init_organism_phase1 launch failed: %d\n", (int)err);
        return;
    }
    printf("[DEBUG-PERSISTENT] init_organism_kernel launched, err=%d\n", (int)err);

    printf("[DEBUG-PERSISTENT] Launching init_organism_phase2_kernel\n");
    init_organism_phase2_kernel<<<1, 1>>>(organism, dataset_array, seed, organism_workspace_genomes);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[PERSISTENT-ERROR] init_organism_kernel launch failed: %d\n", (int)err);
        return;
    }
    printf("[DEBUG-PERSISTENT] init_organism_phase2_kernel launched, err=%d\n", (int)err);

    printf("[DEBUG-PERSISTENT] Calling cudaDeviceSynchronize\n");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[PERSISTENT-ERROR] cudaDeviceSynchronize failed: %d\n", (int)err);
        return;
    }
    printf("[DEBUG-PERSISTENT] cudaDeviceSynchronize complete, entering while loop\n");

    int generation = 0;
    while (true) {
        // Evolution pathway - every generation
        printf("[PERSISTENT-LOOP] gen=%d BEFORE organism_lifecycle_kernel\n", generation);
        organism_lifecycle_kernel<<<1, 1>>>(organism, generation, organism_workspace_genomes);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] organism_lifecycle_kernel launch failed at gen %d: error=%d %s\n",
                   generation, (int)err, cudaGetErrorString(err));
            break;
        }
        err = cudaDeviceSynchronize();
        printf("[PERSISTENT-LOOP] gen=%d AFTER organism_lifecycle_kernel err=%d\n", generation, (int)err);
        if (err != cudaSuccess) {
            printf("[ERROR] organism_lifecycle_kernel sync failed at gen %d: error=%d %s\n",
                   generation, (int)err, cudaGetErrorString(err));
            break;
        }

        // Gradient-based learning pathway - every CHECKPOINT_INTERVAL generations
        // Produces telemetry (classification accuracy, gradient magnitudes) for DIRESA
        if (generation % CHECKPOINT_INTERVAL == 0) {
            hybrid_organism_lifecycle_kernel<<<1, 1>>>(
                organism,
                organism->training_mode,
                organism->param_map,
                generation,
                organism_workspace_genomes
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("[ERROR] hybrid_organism_lifecycle_kernel failed at gen %d: error=%d\n",
                       generation, (int)err);
                break;
            }
        }

        // Update curriculum every CHECKPOINT_INTERVAL generations
        if (generation % CHECKPOINT_INTERVAL == 0 && generation > 0) {
            int num_blocks = (organism->pool->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Collect task accuracies from pool
            collect_pool_task_accuracies_kernel<<<num_blocks, BLOCK_SIZE>>>(
                organism->pool,
                organism->pool_task_accuracies
            );

            // Compute voronoi occupancy histogram
            int voronoi_blocks = (organism->num_voronoi_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
            compute_voronoi_occupancy_kernel<<<voronoi_blocks, BLOCK_SIZE>>>(
                organism->voronoi_cells,
                organism->num_voronoi_cells,
                organism->voronoi_occupancy_histogram
            );

            // Store previous dataset index
            int prev_dataset_idx = organism->curriculum->current_dataset_idx;

            // Update curriculum based on population performance
            update_curriculum_kernel<<<1, 1>>>(
                organism->curriculum,
                organism->pool_task_accuracies,
                organism->voronoi_occupancy_histogram,
                organism->pool->capacity,
                organism->num_voronoi_cells,
                generation
            );

            // If curriculum progressed to new dataset, switch to new pre-loaded dataset
            if (organism->curriculum->current_dataset_idx != prev_dataset_idx) {
                organism->current_dataset = organism->dataset_array[organism->curriculum->current_dataset_idx];
                int new_dataset_id = ACTIVE_DATASET_IDS[organism->curriculum->current_dataset_idx];
                printf("[CURRICULUM] Switched to pre-loaded dataset index %d (dataset_id=%d)\n",
                       organism->curriculum->current_dataset_idx, new_dataset_id);
            }
        }

        int active = Atomics::load_int(organism->pool->active_count);
        printf("[GEN %3d] active=%d archive=%d\n", generation, active, organism->archive_size);

        generation++;
    }

    printf("[SHUTDOWN] Persistent evolution terminated at generation %d\n", generation);
}

#endif

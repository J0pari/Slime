
#ifndef ORGANISM_CU
#define ORGANISM_CU
#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
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

#define CDP_LAUNCH_CHECK(kernel_name) \
    do { \
        cudaError_t _err = cudaGetLastError(); \
        if (_err != cudaSuccess) { \
\
            return; \
        } \
    } while(0)

namespace cg = cooperative_groups;

struct CAParameterMap;
struct HybridTrainingMode;
struct ClassificationHead;
struct AdaptiveCurriculum;
template<int SECTION_SIZE> struct LocalOrganismState;

struct MemoryUpdateParams {
    float flow_lenia_dt;
    float decay_threshold;
    float consolidation_threshold;
    int tube_count;
    int should_compact;
};

struct OrganismPreallocatedBuffers {
    ComponentPool* pool;
    PoolEntry* pool_entries;
    GPUElite* archive;
    VoronoiCell* voronoi_cells;
    BehavioralState* behavioral_agents;
    uint16_t* delta_indices_buffer;
    float* delta_values_buffer;
    float* gradients_buffer;
    float* behavioral_hw_coords_buffer;
    float* behavioral_task_coords_buffer;
    float* behavioral_gen_coords_buffer;
    float* archive_fitness;
    float* archive_coherence;
    float* archive_effective_rank;
    uint64_t* archive_genome_hash;
    uint32_t* archive_parent_ids;
    uint16_t* archive_generation;
    float* archive_hw_coords;
    float* archive_task_coords;
    float* archive_gen_coords;
    float* archive_latent_genome;
    float* archive_hardware_features;
    float* archive_task_performance;
    float* archive_per_class_accuracy;
    uint64_t* archive_hash_table_keys;
    int* archive_hash_table_values;
    float* hw_coords_pool;
    float* task_coords_pool;
    float* gen_coords_pool;
    float* voronoi_hw_centroid_buffer;
    float* voronoi_task_centroid_buffer;
    float* voronoi_gen_centroid_buffer;
    TelemetryBuffer* telemetry;
    MultiHeadCAState* ca_state_pool;
    ChemicalField* chemical_field;
    TemporalTube* chemical_field_history;
    MemoryEntry* chemical_field_history_entries;
    float* history_data_buffer;
    half* all_ca_weights;
    float* all_ca_state;
    float* all_chem_fields;
    float* fitness_history;
    float* effective_rank_history;
    float* coherence_history;
    float* all_rd_fields;
    float* shared_workspace;
    LocalOrganismState<BLOCK_SIZE>* lifecycle_states;
    DIRESAWeights* diresa_hw_weights;
    DIRESAWeights* diresa_task_weights;
    DIRESAWeights* diresa_gen_weights;
    DIRESAWeights* diresa_genome_weights;
    float* diresa_hw_weight_pool;
    float* diresa_task_weight_pool;
    float* diresa_gen_weight_pool;
    float* diresa_genome_weight_pool;
    float* fp32_ca_workspace;
    half* fp16_ca_workspace;
    float* latent_genome_pool;
    float* behavioral_field_pool;
    float* behavioral_gradient_pool;
    float* memory_data_pool;
    float* prediction_error_history;
    TraceBuffer* trace_buffer;
    ExecutionTrace* trace_array;
    HardwareGeometry* hardware_geom;
    uint16_t* delta_indices_pool;
    float* delta_values_pool;
    uint16_t* delta_counts_pool;
    int* memory_compaction_valid_flags;
    int* memory_compaction_scan;
    int* memory_compaction_recursive_workspace;
    MemoryEntry* memory_compaction_buffer;
    float* fitness_rank_pool;
    float* fitness_coherence_pool;
    CAParameterMap* param_map;
    int* lifecycle_phase_counts;
    float* reduction_workspace;
    float* gradient_features_pool;
    float* gradient_logits_pool;
    float* gradient_loss_pool;
    float* gradient_logit_grads_pool;
    float* gradient_magnitudes_pool;
    float* pooling_weights_grad;
    float* fc_weights_grad;
    float* fc_bias_grad;
    float* features_grad;
    float* adam_m_perception_pool;
    float* adam_v_perception_pool;
    float* adam_m_interaction_pool;
    float* adam_v_interaction_pool;
    float* adam_m_value_pool;
    float* adam_v_value_pool;
    float* adam_m_classifier_pool;
    float* adam_v_classifier_pool;
    float* adam_m_pooling;
    float* adam_v_pooling;
    float* adam_m_fc_weights;
    float* adam_v_fc_weights;
    float* adam_m_fc_bias;
    float* adam_v_fc_bias;
    float* batch_ca_states_pool;
    int* batch_labels_pool;
    float* task_loss_pool;
    float* reg_loss_pool;
    float* rank_loss_pool;
    float* coherence_loss_pool;
    float* diversity_loss_pool;
    float* total_loss_pool;
    HybridTrainingMode* training_mode;
    ClassificationHead* classifier;
    float* classifier_workspace;
    AdaptiveCurriculum* curriculum;
    float* voronoi_occupancy_histogram;
    float* pool_task_accuracies;
    Organism* organism;
    float* organism_workspace_genomes;
    float* behavioral_features_buffer;
    float* behavioral_embedding_weights;
    float* behavioral_reconstruction_error;
    // Temporary gradient/workspace buffers for hybrid_organism_lifecycle_kernel
    float* ca_output_grad_buffer;
    float* dL_dperception_buffer;
    float* dL_dinteraction_buffer;
    float* component_workspace_genomes_buffer;
    float* behavioral_workspace_genomes_buffer;
    int* pool_alive_indices;
    bool* pool_alive_flags;
    float* pool_fitness_values;
    int* pool_compaction_flags;
    int* pool_compaction_scan;
    int* pool_compaction_recursive_workspace;
    MemoryUpdateParams* memory_params;
    int* weight_inherit_child_indices;
    int* weight_inherit_parent_indices;
    int* weight_inherit_num_pending;
    // Chunked backward pass workspace buffers (preallocated, sized for BACKWARD_CHUNK_SAMPLES)
    half* backward_ws_fp16_a;
    half* backward_ws_fp16_b;
    float* backward_ws_dW;
    float* backward_ws_dI;
    half* backward_ws_W_T;
    float* backward_ws_im2col;
    float* backward_ws_dpregelu;
    curandState* rng_states;
    // Per-entry tape buffer pools (wired to ca_state->tape)
    TapeEntry* ad_tape_entries_pool;
    float* ad_tape_values_pool;
    float* ad_tape_grads_pool;
    int* ad_tape_levels_pool;
    // Per-entry saved activation pools (wired to ca_state->*_saved)
    float* perception_activations_saved;
    float* interaction_activations_saved;
    float* pre_gelu_values_saved;
    float* batched_ca_output;
    // Batched training buffers - sized for BATCH_SIZE_MAX concurrent samples
    float* batch_affinity_reduced;      // [BATCH_SIZE_MAX * CA_FIELD_SIZE]
    float* batch_flow_field;            // [BATCH_SIZE_MAX * CA_FIELD_SIZE * 2]
    float* batch_reintegration_buffer;  // [BATCH_SIZE_MAX * CA_FIELD_SIZE * CHANNELS_MAX]
    float* batch_prev_concentration;    // [BATCH_SIZE_MAX * CA_FIELD_SIZE * CHANNELS_MAX] - previous step's final state for recurrence
};

struct Dataset;
struct ClassificationHead;
struct AdaptiveCurriculum;

struct Organism {

    ComponentPool* pool;
    GPUElite* archive;
    int archive_size;
    VoronoiCell* voronoi_cells;
    int num_voronoi_cells;
    MultiHeadCAState* ca_state_pool;
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
    DIRESAWeights* diresa_hw_weights;
    DIRESAWeights* diresa_task_weights;
    DIRESAWeights* diresa_gen_weights;
    DIRESAWeights* diresa_genome_weights;
    float* diresa_hw_weight_pool;
    float* diresa_task_weight_pool;
    float* diresa_gen_weight_pool;
    float* diresa_genome_weight_pool;

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

    CAParameterMap* param_map;

    int current_activation_grid_size;  // Track current allocation size for dynamic reallocation

    HybridTrainingMode* training_mode;
    Dataset** dataset_array;
    Dataset* current_dataset;
    Dataset** test_dataset_array;
    Dataset* current_test_dataset;
    ClassificationHead* classifier;
    AdaptiveCurriculum* curriculum;
    float* voronoi_occupancy_histogram;
    float* pool_task_accuracies;

    void* lifecycle_states;

    TelemetryBuffer* telemetry;

    int* memory_compaction_valid_flags;
    int* memory_compaction_scan;
    int* memory_compaction_recursive_workspace;
    MemoryEntry* memory_compaction_buffer;

    int* pool_compaction_flags;
    int* pool_compaction_scan;
    int* pool_compaction_recursive_workspace;

    float* resource_density;
    float* resource_next;
    float* fitness_landscape;
    float* resource_gradient_x;
    float* resource_gradient_y;

    int* lifecycle_phase_counts;

    float* reduction_workspace;  // Partial sums for parallel reduction of chemical field mean
    int reduction_num_blocks;    // Number of blocks used in reduction
    int reduction_total_cells;   // Total cells in chemical field for mean computation

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

    OrganismPreallocatedBuffers* buffers;

    uint8_t* elite_compressed_pool;
    uint32_t* elite_size_pool;

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

    MemoryUpdateParams* memory_params;

    int* weight_inherit_child_indices;
    int* weight_inherit_parent_indices;
    int* weight_inherit_num_pending;

    curandState* rng_states;
};

#include "../training/hybrid_lifecycle.cu"
#include "../lifecycle/lifecycle_stages.cu"
__global__ void memory_update_kernel(TemporalTube* tubes, float* fitness_history, int* valid_flags_workspace, int* scan_workspace, int* scan_recursive_workspace, MemoryEntry* temp_buffer, int generation, float* genome, float* gradients, uint64_t genome_hash, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance);
__global__ void selection_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, VoronoiCell* voronoi_cells, int num_cells, int* archive_size, BehavioralState* behavioral_agents, int generation, float* workspace_genomes);
__global__ void archive_driven_lifecycle_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, int archive_size, VoronoiCell* voronoi_cells, int num_cells, BehavioralState* behavioral_agents, float hunger_threshold, int generation, float* workspace_genomes, DIRESAWeights* diresa_genome_weights);
__global__ void spawn_wave_kernel(Organism* organism, ComponentPool* pool, float spawn_probability, int generation, float* workspace_genomes);
__global__ void culling_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, int archive_size, ChemicalField* chemical_field, ArchitectureParams arch, float fitness_threshold, float hunger_threshold);
__global__ void compute_fitness_from_diresa_kernel(Organism* organism, ComponentPool* pool, int latent_dim);
__global__ void reduce_concentration_mean_kernel(ChemicalField* field, int total_cells, float* partial_sums);
__global__ void finalize_concentration_mean_kernel(ChemicalField* field, float* partial_sums, int num_blocks, int total_cells);
__global__ void coherence_kernel(float* prediction_errors, float* coherence_out, int history_length);
__global__ void initialize_ca_from_field_kernel(
    ComponentPool* pool,
    const float* chem_concentration, const float* chem_gradient_x, const float* chem_gradient_y,
    const float* chem_laplacian, const float* chem_sources, const float* chem_decay_factors,
    const float* rd_resource_density, const float* rd_fitness_landscape,
    const float* rd_resource_gradient_x, const float* rd_resource_gradient_y,
    const float* behavioral_field, const float* attractor_field,
    int max_grid_size, int entry_idx);
__global__ void update_field_from_ca_kernel(ComponentPool* pool, float* chemical_concentration, int max_grid_size, int entry_idx);
__global__ void prepare_ca_fp16_kernel(ComponentPool* pool, int max_grid_size, ArchitectureParams arch, int entry_idx);
__global__ void multi_head_ca_tensor_kernel(ComponentPool* pool, int max_grid_size, ArchitectureParams arch, int entry_idx);
__global__ void reduce_affinity_kernel(ComponentPool* pool, int max_grid_size, int entry_idx);
__global__ void compute_flow_field_kernel(ComponentPool* pool, int max_grid_size, int entry_idx);
__global__ void clear_reintegration_buffer_kernel(ComponentPool* pool, int max_buffer_size, int entry_idx);
__global__ void reintegration_redistribute_kernel(ComponentPool* pool, int max_grid_size, int entry_idx);
__global__ void copy_reintegration_to_concentration_kernel(ComponentPool* pool, int max_buffer_size, int entry_idx);
__global__ void compute_effective_rank_from_latent_kernel(ComponentPool* pool, float* effective_rank_history, float* workspace_genomes, int latent_dim, int entry_idx);
__global__ void store_navigation_history_kernel(Organism* organism, BehavioralState* agents, TemporalTube* tubes, int generation, int hw_dim, int task_dim, int gen_dim);
__global__ void neural_ca_update_kernel(Organism* organism, ChemicalField* chemical_field, float* effective_rank_history, float* fp32_workspace, int generation, ComponentPool* pool, float* fitness_history, float* coherence_history, float* workspace_genomes, TraceBuffer* trace_buffer, int max_grid_size);
__global__ void init_ca_weights_kernel(half* weights, int total_size, uint64_t genome_hash, int fan_in, int fan_out);
__global__ void init_pool_kernel(ComponentPool* pool, int capacity, uint16_t* delta_indices_buffer, float* delta_values_buffer, float* gradients_buffer);
__global__ void compact_pool_alive_indices_kernel(ComponentPool* pool, int* flags, int* scan_workspace, int* scan_recursive_workspace, int capacity);
__global__ void init_organism_ca_weights_kernel(ComponentPool* pool, ArchitectureParams arch);

// get_arch_from_pool moved to pool.cu for proper include ordering

extern "C" __global__ void organism_lifecycle_kernel(
    Organism* organism,
    int generation,
    float* workspace_genomes
) {
    // All 32 threads participate - warp cooperation required for genome reconstruction
    int entry_idx = blockIdx.x;
    ComponentPool* pool = organism->pool;

    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;
    if (entry_idx == 0 && threadIdx.x == 0) printf("V:374 gen=%d cap=%d alive=%d\n", generation, pool->capacity, entry->alive);


    cudaError_t err;

    float* primary_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];
    float* primary_parent_temp = &workspace_genomes[entry_idx * GENOME_SIZE * 2 + GENOME_SIZE];

    reconstruct_genome_from_archive(
        entry->parent_hash,
        (GPUElite*)organism->archive,
        organism->archive_size,
        entry->delta_indices,
        entry->delta_values,
        entry->num_deltas,
        entry->max_deltas,
        primary_genome,
        GENOME_SIZE,
        primary_parent_temp,
        organism->diresa_genome_weights
    );

    if (entry_idx == 0) printf("V:399 gen=%d\n", generation);

    if (entry_idx == 0 && (generation % 5 == 0 || generation < 3)) {
    }

    ArchitectureParams arch;
    arch.num_heads = entry->num_heads;
    arch.channels = entry->channels;
    arch.hidden_dim = entry->hidden_dim;
    arch.head_dim = entry->head_dim;
    arch.grid_size = entry->grid_size;
    // Gate center derived from coherence: low coherence → conservative, high → aggressive
    float coherence = fminf(fmaxf(entry->coherence, 0.0f), 1.0f);
    arch.ca_gate_center = 2.0f - 1.5f * coherence;

    dim3 component_grid((POOL_CAPACITY_MAX + (BLOCK_SIZE - 1)) / BLOCK_SIZE);
    dim3 component_block(BLOCK_SIZE);

    atomicAdd(&organism->lifecycle_phase_counts[0], 1);

    if (entry_idx == 0) {
        printf("V:417 trace=%p hw_geom=%p arch=%p vor=%p n_vor=%d\n",
               (void*)organism->trace_buffer, (void*)organism->hardware_geom,
               (void*)organism->archive, (void*)organism->voronoi_cells,
               organism->num_voronoi_cells);
        aggregate_hardware_geometry_kernel<<<1, BLOCK_SIZE>>>(organism->trace_buffer, organism->hardware_geom);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: aggregate_hardware_geometry_kernel failed err=%d gen=%d\n", (int)err, generation);
            return;
        }

        int selection_blocks = (pool->capacity + WARP_SIZE - 1) / WARP_SIZE;
        selection_kernel<<<selection_blocks, WARP_SIZE>>>(
            organism,
            pool,
            organism->archive,
            organism->voronoi_cells,
            organism->num_voronoi_cells,
            &organism->archive_size,
            organism->behavioral_agents,
            generation,
            workspace_genomes
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: selection_kernel failed err=%d gen=%d\n", (int)err, generation);
            return;
        }

        float* component_workspace_genomes = &workspace_genomes[0];
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: workspace_genomes access failed err=%d gen=%d\n", (int)err, generation);
            return;
        }
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
            printf("FATAL [organism_lifecycle]: component_evolution_kernel failed err=%d gen=%d\n", (int)err, generation);
            return;
        }

        compute_fitness_from_diresa_kernel<<<pool->capacity, BLOCK_SIZE>>>(
            organism, pool, GENOME_LATENT_DIM_MAX);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: compute_fitness_from_diresa_kernel failed err=%d gen=%d\n", (int)err, generation);
            return;
        }
        printf("V:471 gen=%d\n", generation);
    }

    uint64_t pool_genome_hash = entry->genome_hash;
    float ctx_metabolic = entry->fitness;
    float ctx_stress = entry->hunger;

    // Read precomputed mean from parallel reduction (computed once per generation)
    float ctx_morphogen = organism->chemical_field->cached_mean;

    int spawn_rate_slot = derive_param_slot(pool_genome_hash, "lifecycle_spawn_rate");
    int spawn_min_slot = derive_param_slot(pool_genome_hash, "lifecycle_spawn_min");
    float spawn_rate = genome_to_param(primary_genome, entry->gradients, spawn_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, SPAWN_RATE_MIN, SPAWN_RATE_MAX);
    float spawn_min = genome_to_param(primary_genome, entry->gradients, spawn_min_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, SPAWN_PROBABILITY_MIN_MIN, SPAWN_PROBABILITY_MIN_MAX);

    if (entry_idx == 0) {
        printf("V:post471 gen=%d\n", generation);

        if (generation % TELEMETRY_DETAILED == 0) {
            genome_complexity_probe_kernel<<<1, 1>>>(organism->pool, &organism->telemetry->genome_complexity);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [organism_lifecycle]: genome_complexity_probe_kernel failed err=%d gen=%d\n", (int)err, generation);
                return;
            }

            int active = Atomics::load_int(organism->pool->active_count);
            float spawn_prob = spawn_rate * expf(-active / (float)POOL_CAPACITY_MAX);
        }

        if (generation % TELEMETRY_COMPREHENSIVE == 0) {
            GPUElite* arch = (GPUElite*)organism->archive;
            archive_topology_probe_kernel<<<1, 1>>>(
                arch, organism->archive_size,
                organism->voronoi_cells, organism->num_voronoi_cells,
                &organism->telemetry->archive_topology,
                &organism->telemetry->last_checkpoint,
                organism->telemetry->last_occupancy,
                arch->hw_dim, arch->task_dim, arch->gen_dim
            );
            int current_spawned = Atomics::load_int(organism->pool->total_spawned);
            int current_culled = Atomics::load_int(organism->pool->total_culled);
            organism->telemetry->archive_topology.births_since_checkpoint = current_spawned - organism->telemetry->last_total_spawned;
            organism->telemetry->archive_topology.deaths_since_checkpoint = current_culled - organism->telemetry->last_total_culled;
            organism->telemetry->last_total_spawned = current_spawned;
            organism->telemetry->last_total_culled = current_culled;
            organism->telemetry->last_checkpoint = organism->telemetry->archive_topology;
            diresa_evolution_probe_kernel<<<1, 1>>>(arch, organism->archive_size, &organism->telemetry->diresa_evolution);

            // Mark telemetry as valid so populate_audit_buffer copies the data
            organism->telemetry->valid = true;
        }

        float spawn_prob = spawn_rate * expf(-Atomics::load_int(organism->pool->active_count) / (float)organism->pool->capacity);
        printf("V:spawn_check gen=%d prob=%.4f min=%.4f\n", generation, spawn_prob, spawn_min);
        if (spawn_prob > spawn_min) {
            atomicAdd(&organism->lifecycle_phase_counts[1], 1);

            float* spawn_wave_workspace_genomes = &workspace_genomes[4 * GENOME_SIZE + 2 * organism->pool->capacity * GENOME_SIZE];

            spawn_wave_kernel<<<1, BLOCK_SIZE>>>(
                organism,
                organism->pool,
                spawn_prob,
                generation,
                spawn_wave_workspace_genomes
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [organism_lifecycle]: spawn_wave_kernel failed err=%d gen=%d\n", (int)err, generation);
                return;
            }
            printf("V:spawn_done gen=%d\n", generation);
        }

        int hunger_threshold_slot = derive_param_slot(pool_genome_hash, "lifecycle_hunger_threshold");
        float hunger_threshold = genome_to_param(primary_genome, entry->gradients, hunger_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, HUNGER_THRESHOLD_MIN, HUNGER_THRESHOLD_MAX);

        printf("V:pre_archive gen=%d\n", generation);
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
            printf("FATAL [organism_lifecycle]: archive_driven_lifecycle_kernel failed err=%d gen=%d\n", (int)err, generation);
            return;
        }
        printf("V:post_archive gen=%d\n", generation);

        // === WEIGHT INHERITANCE/RESTORATION ===
        // Phase 1: Inherit weights from pool parents (for entries spawned via spawn_component_device)
        cudaMemsetAsync(organism->weight_inherit_num_pending, 0, sizeof(int));

        find_pending_weight_inherits_kernel<<<(pool->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            pool, organism->weight_inherit_child_indices, organism->weight_inherit_parent_indices, organism->weight_inherit_num_pending);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: find_pending_weight_inherits_kernel failed err=%d gen=%d\n", (int)err, generation);
            return;
        }

        int h_num_pending = *organism->weight_inherit_num_pending;
        if (h_num_pending > 0) {
            int weight_blocks = (arch.num_heads * arch.channels * arch.head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
            inherit_ca_weights_kernel<<<dim3(weight_blocks, h_num_pending), BLOCK_SIZE>>>(
                pool, h_num_pending, organism->weight_inherit_child_indices, organism->weight_inherit_parent_indices);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("FATAL [organism_lifecycle]: inherit_ca_weights_kernel failed err=%d gen=%d pending=%d\n", (int)err, generation, h_num_pending);
                return;
            }
        }

        // Phase 2: Restore weights from archive (for entries respawned via replace_from_archive_device)
        for (int idx = 0; idx < pool->capacity; idx++) {
            PoolEntry* e = &pool->entries[idx];
            if (e->alive && e->ca_state && e->ca_state->tape.needs_weight_restore) {
                int elite_idx = e->ca_state->tape.restore_elite_idx;
                if (elite_idx >= 0 && elite_idx < organism->archive_size) {
                    int total_weights = e->num_heads * e->channels * e->head_dim +
                                       e->num_heads * e->head_dim * e->head_dim +
                                       e->num_heads * e->head_dim * e->channels;
                    int weight_blocks = (total_weights + BLOCK_SIZE - 1) / BLOCK_SIZE;

                    restore_elite_weights_kernel<<<weight_blocks, BLOCK_SIZE>>>(
                        organism->archive, elite_idx,
                        e->ca_state->perception_weights,
                        e->ca_state->interaction_weights,
                        e->ca_state->value_weights);

                    apply_weight_deltas_kernel<<<weight_blocks, BLOCK_SIZE>>>(
                        organism->archive, elite_idx,
                        e->ca_state->perception_weights,
                        e->ca_state->interaction_weights,
                        e->ca_state->value_weights);
                }
                e->ca_state->tape.needs_weight_restore = 0;
                e->ca_state->tape.restore_elite_idx = INT_MAX;
            }
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return;
        // === END WEIGHT INHERITANCE/RESTORATION ===

        atomicAdd(&organism->lifecycle_phase_counts[3], 1);

        cudaError_t pre_kern5_err = cudaGetLastError();
        if (pre_kern5_err != cudaSuccess) {
            return;
        }

        printf("V:pre_memory gen=%d\n", generation);
        memory_update_kernel<<<1, BLOCK_SIZE>>>(
            organism->chemical_field->history,
            organism->fitness_history,
            organism->memory_compaction_valid_flags,
            organism->memory_compaction_scan,
            organism->memory_compaction_recursive_workspace,
            organism->memory_compaction_buffer,
            generation,
            primary_genome,
            entry->gradients,
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
            return;
        }
        printf("V:post_memory gen=%d\n", generation);

        float* workspace_genome_buffer = &workspace_genomes[GENOME_SIZE * 2];

        printf("V:pre_neural_ca gen=%d\n", generation);
        neural_ca_update_kernel<<<pool->capacity, WARP_SIZE>>>(
            organism,
            organism->chemical_field,
            organism->effective_rank_history,
            organism->buffers->fp32_ca_workspace,
            generation,
            organism->pool,
            organism->fitness_history,
            organism->coherence_history,
            workspace_genome_buffer,
            organism->trace_buffer,
            arch.grid_size
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return;
        }
        printf("V:post_neural_ca gen=%d\n", generation);

        if (generation % CHECKPOINT_INTERVAL == 0 && generation > 0) {
            int active = Atomics::load_int(organism->pool->active_count);
            float effective_rank = organism->effective_rank_history[generation % 2];
        }

        printf("V:pre_behavioral gen=%d\n", generation);
        // Launch per-entry parallelism: one block per entry, warp cooperation within block
        behavioral_update_kernel<<<pool->capacity, WARP_SIZE>>>(
            organism,
            organism->behavioral_agents,
            organism->chemical_field,
            organism->chemical_field->history,
            generation,
            arch,
            workspace_genomes
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("V:behavioral_launch_error gen=%d err=%d\n", generation, (int)err);
            return;
        }
        printf("V:post_behavioral_launch gen=%d\n", generation);

        // Force synchronization to surface any execution errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("V:behavioral_sync_error gen=%d err=%d\n", generation, (int)err);
            return;
        }
        printf("V:post_behavioral_sync gen=%d\n", generation);

        dim3 field_grid((arch.grid_size + 15) / 16, (arch.grid_size + 15) / 16);
        dim3 field_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

        cudaMemsetAsync(organism->fitness_landscape, 0, arch.grid_size * arch.grid_size * sizeof(float));
        int entry_blocks = (organism->pool->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
        update_fitness_landscape_kernel<<<entry_blocks, BLOCK_SIZE>>>(
            organism->pool,
            organism->behavioral_agents,
            organism->fitness_landscape,
            arch.grid_size
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return;
        }

        uint64_t genome_hash = entry->genome_hash;
        float* genome = primary_genome;
        float* gradients = entry->gradients;

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
            organism->resource_gradient_x,
            organism->resource_gradient_y,
            arch.grid_size,
            rf_dt,
            rf_diffusivity,
            rf_advection
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return;
        }

        float* tmp = organism->resource_density;
        organism->resource_density = organism->resource_next;
        organism->resource_next = tmp;

        int num_blocks = (organism->pool->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;

        float* lifecycle_workspace = &workspace_genomes[4 * GENOME_SIZE];

        lifecycle_transition_kernel<<<num_blocks, BLOCK_SIZE>>>(
            organism->pool,
            (GPUElite*)organism->archive,
            organism->archive_size,
            organism->voronoi_cells,
            organism->num_voronoi_cells,
            organism->behavioral_agents,
            (LocalOrganismState<BLOCK_SIZE>*)organism->lifecycle_states,
            generation,
            organism->chemical_field,
            arch.grid_size,
            organism->telemetry,
            lifecycle_workspace,
            organism->diresa_genome_weights
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return;
        }

        compact_pool_alive_indices_kernel<<<1, 1>>>(
            organism->pool,
            organism->pool_compaction_flags,
            organism->pool_compaction_scan,
            organism->pool_compaction_recursive_workspace,
            organism->pool->capacity
        );
        cudaDeviceSynchronize();

        int alive_blocks = (organism->pool->alive_indices_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (alive_blocks == 0) alive_blocks = 1;

        hierarchical_lifecycle_kernel<<<alive_blocks, BLOCK_SIZE>>>(
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
            lifecycle_workspace,
            organism->diresa_genome_weights
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
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
    if (tid == 0) {
    }
    // Use SoA for coalesced alive read
    if (tid >= pool->capacity || !pool->alive_flags[tid]) {
        return;
    }

    if (tid == 0) {
    }

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

    if (tid == 0) {
    }

    cudaError_t err;

    float current_task_accuracy = organism->telemetry->task_performance.accuracy;
    if (!isnan(current_task_accuracy)) {
        pool->entries[tid].task_accuracy = current_task_accuracy;
    }
    pool->entries[tid].generalization_gap = fabsf(organism->telemetry->task_performance.train_accuracy - organism->telemetry->task_performance.test_accuracy);

    if (tid == 0) {
    }

    organism->fitness_history[(generation % 2) * POOL_CAPACITY_MAX + tid] = pool->entries[tid].task_accuracy;
    organism->coherence_history[(generation % 2) * POOL_CAPACITY_MAX + tid] = pool->entries[tid].coherence;

    if (tid == 0) {
    }

    {
        float hardware_features_temp[HARDWARE_FEATURES_DIM];
        extract_hardware_features(organism->hardware_geom, hardware_features_temp);

        // Cache SoA fitness read for reuse in loop
        float tid_fitness = pool->fitness_values[tid];
        float tid_hunger = pool->entries[tid].hunger;

        float hw_efficiency_sum = safe_epsilon(1.0f);
        for (int i = 0; i < HARDWARE_FEATURES_DIM; i++) {
            int hw_weight_slot = derive_param_slot(pool->entries[tid].genome_hash, "hw_efficiency_weight");
            float hw_weight = genome_to_param(
                primary_genome,
                pool->entries[tid].gradients,
                hw_weight_slot,
                tid_fitness,
                tid_hunger,
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
    }

    {
        __shared__ float s_acc[BLOCK_SIZE], s_gap[BLOCK_SIZE], s_hw[BLOCK_SIZE], s_fit[BLOCK_SIZE];
        int local_tid = threadIdx.x;
        // Use SoA for coalesced alive/fitness reads
        bool alive = (tid < pool->capacity) && pool->alive_flags[tid];
        s_acc[local_tid] = alive ? pool->entries[tid].task_accuracy : 0.0f;
        s_gap[local_tid] = alive ? pool->entries[tid].generalization_gap : 0.0f;
        s_hw[local_tid] = alive ? pool->entries[tid].hardware_efficiency : 0.0f;
        s_fit[local_tid] = alive ? pool->fitness_values[tid] : 0.0f;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (local_tid < s) {
                s_acc[local_tid] += s_acc[local_tid + s];
                s_gap[local_tid] += s_gap[local_tid + s];
                s_hw[local_tid] += s_hw[local_tid + s];
                s_fit[local_tid] += s_fit[local_tid + s];
            }
            __syncthreads();
        }
    }

    // Baldwin Effect: Reinforce genomes that show learning improvement
    // Use SoA for coalesced alive read
    if (generation > 0 && tid < pool->capacity && pool->alive_flags[tid]) {
        float prev_task_accuracy = organism->fitness_history[((generation - 1) % 2) * POOL_CAPACITY_MAX + tid];
        float learning_success = current_task_accuracy - prev_task_accuracy;

        if (is_meaningful(learning_success, 1.0f)) {
            float baldwin_sensitivity = pool->entries[tid].baldwin_sensitivity;
            float scale = learning_success * baldwin_sensitivity;
            float* grads = pool->entries[tid].gradients;
            for (int g = threadIdx.x; g < GENOME_SIZE; g += blockDim.x) {
                float val = grads[g] + scale * primary_genome[g];
                grads[g] = fmaxf(GENOME_VALUE_MIN, fminf(GENOME_VALUE_MAX, val));
            }
        }
    }
}

__global__ void compute_fitness_from_diresa_kernel(
    Organism* organism,
    ComponentPool* pool,
    int latent_dim
) {
    int entry_idx = blockIdx.x;
    // Use SoA for coalesced alive read
    if (entry_idx >= pool->capacity || !pool->alive_flags[entry_idx]) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    float* latent_genome = organism->latent_genome_pool + entry_idx * GENOME_LATENT_DIM_MAX;

    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < latent_dim; i += blockDim.x) {
        sum += latent_genome[i];
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float warp_sums[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    int num_warps = blockDim.x / warpSize;
    if (tid < num_warps) {
        sum = warp_sums[tid];
        unsigned active = __activemask();
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(active, sum, offset);
        }
    }
    __shared__ float mean;
    if (tid == 0) mean = sum / latent_dim;
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tid; i < latent_dim; i += blockDim.x) {
        float diff = latent_genome[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    if (lane == 0) warp_sums[warp_id] = var_sum;
    __syncthreads();
    if (tid < num_warps) {
        var_sum = warp_sums[tid];
        unsigned active = __activemask();
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            var_sum += __shfl_down_sync(active, var_sum, offset);
        }
    }

    if (tid == 0) {
        float variance = var_sum / latent_dim;
        if (variance < 0.0f) {
            // FATAL: Negative variance is mathematically impossible - numerical corruption
            printf("FATAL [compute_fitness]: entry=%d negative variance=%f, culling\n", entry_idx, variance);
            entry->alive = false;
            pool->alive_flags[entry_idx] = false;
            return;
        }
        float effective_rank = sqrtf(variance) * latent_dim;
        float gen_gap_term = 1.0f - entry->generalization_gap;

        // Check for NaN/invalid fitness components - preserve existing fitness if components not yet valid
        // This prevents gen 0 organisms from having their init_fitness_prior overwritten with NaN
        bool has_nan = isnan(entry->task_accuracy) || isnan(entry->generalization_gap) || isnan(entry->hardware_efficiency);
        if (has_nan) {
            // Fitness components not yet computed (gen 0) - keep existing fitness prior
            return;
        }

        if (gen_gap_term <= 0.0f || entry->task_accuracy <= 0.0f || effective_rank <= 0.0f || entry->hardware_efficiency <= 0.0f) {
            // Invalid fitness inputs - cull rather than silently set to zero
            printf("WARN [compute_fitness]: entry=%d invalid inputs (gen_gap=%.3f task_acc=%.3f rank=%.3f hw_eff=%.3f), culling\n",
                   entry_idx, gen_gap_term, entry->task_accuracy, effective_rank, entry->hardware_efficiency);
            entry->alive = false;
            pool->alive_flags[entry_idx] = false;
            return;
        }
        entry->fitness = powf(entry->task_accuracy, entry->fitness_task_exponent)
                       * powf(gen_gap_term, entry->fitness_gen_exponent)
                       * powf(effective_rank, entry->fitness_rank_exponent)
                       * powf(entry->hardware_efficiency, entry->fitness_efficiency_exponent);
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
    int entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    float* organism_genome = &workspace_genomes[entry_idx * 2 * GENOME_SIZE];
    float* temp_parent = &workspace_genomes[entry_idx * 2 * GENOME_SIZE + GENOME_SIZE];

    reconstruct_genome_from_archive(
        entry->parent_hash,
        archive,
        *archive_size,
        entry->delta_indices,
        entry->delta_values,
        entry->num_deltas,
        entry->max_deltas,
        organism_genome,
        GENOME_SIZE,
        temp_parent,
        organism->diresa_genome_weights
    );

    float* latent_genome = organism->latent_genome_pool + entry_idx * GENOME_LATENT_DIM_MAX;
    diresa_encode(organism_genome, latent_genome, &organism->diresa_genome_weights[0]);

    int hw_dim = archive->hw_dim;
    int task_dim = archive->task_dim;
    int gen_dim = archive->gen_dim;

    float hardware_features_temp[HARDWARE_FEATURES_DIM];
    extract_hardware_features(organism->hardware_geom, hardware_features_temp);

    float* hw_coords_component = &organism->hw_coords_pool[entry_idx * hw_dim];
    diresa_encode(hardware_features_temp, hw_coords_component, organism->diresa_hw_weights);

    float task_features[1] = {entry->task_accuracy};
    float* task_coords_component = &organism->task_coords_pool[entry_idx * task_dim];
    diresa_encode(task_features, task_coords_component, &organism->diresa_task_weights[0]);

    float gen_features[1] = {entry->generalization_gap};
    float* gen_coords_component = &organism->gen_coords_pool[entry_idx * gen_dim];
    diresa_encode(gen_features, gen_coords_component, &organism->diresa_gen_weights[0]);

    float* entry_genome = organism_genome;

    int parent_idx = find_parent_by_hash(archive, *archive_size, entry->genome_hash);
    uint32_t parent_id_0 = (parent_idx >= 0) ? parent_idx : 0;
    uint32_t parent_id_1 = 0;

    if (entry->coherence <= 0.0f) {
        return;
    }

    insert_elite_device(
        archive,
        archive_size,
        entry->fitness,
        entry->coherence,
        entry->fitness / entry->coherence,
        gpu_sha256(entry_genome, GENOME_SIZE),
        parent_id_0,
        parent_id_1,
        generation,
        &organism->hw_coords_pool[entry_idx * hw_dim],
        &organism->task_coords_pool[entry_idx * task_dim],
        &organism->gen_coords_pool[entry_idx * gen_dim],
        entry->task_accuracy,
        &archive->per_class_accuracy[entry_idx * NUM_CLASSES_MAX],
        NUM_CLASSES_MAX,
        voronoi_cells,
        num_cells
    );
}

__global__ void spawn_wave_kernel(
    Organism* organism,
    ComponentPool* pool,
    float spawn_probability,
    int generation,
    float* workspace_genomes
) {
    __shared__ int qualifying_parents[BLOCK_SIZE];
    __shared__ int qualifying_count;

    int tid = threadIdx.x;
    int capacity = pool->capacity;

    if (tid == 0) qualifying_count = 0;
    __syncthreads();

    for (int i = tid; i < capacity; i += blockDim.x) {
        // Use SoA for coalesced alive read
        if (!pool->alive_flags[i]) continue;

        float* temp_genome = &workspace_genomes[tid * GENOME_SIZE * SPAWN_WS_COUNT + GENOME_SIZE * SPAWN_WS_TEMP_GENOME];
        float* temp_parent = &workspace_genomes[tid * GENOME_SIZE * SPAWN_WS_COUNT + GENOME_SIZE * SPAWN_WS_TEMP_PARENT];

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
        float local_morphogen = sample_neighborhood(
            organism->chemical_field->concentration, i, pool->entries[i].grid_size);

        // Cache SoA fitness read for reuse
        float entry_fitness = pool->fitness_values[i];
        float entry_hunger = pool->entries[i].hunger;

        int ctx_metabolic_slot = derive_param_slot(temp_hash, "spawn_ctx_metabolic");
        float ctx_metabolic = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_metabolic_slot,
            entry_fitness, entry_hunger, local_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            NORMALIZED_MIN, NORMALIZED_MAX);

        int ctx_stress_slot = derive_param_slot(temp_hash, "spawn_ctx_stress");
        float ctx_stress = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_stress_slot,
            entry_fitness, entry_hunger, local_morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            NORMALIZED_MIN, NORMALIZED_MAX);

        int ctx_morphogen_slot = derive_param_slot(temp_hash, "spawn_ctx_morphogen");
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

        // Use SoA for coalesced fitness read
        if (entry_fitness > fitness_threshold) {
            int slot = atomicAdd(&qualifying_count, 1);
            if (slot < BLOCK_SIZE) {
                qualifying_parents[slot] = i;
            }
        }
    }
    __syncthreads();

    if (qualifying_count == 0) return;

    unsigned int seed = tid * generation * RNG_SEED_MULTIPLIER;
    seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
    float rand = (seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE;

    if (rand >= spawn_probability) return;

    int parent_slot = tid % qualifying_count;
    int parent_idx = qualifying_parents[parent_slot];

    float* workspace_parent_genome = &workspace_genomes[tid * GENOME_SIZE * SPAWN_WS_COUNT + GENOME_SIZE * SPAWN_WS_PARENT_GENOME];
    float* workspace_child_genome = &workspace_genomes[tid * GENOME_SIZE * SPAWN_WS_COUNT + GENOME_SIZE * SPAWN_WS_CHILD_GENOME];
    float* workspace_parent_parent_temp = &workspace_genomes[tid * GENOME_SIZE * SPAWN_WS_COUNT + GENOME_SIZE * SPAWN_WS_PARENT_PARENT_TEMP];

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
    float parent_morphogen = sample_neighborhood(
        organism->chemical_field->concentration, parent_idx, pool->entries[parent_idx].grid_size);

    // Cache SoA fitness read for reuse
    float parent_fitness = pool->fitness_values[parent_idx];
    float parent_hunger = pool->entries[parent_idx].hunger;

    int ctx_metabolic_slot = derive_param_slot(parent_hash, "mutation_ctx_metabolic");
    float ctx_metabolic = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_metabolic_slot,
        parent_fitness, parent_hunger, parent_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        NORMALIZED_MIN, NORMALIZED_MAX);

    int ctx_stress_slot = derive_param_slot(parent_hash, "mutation_ctx_stress");
    float ctx_stress = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_stress_slot,
        parent_fitness, parent_hunger, parent_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        NORMALIZED_MIN, NORMALIZED_MAX);

    int ctx_morphogen_slot = derive_param_slot(parent_hash, "mutation_ctx_morphogen");
    float ctx_morphogen = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_morphogen_slot,
        parent_fitness, parent_hunger, parent_morphogen,
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

    spawn_component_device(
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

        float ctx_morphogen = chemical_field->cached_mean;

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

            float* thread_workspace = &workspace_genomes[idx * (2 * GENOME_SIZE + POOL_CAPACITY_MAX)];
            replace_from_archive_device(
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
                thread_workspace,
                diresa_genome_weights
            );

            Atomics::increment_int(pool->active_count);
        }
    }
}

__global__ void populate_organism_flow_params_kernel(
    Organism* organism,
    ChemicalField* chemical_field,
    ComponentPool* pool,
    float* workspace_genomes,
    int max_grid_size
) {
    int entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    ArchitectureParams arch = get_arch_from_pool(pool, entry_idx);

    float* entry_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];
    float* entry_parent_temp = &workspace_genomes[entry_idx * GENOME_SIZE * 2 + GENOME_SIZE];
    reconstruct_genome_from_archive(
        entry->parent_hash,
        (GPUElite*)organism->archive,
        organism->archive_size,
        entry->delta_indices,
        entry->delta_values,
        entry->num_deltas,
        entry->max_deltas,
        entry_genome,
        GENOME_SIZE,
        entry_parent_temp,
        organism->diresa_genome_weights
    );

    float* genome = entry_genome;
    float* epigenetic = entry->gradients;
    uint64_t genome_hash = entry->genome_hash;

    float context_metabolic = entry->fitness;
    int stress_numerator_slot = derive_param_slot(genome_hash, "context_stress_numerator");
    float stress_numerator = genome_to_param(
        genome, epigenetic, stress_numerator_slot,
        entry->fitness,
        entry->hunger,
        safe_epsilon(1.0f),
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        NORMALIZED_MIN, NORMALIZED_MAX
    );
    if (entry->hunger <= 0.0f) return;
    float context_stress = stress_numerator / entry->hunger;

    float context_morphogen = chemical_field->cached_mean;

    int s_param_slot = derive_param_slot(genome_hash, "flow_lenia_s");
    entry->flow_s = genome_to_param(
        genome, epigenetic, s_param_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_S_MIN, FLOW_LENIA_S_MAX
    );

    int beta_A_slot = derive_param_slot(genome_hash, "flow_lenia_beta_a");
    entry->flow_beta_A = genome_to_param(
        genome, epigenetic, beta_A_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_BETA_A_MIN, FLOW_LENIA_BETA_A_MAX
    );

    int n_param_slot = derive_param_slot(genome_hash, "flow_lenia_n");
    entry->flow_n = genome_to_param(
        genome, epigenetic, n_param_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_N_MIN, FLOW_LENIA_N_MAX
    );

    int alpha_min_slot = derive_param_slot(genome_hash, "flow_lenia_alpha_min");
    entry->flow_alpha_min = genome_to_param(
        genome, epigenetic, alpha_min_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_ALPHA_MIN_MIN, FLOW_LENIA_ALPHA_MIN_MAX
    );

    int alpha_max_slot = derive_param_slot(genome_hash, "flow_lenia_alpha_max");
    entry->flow_alpha_max = genome_to_param(
        genome, epigenetic, alpha_max_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_ALPHA_MAX_MIN, FLOW_LENIA_ALPHA_MAX_MAX
    );

    int sharpness_slot = derive_param_slot(genome_hash, "flow_lenia_sharpness");
    entry->flow_sharpness = genome_to_param(
        genome, epigenetic, sharpness_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_SHARPNESS_MIN, FLOW_LENIA_SHARPNESS_MAX
    );

    int resource_flow_dt_slot = derive_param_slot(genome_hash, "flow_lenia_resource_dt");
    entry->flow_resource_dt = genome_to_param(
        genome, epigenetic, resource_flow_dt_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        RESOURCE_FLOW_DT_MIN, RESOURCE_FLOW_DT_MAX
    );
}

// Pool-based Flow Lenia kernels - extract data from pool entry and use same FlowLeniaOps logic
// These are called via CDP from neural_ca_update_kernel

__global__ void reduce_affinity_kernel(ComponentPool* pool, int max_grid_size, int entry_idx) {
    if (entry_idx >= pool->capacity) return;
    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;
    DEVICE_FATAL_IF(entry->ca_state == nullptr, "alive entry has null ca_state");

    int cell_idx = blockIdx.x;
    int grid_size = entry->grid_size;
    int total_cells = grid_size * grid_size;
    if (cell_idx >= total_cells) return;

    int lane = threadIdx.x;
    float affinity = 0.0f;

    int num_heads = entry->num_heads;
    int head_dim = entry->head_dim;
    int total_elements = num_heads * head_dim;
    float* ca_output = entry->ca_state->ca_output;

    for (int i = lane; i < total_elements; i += WARP_SIZE) {
        int head = i / head_dim;
        int dim = i % head_dim;
        int idx = head * total_cells * head_dim + cell_idx * head_dim + dim;
        affinity += ldg_float(&ca_output[idx]);
    }

    affinity = WarpReduce<32>::sum(affinity);

    if (lane == 0) {
        entry->ca_state->affinity_reduced[cell_idx] = affinity;
    }
}

__global__ void compute_flow_field_kernel(ComponentPool* pool, int max_grid_size, int entry_idx) {
    if (entry_idx >= pool->capacity) return;
    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;
    DEVICE_FATAL_IF(entry->ca_state == nullptr, "alive entry has null ca_state");

    int grid_size = entry->grid_size;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= grid_size || y >= grid_size) return;

    int cell_idx = y * grid_size + x;
    int channels = entry->channels;
    float* affinity = entry->ca_state->affinity_reduced;
    float* concentration = entry->ca_state->ca_concentration;
    float* flow_field = entry->ca_state->flow_field;

    float U_center = affinity[cell_idx];
    int x_E = min(x + 1, grid_size - 1);
    int y_N = min(y + 1, grid_size - 1);
    float U_E = affinity[y * grid_size + x_E];
    float U_N = affinity[y_N * grid_size + x];

    float A_sum_center = 0.0f, A_sum_E = 0.0f, A_sum_N = 0.0f;
    for (int c = 0; c < channels; c++) {
        A_sum_center += concentration[cell_idx * channels + c];
        A_sum_E += concentration[(y * grid_size + x_E) * channels + c];
        A_sum_N += concentration[(y_N * grid_size + x) * channels + c];
    }

    float2 F = FlowLeniaOps::compute_flow_differentiable(
        U_center, U_E, U_N, A_sum_center, A_sum_E, A_sum_N,
        entry->flow_beta_A, entry->flow_n,
        entry->flow_alpha_min, entry->flow_alpha_max, entry->flow_sharpness
    );

    flow_field[cell_idx * 2 + 0] = F.x;
    flow_field[cell_idx * 2 + 1] = F.y;
}

__global__ void clear_reintegration_buffer_kernel(ComponentPool* pool, int max_buffer_size, int entry_idx) {
    if (entry_idx >= pool->capacity) return;
    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;
    DEVICE_FATAL_IF(entry->ca_state == nullptr, "alive entry has null ca_state");

    int buffer_size = entry->grid_size * entry->grid_size * entry->channels;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < buffer_size) {
        entry->ca_state->reintegration_buffer[idx] = 0.0f;
    }
}

__global__ void reintegration_redistribute_kernel(ComponentPool* pool, int max_grid_size, int entry_idx) {
    if (entry_idx >= pool->capacity) return;
    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;
    DEVICE_FATAL_IF(entry->ca_state == nullptr, "alive entry has null ca_state");

    int grid_size = entry->grid_size;
    int source_x = blockIdx.x;
    int source_y = blockIdx.y;
    if (source_x >= grid_size || source_y >= grid_size) return;

    int source_idx = source_y * grid_size + source_x;
    int channels = entry->channels;
    float* flow_field = entry->ca_state->flow_field;
    float* concentration = entry->ca_state->ca_concentration;
    float* buffer = entry->ca_state->reintegration_buffer;

    float Fx = flow_field[source_idx * 2 + 0];
    float Fy = flow_field[source_idx * 2 + 1];
    float dt = entry->flow_resource_dt;

    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        float source_mass = concentration[source_idx * channels + c];
        FlowLeniaOps::bilinear_transport_forward(
            source_mass, (float)source_x, (float)source_y,
            Fx, Fy, dt, grid_size, buffer, c, channels
        );
    }
}

__global__ void copy_reintegration_to_concentration_kernel(ComponentPool* pool, int max_buffer_size, int entry_idx) {
    if (entry_idx >= pool->capacity) return;
    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;
    DEVICE_FATAL_IF(entry->ca_state == nullptr, "alive entry has null ca_state");

    int buffer_size = entry->grid_size * entry->grid_size * entry->channels;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < buffer_size) {
        entry->ca_state->ca_concentration[idx] = entry->ca_state->reintegration_buffer[idx];
    }
}

__global__ void neural_ca_update_kernel(
    Organism* organism,
    ChemicalField* chemical_field,
    float* effective_rank_history,
    float* fp32_workspace,
    int generation,
    ComponentPool* pool,
    float* fitness_history,
    float* coherence_history,
    float* workspace_genomes,
    TraceBuffer* trace_buffer,
    int max_grid_size
) {
    int entry_idx = blockIdx.x;
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    ArchitectureParams arch = get_arch_from_pool(pool, entry_idx);

    dim3 init_grid((max_grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM,
                   (max_grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM,
                   1);
    dim3 init_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

    int max_cells = max_grid_size * max_grid_size;
    if (arch.channels <= 0 || arch.num_heads <= 0 || arch.head_dim <= 0) {
        return;
    }
    dim3 prep_grid((max_cells * arch.channels + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 prep_block(BLOCK_SIZE);
    if (prep_grid.x == 0) {
        return;
    }

    // Convert CA concentration to FP16 workspace
    {
        int grid_size = entry->grid_size;
        int num_cells = grid_size * grid_size;
        int total = num_cells * arch.channels;
        MultiHeadCAState* ca_state = entry->ca_state;
        for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
            ca_state->fp16_workspace[idx] = __float2half(ca_state->ca_concentration[idx]);
        }
        __syncthreads();
    }

    if (arch.num_heads == 0) {
        if (threadIdx.x == 0) printf("V:neural_ca_no_heads entry=%d\n", entry_idx);
        return;
    }

    // Multi-head CA tensor operations (perception/interaction/value)
    {
        int grid_size = entry->grid_size;
        int num_cells = grid_size * grid_size;
        MultiHeadCAState* ca_state = entry->ca_state;
        half* fp16_workspace = ca_state->fp16_workspace;
        float* fp32_workspace = ca_state->fp32_workspace;
        half* perception_weights = ca_state->perception_weights;
        half* interaction_weights = ca_state->interaction_weights;
        half* value_weights = ca_state->value_weights;
        float* ca_output_fp32 = ca_state->ca_output;

        for (int head = 0; head < arch.num_heads; head++) {
            // Perception: WMMA tiles (warp processes tiles sequentially)
            int perception_tiles_m = (num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int perception_tiles_n = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int weight_offset = head * arch.channels * arch.head_dim;

            for (int tile_m = 0; tile_m < perception_tiles_m; tile_m++) {
                for (int tile_n = 0; tile_n < perception_tiles_n; tile_n++) {
                    int tile_row = tile_m * WMMA_TILE_DIM;
                    int tile_col = tile_n * WMMA_TILE_DIM;
                    if (tile_row >= num_cells || tile_col >= arch.head_dim) continue;

                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);

                    for (int k = 0; k < arch.channels; k += WMMA_TILE_DIM) {
                        if (k + WMMA_TILE_DIM <= arch.channels) {
                            wmma::load_matrix_sync(a_frag, fp16_workspace + tile_row * arch.channels + k, arch.channels);
                            wmma::load_matrix_sync(b_frag, perception_weights + weight_offset + k * arch.head_dim + tile_col, arch.head_dim);
                            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                    }
                    int output_offset = head * num_cells * arch.head_dim;
                    wmma::store_matrix_sync(fp32_workspace + output_offset + tile_row * arch.head_dim + tile_col, c_frag, arch.head_dim, wmma::mem_row_major);
                }
            }
            __syncwarp();

            // ReLU activation
            int head_size = num_cells * arch.head_dim;
            float* head_data = fp32_workspace + head * num_cells * arch.head_dim;
            for (int idx = threadIdx.x; idx < head_size; idx += blockDim.x) {
                head_data[idx] = fmaxf(0.0f, head_data[idx]);
            }
            __syncwarp();

            // Convert perception output to FP16 for interaction
            half* interaction_input = fp16_workspace + num_cells * arch.channels;
            for (int idx = threadIdx.x; idx < head_size; idx += blockDim.x) {
                interaction_input[idx] = __float2half(head_data[idx]);
            }
            __syncwarp();

            // Interaction: WMMA matmul
            float* interaction_output = fp32_workspace + arch.num_heads * num_cells * arch.head_dim;
            int interaction_tiles_m = (num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int interaction_tiles_n = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;

            for (int tile_m = 0; tile_m < interaction_tiles_m; tile_m++) {
                for (int tile_n = 0; tile_n < interaction_tiles_n; tile_n++) {
                    int tile_row = tile_m * WMMA_TILE_DIM;
                    int tile_col = tile_n * WMMA_TILE_DIM;
                    if (tile_row >= num_cells || tile_col >= arch.head_dim) continue;

                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);

                    for (int k = 0; k < arch.head_dim; k += WMMA_TILE_DIM) {
                        if (k + WMMA_TILE_DIM <= arch.head_dim) {
                            wmma::load_matrix_sync(a_frag, interaction_input + tile_row * arch.head_dim + k, arch.head_dim);
                            wmma::load_matrix_sync(b_frag, interaction_weights + head * arch.head_dim * arch.head_dim + k * arch.head_dim + tile_col, arch.head_dim);
                            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                    }
                    wmma::store_matrix_sync(interaction_output + tile_row * arch.head_dim + tile_col, c_frag, arch.head_dim, wmma::mem_row_major);
                }
            }
            __syncwarp();

            // GELU activation
            for (int idx = threadIdx.x; idx < head_size; idx += blockDim.x) {
                float x = interaction_output[idx];
                interaction_output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845f * (x + 0.044715f * x * x * x)));
            }
            __syncwarp();

            // Convert interaction output to FP16 for value projection
            half* value_input = interaction_input;
            for (int idx = threadIdx.x; idx < head_size; idx += blockDim.x) {
                value_input[idx] = __float2half(interaction_output[idx]);
            }
            __syncwarp();

            // Value projection: WMMA matmul
            float* head_output = ca_output_fp32 + head * num_cells * arch.channels;
            int value_tiles_m = (num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int value_tiles_n = (arch.channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;

            for (int tile_m = 0; tile_m < value_tiles_m; tile_m++) {
                for (int tile_n = 0; tile_n < value_tiles_n; tile_n++) {
                    int tile_row = tile_m * WMMA_TILE_DIM;
                    int tile_col = tile_n * WMMA_TILE_DIM;
                    if (tile_row >= num_cells || tile_col >= arch.channels) continue;

                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);

                    for (int k = 0; k < arch.head_dim; k += WMMA_TILE_DIM) {
                        if (k + WMMA_TILE_DIM <= arch.head_dim) {
                            wmma::load_matrix_sync(a_frag, value_input + tile_row * arch.head_dim + k, arch.head_dim);
                            wmma::load_matrix_sync(b_frag, value_weights + head * arch.head_dim * arch.channels + k * arch.channels + tile_col, arch.channels);
                            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                    }
                    wmma::store_matrix_sync(head_output + tile_row * arch.channels + tile_col, c_frag, arch.channels, wmma::mem_row_major);
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }
    dim3 affinity_grid((max_cells + WARP_SIZE - 1) / WARP_SIZE, 1);
    dim3 affinity_block(WARP_SIZE);

    // Reduce affinity across heads for each cell
    {
        int grid_size = entry->grid_size;
        int total_cells = grid_size * grid_size;
        int num_heads = entry->num_heads;
        int head_dim = entry->head_dim;
        int total_elements = num_heads * head_dim;
        MultiHeadCAState* ca_state = entry->ca_state;
        float* ca_output = ca_state->ca_output;

        for (int cell_idx = 0; cell_idx < total_cells; cell_idx++) {
            float affinity = 0.0f;
            for (int i = threadIdx.x; i < total_elements; i += WARP_SIZE) {
                int head = i / head_dim;
                int dim = i % head_dim;
                int idx = head * total_cells * head_dim + cell_idx * head_dim + dim;
                affinity += ca_output[idx];
            }
            affinity = WarpReduce<32>::sum(affinity);
            if (threadIdx.x == 0) {
                ca_state->affinity_reduced[cell_idx] = affinity;
            }
        }
        __syncthreads();
    }

    dim3 flow_grid((max_grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM,
                   (max_grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM,
                   1);
    dim3 flow_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

    // Compute flow field from affinity and concentration
    {
        int grid_size = entry->grid_size;
        int total_cells = grid_size * grid_size;
        int channels = entry->channels;
        MultiHeadCAState* ca_state = entry->ca_state;
        float* affinity = ca_state->affinity_reduced;
        float* concentration = ca_state->ca_concentration;
        float* flow_field = ca_state->flow_field;

        for (int cell_idx = threadIdx.x; cell_idx < total_cells; cell_idx += blockDim.x) {
            int x = cell_idx % grid_size;
            int y = cell_idx / grid_size;

            float U_center = affinity[cell_idx];
            int x_E = min(x + 1, grid_size - 1);
            int y_N = min(y + 1, grid_size - 1);
            float U_E = affinity[y * grid_size + x_E];
            float U_N = affinity[y_N * grid_size + x];

            float A_sum_center = 0.0f, A_sum_E = 0.0f, A_sum_N = 0.0f;
            for (int c = 0; c < channels; c++) {
                A_sum_center += concentration[cell_idx * channels + c];
                A_sum_E += concentration[(y * grid_size + x_E) * channels + c];
                A_sum_N += concentration[(y_N * grid_size + x) * channels + c];
            }

            float2 F = FlowLeniaOps::compute_flow_differentiable(
                U_center, U_E, U_N, A_sum_center, A_sum_E, A_sum_N,
                entry->flow_beta_A, entry->flow_n,
                entry->flow_alpha_min, entry->flow_alpha_max, entry->flow_sharpness
            );

            flow_field[cell_idx * 2 + 0] = F.x;
            flow_field[cell_idx * 2 + 1] = F.y;
        }
        __syncthreads();
    }

    int max_buffer_size = max_grid_size * max_grid_size * arch.channels;
    dim3 clear_grid((max_buffer_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 clear_block(BLOCK_SIZE);

    // Clear reintegration buffer
    {
        int buffer_size = entry->grid_size * entry->grid_size * entry->channels;
        float* reint_buf = entry->ca_state->reintegration_buffer;
        for (int idx = threadIdx.x; idx < buffer_size; idx += blockDim.x) {
            reint_buf[idx] = 0.0f;
        }
        __syncthreads();
    }

    dim3 reint_grid(max_grid_size, max_grid_size, 1);
    dim3 reint_block(BLOCK_SIZE);

    // Redistribute mass via flow field transport
    {
        int grid_size = entry->grid_size;
        int total_cells = grid_size * grid_size;
        int channels = entry->channels;
        float* flow_field = entry->ca_state->flow_field;
        float* concentration = entry->ca_state->ca_concentration;
        float* buffer = entry->ca_state->reintegration_buffer;
        float dt = entry->flow_resource_dt;

        for (int source_idx = 0; source_idx < total_cells; source_idx++) {
            int source_x = source_idx % grid_size;
            int source_y = source_idx / grid_size;
            float Fx = flow_field[source_idx * 2 + 0];
            float Fy = flow_field[source_idx * 2 + 1];

            for (int c = threadIdx.x; c < channels; c += blockDim.x) {
                float source_mass = concentration[source_idx * channels + c];
                FlowLeniaOps::bilinear_transport_forward(
                    source_mass, (float)source_x, (float)source_y,
                    Fx, Fy, dt, grid_size, buffer, c, channels
                );
            }
        }
        __syncthreads();
    }

    dim3 copy_grid((max_buffer_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 copy_block(BLOCK_SIZE);

    // Copy reintegration buffer back to concentration
    {
        int buffer_size = entry->grid_size * entry->grid_size * entry->channels;
        float* conc = entry->ca_state->ca_concentration;
        float* reint_buf = entry->ca_state->reintegration_buffer;
        for (int idx = threadIdx.x; idx < buffer_size; idx += blockDim.x) {
            conc[idx] = reint_buf[idx];
        }
        __syncthreads();
    }

    // Update global chemical field from CA concentration
    // CA concentration layout: [grid² × channels], emit channel 0 (concentration channel)
    {
        int grid_size = entry->grid_size;
        int channels = entry->channels;
        int total_cells = grid_size * grid_size;
        float* ca_conc = entry->ca_state->ca_concentration;
        for (int cell_idx = threadIdx.x; cell_idx < total_cells; cell_idx += blockDim.x) {
            float val = ca_conc[cell_idx * channels + 0];  // Read channel 0 from each cell
            if (isfinite(val)) {
                atomicAdd(&chemical_field->concentration[cell_idx], val);
            } else {
                if (threadIdx.x == 0) printf("!W:NaN_blocked entry=%d cell=%d\n", entry_idx, cell_idx);
            }
        }
        __syncthreads();
    }

    dim3 rank_grid(1);
    dim3 rank_block(BLOCK_SIZE);

    // Compute effective rank from latent genome dimensions
    {
        float* latent_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];
        int latent_dim = GENOME_LATENT_DIM_MAX;

        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < latent_dim; i += blockDim.x) {
            local_sum += latent_genome[i];
        }
        float mean = WarpReduce<32>::sum(local_sum) / latent_dim;
        mean = __shfl_sync(0xffffffff, mean, 0);

        float local_var = 0.0f;
        for (int i = threadIdx.x; i < latent_dim; i += blockDim.x) {
            float diff = latent_genome[i] - mean;
            local_var += diff * diff;
        }
        float variance = WarpReduce<32>::sum(local_var) / latent_dim;

        if (threadIdx.x == 0 && variance >= 0.0f) {
            effective_rank_history[entry_idx] = sqrtf(variance) * latent_dim;
        }
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
    // DEBUG: Print at kernel entry to verify execution
    if (blockIdx.x == 0 && threadIdx.x == 0) printf("V:beh_KERNEL_START gen=%d\n", generation);

    int entry_idx = blockIdx.x;
    if (entry_idx == 0 && threadIdx.x == 0) printf("V:beh_ENTER entry=0 gen=%d blockIdx=%d threadIdx=%d\n", generation, blockIdx.x, threadIdx.x);
    ComponentPool* pool = organism->pool;

    if (entry_idx >= pool->capacity) {
        if (entry_idx == 0 && threadIdx.x == 0) printf("V:beh_OVER_CAP entry=0 cap=%d\n", pool->capacity);
        return;
    }

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) {
        if (entry_idx == 0 && threadIdx.x == 0) printf("V:beh_NOT_ALIVE entry=0\n");
        return;
    }
    if (entry_idx == 0 && threadIdx.x == 0) printf("V:beh_ALIVE entry=0 gen=%d\n", generation);

    int num_agents = POOL_CAPACITY_MAX;

    float* primary_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];
    float* primary_parent_temp = &workspace_genomes[entry_idx * GENOME_SIZE * 2 + GENOME_SIZE];

    reconstruct_genome_from_archive(
        entry->parent_hash,
        (GPUElite*)organism->archive,
        organism->archive_size,
        entry->delta_indices,
        entry->delta_values,
        entry->num_deltas,
        entry->max_deltas,
        primary_genome,
        GENOME_SIZE,
        primary_parent_temp,
        organism->diresa_genome_weights
    );

    uint64_t genome_hash = entry->genome_hash;
    float* genome = primary_genome;
    float* gradients = entry->gradients;
    float ctx_metabolic = entry->fitness;
    float ctx_stress = entry->hunger;

    float ctx_morphogen = chemical_field->cached_mean;

    BehavioralDimensions dims;
    dims.derive_from_genome(genome_hash, primary_genome);

    if (threadIdx.x == 0) printf("V:beh_entry gen=%d entry=%d\n", generation, entry_idx);

    // Per-entry behavioral field and gradient buffers with strided access
    int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;
    int behavioral_buffer_size = arch.grid_size * arch.grid_size * behavioral_dim;
    float* behavioral_field = &organism->behavioral_field_pool[entry_idx * behavioral_buffer_size];
    float* behavioral_gradients_pool = &organism->behavioral_gradient_pool[entry_idx * behavioral_buffer_size];

    // Compute behavioral field from agent positions
    {
        int grid_size = arch.grid_size;
        int total_cells = grid_size * grid_size;
        int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;

        ChemotaxisParams chem_params;
        chem_params.derive_from_genome_hash(genome_hash);

        InitContext init_ctx;
        init_ctx.derive_from_genome(genome_hash, primary_genome);

        float behavioral_field_sigma = chem_params.get_behavioral_field_sigma(primary_genome, entry->gradients, init_ctx.metabolic, init_ctx.stress, init_ctx.morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

        for (int cell_idx = threadIdx.x; cell_idx < total_cells; cell_idx += blockDim.x) {
            int x = cell_idx % grid_size;
            int y = cell_idx / grid_size;
            float px = (float)x / grid_size;
            float py = (float)y / grid_size;

            int d_offset = 0;

            for (int d = 0; d < dims.hw_dim; d++) {
                float field_value = 0.0f;
                float weight_sum = 0.0f;
                for (int agent_id = 0; agent_id < num_agents; agent_id++) {
                    BehavioralState* agent = &agents[agent_id];
                    float dx = fabsf(px - agent->position[0]);
                    float dy = fabsf(py - agent->position[1]);
                    dx = fminf(dx, NORMALIZED_MAX - dx);
                    dy = fminf(dy, NORMALIZED_MAX - dy);
                    float dist_sq = dx * dx + dy * dy;
                    float weight = expf(-dist_sq / (GAUSSIAN_VARIANCE_DENOMINATOR * behavioral_field_sigma * behavioral_field_sigma));
                    field_value += weight * agent->hw_coords[d];
                    weight_sum += weight;
                }
                int field_idx = (y * grid_size + x) * behavioral_dim + d_offset;
                if (is_meaningful(weight_sum, 1.0f)) {
                    behavioral_field[field_idx] = field_value / weight_sum;
                }
                // else: preserve existing value (temporal coherence)
                d_offset++;
            }

            for (int d = 0; d < dims.task_dim; d++) {
                float field_value = 0.0f;
                float weight_sum = 0.0f;
                for (int agent_id = 0; agent_id < num_agents; agent_id++) {
                    BehavioralState* agent = &agents[agent_id];
                    float dx = fabsf(px - agent->position[0]);
                    float dy = fabsf(py - agent->position[1]);
                    dx = fminf(dx, NORMALIZED_MAX - dx);
                    dy = fminf(dy, NORMALIZED_MAX - dy);
                    float dist_sq = dx * dx + dy * dy;
                    float weight = expf(-dist_sq / (GAUSSIAN_VARIANCE_DENOMINATOR * behavioral_field_sigma * behavioral_field_sigma));
                    field_value += weight * agent->task_coords[d];
                    weight_sum += weight;
                }
                int field_idx = (y * grid_size + x) * behavioral_dim + d_offset;
                if (is_meaningful(weight_sum, 1.0f)) {
                    behavioral_field[field_idx] = field_value / weight_sum;
                }
                // else: preserve existing value (temporal coherence)
                d_offset++;
            }

            for (int d = 0; d < dims.gen_dim; d++) {
                float field_value = 0.0f;
                float weight_sum = 0.0f;
                for (int agent_id = 0; agent_id < num_agents; agent_id++) {
                    BehavioralState* agent = &agents[agent_id];
                    float dx = fabsf(px - agent->position[0]);
                    float dy = fabsf(py - agent->position[1]);
                    dx = fminf(dx, NORMALIZED_MAX - dx);
                    dy = fminf(dy, NORMALIZED_MAX - dy);
                    float dist_sq = dx * dx + dy * dy;
                    float weight = expf(-dist_sq / (GAUSSIAN_VARIANCE_DENOMINATOR * behavioral_field_sigma * behavioral_field_sigma));
                    field_value += weight * agent->gen_coords[d];
                    weight_sum += weight;
                }
                int field_idx = (y * grid_size + x) * behavioral_dim + d_offset;
                if (is_meaningful(weight_sum, 1.0f)) {
                    behavioral_field[field_idx] = field_value / weight_sum;
                }
                // else: preserve existing value (temporal coherence)
                d_offset++;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) printf("V:beh_3_behavioral_field gen=%d\n", generation);

    // Warp-level CA on behavioral field
    {
        int width = arch.grid_size;
        int height = arch.grid_size;
        int total_cells = width * height;
        int num_tiles = (total_cells + WARP_SIZE - 1) / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        CAParams ca_params;
        ca_params.derive_from_genome_hash(genome_hash);
        float warp_ca_growth_rate = ca_params.get_warp_ca_growth_rate(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

        for (int tile = 0; tile < num_tiles; tile++) {
            int cell_idx = tile * WARP_SIZE + lane_id;
            if (cell_idx >= total_cells) continue;

            int tile_x = cell_idx % width;
            int tile_y = cell_idx / width;

            float my_state = behavioral_field[tile_y * width + tile_x];
            unsigned mask = __ballot_sync(0xffffffff, 1);

            float sum = 0.0f;
            sum += get_neighbor_2d(my_state, -1, -1, width, mask);
            sum += get_neighbor_2d(my_state, 0, -1, width, mask);
            sum += get_neighbor_2d(my_state, 1, -1, width, mask);
            sum += get_neighbor_2d(my_state, -1, 0, width, mask);
            sum += get_neighbor_2d(my_state, 1, 0, width, mask);
            sum += get_neighbor_2d(my_state, -1, 1, width, mask);
            sum += get_neighbor_2d(my_state, 0, 1, width, mask);
            sum += get_neighbor_2d(my_state, 1, 1, width, mask);

            float avg = sum / CA_KERNEL_NEIGHBOR_COUNT;
            float growth = avg * expf(-avg * avg * 2.0f);

            float total_mass = WarpReduce<WARP_SIZE>::sum(my_state);
            float new_val = my_state + warp_ca_growth_rate * growth;
            float new_total = WarpReduce<WARP_SIZE>::sum(new_val);

            if (is_meaningful(new_total, total_mass)) {
                new_val *= total_mass / new_total;
            }

            behavioral_gradients_pool[tile_y * width + tile_x] = new_val;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) printf("V:beh_4_warp_ca gen=%d\n", generation);

    // Compute behavioral gradients
    {
        int grid_size = arch.grid_size;
        int total_cells = grid_size * grid_size;
        int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;

        for (int cell_idx = threadIdx.x; cell_idx < total_cells; cell_idx += blockDim.x) {
            int x = cell_idx % grid_size;
            int y = cell_idx / grid_size;

            for (int dim = 0; dim < behavioral_dim; dim++) {
                float grad_x, grad_y;
                Stencils::gradients_at(grad_x, grad_y, &behavioral_gradients_pool[dim], x, y, grid_size, behavioral_dim);

                float grad_sq = grad_x * grad_x + grad_y * grad_y;
                float magnitude = sqrtf(grad_sq) + safe_epsilon(grad_sq);
                grad_x /= magnitude;
                grad_y /= magnitude;

                int grad_idx = ((y * grid_size + x) * behavioral_dim + dim) * 2;
                behavioral_gradients_pool[grad_idx] = grad_x;
                behavioral_gradients_pool[grad_idx + 1] = grad_y;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) printf("V:beh_5_grad gen=%d\n", generation);

    int chemotaxis_dt_slot = derive_param_slot(genome_hash, "chemotaxis_dt");
    float chemotaxis_dt = genome_to_param(genome, gradients, chemotaxis_dt_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, CHEMOTAXIS_DT_MIN, CHEMOTAXIS_DT_MAX);

    // Chemotactic agent navigation
    {
        int grid_size = arch.grid_size;
        int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;
        float* concentration = chemical_field->concentration;
        float* gradient_x_arr = chemical_field->gradient_x;
        float* gradient_y_arr = chemical_field->gradient_y;

        for (int agent_id = threadIdx.x; agent_id < num_agents; agent_id += blockDim.x) {
            BehavioralState* agent = &agents[agent_id];

            float context_metabolic_agent = agent->sensitivity;
            float context_stress_agent = sqrtf(agent->velocity[0] * agent->velocity[0] + agent->velocity[1] * agent->velocity[1]);
            int grid_x = min(max((int)(agent->position[0] * grid_size), 0), grid_size - 1);
            int grid_y = min(max((int)(agent->position[1] * grid_size), 0), grid_size - 1);
            int idx = grid_y * grid_size + grid_x;
            float context_morphogen_agent = concentration[idx];

            ChemotaxisParams chem_params;
            chem_params.derive_from_genome_hash(agent->genome_hash);

            float theta = chem_params.get_theta(primary_genome, entry->gradients, context_metabolic_agent, context_stress_agent, context_morphogen_agent, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
            float sigma = chem_params.get_sigma(primary_genome, entry->gradients, context_metabolic_agent, context_stress_agent, context_morphogen_agent, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
            float gradient_mix_weight = chem_params.get_gradient_mix_weight(primary_genome, entry->gradients, context_metabolic_agent, context_stress_agent, context_morphogen_agent, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

            float chem_grad_x = gradient_x_arr[idx];
            float chem_grad_y = gradient_y_arr[idx];

            float behav_grad_x = 0.0f, behav_grad_y = 0.0f;
            int d_offset = 0;
            for (int d = 0; d < dims.hw_dim; d++) {
                int grad_idx = ((grid_y * grid_size + grid_x) * behavioral_dim + d_offset) * 2;
                behav_grad_x += behavioral_gradients_pool[grad_idx] * agent->hw_coords[d];
                behav_grad_y += behavioral_gradients_pool[grad_idx + 1] * agent->hw_coords[d];
                d_offset++;
            }
            for (int d = 0; d < dims.task_dim; d++) {
                int grad_idx = ((grid_y * grid_size + grid_x) * behavioral_dim + d_offset) * 2;
                behav_grad_x += behavioral_gradients_pool[grad_idx] * agent->task_coords[d];
                behav_grad_y += behavioral_gradients_pool[grad_idx + 1] * agent->task_coords[d];
                d_offset++;
            }
            for (int d = 0; d < dims.gen_dim; d++) {
                int grad_idx = ((grid_y * grid_size + grid_x) * behavioral_dim + d_offset) * 2;
                behav_grad_x += behavioral_gradients_pool[grad_idx] * agent->gen_coords[d];
                behav_grad_y += behavioral_gradients_pool[grad_idx + 1] * agent->gen_coords[d];
                d_offset++;
            }

            float behav_sq = behav_grad_x * behav_grad_x + behav_grad_y * behav_grad_y;
            float behav_magnitude = sqrtf(behav_sq) + safe_epsilon(behav_sq);
            behav_grad_x /= behav_magnitude;
            behav_grad_y /= behav_magnitude;

            float chem_weight = gradient_mix_weight;
            float behav_weight = NORMALIZED_MAX - gradient_mix_weight;
            float mixed_grad_x = chem_grad_x * chem_weight + behav_grad_x * behav_weight;
            float mixed_grad_y = chem_grad_y * chem_weight + behav_grad_y * behav_weight;

            unsigned int seed = agent_id * RNG_SEED_MULTIPLIER + (unsigned int)(chemotaxis_dt * 1000.0f);
            seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
            float noise_scale = sigma * agent->exploration_noise;
            float noise_x = noise_scale * ((seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE - 0.5f);
            seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
            float noise_y = noise_scale * ((seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE - 0.5f);

            agent->velocity[0] += chemotaxis_dt * (agent->sensitivity * mixed_grad_x - theta * agent->velocity[0] + noise_x);
            agent->velocity[1] += chemotaxis_dt * (agent->sensitivity * mixed_grad_y - theta * agent->velocity[1] + noise_y);

            float vel_magnitude = sqrtf(agent->velocity[0] * agent->velocity[0] + agent->velocity[1] * agent->velocity[1]);
            int max_vel_slot = derive_param_slot(agent->genome_hash, "max_agent_velocity");
            float max_agent_velocity = genome_to_param(primary_genome, entry->gradients, max_vel_slot, context_metabolic_agent, context_stress_agent, context_morphogen_agent, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, MAX_AGENT_VELOCITY_BASE_MIN, MAX_AGENT_VELOCITY_BASE_MAX);
            if (vel_magnitude > max_agent_velocity) {
                agent->velocity[0] *= (max_agent_velocity / vel_magnitude);
                agent->velocity[1] *= (max_agent_velocity / vel_magnitude);
            }

            agent->position[0] += agent->velocity[0] * chemotaxis_dt;
            agent->position[1] += agent->velocity[1] * chemotaxis_dt;
            agent->position[0] = agent->position[0] - floorf(agent->position[0]);
            agent->position[1] = agent->position[1] - floorf(agent->position[1]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) printf("V:beh_6_chemotaxis gen=%d\n", generation);

    // Store navigation history to temporal memory
    {
        int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;
        int memory_entry_size = behavioral_dim + AGENT_SPATIAL_DIMS;
        float* memory_data_pool = organism->memory_data_pool;

        for (int agent_id = threadIdx.x; agent_id < num_agents; agent_id += blockDim.x) {
            float* d_memory_data = memory_data_pool + agent_id * memory_entry_size;

            d_memory_data[0] = agents[agent_id].position[0];
            d_memory_data[1] = agents[agent_id].position[1];
            d_memory_data[2] = agents[agent_id].velocity[0];
            d_memory_data[3] = agents[agent_id].velocity[1];

            int offset = AGENT_SPATIAL_DIMS;
            for (int i = 0; i < dims.hw_dim; i++) {
                d_memory_data[offset++] = agents[agent_id].hw_coords[i];
            }
            for (int i = 0; i < dims.task_dim; i++) {
                d_memory_data[offset++] = agents[agent_id].task_coords[i];
            }
            for (int i = 0; i < dims.gen_dim; i++) {
                d_memory_data[offset++] = agents[agent_id].gen_coords[i];
            }
        }
        __syncthreads();

        // Update tube metadata and copy data to entry
        if (threadIdx.x == 0) {
            float importance = agents[0].exploration_noise;
            float* d_memory_data = memory_data_pool;
            TemporalTube* tube = memory_tubes;

            int idx = tube->head;
            tube->entries[idx].size = memory_entry_size;
            tube->entries[idx].timestamp = tube->global_time;
            tube->entries[idx].importance = importance;
            tube->entries[idx].decay_factor = 1.0f;

            if (d_memory_data && memory_entry_size > 0 && tube->entries[idx].data) {
                for (int i = 0; i < memory_entry_size; i++) {
                    tube->entries[idx].data[i] = d_memory_data[i];
                }
            }

            tube->head = (tube->head + 1) % tube->capacity;
            if (tube->count < tube->capacity) {
                tube->count++;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) printf("V:beh_7_done gen=%d entry=%d\n", generation, entry_idx);
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
    if (tid >= POOL_CAPACITY_MAX) return;

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
        printf("V:nav_hist_pre_store gen=%d\n", generation);
        store_memory_kernel<<<1, 1>>>(
            tubes,
            d_memory_data,
            memory_entry_size,
            importance
        );
        cudaDeviceSynchronize();
        printf("V:nav_hist_post_store gen=%d\n", generation);
    }
}

__global__ void memory_params_kernel(
    TemporalTube* tubes,
    MemoryUpdateParams* params_out,
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
    if (threadIdx.x != 0) return;

    int consolidation_slot = derive_param_slot(genome_hash, "memory_consolidation_threshold");
    params_out->consolidation_threshold = genome_to_param(genome, gradients, consolidation_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CONSOLIDATION_THRESHOLD_MIN, CONSOLIDATION_THRESHOLD_MAX);

    int flow_lenia_dt_slot = derive_param_slot(genome_hash, "flow_lenia_dt");
    params_out->flow_lenia_dt = genome_to_param(genome, gradients, flow_lenia_dt_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, FLOW_LENIA_DT_MIN, FLOW_LENIA_DT_MAX);

    int decay_threshold_slot = derive_param_slot(genome_hash, "memory_decay_threshold");
    params_out->decay_threshold = genome_to_param(genome, gradients, decay_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, DECAY_THRESHOLD_MIN, DECAY_THRESHOLD_MAX);

    params_out->tube_count = tubes->count;
    params_out->should_compact = (generation % TELEMETRY_DETAILED == 0 && tubes->count > 0) ? 1 : 0;
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
    }
}

__global__ void wire_behavioral_agents_kernel(
    BehavioralState* agents,
    int num_agents,
    float* hw_buffer,
    float* task_buffer,
    float* gen_buffer,
    int hw_dim,
    int task_dim,
    int gen_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;

    agents[idx].hw_coords = &hw_buffer[idx * hw_dim];
    agents[idx].task_coords = &task_buffer[idx * task_dim];
    agents[idx].gen_coords = &gen_buffer[idx * gen_dim];
}

__global__ void init_organism_kernel(
    Organism* organism,
    Dataset** dataset_array,
    Dataset** test_dataset_array,
    int pool_capacity,
    float* workspace_genomes,
    OrganismPreallocatedBuffers* buffers
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("V:init_org_enter pool_cap=%d\n", pool_capacity);
        organism->generation = 0;

        float* organism_seed_genome = &workspace_genomes[0];
        uint64_t organism_genome_hash = gpu_sha256(organism_seed_genome, GENOME_SIZE);

        int initial_pool_size_slot = derive_param_slot(organism_genome_hash, "initial_pool_size");
        float initial_pool_size_norm = fmaxf(0.0f, fminf(1.0f, (organism_seed_genome[initial_pool_size_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
        int initial_pool_size = 1 + (int)(initial_pool_size_norm * (pool_capacity - 1));

        organism->active_components = initial_pool_size;

        cudaError_t err;

        organism->pool = buffers->pool;
        organism->pool->entries = buffers->pool_entries;
        organism->pool->alive_indices = buffers->pool_alive_indices;
        organism->pool->alive_indices_count = 0;
        organism->pool->alive_flags = buffers->pool_alive_flags;
        organism->pool->fitness_values = buffers->pool_fitness_values;
        organism->pool->capacity = pool_capacity;
        *((int*)&organism->pool->active_count) = initial_pool_size;
        *((int*)&organism->pool->total_spawned) = 0;
        *((int*)&organism->pool->total_culled) = 0;

        organism->pool_compaction_flags = buffers->pool_compaction_flags;
        organism->pool_compaction_scan = buffers->pool_compaction_scan;
        organism->pool_compaction_recursive_workspace = buffers->pool_compaction_recursive_workspace;
        organism->memory_params = buffers->memory_params;
        organism->weight_inherit_child_indices = buffers->weight_inherit_child_indices;
        organism->weight_inherit_parent_indices = buffers->weight_inherit_parent_indices;
        organism->weight_inherit_num_pending = buffers->weight_inherit_num_pending;

        organism->archive = buffers->archive;
        organism->voronoi_cells = buffers->voronoi_cells;
        organism->behavioral_agents = buffers->behavioral_agents;
        organism->buffers = buffers;

        organism->archive_size = 0;
        organism->num_voronoi_cells = pool_capacity;

        uint16_t* delta_indices_buffer = buffers->delta_indices_buffer;
        float* delta_values_buffer = buffers->delta_values_buffer;
        float* gradients_buffer = buffers->gradients_buffer;

        int pool_blocks = (pool_capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("V:init_org_pre_pool blocks=%d alive_flags=%p fitness_values=%p\n", pool_blocks, (void*)organism->pool->alive_flags, (void*)organism->pool->fitness_values);
        __threadfence();  // Ensure pool pointer writes are visible to child kernel
        init_pool_kernel<<<pool_blocks, BLOCK_SIZE>>>(organism->pool, pool_capacity, delta_indices_buffer, delta_values_buffer, gradients_buffer);
        err = cudaGetLastError();
        printf("V:init_org_post_pool err=%d\n", (int)err);
        if (err != cudaSuccess) { return; }
    }
}

__global__ void init_voronoi_pointers_kernel(
    VoronoiCell* cells,
    int num_cells,
    float* hw_buffer,
    float* task_buffer,
    float* gen_buffer,
    int hw_dim,
    int task_dim,
    int gen_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    
    cells[idx].hw_centroid = &hw_buffer[idx * hw_dim];
    cells[idx].task_centroid = &task_buffer[idx * task_dim];
    cells[idx].gen_centroid = &gen_buffer[idx * gen_dim];
}

__global__ void init_ca_weights_kernel(
    half* weights,
    int total_size,
    uint64_t genome_hash,
    int fan_in,
    int fan_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    // Xavier/Glorot initialization: std = sqrt(2.0 / (fan_in + fan_out))
    // Using genome hash as PRNG seed for reproducibility
    PRNGState rng;
    rng.s0 = genome_hash ^ (idx * XORSHIFT_GOLDEN_RATIO_A);
    rng.s1 = (genome_hash >> 32) ^ (idx * XORSHIFT_GOLDEN_RATIO_B);

    float std_dev = sqrtf(2.0f / (float)(fan_in + fan_out));

    // Box-Muller transform to generate Gaussian random values
    float u1 = rng.next();
    float u2 = rng.next();
    float z0 = sqrtf(-2.0f * safe_log(u1)) * cosf(2.0f * M_PI * u2);
    float value = z0 * std_dev;

    weights[idx] = __float2half(value);
}

__global__ void init_organism_phase2_kernel(
    Organism* organism,
    Dataset** dataset_array,
    Dataset** test_dataset_array,
    unsigned int seed,
    float* workspace_genomes,
    OrganismPreallocatedBuffers* buffers
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cudaError_t err;
        printf("V:p2_enter seed=%u\n", seed);

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

        // Wire behavioral agent coordinate pointers via parallel kernel
        wire_behavioral_agents_kernel<<<(POOL_CAPACITY_MAX + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            organism->behavioral_agents,
            POOL_CAPACITY_MAX,
            buffers->behavioral_hw_coords_buffer,
            buffers->behavioral_task_coords_buffer,
            buffers->behavioral_gen_coords_buffer,
            dims.hw_dim,
            dims.task_dim,
            dims.gen_dim
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 wire_behavioral err=%d\n", (int)err);
            return;
        }

        organism->archive->fitness = buffers->archive_fitness;
        organism->archive->coherence = buffers->archive_coherence;
        organism->archive->effective_rank = buffers->archive_effective_rank;
        organism->archive->genome_hash = buffers->archive_genome_hash;
        organism->archive->parent_ids = buffers->archive_parent_ids;
        organism->archive->generation = buffers->archive_generation;
        organism->archive->hw_coords = buffers->archive_hw_coords;
        organism->archive->task_coords = buffers->archive_task_coords;
        organism->archive->gen_coords = buffers->archive_gen_coords;
        organism->archive->latent_genome = buffers->archive_latent_genome;
        organism->archive->hardware_features = buffers->archive_hardware_features;
        organism->archive->task_performance = buffers->archive_task_performance;
        organism->archive->per_class_accuracy = buffers->archive_per_class_accuracy;
        organism->archive->hash_table_keys = buffers->archive_hash_table_keys;
        organism->archive->hash_table_values = buffers->archive_hash_table_values;

        // Initialize hash table to empty
        int ht_blocks = (GENOME_HASH_TABLE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_hash_table_kernel<<<ht_blocks, BLOCK_SIZE>>>(
            organism->archive->hash_table_keys,
            organism->archive->hash_table_values,
            GENOME_HASH_TABLE_SIZE
        );
        printf("V:p2_sync1_pre ht_blocks=%d\n", ht_blocks);
        err = cudaDeviceSynchronize();
        printf("V:p2_sync1_post err=%d\n", (int)err);
        if (err != cudaSuccess) {
            printf("!E:init2 hash_table_sync err=%d\n", (int)err);
            return;
        }

        int pool_capacity = organism->pool->capacity;

        // Use pre-allocated component pool behavioral coordinate buffers
        organism->hw_coords_pool = buffers->hw_coords_pool;
        organism->task_coords_pool = buffers->task_coords_pool;
        organism->gen_coords_pool = buffers->gen_coords_pool;

        // Use pre-allocated contiguous buffers for all voronoi cell centroids
        float* voronoi_hw_centroid_buffer = buffers->voronoi_hw_centroid_buffer;
        float* voronoi_task_centroid_buffer = buffers->voronoi_task_centroid_buffer;
        float* voronoi_gen_centroid_buffer = buffers->voronoi_gen_centroid_buffer;

        // Launch kernel to set pointers into contiguous buffers
        int pointer_blocks = (organism->num_voronoi_cells + 255) / 256;
        init_voronoi_pointers_kernel<<<pointer_blocks, 256>>>(
            organism->voronoi_cells,
            organism->num_voronoi_cells,
            voronoi_hw_centroid_buffer,
            voronoi_task_centroid_buffer,
            voronoi_gen_centroid_buffer,
            dims.hw_dim,
            dims.task_dim,
            dims.gen_dim
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 voronoi_pointers err=%d\n", (int)err);
            return;
        }

        // Initialize Voronoi cell centroids with genome-derived dimensions
        int voronoi_blocks = (organism->num_voronoi_cells + 255) / 256;
        init_voronoi_cells_kernel<<<voronoi_blocks, 256>>>(
            organism->voronoi_cells,
            organism->num_voronoi_cells,
            dims.hw_dim,
            dims.task_dim,
            dims.gen_dim,
            seed + 555555
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 voronoi_cells err=%d\n", (int)err);
            return;
        }
        int default_decay_rate_slot = derive_param_slot(organism->pool->entries[0].genome_hash, "memory_default_decay_rate");
        float default_decay_rate_norm = (primary_genome[default_decay_rate_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float default_decay_rate = DEFAULT_DECAY_RATE_MIN + default_decay_rate_norm * (DEFAULT_DECAY_RATE_MAX - DEFAULT_DECAY_RATE_MIN);

        ArchitectureParams arch = get_arch_from_pool(organism->pool, 0);
        int field_size = arch.grid_size * arch.grid_size;

        organism->telemetry = buffers->telemetry;
        organism->telemetry->valid = false;
        organism->telemetry->generation = 0;

        // Get actual device heap limit
        size_t heap_limit;
        err = cudaDeviceGetLimit(&heap_limit, cudaLimitMallocHeapSize);
        if (err != cudaSuccess) { printf("!E:init2 heap_limit err=%d\n", (int)err); return; }
        organism->telemetry->memory_allocation.device_heap_limit = heap_limit;
        organism->telemetry->memory_allocation.device_heap_allocated = 0;

        organism->ca_state_pool = buffers->ca_state_pool;
        organism->chemical_field = buffers->chemical_field;

        organism->chemical_field->history = buffers->chemical_field_history;
        organism->chemical_field->history->entries = buffers->chemical_field_history_entries;

        init_tube_kernel<<<(MAX_HISTORY_LENGTH + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
            organism->chemical_field->history,
            MAX_HISTORY_LENGTH,
            default_decay_rate
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 init_tube err=%d\n", (int)err);
            return;
        }

        float* history_data_buffer = buffers->history_data_buffer;
        wire_tube_data_kernel<<<(MAX_HISTORY_LENGTH + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
            organism->chemical_field->history,
            history_data_buffer,
            field_size
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 wire_tube err=%d\n", (int)err);
            return;
        }

        int perception_size = arch.num_heads * arch.channels * arch.head_dim;
        int interaction_size = arch.num_heads * arch.head_dim * arch.head_dim;
        int value_size = arch.num_heads * arch.head_dim * arch.channels;
        int total_weights_size = perception_size + interaction_size + value_size;

        dim3 weight_init_grid((total_weights_size + BLOCK_SIZE - 1) / BLOCK_SIZE, pool_capacity);
        dim3 weight_init_block(BLOCK_SIZE);

        // NOTE: init_organism_ca_weights_kernel moved to after entry->ca_state setup loop
        // (was causing race condition - kernel read entry->ca_state before it was initialized)

        float* all_ca_state = buffers->all_ca_state;
        MultiHeadCAState* ca_state_pool = buffers->ca_state_pool;

        for (int entry_idx = 0; entry_idx < pool_capacity; entry_idx++) {
            PoolEntry* entry = &organism->pool->entries[entry_idx];
            MultiHeadCAState* entry_ca_state = &ca_state_pool[entry_idx];

            float* entry_ca_base = all_ca_state + entry_idx * CA_STATE_STRIDE;

            entry_ca_state->ca_concentration = entry_ca_base;
            entry_ca_state->ca_output = entry_ca_base + CA_CONCENTRATION_SIZE;
            entry_ca_state->affinity_reduced = entry_ca_base + CA_CONCENTRATION_SIZE + CA_OUTPUT_SIZE;
            entry_ca_state->flow_field = entry_ca_base + CA_CONCENTRATION_SIZE + CA_OUTPUT_SIZE + CA_AFFINITY_SIZE;
            entry_ca_state->reintegration_buffer = entry_ca_base + CA_CONCENTRATION_SIZE + CA_OUTPUT_SIZE + CA_AFFINITY_SIZE + CA_FLOW_SIZE;

            half* entry_weights_base = buffers->all_ca_weights + entry_idx * CA_WEIGHTS_PER_ENTRY_STRIDE;
            entry_ca_state->perception_weights = entry_weights_base;
            entry_ca_state->interaction_weights = entry_weights_base + CA_PERCEPTION_WEIGHT_SIZE;
            entry_ca_state->value_weights = entry_weights_base + CA_PERCEPTION_WEIGHT_SIZE + CA_INTERACTION_WEIGHT_SIZE;

            int fp32_stride = CA_FIELD_SIZE * (NUM_HEADS_MAX + 1) * HEAD_DIM_MAX;
            int fp16_stride = CA_FIELD_SIZE * (CHANNELS_MAX + HEAD_DIM_MAX);
            entry_ca_state->fp32_workspace = buffers->fp32_ca_workspace + entry_idx * fp32_stride;
            entry_ca_state->fp16_workspace = buffers->fp16_ca_workspace + entry_idx * fp16_stride;

            // Wire per-entry ADTape buffers from pools
            entry_ca_state->tape.entries = buffers->ad_tape_entries_pool + entry_idx * TAPE_ENTRIES_PER_ENTRY;
            entry_ca_state->tape.capacity = TAPE_ENTRIES_PER_ENTRY;
            entry_ca_state->tape.current_size = 0;
            entry_ca_state->tape.value_buffer = buffers->ad_tape_values_pool + entry_idx * TAPE_VALUES_PER_ENTRY;
            entry_ca_state->tape.grad_buffer = buffers->ad_tape_grads_pool + entry_idx * TAPE_VALUES_PER_ENTRY;
            entry_ca_state->tape.value_levels = buffers->ad_tape_levels_pool + entry_idx * TAPE_VALUES_PER_ENTRY;
            entry_ca_state->tape.value_capacity = TAPE_VALUES_PER_ENTRY;
            entry_ca_state->tape.current_value_idx = 0;
            entry_ca_state->tape.max_level = 0;
            entry_ca_state->tape.needs_weight_restore = 0;
            entry_ca_state->tape.restore_elite_idx = INT_MAX;

            // Wire per-entry TraceBuffer from organism trace pool
            entry_ca_state->trace.traces = buffers->trace_array + entry_idx * TRACE_CAPACITY;
            entry_ca_state->trace.capacity = TRACE_CAPACITY;
            entry_ca_state->trace.current_idx = 0;

            // Wire saved activation buffers (shared - one training entry at a time)
            entry_ca_state->perception_saved = buffers->perception_activations_saved;
            entry_ca_state->interaction_saved = buffers->interaction_activations_saved;
            entry_ca_state->pre_gelu_saved = buffers->pre_gelu_values_saved;

            entry->ca_state = entry_ca_state;
        }

        // Now that entry->ca_state and weight pointers are initialized, launch weight init kernel
        init_organism_ca_weights_kernel<<<weight_init_grid, weight_init_block>>>(
            organism->pool,
            arch
        );

        float* all_chem_fields = buffers->all_chem_fields;
        organism->chemical_field->concentration = all_chem_fields + CA_FIELD_SIZE * CHEM_CONCENTRATION;
        organism->chemical_field->gradient_x = all_chem_fields + CA_FIELD_SIZE * CHEM_GRADIENT_X;
        organism->chemical_field->gradient_y = all_chem_fields + CA_FIELD_SIZE * CHEM_GRADIENT_Y;
        organism->chemical_field->laplacian = all_chem_fields + CA_FIELD_SIZE * CHEM_LAPLACIAN;
        organism->chemical_field->sources = all_chem_fields + CA_FIELD_SIZE * CHEM_SOURCES;
        organism->chemical_field->decay_factors = all_chem_fields + CA_FIELD_SIZE * CHEM_DECAY_FACTORS;

        organism->fitness_history = buffers->fitness_history;
        organism->effective_rank_history = buffers->effective_rank_history;
        organism->coherence_history = buffers->coherence_history;

        float* all_rd_fields = buffers->all_rd_fields;
        organism->resource_density = all_rd_fields + CA_FIELD_SIZE * RD_RESOURCE_DENSITY;
        organism->resource_next = all_rd_fields + CA_FIELD_SIZE * RD_RESOURCE_NEXT;
        organism->fitness_landscape = all_rd_fields + CA_FIELD_SIZE * RD_FITNESS_LANDSCAPE;
        organism->resource_gradient_x = all_rd_fields + CA_FIELD_SIZE * RD_RESOURCE_GRADIENT_X;
        organism->resource_gradient_y = all_rd_fields + CA_FIELD_SIZE * RD_RESOURCE_GRADIENT_Y;

        // Initialize resource_density before gradients can be computed from it
        dim3 init_grid((arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM, (arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM);
        dim3 init_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

        init_resource_fields_kernel<<<init_grid, init_block>>>(
            organism->resource_density,
            organism->fitness_landscape,
            arch.grid_size,
            seed * RNG_SEED_MULTIPLIER,
            primary_genome,
            organism->pool->entries[0].genome_hash
        );
        printf("V:p2_sync2_pre grid_size=%d\n", arch.grid_size);
        err = cudaDeviceSynchronize();
        printf("V:p2_sync2_post err=%d\n", (int)err);

        float* shared_workspace = buffers->shared_workspace;
        organism->coherence_workspace_pool = shared_workspace;
        organism->correlation_matrix_pool = shared_workspace;
        organism->fitness_workspace_pool = shared_workspace;

        organism->lifecycle_states = buffers->lifecycle_states;

        PoolEntry* first_entry = &organism->pool->entries[0];

        // Find max num_classes across all datasets in curriculum
        int max_num_classes = 0;
        for (int i = 0; i < NUM_ACTIVE_DATASETS; i++) {
            int nc = dataset_array[i]->descriptor->num_classes;
            if (nc > max_num_classes) {
                max_num_classes = nc;
            }
        }
        int num_classes = max_num_classes;

        organism->telemetry->memory_allocation.total_gpu_allocated = 0;
        organism->telemetry->memory_allocation.archive_pools_size = 0;
        organism->telemetry->memory_allocation.training_pools_size = 0;
        organism->telemetry->memory_allocation.ca_state_size = 0;
        organism->telemetry->memory_allocation.behavioral_pools_size = 0;
        organism->telemetry->memory_allocation.diresa_weights_size = 0;
        organism->telemetry->memory_allocation.autodiff_tape_size = 0;
        organism->telemetry->memory_allocation.device_heap_limit = DEVICE_MALLOC_HEAP_MB * BYTES_PER_MB;
        organism->telemetry->memory_allocation.device_heap_allocated = 0;

        size_t diresa_struct_size = sizeof(DIRESAWeights);
        int num_replicas = first_entry->num_tempering_replicas;
        size_t diresa_size = diresa_struct_size * num_replicas;
        size_t diresa_mb = diresa_size / BYTES_PER_MB;

        organism->diresa_hw_weights = buffers->diresa_hw_weights;
        organism->diresa_task_weights = buffers->diresa_task_weights;
        organism->diresa_gen_weights = buffers->diresa_gen_weights;
        organism->diresa_genome_weights = buffers->diresa_genome_weights;

        size_t hw_stride = HARDWARE_FEATURES_DIM * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX +
                           DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX +
                           DIRESA_HIDDEN2_MAX * dims.hw_dim + dims.hw_dim +
                           dims.hw_dim * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX +
                           DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX +
                           DIRESA_HIDDEN1_MAX * HARDWARE_FEATURES_DIM + HARDWARE_FEATURES_DIM;

        size_t task_stride = dims.task_dim * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX +
                             DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX +
                             DIRESA_HIDDEN2_MAX * dims.task_dim + dims.task_dim +
                             dims.task_dim * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX +
                             DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX +
                             DIRESA_HIDDEN1_MAX * dims.task_dim + dims.task_dim;

        size_t gen_stride = 1 * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX +
                            DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX +
                            DIRESA_HIDDEN2_MAX * dims.gen_dim + dims.gen_dim +
                            dims.gen_dim * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX +
                            DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX +
                            DIRESA_HIDDEN1_MAX * 1 + 1;

        size_t genome_stride = GENOME_SIZE * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX +
                               DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX +
                               DIRESA_HIDDEN2_MAX * GENOME_LATENT_DIM_MAX + GENOME_LATENT_DIM_MAX +
                               GENOME_LATENT_DIM_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX +
                               DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX +
                               DIRESA_HIDDEN1_MAX * GENOME_SIZE + GENOME_SIZE;

        organism->diresa_hw_weight_pool = buffers->diresa_hw_weight_pool;
        organism->diresa_task_weight_pool = buffers->diresa_task_weight_pool;
        organism->diresa_gen_weight_pool = buffers->diresa_gen_weight_pool;
        organism->diresa_genome_weight_pool = buffers->diresa_genome_weight_pool;

        init_diresa_kernel<<<num_replicas, 1024>>>(
            organism->diresa_hw_weights, organism->diresa_hw_weight_pool, hw_stride,
            HARDWARE_FEATURES_DIM, dims.hw_dim, first_entry, seed + 999999);
        init_diresa_kernel<<<num_replicas, 1024>>>(
            organism->diresa_task_weights, organism->diresa_task_weight_pool, task_stride,
            dims.task_dim, dims.task_dim, first_entry, seed + 888888);
        init_diresa_kernel<<<num_replicas, 1024>>>(
            organism->diresa_gen_weights, organism->diresa_gen_weight_pool, gen_stride,
            1, dims.gen_dim, first_entry, seed + 777777);
        init_diresa_kernel<<<num_replicas, 1024>>>(
            organism->diresa_genome_weights, organism->diresa_genome_weight_pool, genome_stride,
            GENOME_SIZE, GENOME_LATENT_DIM_MAX, first_entry, seed + 666666);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 diresa err=%d\n", (int)err);
            return;
        }

        // Allocate archive compression pools on device
        // DIRESA latent genome storage (replaces SVD compressed_genome_pool)
        organism->latent_genome_pool = buffers->latent_genome_pool;

        organism->behavioral_field_pool = buffers->behavioral_field_pool;
        organism->behavioral_gradient_pool = buffers->behavioral_gradient_pool;
        organism->memory_data_pool = buffers->memory_data_pool;
        organism->prediction_error_history = buffers->prediction_error_history;
        organism->trace_buffer = buffers->trace_buffer;
        organism->trace_buffer->traces = buffers->trace_array;
        organism->hardware_geom = buffers->hardware_geom;
        organism->delta_indices_pool = buffers->delta_indices_pool;
        organism->delta_values_pool = buffers->delta_values_pool;
        organism->delta_counts_pool = buffers->delta_counts_pool;
        organism->memory_compaction_valid_flags = buffers->memory_compaction_valid_flags;
        organism->memory_compaction_scan = buffers->memory_compaction_scan;
        organism->memory_compaction_recursive_workspace = buffers->memory_compaction_recursive_workspace;
        organism->memory_compaction_buffer = buffers->memory_compaction_buffer;

        organism->fitness_rank_pool = buffers->fitness_rank_pool;
        organism->fitness_coherence_pool = buffers->fitness_coherence_pool;
        organism->fitness_history = buffers->fitness_history;
        organism->coherence_history = buffers->coherence_history;
        organism->effective_rank_history = buffers->effective_rank_history;

        organism->rng_states = buffers->rng_states;
        init_rng_states_kernel<<<(POOL_CAPACITY_MAX + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(organism->rng_states, POOL_CAPACITY_MAX, CURAND_DEFAULT_SEED);
        printf("V:p2_sync3_pre rng\n");
        err = cudaDeviceSynchronize();
        printf("V:p2_sync3_post err=%d\n", (int)err);

        // Allocate CA parameter map
        organism->param_map = buffers->param_map;
        init_ca_param_map_kernel<<<1, 1>>>(organism->param_map, arch);

        organism->current_activation_grid_size = arch.grid_size;  // Initialize grid size tracker

        // Allocate lifecycle phase tracking
        organism->lifecycle_phase_counts = buffers->lifecycle_phase_counts;

        // Reduction workspace for parallel mean computation
        organism->reduction_workspace = buffers->reduction_workspace;
        int total_cells = arch.grid_size * arch.grid_size * arch.channels;
        organism->reduction_total_cells = total_cells;
        organism->reduction_num_blocks = (total_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;

        organism->gradient_features_pool = buffers->gradient_features_pool;
        organism->gradient_logits_pool = buffers->gradient_logits_pool;
        organism->gradient_loss_pool = buffers->gradient_loss_pool;
        organism->gradient_logit_grads_pool = buffers->gradient_logit_grads_pool;
        organism->gradient_magnitudes_pool = buffers->gradient_magnitudes_pool;

        // Allocate classifier gradient buffers
        organism->pooling_weights_grad = buffers->pooling_weights_grad;
        organism->fc_weights_grad = buffers->fc_weights_grad;
        organism->fc_bias_grad = buffers->fc_bias_grad;
        organism->features_grad = buffers->features_grad;

        organism->adam_m_perception_pool = buffers->adam_m_perception_pool;
        organism->adam_v_perception_pool = buffers->adam_v_perception_pool;
        organism->adam_m_interaction_pool = buffers->adam_m_interaction_pool;
        organism->adam_v_interaction_pool = buffers->adam_v_interaction_pool;
        organism->adam_m_value_pool = buffers->adam_m_value_pool;
        organism->adam_v_value_pool = buffers->adam_v_value_pool;
        organism->adam_m_classifier_pool = buffers->adam_m_classifier_pool;
        organism->adam_v_classifier_pool = buffers->adam_v_classifier_pool;

        // Allocate Adam optimizer state for classifier weights
        organism->adam_m_pooling = buffers->adam_m_pooling;
        organism->adam_v_pooling = buffers->adam_v_pooling;
        organism->adam_m_fc_weights = buffers->adam_m_fc_weights;
        organism->adam_v_fc_weights = buffers->adam_v_fc_weights;
        organism->adam_m_fc_bias = buffers->adam_m_fc_bias;
        organism->adam_v_fc_bias = buffers->adam_v_fc_bias;

        // Allocate batch training pools
        organism->batch_ca_states_pool = buffers->batch_ca_states_pool;
        organism->batch_labels_pool = buffers->batch_labels_pool;
        organism->task_loss_pool = buffers->task_loss_pool;
        organism->reg_loss_pool = buffers->reg_loss_pool;
        organism->rank_loss_pool = buffers->rank_loss_pool;
        organism->coherence_loss_pool = buffers->coherence_loss_pool;
        organism->diversity_loss_pool = buffers->diversity_loss_pool;
        organism->total_loss_pool = buffers->total_loss_pool;

        // Allocate training mode controller
        organism->training_mode = buffers->training_mode;
        init_training_mode_kernel<<<1, 1>>>(organism->training_mode, organism->pool->entries[0].grid_size, buffers->batch_ca_states_pool, buffers->batch_labels_pool);

        // Allocate classification head
        organism->classifier = buffers->classifier;

        // Use preallocated classifier workspace: pooling_weights + fc_weights + fc_bias
        float* classifier_workspace = buffers->classifier_workspace;

        // Classifier input_dim = num_heads * channels (from spatial pooling of CA output)
        int classifier_input_dim = arch.num_heads * arch.channels;
        int max_classifier_size = max(classifier_input_dim, max(classifier_input_dim * num_classes, num_classes));
        int classifier_blocks = (max_classifier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_classifier_kernel<<<classifier_blocks, BLOCK_SIZE>>>(organism->classifier, classifier_input_dim, num_classes, seed + 777777, classifier_workspace);

        // Wire up training_mode->classifier to point to organism->classifier
        organism->training_mode->classifier = organism->classifier;

        // Wire up training_mode adam buffers to organism pools
        organism->training_mode->adam_m_perception = organism->adam_m_perception_pool;
        organism->training_mode->adam_v_perception = organism->adam_v_perception_pool;
        organism->training_mode->adam_m_interaction = organism->adam_m_interaction_pool;
        organism->training_mode->adam_v_interaction = organism->adam_v_interaction_pool;
        organism->training_mode->adam_m_value = organism->adam_m_value_pool;
        organism->training_mode->adam_v_value = organism->adam_v_value_pool;

        // Allocate adaptive curriculum
        organism->curriculum = buffers->curriculum;

        // Allocate voronoi occupancy histogram for curriculum tracking
        organism->voronoi_occupancy_histogram = buffers->voronoi_occupancy_histogram;

        // Allocate pool task accuracies buffer for curriculum tracking
        organism->pool_task_accuracies = buffers->pool_task_accuracies;

        // Store pre-loaded dataset arrays and set initial datasets
        organism->dataset_array = dataset_array;
        organism->current_dataset = dataset_array[0];
        organism->test_dataset_array = test_dataset_array;
        organism->current_test_dataset = test_dataset_array[0];

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 pre_dataset err=%d\n", (int)err);
            return;
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 dataset err=%d\n", (int)err);
            return;
        }

        BehavioralInitSlots behavioral_slots;
        behavioral_slots.agent_embedding_scale = derive_param_slot(organism->pool->entries[0].genome_hash, "chemotaxis_agent_embedding_scale");
        behavioral_slots.init_exploration = derive_param_slot(organism->pool->entries[0].genome_hash, "chemotaxis_init_exploration");
        behavioral_slots.init_sensitivity = derive_param_slot(organism->pool->entries[0].genome_hash, "chemotaxis_init_sensitivity");
        behavioral_slots.ctx_metabolic = derive_param_slot(organism->pool->entries[0].genome_hash, "init_context_metabolic");
        behavioral_slots.ctx_stress = derive_param_slot(organism->pool->entries[0].genome_hash, "init_context_stress");
        behavioral_slots.ctx_morphogen = derive_param_slot(organism->pool->entries[0].genome_hash, "init_context_morphogen");

        init_behavioral_state_kernel<<<(POOL_CAPACITY_MAX + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(organism->behavioral_agents,
            POOL_CAPACITY_MAX,
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
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 behavioral_state err=%d\n", (int)err);
            return;
        }

        int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;
        int total_weights = behavioral_dim * behavioral_dim;
        int embedding_blocks = (total_weights + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
        init_embedding_weights_kernel<<<embedding_blocks, BLOCK_SIZE>>>(buffers->behavioral_embedding_weights, behavioral_dim, seed + 1);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 embedding_weights err=%d\n", (int)err);
            return;
        }

        dim3 chem_grid((arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, (arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM);
        dim3 chem_block(WMMA_TILE_DIM, WMMA_TILE_DIM);
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
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 chemical_field err=%d\n", (int)err);
            return;
        }
        printf("V:p2_sync4_pre chem_field\n");
        err = cudaDeviceSynchronize();
        printf("V:p2_sync4_post err=%d\n", (int)err);

        set_chemical_sources_from_agents_kernel<<<1, POOL_CAPACITY_MAX>>>(
            organism->chemical_field->sources,
            organism->behavioral_agents,
            POOL_CAPACITY_MAX,
            arch.grid_size,
            primary_genome,
            organism->pool->entries[0].gradients,
            organism->chemical_field->concentration,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 chem_sources err=%d\n", (int)err);
            return;
        }
        printf("V:p2_sync5_pre chem_sources\n");
        err = cudaDeviceSynchronize();
        printf("V:p2_sync5_post err=%d\n", (int)err);

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
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 voronoi_state err=%d\n", (int)err);
            return;
        }

        store_chemical_snapshot_kernel<<<chem_grid, chem_block>>>(organism->chemical_field, field_size, (float)organism->generation, organism->pool->entries[0].genome_hash, primary_genome);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 chem_snapshot err=%d\n", (int)err);
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
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!E:init2 curriculum err=%d\n", (int)err);
            return;
        }

        // Initialize history buffers to zero (ring buffers with depth=2)
        int fitness_coherence_size = 2 * POOL_CAPACITY_MAX;
        int effective_rank_size = 2;
        int fc_blocks = (fitness_coherence_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        clear_buffer_kernel<<<fc_blocks, BLOCK_SIZE>>>(organism->fitness_history, fitness_coherence_size);
        err = cudaGetLastError();
        clear_buffer_kernel<<<fc_blocks, BLOCK_SIZE>>>(organism->coherence_history, fitness_coherence_size);
        err = cudaGetLastError();
        clear_buffer_kernel<<<1, BLOCK_SIZE>>>(organism->effective_rank_history, effective_rank_size);
        err = cudaGetLastError();

        printf("V:init2_complete param_map=%p training_mode=%p ca_state_pool=%p\n",
               (void*)organism->param_map,
               (void*)organism->training_mode, (void*)organism->ca_state_pool);
    }
}

__global__ void check_convergence_kernel(
    Organism* organism,
    int generation,
    bool* converged,
    float* workspace_genomes
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float fitness = organism->fitness_history[(generation % 2) * POOL_CAPACITY_MAX];
        float coherence = organism->coherence_history[(generation % 2) * POOL_CAPACITY_MAX];

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
            *converged = true;
        }
    }
}

__global__ void persistent_evolution_kernel(
    unsigned int seed,
    Dataset** dataset_array,
    Dataset** test_dataset_array,
    OrganismPreallocatedBuffers* buffers,
    AuditBuffer* audit
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // V: kernel entry - received parameters
    printf("V:persistent_entry seed=%u dataset_arr=%p test_arr=%p buffers=%p audit=%p\n",
           seed, (void*)dataset_array, (void*)test_dataset_array, (void*)buffers, (void*)audit);

    Organism* organism = buffers->organism;
    float* organism_workspace_genomes = buffers->organism_workspace_genomes;
    cudaError_t err;

    PRNGState rng;
    rng.s0 = seed * XORSHIFT_GOLDEN_RATIO_A;
    rng.s1 = seed * XORSHIFT_GOLDEN_RATIO_B;

    for (int i = 0; i < GENOME_SIZE; i++) {
        organism_workspace_genomes[i] = rng.next() * GENOME_RANGE_SCALE + GENOME_VALUE_MIN;
    }

    uint64_t organism_genome_hash = gpu_sha256(organism_workspace_genomes, GENOME_SIZE);

    int pool_capacity_slot = derive_param_slot(organism_genome_hash, "pool_capacity");
    float pool_capacity_norm = fmaxf(0.0f, fminf(1.0f, (organism_workspace_genomes[pool_capacity_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    int pool_capacity = POOL_CAPACITY_MIN + (int)(pool_capacity_norm * (POOL_CAPACITY_MAX - POOL_CAPACITY_MIN));
    init_organism_kernel<<<1, 1>>>(organism, dataset_array, test_dataset_array, pool_capacity, organism_workspace_genomes, buffers);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("!E:init1 err=%d\n", (int)err);
        return;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("!E:init1_sync err=%d\n", (int)err);
        return;
    }
    printf("V:init1 org=%p pool=%p\n", (void*)organism, (void*)organism->pool);

    init_organism_phase2_kernel<<<1, 1>>>(organism, dataset_array, test_dataset_array, seed, organism_workspace_genomes, buffers);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("!E:init2 err=%d\n", (int)err);
        return;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("!E:init2_sync err=%d\n", (int)err);
        return;
    }
    printf("V:init2 training_mode=%p dataset=%p test_dataset=%p\n",
           (void*)organism->training_mode, (void*)organism->current_dataset,
           (void*)organism->current_test_dataset);
    printf("V:init2_edges logits=%p telemetry=%p chem=%p conc=%p\n",
           (void*)organism->gradient_logits_pool,
           (void*)organism->telemetry,
           (void*)organism->chemical_field,
           organism->chemical_field ? (void*)organism->chemical_field->concentration : nullptr);

    int generation = 0;
    // V: vertex probe - exposes graph connectivity at persistent loop entry
    printf("V:persistent org=%p pool=%p cap=%d dataset=%p audit=%p\n",
           (void*)organism, (void*)organism->pool, organism->pool->capacity,
           (void*)organism->current_dataset, (void*)audit);
    while (true) {
        // Parallel reduction to compute cached_mean before lifecycle kernels read it
        int reduction_blocks = organism->reduction_num_blocks;
        int total_cells = organism->reduction_total_cells;
        reduce_concentration_mean_kernel<<<reduction_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            organism->chemical_field, total_cells, organism->reduction_workspace);
        cudaDeviceSynchronize();
        finalize_concentration_mean_kernel<<<1, 1>>>(
            organism->chemical_field, organism->reduction_workspace, reduction_blocks, total_cells);
        cudaDeviceSynchronize();

        int capacity = organism->pool->capacity;
        cudaError_t err;

        // ========== PHASE 1: Shared Resource Operations ==========
        // Per architecture.md: CDP depth = 1, launched from coordinator
        printf("V:P1_start gen=%d cap=%d\n", generation, capacity);

        aggregate_hardware_geometry_kernel<<<1, BLOCK_SIZE>>>(organism->trace_buffer, organism->hardware_geom);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P1_hw gen=%d\n", generation); return; }
        printf("V:P1_hw gen=%d\n", generation);

        int selection_blocks = (capacity + WARP_SIZE - 1) / WARP_SIZE;
        selection_kernel<<<selection_blocks, WARP_SIZE>>>(
            organism, organism->pool, organism->archive, organism->voronoi_cells,
            organism->num_voronoi_cells, &organism->archive_size,
            organism->behavioral_agents, generation, organism_workspace_genomes);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P1_sel gen=%d\n", generation); return; }
        printf("V:P1_sel gen=%d\n", generation);

        int behavioral_dim = organism->archive->hw_dim + organism->archive->task_dim + organism->archive->gen_dim;
        update_voronoi_density_kernel<<<(organism->num_voronoi_cells + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            organism->voronoi_cells,
            organism->archive,
            organism->archive_size,
            organism->num_voronoi_cells,
            behavioral_dim,
            organism_workspace_genomes,
            organism->pool->entries[0].gradients,
            organism->pool->entries[0].genome_hash,
            organism->pool->entries[0].fitness,
            organism->pool->entries[0].hunger,
            organism->chemical_field->concentration[0],
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy
        );
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P1_voronoi gen=%d\n", generation); return; }
        printf("V:P1_voronoi gen=%d\n", generation);

        dim3 component_grid((POOL_CAPACITY_MAX + (BLOCK_SIZE - 1)) / BLOCK_SIZE);
        dim3 component_block(BLOCK_SIZE);
        ArchitectureParams arch_p1 = get_arch_from_pool(organism->pool, 0);
        component_evolution_kernel<<<component_grid, component_block>>>(
            organism, organism->pool, organism->archive, organism->voronoi_cells,
            organism->num_voronoi_cells, &organism->archive_size, organism->chemical_field,
            organism->behavioral_agents, organism->fitness_history, organism->coherence_history,
            generation, arch_p1, organism_workspace_genomes);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P1_comp gen=%d\n", generation); return; }
        printf("V:P1_comp gen=%d\n", generation);

        compute_fitness_from_diresa_kernel<<<capacity, BLOCK_SIZE>>>(
            organism, organism->pool, GENOME_LATENT_DIM_MAX);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P1_fitness gen=%d\n", generation); return; }
        printf("V:P1_fitness gen=%d\n", generation);

        // Spawn wave (conditional on population density)
        printf("V:P1_A gen=%d\n", generation);
        int active = Atomics::load_int(organism->pool->active_count);
        printf("V:P1_B gen=%d active=%d\n", generation, active);
        float spawn_prob = SPAWN_RATE_MAX * expf(-active / (float)capacity);
        printf("V:P1_C gen=%d prob=%.6f threshold=%.6f\n", generation, spawn_prob, SPAWN_PROBABILITY_MIN_MIN);
        if (spawn_prob > SPAWN_PROBABILITY_MIN_MIN) {
            float* spawn_workspace = &organism_workspace_genomes[4 * GENOME_SIZE + 2 * capacity * GENOME_SIZE];
            printf("V:P1_D gen=%d entering spawn_wave_kernel\n", generation);
            spawn_wave_kernel<<<1, BLOCK_SIZE>>>(organism, organism->pool, spawn_prob, generation, spawn_workspace);
            cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess) { printf("!E:P1_spawn gen=%d err=%d\n", generation, (int)err); return; }
            printf("V:P1_spawn gen=%d\n", generation);
        } else {
            printf("V:P1_E gen=%d spawn_skipped\n", generation);
        }

        printf("V:P1_F gen=%d entering archive_driven_lifecycle\n", generation);
        int lifecycle_grid = (capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
        archive_driven_lifecycle_kernel<<<lifecycle_grid, BLOCK_SIZE>>>(
            organism, organism->pool, organism->archive, organism->archive_size,
            organism->voronoi_cells, organism->num_voronoi_cells,
            organism->behavioral_agents, HUNGER_THRESHOLD_MAX, generation,
            organism_workspace_genomes, organism->diresa_genome_weights);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P1_lifecycle gen=%d\n", generation); return; }
        printf("V:P1_lifecycle gen=%d\n", generation);

        // Memory update - compute thresholds then apply decay/pruning/consolidation
        memory_params_kernel<<<1, 1>>>(
            organism->chemical_field->history,
            organism->memory_params,
            generation,
            organism_workspace_genomes,
            organism->pool->entries[0].gradients,
            organism->pool->entries[0].genome_hash,
            organism->pool->entries[0].fitness,
            organism->pool->entries[0].hunger,
            organism->chemical_field->cached_mean,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P1_mem_params gen=%d\n", generation); return; }

        // Apply memory operations using computed thresholds
        TemporalTube* tubes = organism->chemical_field->history;
        int tube_count = organism->memory_params->tube_count;
        if (tube_count > 0) {
            int mem_blocks = (tube_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

            apply_decay_kernel<<<mem_blocks, BLOCK_SIZE>>>(
                tubes, organism->memory_params->flow_lenia_dt);
            err = cudaGetLastError();
            if (err != cudaSuccess) { printf("!E:P1_mem_decay gen=%d\n", generation); return; }

            prune_memories_kernel<<<mem_blocks, BLOCK_SIZE>>>(
                tubes, organism->memory_params->decay_threshold);
            err = cudaGetLastError();
            if (err != cudaSuccess) { printf("!E:P1_mem_prune gen=%d\n", generation); return; }

            int consol_threads = (tube_count < BLOCK_SIZE) ? tube_count : BLOCK_SIZE;
            consolidate_memories_kernel<<<1, consol_threads>>>(
                tubes, organism->memory_params->consolidation_threshold);
            err = cudaGetLastError();
            if (err != cudaSuccess) { printf("!E:P1_mem_consol gen=%d\n", generation); return; }

            // Compaction on telemetry intervals
            if (organism->memory_params->should_compact) {
                mark_valid_entries_kernel<<<mem_blocks, BLOCK_SIZE>>>(
                    tubes, organism->memory_compaction_valid_flags,
                    organism->memory_params->decay_threshold);
                cudaDeviceSynchronize();

                // Exclusive scan for write indices
                exclusive_scan_single_kernel<<<1, BLOCK_SIZE>>>(
                    organism->memory_compaction_valid_flags,
                    organism->memory_compaction_scan,
                    tubes->capacity);
                cudaDeviceSynchronize();

                compact_entries_kernel<<<mem_blocks, BLOCK_SIZE>>>(
                    tubes, organism->memory_compaction_valid_flags,
                    organism->memory_compaction_scan,
                    organism->memory_compaction_buffer, tube_count);
                cudaDeviceSynchronize();

                // Finalize compaction and copy entries back
                finalize_and_copy_compacted_kernel<<<mem_blocks, BLOCK_SIZE>>>(
                    tubes, organism->memory_compaction_valid_flags,
                    organism->memory_compaction_scan,
                    organism->memory_compaction_buffer, tube_count);
                err = cudaGetLastError();
                if (err != cudaSuccess) { printf("!E:P1_mem_compact gen=%d\n", generation); return; }
            }
        }

        dim3 field_grid((arch_p1.grid_size + 15) / 16, (arch_p1.grid_size + 15) / 16);
        dim3 field_block(WMMA_TILE_DIM, WMMA_TILE_DIM);
        diffusion_reaction_kernel<<<field_grid, field_block>>>(
            organism->chemical_field->concentration, organism->chemical_field->gradient_x,
            organism->chemical_field->gradient_y, organism->chemical_field->laplacian,
            organism->chemical_field->sources, arch_p1.grid_size, CHEMICAL_DIFFUSION_DT_MAX,
            organism_workspace_genomes, organism->pool->entries[0].gradients,
            organism->pool->entries[0].genome_hash,
            organism->pool->entries[0].fitness, organism->pool->entries[0].hunger,
            organism->chemical_field->cached_mean,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy);
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P1_diff gen=%d\n", generation); return; }

        store_chemical_snapshot_kernel<<<field_grid, field_block>>>(
            organism->chemical_field, arch_p1.grid_size * arch_p1.grid_size,
            (float)generation, organism->pool->entries[0].genome_hash, organism_workspace_genomes);
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P1_snap gen=%d\n", generation); return; }

        cudaDeviceSynchronize();  // BARRIER: Phase 1 complete
        printf("V:P1_done gen=%d\n", generation);

        // ========== PHASE 2: Per-Entry CA Processing ==========
        printf("V:P2_start gen=%d\n", generation);

        // Initialize per-entry CA from shared chemical field (stigmergy loop: chemical_field → per-entry CA)
        // This completes the bidirectional data flow: training→chemical_field→diffusion→per-entry CA
        dim3 init_field_grid((arch_p1.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM,
                             (arch_p1.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM);
        dim3 init_field_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

        // Get attractor field from temporal history for channel 15
        float* attractor_field = nullptr;
        if (organism->chemical_field->history != nullptr &&
            organism->chemical_field->history->count > 0) {
            // Use most recent temporal snapshot
            int recent_idx = (organism->chemical_field->history->head +
                              organism->chemical_field->history->count - 1) %
                             organism->chemical_field->history->capacity;
            attractor_field = organism->chemical_field->history->entries[recent_idx].data;
        }

        for (int entry_idx = 0; entry_idx < capacity; entry_idx++) {
            initialize_ca_from_field_kernel<<<init_field_grid, init_field_block>>>(
                organism->pool,
                // ChemicalField (channels 0-5)
                organism->chemical_field->concentration,
                organism->chemical_field->gradient_x,
                organism->chemical_field->gradient_y,
                organism->chemical_field->laplacian,
                organism->chemical_field->sources,
                organism->chemical_field->decay_factors,
                // RDField (channels 6-9)
                organism->resource_density,
                organism->fitness_landscape,
                organism->resource_gradient_x,
                organism->resource_gradient_y,
                // BehavioralField (channel 10)
                organism->behavioral_field_pool,
                // Temporal retrieval (channel 15)
                attractor_field,
                arch_p1.grid_size,
                entry_idx
            );
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P2_init_ca gen=%d err=%d\n", generation, (int)err); return; }
        cudaDeviceSynchronize();
        printf("V:P2_init_ca gen=%d\n", generation);

        neural_ca_update_kernel<<<capacity, WARP_SIZE>>>(
            organism, organism->chemical_field, organism->effective_rank_history,
            organism->buffers->fp32_ca_workspace, generation, organism->pool,
            organism->fitness_history, organism->coherence_history,
            organism_workspace_genomes, organism->trace_buffer, arch_p1.grid_size);
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P2_ca gen=%d err=%d\n", generation, (int)err); return; }
        cudaDeviceSynchronize();

        // Update shared chemical field from per-entry CA output (stigmergy loop: per-entry CA → chemical_field)
        // This completes the bidirectional data flow: per-entry CA→chemical_field→diffusion→next iteration
        for (int entry_idx = 0; entry_idx < capacity; entry_idx++) {
            update_field_from_ca_kernel<<<init_field_grid, init_field_block>>>(
                organism->pool,
                organism->chemical_field->concentration,
                arch_p1.grid_size,
                entry_idx
            );
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P2_update_field gen=%d err=%d\n", generation, (int)err); return; }

        cudaDeviceSynchronize();  // BARRIER: Phase 2 complete
        printf("V:P2_done gen=%d\n", generation);

        // ========== PHASE 3: Per-Entry Behavioral Processing ==========
        printf("V:P3_start gen=%d\n", generation);
        behavioral_update_kernel<<<capacity, WARP_SIZE>>>(
            organism, organism->behavioral_agents, organism->chemical_field,
            organism->chemical_field->history, generation, arch_p1, organism_workspace_genomes);
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("!E:P3_beh gen=%d err=%d\n", generation, (int)err); return; }

        cudaDeviceSynchronize();  // BARRIER: Phase 3 complete
        printf("V:P3_done gen=%d\n", generation);

        // Gradient-based learning pathway - every CHECKPOINT_INTERVAL generations
        // Produces telemetry (classification accuracy, gradient magnitudes) for DIRESA
        if (generation % CHECKPOINT_INTERVAL == 0) {
            printf("V:checkpoint gen=%d\n", generation);

            int lifecycle_blocks = capacity;
            hybrid_organism_lifecycle_kernel<<<lifecycle_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
                organism,
                organism->training_mode,
                organism->param_map,
                generation,
                organism_workspace_genomes,
                false,  // eval_only=false for training
                audit
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("!E:hybrid_train_launch gen=%d err=%d\n", generation, (int)err);
                return;
            }
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("!E:hybrid_train_sync gen=%d err=%d\n", generation, (int)err);
                return;
            }
            printf("V:hybrid_train gen=%d tm=%p tm->batch_size=%d logits=%p\n",
                   generation, (void*)organism->training_mode,
                   organism->training_mode ? organism->training_mode->batch_size : -1,
                   (void*)organism->gradient_logits_pool);

            // Test evaluation: run forward pass on test dataset to compute test_accuracy
            {
                Dataset* saved_dataset = organism->current_dataset;
                organism->current_dataset = organism->current_test_dataset;
                organism->training_mode->is_train_batch = false;

                hybrid_organism_lifecycle_kernel<<<lifecycle_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
                    organism,
                    organism->training_mode,
                    organism->param_map,
                    generation,
                    organism_workspace_genomes,
                    true,  // eval_only=true for test
                    nullptr  // no audit during test eval
                );
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    return;
                }
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    return;
                }

                // Restore train dataset
                organism->current_dataset = saved_dataset;
                organism->training_mode->is_train_batch = true;
            }

            // Audit buffer population - diagnose any null pointers blocking output
            if (!audit) {
                printf("[audit:gen%d] audit=null, allocated in main.cu via cudaHostAlloc\n", generation);
            } else if (!organism->training_mode) {
                printf("[audit:gen%d] training_mode=null, set from buffers->training_mode in init\n", generation);
            } else if (!organism->gradient_logits_pool) {
                printf("[audit:gen%d] gradient_logits_pool=null, allocated in main.cu buffers_host\n", generation);
            } else if (!organism->current_dataset) {
                printf("[audit:gen%d] current_dataset=null, set by curriculum selection\n", generation);
            } else if (!organism->current_dataset->descriptor) {
                printf("[audit:gen%d] dataset->descriptor=null, set by load_dataset_from_registry\n", generation);
            } else if (!organism->pool) {
                printf("[audit:gen%d] pool=null, created by init_pool_kernel\n", generation);
            } else if (!organism->telemetry) {
                printf("[audit:gen%d] telemetry=null, allocated in main.cu buffers_host\n", generation);
            } else {
                int num_classes = organism->current_dataset->descriptor->num_classes;
                int grid_size = organism->pool->entries[0].grid_size;
                float* ca_conc = organism->pool->entries[0].ca_state ?
                    organism->pool->entries[0].ca_state->ca_concentration : nullptr;
                printf("[audit:gen%d] calling populate_audit_buffer classes=%d grid=%d batch=%d\n",
                       generation, num_classes, grid_size, organism->training_mode->batch_size);
                populate_audit_buffer(
                    audit,
                    generation,
                    organism->gradient_logits_pool,
                    organism->training_mode->batch_labels,
                    organism->training_mode->batch_images,
                    organism->training_mode->batch_size,
                    num_classes,
                    ca_conc,
                    grid_size,
                    organism->telemetry->task_performance.train_accuracy,
                    organism->telemetry->task_performance.test_accuracy,
                    organism->telemetry,
                    organism->pool
                );
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
            }
        }

        generation++;
    }
}

#endif

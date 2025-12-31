
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
            printf("FATAL [%s] entry=%d: %s launch failed: %s\n", __func__, entry_idx, kernel_name, cudaGetErrorString(_err)); \
            return; \
        } \
    } while(0)

namespace cg = cooperative_groups;

struct CAParameterMap;
struct HybridTrainingMode;
struct ClassificationHead;
struct AdaptiveCurriculum;
template<int SECTION_SIZE> struct LocalOrganismState;

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
    ADTape* ad_tape;
    TapeEntry* ad_tape_entries_pool;
    float* ad_tape_values_pool;
    float* ad_tape_grads_pool;
    CAParameterMap* param_map;
    float* perception_activations_saved;
    float* interaction_activations_saved;
    float* pre_gelu_values_saved;
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
    int* pool_compaction_flags;
    int* pool_compaction_scan;
    int* pool_compaction_recursive_workspace;
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

    ADTape* ad_tape;
    CAParameterMap* param_map;

    float* perception_activations_saved;
    float* interaction_activations_saved;
    float* pre_gelu_values_saved;
    int current_activation_grid_size;  // Track current allocation size for dynamic reallocation

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
    int* memory_compaction_recursive_workspace;
    MemoryEntry* memory_compaction_buffer;

    int* pool_compaction_flags;
    int* pool_compaction_scan;
    int* pool_compaction_recursive_workspace;

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

    OrganismPreallocatedBuffers* buffers;

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
__global__ void memory_update_kernel(TemporalTube* tubes, float* fitness_history, int* valid_flags_workspace, int* scan_workspace, int* scan_recursive_workspace, MemoryEntry* temp_buffer, int generation, float* genome, float* gradients, uint64_t genome_hash, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance);
__global__ void selection_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, VoronoiCell* voronoi_cells, int num_cells, int* archive_size, BehavioralState* behavioral_agents, int generation, float* workspace_genomes);
__global__ void archive_driven_lifecycle_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, int archive_size, VoronoiCell* voronoi_cells, int num_cells, BehavioralState* behavioral_agents, float hunger_threshold, int generation, float* workspace_genomes, DIRESAWeights* diresa_genome_weights);
__global__ void spawn_wave_kernel(Organism* organism, ComponentPool* pool, float spawn_probability, int generation, float* workspace_genomes);
__global__ void culling_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, int archive_size, ChemicalField* chemical_field, ArchitectureParams arch, float fitness_threshold, float hunger_threshold);
__global__ void compute_fitness_from_diresa_kernel(float* latent_genome, int latent_dim, float* fitness_out, float task_exponent, float gen_exponent, float rank_exponent, float efficiency_exponent, float task_accuracy, float generalization_gap, float hardware_efficiency, float renyi_q);
__global__ void coherence_kernel(float* prediction_errors, float* coherence_out, int history_length);
__global__ void initialize_ca_from_field_kernel(ComponentPool* pool, float* chemical_concentration, int max_grid_size, int entry_idx);
__global__ void update_field_from_ca_kernel(ComponentPool* pool, float* chemical_concentration, int max_grid_size, int entry_idx);
__global__ void prepare_ca_fp16_kernel(ComponentPool* pool, int max_grid_size, ArchitectureParams arch, int entry_idx);
__global__ void multi_head_ca_tensor_kernel(ComponentPool* pool, int max_grid_size, ArchitectureParams arch, TraceBuffer* trace_buffer, int entry_idx);
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

__device__ __forceinline__ ArchitectureParams get_arch_from_pool(ComponentPool* pool, int idx = 0) {
    ArchitectureParams arch;
    arch.num_heads = pool->entries[idx].num_heads;
    arch.channels = pool->entries[idx].channels;
    arch.hidden_dim = pool->entries[idx].hidden_dim;
    arch.head_dim = pool->entries[idx].head_dim;
    arch.grid_size = pool->entries[idx].grid_size;
    return arch;
}

extern "C" __global__ void organism_lifecycle_kernel(
    Organism* organism,
    int generation,
    float* workspace_genomes
) {
    int entry_idx = blockIdx.x;
    ComponentPool* pool = organism->pool;

    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

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

    if (entry_idx == 0 && (generation % 5 == 0 || generation < 3)) {
        printf("[GEN %3d] Pool: %3d active, %3d spawned, %3d culled | Archive: %2d elites\n",
               generation,
               Atomics::load_int(pool->active_count),
               Atomics::load_int(pool->total_spawned),
               Atomics::load_int(pool->total_culled),
               organism->archive_size);
    }

    ArchitectureParams arch;
    arch.num_heads = entry->num_heads;
    arch.channels = entry->channels;
    arch.hidden_dim = entry->hidden_dim;
    arch.head_dim = entry->head_dim;
    arch.grid_size = entry->grid_size;

    dim3 component_grid((POOL_CAPACITY_MAX + (BLOCK_SIZE - 1)) / BLOCK_SIZE);
    dim3 component_block(BLOCK_SIZE);

    atomicAdd(&organism->lifecycle_phase_counts[0], 1);

    if (entry_idx == 0) {
        aggregate_hardware_geometry_kernel<<<1, BLOCK_SIZE>>>(organism->trace_buffer, organism->hardware_geom);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: aggregate_hardware_geometry launch failed: %s\n", cudaGetErrorString(err));
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
            printf("FATAL [organism_lifecycle]: selection launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        float* component_workspace_genomes = &workspace_genomes[0];
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle]: aggregate_hardware_geometry launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

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
    }

    uint64_t pool_genome_hash = entry->genome_hash;
    float ctx_metabolic = entry->fitness;
    float ctx_stress = entry->hunger;

    float sum_morphogen = 0.0f;
    int total_cells = arch.grid_size * arch.grid_size * arch.channels;
    for (int i = 0; i < total_cells; i++) {
        sum_morphogen += organism->chemical_field->concentration[i];
    }
    float ctx_morphogen = sum_morphogen / (float)total_cells;

    int spawn_rate_slot = derive_param_slot(pool_genome_hash, "lifecycle_spawn_rate");
    int spawn_min_slot = derive_param_slot(pool_genome_hash, "lifecycle_spawn_min");
    float spawn_rate = genome_to_param(primary_genome, entry->gradients, spawn_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, SPAWN_RATE_MIN, SPAWN_RATE_MAX);
    float spawn_min = genome_to_param(primary_genome, entry->gradients, spawn_min_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, SPAWN_PROBABILITY_MIN_MIN, SPAWN_PROBABILITY_MIN_MAX);

    if (entry_idx == 0) {

        printf("[DEBUG-TELEMETRY] gen=%d mod_10=%d mod_100=%d DETAILED=%d COMPREHENSIVE=%d\n",
               generation, generation % 10, generation % 100, TELEMETRY_DETAILED, TELEMETRY_COMPREHENSIVE);

        if (generation % TELEMETRY_DETAILED == 0) {
            printf("[DEBUG-ENTERING-DETAILED] gen=%d\n", generation);
            genome_complexity_probe_kernel<<<1, 1>>>(organism->pool, &organism->telemetry->genome_complexity);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [organism]: genome_complexity_probe_kernel launch failed: %s\n", cudaGetErrorString(err));
                return;
            }

            int active = Atomics::load_int(organism->pool->active_count);
            float spawn_prob = spawn_rate * expf(-active / (float)POOL_CAPACITY_MAX);
            printf("[GEN %3d] active=%d unique=%d entropy=%.2f archive=%d spawn_prob=%.3f\n",
                   generation, active, organism->telemetry->genome_complexity.unique_hashes,
                   organism->telemetry->genome_complexity.hash_entropy, organism->archive_size, spawn_prob);
        }

        if (generation % TELEMETRY_COMPREHENSIVE == 0) {
            printf("[DEBUG-ENTERING-COMPREHENSIVE] gen=%d\n", generation);
            printf("[DEBUG] Launching archive_topology_probe_kernel\n");
            archive_topology_probe_kernel<<<1, 1>>>((GPUElite*)organism->archive, organism->archive_size, organism->voronoi_cells, organism->num_voronoi_cells, &organism->telemetry->archive_topology);
            printf("[DEBUG] archive_topology_probe_kernel launched\n");
            printf("[DEBUG] Launching diresa_evolution_probe_kernel\n");
            diresa_evolution_probe_kernel<<<1, 1>>>((GPUElite*)organism->archive, organism->archive_size, &organism->telemetry->diresa_evolution);
            printf("[DEBUG] diresa_evolution_probe_kernel launched\n");

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
            printf("[DEBUG] Exiting TELEMETRY_COMPREHENSIVE block\n");
        }

        float spawn_prob = spawn_rate * expf(-Atomics::load_int(organism->pool->active_count) / (float)organism->pool->capacity);
        if (spawn_prob > spawn_min) {
            atomicAdd(&organism->lifecycle_phase_counts[1], 1);

            float* spawn_wave_workspace_genomes = &workspace_genomes[4 * GENOME_SIZE + 2 * organism->pool->capacity * GENOME_SIZE];

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
        float hunger_threshold = genome_to_param(primary_genome, entry->gradients, hunger_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, HUNGER_THRESHOLD_MIN, HUNGER_THRESHOLD_MAX);

        atomicAdd(&organism->lifecycle_phase_counts[2], 1);

        printf("[DEBUG] Launching archive_driven_lifecycle_kernel\n");
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
        printf("[DEBUG] archive_driven_lifecycle_kernel launched\n");

        atomicAdd(&organism->lifecycle_phase_counts[3], 1);

        cudaError_t pre_kern5_err = cudaGetLastError();
        if (pre_kern5_err != cudaSuccess) {
            printf("FATAL [PRE-KERN-5] Uncleared CUDA error blocks all subsequent kernels: %s\n", cudaGetErrorString(pre_kern5_err));
            return;
        }

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
            printf("FATAL [organism_lifecycle]: memory_update launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[KERN-5] memory_update_kernel LAUNCHED\n");

        float* workspace_genome_buffer = &workspace_genomes[GENOME_SIZE * 2];

        neural_ca_update_kernel<<<pool->capacity, 1>>>(
            organism,
            organism->chemical_field,
            organism->effective_rank_history,
            organism->fp32_ca_workspace,
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
            printf("FATAL [organism_lifecycle]: neural_ca_update launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[KERN-6] neural_ca_update_kernel LAUNCHED\n");

        if (generation % CHECKPOINT_INTERVAL == 0 && generation > 0) {
            int active = Atomics::load_int(organism->pool->active_count);
            float effective_rank = organism->effective_rank_history[generation % 2];

            printf("[GEN %3d] Topology: active=%d, eff_rank=%.2f, archives=%d\n",
                   generation, active, effective_rank, organism->archive_size);
        }

        behavioral_update_kernel<<<component_grid, component_block>>>(
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
            printf("FATAL [organism_lifecycle]: behavioral_update launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[KERN-7] behavioral_update_kernel LAUNCHED\n");

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
        printf("[KERN-8] update_fitness_landscape_kernel LAUNCHED\n");

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
        printf("[KERN-9] resource_flow_kernel LAUNCHED\n");

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
            printf("FATAL [organism_lifecycle]: lifecycle_transition launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[KERN-10a] lifecycle_transition_kernel<<<num_blocks=%d, BLOCK_SIZE=%d>>> err=%d\n", num_blocks, BLOCK_SIZE, (int)err);

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
            printf("FATAL [organism_lifecycle]: hierarchical_lifecycle launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[KERN-10] hierarchical_lifecycle_kernel<<<alive_blocks=%d, BLOCK_SIZE=%d>>> alive_count=%d err=%d\n", alive_blocks, BLOCK_SIZE, organism->pool->alive_indices_count, (int)err);
        printf("[ORGANISM-LIFECYCLE] All kernels launched - exiting tid=%d\n", threadIdx.x);

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
        printf("[COMPONENT-EVOLUTION-ENTRY] gen=%d capacity=%d\n", generation, pool->capacity);
    }
    if (tid >= pool->capacity || !pool->entries[tid].alive) {
        return;
    }

    if (tid == 0) {
        printf("[COMPONENT-EVOLUTION-A] gen=%d tid=0 about to reconstruct\n", generation);
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
        printf("[COMPONENT-EVOLUTION-B] gen=%d tid=0 reconstruct done\n", generation);
    }

    cudaError_t err;

    float current_task_accuracy = organism->telemetry->task_performance.accuracy;
    if (current_task_accuracy > 0.0f) {
        pool->entries[tid].task_accuracy = current_task_accuracy;
    }
    pool->entries[tid].generalization_gap = fabsf(organism->telemetry->task_performance.train_accuracy - organism->telemetry->task_performance.test_accuracy);

    if (tid == 0) {
        printf("[COMPONENT-EVOLUTION-C] gen=%d tid=0 about to write history\n", generation);
    }

    organism->fitness_history[(generation % 2) * POOL_CAPACITY_MAX + tid] = pool->entries[tid].task_accuracy;
    organism->coherence_history[(generation % 2) * POOL_CAPACITY_MAX + tid] = pool->entries[tid].coherence;

    if (tid == 0) {
        printf("[COMPONENT-EVOLUTION-D] gen=%d tid=0 history written\n", generation);
    }

    {
        float hardware_features_temp[HARDWARE_FEATURES_DIM];
        extract_hardware_features(organism->hardware_geom, hardware_features_temp);

        float hw_efficiency_sum = safe_epsilon(1.0f);
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
    }

    if (tid == 0) {
        float acc_sum = 0.0f, gap_sum = 0.0f, hw_sum = 0.0f, fit_sum = 0.0f;
        float acc_min = 1e9f, gap_min = 1e9f, hw_min = 1e9f, fit_min = 1e9f;
        float acc_max = -1e9f, gap_max = -1e9f, hw_max = -1e9f, fit_max = -1e9f;
        int count = 0;
        for (int i = 0; i < pool->capacity; i++) {
            if (pool->entries[i].alive) {
                count++;
                float acc = pool->entries[i].task_accuracy;
                float gap = pool->entries[i].generalization_gap;
                float hw = pool->entries[i].hardware_efficiency;
                float fit = pool->entries[i].fitness;
                acc_sum += acc; acc_min = fminf(acc_min, acc); acc_max = fmaxf(acc_max, acc);
                gap_sum += gap; gap_min = fminf(gap_min, gap); gap_max = fmaxf(gap_max, gap);
                hw_sum += hw; hw_min = fminf(hw_min, hw); hw_max = fmaxf(hw_max, hw);
                fit_sum += fit; fit_min = fminf(fit_min, fit); fit_max = fmaxf(fit_max, fit);
            }
        }
        if (count > 0) {
            printf("[DIRESA-FITNESS] n=%d acc[%.3f..%.3f]=%.3f gap[%.3f..%.3f]=%.3f hw[%.1f..%.1f]=%.1f fit[%.6f..%.6f]=%.6f\n",
                   count, acc_min, acc_max, acc_sum/count, gap_min, gap_max, gap_sum/count,
                   hw_min, hw_max, hw_sum/count, fit_min, fit_max, fit_sum/count);
        }
    }

    // Baldwin Effect: Reinforce genomes that show learning improvement
    if (generation > 0 && tid < pool->capacity && pool->entries[tid].alive) {
        float prev_task_accuracy = organism->fitness_history[((generation - 1) % 2) * POOL_CAPACITY_MAX + tid];
        float learning_success = current_task_accuracy - prev_task_accuracy;

        if (is_meaningful(learning_success, 1.0f)) {
            float baldwin_sensitivity = pool->entries[tid].baldwin_sensitivity;

            for (int g = 0; g < GENOME_SIZE; g++) {
                pool->entries[tid].gradients[g] += learning_success * baldwin_sensitivity * primary_genome[g];
                pool->entries[tid].gradients[g] = fmaxf(GENOME_VALUE_MIN, fminf(GENOME_VALUE_MAX, pool->entries[tid].gradients[g]));
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

    if (variance < 0.0f) {
        printf("FATAL [power_law_fitness]: variance=%f\n", variance);
        *fitness_out = 0.0f;
        return;
    }
    float effective_rank = sqrtf(variance) * latent_dim;

    float gen_gap_term = 1.0f - generalization_gap;
    if (gen_gap_term <= 0.0f || task_accuracy <= 0.0f || effective_rank <= 0.0f || hardware_efficiency <= 0.0f) {
        printf("FATAL [power_law_fitness]: gen_gap_term=%f task=%f rank=%f hw=%f\n",
               gen_gap_term, task_accuracy, effective_rank, hardware_efficiency);
        *fitness_out = 0.0f;
        return;
    }

    *fitness_out = powf(task_accuracy, task_exponent)
                 * powf(gen_gap_term, gen_exponent)
                 * powf(effective_rank, rank_exponent)
                 * powf(hardware_efficiency, efficiency_exponent);
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
        printf("FATAL [selection]: coherence[%d]=%f\n", entry_idx, entry->coherence);
        return;
    }

    insert_elite_kernel<<<1, 1>>>(
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

    // Collective Voronoi update: coordinator syncs all CDP insert_elite_kernel calls,
    // then launches update using entries[0]'s genome for collective parameter derivation
    if (entry_idx == 0) {
        cudaDeviceSynchronize();

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

                float local_morphogen = sample_neighborhood(
                    organism->chemical_field->concentration, i, pool->entries[i].grid_size);

                float ctx_metabolic = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_metabolic_slot,
                    pool->entries[i].fitness, pool->entries[i].hunger, local_morphogen,
                    organism->telemetry->genome_complexity.hash_entropy,
                    organism->telemetry->archive_topology.novelty_gradient,
                    organism->telemetry->diresa_evolution.behavioral_drift_rate,
                    organism->telemetry->task_performance.accuracy,
                    NORMALIZED_MIN, NORMALIZED_MAX);

                float ctx_stress = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_stress_slot,
                    pool->entries[i].fitness, pool->entries[i].hunger, local_morphogen,
                    organism->telemetry->genome_complexity.hash_entropy,
                    organism->telemetry->archive_topology.novelty_gradient,
                    organism->telemetry->diresa_evolution.behavioral_drift_rate,
                    organism->telemetry->task_performance.accuracy,
                    NORMALIZED_MIN, NORMALIZED_MAX);

                float ctx_morphogen = genome_to_param(temp_genome, pool->entries[i].gradients, ctx_morphogen_slot,
                    pool->entries[i].fitness, pool->entries[i].hunger, local_morphogen,
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

            float parent_morphogen = sample_neighborhood(
                organism->chemical_field->concentration, parent_idx, pool->entries[parent_idx].grid_size);

            float ctx_metabolic = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_metabolic_slot,
                pool->entries[parent_idx].fitness, pool->entries[parent_idx].hunger, parent_morphogen,
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                NORMALIZED_MIN, NORMALIZED_MAX);

            float ctx_stress = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_stress_slot,
                pool->entries[parent_idx].fitness, pool->entries[parent_idx].hunger, parent_morphogen,
                organism->telemetry->genome_complexity.hash_entropy,
                organism->telemetry->archive_topology.novelty_gradient,
                organism->telemetry->diresa_evolution.behavioral_drift_rate,
                organism->telemetry->task_performance.accuracy,
                NORMALIZED_MIN, NORMALIZED_MAX);

            float ctx_morphogen = genome_to_param(workspace_parent_genome, pool->entries[parent_idx].gradients, ctx_morphogen_slot,
                pool->entries[parent_idx].fitness, pool->entries[parent_idx].hunger, parent_morphogen,
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

    float sum_morphogen = 0.0f;
    int morph_total_cells = arch.grid_size * arch.grid_size * arch.channels;
    for (int i = 0; i < morph_total_cells; i++) {
        sum_morphogen += chemical_field->concentration[i];
    }
    float context_morphogen = sum_morphogen / (float)morph_total_cells;

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
    if (threadIdx.x != 0) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    if (entry_idx == 0) printf("[NEURAL-CA] ENTER entry_idx=%d\n", entry_idx);

    ArchitectureParams arch = get_arch_from_pool(pool, entry_idx);

    dim3 init_grid((max_grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM,
                   (max_grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM,
                   1);
    dim3 init_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

    if (entry_idx == 0) printf("[NEURAL-CA] 1. initialize_ca_from_field_kernel grid=(%d,%d,%d) block=(%d,%d)\n",
        init_grid.x, init_grid.y, init_grid.z, init_block.x, init_block.y);
    initialize_ca_from_field_kernel<<<init_grid, init_block>>>(
        pool,
        chemical_field->concentration,
        max_grid_size,
        entry_idx
    );
    if (entry_idx == 0) printf("[NEURAL-CA] 1a. launch call returned\n");
    CDP_LAUNCH_CHECK("initialize_ca_from_field_kernel");
    if (entry_idx == 0) printf("[NEURAL-CA] 1b. CDP_LAUNCH_CHECK passed\n");

    int max_cells = max_grid_size * max_grid_size;
    if (arch.channels <= 0 || arch.num_heads <= 0 || arch.head_dim <= 0) {
        printf("FATAL [neural_ca_update_kernel] entry=%d: invalid arch ch=%d heads=%d head_dim=%d\n",
               entry_idx, arch.channels, arch.num_heads, arch.head_dim);
        return;
    }
    dim3 prep_grid((max_cells * arch.channels + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 prep_block(BLOCK_SIZE);
    if (prep_grid.x == 0) {
        printf("FATAL [neural_ca_update_kernel] entry=%d: prep_grid.x=0 (max_cells=%d ch=%d)\n",
               entry_idx, max_cells, arch.channels);
        return;
    }

    if (entry_idx == 0) printf("[NEURAL-CA] 2. prepare_ca_fp16_kernel grid=(%d) block=(%d)\n", prep_grid.x, prep_block.x);
    prepare_ca_fp16_kernel<<<prep_grid, prep_block>>>(
        pool,
        max_grid_size,
        arch,
        entry_idx
    );
    CDP_LAUNCH_CHECK("prepare_ca_fp16_kernel");

    dim3 ca_grid(1, arch.num_heads, 1);
    dim3 ca_block(1);
    if (ca_grid.y == 0) {
        printf("FATAL [neural_ca_update_kernel] entry=%d: ca_grid.y=0 (num_heads=%d)\n",
               entry_idx, arch.num_heads);
        return;
    }

    if (entry_idx == 0) printf("[NEURAL-CA] 3. multi_head_ca_tensor_kernel grid=(1,%d,1) block=(1)\n", arch.num_heads);
    multi_head_ca_tensor_kernel<<<ca_grid, ca_block>>>(
        pool,
        max_grid_size,
        arch,
        trace_buffer,
        entry_idx
    );
    CDP_LAUNCH_CHECK("multi_head_ca_tensor_kernel");
    if (entry_idx == 0) printf("[NEURAL-CA] 4. after multi_head_ca\n");

    dim3 affinity_grid((max_cells + WARP_SIZE - 1) / WARP_SIZE, 1);
    dim3 affinity_block(WARP_SIZE);

    if (entry_idx == 0) printf("[NEURAL-CA] 5. reduce_affinity_kernel\n");
    reduce_affinity_kernel<<<affinity_grid, affinity_block>>>(
        pool,
        max_grid_size,
        entry_idx
    );
    CDP_LAUNCH_CHECK("reduce_affinity_kernel");

    dim3 flow_grid((max_grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM,
                   (max_grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM,
                   1);
    dim3 flow_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

    if (entry_idx == 0) printf("[NEURAL-CA] 6. compute_flow_field_kernel\n");
    compute_flow_field_kernel<<<flow_grid, flow_block>>>(
        pool,
        max_grid_size,
        entry_idx
    );
    CDP_LAUNCH_CHECK("compute_flow_field_kernel");

    int max_buffer_size = max_grid_size * max_grid_size * arch.channels;
    dim3 clear_grid((max_buffer_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 clear_block(BLOCK_SIZE);

    clear_reintegration_buffer_kernel<<<clear_grid, clear_block>>>(
        pool,
        max_buffer_size,
        entry_idx
    );
    CDP_LAUNCH_CHECK("clear_reintegration_buffer_kernel");

    dim3 reint_grid(max_grid_size, max_grid_size, 1);
    dim3 reint_block(BLOCK_SIZE);

    reintegration_redistribute_kernel<<<reint_grid, reint_block>>>(
        pool,
        max_grid_size,
        entry_idx
    );
    CDP_LAUNCH_CHECK("reintegration_redistribute_kernel");

    dim3 copy_grid((max_buffer_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 copy_block(BLOCK_SIZE);

    copy_reintegration_to_concentration_kernel<<<copy_grid, copy_block>>>(
        pool,
        max_buffer_size,
        entry_idx
    );
    CDP_LAUNCH_CHECK("copy_reintegration_to_concentration_kernel");

    update_field_from_ca_kernel<<<init_grid, init_block>>>(
        pool,
        chemical_field->concentration,
        max_grid_size,
        entry_idx
    );
    CDP_LAUNCH_CHECK("update_field_from_ca_kernel");

    dim3 rank_grid(1);
    dim3 rank_block(BLOCK_SIZE);

    compute_effective_rank_from_latent_kernel<<<rank_grid, rank_block>>>(
        pool,
        effective_rank_history,
        workspace_genomes,
        GENOME_LATENT_DIM_MAX,
        entry_idx
    );
    CDP_LAUNCH_CHECK("compute_effective_rank_from_latent_kernel");

    if (entry_idx == 0) printf("[NEURAL-CA] BEFORE cudaDeviceSynchronize entry_idx=%d\n", entry_idx);
    cudaDeviceSynchronize();
    if (entry_idx == 0) printf("[NEURAL-CA] AFTER cudaDeviceSynchronize entry_idx=%d\n", entry_idx);
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
    int entry_idx = blockIdx.x;
    ComponentPool* pool = organism->pool;

    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

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

    if (entry_idx == 0) {
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

        compute_behavioral_field_kernel<<<field_grid, field_block>>>(behavioral_field, agents, num_agents, arch.grid_size, primary_genome, entry->gradients,
            entry->genome_hash,
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
            entry->gradients,
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
    int* scan_recursive_workspace,
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
        printf("[MEMORY-UPDATE] ENTER gen=%d threads=%d tubes->count=%d\n", generation, blockDim.x * gridDim.x, tubes->count);
        int old_count = tubes->count;

        int consolidation_slot = derive_param_slot(genome_hash, "memory_consolidation_threshold");
        float consolidation_threshold = genome_to_param(genome, gradients, consolidation_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CONSOLIDATION_THRESHOLD_MIN, CONSOLIDATION_THRESHOLD_MAX);

        int flow_lenia_dt_slot = derive_param_slot(genome_hash, "flow_lenia_dt");
        float flow_lenia_dt = genome_to_param(genome, gradients, flow_lenia_dt_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, FLOW_LENIA_DT_MIN, FLOW_LENIA_DT_MAX);

        printf("[MEMORY-UPDATE] Launching apply_decay_kernel count=%d\n", tubes->count);
        apply_decay_kernel<<<(tubes->count + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
            tubes,
            flow_lenia_dt
        );
        printf("[MEMORY-UPDATE] apply_decay_kernel launched\n");

        int decay_threshold_slot = derive_param_slot(genome_hash, "memory_decay_threshold");
        float decay_threshold = genome_to_param(genome, gradients, decay_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, DECAY_THRESHOLD_MIN, DECAY_THRESHOLD_MAX);

        printf("[MEMORY-UPDATE] Launching prune_memories_kernel count=%d\n", tubes->count);
        prune_memories_kernel<<<(tubes->count + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
            tubes,
            decay_threshold
        );
        printf("[MEMORY-UPDATE] prune_memories_kernel launched\n");

        int post_prune_count = tubes->count;

        printf("[MEMORY-UPDATE] Launching consolidate_memories_kernel count=%d\n", tubes->count);
        consolidate_memories_kernel<<<1, min(BLOCK_SIZE, tubes->count)>>>(
            tubes,
            consolidation_threshold
        );
        printf("[MEMORY-UPDATE] consolidate_memories_kernel launched\n");

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

            printf("[MEMORY-UPDATE] Launching prune_and_compact_memories_kernel count=%d\n", tubes->count);
            prune_and_compact_memories_kernel<<<(tubes->count + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(
                tubes,
                valid_flags_workspace,
                scan_workspace,
                scan_recursive_workspace,
                temp_buffer,
                decay_threshold
            );
            printf("[MEMORY-UPDATE] prune_and_compact_memories_kernel launched\n");

            printf("[GEN %3d] Memory compaction: %d->%d entries (removed %d low-value memories)\n",
                   generation, pre_compact_count, tubes->count, pre_compact_count - tubes->count);
        }

        printf("[MEMORY-UPDATE] EXIT gen=%d\n", generation);
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
    int pool_capacity,
    float* workspace_genomes,
    OrganismPreallocatedBuffers* buffers
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
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
        organism->pool->capacity = pool_capacity;
        *((int*)&organism->pool->active_count) = initial_pool_size;
        *((int*)&organism->pool->total_spawned) = 0;
        *((int*)&organism->pool->total_culled) = 0;

        organism->pool_compaction_flags = buffers->pool_compaction_flags;
        organism->pool_compaction_scan = buffers->pool_compaction_scan;
        organism->pool_compaction_recursive_workspace = buffers->pool_compaction_recursive_workspace;

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
        init_pool_kernel<<<pool_blocks, BLOCK_SIZE>>>(organism->pool, pool_capacity, delta_indices_buffer, delta_values_buffer, gradients_buffer);
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("[organism] init_pool launch_err=%s\n", cudaGetErrorString(err)); return; }
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
    rng.s0 = genome_hash ^ (idx * 0x9e3779b97f4a7c15ULL);
    rng.s1 = (genome_hash >> 32) ^ (idx * 0xbf58476d1ce4e5b9ULL);

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
    unsigned int seed,
    float* workspace_genomes,
    OrganismPreallocatedBuffers* buffers
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
            printf("FATAL [init_organism_phase2]: wire_behavioral_agents_kernel failed: %s\n", cudaGetErrorString(err));
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
            printf("FATAL [init_organism_phase2]: init_voronoi_pointers_kernel launch failed: %s\n", cudaGetErrorString(err));
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
            printf("FATAL [organism]: init_voronoi_cells_kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[DEBUG] About to call derive_param_slot\n");
        int default_decay_rate_slot = derive_param_slot(organism->pool->entries[0].genome_hash, "memory_default_decay_rate");
        printf("[DEBUG] derive_param_slot returned %d\n", default_decay_rate_slot);
        float default_decay_rate_norm = (primary_genome[default_decay_rate_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float default_decay_rate = DEFAULT_DECAY_RATE_MIN + default_decay_rate_norm * (DEFAULT_DECAY_RATE_MAX - DEFAULT_DECAY_RATE_MIN);

        ArchitectureParams arch = get_arch_from_pool(organism->pool, 0);
        int field_size = arch.grid_size * arch.grid_size;
        printf("[organism] arch from pool[0]: heads=%d channels=%d hidden=%d head_dim=%d grid=%d field_size=%d\n", arch.num_heads, arch.channels, arch.hidden_dim, arch.head_dim, arch.grid_size, field_size);

        organism->telemetry = buffers->telemetry;
        organism->telemetry->valid = false;
        organism->telemetry->generation = 0;

        // Get actual device heap limit
        size_t heap_limit;
        err = cudaDeviceGetLimit(&heap_limit, cudaLimitMallocHeapSize);
        if (err != cudaSuccess) { printf("FATAL [init_organism_phase2]: cudaDeviceGetLimit failed: %s\n", cudaGetErrorString(err)); return; }
        organism->telemetry->memory_allocation.device_heap_limit = heap_limit;
        organism->telemetry->memory_allocation.device_heap_allocated = 0;

        // Allocate CA workspace
        int weight_count = arch.num_heads * arch.channels * arch.hidden_dim;
        organism->fp32_ca_workspace = buffers->fp32_ca_workspace;
        organism->fp16_ca_workspace = buffers->fp16_ca_workspace;

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
            printf("FATAL [organism]: init_tube_kernel for chemical_field->history failed: %s\n", cudaGetErrorString(err));
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
            printf("FATAL [organism]: wire_tube_data_kernel for chemical_field->history failed: %s\n", cudaGetErrorString(err));
            return;
        }

        int perception_size = arch.num_heads * arch.channels * arch.head_dim;
        int interaction_size = arch.num_heads * arch.head_dim * arch.head_dim;
        int value_size = arch.num_heads * arch.head_dim * arch.channels;
        int total_weights_size = perception_size + interaction_size + value_size;

        dim3 weight_init_grid((total_weights_size + BLOCK_SIZE - 1) / BLOCK_SIZE, pool_capacity);
        dim3 weight_init_block(BLOCK_SIZE);

        init_organism_ca_weights_kernel<<<weight_init_grid, weight_init_block>>>(
            organism->pool,
            arch
        );

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

            entry_ca_state->perception_weights = buffers->all_ca_weights;
            entry_ca_state->interaction_weights = buffers->all_ca_weights + perception_size;
            entry_ca_state->value_weights = buffers->all_ca_weights + perception_size + interaction_size;

            int fp32_stride = CA_FIELD_SIZE * (NUM_HEADS_MAX + 1) * HEAD_DIM_MAX;
            int fp16_stride = CA_FIELD_SIZE * (CHANNELS_MAX + HEAD_DIM_MAX);
            entry_ca_state->fp32_workspace = buffers->fp32_ca_workspace + entry_idx * fp32_stride;
            entry_ca_state->fp16_workspace = buffers->fp16_ca_workspace + entry_idx * fp16_stride;

            entry->ca_state = entry_ca_state;
        }

        float* all_chem_fields = buffers->all_chem_fields;
        organism->chemical_field->concentration = all_chem_fields;
        organism->chemical_field->gradient_x = all_chem_fields + CA_FIELD_SIZE;
        organism->chemical_field->gradient_y = all_chem_fields + CA_FIELD_SIZE * 2;
        organism->chemical_field->laplacian = all_chem_fields + CA_FIELD_SIZE * 3;
        organism->chemical_field->sources = all_chem_fields + CA_FIELD_SIZE * 4;
        organism->chemical_field->decay_factors = all_chem_fields + CA_FIELD_SIZE * 5;

        organism->fitness_history = buffers->fitness_history;
        organism->effective_rank_history = buffers->effective_rank_history;
        organism->coherence_history = buffers->coherence_history;

        float* all_rd_fields = buffers->all_rd_fields;
        organism->resource_density = all_rd_fields;
        organism->resource_next = all_rd_fields + CA_FIELD_SIZE;
        organism->fitness_landscape = all_rd_fields + CA_FIELD_SIZE * 2;
        organism->rd_u_field = all_rd_fields + CA_FIELD_SIZE * 3;
        organism->rd_v_field = all_rd_fields + CA_FIELD_SIZE * 4;
        organism->rd_u_next = all_rd_fields + CA_FIELD_SIZE * 5;
        organism->rd_v_next = all_rd_fields + CA_FIELD_SIZE * 6;

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
            printf("[organism] init_diresa launch_err=%s blocks=%d\n", cudaGetErrorString(err), num_replicas);
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
        printf("[DEBUG-POST-COMPACTION] Completed memory_compaction_buffer allocation\n");

        organism->fitness_rank_pool = buffers->fitness_rank_pool;
        organism->fitness_coherence_pool = buffers->fitness_coherence_pool;
        organism->fitness_history = buffers->fitness_history;
        organism->coherence_history = buffers->coherence_history;
        organism->effective_rank_history = buffers->effective_rank_history;

        organism->ad_tape = buffers->ad_tape;
        organism->ad_tape_entries_pool = buffers->ad_tape_entries_pool;
        organism->ad_tape_values_pool = buffers->ad_tape_values_pool;
        organism->ad_tape_grads_pool = buffers->ad_tape_grads_pool;
        init_ad_tape_kernel<<<1, 1>>>(organism->ad_tape, organism->ad_tape_entries_pool, organism->ad_tape_values_pool, organism->ad_tape_grads_pool, MAX_TAPE_SIZE, MAX_TAPE_VALUES);

        // Allocate CA parameter map
        organism->param_map = buffers->param_map;
        init_ca_param_map_kernel<<<1, 1>>>(organism->param_map, arch);

        // Allocate CA activation buffers for backward pass (implicit differentiation)
        int act_batch = 1;
        int act_size = act_batch * arch.num_heads * arch.grid_size * arch.grid_size * arch.hidden_dim;
        organism->perception_activations_saved = buffers->perception_activations_saved;
        organism->interaction_activations_saved = buffers->interaction_activations_saved;
        organism->pre_gelu_values_saved = buffers->pre_gelu_values_saved;
        organism->current_activation_grid_size = arch.grid_size;  // Initialize grid size tracker

        // Allocate lifecycle phase tracking
        organism->lifecycle_phase_counts = buffers->lifecycle_phase_counts;

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

        int max_classifier_size = max(dims.hw_dim, max(dims.hw_dim * num_classes, num_classes));
        int classifier_blocks = (max_classifier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_classifier_kernel<<<classifier_blocks, BLOCK_SIZE>>>(organism->classifier, dims.hw_dim, num_classes, seed + 777777, classifier_workspace);

        // Wire up training_mode->classifier to point to organism->classifier
        organism->training_mode->classifier = organism->classifier;
        printf("[MEM OK] organism->training_mode->classifier wired to %p\n", organism->classifier);

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

        printf("[MEM] All buffers wired from pre-allocated pool\n");

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
        printf("[DEVICE-DBG] init_behavioral_state_kernel launch returned, calling sync...\n");

        printf("[DEVICE-DBG] cudaDeviceSynchronize returned: %d (%s)\n", (int)err, cudaGetErrorString(err));
        if (err != cudaSuccess) {
            printf("[ERROR] init_behavioral_state_kernel failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[DEVICE] init_behavioral_state_kernel completed\n");

        int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;
        int total_weights = behavioral_dim * behavioral_dim;
        int embedding_blocks = (total_weights + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
        init_embedding_weights_kernel<<<embedding_blocks, BLOCK_SIZE>>>(buffers->behavioral_embedding_weights, behavioral_dim, seed + 1);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] init_embedding_weights_kernel launch failed: %s\n", cudaGetErrorString(err));
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
        printf("[DEVICE] init_chemical_field_kernel launch err=%d\n", (int)err);
        if (err != cudaSuccess) {
            printf("[ERROR] init_chemical_field_kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
        err = cudaDeviceSynchronize();
        printf("[SYNC1] init_chemical_field_kernel sync err=%d\n", (int)err);

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
        printf("[DEVICE] set_chemical_sources launch err=%d\n", (int)err);
        if (err != cudaSuccess) {
            printf("[ERROR] set_chemical_sources launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
        err = cudaDeviceSynchronize();
        printf("[SYNC2] set_chemical_sources sync err=%d\n", (int)err);

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
        printf("[DEVICE] diffusion_reaction launch err=%d\n", (int)err);
        if (err != cudaSuccess) {
            printf("[ERROR] diffusion_reaction launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        store_chemical_snapshot_kernel<<<chem_grid, chem_block>>>(organism->chemical_field, field_size, (float)organism->generation, organism->pool->entries[0].genome_hash, primary_genome);
        err = cudaGetLastError();
        printf("[DEVICE] store_chemical_snapshot launch err=%d\n", (int)err);
        if (err != cudaSuccess) {
            printf("[ERROR] store_chemical_snapshot launch failed: %s\n", cudaGetErrorString(err));
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
        printf("[DEVICE] init_curriculum launch err=%d\n", (int)err);
        if (err != cudaSuccess) {
            printf("[ERROR] init_curriculum launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        // Initialize history buffers to zero (ring buffers with depth=2)
        int fitness_coherence_size = 2 * POOL_CAPACITY_MAX;
        int effective_rank_size = 2;
        int fc_blocks = (fitness_coherence_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        clear_buffer_kernel<<<fc_blocks, BLOCK_SIZE>>>(organism->fitness_history, fitness_coherence_size);
        err = cudaGetLastError();
        printf("[DEVICE] clear fitness_history launch err=%d\n", (int)err);
        clear_buffer_kernel<<<fc_blocks, BLOCK_SIZE>>>(organism->coherence_history, fitness_coherence_size);
        err = cudaGetLastError();
        printf("[DEVICE] clear coherence_history launch err=%d\n", (int)err);
        clear_buffer_kernel<<<1, BLOCK_SIZE>>>(organism->effective_rank_history, effective_rank_size);
        err = cudaGetLastError();
        printf("[DEVICE] clear effective_rank_history<<<blocks=1, BLOCK_SIZE=%d>>> err=%d\n", BLOCK_SIZE, (int)err);
        printf("[RULED-OUT] cudaDeviceSynchronize added here previously - did NOT fix hang - just waits forever for hung child\n");

        printf("[INIT-PHASE2] About to exit if-block tid=%d\n", threadIdx.x);
    }
    printf("[INIT-PHASE2] EXIT tid=%d bid=%d\n", threadIdx.x, blockIdx.x);
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
            printf("[CONVERGED] Gen %d: fitness=%.3f (>%.3f), coherence=%.3f (>%.3f)\n",
                   generation, fitness, fitness_threshold, coherence, coherence_threshold);
            *converged = true;
        }
    }
}

__global__ void persistent_evolution_kernel(
    unsigned int seed,
    Dataset** dataset_array,
    OrganismPreallocatedBuffers* buffers,
    AuditBuffer* audit
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    printf("[DEBUG-PERSISTENT-START] Entering persistent_evolution_kernel\n");

    PRNGState rng;
    rng.s0 = seed * 0x9e3779b97f4a7c15ULL;
    rng.s1 = seed * 0xbf58476d1ce4e5b9ULL;

    float organism_seed_genome[GENOME_SIZE];
    for (int i = 0; i < GENOME_SIZE; i++) {
        organism_seed_genome[i] = rng.next() * GENOME_RANGE_SCALE + GENOME_VALUE_MIN;
    }

    uint64_t organism_genome_hash = gpu_sha256(organism_seed_genome, GENOME_SIZE);

    int pool_capacity_slot = derive_param_slot(organism_genome_hash, "pool_capacity");
    float pool_capacity_norm = fmaxf(0.0f, fminf(1.0f, (organism_seed_genome[pool_capacity_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE));
    int pool_capacity = POOL_CAPACITY_MIN + (int)(pool_capacity_norm * (POOL_CAPACITY_MAX - POOL_CAPACITY_MIN));

    printf("[persistent_evolution] organism seed genome derived pool_capacity=%d\n", pool_capacity);

    Organism* organism = buffers->organism;
    float* organism_workspace_genomes = buffers->organism_workspace_genomes;
    cudaError_t err;

    for (int i = 0; i < GENOME_SIZE; i++) {
        organism_workspace_genomes[i] = organism_seed_genome[i];
    }

    printf("[DEBUG-PERSISTENT] cudaMalloc workspace done\n");

    printf("[DEBUG-PERSISTENT] Launching init_organism_kernel\n");
    init_organism_kernel<<<1, 1>>>(organism, dataset_array, pool_capacity, organism_workspace_genomes, buffers);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FATAL [init_organism_kernel]: launch failed: %d\n", (int)err);
        return;
    }
    printf("[DEBUG-PERSISTENT] init_organism_kernel launched, err=%d\n", (int)err);

    init_organism_phase2_kernel<<<1, 1>>>(organism, dataset_array, seed, organism_workspace_genomes, buffers);
    err = cudaGetLastError();
    printf("[DEBUG-PERSISTENT] init_organism_phase2_kernel launch returned err=%d\n", (int)err);
    if (err != cudaSuccess) {
        printf("FATAL [init_organism_phase2_kernel]: launch failed: %d\n", (int)err);
        return;
    }
    printf("[DEBUG-PERSISTENT] About to sync after init_organism_phase2_kernel\n");
    err = cudaDeviceSynchronize();
    printf("[DEBUG-PERSISTENT] cudaDeviceSynchronize after phase2 returned err=%d\n", (int)err);
    if (err != cudaSuccess) {
        printf("FATAL [init_organism_phase2_kernel]: sync failed: %d\n", (int)err);
        return;
    }

    int generation = 0;
    while (true) {
        int lifecycle_blocks = organism->pool->capacity;
        printf("[PERSISTENT-LOOP] gen=%d organism_lifecycle_kernel<<<%d,1>>>\n", generation, lifecycle_blocks);
        organism_lifecycle_kernel<<<lifecycle_blocks, 1>>>(organism, generation, organism_workspace_genomes);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle_kernel]: launch failed at gen %d: error=%d %s\n",
                   generation, (int)err, cudaGetErrorString(err));
            return;
        }
        printf("[PERSISTENT-LOOP] gen=%d BEFORE cudaDeviceSynchronize (after organism_lifecycle)\n", generation);
        err = cudaDeviceSynchronize();
        printf("[PERSISTENT-LOOP] gen=%d AFTER cudaDeviceSynchronize (after organism_lifecycle) err=%d\n", generation, (int)err);
        if (err != cudaSuccess) {
            printf("FATAL [organism_lifecycle_kernel]: sync failed at gen %d: error=%d %s\n",
                   generation, (int)err, cudaGetErrorString(err));
            return;
        }

        // Gradient-based learning pathway - every CHECKPOINT_INTERVAL generations
        // Produces telemetry (classification accuracy, gradient magnitudes) for DIRESA
        printf("[PERSISTENT-LOOP] gen=%d CHECKPOINT_INTERVAL=%d mod=%d\n", generation, CHECKPOINT_INTERVAL, generation % CHECKPOINT_INTERVAL);
        if (generation % CHECKPOINT_INTERVAL == 0) {
            printf("[PERSISTENT-LOOP] gen=%d CALLING hybrid_organism_lifecycle_kernel<<<%d,1>>>\n", generation, lifecycle_blocks);
            hybrid_organism_lifecycle_kernel<<<lifecycle_blocks, 1>>>(
                organism,
                organism->training_mode,
                organism->param_map,
                generation,
                organism_workspace_genomes
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_organism_lifecycle_kernel]: launch failed at gen %d: error=%d\n",
                       generation, (int)err);
                return;
            }
            printf("[PERSISTENT-LOOP] gen=%d BEFORE cudaDeviceSynchronize (after hybrid_organism_lifecycle)\n", generation);
            err = cudaDeviceSynchronize();
            printf("[PERSISTENT-LOOP] gen=%d AFTER cudaDeviceSynchronize (after hybrid_organism_lifecycle) err=%d\n", generation, (int)err);
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_organism_lifecycle_kernel]: sync failed at gen %d: error=%d %s\n",
                       generation, (int)err, cudaGetErrorString(err));
                return;
            }

            if (audit && organism->training_mode && organism->gradient_logits_pool) {
                int num_classes = organism->current_dataset->descriptor->num_classes;
                int grid_size = organism->pool->entries[0].grid_size;
                float* ca_conc = organism->pool->entries[0].ca_state ?
                    organism->pool->entries[0].ca_state->ca_concentration : nullptr;
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
                    organism->telemetry->task_performance.test_accuracy
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

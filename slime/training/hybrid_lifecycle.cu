
#ifndef HYBRID_LIFECYCLE_CU
#define HYBRID_LIFECYCLE_CU

#include "../config/config.cu"
#include "../training/training_types.cu"
#include "../data/dataset_loader.cu"
#include "../core/chemotaxis.cu"
#include "../training/autodiff_integration.cu"
#include "../training/gradient_fitness.cu"
#include "../training/classification.cu"
#include "../training/optimizer.cu"
#include "../utils/genome_params.cuh"
#include <cuda_runtime.h>

struct Organism;
struct ComponentPool;
struct GPUElite;
struct VoronoiCell;
struct ChemicalField;
struct BehavioralState;
struct TemporalTube;
struct MultiHeadCAState;

extern "C" __global__ void component_evolution_kernel(Organism*, ComponentPool*, GPUElite*, VoronoiCell*, int, int*, ChemicalField*, BehavioralState*, float*, float*, int, ArchitectureParams, float*);
extern "C" __global__ void neural_ca_update_kernel(Organism*, MultiHeadCAState*, ChemicalField*, float*, float*, int, ComponentPool*, float*, float*, float*);
extern "C" __global__ void behavioral_update_kernel(Organism*, BehavioralState*, ChemicalField*, TemporalTube*, int, ArchitectureParams, float*);
__global__ void memory_update_kernel(TemporalTube*, float*, int*, int*, MemoryEntry*, int, float*, float*, uint64_t, float, float, float, float, float, float, float);

__global__ void zero_scalar_kernel(float* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *ptr = 0.0f;
    }
}

extern "C" __global__ void hybrid_organism_lifecycle_kernel(
    Organism* organism,
    HybridTrainingMode* training_mode,
    CAParameterMap* param_map,
    int generation,
    float* workspace_genomes
) {
    printf("[HYBRID] Kernel ENTERED: tid=%d bid=%d gen=%d\n", threadIdx.x, blockIdx.x, generation);

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    printf("[HYBRID] Kernel STARTED on device (gen=%d)\n", generation);

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

    int num_classes_slot = derive_param_slot(organism->pool->entries[0].genome_hash, "num_classes");
    int num_classes = NUM_CLASSES_MIN +
        ((primary_genome[num_classes_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE) *
        (NUM_CLASSES_MAX - NUM_CLASSES_MIN);

    BehavioralDimensions dims;
    dims.derive_from_genome(organism->pool->entries[0].genome_hash, primary_genome);
    int behavioral_dim = dims.total();

    ArchitectureParams arch;
    arch.num_heads = organism->pool->entries[0].num_heads;
    arch.channels = organism->pool->entries[0].channels;
    arch.hidden_dim = organism->pool->entries[0].hidden_dim;
    arch.head_dim = organism->pool->entries[0].head_dim;
    arch.grid_size = organism->pool->entries[0].grid_size;

    dim3 component_grid((MAX_COMPONENTS + (BLOCK_SIZE - 1)) / BLOCK_SIZE);
    dim3 component_block(BLOCK_SIZE);

    dim3 ca_grid(arch.grid_size / WMMA_TILE_DIM, arch.num_heads, 1);
    dim3 ca_block(WMMA_TILE_DIM, WMMA_TILE_DIM, 1);

    dim3 field_grid((arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, (arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM);
    dim3 field_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

    cudaError_t err;

    if (training_mode->use_gradients) {

        // Sample batch from current pre-loaded dataset (switches per curriculum)
        sample_batch_kernel<<<training_mode->batch_size, BLOCK_SIZE>>>(
            organism->current_dataset,
            training_mode,
            training_mode->batch_size,
            generation * training_mode->batch_size,
            arch.grid_size
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: sample_batch launch failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }

        if (organism == nullptr) {
            printf("FATAL [hybrid_lifecycle]: organism is NULL\n");
            asm("trap;");
        }
        if (organism->ad_tape == nullptr) {
            printf("FATAL [hybrid_lifecycle]: ad_tape is NULL\n");
            asm("trap;");
        }
        if (organism->ca_state == nullptr) {
            printf("FATAL [hybrid_lifecycle]: ca_state is NULL\n");
            asm("trap;");
        }
        if (training_mode == nullptr) {
            printf("FATAL [hybrid_lifecycle]: training_mode is NULL\n");
            asm("trap;");
        }
        if (param_map == nullptr) {
            printf("FATAL [hybrid_lifecycle]: param_map is NULL\n");
            asm("trap;");
        }

        printf("[HYBRID-DBG1] About to launch reset_tape_kernel: blocks=%d, threads=%d, tape=%p\n",
               (VALUE_CAPACITY + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, organism->ad_tape);
        printf("[HYBRID-DBG2] tape->entries=%p tape->value_buffer=%p tape->grad_buffer=%p\n",
               organism->ad_tape->entries, organism->ad_tape->value_buffer, organism->ad_tape->grad_buffer);
        printf("[HYBRID-DBG3] tape->capacity=%d tape->value_capacity=%d\n",
               organism->ad_tape->capacity, organism->ad_tape->value_capacity);

        reset_tape_kernel<<<(VALUE_CAPACITY + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(organism->ad_tape);

        err = cudaGetLastError();
        printf("[HYBRID-DBG4] reset_tape_kernel launch returned, err=%d\n", (int)err);
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: reset_tape launch failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }
        printf("[HYBRID-DBG5] reset_tape_kernel launched (async, no sync needed)\n");

        if (training_mode->batch_images != nullptr) {

            dim3 sample_grid((arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM, (arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM, 1);
            dim3 sample_block(WMMA_TILE_DIM, WMMA_TILE_DIM, 1);

            inject_sample_to_ca_kernel<<<sample_grid, sample_block>>>(
                training_mode->batch_images,
                organism->ca_state->ca_concentration,
                1,
                arch.channels,
                arch.grid_size
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: Sample injection failed: %s\n", cudaGetErrorString(err));
                asm("trap;");
            }
        } else {

            initialize_ca_from_field_kernel<<<field_grid, field_block>>>(
                organism->ca_state->ca_concentration,
                organism->chemical_field->concentration,
                arch.grid_size
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: initialize_ca_from_field launch failed: %s\n", cudaGetErrorString(err));
                asm("trap;");
            }
        }

        multi_head_ca_with_tape_kernel<<<ca_grid, ca_block>>>(
            organism->ca_state->ca_concentration,
            organism->ca_state,
            organism->ca_state->ca_output,
            organism->ad_tape,
            param_map,
            1,
            arch.grid_size,
            arch
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: multi_head_ca_with_tape launch failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }

        if (training_mode->batch_images != nullptr) {

            float* features = organism->gradient_features_pool;

            spatial_pooling_kernel<<<1, BLOCK_SIZE>>>(
                organism->ca_state->ca_output,
                training_mode->classifier->pooling_weights,
                features,
                1,
                arch.grid_size,
                arch.channels
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: spatial_pooling launch failed: %s\n", cudaGetErrorString(err));
                asm("trap;");
            }

            float* logits = organism->gradient_logits_pool;

            classification_head_kernel<<<1, num_classes>>>(
                features,
                training_mode->classifier->fc_weights,
                training_mode->classifier->fc_bias,
                logits,
                1,
                arch.channels,
                num_classes
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: classification_head launch failed: %s\n", cudaGetErrorString(err));
                asm("trap;");
            }

            float* loss_out = organism->gradient_loss_pool;
            float* logit_grads = organism->gradient_logit_grads_pool;

            zero_scalar_kernel<<<1, 1>>>(loss_out);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: zero_scalar launch failed: %s\n", cudaGetErrorString(err));
                asm("trap;");
            }

            cross_entropy_loss_kernel<<<1, WARP_SIZE>>>(
                logits,
                training_mode->batch_labels,
                loss_out,
                logit_grads,
                1,
                num_classes
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: cross_entropy_loss launch failed: %s\n", cudaGetErrorString(err));
                asm("trap;");
            }

            task_performance_probe_kernel<<<1, 1>>>(
                logits,
                training_mode->batch_labels,
                1,
                num_classes,
                &organism->telemetry->task_performance,
                training_mode->is_train_batch
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: cross_entropy_loss launch failed: %s\n", cudaGetErrorString(err));
                asm("trap;");
            }

            // Backprop through classifier
            classification_head_backward_kernel<<<1, num_classes>>>(
                logit_grads,
                features,
                training_mode->classifier->fc_weights,
                organism->fc_weights_grad,
                organism->fc_bias_grad,
                organism->features_grad,
                1,
                arch.channels,
                num_classes
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: classification_head_backward launch failed: %s\n", cudaGetErrorString(err));
                asm("trap;");
            }

            // Backprop through spatial pooling
            dim3 pooling_grid(1, (arch.channels + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 pooling_block(BLOCK_SIZE);
            spatial_pooling_backward_kernel<<<pooling_grid, pooling_block>>>(
                organism->features_grad,
                organism->ca_state->ca_concentration,
                training_mode->classifier->pooling_weights,
                organism->pooling_weights_grad,
                organism->ca_state->ca_concentration,
                1,
                arch.grid_size,
                arch.channels
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: spatial_pooling_backward launch failed: %s\n", cudaGetErrorString(err));
                asm("trap;");
            }
        }

        if (organism->ad_tape->current_value_idx > 0) {
            ad_backward_kernel<<<1, BLOCK_SIZE>>>(
                organism->ad_tape,
                organism->ad_tape->current_value_idx - 1,
                1.0f
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: Backward pass failed: %s\n", cudaGetErrorString(err));
                asm("trap;");
            }
        } else {
            printf("WARN [hybrid_lifecycle]: Tape empty, skipping backward pass\n");
        }

        int total_ca_params = arch.num_heads * arch.channels * arch.hidden_dim * 3;

        float ctx_metabolic = organism->pool->entries[0].fitness;
        float ctx_stress = organism->pool->entries[0].hunger;
        float sum_morphogen = 0.0f;
        int total_cells = arch.grid_size * arch.grid_size * arch.channels;
        for (int i = 0; i < total_cells; i++) {
            sum_morphogen += organism->chemical_field->concentration[i];
        }
        float ctx_morphogen = sum_morphogen / (float)total_cells;

        TrainingParams train_params;
        train_params.derive_from_genome_hash(organism->pool->entries[0].genome_hash);
        float adam_beta1 = train_params.get_adam_beta1(primary_genome, organism->pool->entries[0].gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
        float adam_beta2 = train_params.get_adam_beta2(primary_genome, organism->pool->entries[0].gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
        float adam_epsilon = train_params.get_adam_epsilon(primary_genome, organism->pool->entries[0].gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
        float gradient_clip_norm = train_params.get_gradient_clip_norm(primary_genome, organism->pool->entries[0].gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

        dim3 adam_grid((total_ca_params + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 adam_block(BLOCK_SIZE);

        adam_update_fp16_kernel<<<adam_grid, adam_block>>>(
            organism->ca_state->perception_weights,
            organism->ad_tape->grad_buffer,
            training_mode->adam_m_perception,
            training_mode->adam_v_perception,
            total_ca_params,
            training_mode->learning_rate,
            adam_beta1,
            adam_beta2,
            adam_epsilon,
            training_mode->adam_timestep,
            gradient_clip_norm
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: Adam update (CA FP16) failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }

        dim3 pooling_adam_grid((arch.channels + BLOCK_SIZE - 1) / BLOCK_SIZE);
        adam_update_kernel<<<pooling_adam_grid, adam_block>>>(
            training_mode->classifier->pooling_weights,
            organism->pooling_weights_grad,
            organism->adam_m_pooling,
            organism->adam_v_pooling,
            arch.channels,
            training_mode->learning_rate,
            adam_beta1,
            adam_beta2,
            adam_epsilon,
            training_mode->adam_timestep,
            gradient_clip_norm
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: Adam update (pooling weights FP32) failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }

        int fc_weights_size = num_classes * behavioral_dim;
        dim3 fc_weights_adam_grid((fc_weights_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        adam_update_kernel<<<fc_weights_adam_grid, adam_block>>>(
            training_mode->classifier->fc_weights,
            organism->fc_weights_grad,
            organism->adam_m_fc_weights,
            organism->adam_v_fc_weights,
            fc_weights_size,
            training_mode->learning_rate,
            adam_beta1,
            adam_beta2,
            adam_epsilon,
            training_mode->adam_timestep,
            gradient_clip_norm
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: Adam update (fc_weights FP32) failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }

        dim3 fc_bias_adam_grid((num_classes + BLOCK_SIZE - 1) / BLOCK_SIZE);
        adam_update_kernel<<<fc_bias_adam_grid, adam_block>>>(
            training_mode->classifier->fc_bias,
            organism->fc_bias_grad,
            organism->adam_m_fc_bias,
            organism->adam_v_fc_bias,
            num_classes,
            training_mode->learning_rate,
            adam_beta1,
            adam_beta2,
            adam_epsilon,
            training_mode->adam_timestep,
            gradient_clip_norm
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: Adam update (fc_bias FP32) failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }

        training_mode->adam_timestep++;

        float* gradient_magnitudes = organism->gradient_magnitudes_pool;

        extract_head_gradient_magnitudes_kernel<<<1, BLOCK_SIZE>>>(
            organism->ad_tape,
            param_map->head_param_offsets,
            param_map->head_param_counts,
            gradient_magnitudes,
            arch.num_heads
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: extract_head_gradient_magnitudes launch failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }

        compute_gradient_fitness_kernel<<<(MAX_COMPONENTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            gradient_magnitudes,
            organism->coherence_history + generation * MAX_COMPONENTS,
            organism->fitness_history + generation * MAX_COMPONENTS,
            MAX_COMPONENTS,
            training_mode->gradient_fitness_weight,
            training_mode->coherence_fitness_weight
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: compute_gradient_fitness launch failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }
    }

    float* component_workspace_genomes;
    cudaMalloc(&component_workspace_genomes, sizeof(float) * GENOME_SIZE * 2);

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
        printf("FATAL [hybrid_lifecycle]: component_evolution launch failed: %s\n", cudaGetErrorString(err));
        asm("trap;");
    }

    if (!training_mode->use_gradients) {
        float* workspace_genome_buffer;
        cudaMalloc(&workspace_genome_buffer, sizeof(float) * GENOME_SIZE);

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

        cudaFree(workspace_genome_buffer);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: neural_ca_update launch failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }
    } else {

        update_field_from_ca_kernel<<<field_grid, field_block>>>(
            organism->chemical_field->concentration,
            organism->ca_state->ca_output,
            arch.grid_size
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: update_field_from_ca launch failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }

        int weight_count = arch.num_heads * arch.channels * arch.hidden_dim;

        int convert_threads = BLOCK_SIZE;
        int convert_blocks = (weight_count + convert_threads - 1) / convert_threads;
        convert_weights_to_fp32<<<convert_blocks, convert_threads>>>(
            organism->ca_state->perception_weights,
            organism->fp32_ca_workspace,
            weight_count
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: convert_weights_to_fp32 launch failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }

        compute_effective_rank_from_latent_kernel<<<1, BLOCK_SIZE>>>(
            organism->pool->entries[0].latent_genome,
            &organism->effective_rank_history[generation],
            GENOME_LATENT_DIM_MAX
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: compute_effective_rank launch failed: %s\n", cudaGetErrorString(err));
            asm("trap;");
        }
    }

    float* behavioral_workspace_genomes;
    cudaMalloc(&behavioral_workspace_genomes, sizeof(float) * GENOME_SIZE * 2);

    behavioral_update_kernel<<<component_grid, component_block>>>(
        organism,
        organism->behavioral_agents,
        organism->chemical_field,
        organism->memory_tubes,
        generation,
        arch,
        behavioral_workspace_genomes
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FATAL [hybrid_lifecycle]: behavioral_update launch failed: %s\n", cudaGetErrorString(err));
        asm("trap;");
    }

    uint64_t mem_genome_hash = organism->pool->entries[0].genome_hash;
    float ctx_metabolic = organism->pool->entries[0].fitness;
    float ctx_stress = organism->pool->entries[0].hunger;

    float ctx_morphogen = 0.0f;
    int chem_grid_size = arch.grid_size;
    int chem_field_size = chem_grid_size * chem_grid_size;
    for (int i = 0; i < chem_field_size; i++) {
        ctx_morphogen += organism->chemical_field->concentration[i];
    }
    ctx_morphogen /= (float)chem_field_size;

    memory_update_kernel<<<1, BLOCK_SIZE>>>(
        organism->memory_tubes,
        organism->fitness_history,
        organism->memory_compaction_valid_flags,
        organism->memory_compaction_scan,
        organism->memory_compaction_buffer,
        generation,
        primary_genome,
        organism->pool->entries[0].gradients,
        mem_genome_hash,
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
        printf("FATAL [hybrid_lifecycle]: memory_update launch failed: %s\n", cudaGetErrorString(err));
        asm("trap;");
    }

    cudaFree(component_workspace_genomes);
    cudaFree(behavioral_workspace_genomes);
}


#endif

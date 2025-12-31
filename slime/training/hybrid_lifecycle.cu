
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
__global__ void neural_ca_update_kernel(Organism*, ChemicalField*, float*, float*, int, ComponentPool*, float*, float*, float*, TraceBuffer*, int);
__global__ void update_field_from_ca_kernel(ComponentPool*, float*, int, int);
__global__ void compute_effective_rank_from_latent_kernel(ComponentPool*, float*, float*, int, int);
extern "C" __global__ void behavioral_update_kernel(Organism*, BehavioralState*, ChemicalField*, TemporalTube*, int, ArchitectureParams, float*);
__global__ void memory_update_kernel(TemporalTube*, float*, int*, int*, int*, MemoryEntry*, int, float*, float*, uint64_t, float, float, float, float, float, float, float);
__global__ void update_behavioral_embedding_kernel(BehavioralState*, float*, float*, int, float, const float*, const float*, float, float, float, float, int, int, int, int, float*);

__global__ void zero_scalar_kernel(float* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *ptr = 0.0f;
    }
}

__global__ void zero_buffer_kernel(float* buffer, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        buffer[idx] = 0.0f;
    }
}

extern "C" __global__ void hybrid_organism_lifecycle_kernel(
    Organism* organism,
    HybridTrainingMode* training_mode,
    CAParameterMap* param_map,
    int generation,
    float* workspace_genomes
) {
    int entry_idx = blockIdx.x;
    ComponentPool* pool = organism->pool;

    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    MultiHeadCAState* ca_state = entry->ca_state;

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

    int num_classes = organism->current_dataset->descriptor->num_classes;

    BehavioralDimensions dims;
    dims.derive_from_genome(entry->genome_hash, primary_genome);
    int behavioral_dim = dims.total();

    ArchitectureParams arch;
    arch.num_heads = entry->num_heads;
    arch.channels = entry->channels;
    arch.hidden_dim = entry->hidden_dim;
    arch.head_dim = entry->head_dim;
    arch.grid_size = entry->grid_size;

    dim3 component_grid((POOL_CAPACITY_MAX + (BLOCK_SIZE - 1)) / BLOCK_SIZE);
    dim3 component_block(BLOCK_SIZE);

    dim3 ca_grid(arch.grid_size / WMMA_TILE_DIM, arch.num_heads, 1);
    dim3 ca_block(WMMA_TILE_DIM, WMMA_TILE_DIM, 1);

    dim3 field_grid((arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, (arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM);
    dim3 field_block(WMMA_TILE_DIM, WMMA_TILE_DIM);

    cudaError_t err;

    printf("[HYBRID] gen=%d batch=%d grid=%d h=%d ch=%d classes=%d\n",
           generation, training_mode->batch_size, arch.grid_size,
           arch.num_heads, arch.channels, num_classes);

    // Initialize current_activation_grid_size on first run (preallocated buffers already assigned)
    if (organism->current_activation_grid_size == 0) {
        organism->current_activation_grid_size = arch.grid_size;
        printf("[HYBRID] Initialized activation grid_size=%d (using preallocated buffers)\n", arch.grid_size);
    }

    // Warn if grid size changed (buffers are preallocated to MAX size, so no reallocation needed)
    if (arch.grid_size != organism->current_activation_grid_size) {
        printf("[HYBRID] WARNING: Grid size changed from %d to %d (preallocated buffers sized for MAX)\n",
               organism->current_activation_grid_size, arch.grid_size);
        organism->current_activation_grid_size = arch.grid_size;
    }

    if (!training_mode->use_gradients) {
        printf("ERROR [hybrid_lifecycle]: use_gradients=FALSE, training pipeline WILL NOT RUN\n");
    }

    if (training_mode->use_gradients) {
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
                return;
        }
        printf("[HYBRID-CHK] sample_batch_kernel launched\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: sample_batch sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[HYBRID-CHK] sample_batch_kernel completed\n");

        if (organism == nullptr) {
            printf("FATAL [hybrid_lifecycle]: organism is NULL\n");
                return;
        }
        if (organism->ad_tape == nullptr) {
            printf("FATAL [hybrid_lifecycle]: ad_tape is NULL\n");
                return;
        }
        if (ca_state == nullptr) {
            printf("FATAL [hybrid_lifecycle]: ca_state is NULL for entry %d\n", entry_idx);
                return;
        }
        if (training_mode == nullptr) {
            printf("FATAL [hybrid_lifecycle]: training_mode is NULL\n");
                return;
        }
        if (param_map == nullptr) {
            printf("FATAL [hybrid_lifecycle]: param_map is NULL\n");
                return;
        }

        reset_tape_kernel<<<(VALUE_CAPACITY + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(organism->ad_tape);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: reset_tape launch failed: %s\n", cudaGetErrorString(err));
                return;
        }
        printf("[HYBRID-CHK] reset_tape_kernel launched\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: reset_tape sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[HYBRID-CHK] reset_tape_kernel completed\n");

        if (training_mode->batch_images == nullptr) {
            printf("FATAL [hybrid_lifecycle]: batch_images=NULL at sample injection\n");
            return;
        }

        dim3 sample_grid((arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM, (arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM, training_mode->batch_size);
        dim3 sample_block(WMMA_TILE_DIM, WMMA_TILE_DIM, 1);

        inject_sample_to_ca_kernel<<<sample_grid, sample_block>>>(
            training_mode->batch_images,
            ca_state->ca_concentration,
            training_mode->batch_size,
            arch.channels,
            arch.grid_size
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: Sample injection failed: %s\n", cudaGetErrorString(err));
            return;

        }
        printf("[HYBRID-CHK] inject_sample_to_ca_kernel launched\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: inject_sample sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[HYBRID-CHK] inject_sample_to_ca_kernel completed\n");

        int ca_output_size = training_mode->batch_size * arch.num_heads * arch.grid_size * arch.grid_size * arch.head_dim;
        zero_buffer_kernel<<<(ca_output_size + 255) / 256, 256>>>(ca_state->ca_output, ca_output_size);

        size_t buffer_capacity = NUM_HEADS_MAX * CA_FIELD_SIZE * HEAD_DIM_MAX;
        size_t per_sample_size = arch.num_heads * arch.grid_size * arch.grid_size * arch.head_dim;
        int samples_per_pass = (int)(buffer_capacity / per_sample_size);
        if (samples_per_pass < 1) samples_per_pass = 1;
        if (samples_per_pass > training_mode->batch_size) samples_per_pass = training_mode->batch_size;

        int num_passes = (training_mode->batch_size + samples_per_pass - 1) / samples_per_pass;

        int x_tiles = (arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
        int y_tiles = (arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;

        for (int pass = 0; pass < num_passes; pass++) {
            int micro_batch_offset = pass * samples_per_pass;
            int micro_batch_size = min(samples_per_pass, training_mode->batch_size - micro_batch_offset);

            dim3 ca_grid_batched(x_tiles, y_tiles, arch.num_heads * micro_batch_size);
            multi_head_ca_with_tape_kernel<<<ca_grid_batched, ca_block>>>(
                ca_state->ca_concentration,
                ca_state,
                ca_state->ca_output,
                organism->perception_activations_saved,
                organism->interaction_activations_saved,
                organism->pre_gelu_values_saved,
                param_map,
                micro_batch_size,
                micro_batch_offset,
                arch.grid_size,
                arch
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: multi_head_ca_with_tape pass %d launch failed: %s\n", pass, cudaGetErrorString(err));
                return;
            }
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: multi_head_ca_with_tape sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        float* ca_output_grad = nullptr;
        
        if (training_mode->batch_images == nullptr) {
            printf("ERROR [hybrid_lifecycle]: batch_images=NULL at classification pipeline, SKIPPING task_performance_probe_kernel and ALL classification\n");
        }

        if (training_mode->batch_images != nullptr) {

            float* features = organism->gradient_features_pool;

            if (training_mode->classifier == nullptr) {
                printf("FATAL [hybrid_lifecycle]: classifier is NULL\n");
                return;
            }
            spatial_pooling_kernel<<<training_mode->batch_size, BLOCK_SIZE>>>(
                ca_state->ca_output,
                training_mode->classifier->pooling_weights,
                features,
                training_mode->batch_size,
                arch.grid_size,
                arch.channels
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: spatial_pooling launch failed: %s\n", cudaGetErrorString(err));
                return;

            }
            printf("[HYBRID-CHK] spatial_pooling_kernel launched\n");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: spatial_pooling sync failed: %s\n", cudaGetErrorString(err));
                return;
            }
            printf("[HYBRID-CHK] spatial_pooling_kernel completed\n");

            float* logits = organism->gradient_logits_pool;

            classification_head_kernel<<<training_mode->batch_size, num_classes>>>(
                features,
                training_mode->classifier->fc_weights,
                training_mode->classifier->fc_bias,
                logits,
                training_mode->batch_size,
                arch.channels,
                num_classes
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: classification_head launch failed: %s\n", cudaGetErrorString(err));
                return;

            }
            printf("[HYBRID-CHK] classification_head_kernel launched\n");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: classification_head sync failed: %s\n", cudaGetErrorString(err));
                return;
            }
            printf("[HYBRID-CHK] classification_head_kernel completed\n");

            float* loss_out = organism->gradient_loss_pool;
            float* logit_grads = organism->gradient_logit_grads_pool;

            zero_scalar_kernel<<<1, 1>>>(loss_out);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: zero_scalar launch failed: %s\n", cudaGetErrorString(err));
                return;

            }
            printf("[HYBRID-CHK] zero_scalar_kernel launched\n");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: zero_scalar sync failed: %s\n", cudaGetErrorString(err));
                return;
            }
            printf("[HYBRID-CHK] zero_scalar_kernel completed\n");

            cross_entropy_loss_kernel<<<training_mode->batch_size, WARP_SIZE>>>(
                logits,
                training_mode->batch_labels,
                loss_out,
                logit_grads,
                training_mode->batch_size,
                num_classes
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: cross_entropy_loss launch failed: %s\n", cudaGetErrorString(err));
                return;

            }
            printf("[HYBRID-CHK] cross_entropy_loss_kernel launched\n");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: cross_entropy_loss sync failed: %s\n", cudaGetErrorString(err));
                return;
            }
            printf("[HYBRID-CHK] cross_entropy_loss_kernel completed\n");

            task_performance_probe_kernel<<<1, 1>>>(
                logits,
                training_mode->batch_labels,
                training_mode->batch_size,
                num_classes,
                &organism->telemetry->task_performance,
                training_mode->is_train_batch
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: task_performance_probe launch failed: %s\n", cudaGetErrorString(err));
                return;

            }
            printf("[HYBRID-CHK] task_performance_probe_kernel launched\n");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: task_performance_probe sync failed: %s\n", cudaGetErrorString(err));
                return;
            }
            printf("[HYBRID-CHK] task_performance_probe_kernel completed\n");

            // Zero gradient buffers before backward pass (uses atomicAdd)
            int fc_weights_size = num_classes * arch.channels;
            zero_buffer_kernel<<<(fc_weights_size + 255) / 256, 256>>>(organism->fc_weights_grad, fc_weights_size);
            zero_buffer_kernel<<<(num_classes + 255) / 256, 256>>>(organism->fc_bias_grad, num_classes);
            zero_buffer_kernel<<<(arch.channels + 255) / 256, 256>>>(organism->features_grad, arch.channels);
            zero_buffer_kernel<<<(arch.channels + 255) / 256, 256>>>(organism->pooling_weights_grad, arch.channels);

            classification_head_backward_kernel<<<training_mode->batch_size, num_classes>>>(
                logit_grads,
                features,
                training_mode->classifier->fc_weights,
                organism->fc_weights_grad,
                organism->fc_bias_grad,
                organism->features_grad,
                training_mode->batch_size,
                arch.channels,
                num_classes
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: classification_head_backward launch failed: %s\n", cudaGetErrorString(err));
                return;

            }
            printf("[HYBRID-CHK] classification_head_backward_kernel launched\n");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: classification_head_backward sync failed: %s\n", cudaGetErrorString(err));
                return;
            }
            printf("[HYBRID-CHK] classification_head_backward_kernel completed\n");

            // Backprop through spatial pooling
            dim3 pooling_grid(1, (arch.channels + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 pooling_block(BLOCK_SIZE);
            ca_output_grad = organism->buffers->ca_output_grad_buffer;

            // Zero ca_output_grad before spatial_pooling_backward (uses atomicAdd)
            int ca_grad_size = training_mode->batch_size * arch.grid_size * arch.grid_size * arch.channels;
            zero_buffer_kernel<<<(ca_grad_size + 255) / 256, 256>>>(ca_output_grad, ca_grad_size);

            spatial_pooling_backward_kernel<<<pooling_grid, pooling_block>>>(
                organism->features_grad,
                ca_state->ca_output,  // Use ca_output as input
                training_mode->classifier->pooling_weights,
                organism->pooling_weights_grad,
                ca_output_grad,  // Write gradients to separate buffer
                training_mode->batch_size,
                arch.grid_size,
                arch.channels
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: spatial_pooling_backward launch failed: %s\n", cudaGetErrorString(err));
                return;

            }
            printf("[HYBRID-CHK] spatial_pooling_backward_kernel launched\n");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: spatial_pooling_backward sync failed: %s\n", cudaGetErrorString(err));
                return;
            }
            printf("[HYBRID-CHK] spatial_pooling_backward_kernel completed\n");
        }

        // CA Backward Pass - Replace broken tape-based autodiff
        printf("[HYBRID-CHK] === CA BACKWARD PASS STARTING ===\n");
        {
            float* dL_dperception = organism->buffers->dL_dperception_buffer;
            float* dL_dinteraction = organism->buffers->dL_dinteraction_buffer;

            printf("[HYBRID-CHK] dL_dperception buffer: %p\n", (void*)dL_dperception);
            printf("[HYBRID-CHK] dL_dinteraction buffer: %p\n", (void*)dL_dinteraction);
            printf("[HYBRID-CHK] ca_output_grad buffer: %p\n", (void*)ca_output_grad);

            if (dL_dperception == nullptr) {
                printf("FATAL [hybrid_lifecycle]: dL_dperception is nullptr!\n");
                return;
            }
            if (dL_dinteraction == nullptr) {
                printf("FATAL [hybrid_lifecycle]: dL_dinteraction is nullptr!\n");
                return;
            }
            if (ca_output_grad == nullptr) {
                printf("FATAL [hybrid_lifecycle]: ca_output_grad is nullptr!\n");
                return;
            }

            // Zero gradient buffers using kernel (can't use cudaMemset from device)
            int num_elements = training_mode->batch_size * arch.num_heads * arch.grid_size * arch.grid_size * arch.hidden_dim;
            printf("[HYBRID-CHK] num_elements for zero buffers: %d\n", num_elements);
            printf("[HYBRID-CHK] total_params: %d\n", param_map->total_params);

            printf("[HYBRID-CHK] Launching zero_buffer_kernel for dL_dperception...\n");
            zero_buffer_kernel<<<(num_elements + 255) / 256, 256>>>(dL_dperception, num_elements);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: zero_buffer dL_dperception failed: %s\n", cudaGetErrorString(err));
                return;
            }

            printf("[HYBRID-CHK] Launching zero_buffer_kernel for dL_dinteraction...\n");
            zero_buffer_kernel<<<(num_elements + 255) / 256, 256>>>(dL_dinteraction, num_elements);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: zero_buffer dL_dinteraction failed: %s\n", cudaGetErrorString(err));
                return;
            }

            printf("[HYBRID-CHK] Launching zero_buffer_kernel for grad_buffer...\n");
            zero_buffer_kernel<<<(num_elements + 255) / 256, 256>>>(organism->ad_tape->grad_buffer, param_map->total_params);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: zero_buffer grad_buffer failed: %s\n", cudaGetErrorString(err));
                return;
            }

            printf("[HYBRID-CHK] Syncing after zero_buffer_kernels...\n");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: zero_buffer sync failed: %s\n", cudaGetErrorString(err));
                return;
            }
            printf("[HYBRID-CHK] Zero buffer kernels completed\n");
            
            // Allocate workspace buffers for GEMM-based backward pass
            int num_cells = arch.grid_size * arch.grid_size;
            int total_samples = training_mode->batch_size * num_cells;
            int col_width = 9 * arch.channels;

            // Per-head workspace sizes (enables parallel head processing - no loops)
            size_t ws_fp16_a_size = arch.num_heads * total_samples * max(arch.hidden_dim, col_width) * sizeof(half);
            size_t ws_fp16_b_size = arch.num_heads * total_samples * arch.hidden_dim * sizeof(half);
            size_t ws_dW_size = arch.num_heads * max(arch.hidden_dim * arch.hidden_dim, col_width * arch.hidden_dim) * sizeof(float);
            size_t ws_dI_size = arch.num_heads * total_samples * arch.hidden_dim * sizeof(float);
            size_t ws_W_T_size = arch.num_heads * max(arch.hidden_dim * arch.hidden_dim, arch.channels * arch.hidden_dim) * sizeof(half);
            size_t ws_im2col_size = arch.num_heads * total_samples * col_width * sizeof(float);
            size_t ws_dpregelu_size = arch.num_heads * total_samples * arch.hidden_dim * sizeof(float);

            half *ws_fp16_a, *ws_fp16_b, *ws_W_T;
            float *ws_dW, *ws_dI, *ws_im2col, *ws_dpregelu;

            printf("[HYBRID-CHK] Allocating workspace buffers...\n");
            printf("[HYBRID-CHK] Sizes: fp16_a=%d fp16_b=%d dW=%d dI=%d W_T=%d im2col=%d dpregelu=%d\n",
                   (int)ws_fp16_a_size, (int)ws_fp16_b_size, (int)ws_dW_size,
                   (int)ws_dI_size, (int)ws_W_T_size, (int)ws_im2col_size, (int)ws_dpregelu_size);

            err = cudaMalloc(&ws_fp16_a, ws_fp16_a_size);
            printf("[HYBRID-CHK] cudaMalloc ws_fp16_a: err=%d ptr=%p\n", (int)err, ws_fp16_a);
            if (err != cudaSuccess || ws_fp16_a == nullptr) {
                printf("FATAL [hybrid_lifecycle]: cudaMalloc ws_fp16_a failed\n");
                return;
            }
            err = cudaMalloc(&ws_fp16_b, ws_fp16_b_size);
            printf("[HYBRID-CHK] cudaMalloc ws_fp16_b: err=%d ptr=%p\n", (int)err, ws_fp16_b);
            if (err != cudaSuccess || ws_fp16_b == nullptr) {
                printf("FATAL [hybrid_lifecycle]: cudaMalloc ws_fp16_b failed\n");
                return;
            }
            err = cudaMalloc(&ws_dW, ws_dW_size);
            printf("[HYBRID-CHK] cudaMalloc ws_dW: err=%d ptr=%p\n", (int)err, ws_dW);
            if (err != cudaSuccess || ws_dW == nullptr) {
                printf("FATAL [hybrid_lifecycle]: cudaMalloc ws_dW failed\n");
                return;
            }
            err = cudaMalloc(&ws_dI, ws_dI_size);
            printf("[HYBRID-CHK] cudaMalloc ws_dI: err=%d ptr=%p\n", (int)err, ws_dI);
            if (err != cudaSuccess || ws_dI == nullptr) {
                printf("FATAL [hybrid_lifecycle]: cudaMalloc ws_dI failed\n");
                return;
            }
            err = cudaMalloc(&ws_W_T, ws_W_T_size);
            printf("[HYBRID-CHK] cudaMalloc ws_W_T: err=%d ptr=%p\n", (int)err, ws_W_T);
            if (err != cudaSuccess || ws_W_T == nullptr) {
                printf("FATAL [hybrid_lifecycle]: cudaMalloc ws_W_T failed\n");
                return;
            }
            err = cudaMalloc(&ws_im2col, ws_im2col_size);
            printf("[HYBRID-CHK] cudaMalloc ws_im2col: err=%d ptr=%p\n", (int)err, ws_im2col);
            if (err != cudaSuccess || ws_im2col == nullptr) {
                printf("FATAL [hybrid_lifecycle]: cudaMalloc ws_im2col failed\n");
                return;
            }
            err = cudaMalloc(&ws_dpregelu, ws_dpregelu_size);
            printf("[HYBRID-CHK] cudaMalloc ws_dpregelu: err=%d ptr=%p\n", (int)err, ws_dpregelu);
            if (err != cudaSuccess || ws_dpregelu == nullptr) {
                printf("FATAL [hybrid_lifecycle]: cudaMalloc ws_dpregelu failed\n");
                return;
            }

            // Stride calculations
            int I_head_stride = num_cells * arch.hidden_dim;
            int V_head_stride = num_cells * arch.head_dim;
            int I_batch_stride = arch.num_heads * I_head_stride;
            int V_batch_stride = arch.num_heads * V_head_stride;
            int ws_fp16_a_stride = total_samples * arch.hidden_dim;
            int ws_fp16_b_stride = total_samples * arch.head_dim;
            int ws_dW_stride = arch.hidden_dim * arch.head_dim;
            int ws_dI_stride = total_samples * arch.hidden_dim;
            int ws_W_T_stride = arch.head_dim * arch.hidden_dim;

            // === VALUE BACKWARD ===
            int total_I = arch.num_heads * total_samples * arch.hidden_dim;
            batched_convert_fp32_to_fp16_strided<<<(total_I + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                organism->interaction_activations_saved, ws_fp16_a,
                arch.num_heads, training_mode->batch_size, I_head_stride,
                I_head_stride, I_batch_stride, ws_fp16_a_stride
            );

            int total_V = arch.num_heads * total_samples * arch.head_dim;
            batched_convert_fp32_to_fp16_strided<<<(total_V + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ca_output_grad, ws_fp16_b,
                arch.num_heads, training_mode->batch_size, V_head_stride,
                V_head_stride, V_batch_stride, ws_fp16_b_stride
            );

            int tiles_M = (arch.hidden_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int tiles_N = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            batched_tensor_core_gemm_transA_kernel<<<dim3((tiles_M * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, tiles_N, arch.num_heads), BLOCK_SIZE>>>(
                ws_fp16_a, ws_fp16_b, ws_dW,
                arch.hidden_dim, arch.head_dim, total_samples,
                ws_fp16_a_stride, ws_fp16_b_stride, ws_dW_stride
            );

            batched_accumulate_weight_grads_kernel<<<(arch.num_heads * arch.hidden_dim * arch.head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ws_dW, organism->ad_tape->grad_buffer, param_map->value_start,
                arch.hidden_dim * arch.head_dim, arch.num_heads, ws_dW_stride
            );

            int t_tiles_x = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int t_tiles_y = (arch.hidden_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            batched_transpose_fp16_kernel<<<dim3(t_tiles_x, t_tiles_y, arch.num_heads), dim3(WMMA_TILE_DIM, WMMA_TILE_DIM)>>>(
                ca_state->value_weights, ws_W_T,
                arch.hidden_dim, arch.head_dim,
                arch.hidden_dim * arch.head_dim, ws_W_T_stride
            );

            tiles_M = (total_samples + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            tiles_N = (arch.hidden_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            batched_tensor_core_gemm_kernel<<<dim3((tiles_M * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, tiles_N, arch.num_heads), BLOCK_SIZE>>>(
                ws_fp16_b, ws_W_T, ws_dI,
                total_samples, arch.hidden_dim, arch.head_dim,
                ws_fp16_b_stride, ws_W_T_stride, ws_dI_stride
            );

            batched_memcpy_to_strided<<<(arch.num_heads * total_samples * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ws_dI, dL_dinteraction,
                arch.num_heads, training_mode->batch_size, I_head_stride,
                ws_dI_stride, I_head_stride, I_batch_stride
            );

            // === INTERACTION BACKWARD ===
            int ws_dpregelu_stride = total_samples * arch.hidden_dim;
            batched_gelu_backward_kernel<<<(arch.num_heads * total_samples * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                dL_dinteraction, organism->pre_gelu_values_saved, ws_dpregelu,
                arch.num_heads, total_samples * arch.hidden_dim,
                I_head_stride, ws_dpregelu_stride
            );

            batched_convert_fp32_to_fp16_strided<<<(total_I + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                organism->perception_activations_saved, ws_fp16_a,
                arch.num_heads, training_mode->batch_size, I_head_stride,
                I_head_stride, I_batch_stride, ws_fp16_a_stride
            );

            fp32_to_fp16_kernel<<<(arch.num_heads * total_samples * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ws_dpregelu, ws_fp16_b, arch.num_heads * total_samples * arch.hidden_dim
            );

            int dW_tiles = (arch.hidden_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int ws_dW_interaction_stride = arch.hidden_dim * arch.hidden_dim;
            batched_tensor_core_gemm_transA_kernel<<<dim3((dW_tiles * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, dW_tiles, arch.num_heads), BLOCK_SIZE>>>(
                ws_fp16_a, ws_fp16_b, ws_dW,
                arch.hidden_dim, arch.hidden_dim, total_samples,
                ws_fp16_a_stride, ws_fp16_a_stride, ws_dW_interaction_stride
            );

            batched_accumulate_weight_grads_kernel<<<(arch.num_heads * arch.hidden_dim * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ws_dW, organism->ad_tape->grad_buffer, param_map->interaction_start,
                arch.hidden_dim * arch.hidden_dim, arch.num_heads, ws_dW_interaction_stride
            );

            int ws_W_T_interaction_stride = arch.hidden_dim * arch.hidden_dim;
            batched_transpose_fp16_kernel<<<dim3(dW_tiles, dW_tiles, arch.num_heads), dim3(WMMA_TILE_DIM, WMMA_TILE_DIM)>>>(
                ca_state->interaction_weights, ws_W_T,
                arch.hidden_dim, arch.hidden_dim,
                arch.hidden_dim * arch.hidden_dim, ws_W_T_interaction_stride
            );

            batched_tensor_core_gemm_kernel<<<dim3((tiles_M * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, dW_tiles, arch.num_heads), BLOCK_SIZE>>>(
                ws_fp16_b, ws_W_T, ws_dI,
                total_samples, arch.hidden_dim, arch.hidden_dim,
                ws_fp16_a_stride, ws_W_T_interaction_stride, ws_dI_stride
            );

            batched_memcpy_to_strided<<<(arch.num_heads * total_samples * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ws_dI, dL_dperception,
                arch.num_heads, training_mode->batch_size, I_head_stride,
                ws_dI_stride, I_head_stride, I_batch_stride
            );

            // === PERCEPTION BACKWARD ===
            batched_relu_backward_kernel<<<(arch.num_heads * total_samples * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                dL_dperception, organism->perception_activations_saved, ws_dpregelu,
                arch.num_heads, total_samples * arch.hidden_dim,
                I_head_stride, ws_dpregelu_stride
            );

            int ws_im2col_stride = total_samples * col_width;
            batched_im2col_kernel<<<(arch.num_heads * total_samples + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ca_state->ca_concentration, ws_im2col,
                arch.num_heads, training_mode->batch_size, arch.grid_size, arch.channels,
                0, ws_im2col_stride
            );

            int ws_fp16_col_stride = total_samples * col_width;
            fp32_to_fp16_kernel<<<(arch.num_heads * total_samples * col_width + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ws_im2col, ws_fp16_a, arch.num_heads * total_samples * col_width
            );

            fp32_to_fp16_kernel<<<(arch.num_heads * total_samples * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ws_dpregelu, ws_fp16_b, arch.num_heads * total_samples * arch.hidden_dim
            );

            int ws_dW_perception_stride = col_width * arch.hidden_dim;
            int dW_tiles_col = (col_width + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int dW_tiles_hidden = (arch.hidden_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            batched_tensor_core_gemm_transA_kernel<<<dim3((dW_tiles_col * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, dW_tiles_hidden, arch.num_heads), BLOCK_SIZE>>>(
                ws_fp16_a, ws_fp16_b, ws_dW,
                col_width, arch.hidden_dim, total_samples,
                ws_fp16_col_stride, ws_fp16_a_stride, ws_dW_perception_stride
            );

            batched_accumulate_weight_grads_kernel<<<(arch.num_heads * arch.channels * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ws_dW, organism->ad_tape->grad_buffer, param_map->perception_start,
                arch.channels * arch.hidden_dim, arch.num_heads, ws_dW_perception_stride
            );

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("FATAL [hybrid_lifecycle]: batched backward pass failed: %s\n", cudaGetErrorString(err));
                return;
            }

            // Clean up workspace and gradient buffers
            cudaFree(ws_fp16_a);
            cudaFree(ws_fp16_b);
            cudaFree(ws_dW);
            cudaFree(ws_dI);
            cudaFree(ws_W_T);
            cudaFree(ws_im2col);
            cudaFree(ws_dpregelu);
            cudaFree(dL_dperception);
            cudaFree(dL_dinteraction);
            cudaFree(ca_output_grad);
        }

        int total_ca_params = arch.num_heads * arch.channels * arch.hidden_dim * 3;

        float ctx_metabolic = entry->fitness;
        float ctx_stress = entry->hunger;
        float sum_morphogen = 0.0f;
        int total_cells = arch.grid_size * arch.grid_size * arch.channels;
        for (int i = 0; i < total_cells; i++) {
            sum_morphogen += organism->chemical_field->concentration[i];
        }
        float ctx_morphogen = sum_morphogen / (float)total_cells;

        TrainingParams train_params;
        train_params.derive_from_genome_hash(entry->genome_hash);
        float adam_beta1 = train_params.get_adam_beta1(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
        float adam_beta2 = train_params.get_adam_beta2(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
        float adam_epsilon = train_params.get_adam_epsilon(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
        float gradient_clip_norm = train_params.get_gradient_clip_norm(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

        dim3 adam_grid((total_ca_params + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 adam_block(BLOCK_SIZE);

        adam_update_fp16_kernel<<<adam_grid, adam_block>>>(
            ca_state->perception_weights,
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
                return;

        }
        printf("[HYBRID-CHK] adam_update_fp16_kernel launched\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: adam_update_fp16 sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[HYBRID-CHK] adam_update_fp16_kernel completed\n");

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
                return;

        }
        printf("[HYBRID-CHK] adam_update_pooling_kernel launched\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: adam_update_pooling sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[HYBRID-CHK] adam_update_pooling_kernel completed\n");

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
                return;

        }
        printf("[HYBRID-CHK] adam_update_fc_weights_kernel launched\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: adam_update_fc_weights sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[HYBRID-CHK] adam_update_fc_weights_kernel completed\n");

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
                return;

        }
        printf("[HYBRID-CHK] adam_update_fc_bias_kernel launched\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: adam_update_fc_bias sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[HYBRID-CHK] adam_update_fc_bias_kernel completed\n");

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
                return;

        }
        printf("[HYBRID-CHK] extract_head_gradient_magnitudes_kernel launched\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: extract_head_gradient_magnitudes sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[HYBRID-CHK] extract_head_gradient_magnitudes_kernel completed\n");

        compute_gradient_fitness_kernel<<<(POOL_CAPACITY_MAX + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            gradient_magnitudes,
            organism->coherence_history + (generation % 2) * POOL_CAPACITY_MAX,
            organism->fitness_history + (generation % 2) * POOL_CAPACITY_MAX,
            POOL_CAPACITY_MAX,
            training_mode->gradient_fitness_weight,
            training_mode->coherence_fitness_weight
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: compute_gradient_fitness launch failed: %s\n", cudaGetErrorString(err));
                return;

        }
        printf("[HYBRID-CHK] compute_gradient_fitness_kernel launched\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: compute_gradient_fitness sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[HYBRID-CHK] compute_gradient_fitness_kernel completed\n");
    }

    float* component_workspace_genomes = organism->buffers->component_workspace_genomes_buffer;

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
                return;

    }
    printf("[HYBRID-CHK] component_evolution_kernel launched\n");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("FATAL [hybrid_lifecycle]: component_evolution sync failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("[HYBRID-CHK] component_evolution_kernel completed\n");

    if (!training_mode->use_gradients) {
        float* workspace_genomes = organism->buffers->organism_workspace_genomes;

        neural_ca_update_kernel<<<pool->capacity, 1>>>(
            organism,
            organism->chemical_field,
            organism->effective_rank_history,
            organism->fp32_ca_workspace,
            generation,
            organism->pool,
            organism->fitness_history,
            organism->coherence_history,
            workspace_genomes,
            organism->trace_buffer,
            arch.grid_size
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: neural_ca_update launch failed: %s\n", cudaGetErrorString(err));
                return;

        }
        printf("[HYBRID-CHK] neural_ca_update_kernel launched\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: neural_ca_update sync failed: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("[HYBRID-CHK] neural_ca_update_kernel completed\n");
    } else {

        dim3 update_grid(field_grid.x, field_grid.y, 1);
        update_field_from_ca_kernel<<<update_grid, field_block>>>(
            pool,
            organism->chemical_field->concentration,
            arch.grid_size,
            entry_idx
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: update_field_from_ca launch failed: %s\n", cudaGetErrorString(err));
                return;

        }

        int weight_count = arch.num_heads * arch.channels * arch.hidden_dim;

        int convert_threads = BLOCK_SIZE;
        int convert_blocks = (weight_count + convert_threads - 1) / convert_threads;
        convert_weights_to_fp32<<<convert_blocks, convert_threads>>>(
            ca_state->perception_weights,
            organism->fp32_ca_workspace,
            weight_count
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: convert_weights_to_fp32 launch failed: %s\n", cudaGetErrorString(err));
                return;

        }

        float* temp_latent = primary_parent_temp;
        diresa_encode(primary_genome, temp_latent, &organism->diresa_genome_weights[0]);

        compute_effective_rank_from_latent_kernel<<<1, BLOCK_SIZE>>>(
            pool,
            organism->effective_rank_history,
            workspace_genomes,
            GENOME_LATENT_DIM_MAX,
            entry_idx
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: compute_effective_rank launch failed: %s\n", cudaGetErrorString(err));
                return;

        }
    }

    float* behavioral_workspace_genomes = organism->buffers->behavioral_workspace_genomes_buffer;

    behavioral_update_kernel<<<component_grid, component_block>>>(
        organism,
        organism->behavioral_agents,
        organism->chemical_field,
        organism->chemical_field->history,
        generation,
        arch,
        behavioral_workspace_genomes
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FATAL [hybrid_lifecycle]: behavioral_update launch failed: %s\n", cudaGetErrorString(err));
                return;

    }

    if (generation % EMBEDDING_UPDATE_FREQ == 0) {
        zero_scalar_kernel<<<1, 1>>>(organism->buffers->behavioral_reconstruction_error);

        int hw_dim = BEHAVIORAL_DIM_HW_MAX;
        int task_dim = BEHAVIORAL_DIM_TASK_MAX;
        int gen_dim = BEHAVIORAL_DIM_GEN_MAX;
        int behavioral_dim = hw_dim + task_dim + gen_dim;

        update_behavioral_embedding_kernel<<<component_grid, component_block>>>(
            organism->behavioral_agents,
            organism->buffers->behavioral_embedding_weights,
            organism->buffers->behavioral_reconstruction_error,
            POOL_CAPACITY_MAX,
            0.01f,
            primary_genome,
            entry->gradients,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            behavioral_dim,
            hw_dim,
            task_dim,
            gen_dim,
            organism->buffers->behavioral_features_buffer
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("FATAL [hybrid_lifecycle]: update_behavioral_embedding launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    uint64_t mem_genome_hash = entry->genome_hash;
    float ctx_metabolic = entry->fitness;
    float ctx_stress = entry->hunger;

    float ctx_morphogen = 0.0f;
    int chem_grid_size = arch.grid_size;
    int chem_field_size = chem_grid_size * chem_grid_size;
    for (int i = 0; i < chem_field_size; i++) {
        ctx_morphogen += organism->chemical_field->concentration[i];
    }
    ctx_morphogen /= (float)chem_field_size;

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
                return;
        
    }

    cudaFree(component_workspace_genomes);
    cudaFree(behavioral_workspace_genomes);

    // Synchronize to ensure ALL child kernels complete before exiting
    // This prevents persistent loop's cudaDeviceSynchronize from blocking forever
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[HYBRID-ERROR] Final cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("[HYBRID-EXIT] gen=%d completed\n", generation);
}


#endif

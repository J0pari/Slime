
#ifndef HYBRID_LIFECYCLE_CU
#define HYBRID_LIFECYCLE_CU

#include "../config/config.cu"
#include "../training/training_types.cu"
#include "../data/dataset_loader.cu"
#include "../core/chemotaxis.cu"
#include "../core/ca_state.cuh"
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

extern "C" __global__ void component_evolution_kernel(Organism*, ComponentPool*, GPUElite*, VoronoiCell*, int, int*, ChemicalField*, BehavioralState*, float*, float*, int, ArchitectureParams, float*);
__global__ void neural_ca_update_kernel(Organism*, ChemicalField*, float*, float*, int, ComponentPool*, float*, float*, float*, TraceBuffer*, int);
__global__ void update_field_from_ca_kernel(ComponentPool*, float*, int, int);
__global__ void compute_effective_rank_from_latent_kernel(ComponentPool*, float*, float*, int, int);
extern "C" __global__ void behavioral_update_kernel(Organism*, BehavioralState*, ChemicalField*, TemporalTube*, int, ArchitectureParams, float*);
__global__ void memory_update_kernel(TemporalTube*, float*, int*, int*, int*, MemoryEntry*, int, float*, float*, uint64_t, float, float, float, float, float, float, float);
__global__ void update_behavioral_embedding_kernel(BehavioralState*, float*, float*, int, float, const float*, const float*, float, float, float, float, int, int, int, int, float*, const float*, int);

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
    float* workspace_genomes,
    bool eval_only,
    AuditBuffer* audit
) {
    extern __shared__ float sdata[];
    int entry_idx = blockIdx.x;
    int tid = threadIdx.x;
    ComponentPool* pool = organism->pool;

    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    MultiHeadCAState* ca_state = entry->ca_state;

    int local_cells = entry->channels * entry->grid_size * entry->grid_size;
    float thread_sum = 0.0f;
    for (int i = tid; i < local_cells; i += blockDim.x) {
        thread_sum += ca_state->ca_concentration[i];
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float local_ca_mean = sdata[0] / (float)local_cells;
    // Broadcast to all threads via shared memory (threads stay alive for parallel Flow Lenia)
    if (tid == 0) sdata[0] = local_ca_mean;
    __syncthreads();
    local_ca_mean = sdata[0];

    // Shared error flag - if tid==0 detects an error, all threads must exit together
    __shared__ int s_error_flag;
    if (tid == 0) s_error_flag = 0;
    __syncthreads();

    // Sequential setup - only thread 0 (other threads wait for parallel Flow Lenia)
    __shared__ float* s_primary_genome;
    __shared__ int s_num_classes;
    __shared__ int s_behavioral_dim;
    __shared__ ArchitectureParams s_arch;
    __shared__ dim3 s_component_grid;
    __shared__ dim3 s_component_block;
    __shared__ dim3 s_ca_grid;
    __shared__ dim3 s_ca_block;
    __shared__ dim3 s_field_grid;
    __shared__ dim3 s_field_block;

    float* primary_genome;
    float* primary_parent_temp;
    int num_classes;
    int behavioral_dim;
    ArchitectureParams arch;
    dim3 component_grid;
    dim3 component_block;
    dim3 ca_grid;
    dim3 ca_block;
    dim3 field_grid;
    dim3 field_block;
    cudaError_t err;

    if (tid == 0) {
        primary_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];
        primary_parent_temp = &workspace_genomes[entry_idx * GENOME_SIZE * 2 + GENOME_SIZE];

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

        num_classes = organism->current_dataset->descriptor->num_classes;

        BehavioralDimensions dims;
        dims.derive_from_genome(entry->genome_hash, primary_genome);
        behavioral_dim = dims.total();

        arch.num_heads = entry->num_heads;
        arch.channels = entry->channels;
        arch.hidden_dim = entry->hidden_dim;
        arch.head_dim = entry->head_dim;
        arch.grid_size = entry->grid_size;
        float accuracy = organism->telemetry->task_performance.accuracy;
        arch.ca_gate_center = 2.0f - 1.5f * fminf(fmaxf(accuracy, 0.0f), 1.0f);

        component_grid = dim3(POOL_CAPACITY_MAX);  // One block per pool entry
        component_block = dim3(WARP_SIZE);  // Match behavioral_update_kernel expectations
        ca_grid = dim3(arch.grid_size / WMMA_TILE_DIM, arch.num_heads, 1);
        ca_block = dim3(WMMA_TILE_DIM, WMMA_TILE_DIM, 1);
        field_grid = dim3((arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, (arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM);
        field_block = dim3(WMMA_TILE_DIM, WMMA_TILE_DIM);

        // Broadcast to shared for all threads
        s_primary_genome = primary_genome;
        s_num_classes = num_classes;
        s_behavioral_dim = behavioral_dim;
        s_arch = arch;
        s_component_grid = component_grid;
        s_component_block = component_block;
        s_ca_grid = ca_grid;
        s_ca_block = ca_block;
        s_field_grid = field_grid;
        s_field_block = field_block;
    }
    __syncthreads();

    // All threads read shared values
    primary_genome = s_primary_genome;
    primary_parent_temp = primary_genome + GENOME_SIZE;
    num_classes = s_num_classes;
    behavioral_dim = s_behavioral_dim;
    arch = s_arch;
    component_grid = s_component_grid;
    component_block = s_component_block;
    ca_grid = s_ca_grid;
    ca_block = s_ca_block;
    field_grid = s_field_grid;
    field_block = s_field_block;

    // Initialize current_activation_grid_size on first run (preallocated buffers already assigned)
    if (organism->current_activation_grid_size == 0) {
        organism->current_activation_grid_size = arch.grid_size;
    }

    // Warn if grid size changed (buffers are preallocated to MAX size, so no reallocation needed)
    if (arch.grid_size != organism->current_activation_grid_size) {
        organism->current_activation_grid_size = arch.grid_size;
    }

    if (!training_mode->use_gradients) {
    }

    if (training_mode->use_gradients) {
        if (tid == 0) {
        if (organism == nullptr || ca_state == nullptr ||
            training_mode == nullptr || param_map == nullptr) {
            printf("!E:hybrid null_check org=%p ca_state=%p tm=%p pm=%p\n",
                   (void*)organism,
                   (void*)ca_state,
                   (void*)training_mode,
                   (void*)param_map);
            s_error_flag = 1;
        }

        if (!s_error_flag && (training_mode->batch_images == nullptr || training_mode->batch_labels == nullptr)) {
            printf("!E:hybrid batch_buffers_null images=%p labels=%p\n",
                   (void*)training_mode->batch_images,
                   (void*)training_mode->batch_labels);
            s_error_flag = 1;
        }

        if (!s_error_flag) sample_batch_kernel<<<training_mode->batch_size, BLOCK_SIZE>>>(
            organism->current_dataset,
            training_mode,
            training_mode->batch_size,
            generation * training_mode->batch_size,
            arch.grid_size
        );
        CUDA_LAUNCH_CHECK();

        SLIME_DEBUG_PRINT("V:hybrid:155 pre_reset_tape\n");
        reset_tape_kernel<<<(VALUE_CAPACITY + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(&ca_state->tape);
        CUDA_LAUNCH_CHECK();
        SLIME_DEBUG_PRINT("V:hybrid:165 post_reset_tape\n");

        dim3 sample_grid((arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM, (arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM, training_mode->batch_size);
        dim3 sample_block(WMMA_TILE_DIM, WMMA_TILE_DIM, 1);

        SLIME_DEBUG_PRINT("V:hybrid:225 pre_inject grid=(%d,%d,%d) block=(%d,%d,%d)\n", sample_grid.x, sample_grid.y, sample_grid.z, sample_block.x, sample_block.y, sample_block.z);
        inject_sample_to_ca_kernel<<<sample_grid, sample_block>>>(
            organism->batch_ca_states_pool,                // output: batched CA state [batch × grid² × channels]
            training_mode->batch_size,                     // batch_size
            arch.channels,                                 // channels
            arch.grid_size,                                // grid_size
            // ChemicalField sources (channels 0-5)
            organism->chemical_field->concentration,
            organism->chemical_field->gradient_x,
            organism->chemical_field->gradient_y,
            organism->chemical_field->laplacian,
            organism->chemical_field->sources,
            organism->chemical_field->decay_factors,
            // RD field sources (channels 6-9)
            organism->resource_density,
            organism->fitness_landscape,
            organism->resource_gradient_x,
            organism->resource_gradient_y,
            // Behavioral field (channel 10)
            organism->behavioral_field_pool,
            // Dataset sample (channels 11-13)
            training_mode->batch_images,
            (int)organism->current_dataset->descriptor->channels,
            // Recurrence (channel 14) - previous step's final concentration
            organism->batch_prev_concentration,
            // Temporal retrieval (channel 15) - use chemical field for temporal coherence
            organism->chemical_field->concentration
        );
        CUDA_LAUNCH_CHECK();
        SLIME_DEBUG_PRINT("V:hybrid:235 post_inject\n");

        // CA output layout: [batch × num_heads × grid² × channels]
        size_t buffer_capacity = NUM_HEADS_MAX * CA_FIELD_SIZE * CHANNELS_MAX;
        size_t per_sample_size = arch.num_heads * arch.grid_size * arch.grid_size * arch.channels;
        int samples_per_pass = (int)(buffer_capacity / per_sample_size);
        if (samples_per_pass < 1) samples_per_pass = 1;
        if (samples_per_pass > training_mode->batch_size) samples_per_pass = training_mode->batch_size;

        int num_passes = (training_mode->batch_size + samples_per_pass - 1) / samples_per_pass;

        int x_tiles = (arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
        int y_tiles = (arch.grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;

        SLIME_DEBUG_PRINT("V:hybrid:255 pre_ca_loop num_passes=%d samples_per_pass=%d\n", num_passes, samples_per_pass);
        for (int pass = 0; pass < num_passes; pass++) {
            int micro_batch_offset = pass * samples_per_pass;
            int micro_batch_size = min(samples_per_pass, training_mode->batch_size - micro_batch_offset);

            dim3 ca_grid_batched(x_tiles, y_tiles, arch.num_heads * micro_batch_size);
            SLIME_DEBUG_PRINT("V:hybrid:260 pre_ca_kernel pass=%d grid=(%d,%d,%d)\n", pass, ca_grid_batched.x, ca_grid_batched.y, ca_grid_batched.z);
            multi_head_ca_with_tape_kernel<<<ca_grid_batched, ca_block>>>(
                organism->batch_ca_states_pool,       // batched input: [batch × grid² × channels]
                ca_state,
                organism->buffers->batched_ca_output, // batched output: [batch × num_heads × grid² × channels]
                ca_state->perception_saved,
                ca_state->interaction_saved,
                ca_state->pre_gelu_saved,
                param_map,
                micro_batch_size,
                micro_batch_offset,
                arch.grid_size,
                arch
            );

            CUDA_LAUNCH_CHECK();
            SLIME_DEBUG_PRINT("V:hybrid:274 post_ca_kernel pass=%d\n", pass);
        }

        SLIME_DEBUG_PRINT("V:hybrid:276 pre_sync\n");
        }  // end if (tid == 0)

        // All threads check error flag and exit together if error detected
        __syncthreads();
        if (s_error_flag) return;

        // Skip cudaDeviceSynchronize to avoid CDP blocking across blocks
        // Child kernels may still be running - Flow Lenia will use current buffer state
        SLIME_DEBUG_PRINT("V:hybrid:278 post_sync\n");

    // Flow Lenia: transport mass based on CA affinity gradients (ALL threads of ALL blocks)
    {
            int total_cells = arch.grid_size * arch.grid_size;
            int buffer_size = training_mode->batch_size * total_cells * arch.channels;
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads = arch.num_heads;
            int head_dim = arch.head_dim;
            float flow_beta_A = entry->flow_beta_A;
            float flow_n = entry->flow_n;
            float flow_alpha_min = entry->flow_alpha_min;
            float flow_alpha_max = entry->flow_alpha_max;
            float flow_sharpness = entry->flow_sharpness;
            float flow_dt = entry->flow_resource_dt;

            // CA output layout: [batch × num_heads × grid² × channels]
            int total_affinity_work = batch_size * total_cells;
            for (int work_idx = tid; work_idx < total_affinity_work; work_idx += blockDim.x) {
                int batch_idx = work_idx / total_cells;
                int cell_idx = work_idx % total_cells;

                float affinity = 0.0f;
                int total_elements = num_heads * channels;
                for (int i = 0; i < total_elements; i++) {
                    int head = i / channels;
                    int c = i % channels;
                    int idx = batch_idx * num_heads * total_cells * channels +
                              head * total_cells * channels +
                              cell_idx * channels + c;
                    affinity += organism->buffers->batched_ca_output[idx];
                }
                organism->batch_affinity_reduced[batch_idx * total_cells + cell_idx] = affinity;
            }
            __syncthreads();

            for (int work_idx = tid; work_idx < total_affinity_work; work_idx += blockDim.x) {
                int batch_idx = work_idx / total_cells;
                int cell_idx = work_idx % total_cells;
                int x = cell_idx % grid_size;
                int y = cell_idx / grid_size;

                int batch_offset = batch_idx * total_cells;
                float U_center = organism->batch_affinity_reduced[batch_offset + cell_idx];
                int x_E = min(x + 1, grid_size - 1);
                int y_N = min(y + 1, grid_size - 1);
                float U_E = organism->batch_affinity_reduced[batch_offset + y * grid_size + x_E];
                float U_N = organism->batch_affinity_reduced[batch_offset + y_N * grid_size + x];

                int conc_batch_offset = batch_idx * total_cells * channels;
                float A_sum_center = 0.0f, A_sum_E = 0.0f, A_sum_N = 0.0f;
                for (int c = 0; c < channels; c++) {
                    A_sum_center += organism->batch_ca_states_pool[conc_batch_offset + cell_idx * channels + c];
                    A_sum_E += organism->batch_ca_states_pool[conc_batch_offset + (y * grid_size + x_E) * channels + c];
                    A_sum_N += organism->batch_ca_states_pool[conc_batch_offset + (y_N * grid_size + x) * channels + c];
                }

                float2 F = FlowLeniaOps::compute_flow_differentiable(
                    U_center, U_E, U_N, A_sum_center, A_sum_E, A_sum_N,
                    flow_beta_A, flow_n, flow_alpha_min, flow_alpha_max, flow_sharpness
                );

                int flow_idx = batch_idx * total_cells * 2 + cell_idx * 2;
                organism->batch_flow_field[flow_idx + 0] = F.x;
                organism->batch_flow_field[flow_idx + 1] = F.y;
            }

            // Clear reintegration buffer
            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                organism->batch_reintegration_buffer[idx] = 0.0f;
            }
            __syncthreads();

            // Phase 4: Bilinear splatting transport (all threads parallel)
            int total_splat_work = batch_size * total_cells;
            for (int work_idx = tid; work_idx < total_splat_work; work_idx += blockDim.x) {
                int batch_idx = work_idx / total_cells;
                int cell_idx = work_idx % total_cells;
                int source_x = cell_idx % grid_size;
                int source_y = cell_idx / grid_size;

                int batch_offset = batch_idx * total_cells;
                int flow_idx = batch_offset * 2 + cell_idx * 2;
                float Fx = organism->batch_flow_field[flow_idx + 0];
                float Fy = organism->batch_flow_field[flow_idx + 1];

                int conc_batch_offset = batch_idx * total_cells * channels;
                float* batch_buffer = organism->batch_reintegration_buffer + conc_batch_offset;
                const float* batch_conc = organism->batch_ca_states_pool + conc_batch_offset;

                for (int c = 0; c < channels; c++) {
                    float source_mass = batch_conc[cell_idx * channels + c];
                    FlowLeniaOps::bilinear_transport_forward(
                        source_mass,
                        (float)source_x, (float)source_y,
                        Fx, Fy, flow_dt, grid_size,
                        batch_buffer, c, channels
                    );
                }
            }
            __syncthreads();

            // Phase 5: Copy transported mass back to concentration (all threads parallel)
            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                organism->batch_ca_states_pool[idx] = organism->batch_reintegration_buffer[idx];
            }
            __syncthreads();

            // Phase 6: Save concentration for next iteration's recurrence channel
            // batch_prev_concentration will be read by inject_sample_to_ca_kernel on next step
            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                organism->batch_prev_concentration[idx] = organism->batch_ca_states_pool[idx];
            }
        }  // end Flow Lenia
        __syncthreads();

        SLIME_DEBUG_PRINT("V:hybrid:412 post_flow_lenia tid=%d\n", tid);

        // Continue training code - thread 0 in each block launches per-entry kernels
        if (tid == 0) {
        SLIME_DEBUG_PRINT("V:hybrid:415 tid0_entry\n");
        float* ca_output_grad = nullptr;
        
        if (training_mode->batch_images == nullptr) {
        }

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {

            float* features = organism->gradient_features_pool;

            SLIME_DEBUG_PRINT("V:hybrid:428 pre_spatial_pooling\n");
            int num_features = arch.num_heads * arch.channels;
            dim3 pooling_grid(training_mode->batch_size, (num_features + BLOCK_SIZE - 1) / BLOCK_SIZE);
            spatial_pooling_kernel<<<pooling_grid, BLOCK_SIZE>>>(
                organism->buffers->batched_ca_output,
                training_mode->classifier->pooling_weights,
                features,
                training_mode->batch_size,
                arch.num_heads,
                arch.grid_size,
                arch.channels
            );

            CUDA_LAUNCH_CHECK();
            SLIME_DEBUG_PRINT("V:hybrid:438 post_spatial_pooling\n");

            float* logits = organism->gradient_logits_pool;

            classification_head_kernel<<<training_mode->batch_size, num_classes>>>(
                features,
                training_mode->classifier->fc_weights,
                training_mode->classifier->fc_bias,
                logits,
                training_mode->batch_size,
                num_features,
                num_classes
            );
            CUDA_LAUNCH_CHECK();
            SLIME_DEBUG_PRINT("V:hybrid:456 post_classification_head\n");

            float* loss_out = organism->gradient_loss_pool;
            float* logit_grads = organism->gradient_logit_grads_pool;

            zero_scalar_kernel<<<1, 1>>>(loss_out);
            CUDA_LAUNCH_CHECK();
            SLIME_DEBUG_PRINT("V:hybrid:462 post_zero_scalar\n");

            cross_entropy_loss_kernel<<<training_mode->batch_size, WARP_SIZE>>>(
                logits,
                training_mode->batch_labels,
                loss_out,
                logit_grads,
                training_mode->batch_size,
                num_classes
            );
            CUDA_LAUNCH_CHECK();
            SLIME_DEBUG_PRINT("V:hybrid:472 post_cross_entropy\n");

            task_performance_probe_kernel<<<1, 1>>>(
                logits,
                training_mode->batch_labels,
                training_mode->batch_size,
                num_classes,
                &organism->telemetry->task_performance,
                training_mode->is_train_batch
            );
            CUDA_LAUNCH_CHECK();
            SLIME_DEBUG_PRINT("V:hybrid:482 post_task_performance\n");

            // Block 0 syncs and populates audit buffer with forward pass results
            if (entry_idx == 0 && audit) {
                cudaDeviceSynchronize();
                populate_audit_buffer(
                    audit,
                    generation,
                    logits,
                    training_mode->batch_labels,
                    training_mode->batch_images,
                    training_mode->batch_size,
                    num_classes,
                    organism->pool->entries[0].ca_state->ca_concentration,
                    arch.grid_size,
                    organism->telemetry->task_performance.train_accuracy,
                    organism->telemetry->task_performance.test_accuracy,
                    organism->telemetry,
                    organism->pool
                );
                SLIME_DEBUG_PRINT("V:hybrid:audit_done gen=%d\n", generation);
            }

            SLIME_DEBUG_PRINT("V:hybrid:484 post_sync\n");

            // Skip backward pass for eval-only mode (test evaluation)
            if (!eval_only) {
            SLIME_DEBUG_PRINT("V:hybrid:486 enter_backward eval_only=%d\n", eval_only);

            // Zero gradient buffers before backward pass (uses atomicAdd)
            int fc_weights_size = num_classes * arch.channels;
            zero_buffer_kernel<<<(fc_weights_size + 255) / 256, 256>>>(organism->fc_weights_grad, fc_weights_size);
            zero_buffer_kernel<<<(num_classes + 255) / 256, 256>>>(organism->fc_bias_grad, num_classes);
            zero_buffer_kernel<<<(arch.channels + 255) / 256, 256>>>(organism->features_grad, arch.channels);
            zero_buffer_kernel<<<(arch.channels + 255) / 256, 256>>>(organism->pooling_weights_grad, arch.channels);

            SLIME_DEBUG_PRINT("V:hybrid:495 pre_class_backward\n");
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
            CUDA_LAUNCH_CHECK();
            SLIME_DEBUG_PRINT("V:hybrid:507 post_class_backward\n");

            // Backprop through spatial pooling
            int num_features_bwd = arch.num_heads * arch.channels;
            dim3 pooling_grid_bwd(training_mode->batch_size, (num_features_bwd + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 pooling_block_bwd(BLOCK_SIZE);
            ca_output_grad = organism->buffers->ca_output_grad_buffer;

            // Zero ca_output_grad before spatial_pooling_backward (uses atomicAdd)
            // Layout: [batch × num_heads × grid² × channels]
            int ca_grad_size = training_mode->batch_size * arch.num_heads * arch.grid_size * arch.grid_size * arch.channels;
            zero_buffer_kernel<<<(ca_grad_size + 255) / 256, 256>>>(ca_output_grad, ca_grad_size);

            SLIME_DEBUG_PRINT("V:hybrid:520 pre_pooling_backward\n");
            spatial_pooling_backward_kernel<<<pooling_grid_bwd, pooling_block_bwd>>>(
                organism->features_grad,
                organism->buffers->batched_ca_output,
                training_mode->classifier->pooling_weights,
                organism->pooling_weights_grad,
                ca_output_grad,
                training_mode->batch_size,
                arch.num_heads,
                arch.grid_size,
                arch.channels
            );
            CUDA_LAUNCH_CHECK();
            SLIME_DEBUG_PRINT("V:hybrid:530 post_pooling_backward\n");
        }

        // CA Backward Pass - Replace broken tape-based autodiff
        if (!eval_only) {
            SLIME_DEBUG_PRINT("V:hybrid:535 enter_ca_backward\n");
            float* dL_dperception = organism->buffers->dL_dperception_buffer;
            float* dL_dinteraction = organism->buffers->dL_dinteraction_buffer;

            if (dL_dperception != nullptr && dL_dinteraction != nullptr && ca_output_grad != nullptr) {

            // Use preallocated workspace buffers for GEMM-based backward pass
            int num_cells = arch.grid_size * arch.grid_size;
            int total_samples = training_mode->batch_size * num_cells;
            int col_width = 9 * arch.channels;

            half* ws_fp16_a = organism->buffers->backward_ws_fp16_a;
            half* ws_fp16_b = organism->buffers->backward_ws_fp16_b;
            float* ws_dW = organism->buffers->backward_ws_dW;
            float* ws_dI = organism->buffers->backward_ws_dI;
            half* ws_W_T = organism->buffers->backward_ws_W_T;
            float* ws_im2col = organism->buffers->backward_ws_im2col;
            float* ws_dpregelu = organism->buffers->backward_ws_dpregelu;

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
                ca_state->interaction_saved, ws_fp16_a,
                arch.num_heads, training_mode->batch_size, I_head_stride,
                I_head_stride, I_batch_stride, ws_fp16_a_stride, 0
            );

            int total_V = arch.num_heads * total_samples * arch.head_dim;
            batched_convert_fp32_to_fp16_strided<<<(total_V + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ca_output_grad, ws_fp16_b,
                arch.num_heads, training_mode->batch_size, V_head_stride,
                V_head_stride, V_batch_stride, ws_fp16_b_stride, 0
            );

            int tiles_M = (arch.hidden_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int tiles_N = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            batched_tensor_core_gemm_transA_kernel<<<dim3((tiles_M * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, tiles_N, arch.num_heads), BLOCK_SIZE>>>(
                ws_fp16_a, ws_fp16_b, ws_dW,
                arch.hidden_dim, arch.head_dim, total_samples,
                ws_fp16_a_stride, ws_fp16_b_stride, ws_dW_stride
            );

            batched_accumulate_weight_grads_kernel<<<(arch.num_heads * arch.hidden_dim * arch.head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ws_dW, ca_state->tape.grad_buffer, param_map->value_start,
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
                ws_dI_stride, I_head_stride, I_batch_stride, 0
            );

            // === INTERACTION BACKWARD ===
            int ws_dpregelu_stride = total_samples * arch.hidden_dim;
            batched_gelu_backward_kernel<<<(arch.num_heads * total_samples * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                dL_dinteraction, ca_state->pre_gelu_saved, ws_dpregelu,
                arch.num_heads, total_samples * arch.hidden_dim,
                I_head_stride, ws_dpregelu_stride
            );

            batched_convert_fp32_to_fp16_strided<<<(total_I + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                ca_state->perception_saved, ws_fp16_a,
                arch.num_heads, training_mode->batch_size, I_head_stride,
                I_head_stride, I_batch_stride, ws_fp16_a_stride, 0
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
                ws_dW, ca_state->tape.grad_buffer, param_map->interaction_start,
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
                ws_dI_stride, I_head_stride, I_batch_stride, 0
            );

            // === PERCEPTION BACKWARD ===
            batched_relu_backward_kernel<<<(arch.num_heads * total_samples * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                dL_dperception, ca_state->perception_saved, ws_dpregelu,
                arch.num_heads, total_samples * arch.hidden_dim,
                I_head_stride, ws_dpregelu_stride
            );

            int ws_im2col_stride = total_samples * col_width;
            batched_im2col_kernel<<<(arch.num_heads * total_samples + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                organism->batch_ca_states_pool, ws_im2col,
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
                ws_dW, ca_state->tape.grad_buffer, param_map->perception_start,
                arch.channels * arch.hidden_dim, arch.num_heads, ws_dW_perception_stride
            );

            SLIME_DEBUG_PRINT("V:hybrid:712 post_perception_backward\n");
            }  // end if (buffers != nullptr)
        }  // end if (!eval_only) for CA backward
        SLIME_DEBUG_PRINT("V:hybrid:715 post_ca_backward\n");

        int total_ca_params = arch.num_heads * arch.channels * arch.hidden_dim * 3;

        float ctx_metabolic = entry->fitness;
        float ctx_stress = entry->hunger;
        float ctx_morphogen = local_ca_mean;

        TrainingParams train_params;
        train_params.derive_from_genome_hash(entry->genome_hash);
        float adam_beta1 = train_params.get_adam_beta1(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
        float adam_beta2 = train_params.get_adam_beta2(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
        float adam_epsilon = train_params.get_adam_epsilon(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
        float gradient_clip_norm = train_params.get_gradient_clip_norm(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

        dim3 adam_grid((total_ca_params + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 adam_block(BLOCK_SIZE);

        SLIME_DEBUG_PRINT("V:hybrid:733 pre_adam_fp16\n");
        adam_update_fp16_kernel<<<adam_grid, adam_block>>>(
            ca_state->perception_weights,
            ca_state->tape.grad_buffer,
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
        CUDA_LAUNCH_CHECK();
        SLIME_DEBUG_PRINT("V:hybrid:746 post_adam_fp16\n");

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
        CUDA_LAUNCH_CHECK();

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
        CUDA_LAUNCH_CHECK();

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
        CUDA_LAUNCH_CHECK();

        training_mode->adam_timestep++;

        float* gradient_magnitudes = organism->gradient_magnitudes_pool;

        extract_head_gradient_magnitudes_kernel<<<1, BLOCK_SIZE>>>(
            &ca_state->tape,
            param_map->head_param_offsets,
            param_map->head_param_counts,
            gradient_magnitudes,
            arch.num_heads
        );
        CUDA_LAUNCH_CHECK();

        compute_gradient_fitness_kernel<<<(POOL_CAPACITY_MAX + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            gradient_magnitudes,
            organism->coherence_history + (generation % 2) * POOL_CAPACITY_MAX,
            organism->fitness_history + (generation % 2) * POOL_CAPACITY_MAX,
            POOL_CAPACITY_MAX,
            training_mode->gradient_fitness_weight,
            training_mode->coherence_fitness_weight
        );
        CUDA_LAUNCH_CHECK();
        SLIME_DEBUG_PRINT("V:hybrid:821 post_gradient_fitness\n");
        }  // end if (training_mode->batch_images != nullptr)
    }  // end if (tid == 0) for training code
    }  // end if (training_mode->use_gradients)
    SLIME_DEBUG_PRINT("V:hybrid:825 post_training entry_idx=%d\n", entry_idx);

    // GLOBAL operation - processes all entries internally, launch once
    if (tid == 0 && entry_idx == 0) {
        SLIME_DEBUG_PRINT("V:hybrid:828 pre_component_evolution\n");
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
        CUDA_LAUNCH_CHECK();
        // Component evolution sync handled by parent after this kernel returns

        if (!training_mode->use_gradients) {
            SLIME_DEBUG_PRINT("V:hybrid:852 pre_neural_ca\n");
            float* workspace_genomes = organism->buffers->organism_workspace_genomes;
            neural_ca_update_kernel<<<pool->capacity, 1>>>(
                organism,
                organism->chemical_field,
                organism->effective_rank_history,
                ca_state->fp32_workspace,
                generation,
                organism->pool,
                organism->fitness_history,
                organism->coherence_history,
                workspace_genomes,
                organism->trace_buffer,
                arch.grid_size
            );
            CUDA_LAUNCH_CHECK();
            // Neural CA sync handled by parent after this kernel returns
        }
    }

    // PER-ENTRY operations - each block handles its own entry in parallel
    SLIME_DEBUG_PRINT("V:hybrid:873 pre_per_entry entry_idx=%d\n", entry_idx);
    if (tid == 0 && training_mode->use_gradients) {
        SLIME_DEBUG_PRINT("V:hybrid:875 per_entry_start entry_idx=%d\n", entry_idx);
        dim3 update_grid(field_grid.x, field_grid.y, 1);
        update_field_from_ca_kernel<<<update_grid, field_block>>>(
            pool,
            organism->chemical_field->concentration,
            arch.grid_size,
            entry_idx
        );
        CUDA_LAUNCH_CHECK();

        int weight_count = arch.num_heads * arch.channels * arch.hidden_dim;
        int convert_threads = BLOCK_SIZE;
        int convert_blocks = (weight_count + convert_threads - 1) / convert_threads;
        convert_weights_to_fp32<<<convert_blocks, convert_threads>>>(
            ca_state->perception_weights,
            ca_state->fp32_workspace,
            weight_count
        );
        CUDA_LAUNCH_CHECK();

        float* temp_latent = primary_parent_temp;
        diresa_encode(primary_genome, temp_latent, &organism->diresa_genome_weights[0]);

        compute_effective_rank_from_latent_kernel<<<1, BLOCK_SIZE>>>(
            pool,
            organism->effective_rank_history,
            workspace_genomes,
            GENOME_LATENT_DIM_MAX,
            entry_idx
        );
        CUDA_LAUNCH_CHECK();
    }

    float* behavioral_workspace_genomes = organism->buffers->behavioral_workspace_genomes_buffer;

    // behavioral_update_kernel removed - already called in Phase 3 of persistent_evolution_kernel

    // Embedding update only needs to run once, not per-entry
    if (tid == 0 && entry_idx == 0 && generation % EMBEDDING_UPDATE_FREQ == 0) {
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
            organism->buffers->behavioral_features_buffer,
            organism->chemical_field->concentration,
            arch.grid_size
        );
        CUDA_LAUNCH_CHECK();
    }

    // Memory update - only from block 0 to avoid redundant launches
    if (tid == 0 && entry_idx == 0) {
        uint64_t mem_genome_hash = entry->genome_hash;
        float ctx_metabolic = entry->fitness;
        float ctx_stress = entry->hunger;

        float ctx_morphogen = local_ca_mean;

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
        CUDA_LAUNCH_CHECK();
        SLIME_DEBUG_PRINT("V:hybrid:973 post_memory_update\n");

        // Child kernel sync handled by parent's cudaDeviceSynchronize after this kernel returns
    }  // end if (tid == 0 && entry_idx == 0)

    // NOTE: No __syncthreads here - it would deadlock since entry 0's tid=0
    // is blocked at cudaDeviceSynchronize while other threads wait at syncthreads
    SLIME_DEBUG_PRINT("V:hybrid:983 kernel_exit entry_idx=%d tid=%d\n", entry_idx, tid);
}


#endif

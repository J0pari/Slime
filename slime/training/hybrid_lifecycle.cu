
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

    // Cache entry->alive in shared memory to ensure all warps see same value
    // Prevents race where different warps read different cached global memory values
    __shared__ bool s_entry_alive;
    if (tid == 0) s_entry_alive = entry->alive;
    __syncthreads();
    if (!s_entry_alive) return;

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
    // Cache use_gradients to ensure all warps take same branch (contains __syncthreads)
    __shared__ bool s_use_gradients;
    if (tid == 0) {
        s_error_flag = 0;
        s_use_gradients = training_mode->use_gradients;
    }
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

    // Initialize current_activation_grid_size on first run (tid 0 only to avoid race)
    if (tid == 0 && organism->current_activation_grid_size == 0) {
        organism->current_activation_grid_size = arch.grid_size;
    }

    // Update if grid size changed (tid 0 only to avoid race)
    if (tid == 0 && arch.grid_size != organism->current_activation_grid_size) {
        organism->current_activation_grid_size = arch.grid_size;
    }

    if (s_use_gradients) {
        // Error checks - tid==0 sets shared flag, then broadcast
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
        }
        __syncthreads();
        if (s_error_flag) return;  // All threads exit together

        // ========== SAMPLE_BATCH (replaced CDP launch) ==========
        // Global operation: only entry_idx==0 block processes, ALL threads participate
        if (entry_idx == 0) {
            Dataset* dataset = organism->current_dataset;
            unsigned char* all_images = dataset->samples;
            unsigned char* all_labels = dataset->labels;
            float* batch_images_out = training_mode->batch_images;
            int* batch_labels_out = training_mode->batch_labels;
            int dataset_size = dataset->num_samples;
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int offset = generation * batch_size;
            int sample_rows = dataset->descriptor->sample_rows;
            int sample_cols = dataset->descriptor->sample_cols;
            int sample_channels = dataset->descriptor->channels;
            int sample_size = sample_rows * sample_cols * sample_channels;

            // Thread 0 writes all batch labels (small loop)
            if (tid == 0) {
                for (int idx = 0; idx < batch_size; idx++) {
                    int src_idx = (offset + idx) % dataset_size;
                    batch_labels_out[idx] = all_labels[src_idx];
                }
            }

            // All threads do bilinear interpolation via thread loop
            // Total work: batch_size × grid_size² pixels
            int total_pixels = batch_size * grid_size * grid_size;
            for (int work_idx = tid; work_idx < total_pixels; work_idx += blockDim.x) {
                int idx = work_idx / (grid_size * grid_size);  // batch index
                int pixel_idx = work_idx % (grid_size * grid_size);  // pixel within sample

                int src_idx = (offset + idx) % dataset_size;
                int out_y = pixel_idx / grid_size;
                int out_x = pixel_idx % grid_size;

                float src_y = out_y * (float)sample_rows / grid_size;
                float src_x = out_x * (float)sample_cols / grid_size;

                int y0 = (int)src_y;
                int x0 = (int)src_x;
                int y1 = min(y0 + 1, sample_rows - 1);
                int x1 = min(x0 + 1, sample_cols - 1);

                float fy = src_y - y0;
                float fx = src_x - x0;

                float p00 = all_images[src_idx * sample_size + y0 * sample_cols + x0] / 255.0f;
                float p01 = all_images[src_idx * sample_size + y0 * sample_cols + x1] / 255.0f;
                float p10 = all_images[src_idx * sample_size + y1 * sample_cols + x0] / 255.0f;
                float p11 = all_images[src_idx * sample_size + y1 * sample_cols + x1] / 255.0f;

                float value = p00 * (1 - fx) * (1 - fy) +
                             p01 * fx * (1 - fy) +
                             p10 * (1 - fx) * fy +
                             p11 * fx * fy;

                batch_images_out[idx * grid_size * grid_size + pixel_idx] = value;
            }
        }
        __syncthreads();  // Ensure sample_batch complete before continuing
        SLIME_DEBUG_PRINT("V:hybrid:sample_batch_complete entry=%d tid=%d\n", entry_idx, tid);

        // ========== RESET_TAPE (replaced CDP launch) ==========
        // Per-entry operation: ALL threads in ALL blocks participate
        {
            ADTape* tape = &ca_state->tape;
            int tape_capacity = tape->value_capacity;

            // All threads zero grad_buffer and value_levels via thread loop
            for (int i = tid; i < tape_capacity; i += blockDim.x) {
                tape->grad_buffer[i] = 0.0f;
                tape->value_levels[i] = 0;
            }

            // Thread 0 zeros scalar fields
            if (tid == 0) {
                tape->current_size = 0;
                tape->current_value_idx = 0;
                tape->max_level = 0;
            }
        }
        __syncthreads();  // Ensure reset_tape complete before continuing
        SLIME_DEBUG_PRINT("V:hybrid:reset_tape_complete entry=%d tid=%d\n", entry_idx, tid);

        // ========== INJECT_SAMPLE_TO_CA (replaced CDP launch) ==========
        // Global operation: only entry_idx==0 block processes, ALL threads participate
        if (entry_idx == 0) {
            float* ca_out = organism->batch_ca_states_pool;
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels_out = arch.channels;
            int image_channels = (int)organism->current_dataset->descriptor->channels;

            // Source fields (shared across batches)
            float* chem_concentration = organism->chemical_field->concentration;
            float* chem_gradient_x = organism->chemical_field->gradient_x;
            float* chem_gradient_y = organism->chemical_field->gradient_y;
            float* chem_laplacian = organism->chemical_field->laplacian;
            float* chem_sources = organism->chemical_field->sources;
            float* chem_decay_factors = organism->chemical_field->decay_factors;
            float* rd_resource_density = organism->resource_density;
            float* rd_fitness_landscape = organism->fitness_landscape;
            float* rd_resource_gradient_x = organism->resource_gradient_x;
            float* rd_resource_gradient_y = organism->resource_gradient_y;
            float* behavioral_field = organism->behavioral_field_pool;
            float* batch_images = training_mode->batch_images;
            float* prev_concentration = organism->buffers->batch_prev_concentration;
            float* attractor_field = organism->chemical_field->concentration;

            // Total work: batch_size × grid_size² spatial positions
            int total_positions = batch_size * grid_size * grid_size;
            for (int work_idx = tid; work_idx < total_positions; work_idx += blockDim.x) {
                int batch_idx = work_idx / (grid_size * grid_size);
                int spatial_idx = work_idx % (grid_size * grid_size);

                int base_idx = batch_idx * grid_size * grid_size * channels_out + spatial_idx * channels_out;

                // Channel 0-5: ChemicalField (shared across batches)
                ca_out[base_idx + 0] = chem_concentration[spatial_idx];
                ca_out[base_idx + 1] = chem_gradient_x[spatial_idx];
                ca_out[base_idx + 2] = chem_gradient_y[spatial_idx];
                ca_out[base_idx + 3] = chem_laplacian[spatial_idx];
                ca_out[base_idx + 4] = chem_sources[spatial_idx];
                ca_out[base_idx + 5] = chem_decay_factors[spatial_idx];

                // Channel 6-9: RD fields (shared across batches)
                ca_out[base_idx + 6] = rd_resource_density[spatial_idx];
                ca_out[base_idx + 7] = rd_fitness_landscape[spatial_idx];
                ca_out[base_idx + 8] = rd_resource_gradient_x[spatial_idx];
                ca_out[base_idx + 9] = rd_resource_gradient_y[spatial_idx];

                // Channel 10: Behavioral field (shared across batches)
                ca_out[base_idx + 10] = behavioral_field[spatial_idx];

                // Channel 11-13: Dataset sample (per-batch, replicate channel 0 if grayscale)
                int img_base = batch_idx * grid_size * grid_size * image_channels;
                ca_out[base_idx + 11] = batch_images[img_base + spatial_idx];
                ca_out[base_idx + 12] = batch_images[img_base + ((image_channels > 1) ? grid_size * grid_size : 0) + spatial_idx];
                ca_out[base_idx + 13] = batch_images[img_base + ((image_channels > 2) ? 2 * grid_size * grid_size : 0) + spatial_idx];

                // Channel 14: Previous concentration (recurrence) - channel 0 from previous step
                int prev_idx = batch_idx * grid_size * grid_size * channels_out + spatial_idx * channels_out;
                ca_out[base_idx + 14] = prev_concentration[prev_idx + 0];

                // Channel 15: Temporal/attractor retrieval (shared across batches)
                ca_out[base_idx + 15] = attractor_field[spatial_idx];
            }
        }
        __syncthreads();  // Ensure inject_sample_to_ca complete before continuing
        SLIME_DEBUG_PRINT("V:hybrid:inject_sample_complete entry=%d tid=%d\n", entry_idx, tid);

        // ========== MULTI_HEAD_CA (replaced CDP launch, register-based) ==========
        // Global operation: only entry_idx==0 block processes, ALL threads participate
        // Total work: batch_size × num_heads × grid_size² cells
        if (entry_idx == 0) {
            float* ca_input = organism->batch_ca_states_pool;
            float* ca_output = organism->buffers->batched_ca_output;
            float* perception_saved = ca_state->perception_saved;
            float* interaction_saved = ca_state->interaction_saved;
            float* pre_gelu_saved = ca_state->pre_gelu_saved;

            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads = arch.num_heads;
            int head_dim = arch.head_dim;
            int cells_per_grid = grid_size * grid_size;

            // Total cells to process: batch × heads × grid²
            int total_cells = batch_size * num_heads * cells_per_grid;

            for (int work_idx = tid; work_idx < total_cells; work_idx += blockDim.x) {
                // Decode work index → batch_id, head_id, cell_x, cell_y
                int cells_per_batch_head = cells_per_grid;
                int heads_times_cells = num_heads * cells_per_grid;

                int batch_id = work_idx / heads_times_cells;
                int remainder = work_idx % heads_times_cells;
                int head_id = remainder / cells_per_grid;
                int cell_idx = remainder % cells_per_grid;
                int cell_y = cell_idx / grid_size;
                int cell_x = cell_idx % grid_size;

                // Get weight pointers for this head
                half* perc_w = &ca_state->perception_weights[head_id * channels * head_dim];
                half* inter_w = &ca_state->interaction_weights[head_id * head_dim * head_dim];
                half* val_w = &ca_state->value_weights[head_id * head_dim * channels];

                // Load 3×3 neighborhood into registers (register-based, no shared memory)
                float neighborhood[3][3][MAX_CHANNELS];
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = min(max(cell_x + dx, 0), grid_size - 1);
                        int ny = min(max(cell_y + dy, 0), grid_size - 1);
                        int state_idx = batch_id * cells_per_grid * channels +
                                       ny * grid_size * channels +
                                       nx * channels;
                        for (int c = 0; c < channels; c++) {
                            neighborhood[dy + 1][dx + 1][c] = ca_input[state_idx + c];
                        }
                    }
                }

                // Perception: conv over 3×3 neighborhood → head_dim outputs, ReLU
                float perception[MAX_HEAD_DIM];
                for (int h = 0; h < head_dim; h++) {
                    float acc = 0.0f;
                    for (int dy = 0; dy < 3; dy++) {
                        for (int dx = 0; dx < 3; dx++) {
                            for (int c = 0; c < channels; c++) {
                                acc += neighborhood[dy][dx][c] * __half2float(perc_w[c * head_dim + h]);
                            }
                        }
                    }
                    perception[h] = fmaxf(0.0f, acc);  // ReLU
                }

                // Interaction: linear + GELU
                float interaction[MAX_HEAD_DIM];
                float interaction_sum = 0.0f;
                for (int h = 0; h < head_dim; h++) {
                    float acc = 0.0f;
                    for (int j = 0; j < head_dim; j++) {
                        acc += perception[j] * __half2float(inter_w[j * head_dim + h]);
                    }
                    // GELU activation
                    float x = acc;
                    float gelu = GELU_SCALE * x * (GELU_OFFSET + tanhf(GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC_COEFFICIENT * x * x * x)));
                    interaction[h] = gelu;
                    interaction_sum += fabsf(gelu);
                }

                // Value projection: interaction → channels
                float output[MAX_CHANNELS];
                for (int c = 0; c < channels; c++) {
                    float acc = 0.0f;
                    for (int h = 0; h < head_dim; h++) {
                        acc += interaction[h] * __half2float(val_w[h * channels + c]);
                    }
                    output[c] = acc;
                }

                // Gating: sigmoid based on interaction magnitude
                float gate = 1.0f / (1.0f + expf(-(interaction_sum / (float)head_dim - arch.ca_gate_center)));

                // Save activations for backward pass
                int saved_base = batch_id * num_heads * cells_per_grid * head_dim +
                                head_id * cells_per_grid * head_dim +
                                cell_y * grid_size * head_dim +
                                cell_x * head_dim;
                for (int h = 0; h < head_dim; h++) {
                    perception_saved[saved_base + h] = perception[h];
                    interaction_saved[saved_base + h] = interaction[h];
                    // pre_gelu_saved stores the pre-activation value (before GELU)
                    // Recompute since we need it for backward
                    float pre_gelu = 0.0f;
                    for (int j = 0; j < head_dim; j++) {
                        pre_gelu += perception[j] * __half2float(inter_w[j * head_dim + h]);
                    }
                    pre_gelu_saved[saved_base + h] = pre_gelu;
                }

                // Write gated output: blend input center with transformed output
                int out_idx = batch_id * num_heads * cells_per_grid * channels +
                             head_id * cells_per_grid * channels +
                             cell_y * grid_size * channels +
                             cell_x * channels;
                for (int c = 0; c < channels; c++) {
                    float input_val = neighborhood[1][1][c];  // Center cell
                    ca_output[out_idx + c] = input_val * (1.0f - gate) + output[c] * gate;
                }
            }
        }
        __syncthreads();  // Ensure multi_head_ca complete before continuing
        SLIME_DEBUG_PRINT("V:hybrid:multi_head_ca_complete entry=%d tid=%d\n", entry_idx, tid);

        // Remaining CDP launches still inside tid==0 (to be transformed)
        if (tid == 0) {
        SLIME_DEBUG_PRINT("V:hybrid:post_ca tid0_section\n");
        }  // end if (tid == 0) - placeholder for remaining transforms

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
                organism->buffers->batch_affinity_reduced[batch_idx * total_cells + cell_idx] = affinity;
            }
            __syncthreads();

            for (int work_idx = tid; work_idx < total_affinity_work; work_idx += blockDim.x) {
                int batch_idx = work_idx / total_cells;
                int cell_idx = work_idx % total_cells;
                int x = cell_idx % grid_size;
                int y = cell_idx / grid_size;

                int batch_offset = batch_idx * total_cells;
                float U_center = organism->buffers->batch_affinity_reduced[batch_offset + cell_idx];
                int x_E = min(x + 1, grid_size - 1);
                int y_N = min(y + 1, grid_size - 1);
                float U_E = organism->buffers->batch_affinity_reduced[batch_offset + y * grid_size + x_E];
                float U_N = organism->buffers->batch_affinity_reduced[batch_offset + y_N * grid_size + x];

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
                organism->buffers->batch_flow_field[flow_idx + 0] = F.x;
                organism->buffers->batch_flow_field[flow_idx + 1] = F.y;
            }

            // Clear reintegration buffer
            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                organism->buffers->batch_reintegration_buffer[idx] = 0.0f;
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
                float Fx = organism->buffers->batch_flow_field[flow_idx + 0];
                float Fy = organism->buffers->batch_flow_field[flow_idx + 1];

                int conc_batch_offset = batch_idx * total_cells * channels;
                float* batch_buffer = organism->buffers->batch_reintegration_buffer + conc_batch_offset;
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
                organism->batch_ca_states_pool[idx] = organism->buffers->batch_reintegration_buffer[idx];
            }
            __syncthreads();

            // Phase 6: Save concentration for next iteration's recurrence channel
            // batch_prev_concentration will be read by inject_sample_to_ca_kernel on next step
            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                organism->buffers->batch_prev_concentration[idx] = organism->batch_ca_states_pool[idx];
            }
        }  // end Flow Lenia
        __syncthreads();

        SLIME_DEBUG_PRINT("V:hybrid:412 post_flow_lenia tid=%d\n", tid);

        // ========== FORWARD PASS (replaced CDP launches) ==========
        // Global operations: only entry_idx==0 block processes, ALL threads participate
        float* ca_output_grad = nullptr;

        if (entry_idx == 0 && training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
            float* features = organism->gradient_features_pool;
            float* ca_out = organism->buffers->batched_ca_output;
            float* pooling_weights = training_mode->classifier->pooling_weights;
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int num_features = num_heads_local * channels;
            int spatial_size = grid_size * grid_size;

            // ========== SPATIAL_POOLING (replaced CDP launch) ==========
            // Total work: batch_size × num_features
            int total_pool_work = batch_size * num_features;
            for (int work_idx = tid; work_idx < total_pool_work; work_idx += blockDim.x) {
                int batch_idx = work_idx / num_features;
                int feature_idx = work_idx % num_features;
                int head = feature_idx / channels;
                int channel = feature_idx % channels;

                float sum = 0.0f;
                int batch_stride = num_heads_local * spatial_size * channels;
                int head_stride = spatial_size * channels;
                int base_idx = batch_idx * batch_stride + head * head_stride;

                for (int spatial = 0; spatial < spatial_size; spatial++) {
                    int idx = base_idx + spatial * channels + channel;
                    sum += ca_out[idx];
                }

                float avg = sum / spatial_size;
                float weight = pooling_weights[feature_idx];
                float weighted = avg * weight;

                if (!isnan(weighted) && !isinf(weighted)) {
                    features[batch_idx * num_features + feature_idx] = weighted;
                }
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:spatial_pooling_complete tid=%d\n", tid);

            // ========== CLASSIFICATION_HEAD (replaced CDP launch) ==========
            float* logits = organism->gradient_logits_pool;
            float* fc_weights = training_mode->classifier->fc_weights;
            float* fc_bias = training_mode->classifier->fc_bias;

            // Total work: batch_size × num_classes
            int total_class_work = batch_size * num_classes;
            for (int work_idx = tid; work_idx < total_class_work; work_idx += blockDim.x) {
                int batch_idx = work_idx / num_classes;
                int class_idx = work_idx % num_classes;

                float acc = fc_bias[class_idx];
                for (int feat = 0; feat < num_features; feat++) {
                    float feature_val = features[batch_idx * num_features + feat];
                    float weight = fc_weights[class_idx * num_features + feat];
                    acc += feature_val * weight;
                }

                if (!isnan(acc) && !isinf(acc)) {
                    logits[batch_idx * num_classes + class_idx] = acc;
                }
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:classification_head_complete tid=%d\n", tid);

            // ========== ZERO_SCALAR (replaced CDP launch) ==========
            float* loss_out = organism->gradient_loss_pool;
            if (tid == 0) {
                *loss_out = 0.0f;
            }
            __syncthreads();

            // ========== CROSS_ENTROPY_LOSS (replaced CDP launch) ==========
            // Process samples sequentially, each using warp 0 for reduction
            float* logit_grads = organism->gradient_logit_grads_pool;
            int* batch_labels = training_mode->batch_labels;

            // Each warp processes one sample at a time, iterate over all samples
            int warp_id = tid / WARP_SIZE;
            int lane_id = tid % WARP_SIZE;
            int num_warps = blockDim.x / WARP_SIZE;

            for (int sample_base = 0; sample_base < batch_size; sample_base += num_warps) {
                int batch_idx = sample_base + warp_id;
                if (batch_idx < batch_size) {
                    int label = batch_labels[batch_idx];
                    float* batch_logits = &logits[batch_idx * num_classes];
                    float* batch_grads = &logit_grads[batch_idx * num_classes];

                    if (label >= 0 && label < num_classes) {
                        // Warp-level max reduction
                        float local_val = (lane_id < num_classes) ? batch_logits[lane_id] : -INFINITY;
                        float max_logit = warp_reduce_max(local_val);
                        max_logit = __shfl_sync(0xffffffff, max_logit, 0);

                        // Warp-level sum reduction for exp
                        float local_exp = (lane_id < num_classes) ? expf(local_val - max_logit) : 0.0f;
                        float sum_exp = warp_reduce_sum(local_exp);
                        sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

                        // Compute softmax and gradients
                        if (lane_id < num_classes) {
                            float prob = local_exp / sum_exp;
                            float grad = prob - ((lane_id == label) ? 1.0f : 0.0f);
                            batch_grads[lane_id] = grad / batch_size;
                        }

                        // Lane 0 adds to loss
                        if (lane_id == 0) {
                            float log_sum_exp = logf(sum_exp) + max_logit;
                            float nll = log_sum_exp - batch_logits[label];
                            if (!isnan(nll) && !isinf(nll) && nll >= 0.0f) {
                                atomicAdd(loss_out, nll / batch_size);
                            }
                        }
                    }
                }
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:cross_entropy_complete tid=%d\n", tid);

            // ========== TASK_PERFORMANCE_PROBE (replaced CDP launch) ==========
            // Thread 0 computes accuracy metrics
            if (tid == 0) {
                int correct = 0;
                float avg_confidence = 0.0f;

                for (int b = 0; b < batch_size; b++) {
                    float* batch_logits = &logits[b * num_classes];
                    int true_label = batch_labels[b];

                    // Find predicted class and max logit
                    int pred_class = 0;
                    float max_logit = batch_logits[0];
                    for (int c = 1; c < num_classes; c++) {
                        if (batch_logits[c] > max_logit) {
                            max_logit = batch_logits[c];
                            pred_class = c;
                        }
                    }

                    if (pred_class == true_label) correct++;

                    // Compute softmax for confidence
                    float sum_exp = 0.0f;
                    for (int c = 0; c < num_classes; c++) {
                        sum_exp += expf(batch_logits[c] - max_logit);
                    }
                    float confidence = expf(batch_logits[pred_class] - max_logit) / sum_exp;
                    avg_confidence += confidence;
                }

                float accuracy = (float)correct / batch_size;
                avg_confidence /= batch_size;

                // Update telemetry
                if (training_mode->is_train_batch) {
                    organism->telemetry->task_performance.train_accuracy =
                        0.9f * organism->telemetry->task_performance.train_accuracy + 0.1f * accuracy;
                } else {
                    organism->telemetry->task_performance.accuracy = accuracy;
                }
                organism->telemetry->task_performance.avg_confidence = avg_confidence;
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:task_performance_complete tid=%d\n", tid);
        }
        __syncthreads();  // All entries sync before backward pass

        // ========== BACKWARD PASS (replaced CDP launches) ==========
        // Global operations: only entry_idx==0 block processes, ALL threads participate
        if (entry_idx == 0 && training_mode->batch_images != nullptr && training_mode->classifier != nullptr && !eval_only) {
            SLIME_DEBUG_PRINT("V:hybrid:enter_backward eval_only=%d\n", eval_only);

            // Re-access variables for backward pass
            float* features = organism->gradient_features_pool;
            float* logit_grads = organism->gradient_logit_grads_pool;
            float* fc_weights = training_mode->classifier->fc_weights;
            float* fc_weights_grad = organism->fc_weights_grad;
            float* fc_bias_grad = organism->fc_bias_grad;
            float* features_grad = organism->features_grad;
            float* pooling_weights = training_mode->classifier->pooling_weights;
            float* pooling_weights_grad = organism->pooling_weights_grad;
            float* ca_out = organism->buffers->batched_ca_output;
            ca_output_grad = organism->buffers->ca_output_grad_buffer;

            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int num_features = num_heads_local * channels;
            int spatial_size = grid_size * grid_size;

            // ========== ZERO GRADIENT BUFFERS (replaced 5 CDP launches) ==========
            int fc_weights_size = num_classes * channels;
            int ca_grad_size = batch_size * num_heads_local * spatial_size * channels;

            // Zero all gradient buffers via thread loop
            for (int i = tid; i < fc_weights_size; i += blockDim.x) {
                fc_weights_grad[i] = 0.0f;
            }
            for (int i = tid; i < num_classes; i += blockDim.x) {
                fc_bias_grad[i] = 0.0f;
            }
            for (int i = tid; i < num_features; i += blockDim.x) {
                features_grad[i] = 0.0f;
                pooling_weights_grad[i] = 0.0f;
            }
            for (int i = tid; i < ca_grad_size; i += blockDim.x) {
                ca_output_grad[i] = 0.0f;
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:zero_grads_complete tid=%d\n", tid);

            // ========== CLASSIFICATION_HEAD_BACKWARD (replaced CDP launch) ==========
            // Total work: batch_size × num_classes, uses atomicAdd for gradient accumulation
            int total_class_bwd = batch_size * num_classes;
            for (int work_idx = tid; work_idx < total_class_bwd; work_idx += blockDim.x) {
                int batch_idx = work_idx / num_classes;
                int class_idx = work_idx % num_classes;

                float logit_grad = logit_grads[batch_idx * num_classes + class_idx];

                // Bias gradient
                atomicAdd(&fc_bias_grad[class_idx], logit_grad);

                // Weight and feature gradients
                for (int feat = 0; feat < num_features; feat++) {
                    float feature_val = features[batch_idx * num_features + feat];
                    float weight_val = fc_weights[class_idx * num_features + feat];
                    atomicAdd(&fc_weights_grad[class_idx * num_features + feat], logit_grad * feature_val);
                    atomicAdd(&features_grad[batch_idx * num_features + feat], logit_grad * weight_val);
                }
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:class_backward_complete tid=%d\n", tid);

            // ========== SPATIAL_POOLING_BACKWARD (replaced CDP launch) ==========
            // Total work: batch_size × num_features
            int total_pool_bwd = batch_size * num_features;
            for (int work_idx = tid; work_idx < total_pool_bwd; work_idx += blockDim.x) {
                int batch_idx = work_idx / num_features;
                int feature_idx = work_idx % num_features;
                int head = feature_idx / channels;
                int channel = feature_idx % channels;

                float feat_grad = features_grad[batch_idx * num_features + feature_idx];

                if (!isnan(feat_grad) && !isinf(feat_grad)) {
                    int batch_stride = num_heads_local * spatial_size * channels;
                    int head_stride = spatial_size * channels;
                    int base_idx = batch_idx * batch_stride + head * head_stride;

                    // Compute ca_avg for pooling weight grad
                    float ca_avg = 0.0f;
                    for (int spatial = 0; spatial < spatial_size; spatial++) {
                        int idx = base_idx + spatial * channels + channel;
                        ca_avg += ca_out[idx];
                    }
                    ca_avg /= spatial_size;

                    atomicAdd(&pooling_weights_grad[feature_idx], feat_grad * ca_avg);

                    // Distribute gradient to ca_output_grad
                    float ca_grad_val = feat_grad * pooling_weights[feature_idx] / spatial_size;
                    for (int spatial = 0; spatial < spatial_size; spatial++) {
                        int idx = base_idx + spatial * channels + channel;
                        atomicAdd(&ca_output_grad[idx], ca_grad_val);
                    }
                }
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:pooling_backward_complete tid=%d\n", tid);
        }

        // ========== CA BACKWARD PASS (replaced CDP launches) ==========
        // Global operation: only entry_idx==0 block processes, ALL threads participate
        if (entry_idx == 0 && !eval_only) {
            SLIME_DEBUG_PRINT("V:hybrid:enter_ca_backward\n");
            float* dL_dperception = organism->buffers->dL_dperception_buffer;
            float* dL_dinteraction = organism->buffers->dL_dinteraction_buffer;

            {

            // Use preallocated workspace buffers for GEMM-based backward pass
            int num_cells = arch.grid_size * arch.grid_size;
            int total_samples = training_mode->batch_size * num_cells;

            half* ws_fp16_a = organism->buffers->backward_ws_fp16_a;
            half* ws_fp16_b = organism->buffers->backward_ws_fp16_b;
            float* ws_dW = organism->buffers->backward_ws_dW;
            float* ws_dI = organism->buffers->backward_ws_dI;
            half* ws_W_T = organism->buffers->backward_ws_W_T;
            float* ws_im2col = organism->buffers->backward_ws_im2col;
            float* ws_dpregelu = organism->buffers->backward_ws_dpregelu;

            // Stride calculations for backward pass
            // Saved activations layout: [batch][head][cell][head_dim]
            int I_head_stride = num_cells * arch.head_dim;  // interaction/perception saved
            int I_batch_stride = arch.num_heads * I_head_stride;
            // CA output layout: [batch][head][cell][channels]
            int V_head_stride = num_cells * arch.channels;  // ca_output_grad
            int V_batch_stride = arch.num_heads * V_head_stride;
            // Workspace strides (reused for value and interaction backward)
            int ws_fp16_a_stride = total_samples * arch.head_dim;  // interaction_saved FP16
            int ws_fp16_b_stride = total_samples * arch.channels;  // ca_output_grad FP16 (value) / dpregelu FP16 (interaction)
            // Value weight gradients: [head_dim × channels]
            int ws_dW_value_stride = arch.head_dim * arch.channels;
            int ws_W_T_value_stride = arch.channels * arch.head_dim;
            // Interaction weight gradients: [head_dim × head_dim]
            int ws_dW_interaction_stride = arch.head_dim * arch.head_dim;
            int ws_W_T_interaction_stride = arch.head_dim * arch.head_dim;
            // Gradient to interaction output
            int ws_dI_stride = total_samples * arch.head_dim;

            // Warp-level identifiers for WMMA operations
            int warp_id = tid / WARP_SIZE;
            int num_warps = blockDim.x / WARP_SIZE;

            // === VALUE BACKWARD (replaced CDP launches) ===
            // Convert interaction_saved to FP16 (thread loop)
            {
                int total_I = arch.num_heads * total_samples * arch.head_dim;
                for (int idx = tid; idx < total_I; idx += blockDim.x) {
                    int head_id = idx / (training_mode->batch_size * I_head_stride);
                    int remainder = idx % (training_mode->batch_size * I_head_stride);
                    int batch_id = remainder / I_head_stride;
                    int local_idx = remainder % I_head_stride;
                    int src_idx = head_id * I_head_stride + batch_id * I_batch_stride + local_idx;
                    int dst_idx = head_id * ws_fp16_a_stride + batch_id * I_head_stride + local_idx;
                    ws_fp16_a[dst_idx] = __float2half(ca_state->interaction_saved[src_idx]);
                }
            }
            __syncthreads();

            // Convert ca_output_grad to FP16 (thread loop)
            // ca_output layout: [batch][head][cell][channels]
            {
                int total_V = arch.num_heads * total_samples * arch.channels;
                for (int idx = tid; idx < total_V; idx += blockDim.x) {
                    int head_id = idx / (training_mode->batch_size * V_head_stride);
                    int remainder = idx % (training_mode->batch_size * V_head_stride);
                    int batch_id = remainder / V_head_stride;
                    int local_idx = remainder % V_head_stride;
                    int src_idx = head_id * V_head_stride + batch_id * V_batch_stride + local_idx;
                    int dst_idx = head_id * ws_fp16_b_stride + batch_id * V_head_stride + local_idx;
                    ws_fp16_b[dst_idx] = __float2half(ca_output_grad[src_idx]);
                }
            }
            __syncthreads();

            // GEMM transA: dW_value = interaction^T @ ca_output_grad
            // Value weights: [head_dim × channels] per head
            // A (transposed): interaction_saved [total_samples × head_dim]
            // B: ca_output_grad [total_samples × channels]
            // C: dW_value [head_dim × channels]
            {
                int tiles_M = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int tiles_N = (arch.channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int total_tiles = tiles_M * tiles_N * arch.num_heads;

                for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                    int head_id = tile_idx / (tiles_M * tiles_N);
                    int tile_flat = tile_idx % (tiles_M * tiles_N);
                    int warpM = tile_flat / tiles_N;
                    int warpN = tile_flat % tiles_N;

                    int tile_row = warpM * WMMA_TILE_DIM;
                    int tile_col = warpN * WMMA_TILE_DIM;

                    if (tile_row < arch.head_dim && tile_col < arch.channels) {
                        const half* A_head = ws_fp16_a + head_id * ws_fp16_a_stride;
                        const half* B_head = ws_fp16_b + head_id * ws_fp16_b_stride;
                        float* C_head = ws_dW + head_id * ws_dW_value_stride;

                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

                        nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                        for (int k_tile = 0; k_tile < total_samples; k_tile += WMMA_TILE_DIM) {
                            if (k_tile + WMMA_TILE_DIM <= total_samples) {
                                nvcuda::wmma::load_matrix_sync(a_frag, A_head + k_tile * arch.head_dim + tile_row, arch.head_dim);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + k_tile * arch.channels + tile_col, arch.channels);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                        }
                        nvcuda::wmma::store_matrix_sync(C_head + tile_row * arch.channels + tile_col, c_frag, arch.channels, nvcuda::wmma::mem_row_major);
                    }
                }
            }
            __syncthreads();

            // Accumulate value weight gradients (thread loop)
            // Value weights: [head_dim × channels] per head
            {
                int total_grads = arch.num_heads * arch.head_dim * arch.channels;
                for (int idx = tid; idx < total_grads; idx += blockDim.x) {
                    int head_id = idx / (arch.head_dim * arch.channels);
                    int local_idx = idx % (arch.head_dim * arch.channels);
                    int src_idx = head_id * ws_dW_value_stride + local_idx;
                    int dst_idx = param_map->value_start[head_id] + local_idx;
                    ca_state->tape.grad_buffer[dst_idx] = ws_dW[src_idx];
                }
            }
            __syncthreads();

            // Transpose value weights: W [head_dim × channels] -> W^T [channels × head_dim]
            {
                int t_tiles_x = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int t_tiles_y = (arch.channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int total_tile_elements = t_tiles_x * t_tiles_y * arch.num_heads * WMMA_TILE_DIM * WMMA_TILE_DIM;

                for (int work_idx = tid; work_idx < total_tile_elements; work_idx += blockDim.x) {
                    int elements_per_tile = WMMA_TILE_DIM * WMMA_TILE_DIM;
                    int tiles_per_head = t_tiles_x * t_tiles_y;
                    int head_id = work_idx / (tiles_per_head * elements_per_tile);
                    int remainder = work_idx % (tiles_per_head * elements_per_tile);
                    int tile_idx = remainder / elements_per_tile;
                    int elem_idx = remainder % elements_per_tile;
                    int tile_x = tile_idx % t_tiles_x;
                    int tile_y = tile_idx / t_tiles_x;
                    int local_x = elem_idx % WMMA_TILE_DIM;
                    int local_y = elem_idx / WMMA_TILE_DIM;

                    int bx = tile_x * WMMA_TILE_DIM;
                    int by = tile_y * WMMA_TILE_DIM;
                    int h = bx + local_x;  // head_dim index
                    int c = by + local_y;  // channel index

                    if (c < arch.channels && h < arch.head_dim) {
                        // W stored as [head_dim × channels]: W[h * channels + c]
                        // W^T stored as [channels × head_dim]: W_T[c * head_dim + h]
                        const half* W_head = ca_state->value_weights + head_id * arch.head_dim * arch.channels;
                        half* W_T_head = ws_W_T + head_id * ws_W_T_value_stride;
                        W_T_head[c * arch.head_dim + h] = W_head[h * arch.channels + c];
                    }
                }
            }
            __syncthreads();

            // GEMM: dI = dV @ W^T (gradient to interaction output)
            // dV: [total_samples × channels], W^T: [channels × head_dim], dI: [total_samples × head_dim]
            {
                int tiles_M = (total_samples + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int tiles_N = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int total_tiles = tiles_M * tiles_N * arch.num_heads;

                for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                    int head_id = tile_idx / (tiles_M * tiles_N);
                    int tile_flat = tile_idx % (tiles_M * tiles_N);
                    int warpM = tile_flat / tiles_N;
                    int warpN = tile_flat % tiles_N;

                    int tile_row = warpM * WMMA_TILE_DIM;
                    int tile_col = warpN * WMMA_TILE_DIM;

                    if (tile_row < total_samples && tile_col < arch.head_dim) {
                        const half* A_head = ws_fp16_b + head_id * ws_fp16_b_stride;  // dV [samples × channels]
                        const half* B_head = ws_W_T + head_id * ws_W_T_value_stride;  // W^T [channels × head_dim]
                        float* C_head = ws_dI + head_id * ws_dI_stride;               // dI [samples × head_dim]

                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> a_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

                        nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                        for (int k_tile = 0; k_tile < arch.channels; k_tile += WMMA_TILE_DIM) {
                            if (k_tile + WMMA_TILE_DIM <= arch.channels) {
                                nvcuda::wmma::load_matrix_sync(a_frag, A_head + tile_row * arch.channels + k_tile, arch.channels);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + k_tile * arch.head_dim + tile_col, arch.head_dim);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                        }
                        nvcuda::wmma::store_matrix_sync(C_head + tile_row * arch.head_dim + tile_col, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                    }
                }
            }
            __syncthreads();

            // Copy dI to dL_dinteraction with strided layout (thread loop)
            {
                int total_copy = arch.num_heads * total_samples * arch.head_dim;
                for (int idx = tid; idx < total_copy; idx += blockDim.x) {
                    int head_id = idx / (training_mode->batch_size * I_head_stride);
                    int remainder = idx % (training_mode->batch_size * I_head_stride);
                    int batch_id = remainder / I_head_stride;
                    int local_idx = remainder % I_head_stride;
                    int src_idx = head_id * ws_dI_stride + batch_id * I_head_stride + local_idx;
                    int dst_idx = head_id * I_head_stride + batch_id * I_batch_stride + local_idx;
                    dL_dinteraction[dst_idx] = ws_dI[src_idx];
                }
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:value_backward_complete tid=%d\n", tid);

            // === INTERACTION BACKWARD (replaced CDP launches) ===
            // Interaction weights are [head_dim × head_dim] per head
            int ws_dpregelu_stride = total_samples * arch.head_dim;

            // GELU backward (thread loop)
            // Input: dL_dinteraction [batch × head × cell × head_dim]
            // Output: ws_dpregelu [total_samples × head_dim] per head
            {
                int total_gelu = arch.num_heads * total_samples * arch.head_dim;
                int elements_per_head = total_samples * arch.head_dim;
                for (int idx = tid; idx < total_gelu; idx += blockDim.x) {
                    int head_id = idx / elements_per_head;
                    int local_idx = idx % elements_per_head;
                    int src_idx = head_id * I_head_stride + local_idx;
                    int dst_idx = head_id * ws_dpregelu_stride + local_idx;

                    float x = ca_state->pre_gelu_saved[src_idx];
                    float x2 = x * x, x3 = x2 * x;
                    float inner = GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC_COEFFICIENT * x3);
                    float tanh_inner = tanhf(inner);
                    float sech2 = 1.0f - tanh_inner * tanh_inner;
                    float d_inner = GELU_SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_CUBIC_COEFFICIENT * x2);

                    ws_dpregelu[dst_idx] = dL_dinteraction[src_idx] * GELU_SCALE *
                        ((GELU_OFFSET + tanh_inner) + x * sech2 * d_inner);
                }
            }
            __syncthreads();

            // Convert perception_saved to FP16 (thread loop)
            {
                int total_conv = arch.num_heads * total_samples * arch.head_dim;
                for (int idx = tid; idx < total_conv; idx += blockDim.x) {
                    int head_id = idx / (training_mode->batch_size * I_head_stride);
                    int remainder = idx % (training_mode->batch_size * I_head_stride);
                    int batch_id = remainder / I_head_stride;
                    int local_idx = remainder % I_head_stride;
                    int src_idx = head_id * I_head_stride + batch_id * I_batch_stride + local_idx;
                    int dst_idx = head_id * ws_fp16_a_stride + batch_id * I_head_stride + local_idx;
                    ws_fp16_a[dst_idx] = __float2half(ca_state->perception_saved[src_idx]);
                }
            }
            __syncthreads();

            // Convert dpregelu to FP16 (thread loop)
            {
                int total_conv = arch.num_heads * total_samples * arch.head_dim;
                for (int idx = tid; idx < total_conv; idx += blockDim.x) {
                    ws_fp16_b[idx] = __float2half(ws_dpregelu[idx]);
                }
            }
            __syncthreads();

            // GEMM transA: dW_interaction = perception^T @ d_pregelu [head_dim × head_dim]
            {
                int dW_tiles = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int total_tiles = dW_tiles * dW_tiles * arch.num_heads;

                for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                    int head_id = tile_idx / (dW_tiles * dW_tiles);
                    int tile_flat = tile_idx % (dW_tiles * dW_tiles);
                    int warpM = tile_flat / dW_tiles;
                    int warpN = tile_flat % dW_tiles;

                    int tile_row = warpM * WMMA_TILE_DIM;
                    int tile_col = warpN * WMMA_TILE_DIM;

                    if (tile_row < arch.head_dim && tile_col < arch.head_dim) {
                        const half* A_head = ws_fp16_a + head_id * ws_fp16_a_stride;
                        const half* B_head = ws_fp16_b + head_id * ws_fp16_a_stride;
                        float* C_head = ws_dW + head_id * ws_dW_interaction_stride;

                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

                        nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                        for (int k_tile = 0; k_tile < total_samples; k_tile += WMMA_TILE_DIM) {
                            if (k_tile + WMMA_TILE_DIM <= total_samples) {
                                nvcuda::wmma::load_matrix_sync(a_frag, A_head + k_tile * arch.head_dim + tile_row, arch.head_dim);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + k_tile * arch.head_dim + tile_col, arch.head_dim);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                        }
                        nvcuda::wmma::store_matrix_sync(C_head + tile_row * arch.head_dim + tile_col, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                    }
                }
            }
            __syncthreads();

            // Accumulate interaction weight gradients (thread loop)
            {
                int total_grads = arch.num_heads * arch.head_dim * arch.head_dim;
                for (int idx = tid; idx < total_grads; idx += blockDim.x) {
                    int head_id = idx / (arch.head_dim * arch.head_dim);
                    int local_idx = idx % (arch.head_dim * arch.head_dim);
                    int src_idx = head_id * ws_dW_interaction_stride + local_idx;
                    int dst_idx = param_map->interaction_start[head_id] + local_idx;
                    ca_state->tape.grad_buffer[dst_idx] = ws_dW[src_idx];
                }
            }
            __syncthreads();

            // Transpose interaction weights: W [head_dim × head_dim] -> W^T [head_dim × head_dim]
            {
                int dW_tiles = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int total_tile_elements = dW_tiles * dW_tiles * arch.num_heads * WMMA_TILE_DIM * WMMA_TILE_DIM;

                for (int work_idx = tid; work_idx < total_tile_elements; work_idx += blockDim.x) {
                    int elements_per_tile = WMMA_TILE_DIM * WMMA_TILE_DIM;
                    int tiles_per_head = dW_tiles * dW_tiles;
                    int head_id = work_idx / (tiles_per_head * elements_per_tile);
                    int remainder = work_idx % (tiles_per_head * elements_per_tile);
                    int tile_idx = remainder / elements_per_tile;
                    int elem_idx = remainder % elements_per_tile;
                    int tile_x = tile_idx % dW_tiles;
                    int tile_y = tile_idx / dW_tiles;
                    int local_x = elem_idx % WMMA_TILE_DIM;
                    int local_y = elem_idx / WMMA_TILE_DIM;

                    int bx = tile_x * WMMA_TILE_DIM;
                    int by = tile_y * WMMA_TILE_DIM;
                    int x = bx + local_x;
                    int y = by + local_y;

                    if (y < arch.head_dim && x < arch.head_dim) {
                        const half* W_head = ca_state->interaction_weights + head_id * arch.head_dim * arch.head_dim;
                        half* W_T_head = ws_W_T + head_id * ws_W_T_interaction_stride;
                        W_T_head[x * arch.head_dim + y] = W_head[y * arch.head_dim + x];
                    }
                }
            }
            __syncthreads();

            // GEMM: dP = d_pregelu @ W_interaction^T [total_samples × head_dim]
            {
                int dW_tiles = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int tiles_M_int = (total_samples + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int total_tiles = tiles_M_int * dW_tiles * arch.num_heads;

                for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                    int head_id = tile_idx / (tiles_M_int * dW_tiles);
                    int tile_flat = tile_idx % (tiles_M_int * dW_tiles);
                    int warpM = tile_flat / dW_tiles;
                    int warpN = tile_flat % dW_tiles;

                    int tile_row = warpM * WMMA_TILE_DIM;
                    int tile_col = warpN * WMMA_TILE_DIM;

                    if (tile_row < total_samples && tile_col < arch.head_dim) {
                        const half* A_head = ws_fp16_b + head_id * ws_fp16_a_stride;
                        const half* B_head = ws_W_T + head_id * ws_W_T_interaction_stride;
                        float* C_head = ws_dI + head_id * ws_dI_stride;

                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> a_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

                        nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                        for (int k_tile = 0; k_tile < arch.head_dim; k_tile += WMMA_TILE_DIM) {
                            if (k_tile + WMMA_TILE_DIM <= arch.head_dim) {
                                nvcuda::wmma::load_matrix_sync(a_frag, A_head + tile_row * arch.head_dim + k_tile, arch.head_dim);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + k_tile * arch.head_dim + tile_col, arch.head_dim);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                        }
                        nvcuda::wmma::store_matrix_sync(C_head + tile_row * arch.head_dim + tile_col, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                    }
                }
            }
            __syncthreads();

            // Copy dI to dL_dperception with strided layout (thread loop)
            {
                int total_copy = arch.num_heads * total_samples * arch.head_dim;
                for (int idx = tid; idx < total_copy; idx += blockDim.x) {
                    int head_id = idx / (training_mode->batch_size * I_head_stride);
                    int remainder = idx % (training_mode->batch_size * I_head_stride);
                    int batch_id = remainder / I_head_stride;
                    int local_idx = remainder % I_head_stride;
                    int src_idx = head_id * ws_dI_stride + batch_id * I_head_stride + local_idx;
                    int dst_idx = head_id * I_head_stride + batch_id * I_batch_stride + local_idx;
                    dL_dperception[dst_idx] = ws_dI[src_idx];
                }
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:interaction_backward_complete tid=%d\n", tid);

            // === PERCEPTION BACKWARD (replaced CDP launches) ===
            // Forward pass: perception[h] = relu(sum_{dy,dx,c} neighborhood[dy][dx][c] * W[c,h])
            // Equivalent to: perception = relu(pooled_input @ W) where pooled_input[cell,c] = sum over 3x3 of input[c]
            // Perception weights: [channels × head_dim] per head (NOT im2col - forward uses spatial sum)
            // Saved perception activations: [batch][head][cell][head_dim]

            // ReLU backward (thread loop)
            int ws_dprerelu_stride = total_samples * arch.head_dim;  // matches saved activation layout
            {
                int total_relu = arch.num_heads * total_samples * arch.head_dim;
                int elements_per_head = total_samples * arch.head_dim;
                for (int idx = tid; idx < total_relu; idx += blockDim.x) {
                    int head_id = idx / elements_per_head;
                    int local_idx = idx % elements_per_head;
                    int src_idx = head_id * I_head_stride + local_idx;
                    int dst_idx = head_id * ws_dprerelu_stride + local_idx;
                    ws_dpregelu[dst_idx] = dL_dperception[src_idx] * ((ca_state->perception_saved[src_idx] > 0.0f) ? 1.0f : 0.0f);
                }
            }
            __syncthreads();

            // Spatial sum pooling: pool 3x3 neighborhoods to get pooled_input [batch × grid² × channels]
            // This matches forward pass which sums over 3x3 before applying weights
            int ws_pooled_stride = total_samples * arch.channels;  // per head (same input pooled for all heads)
            {
                int num_cells_local = arch.grid_size * arch.grid_size;
                int total_pool = training_mode->batch_size * num_cells_local;
                for (int idx = tid; idx < total_pool; idx += blockDim.x) {
                    int batch_id = idx / num_cells_local;
                    int cell_idx = idx % num_cells_local;
                    int cell_y = cell_idx / arch.grid_size;
                    int cell_x = cell_idx % arch.grid_size;

                    int out_row = batch_id * num_cells_local + cell_idx;
                    const float* input_batch = organism->batch_ca_states_pool;

                    // Sum over 3x3 neighborhood for each channel
                    for (int c = 0; c < arch.channels; c++) {
                        float sum = 0.0f;
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                int ny = max(0, min(arch.grid_size - 1, cell_y + dy));
                                int nx = max(0, min(arch.grid_size - 1, cell_x + dx));
                                int input_idx = batch_id * arch.grid_size * arch.grid_size * arch.channels +
                                               ny * arch.grid_size * arch.channels + nx * arch.channels + c;
                                sum += input_batch[input_idx];
                            }
                        }
                        // Store in ws_im2col (reusing buffer, now [total_samples × channels])
                        ws_im2col[out_row * arch.channels + c] = sum;
                    }
                }
            }
            __syncthreads();

            // Convert pooled_input to FP16 (thread loop) - [total_samples × channels]
            {
                int total_conv = total_samples * arch.channels;
                for (int idx = tid; idx < total_conv; idx += blockDim.x) {
                    ws_fp16_a[idx] = __float2half(ws_im2col[idx]);
                }
            }
            __syncthreads();

            // Convert dprerelu to FP16 (thread loop) - [num_heads × total_samples × head_dim]
            {
                int total_conv = arch.num_heads * total_samples * arch.head_dim;
                for (int idx = tid; idx < total_conv; idx += blockDim.x) {
                    ws_fp16_b[idx] = __float2half(ws_dpregelu[idx]);
                }
            }
            __syncthreads();

            // GEMM for perception weight gradients: dW = pooled_input^T @ d_prerelu (per head)
            // pooled_input: [total_samples × channels] (same for all heads)
            // d_prerelu: [total_samples × head_dim] (per head)
            // dW = pooled^T @ d_prerelu = [channels × total_samples] @ [total_samples × head_dim] = [channels × head_dim]
            {
                int ws_dW_perception_stride = arch.channels * arch.head_dim;
                int dW_tiles_c = (arch.channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int dW_tiles_h = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int total_tiles = dW_tiles_c * dW_tiles_h * arch.num_heads;

                for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                    int head_id = tile_idx / (dW_tiles_c * dW_tiles_h);
                    int tile_flat = tile_idx % (dW_tiles_c * dW_tiles_h);
                    int warpM = tile_flat / dW_tiles_h;
                    int warpN = tile_flat % dW_tiles_h;

                    int tile_row = warpM * WMMA_TILE_DIM;  // channels dimension
                    int tile_col = warpN * WMMA_TILE_DIM;  // head_dim dimension

                    if (tile_row < arch.channels && tile_col < arch.head_dim) {
                        // A = pooled_input^T: load from [total_samples × channels] in column-major (transposed)
                        const half* A_ptr = ws_fp16_a;  // pooled [total_samples × channels]
                        const half* B_head = ws_fp16_b + head_id * ws_dprerelu_stride;  // d_prerelu [total_samples × head_dim]
                        float* C_head = ws_dW + head_id * ws_dW_perception_stride;

                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

                        nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                        for (int k_tile = 0; k_tile < total_samples; k_tile += WMMA_TILE_DIM) {
                            if (k_tile + WMMA_TILE_DIM <= total_samples) {
                                // A^T: accessing pooled_input[k_tile:, tile_row:] in col-major = accessing [k,c] with stride=channels
                                nvcuda::wmma::load_matrix_sync(a_frag, A_ptr + k_tile * arch.channels + tile_row, arch.channels);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + k_tile * arch.head_dim + tile_col, arch.head_dim);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                        }
                        nvcuda::wmma::store_matrix_sync(C_head + tile_row * arch.head_dim + tile_col, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                    }
                }
            }
            __syncthreads();

            // Accumulate perception weight gradients (thread loop)
            // Perception weights: [channels × head_dim] per head
            {
                int ws_dW_perception_stride = arch.channels * arch.head_dim;
                int weights_per_head = arch.channels * arch.head_dim;
                int total_grads = arch.num_heads * weights_per_head;
                for (int idx = tid; idx < total_grads; idx += blockDim.x) {
                    int head_id = idx / weights_per_head;
                    int local_idx = idx % weights_per_head;
                    int src_idx = head_id * ws_dW_perception_stride + local_idx;
                    int dst_idx = param_map->perception_start[head_id] + local_idx;
                    ca_state->tape.grad_buffer[dst_idx] = ws_dW[src_idx];
                }
            }
            __syncthreads();

            SLIME_DEBUG_PRINT("V:hybrid:perception_backward_complete tid=%d\n", tid);
            }  // end CA backward scope
        }  // end if (!eval_only) for CA backward
        SLIME_DEBUG_PRINT("V:hybrid:715 post_ca_backward\n");

        // ========== ADAM UPDATES (replaced CDP launches) ==========
        // Global operation: only entry_idx==0 block processes, ALL threads participate
        if (entry_idx == 0 && training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
            // CA weights per type (each has separate Adam state buffers)
            // Perception: [num_heads × channels × head_dim]
            // Interaction: [num_heads × head_dim × head_dim]
            // Value: [num_heads × head_dim × channels]
            int perception_params = arch.num_heads * arch.channels * arch.head_dim;
            int interaction_params = arch.num_heads * arch.head_dim * arch.head_dim;
            int value_params = arch.num_heads * arch.head_dim * arch.channels;

            float ctx_metabolic = entry->fitness;
            float ctx_stress = entry->hunger;
            float ctx_morphogen = local_ca_mean;

            TrainingParams train_params;
            train_params.derive_from_genome_hash(entry->genome_hash);
            float adam_beta1 = train_params.get_adam_beta1(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
            float adam_beta2 = train_params.get_adam_beta2(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
            float adam_epsilon = train_params.get_adam_epsilon(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
            float gradient_clip_norm = train_params.get_gradient_clip_norm(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

            float lr = training_mode->learning_rate;
            int timestep = training_mode->adam_timestep;

            SLIME_DEBUG_PRINT("V:hybrid:adam_fp16_start tid=%d\n", tid);

            // Adam update for CA PERCEPTION weights (FP16) - thread loop
            {
                half* weights_fp16 = ca_state->perception_weights;
                float* gradients = ca_state->tape.grad_buffer;
                float* m = training_mode->adam_m_perception;
                float* v = training_mode->adam_v_perception;

                for (int idx = tid; idx < perception_params; idx += blockDim.x) {
                    float weight = __half2float(weights_fp16[idx]);
                    float g = gradients[idx];

                    if (!isnan(g) && !isinf(g)) {
                        float g_clipped = fmaxf(-gradient_clip_norm, fminf(gradient_clip_norm, g));
                        float m_new = adam_beta1 * m[idx] + (1.0f - adam_beta1) * g_clipped;
                        float v_new = adam_beta2 * v[idx] + (1.0f - adam_beta2) * g_clipped * g_clipped;
                        m[idx] = m_new;
                        v[idx] = v_new;

                        float m_hat = m_new / (1.0f - powf(adam_beta1, (float)(timestep + 1)));
                        float v_hat = v_new / (1.0f - powf(adam_beta2, (float)(timestep + 1)));
                        float update = lr * m_hat / (sqrtf(v_hat) + adam_epsilon);

                        float new_weight = weight - update;
                        if (!isnan(new_weight) && !isinf(new_weight)) {
                            weights_fp16[idx] = __float2half(new_weight);
                        }
                    }
                }
            }
            __syncthreads();

            // Adam update for CA INTERACTION weights (FP16) - thread loop
            {
                half* weights_fp16 = ca_state->interaction_weights;
                float* gradients = ca_state->tape.grad_buffer + perception_params;
                float* m = training_mode->adam_m_interaction;
                float* v = training_mode->adam_v_interaction;

                for (int idx = tid; idx < interaction_params; idx += blockDim.x) {
                    float weight = __half2float(weights_fp16[idx]);
                    float g = gradients[idx];

                    if (!isnan(g) && !isinf(g)) {
                        float g_clipped = fmaxf(-gradient_clip_norm, fminf(gradient_clip_norm, g));
                        float m_new = adam_beta1 * m[idx] + (1.0f - adam_beta1) * g_clipped;
                        float v_new = adam_beta2 * v[idx] + (1.0f - adam_beta2) * g_clipped * g_clipped;
                        m[idx] = m_new;
                        v[idx] = v_new;

                        float m_hat = m_new / (1.0f - powf(adam_beta1, (float)(timestep + 1)));
                        float v_hat = v_new / (1.0f - powf(adam_beta2, (float)(timestep + 1)));
                        float update = lr * m_hat / (sqrtf(v_hat) + adam_epsilon);

                        float new_weight = weight - update;
                        if (!isnan(new_weight) && !isinf(new_weight)) {
                            weights_fp16[idx] = __float2half(new_weight);
                        }
                    }
                }
            }
            __syncthreads();

            // Adam update for CA VALUE weights (FP16) - thread loop
            {
                half* weights_fp16 = ca_state->value_weights;
                float* gradients = ca_state->tape.grad_buffer + perception_params + interaction_params;
                float* m = training_mode->adam_m_value;
                float* v = training_mode->adam_v_value;

                for (int idx = tid; idx < value_params; idx += blockDim.x) {
                    float weight = __half2float(weights_fp16[idx]);
                    float g = gradients[idx];

                    if (!isnan(g) && !isinf(g)) {
                        float g_clipped = fmaxf(-gradient_clip_norm, fminf(gradient_clip_norm, g));
                        float m_new = adam_beta1 * m[idx] + (1.0f - adam_beta1) * g_clipped;
                        float v_new = adam_beta2 * v[idx] + (1.0f - adam_beta2) * g_clipped * g_clipped;
                        m[idx] = m_new;
                        v[idx] = v_new;

                        float m_hat = m_new / (1.0f - powf(adam_beta1, (float)(timestep + 1)));
                        float v_hat = v_new / (1.0f - powf(adam_beta2, (float)(timestep + 1)));
                        float update = lr * m_hat / (sqrtf(v_hat) + adam_epsilon);

                        float new_weight = weight - update;
                        if (!isnan(new_weight) && !isinf(new_weight)) {
                            weights_fp16[idx] = __float2half(new_weight);
                        }
                    }
                }
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:adam_fp16_complete tid=%d\n", tid);

            // Adam update for pooling weights (FP32) - thread loop
            {
                float* weights = training_mode->classifier->pooling_weights;
                float* gradients = organism->pooling_weights_grad;
                float* m = organism->adam_m_pooling;
                float* v = organism->adam_v_pooling;
                int num_params = arch.channels;

                for (int idx = tid; idx < num_params; idx += blockDim.x) {
                    float g = gradients[idx];
                    if (!isnan(g) && !isinf(g)) {
                        float g_clipped = fmaxf(-gradient_clip_norm, fminf(gradient_clip_norm, g));
                        float m_new = adam_beta1 * m[idx] + (1.0f - adam_beta1) * g_clipped;
                        float v_new = adam_beta2 * v[idx] + (1.0f - adam_beta2) * g_clipped * g_clipped;
                        m[idx] = m_new;
                        v[idx] = v_new;

                        float m_hat = m_new / (1.0f - powf(adam_beta1, (float)(timestep + 1)));
                        float v_hat = v_new / (1.0f - powf(adam_beta2, (float)(timestep + 1)));
                        float update = lr * m_hat / (sqrtf(v_hat) + adam_epsilon);

                        float new_weight = weights[idx] - update;
                        if (!isnan(new_weight) && !isinf(new_weight)) {
                            weights[idx] = new_weight;
                        }
                    }
                }
            }
            __syncthreads();

            // Adam update for FC weights (FP32) - thread loop
            {
                int fc_weights_size = num_classes * behavioral_dim;
                float* weights = training_mode->classifier->fc_weights;
                float* gradients = organism->fc_weights_grad;
                float* m = organism->adam_m_fc_weights;
                float* v = organism->adam_v_fc_weights;

                for (int idx = tid; idx < fc_weights_size; idx += blockDim.x) {
                    float g = gradients[idx];
                    if (!isnan(g) && !isinf(g)) {
                        float g_clipped = fmaxf(-gradient_clip_norm, fminf(gradient_clip_norm, g));
                        float m_new = adam_beta1 * m[idx] + (1.0f - adam_beta1) * g_clipped;
                        float v_new = adam_beta2 * v[idx] + (1.0f - adam_beta2) * g_clipped * g_clipped;
                        m[idx] = m_new;
                        v[idx] = v_new;

                        float m_hat = m_new / (1.0f - powf(adam_beta1, (float)(timestep + 1)));
                        float v_hat = v_new / (1.0f - powf(adam_beta2, (float)(timestep + 1)));
                        float update = lr * m_hat / (sqrtf(v_hat) + adam_epsilon);

                        float new_weight = weights[idx] - update;
                        if (!isnan(new_weight) && !isinf(new_weight)) {
                            weights[idx] = new_weight;
                        }
                    }
                }
            }
            __syncthreads();

            // Adam update for FC bias (FP32) - thread loop
            {
                float* weights = training_mode->classifier->fc_bias;
                float* gradients = organism->fc_bias_grad;
                float* m = organism->adam_m_fc_bias;
                float* v = organism->adam_v_fc_bias;

                for (int idx = tid; idx < num_classes; idx += blockDim.x) {
                    float g = gradients[idx];
                    if (!isnan(g) && !isinf(g)) {
                        float g_clipped = fmaxf(-gradient_clip_norm, fminf(gradient_clip_norm, g));
                        float m_new = adam_beta1 * m[idx] + (1.0f - adam_beta1) * g_clipped;
                        float v_new = adam_beta2 * v[idx] + (1.0f - adam_beta2) * g_clipped * g_clipped;
                        m[idx] = m_new;
                        v[idx] = v_new;

                        float m_hat = m_new / (1.0f - powf(adam_beta1, (float)(timestep + 1)));
                        float v_hat = v_new / (1.0f - powf(adam_beta2, (float)(timestep + 1)));
                        float update = lr * m_hat / (sqrtf(v_hat) + adam_epsilon);

                        float new_weight = weights[idx] - update;
                        if (!isnan(new_weight) && !isinf(new_weight)) {
                            weights[idx] = new_weight;
                        }
                    }
                }
            }
            __syncthreads();

            // Increment timestep (thread 0 only)
            if (tid == 0) {
                training_mode->adam_timestep++;
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:adam_complete tid=%d\n", tid);

            // Extract head gradient magnitudes - thread loop
            float* gradient_magnitudes = organism->gradient_magnitudes_pool;
            {
                ADTape* tape = &ca_state->tape;
                for (int head = tid; head < arch.num_heads; head += blockDim.x) {
                    int offset = param_map->head_param_offsets[head];
                    int count = param_map->head_param_counts[head];
                    float sum = 0.0f;
                    for (int i = 0; i < count; i++) {
                        float g = tape->grad_buffer[offset + i];
                        sum += g * g;
                    }
                    gradient_magnitudes[head] = sqrtf(sum / (float)count);
                }
            }
            __syncthreads();

            // Compute gradient fitness - thread loop
            {
                for (int idx = tid; idx < POOL_CAPACITY_MAX; idx += blockDim.x) {
                    float grad_mag = gradient_magnitudes[idx % arch.num_heads];
                    float coherence = organism->coherence_history[(generation % 2) * POOL_CAPACITY_MAX + idx];
                    float fitness = organism->fitness_history[(generation % 2) * POOL_CAPACITY_MAX + idx];

                    float grad_fitness = training_mode->gradient_fitness_weight * grad_mag;
                    float coh_fitness = training_mode->coherence_fitness_weight * coherence;
                    organism->fitness_history[(generation % 2) * POOL_CAPACITY_MAX + idx] = fitness + grad_fitness + coh_fitness;
                }
            }
            __syncthreads();
            SLIME_DEBUG_PRINT("V:hybrid:gradient_fitness_complete tid=%d\n", tid);
        }  // end if (entry_idx == 0 && batch_images != nullptr)
    }  // end if (training_mode->use_gradients)
    SLIME_DEBUG_PRINT("V:hybrid:825 post_training entry_idx=%d\n", entry_idx);

    // ========== COMPONENT EVOLUTION (replaced CDP launch, thread loops) ==========
    // GLOBAL operation: only entry_idx==0 block processes, ALL threads participate
    if (entry_idx == 0) {
        SLIME_DEBUG_PRINT("V:hybrid:828 pre_component_evolution\n");
        float* component_workspace_genomes = organism->buffers->component_workspace_genomes_buffer;
        GPUElite* archive = organism->archive;
        int archive_size_val = organism->archive_size;

        float current_task_accuracy = organism->telemetry->task_performance.accuracy;
        float train_acc = organism->telemetry->task_performance.train_accuracy;
        float test_acc = organism->telemetry->task_performance.test_accuracy;

        // Phase 1: Per-entry independent work (genome reconstruction, metrics, hardware efficiency)
        // Each thread handles multiple entries via grid-stride loop
        for (int eid = tid; eid < pool->capacity; eid += blockDim.x) {
            if (!pool->alive_flags[eid]) continue;

            PoolEntry* ent = &pool->entries[eid];
            float* eid_primary_genome = &component_workspace_genomes[eid * 2 * GENOME_SIZE];
            float* eid_parent_temp = &component_workspace_genomes[eid * 2 * GENOME_SIZE + GENOME_SIZE];

            // Genome reconstruction
            reconstruct_genome_from_archive(
                ent->parent_hash,
                archive,
                archive_size_val,
                ent->delta_indices,
                ent->delta_values,
                ent->num_deltas,
                ent->max_deltas,
                eid_primary_genome,
                GENOME_SIZE,
                eid_parent_temp,
                organism->diresa_genome_weights
            );

            // Task accuracy and generalization gap
            if (!isnan(current_task_accuracy)) {
                ent->task_accuracy = current_task_accuracy;
            }
            ent->generalization_gap = fabsf(train_acc - test_acc);

            // History updates
            organism->fitness_history[(generation % 2) * POOL_CAPACITY_MAX + eid] = ent->task_accuracy;
            organism->coherence_history[(generation % 2) * POOL_CAPACITY_MAX + eid] = ent->coherence;

            // Hardware efficiency computation
            {
                float hardware_features_temp[HARDWARE_FEATURES_DIM];
                extract_hardware_features(organism->hardware_geom, hardware_features_temp);

                float eid_fitness = pool->fitness_values[eid];
                float eid_hunger = ent->hunger;

                float hw_efficiency_sum = safe_epsilon(1.0f);
                for (int i = 0; i < HARDWARE_FEATURES_DIM; i++) {
                    int hw_weight_slot = derive_param_slot(ent->genome_hash, "hw_efficiency_weight");
                    float hw_weight = genome_to_param(
                        eid_primary_genome,
                        ent->gradients,
                        hw_weight_slot,
                        eid_fitness,
                        eid_hunger,
                        organism->chemical_field->concentration[0],
                        organism->telemetry->genome_complexity.hash_entropy,
                        organism->telemetry->archive_topology.novelty_gradient,
                        organism->telemetry->diresa_evolution.behavioral_drift_rate,
                        organism->telemetry->task_performance.accuracy,
                        FITNESS_EFFICIENCY_EXPONENT_MIN, FITNESS_EFFICIENCY_EXPONENT_MAX
                    );
                    hw_efficiency_sum += hw_weight * hardware_features_temp[i];
                }
                ent->hardware_efficiency = hw_efficiency_sum;
            }
        }
        __syncthreads();

        // Phase 2: Block reduction infrastructure (accumulates population-wide metrics)
        // Reduce in batches across all pool entries
        {
            float local_acc = 0.0f, local_gap = 0.0f, local_hw = 0.0f, local_fit = 0.0f;
            for (int eid = tid; eid < pool->capacity; eid += blockDim.x) {
                if (pool->alive_flags[eid]) {
                    local_acc += pool->entries[eid].task_accuracy;
                    local_gap += pool->entries[eid].generalization_gap;
                    local_hw += pool->entries[eid].hardware_efficiency;
                    local_fit += pool->fitness_values[eid];
                }
            }
            // Store thread-local sums for reduction
            sdata[tid] = local_acc;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                __syncthreads();
            }
            float total_acc = sdata[0];

            sdata[tid] = local_gap;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                __syncthreads();
            }
            float total_gap = sdata[0];

            sdata[tid] = local_hw;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                __syncthreads();
            }
            float total_hw = sdata[0];

            sdata[tid] = local_fit;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                __syncthreads();
            }
            float total_fit = sdata[0];

            // Store population-wide aggregates (infrastructure for downstream wiring)
            if (tid == 0) {
                organism->telemetry->population_metrics.total_accuracy = total_acc;
                organism->telemetry->population_metrics.total_generalization_gap = total_gap;
                organism->telemetry->population_metrics.total_hardware_efficiency = total_hw;
                organism->telemetry->population_metrics.total_fitness = total_fit;
            }
        }
        __syncthreads();

        // Phase 3: Baldwin Effect - reinforce genomes showing learning improvement
        if (generation > 0) {
            for (int eid = tid; eid < pool->capacity; eid += blockDim.x) {
                if (!pool->alive_flags[eid]) continue;

                float prev_task_accuracy = organism->fitness_history[((generation - 1) % 2) * POOL_CAPACITY_MAX + eid];
                float learning_success = current_task_accuracy - prev_task_accuracy;

                if (is_meaningful(learning_success, 1.0f)) {
                    PoolEntry* ent = &pool->entries[eid];
                    float baldwin_sensitivity = ent->baldwin_sensitivity;
                    float scale = learning_success * baldwin_sensitivity;
                    float* grads = ent->gradients;
                    float* eid_primary_genome = &component_workspace_genomes[eid * 2 * GENOME_SIZE];

                    // Each thread updates its own entries' genomes (element-wise)
                    for (int g = 0; g < GENOME_SIZE; g++) {
                        float val = grads[g] + scale * eid_primary_genome[g];
                        grads[g] = fmaxf(GENOME_VALUE_MIN, fminf(GENOME_VALUE_MAX, val));
                    }
                }
            }
        }
        __syncthreads();
        SLIME_DEBUG_PRINT("V:hybrid:component_evolution_complete tid=%d\n", tid);

        // ========== NEURAL CA UPDATE (replaced CDP launch, warp-level WMMA) ==========
        // Only when NOT using gradients (non-training path)
        if (!training_mode->use_gradients) {
            SLIME_DEBUG_PRINT("V:hybrid:852 pre_neural_ca\n");
            float* nca_workspace_genomes = organism->buffers->organism_workspace_genomes;
            int warp_id = tid / WARP_SIZE;
            int lane_id = tid % WARP_SIZE;
            int num_warps = blockDim.x / WARP_SIZE;

            // Process each alive entry sequentially (warp cooperates on WMMA for each)
            for (int eid = 0; eid < pool->capacity; eid++) {
                if (!pool->alive_flags[eid]) continue;

                PoolEntry* ent = &pool->entries[eid];
                MultiHeadCAState* ent_ca_state = ent->ca_state;
                ArchitectureParams ent_arch = get_arch_from_pool(pool, eid);

                int grid_size = ent->grid_size;
                int num_cells = grid_size * grid_size;
                int channels = ent_arch.channels;
                int num_heads = ent_arch.num_heads;
                int head_dim = ent_arch.head_dim;

                if (channels <= 0 || num_heads <= 0 || head_dim <= 0) continue;

                // Convert CA concentration to FP16 workspace (thread loop)
                {
                    int total = num_cells * channels;
                    for (int idx = tid; idx < total; idx += blockDim.x) {
                        ent_ca_state->fp16_workspace[idx] = __float2half(ent_ca_state->ca_concentration[idx]);
                    }
                }
                __syncthreads();

                half* fp16_workspace = ent_ca_state->fp16_workspace;
                float* fp32_workspace = ent_ca_state->fp32_workspace;
                half* perception_weights = ent_ca_state->perception_weights;
                half* interaction_weights = ent_ca_state->interaction_weights;
                half* value_weights = ent_ca_state->value_weights;
                float* ca_output_fp32 = ent_ca_state->ca_output;

                // Multi-head CA: Perception → ReLU → Interaction → GELU → Value
                for (int head = 0; head < num_heads; head++) {
                    int weight_offset = head * channels * head_dim;
                    int head_size = num_cells * head_dim;

                    // Perception GEMM: [num_cells × channels] @ [channels × head_dim] → [num_cells × head_dim]
                    // Warp-level WMMA tile iteration
                    {
                        int tiles_M = (num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                        int tiles_N = (head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                        int total_tiles = tiles_M * tiles_N;

                        for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                            int tile_m = tile_idx / tiles_N;
                            int tile_n = tile_idx % tiles_N;
                            int tile_row = tile_m * WMMA_TILE_DIM;
                            int tile_col = tile_n * WMMA_TILE_DIM;

                            if (tile_row < num_cells && tile_col < head_dim) {
                                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> a_frag;
                                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                                nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                                for (int k = 0; k < channels; k += WMMA_TILE_DIM) {
                                    if (k + WMMA_TILE_DIM <= channels) {
                                        nvcuda::wmma::load_matrix_sync(a_frag, fp16_workspace + tile_row * channels + k, channels);
                                        nvcuda::wmma::load_matrix_sync(b_frag, perception_weights + weight_offset + k * head_dim + tile_col, head_dim);
                                        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                                    }
                                }
                                int output_offset = head * num_cells * head_dim;
                                nvcuda::wmma::store_matrix_sync(fp32_workspace + output_offset + tile_row * head_dim + tile_col, c_frag, head_dim, nvcuda::wmma::mem_row_major);
                            }
                        }
                    }
                    __syncthreads();

                    // ReLU activation (thread loop)
                    {
                        float* head_data = fp32_workspace + head * num_cells * head_dim;
                        for (int idx = tid; idx < head_size; idx += blockDim.x) {
                            head_data[idx] = fmaxf(0.0f, head_data[idx]);
                        }
                    }
                    __syncthreads();

                    // Convert perception output to FP16 for interaction (thread loop)
                    half* interaction_input = fp16_workspace + num_cells * channels;
                    {
                        float* head_data = fp32_workspace + head * num_cells * head_dim;
                        for (int idx = tid; idx < head_size; idx += blockDim.x) {
                            interaction_input[idx] = __float2half(head_data[idx]);
                        }
                    }
                    __syncthreads();

                    // Interaction GEMM: [num_cells × head_dim] @ [head_dim × head_dim] → [num_cells × head_dim]
                    float* interaction_output = fp32_workspace + num_heads * num_cells * head_dim;
                    {
                        int tiles_M = (num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                        int tiles_N = (head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                        int total_tiles = tiles_M * tiles_N;

                        for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                            int tile_m = tile_idx / tiles_N;
                            int tile_n = tile_idx % tiles_N;
                            int tile_row = tile_m * WMMA_TILE_DIM;
                            int tile_col = tile_n * WMMA_TILE_DIM;

                            if (tile_row < num_cells && tile_col < head_dim) {
                                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> a_frag;
                                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> b_frag;
                                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                                nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                                for (int k = 0; k < head_dim; k += WMMA_TILE_DIM) {
                                    if (k + WMMA_TILE_DIM <= head_dim) {
                                        nvcuda::wmma::load_matrix_sync(a_frag, interaction_input + tile_row * head_dim + k, head_dim);
                                        nvcuda::wmma::load_matrix_sync(b_frag, interaction_weights + head * head_dim * head_dim + k * head_dim + tile_col, head_dim);
                                        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                                    }
                                }
                                nvcuda::wmma::store_matrix_sync(interaction_output + tile_row * head_dim + tile_col, c_frag, head_dim, nvcuda::wmma::mem_row_major);
                            }
                        }
                    }
                    __syncthreads();

                    // GELU activation (thread loop)
                    {
                        for (int idx = tid; idx < head_size; idx += blockDim.x) {
                            float x = interaction_output[idx];
                            interaction_output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845f * (x + 0.044715f * x * x * x)));
                        }
                    }
                    __syncthreads();

                    // Convert interaction output to FP16 for value projection (thread loop)
                    half* value_input = interaction_input;
                    {
                        for (int idx = tid; idx < head_size; idx += blockDim.x) {
                            value_input[idx] = __float2half(interaction_output[idx]);
                        }
                    }
                    __syncthreads();

                    // Value projection GEMM: [num_cells × head_dim] @ [head_dim × channels] → [num_cells × channels]
                    float* head_output = ca_output_fp32 + head * num_cells * channels;
                    {
                        int tiles_M = (num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                        int tiles_N = (channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                        int total_tiles = tiles_M * tiles_N;

                        for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                            int tile_m = tile_idx / tiles_N;
                            int tile_n = tile_idx % tiles_N;
                            int tile_row = tile_m * WMMA_TILE_DIM;
                            int tile_col = tile_n * WMMA_TILE_DIM;

                            if (tile_row < num_cells && tile_col < channels) {
                                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> a_frag;
                                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> b_frag;
                                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                                nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                                for (int k = 0; k < head_dim; k += WMMA_TILE_DIM) {
                                    if (k + WMMA_TILE_DIM <= head_dim) {
                                        nvcuda::wmma::load_matrix_sync(a_frag, value_input + tile_row * head_dim + k, head_dim);
                                        nvcuda::wmma::load_matrix_sync(b_frag, value_weights + head * head_dim * channels + k * channels + tile_col, channels);
                                        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                                    }
                                }
                                nvcuda::wmma::store_matrix_sync(head_output + tile_row * channels + tile_col, c_frag, channels, nvcuda::wmma::mem_row_major);
                            }
                        }
                    }
                    __syncthreads();
                }  // end head loop

                // Reduce affinity across heads for each cell
                {
                    int total_elements = num_heads * head_dim;
                    for (int cell_idx = 0; cell_idx < num_cells; cell_idx++) {
                        float affinity = 0.0f;
                        for (int i = tid; i < total_elements; i += blockDim.x) {
                            int h = i / head_dim;
                            int d = i % head_dim;
                            int idx = h * num_cells * head_dim + cell_idx * head_dim + d;
                            affinity += fp32_workspace[idx];
                        }
                        // Warp reduction
                        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                            affinity += __shfl_down_sync(0xffffffff, affinity, offset);
                        }
                        if (lane_id == 0) {
                            atomicAdd(&ent_ca_state->affinity_reduced[cell_idx], affinity);
                        }
                    }
                }
                __syncthreads();

                // Compute flow field from affinity and concentration
                {
                    float* affinity = ent_ca_state->affinity_reduced;
                    float* concentration = ent_ca_state->ca_concentration;
                    float* flow_field = ent_ca_state->flow_field;

                    for (int cell_idx = tid; cell_idx < num_cells; cell_idx += blockDim.x) {
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
                            ent->flow_beta_A, ent->flow_n,
                            ent->flow_alpha_min, ent->flow_alpha_max, ent->flow_sharpness
                        );

                        flow_field[cell_idx * 2 + 0] = F.x;
                        flow_field[cell_idx * 2 + 1] = F.y;
                    }
                }
                __syncthreads();

                // Clear reintegration buffer
                {
                    int buffer_size = num_cells * channels;
                    float* reint_buf = ent_ca_state->reintegration_buffer;
                    for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                        reint_buf[idx] = 0.0f;
                    }
                }
                __syncthreads();

                // Redistribute mass via flow field transport
                {
                    float* flow_field = ent_ca_state->flow_field;
                    float* concentration = ent_ca_state->ca_concentration;
                    float* buffer = ent_ca_state->reintegration_buffer;
                    float dt = ent->flow_resource_dt;

                    for (int source_idx = 0; source_idx < num_cells; source_idx++) {
                        int source_x = source_idx % grid_size;
                        int source_y = source_idx / grid_size;
                        float Fx = flow_field[source_idx * 2 + 0];
                        float Fy = flow_field[source_idx * 2 + 1];

                        for (int c = tid; c < channels; c += blockDim.x) {
                            float source_mass = concentration[source_idx * channels + c];
                            FlowLeniaOps::bilinear_transport_forward(
                                source_mass, (float)source_x, (float)source_y,
                                Fx, Fy, dt, grid_size, buffer, c, channels
                            );
                        }
                    }
                }
                __syncthreads();

                // Copy reintegration buffer back to concentration
                {
                    int buffer_size = num_cells * channels;
                    float* conc = ent_ca_state->ca_concentration;
                    float* reint_buf = ent_ca_state->reintegration_buffer;
                    for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                        conc[idx] = reint_buf[idx];
                    }
                }
                __syncthreads();

                // Update global chemical field from CA concentration (channel 0)
                {
                    for (int cell_idx = tid; cell_idx < num_cells; cell_idx += blockDim.x) {
                        float val = ent_ca_state->ca_concentration[cell_idx * channels + 0];
                        if (isfinite(val)) {
                            atomicAdd(&organism->chemical_field->concentration[cell_idx], val);
                        }
                    }
                }
                __syncthreads();

            }  // end entry loop
            SLIME_DEBUG_PRINT("V:hybrid:neural_ca_complete tid=%d\n", tid);
        }  // end if (!use_gradients)
    }  // end if (entry_idx == 0)

    // PER-ENTRY operations - each block handles its own entry in parallel (thread loops)
    SLIME_DEBUG_PRINT("V:hybrid:873 pre_per_entry entry_idx=%d\n", entry_idx);
    if (training_mode->use_gradients) {
        SLIME_DEBUG_PRINT("V:hybrid:875 per_entry_start entry_idx=%d\n", entry_idx);

        // update_field_from_ca - thread loop (replaced CDP launch)
        {
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int total_cells = grid_size * grid_size;
            float* ca_concentration = entry->ca_state->ca_concentration;
            float* chemical_concentration = organism->chemical_field->concentration;

            for (int cell_idx = tid; cell_idx < total_cells; cell_idx += blockDim.x) {
                // CA concentration layout: [grid² × channels], read channel 0 for chemical field
                float val = ca_concentration[cell_idx * channels + 0];
                if (isfinite(val)) {
                    atomicAdd(&chemical_concentration[cell_idx], val);
                }
            }
        }
        __syncthreads();

        // convert_weights_to_fp32 - thread loop (replaced CDP launch)
        // Convert perception weights only: [num_heads × channels × head_dim]
        {
            int weight_count = arch.num_heads * arch.channels * arch.head_dim;
            half* weights_fp16 = ca_state->perception_weights;
            float* weights_fp32 = ca_state->fp32_workspace;

            for (int idx = tid; idx < weight_count; idx += blockDim.x) {
                weights_fp32[idx] = __half2float(weights_fp16[idx]);
            }
        }
        __syncthreads();

        // diresa_encode - thread 0 only (device function, not parallelizable)
        if (tid == 0) {
            float* temp_latent = primary_parent_temp;
            diresa_encode(primary_genome, temp_latent, &organism->diresa_genome_weights[0]);
        }
        __syncthreads();

        // compute_effective_rank_from_latent - thread loop with block reduction (replaced CDP launch)
        {
            float* latent_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];

            // Compute mean via block reduction
            float local_sum = 0.0f;
            for (int i = tid; i < GENOME_LATENT_DIM_MAX; i += blockDim.x) {
                local_sum += latent_genome[i];
            }
            sdata[tid] = local_sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            float mean = sdata[0] / (float)GENOME_LATENT_DIM_MAX;
            __syncthreads();

            // Compute variance via block reduction
            float local_var = 0.0f;
            for (int i = tid; i < GENOME_LATENT_DIM_MAX; i += blockDim.x) {
                float diff = latent_genome[i] - mean;
                local_var += diff * diff;
            }
            sdata[tid] = local_var;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            float variance = sdata[0] / (float)GENOME_LATENT_DIM_MAX;

            if (tid == 0 && variance >= 0.0f) {
                organism->effective_rank_history[entry_idx] = sqrtf(variance) * (float)GENOME_LATENT_DIM_MAX;
            }
        }
        __syncthreads();
        SLIME_DEBUG_PRINT("V:hybrid:per_entry_complete entry_idx=%d\n", entry_idx);
    }

    float* behavioral_workspace_genomes = organism->buffers->behavioral_workspace_genomes_buffer;

    // behavioral_update_kernel removed - already called in Phase 3 of persistent_evolution_kernel

    // Embedding update only needs to run once, not per-entry (thread loops, replaced CDP)
    if (entry_idx == 0 && generation % EMBEDDING_UPDATE_FREQ == 0) {
        // zero_scalar - thread 0 assignment (replaced CDP launch)
        if (tid == 0) {
            *organism->buffers->behavioral_reconstruction_error = 0.0f;
        }
        __syncthreads();

        int hw_dim = BEHAVIORAL_DIM_HW_MAX;
        int task_dim = BEHAVIORAL_DIM_TASK_MAX;
        int gen_dim = BEHAVIORAL_DIM_GEN_MAX;
        int embed_behavioral_dim = hw_dim + task_dim + gen_dim;

        // update_behavioral_embedding - thread loop (replaced CDP launch)
        {
            BehavioralState* agents = organism->behavioral_agents;
            float* embedding_weights = organism->buffers->behavioral_embedding_weights;
            float* reconstruction_error = organism->buffers->behavioral_reconstruction_error;
            int num_agents = POOL_CAPACITY_MAX;
            float learning_rate = 0.01f;
            float* features_buffer = organism->buffers->behavioral_features_buffer;
            float* chemical_concentration = organism->chemical_field->concentration;
            int grid_size = arch.grid_size;

            float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
            float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
            float ctx_learning = organism->telemetry->diresa_evolution.behavioral_drift_rate;
            float ctx_performance = organism->telemetry->task_performance.accuracy;

            for (int agent_id = tid; agent_id < num_agents; agent_id += blockDim.x) {
                BehavioralState* agent = &agents[agent_id];
                float* features = &features_buffer[agent_id * embed_behavioral_dim];

                float context_metabolic = agent->sensitivity;
                float context_stress = sqrtf(agent->velocity[0] * agent->velocity[0] +
                                             agent->velocity[1] * agent->velocity[1]);
                // Sample morphogen from chemical field at agent position
                int cx = (int)(agent->position[0] * grid_size) % grid_size;
                int cy = (int)(agent->position[1] * grid_size) % grid_size;
                float context_morphogen = chemical_concentration[cy * grid_size + cx];

                int base_freq_slot = derive_param_slot(agent->genome_hash, "fourier_base_freq");
                float fourier_base_freq = genome_to_param(primary_genome, entry->gradients, base_freq_slot,
                    context_metabolic, context_stress, context_morphogen,
                    ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
                    FOURIER_BASE_FREQ_MIN, FOURIER_BASE_FREQ_MAX);

                int num_octaves_slot = derive_param_slot(agent->genome_hash, "fourier_num_octaves");
                int fourier_num_octaves_raw = (int)genome_to_param(primary_genome, entry->gradients, num_octaves_slot,
                    context_metabolic, context_stress, context_morphogen,
                    ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
                    (float)FOURIER_NUM_OCTAVES_MIN, (float)FOURIER_NUM_OCTAVES_MAX);
                int fourier_num_octaves = min(fourier_num_octaves_raw, embed_behavioral_dim - 4);

                int spectrum_exp_slot = derive_param_slot(agent->genome_hash, "fourier_spectrum_exponent");
                float fourier_spectrum_exponent = genome_to_param(primary_genome, entry->gradients, spectrum_exp_slot,
                    context_metabolic, context_stress, context_morphogen,
                    ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
                    FOURIER_SPECTRUM_EXPONENT_MIN, FOURIER_SPECTRUM_EXPONENT_MAX);

                // Extract features
                features[0] = sqrtf(agent->velocity[0] * agent->velocity[0] +
                                   agent->velocity[1] * agent->velocity[1]);

                float turn_rate = 0.0f;
                for (int i = 1; i < GRADIENT_HISTORY; i++) {
                    float dx = agent->gradient_memory[i][0] - agent->gradient_memory[i-1][0];
                    float dy = agent->gradient_memory[i][1] - agent->gradient_memory[i-1][1];
                    turn_rate += sqrtf(dx * dx + dy * dy);
                }
                features[1] = turn_rate / GRADIENT_HISTORY;
                features[2] = agent->exploration_noise;
                features[3] = agent->sensitivity;

                // Fourier features
                for (int k = 0; k < fourier_num_octaves; k++) {
                    float freq = fourier_base_freq * powf(OCTAVE_MULTIPLIER, (float)k);
                    float cos_sum = 0.0f;
                    float sin_sum = 0.0f;

                    for (int i = 0; i < GRADIENT_HISTORY; i++) {
                        cos_sum += agent->gradient_memory[i][0] * cosf(freq * i);
                        sin_sum += agent->gradient_memory[i][1] * sinf(freq * i);
                    }

                    float magnitude = sqrtf(cos_sum * cos_sum + sin_sum * sin_sum) / GRADIENT_HISTORY;
                    if (freq > 0.0f) {
                        float amplitude_weight = powf(freq, -fourier_spectrum_exponent);
                        features[BASE_FEATURES_COUNT + k] = magnitude * amplitude_weight;
                    }
                }

                // Project features and update coords
                for (int d = 0; d < embed_behavioral_dim; d++) {
                    float reconstruction = 0.0f;
                    for (int f = 0; f < embed_behavioral_dim; f++) {
                        reconstruction += features[f] * embedding_weights[f * embed_behavioral_dim + d];
                    }

                    float* target_coord;
                    int local_idx;
                    if (d < hw_dim) {
                        target_coord = agent->hw_coords;
                        local_idx = d;
                    } else if (d < hw_dim + task_dim) {
                        target_coord = agent->task_coords;
                        local_idx = d - hw_dim;
                    } else {
                        target_coord = agent->gen_coords;
                        local_idx = d - hw_dim - task_dim;
                    }

                    float error = reconstruction - target_coord[local_idx];
                    target_coord[local_idx] += learning_rate * error;
                    atomicAdd(reconstruction_error, error * error);
                }
            }
        }
        __syncthreads();
        SLIME_DEBUG_PRINT("V:hybrid:embedding_update_complete tid=%d\n", tid);
    }

    // Memory update - thread loops (replaced CDP launch)
    if (entry_idx == 0) {
        TemporalTube* tubes = organism->chemical_field->history;

        // Skip if no tubes or empty
        if (tubes != nullptr && tubes->count > 0) {
            uint64_t mem_genome_hash = entry->genome_hash;
            float ctx_metabolic = entry->fitness;
            float ctx_stress = entry->hunger;
            float ctx_morphogen = local_ca_mean;
            float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
            float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
            float ctx_learning = organism->telemetry->diresa_evolution.behavioral_drift_rate;
            float ctx_performance = organism->telemetry->task_performance.accuracy;

            // Derive genome parameters
            int decay_threshold_slot = derive_param_slot(mem_genome_hash, "memory_decay_threshold");
            int flow_dt_slot = derive_param_slot(mem_genome_hash, "memory_flow_lenia_dt");

            float decay_threshold = genome_to_param(
                primary_genome, entry->gradients, decay_threshold_slot,
                ctx_metabolic, ctx_stress, ctx_morphogen,
                ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
                DECAY_THRESHOLD_MIN, DECAY_THRESHOLD_MAX
            );

            float flow_lenia_dt = genome_to_param(
                primary_genome, entry->gradients, flow_dt_slot,
                ctx_metabolic, ctx_stress, ctx_morphogen,
                ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
                FLOW_LENIA_DT_MIN, FLOW_LENIA_DT_MAX
            );

            int tube_count = tubes->count;

            // apply_decay - thread loop
            {
                for (int idx = tid; idx < tube_count; idx += blockDim.x) {
                    int entry_idx_tube = (tubes->head - tube_count + idx + tubes->capacity) % tubes->capacity;
                    MemoryEntry* mem_entry = &tubes->entries[entry_idx_tube];

                    float age = tubes->global_time - mem_entry->timestamp;
                    mem_entry->decay_factor = expf(-age * tubes->decay_rate);
                    mem_entry->decay_factor *= (1.0f + mem_entry->importance * 0.5f);
                }
                if (tid == 0) {
                    tubes->global_time += flow_lenia_dt;
                }
            }
            __syncthreads();

            // prune_memories - thread loop for marking, thread 0 for compaction
            {
                // Mark pruned entries
                for (int idx = tid; idx < tube_count; idx += blockDim.x) {
                    int entry_idx_tube = (tubes->head - tube_count + idx + tubes->capacity) % tubes->capacity;
                    MemoryEntry* mem_entry = &tubes->entries[entry_idx_tube];

                    if (mem_entry->decay_factor < decay_threshold) {
                        mem_entry->size = 0;
                    }
                }
            }
            __syncthreads();

            // Sequential compaction by thread 0 (simple fallback)
            if (tid == 0) {
                int write_idx = 0;
                for (int i = 0; i < tube_count; i++) {
                    int read_idx = (tubes->head - tube_count + i + tubes->capacity) % tubes->capacity;
                    if (tubes->entries[read_idx].size > 0) {
                        if (read_idx != write_idx) {
                            tubes->entries[write_idx] = tubes->entries[read_idx];
                        }
                        write_idx = (write_idx + 1) % tubes->capacity;
                    }
                }
                tubes->count = write_idx;
            }
            __syncthreads();
        }
        SLIME_DEBUG_PRINT("V:hybrid:memory_update_complete tid=%d\n", tid);

        // === POPULATE AUDIT BUFFER ===
        // Signal host with comprehensive telemetry from all 47+ system components
        if (tid == 0 && audit != nullptr && training_mode->batch_images != nullptr) {
            float* logits = organism->gradient_logits_pool;
            int* labels = training_mode->batch_labels;
            float* batch_images = training_mode->batch_images;
            int batch_size = training_mode->batch_size;
            int num_classes = organism->current_dataset->descriptor->num_classes;
            float* ca_concentration = ca_state->ca_concentration;
            int grid_size = entry->grid_size;
            float train_acc = organism->telemetry->task_performance.train_accuracy;
            float test_acc = organism->telemetry->task_performance.test_accuracy;

            populate_audit_buffer(
                audit,
                generation,
                logits,
                labels,
                batch_images,
                batch_size,
                num_classes,
                ca_concentration,
                grid_size,
                train_acc,
                test_acc,
                organism->telemetry,
                organism->pool
            );
            SLIME_DEBUG_PRINT("V:hybrid:audit_populated gen=%d\n", generation);
        }
    }  // end if (entry_idx == 0)

    // NOTE: No __syncthreads here - it would deadlock since entry 0's tid=0
    // is blocked at cudaDeviceSynchronize while other threads wait at syncthreads
    SLIME_DEBUG_PRINT("V:hybrid:983 kernel_exit entry_idx=%d tid=%d\n", entry_idx, tid);
}


#endif

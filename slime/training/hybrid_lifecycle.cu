
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

// Loads batch_images, batch_labels, and injects into batch_ca_states_pool
// Runs once per generation; per-entry kernels consume this shared batch data
extern "C" __global__ void load_batch_kernel(
    Organism* organism,
    HybridTrainingMode* training_mode,
    int generation,
    int grid_size  // Architecture grid size for bilinear interpolation
) {
    int tid = threadIdx.x;

    DEVICE_FATAL_IF(organism == nullptr, "load_batch_kernel: organism is null");
    DEVICE_FATAL_IF(training_mode == nullptr, "load_batch_kernel: training_mode is null");
    DEVICE_FATAL_IF(training_mode->batch_images == nullptr, "load_batch_kernel: batch_images is null");
    DEVICE_FATAL_IF(training_mode->batch_labels == nullptr, "load_batch_kernel: batch_labels is null");

    Dataset* dataset = organism->current_dataset;
    DEVICE_FATAL_IF(dataset == nullptr, "load_batch_kernel: dataset is null");
    DEVICE_FATAL_IF(dataset->samples == nullptr, "load_batch_kernel: dataset samples is null");
    DEVICE_FATAL_IF(dataset->labels == nullptr, "load_batch_kernel: dataset labels is null");

    unsigned char* all_images = dataset->samples;
    unsigned char* all_labels = dataset->labels;
    float* batch_images_out = training_mode->batch_images;
    int* batch_labels_out = training_mode->batch_labels;
    int dataset_size = dataset->num_samples;
    int batch_size = training_mode->batch_size;
    int offset = generation * batch_size;
    int sample_rows = dataset->descriptor->sample_rows;
    int sample_cols = dataset->descriptor->sample_cols;
    int sample_channels = dataset->descriptor->channels;
    int sample_size = sample_rows * sample_cols * sample_channels;

    // Thread 0 writes all batch labels
    if (tid == 0) {
        for (int idx = 0; idx < batch_size; idx++) {
            int src_idx = (offset + idx) % dataset_size;
            batch_labels_out[idx] = all_labels[src_idx];
        }
    }
    __syncthreads();

    // Validate dimensions - these must be correct, no silent failures
    DEVICE_FATAL_IF(grid_size <= 0 || grid_size > 512, "load_batch_kernel: invalid grid_size");
    DEVICE_FATAL_IF(sample_rows <= 0, "load_batch_kernel: invalid sample_rows");
    DEVICE_FATAL_IF(sample_cols <= 0, "load_batch_kernel: invalid sample_cols");

    // All threads do bilinear interpolation via thread loop
    int total_pixels = batch_size * grid_size * grid_size;
    for (int work_idx = tid; work_idx < total_pixels; work_idx += blockDim.x) {
        int idx = work_idx / (grid_size * grid_size);
        int pixel_idx = work_idx % (grid_size * grid_size);

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
    __syncthreads();

    // INJECT_SAMPLE_TO_CA: inject batch into CA input buffer
    float* ca_out = organism->batch_ca_states_pool;
    int channels_out = CHANNELS_MAX;  // Use max for consistent buffer layout
    int image_channels = (int)dataset->descriptor->channels;

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

    int total_positions = batch_size * grid_size * grid_size;
    for (int work_idx = tid; work_idx < total_positions; work_idx += blockDim.x) {
        int batch_idx = work_idx / (grid_size * grid_size);
        int spatial_idx = work_idx % (grid_size * grid_size);

        int base_idx = batch_idx * grid_size * grid_size * channels_out + spatial_idx * channels_out;

        // Channel 0-5: ChemicalField
        ca_out[base_idx + 0] = chem_concentration[spatial_idx];
        ca_out[base_idx + 1] = chem_gradient_x[spatial_idx];
        ca_out[base_idx + 2] = chem_gradient_y[spatial_idx];
        ca_out[base_idx + 3] = chem_laplacian[spatial_idx];
        ca_out[base_idx + 4] = chem_sources[spatial_idx];
        ca_out[base_idx + 5] = chem_decay_factors[spatial_idx];

        // Channel 6-9: RD fields
        ca_out[base_idx + 6] = rd_resource_density[spatial_idx];
        ca_out[base_idx + 7] = rd_fitness_landscape[spatial_idx];
        ca_out[base_idx + 8] = rd_resource_gradient_x[spatial_idx];
        ca_out[base_idx + 9] = rd_resource_gradient_y[spatial_idx];

        // Channel 10: Behavioral field
        ca_out[base_idx + 10] = behavioral_field[spatial_idx];

        // Channel 11-13: Dataset sample
        int img_base = batch_idx * grid_size * grid_size * image_channels;
        ca_out[base_idx + 11] = batch_images[img_base + spatial_idx];
        ca_out[base_idx + 12] = batch_images[img_base + ((image_channels > 1) ? grid_size * grid_size : 0) + spatial_idx];
        ca_out[base_idx + 13] = batch_images[img_base + ((image_channels > 2) ? 2 * grid_size * grid_size : 0) + spatial_idx];

        // Channel 14: Previous concentration (recurrence)
        int prev_idx = batch_idx * grid_size * grid_size * channels_out + spatial_idx * channels_out;
        ca_out[base_idx + 14] = prev_concentration[prev_idx + 0];

        // Channel 15: Temporal/attractor retrieval
        ca_out[base_idx + 15] = attractor_field[spatial_idx];
    }
}

extern "C" __global__ void hybrid_organism_lifecycle_kernel(
    Organism* organism,
    HybridTrainingMode* training_mode,
    CAParameterMap* param_map,
    int generation,
    float* workspace_genomes,
    AuditBuffer* audit,
    int wave_start
) {
    extern __shared__ float sdata[];
    int entry_idx = wave_start + blockIdx.x;
    int wave_position = blockIdx.x;
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
    __shared__ int s_num_features;
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
    int num_features;
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
        float accuracy = entry->task_accuracy;
        arch.ca_gate_center = 2.0f - 1.5f * fminf(fmaxf(accuracy, 0.0f), 1.0f);

        num_features = arch.num_heads * arch.channels;

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
        s_num_features = num_features;
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
    num_features = s_num_features;
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

        // ========== RESET_TAPE ==========
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
        __syncthreads();

        // ========== MULTI_HEAD_CA ==========
        // Per-entry operation: ALL entries process with wave-indexed output buffers
        // Total work: batch_size × num_heads × grid_size² cells
        {
            // Record hardware execution trace for CA forward pass
            TraceBuffer* trace_buffer = &ca_state->trace;
            if (tid == 0 && trace_buffer->traces != nullptr) {
                int trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
                if (trace_idx < trace_buffer->capacity) {
                    record_warp_metrics(&trace_buffer->traces[trace_idx], blockIdx.x);
                }
            }

            float* ca_input = organism->batch_ca_states_pool;
            float* perception_saved = ca_state->perception_saved;
            float* interaction_saved = ca_state->interaction_saved;
            float* pre_gelu_saved = ca_state->pre_gelu_saved;

            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads = arch.num_heads;
            int head_dim = arch.head_dim;
            int cells_per_grid = grid_size * grid_size;

            int wave_ca_stride = batch_size * num_heads * cells_per_grid * channels;
            float* ca_output = organism->buffers->batched_ca_output + wave_position * wave_ca_stride;

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
        __syncthreads();

        // All threads check error flag and exit together if error detected
        __syncthreads();
        if (s_error_flag) return;

    // Flow Lenia: transport mass based on CA affinity gradients (per-entry with wave-indexed buffers)
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

            int wave_ca_stride = batch_size * num_heads * total_cells * channels;
            int wave_affinity_stride = batch_size * total_cells;
            int wave_flow_stride = batch_size * total_cells * 2;
            int wave_reint_stride = batch_size * total_cells * channels;

            float* wave_ca_output = organism->buffers->batched_ca_output + wave_position * wave_ca_stride;
            float* wave_affinity = organism->buffers->batch_affinity_reduced + wave_position * wave_affinity_stride;
            float* wave_flow = organism->buffers->batch_flow_field + wave_position * wave_flow_stride;
            float* wave_reint = organism->buffers->batch_reintegration_buffer + wave_position * wave_reint_stride;

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
                    affinity += wave_ca_output[idx];
                }
                wave_affinity[batch_idx * total_cells + cell_idx] = affinity;
            }
            __syncthreads();

            for (int work_idx = tid; work_idx < total_affinity_work; work_idx += blockDim.x) {
                int batch_idx = work_idx / total_cells;
                int cell_idx = work_idx % total_cells;
                int x = cell_idx % grid_size;
                int y = cell_idx / grid_size;

                int batch_offset = batch_idx * total_cells;
                float U_center = wave_affinity[batch_offset + cell_idx];
                int x_E = min(x + 1, grid_size - 1);
                int y_N = min(y + 1, grid_size - 1);
                float U_E = wave_affinity[batch_offset + y * grid_size + x_E];
                float U_N = wave_affinity[batch_offset + y_N * grid_size + x];

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
                wave_flow[flow_idx + 0] = F.x;
                wave_flow[flow_idx + 1] = F.y;
            }

            // Clear reintegration buffer
            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                wave_reint[idx] = 0.0f;
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
                float Fx = wave_flow[flow_idx + 0];
                float Fy = wave_flow[flow_idx + 1];

                int conc_batch_offset = batch_idx * total_cells * channels;
                float* batch_buffer = wave_reint + conc_batch_offset;
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
                organism->batch_ca_states_pool[idx] = wave_reint[idx];
            }
            __syncthreads();

            // Phase 6: Save concentration for next iteration's recurrence channel
            // batch_prev_concentration will be read by inject_sample_to_ca_kernel on next step
            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                organism->buffers->batch_prev_concentration[idx] = organism->batch_ca_states_pool[idx];
            }
        }  // end Flow Lenia
        __syncthreads();

        // ========== FORWARD PASS ==========
        // Per-entry operation: each entry processes with entry-indexed output buffers
        float* ca_output_grad = nullptr;

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int spatial_size = grid_size * grid_size;

            int wave_ca_stride = batch_size * num_heads_local * spatial_size * channels;
            float* ca_out = organism->buffers->batched_ca_output + wave_position * wave_ca_stride;
            float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
            float* pooling_weights = training_mode->classifier->pooling_weights;

            // ========== SPATIAL_POOLING ==========
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

            // ========== DIRESA TASK ENCODING ==========
            if (tid == 0) {
                BehavioralDimensions dims;
                dims.derive_from_genome(entry->genome_hash, primary_genome);
                int task_dim = dims.task_dim;

                DEVICE_FATAL_IF(organism->diresa_task_weights->input_dim != num_features,
                    "hybrid_lifecycle: diresa_task_weights->input_dim mismatch with num_features");
                for (int b = 0; b < batch_size; b++) {
                    diresa_encode(&features[b * num_features], &organism->task_coords_pool[b * task_dim], organism->diresa_task_weights);
                }
            }
            __syncthreads();

            // ========== CLASSIFICATION_HEAD ==========
            float* logits = organism->gradient_logits_pool + entry_idx * batch_size * num_classes;
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

                DEVICE_FATAL_IF(isnan(acc) || isinf(acc), "hybrid_lifecycle: logit accumulator is NaN/Inf");
                logits[batch_idx * num_classes + class_idx] = acc;
            }
            __syncthreads();

            // ========== ZERO_SCALAR ==========
            float* loss_out = &organism->gradient_loss_pool[entry_idx];
            if (tid == 0) {
                *loss_out = 0.0f;
            }
            __syncthreads();

            // ========== CROSS_ENTROPY_LOSS ==========
            // Process samples sequentially, each using warp 0 for reduction
            float* logit_grads = organism->gradient_logit_grads_pool + entry_idx * batch_size * num_classes;
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

            // ========== TASK_PERFORMANCE_PROBE ==========
            // Thread 0 computes accuracy metrics
            if (tid == 0) {
                int correct = 0;
                float avg_confidence = 0.0f;
                int local_per_class_correct[NUM_CLASSES_MAX];
                int local_per_class_total[NUM_CLASSES_MAX];
                for (int c = 0; c < NUM_CLASSES_MAX; c++) {
                    local_per_class_correct[c] = 0;
                    local_per_class_total[c] = 0;
                }

                int grid_size = entry->grid_size;

                // Level 1: Partition field by entry_idx (spatial partitioning to avoid race conditions)
                int pool_capacity = organism->pool->capacity;
                int entries_per_side = (int)ceilf(sqrtf((float)pool_capacity));
                int entry_region_size = grid_size / entries_per_side;

                int entry_tile_y = entry_idx / entries_per_side;
                int entry_tile_x = entry_idx % entries_per_side;
                int entry_offset_y = entry_tile_y * entry_region_size;
                int entry_offset_x = entry_tile_x * entry_region_size;

                // Level 2: Within entry's exclusive region, tile by batch samples
                int tiles_per_side = (int)ceilf(sqrtf((float)batch_size));
                int tile_size = entry_region_size / tiles_per_side;

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

                    bool is_correct = (pred_class == true_label);
                    if (is_correct) correct++;
                    local_per_class_total[true_label]++;
                    if (is_correct) local_per_class_correct[true_label]++;

                    // Tile batch samples within entry's exclusive spatial region
                    int tile_y = b / tiles_per_side;
                    int tile_x = b % tiles_per_side;
                    float sample_accuracy = is_correct ? 1.0f : 0.0f;

                    for (int dy = 0; dy < tile_size; dy++) {
                        for (int dx = 0; dx < tile_size; dx++) {
                            int y = entry_offset_y + tile_y * tile_size + dy;
                            int x = entry_offset_x + tile_x * tile_size + dx;
                            if (y < grid_size && x < grid_size) {
                                int pos = y * grid_size + x;
                                organism->chemical_field->concentration[pos] = sample_accuracy;
                            }
                        }
                    }

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

                // Update per-entry metrics
                if (training_mode->is_train_batch) {
                    entry->train_accuracy = 0.9f * entry->train_accuracy + 0.1f * accuracy;
                } else {
                    entry->test_accuracy = accuracy;
                }
                entry->task_accuracy = accuracy;
                entry->avg_confidence = avg_confidence;
                for (int c = 0; c < NUM_CLASSES_MAX; c++) {
                    entry->per_class_correct[c] = local_per_class_correct[c];
                    entry->per_class_total[c] = local_per_class_total[c];
                }
            }
            __syncthreads();
        }
        __syncthreads();

        // ========== BACKWARD PASS ==========
        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {

            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int spatial_size = grid_size * grid_size;

            int wave_ca_stride = batch_size * num_heads_local * spatial_size * channels;
            float* ca_out = organism->buffers->batched_ca_output + wave_position * wave_ca_stride;
            ca_output_grad = organism->buffers->ca_output_grad_buffer + wave_position * wave_ca_stride;

            float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
            float* logit_grads = organism->gradient_logit_grads_pool + entry_idx * batch_size * num_classes;
            float* features_grad = organism->features_grad + entry_idx * batch_size * num_features;
            float* fc_weights = training_mode->classifier->fc_weights;
            float* fc_weights_grad = organism->fc_weights_grad;
            float* fc_bias_grad = organism->fc_bias_grad;
            float* pooling_weights = training_mode->classifier->pooling_weights;
            float* pooling_weights_grad = organism->pooling_weights_grad;

            // ========== ZERO GRADIENT BUFFERS ==========
            int fc_weights_size = num_classes * num_features;
            int features_grad_size = batch_size * num_features;
            int ca_grad_size = batch_size * num_heads_local * spatial_size * channels;

            // Zero all gradient buffers via thread loop
            for (int i = tid; i < fc_weights_size; i += blockDim.x) {
                fc_weights_grad[i] = 0.0f;
            }
            for (int i = tid; i < num_classes; i += blockDim.x) {
                fc_bias_grad[i] = 0.0f;
            }
            for (int i = tid; i < features_grad_size; i += blockDim.x) {
                features_grad[i] = 0.0f;
            }
            for (int i = tid; i < num_features; i += blockDim.x) {
                pooling_weights_grad[i] = 0.0f;
            }
            for (int i = tid; i < ca_grad_size; i += blockDim.x) {
                ca_output_grad[i] = 0.0f;
            }
            __syncthreads();

            // ========== CLASSIFICATION_HEAD_BACKWARD ==========
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

            // ========== SPATIAL_POOLING_BACKWARD ==========
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
        }

        // ========== CA BACKWARD PASS ==========
        // Per-entry operation with wave-indexed buffers
        {
            // Record hardware execution trace for CA backward pass
            TraceBuffer* trace_buffer = &ca_state->trace;
            if (tid == 0 && trace_buffer->traces != nullptr) {
                int trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
                if (trace_idx < trace_buffer->capacity) {
                    record_warp_metrics(&trace_buffer->traces[trace_idx], blockIdx.x);
                }
            }

            int num_cells = arch.grid_size * arch.grid_size;
            int total_samples = training_mode->batch_size * num_cells;
            int wave_dL_stride = training_mode->batch_size * arch.num_heads * num_cells * arch.head_dim;

            float* dL_dperception = organism->buffers->dL_dperception_buffer + wave_position * wave_dL_stride;
            float* dL_dinteraction = organism->buffers->dL_dinteraction_buffer + wave_position * wave_dL_stride;

            {

            // Use preallocated workspace buffers for GEMM-based backward pass

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
            int lane_id = tid % WARP_SIZE;
            int num_warps = blockDim.x / WARP_SIZE;

            // === VALUE BACKWARD ===
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

            // === INTERACTION BACKWARD ===
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

            // === PERCEPTION BACKWARD ===
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
                        // Store in ws_im2col [total_samples × channels]
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

            // Compute input gradients: d_pooled_input = sum over heads (d_prerelu @ W^T)
            // This backpropagates through perception layer to CA inputs (channels 0-5 are chemical field)
            // d_pooled_input: [total_samples × channels]
            float* d_pooled_input = ws_im2col;  // reuse buffer
            {
                int num_cells_local = arch.grid_size * arch.grid_size;
                int total_inputs = training_mode->batch_size * num_cells_local * arch.channels;
                for (int idx = tid; idx < total_inputs; idx += blockDim.x) {
                    d_pooled_input[idx] = 0.0f;
                }
            }
            __syncthreads();

            // Backprop through perception: d_pooled_input += d_prerelu @ W^T (sum over all heads)
            {
                int num_cells_local = arch.grid_size * arch.grid_size;
                int total_samples_local = training_mode->batch_size * num_cells_local;
                int total_work = total_samples_local * arch.channels;

                for (int work_idx = tid; work_idx < total_work; work_idx += blockDim.x) {
                    int sample_idx = work_idx / arch.channels;
                    int channel_idx = work_idx % arch.channels;

                    float d_input_accum = 0.0f;
                    // Sum over all heads
                    for (int h = 0; h < arch.num_heads; h++) {
                        // Sum over all head dimensions
                        for (int hd = 0; hd < arch.head_dim; hd++) {
                            int W_idx = param_map->perception_start[h] + channel_idx * arch.head_dim + hd;
                            float W_val = __half2float(ca_state->perception_weights[W_idx]);
                            int dprerelu_idx = h * ws_dprerelu_stride + sample_idx * arch.head_dim + hd;
                            d_input_accum += W_val * ws_dpregelu[dprerelu_idx];
                        }
                    }
                    d_pooled_input[work_idx] = d_input_accum;
                }
            }
            __syncthreads();

            // Spread pooled gradients back to 3x3 neighborhoods (inverse of pooling)
            // Input gradient buffer: organism->batch_ca_input_grads (needs to exist)
            float* d_ca_input = organism->batch_ca_input_grads;
            if (d_ca_input != nullptr) {
                int num_cells_local = arch.grid_size * arch.grid_size;
                int total_inputs = training_mode->batch_size * num_cells_local * arch.channels;
                for (int idx = tid; idx < total_inputs; idx += blockDim.x) {
                    int batch_id = idx / (num_cells_local * arch.channels);
                    int remainder = idx % (num_cells_local * arch.channels);
                    int cell_idx = remainder / arch.channels;
                    int channel_idx = remainder % arch.channels;
                    int cell_y = cell_idx / arch.grid_size;
                    int cell_x = cell_idx % arch.grid_size;

                    float grad_accum = 0.0f;
                    // This cell contributes to 9 pooled cells (3x3 window around it)
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int ny = cell_y + dy;
                            int nx = cell_x + dx;
                            if (ny >= 0 && ny < arch.grid_size && nx >= 0 && nx < arch.grid_size) {
                                int pooled_cell_idx = ny * arch.grid_size + nx;
                                int pooled_idx = batch_id * num_cells_local + pooled_cell_idx;
                                grad_accum += d_pooled_input[pooled_idx * arch.channels + channel_idx];
                            }
                        }
                    }
                    d_ca_input[idx] = grad_accum;
                }
            }
            __syncthreads();

            if (d_ca_input != nullptr) {
                int num_cells_local = arch.grid_size * arch.grid_size;
                float* grad_conc = organism->buffers->grad_concentration_buffer;

                for (int cell = tid; cell < num_cells_local; cell += blockDim.x) {
                    grad_conc[cell] = d_ca_input[cell * arch.channels];
                }
            }
            __syncthreads();

            if (d_ca_input != nullptr && tid == 0) {
                dim3 diff_grid((arch.grid_size + 15) / 16, (arch.grid_size + 15) / 16);
                dim3 diff_block(16, 16);

                float ctx_metabolic = entry->fitness;
                float ctx_stress = entry->hunger;
                float ctx_morphogen = organism->chemical_field->cached_mean;
                float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
                float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
                float ctx_learning = training_mode->learning_rate;
                float ctx_performance = entry->task_accuracy;

                diffusion_reaction_backward_kernel<<<diff_grid, diff_block>>>(
                    organism->buffers->grad_concentration_buffer,
                    organism->chemical_field->concentration,
                    organism->chemical_field->laplacian,
                    entry->gradients,
                    arch.grid_size,
                    CHEMICAL_DIFFUSION_DT_MAX,
                    primary_genome,
                    entry->gradients,
                    entry->genome_hash,
                    ctx_metabolic, ctx_stress, ctx_morphogen,
                    ctx_complexity, ctx_niche, ctx_learning, ctx_performance
                );
            }
            __syncthreads();

            }  // end CA backward scope

            // ========== FLOW-LENIA BACKWARD (gradients to flow parameters) ==========
            // Gradient flow: ca_output_grad → transport backward → flow backward → d_beta_A, d_n
            {
                int batch_size = training_mode->batch_size;
                int grid_size = arch.grid_size;
                int channels = arch.channels;
                int num_heads = arch.num_heads;
                int total_cells = grid_size * grid_size;

                float flow_beta_A = entry->flow_beta_A;
                float flow_n = entry->flow_n;
                float flow_alpha_min = entry->flow_alpha_min;
                float flow_alpha_max = entry->flow_alpha_max;
                float flow_sharpness = entry->flow_sharpness;
                float flow_dt = entry->flow_resource_dt;

                // Allocate gradient accumulators in shared memory
                __shared__ float s_d_beta_A;
                __shared__ float s_d_n;
                if (tid == 0) {
                    s_d_beta_A = 0.0f;
                    s_d_n = 0.0f;
                }
                __syncthreads();

                // Thread-local gradient accumulators
                float local_d_beta_A = 0.0f;
                float local_d_n = 0.0f;

                // Backward through bilinear transport
                int total_work = batch_size * total_cells;
                for (int work_idx = tid; work_idx < total_work; work_idx += blockDim.x) {
                    int batch_idx = work_idx / total_cells;
                    int cell_idx = work_idx % total_cells;
                    int source_x = cell_idx % grid_size;
                    int source_y = cell_idx / grid_size;

                    int batch_offset = batch_idx * total_cells;
                    int flow_idx = batch_offset * 2 + cell_idx * 2;
                    float Fx = organism->buffers->batch_flow_field[flow_idx + 0];
                    float Fy = organism->buffers->batch_flow_field[flow_idx + 1];

                    int conc_batch_offset = batch_idx * total_cells * channels;
                    const float* batch_conc = organism->batch_ca_states_pool + conc_batch_offset;

                    float d_flow_x_accum = 0.0f;
                    float d_flow_y_accum = 0.0f;

                    for (int c = 0; c < channels; c++) {
                        float source_mass = batch_conc[cell_idx * channels + c];
                        float d_source_mass, d_flow_x_local, d_flow_y_local;

                        // ca_output_grad has layout [batch × num_heads × grid² × channels]
                        // Reduce across heads to get per-channel gradient
                        float channel_grad = 0.0f;
                        for (int h = 0; h < num_heads; h++) {
                            int grad_idx = batch_idx * num_heads * total_cells * channels +
                                          h * total_cells * channels +
                                          cell_idx * channels + c;
                            channel_grad += ca_output_grad[grad_idx];
                        }

                        FlowLeniaOps::bilinear_transport_backward(
                            source_mass,
                            (float)source_x, (float)source_y,
                            Fx, Fy, flow_dt, grid_size,
                            &channel_grad,
                            &d_source_mass, &d_flow_x_local, &d_flow_y_local,
                            c, channels
                        );

                        d_flow_x_accum += d_flow_x_local;
                        d_flow_y_accum += d_flow_y_local;
                    }

                    // Backward through flow computation
                    int y = source_y;
                    int x = source_x;
                    int x_E = min(x + 1, grid_size - 1);
                    int y_N = min(y + 1, grid_size - 1);

                    float U_center = organism->buffers->batch_affinity_reduced[batch_offset + cell_idx];
                    float U_E = organism->buffers->batch_affinity_reduced[batch_offset + y * grid_size + x_E];
                    float U_N = organism->buffers->batch_affinity_reduced[batch_offset + y_N * grid_size + x];

                    float A_sum_center = 0.0f, A_sum_E = 0.0f, A_sum_N = 0.0f;
                    for (int c = 0; c < channels; c++) {
                        A_sum_center += batch_conc[cell_idx * channels + c];
                        A_sum_E += batch_conc[(y * grid_size + x_E) * channels + c];
                        A_sum_N += batch_conc[(y_N * grid_size + x) * channels + c];
                    }

                    float d_beta_A_local, d_n_local;
                    FlowLeniaOps::compute_flow_backward(
                        d_flow_x_accum, d_flow_y_accum,
                        U_center, U_E, U_N,
                        A_sum_center, A_sum_E, A_sum_N,
                        flow_beta_A, flow_n,
                        flow_alpha_min, flow_alpha_max, flow_sharpness,
                        &d_beta_A_local, &d_n_local
                    );

                    local_d_beta_A += d_beta_A_local;
                    local_d_n += d_n_local;
                }

                // Reduce thread-local gradients to shared
                atomicAdd(&s_d_beta_A, local_d_beta_A);
                atomicAdd(&s_d_n, local_d_n);
                __syncthreads();

                // Apply gradients to Flow-Lenia parameters (thread 0 only)
                if (tid == 0) {
                    float d_beta_A = s_d_beta_A / (batch_size * total_cells);
                    float d_n = s_d_n / (batch_size * total_cells);

                    // Derive Flow-Lenia learning rate from genome
                    float ctx_metabolic = entry->fitness;
                    float ctx_stress = entry->hunger;
                    float ctx_morphogen = local_ca_mean;
                    float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
                    float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
                    float ctx_learning = organism->telemetry->diresa_evolution.behavioral_drift_rate;
                    float ctx_performance = organism->telemetry->task_performance.accuracy;

                    TrainingParams train_params;
                    train_params.derive_from_genome_hash(entry->genome_hash);
                    float flow_lr = train_params.get_flow_lenia_lr(primary_genome, entry->gradients,
                        ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity,
                        ctx_niche, ctx_learning, ctx_performance);
                    float clip_norm = train_params.get_gradient_clip_norm(primary_genome, entry->gradients,
                        ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity,
                        ctx_niche, ctx_learning, ctx_performance);

                    // Clip gradients
                    d_beta_A = fmaxf(-clip_norm, fminf(clip_norm, d_beta_A));
                    d_n = fmaxf(-clip_norm, fminf(clip_norm, d_n));

                    // Gradient descent update
                    float new_beta_A = entry->flow_beta_A - flow_lr * d_beta_A;
                    float new_n = entry->flow_n - flow_lr * d_n;

                    // Clamp to valid ranges
                    entry->flow_beta_A = fmaxf(FLOW_LENIA_BETA_A_MIN, fminf(FLOW_LENIA_BETA_A_MAX, new_beta_A));
                    entry->flow_n = fmaxf(FLOW_LENIA_N_MIN, fminf(FLOW_LENIA_N_MAX, new_n));

                    // Store raw gradients for telemetry
                    organism->buffers->flow_beta_A_grad = d_beta_A;
                    organism->buffers->flow_n_grad = d_n;
                }
                __syncthreads();
            }
        }  // end CA backward pass

        // ========== COMPUTE EFFECTIVE_RANK FROM GRADIENTS ==========
        // Architecture line 688-689: gradient magnitudes → effective_rank
        // Compute per-head gradient RMSE, then Renyi entropy to measure learning diversity
        {
            int num_heads = arch.num_heads;
            int perception_per_head = arch.channels * arch.head_dim;
            int interaction_per_head = arch.head_dim * arch.head_dim;
            int value_per_head = arch.head_dim * arch.channels;
            int params_per_head = perception_per_head + interaction_per_head + value_per_head;

            float* grad_buf = ca_state->tape.grad_buffer;

            // Compute per-head squared gradient sum using warp reduction
            __shared__ float head_grad_sq[NUM_HEADS_MAX];
            for (int h = 0; h < num_heads && h < NUM_HEADS_MAX; h++) {
                float local_sq = 0.0f;
                // Each head's gradients are spread across perception, interaction, value sections
                int p_start = h * perception_per_head;
                int i_start = num_heads * perception_per_head + h * interaction_per_head;
                int v_start = num_heads * perception_per_head + num_heads * interaction_per_head + h * value_per_head;

                for (int i = tid; i < perception_per_head; i += blockDim.x) {
                    float g = grad_buf[p_start + i];
                    local_sq += g * g;
                }
                for (int i = tid; i < interaction_per_head; i += blockDim.x) {
                    float g = grad_buf[i_start + i];
                    local_sq += g * g;
                }
                for (int i = tid; i < value_per_head; i += blockDim.x) {
                    float g = grad_buf[v_start + i];
                    local_sq += g * g;
                }

                // Block reduction
                unsigned mask = __activemask();
                for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                    local_sq += __shfl_down_sync(mask, local_sq, offset);
                }
                __shared__ float warp_sums_eff[32];
                int lane = tid % warpSize;
                int warp_id = tid / warpSize;
                if (lane == 0) warp_sums_eff[warp_id] = local_sq;
                __syncthreads();
                if (tid < blockDim.x / warpSize) {
                    local_sq = warp_sums_eff[tid];
                    unsigned active = __activemask();
                    for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
                        local_sq += __shfl_down_sync(active, local_sq, offset);
                    }
                }
                if (tid == 0) {
                    // RMSE for this head
                    head_grad_sq[h] = sqrtf(local_sq / (float)params_per_head);
                }
                __syncthreads();
            }

            // Compute effective rank from gradient magnitude distribution (Renyi entropy)
            if (tid == 0) {
                float total_sq = 0.0f;
                for (int h = 0; h < num_heads; h++) {
                    total_sq += head_grad_sq[h] * head_grad_sq[h];
                }

                DEVICE_FATAL_IF(total_sq < 1e-12f, "effective_rank: zero gradient magnitude after backward pass - training catastrophically broken");

                // Shannon entropy: H = -Σ p_i log(p_i) where p_i = g_i² / Σg_j²
                float entropy = 0.0f;
                for (int h = 0; h < num_heads; h++) {
                    float g = head_grad_sq[h];
                    float p = (g * g) / total_sq;
                    if (p > 1e-12f) {
                        entropy -= p * logf(p);
                    }
                }
                // Effective rank = exp(entropy), clamped to [1, num_heads]
                float eff_rank = expf(entropy);
                entry->effective_rank = fmaxf(1.0f, fminf((float)num_heads, eff_rank));
            }
            __syncthreads();
        }

        // ========== ADAM UPDATES ==========
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

            // CDP: Adam update for CA weights via canonical kernel
            if (tid == 0) {
                int ts = timestep + 1;

                // Perception weights
                int blocks_p = (perception_params + 255) / 256;
                adam_update_fp16_kernel<<<blocks_p, 256>>>(
                    ca_state->perception_weights,
                    ca_state->tape.grad_buffer,
                    training_mode->adam_m,
                    training_mode->adam_v,
                    perception_params,
                    lr, adam_beta1, adam_beta2, adam_epsilon, ts, gradient_clip_norm
                );

                // Interaction weights
                int blocks_i = (interaction_params + 255) / 256;
                adam_update_fp16_kernel<<<blocks_i, 256>>>(
                    ca_state->interaction_weights,
                    ca_state->tape.grad_buffer + perception_params,
                    training_mode->adam_m + training_mode->perception_size,
                    training_mode->adam_v + training_mode->perception_size,
                    interaction_params,
                    lr, adam_beta1, adam_beta2, adam_epsilon, ts, gradient_clip_norm
                );

                // Value weights
                int blocks_v = (value_params + 255) / 256;
                adam_update_fp16_kernel<<<blocks_v, 256>>>(
                    ca_state->value_weights,
                    ca_state->tape.grad_buffer + perception_params + interaction_params,
                    training_mode->adam_m + training_mode->perception_size + training_mode->interaction_size,
                    training_mode->adam_v + training_mode->perception_size + training_mode->interaction_size,
                    value_params,
                    lr, adam_beta1, adam_beta2, adam_epsilon, ts, gradient_clip_norm
                );

                cudaDeviceSynchronize();
            }
            __syncthreads();

            // CDP: Adam update for classifier weights (FP32) via canonical kernel
            if (tid == 0) {
                int ts = timestep + 1;
                int fc_weights_size = num_classes * num_features;

                // Pooling weights
                int blocks_pool = (arch.channels + 255) / 256;
                adam_update_kernel<<<blocks_pool, 256>>>(
                    training_mode->classifier->pooling_weights,
                    organism->pooling_weights_grad,
                    organism->adam_m_pooling,
                    organism->adam_v_pooling,
                    arch.channels,
                    lr, adam_beta1, adam_beta2, adam_epsilon, ts, gradient_clip_norm
                );

                // FC weights
                int blocks_fc = (fc_weights_size + 255) / 256;
                adam_update_kernel<<<blocks_fc, 256>>>(
                    training_mode->classifier->fc_weights,
                    organism->fc_weights_grad,
                    organism->adam_m_fc_weights,
                    organism->adam_v_fc_weights,
                    fc_weights_size,
                    lr, adam_beta1, adam_beta2, adam_epsilon, ts, gradient_clip_norm
                );

                // FC bias
                int blocks_bias = (num_classes + 255) / 256;
                adam_update_kernel<<<blocks_bias, 256>>>(
                    training_mode->classifier->fc_bias,
                    organism->fc_bias_grad,
                    organism->adam_m_fc_bias,
                    organism->adam_v_fc_bias,
                    num_classes,
                    lr, adam_beta1, adam_beta2, adam_epsilon, ts, gradient_clip_norm
                );

                cudaDeviceSynchronize();
            }
            __syncthreads();

            // Increment timestep (thread 0 only)
            if (tid == 0) {
                training_mode->adam_timestep++;
            }
            __syncthreads();

        }  // end if (entry_idx == 0 && batch_images != nullptr)
    }  // end if (training_mode->use_gradients)

    // ========== COMPONENT EVOLUTION ==========
    // GLOBAL operation: only entry_idx==0 block processes, ALL threads participate
    if (entry_idx == 0) {
        float* component_workspace_genomes = organism->buffers->component_workspace_genomes_buffer;
        GPUElite* archive = organism->archive;
        int archive_size_val = organism->archive_size;

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

            // History updates (task_accuracy computed per-entry during training, not copied from global telemetry)
            organism->fitness_history[(generation % 2) * POOL_CAPACITY_MAX + eid] = ent->task_accuracy;
            organism->coherence_history[(generation % 2) * POOL_CAPACITY_MAX + eid] = ent->coherence;
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
                float current_task_accuracy = pool->entries[eid].task_accuracy;
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
    }

    if (training_mode->use_gradients) {

        // update_field_from_ca - thread loop
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

        // convert_weights_to_fp32 - thread loop
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
    }

    float* behavioral_workspace_genomes = organism->buffers->behavioral_workspace_genomes_buffer;

    // behavioral_update_kernel removed - already called in Phase 3 of persistent_evolution_kernel

    // Embedding update only needs to run once, not per-entry
    if (entry_idx == 0 && generation % EMBEDDING_UPDATE_FREQ == 0) {
        // zero_scalar - thread 0 assignment
        if (tid == 0) {
            *organism->buffers->behavioral_reconstruction_error = 0.0f;
        }
        __syncthreads();

        int hw_dim = BEHAVIORAL_DIM_HW_MAX;
        int task_dim = BEHAVIORAL_DIM_TASK_MAX;
        int gen_dim = BEHAVIORAL_DIM_GEN_MAX;
        int embed_behavioral_dim = hw_dim + task_dim + gen_dim;

        // update_behavioral_embedding - thread loop
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
    }

    // === SYNC BATCH CA → ENTRY CA STATE (closes training→archive loop) ===
    // Training operates on batch buffers. Archive/fitness/telemetry read entry state.
    // Without this copy, the eight loops are severed - archive sees zeros, fitness meaningless.
    if (entry_idx == 0 && organism->batch_ca_states_pool) {
        int grid_size_sync = arch.grid_size;
        int channels_sync = arch.channels;
        int spatial = grid_size_sync * grid_size_sync;
        int copy_size = spatial * channels_sync;
        int pool_cap = organism->pool->capacity;
        int batch_sz = training_mode->batch_size;

        // Copy batch samples back to corresponding entry CA states
        // This connects: training CA output → entry state → archive insertion → fitness
        for (int e = 0; e < pool_cap && e < batch_sz; e++) {
            PoolEntry* ent = &organism->pool->entries[e];
            if (ent->alive && ent->ca_state && ent->ca_state->ca_concentration) {
                int src_offset = e * copy_size;
                for (int idx = tid; idx < copy_size; idx += blockDim.x) {
                    ent->ca_state->ca_concentration[idx] = organism->batch_ca_states_pool[src_offset + idx];
                }
            }
        }
    }
    __syncthreads();

    // === POPULATE AUDIT BUFFER ===
    // Signal host with comprehensive telemetry from all 47+ system components
    if (tid == 0 && audit != nullptr && training_mode->batch_images != nullptr) {
        run_telemetry_probes(organism, generation);
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
            organism->pool,
            organism->chemical_field,
            ca_state,
            organism->hardware_geom
        );
    }
}


#endif

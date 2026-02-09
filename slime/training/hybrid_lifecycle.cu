
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

__device__ int g_block_counter = 0;


struct WaveBufferOffsets {
    int ca_states_offset;       
    int ca_output_offset;       
    int activations_offset;     
    int affinity_offset;        
    int flow_offset;            
    size_t backward_ws_offset;  
};


struct BackwardWorkspaceLayout {
    size_t fp16_a_offset;       
    size_t fp16_b_offset;       
    size_t dW_offset;           
    size_t dI_offset;           
    size_t W_T_offset;          
    size_t im2col_offset;       
    size_t dpregelu_offset;     
    size_t total_bytes;         
};


__device__ __forceinline__ BackwardWorkspaceLayout compute_backward_ws_layout(PoolEntry* entry) {
    BackwardWorkspaceLayout layout;
    int cells = entry->grid_size * entry->grid_size;
    int num_heads = entry->num_heads;
    int hidden_dim = entry->hidden_dim;
    int channels = entry->channels;
    int head_dim = entry->head_dim;

    size_t fp16_a_elements = (size_t)num_heads * BACKWARD_CHUNK_SAMPLES * hidden_dim;
    size_t fp16_b_elements = (size_t)num_heads * BACKWARD_CHUNK_SAMPLES * hidden_dim;
    size_t dW_elements = (size_t)num_heads * hidden_dim * hidden_dim;
    size_t dI_elements = (size_t)num_heads * BACKWARD_CHUNK_SAMPLES * hidden_dim;
    size_t W_T_elements = (size_t)hidden_dim * hidden_dim;
    size_t im2col_elements = (size_t)channels * cells * 9;
    size_t dpregelu_elements = (size_t)BACKWARD_CHUNK_SAMPLES * num_heads * cells * head_dim;

    size_t offset = 0;
    layout.fp16_a_offset = offset;
    offset += fp16_a_elements * sizeof(half);

    layout.fp16_b_offset = offset;
    offset += fp16_b_elements * sizeof(half);

    layout.dW_offset = offset;
    offset += dW_elements * sizeof(float);

    layout.dI_offset = offset;
    offset += dI_elements * sizeof(float);

    layout.W_T_offset = offset;
    offset += W_T_elements * sizeof(half);

    layout.im2col_offset = offset;
    offset += im2col_elements * sizeof(float);

    layout.dpregelu_offset = offset;
    offset += dpregelu_elements * sizeof(float);

    layout.total_bytes = offset;
    return layout;
}


__device__ __forceinline__ int compute_entry_ca_states_size(PoolEntry* entry, int batch_size) {
    int cells = entry->grid_size * entry->grid_size;
    return batch_size * cells * entry->channels;
}

__device__ __forceinline__ int compute_entry_ca_output_size(PoolEntry* entry, int batch_size) {
    int cells = entry->grid_size * entry->grid_size;
    return batch_size * entry->num_heads * cells * entry->channels;
}

__device__ __forceinline__ int compute_entry_activations_size(PoolEntry* entry, int batch_size) {
    int cells = entry->grid_size * entry->grid_size;
    return batch_size * entry->num_heads * cells * entry->head_dim;
}

__device__ __forceinline__ int compute_entry_affinity_size(PoolEntry* entry, int batch_size) {
    int cells = entry->grid_size * entry->grid_size;
    return batch_size * cells;
}

__device__ __forceinline__ int compute_entry_flow_size(PoolEntry* entry, int batch_size) {
    int cells = entry->grid_size * entry->grid_size;
    return batch_size * cells * 2;
}


__device__ WaveBufferOffsets compute_wave_offsets(
    ComponentPool* pool,
    int wave_start,
    int wave_position,
    int batch_size
) {
    WaveBufferOffsets offsets;
    offsets.ca_states_offset = 0;
    offsets.ca_output_offset = 0;
    offsets.activations_offset = 0;
    offsets.affinity_offset = 0;
    offsets.flow_offset = 0;
    offsets.backward_ws_offset = 0;

    
    for (int i = 0; i < wave_position; i++) {
        int compact_idx = wave_start + i;
        if (compact_idx >= pool->alive_indices_count) break;
        int entry_idx = pool->alive_indices[compact_idx];
        PoolEntry* entry = &pool->entries[entry_idx];

        offsets.ca_states_offset += compute_entry_ca_states_size(entry, batch_size);
        offsets.ca_output_offset += compute_entry_ca_output_size(entry, batch_size);
        offsets.activations_offset += compute_entry_activations_size(entry, batch_size);
        offsets.affinity_offset += compute_entry_affinity_size(entry, batch_size);
        offsets.flow_offset += compute_entry_flow_size(entry, batch_size);

        BackwardWorkspaceLayout ws_layout = compute_backward_ws_layout(entry);
        offsets.backward_ws_offset += ws_layout.total_bytes;
    }

    return offsets;
}

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

extern "C" __global__ void load_batch_kernel(
    Organism* organism,
    HybridTrainingMode* training_mode,
    int generation,
    int grid_size,
    int wave_start
) {
    int tid = threadIdx.x;
    int wave_position = blockIdx.x;
    int compact_idx = wave_start + blockIdx.x;

    __syncthreads();

    DEVICE_FATAL_IF(organism == nullptr, "load_batch_kernel: organism is null");
    DEVICE_FATAL_IF(training_mode == nullptr, "load_batch_kernel: training_mode is null");
    DEVICE_FATAL_IF(training_mode->batch_images == nullptr, "load_batch_kernel: batch_images is null");
    DEVICE_FATAL_IF(training_mode->batch_labels == nullptr, "load_batch_kernel: batch_labels is null");

    ComponentPool* pool = organism->pool;
    if (compact_idx >= pool->alive_indices_count) return;
    int entry_idx = pool->alive_indices[compact_idx];
    PoolEntry* entry = &pool->entries[entry_idx];

    
    __shared__ WaveBufferOffsets s_wave_offsets;
    if (tid == 0) {
        s_wave_offsets = compute_wave_offsets(pool, wave_start, wave_position, training_mode->batch_size);
    }
    __syncthreads();

    Dataset* dataset = organism->current_dataset;
    DEVICE_FATAL_IF(dataset == nullptr, "load_batch_kernel: dataset is null");
    DEVICE_FATAL_IF(dataset->samples == nullptr, "load_batch_kernel: dataset samples is null");
    DEVICE_FATAL_IF(dataset->labels == nullptr, "load_batch_kernel: dataset labels is null");
    DEVICE_FATAL_IF(dataset->descriptor == nullptr, "load_batch_kernel: dataset descriptor is null");
    int dataset_size = dataset->num_samples;
    int batch_size = training_mode->batch_size;
    DEVICE_FATAL_IF(dataset_size <= 0 || dataset_size > DATASET_SIZE_MAX, "load_batch_kernel: dataset num_samples invalid");
    DEVICE_FATAL_IF(batch_size <= 0 || batch_size > BATCH_SIZE_MAX, "load_batch_kernel: batch_size invalid");
    DEVICE_FATAL_IF(grid_size <= 0 || grid_size > GRID_SIZE_MAX, "load_batch_kernel: grid_size invalid");
    DEVICE_FATAL_IF(generation < 0, "load_batch_kernel: generation negative");

    int sample_rows = dataset->descriptor->sample_rows;
    int sample_cols = dataset->descriptor->sample_cols;
    int sample_channels = dataset->descriptor->channels;
    DEVICE_FATAL_IF(sample_rows <= 0 || sample_rows > SAMPLE_DIM_MAX, "load_batch_kernel: sample_rows invalid");
    DEVICE_FATAL_IF(sample_cols <= 0 || sample_cols > SAMPLE_DIM_MAX, "load_batch_kernel: sample_cols invalid");
    DEVICE_FATAL_IF(sample_channels <= 0 || sample_channels > CHANNELS_MAX, "load_batch_kernel: sample_channels invalid");

    int sample_size = sample_rows * sample_cols * sample_channels;
    int batch_stride = grid_size * grid_size;
    int total_pixels = batch_size * batch_stride;
    int offset = (generation % ((dataset_size + batch_size - 1) / batch_size)) * batch_size;

    DEVICE_FATAL_IF(sample_size <= 0 || sample_size > SAMPLE_DIM_MAX * SAMPLE_DIM_MAX * CHANNELS_MAX, "load_batch_kernel: sample_size overflow");
    DEVICE_FATAL_IF(batch_stride <= 0 || batch_stride > CA_FIELD_SIZE, "load_batch_kernel: batch_stride overflow");
    DEVICE_FATAL_IF(total_pixels <= 0 || total_pixels > BATCH_SIZE_MAX * CA_FIELD_SIZE, "load_batch_kernel: total_pixels overflow");
    DEVICE_FATAL_IF(offset < 0, "load_batch_kernel: offset overflow");

    unsigned char* all_images = dataset->samples;
    unsigned char* all_labels = dataset->labels;
    float* batch_images_out = training_mode->batch_images;
    int* batch_labels_out = training_mode->batch_labels;

    if (tid == 0) printf("V:L1_pre\n");
    if (tid == 0) {
        for (int idx = 0; idx < batch_size; idx++) {
            int src_idx = (offset + idx) % dataset_size;
            DEVICE_FATAL_IF(src_idx < 0 || src_idx >= dataset_size, "load_batch: label src_idx OOB");
            DEVICE_FATAL_IF(idx < 0 || idx >= BATCH_SIZE_MAX, "load_batch: label idx OOB");
            batch_labels_out[idx] = all_labels[src_idx];
        }
    }
    if (tid == 0) printf("V:L1_post\n");
    __syncthreads();

    if (tid == 0) printf("V:L2_pre\n");
    for (int work_idx = tid; work_idx < total_pixels; work_idx += blockDim.x) {
        int idx = work_idx / batch_stride;
        int pixel_idx = work_idx % batch_stride;

        DEVICE_FATAL_IF(idx < 0 || idx >= batch_size, "load_batch: img idx OOB");
        DEVICE_FATAL_IF(pixel_idx < 0 || pixel_idx >= batch_stride, "load_batch: img pixel_idx OOB");

        int src_idx = (offset + idx) % dataset_size;
        DEVICE_FATAL_IF(src_idx < 0 || src_idx >= dataset_size, "load_batch: img src_idx OOB");

        int out_y = pixel_idx / grid_size;
        int out_x = pixel_idx % grid_size;

        float src_y = out_y * (float)sample_rows / grid_size;
        float src_x = out_x * (float)sample_cols / grid_size;

        int y0 = (int)src_y;
        int x0 = (int)src_x;
        int y1 = min(y0 + 1, sample_rows - 1);
        int x1 = min(x0 + 1, sample_cols - 1);

        DEVICE_FATAL_IF(y0 < 0 || y0 >= sample_rows, "load_batch: y0 OOB");
        DEVICE_FATAL_IF(x0 < 0 || x0 >= sample_cols, "load_batch: x0 OOB");
        DEVICE_FATAL_IF(y1 < 0 || y1 >= sample_rows, "load_batch: y1 OOB");
        DEVICE_FATAL_IF(x1 < 0 || x1 >= sample_cols, "load_batch: x1 OOB");

        float fy = src_y - y0;
        float fx = src_x - x0;

        int out_idx_base = idx * batch_stride * 3;
        DEVICE_FATAL_IF(out_idx_base < 0 || out_idx_base + 2 * batch_stride + pixel_idx >= BATCH_SIZE_MAX * CA_FIELD_SIZE * 3, "load_batch: batch_images_out OOB");

        if (sample_channels >= 3) {
            for (int c = 0; c < 3; c++) {
                int channel_offset = c * sample_rows * sample_cols;
                int img_idx = src_idx * sample_size + channel_offset;
                DEVICE_FATAL_IF(img_idx + y1 * sample_cols + x1 >= dataset_size * sample_size, "load_batch: all_images OOB");
                float tl = all_images[img_idx + y0 * sample_cols + x0] / 255.0f;
                float tr = all_images[img_idx + y0 * sample_cols + x1] / 255.0f;
                float bl = all_images[img_idx + y1 * sample_cols + x0] / 255.0f;
                float br = all_images[img_idx + y1 * sample_cols + x1] / 255.0f;
                batch_images_out[out_idx_base + c * batch_stride + pixel_idx] = Interpolation::bilinear(tl, tr, bl, br, fx, fy);
            }
        } else {
            int img_idx = src_idx * sample_size;
            DEVICE_FATAL_IF(img_idx + y1 * sample_cols + x1 >= dataset_size * sample_size, "load_batch: all_images gray OOB");
            float tl = all_images[img_idx + y0 * sample_cols + x0] / 255.0f;
            float tr = all_images[img_idx + y0 * sample_cols + x1] / 255.0f;
            float bl = all_images[img_idx + y1 * sample_cols + x0] / 255.0f;
            float br = all_images[img_idx + y1 * sample_cols + x1] / 255.0f;
            float3 vg = Interpolation::bilinear_with_grad(tl, tr, bl, br, fx, fy);
            batch_images_out[out_idx_base + 0 * batch_stride + pixel_idx] = vg.x;
            batch_images_out[out_idx_base + 1 * batch_stride + pixel_idx] = vg.y;
            batch_images_out[out_idx_base + 2 * batch_stride + pixel_idx] = vg.z;
        }
    }
    if (tid == 0) printf("V:L2_post\n");
    __syncthreads();

    DEVICE_FATAL_IF(organism->batch_ca_states_pool == nullptr, "load_batch: batch_ca_states_pool null");
    DEVICE_FATAL_IF(organism->buffers == nullptr, "load_batch: buffers null");
    DEVICE_FATAL_IF(organism->buffers->batch_prev_concentration == nullptr, "load_batch: batch_prev_concentration null");

    
    float* ca_out = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
    int channels_out = entry->channels;
    float* batch_images = training_mode->batch_images;
    float* prev_concentration = organism->buffers->batch_prev_concentration + s_wave_offsets.ca_states_offset;

    int total_positions = batch_size * batch_stride;
    DEVICE_FATAL_IF(total_positions <= 0 || total_positions > BATCH_SIZE_MAX * CA_FIELD_SIZE, "load_batch: total_positions OOB");
    if (tid == 0) printf("V:L3_loop_pre total=%d\n", total_positions);

    
    constexpr int IMG_CHANNEL_START = 11;
    constexpr int IMG_CHANNEL_COUNT = 3;

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int spatial_idx = tid; spatial_idx < batch_stride; spatial_idx += blockDim.x) {
            int base_idx = batch_idx * batch_stride * channels_out + spatial_idx * channels_out;
            int prev_idx = batch_idx * batch_stride * channels_out + spatial_idx * channels_out;

            
            for (int c = 0; c < channels_out; c++) {
                ca_out[base_idx + c] = prev_concentration[prev_idx + c];
            }

            
            if (channels_out > IMG_CHANNEL_START + IMG_CHANNEL_COUNT - 1) {
                int img_base = batch_idx * batch_stride * 3;
                ca_out[base_idx + 11] = batch_images[img_base + 0 * batch_stride + spatial_idx];
                ca_out[base_idx + 12] = batch_images[img_base + 1 * batch_stride + spatial_idx];
                ca_out[base_idx + 13] = batch_images[img_base + 2 * batch_stride + spatial_idx];
            }
        }
        __syncthreads();
    }
    if (tid == 0) printf("V:L3_loop_done\n");
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
    int compact_idx = wave_start + blockIdx.x;
    int wave_position = blockIdx.x;
    int tid = threadIdx.x;
    if (blockIdx.x == 0 && tid == 0) g_block_counter = 0;
    __threadfence();
    ComponentPool* pool = organism->pool;

    if (compact_idx >= pool->alive_indices_count) return;

    int entry_idx = pool->alive_indices[compact_idx];
    PoolEntry* entry = &pool->entries[entry_idx];

    
    __shared__ WaveBufferOffsets s_wave_offsets;
    if (tid == 0) {
        s_wave_offsets = compute_wave_offsets(pool, wave_start, wave_position, training_mode->batch_size);
    }
    __syncthreads();

    __shared__ bool s_entry_alive;
    if (tid == 0) s_entry_alive = pool->alive_flags[entry_idx];
    __syncthreads();
    DEVICE_FATAL_IF(!s_entry_alive, "hybrid_organism_lifecycle_kernel: dead entry in alive_indices");
    if (tid == 0) atomicAdd(&g_block_counter, 1);

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
    if (tid == 0) sdata[0] = local_ca_mean;
    __syncthreads();
    local_ca_mean = sdata[0];

    __shared__ int s_error_flag;
    __shared__ bool s_use_gradients;
    if (tid == 0) {
        s_error_flag = 0;
        s_use_gradients = training_mode->use_gradients;
    }
    __syncthreads();

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

        reconstruct_genome_from_archive(entry->parent_hash, (GPUElite*)organism->archive, organism->archive_size,
            entry->delta_indices, entry->delta_values, entry->num_deltas,
            entry->max_deltas, primary_genome, GENOME_SIZE, primary_parent_temp, organism->diresa_genome_weights);

        num_classes = organism->current_dataset->descriptor->num_classes;

        BehavioralDimensions dims;
        dims.derive_from_genome(primary_genome, entry->gradients);
        behavioral_dim = dims.total();

        arch.num_heads = entry->num_heads;
        arch.channels = entry->channels;
        arch.hidden_dim = entry->hidden_dim;
        arch.head_dim = entry->head_dim;
        arch.grid_size = entry->grid_size;
        float accuracy = entry->task_accuracy.value;
        arch.ca_gate_center = 2.0f - 1.5f * fminf(fmaxf(accuracy, 0.0f), 1.0f);

        num_features = arch.num_heads * arch.channels;

        component_grid = dim3(POOL_CAPACITY_MAX);  
        component_block = dim3(WARP_SIZE);  
        ca_grid = dim3(arch.grid_size / WMMA_TILE_DIM, arch.num_heads, 1);
        ca_block = dim3(WMMA_TILE_DIM, WMMA_TILE_DIM, 1);
        field_grid = dim3((arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM, (arch.grid_size + (WMMA_TILE_DIM - 1)) / WMMA_TILE_DIM);
        field_block = dim3(WMMA_TILE_DIM, WMMA_TILE_DIM);

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

    if (tid == 0 && organism->current_activation_grid_size == 0) {
        organism->current_activation_grid_size = arch.grid_size;
    }

    if (tid == 0 && arch.grid_size != organism->current_activation_grid_size) {
        organism->current_activation_grid_size = arch.grid_size;
    }
    if (tid == 0) printf("V:BLK_ARCH blk=%d grid=%d heads=%d hdim=%d ch=%d\n", blockIdx.x, arch.grid_size, arch.num_heads, arch.head_dim, arch.channels);

    if (s_use_gradients) {
        if (tid == 0) {
            DEVICE_FATAL_IF(organism == nullptr, "hybrid: organism is null");
            DEVICE_FATAL_IF(ca_state == nullptr, "hybrid: ca_state is null");
            DEVICE_FATAL_IF(training_mode == nullptr, "hybrid: training_mode is null");
            DEVICE_FATAL_IF(param_map == nullptr, "hybrid: param_map is null");
            DEVICE_FATAL_IF(training_mode->batch_images == nullptr, "hybrid: batch_images is null");
            DEVICE_FATAL_IF(training_mode->batch_labels == nullptr, "hybrid: batch_labels is null");
        }
        __syncthreads();
        if (tid == 0) printf("V:GRAD_ENTER blk=%d\n", blockIdx.x);

        {
            ADTape* tape = &ca_state->tape;
            int tape_capacity = tape->value_capacity;

            for (int i = tid; i < tape_capacity; i += blockDim.x) {
                tape->grad_buffer[i] = 0.0f;
                tape->value_levels[i] = 0;
            }

            if (tid == 0) {
                tape->current_size = 0;
                tape->current_value_idx = 0;
                tape->max_level = 0;
            }
        }
        __syncthreads();

        {
            TraceBuffer* trace_buffer = &ca_state->trace;
            if (tid == 0 && trace_buffer->traces != nullptr) {
                int trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
                if (trace_idx < trace_buffer->capacity) {
                    record_warp_metrics(&trace_buffer->traces[trace_idx], blockIdx.x);
                }
            }

            
            float* perception_saved = ca_state->perception_saved + s_wave_offsets.activations_offset;
            float* interaction_saved = ca_state->interaction_saved + s_wave_offsets.activations_offset;
            float* pre_gelu_saved = ca_state->pre_gelu_saved + s_wave_offsets.activations_offset;

            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads = arch.num_heads;
            int head_dim = arch.head_dim;
            int cells_per_grid = grid_size * grid_size;

            float* ca_input = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
            float* ca_output = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;

            int total_cells = batch_size * num_heads * cells_per_grid;
            if (tid == 0) printf("V:CA_LOOP_ENTER blk=%d total=%d batch=%d heads=%d cells=%d\n", blockIdx.x, total_cells, batch_size, num_heads, cells_per_grid);

            for (int work_idx = tid; work_idx < total_cells; work_idx += blockDim.x) {
                int cells_per_batch_head = cells_per_grid;
                int heads_times_cells = num_heads * cells_per_grid;

                int batch_id = work_idx / heads_times_cells;
                int remainder = work_idx % heads_times_cells;
                int head_id = remainder / cells_per_grid;
                int cell_idx = remainder % cells_per_grid;
                int cell_y = cell_idx / grid_size;
                int cell_x = cell_idx % grid_size;

                half* perc_w = &ca_state->perception_weights[head_id * channels * head_dim];
                half* inter_w = &ca_state->interaction_weights[head_id * head_dim * head_dim];
                half* val_w = &ca_state->value_weights[head_id * head_dim * channels];

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
                    perception[h] = activation_relu(acc);
                }
                if (tid == 0 && work_idx == 0) printf("V:CA_PERC_DONE blk=%d\n", blockIdx.x);

                float interaction[MAX_HEAD_DIM];
                float pre_gelu_vals[MAX_HEAD_DIM];
                float interaction_sum = 0.0f;
                for (int h = 0; h < head_dim; h++) {
                    float acc = 0.0f;
                    for (int j = 0; j < head_dim; j++) {
                        acc += perception[j] * __half2float(inter_w[j * head_dim + h]);
                    }
                    pre_gelu_vals[h] = acc;
                    float gelu = activation_gelu(acc);
                    interaction[h] = gelu;
                    interaction_sum += fabsf(gelu);
                }
                if (tid == 0 && work_idx == 0) printf("V:CA_INTER_DONE blk=%d\n", blockIdx.x);

                float output[MAX_CHANNELS];
                for (int c = 0; c < channels; c++) {
                    float acc = 0.0f;
                    for (int h = 0; h < head_dim; h++) {
                        acc += interaction[h] * __half2float(val_w[h * channels + c]);
                    }
                    output[c] = acc;
                }
                if (tid == 0 && work_idx == 0) printf("V:CA_VAL_DONE blk=%d\n", blockIdx.x);

                float gate = activation_sigmoid(interaction_sum / (float)head_dim - arch.ca_gate_center);

                int saved_base = batch_id * num_heads * cells_per_grid * head_dim +
                                head_id * cells_per_grid * head_dim +
                                cell_y * grid_size * head_dim +
                                cell_x * head_dim;
                for (int h = 0; h < head_dim; h++) {
                    perception_saved[saved_base + h] = perception[h];
                    interaction_saved[saved_base + h] = interaction[h];
                    pre_gelu_saved[saved_base + h] = pre_gelu_vals[h];
                }
                if (tid == 0 && work_idx == 0) printf("V:CA_SAVE_DONE blk=%d\n", blockIdx.x);

                int out_idx = batch_id * num_heads * cells_per_grid * channels +
                             head_id * cells_per_grid * channels +
                             cell_y * grid_size * channels +
                             cell_x * channels;
                for (int c = 0; c < channels; c++) {
                    float input_val = neighborhood[1][1][c];
                    ca_output[out_idx + c] = input_val * (1.0f - gate) + output[c] * gate;
                }
                if (tid == 0 && work_idx == 0) printf("V:CA_OUT_DONE blk=%d\n", blockIdx.x);
            }
            if (tid == 0) printf("V:CA_LOOP_EXIT blk=%d\n", blockIdx.x);
        }
        __syncthreads();
        if (tid == 0) printf("V:CA_FWD_DONE blk=%d\n", blockIdx.x);

        __syncthreads();
        if (s_error_flag) return;

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

            
            float* wave_ca_output = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            float* wave_affinity = organism->buffers->batch_affinity_reduced + s_wave_offsets.affinity_offset;
            float* wave_flow = organism->buffers->batch_flow_field + s_wave_offsets.flow_offset;
            float* wave_reint = organism->buffers->batch_reintegration_buffer + s_wave_offsets.ca_states_offset;
            float* wave_ca_pool = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
            float* wave_prev_conc = organism->buffers->batch_prev_concentration + s_wave_offsets.ca_states_offset;

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
                    A_sum_center += wave_ca_pool[conc_batch_offset + cell_idx * channels + c];
                    A_sum_E += wave_ca_pool[conc_batch_offset + (y * grid_size + x_E) * channels + c];
                    A_sum_N += wave_ca_pool[conc_batch_offset + (y_N * grid_size + x) * channels + c];
                }

                float2 F = FlowLeniaOps::compute_flow_differentiable(
                    U_center, U_E, U_N, A_sum_center, A_sum_E, A_sum_N,
                    flow_beta_A, flow_n, flow_alpha_min, flow_alpha_max, flow_sharpness
                );

                int flow_idx = batch_idx * total_cells * 2 + cell_idx * 2;
                wave_flow[flow_idx + 0] = F.x;
                wave_flow[flow_idx + 1] = F.y;
            }

            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                wave_reint[idx] = 0.0f;
            }
            __syncthreads();

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
                const float* batch_conc = wave_ca_pool + conc_batch_offset;

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

            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                wave_ca_pool[idx] = wave_reint[idx];
            }
            __syncthreads();

            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                wave_prev_conc[idx] = wave_ca_pool[idx];
            }
        }
        __syncthreads();
        if (tid == 0) printf("V:FLOW_DONE blk=%d\n", blockIdx.x);

        float* ca_output_grad = nullptr;

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int spatial_size = grid_size * grid_size;

            
            float* ca_out = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
            float* pooling_weights = training_mode->classifier->pooling_weights;

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

            if (tid == 0) {
                BehavioralDimensions dims;
                dims.derive_from_genome(primary_genome, entry->gradients);
                int task_dim = dims.task_dim;

                DEVICE_FATAL_IF(entry->diresa_task_weights->input_dim != num_features,
                    "hybrid_lifecycle: diresa_task_weights->input_dim mismatch with num_features (entry %d)", entry_idx);
                for (int b = 0; b < batch_size; b++) {
                    diresa_encode(&features[b * num_features], &organism->task_coords_pool[b * task_dim], entry->diresa_task_weights);
                }
            }
            __syncthreads();

            float* logits = organism->gradient_logits_pool + entry_idx * batch_size * num_classes;
            float* fc_weights = training_mode->classifier->fc_weights;
            float* fc_bias = training_mode->classifier->fc_bias;

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

            float* loss_out = &organism->gradient_loss_pool[entry_idx];
            if (tid == 0) {
                *loss_out = 0.0f;
            }
            __syncthreads();

            float* logit_grads = organism->gradient_logit_grads_pool + entry_idx * batch_size * num_classes;
            int* batch_labels = training_mode->batch_labels;

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
                        float local_val = (lane_id < num_classes) ? batch_logits[lane_id] : -INFINITY;
                        float max_logit = warp_reduce_max(local_val);
                        max_logit = __shfl_sync(0xffffffff, max_logit, 0);

                        float local_exp = (lane_id < num_classes) ? expf(local_val - max_logit) : 0.0f;
                        float sum_exp = warp_reduce_sum(local_exp);
                        sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

                        if (lane_id < num_classes) {
                            float prob = local_exp / sum_exp;
                            float grad = prob - ((lane_id == label) ? 1.0f : 0.0f);
                            batch_grads[lane_id] = grad / batch_size;
                        }

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

                int pool_capacity = organism->pool->capacity;
                int entries_per_side = (int)ceilf(sqrtf((float)pool_capacity));
                int entry_region_size = grid_size / entries_per_side;

                int entry_tile_y = entry_idx / entries_per_side;
                int entry_tile_x = entry_idx % entries_per_side;
                int entry_offset_y = entry_tile_y * entry_region_size;
                int entry_offset_x = entry_tile_x * entry_region_size;

                int tiles_per_side = (int)ceilf(sqrtf((float)batch_size));
                int tile_size = entry_region_size / tiles_per_side;

                for (int b = 0; b < batch_size; b++) {
                    float* batch_logits = &logits[b * num_classes];
                    int true_label = batch_labels[b];

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

                    float sum_exp = 0.0f;
                    for (int c = 0; c < num_classes; c++) {
                        sum_exp += expf(batch_logits[c] - max_logit);
                    }
                    float confidence = expf(batch_logits[pred_class] - max_logit) / sum_exp;
                    avg_confidence += confidence;
                }

                float accuracy = (float)correct / batch_size;
                avg_confidence /= batch_size;

                int ema_slot = GenomeParamTable::accuracy_ema_smoothing;
                float ema_smoothing = genome_slot_to_unit(primary_genome, ema_slot);
                ema_smoothing = EMA_SMOOTHING_MIN + ema_smoothing * (EMA_SMOOTHING_MAX - EMA_SMOOTHING_MIN);

                int current_gen = organism->generation;
                if (training_mode->is_train_batch) {
                    float smoothed = ema_smoothing * entry->train_accuracy.value + (1.0f - ema_smoothing) * accuracy;
                    entry->train_accuracy.set_computed(smoothed, current_gen, entry->genome_hash);
                } else {
                    entry->test_accuracy.set_computed(accuracy, current_gen, entry->genome_hash);
                }
                entry->task_accuracy.set_computed(accuracy, current_gen, entry->genome_hash);
                entry->avg_confidence.set_computed(avg_confidence, current_gen, entry->genome_hash);
                for (int c = 0; c < NUM_CLASSES_MAX; c++) {
                    entry->per_class_correct[c] = local_per_class_correct[c];
                    entry->per_class_total[c] = local_per_class_total[c];
                }
            }
            __syncthreads();
        }
        __syncthreads();

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {

            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int spatial_size = grid_size * grid_size;

            
            float* ca_out = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            ca_output_grad = organism->buffers->ca_output_grad_buffer + s_wave_offsets.ca_output_offset;

            float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
            float* logit_grads = organism->gradient_logit_grads_pool + entry_idx * batch_size * num_classes;
            float* features_grad = organism->features_grad + entry_idx * batch_size * num_features;
            float* fc_weights = training_mode->classifier->fc_weights;
            float* fc_weights_grad = organism->fc_weights_grad;
            float* fc_bias_grad = organism->fc_bias_grad;
            float* pooling_weights = training_mode->classifier->pooling_weights;
            float* pooling_weights_grad = organism->pooling_weights_grad;

            int fc_weights_size = num_classes * num_features;
            int features_grad_size = batch_size * num_features;
            int ca_grad_size = batch_size * num_heads_local * spatial_size * channels;

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

            int total_class_bwd = batch_size * num_classes;
            for (int work_idx = tid; work_idx < total_class_bwd; work_idx += blockDim.x) {
                int batch_idx = work_idx / num_classes;
                int class_idx = work_idx % num_classes;

                float logit_grad = logit_grads[batch_idx * num_classes + class_idx];

                atomicAdd(&fc_bias_grad[class_idx], logit_grad);

                for (int feat = 0; feat < num_features; feat++) {
                    float feature_val = features[batch_idx * num_features + feat];
                    float weight_val = fc_weights[class_idx * num_features + feat];
                    atomicAdd(&fc_weights_grad[class_idx * num_features + feat], logit_grad * feature_val);
                    atomicAdd(&features_grad[batch_idx * num_features + feat], logit_grad * weight_val);
                }
            }
            __syncthreads();

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

                    float ca_avg = 0.0f;
                    for (int spatial = 0; spatial < spatial_size; spatial++) {
                        int idx = base_idx + spatial * channels + channel;
                        ca_avg += ca_out[idx];
                    }
                    ca_avg /= spatial_size;

                    atomicAdd(&pooling_weights_grad[feature_idx], feat_grad * ca_avg);

                    float ca_grad_val = feat_grad * pooling_weights[feature_idx] / spatial_size;
                    for (int spatial = 0; spatial < spatial_size; spatial++) {
                        int idx = base_idx + spatial * channels + channel;
                        atomicAdd(&ca_output_grad[idx], ca_grad_val);
                    }
                }
            }
            __syncthreads();
        }

        {
            TraceBuffer* trace_buffer = &ca_state->trace;
            if (tid == 0 && trace_buffer->traces != nullptr) {
                int trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
                if (trace_idx < trace_buffer->capacity) {
                    record_warp_metrics(&trace_buffer->traces[trace_idx], blockIdx.x);
                }
            }

            int num_cells = arch.grid_size * arch.grid_size;
            int total_samples = training_mode->batch_size * num_cells;

            
            float* dL_dperception = organism->buffers->dL_dperception_buffer + s_wave_offsets.activations_offset;
            float* dL_dinteraction = organism->buffers->dL_dinteraction_buffer + s_wave_offsets.activations_offset;
            if (tid == 0) printf("V:BWD_ENTER blk=%d\n", blockIdx.x);

            {

            
            char* backward_ws_base = organism->buffers->backward_workspace + s_wave_offsets.backward_ws_offset;
            BackwardWorkspaceLayout ws_layout = compute_backward_ws_layout(entry);

            
            half* ws_fp16_a = (half*)(backward_ws_base + ws_layout.fp16_a_offset);
            half* ws_fp16_b = (half*)(backward_ws_base + ws_layout.fp16_b_offset);
            float* ws_dW = (float*)(backward_ws_base + ws_layout.dW_offset);
            float* ws_dI = (float*)(backward_ws_base + ws_layout.dI_offset);
            half* ws_W_T = (half*)(backward_ws_base + ws_layout.W_T_offset);
            float* ws_im2col = (float*)(backward_ws_base + ws_layout.im2col_offset);
            float* ws_dpregelu = (float*)(backward_ws_base + ws_layout.dpregelu_offset);

            
            float* perception_saved = ca_state->perception_saved + s_wave_offsets.activations_offset;
            float* interaction_saved = ca_state->interaction_saved + s_wave_offsets.activations_offset;
            float* pre_gelu_saved = ca_state->pre_gelu_saved + s_wave_offsets.activations_offset;

            int I_head_stride = num_cells * arch.head_dim;
            int I_batch_stride = arch.num_heads * I_head_stride;
            int V_head_stride = num_cells * arch.channels;
            int V_batch_stride = arch.num_heads * V_head_stride;
            int ws_dW_value_stride = arch.head_dim * arch.channels;
            int ws_W_T_value_stride = arch.channels * arch.head_dim;
            int ws_dW_interaction_stride = arch.head_dim * arch.head_dim;
            int ws_W_T_interaction_stride = arch.head_dim * arch.head_dim;

            int warp_id = tid / WARP_SIZE;
            int lane_id = tid % WARP_SIZE;
            int num_warps = blockDim.x / WARP_SIZE;

            
            int chunk_ws_a_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
            int chunk_ws_b_stride = BACKWARD_CHUNK_SAMPLES * arch.channels;

            
            {
                int total_dW = arch.num_heads * arch.head_dim * arch.channels;
                for (int idx = tid; idx < total_dW; idx += blockDim.x) {
                    ws_dW[idx] = 0.0f;
                }
            }
            __syncthreads();

            
            int num_chunks = (total_samples + BACKWARD_CHUNK_SAMPLES - 1) / BACKWARD_CHUNK_SAMPLES;
            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                int chunk_start = chunk_idx * BACKWARD_CHUNK_SAMPLES;
                int chunk_samples = min(BACKWARD_CHUNK_SAMPLES, total_samples - chunk_start);
                
                int chunk_samples_aligned = (chunk_samples / WMMA_TILE_DIM) * WMMA_TILE_DIM;
                if (chunk_samples_aligned == 0) continue;

                
                {
                    int total_I = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    for (int idx = tid; idx < total_I; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int sample_in_chunk = remainder / arch.head_dim;
                        int dim_idx = remainder % arch.head_dim;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_id = global_sample % num_cells;
                        int src_idx = head_id * I_head_stride + batch_id * I_batch_stride + cell_id * arch.head_dim + dim_idx;
                        int dst_idx = head_id * chunk_ws_a_stride + sample_in_chunk * arch.head_dim + dim_idx;
                        ws_fp16_a[dst_idx] = __float2half(interaction_saved[src_idx]);
                    }
                }
                __syncthreads();

                
                {
                    int total_V = arch.num_heads * chunk_samples_aligned * arch.channels;
                    for (int idx = tid; idx < total_V; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.channels);
                        int remainder = idx % (chunk_samples_aligned * arch.channels);
                        int sample_in_chunk = remainder / arch.channels;
                        int ch_idx = remainder % arch.channels;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_id = global_sample % num_cells;
                        int src_idx = head_id * V_head_stride + batch_id * V_batch_stride + cell_id * arch.channels + ch_idx;
                        int dst_idx = head_id * chunk_ws_b_stride + sample_in_chunk * arch.channels + ch_idx;
                        ws_fp16_b[dst_idx] = __float2half(ca_output_grad[src_idx]);
                    }
                }
                __syncthreads();

                
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
                            const half* A_head = ws_fp16_a + head_id * chunk_ws_a_stride;
                            const half* B_head = ws_fp16_b + head_id * chunk_ws_b_stride;
                            float* C_head = ws_dW + head_id * ws_dW_value_stride;

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

                            
                            nvcuda::wmma::load_matrix_sync(c_frag, C_head + tile_row * arch.channels + tile_col, arch.channels, nvcuda::wmma::mem_row_major);

                            for (int k_tile = 0; k_tile < chunk_samples_aligned; k_tile += WMMA_TILE_DIM) {
                                nvcuda::wmma::load_matrix_sync(a_frag, A_head + k_tile * arch.head_dim + tile_row, arch.head_dim);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + k_tile * arch.channels + tile_col, arch.channels);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + tile_row * arch.channels + tile_col, c_frag, arch.channels, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                __syncthreads();
            } 

            
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
                    int h = bx + local_x;
                    int c = by + local_y;

                    if (c < arch.channels && h < arch.head_dim) {
                        const half* W_head = ca_state->value_weights + head_id * arch.head_dim * arch.channels;
                        half* W_T_head = ws_W_T + head_id * ws_W_T_value_stride;
                        W_T_head[c * arch.head_dim + h] = W_head[h * arch.channels + c];
                    }
                }
            }
            __syncthreads();

            
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
                        half* W_T_head = ws_W_T + arch.num_heads * ws_W_T_value_stride + head_id * ws_W_T_interaction_stride;
                        W_T_head[x * arch.head_dim + y] = W_head[y * arch.head_dim + x];
                    }
                }
            }
            __syncthreads();

            
            {
                int total_interaction = arch.num_heads * arch.head_dim * arch.head_dim;
                int total_perception = arch.num_heads * arch.channels * arch.head_dim;
                int total_zero = total_interaction + total_perception;
                float* ws_dW_interaction = ws_dW + arch.num_heads * ws_dW_value_stride;
                for (int idx = tid; idx < total_zero; idx += blockDim.x) {
                    ws_dW_interaction[idx] = 0.0f;
                }
            }
            __syncthreads();

            
            int chunk_ws_dI_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
            int chunk_ws_dpregelu_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
            int chunk_ws_pooled_stride = BACKWARD_CHUNK_SAMPLES * arch.channels;
            float* ws_dW_interaction = ws_dW + arch.num_heads * ws_dW_value_stride;
            float* ws_dW_perception = ws_dW_interaction + arch.num_heads * ws_dW_interaction_stride;
            half* ws_W_T_interaction = ws_W_T + arch.num_heads * ws_W_T_value_stride;

            
            float* d_ca_input = organism->batch_ca_input_grads ?
                organism->batch_ca_input_grads + s_wave_offsets.ca_states_offset : nullptr;
            if (d_ca_input != nullptr) {
                for (int idx = tid; idx < total_samples * arch.channels; idx += blockDim.x) {
                    d_ca_input[idx] = 0.0f;
                }
                __syncthreads();
            }

            
            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                int chunk_start = chunk_idx * BACKWARD_CHUNK_SAMPLES;
                int chunk_samples = min(BACKWARD_CHUNK_SAMPLES, total_samples - chunk_start);
                int chunk_samples_aligned = (chunk_samples / WMMA_TILE_DIM) * WMMA_TILE_DIM;
                if (chunk_samples_aligned == 0) continue;

                
                
                {
                    int total_V = arch.num_heads * chunk_samples_aligned * arch.channels;
                    for (int idx = tid; idx < total_V; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.channels);
                        int remainder = idx % (chunk_samples_aligned * arch.channels);
                        int sample_in_chunk = remainder / arch.channels;
                        int ch_idx = remainder % arch.channels;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_id = global_sample % num_cells;
                        int src_idx = head_id * V_head_stride + batch_id * V_batch_stride + cell_id * arch.channels + ch_idx;
                        int dst_idx = head_id * chunk_ws_b_stride + sample_in_chunk * arch.channels + ch_idx;
                        ws_fp16_b[dst_idx] = __float2half(ca_output_grad[src_idx]);
                    }
                }
                __syncthreads();

                
                {
                    int tiles_M = (chunk_samples_aligned + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int tiles_N = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = tiles_M * tiles_N * arch.num_heads;

                    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                        int head_id = tile_idx / (tiles_M * tiles_N);
                        int tile_flat = tile_idx % (tiles_M * tiles_N);
                        int warpM = tile_flat / tiles_N;
                        int warpN = tile_flat % tiles_N;
                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        if (tile_row < chunk_samples_aligned && tile_col < arch.head_dim) {
                            const half* A_head = ws_fp16_b + head_id * chunk_ws_b_stride;
                            const half* B_head = ws_W_T + head_id * ws_W_T_value_stride;
                            float* C_head = ws_dI + head_id * chunk_ws_dI_stride;

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

                
                {
                    int total_elem = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    for (int idx = tid; idx < total_elem; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int sample_in_chunk = remainder / arch.head_dim;
                        int dim_idx = remainder % arch.head_dim;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_id = global_sample % num_cells;

                        int ws_idx = head_id * chunk_ws_dI_stride + sample_in_chunk * arch.head_dim + dim_idx;
                        int out_idx = head_id * I_head_stride + batch_id * I_batch_stride + cell_id * arch.head_dim + dim_idx;
                        int saved_idx = batch_id * I_batch_stride + head_id * I_head_stride + cell_id * arch.head_dim + dim_idx;

                        float dL_dI_val = ws_dI[ws_idx];
                        dL_dinteraction[out_idx] = dL_dI_val;
                        ws_dpregelu[head_id * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + dim_idx] =
                            activation_gelu_backward(pre_gelu_saved[saved_idx], dL_dI_val);
                    }
                }
                __syncthreads();

                
                
                {
                    int total_P = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    for (int idx = tid; idx < total_P; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int sample_in_chunk = remainder / arch.head_dim;
                        int dim_idx = remainder % arch.head_dim;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_id = global_sample % num_cells;
                        int src_idx = head_id * I_head_stride + batch_id * I_batch_stride + cell_id * arch.head_dim + dim_idx;
                        int dst_idx = head_id * chunk_ws_a_stride + sample_in_chunk * arch.head_dim + dim_idx;
                        ws_fp16_a[dst_idx] = __float2half(perception_saved[src_idx]);
                    }
                }
                
                {
                    int total_D = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    for (int idx = tid; idx < total_D; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int src_idx = head_id * chunk_ws_dpregelu_stride + remainder;
                        int dst_idx = head_id * chunk_ws_a_stride + remainder;
                        ws_fp16_b[dst_idx] = __float2half(ws_dpregelu[src_idx]);
                    }
                }
                __syncthreads();

                
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
                            const half* A_head = ws_fp16_a + head_id * chunk_ws_a_stride;
                            const half* B_head = ws_fp16_b + head_id * chunk_ws_a_stride;
                            float* C_head = ws_dW_interaction + head_id * ws_dW_interaction_stride;

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                            nvcuda::wmma::load_matrix_sync(c_frag, C_head + tile_row * arch.head_dim + tile_col, arch.head_dim, nvcuda::wmma::mem_row_major);

                            for (int k_tile = 0; k_tile < chunk_samples_aligned; k_tile += WMMA_TILE_DIM) {
                                nvcuda::wmma::load_matrix_sync(a_frag, A_head + k_tile * arch.head_dim + tile_row, arch.head_dim);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + k_tile * arch.head_dim + tile_col, arch.head_dim);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + tile_row * arch.head_dim + tile_col, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                __syncthreads();

                
                {
                    int tiles_M = (chunk_samples_aligned + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int tiles_N = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = tiles_M * tiles_N * arch.num_heads;

                    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                        int head_id = tile_idx / (tiles_M * tiles_N);
                        int tile_flat = tile_idx % (tiles_M * tiles_N);
                        int warpM = tile_flat / tiles_N;
                        int warpN = tile_flat % tiles_N;
                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        if (tile_row < chunk_samples_aligned && tile_col < arch.head_dim) {
                            const half* A_head = ws_fp16_b + head_id * chunk_ws_a_stride;
                            const half* B_head = ws_W_T_interaction + head_id * ws_W_T_interaction_stride;
                            float* C_head = ws_dI + head_id * chunk_ws_dI_stride;

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

                
                {
                    int total_elem = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    for (int idx = tid; idx < total_elem; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int sample_in_chunk = remainder / arch.head_dim;
                        int dim_idx = remainder % arch.head_dim;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_id = global_sample % num_cells;

                        int ws_idx = head_id * chunk_ws_dI_stride + sample_in_chunk * arch.head_dim + dim_idx;
                        int out_idx = head_id * I_head_stride + batch_id * I_batch_stride + cell_id * arch.head_dim + dim_idx;

                        float dL_dP_val = ws_dI[ws_idx];
                        dL_dperception[out_idx] = dL_dP_val;
                        float perc_val = perception_saved[out_idx];
                        ws_dpregelu[head_id * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + dim_idx] =
                            dL_dP_val * ((perc_val > 0.0f) ? 1.0f : 0.0f);
                    }
                }
                __syncthreads();

                
                
                {
                    for (int idx = tid; idx < chunk_samples_aligned; idx += blockDim.x) {
                        int global_sample = chunk_start + idx;
                        int batch_id = global_sample / num_cells;
                        int cell_idx = global_sample % num_cells;
                        int cell_y = cell_idx / arch.grid_size;
                        int cell_x = cell_idx % arch.grid_size;

                        const float* input_batch = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;

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
                            ws_im2col[idx * arch.channels + c] = sum;
                        }
                    }
                }
                __syncthreads();

                
                {
                    int total_conv = chunk_samples_aligned * arch.channels;
                    for (int idx = tid; idx < total_conv; idx += blockDim.x) {
                        ws_fp16_a[idx] = __float2half(ws_im2col[idx]);
                    }
                }
                
                {
                    int total_D = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    for (int idx = tid; idx < total_D; idx += blockDim.x) {
                        ws_fp16_b[idx] = __float2half(ws_dpregelu[idx]);
                    }
                }
                __syncthreads();

                
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
                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        if (tile_row < arch.channels && tile_col < arch.head_dim) {
                            const half* A_ptr = ws_fp16_a;
                            const half* B_head = ws_fp16_b + head_id * chunk_samples_aligned * arch.head_dim;
                            float* C_head = ws_dW_perception + head_id * ws_dW_perception_stride;

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                            nvcuda::wmma::load_matrix_sync(c_frag, C_head + tile_row * arch.head_dim + tile_col, arch.head_dim, nvcuda::wmma::mem_row_major);

                            for (int k_tile = 0; k_tile < chunk_samples_aligned; k_tile += WMMA_TILE_DIM) {
                                nvcuda::wmma::load_matrix_sync(a_frag, A_ptr + k_tile * arch.channels + tile_row, arch.channels);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + k_tile * arch.head_dim + tile_col, arch.head_dim);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + tile_row * arch.head_dim + tile_col, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                __syncthreads();

                
                
                if (d_ca_input != nullptr) {
                    
                    for (int idx = tid; idx < chunk_samples_aligned * arch.channels; idx += blockDim.x) {
                        int sample_in_chunk = idx / arch.channels;
                        int channel_idx = idx % arch.channels;

                        float d_input_accum = 0.0f;
                        for (int h = 0; h < arch.num_heads; h++) {
                            for (int hd = 0; hd < arch.head_dim; hd++) {
                                int W_idx = param_map->perception_start[h] + channel_idx * arch.head_dim + hd;
                                float W_val = __half2float(ca_state->perception_weights[W_idx]);
                                float dprerelu_val = ws_dpregelu[h * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + hd];
                                d_input_accum += W_val * dprerelu_val;
                            }
                        }
                        ws_im2col[idx] = d_input_accum;
                    }
                    __syncthreads();

                    
                    for (int idx = tid; idx < chunk_samples_aligned * arch.channels; idx += blockDim.x) {
                        int sample_in_chunk = idx / arch.channels;
                        int channel_idx = idx % arch.channels;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_idx = global_sample % num_cells;
                        int cell_y = cell_idx / arch.grid_size;
                        int cell_x = cell_idx % arch.grid_size;
                        float d_pooled_val = ws_im2col[idx];

                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                int ny = cell_y + dy;
                                int nx = cell_x + dx;
                                if (ny >= 0 && ny < arch.grid_size && nx >= 0 && nx < arch.grid_size) {
                                    int out_cell_idx = ny * arch.grid_size + nx;
                                    int out_idx = batch_id * num_cells * arch.channels + out_cell_idx * arch.channels + channel_idx;
                                    atomicAdd(&d_ca_input[out_idx], d_pooled_val);
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
            } 

            if (tid == 0) printf("V:BWD_CHUNKS_DONE blk=%d\n", blockIdx.x);

            
            {
                int total_grads = arch.num_heads * arch.head_dim * arch.head_dim;
                for (int idx = tid; idx < total_grads; idx += blockDim.x) {
                    int head_id = idx / (arch.head_dim * arch.head_dim);
                    int local_idx = idx % (arch.head_dim * arch.head_dim);
                    int src_idx = head_id * ws_dW_interaction_stride + local_idx;
                    int dst_idx = param_map->interaction_start[head_id] + local_idx;
                    ca_state->tape.grad_buffer[dst_idx] = ws_dW_interaction[src_idx];
                }
            }
            __syncthreads();

            
            {
                int ws_dW_perception_stride = arch.channels * arch.head_dim;
                int weights_per_head = arch.channels * arch.head_dim;
                int total_grads = arch.num_heads * weights_per_head;
                for (int idx = tid; idx < total_grads; idx += blockDim.x) {
                    int head_id = idx / weights_per_head;
                    int local_idx = idx % weights_per_head;
                    int src_idx = head_id * ws_dW_perception_stride + local_idx;
                    int dst_idx = param_map->perception_start[head_id] + local_idx;
                    ca_state->tape.grad_buffer[dst_idx] = ws_dW_perception[src_idx];
                }
            }
            __syncthreads();

            
            
            if (d_ca_input != nullptr) {
                float* grad_conc = organism->buffers->grad_concentration_buffer;
                for (int cell = tid; cell < num_cells; cell += blockDim.x) {
                    grad_conc[cell] = d_ca_input[cell * arch.channels];
                }
            }
            __syncthreads();

            if (d_ca_input != nullptr && tid == 0) {
                dim3 diff_grid((arch.grid_size + 15) / 16, (arch.grid_size + 15) / 16);
                dim3 diff_block(16, 16);

                float ctx_metabolic = entry->fitness.value;
                float ctx_stress = entry->hunger.value;
                float ctx_morphogen = organism->chemical_field->cached_mean;
                float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
                float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
                float ctx_learning = training_mode->learning_rate;
                float ctx_performance = entry->task_accuracy.value;

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
            if (tid == 0 && blockIdx.x == 0) printf("V:HYB_diffusion_bwd_done\n");

            }  

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

                __shared__ float s_d_beta_A;
                __shared__ float s_d_n;
                if (tid == 0) {
                    s_d_beta_A = 0.0f;
                    s_d_n = 0.0f;
                }
                __syncthreads();

                float local_d_beta_A = 0.0f;
                float local_d_n = 0.0f;

                int total_work = batch_size * total_cells;
                for (int work_idx = tid; work_idx < total_work; work_idx += blockDim.x) {
                    int batch_idx = work_idx / total_cells;
                    int cell_idx = work_idx % total_cells;
                    int source_x = cell_idx % grid_size;
                    int source_y = cell_idx / grid_size;

                    int batch_offset = batch_idx * total_cells;
                    int flow_idx = batch_offset * 2 + cell_idx * 2;
                    float Fx = organism->buffers->batch_flow_field[s_wave_offsets.flow_offset + flow_idx + 0];
                    float Fy = organism->buffers->batch_flow_field[s_wave_offsets.flow_offset + flow_idx + 1];

                    int conc_batch_offset = batch_idx * total_cells * channels;
                    const float* batch_conc = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset + conc_batch_offset;

                    float d_flow_x_accum = 0.0f;
                    float d_flow_y_accum = 0.0f;

                    for (int c = 0; c < channels; c++) {
                        float source_mass = batch_conc[cell_idx * channels + c];
                        float d_source_mass, d_flow_x_local, d_flow_y_local;

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

                atomicAdd(&s_d_beta_A, local_d_beta_A);
                atomicAdd(&s_d_n, local_d_n);
                __syncthreads();

                if (tid == 0) {
                    float d_beta_A = s_d_beta_A / (batch_size * total_cells);
                    float d_n = s_d_n / (batch_size * total_cells);

                    float ctx_metabolic = entry->fitness.value;
                    float ctx_stress = entry->hunger.value;
                    float ctx_morphogen = local_ca_mean;
                    float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
                    float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
                    float ctx_learning = organism->telemetry->diresa_evolution.behavioral_drift_rate;
                    float ctx_performance = organism->telemetry->task_performance.accuracy;

                    TrainingParams train_params;
                    float flow_lr = train_params.get_flow_lenia_lr(primary_genome, entry->gradients,
                        ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity,
                        ctx_niche, ctx_learning, ctx_performance);
                    float clip_norm = train_params.get_gradient_clip_norm(primary_genome, entry->gradients,
                        ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity,
                        ctx_niche, ctx_learning, ctx_performance);

                    d_beta_A = fmaxf(-clip_norm, fminf(clip_norm, d_beta_A));
                    d_n = fmaxf(-clip_norm, fminf(clip_norm, d_n));

                    float new_beta_A = entry->flow_beta_A - flow_lr * d_beta_A;
                    float new_n = entry->flow_n - flow_lr * d_n;

                    entry->flow_beta_A = fmaxf(FLOW_LENIA_BETA_A_MIN, fminf(FLOW_LENIA_BETA_A_MAX, new_beta_A));
                    entry->flow_n = fmaxf(FLOW_LENIA_N_MIN, fminf(FLOW_LENIA_N_MAX, new_n));

                    organism->buffers->flow_beta_A_grad = d_beta_A;
                    organism->buffers->flow_n_grad = d_n;
                }
                __syncthreads();
            }
        }  

        {
            int num_heads = arch.num_heads;
            int perception_per_head = arch.channels * arch.head_dim;
            int interaction_per_head = arch.head_dim * arch.head_dim;
            int value_per_head = arch.head_dim * arch.channels;
            int params_per_head = perception_per_head + interaction_per_head + value_per_head;

            float* grad_buf = ca_state->tape.grad_buffer;

            __shared__ float head_grad_sq[NUM_HEADS_MAX];
            for (int h = 0; h < num_heads && h < NUM_HEADS_MAX; h++) {
                float local_sq = 0.0f;
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
                    head_grad_sq[h] = sqrtf(local_sq / (float)params_per_head);
                }
                __syncthreads();
            }

            if (tid == 0) {
                float total_sq = 0.0f;
                for (int h = 0; h < num_heads; h++) {
                    total_sq += head_grad_sq[h] * head_grad_sq[h];
                }

                DEVICE_FATAL_IF(total_sq < 1e-12f, "effective_rank: zero gradient magnitude after backward pass - training catastrophically broken");

                float entropy = 0.0f;
                for (int h = 0; h < num_heads; h++) {
                    float g = head_grad_sq[h];
                    float p = (g * g) / total_sq;
                    if (p > 1e-12f) {
                        entropy -= p * logf(p);
                    }
                }
                float eff_rank = expf(entropy);
                float clamped_rank = fmaxf(1.0f, fminf((float)num_heads, eff_rank));
                entry->effective_rank.set_computed(clamped_rank, organism->generation, entry->genome_hash);
            }
            __syncthreads();
        }
        if (tid == 0 && blockIdx.x == 0) printf("V:HYB_effrank_done\n");

        if (entry_idx == 0 && training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
            int perception_params = arch.num_heads * arch.channels * arch.head_dim;
            int interaction_params = arch.num_heads * arch.head_dim * arch.head_dim;
            int value_params = arch.num_heads * arch.head_dim * arch.channels;

            float ctx_metabolic = entry->fitness.value;
            float ctx_stress = entry->hunger.value;
            float ctx_morphogen = local_ca_mean;

            TrainingParams train_params;
            float adam_beta1 = train_params.get_adam_beta1(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
            float adam_beta2 = train_params.get_adam_beta2(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
            float adam_epsilon = train_params.get_adam_epsilon(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
            float gradient_clip_norm = train_params.get_gradient_clip_norm(primary_genome, entry->gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

            float lr = training_mode->learning_rate;
            int timestep = training_mode->adam_timestep;

            if (tid == 0) {
                int ts = timestep + 1;

                int blocks_p = (perception_params + 255) / 256;
                adam_update_fp16_kernel<<<blocks_p, 256>>>(
                    ca_state->perception_weights,
                    ca_state->tape.grad_buffer,
                    training_mode->adam_m,
                    training_mode->adam_v,
                    perception_params,
                    lr, adam_beta1, adam_beta2, adam_epsilon, ts, gradient_clip_norm
                );

                int blocks_i = (interaction_params + 255) / 256;
                adam_update_fp16_kernel<<<blocks_i, 256>>>(
                    ca_state->interaction_weights,
                    ca_state->tape.grad_buffer + perception_params,
                    training_mode->adam_m + training_mode->perception_size,
                    training_mode->adam_v + training_mode->perception_size,
                    interaction_params,
                    lr, adam_beta1, adam_beta2, adam_epsilon, ts, gradient_clip_norm
                );

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

            if (tid == 0) {
                int ts = timestep + 1;
                int fc_weights_size = num_classes * num_features;

                int blocks_pool = (arch.channels + 255) / 256;
                adam_update_kernel<<<blocks_pool, 256>>>(
                    training_mode->classifier->pooling_weights,
                    organism->pooling_weights_grad,
                    organism->adam_m_pooling,
                    organism->adam_v_pooling,
                    arch.channels,
                    lr, adam_beta1, adam_beta2, adam_epsilon, ts, gradient_clip_norm
                );

                int blocks_fc = (fc_weights_size + 255) / 256;
                adam_update_kernel<<<blocks_fc, 256>>>(
                    training_mode->classifier->fc_weights,
                    organism->fc_weights_grad,
                    organism->adam_m_fc_weights,
                    organism->adam_v_fc_weights,
                    fc_weights_size,
                    lr, adam_beta1, adam_beta2, adam_epsilon, ts, gradient_clip_norm
                );

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

            if (tid == 0) {
                training_mode->adam_timestep++;
            }
            __syncthreads();

        }  
    }  

    if (entry_idx == 0) {
        float* component_workspace_genomes = organism->buffers->component_workspace_genomes_buffer;
        GPUElite* archive = organism->archive;
        int archive_size_val = organism->archive_size;

        int alive_ct = pool->alive_indices_count;
        for (int compact = tid; compact < alive_ct; compact += blockDim.x) {
            int eid = pool->alive_indices[compact];
            DEVICE_FATAL_IF(!pool->alive_flags[eid], "hybrid_lifecycle: dead entry in alive_indices (metrics loop)");

            PoolEntry* ent = &pool->entries[eid];
            float* eid_primary_genome = &component_workspace_genomes[eid * 2 * GENOME_SIZE];
            float* eid_parent_temp = &component_workspace_genomes[eid * 2 * GENOME_SIZE + GENOME_SIZE];

            reconstruct_genome_from_archive(ent->parent_hash, archive, archive_size_val,
                ent->delta_indices, ent->delta_values, ent->num_deltas,
                ent->max_deltas, eid_primary_genome, GENOME_SIZE, eid_parent_temp, organism->diresa_genome_weights);

            float gen_gap_val = fabsf(ent->train_accuracy.value - ent->test_accuracy.value);
            ent->generalization_gap.set_computed(gen_gap_val, generation, ent->genome_hash);

            DEVICE_FATAL_IF(ent->cycles_elapsed == 0, "cycles_elapsed is 0 - no execution data");
            DEVICE_FATAL_IF(ent->total_branches == 0, "total_branches is 0 - no branch data");

            float ipc = (float)ent->inst_executed / (float)ent->cycles_elapsed;
            float tensor_util = (float)ent->tensor_core_cycles / (float)ent->cycles_elapsed;
            float branch_efficiency = (float)(ent->total_branches - ent->divergent_branches) / (float)ent->total_branches;
            float hw_eff_val = ipc * tensor_util * branch_efficiency;
            ent->hardware_efficiency.set_computed(hw_eff_val, generation, ent->genome_hash);

            DEVICE_FATAL_IF(generation == 0, "coherence requires previous generation");
            float prev_acc = organism->fitness_history[((generation - 1) % 2) * POOL_CAPACITY_MAX + eid];
            float coherence_val = ent->task_accuracy.value - prev_acc;
            ent->coherence.set_computed(coherence_val, generation, ent->genome_hash);

            DEVICE_VALIDATE_FINITE(ent->task_accuracy.value);
            DEVICE_VALIDATE_FINITE(ent->coherence.value);
            DEVICE_VALIDATE_FINITE(ent->hardware_efficiency.value);
            DEVICE_VALIDATE_FINITE(ent->generalization_gap.value);
            DEVICE_VALIDATE_PROBABILITY(ent->task_accuracy.value);
            DEVICE_VALIDATE_HW_COUNTER(ent->cycles_elapsed, 1ULL, 0xFFFFFFFFFFFFULL);
            DEVICE_VALIDATE_HW_COUNTER(ent->inst_executed, 1ULL, 0xFFFFFFFFFFFFULL);
            DEVICE_VALIDATE_HW_COUNTER(ent->tensor_core_cycles, 0ULL, ent->cycles_elapsed);
            device_validate_fitness_components(pool->fitness_values[eid], ent->coherence.value, ent->effective_rank.value, "pool_entry_fitness");

            organism->fitness_history[(generation % 2) * POOL_CAPACITY_MAX + eid] = ent->task_accuracy.value;
            organism->coherence_history[(generation % 2) * POOL_CAPACITY_MAX + eid] = ent->coherence.value;
        }
        __syncthreads();

        {
            float local_acc = 0.0f, local_gap = 0.0f, local_hw = 0.0f, local_fit = 0.0f;
            for (int compact = tid; compact < alive_ct; compact += blockDim.x) {
                int eid = pool->alive_indices[compact];
                local_acc += pool->entries[eid].task_accuracy.value;
                local_gap += pool->entries[eid].generalization_gap.value;
                local_hw += pool->entries[eid].hardware_efficiency.value;
                local_fit += pool->fitness_values[eid];
            }
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

            if (tid == 0) {
                organism->telemetry->population_metrics.total_accuracy = total_acc;
                organism->telemetry->population_metrics.total_generalization_gap = total_gap;
                organism->telemetry->population_metrics.total_hardware_efficiency = total_hw;
                organism->telemetry->population_metrics.total_fitness = total_fit;
            }
        }
        __syncthreads();

        if (generation > 0) {
            for (int compact = tid; compact < alive_ct; compact += blockDim.x) {
                int eid = pool->alive_indices[compact];
                DEVICE_FATAL_IF(!pool->alive_flags[eid], "hybrid_lifecycle: dead entry in alive_indices (baldwin loop)");

                float prev_task_accuracy = organism->fitness_history[((generation - 1) % 2) * POOL_CAPACITY_MAX + eid];
                float current_task_accuracy = pool->entries[eid].task_accuracy.value;
                float learning_success = current_task_accuracy - prev_task_accuracy;

                if (is_meaningful(learning_success, 1.0f)) {
                    PoolEntry* ent = &pool->entries[eid];
                    float baldwin_sensitivity = ent->baldwin_sensitivity;
                    float scale = learning_success * baldwin_sensitivity;
                    float* grads = ent->gradients;
                    float* eid_primary_genome = &component_workspace_genomes[eid * 2 * GENOME_SIZE];

                    for (int g = 0; g < GENOME_SIZE; g++) {
                        float val = grads[g] + scale * eid_primary_genome[g];
                        grads[g] = fmaxf(GENOME_VALUE_MIN, fminf(GENOME_VALUE_MAX, val));
                    }
                }
            }
        }
        __syncthreads();
    }
    if (tid == 0 && blockIdx.x == 0) printf("V:HYB_baldwin_done\n");

    if (training_mode->use_gradients) {

        {
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int total_cells = grid_size * grid_size;
            float* ca_concentration = entry->ca_state->ca_concentration;
            float* chemical_concentration = organism->chemical_field->concentration;

            for (int cell_idx = tid; cell_idx < total_cells; cell_idx += blockDim.x) {
                float val = ca_concentration[cell_idx * channels + 0];
                if (isfinite(val)) {
                    atomicAdd(&chemical_concentration[cell_idx], val);
                }
            }
        }
        __syncthreads();

        {
            int weight_count = arch.num_heads * arch.channels * arch.head_dim;
            half* weights_fp16 = ca_state->perception_weights;
            float* weights_fp32 = ca_state->fp32_workspace;

            for (int idx = tid; idx < weight_count; idx += blockDim.x) {
                weights_fp32[idx] = __half2float(weights_fp16[idx]);
            }
        }
        __syncthreads();

        if (tid == 0) {
            float* temp_latent = primary_parent_temp;
            diresa_encode(primary_genome, temp_latent, &organism->diresa_genome_weights[0]);
        }
        __syncthreads();
    }

    float* behavioral_workspace_genomes = organism->buffers->behavioral_workspace_genomes_buffer;


    if (entry_idx == 0 && generation % EMBEDDING_UPDATE_FREQ == 0) {
        if (tid == 0) {
            *organism->buffers->behavioral_reconstruction_error = 0.0f;
        }
        __syncthreads();

        int hw_dim = BEHAVIORAL_DIM_HW_MAX;
        int task_dim = BEHAVIORAL_DIM_TASK_MAX;
        int gen_dim = BEHAVIORAL_DIM_GEN_MAX;
        int embed_behavioral_dim = hw_dim + task_dim + gen_dim;

        {
            BehavioralState* agents = organism->behavioral_agents;
            float* embedding_weights = organism->buffers->behavioral_embedding_weights;
            float* reconstruction_error = organism->buffers->behavioral_reconstruction_error;
            int num_agents = POOL_CAPACITY_MAX;
            float* features_buffer = organism->buffers->behavioral_features_buffer;
            float* chemical_concentration = organism->chemical_field->concentration;
            int grid_size = arch.grid_size;

            float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
            float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
            float ctx_learning = organism->telemetry->diresa_evolution.behavioral_drift_rate;
            float ctx_performance = organism->telemetry->task_performance.accuracy;

            int embed_ctx_metabolic_slot = GenomeParamTable::embed_ctx_metabolic;
            int embed_ctx_stress_slot = GenomeParamTable::embed_ctx_stress;
            int embed_ctx_morphogen_slot = GenomeParamTable::embed_ctx_morphogen;
            float embed_ctx_metabolic = genome_slot_to_unit(primary_genome, embed_ctx_metabolic_slot);
            float embed_ctx_stress = genome_slot_to_unit(primary_genome, embed_ctx_stress_slot);
            float embed_ctx_morphogen = genome_slot_to_unit(primary_genome, embed_ctx_morphogen_slot);

            TrainingParams embed_training_params;
            float learning_rate = embed_training_params.get_behavioral_learning_rate(
                primary_genome, entry->gradients,
                embed_ctx_metabolic, embed_ctx_stress, embed_ctx_morphogen,
                ctx_complexity, ctx_niche, ctx_learning, ctx_performance
            );

            for (int agent_id = tid; agent_id < num_agents; agent_id += blockDim.x) {
                BehavioralState* agent = &agents[agent_id];
                float* features = &features_buffer[agent_id * embed_behavioral_dim];

                float context_metabolic = agent->sensitivity;
                float context_stress = sqrtf(agent->velocity[0] * agent->velocity[0] +
                                             agent->velocity[1] * agent->velocity[1]);
                int cx = (int)(agent->position[0] * grid_size) % grid_size;
                int cy = (int)(agent->position[1] * grid_size) % grid_size;
                float context_morphogen = chemical_concentration[cy * grid_size + cx];

                int base_freq_slot = GenomeParamTable::fourier_base_freq;
                float fourier_base_freq = genome_to_param(primary_genome, entry->gradients, base_freq_slot,
                    context_metabolic, context_stress, context_morphogen,
                    ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
                    FOURIER_BASE_FREQ_MIN, FOURIER_BASE_FREQ_MAX);

                int num_octaves_slot = GenomeParamTable::fourier_num_octaves;
                int fourier_num_octaves_raw = (int)genome_to_param(primary_genome, entry->gradients, num_octaves_slot,
                    context_metabolic, context_stress, context_morphogen,
                    ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
                    (float)FOURIER_NUM_OCTAVES_MIN, (float)FOURIER_NUM_OCTAVES_MAX);
                int fourier_num_octaves = min(fourier_num_octaves_raw, embed_behavioral_dim - 4);

                int spectrum_exp_slot = GenomeParamTable::fourier_spectrum_exponent;
                float fourier_spectrum_exponent = genome_to_param(primary_genome, entry->gradients, spectrum_exp_slot,
                    context_metabolic, context_stress, context_morphogen,
                    ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
                    FOURIER_SPECTRUM_EXPONENT_MIN, FOURIER_SPECTRUM_EXPONENT_MAX);

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

    if (entry_idx == 0 && organism->batch_ca_states_pool) {
        int grid_size_sync = arch.grid_size;
        int channels_sync = arch.channels;
        int spatial = grid_size_sync * grid_size_sync;
        int copy_size = spatial * channels_sync;
        int pool_cap = organism->pool->capacity;
        int batch_sz = training_mode->batch_size;

        for (int e = 0; e < pool_cap && e < batch_sz; e++) {
            PoolEntry* ent = &organism->pool->entries[e];
            if (organism->pool->alive_flags[e] && ent->ca_state && ent->ca_state->ca_concentration) {
                int src_offset = e * copy_size;
                for (int idx = tid; idx < copy_size; idx += blockDim.x) {
                    ent->ca_state->ca_concentration[idx] = organism->batch_ca_states_pool[src_offset + idx];
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0 && blockIdx.x == 0) printf("V:HYB_sync_done\n");

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

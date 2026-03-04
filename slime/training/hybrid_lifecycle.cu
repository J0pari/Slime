
#ifndef HYBRID_LIFECYCLE_CU
#define HYBRID_LIFECYCLE_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include "../debug/provenance.cuh"
#include "../training/training_types.cu"
#include "../data/dataset_loader.cu"
#include "../core/chemotaxis.cu"
#include "../core/ca_state.cuh"
#include "../training/autodiff_integration.cu"
#include "../training/gradient_fitness.cu"
#include "../training/classification.cu"
#include "../training/optimizer.cu"
#include "../utils/genome_params.cuh"
#include "../diagnostics/telemetry_probes.cu"
#include <cuda_runtime.h>

struct Organism;
struct ComponentPool;
struct GPUElite;
struct VoronoiCell;
struct ChemicalField;
struct BehavioralState;
struct TemporalTube;

__device__ int g_block_counter = 0;
__device__ int g_grid_barrier_count = 0;
__device__ int g_grid_barrier_sense = 0;

// Aggregate printf counters - atomicAdd to count, grid_barrier, then one thread prints total
__device__ int g_blocks_entered = 0;
__device__ int g_blocks_grad = 0;
__device__ int g_blocks_ca_fwd = 0;
__device__ int g_blocks_flow = 0;
__device__ int g_blocks_bwd = 0;
__device__ int g_blocks_complete = 0;
// V: verbose progress counters for backward pass
__device__ int g_v_flow_done_count = 0;
__device__ int g_v_bwd_enter_count = 0;
__device__ int g_v_bwd_fatal_checks_count = 0;
__device__ int g_v_bwd_chunk_count = 0;
__device__ int g_v_bwd_value_grad_count = 0;
__device__ int g_v_bwd_inter_grad_count = 0;
__device__ int g_v_bwd_perc_grad_count = 0;
__device__ int g_v_bwd_done_count = 0;
__device__ int g_v_bwd_zero_dw_count = 0;
__device__ int g_v_bwd_setup_done_count = 0;
__device__ int g_v_bwd_chunks_done_count = 0;
__device__ int g_v_bwd_inter_grad_copy_count = 0;
__device__ int g_v_bwd_perc_grad_copy_count = 0;
__device__ int g_v_bwd_grad_conc_count = 0;
__device__ int g_v_bwd_diffusion_launch_count = 0;
__device__ int g_v_bwd_chunk0_count = 0;
__device__ int g_v_bwd_i_done_count = 0;
__device__ int g_v_bwd_v_done_count = 0;
__device__ int g_v_bwd_chunk2_enter_count = 0;
__device__ int g_v_bwd_di_write_count = 0;
__device__ int g_v_bwd_perc_load_count = 0;
__device__ int g_v_bwd_dp_write_count = 0;
__device__ int g_v_bwd_im2col_count = 0;
__device__ int g_v_bwd_conv_fp16_count = 0;

// Backward pass broadcast bounds - blocks with work atomicMax, all blocks read after sync
__device__ int g_bwd_max_num_chunks = 0;
__device__ int g_bwd_max_num_cells = 0;
__device__ int g_bwd_max_total_samples = 0;
__device__ int g_v_bwd_input_grad_count = 0;
__device__ int g_v_bwd_scatter_count = 0;
__device__ int g_v_bwd_diff_device_count = 0;
__device__ int g_v_post_bwd_barrier_count = 0;

__device__ __forceinline__ void grid_barrier(int num_blocks) {
    cg::this_grid().sync();
}

// Helper: after grid_barrier, one thread prints aggregate and resets counter
__device__ __forceinline__ void print_aggregate_and_reset(const char* label, int* counter, int num_blocks) {
    grid_barrier(num_blocks);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("V:%s blocks=%d\n", label, *counter);
        *counter = 0;
    }
}

__device__ __forceinline__ BackwardWorkspaceLayout compute_backward_ws_layout(PoolEntry* entry) {
    BackwardWorkspaceLayout layout;
    int cells = entry->grid_size * entry->grid_size;
    int num_heads = entry->num_heads;
    int hidden_dim = entry->hidden_dim;
    int channels = entry->channels;
    int head_dim = entry->head_dim;

    size_t fp16_a_elements = (size_t)num_heads * BACKWARD_CHUNK_SAMPLES * head_dim;
    size_t fp16_b_elements = (size_t)num_heads * BACKWARD_CHUNK_SAMPLES * head_dim;
    // dW stores: interaction(head_dim*head_dim) + perception(channels*head_dim)
    size_t dW_elements = (size_t)num_heads * (head_dim * head_dim + channels * head_dim);
    size_t dI_elements = (size_t)num_heads * BACKWARD_CHUNK_SAMPLES * head_dim;
    // W_T stores: interaction(head_dim*head_dim)
    size_t W_T_elements = (size_t)num_heads * head_dim * head_dim;
    size_t im2col_elements = (size_t)num_heads * BACKWARD_CHUNK_SAMPLES * channels;
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
    return batch_size * entry->num_heads * cells * entry->channels;
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

__device__ void zero_scalar_device(Organism* organism, float* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *ptr = 0.0f;
    }
}

__device__ void zero_buffer_device(Organism* organism, float* buffer, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        buffer[idx] = 0.0f;
    }
}

// Load initial batch samples during initialization (before first training iteration)
// This must be called before init_batch_prev_concentration_device so channels 0-2 have real data
__device__ void load_initial_batch_samples_device(Organism* organism) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    HybridTrainingMode* training_mode = organism->training_mode;
    if (!training_mode) return;

    Dataset* dataset = organism->current_dataset;
    if (!dataset) return;

    int batch_size = training_mode->batch_size;
    int grid_size = organism->current_arch.grid_size;
    int batch_stride = grid_size * grid_size;

    int dataset_size = dataset->num_samples;
    int sample_rows = dataset->descriptor->sample_rows;
    int sample_cols = dataset->descriptor->sample_cols;
    int sample_channels = dataset->descriptor->channels;
    FeatureEncoding encoding = dataset->descriptor->encoding;
    unsigned char* all_samples = dataset->samples;
    unsigned char* all_labels = dataset->labels;
    int sample_size = sample_rows * sample_cols * sample_channels;

    float* batch_samples_out = training_mode->batch_samples;
    int* batch_labels_out = training_mode->batch_labels;

    // Load first batch (generation 0, offset 0)
    int offset = 0;

    // Load labels
    if (global_tid == 0) {
        for (int idx = 0; idx < batch_size; idx++) {
            int src_idx = (offset + idx) % dataset_size;
            batch_labels_out[idx] = all_labels[src_idx];
        }
    }

    // Load and interpolate samples based on encoding
    int total_cells = batch_size * batch_stride;

    if (encoding == ENCODING_TEMPORAL_1D) {
        // Timeseries: resample 1D temporal data to 2D grid
        int timesteps = sample_rows;
        int features = sample_cols;

        for (int work_idx = global_tid; work_idx < total_cells; work_idx += total_threads) {
            int idx = work_idx / batch_stride;
            int cell_idx = work_idx % batch_stride;
            int src_idx = (offset + idx) % dataset_size;
            int cell_y = cell_idx / grid_size;
            int cell_x = cell_idx % grid_size;

            float src_t = cell_y * (float)timesteps / grid_size;
            float src_f = cell_x * (float)features / grid_size;
            int t0 = (int)src_t;
            int f0 = (int)src_f;
            int t1 = min(t0 + 1, timesteps - 1);
            int f1 = min(f0 + 1, features - 1);
            float ft = src_t - t0;
            float ff = src_f - f0;

            int out_base = idx * batch_stride * 3;
            int sample_base = src_idx * sample_size;

            float v00 = all_samples[sample_base + t0 * features + f0] / 255.0f;
            float v01 = all_samples[sample_base + t0 * features + f1] / 255.0f;
            float v10 = all_samples[sample_base + t1 * features + f0] / 255.0f;
            float v11 = all_samples[sample_base + t1 * features + f1] / 255.0f;
            float val = Interpolation::bilinear(v00, v01, v10, v11, ff, ft);

            batch_samples_out[out_base + 0 * batch_stride + cell_idx] = val;
            batch_samples_out[out_base + 1 * batch_stride + cell_idx] = ft;
            batch_samples_out[out_base + 2 * batch_stride + cell_idx] = ff;
        }
    } else {
        // ENCODING_SPATIAL_2D or ENCODING_SPECTRAL_AUDIO: 2D bilinear interpolation
        for (int work_idx = global_tid; work_idx < total_cells; work_idx += total_threads) {
            int idx = work_idx / batch_stride;
            int pixel_idx = work_idx % batch_stride;
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
            int out_idx_base = idx * batch_stride * 3;

            if (sample_channels >= 3) {
                for (int c = 0; c < 3; c++) {
                    int channel_offset = c * sample_rows * sample_cols;
                    int sample_base = src_idx * sample_size + channel_offset;
                    float tl = all_samples[sample_base + y0 * sample_cols + x0] / 255.0f;
                    float tr = all_samples[sample_base + y0 * sample_cols + x1] / 255.0f;
                    float bl = all_samples[sample_base + y1 * sample_cols + x0] / 255.0f;
                    float br = all_samples[sample_base + y1 * sample_cols + x1] / 255.0f;
                    batch_samples_out[out_idx_base + c * batch_stride + pixel_idx] = Interpolation::bilinear(tl, tr, bl, br, fx, fy);
                }
            } else {
                int sample_base = src_idx * sample_size;
                float tl = all_samples[sample_base + y0 * sample_cols + x0] / 255.0f;
                float tr = all_samples[sample_base + y0 * sample_cols + x1] / 255.0f;
                float bl = all_samples[sample_base + y1 * sample_cols + x0] / 255.0f;
                float br = all_samples[sample_base + y1 * sample_cols + x1] / 255.0f;
                float3 vg = Interpolation::bilinear_with_grad(tl, tr, bl, br, fx, fy);
                batch_samples_out[out_idx_base + 0 * batch_stride + pixel_idx] = vg.x;
                batch_samples_out[out_idx_base + 1 * batch_stride + pixel_idx] = vg.y;
                batch_samples_out[out_idx_base + 2 * batch_stride + pixel_idx] = vg.z;
            }
        }
    }
}

__device__ void load_batch_device(Organism* organism) {
    HybridTrainingMode* training_mode = organism->training_mode;
    int generation = organism->generation;
    int wave_start = organism->current_wave_start;
    int tid = threadIdx.x;
    int wave_position = blockIdx.x;
    int compact_idx = wave_start + blockIdx.x;

    // Shared state accessible by all blocks
    DEVICE_FATAL_IF(organism == nullptr, "load_batch_device: organism is null");
    DEVICE_FATAL_IF(training_mode == nullptr, "load_batch_device: training_mode is null");
    DEVICE_FATAL_IF(training_mode->batch_samples == nullptr, "load_batch_device: batch_samples is null");
    DEVICE_FATAL_IF(training_mode->batch_labels == nullptr, "load_batch_device: batch_labels is null");
    DEVICE_FATAL_IF(generation < 0, "load_batch_kernel: generation negative");

    ComponentPool* pool = organism->pool;
    bool has_work = compact_idx < pool->alive_indices_count;
    int batch_size = training_mode->batch_size;
    DEVICE_FATAL_IF(batch_size <= 0 || batch_size > BATCH_SIZE, "load_batch_kernel: batch_size invalid");

    if (tid == 0 && blockIdx.x == 0) {
        training_mode->is_train_batch = ((generation % 2) == 0);
    }

    // Per-entry state - only assigned and accessed when has_work
    int entry_idx;
    PoolEntry* entry;
    int grid_size;
    int channels_out;
    if (has_work) {
        entry_idx = pool->alive_indices[compact_idx];
        entry = &pool->entries[entry_idx];
        grid_size = entry->grid_size;
        channels_out = entry->channels;
    }

    __shared__ WaveBufferOffsets s_wave_offsets;
    if (has_work && tid == 0) {
        s_wave_offsets = compute_wave_offsets(pool, wave_start, wave_position, batch_size);
    }

    cg::this_grid().sync();  // SYNC 1: after wave offset computation

    Dataset* dataset = organism->current_dataset;
    int batch_stride = has_work ? grid_size * grid_size : 0;

    if (has_work && tid == 0 && blockIdx.x == 0) {
        load_batch_labels_device(dataset, training_mode->batch_labels, batch_size, generation);
    }

    cg::this_grid().sync();  // SYNC 2: after label loading

    if (has_work) {
        load_batch_samples_device(dataset, training_mode->batch_samples, batch_size, grid_size, generation, tid, blockDim.x);
    }

    cg::this_grid().sync();  // SYNC 3: after sample loading

    float* ca_out;
    float* prev_concentration;
    float* batch_samples = training_mode->batch_samples;

    if (has_work) {
        ca_out = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
        prev_concentration = organism->buffers->batch_prev_concentration + s_wave_offsets.ca_states_offset;
    }

    ChemicalField* chem = organism->chemical_field;

    // Field read positioning: gen >= 1 uses prev-gen DIRESA-derived coords, gen 0 uses agent_pos + stride
    bool use_diresa_coords = (generation >= 1) && has_work;
    float* prev_field_coords = nullptr;
    int agent_gx = 0, agent_gy = 0;
    int sample_stride_field = has_work ? (grid_size / batch_size) : 0;
    if (has_work) {
        if (use_diresa_coords) {
            prev_field_coords = organism->prev_sample_field_coords + entry_idx * batch_size * 2;
        } else {
            BehavioralState* agent = &organism->behavioral_agents[entry_idx];
            agent_gx = min(max((int)(agent->position[0] * grid_size), 0), grid_size - 1);
            agent_gy = min(max((int)(agent->position[1] * grid_size), 0), grid_size - 1);
        }
    }

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        if (has_work) {
            int center_x, center_y;
            if (use_diresa_coords) {
                center_x = (int)prev_field_coords[batch_idx * 2];
                center_y = (int)prev_field_coords[batch_idx * 2 + 1];
            } else {
                center_x = agent_gx + batch_idx * sample_stride_field;
                center_y = agent_gy + batch_idx * sample_stride_field;
            }
            int num_heads_local = entry->num_heads;
            for (int head = 0; head < num_heads_local; head++) {
                for (int spatial_idx = tid; spatial_idx < batch_stride; spatial_idx += blockDim.x) {
                    int cx = spatial_idx % grid_size;
                    int cy = spatial_idx / grid_size;
                    int field_x = (cx + center_x) % grid_size;
                    int field_y = (cy + center_y) % grid_size;
                    int field_spatial_idx = field_y * grid_size + field_x;

                    int base_idx = (batch_idx * num_heads_local + head) * batch_stride * channels_out + spatial_idx * channels_out;
                    inject_ca_cell_device(
                        ca_out, base_idx, channels_out, generation,
                        chem->concentration, chem->channels, batch_stride, spatial_idx,
                        field_spatial_idx,
                        batch_samples, batch_idx,
                        prev_concentration, base_idx
                    );
                }
            }
        }
        cg::this_grid().sync();  // SYNC 4+: per batch iteration
    }
}

__device__ void hybrid_organism_lifecycle_device(Organism* organism) {
    HybridTrainingMode* training_mode = organism->training_mode;
    CAParameterMap* param_map = organism->param_map;
    int generation = organism->generation;
    float* workspace_genomes = organism->workspace_genomes;
    AuditBuffer* audit = organism->audit_buffer;
    int wave_start = organism->current_wave_start;

    extern __shared__ float sdata[];
    int compact_idx = wave_start + blockIdx.x;
    int wave_position = blockIdx.x;
    int tid = threadIdx.x;
    if (blockIdx.x == 0 && tid == 0) {
        g_block_counter = 0;
        g_blocks_entered = 0;
        g_blocks_grad = 0;
        g_blocks_ca_fwd = 0;
        g_blocks_flow = 0;
        g_blocks_bwd = 0;
        g_blocks_complete = 0;
    }
    ComponentPool* pool = organism->pool;

    bool has_work = compact_idx < pool->alive_indices_count;

    int entry_idx;
    PoolEntry* entry;
    if (has_work) {
        entry_idx = pool->alive_indices[compact_idx];
        entry = &pool->entries[entry_idx];
    }

    if (has_work && tid == 0) {
        organism->lifecycle_entry_idx = entry_idx;
        organism->lifecycle_workspace_genomes = workspace_genomes;
        organism->lifecycle_wave_start = wave_start;
    }
    cg::this_grid().sync();

    __shared__ WaveBufferOffsets s_wave_offsets;
    if (has_work && tid == 0) {
        s_wave_offsets = compute_wave_offsets(pool, wave_start, wave_position, training_mode->batch_size);
    }
    cg::this_grid().sync();

    __shared__ bool s_entry_alive;
    if (has_work && tid == 0) {
        s_entry_alive = pool->alive_flags[entry_idx];
    }
    cg::this_grid().sync();
    if (has_work) {
        DEVICE_FATAL_IF(!s_entry_alive, "hybrid_organism_lifecycle_kernel: dead entry in alive_indices");
        if (tid == 0) atomicAdd(&g_block_counter, 1);
    }

    MultiHeadCAState* ca_state;
    int local_cells;
    float thread_sum = 0.0f;

    if (has_work) {
        ca_state = entry->ca_state;
        local_cells = entry->channels * entry->grid_size * entry->grid_size;
        for (int i = tid; i < local_cells; i += blockDim.x) {
            thread_sum += ca_state->ca_concentration[i];
        }
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float local_ca_mean;
    if (has_work) {
        local_ca_mean = sdata[0] / (float)local_cells;
        if (tid == 0) sdata[0] = local_ca_mean;
    }
    __syncthreads();
    if (has_work) {
        local_ca_mean = sdata[0];
    }

    __shared__ int s_error_flag;
    __shared__ bool s_use_gradients;
    if (tid == 0) {
        s_error_flag = 0;
        s_use_gradients = training_mode->use_gradients;
    }
    cg::this_grid().sync();

    __shared__ float* s_primary_genome;
    __shared__ int s_num_classes;
    __shared__ int s_behavioral_dim;
    __shared__ int s_num_features;
    __shared__ Architecture s_arch;
    __shared__ float s_task_accuracy;
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
    Architecture arch;
    dim3 component_grid;
    dim3 component_block;
    dim3 ca_grid;
    dim3 ca_block;
    dim3 field_grid;
    dim3 field_block;
    cudaError_t err;

    if (has_work && tid == 0) {
        primary_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];
        primary_parent_temp = &workspace_genomes[entry_idx * GENOME_SIZE * 2 + GENOME_SIZE];

        reconstruct_genome_from_archive(entry->parent_hash, (GPUElite*)organism->archive, organism->archive_size,
            entry->delta_indices, entry->delta_values, entry->num_deltas,
            entry->max_deltas, primary_genome, GENOME_SIZE, primary_parent_temp, organism->diresa_genome_weights);

        num_classes = organism->current_dataset->descriptor->num_classes;

        BehavioralDimensions dims;
        dims.derive_from_genome();
        behavioral_dim = dims.total();

        arch.num_heads = entry->num_heads;
        arch.channels = entry->channels;
        arch.hidden_dim = entry->hidden_dim;
        arch.head_dim = entry->head_dim;
        arch.grid_size = entry->grid_size;
        s_task_accuracy = fminf(fmaxf(entry->task_accuracy.value, 0.0f), 1.0f);

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
    cg::this_grid().sync();

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

    if (has_work && tid == 0 && organism->current_activation_grid_size == 0) {
        organism->current_activation_grid_size = arch.grid_size;
    }

    if (has_work && tid == 0 && arch.grid_size != organism->current_activation_grid_size) {
        organism->current_activation_grid_size = arch.grid_size;
    }
    if (has_work && tid == 0 && blockIdx.x == 0) printf("V:BLK_ARCH grid=%d heads=%d hdim=%d ch=%d blocks=%d\n", arch.grid_size, arch.num_heads, arch.head_dim, arch.channels, gridDim.x);

    float* ca_output_grad = organism->buffers->ca_output_grad_buffer + s_wave_offsets.ca_output_offset;
    float* wave_prev_conc = organism->buffers->batch_prev_concentration + s_wave_offsets.ca_states_offset;

    if (s_use_gradients) {
        cg::this_grid().sync();
        if (has_work && tid == 0 && blockIdx.x == 0) printf("V:GRAD_ENTER\n");

        if (has_work) {
            reset_tape_device(&ca_state->tape, tid);
        }
        cg::this_grid().sync();

        if (has_work) {
            float* ca_input = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
            float* ca_output = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;

            multi_head_ca_tensor_device(entry, training_mode->batch_size, ca_input, ca_output, s_task_accuracy);
        }

        cg::this_grid().sync();

        if (has_work && tid == 0) {
            atomicAdd(&g_blocks_ca_fwd, 1);
        }

        cg::this_grid().sync();
        if (has_work) {
            DEVICE_FATAL_IF(s_error_flag, "hybrid_lifecycle: error flag set after CA forward");
        }

        // Persist per-head state: ca_output → prev_concentration for next generation
        if (has_work) {
            int state_size = training_mode->batch_size * arch.num_heads *
                             arch.grid_size * arch.grid_size * arch.channels;
            float* wave_ca_output = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;

            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                wave_prev_conc[idx] = wave_ca_output[idx];
            }
        }
        cg::this_grid().sync();

        if (has_work && tid == 0) {
            atomicAdd(&g_v_flow_done_count, 1);
        }

        grid_barrier(gridDim.x);

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {
            // Pooling - work guarded
            if (has_work) {
                int batch_size = training_mode->batch_size;
                int grid_size = arch.grid_size;
                int channels = arch.channels;
                int num_heads_local = arch.num_heads;
                int spatial_size = grid_size * grid_size;

                float* ca_out = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
                float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
                float* pooling_weights = training_mode->classifier[entry_idx].pooling_weights;

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
            }
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {
            if (has_work && tid == 0) {
                BehavioralDimensions dims;
                dims.derive_from_genome();
                int task_dim = dims.task_dim;

                DEVICE_FATAL_IF(entry->diresa_task_weights->input_dim != num_features,
                    "hybrid_lifecycle: diresa_task_weights->input_dim mismatch with num_features (entry %d)", entry_idx);

                // Per-sample DIRESA encode → fractal fold → 2D field coords
                int batch_size_local = training_mode->batch_size;
                float* features_base = organism->gradient_features_pool + entry_idx * batch_size_local * num_features;
                float* field_coords = organism->sample_field_coords + entry_idx * batch_size_local * 2;
                float sample_latent[BEHAVIORAL_DIM_TASK];
                for (int s = 0; s < batch_size_local; s++) {
                    diresa_encode(features_base + s * num_features, sample_latent, entry->diresa_task_weights);
                    fractal_fold_2d(sample_latent, task_dim, arch.grid_size, &field_coords[s * 2], &field_coords[s * 2 + 1]);
                }
            }
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {
            // Classification - work guarded
            if (has_work) {
                int batch_size = training_mode->batch_size;
                float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
                float* logits = organism->gradient_logits_pool + entry_idx * batch_size * num_classes;
                float* fc_weights = training_mode->classifier[entry_idx].fc_weights;
                float* fc_bias = training_mode->classifier[entry_idx].fc_bias;

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
            }
        }
        cg::this_grid().sync();

        // Per-block loss pointer - uninitialized, only valid when has_work
        float* loss_out;
        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work) {
            loss_out = &organism->gradient_loss_pool[entry_idx];
            if (tid == 0) {
                *loss_out = 0.0f;
            }
        }
        cg::this_grid().sync();

        // Softmax/loss - work guarded
        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work) {
            int batch_size = training_mode->batch_size;
            float* logits = organism->gradient_logits_pool + entry_idx * batch_size * num_classes;
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
        }
        cg::this_grid().sync();
        if (tid == 0 && blockIdx.x == 0) printf("V:softmax_done\n");

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr && has_work && tid == 0) {
                int batch_size = training_mode->batch_size;
                float* logits = organism->gradient_logits_pool + entry_idx * batch_size * num_classes;
                int* batch_labels = training_mode->batch_labels;

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
                                int cells = grid_size * grid_size;
                                int chem_ch = organism->chemical_field->channels;
                                for (int cc = 0; cc < chem_ch; cc++) {
                                    organism->chemical_field->concentration[cc * cells + pos] = sample_accuracy;
                                }
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
                    measured_value_set_computed(&entry->train_accuracy, smoothed, current_gen, entry->genome_hash);
                } else {
                    measured_value_set_computed(&entry->test_accuracy, accuracy, current_gen, entry->genome_hash);
                }
                measured_value_set_computed(&entry->task_accuracy, accuracy, current_gen, entry->genome_hash);
                measured_value_set_computed(&entry->avg_confidence, avg_confidence, current_gen, entry->genome_hash);
                for (int c = 0; c < NUM_CLASSES_MAX; c++) {
                    entry->per_class_correct[c] = local_per_class_correct[c];
                    entry->per_class_total[c] = local_per_class_total[c];
                }
            }
        }
        cg::this_grid().sync();

        // Audit after forward pass - logits and accuracy are computed, capture state before gradients
        if (tid == 0 && blockIdx.x == 0 && audit != nullptr && training_mode->batch_samples != nullptr) {
            run_telemetry_probes(organism, generation);
            float* logits = organism->gradient_logits_pool;
            int* labels = training_mode->batch_labels;
            float* batch_samples = training_mode->batch_samples;
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
                batch_samples,
                batch_size,
                num_classes,
                ca_concentration,
                grid_size,
                train_acc,
                test_acc,
                training_mode->is_train_batch,
                organism->telemetry,
                organism->pool,
                organism->chemical_field,
                ca_state,
                organism->hardware_geom,
                organism->archive_size
            );
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {

            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int spatial_size = grid_size * grid_size;

            float* ca_out = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
            float* logit_grads = organism->gradient_logit_grads_pool + entry_idx * batch_size * num_classes;
            float* features_grad = organism->features_grad + entry_idx * batch_size * num_features;
            float* fc_weights = training_mode->classifier[entry_idx].fc_weights;
            constexpr int FC_WEIGHTS_ENTRY_STRIDE = NUM_CLASSES_MAX * NUM_HEADS * CHANNELS;
            constexpr int FC_BIAS_ENTRY_STRIDE = NUM_CLASSES_MAX;
            constexpr int POOLING_ENTRY_STRIDE = NUM_HEADS * CHANNELS;
            float* fc_weights_grad = organism->fc_weights_grad + entry_idx * FC_WEIGHTS_ENTRY_STRIDE;
            float* fc_bias_grad = organism->fc_bias_grad + entry_idx * FC_BIAS_ENTRY_STRIDE;
            float* pooling_weights = training_mode->classifier[entry_idx].pooling_weights;
            float* pooling_weights_grad = organism->pooling_weights_grad + entry_idx * POOLING_ENTRY_STRIDE;

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
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int spatial_size = grid_size * grid_size;
            float* ca_out = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            float* features = organism->gradient_features_pool + entry_idx * batch_size * num_features;
            float* logit_grads = organism->gradient_logit_grads_pool + entry_idx * batch_size * num_classes;
            float* features_grad = organism->features_grad + entry_idx * batch_size * num_features;
            float* fc_weights = training_mode->classifier[entry_idx].fc_weights;
            constexpr int FC_WEIGHTS_ENTRY_STRIDE = NUM_CLASSES_MAX * NUM_HEADS * CHANNELS;
            constexpr int FC_BIAS_ENTRY_STRIDE = NUM_CLASSES_MAX;
            float* fc_weights_grad = organism->fc_weights_grad + entry_idx * FC_WEIGHTS_ENTRY_STRIDE;
            float* fc_bias_grad = organism->fc_bias_grad + entry_idx * FC_BIAS_ENTRY_STRIDE;
            float* pooling_weights = training_mode->classifier[entry_idx].pooling_weights;

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
        }
        cg::this_grid().sync();

        if (training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int spatial_size = grid_size * grid_size;
            float* ca_out = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            float* features_grad = organism->features_grad + entry_idx * batch_size * num_features;
            float* pooling_weights = training_mode->classifier[entry_idx].pooling_weights;
            constexpr int POOLING_ENTRY_STRIDE = NUM_HEADS * CHANNELS;
            float* pooling_weights_grad = organism->pooling_weights_grad + entry_idx * POOLING_ENTRY_STRIDE;

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
        }
        cg::this_grid().sync();
        if (tid == 0 && blockIdx.x == 0) printf("V:pre_bwd\n");

        // Backward pass - work guarded, syncs outside
        // Blocks without work skip work but hit all grid syncs
        {
            // Reset broadcast bounds
            if (tid == 0 && blockIdx.x == 0) {
                g_bwd_max_num_chunks = 0;
                g_bwd_max_num_cells = 0;
                g_bwd_max_total_samples = 0;
            }
            cg::this_grid().sync();

            // Blocks with work broadcast their bounds via atomicMax
            if (has_work && tid == 0) {
                int num_cells = arch.grid_size * arch.grid_size;
                int total_samples = training_mode->batch_size * num_cells;
                int num_chunks = (total_samples + BACKWARD_CHUNK_SAMPLES - 1) / BACKWARD_CHUNK_SAMPLES;
                atomicMax(&g_bwd_max_num_chunks, num_chunks);
                atomicMax(&g_bwd_max_num_cells, num_cells);
                atomicMax(&g_bwd_max_total_samples, total_samples);
            }
            cg::this_grid().sync();

            // All blocks read broadcast bounds
            int bwd_num_chunks = g_bwd_max_num_chunks;
            int bwd_num_cells = g_bwd_max_num_cells;
            int bwd_total_samples = g_bwd_max_total_samples;

            // Per-block variables - uninitialized, only valid when has_work
            // Blocks without work never read these - garbage is fine
            TraceBuffer* trace_buffer;
            float* dL_dperception;
            float* dL_dinteraction;
            char* backward_ws_base;
            BackwardWorkspaceLayout ws_layout;
            half* ws_fp16_a;
            half* ws_fp16_b;
            float* ws_dW;
            float* ws_dI;
            half* ws_W_T;
            float* ws_im2col;
            float* ws_dpregelu;
            float* perception_saved;
            float* interaction_saved;
            float* pre_gelu_saved;
            int num_cells;
            int total_samples;
            int I_head_stride;
            int I_batch_stride;
            int V_head_stride;
            int V_batch_stride;
            int ws_dW_interaction_stride;
            int ws_W_T_interaction_stride;
            int chunk_ws_a_stride;
            int chunk_ws_b_stride;

            ExecutionTrace* bwd_trace_slot = nullptr;
            unsigned long long bwd_cycle_start = 0;
            if (has_work) {
                trace_buffer = &ca_state->trace;
                {
                    int trace_idx = -1;
                    if (tid == 0 && trace_buffer->traces != nullptr &&
                        trace_buffer->current_idx < trace_buffer->capacity) {
                        trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
                        printf("TRACE_BWD_ALLOC b%d eid=%d idx=%d cidx_now=%d cidx_addr=%p\n",
                            blockIdx.x, entry_idx, trace_idx, trace_buffer->current_idx, &trace_buffer->current_idx);
                    } else if (tid == 0) {
                        printf("TRACE_BWD_SKIP b%d traces=%p cidx=%d cap=%d\n",
                            blockIdx.x, trace_buffer->traces, trace_buffer->current_idx, trace_buffer->capacity);
                    }
                    if (tid < WARP_SIZE) {
                        trace_idx = __shfl_sync(0xFFFFFFFF, trace_idx, 0);
                        if (trace_idx >= 0 && trace_idx < trace_buffer->capacity) {
                            bwd_trace_slot = &trace_buffer->traces[trace_idx];
                            record_warp_metrics(bwd_trace_slot, blockIdx.x);
                            record_memory_access(bwd_trace_slot, (void*)&ca_state->perception_weights[tid], true);
                            record_shared_memory_access(bwd_trace_slot, false, false);
                        }
                    }
                    bwd_cycle_start = clock64();
                }

                num_cells = arch.grid_size * arch.grid_size;
                total_samples = training_mode->batch_size * num_cells;

                DEVICE_FATAL_IF(organism->buffers == nullptr, "BWD: organism->buffers is null");
                DEVICE_FATAL_IF(organism->buffers->dL_dperception_buffer == nullptr, "BWD: dL_dperception_buffer is null");
                DEVICE_FATAL_IF(organism->buffers->dL_dinteraction_buffer == nullptr, "BWD: dL_dinteraction_buffer is null");
                DEVICE_FATAL_IF(s_wave_offsets.activations_offset < 0, "BWD: activations_offset negative");

                dL_dperception = organism->buffers->dL_dperception_buffer + s_wave_offsets.activations_offset;
                dL_dinteraction = organism->buffers->dL_dinteraction_buffer + s_wave_offsets.activations_offset;
                if (tid == 0) atomicAdd(&g_v_bwd_enter_count, 1);

                if (tid == 0) atomicAdd(&g_v_bwd_fatal_checks_count, 1);
                DEVICE_FATAL_IF(organism->buffers->backward_workspace == nullptr, "BWD: backward_workspace is null");
                DEVICE_FATAL_IF(s_wave_offsets.backward_ws_offset < 0, "BWD: backward_ws_offset negative");
                backward_ws_base = organism->buffers->backward_workspace + s_wave_offsets.backward_ws_offset;
                ws_layout = compute_backward_ws_layout(entry);
                DEVICE_FATAL_IF(ws_layout.fp16_a_offset < 0, "BWD: ws_layout.fp16_a_offset negative");
                DEVICE_FATAL_IF(ws_layout.fp16_b_offset < 0, "BWD: ws_layout.fp16_b_offset negative");
                DEVICE_FATAL_IF(ws_layout.dW_offset < 0, "BWD: ws_layout.dW_offset negative");
                DEVICE_FATAL_IF(ws_layout.dI_offset < 0, "BWD: ws_layout.dI_offset negative");
                DEVICE_FATAL_IF(ws_layout.W_T_offset < 0, "BWD: ws_layout.W_T_offset negative");
                DEVICE_FATAL_IF(ws_layout.im2col_offset < 0, "BWD: ws_layout.im2col_offset negative");
                DEVICE_FATAL_IF(ws_layout.dpregelu_offset < 0, "BWD: ws_layout.dpregelu_offset negative");

                ws_fp16_a = (half*)(backward_ws_base + ws_layout.fp16_a_offset);
                ws_fp16_b = (half*)(backward_ws_base + ws_layout.fp16_b_offset);
                ws_dW = (float*)(backward_ws_base + ws_layout.dW_offset);
                ws_dI = (float*)(backward_ws_base + ws_layout.dI_offset);
                ws_W_T = (half*)(backward_ws_base + ws_layout.W_T_offset);
                ws_im2col = (float*)(backward_ws_base + ws_layout.im2col_offset);
                ws_dpregelu = (float*)(backward_ws_base + ws_layout.dpregelu_offset);

                DEVICE_FATAL_IF(ca_state == nullptr, "BWD: ca_state is null");
                DEVICE_FATAL_IF(ca_state->perception_saved == nullptr, "BWD: perception_saved is null");
                DEVICE_FATAL_IF(ca_state->interaction_saved == nullptr, "BWD: interaction_saved is null");
                DEVICE_FATAL_IF(ca_state->pre_gelu_saved == nullptr, "BWD: pre_gelu_saved is null");
                DEVICE_FATAL_IF(ca_state->flow_projection_weights == nullptr, "BWD: flow_projection_weights is null");
                DEVICE_FATAL_IF(ca_state->interaction_weights == nullptr, "BWD: interaction_weights is null");
                DEVICE_FATAL_IF(ca_state->perception_weights == nullptr, "BWD: perception_weights is null");
                DEVICE_FATAL_IF(ca_state->tape.grad_buffer == nullptr, "BWD: tape.grad_buffer is null");
                DEVICE_FATAL_IF(param_map == nullptr, "BWD: param_map is null");
                DEVICE_FATAL_IF(param_map->interaction_start == nullptr, "BWD: param_map->interaction_start is null");
                DEVICE_FATAL_IF(param_map->perception_start == nullptr, "BWD: param_map->perception_start is null");
                DEVICE_FATAL_IF(organism->batch_ca_states_pool == nullptr, "BWD: batch_ca_states_pool is null");
                perception_saved = ca_state->perception_saved;
                interaction_saved = ca_state->interaction_saved;
                pre_gelu_saved = ca_state->pre_gelu_saved;

                I_head_stride = num_cells * arch.head_dim;
                I_batch_stride = arch.num_heads * I_head_stride;
                V_head_stride = num_cells * arch.channels;
                V_batch_stride = arch.num_heads * V_head_stride;
                ws_dW_interaction_stride = arch.head_dim * arch.head_dim;
                ws_W_T_interaction_stride = arch.head_dim * arch.head_dim;

                DEVICE_FATAL_IF(arch.num_heads <= 0, "BWD: arch.num_heads <= 0");
                DEVICE_FATAL_IF(arch.head_dim <= 0, "BWD: arch.head_dim <= 0");
                DEVICE_FATAL_IF(arch.channels <= 0, "BWD: arch.channels <= 0");
                DEVICE_FATAL_IF(total_samples <= 0, "BWD: total_samples <= 0");
                DEVICE_FATAL_IF(ca_output_grad == nullptr, "BWD: ca_output_grad is null");

                chunk_ws_a_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
                chunk_ws_b_stride = BACKWARD_CHUNK_SAMPLES * arch.channels;
            }

            int warp_id = tid / WARP_SIZE;
            int lane_id = tid % WARP_SIZE;
            int num_warps = blockDim.x / WARP_SIZE;

            // Zero d_ca_input before transport backward (d_source accumulates here)
            float* d_ca_input = nullptr;
            if (has_work) {
                DEVICE_FATAL_IF(organism->batch_ca_input_grads == nullptr,
                    "batch_ca_input_grads must be allocated for backward pass");
                d_ca_input = organism->batch_ca_input_grads + s_wave_offsets.ca_states_offset;
                int d_ca_total = training_mode->batch_size * arch.num_heads * num_cells * arch.channels;
                for (int idx = tid; idx < d_ca_total; idx += blockDim.x) {
                    d_ca_input[idx] = 0.0f;
                }
            }
            cg::this_grid().sync();

            {
            // Transport backward: ca_output_grad → d_interaction, d_source, d_flow_projection_weights
            if (has_work) {
                float* ca_input = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
                half* flow_proj_w = ca_state->flow_projection_weights;
                int batch_size = training_mode->batch_size;
                int channels = arch.channels;
                int head_dim = arch.head_dim;
                int num_heads_bwd = arch.num_heads;
                int max_grad_buffer = ca_state->tape.value_capacity;

                // Zero flow projection grads in tape
                int fp_grad_total = num_heads_bwd * 2 * head_dim;
                for (int idx = tid; idx < fp_grad_total; idx += blockDim.x) {
                    int h = idx / (2 * head_dim);
                    int local = idx % (2 * head_dim);
                    ca_state->tape.grad_buffer[param_map->flow_projection_start[h] + local] = 0.0f;
                }

                // Zero dL_dinteraction
                int total_dI = num_heads_bwd * batch_size * num_cells * head_dim;
                for (int idx = tid; idx < total_dI; idx += blockDim.x) {
                    dL_dinteraction[idx] = 0.0f;
                }

                for (int batch_id = 0; batch_id < batch_size; batch_id++) {
                    for (int head = 0; head < num_heads_bwd; head++) {
                        int saved_base = batch_id * I_batch_stride + head * I_head_stride;
                        int flow_weight_offset = head * 2 * head_dim;
                        int batch_cell_offset = (batch_id * num_heads_bwd + head) * num_cells * channels;

                        for (int cell_idx = tid; cell_idx < num_cells; cell_idx += blockDim.x) {
                            float interaction_local[HEAD_DIM];
                            float interaction_sum = 0.0f;
                            for (int d = 0; d < head_dim; d++) {
                                float val = interaction_saved[saved_base + cell_idx * head_dim + d];
                                interaction_local[d] = val;
                                interaction_sum += fabsf(val);
                            }

                            float2 flow = FlowLeniaOps::project_to_flow(
                                interaction_local, head_dim, &flow_proj_w[flow_weight_offset]);
                            float gate_input = interaction_sum / (float)head_dim - compute_ca_gate_center(s_task_accuracy);
                            float gate = activation_sigmoid(gate_input);

                            float d_source[CHANNELS];
                            float d_flow_x, d_flow_y, d_gate_val;
                            FlowLeniaOps::bilinear_transport_backward(
                                ca_input, cell_idx, flow, gate, entry->flow_resource_dt,
                                arch.grid_size, ca_output_grad,
                                d_source, &d_flow_x, &d_flow_y, &d_gate_val,
                                channels, batch_cell_offset);

                            // d_source → d_ca_input (written later after zeroing)
                            // For now store in dL_dperception buffer temporarily
                            // (will be accumulated after d_ca_input is zeroed)

                            // Flow projection backward
                            float d_interaction_local[HEAD_DIM];
                            float d_weights_local[2 * HEAD_DIM];
                            for (int d = 0; d < head_dim; d++) {
                                d_interaction_local[d] = 0.0f;
                            }
                            for (int d = 0; d < 2 * head_dim; d++) {
                                d_weights_local[d] = 0.0f;
                            }
                            FlowLeniaOps::project_to_flow_backward(
                                d_flow_x, d_flow_y, interaction_local, head_dim,
                                &flow_proj_w[flow_weight_offset],
                                d_interaction_local, d_weights_local);

                            // Gate backward: d_gate → sigmoid backward → d_interaction_sum
                            float d_sigmoid = d_gate_val * gate * (1.0f - gate);
                            float d_interaction_sum = d_sigmoid / (float)head_dim;
                            for (int d = 0; d < head_dim; d++) {
                                float sign_val = (interaction_local[d] > 0.0f) ? 1.0f :
                                                 (interaction_local[d] < 0.0f) ? -1.0f : 0.0f;
                                d_interaction_local[d] += d_interaction_sum * sign_val;
                            }

                            // Write d_interaction to dL_dinteraction
                            int out_base = head * I_head_stride + batch_id * I_batch_stride + cell_idx * head_dim;
                            for (int d = 0; d < head_dim; d++) {
                                dL_dinteraction[out_base + d] = d_interaction_local[d];
                            }

                            // Accumulate flow projection weight grads
                            int fp_grad_base = param_map->flow_projection_start[head];
                            for (int d = 0; d < 2 * head_dim; d++) {
                                atomicAdd(&ca_state->tape.grad_buffer[fp_grad_base + d], d_weights_local[d]);
                            }

                            // Accumulate d_source into d_ca_input (zeroed before this section)
                            for (int c = 0; c < channels; c++) {
                                atomicAdd(&d_ca_input[batch_cell_offset + cell_idx * channels + c], d_source[c]);
                            }
                        }
                    }
                }
            }
            cg::this_grid().sync();

            // V:BWD_INTER_TRANSPOSE - work guarded, sync outside
            if (has_work) {
                int dW_tiles = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int total_tile_elements = dW_tiles * dW_tiles * arch.num_heads * WMMA_TILE_DIM * WMMA_TILE_DIM;
                int max_W_per_head = arch.head_dim * arch.head_dim;
                int max_W_T_per_head = ws_W_T_interaction_stride;

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
                        int W_src_idx = y * arch.head_dim + x;
                        int W_T_dst_idx = x * arch.head_dim + y;
                        PROVENANCE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "BWD inter transpose: head_id OOB");
                        PROVENANCE_FATAL_IF(W_src_idx < 0 || W_src_idx >= max_W_per_head, "BWD inter transpose: src OOB");
                        PROVENANCE_FATAL_IF(W_T_dst_idx < 0 || W_T_dst_idx >= max_W_T_per_head, "BWD inter transpose: dst OOB");
                        const half* W_head = ca_state->interaction_weights + head_id * arch.head_dim * arch.head_dim;
                        half* W_T_head = ws_W_T + head_id * ws_W_T_interaction_stride;
                        half w_val = W_head[W_src_idx];
                        PROVENANCE_FATAL_IF(!isfinite(__half2float(w_val)), "BWD inter transpose: W NaN/Inf");
                        W_T_head[W_T_dst_idx] = w_val;
                    }
                }
            }
            cg::this_grid().sync();

            // Zero interaction/perception dW - work guarded, sync outside
            if (has_work && tid == 0) atomicAdd(&g_v_bwd_zero_dw_count, 1);
            if (has_work) {
                int total_interaction = arch.num_heads * arch.head_dim * arch.head_dim;
                int total_perception = arch.num_heads * arch.channels * arch.head_dim;
                int total_zero = total_interaction + total_perception;
                PROVENANCE_FATAL_IF(total_zero <= 0, "BWD zero dW: total_zero overflow");
                float* ws_dW_inter = ws_dW;
                for (int idx = tid; idx < total_zero; idx += blockDim.x) {
                    ws_dW_inter[idx] = 0.0f;
                }
            }
            cg::this_grid().sync();
            if (tid == 0 && blockIdx.x == 0) printf("V:bwd_zero_dW\n");
            if (has_work && tid == 0 && blockIdx.x == 0) {
                TraceBuffer* mid_tb = &organism->ca_state_pool[pool->alive_indices[wave_start]].trace;
                printf("TRACE_MID_A eid=%d cidx=%d cyc0=%llu\n",
                    pool->alive_indices[wave_start], mid_tb->current_idx, mid_tb->traces[0].cycles_elapsed);
            }

            // Per-block variables for second chunk loop - uninitialized, only valid when has_work
            int chunk_ws_dI_stride;
            int chunk_ws_dpregelu_stride;
            int chunk_ws_pooled_stride;
            float* ws_dW_interaction;
            float* ws_dW_perception;
            half* ws_W_T_interaction;

            if (has_work) {
                chunk_ws_dI_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
                chunk_ws_dpregelu_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
                chunk_ws_pooled_stride = BACKWARD_CHUNK_SAMPLES * arch.channels;
                ws_dW_interaction = ws_dW;
                ws_dW_perception = ws_dW_interaction + arch.num_heads * ws_dW_interaction_stride;
                ws_W_T_interaction = ws_W_T;
            }
            cg::this_grid().sync();

            if (has_work && tid == 0) atomicAdd(&g_v_bwd_setup_done_count, 1);

            // Second chunk loop - all blocks iterate bwd_num_chunks, only blocks with work do actual work
            for (int chunk_idx = 0; chunk_idx < bwd_num_chunks; chunk_idx++) {
                int chunk_start = chunk_idx * BACKWARD_CHUNK_SAMPLES;
                int chunk_samples = has_work ? min(BACKWARD_CHUNK_SAMPLES, total_samples - chunk_start) : 0;
                int chunk_samples_aligned = (chunk_samples / WMMA_TILE_DIM) * WMMA_TILE_DIM;
                bool chunk_has_work = has_work && (chunk_samples_aligned > 0);

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_chunk2_enter_count, 1);

                // GELU backward: dL_dinteraction (from transport bwd) → ws_dpregelu
                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_di_write_count, 1);
                if (chunk_has_work) {
                    int total_elem = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_out = arch.num_heads * training_mode->batch_size * num_cells * arch.head_dim;
                    int max_saved = training_mode->batch_size * I_batch_stride + arch.num_heads * I_head_stride;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
                    for (int idx = tid; idx < total_elem; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int sample_in_chunk = remainder / arch.head_dim;
                        int dim_idx = remainder % arch.head_dim;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_id = global_sample % num_cells;

                        int out_idx = head_id * I_head_stride + batch_id * I_batch_stride + cell_id * arch.head_dim + dim_idx;
                        int saved_idx = batch_id * I_batch_stride + head_id * I_head_stride + cell_id * arch.head_dim + dim_idx;
                        int dpregelu_idx = head_id * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + dim_idx;

                        PROVENANCE_FATAL_IF(out_idx < 0 || out_idx >= max_out, "BWD dI: out_idx OOB");
                        PROVENANCE_FATAL_IF(saved_idx < 0 || saved_idx >= max_saved, "BWD dI: saved_idx OOB");
                        PROVENANCE_FATAL_IF(dpregelu_idx < 0 || dpregelu_idx >= max_dpregelu, "BWD dI: dpregelu_idx OOB");

                        float dL_dI_val = dL_dinteraction[out_idx];
                        PROVENANCE_FATAL_IF(!isfinite(dL_dI_val), "BWD dI: dL_dinteraction NaN/Inf");
                        float pre_val = pre_gelu_saved[saved_idx];
                        PROVENANCE_FATAL_IF(!isfinite(pre_val), "BWD dI: pre_gelu_saved NaN/Inf");
                        float gelu_bwd = activation_gelu_backward(pre_val, dL_dI_val);
                        PROVENANCE_FATAL_IF(!isfinite(gelu_bwd), "BWD dI: gelu_bwd NaN/Inf");
                        ws_dpregelu[dpregelu_idx] = gelu_bwd;
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_perc_load_count, 1);
                if (chunk_has_work) {
                    int total_P = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_src_P = arch.num_heads * training_mode->batch_size * num_cells * arch.head_dim;
                    int max_dst_P = arch.num_heads * chunk_ws_a_stride;
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
                        PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_src_P, "BWD perc: src_idx OOB");
                        PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_dst_P, "BWD perc: dst_idx OOB");
                        float val = perception_saved[src_idx];
                        PROVENANCE_FATAL_IF(!isfinite(val), "BWD perc: perception_saved NaN/Inf");
                        ws_fp16_a[dst_idx] = __float2half(val);
                    }
                }
                if (chunk_has_work) {
                    int total_D = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
                    int max_dst_D = arch.num_heads * chunk_ws_a_stride;
                    for (int idx = tid; idx < total_D; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int src_idx = head_id * chunk_ws_dpregelu_stride + remainder;
                        int dst_idx = head_id * chunk_ws_a_stride + remainder;
                        PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_dpregelu, "BWD D: src_idx OOB");
                        PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_dst_D, "BWD D: dst_idx OOB");
                        float val = ws_dpregelu[src_idx];
                        PROVENANCE_FATAL_IF(!isfinite(val), "BWD D: ws_dpregelu NaN/Inf");
                        ws_fp16_b[dst_idx] = __float2half(val);
                    }
                }
                cg::this_grid().sync();


                if (chunk_has_work) {
                    int dW_tiles = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = dW_tiles * dW_tiles * arch.num_heads;
                    int max_ws_fp16_a_inter = arch.num_heads * chunk_ws_a_stride;
                    int max_ws_dW_inter = arch.num_heads * ws_dW_interaction_stride;

                    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                        int head_id = tile_idx / (dW_tiles * dW_tiles);
                        int tile_flat = tile_idx % (dW_tiles * dW_tiles);
                        int warpM = tile_flat / dW_tiles;
                        int warpN = tile_flat % dW_tiles;
                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        DEVICE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "WMMA inter dW: head_id OOB");

                        if (tile_row < arch.head_dim && tile_col < arch.head_dim) {
                            int A_base_offset = head_id * chunk_ws_a_stride;
                            int B_base_offset = head_id * chunk_ws_a_stride;
                            int C_base_offset = head_id * ws_dW_interaction_stride;
                            DEVICE_FATAL_IF(A_base_offset < 0 || A_base_offset >= max_ws_fp16_a_inter, "WMMA inter dW: A_base OOB");
                            DEVICE_FATAL_IF(B_base_offset < 0 || B_base_offset >= max_ws_fp16_a_inter, "WMMA inter dW: B_base OOB");
                            DEVICE_FATAL_IF(C_base_offset < 0 || C_base_offset >= max_ws_dW_inter, "WMMA inter dW: C_base OOB");

                            const half* A_head = ws_fp16_a + A_base_offset;
                            const half* B_head = ws_fp16_b + B_base_offset;
                            float* C_head = ws_dW_interaction + C_base_offset;

                            int C_tile_offset = tile_row * arch.head_dim + tile_col;
                            int C_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                            DEVICE_FATAL_IF(C_tile_offset < 0 || C_tile_max >= ws_dW_interaction_stride, "WMMA inter dW: C_tile OOB");

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                            nvcuda::wmma::load_matrix_sync(c_frag, C_head + C_tile_offset, arch.head_dim, nvcuda::wmma::mem_row_major);

                            for (int k_tile = 0; k_tile < chunk_samples_aligned; k_tile += WMMA_TILE_DIM) {
                                int A_tile_offset = k_tile * arch.head_dim + tile_row;
                                int A_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_row + WMMA_TILE_DIM - 1);
                                int B_tile_offset = k_tile * arch.head_dim + tile_col;
                                int B_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                                DEVICE_FATAL_IF(A_tile_offset < 0 || A_tile_max >= chunk_ws_a_stride, "WMMA inter dW: A_tile OOB");
                                DEVICE_FATAL_IF(B_tile_offset < 0 || B_tile_max >= chunk_ws_a_stride, "WMMA inter dW: B_tile OOB");
                                nvcuda::wmma::load_matrix_sync(a_frag, A_head + A_tile_offset, arch.head_dim);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + B_tile_offset, arch.head_dim);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + C_tile_offset, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                cg::this_grid().sync();
                if (tid == 0 && blockIdx.x == 0 && chunk_idx == 0) printf("V:bwd_inter_dW\n");


                if (chunk_has_work) {
                    int tiles_M = (chunk_samples_aligned + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int tiles_N = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = tiles_M * tiles_N * arch.num_heads;
                    int max_ws_fp16_b_dP = arch.num_heads * chunk_ws_a_stride;
                    int max_ws_W_T_inter = arch.num_heads * ws_W_T_interaction_stride;
                    int max_ws_dI_dP = arch.num_heads * chunk_ws_dI_stride;

                    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                        int head_id = tile_idx / (tiles_M * tiles_N);
                        int tile_flat = tile_idx % (tiles_M * tiles_N);
                        int warpM = tile_flat / tiles_N;
                        int warpN = tile_flat % tiles_N;
                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        DEVICE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "WMMA dP: head_id OOB");

                        if (tile_row < chunk_samples_aligned && tile_col < arch.head_dim) {
                            int A_base_offset = head_id * chunk_ws_a_stride;
                            int B_base_offset = head_id * ws_W_T_interaction_stride;
                            int C_base_offset = head_id * chunk_ws_dI_stride;
                            DEVICE_FATAL_IF(A_base_offset < 0 || A_base_offset >= max_ws_fp16_b_dP, "WMMA dP: A_base OOB");
                            DEVICE_FATAL_IF(B_base_offset < 0 || B_base_offset >= max_ws_W_T_inter, "WMMA dP: B_base OOB");
                            DEVICE_FATAL_IF(C_base_offset < 0 || C_base_offset >= max_ws_dI_dP, "WMMA dP: C_base OOB");

                            const half* A_head = ws_fp16_b + A_base_offset;
                            const half* B_head = ws_W_T_interaction + B_base_offset;
                            float* C_head = ws_dI + C_base_offset;

                            int C_tile_offset = tile_row * arch.head_dim + tile_col;
                            int C_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                            DEVICE_FATAL_IF(C_tile_offset < 0 || C_tile_max >= chunk_ws_dI_stride, "WMMA dP: C_tile OOB");

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                            nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                            for (int k_tile = 0; k_tile < arch.head_dim; k_tile += WMMA_TILE_DIM) {
                                if (k_tile + WMMA_TILE_DIM <= arch.head_dim) {
                                    int A_tile_offset = tile_row * arch.head_dim + k_tile;
                                    int A_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.head_dim + (k_tile + WMMA_TILE_DIM - 1);
                                    int B_tile_offset = k_tile * arch.head_dim + tile_col;
                                    int B_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                                    DEVICE_FATAL_IF(A_tile_offset < 0 || A_tile_max >= chunk_ws_a_stride, "WMMA dP: A_tile OOB");
                                    DEVICE_FATAL_IF(B_tile_offset < 0 || B_tile_max >= ws_W_T_interaction_stride, "WMMA dP: B_tile OOB");
                                    nvcuda::wmma::load_matrix_sync(a_frag, A_head + A_tile_offset, arch.head_dim);
                                    nvcuda::wmma::load_matrix_sync(b_frag, B_head + B_tile_offset, arch.head_dim);
                                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                                }
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + C_tile_offset, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_dp_write_count, 1);
                if (chunk_has_work) {
                    int total_elem = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_ws_dI = arch.num_heads * chunk_ws_dI_stride;
                    int max_out = arch.num_heads * training_mode->batch_size * num_cells * arch.head_dim;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
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
                        int dpregelu_idx = head_id * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + dim_idx;

                        PROVENANCE_FATAL_IF(ws_idx < 0 || ws_idx >= max_ws_dI, "BWD dP: ws_idx OOB");
                        PROVENANCE_FATAL_IF(out_idx < 0 || out_idx >= max_out, "BWD dP: out_idx OOB");
                        PROVENANCE_FATAL_IF(dpregelu_idx < 0 || dpregelu_idx >= max_dpregelu, "BWD dP: dpregelu_idx OOB");

                        float dL_dP_val = ws_dI[ws_idx];
                        PROVENANCE_FATAL_IF(!isfinite(dL_dP_val), "BWD dP: ws_dI NaN/Inf");
                        dL_dperception[out_idx] = dL_dP_val;
                        float perc_val = perception_saved[out_idx];
                        PROVENANCE_FATAL_IF(!isfinite(perc_val), "BWD dP: perception_saved NaN/Inf");
                        float relu_grad = dL_dP_val * ((perc_val > 0.0f) ? 1.0f : 0.0f);
                        PROVENANCE_FATAL_IF(!isfinite(relu_grad), "BWD dP: relu_grad NaN/Inf");
                        ws_dpregelu[dpregelu_idx] = relu_grad;
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_im2col_count, 1);
                if (chunk_has_work) {
                    int im2col_sample_stride = chunk_samples_aligned * arch.channels;
                    int total_im2col_work = arch.num_heads * chunk_samples_aligned;
                    int max_input = training_mode->batch_size * arch.num_heads * num_cells * arch.channels;
                    int max_im2col = arch.num_heads * im2col_sample_stride;
                    const float* input_batch = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;

                    for (int idx = tid; idx < total_im2col_work; idx += blockDim.x) {
                        int head_id = idx / chunk_samples_aligned;
                        int sample_in_chunk = idx % chunk_samples_aligned;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_idx = global_sample % num_cells;
                        int cell_y = cell_idx / arch.grid_size;
                        int cell_x = cell_idx % arch.grid_size;

                        int head_input_offset = (batch_id * arch.num_heads + head_id) * num_cells * arch.channels;

                        for (int c = 0; c < arch.channels; c++) {
                            float sum = 0.0f;
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    int ny = max(0, min(arch.grid_size - 1, cell_y + dy));
                                    int nx = max(0, min(arch.grid_size - 1, cell_x + dx));
                                    int input_idx = head_input_offset + ny * arch.grid_size * arch.channels + nx * arch.channels + c;
                                    PROVENANCE_FATAL_IF(input_idx < 0 || input_idx >= max_input, "BWD im2col: input_idx OOB");
                                    sum += input_batch[input_idx];
                                }
                            }
                            int im2col_idx = head_id * im2col_sample_stride + sample_in_chunk * arch.channels + c;
                            PROVENANCE_FATAL_IF(im2col_idx < 0 || im2col_idx >= max_im2col, "BWD im2col: im2col_idx OOB");
                            ws_im2col[im2col_idx] = sum;
                        }
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_conv_fp16_count, 1);
                if (chunk_has_work) {
                    int total_conv = arch.num_heads * chunk_samples_aligned * arch.channels;
                    for (int idx = tid; idx < total_conv; idx += blockDim.x) {
                        float val = ws_im2col[idx];
                        PROVENANCE_FATAL_IF(!isfinite(val), "BWD conv fp16: ws_im2col NaN/Inf");
                        ws_fp16_a[idx] = __float2half(val);
                    }
                }
                if (chunk_has_work) {
                    int total_D = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
                    int max_dst_D = arch.num_heads * chunk_ws_a_stride;
                    for (int idx = tid; idx < total_D; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.head_dim);
                        int remainder = idx % (chunk_samples_aligned * arch.head_dim);
                        int src_idx = head_id * chunk_ws_dpregelu_stride + remainder;
                        int dst_idx = head_id * chunk_ws_a_stride + remainder;
                        PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_dpregelu, "BWD D fp16: src_idx OOB");
                        PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_dst_D, "BWD D fp16: dst_idx OOB");
                        float val = ws_dpregelu[src_idx];
                        ws_fp16_b[dst_idx] = __float2half(val);
                    }
                }
                cg::this_grid().sync();


                if (chunk_has_work) {
                    int ws_dW_perception_stride = arch.channels * arch.head_dim;
                    int dW_tiles_c = (arch.channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int dW_tiles_h = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = dW_tiles_c * dW_tiles_h * arch.num_heads;
                    int im2col_head_stride = chunk_samples_aligned * arch.channels;
                    int max_ws_fp16_a_perc = arch.num_heads * im2col_head_stride;
                    int chunk_B_stride = chunk_samples_aligned * arch.head_dim;
                    int max_ws_fp16_b_perc = arch.num_heads * chunk_B_stride;
                    int max_ws_dW_perc = arch.num_heads * ws_dW_perception_stride;

                    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                        int head_id = tile_idx / (dW_tiles_c * dW_tiles_h);
                        int tile_flat = tile_idx % (dW_tiles_c * dW_tiles_h);
                        int warpM = tile_flat / dW_tiles_h;
                        int warpN = tile_flat % dW_tiles_h;
                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        DEVICE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "WMMA perc dW: head_id OOB");

                        if (tile_row < arch.channels && tile_col < arch.head_dim) {
                            int A_base_offset = head_id * im2col_head_stride;
                            int B_base_offset = head_id * chunk_B_stride;
                            int C_base_offset = head_id * ws_dW_perception_stride;
                            DEVICE_FATAL_IF(A_base_offset < 0 || A_base_offset >= max_ws_fp16_a_perc, "WMMA perc dW: A_base OOB");
                            DEVICE_FATAL_IF(B_base_offset < 0 || B_base_offset >= max_ws_fp16_b_perc, "WMMA perc dW: B_base OOB");
                            DEVICE_FATAL_IF(C_base_offset < 0 || C_base_offset >= max_ws_dW_perc, "WMMA perc dW: C_base OOB");

                            const half* A_ptr = ws_fp16_a + A_base_offset;
                            const half* B_head = ws_fp16_b + B_base_offset;
                            float* C_head = ws_dW_perception + C_base_offset;

                            int C_tile_offset = tile_row * arch.head_dim + tile_col;
                            int C_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                            DEVICE_FATAL_IF(C_tile_offset < 0 || C_tile_max >= ws_dW_perception_stride, "WMMA perc dW: C_tile OOB");

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                            nvcuda::wmma::load_matrix_sync(c_frag, C_head + C_tile_offset, arch.head_dim, nvcuda::wmma::mem_row_major);

                            for (int k_tile = 0; k_tile < chunk_samples_aligned; k_tile += WMMA_TILE_DIM) {
                                int A_tile_offset = k_tile * arch.channels + tile_row;
                                int A_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.channels + (tile_row + WMMA_TILE_DIM - 1);
                                int B_tile_offset = k_tile * arch.head_dim + tile_col;
                                int B_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                                DEVICE_FATAL_IF(A_tile_offset < 0 || A_tile_max >= im2col_head_stride, "WMMA perc dW: A_tile OOB");
                                DEVICE_FATAL_IF(B_tile_offset < 0 || B_tile_max >= chunk_B_stride, "WMMA perc dW: B_tile OOB");
                                nvcuda::wmma::load_matrix_sync(a_frag, A_ptr + A_tile_offset, arch.channels);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + B_tile_offset, arch.head_dim);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + C_tile_offset, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_input_grad_count, 1);
                if (chunk_has_work) {
                    int weights_per_head = arch.channels * arch.head_dim;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
                    int im2col_head_stride_ig = chunk_samples_aligned * arch.channels;
                    int max_im2col = arch.num_heads * im2col_head_stride_ig;
                    int total_input_grad = arch.num_heads * chunk_samples_aligned * arch.channels;
                    for (int idx = tid; idx < total_input_grad; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.channels);
                        int remainder = idx % (chunk_samples_aligned * arch.channels);
                        int sample_in_chunk = remainder / arch.channels;
                        int channel_idx = remainder % arch.channels;

                        float d_input_accum = 0.0f;
                        int head_weight_start = head_id * weights_per_head;
                        for (int hd = 0; hd < arch.head_dim; hd++) {
                            int W_idx = head_weight_start + channel_idx * arch.head_dim + hd;
                            int dprerelu_idx = head_id * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + hd;
                            PROVENANCE_FATAL_IF(W_idx < 0 || W_idx >= arch.num_heads * weights_per_head, "BWD input: W_idx OOB");
                            PROVENANCE_FATAL_IF(dprerelu_idx < 0 || dprerelu_idx >= max_dpregelu, "BWD input: dprerelu_idx OOB");
                            d_input_accum += __half2float(ca_state->perception_weights[W_idx]) * ws_dpregelu[dprerelu_idx];
                        }
                        int im2col_idx = head_id * im2col_head_stride_ig + sample_in_chunk * arch.channels + channel_idx;
                        PROVENANCE_FATAL_IF(im2col_idx < 0 || im2col_idx >= max_im2col, "BWD input: im2col idx OOB");
                        ws_im2col[im2col_idx] = d_input_accum;
                    }
                }
                cg::this_grid().sync();
                if (tid == 0 && blockIdx.x == 0 && chunk_idx == 0) printf("V:bwd_input_grad\n");
                if (has_work && tid == 0 && blockIdx.x == 0 && chunk_idx == 0) {
                    TraceBuffer* mid_tb2 = &organism->ca_state_pool[pool->alive_indices[wave_start]].trace;
                    printf("TRACE_MID_B cidx=%d cyc0=%llu\n", mid_tb2->current_idx, mid_tb2->traces[0].cycles_elapsed);
                }

                if (chunk_has_work) {
                    if (tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_scatter_count, 1);
                    int im2col_head_stride_sc = chunk_samples_aligned * arch.channels;
                    int scatter_loop_bound = arch.num_heads * chunk_samples_aligned * arch.channels;
                    int max_d_ca = training_mode->batch_size * arch.num_heads * num_cells * arch.channels;
                    for (int idx = tid; idx < scatter_loop_bound; idx += blockDim.x) {
                        int head_id = idx / (chunk_samples_aligned * arch.channels);
                        int remainder = idx % (chunk_samples_aligned * arch.channels);
                        int sample_in_chunk = remainder / arch.channels;
                        int channel_idx = remainder % arch.channels;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_idx = global_sample % num_cells;
                        int cell_y = cell_idx / arch.grid_size;
                        int cell_x = cell_idx % arch.grid_size;
                        int im2col_idx = head_id * im2col_head_stride_sc + sample_in_chunk * arch.channels + channel_idx;
                        float d_pooled_val = ws_im2col[im2col_idx];

                        int head_d_ca_base = (batch_id * arch.num_heads + head_id) * num_cells * arch.channels;
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                int ny = cell_y + dy;
                                int nx = cell_x + dx;
                                if (ny >= 0 && ny < arch.grid_size && nx >= 0 && nx < arch.grid_size) {
                                    int out_cell_idx = ny * arch.grid_size + nx;
                                    int out_idx = head_d_ca_base + out_cell_idx * arch.channels + channel_idx;
                                    PROVENANCE_FATAL_IF(out_idx < 0 || out_idx >= max_d_ca, "BWD scatter: out_idx OOB");
                                    atomicAdd(&d_ca_input[out_idx], d_pooled_val);
                                }
                            }
                        }
                    }
                }
                cg::this_grid().sync();
            }

            if (has_work && tid == 0) atomicAdd(&g_v_bwd_chunks_done_count, 1);

            // Post-loop grad copy - work guarded, sync outside
            if (has_work && tid == 0) atomicAdd(&g_v_bwd_inter_grad_copy_count, 1);
            if (has_work) {
                int total_grads = arch.num_heads * arch.head_dim * arch.head_dim;
                int max_src = arch.num_heads * ws_dW_interaction_stride;
                int max_grad_buffer = ca_state->tape.capacity;
                PROVENANCE_FATAL_IF(total_grads <= 0, "BWD inter grad: total overflow");
                for (int idx = tid; idx < total_grads; idx += blockDim.x) {
                    int head_id = idx / (arch.head_dim * arch.head_dim);
                    int local_idx = idx % (arch.head_dim * arch.head_dim);
                    int src_idx = head_id * ws_dW_interaction_stride + local_idx;
                    int dst_idx = param_map->interaction_start[head_id] + local_idx;
                    PROVENANCE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "BWD inter grad: head_id OOB");
                    PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_src, "BWD inter grad: src_idx OOB");
                    PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_grad_buffer, "BWD inter grad: dst_idx OOB");
                    float val = ws_dW_interaction[src_idx];
                    PROVENANCE_FATAL_IF(!isfinite(val), "BWD inter grad: src NaN/Inf");
                    ca_state->tape.grad_buffer[dst_idx] = val;
                }
            }
            cg::this_grid().sync();

            if (has_work && tid == 0) atomicAdd(&g_v_bwd_perc_grad_copy_count, 1);
            if (has_work) {
                int ws_dW_perception_stride = arch.channels * arch.head_dim;
                int weights_per_head = arch.channels * arch.head_dim;
                int total_grads = arch.num_heads * weights_per_head;
                int max_src = arch.num_heads * ws_dW_perception_stride;
                int max_grad_buffer = ca_state->tape.capacity;
                PROVENANCE_FATAL_IF(total_grads <= 0, "BWD perc grad: total overflow");
                for (int idx = tid; idx < total_grads; idx += blockDim.x) {
                    int head_id = idx / weights_per_head;
                    int local_idx = idx % weights_per_head;
                    int src_idx = head_id * ws_dW_perception_stride + local_idx;
                    int dst_idx = param_map->perception_start[head_id] + local_idx;
                    PROVENANCE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "BWD perc grad: head_id OOB");
                    PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_src, "BWD perc grad: src_idx OOB");
                    PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_grad_buffer, "BWD perc grad: dst_idx OOB");
                    float val = ws_dW_perception[src_idx];
                    PROVENANCE_FATAL_IF(!isfinite(val), "BWD perc grad: src NaN/Inf");
                    ca_state->tape.grad_buffer[dst_idx] = val;
                }
            }
            cg::this_grid().sync();

            if (has_work && tid == 0) atomicAdd(&g_v_bwd_grad_conc_count, 1);
            if (has_work && d_ca_input != nullptr) {
                DEVICE_FATAL_IF(organism->buffers->grad_concentration_buffer == nullptr, "BWD: grad_concentration_buffer is null");
                float* grad_conc = organism->buffers->grad_concentration_buffer;
                int max_d_ca = total_samples * arch.channels;
                for (int cell = tid; cell < num_cells; cell += blockDim.x) {
                    int src_idx = cell * arch.channels;
                    PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_d_ca, "BWD grad_conc: src_idx OOB");
                    PROVENANCE_FATAL_IF(cell < 0 || cell >= num_cells, "BWD grad_conc: cell OOB");
                    float val = d_ca_input[src_idx];
                    PROVENANCE_FATAL_IF(!isfinite(val), "BWD grad_conc: d_ca_input NaN/Inf");
                    grad_conc[cell] = val;
                }
            }
            cg::this_grid().sync();
        }

        cg::this_grid().sync();

            // Record elapsed cycles for backward pass
            if (bwd_trace_slot != nullptr && tid == 0) {
                unsigned long long bwd_cycle_end = clock64();
                unsigned long long elapsed = bwd_cycle_end - bwd_cycle_start;
                bwd_trace_slot->cycles_elapsed = elapsed;
                bwd_trace_slot->tensor_core_cycles = elapsed;
            }
        }

        // CHECKPOINT: verify trace buffer state after backward
        if (has_work && tid == 0 && blockIdx.x == 0) {
            TraceBuffer* chk_tb = &organism->ca_state_pool[pool->alive_indices[wave_start]].trace;
            printf("TRACE_CHK1 eid=%d cidx=%d cyc0=%llu br0=%llu cidx_addr=%p ca_state=%p\n",
                pool->alive_indices[wave_start], chk_tb->current_idx,
                chk_tb->traces[0].cycles_elapsed, chk_tb->traces[0].total_branches,
                &chk_tb->current_idx, (void*)&organism->ca_state_pool[pool->alive_indices[wave_start]]);
        }

        // Effective rank computation - uses param_map interleaved layout [P0|I0|V0|P1|I1|V1|...]
        // Blocks without work only participate in syncs, never touch data.
        {
            __shared__ float head_grad_sq[NUM_HEADS];
            __shared__ float warp_sums_eff[32];

            for (int h = 0; h < NUM_HEADS; h++) {
                if (has_work && h < arch.num_heads) {
                    int head_start = param_map->head_param_offsets[h];
                    int head_count = param_map->head_param_counts[h];
                    float* grad_buf = ca_state->tape.grad_buffer;

                    float local_sq = 0.0f;
                    for (int i = tid; i < head_count; i += blockDim.x) {
                        float g = grad_buf[head_start + i];
                        local_sq += g * g;
                    }

                    unsigned mask = __activemask();
                    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                        local_sq += __shfl_down_sync(mask, local_sq, offset);
                    }
                    int lane = tid % warpSize;
                    int warp_id = tid / warpSize;
                    if (lane == 0) warp_sums_eff[warp_id] = local_sq;
                }
                cg::this_grid().sync();

                if (has_work && h < arch.num_heads) {
                    float local_sq = warp_sums_eff[tid < blockDim.x / warpSize ? tid : 0];
                    if (tid < blockDim.x / warpSize) {
                        unsigned active = __activemask();
                        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
                            local_sq += __shfl_down_sync(active, local_sq, offset);
                        }
                    }
                    if (tid == 0) {
                        head_grad_sq[h] = sqrtf(local_sq / (float)param_map->head_param_counts[h]);
                    }
                }
                cg::this_grid().sync();
            }

            if (has_work && tid == 0) {
                float total_sq = 0.0f;
                for (int h = 0; h < arch.num_heads; h++) {
                    total_sq += head_grad_sq[h] * head_grad_sq[h];
                }

                DEVICE_FATAL_IF(total_sq < 1e-12f, "effective_rank: zero gradient magnitude after backward pass - training catastrophically broken");

                float entropy = 0.0f;
                for (int h = 0; h < arch.num_heads; h++) {
                    float g = head_grad_sq[h];
                    float p = (g * g) / total_sq;
                    if (p > 1e-12f) {
                        entropy -= p * logf(p);
                    }
                }
                float eff_rank = expf(entropy);
                float clamped_rank = fmaxf(1.0f, fminf((float)arch.num_heads, eff_rank));
                measured_value_set_computed(&entry->effective_rank, clamped_rank, organism->generation, entry->genome_hash);
            }
            cg::this_grid().sync();
        }
        if (tid == 0 && blockIdx.x == 0) printf("V:HYB_effrank_done\n");

    int alive_ct = pool->alive_indices_count;
    int wave_end_compact = min(wave_start + (int)gridDim.x, alive_ct);
    bool is_last_wave = (wave_end_compact >= alive_ct);

        // Per-entry Adam updates: each block updates its own entry using actual entry_idx
        if (has_work && training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {
            adam_update_perception_device(organism, entry_idx);
            adam_update_interaction_device(organism, entry_idx);
            adam_update_flow_projection_device(organism, entry_idx);
            adam_update_pooling_device(organism, entry_idx);
            adam_update_fc_weights_device(organism, entry_idx);
            adam_update_fc_bias_device(organism, entry_idx);
        }
        cg::this_grid().sync();

        // Adam timestep: increment once per generation (last wave only)
        if (has_work && blockIdx.x == 0 && is_last_wave && training_mode->batch_samples != nullptr && training_mode->classifier != nullptr) {
            if (tid == 0) {
                training_mode->adam_timestep++;
            }
        }
        cg::this_grid().sync();

    grid_barrier(gridDim.x);
    if (tid == 0) atomicAdd(&g_v_post_bwd_barrier_count, 1);

    float* component_workspace_genomes = organism->buffers->component_workspace_genomes_buffer;
    GPUElite* archive = organism->archive;
    int archive_size_val = organism->archive_size;

    // CHECKPOINT 2: verify trace buffer state before aggregation
    if (has_work && tid == 0 && blockIdx.x == 0) {
        TraceBuffer* chk_tb2 = &organism->ca_state_pool[pool->alive_indices[wave_start]].trace;
        printf("TRACE_CHK2 eid=%d cidx=%d cyc0=%llu br0=%llu\n",
            pool->alive_indices[wave_start], chk_tb2->current_idx,
            chk_tb2->traces[0].cycles_elapsed, chk_tb2->traces[0].total_branches);
    }

    // Aggregate per-entry trace buffers into PoolEntry HW counters
    // Wave-scoped: only aggregate entries processed by this wave
    if (has_work && blockIdx.x == 0) {
        for (int compact = wave_start + tid; compact < wave_end_compact; compact += blockDim.x) {
            int eid = pool->alive_indices[compact];
            PoolEntry* ent = &pool->entries[eid];
            TraceBuffer* tb = &organism->ca_state_pool[eid].trace;
            ent->cycles_elapsed = 0;
            ent->inst_executed = 0;
            ent->inst_issued = 0;
            ent->tensor_core_cycles = 0;
            ent->divergent_branches = 0;
            ent->total_branches = 0;
            int trace_count = tb->current_idx;
            printf("TRACE_AGG eid=%d cidx=%d cap=%d traces=%p cyc0=%llu br0=%llu inst0=%llu\n",
                eid, trace_count, tb->capacity, tb->traces,
                tb->traces[0].cycles_elapsed,
                tb->traces[0].total_branches,
                tb->traces[0].inst_executed);
            for (int i = 0; i < trace_count; i++) {
                ExecutionTrace* t = &tb->traces[i];
                ent->cycles_elapsed += t->cycles_elapsed;
                ent->inst_executed += t->inst_executed;
                ent->inst_issued += t->inst_issued;
                ent->tensor_core_cycles += t->tensor_core_cycles;
                ent->divergent_branches += t->divergent_branches;
                ent->total_branches += t->total_branches;
            }
        }
    }
    cg::this_grid().sync();

    // Wave-scoped: compute metrics for entries processed by this wave
    if (has_work && blockIdx.x == 0) {
        for (int compact = wave_start + tid; compact < wave_end_compact; compact += blockDim.x) {
            int eid = pool->alive_indices[compact];
            DEVICE_FATAL_IF(!pool->alive_flags[eid], "hybrid_lifecycle: dead entry in alive_indices (metrics loop)");

            PoolEntry* ent = &pool->entries[eid];
            float* eid_primary_genome = &component_workspace_genomes[eid * 2 * GENOME_SIZE];
            float* eid_parent_temp = &component_workspace_genomes[eid * 2 * GENOME_SIZE + GENOME_SIZE];

            reconstruct_genome_from_archive(ent->parent_hash, archive, archive_size_val,
                ent->delta_indices, ent->delta_values, ent->num_deltas,
                ent->max_deltas, eid_primary_genome, GENOME_SIZE, eid_parent_temp, organism->diresa_genome_weights);

            float gen_gap_val = fabsf(ent->train_accuracy.value - ent->test_accuracy.value);
            measured_value_set_computed(&ent->generalization_gap, gen_gap_val, generation, ent->genome_hash);

            float ipc = (float)ent->inst_executed / (float)ent->cycles_elapsed;
            float tensor_util = (float)ent->tensor_core_cycles / (float)ent->cycles_elapsed;
            float branch_efficiency = (float)(ent->total_branches - ent->divergent_branches) / (float)ent->total_branches;
            float hw_eff_val = ipc * tensor_util * branch_efficiency;
            measured_value_set_computed(&ent->hardware_efficiency, hw_eff_val, generation, ent->genome_hash);

            DEVICE_VALIDATE_FINITE(ent->task_accuracy.value);
            DEVICE_VALIDATE_FINITE(ent->hardware_efficiency.value);
            DEVICE_VALIDATE_FINITE(ent->generalization_gap.value);
            DEVICE_VALIDATE_PROBABILITY(ent->task_accuracy.value);
            DEVICE_VALIDATE_HW_COUNTER(ent->cycles_elapsed, 1ULL, 0xFFFFFFFFFFFFULL);
            DEVICE_VALIDATE_HW_COUNTER(ent->inst_executed, 1ULL, 0xFFFFFFFFFFFFULL);
            DEVICE_VALIDATE_HW_COUNTER(ent->tensor_core_cycles, 0ULL, ent->cycles_elapsed);

            organism->fitness_history[(generation % 2) * POOL_CAPACITY_MAX + eid] = ent->task_accuracy.value;

            if (generation > 0) {
                float prev_acc = organism->fitness_history[((generation - 1) % 2) * POOL_CAPACITY_MAX + eid];
                float coherence_val = ent->task_accuracy.value - prev_acc;
                measured_value_set_computed(&ent->coherence, coherence_val, generation, ent->genome_hash);
                DEVICE_VALIDATE_FINITE(ent->coherence.value);
                device_validate_fitness_components(pool->fitness_values[eid], ent->coherence.value, ent->effective_rank.value, "pool_entry_fitness");
                organism->coherence_history[(generation % 2) * POOL_CAPACITY_MAX + eid] = ent->coherence.value;
            }
        }
    }
    cg::this_grid().sync();

    {
        // Population-level reduction: must sum ALL entries, so only run on last wave
        // (all per-entry metrics have been computed by now across all waves)
        float local_acc = 0.0f, local_gap = 0.0f, local_hw = 0.0f, local_fit = 0.0f;
        if (is_last_wave && has_work && blockIdx.x == 0) {
            for (int compact = tid; compact < alive_ct; compact += blockDim.x) {
                int eid = pool->alive_indices[compact];
                local_acc += pool->entries[eid].task_accuracy.value;
                local_gap += pool->entries[eid].generalization_gap.value;
                local_hw += pool->entries[eid].hardware_efficiency.value;
                local_fit += pool->fitness_values[eid];
            }
        }
        sdata[tid] = local_acc;
        cg::this_grid().sync();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            cg::this_grid().sync();
        }
        float total_acc = sdata[0];

        sdata[tid] = local_gap;
        cg::this_grid().sync();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            cg::this_grid().sync();
        }
        float total_gap = sdata[0];

        sdata[tid] = local_hw;
        cg::this_grid().sync();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            cg::this_grid().sync();
        }
        float total_hw = sdata[0];

        sdata[tid] = local_fit;
        cg::this_grid().sync();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            cg::this_grid().sync();
        }
        float total_fit = sdata[0];

        if (tid == 0 && is_last_wave && has_work && blockIdx.x == 0) {
            organism->telemetry->population_metrics.total_accuracy = total_acc;
            organism->telemetry->population_metrics.total_generalization_gap = total_gap;
            organism->telemetry->population_metrics.total_hardware_efficiency = total_hw;
            organism->telemetry->population_metrics.total_fitness = total_fit;
        }
    }
    cg::this_grid().sync();

    // Wave-scoped: Baldwin learning for entries in this wave
    if (has_work && blockIdx.x == 0 && generation > 0) {
        for (int compact = wave_start + tid; compact < wave_end_compact; compact += blockDim.x) {
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
    cg::this_grid().sync();
    if (tid == 0 && blockIdx.x == 0) printf("V:HYB_baldwin_done\n");

    if (training_mode->use_gradients) {
        {
            int grid_size = arch.grid_size;
            int ca_channels = arch.channels;
            int total_cells = grid_size * grid_size;
            int chem_channels = organism->chemical_field->channels;
            float* ca_concentration = entry->ca_state->ca_concentration;
            float* chemical_concentration = organism->chemical_field->concentration;

            for (int cell_idx = tid; cell_idx < total_cells; cell_idx += blockDim.x) {
                // Map CA channels 0..chem_channels-1 to chemical field channels
                for (int c = 0; c < chem_channels && c < ca_channels; c++) {
                    float val = ca_concentration[cell_idx * ca_channels + c];
                    if (isfinite(val)) {
                        atomicAdd(&chemical_concentration[c * total_cells + cell_idx], val);
                    }
                }
            }
        }
    }
    cg::this_grid().sync();

    if (training_mode->use_gradients) {
        {
            int weight_count = arch.num_heads * arch.channels * arch.head_dim;
            half* weights_fp16 = ca_state->perception_weights;
            float* weights_fp32 = ca_state->fp32_workspace;

            for (int idx = tid; idx < weight_count; idx += blockDim.x) {
                weights_fp32[idx] = __half2float(weights_fp16[idx]);
            }
        }
    }
    cg::this_grid().sync();

    float* behavioral_workspace_genomes = organism->buffers->behavioral_workspace_genomes_buffer;


    // Behavioral embedding: runs once over all agents, so last wave only
    if (is_last_wave && has_work && blockIdx.x == 0 && generation % EMBEDDING_UPDATE_FREQ == 0) {
        if (tid == 0) {
            *organism->buffers->behavioral_reconstruction_error = 0.0f;
        }
    }
    cg::this_grid().sync();

    if (is_last_wave && has_work && blockIdx.x == 0 && generation % EMBEDDING_UPDATE_FREQ == 0) {
        int hw_dim = BEHAVIORAL_DIM_HW;
        int task_dim = BEHAVIORAL_DIM_TASK;
        int gen_dim = BEHAVIORAL_DIM_GEN;
        int embed_behavioral_dim = hw_dim + task_dim + gen_dim;

        {
            BehavioralState* agents = organism->behavioral_agents;
            float* embedding_weights = organism->buffers->behavioral_embedding_weights;
            float* reconstruction_error = organism->buffers->behavioral_reconstruction_error;
            int num_agents = POOL_CAPACITY_MAX;
            float* features_buffer = organism->buffers->behavioral_features_buffer;

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

            float fourier_base_freq = FOURIER_BASE_FREQ;
            int fourier_num_octaves = min(FOURIER_NUM_OCTAVES, embed_behavioral_dim - 4);
            float fourier_spectrum_exponent = FOURIER_SPECTRUM_EXPONENT;

            for (int agent_id = tid; agent_id < num_agents; agent_id += blockDim.x) {
                BehavioralState* agent = &agents[agent_id];
                float* features = &features_buffer[agent_id * embed_behavioral_dim];

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
    }
    cg::this_grid().sync();

    // All heads deposit ch0 into shared chemical field
    if (has_work && organism->buffers->batch_prev_concentration && organism->chemical_field) {
        ChemicalField* field = organism->chemical_field;
        int grid_size_dep = arch.grid_size;
        int cells_dep = grid_size_dep * grid_size_dep;
        int channels_dep = arch.channels;
        int num_heads_dep = arch.num_heads;
        int batch_sz = training_mode->batch_size;
        int head_stride = cells_dep * channels_dep;
        float acc = fminf(fmaxf(entry->task_accuracy.value, 0.0f), 1.0f);
        float deposition_strength = 0.01f + 0.09f * acc;
        float* field_coords = organism->sample_field_coords + entry_idx * batch_sz * 2;

        for (int s = 0; s < batch_sz; s++) {
            int center_x = (int)field_coords[s * 2];
            int center_y = (int)field_coords[s * 2 + 1];
            int batch_base = s * num_heads_dep * head_stride;
            for (int cell = tid; cell < cells_dep; cell += blockDim.x) {
                int cx = cell % grid_size_dep;
                int cy = cell / grid_size_dep;
                int field_x = (cx + center_x) % grid_size_dep;
                int field_y = (cy + center_y) % grid_size_dep;
                int field_cell = field_y * grid_size_dep + field_x;

                float ch0_sum = 0.0f;
                for (int h = 0; h < num_heads_dep; h++) {
                    ch0_sum += wave_prev_conc[batch_base + h * head_stride + cell * channels_dep];
                }
                atomicAdd(&field->concentration[field_cell], ch0_sum * deposition_strength);
            }
        }
    }
    cg::this_grid().sync();

    // Only block 0 does telemetry and audit - prevents N blocks each printing
    if (tid == 0 && blockIdx.x == 0 && audit != nullptr && training_mode->batch_samples != nullptr) {
        run_telemetry_probes(organism, generation);
        float* logits = organism->gradient_logits_pool;
        int* labels = training_mode->batch_labels;
        float* batch_samples = training_mode->batch_samples;
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
            batch_samples,
            batch_size,
            num_classes,
            ca_concentration,
            grid_size,
            train_acc,
            test_acc,
            training_mode->is_train_batch,
            organism->telemetry,
            organism->pool,
            organism->chemical_field,
            ca_state,
            organism->hardware_geom,
            organism->archive_size
        );
    }
}


__device__ void check_convergence_device(Organism* organism, bool* converged) {
    float* workspace_genomes = organism->workspace_genomes;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float fitness = organism->fitness_history[(organism->generation % 2) * POOL_CAPACITY_MAX];
        float coherence = organism->coherence_history[(organism->generation % 2) * POOL_CAPACITY_MAX];

        // Find the best entry by fitness
        int best_idx = 0;
        float best_fitness = -1e10f;
        ComponentPool* pool = organism->pool;
        for (int i = 0; i < pool->capacity; i++) {
            if (pool->alive_flags[i] && pool->fitness_values[i] > best_fitness) {
                best_fitness = pool->fitness_values[i];
                best_idx = i;
            }
        }

        float* convergence_genome = &workspace_genomes[0];
        float* convergence_parent_temp = &workspace_genomes[GENOME_SIZE];
        PoolEntry* best_entry = &pool->entries[best_idx];
        reconstruct_genome_from_archive(best_entry->parent_hash, organism->archive, organism->archive_size,
            best_entry->delta_indices, best_entry->delta_values, best_entry->num_deltas,
            best_entry->max_deltas, convergence_genome, GENOME_SIZE, convergence_parent_temp, organism->diresa_genome_weights);

        float* genome = convergence_genome;

        int fitness_conv_slot = GenomeParamTable::convergence_fitness_threshold;
        int coherence_conv_slot = GenomeParamTable::convergence_coherence_threshold;
        int fitness_min_slot = GenomeParamTable::convergence_fitness_min;
        int fitness_max_slot = GenomeParamTable::convergence_fitness_max;
        int coherence_min_slot = GenomeParamTable::convergence_coherence_min;
        int coherence_max_slot = GenomeParamTable::convergence_coherence_max;

        float fitness_min = genome_slot_to_unit(genome, fitness_min_slot);
        float fitness_max = genome_slot_to_unit(genome, fitness_max_slot);
        float coherence_min = genome_slot_to_unit(genome, coherence_min_slot);
        float coherence_max = genome_slot_to_unit(genome, coherence_max_slot);

        InitContext conv_ctx;
        conv_ctx.derive_from_genome(genome, best_entry->gradients);

        float fitness_threshold = genome_to_param(
            genome,
            best_entry->gradients,
            fitness_conv_slot,
            conv_ctx.metabolic, conv_ctx.stress, conv_ctx.morphogen,
            organism->telemetry->genome_complexity.hash_entropy,
            organism->telemetry->archive_topology.novelty_gradient,
            organism->telemetry->diresa_evolution.behavioral_drift_rate,
            organism->telemetry->task_performance.accuracy,
            fitness_min, fitness_max
        );

        float coherence_threshold = genome_to_param(
            genome,
            best_entry->gradients,
            coherence_conv_slot,
            conv_ctx.metabolic, conv_ctx.stress, conv_ctx.morphogen,
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

    // Print aggregate DIAG counts at end of lifecycle
    if (threadIdx.x == 0) printf("DIAG_PRE_FINAL_BARRIER b%d\n", blockIdx.x);
    grid_barrier(gridDim.x);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("DIAG_AGGREGATE blocks_ca_fwd=%d blocks_entered=%d\n", g_blocks_ca_fwd, g_blocks_entered);
        // V:BWD aggregate counters
        printf("V:BWD_FLOW_DONE total=%d\n", g_v_flow_done_count);
        printf("V:BWD_ENTER total=%d\n", g_v_bwd_enter_count);
        printf("V:BWD_FATAL_CHECKS total=%d\n", g_v_bwd_fatal_checks_count);
        printf("V:BWD_CHUNK total=%d\n", g_v_bwd_chunk_count);
        printf("V:BWD_VALUE_GRAD total=%d\n", g_v_bwd_value_grad_count);
        printf("V:BWD_INTER_GRAD total=%d\n", g_v_bwd_inter_grad_count);
        printf("V:BWD_PERC_GRAD total=%d\n", g_v_bwd_perc_grad_count);
        printf("V:BWD_DONE total=%d\n", g_v_bwd_done_count);
        printf("V:BWD_ZERO_DW total=%d\n", g_v_bwd_zero_dw_count);
        printf("V:BWD_SETUP_DONE total=%d\n", g_v_bwd_setup_done_count);
        printf("V:BWD_CHUNKS_DONE total=%d\n", g_v_bwd_chunks_done_count);
        printf("V:BWD_INTER_GRAD_COPY total=%d\n", g_v_bwd_inter_grad_copy_count);
        printf("V:BWD_PERC_GRAD_COPY total=%d\n", g_v_bwd_perc_grad_copy_count);
        printf("V:BWD_GRAD_CONC total=%d\n", g_v_bwd_grad_conc_count);
        printf("V:BWD_DIFFUSION_LAUNCH total=%d\n", g_v_bwd_diffusion_launch_count);
        printf("V:BWD_CHUNK0 total=%d\n", g_v_bwd_chunk0_count);
        printf("V:BWD_I_DONE total=%d\n", g_v_bwd_i_done_count);
        printf("V:BWD_V_DONE total=%d\n", g_v_bwd_v_done_count);
        printf("V:BWD_CHUNK2_ENTER total=%d\n", g_v_bwd_chunk2_enter_count);
        printf("V:BWD_DI_WRITE total=%d\n", g_v_bwd_di_write_count);
        printf("V:BWD_PERC_LOAD total=%d\n", g_v_bwd_perc_load_count);
        printf("V:BWD_DP_WRITE total=%d\n", g_v_bwd_dp_write_count);
        printf("V:BWD_IM2COL total=%d\n", g_v_bwd_im2col_count);
        printf("V:BWD_CONV_FP16 total=%d\n", g_v_bwd_conv_fp16_count);
        printf("V:BWD_INPUT_GRAD total=%d\n", g_v_bwd_input_grad_count);
        printf("V:BWD_SCATTER total=%d\n", g_v_bwd_scatter_count);
        printf("V:BWD_DIFF_DEVICE total=%d\n", g_v_bwd_diff_device_count);
        printf("V:POST_BWD_BARRIER total=%d\n", g_v_post_bwd_barrier_count);

        // Reset all counters
        g_blocks_ca_fwd = 0;
        g_blocks_entered = 0;
        // Reset V:BWD counters
        g_v_flow_done_count = 0;
        g_v_bwd_enter_count = 0;
        g_v_bwd_fatal_checks_count = 0;
        g_v_bwd_chunk_count = 0;
        g_v_bwd_value_grad_count = 0;
        g_v_bwd_inter_grad_count = 0;
        g_v_bwd_perc_grad_count = 0;
        g_v_bwd_done_count = 0;
        g_v_bwd_zero_dw_count = 0;
        g_v_bwd_setup_done_count = 0;
        g_v_bwd_chunks_done_count = 0;
        g_v_bwd_inter_grad_copy_count = 0;
        g_v_bwd_perc_grad_copy_count = 0;
        g_v_bwd_grad_conc_count = 0;
        g_v_bwd_diffusion_launch_count = 0;
        g_v_bwd_chunk0_count = 0;
        g_v_bwd_i_done_count = 0;
        g_v_bwd_v_done_count = 0;
        g_v_bwd_chunk2_enter_count = 0;
        g_v_bwd_di_write_count = 0;
        g_v_bwd_perc_load_count = 0;
        g_v_bwd_dp_write_count = 0;
        g_v_bwd_im2col_count = 0;
        g_v_bwd_conv_fp16_count = 0;
        g_v_bwd_input_grad_count = 0;
        g_v_bwd_scatter_count = 0;
        g_v_bwd_diff_device_count = 0;
        g_v_post_bwd_barrier_count = 0;
    }
}

#endif

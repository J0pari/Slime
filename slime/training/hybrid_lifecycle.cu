
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
__device__ int g_diag6_progress_count = 0;
__device__ int g_diag7_tdr_warning_count = 0;
__device__ int g_diag12_weight_ptr_count = 0;
__device__ int g_diag13_neigh_read_count = 0;
__device__ int g_diag15_input_bad_count = 0;
__device__ int g_diag16_post_neigh_count = 0;
__device__ int g_diag17_pre_perc_count = 0;
__device__ int g_diag19_perc_w_bad_count = 0;
__device__ int g_diag20_perc_acc_bad_count = 0;
__device__ int g_diag21_perc_bad_count = 0;
__device__ int g_diag22_post_perc_count = 0;
__device__ int g_diag23_pre_inter_count = 0;
__device__ int g_diag25_inter_prog_count = 0;
__device__ int g_diag26_inter_sample_count = 0;
__device__ int g_diag27_pregelu_bad_count = 0;
__device__ int g_diag28_gelu_bad_count = 0;
__device__ int g_diag29_post_inter_count = 0;
__device__ int g_diag30_inter_slow_count = 0;
__device__ int g_diag31_pre_out_count = 0;
__device__ int g_diag32_out_bad_count = 0;
__device__ int g_diag33_gate_bad_count = 0;
__device__ int g_diag34_post_out_count = 0;
__device__ int g_diag35_pre_save_count = 0;
__device__ int g_diag37_perc_save_verify_count = 0;
__device__ int g_diag40_post_save_count = 0;
__device__ int g_diag42_pre_caout_count = 0;
__device__ int g_diag43_caout_bad_count = 0;
__device__ int g_diag44_post_caout_count = 0;
__device__ int g_diag45_timing_count = 0;
__device__ int g_diag46_iter_slow_count = 0;
__device__ int g_diag47_stack_corrupt_count = 0;
__device__ int g_diag48_nancheck_count = 0;
__device__ int g_diag49_thread_prog_count = 0;
__device__ int g_diag50_thread_exit_count = 0;
__device__ int g_diag51_warp_count = 0;
__device__ int g_diag52_block_done_count = 0;
__device__ int g_diag53_sync_count = 0;
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
    size_t dW_elements = (size_t)num_heads * head_dim * head_dim;
    size_t dI_elements = (size_t)num_heads * BACKWARD_CHUNK_SAMPLES * head_dim;
    size_t W_T_elements = (size_t)num_heads * head_dim * head_dim;
    size_t im2col_elements = (size_t)channels * cells * CA_KERNEL_CELL_COUNT;
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

// Load initial batch images during initialization (before first training iteration)
// This must be called before init_batch_prev_concentration_device so channels 11-13 have real data
__device__ void load_initial_batch_images_device(Organism* organism) {
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
    unsigned char* all_images = dataset->samples;
    unsigned char* all_labels = dataset->labels;
    int sample_size = sample_rows * sample_cols * sample_channels;

    float* batch_images_out = training_mode->batch_images;
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

    // Load and interpolate images
    int total_pixels = batch_size * batch_stride;
    for (int work_idx = global_tid; work_idx < total_pixels; work_idx += total_threads) {
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
                int img_idx = src_idx * sample_size + channel_offset;
                float tl = all_images[img_idx + y0 * sample_cols + x0] / 255.0f;
                float tr = all_images[img_idx + y0 * sample_cols + x1] / 255.0f;
                float bl = all_images[img_idx + y1 * sample_cols + x0] / 255.0f;
                float br = all_images[img_idx + y1 * sample_cols + x1] / 255.0f;
                batch_images_out[out_idx_base + c * batch_stride + pixel_idx] = Interpolation::bilinear(tl, tr, bl, br, fx, fy);
            }
        } else {
            int img_idx = src_idx * sample_size;
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
    DEVICE_FATAL_IF(training_mode->batch_images == nullptr, "load_batch_device: batch_images is null");
    DEVICE_FATAL_IF(training_mode->batch_labels == nullptr, "load_batch_device: batch_labels is null");
    DEVICE_FATAL_IF(generation < 0, "load_batch_kernel: generation negative");

    ComponentPool* pool = organism->pool;
    bool has_work = compact_idx < pool->alive_indices_count;
    int batch_size = training_mode->batch_size;
    DEVICE_FATAL_IF(batch_size <= 0 || batch_size > BATCH_SIZE, "load_batch_kernel: batch_size invalid");

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

    // Dataset - always valid, allocated before computation
    Dataset* dataset = organism->current_dataset;
    int dataset_size = dataset->num_samples;
    int sample_rows = dataset->descriptor->sample_rows;
    int sample_cols = dataset->descriptor->sample_cols;
    int sample_channels = dataset->descriptor->channels;
    unsigned char* all_images = dataset->samples;
    unsigned char* all_labels = dataset->labels;

    float* batch_images_out = training_mode->batch_images;
    int* batch_labels_out = training_mode->batch_labels;
    int batch_stride = grid_size * grid_size;
    int sample_size = sample_rows * sample_cols * sample_channels;
    int offset = has_work ? (generation % ((dataset_size + batch_size - 1) / batch_size)) * batch_size : 0;

    // Phase 1: Load labels (only block 0, thread 0 does this for shared output)
    if (has_work && tid == 0 && blockIdx.x == 0) {
        for (int idx = 0; idx < batch_size; idx++) {
            int src_idx = (offset + idx) % dataset_size;
            batch_labels_out[idx] = all_labels[src_idx];
        }
    }

    cg::this_grid().sync();  // SYNC 2: after label loading

    // Phase 2: Load and interpolate images
    if (has_work) {
        int total_pixels = batch_size * batch_stride;
        for (int work_idx = tid; work_idx < total_pixels; work_idx += blockDim.x) {
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
                    int img_idx = src_idx * sample_size + channel_offset;
                    float tl = all_images[img_idx + y0 * sample_cols + x0] / 255.0f;
                    float tr = all_images[img_idx + y0 * sample_cols + x1] / 255.0f;
                    float bl = all_images[img_idx + y1 * sample_cols + x0] / 255.0f;
                    float br = all_images[img_idx + y1 * sample_cols + x1] / 255.0f;
                    batch_images_out[out_idx_base + c * batch_stride + pixel_idx] = Interpolation::bilinear(tl, tr, bl, br, fx, fy);
                }
            } else {
                int img_idx = src_idx * sample_size;
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
    }

    cg::this_grid().sync();  // SYNC 3: after image loading

    // Phase 3: Initialize CA state from previous concentration + inject images
    float* ca_out;
    float* prev_concentration;
    float* batch_images = training_mode->batch_images;

    if (has_work) {
        ca_out = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
        prev_concentration = organism->buffers->batch_prev_concentration + s_wave_offsets.ca_states_offset;
    }

    constexpr int IMG_CHANNEL_START = 11;
    constexpr int IMG_CHANNEL_COUNT = 3;

    // Loop with sync - all blocks iterate same count, work guarded inside
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        if (has_work) {
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
    cg::this_grid().sync();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        cg::this_grid().sync();
    }

    float local_ca_mean;
    if (has_work) {
        local_ca_mean = sdata[0] / (float)local_cells;
        if (tid == 0) sdata[0] = local_ca_mean;
    }
    cg::this_grid().sync();
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

    if (s_use_gradients) {
        cg::this_grid().sync();
        if (has_work && tid == 0 && blockIdx.x == 0) printf("V:GRAD_ENTER\n");

        if (has_work) {
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
        cg::this_grid().sync();

        // Trace recording and CA setup - work guarded
        if (has_work) {
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

            DEVICE_FATAL_IF(organism->batch_ca_states_pool == nullptr, "CA_FWD: batch_ca_states_pool is null");
            DEVICE_FATAL_IF(organism->buffers->batched_ca_output == nullptr, "CA_FWD: batched_ca_output is null");
            DEVICE_FATAL_IF(s_wave_offsets.ca_states_offset < 0, "CA_FWD: ca_states_offset negative");
            DEVICE_FATAL_IF(s_wave_offsets.ca_output_offset < 0, "CA_FWD: ca_output_offset negative");
            float* ca_input = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
            float* ca_output = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;

            int total_cells = batch_size * num_heads * cells_per_grid;
            DEVICE_FATAL_IF(total_cells <= 0, "CA_LOOP: total_cells <= 0");
            DEVICE_FATAL_IF(ca_input == nullptr, "CA_LOOP: ca_input is null");
            DEVICE_FATAL_IF(ca_output == nullptr, "CA_LOOP: ca_output is null");
            DEVICE_FATAL_IF(ca_state == nullptr, "CA_LOOP: ca_state is null");
            DEVICE_FATAL_IF(ca_state->perception_weights == nullptr, "CA_LOOP: perception_weights is null");
            DEVICE_FATAL_IF(ca_state->interaction_weights == nullptr, "CA_LOOP: interaction_weights is null");
            DEVICE_FATAL_IF(ca_state->value_weights == nullptr, "CA_LOOP: value_weights is null");
            DEVICE_FATAL_IF(perception_saved == nullptr, "CA_LOOP: perception_saved is null");
            DEVICE_FATAL_IF(interaction_saved == nullptr, "CA_LOOP: interaction_saved is null");
            DEVICE_FATAL_IF(pre_gelu_saved == nullptr, "CA_LOOP: pre_gelu_saved is null");
            int my_act_size = batch_size * num_heads * cells_per_grid * head_dim;
            int my_max_abs_idx = s_wave_offsets.activations_offset + my_act_size - 1;

            if (total_cells <= 0 && tid == 0) printf("E_INIT b%d total_cells=%d\n", blockIdx.x, total_cells);
            if (total_cells > 1000000 && tid == 0) printf("E_INIT b%d total_cells=%d huge\n", blockIdx.x, total_cells);
            if (blockDim.x <= 0 && tid == 0) printf("E_INIT b%d blockDim.x=%d\n", blockIdx.x, (int)blockDim.x);
            if (grid_size <= 0 && tid == 0) printf("E_INIT b%d grid_size=%d\n", blockIdx.x, grid_size);
            if (channels <= 0 && tid == 0) printf("E_INIT b%d channels=%d\n", blockIdx.x, channels);
            if (head_dim <= 0 && tid == 0) printf("E_INIT b%d head_dim=%d\n", blockIdx.x, head_dim);
            if (num_heads <= 0 && tid == 0) printf("E_INIT b%d num_heads=%d\n", blockIdx.x, num_heads);

            // Stack canary to detect overflow
            volatile float stack_canary_start = 3.14159265f;

            int expected_iters = (total_cells + blockDim.x - 1) / blockDim.x;
            long long kernel_start_clock = clock64();

            if (tid == 0) {
                int count = atomicAdd(&g_blocks_ca_fwd, 1) + 1;
                int blocks_with_work = min(pool->alive_indices_count - wave_start, (int)gridDim.x);
                if (count == blocks_with_work) {
                    printf("DIAG1_INIT blocks_work=%d hdim=%d heads=%d cells=%d batch=%d grid=%d chan=%d\n",
                           blocks_with_work, head_dim, num_heads, cells_per_grid, batch_size, grid_size, channels);
                }
            }

            PROVENANCE_FATAL_IF(ca_input == nullptr, "DIAG2: ca_input null");
            PROVENANCE_FATAL_IF(ca_output == nullptr, "DIAG2: ca_output null");
            PROVENANCE_FATAL_IF(ca_state->perception_weights == nullptr, "DIAG2: perception_weights null");
            PROVENANCE_FATAL_IF(ca_state->interaction_weights == nullptr, "DIAG2: interaction_weights null");
            PROVENANCE_FATAL_IF(ca_state->value_weights == nullptr, "DIAG2: value_weights null");
            PROVENANCE_FATAL_IF(perception_saved == nullptr, "DIAG2: perception_saved null");
            PROVENANCE_FATAL_IF(interaction_saved == nullptr, "DIAG2: interaction_saved null");
            PROVENANCE_FATAL_IF(pre_gelu_saved == nullptr, "DIAG2: pre_gelu_saved null");

            PROVENANCE_FATAL_IF(head_dim <= 0 || head_dim > 128, "DIAG3: head_dim insane");
            PROVENANCE_FATAL_IF(num_heads <= 0 || num_heads > 64, "DIAG3: num_heads insane");
            PROVENANCE_FATAL_IF(channels <= 0 || channels > 64, "DIAG3: channels insane");
            PROVENANCE_FATAL_IF(grid_size <= 0 || grid_size > 256, "DIAG3: grid_size insane");
            PROVENANCE_FATAL_IF(batch_size <= 0 || batch_size > 128, "DIAG3: batch_size insane");
            PROVENANCE_FATAL_IF(cells_per_grid <= 0 || cells_per_grid > 65536, "DIAG3: cells_per_grid insane");
            PROVENANCE_FATAL_IF(total_cells <= 0 || total_cells > 10000000, "DIAG3: total_cells insane");

            // DIAG 4: Per-thread initial state - count threads entering
            if (tid == 0) printf("DIAG_PRE_SYNC4 block=%d\n", blockIdx.x);
            __shared__ int s_diag4_threads;
            if (tid == 0) s_diag4_threads = 0;
            __syncthreads();
            if (tid == 0) printf("DIAG_POST_SYNC4a block=%d\n", blockIdx.x);
            atomicAdd(&s_diag4_threads, 1);
            __syncthreads();
            if (tid == 0) printf("DIAG_POST_SYNC4b block=%d\n", blockIdx.x);
            if (tid == 0) {
                int total_threads = atomicAdd(&g_blocks_entered, s_diag4_threads);
                if (blockIdx.x == gridDim.x - 1) {
                    printf("DIAG4_THREADS total=%d blocks=%d\n", total_threads + s_diag4_threads, gridDim.x);
                }
            }

            int loop_iter = 0;
            int max_iter_safety = expected_iters + 100;
            long long prev_iter_end = kernel_start_clock;
            int last_completed_iter = -1;
            int stuck_detection_counter = 0;

            // Track per-warp completion for divergence detection
            __shared__ int s_warp_last_iter[8]; // 256 threads = 8 warps
            if (tid < 8) s_warp_last_iter[tid] = -1;
            __syncthreads();

            // CA forward work loop
            if (tid == 0) printf("DIAG_CA_LOOP_START block=%d total_cells=%d\n", blockIdx.x, total_cells);
            for (int work_idx = tid; work_idx < total_cells; work_idx += blockDim.x, loop_iter++) {

            if (tid == 0 && loop_iter == 0) printf("DIAG_LOOP_ITER0 block=%d\n", blockIdx.x);

            // DIAG 5: Infinite loop protection - count occurrences
            if (loop_iter > max_iter_safety) {
                atomicAdd(&g_blocks_flow, 1);  // reuse counter for infinite loop count
                break;
            }

            long long iter_start = clock64();
            long long iter_since_start_ms = (iter_start - kernel_start_clock) / 1000000;

            // DIAG 6: Progress tracking - count all thread iterations at 5-iter checkpoints
            if (loop_iter % 5 == 0) {
                atomicAdd(&g_diag6_progress_count, 1);
            }

            // DIAG 7: TDR warning threshold - count all threads exceeding 2s from start
            if (iter_since_start_ms > 2000) {
                atomicAdd(&g_diag7_tdr_warning_count, 1);
            }

            // DIAG 8: Verify loop variables not corrupted
            PROVENANCE_FATAL_IF(work_idx < 0, "DIAG8: work_idx negative");
            PROVENANCE_FATAL_IF(work_idx >= total_cells + blockDim.x, "DIAG8: work_idx overflow");
            PROVENANCE_FATAL_IF(loop_iter < 0, "DIAG8: loop_iter negative");

            // DIAG 9: Verify thread index not corrupted
            int check_tid = threadIdx.x;
            int check_blk = blockIdx.x;
            PROVENANCE_FATAL_IF(check_tid != tid, "DIAG9: tid corrupted");
            PROVENANCE_FATAL_IF(check_blk < 0 || check_blk >= gridDim.x, "DIAG9: blockIdx corrupted");

                // Index calculations
                int heads_times_cells = num_heads * cells_per_grid;
                int batch_id = work_idx / heads_times_cells;
                int remainder = work_idx % heads_times_cells;
                int head_id = remainder / cells_per_grid;
                int cell_idx = remainder % cells_per_grid;
                int cell_y = cell_idx / grid_size;
                int cell_x = cell_idx % grid_size;

                // DIAG 10: Validate all computed indices
                PROVENANCE_FATAL_IF(batch_id < 0 || batch_id >= batch_size, "DIAG10: batch_id OOB");
                PROVENANCE_FATAL_IF(head_id < 0 || head_id >= num_heads, "DIAG10: head_id OOB");
                PROVENANCE_FATAL_IF(cell_idx < 0 || cell_idx >= cells_per_grid, "DIAG10: cell_idx OOB");
                PROVENANCE_FATAL_IF(cell_x < 0 || cell_x >= grid_size, "DIAG10: cell_x OOB");
                PROVENANCE_FATAL_IF(cell_y < 0 || cell_y >= grid_size, "DIAG10: cell_y OOB");

                // Weight pointer calculations
                int perc_w_offset = head_id * channels * head_dim;
                int inter_w_offset = head_id * head_dim * head_dim;
                int val_w_offset = head_id * head_dim * channels;

                // DIAG 11: Validate weight offsets
                int max_perc_w = num_heads * channels * head_dim;
                int max_inter_w = num_heads * head_dim * head_dim;
                int max_val_w = num_heads * head_dim * channels;
                PROVENANCE_FATAL_IF(perc_w_offset < 0 || perc_w_offset >= max_perc_w, "DIAG11: perc_w_offset OOB");
                PROVENANCE_FATAL_IF(inter_w_offset < 0 || inter_w_offset >= max_inter_w, "DIAG11: inter_w_offset OOB");
                PROVENANCE_FATAL_IF(val_w_offset < 0 || val_w_offset >= max_val_w, "DIAG11: val_w_offset OOB");

                half* perc_w = &ca_state->perception_weights[perc_w_offset];
                half* inter_w = &ca_state->interaction_weights[inter_w_offset];
                half* val_w = &ca_state->value_weights[val_w_offset];

                // DIAG 12: Weight pointer calculations done - count all threads
                atomicAdd(&g_diag12_weight_ptr_count, 1);

                float neighborhood[3][3][CHANNELS];

                // Stack canary check mid-allocation
                volatile float stack_canary_mid1 = 2.71828f;

                // ============ NEIGHBORHOOD READ ============
                long long t_neigh_start = clock64();

                // DIAG 13: Before neighborhood read - count all threads
                atomicAdd(&g_diag13_neigh_read_count, 1);

                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = min(max(cell_x + dx, 0), grid_size - 1);
                        int ny = min(max(cell_y + dy, 0), grid_size - 1);
                        int state_idx = batch_id * cells_per_grid * channels +
                                       ny * grid_size * channels +
                                       nx * channels;

                        // DIAG 14: Validate neighborhood read index
                        int max_ca_input_idx = batch_size * cells_per_grid * channels;
                        PROVENANCE_FATAL_IF(state_idx < 0 || state_idx + channels > max_ca_input_idx,
                                           "DIAG14: ca_input read OOB");

                        for (int c = 0; c < channels; c++) {
                            float val = ca_input[state_idx + c];
                            neighborhood[dy + 1][dx + 1][c] = val;

                            // DIAG 15: Check for NaN/Inf in input - count all occurrences
                            if (isnan(val) || isinf(val)) {
                                atomicAdd(&g_diag15_input_bad_count, 1);
                            }
                        }
                    }
                }

                long long t_neigh_end = clock64();

                // DIAG 16: After neighborhood read - count all threads
                atomicAdd(&g_diag16_post_neigh_count, 1);
                if (tid == 0 && loop_iter == 0) printf("DIAG_POST_NEIGH block=%d\n", blockIdx.x);

                // ============ PERCEPTION COMPUTATION ============
                float perception[HEAD_DIM];

                // Stack canary check
                volatile float stack_canary_mid2 = 1.41421f;

                long long t_perc_start = clock64();

                // DIAG 17: Before perception loop - count all threads
                atomicAdd(&g_diag17_pre_perc_count, 1);

                for (int h = 0; h < head_dim; h++) {
                    // DIAG 18: Check perception loop variable
                    PROVENANCE_FATAL_IF(h < 0 || h >= HEAD_DIM, "DIAG18: perception h OOB");

                    float acc = 0.0f;
                    for (int dy = 0; dy < 3; dy++) {
                        for (int dx = 0; dx < 3; dx++) {
                            for (int c = 0; c < channels; c++) {
                                int w_idx = c * head_dim + h;
                                PROVENANCE_FATAL_IF(w_idx < 0 || w_idx >= channels * head_dim,
                                                   "DIAG18: perc_w index OOB");

                                float neigh_val = neighborhood[dy][dx][c];
                                half w_half = perc_w[w_idx];
                                float w_float = __half2float(w_half);

                                // DIAG 19: Check half conversion - count all bad weights
                                if (isnan(w_float) || isinf(w_float)) {
                                    atomicAdd(&g_diag19_perc_w_bad_count, 1);
                                }

                                acc += neigh_val * w_float;
                            }
                        }
                    }

                    // DIAG 20: Check accumulator before activation - count all bad
                    if (isnan(acc) || isinf(acc)) {
                        atomicAdd(&g_diag20_perc_acc_bad_count, 1);
                    }

                    perception[h] = activation_relu(acc);

                    // DIAG 21: Check perception value after activation - count all bad
                    if (isnan(perception[h]) || isinf(perception[h])) {
                        atomicAdd(&g_diag21_perc_bad_count, 1);
                    }
                }

                long long t_perc_end = clock64();

                // DIAG 22: After perception - count all threads
                atomicAdd(&g_diag22_post_perc_count, 1);
                if (tid == 0 && loop_iter == 0) printf("DIAG_POST_PERC block=%d\n", blockIdx.x);

                // ============ INTERACTION COMPUTATION (THE BIG ONE - hdim^2 MACs) ============
                float interaction[HEAD_DIM];
                float pre_gelu_vals[HEAD_DIM];
                float interaction_sum = 0.0f;

                // Stack canary
                volatile float stack_canary_mid3 = 1.73205f;

                long long t_inter_start = clock64();

                // DIAG 23: Before interaction - count all threads
                atomicAdd(&g_diag23_pre_inter_count, 1);
                if (tid == 0 && loop_iter == 0) printf("DIAG_PRE_INTER block=%d\n", blockIdx.x);

                // DIAG 24: Detailed interaction loop tracking for hdim=64
                int inter_checkpoint = head_dim / 4; // Report every 25%

                for (int h = 0; h < head_dim; h++) {
                    PROVENANCE_FATAL_IF(h < 0 || h >= HEAD_DIM, "DIAG24: interaction h OOB");

                    // DIAG 25: Track progress through interaction for large hdim - count checkpoints
                    if (head_dim >= 32 && inter_checkpoint > 0 && h % inter_checkpoint == 0) {
                        atomicAdd(&g_diag25_inter_prog_count, 1);
                    }

                    float acc = 0.0f;
                    for (int j = 0; j < head_dim; j++) {
                        PROVENANCE_FATAL_IF(j < 0 || j >= HEAD_DIM, "DIAG24: interaction j OOB");

                        int w_idx = j * head_dim + h;
                        PROVENANCE_FATAL_IF(w_idx < 0 || w_idx >= head_dim * head_dim,
                                           "DIAG24: inter_w index OOB");

                        float p_val = perception[j];
                        half w_half = inter_w[w_idx];
                        float w_float = __half2float(w_half);

                        // DIAG 26: Check values in inner loop - count first samples
                        if (h == 0 && j == 0 && loop_iter == 0) {
                            atomicAdd(&g_diag26_inter_sample_count, 1);
                        }

                        acc += p_val * w_float;
                    }

                    pre_gelu_vals[h] = acc;

                    // DIAG 27: Check pre-gelu value - count all bad
                    if (isnan(acc) || isinf(acc)) {
                        atomicAdd(&g_diag27_pregelu_bad_count, 1);
                    }

                    float gelu = activation_gelu(acc);

                    // DIAG 28: Check gelu result - count all bad
                    if (isnan(gelu) || isinf(gelu)) {
                        atomicAdd(&g_diag28_gelu_bad_count, 1);
                    }

                    interaction[h] = gelu;
                    interaction_sum += fabsf(gelu);
                }

                long long t_inter_end = clock64();
                long long inter_time_us = (t_inter_end - t_inter_start) / 1000;

                // DIAG 29: After interaction - count all threads
                atomicAdd(&g_diag29_post_inter_count, 1);
                if (tid == 0 && loop_iter == 0) printf("DIAG_POST_INTER block=%d\n", blockIdx.x);

                // DIAG 30: Warn if interaction taking too long - count all slow threads
                if (inter_time_us > 100000) { // > 100ms
                    atomicAdd(&g_diag30_inter_slow_count, 1);
                }

                // ============ OUTPUT COMPUTATION ============
                float output[CHANNELS];

                long long t_out_start = clock64();

                // DIAG 31: Before output - count all threads
                atomicAdd(&g_diag31_pre_out_count, 1);
                if (tid == 0 && loop_iter == 0) printf("DIAG_PRE_OUTPUT block=%d\n", blockIdx.x);

                for (int c = 0; c < channels; c++) {
                    PROVENANCE_FATAL_IF(c < 0 || c >= CHANNELS, "DIAG31: output c OOB");

                    float acc = 0.0f;
                    for (int h = 0; h < head_dim; h++) {
                        int w_idx = h * channels + c;
                        PROVENANCE_FATAL_IF(w_idx < 0 || w_idx >= head_dim * channels,
                                           "DIAG31: val_w index OOB");

                        float i_val = interaction[h];
                        half w_half = val_w[w_idx];
                        float w_float = __half2float(w_half);
                        acc += i_val * w_float;
                    }
                    output[c] = acc;

                    // DIAG 32: Check output value - count all bad
                    if (isnan(output[c]) || isinf(output[c])) {
                        atomicAdd(&g_diag32_out_bad_count, 1);
                    }
                }
                if (tid == 0 && loop_iter == 0) printf("DIAG_PRE_GATE block=%d\n", blockIdx.x);

                // DIAG 33: Gate computation
                float gate_input = interaction_sum / (float)head_dim - compute_ca_gate_center(s_task_accuracy);
                float gate = activation_sigmoid(gate_input);
                if (tid == 0 && loop_iter == 0) printf("DIAG_POST_GATE block=%d gate=%f\n", blockIdx.x, gate);

                // DIAG 33: Check gate value - count all bad
                if (isnan(gate) || isinf(gate)) {
                    atomicAdd(&g_diag33_gate_bad_count, 1);
                }

                long long t_out_end = clock64();

                // DIAG 34: After output - count all threads
                atomicAdd(&g_diag34_post_out_count, 1);
                if (tid == 0 && loop_iter == 0) printf("DIAG_PRE_SAVE block=%d\n", blockIdx.x);

                // ============ SAVE ACTIVATIONS ============
                long long t_save_start = clock64();

                int saved_base = batch_id * num_heads * cells_per_grid * head_dim +
                                head_id * cells_per_grid * head_dim +
                                cell_idx * head_dim;
                int max_saved_idx = saved_base + head_dim - 1;
                int expected_saved_size = batch_size * num_heads * cells_per_grid * head_dim;

                // DIAG 35: Validate save indices
                PROVENANCE_FATAL_IF(saved_base < 0, "DIAG35: saved_base negative");
                PROVENANCE_FATAL_IF(max_saved_idx >= expected_saved_size, "DIAG35: saved OOB");

                // DIAG 35: Pre-save - count all threads
                atomicAdd(&g_diag35_pre_save_count, 1);

                // DIAG 36: Save perception
                for (int h = 0; h < head_dim; h++) {
                    int idx = saved_base + h;
                    PROVENANCE_FATAL_IF(idx < 0 || idx >= expected_saved_size, "DIAG36: perc_save idx OOB");
                    perception_saved[idx] = perception[h];
                }

                // DIAG 37: Verify perception save - count all first-iteration saves
                if (loop_iter == 0) {
                    atomicAdd(&g_diag37_perc_save_verify_count, 1);
                }
                if (tid == 0 && loop_iter == 0) printf("DIAG_POST_PERC_SAVE block=%d\n", blockIdx.x);

                // DIAG 38: Save interaction
                for (int h = 0; h < head_dim; h++) {
                    int idx = saved_base + h;
                    PROVENANCE_FATAL_IF(idx < 0 || idx >= expected_saved_size, "DIAG38: inter_save idx OOB");
                    interaction_saved[idx] = interaction[h];
                }
                if (tid == 0 && loop_iter == 0) printf("DIAG_POST_INTER_SAVE block=%d\n", blockIdx.x);

                // DIAG 39: Save pre_gelu
                for (int h = 0; h < head_dim; h++) {
                    int idx = saved_base + h;
                    PROVENANCE_FATAL_IF(idx < 0 || idx >= expected_saved_size, "DIAG39: pregelu_save idx OOB");
                    pre_gelu_saved[idx] = pre_gelu_vals[h];
                }
                if (tid == 0 && loop_iter == 0) printf("DIAG_POST_PREGELU_SAVE block=%d\n", blockIdx.x);

                long long t_save_end = clock64();

                // DIAG 40: After saves - count all threads
                if (tid == 0 && loop_iter == 0) printf("DIAG_PRE_ATOMIC40 block=%d\n", blockIdx.x);
                atomicAdd(&g_diag40_post_save_count, 1);
                if (tid == 0 && loop_iter == 0) printf("DIAG_POST_ATOMIC40 block=%d\n", blockIdx.x);

                // ============ WRITE CA OUTPUT ============
                if (tid == 0 && loop_iter == 0) printf("DIAG_PRE_CAOUT block=%d\n", blockIdx.x);
                long long t_caout_start = clock64();

                int out_idx = batch_id * num_heads * cells_per_grid * channels +
                             head_id * cells_per_grid * channels +
                             cell_y * grid_size * channels +
                             cell_x * channels;
                int max_out_idx = out_idx + channels - 1;
                int expected_out_size = batch_size * num_heads * cells_per_grid * channels;

                // DIAG 41: Validate output indices
                PROVENANCE_FATAL_IF(out_idx < 0, "DIAG41: out_idx negative");
                PROVENANCE_FATAL_IF(max_out_idx >= expected_out_size, "DIAG41: ca_out OOB");

                // DIAG 42: Before ca_output write
                atomicAdd(&g_diag42_pre_caout_count, 1);

                for (int c = 0; c < channels; c++) {
                    int idx = out_idx + c;
                    PROVENANCE_FATAL_IF(idx < 0 || idx >= expected_out_size, "DIAG42: ca_out write OOB");

                    float input_val = neighborhood[1][1][c];
                    float out_val = input_val * (1.0f - gate) + output[c] * gate;

                    // DIAG 43: Check output value
                    if (isnan(out_val) || isinf(out_val)) {
                        atomicAdd(&g_diag43_caout_bad_count, 1);
                    }

                    ca_output[idx] = out_val;
                }

                long long t_caout_end = clock64();

                // ============ ITERATION COMPLETE ============
                long long iter_end = clock64();
                long long iter_total_us = (iter_end - iter_start) / 1000;
                long long total_ms = (iter_end - kernel_start_clock) / 1000000;

                // DIAG 45: Timing breakdown - count timing checkpoints
                if (loop_iter % 10 == 0) {
                    atomicAdd(&g_diag45_timing_count, 1);
                }

                // DIAG 46: Warn if iteration took too long - count all slow threads
                if (iter_total_us > 500000) { // > 500ms
                    atomicAdd(&g_diag46_iter_slow_count, 1);
                }

                // DIAG 47: Stack canary check - count all corruptions
                if (stack_canary_start != 3.14159265f || stack_canary_mid1 != 2.71828f ||
                    stack_canary_mid2 != 1.41421f || stack_canary_mid3 != 1.73205f) {
                    atomicAdd(&g_diag47_stack_corrupt_count, 1);
                }

                // DIAG 48: Check for NaN/Inf summary - count all bad
                int any_nan = isnan(perception[0]) || isnan(interaction[0]) || isnan(output[0]) || isnan(gate);
                int any_inf = isinf(perception[0]) || isinf(interaction[0]) || isinf(output[0]) || isinf(gate);
                if (any_nan || any_inf) {
                    atomicAdd(&g_diag48_nancheck_count, 1);
                }

                // Update warp progress tracking (only lane 0 of each warp writes)
                if ((tid % 32) == 0) {
                    s_warp_last_iter[tid / 32] = loop_iter;
                }

                last_completed_iter = loop_iter;
                prev_iter_end = iter_end;

                // DIAG 49: Per-thread completion tracking - count progress checkpoints
                if (loop_iter % 50 == 0) {
                    atomicAdd(&g_diag49_thread_prog_count, 1);
                }
            }

            // ============ LOOP EXIT ============
            long long loop_end = clock64();
            long long total_loop_ms = (loop_end - kernel_start_clock) / 1000000;

            // DIAG 50: Loop exit for all threads
            atomicAdd(&g_diag50_thread_exit_count, 1);

            // DIAG 51: Warp-level completion - count all warps
            if (tid % 32 == 0) {
                atomicAdd(&g_diag51_warp_count, 1);
            }

            // DIAG 52: Block-level summary - count all blocks
            if (tid == 0) {
                atomicAdd(&g_diag52_block_done_count, 1);
            }

            // Final stack canary check
            if (stack_canary_start != 3.14159265f && tid == 0) {
                printf("E_STACK b%d canary=%.8f\n", blockIdx.x, stack_canary_start);
            }
        }

        // DIAG 53: Sync after loop - count threads reaching sync (only blocks with work did actual work)
        if (tid == 0) printf("DIAG_PRE_GRIDSYNC block=%d has_work=%d\n", blockIdx.x, has_work ? 1 : 0);
        if (has_work) atomicAdd(&g_diag53_sync_count, 1);
        cg::this_grid().sync();
        if (tid == 0) printf("DIAG_POST_GRIDSYNC block=%d\n", blockIdx.x);

        // V:CA_FWD_DONE - count blocks completing CA forward
        if (has_work && tid == 0) {
            atomicAdd(&g_blocks_ca_fwd, 1);
        }

        if (tid == 0) printf("DIAG_PRE_GRIDSYNC2 block=%d has_work=%d\n", blockIdx.x, has_work ? 1 : 0);
        cg::this_grid().sync();
        if (tid == 0) printf("DIAG_POST_GRIDSYNC2 block=%d\n", blockIdx.x);
        if (has_work) {
            DEVICE_FATAL_IF(s_error_flag, "hybrid_lifecycle: error flag set after CA forward");
        }
        if (tid == 0) printf("DIAG_POST_ERRCHECK block=%d\n", blockIdx.x);

        // Flow computation - work guarded, syncs outside
        if (has_work) {
            int total_cells = arch.grid_size * arch.grid_size;
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads = arch.num_heads;

            float* wave_ca_output = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            float* wave_affinity = organism->buffers->batch_affinity_reduced + s_wave_offsets.affinity_offset;

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
        }
        if (tid == 0) printf("DIAG_PRE_GRIDSYNC3 block=%d has_work=%d\n", blockIdx.x, has_work ? 1 : 0);
        cg::this_grid().sync();
        if (tid == 0) printf("DIAG_POST_GRIDSYNC3 block=%d\n", blockIdx.x);

        if (has_work) {
            int total_cells = arch.grid_size * arch.grid_size;
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            float flow_beta_A = entry->flow_beta_A;
            float flow_n = entry->flow_n;
            float flow_alpha_min = entry->flow_alpha_min;
            float flow_alpha_max = entry->flow_alpha_max;
            float flow_sharpness = entry->flow_sharpness;

            float* wave_affinity = organism->buffers->batch_affinity_reduced + s_wave_offsets.affinity_offset;
            float* wave_flow = organism->buffers->batch_flow_field + s_wave_offsets.flow_offset;
            float* wave_ca_pool = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;

            int total_affinity_work = batch_size * total_cells;
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

            float* wave_reint = organism->buffers->batch_reintegration_buffer + s_wave_offsets.ca_states_offset;
            int buffer_size = batch_size * total_cells * channels;
            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                wave_reint[idx] = 0.0f;
            }
        }
        cg::this_grid().sync();
        if (tid == 0 && blockIdx.x == 0) printf("DIAG_FLOW1\n");

        if (has_work) {
            int total_cells = arch.grid_size * arch.grid_size;
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            float flow_dt = entry->flow_resource_dt;

            float* wave_flow = organism->buffers->batch_flow_field + s_wave_offsets.flow_offset;
            float* wave_reint = organism->buffers->batch_reintegration_buffer + s_wave_offsets.ca_states_offset;
            float* wave_ca_pool = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;

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
        }
        cg::this_grid().sync();
        if (tid == 0 && blockIdx.x == 0) printf("DIAG_FLOW2\n");

        if (has_work) {
            int total_cells = arch.grid_size * arch.grid_size;
            int batch_size = training_mode->batch_size;
            int channels = arch.channels;
            int buffer_size = batch_size * total_cells * channels;

            float* wave_reint = organism->buffers->batch_reintegration_buffer + s_wave_offsets.ca_states_offset;
            float* wave_ca_pool = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;

            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                wave_ca_pool[idx] = wave_reint[idx];
            }
        }
        cg::this_grid().sync();
        if (tid == 0 && blockIdx.x == 0) printf("DIAG_FLOW3\n");

        if (has_work) {
            int total_cells = arch.grid_size * arch.grid_size;
            int batch_size = training_mode->batch_size;
            int channels = arch.channels;
            int buffer_size = batch_size * total_cells * channels;

            float* wave_ca_pool = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;
            float* wave_prev_conc = organism->buffers->batch_prev_concentration + s_wave_offsets.ca_states_offset;

            for (int idx = tid; idx < buffer_size; idx += blockDim.x) {
                wave_prev_conc[idx] = wave_ca_pool[idx];
            }
        }
        cg::this_grid().sync();

        if (has_work && tid == 0) {
            atomicAdd(&g_v_flow_done_count, 1);
        }

        if (tid == 0) printf("DIAG_PRE_POOL_BARRIER b%d\n", blockIdx.x);
        grid_barrier(gridDim.x);
        if (tid == 0 && blockIdx.x == 0) printf("DIAG_POST_POOL_BARRIER\n");

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
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

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
            if (has_work && tid == 0) {
                BehavioralDimensions dims;
                dims.derive_from_genome();
                int task_dim = dims.task_dim;

                DEVICE_FATAL_IF(entry->diresa_task_weights->input_dim != num_features,
                    "hybrid_lifecycle: diresa_task_weights->input_dim mismatch with num_features (entry %d)", entry_idx);
            }
        }
        cg::this_grid().sync();

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
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
        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr && has_work) {
            loss_out = &organism->gradient_loss_pool[entry_idx];
            if (tid == 0) {
                *loss_out = 0.0f;
            }
        }
        cg::this_grid().sync();

        // Softmax/loss - work guarded
        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr && has_work) {
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

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr && has_work && tid == 0) {
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
        if (tid == 0 && blockIdx.x == 0 && audit != nullptr && training_mode->batch_images != nullptr) {
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
        cg::this_grid().sync();

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {

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
            float* fc_weights_grad = organism->fc_weights_grad;
            float* fc_bias_grad = organism->fc_bias_grad;
            float* pooling_weights = training_mode->classifier[entry_idx].pooling_weights;
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
        }
        cg::this_grid().sync();

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
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
            float* fc_weights_grad = organism->fc_weights_grad;
            float* fc_bias_grad = organism->fc_bias_grad;
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

        if (training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
            int batch_size = training_mode->batch_size;
            int grid_size = arch.grid_size;
            int channels = arch.channels;
            int num_heads_local = arch.num_heads;
            int spatial_size = grid_size * grid_size;
            float* ca_out = organism->buffers->batched_ca_output + s_wave_offsets.ca_output_offset;
            float* features_grad = organism->features_grad + entry_idx * batch_size * num_features;
            float* pooling_weights = training_mode->classifier[entry_idx].pooling_weights;
            float* pooling_weights_grad = organism->pooling_weights_grad;

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
            int ws_dW_value_stride;
            int ws_W_T_value_stride;
            int ws_dW_interaction_stride;
            int ws_W_T_interaction_stride;
            int chunk_ws_a_stride;
            int chunk_ws_b_stride;

            if (has_work) {
                trace_buffer = &ca_state->trace;
                if (tid == 0 && trace_buffer->traces != nullptr) {
                    int trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
                    if (trace_idx < trace_buffer->capacity) {
                        record_warp_metrics(&trace_buffer->traces[trace_idx], blockIdx.x);
                    }
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
                DEVICE_FATAL_IF(ca_state->value_weights == nullptr, "BWD: value_weights is null");
                DEVICE_FATAL_IF(ca_state->interaction_weights == nullptr, "BWD: interaction_weights is null");
                DEVICE_FATAL_IF(ca_state->perception_weights == nullptr, "BWD: perception_weights is null");
                DEVICE_FATAL_IF(ca_state->tape.grad_buffer == nullptr, "BWD: tape.grad_buffer is null");
                DEVICE_FATAL_IF(param_map == nullptr, "BWD: param_map is null");
                DEVICE_FATAL_IF(param_map->value_start == nullptr, "BWD: param_map->value_start is null");
                DEVICE_FATAL_IF(param_map->interaction_start == nullptr, "BWD: param_map->interaction_start is null");
                DEVICE_FATAL_IF(param_map->perception_start == nullptr, "BWD: param_map->perception_start is null");
                DEVICE_FATAL_IF(organism->batch_ca_states_pool == nullptr, "BWD: batch_ca_states_pool is null");
                perception_saved = ca_state->perception_saved + s_wave_offsets.activations_offset;
                interaction_saved = ca_state->interaction_saved + s_wave_offsets.activations_offset;
                pre_gelu_saved = ca_state->pre_gelu_saved + s_wave_offsets.activations_offset;

                I_head_stride = num_cells * arch.head_dim;
                I_batch_stride = arch.num_heads * I_head_stride;
                V_head_stride = num_cells * arch.channels;
                V_batch_stride = arch.num_heads * V_head_stride;
                ws_dW_value_stride = arch.head_dim * arch.channels;
                ws_W_T_value_stride = arch.channels * arch.head_dim;
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

            {
            // Zero dW - work guarded
            if (has_work) {
                int total_dW = arch.num_heads * arch.head_dim * arch.channels;
                DEVICE_FATAL_IF(total_dW <= 0, "BWD: total_dW overflow or zero");
                for (int idx = tid; idx < total_dW; idx += blockDim.x) {
                    ws_dW[idx] = 0.0f;
                }
            }
            cg::this_grid().sync();

            // All blocks iterate broadcast num_chunks, only blocks with work do actual work
            for (int chunk_idx = 0; chunk_idx < bwd_num_chunks; chunk_idx++) {
                // Per-block chunk computation - only valid for blocks with work
                int chunk_start = chunk_idx * BACKWARD_CHUNK_SAMPLES;
                int chunk_samples = has_work ? min(BACKWARD_CHUNK_SAMPLES, total_samples - chunk_start) : 0;
                int chunk_samples_aligned = (chunk_samples / WMMA_TILE_DIM) * WMMA_TILE_DIM;
                bool chunk_has_work = has_work && (chunk_samples_aligned > 0);

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_chunk0_count, 1);

                if (chunk_has_work) {
                    int total_I = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_src_I = arch.num_heads * training_mode->batch_size * num_cells * arch.head_dim;
                    int max_dst_I = arch.num_heads * chunk_ws_a_stride;
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
                        PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_src_I, "BWD interaction_saved OOB");
                        PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_dst_I, "BWD ws_fp16_a OOB");
                        float src_val = interaction_saved[src_idx];
                        PROVENANCE_FATAL_IF(!isfinite(src_val), "BWD interaction_saved NaN/Inf");
                        ws_fp16_a[dst_idx] = __float2half(src_val);
                    }
                }
                cg::this_grid().sync();
                if (tid == 0 && blockIdx.x == 0 && chunk_idx == 0) printf("V:bwd_I_load\n");

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_i_done_count, 1);

                if (chunk_has_work) {
                    int total_V = arch.num_heads * chunk_samples_aligned * arch.channels;
                    int max_src_V = arch.num_heads * training_mode->batch_size * num_cells * arch.channels;
                    int max_dst_V = arch.num_heads * chunk_ws_b_stride;
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
                        PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_src_V, "BWD ca_output_grad OOB");
                        PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_dst_V, "BWD ws_fp16_b OOB");
                        float src_val = ca_output_grad[src_idx];
                        PROVENANCE_FATAL_IF(!isfinite(src_val), "BWD ca_output_grad NaN/Inf");
                        ws_fp16_b[dst_idx] = __float2half(src_val);
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_v_done_count, 1);

                if (chunk_has_work) {
                    PROVENANCE_ASSERT_INITIALIZED_INT(arch.head_dim, "arch.head_dim");
                    PROVENANCE_ASSERT_INITIALIZED_INT(arch.channels, "arch.channels");
                    PROVENANCE_ASSERT_INITIALIZED_INT(arch.num_heads, "arch.num_heads");
                    PROVENANCE_ASSERT_INITIALIZED_INT(chunk_ws_a_stride, "chunk_ws_a_stride");
                    PROVENANCE_ASSERT_INITIALIZED_INT(chunk_ws_b_stride, "chunk_ws_b_stride");
                    PROVENANCE_ASSERT_INITIALIZED_INT(ws_dW_value_stride, "ws_dW_value_stride");
                    PROVENANCE_ASSERT_INITIALIZED_INT(chunk_samples_aligned, "chunk_samples_aligned");
                    PROVENANCE_FATAL_IF(ws_fp16_a == nullptr, "ws_fp16_a null");
                    PROVENANCE_FATAL_IF(ws_fp16_b == nullptr, "ws_fp16_b null");
                    PROVENANCE_FATAL_IF(ws_dW == nullptr, "ws_dW null");

                    int tiles_M = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int tiles_N = (arch.channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = tiles_M * tiles_N * arch.num_heads;
                    int max_ws_fp16_a = arch.num_heads * chunk_ws_a_stride;
                    int max_ws_fp16_b = arch.num_heads * chunk_ws_b_stride;
                    int max_ws_dW = arch.num_heads * ws_dW_value_stride;

                    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                        int head_id = tile_idx / (tiles_M * tiles_N);
                        int tile_flat = tile_idx % (tiles_M * tiles_N);
                        int warpM = tile_flat / tiles_N;
                        int warpN = tile_flat % tiles_N;

                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        PROVENANCE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "head_id OOB");

                        if (tile_row < arch.head_dim && tile_col < arch.channels) {
                            int A_base_offset = head_id * chunk_ws_a_stride;
                            int B_base_offset = head_id * chunk_ws_b_stride;
                            int C_base_offset = head_id * ws_dW_value_stride;
                            PROVENANCE_FATAL_IF(A_base_offset < 0 || A_base_offset >= max_ws_fp16_a, "A_base OOB");
                            PROVENANCE_FATAL_IF(B_base_offset < 0 || B_base_offset >= max_ws_fp16_b, "B_base OOB");
                            PROVENANCE_FATAL_IF(C_base_offset < 0 || C_base_offset >= max_ws_dW, "C_base OOB");

                            const half* A_head = ws_fp16_a + A_base_offset;
                            const half* B_head = ws_fp16_b + B_base_offset;
                            float* C_head = ws_dW + C_base_offset;

                            int C_tile_offset = tile_row * arch.channels + tile_col;
                            int C_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.channels + (tile_col + WMMA_TILE_DIM - 1);
                            PROVENANCE_FATAL_IF(C_tile_offset < 0 || C_tile_max >= ws_dW_value_stride, "C_tile OOB");

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

                            nvcuda::wmma::load_matrix_sync(c_frag, C_head + C_tile_offset, arch.channels, nvcuda::wmma::mem_row_major);

                            for (int k_tile = 0; k_tile < chunk_samples_aligned; k_tile += WMMA_TILE_DIM) {
                                int A_tile_offset = k_tile * arch.head_dim + tile_row;
                                int A_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_row + WMMA_TILE_DIM - 1);
                                int B_tile_offset = k_tile * arch.channels + tile_col;
                                int B_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.channels + (tile_col + WMMA_TILE_DIM - 1);
                                PROVENANCE_FATAL_IF(A_tile_offset < 0 || A_tile_max >= chunk_ws_a_stride, "A_tile OOB");
                                PROVENANCE_FATAL_IF(B_tile_offset < 0 || B_tile_max >= chunk_ws_b_stride, "B_tile OOB");
                                PROVENANCE_FATAL_IF(A_base_offset + A_tile_max >= max_ws_fp16_a, "A_abs OOB");
                                PROVENANCE_FATAL_IF(B_base_offset + B_tile_max >= max_ws_fp16_b, "B_abs OOB");
                                nvcuda::wmma::load_matrix_sync(a_frag, A_head + A_tile_offset, arch.head_dim);
                                nvcuda::wmma::load_matrix_sync(b_frag, B_head + B_tile_offset, arch.channels);
                                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + C_tile_offset, c_frag, arch.channels, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                cg::this_grid().sync();
            }

            // V:BWD_VALUE_GRAD_COPY - count blocks
            if (has_work && tid == 0) atomicAdd(&g_v_bwd_value_grad_count, 1);
            if (has_work) {
                int total_grads = arch.num_heads * arch.head_dim * arch.channels;
                int max_ws_dW = arch.num_heads * ws_dW_value_stride;
                int max_grad_buffer = ca_state->tape.capacity;
                PROVENANCE_FATAL_IF(total_grads <= 0, "BWD value grad: total_grads overflow");
                for (int idx = tid; idx < total_grads; idx += blockDim.x) {
                    int head_id = idx / (arch.head_dim * arch.channels);
                    int local_idx = idx % (arch.head_dim * arch.channels);
                    int src_idx = head_id * ws_dW_value_stride + local_idx;
                    int dst_idx = param_map->value_start[head_id] + local_idx;
                    PROVENANCE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "BWD value grad: head_id OOB");
                    PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_ws_dW, "BWD value grad: src_idx OOB");
                    PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_grad_buffer, "BWD value grad: dst_idx OOB");
                    float src_val = ws_dW[src_idx];
                    PROVENANCE_FATAL_IF(!isfinite(src_val), "BWD value grad: ws_dW NaN/Inf");
                    ca_state->tape.grad_buffer[dst_idx] = src_val;
                }
            }
            cg::this_grid().sync();

            // V:BWD_VALUE_TRANSPOSE - work guarded, sync outside
            if (has_work) {
                int t_tiles_x = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int t_tiles_y = (arch.channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                int total_tile_elements = t_tiles_x * t_tiles_y * arch.num_heads * WMMA_TILE_DIM * WMMA_TILE_DIM;
                int max_W_per_head = arch.head_dim * arch.channels;
                int max_W_T_per_head = ws_W_T_value_stride;

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
                        int W_src_idx = h * arch.channels + c;
                        int W_T_dst_idx = c * arch.head_dim + h;
                        PROVENANCE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "BWD val transpose: head_id OOB");
                        PROVENANCE_FATAL_IF(W_src_idx < 0 || W_src_idx >= max_W_per_head, "BWD val transpose: src OOB");
                        PROVENANCE_FATAL_IF(W_T_dst_idx < 0 || W_T_dst_idx >= max_W_T_per_head, "BWD val transpose: dst OOB");
                        const half* W_head = ca_state->value_weights + head_id * arch.head_dim * arch.channels;
                        half* W_T_head = ws_W_T + head_id * ws_W_T_value_stride;
                        half w_val = W_head[W_src_idx];
                        PROVENANCE_FATAL_IF(!isfinite(__half2float(w_val)), "BWD val transpose: W NaN/Inf");
                        W_T_head[W_T_dst_idx] = w_val;
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
                        half* W_T_head = ws_W_T + arch.num_heads * ws_W_T_value_stride + head_id * ws_W_T_interaction_stride;
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
                float* ws_dW_inter = ws_dW + arch.num_heads * ws_dW_value_stride;
                for (int idx = tid; idx < total_zero; idx += blockDim.x) {
                    ws_dW_inter[idx] = 0.0f;
                }
            }
            cg::this_grid().sync();
            if (tid == 0 && blockIdx.x == 0) printf("V:bwd_zero_dW\n");

            // Per-block variables for second chunk loop - uninitialized, only valid when has_work
            int chunk_ws_dI_stride;
            int chunk_ws_dpregelu_stride;
            int chunk_ws_pooled_stride;
            float* ws_dW_interaction;
            float* ws_dW_perception;
            half* ws_W_T_interaction;
            float* d_ca_input;

            if (has_work) {
                chunk_ws_dI_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
                chunk_ws_dpregelu_stride = BACKWARD_CHUNK_SAMPLES * arch.head_dim;
                chunk_ws_pooled_stride = BACKWARD_CHUNK_SAMPLES * arch.channels;
                ws_dW_interaction = ws_dW + arch.num_heads * ws_dW_value_stride;
                ws_dW_perception = ws_dW_interaction + arch.num_heads * ws_dW_interaction_stride;
                ws_W_T_interaction = ws_W_T + arch.num_heads * ws_W_T_value_stride;

                d_ca_input = organism->batch_ca_input_grads ?
                    organism->batch_ca_input_grads + s_wave_offsets.ca_states_offset : nullptr;
                if (d_ca_input != nullptr) {
                    int max_d_ca = total_samples * arch.channels;
                    PROVENANCE_FATAL_IF(max_d_ca <= 0, "BWD d_ca_input: max_d_ca overflow");
                    for (int idx = tid; idx < max_d_ca; idx += blockDim.x) {
                        d_ca_input[idx] = 0.0f;
                    }
                }
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

                if (chunk_has_work) {
                    int total_V = arch.num_heads * chunk_samples_aligned * arch.channels;
                    int max_src_V = arch.num_heads * training_mode->batch_size * num_cells * arch.channels;
                    int max_dst_V = arch.num_heads * chunk_ws_b_stride;
                    PROVENANCE_FATAL_IF(total_V <= 0, "BWD chunk2 V: total overflow");
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
                        PROVENANCE_FATAL_IF(src_idx < 0 || src_idx >= max_src_V, "BWD chunk2: src_idx OOB");
                        PROVENANCE_FATAL_IF(dst_idx < 0 || dst_idx >= max_dst_V, "BWD chunk2: dst_idx OOB");
                        float val = ca_output_grad[src_idx];
                        PROVENANCE_FATAL_IF(!isfinite(val), "BWD chunk2: ca_output_grad NaN/Inf");
                        ws_fp16_b[dst_idx] = __float2half(val);
                    }
                }
                cg::this_grid().sync();


                if (chunk_has_work) {
                    int tiles_M = (chunk_samples_aligned + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int tiles_N = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = tiles_M * tiles_N * arch.num_heads;
                    int max_ws_fp16_b_dI = arch.num_heads * chunk_ws_b_stride;
                    int max_ws_W_T = arch.num_heads * ws_W_T_value_stride;
                    int max_ws_dI_total = arch.num_heads * chunk_ws_dI_stride;

                    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
                        int head_id = tile_idx / (tiles_M * tiles_N);
                        int tile_flat = tile_idx % (tiles_M * tiles_N);
                        int warpM = tile_flat / tiles_N;
                        int warpN = tile_flat % tiles_N;
                        int tile_row = warpM * WMMA_TILE_DIM;
                        int tile_col = warpN * WMMA_TILE_DIM;

                        DEVICE_FATAL_IF(head_id < 0 || head_id >= arch.num_heads, "WMMA dI: head_id OOB");

                        if (tile_row < chunk_samples_aligned && tile_col < arch.head_dim) {
                            int A_base_offset = head_id * chunk_ws_b_stride;
                            int B_base_offset = head_id * ws_W_T_value_stride;
                            int C_base_offset = head_id * chunk_ws_dI_stride;
                            DEVICE_FATAL_IF(A_base_offset < 0 || A_base_offset >= max_ws_fp16_b_dI, "WMMA dI: A_base OOB");
                            DEVICE_FATAL_IF(B_base_offset < 0 || B_base_offset >= max_ws_W_T, "WMMA dI: B_base OOB");
                            DEVICE_FATAL_IF(C_base_offset < 0 || C_base_offset >= max_ws_dI_total, "WMMA dI: C_base OOB");

                            const half* A_head = ws_fp16_b + A_base_offset;
                            const half* B_head = ws_W_T + B_base_offset;
                            float* C_head = ws_dI + C_base_offset;

                            int C_tile_offset = tile_row * arch.head_dim + tile_col;
                            int C_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                            DEVICE_FATAL_IF(C_tile_offset < 0 || C_tile_max >= chunk_ws_dI_stride, "WMMA dI: C_tile OOB");

                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> a_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
                            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                            nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                            for (int k_tile = 0; k_tile < arch.channels; k_tile += WMMA_TILE_DIM) {
                                if (k_tile + WMMA_TILE_DIM <= arch.channels) {
                                    int A_tile_offset = tile_row * arch.channels + k_tile;
                                    int A_tile_max = (tile_row + WMMA_TILE_DIM - 1) * arch.channels + (k_tile + WMMA_TILE_DIM - 1);
                                    int B_tile_offset = k_tile * arch.head_dim + tile_col;
                                    int B_tile_max = (k_tile + WMMA_TILE_DIM - 1) * arch.head_dim + (tile_col + WMMA_TILE_DIM - 1);
                                    DEVICE_FATAL_IF(A_tile_offset < 0 || A_tile_max >= chunk_ws_b_stride, "WMMA dI: A_tile OOB");
                                    DEVICE_FATAL_IF(B_tile_offset < 0 || B_tile_max >= ws_W_T_value_stride, "WMMA dI: B_tile OOB");
                                    nvcuda::wmma::load_matrix_sync(a_frag, A_head + A_tile_offset, arch.channels);
                                    nvcuda::wmma::load_matrix_sync(b_frag, B_head + B_tile_offset, arch.head_dim);
                                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                                }
                            }
                            nvcuda::wmma::store_matrix_sync(C_head + C_tile_offset, c_frag, arch.head_dim, nvcuda::wmma::mem_row_major);
                        }
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_di_write_count, 1);
                if (chunk_has_work) {
                    int total_elem = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    int max_ws_dI = arch.num_heads * chunk_ws_dI_stride;
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

                        int ws_idx = head_id * chunk_ws_dI_stride + sample_in_chunk * arch.head_dim + dim_idx;
                        int out_idx = head_id * I_head_stride + batch_id * I_batch_stride + cell_id * arch.head_dim + dim_idx;
                        int saved_idx = batch_id * I_batch_stride + head_id * I_head_stride + cell_id * arch.head_dim + dim_idx;
                        int dpregelu_idx = head_id * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + dim_idx;

                        PROVENANCE_FATAL_IF(ws_idx < 0 || ws_idx >= max_ws_dI, "BWD dI: ws_idx OOB");
                        PROVENANCE_FATAL_IF(out_idx < 0 || out_idx >= max_out, "BWD dI: out_idx OOB");
                        PROVENANCE_FATAL_IF(saved_idx < 0 || saved_idx >= max_saved, "BWD dI: saved_idx OOB");
                        PROVENANCE_FATAL_IF(dpregelu_idx < 0 || dpregelu_idx >= max_dpregelu, "BWD dI: dpregelu_idx OOB");

                        float dL_dI_val = ws_dI[ws_idx];
                        PROVENANCE_FATAL_IF(!isfinite(dL_dI_val), "BWD dI: ws_dI NaN/Inf");
                        dL_dinteraction[out_idx] = dL_dI_val;
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
                    int max_input = training_mode->batch_size * arch.grid_size * arch.grid_size * arch.channels;
                    int max_im2col = chunk_samples_aligned * arch.channels;
                    for (int idx = tid; idx < chunk_samples_aligned; idx += blockDim.x) {
                        int global_sample = chunk_start + idx;
                        int batch_id = global_sample / num_cells;
                        int cell_idx = global_sample % num_cells;
                        int cell_y = cell_idx / arch.grid_size;
                        int cell_x = cell_idx % arch.grid_size;

                        PROVENANCE_FATAL_IF(batch_id < 0 || batch_id >= training_mode->batch_size, "BWD im2col: batch_id OOB");
                        PROVENANCE_FATAL_IF(cell_y < 0 || cell_y >= arch.grid_size, "BWD im2col: cell_y OOB");
                        PROVENANCE_FATAL_IF(cell_x < 0 || cell_x >= arch.grid_size, "BWD im2col: cell_x OOB");

                        const float* input_batch = organism->batch_ca_states_pool + s_wave_offsets.ca_states_offset;

                        for (int c = 0; c < arch.channels; c++) {
                            float sum = 0.0f;
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    int ny = max(0, min(arch.grid_size - 1, cell_y + dy));
                                    int nx = max(0, min(arch.grid_size - 1, cell_x + dx));
                                    int input_idx = batch_id * arch.grid_size * arch.grid_size * arch.channels +
                                                   ny * arch.grid_size * arch.channels + nx * arch.channels + c;
                                    PROVENANCE_FATAL_IF(input_idx < 0 || input_idx >= max_input, "BWD im2col: input_idx OOB");
                                    float in_val = input_batch[input_idx];
                                    PROVENANCE_FATAL_IF(!isfinite(in_val), "BWD im2col: input NaN/Inf");
                                    sum += in_val;
                                }
                            }
                            int im2col_idx = idx * arch.channels + c;
                            PROVENANCE_FATAL_IF(im2col_idx < 0 || im2col_idx >= max_im2col, "BWD im2col: im2col_idx OOB");
                            PROVENANCE_FATAL_IF(!isfinite(sum), "BWD im2col: sum NaN/Inf");
                            ws_im2col[im2col_idx] = sum;
                        }
                    }
                }
                cg::this_grid().sync();

                if (chunk_has_work && tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_conv_fp16_count, 1);
                if (chunk_has_work) {
                    int total_conv = chunk_samples_aligned * arch.channels;
                    for (int idx = tid; idx < total_conv; idx += blockDim.x) {
                        float val = ws_im2col[idx];
                        PROVENANCE_FATAL_IF(!isfinite(val), "BWD conv fp16: ws_im2col NaN/Inf");
                        ws_fp16_a[idx] = __float2half(val);
                    }
                }
                if (chunk_has_work) {
                    int total_D = arch.num_heads * chunk_samples_aligned * arch.head_dim;
                    for (int idx = tid; idx < total_D; idx += blockDim.x) {
                        float val = ws_dpregelu[idx];
                        PROVENANCE_FATAL_IF(!isfinite(val), "BWD D fp16: ws_dpregelu NaN/Inf");
                        ws_fp16_b[idx] = __float2half(val);
                    }
                }
                cg::this_grid().sync();


                if (chunk_has_work) {
                    int ws_dW_perception_stride = arch.channels * arch.head_dim;
                    int dW_tiles_c = (arch.channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int dW_tiles_h = (arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
                    int total_tiles = dW_tiles_c * dW_tiles_h * arch.num_heads;
                    int max_ws_fp16_a_perc = chunk_samples_aligned * arch.channels;
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
                            int B_base_offset = head_id * chunk_B_stride;
                            int C_base_offset = head_id * ws_dW_perception_stride;
                            DEVICE_FATAL_IF(B_base_offset < 0 || B_base_offset >= max_ws_fp16_b_perc, "WMMA perc dW: B_base OOB");
                            DEVICE_FATAL_IF(C_base_offset < 0 || C_base_offset >= max_ws_dW_perc, "WMMA perc dW: C_base OOB");

                            const half* A_ptr = ws_fp16_a;
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
                                DEVICE_FATAL_IF(A_tile_offset < 0 || A_tile_max >= max_ws_fp16_a_perc, "WMMA perc dW: A_tile OOB");
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
                if (chunk_has_work && d_ca_input != nullptr) {
                    int weights_per_head = arch.channels * arch.head_dim;
                    int max_dpregelu = arch.num_heads * chunk_ws_dpregelu_stride;
                    int max_im2col = chunk_samples_aligned * arch.channels;
                    for (int idx = tid; idx < chunk_samples_aligned * arch.channels; idx += blockDim.x) {
                        int sample_in_chunk = idx / arch.channels;
                        int channel_idx = idx % arch.channels;

                        float d_input_accum = 0.0f;
                        for (int h = 0; h < arch.num_heads; h++) {
                            int head_start = param_map->perception_start[h];
                            int head_max = head_start + weights_per_head;
                            for (int hd = 0; hd < arch.head_dim; hd++) {
                                int local_idx = channel_idx * arch.head_dim + hd;
                                int W_idx = head_start + local_idx;
                                int dprerelu_idx = h * chunk_ws_dpregelu_stride + sample_in_chunk * arch.head_dim + hd;
                                PROVENANCE_FATAL_IF(local_idx < 0 || local_idx >= weights_per_head, "BWD input: local_idx OOB");
                                PROVENANCE_FATAL_IF(W_idx < head_start || W_idx >= head_max, "BWD input: W_idx OOB");
                                PROVENANCE_FATAL_IF(dprerelu_idx < 0 || dprerelu_idx >= max_dpregelu, "BWD input: dprerelu_idx OOB");
                                float W_val = __half2float(ca_state->perception_weights[W_idx]);
                                PROVENANCE_FATAL_IF(!isfinite(W_val), "BWD input: W_val NaN/Inf");
                                float dprerelu_val = ws_dpregelu[dprerelu_idx];
                                PROVENANCE_FATAL_IF(!isfinite(dprerelu_val), "BWD input: dprerelu_val NaN/Inf");
                                d_input_accum += W_val * dprerelu_val;
                            }
                        }
                        PROVENANCE_FATAL_IF(idx < 0 || idx >= max_im2col, "BWD input: im2col idx OOB");
                        PROVENANCE_FATAL_IF(!isfinite(d_input_accum), "BWD input: d_input_accum NaN/Inf");
                        ws_im2col[idx] = d_input_accum;
                    }
                }
                cg::this_grid().sync();
                if (tid == 0 && blockIdx.x == 0 && chunk_idx == 0) printf("V:bwd_input_grad\n");

                if (chunk_has_work && d_ca_input != nullptr) {
                    if (tid == 0 && chunk_idx == 0) atomicAdd(&g_v_bwd_scatter_count, 1);
                    PROVENANCE_FATAL_IF(d_ca_input == nullptr, "d_ca_input null");
                    PROVENANCE_FATAL_IF(ws_im2col == nullptr, "ws_im2col null");
                    PROVENANCE_ASSERT_INITIALIZED_INT(training_mode->batch_size, "batch_size");
                    PROVENANCE_ASSERT_INITIALIZED_INT(num_cells, "num_cells");
                    PROVENANCE_ASSERT_INITIALIZED_INT(arch.grid_size, "grid_size");
                    PROVENANCE_ASSERT_INITIALIZED_INT(chunk_samples_aligned, "chunk_samples_aligned");
                    PROVENANCE_ASSERT_INITIALIZED_INT(arch.channels, "arch.channels");
                    PROVENANCE_FATAL_IF(chunk_samples_aligned <= 0, "chunk_samples_aligned <= 0");
                    PROVENANCE_FATAL_IF(arch.channels <= 0, "arch.channels <= 0");
                    int scatter_loop_bound = chunk_samples_aligned * arch.channels;
                    PROVENANCE_FATAL_IF(scatter_loop_bound <= 0, "scatter_loop_bound overflow");
                    PROVENANCE_FATAL_IF(scatter_loop_bound > 1000000, "scatter_loop_bound huge");
                    int max_d_ca = training_mode->batch_size * num_cells * arch.channels;
                    PROVENANCE_FATAL_IF(max_d_ca <= 0, "max_d_ca <= 0");
                    for (int idx = tid; idx < scatter_loop_bound; idx += blockDim.x) {
                        int sample_in_chunk = idx / arch.channels;
                        int channel_idx = idx % arch.channels;
                        int global_sample = chunk_start + sample_in_chunk;
                        int batch_id = global_sample / num_cells;
                        int cell_idx = global_sample % num_cells;
                        int cell_y = cell_idx / arch.grid_size;
                        int cell_x = cell_idx % arch.grid_size;
                        float d_pooled_val = ws_im2col[idx];
                        PROVENANCE_FATAL_IF(!isfinite(d_pooled_val), "BWD scatter: d_pooled_val NaN/Inf");

                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                int ny = cell_y + dy;
                                int nx = cell_x + dx;
                                if (ny >= 0 && ny < arch.grid_size && nx >= 0 && nx < arch.grid_size) {
                                    int out_cell_idx = ny * arch.grid_size + nx;
                                    int out_idx = batch_id * num_cells * arch.channels + out_cell_idx * arch.channels + channel_idx;
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

            if (has_work && tid == 0) atomicAdd(&g_v_bwd_diffusion_launch_count, 1);
            if (has_work && d_ca_input != nullptr && tid == 0) {
                dim3 diff_grid((arch.grid_size + 15) / 16, (arch.grid_size + 15) / 16);
                dim3 diff_block(16, 16);

                DEVICE_FATAL_IF(organism->chemical_field == nullptr, "BWD diff: chemical_field is null");
                DEVICE_FATAL_IF(organism->chemical_field->concentration == nullptr, "BWD diff: concentration is null");
                DEVICE_FATAL_IF(organism->chemical_field->laplacian == nullptr, "BWD diff: laplacian is null");
                DEVICE_FATAL_IF(entry->gradients == nullptr, "BWD diff: entry->gradients is null");
                DEVICE_FATAL_IF(primary_genome == nullptr, "BWD diff: primary_genome is null");

                float ctx_metabolic = entry->fitness.value;
                float ctx_stress = entry->hunger.value;
                float ctx_morphogen = organism->chemical_field->cached_mean;
                float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
                float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
                float ctx_learning = training_mode->learning_rate;
                float ctx_performance = entry->task_accuracy.value;

                atomicAdd(&g_v_bwd_diff_device_count, 1);
                diffusion_reaction_backward_device(organism);
            }
            cg::this_grid().sync();
            if (has_work && tid == 0 && blockIdx.x == 0) printf("V:HYB_diffusion_bwd_done\n");

        }

        // Flow Lenia backward - work guarded, syncs outside
        {
                // Shared memory for reductions - all blocks participate in syncs
                __shared__ float s_d_beta_A;
                __shared__ float s_d_n;
                if (tid == 0) {
                    s_d_beta_A = 0.0f;
                    s_d_n = 0.0f;
                }
                cg::this_grid().sync();

                // Per-block variables - uninitialized, only valid when has_work
                int batch_size;
                int grid_size;
                int channels;
                int num_heads;
                int total_cells;
                float flow_beta_A;
                float flow_n;
                float flow_alpha_min;
                float flow_alpha_max;
                float flow_sharpness;
                float flow_dt;
                float local_d_beta_A;
                float local_d_n;

                if (has_work) {
                    batch_size = training_mode->batch_size;
                    grid_size = arch.grid_size;
                    channels = arch.channels;
                    num_heads = arch.num_heads;
                    total_cells = grid_size * grid_size;

                    flow_beta_A = entry->flow_beta_A;
                    flow_n = entry->flow_n;
                    flow_alpha_min = entry->flow_alpha_min;
                    flow_alpha_max = entry->flow_alpha_max;
                    flow_sharpness = entry->flow_sharpness;
                    flow_dt = entry->flow_resource_dt;

                    local_d_beta_A = 0.0f;
                    local_d_n = 0.0f;

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
                }
                cg::this_grid().sync();
                if (tid == 0 && blockIdx.x == 0) printf("V:flow_bwd_reduce\n");

                if (has_work && tid == 0) {
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
                cg::this_grid().sync();
            }
        }

        // Effective rank computation - all blocks iterate NUM_HEADS times for sync alignment
        {
            int num_heads = has_work ? arch.num_heads : 0;
            int perception_per_head = has_work ? arch.channels * arch.head_dim : 0;
            int interaction_per_head = has_work ? arch.head_dim * arch.head_dim : 0;
            int value_per_head = has_work ? arch.head_dim * arch.channels : 0;
            int params_per_head = perception_per_head + interaction_per_head + value_per_head;
            float* grad_buf = has_work ? ca_state->tape.grad_buffer : nullptr;

            __shared__ float head_grad_sq[NUM_HEADS];
            __shared__ float warp_sums_eff[32];

            for (int h = 0; h < NUM_HEADS; h++) {
                float local_sq = 0.0f;
                if (has_work && h < num_heads) {
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
                    int lane = tid % warpSize;
                    int warp_id = tid / warpSize;
                    if (lane == 0) warp_sums_eff[warp_id] = local_sq;
                }
                cg::this_grid().sync();

                if (has_work && h < num_heads) {
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
                }
                cg::this_grid().sync();
            }

            if (has_work && tid == 0) {
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
                measured_value_set_computed(&entry->effective_rank, clamped_rank, organism->generation, entry->genome_hash);
            }
            cg::this_grid().sync();
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

            adam_update_perception_device(organism);
            adam_update_interaction_device(organism);
            adam_update_value_device(organism);
        }
        cg::this_grid().sync();

        if (entry_idx == 0 && training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
            adam_update_pooling_device(organism);
            adam_update_fc_weights_device(organism);
            adam_update_fc_bias_device(organism);
        }
        cg::this_grid().sync();

        if (entry_idx == 0 && training_mode->batch_images != nullptr && training_mode->classifier != nullptr) {
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
    int alive_ct = pool->alive_indices_count;

    if (entry_idx == 0) {
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
            measured_value_set_computed(&ent->generalization_gap, gen_gap_val, generation, ent->genome_hash);

            DEVICE_FATAL_IF(ent->cycles_elapsed == 0, "cycles_elapsed is 0 - no execution data");
            DEVICE_FATAL_IF(ent->total_branches == 0, "total_branches is 0 - no branch data");

            float ipc = (float)ent->inst_executed / (float)ent->cycles_elapsed;
            float tensor_util = (float)ent->tensor_core_cycles / (float)ent->cycles_elapsed;
            float branch_efficiency = (float)(ent->total_branches - ent->divergent_branches) / (float)ent->total_branches;
            float hw_eff_val = ipc * tensor_util * branch_efficiency;
            measured_value_set_computed(&ent->hardware_efficiency, hw_eff_val, generation, ent->genome_hash);

            DEVICE_FATAL_IF(generation == 0, "coherence requires previous generation");
            float prev_acc = organism->fitness_history[((generation - 1) % 2) * POOL_CAPACITY_MAX + eid];
            float coherence_val = ent->task_accuracy.value - prev_acc;
            measured_value_set_computed(&ent->coherence, coherence_val, generation, ent->genome_hash);

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
    }
    cg::this_grid().sync();

    {
        float local_acc = 0.0f, local_gap = 0.0f, local_hw = 0.0f, local_fit = 0.0f;
        if (entry_idx == 0) {
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

        if (tid == 0 && entry_idx == 0) {
            organism->telemetry->population_metrics.total_accuracy = total_acc;
            organism->telemetry->population_metrics.total_generalization_gap = total_gap;
            organism->telemetry->population_metrics.total_hardware_efficiency = total_hw;
            organism->telemetry->population_metrics.total_fitness = total_fit;
        }
    }
    cg::this_grid().sync();

    if (entry_idx == 0 && generation > 0) {
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
    cg::this_grid().sync();
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


    if (entry_idx == 0 && generation % EMBEDDING_UPDATE_FREQ == 0) {
        if (tid == 0) {
            *organism->buffers->behavioral_reconstruction_error = 0.0f;
        }
    }
    cg::this_grid().sync();

    if (entry_idx == 0 && generation % EMBEDDING_UPDATE_FREQ == 0) {
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
    }
    cg::this_grid().sync();

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
    cg::this_grid().sync();

    // Only block 0 does telemetry and audit - prevents N blocks each printing
    if (tid == 0 && blockIdx.x == 0 && audit != nullptr && training_mode->batch_images != nullptr) {
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
        printf("DIAG6_PROGRESS total=%d\n", g_diag6_progress_count);
        printf("DIAG7_TDR_WARN total=%d\n", g_diag7_tdr_warning_count);
        printf("DIAG12_WEIGHT_PTR total=%d\n", g_diag12_weight_ptr_count);
        printf("DIAG13_NEIGH_READ total=%d\n", g_diag13_neigh_read_count);
        printf("DIAG15_INPUT_BAD total=%d\n", g_diag15_input_bad_count);
        printf("DIAG16_POST_NEIGH total=%d\n", g_diag16_post_neigh_count);
        printf("DIAG17_PRE_PERC total=%d\n", g_diag17_pre_perc_count);
        printf("DIAG19_PERC_W_BAD total=%d\n", g_diag19_perc_w_bad_count);
        printf("DIAG20_PERC_ACC_BAD total=%d\n", g_diag20_perc_acc_bad_count);
        printf("DIAG21_PERC_BAD total=%d\n", g_diag21_perc_bad_count);
        printf("DIAG22_POST_PERC total=%d\n", g_diag22_post_perc_count);
        printf("DIAG23_PRE_INTER total=%d\n", g_diag23_pre_inter_count);
        printf("DIAG25_INTER_PROG total=%d\n", g_diag25_inter_prog_count);
        printf("DIAG26_INTER_SAMPLE total=%d\n", g_diag26_inter_sample_count);
        printf("DIAG27_PREGELU_BAD total=%d\n", g_diag27_pregelu_bad_count);
        printf("DIAG28_GELU_BAD total=%d\n", g_diag28_gelu_bad_count);
        printf("DIAG29_POST_INTER total=%d\n", g_diag29_post_inter_count);
        printf("DIAG30_INTER_SLOW total=%d\n", g_diag30_inter_slow_count);
        printf("DIAG31_PRE_OUT total=%d\n", g_diag31_pre_out_count);
        printf("DIAG32_OUT_BAD total=%d\n", g_diag32_out_bad_count);
        printf("DIAG33_GATE_BAD total=%d\n", g_diag33_gate_bad_count);
        printf("DIAG34_POST_OUT total=%d\n", g_diag34_post_out_count);
        printf("DIAG35_PRE_SAVE total=%d\n", g_diag35_pre_save_count);
        printf("DIAG37_PERC_SAVE_VERIFY total=%d\n", g_diag37_perc_save_verify_count);
        printf("DIAG40_POST_SAVE total=%d\n", g_diag40_post_save_count);
        printf("DIAG42_PRE_CAOUT total=%d\n", g_diag42_pre_caout_count);
        printf("DIAG43_CAOUT_BAD total=%d\n", g_diag43_caout_bad_count);
        printf("DIAG44_POST_CAOUT total=%d\n", g_diag44_post_caout_count);
        printf("DIAG45_TIMING total=%d\n", g_diag45_timing_count);
        printf("DIAG46_ITER_SLOW total=%d\n", g_diag46_iter_slow_count);
        printf("DIAG47_STACK_CORRUPT total=%d\n", g_diag47_stack_corrupt_count);
        printf("DIAG48_NANCHECK total=%d\n", g_diag48_nancheck_count);
        printf("DIAG49_THREAD_PROG total=%d\n", g_diag49_thread_prog_count);
        printf("DIAG50_THREAD_EXIT total=%d\n", g_diag50_thread_exit_count);
        printf("DIAG51_WARP total=%d\n", g_diag51_warp_count);
        printf("DIAG52_BLOCK_DONE total=%d\n", g_diag52_block_done_count);
        printf("DIAG53_SYNC total=%d\n", g_diag53_sync_count);
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
        g_diag6_progress_count = 0;
        g_diag7_tdr_warning_count = 0;
        g_diag12_weight_ptr_count = 0;
        g_diag13_neigh_read_count = 0;
        g_diag15_input_bad_count = 0;
        g_diag16_post_neigh_count = 0;
        g_diag17_pre_perc_count = 0;
        g_diag19_perc_w_bad_count = 0;
        g_diag20_perc_acc_bad_count = 0;
        g_diag21_perc_bad_count = 0;
        g_diag22_post_perc_count = 0;
        g_diag23_pre_inter_count = 0;
        g_diag25_inter_prog_count = 0;
        g_diag26_inter_sample_count = 0;
        g_diag27_pregelu_bad_count = 0;
        g_diag28_gelu_bad_count = 0;
        g_diag29_post_inter_count = 0;
        g_diag30_inter_slow_count = 0;
        g_diag31_pre_out_count = 0;
        g_diag32_out_bad_count = 0;
        g_diag33_gate_bad_count = 0;
        g_diag34_post_out_count = 0;
        g_diag35_pre_save_count = 0;
        g_diag37_perc_save_verify_count = 0;
        g_diag40_post_save_count = 0;
        g_diag42_pre_caout_count = 0;
        g_diag43_caout_bad_count = 0;
        g_diag44_post_caout_count = 0;
        g_diag45_timing_count = 0;
        g_diag46_iter_slow_count = 0;
        g_diag47_stack_corrupt_count = 0;
        g_diag48_nancheck_count = 0;
        g_diag49_thread_prog_count = 0;
        g_diag50_thread_exit_count = 0;
        g_diag51_warp_count = 0;
        g_diag52_block_done_count = 0;
        g_diag53_sync_count = 0;
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


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

__device__ int g_blocks_entered = 0;
__device__ int g_blocks_grad = 0;
__device__ int g_blocks_ca_fwd = 0;
__device__ int g_blocks_flow = 0;
__device__ int g_blocks_bwd = 0;
__device__ int g_blocks_complete = 0;
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
__device__ int g_v_bwd_chunk0_count = 0;
__device__ int g_v_bwd_i_done_count = 0;
__device__ int g_v_bwd_v_done_count = 0;
__device__ int g_v_bwd_chunk2_enter_count = 0;
__device__ int g_v_bwd_di_write_count = 0;
__device__ int g_v_bwd_perc_load_count = 0;
__device__ int g_v_bwd_dp_write_count = 0;
__device__ int g_v_bwd_im2col_count = 0;
__device__ int g_v_bwd_conv_fp16_count = 0;

__device__ int g_bwd_max_num_chunks = 0;
__device__ int g_bwd_max_num_cells = 0;
__device__ int g_bwd_max_total_samples = 0;
__device__ int g_v_bwd_input_grad_count = 0;
__device__ int g_v_bwd_scatter_count = 0;
__device__ int g_v_post_bwd_barrier_count = 0;

#include "../diagnostics/telemetry_probes.cu"

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
    bool has_work = compact_idx < pool->alive_indices_count && blockIdx.x < WAVE_SIZE;
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
                        ca_out, base_idx, channels_out,
                        organism, batch_stride, spatial_idx,
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

#include "lifecycle_forward.cu"
#include "lifecycle_backward.cu"
#include "lifecycle_metrics.cu"

__device__ void lifecycle_audit_device(Organism* organism) {
    int tid = threadIdx.x;
    if (tid != 0 || blockIdx.x != 0) return;

    HybridTrainingMode* training_mode = organism->training_mode;
    AuditBuffer* audit = organism->audit_buffer;
    if (audit == nullptr || training_mode->batch_samples == nullptr) return;
    if (!organism->telemetry->valid) return;

    ComponentPool* pool = organism->pool;
    int first_entry_idx = pool->alive_indices[0];
    PoolEntry* entry = &pool->entries[first_entry_idx];
    MultiHeadCAState* ca_state = entry->ca_state;

    populate_audit_buffer(
        audit,
        organism->generation,
        organism->gradient_logits_pool,
        training_mode->batch_labels,
        training_mode->batch_samples,
        training_mode->batch_size,
        organism->current_dataset->descriptor->num_classes,
        ca_state->ca_concentration,
        entry->grid_size,
        entry->num_heads,
        entry->head_dim,
        entry->channels,
        gridDim.x,
        training_mode->is_train_batch,
        organism->telemetry,
        pool,
        organism->chemical_field,
        ca_state,
        organism->hardware_geom,
        organism->archive_size
    );
}


#endif

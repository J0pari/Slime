
#ifndef TENSOR_CORE_CA_CU
#define TENSOR_CORE_CA_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include "../utils/genome_params.cuh"
#include "../core/pseudopod.cu"
#include "../memory/pool.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__device__ void convert_fp32_to_fp16_device(Organism* organism) {
    float* fp32_data = organism->tensor_fp32_data;
    half* fp16_data = organism->tensor_fp16_data;
    int M = organism->tensor_M;
    int N = organism->tensor_N;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        fp16_data[idx] = __float2half(fp32_data[idx]);
    }
}

__device__ void convert_fp16_to_fp32_device(Organism* organism) {
    half* fp16_data = organism->tensor_fp16_data;
    float* fp32_data = organism->tensor_fp32_data;
    int M = organism->tensor_M;
    int N = organism->tensor_N;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        fp32_data[idx] = __half2float(fp16_data[idx]);
    }
}

__device__ void tensor_core_matmul_device(Organism* organism) {
    half* A = organism->tensor_A;
    half* B = organism->tensor_B;
    float* C = organism->tensor_C;
    int M = organism->tensor_M;
    int N = organism->tensor_N;
    int K = organism->tensor_K;

    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    const int tile_row = warpM * WMMA_TILE_DIM;
    const int tile_col = warpN * WMMA_TILE_DIM;

    if (tile_row < M && tile_col < N) {
        wmma::fragment<wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        for (int k_tile = 0; k_tile < K; k_tile += WMMA_TILE_DIM) {

            if (k_tile + WMMA_TILE_DIM <= K) {

                wmma::load_matrix_sync(a_frag, A + tile_row * K + k_tile, K);

                wmma::load_matrix_sync(b_frag, B + k_tile * N + tile_col, N);

                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }

        wmma::store_matrix_sync(C + tile_row * N + tile_col, c_frag, N, wmma::mem_row_major);
    }
}

__device__ void relu_device(Organism* organism) {
    float* data = organism->activation_data;
    int size = organism->activation_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = activation_relu(data[idx]);
    }
}

__device__ void gelu_device(Organism* organism) {
    float* data = organism->activation_data;
    int size = organism->activation_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = activation_gelu(x);
    }
}

__device__ void prepare_ca_fp16_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    Architecture arch = Architecture::maxBounds();
    int entry_idx = organism->current_entry_idx;

    DEVICE_FATAL_IF(entry_idx >= pool->capacity, "prepare_ca_fp16_device: entry_idx out of bounds");

    PoolEntry* entry = &pool->entries[entry_idx];
    DEVICE_FATAL_IF(!pool->alive_flags[entry_idx], "prepare_ca_fp16_device: dead entry passed");

    int grid_size = entry->grid_size;
    int num_cells = grid_size * grid_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_cells * arch.channels;

    if (idx < total) {
        MultiHeadCAState* ca_state = entry->ca_state;
        ca_state->fp16_workspace[idx] = __float2half(ca_state->ca_concentration[idx]);
    }
}

__device__ void multi_head_ca_tensor_device(
    PoolEntry* entry,
    int batch_size,
    float* ca_input,
    float* ca_output,
    float task_accuracy
) {
    // Block-local: 256 threads per block, __syncthreads between stages.
    // Each block processes its own entry independently. 8 warps for WMMA.
    int tid = threadIdx.x;
    int block_threads = blockDim.x;

    MultiHeadCAState* ca_state = entry->ca_state;
    int grid_size = entry->grid_size;
    int channels = entry->channels;
    int num_heads = entry->num_heads;
    int head_dim = entry->head_dim;
    int num_cells = grid_size * grid_size;

    float* perception_saved = ca_state->perception_saved;
    float* interaction_saved = ca_state->interaction_saved;
    float* pre_gelu_saved = ca_state->pre_gelu_saved;

    half* perception_weights = ca_state->perception_weights;
    half* interaction_weights = ca_state->interaction_weights;
    half* flow_projection_weights = ca_state->flow_projection_weights;

    half* fp16_workspace = ca_state->fp16_workspace;
    float* fp32_workspace = ca_state->fp32_workspace;

    // Trace recording
    if (threadIdx.x == 0) {
        TraceBuffer* trace_buffer = &ca_state->trace;
        int trace_idx = -1;
        if (trace_buffer->traces != nullptr &&
            trace_buffer->current_idx < trace_buffer->capacity) {
            trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
        }
        if (trace_idx >= 0 && trace_idx < trace_buffer->capacity) {
            ExecutionTrace* t = &trace_buffer->traces[trace_idx];
            record_warp_metrics(t, blockIdx.x);
            record_memory_access(t, (void*)perception_weights, true);
        }
    }

    // ============ STAGE 0: Zero ca_output (transport accumulates via atomicAdd) ============
    int total_output = batch_size * num_heads * num_cells * channels;
    for (int i = tid; i < total_output; i += block_threads) {
        ca_output[i] = 0.0f;
    }
    __syncthreads();

    int warp_id = tid / WARP_SIZE;
    int block_warps = block_threads / WARP_SIZE;

    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        for (int head = 0; head < num_heads; head++) {

            // ============ STAGE 1: Gather 3x3 neighborhood for this head ============
            int input_offset = (batch_id * num_heads + head) * num_cells * channels;
            int total_gather = num_cells * channels;
            for (int i = tid; i < total_gather; i += block_threads) {
                int cell_idx = i / channels;
                int c = i % channels;
                int cell_y = cell_idx / grid_size;
                int cell_x = cell_idx % grid_size;

                float sum = 0.0f;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int ny = min(max(cell_y + dy, 0), grid_size - 1);
                        int nx = min(max(cell_x + dx, 0), grid_size - 1);
                        sum += ca_input[input_offset + ny * grid_size * channels + nx * channels + c];
                    }
                }
                fp16_workspace[i] = __float2half(sum);
            }
            __syncthreads();

            int perc_weight_offset = head * channels * head_dim;
            int inter_weight_offset = head * head_dim * head_dim;
            int flow_weight_offset = head * 2 * head_dim;
            int perception_size = num_cells * head_dim;

            int perc_out_offset = head * num_cells * head_dim;
            int inter_out_offset = num_heads * num_cells * head_dim + head * num_cells * head_dim;

            // ============ STAGE 2: Perception matmul (WMMA) ============
            // [num_cells × channels] × [channels × head_dim] → [num_cells × head_dim]
            int perc_tiles_M = (num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int perc_tiles_N = (head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int perc_total_tiles = perc_tiles_M * perc_tiles_N;

            for (int tile_idx = warp_id; tile_idx < perc_total_tiles; tile_idx += block_warps) {
                int warpM = tile_idx / perc_tiles_N;
                int warpN = tile_idx % perc_tiles_N;
                int tile_row = warpM * WMMA_TILE_DIM;
                int tile_col = warpN * WMMA_TILE_DIM;

                if (tile_row < num_cells && tile_col < head_dim) {
                    wmma::fragment<wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, wmma::row_major> b_frag;
                    wmma::fragment<wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);

                    for (int k = 0; k < channels; k += WMMA_TILE_DIM) {
                        if (k + WMMA_TILE_DIM <= channels) {
                            wmma::load_matrix_sync(a_frag,
                                fp16_workspace + tile_row * channels + k,
                                channels);
                            wmma::load_matrix_sync(b_frag,
                                perception_weights + perc_weight_offset + k * head_dim + tile_col,
                                head_dim);
                            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                    }
                    wmma::store_matrix_sync(
                        fp32_workspace + perc_out_offset + tile_row * head_dim + tile_col,
                        c_frag, head_dim, wmma::mem_row_major);
                }
            }
            __syncthreads();

            // ============ STAGE 3: ReLU + save perception ============
            int saved_base = batch_id * num_heads * num_cells * head_dim +
                             head * num_cells * head_dim;

            for (int i = tid; i < perception_size; i += block_threads) {
                float val = activation_relu(fp32_workspace[perc_out_offset + i]);
                fp32_workspace[perc_out_offset + i] = val;
                perception_saved[saved_base + i] = val;
            }
            __syncthreads();

            // ============ STAGE 4: fp32 → fp16 for interaction input ============
            half* interaction_input = fp16_workspace + num_cells * channels;
            for (int i = tid; i < perception_size; i += block_threads) {
                interaction_input[i] = __float2half(fp32_workspace[perc_out_offset + i]);
            }
            __syncthreads();

            // ============ STAGE 5: Interaction matmul (WMMA) ============
            // [num_cells × head_dim] × [head_dim × head_dim] → [num_cells × head_dim]
            int inter_tiles_M = (num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int inter_tiles_N = (head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM;
            int inter_total_tiles = inter_tiles_M * inter_tiles_N;

            for (int tile_idx = warp_id; tile_idx < inter_total_tiles; tile_idx += block_warps) {
                int warpM = tile_idx / inter_tiles_N;
                int warpN = tile_idx % inter_tiles_N;
                int tile_row = warpM * WMMA_TILE_DIM;
                int tile_col = warpN * WMMA_TILE_DIM;

                if (tile_row < num_cells && tile_col < head_dim) {
                    wmma::fragment<wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, wmma::row_major> b_frag;
                    wmma::fragment<wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);

                    for (int k = 0; k < head_dim; k += WMMA_TILE_DIM) {
                        if (k + WMMA_TILE_DIM <= head_dim) {
                            wmma::load_matrix_sync(a_frag,
                                interaction_input + tile_row * head_dim + k,
                                head_dim);
                            wmma::load_matrix_sync(b_frag,
                                interaction_weights + inter_weight_offset + k * head_dim + tile_col,
                                head_dim);
                            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                    }
                    wmma::store_matrix_sync(
                        fp32_workspace + inter_out_offset + tile_row * head_dim + tile_col,
                        c_frag, head_dim, wmma::mem_row_major);
                }
            }
            __syncthreads();

            // ============ STAGE 6: Save pre_gelu + GELU + save interaction ============
            for (int i = tid; i < perception_size; i += block_threads) {
                float pre_gelu = fp32_workspace[inter_out_offset + i];
                pre_gelu_saved[saved_base + i] = pre_gelu;
                float gelu = activation_gelu(pre_gelu);
                fp32_workspace[inter_out_offset + i] = gelu;
                interaction_saved[saved_base + i] = gelu;
            }
            __syncthreads();

            // ============ STAGE 7: Flow projection + gated transport ============
            for (int cell_idx = tid; cell_idx < num_cells; cell_idx += block_threads) {
                float interaction_local[HEAD_DIM];
                float interaction_sum = 0.0f;
                for (int h = 0; h < head_dim; h++) {
                    float val = fp32_workspace[inter_out_offset + cell_idx * head_dim + h];
                    interaction_local[h] = val;
                    interaction_sum += fabsf(val);
                }

                float2 flow = FlowLeniaOps::project_to_flow(
                    interaction_local, head_dim,
                    &flow_projection_weights[flow_weight_offset]);

                float gate_input = interaction_sum / (float)head_dim - compute_ca_gate_center(task_accuracy);
                float gate = activation_sigmoid(gate_input);

                int batch_cell_offset = batch_id * num_heads * num_cells * channels +
                                        head * num_cells * channels;
                FlowLeniaOps::bilinear_transport_forward(
                    ca_input, cell_idx, flow, gate,
                    entry->flow_resource_dt, grid_size,
                    ca_output, channels, batch_cell_offset);
            }
            __syncthreads();
        }
    }
}

#endif

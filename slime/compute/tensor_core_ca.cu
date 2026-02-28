
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

__device__ void tensor_core_perception_device(Organism* organism) {
    half* neighborhood_fp16 = organism->tensor_neighborhood_fp16;
    MultiHeadCAState* mh_state = organism->multihead_ca_state;
    half* perception_weights = mh_state->perception_weights;
    float* perception_out = organism->tensor_perception_out;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;
    int head_id = organism->current_head_id;

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int num_cells = grid_size * grid_size;

    const int tile_row = (warp_id / ((arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM)) * WMMA_TILE_DIM;
    const int tile_col = (warp_id % ((arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM)) * WMMA_TILE_DIM;

    if (tile_row < num_cells && tile_col < arch.head_dim) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        int weight_offset = head_id * arch.channels * arch.head_dim;

        for (int k = 0; k < arch.channels; k += WMMA_TILE_DIM) {
            if (k + WMMA_TILE_DIM <= arch.channels) {

                wmma::load_matrix_sync(a_frag,
                    neighborhood_fp16 + tile_row * arch.channels + k,
                    arch.channels);

                wmma::load_matrix_sync(b_frag,
                    perception_weights + weight_offset + k * arch.head_dim + tile_col,
                    arch.head_dim);

                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }

        int output_offset = head_id * num_cells * arch.head_dim;
        wmma::store_matrix_sync(
            perception_out + output_offset + tile_row * arch.head_dim + tile_col,
            c_frag,
            arch.head_dim,
            wmma::mem_row_major
        );
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

__device__ void multi_head_ca_tensor_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    int max_grid_size = organism->max_grid_size;
    Architecture arch = Architecture::maxBounds();
    int entry_idx = organism->current_entry_idx;

    int head = blockIdx.y;
    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    bool valid_head = (head < arch.num_heads);

    DEVICE_FATAL_IF(entry_idx >= pool->capacity, "multi_head_ca_tensor_device: entry_idx out of bounds");
    DEVICE_FATAL_IF(!pool->alive_flags[entry_idx], "multi_head_ca_tensor_device: dead entry passed");

    PoolEntry* entry = &pool->entries[entry_idx];
    int grid_size = entry->grid_size;
    int num_cells = grid_size * grid_size;
    MultiHeadCAState* ca_state = entry->ca_state;

    half* fp16_workspace = ca_state->fp16_workspace;

    if (valid_head) {
        TraceBuffer* trace_buffer = &ca_state->trace;
        int trace_idx = -1;
        if (tid == 0 && trace_buffer->traces != nullptr &&
            trace_buffer->current_idx < trace_buffer->capacity) {
            trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
        }
        if (tid < WARP_SIZE) {
            trace_idx = __shfl_sync(0xFFFFFFFF, trace_idx, 0);
            if (trace_idx >= 0 && trace_idx < trace_buffer->capacity) {
                ExecutionTrace* t = &trace_buffer->traces[trace_idx];
                record_warp_metrics(t, blockIdx.x);
                record_memory_access(t, (void*)&fp16_workspace[tid], true);
            }
        }
    }
    float* fp32_workspace = ca_state->fp32_workspace;
    half* perception_weights = ca_state->perception_weights;
    half* interaction_weights = ca_state->interaction_weights;
    half* value_weights = ca_state->value_weights;
    float* ca_output_fp32 = ca_state->ca_output;

    int perception_size = num_cells * arch.head_dim;
    int weight_offset = head * arch.channels * arch.head_dim;
    int output_offset = head * num_cells * arch.head_dim;

    if (valid_head) {
        for (int out_idx = tid; out_idx < perception_size; out_idx += block_threads) {
            int cell = out_idx / arch.head_dim;
            int dim = out_idx % arch.head_dim;
            float acc = 0.0f;
            for (int c = 0; c < arch.channels; c++) {
                acc += __half2float(fp16_workspace[cell * arch.channels + c]) *
                       __half2float(perception_weights[weight_offset + c * arch.head_dim + dim]);
            }
            fp32_workspace[output_offset + out_idx] = acc;
        }
    }
    cg::this_grid().sync();

    if (valid_head) {
        for (int i = tid; i < perception_size; i += block_threads) {
            fp32_workspace[output_offset + i] = activation_relu(fp32_workspace[output_offset + i]);
        }
    }
    cg::this_grid().sync();

    half* interaction_input = fp16_workspace + num_cells * arch.channels;
    if (valid_head) {
        for (int i = tid; i < perception_size; i += block_threads) {
            interaction_input[i] = __float2half(fp32_workspace[output_offset + i]);
        }
    }
    cg::this_grid().sync();

    float* interaction_output = fp32_workspace + arch.num_heads * num_cells * arch.head_dim;
    int interaction_weight_offset = head * arch.head_dim * arch.head_dim;
    if (valid_head) {
        for (int out_idx = tid; out_idx < perception_size; out_idx += block_threads) {
            int cell = out_idx / arch.head_dim;
            int dim = out_idx % arch.head_dim;
            float acc = 0.0f;
            for (int k = 0; k < arch.head_dim; k++) {
                acc += __half2float(interaction_input[cell * arch.head_dim + k]) *
                       __half2float(interaction_weights[interaction_weight_offset + k * arch.head_dim + dim]);
            }
            interaction_output[out_idx] = acc;
        }
    }
    cg::this_grid().sync();

    if (valid_head) {
        for (int i = tid; i < perception_size; i += block_threads) {
            float x = interaction_output[i];
            interaction_output[i] = activation_gelu(x);
        }
    }
    cg::this_grid().sync();

    half* value_input = interaction_input;
    if (valid_head) {
        for (int i = tid; i < perception_size; i += block_threads) {
            value_input[i] = __float2half(interaction_output[i]);
        }
    }
    cg::this_grid().sync();

    float* head_output = ca_output_fp32 + head * num_cells * arch.channels;
    int value_weight_offset = head * arch.head_dim * arch.channels;
    int value_output_size = num_cells * arch.channels;
    if (valid_head) {
        for (int out_idx = tid; out_idx < value_output_size; out_idx += block_threads) {
            int cell = out_idx / arch.channels;
            int chan = out_idx % arch.channels;
            float acc = 0.0f;
            for (int k = 0; k < arch.head_dim; k++) {
                acc += __half2float(value_input[cell * arch.head_dim + k]) *
                       __half2float(value_weights[value_weight_offset + k * arch.channels + chan]);
            }
            head_output[out_idx] = acc;
        }
    }
}

#endif

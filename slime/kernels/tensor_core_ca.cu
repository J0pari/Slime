
#ifndef TENSOR_CORE_CA_CU
#define TENSOR_CORE_CA_CU

#include "../config/config.cu"
#include "../utils/genome_params.cuh"
#include "../core/pseudopod.cu"
#include "../memory/pool.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__global__ void convert_fp32_to_fp16_kernel(
    float* __restrict__ fp32_data,
    half* __restrict__ fp16_data,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        fp16_data[idx] = __float2half(fp32_data[idx]);
    }
}

__global__ void convert_fp16_to_fp32_kernel(
    half* __restrict__ fp16_data,
    float* __restrict__ fp32_data,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        fp32_data[idx] = __half2float(fp16_data[idx]);
    }
}

__global__ void tensor_core_matmul_kernel(
    half* __restrict__ A,
    half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {

    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    const int tile_row = warpM * WMMA_TILE_DIM;
    const int tile_col = warpN * WMMA_TILE_DIM;

    if (tile_row >= M || tile_col >= N) return;

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

__global__ void relu_kernel(
    float* __restrict__ data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = activation_relu(data[idx]);
    }
}

__global__ void gelu_kernel(
    float* __restrict__ data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = activation_gelu(x);
    }
}

__global__ void tensor_core_perception_kernel(
    half* __restrict__ neighborhood_fp16,
    half* __restrict__ perception_weights,
    float* __restrict__ perception_out,
    int grid_size,
    int head_id,
    ArchitectureParams arch
) {

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int num_cells = grid_size * grid_size;

    const int tile_row = (warp_id / ((arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM)) * WMMA_TILE_DIM;
    const int tile_col = (warp_id % ((arch.head_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM)) * WMMA_TILE_DIM;

    if (tile_row >= num_cells || tile_col >= arch.head_dim) return;

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

__global__ void prepare_ca_fp16_kernel(
    ComponentPool* __restrict__ pool,
    int max_grid_size,
    ArchitectureParams arch,
    int entry_idx
) {
    DEVICE_FATAL_IF(entry_idx >= pool->capacity, "prepare_ca_fp16_kernel: entry_idx out of bounds");

    PoolEntry* entry = &pool->entries[entry_idx];
    DEVICE_FATAL_IF(!entry->alive, "prepare_ca_fp16_kernel: dead entry passed");

    int grid_size = entry->grid_size;
    int num_cells = grid_size * grid_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_cells * arch.channels;

    if (idx >= total) return;

    MultiHeadCAState* ca_state = entry->ca_state;
    ca_state->fp16_workspace[idx] = __float2half(ca_state->ca_concentration[idx]);
}

__global__ void multi_head_ca_tensor_kernel(
    ComponentPool* __restrict__ pool,
    int max_grid_size,
    ArchitectureParams arch,
    int entry_idx
) {
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int block_threads = blockDim.x;

    DEVICE_FATAL_IF(entry_idx >= pool->capacity, "multi_head_ca_tensor_kernel: entry_idx out of bounds");
    if (head >= arch.num_heads) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    DEVICE_FATAL_IF(!entry->alive, "multi_head_ca_tensor_kernel: dead entry passed");

    int grid_size = entry->grid_size;
    int num_cells = grid_size * grid_size;

    MultiHeadCAState* ca_state = entry->ca_state;

    if (tid == 0) {
        TraceBuffer* trace_buffer = &ca_state->trace;
        if (trace_buffer->current_idx < trace_buffer->capacity) {
            int trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
            if (trace_idx < trace_buffer->capacity) {
                record_warp_metrics(&trace_buffer->traces[trace_idx], blockIdx.x);
            }
        }
    }

    half* fp16_workspace = ca_state->fp16_workspace;
    float* fp32_workspace = ca_state->fp32_workspace;
    half* perception_weights = ca_state->perception_weights;
    half* interaction_weights = ca_state->interaction_weights;
    half* value_weights = ca_state->value_weights;
    float* ca_output_fp32 = ca_state->ca_output;

    int perception_size = num_cells * arch.head_dim;
    int weight_offset = head * arch.channels * arch.head_dim;
    int output_offset = head * num_cells * arch.head_dim;

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
    __syncthreads();

    for (int i = tid; i < perception_size; i += block_threads) {
        fp32_workspace[output_offset + i] = activation_relu(fp32_workspace[output_offset + i]);
    }
    __syncthreads();

    half* interaction_input = fp16_workspace + num_cells * arch.channels;
    for (int i = tid; i < perception_size; i += block_threads) {
        interaction_input[i] = __float2half(fp32_workspace[output_offset + i]);
    }
    __syncthreads();

    float* interaction_output = fp32_workspace + arch.num_heads * num_cells * arch.head_dim;
    int interaction_weight_offset = head * arch.head_dim * arch.head_dim;
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
    __syncthreads();

    for (int i = tid; i < perception_size; i += block_threads) {
        float x = interaction_output[i];
        interaction_output[i] = activation_gelu(x);
    }
    __syncthreads();

    half* value_input = interaction_input;
    for (int i = tid; i < perception_size; i += block_threads) {
        value_input[i] = __float2half(interaction_output[i]);
    }
    __syncthreads();

    float* head_output = ca_output_fp32 + head * num_cells * arch.channels;
    int value_weight_offset = head * arch.head_dim * arch.channels;
    int value_output_size = num_cells * arch.channels;
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

#endif

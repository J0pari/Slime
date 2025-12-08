
#ifndef TENSOR_CORE_CA_CU
#define TENSOR_CORE_CA_CU

#include "../config/config.cu"
#include "../utils/genome_params.cuh"
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

    const int tile_row = (warp_id / ((arch.hidden_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM)) * WMMA_TILE_DIM;
    const int tile_col = (warp_id % ((arch.hidden_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM)) * WMMA_TILE_DIM;

    if (tile_row >= num_cells || tile_col >= arch.hidden_dim) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int weight_offset = head_id * arch.head_dim * arch.hidden_dim;

    for (int k = 0; k < arch.head_dim; k += WMMA_TILE_DIM) {
        if (k + WMMA_TILE_DIM <= arch.head_dim) {

            wmma::load_matrix_sync(a_frag,
                neighborhood_fp16 + tile_row * arch.head_dim + k,
                arch.head_dim);

            wmma::load_matrix_sync(b_frag,
                perception_weights + weight_offset + k * arch.hidden_dim + tile_col,
                arch.hidden_dim);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int output_offset = head_id * num_cells * arch.hidden_dim;
    wmma::store_matrix_sync(
        perception_out + output_offset + tile_row * arch.hidden_dim + tile_col,
        c_frag,
        arch.hidden_dim,
        wmma::mem_row_major
    );
}

__global__ void relu_kernel(
    float* __restrict__ data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void gelu_kernel(
    float* __restrict__ data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = 0.5f * x * (1.0f + tanhf(0.7978845f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void multi_head_ca_tensor_kernel(
    float* __restrict__ ca_state_fp32,
    half* __restrict__ perception_weights,
    half* __restrict__ interaction_weights,
    half* __restrict__ value_weights,
    float* __restrict__ ca_output_fp32,
    half* __restrict__ fp16_workspace,
    float* __restrict__ fp32_workspace,
    int grid_size,
    ArchitectureParams arch
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int num_cells = grid_size * grid_size;

    convert_fp32_to_fp16_kernel<<<(num_cells * arch.channels + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ca_state_fp32,
        fp16_workspace,
        num_cells,
        arch.channels
    );

    for (int head = 0; head < arch.num_heads; head++) {

        int num_warps = ((num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM) * ((arch.hidden_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM);

        tensor_core_perception_kernel<<<(num_warps * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            fp16_workspace,
            perception_weights,
            fp32_workspace,
            grid_size,
            head,
            arch
        );

        relu_kernel<<<(num_cells * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            fp32_workspace + head * num_cells * arch.hidden_dim,
            num_cells * arch.hidden_dim
        );

        half* interaction_input = fp16_workspace + num_cells * arch.channels;
        convert_fp32_to_fp16_kernel<<<(num_cells * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            fp32_workspace + head * num_cells * arch.hidden_dim,
            interaction_input,
            num_cells,
            arch.hidden_dim
        );

        float* interaction_output = fp32_workspace + arch.num_heads * num_cells * arch.hidden_dim;

        int num_tiles = ((num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM) * ((arch.hidden_dim + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM);
        tensor_core_matmul_kernel<<<dim3((num_tiles + WARP_SIZE - 1) / WARP_SIZE, 1), dim3(WARP_SIZE, 1)>>>(
            interaction_input,
            interaction_weights + head * arch.hidden_dim * arch.hidden_dim,
            interaction_output,
            num_cells,
            arch.hidden_dim,
            arch.hidden_dim
        );

        gelu_kernel<<<(num_cells * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            interaction_output,
            num_cells * arch.hidden_dim
        );

        half* value_input = interaction_input;
        convert_fp32_to_fp16_kernel<<<(num_cells * arch.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            interaction_output,
            value_input,
            num_cells,
            arch.hidden_dim
        );

        float* head_output = ca_output_fp32 + head * num_cells * arch.channels;

        num_tiles = ((num_cells + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM) * ((arch.channels + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM);
        tensor_core_matmul_kernel<<<dim3((num_tiles + WARP_SIZE - 1) / WARP_SIZE, 1), dim3(WARP_SIZE, 1)>>>(
            value_input,
            value_weights + head * arch.hidden_dim * arch.channels,
            head_output,
            num_cells,
            arch.channels,
            arch.hidden_dim
        );
    }

}

__global__ void init_tensor_weights_kernel(
    half* __restrict__ weights,
    int size,
    unsigned int seed,
    int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    curandState rand_state;
    curand_init(seed + idx, 0, 0, &rand_state);

    float val = curand_normal(&rand_state) * sqrtf(2.0f / channels);
    weights[idx] = __float2half(val);
}

#endif

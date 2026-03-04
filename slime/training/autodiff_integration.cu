
#ifndef AUTODIFF_INTEGRATION_CU
#define AUTODIFF_INTEGRATION_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include "training_types.cu"
#include "../learning/autodiff.cu"
#include "../core/pseudopod.cu"
#include "../metrics/hardware_geometry.cu"
#include "../utils/cuda_primitives.cuh"
#include "../utils/genome_params.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__device__ void transpose_matrix_device(Organism* organism) {
    const float* A = organism->transpose_A_fp32;
    float* B = organism->transpose_B_fp32;
    int M = organism->transpose_M;
    int N = organism->transpose_N;

    __shared__ float tile[WMMA_TILE_DIM][WMMA_TILE_DIM + BANK_PAD];
    int bx = blockIdx.x * WMMA_TILE_DIM, by = blockIdx.y * WMMA_TILE_DIM;
    int x = bx + threadIdx.x, y = by + threadIdx.y;

    if (y < M && x < N) tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    cg::this_grid().sync();

    int out_x = by + threadIdx.x, out_y = bx + threadIdx.y;
    if (out_y < N && out_x < M) B[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];
}

__device__ void tensor_core_gemm_device(Organism* organism) {
    const half* A = organism->gemm_A;
    const half* B = organism->gemm_B;
    float* C = organism->gemm_C;
    int M = organism->gemm_M;
    int N = organism->gemm_N;
    int K = organism->gemm_K;

    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warpN = blockIdx.y;

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

__device__ void fp32_to_fp16_device(Organism* organism) {
    const float* fp32 = organism->conv_fp32;
    half* fp16 = organism->conv_fp16;
    int size = organism->conv_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) fp16[idx] = __float2half(fp32[idx]);
}

__device__ void fp16_to_fp32_device(Organism* organism) {
    const half* fp16 = organism->conv_fp16;
    float* fp32 = organism->conv_fp32;
    int size = organism->conv_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) fp32[idx] = __half2float(fp16[idx]);
}

__device__ void accumulate_weight_grads_device(Organism* organism) {
    const float* dW = organism->weight_grads_src;
    float* grad_buffer = organism->weight_grads_dst;
    int offset = organism->weight_grads_offset;
    int size = organism->weight_grads_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad_buffer[offset + idx] = dW[idx];
}

__device__ void tensor_core_gemm_transA_device(Organism* organism) {
    const half* A = organism->gemm_A;
    const half* B = organism->gemm_B;
    float* C = organism->gemm_C;
    int M = organism->gemm_M;
    int N = organism->gemm_N;
    int K = organism->gemm_K;

    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warpN = blockIdx.y;
    const int tile_row = warpM * WMMA_TILE_DIM;
    const int tile_col = warpN * WMMA_TILE_DIM;

    if (tile_row < M && tile_col < N) {
        wmma::fragment<wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, wmma::col_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        for (int k_tile = 0; k_tile < K; k_tile += WMMA_TILE_DIM) {
            if (k_tile + WMMA_TILE_DIM <= K) {
                wmma::load_matrix_sync(a_frag, A + k_tile * M + tile_row, M);
                wmma::load_matrix_sync(b_frag, B + k_tile * N + tile_col, N);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        wmma::store_matrix_sync(C + tile_row * N + tile_col, c_frag, N, wmma::mem_row_major);
    }
}

__device__ void transpose_fp16_device(Organism* organism) {
    const half* A = organism->transpose_A;
    half* B = organism->transpose_B;
    int M = organism->transpose_M;
    int N = organism->transpose_N;

    __shared__ half tile[WMMA_TILE_DIM][WMMA_TILE_DIM + 1];
    int bx = blockIdx.x * WMMA_TILE_DIM, by = blockIdx.y * WMMA_TILE_DIM;
    int x = bx + threadIdx.x, y = by + threadIdx.y;

    if (y < M && x < N) tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    cg::this_grid().sync();

    int out_x = by + threadIdx.x, out_y = bx + threadIdx.y;
    if (out_y < N && out_x < M) B[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];
}

// The canonical forward pass lives in hybrid_lifecycle.cu.

// All gradients flow through per-entry ca_state->tape.grad_buffer → Adam.

__device__ void init_ca_parameter_map_device(Organism* organism) {
    CAParameterMap* map = organism->param_map;
    Architecture arch = organism->current_arch;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        map->perception_size = arch.channels * arch.head_dim;
        map->interaction_size = arch.head_dim * arch.head_dim;
        map->flow_projection_size = 2 * arch.head_dim;

        int offset = 0;
        for (int h = 0; h < arch.num_heads; h++) {
            map->head_param_offsets[h] = offset;

            map->perception_start[h] = offset;
            offset += map->perception_size;

            map->interaction_start[h] = offset;
            offset += map->interaction_size;

            map->flow_projection_start[h] = offset;
            offset += map->flow_projection_size;

            map->head_param_counts[h] = map->perception_size + map->interaction_size + map->flow_projection_size;
        }

        map->total_params = offset;
        map->total_ca_params = offset;
    }
}

__device__ void im2col_device(Organism* organism) {
    const float* input = organism->im2col_input;
    float* col = organism->im2col_col;
    int batch_size = organism->im2col_batch_size;
    int grid_size = organism->ca_grid_size;
    int channels = organism->ca_channels;
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = grid_size * grid_size;
    if (cell_idx < batch_size * num_cells) {
        int batch_id = cell_idx / num_cells;
        int local_cell = cell_idx % num_cells;
        int cell_y = local_cell / grid_size;
        int cell_x = local_cell % grid_size;

        int col_row = cell_idx;
        int col_width = 9 * channels;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = max(0, min(grid_size - 1, cell_y + dy));
                int nx = max(0, min(grid_size - 1, cell_x + dx));
                int patch_idx = (dy + 1) * 3 + (dx + 1);

                int input_base = batch_id * grid_size * grid_size * channels +
                                ny * grid_size * channels + nx * channels;

                for (int c = 0; c < channels; c++) {
                    col[col_row * col_width + patch_idx * channels + c] = input[input_base + c];
                }
            }
        }
    }
}

__device__ void col2im_device(Organism* organism) {
    const float* col = organism->im2col_col;
    float* input_grad = organism->col2im_output_grad;
    int batch_size = organism->im2col_batch_size;
    int grid_size = organism->ca_grid_size;
    int channels = organism->ca_channels;
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = grid_size * grid_size;
    if (cell_idx < batch_size * num_cells) {
        int batch_id = cell_idx / num_cells;
        int local_cell = cell_idx % num_cells;
        int cell_y = local_cell / grid_size;
        int cell_x = local_cell % grid_size;

        int col_width = 9 * channels;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int out_y = cell_y - dy;
                int out_x = cell_x - dx;
                if (out_y >= 0 && out_y < grid_size && out_x >= 0 && out_x < grid_size) {
                    int out_cell = batch_id * num_cells + out_y * grid_size + out_x;
                    int patch_idx = (dy + 1) * 3 + (dx + 1);

                    int input_base = batch_id * grid_size * grid_size * channels +
                                    cell_y * grid_size * channels + cell_x * channels;

                    for (int c = 0; c < channels; c++) {
                        atomicAdd(&input_grad[input_base + c],
                                 col[out_cell * col_width + patch_idx * channels + c]);
                    }
                }
            }
        }
    }
}

__device__ void relu_backward_device(Organism* organism) {
    const float* dL_dP = organism->backward_dL_dP;
    const float* P = organism->backward_P;
    float* dL_dprerelu = organism->backward_dL_dprerelu;
    int size = organism->backward_elements_per_head;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dL_dprerelu[idx] = dL_dP[idx] * ((P[idx] > 0.0f) ? 1.0f : 0.0f);
    }
}

// route_autodiff_to_unified_device REMOVED: unified gradient buffer system eliminated.
// Adam reads directly from ca_state->tape.grad_buffer (CA weights) and per-entry
// fc_weights_grad/fc_bias_grad/pooling_weights_grad (classifier). NaN/Inf validation
// merged into per-entry Adam functions.

// route_classification_to_unified_device REMOVED: same reason as above.

#endif

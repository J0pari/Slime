
#ifndef AUTODIFF_INTEGRATION_CU
#define AUTODIFF_INTEGRATION_CU

#include "../config/config.cu"
#include "training_types.cu"
#include "../learning/autodiff.cu"
#include "../core/pseudopod.cu"
#include "../utils/cuda_primitives.cuh"
#include "../utils/genome_params.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__global__ void transpose_matrix_kernel(
    const float* __restrict__ A, float* __restrict__ B, int M, int N
) {
    __shared__ float tile[WMMA_TILE_DIM][WMMA_TILE_DIM + BANK_PAD];
    int bx = blockIdx.x * WMMA_TILE_DIM, by = blockIdx.y * WMMA_TILE_DIM;
    int x = bx + threadIdx.x, y = by + threadIdx.y;

    if (y < M && x < N) tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    __syncthreads();

    int out_x = by + threadIdx.x, out_y = bx + threadIdx.y;
    if (out_y < N && out_x < M) B[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void tensor_core_gemm_kernel(
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C,
    int M, int N, int K
) {
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warpN = blockIdx.y;

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

__global__ void fp32_to_fp16_kernel(const float* __restrict__ fp32, half* __restrict__ fp16, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) fp16[idx] = __float2half(fp32[idx]);
}

__global__ void fp16_to_fp32_kernel(const half* __restrict__ fp16, float* __restrict__ fp32, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) fp32[idx] = __half2float(fp16[idx]);
}

__global__ void accumulate_weight_grads_kernel(
    const float* __restrict__ dW, float* __restrict__ grad_buffer, int offset, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad_buffer[offset + idx] = dW[idx];
}

__global__ void tensor_core_gemm_transA_kernel(
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C,
    int M, int N, int K
) {
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warpN = blockIdx.y;
    const int tile_row = warpM * WMMA_TILE_DIM;
    const int tile_col = warpN * WMMA_TILE_DIM;

    if (tile_row >= M || tile_col >= N) return;

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

__global__ void transpose_fp16_kernel(const half* __restrict__ A, half* __restrict__ B, int M, int N) {
    __shared__ half tile[WMMA_TILE_DIM][WMMA_TILE_DIM + 1];
    int bx = blockIdx.x * WMMA_TILE_DIM, by = blockIdx.y * WMMA_TILE_DIM;
    int x = bx + threadIdx.x, y = by + threadIdx.y;

    if (y < M && x < N) tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    __syncthreads();

    int out_x = by + threadIdx.x, out_y = bx + threadIdx.y;
    if (out_y < N && out_x < M) B[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void multi_head_ca_with_tape_kernel(
    float* __restrict__ ca_state,
    MultiHeadCAState* __restrict__ ca_heads,
    float* __restrict__ ca_output,
    float* __restrict__ perception_saved,
    float* __restrict__ interaction_saved,
    float* __restrict__ pre_gelu_saved,
    CAParameterMap* __restrict__ param_map,
    int micro_batch_size,
    int micro_batch_offset,
    int grid_size,
    ArchitectureParams arch
) {
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int head_batch_id = blockIdx.z;
    int head_id = head_batch_id % arch.num_heads;
    int micro_batch_id = head_batch_id / arch.num_heads;
    int batch_id = micro_batch_offset + micro_batch_id;

    int cell_x = tile_x * blockDim.x + threadIdx.x;
    int cell_y = tile_y * blockDim.y + threadIdx.y;

    if (cell_x >= grid_size || cell_y >= grid_size) return;

    if (head_id == 0 && batch_id == 0 && cell_x == 0 && cell_y == 0 && threadIdx.z == 0) {
        if (isnan(ca_state[0])) {
            printf("FATAL [multi_head_ca]: ca_state[0]=NaN at entry\n");
            return;
        }
        half* perception_weights = &ca_heads->perception_weights[0];
        float w0 = __half2float(perception_weights[0]);
        if (isnan(w0)) {
            printf("FATAL [multi_head_ca]: perception_weights[0]=NaN\n");
            return;
        }
        half* interaction_weights = &ca_heads->interaction_weights[0];
        float iw0 = __half2float(interaction_weights[0]);
        if (isnan(iw0)) {
            printf("FATAL [multi_head_ca]: interaction_weights[0]=NaN\n");
            return;
        }
        half* value_weights = &ca_heads->value_weights[0];
        float vw0 = __half2float(value_weights[0]);
        if (isnan(vw0)) {
            printf("FATAL [multi_head_ca]: value_weights[0]=NaN\n");
            return;
        }
    }

    __shared__ float neighborhood[3][3][MAX_HEAD_DIM + BANK_PAD];

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;

            nx = (nx < 0) ? 0 : ((nx >= grid_size) ? grid_size - 1 : nx);
            ny = (ny < 0) ? 0 : ((ny >= grid_size) ? grid_size - 1 : ny);

            int idx = batch_id * grid_size * grid_size * arch.head_dim +
                     ny * grid_size * arch.head_dim +
                     nx * arch.head_dim;

            if (threadIdx.z < arch.head_dim) {
                neighborhood[dy + 1][dx + 1][threadIdx.z] = ldg_float(&ca_state[idx + threadIdx.z]);
            }
        }
    }
    __syncthreads();

    float perception[MAX_HEAD_DIM];
    float interaction[MAX_HEAD_DIM];
    float output[MAX_HEAD_DIM];

    half* perception_weights_fp16 = &ca_heads->perception_weights[head_id * arch.channels * arch.head_dim];

    for (int h = 0; h < arch.head_dim; h++) {
        float acc = 0.0f;
        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
                for (int c = 0; c < arch.channels; c++) {
                    int weight_idx = c * arch.head_dim + h;
                    float weight_val = __half2float(perception_weights_fp16[weight_idx]);
                    float neigh_val = neighborhood[dy][dx][c];
                    acc += weight_val * neigh_val;
                }
            }
        }
        perception[h] = fmaxf(0.0f, acc);
        // Use micro_batch_id for saved buffer indexing (fits within buffer capacity)
        int idx = micro_batch_id * (arch.num_heads * grid_size * grid_size * arch.head_dim) +
                  head_id * (grid_size * grid_size * arch.head_dim) +
                  cell_y * (grid_size * arch.head_dim) +
                  cell_x * arch.head_dim + h;
        perception_saved[idx] = perception[h];
    }

    half* interaction_weights_fp16 = &ca_heads->interaction_weights[head_id * arch.head_dim * arch.head_dim];

    for (int h = 0; h < arch.head_dim; h++) {
        float acc = 0.0f;
        for (int j = 0; j < arch.head_dim; j++) {
            int weight_idx = j * arch.head_dim + h;
            float weight_val = __half2float(interaction_weights_fp16[weight_idx]);
            acc += weight_val * perception[j];

            if (isnan(weight_val)) {
                printf("FATAL [multi_head_ca]: interaction_weight[%d,%d]=nan\n", j, h);
                return;
            }
        }
        float x = acc;
        float x_cubed = x * x * x;
        float inner = GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC_COEFFICIENT * x_cubed);
        float tanh_val = tanhf(inner);
        interaction[h] = GELU_SCALE * x * (GELU_OFFSET + tanh_val);

        if (isnan(interaction[h]) || isinf(interaction[h])) {
            printf("FATAL [multi_head_ca]: interaction[%d]=nan/inf acc=%.6f x=%.6f inner=%.6f tanh=%.6f\n", h, acc, x, inner, tanh_val);
            return;
        }

        // Use micro_batch_id for saved buffer indexing (fits within buffer capacity)
        int idx = micro_batch_id * (arch.num_heads * grid_size * grid_size * arch.head_dim) +
                  head_id * (grid_size * grid_size * arch.head_dim) +
                  cell_y * (grid_size * arch.head_dim) +
                  cell_x * arch.head_dim + h;
        pre_gelu_saved[idx] = x;
        interaction_saved[idx] = interaction[h];
    }

    half* value_weights_fp16 = &ca_heads->value_weights[head_id * arch.head_dim * arch.channels];

    for (int i = 0; i < arch.channels; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < arch.head_dim; j++) {
            int weight_idx = j * arch.channels + i;
            float weight_val = __half2float(value_weights_fp16[weight_idx]);
            output[i] += interaction[j] * weight_val;
        }
    }

    int out_idx = batch_id * arch.num_heads * grid_size * grid_size * arch.channels +
                  head_id * grid_size * grid_size * arch.channels +
                  cell_y * grid_size * arch.channels +
                  cell_x * arch.channels;

    for (int i = 0; i < arch.channels; i++) {
        if (isnan(output[i]) || isinf(output[i])) {
            printf("FATAL [multi_head_ca]: output[%d]=nan/inf perception[0]=%.6f interaction[0]=%.6f\n", i, perception[0], interaction[0]);
            return;
        }
        ca_output[out_idx + i] = output[i];
    }
}

__global__ void apply_ca_gradients_kernel(
    ADTape* __restrict__ tape,
    CAParameterMap* __restrict__ param_map,
    MultiHeadCAState* __restrict__ ca_heads,
    float learning_rate,
    float gradient_clip,
    ArchitectureParams arch
) {
    int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_id = blockIdx.y;

    if (head_id >= arch.num_heads) return;

    int perception_base = param_map->perception_start[head_id];
    int interaction_base = param_map->interaction_start[head_id];
    int value_base = param_map->value_start[head_id];

    half* param_ptr_fp16 = nullptr;
    float* param_ptr_fp32 = nullptr;
    int tape_idx = -1;
    bool is_fp16 = true;

    if (param_idx < param_map->perception_size) {
        param_ptr_fp16 = &ca_heads->perception_weights[head_id * param_map->perception_size + param_idx];
        tape_idx = perception_base + param_idx;
    } else if (param_idx < param_map->perception_size + param_map->interaction_size) {
        int local_idx = param_idx - param_map->perception_size;
        param_ptr_fp16 = &ca_heads->interaction_weights[head_id * param_map->interaction_size + local_idx];
        tape_idx = interaction_base + local_idx;
    } else if (param_idx < param_map->perception_size + param_map->interaction_size + param_map->value_size) {
        int local_idx = param_idx - param_map->perception_size - param_map->interaction_size;
        param_ptr_fp16 = &ca_heads->value_weights[head_id * param_map->value_size + local_idx];
        tape_idx = value_base + local_idx;
    }

    if ((param_ptr_fp16 != nullptr || param_ptr_fp32 != nullptr) && tape_idx >= 0 && tape_idx < tape->current_value_idx) {
        float grad = tape->grad_buffer[tape_idx];

        if (isnan(grad) || isinf(grad)) {
            printf("FATAL [apply_ca_gradients]: grad[%d]=%.6f for head=%d param=%d\n", tape_idx, grad, head_id, param_idx);
            return;
        }

        if (fabsf(grad) > gradient_clip) {
            grad = copysignf(gradient_clip, grad);
        }

        if (is_fp16 && param_ptr_fp16 != nullptr) {
            float val = __half2float(*param_ptr_fp16);
            if (isnan(val)) {
                printf("FATAL [apply_ca_gradients]: weight already NaN at head=%d param=%d\n", head_id, param_idx);
                return;
            }
            val -= learning_rate * grad;
            if (isnan(val) || isinf(val)) {
                printf("FATAL [apply_ca_gradients]: updated weight=%.6f lr=%.6f grad=%.6f\n", val, learning_rate, grad);
                return;
            }
            *param_ptr_fp16 = __float2half(val);
        } else if (param_ptr_fp32 != nullptr) {
            *param_ptr_fp32 -= learning_rate * grad;
        }

        tape->grad_buffer[tape_idx] = 0.0f;
    }
}

__global__ void init_ca_parameter_map_kernel(CAParameterMap* map, ArchitectureParams arch) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    map->perception_size = arch.channels * arch.head_dim;
    map->interaction_size = arch.head_dim * arch.head_dim;
    map->value_size = arch.head_dim * arch.channels;

    int offset = 0;
    for (int h = 0; h < arch.num_heads; h++) {
        map->head_param_offsets[h] = offset;

        map->perception_start[h] = offset;
        offset += map->perception_size;

        map->interaction_start[h] = offset;
        offset += map->interaction_size;

        map->value_start[h] = offset;
        offset += map->value_size;

        map->head_param_counts[h] = map->perception_size + map->interaction_size + map->value_size;
    }

    map->total_params = offset;
    map->total_ca_params = offset;

    printf("[DEVICE] CA param map: %d params, %d heads\n", map->total_params, arch.num_heads);
}

__global__ void im2col_kernel(
    const float* __restrict__ input,
    float* __restrict__ col,
    int batch_size, int grid_size, int channels
) {
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = grid_size * grid_size;
    if (cell_idx >= batch_size * num_cells) return;

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

__global__ void col2im_kernel(
    const float* __restrict__ col,
    float* __restrict__ input_grad,
    int batch_size, int grid_size, int channels
) {
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = grid_size * grid_size;
    if (cell_idx >= batch_size * num_cells) return;

    int batch_id = cell_idx / num_cells;
    int local_cell = cell_idx % num_cells;
    int cell_y = local_cell / grid_size;
    int cell_x = local_cell % grid_size;

    int col_width = 9 * channels;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int out_y = cell_y - dy;
            int out_x = cell_x - dx;
            if (out_y < 0 || out_y >= grid_size || out_x < 0 || out_x >= grid_size) continue;

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

__global__ void relu_backward_kernel(
    const float* __restrict__ dL_dP,
    const float* __restrict__ P,
    float* __restrict__ dL_dprerelu,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dL_dprerelu[idx] = dL_dP[idx] * ((P[idx] > 0.0f) ? 1.0f : 0.0f);
    }
}

#endif

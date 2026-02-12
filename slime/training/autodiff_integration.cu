
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
    __syncthreads();

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

__device__ void transpose_fp16_device(Organism* organism) {
    const half* A = organism->transpose_A;
    half* B = organism->transpose_B;
    int M = organism->transpose_M;
    int N = organism->transpose_N;

    __shared__ half tile[WMMA_TILE_DIM][WMMA_TILE_DIM + 1];
    int bx = blockIdx.x * WMMA_TILE_DIM, by = blockIdx.y * WMMA_TILE_DIM;
    int x = bx + threadIdx.x, y = by + threadIdx.y;

    if (y < M && x < N) tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    __syncthreads();

    int out_x = by + threadIdx.x, out_y = bx + threadIdx.y;
    if (out_y < N && out_x < M) B[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];
}

__device__ void multi_head_ca_with_tape_device(Organism* organism) {
    float* ca_state = organism->ca_state;
    MultiHeadCAState* ca_heads = organism->multihead_ca_state;
    float* ca_output = organism->ca_output;
    float* perception_saved = organism->perception_saved;
    float* interaction_saved = organism->interaction_saved;
    float* pre_gelu_saved = organism->pre_gelu_saved;
    CAParameterMap* param_map = organism->ca_param_map;
    int micro_batch_size = organism->micro_batch_size;
    int micro_batch_offset = organism->micro_batch_offset;
    int grid_size = organism->ca_grid_size;
    Architecture arch = organism->current_arch;
    const int head_id = blockIdx.z % arch.num_heads;
    const int micro_batch_id = blockIdx.z / arch.num_heads;
    const int batch_id = micro_batch_offset + micro_batch_id;
    const int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (cell_x >= grid_size || cell_y >= grid_size) return;

    TraceBuffer* trace_buffer = &ca_heads->trace;
    if (trace_buffer->current_idx < trace_buffer->capacity && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        int trace_idx = atomicAdd(&trace_buffer->current_idx, 1);
        if (trace_idx < trace_buffer->capacity) {
            record_warp_metrics(&trace_buffer->traces[trace_idx], blockIdx.x);
        }
    }

    const int cells_per_grid = grid_size * grid_size;
    const int saved_base = micro_batch_id * arch.num_heads * cells_per_grid * arch.head_dim +
                           head_id * cells_per_grid * arch.head_dim +
                           cell_y * grid_size * arch.head_dim +
                           cell_x * arch.head_dim;

    __shared__ float neighborhood[3][3][MAX_CHANNELS + BANK_PAD];

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = min(max(cell_x + dx, 0), grid_size - 1);
            int ny = min(max(cell_y + dy, 0), grid_size - 1);
            int state_idx = batch_id * cells_per_grid * arch.channels +
                           ny * grid_size * arch.channels +
                           nx * arch.channels;

            if (threadIdx.z < arch.channels) {
                neighborhood[dy + 1][dx + 1][threadIdx.z] = ldg_float(&ca_state[state_idx + threadIdx.z]);
            }
        }
    }
    __syncthreads();

    float perception[MAX_HEAD_DIM];
    float interaction[MAX_HEAD_DIM];
    float output[MAX_CHANNELS];

    half* perc_w = &ca_heads->perception_weights[head_id * arch.channels * arch.head_dim];
    half* inter_w = &ca_heads->interaction_weights[head_id * arch.head_dim * arch.head_dim];
    half* val_w = &ca_heads->value_weights[head_id * arch.head_dim * arch.channels];

    for (int h = 0; h < arch.head_dim; h++) {
        float acc = 0.0f;
        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
                for (int c = 0; c < arch.channels; c++) {
                    acc += neighborhood[dy][dx][c] * __half2float(perc_w[c * arch.head_dim + h]);
                }
            }
        }
        perception[h] = activation_relu(acc);
        perception_saved[saved_base + h] = perception[h];
    }

    float interaction_sum = 0.0f;
    for (int h = 0; h < arch.head_dim; h++) {
        float acc = 0.0f;
        for (int j = 0; j < arch.head_dim; j++) {
            acc += perception[j] * __half2float(inter_w[j * arch.head_dim + h]);
        }
        float x = acc;
        interaction[h] = activation_gelu(x);
        interaction_sum += fabsf(interaction[h]);
        pre_gelu_saved[saved_base + h] = x;
        interaction_saved[saved_base + h] = interaction[h];
    }

    for (int c = 0; c < arch.channels; c++) {
        float acc = 0.0f;
        for (int h = 0; h < arch.head_dim; h++) {
            acc += interaction[h] * __half2float(val_w[h * arch.channels + c]);
        }
        output[c] = acc;
    }

    float coherence = organism->coherence_history[(organism->generation % 2) * POOL_CAPACITY_MAX];
    float gate = activation_sigmoid(interaction_sum / (float)arch.head_dim - compute_ca_gate_center(coherence));

    int out_idx = batch_id * arch.num_heads * cells_per_grid * arch.channels +
                  head_id * cells_per_grid * arch.channels +
                  cell_y * grid_size * arch.channels +
                  cell_x * arch.channels;

    for (int c = 0; c < arch.channels; c++) {
        float input_val = neighborhood[1][1][c];
        ca_output[out_idx + c] = input_val * (1.0f - gate) + output[c] * gate;
    }
}

__device__ void apply_ca_gradients_device(Organism* organism) {
    ADTape* tape = organism->ad_tape;
    CAParameterMap* param_map = organism->ca_param_map;
    MultiHeadCAState* ca_heads = organism->multihead_ca_state;
    float learning_rate = organism->learning_rate;
    float gradient_clip = organism->gradient_clip_norm;
    Architecture arch = organism->current_arch;
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

        DEVICE_FATAL_IF(isnan(grad), "apply_ca_gradients: gradient is NaN - autodiff tape corrupted");
        DEVICE_FATAL_IF(isinf(grad), "apply_ca_gradients: gradient is Inf - autodiff tape corrupted");

        if (fabsf(grad) > gradient_clip) {
            grad = copysignf(gradient_clip, grad);
        }

        if (is_fp16 && param_ptr_fp16 != nullptr) {
            float val = __half2float(*param_ptr_fp16);
            DEVICE_FATAL_IF(isnan(val), "apply_ca_gradients: weight is NaN before update - data corrupted");
            val -= learning_rate * grad;
            DEVICE_FATAL_IF(isnan(val), "apply_ca_gradients: weight became NaN after update");
            DEVICE_FATAL_IF(isinf(val), "apply_ca_gradients: weight became Inf after update");
            *param_ptr_fp16 = __float2half(val);
        } else if (param_ptr_fp32 != nullptr) {
            *param_ptr_fp32 -= learning_rate * grad;
        }

        tape->grad_buffer[tape_idx] = 0.0f;
    }
}

__device__ void init_ca_parameter_map_device(Organism* organism) {
    CAParameterMap* map = organism->ca_param_map;
    Architecture arch = organism->current_arch;
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
}

__device__ void im2col_device(Organism* organism) {
    const float* input = organism->im2col_input;
    float* col = organism->im2col_col;
    int batch_size = organism->im2col_batch_size;
    int grid_size = organism->ca_grid_size;
    int channels = organism->ca_channels;
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

__device__ void col2im_device(Organism* organism) {
    const float* col = organism->im2col_col;
    float* input_grad = organism->col2im_output_grad;
    int batch_size = organism->im2col_batch_size;
    int grid_size = organism->ca_grid_size;
    int channels = organism->ca_channels;
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

__device__ void route_autodiff_to_unified_device(Organism* organism) {
    ADTape* tape = organism->ad_tape;
    CAParameterMap* param_map = organism->ca_param_map;
    UnifiedGradientBuffer* grad_buf = organism->unified_grad_buffer;
    float gradient_clip = organism->gradient_clip_norm;
    Architecture arch = organism->current_arch;
    int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_id = blockIdx.y;

    if (head_id >= arch.num_heads) return;

    int perception_base = param_map->perception_start[head_id];
    int interaction_base = param_map->interaction_start[head_id];
    int value_base = param_map->value_start[head_id];

    int per_head_perception = arch.channels * arch.head_dim;
    int per_head_interaction = arch.head_dim * arch.head_dim;
    int per_head_value = arch.head_dim * arch.channels;
    int total_per_head = per_head_perception + per_head_interaction + per_head_value;

    if (param_idx >= total_per_head) return;

    int tape_idx = -1;
    float* target = nullptr;
    int target_idx = -1;

    if (param_idx < per_head_perception) {
        tape_idx = perception_base + param_idx;
        target = grad_buf->perception_grads;
        target_idx = head_id * per_head_perception + param_idx;
    } else if (param_idx < per_head_perception + per_head_interaction) {
        int local_idx = param_idx - per_head_perception;
        tape_idx = interaction_base + local_idx;
        target = grad_buf->interaction_grads;
        target_idx = head_id * per_head_interaction + local_idx;
    } else {
        int local_idx = param_idx - per_head_perception - per_head_interaction;
        tape_idx = value_base + local_idx;
        target = grad_buf->value_grads;
        target_idx = head_id * per_head_value + local_idx;
    }

    if (tape_idx >= 0 && tape_idx < tape->current_value_idx && target != nullptr) {
        float grad = tape->grad_buffer[tape_idx];

        DEVICE_FATAL_IF(isnan(grad), "route_autodiff: gradient is NaN");
        DEVICE_FATAL_IF(isinf(grad), "route_autodiff: gradient is Inf");

        if (fabsf(grad) > gradient_clip) {
            grad = copysignf(gradient_clip, grad);
        }

        atomicAdd(&target[target_idx], grad);
        tape->grad_buffer[tape_idx] = 0.0f;

        if (param_idx == 0 && head_id == 0) {
            grad_buf->has_autodiff_grads = 1;
        }
    }
}

__device__ void route_classification_to_unified_device(Organism* organism) {
    float* pooling_grads_in = organism->cls_pooling_weights_grad;
    float* fc_weight_grads_in = organism->cls_fc_weights_grad;
    float* fc_bias_grads_in = organism->cls_fc_bias_grad;
    UnifiedGradientBuffer* grad_buf = organism->unified_grad_buffer;
    int num_features = organism->cls_num_features;
    int num_classes = organism->cls_num_classes;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_features) {
        float grad = pooling_grads_in[idx];
        DEVICE_FATAL_IF(isnan(grad), "route_classification: pooling grad is NaN");
        atomicAdd(&grad_buf->pooling_weight_grads[idx], grad);
        pooling_grads_in[idx] = 0.0f;
    }

    if (idx < num_classes * num_features) {
        float grad = fc_weight_grads_in[idx];
        DEVICE_FATAL_IF(isnan(grad), "route_classification: fc_weight grad is NaN");
        atomicAdd(&grad_buf->fc_weight_grads[idx], grad);
        fc_weight_grads_in[idx] = 0.0f;
    }

    if (idx < num_classes) {
        float grad = fc_bias_grads_in[idx];
        DEVICE_FATAL_IF(isnan(grad), "route_classification: fc_bias grad is NaN");
        atomicAdd(&grad_buf->fc_bias_grads[idx], grad);
        fc_bias_grads_in[idx] = 0.0f;
    }

    if (idx == 0) {
        grad_buf->has_backprop_grads = 1;
    }
}

#endif

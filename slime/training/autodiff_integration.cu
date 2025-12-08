
#ifndef AUTODIFF_INTEGRATION_CU
#define AUTODIFF_INTEGRATION_CU

#include "../config/config.cu"
#include "training_types.cu"
#include "../learning/autodiff.cu"
#include "../core/pseudopod.cu"
#include "../utils/tile_ops.cuh"
#include "../utils/genome_params.cuh"
#include <cuda_runtime.h>

enum CAOpType {
    CA_PERCEPTION,
    CA_INTERACTION,
    CA_VALUE,
    CA_FLOW_LENIA,
    CA_HEAD_MIXING
};

__device__ int record_ca_perception(
    ADTape* tape,
    float* neighborhood,
    float* weights,
    float* output,
    int head_id,
    int hidden_dim,
    int channels
) {

    for (int h = 0; h < hidden_dim; h++) {
        float acc = 0.0f;

        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
                for (int c = 0; c < channels; c++) {
                    int weight_idx = c * hidden_dim + h;
                    int neigh_idx = dy * 3 * channels + dx * channels + c;

                    float weight_val = weights[weight_idx];
                    float neigh_val = neighborhood[neigh_idx];
                    float prod = weight_val * neigh_val;

                    int weight_tape_idx = tape_record_unary(tape, OP_MUL, weight_idx, weight_val, neigh_val);
                    int prod_tape_idx = tape_record_binary(tape, OP_MUL, weight_tape_idx, neigh_idx, prod);

                    int prev_acc_idx = (acc == 0.0f) ? -1 : h;
                    acc += prod;
                    tape_record_binary(tape, OP_ADD, prev_acc_idx, prod_tape_idx, acc);
                }
            }
        }

        output[h] = fmaxf(0.0f, acc);
        tape_record_unary(tape, OP_TANH, h, output[h], acc);
    }

    return 0;
}

__device__ int record_ca_interaction(
    ADTape* tape,
    float* perception,
    float* weights,
    float* output,
    int head_id,
    int hidden_dim,
    int channels
) {

    for (int h = 0; h < hidden_dim; h++) {
        float acc = 0.0f;

        for (int j = 0; j < hidden_dim; j++) {
            int weight_idx = j * hidden_dim + h;
            float weight_val = weights[weight_idx];
            float percept_val = perception[j];
            float prod = weight_val * percept_val;

            int weight_tape_idx = tape_record_unary(tape, OP_MUL, weight_idx, weight_val, 0.0f);
            int prod_tape_idx = tape_record_binary(tape, OP_MUL, weight_tape_idx, j, prod);

            acc += prod;
        }

        float x = acc;
        float x_cubed = x * x * x;
        float inner = GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC_COEFFICIENT * x_cubed);
        float tanh_val = tanhf(inner);
        output[h] = GELU_SCALE * x * (GELU_OFFSET + tanh_val);

        tape_record_unary(tape, OP_TANH, h, output[h], inner);
    }

    return 0;
}

__global__ void multi_head_ca_with_tape_kernel(
    float* __restrict__ ca_state,
    MultiHeadCAState* __restrict__ ca_heads,
    float* __restrict__ ca_output,
    ADTape* __restrict__ tape,
    CAParameterMap* __restrict__ param_map,
    int batch_size,
    int grid_size,
    ArchitectureParams arch
) {
    int head_id = blockIdx.y;
    int batch_id = blockIdx.z;
    int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_y = blockIdx.x * blockDim.y + threadIdx.y;

    if (cell_x >= grid_size || cell_y >= grid_size) return;

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

    half* perception_weights_fp16 = &ca_heads->perception_weights[head_id * arch.channels * arch.hidden_dim];
    float perception_weights_fp32[MAX_HEAD_DIM];
    const half2* weights_vec = reinterpret_cast<const half2*>(perception_weights_fp16);
    for (int i = 0; i < arch.head_dim / 2; i++) {
        half2 h2 = weights_vec[i];
        perception_weights_fp32[i * 2] = __half2float(h2.x);
        perception_weights_fp32[i * 2 + 1] = __half2float(h2.y);
    }
    record_ca_perception(tape, (float*)neighborhood, perception_weights_fp32, perception,
                        head_id, arch.head_dim, arch.channels);

    half* interaction_weights_fp16 = &ca_heads->interaction_weights[head_id * arch.channels * arch.hidden_dim];
    float interaction_weights_fp32[MAX_HEAD_DIM];
    const half2* inter_vec = reinterpret_cast<const half2*>(interaction_weights_fp16);
    for (int i = 0; i < arch.head_dim / 2; i++) {
        half2 h2 = inter_vec[i];
        interaction_weights_fp32[i * 2] = __half2float(h2.x);
        interaction_weights_fp32[i * 2 + 1] = __half2float(h2.y);
    }
    record_ca_interaction(tape, perception, interaction_weights_fp32, interaction,
                         head_id, arch.hidden_dim, arch.channels);

    half* value_weights_fp16 = &ca_heads->value_weights[head_id * arch.hidden_dim * arch.channels];
    float* value_weights_fp32;
    cudaError_t err = cudaMalloc(&value_weights_fp32, arch.hidden_dim * arch.channels * sizeof(float));
    if (err != cudaSuccess) {
        printf("[DEVICE-ERR] cudaMalloc failed in multi_head_ca_with_tape_kernel: %d\n", (int)err);
        return;
    }

    const half2* value_vec = reinterpret_cast<const half2*>(value_weights_fp16);
    for (int i = 0; i < (arch.hidden_dim * arch.channels) / 2; i++) {
        half2 h2 = value_vec[i];
        value_weights_fp32[i * 2] = __half2float(h2.x);
        value_weights_fp32[i * 2 + 1] = __half2float(h2.y);
    }
    for (int i = 0; i < arch.head_dim; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < arch.hidden_dim; j++) {
            int weight_idx = j * arch.channels + i;
            output[i] += interaction[j % arch.head_dim] * value_weights_fp32[weight_idx];
        }
    }
    cudaFree(value_weights_fp32);

    int out_idx = batch_id * arch.num_heads * grid_size * grid_size * arch.head_dim +
                  head_id * grid_size * grid_size * arch.head_dim +
                  cell_y * grid_size * arch.head_dim +
                  cell_x * arch.head_dim;

    for (int i = 0; i < arch.head_dim; i++) {
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

        if (fabsf(grad) > gradient_clip) {
            grad = copysignf(gradient_clip, grad);
        }

        if (is_fp16 && param_ptr_fp16 != nullptr) {
            float val = __half2float(*param_ptr_fp16);
            val -= learning_rate * grad;
            *param_ptr_fp16 = __float2half(val);
        } else if (param_ptr_fp32 != nullptr) {
            *param_ptr_fp32 -= learning_rate * grad;
        }

        tape->grad_buffer[tape_idx] = 0.0f;
    }
}

__global__ void init_ca_parameter_map_kernel(CAParameterMap* map, ArchitectureParams arch) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    map->perception_size = arch.channels * arch.hidden_dim;
    map->interaction_size = arch.channels * arch.hidden_dim;
    map->value_size = arch.hidden_dim * arch.channels;

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

    printf("[DEVICE] CA parameter map initialized: %d total parameters across %d heads\n",
           map->total_params, arch.num_heads);
}

#endif

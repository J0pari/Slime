
#ifndef AUTODIFF_CU
#define AUTODIFF_CU
#include "../config/config.cu"
#include "../core/organism.cu"
#include "../utils/cuda_primitives.cuh"
#include <cuda_runtime.h>

__device__ void init_ad_tape_device(Organism* organism) {
    ADTape* tape = organism->ad_tape;
    TapeEntry* entries_pool = organism->ad_entries_pool;
    float* values_pool = organism->ad_values_pool;
    float* grads_pool = organism->ad_grads_pool;
    int* levels_pool = organism->ad_levels_pool;
    int tape_capacity = organism->ad_tape_capacity;
    int value_capacity = organism->ad_value_capacity;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        tape->entries = entries_pool;
        tape->value_buffer = values_pool;
        tape->grad_buffer = grads_pool;
        tape->value_levels = levels_pool;
        tape->capacity = tape_capacity;
        tape->current_size = 0;
        tape->value_capacity = value_capacity;
        tape->current_value_idx = 0;
        tape->max_level = 0;
        tape->needs_weight_restore = 0;
        tape->restore_elite_idx = INT_MAX;
    }
}

__device__ int tape_record_unary(ADTape* tape, TapeOp op, int input_idx, float output_value, float aux_data) {
    int entry_idx = atomicAdd(&tape->current_size, 1);
    if (entry_idx >= tape->capacity) {
        return -1;
    }

    int output_idx = atomicAdd(&tape->current_value_idx, 1);
    if (output_idx >= tape->value_capacity) {
        return -1;
    }

    int input_level = (input_idx >= 0 && input_idx < tape->value_capacity) ? tape->value_levels[input_idx] : 0;
    int level = input_level + 1;

    tape->entries[entry_idx].op = op;
    tape->entries[entry_idx].output_idx = output_idx;
    tape->entries[entry_idx].input1_idx = input_idx;
    tape->entries[entry_idx].input2_idx = -1;
    tape->entries[entry_idx].aux_data = aux_data;
    tape->entries[entry_idx].level = level;

    tape->value_buffer[output_idx] = output_value;
    tape->grad_buffer[output_idx] = 0.0f;
    tape->value_levels[output_idx] = level;

    atomicMax(&tape->max_level, level);

    return output_idx;
}

__device__ int tape_record_binary(ADTape* tape, TapeOp op, int input1_idx, int input2_idx, float output_value) {
    int entry_idx = atomicAdd(&tape->current_size, 1);
    if (entry_idx >= tape->capacity) {
        return -1;
    }

    int output_idx = atomicAdd(&tape->current_value_idx, 1);
    if (output_idx >= tape->value_capacity) {
        return -1;
    }

    int level1 = (input1_idx >= 0 && input1_idx < tape->value_capacity) ? tape->value_levels[input1_idx] : 0;
    int level2 = (input2_idx >= 0 && input2_idx < tape->value_capacity) ? tape->value_levels[input2_idx] : 0;
    int level = max(level1, level2) + 1;

    tape->entries[entry_idx].op = op;
    tape->entries[entry_idx].output_idx = output_idx;
    tape->entries[entry_idx].input1_idx = input1_idx;
    tape->entries[entry_idx].input2_idx = input2_idx;
    tape->entries[entry_idx].aux_data = 0.0f;
    tape->entries[entry_idx].level = level;

    tape->value_buffer[output_idx] = output_value;
    tape->grad_buffer[output_idx] = 0.0f;
    tape->value_levels[output_idx] = level;

    atomicMax(&tape->max_level, level);

    return output_idx;
}

__device__ int ad_add(ADTape* tape, int x_idx, int y_idx) {
    float result = tape->value_buffer[x_idx] + tape->value_buffer[y_idx];
    return tape_record_binary(tape, OP_ADD, x_idx, y_idx, result);
}

__device__ int ad_mul(ADTape* tape, int x_idx, int y_idx) {
    float result = tape->value_buffer[x_idx] * tape->value_buffer[y_idx];
    return tape_record_binary(tape, OP_MUL, x_idx, y_idx, result);
}

__device__ int ad_tanh(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = tanhf(x);
    return tape_record_unary(tape, OP_TANH, x_idx, result, result);
}

__device__ int ad_exp(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = expf(x);
    return tape_record_unary(tape, OP_EXP, x_idx, result, result);
}

__device__ int ad_log(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    if (x <= 0.0f) {
        return -1;
    }
    float result = logf(x);
    return tape_record_unary(tape, OP_LOG, x_idx, result, x);
}

__device__ int ad_sqrt(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    if (x < 0.0f) {
        return -1;
    }
    float result = sqrtf(x);
    return tape_record_unary(tape, OP_SQRT, x_idx, result, result);
}

__device__ int ad_sin(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = sinf(x);
    return tape_record_unary(tape, OP_SIN, x_idx, result, x);
}

__device__ int ad_cos(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = cosf(x);
    return tape_record_unary(tape, OP_COS, x_idx, result, x);
}

__device__ int ad_relu(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = activation_relu(x);
    return tape_record_unary(tape, OP_RELU, x_idx, result, x);
}

__device__ void backward_op(TapeEntry* entry, float* value_buffer, float* grad_buffer) {
    float out_grad = grad_buffer[entry->output_idx];
    if (out_grad != 0.0f) {
        switch (entry->op) {
        case OP_ADD:
            atomicAdd(&grad_buffer[entry->input1_idx], out_grad);
            atomicAdd(&grad_buffer[entry->input2_idx], out_grad);
            break;
        case OP_MUL: {
            float x = value_buffer[entry->input1_idx];
            float y = value_buffer[entry->input2_idx];
            atomicAdd(&grad_buffer[entry->input1_idx], out_grad * y);
            atomicAdd(&grad_buffer[entry->input2_idx], out_grad * x);
            break;
        }
        case OP_TANH: {
            float tanh_x = entry->aux_data;
            atomicAdd(&grad_buffer[entry->input1_idx], out_grad * (1.0f - tanh_x * tanh_x));
            break;
        }
        case OP_RELU: {
            float x = entry->aux_data;
            if (x > 0.0f) {
                atomicAdd(&grad_buffer[entry->input1_idx], out_grad);
            }
            break;
        }
        case OP_EXP:
            atomicAdd(&grad_buffer[entry->input1_idx], out_grad * entry->aux_data);
            break;
        case OP_LOG:
            atomicAdd(&grad_buffer[entry->input1_idx], safe_div(out_grad, entry->aux_data));
            break;
        case OP_SQRT:
            atomicAdd(&grad_buffer[entry->input1_idx], safe_div(out_grad, 2.0f * entry->aux_data));
            break;
        case OP_SIN:
            atomicAdd(&grad_buffer[entry->input1_idx], out_grad * cosf(entry->aux_data));
            break;
        case OP_COS:
            atomicAdd(&grad_buffer[entry->input1_idx], -out_grad * sinf(entry->aux_data));
            break;
        default:
            break;
        }
    }
}

__device__ void ad_backward_device(Organism* organism) {
    ADTape* tape = organism->ad_tape;
    int output_idx = organism->ad_output_idx;
    float output_grad = organism->ad_output_grad;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int tape_size = tape->current_size;
    int max_level = tape->max_level;

    if (tid == 0) {
        tape->grad_buffer[output_idx] = output_grad;
    }
    cg::this_grid().sync();

    for (int level = max_level; level >= 1; level--) {
        for (int i = tid; i < tape_size; i += num_threads) {
            TapeEntry* entry = &tape->entries[i];
            if (entry->level == level) {
                backward_op(entry, tape->value_buffer, tape->grad_buffer);
            }
        }
        cg::this_grid().sync();
    }
}

__device__ void reset_tape_device(Organism* organism) {
    ADTape* tape = organism->ad_tape;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < tape->value_capacity) {
        tape->grad_buffer[tid] = 0.0f;
        tape->value_levels[tid] = 0;
    }

    if (tid == 0) {
        tape->current_size = 0;
        tape->current_value_idx = 0;
        tape->max_level = 0;
    }
}

__device__ void extract_genome_gradients_device(Organism* organism) {
    ADTape* tape = organism->ad_tape;
    int* genome_param_indices = organism->genome_param_indices;
    int num_params = organism->num_genome_params;
    float* output_gradients = organism->output_gradients;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_params) {
        int param_idx = genome_param_indices[tid];
        output_gradients[tid] = tape->grad_buffer[param_idx];
    }
}

__device__ void apply_gradients_device(Organism* organism) {
    int entry_idx = blockIdx.x;
    float* genome = &organism->workspace_genomes[entry_idx * GENOME_SIZE * 2];
    float* gradients = organism->output_gradients;
    int genome_size = GENOME_SIZE;
    float learning_rate = organism->learning_rate;
    float gradient_clip_norm = organism->gradient_clip_norm;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < genome_size) {
        float grad = gradients[tid];

        if (fabsf(grad) > gradient_clip_norm) {
            grad = copysignf(gradient_clip_norm, grad);
        }

        genome[tid] -= learning_rate * grad;

        genome[tid] = clamp(genome[tid], -1.0f, 1.0f);
    }
}

#endif

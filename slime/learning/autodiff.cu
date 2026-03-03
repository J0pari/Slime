
#ifndef AUTODIFF_CU
#define AUTODIFF_CU
#include "../config/config.cu"
#include "../core/organism.cu"
#include "../utils/cuda_primitives.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Per-entry tape initialization occurs in runtime.cu (init_organism_entries_device)

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

// Scalar backward: handles all ops except OP_MATMUL (which needs cooperative parallelism)
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

// OP_MATMUL backward: cooperative parallel across all threads in the grid.
// Metadata layout in value_buffer at entry->input2_idx:
//   [meta+0] = __int_as_float(num_inputs)
//   [meta+1] = __int_as_float(num_outputs)
//   [meta+2..3] = weight matrix pointer (8 bytes as 2 floats)
//   [meta+4..5] = bias pointer (8 bytes as 2 floats, unused in backward)
// Backward: dInput[j] += sum_o(dOutput[o] * W[j * O + o])
__device__ void backward_matmul_parallel(TapeEntry* entry, float* value_buffer, float* grad_buffer,
                                         int tid, int num_threads) {
    int meta = entry->input2_idx;
    int I = __float_as_int(value_buffer[meta]);
    int O = __float_as_int(value_buffer[meta + 1]);
    float* W;
    memcpy(&W, &value_buffer[meta + 2], sizeof(float*));

    int out_start = entry->output_idx;
    int in_start = entry->input1_idx;

    for (int j = tid; j < I; j += num_threads) {
        float grad_sum = 0.0f;
        for (int o = 0; o < O; o++) {
            grad_sum += grad_buffer[out_start + o] * W[j * O + o];
        }
        atomicAdd(&grad_buffer[in_start + j], grad_sum);
    }
}

// Tape-recorded DIRESA genome decode forward pass.
// All threads cooperate on MATMUL layers. Requires cooperative grid context.
// Caller must pre-load latent values into tape->value_buffer[0..output_dim-1].
// Deterministic value buffer layout (latent=output_dim, h2, h1, genome=input_dim):
//   [0..output_dim-1]:           latent leaves (level 0)
//   [output_dim..output_dim+5]:  meta1
//   [+6..+6+h2-1]:               layer1 linear (level 1)
//   [+h2..+2*h2-1]:              layer1 relu (level 2)
//   [...]:                        meta2, layer2 linear (level 3), relu (level 4)
//   [...]:                        meta3, layer3 output (level 5)
// Returns: value_buffer index of first genome output element.
__device__ int diresa_decode_taped_forward(ADTape* tape, const DIRESAWeights* weights,
                                           int tid, int num_threads) {
    int out_dim = weights->output_dim;   // latent dim (128)
    int h2 = weights->hidden2;           // 64
    int h1 = weights->hidden1;           // 128
    int in_dim = weights->input_dim;     // genome size (1024)

    // Pre-computed deterministic offsets
    int meta1 = out_dim;
    int lin1  = meta1 + 6;
    int relu1 = lin1 + h2;
    int meta2 = relu1 + h2;
    int lin2  = meta2 + 6;
    int relu2 = lin2 + h1;
    int meta3 = relu2 + h1;
    int lin3  = meta3 + 6;
    int total_values  = lin3 + in_dim;
    int total_entries = 3 + h2 + h1;

    // Phase 1: Thread 0 writes all tape entries and metadata
    if (tid == 0) {
        // MATMUL 1: latent → hidden2 (out_dim → h2)
        tape->entries[0].op = OP_MATMUL;
        tape->entries[0].input1_idx = 0;
        tape->entries[0].input2_idx = meta1;
        tape->entries[0].output_idx = lin1;
        tape->entries[0].aux_data = 0.0f;
        tape->entries[0].level = 1;

        tape->value_buffer[meta1]     = __int_as_float(out_dim);
        tape->value_buffer[meta1 + 1] = __int_as_float(h2);
        memcpy(&tape->value_buffer[meta1 + 2], &weights->decoder_w1, sizeof(float*));
        memcpy(&tape->value_buffer[meta1 + 4], &weights->decoder_b1, sizeof(float*));

        // RELU entries for layer 1
        for (int i = 0; i < h2; i++) {
            int eidx = 1 + i;
            tape->entries[eidx].op = OP_RELU;
            tape->entries[eidx].input1_idx = lin1 + i;
            tape->entries[eidx].input2_idx = -1;
            tape->entries[eidx].output_idx = relu1 + i;
            tape->entries[eidx].aux_data = 0.0f;
            tape->entries[eidx].level = 2;
        }

        // MATMUL 2: hidden2 → hidden1 (h2 → h1)
        int mm2 = 1 + h2;
        tape->entries[mm2].op = OP_MATMUL;
        tape->entries[mm2].input1_idx = relu1;
        tape->entries[mm2].input2_idx = meta2;
        tape->entries[mm2].output_idx = lin2;
        tape->entries[mm2].aux_data = 0.0f;
        tape->entries[mm2].level = 3;

        tape->value_buffer[meta2]     = __int_as_float(h2);
        tape->value_buffer[meta2 + 1] = __int_as_float(h1);
        memcpy(&tape->value_buffer[meta2 + 2], &weights->decoder_w2, sizeof(float*));
        memcpy(&tape->value_buffer[meta2 + 4], &weights->decoder_b2, sizeof(float*));

        // RELU entries for layer 2
        for (int i = 0; i < h1; i++) {
            int eidx = mm2 + 1 + i;
            tape->entries[eidx].op = OP_RELU;
            tape->entries[eidx].input1_idx = lin2 + i;
            tape->entries[eidx].input2_idx = -1;
            tape->entries[eidx].output_idx = relu2 + i;
            tape->entries[eidx].aux_data = 0.0f;
            tape->entries[eidx].level = 4;
        }

        // MATMUL 3: hidden1 → genome (h1 → in_dim), no activation
        int mm3 = mm2 + 1 + h1;
        tape->entries[mm3].op = OP_MATMUL;
        tape->entries[mm3].input1_idx = relu2;
        tape->entries[mm3].input2_idx = meta3;
        tape->entries[mm3].output_idx = lin3;
        tape->entries[mm3].aux_data = 0.0f;
        tape->entries[mm3].level = 5;

        tape->value_buffer[meta3]     = __int_as_float(h1);
        tape->value_buffer[meta3 + 1] = __int_as_float(in_dim);
        memcpy(&tape->value_buffer[meta3 + 2], &weights->decoder_w3, sizeof(float*));
        memcpy(&tape->value_buffer[meta3 + 4], &weights->decoder_b3, sizeof(float*));

        tape->current_size = total_entries;
        tape->current_value_idx = total_values;
        tape->max_level = 5;

        // Set value_levels
        for (int i = 0; i < out_dim; i++) tape->value_levels[i] = 0;
        for (int i = 0; i < h2; i++) { tape->value_levels[lin1 + i] = 1; tape->value_levels[relu1 + i] = 2; }
        for (int i = 0; i < h1; i++) { tape->value_levels[lin2 + i] = 3; tape->value_levels[relu2 + i] = 4; }
        for (int i = 0; i < in_dim; i++) tape->value_levels[lin3 + i] = 5;

        // Zero grad_buffer for all value slots
        for (int i = 0; i < total_values; i++) tape->grad_buffer[i] = 0.0f;
    }
    cg::this_grid().sync();

    // Phase 2: Cooperative forward computation

    // Layer 1 matmul: Y[i] = B1[i] + sum_j(input[j] * W1[j * h2 + i])
    for (int i = tid; i < h2; i += num_threads) {
        float sum = weights->decoder_b1[i];
        for (int j = 0; j < out_dim; j++) {
            sum += tape->value_buffer[j] * weights->decoder_w1[j * h2 + i];
        }
        tape->value_buffer[lin1 + i] = sum;
    }
    cg::this_grid().sync();

    // Layer 1 relu + store pre-relu in aux_data for backward
    for (int i = tid; i < h2; i += num_threads) {
        float x = tape->value_buffer[lin1 + i];
        tape->value_buffer[relu1 + i] = (x > 0.0f) ? x : 0.0f;
        tape->entries[1 + i].aux_data = x;
    }
    cg::this_grid().sync();

    // Layer 2 matmul
    for (int i = tid; i < h1; i += num_threads) {
        float sum = weights->decoder_b2[i];
        for (int j = 0; j < h2; j++) {
            sum += tape->value_buffer[relu1 + j] * weights->decoder_w2[j * h1 + i];
        }
        tape->value_buffer[lin2 + i] = sum;
    }
    cg::this_grid().sync();

    // Layer 2 relu
    for (int i = tid; i < h1; i += num_threads) {
        float x = tape->value_buffer[lin2 + i];
        tape->value_buffer[relu2 + i] = (x > 0.0f) ? x : 0.0f;
        tape->entries[1 + h2 + 1 + i].aux_data = x;
    }
    cg::this_grid().sync();

    // Layer 3 matmul (no activation — final output)
    for (int i = tid; i < in_dim; i += num_threads) {
        float sum = weights->decoder_b3[i];
        for (int j = 0; j < h1; j++) {
            sum += tape->value_buffer[relu2 + j] * weights->decoder_w3[j * in_dim + i];
        }
        tape->value_buffer[lin3 + i] = sum;
    }
    cg::this_grid().sync();

    return lin3;
}

__device__ void ad_backward_device(ADTape* tape, int output_idx, float output_grad) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int tape_size = tape->current_size;
    int max_level = tape->max_level;

    if (tid == 0) {
        tape->grad_buffer[output_idx] = output_grad;
    }
    cg::this_grid().sync();

    for (int level = max_level; level >= 1; level--) {
        // Pass 1: scalar ops — distributed one entry per thread
        for (int i = tid; i < tape_size; i += num_threads) {
            TapeEntry* entry = &tape->entries[i];
            if (entry->level == level && entry->op != OP_MATMUL) {
                backward_op(entry, tape->value_buffer, tape->grad_buffer);
            }
        }

        // Pass 2: MATMUL ops — ALL threads cooperate on each entry
        for (int i = 0; i < tape_size; i++) {
            TapeEntry* entry = &tape->entries[i];
            if (entry->level == level && entry->op == OP_MATMUL) {
                backward_matmul_parallel(entry, tape->value_buffer, tape->grad_buffer, tid, num_threads);
            }
        }

        cg::this_grid().sync();
    }
}

__device__ void reset_tape_device(ADTape* tape, int tid) {
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


#endif

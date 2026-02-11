
#include "../../config/config.cu"
#include "../../core/organism.cu"
#include "../../learning/autodiff.cu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
\
            exit(1); \
        } \
    } while(0)

struct TestResult {
    bool passed;
    float measured_value;
    float expected_value;
    float tolerance;
    const char* test_name;
};

__global__ void extract_tape_state_kernel(
    ADTape* tape,
    int* current_size_out,
    int* current_value_idx_out,
    float* first_grad_out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *current_size_out = tape->current_size;
        *current_value_idx_out = tape->current_value_idx;
        if (first_grad_out && tape->current_value_idx > 0) {
            *first_grad_out = tape->grad_buffer[0];
        }
    }
}

bool test_tape_initialization() {
    ADTape* d_tape;
    CUDA_CHECK(cudaMalloc(&d_tape, sizeof(ADTape)));

    TapeEntry* entries;
    float* values;
    float* grads;
    int* levels;
    CUDA_CHECK(cudaMalloc(&entries, TAPE_CAPACITY * sizeof(TapeEntry)));
    CUDA_CHECK(cudaMalloc(&values, VALUE_CAPACITY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads, VALUE_CAPACITY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&levels, VALUE_CAPACITY * sizeof(int)));
    CUDA_CHECK(cudaMemset(levels, 0, VALUE_CAPACITY * sizeof(int)));

    ADTape h_tape;
    h_tape.entries = entries;
    h_tape.value_buffer = values;
    h_tape.grad_buffer = grads;
    h_tape.value_levels = levels;
    h_tape.capacity = TAPE_CAPACITY;
    h_tape.value_capacity = VALUE_CAPACITY;
    h_tape.current_size = 0;
    h_tape.current_value_idx = 0;
    h_tape.max_level = 0;
    CUDA_CHECK(cudaMemcpy(d_tape, &h_tape, sizeof(ADTape), cudaMemcpyHostToDevice));

    init_ad_tape_kernel<<<1, 1>>>(d_tape, entries, values, grads, levels, TAPE_CAPACITY, VALUE_CAPACITY);
    cudaDeviceSynchronize();

    int* d_size;
    int* d_value_idx;
    CUDA_CHECK(cudaMalloc(&d_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_value_idx, sizeof(int)));

    extract_tape_state_kernel<<<1, 1>>>(d_tape, d_size, d_value_idx, nullptr);
    cudaDeviceSynchronize();

    int h_size, h_value_idx;
    CUDA_CHECK(cudaMemcpy(&h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_value_idx, d_value_idx, sizeof(int), cudaMemcpyDeviceToHost));

    bool passed = (h_size == 0) && (h_value_idx == 0);

    cudaFree(d_size);
    cudaFree(d_value_idx);
    cudaFree(entries);
    cudaFree(values);
    cudaFree(grads);
    cudaFree(d_tape);

    return passed;
}

__global__ void record_single_op_kernel(ADTape* tape) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        float x = 0.5f;
        float y = tanhf(x);
        int y_idx = tape_record_unary(tape, OP_TANH, 0, y, y);
    }
}

bool test_tape_record_single_op() {
    ADTape* d_tape;
    CUDA_CHECK(cudaMalloc(&d_tape, sizeof(ADTape)));

    TapeEntry* entries;
    float* values;
    float* grads;
    int* levels;
    CUDA_CHECK(cudaMalloc(&entries, TAPE_CAPACITY * sizeof(TapeEntry)));
    CUDA_CHECK(cudaMalloc(&values, VALUE_CAPACITY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads, VALUE_CAPACITY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&levels, VALUE_CAPACITY * sizeof(int)));
    CUDA_CHECK(cudaMemset(levels, 0, VALUE_CAPACITY * sizeof(int)));

    ADTape h_tape;
    h_tape.entries = entries;
    h_tape.value_buffer = values;
    h_tape.grad_buffer = grads;
    h_tape.value_levels = levels;
    h_tape.capacity = TAPE_CAPACITY;
    h_tape.value_capacity = VALUE_CAPACITY;
    h_tape.current_size = 0;
    h_tape.current_value_idx = 0;
    h_tape.max_level = 0;
    CUDA_CHECK(cudaMemcpy(d_tape, &h_tape, sizeof(ADTape), cudaMemcpyHostToDevice));

    init_ad_tape_kernel<<<1, 1>>>(d_tape, entries, values, grads, levels, TAPE_CAPACITY, VALUE_CAPACITY);
    cudaDeviceSynchronize();

    record_single_op_kernel<<<1, 1>>>(d_tape);
    cudaDeviceSynchronize();

    int* d_size;
    int* d_value_idx;
    CUDA_CHECK(cudaMalloc(&d_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_value_idx, sizeof(int)));

    extract_tape_state_kernel<<<1, 1>>>(d_tape, d_size, d_value_idx, nullptr);
    cudaDeviceSynchronize();

    int h_size, h_value_idx;
    CUDA_CHECK(cudaMemcpy(&h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_value_idx, d_value_idx, sizeof(int), cudaMemcpyDeviceToHost));

    bool passed = (h_size == 1) && (h_value_idx == 1);

    cudaFree(d_size);
    cudaFree(d_value_idx);
    cudaFree(entries);
    cudaFree(values);
    cudaFree(grads);
    cudaFree(d_tape);

    return passed;
}

__global__ void forward_test_kernel(ADTape* tape, int* y_idx_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float x = 3.0f;

        int x_idx = atomicAdd(&tape->current_value_idx, 1);
        tape->value_buffer[x_idx] = x;
        tape->grad_buffer[x_idx] = 0.0f;

        float y = x * x;
        int y_idx = tape_record_binary(tape, OP_MUL, x_idx, x_idx, y);
        *y_idx_out = y_idx;

    }
}

bool test_backward_gradient_computation() {
    ADTape* d_tape;
    CUDA_CHECK(cudaMalloc(&d_tape, sizeof(ADTape)));

    TapeEntry* entries;
    float* values;
    float* grads;
    int* levels;
    CUDA_CHECK(cudaMalloc(&entries, TAPE_CAPACITY * sizeof(TapeEntry)));
    CUDA_CHECK(cudaMalloc(&values, VALUE_CAPACITY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads, VALUE_CAPACITY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&levels, VALUE_CAPACITY * sizeof(int)));
    CUDA_CHECK(cudaMemset(levels, 0, VALUE_CAPACITY * sizeof(int)));

    ADTape h_tape;
    h_tape.entries = entries;
    h_tape.value_buffer = values;
    h_tape.grad_buffer = grads;
    h_tape.value_levels = levels;
    h_tape.capacity = TAPE_CAPACITY;
    h_tape.value_capacity = VALUE_CAPACITY;
    h_tape.current_size = 0;
    h_tape.current_value_idx = 0;
    h_tape.max_level = 0;
    CUDA_CHECK(cudaMemcpy(d_tape, &h_tape, sizeof(ADTape), cudaMemcpyHostToDevice));

    init_ad_tape_kernel<<<1, 1>>>(d_tape, entries, values, grads, levels, TAPE_CAPACITY, VALUE_CAPACITY);
    cudaDeviceSynchronize();

    int* d_y_idx;
    CUDA_CHECK(cudaMalloc(&d_y_idx, sizeof(int)));

    forward_test_kernel<<<1, 1>>>(d_tape, d_y_idx);
    cudaDeviceSynchronize();

    int h_y_idx;
    CUDA_CHECK(cudaMemcpy(&h_y_idx, d_y_idx, sizeof(int), cudaMemcpyDeviceToHost));

    ad_backward_kernel<<<1, 32>>>(d_tape, h_y_idx, 1.0f);
    cudaDeviceSynchronize();

    float h_grad;
    float* d_grad_buffer;
    CUDA_CHECK(cudaMemcpy(&d_grad_buffer, &(((ADTape*)d_tape)->grad_buffer), sizeof(void*), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_grad, &d_grad_buffer[0], sizeof(float), cudaMemcpyDeviceToHost));

    float expected = 6.0f;
    bool passed = fabsf(h_grad - expected) < safe_epsilon(expected);

    cudaFree(d_y_idx);
    cudaFree(entries);
    cudaFree(values);
    cudaFree(grads);
    cudaFree(d_tape);

    return passed;
}

bool test_tape_reset() {
    ADTape* d_tape;
    CUDA_CHECK(cudaMalloc(&d_tape, sizeof(ADTape)));

    TapeEntry* entries;
    float* values;
    float* grads;
    int* levels;
    CUDA_CHECK(cudaMalloc(&entries, TAPE_CAPACITY * sizeof(TapeEntry)));
    CUDA_CHECK(cudaMalloc(&values, VALUE_CAPACITY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads, VALUE_CAPACITY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&levels, VALUE_CAPACITY * sizeof(int)));
    CUDA_CHECK(cudaMemset(levels, 0, VALUE_CAPACITY * sizeof(int)));

    ADTape h_tape;
    h_tape.entries = entries;
    h_tape.value_buffer = values;
    h_tape.grad_buffer = grads;
    h_tape.value_levels = levels;
    h_tape.capacity = TAPE_CAPACITY;
    h_tape.value_capacity = VALUE_CAPACITY;
    h_tape.current_size = 0;
    h_tape.current_value_idx = 0;
    h_tape.max_level = 0;
    CUDA_CHECK(cudaMemcpy(d_tape, &h_tape, sizeof(ADTape), cudaMemcpyHostToDevice));

    init_ad_tape_kernel<<<1, 1>>>(d_tape, entries, values, grads, levels, TAPE_CAPACITY, VALUE_CAPACITY);
    record_single_op_kernel<<<1, 1>>>(d_tape);
    cudaDeviceSynchronize();

    reset_tape_kernel<<<(VALUE_CAPACITY  + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_tape);
    cudaDeviceSynchronize();

    int* d_size;
    int* d_value_idx;
    CUDA_CHECK(cudaMalloc(&d_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_value_idx, sizeof(int)));

    extract_tape_state_kernel<<<1, 1>>>(d_tape, d_size, d_value_idx, nullptr);
    cudaDeviceSynchronize();

    int h_size, h_value_idx;
    CUDA_CHECK(cudaMemcpy(&h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_value_idx, d_value_idx, sizeof(int), cudaMemcpyDeviceToHost));

    bool passed = (h_size == 0) && (h_value_idx == 0);

    cudaFree(d_size);
    cudaFree(d_value_idx);
    cudaFree(entries);
    cudaFree(values);
    cudaFree(grads);
    cudaFree(d_tape);

    return passed;
}

int main() {
    int passed = 0;
    int total = 0;

    total++; if (test_tape_initialization()) passed++;
    total++; if (test_tape_record_single_op()) passed++;
    total++; if (test_backward_gradient_computation()) passed++;
    total++; if (test_tape_reset()) passed++;

    return (passed == total) ? 0 : 1;
}

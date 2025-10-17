#pragma once
#include <cuda_runtime.h>

struct ModelProtocol {
    virtual __device__ void forward(float* input, float* output) = 0;
    virtual __device__ float effective_rank() = 0;
    virtual __device__ float coherence() = 0;
    virtual __device__ void reset() = 0;
};

struct Pseudopod : ModelProtocol {
    float* ca_weights;
    float* ca_state;
    float* learning_history;
    int grid_size;
    int history_len;
    float mass_before;

    __device__ void forward(float* input, float* output) override {
        extern __shared__ float tile[];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int gx = blockIdx.x * blockDim.x + tx;
        int gy = blockIdx.y * blockDim.y + ty;

        if (gx < grid_size && gy < grid_size) {
            int idx = gy * grid_size + gx;
            tile[(ty + 1) * (blockDim.x + 2) + (tx + 1)] = input[idx];
            mass_before += input[idx];
        }

        if (tx == 0 && gx > 0) {
            tile[(ty + 1) * (blockDim.x + 2)] = input[gy * grid_size + (gx - 1)];
        }
        if (tx == blockDim.x - 1 && gx < grid_size - 1) {
            tile[(ty + 1) * (blockDim.x + 2) + (tx + 2)] = input[gy * grid_size + (gx + 1)];
        }
        __syncthreads();

        float sum = 0.0f;
        #pragma unroll
        for (int dy = -1; dy <= 1; dy++) {
            #pragma unroll
            for (int dx = -1; dx <= 1; dx++) {
                sum += tile[(ty + 1 + dy) * (blockDim.x + 2) + (tx + 1 + dx)] *
                       ca_weights[(dy + 1) * 3 + (dx + 1)];
            }
        }

        float mu = ca_weights[9];
        float sigma = ca_weights[10];
        float growth = expf(-(sum - mu) * (sum - mu) / (2.0f * sigma * sigma));

        if (gx < grid_size && gy < grid_size) {
            int idx = gy * grid_size + gx;
            output[idx] = input[idx] * growth;
        }
    }

    __device__ float effective_rank() override {
        float trace = 0.0f;
        float frob_sq = 0.0f;

        for (int i = 0; i < grid_size * grid_size; i++) {
            float val = ca_state[i];
            trace += val;
            frob_sq += val * val;
        }

        return (trace * trace) / (frob_sq + 1e-10f);
    }

    __device__ float coherence() override {
        float sum_x = 0.0f, sum_y = 0.0f, sum_xx = 0.0f, sum_xy = 0.0f;

        for (int i = threadIdx.x; i < history_len; i += blockDim.x) {
            float x = (float)i;
            float y = learning_history[i];
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_x += __shfl_down_sync(0xFFFFFFFF, sum_x, offset);
            sum_y += __shfl_down_sync(0xFFFFFFFF, sum_y, offset);
            sum_xx += __shfl_down_sync(0xFFFFFFFF, sum_xx, offset);
            sum_xy += __shfl_down_sync(0xFFFFFFFF, sum_xy, offset);
        }

        float n = (float)history_len;
        float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x + 1e-10f);
        float learning_progress = -slope;

        return 1.0f / (1.0f + expf(-learning_progress * 10.0f));
    }

    __device__ void reset() override {
        for (int i = threadIdx.x; i < grid_size * grid_size; i += blockDim.x) {
            ca_state[i] = 0.0f;
        }
        mass_before = 0.0f;
    }
};
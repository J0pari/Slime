
#ifndef LOSSES_CU
#define LOSSES_CU

#include "../config/config.cu"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void mse_loss_kernel(
    float* predictions,
    float* targets,
    float* loss_out,
    int batch_size,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim) return;

    float diff = predictions[idx] - targets[idx];
    float squared_error = diff * diff;

    atomicAdd(loss_out, squared_error / (float)(batch_size * dim));
}

#endif

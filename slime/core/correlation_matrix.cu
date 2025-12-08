#ifndef CORRELATION_MATRIX_CU
#define CORRELATION_MATRIX_CU

#include "../config/config.cu"
#include "../utils/tile_ops.cuh"
#include <cuda_runtime.h>

__global__ void compute_correlation_matrix_kernel(
    float* genome,
    float* correlation_matrix,
    int genome_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= genome_size || col >= genome_size) return;

    int window = 16;
    int half_window = window / 2;

    float sum_row = 0.0f;
    float sum_col = 0.0f;
    int count = 0;

    for (int offset = -half_window; offset <= half_window; offset++) {
        int idx_row = row + offset;
        int idx_col = col + offset;
        if (idx_row >= 0 && idx_row < genome_size && idx_col >= 0 && idx_col < genome_size) {
            sum_row += genome[idx_row];
            sum_col += genome[idx_col];
            count++;
        }
    }

    float mean_row = sum_row / count;
    float mean_col = sum_col / count;

    float covariance = 0.0f;
    float var_row = 0.0f;
    float var_col = 0.0f;

    for (int offset = -half_window; offset <= half_window; offset++) {
        int idx_row = row + offset;
        int idx_col = col + offset;
        if (idx_row >= 0 && idx_row < genome_size && idx_col >= 0 && idx_col < genome_size) {
            float val_row = genome[idx_row] - mean_row;
            float val_col = genome[idx_col] - mean_col;
            covariance += val_row * val_col;
            var_row += val_row * val_row;
            var_col += val_col * val_col;
        }
    }

    float correlation = covariance / (sqrtf(var_row * var_col) + EPSILON);
    correlation_matrix[row * genome_size + col] = correlation;
}

#endif


#include "../../config/config.cu"
#include "../../core/organism.cu"
#include "../../core/chemotaxis.cu"
#include "../../lifecycle/lifecycle_stages.cu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void extract_rd_stats_kernel(
    float* rd_field,
    int grid_size,
    float* min_out,
    float* max_out,
    float* mean_out,
    int* nan_count_out
) {
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ int s_nan_count;

    if (threadIdx.x == 0) {
        s_min = 1e10f;
        s_max = -1e10f;
        s_sum = 0.0f;
        s_nan_count = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grid_size * grid_size;

    if (idx < total) {
        float val = rd_field[idx];
        if (isnan(val) || isinf(val)) {
            atomicAdd(&s_nan_count, 1);
        } else {
            atomicMin((int*)&s_min, __float_as_int(val));
            atomicMax((int*)&s_max, __float_as_int(val));
            atomicAdd(&s_sum, val);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMin((int*)min_out, __float_as_int(s_min));
        atomicMax((int*)max_out, __float_as_int(s_max));
        atomicAdd(mean_out, s_sum);
        atomicAdd(nan_count_out, s_nan_count);
    }
}

bool test_rd_initialization() {

    float* d_u_field;
    float* d_v_field;
    CUDA_CHECK(cudaMalloc(&d_u_field, GRID_SIZE * GRID_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_field, GRID_SIZE * GRID_SIZE * sizeof(float)));

    dim3 grid(GRID_SIZE/WMMA_TILE_DIM, GRID_SIZE/WMMA_TILE_DIM);
    dim3 block(WMMA_TILE_DIM, WMMA_TILE_DIM);

    init_rd_fields_kernel<<<grid, block>>>(d_u_field, d_v_field, GRID_SIZE, 42);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_u_min, *d_u_max, *d_u_mean, *d_v_min, *d_v_max, *d_v_mean;
    int *d_u_nan, *d_v_nan;

    CUDA_CHECK(cudaMalloc(&d_u_min, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_max, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_mean, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_nan, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_v_min, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_max, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_mean, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_nan, sizeof(int)));

    float init_min = 1e10f, init_max = -1e10f, init_mean = 0.0f;
    int init_nan = 0;
    CUDA_CHECK(cudaMemcpy(d_u_min, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_mean, &init_mean, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_nan, &init_nan, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_min, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_mean, &init_mean, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_nan, &init_nan, sizeof(int), cudaMemcpyHostToDevice));

    extract_rd_stats_kernel<<<256, BLOCK_SIZE>>>(d_u_field, GRID_SIZE, d_u_min, d_u_max, d_u_mean, d_u_nan);
    extract_rd_stats_kernel<<<256, BLOCK_SIZE>>>(d_v_field, GRID_SIZE, d_v_min, d_v_max, d_v_mean, d_v_nan);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_u_min, h_u_max, h_u_mean, h_v_min, h_v_max, h_v_mean;
    int h_u_nan, h_v_nan;
    CUDA_CHECK(cudaMemcpy(&h_u_min, d_u_min, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_u_max, d_u_max, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_u_mean, d_u_mean, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_u_nan, d_u_nan, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_v_min, d_v_min, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_v_max, d_v_max, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_v_mean, d_v_mean, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_v_nan, d_v_nan, sizeof(int), cudaMemcpyDeviceToHost));

    h_u_mean /= (GRID_SIZE * GRID_SIZE);
    h_v_mean /= (GRID_SIZE * GRID_SIZE);

    bool u_valid = (h_u_nan == 0) && (fabsf(h_u_mean - 1.0f) < 0.1f);
    bool v_valid = (h_v_nan == 0) && (h_v_mean >= 0.0f) && (h_v_mean < 0.3f) && (h_v_max > 0.0f);

    printf("[rd_init] u_mean=%.4f v_mean=%.4f u_nan=%d v_nan=%d %s\n",
           h_u_mean, h_v_mean, h_u_nan, h_v_nan,
           (u_valid && v_valid) ? "OK" : "FAIL");

    cudaFree(d_u_field);
    cudaFree(d_v_field);
    cudaFree(d_u_min);
    cudaFree(d_u_max);
    cudaFree(d_u_mean);
    cudaFree(d_u_nan);
    cudaFree(d_v_min);
    cudaFree(d_v_max);
    cudaFree(d_v_mean);
    cudaFree(d_v_nan);

    return u_valid && v_valid;
}

bool test_chemical_field_initialization() {

    float* d_chemical_field;
    float* d_genome;
    CUDA_CHECK(cudaMalloc(&d_chemical_field, GRID_SIZE * GRID_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_genome, GENOME_SIZE * sizeof(float)));

    float h_genome[GENOME_SIZE];
    for (int i = 0; i < GENOME_SIZE; i++) {
        h_genome[i] = (float)i / (float)GENOME_SIZE;
    }
    CUDA_CHECK(cudaMemcpy(d_genome, h_genome, GENOME_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid((GRID_SIZE + 15) / 16, (GRID_SIZE + 15) / 16);
    dim3 block(16, 16);

    initialize_chemical_field_kernel<<<grid, block>>>(d_chemical_field, d_genome, GRID_SIZE, 12345);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_field = new float[GRID_SIZE * GRID_SIZE];
    CUDA_CHECK(cudaMemcpy(h_field, d_chemical_field, GRID_SIZE * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    int nan_count = 0;
    int valid_range = 0;
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        if (isnan(h_field[i])) {
            nan_count++;
        } else {
            sum += h_field[i];
            if (h_field[i] >= 0.0f && h_field[i] <= 1.0f) {
                valid_range++;
            }
        }
    }
    float mean = sum / (float)(GRID_SIZE * GRID_SIZE);

    bool valid = (nan_count == 0) && (valid_range == GRID_SIZE * GRID_SIZE) && (mean > 0.3f && mean < 0.7f);

    printf("[chemical_init] mean=%.6f range_valid=%d/%d nan=%d %s\n",
           mean, valid_range, GRID_SIZE * GRID_SIZE, nan_count, valid ? "OK" : "FAIL");

    delete[] h_field;
    cudaFree(d_chemical_field);
    cudaFree(d_genome);

    return valid;
}

bool test_resource_flow() {

    float* d_resource;
    float* d_resource_next;
    float* d_fitness_landscape;
    CUDA_CHECK(cudaMalloc(&d_resource, GRID_SIZE * GRID_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_resource_next, GRID_SIZE * GRID_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fitness_landscape, GRID_SIZE * GRID_SIZE * sizeof(float)));

    dim3 grid(GRID_SIZE/WMMA_TILE_DIM, GRID_SIZE/WMMA_TILE_DIM);
    dim3 block(WMMA_TILE_DIM, WMMA_TILE_DIM);

    init_resource_fields_kernel<<<grid, block>>>(d_resource, d_fitness_landscape, GRID_SIZE, 456);
    CUDA_CHECK(cudaDeviceSynchronize());

    resource_flow_kernel<<<grid, block>>>(d_resource, d_resource_next, d_fitness_landscape, GRID_SIZE, 0.01f, 0.1f, 0.5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_min, *d_max, *d_mean;
    int *d_nan;
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nan, sizeof(int)));

    float init_min = 1e10f, init_max = -1e10f, init_mean = 0.0f;
    int init_nan = 0;
    CUDA_CHECK(cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean, &init_mean, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nan, &init_nan, sizeof(int), cudaMemcpyHostToDevice));

    extract_rd_stats_kernel<<<256, BLOCK_SIZE>>>(d_resource_next, GRID_SIZE, d_min, d_max, d_mean, d_nan);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_min, h_max, h_mean;
    int h_nan;
    CUDA_CHECK(cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_nan, d_nan, sizeof(int), cudaMemcpyDeviceToHost));

    h_mean /= (GRID_SIZE * GRID_SIZE);

    bool valid = (h_nan == 0) && (h_min >= 0.0f) && (h_mean > 0.0f);

    printf("[resource_flow] min=%.4f max=%.4f mean=%.4f nan=%d %s\n",
           h_min, h_max, h_mean, h_nan, valid ? "OK" : "FAIL");

    cudaFree(d_resource);
    cudaFree(d_resource_next);
    cudaFree(d_fitness_landscape);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_mean);
    cudaFree(d_nan);

    return valid;
}

int main() {
    int passed = 0;
    int total = 0;

    total++; if (test_rd_initialization()) passed++;
    total++; if (test_chemical_field_initialization()) passed++;
    total++; if (test_resource_flow()) passed++;

    printf("%d/%d passed\n", passed, total);

    return (passed == total) ? 0 : 1;
}

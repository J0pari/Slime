// Minimal test to verify CUDA compilation works
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void simple_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}

int main() {
    printf("Testing minimal CUDA compilation...\n");

    // Allocate and test
    float* d_data;
    int size = 1024;
    cudaMalloc(&d_data, size * sizeof(float));

    // Launch kernel
    simple_kernel<<<(size + 255) / 256, 256>>>(d_data, size);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Success! CUDA is working.\n");

    cudaFree(d_data);
    return 0;
}
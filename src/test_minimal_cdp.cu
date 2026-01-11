#include <stdio.h>
#include <cuda_runtime.h>

__global__ void child_kernel(int value) {
}

__global__ void parent_kernel(int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        child_kernel<<<1, 4>>>(value + 100);
        cudaDeviceSynchronize();
    }
}

int main() {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    parent_kernel<<<1, 1>>>(42);

    cudaError_t err = cudaDeviceSynchronize();

    return err == cudaSuccess ? 0 : 1;
}

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void child_kernel(int value) {
    printf("[CHILD] value=%d tid=%d\n", value, threadIdx.x);
}

__global__ void parent_kernel(int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[PARENT] Launching child kernel with value=%d\n", value);
        child_kernel<<<1, 4>>>(value + 100);
        cudaDeviceSynchronize();
        printf("[PARENT] Child kernel completed\n");
    }
}

int main() {
    printf("[HOST] Testing CUDA Dynamic Parallelism...\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[HOST] Device: %s, Compute: %d.%d\n", prop.name, prop.major, prop.minor);

    printf("[HOST] Launching parent kernel...\n");
    parent_kernel<<<1, 1>>>(42);

    cudaError_t err = cudaDeviceSynchronize();
    printf("[HOST] Result: %s\n", cudaGetErrorString(err));

    return err == cudaSuccess ? 0 : 1;
}

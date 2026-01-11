#include <stdio.h>
#include <cuda_runtime.h>

struct Organism;
struct HybridTrainingMode;
struct CAParameterMap;

extern "C" __global__ void hybrid_organism_lifecycle_kernel(Organism*, HybridTrainingMode*, CAParameterMap*, int);

__global__ void test_wrapper_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
    }
}

int main() {
    test_wrapper_kernel<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return 1;
    }

    cudaFuncAttributes attr;
    err = cudaFuncGetAttributes(&attr, hybrid_organism_lifecycle_kernel);
    if (err == cudaSuccess) {
    } else {
        return 1;
    }

    return 0;
}

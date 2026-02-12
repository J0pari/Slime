#include "../config/config.cu"
#include "../core/organism.cu"
#include <stdio.h>
#include <cuda_runtime.h>

extern "C" __global__ void hybrid_organism_lifecycle_kernel(Organism*, HybridTrainingMode*, CAParameterMap*, int, float*, bool, AuditBuffer*);

__device__ void test_wrapper_device(Organism* organism) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
    }
}

int main() {
    test_wrapper_kernel<<<1, 1>>>();
    CUDA_LAUNCH_CHECK_VAL(cudaDeviceSynchronize());

    cudaFuncAttributes attr;
    CUDA_LAUNCH_CHECK_VAL(cudaFuncGetAttributes(&attr, hybrid_organism_lifecycle_kernel));

    return 0;
}

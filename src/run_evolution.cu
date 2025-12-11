#include <cuda_runtime.h>
#include <stdio.h>
#include "../slime/core/organism.cu"

int main(int argc, char** argv) {
    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    printf("[GPU] %s (SM %d.%d, %d MPs, %.1f GB)\n\n",
           props.name, props.major, props.minor,
           props.multiProcessorCount,
           props.totalGlobalMem / 1e9);

    int max_generations = (argc > 1) ? atoi(argv[1]) : 100000;

    printf("[EVOLUTION] Starting %d generations\n", max_generations);

    Organism* d_organism;
    cudaMalloc(&d_organism, sizeof(Organism));

    unsigned int seed = 12345;
    init_organism_kernel<<<1, 1>>>(d_organism, nullptr, seed);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR init: %s\n", cudaGetErrorString(err));
        return 1;
    }

    for (int gen = 0; gen < max_generations; gen++) {
        organism_lifecycle_kernel<<<1, 1>>>(d_organism, gen);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("ERROR gen %d: %s\n", gen, cudaGetErrorString(err));
            break;
        }
    }

    printf("\n[SHUTDOWN] Evolution complete\n");
    cudaFree(d_organism);

    return 0;
}

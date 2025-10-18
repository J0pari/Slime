// main.cu - Minimal launcher for Slime Mold Transformer
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "slime/runtime.cu"

int main(int argc, char** argv) {
    printf("[INIT] Neural CA Transformer\n");
    printf("[EXEC] Beginning computation\n\n");

    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to set CUDA device (code %d): %s\n", err, cudaGetErrorString(err));
        return 1;
    }

    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to get CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device: %s\n", props.name);
    printf("Compute capability: %d.%d\n", props.major, props.minor);
    printf("Dynamic parallelism: %s\n", (props.major > 3 || (props.major == 3 && props.minor >= 5)) ? "YES" : "NO");
    printf("Tensor cores: %s\n", props.major >= 7 ? "YES" : "NO");
    printf("\n");

    if (props.major < 3 || (props.major == 3 && props.minor < 5)) {
        printf("ERROR: GPU does not support dynamic parallelism (requires CC 3.5+)\n");
        return 1;
    }

    printf("Creating organism...\n");
    Organism* organism = create_organism();

    int max_generations = (argc > 1) ? atoi(argv[1]) : 0;
    
    run_organism(organism, max_generations);

    if (argc > 2 && strcmp(argv[2], "--visualize") == 0) {
        printf("\nExtracting final CA state...\n");
        float* ca_state = new float[GRID_SIZE * GRID_SIZE];
        get_ca_state(organism, ca_state, GRID_SIZE * GRID_SIZE);

        printf("CA state sample (top-left 8x8):\n");
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                printf("%.2f ", ca_state[y * GRID_SIZE + x]);
            }
            printf("\n");
        }

        delete[] ca_state;
    }

    printf("\nCleaning up...\n");
    destroy_organism(organism);

    printf("Done.\n");
    return 0;
}

// main.cu - Minimal launcher for Slime Mold Transformer
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "slime/api/gpu_native.cu"

int main(int argc, char** argv) {
    printf("=== Slime Mold Transformer - GPU Native ===\n");
    printf("Starting emergent CA evolution...\n\n");

    // Initialize CUDA and check device
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        const char* errStr = cudaGetErrorString(err);
        printf("ERROR: Failed to set CUDA device (code %d): %s\n", err, errStr ? errStr : "unknown");
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

    // Verify dynamic parallelism support
    if (props.major < 3 || (props.major == 3 && props.minor < 5)) {
        printf("ERROR: GPU does not support dynamic parallelism (requires CC 3.5+)\n");
        return 1;
    }

    // Create organism
    printf("Creating organism...\n");
    Organism* organism = create_organism();

    // Run evolution
    int generations = (argc > 1) ? atoi(argv[1]) : 100;
    printf("Running for %d generations...\n\n", generations);

    run_organism(organism, generations);

    // Optional: Get final CA state for visualization
    if (argc > 2 && strcmp(argv[2], "--visualize") == 0) {
        printf("\nExtracting final CA state...\n");
        float* ca_state = new float[GRID_SIZE * GRID_SIZE];
        get_ca_state(organism, ca_state, GRID_SIZE * GRID_SIZE);

        // Print a small sample
        printf("CA state sample (top-left 8x8):\n");
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                printf("%.2f ", ca_state[y * GRID_SIZE + x]);
            }
            printf("\n");
        }

        delete[] ca_state;
    }

    // Cleanup
    printf("\nCleaning up...\n");
    destroy_organism(organism);

    printf("Done.\n");
    return 0;
}
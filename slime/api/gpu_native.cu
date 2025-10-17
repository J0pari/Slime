// slime/api/gpu_native.cu - Minimal API to launch the organism
#ifndef GPU_NATIVE_CU
#define GPU_NATIVE_CU
#include <cuda_runtime.h>
#include <stdio.h>
#include "../core/organism.cu"

// Forward declaration
__global__ void extract_ca_channel_kernel(float* output, float* concentration, int grid_size);

// Allocate and initialize organism on GPU
extern "C" Organism* create_organism() {
    Organism* h_organism = new Organism();
    Organism* d_organism;

    // Allocate organism structure on device
    cudaMalloc(&d_organism, sizeof(Organism));

    // Allocate all components
    ComponentPool* d_pool;
    cudaMalloc(&d_pool, sizeof(ComponentPool));
    cudaMalloc(&d_pool->entries, MAX_POOL_SIZE * sizeof(PoolEntry));

    GPUElite* d_archive;
    cudaMalloc(&d_archive, MAX_ARCHIVE_SIZE * sizeof(GPUElite));

    TemporalTube* d_tubes;
    cudaMalloc(&d_tubes, sizeof(TemporalTube));
    cudaMalloc(&d_tubes->entries, MAX_MEMORY_SIZE * sizeof(MemoryEntry));

    MultiHeadCAState* d_ca_state;
    cudaMalloc(&d_ca_state, sizeof(MultiHeadCAState));
    cudaMalloc(&d_ca_state->perception_weights, NUM_HEADS * CHANNELS * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_ca_state->interaction_weights, NUM_HEADS * CHANNELS * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_ca_state->value_weights, NUM_HEADS * HIDDEN_DIM * CHANNELS * sizeof(float));
    cudaMalloc(&d_ca_state->head_mixing_weights, NUM_HEADS * NUM_HEADS * sizeof(float));
    cudaMalloc(&d_ca_state->flow_kernels, NUM_HEADS * 9 * sizeof(float));
    cudaMalloc(&d_ca_state->mass_buffer, NUM_HEADS * sizeof(float));

    BehavioralState* d_behavioral;
    cudaMalloc(&d_behavioral, MAX_COMPONENTS * sizeof(BehavioralState));

    ChemicalField* d_chemical;
    cudaMalloc(&d_chemical, sizeof(ChemicalField));
    cudaMalloc(&d_chemical->concentration, GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMalloc(&d_chemical->gradient_x, GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMalloc(&d_chemical->gradient_y, GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMalloc(&d_chemical->laplacian, GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMalloc(&d_chemical->sources, GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMalloc(&d_chemical->decay_factors, GRID_SIZE * GRID_SIZE * sizeof(float));

    // Allocate history buffers
    float* d_fitness_history;
    float* d_coherence_history;
    float* d_effective_rank_history;
    cudaMalloc(&d_fitness_history, MAX_GENERATIONS * MAX_COMPONENTS * sizeof(float));
    cudaMalloc(&d_coherence_history, MAX_GENERATIONS * MAX_COMPONENTS * sizeof(float));
    cudaMalloc(&d_effective_rank_history, MAX_GENERATIONS * sizeof(float));

    // Set pointers in organism structure
    h_organism->pool = d_pool;
    h_organism->archive = d_archive;
    h_organism->memory_tubes = d_tubes;
    h_organism->ca_state = d_ca_state;
    h_organism->behavioral_agents = d_behavioral;
    h_organism->chemical_field = d_chemical;
    h_organism->fitness_history = d_fitness_history;
    h_organism->coherence_history = d_coherence_history;
    h_organism->effective_rank_history = d_effective_rank_history;

    // Copy organism structure to device
    cudaMemcpy(d_organism, h_organism, sizeof(Organism), cudaMemcpyHostToDevice);

    // Initialize with seed
    unsigned int seed = 42;
    init_organism_kernel<<<1, 1>>>(d_organism, seed);
    cudaDeviceSynchronize();

    delete h_organism;
    return d_organism;
}

// Run organism for N generations
extern "C" void run_organism(Organism* d_organism, int generations) {
    printf("Starting organism evolution for %d generations...\n", generations);

    for (int gen = 0; gen < generations; gen++) {
        // Launch lifecycle with dynamic parallelism
        organism_lifecycle_kernel<<<1, 1>>>(d_organism, gen);
        cudaDeviceSynchronize();

        // Print progress every 10 generations
        if (gen % 10 == 0) {
            // Get current stats
            Organism h_organism;
            cudaMemcpy(&h_organism, d_organism, sizeof(Organism), cudaMemcpyDeviceToHost);

            float fitness, coherence;
            cudaMemcpy(&fitness, h_organism.fitness_history + gen * MAX_COMPONENTS,
                      sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&coherence, h_organism.coherence_history + gen * MAX_COMPONENTS,
                      sizeof(float), cudaMemcpyDeviceToHost);

            printf("Gen %4d: fitness=%.4f, coherence=%.4f\n", gen, fitness, coherence);

            // Check for convergence
            if (fitness > FITNESS_THRESHOLD && coherence > 0.9f) {
                printf("CONVERGED! Emergent behavior achieved.\n");
                break;
            }
        }
    }
}

// Cleanup organism
extern "C" void destroy_organism(Organism* d_organism) {
    // Get pointers from device
    Organism h_organism;
    cudaMemcpy(&h_organism, d_organism, sizeof(Organism), cudaMemcpyDeviceToHost);

    // Free all allocations
    cudaFree(h_organism.pool->entries);
    cudaFree(h_organism.pool);
    cudaFree(h_organism.archive);
    cudaFree(h_organism.memory_tubes->entries);
    cudaFree(h_organism.memory_tubes);
    cudaFree(h_organism.ca_state->perception_weights);
    cudaFree(h_organism.ca_state->interaction_weights);
    cudaFree(h_organism.ca_state->value_weights);
    cudaFree(h_organism.ca_state->head_mixing_weights);
    cudaFree(h_organism.ca_state->flow_kernels);
    cudaFree(h_organism.ca_state->mass_buffer);
    cudaFree(h_organism.ca_state);
    cudaFree(h_organism.behavioral_agents);
    cudaFree(h_organism.chemical_field->concentration);
    cudaFree(h_organism.chemical_field->gradient_x);
    cudaFree(h_organism.chemical_field->gradient_y);
    cudaFree(h_organism.chemical_field->laplacian);
    cudaFree(h_organism.chemical_field->sources);
    cudaFree(h_organism.chemical_field->decay_factors);
    cudaFree(h_organism.chemical_field);
    cudaFree(h_organism.fitness_history);
    cudaFree(h_organism.coherence_history);
    cudaFree(h_organism.effective_rank_history);
    cudaFree(d_organism);

    printf("Organism destroyed.\n");
}

// Get current CA state for visualization
extern "C" void get_ca_state(Organism* d_organism, float* h_buffer, int size) {
    Organism h_organism;
    cudaMemcpy(&h_organism, d_organism, sizeof(Organism), cudaMemcpyDeviceToHost);

    // Create temporary buffer for CA state
    float* d_ca_buffer;
    cudaMalloc(&d_ca_buffer, size * sizeof(float));

    // Extract first channel of CA state
    dim3 grid((GRID_SIZE + 15) / 16, (GRID_SIZE + 15) / 16);
    dim3 block(16, 16);

    extract_ca_channel_kernel<<<grid, block>>>(
        d_ca_buffer,
        h_organism.chemical_field->concentration,
        GRID_SIZE
    );

    cudaMemcpy(h_buffer, d_ca_buffer, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ca_buffer);
}

// Helper kernel to extract CA channel
__global__ void extract_ca_channel_kernel(
    float* output,
    float* concentration,
    int grid_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size) {
        output[y * grid_size + x] = concentration[y * grid_size + x];
    }
}

#endif // GPU_NATIVE_CU

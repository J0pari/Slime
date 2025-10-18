// slime/api/gpu_native.cu - Minimal API to launch the organism
#ifndef GPU_NATIVE_CU
#define GPU_NATIVE_CU
#include <cuda_runtime.h>
#include <stdio.h>
#include "../core/organism.cu"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA ERROR at %s:%d: %s (code %d)\n", \
                   __FILE__, __LINE__, cudaGetErrorString(err), err); \
            printf("Failed call: %s\n", #call); \
            exit(1); \
        } \
    } while(0)

// Forward declaration
__global__ void extract_ca_channel_kernel(float* output, float* concentration, int grid_size);

// Allocate and initialize organism on GPU
extern "C" Organism* create_organism() {
    printf("DEBUG: Creating organism...\n");
    Organism* h_organism = new Organism();
    printf("DEBUG: Host organism allocated\n");
    Organism* d_organism;

    // Allocate organism structure on device
    CUDA_CHECK(cudaMalloc(&d_organism, sizeof(Organism)));
    printf("DEBUG: Device organism allocated (%zu bytes)\n", sizeof(Organism));

    // Allocate all components
    ComponentPool* d_pool;
    PoolEntry* d_pool_entries;
    CUDA_CHECK(cudaMalloc(&d_pool, sizeof(ComponentPool)));
    printf("DEBUG: ComponentPool allocated (%zu bytes)\n", sizeof(ComponentPool));
    CUDA_CHECK(cudaMalloc(&d_pool_entries, MAX_POOL_SIZE * sizeof(PoolEntry)));
    printf("DEBUG: PoolEntry array allocated (%zu bytes)\n", MAX_POOL_SIZE * sizeof(PoolEntry));

    GPUElite* d_archive;
    VoronoiCell* d_voronoi_cells;
    CUDA_CHECK(cudaMalloc(&d_archive, MAX_ARCHIVE_SIZE * sizeof(GPUElite)));
    printf("DEBUG: Archive allocated (%zu bytes)\n", MAX_ARCHIVE_SIZE * sizeof(GPUElite));
    CUDA_CHECK(cudaMalloc(&d_voronoi_cells, MAX_CELLS * sizeof(VoronoiCell)));
    printf("DEBUG: Voronoi cells allocated (%zu bytes)\n", MAX_CELLS * sizeof(VoronoiCell));

    TemporalTube* d_tubes;
    MemoryEntry* d_tube_entries;
    CUDA_CHECK(cudaMalloc(&d_tubes, sizeof(TemporalTube)));
    printf("DEBUG: TemporalTube allocated (%zu bytes)\n", sizeof(TemporalTube));
    CUDA_CHECK(cudaMalloc(&d_tube_entries, MAX_MEMORY_SIZE * sizeof(MemoryEntry)));
    printf("DEBUG: MemoryEntry array allocated (%zu bytes)\n", MAX_MEMORY_SIZE * sizeof(MemoryEntry));

    MultiHeadCAState* d_ca_state;
    float* d_perception_weights;
    float* d_interaction_weights;
    float* d_value_weights;
    float* d_head_mixing_weights;
    float* d_flow_kernels;
    float* d_mass_buffer;
    float* d_ca_input;
    float* d_ca_output;
    CUDA_CHECK(cudaMalloc(&d_ca_state, sizeof(MultiHeadCAState)));
    printf("DEBUG: MultiHeadCAState allocated (%zu bytes)\n", sizeof(MultiHeadCAState));
    CUDA_CHECK(cudaMalloc(&d_perception_weights, NUM_HEADS * CHANNELS * HIDDEN_DIM * sizeof(float)));
    printf("DEBUG: Perception weights allocated\n");
    CUDA_CHECK(cudaMalloc(&d_interaction_weights, NUM_HEADS * CHANNELS * HIDDEN_DIM * sizeof(float)));
    printf("DEBUG: Interaction weights allocated\n");
    CUDA_CHECK(cudaMalloc(&d_value_weights, NUM_HEADS * HIDDEN_DIM * CHANNELS * sizeof(float)));
    printf("DEBUG: Value weights allocated\n");
    CUDA_CHECK(cudaMalloc(&d_head_mixing_weights, NUM_HEADS * NUM_HEADS * sizeof(float)));
    printf("DEBUG: Head mixing weights allocated\n");
    CUDA_CHECK(cudaMalloc(&d_flow_kernels, NUM_HEADS * 9 * sizeof(float)));
    printf("DEBUG: Flow kernels allocated\n");
    CUDA_CHECK(cudaMalloc(&d_mass_buffer, NUM_HEADS * sizeof(float)));
    printf("DEBUG: Mass buffer allocated\n");
    CUDA_CHECK(cudaMalloc(&d_ca_input, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float)));
    printf("DEBUG: CA input buffer allocated\n");
    CUDA_CHECK(cudaMalloc(&d_ca_output, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float)));
    printf("DEBUG: CA output buffer allocated\n");

    BehavioralState* d_behavioral;
    CUDA_CHECK(cudaMalloc(&d_behavioral, MAX_COMPONENTS * sizeof(BehavioralState)));
    printf("DEBUG: BehavioralState allocated (%zu bytes)\n", MAX_COMPONENTS * sizeof(BehavioralState));

    ChemicalField* d_chemical;
    float* d_concentration;
    float* d_gradient_x;
    float* d_gradient_y;
    float* d_laplacian;
    float* d_sources;
    float* d_decay_factors;
    TemporalTube* d_chemical_history;
    MemoryEntry* d_chemical_history_entries;
    CUDA_CHECK(cudaMalloc(&d_chemical, sizeof(ChemicalField)));
    printf("DEBUG: ChemicalField allocated (%zu bytes)\n", sizeof(ChemicalField));
    CUDA_CHECK(cudaMalloc(&d_concentration, GRID_SIZE * GRID_SIZE * sizeof(float)));
    printf("DEBUG: Chemical concentration allocated\n");
    CUDA_CHECK(cudaMalloc(&d_gradient_x, GRID_SIZE * GRID_SIZE * sizeof(float)));
    printf("DEBUG: Chemical gradient_x allocated\n");
    CUDA_CHECK(cudaMalloc(&d_gradient_y, GRID_SIZE * GRID_SIZE * sizeof(float)));
    printf("DEBUG: Chemical gradient_y allocated\n");
    CUDA_CHECK(cudaMalloc(&d_laplacian, GRID_SIZE * GRID_SIZE * sizeof(float)));
    printf("DEBUG: Chemical laplacian allocated\n");
    CUDA_CHECK(cudaMalloc(&d_sources, GRID_SIZE * GRID_SIZE * sizeof(float)));
    printf("DEBUG: Chemical sources allocated\n");
    CUDA_CHECK(cudaMalloc(&d_decay_factors, GRID_SIZE * GRID_SIZE * sizeof(float)));
    printf("DEBUG: Chemical decay_factors allocated\n");
    CUDA_CHECK(cudaMalloc(&d_chemical_history, sizeof(TemporalTube)));
    printf("DEBUG: Chemical history TemporalTube allocated\n");
    CUDA_CHECK(cudaMalloc(&d_chemical_history_entries, MAX_HISTORY_LENGTH * sizeof(MemoryEntry)));
    printf("DEBUG: Chemical history entries allocated (%zu bytes)\n", MAX_HISTORY_LENGTH * sizeof(MemoryEntry));

    // Allocate history buffers
    float* d_fitness_history;
    float* d_coherence_history;
    float* d_effective_rank_history;
    CUDA_CHECK(cudaMalloc(&d_fitness_history, MAX_GENERATIONS * MAX_COMPONENTS * sizeof(float)));
    printf("DEBUG: Fitness history allocated (%zu bytes)\n", MAX_GENERATIONS * MAX_COMPONENTS * sizeof(float));
    CUDA_CHECK(cudaMalloc(&d_coherence_history, MAX_GENERATIONS * MAX_COMPONENTS * sizeof(float)));
    printf("DEBUG: Coherence history allocated (%zu bytes)\n", MAX_GENERATIONS * MAX_COMPONENTS * sizeof(float));
    CUDA_CHECK(cudaMalloc(&d_effective_rank_history, MAX_GENERATIONS * sizeof(float)));
    printf("DEBUG: Effective rank history allocated (%zu bytes)\n", MAX_GENERATIONS * sizeof(float));

    // Now set the nested pointers in the allocated structures
    ComponentPool h_pool;
    h_pool.entries = d_pool_entries;
    h_pool.capacity = MAX_POOL_SIZE;
    h_pool.active_count = 0;
    h_pool.total_spawned = 0;
    h_pool.total_culled = 0;
    CUDA_CHECK(cudaMemcpy(d_pool, &h_pool, sizeof(ComponentPool), cudaMemcpyHostToDevice));
    printf("DEBUG: ComponentPool initialized with capacity=%d\n", MAX_POOL_SIZE);

    TemporalTube h_tubes;
    h_tubes.entries = d_tube_entries;
    h_tubes.capacity = MAX_MEMORY_SIZE;
    h_tubes.head = 0;
    h_tubes.count = 0;
    h_tubes.global_time = 0.0f;
    h_tubes.decay_rate = 0.95f;
    CUDA_CHECK(cudaMemcpy(d_tubes, &h_tubes, sizeof(TemporalTube), cudaMemcpyHostToDevice));
    printf("DEBUG: TemporalTube initialized with capacity=%d\n", MAX_MEMORY_SIZE);

    MultiHeadCAState h_ca_state;
    h_ca_state.perception_weights = d_perception_weights;
    h_ca_state.interaction_weights = d_interaction_weights;
    h_ca_state.value_weights = d_value_weights;
    h_ca_state.head_mixing_weights = d_head_mixing_weights;
    h_ca_state.flow_kernels = d_flow_kernels;
    h_ca_state.mass_buffer = d_mass_buffer;
    h_ca_state.ca_input = d_ca_input;
    h_ca_state.ca_output = d_ca_output;
    CUDA_CHECK(cudaMemcpy(d_ca_state, &h_ca_state, sizeof(MultiHeadCAState), cudaMemcpyHostToDevice));

    // Initialize chemical history TemporalTube
    TemporalTube h_chemical_history;
    h_chemical_history.entries = d_chemical_history_entries;
    h_chemical_history.capacity = MAX_HISTORY_LENGTH;
    h_chemical_history.head = 0;
    h_chemical_history.count = 0;
    h_chemical_history.global_time = 0.0f;
    h_chemical_history.decay_rate = 1.0f;  // No decay for chemical field snapshots
    
    // Allocate data buffers for each history entry
    int field_size = GRID_SIZE * GRID_SIZE;
    for (int i = 0; i < MAX_HISTORY_LENGTH; i++) {
        float* entry_data;
        CUDA_CHECK(cudaMalloc(&entry_data, field_size * sizeof(float)));
        MemoryEntry entry;
        entry.data = entry_data;
        entry.size = field_size;
        entry.timestamp = 0.0f;
        entry.decay_factor = 1.0f;
        entry.importance = 1.0f;
        CUDA_CHECK(cudaMemcpy(&d_chemical_history_entries[i], &entry, sizeof(MemoryEntry), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_chemical_history, &h_chemical_history, sizeof(TemporalTube), cudaMemcpyHostToDevice));
    printf("DEBUG: Chemical history TemporalTube initialized with %d entries\n", MAX_HISTORY_LENGTH);

    ChemicalField h_chemical;
    h_chemical.concentration = d_concentration;
    h_chemical.gradient_x = d_gradient_x;
    h_chemical.gradient_y = d_gradient_y;
    h_chemical.laplacian = d_laplacian;
    h_chemical.sources = d_sources;
    h_chemical.decay_factors = d_decay_factors;
    h_chemical.history = d_chemical_history;
    printf("DEBUG: ChemicalField host pointers: concentration=%p, gradient_x=%p, history=%p\n", 
           h_chemical.concentration, h_chemical.gradient_x, h_chemical.history);
    CUDA_CHECK(cudaMemcpy(d_chemical, &h_chemical, sizeof(ChemicalField), cudaMemcpyHostToDevice));
    
    // Verify what was copied
    ChemicalField h_chemical_verify;
    CUDA_CHECK(cudaMemcpy(&h_chemical_verify, d_chemical, sizeof(ChemicalField), cudaMemcpyDeviceToHost));
    printf("DEBUG: ChemicalField device pointers (read back): concentration=%p, gradient_x=%p\n",
           h_chemical_verify.concentration, h_chemical_verify.gradient_x);
    printf("DEBUG: All nested pointers set\n");

    // Initialize Voronoi cells
    printf("DEBUG: Initializing Voronoi cells...\n");
    int behavioral_dim = ARCH_BEHAVIORAL_DIM;  // From archive.cu
    unsigned int seed = 42;  // Fixed seed for reproducibility
    init_voronoi_cells_kernel<<<(MAX_CELLS + 255) / 256, 256>>>(
        d_voronoi_cells,
        MAX_CELLS,
        behavioral_dim,
        seed
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("DEBUG: Voronoi cells initialized with %d cells in %dD space\n", MAX_CELLS, behavioral_dim);

    // Set pointers in organism structure
    h_organism->pool = d_pool;
    h_organism->archive = d_archive;
    h_organism->archive_size = 0;
    h_organism->voronoi_cells = d_voronoi_cells;
    h_organism->num_voronoi_cells = MAX_CELLS;
    h_organism->memory_tubes = d_tubes;
    h_organism->ca_state = d_ca_state;
    h_organism->behavioral_agents = d_behavioral;
    h_organism->chemical_field = d_chemical;
    h_organism->fitness_history = d_fitness_history;
    h_organism->coherence_history = d_coherence_history;
    h_organism->effective_rank_history = d_effective_rank_history;
    printf("DEBUG: All pointers set in host organism\n");

    // Copy organism structure to device
    CUDA_CHECK(cudaMemcpy(d_organism, h_organism, sizeof(Organism), cudaMemcpyHostToDevice));
    printf("DEBUG: Organism structure copied to device\n");
    
    // Verify organism was copied correctly
    Organism h_organism_verify;
    CUDA_CHECK(cudaMemcpy(&h_organism_verify, d_organism, sizeof(Organism), cudaMemcpyDeviceToHost));
    printf("DEBUG: Organism pointers (read back): pool=%p, chemical_field=%p\n",
           h_organism_verify.pool, h_organism_verify.chemical_field);

    // Initialize organism with same seed as Voronoi cells
    printf("DEBUG: Launching init_organism_kernel...\n");
    init_organism_kernel<<<1, 1>>>(d_organism, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("DEBUG: init_organism_kernel completed\n");

    delete h_organism;
    printf("DEBUG: create_organism completed successfully\n");
    return d_organism;
}

// Run organism for N generations
extern "C" void run_organism(Organism* d_organism, int generations) {
    printf("Starting organism evolution for %d generations...\n", generations);

    for (int gen = 0; gen < generations; gen++) {
        // Launch lifecycle with dynamic parallelism
        organism_lifecycle_kernel<<<1, 1>>>(d_organism, gen);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Print progress every 10 generations
        if (gen % 10 == 0) {
            // Get current stats
            Organism h_organism;
            CUDA_CHECK(cudaMemcpy(&h_organism, d_organism, sizeof(Organism), cudaMemcpyDeviceToHost));

            float fitness, coherence;
            CUDA_CHECK(cudaMemcpy(&fitness, h_organism.fitness_history + gen * MAX_COMPONENTS,
                      sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&coherence, h_organism.coherence_history + gen * MAX_COMPONENTS,
                      sizeof(float), cudaMemcpyDeviceToHost));

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
    printf("DEBUG: Destroying organism...\n");
    // Get pointers from device
    Organism h_organism;
    CUDA_CHECK(cudaMemcpy(&h_organism, d_organism, sizeof(Organism), cudaMemcpyDeviceToHost));

    // Get nested structure pointers
    ComponentPool h_pool;
    TemporalTube h_tubes;
    MultiHeadCAState h_ca_state;
    ChemicalField h_chemical;
    CUDA_CHECK(cudaMemcpy(&h_pool, h_organism.pool, sizeof(ComponentPool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_tubes, h_organism.memory_tubes, sizeof(TemporalTube), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_ca_state, h_organism.ca_state, sizeof(MultiHeadCAState), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_chemical, h_organism.chemical_field, sizeof(ChemicalField), cudaMemcpyDeviceToHost));

    // Get chemical history and free its data buffers
    TemporalTube h_chemical_history;
    CUDA_CHECK(cudaMemcpy(&h_chemical_history, h_chemical.history, sizeof(TemporalTube), cudaMemcpyDeviceToHost));
    
    // Free each chemical history entry's data buffer
    for (int i = 0; i < MAX_HISTORY_LENGTH; i++) {
        MemoryEntry entry;
        CUDA_CHECK(cudaMemcpy(&entry, &h_chemical_history.entries[i], sizeof(MemoryEntry), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(entry.data));
    }
    CUDA_CHECK(cudaFree(h_chemical_history.entries));
    CUDA_CHECK(cudaFree(h_chemical.history));

    // Free all allocations
    CUDA_CHECK(cudaFree(h_pool.entries));
    CUDA_CHECK(cudaFree(h_organism.pool));
    CUDA_CHECK(cudaFree(h_organism.archive));
    CUDA_CHECK(cudaFree(h_organism.voronoi_cells));
    CUDA_CHECK(cudaFree(h_tubes.entries));
    CUDA_CHECK(cudaFree(h_organism.memory_tubes));
    CUDA_CHECK(cudaFree(h_ca_state.perception_weights));
    CUDA_CHECK(cudaFree(h_ca_state.interaction_weights));
    CUDA_CHECK(cudaFree(h_ca_state.value_weights));
    CUDA_CHECK(cudaFree(h_ca_state.head_mixing_weights));
    CUDA_CHECK(cudaFree(h_ca_state.flow_kernels));
    CUDA_CHECK(cudaFree(h_ca_state.mass_buffer));
    CUDA_CHECK(cudaFree(h_ca_state.ca_input));
    CUDA_CHECK(cudaFree(h_ca_state.ca_output));
    CUDA_CHECK(cudaFree(h_organism.ca_state));
    CUDA_CHECK(cudaFree(h_organism.behavioral_agents));
    CUDA_CHECK(cudaFree(h_chemical.concentration));
    CUDA_CHECK(cudaFree(h_chemical.gradient_x));
    CUDA_CHECK(cudaFree(h_chemical.gradient_y));
    CUDA_CHECK(cudaFree(h_chemical.laplacian));
    CUDA_CHECK(cudaFree(h_chemical.sources));
    CUDA_CHECK(cudaFree(h_chemical.decay_factors));
    CUDA_CHECK(cudaFree(h_organism.chemical_field));
    CUDA_CHECK(cudaFree(h_organism.fitness_history));
    CUDA_CHECK(cudaFree(h_organism.coherence_history));
    CUDA_CHECK(cudaFree(h_organism.effective_rank_history));
    CUDA_CHECK(cudaFree(d_organism));

    printf("Organism destroyed.\n");
}

// Get current CA state for visualization
extern "C" void get_ca_state(Organism* d_organism, float* h_buffer, int size) {
    Organism h_organism;
    CUDA_CHECK(cudaMemcpy(&h_organism, d_organism, sizeof(Organism), cudaMemcpyDeviceToHost));

    ChemicalField h_chemical;
    CUDA_CHECK(cudaMemcpy(&h_chemical, h_organism.chemical_field, sizeof(ChemicalField), cudaMemcpyDeviceToHost));

    // Create temporary buffer for CA state
    float* d_ca_buffer;
    CUDA_CHECK(cudaMalloc(&d_ca_buffer, size * sizeof(float)));

    // Extract first channel of CA state
    dim3 grid((GRID_SIZE + 15) / 16, (GRID_SIZE + 15) / 16);
    dim3 block(16, 16);

    extract_ca_channel_kernel<<<grid, block>>>(
        d_ca_buffer,
        h_chemical.concentration,
        GRID_SIZE
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_buffer, d_ca_buffer, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_ca_buffer));
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

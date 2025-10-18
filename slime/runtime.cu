// slime/runtime.cu - GPU runtime for organism lifecycle
#ifndef RUNTIME_CU
#define RUNTIME_CU
#include <cuda_runtime.h>
#include <stdio.h>
#include "core/organism.cu"

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
    size_t total_mem = 0;
    printf("[ALLOC] Starting organism initialization\n");
    Organism* h_organism = new Organism();
    Organism* d_organism;

    // Allocate organism structure on device
    CUDA_CHECK(cudaMalloc(&d_organism, sizeof(Organism)));
    total_mem += sizeof(Organism);
    printf("[VERIFY] Device organism ptr=%p, size=%zu bytes, total=%zu MB\n", 
           d_organism, sizeof(Organism), total_mem / (1024*1024));

    // Allocate all components
    ComponentPool* d_pool;
    PoolEntry* d_pool_entries;
    CUDA_CHECK(cudaMalloc(&d_pool, sizeof(ComponentPool)));
    total_mem += sizeof(ComponentPool);
    CUDA_CHECK(cudaMalloc(&d_pool_entries, MAX_POOL_SIZE * sizeof(PoolEntry)));
    total_mem += MAX_POOL_SIZE * sizeof(PoolEntry);
    printf("[VERIFY] ComponentPool ptr=%p, entries ptr=%p, pool entries=%zu MB, total=%zu MB\n",
           d_pool, d_pool_entries, (MAX_POOL_SIZE * sizeof(PoolEntry)) / (1024*1024), total_mem / (1024*1024));

    GPUElite* d_archive;
    VoronoiCell* d_voronoi_cells;
    CUDA_CHECK(cudaMalloc(&d_archive, MAX_ARCHIVE_SIZE * sizeof(GPUElite)));
    total_mem += MAX_ARCHIVE_SIZE * sizeof(GPUElite);
    CUDA_CHECK(cudaMalloc(&d_voronoi_cells, MAX_CELLS * sizeof(VoronoiCell)));
    total_mem += MAX_CELLS * sizeof(VoronoiCell);
    printf("[VERIFY] Archive ptr=%p, Voronoi ptr=%p, archive=%zu MB, total=%zu MB\n",
           d_archive, d_voronoi_cells, (MAX_ARCHIVE_SIZE * sizeof(GPUElite)) / (1024*1024), total_mem / (1024*1024));

    TemporalTube* d_tubes;
    MemoryEntry* d_tube_entries;
    CUDA_CHECK(cudaMalloc(&d_tubes, sizeof(TemporalTube)));
    total_mem += sizeof(TemporalTube);
    CUDA_CHECK(cudaMalloc(&d_tube_entries, MAX_MEMORY_SIZE * sizeof(MemoryEntry)));
    total_mem += MAX_MEMORY_SIZE * sizeof(MemoryEntry);
    printf("[VERIFY] TemporalTube ptr=%p, entries ptr=%p, total=%zu MB\n",
           d_tubes, d_tube_entries, total_mem / (1024*1024));

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
    total_mem += sizeof(MultiHeadCAState);
    CUDA_CHECK(cudaMalloc(&d_perception_weights, NUM_HEADS * CHANNELS * HIDDEN_DIM * sizeof(float)));
    total_mem += NUM_HEADS * CHANNELS * HIDDEN_DIM * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_interaction_weights, NUM_HEADS * CHANNELS * HIDDEN_DIM * sizeof(float)));
    total_mem += NUM_HEADS * CHANNELS * HIDDEN_DIM * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_value_weights, NUM_HEADS * HIDDEN_DIM * CHANNELS * sizeof(float)));
    total_mem += NUM_HEADS * HIDDEN_DIM * CHANNELS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_head_mixing_weights, NUM_HEADS * NUM_HEADS * sizeof(float)));
    total_mem += NUM_HEADS * NUM_HEADS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_flow_kernels, NUM_HEADS * 9 * sizeof(float)));
    total_mem += NUM_HEADS * 9 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_mass_buffer, NUM_HEADS * sizeof(float)));
    total_mem += NUM_HEADS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_ca_input, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float)));
    total_mem += GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_ca_output, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float)));
    total_mem += GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float);
    printf("[VERIFY] MultiHeadCA state ptr=%p, 8 weight buffers allocated, total=%zu MB\n",
           d_ca_state, total_mem / (1024*1024));

    BehavioralState* d_behavioral;
    CUDA_CHECK(cudaMalloc(&d_behavioral, MAX_COMPONENTS * sizeof(BehavioralState)));
    total_mem += MAX_COMPONENTS * sizeof(BehavioralState);
    printf("[VERIFY] BehavioralState ptr=%p, size=%zu KB, total=%zu MB\n",
           d_behavioral, (MAX_COMPONENTS * sizeof(BehavioralState)) / 1024, total_mem / (1024*1024));

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
    total_mem += sizeof(ChemicalField);
    CUDA_CHECK(cudaMalloc(&d_concentration, GRID_SIZE * GRID_SIZE * sizeof(float)));
    total_mem += GRID_SIZE * GRID_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_gradient_x, GRID_SIZE * GRID_SIZE * sizeof(float)));
    total_mem += GRID_SIZE * GRID_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_gradient_y, GRID_SIZE * GRID_SIZE * sizeof(float)));
    total_mem += GRID_SIZE * GRID_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_laplacian, GRID_SIZE * GRID_SIZE * sizeof(float)));
    total_mem += GRID_SIZE * GRID_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_sources, GRID_SIZE * GRID_SIZE * sizeof(float)));
    total_mem += GRID_SIZE * GRID_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_decay_factors, GRID_SIZE * GRID_SIZE * sizeof(float)));
    total_mem += GRID_SIZE * GRID_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_chemical_history, sizeof(TemporalTube)));
    total_mem += sizeof(TemporalTube);
    CUDA_CHECK(cudaMalloc(&d_chemical_history_entries, MAX_HISTORY_LENGTH * sizeof(MemoryEntry)));
    total_mem += MAX_HISTORY_LENGTH * sizeof(MemoryEntry);
    printf("[VERIFY] ChemicalField ptr=%p, 6 field arrays + history, total=%zu MB\n",
           d_chemical, total_mem / (1024*1024));

    // Allocate history buffers
    float* d_fitness_history;
    float* d_coherence_history;
    float* d_effective_rank_history;
    CUDA_CHECK(cudaMalloc(&d_fitness_history, MAX_GENERATIONS * MAX_COMPONENTS * sizeof(float)));
    total_mem += MAX_GENERATIONS * MAX_COMPONENTS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_coherence_history, MAX_GENERATIONS * MAX_COMPONENTS * sizeof(float)));
    total_mem += MAX_GENERATIONS * MAX_COMPONENTS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_effective_rank_history, MAX_GENERATIONS * sizeof(float)));
    total_mem += MAX_GENERATIONS * sizeof(float);
    printf("[VERIFY] History buffers: fitness=%p, coherence=%p, rank=%p, history=%zu MB, TOTAL GPU=%zu MB\n",
           d_fitness_history, d_coherence_history, d_effective_rank_history,
           (2 * MAX_GENERATIONS * MAX_COMPONENTS + MAX_GENERATIONS) * sizeof(float) / (1024*1024),
           total_mem / (1024*1024));

    // Now set the nested pointers in the allocated structures
    ComponentPool h_pool;
    h_pool.entries = d_pool_entries;
    h_pool.capacity = MAX_POOL_SIZE;
    h_pool.active_count = 0;
    h_pool.total_spawned = 0;
    h_pool.total_culled = 0;
    CUDA_CHECK(cudaMemcpy(d_pool, &h_pool, sizeof(ComponentPool), cudaMemcpyHostToDevice));
    ComponentPool verify_pool;
    CUDA_CHECK(cudaMemcpy(&verify_pool, d_pool, sizeof(ComponentPool), cudaMemcpyDeviceToHost));
    printf("[VERIFY] ComponentPool: capacity=%d (actual on device), active=%d, spawned=%d\n",
           verify_pool.capacity, verify_pool.active_count.load(), verify_pool.total_spawned.load());

    TemporalTube h_tubes;
    h_tubes.entries = d_tube_entries;
    h_tubes.capacity = MAX_MEMORY_SIZE;
    h_tubes.head = 0;
    h_tubes.count = 0;
    h_tubes.global_time = 0.0f;
    h_tubes.decay_rate = 0.95f;
    CUDA_CHECK(cudaMemcpy(d_tubes, &h_tubes, sizeof(TemporalTube), cudaMemcpyHostToDevice));
    TemporalTube verify_tubes;
    CUDA_CHECK(cudaMemcpy(&verify_tubes, d_tubes, sizeof(TemporalTube), cudaMemcpyDeviceToHost));
    printf("[VERIFY] TemporalTube: capacity=%d (actual), count=%d, decay_rate=%.3f\n",
           verify_tubes.capacity, verify_tubes.count, verify_tubes.decay_rate);

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
    TemporalTube verify_chem_hist;
    CUDA_CHECK(cudaMemcpy(&verify_chem_hist, d_chemical_history, sizeof(TemporalTube), cudaMemcpyDeviceToHost));
    printf("[VERIFY] Chemical history: capacity=%d (actual), count=%d, %d data buffers allocated\n",
           verify_chem_hist.capacity, verify_chem_hist.count, MAX_HISTORY_LENGTH);

    ChemicalField h_chemical;
    h_chemical.concentration = d_concentration;
    h_chemical.gradient_x = d_gradient_x;
    h_chemical.gradient_y = d_gradient_y;
    h_chemical.laplacian = d_laplacian;
    h_chemical.sources = d_sources;
    h_chemical.decay_factors = d_decay_factors;
    h_chemical.history = d_chemical_history;
    CUDA_CHECK(cudaMemcpy(d_chemical, &h_chemical, sizeof(ChemicalField), cudaMemcpyHostToDevice));
    
    // Verify what was copied
    ChemicalField verify_chem;
    CUDA_CHECK(cudaMemcpy(&verify_chem, d_chemical, sizeof(ChemicalField), cudaMemcpyDeviceToHost));
    bool pointers_match = (verify_chem.concentration == d_concentration) && 
                         (verify_chem.gradient_x == d_gradient_x) &&
                         (verify_chem.history == d_chemical_history);
    printf("[VERIFY] ChemicalField: concentration=%p, gradient_x=%p, history=%p, pointers_valid=%s\n",
           verify_chem.concentration, verify_chem.gradient_x, verify_chem.history,
           pointers_match ? "YES" : "CORRUPTED");
    if (!pointers_match) {
        printf("[ERROR] ChemicalField pointer corruption detected!\n");
        exit(1);
    }

    // Initialize Voronoi cells
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
    
    // Verify cells were actually initialized
    VoronoiCell first_cell;
    CUDA_CHECK(cudaMemcpy(&first_cell, d_voronoi_cells, sizeof(VoronoiCell), cudaMemcpyDeviceToHost));
    bool initialized = false;
    for (int i = 0; i < behavioral_dim && i < 10; i++) {
        if (first_cell.centroid[i] != 0.0f) {
            initialized = true;
            break;
        }
    }
    printf("[VERIFY] Voronoi: %d cells in %dD, first_cell.centroid[0]=%.4f, initialized=%s\n",
           MAX_CELLS, behavioral_dim, first_cell.centroid[0], initialized ? "YES" : "NO");

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

    // Allocate pre-allocated memory pools to replace device-side cudaMalloc
    uint8_t* d_compressed_genome_pool;
    uint32_t* d_compressed_size_pool;
    GPUElite* d_elite_staging_pool;
    float* d_behavioral_field_pool;
    float* d_behavioral_gradient_pool;
    float* d_svd_workspace_pool;
    float* d_svd_singular_values_pool;
    float* d_coherence_workspace_pool;
    float* d_memory_data_pool;
    
    CUDA_CHECK(cudaMalloc(&d_compressed_genome_pool, MAX_ARCHIVE_SIZE * GENOME_SIZE * sizeof(uint8_t)));
    total_mem += MAX_ARCHIVE_SIZE * GENOME_SIZE * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&d_compressed_size_pool, MAX_ARCHIVE_SIZE * sizeof(uint32_t)));
    total_mem += MAX_ARCHIVE_SIZE * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&d_elite_staging_pool, MAX_ARCHIVE_SIZE * sizeof(GPUElite)));
    total_mem += MAX_ARCHIVE_SIZE * sizeof(GPUElite);
    CUDA_CHECK(cudaMalloc(&d_behavioral_field_pool, MAX_COMPONENTS * GRID_SIZE * GRID_SIZE * BEHAVIORAL_DIM * sizeof(float)));
    total_mem += MAX_COMPONENTS * GRID_SIZE * GRID_SIZE * BEHAVIORAL_DIM * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_behavioral_gradient_pool, MAX_COMPONENTS * GRID_SIZE * GRID_SIZE * BEHAVIORAL_DIM * 2 * sizeof(float)));
    total_mem += MAX_COMPONENTS * GRID_SIZE * GRID_SIZE * BEHAVIORAL_DIM * 2 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_svd_workspace_pool, MAX_COMPONENTS * GENOME_SIZE * GENOME_SIZE * sizeof(float)));
    total_mem += MAX_COMPONENTS * GENOME_SIZE * GENOME_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_svd_singular_values_pool, MAX_COMPONENTS * GENOME_SIZE * sizeof(float)));
    total_mem += MAX_COMPONENTS * GENOME_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_coherence_workspace_pool, MAX_COMPONENTS * sizeof(float)));
    total_mem += MAX_COMPONENTS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_memory_data_pool, MAX_COMPONENTS * (BEHAVIORAL_DIM + 4) * sizeof(float)));
    total_mem += MAX_COMPONENTS * (BEHAVIORAL_DIM + 4) * sizeof(float);
    
    float* d_fitness_svd_pool;
    float* d_fitness_rank_pool;
    float* d_fitness_coherence_pool;
    CUDA_CHECK(cudaMalloc(&d_fitness_svd_pool, MAX_COMPONENTS * GENOME_SIZE * sizeof(float)));
    total_mem += MAX_COMPONENTS * GENOME_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_fitness_rank_pool, MAX_COMPONENTS * sizeof(float)));
    total_mem += MAX_COMPONENTS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_fitness_coherence_pool, MAX_COMPONENTS * sizeof(float)));
    total_mem += MAX_COMPONENTS * sizeof(float);
    
    h_organism->compressed_genome_pool = d_compressed_genome_pool;
    h_organism->compressed_size_pool = d_compressed_size_pool;
    h_organism->elite_staging_pool = d_elite_staging_pool;
    h_organism->behavioral_field_pool = d_behavioral_field_pool;
    h_organism->behavioral_gradient_pool = d_behavioral_gradient_pool;
    h_organism->svd_workspace_pool = d_svd_workspace_pool;
    h_organism->svd_singular_values_pool = d_svd_singular_values_pool;
    h_organism->coherence_workspace_pool = d_coherence_workspace_pool;
    h_organism->memory_data_pool = d_memory_data_pool;
    h_organism->fitness_svd_pool = d_fitness_svd_pool;
    h_organism->fitness_rank_pool = d_fitness_rank_pool;
    h_organism->fitness_coherence_pool = d_fitness_coherence_pool;

    // Copy organism structure to device
    CUDA_CHECK(cudaMemcpy(d_organism, h_organism, sizeof(Organism), cudaMemcpyHostToDevice));
    
    // Verify organism was copied correctly
    Organism verify_organism;
    CUDA_CHECK(cudaMemcpy(&verify_organism, d_organism, sizeof(Organism), cudaMemcpyDeviceToHost));
    bool organism_valid = (verify_organism.pool == d_pool) && 
                         (verify_organism.chemical_field == d_chemical) &&
                         (verify_organism.fitness_history == d_fitness_history);
    printf("[VERIFY] Organism: pool=%p, chemical=%p, fitness_hist=%p, structure_valid=%s\n",
           verify_organism.pool, verify_organism.chemical_field, verify_organism.fitness_history,
           organism_valid ? "YES" : "CORRUPTED");
    if (!organism_valid) {
        printf("[ERROR] Organism structure corruption detected!\n");
        exit(1);
    }

    // Initialize organism with same seed as Voronoi cells
    init_organism_kernel<<<1, 1>>>(d_organism, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Verify pool was actually initialized with live organisms
    ComponentPool verify_pool_post_init;
    CUDA_CHECK(cudaMemcpy(&verify_pool_post_init, verify_organism.pool, sizeof(ComponentPool), cudaMemcpyDeviceToHost));
    int active_count = verify_pool_post_init.active_count.load();
    int total_spawned = verify_pool_post_init.total_spawned.load();
    
    // Read back first organism to verify genome is initialized
    PoolEntry first_organism;
    CUDA_CHECK(cudaMemcpy(&first_organism, verify_pool_post_init.entries, sizeof(PoolEntry), cudaMemcpyDeviceToHost));
    
    float genome_sum = 0.0f;
    for (int i = 0; i < 10 && i < GENOME_SIZE; i++) {
        genome_sum += fabsf(first_organism.genome[i]);
    }
    
    printf("[VERIFY] Post-init pool: active=%d/%d, spawned=%d, first_organism: id=%d, alive=%s, genome[0..9]_sum=%.4f\n",
           active_count, verify_pool_post_init.capacity, total_spawned,
           first_organism.id, first_organism.alive ? "YES" : "NO", genome_sum);
    
    if (active_count == 0) {
        printf("[WARNING] Pool initialized with ZERO active organisms - evolution will not occur!\n");
    }
    if (!first_organism.alive && active_count > 0) {
        printf("[WARNING] First organism dead but active_count=%d - index mismatch!\n", active_count);
    }
    if (genome_sum < 0.001f && first_organism.alive) {
        printf("[WARNING] Live organism has near-zero genome - may produce zero fitness!\n");
    }

    delete h_organism;
    printf("[COMPLETE] Organism created: %zu MB GPU memory, %d organisms alive\n",
           total_mem / (1024*1024), active_count);
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

#endif // RUNTIME_CU

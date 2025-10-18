// slime/core/organism.cu - Top-level orchestrator with dynamic parallelism
#ifndef ORGANISM_CU
#define ORGANISM_CU
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include "../memory/archive.cu"
#include "../memory/pool.cu"
#include "../memory/tubes.cu"
#include "pseudopod.cu"
#include "chemotaxis.cu"

namespace cg = cooperative_groups;

// Organism configuration from blueprint
constexpr int MAX_COMPONENTS = 256;
constexpr int MAX_GENERATIONS = 10000;
constexpr int KERNEL_HIERARCHY_DEPTH = 5;  // Dynamic parallelism depth
constexpr float FITNESS_THRESHOLD = 0.8f;
constexpr float HUNGER_DEATH_THRESHOLD = 0.95f;
constexpr float SPAWN_RATE = 0.1f;
constexpr float CULL_RATE = 0.05f;
constexpr int MAX_JACOBI_SWEEPS = 30;  // For SVD computation

// Complete organism state
struct Organism {
    // Core components
    ComponentPool* pool;
    GPUElite* archive;
    int archive_size;
    VoronoiCell* voronoi_cells;
    int num_voronoi_cells;
    TemporalTube* memory_tubes;
    MultiHeadCAState* ca_state;
    BehavioralState* behavioral_agents;

    // Chemical fields
    ChemicalField* chemical_field;

    // Metrics
    float* fitness_history;
    float* coherence_history;
    float* effective_rank_history;
    int generation;
    int active_components;

    // Control parameters
    float learning_rate;
    float mutation_rate;
    float exploration_rate;

    // Pre-allocated memory pools to replace device-side cudaMalloc
    uint8_t* compressed_genome_pool;      // Pool for compressed genomes (MAX_ARCHIVE_SIZE * GENOME_SIZE)
    uint32_t* compressed_size_pool;       // Pool for compressed sizes (MAX_ARCHIVE_SIZE)
    GPUElite* elite_staging_pool;         // Pool for elite staging (MAX_ARCHIVE_SIZE)
    float* behavioral_field_pool;         // Pool for behavioral fields (MAX_COMPONENTS * GRID_SIZE * GRID_SIZE * BEHAVIORAL_DIM)
    float* behavioral_gradient_pool;      // Pool for behavioral gradients (MAX_COMPONENTS * GRID_SIZE * GRID_SIZE * BEHAVIORAL_DIM * 2)
    float* svd_workspace_pool;            // Pool for SVD matrices (MAX_COMPONENTS * GENOME_SIZE * GENOME_SIZE)
    float* svd_singular_values_pool;      // Pool for SVD singular values (MAX_COMPONENTS * GENOME_SIZE)
    float* coherence_workspace_pool;      // Pool for coherence computation (MAX_COMPONENTS)
    float* memory_data_pool;              // Pool for memory entries (MAX_COMPONENTS * (BEHAVIORAL_DIM + 4))
    float* fitness_svd_pool;              // Pool for fitness SVD singular values (MAX_COMPONENTS * GENOME_SIZE)
    float* fitness_rank_pool;             // Pool for fitness effective rank (MAX_COMPONENTS)
    float* fitness_coherence_pool;        // Pool for fitness coherence (MAX_COMPONENTS)
};

// Forward declarations for all kernels
__global__ void component_evolution_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, VoronoiCell* voronoi_cells, int num_cells, int* archive_size, ChemicalField* chemical_field, BehavioralState* behavioral_agents, float* fitness_history, float* coherence_history, int generation);
__global__ void neural_ca_update_kernel(MultiHeadCAState* ca_state, ChemicalField* chemical_field, float* effective_rank_history, int generation);
__global__ void behavioral_update_kernel(Organism* organism, BehavioralState* agents, ChemicalField* chemical_field, TemporalTube* memory_tubes, int generation);
__global__ void memory_update_kernel(TemporalTube* tubes, float* fitness_history, float* coherence_history, int generation);
__global__ void fitness_computation_kernel(ComponentPool* pool, ChemicalField* chemical_field, float* fitness_history, float* coherence_history, int generation);
__global__ void selection_kernel(Organism* organism, ComponentPool* pool, GPUElite* archive, VoronoiCell* voronoi_cells, int num_cells, int* archive_size, BehavioralState* behavioral_agents, int generation);
__global__ void spawn_wave_kernel(ComponentPool* pool, float spawn_probability, int generation);
__global__ void culling_kernel(ComponentPool* pool, float fitness_threshold, float hunger_threshold);
__global__ void svd_kernel(float* matrix, float* singular_values, int size);
__global__ void coherence_computation_kernel(float* genome, float* chemical_field_prev, float* chemical_field_current, int field_size, float* coherence_out);
__global__ void effective_rank_from_svd_kernel(float* singular_values, float* fitness_out, float* coherence_out, int num_values);
__global__ void jacobi_sweep_kernel(float* matrix, float* workspace, int size, int sweep);
__global__ void initialize_ca_from_field_kernel(float* ca_state, float* chemical_concentration, int grid_size);
__global__ void update_field_from_ca_kernel(float* chemical_concentration, float* ca_state, int grid_size);
__global__ void store_navigation_history_kernel(Organism* organism, BehavioralState* agents, TemporalTube* tubes, int generation);

// Level 1: Top-level organism kernel (launches Level 2 kernels)
__global__ void organism_lifecycle_kernel(
    Organism* organism,
    int generation
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Generation counter handled by host before kernel launch

        dim3 component_grid((MAX_COMPONENTS + 255) / 256);
        dim3 component_block(256);

        // Launch Level 2 kernels for each subsystem

        // Component evolution (Level 2)
        component_evolution_kernel<<<component_grid, component_block>>>(
            organism,
            organism->pool,
            organism->archive,
            organism->voronoi_cells,
            organism->num_voronoi_cells,
            &organism->archive_size,
            organism->chemical_field,
            organism->behavioral_agents,
            organism->fitness_history,
            organism->coherence_history,
            generation
        );

        // Neural CA update (Level 2)
        dim3 ca_grid(GRID_SIZE / 16, NUM_HEADS, 1);
        dim3 ca_block(16, 16, 1);
        neural_ca_update_kernel<<<ca_grid, ca_block>>>(
            organism->ca_state,
            organism->chemical_field,
            organism->effective_rank_history,
            generation
        );

        // Behavioral navigation (Level 2)
        behavioral_update_kernel<<<component_grid, component_block>>>(
            organism,
            organism->behavioral_agents,
            organism->chemical_field,
            organism->memory_tubes,
            generation
        );

        // Memory consolidation (Level 2)
        memory_update_kernel<<<1, 256>>>(
            organism->memory_tubes,
            organism->fitness_history,
            organism->coherence_history,
            generation
        );

        // Child kernels complete before parent returns
    }
}

// Level 2: Component evolution (launches Level 3 kernels)
__global__ void component_evolution_kernel(
    Organism* organism,
    ComponentPool* pool,
    GPUElite* archive,
    VoronoiCell* voronoi_cells,
    int num_cells,
    int* archive_size,
    ChemicalField* chemical_field,
    BehavioralState* behavioral_agents,
    float* fitness_history,
    float* coherence_history,
    int generation
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Compute fitness for all components (Level 3)
        // Each warp processes one organism, 8 warps per block of 256 threads
        int warps_per_block = 256 / 32;  // 8 warps
        int num_blocks = (pool->capacity + warps_per_block - 1) / warps_per_block;
        fitness_computation_kernel<<<num_blocks, 256>>>(
            pool,
            chemical_field,
            fitness_history,
            coherence_history,
            generation
        );
        // Parent waits for child kernels implicitly

        // Selection and reproduction (Level 3)
        selection_kernel<<<1, 32>>>(
            organism,
            pool,
            archive,
            voronoi_cells,
            num_cells,
            archive_size,
            behavioral_agents,
            generation
        );
        // Dynamic parallelism: parent waits for children

        // Spawn new components (Level 3)
        float spawn_prob = SPAWN_RATE * expf(-pool->active_count.load() / (float)MAX_COMPONENTS);
        if (spawn_prob > 0.01f) {
            spawn_wave_kernel<<<1, 32>>>(
                pool,
                spawn_prob,
                generation
            );
        }

        // Cull weak/hungry components (Level 3)
        culling_kernel<<<(pool->capacity + 255) / 256, 256>>>(
            pool,
            FITNESS_THRESHOLD,
            HUNGER_DEATH_THRESHOLD
        );
    }
}

// Level 3: Fitness computation (launches Level 4 kernels)
__global__ void fitness_computation_kernel(
    ComponentPool* pool,
    ChemicalField* chemical_field,
    float* fitness_history,
    float* coherence_history,
    int generation
) {
    // Each warp processes one organism collaboratively
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_block = blockDim.x / 32;
    const int idx = blockIdx.x * warps_per_block + warp_id;

    if (idx >= pool->capacity || !pool->entries[idx].alive) return;

    // Warp-level fitness computation - no memory allocations, pure register operations

    float* genome = pool->entries[idx].genome;
    
    // Compute genome L2 norm via warp reduction (fitness proxy)
    float local_sum = 0.0f;
    for (int i = lane; i < GENOME_SIZE; i += 32) {
        float val = genome[i];
        local_sum += val * val;
    }
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    float genome_magnitude = __shfl_sync(0xffffffff, local_sum, 0);
    
    // Compute coherence as correlation between genome and chemical field
    float* curr_field = chemical_field->concentration;
    int field_size = GRID_SIZE * GRID_SIZE;
    int sample_stride = field_size / GENOME_SIZE;
    
    float correlation = 0.0f;
    for (int i = lane; i < GENOME_SIZE; i += 32) {
        int field_idx = i * sample_stride;
        if (field_idx < field_size) {
            correlation += genome[i] * curr_field[field_idx];
        }
    }
    
    // Warp reduction for correlation
    for (int offset = 16; offset > 0; offset /= 2) {
        correlation += __shfl_down_sync(0xffffffff, correlation, offset);
    }
    
    float coherence_val = __shfl_sync(0xffffffff, correlation, 0);
    coherence_val = tanhf(coherence_val / (sqrtf(genome_magnitude) + 1e-6f));
    coherence_val = (coherence_val + 1.0f) * 0.5f;  // Map [-1,1] to [0,1]
    
    // Effective rank approximation via genome entropy
    float entropy = 0.0f;
    for (int i = lane; i < GENOME_SIZE; i += 32) {
        float p = fabsf(genome[i]) / (sqrtf(genome_magnitude) + 1e-6f);
        if (p > 1e-6f) {
            entropy -= p * log2f(p);
        }
    }
    
    // Warp reduction for entropy
    for (int offset = 16; offset > 0; offset /= 2) {
        entropy += __shfl_down_sync(0xffffffff, entropy, offset);
    }
    
    float effective_rank = __shfl_sync(0xffffffff, entropy, 0);
    
    // Only lane 0 writes results
    if (lane == 0) {
        float fitness_val = effective_rank * coherence_val;
        
        pool->entries[idx].fitness = fitness_val;
        pool->entries[idx].coherence = coherence_val;
        pool->entries[idx].hunger = 1.0f - coherence_val;
        
        int history_idx = generation * pool->capacity + idx;
        fitness_history[history_idx] = fitness_val;
        coherence_history[history_idx] = coherence_val;
        
        // Debug: print first organism computation on first generation
        if (idx == 0 && generation == 0) {
            printf("[WARP_COMPUTE] idx=%d, genome_mag=%.4f, coherence=%.4f, entropy=%.4f, fitness=%.4f\n",
                   idx, genome_magnitude, coherence_val, effective_rank, fitness_val);
        }
    }
}

// Level 4: SVD computation (launches Level 5 Jacobi rotations)
__global__ void svd_kernel(
    float* matrix,
    float* singular_values,
    int size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Launch Level 5 Jacobi rotation kernels
        for (int sweep = 0; sweep < MAX_JACOBI_SWEEPS; sweep++) {
            jacobi_sweep_kernel<<<1, size>>>(
                matrix,
                singular_values,
                size,
                sweep
            );
            // Dynamic parallelism: parent waits for children
        }

        // Extract singular values
        for (int i = 0; i < min(256, size); i++) {
            singular_values[i] = sqrtf(fabsf(matrix[i * size + i]));
        }
    }
}

// Level 5: Jacobi rotation sweep (deepest level)
__global__ void jacobi_sweep_kernel(
    float* matrix,
    float* workspace,
    int size,
    int sweep
) {
    int tid = threadIdx.x;
    if (tid >= size) return;

    // Jacobi rotations for this sweep
    for (int p = 0; p < size - 1; p++) {
        for (int q = p + 1; q < size; q++) {
            if ((p + q + sweep) % 3 == 0) {  // Cyclic sweep pattern
                float app = matrix[p * size + p];
                float aqq = matrix[q * size + q];
                float apq = matrix[p * size + q];

                // Compute rotation angle
                float tau = (aqq - app) / (2.0f * apq + 1e-10f);
                float t = (tau >= 0.0f) ?
                    1.0f / (tau + sqrtf(1.0f + tau * tau)) :
                    -1.0f / (-tau + sqrtf(1.0f + tau * tau));

                float c = 1.0f / sqrtf(1.0f + t * t);
                float s = t * c;

                // Apply rotation to row tid
                if (tid != p && tid != q) {
                    float aip = matrix[tid * size + p];
                    float aiq = matrix[tid * size + q];
                    matrix[tid * size + p] = c * aip - s * aiq;
                    matrix[tid * size + q] = s * aip + c * aiq;
                }
            }
        }
    }
}

// Level 4: Coherence computation - measures genome's ability to predict chemical field evolution
__global__ void coherence_computation_kernel(
    float* genome,
    float* chemical_field_prev,
    float* chemical_field_current,
    int field_size,
    float* coherence_out
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float error_sum;
    if (threadIdx.x == 0) {
        error_sum = 0.0f;
    }
    __syncthreads();

    // Use genome to predict next field state from previous state
    // Genome acts as a linear prediction matrix
    if (tid < field_size) {
        float prediction = 0.0f;

        // Linear combination of genome weights with previous field values
        // Sample subset of field for efficiency (every 8th position)
        int step = field_size / min(GENOME_SIZE, field_size);
        for (int i = 0; i < min(GENOME_SIZE, field_size / step); i++) {
            int field_idx = i * step;
            if (field_idx < field_size) {
                prediction += genome[i] * chemical_field_prev[field_idx];
            }
        }

        // Compute squared prediction error
        float actual = chemical_field_current[tid];
        float error = (prediction - actual) * (prediction - actual);

        // Accumulate error
        atomicAdd(&error_sum, error);
    }
    __syncthreads();

    // Coherence is inverse of mean squared error (higher is better)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float mse = error_sum / field_size;
        *coherence_out = 1.0f / (1.0f + mse);  // Maps [0, inf) error to (0, 1] coherence
    }
}

// Level 4: Effective rank from SVD
__global__ void effective_rank_from_svd_kernel(
    float* singular_values,
    float* fitness_out,
    float* coherence_out,
    int num_values
) {
    // Compute normalized singular value distribution
    float sum = 0.0f;
    for (int i = 0; i < num_values; i++) {
        sum += singular_values[i];
    }

    float entropy = 0.0f;
    for (int i = 0; i < num_values; i++) {
        float p = singular_values[i] / (sum + 1e-10f);
        if (p > 1e-10f) {
            entropy -= p * logf(p);
        }
    }

    float effective_rank = expf(entropy);

    // FITNESS = effective_rank × coherence
    *fitness_out = effective_rank * (*coherence_out);
}

// Level 3: Selection and archiving
__global__ void selection_kernel(
    Organism* organism,
    ComponentPool* pool,
    GPUElite* archive,
    VoronoiCell* voronoi_cells,
    int num_cells,
    int* archive_size,
    BehavioralState* behavioral_agents,
    int generation
) {
    int tid = threadIdx.x;

    // Find best components
    __shared__ float best_fitness[32];
    __shared__ int best_indices[32];

    best_fitness[tid] = -1.0f;
    best_indices[tid] = -1;

    // Each thread scans a portion
    int chunk_size = pool->capacity / 32;
    for (int i = tid * chunk_size; i < (tid + 1) * chunk_size; i++) {
        if (i < pool->capacity && pool->entries[i].alive) {
            if (pool->entries[i].fitness > best_fitness[tid]) {
                best_fitness[tid] = pool->entries[i].fitness;
                best_indices[tid] = i;
            }
        }
    }
    __syncthreads();

    // Reduction to find global best
    for (int stride = 16; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (best_fitness[tid + stride] > best_fitness[tid]) {
                best_fitness[tid] = best_fitness[tid + stride];
                best_indices[tid] = best_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    // Archive the best
    if (tid == 0 && best_indices[0] >= 0) {
        PoolEntry* best = &pool->entries[best_indices[0]];
        BehavioralState* agent = &behavioral_agents[best_indices[0]];

        // Use pre-allocated compressed genome pool
        uint8_t* d_compressed = organism->compressed_genome_pool + (generation % MAX_ARCHIVE_SIZE) * GENOME_SIZE;
        uint32_t* d_compressed_size = organism->compressed_size_pool + (generation % MAX_ARCHIVE_SIZE);

        // Compress genome using SVD-based compression
        compress_genome_kernel<<<1, 256>>>(
            best->genome,
            d_compressed,
            d_compressed_size,
            GENOME_SIZE,
            MAX_RANK
        );

        // Use pre-allocated elite staging pool
        GPUElite* d_elite = organism->elite_staging_pool + (generation % MAX_ARCHIVE_SIZE);
        
        // Create elite from best component
        GPUElite elite;
        elite.fitness = best->fitness;
        elite.coherence = best->coherence;
        elite.effective_rank = best->fitness / (best->coherence + 1e-10f);
        elite.genome_hash = gpu_sha256(best->genome, GENOME_SIZE);
        elite.generation = generation;
        elite.compressed_genome = d_compressed;
        elite.compressed_size = 0;  // Will be read from d_compressed_size after kernel completes

        // Copy behavioral_coords from agent's DIRESA embedding
        for (int i = 0; i < 10; i++) {
            elite.behavioral_coords[i] = agent->behavioral_coords[i];
        }

        // Extract raw behavioral metrics from agent state
        elite.raw_metrics[0] = agent->position[0];
        elite.raw_metrics[1] = agent->position[1];
        elite.raw_metrics[2] = agent->velocity[0];
        elite.raw_metrics[3] = agent->velocity[1];
        elite.raw_metrics[4] = agent->exploration_noise;
        elite.raw_metrics[5] = agent->sensitivity;

        // Gradient memory statistics (32 * 2 = 64)
        for (int i = 0; i < GRADIENT_HISTORY; i++) {
            elite.raw_metrics[6 + i * 2] = agent->gradient_memory[i][0];
            elite.raw_metrics[6 + i * 2 + 1] = agent->gradient_memory[i][1];
        }

        // Fitness metrics (4)
        elite.raw_metrics[70] = best->fitness;
        elite.raw_metrics[71] = best->coherence;
        elite.raw_metrics[72] = best->hunger;
        elite.raw_metrics[73] = (float)best->age;
        
        // Copy to device memory manually (device-to-device)
        d_elite->fitness = elite.fitness;
        d_elite->coherence = elite.coherence;
        d_elite->effective_rank = elite.effective_rank;
        d_elite->genome_hash = elite.genome_hash;
        d_elite->generation = elite.generation;
        d_elite->compressed_genome = elite.compressed_genome;
        d_elite->compressed_size = elite.compressed_size;
        for (int i = 0; i < 10; i++) {
            d_elite->behavioral_coords[i] = elite.behavioral_coords[i];
        }
        for (int i = 0; i < 75; i++) {
            d_elite->raw_metrics[i] = elite.raw_metrics[i];
        }

        // Add to archive (with deduplication check)
        insert_elite_kernel<<<1, 1>>>(
            archive,
            archive_size,
            d_elite,
            voronoi_cells,
            num_cells
        );
        
        // No cudaFree needed - using pre-allocated pools
    }
}

// Level 3: Spawn wave
__global__ void spawn_wave_kernel(
    ComponentPool* pool,
    float spawn_probability,
    int generation
) {
    int tid = threadIdx.x;

    // Each thread attempts to spawn
    unsigned int seed = tid * generation * 1337;
    seed = seed * 1664525u + 1013904223u;
    float rand = (seed & 0xFFFFFF) / 16777216.0f;

    if (rand < spawn_probability) {
        // Find a parent
        int parent_idx = -1;
        for (int i = 0; i < pool->capacity; i++) {
            if (pool->entries[i].alive && pool->entries[i].fitness > 0.5f) {
                parent_idx = i;
                break;
            }
        }

        if (parent_idx >= 0) {
            spawn_component_kernel<<<1, 1>>>(
                pool,
                parent_idx,
                0.01f  // Mutation rate
            );
        }
    }
}

// Level 3: Culling
__global__ void culling_kernel(
    ComponentPool* pool,
    float fitness_threshold,
    float hunger_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[idx];

    if (entry->alive) {
        // Cull based on fitness
        if (entry->fitness < fitness_threshold * 0.1f) {  // 10% of threshold
            entry->alive = false;
            pool->total_culled.fetch_add(1);
            pool->active_count.fetch_sub(1);
        }
        // Cull based on hunger (low coherence)
        else if (entry->hunger > hunger_threshold) {
            entry->alive = false;
            pool->total_culled.fetch_add(1);
            pool->active_count.fetch_sub(1);
        }
    }
}

// Level 2: Neural CA update
__global__ void neural_ca_update_kernel(
    MultiHeadCAState* ca_state,
    ChemicalField* chemical_field,
    float* effective_rank_history,
    int generation
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Use pre-allocated CA buffers from ca_state
        float* ca_input = ca_state->ca_input;
        float* ca_output = ca_state->ca_output;

        // Initialize from chemical field
        dim3 init_grid((GRID_SIZE + 15) / 16, (GRID_SIZE + 15) / 16);
        dim3 init_block(16, 16);

        initialize_ca_from_field_kernel<<<init_grid, init_block>>>(
            ca_input,
            chemical_field->concentration,
            GRID_SIZE
        );

        // Run multi-head CA (Level 3)
        dim3 ca_grid(GRID_SIZE / 16, NUM_HEADS, 1);
        dim3 ca_block(16, 16, 1);

        multi_head_ca_kernel<<<ca_grid, ca_block>>>(
            ca_input,
            ca_state->perception_weights,
            ca_state->interaction_weights,
            ca_state->value_weights,
            ca_output,
            1,  // batch size
            GRID_SIZE
        );

        // Apply Flow-Lenia dynamics (Level 3)
        flow_lenia_dynamics_kernel<<<init_grid, init_block>>>(
            ca_input,
            ca_output,
            ca_state->flow_kernels,
            ca_state->mass_buffer,
            GRID_SIZE,
            0.1f  // dt
        );

        // Mix heads (Level 3)
        mix_heads_kernel<<<init_grid, init_block>>>(
            ca_output,
            ca_state->head_mixing_weights,
            ca_input,  // Reuse as final output
            1,
            GRID_SIZE
        );

        // Update chemical field from CA
        update_field_from_ca_kernel<<<init_grid, init_block>>>(
            chemical_field->concentration,
            ca_input,
            GRID_SIZE
        );

        // Compute effective rank (Level 3)
        compute_effective_rank_kernel<<<1, 256>>>(
            ca_state->perception_weights,
            &effective_rank_history[generation],
            NUM_HEADS * CHANNELS * HIDDEN_DIM
        );

        // Dynamic parallelism: parent waits for children
        // CA buffers are persistent and freed in destroy_organism
    }
}

// Helper: Initialize CA from chemical field
__global__ void initialize_ca_from_field_kernel(
    float* ca_state,
    float* chemical_concentration,
    int grid_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int field_idx = y * grid_size + x;
    float concentration = chemical_concentration[field_idx];

    // Initialize all channels based on concentration
    for (int c = 0; c < CHANNELS; c++) {
        int ca_idx = (y * grid_size + x) * CHANNELS + c;

        // Different initialization for different channel groups
        if (c < NUM_HEADS * HEAD_DIM) {
            // Direct mapping for head channels
            ca_state[ca_idx] = concentration * sinf(c * 0.1f);
        } else {
            // Random initialization for other channels
            unsigned int seed = x * grid_size + y + c * 1337;
            seed = seed * 1664525u + 1013904223u;
            float rand = (seed & 0xFFFFFF) / 16777216.0f;
            ca_state[ca_idx] = concentration * rand;
        }
    }
}

// Helper: Update chemical field from CA
__global__ void update_field_from_ca_kernel(
    float* chemical_concentration,
    float* ca_state,
    int grid_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int field_idx = y * grid_size + x;

    // Average across all channels
    float sum = 0.0f;
    for (int c = 0; c < CHANNELS; c++) {
        int ca_idx = (y * grid_size + x) * CHANNELS + c;
        sum += ca_state[ca_idx];
    }

    chemical_concentration[field_idx] = sum / CHANNELS;
}

// Level 2: Behavioral update
__global__ void behavioral_update_kernel(
    Organism* organism,
    BehavioralState* agents,
    ChemicalField* chemical_field,
    TemporalTube* memory_tubes,
    int generation
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int num_agents = MAX_COMPONENTS;

        // Update chemical field (Level 3)
        dim3 field_grid((GRID_SIZE + 15) / 16, (GRID_SIZE + 15) / 16);
        dim3 field_block(16, 16);

        diffusion_reaction_kernel<<<field_grid, field_block>>>(
            chemical_field->concentration,
            chemical_field->gradient_x,
            chemical_field->gradient_y,
            chemical_field->laplacian,
            chemical_field->sources,
            GRID_SIZE,
            0.01f  // dt
        );

        // Store chemical field snapshot after diffusion
        int field_size = GRID_SIZE * GRID_SIZE;
        float global_time = (float)generation;
        store_chemical_snapshot_kernel<<<field_grid, field_block>>>(
            chemical_field,
            field_size,
            global_time
        );

        // Use pre-allocated behavioral pools
        float* behavioral_field = organism->behavioral_field_pool;
        float* behavioral_gradients = organism->behavioral_gradient_pool;

        // Compute behavioral field from agent embeddings (Level 3)
        compute_behavioral_field_kernel<<<field_grid, field_block>>>(
            behavioral_field,
            agents,
            num_agents,
            GRID_SIZE
        );

        // Compute behavioral gradients (Level 3)
        dim3 grad_grid((GRID_SIZE + 15) / 16, (GRID_SIZE + 15) / 16, BEHAVIORAL_DIM);
        behavioral_gradient_kernel<<<grad_grid, field_block>>>(
            behavioral_field,
            behavioral_gradients,
            GRID_SIZE
        );

        // Navigate agents (Level 3)
        chemotactic_navigation_kernel<<<(num_agents + 255) / 256, 256>>>(
            agents,
            chemical_field->concentration,
            chemical_field->gradient_x,
            chemical_field->gradient_y,
            behavioral_gradients,
            num_agents,
            GRID_SIZE,
            0.01f  // dt
        );

        // Store navigation history in memory tubes (Level 3)
        store_navigation_history_kernel<<<1, 256>>>(
            organism,
            agents,
            memory_tubes,
            generation
        );

        // No cudaFree needed - using pre-allocated pools

        // Dynamic parallelism: parent waits for children
    }
}

// Helper: Store navigation history
__global__ void store_navigation_history_kernel(
    Organism* organism,
    BehavioralState* agents,
    TemporalTube* tubes,
    int generation
) {
    int tid = threadIdx.x;
    if (tid >= MAX_COMPONENTS) return;

    // Use pre-allocated memory data pool
    float* d_memory_data = organism->memory_data_pool + tid * (BEHAVIORAL_DIM + 4);
    
    float memory_data[BEHAVIORAL_DIM + 4];

    // Position and velocity
    memory_data[0] = agents[tid].position[0];
    memory_data[1] = agents[tid].position[1];
    memory_data[2] = agents[tid].velocity[0];
    memory_data[3] = agents[tid].velocity[1];

    // Behavioral coordinates
    for (int i = 0; i < BEHAVIORAL_DIM; i++) {
        memory_data[4 + i] = agents[tid].behavioral_coords[i];
    }
    
    // Copy to device memory
    for (int i = 0; i < BEHAVIORAL_DIM + 4; i++) {
        d_memory_data[i] = memory_data[i];
    }

    // Store with importance based on exploration
    float importance = agents[tid].exploration_noise;

    if (tid == 0) {
        store_memory_kernel<<<1, 1>>>(
            tubes,
            d_memory_data,
            BEHAVIORAL_DIM + 4,
            importance
        );
    }
}

// Level 2: Memory update
__global__ void memory_update_kernel(
    TemporalTube* tubes,
    float* fitness_history,
    float* coherence_history,
    int generation
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Apply decay (Level 3)
        apply_decay_kernel<<<(tubes->count + 255) / 256, 256>>>(
            tubes,
            0.01f  // timestep
        );

        // Prune old memories (Level 3)
        prune_memories_kernel<<<(tubes->count + 255) / 256, 256>>>(
            tubes,
            0.01f  // threshold
        );

        // Consolidate similar memories (Level 3)
        consolidate_memories_kernel<<<1, min(256, tubes->count)>>>(
            tubes,
            0.8f  // similarity threshold
        );

        // Dynamic parallelism: parent waits for children
    }
}

// Initialize organism
__global__ void init_organism_kernel(
    Organism* organism,
    unsigned int seed
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize all subsystems
        organism->generation = 0;
        organism->active_components = MIN_POOL_SIZE;
        organism->learning_rate = 0.001f;
        organism->mutation_rate = 0.01f;
        organism->exploration_rate = 0.5f;

        // Initialize component pool
        init_pool_kernel<<<(MAX_POOL_SIZE + 255) / 256, 256>>>(
            organism->pool,
            MAX_POOL_SIZE
        );

        // Initialize memory tubes
        init_tube_kernel<<<(MAX_MEMORY_SIZE + 255) / 256, 256>>>(
            organism->memory_tubes,
            MAX_MEMORY_SIZE,
            DEFAULT_DECAY_RATE
        );

        // Initialize multi-head CA
        init_multihead_ca_kernel<<<(NUM_HEADS * CHANNELS * HIDDEN_DIM + 255) / 256, 256>>>(
            organism->ca_state,
            seed
        );

        // Initialize behavioral agents
        init_behavioral_state_kernel<<<(MAX_COMPONENTS + 255) / 256, 256>>>(
            organism->behavioral_agents,
            MAX_COMPONENTS,
            seed
        );

        // Initialize chemical field
        dim3 chem_grid((GRID_SIZE + 15) / 16, (GRID_SIZE + 15) / 16);
        dim3 chem_block(16, 16);
        init_chemical_field_kernel<<<chem_grid, chem_block>>>(
            organism->chemical_field,
            GRID_SIZE
        );

        // Dynamic parallelism: child kernel synchronizes implicitly when parent returns
        
        // Then set sources from agent positions
        set_chemical_sources_from_agents_kernel<<<1, MAX_COMPONENTS>>>(
            organism->chemical_field->sources,
            organism->behavioral_agents,
            MAX_COMPONENTS,
            GRID_SIZE
        );
        
        // Run initial diffusion to populate concentration from sources
        diffusion_reaction_kernel<<<chem_grid, chem_block>>>(
            organism->chemical_field->concentration,
            organism->chemical_field->gradient_x,
            organism->chemical_field->gradient_y,
            organism->chemical_field->laplacian,
            organism->chemical_field->sources,
            GRID_SIZE,
            0.1f  // larger dt for initial spread
        );

        // Store initial chemical field snapshot
        int field_size = GRID_SIZE * GRID_SIZE;
        store_chemical_snapshot_kernel<<<chem_grid, chem_block>>>(
            organism->chemical_field,
            field_size,
            0.0f  // initial time
        );

        // Dynamic parallelism: parent waits for children
    }
}

// Main evolution loop
extern "C" void evolve_organism(Organism* d_organism, int num_generations) {
    for (int gen = 0; gen < num_generations; gen++) {
        // Launch top-level kernel with dynamic parallelism
        organism_lifecycle_kernel<<<1, 1>>>(d_organism, gen);
        // Dynamic parallelism: parent waits for children

        // Check for convergence
        float fitness, coherence;
        cudaMemcpy(&fitness, &d_organism->fitness_history[gen * MAX_COMPONENTS],
                  sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&coherence, &d_organism->coherence_history[gen * MAX_COMPONENTS],
                  sizeof(float), cudaMemcpyDeviceToHost);

        if (fitness > FITNESS_THRESHOLD && coherence > 0.9f) {
            printf("Converged at generation %d: fitness=%.3f, coherence=%.3f\n",
                  gen, fitness, coherence);
            break;
        }
    }
}

#endif // ORGANISM_CU

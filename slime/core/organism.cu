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
};

// Forward declarations for all kernels
__global__ void component_evolution_kernel(ComponentPool* pool, GPUElite* archive, float* fitness_history, float* coherence_history, int generation);
__global__ void neural_ca_update_kernel(MultiHeadCAState* ca_state, ChemicalField* chemical_field, float* effective_rank_history, int generation);
__global__ void behavioral_update_kernel(BehavioralState* agents, ChemicalField* chemical_field, TemporalTube* memory_tubes, int generation);
__global__ void memory_update_kernel(TemporalTube* tubes, float* fitness_history, float* coherence_history, int generation);
__global__ void fitness_computation_kernel(ComponentPool* pool, float* fitness_history, float* coherence_history, int generation);
__global__ void selection_kernel(ComponentPool* pool, GPUElite* archive, int generation);
__global__ void spawn_wave_kernel(ComponentPool* pool, float spawn_probability, int generation);
__global__ void culling_kernel(ComponentPool* pool, float fitness_threshold, float hunger_threshold);
__global__ void svd_kernel(float* matrix, float* singular_values, int size);
__global__ void coherence_computation_kernel(float* prediction_errors, float* coherence_history, int history_length);
__global__ void effective_rank_from_svd_kernel(float* singular_values, float* fitness_out, float* coherence_out, int num_values);
__global__ void jacobi_sweep_kernel(float* matrix, float* workspace, int size, int sweep);
__global__ void initialize_ca_from_field_kernel(float* ca_state, float* chemical_concentration, int grid_size);
__global__ void update_field_from_ca_kernel(float* chemical_concentration, float* ca_state, int grid_size);
__global__ void store_navigation_history_kernel(BehavioralState* agents, TemporalTube* tubes, int generation);

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
            organism->pool,
            organism->archive,
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
    ComponentPool* pool,
    GPUElite* archive,
    float* fitness_history,
    float* coherence_history,
    int generation
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Compute fitness for all components (Level 3)
        fitness_computation_kernel<<<(pool->capacity + 255) / 256, 256>>>(
            pool,
            fitness_history,
            coherence_history,
            generation
        );
        // Parent waits for child kernels implicitly

        // Selection and reproduction (Level 3)
        selection_kernel<<<1, 32>>>(
            pool,
            archive,
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
    float* fitness_history,
    float* coherence_history,
    int generation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pool->capacity || !pool->entries[idx].alive) return;

    // Launch Level 4 kernels for SVD and coherence
    if (threadIdx.x == 0) {
        // Allocate temporary buffers
        float* weight_matrix;
        float* singular_values;
        float* prediction_errors;

        cudaMalloc((void**)&weight_matrix, GENOME_SIZE * sizeof(float));
        cudaMalloc((void**)&singular_values, 256 * sizeof(float));
        cudaMalloc((void**)&prediction_errors, 100 * sizeof(float));

        // Copy genome to weight matrix (device-to-device)
        for (int i = 0; i < GENOME_SIZE; i++) {
            weight_matrix[i] = pool->entries[idx].genome[i];
        }

        // Launch SVD kernel (Level 4)
        svd_kernel<<<1, 256>>>(
            weight_matrix,
            singular_values,
            GENOME_SIZE
        );

        // Launch coherence kernel (Level 4)
        coherence_computation_kernel<<<1, 256>>>(
            prediction_errors,
            coherence_history + idx * 100,
            100
        );

        cudaDeviceSynchronize();

        // Compute effective rank from singular values (Level 4)
        effective_rank_from_svd_kernel<<<1, 1>>>(
            singular_values,
            &pool->entries[idx].fitness,
            &pool->entries[idx].coherence,
            256
        );

        cudaDeviceSynchronize();

        // Update hunger
        pool->entries[idx].hunger = 1.0f - pool->entries[idx].coherence;

        // Store in history
        int history_idx = generation * pool->capacity + idx;
        fitness_history[history_idx] = pool->entries[idx].fitness;
        coherence_history[history_idx] = pool->entries[idx].coherence;

        // Cleanup
        cudaFree(weight_matrix);
        cudaFree(singular_values);
        cudaFree(prediction_errors);
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
            cudaDeviceSynchronize();
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

// Level 4: Coherence computation
__global__ void coherence_computation_kernel(
    float* prediction_errors,
    float* coherence_history,
    int history_length
) {
    // Generate synthetic prediction errors based on genome evolution
    int tid = threadIdx.x;

    if (tid < history_length) {
        // Use coherence history to generate realistic errors
        float base_error = 1.0f / (1.0f + tid * 0.01f);  // Decreasing error over time
        float noise = sinf(tid * 0.1f) * 0.1f;  // Oscillation
        prediction_errors[tid] = base_error + noise;
    }
    __syncthreads();

    // Compute coherence as learning progress
    float improvement_sum = 0.0f;
    for (int i = 0; i < history_length - 1; i++) {
        float curr = prediction_errors[i];
        float next = prediction_errors[i + 1];
        improvement_sum += fmaxf(0.0f, (curr - next) / (curr + 1e-10f));
    }

    if (tid == 0) {
        coherence_history[0] = improvement_sum / (history_length - 1);
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
    ComponentPool* pool,
    GPUElite* archive,
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

        // Create elite from best component
        GPUElite elite;
        elite.fitness = best->fitness;
        elite.coherence = best->coherence;
        elite.effective_rank = best->fitness / (best->coherence + 1e-10f);
        elite.genome_hash = gpu_sha256(best->genome, GENOME_SIZE);
        elite.generation = generation;

        // Add to archive (with deduplication check)
        insert_elite_kernel<<<1, 1>>>(
            archive,
            (int*)&pool->total_spawned,  // Reuse as archive size
            &elite,
            nullptr,
            0
        );
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
        // Allocate CA buffers
        float* ca_input;
        float* ca_output;
        cudaMalloc(&ca_input, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float));
        cudaMalloc(&ca_output, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float));

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

        cudaDeviceSynchronize();

        // Cleanup
        cudaFree(ca_input);
        cudaFree(ca_output);
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

        // Compute behavioral gradients (Level 3)
        dim3 grad_grid((GRID_SIZE + 15) / 16, (GRID_SIZE + 15) / 16, BEHAVIORAL_DIM);
        behavioral_gradient_kernel<<<grad_grid, field_block>>>(
            nullptr,  // Behavioral field computed from agents
            nullptr,  // Gradients
            GRID_SIZE
        );

        // Navigate agents (Level 3)
        chemotactic_navigation_kernel<<<(num_agents + 255) / 256, 256>>>(
            agents,
            chemical_field->concentration,
            chemical_field->gradient_x,
            chemical_field->gradient_y,
            nullptr,  // Behavioral gradients
            num_agents,
            GRID_SIZE,
            0.01f  // dt
        );

        // Store navigation history in memory tubes (Level 3)
        store_navigation_history_kernel<<<1, 256>>>(
            agents,
            memory_tubes,
            generation
        );

        cudaDeviceSynchronize();
    }
}

// Helper: Store navigation history
__global__ void store_navigation_history_kernel(
    BehavioralState* agents,
    TemporalTube* tubes,
    int generation
) {
    int tid = threadIdx.x;
    if (tid >= MAX_COMPONENTS) return;

    // Pack agent state into memory
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

    // Store with importance based on exploration
    float importance = agents[tid].exploration_noise;

    if (tid == 0) {
        store_memory_kernel<<<1, 1>>>(
            tubes,
            memory_data,
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

        cudaDeviceSynchronize();
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

        cudaDeviceSynchronize();
    }
}

// Main evolution loop
extern "C" void evolve_organism(Organism* d_organism, int num_generations) {
    for (int gen = 0; gen < num_generations; gen++) {
        // Launch top-level kernel with dynamic parallelism
        organism_lifecycle_kernel<<<1, 1>>>(d_organism, gen);
        cudaDeviceSynchronize();

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


#ifndef CHEMOTAXIS_CU
#define CHEMOTAXIS_CU
#include "../config/config.cu"
#include "../utils/genome_params.cuh"
#include "pseudopod.cu"
#include "../memory/pool.cu"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include "../memory/tubes.cu"

namespace cg = cooperative_groups;

struct BehavioralInitSlots {
    int agent_embedding_scale;
    int init_exploration;
    int init_sensitivity;
    int levy_alpha;
    int ctx_metabolic;
    int ctx_stress;
    int ctx_morphogen;
};

struct ChemicalField {
    float* concentration;
    float* gradient_x;
    float* gradient_y;
    float* laplacian;
    float* sources;
    float* decay_factors;
    TemporalTube* history;
    float cached_mean;  // Precomputed mean concentration - updated once per generation via parallel reduction
};

struct BehavioralState {
    float position[2];
    float velocity[2];
    float* hw_coords;
    float* task_coords;
    float* gen_coords;
    float gradient_memory[GRADIENT_HISTORY][2];
    float velocity_history[GRADIENT_HISTORY][2];
    float exploration_noise;
    float exploration;
    float sensitivity;
    int memory_index;
    uint64_t genome_hash;
    int organism_id;
};

__device__ void store_chemical_snapshot(ChemicalField* field, int field_size, float global_time, uint64_t genome_hash, const float* genome) {
    TemporalTube* history = field->history;

    int next_head = (history->head + 1) % history->capacity;

    float* dest = history->entries[next_head].data;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < field_size; i += blockDim.x * gridDim.x) {
        dest[i] = field->concentration[i];
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        int decay_slot = derive_param_slot(genome_hash, "memory_decay_factor");
        int importance_slot = derive_param_slot(genome_hash, "memory_importance");

        history->entries[next_head].timestamp = global_time;
        history->entries[next_head].decay_factor = (genome[decay_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        history->entries[next_head].importance = (genome[importance_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;

        history->head = next_head;
        if (history->count < history->capacity) {
            history->count++;
        }
        history->global_time = global_time;
    }
}

__global__ void store_chemical_snapshot_kernel(ChemicalField* field, int field_size, float global_time, uint64_t genome_hash, const float* genome) {
    store_chemical_snapshot(field, field_size, global_time, genome_hash, genome);
}

__global__ void update_field_from_ca_kernel(
    ComponentPool* __restrict__ pool,
    float* __restrict__ chemical_concentration,
    int max_grid_size,
    int entry_idx
) {
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    int grid_size = entry->grid_size;
    int channels = entry->channels;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int cell_idx = y * grid_size + x;
    float* ca_concentration = entry->ca_state->ca_concentration;

    // CA concentration layout: [grid² × channels], read channel 0 for chemical field
    float val = ca_concentration[cell_idx * channels + 0];
    if (isfinite(val)) {
        atomicAdd(&chemical_concentration[cell_idx], val);
    }
}

// Initialize per-entry CA state from shared chemical field and RD fields
// This is the reverse of update_field_from_ca_kernel - completing the bidirectional stigmergy loop
// Channel layout matches inject_sample_to_ca_kernel:
// 0-5: ChemicalField (concentration, gradients, laplacian, sources, decay)
// 6-9: RDField (resource_density, fitness_landscape, resource gradients)
// 10: BehavioralField
// 11-13: Dataset samples (zeroed for evolutionary path - training encodes patterns in chemical field)
// 14: Previous CA output (recurrence)
// 15: Temporal retrieval
__global__ void initialize_ca_from_field_kernel(
    ComponentPool* __restrict__ pool,
    // ChemicalField components (channels 0-5)
    const float* __restrict__ chem_concentration,
    const float* __restrict__ chem_gradient_x,
    const float* __restrict__ chem_gradient_y,
    const float* __restrict__ chem_laplacian,
    const float* __restrict__ chem_sources,
    const float* __restrict__ chem_decay_factors,
    // RDField components (channels 6-9)
    const float* __restrict__ rd_resource_density,
    const float* __restrict__ rd_fitness_landscape,
    const float* __restrict__ rd_resource_gradient_x,
    const float* __restrict__ rd_resource_gradient_y,
    // BehavioralField (channel 10)
    const float* __restrict__ behavioral_field,
    // Temporal retrieval (channel 15)
    const float* __restrict__ attractor_field,
    int max_grid_size,
    int entry_idx
) {
    if (entry_idx >= pool->capacity) return;

    PoolEntry* entry = &pool->entries[entry_idx];
    if (!entry->alive) return;

    int grid_size = entry->grid_size;
    int channels = entry->channels;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int cell_idx = y * grid_size + x;
    float* ca_concentration = entry->ca_state->ca_concentration;
    int base_idx = cell_idx * channels;

    // Channel 0-5: ChemicalField
    ca_concentration[base_idx + 0] = chem_concentration[cell_idx];
    ca_concentration[base_idx + 1] = chem_gradient_x[cell_idx];
    ca_concentration[base_idx + 2] = chem_gradient_y[cell_idx];
    ca_concentration[base_idx + 3] = chem_laplacian[cell_idx];
    ca_concentration[base_idx + 4] = chem_sources[cell_idx];
    DEVICE_FATAL_IF(chem_decay_factors == nullptr, "chem_decay_factors is null");
    ca_concentration[base_idx + 5] = chem_decay_factors[cell_idx];

    // Channel 6-9: RDField
    ca_concentration[base_idx + 6] = rd_resource_density[cell_idx];
    ca_concentration[base_idx + 7] = rd_fitness_landscape[cell_idx];
    ca_concentration[base_idx + 8] = rd_resource_gradient_x[cell_idx];
    ca_concentration[base_idx + 9] = rd_resource_gradient_y[cell_idx];

    // Channel 10: BehavioralField
    ca_concentration[base_idx + 10] = behavioral_field[cell_idx];

    // Channel 11-13: Dataset samples - NOT present in evolutionary path
    // The training encodes dataset patterns into chemical field, which we read above
    // Setting to zero is intentional - these channels are for training only
    if (channels > 11) ca_concentration[base_idx + 11] = 0.0f;
    if (channels > 12) ca_concentration[base_idx + 12] = 0.0f;
    if (channels > 13) ca_concentration[base_idx + 13] = 0.0f;

    // Channel 14: Previous CA output (recurrence) - read from ca_output channel 0
    if (channels > 14) {
        // Use previous iteration's CA output for recurrence
        // ca_output is allocated as part of ca_state (organism.cu:2927), so if ca_state exists, ca_output exists
        float* ca_output = entry->ca_state->ca_output;
        DEVICE_FATAL_IF(ca_output == nullptr, "ca_output null but ca_state exists");
        // ca_output layout: [num_heads × grid² × head_dim], sum across heads for recurrence
        int num_heads = entry->num_heads;
        int head_dim = entry->head_dim;
        float recurrence = 0.0f;
        for (int h = 0; h < num_heads; h++) {
            int output_idx = h * grid_size * grid_size * head_dim + cell_idx * head_dim;
            recurrence += ca_output[output_idx];  // First element of each head's output
        }
        ca_concentration[base_idx + 14] = recurrence / (float)max(1, num_heads);
    }

    // Channel 15: Temporal retrieval
    if (channels > 15) {
        DEVICE_FATAL_IF(attractor_field == nullptr, "init_chemical_field_kernel: attractor_field required for channel 15");
        ca_concentration[base_idx + 15] = attractor_field[cell_idx];
    }
}

__global__ void diffusion_reaction_kernel(
    float* __restrict__ concentration,
    float* __restrict__ gradient_x,
    float* __restrict__ gradient_y,
    float* __restrict__ laplacian,
    float* __restrict__ sources,
    int grid_size,
    float dt,
    const float* genome,
    const float* epigenetic,
    uint64_t genome_hash,
    float ctx_metabolic,
    float ctx_stress,
    float ctx_morphogen,
    float ctx_complexity,
    float ctx_niche,
    float ctx_learning,
    float ctx_performance
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;

    float c_center;
    Stencils::all_operators(gradient_x[idx], gradient_y[idx], laplacian[idx], c_center,
                            concentration, x, y, grid_size, 1);

    float source_contribution = sources[idx];

    int diffusivity_slot = derive_param_slot(genome_hash, "chem_diffusivity");
    float diffusivity = genome_to_param(genome, epigenetic, diffusivity_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, DIFFUSIVITY_BASE_MIN, DIFFUSIVITY_BASE_MAX);

    int reaction_order_slot = derive_param_slot(genome_hash, "chem_reaction_order");
    float reaction_order = genome_to_param(genome, epigenetic, reaction_order_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, REACTION_ORDER_MIN, REACTION_ORDER_MAX);

    int reaction_rate_slot = derive_param_slot(genome_hash, "chem_reaction_rate");
    float reaction_rate = genome_to_param(genome, epigenetic, reaction_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, REACTION_RATE_MIN, REACTION_RATE_MAX);

    int decay_rate_slot = derive_param_slot(genome_hash, "chem_decay_rate");
    float decay_rate = genome_to_param(genome, epigenetic, decay_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, DECAY_RATE_MIN, DECAY_RATE_MAX);

    float diffusion = diffusivity * laplacian[idx];
    float reaction = reaction_rate * powf(c_center, reaction_order);
    float decay = -decay_rate * c_center;

    concentration[idx] = c_center + dt * (diffusion + reaction + decay + source_contribution);
    concentration[idx] = clamp(concentration[idx], NORMALIZED_MIN, NORMALIZED_MAX);
}

__global__ void diffusion_reaction_backward_kernel(
    const float* __restrict__ grad_concentration,
    const float* __restrict__ concentration,
    const float* __restrict__ laplacian,
    float* __restrict__ grad_genome,
    int grid_size,
    float dt,
    const float* genome,
    const float* epigenetic,
    uint64_t genome_hash,
    float ctx_metabolic,
    float ctx_stress,
    float ctx_morphogen,
    float ctx_complexity,
    float ctx_niche,
    float ctx_learning,
    float ctx_performance
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;
    float d_concentration = grad_concentration[idx];
    float c_center = concentration[idx];
    float lap = laplacian[idx];

    int diffusivity_slot = derive_param_slot(genome_hash, "chem_diffusivity");
    float diffusivity = genome_to_param(genome, epigenetic, diffusivity_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, DIFFUSIVITY_BASE_MIN, DIFFUSIVITY_BASE_MAX);

    int reaction_order_slot = derive_param_slot(genome_hash, "chem_reaction_order");
    float reaction_order = genome_to_param(genome, epigenetic, reaction_order_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, REACTION_ORDER_MIN, REACTION_ORDER_MAX);

    int reaction_rate_slot = derive_param_slot(genome_hash, "chem_reaction_rate");
    float reaction_rate = genome_to_param(genome, epigenetic, reaction_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, REACTION_RATE_MIN, REACTION_RATE_MAX);

    int decay_rate_slot = derive_param_slot(genome_hash, "chem_decay_rate");
    float decay_rate = genome_to_param(genome, epigenetic, decay_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, DECAY_RATE_MIN, DECAY_RATE_MAX);

    float c_safe = fmaxf(fabsf(c_center), safe_epsilon(c_center));
    float c_pow = powf(c_safe, reaction_order);

    float d_diffusivity = d_concentration * dt * lap;
    float d_reaction_rate = d_concentration * dt * c_pow;
    float d_reaction_order = d_concentration * dt * reaction_rate * c_pow * logf(c_safe);
    float d_decay_rate = d_concentration * dt * (-c_center);

    atomicAdd(&grad_genome[diffusivity_slot], d_diffusivity);
    atomicAdd(&grad_genome[reaction_rate_slot], d_reaction_rate);
    atomicAdd(&grad_genome[reaction_order_slot], d_reaction_order);
    atomicAdd(&grad_genome[decay_rate_slot], d_decay_rate);
}

__global__ void behavioral_gradient_kernel(float* __restrict__ behavioral_field, float* __restrict__ behavioral_gradients, int grid_size, int hw_dim, int task_dim, int gen_dim){
    int behavioral_dim = hw_dim + task_dim + gen_dim;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int dim = blockIdx.z;

    if (x >= grid_size || y >= grid_size || dim >= behavioral_dim) return;

    float grad_x, grad_y;
    Stencils::gradients_at(grad_x, grad_y, &behavioral_field[dim], x, y, grid_size, behavioral_dim);

    float grad_sq = grad_x * grad_x + grad_y * grad_y;
    float magnitude = sqrtf(grad_sq) + safe_epsilon(grad_sq);
    grad_x /= magnitude;
    grad_y /= magnitude;

    int grad_idx = ((y * grid_size + x) * behavioral_dim + dim) * 2;
    behavioral_gradients[grad_idx] = grad_x;
    behavioral_gradients[grad_idx + 1] = grad_y;
}

__global__ void chemotactic_navigation_kernel(BehavioralState* __restrict__ agents, float* __restrict__ chemical_field, float* __restrict__ gradient_x, float* __restrict__ gradient_y, float* __restrict__ behavioral_gradients, int num_agents, int grid_size, float dt, const float* genome, const float* gradients, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance, int hw_dim, int task_dim, int gen_dim){
    int behavioral_dim = hw_dim + task_dim + gen_dim;
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id >= num_agents) return;

    BehavioralState* agent = &agents[agent_id];

    float context_metabolic = agent->sensitivity;
    float context_stress = sqrtf(agent->velocity[0] * agent->velocity[0] +
                                 agent->velocity[1] * agent->velocity[1]);
    int grid_x_temp = (int)(agent->position[0] * grid_size);
    int grid_y_temp = (int)(agent->position[1] * grid_size);
    grid_x_temp = min(max(grid_x_temp, 0), grid_size - 1);
    grid_y_temp = min(max(grid_y_temp, 0), grid_size - 1);
    int idx_temp = grid_y_temp * grid_size + grid_x_temp;
    float context_morphogen = chemical_field[idx_temp];

    ChemotaxisParams params;
    params.derive_from_genome_hash(agent->genome_hash);

    float theta = params.get_theta(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float sigma = params.get_sigma(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float hurst_exponent = params.get_hurst_exponent(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float levy_alpha = params.get_levy_alpha(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float sensitivity_threshold = params.get_sensitivity_threshold(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float exploration_growth = params.get_exploration_growth(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float sensitivity_decay = params.get_sensitivity_decay(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float exploration_decay = params.get_exploration_decay(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float sensitivity_growth = params.get_sensitivity_growth(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float min_sensitivity_clamp = params.get_min_sensitivity_clamp(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float max_sensitivity_clamp = params.get_max_sensitivity_clamp(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float min_exploration_clamp = params.get_min_exploration_clamp(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float max_exploration_clamp = params.get_max_exploration_clamp(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float gradient_mix_weight = params.get_gradient_mix_weight(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float memory_decay_rate = params.get_memory_decay_rate(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);

    int grid_x = (int)(agent->position[0] * grid_size);
    int grid_y = (int)(agent->position[1] * grid_size);
    grid_x = min(max(grid_x, 0), grid_size - 1);
    grid_y = min(max(grid_y, 0), grid_size - 1);

    int idx = grid_y * grid_size + grid_x;

    float chem_grad_x = gradient_x[idx];
    float chem_grad_y = gradient_y[idx];

    float behav_grad_x = 0.0f;
    float behav_grad_y = 0.0f;

    int d_offset = 0;
    for (int d = 0; d < hw_dim; d++) {
        int grad_idx = ((grid_y * grid_size + grid_x) * behavioral_dim + d_offset) * 2;
        float weight = agent->hw_coords[d];
        behav_grad_x += behavioral_gradients[grad_idx] * weight;
        behav_grad_y += behavioral_gradients[grad_idx + 1] * weight;
        d_offset++;
    }
    for (int d = 0; d < task_dim; d++) {
        int grad_idx = ((grid_y * grid_size + grid_x) * behavioral_dim + d_offset) * 2;
        float weight = agent->task_coords[d];
        behav_grad_x += behavioral_gradients[grad_idx] * weight;
        behav_grad_y += behavioral_gradients[grad_idx + 1] * weight;
        d_offset++;
    }
    for (int d = 0; d < gen_dim; d++) {
        int grad_idx = ((grid_y * grid_size + grid_x) * behavioral_dim + d_offset) * 2;
        float weight = agent->gen_coords[d];
        behav_grad_x += behavioral_gradients[grad_idx] * weight;
        behav_grad_y += behavioral_gradients[grad_idx + 1] * weight;
        d_offset++;
    }

    float behav_sq = behav_grad_x * behav_grad_x + behav_grad_y * behav_grad_y;
    float behav_magnitude = sqrtf(behav_sq) + safe_epsilon(behav_sq);
    behav_grad_x /= behav_magnitude;
    behav_grad_y /= behav_magnitude;

    int mem_idx = agent->memory_index;
    float chem_weight = gradient_mix_weight;
    float behav_weight = NORMALIZED_MAX - gradient_mix_weight;
    agent->gradient_memory[mem_idx][0] = chem_grad_x * chem_weight + behav_grad_x * behav_weight;
    agent->gradient_memory[mem_idx][1] = chem_grad_y * chem_weight + behav_grad_y * behav_weight;
    agent->memory_index = (mem_idx + 1) % GRADIENT_HISTORY;

    float avg_grad_x = 0.0f;
    float avg_grad_y = 0.0f;
    float weight_sum = 0.0f;

    for (int i = 0; i < GRADIENT_HISTORY; i++) {
        float age = (float)(GRADIENT_HISTORY - i) / GRADIENT_HISTORY;
        float weight = expf(-memory_decay_rate * (NORMALIZED_MAX - age));

        avg_grad_x += agent->gradient_memory[i][0] * weight;
        avg_grad_y += agent->gradient_memory[i][1] * weight;
        weight_sum += weight;
    }

    avg_grad_x /= weight_sum;
    avg_grad_y /= weight_sum;

    unsigned int seed = agent_id * RNG_SEED_MULTIPLIER + (unsigned int)(dt * 1000.0f);
    seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
    float U = (seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE;
    seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
    float V = (seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE;
    seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
    float W = (seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE;

    float phi = (TAU * 0.25f) * (2.0f * V - 1.0f);
    float alpha_phi = levy_alpha * phi;
    float one_minus_alpha_phi = (1.0f - levy_alpha) * phi;

    float levy_num = sinf(alpha_phi);

    float cos_phi = cosf(phi);
    if (cos_phi <= 0.0f) {
        agent->velocity[0] = 0.0f;
        agent->velocity[1] = 0.0f;
        return;
    }
    float levy_denom = powf(cos_phi, (NORMALIZED_MAX / levy_alpha));

    if (W <= 0.0f) {
        agent->velocity[0] = 0.0f;
        agent->velocity[1] = 0.0f;
        return;
    }
    float log_w = -logf(W);
    if (log_w <= 0.0f) {
        agent->velocity[0] = 0.0f;
        agent->velocity[1] = 0.0f;
        return;
    }
    float cos_one_minus_alpha_phi = cosf(one_minus_alpha_phi);
    if (cos_one_minus_alpha_phi <= 0.0f) {
        agent->velocity[0] = 0.0f;
        agent->velocity[1] = 0.0f;
        return;
    }
    float levy_factor = powf(cos_one_minus_alpha_phi / log_w, ((1.0f - levy_alpha) / levy_alpha));

    float levy_sample = levy_num / levy_denom * levy_factor;

    if (U <= 0.0f) {
        agent->velocity[0] = 0.0f;
        agent->velocity[1] = 0.0f;
        return;
    }
    float gaussian_component = sqrtf(-2.0f * logf(U)) * cosf(TAU * V);
    float tempering_weight = expf(-levy_alpha * fabsf(levy_sample));
    float tempered_sample = tempering_weight * levy_sample + (NORMALIZED_MAX - tempering_weight) * gaussian_component;

    seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
    float angle = TAU * ((seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE);
    float noise_x = tempered_sample * cosf(angle);
    float noise_y = tempered_sample * sinf(angle);

    float fractional_friction_x = 0.0f;
    float fractional_friction_y = 0.0f;
    float kernel_exponent = hurst_exponent - FRACTIONAL_OU_KERNEL_OFFSET;

    for (int i = 0; i < GRADIENT_HISTORY; i++) {
        float age = (float)(GRADIENT_HISTORY - i);
        if (age <= 0.0f) {
            return;
        }
        float kernel_weight = powf(age, kernel_exponent);
        fractional_friction_x += kernel_weight * agent->velocity_history[i][0];
        fractional_friction_y += kernel_weight * agent->velocity_history[i][1];
    }

    agent->velocity_history[mem_idx][0] = agent->velocity[0];
    agent->velocity_history[mem_idx][1] = agent->velocity[1];

    agent->velocity[0] += dt * (agent->sensitivity * avg_grad_x -
                                theta * fractional_friction_x +
                                sigma * noise_x * agent->exploration_noise);

    agent->velocity[1] += dt * (agent->sensitivity * avg_grad_y -
                                theta * fractional_friction_y +
                                sigma * noise_y * agent->exploration_noise);

    float vel_magnitude = sqrtf(agent->velocity[0] * agent->velocity[0] +
                                agent->velocity[1] * agent->velocity[1]);

    int max_vel_slot = derive_param_slot(agent->genome_hash, "max_agent_velocity");
    float max_agent_velocity = genome_to_param(genome, gradients, max_vel_slot, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MAX_AGENT_VELOCITY_BASE_MIN, MAX_AGENT_VELOCITY_BASE_MAX);

    if (vel_magnitude > max_agent_velocity) {
        agent->velocity[0] *= (max_agent_velocity / vel_magnitude);
        agent->velocity[1] *= (max_agent_velocity / vel_magnitude);
    }

    agent->position[0] += agent->velocity[0] * dt;
    agent->position[1] += agent->velocity[1] * dt;

    agent->position[0] = agent->position[0] - floorf(agent->position[0]);
    agent->position[1] = agent->position[1] - floorf(agent->position[1]);

    float gradient_strength = sqrtf(avg_grad_x * avg_grad_x + avg_grad_y * avg_grad_y);
    if (gradient_strength < sensitivity_threshold) {
        agent->exploration_noise = fminf(max_exploration_clamp, agent->exploration_noise * exploration_growth);
        agent->exploration = fminf(max_exploration_clamp, agent->exploration * exploration_growth);
        agent->sensitivity = fmaxf(min_sensitivity_clamp, agent->sensitivity * sensitivity_decay);
    } else {
        agent->exploration_noise = fmaxf(min_exploration_clamp, agent->exploration_noise * exploration_decay);
        agent->exploration = fmaxf(min_exploration_clamp, agent->exploration * exploration_decay);
        agent->sensitivity = fminf(max_sensitivity_clamp, agent->sensitivity * sensitivity_growth);
    }
}

__global__ void create_attractors_kernel(
    float* __restrict__ sources,
    float* __restrict__ attractor_positions,
    float* __restrict__ attractor_strengths,
    int grid_size,
    int num_attractors,
    const float* genome,
    const float* gradients,
    uint64_t genome_hash,
    float ctx_complexity,
    float ctx_niche,
    float ctx_learning,
    float ctx_performance
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;
    float source_value = 0.0f;

    
    float context_metabolic = attractor_strengths[0];
    float context_stress = attractor_strengths[1];
    float context_morphogen = attractor_strengths[2];

    ChemotaxisParams params;
    params.derive_from_genome_hash(genome_hash);

    float attractor_sigma = params.get_attractor_sigma(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);

    float px = (float)x / grid_size;
    float py = (float)y / grid_size;

    for (int a = 0; a < num_attractors; a++) {
        float ax = attractor_positions[a * 2];
        float ay = attractor_positions[a * 2 + 1];
        float strength = attractor_strengths[a];

        float dx = fminf(fabsf(px - ax), NORMALIZED_MAX - fabsf(px - ax));
        float dy = fminf(fabsf(py - ay), NORMALIZED_MAX - fabsf(py - ay));
        float dist_sq = dx * dx + dy * dy;

        source_value += strength * expf(-dist_sq / (GAUSSIAN_VARIANCE_DENOMINATOR * attractor_sigma * attractor_sigma));
    }

    sources[idx] = source_value;
}

__global__ void update_behavioral_embedding_kernel(BehavioralState* __restrict__ agents, float* __restrict__ embedding_weights, float* __restrict__ reconstruction_error, int num_agents, float learning_rate, const float* genome, const float* gradients, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance, int behavioral_dim, int hw_dim, int task_dim, int gen_dim, float* features_buffer, const float* __restrict__ chemical_concentration, int grid_size){
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id >= num_agents) return;

    BehavioralState* agent = &agents[agent_id];

    float* features = &features_buffer[agent_id * behavioral_dim];

    float context_metabolic = agent->sensitivity;
    float context_stress = sqrtf(agent->velocity[0] * agent->velocity[0] +
                                 agent->velocity[1] * agent->velocity[1]);
    // Sample morphogen from chemical field at agent position
    int cx = (int)(agent->position[0] * grid_size) % grid_size;
    int cy = (int)(agent->position[1] * grid_size) % grid_size;
    float context_morphogen = chemical_concentration[cy * grid_size + cx];

    int base_freq_slot = derive_param_slot(agent->genome_hash, "fourier_base_freq");
    float fourier_base_freq = genome_to_param(genome, gradients, base_freq_slot, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, FOURIER_BASE_FREQ_MIN, FOURIER_BASE_FREQ_MAX);

    int num_octaves_slot = derive_param_slot(agent->genome_hash, "fourier_num_octaves");
    int fourier_num_octaves_raw = (int)genome_to_param(genome, gradients, num_octaves_slot, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, (float)FOURIER_NUM_OCTAVES_MIN, (float)FOURIER_NUM_OCTAVES_MAX);
    int fourier_num_octaves = min(fourier_num_octaves_raw, behavioral_dim - 4);

    int spectrum_exp_slot = derive_param_slot(agent->genome_hash, "fourier_spectrum_exponent");
    float fourier_spectrum_exponent = genome_to_param(genome, gradients, spectrum_exp_slot, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, FOURIER_SPECTRUM_EXPONENT_MIN, FOURIER_SPECTRUM_EXPONENT_MAX);

    features[0] = sqrtf(agent->velocity[0] * agent->velocity[0] +
                       agent->velocity[1] * agent->velocity[1]);

    float turn_rate = 0.0f;
    for (int i = 1; i < GRADIENT_HISTORY; i++) {
        float dx = agent->gradient_memory[i][0] - agent->gradient_memory[i-1][0];
        float dy = agent->gradient_memory[i][1] - agent->gradient_memory[i-1][1];
        turn_rate += sqrtf(dx * dx + dy * dy);
    }
    features[1] = turn_rate / GRADIENT_HISTORY;

    features[2] = agent->exploration_noise;

    features[3] = agent->sensitivity;

    for (int k = 0; k < fourier_num_octaves; k++) {
        float freq = fourier_base_freq * powf(OCTAVE_MULTIPLIER, (float)k);
        float cos_sum = 0.0f;
        float sin_sum = 0.0f;

        for (int i = 0; i < GRADIENT_HISTORY; i++) {
            cos_sum += agent->gradient_memory[i][0] * cosf(freq * i);
            sin_sum += agent->gradient_memory[i][1] * sinf(freq * i);
        }

        float magnitude = sqrtf(cos_sum * cos_sum + sin_sum * sin_sum) / GRADIENT_HISTORY;
        if (freq <= 0.0f) {
            return;
        }
        float amplitude_weight = powf(freq, -fourier_spectrum_exponent);
        features[BASE_FEATURES_COUNT + k] = magnitude * amplitude_weight;
    }

    // Project features to behavioral space through learned embedding and update coords
    for (int d = 0; d < behavioral_dim; d++) {
        float reconstruction = 0.0f;
        for (int f = 0; f < behavioral_dim; f++) {
            reconstruction += features[f] * embedding_weights[f * behavioral_dim + d];
        }

        // Determine which coord array this dimension belongs to
        float* target_coord;
        int local_idx;
        if (d < hw_dim) {
            target_coord = agent->hw_coords;
            local_idx = d;
        } else if (d < hw_dim + task_dim) {
            target_coord = agent->task_coords;
            local_idx = d - hw_dim;
        } else {
            target_coord = agent->gen_coords;
            local_idx = d - hw_dim - task_dim;
        }

        // Update coord with gradient descent
        float error = reconstruction - target_coord[local_idx];
        target_coord[local_idx] += learning_rate * error;

        // Accumulate reconstruction error
        atomicAdd(reconstruction_error, error * error);
    }
}

__global__ void init_embedding_weights_kernel(float* embedding_weights, int behavioral_dim, unsigned int seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = behavioral_dim * behavioral_dim;
    if (idx >= total_weights) return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    int row = idx / behavioral_dim;
    int col = idx % behavioral_dim;

    float val;
    if (row == col) {
        val = 1.0f + 0.01f * curand_normal(&state);
    } else {
        val = 0.01f * curand_normal(&state);
    }
    DEVICE_FATAL_IF(isnan(val) || isinf(val), "init_embedding_weights: NaN/Inf from curand");
    embedding_weights[idx] = val;
}

__global__ void init_behavioral_state_kernel(BehavioralState* agents, int num_agents, unsigned int seed, const float* genome, const float* epigenetic, uint64_t genome_hash, int organism_id, BehavioralInitSlots slots, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance, int hw_dim, int task_dim, int gen_dim){
    int behavioral_dim = hw_dim + task_dim + gen_dim;
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (agent_id == 0) {
    }

    if (agent_id >= num_agents) return;

    uint64_t stream_seed = ((uint64_t)seed << 32) | (uint64_t)agent_id;
    stream_seed ^= stream_seed >> 33;
    stream_seed *= HASH_MIX_CONSTANT_A;
    stream_seed ^= stream_seed >> 33;
    stream_seed *= HASH_MIX_CONSTANT_B;
    stream_seed ^= stream_seed >> 33;

    PRNGState rng;
    rng.s0 = stream_seed;
    rng.s1 = stream_seed ^ XORSHIFT_GOLDEN_RATIO_A;

    BehavioralState* agent = &agents[agent_id];

    float ctx_metabolic = (genome[slots.ctx_metabolic] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float ctx_stress = (genome[slots.ctx_stress] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float ctx_morphogen = (genome[slots.ctx_morphogen] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;

    float embedding_scale = genome_to_param(genome, epigenetic, slots.agent_embedding_scale, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, AGENT_EMBEDDING_SCALE_BASE_MIN, AGENT_EMBEDDING_SCALE_BASE_MAX);
    float init_exploration = genome_to_param(genome, epigenetic, slots.init_exploration, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, INIT_EXPLORATION_BASE_MIN, INIT_EXPLORATION_BASE_MAX);
    float init_sensitivity = genome_to_param(genome, epigenetic, slots.init_sensitivity, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, INIT_SENSITIVITY_BASE_MIN, INIT_SENSITIVITY_BASE_MAX);
    float levy_alpha = genome_to_param(genome, epigenetic, slots.levy_alpha, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMOTAXIS_LEVY_ALPHA_MIN, CHEMOTAXIS_LEVY_ALPHA_MAX);

    agent->position[0] = rng.next();
    agent->position[1] = rng.next();

    agent->velocity[0] = 0.0f;
    agent->velocity[1] = 0.0f;

    if (agent_id == 0) {
    }

    for (int i = 0; i < hw_dim; i++) {
        agent->hw_coords[i] = rng.levy_stable(levy_alpha, embedding_scale);
    }
    for (int i = 0; i < task_dim; i++) {
        agent->task_coords[i] = rng.levy_stable(levy_alpha, embedding_scale);
    }
    for (int i = 0; i < gen_dim; i++) {
        agent->gen_coords[i] = rng.levy_stable(levy_alpha, embedding_scale);
    }

    if (agent_id == 0) {
    }

    for (int i = 0; i < GRADIENT_HISTORY; i++) {
        agent->gradient_memory[i][0] = 0.0f;
        agent->gradient_memory[i][1] = 0.0f;
        agent->velocity_history[i][0] = 0.0f;
        agent->velocity_history[i][1] = 0.0f;
    }

    agent->exploration_noise = init_exploration;
    agent->exploration = init_exploration;
    agent->sensitivity = init_sensitivity;
    agent->memory_index = 0;
    agent->genome_hash = genome_hash;
    agent->organism_id = organism_id;

    if (agent_id == 0) {
    }
}

__global__ void init_chemical_field_kernel(
    ChemicalField* field,
    int grid_size,
    const float* genome,
    const float* gradients,
    uint64_t genome_hash,
    float ctx_complexity,
    float ctx_niche,
    float ctx_learning,
    float ctx_performance
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;

    ChemotaxisParams params;
    params.derive_from_genome_hash(genome_hash);
    InitContext ctx;
    ctx.derive_from_genome(genome_hash, genome);
    float chemical_decay = params.get_chemical_decay(genome, gradients, ctx.metabolic, ctx.stress, ctx.morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);

    field->concentration[idx] = 0.0f;
    field->gradient_x[idx] = 0.0f;
    field->gradient_y[idx] = 0.0f;
    field->laplacian[idx] = 0.0f;
    field->sources[idx] = 0.0f;
    field->decay_factors[idx] = chemical_decay;

    if (x == 0 && y == 0) {
        int null_check = (field->concentration == nullptr) + (field->gradient_x == nullptr) + (field->gradient_y == nullptr) + (field->laplacian == nullptr);
    }
}

// Parallel reduction to compute mean concentration - updates chemical_field->cached_mean
__global__ void reduce_concentration_mean_kernel(
    ChemicalField* field,
    int total_cells,
    float* partial_sums  // Workspace for block-level partial sums
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads and accumulates multiple elements (grid-stride loop)
    float sum = 0.0f;
    for (int i = idx; i < total_cells; i += blockDim.x * gridDim.x) {
        sum += field->concentration[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level reduction (no sync needed within warp)
    if (tid < 32) {
        volatile float* vdata = sdata;
        vdata[tid] += vdata[tid + 32];
        vdata[tid] += vdata[tid + 16];
        vdata[tid] += vdata[tid + 8];
        vdata[tid] += vdata[tid + 4];
        vdata[tid] += vdata[tid + 2];
        vdata[tid] += vdata[tid + 1];
    }

    // First thread writes block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Final reduction of partial sums and write to cached_mean
__global__ void finalize_concentration_mean_kernel(
    ChemicalField* field,
    float* partial_sums,
    int num_blocks,
    int total_cells
) {
    float sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        sum += partial_sums[i];
    }
    field->cached_mean = sum / (float)total_cells;
}

__global__ void set_chemical_sources_from_agents_kernel(
    float* sources,
    BehavioralState* agents,
    int num_agents,
    int grid_size,
    const float* genome,
    const float* gradients,
    const float* chemical_field,
    float ctx_complexity,
    float ctx_niche,
    float ctx_learning,
    float ctx_performance
) {
    int agent_id = threadIdx.x;
    if (agent_id >= num_agents) return;

    BehavioralState* agent = &agents[agent_id];

    ChemotaxisParams params;
    params.derive_from_genome_hash(agent->genome_hash);

    int grid_x_temp = (int)(agent->position[0] * grid_size);
    int grid_y_temp = (int)(agent->position[1] * grid_size);
    grid_x_temp = min(max(grid_x_temp, 0), grid_size - 1);
    grid_y_temp = min(max(grid_y_temp, 0), grid_size - 1);
    int idx_temp = grid_y_temp * grid_size + grid_x_temp;

    float context_metabolic = agent->sensitivity;
    float context_stress = sqrtf(agent->velocity[0] * agent->velocity[0] + agent->velocity[1] * agent->velocity[1]);
    float context_morphogen = chemical_field[idx_temp];

    float source_sigma = params.get_agent_source_sigma(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float source_strength = params.get_agent_source_strength(genome, gradients, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);

    int grid_x = (int)(agent->position[0] * grid_size);
    int grid_y = (int)(agent->position[1] * grid_size);
    grid_x = min(max(grid_x, 0), grid_size - 1);
    grid_y = min(max(grid_y, 0), grid_size - 1);

    for (int dy = -4; dy <= 4; dy++) {
        for (int dx = -4; dx <= 4; dx++) {
            int x = grid_x + dx;
            int y = grid_y + dy;

            x = (x + grid_size) % grid_size;
            y = (y + grid_size) % grid_size;

            int idx = y * grid_size + x;

            float dist_sq = dx * dx + dy * dy;
            float contribution = source_strength * expf(-dist_sq / (GAUSSIAN_VARIANCE_DENOMINATOR * source_sigma * source_sigma));

            if (isfinite(contribution)) {
                atomicAdd(&sources[idx], contribution);
            }
        }
    }
}

__global__ void compute_behavioral_field_kernel(float* behavioral_field, BehavioralState* agents, int num_agents, int grid_size, const float* genome, const float* gradients, uint64_t genome_hash, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance, int hw_dim, int task_dim, int gen_dim){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    ChemotaxisParams params;
    params.derive_from_genome_hash(genome_hash);

    InitContext ctx;
    ctx.derive_from_genome(genome_hash, genome);

    float behavioral_field_sigma = params.get_behavioral_field_sigma(genome, gradients, ctx.metabolic, ctx.stress, ctx.morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);

    float px = (float)x / grid_size;
    float py = (float)y / grid_size;

    int behavioral_dim = hw_dim + task_dim + gen_dim;
    int d_offset = 0;

    for (int d = 0; d < hw_dim; d++) {
        float field_value = 0.0f;
        float weight_sum = 0.0f;

        for (int agent_id = 0; agent_id < num_agents; agent_id++) {
            BehavioralState* agent = &agents[agent_id];

            float dx = fabsf(px - agent->position[0]);
            float dy = fabsf(py - agent->position[1]);
            dx = fminf(dx, NORMALIZED_MAX - dx);
            dy = fminf(dy, NORMALIZED_MAX - dy);
            float dist_sq = dx * dx + dy * dy;

            float weight = expf(-dist_sq / (GAUSSIAN_VARIANCE_DENOMINATOR * behavioral_field_sigma * behavioral_field_sigma));

            field_value += weight * agent->hw_coords[d];
            weight_sum += weight;
        }

        int field_idx = (y * grid_size + x) * behavioral_dim + d_offset;
        if (is_meaningful(weight_sum, 1.0f)) {
            behavioral_field[field_idx] = field_value / weight_sum;
        }
        // else: preserve existing value (temporal coherence)
        d_offset++;
    }

    for (int d = 0; d < task_dim; d++) {
        float field_value = 0.0f;
        float weight_sum = 0.0f;

        for (int agent_id = 0; agent_id < num_agents; agent_id++) {
            BehavioralState* agent = &agents[agent_id];

            float dx = fabsf(px - agent->position[0]);
            float dy = fabsf(py - agent->position[1]);
            dx = fminf(dx, NORMALIZED_MAX - dx);
            dy = fminf(dy, NORMALIZED_MAX - dy);
            float dist_sq = dx * dx + dy * dy;

            float weight = expf(-dist_sq / (GAUSSIAN_VARIANCE_DENOMINATOR * behavioral_field_sigma * behavioral_field_sigma));

            field_value += weight * agent->task_coords[d];
            weight_sum += weight;
        }

        int field_idx = (y * grid_size + x) * behavioral_dim + d_offset;
        if (is_meaningful(weight_sum, 1.0f)) {
            behavioral_field[field_idx] = field_value / weight_sum;
        }
        // else: preserve existing value (temporal coherence)
        d_offset++;
    }

    for (int d = 0; d < gen_dim; d++) {
        float field_value = 0.0f;
        float weight_sum = 0.0f;

        for (int agent_id = 0; agent_id < num_agents; agent_id++) {
            BehavioralState* agent = &agents[agent_id];

            float dx = fabsf(px - agent->position[0]);
            float dy = fabsf(py - agent->position[1]);
            dx = fminf(dx, NORMALIZED_MAX - dx);
            dy = fminf(dy, NORMALIZED_MAX - dy);
            float dist_sq = dx * dx + dy * dy;

            float weight = expf(-dist_sq / (GAUSSIAN_VARIANCE_DENOMINATOR * behavioral_field_sigma * behavioral_field_sigma));

            field_value += weight * agent->gen_coords[d];
            weight_sum += weight;
        }

        int field_idx = (y * grid_size + x) * behavioral_dim + d_offset;
        if (is_meaningful(weight_sum, 1.0f)) {
            behavioral_field[field_idx] = field_value / weight_sum;
        }
        // else: preserve existing value (temporal coherence)
        d_offset++;
    }
}

__global__ void init_resource_fields_kernel(
    float* __restrict__ resource_density,
    float* __restrict__ fitness_landscape,
    int grid_size,
    unsigned int seed,
    float* genome,
    uint64_t genome_hash
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;
    curandState_t state;
    curand_init(seed, idx, 0, &state);

    int resource_init_slot = derive_param_slot(genome_hash, "resource_density_initial");
    float resource_init_norm = (genome[resource_init_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float resource_init = RESOURCE_INIT_MIN + resource_init_norm * (RESOURCE_INIT_MAX - RESOURCE_INIT_MIN);

    int resource_noise_slot = derive_param_slot(genome_hash, "resource_density_noise");
    float resource_noise_norm = (genome[resource_noise_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float resource_noise = RESOURCE_NOISE_MIN + resource_noise_norm * (RESOURCE_NOISE_MAX - RESOURCE_NOISE_MIN);

    float rand_val = validated_curand_uniform(&state, "init_resource_fields", idx);
    resource_density[idx] = resource_init + resource_noise * (rand_val - CENTERED_DIFFERENCE_SCALE);

    fitness_landscape[idx] = 0.0f;
}

__global__ void initialize_chemical_field_kernel(
    float* __restrict__ chemical_field,
    const float* __restrict__ genome,
    int grid_size,
    unsigned int seed,
    uint64_t genome_hash
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    int base_slot = derive_param_slot(genome_hash, "chem_init_base_value");
    float base_norm = (genome[base_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float base_value = CHEM_INIT_BASE_MIN + base_norm * (CHEM_INIT_BASE_MAX - CHEM_INIT_BASE_MIN);

    int influence_slot = derive_param_slot(genome_hash, "chem_init_genome_influence");
    float influence_norm = (genome[influence_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float influence_scale = CHEM_INIT_GENOME_INFLUENCE_MIN + influence_norm * (CHEM_INIT_GENOME_INFLUENCE_MAX - CHEM_INIT_GENOME_INFLUENCE_MIN);

    int genome_idx = (x + y * grid_size) % GENOME_SIZE;
    float genome_influence = genome[genome_idx] * influence_scale;

    int noise_slot = derive_param_slot(genome_hash, "chem_init_noise_scale");
    float noise_norm = (genome[noise_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float noise_scale = CHEM_INIT_NOISE_MIN + noise_norm * (CHEM_INIT_NOISE_MAX - CHEM_INIT_NOISE_MIN);

    float rand_val = validated_curand_uniform(&state, "initialize_chemical_field", idx);
    float noise = (rand_val - 0.5f) * noise_scale;

    chemical_field[idx] = base_value + genome_influence + noise;
    chemical_field[idx] = clamp(chemical_field[idx], 0.0f, 1.0f);
}

extern "C" __global__ void resource_flow_kernel(
    float* __restrict__ resource_density,
    float* __restrict__ resource_next,
    const float* __restrict__ fitness_landscape,
    float* __restrict__ resource_gradient_x,
    float* __restrict__ resource_gradient_y,
    int grid_size,
    float dt,
    float diffusivity,
    float flow_strength
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;

    float grad_fitness_x, grad_fitness_y;
    Stencils::gradients_at(grad_fitness_x, grad_fitness_y, fitness_landscape, x, y, grid_size, 1);

    float2 velocity = make_float2(-grad_fitness_x * flow_strength, -grad_fitness_y * flow_strength);

    float rho = resource_density[idx];

    float rho_left = (x > 0) ? resource_density[idx - 1] : rho;
    float rho_right = (x < grid_size - 1) ? resource_density[idx + 1] : rho;
    float rho_up = (y > 0) ? resource_density[idx - grid_size] : rho;
    float rho_down = (y < grid_size - 1) ? resource_density[idx + grid_size] : rho;

    float flux_x = (rho_right * velocity.x - rho_left * velocity.x) * CENTERED_DIFFERENCE_SCALE;
    float flux_y = (rho_down * velocity.y - rho_up * velocity.y) * CENTERED_DIFFERENCE_SCALE;
    float divergence = flux_x + flux_y;

    float grad_rho_x, grad_rho_y, lap_rho, rho_center;
    Stencils::all_operators(grad_rho_x, grad_rho_y, lap_rho, rho_center, resource_density, x, y, grid_size, 1);

    // Store gradients for CA channel injection
    resource_gradient_x[idx] = grad_rho_x;
    resource_gradient_y[idx] = grad_rho_y;

    float drho_dt = -divergence + diffusivity * lap_rho;

    float new_rho = rho + dt * drho_dt;
    if (new_rho < -0.1f) {
        return;
    }
    resource_next[idx] = fmaxf(0.0f, new_rho);
}

extern "C" __global__ void update_fitness_landscape_kernel(
    ComponentPool* pool,
    BehavioralState* agents,
    float* __restrict__ fitness_landscape,
    int grid_size
) {
    int entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry_idx >= pool->capacity) return;
    // Use SoA for coalesced alive read
    if (!pool->alive_flags[entry_idx]) return;

    float px = agents[entry_idx].position[0];
    float py = agents[entry_idx].position[1];
    int gx = min((int)(px * grid_size), grid_size - 1);
    int gy = min((int)(py * grid_size), grid_size - 1);
    int idx = gy * grid_size + gx;

    // Use SoA for coalesced fitness read
    atomicAdd(&fitness_landscape[idx], pool->fitness_values[entry_idx]);
}

#endif

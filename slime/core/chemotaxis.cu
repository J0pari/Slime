
#ifndef CHEMOTAXIS_CU
#define CHEMOTAXIS_CU
#include "../config/config.cu"
#include "organism.cu"
#include "../utils/genome_params.cuh"
#include "pseudopod.cu"
#include "../memory/pool.cu"
#include "../compute/warp_ca.cu"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include "../memory/tubes.cu"

namespace cg = cooperative_groups;

__device__ void store_chemical_snapshot_device(Organism* organism) {
    ChemicalField* field = organism->chemical_field;
    int field_size = organism->field_size;
    float global_time = organism->global_time;
    TemporalTube* history = field->history;

    int next_head = (history->head + 1) % history->capacity;
    float* dest = history->entries[next_head].data;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < field_size; i += blockDim.x * gridDim.x) {
        dest[i] = field->concentration[i];
    }
    cg::this_grid().sync();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        history->entries[next_head].timestamp = global_time;
        history->entries[next_head].decay_factor = FIELD_MEMORY_DECAY;
        history->entries[next_head].importance = FIELD_MEMORY_IMPORTANCE;

        history->head = next_head;
        if (history->count < history->capacity) {
            history->count++;
        }
        history->global_time = global_time;
    }
}

__device__ void update_field_from_ca_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    float* chemical_concentration = organism->chemical_field->concentration;
    int entry_idx = organism->current_entry_idx;

    DEVICE_FATAL_IF(entry_idx >= pool->capacity, "update_field_from_ca_device: entry_idx out of bounds");

    PoolEntry* entry = &pool->entries[entry_idx];
    DEVICE_FATAL_IF(!pool->alive_flags[entry_idx], "update_field_from_ca_device: dead entry passed");

    int grid_size = entry->grid_size;
    int channels = entry->channels;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size) {
        int cell_idx = y * grid_size + x;
        float* ca_concentration = entry->ca_state->ca_concentration;

        float val = ca_concentration[cell_idx * channels + 0];
        if (isfinite(val)) {
            atomicAdd(&chemical_concentration[cell_idx], val);
        }
    }
}

__device__ void initialize_ca_from_field_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    ChemicalField* chem = organism->chemical_field;
    const float* chem_concentration = chem->concentration;
    const float* chem_gradient_x = chem->gradient_x;
    const float* chem_gradient_y = chem->gradient_y;
    const float* chem_laplacian = chem->laplacian;
    const float* chem_sources = chem->sources;
    const float* chem_decay_factors = chem->decay_factors;
    const float* rd_resource_density = organism->rd_resource_density;
    const float* rd_fitness_landscape = organism->rd_fitness_landscape;
    const float* rd_resource_gradient_x = organism->rd_resource_gradient_x;
    const float* rd_resource_gradient_y = organism->rd_resource_gradient_y;
    const float* behavioral_field = organism->behavioral_field;
    const float* attractor_field = organism->attractor_field;
    int entry_idx = organism->current_entry_idx;

    DEVICE_FATAL_IF(entry_idx >= pool->capacity, "initialize_ca_from_field_device: entry_idx out of bounds");

    PoolEntry* entry = &pool->entries[entry_idx];
    DEVICE_FATAL_IF(!pool->alive_flags[entry_idx], "initialize_ca_from_field_device: dead entry passed");

    int grid_size = entry->grid_size;
    int channels = entry->channels;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size) {
        int cell_idx = y * grid_size + x;
        float* ca_conc = entry->ca_state->ca_concentration;
        int base_idx = cell_idx * channels;

        ca_conc[base_idx + 0] = chem_concentration[cell_idx];
        ca_conc[base_idx + 1] = chem_gradient_x[cell_idx];
        ca_conc[base_idx + 2] = chem_gradient_y[cell_idx];
        ca_conc[base_idx + 3] = chem_laplacian[cell_idx];
        ca_conc[base_idx + 4] = chem_sources[cell_idx];
        DEVICE_FATAL_IF(chem_decay_factors == nullptr, "chem_decay_factors is null");
        ca_conc[base_idx + 5] = chem_decay_factors[cell_idx];

        ca_conc[base_idx + 6] = rd_resource_density[cell_idx];
        ca_conc[base_idx + 7] = rd_fitness_landscape[cell_idx];
        ca_conc[base_idx + 8] = rd_resource_gradient_x[cell_idx];
        ca_conc[base_idx + 9] = rd_resource_gradient_y[cell_idx];

        ca_conc[base_idx + 10] = behavioral_field[cell_idx];

        if (channels > 11) ca_conc[base_idx + 11] = 0.0f;
        if (channels > 12) ca_conc[base_idx + 12] = 0.0f;
        if (channels > 13) ca_conc[base_idx + 13] = 0.0f;

        if (channels > 14) {
            float* ca_output = entry->ca_state->ca_output;
            DEVICE_FATAL_IF(ca_output == nullptr, "ca_output null but ca_state exists");
            int num_heads = entry->num_heads;
            int head_dim = entry->head_dim;
            float recurrence = 0.0f;
            for (int h = 0; h < num_heads; h++) {
                int output_idx = h * grid_size * grid_size * head_dim + cell_idx * head_dim;
                recurrence += ca_output[output_idx];
            }
            ca_conc[base_idx + 14] = recurrence / (float)max(1, num_heads);
        }

        if (channels > 15) {
            DEVICE_FATAL_IF(attractor_field == nullptr, "initialize_ca_from_field_device: attractor_field required for channel 15");
            ca_conc[base_idx + 15] = attractor_field[cell_idx];
        }
    }
}

__device__ void diffusion_reaction_device(Organism* organism) {
    ChemicalField* chem = organism->chemical_field;
    float* concentration = chem->concentration;
    float* gradient_x = chem->gradient_x;
    float* gradient_y = chem->gradient_y;
    float* laplacian = chem->laplacian;
    float* sources = chem->sources;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;
    float dt = organism->dt;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size) {
        int idx = y * grid_size + x;

        float c_center;
        Stencils::all_operators(gradient_x[idx], gradient_y[idx], laplacian[idx], c_center,
                                concentration, x, y, grid_size, 1);

        float source_contribution = sources[idx];

        float diffusion = FIELD_DIFFUSIVITY * laplacian[idx];
        float reaction = FIELD_REACTION_RATE * powf(c_center, FIELD_REACTION_ORDER);
        float decay = -FIELD_DECAY_RATE * c_center;

        concentration[idx] = c_center + dt * (diffusion + reaction + decay + source_contribution);
        concentration[idx] = clamp(concentration[idx], NORMALIZED_MIN, NORMALIZED_MAX);
    }
}

__device__ void diffusion_reaction_backward_device(Organism* organism) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int entry_idx = organism->lifecycle_entry_idx;
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    Architecture arch = Architecture::maxBounds();

    const float* grad_concentration = organism->buffers->grad_concentration_buffer;
    const float* concentration = organism->chemical_field->concentration;
    const float* laplacian = organism->chemical_field->laplacian;
    float* grad_genome = entry->gradients;
    int grid_size = arch.grid_size;
    float dt = CHEMICAL_DIFFUSION_DT_MAX;
    const float* genome = &organism->lifecycle_workspace_genomes[entry_idx * GENOME_SIZE * 2];
    const float* epigenetic = entry->gradients;

    float ctx_metabolic = entry->fitness.value;
    float ctx_stress = entry->hunger.value;
    float ctx_morphogen = organism->chemical_field->cached_mean;
    float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
    float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
    float ctx_learning = organism->training_mode->learning_rate;
    float ctx_performance = entry->task_accuracy.value;

    if (x < grid_size && y < grid_size) {
        int idx = y * grid_size + x;
        float d_concentration = grad_concentration[idx];
        float c_center = concentration[idx];
        float lap = laplacian[idx];

        int diffusivity_slot = GenomeParamTable::chem_diffusivity;
        float diffusivity = genome_to_param(genome, epigenetic, diffusivity_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, DIFFUSIVITY_BASE_MIN, DIFFUSIVITY_BASE_MAX);

        int reaction_order_slot = GenomeParamTable::chem_reaction_order;
        float reaction_order = genome_to_param(genome, epigenetic, reaction_order_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, REACTION_ORDER_MIN, REACTION_ORDER_MAX);

        int reaction_rate_slot = GenomeParamTable::chem_reaction_rate;
        float reaction_rate = genome_to_param(genome, epigenetic, reaction_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, REACTION_RATE_MIN, REACTION_RATE_MAX);

        int decay_rate_slot = GenomeParamTable::chem_decay_rate;
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
}

__device__ void behavioral_gradient_device(Organism* organism) {
    float* behavioral_field = organism->behavioral_field;
    float* behavioral_gradients = organism->behavioral_gradients;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;
    int hw_dim = organism->hw_dim;
    int task_dim = organism->task_dim;
    int gen_dim = organism->gen_dim;
    int behavioral_dim = hw_dim + task_dim + gen_dim;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int dim = blockIdx.z;

    if (x < grid_size && y < grid_size && dim < behavioral_dim) {
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
}

__device__ void chemotactic_navigation_device(Organism* organism) {
    BehavioralState* agents = organism->behavioral_agents;
    float* chemical_field = organism->chemical_field->concentration;
    float* gradient_x = organism->chemical_field->gradient_x;
    float* gradient_y = organism->chemical_field->gradient_y;
    float* behavioral_gradients = organism->behavioral_gradients;
    int num_agents = organism->pool->capacity;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;
    float dt = organism->dt;
    float ctx_complexity = organism->ctx_complexity;
    float ctx_niche = organism->ctx_niche;
    float ctx_learning = organism->ctx_learning;
    float ctx_performance = organism->ctx_performance;
    int hw_dim = organism->hw_dim;
    int task_dim = organism->task_dim;
    int gen_dim = organism->gen_dim;
    int behavioral_dim = hw_dim + task_dim + gen_dim;
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id < num_agents) {
        const float* genome = &organism->workspace_genomes[agent_id * GENOME_SIZE * 2];
    const float* gradients = organism->pool->entries[agent_id].gradients;

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
    DEVICE_FATAL_IF(cos_phi <= 0.0f, "cos_phi <= 0: phi=%f agent=%d", phi, agent_id);
    float levy_denom = powf(cos_phi, (NORMALIZED_MAX / levy_alpha));

    DEVICE_FATAL_IF(W <= 0.0f, "W <= 0: W=%f agent=%d", W, agent_id);
    float log_w = -logf(W);
    DEVICE_FATAL_IF(log_w <= 0.0f, "log_w <= 0: W=%f log_w=%f agent=%d", W, log_w, agent_id);
    float cos_one_minus_alpha_phi = cosf(one_minus_alpha_phi);
    DEVICE_FATAL_IF(cos_one_minus_alpha_phi <= 0.0f, "cos_one_minus_alpha_phi <= 0: val=%f agent=%d", cos_one_minus_alpha_phi, agent_id);
    float levy_factor = powf(cos_one_minus_alpha_phi / log_w, ((1.0f - levy_alpha) / levy_alpha));

    float levy_sample = levy_num / levy_denom * levy_factor;

    DEVICE_FATAL_IF(U <= 0.0f, "U <= 0: U=%f agent=%d", U, agent_id);
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
        DEVICE_FATAL_IF(age <= 0.0f, "age <= 0 in friction loop: i=%d GRADIENT_HISTORY=%d agent=%d", i, GRADIENT_HISTORY, agent_id);
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

    int max_vel_slot = GenomeParamTable::max_agent_velocity;
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
}

__device__ void create_attractors_device(Organism* organism) {
    float* sources = organism->chemical_field->sources;
    float* attractor_positions = organism->attractor_positions;
    float* attractor_strengths = organism->attractor_strengths;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;
    int num_attractors = organism->num_attractors;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size) {
        int idx = y * grid_size + x;
        float source_value = 0.0f;

        float px = (float)x / grid_size;
        float py = (float)y / grid_size;

        for (int a = 0; a < num_attractors; a++) {
            float ax = attractor_positions[a * 2];
            float ay = attractor_positions[a * 2 + 1];
            float strength = attractor_strengths[a];

            float dx = fminf(fabsf(px - ax), NORMALIZED_MAX - fabsf(px - ax));
            float dy = fminf(fabsf(py - ay), NORMALIZED_MAX - fabsf(py - ay));
            float dist_sq = dx * dx + dy * dy;

            source_value += strength * expf(-dist_sq / (GAUSSIAN_VARIANCE_DENOMINATOR * FIELD_ATTRACTOR_SIGMA * FIELD_ATTRACTOR_SIGMA));
        }

        sources[idx] = source_value;
    }
}

__device__ void update_behavioral_embedding_device(Organism* organism) {
    int entry_idx = blockIdx.x;
    BehavioralState* agents = organism->behavioral_agents;
    float* embedding_weights = organism->embedding_weights;
    float* reconstruction_error = organism->reconstruction_error;
    int num_agents = organism->pool->capacity;
    float learning_rate = organism->embedding_learning_rate;
    const float* genome = &organism->workspace_genomes[entry_idx * GENOME_SIZE * 2];
    const float* gradients = organism->pool->entries[entry_idx].gradients;
    float ctx_complexity = organism->ctx_complexity;
    float ctx_niche = organism->ctx_niche;
    float ctx_learning = organism->ctx_learning;
    float ctx_performance = organism->ctx_performance;
    int hw_dim = organism->hw_dim;
    int task_dim = organism->task_dim;
    int gen_dim = organism->gen_dim;
    int behavioral_dim = hw_dim + task_dim + gen_dim;
    float* features_buffer = organism->features_buffer;
    const float* chemical_concentration = organism->chemical_field->concentration;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;

    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id < num_agents) {
        BehavioralState* agent = &agents[agent_id];

    float* features = &features_buffer[agent_id * behavioral_dim];

    float context_metabolic = agent->sensitivity;
    float context_stress = sqrtf(agent->velocity[0] * agent->velocity[0] +
                                 agent->velocity[1] * agent->velocity[1]);
    int cx = (int)(agent->position[0] * grid_size) % grid_size;
    int cy = (int)(agent->position[1] * grid_size) % grid_size;
    float context_morphogen = chemical_concentration[cy * grid_size + cx];

    int base_freq_slot = GenomeParamTable::fourier_base_freq;
    float fourier_base_freq = genome_to_param(genome, gradients, base_freq_slot, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, FOURIER_BASE_FREQ_MIN, FOURIER_BASE_FREQ_MAX);

    int num_octaves_slot = GenomeParamTable::fourier_num_octaves;
    int fourier_num_octaves_raw = (int)genome_to_param(genome, gradients, num_octaves_slot, context_metabolic, context_stress, context_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, (float)FOURIER_NUM_OCTAVES_MIN, (float)FOURIER_NUM_OCTAVES_MAX);
    int fourier_num_octaves = min(fourier_num_octaves_raw, behavioral_dim - 4);

    int spectrum_exp_slot = GenomeParamTable::fourier_spectrum_exponent;
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
        DEVICE_FATAL_IF(freq <= 0.0f, "freq <= 0 in fourier: k=%d base=%f agent=%d", k, fourier_base_freq, agent_id);
        float amplitude_weight = powf(freq, -fourier_spectrum_exponent);
        features[BASE_FEATURES_COUNT + k] = magnitude * amplitude_weight;
    }

    for (int d = 0; d < behavioral_dim; d++) {
        float reconstruction = 0.0f;
        for (int f = 0; f < behavioral_dim; f++) {
            reconstruction += features[f] * embedding_weights[f * behavioral_dim + d];
        }

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

        float error = reconstruction - target_coord[local_idx];
        target_coord[local_idx] += learning_rate * error;

        atomicAdd(reconstruction_error, error * error);
    }
    }
}

__device__ void init_embedding_weights_device(Organism* organism) {
    float* embedding_weights = organism->embedding_weights;
    int hw_dim = organism->hw_dim;
    int task_dim = organism->task_dim;
    int gen_dim = organism->gen_dim;
    int behavioral_dim = hw_dim + task_dim + gen_dim;
    unsigned int seed = organism->init_seed;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = behavioral_dim * behavioral_dim;

    if (idx < total_weights) {
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
}

__device__ void init_behavioral_state_device(Organism* organism) {
    int entry_idx = blockIdx.x;
    BehavioralState* agents = organism->behavioral_agents;
    int num_agents = organism->pool->capacity;
    unsigned int seed = organism->init_seed;
    const float* genome = &organism->workspace_genomes[entry_idx * GENOME_SIZE * 2];
    const float* epigenetic = organism->pool->entries[entry_idx].gradients;
    uint64_t genome_hash = organism->pool->entries[entry_idx].genome_hash;
    int organism_id = entry_idx;
    BehavioralInitSlots slots = organism->behavioral_slots;
    float ctx_complexity = organism->telemetry->genome_complexity.hash_entropy;
    float ctx_niche = organism->telemetry->archive_topology.novelty_gradient;
    float ctx_learning = organism->telemetry->diresa_evolution.behavioral_drift_rate;
    float ctx_performance = organism->telemetry->task_performance.accuracy;
    int hw_dim = organism->archive->hw_dim;
    int task_dim = organism->archive->task_dim;
    int gen_dim = organism->archive->gen_dim;

    int behavioral_dim = hw_dim + task_dim + gen_dim;
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (agent_id < num_agents) {
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

        InitContext ctx;
        ctx.derive_from_genome(genome, epigenetic);

        int embedding_scale_min_slot = GenomeParamTable::agent_embedding_scale_min;
        int embedding_scale_max_slot = GenomeParamTable::agent_embedding_scale_max;
        float embedding_scale_min = genome_slot_to_unit(genome, embedding_scale_min_slot) * AGENT_EMBEDDING_SCALE_BASE_MAX;
        float embedding_scale_max = embedding_scale_min + genome_slot_to_unit(genome, embedding_scale_max_slot) * (AGENT_EMBEDDING_SCALE_BASE_MAX - embedding_scale_min);

        float embedding_scale = genome_to_param(genome, epigenetic, slots.agent_embedding_scale, ctx.metabolic, ctx.stress, ctx.morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, embedding_scale_min, embedding_scale_max);
        int exploration_min_slot = GenomeParamTable::init_exploration_min;
        int exploration_max_slot = GenomeParamTable::init_exploration_max;
        float exploration_min = genome_slot_to_unit(genome, exploration_min_slot) * INIT_EXPLORATION_BASE_MAX;
        float exploration_max = exploration_min + genome_slot_to_unit(genome, exploration_max_slot) * (INIT_EXPLORATION_BASE_MAX - exploration_min);
        float init_exploration = genome_to_param(genome, epigenetic, slots.init_exploration, ctx.metabolic, ctx.stress, ctx.morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, exploration_min, exploration_max);

        int sensitivity_min_slot = GenomeParamTable::init_sensitivity_min;
        int sensitivity_max_slot = GenomeParamTable::init_sensitivity_max;
        float sensitivity_min = genome_slot_to_unit(genome, sensitivity_min_slot) * INIT_SENSITIVITY_BASE_MAX;
        float sensitivity_max = sensitivity_min + genome_slot_to_unit(genome, sensitivity_max_slot) * (INIT_SENSITIVITY_BASE_MAX - sensitivity_min);
        float init_sensitivity = genome_to_param(genome, epigenetic, slots.init_sensitivity, ctx.metabolic, ctx.stress, ctx.morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, sensitivity_min, sensitivity_max);

        int levy_alpha_min_slot = GenomeParamTable::levy_alpha_min;
        int levy_alpha_max_slot = GenomeParamTable::levy_alpha_max;
        float levy_alpha_min = genome_slot_to_unit(genome, levy_alpha_min_slot) * CHEMOTAXIS_LEVY_ALPHA_MAX;
        float levy_alpha_max = levy_alpha_min + genome_slot_to_unit(genome, levy_alpha_max_slot) * (CHEMOTAXIS_LEVY_ALPHA_MAX - levy_alpha_min);
        float levy_alpha = genome_to_param(genome, epigenetic, slots.levy_alpha, ctx.metabolic, ctx.stress, ctx.morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, levy_alpha_min, levy_alpha_max);

        agent->position[0] = prng_next(&rng);
        agent->position[1] = prng_next(&rng);

        agent->velocity[0] = 0.0f;
        agent->velocity[1] = 0.0f;

        for (int i = 0; i < hw_dim; i++) {
            agent->hw_coords[i] = prng_levy_stable(&rng, levy_alpha, embedding_scale);
        }
        for (int i = 0; i < task_dim; i++) {
            agent->task_coords[i] = prng_levy_stable(&rng, levy_alpha, embedding_scale);
        }
        for (int i = 0; i < gen_dim; i++) {
            agent->gen_coords[i] = prng_levy_stable(&rng, levy_alpha, embedding_scale);
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
    }
}

__device__ void init_chemical_field_device(Organism* organism) {
    ChemicalField* field = organism->chemical_field;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size) {
        int idx = y * grid_size + x;

        field->concentration[idx] = 0.0f;
        field->gradient_x[idx] = 0.0f;
        field->gradient_y[idx] = 0.0f;
        field->laplacian[idx] = 0.0f;
        field->sources[idx] = 0.0f;
        field->decay_factors[idx] = FIELD_CHEMICAL_DECAY;
    }
}

__device__ void reduce_concentration_mean_device(Organism* organism) {
    ChemicalField* field = organism->chemical_field;
    Architecture arch = Architecture::maxBounds();
    int total_cells = arch.grid_size * arch.grid_size;
    float* partial_sums = organism->reduction_partial_sums;

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = idx; i < total_cells; i += blockDim.x * gridDim.x) {
        sum += field->concentration[i];
    }
    sdata[tid] = sum;
    cg::this_grid().sync();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        cg::this_grid().sync();
    }

    if (tid < 32) {
        volatile float* vdata = sdata;
        vdata[tid] += vdata[tid + 32];
        vdata[tid] += vdata[tid + 16];
        vdata[tid] += vdata[tid + 8];
        vdata[tid] += vdata[tid + 4];
        vdata[tid] += vdata[tid + 2];
        vdata[tid] += vdata[tid + 1];
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__device__ void finalize_concentration_mean_device(Organism* organism) {
    ChemicalField* field = organism->chemical_field;
    float* partial_sums = organism->reduction_partial_sums;
    int num_blocks = organism->reduction_num_blocks;
    Architecture arch = Architecture::maxBounds();
    int total_cells = arch.grid_size * arch.grid_size;

    float sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        sum += partial_sums[i];
    }
    field->cached_mean = sum / (float)total_cells;
}

__device__ void set_chemical_sources_from_agents_device(Organism* organism) {
    int entry_idx = blockIdx.x;
    int num_entries = organism->pool->capacity;

    if (entry_idx < num_entries) {
        float* sources = organism->chemical_field->sources;
        BehavioralState* agents = organism->behavioral_agents;
        Architecture arch = Architecture::maxBounds();
        int grid_size = arch.grid_size;

        // Per-entry genome determines this organism's secretion behavior
        const float* genome = &organism->workspace_genomes[entry_idx * GENOME_SIZE * 2];
        const float* gradients = organism->pool->entries[entry_idx].gradients;
        const float* chemical_field = organism->chemical_field->concentration;
        float ctx_complexity = organism->ctx_complexity;
        float ctx_niche = organism->ctx_niche;
        float ctx_learning = organism->ctx_learning;
        float ctx_performance = organism->ctx_performance;

        // 1:1 mapping: entry i owns agent i
        int agent_id = entry_idx;
        BehavioralState* agent = &agents[agent_id];

        ChemotaxisParams params;

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
}

__device__ void compute_behavioral_field_device(Organism* organism) {
    float* behavioral_field = organism->behavioral_field;
    BehavioralState* agents = organism->behavioral_agents;
    int num_agents = organism->pool->capacity;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;
    int hw_dim = organism->hw_dim;
    int task_dim = organism->task_dim;
    int gen_dim = organism->gen_dim;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size) {
        float behavioral_field_sigma = FIELD_BEHAVIORAL_SIGMA;

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
        d_offset++;
    }
    }
}

__device__ void init_resource_fields_device(Organism* organism) {
    float* resource_density = organism->resource_density;
    float* fitness_landscape = organism->fitness_landscape;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;
    unsigned int seed = organism->init_seed * RNG_SEED_MULTIPLIER;

    int thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_y = threadIdx.y + blockIdx.y * blockDim.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int y = thread_y; y < grid_size; y += stride_y) {
        for (int x = thread_x; x < grid_size; x += stride_x) {
            int idx = y * grid_size + x;
            curandState_t state;
            curand_init(seed, idx, 0, &state);

            float rand_val = validated_curand_uniform(&state, "init_resource_fields", idx);
            resource_density[idx] = RESOURCE_INIT + RESOURCE_NOISE * (rand_val - CENTERED_DIFFERENCE_SCALE);

            fitness_landscape[idx] = 0.0f;
        }
    }
}

__device__ void initialize_chemical_field_device(Organism* organism) {
    float* chemical_field = organism->chemical_field->concentration;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;
    unsigned int seed = organism->init_seed;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size) {
        int idx = y * grid_size + x;

        curandState state;
        curand_init(seed, idx, 0, &state);

        float rand_val = validated_curand_uniform(&state, "initialize_chemical_field", idx);
        float noise = (rand_val - CENTERED_DIFFERENCE_SCALE) * CHEM_INIT_NOISE;

        chemical_field[idx] = clamp(CHEM_INIT_BASE + noise, NORMALIZED_MIN, NORMALIZED_MAX);
    }
}

__device__ void resource_flow_device(Organism* organism) {
    float* resource_density = organism->rd_resource_density;
    float* resource_next = organism->rd_resource_next;
    const float* fitness_landscape = organism->rd_fitness_landscape;
    float* resource_gradient_x = organism->rd_resource_gradient_x;
    float* resource_gradient_y = organism->rd_resource_gradient_y;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;
    float dt = organism->dt;
    float diffusivity = organism->rd_diffusivity;
    float flow_strength = organism->rd_flow_strength;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size) {
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

        resource_gradient_x[idx] = grad_rho_x;
        resource_gradient_y[idx] = grad_rho_y;

        float drho_dt = -divergence + diffusivity * lap_rho;

        float new_rho = rho + dt * drho_dt;
        DEVICE_FATAL_IF(new_rho < -0.1f, "new_rho < -0.1: rho=%f drho_dt=%f dt=%f x=%d y=%d", rho, drho_dt, dt, x, y);
        resource_next[idx] = fmaxf(0.0f, new_rho);
    }
}

__device__ void update_fitness_landscape_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    BehavioralState* agents = organism->behavioral_agents;
    float* fitness_landscape = organism->rd_fitness_landscape;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;

    int compact_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_idx < pool->alive_indices_count) {
        int entry_idx = pool->alive_indices[compact_idx];
        DEVICE_FATAL_IF(!pool->alive_flags[entry_idx], "update_fitness_landscape_device: dead entry in alive_indices");

        float px = agents[entry_idx].position[0];
        float py = agents[entry_idx].position[1];
        int gx = min((int)(px * grid_size), grid_size - 1);
        int gy = min((int)(py * grid_size), grid_size - 1);
        int idx = gy * grid_size + gx;

        atomicAdd(&fitness_landscape[idx], pool->fitness_values[entry_idx]);
    }
}

__device__ void populate_organism_flow_params_device(Organism* organism) {
    ComponentPool* pool = organism->pool;
    ChemicalField* chemical_field = organism->chemical_field;
    float* workspace_genomes = organism->workspace_genomes;

    int compact_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_idx < pool->alive_indices_count) {
        int entry_idx = pool->alive_indices[compact_idx];
    PoolEntry* entry = &pool->entries[entry_idx];
    DEVICE_FATAL_IF(!pool->alive_flags[entry_idx], "populate_organism_flow_params_device: dead entry in alive_indices");

    Architecture arch = get_arch_from_pool(organism, entry_idx);

    float* entry_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];
    float* entry_parent_temp = &workspace_genomes[entry_idx * GENOME_SIZE * 2 + GENOME_SIZE];
    reconstruct_genome_from_archive(entry->parent_hash, organism->archive, organism->archive_size,
        entry->delta_indices, entry->delta_values, entry->num_deltas,
        entry->max_deltas, entry_genome, GENOME_SIZE, entry_parent_temp, organism->diresa_genome_weights);

    float* genome = entry_genome;
    float* epigenetic = entry->gradients;
    uint64_t genome_hash = entry->genome_hash;

    float context_metabolic = entry->fitness.value;
    int stress_numerator_slot = GenomeParamTable::context_stress_numerator;
    float stress_numerator = genome_to_param(
        genome, epigenetic, stress_numerator_slot,
        entry->fitness.value,
        entry->hunger.value,
        safe_epsilon(1.0f),
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        NORMALIZED_MIN, NORMALIZED_MAX
    );
    DEVICE_FATAL_IF(entry->hunger.value <= 0.0f, "archive_driven_lifecycle: hunger <= 0 before stress division");
    float context_stress = stress_numerator / entry->hunger.value;

    float context_morphogen = chemical_field->cached_mean;

    int s_param_slot = GenomeParamTable::flow_lenia_s;
    entry->flow_s = genome_to_param(
        genome, epigenetic, s_param_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_S_MIN, FLOW_LENIA_S_MAX
    );

    int beta_A_slot = GenomeParamTable::flow_lenia_beta_A;
    entry->flow_beta_A = genome_to_param(
        genome, epigenetic, beta_A_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_BETA_A_MIN, FLOW_LENIA_BETA_A_MAX
    );

    int n_param_slot = GenomeParamTable::flow_lenia_n;
    entry->flow_n = genome_to_param(
        genome, epigenetic, n_param_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_N_MIN, FLOW_LENIA_N_MAX
    );

    int alpha_min_slot = GenomeParamTable::flow_alpha_min;
    entry->flow_alpha_min = genome_to_param(
        genome, epigenetic, alpha_min_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_ALPHA_MIN_MIN, FLOW_LENIA_ALPHA_MIN_MAX
    );

    int alpha_max_slot = GenomeParamTable::flow_alpha_max;
    entry->flow_alpha_max = genome_to_param(
        genome, epigenetic, alpha_max_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_ALPHA_MAX_MIN, FLOW_LENIA_ALPHA_MAX_MAX
    );

    int sharpness_slot = GenomeParamTable::flow_sharpness;
    entry->flow_sharpness = genome_to_param(
        genome, epigenetic, sharpness_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        FLOW_LENIA_SHARPNESS_MIN, FLOW_LENIA_SHARPNESS_MAX
    );

    int resource_flow_dt_slot = GenomeParamTable::flow_resource_dt;
    entry->flow_resource_dt = genome_to_param(
        genome, epigenetic, resource_flow_dt_slot,
        context_metabolic, context_stress, context_morphogen,
        organism->telemetry->genome_complexity.hash_entropy,
        organism->telemetry->archive_topology.novelty_gradient,
        organism->telemetry->diresa_evolution.behavioral_drift_rate,
        organism->telemetry->task_performance.accuracy,
        RESOURCE_FLOW_DT_MIN, RESOURCE_FLOW_DT_MAX
    );
    }
}

__device__ void behavioral_update_device(Organism* organism) {
    BehavioralState* agents = organism->behavioral_agents;
    ChemicalField* chemical_field = organism->chemical_field;
    float* workspace_genomes = organism->workspace_genomes;
    Architecture arch = get_arch_from_pool(organism, blockIdx.x);

    int entry_idx = blockIdx.x;
    ComponentPool* pool = organism->pool;

    if (entry_idx < pool->capacity && pool->alive_flags[entry_idx]) {
        PoolEntry* entry = &pool->entries[entry_idx];
        int num_agents = POOL_CAPACITY_MAX;

    float* primary_genome = &workspace_genomes[entry_idx * GENOME_SIZE * 2];
    float* primary_parent_temp = &workspace_genomes[entry_idx * GENOME_SIZE * 2 + GENOME_SIZE];

    reconstruct_genome_from_archive(entry->parent_hash, (GPUElite*)organism->archive, organism->archive_size,
        entry->delta_indices, entry->delta_values, entry->num_deltas,
        entry->max_deltas, primary_genome, GENOME_SIZE, primary_parent_temp, organism->diresa_genome_weights);

    uint64_t genome_hash = entry->genome_hash;
    float* genome = primary_genome;
    float* gradients = entry->gradients;
    float ctx_metabolic = entry->fitness.value;
    float ctx_stress = entry->hunger.value;

    float ctx_morphogen = chemical_field->cached_mean;

    BehavioralDimensions dims;
    dims.derive_from_genome();

    int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;
    int behavioral_buffer_size = arch.grid_size * arch.grid_size * behavioral_dim;
    float* behavioral_field = &organism->behavioral_field_pool[entry_idx * behavioral_buffer_size];
    float* behavioral_gradients_pool = &organism->behavioral_gradient_pool[entry_idx * behavioral_buffer_size];

    {
        int grid_size = arch.grid_size;
        int total_cells = grid_size * grid_size;
        int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;

        ChemotaxisParams chem_params;
        

        InitContext init_ctx;
        init_ctx.derive_from_genome(primary_genome, entry->gradients);

        float behavioral_field_sigma = chem_params.get_behavioral_field_sigma(primary_genome, entry->gradients, init_ctx.metabolic, init_ctx.stress, init_ctx.morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

        for (int cell_idx = threadIdx.x; cell_idx < total_cells; cell_idx += blockDim.x) {
            int x = cell_idx % grid_size;
            int y = cell_idx / grid_size;
            float px = (float)x / grid_size;
            float py = (float)y / grid_size;

            int d_offset = 0;

            for (int d = 0; d < dims.hw_dim; d++) {
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
                DEVICE_FATAL_IF(!is_meaningful(weight_sum, 1.0f), "behavioral_update: hw weight_sum not meaningful");
                behavioral_field[field_idx] = field_value / weight_sum;
                d_offset++;
            }

            for (int d = 0; d < dims.task_dim; d++) {
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
                DEVICE_FATAL_IF(!is_meaningful(weight_sum, 1.0f), "behavioral_update: task weight_sum not meaningful");
                behavioral_field[field_idx] = field_value / weight_sum;
                d_offset++;
            }

            for (int d = 0; d < dims.gen_dim; d++) {
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
                DEVICE_FATAL_IF(!is_meaningful(weight_sum, 1.0f), "behavioral_update: gen weight_sum not meaningful");
                behavioral_field[field_idx] = field_value / weight_sum;
                d_offset++;
            }
        }
        cg::this_grid().sync();
    }

    {
        int width = arch.grid_size;
        int height = arch.grid_size;
        int total_cells = width * height;
        int num_tiles = (total_cells + WARP_SIZE - 1) / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        CAParams ca_params;
        float warp_ca_growth_rate = ca_params.get_warp_ca_growth_rate(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

        for (int tile = 0; tile < num_tiles; tile++) {
            int cell_idx = tile * WARP_SIZE + lane_id;
            if (cell_idx >= total_cells) continue;

            int tile_x = cell_idx % width;
            int tile_y = cell_idx / width;

            float my_state = behavioral_field[tile_y * width + tile_x];
            unsigned mask = 0xffffffff;
            float sum = 0.0f;
            sum += get_neighbor_2d(my_state, -1, -1, width, mask);
            sum += get_neighbor_2d(my_state, 0, -1, width, mask);
            sum += get_neighbor_2d(my_state, 1, -1, width, mask);
            sum += get_neighbor_2d(my_state, -1, 0, width, mask);
            sum += get_neighbor_2d(my_state, 1, 0, width, mask);
            sum += get_neighbor_2d(my_state, -1, 1, width, mask);
            sum += get_neighbor_2d(my_state, 0, 1, width, mask);
            sum += get_neighbor_2d(my_state, 1, 1, width, mask);

            float avg = sum / CA_KERNEL_NEIGHBOR_COUNT;
            float growth = avg * expf(-avg * avg * 2.0f);

            float total_mass = WarpReduce<WARP_SIZE>::sum(my_state);
            float new_val = my_state + warp_ca_growth_rate * growth;
            float new_total = WarpReduce<WARP_SIZE>::sum(new_val);

            DEVICE_FATAL_IF(!is_meaningful(new_total, total_mass), "behavioral_update: mass conservation failed - new_total not meaningful");
            new_val *= total_mass / new_total;

            behavioral_gradients_pool[tile_y * width + tile_x] = new_val;
        }
        cg::this_grid().sync();
    }

    {
        int grid_size = arch.grid_size;
        int total_cells = grid_size * grid_size;
        int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;

        for (int cell_idx = threadIdx.x; cell_idx < total_cells; cell_idx += blockDim.x) {
            int x = cell_idx % grid_size;
            int y = cell_idx / grid_size;

            for (int dim = 0; dim < behavioral_dim; dim++) {
                float grad_x, grad_y;
                Stencils::gradients_at(grad_x, grad_y, &behavioral_gradients_pool[dim], x, y, grid_size, behavioral_dim);

                float grad_sq = grad_x * grad_x + grad_y * grad_y;
                float magnitude = sqrtf(grad_sq) + safe_epsilon(grad_sq);
                grad_x /= magnitude;
                grad_y /= magnitude;

                int grad_idx = ((y * grid_size + x) * behavioral_dim + dim) * 2;
                behavioral_gradients_pool[grad_idx] = grad_x;
                behavioral_gradients_pool[grad_idx + 1] = grad_y;
            }
        }
        cg::this_grid().sync();
    }

    int chemotaxis_dt_slot = GenomeParamTable::chemotaxis_dt;
    float chemotaxis_dt = genome_to_param(genome, gradients, chemotaxis_dt_slot, ctx_metabolic, ctx_stress, ctx_morphogen, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, CHEMOTAXIS_DT_MIN, CHEMOTAXIS_DT_MAX);

    {
        int grid_size = arch.grid_size;
        int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;
        float* concentration = chemical_field->concentration;
        float* gradient_x_arr = chemical_field->gradient_x;
        float* gradient_y_arr = chemical_field->gradient_y;

        for (int agent_id = threadIdx.x; agent_id < num_agents; agent_id += blockDim.x) {
            BehavioralState* agent = &agents[agent_id];

            float context_metabolic_agent = agent->sensitivity;
            float context_stress_agent = sqrtf(agent->velocity[0] * agent->velocity[0] + agent->velocity[1] * agent->velocity[1]);
            int grid_x = min(max((int)(agent->position[0] * grid_size), 0), grid_size - 1);
            int grid_y = min(max((int)(agent->position[1] * grid_size), 0), grid_size - 1);
            int idx = grid_y * grid_size + grid_x;
            float context_morphogen_agent = concentration[idx];

            ChemotaxisParams chem_params;

            float theta = chem_params.get_theta(primary_genome, entry->gradients, context_metabolic_agent, context_stress_agent, context_morphogen_agent, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
            float sigma = chem_params.get_sigma(primary_genome, entry->gradients, context_metabolic_agent, context_stress_agent, context_morphogen_agent, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);
            float gradient_mix_weight = chem_params.get_gradient_mix_weight(primary_genome, entry->gradients, context_metabolic_agent, context_stress_agent, context_morphogen_agent, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy);

            float chem_grad_x = gradient_x_arr[idx];
            float chem_grad_y = gradient_y_arr[idx];

            float behav_grad_x = 0.0f, behav_grad_y = 0.0f;
            int d_offset = 0;
            for (int d = 0; d < dims.hw_dim; d++) {
                int grad_idx = ((grid_y * grid_size + grid_x) * behavioral_dim + d_offset) * 2;
                behav_grad_x += behavioral_gradients_pool[grad_idx] * agent->hw_coords[d];
                behav_grad_y += behavioral_gradients_pool[grad_idx + 1] * agent->hw_coords[d];
                d_offset++;
            }
            for (int d = 0; d < dims.task_dim; d++) {
                int grad_idx = ((grid_y * grid_size + grid_x) * behavioral_dim + d_offset) * 2;
                behav_grad_x += behavioral_gradients_pool[grad_idx] * agent->task_coords[d];
                behav_grad_y += behavioral_gradients_pool[grad_idx + 1] * agent->task_coords[d];
                d_offset++;
            }
            for (int d = 0; d < dims.gen_dim; d++) {
                int grad_idx = ((grid_y * grid_size + grid_x) * behavioral_dim + d_offset) * 2;
                behav_grad_x += behavioral_gradients_pool[grad_idx] * agent->gen_coords[d];
                behav_grad_y += behavioral_gradients_pool[grad_idx + 1] * agent->gen_coords[d];
                d_offset++;
            }

            float behav_sq = behav_grad_x * behav_grad_x + behav_grad_y * behav_grad_y;
            float behav_magnitude = sqrtf(behav_sq) + safe_epsilon(behav_sq);
            behav_grad_x /= behav_magnitude;
            behav_grad_y /= behav_magnitude;

            float chem_weight = gradient_mix_weight;
            float behav_weight = NORMALIZED_MAX - gradient_mix_weight;
            float mixed_grad_x = chem_grad_x * chem_weight + behav_grad_x * behav_weight;
            float mixed_grad_y = chem_grad_y * chem_weight + behav_grad_y * behav_weight;

            unsigned int seed = agent_id * RNG_SEED_MULTIPLIER + (unsigned int)(chemotaxis_dt * 1000.0f);
            seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
            float noise_scale = sigma * agent->exploration_noise;
            float noise_x = noise_scale * ((seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE - 0.5f);
            seed = seed * LCG_MULTIPLIER + LCG_INCREMENT;
            float noise_y = noise_scale * ((seed & 0xFFFFFF) / RNG_NORMALIZATION_SCALE - 0.5f);

            agent->velocity[0] += chemotaxis_dt * (agent->sensitivity * mixed_grad_x - theta * agent->velocity[0] + noise_x);
            agent->velocity[1] += chemotaxis_dt * (agent->sensitivity * mixed_grad_y - theta * agent->velocity[1] + noise_y);

            float vel_magnitude = sqrtf(agent->velocity[0] * agent->velocity[0] + agent->velocity[1] * agent->velocity[1]);
            int max_vel_slot = GenomeParamTable::max_agent_velocity;
            float max_agent_velocity = genome_to_param(primary_genome, entry->gradients, max_vel_slot, context_metabolic_agent, context_stress_agent, context_morphogen_agent, organism->telemetry->genome_complexity.hash_entropy, organism->telemetry->archive_topology.novelty_gradient, organism->telemetry->diresa_evolution.behavioral_drift_rate, organism->telemetry->task_performance.accuracy, MAX_AGENT_VELOCITY_BASE_MIN, MAX_AGENT_VELOCITY_BASE_MAX);
            if (vel_magnitude > max_agent_velocity) {
                agent->velocity[0] *= (max_agent_velocity / vel_magnitude);
                agent->velocity[1] *= (max_agent_velocity / vel_magnitude);
            }

            agent->position[0] += agent->velocity[0] * chemotaxis_dt;
            agent->position[1] += agent->velocity[1] * chemotaxis_dt;
            agent->position[0] = agent->position[0] - floorf(agent->position[0]);
            agent->position[1] = agent->position[1] - floorf(agent->position[1]);
        }
        cg::this_grid().sync();
    }

    {
        int behavioral_dim = dims.hw_dim + dims.task_dim + dims.gen_dim;
        int memory_entry_size = behavioral_dim + AGENT_SPATIAL_DIMS;
        float* memory_data_pool = organism->memory_data_pool;

        for (int agent_id = threadIdx.x; agent_id < num_agents; agent_id += blockDim.x) {
            float* d_memory_data = memory_data_pool + agent_id * memory_entry_size;

            d_memory_data[0] = agents[agent_id].position[0];
            d_memory_data[1] = agents[agent_id].position[1];
            d_memory_data[2] = agents[agent_id].velocity[0];
            d_memory_data[3] = agents[agent_id].velocity[1];

            int offset = AGENT_SPATIAL_DIMS;
            for (int i = 0; i < dims.hw_dim; i++) {
                d_memory_data[offset++] = agents[agent_id].hw_coords[i];
            }
            for (int i = 0; i < dims.task_dim; i++) {
                d_memory_data[offset++] = agents[agent_id].task_coords[i];
            }
            for (int i = 0; i < dims.gen_dim; i++) {
                d_memory_data[offset++] = agents[agent_id].gen_coords[i];
            }
        }
        cg::this_grid().sync();

        if (threadIdx.x == 0) {
            float importance = agents[0].exploration_noise;
            float* d_memory_data = memory_data_pool;
            TemporalTube* tube = organism->memory_tubes;

            int idx = tube->head;
            tube->entries[idx].size = memory_entry_size;
            tube->entries[idx].timestamp = tube->global_time;
            tube->entries[idx].importance = importance;
            tube->entries[idx].decay_factor = 1.0f;

            DEVICE_FATAL_IF(d_memory_data == nullptr, "behavioral_update: d_memory_data is null");
            DEVICE_FATAL_IF(memory_entry_size <= 0, "behavioral_update: memory_entry_size <= 0");
            DEVICE_FATAL_IF(tube->entries[idx].data == nullptr, "behavioral_update: tube entry data is null");
            for (int i = 0; i < memory_entry_size; i++) {
                tube->entries[idx].data[i] = d_memory_data[i];
            }

            tube->head = (tube->head + 1) % tube->capacity;
            if (tube->count < tube->capacity) {
                tube->count++;
            }
        }
        cg::this_grid().sync();
    }
    }
}

#endif

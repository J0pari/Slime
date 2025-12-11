
#ifndef CHEMOTAXIS_CU
#define CHEMOTAXIS_CU
#include "../config/config.cu"
#include "../utils/genome_params.cuh"
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
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) printf("[SNAPSHOT1] Entry field_size=%d\n", field_size);

    TemporalTube* history = field->history;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) printf("[SNAPSHOT2] history=%p\n", history);

    int next_head = (history->head + 1) % history->capacity;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) printf("[SNAPSHOT3] next_head=%d capacity=%d\n", next_head, history->capacity);

    float* dest = history->entries[next_head].data;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) printf("[SNAPSHOT4] dest=%p\n", dest);

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < field_size; i += blockDim.x * gridDim.x) {
        dest[i] = field->concentration[i];
    }
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) printf("[SNAPSHOT5] Data copied\n");
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("[SNAPSHOT6] Thread 0 updating metadata\n");
        int decay_slot = derive_param_slot(genome_hash, "memory_decay_factor");
        printf("[SNAPSHOT7] decay_slot=%d\n", decay_slot);
        int importance_slot = derive_param_slot(genome_hash, "memory_importance");
        printf("[SNAPSHOT8] importance_slot=%d\n", importance_slot);

        history->entries[next_head].timestamp = global_time;
        printf("[SNAPSHOT9] timestamp set\n");
        history->entries[next_head].decay_factor = (genome[decay_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        printf("[SNAPSHOT10] decay_factor set\n");
        history->entries[next_head].importance = (genome[importance_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        printf("[SNAPSHOT11] importance set\n");

        history->head = next_head;
        if (history->count < history->capacity) {
            history->count++;
        }
        history->global_time = global_time;
        printf("[SNAPSHOT12] Exit SUCCESS\n");
    }
}

__global__ void store_chemical_snapshot_kernel(ChemicalField* field, int field_size, float global_time, uint64_t genome_hash, const float* genome) {
    store_chemical_snapshot(field, field_size, global_time, genome_hash, genome);
}

__global__ void initialize_ca_from_field_kernel(
    float* __restrict__ ca_concentration,
    float* __restrict__ chemical_concentration,
    int grid_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;

    
    ca_concentration[idx] = chemical_concentration[idx];
}

__global__ void update_field_from_ca_kernel(
    float* __restrict__ chemical_concentration,
    float* __restrict__ ca_output,
    int grid_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;

    
    chemical_concentration[idx] = ca_output[idx];
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
                            concentration, x, y, grid_size);

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

__global__ void behavioral_gradient_kernel(float* __restrict__ behavioral_field, float* __restrict__ behavioral_gradients, int grid_size, int hw_dim, int task_dim, int gen_dim){
    int behavioral_dim = hw_dim + task_dim + gen_dim;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int dim = blockIdx.z;

    if (x >= grid_size || y >= grid_size || dim >= behavioral_dim) return;

    float grad_x, grad_y;
    Stencils::gradients_at(grad_x, grad_y, &behavioral_field[dim], x, y, grid_size * behavioral_dim);

    float magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y) + EPSILON_SMALL;
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

    float behav_magnitude = sqrtf(behav_grad_x * behav_grad_x +
                                  behav_grad_y * behav_grad_y) + EPSILON_SMALL;
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

    float phi = TAU * (V - CENTERED_DIFFERENCE_SCALE);
    float half_pi_alpha = (CUDART_PI_F * CENTERED_DIFFERENCE_SCALE) * levy_alpha;
    float xi = (CUDART_PI_F * CENTERED_DIFFERENCE_SCALE) - half_pi_alpha + phi;
    float alpha_phi = levy_alpha * phi;

    float levy_num = sinf(alpha_phi);
    float levy_denom = powf(cosf(phi) + EPSILON, (NORMALIZED_MAX / levy_alpha));
    float levy_factor = powf(cosf(xi) / (-logf(W + EPSILON) + EPSILON), ((NORMALIZED_MAX - levy_alpha) / levy_alpha));

    float levy_sample = levy_num / levy_denom * levy_factor;

    // Tempered L\u00e9vy flight: Mix with Gaussian for bounded jumps (prevents infinite excursions)
    // Box-Muller transform using U for Gaussian component
    float gaussian_component = sqrtf(-2.0f * logf(U + EPSILON)) * cosf(TAU * V);
    float tempering_weight = expf(-levy_alpha * fabsf(levy_sample));  // Exponential tempering
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
        float kernel_weight = powf(age + EPSILON, kernel_exponent);
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
    float context_stress = 0.0f;
    float context_morphogen = 0.0f;

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

__global__ void update_behavioral_embedding_kernel(BehavioralState* __restrict__ agents, float* __restrict__ embedding_weights, float* __restrict__ reconstruction_error, int num_agents, float learning_rate, const float* genome, const float* gradients, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance, int behavioral_dim){
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id >= num_agents) return;

    BehavioralState* agent = &agents[agent_id];

    float* features;
    cudaError_t malloc_err = cudaMalloc(&features, sizeof(float) * behavioral_dim);
    if (malloc_err != cudaSuccess || features == nullptr) {
        printf("[ERROR] cudaMalloc failed for features in agent %d: %s\n", agent_id, cudaGetErrorString(malloc_err));
        return;
    }

    float context_metabolic = agent->sensitivity;
    float context_stress = sqrtf(agent->velocity[0] * agent->velocity[0] +
                                 agent->velocity[1] * agent->velocity[1]);
    float context_morphogen = 0.0f;

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
        float amplitude_weight = powf(freq + EPSILON, -fourier_spectrum_exponent);
        features[BASE_FEATURES_COUNT + k] = magnitude * amplitude_weight;
    }

    cudaFree(features);
}

__global__ void init_behavioral_state_kernel(BehavioralState* agents, int num_agents, unsigned int seed, const float* genome, const float* epigenetic, uint64_t genome_hash, int organism_id, BehavioralInitSlots slots, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance, int hw_dim, int task_dim, int gen_dim){
    int behavioral_dim = hw_dim + task_dim + gen_dim;
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (agent_id == 0) {
        printf("[behavioral] agents=%p num=%d seed=%u genome=%p genome_hash=%llu organism_id=%d\n",
               agents, num_agents, seed, genome, genome_hash, organism_id);
        printf("[behavioral] dims: hw=%d task=%d gen=%d total=%d hw_coords=%p task_coords=%p gen_coords=%p\n",
               hw_dim, task_dim, gen_dim, behavioral_dim, agents[0].hw_coords, agents[0].task_coords, agents[0].gen_coords);
    }

    if (agent_id >= num_agents) return;

    uint64_t stream_seed = ((uint64_t)seed << 32) | (uint64_t)agent_id;
    stream_seed ^= stream_seed >> 33;
    stream_seed *= 0xff51afd7ed558ccdULL;
    stream_seed ^= stream_seed >> 33;
    stream_seed *= 0xc4ceb9fe1a85ec53ULL;
    stream_seed ^= stream_seed >> 33;

    PRNGState rng;
    rng.s0 = stream_seed;
    rng.s1 = stream_seed ^ 0x9e3779b97f4a7c15ULL;

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
        printf("[behavioral] agent0 levy params: alpha=%f scale=%f\n", levy_alpha, embedding_scale);
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
        printf("[behavioral] agent0 coords assigned: hw[0]=%f task[0]=%f gen[0]=%f\n",
               agent->hw_coords[0], agent->task_coords[0], agent->gen_coords[0]);
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
        printf("[behavioral] agent0: pos=[%f,%f] vel=[%f,%f] exploration=%f sensitivity=%f levy_alpha=%f\n",
               agent->position[0], agent->position[1], agent->velocity[0], agent->velocity[1],
               init_exploration, init_sensitivity, levy_alpha);
        printf("[behavioral] agent0: hw_coords[0]=%f task_coords[0]=%f gen_coords[0]=%f\n",
               agent->hw_coords[0], agent->task_coords[0], agent->gen_coords[0]);
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
        printf("[chemical_field] field=%p grid=%d field_size=%d decay=%f nulls=%d conc=%p grad_x=%p laplacian=%p\n",
               field, grid_size, grid_size*grid_size, chemical_decay, null_check, field->concentration, field->gradient_x, field->laplacian);
        printf("[chemical_field] verify idx0: conc=%f grad_x=%f grad_y=%f decay=%f\n",
               field->concentration[0], field->gradient_x[0], field->gradient_y[0], field->decay_factors[0]);
    }
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

            atomicAdd(&sources[idx], contribution);
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
        behavioral_field[field_idx] = weight_sum > EPSILON_SMALL ? field_value / weight_sum : 0.0f;
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
        behavioral_field[field_idx] = weight_sum > EPSILON_SMALL ? field_value / weight_sum : 0.0f;
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
        behavioral_field[field_idx] = weight_sum > EPSILON_SMALL ? field_value / weight_sum : 0.0f;
        d_offset++;
    }
}

__global__ void init_rd_fields_kernel(
    float* __restrict__ u_field,
    float* __restrict__ v_field,
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

    int u_init_slot = derive_param_slot(genome_hash, "rd_u_field_initial");
    float rd_u_init = (genome[u_init_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float rd_u_value = RD_U_INIT_MIN + rd_u_init * (RD_U_INIT_MAX - RD_U_INIT_MIN);

    u_field[idx] = rd_u_value;

    int cx = grid_size / 2;
    int cy = grid_size / 2;
    float dx = x - cx;
    float dy = y - cy;
    float r = sqrtf(dx*dx + dy*dy);

    int perturbation_radius_slot = derive_param_slot(genome_hash, "rd_perturbation_radius");
    float perturbation_radius_norm = (genome[perturbation_radius_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float perturbation_radius = RD_PERTURBATION_RADIUS_MIN + perturbation_radius_norm * (RD_PERTURBATION_RADIUS_MAX - RD_PERTURBATION_RADIUS_MIN);

    int v_perturbation_slot = derive_param_slot(genome_hash, "rd_v_perturbation");
    float v_perturbation_norm = (genome[v_perturbation_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float v_perturbation_base = RD_V_PERTURBATION_MIN + v_perturbation_norm * (RD_V_PERTURBATION_MAX - RD_V_PERTURBATION_MIN);

    int v_scale_slot = derive_param_slot(genome_hash, "rd_v_perturbation_scale");
    float v_scale_norm = (genome[v_scale_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    float v_scale = v_scale_norm * (RD_V_PERTURBATION_MAX - RD_V_PERTURBATION_MIN);

    v_field[idx] = (r < perturbation_radius) ? v_perturbation_base + curand_uniform(&state) * v_scale : NORMALIZED_MIN;
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

    resource_density[idx] = resource_init + resource_noise * (curand_uniform(&state) - CENTERED_DIFFERENCE_SCALE);

    fitness_landscape[idx] = DEFAULT_FITNESS;
}

__global__ void initialize_chemical_field_kernel(
    float* __restrict__ chemical_field,
    const float* __restrict__ genome,
    int grid_size,
    unsigned int seed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float base_value = 0.5f;
    float genome_influence = 0.0f;

    int genome_idx = (x + y * grid_size) % GENOME_SIZE;
    if (genome != nullptr) {
        genome_influence = genome[genome_idx] * 0.1f;
    }

    float noise = (curand_uniform(&state) - 0.5f) * 0.2f;

    chemical_field[idx] = base_value + genome_influence + noise;
    chemical_field[idx] = clamp(chemical_field[idx], 0.0f, 1.0f);
}

extern "C" __global__ void resource_flow_kernel(
    float* __restrict__ resource_density,
    float* __restrict__ resource_next,
    const float* __restrict__ fitness_landscape,
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
    Stencils::gradients_at(grad_fitness_x, grad_fitness_y, fitness_landscape, x, y, grid_size);

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
    Stencils::all_operators(grad_rho_x, grad_rho_y, lap_rho, rho_center, resource_density, x, y, grid_size);

    float drho_dt = -divergence + diffusivity * lap_rho;

    resource_next[idx] = rho + dt * drho_dt;
    resource_next[idx] = fmaxf(0.0f, resource_next[idx]);
}

extern "C" __global__ void update_fitness_landscape_kernel(
    ComponentPool* pool,
    float* __restrict__ fitness_landscape,
    int grid_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;

    float total_fitness = 0.0f;
    int count = 0;

    for (int i = 0; i < pool->capacity; i++) {
        if (pool->entries[i].alive) {
            total_fitness += pool->entries[i].fitness;
            count++;
        }
    }

    fitness_landscape[idx] = (count > 0) ? (total_fitness / count) : DEFAULT_FITNESS;
}

#endif

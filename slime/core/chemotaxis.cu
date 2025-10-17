// slime/core/chemotaxis.cu - Behavioral navigation and gradient following
#ifndef CHEMOTAXIS_CU
#define CHEMOTAXIS_CU
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Configuration
constexpr int CHEM_GRID_SIZE = 128;
constexpr int BEHAVIORAL_DIM = 10;  // DIRESA embedding dimension
constexpr int GRADIENT_HISTORY = 32;
constexpr int NUM_ATTRACTORS = 8;
constexpr float DIFFUSION_RATE = 0.1f;
constexpr float DECAY_RATE = 0.95f;
constexpr float SENSITIVITY_THRESHOLD = 0.01f;

// Chemical field structure
struct ChemicalField {
    float* concentration;      // [CHEM_GRID_SIZE][CHEM_GRID_SIZE]
    float* gradient_x;         // Spatial gradients
    float* gradient_y;
    float* laplacian;          // For diffusion
    float* sources;            // Attractor positions
    float* decay_factors;      // Per-cell decay
};

// Behavioral state for navigation
struct BehavioralState {
    float position[2];         // Current position
    float velocity[2];         // Current velocity
    float behavioral_coords[BEHAVIORAL_DIM];  // DIRESA embedding
    float gradient_memory[GRADIENT_HISTORY][2];  // Gradient history
    float exploration_noise;   // Exploration vs exploitation
    float sensitivity;         // Response strength
    int memory_index;          // Circular buffer index
};

// Compute chemical diffusion with reaction
__global__ void diffusion_reaction_kernel(
    float* __restrict__ concentration,
    float* __restrict__ gradient_x,
    float* __restrict__ gradient_y,
    float* __restrict__ laplacian,
    float* __restrict__ sources,
    int grid_size,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;

    // Compute spatial gradients using central differences
    float c_center = concentration[idx];
    float c_left = x > 0 ? concentration[idx - 1] : c_center;
    float c_right = x < grid_size - 1 ? concentration[idx + 1] : c_center;
    float c_up = y > 0 ? concentration[idx - grid_size] : c_center;
    float c_down = y < grid_size - 1 ? concentration[idx + grid_size] : c_center;

    // Gradients
    gradient_x[idx] = (c_right - c_left) * 0.5f;
    gradient_y[idx] = (c_down - c_up) * 0.5f;

    // Laplacian for diffusion
    laplacian[idx] = (c_left + c_right + c_up + c_down - 4.0f * c_center);

    // Reaction-diffusion update (Gray-Scott model parameters)
    float feed = 0.055f;
    float kill = 0.062f;

    // Add sources
    float source_contribution = sources[idx];

    // Update concentration
    float diffusion = DIFFUSION_RATE * laplacian[idx];
    float reaction = -c_center * c_center * c_center + feed * (1.0f - c_center);
    float decay = -kill * c_center;

    concentration[idx] = c_center + dt * (diffusion + reaction + decay + source_contribution);
    concentration[idx] = fmaxf(0.0f, fminf(1.0f, concentration[idx]));
}

// Compute behavioral gradients in DIRESA space
__global__ void behavioral_gradient_kernel(
    float* __restrict__ behavioral_field,     // [CHEM_GRID_SIZE][CHEM_GRID_SIZE][BEHAVIORAL_DIM]
    float* __restrict__ behavioral_gradients, // [CHEM_GRID_SIZE][CHEM_GRID_SIZE][BEHAVIORAL_DIM][2]
    int grid_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int dim = blockIdx.z;

    if (x >= grid_size || y >= grid_size || dim >= BEHAVIORAL_DIM) return;

    int field_idx = (y * grid_size + x) * BEHAVIORAL_DIM + dim;

    // Sobel operator for robust gradient estimation
    float sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    float grad_x = 0.0f;
    float grad_y = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = min(max(x + dx, 0), grid_size - 1);
            int ny = min(max(y + dy, 0), grid_size - 1);

            int neighbor_idx = (ny * grid_size + nx) * BEHAVIORAL_DIM + dim;
            float value = behavioral_field[neighbor_idx];

            grad_x += value * sobel_x[dy + 1][dx + 1];
            grad_y += value * sobel_y[dy + 1][dx + 1];
        }
    }

    // Normalize gradients
    float magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y) + 1e-8f;
    grad_x /= magnitude;
    grad_y /= magnitude;

    // Store gradients
    int grad_idx = ((y * grid_size + x) * BEHAVIORAL_DIM + dim) * 2;
    behavioral_gradients[grad_idx] = grad_x;
    behavioral_gradients[grad_idx + 1] = grad_y;
}

// Navigate using chemotactic strategy
__global__ void chemotactic_navigation_kernel(
    BehavioralState* __restrict__ agents,
    float* __restrict__ chemical_field,
    float* __restrict__ gradient_x,
    float* __restrict__ gradient_y,
    float* __restrict__ behavioral_gradients,
    int num_agents,
    int grid_size,
    float dt
) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id >= num_agents) return;

    BehavioralState* agent = &agents[agent_id];

    // Get current grid position
    int grid_x = (int)(agent->position[0] * grid_size);
    int grid_y = (int)(agent->position[1] * grid_size);
    grid_x = min(max(grid_x, 0), grid_size - 1);
    grid_y = min(max(grid_y, 0), grid_size - 1);

    int idx = grid_y * grid_size + grid_x;

    // Sample chemical gradient
    float chem_grad_x = gradient_x[idx];
    float chem_grad_y = gradient_y[idx];

    // Sample behavioral gradient (weighted by embedding)
    float behav_grad_x = 0.0f;
    float behav_grad_y = 0.0f;

    for (int d = 0; d < BEHAVIORAL_DIM; d++) {
        int grad_idx = ((grid_y * grid_size + grid_x) * BEHAVIORAL_DIM + d) * 2;
        float weight = agent->behavioral_coords[d];

        behav_grad_x += behavioral_gradients[grad_idx] * weight;
        behav_grad_y += behavioral_gradients[grad_idx + 1] * weight;
    }

    // Normalize behavioral gradient
    float behav_magnitude = sqrtf(behav_grad_x * behav_grad_x +
                                  behav_grad_y * behav_grad_y) + 1e-8f;
    behav_grad_x /= behav_magnitude;
    behav_grad_y /= behav_magnitude;

    // Store in gradient memory
    int mem_idx = agent->memory_index;
    agent->gradient_memory[mem_idx][0] = chem_grad_x * 0.5f + behav_grad_x * 0.5f;
    agent->gradient_memory[mem_idx][1] = chem_grad_y * 0.5f + behav_grad_y * 0.5f;
    agent->memory_index = (mem_idx + 1) % GRADIENT_HISTORY;

    // Compute temporal average of gradients
    float avg_grad_x = 0.0f;
    float avg_grad_y = 0.0f;
    float weight_sum = 0.0f;

    for (int i = 0; i < GRADIENT_HISTORY; i++) {
        float age = (float)(GRADIENT_HISTORY - i) / GRADIENT_HISTORY;
        float weight = expf(-2.0f * (1.0f - age));  // Exponential weighting

        avg_grad_x += agent->gradient_memory[i][0] * weight;
        avg_grad_y += agent->gradient_memory[i][1] * weight;
        weight_sum += weight;
    }

    avg_grad_x /= weight_sum;
    avg_grad_y /= weight_sum;

    // Biased random walk with Ornstein-Uhlenbeck noise
    float theta = 0.15f;  // Mean reversion strength
    float sigma = 0.2f;   // Noise strength

    // Generate correlated noise using agent ID as seed
    unsigned int seed = agent_id * 1337 + (unsigned int)(dt * 1000);
    seed = seed * 1664525u + 1013904223u;  // LCG
    float rand1 = (seed & 0xFFFFFF) / 16777216.0f;
    seed = seed * 1664525u + 1013904223u;
    float rand2 = (seed & 0xFFFFFF) / 16777216.0f;

    // Box-Muller transform for Gaussian noise
    float noise_x = sqrtf(-2.0f * logf(rand1 + 1e-10f)) * cosf(2.0f * 3.14159f * rand2);
    float noise_y = sqrtf(-2.0f * logf(rand1 + 1e-10f)) * sinf(2.0f * 3.14159f * rand2);

    // Update velocity with chemotactic bias and OU noise
    agent->velocity[0] += dt * (agent->sensitivity * avg_grad_x -
                                theta * agent->velocity[0] +
                                sigma * noise_x * agent->exploration_noise);

    agent->velocity[1] += dt * (agent->sensitivity * avg_grad_y -
                                theta * agent->velocity[1] +
                                sigma * noise_y * agent->exploration_noise);

    // Limit velocity magnitude
    float vel_magnitude = sqrtf(agent->velocity[0] * agent->velocity[0] +
                                agent->velocity[1] * agent->velocity[1]);
    if (vel_magnitude > 1.0f) {
        agent->velocity[0] /= vel_magnitude;
        agent->velocity[1] /= vel_magnitude;
    }

    // Update position
    agent->position[0] += agent->velocity[0] * dt;
    agent->position[1] += agent->velocity[1] * dt;

    // Toroidal boundary conditions
    agent->position[0] = agent->position[0] - floorf(agent->position[0]);
    agent->position[1] = agent->position[1] - floorf(agent->position[1]);

    // Adapt sensitivity based on gradient strength
    float gradient_strength = sqrtf(avg_grad_x * avg_grad_x + avg_grad_y * avg_grad_y);
    if (gradient_strength < SENSITIVITY_THRESHOLD) {
        // Increase exploration when gradient is weak
        agent->exploration_noise = fminf(1.0f, agent->exploration_noise * 1.1f);
        agent->sensitivity = fmaxf(0.1f, agent->sensitivity * 0.95f);
    } else {
        // Increase exploitation when gradient is strong
        agent->exploration_noise = fmaxf(0.1f, agent->exploration_noise * 0.9f);
        agent->sensitivity = fminf(2.0f, agent->sensitivity * 1.05f);
    }
}

// Create attractor sources in chemical field
__global__ void create_attractors_kernel(
    float* __restrict__ sources,
    float* __restrict__ attractor_positions,  // [NUM_ATTRACTORS][2]
    float* __restrict__ attractor_strengths,  // [NUM_ATTRACTORS]
    int grid_size,
    int num_attractors
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;
    float source_value = 0.0f;

    // Normalized coordinates
    float px = (float)x / grid_size;
    float py = (float)y / grid_size;

    // Sum contributions from all attractors
    for (int a = 0; a < num_attractors; a++) {
        float ax = attractor_positions[a * 2];
        float ay = attractor_positions[a * 2 + 1];
        float strength = attractor_strengths[a];

        // Gaussian attractor with toroidal distance
        float dx = fminf(fabsf(px - ax), 1.0f - fabsf(px - ax));
        float dy = fminf(fabsf(py - ay), 1.0f - fabsf(py - ay));
        float dist_sq = dx * dx + dy * dy;

        float sigma = 0.05f;  // Attractor size
        source_value += strength * expf(-dist_sq / (2.0f * sigma * sigma));
    }

    sources[idx] = source_value;
}

// Update DIRESA behavioral embedding based on navigation history
__global__ void update_behavioral_embedding_kernel(
    BehavioralState* __restrict__ agents,
    float* __restrict__ embedding_weights,  // [BEHAVIORAL_DIM][BEHAVIORAL_DIM]
    float* __restrict__ reconstruction_error,
    int num_agents,
    float learning_rate
) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id >= num_agents) return;

    BehavioralState* agent = &agents[agent_id];

    // Extract features from navigation history
    float features[BEHAVIORAL_DIM];

    // Feature 0: Average velocity magnitude
    features[0] = sqrtf(agent->velocity[0] * agent->velocity[0] +
                       agent->velocity[1] * agent->velocity[1]);

    // Feature 1: Turning rate (velocity change)
    float turn_rate = 0.0f;
    for (int i = 1; i < GRADIENT_HISTORY; i++) {
        float dx = agent->gradient_memory[i][0] - agent->gradient_memory[i-1][0];
        float dy = agent->gradient_memory[i][1] - agent->gradient_memory[i-1][1];
        turn_rate += sqrtf(dx * dx + dy * dy);
    }
    features[1] = turn_rate / GRADIENT_HISTORY;

    // Feature 2: Exploration tendency
    features[2] = agent->exploration_noise;

    // Feature 3: Sensitivity
    features[3] = agent->sensitivity;

    // Feature 4-9: Fourier components of trajectory
    for (int k = 0; k < 6; k++) {
        float freq = (k + 1) * 2.0f * 3.14159f / GRADIENT_HISTORY;
        float cos_sum = 0.0f;
        float sin_sum = 0.0f;

        for (int i = 0; i < GRADIENT_HISTORY; i++) {
            cos_sum += agent->gradient_memory[i][0] * cosf(freq * i);
            sin_sum += agent->gradient_memory[i][1] * sinf(freq * i);
        }

        features[4 + k] = sqrtf(cos_sum * cos_sum + sin_sum * sin_sum) / GRADIENT_HISTORY;
    }

    // Update embedding using online PCA
    float reconstructed[BEHAVIORAL_DIM] = {0};

    // Forward pass: project to embedding
    for (int i = 0; i < BEHAVIORAL_DIM; i++) {
        agent->behavioral_coords[i] = 0.0f;
        for (int j = 0; j < BEHAVIORAL_DIM; j++) {
            agent->behavioral_coords[i] += features[j] *
                                          embedding_weights[j * BEHAVIORAL_DIM + i];
        }
        agent->behavioral_coords[i] = tanhf(agent->behavioral_coords[i]);
    }

    // Backward pass: reconstruct features
    float error = 0.0f;
    for (int i = 0; i < BEHAVIORAL_DIM; i++) {
        reconstructed[i] = 0.0f;
        for (int j = 0; j < BEHAVIORAL_DIM; j++) {
            reconstructed[i] += agent->behavioral_coords[j] *
                               embedding_weights[i * BEHAVIORAL_DIM + j];
        }
        error += (features[i] - reconstructed[i]) * (features[i] - reconstructed[i]);
    }

    // Update weights using gradient descent
    for (int i = 0; i < BEHAVIORAL_DIM; i++) {
        for (int j = 0; j < BEHAVIORAL_DIM; j++) {
            float gradient = 2.0f * (features[i] - reconstructed[i]) *
                            agent->behavioral_coords[j];
            atomicAdd(&embedding_weights[i * BEHAVIORAL_DIM + j],
                     learning_rate * gradient);
        }
    }

    // Store reconstruction error
    if (agent_id < BEHAVIORAL_DIM) {
        reconstruction_error[agent_id] = error;
    }
}

// Initialize behavioral state
__global__ void init_behavioral_state_kernel(
    BehavioralState* agents,
    int num_agents,
    unsigned int seed
) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id >= num_agents) return;

    // Initialize PRNG
    curandState_t state;
    curand_init(seed, agent_id, 0, &state);

    BehavioralState* agent = &agents[agent_id];

    // Random initial position
    agent->position[0] = curand_uniform(&state);
    agent->position[1] = curand_uniform(&state);

    // Zero initial velocity
    agent->velocity[0] = 0.0f;
    agent->velocity[1] = 0.0f;

    // Random behavioral coordinates
    for (int i = 0; i < BEHAVIORAL_DIM; i++) {
        agent->behavioral_coords[i] = curand_normal(&state) * 0.1f;
    }

    // Clear gradient memory
    for (int i = 0; i < GRADIENT_HISTORY; i++) {
        agent->gradient_memory[i][0] = 0.0f;
        agent->gradient_memory[i][1] = 0.0f;
    }

    // Initial parameters
    agent->exploration_noise = 0.5f;
    agent->sensitivity = 1.0f;
    agent->memory_index = 0;
}

#endif // CHEMOTAXIS_CU

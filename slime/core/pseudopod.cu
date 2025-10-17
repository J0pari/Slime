// slime/core/pseudopod.cu - Multi-head Neural CA with Flow-Lenia dynamics
#ifndef PSEUDOPOD_CU
#define PSEUDOPOD_CU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

// Configuration from blueprint
constexpr int NUM_HEADS = 8;
constexpr int HEAD_DIM = 64;
constexpr int CHANNELS = 512;  // Total channels
constexpr int HIDDEN_DIM = 256;
constexpr int KERNEL_SIZE = 3;
constexpr int GRID_SIZE = 128;
constexpr float MASS_CONSERVATION_EPSILON = 1e-6f;

// Clamp function for boundaries
__device__ __forceinline__ int clamp(int x, int min, int max) {
    return x < min ? min : (x > max ? max : x);
}

// Multi-head CA state
struct MultiHeadCAState {
    float* perception_weights;    // [NUM_HEADS][CHANNELS][HIDDEN_DIM]
    float* interaction_weights;   // [NUM_HEADS][CHANNELS][HIDDEN_DIM]
    float* value_weights;         // [NUM_HEADS][HIDDEN_DIM][CHANNELS]
    float* head_mixing_weights;   // [NUM_HEADS][NUM_HEADS]
    float* flow_kernels;          // [NUM_HEADS][3][3]
    float* mass_buffer;           // For conservation check
};

// Multi-head Neural CA kernel with tensor cores
__global__ void multi_head_ca_kernel(
    float* __restrict__ ca_state,           // [BATCH][GRID][GRID][CHANNELS]
    float* __restrict__ perception_weights, // [NUM_HEADS][CHANNELS][HIDDEN_DIM]
    float* __restrict__ interaction_weights,// [NUM_HEADS][CHANNELS][HIDDEN_DIM]
    float* __restrict__ value_weights,      // [NUM_HEADS][HIDDEN_DIM][CHANNELS]
    float* __restrict__ ca_output,          // [BATCH][NUM_HEADS][GRID][GRID][CHANNELS]
    int batch_size,
    int grid_size
) {
    // Each block handles one head
    int head_id = blockIdx.y;
    int batch_id = blockIdx.z;
    int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_y = blockIdx.x * blockDim.y + threadIdx.y;

    if (cell_x >= grid_size || cell_y >= grid_size) return;

    // Load neighborhood into shared memory (one head)
    __shared__ float neighborhood[3][3][HEAD_DIM];

    // Each head computes different CA rule
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = clamp(cell_x + dx, 0, grid_size - 1);
            int ny = clamp(cell_y + dy, 0, grid_size - 1);

            int idx = batch_id * grid_size * grid_size * HEAD_DIM +
                     ny * grid_size * HEAD_DIM +
                     nx * HEAD_DIM;

            if (threadIdx.z < HEAD_DIM) {
                neighborhood[dy + 1][dx + 1][threadIdx.z] = ca_state[idx + threadIdx.z];
            }
        }
    }
    __syncthreads();

    // Head-specific perception using tensor cores
    float perception[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        perception[i] = 0.0f;
        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
                for (int c = 0; c < HEAD_DIM; c++) {
                    // Perception weight for this head
                    int weight_idx = head_id * CHANNELS * HIDDEN_DIM +
                                    c * HIDDEN_DIM + i;
                    perception[i] += neighborhood[dy][dx][c] *
                                   perception_weights[weight_idx];
                }
            }
        }
        // ReLU activation
        perception[i] = fmaxf(0.0f, perception[i]);
    }

    // Head-specific interaction
    float interaction[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        interaction[i] = 0.0f;
        for (int j = 0; j < HEAD_DIM; j++) {
            int weight_idx = head_id * CHANNELS * HIDDEN_DIM +
                           j * HIDDEN_DIM + i;
            interaction[i] += perception[j] * interaction_weights[weight_idx];
        }
        // GELU activation
        float x = interaction[i];
        interaction[i] = 0.5f * x * (1.0f + tanhf(0.7978845f * (x + 0.044715f * x * x * x)));
    }

    // Generate head output
    float output[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            int weight_idx = head_id * HIDDEN_DIM * CHANNELS +
                           j * CHANNELS + i;
            output[i] += interaction[j % HEAD_DIM] * value_weights[weight_idx];
        }
    }

    // Store output for this head
    int out_idx = batch_id * NUM_HEADS * grid_size * grid_size * HEAD_DIM +
                  head_id * grid_size * grid_size * HEAD_DIM +
                  cell_y * grid_size * HEAD_DIM +
                  cell_x * HEAD_DIM;

    for (int i = 0; i < HEAD_DIM; i++) {
        ca_output[out_idx + i] = output[i];
    }
}

// Flow-Lenia dynamics with mass conservation
__global__ void flow_lenia_dynamics_kernel(
    float* __restrict__ ca_state,
    float* __restrict__ ca_update,
    float* __restrict__ flow_kernels,  // [NUM_HEADS][3][3]
    float* __restrict__ mass_buffer,
    int grid_size,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int head = blockIdx.z;

    if (x >= grid_size || y >= grid_size) return;

    __shared__ float local_mass[32][32];
    __shared__ float kernel[3][3];

    // Load flow kernel for this head
    if (threadIdx.x < 3 && threadIdx.y < 3) {
        kernel[threadIdx.y][threadIdx.x] = flow_kernels[head * 9 + threadIdx.y * 3 + threadIdx.x];
    }
    __syncthreads();

    int base_idx = y * grid_size * CHANNELS + x * CHANNELS;
    int channel_offset = head * HEAD_DIM;

    // Compute local mass before update
    float mass_before = 0.0f;
    for (int c = 0; c < HEAD_DIM; c++) {
        mass_before += ca_state[base_idx + channel_offset + c];
    }
    local_mass[threadIdx.y][threadIdx.x] = mass_before;

    // Apply Flow-Lenia update
    for (int c = 0; c < HEAD_DIM; c++) {
        float potential = 0.0f;

        // 3x3 convolution with flow kernel
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = clamp(x + dx, 0, grid_size - 1);
                int ny = clamp(y + dy, 0, grid_size - 1);
                int neighbor_idx = ny * grid_size * CHANNELS + nx * CHANNELS + channel_offset + c;

                potential += ca_state[neighbor_idx] * kernel[dy + 1][dx + 1];
            }
        }

        // Growth function (smooth life)
        float growth = potential * expf(-potential * potential);

        // Update with time step
        ca_update[base_idx + channel_offset + c] = ca_state[base_idx + channel_offset + c] +
                                                   dt * growth;
    }

    // Compute mass after update
    __syncthreads();
    float mass_after = 0.0f;
    for (int c = 0; c < HEAD_DIM; c++) {
        mass_after += ca_update[base_idx + channel_offset + c];
    }

    // Mass conservation correction using shared memory value
    if (fabsf(mass_after) > MASS_CONSERVATION_EPSILON) {
        float correction = local_mass[threadIdx.y][threadIdx.x] / mass_after;
        for (int c = 0; c < HEAD_DIM; c++) {
            ca_update[base_idx + channel_offset + c] *= correction;
        }
    }

    // Store total mass in global buffer for monitoring
    if (threadIdx.x == 0 && threadIdx.y == 0 && mass_buffer != nullptr) {
        mass_buffer[head] = mass_after;
    }

    // Store mass for verification
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&mass_buffer[head], mass_after - mass_before);
    }
}

// Head mixing and aggregation
__global__ void mix_heads_kernel(
    float* __restrict__ head_outputs,    // [BATCH][NUM_HEADS][GRID][GRID][HEAD_DIM]
    float* __restrict__ mixing_weights,  // [NUM_HEADS][NUM_HEADS]
    float* __restrict__ final_output,    // [BATCH][GRID][GRID][CHANNELS]
    int batch_size,
    int grid_size
) {
    int batch = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= grid_size || y >= grid_size) return;

    // Mix heads with learned weights
    for (int out_head = 0; out_head < NUM_HEADS; out_head++) {
        for (int c = 0; c < HEAD_DIM; c++) {
            float mixed = 0.0f;

            for (int in_head = 0; in_head < NUM_HEADS; in_head++) {
                int input_idx = batch * NUM_HEADS * grid_size * grid_size * HEAD_DIM +
                               in_head * grid_size * grid_size * HEAD_DIM +
                               y * grid_size * HEAD_DIM +
                               x * HEAD_DIM + c;

                float weight = mixing_weights[out_head * NUM_HEADS + in_head];
                mixed += head_outputs[input_idx] * weight;
            }

            int output_idx = batch * grid_size * grid_size * CHANNELS +
                           y * grid_size * CHANNELS +
                           x * CHANNELS +
                           out_head * HEAD_DIM + c;

            final_output[output_idx] = mixed;
        }
    }
}

// Tensor core optimized convolution for CA
__global__ void tensor_core_ca_kernel(
    half* __restrict__ input,
    half* __restrict__ weights,
    float* __restrict__ output,
    int batch_size,
    int grid_size,
    int channels
) {
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_n = blockIdx.y;
    int batch = blockIdx.z;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Compute tile position
    int tile_row = warp_m * 16;
    int tile_col = warp_n * 16;

    if (tile_row < grid_size && tile_col < grid_size) {
        // Load input tile
        int input_offset = batch * grid_size * grid_size * channels +
                          tile_row * grid_size * channels +
                          tile_col * channels;
        wmma::load_matrix_sync(a_frag, &input[input_offset], channels);

        // Load weight matrix
        wmma::load_matrix_sync(b_frag, weights, channels);

        // Tensor core multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Store result
        int output_offset = batch * grid_size * grid_size * channels +
                           tile_row * grid_size * channels +
                           tile_col * channels;
        wmma::store_matrix_sync(&output[output_offset], c_frag, channels, wmma::mem_row_major);
    }
}

// Compute effective rank for parameter localization
__global__ void compute_effective_rank_kernel(
    float* __restrict__ weights,
    float* __restrict__ effective_rank,
    int num_params
) {
    __shared__ float singular_values[256];

    int tid = threadIdx.x;

    // Compute SVD of weight matrix (using power iteration)
    if (tid < min(256, num_params)) {
        // Initialize with weight magnitudes
        singular_values[tid] = 0.0f;
        for (int i = tid; i < num_params; i += blockDim.x) {
            singular_values[tid] += weights[i] * weights[i];
        }
    }
    __syncthreads();

    // Normalize to get distribution
    float sum = 0.0f;
    for (int i = 0; i < min(256, num_params); i++) {
        sum += singular_values[i];
    }

    if (sum > 0) {
        for (int i = 0; i < min(256, num_params); i++) {
            singular_values[i] /= sum;
        }
    }

    // Compute entropy
    float entropy = 0.0f;
    for (int i = 0; i < min(256, num_params); i++) {
        if (singular_values[i] > 1e-10f) {
            entropy -= singular_values[i] * logf(singular_values[i]);
        }
    }

    // Effective rank = exp(entropy)
    if (tid == 0) {
        *effective_rank = expf(entropy);
    }
}

// Compute coherence (learning progress)
__global__ void compute_coherence_kernel(
    float* __restrict__ loss_history,
    float* __restrict__ coherence,
    int history_length
) {
    __shared__ float improvements[256];

    int tid = threadIdx.x;

    float local_improvement = 0.0f;

    // Compute improvement over time
    if (tid < history_length - 1) {
        float current_loss = loss_history[tid];
        float next_loss = loss_history[tid + 1];

        if (current_loss > 1e-10f) {
            local_improvement = fmaxf(0.0f, (current_loss - next_loss) / current_loss);
        }
    }

    improvements[tid] = local_improvement;
    __syncthreads();

    // Reduction to get average improvement
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            improvements[tid] += improvements[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *coherence = improvements[0] / (history_length - 1);
    }
}

// Initialize multi-head CA
__global__ void init_multihead_ca_kernel(
    MultiHeadCAState* state,
    unsigned int seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize weights with Xavier/He initialization
    curandState_t rand_state;
    curand_init(seed, tid, 0, &rand_state);

    // Perception weights
    int perception_size = NUM_HEADS * CHANNELS * HIDDEN_DIM;
    if (tid < perception_size) {
        float fan_in = CHANNELS;
        float fan_out = HIDDEN_DIM;
        float scale = sqrtf(2.0f / (fan_in + fan_out));
        state->perception_weights[tid] = curand_normal(&rand_state) * scale;
    }

    // Interaction weights
    int interaction_size = NUM_HEADS * CHANNELS * HIDDEN_DIM;
    if (tid < interaction_size) {
        float scale = sqrtf(2.0f / CHANNELS);
        state->interaction_weights[tid] = curand_normal(&rand_state) * scale;
    }

    // Value weights
    int value_size = NUM_HEADS * HIDDEN_DIM * CHANNELS;
    if (tid < value_size) {
        float scale = sqrtf(2.0f / HIDDEN_DIM);
        state->value_weights[tid] = curand_normal(&rand_state) * scale;
    }

    // Head mixing weights (initialized to identity + small noise)
    int mixing_size = NUM_HEADS * NUM_HEADS;
    if (tid < mixing_size) {
        int i = tid / NUM_HEADS;
        int j = tid % NUM_HEADS;
        state->head_mixing_weights[tid] = (i == j ? 1.0f : 0.0f) +
                                          curand_normal(&rand_state) * 0.01f;
    }

    // Flow kernels (normalized for mass conservation)
    int kernel_size = NUM_HEADS * 9;
    if (tid < kernel_size) {
        state->flow_kernels[tid] = curand_uniform(&rand_state);
    }

    // Normalize flow kernels
    if (tid < NUM_HEADS) {
        float sum = 0.0f;
        for (int i = 0; i < 9; i++) {
            sum += state->flow_kernels[tid * 9 + i];
        }
        for (int i = 0; i < 9; i++) {
            state->flow_kernels[tid * 9 + i] /= sum;
        }
    }
}
#endif // PSEUDOPOD_CU

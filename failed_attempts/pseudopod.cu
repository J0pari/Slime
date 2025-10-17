#include <cuda_runtime.h>
#include <cuda/std/atomic>
#include <nvcuda/wmma.hpp>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

constexpr int NUM_HEADS = 8;
constexpr int HEAD_DIM = 64;
constexpr int GRID_SIZE = 256;
constexpr int CHANNELS = 16;
constexpr int NEIGHBORHOOD_SIZE = 3;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

struct NeuralCAState {
    float* grid[NUM_HEADS];
    float* perception_weights[NUM_HEADS];
    float* interaction_weights[NUM_HEADS];
    float* value_weights[NUM_HEADS];
    float* growth_params[NUM_HEADS];
    float* attention_scores;
    float* mass_buffer;
    float* coherence_history;
    float* effective_rank_history;
    cuda::std::atomic<float> total_mass;
    cuda::std::atomic<float> total_variance;
    int timestep;
    int generation;
};

struct FlowLeniaParams {
    float mu[NUM_HEADS];
    float sigma[NUM_HEADS];
    float beta[NUM_HEADS];
    float dt;
    float diffusion_rate;
    float mass_conservation_strength;
    float growth_clip_min;
    float growth_clip_max;
};

__device__ float bell_curve_growth(float x, float mu, float sigma) {
    return expf(-powf(x - mu, 2.0f) / (2.0f * sigma * sigma));
}

__device__ float soft_clip(float x, float min_val, float max_val) {
    float alpha = 10.0f;
    float lower = min_val + logf(1.0f + expf(alpha * (x - min_val))) / alpha;
    float upper = max_val - logf(1.0f + expf(alpha * (max_val - x))) / alpha;
    return fminf(lower, upper);
}

__global__ void multi_head_perception_kernel(NeuralCAState* state, int head_id) {
    extern __shared__ float shared_mem[];

    float* tile = shared_mem;
    float* perception = shared_mem + (NEIGHBORHOOD_SIZE + 2) * (NEIGHBORHOOD_SIZE + 2) * CHANNELS;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int gx = bx * blockDim.x + tx;
    int gy = by * blockDim.y + ty;
    int ch = tz;

    if (gx >= GRID_SIZE || gy >= GRID_SIZE || ch >= CHANNELS) return;

    int tile_width = blockDim.x + 2;
    int tile_height = blockDim.y + 2;

    for (int dy = -1; dy <= blockDim.y; dy++) {
        for (int dx = -1; dx <= blockDim.x; dx++) {
            if ((tx == 0 && dx == -1) || (tx == blockDim.x - 1 && dx == 1) ||
                (ty == 0 && dy == -1) || (ty == blockDim.y - 1 && dy == 1)) {

                int fetch_x = clamp(gx + dx, 0, GRID_SIZE - 1);
                int fetch_y = clamp(gy + dy, 0, GRID_SIZE - 1);
                int tile_idx = (ty + dy + 1) * tile_width * CHANNELS +
                              (tx + dx + 1) * CHANNELS + ch;
                int grid_idx = fetch_y * GRID_SIZE * CHANNELS + fetch_x * CHANNELS + ch;

                tile[tile_idx] = state->grid[head_id][grid_idx];
            }
        }
    }

    if (tx < blockDim.x && ty < blockDim.y) {
        int center_idx = (ty + 1) * tile_width * CHANNELS + (tx + 1) * CHANNELS + ch;
        int grid_idx = gy * GRID_SIZE * CHANNELS + gx * CHANNELS + ch;
        tile[center_idx] = state->grid[head_id][grid_idx];
    }
    __syncthreads();

    float perceived = 0.0f;

    #pragma unroll
    for (int ky = -1; ky <= 1; ky++) {
        #pragma unroll
        for (int kx = -1; kx <= 1; kx++) {
            #pragma unroll
            for (int c = 0; c < CHANNELS; c++) {
                int tile_idx = (ty + ky + 1) * tile_width * CHANNELS +
                              (tx + kx + 1) * CHANNELS + c;

                int weight_idx = ((ky + 1) * 3 + (kx + 1)) * CHANNELS * HEAD_DIM +
                                c * HEAD_DIM + (ch * HEAD_DIM / CHANNELS);

                perceived += tile[tile_idx] * state->perception_weights[head_id][weight_idx];
            }
        }
    }

    int perception_idx = gy * GRID_SIZE * HEAD_DIM + gx * HEAD_DIM + (ch * HEAD_DIM / CHANNELS);
    perception[perception_idx] = tanhf(perceived);
}

__global__ void multi_head_interaction_kernel(NeuralCAState* state, int head_id) {
    extern __shared__ float shared_mem[];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y) / 32;

    __shared__ half a_shared[WMMA_M][WMMA_K];
    __shared__ half b_shared[WMMA_K][WMMA_N];
    __shared__ float c_shared[WMMA_M][WMMA_N];

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_tile = 0; k_tile < HEAD_DIM / WMMA_K; k_tile++) {
        int row_offset = warp_m * WMMA_M;
        int col_offset = k_tile * WMMA_K;

        for (int i = threadIdx.x; i < WMMA_M * WMMA_K; i += blockDim.x) {
            int local_row = i / WMMA_K;
            int local_col = i % WMMA_K;
            int global_row = row_offset + local_row;
            int global_col = col_offset + local_col;

            if (global_row < GRID_SIZE && global_col < HEAD_DIM) {
                float val = state->grid[head_id][global_row * HEAD_DIM + global_col];
                a_shared[local_row][local_col] = __float2half(val);
            } else {
                a_shared[local_row][local_col] = __float2half(0.0f);
            }
        }

        for (int i = threadIdx.x; i < WMMA_K * WMMA_N; i += blockDim.x) {
            int local_row = i / WMMA_N;
            int local_col = i % WMMA_N;
            int weight_idx = (k_tile * WMMA_K + local_row) * HEAD_DIM + (warp_n * WMMA_N + local_col);

            if (weight_idx < HEAD_DIM * HEAD_DIM) {
                float val = state->interaction_weights[head_id][weight_idx];
                b_shared[local_row][local_col] = __float2half(val);
            } else {
                b_shared[local_row][local_col] = __float2half(0.0f);
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, (half*)a_shared, WMMA_K);
        wmma::load_matrix_sync(b_frag, (half*)b_shared, WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync((float*)c_shared, c_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    int gx = blockIdx.x * WMMA_M + threadIdx.x % WMMA_M;
    int gy = blockIdx.y * WMMA_N + threadIdx.x / WMMA_M;

    if (gx < GRID_SIZE && gy < HEAD_DIM && threadIdx.x < WMMA_M * WMMA_N) {
        int out_idx = gx * HEAD_DIM + gy;
        state->grid[head_id][out_idx] = tanhf(c_shared[threadIdx.x % WMMA_M][threadIdx.x / WMMA_M]);
    }
}

__global__ void compute_attention_scores_kernel(NeuralCAState* state, int head_id) {
    extern __shared__ float shared_scores[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int gx = bx * blockDim.x + tx;
    int gy = by * blockDim.y + ty;

    if (gx >= GRID_SIZE || gy >= GRID_SIZE) return;

    float query[HEAD_DIM / NUM_HEADS];
    float key[HEAD_DIM / NUM_HEADS];

    int head_offset = head_id * (HEAD_DIM / NUM_HEADS);

    #pragma unroll
    for (int i = 0; i < HEAD_DIM / NUM_HEADS; i++) {
        query[i] = state->grid[head_id][gx * HEAD_DIM + head_offset + i];
    }

    __shared__ float local_attention[9];

    if (tx < 3 && ty < 3) {
        int nx = clamp(gx + tx - 1, 0, GRID_SIZE - 1);
        int ny = clamp(gy + ty - 1, 0, GRID_SIZE - 1);

        #pragma unroll
        for (int i = 0; i < HEAD_DIM / NUM_HEADS; i++) {
            key[i] = state->grid[head_id][ny * GRID_SIZE * HEAD_DIM + nx * HEAD_DIM + head_offset + i];
        }

        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM / NUM_HEADS; i++) {
            score += query[i] * key[i];
        }

        score /= sqrtf((float)(HEAD_DIM / NUM_HEADS));
        local_attention[ty * 3 + tx] = score;
    }
    __syncthreads();

    if (tx == 0 && ty == 0) {
        float max_score = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            max_score = fmaxf(max_score, local_attention[i]);
        }

        float sum_exp = 0.0f;
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            local_attention[i] = expf(local_attention[i] - max_score);
            sum_exp += local_attention[i];
        }

        #pragma unroll
        for (int i = 0; i < 9; i++) {
            local_attention[i] /= sum_exp;
        }

        int attention_idx = head_id * GRID_SIZE * GRID_SIZE * 9 +
                           gy * GRID_SIZE * 9 + gx * 9;

        for (int i = 0; i < 9; i++) {
            state->attention_scores[attention_idx + i] = local_attention[i];
        }
    }
}

__global__ void flow_lenia_update_kernel(NeuralCAState* state, FlowLeniaParams* params, int head_id) {
    extern __shared__ float shared_mem[];

    float* neighborhood = shared_mem;
    float* growth_field = shared_mem + (blockDim.x + 2) * (blockDim.y + 2) * CHANNELS;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;
    int ch = tz;

    if (gx >= GRID_SIZE || gy >= GRID_SIZE || ch >= CHANNELS) return;

    int idx = gy * GRID_SIZE * CHANNELS + gx * CHANNELS + ch;
    float current_val = state->grid[head_id][idx];

    float mass_before = current_val;
    atomicAdd(&state->total_mass, mass_before);

    int tile_width = blockDim.x + 2;
    int center_idx = (ty + 1) * tile_width * CHANNELS + (tx + 1) * CHANNELS + ch;
    neighborhood[center_idx] = current_val;

    __syncthreads();

    float weighted_sum = 0.0f;
    float total_attention = 0.0f;

    int attention_base = head_id * GRID_SIZE * GRID_SIZE * 9 +
                        gy * GRID_SIZE * 9 + gx * 9;

    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            int neighbor_idx = (ty + dy + 1) * tile_width * CHANNELS +
                             (tx + dx + 1) * CHANNELS + ch;
            int attention_idx = (dy + 1) * 3 + (dx + 1);

            float neighbor_val = neighborhood[neighbor_idx];
            float attention = state->attention_scores[attention_base + attention_idx];

            weighted_sum += neighbor_val * attention;
            total_attention += attention;
        }
    }

    weighted_sum /= (total_attention + 1e-10f);

    float growth = bell_curve_growth(weighted_sum, params->mu[head_id], params->sigma[head_id]);
    growth = soft_clip(growth, params->growth_clip_min, params->growth_clip_max);

    float laplacian = 0.0f;
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            float weight = (dy == 0 && dx == 0) ? -4.0f : ((dy == 0 || dx == 0) ? 1.0f : 0.0f);
            int neighbor_idx = (ty + dy + 1) * tile_width * CHANNELS +
                             (tx + dx + 1) * CHANNELS + ch;
            laplacian += neighborhood[neighbor_idx] * weight;
        }
    }

    float diffusion = params->diffusion_rate * laplacian;

    float new_val = current_val + params->dt * (growth * current_val + diffusion);

    float local_mass_after = new_val;
    float mass_error = local_mass_after - mass_before;

    __shared__ float total_mass_error;
    if (tx == 0 && ty == 0 && tz == 0) {
        total_mass_error = 0.0f;
    }
    __syncthreads();

    atomicAdd(&total_mass_error, mass_error);
    __syncthreads();

    float correction = -total_mass_error / (blockDim.x * blockDim.y * blockDim.z);
    new_val += params->mass_conservation_strength * correction;

    new_val = fmaxf(0.0f, new_val);

    state->grid[head_id][idx] = new_val;

    atomicAdd(&state->mass_buffer[head_id], new_val - mass_before);

    float variance_contrib = (new_val - weighted_sum) * (new_val - weighted_sum);
    atomicAdd(&state->total_variance, variance_contrib);
}

__global__ void compute_effective_rank_kernel(NeuralCAState* state, float* output) {
    cg::grid_group grid = cg::this_grid();

    extern __shared__ float svd_workspace[];
    float* correlation_matrix = svd_workspace;
    float* singular_values = svd_workspace + HEAD_DIM * HEAD_DIM;

    int tid = grid.thread_rank();

    if (tid < HEAD_DIM * HEAD_DIM) {
        int i = tid / HEAD_DIM;
        int j = tid % HEAD_DIM;

        float sum = 0.0f;
        for (int k = 0; k < GRID_SIZE * GRID_SIZE; k++) {
            int idx1 = k * HEAD_DIM + i;
            int idx2 = k * HEAD_DIM + j;

            float sum_heads = 0.0f;
            #pragma unroll
            for (int h = 0; h < NUM_HEADS; h++) {
                sum_heads += state->grid[h][idx1] * state->grid[h][idx2];
            }
            sum += sum_heads / NUM_HEADS;
        }

        correlation_matrix[tid] = sum / (GRID_SIZE * GRID_SIZE);
    }
    __syncthreads();

    if (tid == 0) {
        const int MAX_JACOBI_SWEEPS = 30;
        const float TOLERANCE = 1e-6f;

        for (int sweep = 0; sweep < MAX_JACOBI_SWEEPS; sweep++) {
            float off_diagonal_sum = 0.0f;

            for (int p = 0; p < HEAD_DIM - 1; p++) {
                for (int q = p + 1; q < HEAD_DIM; q++) {
                    float app = correlation_matrix[p * HEAD_DIM + p];
                    float aqq = correlation_matrix[q * HEAD_DIM + q];
                    float apq = correlation_matrix[p * HEAD_DIM + q];

                    if (fabsf(apq) > TOLERANCE) {
                        float tau = (aqq - app) / (2.0f * apq);
                        float t = (tau >= 0) ? 1.0f / (tau + sqrtf(1.0f + tau * tau))
                                             : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                        float c = 1.0f / sqrtf(1.0f + t * t);
                        float s = t * c;

                        for (int i = 0; i < HEAD_DIM; i++) {
                            float aip = correlation_matrix[i * HEAD_DIM + p];
                            float aiq = correlation_matrix[i * HEAD_DIM + q];
                            correlation_matrix[i * HEAD_DIM + p] = c * aip - s * aiq;
                            correlation_matrix[i * HEAD_DIM + q] = s * aip + c * aiq;
                        }

                        for (int j = 0; j < HEAD_DIM; j++) {
                            float apj = correlation_matrix[p * HEAD_DIM + j];
                            float aqj = correlation_matrix[q * HEAD_DIM + j];
                            correlation_matrix[p * HEAD_DIM + j] = c * apj - s * aqj;
                            correlation_matrix[q * HEAD_DIM + j] = s * apj + c * aqj;
                        }

                        correlation_matrix[p * HEAD_DIM + p] = c * c * app + s * s * aqq - 2.0f * c * s * apq;
                        correlation_matrix[q * HEAD_DIM + q] = s * s * app + c * c * aqq + 2.0f * c * s * apq;
                        correlation_matrix[p * HEAD_DIM + q] = 0.0f;
                        correlation_matrix[q * HEAD_DIM + p] = 0.0f;
                    }

                    off_diagonal_sum += fabsf(apq);
                }
            }

            if (off_diagonal_sum < TOLERANCE * HEAD_DIM) break;
        }

        for (int i = 0; i < HEAD_DIM; i++) {
            singular_values[i] = sqrtf(fabsf(correlation_matrix[i * HEAD_DIM + i]));
        }

        for (int i = 0; i < HEAD_DIM - 1; i++) {
            for (int j = i + 1; j < HEAD_DIM; j++) {
                if (singular_values[i] < singular_values[j]) {
                    float tmp = singular_values[i];
                    singular_values[i] = singular_values[j];
                    singular_values[j] = tmp;
                }
            }
        }

        float sum_normalized = 0.0f;
        float sum_sq = 0.0f;

        for (int i = 0; i < HEAD_DIM; i++) {
            sum_normalized += singular_values[i];
            sum_sq += singular_values[i] * singular_values[i];
        }

        float effective_rank = (sum_normalized * sum_normalized) / (sum_sq + 1e-10f);
        *output = effective_rank;

        state->effective_rank_history[state->timestep % 1000] = effective_rank;
    }
}

__global__ void compute_coherence_kernel(NeuralCAState* state, float* output) {
    __shared__ float error_history[1000];
    __shared__ float regression_workspace[4];

    int tid = threadIdx.x;
    int history_len = min(state->timestep, 1000);

    if (tid < history_len) {
        int idx = (state->timestep - history_len + tid) % 1000;
        error_history[tid] = state->coherence_history[idx];
    }
    __syncthreads();

    if (tid == 0) {
        float sum_x = 0.0f, sum_y = 0.0f, sum_xx = 0.0f, sum_xy = 0.0f;

        for (int i = 0; i < history_len; i++) {
            float x = (float)i;
            float y = error_history[i];
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        float n = (float)history_len;
        float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x + 1e-10f);

        float learning_progress = -slope;

        float coherence = 1.0f / (1.0f + expf(-learning_progress * 10.0f));

        *output = coherence;

        state->coherence_history[state->timestep % 1000] = coherence;
    }
}

__global__ void compute_hunger_kernel(NeuralCAState* state, float coherence, float* hunger_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float hunger = 1.0f - coherence;

        float variance = state->total_variance.load() / (GRID_SIZE * GRID_SIZE * CHANNELS * NUM_HEADS);
        float entropy_bonus = logf(1.0f + variance) / 10.0f;

        hunger = fminf(1.0f, hunger + entropy_bonus);

        *hunger_out = hunger;
    }
}

__global__ void spawn_child_ca_kernels(NeuralCAState* state, FlowLeniaParams* params) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dim3 perception_grid(GRID_SIZE / 16, GRID_SIZE / 16);
        dim3 perception_block(16, 16, CHANNELS);

        for (int h = 0; h < NUM_HEADS; h++) {
            multi_head_perception_kernel<<<perception_grid, perception_block,
                (18 * 18 * CHANNELS + 16 * 16 * HEAD_DIM) * sizeof(float)>>>(state, h);
        }
        cudaDeviceSynchronize();

        dim3 interaction_grid(GRID_SIZE / WMMA_M, HEAD_DIM / WMMA_N);
        dim3 interaction_block(32, 8);

        for (int h = 0; h < NUM_HEADS; h++) {
            multi_head_interaction_kernel<<<interaction_grid, interaction_block>>>(state, h);
        }
        cudaDeviceSynchronize();

        dim3 attention_grid(GRID_SIZE / 8, GRID_SIZE / 8);
        dim3 attention_block(8, 8);

        for (int h = 0; h < NUM_HEADS; h++) {
            compute_attention_scores_kernel<<<attention_grid, attention_block,
                9 * sizeof(float)>>>(state, h);
        }
        cudaDeviceSynchronize();

        dim3 flow_grid(GRID_SIZE / 8, GRID_SIZE / 8);
        dim3 flow_block(8, 8, CHANNELS);

        for (int h = 0; h < NUM_HEADS; h++) {
            flow_lenia_update_kernel<<<flow_grid, flow_block,
                ((10 * 10 * CHANNELS) + (8 * 8 * CHANNELS)) * sizeof(float)>>>(state, params, h);
        }
        cudaDeviceSynchronize();

        state->timestep++;
    }
}

__global__ void initialize_pseudopod_kernel(NeuralCAState* state, FlowLeniaParams* params,
                                           curandState* rng_states, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, tid, 0, &rng_states[tid]);

    if (tid < GRID_SIZE * GRID_SIZE * CHANNELS * NUM_HEADS) {
        int head = tid / (GRID_SIZE * GRID_SIZE * CHANNELS);
        int remainder = tid % (GRID_SIZE * GRID_SIZE * CHANNELS);

        float val = curand_normal(&rng_states[tid]) * 0.1f;
        state->grid[head][remainder] = val;

        if (tid == 0) {
            state->total_mass.store(0.0f);
            state->total_variance.store(0.0f);
            state->timestep = 0;
            state->generation = 0;
        }
    }

    if (tid < NUM_HEADS) {
        params->mu[tid] = 0.3f + curand_uniform(&rng_states[tid]) * 0.4f;
        params->sigma[tid] = 0.1f + curand_uniform(&rng_states[tid]) * 0.2f;
        params->beta[tid] = 0.01f + curand_uniform(&rng_states[tid]) * 0.02f;

        if (tid == 0) {
            params->dt = 0.01f;
            params->diffusion_rate = 0.1f;
            params->mass_conservation_strength = 0.95f;
            params->growth_clip_min = -2.0f;
            params->growth_clip_max = 2.0f;
        }
    }

    if (tid < HEAD_DIM * HEAD_DIM * NUM_HEADS) {
        int head = tid / (HEAD_DIM * HEAD_DIM);
        int idx = tid % (HEAD_DIM * HEAD_DIM);

        state->perception_weights[head][idx] = curand_normal(&rng_states[tid]) * 0.1f;
        state->interaction_weights[head][idx] = curand_normal(&rng_states[tid]) * 0.1f;
        state->value_weights[head][idx] = curand_normal(&rng_states[tid]) * 0.1f;
    }
}

__global__ void mutate_pseudopod_kernel(NeuralCAState* parent, NeuralCAState* offspring,
                                       float mutation_strength, curandState* rng_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < HEAD_DIM * HEAD_DIM * NUM_HEADS) {
        int head = tid / (HEAD_DIM * HEAD_DIM);
        int idx = tid % (HEAD_DIM * HEAD_DIM);

        float parent_weight = parent->perception_weights[head][idx];
        float mutation = curand_normal(&rng_states[tid]) * mutation_strength;

        offspring->perception_weights[head][idx] = parent_weight + mutation;

        parent_weight = parent->interaction_weights[head][idx];
        mutation = curand_normal(&rng_states[tid]) * mutation_strength;
        offspring->interaction_weights[head][idx] = parent_weight + mutation;

        parent_weight = parent->value_weights[head][idx];
        mutation = curand_normal(&rng_states[tid]) * mutation_strength;
        offspring->value_weights[head][idx] = parent_weight + mutation;
    }

    if (tid == 0) {
        offspring->generation = parent->generation + 1;
        offspring->timestep = 0;
        offspring->total_mass.store(parent->total_mass.load());
        offspring->total_variance.store(0.0f);
    }
}

__global__ void crossover_pseudopods_kernel(NeuralCAState* parent1, NeuralCAState* parent2,
                                           NeuralCAState* offspring, curandState* rng_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < NUM_HEADS) {
        float crossover_point = curand_uniform(&rng_states[tid]);

        for (int i = 0; i < HEAD_DIM * HEAD_DIM; i++) {
            float rand = curand_uniform(&rng_states[tid * HEAD_DIM * HEAD_DIM + i]);

            if (rand < crossover_point) {
                offspring->perception_weights[tid][i] = parent1->perception_weights[tid][i];
                offspring->interaction_weights[tid][i] = parent1->interaction_weights[tid][i];
                offspring->value_weights[tid][i] = parent1->value_weights[tid][i];
            } else {
                offspring->perception_weights[tid][i] = parent2->perception_weights[tid][i];
                offspring->interaction_weights[tid][i] = parent2->interaction_weights[tid][i];
                offspring->value_weights[tid][i] = parent2->value_weights[tid][i];
            }
        }
    }

    if (tid == 0) {
        offspring->generation = max(parent1->generation, parent2->generation) + 1;
        offspring->timestep = 0;
        offspring->total_mass.store(0.0f);
        offspring->total_variance.store(0.0f);
    }
}
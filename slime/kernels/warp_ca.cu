
#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>

struct Organism;

namespace cg = cooperative_groups;
using namespace nvcuda::wmma;

__device__ void jacobi_rotation(float* A, int n, int p, int q, float* s, float* c) {
    float app = A[p * n + p];
    float aqq = A[q * n + q];
    float apq = A[p * n + q];

    if (fabsf(apq) <= 0.0f) {
        *s = 0.0f;
        *c = 1.0f;
        return;
    }
    float tau = (aqq - app) / (2.0f * apq);
    float t = (tau >= 0.0f) ?
        1.0f / (tau + sqrtf(1.0f + tau * tau)) :
        -1.0f / (-tau + sqrtf(1.0f + tau * tau));

    *c = 1.0f / sqrtf(1.0f + t * t);
    *s = t * (*c);
}

__global__ void gpu_svd_kernel(
    float* __restrict__ A,
    float* __restrict__ U,
    float* __restrict__ S,
    float* __restrict__ V,
    int m, int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                U[i * n + j] = (i == j) ? 1.0f : 0.0f;
                V[i * n + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    __syncthreads();

    int max_sweeps = min(MAX_JACOBI_SWEEPS, 10);
    int tile_dim = min(n, WARP_SIZE);

    for (int tile_row = 0; tile_row < (n + tile_dim - 1) / tile_dim; tile_row++) {
        for (int tile_col = 0; tile_col < (n + tile_dim - 1) / tile_dim; tile_col++) {
            __shared__ float shared_A[WARP_SIZE][WARP_SIZE + BANK_PAD];
            __shared__ float shared_U[WARP_SIZE][WARP_SIZE + BANK_PAD];
            __shared__ float shared_V[WARP_SIZE][WARP_SIZE + BANK_PAD];

            int base_row = tile_row * tile_dim;
            int base_col = tile_col * tile_dim;

            if (tid < tile_dim * tile_dim) {
                int local_row = tid / tile_dim;
                int local_col = tid % tile_dim;
                int global_row = base_row + local_row;
                int global_col = base_col + local_col;

                if (global_row < n && global_col < n) {
                    shared_A[local_row][local_col] = A[global_row * n + global_col];
                    shared_U[local_row][local_col] = U[global_row * n + global_col];
                    shared_V[local_row][local_col] = V[global_row * n + global_col];
                } else {
                    shared_A[local_row][local_col] = 0.0f;
                    shared_U[local_row][local_col] = 0.0f;
                    shared_V[local_row][local_col] = 0.0f;
                }
            }
            __syncthreads();

            int effective_size = min(tile_dim, n - base_row);
            effective_size = min(effective_size, n - base_col);

            for (int sweep = 0; sweep < max_sweeps; sweep++) {
                for (int p = 0; p < effective_size - 1; p++) {
                    for (int q = p + 1; q < effective_size; q++) {
                        float c, s;
                        jacobi_rotation((float*)shared_A, effective_size, p, q, &s, &c);

                        if (threadIdx.x < effective_size) {
                            int i = threadIdx.x;
                            float aip = shared_A[i][p];
                            float aiq = shared_A[i][q];
                            shared_A[i][p] = c * aip - s * aiq;
                            shared_A[i][q] = s * aip + c * aiq;

                            float uip = shared_U[i][p];
                            float uiq = shared_U[i][q];
                            shared_U[i][p] = c * uip - s * uiq;
                            shared_U[i][q] = s * uip + c * uiq;

                            float vip = shared_V[p][i];
                            float viq = shared_V[q][i];
                            shared_V[p][i] = c * vip - s * viq;
                            shared_V[q][i] = s * vip + c * viq;
                        }
                        __syncthreads();
                    }
                }
            }

            if (tid < tile_dim * tile_dim) {
                int local_row = tid / tile_dim;
                int local_col = tid % tile_dim;
                int global_row = base_row + local_row;
                int global_col = base_col + local_col;

                if (global_row < n && global_col < n) {
                    U[global_row * n + global_col] = shared_U[local_row][local_col];
                    V[global_row * n + global_col] = shared_V[local_row][local_col];
                }
            }

            if (tid < effective_size) {
                int global_idx = base_row + tid;
                if (global_idx < n && tile_row == tile_col) {
                    float diag_val = shared_A[tid][tid];
                    // Negative diagonal: singular value is 0 (no early return - all threads must reach syncthreads)
                    S[global_idx] = (diag_val >= 0.0f) ? sqrtf(diag_val) : 0.0f;
                }
            }
            __syncthreads();
        }
    }
}

__global__ void effective_rank_from_latent_kernel(
    float* __restrict__ latent_genome,
    float* __restrict__ rank_out,
    int latent_dim
) {
    int tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int i = tid; i < latent_dim; i += blockDim.x) {
        local_sum += latent_genome[i];
    }
    float mean = BlockReduce<BLOCK_SIZE>::sum(local_sum) / latent_dim;
    mean = __shfl_sync(0xffffffff, mean, 0);

    float local_var = 0.0f;
    for (int i = tid; i < latent_dim; i += blockDim.x) {
        float diff = latent_genome[i] - mean;
        local_var += diff * diff;
    }
    float variance = BlockReduce<BLOCK_SIZE>::sum(local_var) / latent_dim;

    if (tid == 0 && variance >= 0.0f) {
        *rank_out = sqrtf(variance) * latent_dim;
    }
}

__global__ void coherence_kernel(
    float* __restrict__ prediction_errors,
    float* __restrict__ coherence_out,
    int history_length
) {
    DEVICE_FATAL_IF(prediction_errors == nullptr, "coherence_kernel: prediction_errors is null");
    DEVICE_FATAL_IF(coherence_out == nullptr, "coherence_kernel: coherence_out is null");
    DEVICE_FATAL_IF(history_length < 2, "coherence_kernel: history_length must be >= 2");

    __shared__ float learning_progress;

    if (threadIdx.x == 0) learning_progress = 0.0f;
    __syncthreads();

    float local_progress = 0.0f;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < history_length - 1) {
        float curr_error = prediction_errors[tid];
        float next_error = prediction_errors[tid + 1];
        // Valid data: compute progress; invalid data (curr_error <= 0): local_progress stays 0
        if (curr_error > 0.0f) {
            local_progress = fmaxf(0.0f, (curr_error - next_error) / curr_error);
        }
    }

    __syncthreads();
    local_progress = BlockReduce<BLOCK_SIZE>::sum(local_progress);

    if (threadIdx.x == 0) {
        learning_progress = local_progress / (history_length - 1);
        *coherence_out = learning_progress;
    }
}

__global__ void hunger_kernel(
    float* __restrict__ coherence_values,
    float* __restrict__ hunger_out,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {

        hunger_out[tid] = 1.0f - coherence_values[tid];
    }
}

__global__ void neural_ca_tensor_kernel(
    half* __restrict__ input,
    half* __restrict__ weights,
    half* __restrict__ output,
    int width, int height, int channels
) {

    wmma::fragment<wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fill_fragment(c_frag, 0.0f);

    int tile_row = warp_m * WMMA_TILE_DIM;
    int tile_col = warp_n * WMMA_TILE_DIM;

    if (tile_row < height && tile_col < width) {

        wmma::load_matrix_sync(a_frag,
            &input[tile_row * width + tile_col], width);

        wmma::load_matrix_sync(b_frag,
            &weights[0], CA_KERNEL_SIZE * CA_KERNEL_SIZE);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __shared__ float temp[WMMA_TILE_DIM][WMMA_TILE_DIM + BANK_PAD];
        wmma::store_matrix_sync(&temp[0][0], c_frag, WMMA_TILE_DIM, wmma::mem_row_major);

        __syncthreads();

        int lane = threadIdx.x % WARP_SIZE;
        if (lane < WMMA_TILE_DIM * WMMA_TILE_DIM) {
            int local_row = lane / WMMA_TILE_DIM;
            int local_col = lane % WMMA_TILE_DIM;
            int global_row = tile_row + local_row;
            int global_col = tile_col + local_col;
            if (global_row < height && global_col < width) {
                output[global_row * width + global_col] = __float2half(temp[local_row][local_col]);
            }
        }
    }
}

__global__ void flow_lenia_kernel(
    float* __restrict__ state,
    float* __restrict__ next_state,
    float* __restrict__ kernels,
    int width, int height,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float center = state[idx];

    float potential = 0.0f;
    float total_mass = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            int nidx = ny * width + nx;

            float neighbor = state[nidx];
            float kernel_val = kernels[(dy + 1) * 3 + (dx + 1)];

            potential += neighbor * kernel_val;
            total_mass += neighbor;
        }
    }

    float growth = potential * expf(-potential * potential);

    float next_val = center + dt * growth;

    float denom = CA_KERNEL_CELL_COUNT * next_val;
    if (denom <= 0.0f) {
        next_state[idx] = 0.0f;
        return;
    }
    float mass_ratio = total_mass / denom;
    next_state[idx] = next_val * mass_ratio;
}

__device__ float get_neighbor_2d(
    float val,
    int dx, int dy,
    int width, unsigned mask
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int x = lane_id % width;
    int y = lane_id / width;

    int nx = (x + dx + width) % width;
    int ny = (y + dy + WARP_SIZE / width) % (WARP_SIZE / width);

    int neighbor_lane = ny * width + nx;
    return __shfl_sync(mask, val, neighbor_lane);
}

__global__ void warp_ca_kernel(
    float* __restrict__ state,
    float* __restrict__ next_state,
    int width, int height,
    float* genome,
    float* gradients,
    uint64_t genome_hash,
    float ctx_metabolic,
    float ctx_stress,
    float ctx_morphogen,
    float ctx_complexity,
    float ctx_niche,
    float ctx_learning,
    float ctx_performance
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= (width * height) / WARP_SIZE) return;

    int tile_x = (warp_id * WARP_SIZE + lane_id) % width;
    int tile_y = (warp_id * WARP_SIZE + lane_id) / width;

    if (tile_x >= width || tile_y >= height) return;

    float my_state = state[tile_y * width + tile_x];
    unsigned mask = warp.ballot(1);

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

    CAParams ca_params;
    ca_params.derive_from_genome_hash(genome_hash);
    float warp_ca_growth_rate = ca_params.get_warp_ca_growth_rate(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);

    float total_mass = WarpReduce<WARP_SIZE>::sum(my_state);
    float new_val = my_state + warp_ca_growth_rate * growth;
    float new_total = WarpReduce<WARP_SIZE>::sum(new_val);

    if (is_meaningful(new_total, total_mass)) {
        new_val *= total_mass / new_total;
    }

    next_state[tile_y * width + tile_x] = new_val;
}
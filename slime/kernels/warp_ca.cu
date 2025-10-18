// slime/kernels/warp_ca.cu - Warp-level Neural CA with dynamic parallelism
// Implements FITNESS = effective_rank() × coherence()

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>

// Forward declaration of Organism struct
struct Organism;
constexpr int MAX_COMPONENTS = 256;
constexpr int GENOME_SIZE = 1024;

namespace cg = cooperative_groups;
using namespace nvcuda::wmma;

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int CA_KERNEL_SIZE = 3;
constexpr int MAX_JACOBI_SWEEPS = 30;
constexpr float EPSILON = 1e-7f;

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Block-level reduction
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

// Jacobi rotation for SVD
__device__ void jacobi_rotation(float* A, int n, int p, int q, float* s, float* c) {
    float app = A[p * n + p];
    float aqq = A[q * n + q];
    float apq = A[p * n + q];

    float tau = (aqq - app) / (2.0f * apq + EPSILON);
    float t = (tau >= 0.0f) ?
        1.0f / (tau + sqrtf(1.0f + tau * tau)) :
        -1.0f / (-tau + sqrtf(1.0f + tau * tau));

    *c = 1.0f / sqrtf(1.0f + t * t);
    *s = t * (*c);
}

// GPU-native Jacobi SVD kernel (child kernel for dynamic parallelism)
__global__ void gpu_svd_kernel(
    float* __restrict__ A,
    float* __restrict__ U,
    float* __restrict__ S,
    float* __restrict__ V,
    int m, int n
) {
    __shared__ float shared_A[32][32];

    // Copy to shared memory
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < m * n) {
        shared_A[tid / n][tid % n] = A[tid];
    }
    __syncthreads();

    // Jacobi sweeps
    for (int sweep = 0; sweep < MAX_JACOBI_SWEEPS; sweep++) {
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                float c, s;
                jacobi_rotation((float*)shared_A, n, p, q, &s, &c);

                // Apply rotation
                for (int i = 0; i < n; i++) {
                    float aip = shared_A[i][p];
                    float aiq = shared_A[i][q];
                    shared_A[i][p] = c * aip - s * aiq;
                    shared_A[i][q] = s * aip + c * aiq;
                }
            }
        }
    }

    // Extract singular values
    if (threadIdx.x < n) {
        S[threadIdx.x] = sqrtf(fabsf(shared_A[threadIdx.x][threadIdx.x]));
    }
}

// Effective rank computation kernel (child kernel)
__global__ void effective_rank_kernel(
    float* __restrict__ S,
    float* __restrict__ rank_out,
    int n
) {
    __shared__ float s_normalized[256];
    __shared__ float entropy;

    if (threadIdx.x == 0) entropy = 0.0f;
    __syncthreads();

    // Normalize singular values
    float sum = 0.0f;
    if (threadIdx.x < n) {
        sum = S[threadIdx.x];
    }

    // Block reduction for sum
    __syncthreads();
    sum = block_reduce_sum(sum);

    if (threadIdx.x < n) {
        s_normalized[threadIdx.x] = S[threadIdx.x] / (sum + EPSILON);
    }
    __syncthreads();

    // Compute entropy
    if (threadIdx.x < n) {
        float p = s_normalized[threadIdx.x];
        atomicAdd(&entropy, -p * logf(p + EPSILON));
    }
    __syncthreads();

    // Effective rank = exp(entropy)
    if (threadIdx.x == 0) {
        *rank_out = expf(entropy);
    }
}

// Coherence computation kernel (child kernel)
__global__ void coherence_kernel(
    float* __restrict__ prediction_errors,
    float* __restrict__ coherence_out,
    int history_length
) {
    __shared__ float learning_progress;

    if (threadIdx.x == 0) learning_progress = 0.0f;
    __syncthreads();

    // Compute learning progress from prediction error reduction
    float local_progress = 0.0f;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < history_length - 1) {
        float curr_error = prediction_errors[tid];
        float next_error = prediction_errors[tid + 1];
        local_progress = fmaxf(0.0f, (curr_error - next_error) / (curr_error + EPSILON));
    }

    // Reduction
    __syncthreads();
    local_progress = block_reduce_sum(local_progress);

    if (threadIdx.x == 0) {
        learning_progress = local_progress / (history_length - 1);
        *coherence_out = learning_progress;
    }
}

// Fused fitness kernel with dynamic parallelism (parent kernel)
__global__ void fitness_fused_kernel(
    Organism* organism,
    float* __restrict__ correlation_matrix,
    float* __restrict__ prediction_errors,
    float* __restrict__ fitness_out,
    int matrix_size,
    int history_length
) {
    // Use pre-allocated pools for intermediate results
    int tid = blockIdx.x;
    if (tid >= MAX_COMPONENTS) return;
    
    float* S = organism->fitness_svd_pool + tid * GENOME_SIZE;
    float* effective_rank_val = organism->fitness_rank_pool + tid;
    float* coherence_val = organism->fitness_coherence_pool + tid;

    if (threadIdx.x == 0 && blockIdx.x == 0) {

        // Launch child kernel for SVD
        gpu_svd_kernel<<<1, min(matrix_size, BLOCK_SIZE)>>>(
            correlation_matrix, nullptr, S, nullptr, matrix_size, matrix_size
        );

        // Launch child kernel for effective rank
        effective_rank_kernel<<<1, min(matrix_size, BLOCK_SIZE)>>>(
            S, effective_rank_val, matrix_size
        );

        // Launch child kernel for coherence
        coherence_kernel<<<1, min(history_length, BLOCK_SIZE)>>>(
            prediction_errors, coherence_val, history_length
        );

        // Synchronize child kernels
        cudaDeviceSynchronize();

        // Compute final fitness
        float effective_rank, coherence;
        cudaMemcpy(&effective_rank, effective_rank_val, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&coherence, coherence_val, sizeof(float), cudaMemcpyDeviceToDevice);

        *fitness_out = effective_rank * coherence;

        // Cleanup
        cudaFree(S);
        cudaFree(effective_rank_val);
        cudaFree(coherence_val);
    }
}

// Hunger computation kernel
__global__ void hunger_kernel(
    float* __restrict__ coherence_values,
    float* __restrict__ hunger_out,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        // hunger = 1.0 - coherence (curiosity-driven lifecycle)
        hunger_out[tid] = 1.0f - coherence_values[tid];
    }
}

// Tensor Core Neural CA convolution using WMMA
__global__ void neural_ca_tensor_kernel(
    half* __restrict__ input,
    half* __restrict__ weights,
    half* __restrict__ output,
    int width, int height, int channels
) {
    // Fragment declarations for WMMA
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Tile indices
    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y);

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Load neighborhood into matrix A (16x16 tile)
    int tile_row = warp_m * 16;
    int tile_col = warp_n * 16;

    if (tile_row < height && tile_col < width) {
        // Load input tile
        wmma::load_matrix_sync(a_frag,
            &input[tile_row * width + tile_col], width);

        // Load convolution weights
        wmma::load_matrix_sync(b_frag,
            &weights[0], CA_KERNEL_SIZE * CA_KERNEL_SIZE);

        // Tensor Core matrix multiply
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Store result
        wmma::store_matrix_sync(&output[tile_row * width + tile_col],
            c_frag, width, wmma::mem_row_major);
    }
}

// Flow-Lenia dynamics with mass conservation
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

    // 3x3 convolution for Flow-Lenia update
    float potential = 0.0f;
    float total_mass = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = (x + dx + width) % width;  // Toroidal boundary
            int ny = (y + dy + height) % height;
            int nidx = ny * width + nx;

            float neighbor = state[nidx];
            float kernel_val = kernels[(dy + 1) * 3 + (dx + 1)];

            potential += neighbor * kernel_val;
            total_mass += neighbor;
        }
    }

    // Growth function (smooth step)
    float growth = potential * expf(-potential * potential);

    // Update with mass conservation
    float next_val = center + dt * growth;

    // Normalize to conserve mass
    float mass_ratio = total_mass / (9.0f * next_val + EPSILON);
    next_state[idx] = next_val * mass_ratio;
}

// Warp-level CA using shuffle operations (no shared memory)
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
    int width, int height
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= (width * height) / WARP_SIZE) return;

    // Each warp processes a tile
    int tile_x = (warp_id * WARP_SIZE + lane_id) % width;
    int tile_y = (warp_id * WARP_SIZE + lane_id) / width;

    if (tile_x >= width || tile_y >= height) return;

    float my_state = state[tile_y * width + tile_x];
    unsigned mask = warp.ballot(1);

    // Compute CA update using warp shuffles
    float sum = 0.0f;

    // Get all 8 neighbors via shuffle
    sum += get_neighbor_2d(my_state, -1, -1, width, mask);
    sum += get_neighbor_2d(my_state, 0, -1, width, mask);
    sum += get_neighbor_2d(my_state, 1, -1, width, mask);
    sum += get_neighbor_2d(my_state, -1, 0, width, mask);
    sum += get_neighbor_2d(my_state, 1, 0, width, mask);
    sum += get_neighbor_2d(my_state, -1, 1, width, mask);
    sum += get_neighbor_2d(my_state, 0, 1, width, mask);
    sum += get_neighbor_2d(my_state, 1, 1, width, mask);

    // Apply CA rule (Lenia-style smooth life)
    float avg = sum / 8.0f;
    float growth = avg * expf(-avg * avg * 2.0f);

    // Mass conservation via warp reduction
    float total_mass = warp_reduce_sum(my_state);
    float new_val = my_state + 0.1f * growth;
    float new_total = warp_reduce_sum(new_val);

    // Normalize to conserve mass
    if (new_total > EPSILON) {
        new_val *= total_mass / new_total;
    }

    next_state[tile_y * width + tile_x] = new_val;
}
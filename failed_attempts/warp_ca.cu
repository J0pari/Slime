

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32
#define KERNEL_SIZE 3
#define BLOCK_DIM 16
#define MAX_JACOBI_ITERS 30
#define EPSILON 1e-7f


__device__ __forceinline__ float get_left_neighbor(float val, unsigned mask) {
    return __shfl_up_sync(mask, val, 1);
}

__device__ __forceinline__ float get_right_neighbor(float val, unsigned mask) {
    return __shfl_down_sync(mask, val, 1);
}


__device__ __forceinline__ float get_neighbor_2d(
    float val,
    int dx,
    int dy,
    int width,
    unsigned mask
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int x = lane_id % width;
    int y = lane_id / width;

    int nx = x + dx;
    int ny = y + dy;

    // Wrap around (toroidal boundary)
    if (nx < 0) nx += width;
    if (nx >= width) nx -= width;
    int height = WARP_SIZE / width;
    if (ny < 0) ny += height;
    if (ny >= height) ny -= height;

    int neighbor_lane = ny * width + nx;
    return __shfl_sync(mask, val, neighbor_lane);
}

__device__ __forceinline__ float growth_function(
    float potential,
    float center,
    float width
) {
    float z = (potential - center) / (width + 1e-7f);
    return 2.0f * expf(-z * z) - 1.0f;
}


__global__ void neural_ca_1d_warp_shuffle(
    const float* __restrict__ state,
    const float* __restrict__ kernel_weights,
    const float* __restrict__ growth_center,
    const float* __restrict__ growth_width,
    float* __restrict__ out,
    int batch,
    int seq_len,
    int channels
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (b >= batch || c >= channels) return;

    // Active threads mask (all threads in warp)
    unsigned mask = 0xFFFFFFFF;

    // Load CA state into register (one cell per thread)
    int seq_idx = warp_id * WARP_SIZE + lane_id;
    float cell_state = 0.0f;
    if (seq_idx < seq_len) {
        cell_state = state[b * seq_len * channels + seq_idx * channels + c];
    }

    // CA convolution via warp shuffles (no shared memory)
    float potential = 0.0f;
    for (int k = 0; k < KERNEL_SIZE; k++) {
        int offset = k - (KERNEL_SIZE / 2);
        float neighbor;

        if (offset < 0) {
            neighbor = get_left_neighbor(cell_state, mask);
        } else if (offset > 0) {
            neighbor = get_right_neighbor(cell_state, mask);
        } else {
            neighbor = cell_state;
        }

        // Boundary handling: zero-pad at edges
        if ((lane_id == 0 && offset < 0) ||
            (lane_id == WARP_SIZE - 1 && offset > 0)) {
            neighbor = 0.0f;
        }

        float weight = kernel_weights[c * channels * KERNEL_SIZE + c * KERNEL_SIZE + k];
        potential += neighbor * weight;
    }

    // Flow-Lenia growth function
    float center = growth_center[c];
    float width = growth_width[c];
    float growth = growth_function(potential, center, width);

    // CA update with mass conservation
    float new_state = cell_state + growth;

    // Mass conservation: sum across warp, normalize
    float mass_in = __shfl_down_sync(mask, cell_state, 16);
    mass_in += __shfl_down_sync(mask, mass_in, 8);
    mass_in += __shfl_down_sync(mask, mass_in, 4);
    mass_in += __shfl_down_sync(mask, mass_in, 2);
    mass_in += __shfl_down_sync(mask, mass_in, 1);

    float mass_out = __shfl_down_sync(mask, new_state, 16);
    mass_out += __shfl_down_sync(mask, mass_out, 8);
    mass_out += __shfl_down_sync(mask, mass_out, 4);
    mass_out += __shfl_down_sync(mask, mass_out, 2);
    mass_out += __shfl_down_sync(mask, mass_out, 1);

    // Broadcast mass ratio to all threads
    float mass_ratio = __shfl_sync(mask, mass_in / (mass_out + 1e-7f), 0);
    new_state *= mass_ratio;

    // Write output
    if (seq_idx < seq_len) {
        out[b * seq_len * channels + seq_idx * channels + c] = new_state;
    }
}

__device__ void jacobi_rotation(float* A, int n, int p, int q, float* s, float* c) {
    float app = A[p * n + p];
    float aqq = A[q * n + q];
    float apq = A[p * n + q];
    
    float tau = (aqq - app) / (2.0f * apq + EPSILON);
    float t = 1.0f / (fabsf(tau) + sqrtf(1.0f + tau * tau));
    if (tau < 0) t = -t;
    
    *c = 1.0f / sqrtf(1.0f + t * t);
    *s = t * (*c);
}

__global__ void effective_rank_kernel(
    float* __restrict__ correlation_matrix,
    float* __restrict__ rank_out,
    int n
) {
    __shared__ float A[32][32];
    __shared__ float singular_values[32];
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    
    if (tid < n * n) {
        A[tid / n][tid % n] = correlation_matrix[tid];
    }
    __syncthreads();
    
    for (int sweep = 0; sweep < MAX_JACOBI_ITERS; sweep++) {
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                float c, s;
                jacobi_rotation((float*)A, n, p, q, &s, &c);
                
                for (int i = 0; i < n; i++) {
                    float aip = A[i][p];
                    float aiq = A[i][q];
                    A[i][p] = c * aip - s * aiq;
                    A[i][q] = s * aip + c * aiq;
                }
            }
        }
    }
    
    if (tid < n) {
        singular_values[tid] = sqrtf(fabsf(A[tid][tid]));
    }
    __syncthreads();
    
    float sum = 0.0f;
    if (tid < n) {
        sum = singular_values[tid];
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    if (tid == 0) {
        float total_sum = sum;
        float entropy = 0.0f;
        for (int i = 0; i < n; i++) {
            float p = singular_values[i] / (total_sum + EPSILON);
            entropy -= p * logf(p + EPSILON);
        }
        *rank_out = expf(entropy);
    }
}

__global__ void coherence_kernel(
    float* __restrict__ prediction_errors,
    float* __restrict__ coherence_out,
    int history_length
) {
    int tid = threadIdx.x;
    unsigned mask = 0xFFFFFFFF;
    
    float sum_x = 0.0f, sum_y = 0.0f, sum_xx = 0.0f, sum_xy = 0.0f;
    
    for (int i = tid; i < history_length; i += WARP_SIZE) {
        float x = (float)i;
        float y = prediction_errors[i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_x += __shfl_down_sync(mask, sum_x, offset);
        sum_y += __shfl_down_sync(mask, sum_y, offset);
        sum_xx += __shfl_down_sync(mask, sum_xx, offset);
        sum_xy += __shfl_down_sync(mask, sum_xy, offset);
    }
    
    if (tid == 0) {
        float n = (float)history_length;
        float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x + EPSILON);
        float learning_progress = -slope;
        *coherence_out = 1.0f / (1.0f + expf(-learning_progress * 10.0f));
    }
}

__global__ void fitness_fused_kernel(
    float* __restrict__ correlation_matrix,
    float* __restrict__ prediction_errors,
    float* __restrict__ fitness_out,
    int matrix_size,
    int history_length
) {
    __shared__ float effective_rank;
    __shared__ float coherence;
    
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        effective_rank_kernel<<<1, 32>>>(correlation_matrix, &effective_rank, matrix_size);
    }
    __syncthreads();
    
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        coherence_kernel<<<1, 32>>>(prediction_errors, &coherence, history_length);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        fitness_out[blockIdx.x] = effective_rank * coherence;
    }
}

__global__ void compute_hunger_kernel(
    float* __restrict__ coherence_values,
    float* __restrict__ hunger_values,
    bool* __restrict__ should_survive,
    int num_organisms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_organisms) return;
    
    float coherence = coherence_values[idx];
    float hunger = 1.0f - coherence;
    hunger_values[idx] = hunger;
    should_survive[idx] = (hunger < 0.5f);
}


__global__ void neural_ca_2d_tensor_core(
    const half* __restrict__ state,
    const half* __restrict__ kernel_weights,
    const half* __restrict__ growth_center,
    const half* __restrict__ growth_width,
    half* __restrict__ out,
    int batch,
    int height,
    int width,
    int channels
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int tile_y = blockIdx.z / (width / BLOCK_DIM);
    int tile_x = blockIdx.z % (width / BLOCK_DIM);

    if (b >= batch || c >= channels) return;

    // Declare wmma fragments (tensor core registers)
    wmma::fragment<wmma::matrix_a, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, half> acc_frag;

    // Initialize accumulator
    wmma::fill_fragment(acc_frag, 0.0f);

    // Load 16x16 tile into shared memory (for tensor core access)
    __shared__ half tile_state[BLOCK_DIM][BLOCK_DIM];
    __shared__ half tile_kernel[BLOCK_DIM][BLOCK_DIM];

    int lane_id = threadIdx.x % WARP_SIZE;
    int ty = lane_id / BLOCK_DIM;
    int tx = lane_id % BLOCK_DIM;

    int global_y = tile_y * BLOCK_DIM + ty;
    int global_x = tile_x * BLOCK_DIM + tx;

    if (global_y < height && global_x < width) {
        int idx = b * height * width * channels +
                  global_y * width * channels +
                  global_x * channels + c;
        tile_state[ty][tx] = state[idx];
    } else {
        tile_state[ty][tx] = __float2half(0.0f);
    }

    __syncwarp();

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            __shared__ half neighbor_tile[BLOCK_DIM][BLOCK_DIM];
            __shared__ half kernel_tile[BLOCK_DIM][BLOCK_DIM];
            
            for (int dy = 0; dy < BLOCK_DIM; dy++) {
                for (int dx = 0; dx < BLOCK_DIM; dx++) {
                    int global_ny = tile_y * BLOCK_DIM + dy + ky;
                    int global_nx = tile_x * BLOCK_DIM + dx + kx;
                    
                    global_ny = max(0, min(global_ny, height - 1));
                    global_nx = max(0, min(global_nx, width - 1));
                    
                    if (threadIdx.x == dy * BLOCK_DIM + dx && threadIdx.x < BLOCK_DIM * BLOCK_DIM) {
                        int n_idx = b * height * width * channels +
                                   global_ny * width * channels +
                                   global_nx * channels + c;
                        neighbor_tile[dy][dx] = state[n_idx];
                        
                        int kernel_idx = c * channels * 9 + c * 9 + (ky + 1) * 3 + (kx + 1);
                        kernel_tile[dy][dx] = kernel_weights[kernel_idx];
                    }
                }
            }
            
            __syncthreads();
            
            wmma::fragment<wmma::matrix_a, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, half, wmma::row_major> n_frag;
            wmma::fragment<wmma::matrix_b, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, half, wmma::col_major> k_frag;
            
            wmma::load_matrix_sync(n_frag, &neighbor_tile[0][0], BLOCK_DIM);
            wmma::load_matrix_sync(k_frag, &kernel_tile[0][0], BLOCK_DIM);
            wmma::mma_sync(acc_frag, n_frag, k_frag, acc_frag);
        }
    }

    // Flow-Lenia growth function
    half center = growth_center[c];
    half width_param = growth_width[c];

    // Apply growth to accumulator fragments
    for (int i = 0; i < acc_frag.num_elements; i++) {
        float potential = __half2float(acc_frag.x[i]);
        float growth = growth_function(
            potential,
            __half2float(center),
            __half2float(width_param)
        );
        acc_frag.x[i] = __float2half(
            __half2float(tile_state[ty][tx]) + growth
        );
    }

    // Store result
    wmma::store_matrix_sync(&tile_state[0][0], acc_frag, BLOCK_DIM, wmma::mem_row_major);

    __syncwarp();

    if (global_y < height && global_x < width) {
        int idx = b * height * width * channels +
                  global_y * width * channels +
                  global_x * channels + c;
        out[idx] = tile_state[ty][tx];
    }
}


__global__ void mass_conservation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int spatial_size,
    int channels
) {
    int b = blockIdx.x;
    int c = blockIdx.y;

    if (b >= batch || c >= channels) return;

    unsigned mask = 0xFFFFFFFF;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Warp reduce for input mass
    float input_mass = 0.0f;
    for (int i = lane_id; i < spatial_size; i += WARP_SIZE) {
        int idx = b * spatial_size * channels + i * channels + c;
        input_mass += input[idx];
    }

    input_mass = __shfl_down_sync(mask, input_mass, 16);
    input_mass += __shfl_down_sync(mask, input_mass, 8);
    input_mass += __shfl_down_sync(mask, input_mass, 4);
    input_mass += __shfl_down_sync(mask, input_mass, 2);
    input_mass += __shfl_down_sync(mask, input_mass, 1);
    input_mass = __shfl_sync(mask, input_mass, 0);

    // Warp reduce for output mass
    float output_mass = 0.0f;
    for (int i = lane_id; i < spatial_size; i += WARP_SIZE) {
        int idx = b * spatial_size * channels + i * channels + c;
        output_mass += output[idx];
    }

    output_mass = __shfl_down_sync(mask, output_mass, 16);
    output_mass += __shfl_down_sync(mask, output_mass, 8);
    output_mass += __shfl_down_sync(mask, output_mass, 4);
    output_mass += __shfl_down_sync(mask, output_mass, 2);
    output_mass += __shfl_down_sync(mask, output_mass, 1);
    output_mass = __shfl_sync(mask, output_mass, 0);

    // Normalize output to match input mass
    float mass_ratio = input_mass / (output_mass + 1e-7f);
    for (int i = lane_id; i < spatial_size; i += WARP_SIZE) {
        int idx = b * spatial_size * channels + i * channels + c;
        output[idx] *= mass_ratio;
    }
}

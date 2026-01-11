
#include "../../config/config.cu"
#include "../../data/dataset_loader.cu"
#include "../../training/training_types.cu"
#include "../../core/pseudopod.cu"
#include "../../utils/genome_params.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
\
            exit(1); \
        } \
    } while(0)

__global__ void extract_ca_mass_kernel(
    float* ca_state,
    float* total_mass_out,
    int grid_size,
    int channels
) {
    __shared__ float block_sum;

    if (threadIdx.x == 0) {
        block_sum = 0.0f;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = grid_size * grid_size * channels;

    float local_sum = 0.0f;
    for (int i = tid; i < total_size; i += gridDim.x * blockDim.x) {
        local_sum += ca_state[i];
    }

    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = WMMA_TILE_DIM; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    if ((threadIdx.x % WARP_SIZE) == 0) {
        atomicAdd(&block_sum, local_sum);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(total_mass_out, block_sum);
    }
}

__device__ void compute_test_arch(uint64_t test_genome_hash, int* grid_size, int* channels) {
    int grid_size_slot = derive_param_slot(test_genome_hash, "grid_size");
    *grid_size = GRID_SIZE_MIN + (grid_size_slot % (GRID_SIZE_MAX - GRID_SIZE_MIN + 1));

    int channels_slot = derive_param_slot(test_genome_hash, "channels");
    *channels = CHANNELS_MIN + (channels_slot % (CHANNELS_MAX - CHANNELS_MIN + 1));
}

__global__ void get_arch_kernel(uint64_t hash, int* grid_size, int* channels) {
    compute_test_arch(hash, grid_size, channels);
}

bool test_mnist_to_ca_grid_conversion() {
    uint64_t test_genome_hash = 12345ULL;
    int* d_grid_size;
    int* d_channels;
    CUDA_CHECK(cudaMalloc(&d_grid_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_channels, sizeof(int)));

    get_arch_kernel<<<1, 1>>>(test_genome_hash, d_grid_size, d_channels);
    cudaDeviceSynchronize();

    int grid_size, channels;
    CUDA_CHECK(cudaMemcpy(&grid_size, d_grid_size, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&channels, d_channels, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_grid_size);
    cudaFree(d_channels);

    unsigned char* d_mnist_img;
    CUDA_CHECK(cudaMalloc(&d_mnist_img, MNIST_IMAGE_PIXELS));

    unsigned char h_img[MNIST_IMAGE_PIXELS];
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            h_img[y * 28 + x] = (x + y) * 3;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_mnist_img, h_img, MNIST_IMAGE_PIXELS, cudaMemcpyHostToDevice));

    float* d_ca_grid;
    CUDA_CHECK(cudaMalloc(&d_ca_grid, GRID_SIZE_MAX * GRID_SIZE_MAX * MAX_CHANNELS * sizeof(float)));

    dim3 grid_dim((grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM, (grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM);
    dim3 block(WMMA_TILE_DIM, WMMA_TILE_DIM);

    sample_to_ca_grid_kernel<<<grid_dim, block>>>(
        d_mnist_img,
        d_ca_grid,
        MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE,
        grid_size,
        channels
    );
    cudaDeviceSynchronize();

    float* h_ca_sample = (float*)malloc(MNIST_NUM_CLASSES * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_ca_sample, d_ca_grid, MNIST_NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));

    bool in_range = true;
    for (int i = 0; i < MNIST_NUM_CLASSES; i++) {
        if (h_ca_sample[i] < -1.0f || h_ca_sample[i] > 1.0f) {
            in_range = false;
        }
    }

    float first_channel = h_ca_sample[0];
    bool channel_broadcast = true;
    for (int c = 1; c < channels && c < MNIST_NUM_CLASSES; c++) {
        if (fabsf(h_ca_sample[c]) > safe_epsilon(first_channel)) {
            channel_broadcast = false;
        }
    }

    bool passed = in_range && channel_broadcast;

    free(h_ca_sample);
    cudaFree(d_mnist_img);
    cudaFree(d_ca_grid);

    return passed;
}


__device__ void compute_full_arch(uint64_t hash, int* grid_size, int* num_heads, int* channels, int* head_dim, int* hidden_dim) {
    int grid_size_slot = derive_param_slot(hash, "grid_size");
    *grid_size = GRID_SIZE_MIN + (grid_size_slot % (GRID_SIZE_MAX - GRID_SIZE_MIN + 1));

    int num_heads_slot = derive_param_slot(hash, "num_heads");
    *num_heads = NUM_HEADS_MIN + (num_heads_slot % (NUM_HEADS_MAX - NUM_HEADS_MIN + 1));

    int channels_slot = derive_param_slot(hash, "channels");
    *channels = CHANNELS_MIN + (channels_slot % (CHANNELS_MAX - CHANNELS_MIN + 1));

    int head_dim_slot = derive_param_slot(hash, "head_dim");
    *head_dim = HEAD_DIM_MIN + (head_dim_slot % (HEAD_DIM_MAX - HEAD_DIM_MIN + 1));

    *hidden_dim = (*num_heads) * (*head_dim);
}

__global__ void get_full_arch_kernel(uint64_t hash, int* grid_size, int* num_heads, int* channels, int* head_dim, int* hidden_dim) {
    compute_full_arch(hash, grid_size, num_heads, channels, head_dim, hidden_dim);
}

bool test_multi_head_ca_diversity() {
    uint64_t test_genome_hash = 54321ULL;

    int* d_grid_size;
    int* d_num_heads;
    int* d_channels;
    int* d_head_dim;
    int* d_hidden_dim;
    CUDA_CHECK(cudaMalloc(&d_grid_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_heads, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_channels, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_head_dim, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hidden_dim, sizeof(int)));

    get_full_arch_kernel<<<1, 1>>>(test_genome_hash, d_grid_size, d_num_heads, d_channels, d_head_dim, d_hidden_dim);
    cudaDeviceSynchronize();

    int grid_size, num_heads, channels, head_dim, hidden_dim;
    CUDA_CHECK(cudaMemcpy(&grid_size, d_grid_size, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&num_heads, d_num_heads, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&channels, d_channels, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&head_dim, d_head_dim, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hidden_dim, d_hidden_dim, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_grid_size);
    cudaFree(d_num_heads);
    cudaFree(d_channels);
    cudaFree(d_head_dim);
    cudaFree(d_hidden_dim);

    ArchitectureParams arch;
    arch.num_heads = num_heads;
    arch.channels = channels;
    arch.head_dim = head_dim;
    arch.hidden_dim = hidden_dim;
    arch.grid_size = grid_size;
    arch.ca_gate_center = 1.0f;  // Default for tests

    float* d_ca_state;
    float* d_ca_output;
    half* d_perception_weights;
    half* d_interaction_weights;
    half* d_value_weights;

    int state_size = GRID_SIZE_MAX * GRID_SIZE_MAX * MAX_CHANNELS;
    int output_size = NUM_HEADS_MAX * GRID_SIZE_MAX * GRID_SIZE_MAX * MAX_HEAD_DIM;

    CUDA_CHECK(cudaMalloc(&d_ca_state, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ca_output, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_perception_weights, HIDDEN_DIM_MAX * MAX_CHANNELS * HIDDEN_DIM_MAX * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_interaction_weights, HIDDEN_DIM_MAX * MAX_CHANNELS * HIDDEN_DIM_MAX * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_value_weights, HIDDEN_DIM_MAX * HIDDEN_DIM_MAX * MAX_CHANNELS * sizeof(half)));

    float* h_state = (float*)malloc(state_size * sizeof(float));
    for (int i = 0; i < state_size; i++) {
        h_state[i] = ((float)rand() / RAND_MAX) * 0.2f;
    }
    CUDA_CHECK(cudaMemcpy(d_ca_state, h_state, state_size * sizeof(float), cudaMemcpyHostToDevice));

    srand(12345);
    float* h_weights_fp32 = (float*)malloc(HIDDEN_DIM_MAX * MAX_CHANNELS * HIDDEN_DIM_MAX * sizeof(float));
    half* h_weights_fp16 = (half*)malloc(HIDDEN_DIM_MAX * MAX_CHANNELS * HIDDEN_DIM_MAX * sizeof(half));
    for (int i = 0; i < arch.num_heads * arch.channels * arch.hidden_dim; i++) {
        h_weights_fp32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_weights_fp16[i] = __float2half(h_weights_fp32[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_perception_weights, h_weights_fp16, HIDDEN_DIM_MAX * MAX_CHANNELS * HIDDEN_DIM_MAX * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interaction_weights, h_weights_fp16, HIDDEN_DIM_MAX * MAX_CHANNELS * HIDDEN_DIM_MAX * sizeof(half), cudaMemcpyHostToDevice));

    for (int i = 0; i < arch.num_heads * arch.hidden_dim * arch.channels; i++) {
        h_weights_fp32[i % (arch.num_heads * arch.channels * arch.hidden_dim)] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_weights_fp16[i % (arch.num_heads * arch.channels * arch.hidden_dim)] = __float2half(h_weights_fp32[i % (arch.num_heads * arch.channels * arch.hidden_dim)]);
    }
    CUDA_CHECK(cudaMemcpy(d_value_weights, h_weights_fp16, HIDDEN_DIM_MAX * HIDDEN_DIM_MAX * MAX_CHANNELS * sizeof(half), cudaMemcpyHostToDevice));

    dim3 grid_dim((grid_size + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM, arch.num_heads, 1);
    dim3 block(WMMA_TILE_DIM, WMMA_TILE_DIM);

    multi_head_ca_kernel<<<grid_dim, block>>>(
        d_ca_state,
        d_perception_weights,
        d_interaction_weights,
        d_value_weights,
        d_ca_output,
        1,
        grid_size,
        arch
    );
    cudaDeviceSynchronize();

    int head_output_size = grid_size * grid_size * arch.head_dim;
    float* h_head0 = (float*)malloc(head_output_size * sizeof(float));
    float* h_head1 = (float*)malloc(head_output_size * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_head0, d_ca_output, head_output_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_head1, &d_ca_output[head_output_size], head_output_size * sizeof(float), cudaMemcpyDeviceToHost));

    float dot = 0.0f, norm0 = 0.0f, norm1 = 0.0f;
    for (int i = 0; i < head_output_size; i++) {
        dot += h_head0[i] * h_head1[i];
        norm0 += h_head0[i] * h_head0[i];
        norm1 += h_head1[i] * h_head1[i];
    }
    float correlation = dot / fmaxf(sqrtf(norm0 * norm1), 1e-10f);

    bool diverse = (correlation < 0.95f);

    free(h_state);
    free(h_weights_fp32);
    free(h_weights_fp16);
    free(h_head0);
    free(h_head1);
    cudaFree(d_ca_state);
    cudaFree(d_ca_output);
    cudaFree(d_perception_weights);
    cudaFree(d_interaction_weights);
    cudaFree(d_value_weights);

    return diverse;
}

int main() {
    int passed = 0;
    int total = 0;

    total++; if (test_mnist_to_ca_grid_conversion()) passed++;
    total++; if (test_multi_head_ca_diversity()) passed++;

    return (passed == total) ? 0 : 1;
}

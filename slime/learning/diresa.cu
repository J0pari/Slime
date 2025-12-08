#ifndef DIRESA_CU
#define DIRESA_CU

#include "../config/config.cu"
#include <cuda_runtime.h>
#include <curand_kernel.h>

struct PoolEntry;
// DIRESA: Distance-Regularized Siamese Twin Autoencoder
// Paper: https://arxiv.org/abs/2404.18314
// Purpose: Compress behavioral features while preserving distance ordering for QD archive

struct DIRESAWeights {
    // Dimensions (makes struct reusable for hw/task/gen paths)
    int input_dim;   // HARDWARE_FEATURES_DIM or task/gen feature counts
    int output_dim;  // DIM_HW or DIM_TASK or DIM_GEN
    int hidden1;     // Genome-derived
    int hidden2;     // Genome-derived

    // Encoder pointers
    float* encoder_w1;  // [input_dim * hidden1]
    float* encoder_b1;  // [hidden1]
    float* encoder_w2;  // [hidden1 * hidden2]
    float* encoder_b2;  // [hidden2]
    float* encoder_w3;  // [hidden2 * output_dim]
    float* encoder_b3;  // [output_dim]

    // Decoder pointers
    float* decoder_w1;  // [output_dim * hidden2]
    float* decoder_b1;  // [hidden2]
    float* decoder_w2;  // [hidden2 * hidden1]
    float* decoder_b2;  // [hidden1]
    float* decoder_w3;  // [hidden1 * input_dim]
    float* decoder_b3;  // [input_dim]

    // Training state
    float cov_weight;
    float learning_rate;
    uint32_t training_step;

    // Parallel tempering
    float temperature;
    int replica_id;

    // Critical exponent for power-law distance scaling
    float distance_exponent;
    float quality_weight;
};

struct DIRESABatch {
    // Dimensions
    int input_dim;   // Feature dimension (must match DIRESAWeights.input_dim)
    int output_dim;  // Latent dimension (must match DIRESAWeights.output_dim)
    int batch_size;  // Genome-derived

    // Input features
    float* features;              // [batch_size * input_dim]
    float* features_shuffled;     // [batch_size * input_dim]
    int* shuffle_indices;         // [batch_size]

    // Latent representations
    float* latent;                // [batch_size * output_dim]
    float* latent_shuffled;       // [batch_size * output_dim]

    // Reconstructions
    float* reconstructed;         // [batch_size * input_dim]

    // Distance pairs
    float* orig_distances;        // [batch_size]
    float* latent_distances;      // [batch_size]

    // Loss values
    float recon_loss;
    float dist_loss;
    float cov_loss;
};

// Xavier initialization
__global__ void init_diresa_kernel(DIRESAWeights* replicas, int input_dim, int output_dim, PoolEntry* entry, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int replica_id = tid / 1024;
    int local_tid = tid % 1024;

    if (replica_id >= entry->num_tempering_replicas) return;

    DIRESAWeights* weights = &replicas[replica_id];

    // Allocate weight arrays using genome-derived + path-specific dimensions
    if (tid == 0 && replica_id == 0) {
        for (int r = 0; r < entry->num_tempering_replicas; r++) {
            cudaMalloc(&replicas[r].encoder_w1, sizeof(float) * input_dim * entry->diresa_hidden1);
            cudaMalloc(&replicas[r].encoder_b1, sizeof(float) * entry->diresa_hidden1);
            cudaMalloc(&replicas[r].encoder_w2, sizeof(float) * entry->diresa_hidden1 * entry->diresa_hidden2);
            cudaMalloc(&replicas[r].encoder_b2, sizeof(float) * entry->diresa_hidden2);
            cudaMalloc(&replicas[r].encoder_w3, sizeof(float) * entry->diresa_hidden2 * output_dim);
            cudaMalloc(&replicas[r].encoder_b3, sizeof(float) * output_dim);

            cudaMalloc(&replicas[r].decoder_w1, sizeof(float) * output_dim * entry->diresa_hidden2);
            cudaMalloc(&replicas[r].decoder_b1, sizeof(float) * entry->diresa_hidden2);
            cudaMalloc(&replicas[r].decoder_w2, sizeof(float) * entry->diresa_hidden2 * entry->diresa_hidden1);
            cudaMalloc(&replicas[r].decoder_b2, sizeof(float) * entry->diresa_hidden1);
            cudaMalloc(&replicas[r].decoder_w3, sizeof(float) * entry->diresa_hidden1 * input_dim);
            cudaMalloc(&replicas[r].decoder_b3, sizeof(float) * input_dim);

            replicas[r].input_dim = input_dim;
            replicas[r].output_dim = output_dim;
            replicas[r].hidden1 = entry->diresa_hidden1;
            replicas[r].hidden2 = entry->diresa_hidden2;
        }
    }
    __syncthreads();

    curandState state;
    curand_init(seed + tid, 0, 0, &state);

    int hidden1 = weights->hidden1;
    int hidden2 = weights->hidden2;
    int in_dim = weights->input_dim;
    int out_dim = weights->output_dim;

    // Encoder layer 1
    if (local_tid < in_dim * hidden1) {
        float scale = sqrtf(2.0f / (in_dim + hidden1));
        weights->encoder_w1[local_tid] = curand_normal(&state) * scale;
    }
    if (local_tid < hidden1) {
        weights->encoder_b1[local_tid] = 0.0f;
    }

    // Encoder layer 2
    if (local_tid < hidden1 * hidden2) {
        float scale = sqrtf(2.0f / (hidden1 + hidden2));
        weights->encoder_w2[local_tid] = curand_normal(&state) * scale;
    }
    if (local_tid < hidden2) {
        weights->encoder_b2[local_tid] = 0.0f;
    }

    // Encoder layer 3
    if (local_tid < hidden2 * out_dim) {
        float scale = sqrtf(2.0f / (hidden2 + out_dim));
        weights->encoder_w3[local_tid] = curand_normal(&state) * scale;
    }
    if (local_tid < out_dim) {
        weights->encoder_b3[local_tid] = 0.0f;
    }

    // Decoder mirrors encoder
    if (local_tid < out_dim * hidden2) {
        float scale = sqrtf(2.0f / (out_dim + hidden2));
        weights->decoder_w1[local_tid] = curand_normal(&state) * scale;
    }
    if (local_tid < hidden2) {
        weights->decoder_b1[local_tid] = 0.0f;
    }

    if (local_tid < hidden2 * hidden1) {
        float scale = sqrtf(2.0f / (hidden2 + hidden1));
        weights->decoder_w2[local_tid] = curand_normal(&state) * scale;
    }
    if (local_tid < hidden1) {
        weights->decoder_b2[local_tid] = 0.0f;
    }

    if (local_tid < hidden1 * in_dim) {
        float scale = sqrtf(2.0f / (hidden1 + in_dim));
        weights->decoder_w3[local_tid] = curand_normal(&state) * scale;
    }
    if (local_tid < in_dim) {
        weights->decoder_b3[local_tid] = 0.0f;
    }

    // Initialize training state
    if (local_tid == 0) {
        weights->cov_weight = 0.0f;  // Annealing starts at 0
        weights->learning_rate = 0.005f;
        weights->training_step = 0;
        weights->replica_id = replica_id;
        weights->temperature = 1.0f + replica_id * 0.5f;  // [1.0, 1.5, 2.0, 2.5]
        weights->distance_exponent = entry->distance_exponent;
        weights->quality_weight = entry->quality_weight;
    }
}

// ReLU activation
__device__ inline float relu(float x) {
    return fmaxf(0.0f, x);
}

// Encoder forward pass (shared weights for Siamese twins)
__device__ void diresa_encode(const float* features, float* latent, const DIRESAWeights* weights) {
    float hidden1[DIRESA_HIDDEN1_MAX];
    float hidden2[DIRESA_HIDDEN2_MAX];

    // Layer 1: input_dim -> HIDDEN1
    for (int i = 0; i < weights->hidden1; i++) {
        float sum = weights->encoder_b1[i];
        for (int j = 0; j < weights->input_dim; j++) {
            sum += features[j] * weights->encoder_w1[j * weights->hidden1 + i];
        }
        hidden1[i] = relu(sum);
    }

    // Layer 2: HIDDEN1 -> HIDDEN2
    for (int i = 0; i < weights->hidden2; i++) {
        float sum = weights->encoder_b2[i];
        for (int j = 0; j < weights->hidden1; j++) {
            sum += hidden1[j] * weights->encoder_w2[j * weights->hidden2 + i];
        }
        hidden2[i] = relu(sum);
    }

    // Layer 3: HIDDEN2 -> output_dim (linear output)
    for (int i = 0; i < weights->output_dim; i++) {
        float sum = weights->encoder_b3[i];
        for (int j = 0; j < weights->hidden2; j++) {
            sum += hidden2[j] * weights->encoder_w3[j * weights->output_dim + i];
        }
        latent[i] = sum;  // Linear activation for latent space
    }
}

// Decoder forward pass
__device__ void diresa_decode(const float* latent, float* reconstructed, const DIRESAWeights* weights) {
    float hidden1[DIRESA_HIDDEN2_MAX];
    float hidden2[DIRESA_HIDDEN1_MAX];

    // Layer 1: output_dim -> HIDDEN2
    for (int i = 0; i < weights->hidden2; i++) {
        float sum = weights->decoder_b1[i];
        for (int j = 0; j < weights->output_dim; j++) {
            sum += latent[j] * weights->decoder_w1[j * weights->hidden2 + i];
        }
        hidden1[i] = relu(sum);
    }

    // Layer 2: HIDDEN2 -> HIDDEN1
    for (int i = 0; i < weights->hidden1; i++) {
        float sum = weights->decoder_b2[i];
        for (int j = 0; j < weights->hidden2; j++) {
            sum += hidden1[j] * weights->decoder_w2[j * weights->hidden1 + i];
        }
        hidden2[i] = relu(sum);
    }

    // Layer 3: HIDDEN1 -> input_dim (linear output)
    for (int i = 0; i < weights->input_dim; i++) {
        float sum = weights->decoder_b3[i];
        for (int j = 0; j < weights->hidden1; j++) {
            sum += hidden2[j] * weights->decoder_w3[j * weights->input_dim + i];
        }
        reconstructed[i] = sum;
    }
}

// Siamese twin forward pass: encode both original and shuffled inputs
__global__ void diresa_forward_kernel(DIRESABatch* batch, const DIRESAWeights* weights) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch->batch_size) return;

    const float* features = batch->features + tid * batch->input_dim;
    const float* features_shuffled = batch->features_shuffled + tid * batch->input_dim;
    float* latent = batch->latent + tid * batch->output_dim;
    float* latent_shuffled = batch->latent_shuffled + tid * batch->output_dim;
    float* reconstructed = batch->reconstructed + tid * batch->input_dim;

    diresa_encode(features, latent, weights);
    CooperativeSync::sync_warp();

    diresa_encode(features_shuffled, latent_shuffled, weights);
    CooperativeSync::sync_warp();

    diresa_decode(latent, reconstructed, weights);
    CooperativeSync::sync_warp();
}

// Distance layer: compute ||x - x'|| and ||z - z'|| for correlation loss
__global__ void diresa_distance_kernel(DIRESABatch* batch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch->batch_size) return;

    int shuffled_idx = batch->shuffle_indices[tid];

    const float* features_i = batch->features + tid * batch->input_dim;
    const float* features_j = batch->features + shuffled_idx * batch->input_dim;
    const float* latent_i = batch->latent + tid * batch->output_dim;
    const float* latent_j = batch->latent + shuffled_idx * batch->output_dim;

    // Compute original space distance: ||x_i - x_j||
    float orig_dist_sq = 0.0f;
    for (int k = 0; k < batch->input_dim; k++) {
        float diff = features_i[k] - features_j[k];
        orig_dist_sq += diff * diff;
    }
    batch->orig_distances[tid] = sqrtf(orig_dist_sq);

    // Compute latent space distance: ||z_i - z_j||
    float latent_dist_sq = 0.0f;
    for (int k = 0; k < batch->output_dim; k++) {
        float diff = latent_i[k] - latent_j[k];
        latent_dist_sq += diff * diff;
    }
    batch->latent_distances[tid] = sqrtf(latent_dist_sq);
}

// Loss computation: 3 components (reconstruction, distance correlation, covariance)
__global__ void diresa_loss_kernel(DIRESABatch* batch, const DIRESAWeights* weights) {
    __shared__ float shared_recon[256];
    __shared__ float shared_orig_mean[1];
    __shared__ float shared_orig_var[1];
    __shared__ float shared_latent_mean[1];
    __shared__ float shared_latent_var[1];
    __shared__ float shared_cov_sum[1];

    int tid = threadIdx.x;
    int sample_idx = blockIdx.x * blockDim.x + tid;

    // 1. Reconstruction Loss: MSE between original and reconstructed
    float local_recon = 0.0f;
    if (sample_idx < batch->batch_size) {
        const float* orig = batch->features + sample_idx * batch->input_dim;
        const float* recon = batch->reconstructed + sample_idx * batch->input_dim;
        for (int i = 0; i < batch->input_dim; i++) {
            float diff = orig[i] - recon[i];
            local_recon += diff * diff;
        }
        local_recon /= batch->input_dim;
    }
    shared_recon[tid] = local_recon;
    __syncthreads();

    // Reduce reconstruction loss
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_recon[tid] += shared_recon[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&batch->recon_loss, shared_recon[0] / batch->batch_size);
    }

    // 2. Power-Law Distance Scaling: d_latent ∝ d_orig^α
    // Fit in log-log space, generalization of Pearson correlation to power-law regime
    if (blockIdx.x == 0) {
        float target_alpha = weights->distance_exponent;

        // Compute mean of log(original distances) - parallel with tid 0
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < batch->batch_size; i++) {
                sum += logf(batch->orig_distances[i] + EPSILON);
            }
            shared_orig_mean[0] = sum / batch->batch_size;
        }
        __syncthreads();

        // Compute variance of log(original distances)
        if (tid == 0) {
            float sum_sq = 0.0f;
            float mean = shared_orig_mean[0];
            for (int i = 0; i < batch->batch_size; i++) {
                float diff = logf(batch->orig_distances[i] + EPSILON) - mean;
                sum_sq += diff * diff;
            }
            shared_orig_var[0] = sum_sq / batch->batch_size;
        }

        // Compute mean of log(latent distances) - parallel with tid 1
        if (tid == 1) {
            float sum = 0.0f;
            for (int i = 0; i < batch->batch_size; i++) {
                sum += logf(batch->latent_distances[i] + EPSILON);
            }
            shared_latent_mean[0] = sum / batch->batch_size;
        }
        __syncthreads();

        // Compute variance and log-log covariance (slope = power-law exponent) - PARALLEL
        float latent_mean = shared_latent_mean[0];
        float orig_mean = shared_orig_mean[0];

        float local_var = 0.0f;
        float local_cov = 0.0f;

        for (int i = tid; i < batch->batch_size; i += blockDim.x) {
            float latent_diff = logf(batch->latent_distances[i] + EPSILON) - latent_mean;
            float orig_diff = logf(batch->orig_distances[i] + EPSILON) - orig_mean;
            local_var += latent_diff * latent_diff;
            local_cov += latent_diff * orig_diff;
        }

        shared_recon[tid] = local_var;
        shared_cov_sum[0] = 0.0f;
        __syncthreads();

        // Reduce variance
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_recon[tid] += shared_recon[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_latent_var[0] = shared_recon[0] / batch->batch_size;
        }

        // Reduce covariance
        shared_recon[tid] = local_cov;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_recon[tid] += shared_recon[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_cov_sum[0] = shared_recon[0];

            // Power-law exponent from log-log slope: α = cov(log d_latent, log d_orig) / var(log d_orig)
            float alpha_measured = shared_cov_sum[0] / (shared_orig_var[0] * batch->batch_size + EPSILON);

            // Log-log correlation quality (how well data fits power law)
            float log_correlation = shared_cov_sum[0] / (sqrtf(shared_orig_var[0] * shared_latent_var[0]) * batch->batch_size + EPSILON);

            // Combined loss: penalize both deviation from target exponent AND poor power-law fit quality
            float exponent_loss = (alpha_measured - target_alpha) * (alpha_measured - target_alpha);
            float quality_loss = 1.0f - fabsf(log_correlation);  // 0 when perfect power-law fit
            batch->dist_loss = exponent_loss + weights->quality_weight * quality_loss;
        }
    }

    // 3. Covariance Loss: L_cov = 1/(L*(L-1)) * Σ(i≠j) cov²(i,j)(Z)
    // Forces latent components to be statistically independent
    if (blockIdx.x == 0 && tid == 0) {
        float latent_means[BEHAVIORAL_DIM_MAX] = {0};

        // Compute means per latent dimension
        for (int dim = 0; dim < batch->output_dim; dim++) {
            float sum = 0.0f;
            for (int i = 0; i < batch->batch_size; i++) {
                sum += batch->latent[i * batch->output_dim + dim];
            }
            latent_means[dim] = sum / batch->batch_size;
        }

        // Compute off-diagonal covariance terms
        float cov_sum = 0.0f;
        for (int i = 0; i < batch->output_dim; i++) {
            for (int j = i + 1; j < batch->output_dim; j++) {
                float cov_ij = 0.0f;
                for (int k = 0; k < batch->batch_size; k++) {
                    float zi = batch->latent[k * batch->output_dim + i] - latent_means[i];
                    float zj = batch->latent[k * batch->output_dim + j] - latent_means[j];
                    cov_ij += zi * zj;
                }
                cov_ij /= batch->batch_size;
                cov_sum += cov_ij * cov_ij;
            }
        }

        // Normalize by number of off-diagonal terms
        int num_pairs = batch->output_dim * (batch->output_dim - 1) / 2;
        batch->cov_loss = cov_sum / num_pairs;
    }
}

// Annealing logic: gradually increase covariance weight
__device__ void update_annealing(DIRESAWeights* weights, float cov_loss, PoolEntry* entry) {
    if (cov_loss > entry->cov_target && weights->cov_weight < 10.0f) {
        weights->cov_weight += entry->anneal_step;
    }
}

// Parallel tempering: replica exchange based on energy difference
__global__ void replica_exchange_kernel(DIRESAWeights* replicas, DIRESABatch* batches, PoolEntry* entry, curandState* rand_states) {
    int tid = threadIdx.x;
    if (tid >= entry->num_tempering_replicas - 1) return;

    // Try to swap replica i with replica i+1
    int i = tid;
    int j = tid + 1;

    float E_i = batches[i].recon_loss * entry->recon_weight +
                batches[i].dist_loss * entry->dist_weight +
                batches[i].cov_loss * replicas[i].cov_weight;

    float E_j = batches[j].recon_loss * entry->recon_weight +
                batches[j].dist_loss * entry->dist_weight +
                batches[j].cov_loss * replicas[j].cov_weight;

    float beta_i = 1.0f / replicas[i].temperature;
    float beta_j = 1.0f / replicas[j].temperature;

    // Metropolis acceptance criterion
    float delta = (beta_j - beta_i) * (E_i - E_j);
    float accept_prob = fminf(1.0f, expf(delta));

    float rand = curand_uniform(&rand_states[tid]);
    if (rand < accept_prob) {
        // Swap temperatures (not weights - we want to explore same landscape)
        float temp_swap = replicas[i].temperature;
        replicas[i].temperature = replicas[j].temperature;
        replicas[j].temperature = temp_swap;
    }
}

#endif
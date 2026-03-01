#ifndef DIRESA_CU
#define DIRESA_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include "../utils/cuda_primitives.cuh"
#include "../utils/genome_params.cuh"
#include "diresa_types.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Parallel entry-based DIRESA init - each block handles entries[blockIdx.x]
__device__ void init_diresa_entry_device(
    DIRESAWeights* weights,
    float* weight_pool,
    float* grad_pool,
    size_t stride,
    int input_dim,
    int output_dim,
    int hidden1,
    int hidden2,
    float distance_exponent,
    float quality_weight,
    float* genome,
    float* gradients,
    unsigned int seed
) {
    int local_tid = threadIdx.x;

    // Thread 0 sets up weight pointers
    if (local_tid == 0) {
        size_t offset = 0;

        weights->encoder_w1 = &weight_pool[offset];
        offset += input_dim * DIRESA_HIDDEN1_MAX;

        weights->encoder_b1 = &weight_pool[offset];
        offset += DIRESA_HIDDEN1_MAX;

        weights->encoder_w2 = &weight_pool[offset];
        offset += DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX;

        weights->encoder_b2 = &weight_pool[offset];
        offset += DIRESA_HIDDEN2_MAX;

        weights->encoder_w3 = &weight_pool[offset];
        offset += DIRESA_HIDDEN2_MAX * output_dim;

        weights->encoder_b3 = &weight_pool[offset];
        offset += output_dim;

        weights->decoder_w1 = &weight_pool[offset];
        offset += output_dim * DIRESA_HIDDEN2_MAX;

        weights->decoder_b1 = &weight_pool[offset];
        offset += DIRESA_HIDDEN2_MAX;

        weights->decoder_w2 = &weight_pool[offset];
        offset += DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX;

        weights->decoder_b2 = &weight_pool[offset];
        offset += DIRESA_HIDDEN1_MAX;

        weights->decoder_w3 = &weight_pool[offset];
        offset += DIRESA_HIDDEN1_MAX * input_dim;

        weights->decoder_b3 = &weight_pool[offset];

        // Gradient pointers mirror weight layout
        size_t goffset = 0;
        weights->encoder_w1_grad = &grad_pool[goffset]; goffset += input_dim * DIRESA_HIDDEN1_MAX;
        weights->encoder_b1_grad = &grad_pool[goffset]; goffset += DIRESA_HIDDEN1_MAX;
        weights->encoder_w2_grad = &grad_pool[goffset]; goffset += DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX;
        weights->encoder_b2_grad = &grad_pool[goffset]; goffset += DIRESA_HIDDEN2_MAX;
        weights->encoder_w3_grad = &grad_pool[goffset]; goffset += DIRESA_HIDDEN2_MAX * output_dim;
        weights->encoder_b3_grad = &grad_pool[goffset]; goffset += output_dim;
        weights->decoder_w1_grad = &grad_pool[goffset]; goffset += output_dim * DIRESA_HIDDEN2_MAX;
        weights->decoder_b1_grad = &grad_pool[goffset]; goffset += DIRESA_HIDDEN2_MAX;
        weights->decoder_w2_grad = &grad_pool[goffset]; goffset += DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX;
        weights->decoder_b2_grad = &grad_pool[goffset]; goffset += DIRESA_HIDDEN1_MAX;
        weights->decoder_w3_grad = &grad_pool[goffset]; goffset += DIRESA_HIDDEN1_MAX * input_dim;
        weights->decoder_b3_grad = &grad_pool[goffset];

        weights->input_dim = input_dim;
        weights->output_dim = output_dim;
        weights->hidden1 = hidden1;
        weights->hidden2 = hidden2;
    }
    __syncthreads();

    // Initialize random state per thread
    curandState state;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed + global_tid, 0, 0, &state);

    int h1 = weights->hidden1;
    int h2 = weights->hidden2;
    int in_dim = weights->input_dim;
    int out_dim = weights->output_dim;

    // Parallel weight initialization across threads in this block
    if (local_tid < in_dim * h1) {
        float scale = sqrtf(2.0f / (in_dim + h1));
        weights->encoder_w1[local_tid] = validated_curand_normal(&state, "diresa_init_enc1", local_tid) * scale;
    }
    if (local_tid < h1) {
        weights->encoder_b1[local_tid] = 0.0f;
    }

    if (local_tid < h1 * h2) {
        float scale = sqrtf(2.0f / (h1 + h2));
        weights->encoder_w2[local_tid] = validated_curand_normal(&state, "diresa_init_enc2", local_tid) * scale;
    }
    if (local_tid < h2) {
        weights->encoder_b2[local_tid] = 0.0f;
    }

    if (local_tid < h2 * out_dim) {
        float scale = sqrtf(2.0f / (h2 + out_dim));
        weights->encoder_w3[local_tid] = validated_curand_normal(&state, "diresa_init_enc3", local_tid) * scale;
    }
    if (local_tid < out_dim) {
        weights->encoder_b3[local_tid] = 0.0f;
    }

    if (local_tid < out_dim * h2) {
        float scale = sqrtf(2.0f / (out_dim + h2));
        weights->decoder_w1[local_tid] = validated_curand_normal(&state, "diresa_init_dec1", local_tid) * scale;
    }
    if (local_tid < h2) {
        weights->decoder_b1[local_tid] = 0.0f;
    }

    if (local_tid < h2 * h1) {
        float scale = sqrtf(2.0f / (h2 + h1));
        weights->decoder_w2[local_tid] = validated_curand_normal(&state, "diresa_init_dec2", local_tid) * scale;
    }
    if (local_tid < h1) {
        weights->decoder_b2[local_tid] = 0.0f;
    }

    if (local_tid < h1 * in_dim) {
        float scale = sqrtf(2.0f / (h1 + in_dim));
        weights->decoder_w3[local_tid] = validated_curand_normal(&state, "diresa_init_dec3", local_tid) * scale;
    }
    if (local_tid < in_dim) {
        weights->decoder_b3[local_tid] = 0.0f;
    }

    if (local_tid == 0) {
        weights->cov_weight = 0.0f;
        weights->training_step = 0;
        weights->replica_id = 0;
        weights->distance_exponent = distance_exponent;
        weights->quality_weight = quality_weight;

        int diresa_ctx_metabolic_slot = GenomeParamTable::diresa_ctx_metabolic;
        int diresa_ctx_stress_slot = GenomeParamTable::diresa_ctx_stress;
        int diresa_ctx_morphogen_slot = GenomeParamTable::diresa_ctx_morphogen;
        float diresa_ctx_metabolic = genome_slot_to_unit(genome, diresa_ctx_metabolic_slot);
        float diresa_ctx_stress = genome_slot_to_unit(genome, diresa_ctx_stress_slot);
        float diresa_ctx_morphogen = genome_slot_to_unit(genome, diresa_ctx_morphogen_slot);

        int diresa_temp_base_slot = GenomeParamTable::diresa_temp_base;
        int diresa_temp_scale_slot = GenomeParamTable::diresa_temp_scale;
        float temp_base = genome_slot_to_unit(genome, diresa_temp_base_slot);
        float temp_scale = genome_slot_to_unit(genome, diresa_temp_scale_slot);
        weights->temperature = DIRESA_TEMP_BASE_MIN + temp_base * (DIRESA_TEMP_BASE_MAX - DIRESA_TEMP_BASE_MIN);

        int diresa_ctx_complexity_slot = GenomeParamTable::diresa_ctx_complexity;
        int diresa_ctx_niche_slot = GenomeParamTable::diresa_ctx_niche;
        int diresa_ctx_learning_slot = GenomeParamTable::diresa_ctx_learning;
        int diresa_ctx_performance_slot = GenomeParamTable::diresa_ctx_performance;
        float diresa_ctx_complexity = genome_slot_to_unit(genome, diresa_ctx_complexity_slot);
        float diresa_ctx_niche = genome_slot_to_unit(genome, diresa_ctx_niche_slot);
        float diresa_ctx_learning = genome_slot_to_unit(genome, diresa_ctx_learning_slot);
        float diresa_ctx_performance = genome_slot_to_unit(genome, diresa_ctx_performance_slot);

        TrainingParams diresa_training_params;
        weights->learning_rate = diresa_training_params.get_behavioral_learning_rate(
            genome, gradients,
            diresa_ctx_metabolic, diresa_ctx_stress, diresa_ctx_morphogen,
            diresa_ctx_complexity, diresa_ctx_niche, diresa_ctx_learning, diresa_ctx_performance
        );
    }
}

__device__ void diresa_encode(const float* features, float* latent, const DIRESAWeights* weights) {
    float hidden1[DIRESA_HIDDEN1_MAX];
    float hidden2[DIRESA_HIDDEN2_MAX];

    for (int i = 0; i < weights->hidden1; i++) {
        float sum = weights->encoder_b1[i];
        for (int j = 0; j < weights->input_dim; j++) {
            sum += features[j] * weights->encoder_w1[j * weights->hidden1 + i];
        }
        hidden1[i] = activation_relu(sum);
    }

    for (int i = 0; i < weights->hidden2; i++) {
        float sum = weights->encoder_b2[i];
        for (int j = 0; j < weights->hidden1; j++) {
            sum += hidden1[j] * weights->encoder_w2[j * weights->hidden2 + i];
        }
        hidden2[i] = activation_relu(sum);
    }

    int vec_output_dim = weights->output_dim / 4;
    int remainder_dim = weights->output_dim % 4;

    for (int i = 0; i < vec_output_dim; i++) {
        float4 sum4 = make_float4(
            weights->encoder_b3[i * 4 + 0],
            weights->encoder_b3[i * 4 + 1],
            weights->encoder_b3[i * 4 + 2],
            weights->encoder_b3[i * 4 + 3]
        );

        for (int j = 0; j < weights->hidden2; j++) {
            float h = hidden2[j];
            sum4.x += h * weights->encoder_w3[j * weights->output_dim + i * 4 + 0];
            sum4.y += h * weights->encoder_w3[j * weights->output_dim + i * 4 + 1];
            sum4.z += h * weights->encoder_w3[j * weights->output_dim + i * 4 + 2];
            sum4.w += h * weights->encoder_w3[j * weights->output_dim + i * 4 + 3];
        }

        reinterpret_cast<float4*>(latent)[i] = sum4;
    }

    for (int i = vec_output_dim * 4; i < weights->output_dim; i++) {
        float sum = weights->encoder_b3[i];
        for (int j = 0; j < weights->hidden2; j++) {
            sum += hidden2[j] * weights->encoder_w3[j * weights->output_dim + i];
        }
        latent[i] = sum;
    }
}

// Encoder backward: computes feature gradients AND accumulates weight gradients
__device__ void diresa_encode_backward(const float* features, const float* latent_grad, float* features_grad, DIRESAWeights* weights) {
    float hidden1[DIRESA_HIDDEN1_MAX];
    float hidden2[DIRESA_HIDDEN2_MAX];
    float hidden1_grad[DIRESA_HIDDEN1_MAX];
    float hidden2_grad[DIRESA_HIDDEN2_MAX];

    // Forward pass to cache activations
    for (int i = 0; i < weights->hidden1; i++) {
        float sum = weights->encoder_b1[i];
        for (int j = 0; j < weights->input_dim; j++) {
            sum += features[j] * weights->encoder_w1[j * weights->hidden1 + i];
        }
        hidden1[i] = activation_relu(sum);
    }

    for (int i = 0; i < weights->hidden2; i++) {
        float sum = weights->encoder_b2[i];
        for (int j = 0; j < weights->hidden1; j++) {
            sum += hidden1[j] * weights->encoder_w2[j * weights->hidden2 + i];
        }
        hidden2[i] = activation_relu(sum);
    }

    // Layer 3 (hidden2 -> latent): weight grads and hidden2 grad
    for (int i = 0; i < weights->hidden2; i++) {
        hidden2_grad[i] = 0.0f;
    }
    for (int i = 0; i < weights->output_dim; i++) {
        float grad = latent_grad[i];
        weights->encoder_b3_grad[i] += grad;
        for (int j = 0; j < weights->hidden2; j++) {
            weights->encoder_w3_grad[j * weights->output_dim + i] += grad * hidden2[j];
            hidden2_grad[j] += grad * weights->encoder_w3[j * weights->output_dim + i];
        }
    }

    // Layer 2 (hidden1 -> hidden2): apply ReLU derivative, weight grads, hidden1 grad
    for (int i = 0; i < weights->hidden1; i++) {
        hidden1_grad[i] = 0.0f;
    }
    for (int i = 0; i < weights->hidden2; i++) {
        float grad = hidden2_grad[i] * (hidden2[i] > 0.0f ? 1.0f : 0.0f);
        weights->encoder_b2_grad[i] += grad;
        for (int j = 0; j < weights->hidden1; j++) {
            weights->encoder_w2_grad[j * weights->hidden2 + i] += grad * hidden1[j];
            hidden1_grad[j] += grad * weights->encoder_w2[j * weights->hidden2 + i];
        }
    }

    // Layer 1 (input -> hidden1): apply ReLU derivative, weight grads, features grad
    for (int i = 0; i < weights->input_dim; i++) {
        features_grad[i] = 0.0f;
    }
    for (int i = 0; i < weights->hidden1; i++) {
        float grad = hidden1_grad[i] * (hidden1[i] > 0.0f ? 1.0f : 0.0f);
        weights->encoder_b1_grad[i] += grad;
        for (int j = 0; j < weights->input_dim; j++) {
            weights->encoder_w1_grad[j * weights->hidden1 + i] += grad * features[j];
            features_grad[j] += grad * weights->encoder_w1[j * weights->hidden1 + i];
        }
    }
}

// Decoder backward: computes latent gradients AND accumulates weight gradients
// Mirror structure of diresa_decode: latent -> hidden1(h2 size) -> hidden2(h1 size) -> reconstructed(input_dim)
__device__ void diresa_decode_backward(const float* latent, const float* reconstructed_grad, float* latent_grad, DIRESAWeights* weights) {
    float hidden1[DIRESA_HIDDEN2_MAX];  // decoder layer 1 has hidden2 units
    float hidden2[DIRESA_HIDDEN1_MAX];  // decoder layer 2 has hidden1 units
    float hidden1_grad[DIRESA_HIDDEN2_MAX];
    float hidden2_grad[DIRESA_HIDDEN1_MAX];

    // Forward pass to cache activations (mirrors diresa_decode)
    for (int i = 0; i < weights->hidden2; i++) {
        float sum = weights->decoder_b1[i];
        for (int j = 0; j < weights->output_dim; j++) {
            sum += latent[j] * weights->decoder_w1[j * weights->hidden2 + i];
        }
        hidden1[i] = activation_relu(sum);
    }

    for (int i = 0; i < weights->hidden1; i++) {
        float sum = weights->decoder_b2[i];
        for (int j = 0; j < weights->hidden2; j++) {
            sum += hidden1[j] * weights->decoder_w2[j * weights->hidden1 + i];
        }
        hidden2[i] = activation_relu(sum);
    }

    // Layer 3 (hidden2 -> reconstructed): weight grads and hidden2 grad
    for (int i = 0; i < weights->hidden1; i++) {
        hidden2_grad[i] = 0.0f;
    }
    for (int i = 0; i < weights->input_dim; i++) {
        float grad = reconstructed_grad[i];
        weights->decoder_b3_grad[i] += grad;
        for (int j = 0; j < weights->hidden1; j++) {
            weights->decoder_w3_grad[j * weights->input_dim + i] += grad * hidden2[j];
            hidden2_grad[j] += grad * weights->decoder_w3[j * weights->input_dim + i];
        }
    }

    // Layer 2 (hidden1 -> hidden2): ReLU derivative, weight grads, hidden1 grad
    for (int i = 0; i < weights->hidden2; i++) {
        hidden1_grad[i] = 0.0f;
    }
    for (int i = 0; i < weights->hidden1; i++) {
        float grad = hidden2_grad[i] * (hidden2[i] > 0.0f ? 1.0f : 0.0f);
        weights->decoder_b2_grad[i] += grad;
        for (int j = 0; j < weights->hidden2; j++) {
            weights->decoder_w2_grad[j * weights->hidden1 + i] += grad * hidden1[j];
            hidden1_grad[j] += grad * weights->decoder_w2[j * weights->hidden1 + i];
        }
    }

    // Layer 1 (latent -> hidden1): ReLU derivative, weight grads, latent grad
    for (int i = 0; i < weights->output_dim; i++) {
        latent_grad[i] = 0.0f;
    }
    for (int i = 0; i < weights->hidden2; i++) {
        float grad = hidden1_grad[i] * (hidden1[i] > 0.0f ? 1.0f : 0.0f);
        weights->decoder_b1_grad[i] += grad;
        for (int j = 0; j < weights->output_dim; j++) {
            weights->decoder_w1_grad[j * weights->hidden2 + i] += grad * latent[j];
            latent_grad[j] += grad * weights->decoder_w1[j * weights->hidden2 + i];
        }
    }
}

__device__ void diresa_decode(const float* latent, float* reconstructed, const DIRESAWeights* weights) {
    float hidden1[DIRESA_HIDDEN2_MAX];
    float hidden2[DIRESA_HIDDEN1_MAX];

    int vec_output_dim = weights->output_dim / 4;
    for (int i = 0; i < weights->hidden2; i++) {
        float sum = weights->decoder_b1[i];

        for (int j = 0; j < vec_output_dim; j++) {
            float4 latent4 = reinterpret_cast<const float4*>(latent)[j];
            sum += latent4.x * weights->decoder_w1[(j * 4 + 0) * weights->hidden2 + i];
            sum += latent4.y * weights->decoder_w1[(j * 4 + 1) * weights->hidden2 + i];
            sum += latent4.z * weights->decoder_w1[(j * 4 + 2) * weights->hidden2 + i];
            sum += latent4.w * weights->decoder_w1[(j * 4 + 3) * weights->hidden2 + i];
        }

        for (int j = vec_output_dim * 4; j < weights->output_dim; j++) {
            sum += latent[j] * weights->decoder_w1[j * weights->hidden2 + i];
        }

        hidden1[i] = activation_relu(sum);
    }

    for (int i = 0; i < weights->hidden1; i++) {
        float sum = weights->decoder_b2[i];
        for (int j = 0; j < weights->hidden2; j++) {
            sum += hidden1[j] * weights->decoder_w2[j * weights->hidden1 + i];
        }
        hidden2[i] = activation_relu(sum);
    }

    for (int i = 0; i < weights->input_dim; i++) {
        float sum = weights->decoder_b3[i];
        for (int j = 0; j < weights->hidden1; j++) {
            sum += hidden2[j] * weights->decoder_w3[j * weights->input_dim + i];
        }
        reconstructed[i] = sum;
    }
}

// Zero all weight gradient accumulators
__device__ void diresa_zero_grads(DIRESAWeights* w) {
    int h1 = w->hidden1, h2 = w->hidden2;
    int in_d = w->input_dim, out_d = w->output_dim;
    for (int i = 0; i < in_d * h1; i++) w->encoder_w1_grad[i] = 0.0f;
    for (int i = 0; i < h1; i++) w->encoder_b1_grad[i] = 0.0f;
    for (int i = 0; i < h1 * h2; i++) w->encoder_w2_grad[i] = 0.0f;
    for (int i = 0; i < h2; i++) w->encoder_b2_grad[i] = 0.0f;
    for (int i = 0; i < h2 * out_d; i++) w->encoder_w3_grad[i] = 0.0f;
    for (int i = 0; i < out_d; i++) w->encoder_b3_grad[i] = 0.0f;
    for (int i = 0; i < out_d * h2; i++) w->decoder_w1_grad[i] = 0.0f;
    for (int i = 0; i < h2; i++) w->decoder_b1_grad[i] = 0.0f;
    for (int i = 0; i < h2 * h1; i++) w->decoder_w2_grad[i] = 0.0f;
    for (int i = 0; i < h1; i++) w->decoder_b2_grad[i] = 0.0f;
    for (int i = 0; i < h1 * in_d; i++) w->decoder_w3_grad[i] = 0.0f;
    for (int i = 0; i < in_d; i++) w->decoder_b3_grad[i] = 0.0f;
}

// SGD weight update: w -= lr * grad, then zero grad
__device__ void diresa_sgd_update(float* weight, float* grad, int count, float lr) {
    for (int i = 0; i < count; i++) {
        weight[i] -= lr * grad[i];
        grad[i] = 0.0f;
    }
}

__device__ void diresa_weight_update_sgd(DIRESAWeights* w) {
    float lr = w->learning_rate;
    int h1 = w->hidden1, h2 = w->hidden2;
    int in_d = w->input_dim, out_d = w->output_dim;
    diresa_sgd_update(w->encoder_w1, w->encoder_w1_grad, in_d * h1, lr);
    diresa_sgd_update(w->encoder_b1, w->encoder_b1_grad, h1, lr);
    diresa_sgd_update(w->encoder_w2, w->encoder_w2_grad, h1 * h2, lr);
    diresa_sgd_update(w->encoder_b2, w->encoder_b2_grad, h2, lr);
    diresa_sgd_update(w->encoder_w3, w->encoder_w3_grad, h2 * out_d, lr);
    diresa_sgd_update(w->encoder_b3, w->encoder_b3_grad, out_d, lr);
    diresa_sgd_update(w->decoder_w1, w->decoder_w1_grad, out_d * h2, lr);
    diresa_sgd_update(w->decoder_b1, w->decoder_b1_grad, h2, lr);
    diresa_sgd_update(w->decoder_w2, w->decoder_w2_grad, h2 * h1, lr);
    diresa_sgd_update(w->decoder_b2, w->decoder_b2_grad, h1, lr);
    diresa_sgd_update(w->decoder_w3, w->decoder_w3_grad, h1 * in_d, lr);
    diresa_sgd_update(w->decoder_b3, w->decoder_b3_grad, in_d, lr);
    w->training_step++;
}

// Per-entry DIRESA training step: forward -> recon loss -> backward -> SGD update
// features: [input_dim] input vector
// latent: [output_dim] output latent coords (written)
// Returns recon_loss MSE
__device__ float diresa_train_step(const float* features, float* latent, DIRESAWeights* weights) {
    float reconstructed[DIRESA_TASK_INPUT_DIM];  // max(input_dim) across task/hw/gen
    float recon_grad[DIRESA_TASK_INPUT_DIM];
    float latent_grad[BEHAVIORAL_DIM_TASK];      // max(output_dim) across task/hw/gen
    float features_grad[DIRESA_TASK_INPUT_DIM];

    DEVICE_FATAL_IF(weights->input_dim > DIRESA_TASK_INPUT_DIM,
        "diresa_train_step: input_dim %d > DIRESA_TASK_INPUT_DIM %d", weights->input_dim, DIRESA_TASK_INPUT_DIM);
    DEVICE_FATAL_IF(weights->output_dim > BEHAVIORAL_DIM_TASK,
        "diresa_train_step: output_dim %d > BEHAVIORAL_DIM_TASK %d", weights->output_dim, BEHAVIORAL_DIM_TASK);

    // Forward: encode + decode
    diresa_encode(features, latent, weights);
    diresa_decode(latent, reconstructed, weights);

    // Reconstruction loss: MSE
    float mse = 0.0f;
    for (int i = 0; i < weights->input_dim; i++) {
        float diff = features[i] - reconstructed[i];
        mse += diff * diff;
        recon_grad[i] = 2.0f * diff / weights->input_dim;  // d(MSE)/d(reconstructed) = -2*(features-recon)/N, but we want to minimize so grad is negative of loss grad
    }
    mse /= weights->input_dim;

    // Negate recon_grad: loss = mean((features - recon)^2), d_loss/d_recon = -2*(features - recon)/N
    for (int i = 0; i < weights->input_dim; i++) {
        recon_grad[i] = -recon_grad[i];
    }

    // Backward through decoder: recon_grad -> latent_grad, accumulates decoder weight grads
    diresa_zero_grads(weights);
    diresa_decode_backward(latent, recon_grad, latent_grad, weights);

    // Backward through encoder: latent_grad -> features_grad, accumulates encoder weight grads
    diresa_encode_backward(features, latent_grad, features_grad, weights);

    // SGD update
    diresa_weight_update_sgd(weights);

    return mse;
}

__device__ void diresa_forward_device(DIRESABatch* batch, DIRESAWeights* weights) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch->batch_size) {
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
}

__device__ void diresa_distance_device(DIRESABatch* batch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch->batch_size) {
        int shuffled_idx = batch->shuffle_indices[tid];

        const float* features_i = batch->features + tid * batch->input_dim;
        const float* features_j = batch->features + shuffled_idx * batch->input_dim;
        const float* latent_i = batch->latent + tid * batch->output_dim;
        const float* latent_j = batch->latent + shuffled_idx * batch->output_dim;

        float orig_dist_sq = 0.0f;
        for (int k = 0; k < batch->input_dim; k++) {
            float diff = features_i[k] - features_j[k];
            orig_dist_sq += diff * diff;
        }
        batch->orig_distances[tid] = sqrtf(orig_dist_sq);

        float latent_dist_sq = DIRESAOps::compute_latent_distance_sq(latent_i, latent_j, batch->output_dim);
        batch->latent_distances[tid] = sqrtf(latent_dist_sq);
    }
}

__device__ void diresa_loss_device(DIRESABatch* batch, const DIRESAWeights* weights) {
    __shared__ float shared_recon[256];
    __shared__ float shared_orig_mean[1];
    __shared__ float shared_orig_var[1];
    __shared__ float shared_latent_mean[1];
    __shared__ float shared_latent_var[1];
    __shared__ float shared_cov_sum[1];
    __shared__ int diresa_error_flag;  // 0 = ok, nonzero = error code

    int tid = threadIdx.x;
    int sample_idx = blockIdx.x * blockDim.x + tid;

    if (tid == 0) diresa_error_flag = 0;

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
    cg::this_grid().sync();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_recon[tid] += shared_recon[tid + stride];
        }
        cg::this_grid().sync();
    }
    if (tid == 0) {
        atomicAdd(&batch->recon_loss, shared_recon[0] / batch->batch_size);
    }

    if (blockIdx.x == 0) {
        float target_alpha = weights->distance_exponent;

        if (tid == 0) {
            float sum = 0.0f;
            bool valid = true;
            for (int i = 0; i < batch->batch_size && valid; i++) {
                if (batch->orig_distances[i] <= 0.0f) {
                    diresa_error_flag = 1;  // orig_distances <= 0
                    valid = false;
                } else {
                    sum += logf(batch->orig_distances[i]);
                }
            }
            shared_orig_mean[0] = valid ? (sum / batch->batch_size) : 0.0f;
        }
        cg::this_grid().sync();

        if (tid == 0 && diresa_error_flag == 0) {
            float sum_sq = 0.0f;
            float mean = shared_orig_mean[0];
            bool valid = true;
            for (int i = 0; i < batch->batch_size && valid; i++) {
                if (batch->orig_distances[i] <= 0.0f) {
                    diresa_error_flag = 2;  // orig_distances <= 0 (second check)
                    valid = false;
                } else {
                    float diff = logf(batch->orig_distances[i]) - mean;
                    sum_sq += diff * diff;
                }
            }
            if (valid) shared_orig_var[0] = sum_sq / batch->batch_size;
        }

        if (tid == 1 && diresa_error_flag == 0) {
            float sum = 0.0f;
            bool valid = true;
            for (int i = 0; i < batch->batch_size && valid; i++) {
                if (batch->latent_distances[i] <= 0.0f) {
                    diresa_error_flag = 3;  // latent_distances <= 0
                    valid = false;
                } else {
                    sum += logf(batch->latent_distances[i]);
                }
            }
            if (valid) shared_latent_mean[0] = sum / batch->batch_size;
        }
        cg::this_grid().sync();

        float latent_mean = shared_latent_mean[0];
        float orig_mean = shared_orig_mean[0];

        float local_var = 0.0f;
        float local_cov = 0.0f;

        if (diresa_error_flag == 0) {
            for (int i = tid; i < batch->batch_size; i += blockDim.x) {
                if (batch->latent_distances[i] <= 0.0f || batch->orig_distances[i] <= 0.0f) {
                    atomicCAS(&diresa_error_flag, 0, 4);  // distances <= 0 in loop
                } else {
                    float latent_diff = logf(batch->latent_distances[i]) - latent_mean;
                    float orig_diff = logf(batch->orig_distances[i]) - orig_mean;
                    local_var += latent_diff * latent_diff;
                    local_cov += latent_diff * orig_diff;
                }
            }
        }

        shared_recon[tid] = local_var;
        shared_cov_sum[0] = 0.0f;
        cg::this_grid().sync();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_recon[tid] += shared_recon[tid + stride];
            }
            cg::this_grid().sync();
        }
        if (tid == 0) {
            shared_latent_var[0] = shared_recon[0] / batch->batch_size;
        }

        shared_recon[tid] = local_cov;
        cg::this_grid().sync();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_recon[tid] += shared_recon[tid + stride];
            }
            cg::this_grid().sync();
        }
        if (tid == 0 && diresa_error_flag == 0) {
            shared_cov_sum[0] = shared_recon[0];

            float alpha_denom = shared_orig_var[0] * batch->batch_size;
            if (alpha_denom <= 0.0f) {
                diresa_error_flag = 5;  // alpha_denom <= 0
            } else {
                float alpha_measured = shared_cov_sum[0] / alpha_denom;

                float corr_denom = sqrtf(shared_orig_var[0] * shared_latent_var[0]) * batch->batch_size;
                if (corr_denom <= 0.0f || isnan(corr_denom) || isinf(corr_denom)) {
                    diresa_error_flag = 6;  // corr_denom invalid
                } else {
                    float log_correlation = shared_cov_sum[0] / corr_denom;

                    float exponent_loss = (alpha_measured - target_alpha) * (alpha_measured - target_alpha);
                    float quality_loss = 1.0f - fabsf(log_correlation);
                    batch->dist_loss = exponent_loss + weights->quality_weight * quality_loss;
                }
            }
        }
    }

    if (blockIdx.x == 0 && tid == 0) {
        float latent_means[BEHAVIORAL_DIM_TASK] = {0};  // TASK is the largest single dimension

        for (int dim = 0; dim < batch->output_dim; dim++) {
            float sum = 0.0f;
            for (int i = 0; i < batch->batch_size; i++) {
                sum += batch->latent[i * batch->output_dim + dim];
            }
            latent_means[dim] = sum / batch->batch_size;
        }

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

        int num_pairs = batch->output_dim * (batch->output_dim - 1) / 2;
        batch->cov_loss = cov_sum / num_pairs;
    }

    // Check error flag after all syncs - trap loudly if there was an error
    __syncthreads();
    if (diresa_error_flag != 0 && tid == 0 && blockIdx.x == 0) {
        printf("!E:DIRESA_LOSS code=%d batch_size=%d\n", diresa_error_flag, batch->batch_size);
    }
}

__device__ void update_annealing(DIRESAWeights* weights, float cov_loss, PoolEntry* entry) {
    if (cov_loss > entry->cov_target && weights->cov_weight < 10.0f) {
        weights->cov_weight += entry->anneal_step;
    }
}

__device__ void replica_exchange_device(DIRESAWeights* replicas, DIRESABatch* batches, PoolEntry* entry, curandState* rand_states) {
    int tid = threadIdx.x;
    if (tid < entry->num_tempering_replicas - 1) {
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

        float delta = (beta_j - beta_i) * (E_i - E_j);
        float accept_prob = fminf(1.0f, expf(delta));

        float rand = validated_curand_uniform(&rand_states[tid], "replica_exchange", tid);
        if (rand < accept_prob) {
            float temp_swap = replicas[i].temperature;
            replicas[i].temperature = replicas[j].temperature;
            replicas[j].temperature = temp_swap;
        }
    }
}

#endif
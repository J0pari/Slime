#ifndef DIRESA_CU
#define DIRESA_CU

#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
#include "../utils/genome_params.cuh"
#include "diresa_types.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

struct PoolEntry;

struct DIRESABatch {
    int input_dim;   
    int output_dim;  
    int batch_size;  

    float* features;              
    float* features_shuffled;     
    int* shuffle_indices;         

    float* latent;                
    float* latent_shuffled;       

    float* reconstructed;         

    float* orig_distances;        
    float* latent_distances;      

    float recon_loss;
    float dist_loss;
    float cov_loss;
};

__global__ void init_diresa_kernel(DIRESAWeights* replicas, float* preallocated_weight_pool, size_t replica_stride, int input_dim, int output_dim, PoolEntry* entry, unsigned int seed, float* genome) {
    int replica_id = blockIdx.x;
    int local_tid = threadIdx.x;

    if (replica_id >= entry->num_tempering_replicas) return;

    DIRESAWeights* weights = &replicas[replica_id];

    if (local_tid == 0) {
        size_t offset = replica_id * replica_stride;

        weights->encoder_w1 = &preallocated_weight_pool[offset];
        offset += input_dim * DIRESA_HIDDEN1_MAX;

        weights->encoder_b1 = &preallocated_weight_pool[offset];
        offset += DIRESA_HIDDEN1_MAX;

        weights->encoder_w2 = &preallocated_weight_pool[offset];
        offset += DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX;

        weights->encoder_b2 = &preallocated_weight_pool[offset];
        offset += DIRESA_HIDDEN2_MAX;

        weights->encoder_w3 = &preallocated_weight_pool[offset];
        offset += DIRESA_HIDDEN2_MAX * output_dim;

        weights->encoder_b3 = &preallocated_weight_pool[offset];
        offset += output_dim;

        weights->decoder_w1 = &preallocated_weight_pool[offset];
        offset += output_dim * DIRESA_HIDDEN2_MAX;

        weights->decoder_b1 = &preallocated_weight_pool[offset];
        offset += DIRESA_HIDDEN2_MAX;

        weights->decoder_w2 = &preallocated_weight_pool[offset];
        offset += DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX;

        weights->decoder_b2 = &preallocated_weight_pool[offset];
        offset += DIRESA_HIDDEN1_MAX;

        weights->decoder_w3 = &preallocated_weight_pool[offset];
        offset += DIRESA_HIDDEN1_MAX * input_dim;

        weights->decoder_b3 = &preallocated_weight_pool[offset];

        weights->input_dim = input_dim;
        weights->output_dim = output_dim;
        weights->hidden1 = entry->diresa_hidden1;
        weights->hidden2 = entry->diresa_hidden2;
    }
    __syncthreads();

    if (local_tid == 0) {
    }

    curandState state;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed + global_tid, 0, 0, &state);

    if (local_tid == 0) {
    }

    int hidden1 = weights->hidden1;
    int hidden2 = weights->hidden2;
    int in_dim = weights->input_dim;
    int out_dim = weights->output_dim;

    if (local_tid < in_dim * hidden1) {
        float scale = sqrtf(2.0f / (in_dim + hidden1));
        weights->encoder_w1[local_tid] = validated_curand_normal(&state, "diresa_init_enc1", local_tid) * scale;
    }
    if (local_tid < hidden1) {
        weights->encoder_b1[local_tid] = 0.0f;
    }

    if (local_tid < hidden1 * hidden2) {
        float scale = sqrtf(2.0f / (hidden1 + hidden2));
        weights->encoder_w2[local_tid] = validated_curand_normal(&state, "diresa_init_enc2", local_tid) * scale;
    }
    if (local_tid < hidden2) {
        weights->encoder_b2[local_tid] = 0.0f;
    }

    if (local_tid < hidden2 * out_dim) {
        float scale = sqrtf(2.0f / (hidden2 + out_dim));
        weights->encoder_w3[local_tid] = validated_curand_normal(&state, "diresa_init_enc3", local_tid) * scale;
    }
    if (local_tid < out_dim) {
        weights->encoder_b3[local_tid] = 0.0f;
    }

    if (local_tid < out_dim * hidden2) {
        float scale = sqrtf(2.0f / (out_dim + hidden2));
        weights->decoder_w1[local_tid] = validated_curand_normal(&state, "diresa_init_dec1", local_tid) * scale;
    }
    if (local_tid < hidden2) {
        weights->decoder_b1[local_tid] = 0.0f;
    }

    if (local_tid < hidden2 * hidden1) {
        float scale = sqrtf(2.0f / (hidden2 + hidden1));
        weights->decoder_w2[local_tid] = validated_curand_normal(&state, "diresa_init_dec2", local_tid) * scale;
    }
    if (local_tid < hidden1) {
        weights->decoder_b2[local_tid] = 0.0f;
    }

    if (local_tid < hidden1 * in_dim) {
        float scale = sqrtf(2.0f / (hidden1 + in_dim));
        weights->decoder_w3[local_tid] = validated_curand_normal(&state, "diresa_init_dec3", local_tid) * scale;
    }
    if (local_tid < in_dim) {
        weights->decoder_b3[local_tid] = 0.0f;
    }

    if (local_tid == 0) {
        weights->cov_weight = 0.0f;
        weights->training_step = 0;
        weights->replica_id = replica_id;
        weights->distance_exponent = entry->distance_exponent;
        weights->quality_weight = entry->quality_weight;

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
        weights->temperature = DIRESA_TEMP_BASE_MIN + temp_base * (DIRESA_TEMP_BASE_MAX - DIRESA_TEMP_BASE_MIN)
                             + replica_id * (DIRESA_TEMP_SCALE_MIN + temp_scale * (DIRESA_TEMP_SCALE_MAX - DIRESA_TEMP_SCALE_MIN));

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
            genome, entry->gradients,
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

__device__ void diresa_encode_backward(const float* features, const float* latent_grad, float* features_grad, const DIRESAWeights* weights) {
    float hidden1[DIRESA_HIDDEN1_MAX];
    float hidden2[DIRESA_HIDDEN2_MAX];
    float hidden1_grad[DIRESA_HIDDEN1_MAX];
    float hidden2_grad[DIRESA_HIDDEN2_MAX];

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

    for (int i = 0; i < weights->hidden2; i++) {
        hidden2_grad[i] = 0.0f;
    }

    for (int i = 0; i < weights->output_dim; i++) {
        float grad = latent_grad[i];
        for (int j = 0; j < weights->hidden2; j++) {
            hidden2_grad[j] += grad * weights->encoder_w3[j * weights->output_dim + i];
        }
    }

    for (int i = 0; i < weights->hidden1; i++) {
        hidden1_grad[i] = 0.0f;
    }

    for (int i = 0; i < weights->hidden2; i++) {
        float grad = hidden2_grad[i] * (hidden2[i] > 0.0f ? 1.0f : 0.0f);
        for (int j = 0; j < weights->hidden1; j++) {
            hidden1_grad[j] += grad * weights->encoder_w2[j * weights->hidden2 + i];
        }
    }

    for (int i = 0; i < weights->input_dim; i++) {
        features_grad[i] = 0.0f;
    }

    for (int i = 0; i < weights->hidden1; i++) {
        float grad = hidden1_grad[i] * (hidden1[i] > 0.0f ? 1.0f : 0.0f);
        for (int j = 0; j < weights->input_dim; j++) {
            features_grad[j] += grad * weights->encoder_w1[j * weights->hidden1 + i];
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

__global__ void diresa_distance_kernel(DIRESABatch* batch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch->batch_size) return;

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

__global__ void diresa_loss_kernel(DIRESABatch* batch, const DIRESAWeights* weights) {
    __shared__ float shared_recon[256];
    __shared__ float shared_orig_mean[1];
    __shared__ float shared_orig_var[1];
    __shared__ float shared_latent_mean[1];
    __shared__ float shared_latent_var[1];
    __shared__ float shared_cov_sum[1];

    int tid = threadIdx.x;
    int sample_idx = blockIdx.x * blockDim.x + tid;

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

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_recon[tid] += shared_recon[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&batch->recon_loss, shared_recon[0] / batch->batch_size);
    }

    if (blockIdx.x == 0) {
        float target_alpha = weights->distance_exponent;

        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < batch->batch_size; i++) {
                if (batch->orig_distances[i] <= 0.0f) {
                    return;
                }
                sum += logf(batch->orig_distances[i]);
            }
            shared_orig_mean[0] = sum / batch->batch_size;
        }
        __syncthreads();

        if (tid == 0) {
            float sum_sq = 0.0f;
            float mean = shared_orig_mean[0];
            for (int i = 0; i < batch->batch_size; i++) {
                if (batch->orig_distances[i] <= 0.0f) {
                    return;
                }
                float diff = logf(batch->orig_distances[i]) - mean;
                sum_sq += diff * diff;
            }
            shared_orig_var[0] = sum_sq / batch->batch_size;
        }

        if (tid == 1) {
            float sum = 0.0f;
            for (int i = 0; i < batch->batch_size; i++) {
                if (batch->latent_distances[i] <= 0.0f) {
                    return;
                }
                sum += logf(batch->latent_distances[i]);
            }
            shared_latent_mean[0] = sum / batch->batch_size;
        }
        __syncthreads();

        float latent_mean = shared_latent_mean[0];
        float orig_mean = shared_orig_mean[0];

        float local_var = 0.0f;
        float local_cov = 0.0f;

        for (int i = tid; i < batch->batch_size; i += blockDim.x) {
            if (batch->latent_distances[i] <= 0.0f || batch->orig_distances[i] <= 0.0f) {
                return;
            }
            float latent_diff = logf(batch->latent_distances[i]) - latent_mean;
            float orig_diff = logf(batch->orig_distances[i]) - orig_mean;
            local_var += latent_diff * latent_diff;
            local_cov += latent_diff * orig_diff;
        }

        shared_recon[tid] = local_var;
        shared_cov_sum[0] = 0.0f;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_recon[tid] += shared_recon[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_latent_var[0] = shared_recon[0] / batch->batch_size;
        }

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

            float alpha_denom = shared_orig_var[0] * batch->batch_size;
            if (alpha_denom <= 0.0f) {
                return;
            }
            float alpha_measured = shared_cov_sum[0] / alpha_denom;

            float corr_denom = sqrtf(shared_orig_var[0] * shared_latent_var[0]) * batch->batch_size;
            if (corr_denom <= 0.0f || isnan(corr_denom) || isinf(corr_denom)) {
                return;
            }
            float log_correlation = shared_cov_sum[0] / corr_denom;

            float exponent_loss = (alpha_measured - target_alpha) * (alpha_measured - target_alpha);
            float quality_loss = 1.0f - fabsf(log_correlation);
            batch->dist_loss = exponent_loss + weights->quality_weight * quality_loss;
        }
    }

    if (blockIdx.x == 0 && tid == 0) {
        float latent_means[BEHAVIORAL_DIM_MAX] = {0};

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
}

__device__ void update_annealing(DIRESAWeights* weights, float cov_loss, PoolEntry* entry) {
    if (cov_loss > entry->cov_target && weights->cov_weight < 10.0f) {
        weights->cov_weight += entry->anneal_step;
    }
}

__global__ void replica_exchange_kernel(DIRESAWeights* replicas, DIRESABatch* batches, PoolEntry* entry, curandState* rand_states) {
    int tid = threadIdx.x;
    if (tid >= entry->num_tempering_replicas - 1) return;

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

#endif
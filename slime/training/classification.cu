
#ifndef CLASSIFICATION_CU
#define CLASSIFICATION_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include "../utils/cuda_primitives.cuh"
#include "training_types.cu"
#include "losses.cu"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void spatial_pooling_device(Organism* organism) {
    float* ca_output = organism->ca_output;
    float* pooling_weights = organism->cls_pooling_weights;
    float* features = organism->cls_features;
    int batch_size = organism->cls_batch_size;
    int num_heads = organism->cls_num_heads;
    int grid_size = organism->ca_grid_size;
    int channels = organism->ca_channels;
    int batch_idx = blockIdx.x;
    int feature_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int num_features = num_heads * channels;

    if (batch_idx >= batch_size || feature_idx >= num_features) return;

    int head = feature_idx / channels;
    int channel = feature_idx % channels;

    float sum = 0.0f;
    int spatial_size = grid_size * grid_size;

    int batch_stride = num_heads * spatial_size * channels;
    int head_stride = spatial_size * channels;

    int base_idx = batch_idx * batch_stride + head * head_stride;

    for (int spatial = 0; spatial < spatial_size; spatial++) {
        int idx = base_idx + spatial * channels + channel;
        sum += ldg_float(&ca_output[idx]);
    }

    float avg = sum / spatial_size;
    float weight = ldg_float(&pooling_weights[feature_idx]);
    float weighted = avg * weight;

    DEVICE_FATAL_IF(isnan(weighted), "spatial_pooling: weighted result is NaN - CA output or weights corrupted");
    DEVICE_FATAL_IF(isinf(weighted), "spatial_pooling: weighted result is Inf - CA output or weights corrupted");

    features[batch_idx * num_features + feature_idx] = weighted;
}

__device__ void classification_head_device(Organism* organism) {
    float* features = organism->cls_features;
    float* fc_weights = organism->cls_fc_weights;
    float* fc_bias = organism->cls_fc_bias;
    float* logits = organism->cls_logits;
    int batch_size = organism->cls_batch_size;
    int num_features = organism->cls_num_features;
    int num_classes = organism->cls_num_classes;
    int batch_idx = blockIdx.x;
    int class_idx = threadIdx.x;

    if (batch_idx >= batch_size || class_idx >= num_classes) return;

    DEVICE_FATAL_IF(features == nullptr, "classification_head: features is null");
    DEVICE_FATAL_IF(fc_weights == nullptr, "classification_head: fc_weights is null");
    DEVICE_FATAL_IF(fc_bias == nullptr, "classification_head: fc_bias is null");

    if (batch_idx == 0 && class_idx == 0) {
        DEVICE_FATAL_IF(isnan(features[0]), "classification_head: features[0] is NaN - data corrupted");
        DEVICE_FATAL_IF(isnan(fc_weights[0]), "classification_head: fc_weights[0] is NaN - weights corrupted");
        DEVICE_FATAL_IF(isnan(fc_bias[0]), "classification_head: fc_bias[0] is NaN - bias corrupted");
    }

    __shared__ float dot_products[NUM_CLASSES_MAX];

    float acc = ldg_float(&fc_bias[class_idx]);

    for (int feat = 0; feat < num_features; feat++) {
        float feature_val = ldg_float(&features[batch_idx * num_features + feat]);
        float weight = ldg_float(&fc_weights[class_idx * num_features + feat]);
        acc += feature_val * weight;
    }

    if (class_idx < num_classes) {
        dot_products[class_idx] = acc;
    }
    __syncthreads();

    if (class_idx < num_classes) {
        DEVICE_FATAL_IF(isnan(dot_products[class_idx]), "classification_head: logit is NaN - computation corrupted");
        DEVICE_FATAL_IF(isinf(dot_products[class_idx]), "classification_head: logit is Inf - computation corrupted");
        logits[batch_idx * num_classes + class_idx] = dot_products[class_idx];
    }
}

__device__ void softmax_device(Organism* organism) {
    float* logits = organism->cls_logits;
    float* probabilities = organism->cls_probabilities;
    int batch_size = organism->cls_batch_size;
    int num_classes = organism->cls_num_classes;
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    float* batch_logits = &logits[batch_idx * num_classes];
    float* batch_probs = &probabilities[batch_idx * num_classes];

    float local_val = (tid < num_classes) ? batch_logits[tid] : -INFINITY;
    float max_val = warp_reduce_max(local_val);
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    float local_exp = (tid < num_classes) ? expf(local_val - max_val) : 0.0f;
    float sum_exp = warp_reduce_sum(local_exp);
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

    if (tid < num_classes) {
        batch_probs[tid] = local_exp / sum_exp;
    }
}

__device__ void accuracy_device(Organism* organism) {
    float* logits = organism->cls_logits;
    int* labels = organism->cls_labels;
    int* correct_count = organism->cls_correct_count;
    int batch_size = organism->cls_batch_size;
    int num_classes = organism->cls_num_classes;
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    float* batch_logits = &logits[batch_idx * num_classes];
    int predicted_class = 0;
    float max_logit = batch_logits[0];

    for (int c = 1; c < num_classes; c++) {
        if (batch_logits[c] > max_logit) {
            max_logit = batch_logits[c];
            predicted_class = c;
        }
    }

    int true_label = labels[batch_idx];
    if (predicted_class == true_label) {
        atomicAdd(correct_count, 1);
    }
}

__device__ void classification_cross_entropy_device(Organism* organism) {
    float* logits = organism->cls_logits;
    int* labels = organism->cls_labels;
    float* loss_out = organism->cls_loss_out;
    float* logit_grads = organism->cls_logit_grads;
    int batch_size = organism->cls_batch_size;
    int num_classes = organism->cls_num_classes;
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    int label = labels[batch_idx];
    DEVICE_FATAL_IF(label < 0 || label >= num_classes, "classification_cross_entropy: label out of range");

    float* batch_logits = &logits[batch_idx * num_classes];

    float local_val = (tid < num_classes) ? batch_logits[tid] : -INFINITY;

    if (tid < num_classes) {
        DEVICE_FATAL_IF(isnan(local_val), "classification_cross_entropy: logit is NaN");
        DEVICE_FATAL_IF(isinf(local_val), "classification_cross_entropy: logit is Inf");
    }

    float max_logit = warp_reduce_max(local_val);
    max_logit = __shfl_sync(0xffffffff, max_logit, 0);

    float local_exp = (tid < num_classes) ? expf(local_val - max_logit) : 0.0f;
    float sum_exp = warp_reduce_sum(local_exp);
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

    float log_sum_exp = logf(sum_exp) + max_logit;

    if (tid == 0) {
        float nll = log_sum_exp - batch_logits[label];
        DEVICE_FATAL_IF(isnan(nll), "classification_cross_entropy: NLL is NaN");
        DEVICE_FATAL_IF(isinf(nll), "classification_cross_entropy: NLL is Inf");
        DEVICE_FATAL_IF(nll < 0.0f, "classification_cross_entropy: NLL is negative");
        atomicAdd(loss_out, nll / batch_size);
    }

    if (tid < num_classes) {
        float prob = local_exp / sum_exp;
        float grad = (tid == label) ? (prob - 1.0f) : prob;
        logit_grads[batch_idx * num_classes + tid] = grad / batch_size;
    }
}

__device__ void classification_head_backward_device(Organism* organism) {
    float* logit_grads = organism->cls_logit_grads;
    float* features = organism->cls_features;
    float* fc_weights = organism->cls_fc_weights;
    float* fc_weights_grad = organism->cls_fc_weights_grad;
    float* fc_bias_grad = organism->cls_fc_bias_grad;
    float* features_grad = organism->cls_features_grad;
    int batch_size = organism->cls_batch_size;
    int num_features = organism->cls_num_features;
    int num_classes = organism->cls_num_classes;
    int batch_idx = blockIdx.x;
    int class_idx = threadIdx.x;

    if (batch_idx >= batch_size || class_idx >= num_classes) return;

    float logit_grad = ldg_float(&logit_grads[batch_idx * num_classes + class_idx]);

    if (class_idx < num_classes) {
        Atomics::add_float(&fc_bias_grad[class_idx], logit_grad);
    }

    for (int feat = 0; feat < num_features; feat++) {
        float feature_val = ldg_float(&features[batch_idx * num_features + feat]);
        float weight_val = ldg_float(&fc_weights[class_idx * num_features + feat]);
        Atomics::add_float(&fc_weights_grad[class_idx * num_features + feat], logit_grad * feature_val);
        Atomics::add_float(&features_grad[batch_idx * num_features + feat], logit_grad * weight_val);
    }
}

__device__ void spatial_pooling_backward_device(Organism* organism) {
    float* features_grad = organism->cls_features_grad;
    float* ca_output = organism->ca_output;
    float* pooling_weights = organism->cls_pooling_weights;
    float* pooling_weights_grad = organism->cls_pooling_weights_grad;
    float* ca_output_grad = organism->cls_ca_output_grad;
    int batch_size = organism->cls_batch_size;
    int num_heads = organism->cls_num_heads;
    int grid_size = organism->ca_grid_size;
    int channels = organism->ca_channels;
    int batch_idx = blockIdx.x;
    int feature_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int num_features = num_heads * channels;

    if (batch_idx >= batch_size || feature_idx >= num_features) return;

    int head = feature_idx / channels;
    int channel = feature_idx % channels;

    float feat_grad = ldg_float(&features_grad[batch_idx * num_features + feature_idx]);

    DEVICE_FATAL_IF(isnan(feat_grad), "spatial_pooling_backward: features_grad is NaN - backprop corrupted");
    DEVICE_FATAL_IF(isinf(feat_grad), "spatial_pooling_backward: features_grad is Inf - backprop corrupted");

    int spatial_size = grid_size * grid_size;

    int batch_stride = num_heads * spatial_size * channels;
    int head_stride = spatial_size * channels;

    int base_idx = batch_idx * batch_stride + head * head_stride;

    float ca_avg = 0.0f;
    for (int spatial = 0; spatial < spatial_size; spatial++) {
        int idx = base_idx + spatial * channels + channel;
        ca_avg += ldg_float(&ca_output[idx]);
    }
    ca_avg /= spatial_size;

    Atomics::add_float(&pooling_weights_grad[feature_idx], feat_grad * ca_avg);

    float ca_grad_val = feat_grad * ldg_float(&pooling_weights[feature_idx]) / spatial_size;
    for (int spatial = 0; spatial < spatial_size; spatial++) {
        int idx = base_idx + spatial * channels + channel;
        ca_output_grad[idx] += ca_grad_val;
    }
}

__device__ void init_classification_head_device(Organism* organism) {
    ClassificationHead* head = organism->classification_head;
    float* pooling_weights = organism->cls_pooling_weights;
    float* fc_weights = organism->cls_fc_weights;
    float* fc_bias = organism->cls_fc_bias;
    int num_features = organism->cls_num_features;
    int num_classes = organism->cls_num_classes;
    unsigned int seed = organism->init_seed;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    head->pooling_weights = pooling_weights;
    head->fc_weights = fc_weights;
    head->fc_bias = fc_bias;

    curandState rng;
    curand_init(seed, 0, 0, &rng);

    float scale = sqrtf(2.0f / (num_features + num_classes));

    for (int i = 0; i < num_features; i++) {
        pooling_weights[i] = 1.0f / num_features;
    }

    for (int c = 0; c < num_classes; c++) {
        for (int f = 0; f < num_features; f++) {
            int idx = c * num_features + f;
            float rand_val = validated_curand_uniform(&rng, "init_classification_fc", idx);
            fc_weights[idx] = (rand_val - 0.5f) * 2.0f * scale;
        }
    }

    for (int i = 0; i < num_classes; i++) {
        fc_bias[i] = 0.0f;
    }
}

#endif


#ifndef CLASSIFICATION_CU
#define CLASSIFICATION_CU

#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
#include "training_types.cu"
#include "losses.cu"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Input layout: [batch × num_heads × grid² × channels]
// Output layout: [batch × (num_heads × channels)]
__global__ void spatial_pooling_kernel(
    float* __restrict__ ca_output,
    float* __restrict__ pooling_weights,
    float* __restrict__ features,
    int batch_size,
    int num_heads,
    int grid_size,
    int channels
) {
    int batch_idx = blockIdx.x;
    int feature_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int num_features = num_heads * channels;

    if (batch_idx >= batch_size || feature_idx >= num_features) return;

    int head = feature_idx / channels;
    int channel = feature_idx % channels;

    float sum = 0.0f;
    int spatial_size = grid_size * grid_size;

    // CA output layout: [batch × num_heads × grid² × channels]
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

    if (isnan(weighted) || isinf(weighted)) {
        return;
    }
    features[batch_idx * num_features + feature_idx] = weighted;
}

__global__ void classification_head_kernel(
    float* __restrict__ features,
    float* __restrict__ fc_weights,
    float* __restrict__ fc_bias,
    float* __restrict__ logits,
    int batch_size,
    int num_features,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    int class_idx = threadIdx.x;

    if (batch_idx >= batch_size || class_idx >= num_classes) return;

    // Validate pointers and first values
    if (batch_idx == 0 && class_idx == 0) {
        if (features == nullptr) {
            return;
        }
        if (fc_weights == nullptr) {
            return;
        }
        if (fc_bias == nullptr) {
            return;
        }
        if (isnan(features[0])) {
            return;
        }
        if (isnan(fc_weights[0])) {
            return;
        }
        if (isnan(fc_bias[0])) {
            return;
        }
    }

    __shared__ float dot_products[NUM_CLASSES_MAX];

    // Use texture cache for read-only weight/feature access
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
        if (isnan(dot_products[class_idx]) || isinf(dot_products[class_idx])) {
            return;
        }
        logits[batch_idx * num_classes + class_idx] = dot_products[class_idx];
    }
}

__global__ void softmax_kernel(
    float* __restrict__ logits,
    float* __restrict__ probabilities,
    int batch_size,
    int num_classes
) {
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

__global__ void accuracy_kernel(
    float* __restrict__ logits,
    int* __restrict__ labels,
    int* __restrict__ correct_count,
    int batch_size,
    int num_classes
) {
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

__global__ void cross_entropy_loss_kernel(
    float* __restrict__ logits,
    int* __restrict__ labels,
    float* __restrict__ loss_out,
    float* __restrict__ logit_grads,
    int batch_size,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    int label = labels[batch_idx];
    if (label < 0 || label >= num_classes) return;

    float* batch_logits = &logits[batch_idx * num_classes];

    float local_val = (tid < num_classes) ? batch_logits[tid] : -INFINITY;

    if (tid < num_classes && (isnan(local_val) || isinf(local_val))) {
        return;
    }

    float max_logit = warp_reduce_max(local_val);
    max_logit = __shfl_sync(0xffffffff, max_logit, 0);

    float local_exp = (tid < num_classes) ? expf(local_val - max_logit) : 0.0f;
    float sum_exp = warp_reduce_sum(local_exp);
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

    float log_sum_exp = logf(sum_exp) + max_logit;

    if (tid == 0) {
        float nll = log_sum_exp - batch_logits[label];
        if (!isnan(nll) && !isinf(nll) && nll >= 0.0f) {
            atomicAdd(loss_out, nll / batch_size);
        }
    }

    if (tid < num_classes) {
        float prob = local_exp / sum_exp;
        float grad = (tid == label) ? (prob - 1.0f) : prob;
        logit_grads[batch_idx * num_classes + tid] = grad / batch_size;
    }
}

__global__ void classification_head_backward_kernel(
    float* __restrict__ logit_grads,
    float* __restrict__ features,
    float* __restrict__ fc_weights,
    float* __restrict__ fc_weights_grad,
    float* __restrict__ fc_bias_grad,
    float* __restrict__ features_grad,
    int batch_size,
    int num_features,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    int class_idx = threadIdx.x;

    if (batch_idx >= batch_size || class_idx >= num_classes) return;

    // Use texture cache for read-only gradient input
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

// Input grad layout: [batch × (num_heads × channels)]
// CA output layout: [batch × num_heads × grid² × channels]
__global__ void spatial_pooling_backward_kernel(
    float* __restrict__ features_grad,
    float* __restrict__ ca_output,
    float* __restrict__ pooling_weights,
    float* __restrict__ pooling_weights_grad,
    float* __restrict__ ca_output_grad,
    int batch_size,
    int num_heads,
    int grid_size,
    int channels
) {
    int batch_idx = blockIdx.x;
    int feature_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int num_features = num_heads * channels;

    if (batch_idx >= batch_size || feature_idx >= num_features) return;

    int head = feature_idx / channels;
    int channel = feature_idx % channels;

    float feat_grad = ldg_float(&features_grad[batch_idx * num_features + feature_idx]);

    if (isnan(feat_grad) || isinf(feat_grad)) {
        return;
    }

    int spatial_size = grid_size * grid_size;

    // CA output layout: [batch × num_heads × grid² × channels]
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

__global__ void init_classification_head_kernel(
    ClassificationHead* head,
    float* pooling_weights,
    float* fc_weights,
    float* fc_bias,
    int num_features,
    int num_classes,
    unsigned int seed
) {
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

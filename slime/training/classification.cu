
#ifndef CLASSIFICATION_CU
#define CLASSIFICATION_CU

#include "../config/config.cu"
#include "../utils/tile_ops.cuh"
#include "training_types.cu"
#include "losses.cu"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void spatial_pooling_kernel(
    float* __restrict__ ca_state,
    float* __restrict__ pooling_weights,
    float* __restrict__ features,
    int batch_size,
    int grid_size,
    int channels
) {
    int batch_idx = blockIdx.x;
    int channel = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || channel >= channels) return;

    float sum = 0.0f;
    int spatial_size = grid_size * grid_size;
    int base_idx = batch_idx * spatial_size * channels;

    for (int spatial = threadIdx.x; spatial < spatial_size; spatial += blockDim.x) {
        int y = spatial / grid_size;
        int x = spatial % grid_size;
        int idx = base_idx + y * grid_size * channels + x * channels + channel;
        sum += ca_state[idx];
    }

    sum = WarpReduce<WARP_SIZE>::sum(sum);

    if ((threadIdx.x % WARP_SIZE) == 0) {
        float avg = sum / spatial_size;
        float weighted = avg * pooling_weights[channel];
        features[batch_idx * channels + channel] = weighted;
    }
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

    __shared__ float dot_products[NUM_CLASSES_MAX];

    float acc = fc_bias[class_idx];

    for (int feat = 0; feat < num_features; feat++) {
        float feature_val = features[batch_idx * num_features + feat];
        float weight = fc_weights[class_idx * num_features + feat];
        acc += feature_val * weight;
    }

    if (class_idx < num_classes) {
        dot_products[class_idx] = acc;
    }
    __syncthreads();

    if (class_idx < num_classes) {
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

    if (batch_idx >= batch_size) return;

    __shared__ float max_val;
    __shared__ float sum_exp;

    float* batch_logits = &logits[batch_idx * num_classes];
    float* batch_probs = &probabilities[batch_idx * num_classes];

    if (threadIdx.x == 0) {
        max_val = batch_logits[0];
        for (int i = 1; i < num_classes; i++) {
            max_val = fmaxf(max_val, batch_logits[i]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += expf(batch_logits[i] - max_val);
        }
    }
    __syncthreads();

    if (threadIdx.x < num_classes) {
        batch_probs[threadIdx.x] = expf(batch_logits[threadIdx.x] - max_val) / sum_exp;
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
    if (batch_idx >= batch_size) return;

    int label = labels[batch_idx];

    if (logits == nullptr || labels == nullptr || loss_out == nullptr || logit_grads == nullptr) {
        if (threadIdx.x == 0 && batch_idx == 0) {
            printf("FATAL [cross_entropy_loss]: NULL pointer detected\n");
        }
        
        return;
    }

    if (label < 0 || label >= num_classes) {
        printf("FATAL [cross_entropy_loss]: Invalid label %d at batch %d (must be 0-%d)\n",
               label, batch_idx, num_classes - 1);
        
        return;
    }

    __shared__ float max_logit;
    __shared__ float log_sum_exp;

    if (threadIdx.x == 0) {
        max_logit = logits[batch_idx * num_classes];
        for (int i = 1; i < num_classes; i++) {
            float val = logits[batch_idx * num_classes + i];
            if (isnan(val) || isinf(val)) {
                printf("FATAL [cross_entropy_loss]: Invalid logit %f at batch %d class %d\n",
                       val, batch_idx, i);
                
            }
            max_logit = fmaxf(max_logit, val);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += expf(logits[batch_idx * num_classes + i] - max_logit);
        }

        log_sum_exp = logf(sum_exp) + max_logit;
        float nll = log_sum_exp - logits[batch_idx * num_classes + label];

        if (isnan(nll) || isinf(nll) || nll < 0.0f) {
            printf("FATAL [cross_entropy_loss]: Invalid NLL %f at batch %d\n", nll, batch_idx);
            
        }

        atomicAdd(loss_out, nll / batch_size);
    }
    __syncthreads();

    int tid = threadIdx.x;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float prob = expf(logits[batch_idx * num_classes + i] - log_sum_exp);
        float grad = (i == label) ? (prob - 1.0f) : prob;
        logit_grads[batch_idx * num_classes + i] = grad / batch_size;
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

    float logit_grad = logit_grads[batch_idx * num_classes + class_idx];

    if (class_idx < num_classes) {
        atomicAdd(&fc_bias_grad[class_idx], logit_grad);
    }

    for (int feat = 0; feat < num_features; feat++) {
        float feature_val = features[batch_idx * num_features + feat];
        atomicAdd(&fc_weights_grad[class_idx * num_features + feat], logit_grad * feature_val);
        atomicAdd(&features_grad[batch_idx * num_features + feat], logit_grad * fc_weights[class_idx * num_features + feat]);
    }
}

__global__ void spatial_pooling_backward_kernel(
    float* __restrict__ features_grad,
    float* __restrict__ ca_state,
    float* __restrict__ pooling_weights,
    float* __restrict__ pooling_weights_grad,
    float* __restrict__ ca_state_grad,
    int batch_size,
    int grid_size,
    int channels
) {
    int batch_idx = blockIdx.x;
    int channel = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || channel >= channels) return;

    float feat_grad = features_grad[batch_idx * channels + channel];
    int spatial_size = grid_size * grid_size;
    int base_idx = batch_idx * spatial_size * channels;

    float ca_avg = 0.0f;
    for (int spatial = threadIdx.x; spatial < spatial_size; spatial += blockDim.x) {
        int y = spatial / grid_size;
        int x = spatial % grid_size;
        int idx = base_idx + y * grid_size * channels + x * channels + channel;
        ca_avg += ca_state[idx];
    }

    ca_avg = WarpReduce<WARP_SIZE>::sum(ca_avg);
    ca_avg /= spatial_size;

    if ((threadIdx.x % WARP_SIZE) == 0) {
        atomicAdd(&pooling_weights_grad[channel], feat_grad * ca_avg);
    }

    float ca_grad_val = feat_grad * pooling_weights[channel] / spatial_size;
    for (int spatial = threadIdx.x; spatial < spatial_size; spatial += blockDim.x) {
        int y = spatial / grid_size;
        int x = spatial % grid_size;
        int idx = base_idx + y * grid_size * channels + x * channels + channel;
        atomicAdd(&ca_state_grad[idx], ca_grad_val);
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
            fc_weights[c * num_features + f] = (curand_uniform(&rng) - 0.5f) * 2.0f * scale;
        }
    }

    for (int i = 0; i < num_classes; i++) {
        fc_bias[i] = 0.0f;
    }

    printf("[DEVICE] Classification head initialized: %d features, %d classes\n", num_features, num_classes);
}

#endif

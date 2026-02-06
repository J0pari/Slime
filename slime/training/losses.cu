
#ifndef LOSSES_CU
#define LOSSES_CU

#include "../config/config.cu"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void mse_loss_kernel(
    float* __restrict__ predictions,
    float* __restrict__ targets,
    float* __restrict__ loss_out,
    int batch_size,
    int dim
) {
    DEVICE_FATAL_IF(predictions == nullptr, "mse_loss: predictions is null");
    DEVICE_FATAL_IF(targets == nullptr, "mse_loss: targets is null");
    DEVICE_FATAL_IF(loss_out == nullptr, "mse_loss: loss_out is null");
    DEVICE_FATAL_IF(batch_size <= 0, "mse_loss: batch_size must be positive");
    DEVICE_FATAL_IF(dim <= 0, "mse_loss: dim must be positive");

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim) return;

    float pred = predictions[idx];
    float target = targets[idx];

    DEVICE_FATAL_IF(isnan(pred), "mse_loss: prediction is NaN");
    DEVICE_FATAL_IF(isnan(target), "mse_loss: target is NaN");

    float diff = pred - target;
    float squared_error = diff * diff;

    atomicAdd(loss_out, squared_error / (float)(batch_size * dim));
}

// Cross-entropy loss with numerically stable softmax
// logits: [batch_size × num_classes], labels: [batch_size] (class indices)
__global__ void cross_entropy_loss_kernel(
    float* __restrict__ logits,
    int* __restrict__ labels,
    float* __restrict__ loss_out,
    float* __restrict__ gradients,  // Output: dL/d_logits [batch_size × num_classes]
    int batch_size,
    int num_classes
) {
    DEVICE_FATAL_IF(logits == nullptr, "cross_entropy: logits is null");
    DEVICE_FATAL_IF(labels == nullptr, "cross_entropy: labels is null");
    DEVICE_FATAL_IF(loss_out == nullptr, "cross_entropy: loss_out is null");
    DEVICE_FATAL_IF(batch_size <= 0, "cross_entropy: batch_size must be positive");
    DEVICE_FATAL_IF(num_classes <= 0, "cross_entropy: num_classes must be positive");

    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch_size) return;

    int label = labels[sample];
    DEVICE_FATAL_IF(label < 0 || label >= num_classes, "cross_entropy: label out of range");

    float* sample_logits = &logits[sample * num_classes];

    // Numerically stable softmax: subtract max before exp
    float max_logit = sample_logits[0];
    for (int c = 1; c < num_classes; c++) {
        float val = sample_logits[c];
        DEVICE_FATAL_IF(isnan(val), "cross_entropy: logit is NaN");
        if (val > max_logit) max_logit = val;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        sum_exp += expf(sample_logits[c] - max_logit);
    }

    DEVICE_FATAL_IF(sum_exp <= 0.0f, "cross_entropy: sum_exp is non-positive");

    float log_sum_exp = max_logit + logf(sum_exp);
    float sample_loss = log_sum_exp - sample_logits[label];

    DEVICE_FATAL_IF(isnan(sample_loss), "cross_entropy: loss became NaN");
    DEVICE_FATAL_IF(isinf(sample_loss), "cross_entropy: loss became Inf");

    atomicAdd(loss_out, sample_loss / (float)batch_size);

    DEVICE_FATAL_IF(gradients == nullptr, "cross_entropy_kernel: gradients buffer required");
    float* sample_grads = &gradients[sample * num_classes];
    for (int c = 0; c < num_classes; c++) {
        float softmax_c = expf(sample_logits[c] - max_logit) / sum_exp;
        float grad = (softmax_c - (c == label ? 1.0f : 0.0f)) / (float)batch_size;
        sample_grads[c] = grad;
    }
}

// Cross-entropy with label smoothing
__global__ void cross_entropy_label_smoothing_kernel(
    float* __restrict__ logits,
    int* __restrict__ labels,
    float* __restrict__ loss_out,
    float* __restrict__ gradients,
    int batch_size,
    int num_classes,
    float smoothing  // e.g., 0.1 means 90% on true label, 10% spread across others
) {
    DEVICE_FATAL_IF(logits == nullptr, "cross_entropy_smooth: logits is null");
    DEVICE_FATAL_IF(labels == nullptr, "cross_entropy_smooth: labels is null");
    DEVICE_FATAL_IF(loss_out == nullptr, "cross_entropy_smooth: loss_out is null");
    DEVICE_FATAL_IF(smoothing < 0.0f || smoothing > 1.0f, "cross_entropy_smooth: smoothing must be in [0,1]");

    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch_size) return;

    int label = labels[sample];
    DEVICE_FATAL_IF(label < 0 || label >= num_classes, "cross_entropy_smooth: label out of range");

    float* sample_logits = &logits[sample * num_classes];

    // Numerically stable softmax
    float max_logit = sample_logits[0];
    for (int c = 1; c < num_classes; c++) {
        if (sample_logits[c] > max_logit) max_logit = sample_logits[c];
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        sum_exp += expf(sample_logits[c] - max_logit);
    }

    float log_sum_exp = max_logit + logf(sum_exp);

    // Smoothed target: (1-smoothing) on true label, smoothing/(num_classes) on all
    float true_weight = 1.0f - smoothing;
    float smooth_weight = smoothing / (float)num_classes;

    float sample_loss = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        float target = smooth_weight + (c == label ? true_weight : 0.0f);
        float log_prob = sample_logits[c] - log_sum_exp;
        sample_loss -= target * log_prob;
    }

    atomicAdd(loss_out, sample_loss / (float)batch_size);

    DEVICE_FATAL_IF(gradients == nullptr, "cross_entropy_label_smoothing_kernel: gradients buffer required");
    float* sample_grads = &gradients[sample * num_classes];
    for (int c = 0; c < num_classes; c++) {
        float softmax_c = expf(sample_logits[c] - max_logit) / sum_exp;
        float target = smooth_weight + (c == label ? true_weight : 0.0f);
        sample_grads[c] = (softmax_c - target) / (float)batch_size;
    }
}

#endif

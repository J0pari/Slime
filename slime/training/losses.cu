
#ifndef LOSSES_CU
#define LOSSES_CU

#include "../config/config.cu"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void cross_entropy_loss_kernel(
    float* logits,
    int* labels,
    float* loss_out,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float max_logit = -1e9f;
    for (int c = 0; c < num_classes; c++) {
        max_logit = fmaxf(max_logit, logits[idx * num_classes + c]);
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        sum_exp += expf(logits[idx * num_classes + c] - max_logit);
    }

    int true_label = labels[idx];
    float log_prob = (logits[idx * num_classes + true_label] - max_logit) - logf(sum_exp);
    float sample_loss = -log_prob;

    atomicAdd(loss_out, sample_loss / (float)batch_size);
}

__global__ void cross_entropy_backward_kernel(
    float* logits,
    int* labels,
    float* logit_grads,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float max_logit = -1e9f;
    for (int c = 0; c < num_classes; c++) {
        max_logit = fmaxf(max_logit, logits[idx * num_classes + c]);
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        sum_exp += expf(logits[idx * num_classes + c] - max_logit);
    }

    int true_label = labels[idx];
    for (int c = 0; c < num_classes; c++) {
        float softmax_c = expf(logits[idx * num_classes + c] - max_logit) / sum_exp;
        float grad = softmax_c - ((c == true_label) ? 1.0f : 0.0f);
        logit_grads[idx * num_classes + c] = grad / (float)batch_size;
    }
}

__global__ void mse_loss_kernel(
    float* predictions,
    float* targets,
    float* loss_out,
    int batch_size,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim) return;

    float diff = predictions[idx] - targets[idx];
    float squared_error = diff * diff;

    atomicAdd(loss_out, squared_error / (float)(batch_size * dim));
}

#endif

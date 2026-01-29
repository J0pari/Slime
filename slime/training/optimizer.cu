
#ifndef OPTIMIZER_CU
#define OPTIMIZER_CU

#include "../config/config.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void adam_update_kernel(
    float* __restrict__ weights,
    float* __restrict__ gradients,
    float* __restrict__ m,
    float* __restrict__ v,
    int num_params,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int timestep,
    float gradient_clip_norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_params) return;

    // FATAL on null pointers - caller must provide valid buffers
    DEVICE_FATAL_IF(weights == nullptr, "adam_update: weights is null");
    DEVICE_FATAL_IF(gradients == nullptr, "adam_update: gradients is null");
    DEVICE_FATAL_IF(m == nullptr, "adam_update: m is null");
    DEVICE_FATAL_IF(v == nullptr, "adam_update: v is null");

    float g = gradients[idx];

    // FATAL on NaN/inf gradients - indicates upstream computation error
    DEVICE_FATAL_IF(isnan(g), "adam_update: gradient is NaN - upstream computation corrupted");
    DEVICE_FATAL_IF(isinf(g), "adam_update: gradient is Inf - upstream computation corrupted");

    if (fabsf(g) > gradient_clip_norm) {
        g = copysignf(gradient_clip_norm, g);
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v[idx] / (1.0f - powf(beta2, (float)timestep));

    float denom = sqrtf(v_hat) + epsilon;
    DEVICE_FATAL_IF(isnan(denom) || isinf(denom), "adam_update: denom is NaN/Inf - numerical instability");

    float weight_delta = lr * m_hat / denom;
    weights[idx] -= weight_delta;

    DEVICE_FATAL_IF(isnan(weights[idx]), "adam_update: weight became NaN after update");
    DEVICE_FATAL_IF(isinf(weights[idx]), "adam_update: weight became Inf after update");

    gradients[idx] = 0.0f;
}

__global__ void adam_update_fp16_kernel(
    half* __restrict__ weights_fp16,
    float* __restrict__ gradients,
    float* __restrict__ m,
    float* __restrict__ v,
    int num_params,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int timestep,
    float gradient_clip_norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_params) return;

    // FATAL on null pointers - caller must provide valid buffers
    DEVICE_FATAL_IF(weights_fp16 == nullptr, "adam_update_fp16: weights_fp16 is null");
    DEVICE_FATAL_IF(gradients == nullptr, "adam_update_fp16: gradients is null");
    DEVICE_FATAL_IF(m == nullptr, "adam_update_fp16: m is null");
    DEVICE_FATAL_IF(v == nullptr, "adam_update_fp16: v is null");

    float weight = __half2float(weights_fp16[idx]);
    float g = gradients[idx];

    // FATAL on NaN/inf - indicates upstream computation error
    DEVICE_FATAL_IF(isnan(g), "adam_update_fp16: gradient is NaN - upstream computation corrupted");
    DEVICE_FATAL_IF(isinf(g), "adam_update_fp16: gradient is Inf - upstream computation corrupted");
    DEVICE_FATAL_IF(isnan(weight), "adam_update_fp16: weight is NaN - data corrupted");
    DEVICE_FATAL_IF(isinf(weight), "adam_update_fp16: weight is Inf - data corrupted");

    if (fabsf(g) > gradient_clip_norm) {
        g = copysignf(gradient_clip_norm, g);
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v[idx] / (1.0f - powf(beta2, (float)timestep));

    float denom = sqrtf(v_hat) + epsilon;
    DEVICE_FATAL_IF(isnan(denom) || isinf(denom), "adam_update_fp16: denom is NaN/Inf - numerical instability");

    weight -= lr * m_hat / denom;

    DEVICE_FATAL_IF(isnan(weight), "adam_update_fp16: weight became NaN after update");
    DEVICE_FATAL_IF(isinf(weight), "adam_update_fp16: weight became Inf after update");

    weights_fp16[idx] = __float2half(weight);
    gradients[idx] = 0.0f;
}

// Forward declare UnifiedGradientBuffer from training_types.cu
struct UnifiedGradientBuffer;

// Apply unified gradient buffer through Adam to FP16 CA weights
__global__ void adam_apply_unified_ca_grads_kernel(
    UnifiedGradientBuffer* __restrict__ grad_buf,
    half* __restrict__ perception_weights,
    half* __restrict__ interaction_weights,
    half* __restrict__ value_weights,
    float* __restrict__ adam_m,
    float* __restrict__ adam_v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int timestep,
    float gradient_clip_norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int perception_size = grad_buf->perception_size;
    int interaction_size = grad_buf->interaction_size;
    int value_size = grad_buf->value_size;
    int total = perception_size + interaction_size + value_size;

    if (idx >= total) return;

    float grad;
    half* weight_ptr;
    int adam_idx = idx;

    if (idx < perception_size) {
        grad = grad_buf->perception_grads[idx];
        weight_ptr = &perception_weights[idx];
    } else if (idx < perception_size + interaction_size) {
        int local = idx - perception_size;
        grad = grad_buf->interaction_grads[local];
        weight_ptr = &interaction_weights[local];
    } else {
        int local = idx - perception_size - interaction_size;
        grad = grad_buf->value_grads[local];
        weight_ptr = &value_weights[local];
    }

    DEVICE_FATAL_IF(isnan(grad), "adam_unified_ca: gradient is NaN");
    DEVICE_FATAL_IF(isinf(grad), "adam_unified_ca: gradient is Inf");

    if (fabsf(grad) > gradient_clip_norm) {
        grad = copysignf(gradient_clip_norm, grad);
    }

    float m_prev = adam_m[adam_idx];
    float v_prev = adam_v[adam_idx];

    float m = beta1 * m_prev + (1.0f - beta1) * grad;
    float v = beta2 * v_prev + (1.0f - beta2) * grad * grad;

    adam_m[adam_idx] = m;
    adam_v[adam_idx] = v;

    float m_hat = m / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v / (1.0f - powf(beta2, (float)timestep));

    float denom = sqrtf(v_hat) + epsilon;
    DEVICE_FATAL_IF(isnan(denom) || isinf(denom), "adam_unified_ca: denom is NaN/Inf");

    float weight = __half2float(*weight_ptr);
    weight -= lr * m_hat / denom;

    DEVICE_FATAL_IF(isnan(weight), "adam_unified_ca: weight became NaN");
    DEVICE_FATAL_IF(isinf(weight), "adam_unified_ca: weight became Inf");

    *weight_ptr = __float2half(weight);
}

// Apply unified gradient buffer through Adam to FP32 classification head weights
__global__ void adam_apply_unified_classifier_grads_kernel(
    UnifiedGradientBuffer* __restrict__ grad_buf,
    float* __restrict__ pooling_weights,
    float* __restrict__ fc_weights,
    float* __restrict__ fc_bias,
    float* __restrict__ adam_m,
    float* __restrict__ adam_v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int timestep,
    float gradient_clip_norm,
    int num_features,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int fc_weights_size = num_classes * num_features;
    int total = num_features + fc_weights_size + num_classes;

    if (idx >= total) return;

    float grad;
    float* weight_ptr;
    int adam_idx = idx;

    if (idx < num_features) {
        grad = grad_buf->pooling_weight_grads[idx];
        weight_ptr = &pooling_weights[idx];
    } else if (idx < num_features + fc_weights_size) {
        int local = idx - num_features;
        grad = grad_buf->fc_weight_grads[local];
        weight_ptr = &fc_weights[local];
    } else {
        int local = idx - num_features - fc_weights_size;
        grad = grad_buf->fc_bias_grads[local];
        weight_ptr = &fc_bias[local];
    }

    DEVICE_FATAL_IF(isnan(grad), "adam_unified_classifier: gradient is NaN");
    DEVICE_FATAL_IF(isinf(grad), "adam_unified_classifier: gradient is Inf");

    if (fabsf(grad) > gradient_clip_norm) {
        grad = copysignf(gradient_clip_norm, grad);
    }

    float m_prev = adam_m[adam_idx];
    float v_prev = adam_v[adam_idx];

    float m = beta1 * m_prev + (1.0f - beta1) * grad;
    float v = beta2 * v_prev + (1.0f - beta2) * grad * grad;

    adam_m[adam_idx] = m;
    adam_v[adam_idx] = v;

    float m_hat = m / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v / (1.0f - powf(beta2, (float)timestep));

    float denom = sqrtf(v_hat) + epsilon;
    DEVICE_FATAL_IF(isnan(denom) || isinf(denom), "adam_unified_classifier: denom is NaN/Inf");

    float weight = *weight_ptr;
    weight -= lr * m_hat / denom;

    DEVICE_FATAL_IF(isnan(weight), "adam_unified_classifier: weight became NaN");
    DEVICE_FATAL_IF(isinf(weight), "adam_unified_classifier: weight became Inf");

    *weight_ptr = weight;
}

#endif

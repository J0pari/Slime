
#ifndef OPTIMIZER_CU
#define OPTIMIZER_CU

#include "../config/config.cu"
#include <cuda_runtime.h>

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

    if (weights == nullptr || gradients == nullptr || m == nullptr || v == nullptr) {
        if (idx == 0) {
        }
        
        return;
    }

    float g = gradients[idx];

    if (isnan(g) || isinf(g)) {
        
        return;
    }

    if (fabsf(g) > gradient_clip_norm) {
        g = copysignf(gradient_clip_norm, g);
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;

    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - powf(beta1, (float)timestep));

    float v_hat = v[idx] / (1.0f - powf(beta2, (float)timestep));

    float denom = sqrtf(v_hat) + epsilon;
    if (isnan(denom) || isinf(denom)) {
        return;
    }

    float weight_delta = lr * m_hat / denom;
    weights[idx] -= weight_delta;

    if (isnan(weights[idx]) || isinf(weights[idx])) {
        return;
    }

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

    if (weights_fp16 == nullptr || gradients == nullptr || m == nullptr || v == nullptr) {
        if (idx == 0) {
        }
        
        return;
    }

    float weight = __half2float(weights_fp16[idx]);
    float g = gradients[idx];

    if (isnan(g) || isinf(g) || isnan(weight) || isinf(weight)) {
        
        return;
    }

    if (fabsf(g) > gradient_clip_norm) {
        g = copysignf(gradient_clip_norm, g);
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v[idx] / (1.0f - powf(beta2, (float)timestep));

    float denom = sqrtf(v_hat) + epsilon;
    if (isnan(denom) || isinf(denom)) {
        return;
    }

    weight -= lr * m_hat / denom;

    if (isnan(weight) || isinf(weight)) {
        return;
    }

    weights_fp16[idx] = __float2half(weight);
    gradients[idx] = 0.0f;
}

#endif

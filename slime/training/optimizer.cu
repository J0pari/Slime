
#ifndef OPTIMIZER_CU
#define OPTIMIZER_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ void adam_update_perception_device(Organism* organism) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    MultiHeadCAState* ca_state = &organism->ca_state_pool[0];
    HybridTrainingMode* training_mode = organism->training_mode;

    half* weights_fp16 = ca_state->perception_weights;
    float* gradients = ca_state->tape.grad_buffer;
    float* m = training_mode->adam_m;
    float* v = training_mode->adam_v;
    int num_params = training_mode->perception_size;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (idx >= num_params) return;

    float weight = __half2float(weights_fp16[idx]);
    float g = gradients[idx];

    if (fabsf(g) > gradient_clip_norm) {
        g = copysignf(gradient_clip_norm, g);
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v[idx] / (1.0f - powf(beta2, (float)timestep));

    weight -= lr * m_hat / (sqrtf(v_hat) + epsilon);

    weights_fp16[idx] = __float2half(weight);
    gradients[idx] = 0.0f;
}

__device__ void adam_update_interaction_device(Organism* organism) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    MultiHeadCAState* ca_state = &organism->ca_state_pool[0];
    HybridTrainingMode* training_mode = organism->training_mode;

    int perception_params = training_mode->perception_size;
    half* weights_fp16 = ca_state->interaction_weights;
    float* gradients = ca_state->tape.grad_buffer + perception_params;
    float* m = training_mode->adam_m + perception_params;
    float* v = training_mode->adam_v + perception_params;
    int num_params = training_mode->interaction_size;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (idx >= num_params) return;

    float weight = __half2float(weights_fp16[idx]);
    float g = gradients[idx];

    if (fabsf(g) > gradient_clip_norm) {
        g = copysignf(gradient_clip_norm, g);
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v[idx] / (1.0f - powf(beta2, (float)timestep));

    weight -= lr * m_hat / (sqrtf(v_hat) + epsilon);

    weights_fp16[idx] = __float2half(weight);
    gradients[idx] = 0.0f;
}

__device__ void adam_update_value_device(Organism* organism) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    MultiHeadCAState* ca_state = &organism->ca_state_pool[0];
    HybridTrainingMode* training_mode = organism->training_mode;

    int perception_params = training_mode->perception_size;
    int interaction_params = training_mode->interaction_size;
    half* weights_fp16 = ca_state->value_weights;
    float* gradients = ca_state->tape.grad_buffer + perception_params + interaction_params;
    float* m = training_mode->adam_m + perception_params + interaction_params;
    float* v = training_mode->adam_v + perception_params + interaction_params;
    int num_params = training_mode->value_size;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (idx >= num_params) return;

    float weight = __half2float(weights_fp16[idx]);
    float g = gradients[idx];

    if (fabsf(g) > gradient_clip_norm) {
        g = copysignf(gradient_clip_norm, g);
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v[idx] / (1.0f - powf(beta2, (float)timestep));

    weight -= lr * m_hat / (sqrtf(v_hat) + epsilon);

    weights_fp16[idx] = __float2half(weight);
    gradients[idx] = 0.0f;
}

__device__ void adam_update_pooling_device(Organism* organism) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    ClassificationHead* classifier = organism->classifier;
    HybridTrainingMode* training_mode = organism->training_mode;
    Architecture arch = Architecture::fromConfig();

    float* weights = classifier->pooling_weights;
    float* gradients = organism->pooling_weights_grad;
    float* m = organism->adam_m_pooling;
    float* v = organism->adam_v_pooling;
    int num_params = arch.channels;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (idx >= num_params) return;

    float g = gradients[idx];

    if (fabsf(g) > gradient_clip_norm) {
        g = copysignf(gradient_clip_norm, g);
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v[idx] / (1.0f - powf(beta2, (float)timestep));

    weights[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    gradients[idx] = 0.0f;
}

__device__ void adam_update_fc_weights_device(Organism* organism) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    ClassificationHead* classifier = organism->classifier;
    HybridTrainingMode* training_mode = organism->training_mode;
    int num_classes = organism->current_dataset->num_classes;
    int num_features = CLASSIFIER_FEATURES;

    float* weights = classifier->fc_weights;
    float* gradients = organism->fc_weights_grad;
    float* m = organism->adam_m_fc_weights;
    float* v = organism->adam_v_fc_weights;
    int num_params = num_classes * num_features;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (idx >= num_params) return;

    float g = gradients[idx];

    if (fabsf(g) > gradient_clip_norm) {
        g = copysignf(gradient_clip_norm, g);
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v[idx] / (1.0f - powf(beta2, (float)timestep));

    weights[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    gradients[idx] = 0.0f;
}

__device__ void adam_update_fc_bias_device(Organism* organism) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    ClassificationHead* classifier = organism->classifier;
    HybridTrainingMode* training_mode = organism->training_mode;
    int num_classes = organism->current_dataset->num_classes;

    float* weights = classifier->fc_bias;
    float* gradients = organism->fc_bias_grad;
    float* m = organism->adam_m_fc_bias;
    float* v = organism->adam_v_fc_bias;
    int num_params = num_classes;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (idx >= num_params) return;

    float g = gradients[idx];

    if (fabsf(g) > gradient_clip_norm) {
        g = copysignf(gradient_clip_norm, g);
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - powf(beta1, (float)timestep));
    float v_hat = v[idx] / (1.0f - powf(beta2, (float)timestep));

    weights[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    gradients[idx] = 0.0f;
}

struct UnifiedGradientBuffer;

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

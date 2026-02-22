
#ifndef OPTIMIZER_CU
#define OPTIMIZER_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ void adam_update_perception_device(Organism* organism) {
    int tid = threadIdx.x;
    int entry_idx = blockIdx.x;

    MultiHeadCAState* ca_state = &organism->ca_state_pool[entry_idx];
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;

    constexpr int ADAM_CA_ENTRY_STRIDE =
        (NUM_HEADS * CHANNELS * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * CHANNELS) +
        (NUM_CLASSES_MAX * NUM_HEADS * CHANNELS);

    half* weights_fp16 = ca_state->perception_weights;
    float* gradients = ca_state->tape.grad_buffer;
    float* m = training_mode->adam_m + entry_idx * ADAM_CA_ENTRY_STRIDE;
    float* v = training_mode->adam_v + entry_idx * ADAM_CA_ENTRY_STRIDE;
    int num_params = entry->num_heads * entry->channels * entry->head_dim;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (tid < num_params) {
        float weight = __half2float(weights_fp16[tid]);
        float g = gradients[tid];

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[tid] = beta1 * m[tid] + (1.0f - beta1) * g;
        v[tid] = beta2 * v[tid] + (1.0f - beta2) * g * g;

        float m_hat = m[tid] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[tid] / (1.0f - powf(beta2, (float)timestep));

        weight -= lr * m_hat / (sqrtf(v_hat) + epsilon);

        weights_fp16[tid] = __float2half(weight);
        gradients[tid] = 0.0f;
    }
}

__device__ void adam_update_interaction_device(Organism* organism) {
    int tid = threadIdx.x;
    int entry_idx = blockIdx.x;

    MultiHeadCAState* ca_state = &organism->ca_state_pool[entry_idx];
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;

    constexpr int ADAM_CA_ENTRY_STRIDE =
        (NUM_HEADS * CHANNELS * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * CHANNELS) +
        (NUM_CLASSES_MAX * NUM_HEADS * CHANNELS);

    int perception_params = entry->num_heads * entry->channels * entry->head_dim;
    half* weights_fp16 = ca_state->interaction_weights;
    float* gradients = ca_state->tape.grad_buffer + perception_params;
    float* entry_adam_m = training_mode->adam_m + entry_idx * ADAM_CA_ENTRY_STRIDE;
    float* entry_adam_v = training_mode->adam_v + entry_idx * ADAM_CA_ENTRY_STRIDE;
    float* m = entry_adam_m + perception_params;
    float* v = entry_adam_v + perception_params;
    int num_params = entry->num_heads * entry->head_dim * entry->head_dim;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (tid < num_params) {
        float weight = __half2float(weights_fp16[tid]);
        float g = gradients[tid];

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[tid] = beta1 * m[tid] + (1.0f - beta1) * g;
        v[tid] = beta2 * v[tid] + (1.0f - beta2) * g * g;

        float m_hat = m[tid] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[tid] / (1.0f - powf(beta2, (float)timestep));

        weight -= lr * m_hat / (sqrtf(v_hat) + epsilon);

        weights_fp16[tid] = __float2half(weight);
        gradients[tid] = 0.0f;
    }
}

__device__ void adam_update_value_device(Organism* organism) {
    int tid = threadIdx.x;
    int entry_idx = blockIdx.x;

    MultiHeadCAState* ca_state = &organism->ca_state_pool[entry_idx];
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;

    constexpr int ADAM_CA_ENTRY_STRIDE =
        (NUM_HEADS * CHANNELS * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * CHANNELS) +
        (NUM_CLASSES_MAX * NUM_HEADS * CHANNELS);

    int perception_params = entry->num_heads * entry->channels * entry->head_dim;
    int interaction_params = entry->num_heads * entry->head_dim * entry->head_dim;
    half* weights_fp16 = ca_state->value_weights;
    float* gradients = ca_state->tape.grad_buffer + perception_params + interaction_params;
    float* entry_adam_m = training_mode->adam_m + entry_idx * ADAM_CA_ENTRY_STRIDE;
    float* entry_adam_v = training_mode->adam_v + entry_idx * ADAM_CA_ENTRY_STRIDE;
    float* m = entry_adam_m + perception_params + interaction_params;
    float* v = entry_adam_v + perception_params + interaction_params;
    int num_params = entry->num_heads * entry->head_dim * entry->channels;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (tid < num_params) {
        float weight = __half2float(weights_fp16[tid]);
        float g = gradients[tid];

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[tid] = beta1 * m[tid] + (1.0f - beta1) * g;
        v[tid] = beta2 * v[tid] + (1.0f - beta2) * g * g;

        float m_hat = m[tid] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[tid] / (1.0f - powf(beta2, (float)timestep));

        weight -= lr * m_hat / (sqrtf(v_hat) + epsilon);

        weights_fp16[tid] = __float2half(weight);
        gradients[tid] = 0.0f;
    }
}

__device__ void adam_update_pooling_device(Organism* organism) {
    int tid = threadIdx.x;
    int entry_idx = blockIdx.x;

    ClassificationHead* classifier = &organism->classifier[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;
    Architecture arch = Architecture::maxBounds();

    constexpr int POOLING_ENTRY_STRIDE = NUM_HEADS * CHANNELS;

    float* weights = classifier->pooling_weights;
    float* gradients = organism->pooling_weights_grad + entry_idx * POOLING_ENTRY_STRIDE;
    float* m = organism->adam_m_pooling + entry_idx * POOLING_ENTRY_STRIDE;
    float* v = organism->adam_v_pooling + entry_idx * POOLING_ENTRY_STRIDE;
    int num_params = arch.channels;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (tid < num_params) {
        float g = gradients[tid];

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[tid] = beta1 * m[tid] + (1.0f - beta1) * g;
        v[tid] = beta2 * v[tid] + (1.0f - beta2) * g * g;

        float m_hat = m[tid] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[tid] / (1.0f - powf(beta2, (float)timestep));

        weights[tid] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
        gradients[tid] = 0.0f;
    }
}

__device__ void adam_update_fc_weights_device(Organism* organism) {
    int tid = threadIdx.x;
    int entry_idx = blockIdx.x;

    ClassificationHead* classifier = &organism->classifier[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;
    int num_classes = organism->current_dataset->descriptor->num_classes;
    int num_features = CLASSIFIER_FEATURES;

    constexpr int FC_WEIGHTS_ENTRY_STRIDE = NUM_CLASSES_MAX * NUM_HEADS * CHANNELS;

    float* weights = classifier->fc_weights;
    float* gradients = organism->fc_weights_grad + entry_idx * FC_WEIGHTS_ENTRY_STRIDE;
    float* m = organism->adam_m_fc_weights + entry_idx * FC_WEIGHTS_ENTRY_STRIDE;
    float* v = organism->adam_v_fc_weights + entry_idx * FC_WEIGHTS_ENTRY_STRIDE;
    int num_params = num_classes * num_features;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (tid < num_params) {
        float g = gradients[tid];

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[tid] = beta1 * m[tid] + (1.0f - beta1) * g;
        v[tid] = beta2 * v[tid] + (1.0f - beta2) * g * g;

        float m_hat = m[tid] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[tid] / (1.0f - powf(beta2, (float)timestep));

        weights[tid] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
        gradients[tid] = 0.0f;
    }
}

__device__ void adam_update_fc_bias_device(Organism* organism) {
    int tid = threadIdx.x;
    int entry_idx = blockIdx.x;

    ClassificationHead* classifier = &organism->classifier[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;
    int num_classes = organism->current_dataset->descriptor->num_classes;

    constexpr int FC_BIAS_ENTRY_STRIDE = NUM_CLASSES_MAX;

    float* weights = classifier->fc_bias;
    float* gradients = organism->fc_bias_grad + entry_idx * FC_BIAS_ENTRY_STRIDE;
    float* m = organism->adam_m_fc_bias + entry_idx * FC_BIAS_ENTRY_STRIDE;
    float* v = organism->adam_v_fc_bias + entry_idx * FC_BIAS_ENTRY_STRIDE;
    int num_params = num_classes;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    if (tid < num_params) {
        float g = gradients[tid];

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[tid] = beta1 * m[tid] + (1.0f - beta1) * g;
        v[tid] = beta2 * v[tid] + (1.0f - beta2) * g * g;

        float m_hat = m[tid] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[tid] / (1.0f - powf(beta2, (float)timestep));

        weights[tid] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
        gradients[tid] = 0.0f;
    }
}

struct UnifiedGradientBuffer;

__device__ void adam_apply_unified_ca_grads_device(Organism* organism) {
    UnifiedGradientBuffer* grad_buf = organism->unified_grad_buffer;
    MultiHeadCAState* ca_state = organism->multihead_ca_state;
    half* perception_weights = ca_state->perception_weights;
    half* interaction_weights = ca_state->interaction_weights;
    half* value_weights = ca_state->value_weights;
    float* adam_m = organism->adam_m;
    float* adam_v = organism->adam_v;
    float lr = organism->learning_rate;
    float beta1 = organism->adam_beta1;
    float beta2 = organism->adam_beta2;
    float epsilon = organism->adam_epsilon;
    int timestep = organism->adam_timestep;
    float gradient_clip_norm = organism->gradient_clip_norm;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int perception_size = grad_buf->perception_size;
    int interaction_size = grad_buf->interaction_size;
    int value_size = grad_buf->value_size;
    int total = perception_size + interaction_size + value_size;

    if (idx < total) {
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
}

__device__ void adam_apply_unified_classifier_grads_device(Organism* organism) {
    UnifiedGradientBuffer* grad_buf = organism->unified_grad_buffer;
    ClassificationHead* cls = organism->classification_head;
    float* pooling_weights = cls->pooling_weights;
    float* fc_weights = cls->fc_weights;
    float* fc_bias = cls->fc_bias;
    float* adam_m = organism->adam_m_classifier;
    float* adam_v = organism->adam_v_classifier;
    float lr = organism->learning_rate;
    float beta1 = organism->adam_beta1;
    float beta2 = organism->adam_beta2;
    float epsilon = organism->adam_epsilon;
    int timestep = organism->adam_timestep;
    float gradient_clip_norm = organism->gradient_clip_norm;
    int num_features = organism->cls_num_features;
    int num_classes = organism->cls_num_classes;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int fc_weights_size = num_classes * num_features;
    int total = num_features + fc_weights_size + num_classes;

    if (idx < total) {
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
}

#endif

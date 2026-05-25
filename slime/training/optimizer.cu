
#ifndef OPTIMIZER_CU
#define OPTIMIZER_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ void adam_update_perception_device(Organism* organism, int entry_idx) {
    int tid = threadIdx.x;

    MultiHeadCAState* ca_state = &organism->ca_state_pool[entry_idx];
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;

    constexpr int ADAM_CA_ENTRY_STRIDE =
        (NUM_HEADS * CHANNELS * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * HEAD_DIM) +
        (NUM_HEADS * 2 * HEAD_DIM);

    half* weights_fp16 = ca_state->perception_weights;
    float* gradients = ca_state->tape.grad_buffer;
    float* m = training_mode->adam_m + entry_idx * ADAM_CA_ENTRY_STRIDE;
    float* v = training_mode->adam_v + entry_idx * ADAM_CA_ENTRY_STRIDE;
    int num_params = entry->num_heads * entry->channels * entry->head_dim;
    int per_head_perception = entry->channels * entry->head_dim;
    int per_head_total = per_head_perception + entry->head_dim * entry->head_dim + 2 * entry->head_dim;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    for (int pi = tid; pi < num_params; pi += blockDim.x) {
        int head = pi / per_head_perception;
        int local = pi % per_head_perception;
        int grad_idx = head * per_head_total + local;

        float weight = __half2float(weights_fp16[pi]);
        float g = gradients[grad_idx];

        DEVICE_FATAL_IF(isnan(g), "adam_perception: gradient is NaN");
        DEVICE_FATAL_IF(isinf(g), "adam_perception: gradient is Inf");

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[pi] = beta1 * m[pi] + (1.0f - beta1) * g;
        v[pi] = beta2 * v[pi] + (1.0f - beta2) * g * g;

        float m_hat = m[pi] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[pi] / (1.0f - powf(beta2, (float)timestep));

        float denom = sqrtf(v_hat) + epsilon;
        DEVICE_FATAL_IF(isnan(denom) || isinf(denom), "adam_perception: denom is NaN/Inf");

        weight -= lr * m_hat / denom;

        DEVICE_FATAL_IF(isnan(weight), "adam_perception: weight became NaN");
        DEVICE_FATAL_IF(isinf(weight), "adam_perception: weight became Inf");

        weights_fp16[pi] = __float2half(weight);
        gradients[grad_idx] = 0.0f;
    }
}

__device__ void adam_update_interaction_device(Organism* organism, int entry_idx) {
    int tid = threadIdx.x;

    MultiHeadCAState* ca_state = &organism->ca_state_pool[entry_idx];
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;

    constexpr int ADAM_CA_ENTRY_STRIDE =
        (NUM_HEADS * CHANNELS * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * HEAD_DIM) +
        (NUM_HEADS * 2 * HEAD_DIM);

    int perception_params = entry->num_heads * entry->channels * entry->head_dim;
    int per_head_perception = entry->channels * entry->head_dim;
    int per_head_interaction = entry->head_dim * entry->head_dim;
    int per_head_total = per_head_perception + per_head_interaction + 2 * entry->head_dim;
    half* weights_fp16 = ca_state->interaction_weights;
    float* gradients = ca_state->tape.grad_buffer;
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

    for (int pi = tid; pi < num_params; pi += blockDim.x) {
        int head = pi / per_head_interaction;
        int local = pi % per_head_interaction;
        int grad_idx = head * per_head_total + per_head_perception + local;

        float weight = __half2float(weights_fp16[pi]);
        float g = gradients[grad_idx];

        DEVICE_FATAL_IF(isnan(g), "adam_interaction: gradient is NaN");
        DEVICE_FATAL_IF(isinf(g), "adam_interaction: gradient is Inf");

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[pi] = beta1 * m[pi] + (1.0f - beta1) * g;
        v[pi] = beta2 * v[pi] + (1.0f - beta2) * g * g;

        float m_hat = m[pi] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[pi] / (1.0f - powf(beta2, (float)timestep));

        float denom = sqrtf(v_hat) + epsilon;
        DEVICE_FATAL_IF(isnan(denom) || isinf(denom), "adam_interaction: denom is NaN/Inf");

        weight -= lr * m_hat / denom;

        DEVICE_FATAL_IF(isnan(weight), "adam_interaction: weight became NaN");
        DEVICE_FATAL_IF(isinf(weight), "adam_interaction: weight became Inf");

        weights_fp16[pi] = __float2half(weight);
        gradients[grad_idx] = 0.0f;
    }
}

__device__ void adam_update_flow_projection_device(Organism* organism, int entry_idx) {
    int tid = threadIdx.x;

    MultiHeadCAState* ca_state = &organism->ca_state_pool[entry_idx];
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;

    constexpr int ADAM_CA_ENTRY_STRIDE =
        (NUM_HEADS * CHANNELS * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * HEAD_DIM) +
        (NUM_HEADS * 2 * HEAD_DIM);

    int perception_params = entry->num_heads * entry->channels * entry->head_dim;
    int interaction_params = entry->num_heads * entry->head_dim * entry->head_dim;
    int per_head_perception = entry->channels * entry->head_dim;
    int per_head_interaction = entry->head_dim * entry->head_dim;
    int per_head_flow_projection = 2 * entry->head_dim;
    int per_head_total = per_head_perception + per_head_interaction + per_head_flow_projection;
    half* weights_fp16 = ca_state->flow_projection_weights;
    float* gradients = ca_state->tape.grad_buffer;
    float* entry_adam_m = training_mode->adam_m + entry_idx * ADAM_CA_ENTRY_STRIDE;
    float* entry_adam_v = training_mode->adam_v + entry_idx * ADAM_CA_ENTRY_STRIDE;
    float* m = entry_adam_m + perception_params + interaction_params;
    float* v = entry_adam_v + perception_params + interaction_params;
    int num_params = entry->num_heads * 2 * entry->head_dim;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    for (int pi = tid; pi < num_params; pi += blockDim.x) {
        int head = pi / per_head_flow_projection;
        int local = pi % per_head_flow_projection;
        int grad_idx = head * per_head_total + per_head_perception + per_head_interaction + local;

        float weight = __half2float(weights_fp16[pi]);
        float g = gradients[grad_idx];

        DEVICE_FATAL_IF(isnan(g), "adam_flow_projection: gradient is NaN");
        DEVICE_FATAL_IF(isinf(g), "adam_flow_projection: gradient is Inf");

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[pi] = beta1 * m[pi] + (1.0f - beta1) * g;
        v[pi] = beta2 * v[pi] + (1.0f - beta2) * g * g;

        float m_hat = m[pi] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[pi] / (1.0f - powf(beta2, (float)timestep));

        float denom = sqrtf(v_hat) + epsilon;
        DEVICE_FATAL_IF(isnan(denom) || isinf(denom), "adam_flow_projection: denom is NaN/Inf");

        weight -= lr * m_hat / denom;

        DEVICE_FATAL_IF(isnan(weight), "adam_flow_projection: weight became NaN");
        DEVICE_FATAL_IF(isinf(weight), "adam_flow_projection: weight became Inf");

        weights_fp16[pi] = __float2half(weight);
        gradients[grad_idx] = 0.0f;
    }
}

__device__ void adam_update_pooling_device(Organism* organism, int entry_idx) {
    int tid = threadIdx.x;

    ClassificationHead* classifier = &organism->classifier[entry_idx];
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;

    constexpr int POOLING_ENTRY_STRIDE = CLASSIFIER_FEATURE_DIM;

    float* weights = classifier->pooling_weights;
    float* gradients = organism->pooling_weights_grad + entry_idx * POOLING_ENTRY_STRIDE;
    float* m = organism->adam_m_pooling + entry_idx * POOLING_ENTRY_STRIDE;
    float* v = organism->adam_v_pooling + entry_idx * POOLING_ENTRY_STRIDE;
    int num_params = entry->num_heads * POOLING_NUM_TILES * entry->channels;
    float lr = training_mode->learning_rate;
    float beta1 = ADAM_BETA1;
    float beta2 = ADAM_BETA2;
    float epsilon = ADAM_EPSILON;
    int timestep = training_mode->adam_timestep + 1;
    float gradient_clip_norm = GRADIENT_CLIP_NORM;

    for (int pi = tid; pi < num_params; pi += blockDim.x) {
        float g = gradients[pi];

        DEVICE_FATAL_IF(isnan(g), "adam_pooling: gradient is NaN");
        DEVICE_FATAL_IF(isinf(g), "adam_pooling: gradient is Inf");

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[pi] = beta1 * m[pi] + (1.0f - beta1) * g;
        v[pi] = beta2 * v[pi] + (1.0f - beta2) * g * g;

        float m_hat = m[pi] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[pi] / (1.0f - powf(beta2, (float)timestep));

        float denom = sqrtf(v_hat) + epsilon;
        DEVICE_FATAL_IF(isnan(denom) || isinf(denom), "adam_pooling: denom is NaN/Inf");

        float weight = weights[pi];
        weight -= lr * m_hat / denom;

        DEVICE_FATAL_IF(isnan(weight), "adam_pooling: weight became NaN");
        DEVICE_FATAL_IF(isinf(weight), "adam_pooling: weight became Inf");

        weights[pi] = weight;
        gradients[pi] = 0.0f;
    }
}

__device__ void adam_update_fc_weights_device(Organism* organism, int entry_idx) {
    int tid = threadIdx.x;

    ClassificationHead* classifier = &organism->classifier[entry_idx];
    HybridTrainingMode* training_mode = organism->training_mode;
    int num_classes = organism->current_dataset->descriptor->num_classes;
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    int num_features = entry->num_heads * POOLING_NUM_TILES * entry->channels;

    constexpr int FC_WEIGHTS_ENTRY_STRIDE = NUM_CLASSES_MAX * CLASSIFIER_FEATURE_DIM;

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

    for (int pi = tid; pi < num_params; pi += blockDim.x) {
        float g = gradients[pi];

        DEVICE_FATAL_IF(isnan(g), "adam_fc_weights: gradient is NaN");
        DEVICE_FATAL_IF(isinf(g), "adam_fc_weights: gradient is Inf");

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[pi] = beta1 * m[pi] + (1.0f - beta1) * g;
        v[pi] = beta2 * v[pi] + (1.0f - beta2) * g * g;

        float m_hat = m[pi] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[pi] / (1.0f - powf(beta2, (float)timestep));

        float denom = sqrtf(v_hat) + epsilon;
        DEVICE_FATAL_IF(isnan(denom) || isinf(denom), "adam_fc_weights: denom is NaN/Inf");

        float weight = weights[pi];
        weight -= lr * m_hat / denom;

        DEVICE_FATAL_IF(isnan(weight), "adam_fc_weights: weight became NaN");
        DEVICE_FATAL_IF(isinf(weight), "adam_fc_weights: weight became Inf");

        weights[pi] = weight;
        gradients[pi] = 0.0f;
    }
}

__device__ void adam_update_fc_bias_device(Organism* organism, int entry_idx) {
    int tid = threadIdx.x;

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

    for (int pi = tid; pi < num_params; pi += blockDim.x) {
        float g = gradients[pi];

        DEVICE_FATAL_IF(isnan(g), "adam_fc_bias: gradient is NaN");
        DEVICE_FATAL_IF(isinf(g), "adam_fc_bias: gradient is Inf");

        if (fabsf(g) > gradient_clip_norm) {
            g = copysignf(gradient_clip_norm, g);
        }

        m[pi] = beta1 * m[pi] + (1.0f - beta1) * g;
        v[pi] = beta2 * v[pi] + (1.0f - beta2) * g * g;

        float m_hat = m[pi] / (1.0f - powf(beta1, (float)timestep));
        float v_hat = v[pi] / (1.0f - powf(beta2, (float)timestep));

        float denom = sqrtf(v_hat) + epsilon;
        DEVICE_FATAL_IF(isnan(denom) || isinf(denom), "adam_fc_bias: denom is NaN/Inf");

        float weight = weights[pi];
        weight -= lr * m_hat / denom;

        DEVICE_FATAL_IF(isnan(weight), "adam_fc_bias: weight became NaN");
        DEVICE_FATAL_IF(isinf(weight), "adam_fc_bias: weight became Inf");

        weights[pi] = weight;
        gradients[pi] = 0.0f;
    }
}

#endif

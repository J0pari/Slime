
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

__device__ void spatial_pooling_device(
    const float* __restrict__ ca_output,
    const float* __restrict__ pooling_weights,
    float* __restrict__ features,
    int batch_size, int grid_size, int num_heads, int channels,
    int tid, int block_threads
) {
    int num_features = num_heads * POOLING_NUM_TILES * channels;
    int spatial_size = grid_size * grid_size;
    int tile_cell_size = grid_size / POOLING_TILES_K;
    int tile_cells = tile_cell_size * tile_cell_size;

    int total_work = batch_size * num_features;
    for (int work_idx = tid; work_idx < total_work; work_idx += block_threads) {
        int batch_idx = work_idx / num_features;
        int feature_idx = work_idx % num_features;

        int head = feature_idx / (POOLING_NUM_TILES * channels);
        int tile_and_channel = feature_idx % (POOLING_NUM_TILES * channels);
        int tile_idx = tile_and_channel / channels;
        int channel = tile_and_channel % channels;

        int tile_x = tile_idx % POOLING_TILES_K;
        int tile_y = tile_idx / POOLING_TILES_K;

        int batch_stride = num_heads * spatial_size * channels;
        int head_stride = spatial_size * channels;
        int base_idx = batch_idx * batch_stride + head * head_stride;

        float sum = 0.0f;
        for (int dy = 0; dy < tile_cell_size; dy++) {
            for (int dx = 0; dx < tile_cell_size; dx++) {
                int cell_x = tile_x * tile_cell_size + dx;
                int cell_y = tile_y * tile_cell_size + dy;
                int spatial_idx = cell_y * grid_size + cell_x;
                sum += ca_output[base_idx + spatial_idx * channels + channel];
            }
        }
        float avg = sum / (float)tile_cells;
        float weight = pooling_weights[feature_idx];
        features[batch_idx * num_features + feature_idx] = avg * weight;
    }
}

__device__ void classification_head_device(
    const float* __restrict__ features,
    const float* __restrict__ fc_weights,
    const float* __restrict__ fc_bias,
    float* __restrict__ logits,
    int batch_size, int num_features, int num_classes,
    int tid, int block_threads
) {
    int total_work = batch_size * num_classes;
    for (int work_idx = tid; work_idx < total_work; work_idx += block_threads) {
        int batch_idx = work_idx / num_classes;
        int class_idx = work_idx % num_classes;

        float acc = fc_bias[class_idx];
        for (int feat = 0; feat < num_features; feat++) {
            float feature_val = features[batch_idx * num_features + feat];
            float weight = fc_weights[class_idx * num_features + feat];
            acc += feature_val * weight;
        }

        DEVICE_FATAL_IF(!isfinite(acc), "classification_head: logit is NaN/Inf");
        logits[batch_idx * num_classes + class_idx] = acc;
    }
}

__device__ void classification_head_backward_device(
    const float* __restrict__ logit_grads,
    const float* __restrict__ features,
    const float* __restrict__ fc_weights,
    float* __restrict__ fc_weights_grad,
    float* __restrict__ fc_bias_grad,
    float* __restrict__ features_grad,
    int batch_size, int num_features, int num_classes,
    int tid, int block_threads
) {
    int total_work = batch_size * num_classes;
    for (int work_idx = tid; work_idx < total_work; work_idx += block_threads) {
        int batch_idx = work_idx / num_classes;
        int class_idx = work_idx % num_classes;

        float logit_grad = logit_grads[batch_idx * num_classes + class_idx];
        atomicAdd(&fc_bias_grad[class_idx], logit_grad);

        for (int feat = 0; feat < num_features; feat++) {
            float feature_val = features[batch_idx * num_features + feat];
            float weight_val = fc_weights[class_idx * num_features + feat];
            atomicAdd(&fc_weights_grad[class_idx * num_features + feat], logit_grad * feature_val);
            atomicAdd(&features_grad[batch_idx * num_features + feat], logit_grad * weight_val);
        }
    }
}

__device__ void spatial_pooling_backward_device(
    const float* __restrict__ ca_output,
    const float* __restrict__ features_grad,
    const float* __restrict__ pooling_weights,
    float* __restrict__ ca_output_grad,
    float* __restrict__ pooling_weights_grad,
    int batch_size, int grid_size, int num_heads, int channels,
    int tid, int block_threads
) {
    int num_features = num_heads * POOLING_NUM_TILES * channels;
    int spatial_size = grid_size * grid_size;
    int tile_cell_size = grid_size / POOLING_TILES_K;
    int tile_cells = tile_cell_size * tile_cell_size;
    float inv_tile_cells = 1.0f / (float)tile_cells;

    int total_work = batch_size * num_features;
    for (int work_idx = tid; work_idx < total_work; work_idx += block_threads) {
        int batch_idx = work_idx / num_features;
        int feature_idx = work_idx % num_features;

        int head = feature_idx / (POOLING_NUM_TILES * channels);
        int tile_and_channel = feature_idx % (POOLING_NUM_TILES * channels);
        int tile_idx = tile_and_channel / channels;
        int channel = tile_and_channel % channels;

        float feat_grad = features_grad[batch_idx * num_features + feature_idx];

        int tile_x = tile_idx % POOLING_TILES_K;
        int tile_y = tile_idx / POOLING_TILES_K;

        int batch_stride = num_heads * spatial_size * channels;
        int head_stride = spatial_size * channels;
        int base_idx = batch_idx * batch_stride + head * head_stride;

        float grad_per_cell = feat_grad * pooling_weights[feature_idx] * inv_tile_cells;
        float sum_ca = 0.0f;

        for (int dy = 0; dy < tile_cell_size; dy++) {
            for (int dx = 0; dx < tile_cell_size; dx++) {
                int cell_x = tile_x * tile_cell_size + dx;
                int cell_y = tile_y * tile_cell_size + dy;
                int spatial_idx = cell_y * grid_size + cell_x;
                int idx = base_idx + spatial_idx * channels + channel;
                atomicAdd(&ca_output_grad[idx], grad_per_cell);
                sum_ca += ca_output[idx];
            }
        }
        float avg_ca = sum_ca * inv_tile_cells;
        atomicAdd(&pooling_weights_grad[feature_idx], feat_grad * avg_ca);
    }
}

#endif

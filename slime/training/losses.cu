
#ifndef LOSSES_CU
#define LOSSES_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include <cuda_runtime.h>

__device__ void cross_entropy_label_smoothing_device(
    const float* __restrict__ logits,
    const int* __restrict__ labels,
    float* __restrict__ logit_grads,
    float* loss_out,
    int batch_size, int num_classes,
    float smoothing,
    int tid, int block_threads
) {
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_threads / WARP_SIZE;

    float true_weight = 1.0f - smoothing;
    float smooth_weight = smoothing / (float)num_classes;

    for (int sample_base = 0; sample_base < batch_size; sample_base += num_warps) {
        int batch_idx = sample_base + warp_id;
        if (batch_idx < batch_size) {
            int label = labels[batch_idx];
            DEVICE_FATAL_IF(label < 0 || label >= num_classes, "cross_entropy_smooth: label out of range");

            const float* batch_logits = &logits[batch_idx * num_classes];
            float* batch_grads = &logit_grads[batch_idx * num_classes];

            float local_val = (lane_id < num_classes) ? batch_logits[lane_id] : -INFINITY;
            float max_logit = warp_reduce_max(local_val);
            max_logit = __shfl_sync(0xffffffff, max_logit, 0);

            float local_exp = (lane_id < num_classes) ? expf(local_val - max_logit) : 0.0f;
            float sum_exp = warp_reduce_sum(local_exp);
            sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

            if (lane_id < num_classes) {
                float prob = local_exp / sum_exp;
                float target = smooth_weight + ((lane_id == label) ? true_weight : 0.0f);
                batch_grads[lane_id] = (prob - target) / batch_size;
            }

            if (lane_id == 0) {
                float log_sum_exp = logf(sum_exp) + max_logit;
                float sample_loss = 0.0f;
                for (int c = 0; c < num_classes; c++) {
                    float target_c = smooth_weight + ((c == label) ? true_weight : 0.0f);
                    float log_prob = batch_logits[c] - log_sum_exp;
                    sample_loss -= target_c * log_prob;
                }
                DEVICE_FATAL_IF(!isfinite(sample_loss), "cross_entropy_smooth: loss is NaN/Inf");
                atomicAdd(loss_out, sample_loss / batch_size);
            }
        }
    }
}

#endif

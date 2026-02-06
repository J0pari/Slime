#ifndef FLOW_LENIA_OPS_CUH
#define FLOW_LENIA_OPS_CUH

#include "../utils/cuda_primitives.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct FlowLeniaOps {
    __device__ static float reduce_affinity_warp(
        const float* __restrict__ ca_output,
        int cell_idx,
        int grid_size,
        int num_heads,
        int head_dim
    ) {
        int lane = threadIdx.x % 32;
        float affinity = 0.0f;

        int total_elements = num_heads * head_dim;
        for (int i = lane; i < total_elements; i += 32) {
            int head = i / head_dim;
            int dim = i % head_dim;
            int idx = head * grid_size * grid_size * head_dim + cell_idx * head_dim + dim;
            affinity += ldg_float(&ca_output[idx]);
        }

        return WarpReduce<32>::sum(affinity);
    }

    __device__ static float2 compute_flow_at(
        float U_center,
        float U_E, float U_N,
        float A_sum_center,
        float A_sum_E, float A_sum_N,
        float beta_A,
        float n
    ) {
        float ratio = A_sum_center / fmaxf(fabsf(beta_A), safe_epsilon(beta_A));
        float safe_ratio = fmaxf(ratio, safe_epsilon(ratio));
        float safe_n = fmaxf(fabsf(n), safe_epsilon(n));
        float alpha = powf(safe_ratio, safe_n);
        alpha = clamp(alpha, 0.0f, 1.0f);

        float grad_U_x = (U_E - U_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_U_y = (U_N - U_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_A_x = (A_sum_E - A_sum_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_A_y = (A_sum_N - A_sum_center) * CENTERED_DIFFERENCE_SCALE;

        return make_float2(
            (1.0f - alpha) * grad_U_x - alpha * grad_A_x,
            (1.0f - alpha) * grad_U_y - alpha * grad_A_y
        );
    }

    __device__ static void bilinear_splat_weights(
        float cx, float cy,
        int grid_size,
        int* x0, int* y0, int* x1, int* y1,
        float* w00, float* w10, float* w01, float* w11
    ) {
        *x0 = (int)floorf(cx);
        *y0 = (int)floorf(cy);
        *x1 = *x0 + 1;
        *y1 = *y0 + 1;

        float fx = cx - (float)*x0;
        float fy = cy - (float)*y0;

        *w00 = (1.0f - fx) * (1.0f - fy);
        *w10 = fx * (1.0f - fy);
        *w01 = (1.0f - fx) * fy;
        *w11 = fx * fy;

        *x0 = max(0, min(*x0, grid_size - 1));
        *y0 = max(0, min(*y0, grid_size - 1));
        *x1 = max(0, min(*x1, grid_size - 1));
        *y1 = max(0, min(*y1, grid_size - 1));
    }

    __device__ static void bilinear_transport_forward(
        float source_mass,
        float source_x, float source_y,
        float flow_x, float flow_y,
        float dt,
        int grid_size,
        float* __restrict__ target_buffer,
        int channel_offset,
        int channels
    ) {
        float displaced_x = source_x + dt * flow_x;
        float displaced_y = source_y + dt * flow_y;

        int x0, y0, x1, y1;
        float w00, w10, w01, w11;
        bilinear_splat_weights(displaced_x, displaced_y, grid_size,
                               &x0, &y0, &x1, &y1, &w00, &w10, &w01, &w11);

        int idx00 = (y0 * grid_size + x0) * channels + channel_offset;
        int idx10 = (y0 * grid_size + x1) * channels + channel_offset;
        int idx01 = (y1 * grid_size + x0) * channels + channel_offset;
        int idx11 = (y1 * grid_size + x1) * channels + channel_offset;

        atomicAdd(&target_buffer[idx00], source_mass * w00);
        atomicAdd(&target_buffer[idx10], source_mass * w10);
        atomicAdd(&target_buffer[idx01], source_mass * w01);
        atomicAdd(&target_buffer[idx11], source_mass * w11);
    }

    __device__ static void bilinear_transport_backward(
        float source_mass,
        float source_x, float source_y,
        float flow_x, float flow_y,
        float dt,
        int grid_size,
        const float* __restrict__ target_grad,
        float* d_source_mass,
        float* d_flow_x,
        float* d_flow_y,
        int channel_offset,
        int channels
    ) {
        float displaced_x = source_x + dt * flow_x;
        float displaced_y = source_y + dt * flow_y;

        int x0 = (int)floorf(displaced_x);
        int y0 = (int)floorf(displaced_y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float fx = displaced_x - (float)x0;
        float fy = displaced_y - (float)y0;

        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx * fy;

        int cx0 = max(0, min(x0, grid_size - 1));
        int cy0 = max(0, min(y0, grid_size - 1));
        int cx1 = max(0, min(x1, grid_size - 1));
        int cy1 = max(0, min(y1, grid_size - 1));

        int idx00 = (cy0 * grid_size + cx0) * channels + channel_offset;
        int idx10 = (cy0 * grid_size + cx1) * channels + channel_offset;
        int idx01 = (cy1 * grid_size + cx0) * channels + channel_offset;
        int idx11 = (cy1 * grid_size + cx1) * channels + channel_offset;

        float g00 = target_grad[idx00];
        float g10 = target_grad[idx10];
        float g01 = target_grad[idx01];
        float g11 = target_grad[idx11];

        *d_source_mass = w00 * g00 + w10 * g10 + w01 * g01 + w11 * g11;

        float d_fx = source_mass * (-(1.0f - fy) * g00 + (1.0f - fy) * g10 - fy * g01 + fy * g11);
        float d_fy = source_mass * (-(1.0f - fx) * g00 - fx * g10 + (1.0f - fx) * g01 + fx * g11);

        *d_flow_x = dt * d_fx;
        *d_flow_y = dt * d_fy;
    }

    __device__ static float soft_clamp(float x, float min_val, float max_val, float sharpness) {
        float range = max_val - min_val;
        float safe_range = fmaxf(fabsf(range), safe_epsilon(range));
        float scaled = (x - min_val) / safe_range;
        float exp_arg = -sharpness * (scaled - CENTERED_DIFFERENCE_SCALE);
        exp_arg = fmaxf(fminf(exp_arg, EXPF_ARG_LIMIT), -EXPF_ARG_LIMIT);
        float sigmoid_val = 1.0f / (1.0f + expf(exp_arg));
        return min_val + range * sigmoid_val;
    }

    __device__ static float2 compute_flow_differentiable(
        float U_center,
        float U_E, float U_N,
        float A_sum_center,
        float A_sum_E, float A_sum_N,
        float beta_A,
        float n,
        float alpha_min,
        float alpha_max,
        float sharpness
    ) {
        float ratio = A_sum_center / fmaxf(fabsf(beta_A), safe_epsilon(beta_A));
        float safe_ratio = fmaxf(ratio, safe_epsilon(ratio));
        float safe_n = fmaxf(fabsf(n), safe_epsilon(n));
        float alpha = powf(safe_ratio, safe_n);
        alpha = soft_clamp(alpha, alpha_min, alpha_max, sharpness);

        float grad_U_x = (U_E - U_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_U_y = (U_N - U_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_A_x = (A_sum_E - A_sum_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_A_y = (A_sum_N - A_sum_center) * CENTERED_DIFFERENCE_SCALE;

        return make_float2(
            (1.0f - alpha) * grad_U_x - alpha * grad_A_x,
            (1.0f - alpha) * grad_U_y - alpha * grad_A_y
        );
    }

    __device__ static void compute_flow_backward(
        float d_flow_x, float d_flow_y,
        float U_center, float U_E, float U_N,
        float A_sum_center, float A_sum_E, float A_sum_N,
        float beta_A, float n,
        float alpha_min, float alpha_max,
        float sharpness,
        float* d_beta_A,
        float* d_n
    ) {
        float ratio = A_sum_center / fmaxf(fabsf(beta_A), safe_epsilon(beta_A));
        float safe_ratio = fmaxf(ratio, safe_epsilon(ratio));
        float safe_n = fmaxf(fabsf(n), safe_epsilon(n));

        float alpha_unclamped = powf(safe_ratio, safe_n);

        float range = alpha_max - alpha_min;
        float safe_range = fmaxf(fabsf(range), safe_epsilon(range));
        float scaled = (alpha_unclamped - alpha_min) / safe_range;
        float exp_arg = -sharpness * (scaled - CENTERED_DIFFERENCE_SCALE);
        exp_arg = fmaxf(fminf(exp_arg, EXPF_ARG_LIMIT), -EXPF_ARG_LIMIT);
        float exp_val = expf(exp_arg);
        float sigmoid_val = 1.0f / (1.0f + exp_val);
        float d_sigmoid = sigmoid_val * (1.0f - sigmoid_val) * sharpness / safe_range;

        float alpha = alpha_min + range * sigmoid_val;

        float grad_U_x = (U_E - U_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_U_y = (U_N - U_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_A_x = (A_sum_E - A_sum_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_A_y = (A_sum_N - A_sum_center) * CENTERED_DIFFERENCE_SCALE;

        float d_alpha_x = d_flow_x * (-grad_U_x - grad_A_x);
        float d_alpha_y = d_flow_y * (-grad_U_y - grad_A_y);
        float d_alpha = d_alpha_x + d_alpha_y;

        float d_alpha_unclamped = d_alpha * range * d_sigmoid;

        float d_ratio = d_alpha_unclamped * safe_n * powf(safe_ratio, safe_n - 1.0f);
        float d_n_local = d_alpha_unclamped * alpha_unclamped * logf(safe_ratio);

        float beta_A_safe = fmaxf(fabsf(beta_A), safe_epsilon(beta_A));
        *d_beta_A = d_ratio * (-A_sum_center / (beta_A_safe * beta_A_safe));

        *d_n = (fabsf(n) > safe_epsilon(n)) ? d_n_local : 0.0f;
    }
};



#endif

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
        DEVICE_VALIDATE_FINITE(U_center);
        DEVICE_VALIDATE_FINITE(A_sum_center);
        DEVICE_VALIDATE_POSITIVE_DEFINITE(fabsf(beta_A) + safe_epsilon(1.0f));

        float ratio = A_sum_center / fmaxf(fabsf(beta_A), safe_epsilon(beta_A));
        float safe_ratio = fmaxf(ratio, safe_epsilon(ratio));
        float safe_n = fmaxf(fabsf(n), safe_epsilon(n));
        float alpha = powf(safe_ratio, safe_n);
        alpha = clamp(alpha, 0.0f, 1.0f);

        float grad_U_x = (U_E - U_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_U_y = (U_N - U_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_A_x = (A_sum_E - A_sum_center) * CENTERED_DIFFERENCE_SCALE;
        float grad_A_y = (A_sum_N - A_sum_center) * CENTERED_DIFFERENCE_SCALE;

        float flow_x = (1.0f - alpha) * grad_U_x - alpha * grad_A_x;
        float flow_y = (1.0f - alpha) * grad_U_y - alpha * grad_A_y;

        DEVICE_VALIDATE_FLOW_LENIA(A_sum_center, flow_x, flow_y);

        return make_float2(flow_x, flow_y);
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

        int x0 = (int)floorf(displaced_x);
        int y0 = (int)floorf(displaced_y);
        float tx = displaced_x - (float)x0;
        float ty = displaced_y - (float)y0;

        float4 w = Interpolation::bilinear_weights(tx, ty);

        int cx0 = max(0, min(x0, grid_size - 1));
        int cy0 = max(0, min(y0, grid_size - 1));
        int cx1 = max(0, min(x0 + 1, grid_size - 1));
        int cy1 = max(0, min(y0 + 1, grid_size - 1));

        int idx_tl = (cy0 * grid_size + cx0) * channels + channel_offset;
        int idx_tr = (cy0 * grid_size + cx1) * channels + channel_offset;
        int idx_bl = (cy1 * grid_size + cx0) * channels + channel_offset;
        int idx_br = (cy1 * grid_size + cx1) * channels + channel_offset;

        atomicAdd(&target_buffer[idx_tl], source_mass * w.x);
        atomicAdd(&target_buffer[idx_tr], source_mass * w.y);
        atomicAdd(&target_buffer[idx_bl], source_mass * w.z);
        atomicAdd(&target_buffer[idx_br], source_mass * w.w);
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
        float tx = displaced_x - (float)x0;
        float ty = displaced_y - (float)y0;

        float4 w = Interpolation::bilinear_weights(tx, ty);
        float4 dw_dtx, dw_dty;
        Interpolation::bilinear_weight_grads(tx, ty, &dw_dtx, &dw_dty);

        int cx0 = max(0, min(x0, grid_size - 1));
        int cy0 = max(0, min(y0, grid_size - 1));
        int cx1 = max(0, min(x0 + 1, grid_size - 1));
        int cy1 = max(0, min(y0 + 1, grid_size - 1));

        int idx_tl = (cy0 * grid_size + cx0) * channels + channel_offset;
        int idx_tr = (cy0 * grid_size + cx1) * channels + channel_offset;
        int idx_bl = (cy1 * grid_size + cx0) * channels + channel_offset;
        int idx_br = (cy1 * grid_size + cx1) * channels + channel_offset;

        float g_tl = target_grad[idx_tl];
        float g_tr = target_grad[idx_tr];
        float g_bl = target_grad[idx_bl];
        float g_br = target_grad[idx_br];

        *d_source_mass = w.x * g_tl + w.y * g_tr + w.z * g_bl + w.w * g_br;

        float d_tx = source_mass * (dw_dtx.x * g_tl + dw_dtx.y * g_tr + dw_dtx.z * g_bl + dw_dtx.w * g_br);
        float d_ty = source_mass * (dw_dty.x * g_tl + dw_dty.y * g_tr + dw_dty.z * g_bl + dw_dty.w * g_br);

        *d_flow_x = dt * d_tx;
        *d_flow_y = dt * d_ty;
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

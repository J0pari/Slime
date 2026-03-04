#ifndef FLOW_LENIA_OPS_CUH
#define FLOW_LENIA_OPS_CUH

#include "../config/config.cu"
#include "organism.cu"
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
        int lane = threadIdx.x % WARP_SIZE;
        float affinity = 0.0f;

        int total_elements = num_heads * head_dim;
        for (int i = lane; i < total_elements; i += WARP_SIZE) {
            int head = i / head_dim;
            int dim = i % head_dim;
            int idx = head * grid_size * grid_size * head_dim + cell_idx * head_dim + dim;
            affinity += ldg_float(&ca_output[idx]);
        }

        return WarpReduce<WARP_SIZE>::sum(affinity);
    }

    __device__ static float soft_clamp(float x, float min_val, float max_val, float sharpness) {
        float range = max_val - min_val;
        float safe_range = fmaxf(fabsf(range), safe_epsilon(range));
        float scaled = (x - min_val) / safe_range;
        float exp_arg = -sharpness * (scaled - CENTERED_DIFFERENCE_SCALE);
        exp_arg = fmaxf(fminf(exp_arg, EXPF_ARG_LIMIT), -EXPF_ARG_LIMIT);
        float sigmoid_val = activation_sigmoid(-exp_arg);
        return min_val + range * sigmoid_val;
    }

    // ============ Flow field computation from concentration gradients ============
    // Used for chemical field transport (not CA state update)

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

    // ============ Flow projection: learned head_dim → 2D ============
    // Projects head_dim-dimensional interaction output to 2D flow via learned weights
    // weights layout: [2 × head_dim] stored as half (row 0 = flow_x, row 1 = flow_y)

    __device__ static float2 project_to_flow(
        const float* interaction,
        int head_dim,
        const half* flow_projection_weights
    ) {
        float fx = 0.0f;
        float fy = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float val = interaction[d];
            fx += val * __half2float(flow_projection_weights[d]);
            fy += val * __half2float(flow_projection_weights[head_dim + d]);
        }
        return make_float2(fx, fy);
    }

    __device__ static void project_to_flow_backward(
        float d_flow_x, float d_flow_y,
        const float* interaction,
        int head_dim,
        const half* flow_projection_weights,
        float* d_interaction,
        float* d_weights
    ) {
        for (int d = 0; d < head_dim; d++) {
            float wx = __half2float(flow_projection_weights[d]);
            float wy = __half2float(flow_projection_weights[head_dim + d]);
            d_interaction[d] += wx * d_flow_x + wy * d_flow_y;
            d_weights[d] += interaction[d] * d_flow_x;
            d_weights[head_dim + d] += interaction[d] * d_flow_y;
        }
    }

    // ============ Mass-conserving bilinear transport ============
    // Transports all channels of a cell together along a single flow vector.
    // gate scales dt: effective_dt = gate * base_dt (gate=0 → stationary, gate=1 → full displacement).
    // Bilinear weights sum to 1.0 — mass is conserved by construction.
    // Target buffer must be zeroed before use (accumulates via atomicAdd).

    __device__ static void bilinear_transport_forward(
        const float* __restrict__ source_state,
        int cell_idx,
        float2 flow,
        float gate,
        float base_dt,
        int grid_size,
        float* __restrict__ target_buffer,
        int channels,
        int batch_cell_offset
    ) {
        float dt = gate * base_dt;
        int cell_x = cell_idx % grid_size;
        int cell_y = cell_idx / grid_size;
        float displaced_x = (float)cell_x + dt * flow.x;
        float displaced_y = (float)cell_y + dt * flow.y;

        int x0 = (int)floorf(displaced_x);
        int y0 = (int)floorf(displaced_y);
        float tx = displaced_x - (float)x0;
        float ty = displaced_y - (float)y0;

        float4 w = Interpolation::bilinear_weights(tx, ty);

        int cx0 = max(0, min(x0, grid_size - 1));
        int cy0 = max(0, min(y0, grid_size - 1));
        int cx1 = max(0, min(x0 + 1, grid_size - 1));
        int cy1 = max(0, min(y0 + 1, grid_size - 1));

        int idx_tl = batch_cell_offset + (cy0 * grid_size + cx0) * channels;
        int idx_tr = batch_cell_offset + (cy0 * grid_size + cx1) * channels;
        int idx_bl = batch_cell_offset + (cy1 * grid_size + cx0) * channels;
        int idx_br = batch_cell_offset + (cy1 * grid_size + cx1) * channels;

        int src_base = batch_cell_offset + cell_idx * channels;
        for (int c = 0; c < channels; c++) {
            float mass = source_state[src_base + c];
            atomicAdd(&target_buffer[idx_tl + c], mass * w.x);
            atomicAdd(&target_buffer[idx_tr + c], mass * w.y);
            atomicAdd(&target_buffer[idx_bl + c], mass * w.z);
            atomicAdd(&target_buffer[idx_br + c], mass * w.w);
        }
    }

    __device__ static void bilinear_transport_backward(
        const float* __restrict__ source_state,
        int cell_idx,
        float2 flow,
        float gate,
        float base_dt,
        int grid_size,
        const float* __restrict__ target_grad,
        float* d_source_state,
        float* d_flow_x,
        float* d_flow_y,
        float* d_gate,
        int channels,
        int batch_cell_offset
    ) {
        float dt = gate * base_dt;
        int cell_x = cell_idx % grid_size;
        int cell_y = cell_idx / grid_size;
        float displaced_x = (float)cell_x + dt * flow.x;
        float displaced_y = (float)cell_y + dt * flow.y;

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

        int idx_tl = batch_cell_offset + (cy0 * grid_size + cx0) * channels;
        int idx_tr = batch_cell_offset + (cy0 * grid_size + cx1) * channels;
        int idx_bl = batch_cell_offset + (cy1 * grid_size + cx0) * channels;
        int idx_br = batch_cell_offset + (cy1 * grid_size + cx1) * channels;

        int src_base = batch_cell_offset + cell_idx * channels;
        float accum_d_tx = 0.0f;
        float accum_d_ty = 0.0f;
        for (int c = 0; c < channels; c++) {
            float mass = source_state[src_base + c];
            float g_tl = target_grad[idx_tl + c];
            float g_tr = target_grad[idx_tr + c];
            float g_bl = target_grad[idx_bl + c];
            float g_br = target_grad[idx_br + c];

            // d_source_mass[c]: how loss changes w.r.t. this cell's mass in channel c
            d_source_state[c] = w.x * g_tl + w.y * g_tr + w.z * g_bl + w.w * g_br;

            accum_d_tx += mass * (dw_dtx.x * g_tl + dw_dtx.y * g_tr + dw_dtx.z * g_bl + dw_dtx.w * g_br);
            accum_d_ty += mass * (dw_dty.x * g_tl + dw_dty.y * g_tr + dw_dty.z * g_bl + dw_dty.w * g_br);
        }

        *d_flow_x = dt * accum_d_tx;
        *d_flow_y = dt * accum_d_ty;
        *d_gate = base_dt * (flow.x * accum_d_tx + flow.y * accum_d_ty);
    }
};

#endif

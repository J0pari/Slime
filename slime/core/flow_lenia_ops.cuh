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
        float alpha = powf(A_sum_center / beta_A, n);
        alpha = clamp(alpha, 0.0f, 1.0f);

        float grad_U_x = (U_E - U_center) * 0.5f;
        float grad_U_y = (U_N - U_center) * 0.5f;
        float grad_A_x = (A_sum_E - A_sum_center) * 0.5f;
        float grad_A_y = (A_sum_N - A_sum_center) * 0.5f;

        return make_float2(
            (1.0f - alpha) * grad_U_x - alpha * grad_A_x,
            (1.0f - alpha) * grad_U_y - alpha * grad_A_y
        );
    }

    __device__ static float gaussian_overlap_integral(
        float cx, float cy,
        float s,
        int tx, int ty
    ) {
        float x_min = (float)tx;
        float x_max = (float)(tx + 1);
        float y_min = (float)ty;
        float y_max = (float)(ty + 1);

        float integral = 0.0f;
        const int samples = 3;
        float dx = (x_max - x_min) / (float)samples;
        float dy = (y_max - y_min) / (float)samples;

        float norm = 1.0f / (2.0f * M_PI * s * s);

        for (int sy = 0; sy < samples; sy++) {
            for (int sx = 0; sx < samples; sx++) {
                float x = x_min + (sx + 0.5f) * dx;
                float y = y_min + (sy + 0.5f) * dy;
                float r_sq = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                integral += norm * expf(-r_sq / (2.0f * s * s));
            }
        }

        integral *= dx * dy;
        return integral;
    }
};

#endif

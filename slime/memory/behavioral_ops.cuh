#ifndef BEHAVIORAL_OPS_CUH
#define BEHAVIORAL_OPS_CUH

#include "../config/config.cu"

struct GPUElite;
struct VoronoiCell;

__device__ __forceinline__ float compute_three_axis_distance_sq(
    const float* hw_coords_a, const float* task_coords_a, const float* gen_coords_a,
    const float* hw_coords_b, const float* task_coords_b, const float* gen_coords_b,
    int hw_dim, int task_dim, int gen_dim
) {
    float dist_sq = 0.0f;

    for (int d = 0; d < hw_dim; d++) {
        float diff = hw_coords_a[d] - hw_coords_b[d];
        dist_sq += diff * diff;
    }

    for (int d = 0; d < task_dim; d++) {
        float diff = task_coords_a[d] - task_coords_b[d];
        dist_sq += diff * diff;
    }

    for (int d = 0; d < gen_dim; d++) {
        float diff = gen_coords_a[d] - gen_coords_b[d];
        dist_sq += diff * diff;
    }

    return dist_sq;
}

__device__ __forceinline__ float elite_to_cell_distance_sq(const GPUElite* elite, const VoronoiCell* cell);

__device__ __forceinline__ void concatenate_behavioral_coords(
    float* out_coords,
    const float* hw_coords,
    const float* task_coords,
    const float* gen_coords,
    int hw_dim, int task_dim, int gen_dim
) {
    for (int d = 0; d < hw_dim; d++) {
        out_coords[d] = hw_coords[d];
    }
    for (int d = 0; d < task_dim; d++) {
        out_coords[hw_dim + d] = task_coords[d];
    }
    for (int d = 0; d < gen_dim; d++) {
        out_coords[hw_dim + task_dim + d] = gen_coords[d];
    }
}

__device__ __forceinline__ void split_behavioral_coords(
    const float* coords,
    float* hw_coords,
    float* task_coords,
    float* gen_coords,
    int hw_dim, int task_dim, int gen_dim
) {
    for (int d = 0; d < hw_dim; d++) {
        hw_coords[d] = coords[d];
    }
    for (int d = 0; d < task_dim; d++) {
        task_coords[d] = coords[hw_dim + d];
    }
    for (int d = 0; d < gen_dim; d++) {
        gen_coords[d] = coords[hw_dim + task_dim + d];
    }
}

#endif

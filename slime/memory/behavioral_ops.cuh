#ifndef BEHAVIORAL_OPS_CUH
#define BEHAVIORAL_OPS_CUH

#include "../config/config.cu"

struct GPUElite;
struct VoronoiCell;

__device__ __forceinline__ float compute_three_axis_distance_sq(
    const float* hw_coords_a, const float* task_coords_a, const float* gen_coords_a,
    const float* hw_coords_b, const float* task_coords_b, const float* gen_coords_b
) {
    float dist_sq = 0.0f;

    for (int d = 0; d < BEHAVIORAL_DIM_HW_MAX; d++) {
        float diff = hw_coords_a[d] - hw_coords_b[d];
        dist_sq += diff * diff;
    }

    for (int d = 0; d < BEHAVIORAL_DIM_TASK_MAX; d++) {
        float diff = task_coords_a[d] - task_coords_b[d];
        dist_sq += diff * diff;
    }

    for (int d = 0; d < BEHAVIORAL_DIM_GEN_MAX; d++) {
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
    const float* gen_coords
) {
    for (int d = 0; d < BEHAVIORAL_DIM_HW_MAX; d++) {
        out_coords[d] = hw_coords[d];
    }
    for (int d = 0; d < BEHAVIORAL_DIM_TASK_MAX; d++) {
        out_coords[BEHAVIORAL_DIM_HW_MAX + d] = task_coords[d];
    }
    for (int d = 0; d < BEHAVIORAL_DIM_GEN_MAX; d++) {
        out_coords[BEHAVIORAL_DIM_HW_MAX + BEHAVIORAL_DIM_TASK_MAX + d] = gen_coords[d];
    }
}

__device__ __forceinline__ void split_behavioral_coords(
    const float* coords,
    float* hw_coords,
    float* task_coords,
    float* gen_coords
) {
    for (int d = 0; d < BEHAVIORAL_DIM_HW_MAX; d++) {
        hw_coords[d] = coords[d];
    }
    for (int d = 0; d < BEHAVIORAL_DIM_TASK_MAX; d++) {
        task_coords[d] = coords[BEHAVIORAL_DIM_HW_MAX + d];
    }
    for (int d = 0; d < BEHAVIORAL_DIM_GEN_MAX; d++) {
        gen_coords[d] = coords[BEHAVIORAL_DIM_HW_MAX + BEHAVIORAL_DIM_TASK_MAX + d];
    }
}

#endif

// Task-conditioning witness for A201.task-conditioning-complete.
//
// The NCA advertises TASK_EMBED_DIM = 16 but only task dimensions 0..4 are
// copied into channels 6..10; dimensions 5..15 reach no part of the forward.
// This witness perturbs ONLY task_embedding[12] and requires the descriptor
// to change. It currently FAILS, which keeps the claim red in the generated
// status until the 16->5 projection (or another explicit conditioning path)
// exists.
//
// Build: make task-conditioning-test
// Expected exit status: 1 (witness failing, by design, until the invariant
// is implemented).

#include "../autodiff/warp_tape.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace slime;
using namespace slime::autodiff;
using namespace slime::nca;

static void fill_weights(float* w, int n, uint32_t seed) {
    uint32_t s = seed ? seed : 0x9E3779B9u;
    for (int i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        float u = static_cast<float>(s) * (1.0f / 4294967296.0f);
        w[i] = (u - 0.5f) * 0.2f;
    }
}

static int check_task_conditioning() {
    // [claim:A201.task-conditioning-complete]
    const int N = 1;
    OrganismState* d_org = nullptr;
    ForwardInputs* d_inputs = nullptr;
    float* d_weights = nullptr;
    __half* d_img = nullptr;
    float* d_task = nullptr;

    if (cudaMalloc(&d_org, sizeof(OrganismState) * N) != cudaSuccess) return 1;
    if (cudaMalloc(&d_inputs, sizeof(ForwardInputs) * N) != cudaSuccess) return 1;
    if (cudaMalloc(&d_weights, sizeof(float) * TOTAL_WEIGHTS) != cudaSuccess) return 1;
    if (cudaMalloc(&d_img, sizeof(__half) * GRID_SIZE * GRID_SIZE * 3) != cudaSuccess) return 1;
    if (cudaMalloc(&d_task, sizeof(float) * TASK_EMBED_DIM) != cudaSuccess) return 1;

    float* h_weights = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS);
    fill_weights(h_weights, TOTAL_WEIGHTS, 42u);
    if (cudaMemcpy(d_weights, h_weights, sizeof(float) * TOTAL_WEIGHTS,
                   cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    __half* h_img = (__half*)malloc(sizeof(__half) * GRID_SIZE * GRID_SIZE * 3);
    for (int i = 0; i < GRID_SIZE * GRID_SIZE * 3; ++i)
        h_img[i] = __float2half(static_cast<float>((i * 37) % 64) / 64.0f);
    if (cudaMemcpy(d_img, h_img, sizeof(__half) * GRID_SIZE * GRID_SIZE * 3,
                   cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    float h_task[TASK_EMBED_DIM];
    for (int i = 0; i < TASK_EMBED_DIM; ++i) h_task[i] = 0.1f * static_cast<float>(i + 1);
    if (cudaMemcpy(d_task, h_task, sizeof(float) * TASK_EMBED_DIM,
                   cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    ForwardInputs h_in;
    h_in.role = Role::Classifier;
    h_in.task_embedding = d_task;
    h_in.image_rgb = d_img;
    h_in.target_bmap_32 = nullptr;
    if (cudaMemcpy(d_inputs, &h_in, sizeof(ForwardInputs),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    auto descriptor = [&](float* out) -> int {
        if (cudaMemcpy(out,
                       (char*)d_org + offsetof(OrganismState, bmap_traj) +
                           (BTRAJ_SAMPLES - 1) * BMAP_DIM * sizeof(float),
                       sizeof(float) * BMAP_DIM, cudaMemcpyDeviceToHost) != cudaSuccess)
            return 1;
        return 0;
    };

    launch_forward(d_org, d_inputs, nullptr,
                   &d_weights[OFF_PERC], &d_weights[OFF_INTER],
                   &d_weights[OFF_FLOW], &d_weights[OFF_BMAP], RESIDUAL_ALPHA, N, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    float desc0[BMAP_DIM];
    if (descriptor(desc0)) return 1;

    // Perturb ONLY task dimension 12 (outside the 0..4 range that reaches
    // channels 6..10).
    h_task[12] += 0.37f;
    if (cudaMemcpy(d_task, h_task, sizeof(float) * TASK_EMBED_DIM,
                   cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    launch_forward(d_org, d_inputs, nullptr,
                   &d_weights[OFF_PERC], &d_weights[OFF_INTER],
                   &d_weights[OFF_FLOW], &d_weights[OFF_BMAP], RESIDUAL_ALPHA, N, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    float desc1[BMAP_DIM];
    if (descriptor(desc1)) return 1;

    float max_diff = 0.f;
    for (int d = 0; d < BMAP_DIM; ++d) {
        float diff = std::fabs(desc0[d] - desc1[d]);
        if (diff > max_diff) max_diff = diff;
    }
    std::printf("max |descriptor(T) - descriptor(T')| with only task[12] perturbed = %.6e\n",
                max_diff);

    free(h_weights);
    free(h_img);
    cudaFree(d_org);
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_img);
    cudaFree(d_task);

    return (max_diff > 1e-6f) ? 0 : 1;
}

int main() {
    std::printf("Task-conditioning witness (A201.task-conditioning-complete)\n");
    std::fflush(stdout);
    int rc = check_task_conditioning();
    if (rc == 0) {
        std::printf("TASK CONDITIONING: PASS\n");
    } else {
        std::printf("TASK CONDITIONING: FAIL - task embedding dims 5..15 do not "
                    "affect the NCA output\n");
    }
    return rc;
}


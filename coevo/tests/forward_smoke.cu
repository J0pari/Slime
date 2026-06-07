// Stage-1 forward smoke test (A-201). FIRST GPU BUILD TARGET — UNRUN.
//
// This is the minimal thing to run on a real CUDA device: allocate one
// classifier organism, fill the substrate weights deterministically, run the
// 64-step forward pass, and check the resulting BTRAJ is finite and not a flat
// constant (i.e. the CA actually produced spatial structure). It does NOT
// validate correctness of the dynamics — only that the kernel compiles,
// launches, completes without an error or NaN, and produces non-degenerate
// output. That is the Stage-1 acceptance bar in docs/construction_plan.md.
//
// Build + run (needs nvcc + an NVIDIA GPU):
//   make forward-smoke
// or directly:
//   nvcc -std=c++17 -arch=sm_86 -rdc=true --extended-lambda \
//        --expt-relaxed-constexpr tests/forward_smoke.cu -o build/forward_smoke \
//        -lcudadevrt && ./build/forward_smoke
//
// Reaction-diffusion is passed as null here on purpose: the seeding zeroes the
// chemical channels and rd_step has no source term, so a non-null coeffs array
// would evolve an all-zero field to an all-zero field and change nothing. RD is
// wired (see forward_kernel) but functionally dormant until something seeds the
// chemical field; testing it needs a non-zero chemical seed that the spec does
// not yet define.

#include "../nca/engine.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>

using slime::nca::OrganismState;
using slime::nca::ForwardInputs;

// Deterministic small weights from a seed (xorshift32, same generator family as
// the genome RNG). Range ~[-0.1, 0.1] to keep the 64-step rollout bounded.
static void fill_weights(float* w, int n, uint32_t seed) {
    uint32_t s = seed ? seed : 0x9E3779B9u;
    for (int i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        float u = static_cast<float>(s) * (1.0f / 4294967296.0f);  // [0,1)
        w[i] = (u - 0.5f) * 0.2f;
    }
}

int main() {
    const int N = 1;

    OrganismState* org = nullptr;
    ForwardInputs* in  = nullptr;
    float*  task  = nullptr;
    __half* img   = nullptr;
    float*  W_inter = nullptr;
    float*  W_flow  = nullptr;
    float*  W_bmap  = nullptr;

    const int n_inter = slime::nca::PERC_DIM * slime::nca::HIDDEN_DIM;
    const int n_flow  = slime::nca::HIDDEN_DIM * CA_OUT_CHANNELS;
    const int n_bmap  = CA_CHANNELS * BMAP_DIM;
    const int n_img   = GRID_SIZE * GRID_SIZE * 3;

    bool alloc_ok = true;
    alloc_ok &= cudaMallocManaged(&org, sizeof(OrganismState) * N) == cudaSuccess;
    alloc_ok &= cudaMallocManaged(&in,  sizeof(ForwardInputs) * N) == cudaSuccess;
    alloc_ok &= cudaMallocManaged(&task, sizeof(float) * TASK_EMBED_DIM) == cudaSuccess;
    alloc_ok &= cudaMallocManaged(&img,  sizeof(__half) * n_img) == cudaSuccess;
    alloc_ok &= cudaMallocManaged(&W_inter, sizeof(float) * n_inter) == cudaSuccess;
    alloc_ok &= cudaMallocManaged(&W_flow,  sizeof(float) * n_flow) == cudaSuccess;
    alloc_ok &= cudaMallocManaged(&W_bmap,  sizeof(float) * n_bmap) == cudaSuccess;
    if (!alloc_ok) { std::printf("FAIL: cudaMallocManaged\n"); return 1; }

    for (int i = 0; i < TASK_EMBED_DIM; ++i) task[i] = 0.1f * static_cast<float>(i + 1);
    for (int i = 0; i < n_img; ++i) img[i] = __float2half(static_cast<float>((i * 37) % 64) / 64.0f);
    fill_weights(W_inter, n_inter, 1234u);
    fill_weights(W_flow,  n_flow,  5678u);
    fill_weights(W_bmap,  n_bmap,  9012u);

    org[0].role = Role::Classifier;        // Role is a global enum (constants.cuh)
    in[0].role            = Role::Classifier;
    in[0].task_embedding  = task;
    in[0].image_rgb       = img;
    in[0].target_bmap_32  = nullptr;

    slime::nca::launch_forward(org, in, /*coeffs=*/nullptr,
                               W_inter, W_flow, W_bmap, N, /*stream=*/0);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::printf("FAIL: launch/sync: %s\n", cudaGetErrorString(e)); return 1; }

    int nonfinite = 0;
    bool all_same = true;
    const float first = org[0].bmap_traj[0];
    for (int i = 0; i < BTRAJ_SAMPLES * BMAP_DIM; ++i) {
        float v = org[0].bmap_traj[i];
        if (!std::isfinite(v)) nonfinite++;
        if (std::fabs(v - first) > 1e-6f) all_same = false;
    }

    std::printf("bmap_64[0..7] =");
    for (int i = 0; i < 8; ++i) std::printf(" %.4f", org[0].bmap_traj[3 * BMAP_DIM + i]);
    std::printf("\n");

    if (nonfinite) { std::printf("FAIL: %d non-finite BTRAJ entries\n", nonfinite); return 1; }
    if (all_same)  { std::printf("FAIL: BTRAJ is constant (CA produced no structure)\n"); return 1; }
    std::printf("PASS: forward ran; BTRAJ finite and non-constant\n");
    return 0;
}

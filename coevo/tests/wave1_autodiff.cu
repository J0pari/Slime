// Wave 1 Acceptance Test
//
// Per construction_plan.md Wave 1 acceptance criteria:
// 1. Allocate via cudaMalloc. Forward with checkpoints matches forward_kernel.
// 2. Backward produces finite nonzero gradients in all 4 weight groups.
// 3. Stencil adjoint produces nonzero d_state contributions.
// 4. 5 iterations of forward→backward→aggregate→CAME: loss decreases.
// 5. POOL_SIZE=4 gradient aggregation averages correctly.
//
// Build: make wave1-test

#include "../optimizer/came.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace slime;
using namespace slime::autodiff;
using namespace slime::optimizer;
using namespace slime::nca;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::printf("FAIL: %s (line %d)\n", msg, __LINE__); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::printf("CUDA FAIL: %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

// Deterministic weight fill (same as forward_smoke.cu).
static void fill_weights(float* w, int n, uint32_t seed) {
    uint32_t s = seed ? seed : 0x9E3779B9u;
    for (int i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        float u = static_cast<float>(s) * (1.0f / 4294967296.0f);
        w[i] = (u - 0.5f) * 0.2f;
    }
}

// ---- Test 1+2: Forward match + backward nonzero --------------------------
static int test_forward_match_and_backward() {
    std::printf("--- Test: forward_with_checkpoints matches forward_kernel ---\n");
    std::fflush(stdout);

    const int N = 1;

    // Host-side weight buffer.
    float* h_weights = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS);
    fill_weights(h_weights, TOTAL_WEIGHTS, 42u);

    // Device allocations.
    OrganismState* d_org_fwd = nullptr;   // for forward_kernel (reference)
    OrganismState* d_org_ckpt = nullptr;  // for forward_with_checkpoints
    ForwardInputs* d_inputs = nullptr;
    float* d_weights = nullptr;
    __half* d_img = nullptr;
    float* d_task = nullptr;
    CheckpointBuffer* d_ckpt = nullptr;
    GradBuffers* d_grads = nullptr;
    float* d_seed_grad = nullptr;

    CUDA_CHECK(cudaMalloc(&d_org_fwd,  sizeof(OrganismState) * N));
    CUDA_CHECK(cudaMalloc(&d_org_ckpt, sizeof(OrganismState) * N));
    CUDA_CHECK(cudaMalloc(&d_inputs,   sizeof(ForwardInputs) * N));
    CUDA_CHECK(cudaMalloc(&d_weights,  sizeof(float) * TOTAL_WEIGHTS));
    CUDA_CHECK(cudaMalloc(&d_img,      sizeof(__half) * GRID_SIZE * GRID_SIZE * 3));
    CUDA_CHECK(cudaMalloc(&d_task,     sizeof(float) * TASK_EMBED_DIM));
    CUDA_CHECK(cudaMalloc(&d_seed_grad, sizeof(float) * N * BMAP_DIM));

    allocate_checkpoints(&d_ckpt, N);
    allocate_grad_buffers(&d_grads, N);

    BackwardWorkspace ws;
    allocate_backward_workspace(ws, N);

    // Fill image and task on host, then copy to device.
    __half* h_img = (__half*)malloc(sizeof(__half) * GRID_SIZE * GRID_SIZE * 3);
    float* h_task = (float*)malloc(sizeof(float) * TASK_EMBED_DIM);
    for (int i = 0; i < GRID_SIZE * GRID_SIZE * 3; ++i)
        h_img[i] = __float2half(static_cast<float>((i * 37) % 64) / 64.0f);
    for (int i = 0; i < TASK_EMBED_DIM; ++i)
        h_task[i] = 0.1f * static_cast<float>(i + 1);

    CUDA_CHECK(cudaMemcpy(d_img, h_img, sizeof(__half) * GRID_SIZE * GRID_SIZE * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_task, h_task, sizeof(float) * TASK_EMBED_DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyHostToDevice));

    // Set up ForwardInputs on host, copy to device.
    ForwardInputs h_input;
    h_input.role = Role::Classifier;
    h_input.task_embedding = d_task;
    h_input.image_rgb = d_img;
    h_input.target_bmap_32 = nullptr;
    CUDA_CHECK(cudaMemcpy(d_inputs, &h_input, sizeof(ForwardInputs), cudaMemcpyHostToDevice));

    // Run reference forward_kernel.
    launch_forward(d_org_fwd, d_inputs, nullptr,
                   &d_weights[OFF_PERC], &d_weights[OFF_INTER],
                   &d_weights[OFF_FLOW], &d_weights[OFF_BMAP],
                   N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run forward_with_checkpoints.
    launch_forward_with_checkpoints(d_org_ckpt, d_inputs, nullptr,
                                    d_weights, d_ckpt, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare bmap_traj (all 4 samples).
    float h_btraj_ref[BTRAJ_SAMPLES * BMAP_DIM];
    float h_btraj_ckpt[BTRAJ_SAMPLES * BMAP_DIM];
    size_t btraj_offset = offsetof(OrganismState, bmap_traj);
    CUDA_CHECK(cudaMemcpy(h_btraj_ref,
        (char*)d_org_fwd + btraj_offset,
        sizeof(float) * BTRAJ_SAMPLES * BMAP_DIM, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_btraj_ckpt,
        (char*)d_org_ckpt + btraj_offset,
        sizeof(float) * BTRAJ_SAMPLES * BMAP_DIM, cudaMemcpyDeviceToHost));

    int mismatch = 0;
    for (int i = 0; i < BTRAJ_SAMPLES * BMAP_DIM; ++i) {
        float r = h_btraj_ref[i];
        float c = h_btraj_ckpt[i];
        if (r != c) {
            if (mismatch < 5) {
                uint32_t rb, cb;
                std::memcpy(&rb, &r, 4);
                std::memcpy(&cb, &c, 4);
                std::printf("  mismatch at [%d]: ref=%.10e (0x%08x) ckpt=%.10e (0x%08x)\n",
                    i, r, rb, c, cb);
            }
            mismatch++;
        }
    }
    std::printf("  mismatches: %d / %d\n", mismatch, BTRAJ_SAMPLES * BMAP_DIM);
    CHECK(mismatch == 0, "forward_with_checkpoints BTRAJ matches forward_kernel bitwise");
    std::printf("  bmap_64[0..3] = %.4f %.4f %.4f %.4f\n",
        h_btraj_ckpt[3*BMAP_DIM+0], h_btraj_ckpt[3*BMAP_DIM+1],
        h_btraj_ckpt[3*BMAP_DIM+2], h_btraj_ckpt[3*BMAP_DIM+3]);

    // ---- Backward: set seed_grad, run backward, check gradients ----------
    std::printf("--- Test: backward produces finite nonzero gradients ---\n");
    std::fflush(stdout);

    float h_seed[BMAP_DIM];
    for (int d = 0; d < BMAP_DIM; ++d) h_seed[d] = 0.1f * (d + 1);
    CUDA_CHECK(cudaMemcpy(d_seed_grad, h_seed, sizeof(float) * BMAP_DIM, cudaMemcpyHostToDevice));

    launch_backward_all(d_org_ckpt, d_weights, d_seed_grad, d_ckpt, d_grads, ws, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read gradients back.
    GradBuffers h_grads;
    CUDA_CHECK(cudaMemcpy(&h_grads, d_grads, sizeof(GradBuffers), cudaMemcpyDeviceToHost));

    // Check each weight group is finite and nonzero.
    auto check_group = [](const float* dW, int offset, int size, const char* name) {
        bool all_finite = true;
        bool any_nonzero = false;
        float sum_abs = 0.f;
        for (int i = 0; i < size; ++i) {
            float v = dW[offset + i];
            if (!std::isfinite(v)) all_finite = false;
            if (std::fabs(v) > 1e-30f) any_nonzero = true;
            sum_abs += std::fabs(v);
        }
        std::printf("  %s: sum|dW|=%.6e finite=%d nonzero=%d\n",
            name, sum_abs, all_finite, any_nonzero);
        return all_finite && any_nonzero;
    };

    CHECK(check_group(h_grads.dW, OFF_PERC,  W_PERC_SIZE,  "dW_perc"),
          "dW_perc finite and nonzero");
    CHECK(check_group(h_grads.dW, OFF_INTER, W_INTER_SIZE, "dW_inter"),
          "dW_inter finite and nonzero");
    CHECK(check_group(h_grads.dW, OFF_FLOW,  W_FLOW_SIZE,  "dW_flow"),
          "dW_flow finite and nonzero");
    CHECK(check_group(h_grads.dW, OFF_BMAP,  W_BMAP_SIZE,  "dW_bmap"),
          "dW_bmap finite and nonzero");

    // ---- Check stencil adjoint produces nonzero d_state ------------------
    std::printf("--- Test: stencil adjoint produces nonzero d_state ---\n");
    std::fflush(stdout);

    // Read d_state from backward workspace (whichever buffer ended up as "A").
    // After backward, ws.d_state[0] was the initial dA. After 64 swaps it
    // could be in either buffer. Read both and check if either has nonzero.
    float* h_dstate = (float*)malloc(sizeof(float) * GRID_ELEMS);
    bool dstate_nonzero = false;
    for (int buf = 0; buf < 2; ++buf) {
        CUDA_CHECK(cudaMemcpy(h_dstate, ws.d_state[buf],
            sizeof(float) * GRID_ELEMS, cudaMemcpyDeviceToHost));
        float sum = 0.f;
        for (int i = 0; i < GRID_ELEMS; ++i) sum += std::fabs(h_dstate[i]);
        if (sum > 1e-20f) dstate_nonzero = true;
        std::printf("  d_state[%d] sum|v|=%.6e\n", buf, sum);
    }
    CHECK(dstate_nonzero, "stencil adjoint d_state is nonzero");

    free(h_dstate);
    free(h_img);
    free(h_task);
    free(h_weights);
    cudaFree(d_org_fwd);
    cudaFree(d_org_ckpt);
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_img);
    cudaFree(d_task);
    cudaFree(d_seed_grad);
    free_checkpoints(d_ckpt);
    free_grad_buffers(d_grads);
    free_backward_workspace(ws);

    return 0;
}

// ---- Test 3: Loss decreases over 5 iterations ----------------------------
static int test_loss_decreases() {
    std::printf("--- Test: 5 iterations forward+backward+CAME, loss decreases ---\n");
    std::fflush(stdout);

    const int N = 1;
    const int ITERS = 5;
    const int target_class = 2;

    float* h_weights = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS);
    fill_weights(h_weights, TOTAL_WEIGHTS, 42u);

    OrganismState* d_org = nullptr;
    ForwardInputs* d_inputs = nullptr;
    float* d_weights = nullptr;
    __half* d_img = nullptr;
    float* d_task = nullptr;
    float* d_seed_grad = nullptr;
    CheckpointBuffer* d_ckpt = nullptr;
    GradBuffers* d_grads = nullptr;

    CUDA_CHECK(cudaMalloc(&d_org,      sizeof(OrganismState) * N));
    CUDA_CHECK(cudaMalloc(&d_inputs,   sizeof(ForwardInputs) * N));
    CUDA_CHECK(cudaMalloc(&d_weights,  sizeof(float) * TOTAL_WEIGHTS));
    CUDA_CHECK(cudaMalloc(&d_img,      sizeof(__half) * GRID_SIZE * GRID_SIZE * 3));
    CUDA_CHECK(cudaMalloc(&d_task,     sizeof(float) * TASK_EMBED_DIM));
    CUDA_CHECK(cudaMalloc(&d_seed_grad, sizeof(float) * N * BMAP_DIM));

    allocate_checkpoints(&d_ckpt, N);
    allocate_grad_buffers(&d_grads, N);
    BackwardWorkspace ws;
    allocate_backward_workspace(ws, N);

    CameState came;
    allocate_came(came);

    __half* h_img = (__half*)malloc(sizeof(__half) * GRID_SIZE * GRID_SIZE * 3);
    float* h_task = (float*)malloc(sizeof(float) * TASK_EMBED_DIM);
    for (int i = 0; i < GRID_SIZE * GRID_SIZE * 3; ++i)
        h_img[i] = __float2half(static_cast<float>((i * 37) % 64) / 64.0f);
    for (int i = 0; i < TASK_EMBED_DIM; ++i)
        h_task[i] = 0.1f * static_cast<float>(i + 1);

    CUDA_CHECK(cudaMemcpy(d_img, h_img, sizeof(__half) * GRID_SIZE * GRID_SIZE * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_task, h_task, sizeof(float) * TASK_EMBED_DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyHostToDevice));

    ForwardInputs h_input;
    h_input.role = Role::Classifier;
    h_input.task_embedding = d_task;
    h_input.image_rgb = d_img;
    h_input.target_bmap_32 = nullptr;
    CUDA_CHECK(cudaMemcpy(d_inputs, &h_input, sizeof(ForwardInputs), cudaMemcpyHostToDevice));

    float losses[ITERS];
    float* h_desc = (float*)malloc(sizeof(float) * BMAP_DIM);
    float* h_seed = (float*)malloc(sizeof(float) * BMAP_DIM);

    for (int iter = 0; iter < ITERS; ++iter) {
        // Forward.
        launch_forward_with_checkpoints(d_org, d_inputs, nullptr,
                                        d_weights, d_ckpt, N, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Extract descriptor (bmap_64).
        CUDA_CHECK(cudaMemcpy(h_desc,
            (char*)d_org + offsetof(OrganismState, bmap_traj) + (BTRAJ_SAMPLES - 1) * BMAP_DIM * sizeof(float),
            sizeof(float) * BMAP_DIM, cudaMemcpyDeviceToHost));

        // Compute CE loss and seed gradient on host.
        float loss;
        float dlogits[16];
        classifier_loss(h_desc, target_class, 16, dlogits, &loss);
        losses[iter] = loss;

        // Pack seed_grad: first 16 dims = dlogits, rest = 0.
        for (int d = 0; d < BMAP_DIM; ++d) {
            h_seed[d] = (d < 16) ? dlogits[d] : 0.f;
        }
        CUDA_CHECK(cudaMemcpy(d_seed_grad, h_seed, sizeof(float) * BMAP_DIM, cudaMemcpyHostToDevice));

        // Backward.
        launch_backward_all(d_org, d_weights, d_seed_grad, d_ckpt, d_grads, ws, N, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Aggregate + CAME.
        launch_aggregate_gradients(d_grads, came.d_mean_grad, N, 0);
        launch_came_step(d_weights, came, CAME_DEFAULTS, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::printf("  iter %d: loss = %.6f\n", iter, loss);
    }

    bool monotone = true;
    for (int i = 1; i < ITERS; ++i) {
        if (losses[i] >= losses[i-1]) {
            monotone = false;
            std::printf("  loss[%d]=%.6f >= loss[%d]=%.6f\n", i, losses[i], i-1, losses[i-1]);
        }
    }
    CHECK(monotone, "loss decreases monotonically over 5 iterations");

    free(h_desc);
    free(h_seed);
    free(h_img);
    free(h_task);
    free(h_weights);
    cudaFree(d_org);
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_img);
    cudaFree(d_task);
    cudaFree(d_seed_grad);
    free_checkpoints(d_ckpt);
    free_grad_buffers(d_grads);
    free_backward_workspace(ws);
    free_came(came);

    return 0;
}

// ---- Test 4: Gradient aggregation with 4 organisms -----------------------
static int test_gradient_aggregation() {
    std::printf("--- Test: gradient aggregation with 4 organisms ---\n");
    std::fflush(stdout);

    const int N = 4;

    // Allocate and fill GradBuffers on host with known values.
    GradBuffers* h_grads = (GradBuffers*)calloc(N, sizeof(GradBuffers));
    for (int org = 0; org < N; ++org) {
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
            h_grads[org].dW[i] = static_cast<float>(org + 1) * (i + 1) * 0.001f;
        }
    }

    GradBuffers* d_grads = nullptr;
    float* d_mean = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grads, sizeof(GradBuffers) * N));
    CUDA_CHECK(cudaMalloc(&d_mean,  sizeof(float) * TOTAL_WEIGHTS));
    CUDA_CHECK(cudaMemcpy(d_grads, h_grads, sizeof(GradBuffers) * N, cudaMemcpyHostToDevice));

    launch_aggregate_gradients(d_grads, d_mean, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_mean = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS);
    CUDA_CHECK(cudaMemcpy(h_mean, d_mean, sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyDeviceToHost));

    // Expected: mean[i] = (1+2+3+4)/4 * (i+1) * 0.001 = 2.5 * (i+1) * 0.001
    bool correct = true;
    for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
        float expected = 2.5f * (i + 1) * 0.001f;
        if (std::fabs(h_mean[i] - expected) > 1e-4f * std::fabs(expected) + 1e-8f) {
            if (correct) {
                std::printf("  mismatch at [%d]: got=%.6e expected=%.6e\n",
                    i, h_mean[i], expected);
            }
            correct = false;
        }
    }
    CHECK(correct, "gradient aggregation averages correctly");

    free(h_grads);
    free(h_mean);
    cudaFree(d_grads);
    cudaFree(d_mean);

    return 0;
}

int main() {
    std::printf("Wave 1 Acceptance Test\n");
    std::printf("======================\n");
    std::fflush(stdout);

    // Check cooperative launch support.
    int dev = 0;
    int coop = 0;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
    if (!coop) {
        std::printf("FAIL: device does not support cooperative launch\n");
        return 1;
    }
    std::printf("Cooperative launch: supported\n");

    int rc = 0;
    rc |= test_forward_match_and_backward();
    rc |= test_loss_decreases();
    rc |= test_gradient_aggregation();

    std::printf("\n======================\n");
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail > 0 || rc != 0) {
        std::printf("WAVE 1: FAIL\n");
        return 1;
    }
    std::printf("WAVE 1: PASS\n");
    return 0;
}

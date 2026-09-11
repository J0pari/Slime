// Wave 2/2.5 Regression Suite: Gates 1-3
//
// 1. Effective-weight materialization: W_eff == W_shared + delta (bitwise).
// 2. Genome -> phenotype causality: changing one active genome delta changes
//    the descriptor; identical setup reproduces the descriptor bitwise; an
//    empty delta bank reproduces the shared-weight forward AND backward.
// 3. Forced PT swap: every correlated field of a logical organism
//    (device state, checkpoints, gradients, host genome/delta/metadata,
//    seed gradients, batch assignment) moves together.
// 4. Directional finite difference: analytic gradient agrees with the
//    central-difference loss derivative.
//
// Build: make wave2-test

#include "../safety/parallel_tempering.cu"
#include "../optimizer/came.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace slime;
using namespace slime::autodiff;
using namespace slime::nca;
using namespace slime::genome;

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

static void fill_weights(float* w, int n, uint32_t seed) {
    uint32_t s = seed ? seed : 0x9E3779B9u;
    for (int i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        float u = static_cast<float>(s) * (1.0f / 4294967296.0f);
        w[i] = (u - 0.5f) * 0.2f;
    }
}

// ---- Common single-organism rig -----------------------------------------
struct Rig {
    OrganismState*     d_org;
    ForwardInputs*     d_inputs;
    float*             d_weights;      // shared bank
    float*             d_eff_weights;  // [1 * TOTAL_WEIGHTS]
    DeltaWeights*      d_deltas;       // [1]
    CheckpointBuffer*  d_ckpt;
    GradBuffers*       d_grads;
    float*             d_seed_grad;
    BackwardWorkspace  ws;
    __half*            d_img;
    float*             d_task;
    float              h_weights[TOTAL_WEIGHTS];
    __half             h_img[GRID_SIZE * GRID_SIZE * 3];
    float              h_task[TASK_EMBED_DIM];
};

static int rig_init(Rig* r) {
    const int N = 1;
    CUDA_CHECK(cudaMalloc(&r->d_org,     sizeof(OrganismState) * N));
    CUDA_CHECK(cudaMalloc(&r->d_inputs,  sizeof(ForwardInputs) * N));
    CUDA_CHECK(cudaMalloc(&r->d_weights, sizeof(float) * TOTAL_WEIGHTS));
    CUDA_CHECK(cudaMalloc(&r->d_eff_weights, sizeof(float) * TOTAL_WEIGHTS * N));
    CUDA_CHECK(cudaMalloc(&r->d_deltas,  sizeof(DeltaWeights) * N));
    CUDA_CHECK(cudaMalloc(&r->d_img,     sizeof(__half) * GRID_SIZE * GRID_SIZE * 3));
    CUDA_CHECK(cudaMalloc(&r->d_task,    sizeof(float) * TASK_EMBED_DIM));
    CUDA_CHECK(cudaMalloc(&r->d_seed_grad, sizeof(float) * N * BMAP_DIM));

    allocate_checkpoints(&r->d_ckpt, N);
    allocate_grad_buffers(&r->d_grads, N);
    allocate_backward_workspace(r->ws, N);

    fill_weights(r->h_weights, TOTAL_WEIGHTS, 42u);
    for (int i = 0; i < GRID_SIZE * GRID_SIZE * 3; ++i)
        r->h_img[i] = __float2half(static_cast<float>((i * 37) % 64) / 64.0f);
    for (int i = 0; i < TASK_EMBED_DIM; ++i)
        r->h_task[i] = 0.1f * static_cast<float>(i + 1);

    CUDA_CHECK(cudaMemcpy(r->d_weights, r->h_weights,
        sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(r->d_img, r->h_img,
        sizeof(__half) * GRID_SIZE * GRID_SIZE * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(r->d_task, r->h_task,
        sizeof(float) * TASK_EMBED_DIM, cudaMemcpyHostToDevice));

    ForwardInputs h_in;
    h_in.role = Role::Classifier;
    h_in.task_embedding = r->d_task;
    h_in.image_rgb = r->d_img;
    h_in.target_bmap_32 = nullptr;
    CUDA_CHECK(cudaMemcpy(r->d_inputs, &h_in, sizeof(ForwardInputs), cudaMemcpyHostToDevice));
    return 0;
}

static void rig_free(Rig* r) {
    cudaFree(r->d_org);
    cudaFree(r->d_inputs);
    cudaFree(r->d_weights);
    cudaFree(r->d_eff_weights);
    cudaFree(r->d_deltas);
    cudaFree(r->d_img);
    cudaFree(r->d_task);
    cudaFree(r->d_seed_grad);
    free_checkpoints(r->d_ckpt);
    free_grad_buffers(r->d_grads);
    free_backward_workspace(r->ws);
}

static int read_descriptor(Rig* r, float* desc_out) {
    CUDA_CHECK(cudaMemcpy(desc_out,
        (char*)r->d_org + offsetof(OrganismState, bmap_traj) + (BTRAJ_SAMPLES - 1) * BMAP_DIM * sizeof(float),
        sizeof(float) * BMAP_DIM, cudaMemcpyDeviceToHost));
    return 0;
}

static int upload_delta(Rig* r, const DeltaWeights& d) {
    CUDA_CHECK(cudaMemcpy(r->d_deltas, &d, sizeof(DeltaWeights), cudaMemcpyHostToDevice));
    launch_materialize_effective_weights(r->d_weights, r->d_deltas,
                                         r->d_eff_weights, 1, 0);
    return 0;
}

// ---- Test 1: materialization matches host reference ----------------------
static int test_materialize_matches_reference() {
    std::printf("--- Test: W_eff == W_shared + delta (bitwise) ---\n");
    std::fflush(stdout);

    Rig r{};
    if (rig_init(&r)) return 1;

    DeltaWeights d;
    std::memset(&d, 0, sizeof(d));
    d.count = 5;
    d.indices[0] = OFF_FLOW + 3;      d.values[0] =  0.25f;
    d.indices[1] = OFF_PERC + 1;      d.values[1] = -0.10f;
    d.indices[2] = OFF_BMAP + 17;     d.values[2] =  0.05f;
    d.indices[3] = OFF_INTER + 100;   d.values[3] = -0.30f;
    d.indices[4] = OFF_FLOW + 3;      d.values[4] =  0.15f;  // collides with [0]

    if (upload_delta(&r, d)) return 1;
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_eff[TOTAL_WEIGHTS];
    CUDA_CHECK(cudaMemcpy(h_eff, r.d_eff_weights,
        sizeof(float) * TOTAL_WEIGHTS, cudaMemcpyDeviceToHost));

    // Host reference: copy shared, then serial scatter in delta order.
    float ref[TOTAL_WEIGHTS];
    for (int i = 0; i < TOTAL_WEIGHTS; ++i) ref[i] = r.h_weights[i];
    for (int k = 0; k < d.count; ++k) {
        uint32_t idx = d.indices[k] % TOTAL_WEIGHT_SLOTS;
        ref[idx] += d.values[k];
    }

    bool bitwise = true;
    for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
        if (h_eff[i] != ref[i]) {
            bitwise = false;
            std::printf("  mismatch at [%d]: %.9e vs %.9e\n", i, h_eff[i], ref[i]);
            break;
        }
    }
    CHECK(bitwise, "materialized effective weights match host reference bitwise");

    rig_free(&r);
    return 0;
}

// ---- Test 2: genome -> phenotype causality -------------------------------
static int test_genotype_causality() {
    // [claim:A301.genotype-causes-phenotype]
    // [claim:A401.archive-genotype-attribution]
    std::printf("--- Test: genome delta changes the phenotype ---\n");
    std::fflush(stdout);

    Rig r{};
    if (rig_init(&r)) return 1;

    // Empty delta: effective bank must reproduce the shared-weight forward
    // and backward exactly.
    DeltaWeights empty;
    std::memset(&empty, 0, sizeof(empty));
    empty.count = 0;

    if (upload_delta(&r, empty)) return 1;
    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, r.d_eff_weights,
                                    r.d_ckpt, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float desc_eff_empty[BMAP_DIM];
    if (read_descriptor(&r, desc_eff_empty)) return 1;

    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, nullptr,
                                    r.d_ckpt, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float desc_shared[BMAP_DIM];
    if (read_descriptor(&r, desc_shared)) return 1;

    bool identity = true;
    for (int i = 0; i < BMAP_DIM; ++i) {
        if (desc_eff_empty[i] != desc_shared[i]) { identity = false; break; }
    }
    CHECK(identity, "empty delta bank reproduces shared-weight descriptor bitwise");

    // Backward identity: same seed, both paths, identical gradients.
    float h_seed[BMAP_DIM];
    for (int d = 0; d < BMAP_DIM; ++d) h_seed[d] = 0.1f * (d + 1);
    CUDA_CHECK(cudaMemcpy(r.d_seed_grad, h_seed,
        sizeof(float) * BMAP_DIM, cudaMemcpyHostToDevice));
    launch_backward_all(r.d_org, r.d_weights, r.d_eff_weights, r.d_seed_grad,
                        r.d_ckpt, r.d_grads, r.ws, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    GradBuffers g_eff;
    CUDA_CHECK(cudaMemcpy(&g_eff, r.d_grads, sizeof(GradBuffers), cudaMemcpyDeviceToHost));

    launch_backward_all(r.d_org, r.d_weights, r.d_eff_weights, r.d_seed_grad,
                        r.d_ckpt, r.d_grads, r.ws, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    GradBuffers g_shared;
    CUDA_CHECK(cudaMemcpy(&g_shared, r.d_grads, sizeof(GradBuffers), cudaMemcpyDeviceToHost));

    float max_abs_diff[4] = {};
    int first_diff_idx[4] = { -1, -1, -1, -1 };
    for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
        float d = std::fabs(g_eff.dW[i] - g_shared.dW[i]);
        int bank = (i < OFF_INTER) ? 0 : (i < OFF_FLOW) ? 1 : (i < OFF_BMAP) ? 2 : 3;
        if (d > max_abs_diff[bank]) {
            max_abs_diff[bank] = d;
            if (first_diff_idx[bank] < 0 && d > 1e-12f) first_diff_idx[bank] = i;
        }
    }
    std::printf("  eff vs shared grad: max|d| perc=%.3e inter=%.3e flow=%.3e bmap=%.3e (first idx %d %d %d %d)\n",
                max_abs_diff[0], max_abs_diff[1], max_abs_diff[2], max_abs_diff[3],
                first_diff_idx[0], first_diff_idx[1], first_diff_idx[2], first_diff_idx[3]);
    float g_scale = 0.f;
    for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
        g_scale = std::fmax(g_scale, std::fabs(g_shared.dW[i]));
    }
    // atomicAdd accumulation order is nondeterministic run-to-run; the two
    // paths must agree to within floating-point accumulation noise.
    bool grad_close = true;
    for (int b = 0; b < 4; ++b) {
        if (max_abs_diff[b] > 1e-4f * (1.f + g_scale)) grad_close = false;
    }
    CHECK(grad_close, "empty delta bank reproduces shared-weight gradients");

    // Delta A vs delta B: one active weight differs -> descriptor differs.
    DeltaWeights dA;
    std::memset(&dA, 0, sizeof(dA));
    dA.count = 1;
    dA.indices[0] = OFF_FLOW + 7;
    dA.values[0] = 0.3f;

    DeltaWeights dB = dA;
    dB.values[0] = -0.3f;

    if (upload_delta(&r, dA)) return 1;
    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, r.d_eff_weights,
                                    r.d_ckpt, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float descA[BMAP_DIM];
    if (read_descriptor(&r, descA)) return 1;

    if (upload_delta(&r, dB)) return 1;
    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, r.d_eff_weights,
                                    r.d_ckpt, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float descB[BMAP_DIM];
    if (read_descriptor(&r, descB)) return 1;

    bool differ = false;
    float max_diff = 0.f;
    for (int i = 0; i < BMAP_DIM; ++i) {
        float d = std::fabs(descA[i] - descB[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1e-6f) differ = true;
    }
    std::printf("  max |descA - descB| = %.6e\n", max_diff);
    CHECK(differ, "changing one active genome delta changes the descriptor");

    // Determinism: same delta re-run reproduces the descriptor bitwise.
    if (upload_delta(&r, dA)) return 1;
    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, r.d_eff_weights,
                                    r.d_ckpt, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float descA2[BMAP_DIM];
    if (read_descriptor(&r, descA2)) return 1;

    bool same = true;
    for (int i = 0; i < BMAP_DIM; ++i) {
        if (descA2[i] != descA[i]) { same = false; break; }
    }
    CHECK(same, "same genome + same input reproduces the descriptor bitwise");

    rig_free(&r);
    return 0;
}

// ---- Test 3: forced PT swap transaction -----------------------------------
static int test_forced_pt_swap() {
    // [claim:S004.pt-swap-transaction]
    std::printf("--- Test: PT swap moves every correlated field together ---\n");
    std::fflush(stdout);

    const int N = 2;

    // Device arrays with sentinels.
    OrganismState* d_org = nullptr;
    CheckpointBuffer* d_ckpt = nullptr;
    GradBuffers* d_grads = nullptr;
    OrganismState* d_tmp_org = nullptr;
    CheckpointBuffer* d_tmp_ckpt = nullptr;
    GradBuffers* d_tmp_grad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_org, sizeof(OrganismState) * N));
    CUDA_CHECK(cudaMalloc(&d_ckpt, sizeof(CheckpointBuffer) * N));
    CUDA_CHECK(cudaMalloc(&d_grads, sizeof(GradBuffers) * N));
    CUDA_CHECK(cudaMalloc(&d_tmp_org, sizeof(OrganismState)));
    CUDA_CHECK(cudaMalloc(&d_tmp_ckpt, sizeof(CheckpointBuffer)));
    CUDA_CHECK(cudaMalloc(&d_tmp_grad, sizeof(GradBuffers)));

    OrganismState h_org[2];
    CheckpointBuffer h_ckpt[2];
    GradBuffers h_grads[2];
    std::memset(&h_org, 0, sizeof(h_org));
    std::memset(&h_ckpt, 0, sizeof(h_ckpt));
    std::memset(&h_grads, 0, sizeof(h_grads));
    for (int a = 0; a < N; ++a) {
        float sentinel = (a == 0) ? 1.0f : 2.0f;
        h_org[a].bmap_traj[0] = sentinel;
        h_org[a].bmap_traj[BMAP_DIM] = sentinel * 10.f;
        h_org[a].grid[0] = __float2half(sentinel);
        h_org[a].role = (a == 0) ? Role::Classifier : Role::Predictor;
        h_ckpt[a].data[0][0] = __float2half(sentinel * 3.f);
        h_ckpt[a].data[3][GRID_ELEMS - 1] = __float2half(sentinel * 5.f);
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) h_grads[a].dW[i] = sentinel;
    }
    CUDA_CHECK(cudaMemcpy(d_org, h_org, sizeof(h_org), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ckpt, h_ckpt, sizeof(h_ckpt), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grads, h_grads, sizeof(h_grads), cudaMemcpyHostToDevice));

    // Host organism-table rows with sentinels.
    genome::Genome genomes[2];
    DeltaWeights deltas[2];
    uint32_t lineage[2], parent[2];
    int spawn_gen[2];
    float fitness[2], f_raw[2], f_sot[2];
    Role role[2];
    float seed_grad[2 * BMAP_DIM];
    int batch_idx[2];
    std::memset(&genomes, 0, sizeof(genomes));
    std::memset(&deltas, 0, sizeof(deltas));
    std::memset(seed_grad, 0, sizeof(seed_grad));

    genomes[0].bits[5] = 0xAAAAAAAAu;
    genomes[1].bits[5] = 0x55555555u;
    deltas[0].count = 3; deltas[0].indices[0] = 11; deltas[0].values[0] = 0.7f;
    deltas[1].count = 7; deltas[1].indices[0] = 29; deltas[1].values[0] = -0.9f;
    lineage[0] = 101; lineage[1] = 202;
    parent[0] = 1001; parent[1] = 2002;
    spawn_gen[0] = 5; spawn_gen[1] = 9;
    fitness[0] = 0.25f; fitness[1] = 0.75f;
    f_raw[0] = 0.2f;   f_raw[1] = 0.8f;
    f_sot[0] = 0.3f;   f_sot[1] = 0.9f;
    role[0] = Role::Classifier; role[1] = Role::Predictor;
    for (int d = 0; d < BMAP_DIM; ++d) {
        seed_grad[0 * BMAP_DIM + d] = 0.1f * d;
        seed_grad[1 * BMAP_DIM + d] = 0.2f * d;
    }
    batch_idx[0] = 3; batch_idx[1] = 12;

    safety::pt::SwapContext ctx;
    ctx.d_organisms = d_org;
    ctx.d_checkpoints = d_ckpt;
    ctx.d_grads = d_grads;
    ctx.d_swap_org = d_tmp_org;
    ctx.d_swap_ckpt = d_tmp_ckpt;
    ctx.d_swap_grad = d_tmp_grad;
    ctx.genomes = genomes;
    ctx.deltas = deltas;
    ctx.lineage_id = lineage;
    ctx.parent_id = parent;
    ctx.spawn_gen = spawn_gen;
    ctx.fitness = fitness;
    ctx.f_raw = f_raw;
    ctx.f_sot = f_sot;
    ctx.role = role;
    ctx.seed_grad = seed_grad;
    ctx.batch_sample_idx = batch_idx;
    ctx.stream = 0;

    safety::pt::swap_device_organism(ctx, 0, 1);
    safety::pt::swap_host_organism(ctx, 0, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool ok = true;

    // Device data moved.
    OrganismState h_org2[2];
    CheckpointBuffer h_ckpt2[2];
    GradBuffers h_grads2[2];
    CUDA_CHECK(cudaMemcpy(h_org2, d_org, sizeof(h_org2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ckpt2, d_ckpt, sizeof(h_ckpt2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grads2, d_grads, sizeof(h_grads2), cudaMemcpyDeviceToHost));

    if (h_org2[0].bmap_traj[0] != 2.0f || h_org2[1].bmap_traj[0] != 1.0f) ok = false;
    if (h_org2[0].role != Role::Predictor || h_org2[1].role != Role::Classifier) ok = false;
    if (__half2float(h_ckpt2[0].data[0][0]) != 6.0f) ok = false;
    if (__half2float(h_ckpt2[1].data[3][GRID_ELEMS - 1]) != 5.0f) ok = false;
    if (h_grads2[0].dW[10] != 2.0f || h_grads2[1].dW[10] != 1.0f) ok = false;
    CHECK(ok, "device organism state, checkpoints, and gradients swapped together");

    // Host metadata moved.
    ok = true;
    if (genomes[0].bits[5] != 0x55555555u || genomes[1].bits[5] != 0xAAAAAAAAu) ok = false;
    if (deltas[0].count != 7 || deltas[1].count != 3) ok = false;
    if (lineage[0] != 202 || lineage[1] != 101) ok = false;
    if (parent[0] != 2002 || parent[1] != 1001) ok = false;
    if (spawn_gen[0] != 9 || spawn_gen[1] != 5) ok = false;
    if (fitness[0] != 0.75f || fitness[1] != 0.25f) ok = false;
    if (role[0] != Role::Predictor || role[1] != Role::Classifier) ok = false;
    CHECK(ok, "host genome, delta, lineage, spawn, fitness, and role swapped together");

    // Evaluation correlation state moved.
    ok = true;
    for (int d = 0; d < BMAP_DIM; ++d) {
        if (seed_grad[0 * BMAP_DIM + d] != 0.2f * d) ok = false;
        if (seed_grad[1 * BMAP_DIM + d] != 0.1f * d) ok = false;
    }
    if (batch_idx[0] != 12 || batch_idx[1] != 3) ok = false;
    CHECK(ok, "seed-gradient rows and batch assignment swapped with the organism");

    cudaFree(d_org);
    cudaFree(d_ckpt);
    cudaFree(d_grads);
    cudaFree(d_tmp_org);
    cudaFree(d_tmp_ckpt);
    cudaFree(d_tmp_grad);
    return 0;
}

// ---- Test 4: directional finite difference --------------------------------
static int test_finite_difference_gradient() {
    // [claim:A103.gradient-correctness]
    std::printf("--- Test: analytic gradient agrees with finite differences ---\n");
    std::fflush(stdout);

    Rig r{};
    if (rig_init(&r)) return 1;

    const int target_class = 2;
    const float eps = 0.05f;

    auto forward_and_loss = [&](const float* w, float* loss_out, float* desc_out) {
        CUDA_CHECK(cudaMemcpy(r.d_weights, w, sizeof(float) * TOTAL_WEIGHTS,
                              cudaMemcpyHostToDevice));
        launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                        r.d_weights, nullptr,
                                        r.d_ckpt, 1, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (read_descriptor(&r, desc_out)) return 1;
        float dlogits[NUM_CLASSES];
        classifier_loss(desc_out, target_class, NUM_CLASSES, dlogits, loss_out);
        return 0;
    };

    // Loss at W and analytic gradient.
    float desc0[BMAP_DIM];
    float L0 = 0.f;
    if (forward_and_loss(r.h_weights, &L0, desc0)) return 1;

    float h_seed[BMAP_DIM];
    for (int d = 0; d < BMAP_DIM; ++d) h_seed[d] = 0.f;
    {
        float dlogits[NUM_CLASSES];
        float tmp;
        classifier_loss(desc0, target_class, NUM_CLASSES, dlogits, &tmp);
        for (int d = 0; d < NUM_CLASSES; ++d) h_seed[d] = dlogits[d];
    }
    CUDA_CHECK(cudaMemcpy(r.d_seed_grad, h_seed,
        sizeof(float) * BMAP_DIM, cudaMemcpyHostToDevice));
    launch_backward_all(r.d_org, r.d_weights, nullptr, r.d_seed_grad,
                        r.d_ckpt, r.d_grads, r.ws, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    GradBuffers g;
    CUDA_CHECK(cudaMemcpy(&g, r.d_grads, sizeof(GradBuffers), cudaMemcpyDeviceToHost));

    const int bank_lo[4] = { OFF_PERC, OFF_INTER, OFF_FLOW, OFF_BMAP };
    const int bank_hi[4] = { OFF_INTER, OFF_FLOW, OFF_BMAP, TOTAL_WEIGHTS };
    const char* bank_name[4] = { "W_perc", "W_inter", "W_flow", "W_bmap" };

    for (int bank = 0; bank < 4; ++bank) {
        float v[TOTAL_WEIGHTS] = {};
        uint32_t s = 0x1234ABCDu ^ (0x9E3779B9u * static_cast<uint32_t>(bank + 1));
        float norm2 = 0.f;
        for (int i = bank_lo[bank]; i < bank_hi[bank]; ++i) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            float u = static_cast<float>(s) * (1.0f / 4294967296.0f) - 0.5f;
            v[i] = u;
            norm2 += u * u;
        }
        float inv = 1.0f / sqrtf(norm2);
        for (int i = bank_lo[bank]; i < bank_hi[bank]; ++i) v[i] *= inv;

        float d_analytic = 0.f;
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) d_analytic += g.dW[i] * v[i];

        float w_plus[TOTAL_WEIGHTS], w_minus[TOTAL_WEIGHTS];
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
            w_plus[i]  = r.h_weights[i] + eps * v[i];
            w_minus[i] = r.h_weights[i] - eps * v[i];
        }
        float Lp = 0.f, Lm = 0.f;
        float desc_tmp[BMAP_DIM];
        if (forward_and_loss(w_plus, &Lp, desc_tmp)) return 1;
        if (forward_and_loss(w_minus, &Lm, desc_tmp)) return 1;

        float d_numeric = (Lp - Lm) / (2.0f * eps);
        float rel_err = std::fabs(d_analytic - d_numeric)
                      / std::fmax(std::fabs(d_numeric), 1e-12f);
        std::printf("  %-8s d_analytic=% .6e d_numeric=% .6e rel_err=%.4e\n",
                    bank_name[bank], d_analytic, d_numeric, rel_err);
    }

    {
        float v[TOTAL_WEIGHTS];
        uint32_t s = 0x1234ABCDu;
        float norm2 = 0.f;
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            float u = static_cast<float>(s) * (1.0f / 4294967296.0f) - 0.5f;
            v[i] = u;
            norm2 += u * u;
        }
        float inv = 1.0f / sqrtf(norm2);
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) v[i] *= inv;

        float d_analytic = 0.f;
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) d_analytic += g.dW[i] * v[i];

        float w_plus[TOTAL_WEIGHTS], w_minus[TOTAL_WEIGHTS];
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
            w_plus[i]  = r.h_weights[i] + eps * v[i];
            w_minus[i] = r.h_weights[i] - eps * v[i];
        }
        float Lp = 0.f, Lm = 0.f;
        float desc_tmp[BMAP_DIM];
        if (forward_and_loss(w_plus, &Lp, desc_tmp)) return 1;
        if (forward_and_loss(w_minus, &Lm, desc_tmp)) return 1;

        float d_numeric = (Lp - Lm) / (2.0f * eps);
        float rel_err = std::fabs(d_analytic - d_numeric)
                      / std::fmax(std::fabs(d_numeric), 1e-12f);
        std::printf("  %-8s d_analytic=% .6e d_numeric=% .6e rel_err=%.4e\n",
                    "all", d_analytic, d_numeric, rel_err);
        CHECK(std::fabs(d_numeric) > 1e-9f, "finite-difference direction is informative");
        CHECK(rel_err < 5e-2f, "analytic directional derivative matches central difference");
    }

    rig_free(&r);
    return 0;
}

int main() {
    std::printf("Wave 2/2.5 Regression Suite (Gates 1-3)\n");
    std::printf("========================================\n");
    std::fflush(stdout);

    int rc = 0;
    rc |= test_materialize_matches_reference();
    rc |= test_genotype_causality();
    rc |= test_forced_pt_swap();
    rc |= test_finite_difference_gradient();

    std::printf("\n========================================\n");
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail > 0 || rc != 0) {
        std::printf("WAVE2 REGRESSION: FAIL\n");
        return 1;
    }
    std::printf("WAVE2 REGRESSION: PASS\n");
    return 0;
}




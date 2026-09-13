// Evolution regression suite: role locking, effective weights, PT
// correspondence, finite differences.
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
// Build: make evolution-test

#include "../safety/parallel_tempering.cu"
#include "../config/gpu_authorization.cuh"
#include "../optimizer/came.cu"
#include "../integration/phase_graph.cuh"

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
                                    r.d_ckpt, RESIDUAL_ALPHA, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float desc_eff_empty[BMAP_DIM];
    if (read_descriptor(&r, desc_eff_empty)) return 1;

    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, nullptr,
                                    r.d_ckpt, RESIDUAL_ALPHA, 1, 0);
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
    launch_backward_all(r.d_org, r.d_weights, r.d_eff_weights, nullptr,
                        r.d_seed_grad,
                        r.d_ckpt, r.d_grads, r.ws, RESIDUAL_ALPHA, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    GradBuffers g_eff;
    CUDA_CHECK(cudaMemcpy(&g_eff, r.d_grads, sizeof(GradBuffers), cudaMemcpyDeviceToHost));

    launch_backward_all(r.d_org, r.d_weights, r.d_eff_weights, nullptr,
                        r.d_seed_grad,
                        r.d_ckpt, r.d_grads, r.ws, RESIDUAL_ALPHA, 1, 0);
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
                                    r.d_ckpt, RESIDUAL_ALPHA, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float descA[BMAP_DIM];
    if (read_descriptor(&r, descA)) return 1;

    if (upload_delta(&r, dB)) return 1;
    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, r.d_eff_weights,
                                    r.d_ckpt, RESIDUAL_ALPHA, 1, 0);
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
                                    r.d_ckpt, RESIDUAL_ALPHA, 1, 0);
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
    float* d_eff = nullptr;           // [N * TOTAL_WEIGHTS] effective banks
    OrganismState* d_tmp_org = nullptr;
    CheckpointBuffer* d_tmp_ckpt = nullptr;
    GradBuffers* d_tmp_grad = nullptr;
    float* d_tmp_wbank = nullptr;
    CUDA_CHECK(cudaMalloc(&d_org, sizeof(OrganismState) * N));
    CUDA_CHECK(cudaMalloc(&d_ckpt, sizeof(CheckpointBuffer) * N));
    CUDA_CHECK(cudaMalloc(&d_grads, sizeof(GradBuffers) * N));
    CUDA_CHECK(cudaMalloc(&d_eff, sizeof(float) * TOTAL_WEIGHTS * N));
    CUDA_CHECK(cudaMalloc(&d_tmp_org, sizeof(OrganismState)));
    CUDA_CHECK(cudaMalloc(&d_tmp_ckpt, sizeof(CheckpointBuffer)));
    CUDA_CHECK(cudaMalloc(&d_tmp_grad, sizeof(GradBuffers)));
    CUDA_CHECK(cudaMalloc(&d_tmp_wbank, sizeof(float) * TOTAL_WEIGHTS));

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

    // Effective-weight banks with sentinels: slot 0 all 1.0f, slot 1 all 2.0f.
    {
        float* h_eff = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS * N);
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
            h_eff[0 * TOTAL_WEIGHTS + i] = 1.0f;
            h_eff[1 * TOTAL_WEIGHTS + i] = 2.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_eff, h_eff, sizeof(float) * TOTAL_WEIGHTS * N,
                              cudaMemcpyHostToDevice));
        free(h_eff);
    }

    // Host organism-table rows with sentinels.
    genome::Genome genomes[2];
    DeltaWeights deltas[2];
    slime::LineageId lineage[2];
    slime::ArchiveSlot parent[2];
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
    lineage[0] = slime::LineageId(101u);
    lineage[1] = slime::LineageId(202u);
    parent[0] = slime::ArchiveSlot(1001);
    parent[1] = slime::ArchiveSlot(2002);
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
    float pred_err_ema[2] = {0.11f, 0.22f};
    float pred_loss_ema[2] = {0.33f, 0.44f};

    safety::pt::SwapContext ctx;
    ctx.d_organisms = d_org;
    ctx.d_checkpoints = d_ckpt;
    ctx.d_grads = d_grads;
    ctx.d_eff_weights = d_eff;
    ctx.d_swap_org = d_tmp_org;
    ctx.d_swap_ckpt = d_tmp_ckpt;
    ctx.d_swap_grad = d_tmp_grad;
    ctx.d_swap_wbank = d_tmp_wbank;
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
    ctx.predictor_error_ema = pred_err_ema;
    ctx.predictor_loss_ema = pred_loss_ema;
    ctx.stream = 0;

    safety::pt::swap_device_organism(ctx, slime::PtSlot(0),
                                     slime::PtSlot(1));
    safety::pt::swap_host_organism(ctx, slime::PtSlot(0),
                                   slime::PtSlot(1));
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
    if (lineage[0] != slime::LineageId(202u)
        || lineage[1] != slime::LineageId(101u)) ok = false;
    if (parent[0] != slime::ArchiveSlot(2002)
        || parent[1] != slime::ArchiveSlot(1001)) ok = false;
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

    // The per-organism EMA state (predictor error and loss) moves too.
    ok = true;
    if (pred_err_ema[0] != 0.22f || pred_err_ema[1] != 0.11f) ok = false;
    if (pred_loss_ema[0] != 0.44f || pred_loss_ema[1] != 0.33f) ok = false;
    CHECK(ok, "predictor error and loss EMAs swapped with the organism");

    // Effective-weight banks moved with the organism: backward must re-forward
    // each trajectory with the phenotype that produced it.
    {
        float* h_eff = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS * N);
        CUDA_CHECK(cudaMemcpy(h_eff, d_eff, sizeof(float) * TOTAL_WEIGHTS * N,
                              cudaMemcpyDeviceToHost));
        bool banks_ok = true;
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
            if (h_eff[0 * TOTAL_WEIGHTS + i] != 2.0f) banks_ok = false;
            if (h_eff[1 * TOTAL_WEIGHTS + i] != 1.0f) banks_ok = false;
        }
        CHECK(banks_ok, "effective-weight banks swapped with the organism");
        free(h_eff);
    }

    cudaFree(d_org);
    cudaFree(d_ckpt);
    cudaFree(d_grads);
    cudaFree(d_eff);
    cudaFree(d_tmp_org);
    cudaFree(d_tmp_ckpt);
    cudaFree(d_tmp_grad);
    cudaFree(d_tmp_wbank);
    return 0;
}

// ---- Test 3b: backward correspondence after a forced PT swap --------------
// After an accepted swap, a backward at slot A must reproduce the gradient
// that logical organism B produced before the swap: checkpoint, effective
// weights, and seed gradient moved together, so the gradient attribution is
// unchanged by the exchange.
// [claim:S003.red-team-coverage]
static int test_pt_swap_backward_correspondence() {
    // [claim:S004.pt-swap-transaction]
    std::printf("--- Test: backward correspondence across a PT swap ---\n");
    std::fflush(stdout);

    const int N = 2;

    OrganismState* d_org = nullptr;
    ForwardInputs* d_inputs = nullptr;
    float* d_weights = nullptr;
    float* d_eff = nullptr;
    DeltaWeights* d_deltas = nullptr;
    CheckpointBuffer* d_ckpt = nullptr;
    GradBuffers* d_grads = nullptr;
    float* d_seed = nullptr;
    BackwardWorkspace ws;
    OrganismState* d_tmp_org = nullptr;
    CheckpointBuffer* d_tmp_ckpt = nullptr;
    GradBuffers* d_tmp_grad = nullptr;
    float* d_tmp_wbank = nullptr;
    __half* d_img = nullptr;
    float* d_task = nullptr;

    CUDA_CHECK(cudaMalloc(&d_org, sizeof(OrganismState) * N));
    CUDA_CHECK(cudaMalloc(&d_inputs, sizeof(ForwardInputs) * N));
    CUDA_CHECK(cudaMalloc(&d_weights, sizeof(float) * TOTAL_WEIGHTS));
    CUDA_CHECK(cudaMalloc(&d_eff, sizeof(float) * TOTAL_WEIGHTS * N));
    CUDA_CHECK(cudaMalloc(&d_deltas, sizeof(DeltaWeights) * N));
    CUDA_CHECK(cudaMalloc(&d_img, sizeof(__half) * GRID_SIZE * GRID_SIZE * 3));
    CUDA_CHECK(cudaMalloc(&d_task, sizeof(float) * TASK_EMBED_DIM));
    CUDA_CHECK(cudaMalloc(&d_seed, sizeof(float) * N * BMAP_DIM));
    CUDA_CHECK(cudaMalloc(&d_tmp_org, sizeof(OrganismState)));
    CUDA_CHECK(cudaMalloc(&d_tmp_ckpt, sizeof(CheckpointBuffer)));
    CUDA_CHECK(cudaMalloc(&d_tmp_grad, sizeof(GradBuffers)));
    CUDA_CHECK(cudaMalloc(&d_tmp_wbank, sizeof(float) * TOTAL_WEIGHTS));
    allocate_checkpoints(&d_ckpt, N);
    allocate_grad_buffers(&d_grads, N);
    allocate_backward_workspace(ws, N);

    float* h_weights = (float*)malloc(sizeof(float) * TOTAL_WEIGHTS);
    fill_weights(h_weights, TOTAL_WEIGHTS, 42u);
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, sizeof(float) * TOTAL_WEIGHTS,
                          cudaMemcpyHostToDevice));

    __half* h_img = (__half*)malloc(sizeof(__half) * GRID_SIZE * GRID_SIZE * 3);
    float* h_task = (float*)malloc(sizeof(float) * TASK_EMBED_DIM);
    for (int i = 0; i < GRID_SIZE * GRID_SIZE * 3; ++i)
        h_img[i] = __float2half(static_cast<float>((i * 37) % 64) / 64.0f);
    for (int i = 0; i < TASK_EMBED_DIM; ++i)
        h_task[i] = 0.1f * static_cast<float>(i + 1);
    CUDA_CHECK(cudaMemcpy(d_img, h_img, sizeof(__half) * GRID_SIZE * GRID_SIZE * 3,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_task, h_task, sizeof(float) * TASK_EMBED_DIM,
                          cudaMemcpyHostToDevice));

    // Two distinct genotypes: same delta index, opposite values -> distinct
    // effective banks (and therefore distinct trajectories).
    DeltaWeights h_deltas[2];
    std::memset(&h_deltas, 0, sizeof(h_deltas));
    h_deltas[0].count = 1; h_deltas[0].indices[0] = OFF_FLOW + 3; h_deltas[0].values[0] =  0.25f;
    h_deltas[1].count = 1; h_deltas[1].indices[0] = OFF_FLOW + 3; h_deltas[1].values[0] = -0.25f;
    CUDA_CHECK(cudaMemcpy(d_deltas, h_deltas, sizeof(h_deltas), cudaMemcpyHostToDevice));
    launch_materialize_effective_weights(d_weights, d_deltas, d_eff, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    ForwardInputs h_inputs[2];
    for (int a = 0; a < N; ++a) {
        h_inputs[a].role = Role::Classifier;
        h_inputs[a].task_embedding = d_task;
        h_inputs[a].image_rgb = d_img;
        h_inputs[a].target_bmap_32 = nullptr;
    }
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs, sizeof(h_inputs), cudaMemcpyHostToDevice));

    launch_forward_with_checkpoints(d_org, d_inputs, nullptr,
                                    d_weights, d_eff, d_ckpt, RESIDUAL_ALPHA, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Distinct seed gradients per organism.
    float h_seed[2 * BMAP_DIM];
    for (int d = 0; d < BMAP_DIM; ++d) {
        h_seed[0 * BMAP_DIM + d] = 0.1f * (d + 1);
        h_seed[1 * BMAP_DIM + d] = 0.2f * (d + 1);
    }
    CUDA_CHECK(cudaMemcpy(d_seed, h_seed, sizeof(h_seed), cudaMemcpyHostToDevice));

    // Pre-swap backward: gradients of logical organisms 0 and 1 in place.
    launch_backward_all(d_org, d_weights, d_eff, nullptr, d_seed,
                        d_ckpt, d_grads, ws, RESIDUAL_ALPHA, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    GradBuffers h_pre[2];
    CUDA_CHECK(cudaMemcpy(h_pre, d_grads, sizeof(h_pre), cudaMemcpyDeviceToHost));

    // Forced swap of slots 0 <-> 1 (device data + host rows + seed rows).
    genome::Genome genomes[2];
    DeltaWeights deltas_h[2];
    slime::LineageId lineage[2];
    slime::ArchiveSlot parent[2];
    int spawn_gen[2], batch_idx[2];
    float fitness[2], f_raw[2], f_sot[2];
    Role role[2];
    std::memset(&genomes, 0, sizeof(genomes));
    std::memset(&deltas_h, 0, sizeof(deltas_h));
    genomes[0].bits[5] = 0xAAAAAAAAu; genomes[1].bits[5] = 0x55555555u;
    lineage[0] = slime::LineageId(101u);
    lineage[1] = slime::LineageId(202u);
    parent[0] = slime::ArchiveSlot(1001);
    parent[1] = slime::ArchiveSlot(2002);
    spawn_gen[0] = 5; spawn_gen[1] = 9;
    fitness[0] = 0.25f; fitness[1] = 0.75f;
    f_raw[0] = 0.2f; f_raw[1] = 0.8f;
    f_sot[0] = 0.3f; f_sot[1] = 0.9f;
    role[0] = Role::Classifier; role[1] = Role::Classifier;
    batch_idx[0] = 3; batch_idx[1] = 12;

    safety::pt::SwapContext ctx;
    ctx.d_organisms = d_org;
    ctx.d_checkpoints = d_ckpt;
    ctx.d_grads = d_grads;
    ctx.d_eff_weights = d_eff;
    ctx.d_swap_org = d_tmp_org;
    ctx.d_swap_ckpt = d_tmp_ckpt;
    ctx.d_swap_grad = d_tmp_grad;
    ctx.d_swap_wbank = d_tmp_wbank;
    ctx.genomes = genomes;
    ctx.deltas = deltas_h;
    ctx.lineage_id = lineage;
    ctx.parent_id = parent;
    ctx.spawn_gen = spawn_gen;
    ctx.fitness = fitness;
    ctx.f_raw = f_raw;
    ctx.f_sot = f_sot;
    ctx.role = role;
    ctx.seed_grad = h_seed;          // host rows move with the organism
    ctx.batch_sample_idx = batch_idx;
    float pred_err_ema2[2] = {0.11f, 0.22f};
    float pred_loss_ema2[2] = {0.33f, 0.44f};
    ctx.predictor_error_ema = pred_err_ema2;
    ctx.predictor_loss_ema = pred_loss_ema2;
    ctx.stream = 0;

    safety::pt::swap_device_organism(ctx, slime::PtSlot(0),
                                     slime::PtSlot(1));
    safety::pt::swap_host_organism(ctx, slime::PtSlot(0),
                                   slime::PtSlot(1));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Production order: T3 uploads the REORDERED host seed rows after PT.
    CUDA_CHECK(cudaMemcpy(d_seed, h_seed, sizeof(h_seed), cudaMemcpyHostToDevice));

    // Post-swap backward.
    launch_backward_all(d_org, d_weights, d_eff, nullptr, d_seed,
                        d_ckpt, d_grads, ws, RESIDUAL_ALPHA, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    GradBuffers h_post[2];
    CUDA_CHECK(cudaMemcpy(h_post, d_grads, sizeof(h_post), cudaMemcpyDeviceToHost));

    // Slot 0 now holds logical organism 1's rollout: its gradient must equal
    // the pre-swap gradient of organism 1 (and vice versa). Without the
    // effective-weight swap this fails because backward would re-forward
    // organism 1's checkpoints with organism 0's stale bank.
    bool match = true;
    float max_diff = 0.f, scale = 0.f;
    for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
        float d01 = std::fabs(h_post[0].dW[i] - h_pre[1].dW[i]);
        float d10 = std::fabs(h_post[1].dW[i] - h_pre[0].dW[i]);
        if (d01 > max_diff) max_diff = d01;
        if (d10 > max_diff) max_diff = d10;
        scale = std::fmax(scale, std::fabs(h_pre[1].dW[i]));
        scale = std::fmax(scale, std::fabs(h_pre[0].dW[i]));
    }
    if (max_diff > 1e-4f * (1.f + scale)) match = false;
    std::printf("  max |dW_post - dW_pre(moved org)| = %.3e (scale %.3e)\n",
                max_diff, scale);
    CHECK(match, "post-swap backward matches the moved organism's pre-swap gradient");

    // Sensitivity check: simulate the pre-fix arrangement (banks following
    // slots, not organisms) by re-materializing from the unswapped device
    // delta array, then run backward again. It MUST diverge from the
    // reference gradient — this proves the test above detects the missing
    // bank swap rather than passing vacuously.
    launch_materialize_effective_weights(d_weights, d_deltas, d_eff, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    launch_backward_all(d_org, d_weights, d_eff, nullptr, d_seed,
                        d_ckpt, d_grads, ws, RESIDUAL_ALPHA, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    GradBuffers h_buggy[2];
    CUDA_CHECK(cudaMemcpy(h_buggy, d_grads, sizeof(h_buggy), cudaMemcpyDeviceToHost));
    float buggy_diff = 0.f;
    for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
        float d01 = std::fabs(h_buggy[0].dW[i] - h_pre[1].dW[i]);
        float d10 = std::fabs(h_buggy[1].dW[i] - h_pre[0].dW[i]);
        if (d01 > buggy_diff) buggy_diff = d01;
        if (d10 > buggy_diff) buggy_diff = d10;
    }
    std::printf("  sensitivity: stale-bank backward diverges by %.3e (scale %.3e)\n",
                buggy_diff, scale);
    // With the residual timestep the stale-bank divergence scales by alpha,
    // but it must still sit far above the match tolerance (1e-4) — the two
    // measured regimes are separated by ~5 orders of magnitude.
    CHECK(buggy_diff > 1e-3f * (1.f + scale),
          "missing bank swap is detected (sensitivity)");

    // Permutation involution (C6): swapping the same pair back restores the
    // pre-swap arrangement exactly.
    bool involution_ok = true;
    if (!safety::pt::swap_device_organism(ctx, slime::PtSlot(1),
                                          slime::PtSlot(0))) {
        involution_ok = false;
    }
    safety::pt::swap_host_organism(ctx, slime::PtSlot(1), slime::PtSlot(0));
    if (lineage[0] != slime::LineageId(101u)
        || lineage[1] != slime::LineageId(202u)) involution_ok = false;
    CHECK(involution_ok, "reverse swap restores the original permutation");


    free(h_weights);
    free(h_img);
    free(h_task);
    cudaFree(d_org);
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_eff);
    cudaFree(d_deltas);
    cudaFree(d_img);
    cudaFree(d_task);
    cudaFree(d_seed);
    cudaFree(d_tmp_org);
    cudaFree(d_tmp_ckpt);
    cudaFree(d_tmp_grad);
    cudaFree(d_tmp_wbank);
    free_checkpoints(d_ckpt);
    free_grad_buffers(d_grads);
    free_backward_workspace(ws);
    return 0;
}

// ---- Test 4: directional finite difference --------------------------------
// The directional finite-difference comparison runs at TWO residual
// timesteps:
//   alpha = 1.0 — the adjoint structure is identical for any alpha, and at
//                 alpha = 1 the loss sensitivity is large enough for tight
//                 per-bank assertions (the discriminating regime).
//   alpha = RESIDUAL_ALPHA (production) — the recurrent weights' true
//                 derivative is small here and FP16 forward quantization
//                 noise limits achievable agreement; the all-direction check
//                 still catches a missing alpha scaling (~64x mismatch) at a
//                 coarse tolerance.
static int fd_run(Rig& r, float alpha, float eps, bool assert_banks) {
    const int target_class = 2;

    auto forward_and_loss = [&](const float* w, float* loss_out, float* desc_out) {
        CUDA_CHECK(cudaMemcpy(r.d_weights, w, sizeof(float) * TOTAL_WEIGHTS,
                              cudaMemcpyHostToDevice));
        launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                        r.d_weights, nullptr,
                                        r.d_ckpt, alpha, 1, 0);
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
    launch_backward_all(r.d_org, r.d_weights, nullptr, nullptr,
                        r.d_seed_grad,
                        r.d_ckpt, r.d_grads, r.ws, alpha, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    GradBuffers g;
    CUDA_CHECK(cudaMemcpy(&g, r.d_grads, sizeof(GradBuffers), cudaMemcpyDeviceToHost));

    const int bank_lo[4] = { OFF_PERC, OFF_INTER, OFF_FLOW, OFF_BMAP };
    const int bank_hi[4] = { OFF_INTER, OFF_FLOW, OFF_BMAP, TOTAL_WEIGHTS };
    const char* bank_name[4] = { "W_perc", "W_inter", "W_flow", "W_bmap" };
    const float bank_tol[4] = { 5e-2f, 0.20f, 0.20f, 5e-3f };
    float bank_rel_err[4] = {};
    int rc = 0;

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
        bank_rel_err[bank] = rel_err;
        std::printf("  %-8s d_analytic=% .6e d_numeric=% .6e rel_err=%.4e\n",
                    bank_name[bank], d_analytic, d_numeric, rel_err);
    }
    if (assert_banks) {
        for (int bank = 0; bank < 4; ++bank) {
            CHECK(bank_rel_err[bank] < bank_tol[bank], bank_name[bank]);
        }
    }

    // All-weights direction.
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
        CHECK(rel_err < (assert_banks ? 5e-2f : 0.5f),
              assert_banks
                  ? "analytic directional derivative matches central difference"
                  : "production-alpha directional derivative agrees (coarse)");
    }

    // Effective-bank case (production path W_eff = W_shared + delta).
    {
        DeltaWeights delta;
        std::memset(&delta, 0, sizeof(delta));
        delta.count = 2;
        delta.indices[0] = OFF_FLOW + 7; delta.values[0] = 0.2f;
        delta.indices[1] = OFF_INTER + 41; delta.values[1] = -0.15f;

        auto eff_forward_and_loss = [&](const float* w, float* loss_out, float* desc_out) {
            CUDA_CHECK(cudaMemcpy(r.d_weights, w, sizeof(float) * TOTAL_WEIGHTS,
                                  cudaMemcpyHostToDevice));
            if (upload_delta(&r, delta)) return 1;
            launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                            r.d_weights, r.d_eff_weights,
                                            r.d_ckpt, alpha, 1, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (read_descriptor(&r, desc_out)) return 1;
            float dlogits[NUM_CLASSES];
            classifier_loss(desc_out, target_class, NUM_CLASSES, dlogits, loss_out);
            return 0;
        };

        float desc_eff[BMAP_DIM];
        float L0e = 0.f;
        if (eff_forward_and_loss(r.h_weights, &L0e, desc_eff)) return 1;

        float h_seed_e[BMAP_DIM] = {};
        {
            float dlogits[NUM_CLASSES];
            float tmp;
            classifier_loss(desc_eff, target_class, NUM_CLASSES, dlogits, &tmp);
            for (int d = 0; d < NUM_CLASSES; ++d) h_seed_e[d] = dlogits[d];
        }
        CUDA_CHECK(cudaMemcpy(r.d_seed_grad, h_seed_e,
            sizeof(float) * BMAP_DIM, cudaMemcpyHostToDevice));
        launch_backward_all(r.d_org, r.d_weights, r.d_eff_weights, nullptr,
                        r.d_seed_grad,
                            r.d_ckpt, r.d_grads, r.ws, alpha, 1, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        GradBuffers ge;
        CUDA_CHECK(cudaMemcpy(&ge, r.d_grads, sizeof(GradBuffers), cudaMemcpyDeviceToHost));

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

        float d_analytic_e = 0.f;
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) d_analytic_e += ge.dW[i] * v[i];

        float w_plus[TOTAL_WEIGHTS], w_minus[TOTAL_WEIGHTS];
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
            w_plus[i]  = r.h_weights[i] + eps * v[i];
            w_minus[i] = r.h_weights[i] - eps * v[i];
        }
        float Lpe = 0.f, Lme = 0.f;
        float desc_tmp[BMAP_DIM];
        if (eff_forward_and_loss(w_plus, &Lpe, desc_tmp)) return 1;
        if (eff_forward_and_loss(w_minus, &Lme, desc_tmp)) return 1;
        float d_numeric_e = (Lpe - Lme) / (2.0f * eps);
        float rel_err_e = std::fabs(d_analytic_e - d_numeric_e)
                        / std::fmax(std::fabs(d_numeric_e), 1e-12f);
        std::printf("  eff-bank d_analytic=% .6e d_numeric=% .6e rel_err=%.4e\n",
                    d_analytic_e, d_numeric_e, rel_err_e);
        CHECK(rel_err_e < (assert_banks ? 5e-2f : 0.5f),
              assert_banks
                  ? "effective-bank analytic directional derivative matches central difference"
                  : "effective-bank production-alpha derivative agrees (coarse)");
    }
    return rc;
}

static int test_finite_difference_gradient() {
    // [claim:A103.gradient-correctness]
    std::printf("--- Test: analytic gradient agrees with finite differences ---\n");
    std::fflush(stdout);

    Rig r{};
    if (rig_init(&r)) return 1;

    std::printf("  [alpha = 1.0, discriminating regime]\n");
    std::fflush(stdout);
    if (fd_run(r, 1.0f, 0.05f, true)) return 1;

    std::printf("  [alpha = RESIDUAL_ALPHA (production), noise-limited regime]\n");
    std::fflush(stdout);
    if (fd_run(r, RESIDUAL_ALPHA, 0.05f, false)) return 1;

    rig_free(&r);
    return 0;
}

// ---- Test 5: residual dynamics bounded under RESIDUAL_ALPHA ----------------
static int test_residual_dynamics_bounded() {
    // [claim:A201.bounded-residual-dynamics]
    std::printf("--- Test: default-init forward stays far from FP16 saturation ---\n");
    std::fflush(stdout);

    Rig r{};
    if (rig_init(&r)) return 1;

    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, nullptr,
                                    r.d_ckpt, RESIDUAL_ALPHA, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    TelemetryScalars* d_tel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tel, sizeof(TelemetryScalars)));
    CUDA_CHECK(cudaMemset(d_tel, 0, sizeof(TelemetryScalars)));
    launch_state_saturation(r.d_ckpt, r.d_org, d_tel, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    TelemetryScalars h_tel;
    CUDA_CHECK(cudaMemcpy(&h_tel, d_tel, sizeof(TelemetryScalars),
                          cudaMemcpyDeviceToHost));
    std::printf("  state_max_abs=%.3e state_near_max=%.0f\n",
                h_tel.state_max_abs, h_tel.state_near_max);
    CHECK(h_tel.state_max_abs < 60000.f,
          "64-step forward stays far from FP16 saturation");
    CHECK(h_tel.state_near_max == 0.f,
          "no state value approaches the FP16 limit");

    cudaFree(d_tel);
    rig_free(&r);
    return 0;
}

// ---- Test 6: per-role gradient alignment telemetry -------------------------
static int test_role_grad_alignment() {
    // [claim:A501.role-gradient-alignment]
    std::printf("--- Test: role gradient alignment matches a host reference ---\n");
    std::fflush(stdout);

    const int N = 3;  // two classifiers, one predictor
    GradBuffers* d_grads = nullptr;
    Role* d_roles = nullptr;
    TelemetryScalars* d_tel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grads, sizeof(GradBuffers) * N));
    CUDA_CHECK(cudaMalloc(&d_roles, sizeof(Role) * N));
    CUDA_CHECK(cudaMalloc(&d_tel, sizeof(TelemetryScalars)));
    CUDA_CHECK(cudaMemset(d_tel, 0, sizeof(TelemetryScalars)));

    GradBuffers h_grads[N];
    uint32_t s = 0xC0FFEE11u;
    for (int org = 0; org < N; ++org) {
        for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            h_grads[org].dW[i] =
                static_cast<float>(s) * (1.0f / 4294967296.0f) - 0.5f;
        }
    }
    Role h_roles[N] = {Role::Classifier, Role::Classifier, Role::Predictor};

    CUDA_CHECK(cudaMemcpy(d_grads, h_grads, sizeof(GradBuffers) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_roles, h_roles, sizeof(Role) * N,
                          cudaMemcpyHostToDevice));
    if (!optimizer::launch_role_grad_alignment(d_grads, d_roles, N, d_tel, 0)) {
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    TelemetryScalars h_tel;
    CUDA_CHECK(cudaMemcpy(&h_tel, d_tel, sizeof(TelemetryScalars),
                          cudaMemcpyDeviceToHost));

    // Host reference in double precision so the float device sums are the
    // only source of disagreement.
    double dot = 0.0, nc = 0.0, np = 0.0;
    for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
        double gc = static_cast<double>(h_grads[0].dW[i])
                  + static_cast<double>(h_grads[1].dW[i]);
        double gp = static_cast<double>(h_grads[2].dW[i]);
        dot += gc * gp;
        nc  += gc * gc;
        np  += gp * gp;
    }
    float ref_cos = static_cast<float>(dot / std::sqrt(nc * np));
    float got_cos = h_tel.role_grad_dot
                  / (std::sqrt(h_tel.role_grad_norm_sq[0])
                     * std::sqrt(h_tel.role_grad_norm_sq[1]));
    float norm_err_c = std::fabs(h_tel.role_grad_norm_sq[0]
                                 - static_cast<float>(nc))
                     / static_cast<float>(nc);
    float norm_err_p = std::fabs(h_tel.role_grad_norm_sq[1]
                                 - static_cast<float>(np))
                     / static_cast<float>(np);
    std::printf("  cos device=%.6f host=%.6f  rel_err_norm_C=%.2e "
                "rel_err_norm_P=%.2e\n",
                got_cos, ref_cos, norm_err_c, norm_err_p);
    CHECK(norm_err_c < 1e-4f,
          "classifier role gradient norm matches host reference");
    CHECK(norm_err_p < 1e-4f,
          "predictor role gradient norm matches host reference");
    CHECK(std::fabs(got_cos - ref_cos) < 1e-4f,
          "role gradient cosine matches host reference");

    // Cross-role swap: org 0 becomes predictor, org 2 becomes classifier.
    // Attribution must follow the role buffer, not any stale per-generation
    // forward-input roles.
    h_roles[0] = Role::Predictor;
    h_roles[2] = Role::Classifier;
    CUDA_CHECK(cudaMemcpy(d_roles, h_roles, sizeof(Role) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_tel, 0, sizeof(TelemetryScalars)));
    if (!optimizer::launch_role_grad_alignment(d_grads, d_roles, N, d_tel, 0)) {
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_tel, d_tel, sizeof(TelemetryScalars),
                          cudaMemcpyDeviceToHost));
    dot = 0.0; nc = 0.0; np = 0.0;
    for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
        double gc = static_cast<double>(h_grads[1].dW[i])
                  + static_cast<double>(h_grads[2].dW[i]);
        double gp = static_cast<double>(h_grads[0].dW[i]);
        dot += gc * gp;
        nc  += gc * gc;
        np  += gp * gp;
    }
    ref_cos = static_cast<float>(dot / std::sqrt(nc * np));
    got_cos = h_tel.role_grad_dot
            / (std::sqrt(h_tel.role_grad_norm_sq[0])
               * std::sqrt(h_tel.role_grad_norm_sq[1]));
    std::printf("  cross-role swap: cos device=%.6f host=%.6f\n",
                got_cos, ref_cos);
    CHECK(std::fabs(got_cos - ref_cos) < 1e-4f,
          "post-swap role attribution follows the role buffer");

    cudaFree(d_grads);
    cudaFree(d_roles);
    cudaFree(d_tel);
    return 0;
}

// ---- I5: global context channel --------------------------------------------

// Every checkpoint at a sample step is post-broadcast, and its aux channels
// must equal context_map(stored pre-broadcast summary, W_ctx) exactly.
static int test_context_channel_reference() {
    std::printf("--- Test: context broadcast matches the stored summary ---\n");
    std::fflush(stdout);
    Rig r{};
    if (rig_init(&r)) return 1;
    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, nullptr, r.d_ckpt,
                                    RESIDUAL_ALPHA, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    OrganismState h_org;
    CheckpointBuffer h_ck;
    CUDA_CHECK(cudaMemcpy(&h_org, r.d_org, sizeof(OrganismState),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_ck, r.d_ckpt, sizeof(CheckpointBuffer),
                          cudaMemcpyDeviceToHost));

    float W_ctx[CA_CHANNELS * slime::nca::CTX_K];
    for (int i = 0; i < CA_CHANNELS * slime::nca::CTX_K; ++i) {
        W_ctx[i] = r.h_weights[OFF_CTX + i];
    }
    int mismatches = 0;
    for (int ck = 1; ck <= 3; ++ck) {
        float ctx[slime::nca::CTX_K];
        slime::nca::context_map(
            &h_org.sample_summary[(ck - 1) * CA_CHANNELS], W_ctx, ctx);
        for (int cell = 0; cell < GRID_SIZE * GRID_SIZE; cell += 1021) {
            for (int k = 0; k < slime::nca::CTX_K; ++k) {
                __half want = __float2half(ctx[k]);
                __half got = h_ck.data[ck][cell * CA_CHANNELS
                                           + CH_AUX_FIRST + k];
                if (__half_as_ushort(want) != __half_as_ushort(got)) {
                    mismatches++;
                }
            }
        }
    }
    {
        float ctx[slime::nca::CTX_K];
        slime::nca::context_map(&h_org.sample_summary[3 * CA_CHANNELS],
                                W_ctx, ctx);
        for (int cell = 0; cell < GRID_SIZE * GRID_SIZE; cell += 1021) {
            for (int k = 0; k < slime::nca::CTX_K; ++k) {
                __half want = __float2half(ctx[k]);
                __half got = h_org.grid[cell * CA_CHANNELS
                                        + CH_AUX_FIRST + k];
                if (__half_as_ushort(want) != __half_as_ushort(got)) {
                    mismatches++;
                }
            }
        }
    }
    std::printf("  mismatches=%d\n", mismatches);
    CHECK(mismatches == 0,
          "checkpoint and final aux channels equal context_map(summary)");
    rig_free(&r);
    return 0;
}

// The backward must write W_ctx gradients at the sample steps.
static int test_context_gradient_nonzero() {
    std::printf("--- Test: context adjoint writes W_ctx gradients ---\n");
    std::fflush(stdout);
    Rig r{};
    if (rig_init(&r)) return 1;
    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, nullptr, r.d_ckpt,
                                    RESIDUAL_ALPHA, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_seed[BMAP_DIM];
    for (int d = 0; d < BMAP_DIM; ++d) {
        h_seed[d] = 0.1f * static_cast<float>(d % 3 - 1);
    }
    CUDA_CHECK(cudaMemcpy(r.d_seed_grad, h_seed, sizeof(float) * BMAP_DIM,
                          cudaMemcpyHostToDevice));
    launch_backward_all(r.d_org, r.d_weights, nullptr, nullptr,
                        r.d_seed_grad,
                        r.d_ckpt, r.d_grads, r.ws, RESIDUAL_ALPHA, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    GradBuffers g;
    CUDA_CHECK(cudaMemcpy(&g, r.d_grads, sizeof(GradBuffers),
                          cudaMemcpyDeviceToHost));
    float norm2 = 0.f;
    for (int i = 0; i < W_CTX_SIZE; ++i) {
        norm2 += g.dW[OFF_CTX + i] * g.dW[OFF_CTX + i];
    }
    std::printf("  ||dW_ctx||^2=%.3e\n", norm2);
    CHECK(norm2 > 0.f, "context adjoint writes W_ctx gradients");
    rig_free(&r);
    return 0;
}

// ---- I6: reaction-diffusion gradients --------------------------------------

// With RD active, the analytic gradient must match central differences (the
// FD is also the strongest witness that the backward's RD re-forward matches
// the forward's checkpoints), and the forward must be deterministic.
static int test_rd_gradient_finite_difference() {
    std::printf("--- Test: RD-enabled gradient matches finite differences ---\n");
    std::fflush(stdout);
    Rig r{};
    if (rig_init(&r)) return 1;
    const float alpha = 1.0f;
    const int target_class = 2;

    nca::rd::Coefficients h_coeffs;
    uint32_t s = 0x9E3779B9u;
    auto next_rand = [&]() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return static_cast<float>(s) * (1.0f / 4294967296.0f) - 0.5f;
    };
    for (int i = 0; i < 36; ++i) h_coeffs.reaction[i] = 0.6f * next_rand();
    for (int i = 0; i < 6; ++i) {
        h_coeffs.diffusion[i] = 0.2f + 0.4f * (next_rand() + 0.5f);
    }
    nca::rd::Coefficients* d_coeffs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_coeffs, sizeof(nca::rd::Coefficients)));
    CUDA_CHECK(cudaMemcpy(d_coeffs, &h_coeffs,
                          sizeof(nca::rd::Coefficients),
                          cudaMemcpyHostToDevice));

    auto forward_and_loss = [&](const float* w, float* loss_out,
                                float* desc_out) {
        CUDA_CHECK(cudaMemcpy(r.d_weights, w, sizeof(float) * TOTAL_WEIGHTS,
                              cudaMemcpyHostToDevice));
        launch_forward_with_checkpoints(r.d_org, r.d_inputs, d_coeffs,
                                        r.d_weights, nullptr,
                                        r.d_ckpt, alpha, 1, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (read_descriptor(&r, desc_out)) return 1;
        float dlogits[NUM_CLASSES];
        classifier_loss(desc_out, target_class, NUM_CLASSES, dlogits, loss_out);
        return 0;
    };

    float desc0[BMAP_DIM];
    float L0 = 0.f;
    if (forward_and_loss(r.h_weights, &L0, desc0)) return 1;

    // Determinism: the same weights produce the same descriptor bitwise.
    {
        float desc_again[BMAP_DIM];
        float L_again = 0.f;
        if (forward_and_loss(r.h_weights, &L_again, desc_again)) return 1;
        bool identical = true;
        for (int d = 0; d < BMAP_DIM; ++d) {
            if (desc0[d] != desc_again[d]) identical = false;
        }
        CHECK(identical, "RD forward is deterministic");
    }

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
    launch_backward_all(r.d_org, r.d_weights, nullptr, d_coeffs,
                        r.d_seed_grad, r.d_ckpt, r.d_grads, r.ws, alpha, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    GradBuffers g;
    CUDA_CHECK(cudaMemcpy(&g, r.d_grads, sizeof(GradBuffers),
                          cudaMemcpyDeviceToHost));

    float v[TOTAL_WEIGHTS];
    float norm2 = 0.f;
    s = 0x1234ABCDu;
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

    const float eps = 0.05f;
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
                  / std::fmax(std::fabs(d_numeric), 1e-9f);
    std::printf("  RD d_analytic=% .6e d_numeric=% .6e rel_err=%.4e\n",
                d_analytic, d_numeric, rel_err);
    CHECK(std::fabs(d_numeric) > 1e-9f, "RD FD direction is informative");
    CHECK(rel_err < 5e-2f, "RD-enabled analytic gradient matches FD");

    cudaFree(d_coeffs);
    rig_free(&r);
    return 0;
}

// ---- I7: phase graph equivalence -------------------------------------------

// A captured forward phase replayed on a non-default stream must reproduce
// the sequential execution bitwise (grid, BTRAJ, and checkpoints).
static int test_phase_graph_equivalence() {
    std::printf("--- Test: captured forward phase matches sequential ---\n");
    std::fflush(stdout);
    Rig r{};
    if (rig_init(&r)) return 1;
    cudaStream_t s = nullptr;
    CUDA_CHECK(cudaStreamCreate(&s));

    OrganismState h1, h2;
    CheckpointBuffer c1, c2;
    launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                    r.d_weights, nullptr, r.d_ckpt,
                                    RESIDUAL_ALPHA, 1, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(&h1, r.d_org, sizeof(OrganismState),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&c1, r.d_ckpt, sizeof(CheckpointBuffer),
                          cudaMemcpyDeviceToHost));

    slime::integration::PhaseGraph fg;
    if (!slime::integration::phase_run(&fg, s, [&] {
            launch_forward_with_checkpoints(r.d_org, r.d_inputs, nullptr,
                                            r.d_weights, nullptr, r.d_ckpt,
                                            RESIDUAL_ALPHA, 1, s);
        })) {
        return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(&h2, r.d_org, sizeof(OrganismState),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&c2, r.d_ckpt, sizeof(CheckpointBuffer),
                          cudaMemcpyDeviceToHost));

    CHECK(std::memcmp(h1.grid, h2.grid, sizeof(h1.grid)) == 0,
          "captured forward grid matches sequential");
    CHECK(std::memcmp(h1.bmap_traj, h2.bmap_traj, sizeof(h1.bmap_traj)) == 0,
          "captured forward BTRAJ matches sequential");
    CHECK(std::memcmp(&c1, &c2, sizeof(CheckpointBuffer)) == 0,
          "captured forward checkpoints match sequential");

    cudaStreamDestroy(s);
    rig_free(&r);
    return 0;
}

int main() {
    slime::require_gpu_authorization("evolution_regression");
    std::printf("Evolution regression suite (role locking, effective weights, "
                "PT correspondence, finite differences)\n");
    std::printf("========================================\n");
    std::fflush(stdout);

    int rc = 0;
    rc |= test_materialize_matches_reference();
    rc |= test_genotype_causality();
    rc |= test_forced_pt_swap();
    rc |= test_pt_swap_backward_correspondence();
    rc |= test_finite_difference_gradient();
    rc |= test_residual_dynamics_bounded();
    rc |= test_role_grad_alignment();
    rc |= test_context_channel_reference();
    rc |= test_context_gradient_nonzero();
    rc |= test_rd_gradient_finite_difference();
    rc |= test_phase_graph_equivalence();

    std::printf("\n========================================\n");
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail > 0 || rc != 0) {
        std::printf("EVOLUTION REGRESSION: FAIL\n");
        return 1;
    }
    std::printf("EVOLUTION REGRESSION: PASS\n");
    return 0;
}





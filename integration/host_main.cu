// Host main: allocation, initialization, generation loop, entry point
//
// Per cuda_engineering.md sections 2, 3, 5, 6, 8, 9, 12, 13, 15 and
// Predictors participate once the bootstrap injects founders; before that
// every spawn is role-locked to classifier.

#include "main_loop.cu"
#include "../config/strong_ids.cuh"
#include "../safety/alignment.cu"
#include "checkpointing.cu"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thread>

namespace slime::integration {

using namespace slime;
namespace cur = slime::curriculum;

// Default checkpoint location (relative to the working directory). Every
// completed generation is saved here; `--resume` continues from it.
constexpr const char* CHECKPOINT_DEFAULT_PATH = "checkpoints/slime-ckpt.bin";


// ---- Fail-fast CUDA policy (A-501) -----------------------------------------
// A CUDA error invalidates the experiment: allocations, memsets, copies, and
// stream synchronizations all report failure and the run aborts cleanly
// instead of continuing with corrupted state.
#define CUDA_ABORT(expr, ctx) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        std::printf("[FATAL] CUDA %s failed at %s:%d: %s\n", \
                    ctx, __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::fflush(stdout); \
        return false; \
    } \
} while (0)

// Checked enqueue of an async transfer: the copy itself reports failures
// synchronously (invalid pointers, sizes, stream), while kernel/device errors
// surface at the following phase_trace synchronization.
#define TRANSFER_ABORT(expr, ctx) do { \
    cudaError_t _te = (expr); \
    if (_te != cudaSuccess) { \
        std::printf("[FATAL] CUDA transfer %s failed at %s:%d: %s\n", \
                    ctx, __FILE__, __LINE__, cudaGetErrorString(_te)); \
        std::fflush(stdout); \
        return false; \
    } \
} while (0)

// Shutdown-path checked call: reports a failure but cannot abort the run
// (used while releasing resources after the experiment has finished).
#define CUDA_WARN(expr, ctx) do { \
    cudaError_t _we = (expr); \
    if (_we != cudaSuccess) { \
        std::printf("[WARN] CUDA %s failed at %s:%d: %s\n", \
                    ctx, __FILE__, __LINE__, cudaGetErrorString(_we)); \
        std::fflush(stdout); \
    } \
} while (0)

// ---- Probe set evaluation (A-601) ------------------------------------------
// The probe batch is uploaded once at signing; each generation the device
// forward recomputes the per-tuple surprise on real
// (bmap_64, task_embedding, fitness) tuples snapshotted from the replay
// buffer at bootstrap. Returns the mean surprise, or -1 on launch failure.
static float evaluate_probe_reference(World* w) {
    if (!cur::verify_probe_set(w->probe_set, w->host_sot_key)) return 0.f;
    if (!w->probe_set.probe_tuples_signed) return 0.f;
    if (!phase_run(&w->fg_world_predict, w->stream, [&] {
            predictor::launch_reference_forward(
                w->d_reference_reg, w->d_ref_probe_input, PROBE_BATCH,
                nullptr, w->d_ref_probe_target, w->d_ref_surprise, w->stream);
        })) {
        return -1.f;
    }
    TRANSFER_ABORT(cudaMemcpyAsync(w->h_ref_surprise, w->d_ref_surprise,
                    PROBE_BATCH * sizeof(float), cudaMemcpyDeviceToHost,
                    w->stream), "read reference surprise");
    TRANSFER_ABORT(cudaStreamSynchronize(w->stream),
                   "reference surprise sync");
    float total = 0.f;
    for (int i = 0; i < PROBE_BATCH; ++i) total += w->h_ref_surprise[i];
    return total / static_cast<float>(PROBE_BATCH);
}

// Pack the signed probe tuples (bmap_64 + task embedding) into the device
// layout the forward kernel reads. Called at signing and after a checkpoint
// load that restores a signed probe set.
static bool upload_probe_batch(World* w) {
    for (int i = 0; i < PROBE_BATCH; ++i) {
        std::memcpy(&w->h_ref_probe_input[i * predictor::REF_INPUT],
                    &w->probe_set.probe_bmap[i * BMAP_DIM],
                    BMAP_DIM * sizeof(float));
        std::memcpy(&w->h_ref_probe_input[i * predictor::REF_INPUT + BMAP_DIM],
                    &w->probe_set.probe_task_emb[i * TASK_EMBED_DIM],
                    TASK_EMBED_DIM * sizeof(float));
    }
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_ref_probe_input, w->h_ref_probe_input,
                    PROBE_BATCH * predictor::REF_INPUT * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream), "upload probe input");
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_ref_probe_target,
                    w->probe_set.probe_fitness,
                    PROBE_BATCH * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream), "upload probe target");
    return true;
}

// Decode every organism's reaction-diffusion coefficients from its genome
// and upload them for the forward/backward kernels (A-202, I6).
static bool upload_rd_coefficients(World* w) {
    for (int i = 0; i < TOTAL_ORG; ++i) {
        nca::rd::decode_coefficients(w->org_table.genomes[i].bits,
                                     &w->h_rd_coeffs[i]);
    }
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_rd_coeffs, w->h_rd_coeffs,
                    TOTAL_ORG * sizeof(nca::rd::Coefficients),
                    cudaMemcpyHostToDevice, w->stream),
                   "upload rd coefficients");
    return true;
}

// Parameters and AdamW state are device-resident; mirror them into the host
// struct before the serializer writes, and back after a load.
static bool sync_reference_from_device(World* w) {
    TRANSFER_ABORT(cudaMemcpyAsync(&w->reference_reg, w->d_reference_reg,
                    sizeof(predictor::ReferenceRegressor),
                    cudaMemcpyDeviceToHost, w->stream),
                   "sync reference from device");
    TRANSFER_ABORT(cudaStreamSynchronize(w->stream), "reference sync");
    return true;
}

static bool sync_reference_to_device(World* w) {
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_reference_reg, &w->reference_reg,
                    sizeof(predictor::ReferenceRegressor),
                    cudaMemcpyHostToDevice, w->stream),
                   "sync reference to device");
    return true;
}

// ---- GPU buffer allocation/free (section 2) --------------------------------

static bool alloc_gpu_buffers(World* w) {
    CUDA_ABORT(cudaStreamCreate(&w->stream), "stream create");

    // Section 2.1: device-resident buffers.
    CUDA_ABORT(cudaMalloc(&w->d_organisms,    TOTAL_ORG * sizeof(OrganismState)), "alloc d_organisms");
    CUDA_ABORT(cudaMalloc(&w->d_weights,      TOTAL_WEIGHTS * sizeof(float)), "alloc d_weights");
    CUDA_ABORT(cudaMalloc(&w->d_eff_weights,  POOL_SIZE * TOTAL_WEIGHTS * sizeof(float)), "alloc d_eff_weights");
    CUDA_ABORT(cudaMalloc(&w->d_deltas,       POOL_SIZE * sizeof(genome::DeltaWeights)), "alloc d_deltas");
    CUDA_ABORT(cudaMalloc(&w->d_fwd_inputs,   POOL_SIZE * sizeof(ForwardInputs)), "alloc d_fwd_inputs");
    CUDA_ABORT(cudaMalloc(&w->d_roles,        POOL_SIZE * sizeof(Role)), "alloc d_roles");
    CUDA_ABORT(cudaMalloc(&w->d_checkpoints,  POOL_SIZE * sizeof(CheckpointBuffer)), "alloc d_checkpoints");
    CUDA_ABORT(cudaMalloc(&w->d_grads,        POOL_SIZE * sizeof(GradBuffers)), "alloc d_grads");
    CUDA_ABORT(cudaMalloc(&w->d_mean_grad,    TOTAL_WEIGHTS * sizeof(float)), "alloc d_mean_grad");
    CUDA_ABORT(cudaMalloc(&w->d_came_m,       TOTAL_WEIGHTS * sizeof(float)), "alloc d_came_m");
    CUDA_ABORT(cudaMalloc(&w->d_came_v,       TOTAL_WEIGHTS * sizeof(float)), "alloc d_came_v");
    CUDA_ABORT(cudaMalloc(&w->d_came_c,       TOTAL_WEIGHTS * sizeof(float)), "alloc d_came_c");
    CUDA_ABORT(cudaMalloc(&w->d_came_prev_u,  TOTAL_WEIGHTS * sizeof(float)), "alloc d_came_prev_u");
    CUDA_ABORT(cudaMalloc(&w->d_descriptors,  POOL_SIZE * BMAP_DIM * sizeof(float)), "alloc d_descriptors");
    CUDA_ABORT(cudaMalloc(&w->d_seed_grad,    POOL_SIZE * BMAP_DIM * sizeof(float)), "alloc d_seed_grad");
    CUDA_ABORT(cudaMalloc(&w->d_batch_image,  cur::CLASSIFIER_BATCH * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half)), "alloc d_batch_image");
    CUDA_ABORT(cudaMalloc(&w->d_batch_task_emb, TASK_EMBED_DIM * sizeof(float)), "alloc d_batch_task_emb");
    CUDA_ABORT(cudaMalloc(&w->d_btraj,        POOL_SIZE * BTRAJ_SAMPLES * BMAP_DIM * sizeof(float)), "alloc d_btraj");
    CUDA_ABORT(cudaMalloc(&w->d_predictor_bmap32, cur::PREDICTOR_BATCH * BMAP_DIM * sizeof(float)), "alloc d_predictor_bmap32");

    // Device reference state and probe batch (cuda_engineering 4.5-4.6, T5).
    CUDA_ABORT(cudaMalloc(&w->d_reference_reg, sizeof(predictor::ReferenceRegressor)), "alloc d_reference_reg");
    CUDA_ABORT(cudaMalloc(&w->d_ref_batch_input, REF_TRAIN_MINIBATCH * predictor::REF_INPUT * sizeof(float)), "alloc d_ref_batch_input");
    CUDA_ABORT(cudaMalloc(&w->d_ref_batch_target, REF_TRAIN_MINIBATCH * sizeof(float)), "alloc d_ref_batch_target");
    CUDA_ABORT(cudaMalloc(&w->d_ref_probe_input, PROBE_BATCH * predictor::REF_INPUT * sizeof(float)), "alloc d_ref_probe_input");
    CUDA_ABORT(cudaMalloc(&w->d_ref_probe_target, PROBE_BATCH * sizeof(float)), "alloc d_ref_probe_target");
    CUDA_ABORT(cudaMalloc(&w->d_ref_surprise, PROBE_BATCH * sizeof(float)), "alloc d_ref_surprise");
    CUDA_ABORT(cudaMalloc(&w->d_ref_step, sizeof(int)), "alloc d_ref_step");
    CUDA_ABORT(cudaMalloc(&w->d_stress_eff_weights, STRESS_POOL_SIZE * TOTAL_WEIGHTS * sizeof(float)), "alloc d_stress_eff_weights");
    CUDA_ABORT(cudaMalloc(&w->d_stress_batch_image, STRESS_SUBPOP_COUNT * cur::CLASSIFIER_BATCH * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half)), "alloc d_stress_batch_image");
    CUDA_ABORT(cudaMalloc(&w->d_stress_ref_images, STRESS_SUBPOP_COUNT * cur::SOT_SUBBATCH * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half)), "alloc d_stress_ref_images");
    CUDA_ABORT(cudaMalloc(&w->d_stress_task_embs, STRESS_SUBPOP_COUNT * TASK_EMBED_DIM * sizeof(float)), "alloc d_stress_task_embs");
    CUDA_ABORT(cudaMalloc(&w->d_stress_ref_organisms, STRESS_POOL_SIZE * sizeof(OrganismState)), "alloc d_stress_ref_organisms");
    CUDA_ABORT(cudaMalloc(&w->d_stress_targets, 2 * STRESS_POOL_SIZE * BMAP_DIM * sizeof(float)), "alloc d_stress_targets");
    CUDA_ABORT(cudaMalloc(&w->d_rd_coeffs, TOTAL_ORG * sizeof(nca::rd::Coefficients)), "alloc d_rd_coeffs");
    CUDA_ABORT(cudaMalloc(&w->d_sot_ref_coeffs, cur::SOT_MAX_REFS * sizeof(nca::rd::Coefficients)), "alloc d_sot_ref_coeffs");

    // Gradient health: pinned host scalar (section 8).
    CUDA_ABORT(cudaMalloc(&w->d_grad_norm, sizeof(float)), "alloc d_grad_norm");

    // Numerical telemetry (A-501).
    CUDA_ABORT(cudaMalloc(&w->d_tel, sizeof(TelemetryScalars)), "alloc d_tel");

    // SOT reference buffers (section 12): pre-allocated, bounded by SOT_MAX_REFS.
    CUDA_ABORT(cudaMalloc(&w->d_sot_temp_images, cur::SOT_SUBBATCH * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half)), "alloc d_sot_temp_images");
    CUDA_ABORT(cudaMalloc(&w->d_sot_task_emb,    TASK_EMBED_DIM * sizeof(float)), "alloc d_sot_task_emb");
    CUDA_ABORT(cudaMalloc(&w->d_sot_fwd_inputs,  STRESS_POOL_SIZE * sizeof(ForwardInputs)), "alloc d_sot_fwd_inputs");
    CUDA_ABORT(cudaMalloc(&w->d_sot_descriptors, STRESS_POOL_SIZE * BMAP_DIM * sizeof(float)), "alloc d_sot_descriptors");
    CUDA_ABORT(cudaMalloc(&w->d_sot_bank_of,     STRESS_POOL_SIZE * sizeof(int)), "alloc d_sot_bank_of");
    CUDA_ABORT(cudaMalloc(&w->d_sot_ref_organisms, cur::SOT_MAX_REFS * sizeof(OrganismState)), "alloc d_sot_ref_organisms");

    // PT swap temp buffers (section 13): one organism's worth each.
    CUDA_ABORT(cudaMalloc(&w->d_pt_swap_org,  sizeof(OrganismState)), "alloc d_pt_swap_org");
    CUDA_ABORT(cudaMalloc(&w->d_pt_swap_ckpt, sizeof(CheckpointBuffer)), "alloc d_pt_swap_ckpt");
    CUDA_ABORT(cudaMalloc(&w->d_pt_swap_grad, sizeof(GradBuffers)), "alloc d_pt_swap_grad");
    CUDA_ABORT(cudaMalloc(&w->d_pt_swap_wbank, TOTAL_WEIGHTS * sizeof(float)), "alloc d_pt_swap_wbank");

    // Section 10: backward workspace (per-organism, for batched backward).
    constexpr int GRID_ELEMS = GRID_SIZE * GRID_SIZE * CA_CHANNELS;
    constexpr int PERC_ELEMS = GRID_SIZE * GRID_SIZE * autodiff::PERC_DIM;
    w->bwd_workspace.n_organisms = POOL_SIZE;
    CUDA_ABORT(cudaMalloc(&w->bwd_workspace.d_state[0], GRID_ELEMS * sizeof(float) * POOL_SIZE), "alloc bwd d_state[0]");
    CUDA_ABORT(cudaMalloc(&w->bwd_workspace.d_state[1], GRID_ELEMS * sizeof(float) * POOL_SIZE), "alloc bwd d_state[1]");
    CUDA_ABORT(cudaMalloc(&w->bwd_workspace.d_perc,     PERC_ELEMS * sizeof(float) * POOL_SIZE), "alloc bwd d_perc");
    CUDA_ABORT(cudaMalloc(&w->bwd_workspace.d_seg_states,
        static_cast<size_t>(CHECKPOINT_INTERVAL) * POOL_SIZE * GRID_ELEMS
            * sizeof(__half)), "alloc bwd d_seg_states");
    CUDA_ABORT(cudaMalloc(&w->bwd_workspace.d_cell_stage,
        static_cast<size_t>(autodiff::STAGE_STRIDE) * POOL_SIZE
            * sizeof(float)), "alloc bwd d_cell_stage");
    CUDA_ABORT(cudaMalloc(&w->bwd_workspace.d_seed_aux,
        static_cast<size_t>(autodiff::CTX_K) * POOL_SIZE
            * sizeof(float)), "alloc bwd d_seed_aux");
    CUDA_ABORT(cudaMalloc(&w->bwd_workspace.d_rd_g,
        static_cast<size_t>(POOL_SIZE) * GRID_SIZE * GRID_SIZE
            * (CH_CHEM_LAST + 1) * sizeof(float)), "alloc bwd d_rd_g");

    // Section 2.2: pinned host buffers.
    CUDA_ABORT(cudaMallocHost(&w->h_descriptors, POOL_SIZE * BMAP_DIM * sizeof(float)), "allocHost h_descriptors");
    CUDA_ABORT(cudaMallocHost(&w->h_btraj,       POOL_SIZE * BTRAJ_SAMPLES * BMAP_DIM * sizeof(float)), "allocHost h_btraj");
    CUDA_ABORT(cudaMallocHost(&w->h_seed_grad,   POOL_SIZE * BMAP_DIM * sizeof(float)), "allocHost h_seed_grad");
    CUDA_ABORT(cudaMallocHost(&w->h_fwd_inputs,  POOL_SIZE * sizeof(ForwardInputs)), "allocHost h_fwd_inputs");
    CUDA_ABORT(cudaMallocHost(&w->h_weights,     TOTAL_WEIGHTS * sizeof(float)), "allocHost h_weights");
    CUDA_ABORT(cudaMallocHost(&w->h_tel,         sizeof(TelemetryScalars)), "allocHost h_tel");
    CUDA_ABORT(cudaMallocHost(&w->h_ref_surprise, PROBE_BATCH * sizeof(float)), "allocHost h_ref_surprise");

    // Zero CAME state on device.
    CUDA_ABORT(cudaMemset(w->d_came_m,      0, TOTAL_WEIGHTS * sizeof(float)), "memset d_came_m");
    CUDA_ABORT(cudaMemset(w->d_came_v,      0, TOTAL_WEIGHTS * sizeof(float)), "memset d_came_v");
    CUDA_ABORT(cudaMemset(w->d_came_c,      0, TOTAL_WEIGHTS * sizeof(float)), "memset d_came_c");
    CUDA_ABORT(cudaMemset(w->d_came_prev_u, 0, TOTAL_WEIGHTS * sizeof(float)), "memset d_came_prev_u");

    // Zero organism grids.
    CUDA_ABORT(cudaMemset(w->d_organisms, 0, TOTAL_ORG * sizeof(OrganismState)), "memset d_organisms");
    return true;
}

static void free_gpu_buffers(World* w) {
    CUDA_WARN(cudaFree(w->d_organisms), "free d_organisms");
    CUDA_WARN(cudaFree(w->d_weights), "free d_weights");
    CUDA_WARN(cudaFree(w->d_eff_weights), "free d_eff_weights");
    CUDA_WARN(cudaFree(w->d_deltas), "free d_deltas");
    CUDA_WARN(cudaFree(w->d_fwd_inputs), "free d_fwd_inputs");
    CUDA_WARN(cudaFree(w->d_roles), "free d_roles");
    CUDA_WARN(cudaFree(w->d_checkpoints), "free d_checkpoints");
    CUDA_WARN(cudaFree(w->d_grads), "free d_grads");
    CUDA_WARN(cudaFree(w->d_mean_grad), "free d_mean_grad");
    CUDA_WARN(cudaFree(w->d_came_m), "free d_came_m");
    CUDA_WARN(cudaFree(w->d_came_v), "free d_came_v");
    CUDA_WARN(cudaFree(w->d_came_c), "free d_came_c");
    CUDA_WARN(cudaFree(w->d_came_prev_u), "free d_came_prev_u");
    CUDA_WARN(cudaFree(w->d_descriptors), "free d_descriptors");
    CUDA_WARN(cudaFree(w->d_seed_grad), "free d_seed_grad");
    CUDA_WARN(cudaFree(w->d_batch_image), "free d_batch_image");
    CUDA_WARN(cudaFree(w->d_batch_task_emb), "free d_batch_task_emb");
    CUDA_WARN(cudaFree(w->d_btraj), "free d_btraj");
    CUDA_WARN(cudaFree(w->d_predictor_bmap32), "free d_predictor_bmap32");
    CUDA_WARN(cudaFree(w->d_reference_reg), "free d_reference_reg");
    CUDA_WARN(cudaFree(w->d_ref_batch_input), "free d_ref_batch_input");
    CUDA_WARN(cudaFree(w->d_ref_batch_target), "free d_ref_batch_target");
    CUDA_WARN(cudaFree(w->d_ref_probe_input), "free d_ref_probe_input");
    CUDA_WARN(cudaFree(w->d_ref_probe_target), "free d_ref_probe_target");
    CUDA_WARN(cudaFree(w->d_ref_surprise), "free d_ref_surprise");
    CUDA_WARN(cudaFree(w->d_ref_step), "free d_ref_step");
    CUDA_WARN(cudaFree(w->d_stress_eff_weights), "free d_stress_eff_weights");
    CUDA_WARN(cudaFree(w->d_stress_batch_image), "free d_stress_batch_image");
    CUDA_WARN(cudaFree(w->d_stress_ref_images), "free d_stress_ref_images");
    CUDA_WARN(cudaFree(w->d_stress_task_embs), "free d_stress_task_embs");
    CUDA_WARN(cudaFree(w->d_stress_ref_organisms), "free d_stress_ref_organisms");
    CUDA_WARN(cudaFree(w->d_stress_targets), "free d_stress_targets");
    CUDA_WARN(cudaFree(w->d_rd_coeffs), "free d_rd_coeffs");
    CUDA_WARN(cudaFree(w->d_sot_ref_coeffs), "free d_sot_ref_coeffs");
    CUDA_WARN(cudaFree(w->d_grad_norm), "free d_grad_norm");
    CUDA_WARN(cudaFree(w->d_tel), "free d_tel");
    CUDA_WARN(cudaFree(w->d_sot_temp_images), "free d_sot_temp_images");
    CUDA_WARN(cudaFree(w->d_sot_task_emb), "free d_sot_task_emb");
    CUDA_WARN(cudaFree(w->d_sot_fwd_inputs), "free d_sot_fwd_inputs");
    CUDA_WARN(cudaFree(w->d_sot_descriptors), "free d_sot_descriptors");
    CUDA_WARN(cudaFree(w->d_sot_bank_of), "free d_sot_bank_of");
    CUDA_WARN(cudaFree(w->d_sot_ref_organisms), "free d_sot_ref_organisms");
    CUDA_WARN(cudaFree(w->d_pt_swap_org), "free d_pt_swap_org");
    CUDA_WARN(cudaFree(w->d_pt_swap_ckpt), "free d_pt_swap_ckpt");
    CUDA_WARN(cudaFree(w->d_pt_swap_grad), "free d_pt_swap_grad");
    CUDA_WARN(cudaFree(w->d_pt_swap_wbank), "free d_pt_swap_wbank");
    CUDA_WARN(cudaFree(w->bwd_workspace.d_state[0]), "free d_state[0]");
    CUDA_WARN(cudaFree(w->bwd_workspace.d_state[1]), "free d_state[1]");
    CUDA_WARN(cudaFree(w->bwd_workspace.d_perc), "free d_perc");
    CUDA_WARN(cudaFree(w->bwd_workspace.d_seg_states), "free d_seg_states");
    CUDA_WARN(cudaFree(w->bwd_workspace.d_cell_stage), "free d_cell_stage");
    CUDA_WARN(cudaFree(w->bwd_workspace.d_seed_aux), "free d_seed_aux");
    CUDA_WARN(cudaFree(w->bwd_workspace.d_rd_g), "free d_rd_g");
    CUDA_WARN(cudaFreeHost(w->h_descriptors), "freeHost h_descriptors");
    CUDA_WARN(cudaFreeHost(w->h_btraj), "freeHost h_btraj");
    CUDA_WARN(cudaFreeHost(w->h_seed_grad), "freeHost h_seed_grad");
    CUDA_WARN(cudaFreeHost(w->h_fwd_inputs), "freeHost h_fwd_inputs");
    CUDA_WARN(cudaFreeHost(w->h_weights), "freeHost h_weights");
    CUDA_WARN(cudaFreeHost(w->h_tel), "freeHost h_tel");
    CUDA_WARN(cudaFreeHost(w->h_ref_surprise), "freeHost h_ref_surprise");
    CUDA_WARN(cudaStreamDestroy(w->stream), "destroy stream");
}

// ---- Kaiming He weight initialization (section 15.2) ------------------------
// Box-Muller transform: given two uniform [0,1) draws, produce a standard
// normal sample.
static float box_muller_normal(Pcg32* rng) {
    float u1 = pcg32_float(rng);
    float u2 = pcg32_float(rng);
    // Avoid log(0).
    if (u1 < EPS_LOG) u1 = EPS_LOG;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

static void kaiming_he_init(float* weights, Pcg32* host_rng) {
    // Derive a sub-RNG for weight init from one host RNG draw (section 15.2).
    Pcg32 init_rng;
    pcg32_seed(&init_rng, pcg32_random(host_rng), 1);

    // W_perc: fan_in = 9, no activation -> scale = sqrt(1/9).
    {
        float scale = sqrtf(1.0f / 9.0f);
        for (int i = 0; i < W_PERC_SIZE; ++i) {
            weights[autodiff::OFF_PERC + i] = box_muller_normal(&init_rng) * scale;
        }
    }

    // W_inter: fan_in = PERC_DIM (48), GELU activation -> scale = sqrt(2/48).
    {
        float scale = sqrtf(2.0f / static_cast<float>(nca::PERC_DIM));
        for (int i = 0; i < autodiff::W_INTER_SIZE; ++i) {
            weights[autodiff::OFF_INTER + i] = box_muller_normal(&init_rng) * scale;
        }
    }

    // W_flow: fan_in = HIDDEN_DIM (32), linear -> scale = sqrt(1/32).
    {
        float scale = sqrtf(1.0f / static_cast<float>(HIDDEN_DIM));
        for (int i = 0; i < autodiff::W_FLOW_SIZE; ++i) {
            weights[autodiff::OFF_FLOW + i] = box_muller_normal(&init_rng) * scale;
        }
    }

    // W_bmap: fan_in = CA_CHANNELS (16), linear -> scale = sqrt(1/16).
    {
        float scale = sqrtf(1.0f / static_cast<float>(CA_CHANNELS));
        for (int i = 0; i < autodiff::W_BMAP_SIZE; ++i) {
            weights[autodiff::OFF_BMAP + i] = box_muller_normal(&init_rng) * scale;
        }
    }

    // W_ctx (A-203, I5): fan_in = CA_CHANNELS (16), linear -> sqrt(1/16).
    if (GLOBAL_CONTEXT_ENABLED) {
        for (int i = 0; i < autodiff::W_CTX_SIZE; ++i) {
            weights[autodiff::OFF_CTX + i] =
                box_muller_normal(&init_rng) * W_CTX_INIT_SCALE;
        }
    }
}

// ---- Initialization (section 15) --------------------------------------------

bool initialize_world(World* w) {
    std::memset(w, 0, sizeof(World));
    if (!alloc_gpu_buffers(w)) {
        std::printf("[FATAL] buffer allocation failed; experiment aborted\n");
        std::fflush(stdout);
        free_gpu_buffers(w);
        return false;
    }
    if (!safety::emit_cuda_diagnostics(w->stream)) {
        std::printf("[WARN] CUDA startup diagnostics did not complete\n");
    }

    // Section 15.1: PRNG.
    pcg32_seed(&w->rng, PCG32_DEFAULT_STATE, PCG32_DEFAULT_STREAM);

    w->generation = 0;
    w->bootstrap_fired = false;
    w->bootstrap_gen = -1;
    w->s_target = 0.f;
    w->s_target_calibrated = false;
    w->host_sot_key = 0xDEADCAFE42ULL;   // Section 15.3.
    w->grad_health_warn_count = 0;
    w->last_mean_ce = 0.f;
    w->last_max_abs_logit = 0.f;

    // Section 15.2: Kaiming He weight initialization.
    kaiming_he_init(w->h_weights, &w->rng);
    TRANSFER_ABORT(cudaMemcpy(w->d_weights, w->h_weights,
                              TOTAL_WEIGHTS * sizeof(float),
                              cudaMemcpyHostToDevice), "init weights");

    // Section 15.4: Initialize genomes, deltas, metadata.
    // Genome seeds drawn from host PCG32 — no index-based formulas.
    for (int i = 0; i < TOTAL_ORG; ++i) {
        genome::Genome& g = w->org_table.genomes[i];
        std::memset(&g, 0, sizeof(g));
        genome::write_role(g, Role::Classifier);

        // Draw 32-bit seed from PCG32.
        genome::write_seed(g, pcg32_random(&w->rng));

        genome::init_delta_from_prior(g, &w->org_table.deltas[i]);

        w->org_table.lineage_id[i] = LineageId(static_cast<uint32_t>(i));
        w->org_table.parent_id[i] = ArchiveSlot(0);
        w->org_table.spawn_gen[i] = 0;
        w->org_table.role[i] = genome::read_role(g);
        w->org_table.fitness[i] = 0.f;
        w->org_table.f_raw[i] = 0.f;
        w->org_table.f_sot[i] = 1.f;
    }

    // Section 15.5: Initialize replica tags (round-robin across 4 replicas).
    for (int i = 0; i < POOL_SIZE; ++i) {
        w->org_table.replica_tag[i] = static_cast<uint8_t>(i / PT_REPLICA_SIZE);
        w->mutation_ladder.replica_of[i] = w->org_table.replica_tag[i];
    }
    w->mutation_ladder.history_head = 0;
    w->mutation_ladder.beta = 1.0f;
    w->mutation_ladder.accept_ema = PT_TARGET_ACCEPT;
    w->mutation_ladder.swaps_attempted = 0;
    w->mutation_ladder.swaps_accepted = 0;
    std::memset(w->mutation_ladder.best_fitness_history, 0,
                sizeof(w->mutation_ladder.best_fitness_history));

    // Structural pressures (S-003, I4): neutral audit multipliers, no probe
    // baseline, empty sentinel ensemble/history and lineage table.
    std::memset(&w->audit_reg, 0, sizeof(w->audit_reg));
    w->audit_reg.audit_mult_classifier = 1.f;
    w->audit_reg.audit_mult_predictor = 1.f;
    std::memset(&w->probe_panel, 0, sizeof(w->probe_panel));
    w->probe_panel_l_role_baseline = 0.f;
    w->probe_panel_baseline_set = false;
    std::memset(&w->sentinel_ens, 0, sizeof(w->sentinel_ens));
    std::memset(&w->sentinel_history, 0, sizeof(w->sentinel_history));
    std::memset(w->lineage_stats, 0, sizeof(w->lineage_stats));
    w->n_lineage_stats = 0;
    std::memset(w->sentinel_anomaly, 0, sizeof(w->sentinel_anomaly));
    safety::pt::init_stress_ladder(&w->stress_ladder);
    for (int s = 0; s < STRESS_POOL_SIZE; ++s) w->h_stress_f_sot[s] = 1.f;
    for (int i = 0; i < POOL_SIZE; ++i) {
        w->predictor_error_ema[i] = 0.f;
        w->predictor_loss_ema[i] = PREDICTOR_LOSS_EMA_INIT;
    }

    // Section 9.1: Archive initialization.
    std::memset(&w->archive, 0, sizeof(w->archive));
    for (int b = 0; b < ARCHIVE_BINS_X * ARCHIVE_BINS_Y; ++b) {
        w->archive.bins[b].cap_classifier = ARCHIVE_BIN_CAP;
        w->archive.bins[b].cap_predictor  = ARCHIVE_BIN_CAP;
    }
    std::memset(w->archive.inv_var_ema, 0, sizeof(w->archive.inv_var_ema));
    for (int d = 0; d < BMAP_DIM; ++d) w->archive.inv_var_ema[d] = 1.0f;
    w->archive.pca_valid = false;
    w->archive.n_alive_classifier = 0;
    w->archive.n_alive_predictor = 0;

    // RFF projection initialized from PCG32.
    uint32_t rff_seed = pcg32_random(&w->rng);
    archive::init_rff(&w->archive.rff, rff_seed);

    // Section 15.6: CUSUM states (provisional params).
    w->cusum_surprise = {0.f, 0.f, 0.f, 0.5f, 5.0f, 0};
    w->cusum_r        = {0.f, 0.f, 0.f, 0.1f, 3.0f, 0};

    // Reference regressor, replay buffer, probe set, correlation window (A-601).
    predictor::init_reference_regressor(&w->reference_reg, &w->rng);
    TRANSFER_ABORT(cudaMemcpy(w->d_reference_reg, &w->reference_reg,
                    sizeof(predictor::ReferenceRegressor),
                    cudaMemcpyHostToDevice), "upload reference reg");
    std::memset(&w->replay_buffer, 0, sizeof(w->replay_buffer));
    w->replay_buffer.head = 0;
    w->replay_buffer.filled = 0;
    std::memset(&w->corr_window, 0, sizeof(w->corr_window));
    w->corr_window.head = 0;
    w->corr_window.filled = 0;
    cur::init_probe_set(&w->probe_set, w->host_sot_key, &w->rng);
    for (int i = 0; i < PROBE_BATCH; ++i) w->probe_fitness[i] = 0.f;

    // Section 15.7: Assemble first classifier batch with MAIN_SOT_DENSITY.
    cur::assemble_classifier_batch(&w->classifier_batch,
                                   MAIN_SOT_DENSITY,
                                   w->host_sot_key, &w->rng);

    // Zero evaluation correlation buffers before the first generation.
    std::memset(w->h_seed_grad, 0, POOL_SIZE * BMAP_DIM * sizeof(float));
    std::memset(w->h_tel, 0, sizeof(TelemetryScalars));

    std::printf("World initialized: %d pool organisms, %d total, %d weights\n",
                POOL_SIZE, TOTAL_ORG, TOTAL_WEIGHTS);
    std::fflush(stdout);
    return true;
}

// ---- Scoring (section 6) ---------------------------------------------------

// ---- Scoring (section 6, A-601) --------------------------------------------
// Classifiers: cross-entropy on the first NUM_CLASSES bmap_64 dims.
// Predictors: MSE of bmap_64 against their assigned predictor-batch target
// (ground truth), with the prediction-error EMA tracked per target organism
// for the predictor curriculum. The ENTIRE seed-gradient buffer is zeroed
// first: rows belonging to inactive roles must carry no gradient.
static void score_organisms(World* w, float classifier_multiplier,
                            float predictor_multiplier) {
    std::memset(w->h_seed_grad, 0, POOL_SIZE * BMAP_DIM * sizeof(float));

    float sum_loss = 0.f;
    float max_abs_logit = 0.f;
    int n_evaluated = 0;

    for (int org = 0; org < POOL_SIZE; ++org) {
        Role role = canonical_role(w->org_table.role[org]);
        const float* bmap = &w->h_descriptors[org * BMAP_DIM];
        float* sg = &w->h_seed_grad[org * BMAP_DIM];
        float loss = 0.f;
        float task_proxy = 0.f;
        float role_mult = 1.f;
        float audit_mult = 1.f;

        if (role == Role::Classifier) {
            int sample_idx = w->org_table.batch_sample_idx[org];
            int target = w->classifier_batch.label[sample_idx];
            float dlogits[NUM_CLASSES];
            autodiff::classifier_loss(bmap, target, NUM_CLASSES, dlogits, &loss);
            for (int d = 0; d < NUM_CLASSES; ++d) sg[d] = dlogits[d];
            for (int d = 0; d < NUM_CLASSES; ++d) {
                float a = fabsf(bmap[d]);
                if (a > max_abs_logit) max_abs_logit = a;
            }
            task_proxy = expf(-loss);
            role_mult = classifier_multiplier;
            audit_mult = w->audit_reg.audit_mult_classifier;
        } else {
            int slot = cur::predictor_target_slot(org, w->generation);
            const float* target = &w->predictor_batch.target_bmap_64[slot * BMAP_DIM];
            float dpred[BMAP_DIM];
            autodiff::predictor_mse_loss(bmap, target, dpred, &loss);
            for (int d = 0; d < BMAP_DIM; ++d) sg[d] = dpred[d];
            // Fitness aggregates the rotated K-target losses; the seed
            // gradient above remains the current target's training signal.
            w->predictor_loss_ema[org] =
                (1.f - PREDICTOR_LOSS_EMA_ALPHA) * w->predictor_loss_ema[org]
                + PREDICTOR_LOSS_EMA_ALPHA * loss;
            task_proxy = expf(-w->predictor_loss_ema[org]);
            role_mult = predictor_multiplier;
            audit_mult = w->audit_reg.audit_mult_predictor;

            // Ensemble prediction error EMA per pool target organism: the
            // predictor curriculum re-weights toward weak spots. Probe slots
            // carry pool_slot == -1 and have no pool EMA.
            int target_slot = w->predictor_batch.target_pool_slot[slot];
            if (target_slot >= 0 && target_slot < POOL_SIZE) {
                w->predictor_error_ema[target_slot] =
                    (1.f - PREDICTOR_ERROR_EMA_ALPHA) * w->predictor_error_ema[target_slot] +
                    PREDICTOR_ERROR_EMA_ALPHA * loss;
            }
        }

        sum_loss += loss;
        n_evaluated++;
        w->org_table.last_loss[org] = loss;

        float variance_mult = safety::variance_multiplier(bmap);
        w->org_table.f_raw[org] =
            task_proxy * archive::sot_gate(w->org_table.f_sot[org]);
        w->org_table.fitness[org] = archive::compose_fitness(
            w->org_table.f_raw[org], role_mult, audit_mult, variance_mult);
    }

    w->last_mean_ce = (n_evaluated > 0) ? sum_loss / static_cast<float>(n_evaluated) : 0.f;
    w->last_max_abs_logit = max_abs_logit;
}

// ---- Archive insertion (section 5, 9.1) ------------------------------------

static void insert_into_archive(World* w) {
    for (int org = 0; org < POOL_SIZE; ++org) {
        archive::ArchiveEntry cand;
        std::memcpy(cand.descriptor, &w->h_descriptors[org * BMAP_DIM],
                    BMAP_DIM * sizeof(float));
        archive::rff_project(w->archive.rff, cand.descriptor, cand.rff_proj);
        cand.fitness = w->org_table.fitness[org];
        cand.f_raw = w->org_table.f_raw[org];
        cand.f_sot = w->org_table.f_sot[org];
        cand.lineage_id = LineageId(w->org_table.lineage_id[org]);
        cand.parent_id = w->org_table.parent_id[org];
        cand.generation = w->generation;
        cand.role = w->org_table.role[org];
        cand.alive = true;
        cand.genome = w->org_table.genomes[org];

        // Assign bin via PCA projection or hash fallback (section 9.1).
        archive::assign_bin(w->archive, cand.descriptor, cand.bin_x, cand.bin_y);

        archive::insert(&w->archive, cand);
    }
}

// ---- Stress ladder (S-003, I4) ---------------------------------------------
// Deterministic target-dimension permutation drawn from the SOT key.
static void build_target_permutation(uint64_t key, uint8_t perm[BMAP_DIM],
                                     uint8_t inv[BMAP_DIM]) {
    for (int d = 0; d < BMAP_DIM; ++d) perm[d] = static_cast<uint8_t>(d);
    Pcg32 rng;
    pcg32_seed(&rng, key, 0x7A2Bu);
    for (int d = BMAP_DIM - 1; d > 0; --d) {
        int j = static_cast<int>(pcg32_random(&rng)
                                 % static_cast<uint32_t>(d + 1));
        uint8_t t = perm[d]; perm[d] = perm[j]; perm[j] = t;
    }
    for (int d = 0; d < BMAP_DIM; ++d) {
        inv[perm[d]] = static_cast<uint8_t>(d);
    }
}

// Refresh, evaluate, and flag: one pass of the SOT-density stress ladder.
static bool stress_cycle(World* w, int gen) {
    safety::pt::refresh_stress_slots(&w->stress_ladder,
                                     w->org_table.lineage_id,
                                     w->org_table.role, POOL_SIZE, gen,
                                     &w->rng);
    for (int s = 0; s < STRESS_POOL_SIZE; ++s) {
        if (w->stress_ladder.last_refresh_gen[s] != gen) continue;
        uint32_t src = w->stress_ladder.source_pool_idx[s];
        w->org_table.genomes[POOL_SIZE + s] = w->org_table.genomes[src];
        w->org_table.deltas[POOL_SIZE + s] = w->org_table.deltas[src];
        w->org_table.role[POOL_SIZE + s] = w->org_table.role[src];
        w->org_table.lineage_id[POOL_SIZE + s] =
            w->org_table.lineage_id[src];
        w->org_table.fitness[POOL_SIZE + s] = 0.f;
        w->org_table.f_raw[POOL_SIZE + s] = 0.f;
        w->org_table.f_sot[POOL_SIZE + s] = 1.f;
    }
    autodiff::launch_materialize_effective_weights(
        w->d_weights, w->d_deltas + POOL_SIZE, w->d_stress_eff_weights,
        STRESS_POOL_SIZE, w->stream);

    // The refreshed stress slots have new genomes: re-decode their RD
    // coefficients before evaluating them (A-202, I6).
    if (!upload_rd_coefficients(w)) return false;

    for (int s = 0; s < STRESS_POOL_SIZE; ++s) w->h_stress_f_sot[s] = 1.f;

    // Assemble the three elevated batches once and stage every image and task
    // embedding in combined host arrays, so one reference launch and one
    // stress launch carry all 24 slots (I8): the per-sub-population replays
    // were six latency-bound sequences of four-block forwards.
    static __half h_batch_images[STRESS_SUBPOP_COUNT * cur::CLASSIFIER_BATCH
                                 * IMG_HALVES];
    static __half h_unmarked[STRESS_SUBPOP_COUNT * cur::SOT_SUBBATCH
                             * IMG_HALVES];
    static float h_task_embs[STRESS_SUBPOP_COUNT * TASK_EMBED_DIM];
    for (int p = 0; p < STRESS_SUBPOP_COUNT; ++p) {
        cur::assemble_classifier_batch(&w->stress_batch,
                                       STRESS_SOT_DENSITIES[p],
                                       w->host_sot_key, &w->rng);
        std::memcpy(&h_batch_images[p * cur::CLASSIFIER_BATCH * IMG_HALVES],
                    w->stress_batch.image,
                    cur::CLASSIFIER_BATCH * IMG_HALVES * sizeof(__half));
        std::memcpy(&h_task_embs[p * TASK_EMBED_DIM],
                    w->stress_batch.task_embedding,
                    TASK_EMBED_DIM * sizeof(float));
        __half scratch[IMG_HALVES];
        int n_sot = 0;
        for (int s = 0; s < cur::CLASSIFIER_BATCH; ++s) {
            if (!w->stress_batch.is_sot[s]) continue;
            if (n_sot >= cur::SOT_SUBBATCH) break;
            __half* dst = &h_unmarked[(p * cur::SOT_SUBBATCH + n_sot)
                                      * IMG_HALVES];
            std::memcpy(dst, &w->stress_batch.image[s * IMG_HALVES],
                        IMG_HALVES * sizeof(__half));
            cur::apply_sot_permutation(dst, w->host_sot_key, true, scratch);
            n_sot++;
        }
    }
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_stress_batch_image, h_batch_images,
                    STRESS_SUBPOP_COUNT * cur::CLASSIFIER_BATCH * IMG_HALVES
                        * sizeof(__half),
                    cudaMemcpyHostToDevice, w->stream),
                   "upload stress batch images");
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_stress_ref_images, h_unmarked,
                    STRESS_SUBPOP_COUNT * cur::SOT_SUBBATCH * IMG_HALVES
                        * sizeof(__half),
                    cudaMemcpyHostToDevice, w->stream),
                   "upload stress reference images");
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_stress_task_embs, h_task_embs,
                    STRESS_SUBPOP_COUNT * TASK_EMBED_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream),
                   "upload stress task embeddings");

    // Predictor half: nominal and pi-permuted target rows.
    uint8_t perm[BMAP_DIM], inv[BMAP_DIM];
    build_target_permutation(w->host_sot_key, perm, inv);
    static float h_nominal[STRESS_POOL_SIZE * BMAP_DIM];
    static float h_permuted[STRESS_POOL_SIZE * BMAP_DIM];
    for (int s = 0; s < STRESS_POOL_SIZE; ++s) {
        const float* row = &w->predictor_batch.target_bmap_32[
            (s % cur::PREDICTOR_BATCH) * BMAP_DIM];
        for (int d = 0; d < BMAP_DIM; ++d) {
            h_nominal[s * BMAP_DIM + d] = row[d];
            h_permuted[s * BMAP_DIM + perm[d]] = row[d];
        }
    }
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_stress_targets, h_nominal,
                    STRESS_POOL_SIZE * BMAP_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream),
                   "upload stress nominal targets");
    TRANSFER_ABORT(cudaMemcpyAsync(
                    w->d_stress_targets + STRESS_POOL_SIZE * BMAP_DIM,
                    h_permuted, STRESS_POOL_SIZE * BMAP_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream),
                   "upload stress permuted targets");
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_sot_task_emb,
                    w->classifier_batch.task_embedding,
                    TASK_EMBED_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream),
                   "upload stress predictor task embedding");

    if (!safety::alignment::evaluate_stress_all(
            w->d_organisms, w->d_weights, w->d_stress_eff_weights,
            w->d_rd_coeffs + POOL_SIZE,
            w->d_stress_batch_image, w->d_stress_ref_images,
            w->d_stress_task_embs,
            w->d_stress_targets,
            w->d_stress_targets + STRESS_POOL_SIZE * BMAP_DIM,
            inv, w->d_sot_task_emb, w->h_stress_f_sot,
            w->d_sot_fwd_inputs, w->d_sot_descriptors, w->d_sot_bank_of,
            w->d_stress_ref_organisms, &w->fg_stress,
            TOTAL_WEIGHTS, w->stream)) {
        return false;
    }

    safety::pt::update_stress_failures(&w->stress_ladder, w->h_stress_f_sot,
                                       gen);
    return true;
}

// ---- Spawn wave (section 9, A-401 role-proportional) -----------------------

// Spawn `n_spawns` children of `target_role`: unique worst-fitness victims of
// that role, parents from the role's archive live list, mutation by the
// victim slot's replica rate. `force_role` locks the child to target_role
// (pre-bootstrap); otherwise a rare role-bit mutation may migrate the child.
static void spawn_role_wave(World* w, Role target_role, int n_spawns,
                            bool force_role, bool allow_role_conversion = false) {
    if (n_spawns <= 0) return;
    if (target_role == Role::Predictor && w->archive.n_alive_predictor <= 0) return;

    int victims[WAVE_SIZE];
    int n_victims = genome::select_spawn_victims(
        w->org_table.role, w->org_table.fitness, POOL_SIZE,
        target_role, n_spawns, victims);

    // Anti-extinction (A-401 minimum-2 per role): when the pool holds no
    // victim of the target role but the archive can parent that role,
    // convert the worst classifier slots instead of letting the role drain.
    if (n_victims == 0 && allow_role_conversion && target_role == Role::Predictor) {
        n_victims = genome::select_spawn_victims(
            w->org_table.role, w->org_table.fitness, POOL_SIZE,
            Role::Classifier, n_spawns, victims);
        force_role = true;
    }
    if (n_victims <= 0) return;

    for (int spawn = 0; spawn < n_victims; ++spawn) {
        int slot = victims[spawn];

        int parent_archive_idx = -1;
        if (target_role == Role::Classifier && w->archive.n_alive_classifier > 0) {
            int list_idx = static_cast<int>(pcg32_random(&w->rng) % w->archive.n_alive_classifier);
            parent_archive_idx = w->archive.alive_classifier_idx[list_idx];
        } else if (target_role == Role::Predictor && w->archive.n_alive_predictor > 0) {
            int list_idx = static_cast<int>(pcg32_random(&w->rng) % w->archive.n_alive_predictor);
            parent_archive_idx = w->archive.alive_predictor_idx[list_idx];
        }
        if (parent_archive_idx < 0) continue;

        genome::Genome child = w->archive.entries[parent_archive_idx].genome;
        uint8_t replica = w->org_table.replica_tag[slot];
        float mut_rate = PT_MUTATION_RATES[replica];
        genome::mutate(&child, mut_rate, MUTATION_RATE_ROLE, &w->rng);
        if (force_role) genome::force_role(child, target_role);

        w->org_table.genomes[slot] = child;
        w->org_table.role[slot] = genome::runtime_role(child);
        w->org_table.lineage_id[slot] = w->archive.entries[parent_archive_idx].lineage_id;
        w->org_table.parent_id[slot] = ArchiveSlot(parent_archive_idx);
        w->org_table.spawn_gen[slot] = w->generation;
        w->org_table.fitness[slot] = 0.f;
        w->org_table.f_raw[slot] = 0.f;
        w->org_table.f_sot[slot] = 1.f;
        w->predictor_loss_ema[slot] = PREDICTOR_LOSS_EMA_INIT;
        genome::init_delta_from_prior(child, &w->org_table.deltas[slot]);
    }
}

static void spawn_wave(World* w) {
    int asize = archive::archive_size(w->archive);
    if (asize == 0) return;

    if (!w->bootstrap_fired) {
        // Pre-bootstrap: every spawn is a classifier, role-locked.
        spawn_role_wave(w, Role::Classifier, WAVE_SIZE, /*force_role=*/true);
        return;
    }

    // Role-proportional wave: the predictor share tracks the archive role
    // fraction with a minimum of 2 per role to prevent extinction.
    int n_pred = 0;
    if (w->archive.n_alive_predictor > 0) {
        n_pred = static_cast<int>(lroundf(
            static_cast<float>(WAVE_SIZE) *
            static_cast<float>(w->archive.n_alive_predictor) /
            static_cast<float>(asize)));
        if (n_pred < 2) n_pred = 2;
        if (n_pred > WAVE_SIZE - 2) n_pred = WAVE_SIZE - 2;
    }
    spawn_role_wave(w, Role::Predictor, n_pred, /*force_role=*/false,
                    /*allow_role_conversion=*/true);
    spawn_role_wave(w, Role::Classifier, WAVE_SIZE - n_pred, /*force_role=*/false);
}

// ---- Predictor bootstrap (A-601) -------------------------------------------
// One-shot at the archive half-occupancy crossing: sign the stationary
// predictor probe references from current pool trajectories, then inject
// PREDICTOR_FOUNDERS role-flipped copies of the highest-novelty classifier
// genomes into the worst-fitness classifier pool slots.
static bool inject_predictor_founders(World* w) {
    // Sign predictor probes from a deterministic spread of pool organisms.
    LineageId target_ids[cur::PREDICTOR_BATCH];
    float b32[cur::PREDICTOR_BATCH * BMAP_DIM];
    float b64[cur::PREDICTOR_BATCH * BMAP_DIM];
    for (int i = 0; i < cur::PREDICTOR_BATCH; ++i) {
        int org = (i * 7 + 3) % POOL_SIZE;
        target_ids[i] = w->org_table.lineage_id[org];
        // BTRAJ samples: index 1 is bmap_32 (step 32), index 3 is bmap_64.
        std::memcpy(&b32[i * BMAP_DIM],
                    &w->intent_registry.btraj[org][1 * BMAP_DIM],
                    BMAP_DIM * sizeof(float));
        std::memcpy(&b64[i * BMAP_DIM],
                    &w->intent_registry.btraj[org][3 * BMAP_DIM],
                    BMAP_DIM * sizeof(float));
    }
    cur::sign_predictor_probes(&w->probe_set, target_ids, b32, b64,
                               w->host_sot_key);

    // Sign the held-out reference probe tuples from the replay buffer
    // (real evaluated tuples; the reference never trains on them — the
    // snapshotted ring positions are marked held out below).
    if (!w->probe_set.probe_tuples_signed &&
        w->replay_buffer.filled >= PROBE_BATCH) {
        cur::sign_probe_tuples(&w->probe_set,
                               w->replay_buffer.bmap,
                               w->replay_buffer.task_emb,
                               w->replay_buffer.fitness,
                               w->host_sot_key);
        for (int i = 0; i < PROBE_BATCH; ++i) {
            w->replay_buffer.held_out[i] = true;
        }
        if (!upload_probe_batch(w)) return false;
    }

    // Highest-novelty classifier parents.
    static float novelty[MAX_ARCHIVE];
    static int idx_map[MAX_ARCHIVE];
    int n = 0;
    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        const archive::ArchiveEntry& e = w->archive.entries[i];
        if (!e.alive || e.role != Role::Classifier) continue;
        novelty[n] = archive::rff_novelty(e.rff_proj, w->archive.mu_rff_classifier);
        idx_map[n] = i;
        n++;
    }
    if (n == 0) return true;

    int founders[PREDICTOR_FOUNDERS];
    predictor::select_predictor_founders(novelty, n, founders);

    // Victims: the worst-fitness classifier slots, without replacement.
    int victims[WAVE_SIZE];
    int n_victims = genome::select_spawn_victims(
        w->org_table.role, w->org_table.fitness, POOL_SIZE,
        Role::Classifier, PREDICTOR_FOUNDERS, victims);

    int injected = 0;
    for (int k = 0; k < PREDICTOR_FOUNDERS && k < n_victims; ++k) {
        int slot = victims[k];
        int parent_archive = idx_map[founders[k]];
        genome::Genome child = w->archive.entries[parent_archive].genome;
        genome::write_role(child, Role::Predictor);

        w->org_table.genomes[slot] = child;
        w->org_table.role[slot] = Role::Predictor;
        w->org_table.lineage_id[slot] = w->archive.entries[parent_archive].lineage_id;
        w->org_table.parent_id[slot] = ArchiveSlot(parent_archive);
        w->org_table.spawn_gen[slot] = w->generation;
        w->org_table.fitness[slot] = 0.f;
        w->org_table.f_raw[slot] = 0.f;
        w->org_table.f_sot[slot] = 1.f;
        w->predictor_loss_ema[slot] = PREDICTOR_LOSS_EMA_INIT;
        genome::init_delta_from_prior(child, &w->org_table.deltas[slot]);
        injected++;
    }
    std::printf("[BOOTSTRAP] gen %d: %d predictor founders injected; "
                "probe references signed\n", w->generation, injected);
    std::fflush(stdout);
    return true;
}

// ---- Build SwapContext from World ------------------------------------------

static safety::pt::SwapContext make_swap_context(World* w) {
    safety::pt::SwapContext ctx;
    ctx.d_organisms  = w->d_organisms;
    ctx.d_checkpoints = w->d_checkpoints;
    ctx.d_grads      = w->d_grads;
    ctx.d_eff_weights = w->d_eff_weights;
    ctx.d_swap_org   = w->d_pt_swap_org;
    ctx.d_swap_ckpt  = w->d_pt_swap_ckpt;
    ctx.d_swap_grad  = w->d_pt_swap_grad;
    ctx.d_swap_wbank = w->d_pt_swap_wbank;
    ctx.genomes      = w->org_table.genomes;
    ctx.deltas       = w->org_table.deltas;
    ctx.lineage_id   = w->org_table.lineage_id;
    ctx.parent_id    = w->org_table.parent_id;
    ctx.spawn_gen    = w->org_table.spawn_gen;
    ctx.fitness      = w->org_table.fitness;
    ctx.f_raw        = w->org_table.f_raw;
    ctx.f_sot        = w->org_table.f_sot;
    ctx.role         = w->org_table.role;
    ctx.seed_grad    = w->h_seed_grad;
    ctx.batch_sample_idx = w->org_table.batch_sample_idx;
    ctx.predictor_error_ema = w->predictor_error_ema;
    ctx.predictor_loss_ema  = w->predictor_loss_ema;
    ctx.stream       = w->stream;
    return ctx;
}

// ---- Step generation (section 5 data flow) ---------------------------------

// Per-phase timing (I8): each phase ends with a stream sync, so the wall time
// since the previous trace is that phase's execution time. The table is
// printed at run end when --profile is set; the flag also silences the
// per-phase lines so long profiling logs stay compact.
struct PhaseTiming {
    const char* name;
    double total_ms;
    int count;
};
static constexpr int PHASE_TIMING_SLOTS = ::PHASE_TIMING_SLOTS;
static PhaseTiming g_phase_timing[PHASE_TIMING_SLOTS];
static int g_n_phase_timings = 0;
static std::chrono::steady_clock::time_point g_phase_mark;
static bool g_phase_mark_set = false;
static bool g_profile = false;

static void phase_timing_add(const char* tag, double ms) {
    for (int i = 0; i < g_n_phase_timings; ++i) {
        if (std::strcmp(g_phase_timing[i].name, tag) == 0) {
            g_phase_timing[i].total_ms += ms;
            g_phase_timing[i].count++;
            return;
        }
    }
    if (g_n_phase_timings >= PHASE_TIMING_SLOTS) return;
    g_phase_timing[g_n_phase_timings].name = tag;
    g_phase_timing[g_n_phase_timings].total_ms = ms;
    g_phase_timing[g_n_phase_timings].count = 1;
    g_n_phase_timings++;
}

static void print_phase_timings() {
    std::printf("\n[PROFILE] per-phase totals:\n");
    // Insertion sort by total descending (few entries).
    for (int i = 1; i < g_n_phase_timings; ++i) {
        PhaseTiming key = g_phase_timing[i];
        int j = i - 1;
        while (j >= 0 && g_phase_timing[j].total_ms < key.total_ms) {
            g_phase_timing[j + 1] = g_phase_timing[j];
            j--;
        }
        g_phase_timing[j + 1] = key;
    }
    double sum = 0.0;
    for (int i = 0; i < g_n_phase_timings; ++i) sum += g_phase_timing[i].total_ms;
    for (int i = 0; i < g_n_phase_timings; ++i) {
        const PhaseTiming& t = g_phase_timing[i];
        std::printf("[PROFILE] %-28s %10.2f ms  n=%d  %5.1f%%\n",
                    t.name, t.total_ms, t.count,
                    sum > 0.0 ? 100.0 * t.total_ms / sum : 0.0);
    }
    std::printf("[PROFILE] %-28s %10.2f ms\n", "TOTAL", sum);
    std::fflush(stdout);
}

// Phase progress trace: prints phase tag + checks CUDA errors after each sync.
// Always flushed so output is never lost to buffering. Returns false on any
// CUDA error — a failed phase invalidates the run (fail-fast, A-501).
static bool phase_trace(const char* tag, int gen, cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (!g_phase_mark_set) {
        g_phase_mark = std::chrono::steady_clock::now();
        g_phase_mark_set = true;
    }
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(
        now - g_phase_mark).count();
    g_phase_mark = now;
    phase_timing_add(tag, ms);
    if (err != cudaSuccess) {
        std::printf("[FATAL] gen %d %s: %s — run invalidated\n",
                    gen, tag, cudaGetErrorString(err));
        std::fflush(stdout);
        return false;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("[FATAL] gen %d %s (launch): %s — run invalidated\n",
                    gen, tag, cudaGetErrorString(err));
        std::fflush(stdout);
        return false;
    }
    if (!g_profile) {
        std::printf("  gen %d: %s\n", gen, tag);
        std::fflush(stdout);
    }
    return true;
}

// Generation transaction (A-501 ordering):
//   EVALUATE  forward → descriptor → SOT → score → archive
//   PT        swap full organism state INCLUDING seed gradients + batch id
//   T3        upload the reordered seed-gradient buffer
//   LEARN     backward → aggregate → CAME
//   TELEMETRY norms + nonfinite hard abort
//   RECORD    replay push + evaluated-population logging (pre-spawn)
//   EVOLVE    spawn wave
//   MONITOR   reference train, surprise, CUSUM, PCA rebin, operator checks
//
// Returns false when the generation invalidated the run (CUDA error or a
// nonfinite numerical state); run() then aborts the experiment.
bool step_generation(World* w) {
    int gen = w->generation;
    const bool log_this_gen = (gen % TELEMETRY_INTERVAL == 0 || gen < FIRST_GENS_TELEMETRY);
    std::printf("step_generation(%d) begin\n", gen);
    std::fflush(stdout);

    // Curriculum refresh every CURRICULUM_INTERVAL generations (section 15.7).
    if (gen % CURRICULUM_INTERVAL == 0) {
        cur::assemble_classifier_batch(&w->classifier_batch,
                                       MAIN_SOT_DENSITY,
                                       w->host_sot_key, &w->rng);
    }

    // Reaction-diffusion coefficients (A-202, I6): decoded from the current
    // genomes before the forward.
    if (!upload_rd_coefficients(w)) return false;

    // Predictor batch (A-701): assembled from the previous generation's
    // Intent Registry. Stationary probe slots are used once signed.
    if (w->bootstrap_fired) {
        static float bmap32_rows[POOL_SIZE * BMAP_DIM];
        static float bmap64_rows[POOL_SIZE * BMAP_DIM];
        static LineageId pool_ids[POOL_SIZE];
        static bool pool_was_sot[POOL_SIZE];
        for (int i = 0; i < POOL_SIZE; ++i) {
            std::memcpy(&bmap32_rows[i * BMAP_DIM],
                        &w->intent_registry.btraj[i][1 * BMAP_DIM],
                        BMAP_DIM * sizeof(float));
            std::memcpy(&bmap64_rows[i * BMAP_DIM],
                        &w->intent_registry.btraj[i][3 * BMAP_DIM],
                        BMAP_DIM * sizeof(float));
            pool_ids[i] = w->org_table.lineage_id[i];
            int sample = w->org_table.batch_sample_idx[i];
            pool_was_sot[i] = w->classifier_batch.is_sot[sample];
        }
        cur::assemble_predictor_batch(&w->predictor_batch, w->probe_set,
                                      pool_ids, w->predictor_error_ema,
                                      bmap32_rows, bmap64_rows,
                                      w->classifier_batch.task_embedding,
                                      w->org_table.role, pool_was_sot,
                                      &w->rng);
    }

    // Organism-to-batch assignment: deterministic round-robin (A-401).
    for (int i = 0; i < POOL_SIZE; ++i) {
        w->org_table.batch_sample_idx[i] = i % cur::CLASSIFIER_BATCH;
    }

    // Set up h_fwd_inputs for each organism.
    for (int i = 0; i < POOL_SIZE; ++i) {
        int s = w->org_table.batch_sample_idx[i];
        ForwardInputs& fi = w->h_fwd_inputs[i];
        Role role = canonical_role(w->org_table.role[i]);
        fi.role = role;
        fi.task_embedding = w->d_batch_task_emb;
        if (role == Role::Classifier) {
            fi.image_rgb = w->d_batch_image + s * GRID_SIZE * GRID_SIZE * 3;
            fi.target_bmap_32 = nullptr;
        } else {
            fi.image_rgb = nullptr;
            fi.target_bmap_32 = w->d_predictor_bmap32 +
                                (i % cur::PREDICTOR_BATCH) * BMAP_DIM;
        }
    }

    // ---- T1: H→D transfers ----
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_fwd_inputs, w->h_fwd_inputs,
                    POOL_SIZE * sizeof(ForwardInputs),
                    cudaMemcpyHostToDevice, w->stream), "T1 fwd_inputs");
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_batch_image, w->classifier_batch.image,
                    cur::CLASSIFIER_BATCH * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half),
                    cudaMemcpyHostToDevice, w->stream), "T1 batch_image");
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_batch_task_emb, w->classifier_batch.task_embedding,
                    TASK_EMBED_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream), "T1 task_emb");
    // Upload the current genome deltas (updated by the previous spawn wave).
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_deltas, w->org_table.deltas,
                    POOL_SIZE * sizeof(genome::DeltaWeights),
                    cudaMemcpyHostToDevice, w->stream), "T1 deltas");
    // Upload the predictor target bmap_32 rows (used by predictor forwards).
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_predictor_bmap32,
                    w->predictor_batch.target_bmap_32,
                    cur::PREDICTOR_BATCH * BMAP_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream), "T1 predictor targets");
    if (!phase_trace("T1_H2D", gen, w->stream)) return false;

    // ---- GPU: materialize W_eff = W_shared + delta per organism ----
    autodiff::launch_materialize_effective_weights(
        w->d_weights, w->d_deltas, w->d_eff_weights, POOL_SIZE, w->stream);

    // ---- GPU: forward + descriptor + btraj (phase graph, A-102) ----
    // The three launches are captured together: no host synchronization may
    // occur between them, so the phase trace runs once after the graph.
    if (!phase_run(&w->fg_forward, w->stream, [&] {
            autodiff::launch_forward_with_checkpoints(
                w->d_organisms, w->d_fwd_inputs, w->d_rd_coeffs,
                w->d_weights, w->d_eff_weights, w->d_checkpoints,
                RESIDUAL_ALPHA, POOL_SIZE, w->stream);
            nca::extract_descriptor(w->d_organisms, w->d_descriptors,
                                    POOL_SIZE, w->stream);
            autodiff::launch_btraj_gather(w->d_organisms, w->d_btraj,
                                          POOL_SIZE, w->stream);
        })) {
        return false;
    }
    if (!phase_trace("forward+descriptor+btraj", gen, w->stream)) return false;

    // ---- T2: D→H transfers ----
    TRANSFER_ABORT(cudaMemcpyAsync(w->h_descriptors, w->d_descriptors,
                    POOL_SIZE * BMAP_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, w->stream), "T2 descriptors");
    TRANSFER_ABORT(cudaMemcpyAsync(w->h_btraj, w->d_btraj,
                    POOL_SIZE * BTRAJ_SAMPLES * BMAP_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, w->stream), "T2 btraj");
    if (!phase_trace("T2_D2H", gen, w->stream)) return false;

    // Copy BTRAJ into IntentRegistry.
    for (int i = 0; i < POOL_SIZE; ++i) {
        std::memcpy(w->intent_registry.btraj[i],
                    &w->h_btraj[i * BTRAJ_SAMPLES * BMAP_DIM],
                    BTRAJ_SAMPLES * BMAP_DIM * sizeof(float));
    }

    // ---- SOT identity check (section 12, pre-allocated buffers) ----
    // Each SOT-evaluated organism is compared against a reference computed
    // with ITS OWN effective weights (organism-specific phenotype).
    if (!safety::alignment::apply_sot_identity(
            w->d_organisms, w->d_weights, w->d_eff_weights,
            w->classifier_batch, w->host_sot_key,
            w->h_descriptors, w->org_table.batch_sample_idx,
            w->org_table.f_sot,
            w->d_sot_temp_images, w->d_sot_task_emb,
            w->d_sot_fwd_inputs, w->d_sot_descriptors,
            w->d_sot_bank_of, w->d_sot_ref_organisms,
            w->h_rd_coeffs, w->d_sot_ref_coeffs,
            TOTAL_WEIGHTS, w->stream)) {
        return false;
    }
    if (!phase_trace("SOT", gen, w->stream)) return false;

    // ---- Score organisms (section 6, A-601) ----
    // Role-balance multipliers from the surprise history: rho = s_avg /
    // s_target. The mechanism is INACTIVE until the calibration window
    // freezes s_target (spec: calibrated, not declared); the inactive state
    // is explicit and logged once, never a silent substitute ratio.
    float s_avg = 0.f;
    if (w->s_hist_filled > 0) {
        float sum = 0.f;
        for (int i = 0; i < w->s_hist_filled; ++i) sum += w->s_blended_history[i];
        s_avg = sum / static_cast<float>(w->s_hist_filled);
    }
    float classifier_mult = 1.f;
    float predictor_mult = 1.f;
    if (w->s_target_calibrated && w->s_target > 0.f) {
        float rho = archive::surprise_ratio(s_avg, w->s_target);
        classifier_mult = archive::classifier_mult(rho);
        predictor_mult = archive::predictor_mult(rho);
    } else {
        static bool logged_inactive = false;
        if (!logged_inactive) {
            std::printf("[ROLE BALANCE] inactive: s_target uncalibrated "
                        "(multipliers idle until the calibration window closes)\n");
            std::fflush(stdout);
            logged_inactive = true;
        }
    }
    score_organisms(w, classifier_mult, predictor_mult);

    // ---- Structural pressures (S-003, I4) ----
    // Per-role lineage shares and runaway brakes are recomputed before
    // insertion so the brake affects this generation's replacement decisions.
    safety::update_lineage_stats(w->org_table.lineage_id, w->org_table.role,
                                 POOL_SIZE, w->lineage_stats,
                                 &w->n_lineage_stats, gen);
    w->archive.n_lineage_brakes = 0;
    for (Role role : { Role::Classifier, Role::Predictor }) {
        int best = -1;
        for (int s = 0; s < w->n_lineage_stats; ++s) {
            if (w->lineage_stats[s].role != role) continue;
            if (!safety::runaway_detected(w->lineage_stats[s],
                                          LINEAGE_RUNAWAY_THRESHOLD)) {
                continue;
            }
            if (best < 0 || w->lineage_stats[s].archive_share >
                            w->lineage_stats[best].archive_share) {
                best = s;
            }
        }
        if (best >= 0) {
            archive::set_lineage_brake(&w->archive, role,
                                       w->lineage_stats[best].lineage_id,
                                       w->lineage_stats[best].archive_share,
                                       LINEAGE_RUNAWAY_THRESHOLD);
        }
    }

    // ---- Archive insertion (section 9.1) ----
    insert_into_archive(w);

    // Audit cycle: refit the role-aware predictive-sufficiency regressors.
    if (gen % AUDIT_INTERVAL == 0) {
        safety::run_audit_cycle(&w->audit_reg, w->h_descriptors,
                                w->org_table.last_loss, w->org_table.role,
                                POOL_SIZE);
    }

    // Interpretability probe panel and the L_role collapse alarm.
    if (gen % PROBE_PANEL_INTERVAL == 0) {
        safety::refresh_probe_panel(&w->probe_panel, w->h_descriptors,
                                    w->org_table.fitness,
                                    w->org_table.lineage_id,
                                    w->org_table.role, POOL_SIZE);
        if (!w->probe_panel_baseline_set && w->probe_panel.l_role_acc > 0.f) {
            w->probe_panel_l_role_baseline = w->probe_panel.l_role_acc;
            w->probe_panel_baseline_set = true;
        } else if (safety::l_role_collapse(w->probe_panel,
                       w->probe_panel_l_role_baseline)) {
            std::printf("[ALARM] L_role accuracy %.3f dropped below %.0f%% of "
                        "baseline %.3f: representational collapse between "
                        "roles\n", w->probe_panel.l_role_acc,
                        L_ACC_COLLAPSE_FRACTION * 100.f,
                        w->probe_panel_l_role_baseline);
            std::fflush(stdout);
        }
    }

    // Sentinels: score every organism, ingest sampled history examples, and
    // record this generation's descriptors as survived-so-far observations.
    safety::score_sentinels(w->sentinel_ens, w->h_descriptors, POOL_SIZE,
                            w->sentinel_anomaly);
    safety::train_sentinels_from_history(&w->sentinel_ens,
                                         &w->sentinel_history, &w->rng);
    for (int org = 0; org < POOL_SIZE; ++org) {
        safety::sentinel_history_push(&w->sentinel_history,
                                      &w->h_descriptors[org * BMAP_DIM],
                                      0.f, w->org_table.lineage_id[org], gen);
    }

    // ---- Predictor bootstrap (A-601): one-shot at half occupancy ----
    if (!w->bootstrap_fired && archive::bootstrap_trigger(w->archive)) {
        if (!inject_predictor_founders(w)) return false;
        w->bootstrap_fired = true;
        w->bootstrap_gen = gen;
    }

    // ---- PT swaps BEFORE backward (section 13, section 5) ----
    // Transactional: swaps OrganismState, CheckpointBuffer, GradBuffers, and
    // ALL host organism metadata INCLUDING seed-gradient rows and batch sample
    // assignment, so backward differentiates each trajectory with its own
    // organism's objective. T3 runs AFTER the swap so the device sees the
    // reordered seed gradients.
    safety::pt::record_best_fitness(&w->mutation_ladder,
                                    w->org_table.fitness);
    if (gen > 0 && gen % PT_SWAP_INTERVAL == 0) {
        safety::pt::SwapContext ctx = make_swap_context(w);
        if (!safety::pt::propose_swaps(&w->mutation_ladder,
                                       w->org_table.fitness,
                                       &w->rng, ctx)) {
            return false;
        }
    }

    // ---- Stress ladder (S-003, I4): refresh, evaluate, flag ----
    if (!phase_trace("score+archive+PT", gen, w->stream)) return false;
    if (!stress_cycle(w, gen)) return false;

    // ---- T3a: post-PT roles for gradient attribution (C3) ----
    // The alignment kernel groups gradients by the organism's post-swap role;
    // the per-generation ForwardInputs roles are stale after a PT swap.
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_roles, w->org_table.role,
                    POOL_SIZE * sizeof(Role), cudaMemcpyHostToDevice,
                    w->stream), "T3a roles");

    if (!phase_trace("stress", gen, w->stream)) return false;

    // ---- T3: H→D seed_grad (AFTER PT swaps) ----
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_seed_grad, w->h_seed_grad,
                    POOL_SIZE * BMAP_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream), "T3 seed_grad");

    // ---- GPU: backward (phase-decomposed batched, all orgs in parallel) ----
    // COEVO_BACKWARD_PROFILE runs it outside the phase graph so the internal
    // synchronizations used by the sub-phase profiler are legal.
    if (std::getenv("COEVO_BACKWARD_PROFILE") != nullptr) {
        autodiff::launch_backward_all(
            w->d_organisms, w->d_weights, w->d_eff_weights,
            w->d_rd_coeffs, w->d_seed_grad,
            w->d_checkpoints, w->d_grads,
            w->bwd_workspace, RESIDUAL_ALPHA, POOL_SIZE, w->stream);
    } else if (!phase_run(&w->fg_backward, w->stream, [&] {
            autodiff::launch_backward_all(
                w->d_organisms, w->d_weights, w->d_eff_weights,
                w->d_rd_coeffs, w->d_seed_grad,
                w->d_checkpoints, w->d_grads,
                w->bwd_workspace, RESIDUAL_ALPHA, POOL_SIZE, w->stream);
        })) {
        return false;
    }
    if (!phase_trace("backward", gen, w->stream)) return false;

    // ---- GPU: aggregate + CAME (phase graph, A-102) ----
    optimizer::CameState came_state;
    came_state.d_m = w->d_came_m;
    came_state.d_v = w->d_came_v;
    came_state.d_c = w->d_came_c;
    came_state.d_prev_u = w->d_came_prev_u;
    came_state.d_mean_grad = w->d_mean_grad;
    came_state.step = gen;
    if (!phase_run(&w->fg_optimizer, w->stream, [&] {
            optimizer::launch_aggregate_gradients(w->d_grads, w->d_mean_grad,
                                                  POOL_SIZE, w->stream);
            optimizer::launch_came_step(w->d_weights, came_state,
                                        optimizer::CAME_DEFAULTS, w->stream);
        })) {
        return false;
    }

    // ---- GPU: gradient norm via device-side reduction (section 8) ----
    if (!optimizer::launch_grad_norm_reduce(w->d_mean_grad, w->d_grad_norm,
                                            w->stream)) {
        return false;
    }

    // ---- GPU: numerical telemetry (A-501) ----
    // launch_telemetry_kernels zeroes the TelemetryScalars struct first, so
    // it must be enqueued BEFORE the state-saturation/residual kernels write
    // their fields; stream order preserves the accumulation.
    if (!optimizer::launch_telemetry_kernels(
            w->d_mean_grad, w->d_weights,
            w->d_came_m, w->d_came_v, w->d_came_c, w->d_came_prev_u,
            w->d_tel, w->stream)) {
        return false;
    }
    // Enqueued after the telemetry memset so the role sums survive; stream
    // order places this before the readback below.
    if (!optimizer::launch_role_grad_alignment(
            w->d_grads, w->d_roles, POOL_SIZE, w->d_tel, w->stream)) {
        return false;
    }
    autodiff::launch_state_saturation(w->d_checkpoints, w->d_organisms,
                                      w->d_tel, POOL_SIZE, w->stream);
    // The residual-magnitude measurement costs several forward passes worth
    // of work; run it only on generations that will be logged.
    if (log_this_gen) {
        autodiff::launch_residual_magnitude(w->d_checkpoints, w->d_organisms,
                                            w->d_weights, w->d_eff_weights,
                                            w->d_tel, POOL_SIZE, w->stream);
    }
    if (!phase_trace("optimizer", gen, w->stream)) return false;

    // ---- Telemetry readback (single small struct D→H) ----
    TRANSFER_ABORT(cudaMemcpy(w->h_tel, w->d_tel, sizeof(TelemetryScalars),
                              cudaMemcpyDeviceToHost), "telemetry struct");
    float h_grad_norm_sq = 0.f;
    TRANSFER_ABORT(cudaMemcpy(&h_grad_norm_sq, w->d_grad_norm, sizeof(float),
                              cudaMemcpyDeviceToHost), "grad norm scalar");
    float grad_norm = sqrtf(h_grad_norm_sq);

    // Nonfinite numerical state invalidates the run immediately.
    bool tel_nonfinite = !(w->h_tel->nonfinite_count == 0.f)
        || !std::isfinite(grad_norm) || !std::isfinite(w->h_tel->state_max_abs);
    if (log_this_gen) {
        for (int m = 0; m < 5 && !tel_nonfinite; ++m) {
            tel_nonfinite = !std::isfinite(w->h_tel->res_ratio_max[m])
                || !std::isfinite(w->h_tel->res_F_norm2[m]);
        }
    }
    if (tel_nonfinite) {
        std::printf("[FATAL] gen %d: nonfinite numerical state "
                    "(nonfinite_count=%.0f, grad_norm=%.2e, state_max_abs=%.2e) — run invalidated\n",
                    w->generation, w->h_tel->nonfinite_count,
                    grad_norm, w->h_tel->state_max_abs);
        std::fflush(stdout);
        return false;
    }

    // Gradient health monitoring (section 8): too-small gradients only; the
    // explosion direction is covered by the telemetry log and the hard
    // nonfinite abort above.
    float eps_thresh = EPS_GRAD * static_cast<float>(TOTAL_WEIGHTS);
    if (grad_norm < eps_thresh) {
        w->grad_health_warn_count++;
        if (w->grad_health_warn_count >= GRAD_HEALTH_WINDOW) {
            std::printf("[WARN] gen %d: gradient norm %.2e below threshold for %d consecutive gens\n",
                        w->generation, grad_norm, w->grad_health_warn_count);
            std::fflush(stdout);
        }
    } else {
        w->grad_health_warn_count = 0;
    }

    // ---- RECORD: replay + evaluated-population telemetry BEFORE spawning ----
    // Descriptors in h_descriptors belong to the population that was just
    // evaluated; spawn_wave replaces up to 16 of them afterwards, so replay
    // tuples and the generation log must be captured now.
    for (int org = 0; org < POOL_SIZE; ++org) {
        if (canonical_role(w->org_table.role[org]) != Role::Classifier) continue;
        predictor::replay_buffer_push(
            &w->replay_buffer,
            &w->h_descriptors[org * BMAP_DIM],
            w->classifier_batch.task_embedding,
            w->org_table.fitness[org]);
    }

    // ---- Logging (section 15.8): every TELEMETRY_INTERVAL AND first 5 gens ----
    if (log_this_gen) {
        float sum_fit = 0.f, sum_raw = 0.f;
        int count = 0;
        int n_pred = 0;
        for (int i = 0; i < POOL_SIZE; ++i) {
            if (canonical_role(w->org_table.role[i]) == Role::Classifier) {
                sum_fit += w->org_table.fitness[i];
                sum_raw += w->org_table.f_raw[i];
                count++;
            } else {
                n_pred++;
            }
        }
        float mean_fit = count > 0 ? sum_fit / count : 0.f;
        float mean_raw = count > 0 ? sum_raw / count : 0.f;
        int asize = archive::archive_size(w->archive);
        const TelemetryScalars& t = *w->h_tel;
        std::printf("gen %4d  mean_fitness=%.4f  mean_f_raw=%.4f  mean_CE=%.4f  "
                    "max|logit|=%.2e  archive=%d\n",
                    w->generation, mean_fit, mean_raw, w->last_mean_ce,
                    w->last_max_abs_logit, asize);
        std::printf("         grad_norm=%.2e |g|=[%.1e %.1e %.1e %.1e] |w|=[%.2f %.2f %.2f %.2f] "
                    "|u|=[%.2f %.2f %.2f %.2f]\n",
                    grad_norm,
                    sqrtf(t.grad_norm_sq[0]), sqrtf(t.grad_norm_sq[1]),
                    sqrtf(t.grad_norm_sq[2]), sqrtf(t.grad_norm_sq[3]),
                    sqrtf(t.weight_norm_sq[0]), sqrtf(t.weight_norm_sq[1]),
                    sqrtf(t.weight_norm_sq[2]), sqrtf(t.weight_norm_sq[3]),
                    sqrtf(t.update_norm_sq[0]), sqrtf(t.update_norm_sq[1]),
                    sqrtf(t.update_norm_sq[2]), sqrtf(t.update_norm_sq[3]));
        std::printf("         c_mean=%.3e c_max=%.3e conf_mean=%.4f conf_min=%.4f "
                    "state_max=%.2e state_near_max=%.0f\n",
                    t.c_mean, t.c_max, t.conf_mean, t.conf_min,
                    t.state_max_abs, t.state_near_max);
        std::printf("         residual |F| step0/16/32/48/64: [%.2e %.2e %.2e %.2e %.2e]  "
                    "|x|: [%.2e %.2e %.2e %.2e %.2e]\n",
                    sqrtf(t.res_F_norm2[0]), sqrtf(t.res_F_norm2[1]),
                    sqrtf(t.res_F_norm2[2]), sqrtf(t.res_F_norm2[3]),
                    sqrtf(t.res_F_norm2[4]),
                    sqrtf(t.res_x_norm2[0]), sqrtf(t.res_x_norm2[1]),
                    sqrtf(t.res_x_norm2[2]), sqrtf(t.res_x_norm2[3]),
                    sqrtf(t.res_x_norm2[4]));
        std::printf("         residual ratio mean: [%.3f %.3f %.3f %.3f %.3f]  "
                    "max: [%.3f %.3f %.3f %.3f %.3f]\n",
                    t.res_ratio_mean[0], t.res_ratio_mean[1], t.res_ratio_mean[2],
                    t.res_ratio_mean[3], t.res_ratio_mean[4],
                    t.res_ratio_max[0], t.res_ratio_max[1], t.res_ratio_max[2],
                    t.res_ratio_max[3], t.res_ratio_max[4]);
        bool role_cos_valid = (count > 0 && n_pred > 0
                               && t.role_grad_norm_sq[0] > 0.f
                               && t.role_grad_norm_sq[1] > 0.f);
        if (role_cos_valid) {
            float role_cos = t.role_grad_dot
                / (sqrtf(t.role_grad_norm_sq[0])
                   * sqrtf(t.role_grad_norm_sq[1]));
            std::printf("         role grad cos=%.4f  |g_C|/n_C=%.3e  "
                        "|g_P|/n_P=%.3e  n_C=%d n_P=%d\n",
                        role_cos,
                        sqrtf(t.role_grad_norm_sq[0]) / static_cast<float>(count),
                        sqrtf(t.role_grad_norm_sq[1]) / static_cast<float>(n_pred),
                        count, n_pred);
        } else {
            std::printf("         role grad cos=n/a  n_C=%d n_P=%d\n",
                        count, n_pred);
        }
        float sentinel_mean = 0.f;
        for (int i = 0; i < POOL_SIZE; ++i) sentinel_mean += w->sentinel_anomaly[i];
        sentinel_mean /= static_cast<float>(POOL_SIZE);
        std::printf("         audit r2_C=%.3f mult_C=%.3f r2_P=%.3f mult_P=%.3f  "
                    "L_role=%.3f L_fit=%.3f L_lineage=%.3f  "
                    "lineages=%d sentinel_mean=%.3f\n",
                    w->audit_reg.r2_classifier,
                    w->audit_reg.audit_mult_classifier,
                    w->audit_reg.r2_predictor,
                    w->audit_reg.audit_mult_predictor,
                    w->probe_panel.l_role_acc, w->probe_panel.l_fit_acc,
                    w->probe_panel.l_lineage_acc, w->n_lineage_stats,
                    sentinel_mean);
        std::fflush(stdout);
    }

    // ---- MONITOR: reference training (A-601, device kernel) ----
    // The replay buffer is host-only; sample a minibatch that never includes
    // held-out probe tuples, upload it, and run one AdamW step on the device.
    if (w->replay_buffer.filled >= REF_TRAIN_MINIBATCH) {
        bool batch_ok = true;
        for (int mb = 0; mb < REF_TRAIN_MINIBATCH && batch_ok; ++mb) {
            int idx = -1;
            for (int tries = 0; tries < 256 && idx < 0; ++tries) {
                int cand = static_cast<int>(
                    pcg32_random(&w->rng) %
                    static_cast<uint32_t>(w->replay_buffer.filled));
                if (!w->replay_buffer.held_out[cand]) idx = cand;
            }
            if (idx < 0) {
                batch_ok = false;
                break;
            }
            std::memcpy(&w->h_ref_batch_input[mb * predictor::REF_INPUT],
                        &w->replay_buffer.bmap[idx * BMAP_DIM],
                        BMAP_DIM * sizeof(float));
            std::memcpy(&w->h_ref_batch_input[mb * predictor::REF_INPUT + BMAP_DIM],
                        &w->replay_buffer.task_emb[idx * TASK_EMBED_DIM],
                        TASK_EMBED_DIM * sizeof(float));
            w->h_ref_batch_target[mb] = w->replay_buffer.fitness[idx];
        }
        if (batch_ok) {
            w->reference_reg.step++;
            TRANSFER_ABORT(cudaMemcpyAsync(w->d_ref_batch_input,
                            w->h_ref_batch_input,
                            REF_TRAIN_MINIBATCH * predictor::REF_INPUT
                                * sizeof(float),
                            cudaMemcpyHostToDevice, w->stream),
                           "upload reference batch input");
            TRANSFER_ABORT(cudaMemcpyAsync(w->d_ref_batch_target,
                            w->h_ref_batch_target,
                            REF_TRAIN_MINIBATCH * sizeof(float),
                            cudaMemcpyHostToDevice, w->stream),
                           "upload reference batch target");
            TRANSFER_ABORT(cudaMemcpyAsync(w->d_ref_step,
                            &w->reference_reg.step, sizeof(int),
                            cudaMemcpyHostToDevice, w->stream),
                           "upload reference step");
            if (!phase_run(&w->fg_world_train, w->stream, [&] {
                    predictor::launch_reference_train(
                        w->d_reference_reg, w->d_ref_batch_input,
                        w->d_ref_batch_target, w->d_ref_step, w->stream);
                })) {
                return false;
            }
        }
    }

    // ---- Surprise + CUSUM (A-601) ----
    // Reference surprise on the signed probe set (ground truth probe
    // fitness is populated by the probe evaluation below).
    float s_reference = evaluate_probe_reference(w);
    if (s_reference < 0.f) return false;

    // Predictor ensemble surprise: variance across the top-K predictors (by
    // fitness) assigned to each stationary probe slot, on the frozen target.
    float s_predictor = 0.f;
    if (w->bootstrap_fired) {
        int n_probe_slots = 0;
        float slot_var_sum = 0.f;
        for (int slot = 0; slot < cur::PREDICTOR_PROBE_SLOTS; ++slot) {
            // Collect assigned predictors with their fitness.
            int assigned[POOL_SIZE];
            float assigned_fit[POOL_SIZE];
            int k = 0;
            for (int org = 0; org < POOL_SIZE; ++org) {
                if (canonical_role(w->org_table.role[org]) != Role::Predictor) continue;
                if (org % cur::PREDICTOR_BATCH != slot) continue;
                assigned[k] = org;
                assigned_fit[k] = w->org_table.fitness[org];
                k++;
            }
            // Top-K by fitness (insertion sort, K small).
            for (int i = 1; i < k; ++i) {
                int org_key = assigned[i];
                float fit_key = assigned_fit[i];
                int j = i - 1;
                while (j >= 0 && assigned_fit[j] < fit_key) {
                    assigned[j + 1] = assigned[j];
                    assigned_fit[j + 1] = assigned_fit[j];
                    j--;
                }
                assigned[j + 1] = org_key;
                assigned_fit[j + 1] = fit_key;
            }
            int top_k = (k < PREDICTOR_ENSEMBLE_TOP_K) ? k : PREDICTOR_ENSEMBLE_TOP_K;
            if (top_k >= 2) {
                static float preds[POOL_SIZE * BMAP_DIM];
                for (int i = 0; i < top_k; ++i) {
                    std::memcpy(&preds[i * BMAP_DIM],
                                &w->h_descriptors[assigned[i] * BMAP_DIM],
                                BMAP_DIM * sizeof(float));
                }
                slot_var_sum += predictor::ensemble_surprise(preds, top_k);
                n_probe_slots++;
            }
        }
        if (n_probe_slots > 0) {
            s_predictor = slot_var_sum / static_cast<float>(n_probe_slots);
        }
    }

    // The correlation window and the r CUSUM start only when both signals
    // are live (post-bootstrap): before that r is undefined and only
    // reference surprise is used.
    float r = 0.f;
    float s_blended = s_reference;
    if (w->bootstrap_fired) {
        r = predictor::pearson_r_clipped(w->corr_window);
        s_blended = predictor::blend_surprise(s_reference, s_predictor, r);
        predictor::push_correlation(&w->corr_window, s_reference, s_predictor);
        safety::cusum_update(&w->cusum_r, r);
    }
    safety::cusum_update(&w->cusum_surprise, s_blended);

    if (log_this_gen) {
        float rho = (w->s_target_calibrated && w->s_target > EPS_DENOM)
            ? s_blended / w->s_target : 0.f;
        std::printf("         surprise s_ph=%.4e s_pr=%.4e r=%.4f "
                    "s_blend=%.4e rho=%.4f\n",
                    s_reference, s_predictor, r, s_blended, rho);
        // Dashboard surface (I9): role fraction, surprise ratio, ladder swap
        // statistics, and stress-failure flags in one periodic line.
        int n_c = 0;
        int n_p = 0;
        for (int i = 0; i < POOL_SIZE; ++i) {
            if (canonical_role(w->org_table.role[i]) == Role::Classifier) n_c++;
            else n_p++;
        }
        std::printf("         [DASHBOARD] role_frac_C=%.3f role_frac_P=%.3f "
                    "r=%.4f rho=%.4f swaps=%d/%d stress_flagged=%d "
                    "archive=%d\n",
                    static_cast<float>(n_c) / POOL_SIZE,
                    static_cast<float>(n_p) / POOL_SIZE,
                    r, rho,
                    w->mutation_ladder.swaps_accepted,
                    w->mutation_ladder.swaps_attempted,
                    w->stress_ladder.flagged_lineage_count,
                    archive::archive_size(w->archive));
        std::fflush(stdout);
    }

    // Rolling blended-surprise history for rho = s_avg / s_target.
    w->s_blended_history[w->s_hist_head] = s_blended;
    w->s_hist_head = (w->s_hist_head + 1) % HYBRID_R_WINDOW;
    if (w->s_hist_filled < HYBRID_R_WINDOW) w->s_hist_filled++;

    // CUSUM calibration (A-601): collect during the calibration window after
    // bootstrap, then freeze s_target (median) and set k = 0.5 sigma,
    // h = 5 sigma for the surprise CUSUM.
    if (w->bootstrap_fired && !w->s_target_calibrated) {
        int rel = gen - w->bootstrap_gen;
        if (rel >= CALIBRATION_GEN_LO && rel <= CALIBRATION_GEN_HI &&
            w->n_calibration_samples <
                static_cast<int>(sizeof(w->calibration_samples) / sizeof(float))) {
            w->calibration_samples[w->n_calibration_samples++] = s_blended;
        }
        if (rel >= CALIBRATION_GEN_HI && w->n_calibration_samples > 0) {
            static float sorted[CALIBRATION_GEN_HI - CALIBRATION_GEN_LO + 1];
            int n = w->n_calibration_samples;
            std::memcpy(sorted, w->calibration_samples, n * sizeof(float));
            for (int i = 1; i < n; ++i) {
                float key = sorted[i];
                int j = i - 1;
                while (j >= 0 && sorted[j] > key) { sorted[j + 1] = sorted[j]; j--; }
                sorted[j + 1] = key;
            }
            float median = sorted[n / 2];
            float mean = 0.f;
            for (int i = 0; i < n; ++i) mean += sorted[i];
            mean /= static_cast<float>(n);
            float var = 0.f;
            for (int i = 0; i < n; ++i) {
                float d = sorted[i] - mean;
                var += d * d;
            }
            var /= static_cast<float>(n);
            float sigma = sqrtf(var);
            w->s_target = median;
            w->s_target_calibrated = true;
            w->cusum_surprise.reference = median;
            w->cusum_surprise.allowance = 0.5f * sigma;
            w->cusum_surprise.threshold = 5.0f * sigma;
            std::printf("[CALIBRATION] gen %d: s_target=%.6f sigma=%.6f "
                        "(n=%d)\n", gen, median, sigma, n);
            std::fflush(stdout);
        }
    }

    // ---- EVOLVE: spawn wave ----
    spawn_wave(w);

    // ---- Periodic: PCA rebin ----
    if (w->generation > 0 && w->generation % AUDIT_INTERVAL == 0) {
        archive::recompute_bins(&w->archive, w->stream);
    }

    // ---- Operator checks ----
    safety::alignment::apply_operator_command(
        w->org_table.fitness, w->org_table.lineage_id,
        POOL_SIZE, &w->operator_state);

    w->generation++;

    // ---- Checkpoint (S-001): every completed generation is saved, so a
    // killed run resumes at the last generation instead of losing the run.
    {
        bool requested = w->operator_state.checkpoint_requested;
        if (!sync_reference_from_device(w)) return false;
        if (!save_checkpoint(w, w->checkpoint_path)) {
            return false;
        }
        if (requested) {
            std::printf("[OPERATOR] checkpoint written at generation %d\n",
                        w->generation);
            std::fflush(stdout);
            w->operator_state.checkpoint_requested = false;
        }
    }
    return true;
}

// ---- Run -------------------------------------------------------------------

// Poll the operator command file, apply durable effects (pause state, pruned
// lineages zeroed in the pool AND tombstoned in the archive), and return.
static void poll_operator_commands(World* w) {
    if (!safety::alignment::apply_operator_command(
            w->org_table.fitness, w->org_table.lineage_id,
            POOL_SIZE, &w->operator_state)) {
        return;
    }
    for (int p = 0; p < w->operator_state.n_pruned; ++p) {
        archive::prune_lineage(&w->archive,
                               LineageId(w->operator_state.pruned_lineages[p]));
        safety::sentinel_history_mark_pruned(&w->sentinel_history,
                                             w->operator_state.pruned_lineages[p],
                                             w->generation);
    }
}

void run(int n_generations, bool resume, const char* checkpoint_path) {
    World* w = new World;
    if (!initialize_world(w)) {
        delete w;
        std::printf("=== RUN INVALIDATED: initialization failed ===\n");
        std::fflush(stdout);
        return;
    }
    w->checkpoint_path = checkpoint_path ? checkpoint_path
                                         : CHECKPOINT_DEFAULT_PATH;

    if (resume) {
        if (!load_checkpoint(w, w->checkpoint_path)) {
            std::printf("=== RUN INVALIDATED: resume failed (no usable "
                        "checkpoint at %s) ===\n", w->checkpoint_path);
            std::fflush(stdout);
            free_gpu_buffers(w);
            delete w;
            return;
        }
        std::printf("Resumed from %s at generation %d (archive size %d)\n",
                    w->checkpoint_path, w->generation,
                    archive::archive_size(w->archive));
        std::fflush(stdout);
        if (!sync_reference_to_device(w)) {
            std::printf("=== RUN INVALIDATED: reference restore failed ===\n");
            std::fflush(stdout);
            free_gpu_buffers(w);
            delete w;
            return;
        }
        if (w->probe_set.probe_tuples_signed && !upload_probe_batch(w)) {
            std::printf("=== RUN INVALIDATED: probe restore failed ===\n");
            std::fflush(stdout);
            free_gpu_buffers(w);
            delete w;
            return;
        }
    } else {
        std::printf("Fresh run; checkpoints will be written to %s\n",
                    w->checkpoint_path);
        std::fflush(stdout);
    }

    bool valid = true;
    for (int g = w->generation; g < n_generations; ++g) {
        if (safety::alignment::poll_off_switch()) {
            std::printf("Shutdown flag detected at generation %d\n", g);
            std::fflush(stdout);
            break;
        }

        poll_operator_commands(w);

        // Pause gating: while paused, do not advance generations or mutate
        // state; keep polling for resume/shutdown/operator commands.
        while (w->operator_state.paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (safety::alignment::poll_off_switch()) {
                std::printf("Shutdown flag detected while paused\n");
                std::fflush(stdout);
                break;
            }
            poll_operator_commands(w);
        }
        if (w->operator_state.paused) break;  // shutdown while paused

        if (!step_generation(w)) {
            valid = false;
            break;
        }
    }

    // Final report.
    int asize = archive::archive_size(w->archive);
    if (valid) {
        std::printf("\n=== Run complete: %d generations, archive size = %d ===\n",
                    w->generation, asize);
    } else {
        std::printf("\n=== RUN INVALIDATED at generation %d: CUDA error or "
                    "nonfinite numerical state ===\n", w->generation);
    }

    int occupied_bins = 0;
    for (int b = 0; b < ARCHIVE_BINS_X * ARCHIVE_BINS_Y; ++b) {
        if (w->archive.bins[b].count_classifier > 0 ||
            w->archive.bins[b].count_predictor > 0) {
            occupied_bins++;
        }
    }
    std::printf("Occupied PCA bins: %d / %d\n", occupied_bins, ARCHIVE_BINS_X * ARCHIVE_BINS_Y);
    std::printf("Checkpoint: %s (generation %d)\n",
                w->checkpoint_path, w->generation);
    std::fflush(stdout);
    if (g_profile) print_phase_timings();

    free_gpu_buffers(w);
    delete w;
}

}  // namespace slime::integration

// ---- Entry point -----------------------------------------------------------

#ifndef COEVO_NO_MAIN
int main(int argc, char** argv) {
    int n_gen = 100;
    bool resume = false;
    const char* ckpt_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--resume") == 0) {
            resume = true;
        } else if (std::strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) {
            ckpt_path = argv[++i];
        } else if (std::strcmp(argv[i], "--profile") == 0) {
            slime::integration::g_profile = true;
        } else {
            int parsed = std::atoi(argv[i]);
            if (parsed > 0) n_gen = parsed;
        }
    }

    std::printf("Slime Evolution — co-evolving NCA system\n");
    if (ckpt_path != nullptr) {
        std::printf("%s run: %d generations, checkpoints at %s\n",
                    resume ? "Resuming" : "Fresh", n_gen, ckpt_path);
    } else {
        std::printf("%s run: %d generations, no checkpoint path\n",
                    resume ? "Resuming" : "Fresh", n_gen);
    }
    std::fflush(stdout);

    slime::integration::run(n_gen, resume, ckpt_path);
    return 0;
}
#endif  // COEVO_NO_MAIN










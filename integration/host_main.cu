// Wave 2: Host Main — Allocation, Initialization, Generation Loop, Entry Point
//
// Per cuda_engineering.md sections 2, 3, 5, 6, 8, 9, 12, 13, 15 and
// construction_plan.md Wave 2. Classifier-only loop (predictors activate
// in Wave 4 after bootstrap).

#include "main_loop.cu"
#include "../safety/alignment.cu"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace slime::integration {

using namespace slime;
namespace cur = slime::curriculum;

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

// ---- Probe set evaluation (Wave 3, A-601) -----------------------------------
// Evaluate placeholder surprise on the probe set. Returns the mean surprise
// across all 64 probe samples.
static float evaluate_probe_placeholder(
    const predictor::PlaceholderRegressor& reg,
    const cur::ProbeSet& ps,
    const float* probe_fitness,
    uint64_t host_sot_key)
{
    if (!cur::verify_probe_set(ps, host_sot_key)) return 0.f;

    float total = 0.f;
    int count = 0;
    for (int b = 0; b < 4; ++b) {
        const cur::ClassifierBatch& batch = ps.classifier_probes[b];
        for (int s = 0; s < cur::CLASSIFIER_BATCH; ++s) {
            float zero_bmap[BMAP_DIM] = {};
            int probe_idx = b * cur::CLASSIFIER_BATCH + s;
            float surprise = predictor::placeholder_surprise(
                reg, zero_bmap, batch.task_embedding,
                probe_fitness[probe_idx]);
            total += surprise;
            count++;
        }
    }
    return (count > 0) ? total / static_cast<float>(count) : 0.f;
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

    // Gradient health: pinned host scalar (section 8).
    CUDA_ABORT(cudaMalloc(&w->d_grad_norm, sizeof(float)), "alloc d_grad_norm");

    // Numerical telemetry (A-501).
    CUDA_ABORT(cudaMalloc(&w->d_tel, sizeof(TelemetryScalars)), "alloc d_tel");

    // SOT reference buffers (section 12): pre-allocated, bounded by SOT_MAX_REFS.
    CUDA_ABORT(cudaMalloc(&w->d_sot_temp_images, cur::SOT_SUBBATCH * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half)), "alloc d_sot_temp_images");
    CUDA_ABORT(cudaMalloc(&w->d_sot_task_emb,    TASK_EMBED_DIM * sizeof(float)), "alloc d_sot_task_emb");
    CUDA_ABORT(cudaMalloc(&w->d_sot_fwd_inputs,  cur::SOT_MAX_REFS * sizeof(ForwardInputs)), "alloc d_sot_fwd_inputs");
    CUDA_ABORT(cudaMalloc(&w->d_sot_descriptors, cur::SOT_MAX_REFS * BMAP_DIM * sizeof(float)), "alloc d_sot_descriptors");
    CUDA_ABORT(cudaMalloc(&w->d_sot_bank_of,     cur::SOT_MAX_REFS * sizeof(int)), "alloc d_sot_bank_of");

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
    CUDA_ABORT(cudaMalloc(&w->bwd_workspace.recomp[0],  GRID_ELEMS * sizeof(__half) * POOL_SIZE), "alloc bwd recomp[0]");
    CUDA_ABORT(cudaMalloc(&w->bwd_workspace.recomp[1],  GRID_ELEMS * sizeof(__half) * POOL_SIZE), "alloc bwd recomp[1]");

    // Section 2.2: pinned host buffers.
    CUDA_ABORT(cudaMallocHost(&w->h_descriptors, POOL_SIZE * BMAP_DIM * sizeof(float)), "allocHost h_descriptors");
    CUDA_ABORT(cudaMallocHost(&w->h_btraj,       POOL_SIZE * BTRAJ_SAMPLES * BMAP_DIM * sizeof(float)), "allocHost h_btraj");
    CUDA_ABORT(cudaMallocHost(&w->h_seed_grad,   POOL_SIZE * BMAP_DIM * sizeof(float)), "allocHost h_seed_grad");
    CUDA_ABORT(cudaMallocHost(&w->h_fwd_inputs,  POOL_SIZE * sizeof(ForwardInputs)), "allocHost h_fwd_inputs");
    CUDA_ABORT(cudaMallocHost(&w->h_weights,     TOTAL_WEIGHTS * sizeof(float)), "allocHost h_weights");
    CUDA_ABORT(cudaMallocHost(&w->h_tel,         sizeof(TelemetryScalars)), "allocHost h_tel");

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
    cudaFree(w->d_organisms);
    cudaFree(w->d_weights);
    cudaFree(w->d_eff_weights);
    cudaFree(w->d_deltas);
    cudaFree(w->d_fwd_inputs);
    cudaFree(w->d_checkpoints);
    cudaFree(w->d_grads);
    cudaFree(w->d_mean_grad);
    cudaFree(w->d_came_m);
    cudaFree(w->d_came_v);
    cudaFree(w->d_came_c);
    cudaFree(w->d_came_prev_u);
    cudaFree(w->d_descriptors);
    cudaFree(w->d_seed_grad);
    cudaFree(w->d_batch_image);
    cudaFree(w->d_batch_task_emb);
    cudaFree(w->d_btraj);
    cudaFree(w->d_grad_norm);
    cudaFree(w->d_tel);
    cudaFree(w->d_sot_temp_images);
    cudaFree(w->d_sot_task_emb);
    cudaFree(w->d_sot_fwd_inputs);
    cudaFree(w->d_sot_descriptors);
    cudaFree(w->d_sot_bank_of);
    cudaFree(w->d_pt_swap_org);
    cudaFree(w->d_pt_swap_ckpt);
    cudaFree(w->d_pt_swap_grad);
    cudaFree(w->d_pt_swap_wbank);
    cudaFree(w->bwd_workspace.d_state[0]);
    cudaFree(w->bwd_workspace.d_state[1]);
    cudaFree(w->bwd_workspace.d_perc);
    cudaFree(w->bwd_workspace.recomp[0]);
    cudaFree(w->bwd_workspace.recomp[1]);
    cudaFreeHost(w->h_descriptors);
    cudaFreeHost(w->h_btraj);
    cudaFreeHost(w->h_seed_grad);
    cudaFreeHost(w->h_fwd_inputs);
    cudaFreeHost(w->h_weights);
    cudaFreeHost(w->h_tel);
    cudaStreamDestroy(w->stream);
}

// ---- Kaiming He weight initialization (section 15.2) ------------------------
// Box-Muller transform: given two uniform [0,1) draws, produce a standard
// normal sample.
static float box_muller_normal(Pcg32* rng) {
    float u1 = pcg32_float(rng);
    float u2 = pcg32_float(rng);
    // Avoid log(0).
    if (u1 < 1e-30f) u1 = 1e-30f;
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
        float scale = sqrtf(1.0f / static_cast<float>(nca::HIDDEN_DIM));
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

        w->org_table.lineage_id[i] = static_cast<uint32_t>(i);
        w->org_table.parent_id[i] = 0;
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

    // Section 9.1: Archive initialization.
    std::memset(&w->archive, 0, sizeof(w->archive));
    for (int b = 0; b < ARCHIVE_BINS_X * ARCHIVE_BINS_Y; ++b) {
        w->archive.bins[b].cap_classifier = 13;
        w->archive.bins[b].cap_predictor  = 13;
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

    // Wave 3: Placeholder regressor, replay buffer, probe set, correlation window.
    predictor::init_placeholder_regressor(&w->placeholder_reg, &w->rng);
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

// Cross-entropy classification loss + seed gradient computation.
// Uses first NUM_CLASSES dims of bmap_64 as logits (section 6).
// The ENTIRE seed-gradient buffer is zeroed first: rows belonging to
// inactive roles must carry no gradient into backward (cudaMallocHost does
// not guarantee zeros).
static void score_classifiers(World* w) {
    std::memset(w->h_seed_grad, 0, POOL_SIZE * BMAP_DIM * sizeof(float));

    float sum_loss = 0.f;
    float max_abs_logit = 0.f;
    int n_evaluated = 0;

    for (int org = 0; org < POOL_SIZE; ++org) {
        if (canonical_role(w->org_table.role[org]) != Role::Classifier) continue;
        int sample_idx = w->org_table.batch_sample_idx[org];
        int target = w->classifier_batch.label[sample_idx];

        const float* bmap = &w->h_descriptors[org * BMAP_DIM];

        float dlogits[NUM_CLASSES];
        float loss;
        autodiff::classifier_loss(bmap, target, NUM_CLASSES, dlogits, &loss);

        for (int d = 0; d < NUM_CLASSES; ++d) {
            float a = fabsf(bmap[d]);
            if (a > max_abs_logit) max_abs_logit = a;
        }
        sum_loss += loss;
        n_evaluated++;

        float task_fitness_proxy = expf(-loss);  // smooth (0,1] proxy, not accuracy
        w->org_table.f_raw[org] = task_fitness_proxy * archive::sot_gate(w->org_table.f_sot[org]);

        // Before Wave 5: audit_mult = 1.0, variance_mult = 1.0.
        w->org_table.fitness[org] = archive::compose_fitness(
            w->org_table.f_raw[org], 1.0f, 1.0f, 1.0f);

        // Seed gradient: d(CE)/d(logits) in first NUM_CLASSES dims, 0 elsewhere.
        float* sg = &w->h_seed_grad[org * BMAP_DIM];
        for (int d = 0; d < NUM_CLASSES; ++d) sg[d] = dlogits[d];
    }

    w->last_mean_ce = (n_evaluated > 0) ? sum_loss / static_cast<float>(n_evaluated) : 0.f;
    w->last_max_abs_logit = max_abs_logit;
}

// ---- Archive insertion (section 5, 9.1) ------------------------------------

static void insert_into_archive(World* w) {
    for (int org = 0; org < POOL_SIZE; ++org) {
        if (canonical_role(w->org_table.role[org]) != Role::Classifier) continue;

        archive::ArchiveEntry cand;
        std::memcpy(cand.descriptor, &w->h_descriptors[org * BMAP_DIM],
                    BMAP_DIM * sizeof(float));
        archive::rff_project(w->archive.rff, cand.descriptor, cand.rff_proj);
        cand.fitness = w->org_table.fitness[org];
        cand.f_raw = w->org_table.f_raw[org];
        cand.f_sot = w->org_table.f_sot[org];
        cand.lineage_id = w->org_table.lineage_id[org];
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

// ---- Spawn wave (section 9) ------------------------------------------------

static void spawn_wave(World* w) {
    int asize = archive::archive_size(w->archive);
    if (asize == 0) return;

    // Before bootstrap, all spawns are classifiers.
    Role target_role = Role::Classifier;

    // Select all WAVE_SIZE victims up front, without replacement. Selecting
    // them one at a time while zeroing the freshly installed child's fitness
    // would repeatedly pick the same newborn slot and overwrite it.
    int victims[WAVE_SIZE];
    int n_victims = genome::select_spawn_victims(
        w->org_table.role, w->org_table.fitness, POOL_SIZE,
        target_role, WAVE_SIZE, victims);
    if (n_victims <= 0) return;

    for (int spawn = 0; spawn < n_victims; ++spawn) {
        int worst_idx = victims[spawn];

        // Select parent from per-role live index list (section 9).
        int parent_archive_idx = -1;
        if (target_role == Role::Classifier && w->archive.n_alive_classifier > 0) {
            int list_idx = static_cast<int>(pcg32_random(&w->rng) % w->archive.n_alive_classifier);
            parent_archive_idx = w->archive.alive_classifier_idx[list_idx];
        } else if (target_role == Role::Predictor && w->archive.n_alive_predictor > 0) {
            int list_idx = static_cast<int>(pcg32_random(&w->rng) % w->archive.n_alive_predictor);
            parent_archive_idx = w->archive.alive_predictor_idx[list_idx];
        }
        if (parent_archive_idx < 0) continue;

        // Copy genome from archive parent.
        genome::Genome child_genome = w->archive.entries[parent_archive_idx].genome;

        // Mutation rate from the replaced organism's replica (section 9).
        uint8_t replica = w->org_table.replica_tag[worst_idx];
        float mut_rate = PT_MUTATION_RATES[replica];
        genome::mutate(&child_genome, mut_rate, MUTATION_RATE_ROLE, &w->rng);

        // Pre-bootstrap role lock: predictor inputs, loss, seed gradients, and
        // archive behavior do not exist yet, so every spawn is forced back to
        // the classifier role regardless of what the role-bit mutation drew.
        genome::force_role(child_genome, Role::Classifier);

        // Install child into the pool slot.
        w->org_table.genomes[worst_idx] = child_genome;
        w->org_table.role[worst_idx] = genome::runtime_role(child_genome);
        w->org_table.lineage_id[worst_idx] = w->archive.entries[parent_archive_idx].lineage_id;
        w->org_table.parent_id[worst_idx] = static_cast<uint32_t>(parent_archive_idx);
        w->org_table.spawn_gen[worst_idx] = w->generation;
        w->org_table.fitness[worst_idx] = 0.f;
        w->org_table.f_raw[worst_idx] = 0.f;
        w->org_table.f_sot[worst_idx] = 1.f;

        // Re-initialize delta from genome prior.
        genome::init_delta_from_prior(child_genome,
                                      &w->org_table.deltas[worst_idx]);
    }
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
    ctx.stream       = w->stream;
    return ctx;
}

// ---- Step generation (section 5 data flow) ---------------------------------

// Phase progress trace: prints phase tag + checks CUDA errors after each sync.
// Always flushed so output is never lost to buffering. Returns false on any
// CUDA error — a failed phase invalidates the run (fail-fast, A-501).
static bool phase_trace(const char* tag, int gen, cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
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
    std::printf("  gen %d: %s\n", gen, tag);
    std::fflush(stdout);
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
//   MONITOR   placeholder train, surprise, CUSUM, PCA rebin, operator checks
//
// Returns false when the generation invalidated the run (CUDA error or a
// nonfinite numerical state); run() then aborts the experiment.
bool step_generation(World* w) {
    int gen = w->generation;
    std::printf("step_generation(%d) begin\n", gen);
    std::fflush(stdout);

    // Curriculum refresh every CURRICULUM_INTERVAL generations (section 15.7).
    if (gen % CURRICULUM_INTERVAL == 0) {
        cur::assemble_classifier_batch(&w->classifier_batch,
                                       MAIN_SOT_DENSITY,
                                       w->host_sot_key, &w->rng);
    }

    // Organism-to-batch assignment: deterministic round-robin (A-401).
    for (int i = 0; i < POOL_SIZE; ++i) {
        w->org_table.batch_sample_idx[i] = i % cur::CLASSIFIER_BATCH;
    }

    // Set up h_fwd_inputs for each organism.
    for (int i = 0; i < POOL_SIZE; ++i) {
        int s = w->org_table.batch_sample_idx[i];
        ForwardInputs& fi = w->h_fwd_inputs[i];
        fi.role = w->org_table.role[i];
        fi.image_rgb = w->d_batch_image + s * GRID_SIZE * GRID_SIZE * 3;
        fi.task_embedding = w->d_batch_task_emb;
        fi.target_bmap_32 = nullptr;
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
    if (!phase_trace("T1_H2D", gen, w->stream)) return false;

    // ---- GPU: materialize W_eff = W_shared + delta per organism ----
    autodiff::launch_materialize_effective_weights(
        w->d_weights, w->d_deltas, w->d_eff_weights, POOL_SIZE, w->stream);

    // ---- GPU: forward_with_checkpoints (per-organism effective weights) ----
    autodiff::launch_forward_with_checkpoints(
        w->d_organisms, w->d_fwd_inputs, nullptr,
        w->d_weights, w->d_eff_weights, w->d_checkpoints,
        POOL_SIZE, w->stream);
    if (!phase_trace("forward", gen, w->stream)) return false;

    // ---- GPU: extract_descriptor ----
    nca::extract_descriptor(w->d_organisms, w->d_descriptors,
                            POOL_SIZE, w->stream);

    // ---- GPU: btraj_gather ----
    autodiff::launch_btraj_gather(w->d_organisms, w->d_btraj,
                                  POOL_SIZE, w->stream);
    if (!phase_trace("descriptor+btraj", gen, w->stream)) return false;

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
    safety::alignment::apply_sot_identity(
        w->d_organisms, w->d_weights, w->d_eff_weights,
        w->classifier_batch, w->host_sot_key,
        w->h_descriptors, w->org_table.batch_sample_idx,
        w->org_table.f_sot,
        w->d_sot_temp_images, w->d_sot_task_emb,
        w->d_sot_fwd_inputs, w->d_sot_descriptors,
        w->d_sot_bank_of, TOTAL_WEIGHTS,
        w->stream);
    if (!phase_trace("SOT", gen, w->stream)) return false;

    // ---- Score organisms (section 6) ----
    score_classifiers(w);

    // ---- Archive insertion (section 9.1) ----
    insert_into_archive(w);

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
        safety::pt::propose_swaps(&w->mutation_ladder,
                                  w->org_table.fitness,
                                  &w->rng, ctx);
    }
    if (!phase_trace("score+archive+PT", gen, w->stream)) return false;

    // ---- T3: H→D seed_grad (AFTER PT swaps) ----
    TRANSFER_ABORT(cudaMemcpyAsync(w->d_seed_grad, w->h_seed_grad,
                    POOL_SIZE * BMAP_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream), "T3 seed_grad");

    // ---- GPU: backward (phase-decomposed batched, all orgs in parallel) ----
    autodiff::launch_backward_all(
        w->d_organisms, w->d_weights, w->d_eff_weights, w->d_seed_grad,
        w->d_checkpoints, w->d_grads,
        w->bwd_workspace, POOL_SIZE, w->stream);
    if (!phase_trace("backward", gen, w->stream)) return false;

    // ---- GPU: aggregate gradients ----
    optimizer::launch_aggregate_gradients(w->d_grads, w->d_mean_grad,
                                          POOL_SIZE, w->stream);

    // ---- GPU: gradient norm via device-side reduction (section 8) ----
    optimizer::launch_grad_norm_reduce(w->d_mean_grad, w->d_grad_norm,
                                       w->stream);

    // ---- GPU: CAME step ----
    optimizer::CameState came_state;
    came_state.d_m = w->d_came_m;
    came_state.d_v = w->d_came_v;
    came_state.d_c = w->d_came_c;
    came_state.d_prev_u = w->d_came_prev_u;
    came_state.d_mean_grad = w->d_mean_grad;
    came_state.step = gen;
    optimizer::launch_came_step(w->d_weights, came_state,
                                optimizer::CAME_DEFAULTS, w->stream);

    // ---- GPU: numerical telemetry (A-501) ----
    // launch_telemetry_kernels zeroes the TelemetryScalars struct first, so
    // it must be enqueued BEFORE the state-saturation kernel writes its two
    // fields; stream order preserves the accumulation.
    optimizer::launch_telemetry_kernels(
        w->d_mean_grad, w->d_weights,
        w->d_came_m, w->d_came_v, w->d_came_c, w->d_came_prev_u,
        w->d_tel, w->stream);
    autodiff::launch_state_saturation(w->d_checkpoints, w->d_organisms,
                                      w->d_tel, POOL_SIZE, w->stream);
    if (!phase_trace("optimizer", gen, w->stream)) return false;

    // ---- Telemetry readback (single small struct D→H) ----
    TRANSFER_ABORT(cudaMemcpy(w->h_tel, w->d_tel, sizeof(TelemetryScalars),
                              cudaMemcpyDeviceToHost), "telemetry struct");
    float h_grad_norm_sq = 0.f;
    TRANSFER_ABORT(cudaMemcpy(&h_grad_norm_sq, w->d_grad_norm, sizeof(float),
                              cudaMemcpyDeviceToHost), "grad norm scalar");
    float grad_norm = sqrtf(h_grad_norm_sq);

    // Nonfinite numerical state invalidates the run immediately.
    if (!(w->h_tel->nonfinite_count == 0.f) ||
        !std::isfinite(grad_norm) ||
        !std::isfinite(w->h_tel->state_max_abs)) {
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
    if (w->generation % TELEMETRY_INTERVAL == 0 || w->generation < 5) {
        float sum_fit = 0.f, sum_raw = 0.f;
        int count = 0;
        for (int i = 0; i < POOL_SIZE; ++i) {
            if (canonical_role(w->org_table.role[i]) == Role::Classifier) {
                sum_fit += w->org_table.fitness[i];
                sum_raw += w->org_table.f_raw[i];
                count++;
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
        std::fflush(stdout);
    }

    // ---- EVOLVE: spawn wave ----
    spawn_wave(w);

    // ---- MONITOR: placeholder training (A-601, Wave 3) ----
    predictor::placeholder_train_step(&w->placeholder_reg,
                                      &w->replay_buffer, &w->rng);

    // ---- Surprise + CUSUM (A-601, Wave 3) ----
    // Compute s_placeholder on the probe set.
    float s_placeholder = evaluate_probe_placeholder(
        w->placeholder_reg, w->probe_set,
        w->probe_fitness, w->host_sot_key);
    // Before bootstrap, s_predictor = 0 and r = 0 (placeholder dominates).
    float s_predictor = 0.f;
    float r = predictor::pearson_r_clipped(w->corr_window);
    float s_blended = predictor::blend_surprise(s_placeholder, s_predictor, r);
    predictor::push_correlation(&w->corr_window, s_placeholder, s_predictor);
    safety::cusum_update(&w->cusum_surprise, s_blended);
    safety::cusum_update(&w->cusum_r, r);

    // ---- Periodic: PCA rebin ----
    if (w->generation > 0 && w->generation % AUDIT_INTERVAL == 0) {
        archive::recompute_bins(&w->archive, w->stream);
    }

    // ---- Operator checks ----
    bool paused = false;
    bool force_checkpoint = false;
    safety::alignment::apply_operator_command(
        w->org_table.fitness, w->org_table.lineage_id,
        POOL_SIZE, &paused, &force_checkpoint);

    w->generation++;
    return true;
}

// ---- Run -------------------------------------------------------------------

void run(int n_generations) {
    World* w = new World;
    if (!initialize_world(w)) {
        delete w;
        std::printf("=== RUN INVALIDATED: initialization failed ===\n");
        std::fflush(stdout);
        return;
    }

    bool valid = true;
    for (int g = 0; g < n_generations; ++g) {
        if (safety::alignment::poll_off_switch()) {
            std::printf("Shutdown flag detected at generation %d\n", g);
            std::fflush(stdout);
            break;
        }
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
    std::fflush(stdout);

    free_gpu_buffers(w);
    delete w;
}

}  // namespace slime::integration

// ---- Entry point -----------------------------------------------------------

int main(int argc, char** argv) {
    int n_gen = 100;
    if (argc > 1) n_gen = std::atoi(argv[1]);
    if (n_gen <= 0) n_gen = 100;

    std::printf("Slime Evolution — co-evolving NCA system\n");
    std::printf("Running %d generations\n", n_gen);
    std::fflush(stdout);

    slime::integration::run(n_gen);
    return 0;
}

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

// ---- GPU buffer allocation/free (section 2) --------------------------------

static void alloc_gpu_buffers(World* w) {
    cudaStreamCreate(&w->stream);

    // Section 2.1: device-resident buffers.
    cudaMalloc(&w->d_organisms,    TOTAL_ORG * sizeof(OrganismState));
    cudaMalloc(&w->d_weights,      TOTAL_WEIGHTS * sizeof(float));
    cudaMalloc(&w->d_fwd_inputs,   POOL_SIZE * sizeof(ForwardInputs));
    cudaMalloc(&w->d_checkpoints,  POOL_SIZE * sizeof(CheckpointBuffer));
    cudaMalloc(&w->d_grads,        POOL_SIZE * sizeof(GradBuffers));
    cudaMalloc(&w->d_mean_grad,    TOTAL_WEIGHTS * sizeof(float));
    cudaMalloc(&w->d_came_m,       TOTAL_WEIGHTS * sizeof(float));
    cudaMalloc(&w->d_came_v,       TOTAL_WEIGHTS * sizeof(float));
    cudaMalloc(&w->d_came_c,       TOTAL_WEIGHTS * sizeof(float));
    cudaMalloc(&w->d_came_prev_u,  TOTAL_WEIGHTS * sizeof(float));
    cudaMalloc(&w->d_descriptors,  POOL_SIZE * BMAP_DIM * sizeof(float));
    cudaMalloc(&w->d_seed_grad,    POOL_SIZE * BMAP_DIM * sizeof(float));
    cudaMalloc(&w->d_batch_image,  cur::CLASSIFIER_BATCH * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half));
    cudaMalloc(&w->d_batch_task_emb, TASK_EMBED_DIM * sizeof(float));
    cudaMalloc(&w->d_btraj,        POOL_SIZE * BTRAJ_SAMPLES * BMAP_DIM * sizeof(float));

    // Gradient health: pinned host scalar (section 8).
    cudaMalloc(&w->d_grad_norm, sizeof(float));

    // SOT reference buffers (section 12): pre-allocated, bounded by SOT_SUBBATCH.
    cudaMalloc(&w->d_sot_temp_images, cur::SOT_SUBBATCH * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half));
    cudaMalloc(&w->d_sot_task_emb,    TASK_EMBED_DIM * sizeof(float));
    cudaMalloc(&w->d_sot_fwd_inputs,  cur::SOT_SUBBATCH * sizeof(ForwardInputs));
    cudaMalloc(&w->d_sot_descriptors, cur::SOT_SUBBATCH * BMAP_DIM * sizeof(float));

    // PT swap temp buffers (section 13): one organism's worth each.
    cudaMalloc(&w->d_pt_swap_org,  sizeof(OrganismState));
    cudaMalloc(&w->d_pt_swap_ckpt, sizeof(CheckpointBuffer));
    cudaMalloc(&w->d_pt_swap_grad, sizeof(GradBuffers));

    // Section 10: backward workspace (per-organism, for batched backward).
    constexpr int GRID_ELEMS = GRID_SIZE * GRID_SIZE * CA_CHANNELS;
    constexpr int PERC_ELEMS = GRID_SIZE * GRID_SIZE * autodiff::PERC_DIM;
    w->bwd_workspace.n_organisms = POOL_SIZE;
    cudaMalloc(&w->bwd_workspace.d_state[0], GRID_ELEMS * sizeof(float) * POOL_SIZE);
    cudaMalloc(&w->bwd_workspace.d_state[1], GRID_ELEMS * sizeof(float) * POOL_SIZE);
    cudaMalloc(&w->bwd_workspace.d_perc,     PERC_ELEMS * sizeof(float) * POOL_SIZE);
    cudaMalloc(&w->bwd_workspace.recomp[0],  GRID_ELEMS * sizeof(__half) * POOL_SIZE);
    cudaMalloc(&w->bwd_workspace.recomp[1],  GRID_ELEMS * sizeof(__half) * POOL_SIZE);

    // Section 2.2: pinned host buffers.
    cudaMallocHost(&w->h_descriptors, POOL_SIZE * BMAP_DIM * sizeof(float));
    cudaMallocHost(&w->h_btraj,       POOL_SIZE * BTRAJ_SAMPLES * BMAP_DIM * sizeof(float));
    cudaMallocHost(&w->h_seed_grad,   POOL_SIZE * BMAP_DIM * sizeof(float));
    cudaMallocHost(&w->h_fwd_inputs,  POOL_SIZE * sizeof(ForwardInputs));
    cudaMallocHost(&w->h_weights,     TOTAL_WEIGHTS * sizeof(float));

    // Zero CAME state on device.
    cudaMemset(w->d_came_m,      0, TOTAL_WEIGHTS * sizeof(float));
    cudaMemset(w->d_came_v,      0, TOTAL_WEIGHTS * sizeof(float));
    cudaMemset(w->d_came_c,      0, TOTAL_WEIGHTS * sizeof(float));
    cudaMemset(w->d_came_prev_u, 0, TOTAL_WEIGHTS * sizeof(float));

    // Zero organism grids.
    cudaMemset(w->d_organisms, 0, TOTAL_ORG * sizeof(OrganismState));
}

static void free_gpu_buffers(World* w) {
    cudaFree(w->d_organisms);
    cudaFree(w->d_weights);
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
    cudaFree(w->d_sot_temp_images);
    cudaFree(w->d_sot_task_emb);
    cudaFree(w->d_sot_fwd_inputs);
    cudaFree(w->d_sot_descriptors);
    cudaFree(w->d_pt_swap_org);
    cudaFree(w->d_pt_swap_ckpt);
    cudaFree(w->d_pt_swap_grad);
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

void initialize_world(World* w) {
    std::memset(w, 0, sizeof(World));
    alloc_gpu_buffers(w);

    // Section 15.1: PRNG.
    pcg32_seed(&w->rng, PCG32_DEFAULT_STATE, PCG32_DEFAULT_STREAM);

    w->generation = 0;
    w->bootstrap_fired = false;
    w->bootstrap_gen = -1;
    w->s_target = 0.f;
    w->s_target_calibrated = false;
    w->host_sot_key = 0xDEADCAFE42ULL;   // Section 15.3.
    w->grad_health_warn_count = 0;

    // Section 15.2: Kaiming He weight initialization.
    kaiming_he_init(w->h_weights, &w->rng);
    cudaMemcpy(w->d_weights, w->h_weights,
               TOTAL_WEIGHTS * sizeof(float), cudaMemcpyHostToDevice);

    // Section 15.4: Initialize genomes, deltas, metadata.
    // Genome seeds drawn from host PCG32 — no index-based formulas.
    for (int i = 0; i < TOTAL_ORG; ++i) {
        genome::Genome& g = w->org_table.genomes[i];
        std::memset(&g, 0, sizeof(g));
        genome::write_role(g, Role::Classifier);

        // Draw 32-bit seed from PCG32.
        uint32_t seed_val = pcg32_random(&w->rng);
        g.bits[0] = (g.bits[0] & 0x3u) | (seed_val << 2);
        g.bits[1] = (g.bits[1] & ~0x3u) | ((seed_val >> 30) & 0x3u);

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

    // Section 15.7: Assemble first classifier batch with MAIN_SOT_DENSITY.
    cur::assemble_classifier_batch(&w->classifier_batch,
                                   MAIN_SOT_DENSITY,
                                   w->host_sot_key, &w->rng);

    std::printf("World initialized: %d pool organisms, %d total, %d weights\n",
                POOL_SIZE, TOTAL_ORG, TOTAL_WEIGHTS);
    std::fflush(stdout);
}

// ---- Scoring (section 6) ---------------------------------------------------

// Cross-entropy classification loss + seed gradient computation.
// Uses first NUM_CLASSES dims of bmap_64 as logits (section 6).
static void score_classifiers(World* w) {
    for (int org = 0; org < POOL_SIZE; ++org) {
        if (w->org_table.role[org] != Role::Classifier) continue;
        int sample_idx = w->org_table.batch_sample_idx[org];
        int target = w->classifier_batch.label[sample_idx];

        const float* bmap = &w->h_descriptors[org * BMAP_DIM];

        float dlogits[NUM_CLASSES];
        float loss;
        autodiff::classifier_loss(bmap, target, NUM_CLASSES, dlogits, &loss);

        float task_accuracy = expf(-loss);
        w->org_table.f_raw[org] = task_accuracy * archive::sot_gate(w->org_table.f_sot[org]);

        // Before Wave 5: audit_mult = 1.0, variance_mult = 1.0.
        w->org_table.fitness[org] = archive::compose_fitness(
            w->org_table.f_raw[org], 1.0f, 1.0f, 1.0f);

        // Seed gradient: d(CE)/d(logits) in first NUM_CLASSES dims, 0 elsewhere.
        float* sg = &w->h_seed_grad[org * BMAP_DIM];
        for (int d = 0; d < BMAP_DIM; ++d) sg[d] = 0.f;
        for (int d = 0; d < NUM_CLASSES; ++d) sg[d] = dlogits[d];
    }
}

// ---- Archive insertion (section 5, 9.1) ------------------------------------

static void insert_into_archive(World* w) {
    for (int org = 0; org < POOL_SIZE; ++org) {
        if (w->org_table.role[org] != Role::Classifier) continue;

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

    for (int spawn = 0; spawn < WAVE_SIZE; ++spawn) {
        // Before bootstrap, all spawns are classifiers.
        Role target_role = Role::Classifier;

        // Find the worst-fitness organism of the same role in the pool.
        int worst_idx = -1;
        float worst_fit = 1e30f;
        for (int i = 0; i < POOL_SIZE; ++i) {
            if (w->org_table.role[i] != target_role) continue;
            if (w->org_table.fitness[i] < worst_fit) {
                worst_fit = w->org_table.fitness[i];
                worst_idx = i;
            }
        }
        if (worst_idx < 0) continue;

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

        // Install child into the pool slot.
        w->org_table.genomes[worst_idx] = child_genome;
        w->org_table.role[worst_idx] = genome::read_role(child_genome);
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
    ctx.d_swap_org   = w->d_pt_swap_org;
    ctx.d_swap_ckpt  = w->d_pt_swap_ckpt;
    ctx.d_swap_grad  = w->d_pt_swap_grad;
    ctx.genomes      = w->org_table.genomes;
    ctx.deltas       = w->org_table.deltas;
    ctx.lineage_id   = w->org_table.lineage_id;
    ctx.parent_id    = w->org_table.parent_id;
    ctx.spawn_gen    = w->org_table.spawn_gen;
    ctx.fitness      = w->org_table.fitness;
    ctx.f_raw        = w->org_table.f_raw;
    ctx.f_sot        = w->org_table.f_sot;
    ctx.role         = w->org_table.role;
    ctx.stream       = w->stream;
    return ctx;
}

// ---- Step generation (section 5 data flow) ---------------------------------

// Phase progress trace: prints phase tag + checks CUDA errors after each sync.
// Always flushed so output is never lost to buffering.
static void phase_trace(const char* tag, int gen, cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::printf("  [CUDA ERROR] gen %d %s: %s\n", gen, tag, cudaGetErrorString(err));
        std::fflush(stdout);
    }
    std::printf("  gen %d: %s\n", gen, tag);
    std::fflush(stdout);
}

void step_generation(World* w) {
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
    cudaMemcpyAsync(w->d_fwd_inputs, w->h_fwd_inputs,
                    POOL_SIZE * sizeof(ForwardInputs),
                    cudaMemcpyHostToDevice, w->stream);
    cudaMemcpyAsync(w->d_batch_image, w->classifier_batch.image,
                    cur::CLASSIFIER_BATCH * GRID_SIZE * GRID_SIZE * 3 * sizeof(__half),
                    cudaMemcpyHostToDevice, w->stream);
    cudaMemcpyAsync(w->d_batch_task_emb, w->classifier_batch.task_embedding,
                    TASK_EMBED_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream);
    phase_trace("T1_H2D", gen, w->stream);

    // ---- GPU: forward_with_checkpoints ----
    autodiff::launch_forward_with_checkpoints(
        w->d_organisms, w->d_fwd_inputs, nullptr,
        w->d_weights, w->d_checkpoints, POOL_SIZE, w->stream);
    phase_trace("forward", gen, w->stream);

    // ---- GPU: extract_descriptor ----
    nca::extract_descriptor(w->d_organisms, w->d_descriptors,
                            POOL_SIZE, w->stream);

    // ---- GPU: btraj_gather ----
    autodiff::launch_btraj_gather(w->d_organisms, w->d_btraj,
                                  POOL_SIZE, w->stream);
    phase_trace("descriptor+btraj", gen, w->stream);

    // ---- T2: D→H transfers ----
    cudaMemcpyAsync(w->h_descriptors, w->d_descriptors,
                    POOL_SIZE * BMAP_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, w->stream);
    cudaMemcpyAsync(w->h_btraj, w->d_btraj,
                    POOL_SIZE * BTRAJ_SAMPLES * BMAP_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, w->stream);
    phase_trace("T2_D2H", gen, w->stream);

    // Copy BTRAJ into IntentRegistry.
    for (int i = 0; i < POOL_SIZE; ++i) {
        std::memcpy(w->intent_registry.btraj[i],
                    &w->h_btraj[i * BTRAJ_SAMPLES * BMAP_DIM],
                    BTRAJ_SAMPLES * BMAP_DIM * sizeof(float));
    }

    // ---- SOT identity check (section 12, pre-allocated buffers) ----
    safety::alignment::apply_sot_identity(
        w->d_organisms, w->d_weights,
        w->classifier_batch, w->host_sot_key,
        w->h_descriptors, w->org_table.batch_sample_idx,
        w->org_table.f_sot,
        w->d_sot_temp_images, w->d_sot_task_emb,
        w->d_sot_fwd_inputs, w->d_sot_descriptors,
        w->stream);
    phase_trace("SOT", gen, w->stream);

    // ---- Score organisms (section 6) ----
    score_classifiers(w);

    // ---- Archive insertion (section 9.1) ----
    insert_into_archive(w);

    // ---- T3: H→D seed_grad ----
    cudaMemcpyAsync(w->d_seed_grad, w->h_seed_grad,
                    POOL_SIZE * BMAP_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, w->stream);

    // ---- PT swaps BEFORE backward (section 13, section 5) ----
    safety::pt::record_best_fitness(&w->mutation_ladder,
                                    w->org_table.fitness);
    if (gen > 0 && gen % PT_SWAP_INTERVAL == 0) {
        safety::pt::SwapContext ctx = make_swap_context(w);
        safety::pt::propose_swaps(&w->mutation_ladder,
                                  w->org_table.fitness,
                                  &w->rng, ctx);
    }
    phase_trace("score+archive+PT", gen, w->stream);

    // ---- GPU: backward (sequential cooperative per-organism) ----
    autodiff::launch_backward_all(
        w->d_organisms, w->d_weights, w->d_seed_grad,
        w->d_checkpoints, w->d_grads,
        w->bwd_workspace, POOL_SIZE, w->stream);
    phase_trace("backward", gen, w->stream);

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
    phase_trace("optimizer", gen, w->stream);

    // ---- Gradient health monitoring (section 8) ----
    // Read single float from device (not full weight vector).
    float h_grad_norm_sq = 0.f;
    cudaMemcpy(&h_grad_norm_sq, w->d_grad_norm, sizeof(float),
               cudaMemcpyDeviceToHost);
    float grad_norm = sqrtf(h_grad_norm_sq);
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

    // ---- Spawn wave ----
    spawn_wave(w);

    // ---- CUSUM update (placeholder surprise = 0 until Wave 3) ----
    safety::cusum_update(&w->cusum_surprise, 0.f);

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

    // ---- Logging (section 15.8): every TELEMETRY_INTERVAL AND first 5 gens ----
    if (w->generation % TELEMETRY_INTERVAL == 0 || w->generation < 5) {
        float sum_fit = 0.f, sum_raw = 0.f;
        int count = 0;
        for (int i = 0; i < POOL_SIZE; ++i) {
            if (w->org_table.role[i] == Role::Classifier) {
                sum_fit += w->org_table.fitness[i];
                sum_raw += w->org_table.f_raw[i];
                count++;
            }
        }
        float mean_fit = count > 0 ? sum_fit / count : 0.f;
        float mean_raw = count > 0 ? sum_raw / count : 0.f;
        int asize = archive::archive_size(w->archive);
        std::printf("gen %4d  mean_fitness=%.4f  mean_f_raw=%.4f  archive=%d  grad_norm=%.2e\n",
                    w->generation, mean_fit, mean_raw, asize, grad_norm);
        std::fflush(stdout);
    }

    w->generation++;
}

// ---- Run -------------------------------------------------------------------

void run(int n_generations) {
    World* w = new World;
    initialize_world(w);

    for (int g = 0; g < n_generations; ++g) {
        if (safety::alignment::poll_off_switch()) {
            std::printf("Shutdown flag detected at generation %d\n", g);
            std::fflush(stdout);
            break;
        }
        step_generation(w);
    }

    // Final report.
    int asize = archive::archive_size(w->archive);
    std::printf("\n=== Run complete: %d generations, archive size = %d ===\n",
                w->generation, asize);

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

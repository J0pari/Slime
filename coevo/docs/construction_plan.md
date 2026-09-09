# Construction Plan

Derived from `docs/blueprint.md` (behavioral spec) and
`docs/cuda_engineering.md` (GPU engineering spec).

Each wave produces a compiling, running binary that does something real
and testable. No wave leaves stubs, empty bodies, or declared-only functions.
If a function isn't ready, it doesn't exist yet.

---

## Wave 1 — Autodiff + CAME (rewrite)

**Scope**: Replace the deleted `autodiff/warp_tape.cu` and `optimizer/came.cu`
with correct implementations per `cuda_engineering.md` sections 4.1–4.3, 4.7,
4.8, 8, 10.

**Files created**:
- `autodiff/warp_tape.cu`
- `optimizer/came.cu`

**What warp_tape.cu contains**:
- Structs: `CheckpointBuffer`, `GradBuffers`, `BackwardWorkspace` (two FP32
  d_state buffers, one d_perc buffer, two FP16 re-forward buffers — per
  `cuda_engineering.md` section 10).
- `forward_with_checkpoints` kernel — identical logic to the proven
  `forward_kernel` in `nca/engine.cu` but with checkpoint saves at steps
  0, 16, 32, 48. Grid: `<<<n_organisms, dim3(16,16)>>>`.
- `backward_kernel` — Wave 1 initial implementation: processes ONE organism
  per launch using cooperative groups. **Replaced in Wave 2.5** by
  phase-decomposed batched sub-kernels (see Wave 2.5). The algorithmic
  structure (checkpointed reverse-mode AD, two-phase stencil gather,
  re-forward from checkpoints) is preserved; only the parallelization
  strategy changes.
- `backward_project_bmap` — device function. Backpropagates from d_bmap
  through avg-pool + W_bmap projection to d_state and dW_bmap.
- `btraj_gather_kernel` — gathers strided bmap_traj from OrganismState[]
  into contiguous d_btraj buffer. Grid: `<<<POOL_SIZE, 128>>>`.
- Loss functions: `classifier_loss`, `predictor_mse_loss` (existing code,
  proven in host tests — moved here).
- Host API: `allocate_backward_workspace`, `free_backward_workspace`,
  `allocate_checkpoints`, `free_checkpoints`, `allocate_grad_buffers`,
  `free_grad_buffers`, `launch_forward_with_checkpoints`,
  `launch_backward_all` (loops over organisms, launching one cooperative
  backward_kernel per org with `cudaStreamSynchronize` between each).

**What came.cu contains**:
- Structs: `CameHyperparams`, `CameState` (m, v, c, prev_u device pointers
  + size + step).
- `aggregate_gradients_kernel` — averages per-organism GradBuffers into
  a flat d_mean_grad buffer. Grid: `<<<ceil(TOTAL_W/256), 256>>>`.
- `came_step_kernel` — one CAME update per weight. Reads d_mean_grad,
  updates d_weights and CAME state (m, v, c, prev_u). Grid:
  `<<<ceil(TOTAL_W/256), 256>>>`.
- Host API: `allocate_came`, `free_came`, `launch_aggregate_gradients`,
  `launch_came_step`.

**Acceptance test** (`make wave1-test`):
1. Allocate 1 organism, weights, checkpoints, grad buffers, backward
   workspace, CAME state — all via `cudaMalloc`.
2. Forward with checkpoints → verify BTRAJ matches `forward_kernel` output
   (bitwise on bmap_64, since same computation).
3. Set seed_grad to a known pattern. Run backward. Verify all 4 gradient
   buffers (dW_perc, dW_inter, dW_flow, dW_bmap) are finite and nonzero.
4. Verify the stencil adjoint produces nonzero d_state contributions
   (not just straight-through zeros).
5. Run 5 iterations of forward → backward → aggregate → CAME. Verify
   loss decreases monotonically.
6. Test with POOL_SIZE=4 organisms to verify gradient aggregation averages
   correctly.

---

## Wave 2 — Integration Layer + Classifier-Only Loop

**Scope**: Replace the deleted `integration/main_loop.cu`,
`integration/host_main.cu`, and `safety/alignment.cu` with correct
implementations per `cuda_engineering.md` sections 2, 3, 5, 6, 9, 9.1,
12, 13, 15.

**Files created**:
- `integration/main_loop.cu` — World struct, OrganismTable, IntentRegistry.
- `integration/host_main.cu` — `initialize_world`, `step_generation`, `run`, `main`.
- `safety/alignment.cu` — `apply_sot_identity`, `poll_off_switch`,
  `apply_operator_command` (host functions with real bodies).

**Files modified**:
- `config/constants.cuh` — add NUM_CLASSES, MAIN_SOT_DENSITY,
  TELEMETRY_INTERVAL, GRAD_HEALTH_WINDOW, PCG32 struct + functions
  (`__host__ __device__`, rotate via `(32u - rot) & 31u`), default seed.
- `archive/soft_qd_archive.cu` — add per-role live index lists
  (alive_classifier_idx, alive_predictor_idx, n_alive_classifier,
  n_alive_predictor), assign_bin function, PCA storage fields
  (pc, pc_min, pc_max, pc_mean, pca_valid) per section 9.1.
  Update insert/recompute_bins to maintain live lists and PCA state.
  Strip `__device__` from all archive functions — archive is host-only.
- `genome/codec.cu` — migrate `mutate` and `crossover` to `Pcg32*`.
  Migrate `init_delta_from_prior` to seed a local `Pcg32` from genome
  seed. Delete `xorshift32` and `rand_uniform`. Strip `__device__` from
  all genome functions except `apply_delta`.
- `curriculum/problem_generator.cu` — migrate `assemble_classifier_batch`
  to `Pcg32*`. Labels drawn from `[0, NUM_CLASSES)`.
- `safety/parallel_tempering.cu` — implement `record_best_fitness` and
  `propose_swaps` with full organism data swap (device memcpy of
  OrganismState, CheckpointBuffer, GradBuffers + host memcpy of
  OrganismTable rows). Signature: `propose_swaps(MutationLadder*,
  const float*, Pcg32*, SwapContext&)`.
- `optimizer/came.cu` — add `grad_norm_reduce_kernel` for device-side
  gradient L2 norm computation.

**What main_loop.cu contains**:
- `IntentRegistry` struct.
- `OrganismTable` struct (with genome stored per-organism).
- `TOTAL_WEIGHTS` constant (= `TOTAL_WEIGHT_SLOTS` from `genome/codec.cu`,
  2587). Weight offset constants reuse those from codec.cu.
- PCG32 state struct (uint64_t state, uint64_t inc).
- `World` struct with all fields from `cuda_engineering.md` section 2:
  device pointers (d_organisms, d_weights, d_fwd_inputs, d_checkpoints,
  d_grads, d_bwd_d_state[2], d_bwd_d_perc, d_bwd_recomp[2], d_mean_grad,
  d_came_{m,v,c,prev_u}, d_descriptors, d_seed_grad, d_batch_image,
  d_batch_task_emb, d_btraj, d_grad_norm, d_sot_temp_images,
  d_sot_task_emb, d_sot_fwd_inputs, d_sot_descriptors, d_pt_swap_org,
  d_pt_swap_ckpt, d_pt_swap_grad), pinned host buffers (h_descriptors,
  h_btraj, h_seed_grad, h_fwd_inputs, h_weights), and host-only state
  (archive, safety structs, PCG32 rng, scalars).
- Function declarations: `initialize_world`, `step_generation`, `run`.

**What host_main.cu contains**:
- `alloc_gpu_buffers` / `free_gpu_buffers` — allocates EVERY device buffer
  via `cudaMalloc` and every host buffer via `cudaMallocHost` (pinned),
  per `cuda_engineering.md` section 2. No `cudaMallocManaged`. No pointer
  is left null. Includes SOT reference buffers and PT swap temp buffers.
- `initialize_world` — calls alloc, then per section 15:
  - PCG32 RNG seeded per section 15.1.
  - Kaiming He weight init per section 15.2.
  - Genome seeds drawn from PCG32 per section 15.4.
  - RFF init from PCG32.
  - PT ladder state per section 15.5.
  - CUSUM provisional params per section 15.6.
  - SOT key per section 15.3.
  - Main-pool SOT density per section 15.7 (MAIN_SOT_DENSITY, NOT
    STRESS_SOT_DENSITIES[0]).
  - Archive bin caps and pca_valid = false per section 9.1.
  - First classifier batch assembled.
  Every field of World is explicitly initialized.
- `step_generation` — implements the exact data flow from
  `cuda_engineering.md` section 5:
  - Curriculum refresh (using MAIN_SOT_DENSITY).
  - Bootstrap check.
  - Organism-to-batch assignment: round-robin per A-401.
  - ForwardInputs setup on host, T1 transfer (H→D).
  - `launch_forward_with_checkpoints`.
  - `extract_descriptor`.
  - `btraj_gather` + T2 transfer (D→H) + sync.
  - Copy h_btraj to IntentRegistry.
  - SOT identity check (using pre-allocated buffers).
  - Score: cross-entropy loss using NUM_CLASSES per section 6.
  - Compute h_seed_grad from actual loss gradient per section 6.
  - Archive insertion with assign_bin per section 9.1.
  - T3 transfer (H→D) of h_seed_grad.
  - PT swaps (if interval, BEFORE backward).
  - `launch_backward_all` (sequential cooperative per-organism backward;
    replaced by batched sub-kernels in Wave 2.5).
  - `launch_aggregate_gradients`.
  - `grad_norm_reduce` (device-side, 1 float D→H).
  - `launch_came_step`.
  - Sync + gradient health check.
  - Spawn wave from archive live index list per section 9.
  - CUSUM updates.
  - `record_best_fitness`.
  - Telemetry at TELEMETRY_INTERVAL.
  - Periodic: PCA rebin.
  - Operator checks.
- `run` — allocate World, initialize, run N generations, report, cleanup.
- `main` — call run.

**What alignment.cu contains**:
- `apply_sot_identity` — computes f_sot as cosine similarity between
  organism's bmap_64 and reference bmap_64 from the un-permuted SOT image.
  Runs forward on the un-permuted image to get reference, compares.
  Uses pre-allocated device buffers from World (no per-call cudaMalloc).
- `poll_off_switch` — checks for `shutdown.flag` file existence.
- `apply_operator_command` — reads `operator_cmd.txt`, parses prune/pause/
  resume/checkpoint commands.

**What ArchiveEntry gains**: `Genome genome` field (already added to
`archive/soft_qd_archive.cu` per section 9).

**Acceptance test** (`make run`):
1. `./build/coevo` runs 100 generations.
2. Archive size > 0 at end.
3. Multiple PCA bins occupied after recompute_bins (hash fallback distributes
   entries across bins from generation 0).
4. Loss printed per generation shows decrease over the run.
5. Organisms produce distinct bmap_64 descriptors (verified by archive
   having entries in different bins).
6. No CUDA errors reported.
7. Gradient norm reported via device-side reduction (no full D→H copy).
8. SOT density uses MAIN_SOT_DENSITY (0.05), not stress densities.

---

## Wave 2.5 — Batched Backward (Performance-Critical Rewrite)

**Scope**: Replace the sequential cooperative-kernel backward pass with a
phase-decomposed batched approach. The Wave 1 backward launched one
cooperative kernel per organism (64 sequential launches with
`cudaStreamSynchronize` between each), resulting in ~30-60 seconds per
generation. The batched approach decomposes the backward into ~675 small
kernels, each processing all organisms in parallel via
`<<<POOL_SIZE, 256>>>`. Per `cuda_engineering.md` section 4.2 (revised).

**Motivation**: The sequential backward made the system unusably slow.
64 cooperative kernel launches with host-side sync after each defeats
GPU parallelism. Each organism's backward was using 16 blocks on 30 SMs
(~53% occupancy) while 63 other organisms waited.

**Files modified**:
- `autodiff/warp_tape.cu` — delete `backward_kernel` (single cooperative
  kernel). Replace with 6 sub-kernels: `bwd_zero_kernel`,
  `bwd_seed_avgpool_kernel`, `bwd_seed_scatter_kernel`,
  `bwd_load_checkpoint_kernel`, `bwd_reforward_step_kernel`,
  `bwd_weight_grad_kernel`, `bwd_stencil_gather_kernel`. Rewrite
  `launch_backward_all` to call the sub-kernel sequence. Update
  `BackwardWorkspace` to hold per-organism buffers.
- `integration/host_main.cu` — update `allocate_gpu_buffers` to allocate
  per-organism backward workspace (96 MB total). Update
  `free_gpu_buffers` to match.

**Key design decisions**:
- Per-organism private workspace buffers (d_state, d_perc, recomp) so
  no organism reads/writes another's data. Eliminates all need for
  cooperative groups and grid-wide sync.
- Each sub-kernel uses `<<<POOL_SIZE, 256>>>` — one block per organism,
  256 threads per block. Intra-block sync via `__syncthreads()`.
- Host loop manages the segment/local_step iteration and pointer swaps.
- atomicAdd for weight gradient accumulation is intra-block only (no
  inter-block contention).
- No cooperative groups, no `cudaLaunchCooperativeKernel`, no grid.sync().

**VRAM impact**: +96 MB (1.5 MB per organism × 64). Total system VRAM
goes from ~61 MB to ~157 MB. Within the 6 GB budget.

**Prerequisites**: Wave 2 complete (main loop running).

**Acceptance test**:
1. `make all` compiles with zero warnings.
2. `make check` passes all host tests (4181).
3. `make forward-smoke` passes.
4. 10-generation run completes in under 60 seconds (vs. ~5-10 minutes
   with sequential backward).
5. Gradient norm and archive growth match Wave 2 qualitative behavior
   (archive grows, grad norm decreases over generations).
6. No CUDA errors in phase trace output.

---

## Wave 3 — Placeholder Regressor + Probe Set

**Scope**: Implement `placeholder_forward`, `launch_placeholder_train`,
probe set evaluation, CUSUM on placeholder surprise. Per
`cuda_engineering.md` sections 4.5, 4.6 and `blueprint.md` A-601.

**Files modified**:
- `predictor/hybrid_surprise.cu` — replace DECLARED ONLY stubs with
  real kernel bodies for `placeholder_forward` and `launch_placeholder_train`.
- `curriculum/problem_generator.cu` — add probe set initialization
  and host-side probe evaluation.
- `integration/host_main.cu` — wire placeholder training into the
  generation loop (replay buffer push + train launch), compute
  s_placeholder from probe prediction error, feed into CUSUM.

**Acceptance test** (`make wave3-test`):
1. Placeholder regressor trains on synthetic data, MSE decreases.
2. Inject a sudden fitness shift; CUSUM detects within 50 generations.
3. Probe set signature verification passes.
4. s_placeholder is nonzero and varies across generations.

---

## Wave 4 — Predictor Role Activation

**Scope**: Wire BTRAJ to IntentRegistry, implement predictor batch assembly,
bootstrap trigger + founder spawn, predictor forward/loss/training through
the existing backward+CAME pipeline, ensemble surprise, hybrid blending,
role-balance fitness scaling, CUSUM calibration. Per `blueprint.md` A-601,
A-701, A-401.

**Files modified**:
- `curriculum/problem_generator.cu` — implement `assemble_predictor_batch`.
- `integration/host_main.cu` — wire role-switched scoring (classifiers use
  CE loss, predictors use MSE loss), BTRAJ→IntentRegistry copy, bootstrap
  trigger, founder spawn, ensemble surprise, hybrid blending, role-balance
  scaling in archive insertion. Implement CUSUM calibration: during the
  s_target calibration window (generations 200-700 after bootstrap),
  collect surprise and r statistics. On window close, compute σ, set
  k = 0.5σ, h = 5σ (and k_r, h_r for r CUSUM), reset accumulators.
- `predictor/hybrid_surprise.cu` — no new stubs; ensemble_surprise and
  blend_surprise already have bodies.

**Acceptance test** (`make wave4-test`):
1. Run 2000+ generations.
2. Archive fills to 50%, bootstrap fires.
3. Predictor founders spawn (16 organisms flip to predictor role).
4. Predictors reproduce and their MSE loss decreases.
5. Pearson r climbs above 0.5.
6. Role-balance scaling shifts archive composition when synthetic
   surprise is injected.
7. BTRAJ values at steps 16/32/48/64 match reference forward.
8. Role mutation rate over 10000 spawns is within [5e-5, 5e-4].
9. CUSUM calibration fires after generation 700, parameters update from σ.

---

## Wave 5 — Structural Pressures + PT Launchers

**Scope**: Replace all remaining DECLARED ONLY stubs across safety files
with real implementations. Activate audit_mult, variance_mult, and
lineage brake. Per `blueprint.md` S-003, S-004, A-401, Q-001 and
`cuda_engineering.md` section 13.

**Files modified**:
- `safety/structural.cu` — implement: `run_audit_cycle` (least-squares
  fit on bmap→fitness, compute R², emit audit_mult), `refresh_probe_panel`
  (train 4 linear probes, report accuracies), `launch_sentinel_score`
  (score all organisms through ensemble), `update_lineage_stats`
  (per-role lineage share, runaway detection), `compute_variance_mult`
  (variance floor on bmap_64, returns per-organism multiplier).
- `safety/parallel_tempering.cu` — implement:
  `refresh_stress_slots` (copy from main pool), `evaluate_stress`
  (forward on stress organisms at elevated SOT), `flag_stress_failures`
  (scan failure rates). (`record_best_fitness` and `propose_swaps` are
  implemented in Wave 2 since they depend only on fitness arrays.)
- `integration/host_main.cu` — wire all structural pressures into the
  generation loop at their specified intervals. Replace the hardcoded
  `audit_mult = 1.0` and `variance_mult = 1.0` with live values from
  run_audit_cycle and compute_variance_mult. Wire lineage brake into
  archive insertion (call apply_lineage_brake when lineage-share stats
  indicate a runaway).
- `archive/soft_qd_archive.cu` — wire apply_lineage_brake calls from
  lineage stats into insert path.

**Acceptance test** (`make wave5-test`):
1. Audit cycle runs and produces audit_mult in [0.9, 1.0].
2. L_role accuracy is high (>0.8) after bootstrap.
3. Sentinel scoring produces finite scores for all organisms.
4. Lineage stats correctly count per-role archive shares.
5. PT swaps achieve ~25% acceptance rate.
6. Stress ladder identifies SOT-fragile lineages (injected synthetic).
7. variance_mult < 1.0 for organisms with degenerate (near-constant) bmap.
8. Lineage brake activates when synthetic runaway lineage exceeds threshold.

---

## Wave 6 — Red-Team Tests + Phase Graphs

**Scope**: Implement all Q-001 attack class tests (A–F) and captured-graph
execution. Per `blueprint.md` Q-001 and `cuda_engineering.md` section 11.

**Files created**:
- `tests/qa_red_team.cu` — all 6 attack classes with real injection,
  detection verification, and timing assertions.
- `execution/phase_graphs.cu` — CUDA graph capture/replay for the 6
  capturable phases.

**Acceptance test**:
1. All attack classes A–F detected within T_detect_max.
2. Captured-graph execution produces bitwise identical outputs to
   sequential execution over 100 generations.

---

## Wave 7 — Checkpoint + Long-Run Hardening

**Scope**: Full checkpoint serialization, long-run stability tests,
dashboard telemetry. Per `blueprint.md` S-001 and
`cuda_engineering.md` section 14.

**Files modified**:
- `safety/monitoring.cu` — implement `write_checkpoint`, `load_checkpoint`
  with schema hash verification.
- `integration/host_main.cu` — wire checkpoint at intervals and on
  operator command. Load checkpoint at startup if present.

**Acceptance test**:
1. Checkpoint round-trip preserves all state (byte-identical re-run
   from checkpoint for 10 generations).
2. 5000+ generation run with all subsystems: r > 0.5 sustained,
   predictor sub-population alive, no spontaneous Class A–F conditions.
3. Role fraction, r, rho, swap stats, stress failure rates printed
   to telemetry log.

---

## Wave 8 — Global Context Channel

**Scope**: Implement the global context broadcast (A-203) per
`cuda_engineering.md` section 16. Add W_ctx weight layer, modify the forward
kernel to broadcast summary into channels 14–15 at bmap sample steps, modify
the backward kernel with the context broadcast adjoint.

**Prerequisite**: Waves 1–7 complete. Baseline performance data collected from
a Wave 7 long run to establish the A/B comparison baseline.

**Files modified**:
- `config/constants.cuh` — set `GLOBAL_CONTEXT_ENABLED = true`, add
  `W_CTX_SIZE`.
- `autodiff/warp_tape.cu` — add `OFF_CTX`, update `TOTAL_WEIGHTS` (conditioned
  on `GLOBAL_CONTEXT_ENABLED`). Add W_ctx to `GradBuffers`. Add context
  broadcast in `forward_with_checkpoints` at bmap steps. Add context adjoint
  in `backward_kernel`.
- `nca/engine.cu` — add context broadcast in `forward_kernel` (non-checkpointed
  path) for SOT reference forward passes.
- `integration/host_main.cu` — add W_ctx to Kaiming He init table. No other
  changes (TOTAL_WEIGHTS governs all buffer sizes automatically).

**Acceptance test** (`make wave8-test`):
1. Forward with `GLOBAL_CONTEXT_ENABLED = true` produces non-zero values in
   channels 14–15 after step 16 (for classifiers, which start at zero).
2. Backward produces non-zero dW_ctx gradients.
3. 100-generation A/B comparison: context-enabled run vs. context-disabled run
   produce different archive descriptors (the channel is not inert).
4. 5000-generation run: task accuracy matches or exceeds the Wave 7 baseline.
   If accuracy regresses, the context channel is adding noise rather than
   signal — this is a valid negative result that informs architecture decisions.
5. BTRAJ bitwise match between `forward_kernel` and
   `forward_with_checkpoints` (both with context enabled).

---

## Rules

- Each wave's acceptance test must pass before starting the next wave.
- No function is declared without a body.
- No comments containing "for now", "simpler", "placeholder" (except the
  PlaceholderRegressor which is the spec's name for it), "temporary",
  "stub", "later", "let's", "we need", "we could", "we use",
  "approximation", "good enough", "hack", or "deferred".
- Every kernel launch has its grid/block dims specified in the engineering
  spec. Use those dims.
- Every device buffer is allocated via `cudaMalloc`. Every pinned host
  buffer via `cudaMallocHost`. No `cudaMallocManaged` in the generation
  loop. No null pointers at any point after initialization.
- The backward kernel uses `<<<16, 256>>>` with `cudaLaunchCooperativeKernel`
  and the gather-based stencil adjoint. No scatter+atomicAdd on d_state.
  No single-block backward.
- Scoring uses cross-entropy loss, not variance proxies.
- CAME is the spec's algorithm, not an "approximation" of it.
- Spawn copies genomes from archive entries, not from pool[idx % POOL_SIZE].
- All host↔device transfers use explicit `cudaMemcpyAsync` at the 4
  defined transfer points (T1–T4) per `cuda_engineering.md` section 3.

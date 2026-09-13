# Architecture Signals

Non-normative observations and friction notes. This document records things
worth remembering about the architecture-control process; it never changes
behavior. The three binding specifications remain blueprint.md,
cuda_engineering.md, and construction_plan.md.

## Signals

- 2026-09-12: Fallback/default audit opened (type-theory pass, inspired by
  KanForge's sum-type discipline and training-architecture's AST scanners).
  Closed with gates and negative tests: silent switch defaults
  (`gate_enum_no_silent_default`; the one offense removed), masked CUDA
  results (`gate_no_masked_cuda_errors`; seven offenses, all introduced this
  session, fixed), stored-but-unchecked `cudaError_t` variables
  (`gate_no_unchecked_error_vars`; scan found none), value-to-literal ternary
  collapse `x ? x : "literal"` (`gate_no_value_ternary_string_default`; one
  offense in the startup banner fixed), and unreadable evidence manifests
  (`load_manifests` now raises instead of silently dropping a manifest, and a
  missing `result` is refused; negative tests in test_architecture.py).
  Environment reads were scanned clean: all three `getenv` uses are presence
  gates for COEVO_* diagnostics, no literal substitution. The registry
  parsers now refuse missing structural keys (`_require_keys`: build_status
  `items`, documents, transactions + organism_buffers, bridge
  fingerprint/gate/contract, experiments) so a truncated registry can no
  longer validate an empty surface and pass. The null-early-return class was
  scanned: eight `if (!x) return ...` guards exist and every one propagates
  failure (`return false`/`nullptr`) or expresses a semantic no-change
  (`soft_qd_archive.cu` insertion when nothing changed); no silent error
  swallow was found. No gate was added for it: a line-local rule cannot
  distinguish a legitimate guard from a swallow, so the honest state is
  scanned-clean, not enforced. The nested-field remainder was inspected: the
  load-bearing defaults refuse absence rather than substitute (a missing
  build-item `status` fails BUILD_STATES, a missing `mechanisms` list fails
  the names-no-mechanisms check, and `gpu_client.scheduler_root` raises on an
  unset or empty root); the remaining `.get(key, default)` calls are
  display-only rendering, which is the deliberate presentation allowance.
  Still open: interchangeable runtime id types (class 6, no strong typedefs
  yet) — the only class without either a gate or an inspection result. Until
  it has a failing witness of its own, the audit is incomplete; do not
  describe it as closed.
- 2026-09-12: I8 backward work after the combined stress pass. Two measured
  wins: (1) the weight-grad kernel's per-cell dW_perc global atomics (27
  addresses hammered by 4096 cells per organism) became per-thread register
  partials with one deterministic tree reduction per block (main kernel
  3.89 -> 3.40 s); (2) ptxas showed the same kernel at 128 registers with
  3012/3116 bytes of spill stores/loads per thread, so perc/hidden/d_state/
  d_pre_hidden now stage at their last-use points and d_hidden is fused into
  d_pre_hidden — spills drop to 324/328 bytes and the main kernel to 2.38 s.
  Current per-generation budget (BPROFILE, sync-instrumented): backward
  5.58 s (reforward 0.75, weight-grad 3.60 [main 2.38, reduce_inter 0.93,
  reduce_flow 0.29], stencil 1.15, rd 0.08), stress 0.84, forward 0.70,
  SOT ~0.5. Target is 6 s; the remaining candidates are the reduce_inter
  dpre re-reads (48 p-blocks per organism each read all 32 dpre rows) and
  the stencil gather. evolution_regression 36/36 after both changes.
- 2026-09-12: I8 combined stress pass landed: `evaluate_stress_all` replaces
  the six per-sub-population/per-role replays with one reference launch and
  one stress launch over all 24 slots (staged images, task embeddings, and
  nominal/permuted targets in shared World buffers). Measured stress
  4.44 -> 0.84 s/generation (2-generation profile: 1676.64 ms / 2, 5.3x);
  total generation ~9.3 s before the backward work. Observable output is
  identical to the previous build on the same 3-generation run (every
  [STRESS]/[DASHBOARD]/checkpoint line matches). Checkpoint hashes are not a
  determinism witness here: the previous build run twice also produced
  different hashes (and a different `Occupied PCA bins` line), so fresh runs
  already drift run-to-run. The pinned PCG32 seed rules out ambient RNG (no
  entropy seeding exists), and the prime candidate is the documented float
  atomicAdd weight-grad accumulation (order varies in L2), whose low-bit
  differences cross archive thresholds. No claim asserts bit-reproducibility
  of the trajectory (G100.deterministic-seed is about the PCG32 draw
  sequence; A101.sot-schedule-independent is about the host-side schedule),
  so this is a reproducibility observation, not a violated contract; it does
  mean equivalence evidence between builds must compare observable output,
  not checkpoint bytes. The combined pass shares `d_sot_fwd_inputs`/`d_sot_descriptors`/`d_sot_bank_of`
  with the SOT path, so those scratch buffers are now allocated for
  STRESS_POOL_SIZE (24) slots; the 16-slot allocation was the cause of the
  first launch's `invalid argument` + illegal access. The backward reduce
  kernels gained shared-array padding (HIDDEN_DIM+1) to break bank conflicts
  with unchanged summation order (weight_grad 21.5 -> 5.1 s/generation).
- 2026-09-12: I8 profile correction. The "score+archive+PT" phase label
  included stress_cycle; splitting the trace shows score+archive+PT is
  1 ms/generation while stress is 12.9 s / 3 = 4.3 s/generation (15.5%).
  A debug-checks hypothesis was also tested and disproven: building the
  integration binary with SLIME_DEBUG_CHECKS=0 changed nothing measurable
  (13.4 vs 13.2 s), so the archive invariant checker stays enabled in
  release. Current phase budget per generation: backward ~22.3 s, stress
  ~4.3 s, forward ~0.7 s, SOT ~0.36 s, everything else negligible. Next I8
  targets in order: the backward weight-grad atomic accumulation (structural
  tiled rewrite; the shared-memory variant is measured and rejected), then
  stress internals (batch assembly vs the six small phase replays).
- 2026-09-12: I8 measured rejection (second of its kind). The backward
  weight-grad kernel issues ~2,075 global atomicAdds per cell (~35 billion
  per generation); a shared-memory accumulation with one flush per block was
  expected to help and measured 4.7x SLOWER (backward 65.5 -> 307.1 s for 3
  generations): 256 threads in a block serialize on the same shared
  addresses, while global atomics are aggregated by L2. Reverted with data,
  like the forward's shared-atomic experiment. The next attempt must remove
  the atomic accumulation structurally (e.g. a tiled scheme where each
  thread owns bank entries and accumulates over shared-staged per-cell
  perc/d_pre_hidden tiles in registers), which is a deliberate kernel
  rewrite rather than a loop change.
- 2026-09-12: I8 profile (3 generations, RTX 3060 Laptop, --profile). Before
  the cached-segment backward: backward 85.4 s (83.6%), score+archive+PT
  13.2 s (13.0%), forward+descriptor+btraj 2.2 s (2.1%), SOT 1.1 s, total
  102.1 s. After caching each segment's 16 states once instead of
  re-forwarding quadratically (480 -> 64 CA steps per generation): backward
  65.5 s, total 81.5 s. Target is 10 generations in 60 s (6 s/gen); current
  ~27 s/gen. Remaining hotspot: the per-step weight-grad and stencil-gather
  kernels, which round-trip the 48-wide d_perc buffer through global memory
  (~500 MB per call: 64 orgs x 4096 cells x 48 floats written, then read
  with a 9-neighbor stencil); fusing the two kernels (or staging d_perc per
  tile) is the next optimization, then launch-level work. The measured
  shared-memory-atomic variant remains rejected with data.
- 2026-09-12: Cross-repo bridge admission assessment (non-normative; no code,
  no claims, no inventory change). An external `adaptive-ecology/v1` handoff
  from the LLM-Trader side was evaluated against the binding extension gate
  in construction_plan.md. Gate state: CLOSED — I8 and I9 are `missing`, and
  the experimental program (E1-E5) has produced no evidence, so no extension
  axis may begin. Correct outcome: refusal with a precondition report, not
  bridge code.
  Coherence questions to resolve upstream before admission: (1) the join key
  is named `proposal_id` in sample_identity but `sample_id` in outcome_join
  and the consumer brief — either an alias must be declared or the field
  defined; (2) two literal feature names contain spaces
  (`sentiment_strongly bullish`, `sentiment_strongly bearish`) inside the
  fingerprinted ordered list; (3) the contract descriptor lives in Trader, so
  the content-recomputed fingerprint check is an admission precondition that
  cannot be satisfied from this repository; (4) the "three percentile-rank
  intermediates" are not enumerated, so the normalization-refusal check has
  no exact set.
  Narrowest future architecture if and when the gate opens: one host-side
  external task source that validates and packs the pinned contract into a
  Slime-owned typed form (no Trader code, no JSON in the CUDA core, no
  runtime fetch); the 15-d context vector consumed as global conditioning
  only, with no topology claim (v1 persists no spatial or temporal view);
  descriptor separation (z_state / y_task / d_behavior) as the prerequisite
  architectural item because bmap_64 is already overloaded; the raw external
  vector never becomes the archive descriptor; new roles only via input
  wiring and objective semantics on the shared substrate; external surprise
  stays telemetry under competence gating; frozen effective-phenotype
  sentinels as a host-owned, checkpoint-covered evaluation population for
  learner-drift versus environment-drift decomposition; checkpoint coverage
  for stream cursor, partition state, contract fingerprint, task-family
  identity, sentinel identities, and adaptation windows; PT registration for
  any organism-bound buffer; shadow-only, with the reverse contract
  deferred.
- 2026-09-12: I7 notes. Capture boundary rule: a phase graph contains only
  device launches/transfers with stable arguments, so the forward phase's
  three launches are captured together and the phase trace runs after the
  graph; the optimizer captures aggregate+came and the grad-norm reduction
  moved after it. Two graph-stability traps were found and fixed: the
  reference regressor's training step and the CAME step were kernel
  arguments (the CAME kernel turned out not to take one; the reference step
  is now device-resident). StressEval and the SOT reference are deferred:
  their device launches interleave host readbacks and host cosine
  computation, so capturing them requires restructuring the readbacks into
  stable host buffers first. The component formerly called the "placeholder"
  regressor is the reference regressor; the vocabulary was removed because a
  repository whose contract forbids stubs must not name a permanent
  component as if it were unfinished.
- 2026-09-12: I6 notes. (1) The old `rd_disabled` gate parsed single lines,
  so multi-line launcher calls evaded it; it was replaced by
  `rd_adjoint_present`, which triggers on the coefficients plumbing and
  requires the RD re-forward, the clamp-aware d_next workspace, and the RD
  gather in the backward. (2) The finite-difference witness initially failed
  on the saturated case because the loss was returned as float and the
  clamped value (~65504) makes the loss ~2e5, whose ulp swamps the FD
  signal; the loss accumulator is now double. Near the FP16 bound the FD is
  ill-conditioned (ulp > FD step), so the adversarial case saturates well
  past the bound. (3) The reaction encoding changed from centred quantization
  (zero bits = -1) to sign-magnitude (zero bits = 0), which invalidates any
  prior RD coefficient decode; the liveness test for the numeric exemption
  moved with `read_bits` into `nca/rd_codec.cuh`. (4) Stress slots re-decode
  their RD coefficients after the refresh copy, since their genomes change
  mid-generation.
- 2026-09-12: I5 backward adjoint design (complete, unexecuted). Two
  correctness traps found while deriving it, both fixed by storing the
  pre-broadcast summary:
  (1) The forward broadcast overwrites channels 14-15 after computing the
  summary, so the pre-broadcast channel values are unrecoverable from the
  re-forwarded state; the summary must be saved per organism per sample step
  (project_bmap gains a nullable summary-out, or the re-forward stores it
  from its CA output before broadcasting).
  (2) The seed backward's avgpool reads the final grid, which is
  post-broadcast, but the forward's bmap projection used the pre-broadcast
  summary. Recomputing the summary from the final grid would include the
  broadcast-written channels and bias the seed path; the seed scatter must
  consume the stored pre-broadcast summary instead. An earlier note here
  claiming step 64 had no downstream path was wrong for the same reason:
  the final broadcast changes the grid that the seed avgpool reads, so the
  step-64 adjoint exists through the summary, not through a later CA step.
  Remaining design: per replayed sample step the weight-grad kernel consumes
  dA channels 14-15 into d_ctx; adds dW_ctx[c*2+k] += s_t[c]*d_ctx[k]; adds
  sum_k W_ctx[c*2+k]*d_ctx[k] to the mean adjoint; zeroes dA channels 14-15
  before the CA adjoint. The host backward loop must pass the absolute step
  index (seg*CHECKPOINT_INTERVAL + local_step) to the re-forward and
  weight-grad kernels. This is the first implementation task of the next
  session; the gate stays disabled until the finite-difference witness
  exists.
- 2026-09-12: I5 forward broadcast landed; the backward context adjoint needs
  care about which sample steps have a downstream path. `project_bmap` runs
  at steps {16, 32, 48, 64} and the context write happens after the bmap
  projection at each. At step 64 the loop ends, so that broadcast cannot
  reach any later bmap sample: only the broadcasts at 16, 32, and 48 can
  influence the loss, and the adjoint must accumulate d_ctx where the
  backward replays those steps. The step-64 write still mutates the stored
  final grid (telemetry/audit see it), which is why the forward and the
  adjoint cannot share one naive per-sample treatment. The adjoint lands in
  the state backward's sample-step replay, not in bwd_seed_scatter_kernel.
- 2026-09-12: C1a fixed the predictor-target contract: `assemble_predictor_batch`
  now draws classifier-only targets, carries real lineage ids (the host passed
  pool indices), and propagates the target's SOT status; the typed witness
  `test_predictor_batch_contract` pins all of it. C1b chose generation-rotated
  temporal aggregation: the target slot rotates with the generation, a
  per-predictor loss EMA feeds fitness, and the EMA is organism-identity state
  (annotated, PT-swapped, and serialized). True per-generation K forwards were
  rejected for 8x forward cost; population-level aggregation for not giving an
  individual predictor the aggregate the spec describes.
- 2026-09-12: A semantic-coverage review distilled into the construction
  plan's semantic correctness stream: C1 predictor-target contract (classifier
  targets, real lineage ids, SOT status, K-target aggregation), C2 surprise
  population identity across spawn, C3 role-gradient PT identity, C4 genome
  fieldwise perturbation matrix, C5 checkpoint continuation equivalence, C6
  PT permutation test, C7 phase trace, C8 archive attribution semantics.
  C1-C3 gate further BUILD; C4-C7 are VERIFY witnesses; C8 is a blueprint
  decision. The host test build now carries -Werror, and the two
  signed/unsigned comparisons in the archive rebin are fixed.
- 2026-09-12: The stress ladder's classifier half is well-defined (elevated
  SOT-marked images at {10%, 20%, 40%} with per-organism effective-weight
  references), but the blueprint does not define what "fails the SOT gate"
  means for a predictor stress representative: predictors take a target
  descriptor, not an image, so the reversible image transformation does not
  apply. Resolved in blueprint S-003 with the target-permutation gate: the
  host draws a reversible permutation pi of the target bmap_32 dimensions
  from the SOT key, the nominal-target response is the reference, and the
  response to pi(target) must match it after pi^-1.
- 2026-09-12: The SOT identity check uses the 24 tail organism slots
  (`d_organisms + POOL_SIZE`) as reference-rollout scratch, but the blueprint
  assigns the same slots to the SOT-density stress sub-populations. The
  stress ladder (I4 remainder) therefore needs the conflict resolved first:
  either dedicated SOT reference buffers or an explicit time-multiplex
  ordering. Recorded before implementing so the next step starts with the
  decision, not with a collision.
- 2026-09-12: `cuda_engineering.md` was internally inconsistent about the
  reference regressor: §2.3 and §14 listed it host-only while §4.5-4.6,
  §5, and the VRAM budget specified device kernels. Reconciled in favor of
  the kernels (the construction plan's I3 requires them): parameters and
  AdamW state are device-resident, the host struct is a checkpoint/init
  mirror, the replay buffer stays host-only, and §3 gained T5 (minibatch
  upload, probe surprise readback). The now-dead host-side forward/train
  reference was removed rather than kept as a second implementation.
- 2026-09-12: Numeric policy adopted from the LLM-Trader pattern
  (`tests/test_no_magic_numbers.py`): the schema home is
  `config/constants.cuh`, the declared-constant registry is derived from it,
  exemptions are review decisions with reasons and a liveness test, and the
  gate is proven by planted violations. Slime's previous gate only caught
  `% <literal>`; the new `gate_numeric_policy` found 102 real findings
  (FP16 clamp bounds, epsilons, kernel-shape constants, hyperparameters all
  living outside the schema home). All are fixed; `config/constants.cuh`
  gained the canonical block, and four structural exemptions remain
  (bit-layout helpers, kernel-shape loops) each with a reason.
- 2026-09-12: Paranoid audit of the checkpoint/predictor work found and
  fixed: the atomic replace had a remove-then-rename window (now direct
  replace), the payload had no checksum (now FNV-1a 64), new run state was
  unserialized (surprise history, calibration samples, predictor error EMA,
  predictor batch, telemetry scalars — all now in the payload),
  `predictor_error_ema` did not move through PT (now swapped, annotated,
  and registry-checked), probe tuples were not held out from reference
  training (now flagged and skipped), `surprise_ratio` silently returned 1.0
  when uncalibrated (now an explicit gated state logged once), and the
  archive bin cap/occupants sizing and predictor EMA rates were bare
  literals (now named constants).
- 2026-09-12: Slime adopted the training-architecture GPU scheduler
  (`gpu-scheduler/v1`). `contracts/gpu-scheduler-pin.json` pins the contract
  fingerprint; `architecture/gpu_client.py` validates the schema major and
  fingerprint before every submission (loud refusal on drift), submits,
  waits, and reads the scheduler's result ledger;
  `architecture/progress_wrap.py` wraps scheduled commands and emits
  `progress/v1` envelopes for the daemon's estimator. Evidence manifests
  recorded with `--scheduler-job <id>` link claim evidence to the scheduler
  result ledger. All paths come from the environment (`TRAINING_ARCH_ROOT`,
  `SLIME_EVOLUTION_ROOT`, optional `KF_GPU_SCHED_DIR`); nothing is
  hardcoded. Slime is registered as a consumer in the owner's
  `src/integration.py` `_ENV_ROOTS` so the doctor observes the pin.
- 2026-09-12: The shared-atomic weight-gradient experiment was measured and
  rejected: replacing the global atomicAdds with per-block shared-memory
  accumulation ran the autodiff acceptance suite in 166-171s versus 34-40s for the global-atomic
  kernel (same-address shared-atomic replay serializes; L2 handles the
  global atomics at high throughput). The kernel was reverted; the finding
  means the backward bottleneck is launch structure/occupancy, not atomic
  traffic — profiling (Nsight) and CUDA graph capture are the next levers.
- 2026-09-12: The `checked_cuda_calls` allowlist is empty. Every production
  raw CUDA call is in a checked context: `CUDA_ABORT`/`TRANSFER_ABORT` for
  the generation loop, `CUDA_WARN` for shutdown frees, `cuda_diagnostics_ok`
  for the startup probe, and bool-returning allocation helpers
  (`allocate_checkpoints`/`allocate_grad_buffers`/`allocate_backward_workspace`/
  `allocate_came`). The launchers `launch_grad_norm_reduce`,
  `launch_telemetry_kernels`, `propose_swaps`, and `apply_sot_identity` now
  report failure and the run invalidates. `source_gates.py --strict` reports
  0 errors / 0 warnings.
- 2026-09-12: The task-conditioning defect is fixed with an explicit FIXED
  16→5 projection: TASK_PROJ (the first five rows of the 16-point DCT-II)
  folds every task embedding dimension into channels 6..10 at seeding, so no
  advertised dimension is inert. The failing witness now passes (perturbing
  task dim 12 shifts the descriptor by 2.5e-2). A learnable W_task bank
  remains a future architecture decision rather than a silent omission.
  With this, all 24 claims carry current passing evidence.
- 2026-09-12: Operator commands are durable now. Parsing lives in the pure
  `safety/operator_cmds.cuh`; the run loop owns `OperatorState` and applies
  pause gating (no generation steps while paused) and durable prunes
  (`archive::prune_lineage` tombstones entries with exact statistics under
  the invariant checker). Checkpoint reports unsupported honestly until the
  Wave 7 serialization exists. Three source gates joined the layer —
  `operator_polling`, `replay_before_spawn`, `schedule_host_only` — each
  with a planted-violation negative test. The remaining red claim is
  `A201.task-conditioning-complete`; `A201.shared-substrate` and
  `A401.archive-genotype-attribution` stay provisional until the predictor
  role exists (Wave 4) and their strong witnesses are writable.
- 2026-09-12: Gate 4 archive repairs landed. `recompute_bins` is now
  transactional: after reassigning bins it evicts QD-ranked occupants from
  any over-capacity (bin, role) and rebuilds every statistic from the
  survivors. `insert` replacement now displaces the candidate's nearest
  same-role same-bin neighbor by the weighted Euclidean metric (the A-401
  descriptor metric), updates the inverse-variance EMA on every mutation,
  and adjusts the role RFF mean exactly on eviction. PCA power iteration
  starts from dense deterministic vectors with a degenerate-covariance
  fallback. An invariant checker (counts, live lists, bin occupancies, caps,
  RFF means) runs after every insertion, replacement, and rebin under
  SLIME_DEBUG_CHECKS; a 1500-operation randomized property test plus a
  forced 14-into-one-bin collapse test exercise it. Three new/upgraded
  claims (A401.bin-capacity, A401.live-statistics-exact,
  A401.weighted-metric-active) are established with passing witnesses.
- 2026-09-12: Residual-magnitude telemetry (‖F_θ(x_t)‖, ‖x_t‖, per-cell
  ratio at steps 0/16/32/48/64) diagnosed the FP16 saturation: at step 0 the
  residual was already ~0.94× the state norm (per-cell max 1.36), so the
  unnormalized recurrence doubled the state per step and hit the clamp by
  step 16 (74 → 4.5e5 → 1.6e7). Fixed with the explicit residual timestep
  x_{t+1} = x_t + α·F_θ(x_t), α = 1/CA_STEPS, mirrored in the checkpoint
  re-forward and the weight-gradient adjoint (dF = α·d_state_next). The
  finite-difference suite now validates the adjoint at α=1 (tight per-bank
  assertions) and at production α (coarse; still catches a missing α
  scaling, ~64x mismatch). Fresh 5-generation run: state_max ~1.4, CE
  decreasing 2.998→2.713, fitness rising, zero saturation. The residual
  measurement kernel is gated to logging generations (it costs several
  forward passes of work).
- 2026-09-12: External review found the PT swap still left per-organism
  effective-weight banks behind: backward would reconstruct a moved
  trajectory with the stale slot's phenotype. Fixed by swapping the banks
  through a pre-allocated temp buffer; `test_pt_swap_backward_correspondence`
  now proves post-swap gradients equal the moved organism's pre-swap
  gradients, with a sensitivity check that the missing swap is detectable.
  This is the first defect the architecture layer missed because the
  transaction registry was complete only relative to itself — the fix adds
  code-side `[identity:organism]` / `[crosses:pt=...]` annotations checked
  against the registry in both directions.
- 2026-09-12: `A201.task-conditioning-complete` claims what the NCA always
  advertised: every task embedding dimension reaches the forward. The
  witness `tests/task_conditioning.cu` fails (task dims 5..15 are inert),
  so the generated status shows the claim red until the 16→5 projection (or
  another conditioning path) exists.
- 2026-09-12: Evidence manifests now name the witness that passed
  (`claim=witness:result`), and staleness includes a per-claim proposition
  hash. Old-format manifests were replaced. The status surface renders
  source-gate WARN states instead of collapsing them to PASS.
- 2026-09-11: The first claim registry embedded 21 claims into the specs.
  Four claims are intentionally red: `A401.bin-capacity`,
  `A401.live-statistics-exact`, `S002.operator-command-effective`, and
  `A101.sot-schedule-independent` have mechanisms but no strong witness yet,
  and their semantics are known to be incomplete (PCA rebin can exceed bin
  caps; operator pause/checkpoint flags are transient locals). The report
  surfaces them until witnesses exist.
- 2026-09-11: The source gate `checked_cuda_calls` ships with an explicit
  allowlist of wrapper functions whose raw CUDA calls are caught by a later
  stream synchronize or are shutdown-only. `--strict` fails on allowlisted
  hits so the list can only shrink.
- 2026-09-11: The repository gained git history (pushed to
  https://github.com/J0pari/Slime.git). Evidence manifests still rely on
  content hashes (claim/mechanism/witness/binary) rather than commit ids —
  deliberate: hashes also go stale for uncommitted edits.

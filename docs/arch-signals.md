# Architecture Signals

Non-normative observations and friction notes. This document records things
worth remembering about the architecture-control process; it never changes
behavior. The three binding specifications remain blueprint.md,
cuda_engineering.md, and construction_plan.md.

## Signals

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
  placeholder regressor: §2.3 and §14 listed it host-only while §4.5-4.6,
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
  and registry-checked), probe tuples were not held out from placeholder
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

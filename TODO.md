# Slime TODO

This is the action queue. It derives from the binding specifications and the
current implementation tracker; do not use it to redefine behavior. Update
[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) when an item is
completed or its verification state changes.

## P0 — Architecture-control layer

- [x] Embed typed `@claim` blocks in the binding specs (21 initial claims:
  G-100, A-101, A-103, A-201, A-202, A-301, A-401, A-501, S-001, S-002,
  S-004, I-001, C-001) and generate `docs/IMPLEMENTATION_STATUS.md` from the
  registry + evidence manifests.
- [x] Add `architecture/compiler.py` (registry, mechanism/witness resolution,
  bidirectional witness markers, dependency DAG, document ownership,
  capability drift, executable phase model, golden status).
- [x] Add `architecture/source_gates.py` (no ambient PRNG, no managed memory,
  checked CUDA calls, named tunables, RD-disabled, host authority) with its
  own negative tests in `tests/architecture/`.
- [x] Add `architecture/evidence.py` (slime-evidence/v1 manifests, content
  hashes, staleness) and record manifests for today's host/wave1/wave2/run5/
  architecture runs.
- [x] Add `AGENTS.md` (architecture delta protocol) and the concern registry
  (`architecture/documents.yaml`), machine contract (`machine.json`), and
  transaction/phase model (`transactions.yaml`).
- [x] Close the PT attribution bug: effective-weight banks now move with the
  organism through PT swaps (temp bank buffer + swap), the transaction
  registry gained code-side `[crosses:pt=...]` annotations checked in both
  directions, and `test_pt_swap_backward_correspondence` verifies post-swap
  gradients equal the moved organism's pre-swap gradients (with a
  sensitivity check that the missing swap is detectable).
- [x] Evidence is now witness-attributed (`claim=witness:result`) and
  staleness includes a per-claim proposition hash; the status surface renders
  source-gate WARN states.
- [x] Implement the explicit 16→5 task-embedding projection: a FIXED DCT-II
  mixing matrix (TASK_PROJ in config/constants.cuh) folds all 16 task
  dimensions into channels 6..10 at seeding; `A201.task-conditioning-complete`
  is established — the witness (perturb task dim 12 → output must change)
  now passes with a 2.5e-2 descriptor shift. A learnable W_task bank remains
  a future architecture decision.
- [x] Shrink the `checked_cuda_calls` allowlist to empty: every production
  raw CUDA call now routes through a checked wrapper (`CUDA_ABORT`,
  `TRANSFER_ABORT`, `CUDA_WARN`, `cuda_diagnostics_ok`) or a bool-returning
  allocation helper; launchers (`launch_grad_norm_reduce`,
  `launch_telemetry_kernels`, `propose_swaps`, `apply_sot_identity`) report
  failure and the run invalidates. `source_gates.py --strict` reports
  0 errors / 0 warnings.
- [x] Give the archive claims witnesses and repair their semantics:
  `A401.bin-capacity` (transactional rebin with QD-ranked eviction),
  `A401.live-statistics-exact` (invariant checker on every insert/rebin +
  1500-op randomized property test), and the new `A401.weighted-metric-active`
  (inverse-variance EMA updated on every insert/replacement; replacement
  displaces the nearest same-role same-bin neighbor by weighted distance).
  PCA power iteration now uses dense deterministic starting vectors with a
  degenerate-covariance fallback; RFF means adjust exactly on replacement.
- [x] Give the remaining red claims witnesses and make operator commands
  durable: `S002.operator-command-effective` (parse extracted to
  safety/operator_cmds.cuh; pause gates the run loop; prune zeroes pool
  members AND tombstones archive entries via archive::prune_lineage with
  exact statistics; checkpoint reports unsupported until Wave 7),
  `I001.replay-evaluation-identity` (replay-before-spawn source gate with a
  planted-violation test), `A101.sot-schedule-independent`
  (schedule-host-only source gate + batch determinism host test). All three
  established with passing evidence; `A201.task-conditioning-complete` now
  passes after the fixed DCT-II 16→5 projection (see below).
- [x] Implement the explicit 16→5 task-embedding projection: a FIXED DCT-II
  mixing matrix (TASK_PROJ in config/constants.cuh) folds all 16 task
  dimensions into channels 6..10 at seeding; `A201.task-conditioning-complete`
  is established — the witness (perturb task dim 12 → output must change)
  now passes with a 2.5e-2 descriptor shift. A learnable W_task bank remains
  a future architecture decision.

## P0 — Restore a trustworthy baseline

- [x] Rebuild the current source with the configured MSVC/CUDA environment and
  record tool versions, GPU, command, runtime, and outcome.
- [x] Run and record the Wave 1, forward-smoke, host-test, and 10-generation
  Wave 2.5 acceptance checks against that build. The Wave 2.5 timing gate
  failed; see `docs/IMPLEMENTATION_STATUS.md`.
- [x] Gates 1–3 repair pass: unique 16-slot spawn waves, pre-bootstrap role
  lock, full seed-gradient zeroing, transactional PT swaps (seed gradients +
  batch identity move with the organism; T3 after PT), replay/evaluated
  telemetry before spawning, fail-fast CUDA errors, per-organism effective
  weights (W_eff = W_shared + delta) in forward/backward/SOT, genotype
  causality tests, PT transaction tests, directional finite-difference
  gradient validation, production-CAME host tests, and per-bank numerical
  telemetry with nonfinite hard aborts.
- [ ] Reduce the current 10-generation integration run below the binding
  60-second Wave 2.5 limit, then repeat its acceptance run.
- [x] Diagnose and stabilize the training dynamics: residual-magnitude
  telemetry (‖F_θ(x_t)‖, ‖x_t‖, ratio at steps 0/16/32/48/64) showed the
  unnormalized recurrence had ‖F‖ ≈ ‖x‖ at step 0 (ratio max 1.36), doubling
  the state per step into FP16 saturation by step 16. Fixed with the explicit
  residual timestep x_{t+1} = x_t + RESIDUAL_ALPHA·F_θ(x_t),
  RESIDUAL_ALPHA = 1/CA_STEPS (claim A201.bounded-residual-dynamics, witness
  passing). Fresh 5-generation run: state_max ~1.4 (was 65504), mean CE
  decreasing 2.998 → 2.713 (below the random baseline), fitness rising
  0.0539 → 0.0675, zero saturation. The finite-difference suite validates the
  alpha-adjoint at alpha=1 (tight per-bank) and production alpha (coarse,
  catching a missing alpha scaling).
- [ ] Watch long-run dynamics under the new timestep (50+ generations,
  PT swap interval) and re-measure the 10-generation timing gate.
- [ ] Repair archive semantics before Wave 4+: inverse-variance metric update
  and use, exact RFF-mean adjustment on replacement, capacity enforcement on
  PCA rebin, robust PCA initialization, and an archive invariant checker.
- [x] Test the shared-memory weight-gradient hypothesis and record the
  result: per-block shared-atomic accumulation was measured 4.5x SLOWER
  (wave1 166-171s vs 34-40s for the global-atomic kernel; same-address
  shared-atomic replay serializes). Reverted. The backward bottleneck is
  launch structure/occupancy, not atomic traffic — profile (Nsight) and
  pursue CUDA graph capture / kernel fusion next.
- [ ] Profile the backward with Nsight and attack launch structure
  (graph capture, fused re-forward+grad kernels) toward the 10-gen/60s gate.
- [x] Adopt the cross-repo GPU scheduler (training-architecture
  gpu-scheduler/v1): pinned contract + fingerprint check, submit/wait via
  `architecture/gpu_client.py`, progress/v1 wrapping, scheduler-ledger
  provenance in evidence manifests, Slime registered as a consumer.
- [ ] Resolve any current-source failures before starting a later wave.
- [ ] Make every per-generation transfer conform to the four-point asynchronous
  transfer schedule, including the post-CAME gradient scalar read.
- [x] Apply `pause` and `prune` operator commands durably in the run loop
  (checkpoint reports unsupported until Wave 7 serialization exists).
- [ ] Add the missing BOM document or remove the stale `bom.md` reference from
  the blueprint.
- [x] Refresh README navigation after the documentation structure settles: link
  to the progress tracker and action queue as operational aids, while keeping
  the three binding specifications as the only normative sources and avoiding
  duplicated status or requirements.

## P1 — Complete Wave 3

- [ ] Implement the specified GPU placeholder forward/train kernels and wire
  them into the phase model.
- [ ] Evaluate the signed probe set to populate real probe fitness; stop using
  zero-initialized probe targets.
- [ ] Add `wave3-test` and prove placeholder MSE improvement, CUSUM detection,
  probe-signature validity, and varying nonzero placeholder surprise.

## P2 — Complete Wave 4 predictor-role activation

- [ ] Fire bootstrap once at archive half occupancy and inject 16 high-novelty
  classifier-derived predictor founders.
- [ ] Assemble predictor batches from IntentRegistry trajectories and provide
  the role-switched input/target wiring.
- [ ] Score predictor MSE, generate its seed gradients, and include predictors
  in archive insertion, spawning, and role-proportional selection.
- [ ] Implement top-K ensemble surprise, correlation blending, role-balance
  fitness, and 200–700 generation CUSUM calibration.
- [ ] Add `wave4-test` for bootstrap, founder survival, MSE, correlation,
  BTRAJ agreement, role mutation, and calibration requirements.

## P3 — Complete Wave 5 safety and structural pressure work

- [ ] Implement and wire audit, interpretability probes, sentinel scoring,
  lineage statistics/brake, and variance-floor multiplier.
- [ ] Implement and wire stress-slot refresh, elevated-SOT evaluation, and
  stress-failure reporting.
- [x] Ensure PT swaps exchange every required OrganismTable field, including
  batch assignment metadata (done in the Gates 1–3 pass; seed-gradient rows
  also move with the organism).
- [ ] Add `wave5-test` for all structural, stress, and ladder acceptance cases.

## P4 — Complete Waves 6–8

- [ ] Create graph capture/replay and red-team tests for attack classes A–F;
  prove graph/sequential equivalence.
- [ ] Implement full schema-checked checkpoint write/load, command-triggered
  and periodic checkpoints, and long-run telemetry/stability tests.
- [ ] Implement the compile-time global-context channel with `W_ctx`, forward
  broadcast, backward adjoint, and Wave 8 A/B verification.

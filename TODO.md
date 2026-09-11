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
- [ ] Shrink the `checked_cuda_calls` allowlist: convert
  `launch_grad_norm_reduce` / `launch_telemetry_kernels` / `propose_swaps` /
  `apply_sot_identity` to checked wrappers so `--strict` passes.
- [ ] Give the four red claims witnesses: `A401.bin-capacity`,
  `A401.live-statistics-exact` (archive invariant repairs),
  `S002.operator-command-effective` (durable operator state),
  `A101.sot-schedule-independent` (schedule-host-side test), and
  `I001.replay-evaluation-identity` (evaluation-record snapshot test).

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
- [ ] Diagnose and stabilize the training dynamics: telemetry shows the 64-step
  forward saturates ~50% of FP16 state values at ±65504 with logits ~1e5,
  mean CE ~69, and per-bank gradient norms ~1e16–1e17 at generation 0. Follow
  the Gate 3 order: gradient clipping / residual scaling / initialization
  redesign only after the finite-difference suite stays green.
- [ ] Repair archive semantics before Wave 4+: inverse-variance metric update
  and use, exact RFF-mean adjustment on replacement, capacity enforcement on
  PCA rebin, robust PCA initialization, and an archive invariant checker.
- [ ] Redesign the backward weight-gradient accumulation (shared-memory
  reductions instead of ~34.8B global atomicAdds/generation) and remove the
  per-phase stream synchronizations before CUDA graph capture.
- [ ] Resolve any current-source failures before starting a later wave.
- [ ] Make every per-generation transfer conform to the four-point asynchronous
  transfer schedule, including the post-CAME gradient scalar read.
- [ ] Apply `pause` and forced `checkpoint` operator commands in the run loop.
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

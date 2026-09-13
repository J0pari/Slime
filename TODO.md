# Slime TODO

Operational work queue in dependency order. Behavior is defined by the
binding specifications and the claim registry; build status is
machine-checked in `architecture/build_status.yaml` and rendered in
`docs/IMPLEMENTATION_STATUS.md`. This file lists what is being worked on,
not a history.

## P0 — Semantic correctness (gates further BUILD)

Mechanism presence is not contract fulfillment. C1-C3 close defects in
implemented items; no inventory item after I4 starts until they are closed
or explicitly deferred with a recorded decision.

- [x] C1a Predictor-target identity: classifier-only targets, real lineage
  ids, target SOT status, descriptor rows, and a typed contract witness.
- [x] C1b Predictor K-target aggregation: generation-rotated target slots
  (a predictor covers all K targets across K generations) with a
  per-predictor loss EMA feeding fitness; the EMA travels with organism
  identity through PT and checkpoints.
- [x] C2 Surprise population identity: reference training and the surprise
  computation (including the predictor ensemble) now precede the spawn wave;
  the `surprise_before_spawn` source gate pins the order.
- [x] C3 Role-gradient PT identity: the alignment kernel groups by a
  post-PT role buffer (uploaded after the swap), and the cross-role swap
  witness is written in `tests/evolution_regression.cu` (executes in VERIFY
  with the rest of the GPU suite).
- [x] C8 Archive attribution: historical attribution decided and
  documented in A401 (stored genome at insertion time, never a
  replay-equivalent reconstruction; delta entries reconstruct against
  the shared base captured with the entry), with the stored-genome
  witness.

## P0 — Finish the BUILD phase

No GPU integration, acceptance, or production run happens before this list
is empty; `evidence.py record` refuses GPU manifests until then.

I4
- [x] Variance floor multiplier per organism.
- [x] Audit cycle (least-squares fit, R^2, audit_mult) wired into fitness.
- [x] Probe panel (four probes, L_role collapse alarm).
- [x] Sentinel scoring and pruning-history training.
- [x] Per-role lineage stats and the non-mutating expanding-lineage brake.
- [x] SOT reference scratch separated from the stress slots (dedicated
  `d_sot_ref_organisms` buffer).
- [x] Stress ladder host core: lineage-biased refresh policy (one classifier
  and one predictor per sub-population per generation) and per-lineage
  rolling-window failure flagging.
- [x] Stress ladder device evaluation: elevated-SOT classifier forwards and
  pi-permuted predictor forwards on the stress slots, per-slot f_sot
  readback, wired into the generation loop.

I5
- [x] `GLOBAL_CONTEXT_ENABLED` gate, W_ctx in the flat weight space and all
  offsets, forward broadcast of the bmap summary into channels 14-15,
  backward context adjoint.
- [x] Host-verified adjoint reference and finite-difference witness.
- [x] GPU witnesses written (checkpoint aux channels vs stored summary;
  non-zero dW_ctx); they execute in VERIFY with the rest of the suite.

I6
- [x] RD adjoint (replay reproduces residual + reaction + diffusion).
- [x] Neutral genome encoding for RD coefficients (zero bits mean zero).
- [x] Enable RD in the main loop with per-organism decoded coefficients.
- [x] Host-verified RD adjoint reference with an adversarial FD witness;
  GPU FD/determinism witness written (executes in VERIFY).

I7
- [x] Capture/replay of the capturable phases per cuda_engineering §11
  (Forward, Backward, Optimizer, WorldPredict, WorldTrain, StressEval) with
  a debug mode and an equivalence witness.
- [x] StressEval captured after restructuring the readbacks into stable host
  staging; the SOT reference stays host-interleaved by design.

I8
- [x] Profile first (per-phase timing table under --profile; cached-segment
  backward removed the quadratic re-forward: total 102 -> 81.5 s for 3 gens).
- [ ] Meet the 10-generation / 60-second gate with per-generation
  checkpoint writes included. Measured 2026-09-12 at ~7.6 s/gen: backward
  5.6 (weight-grad 3.6 [main 2.34, reduce_inter 0.93, reduce_flow 0.29],
  stencil 1.15, reforward 0.75, rd 0.08), stress 0.84, forward 0.70.
  Measured rejections: shared-memory atomic accumulation (4.7x slower),
  launch bounds (no-op), tiled one-block-per-organism reduce (6x slower:
  64 blocks, 4096-FMA dependent chains), cell-split weight-grad with
  partials+merge (no mechanism for gain: the kernel is register-file
  limited — 128 regs x 512 resident threads fills the 64K register file,
  so extra blocks cannot raise resident threads; FD-clean but reverted as
  unmeasurable under session drift). The systemic limit is that the
  backward runs one block per organism everywhere (64 blocks, ~27%
  occupancy) with 128 registers per thread; the next lever is reducing
  register pressure (fewer live arrays per thread) or an explicit
  3-blocks/SM launch bound with controlled spills, then the same analysis
  for the stencil gather.

I9
- [x] Dashboard surface: role fraction, r, rho, swap stats, stress-failure
  flags in one periodic telemetry line.
- [x] Red-team classes A-F with detection verification: A/B/F in the
  host injection test, C in the attack-framed gate test, D in the
  checkpoint tamper/schema refusals, E in the PT correspondence
  sensitivity check. F's host witness mirrors the CUSUM reference (the
  production path runs on hardware); the claim stays provisional until
  F has a production-path witness.
- [ ] 5000-generation stability verification (checkpoint/restart
  byte-equivalence, no spontaneous class A-F conditions): the chunked
  harness exists (`make stability-run`, `tests/long_run_check.py`,
  default 20 generations in 5-generation chunks) and fails on any
  spontaneous stress flag or nonfinite dashboard value; the 5000-
  generation acceptance run itself remains.

## P1 — VERIFY (after BUILD)

- [ ] Rerun every witness and record fresh evidence: host suite, autodiff
  acceptance, evolution regression, task-conditioning, checkpoint state.
  Regenerate the status until no claim reads `pass/stale`.
- [ ] End-to-end predictor witnesses: bootstrap -> predictor evaluation ->
  predictor archive -> predictor parent -> predictor offspring.
- [ ] Reference witness: device forward/train match a hand-computed
  reference; probe surprise nonzero and varying; MSE decreases over
  generations.
- [ ] Bound the reference's uncertainty output before surprise is treated
  as a robust signal (`A601.trust-weight-composition`).
- [ ] Execute the preregistered experiments (E1-E5) and record manifests;
  update each protocol's status in `architecture/experiments.yaml`.
- [ ] Property witnesses: C4 genome fieldwise perturbation matrix, C5
  checkpoint continuation equivalence, C6 PT permutation test, C7 phase
  trace checked against the declared phase model.
- [ ] Gate mutation testing: remove one transaction move, one CUDA check,
  and one role canonicalization; prove the right witness goes red.

## P2 — INTEGRATE (after VERIFY)

- [ ] 100-generation resumable run: PT exchange at generation 50, zero
  saturation and nonfinite values, archive invariants intact, learning
  signal healthy, restart continuity verified.

## P3 — OPERATE (after INTEGRATE)

- [ ] Long runs with checkpoint/restart, reproducibility checks, and the
  dashboard surface.
- [ ] Portability hardening: cross-platform build paths, configurable CUDA
  architecture, and a GPU CI lane (the current build is Windows/MSVC/CUDA
  specific and CI is host/static only).

## Carryover

- [ ] Make every per-generation transfer conform to the four-point
  asynchronous transfer schedule.

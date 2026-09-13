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
- [x] C2 Surprise population identity: placeholder training and the surprise
  computation (including the predictor ensemble) now precede the spawn wave;
  the `surprise_before_spawn` source gate pins the order.
- [x] C3 Role-gradient PT identity: the alignment kernel groups by a
  post-PT role buffer (uploaded after the swap), and the cross-role swap
  witness is written in `tests/evolution_regression.cu` (executes in VERIFY
  with the rest of the GPU suite).
- [ ] C8 Archive attribution: decide historical-attribution versus
  replay-equivalent entries in the blueprint and document the shared-base
  interaction.

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
- [ ] `GLOBAL_CONTEXT_ENABLED` gate, W_ctx in the flat weight space and all
  offsets, forward broadcast of the bmap summary into channels 14-15,
  backward context adjoint.

I6
- [ ] RD adjoint (replay reproduces residual + reaction + diffusion).
- [ ] Neutral genome encoding for RD coefficients (zero bits mean zero).
- [ ] Enable RD in the main loop with per-organism decoded coefficients.

I7
- [ ] Capture/replay of the capturable phases with a debug mode that
  synchronizes and validates.

I8
- [ ] Profile first (per-phase timing, then Nsight where available); the
  shared-memory-atomic experiment is already measured and rejected.
- [ ] Meet the 10-generation / 60-second gate with per-generation
  checkpoint writes included.

I9
- [ ] 5000-generation stability under all subsystems.
- [ ] Red-team classes A-F with detection verification.
- [ ] Dashboard surface: role fraction, r, rho, swap stats, stress failure
  rates.

## P1 — VERIFY (after BUILD)

- [ ] Rerun every witness and record fresh evidence: host suite, autodiff
  acceptance, evolution regression, task-conditioning, checkpoint state.
  Regenerate the status until no claim reads `pass/stale`.
- [ ] End-to-end predictor witnesses: bootstrap -> predictor evaluation ->
  predictor archive -> predictor parent -> predictor offspring.
- [ ] Placeholder witness: device forward/train match a hand-computed
  reference; probe surprise nonzero and varying; MSE decreases over
  generations.
- [ ] Bound the placeholder's uncertainty output before surprise is treated
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

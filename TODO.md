# Slime TODO

Operational work queue in dependency order. Behavior is defined by the
binding specifications and the claim registry; build status is
machine-checked in `architecture/build_status.yaml` and rendered in
`docs/IMPLEMENTATION_STATUS.md`. This file lists what is being worked on,
not a history.

## P0 — Finish the BUILD phase

No GPU integration, acceptance, or production run happens before this list
is empty; `evidence.py record` refuses GPU manifests until then.

I4
- [ ] Variance floor multiplier per organism.
- [ ] Stress ladder: refresh from the main pool, elevated-SOT evaluation,
  failure flagging.
- [ ] Wire audit_mult and variance_mult into fitness composition; activate
  the expanding-lineage brake in archive insertion.

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

## P2 — INTEGRATE (after VERIFY)

- [ ] 100-generation resumable run: PT exchange at generation 50, zero
  saturation and nonfinite values, archive invariants intact, learning
  signal healthy, restart continuity verified.

## P3 — OPERATE (after INTEGRATE)

- [ ] Long runs with checkpoint/restart, reproducibility checks, and the
  dashboard surface.

## Carryover

- [ ] Make every per-generation transfer conform to the four-point
  asynchronous transfer schedule.

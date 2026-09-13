# Slime TODO

Operational work queue in dependency order. Behavior is defined by the
binding specifications and the claim registry; implementation status is
generated from evidence manifests (`python architecture/compiler.py
status`). This file lists what is being worked on, not a history.

## P0 — Green baseline with current evidence

- [x] Fix the scheduler-client relative-path resolution: a relative
  executable is always resolved against the job working directory (the
  owner validates existence); `test_submit_builds_argv_and_parses_ack`
  passes.
- [ ] Rerun every witness on the current source and record fresh evidence:
  host suite, autodiff acceptance, evolution regression, task-conditioning,
  checkpoint state. Regenerate the status until no claim reads `pass/stale`.
- [ ] Prove checkpoint/restart on hardware: GPU state roundtrip, a real
  process restart with `--resume`, payload-checksum and schema rejection,
  and a refused replacement leaving the previous checkpoint intact.
- [ ] Run a 100-generation resumable classifier baseline: PT exchange at
  generation 50, zero saturation and nonfinite values, archive invariants
  intact, learning signal healthy, and restart continuity verified.
- [ ] Commit no new evidence until
  `python architecture/compiler.py check --golden --strict` passes on a
  clean tree.

## P0 — Close predictor causality

- [x] Predictors enter the archive: `insert_into_archive` no longer filters
  by role, so the predictor live list can parent predictor spawns.
- [x] `PredictorBatch` carries pool-slot and lineage identity separately
  (stationary probes use `pool_slot = -1`); the curriculum error EMA
  updates the correct pool slot.
- [ ] Add end-to-end witnesses: bootstrap -> predictor evaluation ->
  predictor archive -> predictor parent -> predictor offspring.

## P1 — Complete the placeholder per the binding design

- [ ] GPU placeholder forward/train kernels per cuda_engineering 4.5/4.6.
- [ ] Bound the uncertainty parameterization before surprise is treated as
  a robust signal.

## P1 — Transfer and operator surface

- [ ] Make every per-generation transfer conform to the four-point
  asynchronous transfer schedule.
- [ ] Add the missing BOM document or remove the stale reference.

## P2 — Performance (measurement first)

- [ ] Profile the backward (launch structure and occupancy; the
  shared-memory-atomic variant was measured and rejected).
- [ ] Meet the 10-generation / 60-second gate with per-generation
  checkpoint writes included.

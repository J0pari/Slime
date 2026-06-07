# C-001: Construction Sequence

Source of truth: `docs/blueprint.md` (Sheet C-001).

Staged plan. Each stage has a clear exit condition; do not start the next stage
until the previous one passes its acceptance check. Stages are ordered by
dependency, not by calendar — the schedule is whatever the work takes.

Nothing in this tree has been compiled or run on CUDA hardware yet. Every stage
below that involves device kernels or captured graphs is therefore "designed,
not validated" until it executes on an NVIDIA GPU with the CUDA toolkit. The
host-side math has unit coverage (`make check`); the device side does not.

| Stage | Scope                                                              | Acceptance |
| :---- | :----------------------------------------------------------------- | :--------- |
| 1     | Foundation: pool, genome codec, organism table, device build wiring | Tree compiles under nvcc; a single organism decodes and runs one CA step without NaN |
| 2     | AD and optimizer: warp tape, CAME path, SOT library, identity-loss branch | Backward pass produces finite gradients; a classifier learns a trivial task over a few steps |
| 3     | QD core: final-bmap descriptor archive, RFF KDE, sparsity bonus, gate-based fitness; validate classifier-only operation; confirm SOT corrigibility gate | Classifier-only run reproduces baseline accuracy/diversity metrics; SOT gate penalizes an SOT-blind organism |
| 4     | Captured-graph execution                                          | Behavioral equivalence between captured and sequential modes on a short run |
| 5     | Placeholder regressor + probe set: regressor as sole surprise source, probe set generation + signing + host evaluation, CUSUM on placeholder surprise | Placeholder surprise tracks fitness drift on synthetic injections |
| 6     | BTRAJ capture + role machinery: sample bmap at intermediate CA steps; store BTRAJ; role tag in genome and role-switched input pathway; verify role mutations occur at the intended low rate; bootstrap-trigger condition and predictor-founder spawning | BTRAJ correctness + role-switched input unit tests pass; measured role mutation rate ≈ 1e-4 |
| 7     | Predictor role activation: run through the bootstrap trigger in a controlled experiment; verify founders reproduce; prediction loss decreases over generations; ensemble surprise correlates with placeholder; hybrid blending; role-balance fitness scaling | Bootstrap-trigger integration test passes; r climbs above 0.5 within a bounded post-bootstrap window |
| 8     | Structural pressures + PT ladders: role-aware audit + probe panel (incl. L_role); sentinel training on role-blind data; per-role lineage tracking; mutation-rate ladder with Metropolis swaps + adaptive β; SOT-density stress ladder; red-team exercises | All Q-001 attack classes A–F detected within their T_detect_max windows |
| 9     | Full integration + long-run hardening: long stability runs with all subsystems active; optional MPK backend behavioral-equivalence validation | Long post-bootstrap run sustains predictor sub-population, r > 0.5, no spontaneous Class A–F conditions |

Notes:

- Stages 1–4 are the substrate foundation; they carry no role concept and can be
  validated with classifiers alone.
- Stage 6 is the smallest substantive change introducing the role concept; hold
  off on predictor *activation* until stage 7 so the bootstrap trigger can be
  exercised in isolation.
- Hybrid blending (stage 7) and PT ladders (stage 8) are independent in the
  source tree; they can be developed in parallel, but the acceptance runs in
  stage 9 should happen against the merged build.

Current implementation status is tracked in `docs/IMPLEMENTATION_STATUS.md`.

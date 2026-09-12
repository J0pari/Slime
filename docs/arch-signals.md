# Architecture Signals

Non-normative observations and friction notes. This document records things
worth remembering about the architecture-control process; it never changes
behavior. The three binding specifications remain blueprint.md,
cuda_engineering.md, and construction_plan.md.

## Signals

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

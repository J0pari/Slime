# Construction Plan

Derived from `docs/blueprint.md` (behavioral spec) and
`docs/cuda_engineering.md` (GPU engineering spec).

## Order of operations (binding)

1. **BUILD** — implement every intended feature in the dependency order below.
   The build is complete when every item in the feature inventory is
   implemented and compiles. No acceptance testing gates the build; a feature
   is not "done later in a wave", it is either built or not built.
2. **VERIFY** — component and property tests per subsystem (host and GPU
   component tests), then numerical and behavioral checks.
3. **INTEGRATE** — multi-generation integration runs, timing, invariants,
   archive/telemetry acceptance.
4. **OPERATE** — long production runs, checkpoint/restart, reproducibility,
   dashboard telemetry.

Rules:

- No GPU integration, acceptance, or production run happens until the BUILD
  phase is complete. Compilation and host-side unit tests are build tools,
  not acceptance events.
- No feature is represented as working before it is built and verified.
  Partial features are marked partial in the inventory; nothing is "enabled"
  by documentation.
- Checkpoint/resume exists before any run that cannot be trivially repeated.
  An un-resumable multi-hour run is a defect.
- Every feature listed in the inventory is normative: it comes from
  blueprint.md or cuda_engineering.md. If the architecture is wrong, change
  the specs first.

## Where continuous and experimental work sits

- CI (`.github/workflows/architecture.yml`) runs the CPU/static gates on
  every push and pull request: `make check`, the golden architecture check,
  the strict source gates, and the guards' own negative tests. GPU witnesses
  stay scheduler-submitted and are not part of CI.
- The experiment registry (`architecture/experiments.yaml`) is preregistered
  now but executed in VERIFY/OPERATE; each execution records evidence
  manifests and updates the protocol status.
- VERIFY opens with evidence closure: rerun every witness on current source,
  record fresh manifests, and regenerate the status until no claim reads
  stale, so the next session starts from a clean checkpoint.
- Portability hardening (post-BUILD, before OPERATE): the build is currently
  Windows/MSVC/CUDA-specific (`env.sh`, `.exe` outputs, a `/STACK` linker
  flag, a default `sm_86` target, and CI that covers host/static checks but
  not GPU execution). Cross-platform build paths, a configurable CUDA
  architecture, and a GPU CI lane are separate work, not part of any
  inventory item.

## Scope discipline and extension gate

The extension axes in blueprint.md are documented so the current seams stay
general; they are not in scope. None may begin until every inventory item is
implemented and the experimental program has produced evidence on the
current system.

Admission rules: an extension enters this plan as a new inventory item (with
status and mechanisms in architecture/build_status.yaml) before any code;
new roles are schema entries plus input wiring and objectives on the shared
substrate, not new model classes; novelty pressure stays competence-gated;
the host stays the authority that schedules tests. An extension that appears
in code before it appears here is a build defect and is removed or rolled
back.

---

## Semantic correctness stream (gates further BUILD)

Syntactic coverage is not semantic coverage. Mechanism-presence checks can
pass while the described contract is not met. The following defects in
implemented items are closed with semantic witnesses before any inventory
item after I4 starts:

- C1 Predictor-target contract. `assemble_predictor_batch` draws
  classifier-only targets, carries the real lineage id (not the pool index)
  and the SOT status of the target; a predictor's score aggregates over its
  K targets; a typed transaction test asserts targets, lineage attribution,
  SOT status, and target descriptors stay together.
- C2 Surprise population identity. Surprise and the predictor ensemble are
  computed from the evaluated pre-spawn population snapshot, never from
  post-spawn role/fitness metadata (descriptors describe the pre-spawn
  population; roles and fitness must describe the same one).
- C3 Role-gradient PT identity. A forced cross-role PT swap witness proves
  gradient role attribution follows organism identity through the swap.

C4-C7 are property witnesses recorded in VERIFY:

- C4 Genome fieldwise perturbation matrix: each declared genome field is
  mutated independently and either changes the phenotype as declared or is
  marked dormant with a reason (RD bits dormant until I6).
- C5 Checkpoint continuation equivalence: run N -> checkpoint -> run M
  agrees with run N -> checkpoint -> reload -> run M at the required
  determinism level.
- C6 PT permutation test: a random permutation of organism identities
  transforms every identity-bound observable equivariantly.
- C7 Phase trace: `step_generation` emits runtime phase tokens checked
  against the declared phase model, so the executable order is witnessed,
  not only declared.

C8 is a specification decision: the blueprint must state whether archive
entries are historical attributions (descriptor and fitness at capture time
under the then-current shared W) or replay-equivalent solutions, and the
shared-base interaction must be explicit where insertion and selection
consume fitness.

---

## Feature inventory (build order)

Dependencies are semantic: an item may only be built when everything it
depends on exists.

### I1 — Checkpoint and resume (S-001, cuda_engineering §14)

- Full run-state serialization: organism table (genomes, deltas, lineage,
  parent, spawn_gen, replica_tag, fitness, f_raw, f_sot, role,
  batch_sample_idx), archive (entries, bins, live lists, RFF means,
  inverse-variance EMA, descriptor EMA, PCA state), mutation ladder, CUSUM
  states, reference regressor + AdamW moments, replay buffer, correlation
  window, probe set + probe fitness, classifier batch, RNG state, operator
  state, calibrated s_target and flags, generation counter.
- Device state: shared weights and CAME moments (m, v, c, prev_u). Grids and
  checkpoints are re-seeded by the next forward and are not serialized.
- Atomic write (temp file + replace), schema hash and version check on load,
  loud refusal on mismatch.
- Resume from the saved generation; `--resume` and `--ckpt <path>` CLI;
  per-generation saves; the operator `checkpoint` command writes immediately.
- Verification: host archive roundtrip; GPU exact-state roundtrip; resume
  continuity across a process restart.

### I2 — Predictor role activation (A-601, A-701)

- Predictor batch assembly from the Intent Registry: K=8 targets sampled from
  the active pool weighted by ensemble prediction error; target bmap_32 and
  ground-truth bmap_64 per target.
- Bootstrap: one-shot trigger at archive half occupancy; inject
  PREDICTOR_FOUNDERS=16 role-flipped high-novelty classifier copies.
- Predictor forward through the same NCA and checkpointed backward; MSE loss
  on bmap_64 against the target; predictor seed gradients.
- Predictors participate in archive insertion, spawning, role-proportional
  parent selection, and fitness composition.
- Ensemble surprise (top-K predictors, per-descriptor variance), hybrid
  blending with the reference on the correlation window, role-balance
  fitness scaling driven by rho = s_avg / s_target.
- CUSUM calibration over the specified window: k = 0.5 sigma, h = 5 sigma,
  frozen s_target.
- Verification: founder injection, predictor survival and MSE decrease,
  BTRAJ agreement, role mutation rate in range, calibration firing.

### I3 — Probe and reference completion (A-601, cuda_engineering §4.5–4.6)

- GPU reference forward and training kernels per the engineering spec.
- Signed probe set evaluation with real ground-truth fitness (probe targets
  currently zeros are a defect, not a reference).
- Reference surprise from real probe prediction error; the correlation
  window starts only when both signals are live.
- Verification: reference MSE decreases, surprise nonzero and varying,
  probe signature validity, CUSUM detection on an injected shift.

### I4 — Structural pressures (S-003)

- Audit cycle: least-squares fit of bmap to fitness, R^2, audit_mult.
- Probe panel: four linear probes, per-probe accuracy, l_role accuracy.
- Sentinel scoring: ensemble logistic scoring of all organisms.
- Lineage statistics: per-role share, growth, runaway detection; the
  expanding-lineage brake wired into archive insertion.
- Variance floor multiplier per organism.
- Stress ladder: refresh from the main pool (lineage-biased, 25% per
  generation), elevated-SOT evaluation, failure flagging. Classifier stress
  slots evaluate against the elevated SOT-marked images with per-organism
  effective-weight references. Predictor stress slots evaluate the
  target-permutation gate of blueprint S-003: the nominal-target response is
  the reference and the response to pi(target) must match it after pi^-1.
- Wire audit_mult and variance_mult into fitness composition; activate the
  lineage brake.
- Verification: audit_mult range, probe accuracy, sentinel finiteness,
  lineage accounting, brake activation, stress-failure detection.

### I5 — Global context channel (A-203, cuda_engineering §16)

- `GLOBAL_CONTEXT_ENABLED` configuration gate, W_ctx weight layer added to
  the flat weight space and all offsets, forward broadcast of the bmap
  summary into channels 14–15 at sample steps, backward context adjoint.
- Kaiming initialization and telemetry bank coverage for W_ctx.
- Verification: non-zero channels 14–15 after step 16 when enabled; non-zero
  dW_ctx; bitwise BTRAJ match between the plain and checkpointed forward with
  context enabled; A/B descriptor difference.

### I6 — Reaction-diffusion activation (A-202, A-103)

- RD adjoint: checkpoint reconstruction must reproduce the full transition
  (CA residual + reaction + diffusion); the state adjoint must include the
  transposed RD operator (Laplacian symmetric, reaction matrix transposed).
- Neutral genome encoding for RD coefficients (zero genome bits must mean
  zero reaction/diffusion, not the current quantized -1).
- Enable RD in the main loop; RD coefficients decoded per organism and passed
  to the forward.
- Verification: finite-difference gradients with RD enabled; forward/replay
  state equivalence; no saturation regression.

### I7 — Phase graphs (A-102, cuda_engineering §11)

- Capture and replay of the capturable phases per cuda_engineering §11
  (Forward, Backward, Optimizer, WorldPredict, WorldTrain, StressEval) with
  a debug mode that synchronizes and validates. The phase list is the
  engineering spec's; the earlier parenthetical here was a stale summary.
  The SOT reference is host-interleaved by design and not captured.
- Verification: captured execution produces identical outputs to sequential
  execution over a multi-generation run.

### I8 — Performance completion (cuda_engineering, acceptance)

- Meet the binding 10-generations-in-60-seconds target on the specified
  RTX 3060 Laptop GPU without reducing the scientific workload.
- Profile first (per-phase timing, then Nsight where available); attack the
  actual hotspots (launch structure, kernel fusion, graph replay, transfer
  schedule); the measured shared-atomic experiment is already rejected with
  data.
- Verification: the timing gate, with the same acceptance suite passing.

### I9 — Long-run hardening (S-001, Q-001)

- 5000-generation stability under all subsystems.
- Red-team tests for attack classes A–F with detection verification.
- Dashboard/telemetry surface: role fraction, r, rho, swap stats, stress
  failure rates.
- Verification: sustained r > 0.5, no spontaneous class A–F conditions,
  checkpoint/restart byte-equivalence over the long run.

---

## Verification phase (after BUILD completes)

Component and property tests, per subsystem:

- Genome codec roundtrip, seed uniqueness, mutation/role-lock behavior.
- RNG determinism and single-PRNG source gates.
- Forward determinism, checkpoint replay bitwise equivalence, finite-
  difference gradients (shared and effective banks, alpha regimes).
- CAME production equation; PT transaction completeness (including
  effective-weight banks); spawn-wave uniqueness; replay evaluation
  identity; phase order.
- Archive insert/rebin invariants under randomized property testing.
- Task conditioning (all 16 embedding dimensions reach the forward).
- Residual-dynamics bounds under the production timestep.
- Checkpoint roundtrip and resume continuity.
- Reference/probe/probe-signature behavior.
- Predictor: founder injection, MSE decrease, correlation, calibration.
- Structural: audit, sentinel, lineage brake, stress ladder.
- RD: adjoint correctness with RD enabled.
- Graphs: captured vs sequential equivalence.

## Integration phase (after VERIFY passes)

- Multi-generation runs (classifier-only, then full role mixture) with
  archive/telemetry invariants checked every generation.
- Gate mutation testing: deliberately remove one required transaction move,
  one CUDA error check, and one role canonicalization, and prove the
  appropriate witness goes red.
- The C4-C7 property witnesses above.
- 10-generation timing gate on the specified hardware.
- Numerical health: no NaN/Inf, bounded activations, no persistent FP16
  saturation, finite weight/update norms.
- Role/archive composition and surprise/calibration behavior.

## Operation phase (after INTEGRATE passes)

- Long runs (5000+ generations) with checkpoint/restart, reproducibility
  checks, and the dashboard surface.
- Production events only after the above; every run is resumable and every
  result is evidence-linked.

## Experimental program (after INTEGRATE passes)

The co-evolutionary claims are empirical. Protocols are declared in
`architecture/experiments.yaml` — hypothesis, intervention, controls,
metrics, seed set, decision rule — and rendered in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md); results are recorded
as evidence manifests, not as prose. The program:

### E1 — Endogenous prediction pressure

Does co-evolving prediction change search dynamics? Classifier-only,
fixed-reference, and evolved-predictor conditions under identical compute
budgets, measuring rates of behavioral novelty and archive expansion rather
than final accuracy.

### E2 — Shared-substrate transfer versus interference

Does the shared substrate create transferable structure or interference?
Track the cosine similarity of the mean classifier and predictor gradients
across developmental phases and seeds, against completely separate
substrates.

### E3 — Surprise homeostasis and role dynamics

Does surprise homeostasis stabilize complexity? Characterize the
(surprise, classifier, predictor, novelty) series: fixed point, oscillation,
hysteresis, boom/bust cycles, role extinction.

### E4 — Ensemble disagreement tracks held-out error

Is ensemble variance a usable epistemic signal? Rank-correlate it with
squared held-out error on the frozen probes, with calibration and diversity
bounds.

### E5 — CAME versus AdamW

Does the confidence-adjusted update earn its complexity? Matched runs of
CAME against AdamW under identical seeds and compute budgets; the decision
rule asks whether CAME improves a preregistered metric beyond the baseline
band rather than whether it is implementable.

Dynamics questions the series should answer: stable fixed point,
oscillation, hysteresis, boom/bust cycles, role extinction, and
Red-Queen behavior. Treating surprise as a robust safety signal requires
the trust composition of `A601.trust-weight-composition` first: raw
correlation measures agreement, not correctness.

## Current implementation status

The build inventory status is machine-checked in
`architecture/build_status.yaml` and rendered in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). It is not maintained as
prose here. Until every inventory item is `implemented`, no integration,
acceptance, or production GPU run is permitted, and `evidence.py record`
refuses GPU manifests.

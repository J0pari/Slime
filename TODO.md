# Slime TODO

Operational work queue in dependency order. Behavior is defined by the
binding specifications and the claim registry; build status is
machine-checked in `architecture/build_status.yaml` and rendered in
`docs/IMPLEMENTATION_STATUS.md`. This file lists what is being worked on,
not a history; the failure history lives in `docs/arch-signals.md`.

## Work order compliance (directive 64d0820993d2, 2026-09-14)

- §25 item 4 / §11.1 (strong-ID/memset build failures): done. Value
  initialization throughout, the two GCC-only memsets (PredictorBatch,
  ProbeSet) and the second strong-ID printf fixed; g++ 15.2 via WSL with
  -Wall -Wextra -Werror compiles clean and host 4940/4940 passes; clang
  -Wnontrivial-memcall is recorded as `make host-tests-clang`. No toolchain
  warnings were weakened.
- §25 item 16 / §11.3 (blocker prose drifting from machine state): done.
  bridge.yaml declares requires_build, requires_experiments,
  requires_contracts, and requires_evidence; derive_bridge computes the gate
  and the reasons; check_bridge fails on any declared/derived mismatch; the
  compiler rejects implemented items that list missing reasons.
- §11.2 (cross-platform architecture tests): done. The fixture is a .cmd on
  Windows and an executable shell script on POSIX; the Linux architecture
  suite passes 71 tests via WSL.
- §11.4 (keep the Trader bridge closed until prerequisites pass): done. The
  gate is derived CLOSED from the four requirement lists; schema
  compatibility alone opens nothing.
- §11.5 (surprise blending): implemented (trust-weight composition:
  calibration, held-out over-bound, diversity, correlation); GPU validation
  queued; regime-based reliability remains future work.

## In flight

- 5000-generation stability run (I9): scheduler job 65a5d0ea88b24ce9,
  running, resumed from generation 1812. Chunked at 25 with resume-on-start,
  max 900 min, 3 retries; the harness fails only on sustained stress flags
  (more than half the post-warmup chunks), nonfinite dashboard values, or a
  checkpoint that does not advance. Prior attempts and their causes are in
  arch-signals; the scheduler incident record is below.
- Queued witnesses: e7e8e56edc36a67b (autodiff acceptance) and
  14bf207e63808404 (forward smoke), both rebuilt and guard-checked.

## Incident 2026-09-13 (scheduler, relay to owner)

- The earlier 5000-generation run (fe7866b303882f47) was killed silently at
  ~generation 1721 (no coevo process, GPU idle, log stopped mid-generation)
  while the daemon still reported it running and blocked the queue for
  ~89 minutes; `stop --job` cleared it. Two gaps to relay: (a) the RAM
  reclaim excludes the running job's tree, but the tree is python (wrapper)
  -> python (harness) -> coevo (grandchild), so the grandchild may be
  misclassified as reclaimable; (b) there is no liveness check for a running
  job whose process tree has died. The owner has since made transient
  failures (pressure stops, blocked queues) retryable.

## Review 2026-09-13 scope (Slime) — complete

- [x] Strong-ID initialization migration (value initialization; no bytewise
  clearing of ID-bearing structs) and the second strong-ID printf. Verified
  with clang++ -Wall -Wextra -Werror -Wnontrivial-memcall: clean compile,
  host 4940/4940; `make host-tests-clang` records the check. The g++
  cross-check is now done locally via WSL Ubuntu: g++ 15.2 with
  -Wall -Wextra -Werror compiles the host suite clean and it passes
  4940/4940 (this caught two more memsets gcc alone flags:
  PredictorBatch in problem_generator.cu and ProbeSet in the host tests,
  both now value-initialized), and the Linux architecture suite passes
  71 tests (the POSIX fixture is exercised for real).
- [x] Cross-platform e2e fixture (.cmd on Windows, executable shell script
  on POSIX).
- [x] Control-plane drift: bridge.yaml declares requires_build [I9] and
  requires_experiments [E1..E5]; derive_bridge computes the gate and reasons;
  check_bridge fails on declared/derived mismatch; the compiler rejects an
  implemented item that lists missing reasons. Negative tests for both.
- [x] Surprise blending reliability: the trust-weight composition
  (calibration, held-out, diversity, correlation) is implemented and
  provisional.

## P0 — Semantic correctness (gates further BUILD)

- [x] C1a/C1b predictor-target identity and K-target aggregation.
- [x] C2 surprise-before-spawn identity (`surprise_before_spawn` gate).
- [x] C3 role-gradient PT identity (post-PT role buffer, cross-role witness).
- [x] C8 archive attribution: historical attribution documented in A401 with
  the stored-genome witness.

## P0 — Finish the BUILD phase

No GPU integration, acceptance, or production run happens before this list
is empty; `evidence.py record` refuses GPU manifests until then.

- I1-I7: implemented and hardware-validated (see the status surface).
- I8: implemented under the soft target (wall 60.4 s for 10 generations,
  6.04 s/gen). The measured phase budget, rejections, and the fused
  stress-launch win are in arch-signals. Further optimization is optional;
  the next lever if pursued is register-pressure reduction in the backward.
- I9: dashboard done; red-team classes A-F have detection verification (F's
  host witness mirrors the CUSUM reference; the claim stays provisional).
  The 5000-generation stability run is in flight; on completion flip I9 to
  implemented and record the evidence.

## P1 — VERIFY (after BUILD)

- [ ] Rerun every witness and record fresh evidence: host suite, autodiff
  acceptance, evolution regression, task-conditioning, checkpoint state.
  Regenerate the status until no claim reads `pass/stale`.
- [ ] End-to-end predictor witnesses: bootstrap -> predictor evaluation ->
  predictor archive -> predictor parent -> predictor offspring.
- [ ] Reference witness: device forward/train match a hand-computed
  reference; probe surprise nonzero and varying; MSE decreases.
- [ ] Trust-weight GPU validation: the composed weight's telemetry behaves
  (bounded, zero before bootstrap, nonzero after) over a run.
- [ ] Execute the preregistered experiments (E1-E5) and record manifests;
  update each protocol's status in `architecture/experiments.yaml`.
  PREREQUISITE (found while checking turnkey readiness): every experiment's
  controls need a controlled variant the binary cannot currently select —
  classifier-only QD and a fixed reference predictor (E1), separate role
  substrates (E2), frozen s_target with role balance disabled (E3), the
  single-predictor variance proxy (E4), AdamW instead of CAME (E5). The
  protocols are preregistered and the decision rules are stated, but the
  control configurations must be admitted spec-first (a claim-bearing
  configuration surface, or per-experiment builds recorded in the
  manifests) before any experiment can execute. Do not treat E1-E5 as
  turnkey until that mechanism is decided and built.
- [ ] Property witnesses: C4/C5/C6/C7 are written and run in VERIFY.
- [x] Gate mutation testing (transaction move, CUDA check, role
  canonicalization via tests/mutation_check.py).

## P2 — INTEGRATE (after VERIFY)

- [ ] 100-generation resumable run: PT exchange at generation 50, zero
  saturation and nonfinite values, archive invariants intact, learning
  signal healthy, restart continuity verified.

## P3 — OPERATE (after INTEGRATE)

- [ ] Long runs with checkpoint/restart, reproducibility checks, and the
  dashboard surface.
- [ ] Portability hardening: cross-platform build paths and a GPU CI lane
  (the build is Windows/MSVC/CUDA specific; CI is host/static only;
  `ARCH` is already overridable).

## Carryover

- [x] Four-point transfer schedule audited against cuda_engineering section
  3: the core path maps cleanly (T1 forward inputs/batch/deltas/predictor
  targets, T2 descriptors/btraj, T3 roles/seed grads after PT, T5 reference
  minibatch and surprise readback, T4 weights only at the per-generation
  checkpoint write, which is the case the spec names). The stress-block
  uploads/readbacks are governed by section 11's StressEval capture rule
  (stable staging), not section 3, so they are not violations. No changes
  needed.

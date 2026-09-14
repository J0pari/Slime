# Slime TODO

Operational work queue in dependency order. Behavior is defined by the
binding specifications and the claim registry; build status is
machine-checked in `architecture/build_status.yaml` and rendered in
`docs/IMPLEMENTATION_STATUS.md`. This file lists what is being worked on,
not a history.

## Review 2026-09-13 scope (Slime)

- [x] Strong-ID initialization migration: ID-bearing structs are
  value-initialized (World via new World(), StressLadder/StressLineageRecord
  via {}), the bytewise clears of World/Archive/SentinelHistory/LineageStats
  and the test-side Archive/ArchiveEntry memsets are gone; the printf
  strong-id call already used .value(). Verified with clang++
  -Wall -Wextra -Werror -Wnontrivial-memcall (the local stand-in for the
  g++ CI warning class, since g++ is not installed): clean compile and
  4940/4940 passing. That check also caught a second strong-ID printf
  (stress_ladder.cuh's [STRESS] flag line), now .value(). A g++ CI run
  remains the final confirmation.
- [x] Cross-platform e2e fixture: the fake binary is a .cmd on Windows and
  an executable shell script on POSIX, so the architecture suite no longer
  fails on Linux.
- [x] Control-plane drift: bridge.yaml now declares requires_build [I9]
  and requires_experiments [E1..E5]; derive_bridge computes the gate and the
  reasons from the registries, check_bridge fails on any declared/derived
  mismatch, render_bridge emits the derived reasons, and the compiler now
  rejects an implemented item that lists missing reasons (the exact drift).
  The stale ~27-second I8 text and the stale red-team missing line are gone;
  I9's missing list is the 5000-generation run only.
- [x] Surprise blending reliability: the trust-weight composition
  (calibration, held-out, diversity, correlation) is implemented and
  provisional.

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

## GPU authorization guard (complete)

- All six canonical binaries are rebuilt with the guard and refusal-checked
  (bare launch exits 2): coevo, evolution_regression, checkpoint_state,
  autodiff_acceptance, task_conditioning, forward_smoke. Two stale witnesses
  were found by the rebuild and fixed: autodiff_acceptance called
  launch_backward_all without the RD coefficients argument, and
  forward_smoke referenced the moved HIDDEN_DIM; both now compile and their
  runs are queued: e7e8e56edc36a67b (autodiff acceptance) and
  14bf207e63808404 (forward smoke).

## Original guard note

- The startup guard and its source gate are in source; the canonical
  build/*.exe binaries are NOT yet rebuilt with it because the queued jobs
  (5000-gen and the three measurements) run the pre-guard binaries and their
  submitted env predates the marker. Once the queue drains:
  `python tests/rebuild_binaries.py` builds all six canonical binaries and
  asserts each refuses a bare launch (exit 2). Then resubmit any queued
  measurement whose env predates the marker.

## Queued measurements (scheduler)

- [x] slime-i8-phases (b756631a6c3c6720): done. Backward ~4.57 s/gen
  (wg_main 1959, reduce_inter 726, reduce_flow 245, stencil 967, reforward
  608, rd 67 ms); WALL 6.6 s for 1 generation.
- [x] slime-i8-wall (8e1d50739dc8250d): done. WALL 60.4 s for 10
  generations (6.04 s/gen) — the soft target is met on the wall-clock
  measure (backward 45.6 s, stress 5.6, forward 5.4).
- [x] slime-regress-verify (8d7a97da45cbc63a): done, exit 0.
- [ ] slime-i8-phases (b756631a6c3c6720): 1-generation backward phase budget
  via tests/measure_run.py --backward-profile.
- [ ] slime-i8-wall (8e1d50739dc8250d): 10-generation wall clock with
  --profile (the soft-target measurement).
- [ ] slime-regress-verify (8d7a97da45cbc63a): evolution_regression re-run
  for fresh VERIFY evidence.
All three are queued behind slime-5000gen (17f6be1c8143f710) and other
repos' jobs. Their submitted env predates the client marker, so they must
run before the canonical binaries are rebuilt with the guard (the rebuild
script's own guard check is what catches this if the order is violated).
No direct GPU process runs while any of these are pending.

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
- [ ] Approach the soft 10-generation / 60-second target with per-generation
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
  for the stencil gather. Also rejected with interleaved A/B runs:
  __launch_bounds__(256, 3) on the reforward/weight-grad/stencil kernels
  (weight-grad 2.07-2.14 -> 3.60-3.61 s, reforward 0.79-0.80 -> 1.10-1.12 s;
  the 85-register budget spills; stencil neutral). Also rejected: warp-shuffle
  reduce_inter (no shared, high occupancy) measured 12-15% faster on its own
  kernel but throttled the next compute-heavy kernel by ~800 ms (four
  interleaved pairs, unchanged control flat), a net ~0.65 s/generation loss;
  reverted. The occupancy/power coupling between adjacent backward kernels is
  now itself a measured effect. Landed win: the fused 48-slot stress
  launch (two 24-block launches -> one 48-block launch), stress 0.84 -> 0.74
  s/generation with the FD suite green.

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
  -- INVALID RESULT, RESUBMITTED: job 17f6be1c8143f710 reported
  done/PASS 5000/5000 in 443 s, but its log shows every chunk after the
  first resumed at generation 50 and ran zero generations: the harness
  passed the chunk size where the binary's N is the total target, and the
  harness never verified the checkpoint. Both are fixed (cumulative target
  + checkpoint-generation check). The authoritative resubmission is job
  f0a2cf3602d1b74b (fresh checkpoint, 100 chunks of 50, ~10 h, max 900 min,
  3 retries); 9a77a4933f1a353c was cancelled by the operator resource
  floor. slime-run55 was killed at its declared 60-minute max and is
  resubmitted as job 56a6b1511c6bfcd0 (chunked, max 240 min, 3 retries).
  Resilience is enforced: the client refuses bare multi-generation runs,
  the harness resumes on start and fails only on sustained flags.
  f0a2cf3602d1b74b then failed the dispatch RAM gate (464 MiB free vs 6144
  required: declared 4096 + margin; ollama/other load was transient) and
  56a6b1511c6bfcd0 was cancelled by the operator floor again. Neither
  auto-retried (retryAfter null). Current submissions: 409c95bfe907ac33
  (5000-gen) and 9880b19caf880a75 (run55), both --ram 2048 --retries 3.
  Queue state: both queued, feasible; dispatch is held by the daemon's
  global RAM floor (6144 MiB free required, ~2.1 GiB free with OpenCode and
  system processes resident). OLLAMA_KEEP_ALIVE=0 is now set persistently
  for future ollama starts (the daemon was idle). The scheduler then
  changed (owner commits 4deb26b/c8eaccc/5ddcc8b): RAM is reclaimed at
  dispatch (largest non-protected working sets die until the claim fits;
  operator tool and shell last), a resource-blocked queue fails its front
  job loudly after bounded polls (which is what killed the previous two
  submissions), impossible claims are refused at submit, and a queue
  control surface exists. Contract fingerprint still validates. Current
  submissions: c4223d8df9634a4e (5000-gen) and bfd463ba6d5b2a49 (run55).
  Retry semantics (gpu_scheduler._maybe_retry): only nonzero exits retry;
  policy kills, operator cancellations, and dispatch refusals pass exit
  code None and never retry, so those still need manual resubmission.
  Resume-on-start makes each resubmission continue from the last chunk.
  Operator pressure explained and the bug confirmed. Terminology: the
  "operator tool" is not a repo; it is OpenCode.exe (this session) plus
  the Windows shell, named in policy.py RECLAIM_LAST_IMAGES and reclaimed
  only after every other process. The scheduler reserves 2048 MiB free
  RAM + 4096 MiB commit for it;
  after 4 strikes with no reclaimable victims it kills the running job.
  The stop was labelled (transient) and called _maybe_retry(transient=True)
  but the old _maybe_retry refused exit-None failures, so pressure-stopped
  jobs died permanently. The owner fixed it: _maybe_retry now requeues
  transient failures (pressure stops and blocked queues) with backoff and
  bounded retries. Observed contention: the operator floor cancels slime long runs while
  its own jobs (priority -21/-22) run, and the global 6144 MiB free-RAM
  gate refuses dispatch; chunks were reduced to 25 (5000-gen) and 10
  (run55) so each preemption costs less and progress accumulates. History: 233fc0ed8322d3ee (contention), 90f2447f11941fd2 (warmup),
  17f6be1c8143f710 attempt 0 (watchdog), attempt 1 (invalid PASS),
  50-generation chunks, ~10 h). The first submission (233fc0ed8322d3ee)
  failed after 583 s while a competing direct run held the GPU; the
  resubmission raced an owner edit of gpu_scheduler.py and the client
  refused loudly, then succeeded via the owner CLI (the client path should
  be retried first next time).
  50-generation chunks, ~10 h); on completion check the harness exit and
  record the manifest after I9 flips implemented
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
  as a robust signal (`A601.trust-weight-composition`). Design closed in
  the blueprint: four bounded factors (correlation, calibration error
  against the reference's uncertainty bound, held-out error, ensemble
  diversity), second regressor head as the exp-scaled clamped bound,
  surprise read in bound units with an over-bound discount and flag.
  Implementation: host composition function plus wiring, then the GPU
  validation run and its manifest.
- [ ] Execute the preregistered experiments (E1-E5) and record manifests;
  update each protocol's status in `architecture/experiments.yaml`.
- [ ] Property witnesses: C4 genome fieldwise perturbation matrix, C5
  checkpoint continuation equivalence, C6 PT permutation test, C7 phase
  trace checked against the declared phase model.
- [x] Gate mutation testing: the transaction-move and CUDA-check mutations
  are architecture tests; the role-canonicalization mutation is
  tests/mutation_check.py (isolated worktree, mutated build, asserts
  test_canonical_role goes red — validated: Reserved10 line 316).

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

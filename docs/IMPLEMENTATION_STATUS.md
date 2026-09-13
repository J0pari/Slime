# Implementation Status

GENERATED FILE. Do not edit by hand; run `python architecture/compiler.py status`. Evidence tracker only — the binding behavior, engineering constraints, and delivery requirements are respectively in [blueprint.md](blueprint.md), [cuda_engineering.md](cuda_engineering.md), and [construction_plan.md](construction_plan.md).

<!-- architecture-status:start -->
## Build inventory

| Item | Feature | Build status |
| :--- | :------ | :----------- |
| I1 | Checkpoint and resume (S-001, cuda_engineering §14) | implemented |
| I2 | Predictor role activation (A-601, A-701) | implemented |
| I3 | Probe and placeholder completion (A-601, cuda_engineering §4.5–4.6) | implemented |
| I4 | Structural pressures (S-003) | implemented |
| I5 | Global context channel (A-203, cuda_engineering §16) | missing |
| I6 | Reaction-diffusion activation (A-202, A-103) | missing |
| I7 | Phase graphs (A-102, cuda_engineering §11) | missing |
| I8 | Performance completion (cuda_engineering, acceptance) | missing |
| I9 | Long-run hardening (S-001, Q-001) | missing |

BUILD phase incomplete: 5 of 9 inventory items are not implemented (I5, I6, I7, I8, I9). No GPU integration, acceptance, or production run is permitted, and `evidence.py record` refuses GPU manifests.

## Experiment registry

| Experiment | Title | Status | Claims |
| :--------- | :---- | :----- | :----- |
| E1 | Endogenous prediction pressure | planned | A601.ensemble-epistemic-uncertainty, A601.competence-gated-novelty |
| E2 | Shared-substrate transfer versus interference | planned | A501.role-gradient-alignment |
| E3 | Surprise homeostasis and role dynamics | planned | A601.trust-weight-composition |
| E4 | Ensemble disagreement tracks held-out error | planned | A601.ensemble-epistemic-uncertainty, A601.trust-weight-composition |
| E5 | CAME versus AdamW | planned | A501.came-production-equation |

## Claims

| Claim | Kind | Lifecycle | Mechanisms | Strong witness | Latest evidence |
| :---- | :--- | :-------- | :--------- | :------------- | :-------------- |
| A101.sot-schedule-independent | invariant | established | ✅ | ✅ | pass/stale |
| A103.checkpoint-replay-equivalence | invariant | established | ✅ | ✅ | pass/stale |
| A103.gradient-correctness | acceptance | established | ✅ | ✅ | pass/stale |
| A201.bounded-residual-dynamics | empirical | established | ✅ | ✅ | pass/stale |
| A201.role-canonicalization | invariant | established | ✅ | ✅ | pass/stale |
| A201.shared-substrate | invariant | provisional | ✅ | ❌ | pass/stale |
| A201.task-conditioning-complete | contract | established | ✅ | ✅ | pass/stale |
| A202.rd-disabled-until-adjoint | contract | established | ✅ | ✅ | pass/stale |
| A301.genotype-causes-phenotype | invariant | established | ✅ | ✅ | pass/stale |
| A401.archive-genotype-attribution | contract | provisional | ✅ | ❌ | pass/stale |
| A401.bin-capacity | invariant | established | ✅ | ✅ | pass/stale |
| A401.live-statistics-exact | invariant | established | ✅ | ✅ | pass/stale |
| A401.weighted-metric-active | contract | established | ✅ | ✅ | pass/stale |
| A501.came-production-equation | contract | established | ✅ | ✅ | pass/stale |
| A501.role-gradient-alignment | empirical | provisional | ✅ | ✅ | never |
| A601.competence-gated-novelty | policy | planned | ✅ | ❌ | never |
| A601.ensemble-epistemic-uncertainty | empirical | provisional | ✅ | ❌ | never |
| A601.trust-weight-composition | capability | planned | ✅ | ❌ | never |
| C001.acceptance-evidence-current | acceptance | established | ✅ | ✅ | pass/stale |
| G100.deterministic-seed | contract | established | ✅ | ✅ | pass/stale |
| G100.named-tunables | policy | established | ✅ | ✅ | pass/stale |
| G100.single-prng | invariant | established | ✅ | ✅ | pass/stale |
| I001.phase-order | contract | established | ✅ | ✅ | pass/stale |
| I001.replay-evaluation-identity | invariant | established | ✅ | ✅ | pass/stale |
| I001.spawn-wave-unique | invariant | established | ✅ | ✅ | pass/stale |
| S001.checkpoint-roundtrip | invariant | established | ✅ | ✅ | pass/stale |
| S001.cuda-errors-fatal | contract | established | ✅ | ✅ | pass/stale |
| S002.host-authority | invariant | established | ✅ | ✅ | pass/stale |
| S002.operator-command-effective | contract | established | ✅ | ✅ | pass/stale |
| S004.pt-swap-transaction | invariant | established | ✅ | ✅ | pass/stale |

Source gates: no_ambient_rng: PASS, no_managed_memory: PASS, checked_cuda_calls: PASS, named_tunables: PASS, rd_disabled: PASS, host_authority: PASS, operator_polling: PASS, replay_before_spawn: PASS, surprise_before_spawn: PASS, schedule_host_only: PASS, numeric_policy: PASS

Claims: 30 total, 24 established, 4 provisional, 2 planned.
<!-- architecture-status:end -->

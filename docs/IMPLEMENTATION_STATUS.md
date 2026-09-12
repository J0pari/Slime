# Implementation Status

GENERATED FILE. Do not edit by hand; run `python architecture/compiler.py status`. Evidence tracker only — the binding behavior, engineering constraints, and delivery requirements are respectively in [blueprint.md](blueprint.md), [cuda_engineering.md](cuda_engineering.md), and [construction_plan.md](construction_plan.md).

<!-- architecture-status:start -->
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
| C001.acceptance-evidence-current | acceptance | established | ✅ | ✅ | pass/stale |
| G100.deterministic-seed | contract | established | ✅ | ✅ | pass/stale |
| G100.named-tunables | policy | established | ✅ | ✅ | never |
| G100.single-prng | invariant | established | ✅ | ✅ | pass/stale |
| I001.phase-order | contract | established | ✅ | ✅ | pass/stale |
| I001.replay-evaluation-identity | invariant | established | ✅ | ✅ | pass/stale |
| I001.spawn-wave-unique | invariant | established | ✅ | ✅ | pass/stale |
| S001.checkpoint-roundtrip | invariant | established | ✅ | ✅ | never |
| S001.cuda-errors-fatal | contract | established | ✅ | ✅ | pass/stale |
| S002.host-authority | invariant | established | ✅ | ✅ | pass/stale |
| S002.operator-command-effective | contract | established | ✅ | ✅ | pass/stale |
| S004.pt-swap-transaction | invariant | established | ✅ | ✅ | pass/stale |

Source gates: no_ambient_rng: PASS, no_managed_memory: PASS, checked_cuda_calls: PASS, named_tunables: PASS, rd_disabled: PASS, host_authority: PASS, operator_polling: PASS, replay_before_spawn: PASS, schedule_host_only: PASS, numeric_policy: PASS

Claims: 26 total, 24 established, 2 provisional, 0 planned.
<!-- architecture-status:end -->

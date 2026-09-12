# Implementation Status

GENERATED FILE. Do not edit by hand; run `python architecture/compiler.py status`. Evidence tracker only — the binding behavior, engineering constraints, and delivery requirements are respectively in [blueprint.md](blueprint.md), [cuda_engineering.md](cuda_engineering.md), and [construction_plan.md](construction_plan.md).

<!-- architecture-status:start -->
| Claim | Kind | Lifecycle | Mechanisms | Strong witness | Latest evidence |
| :---- | :--- | :-------- | :--------- | :------------- | :-------------- |
| A101.sot-schedule-independent | invariant | provisional | ✅ | ❌ | never |
| A103.checkpoint-replay-equivalence | invariant | established | ✅ | ✅ | pass/current |
| A103.gradient-correctness | acceptance | established | ✅ | ✅ | pass/current |
| A201.role-canonicalization | invariant | established | ✅ | ✅ | pass/current |
| A201.shared-substrate | invariant | provisional | ✅ | ❌ | pass/current |
| A201.task-conditioning-complete | contract | provisional | ✅ | ✅ | fail |
| A202.rd-disabled-until-adjoint | contract | established | ✅ | ✅ | pass/current |
| A301.genotype-causes-phenotype | invariant | established | ✅ | ✅ | pass/current |
| A401.archive-genotype-attribution | contract | provisional | ✅ | ❌ | pass/current |
| A401.bin-capacity | invariant | provisional | ✅ | ❌ | never |
| A401.live-statistics-exact | invariant | provisional | ✅ | ❌ | never |
| A501.came-production-equation | contract | established | ✅ | ✅ | pass/current |
| C001.acceptance-evidence-current | acceptance | established | ✅ | ✅ | pass/current |
| G100.deterministic-seed | contract | established | ✅ | ✅ | pass/current |
| G100.single-prng | invariant | established | ✅ | ✅ | pass/current |
| I001.phase-order | contract | established | ✅ | ✅ | pass/current |
| I001.replay-evaluation-identity | invariant | provisional | ✅ | ❌ | pass/current |
| I001.spawn-wave-unique | invariant | established | ✅ | ✅ | pass/current |
| S001.cuda-errors-fatal | contract | established | ✅ | ✅ | pass/current |
| S002.host-authority | invariant | established | ✅ | ✅ | pass/current |
| S002.operator-command-effective | contract | provisional | ✅ | ❌ | never |
| S004.pt-swap-transaction | invariant | established | ✅ | ✅ | pass/current |

Source gates: no_ambient_rng: PASS, no_managed_memory: PASS, checked_cuda_calls: WARN, named_tunables: PASS, rd_disabled: PASS, host_authority: PASS

Claims: 22 total, 14 established, 8 provisional, 0 planned.
<!-- architecture-status:end -->

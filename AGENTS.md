<!-- agent-authority: AGENTS.md -->

# AGENTS.md — Architecture Delta Protocol

This file is the single authority for agent instructions in this repository.
Any other instruction file must delegate here and carry no independent
rules; the compiler rejects undeclared root-level documents, so a second
authority cannot be introduced silently.

The three binding specifications are `docs/blueprint.md`,
`docs/cuda_engineering.md`, and `docs/construction_plan.md`. Normative
statements inside them are `@claim` blocks. `docs/IMPLEMENTATION_STATUS.md`
is generated output (`python architecture/compiler.py status`) — never edit
it by hand. `docs/arch-signals.md` is the friction log.

## Non-negotiables

- The specs are the build target: implement what they say, not a subset, not
  a simpler version, not a placeholder for later. If a dependency is not
  ready, the dependent feature is not built yet.
- No stubs, husks, empty wrappers, or dead code. If a function exists, it
  has a complete body doing what the spec says; if it cannot exist yet, it
  is removed rather than declared.
- Witnesses must fail unless the real implementation works. A test that
  passes against an empty or zero-initialized state is not a witness.
- Compile with zero warnings; fix broken tooling instead of working around
  it; never claim a result that was not executed.
- The repository is the memory: canonical specs, generated status, claims,
  TODO queue, arch-signals, evidence manifests, source annotations, and
  tests must let a fresh session reconstruct state without conversation
  history.
- Context budget is never a reason to defer, decline, or leave unstarted
  any task the work requires. A session works until forced compaction;
  partial iterations are normal and are completed by the next session from
  the repository state. Never cite context budget as a reason to stop or to
  avoid beginning work. The only valid reasons to stop are: the work is
  done, or a genuine blocker that is recorded in the repository.

## Before changing behavior, classify the change

Every behavior-changing iteration must report, at the top of the work, which
of these it is:

1. **Implementation under existing claims** — no claim changes.
2. **Evidence that an existing claim is wrong** — claim status/confidence
   changes, not the code.
3. **A genuinely missing architectural claim** — the spec must gain a new
   `@claim` block in the same change.

## Required delta report

End every behavior-changing iteration with:

```
CLAIM_DELTA
  + <claim-id>        (new claim, with kind/lifecycle/mechanisms/witnesses)
  Δ <claim-id>        (status, confidence, mechanisms, or witnesses changed)
  - <claim-id>        (claim removed or deprecated)
  none: implementation under existing claims

WITNESS_DELTA
  + <witness-file>::<test>
  Δ <witness-file>::<test>
  none: no witness changed

SPEC_DELTA
  blueprint:
  cuda_engineering:
  construction_plan:

UNCERTAINTY
  disproven:
  still_unknown:
```

## Architecture gates

Run before finishing:

- `python architecture/compiler.py check --golden` — referential integrity
  over the claim registry, code, documents, transactions, build inventory,
  and operational prose (that module owns the check list); the golden status
  file must match.
- `python architecture/source_gates.py` — the source gates (that module owns
  the gate list).
- `python -m unittest discover -s tests/architecture -v` — the guards'
  own negative tests.
- `make check`, `make autodiff-test`, `make evolution-test` — the executable
  witnesses on hardware.

## Claim block reference

```
@claim <id> <kind>            invariant|contract|capability|policy|empirical|acceptance
S <statement, may wrap>
M <path::anchor>              mechanism (repeatable)
W <path::anchor>              exercise witness
W+ <path::anchor>             strong witness: adversarial, negative, tamper,
                              boundary, counterfactual, or able to detect the
                              failure the invariant exists to prevent
P <tag>
T <lifecycle>                 planned|provisional|established|deprecated
C <confidence>                unobserved|inferred|observed|repeatedly_observed
D <claim-id>                  dependency (repeatable)
```

A witness must declare the claim it witnesses inside its own source:

```
// [claim:<claim-id>]
```

An established claim requires established dependencies and at least one
witness. A claim whose witness is only `W` cannot be marked `established` on
the strength of exercising alone.

## Organism-identity buffers

When adding organism-associated state to `World` or `OrganismTable`, annotate
the field:

```
float* d_something;   // [identity:organism] [lifetime:rollout] [crosses:pt=something]
```

and add `something` to BOTH `organism_buffers` and
`pt_swap.organism_identity` in `architecture/transactions.yaml`. The compiler
checks both directions: an annotated field missing from the registry, and a
registry entry missing its annotation, both fail `architecture-check`. State
that must move with the organism through a PT exchange is declared here, not
remembered.

## Evidence discipline

When a run exercises a claim, record a manifest. The witness must be named —
the record entry carries the provenance:

```
python architecture/evidence.py record --name <run> --binary build/coevo.exe \
  --cuda <ver> --gpu "<name>" --seed <seed> \
  --result <claim-id>=<witness-anchor>:pass|fail [--result ...]
```

The witness anchor must be registered for the claim (or be the executed
binary itself for integration-run attestation), and its file must declare
`[claim:<claim-id>]`. Manifests live in `evidence/`. The status renderer only
reads manifests — prose status claims have no effect. Editing a claim's
statement, mechanisms, or witnesses — or any mechanism/witness file — makes
prior evidence STALE until the witness reruns.

When a run executed through the training-architecture scheduler, add
`--scheduler-job <jobId>`; the manifest then links to the scheduler's result
ledger (status, exit code, duration, log file, pinned contract fingerprint).

## GPU scheduling

Every GPU task — training, verification, profiling, and timing measurement —
is submitted to the training-architecture scheduler (`gpu-scheduler/v1`) and
queued. The scheduler owns serialization and the GPU lock. There is no "wait
for a free GPU" step and no such state to wait for: submit the job
(detached), poll `inspect`, and it runs when its turn comes. If the GPU is
busy, the job queues; that is the normal path, not a blocker.

`TRAINING_ARCH_ROOT` locates the scheduler (environment only — no path is
hardcoded); the client validates the pinned contract fingerprint before
submission and refuses loudly on drift. `--direct` is the explicit escape
hatch for machines where the scheduler is unreachable; it acquires the lock
before running. Never launch a GPU process outside these two paths: a direct
process competes with a scheduled job, can kill it, and corrupts both
measurements. This is enforced, not prose: every GPU binary calls
`require_gpu_authorization` at startup and refuses to run without the
marker the client sets (`config/gpu_authorization.cuh`), and
`gate_gpu_authorization` fails any GPU binary that drops the guard.
Scheduled commands are wrapped by
`architecture/progress_wrap.py`, which emits `progress/v1` envelopes for the
daemon. Long jobs are chunked and resumable (`tests/long_run_check.py`
resumes from the checkpoint on start, so a scheduler retry or a
resource-pressure cancellation continues instead of restarting) and
declare a `--max-minutes` that covers the whole run, because the
scheduler kills a job at its declared duration. See README.md for the
commands.

## What not to do

- Do not hand-edit `docs/IMPLEMENTATION_STATUS.md`.
- Do not create a new "architecture" or "plan" markdown document; if the
  architecture is wrong, change the specs. If a note is worth keeping, use
  `docs/arch-signals.md`.
- Do not record GPU evidence while `architecture/build_status.yaml` has any
  item that is not `implemented`; `evidence.py record` refuses, and the
  construction plan's order of operations is binding.
- Do not add a PRNG, `cudaMallocManaged`, an unchecked CUDA call, a bare
  numeric tunable at a live seam, or non-null RD coefficients; the source
  gates fail the build.
- Do not claim a planned capability in present tense in README, TODO, or the
  status surface.

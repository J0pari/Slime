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

GPU work goes through the training-architecture scheduler
(`gpu-scheduler/v1`), never ad hoc. `TRAINING_ARCH_ROOT` locates the
scheduler (environment only — no path is hardcoded); the client validates
the pinned contract fingerprint before submission and refuses loudly on
drift. Scheduled jobs are covered by the scheduler's GPU lock; manual
launches must go through `python architecture/gpu_client.py run --direct`,
which acquires the lock. Scheduled commands are wrapped by
`architecture/progress_wrap.py`, which emits `progress/v1` envelopes for
the daemon. See README.md for the commands.

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

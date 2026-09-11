# AGENTS.md — Architecture Delta Protocol

The three binding specifications are `docs/blueprint.md`,
`docs/cuda_engineering.md`, and `docs/construction_plan.md`. Normative
statements inside them are `@claim` blocks. `docs/IMPLEMENTATION_STATUS.md`
is generated output (`python architecture/compiler.py status`) — never edit
it by hand. `docs/arch-signals.md` is the friction log.

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
  + tests/foo.cpp::test_x
  Δ tests/bar.cu::test_y
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

- `python architecture/compiler.py check --golden` — claim registry,
  mechanisms, bidirectional witnesses, dependency DAG, document ownership,
  capability drift, phase model, golden status file.
- `python architecture/source_gates.py` — no ambient PRNG, no
  `cudaMallocManaged`, checked CUDA calls, named tunables only, RD disabled
  until its adjoint, host authority polling.
- `python -m unittest discover -s tests/architecture -v` — the guards'
  own negative tests.
- `make check`, `make wave1-test`, `make wave2-test` — the executable
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

## Evidence discipline

When a run exercises a claim, record a manifest:

```
python architecture/evidence.py record --name <run> --binary build/coevo.exe \
  --cuda <ver> --gpu "<name>" --seed <seed> \
  --result <claim-id>:pass|fail [--result ...]
```

Manifests live in `evidence/`. The status renderer only reads manifests —
prose status claims have no effect. Editing a mechanism or witness file makes
prior evidence STALE until the witness reruns.

## What not to do

- Do not hand-edit `docs/IMPLEMENTATION_STATUS.md`.
- Do not create a new "architecture" or "plan" markdown document; if the
  architecture is wrong, change the specs. If a note is worth keeping, use
  `docs/arch-signals.md`.
- Do not add a PRNG, `cudaMallocManaged`, an unchecked CUDA call, a bare
  numeric tunable at a live seam, or non-null RD coefficients; the source
  gates fail the build.
- Do not claim a planned capability in present tense in README, TODO, or the
  status surface.

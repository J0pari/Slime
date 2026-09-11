# Slime — Co-evolving Substrate with Emergent Predictive Role

This repository implements Slime. The project’s behavior, engineering constraints,
and staged delivery requirements are defined only in the documents below. Update
those documents when debugging or integration changes the specification; keep this
README as a stable entry point.

## Binding specifications

- [Behavioral blueprint](docs/blueprint.md) — architecture, subsystem behavior,
  invariants, and quality requirements.
- [CUDA engineering specification](docs/cuda_engineering.md) — memory model,
  kernels, transfers, initialization, and execution constraints.
- [Construction plan](docs/construction_plan.md) — dependency-ordered waves and
  their acceptance criteria.

## Project navigation

- [Implementation status](docs/IMPLEMENTATION_STATUS.md) — generated evidence
  tracker; it is not a specification and must not be edited by hand.
- [Action queue](TODO.md) — prioritized implementation and verification work
  derived from the binding specifications.
- [Architecture-control layer](AGENTS.md) — the claim registry embedded in the
  specifications, source gates, evidence manifests, and the delta protocol
  every behavior change must follow.

## Build

Source `env.sh` from Git Bash to configure MSVC, CUDA, and GNU Make, then run:

```sh
make            # build the CUDA integration binary
make check      # run host-only unit tests
make clean      # remove generated build artifacts
```

Architecture gates (run before finishing any behavior change):

```sh
make architecture-check    # claim registry + golden status + source gates + negative tests
make architecture-test     # the gates' own negative tests
make architecture-report   # the generated architecture report
```

GPU verification targets and the currently required acceptance checks are defined
by the construction plan, not duplicated here.

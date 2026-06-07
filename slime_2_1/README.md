# Slime — Co-evolving Substrate with Emergent Predictive Role

Implementation of the blueprint at `docs/blueprint.md`.

A single co-evolving population shares one substrate. Every organism is a
16-channel 64×64 NCA carrying a 2-bit role tag (classifier or predictor). Roles
select an input pathway and fitness function; all other machinery (delta codec,
CAME, archive, audit, sentinels, lineage tracking, SOT pressure, hardware
off-switch) is role-blind.

A hand-coded placeholder predictor is retained as a permanent ground-truth
check. After the archive crosses 50% occupancy, a wave of predictor-role
founders is injected and the dominant surprise signal blends toward an evolved
predictor ensemble using live Pearson correlation as the blending weight.

## Status

This is early, GPU-untested code. **None of it has been compiled with nvcc or
run on a CUDA device.** Host-side math has unit coverage; everything device-side
is designed but unvalidated. The honest per-module breakdown — what is tested,
what is written-but-unverified, and what is declared-only with a
blueprint-in-place comment — lives in `docs/IMPLEMENTATION_STATUS.md`. Read that
before assuming any subsystem works.

The first real milestone is getting the tree to compile under the CUDA toolkit
and pushing one classifier organism through a forward pass without a NaN (Stage 1
of `docs/construction_plan.md`).

## Sheet Index → Source Layout

| Sheet  | Title                                                  | Source                                         |
| :----- | :----------------------------------------------------- | :--------------------------------------------- |
| G-100  | General Notes & Conventions                            | `config/constants.cuh`                         |
| A-101  | System Architecture — Global View                      | `integration/main_loop.cu`                     |
| A-102  | Execution Backend — Phase-Major CUDA Graphs            | `execution/phase_graphs.cu`                    |
| A-103  | Autodiff — Checkpointed Warp Tape (Trajectory-Aware)   | `autodiff/warp_tape.cu`                        |
| A-201  | NCA Engine, Role-Switched Input, Behavioral Trajectory | `nca/engine.cu`                                |
| A-202  | Reaction-Diffusion Field                               | `nca/reaction_diffusion.cu`                    |
| A-301  | Genome & Delta-Weight Codec (Role-Tagged)              | `genome/codec.cu`                              |
| A-401  | Soft Quality-Diversity Archive (Role-Aware)            | `archive/soft_qd_archive.cu`                   |
| A-501  | Optimizer — CAME                                       | `optimizer/came.cu`                            |
| A-601  | Predictor Role & Hybrid Surprise Signal                | `predictor/hybrid_surprise.cu`                 |
| A-701  | Problem Generator & Dual Curriculum                    | `curriculum/problem_generator.cu`              |
| S-001  | Monitoring, Checkpointing & Resilience                 | `safety/monitoring.cu`                         |
| S-002  | Safety & Alignment Architecture                        | `safety/alignment.cu`                          |
| S-003  | Structural Pressures: Audit, Sentinels, Lineage        | `safety/structural.cu`                         |
| S-004  | Parallel Tempering Ladders                             | `safety/parallel_tempering.cu`                 |
| I-001  | Assembly & Integration                                 | `integration/main_loop.cu`                     |
| C-001  | Construction Sequence                                  | `docs/construction_plan.md`                    |
| M-001  | Bill of Materials                                      | `docs/bom.md`                                  |
| Q-001  | Quality Assurance                                      | `tests/qa_red_team.cu`                         |

Every source file opens with the sheet identifier it implements. Declared-only
functions carry a blueprint-in-place comment describing what they must do — that
comment is a specification, not a claim that the function works.

## Build

```
make            # nvcc build (requires CUDA toolkit, sm_86+); unrun so far
make check      # host-only unit tests (no CUDA required)
make clean
```

`make check` exercises the math that doesn't need a GPU: SOT gate, role-balance
multipliers, hybrid surprise blending, PT swap probabilities, genome bit layout,
losses, sentinel logistic, CUSUM reset, role canonicalization, and the optimizer
confidence update. The stub headers under `tests/stubs/` satisfy
`<cuda_runtime.h>` / `<cuda_fp16.h>` during a host-only compile.

`make` (the nvcc target) has not been run; expect to fix compile errors on the
first real build.

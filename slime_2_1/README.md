# Slime 2.1 — Co-evolving Substrate with Emergent Predictive Role

Implementation scaffold for blueprint Issue 2.1 (Revision E, 2026-05-27).

A single co-evolving population shares one substrate. Every organism is a 16-channel
64×64 NCA carrying a 2-bit role tag (classifier or predictor). Roles select an
input pathway and fitness function; all other machinery (delta codec, CAME,
archive, audit, sentinels, lineage tracking, SOT pressure, hardware off-switch)
is role-blind.

The hand-coded placeholder predictor from 2.0 is retained as a permanent
ground-truth check. After the archive crosses 50% occupancy, a wave of
predictor-role founders is injected and the dominant surprise signal blends
toward an evolved predictor ensemble using live Pearson correlation as the
blending weight.

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
| C-001  | Construction Sequence                                  | `build/construction_plan.md`                   |
| M-001  | Bill of Materials                                      | `build/bom.md`                                 |
| Q-001  | Quality Assurance                                      | `tests/qa_red_team.cu`                         |

The full blueprint is reproduced at `docs/blueprint_2_1.md`. Every source file
opens with the sheet identifier it implements; deviate only with a note in that
header.

## Status

Scaffold only. Each module exposes the spec's data structures and entry points
with stubbed bodies marked `// TODO(2.1):`. Phase ordering, public APIs, and
shared types are committed so the modules can be filled in independently
following C-001.

## Build

The legacy `slime/` tree at the repository root still compiles standalone; the
2.1 scaffold lives beside it under `slime_2_1/` to keep the working baseline
unbroken during construction.

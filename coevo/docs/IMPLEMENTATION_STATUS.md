# Implementation Status

Honest accounting of what exists, so nobody mistakes a declaration for a working
subsystem. This tree has **never been compiled with nvcc or run on a GPU**. The
real test — compilation under the CUDA toolkit and execution on an NVIDIA device
— has not happened. Until it does, every device-side item below is *designed,
not validated*: it may have subtle bugs in launch geometry, shared-memory use,
synchronization, or numerics that only surface on hardware.

Three tiers:

- **Tested** — host-compilable logic with assertions in `tests/host_unit_tests.cpp`
  (`make check`). The algorithm is exercised; the CUDA wrapping around it is not.
- **Written, unverified** — a real body exists, but it is device code (or depends
  on device code) and has not been compiled or run.
- **Declared only** — a signature with no body. The accompanying comment is a
  blueprint-in-place: a specification of what the function must do, not a claim
  that it does it.

## By sheet

| Sheet | Module | Tested | Written, unverified | Declared only |
| :---- | :----- | :----- | :------------------ | :------------ |
| G-100 | `config/constants.cuh` | constants, role enum, genome bit layout | — | — |
| A-201 | `nca/engine.cu` | channel-ownership constants | seeding, `ca_step`, `forward_kernel`, `project_bmap`, perception stencils, `launch_forward` | `extract_descriptor` |
| A-202 | `nca/reaction_diffusion.cu` | — | `rd_step` (wired into `forward_kernel`, but dormant — see below), `decode_coefficients` | — |
| A-301 | `genome/codec.cu` | `mutate`, `crossover`, role/seed accessors, xorshift | — | `apply_delta`, `init_delta_from_prior` |
| A-401 | `archive/soft_qd_archive.cu` | `compose_fitness`, role multipliers, SOT gate, `surprise_ratio` | `insert` (bin + fitness replacement only) | `recompute_bins`, `apply_lineage_brake` |
| A-501 | `optimizer/came.cu` | — | `came_update`, `came_step_kernel` | `launch_came_step` |
| A-103 | `autodiff/warp_tape.cu` | `classifier_loss`, `predictor_mse_loss` | — | `launch_backward` |
| A-601 | `predictor/hybrid_surprise.cu` | Pearson r, blending, ensemble surprise, founder selection | — | `placeholder_forward`, `launch_placeholder_train` |
| A-701 | `curriculum/problem_generator.cu` | SOT Feistel permutation (bijection + inverse), probe-set keyed signature/verify | — | `assemble_classifier_batch`, `assemble_predictor_batch` (batch assembly) |
| A-102 | `execution/phase_graphs.cu` | — | capture/build/launch/destroy (CUDA graph API) | — |
| S-001 | `safety/monitoring.cu` | `cusum_update`, checkpoint header + schema hash | — | full-payload `write_checkpoint`/`load_checkpoint` |
| S-002 | `safety/alignment.cu` | — | — | `apply_sot_identity`, `poll_off_switch`, `apply_operator_command` |
| S-003 | `safety/structural.cu` | sentinel score/train/logistic, `runaway_detected`, `l_role_collapse` | — | `run_audit_cycle`, `refresh_probe_panel`, `update_lineage_stats`, `launch_sentinel_score` |
| S-004 | `safety/parallel_tempering.cu` | `improvement_rate`, swap probability, `update_beta` | — | `record_best_fitness`, `propose_swaps`, `refresh_stress_slots`, `evaluate_stress`, `flag_stress_failures` |
| I-001 | `integration/main_loop.cu`, `host_main.cu` | — | host driver skeleton (surprise/CUSUM/β wiring); phase launches commented out | — |
| Q-001 | `tests/qa_red_team.cu` | attack-class table | — | all `run_case` / unit / integ / longrun / perf entry points |

## What "Written, unverified" specifically risks

- **`ca_step` / `forward_kernel`**: block-to-organism mapping, fully-unrolled
  48- and 32-wide inner loops (register pressure — likely spills), `__syncthreads()`
  placement, and the double-buffer parity all assume a 16×16 block and 64 even CA
  steps. Not checked by a compiler yet.
- **`project_bmap`**: per-cell `atomicAdd` into shared memory across 64×64×16
  elements — correct but possibly slow; throughput is a hardware question.
- **CUDA graph capture** (`execution/phase_graphs.cu`): the capture/instantiate
  calls are real API usage but capture nothing yet — the kernel sequences inside
  each phase are not populated.

## The chemical field is dormant (honest gap)

`rd_step` is now wired into `forward_kernel` and `ca_step` correctly leaves the
chemical channels 0–5 to it (A-202). But both seeding paths zero the chemical
channels, and `rd_step` has no source term — linear diffusion + linear reaction
of an all-zero field stays zero. So the chemical/reaction-diffusion subsystem,
though present and correct, contributes **nothing** to behavior as currently
specified: there is no mechanism that ever writes a non-zero value into channels
0–5. This is a real blueprint gap, not a bug in the wiring. Until the spec
defines a chemical seed or source, RD is a structurally-present no-op, and the
forward smoke test runs it with `coeffs=null` because a non-null array would
make no observable difference.

## First GPU build target

`make forward-smoke` (needs nvcc + an NVIDIA GPU) builds and runs
`tests/forward_smoke.cu`: one classifier organism, one forward pass, assert the
BTRAJ is finite and non-constant. It has **not been compiled** — expect to fix
errors on the first real build. This is the smallest concrete step toward Stage
1 of `construction_plan.md` and the first point at which the A-201 forward path
can move from "written, unverified" toward "tested".

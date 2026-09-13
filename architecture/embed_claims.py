"""One-shot: embed the initial claim blocks into the specification documents.

Inserts @claim blocks immediately after the sheet/section headers named below.
Idempotent-ish: refuses to insert a claim id that already exists in the file.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

CLAIM_TEMPLATES = {
    "G100.single-prng": """@claim G100.single-prng invariant
S every outcome-affecting random draw uses PCG32; no second PRNG affects
  execution
M config/constants.cuh::Pcg32
M config/constants.cuh::pcg32_random
M architecture/source_gates.py::gate_no_ambient_rng
W+ tests/architecture/test_architecture.py::test_gate_no_ambient_rng_catches_plant
P reproducibility
T established
C repeatedly_observed
""",
    "G100.deterministic-seed": """@claim G100.deterministic-seed contract
S the same PCG32 seed and stream produce the same draw sequence on host and
  device
M config/constants.cuh::pcg32_seed
W+ tests/host_unit_tests.cpp::test_pcg32_determinism
P reproducibility
T established
C observed
""",
    "A101.sot-schedule-independent": """@claim A101.sot-schedule-independent invariant
S GPU-resident state does not influence the SOT/probe schedule or pruning
  commands; the schedule is host-side, seeded, and reproducible
M curriculum/problem_generator.cu::assemble_classifier_batch
M integration/host_main.cu::step_generation
P safety
T provisional
C inferred
""",
    "A103.checkpoint-replay-equivalence": """@claim A103.checkpoint-replay-equivalence invariant
S forward_with_checkpoints reproduces the plain forward bitwise and replayed
  segments reconstruct checkpointed states
M autodiff/warp_tape.cu::forward_with_checkpoints_kernel
M autodiff/warp_tape.cu::bwd_reforward_step_kernel
W+ tests/autodiff_acceptance.cu::test_forward_match_and_backward
T established
C observed
""",
    "A103.gradient-correctness": """@claim A103.gradient-correctness acceptance
S the analytic backward agrees with directional finite differences of the
  executed forward, and gradients remain finite
M autodiff/warp_tape.cu::launch_backward_all
W+ tests/evolution_regression.cu::test_finite_difference_gradient
D A103.checkpoint-replay-equivalence
T established
C observed
""",
    "A201.role-canonicalization": """@claim A201.role-canonicalization invariant
S reserved role codes canonicalize to a defined role everywhere host and
  device role-switched logic runs, so pathway choice is total
M config/constants.cuh::canonical_role
W+ tests/host_unit_tests.cpp::test_canonical_role
T established
C repeatedly_observed
""",
    "A201.shared-substrate": """@claim A201.shared-substrate invariant
S classifier and predictor roles execute the same NCA architecture, delta
  codec, optimizer, archive, audit, and safety machinery; role changes only
  input wiring and fitness semantics
M nca/engine.cu::forward_one
M genome/codec.cu::read_role
M integration/host_main.cu::step_generation
W tests/autodiff_acceptance.cu::test_forward_match_and_backward
T provisional
C observed
""",
    "A202.rd-disabled-until-adjoint": """@claim A202.rd-disabled-until-adjoint contract
S reaction-diffusion stays disabled in the active loop until the RD adjoint
  and checkpoint replay exist; the forward receives null coefficients
M autodiff/warp_tape.cu::bwd_reforward_step_kernel
M architecture/source_gates.py::gate_rd_disabled
W+ tests/architecture/test_architecture.py::test_gate_rd_disabled_catches_nonnull
T established
C observed
""",
    "A301.genotype-causes-phenotype": """@claim A301.genotype-causes-phenotype invariant
S active genome differences can alter the executed phenotype while shared
  weights and inputs are held fixed
M genome/codec.cu::apply_delta
M autodiff/warp_tape.cu::materialize_effective_weights_kernel
M integration/host_main.cu::step_generation
W+ tests/evolution_regression.cu::test_genotype_causality
T established
C observed
""",
    "A401.archive-genotype-attribution": """@claim A401.archive-genotype-attribution contract
S the archive stores the genome that produced the archived descriptor and
  fitness, and parent selection draws from those stored genomes
M integration/host_main.cu::insert_into_archive
M integration/host_main.cu::spawn_wave
W tests/evolution_regression.cu::test_genotype_causality
D A301.genotype-causes-phenotype
T provisional
C inferred
""",
    "A401.bin-capacity": """@claim A401.bin-capacity invariant
S no archive bin holds more than its per-role capacity after any insertion
  or PCA rebin
M archive/soft_qd_archive.cu::insert
M archive/soft_qd_archive.cu::recompute_bins
T provisional
C unobserved
""",
    "A401.live-statistics-exact": """@claim A401.live-statistics-exact invariant
S per-role counts, live lists, and RFF means exactly describe the alive
  archive after every insertion, replacement, and rebin
M archive/soft_qd_archive.cu::insert
M archive/soft_qd_archive.cu::recompute_bins
T provisional
C unobserved
""",
    "A501.came-production-equation": """@claim A501.came-production-equation contract
S the device CAME step implements confidence = 1/(1+c) with
  w -= lr*confidence*u + weight_decay*w, sharing one scalar equation with
  the host test reference
M optimizer/came_math.cuh::came_step_scalar
M optimizer/came.cu::came_step_kernel
W+ tests/host_unit_tests.cpp::test_came_production_equation
T established
C observed
""",
    "S001.cuda-errors-fatal": """@claim S001.cuda-errors-fatal contract
S any CUDA allocation, copy, launch, or synchronization failure invalidates
  the run; the loop aborts instead of continuing with corrupted state
M integration/host_main.cu::CUDA_ABORT
M integration/host_main.cu::phase_trace
M architecture/source_gates.py::gate_checked_cuda_calls
W+ tests/architecture/test_architecture.py::test_gate_checked_cuda_calls_catches_plant
T established
C observed
""",
    "S002.host-authority": """@claim S002.host-authority invariant
S the host polls the off-switch before every generation and owns SOT keys,
  probe signatures, and operator commands; no evolved state controls them
M safety/alignment.cu::poll_off_switch
M architecture/source_gates.py::gate_host_authority
W+ tests/architecture/test_architecture.py::test_gate_host_authority_catches_plant
T established
C observed
""",
    "S002.operator-command-effective": """@claim S002.operator-command-effective contract
S pause, resume, prune, and checkpoint commands cause durable state
  transitions that the run loop actually applies
M safety/alignment.cu::apply_operator_command
T provisional
C unobserved
""",
    "S004.pt-swap-transaction": """@claim S004.pt-swap-transaction invariant
S an accepted PT swap moves every organism-associated rollout state together:
  device state, checkpoints, genome, delta, role, lineage, batch assignment,
  and seed gradients
M safety/parallel_tempering.cu::propose_swaps
M safety/parallel_tempering.cu::swap_host_organism
M architecture/transactions.yaml::pt_swap
W+ tests/evolution_regression.cu::test_forced_pt_swap
T established
C observed
""",
    "I001.spawn-wave-unique": """@claim I001.spawn-wave-unique invariant
S each spawn wave installs WAVE_SIZE distinct offspring into distinct pool
  slots before any of them is re-evaluated
M genome/codec.cu::select_spawn_victims
M integration/host_main.cu::spawn_wave
W+ tests/host_unit_tests.cpp::test_spawn_victims_unique
T established
C observed
""",
    "I001.replay-evaluation-identity": """@claim I001.replay-evaluation-identity invariant
S replay tuples and generation telemetry are recorded from the evaluated
  population state before spawning replaces pool members
M integration/host_main.cu::step_generation
T provisional
C inferred
""",
    "I001.phase-order": """@claim I001.phase-order contract
S the generation loop follows the executable phase order with the seed-
  gradient transfer after the PT exchange and before backward
M integration/host_main.cu::step_generation
M architecture/transactions.yaml::generation_phases
W+ tests/architecture/test_architecture.py::test_phase_model_rejects_seed_before_pt
T established
C observed
""",
    "C001.acceptance-evidence-current": """@claim C001.acceptance-evidence-current acceptance
S an acceptance result only establishes a claim while the mechanisms and
  witnesses it exercised are unchanged; changed code reopens the claim until
  fresh evidence is recorded
M architecture/evidence.py::verdict
W+ tests/architecture/test_architecture.py::test_stale_evidence_detected
T established
C observed
""",
}

# (file, line-anchor prefix, claims) — the claim block is inserted directly
# after the first line starting with the anchor prefix.
PLAN = [
    ("docs/blueprint.md", "G-100: General Notes", ["G100.single-prng", "G100.deterministic-seed"]),
    ("docs/blueprint.md", "A-101: System Architecture", ["A101.sot-schedule-independent"]),
    ("docs/blueprint.md", "A-103: Autodiff", ["A103.checkpoint-replay-equivalence", "A103.gradient-correctness"]),
    ("docs/blueprint.md", "A-201: NCA Engine", ["A201.role-canonicalization", "A201.shared-substrate"]),
    ("docs/blueprint.md", "A-202: Reaction-Diffusion", ["A202.rd-disabled-until-adjoint"]),
    ("docs/blueprint.md", "A-301: Genome", ["A301.genotype-causes-phenotype"]),
    ("docs/blueprint.md", "A-401: Soft Quality", ["A401.archive-genotype-attribution", "A401.bin-capacity", "A401.live-statistics-exact"]),
    ("docs/blueprint.md", "A-501: Optimizer", ["A501.came-production-equation"]),
    ("docs/blueprint.md", "S-002: Safety", ["S002.host-authority", "S002.operator-command-effective"]),
    ("docs/blueprint.md", "S-004: Parallel Tempering", ["S004.pt-swap-transaction"]),
    ("docs/blueprint.md", "I-001: Assembly", ["I001.spawn-wave-unique", "I001.replay-evaluation-identity", "I001.phase-order"]),
    ("docs/blueprint.md", "C-001: Construction", ["C001.acceptance-evidence-current"]),
    ("docs/cuda_engineering.md", "## 12. Safety/Alignment", ["S001.cuda-errors-fatal"]),
]


def main() -> int:
    for path, anchor, ids in PLAN:
        p = ROOT / path
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        insert_at = None
        for i, line in enumerate(lines):
            if line.startswith(anchor):
                insert_at = i
                break
        if insert_at is None:
            print(f"anchor not found: {path} {anchor}")
            return 1
        for cid in ids:
            if f"@claim {cid} " in text:
                print(f"already present: {cid}")
                continue
            block = CLAIM_TEMPLATES[cid].rstrip("\n")
            lines.insert(insert_at + 1, "")
            lines.insert(insert_at + 1, block)
            insert_at += 2
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"embedded {len(ids)} claims into {path} after '{anchor}'")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

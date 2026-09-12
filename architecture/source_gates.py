"""Static source gates for Slime (architecture-control layer).

Each gate scans production source (everything except tests/, build/,
architecture/, evidence/, .claude/, .vscode/, .continue/) for a forbidden
pattern and reports findings. A finding is an ERROR unless it is explicitly
allowlisted (reported as WARNING). `--strict` promotes allowlist warnings to
errors so the remaining allowlist shrinks over time.

Gates are imported as functions by tests/architecture/test_architecture.py,
which plants violations in temporary trees to prove each gate catches them
(a guard with no negative test is itself only an assumption).

Production sources are also exported for the CUDA-call checker's wrapper
allowlist: wrappers declared here are treated as approved CUDA call sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SRC_SUFFIXES = {".cu", ".cuh", ".cpp", ".h", ".hpp"}
EXCLUDED_DIRS = {"tests", "build", "architecture", "evidence", ".claude", ".vscode", ".continue", ".git"}
EXCLUDED_FILES = {"env.sh"}

# Approved wrappers around raw CUDA calls. The remaining production call sites
# must route through checked wrappers (CUDA_CHECK / CUDA_ABORT) or appear here
# with a justification. Open items are tracked in TODO.md; `--strict` fails on
# any allowlisted hit so the list cannot grow silently.
CUDA_WRAPPER_ALLOWLIST = {
    "autodiff/warp_tape.cu": {
        "allocate_checkpoints", "free_checkpoints", "allocate_grad_buffers",
        "free_grad_buffers", "allocate_backward_workspace", "free_backward_workspace",
    },
    "optimizer/came.cu": {
        "allocate_came", "free_came",
        # TODO(architecture): convert these launcher memsets to a checked
        # wrapper instead of relying on the following phase_trace sync.
        "launch_grad_norm_reduce", "launch_telemetry_kernels",
    },
    "integration/host_main.cu": {"free_gpu_buffers"},
    "safety/parallel_tempering.cu": {
        "swap_device_organism",
        # TODO(architecture): give propose_swaps an error channel; today its
        # stream synchronize surfaces at the caller's next phase_trace.
        "propose_swaps",
    },
    "safety/alignment.cu": {"apply_sot_identity"},
    "safety/monitoring.cu": {"collect_cuda_diagnostics", "benchmark_cuda_transfers", "emit_cuda_diagnostics"},
    "nca/engine.cu": {},   # launchers only (no raw runtime calls expected)
    "nca/reaction_diffusion.cu": {},
    "genome/codec.cu": {},
    "curriculum/problem_generator.cu": {},
    "archive/soft_qd_archive.cu": {},
    "integration/main_loop.cu": {},
    "config/constants.cuh": {},
}

CUDA_CALL_PATTERN = re.compile(
    r"\b(cudaMalloc|cudaMallocHost|cudaMemcpy|cudaMemcpyAsync|cudaMemset|cudaMemsetAsync"
    r"|cudaStreamSynchronize|cudaStreamCreate|cudaFree|cudaFreeHost)\s*\("
)
CHECKED_CONTEXT = re.compile(r"CUDA_CHECK|CUDA_ABORT|TRANSFER_ABORT|cuda_diagnostics_ok|_err|cudaError_t")

ALLOWLISTED_CTX = {
    "cudaGetLastError", "cudaGetErrorString", "cudaDeviceSynchronize",
}


@dataclass
class Finding:
    gate: str
    path: str
    line: int
    text: str
    severity: str = "error"   # "error" | "warning"

    def __str__(self) -> str:
        return f"{self.gate}: {self.path}:{self.line}: {self.text.strip()} [{self.severity}]"


@dataclass
class GateReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def iter_production_sources(root: Path):
    for path in sorted(root.rglob("*")):
        if path.suffix not in SRC_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if rel.parts[0] in EXCLUDED_DIRS or rel.name in EXCLUDED_FILES:
            continue
        yield rel


def source_lines(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for rel in iter_production_sources(root):
        p = root / rel
        try:
            out[str(rel).replace("\\", "/")] = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
    return out


# ---- Gate: no ambient PRNG -------------------------------------------------
AMBIENT_RNG_PATTERNS = [
    re.compile(r"\brand\s*\("),
    re.compile(r"\bsrand\s*\("),
    re.compile(r"curand", re.IGNORECASE),
    re.compile(r"std::random"),
    re.compile(r"mt19937"),
    re.compile(r"\bxorshift\b", re.IGNORECASE),
]


def gate_no_ambient_rng(files: dict[str, list[str]], report: GateReport) -> None:
    for path, lines in files.items():
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                continue
            for pat in AMBIENT_RNG_PATTERNS:
                if pat.search(line):
                    report.findings.append(
                        Finding("no_ambient_rng", path, i, line))
                    break


# ---- Gate: no cudaMallocManaged -------------------------------------------
def gate_no_managed_memory(files: dict[str, list[str]], report: GateReport) -> None:
    for path, lines in files.items():
        for i, line in enumerate(lines, 1):
            if "cudaMallocManaged" in line:
                report.findings.append(Finding("no_managed_memory", path, i, line))


# ---- Gate: CUDA calls must be checked -------------------------------------
def gate_checked_cuda_calls(files: dict[str, list[str]], report: GateReport,
                            strict: bool = False) -> None:
    for path, lines in files.items():
        allow = CUDA_WRAPPER_ALLOWLIST.get(path)
        depth = 0              # global brace depth (namespace braces keep it > 0)
        allow_depth: int | None = None   # brace depth at allowlist entry
        entered_body = False   # have we crossed the function's opening brace
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if allow and allow_depth is None:
                for fn in allow:
                    if re.search(r"\b" + re.escape(fn) + r"\s*\(", line):
                        allow_depth = depth
                        entered_body = False
                        break
            if CHECKED_CONTEXT.search(line):
                depth += line.count("{") - line.count("}")
                if allow_depth is not None:
                    if depth > allow_depth:
                        entered_body = True
                    elif entered_body and depth <= allow_depth:
                        allow_depth = None
                continue
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                continue
            for m in CUDA_CALL_PATTERN.finditer(line):
                fn = m.group(1)
                if fn in ALLOWLISTED_CTX:
                    continue
                severity = "error"
                if allow_depth is not None:
                    severity = "warning" if not strict else "error"
                report.findings.append(
                    Finding("checked_cuda_calls", path, i, line, severity))
            depth += line.count("{") - line.count("}")
            if allow_depth is not None:
                if depth > allow_depth:
                    entered_body = True
                elif entered_body and depth <= allow_depth:
                    allow_depth = None


# ---- Gate: named tunables only --------------------------------------------
# Live-seam numerics must be named constants. Flags a modulo-literal when it
# is compared against a literal (`generation % 50 == 0`) or when the left
# operand is a generation/step-like name. Shape arithmetic (`k % 3`, bit
# offsets `start % 32`) and printf format specifiers are not live seams.
TUNABLE_MOD_RE = re.compile(r"%\s*\d+")
TUNABLE_LEFT_RE = re.compile(r"([A-Za-z_]\w*)\s*$")
TUNABLE_LEFT_NAMES = {"gen", "generation", "step", "epoch", "iter", "iteration",
                      "interval", "n_gen"}


def gate_named_tunables(files: dict[str, list[str]], report: GateReport) -> None:
    for path, lines in files.items():
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                continue
            for m in TUNABLE_MOD_RE.finditer(line):
                after = line[m.end():]
                if after.lstrip().startswith("=="):
                    report.findings.append(Finding("named_tunables", path, i, line))
                    break
                left = line[:m.start()]
                lm = TUNABLE_LEFT_RE.search(left)
                if lm and lm.group(1) in TUNABLE_LEFT_NAMES:
                    # Not a printf-style format string ("gen %4d").
                    name_start = lm.start(1)
                    if name_start > 0 and left[name_start - 1] in ('"', "'"):
                        continue
                    report.findings.append(Finding("named_tunables", path, i, line))
                    break


# ---- Gate: reaction-diffusion stays disabled until its adjoint exists -----
RD_LAUNCHERS = ("launch_forward_with_checkpoints", "launch_forward_effective", "launch_forward")


def _call_args(line: str, fn: str) -> list[str] | None:
    idx = line.find(fn + "(")
    if idx < 0:
        return None
    start = idx + len(fn) + 1
    depth = 1
    args: list[str] = []
    cur = ""
    for ch in line[start:]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                args.append(cur)
                break
        if ch == "," and depth == 1:
            args.append(cur)
            cur = ""
            continue
        cur += ch
    return [a.strip() for a in args]


def gate_rd_disabled(files: dict[str, list[str]], report: GateReport) -> None:
    for path, lines in files.items():
        for i, line in enumerate(lines, 1):
            for fn in RD_LAUNCHERS:
                if fn + "(" not in line:
                    continue
                args = _call_args(line, fn)
                if args is None or len(args) < 4:
                    continue
                if args[2] != "nullptr":
                    report.findings.append(
                        Finding("rd_disabled", path, i, line))


# ---- Gate: host authority polls the off-switch in the run loop ------------
def gate_host_authority(files: dict[str, list[str]], report: GateReport) -> None:
    text = "\n".join(files.get("integration/host_main.cu", []))
    if "poll_off_switch" not in text:
        report.findings.append(
            Finding("host_authority", "integration/host_main.cu", 0,
                    "run() no longer polls poll_off_switch()"))


# ---- Gate: operator commands are polled and applied durably in run() ------
def gate_operator_polling(files: dict[str, list[str]], report: GateReport) -> None:
    text = "\n".join(files.get("integration/host_main.cu", []))
    if "poll_operator_commands(w)" not in text:
        report.findings.append(
            Finding("operator_polling", "integration/host_main.cu", 0,
                    "run() no longer polls the operator command file"))
    if "operator_state.paused" not in text:
        report.findings.append(
            Finding("operator_polling", "integration/host_main.cu", 0,
                    "run() no longer applies the durable paused state"))


# ---- Gate: replay tuples are recorded BEFORE the spawn wave ---------------
def gate_replay_before_spawn(files: dict[str, list[str]], report: GateReport) -> None:
    lines = files.get("integration/host_main.cu", [])
    replay_line = -1
    spawn_line = -1
    for i, line in enumerate(lines, 1):
        if "replay_buffer_push(" in line and "//" not in line.split("replay_buffer_push(")[0]:
            if replay_line < 0:
                replay_line = i
        if "spawn_wave(w);" in line and spawn_line < 0:
            spawn_line = i
    if replay_line < 0:
        report.findings.append(
            Finding("replay_before_spawn", "integration/host_main.cu", 0,
                    "replay buffer push call missing"))
    elif spawn_line < 0:
        report.findings.append(
            Finding("replay_before_spawn", "integration/host_main.cu", 0,
                    "spawn_wave call missing"))
    elif replay_line > spawn_line:
        report.findings.append(
            Finding("replay_before_spawn", "integration/host_main.cu", replay_line,
                    "replay tuples are recorded after the spawn wave (pre-spawn "
                    "evaluation identity violated)"))


# ---- Gate: the SOT/probe schedule is computed host-side only --------------
# Flags device-memory access or device-execution tokens in the schedule file.
# __host__ __device__ annotations on pure helpers (e.g. the Feistel
# permutation) are permitted: they do not touch device state.
SCHEDULE_DEVICE_TOKENS = re.compile(
    r"cudaMalloc|cudaMemcpy|cudaMemset|cudaFree|cudaStream|<<<|__global__"
    r"|atomicAdd|__shared__|cudaDeviceSynchronize|cudaGetLastError")
SCHEDULE_EXEMPT_LINE = re.compile(r"#include\s+[<\"]cuda_runtime\.h")


def gate_schedule_host_only(files: dict[str, list[str]], report: GateReport) -> None:
    path = "curriculum/problem_generator.cu"
    lines = files.get(path, [])
    for i, line in enumerate(lines, 1):
        if SCHEDULE_EXEMPT_LINE.search(line):
            continue
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        if SCHEDULE_DEVICE_TOKENS.search(line):
            report.findings.append(
                Finding("schedule_host_only", path, i, line))


ALL_GATES = [
    gate_no_ambient_rng,
    gate_no_managed_memory,
    gate_checked_cuda_calls,
    gate_named_tunables,
    gate_rd_disabled,
    gate_host_authority,
    gate_operator_polling,
    gate_replay_before_spawn,
    gate_schedule_host_only,
]

GATE_NAMES = [g.__name__.replace("gate_", "") for g in ALL_GATES]


def run_gates(root: Path, strict: bool = False) -> GateReport:
    report = GateReport()
    files = source_lines(root)
    for gate in ALL_GATES:
        if gate is gate_checked_cuda_calls:
            gate(files, report, strict=strict)
        else:
            gate(files, report)
    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Slime static source gates")
    parser.add_argument("--root", default=".")
    parser.add_argument("--strict", action="store_true",
                        help="promote allowlisted CUDA-call warnings to errors")
    args = parser.parse_args()
    report = run_gates(Path(args.root), strict=args.strict)
    for f in report.findings:
        print(f)
    print(f"\nsource gates: {len(report.errors)} errors, {len(report.warnings)} warnings")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

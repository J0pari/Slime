"""Static source gates for Slime (architecture-control layer).

Each gate scans production source (everything except tests/, build/,
architecture/, evidence/, and hidden tool directories) for a forbidden
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
EXCLUDED_DIRS = {"tests", "build", "architecture", "evidence", ".git"}
EXCLUDED_FILES = {"env.sh"}

# Approved wrappers around raw CUDA calls. Every production call site routes
# through checked wrappers (CUDA_CHECK / CUDA_ABORT / TRANSFER_ABORT /
# CUDA_WARN / cuda_diagnostics_ok) or the bool-returning allocation helpers
# in autodiff/warp_tape.cu and optimizer/came.cu. The map is empty: any new
# raw call outside a checked context is an ERROR even without --strict.
CUDA_WRAPPER_ALLOWLIST = {}

CUDA_CALL_PATTERN = re.compile(
    r"\b(cudaMalloc|cudaMallocHost|cudaMemcpy|cudaMemcpyAsync|cudaMemset|cudaMemsetAsync"
    r"|cudaStreamSynchronize|cudaStreamCreate|cudaFree|cudaFreeHost)\s*\("
)
CHECKED_CONTEXT = re.compile(
    r"CUDA_CHECK|CUDA_ABORT|TRANSFER_ABORT|CUDA_WARN|cuda_diagnostics_ok"
    r"|cudaSuccess|_err|_ce|cudaError_t")

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
        if (rel.parts[0] in EXCLUDED_DIRS or rel.parts[0].startswith(".")
                or rel.name in EXCLUDED_FILES):
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


# ---- Gate: reaction-diffusion may be enabled only with its adjoint --------
# RD is enabled when the forward carries per-organism coefficients. Whenever
# the coefficients plumbing is present, the backward must re-forward RD,
# produce the clamp-aware d_next workspace, and run the RD gather; otherwise
# gradients are silently biased (the failure the old rd_disabled gate
# prevented by banning coefficients outright).
RD_ADJOINT_MARKERS = (
    "rd_step(rc, rn, coeffs[org])",
    "bwd_rd_gather_kernel",
    "d_rd_g",
)


def gate_rd_adjoint_present(files: dict[str, list[str]],
                            report: GateReport) -> None:
    host = " ".join(" ".join(files.get("integration/host_main.cu", [])).split())
    if "d_rd_coeffs" not in host:
        return  # reaction-diffusion plumbing absent: nothing to require
    tape = " ".join(files.get("autodiff/warp_tape.cu", []))
    for marker in RD_ADJOINT_MARKERS:
        if marker not in tape:
            report.findings.append(
                Finding("rd_adjoint_present", "autodiff/warp_tape.cu", 0,
                        "reaction-diffusion is enabled but the backward is "
                        f"missing {marker}"))


# ---- Gate: no cross-repo bridge code before admission ----------------------
# The extension gate in construction_plan.md is closed until every inventory
# item is implemented and the experimental program has evidence. Until an
# admitted inventory item exists, no production source may reference the
# external contract; documentation and configuration records are separate
# files and are not scanned here.
BRIDGE_TOKENS = ("adaptive-ecology", "adaptive_ecology", "ecology.ndjson")


def gate_no_bridge_code(files: dict[str, list[str]],
                        report: GateReport) -> None:
    for path, lines in files.items():
        for i, line in enumerate(lines, 1):
            for token in BRIDGE_TOKENS:
                if token in line:
                    report.findings.append(
                        Finding("no_bridge_code", path, i, line))


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


# ---- Gate: surprise reads the evaluated pre-spawn population --------------
# The predictor ensemble ranks and filters organisms by role and fitness.
# Those fields are mutated by the spawn wave, while the descriptors it
# consumes describe the evaluated population, so the surprise computation
# must precede spawn_wave.
def gate_surprise_before_spawn(files: dict[str, list[str]],
                               report: GateReport) -> None:
    lines = files.get("integration/host_main.cu", [])
    surprise_line = -1
    spawn_line = -1
    for i, line in enumerate(lines, 1):
        if "evaluate_probe_reference(w)" in line and surprise_line < 0:
            surprise_line = i
        if "spawn_wave(w);" in line and spawn_line < 0:
            spawn_line = i
    if surprise_line < 0:
        report.findings.append(
            Finding("surprise_before_spawn", "integration/host_main.cu", 0,
                    "reference surprise call missing"))
    elif spawn_line < 0:
        report.findings.append(
            Finding("surprise_before_spawn", "integration/host_main.cu", 0,
                    "spawn_wave call missing"))
    elif surprise_line > spawn_line:
        report.findings.append(
            Finding("surprise_before_spawn", "integration/host_main.cu",
                    surprise_line,
                    "surprise is computed after the spawn wave (pre-spawn "
                    "population identity violated)"))


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


# ---- Gate: numeric policy (schema home = config/constants.cuh) -------------
# The LLM-Trader pattern adapted to C++/CUDA: the schema home is the ONLY
# place for numbers; action code uses named constants. The declared-constant
# registry is DERIVED from the schema home, so adding a constant there
# automatically makes bare uses of its value suspect elsewhere. Rules:
#   N1  numeric constexpr/const definitions outside the schema home
#   N2  comparisons against a non-structural numeric literal
#   N3  ternary fallback to a non-structural numeric literal
#   N4  default-argument numeric literals
#   N5  modulo against a numeric literal (named tunables only)
# Exemptions are review decisions with reasons; the liveness test fails when
# an exemption no longer names real code.
SCHEMA_HOME = "config/constants.cuh"
NUMERIC_STRUCTURAL = {0.0, 1.0, -1.0, 2.0}

# Review-decision exemptions: (path, enclosing-symbol) -> reason.
# Adding one is a review decision, never an allowance for a configurable
# number; tests/architecture/test_architecture.py::test_numeric_exemptions_live
# fails when an exemption no longer names real code.
NUMERIC_POLICY_EXEMPTIONS: dict[tuple[str, str], str] = {
    ("genome/codec.cu", "read_bits"): (
        "the bit-layout helper indexes a 32-bit word: 32 is the word width, "
        "structural to the codec's own representation"),
    ("nca/rd_codec.cuh", "read_bits"): (
        "the bit-layout helper indexes a 32-bit word: 32 is the word width, "
        "structural to the codec's own representation"),
    ("nca/engine.cu", "project_bmap"): (
        "the deterministic tree reduction's NTHREADS and the unrolled "
        "16-channel loops are kernel-shape constants tied to the 16x16 "
        "block, checked by static_assert"),
    ("autodiff/warp_tape.cu", "cell_yx"): (
        "flat-cell indexing by the grid width: structural to the layout"),
}

_CONST_DEF_RE = re.compile(
    r"\b(?:constexpr|const)\s+[\w:<>]+\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")
_NUMBER_TOKEN_RE = re.compile(
    r"(?<![\w.])(?:0[xX][0-9A-Fa-f]+|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)")
_CMP_LITERAL_RE = re.compile(
    r"(?:==|!=|<=|>=|<|>)\s*(-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?f?)")
_MOD_LITERAL_RE = re.compile(
    r"%\s*(-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?f?)")
_DEFAULT_LABEL_RE = re.compile(r"^\s*default\s*:")
_TERNARY_ELSE_RE = re.compile(
    r"\?\s*[^;:]*:\s*(-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?f?)")
_TERNARY_THEN_RE = re.compile(
    r"\?\s*(-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?f?)\s*:")
_DEFAULT_ARG_RE = re.compile(
    r"\([^)]*\b[A-Za-z_]\w*\s*=\s*(-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?f?)")


def strip_code_line(line: str) -> str:
    """Remove comments and string/char literal contents for scanning."""
    out = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == '"' or ch == "'":
            quote = ch
            i += 1
            while i < n and line[i] != quote:
                if line[i] == "\\":
                    i += 1
                i += 1
            i += 1
            continue
        if ch == "/" and i + 1 < n and line[i + 1] == "/":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def _number_value(token: str) -> float | None:
    t = token.rstrip("fFuUlL")
    try:
        if t.lower().startswith("0x"):
            return float(int(t, 16))
        return float(t)
    except ValueError:
        return None


def parse_schema_constants(text: str) -> dict[str, float]:
    """{NAME: value} for numeric constants declared in the schema home."""
    out: dict[str, float] = {}
    for m in _CONST_DEF_RE.finditer(text):
        name, expr = m.group(1), m.group(2)
        tok = _NUMBER_TOKEN_RE.search(expr)
        if not tok:
            continue
        value = _number_value(tok.group(0))
        if value is not None:
            out[name] = value
    return out


def enclosing_symbol(lines: list[str], lineno: int) -> str:
    """Best-effort enclosing function name for an exemption key."""
    for i in range(lineno - 2, max(-1, lineno - 120), -1):
        line = lines[i].strip()
        if not line or line.startswith(("//", "*", "/*", "#", "}")):
            continue
        m = re.match(r"^[\w:<>\*&\s]+?\b([A-Za-z_]\w*)\s*\(", line)
        if m and not line.endswith(";"):
            return m.group(1)
    return ""


def gate_numeric_policy(files: dict[str, list[str]], report: GateReport) -> None:
    schema_text = "\n".join(files.get(SCHEMA_HOME, []))
    declared = parse_schema_constants(schema_text)
    declared_names_by_value: dict[float, list[str]] = {}
    for name, value in declared.items():
        declared_names_by_value.setdefault(value, []).append(name)

    def is_structural(value: float) -> bool:
        return value in NUMERIC_STRUCTURAL

    def hint(value: float) -> str:
        names = declared_names_by_value.get(value)
        if names:
            return f" (equals declared {names[0]})"
        return ""

    def exempt(path: str, lines: list[str], lineno: int) -> bool:
        symbol = enclosing_symbol(lines, lineno)
        return (path, symbol) in NUMERIC_POLICY_EXEMPTIONS

    for path, lines in files.items():
        if path == SCHEMA_HOME:
            continue
        for i, raw in enumerate(lines, 1):
            code = strip_code_line(raw)
            if not code.strip():
                continue
            if exempt(path, lines, i):
                continue
            # Shift operators are not comparisons: neutralise them before the
            # comparison rule (>> 6 and << 2 are layout arithmetic).
            code_noshift = code.replace("<<", " ").replace(">>", " ")
            # Iteration bounds in `for` headers are structural shape, not
            # live seams (the C++ analogue of allowed subscript arithmetic).
            is_for = re.match(r"\s*for\s*\(", code) is not None
            # N1: numeric const definitions outside the schema home.
            m = _CONST_DEF_RE.search(code)
            if m:
                tok = _NUMBER_TOKEN_RE.search(m.group(2))
                if tok:
                    value = _number_value(tok.group(0))
                    if value is not None and not is_structural(value):
                        report.findings.append(Finding(
                            "numeric_policy", path, i,
                            f"{m.group(1)} = {tok.group(0)}: numeric constants "
                            f"live in {SCHEMA_HOME}{hint(value)}"))
                        continue
            if is_for:
                continue
            # N2: comparisons against a non-structural literal.
            fired = False
            for cm in _CMP_LITERAL_RE.finditer(code_noshift):
                value = _number_value(cm.group(1))
                if value is None or is_structural(value):
                    continue
                report.findings.append(Finding(
                    "numeric_policy", path, i,
                    f"comparison against literal {cm.group(1)}{hint(value)}"))
                fired = True
                break
            if fired:
                continue
            # N3: ternary fallback to a non-structural literal.
            for tm in list(_TERNARY_ELSE_RE.finditer(code)) + \
                    list(_TERNARY_THEN_RE.finditer(code)):
                value = _number_value(tm.group(1))
                if value is None or is_structural(value):
                    continue
                report.findings.append(Finding(
                    "numeric_policy", path, i,
                    f"ternary fallback literal {tm.group(1)}{hint(value)}"))
                fired = True
                break
            if fired:
                continue
            # N4: default-argument literals.
            for dm in _DEFAULT_ARG_RE.finditer(code):
                value = _number_value(dm.group(1))
                if value is None or is_structural(value):
                    continue
                report.findings.append(Finding(
                    "numeric_policy", path, i,
                    f"default-argument literal {dm.group(1)}{hint(value)}"))
                fired = True
                break
            if fired:
                continue
            # N5: modulo literals.
            for mm in _MOD_LITERAL_RE.finditer(code):
                value = _number_value(mm.group(1))
                if value is None or is_structural(value):
                    continue
                report.findings.append(Finding(
                    "numeric_policy", path, i,
                    f"modulo literal {mm.group(1)}{hint(value)}"))
                break


# ---- Gate: no silent switch defaults ---------------------------------------
# A `default:` arm turns a new enumerator into a silently-ignored value
# instead of the compiler's missing-case diagnostic; an unreachable arm keeps
# exhaustiveness live while still compiling. Every default label must lead
# with an abort/unreachable marker.
UNREACHABLE_MARKERS = (
    "__builtin_unreachable",
    "std::abort",
    "SLIME_UNREACHABLE",
    "abort(",
)


def gate_enum_no_silent_default(files: dict[str, list[str]],
                                report: GateReport) -> None:
    for path, lines in files.items():
        if not path.endswith((".cu", ".cuh", ".cpp", ".h")):
            continue
        for i, line in enumerate(lines):
            if not _DEFAULT_LABEL_RE.match(line):
                continue
            tail = " ".join(" ".join(lines[i + 1:i + 4]).split())
            if any(marker in tail for marker in UNREACHABLE_MARKERS):
                continue
            report.findings.append(Finding(
                "enum_no_silent_default", path, i + 1,
                "switch default silently handles a value"))


# ---- Gate: no masked CUDA errors -------------------------------------------
# Discarding a CUDA call's result with `(void)` hides a failure the pipeline
# would otherwise report; every CUDA result is either checked or the call is
# removed. The scan is line-local: a line carrying both a CUDA call and a
# `(void)` discard is an offense.
def gate_no_masked_cuda_errors(files: dict[str, list[str]],
                               report: GateReport) -> None:
    for path, lines in files.items():
        if not path.endswith((".cu", ".cuh")):
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            if "cuda" in line and "(void)" in line:
                report.findings.append(Finding(
                    "no_masked_cuda_errors", path, i,
                    "CUDA result discarded with (void)"))


# ---- Gate: no unchecked error variables ------------------------------------
# A `cudaError_t x = ...` whose name is never compared or passed on stores a
# failure nobody reads. The scan is per file: the name must reappear in a
# comparison or another assignment after its declaration.
_ERROR_VAR_RE = re.compile(r"cudaError_t\s+(\w+)\s*=")


def gate_no_unchecked_error_vars(files: dict[str, list[str]],
                                 report: GateReport) -> None:
    for path, lines in files.items():
        if not path.endswith((".cu", ".cuh")):
            continue
        text = "\n".join(lines)
        for m in _ERROR_VAR_RE.finditer(text):
            name = m.group(1)
            rest = text[m.end():]
            if re.search(rf"\b{re.escape(name)}\b\s*(?:!=|==|\))", rest) \
                    or re.search(rf"if\s*\(\s*{re.escape(name)}\b", rest) \
                    or re.search(rf"\b{re.escape(name)}\b\s*=\s*", rest):
                continue
            line = text[:m.start()].count("\n") + 1
            report.findings.append(Finding(
                "no_unchecked_error_vars", path, line,
                f"cudaError_t {name} is never checked"))


# ---- Gate: no value-to-literal ternary collapse ----------------------------
# `x ? x : "literal"` substitutes a fabricated string when x is absent; the
# honest form branches on the absence itself. (Boolean-to-label mappings like
# `flag ? "yes" : "no"` are presentation and stay allowed.)
_VALUE_COLLAPSE_RE = re.compile(r"\b(\w+)\s*\?\s*\1\s*:\s*\"")


def gate_no_value_ternary_string_default(files: dict[str, list[str]],
                                         report: GateReport) -> None:
    for path, lines in files.items():
        if not path.endswith((".cu", ".cuh")):
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            if _VALUE_COLLAPSE_RE.search(line):
                report.findings.append(Finding(
                    "no_value_ternary_string_default", path, i,
                    "ternary substitutes a string literal for an absent value"))


# ---- Gate: identity fields use strong id types -----------------------------
# Fields carrying the [identity:organism] annotation are the registry hook for
# organism state; declaring one as a raw uint32_t reintroduces exactly the
# interchangeable-id class the strong types removed.
def gate_strong_ids_identity_fields(files: dict[str, list[str]],
                                    report: GateReport) -> None:
    for path, lines in files.items():
        if not path.endswith((".cu", ".cuh")):
            continue
        for i, line in enumerate(lines, 1):
            if "[identity:organism]" not in line:
                continue
            if re.search(r"\buint32_t\b", line):
                report.findings.append(Finding(
                    "strong_ids_identity_fields", path, i,
                    "identity field declared uint32_t; use a strong id type"))


ALL_GATES = [
    gate_no_ambient_rng,
    gate_no_managed_memory,
    gate_checked_cuda_calls,
    gate_named_tunables,
    gate_rd_adjoint_present,
    gate_no_bridge_code,
    gate_host_authority,
    gate_operator_polling,
    gate_replay_before_spawn,
    gate_surprise_before_spawn,
    gate_schedule_host_only,
    gate_numeric_policy,
    gate_enum_no_silent_default,
    gate_no_masked_cuda_errors,
    gate_no_unchecked_error_vars,
    gate_no_value_ternary_string_default,
    gate_strong_ids_identity_fields,
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

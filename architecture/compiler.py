"""Slime architecture compiler: referential integrity over the claim registry.

Commands:
  check     parse claims, resolve mechanisms/witnesses (both directions),
            validate the dependency DAG, document ownership, capability-drift
            regexes, and the executable phase model. Exit 1 on any error.
  status    regenerate docs/IMPLEMENTATION_STATUS.md from claims + evidence
            manifests (golden-file source; CI runs `check --golden`).
  report    print the architecture report (claims, mechanism integrity,
            witnesses, dependency blockers, provisional age, source gates).

The compiler asserts referential integrity only: "spawn_wave still exists and
is attached to A401.spawn-wave-unique". Whether the code is CORRECT is the
witness's job, not the compiler's.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ARCH = Path(__file__).parent

SPEC_DOCS = [
    "docs/blueprint.md",
    "docs/cuda_engineering.md",
    "docs/construction_plan.md",
]

STATUS_DOC = "docs/IMPLEMENTATION_STATUS.md"
STATUS_START = "<!-- architecture-status:start -->"
STATUS_END = "<!-- architecture-status:end -->"

sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from architecture.claims import Claim, WitnessRef, load_registry  # noqa: E402
from architecture import evidence as evidence_mod  # noqa: E402
from architecture import source_gates as gates  # noqa: E402

CODE_SUFFIXES = {".cu", ".cuh", ".cpp", ".h", ".hpp"}
CODE_TOKEN_RE_CACHE: dict[str, re.Pattern] = {}


def code_token_re(symbol: str) -> re.Pattern:
    pat = CODE_TOKEN_RE_CACHE.get(symbol)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(symbol) + r"\b")
        CODE_TOKEN_RE_CACHE[symbol] = pat
    return pat


def resolve_mechanism(anchor: str, root: Path, transactions: dict) -> tuple[bool, str]:
    if "::" not in anchor:
        return False, "anchor missing ::symbol"
    path, symbol = anchor.split("::", 1)
    p = root / path
    if not p.exists():
        return False, f"file missing: {path}"
    if path == "architecture/transactions.yaml":
        flat = flatten_keys(transactions)
        leaves = {k.rsplit(".", 1)[-1] for k in flat}
        tops = set(transactions.keys())
        if symbol not in flat and symbol not in leaves and symbol not in tops:
            return False, f"transactions.yaml key missing: {symbol}"
        return True, ""
    if p.suffix in CODE_SUFFIXES:
        text = p.read_text(encoding="utf-8", errors="replace")
        return bool(code_token_re(symbol).search(text)), f"symbol missing: {path}::{symbol}"
    if p.suffix == ".py":
        text = p.read_text(encoding="utf-8", errors="replace")
        return bool(re.search(r"def\s+" + re.escape(symbol) + r"\b", text)), \
            f"python symbol missing: {path}::{symbol}"
    return False, f"unsupported mechanism file type: {path}"


def flatten_keys(d: dict, prefix: str = "") -> set[str]:
    keys = set()
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        keys.add(full)
        if isinstance(v, dict):
            keys |= flatten_keys(v, full)
    return keys


def resolve_witness(w: WitnessRef, root: Path, claim_id: str) -> tuple[bool, str]:
    p = root / w.path
    if not p.exists():
        return False, f"witness file missing: {w.path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if w.symbol:
        if p.suffix == ".py":
            if not re.search(r"def\s+" + re.escape(w.symbol) + r"\b", text):
                return False, f"witness symbol missing: {w.path}::{w.symbol}"
        elif p.suffix in CODE_SUFFIXES:
            if not code_token_re(w.symbol).search(text):
                return False, f"witness symbol missing: {w.path}::{w.symbol}"
        else:
            return False, f"unsupported witness file type: {w.path}"
    # Bidirectional linkage: the witness must declare the claim it witnesses.
    if f"[claim:{claim_id}]" not in text:
        return False, f"witness does not declare [claim:{claim_id}]: {w.path}"
    return True, ""


def reverse_witness_index(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """file -> claim ids declared via [claim:...] markers; plus orphan markers."""
    index: dict[str, list[str]] = {}
    orphans: list[str] = []
    marker_re = re.compile(r"\[claim:([A-Za-z0-9_.\-]+)\]")
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.parts[0] in gates.EXCLUDED_DIRS:
            continue
        if p.suffix not in CODE_SUFFIXES and p.suffix != ".py":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in marker_re.finditer(text):
            key = str(rel).replace("\\", "/")
            index.setdefault(key, []).append(m.group(1))
            orphans.append(f"{key}:{m.group(1)}")
    return index, orphans


def validate_phase_model(order: list[str], invariants: list[dict]) -> list[str]:
    """Validate ordering invariants against the phase list. Returns errors.

    Exported so tests/architecture can prove the model rejects illegal orders.
    """
    errors: list[str] = []
    pos = {name: i for i, name in enumerate(order)}
    for name in order:
        if not isinstance(name, str) or not name:
            errors.append(f"phase order entry invalid: {name!r}")
    for inv in invariants:
        for key in ("after", "before"):
            if key not in inv:
                continue
            pairs = inv[key]
            if not isinstance(pairs, list) or len(pairs) != 2:
                errors.append(f"phase invariant {key} must be [a, b]: {pairs!r}")
                continue
            a, b = pairs
            if a not in pos:
                errors.append(f"phase invariant references unknown phase: {a}")
            if b not in pos:
                errors.append(f"phase invariant references unknown phase: {b}")
            if a in pos and b in pos:
                if key == "after" and not (pos[a] < pos[b]):
                    errors.append(f"phase invariant violation: {a} must be after {b}")
                if key == "before" and not (pos[a] < pos[b]):
                    errors.append(f"phase invariant violation: {a} must be before {b}")
    return errors


def load_configs(root: Path) -> tuple[dict, dict, dict]:
    documents = yaml.safe_load((ARCH / "documents.yaml").read_text(encoding="utf-8"))
    transactions = yaml.safe_load((ARCH / "transactions.yaml").read_text(encoding="utf-8"))
    machine = json.loads((ARCH / "machine.json").read_text(encoding="utf-8"))
    return documents, transactions, machine


def check_claims(claims: list[Claim], root: Path, transactions: dict,
                 errors: list[str], warnings: list[str]) -> None:
    by_id = {c.id: c for c in claims}

    # Mechanisms resolve.
    for c in claims:
        for m in c.mechanisms:
            ok, msg = resolve_mechanism(m, root, transactions)
            if not ok:
                errors.append(f"{c.id}: mechanism {m}: {msg}")

    # Witnesses resolve + bidirectional markers.
    rev_index, orphans = reverse_witness_index(root)
    witnessed_files: dict[str, list[Claim]] = {}
    for c in claims:
        for w in c.witnesses:
            ok, msg = resolve_witness(w, root, c.id)
            if not ok:
                errors.append(f"{c.id}: witness {w.anchor}: {msg}")
            witnessed_files.setdefault(w.path, []).append(c)
    declared = {c.id for c in claims}
    for orphan in orphans:
        path, cid = orphan.rsplit(":", 1)
        if cid not in declared:
            errors.append(f"{path}: [claim:{cid}] marker references unknown claim")
        else:
            claim = by_id[cid]
            files = {w.path for w in claim.witnesses}
            if path not in files:
                errors.append(
                    f"{path}: declares [claim:{cid}] but that claim does not "
                    f"list this file as a witness")

    # Dependency DAG: exists, acyclic, established deps established.
    for c in claims:
        for d in c.deps:
            if d not in by_id:
                errors.append(f"{c.id}: dependency {d} does not exist")
                continue
            if c.lifecycle == "established" and by_id[d].lifecycle != "established":
                errors.append(
                    f"{c.id}: established claim depends on non-established "
                    f"{d} ({by_id[d].lifecycle})")
    state = {}
    visiting = set()

    def visit(cid: str, path: list[str]) -> None:
        if cid in state:
            return
        if cid in visiting:
            errors.append("dependency cycle: " + " -> ".join(path + [cid]))
            return
        visiting.add(cid)
        for d in by_id[cid].deps:
            if d in by_id:
                visit(d, path + [cid])
        visiting.discard(cid)
        state[cid] = True

    for cid in by_id:
        visit(cid, [])

    # Lifecycle sanity.
    for c in claims:
        if c.lifecycle == "established" and not c.witnesses:
            warnings.append(f"{c.id}: established with no witnesses")


def check_documents(root: Path, documents: dict, errors: list[str]) -> None:
    declared_files: dict[str, list[str]] = {}
    generated: list[str] = []
    kinds: dict[str, str] = {}
    for concern, spec in documents.get("documents", {}).items():
        for f in spec.get("files", []):
            declared_files.setdefault(f, []).append(concern)
            kinds[f] = spec.get("kind", "operational")
            if spec.get("kind") == "generated":
                generated.append(f)
    for p in sorted((root / "docs").glob("*.md")):
        rel = f"docs/{p.name}"
        if rel not in declared_files:
            errors.append(f"undeclared document: {rel} (add it to architecture/documents.yaml)")
    for f in declared_files:
        if not (root / f).exists():
            errors.append(f"declared document missing: {f}")
        if len(declared_files[f]) > 1:
            errors.append(f"document {f} has multiple owners: {declared_files[f]}")
        if kinds.get(f) == "generated":
            text = (root / f).read_text(encoding="utf-8", errors="replace")
            if STATUS_START not in text or STATUS_END not in text:
                errors.append(
                    f"generated document {f} lacks the "
                    f"{STATUS_START} / {STATUS_END} block")


def check_capability_drift(root: Path, documents: dict, errors: list[str]) -> None:
    for cap, spec in documents.get("capability-drift", {}).items():
        if spec.get("status") != "planned":
            continue
        for doc in spec.get("surface-docs", []):
            p = root / doc
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                low = line.lower()
                for banned in spec.get("banned-present-tense", []):
                    if banned in low:
                        errors.append(
                            f"capability drift ({cap}): {doc}:{i}: "
                            f"'{line.strip()}' asserts planned behavior in "
                            f"present tense (banned: '{banned}')")


def check_phase_and_transactions(root: Path, transactions: dict,
                                 errors: list[str]) -> None:
    gen = transactions.get("generation_phases", {})
    errors.extend(validate_phase_model(gen.get("order", []), gen.get("invariants", [])))

    # Transaction completeness: every declared organism-identity buffer must be
    # moved by the PT transaction.
    buffers = transactions.get("organism_buffers", [])
    pt_swap = transactions.get("transactions", {}).get("pt_swap", {})
    identity = pt_swap.get("organism_identity", [])
    for b in buffers:
        if b not in identity:
            errors.append(
                f"transactions.yaml: organism buffer '{b}' is not in "
                f"pt_swap.organism_identity")


def render_status(claims: list[Claim], root: Path, manifests: list[dict],
                  transactions: dict) -> str:
    rows = []
    for c in sorted(claims, key=lambda x: x.id):
        v = evidence_mod.verdict(c, manifests, root)
        mech_ok = all(resolve_mechanism(m, root, transactions)[0]
                      for m in c.mechanisms)
        strong = "✅" if c.has_strong_witness else "❌"
        rows.append(
            f"| {c.id} | {c.kind} | {c.lifecycle} | "
            f"{'✅' if mech_ok else '❌'} | {strong} | {v.state} |")
    lines = [
        "| Claim | Kind | Lifecycle | Mechanisms | Strong witness | Latest evidence |",
        "| :---- | :--- | :-------- | :--------- | :------------- | :-------------- |",
    ] + rows
    return "\n".join(lines)


def build_status_file(claims: list[Claim], root: Path, manifests: list[dict],
                      transactions: dict, gates_report: gates.GateReport) -> str:
    table = render_status(claims, root, manifests, transactions)
    gate_summary = ", ".join(
        f"{g}: {'PASS' if not any(f.gate == g for f in gates_report.errors) else 'FAIL'}"
        for g in gates.GATE_NAMES)
    return (
        "# Implementation Status\n"
        "\n"
        "GENERATED FILE. Do not edit by hand; run "
        "`python architecture/compiler.py status`. Evidence tracker only — "
        "the binding behavior, engineering constraints, and delivery "
        "requirements are respectively in [blueprint.md](blueprint.md), "
        "[cuda_engineering.md](cuda_engineering.md), and "
        "[construction_plan.md](construction_plan.md).\n"
        "\n"
        f"{STATUS_START}\n"
        f"{table}\n"
        "\n"
        f"Source gates: {gate_summary}\n"
        "\n"
        f"Claims: {len(claims)} total, "
        f"{sum(1 for c in claims if c.lifecycle == 'established')} established, "
        f"{sum(1 for c in claims if c.lifecycle == 'provisional')} provisional, "
        f"{sum(1 for c in claims if c.lifecycle == 'planned')} planned.\n"
        f"{STATUS_END}\n"
    )


def check(args) -> int:
    root = ROOT
    errors: list[str] = []
    warnings: list[str] = []

    documents, transactions, _machine = load_configs(root)
    claims, parse_errors = load_registry(SPEC_DOCS)
    errors.extend(parse_errors)
    check_claims(claims, root, transactions, errors, warnings)
    check_documents(root, documents, errors)
    check_capability_drift(root, documents, errors)
    check_phase_and_transactions(root, transactions, errors)

    gates_report = gates.run_gates(root, strict=getattr(args, "strict", False))
    errors.extend(str(f) for f in gates_report.errors)
    warnings.extend(str(f) for f in gates_report.warnings)

    if args.golden:
        manifests = evidence_mod.load_manifests(root / "evidence")
        expected = build_status_file(claims, root, manifests, transactions, gates_report)
        status_path = root / STATUS_DOC
        if not status_path.exists():
            errors.append(f"{STATUS_DOC} missing; run `python architecture/compiler.py status`")
        else:
            actual = status_path.read_text(encoding="utf-8")
            if actual != expected:
                errors.append(
                    f"{STATUS_DOC} is stale; regenerate with "
                    f"`python architecture/compiler.py status`")

    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    if errors:
        print(f"\narchitecture check: {len(errors)} errors, {len(warnings)} warnings")
        return 1
    print(f"architecture check: OK ({len(claims)} claims, {len(warnings)} warnings)")
    return 0


def status(args) -> int:
    root = ROOT
    _documents, transactions, _machine = load_configs(root)
    claims, errors = load_registry(SPEC_DOCS)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    manifests = evidence_mod.load_manifests(root / "evidence")
    gates_report = gates.run_gates(root)
    content = build_status_file(claims, root, manifests, transactions, gates_report)
    (root / STATUS_DOC).write_text(content, encoding="utf-8")
    print(f"wrote {STATUS_DOC} ({len(claims)} claims, {len(manifests)} evidence manifests)")
    return 0


def report(args) -> int:
    root = ROOT
    documents, transactions, _machine = load_configs(root)
    claims, errors = load_registry(SPEC_DOCS)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    manifests = evidence_mod.load_manifests(root / "evidence")
    gates_report = gates.run_gates(root, strict=getattr(args, "strict", False))

    established = [c for c in claims if c.lifecycle == "established"]
    provisional = [c for c in claims if c.lifecycle == "provisional"]
    planned = [c for c in claims if c.lifecycle == "planned"]

    print("SLIME ARCHITECTURE REPORT")
    print("=========================")
    print(f"\nClaims: {len(claims)}")
    print(f"  established:  {len(established)}")
    print(f"  provisional:  {len(provisional)}")
    print(f"  planned:      {len(planned)}")

    broken_mech = []
    witness_state: dict[str, int] = {}
    for c in claims:
        if not all(resolve_mechanism(m, root, transactions)[0] for m in c.mechanisms):
            broken_mech.append(c.id)
        for w in c.witnesses:
            ok, _ = resolve_witness(w, root, c.id)
            key = "passing" if ok else "broken"
            witness_state[key] = witness_state.get(key, 0) + 1
    print(f"\nMechanism integrity:")
    print(f"  {len(claims) - len(broken_mech)} complete")
    if broken_mech:
        print(f"  {len(broken_mech)} broken: {', '.join(broken_mech)}")

    print(f"\nStrong witnesses:")
    for state, n in sorted(witness_state.items()):
        print(f"  {state}: {n}")
    verdicts = {c.id: evidence_mod.verdict(c, manifests, root) for c in claims}
    for state in ("pass/current", "pass/stale", "fail", "never"):
        ids = sorted(cid for cid, v in verdicts.items() if v.state == state)
        if ids:
            print(f"  evidence {state}: {', '.join(ids)}")

    print(f"\nDependency blockers:")
    blocked = [c for c in claims
               if any(d in {x.id for x in claims} and
                      next(x for x in claims if x.id == d).lifecycle != "established"
                      for d in c.deps)]
    for c in blocked:
        print(f"  {c.id}")
        for d in c.deps:
            dep = next((x for x in claims if x.id == d), None)
            if dep and dep.lifecycle != "established":
                print(f"    blocked by {d} ({dep.lifecycle})")
    if not blocked:
        print("  none")

    print(f"\nSource gates:")
    for g in gates.GATE_NAMES:
        errs = [f for f in gates_report.errors if f.gate == g]
        warns = [f for f in gates_report.warnings if f.gate == g]
        state = "FAIL" if errs else ("WARN" if warns else "PASS")
        print(f"  {g}: {state} ({len(errs)} errors, {len(warns)} allowlisted)")

    print(f"\nProvisional with no strong witness:")
    for c in provisional:
        if not c.has_strong_witness:
            print(f"  {c.id} ({c.confidence})")
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Slime architecture compiler")
    sub = parser.add_subparsers(dest="cmd")
    c = sub.add_parser("check")
    c.add_argument("--golden", action="store_true",
                   help="also verify docs/IMPLEMENTATION_STATUS.md is up to date")
    c.add_argument("--strict", action="store_true",
                   help="promote allowlisted source-gate warnings to errors")
    sub.add_parser("status")
    r = sub.add_parser("report")
    r.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.cmd == "check":
        return check(args)
    if args.cmd == "status":
        return status(args)
    if args.cmd == "report":
        return report(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

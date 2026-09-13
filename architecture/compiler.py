"""Slime architecture compiler: referential integrity over the claim registry.

Commands:
  check     parse claims, resolve mechanisms/witnesses (both directions),
            validate the dependency DAG, document ownership, capability-drift
            regexes, the executable phase model, the build inventory, and
            prose citations in the operational documents. Exit 1 on any error.
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

sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from architecture.claims import (  # noqa: E402
    Claim, WitnessRef, SPEC_DOCS, CODE_SUFFIXES, code_token_re, flatten_keys,
    load_registry, resolve_mechanism, resolve_witness, claim_hash,
    VALID_KINDS, VALID_LIFECYCLE, VALID_CONFIDENCE,
)
from architecture import evidence as evidence_mod  # noqa: E402
from architecture import source_gates as gates  # noqa: E402

STATUS_DOC = "docs/IMPLEMENTATION_STATUS.md"
STATUS_START = "<!-- architecture-status:start -->"
STATUS_END = "<!-- architecture-status:end -->"
BUILD_STATUS_FILE = "architecture/build_status.yaml"
BRIDGE_FILE = "architecture/bridge.yaml"
BRIDGE_STATES = ("OPEN", "CLOSED")
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
PLAN_DOC = "docs/construction_plan.md"
INVENTORY_RE = re.compile(r"^### (I\d+)\b", re.MULTILINE)
PLAN_HEADING_RE = re.compile(r"^### (I\d+) \u2014 (.+?)\s*$", re.MULTILINE)
BUILD_STATES = ("missing", "partial", "implemented")
EXPERIMENTS_FILE = "architecture/experiments.yaml"
EXPERIMENT_HEADING_RE = re.compile(r"^### (E\d+) \u2014 (.+?)\s*$", re.MULTILINE)
EXPERIMENT_STATES = ("planned", "running", "concluded")
EXPERIMENT_REQUIRED_FIELDS = ("hypothesis", "intervention", "controls",
                              "metrics", "seeds", "decision", "status",
                              "claims")

# ---- Prose citation guards ------------------------------------------------
# Operational docs may not restate derivable facts: they cite paths, make
# targets, scripts, claim ids, and the build inventory, and the compiler
# verifies every citation resolves. The friction log (docs/arch-signals.md)
# is historical and exempt.
OPERATIONAL_DOCS = ("README.md", "AGENTS.md", "TODO.md")
CITATION_DOCS = OPERATIONAL_DOCS + (PLAN_DOC, "docs/blueprint.md",
                                    "docs/cuda_engineering.md")
CITED_PATH_RE = re.compile(
    r"\b((?:architecture|tests|integration|optimizer|autodiff|nca|genome|"
    r"archive|safety|curriculum|predictor|contracts|config|docs)/"
    r"[A-Za-z0-9_./-]+\.(?:cu|cuh|cpp|h|py|md|json|yaml))\b")
MAKE_TARGET_RE = re.compile(r"\bmake ([a-z][a-z0-9-]*)\b")
PYTHON_CITE_RE = re.compile(
    r"python (architecture/[a-z_]+\.py)(?:\s+([a-z][a-z-]*))?")
CLAIM_MENTION_RE = re.compile(r"\b([A-Z]\d{3}\.[a-z][a-z0-9-]+)\b")
TODO_ITEM_RE = re.compile(r"\b(I[1-9])\b")

# Source annotations in integration/main_loop.cu (see its OrganismTable/World
# comments). The compiler enforces registry completeness in BOTH directions:
#   code [crosses:pt=X]  ->  X must be in pt_swap.organism_identity
#   pt_swap.organism_identity entry  ->  must have a [crosses:pt=...]
#                                       annotation in code
ANNOTATION_CROSSES_RE = re.compile(r"\[crosses:pt=([A-Za-z_][\w]*)\]")
ANNOTATION_IDENTITY_RE = re.compile(r"\[identity:organism\]")
MAIN_LOOP_SRC = "integration/main_loop.cu"


def reverse_witness_index(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """file -> claim ids declared via [claim:...] markers; plus orphan markers."""
    index: dict[str, list[str]] = {}
    orphans: list[str] = []
    marker_re = re.compile(r"\[claim:([A-Za-z0-9_.\-]+)\]")
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.parts[0] in gates.EXCLUDED_DIRS or rel.parts[0].startswith("."):
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


def load_build_status() -> dict:
    return yaml.safe_load((ARCH / "build_status.yaml").read_text(encoding="utf-8"))


def load_bridge() -> dict:
    return yaml.safe_load((ARCH / "bridge.yaml").read_text(encoding="utf-8"))


def check_bridge(root: Path, build: dict, errors: list[str]) -> None:
    """Validate the cross-repo bridge admission record.

    The record is non-normative, but its gate state is a claim about the
    build: it may not say OPEN while an inventory item is unimplemented, and
    a CLOSED gate must carry the reasons and the open coherence questions.
    """
    bridge = load_bridge()
    fp = str(bridge.get("declared_fingerprint", ""))
    if not FINGERPRINT_RE.match(fp):
        errors.append(f"{BRIDGE_FILE}: declared_fingerprint is not 64 hex chars")
    gate = bridge.get("gate")
    if gate not in BRIDGE_STATES:
        errors.append(f"{BRIDGE_FILE}: gate {gate!r} is not one of {BRIDGE_STATES}")
    incomplete = [iid for iid, item in sorted(build.get("items", {}).items())
                  if item.get("status") != "implemented"]
    if gate == "OPEN" and incomplete:
        errors.append(f"{BRIDGE_FILE}: gate is OPEN while build items "
                      f"{', '.join(incomplete)} are not implemented")
    if gate == "CLOSED":
        if not bridge.get("gate_reasons"):
            errors.append(f"{BRIDGE_FILE}: CLOSED gate must list reasons")
        if not bridge.get("coherence_questions"):
            errors.append(f"{BRIDGE_FILE}: CLOSED gate must list the open "
                          f"coherence questions")
    record = bridge.get("handoff_record")
    if not record or not (root / record).exists():
        errors.append(f"{BRIDGE_FILE}: handoff_record {record!r} missing")


def render_bridge() -> str:
    bridge = load_bridge()
    lines = [
        f"- External contract: `{bridge.get('external_contract', '')}` "
        f"(owner: {bridge.get('contract_owner', '')})",
        f"- Declared fingerprint: `{bridge.get('declared_fingerprint', '')}` "
        f"(recomputed by Slime: "
        f"{'yes' if bridge.get('fingerprint_recomputed') else 'no'})",
        f"- Admission gate: **{bridge.get('gate', '')}**",
    ]
    for reason in bridge.get("gate_reasons", []):
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def build_incomplete(build: dict) -> list[str]:
    return [iid for iid, item in sorted(build.get("items", {}).items())
            if item.get("status") != "implemented"]


def check_build_status(root: Path, build: dict, transactions: dict,
                       errors: list[str]) -> None:
    """Validate the build inventory against the plan and the code.

    The plan's order of operations is binding: the inventory here is the
    machine-readable truth the GPU acceptance gate reads, so it may not
    reference mechanisms that do not exist and may not silently omit an
    inventory item.
    """
    plan = (root / PLAN_DOC).read_text(encoding="utf-8")
    planned = set(INVENTORY_RE.findall(plan))
    items = build.get("items", {})
    for missing in sorted(planned - set(items)):
        errors.append(f"{PLAN_DOC}: {missing} has no {BUILD_STATUS_FILE} entry")
    for extra in sorted(set(items) - planned):
        errors.append(f"{BUILD_STATUS_FILE}: {extra} is not an inventory "
                      f"item in {PLAN_DOC}")
    for iid, item in sorted(items.items()):
        status = item.get("status")
        if status not in BUILD_STATES:
            errors.append(f"{BUILD_STATUS_FILE}: {iid} status {status!r} "
                          f"is not one of {BUILD_STATES}")
            continue
        mechanisms = item.get("mechanisms", [])
        if status != "missing" and not mechanisms:
            errors.append(f"{BUILD_STATUS_FILE}: {iid} is {status} but names "
                          f"no mechanisms")
        if status == "missing" and mechanisms:
            errors.append(f"{BUILD_STATUS_FILE}: {iid} is missing but names "
                          f"mechanisms")
        for mechanism in mechanisms:
            ok, why = resolve_mechanism(mechanism, root, transactions)
            if not ok:
                errors.append(f"{BUILD_STATUS_FILE}: {iid} mechanism "
                              f"{mechanism}: {why}")


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
    for p in sorted(root.rglob("*")):
        if p.suffix not in (".md", ".mdc"):
            continue
        rel = p.relative_to(root)
        if rel.parts[0] in ("build", ".git"):
            continue
        rel_str = str(rel).replace("\\", "/")
        if rel_str not in declared_files:
            errors.append(
                f"undeclared document: {rel_str} (add it to "
                f"architecture/documents.yaml)")
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

    pt_swap = transactions.get("transactions", {}).get("pt_swap", {})
    identity = pt_swap.get("organism_identity", [])
    buffers = transactions.get("organism_buffers", [])

    # Registry-internal completeness.
    for b in buffers:
        if b not in identity:
            errors.append(
                f"transactions.yaml: organism buffer '{b}' is not in "
                f"pt_swap.organism_identity")
    for b in identity:
        if b not in buffers:
            errors.append(
                f"transactions.yaml: pt_swap.organism_identity entry '{b}' "
                f"is not in organism_buffers")

    # Code-internal completeness: every [crosses:pt=<key>] annotation in
    # integration/main_loop.cu must name an identity entry, and every identity
    # entry must be annotated in code. This catches buffers like d_eff_weights
    # that exist in the World struct but were never told to either list.
    src_path = root / MAIN_LOOP_SRC
    if src_path.exists():
        text = src_path.read_text(encoding="utf-8", errors="replace")
        annotated = set(ANNOTATION_CROSSES_RE.findall(text))
        n_identity = len(ANNOTATION_IDENTITY_RE.findall(text))
        for key in sorted(annotated):
            if key not in identity:
                errors.append(
                    f"{MAIN_LOOP_SRC}: field annotated [crosses:pt={key}] is "
                    f"not in pt_swap.organism_identity")
        for key in sorted(identity):
            if key not in annotated:
                errors.append(
                    f"transactions.yaml: pt_swap.organism_identity entry "
                    f"'{key}' has no [crosses:pt={key}] annotation in "
                    f"{MAIN_LOOP_SRC}")
        if n_identity < len(identity):
            errors.append(
                f"{MAIN_LOOP_SRC}: {n_identity} [identity:organism] "
                f"annotations for {len(identity)} registry buffers")


def check_prose_citations(root: Path, claims: list[Claim], build: dict,
                          errors: list[str]) -> None:
    """Every citation in the operational docs must resolve.

    The compiler enforces referential integrity inside the claim registry;
    this extends it to prose. A path, make target, script, subcommand, claim
    id, or build item mentioned in README/AGENTS/TODO that does not exist is
    a silently drifted restatement, so it fails the check.
    """
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    make_targets = set(re.findall(r"^([a-zA-Z0-9_-]+):", makefile,
                                  re.MULTILINE))
    claim_ids = {c.id for c in claims}
    for doc in OPERATIONAL_DOCS:
        text = (root / doc).read_text(encoding="utf-8")
        for path in sorted(set(CITED_PATH_RE.findall(text))):
            if not (root / path).exists():
                errors.append(f"{doc}: cites missing path {path}")
        for target in sorted(set(MAKE_TARGET_RE.findall(text))):
            if target not in make_targets:
                errors.append(f"{doc}: cites undefined make target "
                              f"`make {target}`")
        for script, sub in sorted(set(PYTHON_CITE_RE.findall(text))):
            script_path = root / script
            if not script_path.exists():
                errors.append(f"{doc}: cites missing script {script}")
            elif sub:
                body = script_path.read_text(encoding="utf-8")
                if f'"{sub}"' not in body and f"'{sub}'" not in body:
                    errors.append(f"{doc}: cites unknown subcommand "
                                  f"{script} {sub}")
    for doc in CITATION_DOCS:
        text = (root / doc).read_text(encoding="utf-8")
        for cid in sorted(set(CLAIM_MENTION_RE.findall(text))):
            if cid not in claim_ids:
                errors.append(f"{doc}: mentions unknown claim {cid}")

    # The work queue must cover the incomplete build inventory.
    todo = (root / "TODO.md").read_text(encoding="utf-8")
    mentioned = set(TODO_ITEM_RE.findall(todo))
    items = set(build.get("items", {}))
    incomplete = {iid for iid, item in build.get("items", {}).items()
                  if item.get("status") != "implemented"}
    for iid in sorted(incomplete - mentioned):
        errors.append(f"TODO.md: incomplete build item {iid} is not listed")
    for iid in sorted(mentioned - items):
        errors.append(f"TODO.md: unknown build item {iid}")


def check_claim_grammar_doc(root: Path, errors: list[str]) -> None:
    """The claim grammar restated in AGENTS.md must match claims.py."""
    text = (root / "AGENTS.md").read_text(encoding="utf-8")
    patterns = (
        ("kind", r"@claim <id> <kind>\s+(.+)", VALID_KINDS),
        ("lifecycle", r"T <lifecycle>\s+(.+)", VALID_LIFECYCLE),
        ("confidence", r"C <confidence>\s+(.+)", VALID_CONFIDENCE),
    )
    for label, pattern, valid in patterns:
        m = re.search(pattern, text)
        if not m:
            errors.append(f"AGENTS.md: claim grammar line for {label} missing")
            continue
        listed = {t.strip() for t in m.group(1).split("|")}
        if listed != valid:
            errors.append(f"AGENTS.md: {label} tokens {sorted(listed)} do not "
                          f"match claims.py {sorted(valid)}")


def check_canonical_doc_list(root: Path, documents: dict,
                             errors: list[str]) -> None:
    """AGENTS.md must list every canonical document and invent none."""
    declared: set[str] = set()
    canonical: set[str] = set()
    for concern in documents.get("documents", {}).values():
        declared.update(concern.get("files", []))
        if concern.get("kind") == "canonical":
            canonical.update(concern.get("files", []))
    text = (root / "AGENTS.md").read_text(encoding="utf-8")
    mentioned = set(re.findall(r"docs/[a-z_]+\.md", text))
    for path in sorted(canonical - mentioned):
        errors.append(f"AGENTS.md: canonical document {path} is not listed")
    for path in sorted(mentioned - declared):
        errors.append(f"AGENTS.md: mentions undeclared document {path}")


def load_experiments() -> dict:
    return yaml.safe_load((ARCH / "experiments.yaml").read_text(encoding="utf-8"))


def check_experiments(root: Path, claims: list[Claim],
                      errors: list[str]) -> None:
    """Validate the experiment registry against the plan and the claims.

    Every `### E<n>` heading in the construction plan needs exactly one
    protocol, every protocol needs the full preregistration fields, and every
    referenced claim must exist.
    """
    plan = (root / PLAN_DOC).read_text(encoding="utf-8")
    planned = {m.group(1) for m in EXPERIMENT_HEADING_RE.finditer(plan)}
    experiments = load_experiments().get("experiments", {})
    for missing in sorted(planned - set(experiments)):
        errors.append(f"{PLAN_DOC}: {missing} has no {EXPERIMENTS_FILE} entry")
    for extra in sorted(set(experiments) - planned):
        errors.append(f"{EXPERIMENTS_FILE}: {extra} is not an experiment in "
                      f"{PLAN_DOC}")
    claim_ids = {c.id for c in claims}
    for eid, e in sorted(experiments.items()):
        for field in EXPERIMENT_REQUIRED_FIELDS:
            if field not in e or not e[field]:
                errors.append(f"{EXPERIMENTS_FILE}: {eid} missing field "
                              f"{field}")
        if e.get("status") not in EXPERIMENT_STATES:
            errors.append(f"{EXPERIMENTS_FILE}: {eid} status "
                          f"{e.get('status')!r} is not one of "
                          f"{EXPERIMENT_STATES}")
        for cid in e.get("claims", []):
            if cid not in claim_ids:
                errors.append(f"{EXPERIMENTS_FILE}: {eid} references unknown "
                              f"claim {cid}")


def render_experiments() -> str:
    experiments = load_experiments().get("experiments", {})
    rows = []
    for eid, e in sorted(experiments.items()):
        rows.append(f"| {eid} | {e.get('title', '')} | "
                    f"{e.get('status', '')} | "
                    f"{', '.join(e.get('claims', []))} |")
    lines = [
        "| Experiment | Title | Status | Claims |",
        "| :--------- | :---- | :----- | :----- |",
    ] + rows
    return "\n".join(lines)


def gate_summary_line(gates_report: gates.GateReport) -> str:
    parts = []
    for g in gates.GATE_NAMES:
        errs = [f for f in gates_report.errors if f.gate == g]
        warns = [f for f in gates_report.warnings if f.gate == g]
        state = "FAIL" if errs else ("WARN" if warns else "PASS")
        parts.append(f"{g}: {state}")
    return ", ".join(parts)


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


def render_build_status(build: dict, titles: dict[str, str]) -> str:
    rows = []
    for iid, item in sorted(build.get("items", {}).items()):
        rows.append(f"| {iid} | {titles.get(iid, '')} | "
                    f"{item.get('status', '')} |")
    lines = [
        "| Item | Feature | Build status |",
        "| :--- | :------ | :----------- |",
    ] + rows
    return "\n".join(lines)


def build_status_file(claims: list[Claim], root: Path, manifests: list[dict],
                      transactions: dict, gates_report: gates.GateReport,
                      build: dict) -> str:
    table = render_status(claims, root, manifests, transactions)
    plan = (root / PLAN_DOC).read_text(encoding="utf-8")
    titles = dict(PLAN_HEADING_RE.findall(plan))
    inventory = render_build_status(build, titles)
    n_items = len(build.get("items", {}))
    incomplete = build_incomplete(build)
    if incomplete:
        build_gate = (
            f"BUILD phase incomplete: {len(incomplete)} of {n_items} inventory "
            f"items are not implemented ({', '.join(incomplete)}). No GPU "
            f"integration, acceptance, or production run is permitted, and "
            f"`evidence.py record` refuses GPU manifests.")
    else:
        build_gate = ("BUILD phase complete: every inventory item is "
                      "implemented; VERIFY and INTEGRATE may proceed.")
    gate_summary = gate_summary_line(gates_report)
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
        "## Build inventory\n"
        "\n"
        f"{inventory}\n"
        "\n"
        f"{build_gate}\n"
        "\n"
        "## Experiment registry\n"
        "\n"
        f"{render_experiments()}\n"
        "\n"
        "## Bridge admission\n"
        "\n"
        f"{render_bridge()}\n"
        "\n"
        "## Claims\n"
        "\n"
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
    build = load_build_status()
    claims, parse_errors = load_registry(SPEC_DOCS)
    errors.extend(parse_errors)
    check_claims(claims, root, transactions, errors, warnings)
    check_documents(root, documents, errors)
    check_capability_drift(root, documents, errors)
    check_phase_and_transactions(root, transactions, errors)
    check_build_status(root, build, transactions, errors)
    check_bridge(root, build, errors)
    check_prose_citations(root, claims, build, errors)
    check_claim_grammar_doc(root, errors)
    check_canonical_doc_list(root, documents, errors)
    check_experiments(root, claims, errors)

    gates_report = gates.run_gates(root, strict=getattr(args, "strict", False))
    errors.extend(str(f) for f in gates_report.errors)
    warnings.extend(str(f) for f in gates_report.warnings)

    if args.golden:
        manifests = evidence_mod.load_manifests(root / "evidence")
        expected = build_status_file(claims, root, manifests, transactions,
                                     gates_report, build)
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
    build = load_build_status()
    claims, errors = load_registry(SPEC_DOCS)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    check_build_status(root, build, transactions, errors)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    manifests = evidence_mod.load_manifests(root / "evidence")
    gates_report = gates.run_gates(root)
    content = build_status_file(claims, root, manifests, transactions,
                                gates_report, build)
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

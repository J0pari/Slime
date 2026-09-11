"""Claim registry model and markdown parser for the Slime architecture layer.

Claim blocks are embedded directly in the binding specification documents
(blueprint.md, cuda_engineering.md, construction_plan.md). Block syntax:

    @claim <id> <kind>
    S <statement text, may wrap onto lines indented two spaces>
    M <path::anchor>            (mechanism, repeatable)
    W <path::anchor>            (exercise witness, repeatable)
    W+ <path::anchor>           (strong witness, repeatable)
    P <tag>                     (free-form tag, e.g. reproducibility)
    T <lifecycle>               planned | provisional | established | deprecated
    C <confidence>              unobserved | inferred | observed | repeatedly_observed
    D <claim-id>                (dependency, repeatable)

The block ends at the first blank line. A field keyword must start a line;
continuation lines are indented at least two spaces.

Kind vocabulary:
  invariant   - must hold in every valid execution
  contract    - boundary/ordering obligation between components
  capability  - named functionality with activation state
  policy      - host-side decision rule (safety/operator policy)
  empirical   - hypothesis about measured behavior (not an invariant)
  acceptance  - gate an experiment must pass to count as evidence

This module is stdlib-only so the compiler runs anywhere python3 is present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

VALID_KINDS = {"invariant", "contract", "capability", "policy", "empirical", "acceptance"}
VALID_LIFECYCLE = {"planned", "provisional", "established", "deprecated"}
VALID_CONFIDENCE = {"unobserved", "inferred", "observed", "repeatedly_observed"}

CLAIM_RE = re.compile(r"^@claim\s+(\S+)\s+(\S+)\s*$")
FIELD_RE = re.compile(r"^(S|M|W\+|W|P|T|C|D|X)\s+(.*)$")
CONTINUATION_RE = re.compile(r"^  (.*)$")


@dataclass
class WitnessRef:
    strength: str          # "W" or "W+"
    anchor: str            # path::symbol
    claim_id: str = ""

    @property
    def path(self) -> str:
        return self.anchor.split("::", 1)[0] if "::" in self.anchor else self.anchor

    @property
    def symbol(self) -> str:
        return self.anchor.split("::", 1)[1] if "::" in self.anchor else ""


@dataclass
class Claim:
    id: str
    kind: str
    doc: str
    line: int
    statement: str = ""
    mechanisms: list[str] = field(default_factory=list)
    witnesses: list[WitnessRef] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    lifecycle: str = "planned"
    confidence: str = "unobserved"
    deps: list[str] = field(default_factory=list)

    @property
    def has_strong_witness(self) -> bool:
        return any(w.strength == "W+" for w in self.witnesses)


def parse_claim_blocks(text: str, doc: str) -> list[Claim]:
    claims: list[Claim] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = CLAIM_RE.match(lines[i])
        if not m:
            i += 1
            continue
        claim_id, kind = m.group(1), m.group(2)
        claim = Claim(id=claim_id, kind=kind, doc=doc, line=i + 1)
        current_field: str | None = None
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() == "":
                break
            fm = FIELD_RE.match(line)
            if fm:
                key, value = fm.group(1), fm.group(2)
                current_field = key
                if key == "S":
                    claim.statement = value
                elif key == "M":
                    claim.mechanisms.append(value)
                elif key in ("W", "W+"):
                    claim.witnesses.append(WitnessRef(strength=key, anchor=value, claim_id=claim_id))
                elif key == "P":
                    claim.tags.append(value)
                elif key == "T":
                    claim.lifecycle = value
                elif key == "C":
                    claim.confidence = value
                elif key == "D":
                    claim.deps.append(value)
                elif key == "X":
                    claim.tags.append("exception:" + value)
                i += 1
                continue
            cm = CONTINUATION_RE.match(line)
            if cm and current_field == "S":
                claim.statement = (claim.statement + " " + cm.group(1)).strip()
                i += 1
                continue
            # Unexpected content: end the block defensively.
            break
        claims.append(claim)
    return claims


def load_registry(spec_docs: list[str]) -> tuple[list[Claim], list[str]]:
    """Parse all claim blocks from the specification documents."""
    claims: list[Claim] = []
    errors: list[str] = []
    seen: dict[str, Claim] = {}
    for doc in spec_docs:
        p = Path(doc)
        if not p.exists():
            errors.append(f"spec document missing: {doc}")
            continue
        text = p.read_text(encoding="utf-8")
        for c in parse_claim_blocks(text, doc):
            if c.id in seen:
                errors.append(
                    f"{doc}:{c.line}: duplicate claim id {c.id} "
                    f"(already defined in {seen[c.id].doc}:{seen[c.id].line})"
                )
                continue
            seen[c.id] = c
            claims.append(c)
    for c in claims:
        if c.kind not in VALID_KINDS:
            errors.append(f"{c.id}: invalid kind '{c.kind}'")
        if c.lifecycle not in VALID_LIFECYCLE:
            errors.append(f"{c.id}: invalid lifecycle '{c.lifecycle}'")
        if c.confidence not in VALID_CONFIDENCE:
            errors.append(f"{c.id}: invalid confidence '{c.confidence}'")
        if not c.statement:
            errors.append(f"{c.id}: missing S statement")
        if not c.mechanisms:
            errors.append(f"{c.id}: no mechanism anchors (M lines)")
    return claims, errors

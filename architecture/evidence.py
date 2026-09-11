"""Evidence manifests and staleness for the Slime architecture layer.

Every accepted test/run records a manifest (schema slime-evidence/v1) with
hashes of the claims, mechanisms, witnesses, and binary it exercised. The
status renderer derives PASS/CURRENT, PASS/STALE, FAIL, and NEVER RUN from
manifests alone — never from prose. When a mechanism file changes after the
latest accepted evidence, the claim becomes STALE until its witness reruns.

A manifest can be created from a real run with:

    python architecture/evidence.py record --name run5 --binary build/coevo.exe \
        --cuda 13.0.48 --gpu "RTX 3060 Laptop" --seed default \
        --result A103.checkpoint-replay-equivalence:pass ...

stdlib-only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "slime-evidence/v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spec_hash(claims) -> str:
    """Canonical hash of the claim registry (order-independent)."""
    rows = []
    for c in claims:
        rows.append(json.dumps({
            "id": c.id, "kind": c.kind, "statement": c.statement,
            "mechanisms": sorted(c.mechanisms),
            "witnesses": sorted((w.strength, w.anchor) for w in c.witnesses),
            "lifecycle": c.lifecycle, "confidence": c.confidence,
            "deps": sorted(c.deps),
        }, sort_keys=True))
    return sha256_text("\n".join(sorted(rows)))


@dataclass
class Verdict:
    claim_id: str
    state: str          # "pass/current" | "pass/stale" | "fail" | "never"
    manifest: dict | None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.state in ("pass/current",)


def mechanism_file_hashes(claims, root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in claims:
        h = hashlib.sha256()
        for m in c.mechanisms:
            path = m.split("::", 1)[0]
            p = root / path
            if p.suffix in (".cu", ".cuh", ".cpp", ".h", ".hpp") and p.exists():
                h.update(sha256_file(p).encode())
        out[c.id] = h.hexdigest()
    return out


def witness_file_hashes(claims, root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in claims:
        h = hashlib.sha256()
        seen = set()
        for w in c.witnesses:
            p = root / w.path
            if p.exists() and str(p) not in seen:
                h.update(sha256_file(p).encode())
                seen.add(str(p))
        out[c.id] = h.hexdigest()
    return out


def new_manifest(*, name: str, root: Path, binary: str, cuda: str, gpu: str,
                 seed: str, results: dict[str, str], claims, machine: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": SCHEMA,
        "runId": f"{name}-{now[:19].replace(':', '')}",
        "name": name,
        "generatedAt": now,
        "binary": binary,
        "binarySha256": sha256_file(root / binary) if (root / binary).exists() else "",
        "cuda": cuda,
        "gpu": gpu,
        "seed": seed,
        "machineContract": machine,
        "specHash": spec_hash(claims),
        "claims": {cid: {"witness": "", "result": res} for cid, res in sorted(results.items())},
        "mechanismHashes": mechanism_file_hashes(claims, root),
        "witnessHashes": witness_file_hashes(claims, root),
    }
    return manifest


def save_manifest(evidence_dir: Path, manifest: dict) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"{manifest['runId']}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_manifests(evidence_dir: Path) -> list[dict]:
    out = []
    if not evidence_dir.exists():
        return out
    for p in sorted(evidence_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("schema") == SCHEMA:
                data["_path"] = str(p)
                out.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def latest_for_claim(manifests: list[dict], claim_id: str) -> dict | None:
    cands = [m for m in manifests if claim_id in m.get("claims", {})]
    if not cands:
        return None
    return max(cands, key=lambda m: m.get("generatedAt", ""))


def verdict(claim, manifests: list[dict], root: Path) -> Verdict:
    man = latest_for_claim(manifests, claim.id)
    if man is None:
        return Verdict(claim.id, "never", None,
                       "no evidence manifest references this claim")
    entry = man["claims"][claim.id]
    result = entry.get("result", "unknown")
    current_mech = mechanism_file_hashes([claim], root)[claim.id]
    stale = man.get("mechanismHashes", {}).get(claim.id) != current_mech
    current_wit = witness_file_hashes([claim], root)[claim.id]
    stale = stale or man.get("witnessHashes", {}).get(claim.id) != current_wit
    if result != "pass":
        return Verdict(claim.id, "fail", man, f"latest witness result: {result}")
    if stale:
        return Verdict(claim.id, "pass/stale", man,
                       "mechanism or witness changed since this evidence")
    return Verdict(claim.id, "pass/current", man, "")


def record(args) -> int:
    """CLI: record a manifest from a completed run."""
    root = Path(args.root)
    results = {}
    for kv in args.result:
        if ":" not in kv:
            print(f"bad --result {kv!r} (expected claim:result)")
            return 2
        cid, res = kv.split(":", 1)
        results[cid] = res
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import architecture.claims as claims_mod
    import architecture.compiler as compiler_mod
    claims, errors = claims_mod.load_registry(compiler_mod.SPEC_DOCS)
    if errors:
        for e in errors:
            print(e)
        return 2
    machine = json.loads((root / "architecture/machine.json").read_text(encoding="utf-8"))
    manifest = new_manifest(name=args.name, root=root, binary=args.binary,
                            cuda=args.cuda, gpu=args.gpu, seed=args.seed,
                            results=results, claims=claims, machine=machine)
    path = save_manifest(root / "evidence", manifest)
    print(f"recorded {path}")
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Slime evidence manifests")
    sub = parser.add_subparsers(dest="cmd")

    rec = sub.add_parser("record")
    rec.add_argument("--root", default=".")
    rec.add_argument("--name", required=True)
    rec.add_argument("--binary", default="")
    rec.add_argument("--cuda", default="unknown")
    rec.add_argument("--gpu", default="unknown")
    rec.add_argument("--seed", default="default")
    rec.add_argument("--result", action="append", default=[],
                     help="claim:pass|fail (repeatable)")

    sub.add_parser("list")

    args = parser.parse_args()
    if args.cmd == "record":
        return record(args)
    if args.cmd == "list":
        for m in load_manifests(Path("evidence")):
            print(m.get("runId"), m.get("gpu"), m.get("cuda"),
                  ",".join(f"{k}={v['result']}" for k, v in sorted(m["claims"].items())))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

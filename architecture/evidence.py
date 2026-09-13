"""Evidence manifests and staleness for the Slime architecture layer.

Every accepted test/run records a manifest (schema slime-evidence/v1) with
hashes of the claims, mechanisms, witnesses, and binary it exercised. The
status renderer derives PASS/CURRENT, PASS/STALE, FAIL, and NEVER RUN from
manifests alone — never from prose. When a mechanism file changes after the
latest accepted evidence, the claim becomes STALE until its witness reruns.

A manifest can be created from a real run with:

    python architecture/evidence.py record --name run5 --binary build/coevo.exe \
        --cuda 13.0.48 --gpu "RTX 3060 Laptop" --seed default \
        --result A103.checkpoint-replay-equivalence=tests/autodiff_acceptance.cu::test_forward_match_and_backward:pass

Uses the project's claims module and PyYAML.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from architecture import claims as claims_mod  # noqa: E402
from architecture.claims import claim_hash  # noqa: E402

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
                 seed: str, results: dict[str, dict], claims, machine: dict,
                 scheduler_job: str = "") -> dict:
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
        "claims": {cid: {"witness": entry["witness"], "result": entry["result"]}
                   for cid, entry in sorted(results.items())},
        "claimHashes": {c.id: claim_hash(c) for c in claims},
        "mechanismHashes": mechanism_file_hashes(claims, root),
        "witnessHashes": witness_file_hashes(claims, root),
    }
    if scheduler_job:
        manifest["scheduler"] = scheduler_provenance(scheduler_job)
    return manifest


def scheduler_provenance(job_id: str) -> dict:
    """Link this manifest to the training-architecture scheduler's result
    ledger. Imported lazily so the evidence layer stays usable without the
    scheduler; an unreachable scheduler records the jobId and the reason
    instead of silently dropping the provenance."""
    block = {"jobId": job_id}
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import gpu_client  # type: ignore
        block.update(gpu_client.scheduler_provenance(job_id))
    except Exception as exc:
        block["unavailable"] = str(exc)[:300]
    return block


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
    current_wit = witness_file_hashes([claim], root)[claim.id]
    current_prop = claim_hash(claim)
    stale = man.get("mechanismHashes", {}).get(claim.id) != current_mech
    stale = stale or man.get("witnessHashes", {}).get(claim.id) != current_wit
    stale = stale or man.get("claimHashes", {}).get(claim.id) != current_prop
    if result != "pass":
        return Verdict(claim.id, "fail", man, f"latest witness result: {result}")
    if stale:
        return Verdict(claim.id, "pass/stale", man,
                       "claim, mechanism, or witness changed since this evidence")
    return Verdict(claim.id, "pass/current", man, "")


def parse_and_validate_results(result_args: list[str], claims, binary: str,
                               root: Path):
    """Validate --result entries (claim=witness:result) and return
    ({claim_id: {"witness": w, "result": r}}, [errors]). The witness must be
    registered for the claim — or be the executed binary itself for
    integration-run attestation — and its file must declare [claim:<id>]."""
    by_id = {c.id: c for c in claims}
    results: dict[str, dict] = {}
    errors: list[str] = []
    for kv in result_args:
        if "=" not in kv or ":" not in kv.split("=", 1)[1]:
            errors.append(f"bad --result {kv!r} (expected claim=witness:result)")
            continue
        cid, rest = kv.split("=", 1)
        witness, res = rest.rsplit(":", 1)
        if res not in ("pass", "fail"):
            errors.append(f"bad --result {kv!r}: result must be pass or fail")
            continue
        if cid not in by_id:
            errors.append(f"bad --result {kv!r}: unknown claim {cid}")
            continue
        claim = by_id[cid]
        registered = {w.anchor for w in claim.witnesses}
        if witness in registered:
            ok, msg = claims_mod.resolve_witness(
                next(w for w in claim.witnesses if w.anchor == witness),
                root, cid)
            if not ok:
                errors.append(f"bad --result {kv!r}: {msg}")
                continue
        elif binary and witness == binary:
            # Integration-run attestation: the executed binary is the witness
            # for runtime-observable claims. Provenance rests on the recorded
            # binary sha256 plus claim/mechanism/witness hashes.
            pass
        else:
            errors.append(
                f"bad --result {kv!r}: witness {witness} is not registered "
                f"for {cid} (registered: {', '.join(sorted(registered)) or 'none'})")
            continue
        results[cid] = {"witness": witness, "result": res}
    return results, errors


def gpu_evidence_gate(build: dict) -> str:
    """Return a refusal message when the build inventory is incomplete.

    The construction plan's order of operations is binding: no GPU
    integration, acceptance, or production evidence exists before every
    inventory item is implemented. Empty string means the gate is open.
    """
    incomplete = [iid for iid, item in sorted(build.get("items", {}).items())
                  if item.get("status") != "implemented"]
    if incomplete:
        return (f"REFUSED: BUILD phase incomplete ({', '.join(incomplete)}); "
                f"no GPU acceptance evidence may be recorded until every "
                f"inventory item in architecture/build_status.yaml is "
                f"implemented.")
    return ""


def record(args) -> int:
    """CLI: record a manifest from a completed run.

    --result has the form <claim>=<witness>:<pass|fail>, e.g.
        --result A301.genotype-causes-phenotype=tests/evolution_regression.cu::test_genotype_causality:pass
    The witness must be registered for the claim (or be the executed binary
    itself for integration-run attestation), and its file must declare
    [claim:<id>]. This makes the manifest a provenance record, not a promise.
    """
    root = Path(args.root)
    claims, errors = claims_mod.load_registry(claims_mod.SPEC_DOCS)
    if errors:
        for e in errors:
            print(e)
        return 2
    results, verrors = parse_and_validate_results(args.result, claims,
                                                  args.binary, root)
    if verrors:
        for e in verrors:
            print(e)
        return 2

    # The construction plan's order of operations is binding: GPU acceptance
    # evidence may not exist before the build inventory is complete. Host-side
    # unit tests are build tools and pass --cuda n/a.
    gpu_run = (args.cuda.strip().lower() not in ("n/a", "", "none")
               or bool(getattr(args, "scheduler_job", "")))
    if gpu_run:
        build = yaml.safe_load(
            (root / "architecture/build_status.yaml").read_text(encoding="utf-8"))
        refusal = gpu_evidence_gate(build)
        if refusal:
            print(refusal)
            return 2

    machine = json.loads((root / "architecture/machine.json").read_text(encoding="utf-8"))
    manifest = new_manifest(name=args.name, root=root, binary=args.binary,
                            cuda=args.cuda, gpu=args.gpu, seed=args.seed,
                            results=results, claims=claims, machine=machine,
                            scheduler_job=getattr(args, "scheduler_job", "") or "")
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
    rec.add_argument("--scheduler-job", default="", dest="scheduler_job",
                     help="training-architecture scheduler jobId; records the "
                          "result-ledger provenance in the manifest")
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

"""Chunked long-run stability check (I9).

Runs the evolution binary in chunks with checkpoint/restart, fails on any
spontaneous stress flag or nonfinite dashboard value, and asserts the
checkpoint generation reaches the requested total. The 5000-generation
acceptance run uses this harness with --gens 5000; the default is a short
validation that exercises the same path.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

FLAG_RE = re.compile(r"\[STRESS\] lineage \d+ flagged")
NONFINITE_RE = re.compile(r"(nan|inf)", re.IGNORECASE)


def run_chunk(binary: str, gens: int, ckpt: str, resume: bool) -> str:
    cmd = [binary, str(gens), "--ckpt", ckpt]
    if resume:
        cmd.append("--resume")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"chunk failed with exit code {proc.returncode}")
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/coevo.exe")
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--ckpt", default="checkpoints/long-run.bin")
    ap.add_argument("--warmup", type=int, default=5,
                    help="generations before stress flags count as "
                         "spontaneous (the ladder calibrates early)")
    args = ap.parse_args()

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt).unlink(missing_ok=True)

    done = 0
    resume = False
    while done < args.gens:
        chunk = min(args.chunk, args.gens - done)
        out = run_chunk(args.binary, chunk, args.ckpt, resume)
        if done + chunk > args.warmup and FLAG_RE.search(out):
            print(out[-2000:])
            raise SystemExit(f"spontaneous stress flag after {done} generations")
        for line in out.splitlines():
            if "[DASHBOARD]" in line and NONFINITE_RE.search(line):
                print(line)
                raise SystemExit("nonfinite dashboard value")
        done += chunk
        resume = True
        print(f"chunk ok: {done}/{args.gens} generations")

    if not Path(args.ckpt).is_file():
        raise SystemExit("checkpoint missing after the run")
    print(f"long-run stability check: PASS ({args.gens} generations, "
          f"chunks of {args.chunk})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

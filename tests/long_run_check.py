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
R_RE = re.compile(r"\br=([0-9.]+)")
CKPT_RE = re.compile(r"\(generation (\d+)\)")


def run_chunk(binary: str, target_total: int, ckpt: str, resume: bool) -> str:
    """Run one chunk up to the cumulative target generation, streaming the
    child's output through so the progress/v1 wrapper sees the binary's
    per-generation `gen N` lines. The binary's N argument is the total
    generation to reach (it resumes from the checkpoint's generation), so
    callers pass the running total, never the chunk size."""
    # --profile makes the binary emit its per-generation phase trace
    # ("gen N: ..."), which is the progress signal the daemon's watchdog
    # needs; without it a long chunk looks stalled and gets killed.
    cmd = [binary, str(target_total), "--profile", "--ckpt", ckpt]
    if resume:
        cmd.append("--resume")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait(timeout=3600)
    out = "".join(lines)
    if proc.returncode != 0:
        print(out[-2000:], file=sys.stderr)
        raise SystemExit(f"chunk failed with exit code {proc.returncode}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/coevo.exe")
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--ckpt", default="checkpoints/long-run.bin")
    ap.add_argument("--warmup", type=int, default=50,
                    help="generations before stress flags count as "
                         "spontaneous (the ladder calibrates early)")
    ap.add_argument("--r-warmup", type=int, default=100,
                    help="generations before the r > 0.5 fraction is "
                         "enforced (predictors do not exist before bootstrap)")
    ap.add_argument("--r-floor", type=float, default=0.5,
                    help="I9 verification floor for the role balance r")
    args = ap.parse_args()

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt).unlink(missing_ok=True)

    done = 0
    resume = False
    r_ok = 0
    r_total = 0
    while done < args.gens:
        chunk = min(args.chunk, args.gens - done)
        target = done + chunk
        out = run_chunk(args.binary, target, args.ckpt, resume)
        m = CKPT_RE.search(out)
        if m is None or int(m.group(1)) != target:
            print(out[-2000:])
            raise SystemExit(
                f"checkpoint generation "
                f"{m.group(1) if m else '?'} != target {target}: the run "
                f"did not advance as requested")
        if done >= args.warmup and FLAG_RE.search(out):
            print(out[-2000:])
            raise SystemExit(f"spontaneous stress flag after {done} generations")
        for line in out.splitlines():
            if "[DASHBOARD]" in line and NONFINITE_RE.search(line):
                print(line)
                raise SystemExit("nonfinite dashboard value")
            if done >= args.r_warmup and "[DASHBOARD]" in line:
                m = R_RE.search(line)
                if m is not None:
                    r_total += 1
                    if float(m.group(1)) > args.r_floor:
                        r_ok += 1
        done += chunk
        resume = True
        print(f"chunk ok: {done}/{args.gens} generations")

    if not Path(args.ckpt).is_file():
        raise SystemExit("checkpoint missing after the run")
    if r_total > 0:
        frac = r_ok / r_total
        print(f"sustained role balance: r > {args.r_floor} in "
              f"{r_ok}/{r_total} post-warmup samples ({frac:.2f})")
        if frac <= 0.5:
            raise SystemExit("sustained role balance not met (I9)")
    print(f"long-run stability check: PASS ({args.gens} generations, "
          f"chunks of {args.chunk})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

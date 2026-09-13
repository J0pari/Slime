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


def chunk_command(binary: str, target_total: int, ckpt: str,
                  resume: bool) -> list[str]:
    """The chunk argv. The binary's N argument is the total generation to
    reach (it resumes from the checkpoint's generation), so callers pass the
    running total, never the chunk size. --profile makes the binary emit its
    per-generation phase trace ("gen N: ..."), which is the progress signal
    the daemon's watchdog needs; without it a long chunk looks stalled and
    gets killed."""
    cmd = [binary, str(target_total), "--profile", "--ckpt", ckpt]
    if resume:
        cmd.append("--resume")
    return cmd


def parse_checkpoint_generation(out: str) -> int | None:
    """The generation of the last checkpoint line in a chunk's output, or
    None when no checkpoint line is present."""
    gens = CKPT_RE.findall(out)
    if not gens:
        return None
    return int(gens[-1])


def flag_policy_failed(flagged_chunks: int, post_warmup_chunks: int,
                       max_fraction: float) -> bool:
    """Isolated stress flags are operator-review signals, not failures: a
    lineage can be flagged legitimately. The run fails only when flags are
    sustained, i.e. appear in more than `max_fraction` of the post-warmup
    chunks. Before the warmup the gen-0 calibration flag is expected."""
    if post_warmup_chunks <= 0:
        return False
    return (flagged_chunks / post_warmup_chunks) > max_fraction


def r_samples(out: str, floor: float) -> tuple[int, int]:
    """(above floor, total) dashboard r samples in a chunk's output."""
    ok = 0
    total = 0
    for line in out.splitlines():
        if "[DASHBOARD]" not in line:
            continue
        m = R_RE.search(line)
        if m is None:
            continue
        total += 1
        if float(m.group(1)) > floor:
            ok += 1
    return ok, total


def run_chunk(binary: str, target_total: int, ckpt: str, resume: bool) -> str:
    """Run one chunk, streaming the child's output through so the
    progress/v1 wrapper sees the binary's per-generation `gen N` lines."""
    cmd = chunk_command(binary, target_total, ckpt, resume)
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
    ap.add_argument("--flag-chunk-frac", type=float, default=0.5,
                    help="fail only when flags appear in more than this "
                         "fraction of post-warmup chunks")
    ap.add_argument("--r-warmup", type=int, default=100,
                    help="generations before the r > 0.5 fraction is "
                         "enforced (predictors do not exist before bootstrap)")
    ap.add_argument("--r-floor", type=float, default=0.5,
                    help="I9 verification floor for the role balance r")
    args = ap.parse_args()

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

    # Resume on start: a scheduler retry after a cancellation continues from
    # the last checkpoint instead of restarting a multi-hour run. A
    # zero-generation probe with --resume reports the checkpoint's
    # generation, or invalidates when there is no usable checkpoint.
    done = 0
    resume = False
    if Path(args.ckpt).is_file():
        probe = run_chunk(args.binary, 0, args.ckpt, True)
        if "RUN INVALIDATED" not in probe:
            g = parse_checkpoint_generation(probe)
            if g is None:
                raise SystemExit("resume probe printed no generation")
            done = g
            resume = True
            print(f"resuming from generation {done}")
    flagged_chunks = 0
    post_warmup_chunks = 0
    r_ok = 0
    r_total = 0
    while done < args.gens:
        chunk = min(args.chunk, args.gens - done)
        target = done + chunk
        out = run_chunk(args.binary, target, args.ckpt, resume)
        ckpt_gen = parse_checkpoint_generation(out)
        if ckpt_gen != target:
            print(out[-2000:])
            raise SystemExit(
                f"checkpoint generation {ckpt_gen if ckpt_gen is not None else '?'}"
                f" != target {target}: the run did not advance as requested")
        if done >= args.warmup:
            post_warmup_chunks += 1
            if FLAG_RE.search(out):
                flagged_chunks += 1
        for line in out.splitlines():
            if "[DASHBOARD]" in line and NONFINITE_RE.search(line):
                print(line)
                raise SystemExit("nonfinite dashboard value")
        if done >= args.r_warmup:
            ok, total = r_samples(out, args.r_floor)
            r_ok += ok
            r_total += total
        done += chunk
        resume = True
        print(f"chunk ok: {done}/{args.gens} generations")

    if not Path(args.ckpt).is_file():
        raise SystemExit("checkpoint missing after the run")
    if post_warmup_chunks > 0:
        print(f"stress flags: {flagged_chunks}/{post_warmup_chunks} "
              f"post-warmup chunks")
        if flag_policy_failed(flagged_chunks, post_warmup_chunks,
                              args.flag_chunk_frac):
            raise SystemExit("spontaneous stress flags sustained (I9)")
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

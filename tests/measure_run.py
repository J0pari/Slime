"""Scheduled measurement harness for I8 (phase budget and wall clock).

Sets the profiling environment, runs the binary, streams its output through
(the progress/v1 wrapper reads the per-generation `gen N` lines), and prints
one WALL line with the elapsed seconds. This is the only way timing numbers
are produced: submitted to the scheduler, never launched ad hoc.
"""
import argparse
import os
import subprocess
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/coevo.exe")
    ap.add_argument("--gens", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backward-profile", action="store_true",
                    help="set COEVO_BACKWARD_PROFILE=1")
    ap.add_argument("--profile", action="store_true",
                    help="pass --profile for the per-phase table")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.backward_profile:
        env["COEVO_BACKWARD_PROFILE"] = "1"
    cmd = [args.binary, str(args.gens), "--ckpt", args.ckpt]
    if args.profile:
        cmd.append("--profile")

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    wall = time.monotonic() - t0
    print(f"WALL {wall:.1f} s for {args.gens} generations")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

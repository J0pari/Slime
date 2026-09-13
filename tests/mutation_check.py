"""Gate mutation check: role canonicalization (VERIFY).

Proves the canonicalization witness detects the failure it exists to
prevent: a detached worktree is mutated (Reserved10 canonicalizes to
Predictor instead of Classifier), the host suite is built and run there,
and the check passes only if test_canonical_role goes red. CPU-only; no GPU
work, so it is safe while scheduled jobs run.

Requires the build toolchain in the environment (nvcc + MSVC), like the
Makefile host-tests target.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTATION_OLD = "case 0b10: return Role::Classifier;"
MUTATION_NEW = "case 0b10: return Role::Predictor;"


def run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvcc", default="nvcc")
    args = ap.parse_args()

    wt = Path(tempfile.mkdtemp(prefix="slime-mutation-"))
    try:
        rc, out = run(["git", "worktree", "add", "--detach", str(wt), "HEAD"],
                      ROOT)
        if rc != 0:
            print(out)
            return 1
        target = wt / "config" / "constants.cuh"
        text = target.read_text(encoding="utf-8")
        if text.count(MUTATION_OLD) != 1:
            print("mutation anchor not found exactly once")
            return 1
        target.write_text(text.replace(MUTATION_OLD, MUTATION_NEW),
                          encoding="utf-8")
        (wt / "build").mkdir(exist_ok=True)
        rc, out = run([args.nvcc, "-Itests/stubs", "-I.", "-std=c++17",
                       "-O2", "tests/host_unit_tests.cpp",
                       "-o", "build/host_tests.exe"], wt)
        if rc != 0:
            print("mutated build failed (unexpected):")
            print(out[-2000:])
            return 1
        rc, out = run([str(wt / "build" / "host_tests.exe")], wt)
        if "Reserved10" not in out or "FAIL" not in out:
            print("mutated suite did not go red on the canonicalization "
                  "witness (the witness does not detect the mutation)")
            print(out[-1500:])
            return 1
        print("mutation detected: test_canonical_role went red")
        print([l for l in out.splitlines() if "Reserved10" in l][:1])
        return 0
    finally:
        run(["git", "worktree", "remove", "--force", str(wt)], ROOT)
        shutil.rmtree(wt, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

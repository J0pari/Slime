"""Rebuild the canonical GPU binaries with the authorization guard.

Run once the scheduler queue is drained (see TODO): the queued jobs run the
pre-guard binaries, so rebuilding earlier would fail them. Builds the six
canonical binaries with the standard flags and asserts each refuses a bare
launch (exit 2) with COEVO_GPU_AUTHORIZED unset.

Requires the build toolchain in the environment (nvcc + MSVC), like the
Makefile targets.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ["-arch=sm_86", "-rdc=true", "--extended-lambda",
          "--expt-relaxed-constexpr", "-std=c++17", "-O2"]
BINARIES = [
    ("integration/host_main.cu", "build/coevo.exe", ["-lcudadevrt"]),
    ("tests/evolution_regression.cu", "build/evolution_regression.exe",
     ["-lcudadevrt", "-Xlinker", "/STACK:33554432"]),
    ("tests/checkpoint_state.cu", "build/checkpoint_state.exe",
     ["-lcudadevrt"]),
    ("tests/autodiff_acceptance.cu", "build/autodiff_acceptance.exe",
     ["-lcudadevrt"]),
    ("tests/task_conditioning.cu", "build/task_conditioning.exe",
     ["-lcudadevrt"]),
    ("tests/forward_smoke.cu", "build/forward_smoke.exe", ["-lcudadevrt"]),
]


def main() -> int:
    nvcc = os.environ.get("NVCC", "nvcc")
    for src, out, extra in BINARIES:
        cmd = [nvcc, *COMMON, src, "-o", out, *extra]
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            raise SystemExit(f"build failed: {src}")
        print(f"built {out}")

    env = {k: v for k, v in os.environ.items()
           if k != "COEVO_GPU_AUTHORIZED"}
    for src, out, extra in BINARIES:
        proc = subprocess.run([str(ROOT / out), "0"], cwd=ROOT, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 2:
            raise SystemExit(
                f"guard check failed for {out}: exit {proc.returncode}")
    print(f"rebuilt and guard-checked {len(BINARIES)} binaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

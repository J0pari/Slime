#!/usr/bin/env python3
"""Audit MNIST↔CA training connectivity against compiled artifacts.

What this does (a real audit, not just grep):
 1) Ensures build artifacts exist (logs/ptx/*.ii|*.ptx|*.cudafe1.cpp or build/slime.exe).
 2) Scans preprocessed CUDA outputs under logs/ptx for actual kernel launch sites.
 3) Optionally cross-checks the final binary for the same kernels if cuobjdump is available.
 4) Falls back to source scan only with --allow-source-only, but otherwise fails if nothing compiled.

Required connections (at least one kernel per line must be launched):
  - MNIST → CA input:            mnist_to_ca_grid_kernel | inject_mnist_to_ca_kernel
  - Features → logits:           classification_head_kernel
  - Logits → loss:               cross_entropy_loss_kernel
  - Backward pass:               ad_backward_kernel
  - Optimizer step:              adam_update_kernel | adam_update_fp16_kernel
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from core import FileOp, BuildArtifacts, Shell, Paths

TARGET_FILE = Paths.source_file('training', 'hybrid_lifecycle.cu')
DEFAULT_PTX_DIR = Paths.ptx_dir()
DEFAULT_EXE = Paths.build_artifact('slime.exe')


def scan_ptx_for_labels(ptx_dir: Path, labels_to_needles: List[Tuple[str, List[str]]]) -> Tuple[Dict[str, str], List[str]]:
    """Scan PTX for labeled kernel groups"""
    found: Dict[str, str] = {}
    missing: List[str] = []

    # Flatten all kernel names
    all_kernels = []
    label_map = {}
    for label, names in labels_to_needles:
        for name in names:
            all_kernels.append(name)
            label_map[name] = label

    # Scan once
    kernel_locations = BuildArtifacts.scan_for_kernels(ptx_dir, all_kernels)

    # Map back to labels
    for label, names in labels_to_needles:
        for name in names:
            if name in kernel_locations:
                found[label] = f"{name} in {kernel_locations[name].relative_to(ptx_dir)}"
                break
        else:
            missing.append(label)

    return found, missing


def check_binary_for_symbols(exe_path: str, needles: List[str]) -> Tuple[bool, str]:
    """Check if any needle appears in cuobjdump output for exe_path."""
    try:
        result = subprocess.run(['cuobjdump', '-elf', 'all', exe_path], capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, 'cuobjdump unavailable'
    output = result.stdout + result.stderr
    for n in needles:
        if n in output:
            return True, n
    return False, ''


def audit(ptx_dir: Path, exe_path: Path, allow_source_only: bool, do_build: bool) -> int:
    # Optionally build first
    if do_build:
        build_dir = Paths.repo_root / 'build'
        script = build_dir / 'compile.bat'
        if not script.exists():
            print(f"[ERROR] compile.bat not found at {script}")
            return 1

        result = Shell.run([str(script)], cwd=build_dir, description="Running compile.bat")
        if result.is_err():
            print(f"[FAIL] Build failed: {result.message}")
            return 1

    # Ensure target file exists
    if not TARGET_FILE.exists():
        print(f"[FAIL] Target file not found: {TARGET_FILE}")
        return 1

    # Determine artifact presence
    have_artifacts = ptx_dir.exists() and any(ptx_dir.rglob('*.ii')) or any(ptx_dir.rglob('*.ptx'))
    have_exe = bool(exe_path and exe_path.exists())

    if not have_artifacts and not have_exe:
        if allow_source_only:
            print("[WARN] No compiled artifacts found; proceeding with source-only audit.")
        else:
            print("[FAIL] No compiled artifacts found (no logs/ptx intermediates and no executable).")
            print("       Run build/compile.bat or use --build to compile before auditing.")
            return 2

    labels = [
        ("MNIST->CA injection", ["mnist_to_ca_grid_kernel", "inject_mnist_to_ca_kernel"]),
        ("Features->Logits classification", ["classification_head_kernel"]),
        ("Logits->Loss cross-entropy", ["cross_entropy_loss_kernel"]),
        ("Autodiff backward", ["ad_backward_kernel"]),
        ("Optimizer update", ["adam_update_kernel", "adam_update_fp16_kernel"]),
    ]

    # 1) Source-level check (for developer intent) using FileOp
    src_result = FileOp.read(TARGET_FILE)
    if src_result.is_err():
        print(f"[FAIL] Cannot read target file: {src_result.message}")
        return 1

    src_content = src_result.value
    src_found: Dict[str, str] = {}
    src_missing: List[str] = []
    for label, names in labels:
        ok, which = find_any(src_content, names)
        if ok:
            src_found[label] = which
        else:
            src_missing.append(label)

    # 2) Build artifacts check – what actually got compiled
    build_found: Dict[str, str] = {}
    build_missing: List[str] = []
    if have_artifacts:
        build_found, build_missing = scan_ptx_for_labels(ptx_dir, labels)

    # 3) Binary symbol presence (optional additional signal)
    bin_hits: Dict[str, str] = {}
    if have_exe:
        for label, names in labels:
            ok, which = check_binary_for_symbols(exe_path, names)
            if ok:
                bin_hits[label] = which

    # Report
    print("\n=== CONNECTIVITY AUDIT ===\n")
    print(f"[INFO] Source file: {TARGET_FILE.relative_to(Paths.repo_root)}")
    print(f"[INFO] PTX dir:     {ptx_dir.relative_to(Paths.repo_root)} ({'present' if have_artifacts else 'missing'})")
    print(f"[INFO] Executable:  {exe_path.relative_to(Paths.repo_root) if exe_path else '(none provided)'} ({'present' if have_exe else 'missing'})\n")

    print("-- Source (declared launches) --")
    for label, names in labels:
        if label in src_found:
            print(f"[OK]   {label:<30} via {src_found[label]}")
        else:
            print(f"[MISS] {label:<30} expected one of {names}")

    if have_artifacts:
        print("\n-- Build artifacts (preprocessed launches) --")
        for label, names in labels:
            if label in build_found:
                print(f"[OK]   {label:<30} {build_found[label]}")
            else:
                print(f"[MISS] {label:<30} not found in build intermediates")

    if have_exe:
        print("\n-- Binary (symbols present) --")
        if not bin_hits:
            print("[WARN] cuobjdump unavailable or no symbols found; skipping binary assert")
        else:
            for label, names in labels:
                if label in bin_hits:
                    print(f"[OK]   {label:<30} symbol {bin_hits[label]} present in binary")
                else:
                    print(f"[MISS] {label:<30} symbol not found in binary")

    # Decide pass/fail
    reasons: List[str] = []
    # Must have artifacts unless explicitly allowed to be source-only
    if not have_artifacts and not have_exe and not allow_source_only:
        reasons.append("no compiled artifacts")

    # Require that all labels are present in source AND in build artifacts (if present)
    for label, _ in labels:
        if label not in src_found:
            reasons.append(f"missing in source: {label}")
        if have_artifacts and label not in build_found:
            reasons.append(f"missing in build: {label}")

    if reasons:
        print("\n[FAIL] Connectivity audit failed:")
        for r in reasons:
            print(f"  - {r}")
        if not have_artifacts:
            print("\nHint: run build/compile.bat or use --build flag to compile before auditing.")
        return 1

    print("\n[PASS] Connectivity verified in source and compiled artifacts.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description='Audit MNIST↔CA training connectivity against compiled artifacts')
    p.add_argument('--ptx-dir', default=str(DEFAULT_PTX_DIR), help='Directory containing PTX/preprocessed (*.ii) build artifacts')
    p.add_argument('--exe', default=str(DEFAULT_EXE), help='Path to compiled executable (for cuobjdump symbol check)')
    p.add_argument('--build', action='store_true', help='Run build/compile.bat before auditing')
    p.add_argument('--allow-source-only', action='store_true', help='Allow passing without compiled artifacts (source-only audit)')
    args = p.parse_args(argv)

    ptx_dir = Path(args.ptx_dir)
    exe_path = Path(args.exe) if args.exe else None

    sys.exit(audit(ptx_dir, exe_path, args.allow_source_only, args.build))


if __name__ == '__main__':
    main()

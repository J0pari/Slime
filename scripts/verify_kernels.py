#!/usr/bin/env python3
"""Verify all expected kernels were compiled into the build artifacts.

Enhancements:
- exe_path is now optional; if omitted or not found, the ELF/cubin check is skipped.
- PTX scan now also scans CUDA preprocessed files (*.ii) for kernel definitions/usages.
- Expanded expected kernel list to better cover the MNIST gradient path.
"""

import sys
import os
import subprocess
from pathlib import Path
from core import BuildArtifacts

EXPECTED_KERNELS = [
    "hybrid_organism_lifecycle_kernel",
    "organism_lifecycle_kernel",
    "component_evolution_kernel",
    "neural_ca_update_kernel",
    "behavioral_update_kernel",
    "hierarchical_lifecycle_kernel",
    "sample_batch_kernel",
    "unified_sample_to_ca_kernel",
    "inject_sample_to_ca_kernel",
    "spatial_pooling_kernel",
    "classification_head_kernel",
    "cross_entropy_loss_kernel",
    "ad_backward_kernel",
    "adam_update_kernel",
    "adam_update_fp16_kernel",
    "resource_flow_kernel",
]

def check_ptx_files(ptx_dir):
    """Check PTX files for kernel definitions"""
    found = BuildArtifacts.scan_for_kernels(Path(ptx_dir), EXPECTED_KERNELS)
    return {k: str(v) for k, v in found.items()}

def check_cubin(exe_path):
    """Use cuobjdump to check compiled kernels in executable."""
    try:
        result = subprocess.run(
            ['cuobjdump', '--list-text', exe_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout + result.stderr
        found = []
        for kernel in EXPECTED_KERNELS:
            if kernel in output:
                found.append(kernel)
        return found
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: verify_kernels.py <ptx_dir> [exe_path]")
        sys.exit(1)

    ptx_dir = sys.argv[1]
    exe_path = sys.argv[2] if len(sys.argv) >= 3 else None

    print("\n=== KERNEL COMPILATION VERIFICATION ===\n")

    # Check PTX files
    ptx_kernels = check_ptx_files(ptx_dir)
    print(f"[PTX] Found {len(ptx_kernels)}/{len(EXPECTED_KERNELS)} kernels in intermediate files:")
    for kernel in EXPECTED_KERNELS:
        if kernel in ptx_kernels:
            rel_path = os.path.relpath(ptx_kernels[kernel], ptx_dir)
            print(f"  [OK] {kernel:<40} in {rel_path}")
        else:
            print(f"  [MISS] {kernel:<40} NOT COMPILED")

    # Check executable
    cubin_kernels = None
    if exe_path and os.path.exists(exe_path):
        print(f"\n[EXE] Checking final binary: {exe_path}")
        cubin_kernels = check_cubin(exe_path)
        if cubin_kernels is not None:
            print(f"  Found {len(cubin_kernels)}/{len(EXPECTED_KERNELS)} kernels in binary")
            missing_in_binary = set(EXPECTED_KERNELS) - set(cubin_kernels)
            if missing_in_binary:
                print(f"  [MISS] Missing from binary: {', '.join(missing_in_binary)}")
        else:
            print("  [SKIP] cuobjdump not available")
    else:
        print("\n[EXE] Skipping binary check (no exe_path provided or file not found)")

    # Summary and RDC nuance detection
    missing_ptx = set(EXPECTED_KERNELS) - set(ptx_kernels.keys())

    if missing_ptx:
        # Check if RDC might be hiding the kernels
        if cubin_kernels is not None:
            missing_in_binary = set(EXPECTED_KERNELS) - set(cubin_kernels)
            kernels_only_in_binary = missing_ptx & set(cubin_kernels)

            if kernels_only_in_binary:
                print(f"\n[RDC-INLINED] {len(kernels_only_in_binary)} kernels missing from PTX but present in binary (device-side calls inlined by RDC):")
                for k in sorted(kernels_only_in_binary):
                    print(f"  - {k}")

            if missing_in_binary:
                print(f"\n[CRITICAL-FAIL] {len(missing_in_binary)} kernels missing from BOTH PTX and binary:")
                for k in sorted(missing_in_binary):
                    print(f"  - {k}")
                print("  These kernels are either not defined, not called, or compilation failed silently!")
                sys.exit(1)

            still_missing_ptx = missing_ptx - kernels_only_in_binary
            if still_missing_ptx:
                print(f"\n[WARN] {len(still_missing_ptx)} kernels missing from PTX and not checked in binary")
        else:
            print(f"\n[FAIL] Missing kernels in PTX: {', '.join(missing_ptx)}")
            print("       These kernels were never compiled OR were inlined by RDC (can't verify without cuobjdump)")
            print("       RISK: Could be real compilation failures hidden by RDC assumption!")
            sys.exit(1)

    # Final verdict
    if cubin_kernels is not None:
        if len(cubin_kernels) == len(EXPECTED_KERNELS):
            print("\n[PASS] All expected kernels verified in final binary")
        else:
            print(f"\n[PARTIAL] {len(cubin_kernels)}/{len(EXPECTED_KERNELS)} kernels in binary")
    else:
        if not missing_ptx:
            print("\n[PASS] All expected kernels found in PTX compilation artifacts")
        else:
            print("\n[UNKNOWN] Cannot verify - kernels missing from PTX, no binary check available")

    print("\n=======================================\n")

if __name__ == '__main__':
    main()

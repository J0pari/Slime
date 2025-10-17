#!/usr/bin/env python3
"""
Fix all cudaDeviceSynchronize() calls in device kernels.
These need to be removed since dynamic parallelism handles synchronization implicitly.
"""

import os
import re

def fix_cuda_sync_in_file(filepath):
    """Fix cudaDeviceSynchronize calls in a single file."""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Pattern to match cudaDeviceSynchronize(); with optional whitespace
    patterns = [
        (r'(\s*)cudaDeviceSynchronize\(\);', r'\1// Dynamic parallelism: parent waits for children'),
        (r'(\s*)::cudaDeviceSynchronize\(\);', r'\1// Dynamic parallelism: parent waits for children'),
    ]

    changes_made = False
    for pattern, replacement in patterns:
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            print(f"  Fixed {count} cudaDeviceSynchronize() calls with pattern: {pattern}")
            content = new_content
            changes_made = True

    if changes_made:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False

def main():
    """Find and fix all CUDA files."""

    cuda_files = [
        'c:/Slime/slime/core/organism.cu',
        'c:/Slime/slime/core/pseudopod.cu',
        'c:/Slime/slime/core/chemotaxis.cu',
        'c:/Slime/slime/memory/archive.cu',
        'c:/Slime/slime/memory/pool.cu',
        'c:/Slime/slime/memory/tubes.cu',
        'c:/Slime/slime/api/gpu_native.cu'
    ]

    print("Scanning for cudaDeviceSynchronize() calls in device kernels...")

    total_fixed = 0
    for filepath in cuda_files:
        if os.path.exists(filepath):
            print(f"\nChecking {filepath}...")
            if fix_cuda_sync_in_file(filepath):
                total_fixed += 1
                print(f"  [FIXED] {os.path.basename(filepath)}")
            else:
                print(f"  No changes needed in {os.path.basename(filepath)}")
        else:
            print(f"  File not found: {filepath}")

    print(f"\n{'='*60}")
    print(f"Summary: Fixed cudaDeviceSynchronize() in {total_fixed} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
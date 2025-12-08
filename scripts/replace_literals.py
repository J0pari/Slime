#!/usr/bin/env python3
"""
Literal replacement - merges replace_dim_literals, replace_mnist_size, replace_specific_literals.

Original: 3 separate scripts, 307 lines total
New: Single composable script using core.Pattern

Replaces:
- Dimension literals: 16 → WMMA_TILE_DIM, 32 → WARP_SIZE, 256 → BLOCK_SIZE
- MNIST sizes: 784 → MNIST_INPUT_SIZE, 28 → MNIST_GRID_DIM
- Grid calculations: (x + 15) / 16 → (x + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM
- Identifiers: MNISTPerformanceMetrics -> TaskPerformanceMetrics, mnist_performance -> task_performance
"""

from pathlib import Path
from core import FileOp, Pattern, Scope, Ok, Err
import sys


# All dimension literal replacements
DIMENSION_REPLACEMENTS = [
    # Grid calculations with rounding
    (r'\(\s*(\w+)\s*\+\s*15\s*\)\s*/\s*16', r'(\1 + WMMA_TILE_DIM - 1) / WMMA_TILE_DIM'),
    (r'\(\s*(\w+)\s*\+\s*31\s*\)\s*/\s*32', r'(\1 + WARP_SIZE - 1) / WARP_SIZE'),
    (r'\(\s*(\w+)\s*\+\s*255\s*\)\s*/\s*256', r'(\1 + BLOCK_SIZE - 1) / BLOCK_SIZE'),

    # dim3 declarations
    (r'dim3\s+(\w+)\s*\(\s*(\w+)\s*/\s*16\s*,\s*(\w+)\s*/\s*16\s*\)', r'dim3 \1(\2/WMMA_TILE_DIM, \3/WMMA_TILE_DIM)'),
    (r'dim3\s+(\w+)\s*\(16\s*,\s*16\)', r'dim3 \1(WMMA_TILE_DIM, WMMA_TILE_DIM)'),
    (r'dim3\s+(\w+)\s*\(16\s*,\s*16\s*,\s*(\d+)\)', r'dim3 \1(WMMA_TILE_DIM, WMMA_TILE_DIM, \2)'),

    # Kernel launches
    (r'<<<\s*dim3\s*\(16\s*,\s*16\)', r'<<<dim3(WMMA_TILE_DIM, WMMA_TILE_DIM)'),
    (r',\s*dim3\s*\(16\s*,\s*16\)', r', dim3(WMMA_TILE_DIM, WMMA_TILE_DIM)'),

    # Thread indexing
    (r'threadIdx\.x\s*/\s*16', r'threadIdx.x / WMMA_TILE_DIM'),
    (r'threadIdx\.x\s*%\s*16', r'threadIdx.x % WMMA_TILE_DIM'),
    (r'threadIdx\.x\s*%\s*32', r'threadIdx.x % WARP_SIZE'),

    # Array declarations
    (r'\[\s*16\s*\]\s*\[\s*16\s*\]', r'[WMMA_TILE_DIM][WMMA_TILE_DIM]'),

    # Comparisons
    (r'if\s*\(\s*(\w+)\s*<\s*16\s*\)', r'if (\1 < WMMA_TILE_DIM)'),

    # Warp operations
    (r'WarpReduce<32>', r'WarpReduce<WARP_SIZE>'),
]

# MNIST-specific literal replacements
MNIST_REPLACEMENTS = [
    (r'\b784\b', 'MNIST_INPUT_SIZE'),
    (r'\b28\b(?!_)', 'MNIST_GRID_DIM'),  # 28 not followed by underscore
]

# Identifier generalization: MNIST->generic dataset terminology
# Core pattern: input SAMPLES (not images) that get mapped to CA grid
IDENTIFIER_REPLACEMENTS = [
    # Struct/type names
    (r'\bMNISTPerformanceMetrics\b', 'TaskPerformanceMetrics'),
    (r'\bMNISTDataset\b', 'Dataset'),

    # Field/variable names - use 'sample' as general pattern
    (r'\bmnist_performance\b', 'task_performance'),
    (r'\bmnist_dataset\b', 'dataset'),
    (r'\bmnist_data\b', 'sample_data'),
    (r'\bmnist_batch_images\b', 'batch_samples'),  # Must come before mnist_images
    (r'\bmnist_images\b', 'samples'),
    (r'\bmnist_labels\b', 'labels'),
    (r'\bmnist_image\b', 'sample'),
    (r'\bmnist_rows\b', 'sample_rows'),
    (r'\bmnist_cols\b', 'sample_cols'),
    (r'\bmnist_batch\b', 'batch'),
    (r'\bmnist_batch_labels\b', 'batch_labels'),
    (r'\bmnist_value\b', 'sample_value'),
    (r'\bimage_rows\b', 'sample_rows'),  # Fix previous incomplete replacement
    (r'\bimage_cols\b', 'sample_cols'),
    (r'\bpixel_value\b', 'sample_value'),
    (r'\bimage_to_ca_grid_kernel\b', 'sample_to_ca_grid_kernel'),
    (r'\binject_image_to_ca_kernel\b', 'inject_sample_to_ca_kernel'),
    (r'\binject_image_to_ca\b', 'inject_sample_to_ca'),
    (r'\bdump_image_raw\b', 'dump_sample_raw'),

    # Dataset struct field renames
    (r'dataset->images\b', 'dataset->samples'),
    (r'->num_images\b', '->num_samples'),
    (r'->num_rows\b', '->sample_rows'),
    (r'->num_cols\b', '->sample_cols'),

    # Function names
    (r'\bmnist_performance_probe_kernel\b', 'task_performance_probe_kernel'),
    (r'\bconvert_mnist_to_bin\b', 'convert_dataset_to_bin'),
    (r'\bconvert_mnist_to_cu\b', 'convert_dataset_to_cu'),
    (r'\bsample_mnist_batch_kernel\b', 'sample_batch_kernel'),
]


def replace_in_file(filepath: Path, replacements: list) -> int:
    """Apply replacements to a file, using scope awareness to avoid breaking code"""
    content_result = FileOp.read(filepath)
    if content_result.is_err():
        print(f"  Warning: {content_result.message}")
        return 0

    original = content_result.value

    # Apply all replacements using Pattern.replace_all
    transformed = Pattern.replace_all(replacements, original)

    if transformed == original:
        return 0

    # Verify at least one constant definition exists in file or is included
    # This prevents blind replacement in files that don't have access to constants
    has_config = '#include' in original and 'config' in original
    has_constexpr = 'constexpr' in original

    if not has_config and not has_constexpr:
        # Check if replacement introduces undefined constants
        new_constants = set()
        for pattern, replacement in replacements:
            if replacement and replacement[0].isupper():
                # Extract constant name from replacement
                import re
                const_match = re.findall(r'\b[A-Z_]+\b', replacement)
                new_constants.update(const_match)

        if new_constants and not has_config:
            print(f"  Warning: {filepath.name} doesn't include config, but replacement would use {new_constants}")
            print(f"           Skipping to avoid undefined constants")
            return 0

    # Write back
    write_result = FileOp.write(filepath, transformed)
    if write_result.is_err():
        print(f"  Error writing {filepath}: {write_result.message}")
        return 0

    # Count changed lines
    orig_lines = original.split('\n')
    new_lines = transformed.split('\n')
    changes = sum(1 for a, b in zip(orig_lines, new_lines) if a != b)

    return changes


def show_file_diff(filepath: Path, original: str, transformed: str, source_dir: Path):
    """Show diff for a single file and ask for approval"""
    rel_path = filepath.relative_to(source_dir)
    orig_lines = original.split('\n')
    new_lines = transformed.split('\n')

    print(f"\n{'='*70}")
    print(f"File: {rel_path}")
    print(f"{'='*70}")

    changes = []
    for i, (old, new) in enumerate(zip(orig_lines, new_lines), 1):
        if old != new:
            changes.append((i, old, new))

    for line_no, old, new in changes:
        print(f"\nLine {line_no}:")
        print(f"  - {old}")
        print(f"  + {new}")

    print(f"\n{len(changes)} lines would change")
    return changes


def main():
    from core import Paths
    source_dir = Paths.source_file()

    # Determine what to replace
    replace_dims = '--dims' in sys.argv or '--all' in sys.argv
    replace_mnist = '--mnist' in sys.argv or '--all' in sys.argv
    replace_identifiers = '--identifiers' in sys.argv or '--all' in sys.argv
    auto_yes = '--yes' in sys.argv

    if not replace_dims and not replace_mnist and not replace_identifiers:
        print("Usage:")
        print("  replace_literals.py --dims          # Replace dimension literals (interactive)")
        print("  replace_literals.py --mnist         # Replace MNIST literals (interactive)")
        print("  replace_literals.py --identifiers   # Replace MNIST→generic identifiers (interactive)")
        print("  replace_literals.py --all           # Replace all (interactive)")
        print("  replace_literals.py --dims --yes    # Auto-approve all changes (DANGEROUS)")
        sys.exit(1)

    print("=== Literal Replacement ===\n")
    if auto_yes:
        print("WARNING: Auto-approve mode - all changes will be applied without review!\n")

    # Scan for .cu and .cuh files
    cu_files_result = FileOp.scan(source_dir, "*.cu")
    cuh_files_result = FileOp.scan(source_dir, "*.cuh")
    if cu_files_result.is_err() or cuh_files_result.is_err():
        print(f"ERROR: {cu_files_result.message if cu_files_result.is_err() else cuh_files_result.message}")
        sys.exit(1)

    all_files = cu_files_result.value + cuh_files_result.value

    # Build replacement list
    replacements = []
    if replace_dims:
        replacements.extend(DIMENSION_REPLACEMENTS)
        print("Will replace dimension literals (16/32/256)")
    if replace_mnist:
        replacements.extend(MNIST_REPLACEMENTS)
        print("Will replace MNIST sizes (28/784)")
    if replace_identifiers:
        replacements.extend(IDENTIFIER_REPLACEMENTS)
        print("Will replace MNIST->generic identifiers (MNISTPerformanceMetrics->TaskPerformanceMetrics, etc.)")

    print()

    # Phase 1: Scan all files and collect proposed changes
    proposed_changes = []
    for filepath in all_files:
        content_result = FileOp.read(filepath)
        if content_result.is_err():
            continue

        original = content_result.value
        transformed = Pattern.replace_all(replacements, original)

        if transformed != original:
            proposed_changes.append((filepath, original, transformed))

    if not proposed_changes:
        print("No changes needed")
        return

    print(f"Found {len(proposed_changes)} files with proposed changes\n")

    # Phase 2: Show each file and ask for approval
    approved_changes = []
    for filepath, original, transformed in proposed_changes:
        changes = show_file_diff(filepath, original, transformed, source_dir)

        if auto_yes:
            print("  [AUTO-APPROVED]")
            approved_changes.append((filepath, transformed))
            continue

        # Interactive approval
        while True:
            response = input("\nApply changes? [y]es / [n]o / [s]kip file / [q]uit: ").lower()

            if response == 'y':
                approved_changes.append((filepath, transformed))
                print("  OK Changes approved")
                break
            elif response == 'n' or response == 's':
                print("  X Changes skipped")
                break
            elif response == 'q':
                print("\nAborting - no files modified")
                return
            else:
                print("Invalid input. Use y/n/s/q")

    # Phase 3: Apply approved changes
    if not approved_changes:
        print("\nNo changes approved")
        return

    print(f"\n{'='*70}")
    print(f"Applying {len(approved_changes)} approved changes...")
    print(f"{'='*70}\n")

    files_modified = 0
    for filepath, new_content in approved_changes:
        write_result = FileOp.write(filepath, new_content)
        if write_result.is_err():
            print(f"  X {filepath.relative_to(source_dir)}: {write_result.message}")
        else:
            print(f"  OK {filepath.relative_to(source_dir)}")
            files_modified += 1

    print(f"\nTotal: {files_modified} files modified")


if __name__ == '__main__':
    main()

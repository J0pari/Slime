#!/usr/bin/env python3
"""
Printf statement removal - removes printf/fprintf statements from CUDA source.

Handles multi-line statements by tracking parenthesis depth.
Preserves fprintf to FILE* (legitimate file I/O) while removing console output.

Usage:
    strip_printfs.py --console          # Remove printf() only (console noise)
    strip_printfs.py --all              # Remove all printf/fprintf
    strip_printfs.py --console --yes    # Auto-approve (no prompts)
"""

from pathlib import Path
from typing import List, Tuple
from core import FileOp, Ok, Err
import sys
import re


def find_statement_end(content: str, start: int) -> int:
    """
    Find the end of a C statement starting at 'start'.
    Tracks parenthesis depth to handle multi-line statements.
    Returns index after the closing semicolon.
    """
    depth = 0
    i = start
    in_string = False
    escape_next = False

    while i < len(content):
        c = content[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if c == '\\':
            escape_next = True
            i += 1
            continue

        if c == '"' and not in_string:
            in_string = True
        elif c == '"' and in_string:
            in_string = False
        elif not in_string:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ';' and depth == 0:
                return i + 1

        i += 1

    return len(content)


def strip_printf_statements(content: str, include_fprintf: bool = False) -> Tuple[str, int]:
    """
    Remove printf (and optionally fprintf) statements from content.

    Handles:
    - Standalone printf statements
    - Inline printf after code (e.g., if (x) printf(...))
    - fprintf to stderr (when include_fprintf=True)
    - Preserves fprintf to FILE* variables (legitimate file I/O)
    - Preserves snprintf (string formatting)

    Returns (new_content, count_removed).
    """
    removed = 0
    result = []
    last_end = 0

    # Find all printf/fprintf calls (not snprintf)
    if include_fprintf:
        # Match printf or fprintf, but not snprintf
        pattern = re.compile(r'(?<![a-z_])(?:printf|fprintf)\s*\(')
    else:
        pattern = re.compile(r'(?<![a-z_])printf\s*\(')

    matches = list(pattern.finditer(content))

    for match in matches:
        call_start = match.start()

        # Skip if already processed (overlapping with previous removal)
        if call_start < last_end:
            continue

        # Find where the statement ends (after semicolon)
        stmt_end = find_statement_end(content, call_start)
        stmt_text = content[call_start:stmt_end]

        # Check if this fprintf writes to a FILE* (not stderr/stdout)
        if 'fprintf' in match.group(0):
            first_arg_match = re.search(r'fprintf\s*\(\s*(\w+)', stmt_text)
            if first_arg_match:
                first_arg = first_arg_match.group(1)
                if first_arg not in ('stderr', 'stdout'):
                    # This is legitimate file I/O, keep it
                    continue

        # Find the start of this statement (go back to find statement boundary)
        # Look for: newline, semicolon, or opening brace before the printf
        stmt_start = call_start
        for i in range(call_start - 1, -1, -1):
            c = content[i]
            if c in ';\n{':
                stmt_start = i + 1
                break
            elif c not in ' \t':
                # Non-whitespace before printf - might be inline
                # Check if it's part of a control structure like "if (x) printf"
                break

        # Skip leading whitespace
        while stmt_start < call_start and content[stmt_start] in ' \t':
            stmt_start += 1

        # Add content before this statement
        result.append(content[last_end:stmt_start])

        # Skip any trailing whitespace/newline after the statement
        while stmt_end < len(content) and content[stmt_end] in ' \t':
            stmt_end += 1
        if stmt_end < len(content) and content[stmt_end] == '\n':
            stmt_end += 1

        # Also remove trailing fflush(stdout); if present
        fflush_match = re.match(r'\s*fflush\s*\(\s*stdout\s*\)\s*;[ \t]*\n?', content[stmt_end:])
        if fflush_match:
            stmt_end += fflush_match.end()

        last_end = stmt_end
        removed += 1

    # Add remaining content
    result.append(content[last_end:])

    return ''.join(result), removed


def process_file(filepath: Path, include_fprintf: bool) -> Tuple[str, str, int]:
    """
    Process a single file.
    Returns (original_content, new_content, count_removed).
    """
    result = FileOp.read(filepath)
    if result.is_err():
        return None, None, 0

    original = result.value
    new_content, count = strip_printf_statements(original, include_fprintf)

    return original, new_content, count


def show_diff(filepath: Path, original: str, new_content: str, source_dir: Path, count: int) -> int:
    """Show summary for a file."""
    rel_path = filepath.relative_to(source_dir)

    print(f"  {rel_path}: {count} printf statements")

    return count


def main():
    from core import Paths

    include_fprintf = '--all' in sys.argv
    console_only = '--console' in sys.argv
    auto_yes = '--yes' in sys.argv

    if not console_only and not include_fprintf:
        print(__doc__)
        print("\nMust specify --console or --all")
        sys.exit(1)

    print("=== Printf Removal ===\n")
    if include_fprintf:
        print("Mode: ALL (printf + fprintf to stderr/stdout)")
    else:
        print("Mode: CONSOLE (printf only)")

    if auto_yes:
        print("WARNING: Auto-approve mode enabled\n")

    # Scan slime/ and src/
    source_dirs = [Paths.repo_root / 'slime', Paths.repo_root / 'src']
    all_files = []

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        cu_result = FileOp.scan(source_dir, "*.cu")
        if not cu_result.is_err():
            all_files.extend(cu_result.value)
        cuh_result = FileOp.scan(source_dir, "*.cuh")
        if not cuh_result.is_err():
            all_files.extend(cuh_result.value)

    print(f"Scanning {len(all_files)} files...\n")

    # Phase 1: Find files with printfs
    proposed = []
    total_statements = 0

    for filepath in all_files:
        original, new_content, count = process_file(filepath, include_fprintf)
        if count > 0:
            proposed.append((filepath, original, new_content, count))
            total_statements += count

    if not proposed:
        print("No printf statements found")
        return

    print(f"Found {total_statements} printf statements in {len(proposed)} files\n")

    # Phase 2: Show diffs and get approval
    approved = []

    for filepath, original, new_content, count in proposed:
        # Determine source_dir for relative path
        source_dir = Paths.repo_root / 'slime' if 'slime' in str(filepath) else Paths.repo_root / 'src'

        show_diff(filepath, original, new_content, source_dir, count)

        if auto_yes:
            approved.append((filepath, new_content))
            continue

        while True:
            response = input("\nApply? [y]es / [n]o / [q]uit: ").lower()
            if response == 'y':
                approved.append((filepath, new_content))
                break
            elif response == 'n':
                print("  Skipped")
                break
            elif response == 'q':
                print("\nAborted")
                return

    # Phase 3: Apply
    if not approved:
        print("\nNo changes approved")
        return

    print(f"\n{'='*70}")
    print(f"Applying {len(approved)} files...")
    print(f"{'='*70}\n")

    for filepath, new_content in approved:
        result = FileOp.write(filepath, new_content)
        if result.is_err():
            print(f"  X {filepath.name}: {result.message}")
        else:
            print(f"  OK {filepath.name}")

    print(f"\nDone: {len(approved)} files modified")


if __name__ == '__main__':
    main()

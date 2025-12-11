#!/usr/bin/env python3
"""
Cut-and-paste lines between files - ONE atomic operation like Ctrl+X, Ctrl+V

Usage:
    python cut_paste.py <source> <dest> <start_line> <end_line>
"""

import sys
from pathlib import Path

def cut_paste_lines(source_file: Path, dest_file: Path, start_line: int, end_line: int):
    """
    Cut lines from source, paste to dest - atomic operation

    Args:
        source_file: File to cut from
        dest_file: File to paste to
        start_line: First line to cut (1-indexed)
        end_line: Last line to cut (1-indexed, inclusive)
    """
    # Read source
    with open(source_file, 'r') as f:
        lines = f.readlines()

    # Validate range
    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        raise ValueError(f"Invalid line range: {start_line}-{end_line} (file has {len(lines)} lines)")

    # CUT: extract lines and remove from source
    cut_content = lines[start_line-1:end_line]
    remaining = lines[:start_line-1] + lines[end_line:]

    # Write back source (atomic: delete happens here)
    with open(source_file, 'w') as f:
        f.writelines(remaining)

    # PASTE: append to dest
    with open(dest_file, 'a') as f:
        f.writelines(cut_content)

    print(f"OK Cut {len(cut_content)} lines from {source_file.name} -> {dest_file.name}")
    return len(cut_content)

def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)

    source = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    start = int(sys.argv[3])
    end = int(sys.argv[4])

    if not source.exists():
        print(f"ERROR: {source} does not exist")
        sys.exit(1)

    cut_paste_lines(source, dest, start, end)

if __name__ == "__main__":
    main()

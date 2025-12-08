#!/usr/bin/env python3
"""
Magic number detection - rewritten using core primitives.

Original: 341 lines with custom FileScanner, PatternMatcher, ConfigManager
New: ~50 lines composing FileOp, Pattern, ConstantSpace

Same semantics, cleaner composition.
"""

from pathlib import Path
from typing import List, Tuple
from core import FileOp, Pattern, ConstantSpace, Scope, Ok, Err, Result, bind
import sys


def find_magic_in_file(filepath: Path, constants: ConstantSpace) -> Result[List[Tuple[str, str, int]]]:
    """
    Find magic numbers in a file.
    Returns list of (value, context_line, line_number)
    """
    def process_content(content: str) -> Result[List[Tuple[str, str, int]]]:
        lines = content.split('\n')
        findings = []
        NUMERIC_PATTERN = r'\b(\d+\.?\d*[fF]?)\b'
        TRIVIAL = {'0', '1', '2', '3', '4', '0.0f', '1.0f', '2.0f', '0.5f', '-1.0f'}

        for line_no, line in enumerate(lines, 1):
            # Skip preprocessor, constexpr, comments
            if line.strip().startswith('#') or 'constexpr' in line:
                continue

            code_part = line.split('//')[0]

            for match in Pattern.match(NUMERIC_PATTERN, code_part):
                number = match.group(1)

                # Skip trivial numbers
                if number in TRIVIAL:
                    continue

                # Skip if already a known constant
                if constants.find_by_value(number):
                    continue

                # Skip scientific notation exponents
                pos = match.start()
                if pos > 0 and code_part[pos - 1] in ['e', 'E']:
                    continue

                findings.append((number, line.strip(), line_no))

        return Ok(findings)

    return bind(FileOp.read(filepath), process_content)


def scan_all_files(source_dir: Path, constants: ConstantSpace) -> Result[dict]:
    """Scan all .cu/.cuh files for magic numbers"""

    # Scan for CUDA files
    cu_result = FileOp.scan(source_dir, "*.cu")
    cuh_result = FileOp.scan(source_dir, "*.cuh")

    if cu_result.is_err():
        return cu_result
    if cuh_result.is_err():
        return cuh_result

    all_files = cu_result.value + cuh_result.value

    # Exclude config.cu
    config_path = source_dir / "config" / "config.cu"
    all_files = [f for f in all_files if f != config_path]

    # Find magic numbers in each file
    all_findings = {}
    for filepath in all_files:
        result = find_magic_in_file(filepath, constants)
        if result.is_err():
            print(f"Warning: {result.message}")
            continue

        if result.value:
            all_findings[filepath] = result.value

    return Ok(all_findings)


def report_findings(findings: dict, source_dir: Path):
    """Print findings grouped by value"""

    # Group by value
    by_value = {}
    for filepath, occurrences in findings.items():
        for value, context, line_no in occurrences:
            if value not in by_value:
                by_value[value] = []
            by_value[value].append((filepath, line_no, context))

    print(f"\n=== Magic Number Report ===")
    print(f"Found {len(by_value)} distinct magic numbers\n")

    for value in sorted(by_value.keys()):
        occurrences = by_value[value]
        print(f"\n{value} ({len(occurrences)} occurrences):")
        for filepath, line_no, context in occurrences[:5]:  # Show first 5
            rel_path = filepath.relative_to(source_dir)
            print(f"  {rel_path}:{line_no}: {context[:80]}")
        if len(occurrences) > 5:
            print(f"  ... and {len(occurrences) - 5} more")


def auto_add_constants(findings: dict, constants: ConstantSpace, config_path: Path, interactive: bool = True) -> Result[int]:
    """Add constants with optional interactive approval, using scope analysis"""

    # Collect proposed constants
    proposed = []
    for filepath, occurrences in findings.items():
        content_result = FileOp.read(filepath)
        if content_result.is_err():
            continue

        content = content_result.value

        for value, context, line_no in occurrences:
            # Only propose if in genome_to_param context
            if 'genome_to_param' not in context:
                continue

            # Use scope analysis to understand WHERE this constant is used
            position = content.find(context)
            scope_info = Scope.get_enclosing_scope(content, position) if position != -1 else None

            # Suggest name based on context AND scope
            suggested_name = constants.suggest_name(value, context)

            # Add scope info to suggestion if available
            if scope_info and scope_info.function:
                # If used in specific function, suggest more specific name
                func_name = scope_info.function
                if 'theta' in func_name or 'chemotaxis' in context.lower():
                    suggested_name = suggested_name.replace('_PARAM', '_THETA')
                elif 'alpha' in func_name:
                    suggested_name = suggested_name.replace('_PARAM', '_ALPHA')

            if '.' in value or 'f' in value.lower():
                const_type = 'float'
            else:
                const_type = 'int'

            if not constants.find_by_name(suggested_name):
                proposed.append((suggested_name, value, const_type, filepath, line_no, context))

    if not proposed:
        print("\nNo constants to add")
        return Ok(0)

    print(f"\n{'='*70}")
    print(f"Proposed Constants ({len(proposed)} total)")
    print(f"{'='*70}\n")

    approved = []
    for name, value, const_type, filepath, line_no, context in proposed:
        rel_path = filepath.relative_to(filepath.parent.parent.parent)
        print(f"\nConstant: {name}")
        print(f"  Type:  {const_type}")
        print(f"  Value: {value}")
        print(f"  From:  {rel_path}:{line_no}")
        print(f"  Context: {context[:80]}")

        if interactive:
            while True:
                response = input("  Add this constant? [y]es / [n]o / [q]uit: ").lower()
                if response == 'y':
                    approved.append((name, value, const_type, filepath, line_no))
                    print("    ✓ Approved")
                    break
                elif response == 'n':
                    print("    ✗ Skipped")
                    break
                elif response == 'q':
                    print("\nAborting")
                    return Ok(0)
                else:
                    print("    Invalid input. Use y/n/q")
        else:
            # Auto-approve mode
            approved.append((name, value, const_type, filepath, line_no))
            print("    [AUTO-APPROVED]")

    if not approved:
        print("\nNo constants approved")
        return Ok(0)

    # Register approved constants
    for name, value, const_type, filepath, line_no in approved:
        constants.register(name, value, const_type, (filepath, line_no))

    # Write to config
    result = constants.write_config(config_path)
    if result.is_err():
        return result

    print(f"\n{'='*70}")
    print(f"Added {len(approved)} constants to {config_path}")
    print(f"{'='*70}")

    return Ok(len(approved))


def main():
    from core import Paths
    source_dir = Paths.source_file()
    config_path = Paths.source_file("config", "config.cu")

    # Initialize constant space (load existing config)
    constants = ConstantSpace()

    print("=== Magic Number Finder ===\n")

    # Scan for magic numbers
    result = scan_all_files(source_dir, constants)
    if result.is_err():
        print(f"ERROR: {result.message}")
        for ctx in result.context:
            print(f"  {ctx}")
        sys.exit(1)

    findings = result.value
    report_findings(findings, source_dir)

    # Auto-add if --fix flag
    if '--fix' in sys.argv:
        interactive = '--yes' not in sys.argv  # --yes disables interactive mode
        if not interactive:
            print("\nWARNING: Auto-approve mode enabled (--yes)")

        auto_result = auto_add_constants(findings, constants, config_path, interactive)
        if auto_result.is_err():
            print(f"ERROR: {auto_result.message}")
            sys.exit(1)


if __name__ == "__main__":
    main()

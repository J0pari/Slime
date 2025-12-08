#!/usr/bin/env python3
"""
Function signature refactoring - add/remove/reorder parameters across codebase.

This tool modifies function signatures AND finds all call sites, but requires
the user to manually specify what argument values to pass at each site.

WHY: Different call sites need different argument expressions:
  - organism.cu calls pass organism->telemetry->*
  - Helper functions pass through their own parameters
  - Kernels might compute values locally

The tool BREAKS all call sites intentionally, showing compilation errors
that guide manual fixes. This is CORRECT behavior for INFINITE_FUTURE.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
from core import FileOp, Scope, FunctionDef, CallSite, Result, Ok, Err
import sys


def analyze_function_signature(source_dir: Path, func_name: str) -> Result[Dict]:
    """
    Find function definition and all call sites.
    Returns analysis without modifying anything.
    """
    files_result = FileOp.scan(source_dir, "*.cu")
    if files_result.is_err():
        return files_result

    cu_files = files_result.value
    cuh_result = FileOp.scan(source_dir, "*.cuh")
    if not cuh_result.is_err():
        cu_files.extend(cuh_result.value)

    # Find function definition
    func_def = None
    func_def_file = None

    for filepath in cu_files:
        content_result = FileOp.read(filepath)
        if content_result.is_err():
            continue

        found_func = Scope.find_function(content_result.value, func_name)
        if found_func:
            func_def = found_func
            func_def_file = filepath
            break

    if not func_def:
        return Err(f"Function {func_name} not found")

    # Find all call sites
    call_sites = []
    for filepath in cu_files:
        content_result = FileOp.read(filepath)
        if content_result.is_err():
            continue

        calls = Scope.find_calls(content_result.value, func_name)
        for call in calls:
            call_sites.append((filepath, call))

    return Ok({
        'function': func_def,
        'definition_file': func_def_file,
        'call_sites': call_sites,
        'total_calls': len(call_sites)
    })


def update_function_signature_only(source_dir: Path, func_name: str,
                                   new_params: List[str], position: int = -1) -> Result[Tuple[int, int]]:
    """
    Update ONLY the function signature, leaving all call sites broken.
    This forces compilation errors that show every place needing manual attention.

    Returns (files_modified, call_sites_broken)
    """
    analysis_result = analyze_function_signature(source_dir, func_name)
    if analysis_result.is_err():
        return analysis_result

    analysis = analysis_result.value
    func_def = analysis['function']
    func_def_file = analysis['definition_file']
    call_sites = analysis['call_sites']

    print(f"\n[SIGNATURE UPDATE]")
    print(f"  Function: {func_name}")
    print(f"  Location: {func_def_file.relative_to(source_dir)}")
    print(f"  Current params: {len(func_def.params)}")
    print(f"  Adding: {len(new_params)} parameters")
    print(f"  Will break: {len(call_sites)} call sites")
    print()

    # Update signature
    new_func = Scope.add_params(func_def, new_params, position)
    content_result = FileOp.read(func_def_file)
    if content_result.is_err():
        return content_result

    new_content = Scope.replace_signature(content_result.value, func_def, new_func)

    write_result = FileOp.write(func_def_file, new_content)
    if write_result.is_err():
        return write_result

    print(f"OK Updated signature in {func_def_file.relative_to(source_dir)}")
    print(f"X {len(call_sites)} call sites now have wrong argument count")
    print()
    print("Call sites that need manual fixing:")
    for filepath, call in call_sites:
        rel_path = filepath.relative_to(source_dir)
        print(f"  {rel_path} (has {len(call.args)} args, needs {len(new_func.params)})")

    return Ok((1, len(call_sites)))


def auto_wire_telemetry(source_dir: Path, func_name: str,
                        new_params: List[str], semantic_needs: List[str],
                        position: int = -1) -> Result[Tuple[int, int]]:
    """
    Automatically wire telemetry parameters by analyzing scope at each call site.

    semantic_needs: ['complexity', 'niche', 'learning', 'performance']

    For each call site:
    1. Find enclosing function
    2. Get available variables
    3. Match semantic needs to available vars:
       - If organism exists → organism->telemetry->*
       - If ctx_complexity exists → ctx_complexity
       - Otherwise → compilation error (manual fix needed)
    """
    analysis_result = analyze_function_signature(source_dir, func_name)
    if analysis_result.is_err():
        return analysis_result

    analysis = analysis_result.value
    func_def = analysis['function']
    func_def_file = analysis['definition_file']
    call_sites = analysis['call_sites']

    print(f"\n[AUTO-WIRE TELEMETRY]")
    print(f"  Function: {func_name}")
    print(f"  Semantic needs: {semantic_needs}")
    print(f"  Analyzing {len(call_sites)} call sites...")
    print()

    # Update signature
    new_func = Scope.add_params(func_def, new_params, position)
    content_result = FileOp.read(func_def_file)
    if content_result.is_err():
        return content_result

    new_content = Scope.replace_signature(content_result.value, func_def, new_func)
    write_result = FileOp.write(func_def_file, new_content)
    if write_result.is_err():
        return write_result

    files_modified = 1
    total_wired = 0
    total_failed = 0

    # Group call sites by file
    files_to_calls = {}
    for filepath, call in call_sites:
        if filepath not in files_to_calls:
            files_to_calls[filepath] = []
        files_to_calls[filepath].append(call)

    # Analyze and update each file
    for filepath, calls in files_to_calls.items():
        content_result = FileOp.read(filepath)
        if content_result.is_err():
            continue

        content = content_result.value
        modified_content = content

        # Analyze each call site
        for call in reversed(calls):
            # Find what variables are available at this call site
            available_vars = Scope.find_available_variables(content, call.start)

            # Match semantic needs to available variables
            mapping = Scope.match_semantic_variables(available_vars, semantic_needs)

            # Build argument list
            new_args = []
            all_found = True
            for need in semantic_needs:
                expr = mapping.get(need)
                if expr:
                    new_args.append(expr)
                else:
                    all_found = False
                    break

            if all_found:
                # Wire it up
                new_call = Scope.add_args(call, new_args, position)
                modified_content = Scope.replace_call(modified_content, call, new_call)
                total_wired += 1
            else:
                # Can't auto-wire - leave broken for manual fix
                rel_path = filepath.relative_to(source_dir)
                print(f"  X {rel_path}:{call.start} - Cannot auto-wire, needs manual fix")
                print(f"      Available: {available_vars}")
                print(f"      Needs: {semantic_needs}")
                total_failed += 1

        # Write if modified
        if modified_content != content:
            write_result = FileOp.write(filepath, modified_content)
            if not write_result.is_err():
                rel_path = filepath.relative_to(source_dir)
                wired_count = len([c for c in calls if True])  # Count wired calls
                print(f"  OK {rel_path}: auto-wired")
                files_modified += 1

    print()
    print(f"Summary:")
    print(f"  Auto-wired: {total_wired} call sites")
    print(f"  Failed: {total_failed} call sites (need manual fix)")
    print(f"  Files modified: {files_modified}")

    return Ok((files_modified, total_wired))


def update_signature_and_calls(source_dir: Path, func_name: str,
                               new_params: List[str], arg_map: Dict[str, List[str]],
                               position: int = -1) -> Result[Tuple[int, int]]:
    """
    Update function signature AND call sites using explicit argument mapping.

    arg_map: {filename: [arg_expressions]} mapping files to argument expressions
    Example: {
        'organism.cu': ['organism->telemetry->genome_complexity.hash_entropy', ...],
        'chemotaxis.cu': ['ctx_complexity', 'ctx_niche', ...]
    }

    This is the ONLY way to correctly update all call sites - explicit user specification.
    """
    analysis_result = analyze_function_signature(source_dir, func_name)
    if analysis_result.is_err():
        return analysis_result

    analysis = analysis_result.value
    func_def = analysis['function']
    func_def_file = analysis['definition_file']
    call_sites = analysis['call_sites']

    print(f"\n[FULL UPDATE]")
    print(f"  Function: {func_name}")
    print(f"  Updating {len(call_sites)} call sites with custom arguments")
    print()

    # Update signature
    new_func = Scope.add_params(func_def, new_params, position)
    content_result = FileOp.read(func_def_file)
    if content_result.is_err():
        return content_result

    new_content = Scope.replace_signature(content_result.value, func_def, new_func)

    write_result = FileOp.write(func_def_file, new_content)
    if write_result.is_err():
        return write_result

    files_modified = 1

    # Group call sites by file
    files_to_calls = {}
    for filepath, call in call_sites:
        if filepath not in files_to_calls:
            files_to_calls[filepath] = []
        files_to_calls[filepath].append(call)

    # Update each file's call sites
    for filepath, calls in files_to_calls.items():
        filename = filepath.name

        # Get argument expressions for this file
        args_for_file = arg_map.get(filename)
        if not args_for_file:
            print(f"X {filename}: No argument mapping provided, skipping")
            continue

        content_result = FileOp.read(filepath)
        if content_result.is_err():
            continue

        modified_content = content_result.value

        # Update calls in reverse order
        for call in reversed(calls):
            new_call = Scope.add_args(call, args_for_file, position)
            modified_content = Scope.replace_call(modified_content, call, new_call)

        write_result = FileOp.write(filepath, modified_content)
        if not write_result.is_err():
            rel_path = filepath.relative_to(source_dir)
            print(f"OK {rel_path}: {len(calls)} call sites updated")
            files_modified += 1

    return Ok((files_modified, len(call_sites)))


def rename_function(source_dir: Path, old_name: str, new_name: str) -> Result[int]:
    """
    Rename function definition and all call sites.
    Simple word-boundary replacement.
    """
    files_result = FileOp.scan(source_dir, "*.cu")
    if files_result.is_err():
        return files_result

    cu_files = files_result.value
    cuh_result = FileOp.scan(source_dir, "*.cuh")
    if not cuh_result.is_err():
        cu_files.extend(cuh_result.value)

    print(f"\n[RENAME]")
    print(f"  {old_name} → {new_name}")
    print()

    files_modified = 0
    import re

    for filepath in cu_files:
        content_result = FileOp.read(filepath)
        if content_result.is_err():
            continue

        new_content = re.sub(rf'\b{re.escape(old_name)}\b', new_name, content_result.value)

        if new_content != content_result.value:
            write_result = FileOp.write(filepath, new_content)
            if not write_result.is_err():
                rel_path = filepath.relative_to(source_dir)
                print(f"OK {rel_path}")
                files_modified += 1

    print()
    print(f"Renamed in {files_modified} files")
    return Ok(files_modified)


def batch_add_parameter(source_dir: Path, func_names: List[str],
                        new_param: str, position: int = -1) -> Result[Dict]:
    """
    Add same parameter to multiple functions in batch.
    Breaks all call sites intentionally.

    Returns summary of what was broken.
    """
    print(f"\n[BATCH PARAMETER ADD]")
    print(f"  Parameter: {new_param}")
    print(f"  Functions: {len(func_names)}")
    print()

    results = {
        'signatures_updated': 0,
        'call_sites_broken': 0,
        'functions_not_found': [],
        'broken_by_function': {}
    }

    for func_name in func_names:
        result = update_function_signature_only(source_dir, func_name, [new_param], position)

        if result.is_err():
            results['functions_not_found'].append(func_name)
            print(f"  X {func_name}: {result.message}")
        else:
            files_modified, call_sites = result.value
            results['signatures_updated'] += files_modified
            results['call_sites_broken'] += call_sites
            results['broken_by_function'][func_name] = call_sites
            print(f"  OK {func_name}: broke {call_sites} call sites")

    print()
    print("Summary:")
    print(f"  Signatures updated: {results['signatures_updated']}")
    print(f"  Total call sites broken: {results['call_sites_broken']}")
    print(f"  Functions not found: {len(results['functions_not_found'])}")

    return Ok(results)


def batch_fix_calls(source_dir: Path, func_names: List[str],
                    arg_expr: str, position: int = -1) -> Result[Dict]:
    """
    Fix all broken call sites for multiple functions by adding same argument expression.

    func_names: List of function names to fix
    arg_expr: Single argument expression to add to ALL call sites
    position: Where to insert (-1 for end)

    Example:
        batch_fix_calls(slime_dir,
                       ['kernel1', 'kernel2', 'kernel3'],
                       'organism->pool->entries[0].behavioral_dim')
    """
    print(f"\n[BATCH FIX CALLS]")
    print(f"  Argument: {arg_expr}")
    print(f"  Functions: {len(func_names)}")
    print()

    files_modified = set()
    total_fixed = 0

    for func_name in func_names:
        # Find all call sites for this function
        files_result = FileOp.scan(source_dir, "*.cu")
        if files_result.is_err():
            return files_result

        cu_files = files_result.value
        cuh_result = FileOp.scan(source_dir, "*.cuh")
        if not cuh_result.is_err():
            cu_files.extend(cuh_result.value)

        # Group call sites by file
        files_to_calls = {}
        for filepath in cu_files:
            content_result = FileOp.read(filepath)
            if content_result.is_err():
                continue

            calls = Scope.find_calls(content_result.value, func_name)
            if calls:
                files_to_calls[filepath] = calls

        # Fix each file
        for filepath, calls in files_to_calls.items():
            content_result = FileOp.read(filepath)
            if content_result.is_err():
                continue

            modified_content = content_result.value

            # Update calls in reverse order to preserve positions
            for call in reversed(calls):
                new_call = Scope.add_args(call, [arg_expr], position)
                modified_content = Scope.replace_call(modified_content, call, new_call)
                total_fixed += 1

            # Write modified file
            write_result = FileOp.write(filepath, modified_content)
            if not write_result.is_err():
                rel_path = filepath.relative_to(source_dir)
                print(f"  OK {rel_path}: fixed {len(calls)} calls to {func_name}")
                files_modified.add(filepath)

    print()
    print(f"Summary:")
    print(f"  Files modified: {len(files_modified)}")
    print(f"  Total calls fixed: {total_fixed}")

    return Ok({
        'files_modified': len(files_modified),
        'calls_fixed': total_fixed
    })


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCommands:")
        print("  analyze <func_name>                       Show function signature and all call sites")
        print("  break <func_name> <params>                Update signature, intentionally break calls")
        print("  batch <param> <func1,func2,...>           Add same param to multiple functions")
        print("  fix <arg> <func1,func2,...>               Fix broken calls by adding same argument")
        print("  autowire <func_name> <params> <needs>     Auto-wire telemetry by scope analysis")
        print("  rename <old_name> <new_name>              Rename function everywhere")
        print()
        print("Examples:")
        print("  # See where genome_to_param is called")
        print("  python refactor_functions.py analyze genome_to_param")
        print()
        print("  # Add params, break all calls (to find them via compilation)")
        print("  python refactor_functions.py break genome_to_param 'float ctx_complexity, float ctx_niche'")
        print()
        print("  # Add same param to multiple kernels in batch")
        print("  python refactor_functions.py batch 'int behavioral_dim' 'behavioral_gradient_kernel,chemotactic_navigation_kernel'")
        print()
        print("  # Fix all broken calls by adding same argument")
        print("  python refactor_functions.py fix 'organism->pool->entries[0].behavioral_dim' 'kernel1,kernel2,kernel3'")
        print()
        print("  # Auto-wire telemetry using scope analysis")
        print("  python refactor_functions.py autowire get_theta 'float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance' 'complexity,niche,learning,performance'")
        print()
        print("  # Rename function")
        print("  python refactor_functions.py rename old_kernel new_kernel")
        sys.exit(1)

    from core import Paths
    source_dir = Paths.source_file()

    command = sys.argv[1]

    if command == "analyze":
        if len(sys.argv) < 3:
            print("Usage: refactor_functions.py analyze <func_name>")
            sys.exit(1)

        func_name = sys.argv[2]
        result = analyze_function_signature(source_dir, func_name)

        if result.is_err():
            print(f"ERROR: {result.message}")
            sys.exit(1)

        analysis = result.value
        print(f"\n=== {func_name} ===")
        print(f"Definition: {analysis['definition_file'].relative_to(source_dir)}")
        print(f"Parameters: {len(analysis['function'].params)}")
        for i, param in enumerate(analysis['function'].params):
            print(f"  [{i}] {param}")
        print()
        print(f"Call sites: {analysis['total_calls']}")
        for filepath, call in analysis['call_sites']:
            rel_path = filepath.relative_to(source_dir)
            print(f"  {rel_path}:{call.start} ({len(call.args)} args)")

    elif command == "break":
        if len(sys.argv) < 4:
            print("Usage: refactor_functions.py break <func_name> <params>")
            print("  params: comma-separated parameter declarations")
            sys.exit(1)

        func_name = sys.argv[2]
        params_str = sys.argv[3]
        new_params = [p.strip() for p in params_str.split(',')]

        position = -1
        if '--position' in sys.argv:
            pos_idx = sys.argv.index('--position')
            position = int(sys.argv[pos_idx + 1])

        result = update_function_signature_only(source_dir, func_name, new_params, position)

        if result.is_err():
            print(f"ERROR: {result.message}")
            sys.exit(1)

    elif command == "batch":
        if len(sys.argv) < 4:
            print("Usage: refactor_functions.py batch <param> <func1,func2,...>")
            print("  param: single parameter declaration")
            print("  funcs: comma-separated function names")
            sys.exit(1)

        param = sys.argv[2]
        funcs_str = sys.argv[3]
        func_names = [f.strip() for f in funcs_str.split(',')]

        position = -1
        if '--position' in sys.argv:
            pos_idx = sys.argv.index('--position')
            position = int(sys.argv[pos_idx + 1])

        result = batch_add_parameter(source_dir, func_names, param, position)

        if result.is_err():
            print(f"ERROR: {result.message}")
            sys.exit(1)

    elif command == "fix":
        if len(sys.argv) < 4:
            print("Usage: refactor_functions.py fix <arg> <func1,func2,...>")
            print("  arg: argument expression to add to all calls")
            print("  funcs: comma-separated function names")
            sys.exit(1)

        arg_expr = sys.argv[2]
        funcs_str = sys.argv[3]
        func_names = [f.strip() for f in funcs_str.split(',')]

        position = -1
        if '--position' in sys.argv:
            pos_idx = sys.argv.index('--position')
            position = int(sys.argv[pos_idx + 1])

        result = batch_fix_calls(source_dir, func_names, arg_expr, position)

        if result.is_err():
            print(f"ERROR: {result.message}")
            sys.exit(1)

    elif command == "autowire":
        if len(sys.argv) < 5:
            print("Usage: refactor_functions.py autowire <func_name> <params> <needs>")
            print("  params: comma-separated parameter declarations")
            print("  needs: comma-separated semantic needs (complexity,niche,learning,performance)")
            sys.exit(1)

        func_name = sys.argv[2]
        params_str = sys.argv[3]
        needs_str = sys.argv[4]

        new_params = [p.strip() for p in params_str.split(',')]
        semantic_needs = [n.strip() for n in needs_str.split(',')]

        position = -1
        if '--position' in sys.argv:
            pos_idx = sys.argv.index('--position')
            position = int(sys.argv[pos_idx + 1])

        result = auto_wire_telemetry(source_dir, func_name, new_params, semantic_needs, position)

        if result.is_err():
            print(f"ERROR: {result.message}")
            sys.exit(1)

    elif command == "rename":
        if len(sys.argv) < 4:
            print("Usage: refactor_functions.py rename <old_name> <new_name>")
            sys.exit(1)

        old_name = sys.argv[2]
        new_name = sys.argv[3]

        result = rename_function(source_dir, old_name, new_name)

        if result.is_err():
            print(f"ERROR: {result.message}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print("Use: analyze, break, autowire, or rename")
        sys.exit(1)


if __name__ == "__main__":
    main()

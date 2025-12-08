#!/usr/bin/env python3
"""Analyze memory allocations in organism.cu to find out-of-memory cause."""

from pathlib import Path
from core import Paths, FileOp, Ok, Err, bind, CUDAParser, Result, GenomeSimulator

def get_type_size(type_name: str, constants: dict) -> Result[int]:
    """Get size of a type using libclang AST parsing."""
    # Primitive types
    type_sizes = {
        'float': 4,
        'half': 2,
        '__half': 2,
        'uint32_t': 4,
        'int': 4,
        'uint8_t': 1,
        'bool': 1,
        'double': 8,
        'uint64_t': 8,
        'int64_t': 8,
        'uint16_t': 2,
        'int16_t': 2,
        'int8_t': 1,
        'char': 1,
    }

    if type_name in type_sizes:
        return Ok(type_sizes[type_name])

    # Use libclang to get struct size
    return CUDAParser.get_struct_size(type_name)

def evaluate_size(size_expr: str, constants: dict, local_vars: dict = None) -> Result[int]:
    """Evaluate a size expression using known constants and local variables."""
    expr = size_expr.strip()
    if local_vars is None:
        local_vars = {}

    # Merge constants and local vars for eval namespace
    eval_namespace = {**constants, **local_vars}

    # Handle sizeof(Type) * Count
    import re
    match = re.match(r'sizeof\s*\(\s*(\w+(?:\s*<[^>]+>)?)\s*\)\s*\*\s*(.+)', expr)
    if match:
        type_name = match.group(1).split('<')[0].strip()
        count_expr = match.group(2)

        size_result = get_type_size(type_name, constants)
        if size_result.is_err():
            return Err(f"Failed to get size of type {type_name}: {size_result.message}")

        base_size = size_result.value

        # Evaluate count with namespace
        try:
            count = eval(count_expr, {"__builtins__": {}}, eval_namespace)
            return Ok(base_size * count)
        except Exception as e:
            return Err(f"Failed to evaluate count expression '{count_expr}': {e}")

    # Handle plain sizeof(Type)
    match = re.match(r'sizeof\s*\(\s*(\w+(?:\s*<[^>]+>)?)\s*\)', expr)
    if match:
        type_name = match.group(1).split('<')[0].strip()
        return get_type_size(type_name, constants)

    # Try direct evaluation with namespace
    try:
        result = eval(expr, {"__builtins__": {}}, eval_namespace)
        return Ok(result)
    except Exception as e:
        return Err(f"Failed to evaluate expression '{expr}': {e}")

def main():
    repo_root = Paths.repo_root
    organism_cu = repo_root / "slime" / "core" / "organism.cu"

    # Extract constants from config.cu
    constant_files = [
        repo_root / "slime" / "config" / "config.cu",
    ]

    all_constants = []
    for const_file in constant_files:
        if const_file.exists():
            result = CUDAParser.extract_constants(const_file)
            if result.is_ok():
                all_constants.extend(result.value)

    constants_result = Ok(all_constants) if all_constants else Err("No constants found")

    # Evaluate constants in dependency order
    constants = {}
    if constants_result.is_ok():
        constants_raw = {c.name: c.value for c in constants_result.value}

        # Iteratively evaluate constants
        max_iterations = 50
        for iteration in range(max_iterations):
            progress = False
            for name, value in constants_raw.items():
                if name not in constants:
                    try:
                        # Handle float literals with 'f' suffix
                        eval_value = value.rstrip('f') if value.endswith('f') else value
                        # Try to evaluate with current context
                        evaluated = eval(eval_value, {"__builtins__": {}}, constants)
                        constants[name] = evaluated
                        progress = True
                    except:
                        pass  # Skip for now, might need more dependencies

            if not progress:
                break  # No more progress possible

        print(f"Evaluated {len(constants)} / {len(constants_raw)} constants from config.cu", flush=True)

    # Find all struct assignments to discover where arch/pool values are set
    print("Finding struct field assignments in organism.cu...", flush=True)
    assignments_result = CUDAParser.find_struct_assignments(organism_cu)
    if assignments_result.is_ok():
        assignments = assignments_result.value

        # Group by struct variable name
        arch_assignments = [a for a in assignments if a.struct_var == 'arch']
        pool_entry_assignments = [a for a in assignments if 'entries[' in a.struct_var and 'pool' in a.struct_var]

        print(f"Found {len(arch_assignments)} arch assignments, {len(pool_entry_assignments)} pool entry assignments", flush=True)

        # Generate genome and derive architecture exactly as CUDA does
        if arch_assignments:
            print("Simulating genome generation and architecture derivation...", flush=True)

            # Verify we have required constants
            required = ['GENOME_SIZE', 'GENOME_RANGE_SCALE', 'GENOME_VALUE_MIN']
            missing = [c for c in required if c not in constants]
            if missing:
                print(f"ERROR: Missing required constants: {missing}", flush=True)
                print("Cannot simulate genome without these constants", flush=True)
            else:
                # Generate genome for pool entry idx=0 exactly as init_pool_kernel does
                genome = GenomeSimulator.generate_genome(
                    idx=0,
                    genome_size=constants['GENOME_SIZE'],
                    genome_range_scale=constants['GENOME_RANGE_SCALE'],
                    genome_value_min=constants['GENOME_VALUE_MIN']
                )

                # Compute genome hash
                genome_hash = GenomeSimulator.gpu_sha256_stub(genome)
                print(f"Generated genome for idx=0, hash={genome_hash:016x}", flush=True)

                # Derive architecture exactly as derive_architecture does
                arch_dict = GenomeSimulator.derive_architecture(genome_hash, genome, constants)

                print(f"Derived architecture: num_heads={arch_dict['num_heads']}, " +
                      f"channels={arch_dict['channels']}, head_dim={arch_dict['head_dim']}, " +
                      f"grid_size={arch_dict['grid_size']}, hidden_dim={arch_dict['hidden_dim']}", flush=True)

                # Create arch object for use in local variable evaluation
                class Arch:
                    pass
                arch = Arch()
                for k, v in arch_dict.items():
                    setattr(arch, k, v)

                constants['arch'] = arch
    else:
        print(f"Failed to find struct assignments: {assignments_result.message}", flush=True)

    # Extract local variables from init_organism_kernel
    print("Extracting local variables from init_organism_kernel...", flush=True)
    local_vars_result = CUDAParser.extract_local_variables(organism_cu, 'init_organism_kernel')
    local_vars_raw = local_vars_result.value if local_vars_result.is_ok() else {}

    # Evaluate local variables with constants in dependency order
    local_vars = {}
    max_iterations = 20
    for iteration in range(max_iterations):
        progress = False
        for var_name, var_expr in local_vars_raw.items():
            if var_name not in local_vars:
                try:
                    # Handle sizeof() expressions by evaluating them using get_type_size
                    import re
                    sizeof_pattern = r'sizeof\s*\(\s*(\w+)\s*\)'
                    def replace_sizeof(match):
                        type_name = match.group(1)
                        size_result = get_type_size(type_name, constants)
                        if size_result.is_ok():
                            return str(size_result.value)
                        else:
                            raise ValueError(f"Cannot get size of {type_name}")

                    eval_expr = re.sub(sizeof_pattern, replace_sizeof, var_expr)

                    # Try to evaluate with current context
                    local_vars[var_name] = eval(eval_expr, {"__builtins__": {}}, {**constants, **local_vars})
                    progress = True
                except Exception as e:
                    pass  # Skip for now, might need more dependencies

        if not progress:
            break  # No more progress possible

    # Show which variables failed to evaluate
    failed_vars = [k for k in local_vars_raw.keys() if k not in local_vars]
    if failed_vars:
        print(f"Failed to evaluate {len(failed_vars)} variables: {failed_vars}", flush=True)

    print(f"Extracted {len(local_vars)} / {len(local_vars_raw)} local variables", flush=True)

    # Extract allocations using tree-sitter
    print("Parsing organism.cu for cudaMalloc calls...", flush=True)
    mallocs_result = CUDAParser.find_mallocs(organism_cu)
    if mallocs_result.is_err():
        print(f"Error: {mallocs_result.message}")
        return 1

    mallocs = mallocs_result.value
    print(f"Found {len(mallocs)} cudaMalloc calls\n", flush=True)

    # Calculate sizes
    total_bytes = 0
    print(f"{'Line':<6} {'Variable':<40} {'Size Expression':<50} {'Bytes':<15} {'MB':<10}")
    print("=" * 130)

    for malloc in mallocs:
        size_result = evaluate_size(malloc.size_expr, constants, local_vars)

        if size_result.is_ok():
            size_bytes = size_result.value
            size_mb = size_bytes / (1024 * 1024)
            size_str = f"{size_bytes:,}"
            total_bytes += size_bytes
        else:
            size_str = f"ERROR: {size_result.message}"
            size_mb = 0

        print(f"{malloc.location[1]:<6} {malloc.var_name:<40} {malloc.size_expr:<50} {size_str:<15} {size_mb:<10.2f}")

    total_mb = total_bytes / (1024 * 1024)
    print("=" * 130)
    print(f"{'TOTAL':<97} {total_bytes:<15,} {total_mb:<10.2f}")
    print(f"\nGPU reported: 6.4 GB available")
    print(f"Allocated: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")

    if total_mb / 1024 > 6.4:
        print(f"\nMEMORY OVERFLOW: Trying to allocate {total_mb/1024:.2f} GB on {6.4} GB GPU!")

    # Find exactly where it fails by showing cumulative allocation
    print("\n=== CUMULATIVE ALLOCATION (in order) ===")
    cumulative = 0
    print(f"{'Line':<6} {'Variable':<40} {'This Alloc MB':<15} {'Cumulative MB':<15} {'Cumulative GB':<15}")
    print("=" * 100)

    for malloc in mallocs:
        size_result = evaluate_size(malloc.size_expr, constants, local_vars)
        if size_result.is_ok():
            size_bytes = size_result.value
            size_mb = size_bytes / (1024 * 1024)
            cumulative += size_bytes
            cumulative_mb = cumulative / (1024 * 1024)
            cumulative_gb = cumulative / (1024 * 1024 * 1024)

            status = "FAIL HERE" if cumulative_gb > 6.4 and cumulative - size_bytes <= 6.4 * 1024 * 1024 * 1024 else ""
            print(f"{malloc.location[1]:<6} {malloc.var_name:<40} {size_mb:<15.2f} {cumulative_mb:<15.2f} {cumulative_gb:<15.2f} {status}")

    return 0

if __name__ == '__main__':
    exit(main())

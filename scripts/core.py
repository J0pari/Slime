#!/usr/bin/env python3
"""
Core transformation primitives for CUDA source refactoring.

Five fundamental operations that all other scripts compose:
1. FileOp - File I/O with error context
2. Pattern - Regex matching and replacement
3. Scope - Code structure analysis (functions, calls, signatures)
4. ConstantSpace - Semantic constant management
5. Pipeline - Transformation composition with dependencies

All 23 existing scripts are compositions of these 5 primitives.
"""

from pathlib import Path
from typing import Callable, List, Dict, Optional, Tuple, TypeVar, Generic, Set
from dataclasses import dataclass
from enum import Enum
import re

# ============================================================================
# Result type for error propagation without exceptions
# ============================================================================

T = TypeVar('T')
U = TypeVar('U')

@dataclass
class Success(Generic[T]):
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

@dataclass
class Error:
    message: str
    context: List[str]

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

Result = Success[T] | Error

def Ok(value: T) -> Result[T]:
    return Success(value)

def Err(message: str, context: List[str] = None) -> Error:
    return Error(message, context or [])


# ============================================================================
# 1. FileOp - File I/O operations with error context
# ============================================================================

class BinaryData:
    """Binary file loading with typed numpy conversion"""

    @staticmethod
    def load_typed(filepath: Path, dtype, shape=None):
        """Load binary file as numpy array"""
        result = FileOp.read_bytes(filepath)
        if result.is_err():
            raise IOError(f"Failed to read {filepath}: {result.message}")

        import numpy as np
        data = np.frombuffer(result.value, dtype=dtype)
        return data.reshape(shape) if shape else data

    @staticmethod
    def softmax(logits):
        """Softmax with numerical stability"""
        import numpy as np
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    @staticmethod
    def cross_entropy_loss(logits, label):
        """Cross-entropy loss"""
        probs = BinaryData.softmax(logits)
        import numpy as np
        return -np.log(probs[label] + 1e-10)

    @staticmethod
    def load_mnist(filepath: Path):
        """Load MNIST image (28x28 uint8)"""
        import numpy as np
        return BinaryData.load_typed(filepath, np.uint8, (28, 28))

    @staticmethod
    def load_ca_state(filepath: Path, grid_size=256, channels=16):
        """Load CA state"""
        import numpy as np
        return BinaryData.load_typed(filepath, np.float32, (grid_size, grid_size, channels))

    @staticmethod
    def load_logits(filepath: Path):
        """Load logits"""
        import numpy as np
        return BinaryData.load_typed(filepath, np.float32)

    @staticmethod
    def load_label(filepath: Path) -> int:
        """Load int32 label"""
        result = FileOp.read_bytes(filepath)
        if result.is_err():
            raise IOError(f"Failed to read {filepath}: {result.message}")
        import struct
        return struct.unpack('i', result.value[:4])[0]

    @staticmethod
    def load_field(filepath: Path, grid_size=256):
        """Load 2D field"""
        import numpy as np
        return BinaryData.load_typed(filepath, np.float32, (grid_size, grid_size))


class BuildArtifacts:
    """PTX/build artifact scanning"""

    @staticmethod
    def scan_for_kernels(ptx_dir: Path, kernel_names: List[str]) -> Dict[str, Path]:
        """Scan for kernel launch sites"""
        found = {}
        if not ptx_dir.exists():
            return found

        import os
        for root, _, files in os.walk(ptx_dir):
            for f in files:
                if f.endswith(('.ii', '.ptx', '.cudafe1.cpp')):
                    path = Path(root) / f
                    result = FileOp.read(path)
                    if result.is_err():
                        continue

                    content = result.value
                    for kernel in kernel_names:
                        # Check for launch site or PTX entry
                        mangled = f"_Z{len(kernel)}{kernel}"
                        if (f"{kernel}<<<" in content or
                            f".entry {kernel}" in content or
                            f".entry {mangled}" in content):
                            found[kernel] = path
                            break

        return found


class Shell:
    """Subprocess execution"""

    @staticmethod
    def run(cmd: List[str], cwd: Path = None, description: str = None) -> Result[int]:
        """Run command, return exit code"""
        if description:
            print(f"[RUN] {description}")
        print("$", " ".join(cmd))

        import subprocess
        try:
            exit_code = subprocess.call(cmd, cwd=str(cwd) if cwd else None)
            if exit_code != 0:
                return Err(f"Command exited with code {exit_code}")
            return Ok(exit_code)
        except Exception as e:
            return Err(f"Command failed: {e}")


class Paths:
    """Path resolution"""

    repo_root = Path(__file__).parent.parent

    @staticmethod
    def build_artifact(name: str) -> Optional[Path]:
        """Get build artifact path if exists"""
        path = Paths.repo_root / 'build' / name
        return path if path.exists() else None

    @staticmethod
    def ptx_dir() -> Path:
        """Get PTX dir (build/logs/ptx or logs/ptx)"""
        candidate = Paths.repo_root / 'build' / 'logs' / 'ptx'
        return candidate if candidate.is_dir() else Paths.repo_root / 'logs' / 'ptx'

    @staticmethod
    def source_file(*parts: str) -> Path:
        """Get slime/ source path"""
        return Paths.repo_root / 'slime' / Path(*parts)

    @staticmethod
    def script(*parts: str) -> Path:
        """Get scripts/ path"""
        return Paths.repo_root / 'scripts' / Path(*parts)


# ============================================================================
# FileOp - Original file operations
# ============================================================================

class FileOp:
    """
    All file operations return Result[T] for composable error handling.
    Eliminates try/catch boilerplate across all scripts.
    """

    @staticmethod
    def read(path: Path, encoding: str = 'utf-8') -> Result[str]:
        """Read file, return content or error with context"""
        try:
            with open(path, 'r', encoding=encoding) as f:
                return Ok(f.read())
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    return Ok(f.read())
            except Exception as e:
                return Err(f"Failed to read {path}", [str(e)])
        except Exception as e:
            return Err(f"Failed to read {path}", [str(e)])

    @staticmethod
    def read_bytes(path: Path) -> Result[bytes]:
        """Read file as bytes, return content or error with context"""
        try:
            with open(path, 'rb') as f:
                return Ok(f.read())
        except Exception as e:
            return Err(f"Failed to read {path}", [str(e)])

    @staticmethod
    def write(path: Path, content: str, encoding: str = 'utf-8') -> Result[Path]:
        """Write file, return path or error with context"""
        try:
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            return Ok(path)
        except Exception as e:
            return Err(f"Failed to write {path}", [str(e)])

    @staticmethod
    def transform(path: Path, f: Callable[[str], str]) -> Result[Path]:
        """Read -> transform -> write. Core composition primitive."""
        result = FileOp.read(path)
        if isinstance(result, Error):
            return result

        try:
            transformed = f(result.value)
            return FileOp.write(path, transformed)
        except Exception as e:
            return Err(f"Transform failed on {path}", [str(e)])

    @staticmethod
    def scan(directory: Path, pattern: str) -> Result[List[Path]]:
        """Find all files matching glob pattern"""
        try:
            files = list(directory.rglob(pattern))
            return Ok(files)
        except Exception as e:
            return Err(f"Failed to scan {directory}", [str(e)])

    @staticmethod
    def bind(result: Result[T], f: Callable[[T], Result[U]]) -> Result[U]:
        """
        Monadic bind for chaining operations that can fail.
        If result is Error, propagate it. Otherwise apply f.
        """
        if isinstance(result, Error):
            return result
        return f(result.value)


# ============================================================================
# 2. Pattern - Regex operations as composable transformations
# ============================================================================

class Pattern:
    """
    Regex matching and replacement as composable functions.
    Replaces PatternMatcher and all replace_* scripts.
    """

    @staticmethod
    def match(regex: str, content: str, flags: int = 0) -> List[re.Match]:
        """Find all matches of pattern in content"""
        return list(re.finditer(regex, content, flags))

    @staticmethod
    def replace(regex: str, replacement: str, content: str, flags: int = 0) -> str:
        """Replace all occurrences of pattern"""
        return re.sub(regex, replacement, content, flags)

    @staticmethod
    def replace_all(replacements: List[Tuple[str, str]], content: str) -> str:
        """
        Chain multiple replacements: content -> r1 -> r2 -> ... -> rn
        Each replacement sees the output of the previous one.
        """
        result = content
        for regex, replacement in replacements:
            result = Pattern.replace(regex, replacement, result)
        return result

    @staticmethod
    def extract(regex: str, content: str, group: int = 0) -> List[str]:
        """Extract all matching groups"""
        matches = Pattern.match(regex, content)
        return [m.group(group) for m in matches]

    @staticmethod
    def test(regex: str, content: str) -> bool:
        """Check if pattern matches"""
        return re.search(regex, content) is not None

    # ========================================================================
    # PATTERN LIBRARY - Common code transformations
    # ========================================================================

    @staticmethod
    def strip_comments(content: str) -> str:
        """
        Remove C/C++ comments from code.
        Needed for: Code cleanup without writing custom regex
        """
        # Remove // line comments
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        # Remove /* block comments */
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        return content

    @staticmethod
    def normalize_float_literals(content: str) -> str:
        """
        Fix invalid float literals: 10f -> 10.0f
        Needed for: Fixing compilation errors without manual intervention
        """
        # Pattern: integer followed immediately by 'f' (invalid C++)
        # Must have decimal point: 10.0f not 10f
        def replace_func(match):
            num = match.group(1)
            return f"{num}.0f"

        return re.sub(r'\b(\d+)f\b', replace_func, content)

    @staticmethod
    def remove_extra_blank_lines(content: str) -> str:
        """
        Collapse multiple blank lines to single blank line.
        Needed for: Code cleanup after comment removal
        """
        lines = content.split('\n')
        result = []
        prev_blank = False

        for line in lines:
            is_blank = len(line.strip()) == 0
            if not (is_blank and prev_blank):
                result.append(line)
            prev_blank = is_blank

        return '\n'.join(result)

    @staticmethod
    def sort_includes(content: str) -> str:
        """
        Sort #include directives alphabetically.
        Needed for: Consistent include ordering
        """
        lines = content.split('\n')
        includes = []
        non_includes = []
        in_include_block = False

        for line in lines:
            if line.strip().startswith('#include'):
                includes.append(line)
                in_include_block = True
            else:
                if in_include_block and includes:
                    # End of include block - sort and add
                    non_includes.extend(sorted(includes))
                    includes = []
                    in_include_block = False
                non_includes.append(line)

        # Handle includes at end of file
        if includes:
            non_includes.extend(sorted(includes))

        return '\n'.join(non_includes)

    @staticmethod
    def cleanup_code(content: str) -> str:
        """
        Apply all cleanup transformations.
        Needed for: General code cleanup in one call
        """
        content = Pattern.strip_comments(content)
        content = Pattern.normalize_float_literals(content)
        content = Pattern.remove_extra_blank_lines(content)
        return content


# ============================================================================
# 3. Scope - Code structure analysis (AST-aware transformations)
# ============================================================================

@dataclass
class FunctionDef:
    """Parsed function/kernel signature"""
    name: str
    prefix: str      # __global__, __device__, etc
    params: List[str]
    suffix: str      # const, override, etc
    start: int       # position in source
    end: int

    def signature_text(self) -> str:
        """Reconstruct full signature"""
        param_str = ', '.join(self.params)
        return f"{self.prefix} {self.name}({param_str}){self.suffix}"

@dataclass
class CallSite:
    """Function/kernel call location"""
    name: str
    args: List[str]
    start: int
    end: int
    is_kernel: bool  # True if <<<grid, block>>> launch


class Scope:
    """
    Understanding code structure - functions, kernels, calls.
    Replaces FunctionSignature, KernelCallFinder, ConstexprExtractor.
    """

    FUNCTION_PATTERN = re.compile(
        r'((?:__global__|__device__|__host__|extern\s+"C"|template<[^>]+>)\s+.*?)\s+'
        r'(\w+)\s*'
        r'\(([^)]*(?:\([^)]*\)[^)]*)*)\)'
        r'(\s*(?:const)?\s*(?:override)?)',
        re.DOTALL
    )

    KERNEL_LAUNCH = re.compile(
        r'(\w+)\s*<<<\s*([^>]+)\s*>>>\s*\(([^;]*(?:\([^)]*\)[^;]*)*)\);',
        re.DOTALL
    )

    FUNCTION_CALL = re.compile(
        r'(?<![a-zA-Z0-9_])(\w+)\s*\(([^;]*(?:\([^)]*\)[^;]*)*)\);',
        re.DOTALL
    )

    @staticmethod
    def find_functions(content: str) -> List[FunctionDef]:
        """Extract all function definitions"""
        functions = []
        for match in Scope.FUNCTION_PATTERN.finditer(content):
            params = Scope._parse_params(match.group(3))
            func = FunctionDef(
                name=match.group(2),
                prefix=match.group(1).strip(),
                params=params,
                suffix=match.group(4).strip() if match.lastindex >= 4 else '',
                start=match.start(),
                end=match.end()
            )
            functions.append(func)
        return functions

    @staticmethod
    def find_function(content: str, name: str) -> Optional[FunctionDef]:
        """Find specific function by name"""
        funcs = Scope.find_functions(content)
        for f in funcs:
            if f.name == name:
                return f
        return None

    @staticmethod
    def find_calls(content: str, func_name: str) -> List[CallSite]:
        """Find all call sites of a function"""
        calls = []

        # Check kernel launches
        for match in Scope.KERNEL_LAUNCH.finditer(content):
            if match.group(1) == func_name:
                args = Scope._parse_params(match.group(3))
                calls.append(CallSite(
                    name=match.group(1),
                    args=args,
                    start=match.start(),
                    end=match.end(),
                    is_kernel=True
                ))

        # Check regular calls
        for match in Scope.FUNCTION_CALL.finditer(content):
            if match.group(1) == func_name:
                args = Scope._parse_params(match.group(2))
                calls.append(CallSite(
                    name=match.group(1),
                    args=args,
                    start=match.start(),
                    end=match.end(),
                    is_kernel=False
                ))

        return calls

    @staticmethod
    def add_params(func: FunctionDef, new_params: List[str], position: int = -1) -> FunctionDef:
        """Add parameters to function signature"""
        if position == -1:
            position = len(func.params)

        new_param_list = func.params[:position] + new_params + func.params[position:]

        return FunctionDef(
            name=func.name,
            prefix=func.prefix,
            params=new_param_list,
            suffix=func.suffix,
            start=func.start,
            end=func.end
        )

    @staticmethod
    def add_args(call: CallSite, new_args: List[str], position: int = -1) -> CallSite:
        """Add arguments to call site"""
        if position == -1:
            position = len(call.args)

        new_arg_list = call.args[:position] + new_args + call.args[position:]

        return CallSite(
            name=call.name,
            args=new_arg_list,
            start=call.start,
            end=call.end,
            is_kernel=call.is_kernel
        )

    @staticmethod
    def replace_signature(content: str, old_func: FunctionDef, new_func: FunctionDef) -> str:
        """Replace function signature in source"""
        new_sig = new_func.signature_text()
        return content[:old_func.start] + new_sig + content[old_func.end:]

    @staticmethod
    def replace_call(content: str, old_call: CallSite, new_call: CallSite) -> str:
        """Replace call site in source"""
        arg_str = ', '.join(new_call.args)
        if new_call.is_kernel:
            # Reconstruct kernel launch (need to extract grid/block from original)
            original_text = content[old_call.start:old_call.end]
            grid_block = re.search(r'<<<([^>]+)>>>', original_text).group(1)
            new_text = f"{new_call.name}<<<{grid_block}>>>({arg_str});"
        else:
            new_text = f"{new_call.name}({arg_str});"

        return content[:old_call.start] + new_text + content[old_call.end:]

    @staticmethod
    def _parse_params(params_str: str) -> List[str]:
        """Parse parameter/argument list respecting nested parens and templates"""
        if not params_str.strip():
            return []

        params = []
        current = []
        depth = 0
        angle_depth = 0

        for char in params_str:
            if char == '<':
                angle_depth += 1
            elif char == '>':
                angle_depth -= 1
            elif char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0 and angle_depth == 0:
                params.append(''.join(current).strip())
                current = []
                continue
            current.append(char)

        if current:
            params.append(''.join(current).strip())

        return params

    # ========================================================================
    # CONTEXT-AWARE SCOPE - Know where you are in the code
    # ========================================================================

    @staticmethod
    def get_function_at(content: str, position: int) -> Optional[FunctionDef]:
        """
        Find which function contains this position.
        Needed for: "Is this cudaMalloc inside a device function?"
        """
        functions = Scope.find_functions(content)

        for func in functions:
            # Find function body bounds
            body_start = content.find('{', func.end)
            if body_start == -1:
                continue

            # Find matching closing brace
            brace_count = 1
            body_end = body_start
            for i in range(body_start + 1, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        body_end = i
                        break

            # Check if position is within this function
            if func.start <= position <= body_end:
                return func

        return None

    @staticmethod
    def is_device_code(content: str, position: int) -> bool:
        """
        Check if position is inside __device__ or __global__ function.
        Needed for: "Is this malloc legal or device-side (illegal)?"
        """
        func = Scope.get_function_at(content, position)
        if not func:
            return False

        prefix_lower = func.prefix.lower()
        return '__device__' in prefix_lower or '__global__' in prefix_lower

    @dataclass
    class ScopeInfo:
        """Information about code location context"""
        function: Optional[FunctionDef]
        is_device: bool
        is_global: bool
        is_host: bool
        available_params: List[str]  # Parameters visible at this scope

    @staticmethod
    def get_enclosing_scope(content: str, position: int) -> 'Scope.ScopeInfo':
        """
        Get full context about where this position is.
        Returns what function it's in and what kind of code.
        """
        func = Scope.get_function_at(content, position)

        if not func:
            return Scope.ScopeInfo(
                function=None,
                is_device=False,
                is_global=False,
                is_host=True,
                available_params=[]
            )

        prefix_lower = func.prefix.lower()
        return Scope.ScopeInfo(
            function=func,
            is_device='__device__' in prefix_lower,
            is_global='__global__' in prefix_lower,
            is_host='__host__' in prefix_lower or '__device__' not in prefix_lower,
            available_params=func.params
        )

    @staticmethod
    def find_available_variables(content: str, position: int) -> List[str]:
        """
        Find all variables/parameters available at this position.
        Returns parameter names from enclosing function.
        """
        scope_info = Scope.get_enclosing_scope(content, position)
        if not scope_info.function:
            return []

        # Extract variable names from parameter declarations
        # "float ctx_metabolic" -> "ctx_metabolic"
        # "Organism* organism" -> "organism"
        # "const float* genome" -> "genome"
        var_names = []
        for param in scope_info.available_params:
            parts = param.strip().split()
            if parts:
                # Last non-operator token is the variable name
                name = parts[-1].strip('*&')
                var_names.append(name)

        return var_names

    @staticmethod
    def match_semantic_variables(available_vars: List[str], semantic_needs: List[str]) -> Dict[str, Optional[str]]:
        """
        Match available variables to semantic needs.

        semantic_needs: ['complexity', 'niche', 'learning', 'performance']
        available_vars: ['organism', 'ctx_metabolic', 'genome', ...]

        Returns mapping: {
            'complexity': 'organism',  # Found organism, can access organism->telemetry->...
            'niche': 'organism',
            'learning': None,  # Not available
            'performance': None
        }

        Heuristics:
        - If 'organism' exists, use organism->telemetry->genome_complexity.hash_entropy etc
        - If 'ctx_complexity' exists, use it directly
        - Otherwise return None (needs manual specification)
        """
        mapping = {}

        # Check if we have direct context variables
        for need in semantic_needs:
            ctx_var = f"ctx_{need}"
            if ctx_var in available_vars:
                mapping[need] = ctx_var
            elif 'organism' in available_vars:
                # Use telemetry path
                telemetry_map = {
                    'complexity': 'organism->telemetry->genome_complexity.hash_entropy',
                    'niche': 'organism->telemetry->archive_topology.novelty_gradient',
                    'learning': 'organism->telemetry->diresa_evolution.behavioral_drift_rate',
                    'performance': 'organism->telemetry->mnist_performance.accuracy'
                }
                mapping[need] = telemetry_map.get(need)
            else:
                mapping[need] = None

        return mapping

    # ========================================================================
    # STRUCT/FIELD OPERATIONS - Handle data structures
    # ========================================================================

    STRUCT_PATTERN = re.compile(
        r'struct\s+(\w+)\s*\{([^}]+)\}',
        re.DOTALL
    )

    FIELD_ACCESS_PATTERN = re.compile(
        r'(\w+)\s*([->]+)\s*(\w+)',
        re.MULTILINE
    )

    @dataclass
    class StructDef:
        """Parsed struct definition"""
        name: str
        fields: List[str]  # field names
        start: int
        end: int

    @staticmethod
    def find_structs(content: str) -> List['Scope.StructDef']:
        """Find all struct definitions"""
        structs = []
        for match in Scope.STRUCT_PATTERN.finditer(content):
            struct_name = match.group(1)
            body = match.group(2)

            # Extract field names (simplified - looks for "type name;" patterns)
            field_pattern = re.compile(r'\b(\w+)\s*;')
            fields = []
            for field_match in field_pattern.finditer(body):
                # The identifier before the semicolon is likely the field name
                # More robust: look for "type identifier;" but this is good enough
                parts = field_match.group(0).strip().rstrip(';').split()
                if len(parts) >= 2:
                    fields.append(parts[-1])  # Last word before ; is field name

            structs.append(Scope.StructDef(
                name=struct_name,
                fields=fields,
                start=match.start(),
                end=match.end()
            ))

        return structs

    @dataclass
    class FieldAccess:
        """A field access like obj->field or obj.field"""
        object_name: str
        accessor: str  # '->' or '.'
        field_name: str
        start: int
        end: int

    @staticmethod
    def find_field_accesses(content: str, struct_name: str, field_name: str) -> List['Scope.FieldAccess']:
        """
        Find all accesses to struct_name->field_name or struct_name.field_name
        Needed for: "Rename all ca_state->ca_input to ca_state->ca_concentration"
        """
        accesses = []

        # Pattern: struct_name followed by -> or . followed by field_name
        pattern = re.compile(
            rf'\b{re.escape(struct_name)}\s*([->\.]+)\s*{re.escape(field_name)}\b'
        )

        for match in pattern.finditer(content):
            accesses.append(Scope.FieldAccess(
                object_name=struct_name,
                accessor=match.group(1).strip(),
                field_name=field_name,
                start=match.start(),
                end=match.end()
            ))

        return accesses

    @staticmethod
    def rename_field(content: str, struct_name: str, old_field: str, new_field: str) -> str:
        """
        Rename all struct_name->old_field to struct_name->new_field
        Needed for: Field refactoring without writing custom regex each time
        """
        # Find all accesses
        accesses = Scope.find_field_accesses(content, struct_name, old_field)

        # Replace in reverse order to preserve positions
        new_content = content
        for access in reversed(accesses):
            replacement = f"{struct_name}{access.accessor}{new_field}"
            new_content = new_content[:access.start] + replacement + new_content[access.end:]

        return new_content


# ============================================================================
# 4. ConstantSpace - Semantic constant management
# ============================================================================

@dataclass
class Constant:
    """A constant with semantic meaning"""
    name: str
    value: str
    type: str  # 'int', 'float', 'unsigned', 'double'
    occurrences: List[Tuple[Path, int]]  # (file, line) where it appears

class ConstantSpace:
    """
    Constants as a semantic space with value->name mapping.
    Replaces ConstantRegistry, ConfigManager, SemanticNameGenerator.
    """

    def __init__(self):
        self.constants: Dict[str, Constant] = {}
        self.value_index: Dict[str, str] = {}  # value -> canonical name

    def register(self, name: str, value: str, const_type: str,
                 location: Tuple[Path, int] = None):
        """Add or update a constant"""
        if name in self.constants:
            const = self.constants[name]
            if location:
                const.occurrences.append(location)
        else:
            const = Constant(
                name=name,
                value=value,
                type=const_type,
                occurrences=[location] if location else []
            )
            self.constants[name] = const
            self.value_index[value] = name

    def find_by_value(self, value: str) -> Optional[str]:
        """Lookup constant name by value"""
        return self.value_index.get(value)

    def find_by_name(self, name: str) -> Optional[Constant]:
        """Lookup constant by name"""
        return self.constants.get(name)

    def suggest_name(self, value: str, context: str) -> str:
        """
        Generate semantic name from value and context.
        Heuristics for common patterns.
        """
        # Check if already exists
        existing = self.find_by_value(value)
        if existing:
            return existing

        # Infer from context
        ctx_lower = context.lower()

        # Genome-related
        if 'genome' in ctx_lower and value in ('512', '1024'):
            return 'GENOME_SIZE'

        # Dimensions
        if 'grid' in ctx_lower and value == '128':
            return 'GRID_SIZE'
        if 'warp' in ctx_lower and value == '32':
            return 'WARP_SIZE'
        if 'block' in ctx_lower and value == '256':
            return 'BLOCK_SIZE'
        if 'tile' in ctx_lower and value == '16':
            return 'WMMA_TILE_DIM'

        # Thresholds
        if 'threshold' in ctx_lower:
            if '0.8' in value:
                return 'FITNESS_THRESHOLD'
            if '0.01' in value:
                return 'DELTA_THRESHOLD'

        # Rates
        if 'rate' in ctx_lower:
            if '0.1' in value:
                return 'LEARNING_RATE'
            if '0.95' in value:
                return 'DECAY_RATE'

        # Generic fallback
        if '.' in value or 'f' in value.lower():
            return f"CONST_FLOAT_{value.replace('.', '_').replace('f', '').replace('-', 'NEG')}"
        else:
            return f"CONST_INT_{value}"

    def write_config(self, path: Path) -> Result[Path]:
        """Write all constants to config.cu"""
        # Group by type
        int_consts = [(n, c) for n, c in self.constants.items() if c.type in ('int', 'unsigned')]
        float_consts = [(n, c) for n, c in self.constants.items() if c.type in ('float', 'double')]

        lines = ['#ifndef CONFIG_CU', '#define CONFIG_CU', '']

        # Integer constants
        if int_consts:
            lines.append('// Integer constants')
            for name, const in sorted(int_consts):
                lines.append(f'constexpr {const.type} {name} = {const.value};')
            lines.append('')

        # Float constants
        if float_consts:
            lines.append('// Floating-point constants')
            for name, const in sorted(float_consts):
                lines.append(f'constexpr {const.type} {name} = {const.value};')
            lines.append('')

        lines.append('#endif')

        content = '\n'.join(lines)
        return FileOp.write(path, content)


# ============================================================================
# 5. Pipeline - Transformation composition with dependencies
# ============================================================================

@dataclass
class Stage:
    """A transformation stage in the pipeline"""
    name: str
    transform: Callable[[Dict], Result[Dict]]  # context -> new context
    dependencies: Set[str]  # names of stages this depends on

class Pipeline:
    """
    Composable transformation pipeline with dependency resolution.
    Replaces RefactoringOrchestrator, MasterRefactorOrchestrator.
    """

    def __init__(self):
        self.stages: Dict[str, Stage] = {}
        self.execution_order: List[str] = []

    def add_stage(self, name: str,
                  transform: Callable[[Dict], Result[Dict]],
                  depends_on: List[str] = None) -> 'Pipeline':
        """Add a stage to the pipeline"""
        stage = Stage(
            name=name,
            transform=transform,
            dependencies=set(depends_on or [])
        )
        self.stages[name] = stage
        return self

    def _resolve_dependencies(self) -> Result[List[str]]:
        """Topological sort of stages by dependencies"""
        visited = set()
        order = []

        def visit(name: str, path: Set[str]):
            if name in path:
                return Err(f"Circular dependency: {' -> '.join(path)} -> {name}")
            if name in visited:
                return Ok(None)

            stage = self.stages[name]
            path.add(name)

            for dep in stage.dependencies:
                if dep not in self.stages:
                    return Err(f"Unknown dependency: {name} depends on {dep}")
                result = visit(dep, path)
                if isinstance(result, Error):
                    return result

            path.remove(name)
            visited.add(name)
            order.append(name)
            return Ok(None)

        for name in self.stages:
            result = visit(name, set())
            if isinstance(result, Error):
                return result

        return Ok(order)

    def run(self, initial_context: Dict = None) -> Result[Dict]:
        """Execute pipeline in dependency order"""
        # Resolve execution order
        order_result = self._resolve_dependencies()
        if isinstance(order_result, Error):
            return order_result

        execution_order = order_result.value
        context = initial_context or {}

        # Execute stages
        for stage_name in execution_order:
            stage = self.stages[stage_name]
            print(f"[Pipeline] Running: {stage_name}")

            result = stage.transform(context)
            if isinstance(result, Error):
                return Err(f"Stage {stage_name} failed",
                          [result.message] + result.context)

            context = result.value

        return Ok(context)


# ============================================================================
# Utility: Bind operation for composing Result-returning functions
# ============================================================================

def bind(result: Result[T], f: Callable[[T], Result[U]]) -> Result[U]:
    """
    Chain operations that return Result.
    If result is Error, propagate it. Otherwise apply f to the value.

    This is the key to composability - no manual error checking needed.
    """
    if isinstance(result, Error):
        return result
    return f(result.value)


# ============================================================================
# 6. CUDAParser - AST-based parsing using tree-sitter
# ============================================================================

class CUDAParser:
    """
    CUDA parser using tree-sitter for AST traversal.
    Handles comments, strings, macros, templates, nested braces.
    """

    @staticmethod
    def parse_file(filepath: Path):
        """Parse CUDA file into tree-sitter AST"""
        try:
            from tree_sitter import Language, Parser
            from tree_sitter_cuda import language
        except ImportError:
            return Err("tree-sitter-cuda not installed. Run: pip install tree-sitter tree-sitter-cuda")

        file_result = FileOp.read(filepath)
        if file_result.is_err():
            return file_result

        source_code = file_result.value.encode('utf-8')
        parser = Parser(Language(language()))
        tree = parser.parse(source_code)

        return Ok(tree)

    @dataclass
    class StructInfo:
        """Complete struct information from AST"""
        name: str
        size_bytes: int
        fields: List[Tuple[str, str, int]]  # (name, type, size)
        location: Tuple[str, int]  # (file, line)

    @staticmethod
    def get_struct_size(struct_name: str, search_paths: List[Path] = None) -> Result[int]:
        """Get struct size by parsing and calculating field sizes"""
        if search_paths is None:
            search_paths = [Paths.repo_root / 'slime']

        # Search for struct definition
        for search_dir in search_paths:
            for cu_file in search_dir.rglob("*.cu"):
                result = CUDAParser._find_struct_in_file(cu_file, struct_name)
                if result.is_ok() and result.value is not None:
                    return Ok(result.value.size_bytes)

            for cuh_file in search_dir.rglob("*.cuh"):
                result = CUDAParser._find_struct_in_file(cuh_file, struct_name)
                if result.is_ok() and result.value is not None:
                    return Ok(result.value.size_bytes)

        return Err(f"Struct {struct_name} not found")

    @staticmethod
    def _find_struct_in_file(filepath: Path, struct_name: str) -> Result[Optional['CUDAParser.StructInfo']]:
        """Find struct in specific file using tree-sitter AST"""
        tree_result = CUDAParser.parse_file(filepath)
        if tree_result.is_err():
            return tree_result

        tree = tree_result.value

        def visit_node(node):
            # Look for struct definition
            if node.type == 'struct_specifier':
                # Get struct name
                name_node = node.child_by_field_name('name')
                if name_node and name_node.text.decode('utf-8') == struct_name:
                    # Get fields and calculate size
                    fields = []
                    total_size = 0

                    body_node = node.child_by_field_name('body')
                    if body_node:
                        for child in body_node.children:
                            if child.type == 'field_declaration':
                                # Extract field type and name
                                field_type = None
                                field_name = None

                                for subchild in child.children:
                                    if subchild.type == 'type_identifier' or subchild.type == 'primitive_type':
                                        field_type = subchild.text.decode('utf-8')
                                    elif subchild.type == 'field_identifier':
                                        field_name = subchild.text.decode('utf-8')

                                if field_type and field_name:
                                    # Calculate field size
                                    field_size = CUDAParser._get_type_size(field_type)
                                    fields.append((field_name, field_type, field_size))
                                    total_size += field_size

                    # Align to 8-byte boundary
                    total_size = ((total_size + 7) // 8) * 8

                    return CUDAParser.StructInfo(
                        name=struct_name,
                        size_bytes=total_size,
                        fields=fields,
                        location=(str(filepath), node.start_point[0] + 1)
                    )

            # Recurse
            for child in node.children:
                result = visit_node(child)
                if result is not None:
                    return result

            return None

        struct_info = visit_node(tree.root_node)
        return Ok(struct_info)

    @staticmethod
    def _get_type_size(type_name: str) -> int:
        """Get size of a primitive type"""
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
        return type_sizes.get(type_name, 8)  # Default to pointer size

    @dataclass
    class MallocCall:
        """cudaMalloc call information from AST"""
        var_name: str
        size_expr: str
        location: Tuple[str, int]
        function_name: str  # Which function contains this malloc

    @staticmethod
    def find_mallocs(filepath: Path) -> Result[List['CUDAParser.MallocCall']]:
        """Find all cudaMalloc AND TRACKED_ALLOC calls using tree-sitter CUDA parser"""
        try:
            from tree_sitter import Language, Parser
            from tree_sitter_cuda import language
        except ImportError:
            return Err("tree-sitter-cuda not installed. Run: pip install tree-sitter tree-sitter-cuda")

        # Read source file
        file_result = FileOp.read(filepath)
        if file_result.is_err():
            return file_result

        source_code = file_result.value.encode('utf-8')

        # Create parser with CUDA language
        parser = Parser(Language(language()))
        tree = parser.parse(source_code)

        mallocs = []

        # Query for cudaMalloc calls
        def visit_node(node, parent_func=None):
            # Track function context
            if node.type == 'function_definition':
                # Get function name from declarator
                for child in node.children:
                    if child.type == 'function_declarator':
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                parent_func = subchild.text.decode('utf-8')
                                break
                        break

            # Look for call expressions
            if node.type == 'call_expression':
                # Get function name
                func_node = node.child_by_field_name('function')
                if func_node and func_node.type == 'identifier':
                    func_name = func_node.text.decode('utf-8')

                    if func_name == 'cudaMalloc':
                        # Get arguments
                        args_node = node.child_by_field_name('arguments')
                        if args_node:
                            args_text = args_node.text.decode('utf-8')

                            # Parse arguments: cudaMalloc(&var, size)
                            import re
                            match = re.match(r'\(\s*&\s*([^,]+)\s*,\s*(.+)\s*\)', args_text)
                            if match:
                                var_name = match.group(1).strip()
                                size_expr = match.group(2).strip()

                                line_num = node.start_point[0] + 1  # 0-indexed to 1-indexed

                                mallocs.append(CUDAParser.MallocCall(
                                    var_name=var_name,
                                    size_expr=size_expr,
                                    location=(str(filepath), line_num),
                                    function_name=parent_func or 'unknown'
                                ))

                    elif func_name == 'TRACKED_ALLOC':
                        # Get arguments: TRACKED_ALLOC(ptr, size, telemetry, counter)
                        args_node = node.child_by_field_name('arguments')
                        if args_node:
                            args_text = args_node.text.decode('utf-8')

                            # Parse arguments: TRACKED_ALLOC(var, size, ...)
                            import re
                            match = re.match(r'\(\s*([^,]+)\s*,\s*([^,]+)\s*,.*\)', args_text)
                            if match:
                                var_name = match.group(1).strip()
                                size_expr = match.group(2).strip()

                                line_num = node.start_point[0] + 1

                                mallocs.append(CUDAParser.MallocCall(
                                    var_name=var_name,
                                    size_expr=size_expr,
                                    location=(str(filepath), line_num),
                                    function_name=parent_func or 'unknown'
                                ))

            # Recurse
            for child in node.children:
                visit_node(child, parent_func)

        visit_node(tree.root_node)
        return Ok(mallocs)

    @dataclass
    class ConstantDef:
        """Constant definition from AST"""
        name: str
        value: str
        type: str
        location: Tuple[str, int]

    @staticmethod
    def extract_constants(filepath: Path) -> Result[List['CUDAParser.ConstantDef']]:
        """Extract all constexpr/const definitions using tree-sitter"""
        tree_result = CUDAParser.parse_file(filepath)
        if tree_result.is_err():
            return tree_result

        tree = tree_result.value
        constants = []

        def visit_node(node):
            # Look for variable declarations with constexpr/const
            if node.type == 'declaration':
                # Check if it has constexpr or const (type_qualifier not storage_class_specifier)
                has_const = False
                for child in node.children:
                    if child.type == 'type_qualifier' and child.text.decode('utf-8') in ['constexpr', 'const']:
                        has_const = True
                        break

                if has_const:
                    # Extract variable name and value
                    name = None
                    value = None
                    type_name = None

                    for child in node.children:
                        if child.type == 'primitive_type' or child.type == 'type_identifier':
                            type_name = child.text.decode('utf-8')
                        elif child.type == 'init_declarator':
                            # Get identifier and initializer
                            # First child is identifier (variable name)
                            # Third child (after '=') is the value
                            init_children = list(child.children)
                            if len(init_children) >= 3:
                                name = init_children[0].text.decode('utf-8')
                                value = init_children[2].text.decode('utf-8')

                    if name and value and type_name:
                        constants.append(CUDAParser.ConstantDef(
                            name=name,
                            value=value,
                            type=type_name,
                            location=(str(filepath), node.start_point[0] + 1)
                        ))

            # Recurse
            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return Ok(constants)

    @staticmethod
    def extract_local_variables(filepath: Path, function_name: str) -> Result[dict]:
        """Extract local variable assignments from a function using tree-sitter"""
        tree_result = CUDAParser.parse_file(filepath)
        if tree_result.is_err():
            return tree_result

        tree = tree_result.value
        local_vars = {}

        def visit_node(node, in_target_func=False):
            # Find the target function
            if node.type == 'function_definition':
                # Get function name
                func_name = None
                for child in node.children:
                    if child.type == 'function_declarator':
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                func_name = subchild.text.decode('utf-8')
                                break
                        break

                if func_name == function_name:
                    in_target_func = True

            # Within target function, find variable declarations with initializers
            if in_target_func and node.type == 'declaration':
                # Extract type, name, and value
                var_type = None
                var_name = None
                var_value = None

                for child in node.children:
                    if child.type == 'primitive_type' or child.type == 'type_identifier':
                        var_type = child.text.decode('utf-8')
                    elif child.type == 'init_declarator':
                        # Get identifier and initializer
                        # First child is identifier, third child (after '=') is the value
                        init_children = list(child.children)
                        if len(init_children) >= 3:
                            var_name = init_children[0].text.decode('utf-8')
                            var_value = init_children[2].text.decode('utf-8')

                if var_type in ('int', 'size_t') and var_name and var_value:
                    local_vars[var_name] = var_value

            # Recurse
            for child in node.children:
                visit_node(child, in_target_func)

        visit_node(tree.root_node)
        return Ok(local_vars)

    @dataclass
    class StructAssignment:
        """A struct field assignment found in code"""
        struct_var: str
        field_name: str
        value_expr: str
        function_name: str
        location: Tuple[str, int]

    @staticmethod
    def find_struct_assignments(filepath: Path, struct_type: str = None) -> Result[List['CUDAParser.StructAssignment']]:
        """Find all struct field assignments, optionally filtered by struct type"""
        tree_result = CUDAParser.parse_file(filepath)
        if tree_result.is_err():
            return tree_result

        tree = tree_result.value
        assignments = []
        current_function = None

        def visit_node(node):
            nonlocal current_function

            # Track current function
            if node.type == 'function_definition':
                func_name = None
                for child in node.children:
                    if child.type == 'function_declarator':
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                func_name = subchild.text.decode('utf-8')
                                break
                        break
                old_func = current_function
                current_function = func_name

            # Find assignment expressions
            if node.type == 'assignment_expression':
                left_node = node.child_by_field_name('left')
                right_node = node.child_by_field_name('right')

                if left_node and right_node and left_node.type == 'field_expression':
                    # Parse struct_var.field_name
                    field_text = left_node.text.decode('utf-8')
                    parts = field_text.split('.')
                    if len(parts) >= 2:
                        struct_var = '.'.join(parts[:-1])
                        field_name = parts[-1]
                        value_expr = right_node.text.decode('utf-8')

                        # If struct_type filter is provided, check if this matches
                        # (we can't easily determine type from AST, so accept all for now)
                        assignments.append(CUDAParser.StructAssignment(
                            struct_var=struct_var,
                            field_name=field_name,
                            value_expr=value_expr,
                            function_name=current_function or 'unknown',
                            location=(str(filepath), node.start_point[0] + 1)
                        ))

            # Recurse
            for child in node.children:
                visit_node(child)

            # Restore function context after visiting children
            if node.type == 'function_definition':
                current_function = old_func

        visit_node(tree.root_node)
        return Ok(assignments)


# ============================================================================
# 7. Genome simulation - exact RNG and derive_architecture from CUDA code
# ============================================================================

class GenomeSimulator:
    """Simulate CUDA genome generation and architecture derivation exactly"""

    @staticmethod
    def xorshift128plus(s0: int, s1: int) -> tuple:
        """
        Exact implementation of PRNGState.next() from pool.cu
        Returns (next_value, new_s0, new_s1)
        """
        XORSHIFT_A = 17
        XORSHIFT_B = 11
        XORSHIFT_C = 25
        XORSHIFT_NORMALIZATION_SCALE = float(1 << 63)

        x = s0
        y = s1
        s0 = y
        x ^= (x << XORSHIFT_A) & 0xFFFFFFFFFFFFFFFF
        s1 = (x ^ y ^ (x >> XORSHIFT_B) ^ (y >> XORSHIFT_C)) & 0xFFFFFFFFFFFFFFFF
        result = ((s1 + y) & 0xFFFFFFFFFFFFFFFF) / XORSHIFT_NORMALIZATION_SCALE
        return (result, s0, s1)

    @staticmethod
    def generate_genome(idx: int, genome_size: int, genome_range_scale: float, genome_value_min: float) -> list:
        """
        Generate genome exactly as init_pool_kernel does
        idx: pool entry index (0 for first entry)
        """
        # Seed RNG exactly as in pool.cu line 345-346
        s0 = (idx * 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
        s1 = (idx * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF

        genome = []
        for i in range(genome_size):
            val, s0, s1 = GenomeSimulator.xorshift128plus(s0, s1)
            genome_val = val * genome_range_scale + genome_value_min
            genome.append(genome_val)

        return genome

    @staticmethod
    def fnv1a_hash(s: str) -> int:
        """FNV-1a hash implementation from tile_ops.cuh"""
        FNV_OFFSET = 0xcbf29ce484222325
        FNV_PRIME = 0x100000001b3

        hash_val = FNV_OFFSET
        for c in s.encode('utf-8'):
            hash_val ^= c
            hash_val = (hash_val * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        return hash_val

    @staticmethod
    def derive_param_slot(genome_hash: int, param_id: str, genome_size: int) -> int:
        """derive_param_slot from tile_ops.cuh line 129"""
        id_hash = GenomeSimulator.fnv1a_hash(param_id)

        combined = genome_hash ^ id_hash
        combined ^= (combined >> 33)
        combined = (combined * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        combined ^= (combined >> 33)
        combined = (combined * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        combined ^= (combined >> 33)

        return int(combined % genome_size)

    @staticmethod
    def derive_architecture(genome_hash: int, genome: list, constants: dict) -> dict:
        """
        Exact implementation of derive_architecture from tile_ops.cuh line 159
        Returns dict with num_heads, channels, hidden_dim, head_dim, grid_size
        Raises KeyError if required constants are missing
        """
        # Extract required constants - fail if not present
        GENOME_TO_UNIT_OFFSET = constants['GENOME_TO_UNIT_OFFSET']
        GENOME_TO_UNIT_SCALE = constants['GENOME_TO_UNIT_SCALE']
        NUM_HEADS_MIN = constants['NUM_HEADS_MIN']
        NUM_HEADS_MAX = constants['NUM_HEADS_MAX']
        CHANNELS_MIN = constants['CHANNELS_MIN']
        CHANNELS_MAX = constants['CHANNELS_MAX']
        HEAD_DIM_MIN = constants['HEAD_DIM_MIN']
        HEAD_DIM_MAX = constants['HEAD_DIM_MAX']
        GRID_SIZE_MIN = constants['GRID_SIZE_MIN']
        GRID_SIZE_MAX = constants['GRID_SIZE_MAX']
        GENOME_SIZE = len(genome)

        num_heads_slot = GenomeSimulator.derive_param_slot(genome_hash, "arch_num_heads", GENOME_SIZE)
        channels_slot = GenomeSimulator.derive_param_slot(genome_hash, "arch_channels", GENOME_SIZE)
        head_dim_slot = GenomeSimulator.derive_param_slot(genome_hash, "arch_head_dim", GENOME_SIZE)
        grid_size_slot = GenomeSimulator.derive_param_slot(genome_hash, "arch_grid_size", GENOME_SIZE)

        num_heads_norm = (genome[num_heads_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE
        channels_norm = (genome[channels_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE
        head_dim_norm = (genome[head_dim_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE
        grid_size_norm = (genome[grid_size_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE

        num_heads = NUM_HEADS_MIN + int(num_heads_norm * (NUM_HEADS_MAX - NUM_HEADS_MIN))
        channels = CHANNELS_MIN + int(channels_norm * (CHANNELS_MAX - CHANNELS_MIN))
        head_dim = HEAD_DIM_MIN + int(head_dim_norm * (HEAD_DIM_MAX - HEAD_DIM_MIN))
        grid_size = GRID_SIZE_MIN + int(grid_size_norm * (GRID_SIZE_MAX - GRID_SIZE_MIN))
        hidden_dim = num_heads * head_dim

        return {
            'num_heads': num_heads,
            'channels': channels,
            'hidden_dim': hidden_dim,
            'head_dim': head_dim,
            'grid_size': grid_size
        }

    @staticmethod
    def gpu_sha256_stub(genome: list) -> int:
        """
        Stub for gpu_sha256 - uses Python hashlib
        Note: This won't match CUDA exactly but gives deterministic hash
        """
        import hashlib
        genome_bytes = b''.join(str(x).encode() for x in genome)
        hash_digest = hashlib.sha256(genome_bytes).digest()
        # Convert first 8 bytes to uint64
        return int.from_bytes(hash_digest[:8], 'little')

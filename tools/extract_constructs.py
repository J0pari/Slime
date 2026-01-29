#!/usr/bin/env python3
"""
Schema-driven code extraction.

Finds structs/functions/kernels by name, detects their bounds via brace matching,
and physically moves them to new files.
"""

import re
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class Construct:
    """A found code construct with its bounds"""
    name: str
    kind: str  # 'struct', 'kernel', 'function', 'block'
    start_line: int
    end_line: int
    content: str

@dataclass
class ExtractionSpec:
    """What to extract and where"""
    name: str
    kind: str  # 'struct', 'kernel', 'function'
    dest_file: str
    action: str = 'move'  # 'move', 'delete', 'copy'
    canonical: str = ''  # If delete, what's the canonical location

@dataclass
class FileSpec:
    """Schema for one source file"""
    source: str
    extractions: List[ExtractionSpec] = field(default_factory=list)

class CUDAParser:
    """Parses CUDA/C++ to find construct boundaries"""

    def __init__(self, content: str):
        self.lines = content.split('\n')
        self.content = content

    def find_struct(self, name: str) -> Optional[Construct]:
        """Find struct by name, return with full bounds"""
        # Match: struct Name { (with word boundary to avoid substring matches)
        # Must have { on same line to distinguish from forward declaration
        # \b ensures we don't match OrganismPreallocatedBuffers when looking for Organism
        pattern = re.compile(rf'^\s*struct\s+{re.escape(name)}\b\s*\{{')

        for i, line in enumerate(self.lines):
            if pattern.match(line):
                start = i
                end = self._find_closing_brace(i, expect_semicolon=True)
                if end:
                    content = '\n'.join(self.lines[start:end+1])
                    return Construct(name, 'struct', start+1, end+1, content)
        return None

    def find_kernel(self, name: str) -> Optional[Construct]:
        """Find __global__ kernel by name"""
        return self._find_function(name, require_global=True)

    def find_function(self, name: str) -> Optional[Construct]:
        """Find any function by name"""
        return self._find_function(name, require_global=False)

    def _find_function(self, name: str, require_global: bool) -> Optional[Construct]:
        """Find function/kernel definition (not declaration)"""
        # Build pattern for function signature
        if require_global:
            sig_pattern = re.compile(
                rf'(?:extern\s+"C"\s+)?__global__\s+void\s+{re.escape(name)}\s*\('
            )
        else:
            sig_pattern = re.compile(
                rf'(?:__device__|__host__|__global__|static|inline|\s)*'
                rf'(?:\w+(?:<[^>]+>)?(?:\s*\*)*\s+)+{re.escape(name)}\s*\('
            )

        i = 0
        while i < len(self.lines):
            line = self.lines[i]

            # Skip forward declarations (line ends with ;)
            if sig_pattern.search(line):
                # Scan ahead to see if this is declaration or definition
                j = i
                found_brace = False
                while j < len(self.lines) and j < i + 20:  # Look ahead max 20 lines
                    if '{' in self.lines[j]:
                        found_brace = True
                        break
                    if self.lines[j].rstrip().endswith(';'):
                        # It's a declaration, skip
                        break
                    j += 1

                if found_brace:
                    # This is a definition
                    start = i
                    end = self._find_closing_brace(i, expect_semicolon=False)
                    if end:
                        content = '\n'.join(self.lines[start:end+1])
                        return Construct(name, 'kernel' if require_global else 'function',
                                        start+1, end+1, content)
            i += 1
        return None

    def find_comment_block(self, marker: str) -> Optional[Construct]:
        """Find a section marked by // ========== MARKER =========="""
        start_pattern = re.compile(rf'//\s*=+\s*{re.escape(marker)}\s*=+')
        end_pattern = re.compile(r'//\s*=+\s*\w+.*=+')

        start = None
        for i, line in enumerate(self.lines):
            if start is None:
                if start_pattern.search(line):
                    start = i
            else:
                if end_pattern.search(line) and i > start:
                    content = '\n'.join(self.lines[start:i])
                    return Construct(marker, 'block', start+1, i, content)

        return None

    def _find_closing_brace(self, start_line: int, expect_semicolon: bool) -> Optional[int]:
        """Find the line with the closing brace, handling nesting"""
        brace_count = 0
        started = False

        for i in range(start_line, len(self.lines)):
            line = self.lines[i]

            for char in line:
                if char == '{':
                    brace_count += 1
                    started = True
                elif char == '}':
                    brace_count -= 1

            if started and brace_count == 0:
                # For structs, might need the semicolon line
                if expect_semicolon:
                    if '};' in line:
                        return i
                    # Check next line for semicolon
                    if i + 1 < len(self.lines) and self.lines[i+1].strip() == '};':
                        return i + 1
                return i

        return None


class Extractor:
    """Performs the actual extraction based on schema"""

    def __init__(self, root_dir: str):
        self.root = root_dir
        self.file_cache: Dict[str, str] = {}

    def load_file(self, path: str) -> str:
        """Load file content"""
        full_path = os.path.join(self.root, path)
        if full_path not in self.file_cache:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                self.file_cache[full_path] = f.read()
        return self.file_cache[full_path]

    def save_file(self, path: str, content: str):
        """Save file content"""
        full_path = os.path.join(self.root, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Wrote: {path}")

    def execute(self, spec: FileSpec, dry_run: bool = True):
        """Execute extraction for a file spec"""
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {spec.source}")

        content = self.load_file(spec.source)
        parser = CUDAParser(content)

        # Find all constructs first
        found: List[Tuple[ExtractionSpec, Construct]] = []

        for ex in spec.extractions:
            if ex.kind == 'struct':
                c = parser.find_struct(ex.name)
            elif ex.kind == 'kernel':
                c = parser.find_kernel(ex.name)
            elif ex.kind == 'function':
                c = parser.find_function(ex.name)
            elif ex.kind == 'block':
                c = parser.find_comment_block(ex.name)
            else:
                print(f"  Unknown kind: {ex.kind}")
                continue

            if c:
                found.append((ex, c))
                print(f"  Found {ex.kind} '{ex.name}' at lines {c.start_line}-{c.end_line} ({c.end_line - c.start_line + 1} lines)")
            else:
                print(f"  NOT FOUND: {ex.kind} '{ex.name}'")

        if dry_run:
            print("\n  Would perform:")
            for ex, c in found:
                if ex.action == 'delete':
                    print(f"    DELETE {c.name} (canonical: {ex.canonical})")
                elif ex.action == 'move':
                    print(f"    MOVE {c.name} -> {ex.dest_file}")
                elif ex.action == 'copy':
                    print(f"    COPY {c.name} -> {ex.dest_file}")
            return

        # Sort by start line descending so we can remove from bottom up
        found.sort(key=lambda x: x[1].start_line, reverse=True)

        lines = content.split('\n')
        dest_contents: Dict[str, List[str]] = {}

        for ex, c in found:
            # Extract the content
            extracted = '\n'.join(lines[c.start_line-1:c.end_line])

            if ex.action == 'move':
                # Add to destination file
                if ex.dest_file not in dest_contents:
                    dest_contents[ex.dest_file] = []
                dest_contents[ex.dest_file].append(f"// Extracted from {spec.source} lines {c.start_line}-{c.end_line}")
                dest_contents[ex.dest_file].append(extracted)
                dest_contents[ex.dest_file].append("")

                # Remove from source (replace with comment)
                lines[c.start_line-1:c.end_line] = [f"// MOVED TO {ex.dest_file}: {c.name}"]

            elif ex.action == 'delete':
                # Just remove, add comment pointing to canonical
                lines[c.start_line-1:c.end_line] = [
                    f"// DELETED: {c.name} - was duplicate of {ex.canonical}"
                ]

            elif ex.action == 'copy':
                if ex.dest_file not in dest_contents:
                    dest_contents[ex.dest_file] = []
                dest_contents[ex.dest_file].append(extracted)
                dest_contents[ex.dest_file].append("")

        # Write destination files
        for dest_path, content_parts in dest_contents.items():
            self.save_file(dest_path, '\n'.join(content_parts))

        # Write modified source
        self.save_file(spec.source, '\n'.join(lines))


# ============================================================================
# SCHEMA DEFINITION - Edit this to define what gets extracted where
# ============================================================================

ORGANISM_SCHEMA = FileSpec(
    source='slime/core/organism.cu',
    extractions=[
        # Types to extract to header
        ExtractionSpec('OrganismPreallocatedBuffers', 'struct', 'slime/core/organism_types.cuh', 'move'),
        ExtractionSpec('Organism', 'struct', 'slime/core/organism_types.cuh', 'move'),
        ExtractionSpec('MemoryUpdateParams', 'struct', 'slime/core/organism_types.cuh', 'move'),

        # Duplicate kernel to DELETE
        ExtractionSpec('neural_ca_update_kernel', 'kernel', '', 'delete',
                      canonical='slime/kernels/tensor_core_ca.cu::multi_head_ca_tensor_kernel'),

        # Unique kernels to extract
        ExtractionSpec('behavioral_update_kernel', 'kernel', 'slime/core/behavioral.cu', 'move'),
        ExtractionSpec('store_navigation_history_kernel', 'kernel', 'slime/core/behavioral.cu', 'move'),
        ExtractionSpec('init_organism_kernel', 'kernel', 'slime/core/organism_init.cu', 'move'),
        ExtractionSpec('init_organism_phase2_kernel', 'kernel', 'slime/core/organism_init.cu', 'move'),
        ExtractionSpec('component_evolution_kernel', 'kernel', 'slime/lifecycle/selection.cu', 'move'),
        ExtractionSpec('selection_kernel', 'kernel', 'slime/lifecycle/selection.cu', 'move'),
        ExtractionSpec('spawn_wave_kernel', 'kernel', 'slime/lifecycle/selection.cu', 'move'),
        ExtractionSpec('culling_kernel', 'kernel', 'slime/lifecycle/selection.cu', 'move'),
    ]
)

HYBRID_SCHEMA = FileSpec(
    source='slime/training/hybrid_lifecycle.cu',
    extractions=[
        # Duplicate blocks to DELETE
        ExtractionSpec('ADAM UPDATES', 'block', '', 'delete',
                      canonical='slime/training/optimizer.cu::adam_apply_unified_*'),
    ]
)


def main():
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extractor = Extractor(root)

    dry_run = '--execute' not in sys.argv

    if dry_run:
        print("=" * 60)
        print("DRY RUN - No files will be modified")
        print("Run with --execute to actually perform extraction")
        print("=" * 60)

    extractor.execute(ORGANISM_SCHEMA, dry_run=dry_run)
    extractor.execute(HYBRID_SCHEMA, dry_run=dry_run)

    if dry_run:
        print("\n" + "=" * 60)
        print("To execute: python tools/extract_constructs.py --execute")
        print("=" * 60)


if __name__ == '__main__':
    main()

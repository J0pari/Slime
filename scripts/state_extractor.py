#!/usr/bin/env python3

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from core import FileOp, Result, Ok, Err, bind


class DiffLineType(Enum):
    ADDITION = "+"
    DELETION = "-"
    CONTEXT = " "


@dataclass
class DiffHunk:
    """Single contiguous change region in a file"""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    additions: List[str]
    deletions: List[str]

    @property
    def net_change(self) -> int:
        return len(self.additions) + len(self.deletions)


@dataclass
class FileDiff:
    """Complete diff state for a single file"""
    filepath: Path
    hunks: List[DiffHunk]

    @property
    def total_additions(self) -> int:
        return sum(len(h.additions) for h in self.hunks)

    @property
    def total_deletions(self) -> int:
        return sum(len(h.deletions) for h in self.hunks)

    @property
    def total_changes(self) -> int:
        return self.total_additions + self.total_deletions

    def extract_pattern_fixes(self, pattern: str) -> List[Tuple[str, str]]:
        """Extract all instances where pattern was replaced"""
        fixes = []
        for hunk in self.hunks:
            for deletion in hunk.deletions:
                if pattern in deletion:
                    # Find corresponding addition
                    for addition in hunk.additions:
                        if addition.replace(pattern, '') in deletion.replace(pattern, ''):
                            fixes.append((deletion.strip(), addition.strip()))
        return fixes


@dataclass
class RepositoryState:
    """Complete repository diff state extracted from git"""
    file_diffs: Dict[Path, FileDiff]

    @property
    def total_files_changed(self) -> int:
        return len(self.file_diffs)

    @property
    def total_lines_changed(self) -> int:
        return sum(fd.total_changes for fd in self.file_diffs.values())

    def files_matching(self, pattern: str) -> List[Path]:
        """Get all changed files matching glob pattern"""
        return [p for p in self.file_diffs.keys() if p.match(pattern)]


class GitStateExtractor:

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def _run_git(self, args: List[str]) -> Result[str]:
        """Execute git command and return result with proper error handling"""
        try:
            result = subprocess.run(
                ['git'] + args,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                return Err(f"git {' '.join(args)} failed: {result.stderr}")
            return Ok(result.stdout)
        except Exception as e:
            return Err(f"git command failed: {e}")

    def extract_numstat(self, path_filter: Optional[str] = None) -> Result[Dict[Path, Tuple[int, int]]]:
        """Extract numerical diff statistics: {filepath: (additions, deletions)}"""
        args = ['diff', '--numstat']
        if path_filter:
            args.append(path_filter)

        def parse_numstat(output: str) -> Result[Dict[Path, Tuple[int, int]]]:
            stats = {}
            for line in output.strip().split('\n'):
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 3:
                    added, removed, filepath = parts
                    if added != '-' and removed != '-':
                        stats[Path(filepath)] = (int(added), int(removed))
            return Ok(stats)

        return bind(self._run_git(args), parse_numstat)

    def extract_full_diff(self, path_filter: Optional[str] = None) -> Result[str]:
        """Extract complete unified diff"""
        args = ['diff']
        if path_filter:
            args.append(path_filter)

        return self._run_git(args)

    def parse_unified_diff(self, diff_text: str) -> Dict[Path, FileDiff]:
        """Parse unified diff into structured FileDiff objects"""
        file_diffs = {}
        current_file = None
        current_hunk = None

        for line in diff_text.split('\n'):
            if line.startswith('diff --git'):
                # Extract filepath from "diff --git a/path b/path"
                parts = line.split()
                if len(parts) >= 4:
                    filepath = Path(parts[2][2:])  # Remove "a/" prefix
                    current_file = filepath
                    file_diffs[filepath] = FileDiff(filepath=filepath, hunks=[])

            elif line.startswith('@@'):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                import re
                match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2) or '1')
                    new_start = int(match.group(3))
                    new_count = int(match.group(4) or '1')

                    current_hunk = DiffHunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        additions=[],
                        deletions=[]
                    )
                    if current_file and current_file in file_diffs:
                        file_diffs[current_file].hunks.append(current_hunk)

            elif current_hunk and line.startswith('+') and not line.startswith('+++'):
                current_hunk.additions.append(line[1:])

            elif current_hunk and line.startswith('-') and not line.startswith('---'):
                current_hunk.deletions.append(line[1:])

        return file_diffs

    def extract_repository_state(self, path_filter: Optional[str] = None) -> Result[RepositoryState]:
        """Extract complete repository state by interrogating git"""
        def create_state(diff_text: str) -> Result[RepositoryState]:
            file_diffs = self.parse_unified_diff(diff_text)
            return Ok(RepositoryState(file_diffs=file_diffs))

        return bind(self.extract_full_diff(path_filter), create_state)

    def verify_pattern_replaced(self, old_pattern: str, new_pattern: str,
                               path_filter: Optional[str] = None) -> Result[Dict[Path, int]]:
        """Verify a pattern was actually replaced by interrogating git diff"""
        def count_replacements(state: RepositoryState) -> Result[Dict[Path, int]]:
            replacements = {}
            for filepath, file_diff in state.file_diffs.items():
                count = 0
                for hunk in file_diff.hunks:
                    for deletion in hunk.deletions:
                        if old_pattern in deletion:
                            # Verify corresponding addition has new pattern
                            for addition in hunk.additions:
                                if new_pattern in addition:
                                    # Check if it's the same line context
                                    deletion_clean = deletion.replace(old_pattern, '').strip()
                                    addition_clean = addition.replace(new_pattern, '').strip()
                                    if deletion_clean == addition_clean:
                                        count += 1

                if count > 0:
                    replacements[filepath] = count

            return Ok(replacements)

        return bind(self.extract_repository_state(path_filter), count_replacements)

    def extract_mangled_scientific_notation(self) -> Result[Dict[Path, List[str]]]:
        """Find all remaining mangled scientific notation in working tree"""
        result = self._run_git(['grep', '-n', '1e-CONST_FLOAT'])
        if result.is_err():
            # grep returns error if no matches - that's OK
            return Ok({})

        findings = {}
        for line in result.value.strip().split('\n'):
            if not line:
                continue
            # Format: filepath:line_number:content
            parts = line.split(':', 2)
            if len(parts) == 3:
                filepath = Path(parts[0])
                content = parts[2]
                if filepath not in findings:
                    findings[filepath] = []
                findings[filepath].append(content.strip())

        return Ok(findings)

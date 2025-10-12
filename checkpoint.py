#!/usr/bin/env python3

import hashlib
import zlib
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import difflib
from datetime import datetime, timezone

class TestResultCheckpointSystem:
    def __init__(self, repo_dir: Path = Path.cwd()):
        self.repo_dir = repo_dir
        self.checkpoint_dir = repo_dir / ".test_checkpoints"
        self.objects_dir = self.checkpoint_dir / "objects"
        self.refs_dir = self.checkpoint_dir / "refs"
        self.head_file = self.checkpoint_dir / "HEAD"
        self.test_results_dir = repo_dir / "test_results"
        self._init_repo()

    def _init_repo(self):
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.objects_dir.mkdir(exist_ok=True)
        self.refs_dir.mkdir(exist_ok=True)
        self.test_results_dir.mkdir(exist_ok=True)

        if not self.head_file.exists():
            self.head_file.write_text("")

    def _hash_object(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _write_object(self, obj_type: str, content: bytes) -> str:
        header = f"{obj_type} {len(content)}\0".encode()
        full_content = header + content
        sha = self._hash_object(full_content)

        obj_dir = self.objects_dir / sha[:2]
        obj_dir.mkdir(exist_ok=True)
        obj_path = obj_dir / sha[2:]

        if not obj_path.exists():
            compressed = zlib.compress(full_content, level=9)
            obj_path.write_bytes(compressed)

        return sha

    def _read_object(self, sha: str) -> Tuple[str, bytes]:
        if len(sha) < 64:
            for obj_dir in self.objects_dir.iterdir():
                if obj_dir.is_dir() and obj_dir.name.startswith(sha[:2]):
                    for obj_file in obj_dir.iterdir():
                        full_sha = obj_dir.name + obj_file.name
                        if full_sha.startswith(sha):
                            sha = full_sha
                            break

        obj_path = self.objects_dir / sha[:2] / sha[2:]
        if not obj_path.exists():
            raise ValueError(f"Object {sha} not found")

        compressed = obj_path.read_bytes()
        full_content = zlib.decompress(compressed)

        header_end = full_content.index(b'\0')
        header = full_content[:header_end].decode()
        obj_type, size_str = header.split(' ')
        content = full_content[header_end + 1:]

        return obj_type, content

    def _create_checkpoint(self, parent_sha: Optional[str], content_sha: str, test_file: str, message: str) -> str:
        checkpoint_data = {
            'parent': parent_sha,
            'content': content_sha,
            'test_file': test_file,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        checkpoint_json = json.dumps(checkpoint_data, indent=2)
        return self._write_object('checkpoint', checkpoint_json.encode('utf-8'))

    def _read_checkpoint(self, checkpoint_sha: str) -> dict:
        if len(checkpoint_sha) < 64:
            for obj_dir in self.objects_dir.iterdir():
                if obj_dir.is_dir():
                    for obj_file in obj_dir.iterdir():
                        full_sha = obj_dir.name + obj_file.name
                        if full_sha.startswith(checkpoint_sha):
                            try:
                                obj_type, content = self._read_object(full_sha)
                                if obj_type == 'checkpoint':
                                    return json.loads(content.decode('utf-8'))
                            except:
                                continue
            raise ValueError(f"Checkpoint {checkpoint_sha} not found")

        obj_type, content = self._read_object(checkpoint_sha)
        if obj_type != 'checkpoint':
            raise ValueError(f"Object {checkpoint_sha} is not a checkpoint")
        return json.loads(content.decode('utf-8'))

    def _json_to_lines(self, json_obj: dict) -> List[str]:
        json_str = json.dumps(json_obj, indent=2, sort_keys=True)
        return [line + '\n' for line in json_str.splitlines()]

    def _lines_to_json(self, lines: List[str]) -> dict:
        json_str = ''.join(lines)
        return json.loads(json_str)

    def _store_content(self, json_data: dict, parent_sha: Optional[str], test_file: str) -> str:
        current_lines = self._json_to_lines(json_data)
        content_bytes = ''.join(current_lines).encode('utf-8')

        if parent_sha:
            try:
                parent_checkpoint = self._read_checkpoint(parent_sha)
                if parent_checkpoint['test_file'] != test_file:
                    return self._write_object('blob', content_bytes)

                parent_type, parent_content = self._read_object(parent_checkpoint['content'])

                if parent_type == 'delta':
                    parent_data = json.loads(parent_content.decode('utf-8'))
                    base_type, base_content = self._read_object(parent_data['base'])
                    parent_lines = base_content.decode('utf-8').splitlines(keepends=True)
                    for delta_sha in parent_data['deltas']:
                        delta_type, delta_content = self._read_object(delta_sha)
                        parent_lines = self._apply_delta(parent_lines, json.loads(delta_content.decode('utf-8')))
                else:
                    parent_lines = parent_content.decode('utf-8').splitlines(keepends=True)

                delta_ops = []
                matcher = difflib.SequenceMatcher(None, parent_lines, current_lines)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'delete':
                        delta_ops.append({'op': 'delete', 'start': i1, 'count': i2 - i1})
                    elif tag == 'insert':
                        delta_ops.append({'op': 'insert', 'start': i1, 'lines': current_lines[j1:j2]})
                    elif tag == 'replace':
                        delta_ops.append({'op': 'delete', 'start': i1, 'count': i2 - i1})
                        delta_ops.append({'op': 'insert', 'start': i1, 'lines': current_lines[j1:j2]})

                delta_json = json.dumps(delta_ops, ensure_ascii=False)
                delta_bytes = delta_json.encode('utf-8')

                if len(delta_bytes) < len(content_bytes) * 0.5:
                    delta_sha = self._write_object('delta_ops', delta_bytes)

                    if parent_type == 'delta':
                        parent_data['deltas'].append(delta_sha)
                        delta_chain_size = sum(len(self._read_object(d)[1]) for d in parent_data['deltas'])

                        if delta_chain_size > len(content_bytes) * 0.7:
                            return self._write_object('blob', content_bytes)
                        else:
                            return self._write_object('delta', json.dumps(parent_data).encode('utf-8'))
                    else:
                        delta_data = {
                            'base': parent_checkpoint['content'],
                            'deltas': [delta_sha]
                        }
                        return self._write_object('delta', json.dumps(delta_data).encode('utf-8'))
                else:
                    return self._write_object('blob', content_bytes)
            except:
                return self._write_object('blob', content_bytes)
        else:
            return self._write_object('blob', content_bytes)

    def _apply_delta(self, lines: List[str], delta_ops: list) -> List[str]:
        lines = list(lines)
        offset = 0

        for op in delta_ops:
            if op['op'] == 'delete':
                del lines[op['start'] + offset:op['start'] + offset + op['count']]
                offset -= op['count']
            elif op['op'] == 'insert':
                lines[op['start'] + offset:op['start'] + offset] = op['lines']
                offset += len(op['lines'])

        return lines

    def _get_content(self, checkpoint_sha: str) -> dict:
        checkpoint = self._read_checkpoint(checkpoint_sha)
        obj_type, content = self._read_object(checkpoint['content'])

        if obj_type == 'blob':
            return json.loads(content.decode('utf-8'))
        elif obj_type == 'delta':
            delta_data = json.loads(content.decode('utf-8'))
            base_type, base_content = self._read_object(delta_data['base'])
            result_lines = base_content.decode('utf-8').splitlines(keepends=True)

            for delta_sha in delta_data['deltas']:
                delta_type, delta_content = self._read_object(delta_sha)
                delta_ops = json.loads(delta_content.decode('utf-8'))
                result_lines = self._apply_delta(result_lines, delta_ops)

            return json.loads(''.join(result_lines))
        else:
            raise ValueError(f"Unknown content type: {obj_type}")

    def _get_current_checkpoint(self, test_file: str) -> Optional[str]:
        ref_file = self.refs_dir / f"{test_file}.ref"
        if not ref_file.exists():
            return None
        ref_content = ref_file.read_text().strip()
        if not ref_content:
            return None
        return ref_content

    def checkpoint_test_result(self, json_file: Path, message: str = None) -> str:
        if not json_file.exists():
            raise FileNotFoundError(f"Test result file {json_file} not found")

        test_name = json_file.stem
        json_data = json.loads(json_file.read_text(encoding='utf-8'))

        parent_sha = self._get_current_checkpoint(test_name)

        if parent_sha:
            try:
                parent_json = self._get_content(parent_sha)
                if parent_json == json_data:
                    print(f"No changes in {test_name}")
                    return parent_sha
            except:
                pass

        content_sha = self._store_content(json_data, parent_sha, test_name)

        if not message:
            message = f"Test run {test_name} at {datetime.now(timezone.utc).isoformat()}"

        checkpoint_sha = self._create_checkpoint(parent_sha, content_sha, test_name, message)

        ref_file = self.refs_dir / f"{test_name}.ref"
        ref_file.write_text(checkpoint_sha + "\n")

        if parent_sha:
            try:
                parent_json = self._get_content(parent_sha)
                parent_lines = self._json_to_lines(parent_json)
                current_lines = self._json_to_lines(json_data)
                diff = list(difflib.unified_diff(parent_lines, current_lines, n=0))
                changes = len([l for l in diff if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))])
                print(f"Checkpointed {test_name} [{checkpoint_sha[:8]}] with {changes} line changes")
            except:
                print(f"Checkpointed {test_name} [{checkpoint_sha[:8]}] (initial)")
        else:
            print(f"Checkpointed {test_name} [{checkpoint_sha[:8]}] (initial)")

        return checkpoint_sha

    def checkpoint_all_test_results(self) -> Dict[str, str]:
        results = {}
        for json_file in self.test_results_dir.glob("*.json"):
            try:
                checkpoint_sha = self.checkpoint_test_result(json_file)
                results[json_file.stem] = checkpoint_sha
            except Exception as e:
                print(f"Error checkpointing {json_file.stem}: {e}")
        return results

    def restore(self, test_name: str, checkpoint_sha: str = None, output_path: Path = None):
        if not checkpoint_sha:
            checkpoint_sha = self._get_current_checkpoint(test_name)

        if not checkpoint_sha:
            print(f"No checkpoints found for {test_name}")
            return

        if len(checkpoint_sha) < 64:
            for obj_dir in self.objects_dir.iterdir():
                if obj_dir.is_dir():
                    for obj_file in obj_dir.iterdir():
                        full_sha = obj_dir.name + obj_file.name
                        if full_sha.startswith(checkpoint_sha):
                            try:
                                obj_type, _ = self._read_object(full_sha)
                                if obj_type == 'checkpoint':
                                    checkpoint_sha = full_sha
                                    break
                            except:
                                continue

        json_data = self._get_content(checkpoint_sha)

        if output_path:
            output_file = output_path
        else:
            output_file = self.test_results_dir / f"{test_name}_restored_{checkpoint_sha[:8]}.json"

        output_file.write_text(json.dumps(json_data, indent=2))
        print(f"Restored {test_name} checkpoint {checkpoint_sha[:8]} to {output_file}")

    def log(self, test_name: str, limit: int = 10):
        checkpoint_sha = self._get_current_checkpoint(test_name)
        if not checkpoint_sha:
            print(f"No checkpoints found for {test_name}")
            return

        count = 0
        print(f"\nCheckpoint history for {test_name}:\n")
        while checkpoint_sha and count < limit:
            try:
                checkpoint = self._read_checkpoint(checkpoint_sha)
                print(f"checkpoint {checkpoint_sha[:8]}")
                print(f"    {checkpoint['message']}")
                print(f"    {checkpoint['timestamp']}")
                print()
                checkpoint_sha = checkpoint['parent']
                count += 1
            except:
                break

    def status(self):
        print("\nTest Result Checkpoint Status:\n")
        for ref_file in self.refs_dir.glob("*.ref"):
            test_name = ref_file.stem
            checkpoint_sha = ref_file.read_text().strip()

            json_file = self.test_results_dir / f"{test_name}.json"
            if json_file.exists():
                try:
                    current_json = json.loads(json_file.read_text())
                    checkpoint_json = self._get_content(checkpoint_sha)

                    if current_json == checkpoint_json:
                        print(f"{test_name}: clean (HEAD at {checkpoint_sha[:8]})")
                    else:
                        current_lines = self._json_to_lines(current_json)
                        checkpoint_lines = self._json_to_lines(checkpoint_json)
                        diff = list(difflib.unified_diff(checkpoint_lines, current_lines, n=0))
                        changes = len([l for l in diff if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))])
                        print(f"{test_name}: modified ({changes} line changes since {checkpoint_sha[:8]})")
                except Exception as e:
                    print(f"{test_name}: error ({e})")
            else:
                print(f"{test_name}: missing file (last checkpoint {checkpoint_sha[:8]})")

    def diff(self, test_name: str, checkpoint_sha1: str = None, checkpoint_sha2: str = None):
        if not checkpoint_sha2:
            checkpoint_sha2 = self._get_current_checkpoint(test_name)
            json_file = self.test_results_dir / f"{test_name}.json"
            if json_file.exists():
                json2 = json.loads(json_file.read_text())
            else:
                json2 = self._get_content(checkpoint_sha2)
        else:
            json2 = self._get_content(checkpoint_sha2)

        if checkpoint_sha1:
            json1 = self._get_content(checkpoint_sha1)
        else:
            checkpoint = self._read_checkpoint(checkpoint_sha2)
            if checkpoint['parent']:
                json1 = self._get_content(checkpoint['parent'])
            else:
                print(f"No parent checkpoint for {checkpoint_sha2[:8]}")
                return

        lines1 = self._json_to_lines(json1)
        lines2 = self._json_to_lines(json2)

        diff = difflib.unified_diff(lines1, lines2,
                                    fromfile=f"{test_name} {checkpoint_sha1[:8] if checkpoint_sha1 else 'parent'}",
                                    tofile=f"{test_name} {checkpoint_sha2[:8] if checkpoint_sha2 else 'current'}",
                                    lineterm='')

        for line in diff:
            print(line)

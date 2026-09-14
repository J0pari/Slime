"""End-to-end tests for the long-run harness against a fake binary.

The fake mimics the real CLI contract (cumulative target, per-generation
`gen N` progress, checkpoint generation line) so the harness's whole loop —
sequencing, warmup, checkpoint verification, PASS/FAIL — is exercised
without a GPU and without a 10-hour job. FAKE_STUCK=1 reproduces the
false-PASS condition (the checkpoint never advances) and the harness must
reject it. CPU-only.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "tests" / "long_run_check.py"
FAKE_PY = ROOT / "tests" / "fake_evolution_binary.py"


def make_fake_binary(td: Path) -> Path:
    """An executable wrapper so the harness can spawn it like a real binary
    on both platforms: a .cmd on Windows, a POSIX shell script with the
    execute bit elsewhere."""
    if os.name == "nt":
        cmd = td / "fake_evo.cmd"
        cmd.write_text(f'@echo off\n"{sys.executable}" "{FAKE_PY}" %*\n',
                       encoding="utf-8")
        return cmd
    script = td / "fake_evo"
    script.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_PY}" "$@"\n',
        encoding="utf-8")
    script.chmod(0o755)
    return script


class HarnessEndToEndTests(unittest.TestCase):
    def run_harness(self, td: Path, gens: int, chunk: int, extra_env=None,
                    extra_args=None):
        env = dict(os.environ)
        env.update(extra_env or {})
        return subprocess.run(
            [sys.executable, str(HARNESS),
             "--binary", str(make_fake_binary(td)),
             "--gens", str(gens), "--chunk", str(chunk),
             "--ckpt", str(td / "ck.bin"), *(extra_args or [])],
            capture_output=True, text=True, timeout=120, env=env)

    def test_end_to_end_pass_and_chunking(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            proc = self.run_harness(td, gens=10, chunk=5)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("PASS (10 generations", proc.stdout)
            self.assertEqual(proc.stdout.count("chunk ok:"), 2)
            self.assertEqual((td / "ck.bin.gen").read_text(
                encoding="utf-8"), "10")

    def test_stuck_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            proc = self.run_harness(td, gens=10, chunk=5,
                                    extra_env={"FAKE_STUCK": "1"})
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("did not advance", proc.stdout + proc.stderr)

    def test_gen_zero_flag_is_ignored_within_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            proc = self.run_harness(td, gens=10, chunk=5)
            self.assertNotIn("spontaneous stress", proc.stdout)

    def test_retry_resumes_from_the_checkpoint(self):
        # A scheduler retry must continue, not restart: run 10, then run 15
        # against the same checkpoint.
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            first = self.run_harness(td, gens=10, chunk=5)
            self.assertEqual(first.returncode, 0, first.stdout)
            second = self.run_harness(td, gens=15, chunk=5)
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn("resuming from generation 10", second.stdout)
            self.assertEqual((td / "ck.bin.gen").read_text(
                encoding="utf-8"), "15")

    def test_sustained_flags_fail_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            proc = self.run_harness(td, gens=20, chunk=10,
                                    extra_env={"FAKE_FLAG_EVERY": "1"},
                                    extra_args=["--warmup", "0"])
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("sustained", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

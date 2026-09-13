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
    """A .cmd wrapper so the harness can spawn it like a real executable."""
    cmd = td / "fake_evo.cmd"
    cmd.write_text(f'@echo off\n"{sys.executable}" "{FAKE_PY}" %*\n',
                   encoding="utf-8")
    return cmd


class HarnessEndToEndTests(unittest.TestCase):
    def run_harness(self, td: Path, gens: int, chunk: int, extra_env=None):
        env = dict(os.environ)
        env.update(extra_env or {})
        return subprocess.run(
            [sys.executable, str(HARNESS),
             "--binary", str(make_fake_binary(td)),
             "--gens", str(gens), "--chunk", str(chunk),
             "--ckpt", str(td / "ck.bin")],
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
            self.assertNotIn("spontaneous stress flag", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

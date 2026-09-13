"""Tests for the long-run harness decision logic (I9).

The harness runs 10-hour jobs, so its logic is tested here against the real
failure fixtures: the gen-0 calibration flag, the false-PASS log where every
chunk resumed at generation 50, and the watchdog/progress command contract.
CPU-only.
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "long_run_check", ROOT / "tests" / "long_run_check.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class HarnessTests(unittest.TestCase):
    def test_chunk_command_carries_target_profile_and_resume(self):
        cmd = harness.chunk_command("build/coevo.exe", 100, "ck.bin", True)
        self.assertEqual(cmd[1], "100")
        self.assertIn("--profile", cmd)
        self.assertIn("--resume", cmd)
        fresh = harness.chunk_command("build/coevo.exe", 50, "ck.bin", False)
        self.assertNotIn("--resume", fresh)

    def test_checkpoint_generation_parsing(self):
        out = ("Resumed from checkpoints/x.bin at generation 50\n"
               "=== Run complete: 50 generations ===\n"
               "Checkpoint: checkpoints/x.bin (generation 50)\n")
        self.assertEqual(harness.parse_checkpoint_generation(out), 50)
        self.assertIsNone(harness.parse_checkpoint_generation("no lines"))

    def test_false_pass_fixture_is_rejected(self):
        # The real invalid run: target 100 but the checkpoint stayed at 50.
        out = ("Resumed from checkpoints/long-run-5000.bin at generation 50\n"
               "=== Run complete: 50 generations ===\n"
               "Checkpoint: checkpoints/long-run-5000.bin (generation 50)\n")
        target = 100
        self.assertNotEqual(harness.parse_checkpoint_generation(out), target)

    def test_warmup_fixture_ignores_the_gen_zero_flag(self):
        flag = ("[STRESS] lineage 0 flagged: 60% SOT-gate failures over the "
                "last 10 stress evaluations (operator review)\n")
        self.assertFalse(harness.flag_is_spontaneous(0, 50, flag))
        self.assertTrue(harness.flag_is_spontaneous(50, 50, flag))
        self.assertFalse(harness.flag_is_spontaneous(50, 50, "clean output"))

    def test_r_samples_counts_post_warmup_dashboard_lines(self):
        out = ("[DASHBOARD] role_frac_C=1.0 r=0.6000 rho=0.5\n"
               "[DASHBOARD] role_frac_C=0.5 r=0.4000 rho=0.5\n"
               "surprise s_ph=1.0 s_pr=1.0 r=0.9000\n")
        ok, total = harness.r_samples(out, 0.5)
        self.assertEqual((ok, total), (1, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)

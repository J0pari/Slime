"""GPU scheduler client tests: contract pin, submission, waiting, result
ledger provenance, progress/v1 wrapping, and loud failure on mismatch.

The tests build a fake training-architecture root in a temp directory and
point TRAINING_ARCH_ROOT at it; the real Slime pin file is validated
against the canonical fingerprint. No absolute paths anywhere.

Run from the repository root:

    python -m unittest discover -s tests/architecture -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "architecture"))

import gpu_client  # noqa: E402
import progress_wrap  # noqa: E402
from architecture import evidence  # noqa: E402

CANONICAL_FP = "5d602cc2574cc3c26c8153d5bf4cfa09066f6ec902fb03aef0747792a509c690"

FAKE_SCHEDULER = '''\
import json, os, sys
state_path = os.environ["FAKE_SCHED_STATE"]
argv_path = os.environ.get("FAKE_ARGV_PATH", "")
mode = sys.argv[1]
if mode == "contract":
    print(json.dumps({"schema": "gpu-scheduler/v1", "owner": "training-architecture",
                      "contractVersion": "1"}))
elif mode == "status":
    print(json.dumps({"running": None, "queue": []}))
elif mode == "submit":
    if argv_path:
        with open(argv_path, "w", encoding="utf-8") as f:
            json.dump(sys.argv, f)
    print(json.dumps({"jobId": "0123456789abcdef", "status": "queued"}))
elif mode == "inspect":
    with open(state_path, "r", encoding="utf-8") as f:
        status = f.read().strip() or "queued"
    print(json.dumps({"jobId": "0123456789abcdef", "name": "fake", "status": status,
                      "attempt": 1, "exitCode": 0 if status == "done" else None,
                      "submittedAt": "2026-09-12T00:00:00Z",
                      "startedAt": "2026-09-12T00:00:01Z",
                      "finishedAt": "2026-09-12T00:00:02Z" if status == "done" else None,
                      "retryAfter": None, "telemetry": None}))
'''

FAKE_HANDOFF = '''\
import os
def fingerprint(manifest):
    override = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fake_fp.txt")
    if os.path.isfile(override):
        with open(override, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "%s"
''' % CANONICAL_FP

FAKE_LOCK = '''\
def state():
    return {"state": "free"}
def acquire(role, jobId=None):
    return {"acquired": True, "role": role}
def release():
    return None
'''


FAKE_CLIENT = '''\
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cli(*args):
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "control", "gpu_scheduler.py"),
         *args], capture_output=True, text=True)
    return json.loads(out.stdout)


def status():
    return _cli("status")


def inspect(job_id=None):
    return _cli("inspect", "--job", job_id)
'''


class FakeScheduler:
    """A temp training-architecture root with a stub scheduler."""

    def __init__(self, td: str):
        self.root = Path(td)
        (self.root / "control").mkdir(parents=True)
        (self.root / "control" / "gpu_scheduler.py").write_text(
            FAKE_SCHEDULER, encoding="utf-8")
        (self.root / "control" / "client.py").write_text(
            FAKE_CLIENT, encoding="utf-8")
        (self.root / "control" / "gpu_lock.py").write_text(
            FAKE_LOCK, encoding="utf-8")
        (self.root / "control" / "__init__.py").write_text("", encoding="utf-8")
        self.sched_dir = self.root / "artifacts" / "gpu-scheduler"
        (self.sched_dir / "results").mkdir(parents=True)
        self.state_path = self.root / "job-state.txt"
        self.state_path.write_text("running", encoding="utf-8")
        self.argv_path = self.root / "argv.json"

    def env(self, extra=None):
        env = dict(os.environ)
        env["TRAINING_ARCH_ROOT"] = str(self.root)
        env["KF_GPU_SCHED_DIR"] = str(self.sched_dir)
        env["FAKE_SCHED_STATE"] = str(self.state_path)
        env["FAKE_ARGV_PATH"] = str(self.argv_path)
        env.update(extra or {})
        # The control API client resolves the daemon from process state, so
        # the fixture publishes the same values for in-process clients and
        # their children.
        os.environ.update(env)
        return env

    def set_job_state(self, status):
        self.state_path.write_text(status, encoding="utf-8")

    def write_result(self, status="done", exit_code=0, duration=12.5):
        rec = {"jobId": "0123456789abcdef", "name": "fake", "repo":
               "slime-evolution", "status": status, "exitCode": exit_code,
               "reason": "", "startedAt": "2026-09-12T00:00:01Z",
               "finishedAt": "2026-09-12T00:00:13Z",
               "durationSec": duration, "telemetry": None, "failure": None,
               "logFile": str(self.sched_dir / "logs" / "fake.log")}
        path = self.sched_dir / "results" / "0123456789abcdef.json"
        path.write_text(json.dumps(rec), encoding="utf-8")
        return path


class PinTests(unittest.TestCase):
    def test_pin_file_is_canonical(self):
        pin = gpu_client.load_pin()
        self.assertEqual(pin["schema"], "gpu-scheduler/v1")
        self.assertEqual(len(pin["fingerprint"]), 64)
        self.assertTrue(all(c in pin["fingerprint"] for c in "0123456789abcdef"))
        self.assertIn("progress/v1", pin["capabilities"])
        self.assertIn("gpu-lock/v1", pin["capabilities"])


class ClientTests(unittest.TestCase):
    def test_missing_root_is_loud(self):
        env = {k: v for k, v in os.environ.items()
               if k != gpu_client.ENV_ROOT}
        env.pop(gpu_client.ENV_SCHED_DIR, None)
        with self.assertRaises(gpu_client.SchedulerUnavailable) as ctx:
            gpu_client.scheduler_root(env)
        self.assertIn(gpu_client.ENV_ROOT, str(ctx.exception))

    def test_contract_check_ok_and_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeScheduler(td)
            manifest = gpu_client.contract_manifest(env=fake.env())
            observed = gpu_client._fingerprint(manifest)
            with unittest.mock.patch.object(
                    gpu_client, "load_pin",
                    return_value={"schema": "gpu-scheduler/v1",
                                  "fingerprint": observed,
                                  "capabilities": ["submit"]}):
                checked = gpu_client.check_contract(env=fake.env())
            self.assertEqual(checked["schema"], "gpu-scheduler/v1")
            # A drifted pin must refuse loudly.
            with unittest.mock.patch.object(
                    gpu_client, "load_pin",
                    return_value={"schema": "gpu-scheduler/v1",
                                  "fingerprint": "0" * 64,
                                  "capabilities": ["submit"]}):
                with self.assertRaises(gpu_client.ContractMismatch):
                    gpu_client.check_contract(env=fake.env())

    def test_submit_builds_argv_and_parses_ack(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeScheduler(td)
            ack = gpu_client.submit(
                "slime-evolution-regression", ["build/evolution_test.exe"],
                2048, 4096, priority=5, max_minutes=30, env=fake.env())
            self.assertEqual(ack, {"jobId": "0123456789abcdef",
                                   "status": "queued"})
            argv = json.loads(fake.argv_path.read_text(encoding="utf-8"))
            self.assertIn("--name", argv)
            self.assertEqual(argv[argv.index("--name") + 1], "slime-evolution-regression")
            self.assertEqual(argv[argv.index("--repo") + 1], "slime-evolution")
            self.assertEqual(argv[argv.index("--vram") + 1], "2048")
            self.assertEqual(argv[argv.index("--ram") + 1], "4096")
            self.assertEqual(argv[argv.index("--cwd") + 1], str(ROOT))
            cmd_idx = argv.index("--cmd")
            resolved = argv[cmd_idx + 1:]
            self.assertEqual(len(resolved), 1)
            # the relative path is resolved against the job cwd for the
            # owner's absolute-executable validation
            self.assertTrue(os.path.isabs(resolved[0]))
            self.assertTrue(resolved[0].endswith("evolution_test.exe"))

    def test_bare_long_run_refused(self):
        # Resilience is enforced at the submission choke point: a bare
        # multi-generation run is refused; short runs and harness-wrapped
        # runs are allowed.
        with tempfile.TemporaryDirectory() as td:
            fake = FakeScheduler(td)
            with self.assertRaises(ValueError):
                gpu_client.submit("x", ["build/coevo.exe", "100"], 2048, 4096,
                                  env=fake.env())
            gpu_client.submit("x", ["build/coevo.exe", "1"], 2048, 4096,
                              env=fake.env())
            gpu_client.submit("x", ["python", "tests/long_run_check.py",
                                    "--gens", "5000"], 2048, 4096,
                              env=fake.env())

    def test_submit_requires_ram_and_scratch_disk(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeScheduler(td)
            with self.assertRaises(ValueError):
                gpu_client.submit("x", ["build/evolution_test.exe"],
                                  2048, 0, env=fake.env())
            with self.assertRaises(ValueError):
                gpu_client.submit("x", ["build/evolution_test.exe"],
                                  2048, 4096, job_kind="scratch",
                                  env=fake.env())
            ack = gpu_client.submit("x", ["build/evolution_test.exe"],
                                    2048, 4096, job_kind="scratch",
                                    disk_mib=8192, env=fake.env())
            argv = json.loads(fake.argv_path.read_text(encoding="utf-8"))
            self.assertEqual(argv[argv.index("--disk") + 1], "8192")
            self.assertEqual(ack["status"], "queued")

    def test_wait_and_result_ledger_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeScheduler(td)
            job = gpu_client.inspect("0123456789abcdef", env=fake.env())
            self.assertEqual(job["status"], "running")
            fake.set_job_state("done")
            final = gpu_client.wait("0123456789abcdef", poll_seconds=0.01,
                                    env=fake.env())
            self.assertEqual(final["status"], "done")
            fake.write_result()
            prov = gpu_client.scheduler_provenance("0123456789abcdef",
                                                   env=fake.env())
            self.assertEqual(prov["jobId"], "0123456789abcdef")
            self.assertEqual(prov["status"], "done")
            self.assertEqual(prov["durationSec"], 12.5)
            self.assertEqual(prov["contractFingerprint"], CANONICAL_FP)

    def test_wrapped_command_has_no_bare_double_dash(self):
        # The owner's submit parser is nargs=REMAINDER: argparse terminates
        # it at a bare "--", so a wrapped command must not contain one.
        wrapped = gpu_client.wrap_progress_command(
            "run10", 10, ["build/coevo.exe", "10"])
        self.assertNotIn("--", wrapped)
        self.assertIn("progress_wrap.py", wrapped[1])
        self.assertEqual(wrapped[-2:], ["build/coevo.exe", "10"])

    def test_evidence_manifest_links_scheduler_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeScheduler(td)
            fake.write_result()
            saved = {k: os.environ.get(k) for k in
                     ("TRAINING_ARCH_ROOT", "KF_GPU_SCHED_DIR")}
            try:
                os.environ["TRAINING_ARCH_ROOT"] = str(fake.root)
                os.environ["KF_GPU_SCHED_DIR"] = str(fake.sched_dir)
                manifest = evidence.new_manifest(
                    name="t", root=ROOT, binary="none", cuda="x", gpu="y",
                    seed="s", results={}, claims=[],
                    machine={"schema": "slime-machine/v1"},
                    scheduler_job="0123456789abcdef")
                self.assertEqual(manifest["scheduler"]["jobId"],
                                 "0123456789abcdef")
                self.assertEqual(manifest["scheduler"]["status"], "done")
                self.assertEqual(manifest["scheduler"]["contractFingerprint"],
                                 CANONICAL_FP)
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class ProgressWrapTests(unittest.TestCase):
    def test_envelope_validator(self):
        good = progress_wrap.make_envelope("train", 1, 10, {})
        self.assertTrue(progress_wrap.envelope_valid(good))
        weak = dict(good)
        del weak["extras"]
        self.assertFalse(progress_wrap.envelope_valid(weak))
        wrong_schema = dict(good, schema="progress/v0")
        self.assertFalse(progress_wrap.envelope_valid(wrong_schema))
        out_of_range = dict(good, done=11, total=10)
        self.assertFalse(progress_wrap.envelope_valid(out_of_range))

    def test_wrapper_emits_valid_progress(self):
        with tempfile.TemporaryDirectory() as td:
            child = Path(td) / "child.py"
            child.write_text(
                "import sys\n"
                "print('step_generation(0) begin')\n"
                "print('gen 0  mean_fitness=0.05')\n"
                "print('step_generation(1) begin')\n"
                "print('gen 1  mean_fitness=0.06')\n"
                "sys.exit(0)\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "architecture" / "progress_wrap.py"),
                 "--name", "test", "--total", "2", "--min-interval", "0", "--",
                 sys.executable, str(child)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("step_generation(0) begin", proc.stdout)
            envelopes = []
            for line in proc.stdout.splitlines():
                if line.startswith("PROGRESS "):
                    env = json.loads(line[len("PROGRESS "):])
                    self.assertTrue(progress_wrap.envelope_valid(env))
                    envelopes.append(env)
            self.assertGreaterEqual(len(envelopes), 4)
            self.assertTrue(any(e["phase"] == "training" for e in envelopes))
            self.assertEqual(envelopes[-1]["phase"], "complete")
            self.assertEqual(envelopes[-1]["done"], 2)
            self.assertEqual(envelopes[-1]["total"], 2)
            self.assertEqual(envelopes[-1]["frac"], 1.0)

    def test_wrapper_propagates_failure(self):
        with tempfile.TemporaryDirectory() as td:
            child = Path(td) / "child.py"
            child.write_text("import sys; sys.exit(3)\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "architecture" / "progress_wrap.py"),
                 "--name", "test", "--total", "1", "--",
                 sys.executable, str(child)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 3)
            self.assertTrue(any(line.startswith("PROGRESS ")
                                for line in proc.stdout.splitlines()))


if __name__ == "__main__":
    unittest.main(verbosity=2)

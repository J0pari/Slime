"""GPU scheduler consumer client (gpu-scheduler/v1).

Slime enters the GPU through the training-architecture scheduler: one
exclusive arbiter for all local repos. This module is the consumer side of
the pinned contract (contracts/gpu-scheduler-pin.json):

  - contract check before any submission (schema major + fingerprint);
  - submit / inspect / status / wait against the daemon;
  - the job result ledger (results/<jobId>.json) for evidence provenance;
  - gpu-lock/v1 acquisition for MANUAL launches (scheduled jobs are covered
    by the scheduler's ownership).

Path discipline: the training-architecture location is discovered from the
environment only. No absolute path is embedded in this file.

    TRAINING_ARCH_ROOT   the training-architecture repo root
                         (the client uses <root>/src/gpu_scheduler.py)
    KF_GPU_SCHED_DIR     optional state-dir override, honored by the owner
                         (defaults to <root>/artifacts/gpu-scheduler)

stdlib-only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ENV_ROOT = "TRAINING_ARCH_ROOT"
ENV_SCHED_DIR = "KF_GPU_SCHED_DIR"
PIN_PATH = "contracts/gpu-scheduler-pin.json"
REPO_ROOT = Path(__file__).resolve().parent.parent


class SchedulerUnavailable(RuntimeError):
    """The scheduler root is not configured or not reachable."""


class ContractMismatch(RuntimeError):
    """The observed scheduler contract disagrees with the pin."""


def _env(env: dict | None = None) -> dict:
    return dict(os.environ if env is None else env)


def scheduler_root(env: dict | None = None) -> Path:
    root = _env(env).get(ENV_ROOT, "").strip()
    if not root:
        raise SchedulerUnavailable(
            f"{ENV_ROOT} is not set; point it at the training-architecture "
            f"repo root to submit GPU work through the scheduler")
    path = Path(root)
    if not path.is_dir():
        raise SchedulerUnavailable(f"{ENV_ROOT} does not exist: {root}")
    return path


def scheduler_script(env: dict | None = None) -> Path:
    script = scheduler_root(env) / "src" / "gpu_scheduler.py"
    if not script.is_file():
        raise SchedulerUnavailable(f"scheduler script missing: {script}")
    return script


def sched_dir(env: dict | None = None) -> Path:
    """The scheduler state dir: KF_GPU_SCHED_DIR, else <root>/artifacts/gpu-scheduler."""
    override = _env(env).get(ENV_SCHED_DIR, "").strip()
    if override:
        return Path(override)
    return scheduler_root(env) / "artifacts" / "gpu-scheduler"


def load_pin() -> dict:
    pin_file = REPO_ROOT / PIN_PATH
    if not pin_file.is_file():
        raise ContractMismatch(f"consumer pin missing: {PIN_PATH}")
    with open(pin_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_cli(args: list[str], env: dict | None = None,
             timeout: float = 120.0) -> dict:
    """Run the owner CLI and parse its JSON stdout."""
    script = scheduler_script(env)
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, timeout=timeout,
        env=_env(env), cwd=str(scheduler_root(env)))
    if proc.returncode != 0:
        raise SchedulerUnavailable(
            f"scheduler {args[0]} failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()[:400]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SchedulerUnavailable(
            f"scheduler {args[0]} returned non-JSON output: "
            f"{proc.stdout[:200]!r}") from exc


def contract_manifest(env: dict | None = None) -> dict:
    return _run_cli(["contract", "--json"], env=env)


def _fingerprint(manifest: dict, env: dict | None = None) -> str:
    """The owner's canonical fingerprint algorithm (src/handoff.py).

    Recomputing it from the returned manifest is the contract's own
    instruction; the algorithm has exactly one authority and we import it
    rather than duplicating it."""
    root = scheduler_root(env)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.handoff import fingerprint  # type: ignore
    except Exception as exc:  # pragma: no cover - only on a broken owner
        raise ContractMismatch(
            f"cannot import the owner fingerprint algorithm from {root}: {exc}"
        ) from exc
    return fingerprint(manifest)


def check_contract(env: dict | None = None) -> dict:
    """Validate the observed contract against the pin. Loud refusal on a
    schema-major or fingerprint mismatch (the contract's compatibility rule)."""
    pin = load_pin()
    manifest = contract_manifest(env)
    observed_schema = manifest.get("schema")
    if observed_schema != pin["schema"]:
        raise ContractMismatch(
            f"gpu-scheduler schema mismatch: pinned {pin['schema']!r}, "
            f"observed {observed_schema!r}")
    observed_fp = _fingerprint(manifest, env)
    if observed_fp != pin["fingerprint"]:
        raise ContractMismatch(
            f"gpu-scheduler fingerprint mismatch: pinned "
            f"{pin['fingerprint'][:16]}..., observed {observed_fp[:16]}... "
            f"(the contract drifted; re-pin after review)")
    return manifest


def status(env: dict | None = None) -> dict:
    return _run_cli(["status"], env=env)


def submit(name: str, command: list[str], vram_mib: int,
           priority: int = 0, max_minutes: float = 0.0, cwd: str = "",
           job_env: dict | None = None, telemetry_log: str = "",
           allow_ollama: bool = False, retries: int | None = None,
           env: dict | None = None) -> dict:
    """Submit a job; returns the SubmitAck {jobId, status}.

    The owner validates that command[0] is an absolute existing file or on
    PATH; relative paths are resolved against the job cwd (default: this
    repo root) so callers can pass build/foo.exe naturally."""
    if not command:
        raise ValueError("command must be non-empty")
    work_dir = cwd or str(REPO_ROOT)
    first = command[0]
    if not os.path.isabs(first):
        # Always resolve against the job cwd: the owner validates that the
        # executable exists, and a relative path would otherwise be checked
        # against the owner's working directory, not the job's.
        command = [str((Path(work_dir) / first).resolve()), *command[1:]]
    args = ["submit", "--name", name, "--repo", "slime-evolution",
            "--vram", str(int(vram_mib)), "--priority", str(int(priority)),
            "--max-minutes", str(float(max_minutes))]
    args += ["--cwd", work_dir]
    if telemetry_log:
        args += ["--telemetry-log", telemetry_log]
    if allow_ollama:
        args += ["--allow-ollama"]
    if retries is not None:
        args += ["--retries", str(int(retries))]
    for key, value in sorted((job_env or {}).items()):
        args += ["--env", f"{key}={value}"]
    args += ["--cmd", *command]
    return _run_cli(args, env=env)


def inspect(job_id: str, env: dict | None = None) -> dict:
    return _run_cli(["inspect", "--job", job_id], env=env)


def wait(job_id: str, poll_seconds: float = 10.0, timeout: float = 0.0,
         env: dict | None = None, on_status=None) -> dict:
    """Poll inspect until the job reaches a terminal status. Returns the
    final JobStatus. timeout=0 waits forever."""
    started = time.monotonic()
    last = None
    while True:
        job = inspect(job_id, env=env)
        state = job.get("status")
        if state != last:
            if on_status is not None:
                on_status(job)
            last = state
        if state in ("done", "failed", "cancelled"):
            return job
        if timeout and (time.monotonic() - started) > timeout:
            raise TimeoutError(
                f"job {job_id} still {state!r} after {timeout:.0f}s")
        time.sleep(poll_seconds)


def result_path(job_id: str, env: dict | None = None) -> Path:
    return sched_dir(env) / "results" / f"{job_id}.json"


def load_result(job_id: str, env: dict | None = None) -> dict | None:
    """The scheduler's JobResult record for a finished job, or None."""
    path = result_path(job_id, env)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scheduler_provenance(job_id: str, env: dict | None = None) -> dict:
    """The evidence-provenance block for a scheduled job: links Slime's
    manifests to the scheduler's result ledger."""
    pin = load_pin()
    result = load_result(job_id, env)
    block = {"jobId": job_id, "contractSchema": pin["schema"],
             "contractFingerprint": pin["fingerprint"]}
    if result is not None:
        block.update({
            "status": result.get("status"),
            "exitCode": result.get("exitCode"),
            "durationSec": result.get("durationSec"),
            "logFile": result.get("logFile"),
        })
    return block


# ---- Manual-launch GPU lock (gpu-lock/v1) --------------------------------
def _gpu_lock_module(env: dict | None = None):
    root = scheduler_root(env)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src import gpu_lock  # type: ignore
    except Exception as exc:
        raise SchedulerUnavailable(
            f"cannot import the owner gpu-lock module from {root}: {exc}"
        ) from exc
    return gpu_lock


def lock_state(env: dict | None = None) -> dict:
    return _gpu_lock_module(env).state()


def acquire_gpu_lock(role: str = "slime-manual", job_id: str = "",
                     env: dict | None = None) -> dict:
    """Acquire the machine-wide GPU lock for a MANUAL launch. Scheduled jobs
    are covered by the scheduler's ownership and must not call this."""
    got = _gpu_lock_module(env).acquire(role, jobId=job_id or None)
    if not got.get("acquired"):
        raise SchedulerUnavailable(
            f"GPU lock refused: {got.get('reason') or got.get('state')}")
    return got


def release_gpu_lock(env: dict | None = None) -> None:
    _gpu_lock_module(env).release()


# ---- CLI -----------------------------------------------------------------
def wrap_progress_command(name: str, total: int,
                          command: list[str]) -> list[str]:
    """Wrap a command in progress_wrap.py. Deliberately NO "--" separator:
    the owner's submit parser uses nargs=REMAINDER, which argparse
    terminates at a bare "--" (the job would fail submission)."""
    wrapper = Path(__file__).resolve().parent / "progress_wrap.py"
    return [sys.executable, str(wrapper), "--name", name,
            "--total", str(total), *command]


def _cmd_run(args) -> int:
    """Submit a job through the scheduler and exit. The daemon owns the job;
    --wait opts into blocking until it reaches a terminal status (never the
    default: a foreground wait defeats the daemon and ties up the caller).
    --direct is the explicit escape hatch for machines without the
    scheduler; it executes locally (under the GPU lock when reachable)."""
    command = list(args.cmd)
    if not command:
        print("run: no command given (use -- <cmd...>)", file=sys.stderr)
        return 2

    if args.direct:
        return _run_direct(command, args)

    try:
        check_contract()
        if args.total > 0:
            command = wrap_progress_command(args.name, args.total, command)
        ack = submit(args.name, command, args.vram, priority=args.priority,
                     max_minutes=args.max_minutes, cwd=args.cwd,
                     telemetry_log=args.telemetry_log)
    except (SchedulerUnavailable, ContractMismatch) as exc:
        print(f"[gpu-client] scheduler unavailable: {exc}", file=sys.stderr)
        print("[gpu-client] re-run with --direct to execute locally, or set "
              f"{ENV_ROOT}", file=sys.stderr)
        return 2

    job_id = ack["jobId"]
    print(f"[gpu-client] submitted {args.name}: {job_id}")
    if not args.wait:
        print(f"[gpu-client] detached; poll with: "
              f"python architecture/gpu_client.py inspect --job {job_id}")
        return 0

    def _on_status(job):
        print(f"[gpu-client] {job_id}: {job['status']}"
              + (f" (attempt {job.get('attempt')})" if job.get("attempt") else ""))

    job = wait(job_id, poll_seconds=args.poll, env=None, on_status=_on_status)
    result = load_result(job_id) or {}
    print(json.dumps({"jobId": job_id, "status": job["status"],
                      "exitCode": job.get("exitCode"),
                      "durationSec": result.get("durationSec"),
                      "logFile": result.get("logFile")}, indent=2))
    return 0 if job["status"] == "done" else 1


def _run_direct(command: list[str], args) -> int:
    """Direct execution (explicit opt-in): manual GPU launches acquire the
    lock when the owner module is reachable; a refusal is loud."""
    held = False
    try:
        acquire_gpu_lock(role="slime-direct", env=None)
        held = True
        print("[gpu-client] direct mode: gpu lock acquired")
    except (SchedulerUnavailable, ContractMismatch) as exc:
        print(f"[gpu-client] direct mode without gpu lock: {exc}",
              file=sys.stderr)
    try:
        proc = subprocess.run(command, cwd=args.cwd or None)
        return proc.returncode
    finally:
        if held:
            try:
                release_gpu_lock()
            except Exception:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Slime GPU scheduler client")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("contract", help="validate the pinned contract")
    sub.add_parser("status", help="daemon status")
    p_pin = sub.add_parser("pin", help="print the consumer pin")
    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("--job", required=True)
    p_result = sub.add_parser("result")
    p_result.add_argument("--job", required=True)
    sub.add_parser("lock-status")

    p_run = sub.add_parser("run", help="submit + wait (or --direct)")
    p_run.add_argument("--name", required=True)
    p_run.add_argument("--vram", type=int, required=True)
    p_run.add_argument("--priority", type=int, default=0)
    p_run.add_argument("--max-minutes", type=float, default=0.0)
    p_run.add_argument("--cwd", default="")
    p_run.add_argument("--telemetry-log", default="")
    p_run.add_argument("--total", type=int, default=0,
                       help="generation/step total for progress/v1 wrapping")
    p_run.add_argument("--poll", type=float, default=10.0)
    p_run.add_argument("--wait", action="store_true",
                       help="block until the job finishes (default: submit "
                            "and exit; the daemon owns the job)")
    p_run.add_argument("--direct", action="store_true",
                       help="execute locally instead of submitting")
    p_run.add_argument("cmd", nargs=argparse.REMAINDER)

    args = ap.parse_args(argv)
    try:
        if args.command == "contract":
            manifest = check_contract()
            print(json.dumps({"schema": manifest.get("schema"),
                              "validated": True}, indent=2))
            return 0
        if args.command == "status":
            print(json.dumps(status(), indent=2))
            return 0
        if args.command == "pin":
            print(json.dumps(load_pin(), indent=2))
            return 0
        if args.command == "inspect":
            print(json.dumps(inspect(args.job), indent=2))
            return 0
        if args.command == "result":
            print(json.dumps(load_result(args.job), indent=2))
            return 0
        if args.command == "lock-status":
            print(json.dumps(lock_state(), indent=2))
            return 0
        if args.command == "run":
            # argparse REMAINDER: the command follows "--"; drop the marker.
            cmd = list(args.cmd)
            if cmd and cmd[0] == "--":
                cmd = cmd[1:]
            args.cmd = cmd
            return _cmd_run(args)
    except (SchedulerUnavailable, ContractMismatch) as exc:
        print(f"[gpu-client] {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""progress/v1 wrapper for scheduled Slime jobs.

Runs a child command, tees its output, and emits progress/v1 envelopes
(the cross-repo contract, contracts/gpu-scheduler-v1.json) so the
training-architecture daemon can estimate the job's progress and ETA.

    python architecture/progress_wrap.py --name run10 --total 10 -- \
        build/coevo.exe 10

Parsing:
  - Slime's own telemetry lines drive the counter:
      "step_generation(N) begin"  -> phase detail
      "gen N  mean_fitness=..."   -> done = N + 1
  - A child line that is itself a PROGRESS envelope is passed through if it
    validates as progress/v1, otherwise ignored.
Envelopes are emitted on counter change (throttled) and once at exit.
The child's exit code is propagated.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

GEN_LINE = re.compile(r"^gen\s+(\d+)\b")
STEP_LINE = re.compile(r"^step_generation\((\d+)\)")
ENVELOPE_REQUIRED = ("schema", "event", "ts", "phase", "done", "total",
                     "frac", "extras")
EMIT_MIN_INTERVAL = 1.0


def envelope_valid(env: dict) -> bool:
    """The progress/v1 consumer contract (mirrors the owner validator)."""
    if not isinstance(env, dict):
        return False
    if env.get("schema") != "progress/v1" or env.get("event") != "progress":
        return False
    if not isinstance(env.get("ts"), (int, float)):
        return False
    if not isinstance(env.get("phase"), str):
        return False
    for key in ("done", "total"):
        if not isinstance(env.get(key), (int, float)):
            return False
    total = env.get("total", 0)
    if total and not (0 <= env.get("done", 0) <= total):
        return False
    frac = env.get("frac")
    if frac is not None and not isinstance(frac, (int, float)):
        return False
    if not isinstance(env.get("extras"), dict):
        return False
    for key in ENVELOPE_REQUIRED:
        if key not in env:
            return False
    return True


def make_envelope(phase: str, done: float, total: float, extras: dict) -> dict:
    frac = (done / total) if total > 0 else None
    return {"schema": "progress/v1", "event": "progress",
            "ts": round(time.time(), 3), "phase": phase,
            "done": done, "total": total, "frac": frac,
            "rateNow": None, "rateEma": None, "rateTrend": None,
            "etaSec": None, "etaLowSec": None, "etaHighSec": None,
            "extras": extras}


def run_child(name: str, total: int, command: list[str],
              min_interval: float = EMIT_MIN_INTERVAL) -> int:
    child = subprocess.Popen(command, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             bufsize=1)
    done = 0.0
    phase = "starting"
    last_emit = 0.0

    def emit(force: bool = False) -> None:
        nonlocal last_emit
        now = time.monotonic()
        if not force and (now - last_emit) < min_interval:
            return
        last_emit = now
        env = make_envelope(phase, done, float(total),
                            {"repo": "slime-evolution", "job": name})
        print("PROGRESS " + json.dumps(env), flush=True)

    emit(force=True)
    assert child.stdout is not None
    for line in child.stdout:
        print(line, end="", flush=True)
        stripped = line.strip()
        if stripped.startswith("PROGRESS "):
            try:
                env = json.loads(stripped[len("PROGRESS "):])
            except (ValueError, TypeError):
                env = None
            if envelope_valid(env):
                # pass-through: the child speaks the contract itself
                done = max(done, float(env.get("done", done)))
                total = max(total, int(env.get("total", total)))
                phase = str(env.get("phase", phase))
            continue
        m = STEP_LINE.match(stripped)
        if m:
            phase = f"generation {m.group(1)}"
            emit()
            continue
        m = GEN_LINE.match(stripped)
        if m:
            done = float(int(m.group(1)) + 1)
            phase = "training"
            emit()
    code = child.wait()
    phase = "complete" if code == 0 else "failed"
    done = float(total) if code == 0 else done
    emit(force=True)
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="progress/v1 wrapper")
    ap.add_argument("--name", default="slime-job")
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--min-interval", type=float, default=EMIT_MIN_INTERVAL,
                    help="minimum seconds between non-final envelopes")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("progress_wrap: no command given (use -- <cmd...>)",
              file=sys.stderr)
        return 2
    return run_child(args.name, args.total, cmd,
                     min_interval=args.min_interval)


if __name__ == "__main__":
    raise SystemExit(main())

"""Fake evolution binary for the long-run harness end-to-end tests.

Mimics the real binary's CLI contract: N is the cumulative target generation
(it resumes from the checkpoint's generation), --profile emits per-generation
`gen N` lines, and the final line reports the checkpoint's generation. With
FAKE_STUCK=1 it ignores the target and never advances, which is the
false-PASS condition the harness must reject. CPU-only.
"""
import os
import sys
from pathlib import Path

target = int(sys.argv[1])
resume = "--resume" in sys.argv
ckpt = None
for i, a in enumerate(sys.argv):
    if a == "--ckpt":
        ckpt = Path(sys.argv[i + 1])
state = Path(str(ckpt) + ".gen")
gen = int(state.read_text()) if (resume and state.is_file()) else 0
stuck = os.environ.get("FAKE_STUCK") == "1"

if resume:
    print(f"Resumed from {ckpt} at generation {gen} (archive size 1)")
else:
    print("World initialized: 1 pool organism")
for g in range(gen, target):
    print(f"gen {g}: forward+descriptor+btraj")
    if g == 0:
        print("[STRESS] lineage 0 flagged: 60% SOT-gate failures over the "
              "last 10 stress evaluations (operator review; no automatic "
              "pruning)")
    print("[DASHBOARD] role_frac_C=1.000 role_frac_P=0.000 r=0.6000 "
          "rho=0.0000 swaps=0/0 stress_flagged=1 archive=64")
new_gen = gen if stuck else target
state.write_text(str(new_gen), encoding="utf-8")
ckpt.write_bytes(b"fake-checkpoint")
print(f"=== Run complete: {target - gen} generations ===")
print(f"Checkpoint: {ckpt} (generation {new_gen})")

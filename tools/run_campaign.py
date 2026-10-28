"""Run the mission scenarios, launch to splashdown, all four tubules each (results/mission/<scenario>).

    python tools/run_campaign.py [--parallel 2] [scenario ...]       (default: every scenario below, in this order)

Each scenario runs `python -m dolle_sim mission configs/mission_quarter_deck.json --scenario <s> --jobs 4`, logs to
results/mission_<s>.log and appends '<s> exit=<code> <seconds> s' to results/mission_scenarios_done.txt.

Scenarios that are identical to another up to some time continue from that one's checkpoint instead of repeating
the same flight (engine.branch verifies that they are identical up to then, and the result is bit-for-bit the run
from lift-off; if it cannot be verified the scenario runs from lift-off):
  kick, no_release, strong_shaker    from nominal at T+89.9 s (the releases start at T+90 s)
  kick+spin_1hz                      from nominal at T+54 s (the residual spin starts with the de-spin)
  rough_grains+hang_nose_up/+shock_oblique   from rough_grains at T+459 s (they differ from the parachute on)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["nominal", "launch_vibration", "kick", "no_release", "strong_shaker", "kick+spin_1hz", "no_drag",
         "kick+lying", "kick+gas_100pa", "kick+gas_1atm", "rough_grains", "rough_grains+hang_nose_up",
         "rough_grains+shock_oblique"]
# Weakly bound dust rolls from lift-off, and every pair of rollers that sticks is a rolling cluster that only the
# full model can integrate: with 800 grains per tubule ~150 grains are in the full model at once (~4 h of
# computing per second of flight on a laptop), so these scenarios use 100 grains per tubule.
GRAINS = {"rough_grains": 100, "rough_grains+hang_nose_up": 100, "rough_grains+shock_oblique": 100}
CHECKPOINTS = {"nominal": [54.0, 89.9], "rough_grains": [459.0]}
BRANCH = {"kick": ("nominal", 89.9), "no_release": ("nominal", 89.9), "strong_shaker": ("nominal", 89.9),
          "kick+spin_1hz": ("nominal", 54.0), "rough_grains+hang_nose_up": ("rough_grains", 459.0),
          "rough_grains+shock_oblique": ("rough_grains", 459.0)}
TUBULES = 4


def _wait_for(src, T):
    d = ROOT / "results" / "mission" / src / "checkpoints"
    while len(list(d.glob(f"*_T{T:08.3f}.pkl"))) < TUBULES:
        time.sleep(30)
    return d


def run(sc):
    arg = [] if sc == "nominal" else ["--scenario", sc]
    if sc in CHECKPOINTS:
        arg += ["--checkpoint-at", *map(str, CHECKPOINTS[sc])]
    if sc in GRAINS:
        arg += ["--grains", str(GRAINS[sc])]
    if sc in BRANCH:
        src, T = BRANCH[sc]
        arg += ["--resume-from", str(_wait_for(src, T).relative_to(ROOT)), "--resume-at", str(T)]
    log = ROOT / "results" / f"mission_{sc}.log"
    env = dict(os.environ, NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS", "2"))
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.call([sys.executable, "-u", "-m", "dolle_sim", "mission", "configs/mission_quarter_deck.json",
                                *arg, "--jobs", str(TUBULES)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    line = f"{sc} exit={code} {time.time() - t0:.0f} s"
    with open(ROOT / "results" / "mission_scenarios_done.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--parallel", type=int, default=2, help="scenarios at once (each runs its 4 tubules in parallel)")
    a = ap.parse_args()
    todo = a.scenarios or ORDER
    with ThreadPoolExecutor(a.parallel) as ex:
        codes = list(ex.map(run, todo))
    return max(codes) if codes else 0


if __name__ == "__main__":
    sys.exit(main())

"""Run the mission scenarios, launch to splashdown, all four tubules each (results/mission/<scenario>).

    python tools/run_campaign.py [--workers 8] [scenario ...]       (default: every scenario below, in this order)

Every tubule of every scenario is one job: `python -m dolle_sim mission configs/mission_quarter_deck.json
--scenario <s> --tubule <i>` (the same configuration and seed as a whole-scenario run, so the result is the same;
tested), logged to results/mission_<s>_T<i>.log. A pool of workers takes the next job that can start; when a
scenario's four tubules are done its outputs are written (`--assemble`, logged to results/mission_<s>.log) and
'<s> exit=<code> <seconds> s' is appended to results/mission_scenarios_done.txt. Scheduling tubules one by one keeps
every core busy (the tubules of one scenario take very different times).

Scenarios that are identical to another up to some time continue from that one's checkpoint instead of repeating
the same flight (engine.branch verifies that they are identical up to then, and the result is bit-for-bit the run
from lift-off; if it cannot be verified the tubule runs from lift-off). A branched tubule starts as soon as the
same tubule of its source has saved that checkpoint:
  kick, no_release, strong_shaker    from nominal at T+89.9 s (the releases start at T+90 s)
  kick+spin_1hz                      from nominal at T+54 s (the residual spin starts with the de-spin)
  rough_grains+hang_nose_up/+shock_oblique   from rough_grains at T+459 s (they differ from the parachute on)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = "configs/mission_quarter_deck.json"
ORDER = ["nominal", "launch_vibration", "rough_grains", "kick+spin_1hz", "kick", "no_release", "strong_shaker",
         "no_drag", "kick+lying", "kick+gas_100pa", "kick+gas_1atm", "rough_grains+hang_nose_up",
         "rough_grains+shock_oblique"]
# Weakly bound dust rolls from lift-off, and every pair of rollers that sticks is a rolling cluster that only the
# full model can integrate: with 800 grains per tubule ~150 grains are in the full model at once (~4 h of
# computing per second of flight on a laptop). The ground-test vibration level wakes every grain for the 17 s of
# both burns, at outer steps of 1/40 of the 2 kHz band (~10^4 s of computing per second of flight with 800).
# These scenarios use 100 grains per tubule.
GRAINS = {"rough_grains": 100, "rough_grains+hang_nose_up": 100, "rough_grains+shock_oblique": 100,
          "launch_vibration": 100}
CHECKPOINTS = {"nominal": [54.0, 89.9], "rough_grains": [459.0]}
BRANCH = {"kick": ("nominal", 89.9), "no_release": ("nominal", 89.9), "strong_shaker": ("nominal", 89.9),
          "kick+spin_1hz": ("nominal", 54.0), "rough_grains+hang_nose_up": ("rough_grains", 459.0),
          "rough_grains+shock_oblique": ("rough_grains", 459.0)}


def tubule_names(sc):
    from dolle_sim.mission import apply_scenario, tubule_chamber
    m = apply_scenario(json.loads((ROOT / CONFIG).read_text(encoding="utf-8")), None if sc == "nominal" else sc)
    return [tubule_chamber(m, t)["name"] for t in m["tubules"]]


def out_dir(sc):
    return ROOT / "results" / "mission" / sc


def base_args(sc):
    arg = [] if sc == "nominal" else ["--scenario", sc]
    if sc in GRAINS:
        arg += ["--grains", str(GRAINS[sc])]
    return arg


def ready(sc, i, names, codes):
    """A branched tubule can start once its source tubule has saved the checkpoint (or has ended without it: then
    it runs from lift-off)."""
    if sc not in BRANCH:
        return True
    src, T = BRANCH[sc]
    if src not in names:
        names[src] = tubule_names(src)
    have = (out_dir(src) / "checkpoints" / f"{names[src][i]}_T{T:08.3f}.pkl").exists()
    return have or i in codes.get(src, {}) or src not in codes


def start(sc, i):
    arg = base_args(sc) + ["--tubule", str(i)]
    if sc in CHECKPOINTS:
        arg += ["--checkpoint-at", *map(str, CHECKPOINTS[sc])]
    if sc in BRANCH:
        src, T = BRANCH[sc]
        arg += ["--resume-from", str((out_dir(src) / "checkpoints").relative_to(ROOT)), "--resume-at", str(T)]
    env = dict(os.environ, NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS", "2"))
    log = open(ROOT / "results" / f"mission_{sc}_T{i + 1}.log", "w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-u", "-m", "dolle_sim", "mission", CONFIG, *arg], cwd=ROOT, stdout=log,
                         stderr=subprocess.STDOUT, env=env)
    return p, log


def assemble(sc, t0, codes):
    code = max(codes)
    if code == 0:
        with open(ROOT / "results" / f"mission_{sc}.log", "w", encoding="utf-8") as f:
            code = subprocess.call([sys.executable, "-u", "-m", "dolle_sim", "mission", CONFIG, *base_args(sc),
                                    "--assemble"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    line = f"{sc} exit={code} {time.time() - t0:.0f} s"
    with open(ROOT / "results" / "mission_scenarios_done.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="tubule jobs at once")
    a = ap.parse_args()
    todo = a.scenarios or ORDER
    names = {sc: tubule_names(sc) for sc in todo}
    jobs = [(sc, i) for sc in todo for i in range(len(names[sc]))]
    running, codes, t0, final = {}, {sc: {} for sc in todo}, time.time(), []
    while jobs or running:
        for job, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                del running[job]
                sc, i = job
                codes[sc][i] = p.returncode
                print(f"{sc} T{i + 1} exit={p.returncode} {time.time() - t0:.0f} s", flush=True)
                if len(codes[sc]) == len(names[sc]):
                    final.append(assemble(sc, t0, list(codes[sc].values())))
        for job in list(jobs):
            if len(running) >= a.workers:
                break
            sc, i = job
            if ready(sc, i, names, codes):
                jobs.remove(job)
                running[job] = start(sc, i)
        time.sleep(10)
    return max(final) if final else 0


if __name__ == "__main__":
    sys.exit(main())

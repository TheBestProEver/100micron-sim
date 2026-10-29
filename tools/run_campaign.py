"""Run the mission scenarios, launch to splashdown, all four tubules each (results/mission/<scenario>).

    python tools/run_campaign.py [--workers 8] [scenario ...]       (default: every scenario below, in this order)

Every tubule of every scenario is one job: `python -m dolle_sim mission configs/mission_quarter_deck.json
--scenario <s> --grains 100 --tubule <i> --autosave-min 30` (the same configuration and seed as a whole-scenario run,
so the result is the same; tested), logged to results/mission_<s>_T<i>.log. A pool of workers takes the next job that
can start; when a scenario's four tubules are done its outputs are written (`--assemble`, logged to
results/mission_<s>.log) and '<s> exit=<code> <seconds> s' is appended to results/mission_scenarios_done.txt.
Scheduling tubules one by one keeps every core busy (the tubules of one scenario take very different times).

Every job saves its whole state every 30 minutes of computing (results/mission/<s>/autosave). A job that dies is
started again and continues from its last autosave, bit-for-bit the uninterrupted run (tested): after running out of
memory as often as it takes (once the machine has the memory again), after any other failure up to three times. A job
starts only when the machine can commit its memory. This script can be stopped and started again: finished tubules
(their final states in results/mission/<s>/tubules) are not run again, and jobs still running are adopted.

Scenarios that are identical to another up to some time continue from that one's checkpoint instead of repeating
the same flight (engine.branch verifies that they are identical up to then, and the result is bit-for-bit the run
from lift-off; if it cannot be verified the tubule runs from lift-off). A branched tubule starts as soon as the
same tubule of its source has saved that checkpoint:
  kick, no_release, strong_shaker    from nominal at T+89.9 s (the releases start at T+90 s)
  kick+spin_1hz                      from nominal at T+54 s (the residual spin starts with the de-spin)
  rough_grains+hang_nose_up/+shock_oblique   from rough_grains at T+459 s (they differ from the parachute on)

Dust amount: 100 grains per tubule in every scenario. The amount is not decided yet (the configuration's 800 is an
example); with the full contact model the dust that rolls along the walls in the coasts and under the spin (T+5 s to
the de-spin at T+54 s) costs ~10^4 s of computing per second of flight at 800 grains, weeks for the campaign on a
laptop, and 100 grains in every scenario keeps the scenarios comparable with one another.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/mission_quarter_deck.json"
ORDER = ["nominal", "launch_vibration", "rough_grains", "kick+spin_1hz", "kick", "no_release", "strong_shaker",
         "no_drag", "kick+lying", "kick+gas_100pa", "kick+gas_1atm", "rough_grains+hang_nose_up",
         "rough_grains+shock_oblique"]
GRAINS = 100                 # per tubule, every scenario (see above)
AUTOSAVE_MIN = 30
RETRIES = 3
JOB_MEMORY_GB = 1.5          # a job's peak commit (~0.7-1.1 GB measured) with a margin
CHECKPOINTS = {"nominal": [54.0, 89.9], "rough_grains": [459.0]}
BRANCH = {"kick": ("nominal", 89.9), "no_release": ("nominal", 89.9), "strong_shaker": ("nominal", 89.9),
          "kick+spin_1hz": ("nominal", 54.0), "rough_grains+hang_nose_up": ("rough_grains", 459.0),
          "rough_grains+shock_oblique": ("rough_grains", 459.0)}


_NAMES = {}


def tubule_names(sc):
    """The scenario's tubule names, computed in a child process (this script stays small: it imports none of the
    simulator, which would hold ~0.5 GB for the whole campaign)."""
    if sc not in _NAMES:
        code = ("import json, sys; from dolle_sim.mission import apply_scenario, tubule_chamber; "
                f"m = apply_scenario(json.loads(open({str(ROOT / CONFIG)!r}, encoding='utf-8').read()), "
                f"{None if sc == 'nominal' else sc!r}); print(json.dumps([tubule_chamber(m, t)['name'] "
                "for t in m['tubules']]))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        _NAMES[sc] = json.loads(out.stdout.strip().splitlines()[-1])
    return _NAMES[sc]


def out_dir(sc):
    return ROOT / "results" / "mission" / sc


def done_tubule(sc, name):
    return (out_dir(sc) / "tubules" / f"{name}.pkl").exists()        # dolle_sim.mission.final_state_path


def free_commit_gb():
    """Memory the system can still commit (Windows: available page file + RAM; elsewhere: available RAM)."""
    if os.name == "nt":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MS()
        s.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullAvailPageFile / 2 ** 30
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except ImportError:          # pragma: no cover
        return float("inf")


def out_of_memory(sc, i):
    """Did the tubule's last attempt die of memory exhaustion?"""
    try:
        with open(ROOT / "results" / f"mission_{sc}_T{i + 1}.log", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 3000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return False
    tail = tail.split("--- started again")[-1]
    # a MemoryError, or a hard crash (no Python traceback) while the machine is short of memory
    return "MemoryError" in tail or ("Traceback" not in tail and free_commit_gb() < 1.0)


class Adopted:
    """A tubule job left running by an earlier run of this script (not our child: its exit code is unknown, so
    success is its final state on disk)."""
    def __init__(self, proc, sc, name):
        self.proc, self.sc, self.name, self.returncode = proc, sc, name, None

    def poll(self):
        import psutil
        try:
            alive = self.proc.is_running() and self.proc.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            alive = False
        if alive:
            return None
        self.returncode = 0 if done_tubule(self.sc, self.name) else 1
        return self.returncode


def running_jobs(names):
    """Tubule jobs of this campaign that are already running: (scenario, tubule) -> Adopted."""
    import psutil
    found = {}
    for pr in psutil.process_iter(["cmdline"]):
        c = pr.info.get("cmdline") or []
        if "dolle_sim" not in c or "mission" not in c or "--tubule" not in c:
            continue
        sc = c[c.index("--scenario") + 1] if "--scenario" in c else "nominal"
        i = int(c[c.index("--tubule") + 1])
        if sc in names:
            found[(sc, i)] = Adopted(pr, sc, names[sc][i])
    return found


class _Null:
    def close(self):
        pass


def base_args(sc):
    arg = [] if sc == "nominal" else ["--scenario", sc]
    return arg + ["--grains", str(GRAINS)]


def ready(sc, i, names, final):
    """A branched tubule can start once its source tubule has saved the checkpoint (or has ended without it: then
    it runs from lift-off)."""
    if sc not in BRANCH:
        return True
    src, T = BRANCH[sc]
    if src not in names:
        names[src] = tubule_names(src)
    have = (out_dir(src) / "checkpoints" / f"{names[src][i]}_T{T:08.3f}.pkl").exists()
    return have or i in final.get(src, {}) or src not in final


def start(sc, i, attempt):
    arg = base_args(sc) + ["--tubule", str(i), "--autosave-min", str(AUTOSAVE_MIN)]
    if sc in CHECKPOINTS:
        arg += ["--checkpoint-at", *map(str, CHECKPOINTS[sc])]
    if sc in BRANCH:
        src, T = BRANCH[sc]
        arg += ["--resume-from", str((out_dir(src) / "checkpoints").relative_to(ROOT)), "--resume-at", str(T)]
    env = dict(os.environ, NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS", "1"))
    log = open(ROOT / "results" / f"mission_{sc}_T{i + 1}.log", "a" if attempt else "w", encoding="utf-8")
    if attempt:
        log.write(f"\n--- started again (attempt {attempt + 1}) {time.ctime()} ---\n")
        log.flush()
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
    final = {sc: {} for sc in todo}          # tubule -> exit code, once it is done for good
    for sc in todo:
        for i, n in enumerate(names[sc]):
            if done_tubule(sc, n):
                final[sc][i] = 0
    assembled = {sc for sc in todo if len(final[sc]) == len(names[sc])
                 and (out_dir(sc) / "mission_summary.json").exists()}
    jobs = [(sc, i) for sc in todo for i in range(len(names[sc])) if i not in final[sc]]
    running, tries, attempts, t0, results, last_start = {}, {}, {}, time.time(), [], 0.0
    for job, p in running_jobs(names).items():
        if job in jobs:
            jobs.remove(job)
            running[job] = (p, _Null())
            print(f"adopted the running job {job[0]} T{job[1] + 1} (pid {p.proc.pid})", flush=True)
    for sc in todo:
        if sc not in assembled and len(final[sc]) == len(names[sc]):
            results.append(assemble(sc, t0, list(final[sc].values())))
    print(f"campaign: {len(jobs)} tubule jobs to run, {sum(map(len, final.values()))} done already", flush=True)
    while jobs or running:
        for job, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                del running[job]
                sc, i = job
                print(f"{sc} T{i + 1} exit={p.returncode} {time.time() - t0:.0f} s", flush=True)
                if p.returncode != 0:
                    oom = out_of_memory(sc, i)
                    if oom or tries.get(job, 0) < RETRIES:
                        if not oom:
                            tries[job] = tries.get(job, 0) + 1
                        attempts[job] = attempts.get(job, 0) + 1
                        jobs.insert(0, job)          # continues from its autosave
                        continue
                final[sc][i] = p.returncode
                if len(final[sc]) == len(names[sc]):
                    results.append(assemble(sc, t0, list(final[sc].values())))
        for job in list(jobs):
            if len(running) >= a.workers:
                break
            # one job at a time, a minute apart, and only while the machine can commit its memory
            if time.time() - last_start < 60 or free_commit_gb() < JOB_MEMORY_GB:
                break
            sc, i = job
            if ready(sc, i, names, final):
                jobs.remove(job)
                running[job] = start(sc, i, attempts.get(job, 0))
                last_start = time.time()
        time.sleep(10)
    return max(results) if results else 0


if __name__ == "__main__":
    sys.exit(main())

"""One tubule of a mission with the v0.6 full-contact engine (engine.py), with checkpoints, branches and autosave.

python -m dolle_extras mission (which sets "engine": "full" unless the config names another) hands run_args to
dolle_sim.mission.run_mission / run_tubule as their runner; dolle_sim itself never imports this module.
The checkpoint, branch and autosave options exist only for this engine, because only its runs are long enough (hours per tubule) to need them:

  checkpoints_s   flight times [s] at which each tubule's state is saved (to <out>/checkpoints)
  resume_from     a checkpoints directory of another scenario's run: continue from it at resume_at_s, if the two
                  scenarios are identical up to then (engine.branch verifies it; otherwise the run starts at lift-off)
  autosave_s      save the state every this many seconds of computing (to <out>/autosave); a run killed before its end
                  continues from its last autosave, bit for bit the uninterrupted run

Provenance: moved from dolle_sim/mission.py (v0.6.2) in v0.7; an autosave that cannot be read (saved before
the move) is now skipped instead of stopping the run. No physics here.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from dolle_sim.mission import autosave_path

from .engine import FullContactSim, branch


def run_args(args) -> FullContactSim:
    """run_one for the (cfg, chamber, seed, out_dir) tuple of dolle_sim.mission._tubule_args: the runner that
    python -m dolle_extras mission passes to dolle_sim.mission.run_mission and run_tubule."""
    return run_one(*args)


def run_one(cfg: dict, ch: dict, seed: int, out_dir) -> FullContactSim:
    """Run one tubule (chamber `ch`) of the mission config `cfg` with the full-contact engine and return the run."""
    s = FullContactSim(cfg, ch, seed)
    if cfg.get("checkpoints_s") or cfg.get("resume_from"):
        s.checkpoints = list(cfg.get("checkpoints_s") or [])
        s.checkpoint_dir = Path(out_dir) / "checkpoints"
        if cfg.get("resume_from"):
            # continue a run of another scenario saved at resume_at_s (identical up to then; branch() verifies it,
            # and if it cannot be shown identical the run starts from lift-off instead)
            try:
                s = branch(s.checkpoint_path(float(cfg["resume_at_s"]), cfg["resume_from"]), s)
                print(f"  [{s.name}] continuing from T+{float(cfg['resume_at_s']):g} s of {cfg['resume_from']}",
                      flush=True)
            except (ValueError, FileNotFoundError) as e:
                print(f"  [{s.name}] not branched ({e}); running from lift-off", flush=True)
    auto = None
    if cfg.get("autosave_s") is not None:
        # a run killed before its end (out of memory, reboot) continues from its last autosave: the same tubule and
        # configuration (the saved run's initial-state signature equals this one's), bit-for-bit the uninterrupted run
        auto = autosave_path(out_dir, s.name)
        A = None
        if auto.exists():
            try:
                with open(auto, "rb") as f:
                    A = pickle.load(f)
            except (ImportError, AttributeError, pickle.UnpicklingError) as e:     # saved by an older version
                print(f"  [{s.name}] autosave {auto.name} cannot be read ({e}); not used", flush=True)
        if A is not None:
            if A.name == s.name and A._init_sig == s._init_sig:
                print(f"  [{s.name}] continuing from its autosave at T+{A.env.t0_flight + A.t:.3f} s", flush=True)
                s = A
            else:
                print(f"  [{s.name}] autosave {auto.name} is of another configuration; not used", flush=True)
    if auto is not None:
        s.run(progress=bool(cfg.get("progress", False)), autosave_path=auto, autosave_s=float(cfg["autosave_s"]))
    else:
        s.run(progress=bool(cfg.get("progress", False)))
    return s

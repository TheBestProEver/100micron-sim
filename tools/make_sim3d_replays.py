"""Write the recorded replays of the 3-D page (docs/sim3d/) with the readable core's 3-D simulation.

    python tools/make_sim3d_replays.py            (about 10 s)

Writes, in docs/sim3d/data/:
  two_tubules.json     two tubules side by side in microgravity, silica and graphite, from the same start
                       (dolle_core/scenes.py two_tubule_start: every input with its source or marked ASSUMED);
                       frames, every impact, the summary, the momentum ledger of every frame
                       (scenes.MomentumRecorder) and the momentum checks of each tubule; plus, for
                       comparison, the summary of the same grains spread through the whole tubule
  impact_closeup.json  one head-on impact of two 50 um radius grains up close, per material at 0.8 and 1.5 times
                       the sticking speed (scenes.closeup_cases)

Only numpy and dolle_core are needed.  The numbers it prints are the ones quoted in notebook 06.
The python files the page's live sandbox loads are copied by tools/build_sim3d.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dolle_core import scenes, sim3d  # noqa: E402

OUT = ROOT / "docs" / "sim3d" / "data"
PALETTE = ["#a9cfe8", "#4b4b4b", "#e6194b", "#3cb44b", "#ffb000", "#4363d8", "#f58231", "#911eb4", "#46c0c0",
           "#f032e6", "#9a6324", "#008080", "#800000", "#808000"]
"""Colour index -> colour: 0 a single silica grain (pale blue), 1 a single graphite grain (dark grey), 2-13 clumps
of two or more grains (2 + clump id mod 12).  The page dims wall-bound grains."""


def commit() -> str | None:
    """The repository's current git commit, if git is available."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def spread_start(start: dict, seed: int = 7) -> dict:
    """The same grains (sizes, velocities, adhesion) placed uniformly through the whole tubule, 2 radii clear of
    the walls, instead of in a compact cloud (positions ASSUMED, for comparison)."""
    size, r = np.asarray(start["tubule"].size), start["radii"]
    pos = scenes.puff_positions(r, size / 2, size - 4 * r.max(), np.random.default_rng(seed + 1000))
    return {**start, "positions": pos}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = scenes.two_tubule_start()
    tub = start["tubule"]
    tubules, report = [], {}
    for m in ("silica", "graphite"):
        sim = sim3d.make_sim(tub, m, start["radii"], start["positions"], start["velocities"],
                             adhesion=start["adhesion"])                  # = scenes.run_tubule, plus the ledger
        rec = scenes.MomentumRecorder(sim)
        sim3d.run(sim, 12.0, on_frame=rec)
        tubules.append(scenes.tubule_replay(sim, m, rec))
        report[m] = sim3d.summary(sim)
        report[m]["max_dev_rel"] = max(r[-1] for r in rec.rows)
    spread = {}
    for m in ("silica", "graphite"):
        s = scenes.run_tubule(spread_start(start), m, 12.0, frame_dt=1.0, slow_until=0.0)
        sm = sim3d.summary(s)
        spread[m] = {k: sm[k] for k in ("counts", "grain_grain_impacts", "largest_clump", "free_clumps_at_end",
                                        "t_end_s", "run_time_s")}
    replay = {
        "schema": "dolle-sim3d-replay/1", "made_by": "python tools/make_sim3d_replays.py", "commit": commit(),
        "engine": "dolle_core/sim3d.py (rigid clumps; contact physics of dolle_core/collide.py)",
        "scenario": {"settings": start["settings"], "sources": scenes.START_SOURCES,
                     "tubule": {"size_m": list(tub.size), "wall": tub.wall, "f_wall": tub.f_wall,
                                "accel_m_s2": list(tub.accel), "dt_s": tub.dt, "mu": "per pair (materials.FRICTION)"
                                if tub.mu is None else tub.mu, "e_t": tub.e_t, "n_sub": tub.n_sub}},
        "frames_note": "every 4 ms step up to 0.5 s (slow motion of the cloud phase), then 30 per second; positions "
                       "in integer micrometres",
        "palette": PALETTE, "tubules": tubules,
        "spread_start_for_comparison": {"note": "the same grains spread uniformly through the whole tubule "
                                                "(ASSUMED positions): grains practically never meet",
                                        **spread}}
    text = scenes.replay_json(replay)
    (OUT / "two_tubules.json").write_text(text, encoding="utf-8")
    close = {"schema": "dolle-sim3d-closeup/1", "made_by": "python tools/make_sim3d_replays.py", "commit": commit(),
             "note": "two 50 um radius grains head-on in the centre-of-mass frame (dolle_core.collide.impact); the "
                     "model has no force before touch (the JKR contact snaps on at touch)",
             "cases": scenes.closeup_cases()}
    ctext = json.dumps(close, separators=(",", ":"))
    (OUT / "impact_closeup.json").write_text(ctext, encoding="utf-8")

    for m, sm in report.items():
        c = sm["counts"]
        print(f"{m:9s} grain-grain impacts {sm['grain_grain_impacts']:3d} (stick {c['stick']}, bounce "
              f"{c['bounce']}, erosion {c['erosion']}, shatter {c['shatter']}); wall stick {c['wall_stick']}, "
              f"wall bounce {c['wall_bounce']}; clumps at end {sm['clump_sizes_at_end']}, largest "
              f"{sm['largest_clump']}; free at end {sm['free_clumps_at_end']} (t = {sm['t_end_s']:.2f} s); "
              f"last grain-grain impact {sm['last_grain_grain_impact_s']:.3f} s; worst event dP {sm['max_event_dP_rel']:.1e}, "
              f"dL {sm['max_event_dL_rel']:.1e}; ledger {sm['ledger_dP_rel']:.1e} / {sm['ledger_dL_rel']:.1e}; "
              f"total momentum constant to {sm['max_dev_rel']:.1e} of sum M|V|; "
              f"run {sm['run_time_s']:.1f} s")
    for m, sm in spread.items():
        print(f"{m:9s} spread through the tubule: grain-grain impacts {sm['grain_grain_impacts']}, "
              f"largest clump {sm['largest_clump']}, run {sm['run_time_s']:.1f} s")
    for case in close["cases"]:
        print(f"close-up {case['material']:9s} {case['v_in_mm_s']:6.2f} mm/s ({case['factor_of_v_stick']} v*): "
              f"{case['outcome']}, e = {case['e']:.3f}, contact {case['contact_time_us']:.2f} us, "
              f"{len(case['rows'])} rows")
    for f in ("two_tubules.json", "impact_closeup.json"):
        print(f"{f}: {(OUT / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.1f} s")

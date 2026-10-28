"""Markdown tables of the mission scenario results, read from results/mission/<scenario>/ (summaries and
trajectories), for the README and docs/data/mission_tables.md.

    python tools/mission_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_campaign import ORDER  # noqa: E402

FACE = {"-z": "bottom", "+z": "top", "-x": "side -x", "+x": "side +x", "-y": "side -y", "+y": "side +y"}


def _impacts(t):
    med = t.get("ug_median_impact_speed_m_s")
    s = f"{t['grains_released']} · {t['ug_grain_grain_impacts']}"
    if t["ug_monomer_impacts"]:
        s += f" ({t['ug_monomer_sticks']}/{t['ug_monomer_impacts']}" + (f", {med * 1e3:.0f} mm/s" if med else "") + ")"
    return s


def _moved(d, name, T_query, t0=0.0):
    """Fraction of grains more than one diameter from where they started, at flight time T_query."""
    f = d / name / "trajectory.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    t, pos, rad = z["t"], z["pos"], z["radius"]
    k = int(np.searchsorted(t + t0, T_query)) if (t + t0)[-1] >= T_query else len(t) - 1
    k = min(k, len(t) - 1)
    disp = np.linalg.norm(pos[k] - pos[0], axis=1)
    return float((disp > 2 * rad).mean()), float(np.median(disp) * 1e6)


def main():
    rows, rows2 = [], []
    names = None
    for sc in ORDER:
        d = ROOT / "results" / "mission" / sc
        f = d / "mission_summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        tubs = s["tubules"]
        if "dem_steps" not in (tubs[0].get("events") or {}):
            continue                                    # a v0.5 result
        names = [t["tubule"] for t in tubs]
        rows.append([f"`{sc}`"] + [_impacts(t) for t in tubs])
        cells = []
        for t in tubs:
            mv = _moved(d, t["tubule"], 332.0)
            faces = ", ".join(f"{n} {FACE.get(k, k)}" for k, n in sorted(t["wall_bound_by_face"].items(),
                                                                       key=lambda kv: -kv[1])[:3])
            ev = t["events"]
            br = f"; from T+{ev['branched_at_T']:g} s of a shared run" if "branched_at_T" in ev else ""
            cells.append((f"{100 * mv[0]:.0f}% moved > 1 diameter by power-off; " if mv else "")
                         + f"end: {faces}; {t['free_at_end']} free; "
                         f"verified steps {ev.get('checked_steps', 0)}, redone {ev.get('redos', 0)}, "
                         f"joins on approach {ev.get('dem_joins', 0)}; "
                         f"{(t.get('wall_time_s') or 0) / 3600:.1f} h{br}")
        rows2.append([f"`{sc}`"] + cells)
    if not rows:
        print("no v0.6 mission results yet")
        return
    head = ["scenario"] + [n.replace("_", " ") for n in names]
    out = ["**Grains released · grain–grain impacts in the microgravity window** (single-grain impacts that stuck / "
           "all single-grain impacts, median speed):", "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out += ["", "**Where the dust went, and what the engine did** (per tubule):", "",
            "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows2]
    txt = "\n".join(out) + "\n"
    (ROOT / "docs" / "data" / "mission_tables.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()

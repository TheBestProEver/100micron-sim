"""Result helpers without plotting dependencies (shared by the CLI and the in-browser app)."""
from __future__ import annotations

import base64
import csv
import io
import json
import math
from pathlib import Path

import numpy as np


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items() if k != "material_obj"}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def csv_text(rows) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def collision_stats(sim):
    pp = [c for c in sim.collisions if c["kind"] == "pp"]
    mono = [c for c in pp if c["n_i"] == 1 and c["n_j"] == 1 and not c.get("frozen_i") and not c.get("frozen_j")
            and not c.get("post_fragment")]
    s = {
        "grain_grain_collisions": len(pp),
        "grain_grain_sticks": sum(c["outcome"] == "stick" for c in pp),
        "monomer_monomer_collisions": len(mono),
        "monomer_monomer_sticks": sum(c["outcome"] == "stick" for c in mono),
        "impacts_on_wall_bound_grains": sum(bool(c.get("frozen_i") or c.get("frozen_j")) for c in pp),
        "wall_collisions": sum(c["kind"] == "wall" for c in sim.collisions),
        "wall_sticks": sum(c["kind"] == "wall" and c["outcome"] == "stick" for c in sim.collisions),
    }
    st = [c["v_n"] for c in pp if c["outcome"] == "stick"]
    bo = [c["v_n"] for c in pp if c["outcome"] == "bounce"]
    s["fastest_stick_m_s"] = max(st) if st else None
    s["slowest_bounce_m_s"] = min(bo) if bo else None
    ts = sim.timeseries[-1]
    s.update({k: ts[k] for k in ("free_bodies", "free_monomers", "wall_monomers",
                                 "mean_aggregate_size", "max_aggregate_size")})
    s["n_grains"] = int(sim.N)
    s["wall_time_s"] = getattr(sim, "wall_time", None)
    return s


def viewer_payload(sims, max_particles=1500, fps=15):
    """Compact (base64) replay data for the HTML viewer."""
    chambers = []
    for s in sims:
        rng = np.random.default_rng(0)
        idx = np.arange(s.N) if s.N <= max_particles else np.sort(rng.choice(s.N, max_particles, replace=False))
        t = np.asarray(s.frame_t)
        step = max(1, int(round((1.0 / fps) / max(s.out_dt, 1e-9))))
        fr = np.asarray(s.frames)[::step][:, idx, :]
        bodies = np.asarray(s.frame_bodies)[::step][:, idx]
        ts = t[::step]
        scale = float(s.size.max()) / 65000.0
        q = np.clip(np.round(fr / scale), 0, 65535).astype("<u2")
        # aggregate size per grain per frame (0 = stuck on a wall)
        full_bodies = np.asarray(s.frame_bodies)[::step]
        agg = np.zeros(bodies.shape, np.uint8)
        frozen_by_frame = np.asarray(s.frame_frozen)[::step]
        for f in range(len(ts)):
            counts = np.bincount(full_bodies[f], minlength=s.N)
            a = np.minimum(counts[bodies[f]], 255).astype(np.uint8)
            a[frozen_by_frame[f][idx]] = 0
            agg[f] = a
        hz = np.asarray(s.frame_spin)[::step][:, idx]
        with np.errstate(divide="ignore"):
            spin_code = np.where(hz > 0, np.clip(np.round(1 + 50 * (np.log10(np.maximum(hz, 1e-30)) + 2)), 1, 255), 0)
        spin_code = spin_code.astype(np.uint8)
        events = [[round(c["t"], 4), round(c["v_n"], 6), 1 if c["outcome"] == "stick" else 0,
                   1 if c["kind"] == "wall" else 0,
                   *[round(float(x) / scale) for x in (c["x"], c["y"], c["z"])]]
                  for c in s.collisions]
        chambers.append({
            "name": s.name, "size": s.size.tolist(), "scale": scale, "n_frames": len(ts),
            "n_shown": int(len(idx)), "n_total": int(s.N),
            "t": [round(float(x), 4) for x in ts],
            "radius_um": [round(float(r) * 1e6, 1) for r in s.rad[idx]],
            "material": s.mat[idx].tolist(), "materials": [m.name for m in s.mats],
            "pos_b64": base64.b64encode(q.tobytes()).decode(),
            "agg_b64": base64.b64encode(agg.tobytes()).decode(),
            "spin_b64": base64.b64encode(spin_code.tobytes()).decode(),
            "events": events,
        })
    return {"chambers": chambers}


def render_viewer(payload, title, template=None) -> str:
    tpl = template or (Path(__file__).parent / "viewer_template.html").read_text(encoding="utf-8")
    return tpl.replace("__TITLE__", title).replace("/*__DATA__*/null", json.dumps(payload))

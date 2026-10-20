"""Synthetic camera + sticking-threshold estimation: what would DOLL-E actually measure?

The flight camera sees a 2D projection of the chamber at a finite frame rate,
with finite depth of field and position noise. For every simulated grain-grain
impact this module decides whether it would be observable and trackable, and
what relative speed would be measured from the images. A logistic fit of
stick/bounce outcome vs speed then gives the sticking velocity (50% point)
with a bootstrap confidence interval -- the quantity the mission reports.

Camera config (per chamber "camera": {...}, or top level):
    view_axis          : direction the camera looks along (default [0,0,1])
    fps                : frame rate [1/s]
    fov_m              : [width, height] of the field of view at the focal plane [m]
    fov_center_m       : 3D point the field of view is centred on (default chamber centre)
    depth_of_field_m   : grains within +-DOF/2 of the focal plane are in focus
    position_noise_m   : centroiding error per frame (~0.3 px)
    track_frames       : frames of free flight needed before and after an impact
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


def _basis(axis):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, t)
    u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


def synthetic_camera(sim, cam: dict, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    a, u, v = _basis(cam.get("view_axis", [0, 0, 1]))
    fps = float(cam.get("fps", 500))
    fov = np.asarray(cam.get("fov_m", sim.size[:2].tolist()), float)
    c0 = np.asarray(cam.get("fov_center_m", (sim.size / 2).tolist()), float)
    dof = float(cam.get("depth_of_field_m", 0.01))
    sig_x = float(cam.get("position_noise_m", 5e-6))
    k = int(cam.get("track_frames", 6))
    # least-squares velocity error from k frames, per grain per image axis; x sqrt(2) for relative
    sig_v = math.sqrt(2) * sig_x * fps * math.sqrt(12.0 / (k * (k * k - 1)))
    t_track = k / fps

    # free-flight intervals: time since the previous / until the next event of each grain
    ev_t = {}
    for c in sim.collisions:
        for g in (c["i"], c["j"]):
            if g >= 0:
                ev_t.setdefault(g, []).append(c["t"])
    rows = []
    for c in sim.collisions:
        if c["kind"] != "pp":
            continue
        p = np.array([c["x"], c["y"], c["z"]]) - c0
        in_fov = abs(p @ u) <= fov[0] / 2 and abs(p @ v) <= fov[1] / 2
        in_focus = abs(p @ a) <= dof / 2
        before, after = math.inf, math.inf
        for g in (c["i"], c["j"]):
            ts = np.asarray(ev_t[g])
            prev = ts[ts < c["t"] - 1e-12]
            nxt = ts[ts > c["t"] + 1e-12]
            before = min(before, c["t"] - prev.max() if len(prev) else c["t"])
            after = min(after, nxt.min() - c["t"] if len(nxt) else sim.duration - c["t"])
        trackable = before >= t_track and after >= t_track
        vrel = np.array([c["vrel_x"], c["vrel_y"], c["vrel_z"]])
        v2 = np.array([vrel @ u, vrel @ v]) + sig_v * rng.standard_normal(2)
        rows.append(dict(t=c["t"], outcome=c["outcome"], n_i=c["n_i"], n_j=c["n_j"],
                         mat_i=c["mat_i"], mat_j=c["mat_j"],
                         v_n_true=c["v_n"], v_rel_true=c["v_rel"],
                         v_rel_measured_2d=float(np.linalg.norm(v2)),
                         in_fov=bool(in_fov), in_focus=bool(in_focus), trackable=bool(trackable),
                         observable=bool(in_fov and in_focus and trackable)))
    return rows


def fit_threshold(v, stuck, n_boot=200, seed=0):
    """Logistic fit P(stick) = 1 / (1 + exp((ln v - ln v50) / w)). Returns v50, CI and width."""
    v = np.asarray(v, float)
    y = np.asarray(stuck, float)
    ok = v > 0
    v, y = v[ok], y[ok]
    if len(v) < 5 or y.min() == y.max():
        return None

    def nll(th, lv, yy):
        z = (lv - th[0]) / math.exp(th[1])
        p = np.clip(1 / (1 + np.exp(np.clip(z, -50, 50))), 1e-9, 1 - 1e-9)
        return -(yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum()

    lv = np.log(v)
    x0 = [np.median(lv), math.log(0.3)]
    best = minimize(nll, x0, args=(lv, y), method="Nelder-Mead")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(v), len(v))
        if y[k].min() == y[k].max():
            continue
        r = minimize(nll, best.x, args=(lv[k], y[k]), method="Nelder-Mead")
        boots.append(math.exp(r.x[0]))
    lo, hi = (np.percentile(boots, [16, 84]) if len(boots) > 10 else (math.nan, math.nan))
    return {"v50_m_s": math.exp(best.x[0]), "v50_lo_1sigma": float(lo), "v50_hi_1sigma": float(hi),
            "log_width": math.exp(best.x[1]), "n": int(len(v)), "n_stick": int(y.sum())}


def measurement_report(sim, cam: dict, seed=0):
    rows = synthetic_camera(sim, cam, seed)
    obs = [r for r in rows if r["observable"]]
    mono = [r for r in rows if r["n_i"] == 1 and r["n_j"] == 1]
    rep = {
        "camera": cam,
        "grain_grain_impacts": len(rows),
        "observable_impacts": len(obs),
        "fraction_in_fov": float(np.mean([r["in_fov"] for r in rows])) if rows else 0.0,
        "fraction_in_focus": float(np.mean([r["in_focus"] for r in rows])) if rows else 0.0,
        "fraction_trackable": float(np.mean([r["trackable"] for r in rows])) if rows else 0.0,
        # 'truth' = what a perfect 3D observer of every monomer-monomer impact would infer
        "threshold_truth_normal_speed": fit_threshold([r["v_n_true"] for r in mono],
                                                      [r["outcome"] == "stick" for r in mono]),
        "threshold_truth_relative_speed": fit_threshold([r["v_rel_true"] for r in mono],
                                                        [r["outcome"] == "stick" for r in mono]),
        "threshold_measured_camera": fit_threshold([r["v_rel_measured_2d"] for r in obs],
                                                   [r["outcome"] == "stick" for r in obs]),
    }
    return rows, rep

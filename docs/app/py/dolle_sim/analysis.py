"""Synthetic camera + sticking-threshold estimation: what would DOLL-E actually measure?

The flight cameras see 2D projections of the chamber at a finite frame rate,
with finite depth of field and centroiding noise. For every simulated
grain-grain impact this module decides whether it would be observable and
trackable, and what impact speed would be inferred from the images. A logistic
fit of stick/bounce outcome vs speed then gives the sticking velocity (50%
point) with a bootstrap confidence interval -- the number the mission reports.

Sticking is controlled by the NORMAL impact speed. A camera, however, measures
projected velocities, so the analysis strategy matters a lot. Three are modelled:

  "2d_relative" : one camera, |projected relative velocity|. Simplest, but
                  oblique (grazing) impacts stick at high relative speed and
                  the line-of-sight component is invisible -> biased high and
                  a very broad transition.
  "2d_normal"   : one camera, relative velocity projected on the measured
                  line of centres at contact (both from the images). Removes
                  most of the impact-parameter smearing.
  "stereo_normal": two orthogonal cameras -> 3D relative velocity and 3D line
                  of centres -> the normal speed itself, up to noise.

All fits use free monomer-monomer impacts only: single grains are
distinguishable from aggregates in the images, and grains resting on a wall are
visibly stationary. Impacts on aggregates or on wall-bound grains have
different effective masses and thresholds, and impacts within 50 ms of a
fragmentation are excluded. The camera measures grain-centroid velocities (grain spin is
invisible at ~5 um centroid precision); the truth uses the contact-point normal speed.

Camera config (top-level "camera": {...}, overridable per chamber):
    view_axis          : direction camera 1 looks along (default [0,0,1])
    stereo_axis        : direction camera 2 looks along (default: orthogonal to view_axis)
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

STRATEGIES = ("2d_relative", "2d_normal", "stereo_normal")


def _basis(axis):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, t)
    u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


class _Camera:
    def __init__(self, axis, size, cam):
        self.a, self.u, self.v = _basis(axis)
        self.fov = np.asarray(cam.get("fov_m", None) or [float(np.max(size))] * 2, float)
        self.c0 = np.asarray(cam.get("fov_center_m", (np.asarray(size) / 2).tolist()), float)
        self.dof = float(cam.get("depth_of_field_m", 0.01))

    def sees(self, p):
        d = p - self.c0
        return (abs(d @ self.u) <= self.fov[0] / 2 and abs(d @ self.v) <= self.fov[1] / 2,
                abs(d @ self.a) <= self.dof / 2)

    def project(self, x):
        return np.array([x @ self.u, x @ self.v])


def synthetic_camera(sim, cam: dict, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    size = np.asarray(sim.size, float)
    ax1 = np.asarray(cam.get("view_axis", [0, 0, 1]), float)
    ax1 = ax1 / np.linalg.norm(ax1)
    ax2 = cam.get("stereo_axis")
    if ax2 is None:
        ax2 = _basis(ax1)[1]
    cam1, cam2 = _Camera(ax1, size, cam), _Camera(ax2, size, cam)
    fps = float(cam.get("fps", 500))
    sig_x = float(cam.get("position_noise_m", 5e-6))
    k = int(cam.get("track_frames", 6))
    # least-squares velocity error from k frames, per grain per image axis; x sqrt(2) for relative
    sig_v = math.sqrt(2) * sig_x * fps * math.sqrt(12.0 / (k * (k * k - 1)))
    t_track = k / fps

    # free-flight intervals: time since the previous / until the next event of each grain
    ev_t: dict[int, list] = {}
    for c in sim.collisions:
        for g in (c["i"], c["j"]):
            if g >= 0:
                ev_t.setdefault(int(g), []).append(c["t"])
    ev_t = {g: np.sort(np.asarray(ts)) for g, ts in ev_t.items()}
    rows = []
    for c in sim.collisions:
        if c["kind"] != "pp":
            continue
        p = np.array([c["x"], c["y"], c["z"]])
        fov1, foc1 = cam1.sees(p)
        fov2, foc2 = cam2.sees(p)
        before, after = math.inf, math.inf
        for g in (int(c["i"]), int(c["j"])):
            ts = ev_t[g]
            kp = np.searchsorted(ts, c["t"] - 1e-12, side="left") - 1     # last event strictly before
            kn = np.searchsorted(ts, c["t"] + 1e-12, side="right")        # first event strictly after
            before = min(before, c["t"] - ts[kp] if kp >= 0 else c["t"])
            after = min(after, ts[kn] - c["t"] if kn < len(ts) else sim.duration - c["t"])
        trackable = before >= t_track and after >= t_track
        # a centroid-tracking camera sees the grain-centre relative velocity, not the contact-point one
        vrel = np.array([c.get("vcm_x", c["vrel_x"]), c.get("vcm_y", c["vrel_y"]), c.get("vcm_z", c["vrel_z"])])
        n = np.array([c["n_x"], c["n_y"], c["n_z"]])
        r_sep = float(c.get("r_sum", 1e-4))                 # centre separation at contact
        # --- camera 1: projected relative velocity and projected line of centres (noisy)
        v1 = cam1.project(vrel) + sig_v * rng.standard_normal(2)
        n1 = cam1.project(n) * r_sep + math.sqrt(2) * sig_x * rng.standard_normal(2)
        n1_hat = n1 / max(np.linalg.norm(n1), 1e-30)
        # --- stereo: two orthogonal views give all three components (shared axis averaged)
        v2 = cam2.project(vrel) + sig_v * rng.standard_normal(2)
        n2 = cam2.project(n) * r_sep + math.sqrt(2) * sig_x * rng.standard_normal(2)
        v3 = _stereo_merge(cam1, cam2, v1, v2)
        n3 = _stereo_merge(cam1, cam2, n1, n2)
        n3_hat = n3 / max(np.linalg.norm(n3), 1e-30)
        obs1 = bool(fov1 and foc1 and trackable)
        rows.append(dict(
            t=c["t"], outcome=c["outcome"], n_i=int(c["n_i"]), n_j=int(c["n_j"]),
            on_wall_partner=bool(c.get("frozen_i", False) or c.get("frozen_j", False)),
            post_fragment=bool(c.get("post_fragment", False)),
            mat_i=c["mat_i"], mat_j=c["mat_j"],
            v_n_true=c["v_n"], v_rel_true=c["v_rel"],
            v_rel_measured_2d=float(np.linalg.norm(v1)),
            v_n_measured_2d=float(abs(v1 @ n1_hat)),
            v_n_measured_stereo=float(abs(v3 @ n3_hat)),
            in_fov=bool(fov1), in_focus=bool(foc1), trackable=bool(trackable),
            observable=obs1, observable_stereo=bool(obs1 and fov2 and foc2)))
    return rows


def _stereo_merge(c1, c2, m1, m2):
    """Least-squares 3D vector from two 2D projections."""
    A = np.vstack([c1.u, c1.v, c2.u, c2.v])
    b = np.concatenate([m1, m2])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def fit_threshold(v, stuck, n_boot=200, seed=0):
    """Logistic fit P(stick) = 1 / (1 + exp((ln v - ln v50) / w)). Returns v50, 1-sigma CI and width."""
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
    # bounded fit: the threshold stays within the sampled speed range and the width within sane limits
    # (perfectly separated samples would otherwise drive the width to 0 and the fit to overflow)
    bounds = [(lv.min() - 1.0, lv.max() + 1.0), (math.log(1e-3), math.log(5.0))]
    x0 = [float(np.median(lv)), math.log(0.3)]
    best = minimize(nll, x0, args=(lv, y), method="L-BFGS-B", bounds=bounds)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(v), len(v))
        if y[k].min() == y[k].max():
            continue
        r = minimize(nll, best.x, args=(lv[k], y[k]), method="L-BFGS-B", bounds=bounds)
        boots.append(math.exp(r.x[0]))
    lo, hi = (np.percentile(boots, [16, 84]) if len(boots) > 10 else (math.nan, math.nan))
    return {"v50_m_s": math.exp(best.x[0]), "v50_lo_1sigma": float(lo), "v50_hi_1sigma": float(hi),
            "log_width": math.exp(best.x[1]), "n": int(len(v)), "n_stick": int(y.sum())}


def measurement_report(sim, cam: dict, seed=0):
    rows = synthetic_camera(sim, cam, seed)
    mono = [r for r in rows if r["n_i"] == 1 and r["n_j"] == 1 and not r["on_wall_partner"]
            and not r["post_fragment"]]
    wallg = [r for r in rows if r["n_i"] == 1 and r["n_j"] == 1 and r["on_wall_partner"] and not r["post_fragment"]]
    obs = [r for r in mono if r["observable"]]
    obs3 = [r for r in mono if r["observable_stereo"]]
    st = lambda rr: [r["outcome"] == "stick" for r in rr]  # noqa: E731
    rep = {
        "camera": cam,
        "grain_grain_impacts": len(rows),
        "monomer_monomer_impacts": len(mono),
        "impacts_on_wall_bound_grains": len(wallg),
        "observable_impacts": sum(r["observable"] for r in rows),
        "observable_monomer_impacts": len(obs),
        "observable_monomer_impacts_stereo": len(obs3),
        "fraction_in_fov": float(np.mean([r["in_fov"] for r in rows])) if rows else 0.0,
        "fraction_in_focus": float(np.mean([r["in_focus"] for r in rows])) if rows else 0.0,
        "fraction_trackable": float(np.mean([r["trackable"] for r in rows])) if rows else 0.0,
        # 'truth' = a perfect 3D observer of every monomer-monomer impact
        "threshold_truth_normal_speed": fit_threshold([r["v_n_true"] for r in mono], st(mono)),
        "threshold_truth_relative_speed": fit_threshold([r["v_rel_true"] for r in mono], st(mono)),
        "threshold_truth_on_wall_bound_grains": fit_threshold([r["v_n_true"] for r in wallg], st(wallg)),
        # what each camera strategy would report
        "threshold_2d_relative": fit_threshold([r["v_rel_measured_2d"] for r in obs], st(obs)),
        "threshold_2d_normal": fit_threshold([r["v_n_measured_2d"] for r in obs], st(obs)),
        "threshold_stereo_normal": fit_threshold([r["v_n_measured_stereo"] for r in obs3], st(obs3)),
    }
    rep["threshold_measured_camera"] = rep["threshold_stereo_normal"] or rep["threshold_2d_normal"]
    return rows, rep

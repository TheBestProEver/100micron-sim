"""JSON API used by the in-browser app (Pyodide). Every function takes and returns JSON strings.

No plotting dependencies: charts are drawn by the web page.

Provenance: JSON wrapper for the browser app (no physics).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict

import numpy as np

from . import __version__
from .materials import MATERIALS, get_material, pair_props, sphere_mass
from .contact import JKRContact
from .design import preflight as _preflight, stick_probability_curve, infer_surface_energy
from .simulation import ChamberSim
from .analysis import measurement_report
from .results import jsonable, collision_stats, csv_text, viewer_payload
from . import verification

WALL_MATERIALS = ["glass_fluorosilane", "glass_bk7", "glass_borosilicate", "glass_soda_lime", "fused_silica_glass",
                  "glass_silanized", "sapphire", "aluminum_6061", "ptfe"]


def _dump(o):
    return json.dumps(jsonable(o))


def default_config():
    """A browser-sized version of the mission configuration (two chambers, 200 grains, 4 s)."""
    def chamber(name, material, v_mean, v_sigma, vib_amp):
        return {
            "name": name, "size_m": [0.01, 0.01, 0.01], "wall_material": "glass_fluorosilane",
            "wall_adhesion_factor": 0.05,
            "wall_vibration": {"amplitude_m": vib_amp, "frequency_hz": 200, "axes": [0, 1, 2]},
            "center_from_spin_axis_m": [0.05, 0.0, 0.0], "center_from_cg_m": [0.05, 0.0, 0.3],
            "particles": [{"material": material, "count": 200, "radius_m": 50e-6, "radius_rel_sigma": 0.05,
                           "surface_energy_log_sigma": 0.3, "charge_e_mean": 0, "charge_e_sigma": 2e4,
                           "spin": {"rate_hz_mean": 5.0, "rate_hz_sigma": 3.0},
                           "velocity": {"mode": "isotropic", "speed_mean_m_s": v_mean, "speed_sigma_m_s": v_sigma}}],
        }
    return {
        "name": "browser_run", "seed": 1, "duration_s": 4.0, "dt_s": 0.004, "output_fps": 15,
        "environment": {"preset": "sounding_rocket", "pressure_pa": 1.0, "temperature_k": 293, "gas": "air",
                        "residual_accel_g": [0.0, 0.0, 1e-4], "jitter_rms_g": 5e-5, "jitter_corr_time": 0.05,
                        "spin_rate_hz": 0.0167, "spin_axis": [0, 0, 1], "gravity_gradient": True,
                        "altitude_m": 130000, "local_vertical": [0, 0, 1],
                        "electric_field_v_m": [0, 0, 0], "magnetic_field_t": [0, 2e-5, -4.5e-5],
                        "temperature_gradient_k_m": [0.0, 0.0, 3.0],
                        "illumination": {"intensity_w_m2": 1000.0, "direction": [1.0, 0.0, 0.0]}},
        "forces": {"coulomb": True, "space_charge": True, "polarization": True, "image_charges": True,
                   "self_gravity": True, "lorentz": True, "external_e_field": True, "thermophoresis": True,
                   "photophoresis": True, "radiation_pressure": True, "brownian": True,
                   "long_range_cutoff_m": 0.003, "long_range_update_every": 1, "space_charge_update_every": 25},
        "contact": {"friction_coefficient": 0.3, "tangential_restitution": 0.0, "wall_sticking": "jkr",
                    "contact_charging": True, "fragmentation": True, "neighbor_skin_m": 0.001,
                    "steps_per_contact": 100},
        "camera": {"view_axis": [0, 0, 1], "fps": 500, "depth_of_field_m": 0.01, "position_noise_m": 5e-6,
                   "track_frames": 6},
        "materials": {},
        "chambers": [chamber("carbon", "graphite", 0.035, 0.02, 2.5e-5),
                     chamber("silica", "silica", 0.007, 0.004, 5e-6)],
    }


def info(_=None):
    mats = {}
    for k, m in MATERIALS.items():
        d = asdict(m)
        if not math.isfinite(d["yield_pressure"]):
            d["yield_pressure"] = None
        mats[k] = d
    return _dump({"version": __version__, "materials": mats, "wall_materials": WALL_MATERIALS,
                  "dust_materials": [k for k in MATERIALS if k not in WALL_MATERIALS],
                  "default_config": default_config(), "checks": verification.list_checks()})


def preflight(cfg_json):
    cfg = json.loads(cfg_json)
    return _dump([_preflight(cfg, ch) for ch in cfg["chambers"]])


def _binned(v, stuck, nbins=14):
    v = np.maximum(np.asarray(v, float), 1e-6)
    y = np.asarray(stuck, float)
    if len(v) == 0:
        return []
    bins = np.logspace(np.log10(v.min()), np.log10(v.max()) + 1e-9, nbins)
    idx = np.digitize(v, bins)
    out = []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() >= 3:
            out.append([float(np.sqrt(bins[b - 1] * bins[b])), float(y[m].mean()), int(m.sum())])
    return out


def run(cfg_json, progress=None):
    cfg = json.loads(cfg_json)
    seed = int(cfg.get("seed", 1))
    sims, out = [], []
    nch = len(cfg["chambers"])
    for k, ch in enumerate(cfg["chambers"]):
        pf = _preflight(cfg, ch)
        sim = ChamberSim(cfg, ch, seed + k)

        def cb(t, T, ev, k=k, name=sim.name):
            if progress is not None:
                progress(json.dumps({"chamber": k, "n_chambers": nch, "name": name, "t": t, "T": T,
                                     "events": ev}))
        sim.run(progress=False, callback=cb)
        cam = dict(cfg.get("camera", {}))
        cam.update(ch.get("camera", {}))
        rows, rep = measurement_report(sim, cam)
        mono = [r for r in rows if r["n_i"] == 1 and r["n_j"] == 1 and not r["on_wall_partner"]
                and not r["post_fragment"]]
        st = lambda rr: [r["outcome"] == "stick" for r in rr]  # noqa: E731
        series = {
            "truth": _binned([r["v_n_true"] for r in mono], st(mono)),
            "2d_relative": _binned([r["v_rel_measured_2d"] for r in mono if r["observable"]],
                                   st([r for r in mono if r["observable"]])),
            "2d_normal": _binned([r["v_n_measured_2d"] for r in mono if r["observable"]],
                                 st([r for r in mono if r["observable"]])),
            "stereo": _binned([r["v_n_measured_stereo"] for r in mono if r["observable_stereo"]],
                              st([r for r in mono if r["observable_stereo"]])),
        }
        ts = sim.timeseries
        stride = max(1, len(ts) // 300)
        out.append({
            "name": sim.name, "stats": collision_stats(sim), "measurement": rep, "preflight": pf,
            "measurement_points": series, "timeseries": ts[::stride] + ([ts[-1]] if len(ts) % stride else []),
            "collisions_csv": csv_text(sim.collisions),
            "camera_csv": csv_text(rows),
        })
        sims.append(sim)
    return _dump({"chambers": out, "viewer": viewer_payload(sims, max_particles=800, fps=15)})


def sticking(args_json):
    a = json.loads(args_json)
    ov = a.get("materials", {})
    ma, mb = get_material(a["mat_a"], ov), get_material(a["mat_b"], ov)
    r = float(a.get("diameter_um", 100)) * 0.5e-6
    c = JKRContact(pair_props(ma, r, mb, r))
    m_eff = 1 / (1 / sphere_mass(ma, r) + 1 / sphere_mass(mb, r))
    v_th = c.threshold(m_eff)
    vs = np.logspace(math.log10(v_th / 20), math.log10(v_th * 20), int(a.get("n_v", 30)))
    probs = stick_probability_curve(a["mat_a"], a["mat_b"], r, vs, float(a.get("radius_rel_sigma", 0.05)),
                                    float(a.get("gamma_log_sigma", 0.3)), int(a.get("samples", 40)), ov)
    e_curve = [c.resolve(v, m_eff).e_n for v in vs]
    return _dump({"v_stick_dem": v_th, "v_stick_energy_estimate": c.v_stick_analytic(m_eff),
                  "v_stick_jkr_hysteresis": c.v_stick_elastic(m_eff), "v_yield": c.v_yield(m_eff),
                  "pull_off_force_N": c.p.F_c, "break_energy_J": c.p.E_break, "m_eff_kg": m_eff,
                  "velocities": vs, "stick_probability": probs, "restitution": e_curve})


def invert(args_json):
    a = json.loads(args_json)
    r = float(a.get("diameter_um", 100)) * 0.5e-6
    g = infer_surface_energy(a["material"], r, float(a["v_stick_m_s"]), a.get("partner") or None,
                             a.get("materials", {}))
    base = get_material(a["material"], a.get("materials", {}))
    return _dump({"gamma_eff": g, "gamma_database": base.gamma, "ratio": g / base.gamma})


def verify(name):
    return _dump(verification.run_check(name))


def launch_loads(cfg_json):
    """Analytic launch-load report for a mission config (mission.py: launch_report)."""
    from .mission import launch_report
    return _dump(launch_report(json.loads(cfg_json)))


def scale_config(cfg_json, factor):
    """Utility for the page: scale grain counts (keeps everything else)."""
    cfg = json.loads(cfg_json)
    for ch in cfg["chambers"]:
        for g in ch["particles"]:
            g["count"] = max(1, int(round(g["count"] * float(factor))))
    return json.dumps(cfg)


def thresholds(args_json):
    """Predicted grain-grain sticking speed for each material (same material pairs) at one grain size."""
    a = json.loads(args_json)
    ov = a.get("materials", {})
    r = float(a.get("diameter_um", 100)) * 0.5e-6
    out = {}
    for name in a["materials_list"]:
        m = get_material(name, ov)
        c = JKRContact(pair_props(m, r, m, r))
        out[name] = c.threshold(sphere_mass(m, r) / 2)
    return _dump(out)

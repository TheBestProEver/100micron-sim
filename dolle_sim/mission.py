"""One-sweep mission: every tubule through the whole flight on one clock, plus a launch-load report.

    python -m dolle_sim mission configs/mission_quarter_deck.json [--scenario NAME[+NAME]] [--out DIR]
        (default DIR: results/mission/<scenario or nominal>)
    python -m dolle_sim mission configs/mission_quarter_deck.json --loads-only

A mission config describes the payload (tubules on the deck), the flight timeline (flight.py) and the gas.
Each tubule becomes one simulated chamber. All tubules share the same flight clock: the simulation starts at
"simulate_from_s" (T+0 = lift-off, with the dust settled on the aft face as it is on the pad) and runs to 5 s
after splashdown (or to "simulate_to_s"), passing through the boost, coast, spin, de-spin, microgravity,
re-entry, parachute, descent and splashdown phases with the correct timing. Impact statistics are counted in the microgravity window only (ug_start_s to
ug_end_s, the ~100 km crossings). "scenarios" are named sets of overrides (gas, release mechanism,
orientation, flight assumptions) for comparisons; "a+b" applies both.

Tubule frame: +z along the rocket axis toward the nose (the aft face is z = 0); x, y in the deck plane,
with the spin axis at the deck centre. A "standing" tubule has its long axis along z; a "lying" one along
deck x (or y with "lying_axis": "y").

The launch-load report is analytic: for each flight phase it gives the loads at the tubule (axial,
centrifugal, Euler, and the guide's lateral ascent load) and which fraction of the dust's per-grain pull-off
thresholds they exceed, plus an estimate of detachment by random vibration at the ground-test levels, in flight
and in the ground vibration test (Rice's up-crossing rate for a Gaussian process). The time-resolved simulation
then shows what the grains actually do under the steady loads (it has no random vibration).

Provenance: geometry and timing from the mission config and flight.py; per-grain pull-off thresholds from
the same JKR relation as the simulation; vibration exceedance from Rice (1944, Bell Syst. Tech. J. 23, 282).
No REBOUND code.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from .environment import G0
from .flight import FlightProfile
from .materials import get_material, pair_props, sphere_mass
from .sizes import sample_diameters

INCH = 0.0254

# ground vibration test (RockSat User Guide 2026, Appendix B): random, flat PSD 20-2000 Hz; and sine
VIBRATION_TEST = {"thrust_grms": 10.0, "lateral_grms": 7.6, "f_lo_hz": 20.0, "f_hi_hz": 2000.0,
                  "test_s_per_axis": 20.0,   # "Each axis will see 20 seconds at full level"
                  "sine_peak_g": 7.0}        # sine test: 7 G from 144 to 2000 Hz on the thrust axis


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def apply_scenario(mission: dict, name: str | None) -> dict:
    """Apply a named scenario; "a+b" applies a, then b (e.g. "kick+lying")."""
    m = copy.deepcopy(mission)
    if not name:
        return m
    sc = m.get("scenarios", {})
    for part in name.split("+"):
        if part not in sc:
            raise KeyError(f"unknown scenario '{part}' (choose from {sorted(sc)}, or combine them with '+')")
        over = copy.deepcopy(sc[part])
        tub_over = over.pop("tubules", None)          # applied to every tubule
        _deep_update(m, over)
        if tub_over:
            for t in m["tubules"]:
                _deep_update(t, tub_over)
    m["scenario"] = name
    return m


def tubule_size_m(t: dict) -> np.ndarray:
    w, h, L = (np.asarray(t.get("inner_size_in", [0.5, 0.5, 1.5]), float) * INCH)
    orient = t.get("orientation", "standing")
    if orient == "standing":
        return np.array([w, h, L])
    if orient == "lying":
        return np.array([L, h, w]) if t.get("lying_axis", "x") == "x" else np.array([h, L, w])
    raise ValueError(f"orientation must be 'standing' or 'lying', not '{orient}'")


def tubule_chamber(m: dict, t: dict) -> dict:
    size = tubule_size_m(t)
    pos = list(t.get("deck_position_m", [0.05, 0.0])) + [0.0]
    dust = dict(t["dust"])
    dust.setdefault("velocity", {"speed_mean_m_s": 0.0})
    # the dust rests on the aft face from the pad until something frees it, whenever the run starts;
    # "initial_state": {"type": "free"} starts it floating instead (an idealised start)
    dust.setdefault("initial_state", {"type": "settled", "face": "-z"})
    ch = {"name": t["name"], "size_m": size.tolist(), "center_from_spin_axis_m": pos, "center_from_cg_m": pos,
          "wall_material": t.get("wall_material", "glass_borosilicate"),
          "wall_adhesion_factor": t.get("wall_adhesion_factor", 1.0),
          "particles": [dust], "release": t.get("release", [])}
    if t.get("wall_vibration"):
        ch["wall_vibration"] = t["wall_vibration"]
    if m.get("vibration"):
        ch["vibration"] = m["vibration"]
    if t.get("camera"):
        ch["camera"] = t["camera"]
    else:
        # camera across the tubule's short horizontal side; field of view [u, v] covers the other two sides
        from .analysis import _basis
        ax = 0 if (t.get("orientation", "standing") == "standing" or t.get("lying_axis", "x") == "y") else 1
        a = np.zeros(3)
        a[ax] = 1.0
        _, u, v = _basis(a)
        ch["camera"] = {"view_axis": a.tolist(), "fov_m": [float(np.abs(u) @ size), float(np.abs(v) @ size)],
                        "depth_of_field_m": float(size[ax])}
    return ch


def mission_cfg(m: dict) -> dict:
    prof = FlightProfile.from_config(m.get("flight_profile"))
    t0 = float(m.get("simulate_from_s", 0.0))
    gas = dict(m.get("gas", {"pressure_pa": 1e-3}))
    env = {"preset": "ideal", "gravity_gradient": True, "start_time_s": t0,
           "flight_profile": m.get("flight_profile", {}), "jitter_rms_g": m.get("jitter_rms_g", 0.0), **gas}
    cfg = {"name": m.get("name", "mission"), "dt_s": m.get("dt_s", 4e-3), "dt_launch_s": m.get("dt_launch_s", 5e-4),
           "duration_s": min(float(m.get("simulate_to_s", prof.end_s)), prof.end_s) - t0,
           "output_fps": m.get("output_fps", 5), "seed": m.get("seed", 0),
           "environment": env, "forces": m.get("forces", {}), "contact": m.get("contact", {}),
           "camera": m.get("camera", {}), "materials": m.get("materials", {}),
           "engine": m.get("engine", "full"), "progress": m.get("progress", False)}
    for k in ("checkpoints_s", "resume_from", "resume_at_s"):      # see _run_one and engine.branch
        if m.get(k) is not None:
            cfg[k] = m[k]
    return cfg


# ---------------------------------------------------------------------------- analytic launch loads
def grain_thresholds_g(m: dict, t: dict, n: int = 4000, seed: int = 0) -> np.ndarray:
    """Per-grain wall pull-off thresholds a_th = F_c / m [g] for a tubule's dust (sizes and adhesion scatter)."""
    rng = np.random.default_rng(seed)
    dust = t["dust"]
    ov = m.get("materials", {})
    mat = get_material(dust["material"], ov)
    wall = get_material(t.get("wall_material", "glass_borosilicate"), ov)
    if "size_distribution" in dust:
        d = sample_diameters(dust["size_distribution"], n, rng)
    else:
        d = np.full(n, 2 * float(dust.get("radius_m", 50e-6)))
    f = np.exp(float(dust.get("surface_energy_log_sigma", 0.0)) * rng.standard_normal(n)) * float(
        t.get("wall_adhesion_factor", 1.0))
    # F_c = 1.5 pi w R is linear in R (flat wall) and in the adhesion factor: scale one pair_props evaluation
    r0 = 50e-6
    p0 = pair_props(mat, r0, wall, None)
    r = d / 2
    return p0.F_c * (r / r0) * f / (sphere_mass(mat, 1.0) * r ** 3) / G0


def _vib_exceed_prob(level_g, grms, duration_s, f_lo, f_hi):
    """Probability that a zero-mean Gaussian random acceleration (flat PSD f_lo-f_hi, rms grms) exceeds level_g at
    least once in duration_s: 1 - exp(-nu0 T exp(-x^2/2 sigma^2)), nu0 = sqrt(int f^2 S / int S) (Rice 1944)."""
    nu0 = math.sqrt((f_hi ** 3 - f_lo ** 3) / (3 * (f_hi - f_lo)))
    n_up = nu0 * duration_s * np.exp(-0.5 * (np.asarray(level_g) / grms) ** 2)
    return 1 - np.exp(-n_up)


def launch_report(m: dict, n_samples: int = 4000) -> dict:
    """Analytic loads vs pull-off thresholds for every tubule and flight phase (dust on the aft face).

    Per phase: the largest load pulling dust off the aft face and the largest load along it (centrifugal + Euler,
    plus the guide's lateral quasi-static load during the ascent as a worst-case sum), the fraction of grains
    pulled off quasi-statically, and the fraction that could slide (tangential load > mu x (adhesion - pull) at
    the same instant). Random-vibration ESTIMATE (Rice up-crossings of a Gaussian at the test level): every
    exceedance of a grain's margin frees it; the thrust is not counted as pressing the dust down (conservative)
    while a drag pull reduces the margin; applied from lift-off to second-stage burnout, and separately to the
    ground vibration test (20 s at full level on the thrust axis, payload upright so 1 g presses the dust) and the
    sine test (7 g peak). Rolling is not estimated (see provenance.py)."""
    prof = FlightProfile.from_config(m.get("flight_profile"))
    mu = float(m.get("contact", {}).get("friction_coefficient", 0.3))
    vib = dict(VIBRATION_TEST, **m.get("vibration_test", {}))
    f_lo, f_hi = vib["f_lo_hz"], vib["f_hi_hz"]
    tubs = []
    for t in m["tubules"]:
        a_th = grain_thresholds_g(m, t, n=n_samples)
        pos = np.array(list(t.get("deck_position_m", [0.05, 0.0])), float)
        size = tubule_size_m(t)
        # farthest point of the aft face from the spin axis (worst centrifugal load)
        corners = [pos + np.array([sx * size[0] / 2, sy * size[1] / 2]) for sx in (-1, 1) for sy in (-1, 1)]
        r_far = max(corners, key=lambda c: np.hypot(*c))
        phases = []
        survive_flight = np.ones_like(a_th)
        for ph in prof.phases():
            ts = list(ph["t0"] + (np.arange(200) + 0.5) / 200 * (ph["t1"] - ph["t0"]))   # strictly inside the phase
            ts += [float(i["t_s"]) + 0.5 * float(i["duration_s"]) for i in prof.impulses       # impulse peaks
                   if ph["t0"] <= float(i["t_s"]) + 0.5 * float(i["duration_s"]) < ph["t1"]]
            ascent = ph["t1"] <= prof.burn2[1] + 1e-9
            pull, tang = [], []           # load off the aft face (+ = away from it) and along it [g]
            for tt in ts:
                L = prof.loads_at(tt, r_far)
                pull.append(L["total"][2])
                tang.append(float(np.hypot(L["total"][0], L["total"][1])) + (prof.lateral_g if ascent else 0.0))
            pull, tang = np.array(pull), np.array(tang)
            pull_max = float(pull.max())
            frac_qs = float((a_th < pull_max).mean())
            # sliding (not modelled): friction capacity mu x (adhesion + pressing load) at the same instant
            cap = mu * np.maximum(a_th[:, None] - pull[None, :], 0.0)
            frac_slide = float((tang[None, :] > cap).any(axis=1).mean())
            p_vib = np.zeros_like(a_th)
            if ascent:
                need = np.maximum(a_th - max(pull_max, 0.0), 0.0)      # thrust pressing ignored (conservative)
                p_vib = _vib_exceed_prob(need, vib["thrust_grms"], ph["t1"] - ph["t0"], f_lo, f_hi)
                survive_flight *= 1 - p_vib
            phases.append(dict(phase=ph["name"], t0=ph["t0"], t1=ph["t1"], source=ph["source"],
                               axial_g_range=ph["axial_g_range"], max_pull_off_aft_g=pull_max,
                               max_press_onto_aft_g=float(max(0.0, -pull.min())),
                               max_tangential_g=float(tang.max()), fraction_pulled_off_quasi_static=frac_qs,
                               fraction_could_slide=frac_slide, fraction_freed_by_vibration_estimate=float(p_vib.mean())))
        p_ground = _vib_exceed_prob(a_th + 1.0, vib["thrust_grms"], vib["test_s_per_axis"], f_lo, f_hi)
        vibration = {"ground_random_test": float(p_ground.mean()),
                     "ground_sine_test": float((a_th + 1.0 < vib["sine_peak_g"]).mean()),
                     "flight_lift_off_to_burnout": float(1 - survive_flight.mean()),
                     "all": float(1 - (survive_flight * (1 - p_ground) * (a_th + 1.0 >= vib["sine_peak_g"])).mean())}
        tubs.append(dict(name=t["name"], orientation=t.get("orientation", "standing"), size_m=size.tolist(),
                         deck_position_m=pos.tolist(), r_far_m=float(np.hypot(*r_far)),
                         dust=t["dust"]["material"], wall=t.get("wall_material", "glass_borosilicate"),
                         threshold_g={"p10": float(np.percentile(a_th, 10)), "median": float(np.median(a_th)),
                                      "p90": float(np.percentile(a_th, 90))},
                         phases=phases, vibration_estimate=vibration))
    return {"profile": prof.to_dict(), "tubules": tubs}


# ---------------------------------------------------------------------------- simulation
def _run_one(args):
    cfg, ch, seed, out_dir = args
    if cfg.get("engine", "full") == "full":
        from .engine import FullContactSim as Sim          # v0.6: every grain a body, full contact mechanics
    else:
        from .simulation import ChamberSim as Sim           # v0.5 event-driven engine with rigid aggregates
    s = Sim(cfg, ch, seed)
    if cfg.get("checkpoints_s") or cfg.get("resume_from"):
        from .engine import branch
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
    s.run(progress=bool(cfg.get("progress", False)))
    return s


def _in_window(c_t, t0, prof):
    return prof.ug_start_s <= c_t + t0 < prof.ug_end_s


def summarize(sim, prof: FlightProfile) -> dict:
    """Per-tubule results. Impact statistics cover the microgravity window [ug_start_s, ug_end_s] only."""
    t0 = sim.env.t0_flight
    pp = [c for c in sim.collisions if c["kind"] == "pp" and _in_window(c["t"], t0, prof)]
    mono = [c for c in pp if c["n_i"] == 1 and c["n_j"] == 1 and not (c["frozen_i"] or c["frozen_j"])]
    det = sim.detach_log
    by_phase = {}
    for ph in prof.phases():
        by_phase[ph["name"]] = sum(1 for d in det if ph["t0"] <= d["t"] < ph["t1"])
    fz = sim.frozen[sim.body]
    face = {}
    for k in range(sim.N):
        if fz[k]:
            b = sim.body[k]
            key = f"{'-+'[int(sim.fw_side[b] > 0)]}{'xyz'[int(sim.fw_ax[b])]}"
            face[key] = face.get(key, 0) + 1
    ts = [r for r in sim.timeseries if r["t"] + t0 <= prof.power_off_s]
    free_po = int(ts[-1]["free_monomers"]) if ts else None
    return {"tubule": sim.name, "n_grains": int(sim.N), "end_s": float(t0 + sim.duration),
            "free_at_power_off": free_po, "free_at_end": int((~fz).sum()), "wall_bound_by_face": face,
            "grains_released": len(sim._ever_freed), "detachments_by_phase": by_phase,
            "detachment_events": len(det),
            "ug_window_s": [prof.ug_start_s, prof.ug_end_s],
            "ug_grain_grain_impacts": len(pp), "ug_monomer_impacts": len(mono),
            "ug_monomer_sticks": sum(c["outcome"] == "stick" for c in mono),
            "ug_median_impact_speed_m_s": float(np.median([c["v_n"] for c in mono])) if mono else None,
            "events": dict(sim.n_events), "wall_time_s": getattr(sim, "wall_time", None)}


def _tubule_args(mission: dict, out: Path, scenario: str | None):
    m = apply_scenario(mission, scenario)
    cfg = mission_cfg(m)
    return m, cfg, [(cfg, tubule_chamber(m, t), int(m.get("seed", 0)) + i, out) for i, t in enumerate(m["tubules"])]


def final_state_path(out: Path, name: str) -> Path:
    return Path(out) / "tubules" / f"{name}.pkl"


def run_tubule(mission: dict, out: Path, scenario: str | None, index: int) -> Path:
    """Run one tubule of a scenario (the same configuration and seed as in run_mission) and save its final state, so
    a campaign can schedule tubules one by one; assemble_mission writes the scenario's outputs from the four."""
    import pickle
    out.mkdir(parents=True, exist_ok=True)
    m, cfg, args = _tubule_args(mission, out, scenario)
    s = _run_one(args[index])
    path = final_state_path(out, s.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def assemble_mission(mission: dict, out: Path, scenario: str | None = None, write_outputs=True):
    """The scenario's outputs from its tubules' saved final states (run_tubule)."""
    import pickle
    m, cfg, args = _tubule_args(mission, out, scenario)
    sims = []
    for a in args:
        with open(final_state_path(out, a[1]["name"]), "rb") as f:
            sims.append(pickle.load(f))
    return _write_mission(m, cfg, out, sims, write_outputs)


def run_mission(mission: dict, out: Path, scenario: str | None = None, jobs: int = 1, write_outputs=True):
    out.mkdir(parents=True, exist_ok=True)
    m, cfg, args = _tubule_args(mission, out, scenario)
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            sims = list(ex.map(_run_one, args))
    else:
        sims = [_run_one(a) for a in args]
    return _write_mission(m, cfg, out, sims, write_outputs)


def _write_mission(m, cfg, out, sims, write_outputs):
    prof = FlightProfile.from_config(m.get("flight_profile"))
    summary = {"name": m.get("name"), "scenario": m.get("scenario"), "flight_profile": prof.to_dict(),
               "gas": m.get("gas"), "launch_loads": launch_report(m),
               "tubules": [summarize(s, prof) for s in sims]}
    if write_outputs:
        from .design import preflight
        from .output import write_chamber, plot_all, write_viewer, _write_csv
        pfs = []
        for s, t in zip(sims, m["tubules"]):
            ch = tubule_chamber(m, t)
            pf = preflight(cfg, ch)
            pfs.append(pf)
            # the camera analysis and the standard outputs use the microgravity window only; impacts before it
            # (boost and coast) and after it (descent below ~100 km) go to their own files
            t0 = s.env.t0_flight
            launch = [c for c in s.collisions if c["t"] + t0 < prof.ug_start_s]
            descent = [c for c in s.collisions if c["t"] + t0 >= prof.ug_end_s]
            s.collisions = [c for c in s.collisions if _in_window(c["t"], t0, prof)]
            (out / s.name).mkdir(parents=True, exist_ok=True)
            _write_csv(out / s.name / "launch_collisions.csv", launch)
            _write_csv(out / s.name / "descent_collisions.csv", descent)
            _write_csv(out / s.name / "detachments.csv", s.detach_log)
            cam = dict(cfg.get("camera", {}))
            cam.update(ch.get("camera", {}))
            write_chamber(s, out / s.name, pf, cam)
        try:
            plot_all(sims, out, pfs)
        except Exception as e:  # pragma: no cover - plotting is optional
            summary["plot_error"] = repr(e)
        write_viewer(sims, out / "viewer.html", f"DOLL-E mission: {m.get('name')} ({m.get('scenario') or 'nominal'})")
    from .results import jsonable
    (out / "mission_summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    return summary, sims

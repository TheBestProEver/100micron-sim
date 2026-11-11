"""Ready-made scenes for the 3-D simulation: the two-tubule comparison, one impact up close, the browser sandbox.

Two tubules (two_tubule_start, run_two_tubules).  Two identical 0.5 x 0.5 x 1.5 in borosilicate tubules side by
side in microgravity, one with silica grains and one with graphite grains, started from the SAME sizes, positions,
velocities and adhesion factors.  Every input is listed in START_SOURCES with its source or marked ASSUMED.  The
one big assumption is the start: a compact cloud (a 2 mm cube at the tubule centre).  Grains spread through the
whole tubule practically never meet (mean free path ~2 m against walls ~6 mm away), so clumps form only from a
compact cloud or with many more grains; how DOLL-E releases its dust is not decided.

One impact up close (impact_closeup).  Two 50 um radius grains, head-on, in the centre-of-mass frame: the trace of
collide.py's impact (force, overlap, patch radius, velocities, energies at every sub-step), for the page's slow
motion view.

Sandbox (sandbox_start, sandbox_advance).  Small runs chosen in the web page and computed in the browser
(Pyodide); they exchange JSON text.

The momentum ledger (MomentumRecorder).  One row per frame: the momentum of the flying grains, what the walls have
received and what the residual acceleration has given; the page plots the tubule's total, which stays constant.

Plain python + numpy (statistics.NormalDist replaces scipy), so this runs under Pyodide too.
"""
from __future__ import annotations

import json
import math
from statistics import NormalDist

import numpy as np

from .clumps import momentum
from .collide import TRACE_KEYS, impact, sticking_threshold
from .materials import G0
from .pair import make_pair
from .sim3d import Sim, Tubule, frame, make_sim, run, summary

START_SOURCES = {
    "tubule": "0.5 x 0.5 x 1.5 in inner, borosilicate, standing (long axis z = rocket axis), wall adhesion factor "
              "1.0: configs/mission_quarter_deck.json (T1, T2)",
    "materials": "silica in one tubule, graphite in the other; values from dolle_core/materials.py",
    "n_grains": "100 per tubule: the value of the full simulator's flight campaign; DOLL-E's dust amount is not "
                "decided",
    "sizes": "diameters from the 90-106 um sieve range with 90 % inside, log-normal centred on the band (the "
             "log-normal shape is ASSUMED, as in dolle_sim/sizes.py); the SAME sample in both tubules",
    "adhesion_scatter": "factor exp(0.6 N(0,1)) per grain on the work of adhesion; pairs sqrt(f_i f_j), wall f_i: "
                        "configs/mission_quarter_deck.json surface_energy_log_sigma, from the ~70 % scatter of AFM "
                        "adhesion on clean glass (Rush et al. 2018, PROVENANCE.md)",
    "positions": "ASSUMED: a compact released cloud, uniform in a 2 mm cube at the tubule centre, centres at "
                 "least 1.06 x the sum of radii apart (the full simulator's spacing rule)",
    "velocities": "ASSUMED: isotropic directions, speeds |N(15, 7.5)| mm/s, no spin. 15 mm/s is the geometric mean "
                  "of the two grain-grain sticking speeds (6.2 and 38 mm/s), so the same start lies above "
                  "silica's threshold and below graphite's",
    "residual_acceleration": "1e-4 g along +z (toward the nose): ASSUMED order of magnitude (PROVENANCE.md)",
    "time_step": "4 ms: configs/mission_quarter_deck.json dt_s",
    "friction": "per pair from dolle_core/materials.py FRICTION (silica-silica 0.6, silica-wall 0.4, graphite 0.5)",
    "left_out": "gas drag, the payload's 1 rpm roll, contact charging, wall vibration: notebook 06 section 9 gives "
                "each one's size",
}


# ---------------------------------------------------------------------------- the start
def sieve_radii(n: int, d_min: float = 90e-6, d_max: float = 106e-6, fraction: float = 0.9, rng=None) -> np.ndarray:
    """n radii [m] from a sieve cut [d_min, d_max] (diameters) with `fraction` of the grains inside: a log-normal
    centred on sqrt(d_min d_max) with ln-sigma s = ln(d_max/d_min) / (2 z), z the standard-normal quantile of
    (1 + fraction)/2.  Same rule as dolle_sim/sizes.py sieve_lognormal (the log-normal shape is ASSUMED there)."""
    rng = np.random.default_rng() if rng is None else rng
    mu = 0.5 * math.log(d_min * d_max)
    s = math.log(d_max / d_min) / (2 * NormalDist().inv_cdf(0.5 * (1 + fraction)))
    return 0.5 * np.exp(mu + s * rng.standard_normal(n))


def puff_positions(radii, centre, box, rng, spacing: float = 1.06, tries: int = 20000) -> np.ndarray:
    """Random sequential placement of grain centres, uniform in a box around centre (box: the side of a cube [m],
    or three side lengths), each pair at least spacing x (r_i + r_j) apart (ValueError if the box is too full)."""
    radii, centre = np.asarray(radii, float), np.asarray(centre, float)
    out = np.zeros((len(radii), 3))
    for k, rk in enumerate(radii):
        for _ in range(tries):
            p = centre + (rng.random(3) - 0.5) * box
            if k == 0 or np.all(((out[:k] - p) ** 2).sum(1) >= (spacing * (radii[:k] + rk)) ** 2):
                out[k] = p
                break
        else:
            raise ValueError(f"cannot place {len(radii)} grains in a box of {np.asarray(box) * 1e3} mm")
    return out


def isotropic_velocities(n: int, mean: float, sigma: float, rng) -> np.ndarray:
    """n velocities [m/s]: uniformly random directions, speeds |N(mean, sigma)| (v0.5's "isotropic" mode)."""
    speed = np.abs(mean + sigma * rng.standard_normal(n))
    d = rng.standard_normal((n, 3))
    return d / np.linalg.norm(d, axis=1)[:, None] * speed[:, None]


def two_tubule_start(n: int = 100, seed: int = 7, box_mm: float = 2.0, speed_mm_s: float = 15.0,
                     sigma_mm_s: float = 7.5, log_sigma_w: float = 0.6, residual_g: float = 1e-4,
                     tubule: Tubule | None = None) -> dict:
    """The common start of both tubules (START_SOURCES): radii [m], positions [m], velocities [m/s], adhesion
    factors, and the Tubule.  seed 7 is the mission config's seed."""
    rng = np.random.default_rng(seed)
    tub = tubule or Tubule(accel=(0.0, 0.0, residual_g * G0))
    radii = sieve_radii(n, rng=rng)
    adhesion = np.exp(log_sigma_w * rng.standard_normal(n))
    pos = puff_positions(radii, np.asarray(tub.size) / 2, box_mm * 1e-3, rng)
    vel = isotropic_velocities(n, speed_mm_s * 1e-3, sigma_mm_s * 1e-3, rng)
    return {"tubule": tub, "radii": radii, "positions": pos, "velocities": vel, "adhesion": adhesion,
            "settings": {"n": n, "seed": seed, "box_mm": box_mm, "speed_mm_s": speed_mm_s, "sigma_mm_s": sigma_mm_s,
                         "log_sigma_w": log_sigma_w, "residual_g": residual_g}}


def run_tubule(start: dict, material: str, t_end: float = 12.0, **run_kw) -> Sim:
    """Run one tubule of `material` from `start` (two_tubule_start) until t_end [s] or all grains are on walls."""
    sim = make_sim(start["tubule"], material, start["radii"], start["positions"], start["velocities"],
                   adhesion=start["adhesion"])
    return run(sim, t_end, **run_kw)


def run_two_tubules(start: dict, t_end: float = 12.0, materials=("silica", "graphite"), **run_kw) -> dict:
    """Both tubules from the same start: {material: Sim}."""
    return {m: run_tubule(start, m, t_end, **run_kw) for m in materials}


def thresholds(material: str, wall: str = "glass_borosilicate", r: float = 50e-6) -> dict:
    """Sticking speeds [m/s] of two r grains of `material` and of one on the wall (collide.sticking_threshold)."""
    return {"grain_grain": sticking_threshold(make_pair(material, r, material, r)),
            "grain_wall": sticking_threshold(make_pair(material, r, wall, None))}


# ---------------------------------------------------------------------------- JSON for the page
EVENT_COLUMNS = ("t_s", "kind", "i", "j", "outcome", "v_n_mm_s", "v_t_mm_s", "e_n", "n_i", "n_j", "x_um", "y_um",
                 "z_um", "fragments", "mode", "dP_rel", "dL_rel")
"""One row per event.  kind 0 grain-grain, 1 grain-wall; outcome 1 stick, 0 bounce; n_i, n_j grains in the two
clumps (n_j = 0 for a wall); x, y, z the contact point; mode "", "erosion" or "shatter"; dP_rel, dL_rel the
event's momentum and angular-momentum errors relative to the momenta involved."""


def event_rows(events) -> list:
    """Event records as compact rows (EVENT_COLUMNS)."""
    return [[round(e["t"], 6), 0 if e["kind"] == "pp" else 1, e["i"], e["j"], int(e["outcome"] == "stick"),
             round(e["v_n"] * 1e3, 4), round(e["v_t"] * 1e3, 4), round(e["e_n"], 4), e["n_i"], e["n_j"],
             round(e["x"] * 1e6, 1), round(e["y"] * 1e6, 1), round(e["z"] * 1e6, 1), e["fragments"], e["mode"],
             float(f"{e['dP_rel']:.2e}"), float(f"{e['dL_rel']:.2e}")] for e in events]


def frame_columns(frames) -> dict:
    """Frames as columns: t [s]; pos (flat x0, y0, z0, x1, ... integer um); clump ids; wall flags as a 0/1 string;
    colour indices."""
    return {"t": [round(f["t"], 4) for f in frames], "pos": [f["pos_um"].ravel().tolist() for f in frames],
            "clump": [f["clump"].tolist() for f in frames], "wall": ["".join(map(str, f["wall"])) for f in frames],
            "colour": [f["colour"].tolist() for f in frames]}


def tubule_replay(sim: Sim, name: str, recorder: "MomentumRecorder | None" = None) -> dict:
    """Everything the page needs for one tubule: sizes, frames, events, summary, sticking speeds, and (if the run
    had a MomentumRecorder) the momentum row of every frame with its scale p_scale [1e-12 kg m/s]."""
    c = sim.clumps
    mats = sorted(set(c.material))
    out = {"name": name, "material": mats[0] if len(mats) == 1 else mats, "wall": sim.tubule.wall,
           "size_um": [round(s * 1e6, 1) for s in sim.tubule.size], "radii_um": [round(r * 1e6, 2) for r in c.r],
           "frames": frame_columns(sim.frames), "event_columns": list(EVENT_COLUMNS),
           "events": event_rows(sim.events), "summary": _rounded(summary(sim)),
           "sticking_speeds_mm_s": {k: round(v * 1e3, 3) for k, v in thresholds(mats[0], sim.tubule.wall).items()}}
    if recorder is not None:
        out.update({"momentum_columns": list(MOMENTUM_COLUMNS), "momentum": recorder.rows,
                    "p_scale": float(f"{recorder.p_scale * 1e12:.6g}")})
    return out


# ---------------------------------------------------------------------------- the momentum ledger, per frame
MOMENTUM_COLUMNS = ("t_s", "Pg_x", "Pg_y", "Pg_z", "Pw_x", "Pw_y", "Pw_z", "Pa_x", "Pa_y", "Pa_z", "dev_rel")
"""One row per frame; momenta in units of 1e-12 kg m/s (= ng mm/s).  Pg: the flying grains and clumps
(clumps.momentum); Pw: the total impulse the walls have received (Sim.tubule_P); Pa: the total impulse the residual
acceleration has given (Sim.external_P).  The tubule's total momentum Pg + Pw - Pa stays equal to its start value
P0: every impact gives the two bodies equal and opposite impulses (Newton's third law), and the walls' and the
acceleration's shares are booked exactly.  dev_rel = |Pg + Pw - Pa - P0| / S0, with S0 = sum of M |V| over the
clumps at the start (p_scale): rounding error only (~1e-14)."""


def momentum_scale(sim: Sim) -> float:
    """S0 = sum over the free clumps of M |V| [kg m/s]: the scale the momentum error is measured against."""
    c = sim.clumps
    idx = np.flatnonzero(c.alive & ~c.frozen)
    return float(np.sum(c.M[idx] * np.linalg.norm(c.V[idx], axis=1)))


class MomentumRecorder:
    """on_frame callback for sim3d.run: stores one MOMENTUM_COLUMNS row per frame in .rows.  Make it before the run
    starts: it takes S0 (.p_scale, kg m/s) from the sim as it is then."""

    def __init__(self, sim: Sim):
        self.p_scale = momentum_scale(sim)
        self.rows = []

    def __call__(self, sim: Sim, f: dict) -> None:
        Pg, _ = momentum(sim.clumps)
        dev = float(np.linalg.norm(Pg + sim.tubule_P - sim.external_P - sim.P0)) / max(self.p_scale, 1e-300)
        vals = np.concatenate((Pg, sim.tubule_P, sim.external_P)) * 1e12
        self.rows.append([round(f["t"], 4)] + [float(f"{v:.7g}") for v in vals] + [float(f"{dev:.3g}")])


def _rounded(d):
    """Round floats in nested dicts for compact JSON (4 significant figures)."""
    if isinstance(d, dict):
        return {k: _rounded(v) for k, v in d.items()}
    if isinstance(d, float):
        return float(f"{d:.4g}")
    return d


def replay_json(replay: dict) -> str:
    """Compact JSON text (no spaces)."""
    return json.dumps(replay, separators=(",", ":"))


# ---------------------------------------------------------------------------- one impact up close
CLOSEUP_COLUMNS = ("t_us", "overlap_nm", "a_um", "F_el_uN", "F_damp_uN", "v1_mm_s", "v2_mm_s", "momentum_rel",
                   "kinetic_fJ", "stored_fJ", "damping_fJ", "hysteresis_fJ", "plastic_fJ", "energy_error_fJ")


def impact_closeup(material: str, v_in: float, r: float = 50e-6, max_rows: int = 400) -> dict:
    """Head-on impact of two grains of radius r [m] at approach speed v_in [m/s] in the centre-of-mass frame
    (collide.impact with record=True), decimated to <= max_rows rows (CLOSEUP_COLUMNS: us, nm, um, uN, mm/s, fJ;
    momentum_rel = total momentum / (m1 |u1|), which stays 0 to rounding).  The approach before touch has no
    force in this model (the JKR contact snaps on at touch); the page animates it from 'approach'."""
    pair = make_pair(material, r, material, r)
    res = impact(pair, v_in, record=True)
    tr = res.trace
    n = len(tr["t"])
    idx = np.unique(np.linspace(0, n - 1, min(n, max_rows)).round().astype(int))
    p_scale = pair.m1 * abs(res.u1)
    cols = {"t_us": tr["t"] * 1e6, "overlap_nm": tr["overlap"] * 1e9, "a_um": tr["contact_radius"] * 1e6,
            "F_el_uN": tr["force"] * 1e6, "F_damp_uN": tr["damping_force"] * 1e6, "v1_mm_s": tr["v1"] * 1e3,
            "v2_mm_s": tr["v2"] * 1e3, "momentum_rel": tr["momentum"] / p_scale, "kinetic_fJ": tr["kinetic"] * 1e15,
            "stored_fJ": tr["stored"] * 1e15, "damping_fJ": tr["damping_loss"] * 1e15,
            "hysteresis_fJ": tr["hysteresis_loss"] * 1e15, "plastic_fJ": tr["plastic_loss"] * 1e15,
            "energy_error_fJ": tr["energy_error"] * 1e15}
    rows = [[float(f"{cols[k][i]:.5g}") for k in CLOSEUP_COLUMNS] for i in idx]
    from .jkr import equilibrium_contact_radius, pull_off_force
    from .collide import contact_law
    law = contact_law(pair)
    v_star = sticking_threshold(pair)
    E_in = 0.5 * pair.m_star * v_in ** 2
    return {"material": material, "radius_um": r * 1e6, "v_in_mm_s": v_in * 1e3, "v_stick_mm_s": v_star * 1e3,
            "outcome": res.outcome, "e": res.e, "contact_time_us": res.contact_time * 1e6,
            "max_overlap_nm": res.max_overlap * 1e9, "max_force_uN": res.max_force * 1e6, "yielded": res.yielded,
            "u1_mm_s": res.u1 * 1e3, "u2_mm_s": res.u2 * 1e3, "v1_mm_s": res.v1 * 1e3, "v2_mm_s": res.v2 * 1e3,
            "pair": {"R_star_um": pair.R_star * 1e6, "E_star_GPa": pair.E_star / 1e9, "w_J_m2": pair.w,
                     "m_star_kg": pair.m_star, "a0_um": equilibrium_contact_radius(pair.R_star, pair.w,
                                                                                   pair.E_star) * 1e6,
                     "F_c_uN": pull_off_force(pair.R_star, pair.w) * 1e6, "delta_break_nm": law.delta_break * 1e9,
                     "U0_fJ": law.U0 * 1e15},
            "E_in_fJ": E_in * 1e15,
            "momentum_error_rel": float(np.max(np.abs(tr["momentum"] - tr["momentum"][0])) / p_scale),
            "energy_error_rel": float(np.max(np.abs(tr["energy_error"])) / E_in),
            "approach": {"gap_start_um": 0.5 * r * 1e6, "duration_s": 0.5 * r / v_in},
            "columns": list(CLOSEUP_COLUMNS), "rows": rows}


def closeup_cases(materials=("silica", "graphite"), factors=(0.8, 1.5)) -> list:
    """The page's four close-ups: per material, factors x the sticking speed (0.8: sticks, 1.5: bounces).  If a
    stick is captured at the first touch (no contact phase to show), 0.9 x is used instead."""
    out = []
    for m in materials:
        v_star = sticking_threshold(make_pair(m, 50e-6, m, 50e-6))
        for f in factors:
            case = impact_closeup(m, f * v_star)
            if len(case["rows"]) < 3 and f < 1:
                f = 0.9
                case = impact_closeup(m, f * v_star)
            case["factor_of_v_stick"] = f
            out.append(case)
    return out


# ---------------------------------------------------------------------------- the browser sandbox
KICK_SPEED_M_S = 5000.0 * G0 / (2 * math.pi * 5000.0)
"""1.56 m/s: the velocity amplitude a/(2 pi f) of a wall moving sinusoidally at the peak acceleration a = 5000 g and
f = 5000 Hz of the release kick in configs/mission_quarter_deck.json ("kick" scenario).  Used as the typical speed of
a kicked grain; how fast grains really leave the wall depends on when they let go: ASSUMED."""

SANDBOX_DEFAULTS = {"materials": ["silica", "graphite"], "n": 30, "box_mm": 1.2, "speed_mm_s": 15.0,
                    "sigma_mm_s": 7.5, "residual_g": 1e-4, "log_sigma_w": 0.6, "seed": 1, "preset": "cloud",
                    "t_end": 12.0, "d_min_um": 90.0, "d_max_um": 106.0}
"""Presets: "cloud" (a compact cloud at the centre), "spread" (grains through the whole tubule: they practically
never meet), "kick" (mean speed KICK_SPEED_M_S, spread 10 %: nothing sticks; not offered on the page).
d_min_um, d_max_um: the sieve cut of the grain diameters [um] (sieve_radii; 90-106 um is DOLL-E's)."""


def sandbox_start(params_json: str = "{}") -> dict:
    """Build the sandbox runs (one Sim per material, the same start) from JSON parameters (SANDBOX_DEFAULTS)."""
    p = dict(SANDBOX_DEFAULTS)
    p.update(json.loads(params_json or "{}"))
    n = int(min(max(p["n"], 2), 60))
    tub = Tubule(accel=(0.0, 0.0, float(p["residual_g"]) * G0))
    rng = np.random.default_rng(int(p["seed"]))
    radii = sieve_radii(n, float(p["d_min_um"]) * 1e-6, float(p["d_max_um"]) * 1e-6, rng=rng)
    adhesion = np.exp(float(p["log_sigma_w"]) * rng.standard_normal(n))
    size = np.asarray(tub.size)
    if p["preset"] == "spread":
        pos = puff_positions(radii, size / 2, size - 4 * radii.max(), rng)
    else:
        pos = puff_positions(radii, size / 2, float(p["box_mm"]) * 1e-3, rng)
    speed = KICK_SPEED_M_S * 1e3 if p["preset"] == "kick" else float(p["speed_mm_s"])
    sigma = 0.1 * speed if p["preset"] == "kick" else float(p["sigma_mm_s"])
    vel = isotropic_velocities(n, speed * 1e-3, sigma * 1e-3, rng)
    sims = [make_sim(tub, m, radii, pos, vel, adhesion=adhesion) for m in p["materials"]]
    return {"params": p, "sims": sims, "sent_frames": [0] * len(sims), "sent_events": [0] * len(sims),
            "momentum": [MomentumRecorder(s) for s in sims]}


def sandbox_advance(state: dict, seconds: float = 0.1) -> str:
    """Run every sandbox Sim on by `seconds` (frames every step for the first 0.5 s, then 30 per second) and return
    JSON with the new frames, events and momentum rows (MOMENTUM_COLUMNS) of each, its counts, and whether it is
    finished."""
    out = []
    for k, sim in enumerate(state["sims"]):
        rec = state["momentum"][k]
        t_end = min(sim.t + seconds, float(state["params"]["t_end"]))
        if sim.t < t_end - 1e-12 and len(np.flatnonzero(sim.clumps.alive & ~sim.clumps.frozen)):
            run(sim, t_end, on_frame=rec, final_frame=t_end >= float(state["params"]["t_end"]) - 1e-9)
        f0, e0 = state["sent_frames"][k], state["sent_events"][k]
        state["sent_frames"][k], state["sent_events"][k] = len(sim.frames), len(sim.events)
        done = sim.t >= float(state["params"]["t_end"]) - 1e-9 or not (sim.clumps.alive & ~sim.clumps.frozen).any()
        out.append({"material": sim.clumps.material[0], "frames": frame_columns(sim.frames[f0:]),
                    "events": event_rows(sim.events[e0:]), "counts": dict(sim.counts), "t": sim.t, "done": bool(done),
                    "momentum": rec.rows[f0:], "p_scale": float(f"{rec.p_scale * 1e12:.6g}"),
                    "radii_um": [round(r * 1e6, 2) for r in sim.clumps.r] if f0 == 0 else None})
    return json.dumps({"tubules": out, "event_columns": list(EVENT_COLUMNS),
                       "momentum_columns": list(MOMENTUM_COLUMNS),
                       "size_um": [round(s * 1e6, 1) for s in state["sims"][0].tubule.size]}, separators=(",", ":"))

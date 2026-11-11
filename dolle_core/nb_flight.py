"""Helpers for the notebooks 00_start_here, 04_grain_on_a_wall and 05_the_flight_in_brief.

These functions only collect, tabulate and plot numbers.  The physics they use is in the other dolle_core modules
(wall.py: the limits of a grain on a wall; collide.py: the sticking speed) and, for the flight loads, in the full
simulator's dolle_sim/flight.py (FlightProfile), which the notebooks create and pass in as `prof`.

This module is NOT imported by `import dolle_core` (use `from dolle_core import nb_flight`), so the core itself stays
independent of dolle_sim.  The two functions that run the full simulator (core_vs_sim, dem_rolling_ramp) import it
only when they are called.

Sign conventions (those of dolle_sim/flight.py).  Tubule frame: +z along the rocket axis toward the nose; the aft
face of a standing tubule is z = 0 and the dust rests on it from the pad.  prof.apparent_accel(t) is the acceleration
a free grain appears to have in the tubule frame: minus the payload's specific force (what an accelerometer reads).
So during a burn it is -25 g along z (pressing the dust onto the aft face) and in the coast drag it is about +3 g
(pulling the dust off the aft face).  All accelerations here are in units of g0 = 9.80665 m/s^2.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .collide import sticking_threshold
from .materials import G0
from .pair import make_pair
from .wall import rolling_acceleration, wall_limits

INCH = 0.0254


# ---------------------------------------------------------------------------------------------------- tables
def print_table(rows, cols, title=None):
    """Print a list of dicts as an aligned text table.  cols: list of (key, header, format) with format a
    format spec such as ".3g" or "" (str)."""
    cells = [[format(r[k], f) if f else str(r[k]) for k, _, f in cols] for r in rows]
    heads = [h for _, h, _ in cols]
    width = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(heads)]
    if title:
        print(title)
    print("  ".join(h.ljust(w) for h, w in zip(heads, width)))
    print("  ".join("-" * w for w in width))
    for c in cells:
        print("  ".join(x.ljust(w) for x, w in zip(c, width)))


# ---------------------------------------------------------------------------------------------------- tubules
_EXAMPLE_TUBULES = [
    # copy of the "tubules" of configs/mission_quarter_deck.json (used only when that file is not on disk, e.g. in
    # Colab, where pip installs the packages but not the configs).  Materials, sizes and positions are EXAMPLES.
    dict(name="T1_silica_90-106um_bare", dust="silica", wall="glass_borosilicate", d_min=9.0e-5, d_max=1.06e-4,
         deck_position_m=(-0.080, 0.050), inner_size_in=(0.5, 0.5, 1.5), wall_adhesion_factor=1.0, log_sigma=0.6),
    dict(name="T2_graphite_90-106um_bare", dust="graphite", wall="glass_borosilicate", d_min=9.0e-5, d_max=1.06e-4,
         deck_position_m=(-0.030, 0.065), inner_size_in=(0.5, 0.5, 1.5), wall_adhesion_factor=1.0, log_sigma=0.6),
    dict(name="T3_silica_45-53um_silanized", dust="silica", wall="glass_silanized", d_min=4.5e-5, d_max=5.3e-5,
         deck_position_m=(0.030, 0.065), inner_size_in=(0.5, 0.5, 1.5), wall_adhesion_factor=1.0, log_sigma=0.6),
    dict(name="T4_graphite_45-53um_silanized", dust="graphite", wall="glass_silanized", d_min=4.5e-5, d_max=5.3e-5,
         deck_position_m=(0.080, 0.050), inner_size_in=(0.5, 0.5, 1.5), wall_adhesion_factor=1.0, log_sigma=0.6),
]


def example_tubules(config_path=None):
    """The four example tubules of the mission config (dust, wall, sieve DIAMETERS d_min..d_max [m], deck position
    [m] from the spin axis, inner size [in], adhesion factor, log-normal scatter sigma of ln w).  Read from
    configs/mission_quarter_deck.json when the repository is checked out (or from config_path), else the copy in
    this file.  Returns (tubules, source), source as a path relative to the repository (never an absolute path, so
    printed outputs are the same on every machine) or "embedded copy"."""
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "configs" / \
        "mission_quarter_deck.json"
    if not path.exists():
        return [dict(t) for t in _EXAMPLE_TUBULES], "embedded copy in dolle_core/nb_flight.py (config file not found)"
    out = []
    for t in json.loads(path.read_text())["tubules"]:
        d, sd = t["dust"], t["dust"]["size_distribution"]
        out.append(dict(name=t["name"], dust=d["material"], wall=t.get("wall_material", "glass_borosilicate"),
                        d_min=float(sd["min_m"]), d_max=float(sd["max_m"]),
                        deck_position_m=tuple(t.get("deck_position_m", (0.05, 0.0))),
                        inner_size_in=tuple(t.get("inner_size_in", (0.5, 0.5, 1.5))),
                        wall_adhesion_factor=float(t.get("wall_adhesion_factor", 1.0)),
                        log_sigma=float(d.get("surface_energy_log_sigma", 0.0))))
    try:
        shown = path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        shown = path.name
    return out, shown


_EXAMPLE_CAMERA = dict(fps=500, position_noise_m=5e-06, track_frames=6)   # copy of the config's "camera" (EXAMPLE values)


def example_camera(config_path=None):
    """The example camera of the mission config: frame rate fps [1/s], position_noise_m [m] (centroid error per
    frame) and track_frames (frames per velocity fit).  EXAMPLE values, read from configs/mission_quarter_deck.json
    when it is on disk, else the copy in this file."""
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "configs" /         "mission_quarter_deck.json"
    if not path.exists():
        return dict(_EXAMPLE_CAMERA)
    cam = json.loads(path.read_text())["camera"]
    return {k: cam[k] for k in _EXAMPLE_CAMERA}


def far_corner(tubule):
    """Corner of a standing tubule's aft face farthest from the spin axis [m] (deck plane), where the centrifugal
    load is largest: the point dolle_sim/mission.py launch_report uses."""
    pos = np.asarray(tubule["deck_position_m"], float)
    w, h = np.asarray(tubule["inner_size_in"][:2], float) * INCH
    corners = [pos + np.array([sx * w / 2, sy * h / 2]) for sx in (-1, 1) for sy in (-1, 1)]
    return max(corners, key=lambda c: float(np.hypot(*c)))


# ---------------------------------------------------------------------------------------------------- notebook 04
def limits_vs_diameter(grain, wall, diameters, f_adh=1.0, h=0.5, press_g=0.0):
    """dolle_core.wall_limits for grains of each DIAMETER in `diameters` [m] (radius = d/2).  Returns a dict of
    arrays: d [m], pull_off_g, slide_g, roll_g [g] and twist_angular_acc [rad/s^2]."""
    L = [wall_limits(grain, 0.5 * d, wall, f_adh=f_adh, h=h, press_g=press_g) for d in diameters]
    return dict(d=np.asarray(diameters, float), pull_off_g=np.array([x.pull_off_g for x in L]),
                slide_g=np.array([x.slide_g for x in L]), roll_g=np.array([x.roll_g for x in L]),
                twist_angular_acc=np.array([x.twist_angular_acc for x in L]))


def plot_limits_vs_diameter(cases, loads, d_range=(45e-6, 106e-6), ax=None, sliding=False):
    """Log-log plot of the pull-off (solid) and rolling (dashed) limits [g] against grain diameter [um], and the
    sliding limit (dotted) if sliding=True.  cases: list of (label, grain, wall, colour); loads: list of
    (label, g) drawn as horizontal grey lines, labelled alternately at the right and left ends."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5.5))
    d = np.geomspace(*d_range, 40)
    for label, grain, wall, colour in cases:
        L = limits_vs_diameter(grain, wall, d)
        ax.plot(d * 1e6, L["pull_off_g"], "-", color=colour, lw=2, label=f"{label}: pull-off")
        if sliding:
            ax.plot(d * 1e6, L["slide_g"], ":", color=colour, lw=1.5, label=f"{label}: sliding")
        ax.plot(d * 1e6, L["roll_g"], "--", color=colour, lw=2, label=f"{label}: rolling")
    for k, (label, g) in enumerate(sorted(loads, key=lambda x: x[1])):
        ax.axhline(g, color="0.45", lw=0.8)
        right = k % 2 == 0
        ax.text(d_range[1 if right else 0] * 1e6 * (0.99 if right else 1.01), g * 1.06, label,
                ha="right" if right else "left", va="bottom", fontsize=8, color="0.2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grain diameter  d  [µm]")
    ax.set_ylabel("acceleration that reaches the limit  [g]")
    ax.set_xticks([45, 53, 60, 70, 80, 90, 106])
    ax.set_xticklabels(["45", "53", "60", "70", "80", "90", "106"])
    ax.minorticks_off()
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def dem_rolling_ramp(t_end=2e-3, n_out=100):
    """Run the full simulator's 3-D contact model (extras/dolle_extras/dem.py DEMSystem) with exactly the set-up of its
    check verify_full.check_rolling_on_wall: a 50 um radius silica grain resting on a borosilicate wall (z = 0),
    pressed on it by 1 g, under a sideways body acceleration ramped linearly from 0 to 3x its rolling limit in t_end
    [s], dt = 1e-8 s.  The run is split into n_out pieces to record the grain's speed v [m/s] and omega R [m/s] (the
    pieces continue one run: same steps, same forces).  Returns a dict with t, v, omega_R, ramp [m/s^2], a_roll_sim
    (the check's rolling limit, from dem.make_params at zero load) [m/s^2]; or None if numba or extras/ is missing (the
    pure-python fallback would take many minutes)."""
    import importlib.util
    if importlib.util.find_spec("numba") is None:
        print("numba is not installed: skipping the 3-D run (pip install numba to run it).")
        return None
    try:
        from dolle_extras import dem              # the optional v0.6 engine (extras/; the notebooks put it on the path)
    except ImportError:
        print("the optional extras/ folder is not in this copy: skipping the 3-D run (use a clone of the repository).")
        return None
    from dolle_sim.materials import get_material, sphere_mass
    s, g = get_material("silica"), get_material("glass_borosilicate")
    R = 50e-6
    m = sphere_mass(s, R)
    wall = dict(E=g.E, nu=g.nu, gamma=g.gamma, e_dry=g.e_dry, p_y=g.yield_pressure)
    p = dem.make_params(R, 0.0, m, math.inf, s.E, s.nu, g.E, g.nu, 2 * math.sqrt(s.gamma * g.gamma), 0.9, math.inf,
                        0.4, 0.5, 100.0, 100.0, dem.DAMP_TSUJI_JKR, dem.BREAK_DISPLACEMENT, 1.0, 0.0)
    a_roll = p[dem.P_MUR] * 2 * p[dem.P_FC] / m
    z0 = R - p[dem.P_A0] ** 2 / (3 * R)
    S = dem.DEMSystem([[0.5, 0.5, z0]], [[0, 0, 0]], [[0, 0, 0]], R, s.rho, s.E, s.nu, s.gamma, s.e_dry,
                      s.yield_pressure, [1.0, 1.0, 1.0], wall, mu_pw=0.4, h_roll=0.5, q_roll=100, q_twist=100,
                      wall_faces=(4,))
    dt = 1e-8
    n = int(t_end / dt)
    ramp = 3 * a_roll / t_end
    t = (np.arange(n) + 1) * dt
    acc = np.column_stack([ramp * t, np.zeros(n), np.full(n, -G0)])
    edges = np.linspace(0, n, n_out + 1).astype(int)
    ts, vs, ws = [0.0], [0.0], [0.0]
    for i0, i1 in zip(edges[:-1], edges[1:]):
        S.run(dt, i1 - i0, acc=acc[i0:i1])
        ts.append(i1 * dt)
        vs.append(float(S.v[0, 0]))
        ws.append(float(S.w[0, 1] * R))
    return dict(t=np.array(ts), v=np.array(vs), omega_R=np.array(ws), ramp=ramp, a_roll_sim=a_roll, R=R)


def rigid_plastic_rolling_speed(t, ramp, a_roll):
    """Speed [m/s] of a grain under a sideways acceleration ramp*t [m/s^2] once it passes the rolling limit a_roll
    [m/s^2], rolling without slip: dv/dt = (ramp t - a_roll)/1.4 (wall.rolling_acceleration), integrated from the
    onset t_on = a_roll/ramp: v = (ramp (t^2 - t_on^2)/2 - a_roll (t - t_on))/1.4; zero before the onset."""
    t = np.asarray(t, float)
    t_on = a_roll / ramp
    v = (0.5 * ramp * (t ** 2 - t_on ** 2) - a_roll * (t - t_on)) / 1.4
    return np.where(t > t_on, v, 0.0)


# ---------------------------------------------------------------------------------------------------- flight loads
def aft_face_loads(prof, r_xy, n=200):
    """For every phase of prof.phases(): the largest apparent load pulling dust off the aft face (+z) and pressing
    it on (-z), and the largest load along the face (centrifugal + Euler + the sideways apparent acceleration), all
    in g, at the point r_xy [m] of the deck plane.  Sampled at n points strictly inside each phase, as
    dolle_sim/mission.py launch_report does (which also adds the guide's ~10 g lateral ascent envelope to the
    along-face load; here that is left out and quoted separately)."""
    rows = []
    for ph in prof.phases():
        ts = ph["t0"] + (np.arange(n) + 0.5) / n * (ph["t1"] - ph["t0"])
        tot = np.array([prof.loads_at(t, np.asarray(r_xy, float))["total"] for t in ts])
        along = np.hypot(tot[:, 0], tot[:, 1])
        rows.append(dict(phase=ph["name"], t0=ph["t0"], t1=ph["t1"], pull_off_aft_g=float(max(tot[:, 2].max(), 0.0)) + 0.0,
                         press_on_aft_g=float(max(-tot[:, 2].min(), 0.0)) + 0.0, along_face_g=float(along.max()),
                         spin_hz_max=float(max(prof.spin_hz(t) for t in ts)), source=ph["source"]))
    return rows


def roll_to_wall(prof, r_xy, a_roll_g, distance, t0=0.0, t1=60.0, dt=1e-3):
    """When the spin's centrifugal + Euler load at r_xy [m] first exceeds a grain's rolling limit a_roll_g [g], and
    when the grain, rolling without slip from rest (dv/dt = (a - a_roll)/1.4, wall.rolling_acceleration), has
    covered `distance` [m].  A one-dimensional estimate: the load is taken at r_xy all the way (the far corner is
    the largest load, so this is the fastest case) and Coriolis is ignored.  Returns (t_onset, t_arrive, v_arrive)
    [s, s, m/s]; None entries if it does not happen before t1."""
    x = v = 0.0
    t_on = None
    for t in np.arange(t0, t1, dt):
        L = prof.loads_at(t, np.asarray(r_xy, float))
        a = math.hypot(L["centrifugal"][0] + L["euler"][0], L["centrifugal"][1] + L["euler"][1])
        if t_on is None and a > a_roll_g:
            t_on = float(t)
        v += rolling_acceleration(a * G0, a_roll_g * G0, v, dt) * dt
        x += v * dt
        if x >= distance:
            return t_on, float(t), v
    return t_on, None, None


def flight_history(prof, r_xy=(0.0, 0.0), n=8000):
    """Sampled loads of a FlightProfile from lift-off to its end: t [s]; the apparent acceleration of a free grain
    along the rocket axis (+ toward the nose) and its sideways size [g]; the centrifugal and Euler loads at r_xy [g];
    the spin [Hz].  Uniform samples plus 400 inside each short event (de-spin, parachute opening, splashdown) so
    their peaks are resolved."""
    ts = list(np.linspace(0.0, prof.end_s - 1e-6, n))
    events = [(prof.despin_s, prof.despin_s + prof.despin_duration_s)] + prof.event_windows()
    for a, b in events:
        ts += list(a + (np.arange(400) + 0.5) / 400 * (b - a))
    ts = np.array(sorted(ts))
    acc = np.array([prof.apparent_accel(t) for t in ts]) / G0
    ld = [prof.loads_at(t, np.asarray(r_xy, float)) for t in ts]
    return dict(t=ts, axial_g=acc[:, 2], sideways_g=np.hypot(acc[:, 0], acc[:, 1]),
                centrifugal_g=np.array([float(np.hypot(*x["centrifugal"][:2])) for x in ld]),
                euler_g=np.array([float(np.hypot(*x["euler"][:2])) for x in ld]),
                spin_hz=np.array([prof.spin_hz(t) for t in ts]))


_SHORT = {"first-stage burn": "burn 1", "coast between burns": "coast", "second-stage burn": "burn 2",
          "coast in the upper atmosphere": "coast", "de-spin": "de-spin", "coast to ~100 km": "coast",
          "microgravity (above ~100 km)": "microgravity window", "re-entry": "re-entry",
          "parachute opening": "chute", "descent under the parachute": "under the parachute",
          "splashdown": "splash", "afloat": "afloat"}


def plot_flight(prof, hist, split_s=100.0):
    """Three rows in two columns (the first split_s seconds: ascent, spin, de-spin; then the rest of the flight):
    (1) the apparent acceleration along the rocket axis, split into its part toward the aft face (pressing the dust
    on it) and toward the nose (pulling the dust off it); (2) the sideways loads at the tubule; (3) the spin rate.
    Log scales; phases shaded and labelled (the thin band at T+54 s is the 0.9 s de-spin)."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(12, 8.5), sharex="col", gridspec_kw=dict(width_ratios=[1, 2]))
    t = hist["t"]
    pos = lambda x: np.where(x > 0, x, np.nan)
    for col, (lo, hi) in enumerate(((0.0, split_s), (split_s, prof.end_s))):
        m = (t >= lo) & (t <= hi)
        a0, a1, a2 = axs[0, col], axs[1, col], axs[2, col]
        a0.plot(t[m], pos(-hist["axial_g"][m]), color="C0", lw=1.3, label="toward the aft face (presses dust on)")
        a0.plot(t[m], pos(hist["axial_g"][m]), color="C3", lw=1.3, label="toward the nose (pulls dust off)")
        a1.plot(t[m], pos(hist["centrifugal_g"][m]), color="C1", lw=1.2, label="centrifugal (spin)")
        a1.plot(t[m], pos(hist["euler_g"][m]), color="C2", lw=1.2, label="Euler (spin rate changing)")
        a1.plot(t[m], pos(hist["sideways_g"][m]), color="C3", lw=1.2, label="sideways apparent accel. (re-entry, afloat)")
        a2.plot(t[m], hist["spin_hz"][m], color="C4", lw=1.2)
        for ax, ylim in ((a0, (1e-5, 300)), (a1, (1e-5, 50)), (a2, (5e-3, 10))):
            ax.set_yscale("log")
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
        a2.set_xlim(lo, hi)
        a2.set_xlabel("time after lift-off  [s]")
        for k, ph in enumerate(prof.phases()):
            if ph["t1"] <= lo or ph["t0"] >= hi:
                continue
            for ax in (a0, a1, a2):
                ax.axvspan(max(ph["t0"], lo), min(ph["t1"], hi), color="0.93" if k % 2 else "0.84", lw=0, zorder=0)
            if ph["t1"] - ph["t0"] > 0.03 * (hi - lo):
                label = _SHORT.get(ph["name"], ph["name"])
                if col == 0:        # narrow column: short labels, rotated, from the bottom of the panel
                    a0.text(0.5 * (max(ph["t0"], lo) + min(ph["t1"], hi)), 2e-5, label.replace("microgravity ", ""),
                            ha="center", va="bottom", fontsize=7, rotation=90)
                else:
                    a0.text(0.5 * (ph["t0"] + min(ph["t1"], hi)), 45, label, ha="center", va="bottom", fontsize=7)
    a0 = axs[0, 1]
    a0.annotate(f"parachute opens: {prof.opening_peak_g:.0f} g", (prof.chute_s, prof.opening_peak_g),
                (prof.chute_s + 40, 2e-2), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    a0.annotate("splashdown: {:.0f} g".format(prof.axial_g(prof.splash_s + 0.5 * prof.splash_stop_s)),
                (prof.splash_s, 30), (prof.splash_s - 250, 2e-3), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    axs[0, 0].set_ylabel("apparent acceleration\nalong the rocket axis [g]")
    axs[1, 0].set_ylabel("sideways loads at the\ntubule's far corner [g]")
    axs[2, 0].set_ylabel("spin rate [Hz]")
    axs[0, 0].legend(fontsize=7, loc="upper right")
    axs[1, 1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def phase_effects(prof, tubules, h=0.5):
    """For every flight phase and example tubule: the largest pull off the aft face and the largest load along it
    (at the tubule's far corner, aft_face_loads), against the limits of the tubule's weakest grain (the largest
    diameter; wall_limits with the nominal adhesion) and strongest (the smallest diameter).  Verdict per tubule:
    'pulled off' if the pull exceeds the weakest grain's pull-off limit, 'all roll' if the along-face load exceeds
    the strongest grain's rolling limit, 'largest roll' if it exceeds only the weakest grain's, else 'held'."""
    lims = []
    for tb in tubules:
        weak = wall_limits(tb["dust"], tb["d_max"] / 2, tb["wall"], f_adh=tb["wall_adhesion_factor"], h=h)
        strong = wall_limits(tb["dust"], tb["d_min"] / 2, tb["wall"], f_adh=tb["wall_adhesion_factor"], h=h)
        lims.append((weak, strong, aft_face_loads(prof, far_corner(tb))))
    rows = []
    for k, ph in enumerate(prof.phases()):
        row = dict(phase=ph["name"], t=f"{ph['t0']:.1f}-{ph['t1']:.1f}", pull_g=lims[0][2][k]["pull_off_aft_g"])
        for tb, (weak, strong, loads) in zip(tubules, lims):
            along = loads[k]["along_face_g"]
            verdict = ("pulled off" if loads[k]["pull_off_aft_g"] > weak.pull_off_g else
                       "all roll" if along > strong.roll_g else "largest roll" if along > weak.roll_g else "held")
            row[tb["name"][:2]] = f"{along:.3g} g: {verdict}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------- v0.5 results
V05_STATUS = ("v0.5 event-driven engine: rigid aggregates; wall grains held in place until pulled off, so no rolling "
              "or sliding along the walls; 800 grains per tubule; docs/MISSION_RUNS.md (the README's 'Mission runs' table at commit f133c3e). "
              "The v0.6 full-contact re-runs were stopped before they finished: there are no v0.6 scenario results.")

V05_SCENARIOS = [
    # (scenario, what it changes, T1, T2, T3, T4): each cell "grains released · grain-grain impacts in the
    # microgravity window (single-grain impacts that stuck / all single-grain impacts, their median speed)",
    # copied verbatim from docs/MISSION_RUNS.md (the README's table until v0.7; v0.5 engine).
    ("nominal", "50 g / 200 Hz shaker, 10 s at T+90 s (vacuum, standing)", "0 · 0", "0 · 0", "0 · 0", "0 · 0"),
    ("no_release", "no release mechanism", "0 · 0", "0 · 0", "0 · 0", "0 · 0"),
    ("no_drag", "nominal, no drag in the coasts", "0 · 0", "0 · 0", "0 · 0", "0 · 0"),
    ("rough_grains", "nominal, stickiness × 0.02 (rough grains or a coating)", "730 · 7051 (223/6189, 471 mm/s)",
     "588 · 3534 (117/2712, 449 mm/s)", "398 · 637 (39/368, 452 mm/s)", "179 · 124 (4/28, 421 mm/s)"),
    ("strong_shaker", "500 g / 1 kHz shaker, 5 s at T+90 s", "88 · 125 (0/39, 1071 mm/s)", "28 · 36 (0/2, 882 mm/s)",
     "6 · 0", "0 · 0"),
    ("kick", "three 5,000 g / 5 kHz single-cycle kicks (T+90, 150, 210 s)", "795 · 186 (3/66, 32 mm/s)",
     "772 · 239 (28/88, 60 mm/s)", "683 · 48 (0/20, 45 mm/s)", "502 · 28 (5/6, 43 mm/s)"),
    ("kick+lying", "kicks, tubules lying flat", "795 · 124 (6/30, 12 mm/s)", "772 · 102 (10/34, 117 mm/s)",
     "691 · 51 (1/14, 15 mm/s)", "510 · 25 (0/1, 160 mm/s)"),
    ("kick+spin_1hz", "kicks, 1 Hz residual spin (the guide's upper value)", "795 · 3740 (7/1737, 155 mm/s)",
     "770 · 1074 (21/551, 301 mm/s)", "683 · 778 (2/376, 224 mm/s)", "498 · 201 (4/94, 345 mm/s)"),
    ("kick+gas_100pa", "kicks, 100 Pa of air", "795 · 48 (0/4, 1045 mm/s)", "771 · 13 (2/7, 973 mm/s)", "683 · 4",
     "499 · 1"),
    ("kick+gas_1atm", "kicks, 1 atm of air", "795 · 57 (0/9, 548 mm/s)", "770 · 9 (1/7, 436 mm/s)",
     "683 · 1; 42 floating at power-off", "498 · 3; 20 floating at power-off"),
]

_CELL = re.compile(r"^(\d+) · (\d+)(?: \((\d+)/(\d+), (\d+) mm/s\))?")


def parse_v05_cell(cell):
    """'795 · 186 (3/66, 32 mm/s)' -> dict(released=795, impacts=186, stuck=3, singles=66, median_mm_s=32);
    missing parts are None."""
    m = _CELL.match(cell)
    k = [None if x is None else int(x) for x in m.groups()]
    return dict(released=k[0], impacts=k[1], stuck=k[2], singles=k[3], median_mm_s=k[4])


def plot_thresholds_and_impacts(tubules, scenarios=("kick", "kick+lying"), d_range=(40e-6, 120e-6), ax=None):
    """Sticking speed of two equal grains (dolle_core.sticking_threshold, the simulator's 300 steps per contact)
    against their diameter, for silica and graphite, with the v0.5 median speed of the single-grain impacts in the
    microgravity window (V05_SCENARIOS) at each example tubule's mean sieve diameter, labelled stuck/all."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    d = np.geomspace(*d_range, 9)
    colours = {"silica": "C0", "graphite": "C3"}
    for grain, colour in colours.items():
        v = [sticking_threshold(make_pair(grain, x / 2, grain, x / 2)) for x in d]
        ax.plot(d * 1e6, np.array(v) * 1e3, "-", color=colour, lw=2, label=f"{grain}-{grain}: sticking speed")
    cells = {sc: c for sc, _, *c in V05_SCENARIOS}
    for sc, marker in zip(scenarios, "osD^"):
        for tb, cell in zip(tubules, cells[sc]):
            c = parse_v05_cell(cell)
            if c["median_mm_s"] is None:
                continue
            x = math.sqrt(tb["d_min"] * tb["d_max"]) * 1e6
            ax.plot(x, c["median_mm_s"], marker, color=colours.get(tb["dust"], "0.3"), mfc="none", ms=9, mew=1.5)
            ax.annotate(f"{c['stuck']}/{c['singles']}", (x, c["median_mm_s"]), (7, 2 if tb["dust"] == "silica" else -9),
                        textcoords="offset points", fontsize=8, color=colours.get(tb["dust"], "0.3"))
        ax.plot([], [], marker, color="0.3", mfc="none", label=f"v0.5 '{sc}': median impact speed (stuck/all)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grain diameter  d  [µm]")
    ax.set_ylabel("speed  [mm/s]")
    ax.set_xticks([45, 53, 70, 90, 106])
    ax.set_xticklabels(["45", "53", "70", "90", "106"])
    ax.minorticks_off()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.figure.tight_layout()
    return ax


def v05_table_matches_readme(readme_path=None):
    """True if every row of V05_SCENARIOS appears verbatim in the scenario table of docs/MISSION_RUNS.md (or of the
    file readme_path); None if that file is not on disk (Colab).  (The table was in README.md until v0.7.)"""
    path = Path(readme_path) if readme_path else Path(__file__).resolve().parents[1] / "docs" / "MISSION_RUNS.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    for sc, desc, *cells in V05_SCENARIOS:
        line = f"| `{sc}`: {desc} | " + " | ".join(cells) + " |"
        if line not in text:
            return False
    return True


# ---------------------------------------------------------------------------------------------------- notebook 00
def package_lines():
    """Lines of python in the dolle_sim package, in the dolle_core physics modules, and in dolle_core's notebook
    helpers (nb_*.py, which only tabulate and plot), counted from the files they were imported from.
    Returns dict label -> (files, lines)."""
    import dolle_core
    import dolle_sim
    count = lambda files: (len(files), sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files))
    core = sorted(Path(dolle_core.__file__).parent.glob("*.py"))
    return {"dolle_sim": count(sorted(Path(dolle_sim.__file__).parent.glob("*.py"))),
            "dolle_core (physics)": count([f for f in core if not f.name.startswith("nb_")]),
            "dolle_core (notebook helpers nb_*.py)": count([f for f in core if f.name.startswith("nb_")])}


def core_vs_sim(mat1, mat2, r=50e-6, wall=False):
    """The same impact computed by dolle_core and by the full simulator's 1-D contact model
    (dolle_sim/contact.py JKRContact): the sticking speed of a head-on collision of two grains of radius r [m]
    (or a grain on a wall of mat2 when wall=True), and the restitution of one impact at twice that speed.
    tests/test_dolle_core.py makes this comparison for many more cases.  Returns a dict (speeds in m/s)."""
    from dolle_sim import materials as sm
    from dolle_sim.contact import JKRContact
    from .collide import impact
    a, b = sm.get_material(mat1), sm.get_material(mat2)
    props = sm.pair_props(a, r, b, None if wall else r)
    m1, m2 = sm.sphere_mass(a, r), sm.sphere_mass(b, r)
    m_eff = m1 if wall else m1 / (1.0 + m1 / m2)       # the reduced mass, written as dolle_core does
    sim = JKRContact(props)
    pair = make_pair(mat1, r, mat2, None if wall else r)
    v_sim, v_core = sim.threshold(m_eff), sticking_threshold(pair)
    e_sim, e_core = sim.resolve(2 * v_core, m_eff).e_n, impact(pair, 2 * v_core).e
    return dict(case=pair.name, v_stick_core=v_core, v_stick_sim=v_sim, e_core=e_core, e_sim=e_sim)

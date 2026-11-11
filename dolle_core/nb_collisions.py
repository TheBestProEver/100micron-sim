"""Helpers for the collision notebooks (notebooks/01_two_grains_colliding, 02_momentum_conservation,
03_silica_vs_carbon): small functions that keep the notebook cells short.

Nothing here adds physics.  Every helper calls the core modules (collide.py, jkr.py, scaling.py, many.py, ...)
with their default settings, which reproduce the full simulator dolle_sim; the only exceptions are clearly
named demonstrations (jkr_total_energy, which re-derives the JKR curve from an energy balance, and
collide_one_sided, a deliberately BROKEN integrator that shows what a momentum violation looks like).

dolle_sim is imported only inside the functions that compare with it (simulator_*), and matplotlib only inside
the plot_* functions, so `import dolle_core` still needs nothing but numpy.

Units: SI everywhere unless a name or label says otherwise (plots use um, nm, us, uN, mm/s for readability).
"""
from __future__ import annotations

from dataclasses import replace
import inspect
import math
from pathlib import Path

import numpy as np

from .collide import (ContactState, contact_force, contact_law, impact, is_bound, sticking_threshold, time_step)
from .damping import damping_beta
from .jkr import (break_overlap, break_radius, contact_work, equilibrium_contact_radius, force_of_radius,
                  hysteresis_energy, overlap_of_radius, pull_off_force, snap_on_radius, tabor_parameter)
from .materials import get_material
from .pair import make_pair
from .plastic import yield_speed
from .scaling import damping_factor, scaling_law_speed, undamped_sticking_speed

PROPERTIES = ("gamma", "E", "nu", "rho", "e_dry", "yield_pressure")
"""The six material numbers that enter the contact physics (materials.py)."""


# ---------------------------------------------------------------------------- the pair and the JKR curve
def describe_pair(pair) -> str:
    """The pair quantities of `pair` as readable text with units (pair.py)."""
    m2 = "infinite (wall)" if math.isinf(pair.m2) else f"{pair.m2:.4e} kg"
    return "\n".join([
        f"pair {pair.name}",
        f"  E*  = {pair.E_star / 1e9:.4f} GPa      reduced modulus",
        f"  R*  = {pair.R_star * 1e6:.4f} um        reduced radius",
        f"  m1  = {pair.m1:.4e} kg,  m2 = {m2}",
        f"  m*  = {pair.m_star:.4e} kg   reduced mass",
        f"  w   = {pair.w:.5f} J/m^2     work of adhesion",
        f"  e_dry = {pair.e_dry:.4f}        dry restitution (sets the damping)",
        f"  p_y = {pair.p_y / 1e9:.3f} GPa        yield pressure"])


def jkr_total_energy(a, delta, R: float, w: float, E: float):
    """Total energy U_el(a, delta) - pi a^2 w [J] of a contact of patch radius a [m] held at overlap delta [m]:
    the JKR energy balance (Johnson, Kendall & Roberts 1971; Johnson 1985, "Contact Mechanics", ch. 5).
    Elastic part: load a Hertz contact to patch radius a (energy 8 E a^5/(15 R^2), load P1 = 4 E a^3/(3R),
    overlap a^2/R), then unload it at fixed patch radius (a flat punch of stiffness 2 a E) by D = a^2/R - delta,
    which releases P1 D - a E D^2.  Works on numpy arrays.  Its minimum over a is the JKR patch radius."""
    a = np.asarray(a, dtype=float)
    P1, D = 4 * E * a ** 3 / (3 * R), a ** 2 / R - delta
    return 8 * E * a ** 5 / (15 * R ** 2) - P1 * D + a * E * D ** 2 - math.pi * a ** 2 * w


def jkr_curve(R: float, w: float, E: float, a_max: float, n: int = 800) -> dict:
    """Points of the JKR curve for patch radii from a0/100 to a_max [m]: arrays a, delta [m], F [N] and
    `stable` (True where a >= a_c, the branch a collision can follow)."""
    a0 = equilibrium_contact_radius(R, w, E)
    a = np.geomspace(a0 / 100, a_max, n)
    delta = np.array([overlap_of_radius(x, R, w, E) for x in a])
    F = np.array([force_of_radius(x, R, w, E) for x in a])
    return {"a": a, "delta": delta, "F": F, "stable": a >= break_radius(R, w, E)}


def plot_jkr_curve(pair, a_max_factor: float = 1.6, ax=None):
    """Plot the JKR force-overlap curve of `pair` (nm, uN) with the Hertz curve and the landmarks
    (snap-on, zero load, pull-off, break)."""
    import matplotlib.pyplot as plt
    R, w, E = pair.R_star, pair.w, pair.E_star
    a0, Fc = equilibrium_contact_radius(R, w, E), pull_off_force(R, w)
    c = jkr_curve(R, w, E, a_max_factor * a0)
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    st = c["stable"]
    ax.plot(c["delta"][st] * 1e9, c["F"][st] * 1e6, "C0", label="JKR, stable branch")
    ax.plot(c["delta"][~st] * 1e9, c["F"][~st] * 1e6, "C0--", lw=1, label="JKR, unstable branch")
    dH = np.linspace(0, c["delta"].max(), 200)
    ax.plot(dH * 1e9, 4 / 3 * E * math.sqrt(R) * dH ** 1.5 * 1e6, "C7:", label="Hertz (no adhesion)")
    pts = {"snap-on": snap_on_radius(R, w, E), "zero load $a_0$": a0, "pull-off $-F_c$": a0 / 4 ** (1 / 3),
           "break $\\delta_c$": break_radius(R, w, E)}
    for (lab, a), col in zip(pts.items(), ("C3", "C2", "C1", "C4")):
        ax.plot(overlap_of_radius(a, R, w, E) * 1e9, force_of_radius(a, R, w, E) * 1e6, "o", color=col, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set(xlabel="overlap $\\delta$ [nm]", ylabel="contact force $F$ [uN]  (> 0 pushes apart)",
           title=f"JKR contact, {pair.name}: $F_c$ = {Fc * 1e6:.2f} uN")
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------- one impact, phase by phase
def impact_phases(result) -> dict:
    """Landmarks of a recorded collision (collide(..., record=True)): times [s], overlaps [m] and forces [N]
    at touch, maximum overlap, return to zero overlap (the neck starts), and the end (break or capture)."""
    tr = result.trace
    t, d, F, vr = tr["t"], tr["overlap"], tr["force"], tr["v1"] - tr["v2"]
    i_max = int(np.argmax(d))
    after = np.flatnonzero(d[i_max:] < 0)
    i_neck = i_max + int(after[0]) if len(after) else None
    back = np.flatnonzero(vr < 0)
    i_reb = int(back[0]) if len(back) else None
    end = len(t) - (2 if result.outcome == "bounce" else 1)          # last row still in contact
    out = {"t_touch": t[0], "F_touch": F[0], "t_max": t[i_max], "delta_max": d[i_max], "F_max": F[i_max],
           "t_rebound": t[i_reb] if i_reb is not None else None,
           "t_neck": t[i_neck] if i_neck is not None else None, "t_end": t[end], "delta_end": d[end],
           "F_end": F[end], "F_min": F.min(), "outcome": result.outcome}
    return out


def plot_impact(results, labels, axes=None):
    """Overlap [nm], patch radius [nm] and contact force [uN] against time [us] for recorded collisions, with
    the break overlap of the first one marked."""
    import matplotlib.pyplot as plt
    if axes is None:
        axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)[1]
    for r, lab in zip(results, labels):
        tr = r.trace
        k = slice(None, -1) if r.outcome == "bounce" else slice(None)      # drop the "after break" row
        for ax, key, s in zip(axes, ("overlap", "contact_radius", "force"), (1e9, 1e9, 1e6)):
            ax.plot(tr["t"][k] * 1e6, tr[key][k] * s, label=lab)
    law = contact_law(results[0].pair)
    axes[0].axhline(law.delta_break * 1e9, color="C4", ls=":", label="break overlap $\\delta_c$")
    axes[1].axhline(law.a_c * 1e9, color="C4", ls=":", label="break radius $a_c$")
    axes[2].axhline(-pull_off_force(law.R, law.w) * 1e6, color="C1", ls=":", label="$-F_c$")
    for ax, lab in zip(axes, ("overlap $\\delta$ [nm]", "patch radius $a$ [nm]", "force $F_{el}$ [uN]")):
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(lab)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time since touch $t$ [us]")
    return axes


def loop_areas(result) -> dict:
    """Work [J] done on the contact over the recorded collision: elastic (integral of F_el d delta, equal to the
    hysteresis |U0| when the contact breaks on its own curve) and damping (integral of F_damp d delta), by the
    trapezoidal rule over the rows in contact.  Also the model's own |U0| and damping-loss bookkeeping."""
    tr = result.trace
    k = slice(None, -1) if result.outcome == "bounce" else slice(None)
    d, F, Fd = tr["overlap"][k], tr["force"][k], tr["damping_force"][k]
    trap = lambda y: float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(d)))
    law = contact_law(result.pair)
    return {"elastic_work": trap(F), "damping_work": trap(Fd), "U0": -law.U0,
            "damping_loss_trace": float(tr["damping_loss"][-1])}


def plot_force_loop(result, ax=None):
    """Force against overlap through one bouncing collision: the elastic force F_el (on the JKR curve) and the
    total F_el + F_damp; shaded: the adhesion-hysteresis area (|U0|) and the damping loop."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    tr = result.trace
    k = slice(None, -1) if result.outcome == "bounce" else slice(None)
    d, F, Ft = tr["overlap"][k] * 1e9, tr["force"][k] * 1e6, (tr["force"][k] + tr["damping_force"][k]) * 1e6
    neck = d <= 0
    ax.fill_between(d[neck], F[neck], 0, color="C1", alpha=0.3, label="adhesion hysteresis area = $|U_0|$")
    ax.fill(np.r_[d, d[::-1]], np.r_[Ft, F[::-1]], color="C2", alpha=0.25, label="damping loop = energy to heat")
    ax.plot(d, F, "C0", label="elastic force $F_{el}$ (JKR curve)")
    ax.plot(d, Ft, "C3", lw=1, label="total force $F_{el} + F_{damp}$")
    ax.plot([d[0], d[0]], [0, F[0]], "C0:", lw=1)
    ax.plot([d[-1], d[-1]], [F[-1], 0], "C0:", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="overlap $\\delta$ [nm]", ylabel="force [uN]  (> 0 pushes apart)",
           title=f"force-overlap loop at {result.v_in * 1e3:.0f} mm/s")
    ax.legend(fontsize=8)
    return ax


def energy_split(result) -> dict:
    """Final energy bookkeeping of a recorded collision as fractions of the incoming 1/2 m* v_in^2: kinetic
    (relative motion), stored, damping, hysteresis, plastic, integrator error (collide.py TRACE_KEYS)."""
    tr = result.trace
    E_in = 0.5 * result.pair.m_star * result.v_in ** 2
    keys = ("kinetic", "stored", "damping_loss", "hysteresis_loss", "plastic_loss", "energy_error")
    return {k: float(tr[k][-1] / E_in) for k in keys} | {"E_in": E_in}


def plot_energy_budget(result, ax=None):
    """Energy terms of a recorded collision against time, as fractions of the incoming energy."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    tr = result.trace
    E_in = 0.5 * result.pair.m_star * result.v_in ** 2
    t = tr["t"] * 1e6
    for key, lab in (("kinetic", "kinetic (relative motion)"), ("stored", "stored in the contact $U$"),
                     ("damping_loss", "lost to damping (heat)"), ("hysteresis_loss", "lost to adhesion hysteresis")):
        ax.plot(t, tr[key] / E_in, label=lab)
    total = tr["kinetic"] + tr["stored"] + tr["damping_loss"] + tr["hysteresis_loss"] + tr["plastic_loss"]
    ax.plot(t, total / E_in, "k--", lw=1, label="sum (= 1 if energy is booked completely)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="time since touch [us]", ylabel="energy / incoming $\\frac{1}{2} m^* v_{in}^2$",
           title=f"energy budget at {result.v_in * 1e3:.1f} mm/s: {result.outcome}")
    ax.legend(fontsize=8)
    return ax


def loss_fractions(pair, speeds) -> dict:
    """For each approach speed [m/s]: restitution e and the fractions of the incoming energy lost to the adhesion
    hysteresis and to damping (recorded impacts, centre-of-mass frame)."""
    rows = {"e": [], "hysteresis": [], "damping": []}
    for v in speeds:
        r = impact(pair, v, record=True)
        s = energy_split(r)
        rows["e"].append(r.e)
        rows["hysteresis"].append(s["hysteresis_loss"])
        rows["damping"].append(s["damping_loss"])
    return {k: np.array(x) for k, x in rows.items()}


def restitution_curve(pair, speeds, n_sub: int = 300):
    """Restitution e (0 = stuck) and whether the contact yielded, at each approach speed [m/s]."""
    res = [impact(pair, v, n_sub=n_sub) for v in speeds]
    return np.array([r.e for r in res]), np.array([r.yielded for r in res])


def dry_restitution(material, e_dry: float, v: float = 0.1) -> float:
    """Restitution of a head-on impact at v [m/s] of two 50 um radius grains of `material` with the stickiness
    switched off (gamma = 1e-7 J/m^2, as the simulator's check 'damping') and yield off: what e_dry calibrates."""
    m = replace(get_material(material), gamma=1e-7, e_dry=e_dry)
    return impact(make_pair(m, 50e-6, m, 50e-6, plastic=False), v).e


def _e_of_beta(beta: float) -> float:
    """Inverse of damping.damping_beta: the restitution e whose beta = ln e / sqrt(ln^2 e + pi^2) is `beta`."""
    return math.exp(math.pi * beta / math.sqrt(1.0 - beta * beta))


def restitution_with_prefactor(k: float, e_dry: float, v: float = 0.1, material: str = "silica") -> float:
    """Restitution of the non-sticky impact of dry_restitution when the damping coefficient is
    c_d = -2 k beta(e_dry) sqrt(2 E* m*) instead of the simulator's k = sqrt(5/6).  Done exactly with the core's
    own loop: a material whose e_dry' has beta(e_dry') = k beta(e_dry)/sqrt(5/6) gets that c_d."""
    b = k * damping_beta(e_dry) / math.sqrt(5.0 / 6.0)
    return dry_restitution(material, _e_of_beta(b), v)


def calibrated_prefactor(e_dry: float, v: float = 0.1, iters: int = 40) -> float:
    """The prefactor k for which the non-sticky impact returns exactly e_dry (bisection on k; more damping, lower
    e).  The simulator uses k = sqrt(5/6) = 0.9129 for every e_dry."""
    lo, hi = 0.5, 1.5
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if restitution_with_prefactor(mid, e_dry, v) > e_dry:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def well_energies(pair) -> dict:
    """Energies of the JKR curve of `pair` in units of F_c |delta_c| (no unit): the hysteresis |U0| =
    G(a_c) - G(a_snap) (what an arriving grain, which starts at the snap-on state, must pay to escape) and the depth
    of the well measured from its bottom, G(a_c) - G(a0) (the "1.54 F_c delta_c" of the dust literature, e.g.
    Dominik & Tielens 1997; dolle_sim materials.pair_props E_break)."""
    R, w, E = pair.R_star, pair.w, pair.E_star
    unit = pull_off_force(R, w) * abs(break_overlap(R, w, E))
    a_c = break_radius(R, w, E)
    G = lambda a: contact_work(a, R, w, E)
    return {"U0": (G(a_c) - G(snap_on_radius(R, w, E))) / unit,
            "from_a0": (G(a_c) - G(equilibrium_contact_radius(R, w, E))) / unit, "unit": unit}


def threshold_convergence(pair, n_subs=(100, 300, 1000, 3000), tol: float = 1e-5) -> dict:
    """Sticking threshold [m/s] with each number of steps per contact time (collide.py time_step)."""
    return {n: sticking_threshold(pair, tol=tol, n_sub=n) for n in n_subs}


# ---------------------------------------------------------------------------- comparisons with dolle_sim
def _simulator_contact(mat1: str, r1: float, mat2: str, r2, steps: int = 300, **override):
    """dolle_sim's JKRContact and the m* it uses, for named materials (optionally with fields overridden)."""
    from dolle_sim import materials as sm
    from dolle_sim.contact import JKRContact
    a, b = replace(sm.get_material(mat1), **override), replace(sm.get_material(mat2), **override)
    p = sm.pair_props(a, r1, b, r2)
    m1 = sm.sphere_mass(a, r1)
    m_eff = m1 if r2 is None else m1 * sm.sphere_mass(b, r2) / (m1 + sm.sphere_mass(b, r2))
    return JKRContact(p, steps=steps), m_eff


def simulator_threshold(mat1: str, r1: float, mat2: str, r2, steps: int = 300, tol: float = 2e-3, **override):
    """Sticking threshold [m/s] from the full simulator (dolle_sim/contact.py JKRContact.threshold)."""
    c, m = _simulator_contact(mat1, r1, mat2, r2, steps, **override)
    return c.threshold(m, tol=tol)


def simulator_restitution(mat1: str, r1: float, mat2: str, r2, speeds, steps: int = 300, **override):
    """Restitution at each approach speed [m/s] from the full simulator (JKRContact.resolve; 0 = stuck)."""
    c, m = _simulator_contact(mat1, r1, mat2, r2, steps, **override)
    return np.array([0.0 if (res := c.resolve(v, m)).stuck else res.e_n for v in speeds])


# ---------------------------------------------------------------------------- momentum
def momentum_drift(result) -> float:
    """Largest |P(t) - P(0)| over a recorded collision, relative to m1 |u1| + m2 |u2| (the momenta involved)."""
    tr, p = result.trace, result.pair
    scale = p.m1 * abs(result.u1) + (0.0 if math.isinf(p.m2) else p.m2 * abs(result.u2))
    return float(np.max(np.abs(tr["momentum"] - result.momentum_before)) / scale)


def plot_momentum(t, m1v1, m2v2, ax=None, title=""):
    """Momentum of grain 1, grain 2 and the total [kg m/s] against time [s] (plotted in us)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    ax.plot(t * 1e6, m1v1, label="grain 1: $m_1 v_1$")
    ax.plot(t * 1e6, m2v2, label="grain 2: $m_2 v_2$")
    ax.plot(t * 1e6, m1v1 + m2v2, "k", lw=2, label="total $P = m_1 v_1 + m_2 v_2$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="time since touch [us]", ylabel="momentum [kg m/s]", title=title)
    ax.legend(fontsize=8)
    return ax


def collide_one_sided(pair, u1: float, u2: float = 0.0, n_sub: int = 300, max_steps: int = 300000) -> dict:
    """A deliberately BROKEN head-on collision: the same contact law and velocity-Verlet loop as
    collide.collide_pair (no trace bookkeeping), except that the contact force is applied to grain 1 only;
    grain 2 never feels the reaction (Newton's third law switched off).  Returns arrays t [s], v1, v2 [m/s],
    momentum [kg m/s] after every step, and the outcome.  For the demonstration in notebook 02 only."""
    law, v_in = contact_law(pair), u1 - u2
    dt, s = time_step(law, v_in, n_sub), ContactState(a=law.a_snap)
    F = force_of_radius(s.a, law.R, law.w, law.E) + law.c_d * math.sqrt(s.a) * v_in
    v1, v2, t = u1, u2, 0.0
    rows = [(0.0, v1, v2)]
    outcome = "stick"
    for _ in range(max_steps):
        v1h, v2h = v1 - 0.5 * dt * F / pair.m1, v2                      # half kick: grain 1 only (the bug)
        s.delta += dt * (v1h - v2h)
        t += dt
        F_el = contact_force(law, s)
        if F_el is None:
            v1, v2, outcome = v1h, v2h, "bounce"
            rows.append((t, v1, v2))
            break
        F = F_el + law.c_d * math.sqrt(s.a) * (v1h - v2h)
        v1, v2 = v1h - 0.5 * dt * F / pair.m1, v2h                      # half kick: grain 1 only (the bug)
        s.delta_peak = max(s.delta_peak, s.delta)
        rows.append((t, v1, v2))
        if is_bound(law, s, v1 - v2):
            break
    t, v1, v2 = (np.array(c) for c in zip(*rows))
    return {"t": t, "v1": v1, "v2": v2, "momentum": pair.m1 * v1 + pair.m2 * v2, "outcome": outcome}


def show_source(target, pattern: str, before: int = 0, after: int = 0) -> None:
    """Print the lines of `target` (a function, or a path to a file) that contain `pattern`, with `before` and
    `after` lines of context and their line numbers in the file."""
    if callable(target):
        lines, start = inspect.getsourcelines(target)
        name = f"{Path(inspect.getsourcefile(target)).name}: {target.__name__}"
    else:
        lines, start, name = Path(target).read_text(encoding="utf-8").splitlines(True), 1, Path(target).name
    hits = [i for i, line in enumerate(lines) if pattern in line]
    print(f"--- {name}")
    for i in hits:
        for j in range(max(i - before, 0), min(i + after + 1, len(lines))):
            print(f"{start + j:5d}  {lines[j].rstrip()}")
        print("  ...")


def simulator_file(name: str) -> Path:
    """Path of a source file of the full simulator, e.g. simulator_file("simulation.py"): in dolle_sim (next to
    dolle_core), or else in the optional extras/dolle_extras (the v0.6 engine, e.g. "dem.py")."""
    root = Path(__file__).resolve().parents[1]
    for folder in (root / "dolle_sim", root / "extras" / "dolle_extras"):
        if (folder / name).exists():
            return folder / name
    return root / "dolle_sim" / name


def aggregates(line, row: int = -1) -> list:
    """Groups of grains that touch their neighbours at recorded time `row` of a many.line_collision result:
    for each group (members, centre-of-mass velocity = group momentum / group mass [m/s], spread of the members'
    velocities max - min [m/s]: the vibration the damping has not yet removed)."""
    groups, cur = [], [0]
    for k, touching in enumerate(line.touching[row]):
        if touching:
            cur.append(k + 1)
        else:
            groups.append(cur)
            cur = [k + 1]
    groups.append(cur)
    m, v = line.masses, line.v[row]
    return [(g, float(np.sum(m[g] * v[g]) / np.sum(m[g])), float(np.ptp(v[g]))) for g in groups]


def group_velocity(line, members) -> np.ndarray:
    """Group momentum / group mass [m/s] of the grains `members` at every recorded time of a line_collision."""
    m = line.masses[members]
    return (line.v[:, members] * m).sum(axis=1) / m.sum()


def last_contact_time(line, pair_index: int) -> float:
    """Last recorded time [s] at which neighbour pair `pair_index` (grains k, k+1) was touching (nan if never)."""
    on = np.flatnonzero(line.touching[:, pair_index])
    return float(line.t[on[-1]]) if len(on) else float("nan")


def dem_threshold(material: str = "silica", r: float = 50e-6, offset: float = 0.0, dt_factor: float = 1.0) -> float:
    """Sticking threshold [m/s] of two equal grains in the full simulator's 3-D contact model (extras/dolle_extras/dem.py,
    needs numba), as a NORMAL impact speed: the set-up of verify_full.check_dem_vs_1d (damping on, no spin, step
    t_osc/400, 16 bisection steps between 0.5 and 2 times the 1-D reference), except that the second grain can be
    displaced sideways by `offset` [m] (impact parameter; 0 = head-on, 2r would be grazing) and the step multiplied by
    dt_factor.  With an offset the grains also slide and roll during the contact (tangential springs loaded)."""
    from dolle_extras import dem                  # the optional v0.6 engine (extras/; the notebooks put it on the path)
    from dolle_extras.verify_full import _pair_system
    from dolle_sim.materials import get_material as sim_material, pair_props, sphere_mass
    from dolle_sim.contact import JKRContact
    mat = sim_material(material)
    p, m_eff = pair_props(mat, r, mat, r), sphere_mass(mat, r) / 2
    ref = JKRContact(p, steps=2000).threshold(m_eff, tol=1e-5)
    dt = 2 * math.pi * math.sqrt(m_eff / (2 * p.E_star * p.a0)) / 400 * dt_factor
    cos_t = math.sqrt(1.0 - (offset / (2 * r)) ** 2)

    def bounces(v_n):
        v, gap = v_n / cos_t, 0.02 * r
        x = np.array([[0, 0, 0], [2 * r * cos_t + gap, offset, 0]], float) + 0.5
        S = _pair_system(mat, r, x, [[v / 2, 0, 0], [-v / 2, 0, 0]], np.zeros((2, 3)), skin=0.5 * r)
        S.run(dt, int(gap / v / dt) + 1)
        for _ in range(400):
            S.run(dt, 2000)
            on = S.hist[:, dem.H_TOUCH].max() > 0.5 if len(S.hist) else False
            if not on and np.linalg.norm(S.x[1] - S.x[0]) > 2 * r:
                return True
        return False
    lo, hi = 0.5 * ref, 2 * ref
    for _ in range(16):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if bounces(mid) else (mid, hi)
    return math.sqrt(lo * hi)


def plot_line(line, axes=None):
    """Grains on a line (many.line_collision): velocities [mm/s] with the centre-of-mass velocity, the
    relative change of the total momentum after every step, and the kinetic energy [J], against time [us]."""
    import matplotlib.pyplot as plt
    if axes is None:
        axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)[1]
    t = line.t * 1e6
    for i in range(line.v.shape[1]):
        axes[0].plot(t, line.v[:, i] * 1e3, label=f"grain {i}")
    M, P0 = np.sum(line.masses), float(np.sum(line.masses * line.v[0]))
    axes[0].axhline(P0 / M * 1e3, color="k", ls="--", label="centre of mass")
    scale = float(np.sum(line.masses * np.abs(line.v[0])))
    axes[1].plot(line.momentum_t * 1e6, (line.momentum - P0) / scale, "k", lw=0.8)
    axes[2].plot(t, line.kinetic, "C3")
    for ax, lab in zip(axes, ("velocity [mm/s]", "$(P - P_0)\\,/\\,\\sum m_i|v_i|$", "kinetic energy [J]")):
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("time [us]")
    return axes


# ---------------------------------------------------------------------------- materials compared
def property_table(names) -> str:
    """The six contact properties of each named material, with units, as aligned text."""
    head = f"{'material':22s} {'rho kg/m3':>10s} {'E GPa':>7s} {'nu':>5s} {'gamma J/m2':>11s} {'e_dry':>6s} {'p_y GPa':>8s}"
    rows = [head]
    for n in names:
        m = get_material(n)
        rows.append(f"{m.name:22s} {m.rho:10.0f} {m.E / 1e9:7.1f} {m.nu:5.2f} {m.gamma:11.4f} {m.e_dry:6.2f} "
                    f"{m.yield_pressure / 1e9:8.2f}")
    return "\n".join(rows)


def pair_summary(mat1, mat2=None, r: float = 50e-6, tol: float = 1e-4) -> dict:
    """Contact numbers of two equal-sized grains of radius r [m] (mat2 defaults to mat1): E* [Pa], w [J/m^2],
    pull-off force F_c [N], zero-load patch a0 [m], break overlap [m], hysteresis |U0| [J], undamped and damped
    sticking speeds [m/s], yield speed [m/s], Tabor parameter (z0 = 0.3 nm)."""
    p = make_pair(mat1, r, mat2 if mat2 is not None else mat1, r)
    R, w, E = p.R_star, p.w, p.E_star
    return {"pair": p, "E_star": E, "w": w, "F_c": pull_off_force(R, w), "a0": equilibrium_contact_radius(R, w, E),
            "delta_c": break_overlap(R, w, E), "U0": hysteresis_energy(R, w, E),
            "v_undamped": undamped_sticking_speed(p), "v_stick": sticking_threshold(p, tol=tol),
            "v_yield": yield_speed(p.m_star, R, E, p.p_y), "tabor": tabor_parameter(R, w, E)}


def threshold_of(material, r: float = 50e-6, tol: float = 1e-4, n_sub: int = 300, **changes) -> float:
    """Sticking threshold [m/s] of two grains of radius r [m] of `material` with some fields changed
    (e.g. gamma=0.19): the full model (collide.py), yield included."""
    m = replace(get_material(material), **changes)
    return sticking_threshold(make_pair(m, r, m, r), tol=tol, n_sub=n_sub)


def scaling_prediction(base, changed, r: float = 50e-6, f_ref: str = "graphite") -> float:
    """Ratio of sticking speeds changed/base predicted by hand: the scaling law
    gamma^(5/6) E*^(-1/3) R*^(-5/6) rho^(-1/2) (scaling.py) times f(e_dry) measured on a DIFFERENT reference pair
    (default graphite, yield off), which works because f depends on e_dry only.  Yield is ignored (it starts far
    above the sticking speeds; check v_yield)."""
    def v(m):
        p = make_pair(m, r, m, r)
        ref = make_pair(f_ref, 50e-6, f_ref, 50e-6, plastic=False)
        return scaling_law_speed(m.gamma, p.E_star, p.R_star, m.rho) * damping_factor(m.e_dry, n_sub=300,
                                                                                       tol=1e-5, pair=ref)
    return v(get_material(changed)) / v(get_material(base))


def scaling_band(ranges: dict, e_grid, f_grid, r: float = 50e-6) -> tuple:
    """Lowest and highest sticking speed [m/s] of two equal grains of radius r [m] when each property varies over
    its range: ranges = {"gamma": (lo, hi), "E": (lo, hi), "nu": (lo, hi), "rho": (lo, hi), "e_dry": (lo, hi)}
    (single numbers allowed).  Uses the scaling law (exact for the undamped model, scaling.py) times f(e_dry)
    interpolated in the measured table (e_grid, f_grid): the speed rises with gamma, falls with E*, rho and e_dry,
    and E* = E/(2(1 - nu^2)) rises with nu, so the extremes are at the corners of the ranges."""
    lo_hi = {k: (v, v) if np.isscalar(v) else tuple(v) for k, v in ranges.items()}

    def v(gamma, E, nu, rho, e):
        E_star = E / (2 * (1 - nu * nu))
        return scaling_law_speed(gamma, E_star, r / 2, rho) * float(np.interp(e, e_grid, f_grid))
    g, E, nu, rho, e = (lo_hi[k] for k in ("gamma", "E", "nu", "rho", "e_dry"))
    return v(g[0], E[1], nu[1], rho[1], e[1]), v(g[1], E[0], nu[0], rho[0], e[0])


def swap_properties(base: str, target: str, r: float = 50e-6, tol: float = 1e-4) -> list:
    """Change ONE property of `base` to its value in `target` at a time; for each: the property, the sticking-
    speed ratio from the full model, the hand prediction (scaling_prediction) and the yield speed [m/s] of the
    changed pair.  Last row: all six changed (= target)."""
    b, tg = get_material(base), get_material(target)
    v0, rows = threshold_of(b, r, tol), []
    for prop in PROPERTIES + ("all",):
        ch = {k: getattr(tg, k) for k in (PROPERTIES if prop == "all" else (prop,))}
        m = replace(b, **ch)
        p = make_pair(m, r, m, r)
        rows.append({"property": prop, "measured": threshold_of(m, r, tol) / v0,
                     "predicted": scaling_prediction(b, m, r), "v_yield": yield_speed(p.m_star, p.R_star, p.E_star, p.p_y)})
    return rows


def plot_swap_bars(rows, title: str, ax=None):
    """Bar chart (log axis) of the measured and predicted sticking-speed ratio for each swapped property, with the
    measured value written on each bar."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["measured"] for r in rows], 0.4, label="full model (collide.py)")
    ax.bar(x + 0.2, [r["predicted"] for r in rows], 0.4, label="hand: scaling law x f(e_dry)")
    for xi, r in zip(x, rows):
        ax.text(xi, max(r["measured"], 1.0) * 1.04, f"x{r['measured']:.2f}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ticks = [0.9, 1, 1.5, 2, 3, 4, 6, 8]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(x, [r["property"] for r in rows])
    ax.set(ylabel="sticking speed / silica's (log scale)", title=title)
    ax.legend(fontsize=8, loc="upper left")
    return ax


def plain_log_axis(ax, which: str = "y"):
    """Label a logarithmic axis with plain numbers at 1, 2 and 5 times each power of ten (e.g. 5, 10, 20)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())
    return ax

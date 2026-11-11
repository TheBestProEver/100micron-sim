"""A few grains on a line: momentum conservation you can watch.

N grains move along the x axis, grain i at x_i [m] with velocity v_i [m/s] and radius r_i [m], in order
x_0 < x_1 < ... .  Only neighbours can touch.  Each touching neighbour pair (i, i+1) carries the contact of
collide.py: the JKR force with its hysteresis (forms at overlap 0, breaks at delta_c) plus the damping, with
F > 0 pushing apart.  The pair force is applied as an equal and opposite pair (Newton's third law):

    force on grain i   -= F,      force on grain i+1 += F.

Summed over all grains every pair force cancels, so the total momentum P = sum m_i v_i cannot change.  The
velocity-Verlet loop changes each v_i by (dt/2) f_i / m_i per half kick, so P changes by (dt/2) sum f_i = 0 up to
rounding; LineResult.momentum records P after EVERY step so the student can see that it stays constant to ~1e-16
of the momenta involved, while the kinetic energy does not (adhesion hysteresis and damping take it).

What is left out on purpose.  Plastic yield: line_collision refuses inputs where any pair could reach its yield
speed (plastic.py yield_speed; it checks the largest relative speed the energy in the system allows: the
centre-of-mass kinetic energy plus every well depth |U(a0)| = 1.54 F_c |delta_c|), so the
contact law is exactly the elastic JKR + damping one of collide.py.  Motion off the line (oblique contacts, spin,
sliding, rolling): that is the full simulator's 3-D model, dolle_sim/dem.py, which conserves linear and angular
momentum to 1e-14 (PROVENANCE.md; verify check dem_conservation).

Time step: dt = min over pairs of min(t_Hz, t_osc)/n_sub (collide.py time_step) at the largest possible relative
speed, i.e. the same rule as a single collision, applied to the fastest contact.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .collide import ContactLaw, contact_law, time_step
from .jkr import contact_radius, contact_work, force_of_radius
from .pair import make_pair
from .plastic import yield_speed


@dataclass
class LineResult:
    """Time series of a line of grains.  t [s], x and v [m, m/s] (rows: recorded times, columns: grains),
    touching (rows: times, columns: neighbour pairs); momentum_t and momentum [kg m/s]: total momentum after every
    step; kinetic [J] at the recorded times; masses [kg]; pairs: the Pair of each neighbour pair."""
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    touching: np.ndarray
    momentum_t: np.ndarray
    momentum: np.ndarray
    kinetic: np.ndarray
    masses: np.ndarray
    pairs: list


def _speed_bound(masses, v0, laws) -> float:
    """Largest relative speed any pair can reach [m/s]: its relative kinetic energy 1/2 m* v^2 cannot exceed the
    kinetic energy in the centre-of-mass frame plus the full depth of every adhesive well, measured from the
    bottom of the well at the zero-load patch a0: G_break - G(a0) = 1.54 F_c |delta_c| (= 1.65 |U0|; energy is
    never created)."""
    m, v = np.asarray(masses), np.asarray(v0, dtype=float)
    depth = sum(law.G_break - contact_work(law.a0, law.R, law.w, law.E) for law in laws)
    E = 0.5 * float(np.sum(m * (v - np.sum(m * v) / np.sum(m)) ** 2)) + depth
    return max(math.sqrt(2 * E / law.m) for law in laws)


def _pair_force(law: ContactLaw, state: list, delta: float, v_rel: float) -> float:
    """Contact force [N] (> 0 pushes apart) of one neighbour pair at overlap delta [m] and approach speed v_rel
    [m/s]; state = [touching, a] is updated: forms at delta >= 0 with a = a_snap, breaks below delta_break."""
    if not state[0]:
        if delta < 0.0:
            return 0.0
        state[0], state[1] = True, law.a_snap
    elif delta < law.delta_break:
        state[0] = False
        return 0.0
    state[1] = contact_radius(delta, law.R, law.w, law.E, state[1], law.a_c)
    return force_of_radius(state[1], law.R, law.w, law.E) + law.c_d * math.sqrt(state[1]) * v_rel


def _forces(laws, states, radii, x, v):
    """Force on every grain [N]: each neighbour pair's force applied equal and opposite."""
    f = np.zeros(len(x))
    for k, law in enumerate(laws):
        delta = radii[k] + radii[k + 1] - (x[k + 1] - x[k])
        F = _pair_force(law, states[k], delta, v[k] - v[k + 1])
        f[k] -= F
        f[k + 1] += F
    return f


def line_collision(materials, radii, x0, v0, t_end: float, n_sub: int = 300, record_every: int = 20,
                   f_adh: float = 1.0) -> LineResult:
    """Integrate grains on a line (materials: names or Materials; radii [m]; x0 [m] increasing centre positions;
    v0 [m/s]) for t_end [s] by velocity Verlet with damping at the half-step velocities (as collide.py).  Total
    momentum is recorded after every step, the rest every `record_every` steps."""
    radii, x, v = (np.array(a, dtype=float) for a in (radii, x0, v0))
    pairs = [make_pair(materials[k], radii[k], materials[k + 1], radii[k + 1], f_adh=f_adh, plastic=False)
             for k in range(len(x) - 1)]
    laws, masses = [contact_law(p) for p in pairs], np.array([pairs[0].m1] + [p.m2 for p in pairs])
    v_max = _speed_bound(masses, v, laws)
    _refuse_yield(pairs, materials, radii, f_adh, v_max)
    dt = min(time_step(law, v_max, n_sub) for law in laws)
    states = [[False, 0.0] for _ in laws]
    f = _forces(laws, states, radii, x, v)
    rows, P = {"t": [], "x": [], "v": [], "touching": [], "kinetic": []}, []
    for step in range(int(math.ceil(t_end / dt)) + 1):
        if step % record_every == 0:
            for key, val in (("t", step * dt), ("x", x.copy()), ("v", v.copy()),
                             ("touching", [s[0] for s in states]), ("kinetic", 0.5 * float(np.sum(masses * v * v)))):
                rows[key].append(val)
        v_half = v + 0.5 * dt * f / masses                       # half kick
        x = x + dt * v_half                                      # drift
        f = _forces(laws, states, radii, x, v_half)              # new forces (damping at half-step velocities)
        v = v_half + 0.5 * dt * f / masses                       # half kick
        P.append(float(np.sum(masses * v)))
    P = np.array(P)
    return LineResult(np.array(rows["t"]), np.array(rows["x"]), np.array(rows["v"]), np.array(rows["touching"]),
                      dt * (np.arange(len(P)) + 1), P, np.array(rows["kinetic"]), masses, pairs)


def _refuse_yield(pairs, materials, radii, f_adh, v_max):
    """Raise ValueError if any pair (with its real yield pressure) could be hit faster than its yield speed."""
    for k, p in enumerate(pairs):
        real = make_pair(materials[k], radii[k], materials[k + 1], radii[k + 1], f_adh=f_adh, plastic=True)
        if v_max >= yield_speed(real.m_star, real.R_star, real.E_star, real.p_y):
            raise ValueError(f"pair {k} could reach its yield speed; many.py has no plasticity, use collide()")


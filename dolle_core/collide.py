"""A head-on collision of two grains (or a grain and a wall), integrated step by step.

Body 1 (left, velocity u1) runs into body 2 (right, velocity u2 < u1) along the line joining their centres.
While they touch, the contact pushes body 1 with -F and body 2 with +F (Newton's third law; F > 0 pushes apart):

    m1 dv1/dt = -F,     m2 dv2/dt = +F,     F = F_el(delta) + c_d sqrt(a) (v1 - v2)

F_el is the JKR force (jkr.py), with Thornton-Ning yield if the impact is hard enough (plastic.py); the second
term is the damping (damping.py).  delta is the overlap [m] and v1 - v2 the approach speed [m/s].

Momentum.  Adding the two equations: d(m1 v1 + m2 v2)/dt = -F + F = 0.  The total momentum cannot change,
whatever the force law.  The loop below applies the SAME number F to both bodies in every half step, so the
discrete momentum is conserved to rounding error (~1e-16 of the momenta involved); the trace records it so you can
see this.  A wall has infinite mass: it does not move and takes the impulse; the trace then records the grain's
momentum plus the impulse given to the wall.

Energy.  The contact stores energy (the JKR potential U), damping turns energy into heat, the adhesion
hysteresis |U0| is lost when the contact forms and breaks (jkr.py), and yield costs plastic work.  The trace
records the kinetic energy of the relative motion, 1/2 m* (v1 - v2)^2 (the centre-of-mass motion carries the rest
and never changes), U, and the three losses; their sum stays equal to the incoming 1/2 m* v_in^2 up to the
integrator's error, which is recorded too.  (Bookkeeping convention, as in the simulator: U = 0 for the broken
contact, so the whole hysteresis loss |U0| is booked when the contact forms; only the total per make-and-break
cycle is physical.)

Stick or bounce (dolle_sim/contact.py _resolve_core, same tests in the same order):
  1. if 1/2 m* v_in^2 < |U0| the grains are captured the moment they touch (they can never climb out of the well);
  2. the overlap falls below the break overlap (of the flattened contact after yield): BOUNCE, restitution
     e = (separation speed)/(approach speed), the separation speed taken at that half step;
  3. the approach speed became negative (rebound started) and then positive again (falling back in): STICK
     (the turn-around happens at a patch smaller than a0, where the force is attractive and U < 0, so the
     relative motion plus U has negative energy: the pair is bound, as in test 4);
  4. the energy of the relative motion plus U is below zero (below the energy of the broken contact; damping can
     only lower it, so the grains can never escape): STICK.
  The loop stops as soon as one of tests 3/4 proves the pair can never separate, while the grains are still
  vibrating in the contact; it then REPORTS both at the centre-of-mass velocity (where the damping would bring
  them), so for a stick v1 = v2 = v_cm and momentum_after are set from momentum conservation, not measured.  The
  trace (trace["momentum"]) is the measurement.

Time step (velocity Verlet: half kick, drift, new force, half kick; Verlet 1967, Swope et al. 1982).  It is
second-order accurate and time-reversible, and conserves momentum exactly for pair forces.  The step is
dt = min(t_Hz, t_osc)/n_sub with n_sub = 300 (the simulator's steps_per_contact): t_Hz = 2.87 (m*^2/(R* E*^2
v_in))^(1/5) is the Hertz impact duration (Johnson 1985, "Contact Mechanics", ch. 11) and t_osc = 2 pi sqrt(m*/(2
E* a0)) a contact time scale: the period of a Hertz spring with patch a0 (stiffness 2 E* a0), as in contact.py
_resolve_core.  The true small-oscillation stiffness about the JKR equilibrium is dF/d(delta) = (dF/da)/(d delta/da)
= (2 E* a0^2/R*)/((5/3) a0/R*) = (6/5) E* a0, so the true period is sqrt(2/1.2) = 1.29x longer; t_osc only caps
the step, and the shorter estimate is the safer one.  For the 50 um radius silica pair 300 steps give a threshold
3.8e-4 above the 3000-step value (notebooks/01_two_grains_colliding.ipynb, section 4).

Because this loop does the same arithmetic as contact.py, it gives the simulator's restitutions and sticking
speeds to rounding error (tests/test_dolle_core.py).  What it does NOT cover: oblique impacts, spin, gas drag,
charging, and more than two bodies (many.py does a line of grains).  In an oblique impact the sliding, rolling and
twisting springs act during the contact; the full simulator's 3-D model dolle_sim/dem.py has them.  For HEAD-ON
impacts its sticking speed equals this 1-D model's to 1.6e-3 (check dem_vs_1d; 5e-4 at half its step).  That
oblique impacts stick at the same NORMAL speed is an assumption of the simulator's v0.5 chamber model
(PROVENANCE.md); notebooks/01, section 12, tests it with the 3-D model at two impact offsets (it holds to 1e-4).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from .damping import damping_coefficient
from .jkr import (break_overlap, break_radius, contact_radius, contact_work, equilibrium_contact_radius,
                  force_of_radius, snap_on_radius)
from .pair import Pair, make_pair
from .plastic import flattened_radius, hertz_force, permanent_indentation, plastic_loading_force, yield_overlap


@dataclass(frozen=True)
class ContactLaw:
    """Constants of one contact (computed once per pair): R = R* [m], E = E* [Pa], w [J/m^2], m = m* [kg], a0,
    a_c (end of the stable branch), a_snap [m]; delta_break [m] and G_break [J] (the broken contact); U0 [J] (< 0,
    well depth at touch); c_d (damping.py); delta_y [m], F_y [N], p_y [Pa] (plastic.py)."""
    R: float
    E: float
    w: float
    m: float
    a0: float
    a_c: float
    a_snap: float
    delta_break: float
    G_break: float
    U0: float
    c_d: float
    delta_y: float
    F_y: float
    p_y: float


def contact_law(pair: Pair, break_mode: str = "displacement") -> ContactLaw:
    """Collect the contact constants of `pair` (as dolle_sim/contact.py JKRContact.__init__ and _resolve_core do).
    Without yield (p_y = inf) delta_y and p_y are set to 1e30 as in contact.py."""
    R, E, w, m = pair.R_star, pair.E_star, pair.w, pair.m_star
    a_b = break_radius(R, w, E, break_mode)
    G_break = contact_work(a_b, R, w, E)
    a_snap = snap_on_radius(R, w, E)
    d_y = yield_overlap(R, E, pair.p_y)
    d_y, p_y = (d_y, pair.p_y) if math.isfinite(d_y) else (1e30, 1e30)
    return ContactLaw(R=R, E=E, w=w, m=m, a0=equilibrium_contact_radius(R, w, E),
                      a_c=break_radius(R, w, E, "displacement"), a_snap=a_snap,
                      delta_break=break_overlap(R, w, E, break_mode), G_break=G_break,
                      U0=contact_work(a_snap, R, w, E) - G_break, c_d=damping_coefficient(pair.e_dry, E, m),
                      delta_y=d_y, F_y=hertz_force(d_y, R, E), p_y=p_y)


def time_step(law: ContactLaw, v_in: float, n_sub: int = 300) -> float:
    """dt = min(t_Hz, t_osc)/n_sub [s], t_Hz = 2.87 (m*^2/(R* E*^2 v_in))^(1/5) the Hertz impact duration,
    t_osc = 2 pi sqrt(m*/(2 E* a0)) the period of a Hertz spring with patch a0 (the true JKR period about a0 is
    1.29x longer, module docstring).  contact.py _resolve_core."""
    v_in = max(v_in, 1e-12)
    t_hz = 2.87 * (law.m ** 2 / (law.R * law.E ** 2 * v_in)) ** 0.2
    t_osc = 2 * math.pi * math.sqrt(law.m / (2 * law.E * law.a0))
    return min(t_hz, t_osc) / n_sub


@dataclass
class ContactState:
    """What the contact remembers during one impact: patch radius a [m], overlap delta [m], deepest overlap so far,
    whether it has yielded and, if so, the flattened contact (radius R_p, dent delta_p, its break radius a_cp,
    break overlap delta_cp and G_cp = G at break), and whether the rebound has started."""
    a: float
    delta: float = 0.0
    delta_peak: float = 0.0
    force_peak: float = 0.0
    yielded: bool = False
    R_p: float = 0.0
    delta_p: float = 0.0
    a_cp: float = 0.0
    delta_cp: float = 0.0
    G_cp: float = 0.0
    rebounding: bool = False


def _yield(law: ContactLaw, s: ContactState) -> float:
    """Plastic loading at overlap s.delta > delta_y (Thornton & Ning 1998): update the flattened contact and return
    the change F_max - F_Hertz to add to the JKR force [N].  contact.py _resolve_core (plastic branch)."""
    F_max = plastic_loading_force(s.delta, law.R, law.E, law.p_y, law.delta_y)
    s.yielded = True
    s.R_p = flattened_radius(F_max, law.F_y, law.E, law.p_y)
    s.delta_p = permanent_indentation(s.delta, F_max, law.E, s.R_p)
    s.a_cp = break_radius(s.R_p, law.w, law.E, "displacement")
    s.delta_cp = break_overlap(s.R_p, law.w, law.E, "displacement")
    s.G_cp = contact_work(s.a_cp, s.R_p, law.w, law.E)
    return F_max - hertz_force(s.delta, law.R, law.E)


def contact_force(law: ContactLaw, s: ContactState) -> float | None:
    """Elastic(-plastic) contact force F_el [N] at the overlap s.delta, updating the patch radius s.a; None if the
    contact has broken.  Unloading a yielded contact: JKR of the flattened contact (R_p, delta - delta_p).
    Otherwise: JKR (jkr.py), plus the Thornton-Ning correction while loading beyond delta_y.  contact.py."""
    if s.yielded and s.delta < s.delta_peak:
        if s.delta - s.delta_p < s.delta_cp:
            return None
        s.a = contact_radius(s.delta - s.delta_p, s.R_p, law.w, law.E, s.a, s.a_cp)
        return force_of_radius(s.a, s.R_p, law.w, law.E)
    if s.delta < law.delta_break:
        return None
    s.a = contact_radius(s.delta, law.R, law.w, law.E, s.a, law.a_c)
    F = force_of_radius(s.a, law.R, law.w, law.E)
    if s.delta > law.delta_y:
        F += _yield(law, s)
    return F


def stored_energy(law: ContactLaw, s: ContactState) -> float:
    """U [J]: energy stored in the contact relative to the broken contact, G(a) - G(break), using the flattened
    contact once yielded (contact.py JKRContact.U and the bound test of _resolve_core)."""
    if s.yielded:
        return contact_work(s.a, s.R_p, law.w, law.E) - s.G_cp
    return contact_work(s.a, law.R, law.w, law.E) - law.G_break


def is_bound(law: ContactLaw, s: ContactState, v_rel: float) -> bool:
    """Stop tests 3 and 4 (module docstring), in contact.py's order: falling back in after the rebound started, or
    energy of the relative motion plus U below zero (not tested while still loading a yielded contact)."""
    if v_rel < 0:
        s.rebounding = True
    elif s.rebounding and v_rel > 0:
        return True
    if not s.yielded:
        return 0.5 * law.m * v_rel * v_rel + contact_work(s.a, law.R, law.w, law.E) - law.G_break < 0.0
    if s.delta < s.delta_peak:
        return 0.5 * law.m * v_rel * v_rel + contact_work(s.a, s.R_p, law.w, law.E) - s.G_cp < 0.0
    return False


@dataclass
class CollisionResult:
    """Outcome of one head-on collision.  outcome "stick" or "bounce"; v_in approach speed [m/s]; e restitution
    (0 if stuck); u1, u2 velocities before and v1, v2 after [m/s] (stuck: both SET to the centre-of-mass velocity,
    not measured; the trace's last row holds the measured, still vibrating, velocities);
    contact_time [s] (for a stick: until the pair was found bound); max_overlap [m]; max_force [N] the largest
    repulsive F_el (0 if the force stayed attractive, as in contact.py); yielded;
    momentum before/after [kg m/s] (with a wall: grain + impulse to the wall; for a stick momentum_after is SET
    equal to momentum_before, measure it from trace["momentum"]); trace: time series or None."""
    pair: Pair
    outcome: str
    v_in: float
    e: float
    u1: float
    u2: float
    v1: float
    v2: float
    contact_time: float
    max_overlap: float
    max_force: float
    yielded: bool
    momentum_before: float
    momentum_after: float
    trace: dict | None = field(default=None, repr=False)

    @property
    def stuck(self) -> bool:
        """True if the grains stuck."""
        return self.outcome == "stick"


def _momentum(pair: Pair, v1: float, v2: float, wall_impulse: float) -> float:
    """m1 v1 + m2 v2 [kg m/s]; for a wall (m2 = inf) m1 v1 + the impulse the wall has received."""
    return pair.m1 * v1 + (wall_impulse if math.isinf(pair.m2) else pair.m2 * v2)


def _centre_of_mass_velocity(pair: Pair, u1: float, u2: float) -> float:
    """v_cm = (m1 u1 + m2 u2)/(m1 + m2) [m/s]; a wall (m2 = inf) keeps its own velocity u2."""
    return u2 if math.isinf(pair.m2) else (pair.m1 * u1 + pair.m2 * u2) / (pair.m1 + pair.m2)


TRACE_KEYS = ("t", "overlap", "contact_radius", "force", "damping_force", "v1", "v2", "momentum", "kinetic",
              "stored", "damping_loss", "hysteresis_loss", "plastic_loss", "energy_error")
"""Columns of CollisionResult.trace (numpy arrays, SI units): time [s]; overlap [m]; contact radius [m]; elastic
contact force F_el and damping force [N]; v1, v2 [m/s]; total momentum [kg m/s]; energies [J]: kinetic energy of
the relative motion, stored contact energy U, damping loss, hysteresis loss |U0|, plastic work; energy_error = sum
of all five minus 1/2 m* v_in^2: the integrator's error, of order dt^2 while the grains touch (velocity Verlet),
plus a jump of order dt in the last row, because the break is detected only to within one step (as in the
simulator; this is why 300 and 2000 steps differ by ~1e-3 in restitution)."""


@dataclass
class _Recorder:
    """Collects the trace and the energy bookkeeping (trapezoidal integrals of the damping and contact work)."""
    law: ContactLaw
    E_in: float
    rows: dict = field(default_factory=lambda: {k: [] for k in TRACE_KEYS})
    last: tuple = (0.0, 0.0, 0.0, 0.0)      # overlap, F_el, F_damp, U at the previous row
    damping_loss: float = 0.0
    plastic_loss: float = 0.0

    def add(self, t, s: ContactState, F_el, F_damp, v1, v2, momentum, U):
        """Append one row.  Damping loss += mean damping force x d(overlap); once yielded, plastic work += mean
        F_el x d(overlap) - change of U.  The hysteresis loss |U0| is booked from the touch on."""
        d0, F0, Fd0, U_prev = self.last
        dd = s.delta - d0
        if self.rows["t"]:
            self.damping_loss += 0.5 * (Fd0 + F_damp) * dd
            if s.yielded:
                self.plastic_loss += 0.5 * (F0 + F_el) * dd - (U - U_prev)
        self.last = (s.delta, F_el, F_damp, U)
        kin = 0.5 * self.law.m * (v1 - v2) ** 2
        hyst = -self.law.U0
        total = kin + U + self.damping_loss + hyst + self.plastic_loss
        for k, x in zip(TRACE_KEYS, (t, s.delta, s.a, F_el, F_damp, v1, v2, momentum, kin, U, self.damping_loss,
                                     hyst, self.plastic_loss, total - self.E_in)):
            self.rows[k].append(x)

    def arrays(self) -> dict:
        """The trace as a dict of numpy arrays (keys TRACE_KEYS)."""
        return {k: np.array(v) for k, v in self.rows.items()}


def collide_pair(pair: Pair, u1: float, u2: float = 0.0, n_sub: int = 300, record: bool = True,
                 break_mode: str = "displacement", max_steps: int = 300000) -> CollisionResult:
    """Integrate the head-on collision of body 1 (velocity u1 [m/s]) with body 2 (u2 < u1; a wall if pair.r2 is
    None) by velocity Verlet until it sticks or bounces (module docstring).  record=True stores the time series
    (TRACE_KEYS).  Same arithmetic and stop tests as dolle_sim/contact.py _resolve_core."""
    law, v_in = contact_law(pair, break_mode), u1 - u2
    if not v_in > 0:
        raise ValueError("the bodies must approach: u1 > u2")
    dt, s, J2 = time_step(law, v_in, n_sub), ContactState(a=law.a_snap), 0.0
    rec = _Recorder(law, 0.5 * law.m * v_in ** 2) if record else None
    F_el = force_of_radius(s.a, law.R, law.w, law.E)                    # touch: patch jumps to a_snap
    F_damp = law.c_d * math.sqrt(s.a) * v_in
    if rec:
        rec.add(0.0, s, F_el, F_damp, u1, u2, _momentum(pair, u1, u2, J2), law.U0)
    if 0.5 * law.m * v_in ** 2 + law.U0 < 0.0:                         # test 1: captured on touch
        return _result(pair, s, "stick", u1, u2, u1, u2, 0.0, J2, rec)
    F_tot = F_el + F_damp
    acc1, acc2, v1, v2, t = -F_tot / pair.m1, F_tot / pair.m2, u1, u2, 0.0
    for _ in range(max_steps):
        v1h, v2h = v1 + 0.5 * dt * acc1, v2 + 0.5 * dt * acc2          # half kick
        J2 += 0.5 * dt * F_tot
        s.delta += dt * (v1h - v2h)                                     # drift
        t += dt
        F_el = contact_force(law, s)                                    # contact law at the new overlap
        if F_el is None:                                                # test 2: broken -> bounce
            return _result(pair, s, "bounce", u1, u2, v1h, v2h, t, J2, rec)
        F_damp = law.c_d * math.sqrt(s.a) * (v1h - v2h)
        F_tot = F_el + F_damp
        acc1, acc2 = -F_tot / pair.m1, F_tot / pair.m2                  # equal and opposite
        v1, v2 = v1h + 0.5 * dt * acc1, v2h + 0.5 * dt * acc2           # half kick
        J2 += 0.5 * dt * F_tot
        if s.delta > s.delta_peak:
            s.delta_peak = s.delta
            s.force_peak = max(s.force_peak, F_el)
        if rec:
            rec.add(t, s, F_el, F_damp, v1, v2, _momentum(pair, v1, v2, J2), stored_energy(law, s))
        if is_bound(law, s, v1 - v2):                                   # tests 3 and 4 -> stick
            return _result(pair, s, "stick", u1, u2, v1, v2, t, J2, rec)
    return _result(pair, s, "stick", u1, u2, v1, v2, t, J2, rec)       # safety cap, as contact.py


def _result(pair, s, outcome, u1, u2, v1, v2, t, J2, rec) -> CollisionResult:
    """Package the outcome.  Bounce: e = |v1 - v2| / v_in at the break, momentum after measured from v1, v2.
    Stick: both bodies are reported at the centre-of-mass velocity (where the damping would bring them) and
    momentum_after = momentum_before by construction; the measured values are in the trace."""
    v_in = u1 - u2
    if outcome == "bounce":
        e, after = abs(v1 - v2) / v_in, _momentum(pair, v1, v2, J2)
        if rec:
            broken = replace(s, a=0.0)
            rec.add(t, broken, 0.0, 0.0, v1, v2, after, 0.0)
    else:
        v1 = v2 = _centre_of_mass_velocity(pair, u1, u2)
        e, after = 0.0, _momentum(pair, u1, u2, 0.0)
    return CollisionResult(pair, outcome, v_in, e, u1, u2, v1, v2, t, s.delta_peak, s.force_peak, s.yielded,
                           _momentum(pair, u1, u2, 0.0), after, rec.arrays() if rec else None)


def centre_of_mass_velocities(pair: Pair, v_in: float) -> tuple[float, float]:
    """(u1, u2) [m/s] with approach speed v_in and the centre of mass at rest: u1 = v_in m2/(m1+m2),
    u2 = -v_in m1/(m1+m2); with a wall u1 = v_in, u2 = 0."""
    if math.isinf(pair.m2):
        return v_in, 0.0
    return v_in * (pair.m2 / (pair.m1 + pair.m2)), -v_in * (pair.m1 / (pair.m1 + pair.m2))


def impact(pair: Pair, v_in: float, n_sub: int = 300, record: bool = False,
           break_mode: str = "displacement") -> CollisionResult:
    """Head-on impact at approach speed v_in [m/s] in the centre-of-mass frame (what the simulator's
    JKRContact(p).resolve(v_in, m*) computes)."""
    u1, u2 = centre_of_mass_velocities(pair, v_in)
    return collide_pair(pair, u1, u2, n_sub=n_sub, record=record, break_mode=break_mode)


def collide(mat1, r1: float, mat2, r2: float | None, u1: float, u2: float = 0.0, f_adh: float = 1.0,
            plastic: bool = True, n_sub: int = 300, record: bool = True,
            break_mode: str = "displacement") -> CollisionResult:
    """Collide a grain of material mat1, radius r1 [m], velocity u1 [m/s] with a grain (mat2, r2, u2), or with a
    flat wall of mat2 when r2 is None.  Builds the Pair (pair.py make_pair) and runs collide_pair."""
    pair = make_pair(mat1, r1, mat2, r2, f_adh=f_adh, plastic=plastic)
    return collide_pair(pair, u1, u2, n_sub=n_sub, record=record, break_mode=break_mode)


def sticking_threshold(pair: Pair, tol: float = 2e-3, n_sub: int = 300, lo: float = 1e-6, hi: float = 1.0,
                       break_mode: str = "displacement") -> float:
    """Approach speed [m/s] below which the pair sticks: geometric bisection on v_in (midpoint sqrt(lo hi)) until
    hi/lo - 1 < tol, starting from lo = 1e-6 m/s and hi = 1 m/s (tripled while it still sticks).  Same search as
    dolle_sim/contact.py JKRContact.threshold.  The model is deterministic: one speed, no probability."""
    def sticks(v):
        """Does an impact at approach speed v [m/s] stick?"""
        return impact(pair, v, n_sub=n_sub, break_mode=break_mode).stuck
    if not sticks(lo):
        return lo
    while sticks(hi) and hi < 1e3:
        hi *= 3
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if sticks(mid):
            lo = mid
        else:
            hi = mid
        if hi / lo - 1 < tol:
            break
    return math.sqrt(lo * hi)

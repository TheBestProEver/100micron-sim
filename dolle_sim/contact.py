"""JKR adhesive contact model, resolved with DEM-style time integration.

A binary impact lasts ~1 microsecond for 100 um grains, while the flight lasts
minutes, so the main simulation does not integrate contacts on its global time
step. Instead every detected impact is handed to `JKRContact.resolve`, which
integrates the normal-direction equation of motion

    m_eff * d2(delta)/dt2 = -( F_JKR(delta) + F_damp(delta, d(delta)/dt) )

with a ~nanosecond step until the grains either separate (bounce) or become
energetically bound (stick). Physics included:

* JKR elastic-adhesive force with the standard hysteresis: the contact snaps
  on at delta = 0 during approach and only breaks at delta = delta_c < 0
  (Johnson, Kendall & Roberts 1971; Chokshi, Tielens & Hollenbach 1993).
* Viscoelastic damping tuned so a non-adhesive impact has restitution e_dry
  (Tsuji et al. 1992 form, as used in LIGGGHTS/EDEM).
* Plastic yield (Thornton & Ning 1998): once the peak (Hertz) contact pressure reaches
  the limiting pressure p_y the loading curve becomes linear,
      F = F_y + pi p_y R (delta - delta_y),
  and unloading is elastic with an enlarged radius of curvature R_p and a
  permanent indentation delta_p. During loading the Hertz part of the JKR
  force is replaced by this law; unloading follows the JKR relations of the
  flattened contact (radius R_p, overlap delta - delta_p), so the pull-off
  force rises to 1.5 pi w R_p and the contact breaks at
  delta_p + delta_c(R_p) (plastic flattening increases adhesion).

The result is a physically-derived sticking threshold that depends on
material (E, nu, gamma, rho), grain size and the effective mass of the bodies
(important once grains are part of an aggregate).

JKR relations used (a = contact radius, w = work of adhesion):
    delta(a) = a^2/R - sqrt(2 pi w a / E*)
    F(a)     = 4 E* a^3 / (3R) - sqrt(8 pi w E* a^3)        (F > 0 repulsive)

Provenance: Equations from the papers cited above (Johnson, Kendall & Roberts
1971; Chokshi et al. 1993; Tsuji et al. 1992; Thornton & Ning 1998); the time
integration of each impact is our own. No REBOUND code. The material values
fed in (materials.py) are inputs, several of them assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .materials import PairProps


@dataclass
class ContactResult:
    stuck: bool
    v_in: float           # normal approach speed [m/s]
    v_out: float          # normal separation speed [m/s] (0 if stuck)
    e_n: float            # normal restitution
    max_overlap: float    # [m]
    max_force: float      # [N]
    contact_time: float   # [s] (time integrated; for sticks, time until bound)
    max_contact_radius: float = 0.0   # [m]
    yielded: bool = False             # plastic deformation occurred


class JKRContact:
    """Normal-direction JKR contact for one material/size pair."""

    def __init__(self, props: PairProps, steps: int = 300, separation: str = "displacement"):
        """separation: where an elastic contact breaks during a collision.
        "displacement" (default): at the end of the stable JKR branch, a = a0/6^(2/3), delta = delta_c, as
            in Chokshi et al. (1993) and Dominik & Tielens (1997);
        "force": at the maximum tensile force, a = a0/4^(1/3), F = -F_c, as LAMMPS pair_style granular jkr
            does (used for the LAMMPS cross-check). The two give different hysteresis energies (see
            provenance.py)."""
        p = self.p = props
        self.steps = int(steps)            # integration sub-steps per contact time (300: ~0.06% accuracy)
        e = max(min(props.e_dry, 0.9999), 1e-4)
        ln_e = math.log(e)
        self.beta = ln_e / math.sqrt(ln_e ** 2 + math.pi ** 2)   # negative
        self.A = 4 * p.E_star / (3 * p.R_star)
        self.B = math.sqrt(8 * math.pi * p.w * p.E_star)
        self.C = math.sqrt(2 * math.pi * p.w / p.E_star)
        self.a_c = p.a0 / 6.0 ** (2.0 / 3.0)
        if separation == "displacement":
            a_b = self.a_c
        elif separation == "force":
            a_b = p.a0 / 4.0 ** (1.0 / 3.0)
        else:
            raise ValueError("separation must be 'displacement' or 'force'")
        self.separation = separation
        self.delta_break = a_b * a_b / p.R_star - self.C * math.sqrt(a_b)
        self.G_c = self._G(a_b)                    # energy reference: the broken contact
        self.a_snap = self.solve_a(0.0, p.a0)
        if math.isfinite(p.p_y):
            self.delta_y = p.R_star * (math.pi * p.p_y / (2 * p.E_star)) ** 2
        else:
            self.delta_y = math.inf
        self.U0 = self.U(self.a_snap)              # well depth at snap-on (< 0)

    # ------------------------------------------------------------ JKR algebra
    def delta_of_a(self, a):
        return a * a / self.p.R_star - self.C * math.sqrt(a)

    def force_of_a(self, a):
        return self.A * a ** 3 - self.B * a ** 1.5

    def solve_a(self, delta: float, a_guess: float) -> float:
        """Invert delta(a) on the stable branch a >= a_c (Newton, warm-started)."""
        R, C, a_c = self.p.R_star, self.C, self.a_c
        a = max(a_guess, a_c)
        for _ in range(30):
            g = a * a / R - C * math.sqrt(a) - delta
            dg = 2 * a / R - 0.5 * C / math.sqrt(a)
            if dg <= 0:
                a = a * 1.5
                continue
            step = g / dg
            a_new = a - step
            if a_new < a_c:
                a_new = 0.5 * (a + a_c)
            if abs(a_new - a) < 1e-12 * a:
                return a_new
            a = a_new
        return a

    def _G(self, a):
        # antiderivative of F(a) * d(delta)/da
        A, B, C, R = self.A, self.B, self.C, self.p.R_star
        return (2 * A / (5 * R)) * a ** 5 - (A * C / 2 + 2 * B / R) / 3.5 * a ** 3.5 + (B * C / 4) * a * a

    def U(self, a):
        """Potential energy relative to the broken contact (U = 0 at delta_c)."""
        return self._G(a) - self.G_c

    # -------------------------------------------------------------- estimates
    def v_stick_analytic(self, m_eff: float) -> float:
        """Simple energy estimate: 1/2 m v^2 = E_break = 1.54 F_c |delta_c| (the contact break energy of
        Chokshi et al. 1993 / Dominik & Tielens 1997). This is NOT the closed-form critical velocity quoted in
        Chokshi et al. 1993, and it ignores viscoelastic loss; stick/bounce is always decided by resolve()."""
        return math.sqrt(2 * self.p.E_break / m_eff)

    def v_stick_elastic(self, m_eff: float) -> float:
        """Threshold from the JKR hysteresis alone (no viscoelastic loss)."""
        return math.sqrt(2 * abs(self.U0) / m_eff)

    def v_yield(self, m_eff: float) -> float:
        """Impact speed at which the peak Hertz pressure reaches the yield pressure."""
        if not math.isfinite(self.delta_y):
            return math.inf
        U = 8.0 / 15.0 * self.p.E_star * math.sqrt(self.p.R_star) * self.delta_y ** 2.5
        return math.sqrt(2 * U / m_eff)

    # ---------------------------------------------------------------- resolve
    def resolve(self, v_in: float, m_eff: float) -> ContactResult:
        """Integrate a head-on (normal) impact at approach speed v_in > 0."""
        p = self.p
        dy = self.delta_y if math.isfinite(self.delta_y) else 1e30
        py = p.p_y if math.isfinite(p.p_y) else 1e30
        out = _resolve_core(p.R_star, self.C, self.A, self.B, self.a_c, self.G_c, self.a_snap,
                            self.U0, self.delta_break, p.E_star, p.a0, self.beta, abs(v_in), m_eff, dy, py,
                            float(self.steps))
        stuck, v_out, d_peak, F_peak, t, a_peak, yielded = out
        v_in = max(abs(v_in), 1e-12)
        return ContactResult(bool(stuck), v_in, v_out, v_out / v_in, d_peak, F_peak, t, a_peak, bool(yielded))

    def threshold(self, m_eff: float, lo: float = 1e-6, hi: float = 1.0, tol: float = 2e-3) -> float:
        """Bisection for the DEM sticking velocity (deterministic model)."""
        if not self.resolve(lo, m_eff).stuck:
            return lo
        while self.resolve(hi, m_eff).stuck and hi < 1e3:
            hi *= 3
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            if self.resolve(mid, m_eff).stuck:
                lo = mid
            else:
                hi = mid
            if hi / lo - 1 < tol:
                break
        return math.sqrt(lo * hi)


# ----------------------------------------------------------------------------
# Numerical core (plain functions so they can be JIT-compiled by numba).
try:
    from numba import njit
except ImportError:          # pragma: no cover - numba is optional
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _solve_a(delta, a_guess, R, C, a_c):
    a = max(a_guess, a_c)
    for _ in range(30):
        g = a * a / R - C * math.sqrt(a) - delta
        dg = 2 * a / R - 0.5 * C / math.sqrt(a)
        if dg <= 0:
            a = a * 1.5
            continue
        a_new = a - g / dg
        if a_new < a_c:
            a_new = 0.5 * (a + a_c)
        if abs(a_new - a) < 1e-12 * a:
            return a_new
        a = a_new
    return a


@njit(cache=True)
def _G(a, A, B, C, R):
    return (2 * A / (5 * R)) * a ** 5 - (A * C / 2 + 2 * B / R) / 3.5 * a ** 3.5 + (B * C / 4) * a * a


@njit(cache=True)
def _resolve_core(R, C, A, B, a_c, G_c, a_snap, U0, delta_c, E_star, a0, beta, v_in, m_eff, delta_y, p_y,
                  n_sub=300.0):
    v_in = max(v_in, 1e-12)
    # time step from the faster of the Hertz impact time and the JKR well oscillation
    t_hz = 2.87 * (m_eff ** 2 / (R * E_star ** 2 * v_in)) ** 0.2
    t_osc = 2 * math.pi * math.sqrt(m_eff / (2 * E_star * a0))
    dt = min(t_hz, t_osc) / n_sub
    damp_c = -2 * math.sqrt(5.0 / 6.0) * beta * math.sqrt(m_eff * 2 * E_star)
    kH = 4.0 / 3.0 * E_star * math.sqrt(R)
    F_y = kH * delta_y ** 1.5
    w_adh = C * C * E_star / (2 * math.pi)              # work of adhesion (C = sqrt(2 pi w / E*))
    # snap-on: contact forms at delta = 0 with a finite contact radius
    if 0.5 * m_eff * v_in ** 2 + U0 < 0.0:
        return True, 0.0, 0.0, 0.0, 0.0, a_snap, False
    d, v, t, a = 0.0, v_in, 0.0, a_snap
    acc = -((A * a ** 3 - B * a ** 1.5) + damp_c * math.sqrt(a) * v) / m_eff
    d_peak = 0.0
    F_peak = 0.0
    a_peak = a
    yielded = False
    # flattened-contact (post-yield) JKR geometry, Thornton & Ning (1998): radius R_p, offset d_p
    R_p = R
    d_p = 0.0
    A_p = A
    a_cp = a_c
    dc_p = delta_c
    G_cp = G_c
    unloaded = False
    for _ in range(300000):
        v_half = v + 0.5 * dt * acc
        d += dt * v_half
        t += dt
        if yielded and d < d_peak:
            # elastic unloading / reloading of the flattened contact: JKR with R_p in d - d_p
            ds = d - d_p
            if ds < dc_p:                                   # flattened contact breaks -> bounce
                return False, abs(v_half), d_peak, F_peak, t, a_peak, yielded
            a = _solve_a(ds, a, R_p, C, a_cp)
            F_el = A_p * a ** 3 - B * a ** 1.5
        else:
            if d < delta_c:                                 # contact broken -> bounce
                return False, abs(v_half), d_peak, F_peak, t, a_peak, yielded
            a = _solve_a(d, a, R, C, a_c)
            F_el = A * a ** 3 - B * a ** 1.5                # JKR (elastic + adhesive)
            if d > delta_y:
                # plastic loading: the Hertz part is replaced by the linear limiting-pressure law
                yielded = True
                F_max = F_y + math.pi * p_y * R * (d - delta_y)
                F_el += F_max - kH * d ** 1.5
                R_p = (4 * E_star / (3 * F_max)) * ((2 * F_max + F_y) / (2 * math.pi * p_y)) ** 1.5
                kP = 4.0 / 3.0 * E_star * math.sqrt(R_p)
                d_p = d - (F_max / kP) ** (2.0 / 3.0)
                A_p = 4 * E_star / (3 * R_p)
                a0_p = (4.5 * math.pi * w_adh * R_p ** 2 / E_star) ** (1.0 / 3.0)
                a_cp = a0_p / 6.0 ** (2.0 / 3.0)
                dc_p = a_cp ** 2 / R_p - C * math.sqrt(a_cp)
                G_cp = _G(a_cp, A_p, B, C, R_p)
        acc = -(F_el + damp_c * math.sqrt(a) * v_half) / m_eff
        v_new = v_half + 0.5 * dt * acc
        if d > d_peak:
            d_peak = d
            a_peak = a
            if F_el > F_peak:
                F_peak = F_el
        if v_new < 0:
            unloaded = True
        elif unloaded and v_new > 0:
            # failed to escape on the rebound and is falling back in: bound
            return True, 0.0, d_peak, F_peak, t, a_peak, yielded
        v = v_new
        # bound: total energy below the break threshold of the current contact -> can never escape
        if not yielded:
            if 0.5 * m_eff * v * v + _G(a, A, B, C, R) - G_c < 0.0:
                return True, 0.0, d_peak, F_peak, t, a_peak, yielded
        elif d < d_peak:
            if 0.5 * m_eff * v * v + _G(a, A_p, B, C, R_p) - G_cp < 0.0:
                return True, 0.0, d_peak, F_peak, t, a_peak, yielded
    return True, 0.0, d_peak, F_peak, t, a_peak, yielded

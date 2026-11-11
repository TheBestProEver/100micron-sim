"""Damping: why a non-sticky impact still does not bounce back at full speed.

Real solids turn part of the elastic energy of an impact into heat (internal friction, elastic waves).  The
simulator models this with a damping force proportional to the approach speed v = d(delta)/dt [m/s]:

    F_damp = c_d sqrt(a) v           [N],   c_d = -2 sqrt(5/6) beta sqrt(2 E* m*)   [kg m^-1/2 s^-1]

a = contact patch radius [m], E* reduced modulus [Pa], m* reduced mass [kg], beta a pure number set by the dry
restitution e_dry (below).  It always opposes the relative motion.

Where it comes from.  A Hertz contact is a spring whose stiffness grows with the patch: k = dF/d(delta) = 2 E* a.
A spring k with a mass m* and a dashpot of "damping ratio" zeta rebounds with restitution
e = exp(-pi zeta / sqrt(1 - zeta^2)); solving for the damping gives  beta = ln(e) / sqrt(ln(e)^2 + pi^2)  (= -zeta)
and a dashpot 2 |beta| sqrt(k m*).  Tsuji, Tanaka & Ishida (1992, Powder Technol. 71, 239) wrote the dashpot of a
Hertz contact as eta = alpha sqrt(m* k_H) delta^(1/4), k_H = (4/3) E* sqrt(R*), with alpha read from a chart
against e.  The form above is the same: with a = sqrt(R* delta) (Hertz),
    c_d sqrt(a) = 2 sqrt(5/6) |beta| sqrt(2 E* a m*) = 2 sqrt(5/6) |beta| sqrt(1.5 k_H m*) delta^(1/4)
                = sqrt(5) |beta| sqrt(k_H m*) delta^(1/4)     (2 E* a = 1.5 k_H delta^(1/2)),
i.e. Tsuji's law with alpha = sqrt(5) |beta|; this is how DEM (discrete element method) codes such as LIGGGHTS
and EDEM write it.  Using the JKR patch radius a (so that the damping also acts while the neck is stretched at
negative overlap) is the simulator's own adaptation (PROVENANCE.md, "Viscoelastic damping": "that adaptation is
ours").

Why the restitution does not depend on speed (Hertz, no adhesion): m* delta'' = -K delta^1.5 - c R*^(1/4)
delta^(1/4) delta' with K = (4/3) E* sqrt(R*), c = c_d.  Put delta = D x, t = T s and choose T so that
m* D/T^2 = K D^1.5; dividing by K D^1.5 leaves x'' = -x^1.5 - [c R*^(1/4)/sqrt(K m*)] x^(1/4) x', whose
damping group contains no D, so the impact speed (which only fixes D) cannot change e.  With the c_d above the
group is sqrt(5) |beta|: e depends on e_dry alone.

Calibration (approximate).  beta(e) is the LINEAR-oscillator relation applied to the non-linear Hertz spring, with
the factor sqrt(5/6); it is not exact.  notebooks/01 (section 3) finds the prefactor that returns e_dry exactly
(0.912-0.914 against sqrt(5/6) = 0.9129) and the residual |e - e_dry| <= 6.1e-4 for e_dry = 0.5-0.95 (the
simulator's check 'damping': 0.8501 for 0.85; tests/test_dolle_core.py repeats it).  An analytic treatment of the
damped Hertz spring is Antypov & Elliott (2011, EPL 94, 50004; not checked here).  The e_dry values themselves
are ASSUMED (materials.py).

Why this form: (1) it is the standard, published form for Hertz contacts; (2) its strength scales with the
contact the same way as the elastic force, so the restitution does not depend on impact speed (as assumed); (3)
with it, the whole sticking problem has one free number besides the materials, e_dry, so the damped sticking
speed is the undamped one times a factor f(e_dry) (scaling.py).  Alternatives: a constant dashpot (restitution
then depends on speed); the viscoelastic law of Brilliantov, Spahn, Hertzsch & Poeschel (1996, Phys. Rev. E 53,
5382), whose restitution falls with speed and which needs dissipative constants not known for these grains.

Clamping: the 1-D impact model clamps e_dry to [1e-4, 0.9999] (dolle_sim/contact.py JKRContact.__init__), so it
can never be exactly undamped there.  The 3-D model (dolle_sim/dem.py make_params) clamps to [1e-4, 1] and uses
beta = 0 at e = 1.  This core follows the 1-D clamp, except that e_dry >= 1 means "no damping" (beta = 0), as in
the 3-D model, so the elastic limit can be shown exactly.
"""
from __future__ import annotations

import math


def damping_beta(e_dry: float) -> float:
    """beta = ln(e)/sqrt(ln(e)^2 + pi^2) (no unit, <= 0), e = e_dry clamped to [1e-4, 0.9999] as in contact.py;
    e_dry >= 1 gives beta = 0 (no damping, as in dem.py make_params)."""
    if e_dry >= 1.0:
        return 0.0
    e = max(min(e_dry, 0.9999), 1e-4)
    ln_e = math.log(e)
    return ln_e / math.sqrt(ln_e ** 2 + math.pi ** 2)


def damping_coefficient(e_dry: float, E: float, m: float) -> float:
    """c_d = -2 sqrt(5/6) beta sqrt(2 E* m*) [kg m^-1/2 s^-1], so that F_damp = c_d sqrt(a) v.  E = E* [Pa],
    m = m* [kg].  Tsuji et al. (1992); contact.py _resolve_core (damp_c), dem.py make_params (P_DAMPC)."""
    return -2 * math.sqrt(5.0 / 6.0) * damping_beta(e_dry) * math.sqrt(m * 2 * E)

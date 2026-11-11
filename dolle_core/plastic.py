"""Plastic yield (Thornton & Ning 1998): what happens when an impact is hard enough to dent the grains.

Source: Thornton & Ning (1998), "A theoretical model for the stick/bounce behaviour of adhesive, elastic-plastic
spheres", Powder Technol. 99, 154.  Code: dolle_sim/contact.py _resolve_core (plastic branch), extras/dolle_extras/dem.py.

In a Hertz contact the pressure is highest at the centre of the patch, p_max = 2 E* a / (pi R*) (Johnson 1985,
"Contact Mechanics").  A real solid cannot carry more than a limiting pressure p_y [Pa].  Setting p_max = p_y with
a = sqrt(R* delta) gives the overlap at which yield starts:

    delta_y = R* (pi p_y / (2 E*))^2            [m],     F_y = (4/3) E* sqrt(R*) delta_y^1.5   [N]

Beyond it the pressure in the yielded part of the patch is capped at p_y.  Thornton & Ning ASSUME that the patch
area keeps growing as pi R* delta, so each extra bit of overlap adds force at the rate pi p_y R*; the
loading curve becomes a straight line (written by them for the NON-adhesive contact):

    F_max(delta) = F_y + pi p_y R* (delta - delta_y)

(Useful identity: pi p_y R* delta_y = pi p_y a_y^2 = (3/2) F_y, with a_y = sqrt(R* delta_y) = pi R* p_y/(2 E*).)

Unloading is elastic again, but from a FLATTENED sphere: Thornton & Ning give its larger radius of curvature R_p
and the permanent dent delta_p,

    R_p     = (4 E* / (3 F_max)) ((2 F_max + F_y) / (2 pi p_y))^1.5
    delta_p = delta_max - (F_max / ((4/3) E* sqrt(R_p)))^(2/3)

and the contact then follows the JKR relations of jkr.py with R* -> R_p and delta -> delta - delta_p (so the
flattened contact is stickier: its pull-off force is 1.5 pi w R_p).  With adhesion, the simulator's CHOICE
(contact.py _resolve_core, plastic branch) while loading beyond delta_y is F_el = F_JKR(a) + (F_max(delta) -
F_Hertz(delta)): the JKR force with the Hertz part replaced by the plastic loading line.

Why plasticity is in the model: the simulator must handle fast impacts as well (restitution falls once grains
dent).  For the sticking speeds it is irrelevant: yield starts at v_y = 11.1 m/s (silica pairs, 50 um radius) and
0.84 m/s (graphite pairs) against thresholds of 6-38 mm/s, and thresholds with yield on and off agree to all
printed digits (notebooks/01, section 11; tests/test_dolle_core.py checks the elastic and plastic branches).  All yield pressures are ASSUMED (materials.py).

Check: without adhesion and damping, the restitution must follow Thornton & Ning's closed form
thornton_ning_restitution (the simulator's check 'thornton_ning': within 1e-4).
"""
from __future__ import annotations

import math


def yield_overlap(R: float, E: float, p_y: float) -> float:
    """delta_y = R (pi p_y/(2E))^2 [m]: overlap at which the peak Hertz pressure reaches p_y [Pa]; inf if p_y is
    inf.  R = R*, E = E*.  Thornton & Ning (1998); contact.py JKRContact.__init__ (delta_y)."""
    if not math.isfinite(p_y):
        return math.inf
    return R * (math.pi * p_y / (2 * E)) ** 2


def hertz_force(delta: float, R: float, E: float) -> float:
    """Hertz force F = (4/3) E sqrt(R) delta^1.5 [N] for overlap delta [m] (Hertz 1882; contact.py kH)."""
    return 4.0 / 3.0 * E * math.sqrt(R) * delta ** 1.5


def plastic_loading_force(delta: float, R: float, E: float, p_y: float, delta_y: float) -> float:
    """F_max = F_y + pi p_y R (delta - delta_y) [N]: force on the plastic loading line at overlap delta [m]
    beyond delta_y.  Thornton & Ning (1998); contact.py _resolve_core."""
    return hertz_force(delta_y, R, E) + math.pi * p_y * R * (delta - delta_y)


def flattened_radius(F_max: float, F_y: float, E: float, p_y: float) -> float:
    """R_p = (4E/(3 F_max)) ((2 F_max + F_y)/(2 pi p_y))^1.5 [m]: radius of curvature of the flattened contact
    after loading to F_max [N].  Thornton & Ning (1998); contact.py _resolve_core."""
    return (4 * E / (3 * F_max)) * ((2 * F_max + F_y) / (2 * math.pi * p_y)) ** 1.5


def permanent_indentation(delta: float, F_max: float, E: float, R_p: float) -> float:
    """delta_p = delta - (F_max/((4/3) E sqrt(R_p)))^(2/3) [m]: the dent left after unloading from overlap delta
    and force F_max.  Thornton & Ning (1998); contact.py _resolve_core."""
    return delta - (F_max / (4.0 / 3.0 * E * math.sqrt(R_p))) ** (2.0 / 3.0)


def yield_speed(m: float, R: float, E: float, p_y: float) -> float:
    """v_y = sqrt(2 U_y / m) [m/s], U_y = (8/15) E sqrt(R) delta_y^2.5 [J] (the Hertz energy at delta_y): the
    impact speed at which yield starts.  m = m*.  contact.py JKRContact.v_yield."""
    d_y = yield_overlap(R, E, p_y)
    if not math.isfinite(d_y):
        return math.inf
    return math.sqrt(2 * (8.0 / 15.0 * E * math.sqrt(R) * d_y ** 2.5) / m)


def thornton_ning_restitution(v: float, v_y: float) -> float:
    """Thornton & Ning (1998) restitution of an elastic-plastic, non-adhesive, undamped impact at speed v > v_y:
    with x = v_y/v,  e = (6 sqrt(3)/5)^(1/2) (1 - x^2/6)^(1/2) (x / (x + 2 sqrt(6/5 - x^2/5)))^(1/4).
    Same expression as dolle_sim/verification.py _e_tn (1 for v <= v_y)."""
    x = v_y / v
    if x >= 1.0:
        return 1.0
    return (6 * math.sqrt(3) / 5) ** 0.5 * (1 - x * x / 6) ** 0.5 * (x / (x + 2 * math.sqrt(6 / 5 - x * x / 5))) ** 0.25

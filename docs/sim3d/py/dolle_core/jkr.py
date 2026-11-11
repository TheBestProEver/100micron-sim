"""JKR contact mechanics: the force between two sticky elastic spheres.

Symbols (units): R = reduced radius R* [m], E = reduced modulus E* [Pa], w = work of adhesion [J/m^2],
a = radius of the circular contact patch [m], delta = overlap [m] (how far the undeformed spheres would
interpenetrate; > 0 pressed together, < 0 a stretched "neck"), F = normal force [N], positive = pushing apart.

1. No stickiness (Hertz 1882).  Pressing a sphere on a flat gives a contact patch of radius a with
       delta = a^2 / R,        F = 4 E a^3 / (3 R)        (so F = (4/3) E sqrt(R) delta^1.5).

2. With stickiness (Johnson, Kendall & Roberts 1971, "Surface energy and the contact of elastic solids", Proc.
   R. Soc. A 324, 301).  Surface energy pulls the surfaces together, so at a given load the patch is LARGER than
   Hertz's.  JKR find the patch radius by an energy balance: the elastic energy stored in the solids plus the
   surface energy -w pi a^2 released by the patch must be stationary in a.  The result is the Hertz pressure plus
   a tensile "flat punch" pressure at the edge, which adds a second term to each relation:
       delta(a) = a^2 / R  -  sqrt(2 pi w a / E)
       F(a)     = 4 E a^3 / (3 R)  -  sqrt(8 pi w E a^3)
   Everything else follows from these two lines (derived step by step in the notebooks):
       a0      = (9 pi w R^2 / (2 E))^(1/3)    patch radius with no load (F = 0)
       F_c     = (3/2) pi w R                  pull-off force: the largest steady pull the contact holds
                                               (reached at a = a0 / 4^(1/3))
       a_snap  = (2 pi w R^2 / E)^(1/3)        patch radius at delta = 0: on first touch the patch jumps from 0 to
                                               a_snap and the force to -(8/9) F_c (attractive)
       a_c     = a0 / 6^(2/3)                  end of the stable branch (d delta / da = 0): below it no
                                               equilibrium exists, so in a collision the contact breaks here, at
       delta_c = -(3/4) (pi^2 w^2 R / E^2)^(1/3)   (Chokshi, Tielens & Hollenbach 1993; Dominik & Tielens 1997)

3. Hysteresis.  The contact forms at delta = 0 but breaks only at delta_c < 0.  Going round that loop costs the
   energy |U0| = G(a_c) - G(a_snap), where G(a) is the work done along the curve (the integral of F d delta).  This
   lost energy is why slow grains stick: if the impact brings less kinetic energy (of the relative motion) than
   |U0| (plus what damping takes, damping.py), the grains cannot escape.

Why JKR and not DMT or Maugis-Dugdale (see tabor_parameter): JKR is the limit of a large Tabor parameter (soft,
sticky, large spheres).  For these 100 um grains it is ~7-14 for graphite (JKR applies) but ~1-2 for silica, in the
JKR-DMT transition, where JKR overstates the hysteresis (PROVENANCE.md, "JKR adhesive contact").  The full
simulator uses JKR for all contacts; so does this core.

Source of every function: dolle_sim/contact.py (JKRContact, _solve_a, _G) and dolle_sim/materials.py
(pair_props); the code below performs the same arithmetic in the same order, so results agree to rounding.
"""
from __future__ import annotations

import math


def equilibrium_contact_radius(R: float, w: float, E: float) -> float:
    """a0 = (4.5 pi w R^2 / E)^(1/3)  [m]: patch radius at zero load, F(a0) = 0.  JKR (1971); pair_props."""
    return (4.5 * math.pi * w * R ** 2 / E) ** (1.0 / 3.0)


def pull_off_force(R: float, w: float) -> float:
    """F_c = 1.5 pi w R  [N]: largest steady pull the contact holds (JKR 1971); pair_props."""
    return 1.5 * math.pi * w * R


def overlap_of_radius(a: float, R: float, w: float, E: float) -> float:
    """delta(a) = a^2/R - sqrt(2 pi w a / E)  [m] (JKR 1971); contact.py JKRContact.delta_of_a."""
    return a * a / R - math.sqrt(2 * math.pi * w / E) * math.sqrt(a)


def force_of_radius(a: float, R: float, w: float, E: float) -> float:
    """F(a) = 4 E a^3/(3R) - sqrt(8 pi w E a^3)  [N], > 0 repulsive (JKR 1971); JKRContact.force_of_a."""
    return (4 * E / (3 * R)) * a ** 3 - math.sqrt(8 * math.pi * w * E) * a ** 1.5


def contact_work(a: float, R: float, w: float, E: float) -> float:
    """G(a) = integral of F(a) (d delta/da) da  [J]: work done on the contact along the JKR curve up to patch
    radius a (an antiderivative; only differences of G mean anything).  With A = 4E/(3R), B = sqrt(8 pi w E),
    C = sqrt(2 pi w/E):  G = (2A/(5R)) a^5 - (A C/2 + 2B/R) a^3.5 / 3.5 + (B C/4) a^2.  contact.py _G."""
    A, B, C = 4 * E / (3 * R), math.sqrt(8 * math.pi * w * E), math.sqrt(2 * math.pi * w / E)
    return (2 * A / (5 * R)) * a ** 5 - (A * C / 2 + 2 * B / R) / 3.5 * a ** 3.5 + (B * C / 4) * a * a


def break_radius(R: float, w: float, E: float, mode: str = "displacement") -> float:
    """Patch radius at which a contact breaks [m].  mode "displacement" (default, every simulator run): a0/6^(2/3),
    the end of the stable branch (Chokshi et al. 1993); "force": a0/4^(1/3), where F = -F_c (what LAMMPS does;
    used only for the simulator's LAMMPS cross-check).  contact.py JKRContact.__init__."""
    a0 = equilibrium_contact_radius(R, w, E)
    if mode == "displacement":
        return a0 / 6.0 ** (2.0 / 3.0)
    if mode == "force":
        return a0 / 4.0 ** (1.0 / 3.0)
    raise ValueError("mode must be 'displacement' or 'force'")


def break_overlap(R: float, w: float, E: float, mode: str = "displacement") -> float:
    """delta at which the contact breaks [m] (< 0).  Displacement mode: -(3/4)(pi^2 w^2 R/E^2)^(1/3).
    Computed as delta(a_break) like contact.py JKRContact.__init__ (delta_break)."""
    a_b = break_radius(R, w, E, mode)
    return a_b * a_b / R - math.sqrt(2 * math.pi * w / E) * math.sqrt(a_b)


def contact_radius(delta: float, R: float, w: float, E: float, a_guess: float, a_min: float) -> float:
    """Patch radius a [m] on the stable branch (a >= a_min, the break radius) that gives overlap delta [m]:
    Newton's method on g(a) = a^2/R - C sqrt(a) - delta, C = sqrt(2 pi w/E), warm-started at a_guess; step
    up x1.5 while g'(a) <= 0; halve towards a_min if a step overshoots it; stop when the change is below 1e-12 a.
    Same iteration as contact.py _solve_a."""
    C = math.sqrt(2 * math.pi * w / E)
    a = max(a_guess, a_min)
    for _ in range(30):
        g = a * a / R - C * math.sqrt(a) - delta
        dg = 2 * a / R - 0.5 * C / math.sqrt(a)
        if dg <= 0:
            a = a * 1.5
            continue
        a_new = a - g / dg
        if a_new < a_min:
            a_new = 0.5 * (a + a_min)
        if abs(a_new - a) < 1e-12 * a:
            return a_new
        a = a_new
    return a


def snap_on_radius(R: float, w: float, E: float) -> float:
    """a_snap [m]: patch radius at delta = 0, the instant the grains touch.  Closed form (2 pi w R^2/E)^(1/3)
    = (4/9)^(1/3) a0; computed here as the simulator does (Newton from a0, contact.py JKRContact.a_snap)."""
    a_c = break_radius(R, w, E, "displacement")
    return contact_radius(0.0, R, w, E, equilibrium_contact_radius(R, w, E), a_c)


def hysteresis_energy(R: float, w: float, E: float, mode: str = "displacement") -> float:
    """|U0| = G(a_break) - G(a_snap)  [J]: energy lost in one make-and-break cycle of the contact (contact.py
    JKRContact.U0, sign flipped).  For the displacement break |U0| = 7.0922 w^(5/3) R^(4/3) E^(-2/3)."""
    a_b = break_radius(R, w, E, mode)
    return contact_work(a_b, R, w, E) - contact_work(snap_on_radius(R, w, E), R, w, E)


def tabor_parameter(R: float, w: float, E: float, z0: float = 0.3e-9) -> float:
    """Tabor parameter mu_T = (R w^2 / (E^2 z0^3))^(1/3) (no unit): the height of the adhesive neck compared with
    the range z0 [m] of the surface forces (Tabor 1977, J. Colloid Interface Sci. 58, 2; Johnson & Greenwood 1997,
    J. Colloid Interface Sci. 192, 326).  mu_T >> 1 (above ~3-5): JKR; mu_T << 1: DMT; between: Maugis-Dugdale.
    z0 = 0.3 nm is ASSUMED (the simulator's notes use an interaction range of 0.2-0.4 nm)."""
    return (R * w * w / (E * E * z0 ** 3)) ** (1.0 / 3.0)

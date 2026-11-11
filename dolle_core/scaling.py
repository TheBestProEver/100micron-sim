"""The sticking speed by hand: an energy balance, and what changes it.

Undamped (no damping, no yield).  The grains stick if the kinetic energy of their relative motion is less than
the energy the contact keeps in one make-and-break cycle, the hysteresis |U0| (jkr.py):

    1/2 m* v^2 < |U0|      ->      v_stick,0 = sqrt(2 |U0| / m*)                         [m/s]

|U0| = G(a_c) - G(a_snap), with G(a) = 8 E* a^5/(15 R*^2) - (4/(3 R*)) sqrt(2 pi w E*) a^3.5 + pi w a^2 the work
done along the JKR curve (jkr.py contact_work).  Units alone do not fix the exponents (w R*^2 and E* R*^3 are
energies too); a substitution does.  Put a = s a0 and use pi w = 2 E* a0^3/(9 R*^2) (the definition of a0), so
sqrt(2 pi w E*) = (2 E*/(3 R*)) a0^1.5:

    G = (E* a0^5 / R*^2) g(s),    g(s) = 8 s^5/15 - 8 s^3.5/9 + 2 s^2/9,
    E* a0^5 / R*^2 = (9 pi/2)^(5/3) w^(5/3) R*^(4/3) E*^(-2/3).

The landmarks are fixed fractions of a0 (a_c = 6^(-2/3) a0, a_snap = (4/9)^(1/3) a0), so

    |U0| = k_U w^(5/3) R*^(4/3) E*^(-2/3),    k_U = (9 pi/2)^(5/3) [g(6^(-2/3)) - g((4/9)^(1/3))] = 7.0922

(hysteresis_constant computes it from the curve; tests/test_dolle_core.py test_scaling_law_prefactors_and_exactness).

Two equal spheres of radius r (R* = r/2, m* = (16/3) pi R*^3 rho, w = 2 gamma) and a sphere on a wall (R* = r,
m* = (4/3) pi r^3 rho) then give

    v_stick,0 = 1.6394 gamma^(5/6) E*^(-1/3) R*^(-5/6) rho^(-1/2)      (equal spheres)
    v_stick,0 = 1.8402 w^(5/6) E*^(-1/3) R*^(-5/6) rho^(-1/2)          (sphere on a wall)

So stickier (gamma up), softer (E* down), smaller (R down) and lighter (rho down) grains stick at higher speeds.

Damped.  With damping the contact law has no other free number than e_dry: measured in units of F_c and delta_c
the damping force is beta x a pure number (damping.py).  Hence the damped threshold is the undamped one times a
factor that depends on e_dry ONLY, not on the material, size or mass (tests/test_dolle_core.py
test_damping_factor_depends_on_e_dry_only: silica, graphite, a wall and 20 um grains agree within 3e-5):

    v_stick = f(e_dry) v_stick,0,     (damping_factor computes f)
    f(0.85) = 1.6142, f(0.70) = 2.4589 at 2000 steps per contact (damping_factor's default);
    f(0.85) = 1.6148, f(0.70) = 2.4611 at 300 steps per contact (the simulator's; notebook 03, section 5).

Silica -> graphite (50 um radius pairs): w 3x -> 3^(5/6) = 2.498; E* (5.57/27.80)^(-1/3) = 1.709; rho
(2200/2000)^(-1/2) = 0.954; product 4.07 (the material values); f(0.70)/f(0.85) = 1.52 (the ASSUMED e_dry);
total 6.20 = 38.3/6.18 mm/s.

Published comparison.  Chokshi, Tielens & Hollenbach (1993, ApJ 407, 806) derived the critical velocity
v_cr = c gamma^(5/6) / (E^(1/3) R^(5/6) rho^(1/2)) with R the reduced radius, assuming the energy is lost mainly
to elastic surface waves.  Beckwith, Henning & Nakagawa (2000), "Dust properties and assembly of large particles
in protoplanetary disks", Protostars and Planets IV, eq. 8 and its footnote, correct c from the 3.86 printed by
Chokshi et al. to 1.07 (and note that Dominik & Tielens 1997 misprinted the E exponent).  With E = E* and
R = R* this is 0.653x this model's undamped 1.6394, because the loss model differs (here: the JKR hysteresis
|U0|); the definitions agree: the review's "about 5 cm/s" for micron-sized quartz comes out at 5.7 cm/s with
the 1.07 formula (R* = 0.5 um, rho = 2600 kg/m^3).  The 1.84 often attributed to Thornton & Ning (1998) equals
the sphere-on-wall coefficient above, but that formula was not confirmed from the paper.  Both are listed only for
comparison; the model's own number is the one that follows from its JKR curve.
"""
from __future__ import annotations

from dataclasses import replace
import math

from .collide import sticking_threshold
from .jkr import hysteresis_energy
from .pair import Pair, make_pair


def hysteresis_constant(mode: str = "displacement") -> float:
    """k_U = |U0| / (w^(5/3) R^(4/3) E^(-2/3)) (no unit), computed from the JKR curve at R = w = E = 1 (exact by
    the scaling argument of the module docstring).  7.0922 for the displacement break."""
    return hysteresis_energy(1.0, 1.0, 1.0, mode)


def undamped_sticking_speed(pair: Pair, mode: str = "displacement") -> float:
    """v_stick,0 = sqrt(2 |U0| / m*) [m/s]: the sticking speed without damping or yield
    (dolle_sim/contact.py JKRContact.v_stick_elastic)."""
    return math.sqrt(2 * hysteresis_energy(pair.R_star, pair.w, pair.E_star, mode) / pair.m_star)


def equal_spheres_prefactor() -> float:
    """c in v_stick,0 = c gamma^(5/6) E*^(-1/3) R*^(-5/6) rho^(-1/2) for two equal spheres:
    c = sqrt(3 k_U / (8 pi)) 2^(5/6) = 1.6394."""
    return math.sqrt(3 * hysteresis_constant() / (8 * math.pi)) * 2 ** (5 / 6)


def wall_prefactor() -> float:
    """c in v_stick,0 = c w^(5/6) E*^(-1/3) R^(-5/6) rho^(-1/2) for a sphere on a wall: c = sqrt(3 k_U/(2 pi))
    = 1.8402."""
    return math.sqrt(3 * hysteresis_constant() / (2 * math.pi))


def scaling_law_speed(gamma: float, E_star: float, R_star: float, rho: float) -> float:
    """Hand formula for two equal spheres: v = 1.6394 gamma^(5/6) E*^(-1/3) R*^(-5/6) rho^(-1/2) [m/s]
    (gamma J/m^2, E* Pa, R* m, rho kg/m^3)."""
    return equal_spheres_prefactor() * gamma ** (5 / 6) * E_star ** (-1 / 3) * R_star ** (-5 / 6) * rho ** -0.5


def chokshi_dominik_tielens_speed(gamma: float, E_star: float, R_star: float, rho: float) -> float:
    """Published formula v = 1.07 gamma^(5/6) / (E^(1/3) R^(5/6) rho^(1/2)) [m/s] (Chokshi et al. 1993, with the
    coefficient 1.07 of Beckwith, Henning & Nakagawa 2000, eq. 8 and footnote), evaluated with E = E*, R = R*.  For
    comparison only: 0.653x this model's undamped speed, because Chokshi et al. assume a different energy loss
    (elastic surface waves; module docstring)."""
    return 1.07 * gamma ** (5 / 6) / (E_star ** (1 / 3) * R_star ** (5 / 6) * rho ** 0.5)


def damping_factor(e_dry: float, n_sub: int = 2000, tol: float = 1e-6, pair: Pair | None = None) -> float:
    """f(e_dry) = (damped sticking speed) / (undamped sticking speed) (no unit), measured by bisection on a
    reference pair (default: two 50 um radius silica grains, yield off) with its e_dry replaced.  Depends on e_dry
    only (module docstring)."""
    ref = pair if pair is not None else make_pair("silica", 50e-6, "silica", 50e-6, plastic=False)
    ref = replace(ref, e_dry=e_dry)
    return sticking_threshold(ref, tol=tol, n_sub=n_sub) / undamped_sticking_speed(ref)

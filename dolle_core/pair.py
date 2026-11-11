"""Two touching bodies seen by the contact law: the "reduced" (effective) quantities.

When two spheres press on each other, the contact does not care about each body separately: the classic
result of Hertz (1882) is that the contact behaves like ONE sphere of reduced radius R* pressing on a rigid flat,
made of a material of reduced modulus E*.  Likewise Newton's second law for the two bodies, written for their
relative coordinate, involves only the reduced mass m*.  These are

    1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2       [Pa]   both bodies deform; the softer one more
    R*   = r1 r2 / (r1 + r2)                     [m]    a flat wall is a sphere of infinite radius: R* = r1
    m*   = m1 m2 / (m1 + m2)                     [kg]   a wall has infinite mass: m* = m1

Why m*: body 1 feels -F, body 2 feels +F (Newton's third law), so the relative acceleration is
d^2(x1 - x2)/dt^2 = -F/m1 - F/m2 = -F/m*.

Surface stickiness of a pair: the work of adhesion w [J/m^2] is the energy needed to pull apart 1 m^2 of
contact.  Two surfaces of the same material: w = 2 gamma (two new surfaces are made).  Different materials: the
simulator uses the geometric mean, w = 2 sqrt(gamma1 gamma2) (a Berthelot-type mixing rule, ASSUMED: common
practice, not measured for these pairs).  An optional factor f_adh multiplies w (roughness or coatings,
ASSUMED knob, default 1).  The pair's dry restitution is sqrt(e1 e2) (ASSUMED rule) and its yield pressure the
smaller of the two (rule ASSUMED: the simulator's choice, dolle_sim materials.pair_props; Thornton & Ning 1998
treat a contact with a single limiting pressure p_y and supply only the yield law itself).

Source of every formula: dolle_sim/materials.py: pair_props (and dolle_sim/dem.py: make_params, identical).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .materials import Material, get_material, sphere_mass


def reduced_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
    """E* = 1 / ((1 - nu1^2)/E1 + (1 - nu2^2)/E2)  [Pa].  Hertz (1882); dolle_sim materials.pair_props."""
    return 1.0 / ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2)


def reduced_radius(r1: float, r2: float | None = None) -> float:
    """R* = r1 r2/(r1 + r2) for two spheres; R* = r1 on a flat wall (r2 None)  [m].  pair_props."""
    return r1 if r2 is None else r1 * r2 / (r1 + r2)


def reduced_mass(m1: float, m2: float = math.inf) -> float:
    """m* = m1 m2/(m1 + m2) [kg], written as m1/(1 + m1/m2) so that a wall (m2 = inf) gives m1 and two equal
    masses give exactly m1/2."""
    return m1 / (1.0 + m1 / m2)


def work_of_adhesion(gamma1: float, gamma2: float, f_adh: float = 1.0) -> float:
    """w = 2 sqrt(gamma1 gamma2) f_adh  [J/m^2], floored at 1e-12 (keeps the JKR algebra finite in the
    no-adhesion limit).  Mixing rule ASSUMED; dolle_sim materials.pair_props."""
    return max(2.0 * math.sqrt(gamma1 * gamma2) * f_adh, 1e-12)


@dataclass(frozen=True)
class Pair:
    """Everything the contact law needs about two bodies.  Body 2 is a wall when r2 is None (then m2 = inf).
    Units: radii m, masses kg, E_star Pa, w J/m^2, p_y Pa."""
    name: str
    r1: float
    m1: float
    r2: float | None
    m2: float
    E_star: float
    R_star: float
    m_star: float
    w: float
    e_dry: float
    p_y: float


def make_pair(mat1, r1: float, mat2, r2: float | None = None, f_adh: float = 1.0,
              plastic: bool = True) -> Pair:
    """Build the Pair for sphere (mat1, radius r1) touching sphere (mat2, r2), or a flat wall of mat2 when
    r2 is None.  Materials may be names or Material objects.  plastic=False switches yield off (p_y = inf).
    Same numbers as dolle_sim materials.pair_props(mat1, r1, mat2, r2, adhesion_factor=f_adh)."""
    a: Material = get_material(mat1)
    b: Material = get_material(mat2)
    m1 = sphere_mass(a.rho, r1)
    m2 = math.inf if r2 is None else sphere_mass(b.rho, r2)
    p_y = min(a.yield_pressure, b.yield_pressure) if plastic else math.inf
    return Pair(name=f"{a.name}-{b.name}" + ("(wall)" if r2 is None else ""), r1=r1, m1=m1, r2=r2, m2=m2,
                E_star=reduced_modulus(a.E, a.nu, b.E, b.nu), R_star=reduced_radius(r1, r2),
                m_star=reduced_mass(m1, m2), w=work_of_adhesion(a.gamma, b.gamma, f_adh),
                e_dry=math.sqrt(a.e_dry * b.e_dry), p_y=p_y)

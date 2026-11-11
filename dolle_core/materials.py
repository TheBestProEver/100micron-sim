"""Material properties: the numbers every other module starts from.

A grain (or a wall) enters the contact physics through six numbers:

    rho    density                                         kg/m^3
    E      Young's modulus: how stiff the solid is         Pa = N/m^2
    nu     Poisson ratio: how much it bulges sideways      (no unit)
           when squeezed
    gamma  surface energy of ONE free surface: the energy  J/m^2
           needed to create 1 m^2 of new surface.  Two
           touching surfaces of the same material hold
           together with the work of adhesion w = 2 gamma
    e_dry  "dry" restitution: speed out / speed in of an   (no unit)
           impact with the stickiness switched off.  It
           sets the strength of the damping (damping.py)
    p_y    yield (limiting contact) pressure: above it     Pa
           the contact flattens permanently (plastic.py)

The values below are EXACTLY those of the full simulator (dolle_sim/materials.py: MATERIALS), field by field;
tests/test_dolle_core.py checks every one.  The `source` text says where each number comes from as the
simulator's own notes and PROVENANCE.md state it; anything not taken from a paper or data sheet is marked
ASSUMED.  Only the mechanical fields are kept here (the simulator's charging and light-pressure fields are not
used by this core).

Which carbon?  The carbon DOLL-E will fly is not decided.  "graphite" is the simulator's carbon (values of
Chokshi, Tielens & Hollenbach 1993, Table 3); "amorphous_carbon" is an ASSUMED placeholder.  See
examples/silica_vs_carbon.py for how the sticking speed moves with each property.

Friction coefficients (FRICTION) are not material properties but properties of a PAIR of surfaces; they are the
simulator's table dolle_sim/engine.py: MU_DEFAULT.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

G0 = 9.80665
"""Standard gravity in m/s^2 (exact by definition); used only to express accelerations in units of g."""


@dataclass(frozen=True)
class Material:
    """One solid.  Units: rho kg/m^3, E Pa, nu -, gamma J/m^2 (per single surface), e_dry -, yield_pressure Pa
    (math.inf = never yields).  `source` says where the numbers come from."""
    name: str
    rho: float
    E: float
    nu: float
    gamma: float
    e_dry: float
    yield_pressure: float
    source: str


def _m(name, rho, E, nu, gamma, e_dry, p_y, source):
    """Shorthand for one table row (same field order as dolle_sim/materials.py)."""
    return Material(name, rho, E, nu, gamma, e_dry, p_y, source)


MATERIALS: dict[str, Material] = {m.name: m for m in [
    # ------------------------------------------------------------------ dust
    _m("silica", 2000.0, 54e9, 0.17, 0.025, 0.85, 3e9,
       "Amorphous SiO2 spheres. gamma = 0.025 J/m^2, E = 54 GPa, nu = 0.17: the quartz values used by Chokshi, "
       "Tielens & Hollenbach (1993) ApJ 407, 806 and Dominik & Tielens (1997) ApJ 480, 647 (measured in air "
       "~0.014-0.025 J/m^2; Kimura et al. 2015, ApJ 812, 67, argue dry silica in vacuum has ~0.15-0.25 J/m^2; "
       "Kreuzig et al. 2024, A&A 688, found no such rise for hydrophobic spheres). rho, e_dry and p_y ASSUMED."),
    _m("graphite", 2200.0, 10e9, 0.32, 0.075, 0.7, 0.3e9,
       "The simulator's carbon. gamma = 0.075 J/m^2 (basal plane), E = 10 GPa, nu = 0.32, rho = 2200 kg/m^3: "
       "Chokshi, Tielens & Hollenbach (1993) Table 3. The cleavage energy of graphite (Wang et al. 2015, Nat. "
       "Commun. 6, 7853) implies ~0.19 J/m^2. e_dry and p_y ASSUMED."),
    _m("amorphous_carbon", 1800.0, 20e9, 0.25, 0.05, 0.7, 1.5e9,
       "Glassy/amorphous carbon spheres: ALL values ASSUMED (properties vary with preparation; measure them)."),
    _m("carbon_silica_50_50", 2100.0, 30e9, 0.20, 0.045, 0.75, 1.5e9,
       "Placeholder for a mixed carbon/silicate grain: ALL values ASSUMED (rough intermediate values, not a "
       "mixing rule)."),
    _m("forsterite", 3270.0, 200e9, 0.24, 0.025, 0.85, 4e9,
       "Mg2SiO4 olivine, crystalline silicate analogue: ALL values ASSUMED (PROVENANCE.md, other materials)."),
    _m("alumina", 3950.0, 370e9, 0.22, 0.05, 0.85, 6e9,
       "Al2O3: ALL values ASSUMED; gamma is poorly constrained for powders (0.03-0.1 J/m^2)."),
    _m("water_ice", 917.0, 7e9, 0.25, 0.19, 0.8, 0.1e9,
       "Reference only. gamma: Gundlach et al. (2011) Icarus 214, 717; E, nu: Chokshi et al. (1993); "
       "rho, e_dry, p_y ASSUMED."),
    # ------------------------------------------------------------------ walls
    _m("glass_borosilicate", 2230.0, 63e9, 0.20, 0.0203, 0.8, 5e9,
       "Borosilicate 3.3 tubing: E 63 GPa, nu 0.2, rho 2.23 g/cm^3 from the DURAN borosilicate 3.3 data sheet "
       "(BOROFLOAT 33: E 64 GPa). gamma is an EFFECTIVE value: with w = 2 sqrt(gamma_grain gamma_wall) it "
       "reproduces w = 0.045 J/m^2 measured for ~90 um glass beads on clean glass in room air (Sansao et al. "
       "2021, Powder Technol. 384, 267) for silica grains; it gives w = 0.078 J/m^2 for graphite (no "
       "measurement). e_dry and p_y ASSUMED."),
    _m("glass_soda_lime", 2530.0, 72e9, 0.23, 0.0203, 0.8, 5e9,
       "Soda-lime float glass: E, nu, rho from a compilation (lower confidence); gamma as borosilicate; "
       "e_dry, p_y ASSUMED."),
    _m("fused_silica_glass", 2200.0, 73e9, 0.16, 0.0203, 0.8, 5e9,
       "Fused silica: E 73 GPa, nu 0.16, rho 2.20 g/cm^3 from the HPFS 7980 data sheet; gamma as "
       "borosilicate; e_dry, p_y ASSUMED."),
    _m("glass_silanized", 2230.0, 63e9, 0.20, 0.0068, 0.8, 5e9,
       "Borosilicate with a hydrophobic silane. gamma: effective value for w = 0.026 J/m^2, the middle of "
       "0.022-0.030 J/m^2 measured for silanised ~90 um beads on silanised glass (Sansao et al. 2021; Zafar et "
       "al. 2014, Powder Technol. 264, 236); E, nu, rho as borosilicate; e_dry, p_y ASSUMED."),
    _m("glass_bk7", 2510.0, 82e9, 0.21, 0.03, 0.8, 5e9, "Window glass: ALL values ASSUMED."),
    _m("glass_fluorosilane", 2510.0, 82e9, 0.21, 0.012, 0.8, 5e9,
       "Glass with a fluorinated anti-stick monolayer: ALL values ASSUMED (a design assumption to be measured)."),
    _m("aluminum_6061", 2700.0, 69e9, 0.33, 0.05, 0.6, 0.45e9, "Bare aluminium wall: ALL values ASSUMED."),
    _m("sapphire", 3980.0, 345e9, 0.29, 0.04, 0.85, 8e9, "Hard window material: ALL values ASSUMED."),
    _m("ptfe", 2200.0, 0.5e9, 0.46, 0.019, 0.6, 0.02e9, "PTFE (Teflon): ALL values ASSUMED."),
]}


FRICTION: dict[tuple[str, str], float] = {
    ("silica", "wall"): 0.4, ("silica", "silica"): 0.6, ("graphite", "wall"): 0.5,
    ("graphite", "graphite"): 0.5, ("graphite", "silica"): 0.5}
"""Sliding friction coefficient mu (no unit) for (grain, other) pairs, "wall" meaning any wall.  Same table as
dolle_sim/engine.py: MU_DEFAULT.  Sources as engine.py gives them: silica on glass 0.4 (Meyer, Fuchs, Staedler &
Jiang 2014, Friction 2, 255); silica on silica 0.6 (Liu et al. 2017, Nat. Commun. 8, 14029, its uncoated-silica
reference; not re-checked here); graphite 0.5 (Buckley 1971, 1973: graphite in dry gas or vacuum).  Liu et al.'s
headline result is ~0.003 for graphene-coated contacts (superlubricity), so a carbon's mu is uncertain by more
than 100x depending on its surface.  graphite-wall and graphite-silica are ASSUMED; every other pair is 0.4,
ASSUMED."""


def get_material(name_or_material) -> Material:
    """Return the Material named `name_or_material` (a Material passes through unchanged)."""
    if isinstance(name_or_material, Material):
        return name_or_material
    if name_or_material not in MATERIALS:
        raise KeyError(f"unknown material {name_or_material!r}; known: {sorted(MATERIALS)}")
    return MATERIALS[name_or_material]


def friction_coefficient(grain: str, other: str) -> float:
    """Sliding friction mu between `grain` and `other` ("wall" for any wall), looked up in either order;
    0.4 (ASSUMED) when the pair is not in FRICTION (the simulator's default, dolle_sim/engine.py)."""
    return FRICTION.get((grain, other), FRICTION.get((other, grain), 0.4))


def sphere_mass(rho: float, r: float) -> float:
    """Mass of a solid sphere, m = (4/3) pi r^3 rho  [kg]  (rho in kg/m^3, r in m).
    Mirrors dolle_sim/materials.py: sphere_mass."""
    return 4.0 / 3.0 * math.pi * r ** 3 * rho

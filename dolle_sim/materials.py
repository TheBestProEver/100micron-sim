"""Material database for DOLL-E dust simulations.

All values are SI. Numbers are representative literature values meant as a
starting point -- surface energy in particular is uncertain by a factor of
several for real powders (roughness, adsorbed water, contamination), and it is
exactly the quantity a sticking-velocity experiment constrains. Override any
value from the run config ("materials": {...}). Values marked "uncertain" in
the notes should be measured on the actual flight powder if at all possible.

gamma          : surface energy per surface [J/m^2]  (work of adhesion w = 2*gamma)
E              : Young's modulus [Pa]
nu             : Poisson ratio
rho            : bulk (grain) density [kg/m^3]
e_dry          : restitution coefficient for non-adhesive (high-speed) impacts,
                 sets the viscoelastic damping in the contact model
work_function  : electron work function [eV]; used for the contact-charging
                 model (charge transfer ~ difference in work functions)
permittivity   : relative dielectric constant; sets how strongly a grain is
                 polarised by nearby charges (conductors: use a large number)
yield_pressure : limiting contact pressure for plastic yield [Pa]
                 (Thornton & Ning 1998); ~1.6x the uniaxial yield stress
thermal_conductivity : [W/m/K]; controls photophoresis (poor conductors are
                 pushed hardest by light)
absorptivity   : fraction of visible light absorbed (photophoresis, radiation
                 pressure); carbon ~0.9, clean silica ~0.05-0.1
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math


@dataclass(frozen=True)
class Material:
    name: str
    rho: float
    E: float
    nu: float
    gamma: float
    e_dry: float = 0.8
    work_function: float = 5.0
    permittivity: float = 4.0
    yield_pressure: float = math.inf
    thermal_conductivity: float = 1.0
    absorptivity: float = 0.5
    note: str = ""

    @property
    def polarizability(self) -> float:
        """Clausius-Mossotti factor (eps-1)/(eps+2) of a sphere (1 for a conductor)."""
        return (self.permittivity - 1) / (self.permittivity + 2)

    @property
    def image_factor(self) -> float:
        """Image-charge factor (eps-1)/(eps+1) of a flat wall (1 for a conductor)."""
        return (self.permittivity - 1) / (self.permittivity + 1)

    def to_dict(self):
        d = asdict(self)
        if not math.isfinite(d["yield_pressure"]):
            d["yield_pressure"] = None
        return d


MATERIALS: dict[str, Material] = {
    # ---------------------------------------------------------------- dust
    "silica": Material(
        "silica", rho=2000.0, E=54e9, nu=0.17, gamma=0.025, e_dry=0.85,
        work_function=5.0, permittivity=3.8, yield_pressure=3e9,
        thermal_conductivity=1.4, absorptivity=0.05,
        note="Amorphous SiO2 spheres; gamma=0.025 J/m^2 (Blum & Wurm 2000). "
             "The most-studied reference case. Yield pressure uncertain."),
    "forsterite": Material(
        "forsterite", rho=3270.0, E=200e9, nu=0.24, gamma=0.025, e_dry=0.85,
        work_function=5.0, permittivity=7.0, yield_pressure=4e9,
        thermal_conductivity=4.0, absorptivity=0.2,
        note="Mg2SiO4 olivine, crystalline silicate analogue."),
    "alumina": Material(
        "alumina", rho=3950.0, E=370e9, nu=0.22, gamma=0.05, e_dry=0.85,
        work_function=4.7, permittivity=9.8, yield_pressure=6e9,
        thermal_conductivity=30.0, absorptivity=0.1,
        note="Al2O3; gamma is poorly constrained for powders (0.03-0.1)."),
    "graphite": Material(
        "graphite", rho=2200.0, E=10e9, nu=0.20, gamma=0.075, e_dry=0.7,
        work_function=4.6, permittivity=1e6, yield_pressure=0.3e9,
        thermal_conductivity=100.0, absorptivity=0.9,
        note="Polycrystalline graphite; gamma=0.075 J/m^2 as in Chokshi+1993. "
             "Soft: yields plastically at modest speeds. Basal-plane gamma is "
             "higher (~0.1-0.2); yield pressure uncertain."),
    "amorphous_carbon": Material(
        "amorphous_carbon", rho=1800.0, E=20e9, nu=0.25, gamma=0.05, e_dry=0.7,
        work_function=4.8, permittivity=1000, yield_pressure=1.5e9,
        thermal_conductivity=1.5, absorptivity=0.95,
        note="Glassy/amorphous carbon spheres (closest lab analogue of "
             "interstellar carbon). Props vary with preparation -- measure them."),
    "carbon_silica_50_50": Material(
        "carbon_silica_50_50", rho=2100.0, E=30e9, nu=0.20, gamma=0.045,
        e_dry=0.75, work_function=4.8, permittivity=20, yield_pressure=1.5e9,
        thermal_conductivity=1.5, absorptivity=0.7,
        note="Placeholder for a mixed carbon/silicate grain (simple averages)."),
    "water_ice": Material(
        "water_ice", rho=1000.0, E=7e9, nu=0.25, gamma=0.19, e_dry=0.8,
        work_function=6.0, permittivity=3.2, yield_pressure=0.1e9,
        thermal_conductivity=2.2, absorptivity=0.02,
        note="Reference only (outer-disk ices)."),
    # ---------------------------------------------------------------- walls
    "aluminum_6061": Material(
        "aluminum_6061", rho=2700.0, E=69e9, nu=0.33, gamma=0.05, e_dry=0.6,
        work_function=4.3, permittivity=1e6, yield_pressure=0.45e9,
        thermal_conductivity=167.0, absorptivity=0.1,
        note="Chamber wall material (bare Al)."),
    "glass_bk7": Material(
        "glass_bk7", rho=2510.0, E=82e9, nu=0.21, gamma=0.03, e_dry=0.8,
        work_function=5.0, permittivity=6.7, yield_pressure=5e9,
        thermal_conductivity=1.1, absorptivity=0.01,
        note="Chamber window / wall."),
    "glass_fluorosilane": Material(
        "glass_fluorosilane", rho=2510.0, E=82e9, nu=0.21, gamma=0.012, e_dry=0.8,
        work_function=5.3, permittivity=6.7, yield_pressure=5e9,
        thermal_conductivity=1.1, absorptivity=0.01,
        note="Glass with a fluorinated anti-stick monolayer (FDTS-type): hard "
             "and low surface energy -> low wall adhesion."),
    "sapphire": Material(
        "sapphire", rho=3980.0, E=345e9, nu=0.29, gamma=0.04, e_dry=0.85,
        work_function=4.7, permittivity=9.4, yield_pressure=8e9,
        thermal_conductivity=35.0, absorptivity=0.01,
        note="Very hard window material."),
    "ptfe": Material(
        "ptfe", rho=2200.0, E=0.5e9, nu=0.46, gamma=0.019, e_dry=0.6,
        work_function=5.75, permittivity=2.1, yield_pressure=0.02e9,
        thermal_conductivity=0.25, absorptivity=0.1,
        note="Teflon. Low surface energy BUT very soft (large contact area) and "
             "strongly tribo-charging -- usually NOT a good anti-stick wall here."),
}


def get_material(name: str, overrides: dict | None = None) -> Material:
    """Return a material, applying config overrides (dict of field -> value)."""
    overrides = overrides or {}
    if name in MATERIALS:
        base = MATERIALS[name]
    else:
        if name not in overrides:
            raise KeyError(f"Unknown material '{name}'. Known: {sorted(MATERIALS)}")
        return Material(name=name, **{k: v for k, v in overrides[name].items() if k != "name"})
    if name in overrides:
        base = replace(base, **overrides[name])
    return base


@dataclass(frozen=True)
class PairProps:
    """Effective contact properties for a pair of materials/radii."""
    E_star: float    # effective modulus [Pa]
    R_star: float    # reduced radius [m] (R1 for sphere-on-flat)
    w: float         # work of adhesion [J/m^2]
    e_dry: float     # combined dry restitution
    F_c: float       # JKR pull-off force [N]
    a0: float        # equilibrium contact radius [m]
    delta_c: float   # (negative) overlap at which the contact breaks [m]
    E_break: float   # energy to break a contact (Chokshi+1993) [J]
    p_y: float       # limiting (yield) contact pressure of the softer body [Pa]


def pair_props(m1: Material, r1: float, m2: Material, r2: float | None,
               adhesion_factor: float = 1.0) -> PairProps:
    """Contact properties for sphere(m1, r1) on sphere(m2, r2); r2=None means a flat wall.

    adhesion_factor scales the work of adhesion (surface roughness / coatings
    reduce the true contact area; per-grain scatter).
    """
    E_star = 1.0 / ((1 - m1.nu ** 2) / m1.E + (1 - m2.nu ** 2) / m2.E)
    R_star = r1 if r2 is None else r1 * r2 / (r1 + r2)
    # Work of adhesion between dissimilar surfaces (Berthelot-type mixing rule)
    w = 2.0 * math.sqrt(m1.gamma * m2.gamma) * adhesion_factor
    w = max(w, 1e-12)            # non-adhesive limit (keeps the JKR algebra finite)
    e_dry = math.sqrt(m1.e_dry * m2.e_dry)
    F_c = 1.5 * math.pi * w * R_star
    a0 = (4.5 * math.pi * w * R_star ** 2 / E_star) ** (1.0 / 3.0)
    # contact radius at break under displacement control: a_c = a0 / 6^(2/3)
    a_c = a0 / 6.0 ** (2.0 / 3.0)
    delta_c = a_c ** 2 / R_star - math.sqrt(2 * math.pi * w * a_c / E_star)
    E_break = 1.54 * F_c * abs(delta_c)
    p_y = min(m1.yield_pressure, m2.yield_pressure)
    return PairProps(E_star, R_star, w, e_dry, F_c, a0, delta_c, E_break, p_y)


def sphere_mass(mat: Material, r: float) -> float:
    return 4.0 / 3.0 * math.pi * r ** 3 * mat.rho

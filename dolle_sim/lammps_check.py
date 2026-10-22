"""Optional cross-check of the DOLL-E contact model against LAMMPS (pip install lammps).

THIS AND rebound_check.py ARE THE ONLY FILES THAT USE EXTERNAL SIMULATION CODES. The simulator never imports
this file (tests/test_provenance.py checks it).

LAMMPS (Thompson et al. 2022, Comput. Phys. Commun. 271, 108171; https://www.lammps.org) is a widely used,
open-source particle simulator. Its GRANULAR package implements the JKR contact model (pair_style granular
... jkr, and fix wall/gran ... jkr) with the same force law, overlap-radius relation, work of adhesion
w = 2 gamma and effective-modulus mixing as DOLL-E (checked in the source, src/GRANULAR/
gran_sub_mod_normal.cpp). It is a fully independent implementation and integrator.

One documented difference: LAMMPS breaks a JKR contact at the maximum tensile force (F = -F_c,
a = a0/4^(1/3)), while DOLL-E, like Chokshi et al. (1993) and Dominik & Tielens (1997), follows the JKR
curve to the end of the stable branch (a = a0/6^(2/3)). For an undamped collision this halves LAMMPS's
adhesive energy loss. DOLL-E's contact model can use either criterion (JKRContact(separation=...)).

What is compared (python -m dolle_sim lammps-check):
  1. Undamped head-on collisions of two equal spheres (silica, graphite): the sticking threshold and the
     restitution e(v) from LAMMPS vs DOLL-E with the same break criterion (expected to agree to the
     integration accuracy), and vs DOLL-E's default criterion (expected ratio 0.70).
  2. Pull-off from a wall: a grain resting on a flat wall (fix wall/gran zplane, which keeps the JKR neck;
     fix wall/gran/region does not) under a slowly ramped body force must detach at F = 1.5 pi w R, the
     criterion DOLL-E uses for wall-bound grains. Detachment is checked every ~0.004% of F_c near F_c; the
     remaining ~0.1% offset is the lag of the finite loading rate.
Damped collisions are not compared: LAMMPS refuses its Tsuji damping with a cohesive normal model such as JKR
("Cannot use tsuji model with a cohesive normal model"), and the laws it does allow with JKR (velocity,
mass_velocity, viscoelastic) differ from DOLL-E's Tsuji-type damping evaluated with the JKR contact radius.
The collision test therefore runs with zero damping; the wall test uses LAMMPS's velocity damping only to let the
resting grain settle before the slow pull (it does not affect the quasi-static pull-off force).

Provenance: drives LAMMPS through its Python module; parameters mapped as in the LAMMPS source (wall:
E_wall = 2 (1 - nu^2) E*, gamma_wall = sqrt(gamma_grain gamma_wall)). No REBOUND code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .contact import JKRContact
from .materials import get_material, pair_props, sphere_mass


def _lmp():
    try:
        import lammps
    except ImportError as e:  # pragma: no cover
        raise ImportError("the LAMMPS cross-check needs LAMMPS: pip install lammps (Windows: also MS-MPI)") from e
    return lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])


def lammps_collision(mat, R, v_rel, n_sub=300):
    """Head-on collision of two equal undamped JKR spheres in LAMMPS. Returns e (0 = stuck together)."""
    E, nu, rho, gamma = mat.E, mat.nu, mat.rho, mat.gamma
    Estar = E / (2 * (1 - nu ** 2))
    Rs = R / 2
    meff = sphere_mass(mat, R) / 2
    t_hz = 2.87 * (meff ** 2 / (Rs * Estar ** 2 * v_rel)) ** 0.2
    a0 = (4.5 * math.pi * 2 * gamma * Rs ** 2 / Estar) ** (1 / 3)
    t_osc = 2 * math.pi * math.sqrt(meff / (2 * Estar * a0))
    dt = min(t_hz, t_osc) / n_sub
    gap = 0.2 * R
    tmax = 2 * gap / v_rel + 30 * max(t_hz, t_osc)
    L = _lmp()
    L.commands_string(f"""
units si
atom_style sphere
atom_modify map array
boundary f f f
newton off
comm_modify vel yes
region box block {-4 * R} {4 * R} {-2 * R} {2 * R} {-2 * R} {2 * R} units box
create_box 1 box
create_atoms 1 single {-(R + gap / 2):.15e} 0 0 units box
create_atoms 1 single {(R + gap / 2):.15e} 0 0 units box
set type 1 diameter {2 * R:.15e}
set type 1 density {rho}
pair_style granular
pair_coeff * * jkr {E} 0.0 {nu} {gamma} damping velocity tangential linear_nohistory 0.0 0.0
velocity all set 0 0 0 units box
set atom 1 vx {v_rel / 2:.15e}
set atom 2 vx {-v_rel / 2:.15e}
fix 1 all nve/sphere
timestep {dt:.8e}
neighbor {R} bin
neigh_modify every 1 delay 0 check yes
run {int(tmax / dt) + 1}
""")
    v = L.numpy.extract_atom("v")
    x = L.numpy.extract_atom("x")
    tag = L.numpy.extract_atom("id")
    vx = {int(t): float(v[i, 0]) for i, t in enumerate(tag)}
    xx = {int(t): float(x[i, 0]) for i, t in enumerate(tag)}
    L.close()
    d_pull = (3 * math.pi ** 2 / 16) ** (1 / 3) * (Rs * gamma ** 2 / Estar ** 2) ** (1 / 3)
    e = (vx[2] - vx[1]) / v_rel
    return e if (xx[2] - xx[1] - 2 * R > 2 * d_pull and e > 0) else 0.0


def lammps_threshold(mat, R, lo, hi, iters=12):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if lammps_collision(mat, R, mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def lammps_pulloff(grain, wall, R, ramp_g_per_s=None):
    """A grain resting on a JKR wall (fix wall/gran zplane) under a body force ramped away from the wall.
    Returns the body force [N] at which it detached."""
    E_star = 1.0 / ((1 - grain.nu ** 2) / grain.E + (1 - wall.nu ** 2) / wall.E)
    nu_w = grain.nu
    E_w = 2 * (1 - nu_w ** 2) * E_star          # LAMMPS treats the wall as the same material: E* = E/(2(1-nu^2))
    gamma_w = math.sqrt(grain.gamma * wall.gamma)
    m = sphere_mass(grain, R)
    Fc = 1.5 * math.pi * 2 * gamma_w * R
    a_c = Fc / m
    a0 = (4.5 * math.pi * 2 * gamma_w * R ** 2 / E_star) ** (1 / 3)
    t_osc = 2 * math.pi * math.sqrt(m / (2 * E_star * a0))
    dt = t_osc / 200
    ramp = ramp_g_per_s or (a_c / 9.80665) / (4000 * t_osc)     # reach the pull-off over ~4000 oscillations
    t_end = 1.5 * (a_c / 9.80665) / ramp
    nsteps = int(t_end / dt)
    L = _lmp()
    L.commands_string(f"""
units si
atom_style sphere
atom_modify map array
boundary f f f
newton off
comm_modify vel yes
region box block {-3 * R} {3 * R} {-3 * R} {3 * R} 0 {20 * R} units box
create_box 1 box
create_atoms 1 single 0 0 {R - a0 ** 2 / (3 * R):.15e} units box
set type 1 diameter {2 * R:.15e}
set type 1 density {grain.rho}
pair_style granular
pair_coeff * * jkr {grain.E} 0.0 {grain.nu} {grain.gamma} damping velocity tangential linear_nohistory 0.0 0.0
variable gmag equal {ramp * 9.80665:.10e}*time
fix wall all wall/gran granular jkr {E_w} {0.5 * m * 2 * math.pi / t_osc:.6e} {nu_w} {gamma_w} damping velocity tangential linear_nohistory 0.0 0.0 zplane 0.0 NULL
fix g all gravity v_gmag vector 0 0 1
fix 1 all nve/sphere
timestep {dt:.8e}
neighbor {R} bin
neigh_modify every 1 delay 0 check yes
variable zz equal z[1]
""")
    a0dist = (3 * math.pi ** 2 / 16) ** (1 / 3) * (R * gamma_w ** 2 / E_star ** 2) ** (1 / 3)
    # coarse checks up to 98% of the expected pull-off, then checks 100x finer (each 0.00375% of F_c)
    t_c = (a_c / 9.80665) / ramp
    coarse = max(int(0.98 * t_c / dt) // 100, 1)
    fine = max(nsteps // 40000, 1)
    F_det, steps = None, 0
    while steps < nsteps:
        chunk = coarse if steps + coarse < 0.98 * t_c / dt else fine
        L.command(f"run {chunk}")
        steps += chunk
        if L.extract_variable("zz") - R > 3 * a0dist:
            F_det = m * ramp * 9.80665 * L.get_thermo("time")
            break
    L.close()
    return F_det, Fc, fine * dt / t_c


def compare(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("silica", "graphite"):
        mat = get_material(name)
        import dataclasses
        el = dataclasses.replace(mat, e_dry=0.99999, yield_pressure=math.inf)
        R = 50e-6
        m_eff = sphere_mass(el, R) / 2
        p = pair_props(el, R, el, R)
        ours_force = JKRContact(p, separation="force").threshold(m_eff, tol=2e-5)
        ours_disp = JKRContact(p, separation="displacement").threshold(m_eff, tol=2e-5)
        lmp = lammps_threshold(el, R, 0.5 * ours_force, 1.5 * ours_force)
        e_rows = []
        for f in (1.2, 2.0, 4.0):
            v = f * ours_force
            e_rows.append(dict(v_m_s=v, lammps_e=lammps_collision(el, R, v),
                               dolle_e=JKRContact(p, separation="force").resolve(v, m_eff).e_n))
        rows.append(dict(material=name, radius_m=R, dolle_threshold_force_break_m_s=ours_force,
                         dolle_threshold_default_m_s=ours_disp, lammps_threshold_m_s=lmp,
                         rel_diff_same_criterion=lmp / ours_force - 1, restitution=e_rows))
    pull = []
    for g, w in (("silica", "glass_borosilicate"), ("graphite", "glass_borosilicate")):
        F_det, Fc, res_rel = lammps_pulloff(get_material(g), get_material(w), 50e-6)
        pull.append(dict(grain=g, wall=w, lammps_detach_force_N=F_det, jkr_pulloff_N=Fc,
                         rel_diff=(F_det / Fc - 1) if F_det else None, detection_step_rel=res_rel))
    res = {"collisions": rows, "pulloff": pull}
    (out / "lammps_check.json").write_text(json.dumps(res, indent=2))
    return res

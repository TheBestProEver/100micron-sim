"""How long would a time-driven DEM code (LAMMPS) take for one DOLL-E tubule? (pip install lammps)

    python tools/lammps_benchmark.py          # grains in flight (no contacts: a lower bound)
    python tools/lammps_benchmark.py --bed    # the settled single layer, every grain in a JKR wall contact

800 silica grains (R = 50 um, E = 54 GPa, nu = 0.17, rho = 2000 kg/m^3, gamma = 0.025 J/m^2) moving at ~1 cm/s in
a 12.7 x 12.7 x 38.1 mm box with JKR walls, real stiffness, time step 20% of the Rayleigh time (EDEM's guidance for
dense packings; Washino et al. 2024 recommend 0.1-0.2). The measured rate is extrapolated to the 332 s from lift-off
to power-off, and to softened grains. Softening the modulus by lambda raises the JKR sticking speed by lambda^(-1/3)
(v_stick ~ gamma^(5/6) E^(-1/3) rho^(-1/2) R^(-5/6): Chokshi et al. 1993; Thornton & Ning 1998). The usual correction
gamma_mod = gamma lambda^(2/5) (Haervig et al. 2017; Chen et al. 2019; Washino et al. 2024) restores the sticking
speed but cuts the pull-off force 1.5 pi w R by lambda^(2/5), the quantity that decides release from the walls.
"""
from __future__ import annotations

import math
import time

import lammps

R, RHO, E, NU, GAMMA = 50e-6, 2000.0, 54e9, 0.17, 0.025
LX, LY, LZ = 0.0127, 0.0127, 0.0381
N_STEPS = 20000


def main(bed: bool = False):
    G = E / (2 * (1 + NU))
    t_ray = math.pi * R * math.sqrt(RHO / G) / (0.1631 * NU + 0.8766)
    dt = 0.2 * t_ray
    m = 4 / 3 * math.pi * R ** 3 * RHO
    T = m * 0.01 ** 2 / (3 * 1.380649e-23)          # a "temperature" that gives ~1 cm/s rms speeds
    L = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    jkr = f"jkr {E} 0.5 {NU} {GAMMA} damping viscoelastic tangential linear_history 0.3 0.0 0.3"
    L.commands_string(f"""
units si
atom_style sphere
atom_modify map array
boundary f f f
newton off
comm_modify vel yes
region box block 0 {LX} 0 {LY} 0 {LZ} units box
create_box 1 box
region inner block {2 * R} {LX - 2 * R} {2 * R} {LY - 2 * R} {(R - 2e-9 if bed else 2 * R)} {(R - 0.5e-9 if bed else LZ - 2 * R)} units box
create_atoms 1 random 800 4928459 inner overlap {2.2 * R} maxtry 2000
set type 1 diameter {2 * R}
set type 1 density {RHO}
pair_style granular
pair_coeff * * {jkr}
fix wx all wall/gran granular {jkr} xplane 0.0 {LX}
fix wy all wall/gran granular {jkr} yplane 0.0 {LY}
fix wz all wall/gran granular {jkr} zplane 0.0 {LZ}
velocity all create {(1e-6 * T if bed else T):.6e} 87287 dist gaussian
fix 1 all nve/sphere
timestep {dt:.6e}
neighbor {0.5 * R} bin
neigh_modify every 1 delay 0 check yes
run 100
""")
    t0 = time.time()
    L.command(f"run {N_STEPS}")
    el = time.time() - t0
    n = L.get_natoms()
    if bed:                                       # check that every grain still touches the aft wall
        z = L.numpy.extract_atom("x")[:, 2]
        print(f"grains touching the aft wall at the end: {int((z <= R + 1e-9).sum())} of {n}")
    L.close()
    rate = n * N_STEPS / el
    steps = 332.0 / dt
    print(f"{'settled single layer on the aft wall (every grain in a JKR wall contact)' if bed else 'grains in flight (no contacts)'}")
    print(f"Rayleigh time {t_ray:.3e} s, dt {dt:.3e} s, {n} grains")
    print(f"{N_STEPS} steps in {el:.1f} s: {rate:.3e} grain-steps per second on one core")
    print(f"T+0 to T+332 s = {steps:.2e} steps: {n * steps / rate / 86400:.0f} core-days per tubule")
    for lam in (1e-2, 1e-3):
        print(f"  E x {lam:g}: {n * steps * math.sqrt(lam) / rate / 86400:.1f} core-days; sticking speed x "
              f"{lam ** (-1 / 3):.1f} unless gamma x {lam ** 0.4:.3f}, which cuts the wall pull-off force "
              f"{lam ** -0.4:.0f}x")


if __name__ == "__main__":
    import sys
    main(bed="--bed" in sys.argv)

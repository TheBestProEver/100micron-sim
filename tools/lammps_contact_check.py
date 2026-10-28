"""Term-by-term comparison of DOLL-E's full contact model (dolle_sim/dem.py) with LAMMPS (pip install lammps).

Both codes run the same physical set-ups with the same contact model (JKR with LAMMPS's break at maximum tension,
LAMMPS "viscoelastic" normal damping, Mindlin sliding, SDS rolling and twisting), the same coefficients and the same
time step, and the final states are compared:

  A  oblique, spinning grain-grain impact at 3 speeds (stick, slow bounce, fast bounce): velocities and spins
  B  a grain on a flat wall under a lateral body force ramped through the rolling limit: displacement and spin
  C  a grain on a wall given a lateral step load below the rolling limit: the rocking oscillation (elastic rolling)
  D  a grain on a wall spinning about the wall normal: the twisting limit
  E  20 grains dropped on a floor in a box (multi-contact, walls): positions and velocities after a short time

python tools/lammps_contact_check.py [out.json]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dolle_sim import dem                                     # noqa: E402
from dolle_sim.materials import get_material                  # noqa: E402

MAT = get_material("silica")
R = 50e-6
ETA0 = 2.0e5          # LAMMPS viscoelastic damping constant [1/(m s)]: F_d = -eta0 m_eff a v_n (moderate damping)
XGT = 1.0
MU_T = 0.6
H_ROLL = 0.5
Q = 30.0


def coeffs(wall):
    """Contact coefficients, computed by DOLL-E's make_params (so both codes use identical numbers)."""
    m = 4 / 3 * math.pi * R ** 3 * MAT.rho
    p = dem.make_params(R, 0.0 if wall else R, m, math.inf if wall else m, MAT.E, MAT.nu, MAT.E, MAT.nu,
                        2 * MAT.gamma, 1.0, math.inf, MU_T, H_ROLL, Q, Q, dem.DAMP_VISCOELASTIC, dem.BREAK_FORCE,
                        XGT, ETA0)
    return p, m


def lammps_model(wall):
    p, m = coeffs(wall)
    # LAMMPS forms E_eff = E/(2(1-nu^2)) for pairs and walls (wall treated as the same material) -> pass the grain E
    return (f"jkr {MAT.E:.10e} {ETA0:.10e} {MAT.nu} {MAT.gamma:.10e} damping viscoelastic "
            f"tangential linear_history {p[dem.P_KT] * p[dem.P_A0]:.10e} {XGT} {MU_T} "
            f"rolling sds {p[dem.P_KR]:.10e} {p[dem.P_GR]:.10e} {p[dem.P_MUR]:.10e} "
            f"twisting sds {p[dem.P_KTW]:.10e} {p[dem.P_GTW]:.10e} {p[dem.P_MUTW]:.10e}")


def run_lammps(x, v, w, nsteps, dt, box, gravity=None, wall=False, ramp=None):
    import lammps
    L = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    lo, hi = box
    cmds = f"""
units si
atom_style sphere
atom_modify map array
boundary f f f
newton off
comm_modify vel yes
region box block {lo[0]} {hi[0]} {lo[1]} {hi[1]} {lo[2]} {hi[2]} units box
create_box 1 box
"""
    for k in range(len(x)):
        cmds += f"create_atoms 1 single {x[k][0]:.17e} {x[k][1]:.17e} {x[k][2]:.17e} units box\n"
    cmds += f"""
set type 1 diameter {2 * R:.17e}
set type 1 density {MAT.rho}
pair_style granular
pair_coeff * * {lammps_model(False)}
"""
    for k in range(len(x)):
        cmds += (f"set atom {k + 1} vx {v[k][0]:.17e} vy {v[k][1]:.17e} vz {v[k][2]:.17e}\n"
                 f"set atom {k + 1} omega {w[k][0]:.17e} {w[k][1]:.17e} {w[k][2]:.17e}\n")
    if wall:
        cmds += f"fix wz all wall/gran granular {lammps_model(True)} zplane 0.0 NULL\n"
    if gravity is not None:
        g = np.asarray(gravity, float)
        cmds += f"fix ga all addforce v_fx v_fy v_fz\n" if False else ""
        m = 4 / 3 * math.pi * R ** 3 * MAT.rho
        if ramp is None:
            cmds += f"fix ga all addforce {m * g[0]:.17e} {m * g[1]:.17e} {m * g[2]:.17e}\n"
        else:
            cmds += (f"variable fx equal {m * g[0]:.17e}+{m * ramp:.17e}*time\n"
                     f"fix ga all addforce v_fx {m * g[1]:.17e} {m * g[2]:.17e}\n")
    cmds += f"""
fix 1 all nve/sphere
timestep {dt:.17e}
neighbor {0.2 * R} bin
neigh_modify every 1 delay 0 check yes
run {nsteps}
"""
    L.commands_string(cmds)
    tag = L.numpy.extract_atom("id").copy()
    order = np.argsort(tag)
    X = L.numpy.extract_atom("x")[order].copy()
    V = L.numpy.extract_atom("v")[order].copy()
    W = L.numpy.extract_atom("omega")[order].copy()
    L.close()
    return X, V, W


def run_dolle(x, v, w, nsteps, dt, box, gravity=None, wall=False, ramp=None):
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    S = dem.DEMSystem(np.asarray(x) - lo, v, w, R, MAT.rho, MAT.E, MAT.nu, MAT.gamma, 1.0, math.inf, hi - lo,
                      dict(E=MAT.E, nu=MAT.nu, gamma=MAT.gamma, e_dry=1.0, p_y=math.inf), mu_pp=MU_T, mu_pw=MU_T,
                      h_roll=H_ROLL, q_roll=Q, q_twist=Q, damp_mode=dem.DAMP_VISCOELASTIC,
                      break_mode=dem.BREAK_FORCE, x_gt=XGT, eta_visc=ETA0, walls=wall, wall_faces=(4,),
                      events=False)   # LAMMPS does not locate contact events
    acc = None
    if gravity is not None:
        g = np.asarray(gravity, float)
        if ramp is None:
            acc = g
        else:
            t = (np.arange(nsteps) + 1) * dt
            acc = np.column_stack([g[0] + ramp * t, np.full(nsteps, g[1]), np.full(nsteps, g[2])])
    S.run(dt, nsteps, acc=acc)
    return S.x + lo, S.v, S.w


def compare(name, a, b, scale):
    out = {}
    for lab, A, B, s in zip(("x", "v", "w"), a, b, scale):
        d = float(np.abs(np.asarray(A) - np.asarray(B)).max())
        out[f"max_abs_diff_{lab}"] = d
        out[f"rel_diff_{lab}"] = d / s if s else None
    print(f"{name}: " + ", ".join(f"{k}={v:.2e}" for k, v in out.items() if k.startswith("rel") and v is not None))
    return out


def main(out_path=None):
    p_pair, m = coeffs(False)
    p_wall, _ = coeffs(True)
    T_s = 2 * math.pi / math.sqrt(p_pair[dem.P_KT] * p_pair[dem.P_A0] * 3.5 / m)   # sliding-mode scale
    dt = T_s / 60
    res = {"dt_s": dt, "coefficients_pair": p_pair.tolist(), "coefficients_wall": p_wall.tolist(), "tests": {}}
    # A: oblique impacts with spin
    for v_imp in (0.003, 0.03, 0.3):
        b = 0.5 * R
        x = [[0.0, 0.0, 0.0], [2 * R + 0.05 * R, b, 0.0]]
        v = [[v_imp, 0.0, 0.0], [0.0, 0.0, 0.0]]
        w = [[0.0, 0.0, 200.0], [50.0, -80.0, 0.0]]
        # first touch after the x-gap closes to sqrt(4 - 0.25) R; then 40 us of contact or separation
        n = int(((2.05 - math.sqrt(4 - 0.25)) * R / v_imp + 40e-6) / dt)
        box = ([-10 * R, -10 * R, -10 * R], [12 * R, 10 * R, 10 * R])
        a = run_lammps(x, v, w, n, dt, box)
        d = run_dolle(x, v, w, n, dt, box)
        res["tests"][f"A_impact_{v_imp}"] = r = compare(f"A oblique impact {v_imp} m/s", a, d, (R, v_imp, 200.0))
        r["gap_after_nm"] = float((np.linalg.norm(d[0][1] - d[0][0]) - 2 * R) * 1e9)
        r["outcome"] = "stuck" if r["gap_after_nm"] < 1.0 else "bounced"
        print(f"   outcome {r['outcome']}, gap {r['gap_after_nm']:.2f} nm, |v2 - v1| {np.linalg.norm(d[1][1] - d[1][0]):.4g} m/s")
    # B: rolling under a ramped lateral load (0 -> 3x the rolling limit over 2 ms), 1 g pressing
    a_roll = p_wall[dem.P_MUR] * 2 * p_wall[dem.P_FC] / m
    z0 = R - p_wall[dem.P_A0] ** 2 / (3 * R)
    x = [[0.0, 0.0, z0]]
    nB = int(2e-3 / dt)
    box = ([-200 * R, -10 * R, 0.0], [200 * R, 10 * R, 10 * R])
    ramp = 3 * a_roll / 2e-3
    a = run_lammps(x, [[0, 0, 0]], [[0, 0, 0]], nB, dt, box, gravity=[0, 0, -9.80665], wall=True, ramp=ramp)
    d = run_dolle(x, [[0, 0, 0]], [[0, 0, 0]], nB, dt, box, gravity=[0, 0, -9.80665], wall=True, ramp=ramp)
    res["tests"]["B_rolling_ramp"] = compare("B rolling ramp", a, d, (max(abs(a[0][0][0]), R * 1e-3),
                                                                      max(abs(a[1][0][0]), 1e-9), 1.0))
    res["tests"]["B_rolling_ramp"]["lammps_x_m"] = float(a[0][0][0])
    res["tests"]["B_rolling_ramp"]["dolle_x_m"] = float(d[0][0][0])
    # C: rocking under a step load at half the rolling limit
    nC = int(0.6e-3 / dt)
    a = run_lammps(x, [[0, 0, 0]], [[0, 0, 0]], nC, dt, box, gravity=[0.5 * a_roll, 0, -9.80665], wall=True)
    d = run_dolle(x, [[0, 0, 0]], [[0, 0, 0]], nC, dt, box, gravity=[0.5 * a_roll, 0, -9.80665], wall=True)
    res["tests"]["C_rocking_step"] = compare("C rocking step", a, d, (R * 1e-6, 1e-6, max(abs(a[2][0][1]), 1e-6)))
    # D: twisting limit: spin about the wall normal
    nD = int(0.3e-3 / dt)
    a = run_lammps(x, [[0, 0, 0]], [[0, 0, 300.0]], nD, dt, box, gravity=[0, 0, -9.80665], wall=True)
    d = run_dolle(x, [[0, 0, 0]], [[0, 0, 300.0]], nD, dt, box, gravity=[0, 0, -9.80665], wall=True)
    res["tests"]["D_twisting"] = compare("D twisting", a, d, (R * 1e-6, 1e-6, 300.0))
    # E: 20 grains dropped onto a floor
    rng = np.random.default_rng(3)
    pts = []
    while len(pts) < 20:
        c = rng.uniform([2 * R, 2 * R, 1.2 * R], [12 * R, 12 * R, 8 * R])
        if all(np.linalg.norm(c - q) > 2.2 * R for q in pts):
            pts.append(c)
    vE = rng.normal(0, 0.01, (20, 3))
    nE = int(0.3e-3 / dt)
    box = ([0.0, 0.0, 0.0], [14 * R, 14 * R, 14 * R])
    a = run_lammps(pts, vE, np.zeros((20, 3)), nE, dt, box, gravity=[0, 0, -100 * 9.80665], wall=True)
    d = run_dolle(pts, vE, np.zeros((20, 3)), nE, dt, box, gravity=[0, 0, -100 * 9.80665], wall=True)
    res["tests"]["E_drop_20"] = compare("E 20 grains", a, d, (R, 0.01, 100.0))
    if out_path:
        Path(out_path).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

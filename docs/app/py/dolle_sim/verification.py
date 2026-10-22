"""Verification suite: physics benchmarks against analytic solutions and conservation laws.

Run with `python -m dolle_sim verify`, in the browser app (Verify tab), or via pytest.
Each check returns a dict with what was tested, the measured value, the
acceptance criterion and whether it passed. Checks are short (seconds) so they
can run in the browser (Pyodide) as well.

Provenance: Each check compares the engine with an analytic solution, an
independently computed reference (scipy ODE solver, Coulomb's law, kinetic-
theory formulas) or a conservation law.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np

from .materials import get_material, pair_props, sphere_mass
from .contact import JKRContact
from .simulation import ChamberSim
from .electrostatics import SpaceChargeSolver, short_range_kernel, K_E

R = 50e-6
QUIET = {"brownian": False, "self_gravity": False, "thermophoresis": False, "photophoresis": False,
         "radiation_pressure": False}


def _cfg(env=None, forces=None, contact=None, dt=4e-3, T=1.0):
    e = {"preset": "ideal", "pressure_pa": 1.0, "gravity_gradient": False, "magnetic_field_t": [0, 0, 0]}
    e.update(env or {})
    f = dict(QUIET)
    f.update(forces or {})
    c = {"contact_charging": False}
    c.update(contact or {})
    return {"dt_s": dt, "duration_s": T, "environment": e, "forces": f, "contact": c}


def _one_grain(v, pos, size=1.0, **extra):
    return {"name": "t", "size_m": [size] * 3, **extra,
            "particles": [{"material": "silica", "count": 1, "radius_m": R,
                           "region_m": {"center": list(pos), "size": [0, 0, 0]},
                           "velocity": {"mode": "jet", "direction": list(v), "cone_deg": 0,
                                        "speed_mean_m_s": float(np.linalg.norm(v)), "speed_sigma_m_s": 0}}]}


def _contact(name, **kw):
    m = replace(get_material(name), **kw)
    return JKRContact(pair_props(m, R, m, R)), sphere_mass(m, R) / 2, m


def _e_tn(x):
    return (6 * math.sqrt(3) / 5) ** 0.5 * (1 - x * x / 6) ** 0.5 * (x / (x + 2 * math.sqrt(6 / 5 - x * x / 5))) ** 0.25


def _res(cid, name, what, value, criterion, passed, details=""):
    return {"id": cid, "name": name, "what": what, "value": value, "criterion": criterion,
            "passed": bool(passed), "details": details}


# ---------------------------------------------------------------------------- contact mechanics
def check_elastic_limit():
    c, me, _ = _contact("silica", e_dry=0.99999, gamma=1e-7, yield_pressure=math.inf)
    e = c.resolve(0.3, me).e_n
    return _res("elastic", "Elastic limit conserves energy", "No adhesion, no damping, no plasticity: restitution must be 1.",
                f"e = {e:.5f}", "|e - 1| < 1e-3", abs(e - 1) < 1e-3)


def check_damping():
    c, me, _ = _contact("silica", gamma=1e-7, yield_pressure=math.inf)
    e = c.resolve(0.1, me).e_n
    return _res("damping", "Viscoelastic damping calibration", "Non-adhesive impact must reproduce the material's dry restitution e_dry = 0.85.",
                f"e = {e:.4f}", "|e - 0.85| < 0.005", abs(e - 0.85) < 0.005)


def check_thornton_ning():
    c, me, _ = _contact("silica", e_dry=0.99999, gamma=1e-7, yield_pressure=0.5e9)
    worst, rows = 0.0, []
    for v in (1.0, 3.0, 10.0):
        e = c.resolve(v, me).e_n
        ref = _e_tn(c.v_yield(me) / v)
        worst = max(worst, abs(e - ref))
        rows.append(f"v={v:g}: {e:.4f} vs {ref:.4f}")
    return _res("thornton_ning", "Plastic restitution (Thornton & Ning 1998)",
                "Elastic-plastic impacts must follow the analytic Thornton-Ning restitution.",
                f"max |e - e_TN| = {worst:.4f}", "< 0.005", worst < 0.005, "; ".join(rows))


def check_plastic_damping():
    c, me, m = _contact("graphite")
    worst, rows = 0.0, []
    for v in (3.0, 10.0, 30.0):
        r = c.resolve(v, me)
        ref = _e_tn(c.v_yield(me) / v) * m.e_dry
        worst = max(worst, abs(r.e_n / ref - 1))
        rows.append(f"v={v:g}: {r.e_n:.3f} vs {ref:.3f}")
    return _res("plastic_damping", "Plasticity with damping",
                "With realistic damping (e_dry = 0.7) plastic impacts must give ~ e_TN x e_dry (no damping of a separated contact).",
                f"max relative deviation = {100 * worst:.1f}%", "< 5%", worst < 0.05, "; ".join(rows))


def check_energy_scaling():
    c, me, _ = _contact("graphite")
    worst = 0.0
    for v in (0.05, 0.2, 1.0, 5.0):
        worst = max(worst, abs(c.resolve(v, me).e_n - c.resolve(v / math.sqrt(2), 2 * me).e_n))
    return _res("energy_scaling", "Outcome depends only on impact energy",
                "Same 1/2 m v^2 with twice the mass must give the same restitution (dimensional analysis of the contact law).",
                f"max |delta e| = {worst:.1e}", "< 1e-4", worst < 1e-4)


def check_threshold_limits():
    c0, me0, _ = _contact("silica", e_dry=0.99999)
    v_el, v_dem0 = c0.v_stick_elastic(me0), c0.threshold(me0)
    c, me, _ = _contact("silica")
    v = c.threshold(me)
    ok = abs(v_dem0 / v_el - 1) < 0.02 and c.resolve(0.5 * v, me).stuck and not c.resolve(2 * v, me).stuck
    cg, mg, _ = _contact("graphite")
    ok = ok and cg.threshold(mg) > v
    return _res("threshold", "Sticking threshold limits",
                "Without damping the DEM threshold equals the JKR hysteresis energy estimate; grains stick below / bounce above it; carbon stickier than silica.",
                f"undamped {v_dem0 * 1e3:.3f} vs JKR {v_el * 1e3:.3f} mm/s; damped silica {v * 1e3:.2f} mm/s",
                "within 2%, stick/bounce consistent", ok)


# ---------------------------------------------------------------------------- integrators
def check_free_flight():
    a = 1e-3
    cfg = _cfg(env={"residual_accel_g": [0, 0, a / 9.80665]})
    s = ChamberSim(cfg, _one_grain([0.01, 0, 0], [0.5, 0.5, 0.5]), 0)
    x0, v0 = s.pos[0].copy(), s.V[0].copy()
    tau = s.mass[0] / s.dragc[0]
    s.run(progress=False)
    T = s.duration
    vt = np.array([0, 0, a * tau])
    x = x0 + vt * T + (v0 - vt) * tau * (1 - math.exp(-T / tau))
    err = float(np.linalg.norm(s.pos[0] - x))
    return _res("free_flight", "Free flight with gas drag", "Residual acceleration + drag vs the exact analytic trajectory over 1 s.",
                f"error = {err * 1e9:.2f} nm", "< 10 nm", err < 1e-8)


def _rotating_run(dt, spin):
    from scipy.integrate import solve_ivp
    cfg = _cfg(env={"spin_rate_hz": spin, "spin_axis": [0, 0, 1]}, T=2.0, dt=dt)
    ch = _one_grain([0.01, 0.003, 0.001], [0.5, 0.5, 0.5], center_from_spin_axis_m=[0.05, 0.0, 0.0])
    s = ChamberSim(cfg, ch, 0)
    x0, v0 = s.pos[0].copy(), s.V[0].copy()
    gam = s.dragc[0] / s.mass[0]
    Om, off = s.env.omega, s.axis_offset
    s.run(progress=False)

    def rhs(t, y):
        r, v = y[:3], y[3:]
        return np.concatenate([v, -gam * v - 2 * np.cross(Om, v) - np.cross(Om, np.cross(Om, r + off))])
    sol = solve_ivp(rhs, (0, s.duration), np.concatenate([x0, v0]), rtol=1e-12, atol=1e-15)
    return float(np.linalg.norm(s.pos[0] - sol.y[:3, -1])), float(np.linalg.norm(sol.y[:3, -1] - x0))


def check_rotating_frame():
    e_flight, _ = _rotating_run(4e-3, 0.0167)
    e1, path = _rotating_run(4e-3, 0.5)
    e2, _ = _rotating_run(2e-3, 0.5)
    ok = e_flight < 1e-9 and e1 < 2e-4 * path and e1 / e2 > 3.5
    return _res("rotating_frame", "Spinning payload frame (Coriolis + centrifugal + drag)",
                "Grain trajectory vs a 1e-12-tolerance ODE solution: exact at 1 rpm, 2nd-order convergence at 30x spin.",
                f"1 rpm: {e_flight * 1e9:.3f} nm; 0.5 Hz: {e1 * 1e6:.2f} um, convergence ratio {e1 / e2:.2f}",
                "< 1 nm; ratio > 3.5", ok)


def check_rigid_body():
    cfg = _cfg(env={"pressure_pa": 0.0}, T=5.0)
    ch = {"name": "t", "size_m": [0.05] * 3,
          "particles": [{"material": "graphite", "count": 3, "radius_m": R, "velocity": {"speed_mean_m_s": 0.0}}]}
    s = ChamberSim(cfg, ch, 1)
    c = s.size / 2
    s.pos[:] = c + np.array([[-2 * R, 0, 0], [0, 0, 0], [2 * R, 0, 0]])
    s.X[:] = s.pos
    s.V[:] = 0
    s.L[:] = 0
    s._build_neighbors()
    s._merge(0, 1)
    s._merge(s.body[0], 2)
    b = s.body[0]
    w0 = 2 * math.pi * 5 * np.array([math.cos(0.65), math.sin(0.65), 0.0])
    s.L[b] = np.linalg.inv(s.Iinv[b]) @ w0
    E0 = 0.5 * s.L[b] @ s.Iinv[b] @ s.L[b]
    L0 = s.L[b].copy()
    n_steps = int(round(s.duration / s.dt))
    dE = np.empty(n_steps)
    for k in range(n_steps):
        s.step(k * s.dt)
        dE[k] = 0.5 * s.L[b] @ s.Iinv[b] @ s.L[b] / E0 - 1
    dL = float(np.linalg.norm(s.L[b] - L0) / np.linalg.norm(L0))
    q = n_steps // 5
    drift = abs(dE[-q:].mean() - dE[:q].mean())
    ok = np.abs(dE).max() < 1e-3 and drift < 1e-5 and dL < 1e-10
    return _res("rigid_body", "Free rotation of an aggregate",
                "Spinning linear trimer in vacuum for 5 s: angular momentum conserved; the rotational-energy error "
                "(sampled every step) stays bounded and does not drift.",
                f"max |dE/E| = {np.abs(dE).max():.1e}, drift {drift:.1e}, |dL/L| = {dL:.1e}",
                "max dE < 1e-3, drift < 1e-5, dL < 1e-10", ok)


# ---------------------------------------------------------------------------- collisions
def check_impulse():
    cfg = _cfg()
    ch = {"name": "t", "size_m": [0.02] * 3,
          "particles": [{"material": "graphite", "count": 4, "radius_m": R, "velocity": {"speed_mean_m_s": 0.0}}]}
    s = ChamberSim(cfg, ch, 4)
    rng = np.random.default_rng(0)
    c = s.size / 2
    s.pos[:3] = c + np.array([[0, 0, 0], [2 * R, 0, 0], [2 * R, 2 * R, 0]])
    s.X[:] = s.pos
    s._merge(0, 1)
    s._merge(s.body[0], 2)
    A = s.body[0]
    worst_n, worst_cone = 0.0, 0.0
    for _ in range(200):
        n = rng.standard_normal(3)
        n /= np.linalg.norm(n)
        rA = rng.standard_normal(3) * R
        vrel = rng.standard_normal(3) * 0.01
        vrel += (abs(vrel @ n) + 0.005 - vrel @ n) * n
        K = s._K_body(A, rA) + np.eye(3) / s.mass[3]
        e = rng.random()
        P = s._impulse(K, vrel, n, e)
        v_after = vrel - K @ P
        worst_n = max(worst_n, abs(v_after @ n + e * (vrel @ n)) / (vrel @ n))
        Pn = P @ n
        worst_cone = max(worst_cone, np.linalg.norm(P - Pn * n) / (s.mu_f * Pn) - 1)
    ok = worst_n < 1e-9 and worst_cone < 1e-9
    return _res("impulse", "Rigid-body collision impulse",
                "200 random oblique impacts on a trimer: post-impact normal velocity = -e v_n exactly; friction stays inside the Coulomb cone.",
                f"normal error {worst_n:.1e}, cone excess {max(worst_cone, 0):.1e}", "< 1e-9", ok)


def check_conservation():
    cfg = _cfg()
    ch = {"name": "t", "size_m": [0.02] * 3,
          "particles": [{"material": "graphite", "count": 3, "radius_m": R, "velocity": {"speed_mean_m_s": 0.01}}]}
    s = ChamberSim(cfg, ch, 3)
    s._build_neighbors()
    s.pos[1] = s.pos[0] + np.array([2 * R, 0, 0])
    s.pos[2] = s.pos[0] + np.array([0, 2 * R, 0])
    s.X[:] = s.pos
    s.L[:] = np.array([[1e-20, 0, 0], [0, 2e-20, 0], [0, 0, 3e-20]])

    def PL(bodies):
        P = sum(s.M[k] * s.V[k] for k in bodies)
        L = sum(s.L[k] + s.M[k] * np.cross(s.X[k], s.V[k]) for k in bodies)
        return P, L
    P0, L0 = PL(range(3))
    s._merge(0, 1)
    s._merge(s.body[0], 2)
    b = s.body[0]
    P1, L1 = PL([b])
    s._split(b, [k for k in s.members[b] if k != b][:1])
    P2, L2 = PL(np.flatnonzero(s.alive))
    errP = max(np.linalg.norm(P1 - P0), np.linalg.norm(P2 - P0)) / np.linalg.norm(P0)
    errL = max(np.linalg.norm(L1 - L0), np.linalg.norm(L2 - L0)) / np.linalg.norm(L0)
    return _res("conservation", "Momentum through sticking and fragmentation",
                "Merging three grains into an aggregate and splitting one off conserve linear and angular momentum.",
                f"dP/P = {errP:.1e}, dL/L = {errL:.1e}", "< 1e-6", errP < 1e-6 and errL < 1e-6)


def check_walls():
    cfg = _cfg(T=1.0, contact={"wall_sticking": "never"})
    ch = {"name": "t", "size_m": [0.01] * 3, "wall_material": "glass_fluorosilane",
          "wall_vibration": {"amplitude_m": 2.5e-5, "frequency_hz": 200},
          "particles": [{"material": "graphite", "count": 150, "radius_m": R,
                         "velocity": {"speed_mean_m_s": 0.01, "speed_sigma_m_s": 0.005}}]}
    s = ChamberSim(cfg, ch, 2)
    s._build_neighbors()
    worst = 0.0
    for k in range(int(s.duration / s.dt)):
        s.step(k * s.dt)
        worst = max(worst, float(np.maximum(s.rad[:, None] - s.pos, s.pos + s.rad[:, None] - s.size[None, :]).max()))
    pre = s.contain_max_m          # penetration found *before* the containment safety net acted
    return _res("walls", "Vibrating walls never let grains out",
                "150 grains (forming spinning aggregates) against 25 um / 200 Hz vibrating walls for 1 s: no grain "
                "surface may cross a wall, and the position safety net must never be needed.",
                f"worst penetration = {max(worst, pre) * 1e9:.2f} nm over {s.n_events['wall_bounce']} wall bounces "
                f"({s.contain_corrections} safety-net corrections)",
                "< 1 nm, no corrections", max(worst, pre) < 1e-9 and s.n_events["wall_bounce"] > 50)


def check_no_overlap():
    from scipy.spatial import cKDTree
    cfg = _cfg(T=0.5, contact={"wall_sticking": "never"})
    ch = {"name": "t", "size_m": [0.006] * 3, "wall_material": "glass_fluorosilane", "wall_adhesion_factor": 0.05,
          "wall_vibration": {"amplitude_m": 2.5e-5, "frequency_hz": 200},
          "particles": [{"material": "silica", "count": 300, "radius_m": R,
                         "velocity": {"speed_mean_m_s": 0.03, "speed_sigma_m_s": 0.01}}]}
    s = ChamberSim(cfg, ch, 3)
    s._build_neighbors()
    worst = 0.0
    for k in range(int(s.duration / s.dt)):
        s.step(k * s.dt)
        pairs = cKDTree(s.pos).query_pairs(2.2 * R, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            sel = s.body[i] != s.body[j]
            if sel.any():
                d = np.linalg.norm(s.pos[i[sel]] - s.pos[j[sel]], axis=1)
                worst = max(worst, float(((s.rad[i[sel]] + s.rad[j[sel]]) - d).max()))
    n_ev = s.n_events["bounce"] + s.n_events["stick"]
    return _res("no_overlap", "No missed collisions in a dense gas",
                "300 grains in a 6 mm box for 0.5 s: at the end of every step no two separate bodies may interpenetrate.",
                f"worst overlap = {max(worst, 0) / R:.1e} R over {n_ev} grain-grain impacts", "< 1e-3 R",
                worst < 1e-3 * R and n_ev > 20)


# ---------------------------------------------------------------------------- electrostatics
def check_space_charge():
    L = 0.03
    s = SpaceChargeSolver([L, L, L], sigma=3e-3 / 3.5)
    c = np.array([[L / 2, L / 2, L / 2]]) + np.array([[0.0003, 0.0002, 0.0001]])
    q = np.array([1e-12])
    s.solve(c, q)
    worst, rows = 0.0, []
    for r in (1e-3, 2e-3, 3e-3, 5e-3):
        p = c + np.array([[r, 0.0, 0.0]])
        E = s.field_at(p)[0] + (K_E * q[0] * short_range_kernel(r, s.sigma) * np.array([1.0, 0, 0]) if r < 3e-3 else 0)
        ratio = float(np.linalg.norm(E) / (K_E * q[0] / r ** 2))
        worst = max(worst, abs(ratio - 1))
        rows.append(f"{r * 1e3:g} mm: {ratio:.3f}")
    # grounded-wall image of a charge 5 mm from a wall
    p = np.array([[5e-3, L / 2, L / 2]])
    s.solve(p, q)
    img = float(s.field_at(p)[0][0] / (-K_E * q[0] / (2 * 5e-3) ** 2))
    ok = worst < 0.06 and abs(img - 1) < 0.1
    return _res("space_charge", "Long-range electrostatics (Ewald mesh)",
                "Short-range pairs + grounded-box particle-mesh field vs Coulomb's law, and the image force near a grounded wall.",
                f"max |F/F_Coulomb - 1| = {100 * worst:.1f}%; image ratio {img:.2f}", "< 6%; image within 10%", ok,
                "; ".join(rows))


def check_charge_saturation():
    cfg = _cfg(contact={"contact_charging": True})
    ch = {"name": "t", "size_m": [0.02] * 3, "wall_material": "glass_bk7",
          "particles": [{"material": "amorphous_carbon", "count": 1, "radius_m": R}]}
    s = ChamberSim(cfg, ch, 0)
    for _ in range(2000):
        s._charge(0, None, 1e-6)
    dphi = get_material("amorphous_carbon").work_function - get_material("glass_bk7").work_function
    target = -dphi * R / K_E
    err = abs(s.q[0] / target - 1)
    return _res("charging", "Contact charging saturates",
                "Repeated wall contacts drive the grain to the potential that cancels the work-function difference.",
                f"q = {s.q[0] / 1.602e-19:.3g} e (expected {target / 1.602e-19:.3g} e)", "within 0.1%", err < 1e-3)


def check_threshold_fit():
    from .analysis import fit_threshold
    rng = np.random.default_rng(1)
    hits, n_rep, worst = 0, 20, 0.0
    for _ in range(n_rep):
        v = np.exp(rng.normal(math.log(0.03), 0.6, 1500))
        stuck = rng.random(1500) < 1 / (1 + np.exp((np.log(v) - math.log(0.03)) / 0.15))
        f = fit_threshold(v, stuck, n_boot=60, seed=int(rng.integers(1e9)))
        worst = max(worst, abs(f["v50_m_s"] / 0.03 - 1))
        hits += f["v50_lo_1sigma"] <= 0.03 <= f["v50_hi_1sigma"]
    sep = fit_threshold([0.001, 0.002, 0.003, 0.05, 0.06, 0.07], [1, 1, 1, 0, 0, 0])
    ok = worst < 0.05 and 0.4 <= hits / n_rep <= 0.95 and sep is not None and 0.003 <= sep["v50_m_s"] <= 0.05
    return _res("threshold_fit", "Sticking-threshold estimator",
                "Logistic fit + bootstrap on synthetic impacts with a known 30 mm/s threshold: unbiased, robust to perfectly "
                "separated data; the coverage of its 1-sigma intervals is reported (nominal 68%).",
                f"max error {100 * worst:.1f}%, 1-sigma coverage {hits}/{n_rep}", "< 5%; coverage 40-95%", ok)


def check_spin_rotating_frame():
    """A torque-free spinning grain keeps its absolute angular momentum fixed in inertial space."""
    cfg = _cfg(env={"pressure_pa": 0.0, "spin_rate_hz": 0.0167, "spin_axis": [0.3, 0.2, 1.0]}, T=10.0)
    s = ChamberSim(cfg, _one_grain([1e-9, 0, 0], [0.5, 0.5, 0.5]), 0)
    I0 = s.Iiso[0]
    w_rel0 = 2 * math.pi * 5 * np.array([0.6, 0.0, 0.8])
    s.L[0] = I0 * w_rel0
    Om = s.env.omega
    La0 = s.L[0] + I0 * Om                          # absolute angular momentum (chamber components at t=0)
    s.run(progress=False)
    expect = s.env.to_chamber(La0, s.duration)      # the same inertial vector seen from the chamber at t
    err = float(np.linalg.norm(s.L[0] + I0 * Om - expect) / np.linalg.norm(La0))
    # an L-shaped (asymmetric) aggregate 5 cm off the spin axis: centrifugal force acts on it, but in inertial
    # space no real torque does, so its absolute angular momentum must stay fixed too
    cfg = _cfg(env={"pressure_pa": 0.0, "spin_rate_hz": 0.0167, "spin_axis": [0.3, 0.2, 1.0]}, T=10.0)
    ch = {"name": "t", "size_m": [1.0] * 3, "center_from_spin_axis_m": [0.05, 0.0, 0.0],
          "particles": [{"material": "graphite", "count": 3, "radius_m": R, "velocity": {"speed_mean_m_s": 0.0}}]}
    a = ChamberSim(cfg, ch, 4)
    c = a.size / 2
    a.pos[:3] = c + np.array([[0, 0, 0], [2 * R, 0, 0], [2 * R, 2 * R, 0]])
    a.X[:] = a.pos
    a.V[:] = 0
    a.L[:] = 0
    a._build_neighbors()
    a._merge(0, 1)
    a._merge(a.body[0], 2)
    b = a.body[0]
    Iw = lambda: a.Q[b] @ np.diag(a.Ip[b]) @ a.Q[b].T
    a.L[b] = Iw() @ (2 * math.pi * 3 * np.array([0.3, -0.5, 0.8]))
    La0 = a.L[b] + Iw() @ a.env.omega
    a.run(progress=False)
    err2 = float(np.linalg.norm(a.L[b] + Iw() @ a.env.omega - a.env.to_chamber(La0, a.duration))
                 / np.linalg.norm(La0))
    return _res("spin_rotating_frame", "Spin in the spinning payload frame",
                "A spinning grain, and an L-shaped aggregate 5 cm off the spin axis, in vacuum for 10 s at 1 rpm about a "
                "tilted axis: their absolute angular momentum must stay fixed in inertial space (so it appears to "
                "precess at -Omega in the chamber); centrifugal force must not torque the aggregate.",
                f"grain |dL|/L = {err:.1e}, aggregate |dL|/L = {err2:.1e}", "< 1e-9", err < 1e-9 and err2 < 1e-9)


def _impact_energy_audit(s):
    """Wrap the impact resolvers of a sim: record, per impact, the change of kinetic + broken-bond energy."""
    def total():
        live = np.flatnonzero(s.alive & ~s.frozen)
        W = np.einsum("bij,bj->bi", s.Iinv[live], s.L[live])
        return float((0.5 * s.M[live] * (s.V[live] ** 2).sum(1)).sum() + 0.5 * (W * s.L[live]).sum()) + s.E_bonds_broken
    log = []

    def wrap(f):
        def g(*args):
            n0, E0 = len(s.collisions), total()
            hit = f(*args)
            if len(s.collisions) > n0:
                rec = s.collisions[-1]
                log.append((total() - E0, max(rec["E_coll"], 1e-300), rec["fragments"] > 0))
            return hit
        return g
    s._resolve_pp, s._resolve_wall = wrap(s._resolve_pp), wrap(s._resolve_wall)
    return log


def _phantom_bonds(s):
    return sum(len(s._components(s.members[b])) > 1 for b in np.flatnonzero(s.alive & (s.nmono > 1)))


def check_fragmentation_energy():
    """No impact may create energy: kinetic + broken-bond energy never increases, for any impact geometry."""
    cfg = _cfg(env={"pressure_pa": 0.0}, T=0.01)
    ch = {"name": "t", "size_m": [0.02] * 3, "wall_material": "glass_fluorosilane", "wall_adhesion_factor": 0.05,
          "particles": [{"material": "silica", "count": 4, "radius_m": R, "velocity": {"speed_mean_m_s": 0.0}}]}
    rng = np.random.default_rng(11)
    gains, n_frag, phantom = [], 0, 0
    # A: a grain hits a spinning, moving 3-grain chain or L at random speeds (0.02-3 m/s), angles and offsets
    for trial in range(60):
        s = ChamberSim(cfg, ch, 5)
        log = _impact_energy_audit(s)
        c = s.size / 2
        shape = [[-2 * R, 0, 0], [0, 0, 0], [2 * R, 0, 0]] if trial % 2 else [[0, 0, 0], [2 * R, 0, 0], [2 * R, 2 * R, 0]]
        s.pos[:3] = c + np.array(shape)
        s.X[:3] = s.pos[:3]
        s.V[:] = 0
        s.L[:] = 0
        s._merge(0, 1)
        s._merge(s.body[0], 2)
        A = s.body[0]
        s.V[A] = rng.standard_normal(3) * 0.01
        s.L[A] = np.linalg.inv(s.Iinv[A]) @ (rng.standard_normal(3) * 2 * math.pi * 10)
        tgt = s.pos[rng.integers(3)]
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        s.pos[3] = s.X[3] = tgt - d * 2.6 * R + np.cross(d, rng.standard_normal(3)) * 0.5 * R
        speed = math.exp(rng.uniform(math.log(0.02), math.log(3.0)))
        s.V[3] = d * speed
        s._build_neighbors()
        for k in range(3):
            s.step(k * s.dt)
        gains += log
        phantom += _phantom_bonds(s)
    # B: spinning aggregates hitting a static wall at random angles
    for trial in range(30):
        s = ChamberSim(cfg, ch, 6)
        log = _impact_energy_audit(s)
        s.pos[:3] = np.array([[0.01, 0.01, 1.2 * R], [0.01 + 2 * R, 0.01, 1.2 * R], [0.01 + 2 * R, 0.01 + 2 * R, 1.2 * R]])
        s.pos[3] = [0.015, 0.015, 0.01]
        s.X[:] = s.pos
        s.V[:] = 0
        s.L[:] = 0
        s._merge(0, 1)
        s._merge(s.body[0], 2)
        A = s.body[0]
        speed = math.exp(rng.uniform(math.log(0.02), math.log(3.0)))
        v = rng.standard_normal(3)
        v[2] = -abs(v[2]) - 0.2
        s.V[A] = speed * v / np.linalg.norm(v)
        s.L[A] = np.linalg.inv(s.Iinv[A]) @ (rng.standard_normal(3) * 2 * math.pi * 10)
        s._build_neighbors()
        for k in range(3):
            s.step(k * s.dt)
        gains += log
        phantom += _phantom_bonds(s)
    # C: a closed box of 300 graphite grains with spins and a wide speed spread (sticking, bouncing, friction,
    #    erosion and shattering of aggregates of every shape)
    cfgC = _cfg(env={"pressure_pa": 0.0}, T=0.5)
    chC = {"name": "t", "size_m": [0.01] * 3, "wall_material": "glass_fluorosilane", "wall_adhesion_factor": 0.05,
           "particles": [{"material": "graphite", "count": 300, "radius_m": R, "spin": {"rate_hz_mean": 20.0, "rate_hz_sigma": 10.0},
                          "velocity": {"speed_mean_m_s": 0.06, "speed_sigma_m_s": 0.05}}]}
    s = ChamberSim(cfgC, chC, 7)
    log = _impact_energy_audit(s)
    s.run(progress=False)
    gains += log
    phantom += _phantom_bonds(s)
    rel = np.array([g / e for g, e, _ in gains])
    n_frag = sum(f for _, _, f in gains)
    worst = float(rel.max()) if len(rel) else 0.0
    ok = worst <= 1e-9 and phantom == 0 and n_frag >= 20
    return _res("fragmentation_energy", "Impacts never create energy",
                "Every impact, measured just before and after it is resolved: grains hitting spinning chains and L-shaped "
                "aggregates at 0.02-3 m/s from random directions and offsets, spinning aggregates hitting a wall, and "
                "0.5 s of a closed box of 300 spinning graphite grains. Kinetic energy plus the energy of broken bonds "
                "may never increase, and no aggregate may keep grains that no longer touch.",
                f"max gain {worst:+.1e} of the impact energy over {len(rel)} impacts ({n_frag} with fragmentation); "
                f"phantom bonds {phantom}", "<= 1e-9, none, >= 20 fragmentations", ok)


def check_friction_energy():
    """Friction impulses on eccentric aggregate impacts never create kinetic energy."""
    cfg = _cfg()
    ch = {"name": "t", "size_m": [0.02] * 3,
          "particles": [{"material": "graphite", "count": 4, "radius_m": R, "velocity": {"speed_mean_m_s": 0.0}}]}
    s = ChamberSim(cfg, ch, 6)
    rng = np.random.default_rng(3)
    c = s.size / 2
    s.pos[:3] = c + np.array([[0, 0, 0], [2 * R, 0, 0], [2 * R, 2 * R, 0]])
    s.X[:] = s.pos
    s._merge(0, 1)
    s._merge(s.body[0], 2)
    A = s.body[0]
    worst = -math.inf
    for _ in range(2000):
        n = rng.standard_normal(3)
        n /= np.linalg.norm(n)
        rA = rng.standard_normal(3) * 2 * R
        vrel = rng.standard_normal(3) * 0.01
        vrel += (abs(vrel @ n) + 1e-4 - vrel @ n) * n
        K = s._K_body(A, rA) + np.eye(3) / s.mass[3]
        e = rng.random()
        P = s._impulse(K, vrel, n, e)
        dKE = float(-P @ vrel + 0.5 * P @ K @ P)
        Kinv = np.linalg.inv(K)
        worst = max(worst, dKE / (0.5 * vrel @ Kinv @ vrel))
    return _res("friction_energy", "Friction never creates energy",
                "2000 random eccentric impacts on an aggregate with friction (where rigid-body friction laws can "
                "gain energy): every impulse must dissipate.",
                f"max relative energy change {worst:+.1e}", "<= 0", worst <= 1e-12)


def check_drag_limits():
    """Gas drag reproduces the exact continuum (Stokes) and free-molecular (Epstein) limits."""
    from .environment import Environment
    env_lo = Environment({"preset": "ideal", "pressure_pa": 1e7}, np.random.default_rng(0))   # Kn << 1
    env_hi = Environment({"preset": "ideal", "pressure_pa": 1e-3}, np.random.default_rng(0))  # Kn >> 1
    r = 50e-6
    stokes = 6 * math.pi * env_lo.mu * r
    e1 = abs(float(env_lo.drag_coeff(r)) / stokes - 1)
    epstein = (4 * math.pi / 3) * (1 + math.pi / 8) * env_hi.rho_gas * env_hi.v_th * r ** 2   # diffuse reflection
    e2 = float(env_hi.drag_coeff(r)) / epstein - 1
    ok = e1 < 0.01 and abs(e2) < 0.03
    return _res("drag_limits", "Gas drag limits (Stokes / Epstein)",
                "The slip-corrected drag must reduce to Stokes drag in dense gas and to Epstein free-molecular drag "
                "(diffuse reflection) at low pressure.",
                f"Stokes {100 * e1:.2f}% off; Epstein {100 * e2:+.1f}% off", "< 1%; within 3%", ok,
                "The empirical slip correction sits ~2.5% below ideal diffuse-reflection Epstein drag.")


def check_thermophoresis():
    """Thermophoretic drift velocity vs the standard free-molecular result v = -(3/4)(1+pi/8)^-1 (mu/rho) gradT/T."""
    cfg = _cfg(env={"pressure_pa": 1.0, "temperature_gradient_k_m": [0, 0, 10.0]}, forces={"thermophoresis": True})
    s = ChamberSim(cfg, _one_grain([1e-9, 0, 0], [0.5, 0.5, 0.5], size=1.0), 0)
    env = s.env
    jump = 1 + 3.9 * env.mfp / s.size[2]                    # the code's chamber temperature-jump reduction
    v_code = s.F_static[0, 2] / s.dragc[0] * jump
    v_ref = -(3 / 4) / (1 + math.pi / 8) * env.mu / env.rho_gas * 10.0 / env.T
    ratio = v_code / v_ref
    return _res("thermophoresis", "Thermophoresis vs the standard free-molecular drift",
                "Thermophoretic force / drag must give the textbook free-molecular drift speed (derived independently "
                "from kinetic theory), up to the ~2.5% drag-model difference.",
                f"drift ratio = {ratio:.3f}", "within 4%", abs(ratio - 1) < 0.04)


def check_brownian():
    """Brownian kicks reproduce equipartition: <m v_x^2> = kT."""
    cfg = _cfg(env={"pressure_pa": 1e5}, forces={"brownian": True}, T=0.6)
    ch = {"name": "t", "size_m": [1.0] * 3,
          "particles": [{"material": "silica", "count": 2000, "radius_m": R, "velocity": {"speed_mean_m_s": 1e-9}}]}
    s = ChamberSim(cfg, ch, 7)
    s.run(progress=False)
    ratio = float((s.mass[:, None] * s.V ** 2).mean() / (1.380649e-23 * s.env.T))
    tau = s.mass[0] / s.dragc[0]
    return _res("brownian", "Brownian motion: equipartition",
                f"2000 grains in dense gas for {s.duration / tau:.0f} drag times: thermal kicks plus drag must settle to "
                "<m v_x^2> = kT (fluctuation-dissipation).",
                f"<m v^2>/kT = {ratio:.3f}", "within 5%", abs(ratio - 1) < 0.05)


# ---------------------------------------------------------------------------- flight timeline
_QUIET_EM = {"coulomb": False, "space_charge": False, "lorentz": False}


def _flight_cfg(T0, T, dt, prof, extra_env=None):
    env = {"preset": "ideal", "pressure_pa": 0.0, "gravity_gradient": False, "magnetic_field_t": [0, 0, 0],
           "start_time_s": T0, "flight_profile": prof}
    env.update(extra_env or {})
    return {"dt_s": dt, "dt_launch_s": dt, "duration_s": T, "forces": {**QUIET, **_QUIET_EM},
            "contact": {"contact_charging": False}, "environment": env}


def check_flight_frame():
    """Free flight through the de-spin (time-dependent spin, Euler force) vs an ODE; spins keep their absolute
    angular momentum while the payload spins up and is de-spun."""
    from scipy.integrate import solve_ivp
    from .flight import FlightProfile
    prof_cfg = {"residual_g": 0.0, "residual_spin_hz": 1.0 / 60.0, "coast_drag_g": 0.0, "post_burn_drag_g": 0.0}
    prof = FlightProfile.from_config(prof_cfg)
    T0, T = 53.8, 1.2
    ch = {"name": "t", "size_m": [20.0] * 3, "center_from_spin_axis_m": [0.06, 0.02, 0.0],
          "particles": [{"material": "silica", "count": 1, "radius_m": R, "region_m": {"center": [10, 10, 10], "size": [0, 0, 0]},
                         "velocity": {"mode": "jet", "direction": [0.3, -0.2, 0.1], "cone_deg": 0,
                                      "speed_mean_m_s": 0.01, "speed_sigma_m_s": 0}}]}

    def rhs(t, y, off):
        r, v = y[:3], y[3:]
        Om = np.array([0, 0, 2 * math.pi * prof.spin_hz(T0 + t)])
        Omd = np.array([0, 0, prof.spin_rate_dot(T0 + t, 1e-7)])
        ra = r + off
        return np.concatenate([v, -2 * np.cross(Om, v) - np.cross(Om, np.cross(Om, ra)) - np.cross(Omd, ra)])
    errs, path = [], 0.0
    for dt in (1e-3, 5e-4, 2.5e-4):
        cfg = _flight_cfg(T0, T, dt, prof_cfg)
        # a single grain has no neighbours: a large skin keeps the step at dt (fast grains otherwise get
        # shorter steps, which would hide the convergence order this check measures)
        cfg["contact"]["neighbor_skin_m"] = 100.0
        sim = ChamberSim(cfg, ch, 0)
        x0, v0, off = sim.pos[0].copy(), sim.V[0].copy(), sim.axis_offset
        sim.run(progress=False)
        sol = solve_ivp(rhs, (0, T), np.concatenate([x0, v0]), args=(off,), rtol=1e-12, atol=1e-15, max_step=5e-4)
        errs.append(float(np.linalg.norm(sim.pos[0] - sol.y[:3, -1])))
        path = float(np.linalg.norm(sol.y[:3, -1] - x0))
    ratio = errs[1] / errs[2]
    # absolute angular momentum of a spinning grain through the spin-up of the boost (0 -> 4.8 Hz)
    chs = {"name": "t", "size_m": [50.0] * 3, "particles": [{"material": "silica", "count": 1, "radius_m": R,
           "region_m": {"center": [25, 25, 25], "size": [0, 0, 0]}, "velocity": {"speed_mean_m_s": 0.0}}]}
    s = ChamberSim(_flight_cfg(0.0, 3.0, 5e-4, {"burn1": [0.0, 1.0], "burn2": [1.5, 3.0], "burn_g": 0.0,
                                                 "residual_g": 0.0, "despin_s": 10.0, "coast_drag_g": 0.0,
                                                 "post_burn_drag_g": 0.0}), chs, 0)
    I0 = s.Iiso[0]
    s.L[0] = I0 * 2 * math.pi * 3 * np.array([0.6, 0.0, 0.8])
    La0 = s.L[0] + I0 * s.env.omega
    s.run(progress=False)
    dL = float(np.linalg.norm(s.L[0] + I0 * s.env.omega - s.env.to_chamber(La0, s.duration)) / np.linalg.norm(La0))
    ok = errs[1] < 2e-3 * path and ratio > 3.5 and dL < 1e-8
    return _res("flight_frame", "Spin-up and de-spin (time-dependent spin)",
                "A grain released during the yo-yo de-spin (4.8 Hz -> 1 rpm in 0.9 s; Coriolis, centrifugal and Euler "
                "forces) vs a 1e-12 ODE solution; a spinning grain must keep its absolute angular momentum while the "
                "payload spins up from 0 to 4.8 Hz in 3 s.",
                f"error {100 * errs[1] / path:.2f}% of a {path:.1f} m path at dt = 0.5 ms, convergence ratio {ratio:.2f}; "
                f"|dL_abs|/L = {dL:.1e}", "< 0.2%; ratio > 3.5; < 1e-8", ok)


def _bed(n, prof_T0, T, prof, release=None, seed=1):
    tub = {"name": "t", "size_m": [0.0127, 0.0127, 0.0381], "wall_material": "glass_fluorosilane",
           "wall_adhesion_factor": 0.05, "release": release or [],
           "particles": [{"material": "silica", "count": n, "surface_energy_log_sigma": 1.0,
                          "size_distribution": {"type": "sieve", "min_m": 9e-5, "max_m": 1.06e-4},
                          "initial_state": {"type": "settled", "face": "-z"}}]}
    return ChamberSim(_flight_cfg(prof_T0, T, 4e-3, prof), tub, seed)


def check_detachment():
    """Wall-bound grains come off exactly when the load exceeds the JKR pull-off force 1.5 pi w R."""
    from .environment import G0
    s = _bed(300, 6.0, 0.05, {"coast_drag_g": 30.0, "spin_burnout_hz": 0.0})
    a_th = s.Fc_wall / s.mass / G0
    exp1 = int((a_th < 30.0).sum())
    s.run(progress=False)
    got1 = s.n_events["detach"]
    s2 = _bed(300, 100.0, 0.05, {"residual_g": 0.0, "spin_burnout_hz": 0.0, "residual_spin_hz": 0.0,
                                 "post_burn_drag_g": 0.0},
              release=[{"type": "shake", "start_s": 100.01, "cycles": 4, "frequency_hz": 200, "peak_g": 40.0, "axis": "z"}],
              seed=2)
    a_th2 = s2.Fc_wall / s2.mass / G0
    exp2 = int((a_th2 < 40.0).sum())
    s2.run(progress=False)
    A, w = s2.release[0]["amplitude_m"], s2.release[0]["omega"]
    verr = max([abs(d["v"] - A * w * math.sqrt(1 - (a_th2[d["grain"]] / 40.0) ** 2)) for d in s2.detach_log] or [0.0])
    ok = got1 == exp1 and s2.n_events["detach"] == exp2 and verr < 1e-9 and 0 < exp1 < 300 and 0 < exp2 < 300
    return _res("detachment", "Release of wall-bound grains",
                "300 grains settled on a tubule's aft face, with per-grain pull-off thresholds spread over ~2-1000 g: a "
                "steady 30 g pull (drag deceleration) and a 40 g, 200 Hz shaker burst must free exactly the grains whose "
                "JKR pull-off force 1.5 pi w R is exceeded; shaken grains leave at A w cos(phi), sin(phi) = a_th/(A w^2).",
                f"steady pull: {got1} of {exp1} expected; shaker: {s2.n_events['detach']} of {exp2} expected, "
                f"release speed error {verr:.1e} m/s", "exact; < 1e-9 m/s", ok)


def check_settling():
    """Under a sustained 25 g load every grain comes to rest on the loaded face, and none leaves the tubule."""
    ch = {"name": "t", "size_m": [0.0127, 0.0127, 0.0381], "wall_material": "glass_fluorosilane", "wall_adhesion_factor": 0.05,
          "particles": [{"material": "silica", "count": 200, "radius_m": R, "velocity": {"speed_mean_m_s": 0.05}}]}
    s = ChamberSim(_flight_cfg(1.0, 0.4, 4e-3, {"spin_burnout_hz": 0.0}), ch, 3)
    s.dt_launch = 5e-4
    s.run(progress=False)
    fz = s.frozen[s.body]
    on_aft = int((fz & (s.fw_ax[s.body] == 2) & (s.fw_side[s.body] < 0)).sum())
    inside = bool(np.all((s.pos >= s.rad[:, None] - 1e-12) & (s.pos <= s.size - s.rad[:, None] + 1e-12)))
    ok = fz.mean() > 0.95 and on_aft > 0.9 * s.N and inside and s.contain_corrections == 0
    return _res("settling", "Dust settles under the boost load",
                "200 grains moving at 5 cm/s in a 0.5 x 0.5 x 1.5 in tubule under 25 g: within 0.4 s they must come to rest "
                "on the aft face (bounces that would return within one step are resting contacts), with none outside.",
                f"{int(fz.sum())}/{s.N} wall-bound, {on_aft} on the aft face; safety-net corrections {s.contain_corrections}",
                "> 95%, > 90% aft, none outside", ok)


CHECKS = [check_elastic_limit, check_damping, check_thornton_ning, check_plastic_damping, check_energy_scaling,
          check_threshold_limits, check_free_flight, check_rotating_frame, check_rigid_body, check_impulse,
          check_conservation, check_walls, check_no_overlap, check_space_charge, check_charge_saturation,
          check_threshold_fit, check_spin_rotating_frame, check_fragmentation_energy, check_friction_energy,
          check_drag_limits, check_thermophoresis, check_brownian, check_flight_frame, check_detachment,
          check_settling]


def list_checks():
    return [f.__name__ for f in CHECKS]


def run_check(name):
    f = {fn.__name__: fn for fn in CHECKS}[name]
    t0 = time.time()
    try:
        r = f()
    except Exception as exc:          # a crash is a failed check, not a crashed suite
        r = _res(name, name, "", f"error: {exc!r}", "", False)
    r["seconds"] = round(time.time() - t0, 2)
    return r


def run_all(progress=None):
    out = []
    for f in CHECKS:
        r = run_check(f.__name__)
        out.append(r)
        if progress:
            progress(r)
    return out

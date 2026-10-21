"""Verification suite: physics benchmarks against analytic solutions and conservation laws.

Run with `python -m dolle_sim verify`, in the browser app (Verify tab), or via pytest.
Each check returns a dict with what was tested, the measured value, the
acceptance criterion and whether it passed. Checks are short (seconds) so they
can run in the browser (Pyodide) as well.
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
    s.run(progress=False)
    dE = abs(0.5 * s.L[b] @ s.Iinv[b] @ s.L[b] / E0 - 1)
    dL = float(np.linalg.norm(s.L[b] - L0) / np.linalg.norm(L0))
    return _res("rigid_body", "Free rotation of an aggregate",
                "Spinning linear trimer in vacuum for 5 s: angular momentum conserved, rotational energy bounded (no pumping).",
                f"|dE/E| = {dE:.1e}, |dL/L| = {dL:.1e}", "dE < 1e-3, dL < 1e-10", dE < 1e-3 and dL < 1e-10)


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
    return _res("walls", "Vibrating walls never let grains out",
                "150 grains against 25 um / 200 Hz vibrating walls for 1 s: no grain surface may ever cross a wall.",
                f"worst penetration = {max(worst, 0) * 1e9:.2f} nm over {s.n_events['wall_bounce']} wall bounces",
                "< 1 nm", worst < 1e-9 and s.n_events["wall_bounce"] > 50)


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
        v = np.exp(rng.normal(math.log(0.03), 0.6, 600))
        stuck = rng.random(600) < 1 / (1 + np.exp((np.log(v) - math.log(0.03)) / 0.15))
        f = fit_threshold(v, stuck, n_boot=60, seed=int(rng.integers(1e9)))
        worst = max(worst, abs(f["v50_m_s"] / 0.03 - 1))
        hits += f["v50_lo_1sigma"] <= 0.03 <= f["v50_hi_1sigma"]
    sep = fit_threshold([0.001, 0.002, 0.003, 0.05, 0.06, 0.07], [1, 1, 1, 0, 0, 0])
    ok = worst < 0.05 and 0.4 <= hits / n_rep <= 0.95 and sep is not None and 0.003 <= sep["v50_m_s"] <= 0.05
    return _res("threshold_fit", "Sticking-threshold estimator",
                "Logistic fit + bootstrap on synthetic impacts with a known 30 mm/s threshold: unbiased, honest 1-sigma "
                "intervals, robust to perfectly separated data.",
                f"max error {100 * worst:.1f}%, 1-sigma coverage {hits}/{n_rep}", "< 5%; coverage 40-95%", ok)


CHECKS = [check_elastic_limit, check_damping, check_thornton_ning, check_plastic_damping, check_energy_scaling,
          check_threshold_limits, check_free_flight, check_rotating_frame, check_rigid_body, check_impulse,
          check_conservation, check_walls, check_no_overlap, check_space_charge, check_charge_saturation,
          check_threshold_fit]


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

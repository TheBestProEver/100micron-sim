"""Engine validation tests (integrators, collision handling, walls, electrostatics)."""
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dolle_sim.materials import get_material, pair_props, sphere_mass
from dolle_sim.contact import JKRContact
from dolle_sim.simulation import ChamberSim
from dolle_sim.electrostatics import SpaceChargeSolver, short_range_kernel, K_E

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


def _rotating_run(dt, spin):
    cfg = _cfg(env={"spin_rate_hz": spin, "spin_axis": [0, 0, 1]}, T=2.0, dt=dt)
    ch = _one_grain([0.01, 0.003, 0.001], [0.5, 0.5, 0.5], center_from_spin_axis_m=[0.05, 0.0, 0.0])
    s = ChamberSim(cfg, ch, 0)
    x0, v0 = s.pos[0].copy(), s.V[0].copy()
    gam = s.dragc[0] / s.mass[0]
    Om, off = s.env.omega, s.axis_offset
    s.run(progress=False)

    def rhs(t, y):
        r, v = y[:3], y[3:]
        a = -gam * v - 2 * np.cross(Om, v) - np.cross(Om, np.cross(Om, r + off))
        return np.concatenate([v, a])
    sol = solve_ivp(rhs, (0, s.duration), np.concatenate([x0, v0]), rtol=1e-12, atol=1e-15)
    return np.linalg.norm(s.pos[0] - sol.y[:3, -1]), np.linalg.norm(sol.y[:3, -1] - x0)


def test_rotating_frame_free_flight_matches_ode():
    """Drag + Coriolis + centrifugal in a spinning frame vs a high-accuracy ODE solution."""
    # the flight case (1 rpm): essentially exact
    err, path = _rotating_run(4e-3, 0.0167)
    assert err < 1e-9
    # a 30x faster spin is a stringent check: small error that converges at 2nd order
    e1, path = _rotating_run(4e-3, 0.5)
    e2, _ = _rotating_run(2e-3, 0.5)
    assert e1 < 2e-4 * path
    assert e1 / e2 > 3.5


def test_free_rigid_body_conserves_energy_and_momentum():
    """A spinning linear trimer in vacuum: DLM rotation must not pump rotational energy."""
    cfg = _cfg(env={"pressure_pa": 0.0}, T=10.0)
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
    w0 = 2 * math.pi * 5 * np.array([math.cos(0.65), math.sin(0.65), 0.0])     # 37 deg to the axis
    s.L[b] = np.linalg.inv(s.Iinv[b]) @ w0
    E0 = 0.5 * s.L[b] @ s.Iinv[b] @ s.L[b]
    L0 = s.L[b].copy()
    s.run(progress=False)
    E1 = 0.5 * s.L[b] @ s.Iinv[b] @ s.L[b]
    assert abs(E1 / E0 - 1) < 1e-3          # symplectic: bounded O(h^2) error, no drift (old code: x7)
    assert np.allclose(s.L[b], L0, rtol=1e-10, atol=1e-10 * np.linalg.norm(L0))


def test_vibrating_walls_never_let_grains_out():
    cfg = _cfg(T=2.0, contact={"wall_sticking": "never"})
    ch = {"name": "t", "size_m": [0.01] * 3, "wall_material": "glass_fluorosilane",
          "wall_vibration": {"amplitude_m": 2.5e-5, "frequency_hz": 200},
          "particles": [{"material": "graphite", "count": 200, "radius_m": R,
                         "velocity": {"speed_mean_m_s": 0.01, "speed_sigma_m_s": 0.005}}]}
    s = ChamberSim(cfg, ch, 2)
    s._build_neighbors()
    worst = 0.0
    for k in range(int(s.duration / s.dt)):
        s.step(k * s.dt)
        out = np.maximum(s.rad[:, None] - s.pos, s.pos + s.rad[:, None] - s.size[None, :]).max()
        worst = max(worst, out)
    assert worst < 1e-9
    assert s.n_events["wall_bounce"] > 100


def test_no_interpenetration_in_dense_gas():
    """Event re-prediction: no pair of different bodies may overlap at the end of any step."""
    cfg = _cfg(T=1.0, contact={"wall_sticking": "never"})
    ch = {"name": "t", "size_m": [0.006] * 3, "wall_material": "glass_fluorosilane",
          "wall_adhesion_factor": 0.05, "wall_vibration": {"amplitude_m": 2.5e-5, "frequency_hz": 200},
          "particles": [{"material": "silica", "count": 400, "radius_m": R,
                         "velocity": {"speed_mean_m_s": 0.03, "speed_sigma_m_s": 0.01}}]}
    s = ChamberSim(cfg, ch, 3)
    s._build_neighbors()
    worst = 0.0
    from scipy.spatial import cKDTree
    for k in range(int(s.duration / s.dt)):
        s.step(k * s.dt)
        pairs = cKDTree(s.pos).query_pairs(2.2 * R, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            sel = s.body[i] != s.body[j]
            d = np.linalg.norm(s.pos[i[sel]] - s.pos[j[sel]], axis=1)
            if len(d):
                worst = max(worst, float(((s.rad[i[sel]] + s.rad[j[sel]]) - d).max()))
    assert s.n_events["bounce"] + s.n_events["stick"] > 50
    assert worst < 1e-3 * R


def test_impulse_gives_exact_normal_restitution_for_aggregates():
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
    for _ in range(200):
        n = rng.standard_normal(3); n /= np.linalg.norm(n)
        rA = rng.standard_normal(3) * R
        vrel = rng.standard_normal(3) * 0.01
        vrel += (abs(vrel @ n) + 0.005 - vrel @ n) * n          # approaching
        K = s._K_body(A, rA) + np.eye(3) / s.mass[3]
        e = rng.random()
        P = s._impulse(K, vrel, n, e)
        v_after = vrel - K @ P
        assert v_after @ n == pytest.approx(-e * (vrel @ n), rel=1e-9, abs=1e-15)
        Pn = P @ n
        assert np.linalg.norm(P - Pn * n) <= s.mu_f * Pn * (1 + 1e-9) + 1e-30


def test_plastic_restitution_with_damping():
    m = get_material("graphite")
    c = JKRContact(pair_props(m, R, m, R))
    me = sphere_mass(m, R) / 2
    for v in (3.0, 10.0, 30.0):
        x = c.v_yield(me) / v
        e_tn = (6 * math.sqrt(3) / 5) ** 0.5 * (1 - x * x / 6) ** 0.5 * (x / (x + 2 * math.sqrt(6 / 5 - x * x / 5))) ** 0.25
        res = c.resolve(v, me)
        assert not res.stuck and res.yielded
        assert res.e_n == pytest.approx(e_tn * m.e_dry, rel=0.05)


def test_zero_adhesion_contact_does_not_crash():
    m = replace(get_material("silica"), gamma=0.0)
    c = JKRContact(pair_props(m, R, m, R))
    assert c.resolve(0.1, sphere_mass(m, R) / 2).e_n == pytest.approx(0.85, abs=0.01)


def test_space_charge_solver_matches_coulomb():
    L = 0.05
    s = SpaceChargeSolver([L, L, L], sigma=3e-3 / 3.5)
    c = np.array([[L / 2, L / 2, L / 2]])
    q = np.array([1e-12])
    s.solve(c, q)
    for r in (1e-3, 2e-3, 3e-3, 5e-3, 8e-3):
        p = c + np.array([[r, 0.0, 0.0]])
        E = s.field_at(p)[0] + (K_E * q[0] * short_range_kernel(r, s.sigma) * np.array([1.0, 0, 0]) if r < 3e-3 else 0)
        assert np.linalg.norm(E) / (K_E * q[0] / r ** 2) == pytest.approx(1.0, abs=0.06)


def test_photophoresis_prefactor():
    cfg = _cfg(env={"illumination": {"intensity_w_m2": 1000.0, "direction": [1, 0, 0]}},
               forces={"photophoresis": True, "radiation_pressure": False})
    s = ChamberSim(cfg, _one_grain([1, 0, 0], [0.5, 0.5, 0.5]), 0)
    m = s.mats[0]
    expect = math.pi / 3 * 0.5 * m.absorptivity * s.env.P * 1000.0 * s.rad[0] ** 3 / (s.env.T * m.thermal_conductivity)
    assert s.F_static[0, 0] == pytest.approx(expect, rel=1e-9)


def test_impacts_on_wall_bound_grains_are_flagged():
    cfg = _cfg(T=0.6, contact={"wall_sticking": "always"})
    ch = {"name": "t", "size_m": [0.01] * 3,
          "particles": [{"material": "graphite", "count": 2, "radius_m": R,
                         "region_m": {"center": [0.0003, 0.005, 0.005], "size": [0, 0, 0]},
                         "velocity": {"mode": "jet", "direction": [-1, 0, 0], "cone_deg": 0,
                                      "speed_mean_m_s": 0.002, "speed_sigma_m_s": 0}}]}
    s = ChamberSim(cfg, ch, 0)
    s.pos[1] = s.pos[0] + np.array([0.0008, 0, 0])      # second grain follows the first to the wall
    s.X[:] = s.pos
    s.run(progress=False)
    pp = [c for c in s.collisions if c["kind"] == "pp"]
    assert pp and all(c["frozen_i"] or c["frozen_j"] for c in pp)

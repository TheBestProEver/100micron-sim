"""Physics validation tests: run with `python -m pytest -q`."""
import math
from dataclasses import replace

import numpy as np
import pytest

from dolle_sim.materials import get_material, pair_props, sphere_mass
from dolle_sim.contact import JKRContact
from dolle_sim.simulation import ChamberSim

R = 50e-6


def contact(name, **kw):
    m = replace(get_material(name), **kw)
    return JKRContact(pair_props(m, R, m, R)), sphere_mass(m, R) / 2


def test_elastic_limit_conserves_energy():
    c, me = contact("silica", e_dry=0.99999, gamma=1e-7, yield_pressure=math.inf)
    assert c.resolve(0.3, me).e_n == pytest.approx(1.0, abs=1e-3)


def test_damping_reproduces_dry_restitution():
    c, me = contact("silica", gamma=1e-7, yield_pressure=math.inf)
    assert c.resolve(0.1, me).e_n == pytest.approx(0.85, abs=0.005)


@pytest.mark.parametrize("v", [1.0, 3.0, 10.0])
def test_plasticity_matches_thornton_ning(v):
    c, me = contact("silica", e_dry=0.99999, gamma=1e-7, yield_pressure=0.5e9)
    x = c.v_yield(me) / v
    e_tn = (6 * math.sqrt(3) / 5) ** 0.5 * (1 - x * x / 6) ** 0.5 * (x / (x + 2 * math.sqrt(6 / 5 - x * x / 5))) ** 0.25
    assert c.resolve(v, me).e_n == pytest.approx(e_tn, abs=0.005)


def test_sticking_threshold_matches_jkr_hysteresis():
    # without damping the DEM threshold equals the JKR hysteresis estimate; with damping it is higher
    c, me = contact("silica", e_dry=0.99999)
    assert c.threshold(me) == pytest.approx(c.v_stick_elastic(me), rel=0.02)
    c2, me2 = contact("silica")
    assert c2.threshold(me2) > c2.v_stick_elastic(me2)
    assert c2.resolve(0.5 * c2.threshold(me2), me2).stuck
    assert not c2.resolve(2.0 * c2.threshold(me2), me2).stuck


def test_carbon_stickier_than_silica():
    cg, mg = contact("graphite")
    cs, ms = contact("silica")
    assert cg.threshold(mg) > cs.threshold(ms)


def _quiet_cfg(**env):
    e = {"preset": "ideal", "pressure_pa": 1.0, "gravity_gradient": False, "magnetic_field_t": [0, 0, 0]}
    e.update(env)
    return {"dt_s": 2e-3, "duration_s": 1.0, "environment": e,
            "forces": {"brownian": False, "self_gravity": False, "thermophoresis": False},
            "contact": {"contact_charging": False}}


def test_free_flight_with_drag_matches_analytic():
    a = 1e-3
    cfg = _quiet_cfg(residual_accel_g=[0, 0, a / 9.80665])
    ch = {"name": "t", "size_m": [1.0, 1.0, 1.0],
          "particles": [{"material": "silica", "count": 1, "radius_m": R,
                         "region_m": {"center": [0.5, 0.5, 0.5], "size": [0, 0, 0]},
                         "velocity": {"mode": "jet", "direction": [1, 0, 0], "cone_deg": 0,
                                      "speed_mean_m_s": 0.01, "speed_sigma_m_s": 0}}]}
    s = ChamberSim(cfg, ch, 0)
    x0, v0 = s.pos[0].copy(), s.V[0].copy()
    tau = s.mass[0] / s.dragc[0]
    s.run(progress=False)
    T = s.duration
    vt = np.array([0, 0, a * tau])
    x_exact = x0 + vt * T + (v0 - vt) * tau * (1 - math.exp(-T / tau))
    assert np.linalg.norm(s.pos[0] - x_exact) < 1e-8


def test_merge_and_split_conserve_momentum():
    cfg = _quiet_cfg()
    ch = {"name": "t", "size_m": [0.02] * 3,
          "particles": [{"material": "graphite", "count": 3, "radius_m": R, "velocity": {"speed_mean_m_s": 0.01}}]}
    s = ChamberSim(cfg, ch, 3)
    s._build_neighbors()
    # place grains in contact and merge them
    s.pos[1] = s.pos[0] + np.array([2 * R, 0, 0])
    s.pos[2] = s.pos[0] + np.array([0, 2 * R, 0])
    s.X[:] = s.pos
    s.L[:] = np.array([[1e-20, 0, 0], [0, 2e-20, 0], [0, 0, 3e-20]])
    P0 = (s.M[:3, None] * s.V[:3]).sum(0)
    L0 = sum(s.L[k] + s.M[k] * np.cross(s.X[k], s.V[k]) for k in range(3))
    s._merge(0, 1)
    s._merge(0, 2)
    b = s.body[0]
    assert s.nmono[b] == 3
    assert np.allclose(s.M[b] * s.V[b], P0, rtol=1e-12, atol=1e-25)
    Lb = s.L[b] + s.M[b] * np.cross(s.X[b], s.V[b])
    assert np.allclose(Lb, L0, rtol=1e-9, atol=1e-30)
    s._split(b, [k for k in s.members[b] if k != b][:1])
    alive = np.flatnonzero(s.alive)
    P1 = (s.M[alive, None] * s.V[alive]).sum(0)
    L1 = sum(s.L[k] + s.M[k] * np.cross(s.X[k], s.V[k]) for k in alive)
    assert np.allclose(P1, P0, rtol=1e-9, atol=1e-25)
    assert np.allclose(L1, L0, rtol=1e-6, atol=1e-30)


def test_contact_charging_saturates():
    cfg = _quiet_cfg()
    cfg["contact"]["contact_charging"] = True
    ch = {"name": "t", "size_m": [0.02] * 3, "wall_material": "glass_bk7",
          "particles": [{"material": "amorphous_carbon", "count": 1, "radius_m": R}]}
    s = ChamberSim(cfg, ch, 0)
    for _ in range(10000):
        s._charge(0, None, 1e-6)
    # saturates when the grain potential cancels the contact potential: k q / R = -(phi_grain - phi_wall)
    dphi = get_material("amorphous_carbon").work_function - get_material("glass_bk7").work_function
    assert s.q[0] == pytest.approx(-dphi * R / 8.9875517923e9, rel=1e-3)

"""Flight timeline (flight.py) and the mission runner (mission.py)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from dolle_sim.flight import FlightProfile, RESIDUAL_SPIN_OPTIONS_HZ
from dolle_sim.mission import launch_report, run_mission, tubule_size_m

ROOT = Path(__file__).resolve().parents[1]


def test_guide_anchors():
    """Default timeline = RockSat User Guide 2026, Table 3-1."""
    p = FlightProfile()
    assert p.burn1 == (0.0, 5.2)
    assert p.burn2[0] == pytest.approx(5.2 + 12.2) and p.burn2[1] == pytest.approx(5.2 + 12.2 + 11.7)
    assert p.axial_g(2.0) == 25.0 and p.axial_g(20.0) == 25.0
    assert p.spin_hz(p.burn2[1] - 1e-9) == pytest.approx(4.8, rel=1e-6)
    assert p.power_off_s == 332.0
    assert RESIDUAL_SPIN_OPTIONS_HZ["guide_upper"] == 1.0


def test_spin_angle_is_the_integral_of_the_spin_rate():
    p = FlightProfile()
    for t in (3.0, 29.1, 54.3, 60.0, 200.0):
        ref, _ = quad(lambda x: 2 * math.pi * p.spin_hz(x), 0, t, limit=400, points=[29.1, 54.0, 54.9])
        assert p.spin_angle(t) == pytest.approx(ref, rel=1e-6)


def test_loads_are_rotating_frame_kinematics():
    p = FlightProfile()
    r = np.array([0.08, 0.05])
    L = p.loads_at(40.0, r)                          # 4.8 Hz, between de-spin and burnout
    w = 2 * math.pi * 4.8
    assert np.hypot(*L["centrifugal"][:2]) == pytest.approx(w * w * np.hypot(*r) / 9.80665, rel=1e-9)
    assert L["axial"][2] == pytest.approx(-p.axial_g(40.0), rel=1e-12)


def test_tubule_geometry():
    t = {"inner_size_in": [0.5, 0.5, 1.5]}
    assert tubule_size_m({**t, "orientation": "standing"}) == pytest.approx([0.0127, 0.0127, 0.0381])
    assert tubule_size_m({**t, "orientation": "lying"}) == pytest.approx([0.0381, 0.0127, 0.0127])


def test_scenarios_combine():
    from dolle_sim.mission import apply_scenario
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    c = apply_scenario(m, "kick+lying")
    assert all(t["orientation"] == "lying" and t["release"][0]["type"] == "kick" for t in c["tubules"])
    assert all(t["orientation"] == "standing" for t in m["tubules"])          # the original is untouched


def _kick_run(force_recompute, count=150):
    from dolle_sim.mission import apply_scenario, mission_cfg, tubule_chamber
    from dolle_sim.simulation import ChamberSim
    m = apply_scenario(json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text()), "kick+spin_1hz")
    m["simulate_from_s"], m["simulate_to_s"] = 89.9, 100.0
    t = m["tubules"][1]                                  # graphite: grains stick, aggregates land on the walls
    t["dust"].update(count=count, charge_e_mean=2e4, charge_e_sigma=1e4)   # nonzero pair and space-charge forces
    s = ChamberSim(mission_cfg(m), tubule_chamber(m, t), 1)
    if force_recompute:                                  # no reuse of forces, no aggregate pre-check
        orig = s._forces_pos

        def always(vm):
            s._pair_stale = s._charge_stale = s._sc_dirty = True
            return orig(vm)
        s._forces_pos = always
        s._aggregate_may_fail = lambda b, a_sh: True
    s.run(progress=False)
    return s


def test_wall_bound_shortcuts_are_exact():
    """Reusing the pair and charge forces while every grain is wall-bound, and skipping the tree statics of
    aggregates that cannot fail, must not change anything (charged grains, a kick, aggregates on the walls)."""
    a, b = _kick_run(False), _kick_run(True)
    fb = np.flatnonzero(a.alive & a.frozen)
    assert a.n_events["detach"] > 0 and a.n_events["stick"] > 0 and (a.nmono[fb] > 1).any()
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.V, b.V) and np.array_equal(a.q, b.q)
    assert a.n_events == b.n_events and len(a.detach_log) == len(b.detach_log)


def test_mission_end_to_end(tmp_path):
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    m["tubules"] = m["tubules"][:2]
    for t in m["tubules"]:
        t["dust"]["count"] = 40
    m["simulate_from_s"] = 89.0
    m["simulate_to_s"] = 91.0
    m["engine"] = "v05"              # the mission pipeline on the fast v0.5 engine (the full engine: test_full_engine)
    rep = launch_report(m)
    assert len(rep["tubules"]) == 2
    summary, sims = run_mission(m, tmp_path, scenario="kick", jobs=1)
    assert (tmp_path / "mission_summary.json").exists()
    for t in summary["tubules"]:
        assert t["n_grains"] == 40
        assert sum(t["wall_bound_by_face"].values()) + t["free_at_end"] == 40


def test_drag_reverses_at_apogee_and_window_ends_at_100_km():
    p = FlightProfile(descent_attitude="tail_first")
    assert p.apparent_accel(150.0)[2] > 0 and p.apparent_accel(250.0)[2] < 0   # toward the nose, then the aft face
    d = FlightProfile()                     # default: ascent attitude until re-entering the air, then side-on
    assert d.apparent_accel(250.0)[2] < 0 and d.apparent_accel(250.0)[0] == 0
    assert d.apparent_accel(d.ug_end_s + 1.0)[0] < 0 and d.apparent_accel(d.ug_end_s + 1.0)[2] == 0
    o = FlightProfile(shock_direction="oblique").apparent_accel(882.0 + 0.025)
    assert o[0] != 0 and o[2] != 0 and abs(abs(o[0]) - abs(o[2])) < 1e-9
    ph = {x["name"]: x for x in p.phases()}
    assert ph["microgravity (above ~100 km)"]["t1"] == p.ug_end_s == ph["re-entry"]["t0"]


def test_descent_to_splashdown():
    """Guide times (chute T+460 s, splash T+882 s) and the fact-sheet chute altitude fix the fall; momentum and the
    handbook's 5 g opening fix the shocks; the parachute descent reaches the sea exactly at splashdown."""
    import math
    from scipy.integrate import quad, trapezoid
    from dolle_sim import atmosphere
    p = FlightProfile()
    assert p.end_s == p.splash_s + p.float_s and p.chute_s == 460.0 and p.splash_s == 882.0
    assert p._h[-1] == pytest.approx(p.chute_altitude_m, rel=1e-3)            # solved fall reaches 4.6 km
    g = 9.80665 * (6.371e6 / (6.371e6 + p._h)) ** 2                              # speed = integral of (g - drag)
    assert p.chute_speed_m_s == pytest.approx(trapezoid(g - p._f, p._t), rel=2e-3)
    rho0 = atmosphere.density(0.0)
    v_d = p.v_sea * math.sqrt(rho0 / atmosphere.density(p.chute_altitude_m))
    t1 = p.chute_s + p.chute_open_s
    J, _ = quad(lambda t: (p.axial_g(t) - 1.0) * 9.80665, p.chute_s, t1, limit=200)
    assert J == pytest.approx(p.chute_speed_m_s - v_d, rel=1e-6)               # opening impulse = momentum change
    assert max(p.axial_g(t) for t in np.linspace(p.chute_s, t1, 2001)) == pytest.approx(p.opening_peak_g, rel=1e-5)
    J, _ = quad(lambda t: (p.axial_g(t) - 1.0) * 9.80665, p.splash_s, p.splash_s + p.splash_stop_s, limit=200)
    assert J == pytest.approx(p.v_sea, rel=1e-6)                               # impact impulse = momentum
    h1 = p.chute_altitude_m - 0.5 * (p.chute_speed_m_s + v_d) * p.chute_open_s
    T, _ = quad(lambda h: 1.0 / (p.v_sea * math.sqrt(rho0 / atmosphere.density(h))), 0.0, h1, limit=200)
    assert t1 + T == pytest.approx(p.splash_s, abs=1.0)                        # reaches the sea at splashdown
    assert 9.1 <= p.v_sea <= 12.2                                              # NASA-MSFC 1975: 30-40 ft/s
    # directions: side-on in the fall, hanging nose down, afloat on its side
    assert p.apparent_accel(p.reentry_peak_s)[0] < 0 and p.apparent_accel(600.0) == pytest.approx([0, 0, 9.80665])
    assert p.apparent_accel(p.end_s - 1.0) == pytest.approx([-9.80665, 0, 0])
    q = FlightProfile(descent_attitude="tumbling", hang_attitude="nose_up", float_attitude="nose_up")
    a = np.array([q.apparent_accel(t) for t in np.linspace(360, 362, 9)])
    assert a[:, 0].min() < 0 < a[:, 0].max() and q.apparent_accel(600.0)[2] < 0

def _bed_cfg(prof, T0, T, **chamber):
    ch = {"name": "t", "size_m": [0.0127, 0.0127, 0.0381], "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": 40, "radius_m": 5e-5,
                         "initial_state": {"type": "settled", "face": "-z"}}]}
    ch.update(chamber)
    cfg = {"dt_s": 4e-3, "duration_s": T, "contact": chamber.pop("contact", {}),
           "environment": {"preset": "ideal", "pressure_pa": 0.0, "start_time_s": T0, "flight_profile": prof}}
    return cfg, ch


def test_non_adhesive_walls_release_resting_grains_when_the_load_turns():
    from dolle_sim.simulation import ChamberSim
    cfg, ch = _bed_cfg({"spin_burnout_hz": 0.0}, 6.0, 0.05)        # coast: drag pulls the dust off the aft face
    cfg["contact"] = {"wall_sticking": "never"}
    s = ChamberSim(cfg, ch, 0).run(progress=False)
    assert s.n_events["detach"] == 40


def test_release_window_without_a_flight_profile():
    from dolle_sim.simulation import ChamberSim
    cfg = {"dt_s": 4e-3, "duration_s": 0.05, "environment": {"preset": "ideal", "pressure_pa": 0.0}}
    ch = {"name": "t", "size_m": [0.0127, 0.0127, 0.0381], "wall_material": "glass_borosilicate",
          "release": [{"type": "kick", "start_s": 0.0213, "cycles": 1, "frequency_hz": 5000, "peak_g": 5000.0,
                       "axis": "z"}],
          "particles": [{"material": "silica", "count": 30, "radius_m": 5e-5,
                         "initial_state": {"type": "settled", "face": "-z"}}]}
    s = ChamberSim(cfg, ch, 0).run(progress=False)
    assert s.n_events["detach"] == 30          # a 0.2 ms kick between two 4 ms steps is not skipped


def test_settled_groups_never_overlap():
    from scipy.spatial import cKDTree
    from dolle_sim.simulation import ChamberSim
    g = {"radius_m": 5e-5, "count": 500, "initial_state": {"type": "settled", "face": "-z"}}
    ch = {"name": "t", "size_m": [0.0127, 0.0127, 0.0381],
          "particles": [dict(g, material="silica"), dict(g, material="graphite"),
                        dict(g, material="silica", initial_state={"type": "settled", "face": "-x"})]}
    s = ChamberSim({"dt_s": 4e-3, "duration_s": 0.01, "environment": {"preset": "ideal"}}, ch, 1)
    assert not len(cKDTree(s.pos).query_pairs(2 * 5e-5 * 1.0599))


def test_negative_spin_keeps_absolute_angular_momentum():
    from dolle_sim.simulation import ChamberSim
    ch = {"name": "t", "size_m": [50.0] * 3, "particles": [{"material": "silica", "count": 1, "radius_m": 5e-5,
          "region_m": {"center": [25, 25, 25], "size": [0, 0, 0]}, "velocity": {"speed_mean_m_s": 0.0}}]}
    T, f = 1.3, -0.5
    cfg = {"dt_s": 1e-3, "duration_s": T, "environment": {"preset": "ideal", "pressure_pa": 0.0,
           "spin_rate_hz": f, "gravity_gradient": False, "magnetic_field_t": [0, 0, 0]}}
    s = ChamberSim(cfg, ch, 0)
    I0 = s.Iiso[0]
    s.L[0] = I0 * 2 * math.pi * 3 * np.array([0.6, 0.0, 0.8])
    La0 = s.L[0] + I0 * s.env.omega
    s.run(progress=False)
    # independent reference: an inertially fixed vector seen from a frame turning at 2 pi f about +z
    th = -2 * math.pi * f * T
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    expected = Rz @ La0
    assert np.linalg.norm(s.L[0] + I0 * s.env.omega - expected) / np.linalg.norm(La0) < 1e-8
    assert np.linalg.norm(s.env.to_chamber(La0, T) - expected) / np.linalg.norm(La0) < 1e-12

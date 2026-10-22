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


def _kick_run(force_recompute, count=60):
    from dolle_sim.mission import apply_scenario, mission_cfg, tubule_chamber
    from dolle_sim.simulation import ChamberSim
    m = apply_scenario(json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text()), "kick")
    m["simulate_from_s"], m["flight_profile"]["power_off_s"] = 89.8, 90.6
    t = m["tubules"][0]
    t["dust"].update(count=count, charge_e_mean=2e4, charge_e_sigma=1e4)   # nonzero pair forces
    s = ChamberSim(mission_cfg(m), tubule_chamber(m, t), 3)
    if force_recompute:
        orig = s._forces_pos

        def always(vm):
            s._pair_stale = True
            return orig(vm)
        s._forces_pos = always
    s.run(progress=False)
    return s


def test_pair_force_cache_is_exact():
    """Reusing the pair forces while every grain is wall-bound must not change anything (charged grains, a kick
    that frees them, bounces, re-sticking)."""
    a, b = _kick_run(False), _kick_run(True)
    assert a.n_events["detach"] > 0 and a.n_events["wall_bounce"] > 0
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.V, b.V) and np.array_equal(a.q, b.q)
    assert a.n_events == b.n_events


def test_mission_end_to_end(tmp_path):
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    m["tubules"] = m["tubules"][:2]
    for t in m["tubules"]:
        t["dust"]["count"] = 40
    m["simulate_from_s"] = 89.0
    m["flight_profile"]["power_off_s"] = 91.0
    rep = launch_report(m)
    assert len(rep["tubules"]) == 2
    summary, sims = run_mission(m, tmp_path, scenario="kick", jobs=1)
    assert (tmp_path / "mission_summary.json").exists()
    for t in summary["tubules"]:
        assert t["n_grains"] == 40
        assert sum(t["wall_bound_by_face"].values()) + t["free_at_end"] == 40


def test_drag_reverses_at_apogee_and_window_ends_at_100_km():
    p = FlightProfile()
    assert p.apparent_accel(150.0)[2] > 0 and p.apparent_accel(250.0)[2] < 0   # toward the nose, then the aft face
    names = [ph["name"] for ph in p.phases()]
    assert names[-1] == "descent below ~100 km" and p.phases()[-2]["t1"] == p.ug_end_s


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

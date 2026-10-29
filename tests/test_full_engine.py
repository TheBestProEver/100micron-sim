"""The v0.6 full-contact engine: contact kernel, statics, slaved rolling, loads, vibration, regimes."""
import json
from pathlib import Path

import numpy as np
import pytest

from dolle_sim import verify_full as V

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("check", V.CHECKS, ids=lambda f: f.__name__)
def test_full_engine_check(check):
    r = check()
    assert r["passed"], f"{r['name']}: {r['value']} (required {r['criterion']})"


def test_jkr_closed_form_matches_newton():
    from dolle_sim import dem
    from dolle_sim.contact import JKRContact
    from dolle_sim.materials import get_material, pair_props
    s = get_material("silica")
    p = pair_props(s, 50e-6, s, 50e-6)
    c = JKRContact(p)
    for d in np.linspace(c.delta_break * 0.999, 5 * p.a0 ** 2 / p.R_star, 200):
        assert dem.jkr_a(d, p.R_star, p.w, p.E_star) == pytest.approx(c.solve_a(d, p.a0), rel=1e-12)


def test_loads_release_waveform_is_whole_cycles_from_rest():
    """A kick is one full cycle of tubule acceleration: no velocity left over, no jump at the start."""
    from dolle_sim.engine import FullContactSim
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    from dolle_sim.mission import apply_scenario, mission_cfg, tubule_chamber
    m = apply_scenario(m, "kick")
    m["simulate_from_s"], m["simulate_to_s"] = 89.0, 89.5
    t = m["tubules"][0]
    t["dust"]["count"] = 5
    S = FullContactSim(mission_cfg(m), tubule_chamber(m, t), 0)
    r = S.loads.release[0]
    T = np.linspace(r["start"], r["end"], 20001)
    a = S.loads.release_accel(T)[:, 2]
    assert a[0] == pytest.approx(0.0, abs=1e-6 * r["a"])
    dv = float(np.sum(0.5 * (a[1:] + a[:-1]) * np.diff(T)))
    assert abs(dv) < 1e-6 * r["a"] / r["f"]


def test_full_engine_mission_smoke(tmp_path):
    """Lift-off of a few grains on the new engine through the first burn and into the coast: every grain stays
    accounted for, nothing comes off while the 25 g boost presses the dust down (burn 1, T+0-5.2 s), the summary is
    written. (In the coast a grain rolling outward can knock a resting one off the floor: that is physics.)"""
    from dolle_sim.mission import run_mission
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    m["tubules"] = m["tubules"][:1]
    m["tubules"][0]["dust"]["count"] = 20
    m["simulate_to_s"] = 6.0
    summary, sims = run_mission(m, tmp_path, jobs=1)
    t = summary["tubules"][0]
    assert t["n_grains"] == 20
    assert sum(t["wall_bound_by_face"].values()) + t["free_at_end"] == 20
    assert t["detachments_by_phase"]["first-stage burn"] == 0
    assert all(d["t"] >= 5.2 for d in sims[0].detach_log)
    assert (tmp_path / "mission_summary.json").exists()


def test_cluster_sleeps_and_wakes_whole():
    """Two touching grains on a floor settle in the full model, go to sleep as one cluster (the exact energy-ellipsoid
    bound), and a forced wake of one member wakes both at the cluster's static equilibrium, at rest."""
    from dolle_sim.engine import FullContactSim, ACTIVE, ASLEEP
    cfg = {"dt_s": 1e-3, "duration_s": 0.3, "output_fps": 100,
           "environment": {"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": [0.0, 0.0, -1.0]},
           "contact": {"contact_charging": False}, "forces": {}}
    ch = {"name": "pair", "size_m": [0.001, 0.001, 0.001], "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": 2, "radius_m": 50e-6, "velocity": {"speed_mean_m_s": 0.0},
                         "initial_state": {"type": "free"}}]}
    S = FullContactSim(cfg, ch, 0)
    S.base.F_static[:] = 0.0
    R = 50e-6
    S.x[0] = [0.5e-3, 0.5e-3, R]
    S.x[1] = [0.5e-3 + 2 * R - 2e-9, 0.5e-3, R]
    S.v[:], S.w[:] = 0.0, 0.0
    S.regime[:] = ACTIVE
    S.run(progress=False)
    assert (S.regime == ASLEEP).all() and 0 in S._island_of and S._island_of[0] == S._island_of[1]
    x_sleep = S.x.copy()
    bad, vf = S._step_once(S.t, 1e-4, {0}, 1.0)
    assert not bad and vf is None
    assert 0 not in S._island_of and 1 not in S._island_of
    assert np.abs(S.x - x_sleep).max() < 1e-9


def test_checkpoint_with_sleeping_cluster_is_exact():
    """A state holding a sleeping cluster (its factorised stiffness is not picklable and is rebuilt on loading)
    saves, loads and continues bit-for-bit as the run kept in memory."""
    import pickle
    from dolle_sim.engine import FullContactSim, ACTIVE, ASLEEP
    cfg = {"dt_s": 1e-3, "duration_s": 0.3, "output_fps": 100,
           "environment": {"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": [0.0, 0.0, -1.0]},
           "contact": {"contact_charging": False}, "forces": {}}
    ch = {"name": "pair", "size_m": [0.001, 0.001, 0.001], "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": 2, "radius_m": 50e-6, "velocity": {"speed_mean_m_s": 0.0},
                         "initial_state": {"type": "free"}}]}
    S = FullContactSim(cfg, ch, 0)
    S.base.F_static[:] = 0.0
    R = 50e-6
    S.x[0] = [0.5e-3, 0.5e-3, R]
    S.x[1] = [0.5e-3 + 2 * R - 2e-9, 0.5e-3, R]
    S.v[:], S.w[:] = 0.0, 0.0
    S.regime[:] = ACTIVE
    S.run(progress=False)
    assert (S.regime == ASLEEP).all() and S._islands
    L = pickle.loads(pickle.dumps(S, protocol=pickle.HIGHEST_PROTOCOL))
    assert all(isl.get("lu") is None for isl in L._islands.values())
    for sim in (S, L):
        sim.duration = 0.35
        sim.run(progress=False)
    assert all(np.array_equal(getattr(S, a), getattr(L, a)) for a in ("x", "v", "w", "regime"))
    assert all(isl.get("lu") is not None for isl in L._islands.values())


def test_branch_from_checkpoint_is_exact(tmp_path):
    """A run continued from another scenario's checkpoint (identical up to it) is bit-for-bit the run of its own
    scenario from lift-off; a branch into a scenario that differs earlier is refused."""
    from dolle_sim.engine import FullContactSim, branch
    from dolle_sim.mission import apply_scenario, mission_cfg, tubule_chamber
    base = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    kick = [{"type": "kick", "start_s": 4.0, "cycles": 1, "frequency_hz": 5000, "peak_g": 2000.0, "axis": "z"}]

    def sim(release, scenario=None, cps=(3.95,)):
        m = apply_scenario(base, scenario)
        m["simulate_to_s"] = 4.1
        t = m["tubules"][0]
        t["dust"]["count"] = 60
        t["release"] = release
        S = FullContactSim(mission_cfg(m), tubule_chamber(m, t), 0)
        S.checkpoints, S.checkpoint_dir = list(cps), tmp_path
        return S
    A = sim([]).run(progress=False)
    cp = A.checkpoint_path(3.95)
    assert cp.exists()
    B = sim(kick, cps=()).run(progress=False)          # the kick scenario from lift-off
    C = branch(cp, sim(kick, cps=()))
    C.run(progress=False)
    assert C.n_events["branched_at_T"] == 3.95
    assert np.array_equal(B.x, C.x) and np.array_equal(B.v, C.v) and np.array_equal(B.w, C.w)
    assert len(B.collisions) == len(C.collisions) and len(B.detach_log) == len(C.detach_log) > 0
    with pytest.raises(ValueError):
        branch(cp, sim(kick, "rough_grains", cps=()))
    late = [dict(kick[0], start_s=3.9)]
    with pytest.raises(ValueError):
        branch(cp, sim(late, cps=()))


def _scene(n, size=0.002, g=(0.0, 0.0, -1.0)):
    from dolle_sim.engine import FullContactSim
    cfg = {"dt_s": 1e-3, "duration_s": 0.3, "output_fps": 100,
           "environment": {"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": list(g)},
           "contact": {"contact_charging": False}, "forces": {}}
    ch = {"name": "scene", "size_m": [size] * 3, "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": n, "radius_m": 50e-6, "velocity": {"speed_mean_m_s": 0.0},
                         "initial_state": {"type": "free"}}]}
    S = FullContactSim(cfg, ch, 0)
    S.base.F_static[:] = 0.0
    S.v[:], S.w[:] = 0.0, 0.0
    return S


def test_rolling_grain_wall_bound_assumes_no_direction():
    """A rolling grain's envelope bounds how far it rolls, not in which direction (Coriolis, the rolling resistance
    and the load along the wall turn it): rolling parallel to a side wall 10 um away at 10 mm/s, its capture time
    for that wall is at most (gap - d_cap) / |v|."""
    from dolle_sim.engine import ROLL
    S = _scene(1)
    R, T0 = 50e-6, float(S.env.t0_flight)
    S.x[0] = [S.L[0] - R - 10e-6, 1e-3, R]
    S._place_at_equilibrium(0, 4, S._row(0, -5), T0)
    S.regime[0] = ROLL
    S._roll_face[0] = 4
    S.v[0] = [0.0, 0.01, 0.0]
    S.w[0] = [-0.01 / R, 0.0, 0.0]
    H = 1e-3
    s, s_off = S._motion_bound(H, T0), np.zeros(1)
    s[:], s_off[:] = S._roll_bound(np.array([0]), H, T0)
    S._vd_step, S._ag_step = np.zeros(1), np.zeros(1)
    t_hit, _ = S._capture_times(np.array([0]), H, T0, s, s_off)
    gap = S.L[0] - S.rad[0] - S.x[0, 0]
    assert t_hit[0] <= (gap - S.d_cap) / 0.01 * (1 + 1e-9)


def test_grains_leaving_together_have_their_speed_bounds_checked():
    """Grains that leave the DEM in one call (each relying on the others' speed bounds) and rejoin later are in the
    speed record of that call, so their speed after rejoining is checked against what the others assumed."""
    from dolle_sim.engine import ACTIVE
    S = _scene(4, g=(0.0, 0.0, 0.0))
    R, c = 50e-6, 1e-3
    S.x[0] = [c, c, c]
    S.x[1] = [c + 2 * R + 0.5e-6, c, c]
    S.x[2] = [c + 4 * R + 1.1e-6, c, c]
    S.x[3] = [c + 6 * R + 1.1e-6 - 2e-9, c, c]
    S.v[2] = S.v[3] = [-1e-3, 0.0, 0.0]
    S.regime[:] = ACTIVE
    S.base._build_neighbors()
    S._step_once(0.0, S._outer_dt(0.0), set(), 1.0)
    left = [(te, k) for k, (te, xe, Se, tr, Ve, Ae) in S._chk["exited"].items() if tr < S._outer_dt(0.0)]
    assert left
    for te, k in left:
        assert any(abs(tc - te) < 1e-15 and k in set(idx.tolist()) for tc, idx, vb in S._chk["speed"])


def test_contact_formed_outside_the_dem_is_redone():
    """Safety net: a grain whose capture-time bound is (deliberately) wrong flies into a wall outside the DEM; the
    overlap without a contact is found, the step is redone with the grain in the DEM throughout, and it bounces."""
    S = _scene(1, g=(0.0, 0.0, 0.0))
    R = 50e-6
    S.x[0] = [1e-3, 1e-3, R + 5e-6]
    S.v[0] = [0.0, 0.0, -0.05]
    S._capture_times = lambda free, H, T, s, s_off: (np.full(len(free), np.inf), np.zeros(S.N, bool))
    S.duration = 2e-3
    S.run(progress=False)
    assert S.n_events.get("redos_missed_contact", 0) >= 1
    assert S.x[0, 2] > R and S.v[0, 2] > 0                        # it bounced off the floor


def test_vibration_window_ends_are_one_sided_and_bounded():
    """A step that starts where the vibration ends feels none of it, a step that ends there feels its last value, and
    the bound over an interval touching a window end includes the end value."""
    from dolle_sim.vibration import VibrationHistory
    h = VibrationHistory(1.0, 20000.0, np.random.default_rng(0).normal(size=(3, 201)))
    end = np.array([h.t0 + h.T])
    assert np.all(h.accel(end, "+") == 0.0) and np.any(h.accel(end, "-") != 0.0)
    assert np.all(h.accel(np.array([h.t0]), "-") == 0.0) and np.any(h.accel(np.array([h.t0]), "+") != 0.0)
    assert np.all(h.abs_max(end[0], end[0] + 1e-4) >= np.abs(h.accel(end, "-")[:, 0]) * (1 - 1e-12))


def test_engine_refuses_loads_it_cannot_bound():
    """g-jitter is random with no bound, so the full-contact engine refuses it rather than ignoring it."""
    from dolle_sim.engine import FullContactSim
    cfg = {"dt_s": 1e-3, "duration_s": 1e-3, "output_fps": 100,
           "environment": {"preset": "ideal", "pressure_pa": 0.0, "jitter_rms_g": 1e-3},
           "contact": {"contact_charging": False}, "forces": {}}
    ch = {"name": "j", "size_m": [0.002] * 3, "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": 1, "radius_m": 50e-6, "velocity": {"speed_mean_m_s": 0.0},
                         "initial_state": {"type": "free"}}]}
    with pytest.raises(ValueError):
        FullContactSim(cfg, ch, 0)


def test_tubule_by_tubule_run_is_the_mission_run(tmp_path):
    """A campaign may run a scenario's tubules one by one (run_tubule) and assemble the outputs: the result is
    exactly run_mission's (same configurations and seeds)."""
    from dolle_sim.mission import run_mission, run_tubule, assemble_mission
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    m["tubules"] = m["tubules"][:2]
    for t in m["tubules"]:
        t["dust"]["count"] = 12
    m["simulate_to_s"] = 2.0
    s1, sims1 = run_mission(m, tmp_path / "a", jobs=1, write_outputs=False)
    for i in range(2):
        run_tubule(m, tmp_path / "b", None, i)
    s2, sims2 = assemble_mission(m, tmp_path / "b", write_outputs=False)
    for s in (s1, s2):                                          # computing time differs, of course
        for t in s["tubules"]:
            t.pop("wall_time_s", None)
    assert json.dumps(s1, sort_keys=True, default=str) == json.dumps(s2, sort_keys=True, default=str)
    for a, b in zip(sims1, sims2):
        assert all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("x", "v", "w", "regime"))

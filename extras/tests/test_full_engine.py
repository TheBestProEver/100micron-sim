"""The v0.6 full-contact engine: contact kernel, statics, slaved rolling, loads, vibration, regimes."""
import json
from pathlib import Path

import numpy as np
import pytest

from dolle_extras import verify_full as V

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("check", V.CHECKS, ids=lambda f: f.__name__)
def test_full_engine_check(check):
    r = check()
    assert r["passed"], f"{r['name']}: {r['value']} (required {r['criterion']})"


def test_jkr_closed_form_matches_newton():
    from dolle_extras import dem
    from dolle_sim.contact import JKRContact
    from dolle_sim.materials import get_material, pair_props
    s = get_material("silica")
    p = pair_props(s, 50e-6, s, 50e-6)
    c = JKRContact(p)
    for d in np.linspace(c.delta_break * 0.999, 5 * p.a0 ** 2 / p.R_star, 200):
        assert dem.jkr_a(d, p.R_star, p.w, p.E_star) == pytest.approx(c.solve_a(d, p.a0), rel=1e-12)


def test_loads_release_waveform_is_whole_cycles_from_rest():
    """A kick is one full cycle of tubule acceleration: no velocity left over, no jump at the start."""
    from dolle_extras.engine import FullContactSim
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
    from dolle_extras.full_mission import run_args
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    m["engine"] = "full"                       # (dolle_sim runs only the v0.5 engine; the runner is the v0.6 one)
    m["tubules"] = m["tubules"][:1]
    m["tubules"][0]["dust"]["count"] = 20
    m["simulate_to_s"] = 6.0
    summary, sims = run_mission(m, tmp_path, jobs=1, runner=run_args)
    t = summary["tubules"][0]
    assert "dem_steps" in t["events"]          # it ran the full-contact engine
    assert t["n_grains"] == 20
    assert sum(t["wall_bound_by_face"].values()) + t["free_at_end"] == 20
    assert t["detachments_by_phase"]["first-stage burn"] == 0
    assert all(d["t"] >= 5.2 for d in sims[0].detach_log)
    assert (tmp_path / "mission_summary.json").exists()


def test_cluster_sleeps_and_wakes_whole():
    """Two touching grains on a floor settle in the full model, go to sleep as one cluster (the exact energy-ellipsoid
    bound), and a forced wake of one member wakes both at the cluster's static equilibrium, at rest."""
    from dolle_extras.engine import FullContactSim, ACTIVE, ASLEEP
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
    from dolle_extras.engine import FullContactSim, ACTIVE, ASLEEP
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
    from dolle_extras.engine import FullContactSim, branch
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
    from dolle_extras.engine import FullContactSim
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
    from dolle_extras.engine import ROLL
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
    from dolle_extras.engine import ACTIVE
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
    from dolle_extras.vibration import VibrationHistory
    h = VibrationHistory(1.0, 20000.0, np.random.default_rng(0).normal(size=(3, 201)))
    end = np.array([h.t0 + h.T])
    assert np.all(h.accel(end, "+") == 0.0) and np.any(h.accel(end, "-") != 0.0)
    assert np.all(h.accel(np.array([h.t0]), "-") == 0.0) and np.any(h.accel(np.array([h.t0]), "+") != 0.0)
    assert np.all(h.abs_max(end[0], end[0] + 1e-4) >= np.abs(h.accel(end, "-")[:, 0]) * (1 - 1e-12))


def test_engine_refuses_loads_it_cannot_bound():
    """g-jitter is random with no bound, so the full-contact engine refuses it rather than ignoring it."""
    from dolle_extras.engine import FullContactSim
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
    from dolle_extras.full_mission import run_args
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    m["engine"] = "full"
    m["tubules"] = m["tubules"][:2]
    for t in m["tubules"]:
        t["dust"]["count"] = 12
    m["simulate_to_s"] = 2.0
    s1, sims1 = run_mission(m, tmp_path / "a", jobs=1, write_outputs=False, runner=run_args)
    for i in range(2):
        run_tubule(m, tmp_path / "b", None, i, runner=run_args)
    s2, sims2 = assemble_mission(m, tmp_path / "b", write_outputs=False)
    for s in (s1, s2):                                          # computing time differs, of course
        for t in s["tubules"]:
            t.pop("wall_time_s", None)
    assert json.dumps(s1, sort_keys=True, default=str) == json.dumps(s2, sort_keys=True, default=str)
    assert "dem_steps" in s1["tubules"][0]["events"]
    for a, b in zip(sims1, sims2):
        assert all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("x", "v", "w", "regime"))


def test_campaign_cli_tubule_jobs_and_assemble(tmp_path):
    """The command lines the campaign runs: one tubule at a time (`python -m dolle_extras mission --tubule i`), then
    `mission --assemble`. dolle_extras runs the full-contact engine unless the config names another."""
    import subprocess
    import sys
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    m["tubules"] = m["tubules"][:2]
    for t in m["tubules"]:
        t["dust"]["count"] = 8
    m["simulate_to_s"] = 1.0
    cfg = tmp_path / "tiny.json"
    cfg.write_text(json.dumps(m))
    out = tmp_path / "out"
    for arg in (["--tubule", "0"], ["--tubule", "1"], ["--assemble"]):
        r = subprocess.run([sys.executable, "-m", "dolle_extras", "mission", str(cfg), "--out", str(out), *arg],
                           cwd=ROOT, capture_output=True, text=True)       # (conftest.py sets PYTHONPATH)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    s = json.loads((out / "mission_summary.json").read_text())
    assert [t["n_grains"] for t in s["tubules"]] == [8, 8]
    assert "dem_steps" in s["tubules"][0]["events"]


def test_killed_tubule_run_continues_from_its_autosave(tmp_path):
    """A tubule run killed part-way (out of memory, reboot) and started again continues from its last autosave; the
    result is bit-for-bit the uninterrupted run, and an autosave of another configuration is not used."""
    import pickle
    from dolle_extras import engine
    from dolle_extras.full_mission import run_args
    from dolle_sim.mission import _tubule_args, autosave_path, run_tubule
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    m["engine"] = "full"
    m["tubules"] = m["tubules"][:1]
    m["tubules"][0]["dust"]["count"] = 40
    m["simulate_to_s"] = 6.0                     # through burnout (grains wake, roll and collide)
    m["autosave_s"] = 0.0                        # save at every outer step
    _, _, args = _tubule_args(m, tmp_path / "a", None)
    A = run_args(args[0])                        # uninterrupted
    assert A.n_events["active_grain_steps"] > 0
    run = engine.FullContactSim.run
    calls = {"n": 0}

    def killed(self, *a, **k):                   # die after ~2/3 of the outer steps
        def cb(t, d, ev):
            calls["n"] += 1
            if t > 0.66 * d:
                raise MemoryError("killed")
        return run(self, *a, callback=cb, callback_interval_s=0.0, **k)
    engine.FullContactSim.run = killed
    try:
        with pytest.raises(MemoryError):
            run_tubule(m, tmp_path / "b", None, 0, runner=run_args)
    finally:
        engine.FullContactSim.run = run
    auto = autosave_path(tmp_path / "b", A.name)
    with open(auto, "rb") as f:
        mid = pickle.load(f)
    assert 0.5 * A.duration < mid.t < A.duration
    path = run_tubule(m, tmp_path / "b", None, 0, runner=run_args)    # started again: continues from the autosave
    assert not auto.exists()
    with open(path, "rb") as f:
        B = pickle.load(f)
    assert all(np.array_equal(getattr(A, k), getattr(B, k)) for k in ("x", "v", "w", "regime"))
    assert len(A.collisions) == len(B.collisions) and A.n_events == B.n_events
    # an autosave of another configuration (here: another dust amount) is ignored
    m2 = json.loads(json.dumps(m))
    m2["tubules"][0]["dust"]["count"] = 41
    m2["simulate_to_s"] = 0.5
    with open(auto, "wb") as f:
        pickle.dump(mid, f)
    _, _, args2 = _tubule_args(m2, tmp_path / "b", None)
    C = run_args(args2[0])
    assert C.N == 41 and abs(C.t - 0.5) < 1e-9


# ---------------------------------------------------------------------------- sleeping clusters at their caps
R50 = 50e-6


def _rows_on_floor(groups, g=(0.0, 0.0, -1.0), roll=None, extra=0, env=None, **contact):
    """Rows of 50 um silica grains along x on the floor of a 1 mm box (groups: (x of the first centre, count) each,
    at y = 0.5 mm), every grain touching the floor and its neighbours, placed at the exact static equilibrium of the
    real contact law under the uniform load g [g] (or a flight profile through env): the contacts formed at their
    zero-load overlaps by one kernel evaluation at rest, then statics.solve_equilibrium, first with the caps lifted,
    then with the real kernel's return map and the released tangent (the engine's own tolerances). roll: every floor
    rolling spring first set to that fraction of its cap along +x, the locked-in rolling deflection an earlier load
    leaves; above 1 the return map brings it back to the cap, and the placed state is snapped to (1 - 2^-40) of the
    caps as a sleeping island's is (statics.snap_to_caps). A row is held against rotation about y by its sliding
    springs (the pair contact and the floor), so its rolling springs at their caps leave no mechanism. extra grains
    stay free, at rest, in the upper corner. Returns the sim (the rows ACTIVE, at rest, neighbour lists built), the
    members of each row and the contact rows."""
    from scipy.sparse.linalg import splu
    from dolle_extras import dem, statics
    from dolle_extras.engine import FullContactSim, ACTIVE
    n = sum(c for _, c in groups)
    cfg = {"dt_s": 1e-3, "duration_s": 0.3, "output_fps": 100,
           "environment": dict({"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": list(g)}, **(env or {})),
           "contact": dict({"contact_charging": False}, **contact), "forces": {}}
    ch = {"name": "rows", "size_m": [0.001] * 3, "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": n + extra, "radius_m": R50,
                         "velocity": {"speed_mean_m_s": 0.0}, "initial_state": {"type": "free"}}]}
    S = FullContactSim(cfg, ch, 0)
    S.base.F_static[:] = 0.0
    S.v[:], S.w[:] = 0.0, 0.0
    pw, pp = S._params(0, -5), S._params(0, 1)
    dw, dp = (p[dem.P_A0] ** 2 / (3 * p[dem.P_REFF]) for p in (pw, pp))    # zero-load JKR overlaps
    keys, mems, k = [], [], 0
    for x0, c in groups:
        mem = np.arange(k, k + c)
        for q, i in enumerate(mem):
            S.x[i] = [x0 + q * (2 * R50 - dp), 0.5e-3, R50 - dw]
        keys += [(int(i), -5) for i in mem] + [(int(i), int(i) + 1) for i in mem[:-1]]
        mems.append(mem)
        k += c
    for q in range(extra):
        S.x[n + q] = [0.85e-3 - 0.15e-3 * q, 0.85e-3, 0.5e-3]
    mem = np.arange(n)
    rows = np.array([S._row(i, j) for i, j in keys])
    ci, cj = np.array([i for i, _ in keys]), np.array([j for _, j in keys])
    x, prm = np.ascontiguousarray(S.x[mem]), np.ascontiguousarray(S._prm[rows])
    prm_el = statics.lift_caps(prm)
    _, h = statics.apply_displacement(x, ci, cj, prm, np.zeros((len(rows), dem.NH)), S.L, np.zeros((n, 6)))
    if roll is not None:
        lam, _, out = statics.state(x, ci, cj, prm_el, h, S.L)
        fl = cj == -5
        h[fl, dem.H_XR] = -roll * statics.caps(lam, prm, out)[fl, 1] / prm[fl, dem.P_KR]
    K, J = statics.jacobians(x, S.rad[mem], ci, cj, prm_el, h, S.L)
    lu = splu(K)
    lam, Fnet, out = statics.state(x, ci, cj, prm_el, h, S.L)
    tt, tr = S._island_tols(statics.caps(lam, prm, out), prm, np.ones(len(rows), bool))
    F = S._island_load(mem, float(S.env.t0_flight))
    E = statics.solve_equilibrium(x, ci, cj, prm_el, prm_el, h, S.L, lu, F, lu.solve(F + Fnet), tt, tr, 20)
    E = statics.solve_equilibrium(x, ci, cj, prm, prm_el, h, S.L, lu, F, E["du"], tt, tr, 60,
                                  tangent=S._island_tangent(dict(K=K, J=J, prm=prm, lu=lu), False))
    assert E["ok"]
    S.x[mem] = E["xn"]
    S._hist[rows] = statics.snap_to_caps(E["hn"], E["lam"], prm, E["out"])[0]
    S.regime[mem] = ACTIVE
    S.base._build_neighbors()
    S._run_started = True
    return S, mems, rows


def _floor_rolling(S, isl):
    """For a sleeping island at its committed state: the floor contacts (mask), |k_r xi_r| over their rolling
    caps, the caps [N] (statics.caps: mu_r |F_ne + 2 F_pull| from the state's own normal forces) and the elastic
    normal forces [N]."""
    from dolle_extras import dem, statics
    x = np.ascontiguousarray(S.x[isl["members"]])
    h = np.ascontiguousarray(S._hist[isl["rows"]])
    lam, _, out = statics.state(x, isl["ci"], isl["cj"], isl["prm_el"], h, S.L)
    cap = statics.caps(lam, isl["prm"], out)[:, 1]
    fr = statics.kcomp(isl["prm"], h)[:, 1] * np.linalg.norm(h[:, dem.H_XR:dem.H_XR + 3], axis=1)
    fl = (isl["cj"] == -5) & (h[:, dem.H_TOUCH] > 0.5)
    return fl, fr / np.maximum(cap, 1e-300), cap, lam.reshape(-1, 8)[:, 0]


def _spring_change(isl, h_a, h_b):
    """The largest change of any sliding, rolling or twisting spring force of an island's touching contacts between
    the histories h_a and h_b, |k (xi_a - xi_b)| (statics.kcomp), relative to that spring's cap (the island's capv)."""
    from dolle_extras import dem, statics
    d = h_a - h_b
    dd = np.column_stack([np.linalg.norm(d[:, dem.H_XT:dem.H_XT + 3], axis=1),
                          np.linalg.norm(d[:, dem.H_XR:dem.H_XR + 3], axis=1), np.abs(d[:, dem.H_XTW])])
    tm = h_b[:, dem.H_TOUCH] > 0.5
    return float((statics.kcomp(isl["prm"], h_b) * dd / np.maximum(isl["capv"], 1e-300))[tm].max())


def _ramp(S, dg, T_ramp):
    """Add to the sim's uniform load a linear ramp of dg [g] (per axis) from the run's start over T_ramp, then held:
    Loads.accel and Loads.accel_bound wrapped. A linear piece has no chord deviation inside a step, so the 0 that
    Loads.chord_dev returns for the constant load underneath stays exact (the tests stop before the ramp's end)."""
    from dolle_sim.environment import G0
    T0 = float(S.env.t0_flight)
    acc, bnd = S.loads.accel, S.loads.accel_bound
    da = np.asarray(dg, float) * G0

    def f(T):
        return np.clip((np.atleast_1d(np.asarray(T, float)) - T0) / T_ramp, 0.0, 1.0)

    def accel(T, coarse=1e-5, side=None):
        return acc(T, coarse, side) + f(T)[:, None] * da[None, :]

    def accel_bound(Ta, Tb):
        return bnd(Ta, Tb) + np.abs(da) * float(f(Tb)[0])
    S.loads.accel, S.loads.accel_bound = accel, accel_bound


def _steps(S, n):
    """n outer steps from S.t (step() itself, no _skip_asleep)."""
    t = S.t
    for _ in range(n):
        H = S._outer_dt(t)
        S.step(t, H)
        t += H
    return t


def test_island_at_its_rolling_cap_sleeps_exactly_under_constant_load():
    """A pair on the floor whose floor rolling springs sit at their caps (set there by hand: an admissible state of
    the contact law, a deflection an earlier load can leave locked in; the -2 g along the row alone holds them at
    1.5e-4 of their caps, measured with the caps lifted) sleeps at once as an island at its caps
    (cluster_sleep_cap, tier 1: the real kernel takes no clip branch, design README section 3), and under a constant
    load it does not move (critique E1: the engine's committed histories bit-identical). The placement is the entry's
    solve, not yet a fixed point of the step's: the first commit's Newton step removes its residual (du ~ 1e-22 m)
    and the second commit's kernel evaluation changes the rounding residue of a rolling spring along its contact
    normal (~5e-24 m) by one ulp; in these two commits every spring force moves by <= island_tol_rel of its cap (the
    Newton tolerance; measured ~1e-15). From then on 50 steps leave the committed histories bit-identical, the
    positions stay within 1e-15 m of the placement, and every step is tier 1."""
    from dolle_extras.engine import ASLEEP
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001)
    S._classify(0.0)
    assert (S.regime[mem] == ASLEEP).all() and S.n_events["cluster_sleep_cap"] == 1
    iid = S._island_of[0]
    fl, ratio, _, _ = _floor_rolling(S, S._islands[iid])
    assert fl.sum() == 2 and np.all(np.abs(ratio[fl] - 1) <= 1e-9)      # (the placement's Newton solve)
    nr = len(S._ckeys)
    x0, h0 = S.x.copy(), S._hist[:nr].copy()
    _steps(S, 2)
    h1 = S._hist[:nr].copy()
    isl = S._islands[iid]
    assert _spring_change(isl, h1[isl["rows"]], h0[isl["rows"]]) <= S.island_tol_rel
    assert np.abs(S.x - x0).max() <= 1e-15
    _steps(S, 50)
    assert np.array_equal(S._hist[:nr], h1) and np.abs(S.x - x0).max() <= 1e-15
    assert (S.regime[mem] == ASLEEP).all() and S._islands[iid]["tier"] == 1
    assert S.n_events["island_static_steps"] == 52 and S.n_events["island_creep_steps"] == 0


def test_creep_island_follows_its_shrinking_cap_and_reports_its_bounds():
    """The pair at its rolling caps while the pressing load falls from 1 g to 0.2 g over 20 ms (the early coast's
    cap shrink, critique C2): every rolling cap shrinks each step, so the island creeps (tier 2, the kernel's return
    map at equilibrium). After every commit each floor rolling spring force |k_r xi_r| equals its new cap to 1e-9
    (it is snapped to 1 - 2^-40 of it), the caps sum to mu_r (the pressing load + 2 F_pull per contact) with the
    load of that moment (vertical equilibrium: an independent check that the limit is the right one), the island
    moves 0 < du_max < island_motion_m, and its bounds are positive and inside their tolerances: per step flow <=
    island_flow_max and path error <= island_path_tol of the deflection at the cap, snap <= 2^-40 (relative), the
    Newton term <= island_tol_rel, the budgeted total <= island_budget_rel; the chord term is exactly 0 (the load
    is linear inside each step; the half-step pair's path error is 0 in some steps, so per step it is >= 0 and in
    sum > 0). The tier-2 error terms that tier 1 certifies through D are in the budget (review): the Newton
    residuals of the committed, half-step and start states through the released structure (resid_m, summed) and the
    lag the drift's dynamics leave (dyn_m, its running maximum), both > 0 and in total_rel, and the per-step inputs
    are reported (island_newton_rel, island_dyn_rel). With island_flow_max made tiny the next creep step wakes the
    island (island_wake_flow), and its re-entry at the end of that step is refused by the same test
    (cluster_awake_certificate)."""
    from dolle_extras import dem
    from dolle_extras.engine import ASLEEP
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001)
    _ramp(S, (0.0, 0.0, 0.8), 0.02)
    S._classify(0.0)
    iid = S._island_of[0]
    assert S._islands[iid]["tier"] == 2 and S.n_events["cluster_sleep_creep"] == 1
    fl, _, cap_prev, _ = _floor_rolling(S, S._islands[iid])
    p = S._islands[iid]["prm"][fl]
    flow_prev, path_prev = 0.0, 0.0
    for q in range(10):
        _steps(S, 1)
        isl = S._islands[iid]
        assert isl["tier"] == 2 and (S.regime[mem] == ASLEEP).all()
        fl, ratio, cap, Fne = _floor_rolling(S, isl)
        assert np.all(np.abs(ratio[fl] - 1) <= 1e-9) and np.all(cap[fl] < cap_prev[fl])
        az = -float(S.loads.accel(np.array([isl["T_ref"]]), side="-")[0, 2])
        tt = S._island_tols(isl["capv"], isl["prm"], fl | (isl["cj"] >= 0))[0]
        assert abs(Fne[fl].sum() - S.m[mem].sum() * az) <= len(mem) * tt       # (the solve's force tolerance)
        assert np.allclose(cap[fl].sum(), (p[:, dem.P_MUR] * (Fne[fl] + 2 * p[:, dem.P_FC])).sum(), rtol=1e-12)
        assert 0.0 < isl["du_max"] < S.island_motion_m
        xi_cap = cap[fl].min() / p[0, dem.P_KR]
        b = isl["bnd"]
        assert 0.0 < b["flow_m"][1] - flow_prev <= S.island_flow_max * xi_cap
        assert 0.0 <= b["path_m"][1] - path_prev <= S.island_path_tol * xi_cap
        flow_prev, path_prev, cap_prev = b["flow_m"][1], b["path_m"][1], cap
    assert b["path_m"][1] > 0.0 and 0.0 < b["snap_rel"] <= 2.0 ** -40 * (1 + 1e-3)
    assert 0.0 < b["newton_rel"] <= S.island_tol_rel and b["dyn_rel"] > 0.0
    assert 0.0 < b["total_rel"] <= S.island_budget_rel and not b["curv_m"].any() and b["curv_rel"] == 0.0
    assert S.n_events["island_creep_steps"] == 10 and S.n_events["island_flow_m"] == b["flow_m"][1]
    assert S.n_events["island_bound_xi"] == b["total_rel"]
    assert b["resid_m"][1] > 0.0 and b["dyn_m"][1] > 0.0
    assert b["total_rel"] >= (b["path_m"][1] + b["resid_m"][1] + b["dyn_m"][1]) / xi_cap * (1 - 1e-9)
    assert S.n_events["island_newton_rel"] == b["newton_rel"] and S.n_events["island_dyn_rel"] == b["dyn_rel"]
    S.island_flow_max = 1e-6
    _steps(S, 1)
    assert S.n_events["island_wake_flow"] == 1 and iid not in S._islands
    assert S.n_events["cluster_awake_certificate"] == 1 and not (S.regime[mem] == ASLEEP).any()


def test_island_wakes_when_a_near_pair_closes():
    """A sleeping pair 0.15 um from the -x wall carries the member-wall pair as an inactive contact of its system
    (island_gap_m = 2 d_cap, critique B1). With the committed positions moved along the floor (an exact equilibrium
    still: no contact of the island depends on the position along x) until that gap is half the step's drift
    allowance s_isl, as a creeping island's drift would leave it, the next step wakes the island into the DEM with
    island_wake_newcontact (a contact could form within the step), and its re-entry at the end of that step is
    refused for the same reason (cluster_awake_certificate)."""
    from dolle_extras import dem
    from dolle_extras.engine import ASLEEP
    S, (mem,), rows = _rows_on_floor([(R50 + 1.5e-7, 2)], g=(-2.0, 0.0, -1.0), roll=1.001)
    S._classify(0.0)
    iid = S._island_of[0]
    _steps(S, 1)
    isl = S._islands[iid]
    near = (isl["cj"] == -1) & (S._hist[isl["rows"], dem.H_TOUCH] < 0.5)
    assert near.sum() == 1 and (S.regime[mem] == ASLEEP).all()
    gap = S.x[0, 0] - S.rad[0]
    assert 1.4e-7 < gap < S.island_gap_m and isl["s_isl"] < 1e-11
    S.x[mem, 0] -= gap - 0.5 * isl["s_isl"]
    _steps(S, 1)
    assert S.n_events["island_wake_newcontact"] == 1 and iid not in S._islands
    assert S.n_events["cluster_awake_certificate"] == 1 and not (S.regime[mem] == ASLEEP).any()


def test_watch_joins_a_grain_within_the_drift_allowance():
    """A DEM grain next to a sleeping island whose members may drift by s_isl within the step (critique B2): at a
    gap of 2 d_cap + 1.5 s_isl it does not join and may move 0.5 s_isl until the next list build (no band in which it
    may move by nothing yet does not join); at 2 d_cap + 0.5 s_isl it would not have joined a static sleeper (its
    allowance is 0.5 s_isl then), but with the drift it joins and wakes the island (island_wake_approach), the
    island handed over at its committed state at rest."""
    from dolle_extras.engine import ACTIVE, ASLEEP
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001, extra=1)
    S._classify(0.0)
    iid = S._island_of[0]
    T = S._islands[iid]["T_ref"]
    s, d2, skin = 4e-8, 2 * S.d_cap, 1e-6
    S.regime[2] = ACTIVE
    x_c = S.x[mem].copy()
    for gap, drift, joins in ((d2 + 1.5 * s, s, False), (d2 + 0.5 * s, 0.0, False), (d2 + 0.5 * s, s, True)):
        S.x[2] = S.x[1] + [2 * R50 + gap, 0.0, 0.0]
        S._s_isl = np.zeros(S.N)
        S._s_isl[mem] = drift
        S._s_isl_max = drift
        ai, lim = S._watch(np.array([2]), T, skin)
        if not joins:
            assert ai.tolist() == [2] and iid in S._islands and (S.regime[mem] == ASLEEP).all()
            assert lim[0] == pytest.approx(0.5 * s, rel=1e-6)
    assert sorted(ai.tolist()) == [0, 1, 2] and iid not in S._islands
    assert all(S.regime[k] == ACTIVE for k in mem) and S.n_events["island_wake_approach"] == 1
    assert np.array_equal(S.x[mem], x_c) and not S.v[mem].any() and not S.w[mem].any()


def test_island_budget_wakes_after_repeated_small_excess():
    """At its rolling caps under a constant load the pair's bounded ratio exceeds 1 by a little every step (the
    deviation bound's Newton terms, well inside 1 + island_tol_rel, so tier 1), and each excess times the deflection
    at the cap is summed into the budget (critique A6). With island_budget_rel = 1e-6 the island sleeps, its total
    grows step by step, and it wakes with island_wake_budget at the first step that would take the total past 1e-6;
    the reported bound (island_bound_xi) never exceeds the budget. At the end of that step the cluster, at rest in
    the full model, is put to sleep again, re-placed exactly with the budget restarted."""
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001, island_budget_rel=1e-6)
    S._classify(0.0)
    iid = S._island_of[0]
    tot = [S._islands[iid]["bnd"]["total_rel"]]
    flow_tol = [0.0]
    while iid in S._islands and len(tot) < 100:
        _steps(S, 1)
        if iid in S._islands:
            b = S._islands[iid]["bnd"]
            assert S._islands[iid]["tier"] == 1
            tot.append(b["total_rel"])
            flow_tol.append(b["flow_tol_m"][1])
    assert S.n_events["island_wake_budget"] == 1 and len(tot) >= 5 and iid not in S._islands
    assert np.all(np.diff(tot) > 0) and np.all(np.diff(flow_tol) > 0) and tot[-1] <= 1e-6
    assert S.n_events["island_bound_xi"] == tot[-1]
    isl = S._islands[S._island_of[0]]
    assert isl["n_steps"] == 0 and isl["bnd"]["total_rel"] == 0.0 and S.n_events["cluster_sleep"] == 2


def test_island_at_a_load_breakpoint_is_committed_at_the_pre_jump_equilibrium():
    """On the flight profile, a pair at its rolling caps sleeps 10 ms before burnout of the first stage (T+5.2 s:
    25 g pressing, then -3 g, a breakpoint). The step ending at the breakpoint is certified against the load's limit
    from the left (critique A2), so the committed state is the pre-jump equilibrium: its reference load is the
    left-limit load, against which its residual is within the Newton tolerance, while against the post-jump load it
    is that of the whole jump. The next step starts at the breakpoint: the island wakes (island_wake_fast) and the
    DEM starts from the committed state."""
    from dolle_extras import dem, statics
    from dolle_extras.engine import ASLEEP
    prof = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())["flight_profile"]
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], roll=1.02, env={"flight_profile": prof, "start_time_s": 5.19})
    bp = S.loads.next_breakpoint(S.env.t0_flight)
    assert bp == pytest.approx(5.2, abs=1e-12)
    S._classify(0.0)
    iid = S._island_of[0]
    assert S.n_events["cluster_sleep_cap"] == 1
    for _ in range(20):
        _steps(S, 1)
        if abs(S._islands[iid]["T_ref"] - bp) <= 1e-9:
            break
    isl = S._islands[iid]
    assert abs(isl["T_ref"] - bp) <= 1e-9 and (S.regime[mem] == ASLEEP).all()
    F_pre, F_post = S._island_load(mem, isl["T_ref"], "-"), S._island_load(mem, isl["T_ref"])
    assert np.allclose(isl["F_ref"], F_pre, rtol=0.0, atol=1e-12 * np.abs(F_pre).max())   # (the spin terms use x)
    x = np.ascontiguousarray(S.x[mem])
    h = np.ascontiguousarray(S._hist[isl["rows"]])
    Fnet = statics.state(x, isl["ci"], isl["cj"], isl["prm_el"], h, S.L)[1]
    tt, tr = S._island_tols(isl["capv"], isl["prm"], h[:, dem.H_TOUCH] > 0.5)
    r_pre, r_post = (F_pre + Fnet).reshape(-1, 6), (F_post + Fnet).reshape(-1, 6)
    assert np.abs(r_pre[:, :3]).max() <= tt and np.abs(r_pre[:, 3:]).max() <= tr
    assert np.abs(r_post[:, 2]).min() >= 0.99 * 28 * 9.80665 * S.m[mem].min()
    x_c = S.x.copy()
    t = S.t
    S._step_once(t, S._outer_dt(t), set(), 1.0)
    assert S.n_events["island_wake_fast"] == 1 and np.array_equal(S._chk["x_seen"], x_c)


def test_refused_island_entry_leaves_the_state_untouched():
    """An entry attempt works on candidate arrays (critique D1): refused by the ratchet test (island_ratchet_rel =
    0, a residual speed of 1 nm/s) or by the first step's certificate after the candidate was placed and advanced
    (island_budget_rel = 0), the cluster's positions, velocities, spins and histories are bit-identical to what they
    were, it stays in the full model and the reason is counted."""
    from dolle_extras.engine import ACTIVE
    for cc, why in (({"island_ratchet_rel": 0.0}, "ratchet"), ({"island_budget_rel": 0.0}, "certificate")):
        S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001, **cc)
        S.v[0] = [1e-9, 0.0, 0.0]
        nr = len(S._ckeys)
        before = (S.x.copy(), S.v.copy(), S.w.copy(), S._hist[:nr].copy())
        assert not S._try_sleep_island(mem, S._touching(), float(S.env.t0_flight), S._outer_dt(0.0))
        assert S.n_events["cluster_awake_" + why] == 1 and not S._islands and (S.regime[mem] == ACTIVE).all()
        assert all(np.array_equal(a, b) for a, b in zip(before, (S.x, S.v, S.w, S._hist[:nr])))


def test_island_entry_certificate_is_not_counted_as_a_step():
    """The entry's first-step certificate (critique D2) is not a sleeping step: after a successful entry, static at
    the caps or creeping, island_static_steps, island_creep_steps, island_newton_iters and island_refactor are 0;
    the first step counts one step."""
    keys = ("island_static_steps", "island_creep_steps", "island_newton_iters", "island_refactor")
    for dg, tier, step_key in (((0.0, 0.0, 0.0), 1, "island_static_steps"), ((0.0, 0.0, 0.8), 2, "island_creep_steps")):
        S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001)
        _ramp(S, dg, 0.02)
        S._classify(0.0)
        assert S._islands[S._island_of[0]]["tier"] == tier and S.n_events["cluster_sleep"] == 1
        assert all(S.n_events[k] == 0 for k in keys)
        _steps(S, 1)
        assert S.n_events[step_key] == 1 and S.n_events["island_newton_iters"] > 0
        assert sum(S.n_events[k] for k in keys[:2]) == 1


def test_forced_redo_leaves_the_island_as_it_was():
    """A step attempt advances a sleeping island only into a pending state; a redo restores the snapshot, which
    shares the island dict (replaced at commits, never mutated): after an ordinary attempt and after an attempt with a
    member forced into the DEM (the island woken), _restore leaves the island dict the same object with the same
    contents, the positions and histories bit-identical and nothing pending; the step then commits once."""
    from dolle_extras.engine import ACTIVE, ASLEEP
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001)
    S._classify(0.0)
    _steps(S, 1)
    iid = S._island_of[0]
    d0 = S._islands[iid]
    ref = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d0.items()
           if isinstance(v, (np.ndarray, int, float))}
    nr = len(S._ckeys)
    x0, h0, ev0 = S.x.copy(), S._hist[:nr].copy(), dict(S.n_events)
    t = S.t
    H = S._outer_dt(t)
    snap = S._snapshot()
    for force in (set(), {int(mem[0])}):
        S._step_once(t, H, force, 1.0)
        if force:
            assert iid not in S._islands and all(S.regime[k] == ACTIVE for k in mem)
        else:
            assert iid in S._isl_pending and S._islands[iid] is d0
        S._restore(snap)
        assert S._islands[iid] is d0 and S._isl_pending == {} and (S.regime[mem] == ASLEEP).all()
        assert all(np.array_equal(d0[k], v) if isinstance(v, np.ndarray) else d0[k] == v for k, v in ref.items())
        assert np.array_equal(S.x, x0) and np.array_equal(S._hist[:nr], h0) and S.n_events == ev0
    S.step(t, H)
    assert S._islands[iid]["n_steps"] == d0["n_steps"] + 1
    assert S.n_events["island_static_steps"] == ev0["island_static_steps"] + 1


def test_checkpoint_with_island_at_its_cap_is_exact():
    """A state holding an island at its rolling caps (its factor is not picklable and is rebuilt on loading) saves,
    loads and continues bit-for-bit as the run kept in memory, through skips in which the island is advanced from
    sample to sample (_skip_asleep)."""
    import pickle
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001)
    S._classify(0.0)
    _steps(S, 3)
    L = pickle.loads(pickle.dumps(S, protocol=pickle.HIGHEST_PROTOCOL))
    assert all(isl.get("lu") is None for isl in L._islands.values())
    end = S.t + 0.03
    for sim in (S, L):
        sim.duration = end
        sim.run(progress=False)
    nr = len(S._ckeys)
    assert all(np.array_equal(getattr(S, a), getattr(L, a)) for a in ("x", "v", "w", "regime"))
    assert np.array_equal(S._hist[:nr], L._hist[:nr]) and S.n_events == L.n_events
    assert S.n_events["skipped_s"] > 0 and S.n_events["island_static_steps"] > 3
    (a,), (b,) = S._islands.values(), L._islands.values()
    assert a["n_steps"] == b["n_steps"] and np.array_equal(a["lam"], b["lam"]) and a["bnd"]["total_rel"] == \
        b["bnd"]["total_rel"]


def test_island_of_the_previous_format_is_upgraded_and_advances():
    """An island saved by the previous code version (critique D8: the v0.6 dict, its touching contacts only, tangents
    with the real kernel; no tier, bounds, lifted parameters, row norms, spring constants, caps, drift allowance or
    near pairs) is upgraded at its first step: the near pair with the wall is back in its system, it is tier 1 with
    a finite ratchet bound, and it advances, static at its caps (positions within 1e-15 m; the first Newton solve
    absorbs the old placement's residual, every spring force within island_tol_rel of its cap)."""
    from dolle_extras import dem, statics
    from dolle_extras.engine import ASLEEP
    S, (mem,), rows = _rows_on_floor([(R50 + 1.5e-7, 2)], g=(-2.0, 0.0, -1.0), roll=1.001)
    S._classify(0.0)
    iid = S._island_of[0]
    new = S._islands[iid]
    tm = S._hist[new["rows"], dem.H_TOUCH] > 0.5
    assert (~tm).sum() == 1
    ci, cj, rw = new["ci"][tm], new["cj"][tm], new["rows"][tm]
    x, prm, hist = np.ascontiguousarray(S.x[mem]), np.ascontiguousarray(S._prm[rw]), np.ascontiguousarray(S._hist[rw])
    K, J = statics.jacobians(x, S.rad[mem], ci, cj, prm, hist, S.L)
    lam, _, out = statics.state(x, ci, cj, prm, hist, S.L)
    S._islands[iid] = dict(members=mem, ci=ci, cj=cj, rows=rw, prm=prm, out=out, lam=lam, J=J, K=K,
                           F_ref=new["F_ref"], T_ref=new["T_ref"])
    nr = len(S._ckeys)
    x0, h0 = S.x.copy(), S._hist[:nr].copy()
    _steps(S, 3)
    isl = S._islands[iid]
    assert isl["tier"] == 1 and isl["n_steps"] == 3 and len(isl["ci"]) == len(new["ci"])
    assert all(k in isl for k in ("bnd", "prm_el", "rn", "kc", "capv", "s_isl", "F_lr_ref", "r_ref"))
    assert np.isfinite(isl["bnd"]["ratchet_m"]).all() and (S.regime[mem] == ASLEEP).all()
    assert S.n_events["island_static_steps"] == 3 and np.abs(S.x - x0).max() <= 1e-15
    assert _spring_change(isl, S._hist[isl["rows"]], h0[isl["rows"]]) <= S.island_tol_rel


def test_skip_asleep_advances_islands_in_lockstep():
    """Everything asleep: _skip_asleep advances both islands sample by sample and commits each sample at once; when
    the second fails at sample q (its certificate made to fail there), the skip stops at the sample before, and both
    islands are committed at that time, q steps each, and counted q steps each (island_static_steps): the first
    island's advance over the failed sample is not committed, the stepping redoes that interval, so it is not a step
    slept (critique D2: a sleeping step is counted once), and none of its counters stays (review: its Newton
    iterations stayed counted): island_newton_iters is that of a skip of the q samples that passed, run alone."""
    S, (m0, m1), rows = _rows_on_floor([(0.25e-3, 2), (0.65e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001)
    S._classify(0.0)
    i0, i1 = S._island_of[0], S._island_of[2]
    assert i0 != i1 and S._island_of[1] == i0 and S._island_of[3] == i1
    T, dts, q = float(S.env.t0_flight), S.base.dt_base, 6
    Ts = T + dts * np.arange(1, 21)
    advance = S._island_advance

    def failing(isl, T0, T1, count=True):
        if isl is S._islands[i1] and T1 >= Ts[q] - 1e-12:
            return False, None, "budget"
        return advance(isl, T0, T1, count)
    S._island_advance = failing
    t_new = S._skip_asleep(0.0, 0.02)
    assert t_new == pytest.approx(q * dts, abs=1e-12)
    for iid in (i0, i1):
        assert S._islands[iid]["T_ref"] == Ts[q - 1] and S._islands[iid]["n_steps"] == q
    assert S.n_events["island_static_steps"] == 2 * q
    R, _, _ = _rows_on_floor([(0.25e-3, 2), (0.65e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001)
    R._classify(0.0)
    assert R._skip_asleep(0.0, (q + 0.5) * dts) == pytest.approx(q * dts, abs=1e-12)
    assert R.n_events["island_static_steps"] == 2 * q
    assert S.n_events["island_newton_iters"] == R.n_events["island_newton_iters"] > 0


def test_island_entry_hysteresis():
    """Against sleep/wake chatter an entry looks island_hyst_steps (20) steps ahead (critique D6, C3): a pair whose
    floor rolling springs carry 0.96 of their caps under a constant load across the row sleeps (tier 1); under a
    load growing so that the springs would reach their caps within 10 steps it is refused with cluster_awake_drift
    when creep_islands is off (its first step's certificate holds), and sleeps as a creep island (tier 2) when it is
    on."""
    S1, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(0.0, 1.0, -1.0))
    S1._classify(0.0)
    fl, r1, _, _ = _floor_rolling(S1, S1._islands[S1._island_of[0]])
    gy = 0.96 / float(r1[fl].max())
    dgy = (1 / 0.96 - 1) * gy * 2.0              # over 20 ms: the caps reached after 10 ms
    for creep, dg, tier in ((True, 0.0, 1), (False, dgy, None), (True, dgy, 2)):
        S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(0.0, gy, -1.0), creep_islands=creep)
        _ramp(S, (0.0, dg, 0.0), 0.02)
        ok = S._try_sleep_island(mem, S._touching(), float(S.env.t0_flight), S._outer_dt(0.0))
        if tier is None:
            assert not ok and S.n_events["cluster_awake_drift"] == 1 and S.n_events["cluster_awake_certificate"] == 0
            continue
        isl = S._islands[S._island_of[0]]
        fl, r, _, _ = _floor_rolling(S, isl)
        assert ok and isl["tier"] == tier and abs(float(r[fl].max()) - 0.96) <= 1e-3
        assert S.n_events["cluster_sleep_creep"] == (tier == 2) and S.n_events["cluster_sleep_cap"] == 0


def test_margins_chord_term_raises_the_ratios():
    """_margins with the loads' chord deviation dF (single grains on one wall: the entry = stay test of the
    critique's corrected criteria, at entry and at every step): the pressing load is lowered by dF, the tangential
    force raised by dF, the twisting and rolling torques by dF R, so every ratio rises, by exactly those amounts; on
    a constant load the chord deviation is 0 and nothing changes."""
    import math
    from dolle_extras import dem
    S = _scene(1, size=0.001)
    S.x[0] = [0.5e-3, 0.5e-3, R50]
    row = S._row(0, -5)
    p = S._prm[row]
    Fc, R = p[dem.P_FC], p[dem.P_REFF]
    idx, faces = np.array([0]), np.array([4])
    F = np.array([[2e-9, -1e-9, 0.3 * Fc]])                   # pulling off the floor at 0.3 of the pull-off force
    Tq = np.array([[1e-14, 2e-14, 3e-15]])
    k0 = S._margins(idx, faces, F, Tq)[6][:, 0]
    dF = 1e-10
    k1 = S._margins(idx, faces, F, Tq, np.array([dF]))[6][:, 0]
    assert np.all(k1 > k0)
    Fn = -0.3 * Fc - dF
    Fncrit = Fn + 2 * Fc
    ft, tt, tw = math.hypot(2e-9, -1e-9) + dF, math.hypot(1e-14, 2e-14) + dF * R, 3e-15 + dF * R
    assert k1 == pytest.approx([-Fn / Fc, ft / (p[dem.P_MUT] * Fncrit), (ft + tt / R) / (p[dem.P_MUR] * Fncrit),
                                tw / (p[dem.P_MUTW] * Fncrit)], rel=1e-12)
    T0 = float(S.env.t0_flight)
    assert S._chord_force(idx, T0, T0 + 1e-3)[0] == 0.0


def test_single_grain_sleeps_within_the_margin_and_rolls_at_its_cap():
    """A grain on one wall: its entry is its stay test (ratios <= 1 - margin with the chord term), no longer
    1 - 5 margin. At 0.95 of its rolling cap (constant lateral load) it falls asleep after one step and does not
    move; at 1.01 a single contact at its cap is a mechanism and it rolls (ROLL)."""
    from dolle_extras import dem
    from dolle_extras.engine import ACTIVE, ASLEEP, ROLL
    from dolle_sim.environment import G0
    for ratio, regime in ((0.95, ASLEEP), (1.01, ROLL)):
        S0 = _scene(1, size=0.001)
        p, m = S0._params(0, -5), S0.m[0]
        cap = p[dem.P_MUR] * (m * G0 + 2 * p[dem.P_FC])        # 1 g pressing; |f_r| = |F_t| for one contact
        S = _scene(1, size=0.001, g=(ratio * cap / m / G0, 0.0, -1.0))
        S.x[0] = [0.5e-3, 0.5e-3, R50]
        T0 = float(S.env.t0_flight)
        S._place_at_equilibrium(0, 4, S._row(0, -5), T0)
        S.regime[0] = ACTIVE
        F, Tq = S._body_force(np.array([0]), T0)
        assert S._margins(np.array([0]), np.array([4]), F, Tq)[0][0] == pytest.approx(ratio, abs=1e-3)
        S.base._build_neighbors()
        _steps(S, 1)
        x1 = S.x[0].copy()
        _steps(S, 2)
        assert S.regime[0] == regime
        if regime == ASLEEP:
            assert 0 in S._sleep_face and S.n_events["sleep"] == 1 and np.array_equal(S.x[0], x1)
        else:
            assert 0 in S._roll_face and 0 not in S._sleep_face


def test_cap_sleep_off_is_the_previous_entry_test():
    """cfg contact.cap_sleep = False runs the entry test of v0.6 (_try_sleep_island_linear): a pair at rest under
    1 g sleeps as a tier-0 island in the previous format (no bounds, lifted parameters or near pairs), placed by one
    linear-response displacement, counted cluster_sleep only, and stepped on the linear certificate (no
    island_static_steps); a pair whose floor rolling springs carry 0.95 of their caps is refused within 5 margins of
    a limit (cluster_awake_near_limit), where cap_sleep on puts it to sleep."""
    from scipy.sparse.linalg import splu
    from dolle_extras import statics
    from dolle_extras.engine import ASLEEP
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], cap_sleep=False)
    ci, cj, rw = S._island_system(mem, S._touching())
    x, prm, hist = np.ascontiguousarray(S.x[mem]), np.ascontiguousarray(S._prm[rw]), np.ascontiguousarray(S._hist[rw])
    K, J = statics.jacobians(x, S.rad[mem], ci, cj, prm, hist, S.L)
    Fnet = statics.state(x, ci, cj, prm, hist, S.L)[1]
    du = splu(K).solve(S._island_load(mem, float(S.env.t0_flight)) + Fnet)
    xn, hn = statics.apply_displacement(x, ci, cj, prm, hist, S.L, du.reshape(-1, 6))
    S._classify(0.0)
    isl = S._islands[S._island_of[0]]
    assert set(isl) == {"members", "ci", "cj", "rows", "prm", "out", "lam", "J", "K", "lu", "F_ref", "T_ref", "tier"}
    assert isl["tier"] == 0 and np.array_equal(S.x[mem], xn) and np.array_equal(S._hist[rw], hn)
    assert S.n_events["cluster_sleep"] == 1 and S.n_events["cluster_sleep_cap"] == 0
    _steps(S, 3)
    assert (S.regime[mem] == ASLEEP).all() and S.n_events["island_static_steps"] == 0
    for cap_sleep in (False, True):
        S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], roll=0.95, cap_sleep=cap_sleep, slaved_clusters=False)
        S._classify(0.0)
        assert bool(S._islands) == cap_sleep
        assert S.n_events.get("cluster_awake_near_limit", 0) == (not cap_sleep)


def test_sleeping_island_wakes_when_it_could_touch_another_sleeper():
    """Two sleepers of different groups are compared with each other at every step (_sleeper_contacts): an island's
    own system lists only its member pairs and walls (_island_near), _watch only DEM grains, so before this test a
    drifting island could close a gap to a separate sleeping island or grain unseen (review: the gap closed and no
    contact formed). Under 1 g, a pair on the floor 0.2 nm from a second pair (two islands), and a pair 0.2 nm from a
    single grain on the floor (an island and a single sleeper): all asleep after the classification (the gap exceeds the
    first step's allowance, so the entries' neighbour test, _sleepers_near, lets them sleep); with the island's
    committed positions moved along the floor until the gap is half its drift allowance s_isl (an exact equilibrium
    still, as a creeping island's drift would leave it), the next step wakes both sides (island_wake_neighbour) and
    the full model takes the step from the committed state; at its end the one put to sleep first keeps the other
    awake (cluster_awake_neighbour or sleep_refused_neighbour) while they are that close."""
    from dolle_extras.engine import ASLEEP
    for groups, single in (([2, 2], False), ([2, 1], True)):
        S0, _, _ = _rows_on_floor([(0.25e-3, 2)])
        dp = 2 * R50 - float(S0.x[1, 0] - S0.x[0, 0])
        x1 = 0.25e-3 + (2 * R50 - dp) + 2 * R50 + 2e-10
        S, (m0, m1), rows = _rows_on_floor([(0.25e-3, groups[0]), (x1, groups[1])])
        S._classify(0.0)
        assert (S.regime == ASLEEP).all() and (2 in S._sleep_face) == single
        i0 = S._island_of[0]
        gap = float(np.linalg.norm(S.x[2] - S.x[1])) - 2 * R50
        s0 = S._islands[i0]["s_isl"]
        assert 1e-10 < gap < 3e-10 and s0 < 1e-11
        assert S._sleeper_contacts(np.full(S.N, 0.4 * gap)) == []
        assert len(S._sleeper_contacts(np.full(S.N, 0.6 * gap))) == 1
        S.x[m0, 0] += gap - 0.5 * s0
        _steps(S, 1)
        assert S.n_events["island_wake_neighbour"] == 1 and i0 not in S._islands
        assert S.n_events["wake"] == len(m0) + len(m1) and S.n_events.get("redos_missed_contact", 0) == 0
        assert not (S.regime == ASLEEP).all()
        refused = S.n_events["cluster_awake_neighbour"] + S.n_events.get("sleep_refused_neighbour", 0)
        assert refused >= 1


def test_single_sleeper_and_tier0_island_woken_at_a_breakpoint_start_from_the_pre_jump_equilibrium():
    """A single sleeper and a tier-0 island (cap_sleep off: the linear certificate and placement) are certified up to
    a load breakpoint with the load's limit from the left, so the step starting there wakes them (critique A2); they
    are then placed at the equilibrium of the load just before the jump, and the full model resolves the jump (a
    placement at the post-jump equilibrium dropped its transient: review, a grain at 0.95 of its rolling limit across
    the de-spin knot whose full model reached 0.99999 of it from the pre-jump state). Burnout of the first stage
    (T+5.2 s, 25 g pressing then -3 g) on the flight profile: the woken grain's position is that of
    _place_at_equilibrium under the left-limit load to 1e-15 m and > 1e-12 m from the post-jump one (2.1e-11 m: the
    normal overlap of 25 g against -3 g); the island's is
    the linear response to the left-limit load from its reference state, bit-identical; at the run's start (placed
    there, nothing before it) a wake uses the load at T itself."""
    import copy
    from dolle_extras import statics
    from dolle_extras.engine import ASLEEP
    prof = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())["flight_profile"]
    env = {"flight_profile": prof, "start_time_s": 5.19}
    S, _, rows = _rows_on_floor([(0.45e-3, 1), (0.75e-3, 1)], env=env)
    bp = S.loads.next_breakpoint(S.env.t0_flight)
    assert bp == pytest.approx(5.2, abs=1e-12) and S._jump_side(S.env.t0_flight, S.env.t0_flight) is None
    _steps(S, 1)
    assert S.regime[0] == ASLEEP and 0 in S._sleep_face
    while S.env.t0_flight + S.t < bp - 1e-9:
        _steps(S, 1)
    T = float(S.env.t0_flight + S.t)
    assert abs(T - bp) <= 1e-9 and S.regime[0] == ASLEEP and S._jump_side(T, S.env.t0_flight) == "-"
    face, row = S._sleep_face[0], S._key[(0, -5)]
    ref = {}
    for side in ("-", None):
        D = copy.deepcopy(S)
        F, Tq = D._body_force(np.array([0]), T, side)
        D._place_at_equilibrium(0, face, row, T, F, Tq)
        ref[side] = D.x[0].copy()
    S._wake(0, T)
    assert np.abs(S.x[0] - ref["-"]).max() <= 1e-15 and np.abs(S.x[0] - ref[None]).max() > 1e-12
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], env=env, cap_sleep=False)
    S._classify(0.0)
    iid = S._island_of[0]
    isl = S._islands[iid]
    assert isl["tier"] == 0
    while S.env.t0_flight + S.t < bp - 1e-9:
        _steps(S, 1)
    T = float(S.env.t0_flight + S.t)
    assert abs(T - bp) <= 1e-9 and (S.regime[mem] == ASLEEP).all() and S._islands[iid] is isl
    ref = {}
    for side in ("-", None):
        du = S._island_lu(isl).solve(S._island_load(mem, T, side) - isl["F_ref"]).reshape(-1, 6)
        ref[side] = statics.apply_displacement(np.ascontiguousarray(S.x[mem]), isl["ci"], isl["cj"], isl["prm"],
                                               np.ascontiguousarray(S._hist[isl["rows"]]), S.L, du)[0]
    S._wake_island(iid, T)
    assert np.array_equal(S.x[mem], ref["-"]) and np.abs(S.x[mem] - ref[None]).max() > 1e-12


def test_entry_ratchet_releases_every_component_it_finds_near_its_cap():
    """_island_ratchet iterates its released system to a fixed point (review): one grain whose rotation is held by a
    rolling spring i at its cap, a second rolling spring j at 0.9796 of its cap and a soft stiffness k_o = 0.01 k.
    The elastic bound flags only i; the bound with i released flags j too (0.9796 + 5e-4 >= 1 - margin); released
    together, j adds no stiffness either, and the slip bound of both is sqrt((k_j + k_o) / k_o) = 10x the one-pass
    value: 5.0e-3 of the deflection at the cap, which the entry refuses (> island_ratchet_rel = 1e-3), where the
    one-pass bound (5.0e-4) let it sleep. The bound returned equals statics.energy_bounds in the system with both
    released, to rounding."""
    from scipy.sparse import csc_matrix, csr_matrix
    from dolle_extras import dem, statics
    from dolle_extras.engine import FullContactSim as F

    class Stub:
        margin = 0.02
        _ratios, _kind_norms, _fhat = staticmethod(F._ratios), staticmethod(F._kind_norms), staticmethod(F._fhat)
        _island_dashpots = staticmethod(F._island_dashpots)
        m, I, rad = np.ones(1), np.ones(1), np.ones(1)

    ki, kj, ko = 1.0, 1.0, 0.01
    K = csc_matrix(np.diag([1e3, 1e3, 1e3, ki + kj + ko, 1e3, 1e3]))
    Jd = np.zeros((16, 6))
    Jd[4, 3], Jd[12, 3] = -ki, -kj
    J = csr_matrix(Jd)
    prm = np.zeros((2, dem.NP))
    prm[:, dem.P_MUT], prm[:, dem.P_MUTW], prm[:, dem.P_MUR] = 1e6, 1e6, 1.0
    hist = np.zeros((2, dem.NH))
    hist[:, dem.H_TOUCH] = 1.0
    lam = np.zeros(16)
    lam[0] = lam[8] = 1.0                                    # rolling caps mu_r |F_ne| = 1
    lam[4], lam[12] = 1.0, 0.9796
    out = np.zeros((2, dem.NOUT))
    kc = np.array([[1.0, ki, 1.0], [1.0, kj, 1.0]])
    capv = statics.caps(lam, prm, out)
    E = (5e-4) ** 2 * (kj + ko) / 2
    with np.errstate(divide="ignore", invalid="ignore"):        # (no pull-off force in the toy: 0/0 ratios)
        dl, slip, near = F._island_ratchet(Stub(), K, J, prm, hist, lam, out, kc, capv, E, np.array([0]))
    assert near[:, 1].tolist() == [True, True]
    both = np.zeros((2, 3), bool)
    both[:, 1] = True
    d_both = F._kind_norms(statics.energy_bounds(statics.released_stiffness(K, J, kc, both, F._fhat(lam)), J, E))
    assert np.allclose(slip[:, 1], d_both[:, 1] / kc[:, 1], rtol=1e-12)
    rel = slip[:, 1] / (capv[:, 1] / kc[:, 1])
    assert np.allclose(rel, 5e-4 * np.sqrt((kj + ko) / ko), rtol=1e-6) and (rel > 1e-3).all()


def test_creep_step_charges_an_excursion_beyond_the_chord_through_the_released_structure_and_the_caps():
    """A creep step's curvature charge (curv_m: how far a load excursion beyond the chord inside the step can flow a
    capped spring) is taken through the released structure, including the cap's change with the normal force: on
    check 16's 5-grain corner pile at its rolling limits (coast load), with an in-step deviation of 0.01 g along z
    added to Loads.chord_dev (ASSUMED, a test load), one creep step's rolling charge covers the rolling flow that an
    excursion of the chord's size causes along each of +-x, +-y, +-z from the committed state (the real kernel, its
    creep tangent; measured flow / charge <= 0.59), while the previous charge, the elastic own-kind row norms, is
    exceeded by the excursion that lowers the pressing load (6.9x, review)."""
    import copy
    from scipy.sparse.linalg import splu
    from dolle_extras import dem, statics
    from dolle_extras import verify_full as V
    G0 = 9.80665
    P, _ = V._cap_pile_prep("coast", (-8.0, 0.0, -2.0))
    S = copy.deepcopy(P)
    S.all_dem = False
    S.duration = S.t + 1.0
    T0 = float(S.env.t0_flight + S.t)

    def coast(T):
        return np.column_stack([np.full(len(T), -8 * G0), np.zeros(len(T)),
                                -(2.0 - 1.8 * np.clip((T - T0) / 0.03, 0.0, 1.0)) * G0])
    V._cap_loads(S, coast, lambda a, b: np.array([0.0, 0.0, 0.01 * G0]))
    mem = np.array([2, 3, 4, 5, 6])
    H = S._outer_dt(S.t)
    assert S._try_sleep_island(mem, list(V._cap_touching(S)), T0, H)
    isl = S._islands[S._island_of[2]]
    ok, new, _ = S._island_advance(isl, T0, T0 + H)
    assert ok and isl["tier"] == 2 and new["tier"] == 2
    charge = float(new["bnd"]["curv_m"][1] - isl["bnd"]["curv_m"][1])
    chord = S._island_chord(mem, T0, T0 + H)
    x, h = np.ascontiguousarray(S.x[mem]), np.ascontiguousarray(S._hist[isl["rows"]])
    prm, pe, ci, cj = isl["prm"], isl["prm_el"], isl["ci"], isl["cj"]
    K, J = statics.jacobians(x, S.rad[mem], ci, cj, pe, h, S.L)
    lu = splu(K)
    tm = h[:, dem.H_TOUCH] > 0.5
    lam, Fnet, out = statics.state(x, ci, cj, pe, h, S.L)
    tt, tr = S._island_tols(statics.caps(lam, prm, out), prm, tm)
    tan = S._island_tangent(dict(K=K, J=J, prm=prm, lu=lu), False)
    old = float((S._kind_norms(S._island_row_norms(lu, J)[:, 0])[tm] * chord / statics.kcomp(prm, h)[tm]).max(0)[1])
    F0 = S._island_load(mem, T0)
    flows = []
    for dv in np.vstack([np.eye(3), -np.eye(3)]):
        dF = np.zeros((len(mem), 6))
        dF[:, :3] = S.m[mem][:, None] * dv[None, :]
        dF = (dF / np.linalg.norm(dF) * chord).reshape(-1)
        C = statics.solve_equilibrium(x, ci, cj, prm, pe, h, S.L, lu, F0 + dF, lu.solve(F0 + Fnet + dF), tt, tr, 40,
                                      tangent=tan)
        assert C["ok"]
        hn_el = statics.apply_displacement(x, ci, cj, pe, h, S.L, C["du"].reshape(-1, 6))[1]
        flows.append(float(statics.flow_increments(hn_el, C["hn"])[tm].max(0)[1]))
    assert 0.0 < max(flows) <= charge and max(flows) > 3 * old


def test_entry_dropping_a_residual_keeps_the_one_percent_test_for_contacts_far_from_their_caps():
    """When a component is near its cap the entry drops the left-over motion (its slip there is bounded by the
    released-system test); every contact with no component near its cap must still pass the previous version's test,
    the residual's largest effect on its load within 1% of its remaining margin (review: it was not applied, so a
    residual that failed it was dropped unreported). Check 16's 5-grain corner pile at its rolling limits (coast,
    exact equilibrium) with kinetic energy given to the top grain: at 1e-20 J it sleeps; at 5e-20 J the slip test
    holds (slip <= island_ratchet_rel of the deflection at the cap for every near component) but a far contact fails
    the 1% test, so it stays awake (cluster_awake_ratchet), its state untouched and _linres not tried."""
    import copy
    import math
    from dolle_extras import verify_full as V
    P, _ = V._cap_pile_prep("coast", (-8.0, 0.0, -2.0))
    mem = np.array([2, 3, 4, 5, 6])
    for E, sleeps in ((1e-20, True), (5e-20, False)):
        D = copy.deepcopy(P)
        D.all_dem = False
        D.duration = D.t + 1.0
        D.v[6] = [math.sqrt(2 * E / D.m[6]), 0.0, 0.0]
        Q = V._cap_system(D, mem)
        _, slip, near = D._island_ratchet(Q["K"], Q["J"], Q["prm"], Q["h"], Q["lam"], Q["out"], Q["kc"], Q["capv"], E,
                                          mem)
        assert near.any() and (slip[near] <= D.island_ratchet_rel * (Q["capv"] / Q["kc"])[near]).all()
        lin = []
        D._linres = lambda *a: lin.append(1)
        nr = len(D._ckeys)
        before = (D.x.copy(), D.v.copy(), D._hist[:nr].copy())
        ok = D._try_sleep_island(mem, list(V._cap_touching(D)), float(D.env.t0_flight + D.t), D._outer_dt(D.t))
        assert ok == sleeps and not lin
        if not sleeps:
            assert D.n_events["cluster_awake_ratchet"] == 1 and not D._islands
            assert all(np.array_equal(a, b) for a, b in zip(before, (D.x, D.v, D._hist[:nr])))


def test_island_of_the_previous_format_is_upgraded_at_the_load_of_its_first_step():
    """An island saved by the previous code version sits where its linear placement left it at its sleep time T_ref,
    the equilibrium of the load then; the upgrade at a later T0 places it at the exact equilibrium of the load at T0,
    with T_ref = T0 and F_ref = F(T0) (review: keeping the old ones made a wake in the first step hand the DEM the
    equilibrium of a load left long ago). The pair sleeps, its island is put in the previous format and stepped by the
    previous version's linear certificate (cap_sleep off) while the pressing load grows by 1 g over 20 ms; at T0 =
    10 ms the upgrade: its residual against F(T0) is within the Newton tolerances, it is within 1e-14 m of the
    previous version's own placement at T0 (the linear response) and > 1e-13 m from where it was; a wake at T0 hands
    exactly that state over with no fallback (island_wake_fallback 0)."""
    from dolle_extras import dem, statics
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(-2.0, 0.0, -1.0))
    _ramp(S, (0.0, 0.0, -1.0), 0.02)
    S._classify(0.0)
    iid = S._island_of[0]
    new = S._islands[iid]
    tm = S._hist[new["rows"], dem.H_TOUCH] > 0.5
    ci, cj, rw = new["ci"][tm], new["cj"][tm], new["rows"][tm]
    x, prm, hist = np.ascontiguousarray(S.x[mem]), np.ascontiguousarray(S._prm[rw]), np.ascontiguousarray(S._hist[rw])
    K, J = statics.jacobians(x, S.rad[mem], ci, cj, prm, hist, S.L)
    lam, _, out = statics.state(x, ci, cj, prm, hist, S.L)
    S._islands[iid] = old = dict(members=mem, ci=ci, cj=cj, rows=rw, prm=prm, out=out, lam=lam, J=J, K=K,
                                 F_ref=new["F_ref"], T_ref=new["T_ref"])
    S.cap_sleep = False
    _steps(S, 10)
    S.cap_sleep = True
    assert S._islands[iid] is old
    T0 = float(S.env.t0_flight + S.t)
    x_old = S.x[mem].copy()
    du = S._island_lu(old).solve(S._island_load(mem, T0) - old["F_ref"]).reshape(-1, 6)
    x_lin = statics.apply_displacement(np.ascontiguousarray(x_old), ci, cj, prm, np.ascontiguousarray(S._hist[rw]),
                                       S.L, du)[0]
    up = S._island_upgrade(old, T0)
    assert up["tier"] in (1, 2) and up["T_ref"] == T0 and np.array_equal(up["F_ref"], S._island_load(mem, T0))
    xu = np.ascontiguousarray(S.x[mem])
    h = np.ascontiguousarray(S._hist[up["rows"]])
    r = (up["F_ref"] + statics.state(xu, up["ci"], up["cj"], up["prm_el"], h, S.L)[1]).reshape(-1, 6)
    tt, tr = S._island_tols(up["capv"], up["prm"], h[:, dem.H_TOUCH] > 0.5)
    assert np.abs(r[:, :3]).max() <= tt and np.abs(r[:, 3:]).max() <= tr
    assert np.abs(xu - x_lin).max() <= 1e-14 and np.abs(xu - x_old).max() > 1e-13
    S._islands[iid] = up
    S._wake_island(iid, T0)
    assert np.array_equal(S.x[mem], xu) and S.n_events["island_wake_fallback"] == 0


def test_drift_allowance_covers_an_out_and_back_load_through_a_stack():
    """s_isl, the most a sleeping member may move inside a step (every other grain's motion bounds use it), covers a
    load that goes out and back within the step and the chain of deflections down a stack (review: the end
    displacement plus the largest single contact deflection missed it by 2-5x). Three grains stacked on the floor
    under 1 g, asleep (tier 1); a lateral half-sine of 1e-3 g over exactly one step (zero at both ends, Loads.chord_dev
    given its exact deviation from the chord, (pi/l)^2 l^2/8 times the peak); every member's largest displacement
    on the exact quasi-static path (the elastic equilibrium at 41 times in the step, every real sliding, rolling and
    twisting ratio far below 1) is within s_isl."""
    import math
    from scipy.sparse.linalg import splu
    from dolle_extras import dem, statics
    from dolle_extras import verify_full as V
    from dolle_extras.engine import ACTIVE, ASLEEP, FullContactSim
    from dolle_sim.environment import G0
    cfg = {"dt_s": 1e-3, "duration_s": 0.3, "output_fps": 100,
           "environment": {"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": [0.0, 0.0, -1.0]},
           "contact": {"contact_charging": False}, "forces": {}}
    ch = {"name": "stack", "size_m": [0.001] * 3, "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": 3, "radius_m": R50, "velocity": {"speed_mean_m_s": 0.0},
                         "initial_state": {"type": "free"}}]}
    S2 = FullContactSim(cfg, ch, 0)
    S2.base.F_static[:] = 0.0
    S2.v[:], S2.w[:] = 0.0, 0.0
    dw, dp = (p[dem.P_A0] ** 2 / (3 * p[dem.P_REFF]) for p in (S2._params(0, -5), S2._params(0, 1)))
    mem = np.arange(3)
    for q in range(3):
        S2.x[q] = [0.5e-3, 0.5e-3, R50 - dw + q * (2 * R50 - dp)]
    keys = [(0, -5), (0, 1), (1, 2)]
    rw = np.array([S2._row(i, j) for i, j in keys])
    ci, cj = np.array([i for i, _ in keys]), np.array([j for _, j in keys])
    x, prm = np.ascontiguousarray(S2.x[mem]), np.ascontiguousarray(S2._prm[rw])
    pe = statics.lift_caps(prm)
    _, h = statics.apply_displacement(x, ci, cj, prm, np.zeros((len(rw), dem.NH)), S2.L, np.zeros((3, 6)))
    K, J = statics.jacobians(x, S2.rad[mem], ci, cj, pe, h, S2.L)
    lu = splu(K)
    lam, Fnet, out = statics.state(x, ci, cj, pe, h, S2.L)
    tt, tr = S2._island_tols(statics.caps(lam, prm, out), prm, np.ones(len(rw), bool))
    F = S2._island_load(mem, float(S2.env.t0_flight))
    E = statics.solve_equilibrium(x, ci, cj, pe, pe, h, S2.L, lu, F, lu.solve(F + Fnet), tt, tr, 20)
    assert E["ok"]
    S2.x[mem], S2._hist[rw] = E["xn"], E["hn"]
    S2.regime[mem] = ACTIVE
    S2.base._build_neighbors()
    S2._run_started = True
    S2._classify(0.0)
    assert (S2.regime[mem] == ASLEEP).all()
    isl = S2._islands[S2._island_of[0]]
    T0, l, A = float(isl["T_ref"]), 1e-3, 1e-3 * G0

    def fn(T):
        p = np.where((T >= T0) & (T < T0 + l), A * np.sin(math.pi * (T - T0) / l), 0.0)
        return np.column_stack([p, np.zeros(len(T)), np.full(len(T), -G0)])

    def curv(Ta, Tb):
        return np.array([A * (math.pi / l) ** 2 * (Tb - Ta) ** 2 / 8, 0.0, 0.0]) if Ta < T0 + l and Tb > T0 \
            else np.zeros(3)
    V._cap_loads(S2, fn, curv)
    ok, new, _ = S2._island_advance(isl, T0, T0 + l, count=False)
    assert ok and new["tier"] == 1
    x0, h0 = np.ascontiguousarray(S2.x[mem]), np.ascontiguousarray(S2._hist[isl["rows"]])
    tm = h0[:, dem.H_TOUCH] > 0.5
    tt, tr = S2._island_tols(isl["capv"], isl["prm"], tm)
    worst, ratio = 0.0, 0.0
    for t in np.linspace(T0, T0 + l, 41):
        F = S2._island_load(mem, t)
        E = statics.solve_equilibrium(x0, isl["ci"], isl["cj"], isl["prm_el"], isl["prm_el"], h0, S2.L,
                                      S2._island_lu(isl), F, S2._island_lu(isl).solve(F - isl["F_ref"]), tt, tr, 20)
        assert E["ok"]
        worst = max(worst, float(np.linalg.norm(E["xn"] - x0, axis=1).max()))
        ratio = max(ratio, float(S2._ratios(E["lam"], None, isl["prm"], E["out"], tm)[1].max()))
    assert worst > 1e-10 and ratio < 0.1 and worst <= new["s_isl"]


def test_mid_step_wake_tests_its_placement_against_this_steps_allowance():
    """A cluster woken inside a step (a DEM grain approaching, _watch) is placed by the Newton solve to the load then
    if that moves no member by more than the drift allowance every motion bound of the step used, this step's s_isl
    (_step_once), not the committed dict's, the previous step's (review). The creeping pair under an extra -2 g along
    the row ramped in over 2 ms: in the first step after entry its allowance grows from the entry's 1e-12 m; woken at
    0.9 of that step, its placement moves it by more than 1e-12 m but within this step's allowance, and is used (no
    island_wake_fallback)."""
    from dolle_extras.engine import ASLEEP
    S, (mem,), rows = _rows_on_floor([(0.45e-3, 2)], g=(-2.0, 0.0, -1.0), roll=1.001)
    _ramp(S, (-2.0, 0.0, 0.0), 0.002)
    S._classify(0.0)
    iid = S._island_of[0]
    isl = S._islands[iid]
    assert isl["s_isl"] == 1e-12
    t = S.t
    H = S._outer_dt(t)
    T = float(S.env.t0_flight + t)
    ok, new, _ = S._island_advance(isl, T, T + H, count=False)
    assert ok and new["s_isl"] > 2e-12
    S._s_isl = np.zeros(S.N)
    S._s_isl[mem] = new["s_isl"]
    S._s_isl_max = new["s_isl"]
    x0 = S.x[mem].copy()
    S._wake_island(iid, T + 0.9 * H)
    moved = float(np.abs(S.x[mem] - x0).max())
    assert 1e-12 < moved <= new["s_isl"] and S.n_events["island_wake_fallback"] == 0
    assert not (S.regime[mem] == ASLEEP).any()

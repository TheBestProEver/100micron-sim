"""The 3-D rigid-clump simulation of dolle_core (clumps.py, impacts.py, sim3d.py, scenes.py): against analytic
results, against the laws it must obey, and against the full simulator's v0.5 chamber engine
(dolle_sim/simulation.py ChamberSim) on identical starts.

Each test runs in well under 30 s.  dolle_sim is imported only inside the helpers that compare with it.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from dolle_core import clumps as cl
from dolle_core import impacts, scenes, sim3d
from dolle_core.collide import impact, sticking_threshold
from dolle_core.materials import G0
from dolle_core.pair import make_pair

ROOT = Path(__file__).resolve().parents[1]
R = 50e-6
QUIET = sim3d.Tubule(accel=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------- helpers
def centre(tub=QUIET):
    return np.asarray(tub.size) / 2


def chain_clump(sim, offsets, b0=0):
    """Merge grains b0, b0+1, ... (already placed touching) into one rigid clump; returns its id."""
    c = sim.clumps
    A = b0
    for k in range(1, len(offsets)):
        A = cl.merge(c, int(c.body[b0]), int(c.body[b0 + k]))
    sim.P0, sim.L0 = cl.momentum(c)
    return int(c.body[b0])


OFF = ("coulomb", "polarization", "image_charges", "self_gravity", "lorentz", "external_e_field", "thermophoresis",
       "photophoresis", "radiation_pressure", "brownian", "space_charge")


def _chamber_sim():
    try:
        from dolle_sim.simulation import ChamberSim
    except ImportError as err:          # the restructure may move it; say so clearly
        raise ImportError("these comparison tests need the full simulator's v0.5 chamber engine "
                          "(dolle_sim.simulation.ChamberSim)") from err
    return ChamberSim


def v05_sim(tub, mats, radii, pos, vel, t_end, spins=None, adhesion=None):
    """A v0.5 ChamberSim on the same start: one particle group per grain, vacuum, no payload roll, every force and
    charging off, tubule settings from `tub` (mu must be a number), then pos, X, V, L (and adhesion) overwritten
    with the core's state."""
    ChamberSim = _chamber_sim()
    cfg = {"dt_s": tub.dt, "duration_s": t_end, "output_fps": 10,
           "environment": {"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": [a / G0 for a in tub.accel],
                           "gravity_gradient": False, "magnetic_field_t": [0, 0, 0], "spin_rate_hz": 0.0},
           "forces": {k: False for k in OFF},
           "contact": {"friction_coefficient": tub.mu, "tangential_restitution": tub.e_t, "contact_charging": False,
                       "fragmentation": tub.fragmentation, "wall_sticking": "jkr", "steps_per_contact": tub.n_sub}}
    groups = [{"material": m, "count": 1, "radius_m": float(r),
               "region_m": {"center": list(np.asarray(tub.size) / 2), "size": [0, 0, 0]},
               "velocity": {"mode": "jet", "direction": [1, 0, 0], "speed_mean_m_s": 0.0, "speed_sigma_m_s": 0.0}}
              for m, r in zip(mats, radii)]
    ch = {"name": "core", "size_m": list(tub.size), "wall_material": tub.wall, "wall_adhesion_factor": tub.f_wall,
          "particles": groups}
    s = ChamberSim(cfg, ch, seed=0)
    s.pos = np.array(pos, float).copy()
    s.X = s.pos.copy()
    s.V = np.array(vel, float).copy()
    if spins is not None:
        s.L = np.array(spins, float) * s.I0[:, None]
    if adhesion is not None:
        s.adh = np.array(adhesion, float).copy()
    return s


def both(tub, mats, radii, pos, vel, t_end, spins=None, prep=None, adhesion=None):
    """Run the core and v0.5 from the same start; prep(core_sim, v05_sim) may pre-form clumps first."""
    a = sim3d.make_sim(tub, mats, radii, pos, vel, spins=spins, adhesion=adhesion)
    b = v05_sim(tub, mats, radii, pos, vel, t_end, spins=spins, adhesion=adhesion)
    if prep:
        prep(a, b)
    a.P0, a.L0 = cl.momentum(a.clumps)
    sim3d.run(a, t_end, stop_when_settled=False)
    b.run(progress=False)
    return a, b


def same_events(a, b, n=None, tol_t=1e-12, tol_rel=1e-12):
    """Number of leading events with the same (kind, i, j, outcome), time to tol_t, v_n, e_n, m_eff to tol_rel."""
    ea, eb = a.events[:n], b.collisions[:n]
    k = 0
    for x, y in zip(ea, eb):
        key_x = (x["kind"], x["i"], x["j"], x["outcome"])
        key_y = (y["kind"], y["i"], y["j"], y["outcome"])
        if key_x != key_y or abs(x["t"] - y["t"]) > tol_t:
            break
        for f in ("v_n", "m_eff"):
            assert abs(x[f] - y[f]) <= tol_rel * abs(y[f]), (k, f, x[f], y[f])
        assert abs(x["e_n"] - y["e_n"]) <= tol_rel, (k, "e_n", x["e_n"], y["e_n"])
        k += 1
    return k


def cloud(n, box, seed, grid=True, mats="silica", speed=0.015, sigma=0.0075, tub=None):
    """A compact random cloud at the tubule centre: radii (on the 0.5 um grid if grid), positions, velocities."""
    rng = np.random.default_rng(seed)
    tub = tub or sim3d.Tubule(mu=0.3)
    r = scenes.sieve_radii(n, rng=rng)
    if grid:
        r = np.round(r * 2e6) / 2e6
    pos = scenes.puff_positions(r, centre(tub), box, rng)
    vel = scenes.isotropic_velocities(n, speed, sigma, rng)
    return tub, [mats] * n, r, pos, vel


# ============================================================================ analytic cases
def test_free_flight_is_an_exact_parabola():
    a = (1e-3, -2e-3, 0.5e-3)
    tub = sim3d.Tubule(size=(2.0, 2.0, 2.0), accel=a)
    x0, v0 = np.array([1.0, 1.0, 1.0]), np.array([1e-3, 2e-3, -1e-3])
    s = sim3d.make_sim(tub, "silica", [R], [x0], [v0])
    sim3d.run(s, 10.0, stop_when_settled=False)
    t = s.t
    exact = x0 + v0 * t + 0.5 * np.asarray(a) * t * t
    assert abs(t - 10.0) < 1e-12 and not s.events
    assert np.abs(s.clumps.pos[0] - exact).max() < 1e-12
    assert np.abs(s.clumps.V[0] - (v0 + np.asarray(a) * t)).max() < 1e-14     # 5000 kicks of rounding


def _trimer(material="graphite", shape="line"):
    c = centre()
    off = [[-2 * R, 0, 0], [0, 0, 0], [2 * R, 0, 0]] if shape == "line" else [[0, 0, 0], [2 * R, 0, 0], [2 * R, 2 * R, 0]]
    s = sim3d.make_sim(QUIET, material, [R] * 3, c + np.array(off), np.zeros((3, 3)))
    return s, chain_clump(s, off)


def test_free_rotation_of_a_trimer_keeps_L_and_energy():
    """v0.5 check rigid_body: a spinning linear trimer for 5 s."""
    s, b = _trimer()
    c = s.clumps
    w0 = 2 * math.pi * 5 * np.array([math.cos(0.65), math.sin(0.65), 0.0])
    c.L[b] = np.linalg.inv(c.Iinv[b]) @ w0
    E0, L0 = 0.5 * c.L[b] @ c.Iinv[b] @ c.L[b], c.L[b].copy()
    dE = []
    for _ in range(int(round(5.0 / s.tubule.dt))):
        sim3d.step(s)
        dE.append(0.5 * c.L[b] @ c.Iinv[b] @ c.L[b] / E0 - 1)
    dE = np.array(dE)
    q = len(dE) // 5
    assert np.linalg.norm(c.L[b] - L0) / np.linalg.norm(L0) < 1e-10
    assert np.abs(dE).max() < 1e-3 and abs(dE[-q:].mean() - dE[:q].mean()) < 1e-5
    # rigid: the members keep their distances
    d = np.linalg.norm(c.pos[0] - c.pos[2])
    assert abs(d - 4 * R) < 1e-15 * 1e3


def test_dimer_precesses_like_a_free_symmetric_top():
    """A dimer is a symmetric top (I1 = I2 = I_perp about its centre, I3 along its axis).  Torque-free, its axis
    e3 turns about the fixed L at the rate |L|/I_perp (Landau & Lifshitz, Mechanics, section 33).  The DLM error is
    second order in the step h."""
    def axis_error(h):
        c0 = centre()
        s = sim3d.make_sim(QUIET, "silica", [R, R], [c0 - [R, 0, 0], c0 + [R, 0, 0]], np.zeros((2, 3)))
        b = chain_clump(s, [0, 1])
        c = s.clumps
        I_perp = 2 * (c.I0[0] + c.m[0] * R * R)
        c.L[b] = np.array([0.3, 0.0, 1.0]) * I_perp * 2 * math.pi * 3
        e0 = (c.pos[1] - c.pos[0]) / (2 * R)
        T = 0.2
        for _ in range(int(round(T / h))):
            cl.rotate(c, np.array([b]), h)
        c.pos = c.X[c.body] + c.off
        Lh = c.L[b] / np.linalg.norm(c.L[b])
        th = np.linalg.norm(c.L[b]) / I_perp * T
        K = impacts.skew(Lh)
        Rm = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
        e = (c.pos[1] - c.pos[0]) / (2 * R)
        return np.linalg.norm(e - Rm @ e0)
    e1, e2 = axis_error(2e-3), axis_error(1e-3)
    assert e1 < 1e-3 and 3.5 < e1 / e2 < 4.5


def test_merge_and_split_conserve_momentum_and_angular_momentum():
    """v0.5 check conservation: three grains stick into one clump, then one grain is split off."""
    c0 = centre()
    rng = np.random.default_rng(1)
    s = sim3d.make_sim(QUIET, ["graphite", "silica", "graphite"], [R, 0.8 * R, 1.1 * R],
                       [c0, c0 + [1.8 * R, 0, 0], c0 + [0, 2.1 * R, 0]], rng.standard_normal((3, 3)) * 0.01,
                       spins=rng.standard_normal((3, 3)) * 100)
    c = s.clumps
    P0, L0 = cl.momentum(c)
    b = cl.merge(c, cl.merge(c, 0, 1), 2)
    P1, L1 = cl.momentum(c)
    direct = cl.inertia_tensor(c, [0, 1, 2], c.X[b])
    stored = c.Q[b] @ np.diag(c.Ip[b]) @ c.Q[b].T
    new = cl.detach(c, b, [[1]])
    P2, L2 = cl.momentum(c)
    for P, L in ((P1, L1), (P2, L2)):
        assert np.linalg.norm(P - P0) / np.linalg.norm(P0) < 1e-14
        assert np.linalg.norm(L - L0) / np.linalg.norm(L0) < 1e-14
    assert np.abs(direct - stored).max() / np.abs(direct).max() < 1e-14
    assert len(new) == 1 and c.n[b] == 2


@pytest.mark.parametrize("material,v", [("silica", 0.02), ("silica", 0.004), ("graphite", 0.06),
                                        ("graphite", 0.02), ("graphite", 1.5)])
def test_head_on_impact_equals_collide_impact(material, v):
    """Two free grains head-on in 3-D give the 1-D model's outcome, restitution and velocities (graphite at
    1.5 m/s: above its yield speed of 0.84 m/s, a dented impact)."""
    c0 = centre()
    s = sim3d.make_sim(QUIET, material, [R, R], [c0 - [1.01 * R, 0, 0], c0 + [1.01 * R, 0, 0]],
                       [[v / 2, 0, 0], [-v / 2, 0, 0]])
    sim3d.step(s)
    e = s.events[0]
    ref = impact(make_pair(material, R, material, R), v)
    assert ref.yielded == (v > 1)
    assert e["outcome"] == ref.outcome and abs(e["e_n"] - ref.e) < 1e-12 and e["yielded"] == ref.yielded
    if ref.outcome == "bounce":
        assert abs(s.clumps.V[0, 0] - ref.v1) < 1e-12 * v and abs(s.clumps.V[1, 0] - ref.v2) < 1e-12 * v
    else:
        assert s.clumps.n[s.clumps.body[0]] == 2 and np.abs(s.clumps.V[s.clumps.body[0]]).max() < 1e-18
    assert e["dP_rel"] < 1e-15 and e["dL_rel"] < 1e-15


def test_3d_sticking_speed_equals_sticking_threshold():
    """Bisection on the approach speed in the 3-D engine finds the 1-D model's sticking speed."""
    def sticks(v):
        c0 = centre()
        s = sim3d.make_sim(QUIET, "silica", [R, R], [c0 - [1.0001 * R, 0, 0], c0 + [1.0001 * R, 0, 0]],
                           [[v / 2, 0, 0], [-v / 2, 0, 0]])
        sim3d.step(s)
        return s.events[0]["outcome"] == "stick"
    lo, hi = 1e-3, 0.1
    while hi / lo - 1 > 2e-3:
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if sticks(mid) else (lo, mid)
    ref = sticking_threshold(make_pair("silica", R, "silica", R))
    assert abs(math.sqrt(lo * hi) / ref - 1) < 2e-3


def test_normal_impact_equals_dolle_sim_resolve():
    """normal_impact(pair, v, m_eff) is contact.py JKRContact(pair_props).resolve(v, m_eff): 200 random cases."""
    try:
        from dolle_sim.contact import JKRContact
        from dolle_sim.materials import get_material, pair_props
    except ImportError as err:
        raise ImportError("this test compares with dolle_sim.contact and dolle_sim.materials") from err
    rng = np.random.default_rng(5)
    worst = 0.0
    for k in range(200):
        g = ("silica", "graphite")[k % 2]
        wall = k % 4 >= 2
        other = "glass_borosilicate" if wall else g
        r1, r2 = rng.uniform(40e-6, 55e-6), (None if wall else rng.uniform(40e-6, 55e-6))
        pair = make_pair(g, r1, other, r2)
        ref = JKRContact(pair_props(get_material(g), r1, get_material(other), r2))
        m_eff = pair.m_star * rng.uniform(0.2, 1.0)
        v = math.exp(rng.uniform(math.log(1e-3), math.log(0.3)))
        a, b = impacts.normal_impact(pair, v, m_eff), ref.resolve(v, m_eff)
        assert a.stuck == b.stuck
        worst = max(worst, abs(a.e - b.e_n), abs(a.max_force - b.max_force) / max(b.max_force, 1e-30))
    assert worst < 1e-12


def _oblique_wall(mu, vn=0.05, vt=0.04):
    tub = sim3d.Tubule(accel=(0.0, 0.0, 0.0), mu=mu)
    s = sim3d.make_sim(tub, "silica", [R], [[0.006, 0.006, 1.001 * R]], [[vt, 0.0, -vn]])
    sim3d.run(s, 0.004, stop_when_settled=False)
    e = s.events[0]
    assert e["kind"] == "wall" and e["outcome"] == "bounce"
    return s, e


def test_oblique_wall_impact_rolls_with_high_friction():
    """High friction: the contact point stops slipping, so (angular momentum about the contact point kept)
    v_t' = (5/7) v_t and omega' = (5/7) v_t / r."""
    s, e = _oblique_wall(mu=10.0)
    c = s.clumps
    w = c.Iinv[0] @ c.L[0]
    assert abs(c.V[0, 0] / 0.04 - 5 / 7) < 1e-12
    assert abs(w[1] * R / 0.04 - 5 / 7) < 1e-12 and abs(c.V[0, 2] / 0.05 - e["e_n"]) < 1e-12


def test_oblique_wall_impact_slides_with_low_friction():
    """Low friction: the surfaces slide throughout, delta v_t = mu (1 + e_n) v_n."""
    s, e = _oblique_wall(mu=0.05)
    dvt = 0.04 - s.clumps.V[0, 0]
    assert abs(dvt / (0.05 * (1 + e["e_n"]) * 0.05) - 1) < 1e-12


def test_off_centre_stick_spins_the_dimer():
    """A graphite grain sticks to a resting one at impact parameter b: the dimer spins at omega = L/I_perp with
    L = (m/2) v b about the common centre of mass (equal masses)."""
    v, b = 0.01, 0.6 * 2 * R
    c0 = centre()
    x = math.sqrt((2 * R) ** 2 - b * b)
    s = sim3d.make_sim(QUIET, "graphite", [R, R], [c0 - [x + 1e-7, b, 0], c0], [[v, 0, 0], [0, 0, 0]])
    sim3d.run(s, 0.01, stop_when_settled=False)
    e = s.events[0]
    c = s.clumps
    B = c.body[0]
    assert e["outcome"] == "stick" and c.n[B] == 2
    m = c.m[0]
    Lz = 0.5 * m * v * b
    I_perp = 2 * (c.I0[0] + m * R * R)
    w = c.Iinv[B] @ c.L[B]
    assert abs(c.L[B][2] / Lz - 1) < 1e-12 and abs(w[2] / (Lz / I_perp) - 1) < 1e-9
    assert np.abs(w[:2]).max() < 1e-9 * abs(w[2])
    assert e["dP_rel"] < 1e-14 and e["dL_rel"] < 1e-14


def test_contact_time_from_the_quadratic_matches_bisection():
    rng = np.random.default_rng(2)
    for _ in range(100):
        c0 = centre()
        d = rng.standard_normal(3)
        d *= 6 * R / np.linalg.norm(d)
        u = -d / np.linalg.norm(d) * 0.02 + rng.standard_normal(3) * 0.004
        s = sim3d.make_sim(QUIET, "silica", [R, 0.9 * R], [c0, c0 + d], [[0, 0, 0], u])
        cand = np.array([[0, 1]])
        vm = cl.grain_velocities(s.clumps)
        ev = sim3d.next_contact(s, vm, 1.0, cand, np.zeros(1, bool), np.zeros((2, 3), bool))
        f = lambda t: np.linalg.norm(d + u * t) - 1.9 * R
        if ev is None or ev[1] != "pp":
            ts = np.linspace(0, 1, 20001)
            assert min(f(t) for t in ts[:2000]) > 0
            continue
        lo, hi = 0.0, ev[0] * 1.5 + 1e-9
        hi = next(t for t in np.linspace(0, hi, 2001) if f(t) <= 0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        assert abs(ev[0] - hi) < 1e-12


def test_spinning_dimer_contact_is_found_on_the_arc():
    """A fast-spinning dimer flies into a resting grain: the engine stops within 1e-3 r of the true contact."""
    worst = 0.0
    rng = np.random.default_rng(4)
    for trial in range(10):
        c0 = centre()
        s = sim3d.make_sim(QUIET, "silica", [R] * 3, [c0 - [R, 0, 0], c0 + [R, 0, 0], c0 + [0, 8 * R, 0]],
                           [[0, 0.02, 0], [0, 0.02, 0], [0, 0, 0]])
        b = chain_clump(s, [0, 1])
        s.clumps.L[b] = np.linalg.inv(s.clumps.Iinv[b]) @ (rng.standard_normal(3) * 2 * math.pi * 60)
        sim3d.run(s, 0.06, stop_when_settled=False)
        pp = [e for e in s.events if e["kind"] == "pp"]
        assert pp
        worst = max(worst, max(abs(e["gap"]) for e in pp))
    assert worst < 1e-3 * R


def test_bounce_impulse_on_a_trimer():
    """v0.5 checks impulse and friction_energy: -e v_n exactly, inside the friction cone, never an energy gain."""
    s, A = _trimer(shape="L")
    c = s.clumps
    rng = np.random.default_rng(0)
    m4 = make_pair("graphite", R, "graphite", R).m1
    worst_n = worst_cone = 0.0
    for _ in range(200):
        n = rng.standard_normal(3)
        n /= np.linalg.norm(n)
        rA = rng.standard_normal(3) * R
        vrel = rng.standard_normal(3) * 0.01
        vrel += (abs(vrel @ n) + 0.005 - vrel @ n) * n
        K = impacts.collision_matrix(c, A, rA) + np.eye(3) / m4
        e = rng.random()
        P, _ = impacts.bounce_impulse(K, vrel, n, e, 0.3)
        worst_n = max(worst_n, abs((vrel - K @ P) @ n + e * (vrel @ n)) / (vrel @ n))
        worst_cone = max(worst_cone, np.linalg.norm(P - (P @ n) * n) / (0.3 * (P @ n)) - 1)
    assert worst_n < 1e-9 and worst_cone < 1e-9
    rng = np.random.default_rng(3)
    worst = -math.inf
    for _ in range(2000):
        n = rng.standard_normal(3)
        n /= np.linalg.norm(n)
        rA = rng.standard_normal(3) * 2 * R
        vrel = rng.standard_normal(3) * 0.01
        vrel += (abs(vrel @ n) + 1e-4 - vrel @ n) * n
        K = impacts.collision_matrix(c, A, rA) + np.eye(3) / m4
        P, _ = impacts.bounce_impulse(K, vrel, n, rng.random(), 0.3)
        worst = max(worst, float(-P @ vrel + 0.5 * P @ K @ P) / (0.5 * vrel @ np.linalg.inv(K) @ vrel))
    assert worst <= 1e-12


def test_fragmentation_never_creates_energy():
    """v0.5 check fragmentation_energy, parts A and B: grains hitting spinning chains and L-shapes at 0.02-3 m/s,
    and spinning trimers hitting a wall.  Kinetic + broken-bond energy may never rise; no clump may keep grains
    that no longer touch; at least 20 impacts fragment."""
    tub = sim3d.Tubule(accel=(0.0, 0.0, 0.0), size=(0.02, 0.02, 0.02), wall="glass_fluorosilane", f_wall=0.05,
                       mu=0.3)
    rng = np.random.default_rng(11)
    gains, phantom = [], 0
    resolve = sim3d.resolve

    def audited(sim, ev, cand):
        """sim3d.resolve, plus the change of kinetic + broken-bond energy across the event."""
        E0 = cl.kinetic_energy(sim.clumps) + sim.E_bonds_broken
        rec = resolve(sim, ev, cand)
        if rec is not None:
            gain = cl.kinetic_energy(sim.clumps) + sim.E_bonds_broken - E0
            gains.append((gain / max(rec["E_coll"], 1e-300), rec["fragments"] > 0))
        return rec
    for trial in range(90):
        c0 = centre(tub)
        if trial < 60:
            shape = [[-2 * R, 0, 0], [0, 0, 0], [2 * R, 0, 0]] if trial % 2 else [[0, 0, 0], [2 * R, 0, 0], [2 * R, 2 * R, 0]]
            pos = list(c0 + np.array(shape))
            tgt = pos[rng.integers(3)]
            d = rng.standard_normal(3)
            d /= np.linalg.norm(d)
            pos.append(tgt - d * 2.6 * R + np.cross(d, rng.standard_normal(3)) * 0.5 * R)
            speed = math.exp(rng.uniform(math.log(0.02), math.log(3.0)))
            vel = [np.zeros(3)] * 3 + [d * speed]
        else:
            pos = [[0.01, 0.01, 1.2 * R], [0.01 + 2 * R, 0.01, 1.2 * R], [0.01 + 2 * R, 0.01 + 2 * R, 1.2 * R],
                   [0.015, 0.015, 0.01]]
            vel = [np.zeros(3)] * 4
        s = sim3d.make_sim(tub, "silica", [R] * 4, pos, vel)
        A = chain_clump(s, [0, 1, 2])
        c = s.clumps
        if trial < 60:
            c.V[A] = rng.standard_normal(3) * 0.01
        else:
            speed = math.exp(rng.uniform(math.log(0.02), math.log(3.0)))
            v = rng.standard_normal(3)
            v[2] = -abs(v[2]) - 0.2
            c.V[A] = speed * v / np.linalg.norm(v)
        c.L[A] = np.linalg.inv(c.Iinv[A]) @ (rng.standard_normal(3) * 2 * math.pi * 10)
        sim3d.resolve = audited
        try:
            for _ in range(3):
                sim3d.step(s)
        finally:
            sim3d.resolve = resolve
        phantom += sum(len(cl.bond_groups(c, b)) > 1 for b in np.flatnonzero(c.alive & (c.n > 1)))
    worst, n_frag = max(g for g, _ in gains), sum(f for _, f in gains)
    assert worst <= 1e-9 and phantom == 0 and n_frag >= 20, (worst, phantom, n_frag)


def test_no_missed_collisions_and_no_wall_penetration():
    """v0.5 checks no_overlap and walls (static walls): 120 silica grains in a 3 mm box, barely sticky walls so
    they keep bouncing, 0.4 s.  After every step no two clumps overlap by more than 1e-3 r, no grain crosses a
    wall, and the safety net never acts."""
    tub = sim3d.Tubule(size=(3e-3, 3e-3, 3e-3), wall="glass_fluorosilane", f_wall=0.05, accel=(0.0, 0.0, 0.0))
    rng = np.random.default_rng(3)
    n = 120
    pos = scenes.puff_positions([R] * n, centre(tub), 2.6e-3, rng)
    vel = scenes.isotropic_velocities(n, 0.03, 0.01, rng)
    s = sim3d.make_sim(tub, "silica", [R] * n, pos, vel)
    worst, worst_wall = 0.0, 0.0
    I, J = np.triu_indices(n, 1)
    for _ in range(100):
        sim3d.step(s)
        c = s.clumps
        sel = c.body[I] != c.body[J]
        d = np.linalg.norm(c.pos[I[sel]] - c.pos[J[sel]], axis=1)
        worst = max(worst, float((2 * R - d).max()))
        worst_wall = max(worst_wall, float(np.maximum(R - c.pos, c.pos + R - np.asarray(tub.size)).max()))
    pp = s.counts["stick"] + s.counts["bounce"]
    assert worst < 1e-3 * R and worst_wall < 1e-12 and s.contain_corrections == 0 and pp > 20, (worst, pp)


def test_momentum_ledger_of_the_two_tubule_run():
    """The page's scenario: every event conserves P and L to rounding (1e-13 of the momenta involved), and the
    ledger P(t) = P(0) + external + what the tubule received holds to 1e-12 over the whole run."""
    st = scenes.two_tubule_start()
    for m in ("silica", "graphite"):
        s = scenes.run_tubule(st, m, 4.0)
        assert s.events and max(e["dP_rel"] for e in s.events) < 1e-13
        assert max(e["dL_rel"] for e in s.events) < 1e-13
        eP, eL = sim3d.conservation_error(s)
        assert eP < 1e-12 and eL < 1e-12 and s.contain_corrections == 0


def test_clumps_are_held_by_their_stored_bonds():
    """Every clump at the end of both tubule runs is held together by its stored bonds (N - 1 bonds for N grains,
    one connected part), every bond is a contact (gap within the arc bound rho theta^2/2 = 1.25e-3 rho of the
    straight-line search), and at least one bond of a grain that stuck to a spinning clump has a gap the old
    distance test (1e-6 of the radius sum) missed.  For such a clump nothing splits off without an impact, and
    eroding the loosely placed grain costs its real bond."""
    st = scenes.two_tubule_start()
    hidden = []
    for m in ("silica", "graphite"):
        s = scenes.run_tubule(st, m, 4.0)
        c = s.clumps
        for b in np.flatnonzero(c.alive & (c.n > 1)):
            assert len(c.bonds[b]) == c.n[b] - 1 and len(cl.bond_groups(c, b)) == 1, (m, b, c.bonds[b])
            for i, j in c.bonds[b]:
                S = c.r[i] + c.r[j]
                gap = np.linalg.norm(c.pos[i] - c.pos[j]) - S
                assert -1e-12 * S < gap < 1.25e-3 * S, (m, b, gap)
                if gap > 1e-6 * S:
                    hidden.append((s, b, i, j))
    assert hidden                                  # the case the stored bonds exist for (graphite trimers)
    s, b, i, j = hidden[0]
    c = s.clumps
    assert cl.split_disconnected(c, b) == [] and cl.n_bonds(c, b, i) >= 1 and cl.n_bonds(c, b, j) >= 1
    assert len(cl.touching_groups(c, c.members[b])) > 1          # what the distance test alone would have done


def test_v05_bodies_are_held_by_their_stored_bonds():
    """The same for the full simulator's v0.5 engine (dolle_sim/simulation.py ChamberSim.bonds) on the graphite
    tubule's start."""
    st = scenes.two_tubule_start()
    tub = sim3d.Tubule(mu=0.3)
    s = v05_sim(tub, ["graphite"] * len(st["radii"]), st["radii"], st["positions"], st["velocities"], 1.0,
                adhesion=st["adhesion"])
    s.run(progress=False)
    multi = np.flatnonzero(s.alive & (s.nmono > 1))
    assert len(multi) >= 5
    for b in multi:
        assert len(s.bonds[b]) == s.nmono[b] - 1 and len(s._components(b)) == 1, (b, s.bonds[b])
        for i, j in s.bonds[b]:
            S = s.rad[i] + s.rad[j]
            assert -1e-12 * S < np.linalg.norm(s.pos[i] - s.pos[j]) - S < 1.25e-3 * S


# ============================================================================ against dolle_sim v0.5
def _case_head_on(material, v):
    c0 = centre()
    return dict(mats=[material] * 2, radii=[R, R], pos=[c0 - [1.5 * R, 0, 0], c0 + [1.5 * R, 0, 0]],
                vel=[[v / 2, 0, 0], [-v / 2, 0, 0]])


def _case_oblique_spin():
    c0 = centre()
    return dict(mats=["silica"] * 2, radii=[R, 45e-6], pos=[c0 - [3 * R, 0.8 * R, 0], c0],
                vel=[[0.03, 0, 0.002], [0, 0, 0]], spins=[[0, 0, 300.0], [100.0, 0, 0]])


def _case_chain():
    c0 = centre()
    return dict(mats=["silica"] * 3, radii=[R, R, 48e-6], pos=[c0 - [3 * R, 0, 0], c0, c0 + [2.3 * R, 0.1 * R, 0]],
                vel=[[0.03, 0, 0], [0, 0, 0], [0, 0, 0]])


def _case_onto_dimer(material, v):
    c0 = centre()
    def prep(a, b):
        A = chain_clump(a, [0, 1])
        b._merge(0, 1)
        Lb = np.array([0.0, 0.0, 1.0]) * 2 * math.pi * 30 / a.clumps.Iinv[A][2, 2]
        a.clumps.L[A] = Lb
        b.L[b.body[0]] = Lb
    return dict(mats=[material] * 3, radii=[R] * 3, pos=[c0 - [R, 0, 0], c0 + [R, 0, 0], c0 + [0, 6 * R, 0]],
                vel=[[0, 0, 0], [0, 0, 0], [0, -v, 0]], prep=prep)


def _case_onto_wall_grain(material, v):
    c0 = centre()
    def prep(a, b):
        cl.freeze(a.clumps, 0, 2, -1.0)
        b.frozen[0], b.fw_ax[0], b.fw_side[0], b.V[0], b.L[0] = True, 2, -1.0, 0, 0
    return dict(mats=[material] * 2, radii=[R, R], pos=[[c0[0], c0[1], R], [c0[0] + 0.5 * R, c0[1], 5 * R]],
                vel=[[0, 0, 0], [0, 0, -v]], prep=prep)


def _case_aggregate_on_wall():
    c0 = centre()
    def prep(a, b):
        A = chain_clump(a, [0, 1, 2])
        b._merge(0, 1)
        b._merge(b.body[0], 2)
        Lb = np.linalg.inv(a.clumps.Iinv[A]) @ np.array([20.0, -40.0, 10.0])
        a.clumps.L[A] = Lb
        b.L[b.body[0]] = Lb
    z = 4 * R
    return dict(mats=["silica"] * 3, radii=[R] * 3,
                pos=[[c0[0], c0[1], z], [c0[0] + 2 * R, c0[1], z], [c0[0] + 2 * R, c0[1] + 2 * R, z]],
                vel=[[0.01, 0, -0.05]] * 3, prep=prep)


def _case_dimer_hit(v):
    c0 = centre()
    def prep(a, b):
        chain_clump(a, [0, 1])
        b._merge(0, 1)
    return dict(mats=["silica"] * 3, radii=[R] * 3, pos=[c0 - [R, 0, 0], c0 + [R, 0, 0], c0 + [-R, 5 * R, 0]],
                vel=[[0, 0, 0], [0, 0, 0], [0, -v, 0]], prep=prep)


SCRIPTED = {"head-on bounce": _case_head_on("silica", 0.02), "head-on stick": _case_head_on("graphite", 0.01),
            "oblique bounce with spin": _case_oblique_spin(), "three-grain chain": _case_chain(),
            "grain onto spinning dimer (stick)": _case_onto_dimer("graphite", 0.01),
            "grain onto spinning dimer (bounce)": _case_onto_dimer("silica", 0.03),
            "grain onto wall-bound grain (stick)": _case_onto_wall_grain("graphite", 0.005),
            "grain onto wall-bound grain (bounce)": _case_onto_wall_grain("silica", 0.03),
            "spinning trimer hits a wall": _case_aggregate_on_wall(),
            "erosion of a dimer": _case_dimer_hit(0.011), "shatter of a dimer": _case_dimer_hit(0.06)}


@pytest.mark.parametrize("name", list(SCRIPTED))
def test_scripted_cases_match_dolle_sim(name):
    case = SCRIPTED[name]
    tub = sim3d.Tubule(mu=0.3)
    a, b = both(tub, case["mats"], case["radii"], case["pos"], case["vel"], 0.2, spins=case.get("spins"),
                prep=case.get("prep"))
    assert a.events and len(a.events) == len(b.collisions)
    assert same_events(a, b) == len(a.events)
    ca = a.clumps
    spinning = (ca.n[ca.body] > 1) & ~ca.frozen[ca.body]
    tol = np.where(spinning, 1e-7, 1e-12)
    assert np.all(np.abs(ca.pos - b.pos).max(1) <= tol), np.abs(ca.pos - b.pos).max()
    alive = np.flatnonzero(ca.alive)
    assert set(alive) == set(np.flatnonzero(b.alive))
    for x, y in ((ca.V[alive], b.V[alive]), (ca.L[alive], b.L[alive])):
        assert np.abs(x - y).max() <= 1e-9 * max(np.abs(y).max(), 1e-300)
    for k in ("stick", "bounce", "wall_stick", "wall_bounce", "erosion", "shatter"):
        assert a.counts[k] == b.n_events[k], k
    if "erosion" in name:
        assert a.counts["erosion"] == 1
    if "shatter" in name:
        assert a.counts["shatter"] == 1


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_30_grain_cloud_matches_dolle_sim(seed):
    """The sandbox size: the first 20 events identical, the final counts equal (both materials)."""
    for m in ("silica", "graphite"):
        tub, mats, r, pos, vel = cloud(30, 1.5e-3, seed, mats=m)
        a, b = both(tub, mats, r, pos, vel, 12.0)
        n = min(20, len(b.collisions))
        assert same_events(a, b, n=n, tol_t=1e-10, tol_rel=1e-6) == n
        for k in ("stick", "bounce", "wall_stick", "wall_bounce", "erosion", "shatter"):
            assert a.counts[k] == b.n_events[k], (m, k, a.counts, b.n_events)


@pytest.mark.parametrize("material", ["silica", "graphite"])
def test_100_grain_ensemble_matches_dolle_sim(material):
    """8 seeds of a 100-grain cloud (2 mm cube) with EXACT (off-grid) radii and the 0.6 adhesion scatter, so v0.5's
    rounding of radii and adhesion in its contact-law cache is included: the grain-grain stick and bounce totals
    agree within 2 sigma (Poisson) and the median grain-grain speeds within 10 %."""
    tot = {"a": [0, 0], "b": [0, 0]}
    vn = {"a": [], "b": []}
    for seed in range(8):
        tub, mats, r, pos, vel = cloud(100, 2e-3, 100 + seed, grid=False, mats=material)
        adh = np.exp(0.6 * np.random.default_rng(seed).standard_normal(100))
        a, b = both(tub, mats, r, pos, vel, 0.6, adhesion=adh)
        for key, evs in (("a", a.events), ("b", b.collisions)):
            pp = [e for e in evs if e["kind"] == "pp"]
            tot[key][0] += sum(e["outcome"] == "stick" for e in pp)
            tot[key][1] += sum(e["outcome"] == "bounce" for e in pp)
            vn[key] += [e["v_n"] for e in pp]
    for k in range(2):
        x, y = tot["a"][k], tot["b"][k]
        assert abs(x - y) <= 2 * math.sqrt(max(x + y, 1)), tot
    assert abs(np.median(vn["a"]) / np.median(vn["b"]) - 1) < 0.1


def _v05_cloud(material, seed, effect):
    """v0.5 run of a 100-grain cloud in the DOLL-E tubule for 0.6 s with one left-out effect switched on
    ("no_separation": fragments get none of the left-over dissipated energy back)."""
    ChamberSim = _chamber_sim()
    size = list(sim3d.Tubule().size)
    env = {"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": [0, 0, 1e-4], "gravity_gradient": False,
           "magnetic_field_t": [0, 0, 0], "spin_rate_hz": 0.0}
    forces = {k: False for k in OFF}
    contact = {"friction_coefficient": 0.3, "contact_charging": False, "fragmentation": True}
    ch = {"name": material, "size_m": size, "wall_material": "glass_borosilicate",
          "particles": [{"material": material, "count": 100, "surface_energy_log_sigma": 0.6,
                         "size_distribution": {"type": "sieve", "min_m": 9e-5, "max_m": 1.06e-4,
                                               "fraction_in_range": 0.9},
                         "region_m": {"center": [x / 2 for x in size], "size": [2e-3] * 3},
                         "velocity": {"mode": "isotropic", "speed_mean_m_s": 0.015, "speed_sigma_m_s": 0.0075}}]}
    if effect == "drag":                 # the mission's residual gas (configs/mission_quarter_deck.json)
        env.update(pressure_pa=1e-3, gas="air", temperature_k=293.0)
    if effect == "roll":                 # 1 rpm residual roll, tubule T1 at its deck position
        env.update(spin_rate_hz=0.0167)
        ch["center_from_spin_axis_m"] = [-0.080, 0.050, 0.0]
    if effect == "no_separation":
        contact["fragment_separation"] = 0.0
    if effect == "charging":
        contact["contact_charging"] = True
        forces.update(coulomb=True, polarization=True, image_charges=True, space_charge=True)
    cfg = {"dt_s": 4e-3, "duration_s": 0.6, "output_fps": 10, "environment": env, "forces": forces, "contact": contact}
    s = ChamberSim(cfg, ch, seed=seed)
    s.run(progress=False)
    pp = [e for e in s.collisions if e["kind"] == "pp"]
    return len(pp), sum(e["outcome"] == "stick" for e in pp)


@pytest.mark.parametrize("effect", ["drag", "roll", "charging"])
def test_left_out_effects_do_not_change_grain_grain_outcomes(effect):
    """In dolle_sim v0.5, the same 6 seeds x 2 materials with the effect off and on: the grain-grain impact and
    stick totals change by less than 2 sigma (Poisson).  (Gas drag at 1e-3 Pa, the 1 rpm payload roll at the
    tubule's deck radius, contact charging with Coulomb, polarisation and image forces.)"""
    for material in ("silica", "graphite"):
        off = np.array([_v05_cloud(material, s, None) for s in range(1, 7)]).sum(0)
        on = np.array([_v05_cloud(material, s, effect) for s in range(1, 7)]).sum(0)
        for x, y in zip(off, on):
            assert abs(x - y) <= 2 * math.sqrt(max(x + y, 1)), (material, effect, off, on)


def test_separation_bound_in_both_engines():
    """The separation energy after a break-up is ASSUMED (impacts.py point 4).  Its opposite bound, none given back:
    in the core's two-tubule run silica's impacts / sticks go from 20 / 4 to 22 / 6 and graphite's from 17 / 17 to
    18 / 18 (notebook 06, section 9); in v0.5, 6 seeds x 100 grains, from 97/21 to 98/22 (silica) and 87/82 to 88/83
    (graphite).  The core and v0.5 also still give the same events with none given back."""
    from dataclasses import replace
    st = scenes.two_tubule_start()
    got = {}
    for sep in (1.0, 0.0):
        for m in ("silica", "graphite"):
            s = scenes.run_tubule({**st, "tubule": replace(st["tubule"], separation=sep)}, m, 4.0)
            got[sep, m] = (sum(e["kind"] == "pp" for e in s.events), s.counts["stick"])
            assert max(e["dP_rel"] for e in s.events) < 1e-13 and max(e["dL_rel"] for e in s.events) < 1e-13
    assert got == {(1.0, "silica"): (20, 4), (1.0, "graphite"): (17, 17),
                   (0.0, "silica"): (22, 6), (0.0, "graphite"): (18, 18)}, got
    for material, off_on in (("silica", ((97, 21), (98, 22))), ("graphite", ((87, 82), (88, 83)))):
        off = tuple(np.array([_v05_cloud(material, s, None) for s in range(1, 7)]).sum(0))
        on = tuple(np.array([_v05_cloud(material, s, "no_separation") for s in range(1, 7)]).sum(0))
        assert (off, on) == off_on, (material, off, on)
    case = _case_dimer_hit(0.06)                                 # shatter of a dimer, none given back: same events
    tub = sim3d.Tubule(mu=0.3, separation=0.0)
    a = sim3d.make_sim(tub, case["mats"], case["radii"], case["pos"], case["vel"])
    b = v05_sim(tub, case["mats"], case["radii"], case["pos"], case["vel"], 0.2)
    b.frag_separation = 0.0
    case["prep"](a, b)
    sim3d.run(a, 0.2, stop_when_settled=False)
    b.run(progress=False)
    assert any(e["mode"] == "shatter" for e in a.events) and same_events(a, b) == len(a.events)


# ============================================================================ the page's data
def test_impact_closeup_outcomes_and_energy():
    for case in scenes.closeup_cases():
        want = "stick" if case["factor_of_v_stick"] < 1 else "bounce"
        assert case["outcome"] == want and len(case["rows"]) <= 400
        assert case["energy_error_rel"] < 1e-3 and case["momentum_error_rel"] < 1e-14


def test_sandbox_runs_and_returns_json():
    t0 = time.perf_counter()
    state = scenes.sandbox_start(json.dumps({"n": 30, "seed": 2, "t_end": 2.0}))
    done = False
    while not done:
        out = json.loads(scenes.sandbox_advance(state, 0.5))
        done = all(t["done"] for t in out["tubules"])
    assert time.perf_counter() - t0 < 5.0
    assert len(out["tubules"]) == 2 and out["event_columns"][0] == "t_s"
    for preset in ("spread", "kick"):
        st = scenes.sandbox_start(json.dumps({"preset": preset, "n": 20, "t_end": 0.2}))
        json.loads(scenes.sandbox_advance(st, 0.2))


def test_replay_files_are_consistent():
    """docs/sim3d/data/two_tubules.json (written by tools/make_sim3d_replays.py): lengths, every grain inside the
    tubule, clump ids consistent with the counts, the summary's counts equal to the event counts, < 3 MB."""
    path = ROOT / "docs" / "sim3d" / "data" / "two_tubules.json"
    if not path.exists():
        pytest.skip("run python tools/make_sim3d_replays.py first")
    assert path.stat().st_size < 3e6
    rep = json.loads(path.read_text(encoding="utf-8"))
    assert rep["schema"] == "dolle-sim3d-replay/1" and len(rep["tubules"]) == 2
    for tub in rep["tubules"]:
        N, fr = len(tub["radii_um"]), tub["frames"]
        size, r = np.array(tub["size_um"]), np.array(tub["radii_um"])
        assert len(fr["t"]) == len(fr["pos"]) == len(fr["clump"]) == len(fr["wall"]) == len(fr["colour"])
        for pos in fr["pos"]:
            p = np.array(pos).reshape(N, 3)
            assert (p >= r[:, None] - 1).all() and (p <= size - r[:, None] + 1).all()
        cols = tub["event_columns"]
        ev = [dict(zip(cols, row)) for row in tub["events"]]
        cnt = tub["summary"]["counts"]
        assert cnt["stick"] == sum(e["kind"] == 0 and e["outcome"] == 1 for e in ev)
        assert cnt["bounce"] == sum(e["kind"] == 0 and e["outcome"] == 0 for e in ev)
        assert cnt["wall_stick"] == sum(e["kind"] == 1 and e["outcome"] == 1 for e in ev)
        assert cnt["wall_bounce"] == sum(e["kind"] == 1 and e["outcome"] == 0 for e in ev)
        assert max(e["dP_rel"] for e in ev) < 1e-13 and max(e["dL_rel"] for e in ev) < 1e-13
        sizes = np.bincount(fr["clump"][-1])
        sizes = sizes[sizes > 0]
        assert {str(k): int((sizes == k).sum()) for k in set(sizes.tolist())} == tub["summary"]["clump_sizes_at_end"]

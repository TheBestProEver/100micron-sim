"""dolle_core (the readable core) against the full simulator dolle_sim, and against the laws it must obey.

Reference numbers quoted in comments were produced by running dolle_sim at commit f133c3e; every comparison with
dolle_sim is made here by calling dolle_sim directly.
"""
import math
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

import dolle_core as dc
from dolle_sim import materials as sim_materials
from dolle_sim.contact import JKRContact
from dolle_sim.verification import _e_tn

ROOT = Path(__file__).resolve().parents[1]
R = 50e-6
CASES = [("silica", "silica", False), ("graphite", "graphite", False),
         ("silica", "glass_borosilicate", True), ("graphite", "glass_borosilicate", True)]


def _sim(a, b, wall, **over):
    """dolle_sim's JKRContact, m* and PairProps for a case (optionally with overridden material fields)."""
    ma, mb = replace(sim_materials.get_material(a), **over), replace(sim_materials.get_material(b), **over)
    p = sim_materials.pair_props(ma, R, mb, None if wall else R)
    m = sim_materials.sphere_mass(ma, R)
    return p, (m if wall else m / 2)


def _core(a, b, wall, plastic=True, **over):
    ma = replace(dc.get_material(a), **over)
    mb = replace(dc.get_material(b), **over)
    return dc.make_pair(ma, R, mb, None if wall else R, plastic=plastic)


# ---------------------------------------------------------------------------- materials
def test_every_material_value_equals_dolle_sim():
    assert set(dc.MATERIALS) == set(sim_materials.MATERIALS)
    for name, m in dc.MATERIALS.items():
        s = sim_materials.MATERIALS[name]
        for f in ("rho", "E", "nu", "gamma", "e_dry", "yield_pressure"):
            assert getattr(m, f) == getattr(s, f), (name, f)
        assert m.source


def test_friction_lookup():
    """(The table itself is compared with the v0.6 engine's MU_DEFAULT in test_core_vs_dem.py.)"""
    assert dc.friction_coefficient("silica", "wall") == 0.4
    assert dc.friction_coefficient("silica", "graphite") == 0.5          # looked up in either order
    assert dc.friction_coefficient("alumina", "wall") == 0.4             # the engine's default


@pytest.mark.parametrize("a,b,wall", CASES)
@pytest.mark.parametrize("f_adh", [1.0, 0.05])
def test_pair_properties_equal_pair_props(a, b, wall, f_adh):
    ma, mb = sim_materials.get_material(a), sim_materials.get_material(b)
    p = sim_materials.pair_props(ma, R, mb, None if wall else R, adhesion_factor=f_adh)
    q = dc.make_pair(a, R, b, None if wall else R, f_adh=f_adh)
    assert (q.E_star, q.R_star, q.w, q.e_dry, q.p_y) == (p.E_star, p.R_star, p.w, p.e_dry, p.p_y)
    assert dc.pull_off_force(q.R_star, q.w) == p.F_c
    assert dc.equilibrium_contact_radius(q.R_star, q.w, q.E_star) == p.a0
    assert dc.break_overlap(q.R_star, q.w, q.E_star) == pytest.approx(p.delta_c, rel=1e-13)
    assert q.m1 == sim_materials.sphere_mass(ma, R)


# ---------------------------------------------------------------------------- JKR algebra
@pytest.mark.parametrize("a,b,wall", CASES)
def test_jkr_identities_and_simulator_constants(a, b, wall):
    q = _core(a, b, wall)
    Rs, w, E = q.R_star, q.w, q.E_star
    c = JKRContact(_sim(a, b, wall)[0])
    a0, Fc = dc.equilibrium_contact_radius(Rs, w, E), dc.pull_off_force(Rs, w)
    a_snap, a_c = dc.snap_on_radius(Rs, w, E), dc.break_radius(Rs, w, E)
    assert a_snap == c.a_snap and dc.break_overlap(Rs, w, E) == c.delta_break
    assert dc.contact_work(a_snap, Rs, w, E) - dc.contact_work(a_c, Rs, w, E) == c.U0
    assert a_snap == pytest.approx((2 * math.pi * w * Rs ** 2 / E) ** (1 / 3), rel=1e-12)
    assert a_snap / a0 == pytest.approx((4 / 9) ** (1 / 3), rel=1e-12)
    assert dc.force_of_radius(a0, Rs, w, E) == pytest.approx(0.0, abs=1e-12 * Fc)
    assert dc.force_of_radius(a_snap, Rs, w, E) == pytest.approx(-8 / 9 * Fc, rel=1e-12)
    assert dc.force_of_radius(a_c, Rs, w, E) == pytest.approx(-5 / 9 * Fc, rel=1e-12)
    assert dc.force_of_radius(a0 / 4 ** (1 / 3), Rs, w, E) == pytest.approx(-Fc, rel=1e-12)
    aa = np.linspace(0.3, 1.5, 2001) * a0                      # -F_c is the most tensile force on the curve
    assert min(dc.force_of_radius(x, Rs, w, E) for x in aa) >= -Fc * (1 + 1e-12)
    assert dc.break_overlap(Rs, w, E) == pytest.approx(-0.75 * (math.pi ** 2 * w ** 2 * Rs / E ** 2) ** (1 / 3),
                                                       rel=1e-12)
    assert dc.hysteresis_energy(Rs, w, E) == pytest.approx(7.092196 * w ** (5 / 3) * Rs ** (4 / 3) * E ** (-2 / 3),
                                                           rel=1e-6)
    assert dc.contact_radius(dc.overlap_of_radius(0.9 * a0, Rs, w, E), Rs, w, E, a0, a_c) == pytest.approx(
        0.9 * a0, rel=1e-12)


def test_tabor_parameter_matches_provenance_ranges():
    """PROVENANCE.md: mu ~1-2 for silica, ~7-14 for graphite (interaction range 0.2-0.4 nm)."""
    for name, lo, hi in (("silica", 1.0, 2.2), ("graphite", 6.5, 13.2)):
        q = _core(name, name, False)
        mus = [dc.tabor_parameter(q.R_star, q.w, q.E_star, z0) for z0 in (0.4e-9, 0.2e-9)]
        assert lo <= mus[0] < mus[1] <= hi


# ---------------------------------------------------------------------------- impacts vs dolle_sim
@pytest.mark.parametrize("a,b,wall", CASES)
def test_sticking_threshold_equals_simulator(a, b, wall):
    p, me = _sim(a, b, wall)
    q = _core(a, b, wall)
    assert q.m_star == me
    ref = JKRContact(p).threshold(me)
    assert dc.sticking_threshold(q) == pytest.approx(ref, rel=1e-12)
    ref_f = JKRContact(p, separation="force").threshold(me)
    assert dc.sticking_threshold(q, break_mode="force") == pytest.approx(ref_f, rel=1e-12)


def test_threshold_reference_numbers():
    """dolle_sim's thresholds at 300 steps per contact (mm/s, commit f133c3e): 6.17518, 38.35856, 6.66206, 18.23326."""
    for (a, b, wall), ref in zip(CASES, (6.17518, 38.35856, 6.66206, 18.23326)):
        assert dc.sticking_threshold(_core(a, b, wall)) * 1e3 == pytest.approx(ref, abs=1e-5)


@pytest.mark.parametrize("a,b,wall", CASES)
def test_restitution_and_impact_details_equal_simulator(a, b, wall):
    p, me = _sim(a, b, wall)
    q = _core(a, b, wall)
    c = JKRContact(p)
    thr = c.threshold(me)
    for v in (0.5 * thr, 1.2 * thr, 2 * thr, 4 * thr, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        r, s = dc.impact(q, v), c.resolve(v, me)
        assert r.stuck == s.stuck and r.yielded == s.yielded
        assert abs(r.e - s.e_n) <= 1e-12
        for x, y in ((r.max_overlap, s.max_overlap), (r.max_force, s.max_force), (r.contact_time, s.contact_time)):
            assert x == pytest.approx(y, rel=1e-12, abs=1e-30)


def test_simulator_contact_checks_repeated():
    """dolle_sim's checks elastic (0.99990), damping (0.8501; graphite 0.6998), thornton_ning (e_TN within 0.005)
    and plastic_damping (within 2.6% of e_TN e_dry), each equal to dolle_sim's own number to 1e-12."""
    def both(name, v, **over):
        p, me = _sim(name, name, False, **over)
        return dc.impact(_core(name, name, False, **over), v), JKRContact(p).resolve(v, me), p, me
    r, s, _, _ = both("silica", 0.3, e_dry=0.99999, gamma=1e-7, yield_pressure=math.inf)
    assert abs(r.e - s.e_n) < 1e-12 and abs(r.e - 1) < 1e-3
    for name, e_dry in (("silica", 0.85), ("graphite", 0.7)):
        r, s, _, _ = both(name, 0.1, gamma=1e-7, yield_pressure=math.inf)
        assert abs(r.e - s.e_n) < 1e-12 and abs(r.e - e_dry) < 0.005
    for v in (1.0, 3.0, 10.0):
        r, s, p, me = both("silica", v, e_dry=0.99999, gamma=1e-7, yield_pressure=0.5e9)
        v_y = dc.yield_speed(me, p.R_star, p.E_star, p.p_y)
        assert v_y == pytest.approx(JKRContact(p).v_yield(me), rel=1e-14)
        assert abs(r.e - s.e_n) < 1e-12 and abs(r.e - dc.thornton_ning_restitution(v, v_y)) < 0.005
        assert dc.thornton_ning_restitution(v, v_y) == _e_tn(v_y / v)
    for v in (3.0, 10.0, 30.0):
        r, s, p, me = both("graphite", v)
        ref = dc.thornton_ning_restitution(v, dc.yield_speed(me, p.R_star, p.E_star, p.p_y)) * 0.7
        assert r.yielded and abs(r.e - s.e_n) < 1e-12 and abs(r.e / ref - 1) < 0.026


def test_undamped_threshold_is_the_hysteresis_energy():
    """With no damping (e_dry = 1) the integrated threshold must be sqrt(2 |U0| / m*)."""
    for name in ("silica", "graphite"):
        q = _core(name, name, False, plastic=False, e_dry=1.0)
        v0 = dc.undamped_sticking_speed(q)
        assert dc.sticking_threshold(q, n_sub=2000, tol=1e-5) == pytest.approx(v0, rel=1e-3)
    # the reference numbers (mm/s): 3.82655 and 15.57579
    assert dc.undamped_sticking_speed(_core("silica", "silica", False)) * 1e3 == pytest.approx(3.82655, abs=1e-5)
    assert dc.undamped_sticking_speed(_core("graphite", "graphite", False)) * 1e3 == pytest.approx(15.57579, abs=1e-5)


# ---------------------------------------------------------------------------- conservation laws
def _momentum_error(r):
    tr = r.trace
    scale = abs(r.pair.m1 * r.u1) + (abs(r.pair.m2 * r.u2) if math.isfinite(r.pair.m2) else 0.0)
    return float(np.max(np.abs(tr["momentum"] - r.momentum_before))) / scale


@pytest.mark.parametrize("u1,u2", [(0.01, 0.0), (0.05, -0.02), (0.003, -0.004), (2.0, 0.0), (30.0, 0.0)])
def test_momentum_conserved_in_every_collision(u1, u2):
    """Grains of different size and material, sticking and bouncing, elastic and plastic: total momentum stays
    constant to rounding at every step, and the final velocities obey the two-body formulas."""
    for a, ra, b, rb in (("silica", 50e-6, "silica", 50e-6), ("graphite", 30e-6, "silica", 70e-6),
                         ("silica", 50e-6, "glass_borosilicate", None)):
        r = dc.collide(a, ra, b, rb, u1, u2 if rb else 0.0)
        assert _momentum_error(r) <= 1e-12
        assert abs(r.momentum_after - r.momentum_before) <= 1e-12 * abs(r.momentum_before) + 1e-30
        p = r.pair
        if r.outcome == "bounce" and rb:
            v_cm, du = (p.m1 * r.u1 + p.m2 * r.u2) / (p.m1 + p.m2), r.e * r.v_in
            assert r.v1 == pytest.approx(v_cm - p.m2 / (p.m1 + p.m2) * du, rel=1e-12, abs=1e-15)
            assert r.v2 == pytest.approx(v_cm + p.m1 / (p.m1 + p.m2) * du, rel=1e-12, abs=1e-15)


def test_energy_conserved_in_the_elastic_hertz_limit():
    """No damping (e_dry = 1), no adhesion (gamma 1e-7 J/m^2), no yield: kinetic + stored energy constant and e = 1.
    Velocity Verlet's energy error is of order dt^2: 1.6e-5 at 300 steps per contact, 1.6e-7 at 3000."""
    m = replace(dc.get_material("silica"), gamma=1e-7, e_dry=1.0, yield_pressure=math.inf)
    q = dc.make_pair(m, R, m, R)
    E_in = 0.5 * q.m_star * 0.3 ** 2
    r = dc.collide_pair(q, 0.3, 0.0)
    assert r.outcome == "bounce" and abs(r.e - 1) <= 1e-3
    err300 = np.max(np.abs(r.trace["energy_error"])) / E_in
    r3 = dc.collide_pair(q, 0.3, 0.0, n_sub=3000)
    err3000 = np.max(np.abs(r3.trace["energy_error"])) / E_in
    assert err3000 <= 1e-6 and abs(r3.e - 1) <= 1e-6
    assert 50 < err300 / err3000 < 200                         # second order: (3000/300)^2 = 100
    assert np.all(r3.trace["damping_loss"] == 0) and np.all(r3.trace["plastic_loss"] == 0)


def test_energy_bookkeeping_with_adhesion_damping_and_yield():
    """Kinetic + stored + damping + hysteresis + plastic = incoming energy, to the integrator's error while in
    contact; the plastic work appears only after yield."""
    for name, v in (("silica", 0.01), ("graphite", 0.05), ("graphite", 10.0)):
        q = _core(name, name, False)
        r = dc.collide_pair(q, v, 0.0)
        err = np.abs(r.trace["energy_error"][:-1]) / (0.5 * q.m_star * v * v)
        assert np.max(err) < 1e-4
        assert (r.trace["plastic_loss"][-1] > 0) == r.yielded


def test_line_of_grains_conserves_momentum():
    """Four grains on a line (two materials, two sizes), contacts forming and breaking: total momentum after
    every step equal to the first to <= 1e-12 of the momenta involved."""
    mats, radii = ["silica", "graphite", "silica", "graphite"], [50e-6, 40e-6, 50e-6, 60e-6]
    x0 = np.cumsum([0.0, 90.3e-6, 90.2e-6, 110.4e-6])
    v0 = [0.03, 0.0, -0.002, -0.01]
    res = dc.line_collision(mats, radii, x0, v0, t_end=1e-4)
    scale = float(np.sum(res.masses * np.abs(v0)))
    P0 = float(np.sum(res.masses * np.array(v0)))
    assert np.max(np.abs(res.momentum - P0)) / scale <= 1e-12
    assert res.touching.any() and len(res.momentum) > 1000
    assert res.kinetic[-1] < res.kinetic[0]                     # hysteresis and damping take energy


def test_two_grain_line_agrees_with_collide():
    """The many-grain integrator with two grains gives the restitution of collide.py within the step error of the
    1-D model itself, and both converge to the same value.  (Its step is set by the largest possible speed and it
    starts with a gap, so its steps fall differently relative to the break, which is detected to within one step:
    collide's own restitution changes by 2.9e-4 between 300 and 3000 steps per contact here.)"""
    v = 0.05
    line = lambda n: dc.line_collision(["silica", "silica"], [R, R], [0.0, 2 * R + 0.05e-6], [v, 0.0], t_end=1.2e-5,
                                       n_sub=n)
    res, res_fine = line(300), line(1000)
    r = dc.collide("silica", R, "silica", R, v, 0.0)
    r_fine = dc.collide("silica", R, "silica", R, v, 0.0, n_sub=3000)
    assert not res.touching[-1, 0]
    assert abs(r.e / r_fine.e - 1) < 5e-4
    assert (res.v[-1, 1] - res.v[-1, 0]) / v == pytest.approx(r.e, rel=1e-3)
    assert (res_fine.v[-1, 1] - res_fine.v[-1, 0]) / v == pytest.approx(r_fine.e, rel=5e-5)


def test_line_speed_bound_uses_the_full_well_depth():
    """many.py bounds the relative speed with the depth of each adhesive well measured from its bottom at a0,
    G_break - G(a0) = 1.5426 F_c |delta_c| (= 1.649 |U0|), not with |U0| (the energy from the snap-on state)."""
    from dolle_core.many import _speed_bound
    p = dc.make_pair("silica", R, "silica", R, plastic=False)
    law = dc.contact_law(p)
    F_dc = dc.pull_off_force(p.R_star, p.w) * abs(dc.break_overlap(p.R_star, p.w, p.E_star))
    depth = law.G_break - dc.contact_work(law.a0, law.R, law.w, law.E)
    assert depth / F_dc == pytest.approx(1.5426, abs=1e-4)
    assert -law.U0 / F_dc == pytest.approx(0.9355, abs=1e-4)
    m = p.m1
    v = _speed_bound([m, m], [0.0, 0.0], [law])
    assert 0.5 * law.m * v * v == pytest.approx(depth, rel=1e-12)


def test_jkr_stiffness_at_rest_and_hysteresis_substitution():
    """dF/d(delta) at the zero-load patch a0 is (6/5) E* a0 (the time step's t_osc uses 2 E* a0, 1.29x shorter
    period); k_U = (9 pi/2)^(5/3) [g(6^(-2/3)) - g((4/9)^(1/3))], g(s) = 8 s^5/15 - 8 s^3.5/9 + 2 s^2/9."""
    p = dc.make_pair("silica", R, "silica", R)
    Rs, w, E = p.R_star, p.w, p.E_star
    a0, h = dc.equilibrium_contact_radius(Rs, w, E), 1e-6
    a1, a2 = a0 * (1 - h), a0 * (1 + h)
    k = (dc.force_of_radius(a2, Rs, w, E) - dc.force_of_radius(a1, Rs, w, E)) /         (dc.overlap_of_radius(a2, Rs, w, E) - dc.overlap_of_radius(a1, Rs, w, E))
    assert k / (E * a0) == pytest.approx(1.2, rel=1e-8)
    g = lambda s: 8 * s ** 5 / 15 - 8 * s ** 3.5 / 9 + 2 * s * s / 9
    assert (4.5 * math.pi) ** (5 / 3) * (g(6 ** (-2 / 3)) - g((4 / 9) ** (1 / 3))) == pytest.approx(
        dc.hysteresis_constant(), rel=1e-10)


def test_rolling_grain_decelerates_below_its_limit():
    """wall.rolling_acceleration: from rest nothing below the limit; a rolling grain is braked by the limit torque
    at (a - a_roll)/1.4 and, with dt given, stops instead of reversing."""
    a_roll = 10.0
    assert dc.rolling_acceleration(5.0, a_roll) == 0.0
    assert dc.rolling_acceleration(5.0, a_roll, v=1.0) == pytest.approx(-5.0 / 1.4, rel=1e-15)
    assert dc.rolling_acceleration(5.0, a_roll, v=1e-4, dt=1e-3) == pytest.approx(-0.1, rel=1e-12)
    assert dc.rolling_acceleration(30.0, a_roll, v=1.0) == pytest.approx(20.0 / 1.4, rel=1e-15)


def test_example_tubules_report_no_absolute_path():
    """nb_flight.example_tubules prints where the tubules came from relative to the repository (no local path)."""
    from dolle_core import nb_flight
    tubules, source = nb_flight.example_tubules()
    assert len(tubules) == 4
    assert not Path(source).is_absolute() and ":" not in source and "Users" not in source


def test_line_refuses_impacts_that_could_yield():
    with pytest.raises(ValueError):
        dc.line_collision(["graphite", "graphite"], [R, R], [0.0, 2 * R + 1e-7], [2.0, 0.0], t_end=1e-6)


# ---------------------------------------------------------------------------- wall statics
def test_wall_reference_numbers():
    """Spec 7.5: pull-off 1033.74 / 1627.71 g, rolling 1.2935 / 3.5897 g (1.2942 / 3.5908 g pressed by 1 g),
    with wall_adh 0.05 rolling 0.0238 / 0.0661 g."""
    for grain, pull, roll, roll1, roll05 in (("silica", 1033.74, 1.2935, 1.2942, 0.0238),
                                             ("graphite", 1627.71, 3.5897, 3.5908, 0.0661)):
        L = dc.wall_limits(grain, R, "glass_borosilicate")
        assert round(L.pull_off_g, 2) == pull and round(L.roll_g, 4) == roll
        assert round(dc.wall_limits(grain, R, "glass_borosilicate", press_g=1.0).roll_g, 4) == roll1
        assert round(dc.wall_limits(grain, R, "glass_borosilicate", f_adh=0.05).roll_g, 4) == roll05


# ---------------------------------------------------------------------------- hand scaling laws
def test_scaling_law_prefactors_and_exactness():
    """Spec 9.1: |U0| = 7.0922 w^(5/3) R*^(4/3) E*^(-2/3); v = 1.6394 gamma^(5/6) E*^(-1/3) R*^(-5/6) rho^(-1/2)
    for equal spheres, 1.8402 w^(5/6) ... for a sphere on a wall; exact for this model ('6 digits')."""
    assert dc.hysteresis_constant() == pytest.approx(7.0922, abs=5e-5)
    assert dc.equal_spheres_prefactor() == pytest.approx(1.6394, abs=5e-5)
    assert dc.wall_prefactor() == pytest.approx(1.8402, abs=5e-5)
    for name, r in (("silica", 50e-6), ("graphite", 50e-6), ("graphite", 20e-6), ("water_ice", 80e-6)):
        m = dc.get_material(name)
        q = dc.make_pair(name, r, name, r)
        assert dc.scaling_law_speed(m.gamma, q.E_star, q.R_star, m.rho) == pytest.approx(
            dc.undamped_sticking_speed(q), rel=1e-6)
    q = dc.make_pair("silica", R, "glass_borosilicate", None)
    v = dc.wall_prefactor() * q.w ** (5 / 6) * q.E_star ** (-1 / 3) * R ** (-5 / 6) * 2000 ** -0.5
    assert v == pytest.approx(dc.undamped_sticking_speed(q), rel=1e-6)


def test_damping_factor_depends_on_e_dry_only():
    """Spec 4.2: v_stick = f(e_dry) v_stick,0 with f(0.85) = 1.6142, f(0.70) = 2.4589 for every material and size."""
    for e_dry, f_ref in ((0.85, 1.6142), (0.70, 2.4589)):
        fs = [dc.damping_factor(e_dry, n_sub=2000, tol=1e-5, pair=replace(q, p_y=math.inf)) for q in
              (_core("silica", "silica", False), _core("graphite", "graphite", False),
               dc.make_pair("silica", R, "glass_borosilicate", None), dc.make_pair("graphite", 20e-6, "graphite", 20e-6))]
        assert max(fs) / min(fs) - 1 < 3e-5
        assert np.mean(fs) == pytest.approx(f_ref, abs=3e-4)


def test_silica_to_graphite_ratio_by_hand():
    """Spec 9.4: 6.20x = 4.07x (material values: 3^(5/6) x 1.709 x 0.954) x 1.52x (assumed e_dry)."""
    s, g = _core("silica", "silica", False), _core("graphite", "graphite", False)
    by_hand = 3 ** (5 / 6) * (g.E_star / s.E_star) ** (-1 / 3) * (2200 / 2000) ** -0.5
    assert by_hand == pytest.approx(dc.undamped_sticking_speed(g) / dc.undamped_sticking_speed(s), rel=1e-12)
    assert by_hand == pytest.approx(4.0705, abs=1e-4)
    total = dc.sticking_threshold(g, n_sub=2000, tol=1e-5) / dc.sticking_threshold(s, n_sub=2000, tol=1e-5)
    assert total == pytest.approx(6.2003, abs=2e-3)


# ---------------------------------------------------------------------------- packaging
def test_core_imports_without_numba_or_dolle_sim():
    code = ("import sys; sys.modules['numba'] = None; import dolle_core as dc; "
            "assert 'dolle_sim' not in sys.modules; "
            "print(dc.sticking_threshold(dc.make_pair('silica', 5e-5, 'silica', 5e-5)))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert float(out.stdout) == pytest.approx(6.175177e-3, rel=1e-6)


def test_pyproject_packages_include_core():
    import tomllib
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "dolle_core" in cfg["tool"]["setuptools"]["packages"]


@pytest.mark.parametrize("script", sorted((ROOT / "examples").glob("*.py")))
def test_examples_run(script):
    out = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip()

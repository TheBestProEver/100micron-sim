"""dolle_core.inverse: the e_dry / surface-energy fit from impact lists.

- the cached table h(x, e_dry) reproduces collide.impact for other materials, sizes, surface energies and a wall;
- the fit recovers known (gamma_scale, e_dry) from synthetic flights within its own 1-sigma error bars in most
  seeds (coverage over 20 seeds; the fractions are printed: run with -s to see them);
- with stick/bounce alone gamma and e_dry are degenerate, and the restitution of bounces breaks the degeneracy;
- the curvature (Hessian) error bars agree with the profile ones where the speed integral used to be too coarse
  (graphite, gentle release, 25-50 impacts), and with a grain-to-grain adhesion spread the gamma pulls have unit
  spread;
- a bounce far below the predicted threshold is evaluated above the threshold (top-tail node);
- the CSV reader round-trips, groups a mixed pair whatever the column order, and rejects malformed rows.
Each test takes well under 30 s.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

import dolle_core as dc
from dolle_core import inverse as inv


# ---------------------------------------------------------------------------- the table
def test_threshold_factor_equals_damping_factor():
    for e in (0.35, 0.5, 0.7, 0.85, 0.95):
        assert inv.threshold_factor(e) == pytest.approx(dc.damping_factor(e, n_sub=300, tol=1e-6), rel=2e-4)
    for e in (0.3, 0.42, 0.85, 0.99):
        assert inv.e_of_threshold_factor(inv.threshold_factor(e)) == pytest.approx(e, abs=1e-6)


def test_pair_speeds_equal_the_core():
    for m1, r1, m2, r2, s in (("silica", 45e-6, "silica", 53e-6, 1.0), ("graphite", 50e-6, "graphite", 50e-6, 2.5),
                              ("silica", 50e-6, "glass_borosilicate", None, 0.3)):
        p = dc.make_pair(m1, r1, m2, r2, f_adh=s)
        v0, vy = inv.pair_speeds(m1, m2, r1, r2, s)
        assert v0 == dc.undamped_sticking_speed(p)
        assert vy == dc.yield_speed(p.m_star, p.R_star, p.E_star, p.p_y)


def test_table_reproduces_collide_for_other_pairs():
    """One table, computed on two 50 um silica grains, gives every pair's outcome (module docstring, section 1)."""
    rng = np.random.default_rng(1)
    cases = [("graphite", 45e-6, "graphite", 53e-6), ("silica", 47e-6, "silica", 51e-6),
             ("silica", 50e-6, "glass_borosilicate", None), ("graphite", 48e-6, "silica", 50e-6)]
    worst, n = 0.0, 0
    for m1, r1, m2, r2 in cases:
        for _ in range(12):
            e, s = rng.uniform(0.3, 0.99), math.exp(rng.uniform(-1.5, 1.5))
            p = replace(dc.make_pair(m1, r1, m2, r2, f_adh=s, plastic=False), e_dry=e)
            v0 = dc.undamped_sticking_speed(p)
            x = inv.threshold_factor(e) * math.exp(rng.uniform(-0.3, 4.0))
            direct = dc.impact(p, x * v0)
            table = float(inv.restitution_of_x(x, e))
            assert (direct.e == 0) == (table == 0), (m1, m2, e, x)
            worst = max(worst, abs(direct.e - table))
            n += 1
    print(f"\ntable vs collide.impact: largest |e difference| over {n} impacts = {worst:.4f}")
    assert worst < 0.005


def test_synthetic_flight_table_vs_exact():
    a = inv.make_synthetic_flight(60, "graphite", seed=4)
    b = inv.make_synthetic_flight(60, "graphite", seed=4, exact=True)
    assert np.mean(a.stuck == b.stuck) >= 0.95
    both = ~a.stuck & ~b.stuck
    assert np.max(np.abs(a.truth["restitution"][both] - b.truth["restitution"][both])) < 0.005


# ---------------------------------------------------------------------------- the fit
def test_fit_recovers_known_values_coverage():
    """20 synthetic silica flights (150 impacts each, true gamma_scale 1.0, e_dry 0.90): how often does each 1-sigma
    interval contain the true value?  Expected about 68 %; required at least half."""
    s_true, e_true, cover_s, cover_e, pull_s, pull_e = 1.0, 0.90, [], [], [], []
    for seed in range(20):
        imp = inv.make_synthetic_flight(150, "silica", gamma_scale=s_true, e_dry=e_true, seed=100 + seed)
        r = inv.fit(imp)
        c = r.covers(s_true, e_true)
        cover_s.append(c["gamma_scale"])
        cover_e.append(c["e_dry"])
        pull_s.append(math.log(r.gamma_scale / s_true) / r.gamma_rel_halfwidth)
        pull_e.append((r.e_dry - e_true) / r.e_dry_halfwidth)
    print(f"\ncoverage over 20 seeds: gamma {np.mean(cover_s):.2f}, e_dry {np.mean(cover_e):.2f}, both "
          f"{np.mean(np.array(cover_s) & np.array(cover_e)):.2f}; mean pull gamma {np.mean(pull_s):+.2f}, "
          f"e_dry {np.mean(pull_e):+.2f}; sd of pulls gamma {np.std(pull_s):.2f}, e_dry {np.std(pull_e):.2f}")
    assert np.mean(cover_s) >= 0.5 and np.mean(cover_e) >= 0.5
    assert abs(np.mean(pull_s)) < 0.5 and abs(np.mean(pull_e)) < 0.5
    # error bars neither too small nor too large (1.5x off would show here; >= 50 % coverage alone barely catches it)
    assert 0.7 < np.std(pull_s) < 1.4 and 0.7 < np.std(pull_e) < 1.4


def test_hessian_agrees_with_profile_graphite_gentle():
    """Graphite, release kept below 3 v_stick, 25-50 impacts: the speed error is small compared with v_stick, where the
    old sigma_v/4 cells made -ln L bumpy and the Hessian error bars 2-2.7x too small (module docstring, section 4)."""
    for n, seed in ((50, 500), (50, 501), (50, 502), (25, 502)):
        imp = inv.make_synthetic_flight(n, "graphite", e_dry=0.70, speed_range=(0.3, 3.0), seed=seed)
        h, p = inv.fit(imp, profile=False), inv.fit(imp)
        print(f"\nN {n} seed {seed}: Hessian +-{h.e_dry_halfwidth:.4f} / {h.gamma_rel_halfwidth:.3f}, "
              f"profile +-{p.e_dry_halfwidth:.4f} / {p.gamma_rel_halfwidth:.3f}")
        assert h.e_dry_halfwidth == pytest.approx(p.e_dry_halfwidth, rel=0.3)
        assert h.gamma_rel_halfwidth == pytest.approx(p.gamma_rel_halfwidth, rel=0.3)


def test_adhesion_spread_pulls():
    """10 silica flights with a per-pair adhesion spread of 0.42 (the simulator's 0.6 per grain, grain-grain), fitted
    with it: the gamma and e_dry pulls must have about unit spread (a 2.3x error underestimate was once found here)."""
    pull_s, pull_e = [], []
    for seed in range(10):
        imp = inv.make_synthetic_flight(150, "silica", e_dry=0.90, sigma_ln_w=0.42, seed=300 + seed)
        r = inv.fit(imp, sigma_ln_w=0.42, profile=False)
        pull_s.append(math.log(r.gamma_scale) / r.gamma_rel_halfwidth)
        pull_e.append((r.e_dry - 0.90) / r.e_dry_halfwidth)
    print(f"\nadhesion spread 0.42: mean pull gamma {np.mean(pull_s):+.2f}, sd {np.std(pull_s):.2f}; "
          f"e_dry {np.mean(pull_e):+.2f}, sd {np.std(pull_e):.2f}")
    assert 0.6 < np.std(pull_s) < 1.5 and abs(np.mean(pull_s)) < 0.8
    assert 0.6 < np.std(pull_e) < 1.5 and abs(np.mean(pull_e)) < 0.8


def test_top_tail_node_lies_above_the_threshold():
    """When the sticking speed lies more than 4 sigma_v above the measured speed (a bounce far below the predicted
    threshold), the restitution of the part above v_s must be evaluated ABOVE v_s: at the mean of the normal
    distribution beyond it, phi(z)/Phi(-z), about z + 1/z (a node below v_s gives e = 0, the restitution of a stuck
    grain).  Such an impact is unlikely but possible, and its -ln L must stay finite and moderate."""
    zi = np.array([[4.5], [6.0], [9.0], [0.3]])
    z = inv._split_node(zi)
    assert np.all(z > zi)
    phi = np.exp(-0.5 * zi[:3] ** 2) / math.sqrt(2 * math.pi)
    np.testing.assert_allclose(z[:3], phi / inv.normal_cdf(-zi[:3]), rtol=1e-6)
    assert z[3, 0] < 0.3 + 1 / 16                              # inside a finite cell: between zi and the cell edge
    v0, _ = inv.pair_speeds("graphite", "graphite", 50e-6, 50e-6)
    vs = inv.threshold_factor(0.7) * v0
    vm = vs - 6 * float(inv.SpeedError().sigma(vs))
    imp = inv.Impacts([vm], [False], [0.15], 50e-6, 50e-6, "graphite", "graphite")
    assert -40 < inv.log_likelihood(imp, 1.0, 0.7, speed_prior="flat") < 0


def test_fit_graphite_and_a_wall():
    imp = inv.make_synthetic_flight(250, "graphite", gamma_scale=1.5, e_dry=0.6, seed=11)
    r = inv.fit(imp)
    assert abs(math.log(r.gamma_scale / 1.5)) < 3 * r.gamma_rel_halfwidth
    assert abs(r.e_dry - 0.6) < 3 * r.e_dry_halfwidth
    wall = inv.make_synthetic_flight(200, "silica", partner="glass_borosilicate", wall=True, seed=12)
    rw = inv.fit(wall)
    e_pair = dc.make_pair("silica", 50e-6, "glass_borosilicate", None).e_dry
    assert abs(math.log(rw.gamma_scale)) < 3 * rw.gamma_rel_halfwidth
    assert abs(rw.e_dry - e_pair) < 3 * rw.e_dry_halfwidth


def test_stick_bounce_alone_is_degenerate_and_restitution_breaks_it():
    imp = inv.make_synthetic_flight(300, "silica", e_dry=0.90, seed=1)
    only = inv.fit(imp, use_restitution=False)
    full = inv.fit(imp)
    # stick/bounce only: e_dry is not constrained at all ...
    assert only.at_bound["e_dry low"] and only.at_bound["e_dry high"]
    assert only.gamma_scale_interval[1] / only.gamma_scale_interval[0] > 5
    # ... because every (s, e_dry) with the same sticking speed f(e_dry) s^(5/6) fits equally well
    P = inv._prepare(imp.without_restitution(), inv.SpeedError(), 0.0, 0.0)
    ls0, e0 = math.log(only.gamma_scale), only.e_dry
    along = [inv._nll(P, ls0 + 1.2 * math.log(inv.threshold_factor(e0) / inv.threshold_factor(e)), e)
             for e in (0.4, 0.6, 0.8, 0.95)]
    across = inv._nll(P, ls0 + 0.4, e0)
    assert max(along) - only.nll_min < 0.5 < across - only.nll_min
    # but the sticking speed itself is well measured either way
    assert only.v_stick == pytest.approx(full.v_stick, rel=0.05)
    # the gamma interval is then set by the e_dry search range, and is flagged as such
    assert only.at_bound["gamma low"] and only.at_bound["gamma high"]
    # with the restitutions of the bounces both numbers are pinned
    assert not any(full.at_bound.values())
    assert full.e_dry_halfwidth < 0.02 and full.gamma_rel_halfwidth < 0.1


def test_predict_and_residuals_are_consistent():
    imp = inv.make_synthetic_flight(300, "graphite", seed=5)
    pr = inv.predict(1.0, 0.7, imp)
    assert np.all((pr.p_stick >= 0) & (pr.p_stick <= 1 + 1e-12))
    # predicted stuck fraction matches the observed one, and restitution pulls scatter like N(0, 1)
    assert abs(pr.p_stick.mean() - imp.stuck.mean()) < 0.06
    res = inv.residuals(imp, 1.0, 0.7)
    pulls = res["e_pull"][np.isfinite(res["e_pull"])]
    assert abs(np.mean(pulls)) < 0.2 and 0.8 < np.std(pulls) < 1.25


# ---------------------------------------------------------------------------- CSV
def test_csv_round_trip(tmp_path):
    a = inv.make_synthetic_flight(40, "silica", seed=2, measure_radii=True)
    b = inv.make_synthetic_flight(30, "graphite", seed=3, restitution_fraction=0.5)          # radii unknown
    c = inv.make_synthetic_flight(20, "silica", partner="glass_borosilicate", wall=True, seed=4)
    path = tmp_path / "impacts.csv"
    inv.write_impacts_csv(path, [a, b, c], comment="SYNTHETIC test data")
    assert path.read_text(encoding="utf-8").startswith("# SYNTHETIC test data")
    back = inv.read_impacts_csv(path)
    assert set(back) == {a.pair, b.pair, c.pair}
    for orig in (a, b, c):
        got = back[orig.pair]
        np.testing.assert_allclose(got.speed, orig.speed, rtol=1e-12)
        np.testing.assert_array_equal(got.stuck, orig.stuck)
        np.testing.assert_array_equal(np.isnan(got.restitution), np.isnan(orig.restitution))
        ok = ~np.isnan(orig.restitution)
        np.testing.assert_allclose(got.restitution[ok], orig.restitution[ok], rtol=1e-12)
        for k in ("r1", "r2"):
            g, o = getattr(got, k), getattr(orig, k)
            np.testing.assert_array_equal(np.isnan(g), np.isnan(o))
            np.testing.assert_array_equal(np.isinf(g), np.isinf(o))
            fin = np.isfinite(o)
            np.testing.assert_allclose(g[fin], o[fin], rtol=1e-12)


def test_csv_groups_a_mixed_pair_in_either_order(tmp_path):
    head = ",".join(inv.CSV_COLUMNS) + "\n"
    p = tmp_path / "mixed.csv"
    p.write_text(head + "5.0,0,0.5,45,52,silica,graphite\n6.0,1,,51,47,graphite,silica\n", encoding="utf-8")
    back = inv.read_impacts_csv(p)
    assert list(back) == ["graphite-silica"]
    g = back["graphite-silica"]
    np.testing.assert_allclose(g.r1 * 1e6, [52, 51])       # graphite radii
    np.testing.assert_allclose(g.r2 * 1e6, [45, 47])       # silica radii


def test_csv_rejects_bad_rows(tmp_path):
    head = ",".join(inv.CSV_COLUMNS) + "\n"
    for bad in ("5.0,2,,,,silica,silica\n", "5.0,1,0.4,,,silica,silica\n", "-1,0,,,,silica,silica\n",
                "5.0,0,,,,silica,unobtainium\n"):
        p = tmp_path / "bad.csv"
        p.write_text(head + bad, encoding="utf-8")
        with pytest.raises(ValueError):
            inv.read_impacts_csv(p)


def test_override_material_is_temporary():
    before = dc.get_material("silica").e_dry
    with inv.override_material("silica", e_dry=0.9) as m:
        assert m.e_dry == 0.9 and dc.make_pair("silica", 50e-6, "silica", 50e-6).e_dry == pytest.approx(0.9)
    assert dc.get_material("silica").e_dry == before

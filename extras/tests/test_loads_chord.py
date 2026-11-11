"""Loads at breakpoints (limits from the left for step ends) and the chord-deviation bounds of the loads over a step
(loads.py: accel(side), omega_dot(side), chord_dev, spin_chord)."""
import math

import numpy as np
import pytest

from dolle_sim.environment import Environment
from dolle_sim.flight import G0
from dolle_extras.loads import Loads

L_STEP = 4e-3


@pytest.fixture(scope="module")
def loads():
    """The nominal flight profile (flight.py defaults) on the standing tubule's spin axis, no release, no vibration."""
    env = Environment({"preset": "ideal", "pressure_pa": 0.0, "start_time_s": 0.0, "flight_profile": {}},
                      np.random.default_rng(0))
    return Loads(env, [], None)


def _exact_dev(L, T0, T1, n=200):
    """The largest per-axis distance, on a grid of n points, between the loads and the chord joining their values at
    T0 (from the right) and T1 (from the left); also the largest |a| on the grid."""
    g = np.linspace(T0, T1, n)
    A = np.array([L.accel(np.array([t]))[0] for t in g[:-1]] + [L.accel(np.array([g[-1]]), side="-")[0]])
    chord = A[0] + (A[-1] - A[0])[None, :] * ((g - T0) / (T1 - T0))[:, None]
    return np.abs(A - chord).max(0), float(np.abs(A).max())


def test_left_limits_at_breakpoints(loads):
    """At the second burn's end the stored load is the post-jump drag and side "-" the thrust; at de-spin onset the
    stored spin rate is the de-spin's and side "-" zero (the acceleration is continuous there). On the interpolation
    grid a segment ending at the burn end is all thrust with side "-", one starting there is drag from its second
    sample on (accel_seg: from its first). Half a second from every breakpoint side "-" changes nothing."""
    L, p = loads, loads.prof
    Tb = np.array([p.burn2[1]])
    assert L.accel(Tb)[0, 2] == pytest.approx(p.post_burn_drag_g * G0, rel=1e-9)
    assert np.array_equal(L.accel(Tb, side="+"), L.accel(Tb))
    assert L.accel(Tb, side="-")[0, 2] == pytest.approx(-p.burn_g * G0, rel=1e-12)
    assert L.omega_dot(Tb)[0, 2] == 0.0
    assert L.omega_dot(Tb, side="-")[0, 2] == pytest.approx(2 * math.pi * p.spin_burnout_hz / p.burn2[1], rel=1e-12)
    Td = np.array([p.despin_s])
    assert L.omega_dot(Td)[0, 2] == pytest.approx(
        2 * math.pi * (p.residual_spin_hz - p.spin_burnout_hz) / p.despin_duration_s, rel=1e-12)
    assert L.omega_dot(Td, side="-")[0, 2] == 0.0
    assert L.accel(Td, side="-")[0] == pytest.approx(L.accel(Td)[0], rel=1e-9)
    assert np.allclose(L.omega(Td), L.omega(Td - 1e-12), rtol=1e-12)
    Ts = np.linspace(p.burn2[1] - L_STEP, p.burn2[1], 41)
    a = L.accel(Ts, side="-")
    assert np.all(a[:, 2] == pytest.approx(-p.burn_g * G0, rel=1e-12))
    assert L.accel(Ts)[-1, 2] == pytest.approx(p.post_burn_drag_g * G0, rel=1e-9)
    Ts = np.linspace(p.burn2[1], p.burn2[1] + 5e-5, 41)
    a = L.accel(Ts, side="-")
    assert a[0, 2] == pytest.approx(-p.burn_g * G0, rel=1e-12) and np.all(a[1:, 2] > 0)
    assert np.all(L.accel_seg(Ts)[:, 2] > 0)
    for bp in L.breakpoints:
        for T in (bp - 0.5, bp + 0.5):
            if T < 0:
                continue
            one, grid = np.array([T]), np.linspace(T, T + 1e-4, 12)
            assert np.array_equal(L.accel(one, side="-"), L.accel(one))
            assert np.array_equal(L.accel(grid, side="-"), L.accel(grid))
            assert np.array_equal(L.omega_dot(one, side="-"), L.omega_dot(one))


def test_chord_dev_vanishes_on_constant_and_linear_pieces(loads):
    """Zero deviation (to rounding) on the coast between the burns and between two points of the descent table."""
    L, p = loads, loads.prof
    for T0 in (8.0, p._t[300] + 0.1):
        cd = L.chord_dev(T0, T0 + L_STEP)
        ex, amax = _exact_dev(L, T0, T0 + L_STEP)
        assert np.all(cd <= 1e-12 * amax) and np.all(ex <= cd + 1e-12 * amax)


def test_chord_dev_is_exact_on_the_splash_half_sine(loads):
    """Inside the splash pulse the bound is peak omega^2 l^2/8 (the exact curvature bound of a sine) and it bounds
    the exact chord deviation; at the pulse's peak the exact deviation comes within 0.2% of it."""
    L, p = loads, loads.prof
    peak = p.v_sea / p.splash_stop_s * 0.5 * math.pi
    ref = peak * (math.pi / p.splash_stop_s) ** 2 * L_STEP ** 2 / 8
    for T0 in (p.splash_s + 0.01, p.splash_s + 0.02, p.splash_s + 0.5 * p.splash_stop_s - L_STEP / 2):
        cd = L.chord_dev(T0, T0 + L_STEP)
        ex, _ = _exact_dev(L, T0, T0 + L_STEP)
        assert cd[2] == pytest.approx(ref, rel=1e-3) and cd[0] == 0.0 and cd[1] == 0.0
        assert np.all(ex <= cd)
    ex, _ = _exact_dev(L, p.splash_s + 0.5 * p.splash_stop_s - L_STEP / 2, p.splash_s + 0.5 * p.splash_stop_s + L_STEP / 2)
    assert ex[2] >= 0.998 * ref


def test_chord_dev_bounds_the_exact_deviation(loads):
    """50 seeded 4 ms steps over the whole flight (none straddling a breakpoint, as the engine's steps never do), a
    step containing a descent-table point (the kink term |Delta slope| l/4 of a slope jump at mid-step), one
    containing the de-spin knot, one ending and one starting at the second burn's end: the bound is never below the
    exact deviation on a 200-point grid."""
    L, p = loads, loads.prof
    rng = np.random.default_rng(1)
    steps = []
    while len(steps) < 50:
        T0 = rng.uniform(0.0, p.end_s - L_STEP)
        if not ((L.breakpoints > T0 - 1e-9) & (L.breakpoints < T0 + L_STEP + 1e-9)).any():
            steps.append(T0)
    i = 300
    steps += [p._t[i] - L_STEP / 2, p.despin_s - L_STEP / 2, p.burn2[1] - L_STEP, p.burn2[1]]
    slack = 0.0
    for T0 in steps:
        cd = L.chord_dev(T0, T0 + L_STEP)
        ex, amax = _exact_dev(L, T0, T0 + L_STEP)
        assert np.all(ex <= cd + 1e-12 * amax), (T0, cd, ex)
        slack = max(slack, float(np.max(cd - ex)))
    print(f"chord_dev: no violation on {len(steps)} steps, largest slack {slack:.3g} m/s^2")
    tt, ff = np.asarray(p._t), np.asarray(p._f)
    dslope = abs((ff[i + 1] - ff[i]) / (tt[i + 1] - tt[i]) - (ff[i] - ff[i - 1]) / (tt[i] - tt[i - 1]))
    ex, _ = _exact_dev(L, tt[i] - L_STEP / 2, tt[i] + L_STEP / 2)
    assert ex[0] == pytest.approx(dslope * L_STEP / 4, rel=1e-2)
    assert L.chord_dev(tt[i] - L_STEP / 2, tt[i] + L_STEP / 2)[0] >= dslope * L_STEP / 4


def test_spin_chord_is_the_exact_deviation_of_omega_squared(loads):
    """On the spin-up piece Omega^2 is a parabola: the bound equals its chord deviation (fine grid with the mid-point)
    to 1e-9 and the analytic Omega_dot^2 l^2/4; zero at constant spin; across the de-spin knot still a bound."""
    L, p = loads, loads.prof

    def exact(T0, T1):
        g = np.linspace(T0, T1, 2001)
        W2 = np.linalg.norm(L.omega(g), axis=1) ** 2
        return float(np.abs(W2 - (W2[0] + (W2[-1] - W2[0]) * (g - T0) / (T1 - T0))).max())
    assert L.spin_chord(10.0, 11.0) == pytest.approx(exact(10.0, 11.0), rel=1e-9)
    wd = 2 * math.pi * p.spin_burnout_hz / p.burn2[1]
    assert L.spin_chord(10.0, 10.0 + L_STEP) == pytest.approx(wd ** 2 * L_STEP ** 2 / 4, rel=1e-12)
    assert L.spin_chord(40.0, 40.0 + L_STEP) == 0.0
    wd2 = 2 * math.pi * (p.residual_spin_hz - p.spin_burnout_hz) / p.despin_duration_s
    assert L.spin_chord(p.despin_s, p.despin_s + L_STEP) == pytest.approx(wd2 ** 2 * L_STEP ** 2 / 4, rel=1e-12)
    assert L.spin_chord(p.despin_s - L_STEP, p.despin_s) == 0.0
    assert L.spin_chord(53.9, 54.1) >= exact(53.9, 54.1)


def test_release_and_vibration_terms_are_present_when_their_windows_overlap(loads):
    """A step overlapping a release window carries twice the waveform's amplitude on its axis, one overlapping a
    vibration window twice the vibration's largest value over the step; neither elsewhere."""
    from dolle_extras.vibration import VibrationHistory
    rel = [dict(start_s=90.0, end_s=90.2, omega=2 * math.pi * 200.0, peak_g=50.0, axis=2, type="shake")]
    hv = VibrationHistory(100.0, 20000.0, np.random.default_rng(0).normal(size=(3, 201)))
    L = Loads(loads.env, rel, [hv])
    assert L.chord_dev(90.1, 90.1 + L_STEP) == pytest.approx([0.0, 0.0, 2 * 50.0 * G0])
    assert L.chord_dev(100.002, 100.006) == pytest.approx(2 * hv.abs_max(100.002, 100.006))
    assert np.all(L.chord_dev(95.0, 95.0 + L_STEP) == 0.0)
    assert np.all(L.accel(np.array([90.2]), side="-") == pytest.approx(L.accel(np.array([90.2])), abs=1e-6 * 50 * G0))

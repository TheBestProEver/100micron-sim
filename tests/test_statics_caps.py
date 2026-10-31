"""statics.py at a cap: lifted caps, the kernel's return map, snapping, the released stiffness, the Newton solves."""
import math

import numpy as np
import pytest
from scipy.sparse.linalg import splu

from dolle_sim import dem, statics
from dolle_sim.materials import get_material, sphere_mass

R = 50e-6
L = np.array([1e-3, 1e-3, 1e-3])
G0 = 9.80665


def _wall_params():
    s, g = get_material("silica"), get_material("glass_borosilicate")
    m = sphere_mass(s, R)
    return dem.make_params(R, 0.0, m, math.inf, s.E, s.nu, g.E, g.nu, 2 * math.sqrt(s.gamma * g.gamma), 0.9, math.inf,
                           0.4, 0.5, 100.0, 100.0, dem.DAMP_TSUJI_JKR, dem.BREAK_DISPLACEMENT, 1.0, 0.0), m


def _pair_params():
    s = get_material("silica")
    m = sphere_mass(s, R)
    return dem.make_params(R, R, m, m, s.E, s.nu, s.E, s.nu, 2 * s.gamma, 0.9, math.inf, 0.4, 0.5, 100.0, 100.0,
                           dem.DAMP_TSUJI_JKR, dem.BREAK_DISPLACEMENT, 1.0, 0.0)


def _d0(p):
    """Zero-load JKR overlap a0^2 / (3 R_eff)."""
    return p[dem.P_A0] ** 2 / (3 * p[dem.P_REFF])


def _formed(x, ci, cj, prm):
    """Histories after one kernel evaluation at rest: the listed contacts formed (overlap >= 0), springs zero."""
    _, h = statics.apply_displacement(x, ci, cj, prm, np.zeros((len(ci), dem.NH)), L, np.zeros((len(x), 6)))
    assert (h[:, dem.H_TOUCH] > 0.5).all()
    return h


def _floor_grain():
    """One grain on the floor (face 4, cj = -5) at its zero-load overlap."""
    p, m = _wall_params()
    x = np.array([[0.5e-3, 0.5e-3, R - _d0(p)]])
    ci, cj, prm = np.array([0]), np.array([-5]), p[None, :].copy()
    return x, ci, cj, prm, _formed(x, ci, cj, prm), m


def _corner_grain():
    """One grain in the edge between the -x wall (face 0, cj = -1) and the floor (face 4, cj = -5)."""
    p, m = _wall_params()
    d = _d0(p)
    x = np.array([[R - d, 0.5e-3, R - d]])
    ci, cj, prm = np.array([0, 0]), np.array([-1, -5]), np.vstack([p, p])
    return x, ci, cj, prm, _formed(x, ci, cj, prm), m


def _chain(n):
    """n grains in a row on the floor, each touching the floor and its neighbours, at zero-load overlaps."""
    pw, m = _wall_params()
    pp = _pair_params()
    dw, dp = _d0(pw), _d0(pp)
    x = np.array([[0.3e-3 + k * (2 * R - dp), 0.5e-3, R - dw] for k in range(n)])
    ci = np.array(list(range(n)) + list(range(n - 1)))
    cj = np.array([-5] * n + list(range(1, n)))
    prm = np.vstack([pw] * n + [pp] * (n - 1))
    return x, ci, cj, prm, _formed(x, ci, cj, prm), np.full(n, m)


def _set_rolling(h, k, prm, lam, out, frac, direction):
    """Rolling history of contact k such that its spring force is frac of the cap along direction (unit, tangential)."""
    cap = statics.caps(lam, prm, out)[k, 1]
    h[k, dem.H_XR:dem.H_XR + 3] = -frac * cap / prm[k, dem.P_KR] * np.asarray(direction, float)


def _roll_du(dxr, n):
    """Displacement (6,) of a grain rolling without slip on a wall of normal n by the rolling increment dxr
    (xi_r += R dtheta x n; the centre moves with the contact point): dtheta = n x dxr / R, dx = R dtheta x n."""
    th = np.cross(n, dxr) / R
    return np.concatenate([R * np.cross(th, n), th])


def _tols(lam, prm, out):
    """Per-DOF tolerances: 1e-6 of the smallest force cap and of the smallest torque cap (twist, or R times rolling)."""
    cp = statics.caps(lam, prm, out)
    return 1e-6 * float(cp[:, :2].min()), 1e-6 * float(min(cp[:, 2].min(), (prm[:, dem.P_REFF] * cp[:, 1]).min()))


def test_lifted_kernel_is_bit_identical_below_the_caps():
    """With every spring at half its cap the real and the lifted kernel take the same branches: state() and
    apply_displacement() agree bit for bit, so the lifted tangents and solves describe the real kernel exactly."""
    x, ci, cj, prm, h, m = _corner_grain()
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    cp = statics.caps(lam, prm, out)
    k = statics.kcomp(prm, h)
    for q, (ts, tr) in enumerate([((0, 1, 0), (0, 0, 1)), ((1, 0, 0), (0, 1, 0))]):   # tangential to n = x, n = z
        h[q, dem.H_XT:dem.H_XT + 3] = -0.5 * cp[q, 0] / k[q, 0] * np.array(ts, float)
        h[q, dem.H_XR:dem.H_XR + 3] = -0.5 * cp[q, 1] / k[q, 1] * np.array(tr, float)
        h[q, dem.H_XTW] = 0.5 * cp[q, 2] / k[q, 2]
    prm_el = statics.lift_caps(prm)
    a, b = statics.state(x, ci, cj, prm, h, L), statics.state(x, ci, cj, prm_el, h, L)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert np.abs(statics.margins(a[0], prm, a[2]) - 0.5).max() < 1e-12
    dU = np.array([[1e-12, -2e-12, 0.5e-12, 1e-8, -2e-8, 3e-8]])
    a, b = (statics.apply_displacement(x, ci, cj, p, h, L, dU, with_out=True) for p in (prm, prm_el))
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert not np.array_equal(a[1], h)


def test_lifted_tangent_is_elastic_at_a_cap_and_the_real_one_is_kinked():
    """A rolling spring at its cap: the lifted jacobian rows of the rolling force are -k_r times the kinematic rows
    (finite differences of the lifted histories, 1e-6), while the real-parameter central differences straddle the
    clip and lose half the stiffness along the force direction (>= 25% off); across it they agree to 1e-5."""
    x, ci, cj, prm, h, m = _floor_grain()
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    fhat = np.array([math.cos(0.3), math.sin(0.3), 0.0])
    _set_rolling(h, 0, prm, lam, out, 1.0, fhat)
    h, _ = statics.snap_to_caps(h, lam, prm, out)
    prm_el = statics.lift_caps(prm)
    K, J = statics.jacobians(x, np.array([R]), ci, cj, prm_el, h, L)
    Jr = J.toarray()[4:7]
    B = np.zeros((3, 6))
    for d in range(6):
        eps = statics.EPS_X if d < 3 else statics.EPS_X / R
        dU = np.zeros((1, 6))
        dU[0, d] = eps
        hp = statics.apply_displacement(x, ci, cj, prm_el, h, L, dU)[1]
        dU[0, d] = -eps
        hm = statics.apply_displacement(x, ci, cj, prm_el, h, L, dU)[1]
        B[:, d] = (hp[0, dem.H_XR:dem.H_XR + 3] - hm[0, dem.H_XR:dem.H_XR + 3]) / (2 * eps)
    kr = prm[0, dem.P_KR]
    assert np.linalg.norm(Jr + kr * B) <= 1e-6 * np.linalg.norm(Jr)
    Jreal = statics.jacobians(x, np.array([R]), ci, cj, prm, h, L)[1].toarray()[4:7]
    radial_el, radial_re = fhat @ Jr[:, 3:], fhat @ Jreal[:, 3:]
    assert np.linalg.norm(radial_re - radial_el) >= 0.25 * np.linalg.norm(radial_el)
    perp = np.array([-fhat[1], fhat[0], 0.0])
    # across the force both agree, up to the clip rescaling the whole trial vector at +eps: first order in the
    # finite-difference force step over the cap, k_r EPS_X / cap ~ 6e-6 (measured 1.3e-6)
    assert np.linalg.norm(perp @ Jreal[:, 3:] - perp @ Jr[:, 3:]) <= 1e-5 * np.linalg.norm(perp @ Jr[:, 3:])


def test_real_state_reads_an_over_cap_twist_as_zero_and_snapping_fixes_it():
    """The real kernel at rest resets an over-cap twist to 0 (contact_force: sg = sign(mt) = 0 at zero rate), so
    state() with the real parameters reports 0 where the lifted state() reads the twist as it is; after snap_to_caps
    the real state() gives (1 - 2^-40) of the cap with the sign kept, and one more evaluation at rest (statics, and
    the DEM kernel itself as at the first force evaluation after a wake) leaves the history within 1e-11 relative."""
    x, ci, cj, prm, h, m = _floor_grain()
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    cap = statics.caps(lam, prm, out)[0, 2]
    ktw = prm[0, dem.P_KTW]
    prm_el = statics.lift_caps(prm)
    for sign in (1.0, -1.0):
        hc = h.copy()
        hc[0, dem.H_XTW] = sign * 1.5 * cap / ktw
        lam_re = statics.state(x, ci, cj, prm, hc, L)[0]
        lam_el = statics.state(x, ci, cj, prm_el, hc, L)[0]
        assert lam_re[7] == 0.0
        assert lam_el[7] == pytest.approx(-sign * 1.5 * cap, rel=1e-12)
        hs, mrel = statics.snap_to_caps(hc, lam_el, prm, out)
        assert mrel == pytest.approx(1 - (1 - 2 ** -40) / 1.5, rel=1e-6)
        assert np.sign(hs[0, dem.H_XTW]) == sign
        lam_s = statics.state(x, ci, cj, prm, hs, L)[0]
        assert lam_s[7] == pytest.approx(-sign * (1 - 2 ** -40) * cap, rel=1e-13)
        hn = statics.apply_displacement(x, ci, cj, prm, hs, L, np.zeros((1, 6)))[1]
        assert hn[0, dem.H_XTW] == pytest.approx(hs[0, dem.H_XTW], rel=1e-11)
        hd = hs[0].copy()
        dxv = np.zeros(3)
        r = dem._wall_dx(x[0], L, 4, dxv)
        o = np.zeros(dem.NOUT)
        z = np.zeros(3)
        dem.contact_force(dxv, r, z, z, z, z, prm[0], hd, 5e-9, o, 0.0)
        assert hd[dem.H_XTW] == pytest.approx(hs[0, dem.H_XTW], rel=1e-11)
        assert o[18] == pytest.approx(-sign * (1 - 2 ** -40) * cap, rel=1e-13)
    # a twist within round-off of its cap is snapped too, so no evaluation can ever clip it
    hc = h.copy()
    hc[0, dem.H_XTW] = (1 - 1e-15) * cap / ktw
    hs, mrel = statics.snap_to_caps(hc, lam, prm, out)
    assert 0 < mrel < 1e-12
    assert statics.state(x, ci, cj, prm, hs, L)[0][7] == pytest.approx(-(1 - 2 ** -40) * cap, rel=1e-13)
    assert statics.snap_to_caps(h, lam, prm, out)[1] == 0.0


def test_apply_displacement_is_the_kernel_return_map():
    """A capped rolling spring given an outward rolling increment (without slip) while the normal force, hence the
    cap, changes: the real kernel clips the history once, to |k_r xi_r + g_r v_l| = mu_r |F_ne + 2 F_pull| with the
    NEW normal force to 1e-12 (the dashpot shift g_r v_l / k_r is 2e-10 of the cap for a 1e-8 m increment, below
    1e-12 for creep-sized ones); the lifted kernel does not clip (its force grows by k_r times the increment)."""
    x, ci, cj, prm, h, m = _floor_grain()
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    fhat = np.array([0.0, 1.0, 0.0])
    _set_rolling(h, 0, prm, lam, out, 1.0, fhat)
    h, _ = statics.snap_to_caps(h, lam, prm, out)
    kr, gr = prm[0, dem.P_KR], prm[0, dem.P_GR]
    n = np.array([0.0, 0.0, 1.0])
    # (a) a large outward increment with a push into the floor (F_ne, hence the cap, rises): the exact return map,
    # including the dashpot shift g_r v_l / k_r of the clipped history (2e-10 of the cap here; v_l = dxi_r / TAU)
    dxr = -1e-8 * fhat                                  # xi_r points along -fhat: this increment is outward
    dU = _roll_du(dxr, n).reshape(1, 6)
    dU[0, 2] = -2e-10
    xn, hr, outr = statics.apply_displacement(x, ci, cj, prm, h, L, dU, with_out=True)
    xe, he, oute = statics.apply_displacement(x, ci, cj, statics.lift_caps(prm), h, L, dU, with_out=True)
    assert outr[0, 11] > lam[0] + 1e-7 and outr[0, 11] == oute[0, 11]
    cap_new = prm[0, dem.P_MUR] * abs(outr[0, 11] + 2 * outr[0, 19])
    fe = kr * np.linalg.norm(he[0, dem.H_XR:dem.H_XR + 3])
    assert fe > 1.1 * cap_new                           # the elastic trial is well outside: the real kernel clipped
    assert fe == pytest.approx(kr * np.linalg.norm(h[0, dem.H_XR:dem.H_XR + 3] + dxr), rel=1e-9)
    vl = dxr / statics.TAU
    assert np.linalg.norm(kr * hr[0, dem.H_XR:dem.H_XR + 3] + gr * vl) == pytest.approx(cap_new, rel=1e-12)
    assert kr * np.linalg.norm(hr[0, dem.H_XR:dem.H_XR + 3]) == pytest.approx(cap_new, rel=1e-9)
    assert np.abs(hr[0, dem.H_XT:dem.H_XT + 3]).max() < 1e-20     # rolling without slip: sliding spring untouched
    # (b) a creep-sized increment (2e-11 m, 1.3e-3 of the cap) while a lift lowers F_ne and shrinks the cap: the
    # clipped spring sits on the NEW cap to 1e-12 (the dashpot shift is 6e-13 of it); the lifted kernel does not clip
    dxr = -2e-11 * fhat
    dU = _roll_du(dxr, n).reshape(1, 6)
    dU[0, 2] = 1e-10
    xn, hr, outr = statics.apply_displacement(x, ci, cj, prm, h, L, dU, with_out=True)
    he = statics.apply_displacement(x, ci, cj, statics.lift_caps(prm), h, L, dU)[1]
    cap_new = prm[0, dem.P_MUR] * abs(outr[0, 11] + 2 * outr[0, 19])
    assert cap_new < 0.97 * statics.caps(lam, prm, out)[0, 1]
    assert kr * np.linalg.norm(he[0, dem.H_XR:dem.H_XR + 3]) > 1.03 * cap_new
    assert kr * np.linalg.norm(hr[0, dem.H_XR:dem.H_XR + 3]) == pytest.approx(cap_new, rel=1e-12)


def test_released_stiffness_is_the_clipped_tangent_and_shows_mechanisms():
    """One capped rolling spring: the lifted stiffness minus the rank-1 outward term equals the forward difference of
    the clipped kernel for a 1e-14 m increment with outward and turning parts (1e-6 of the elastic response; the
    second-order cross term of the clip is 4e-7 of it at this step and grows with it); along the pure
    rolling motion nothing then holds the grain (the released stiffness keeps <= 1e-6 of the elastic one), and
    energy_bounds on it either finds the system not positive definite (None) or returns a bound >= 1e3 times the
    elastic one, a mechanism either way (the singularity is exact only to the finite-difference rounding); a corner
    grain with two capped rolling springs in different directions stays positive definite, bounds within 1.5x."""
    x, ci, cj, prm, h, m = _floor_grain()
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    fhat = np.array([math.cos(1.1), math.sin(1.1), 0.0])
    # exactly at the cap, not snapped: the 2^-40 gap of a snapped state would offset the forward difference below by
    # 2^-40 cap / (k_r |du|) = 1.2e-6 of it at this step size
    _set_rolling(h, 0, prm, lam, out, 1.0, fhat)
    prm_el = statics.lift_caps(prm)
    lam, Fnet0, out = statics.state(x, ci, cj, prm_el, h, L)
    K, J = statics.jacobians(x, np.array([R]), ci, cj, prm_el, h, L)
    kc = statics.kcomp(prm, h)
    mask = np.zeros((1, 3), bool)
    mask[0, 1] = True
    fh = np.zeros((1, 3, 3))
    fh[0, 1] = fhat
    Krel = statics.released_stiffness(K, J, kc, mask, fh)
    assert Krel.shape == K.shape and Krel.format == "csc"
    Kd, Kr = K.toarray(), Krel.toarray()
    v = fhat @ J.toarray()[4:7]
    assert np.allclose(Kr, Kd - np.outer(v, v) / kc[0, 1], rtol=1e-12, atol=0)
    n = np.array([0.0, 0.0, 1.0])
    perp = np.array([-fhat[1], fhat[0], 0.0])
    du = _roll_du(1e-14 * (-fhat) + 0.7e-14 * perp, n)
    xn, hn = statics.apply_displacement(x, ci, cj, prm, h, L, du.reshape(1, 6))
    dF = statics.state(xn, ci, cj, prm_el, hn, L)[1] - Fnet0
    dF_el, dF_rel = -(Kd @ du), -(Kr @ du)
    assert np.linalg.norm(dF - dF_rel) <= 1e-6 * np.linalg.norm(dF_el)
    assert np.linalg.norm(dF - dF_el) > 0.3 * np.linalg.norm(dF_el)
    u = _roll_du(-fhat, n)
    assert u @ Kr @ u <= 1e-6 * (u @ Kd @ u)
    eb_el = statics.energy_bounds(K, J, 1e-20)
    eb = statics.energy_bounds(Krel, J, 1e-20)
    assert eb is None or not np.isfinite(eb).all() or eb.max() > 1e3 * eb_el.max()
    # nothing flagged: K_el itself
    assert (statics.released_stiffness(K, J, kc, np.zeros((1, 3), bool), fh) != K).nnz == 0
    x, ci, cj, prm, h, m = _corner_grain()
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    dirs = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    for q in range(2):
        _set_rolling(h, q, prm, lam, out, 1.0, dirs[q])
    h, _ = statics.snap_to_caps(h, lam, prm, out)
    prm_el = statics.lift_caps(prm)
    K, J = statics.jacobians(x, np.array([R]), ci, cj, prm_el, h, L)
    mask = np.zeros((2, 3), bool)
    mask[:, 1] = True
    fh = np.zeros((2, 3, 3))
    fh[:, 1] = dirs
    Krel = statics.released_stiffness(K, J, statics.kcomp(prm, h), mask, fh)
    eb_el, eb = statics.energy_bounds(K, J, 1e-20), statics.energy_bounds(Krel, J, 1e-20)
    assert eb is not None and np.isfinite(eb).all()
    assert (eb <= 1.5 * eb_el + 1e-300).all() and (eb >= eb_el * (1 - 1e-9)).all()


def test_solve_equilibrium_elastic_converges_in_three_evaluations():
    """Two and three grains on the floor under 1 g from their zero-load overlaps: the lifted Newton solve from the
    linear prediction meets the per-DOF tolerances (1e-6 of the smallest caps) within 3 kernel evaluations with no
    contact broken, formed or yielded, and the real and lifted kernels agree bit for bit at the solution."""
    for n in (2, 3):
        x, ci, cj, prm, h, m = _chain(n)
        prm_el = statics.lift_caps(prm)
        lam, Fnet, out = statics.state(x, ci, cj, prm_el, h, L)
        K, J = statics.jacobians(x, np.full(n, R), ci, cj, prm_el, h, L)
        lu = splu(K)
        Fext = np.zeros((n, 6))
        Fext[:, 2] = -m * G0
        Fext = Fext.reshape(-1)
        du0 = lu.solve(Fext + Fnet)
        tt, tr = _tols(lam, prm, out)
        E = statics.solve_equilibrium(x, ci, cj, prm_el, prm_el, h, L, lu, Fext, du0, tt, tr, 6)
        assert E["ok"] and E["n_it"] <= 3
        assert not (E["broke"] or E["formed"] or E["yielded"] or E["mechanism"])
        r6 = np.abs(E["r"].reshape(n, 6))
        assert r6[:, :3].max() <= tt and r6[:, 3:].max() <= tr
        assert np.array_equal(E["r"], Fext + E["Fnet"])
        a = statics.state(E["xn"], ci, cj, prm, E["hn"], L)
        b = statics.state(E["xn"], ci, cj, prm_el, E["hn"], L)
        assert all(np.array_equal(u, v) for u, v in zip(a, b))
        assert np.array_equal(a[0], E["lam"])
        assert (statics.margins(E["lam"], prm, E["out"]) < 0.5).all()
        assert (statics.overlaps(E["xn"], ci, cj, prm, L) > 0).all()


def test_solve_equilibrium_creeps_with_the_released_tangent_and_reports_mechanisms_and_transitions():
    """A corner grain whose floor rolling spring sits at its cap, balanced by an external load: with the demand on
    that spring raised by 1% (through the linear response) the real-kernel return map inside the Newton loop, with
    the released stiffness as tangent, converges within 10 evaluations and leaves |k_r xi_r| equal to the new cap to
    1e-9 while the lifted solve just overloads the spring; the same spring on a grain held by the floor alone is a
    mechanism (the elastic fixed point does not contract: mechanism True; the released tangent, singular only to
    rounding, throws the grain off the wall or fails: never ok); lifting a grain off the floor returns broke, and a
    closing gap between two grains returns formed."""
    x, ci, cj, prm, h, m = _corner_grain()
    prm_el = statics.lift_caps(prm)
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    _set_rolling(h, 1, prm, lam, out, 1.0, np.array([0.0, 1.0, 0.0]))
    h, _ = statics.snap_to_caps(h, lam, prm, out)
    lam, Fnet, out = statics.state(x, ci, cj, prm_el, h, L)
    Fext0 = -Fnet                                        # the load this state balances exactly
    K, J = statics.jacobians(x, np.array([R]), ci, cj, prm_el, h, L)
    lu = splu(K)
    kc = statics.kcomp(prm, h)
    cap0 = statics.caps(lam, prm, out)[1, 1]
    tt, tr = _tols(lam, prm, out)
    E0 = statics.solve_equilibrium(x, ci, cj, prm, prm_el, h, L, lu, Fext0, np.zeros(6), tt, tr, 6)
    assert E0["ok"] and E0["n_it"] == 1
    # a torque about the line through the two contact points (no slip at either): 1% more demand on the floor spring
    dF1 = np.zeros(6)
    dF1[3:] = np.array([1.0, 0.0, -1.0]) / math.sqrt(2)
    dlam = (J @ lu.solve(dF1)).reshape(2, 8)
    dF = dF1 * 0.01 * cap0 / abs(dlam[1, 5])
    Fext = Fext0 + dF
    du0 = lu.solve(dF)
    El = statics.solve_equilibrium(x, ci, cj, prm_el, prm_el, h, L, lu, Fext, du0, tt, tr, 6)
    assert El["ok"] and El["n_it"] <= 3
    assert statics.ratios_by_kind(El["lam"], prm, El["out"])[2][1] == pytest.approx(1.01, abs=2e-3)

    def tangent(xn, hn, lam_n, out_n):
        rp, rs, rr, rt = statics.ratios_by_kind(lam_n, prm, out_n)
        L8 = lam_n.reshape(-1, 8)
        mask = np.column_stack([rs, rr, rt]) >= 1 - 1e-9
        fh = np.zeros((len(prm), 3, 3))
        fh[:, 0] = L8[:, 1:4] / np.maximum(np.linalg.norm(L8[:, 1:4], axis=1), 1e-300)[:, None]
        fh[:, 1] = L8[:, 4:7] / np.maximum(np.linalg.norm(L8[:, 4:7], axis=1), 1e-300)[:, None]
        return splu(statics.released_stiffness(K, J, kc, mask, fh))
    C = statics.solve_equilibrium(x, ci, cj, prm, prm_el, h, L, lu, Fext, du0, tt, tr, 40, tangent=tangent)
    assert C["ok"] and C["n_it"] <= 10 and not C["mechanism"]
    cap_new = statics.caps(C["lam"], prm, C["out"])[1, 1]
    assert kc[1, 1] * np.linalg.norm(C["hn"][1, dem.H_XR:dem.H_XR + 3]) == pytest.approx(cap_new, rel=1e-9)
    assert statics.ratios_by_kind(C["lam"], prm, C["out"])[2][1] == pytest.approx(1.0, rel=1e-9)
    hn_el = statics.apply_displacement(x, ci, cj, prm_el, h, L, C["du"].reshape(-1, 6))[1]
    dp = statics.flow_increments(hn_el, C["hn"])
    assert dp[1, 1] > 0 and dp[0, 1] == 0 and (dp[:, [0, 2]] == 0).all()
    assert dp[1, 1] < 0.02 * cap0 / kc[1, 1]
    assert np.abs(C["du"][:3]).max() < 1e-9 and np.abs(C["du"][3:]).max() < 1e-5     # a creep-sized motion
    # one contact at its cap: a mechanism
    x, ci, cj, prm, h, m = _floor_grain()
    prm_el = statics.lift_caps(prm)
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    _set_rolling(h, 0, prm, lam, out, 1.0, np.array([1.0, 0.0, 0.0]))
    h, _ = statics.snap_to_caps(h, lam, prm, out)
    lam, Fnet, out = statics.state(x, ci, cj, prm_el, h, L)
    K, J = statics.jacobians(x, np.array([R]), ci, cj, prm_el, h, L)
    lu = splu(K)
    kc = statics.kcomp(prm, h)
    Fext0 = -Fnet
    Fext = Fext0.copy()
    Fext[3:] *= 1.01
    du0 = lu.solve(Fext - Fext0)
    tt, tr = _tols(lam, prm, out)
    M = statics.solve_equilibrium(x, ci, cj, prm, prm_el, h, L, lu, Fext, du0, tt, tr, 40)
    assert not M["ok"] and M["mechanism"] and M["n_it"] == 3
    M2 = statics.solve_equilibrium(x, ci, cj, prm, prm_el, h, L, lu, Fext, du0, tt, tr, 40, tangent=tangent)
    assert not M2["ok"] and (M2["mechanism"] or M2["broke"])
    # a lift-off breaks the contact; a closing gap forms one
    B = statics.solve_equilibrium(x, ci, cj, prm, prm_el, h, L, lu, Fext0, np.array([0, 0, 1e-6, 0, 0, 0.0]), tt, tr, 6)
    assert not B["ok"] and B["broke"] and not B["formed"] and B["n_it"] == 1
    pw, m = _wall_params()
    pp = _pair_params()
    x = np.array([[0.3e-3, 0.5e-3, R - _d0(pw)], [0.3e-3 + 2 * R + 1e-9, 0.5e-3, R - _d0(pw)]])
    ci, cj, prm = np.array([0, 1, 0]), np.array([-5, -5, 1]), np.vstack([pw, pw, pp])
    h = np.vstack([_formed(x[:1], ci[:1], cj[:1], prm[:1]), _formed(x[1:2], ci[1:2], cj[1:2], prm[1:2]),
                   np.zeros((1, dem.NH))])
    prm_el = statics.lift_caps(prm)
    lam, Fnet, out = statics.state(x, ci, cj, prm_el, h, L)
    K, J = statics.jacobians(x, np.full(2, R), ci, cj, prm_el, h, L)
    du0 = np.zeros(12)
    du0[6] = -2e-9
    F = statics.solve_equilibrium(x, ci, cj, prm, prm_el, h, L, splu(K), -Fnet, du0, 1e-14, 1e-19, 6)
    assert not F["ok"] and F["formed"] and not F["broke"] and F["hn"][2, dem.H_TOUCH] > 0.5


def test_row_norms_bound_the_linear_response():
    """rn_k ||f||_2 >= |(J K^-1 f)_k| for random force vectors (Cauchy-Schwarz), with equality for f along
    K^-T J_k^T (1e-9), blocks of any size giving the same norms."""
    x, ci, cj, prm, h, m = _chain(2)
    prm_el = statics.lift_caps(prm)
    K, J = statics.jacobians(x, np.full(2, R), ci, cj, prm_el, h, L)
    lu = splu(K)
    rn = statics.row_norms(lu, J, block=5)
    assert np.array_equal(rn, statics.row_norms(lu, J))
    assert rn.shape == (J.shape[0],) and (rn >= 0).all() and rn.max() > 0
    rng = np.random.default_rng(3)
    for _ in range(20):
        f = rng.normal(size=12) * np.tile([1e-9] * 3 + [1e-14] * 3, 2)
        resp = np.abs(J @ lu.solve(f))
        assert (resp <= rn * np.linalg.norm(f) * (1 + 1e-9) + 1e-300).all()
    Jt = J.T.toarray()
    for k in range(J.shape[0]):
        if rn[k] == 0:
            continue
        f = lu.solve(Jt[:, k], trans='T')
        assert abs((J @ lu.solve(f))[k]) == pytest.approx(rn[k] * np.linalg.norm(f), rel=1e-9)


def test_flow_increments_and_twist_ambiguity():
    """flow_increments is exactly zero when no spring is clipped and equals the clipped excess over k where one is;
    twist_ambiguous flags an over-cap trial twist only for a zero or opposite-sign increment."""
    x, ci, cj, prm, h, m = _floor_grain()
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    prm_el = statics.lift_caps(prm)
    n = np.array([0.0, 0.0, 1.0])
    _set_rolling(h, 0, prm, lam, out, 0.5, np.array([1.0, 0.0, 0.0]))
    du = _roll_du(np.array([-1e-10, 0.0, 0.0]), n).reshape(1, 6)        # outward but still under the cap
    he = statics.apply_displacement(x, ci, cj, prm_el, h, L, du)[1]
    hr = statics.apply_displacement(x, ci, cj, prm, h, L, du)[1]
    assert np.array_equal(he, hr) and not np.array_equal(he, h)
    assert (statics.flow_increments(he, hr) == 0).all()
    _set_rolling(h, 0, prm, lam, out, 1.0, np.array([1.0, 0.0, 0.0]))
    h, _ = statics.snap_to_caps(h, lam, prm, out)
    du = _roll_du(np.array([-1e-8, 0.0, 0.0]), n).reshape(1, 6)
    he = statics.apply_displacement(x, ci, cj, prm_el, h, L, du, with_out=True)
    hr = statics.apply_displacement(x, ci, cj, prm, h, L, du, with_out=True)
    cap_new = prm[0, dem.P_MUR] * abs(he[2][0, 11] + 2 * he[2][0, 19])
    kr = prm[0, dem.P_KR]
    dp = statics.flow_increments(he[1], hr[1])
    assert dp.shape == (1, 3) and dp[0, 0] == 0 and dp[0, 2] == 0
    assert dp[0, 1] == pytest.approx((kr * np.linalg.norm(he[1][0, dem.H_XR:dem.H_XR + 3]) - cap_new) / kr, rel=1e-9)
    cp = statics.caps(lam, prm, out)
    ktw = prm[0, dem.P_KTW]
    h_ref = h.copy()
    h_ref[0, dem.H_XTW] = 0.999 * cp[0, 2] / ktw
    caps_new = cp.copy()
    caps_new[:, 2] *= 0.99                                   # the cap shrank under the twist
    for inc, expect in ((0.0, True), (-1e-9 * cp[0, 2] / ktw, True), (1e-9 * cp[0, 2] / ktw, False)):
        he = h_ref.copy()
        he[0, dem.H_XTW] += inc
        assert bool(statics.twist_ambiguous(h_ref, he, prm, caps_new)[0]) is expect
    assert not statics.twist_ambiguous(h_ref, h_ref, prm, cp)[0]               # not over the cap: never ambiguous
    h_neg = h_ref.copy()
    h_neg[0, dem.H_XTW] *= -1
    he = h_neg.copy()
    he[0, dem.H_XTW] += 1e-9 * cp[0, 2] / ktw                              # opposite to a negative twist
    assert statics.twist_ambiguous(h_neg, he, prm, caps_new)[0]


def test_overlaps_caps_kcomp_and_ratios_are_the_kernel_quantities():
    """overlaps() for walls of both parities and a pair; caps(), kcomp() and ratios_by_kind() reproduce margins()
    and the kernel's own spring constants (Wada, Mindlin and yielded branches); the bounded ratios reproduce
    margins_bounded()."""
    pw, m = _wall_params()
    pp = _pair_params()
    x = np.array([[0.3e-3, 0.5e-3, R - 2e-10], [0.3e-3 + 2 * R - 3e-10, 0.5e-3, R - 2e-10],
                  [0.7e-3, 0.5e-3, L[2] - R + 4e-10]])
    ci, cj = np.array([0, 1, 2, 0, 1]), np.array([-5, -5, -6, 1, 2])
    prm = np.vstack([pw, pw, pw, pp, pp])
    d = statics.overlaps(x, ci, cj, prm, L)
    assert d[:4] == pytest.approx([2e-10, 2e-10, 4e-10, 3e-10], rel=1e-6)
    assert d[4] < -1e-4
    x, ci, cj, prm, h, m = _corner_grain()
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    cp = statics.caps(lam, prm, out)
    k = statics.kcomp(prm, h)
    assert np.array_equal(k[:, 0], prm[:, dem.P_KT] * prm[:, dem.P_A0])
    assert np.array_equal(k[:, 1], prm[:, dem.P_KR]) and np.array_equal(k[:, 2], prm[:, dem.P_KTW])
    for q, (ts, tr) in enumerate([((0, 1, 0), (0, 0, 1)), ((1, 0, 0), (0, 1, 0))]):
        h[q, dem.H_XT:dem.H_XT + 3] = -0.3 * cp[q, 0] / k[q, 0] * np.array(ts, float)
        h[q, dem.H_XR:dem.H_XR + 3] = -0.7 * cp[q, 1] / k[q, 1] * np.array(tr, float)
        h[q, dem.H_XTW] = (0.2 + 0.5 * q) * cp[q, 2] / k[q, 2]
    lam, _, out = statics.state(x, ci, cj, prm, h, L)
    rk = statics.ratios_by_kind(lam, prm, out)
    assert np.array_equal(np.maximum.reduce(list(rk)), statics.margins(lam, prm, out))
    assert rk[1] == pytest.approx([0.3, 0.3], rel=1e-12) and rk[2] == pytest.approx([0.7, 0.7], rel=1e-12)
    assert rk[3] == pytest.approx([0.2, 0.7], rel=1e-12)
    L8 = lam.reshape(2, 8)
    assert np.linalg.norm(L8[:, 1:4], axis=1) == pytest.approx(rk[1] * cp[:, 0], rel=1e-12)
    assert np.abs(L8[:, 7]) == pytest.approx(rk[3] * cp[:, 2], rel=1e-12)
    dlam = np.abs(lam) * 0.1 + 1e-12
    rb = statics.ratios_by_kind_bounded(lam, dlam, prm, out)
    assert np.array_equal(np.maximum.reduce(list(rb)), statics.margins_bounded(lam, dlam, prm, out))
    assert (np.maximum.reduce(list(rb)) > np.maximum.reduce(list(rk))).all()
    # the Mindlin and yielded sliding springs, read off the kernel's own force
    prm_m = prm.copy()
    prm_m[:, dem.P_TMODE] = dem.TANG_MINDLIN
    lam_m, _, out_m = statics.state(x, ci, cj, prm_m, h, L)
    fs = np.linalg.norm(lam_m.reshape(2, 8)[:, 1:4], axis=1)
    xt = np.linalg.norm(h[:, dem.H_XT:dem.H_XT + 3], axis=1)
    assert fs / xt == pytest.approx(statics.kcomp(prm_m, h)[:, 0], rel=1e-12)
    assert statics.kcomp(prm_m, h)[0, 0] == pytest.approx(prm[0, dem.P_KT] * h[0, dem.H_A], rel=1e-15)
    hy = h.copy()
    hy[:, dem.H_YLD] = 1.0
    hy[:, dem.H_RP] = 2 * R
    hy[:, dem.H_DP] = 0.0
    hy[:, dem.H_FCP] = 1.5 * math.pi * prm[:, dem.P_W] * 2 * R
    lam_y, _, out_y = statics.state(x, ci, cj, prm, hy, L)
    fs = np.linalg.norm(lam_y.reshape(2, 8)[:, 1:4], axis=1)
    assert fs / xt == pytest.approx(statics.kcomp(prm, hy)[:, 0], rel=1e-12)
    assert statics.kcomp(prm, hy)[0, 0] == pytest.approx(
        prm[0, dem.P_KT] * (4.5 * math.pi * prm[0, dem.P_W] * (2 * R) ** 2 / prm[0, dem.P_ESTAR]) ** (1 / 3), rel=1e-15)
    assert np.array_equal(out_y[:, 19], hy[:, dem.H_FCP])
    assert statics.lift_caps(prm)[:, [dem.P_MUT, dem.P_MUR, dem.P_MUTW]].min() == 1e300
    assert np.array_equal(np.delete(statics.lift_caps(prm), [dem.P_MUT, dem.P_MUR, dem.P_MUTW], axis=1),
                          np.delete(prm, [dem.P_MUT, dem.P_MUR, dem.P_MUTW], axis=1))

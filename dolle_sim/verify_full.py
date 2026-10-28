"""Checks of the v0.6 full-contact engine (dem.py, statics.py, loads.py, vibration.py, engine.py).

Each check states what it compares, against what, and the pass criterion. They need numba (desktop); the browser app
runs only the checks in verification.py. Run them all with: python -m dolle_sim verify
"""
from __future__ import annotations

import math

import numpy as np

from . import dem, statics
from .materials import get_material, pair_props, sphere_mass


def _res(key, name, what, value, criterion, ok):
    return {"id": key, "name": name, "what": what, "value": value, "criterion": criterion, "passed": bool(ok),
            "details": "", "suite": "full-contact engine (desktop, numba)"}


def _pair_system(mat, R, x, v, w, **kw):
    return dem.DEMSystem(np.asarray(x, float), np.asarray(v, float), np.asarray(w, float), R, mat.rho, mat.E,
                         mat.nu, mat.gamma, kw.pop("e_dry", mat.e_dry), kw.pop("p_y", mat.yield_pressure),
                         kw.pop("L", [1.0, 1.0, 1.0]), kw.pop("wall", {}), walls=kw.pop("walls", False), **kw)


def _collide(mat, R, v, dt, **kw):
    gap = 0.02 * R
    x = np.array([[0, 0, 0], [2 * R + gap, 0, 0]], float) + 0.5
    S = _pair_system(mat, R, x, [[v / 2, 0, 0], [-v / 2, 0, 0]], np.zeros((2, 3)), skin=0.5 * R, **kw)
    S.run(dt, int(gap / v / dt) + 1)
    for _ in range(400):
        S.run(dt, 2000)
        on = S.hist[:, dem.H_TOUCH].max() > 0.5 if len(S.hist) else False
        if not on and np.linalg.norm(S.x[1] - S.x[0]) > 2 * R:
            return (S.v[1, 0] - S.v[0, 0]) / v
    return 0.0


def check_dem_vs_1d():
    """Head-on impacts in the 3-D DEM against the independent 1-D JKR integrator of contact.py (same physics)."""
    from .contact import JKRContact
    worst = 0.0
    vals = []
    for name in ("silica", "graphite"):
        mat = get_material(name)
        R = 50e-6
        p = pair_props(mat, R, mat, R)
        meff = sphere_mass(mat, R) / 2
        ref = JKRContact(p, steps=2000).threshold(meff, tol=1e-5)
        dt = 2 * math.pi * math.sqrt(meff / (2 * p.E_star * p.a0)) / 400
        lo, hi = 0.5 * ref, 2 * ref
        for _ in range(16):
            mid = math.sqrt(lo * hi)
            if _collide(mat, R, mid, dt) > 0:
                hi = mid
            else:
                lo = mid
        th = math.sqrt(lo * hi)
        worst = max(worst, abs(th / ref - 1))
        vals.append(f"{name} {th * 1e3:.3f} vs {ref * 1e3:.3f} mm/s")
    return _res("dem_vs_1d", "3-D contact dynamics vs the 1-D contact model",
                "Sticking speed of head-on impacts of two 50 um grains (silica, graphite): DEM bisection vs the 1-D "
                "JKR + damping integrator of contact.py.", "; ".join(vals) + f"; worst {worst:.1e}", "< 2e-3",
                worst < 2e-3)


def check_dem_conservation():
    """Undamped oblique impact of spinning, adhesive grains: momentum exact; total energy (kinetic + JKR + springs)
    constant while the contact holds (the JKR snap-on and break are the model's only losses)."""
    from .contact import JKRContact
    mat = get_material("silica")
    R = 50e-6
    x = [[0.0, 0.0, 0.0], [math.sqrt(4 - 0.25) * R + 1e-8, 0.5 * R, 0.0]]
    v = [[0.05, 0.0, 0.0], [0.0, 0.0, 0.0]]
    w = [[0.0, 0.0, 300.0], [0.0, 100.0, 0.0]]
    S = _pair_system(mat, R, x, v, w, e_dry=1.0, p_y=math.inf, q_roll=0, q_twist=0, x_gt=0.0, mu_pp=1e6,
                     h_roll=1e6, skin=0.5 * R)
    m, I = S.m, S.I
    jkr = JKRContact(pair_props(mat, R, mat, R))

    def mom():
        P = (m[:, None] * S.v).sum(0)
        Lm = (np.cross(S.x, m[:, None] * S.v) + I[:, None] * S.w).sum(0)
        return P, Lm

    def energy():
        E = 0.5 * (m * (S.v ** 2).sum(1)).sum() + 0.5 * (I * (S.w ** 2).sum(1)).sum()
        on = S.hist[:, dem.H_TOUCH] > 0.5
        if on.any():
            k = int(np.flatnonzero(on)[0])
            h, p = S.hist[k], S.prm[k]
            E += jkr.U(h[dem.H_A])
            E += 0.5 * p[dem.P_KT] * p[dem.P_A0] * float(h[0:3] @ h[0:3])
            E += 0.5 * p[dem.P_KR] * float(h[3:6] @ h[3:6]) + 0.5 * p[dem.P_KTW] * h[6] ** 2
        return E, bool(on.any())
    P0, L0 = mom()
    dt = 2e-10
    Es = []
    for _ in range(20000):
        S.run(dt, 10)
        E, on = energy()
        if on:
            Es.append(E)
        elif Es:
            break
    P1, L1 = mom()
    KE0 = 0.5 * (m * (np.asarray(v) ** 2).sum(1)).sum() + 0.5 * (I * (np.asarray(w) ** 2).sum(1)).sum()
    dE = float((max(Es) - min(Es)) / KE0) if len(Es) > 10 else float("nan")
    dP = float(np.linalg.norm(P1 - P0) / np.linalg.norm(P0))
    dL = float(np.linalg.norm(L1 - L0) / np.linalg.norm(L0))
    ok = dP < 1e-12 and dL < 1e-9 and dE < 1e-6
    return _res("dem_conservation", "Momentum and energy in contact dynamics",
                "Oblique impact (impact parameter 0.25 diameters, 5 cm/s) of two spinning 50 um silica grains, no "
                "damping, no yield, sliding and rolling below their limits: linear and angular momentum through the "
                "impact, and the total energy (kinetic + JKR potential + sliding, rolling and twisting spring energy) "
                "over the whole time in contact.",
                f"|dP|/P = {dP:.1e}, |dL|/L = {dL:.1e}, energy varies by {dE:.1e} of the impact energy over "
                f"{len(Es)} samples in contact", "< 1e-12; < 1e-9; < 1e-6", ok)


def check_rolling_on_wall():
    """A grain on a wall under a lateral load ramped through the rolling limit: onset and rigid-plastic rolling."""
    s, g = get_material("silica"), get_material("glass_borosilicate")
    R = 50e-6
    m = sphere_mass(s, R)
    wall = dict(E=g.E, nu=g.nu, gamma=g.gamma, e_dry=g.e_dry, p_y=g.yield_pressure)
    p = dem.make_params(R, 0.0, m, math.inf, s.E, s.nu, g.E, g.nu, 2 * math.sqrt(s.gamma * g.gamma), 0.9, math.inf,
                        0.4, 0.5, 100.0, 100.0, dem.DAMP_TSUJI_JKR, dem.BREAK_DISPLACEMENT, 1.0, 0.0)
    a_roll = p[dem.P_MUR] * 2 * p[dem.P_FC] / m
    z0 = R - p[dem.P_A0] ** 2 / (3 * R)
    L = [1.0, 1.0, 1.0]
    S = dem.DEMSystem([[0.5, 0.5, z0]], [[0, 0, 0]], [[0, 0, 0]], R, s.rho, s.E, s.nu, s.gamma, s.e_dry,
                      s.yield_pressure, L, wall, mu_pw=0.4, h_roll=0.5, q_roll=100, q_twist=100, wall_faces=(4,))
    dt = 1e-8
    t_end = 2e-3
    n = int(t_end / dt)
    ramp = 3 * a_roll / t_end
    t = (np.arange(n) + 1) * dt
    acc = np.column_stack([ramp * t, np.zeros(n), np.full(n, -9.80665)])
    S.run(dt, n, acc=acc)
    t_on = a_roll / ramp
    v_an = (0.5 * ramp * (t_end ** 2 - t_on ** 2) - a_roll * (t_end - t_on)) / 1.4
    rel = abs(S.v[0, 0] / v_an - 1)
    slip = abs(S.v[0, 0] - S.w[0, 1] * R) / abs(S.v[0, 0])
    return _res("rolling_on_wall", "Rolling on a wall (plastic rolling)",
                "50 um silica grain on borosilicate under 1 g pressing and a lateral body load ramped from 0 to 3x its "
                "rolling limit (4 F_c xi_crit / (m R), xi_crit = h a0/12, h = 0.5) in 2 ms: final speed vs the "
                "rigid-plastic rolling solution (m + I/R^2) dv/dt = m (a - a_roll) from the onset time, and rolling "
                "without slip (v = omega R).",
                f"rolling limit {a_roll / 9.80665:.2f} g; final speed {S.v[0, 0] * 1e3:.3f} vs {v_an * 1e3:.3f} mm/s "
                f"({rel:.1e}); slip {slip:.1e}", "< 1e-2 (elastic pre-roll and damping); slip < 1e-6",
                rel < 1e-2 and slip < 1e-6)


def check_cluster_statics():
    """Finite-difference stiffness of the production kernel vs analytic contact stiffnesses; static response."""
    s, g = get_material("silica"), get_material("glass_borosilicate")
    R = 50e-6
    m = sphere_mass(s, R)
    L = np.array([1.0, 1.0, 1.0])
    p = dem.make_params(R, 0.0, m, math.inf, s.E, s.nu, g.E, g.nu, 2 * math.sqrt(s.gamma * g.gamma), 0.9, math.inf,
                        0.4, 0.5, 100.0, 100.0, dem.DAMP_TSUJI_JKR, dem.BREAK_DISPLACEMENT, 1.0, 0.0)
    a0 = p[dem.P_A0]
    x = np.array([[0.5, 0.5, R - a0 ** 2 / (3 * R)]])
    h = np.zeros((1, dem.NH))
    h[0, dem.H_TOUCH] = 1.0
    K, J = statics.jacobians(x, np.array([R]), np.array([0]), np.array([-5]), p[None, :], h, L)
    Kd = K.toarray()
    e_n = abs(Kd[2, 2] / (1.2 * p[dem.P_ESTAR] * a0) - 1)
    e_t = abs(Kd[0, 0] / (p[dem.P_KT] * a0) - 1)
    dF = np.zeros(6)
    dF[0] = m * 9.80665
    du = np.linalg.solve(Kd, dF)
    dl = J.toarray() @ du
    e_s = abs(dl[1] / (-m * 9.80665) - 1)
    e_r = abs(dl[4] / (-m * 9.80665) - 1)
    worst = max(e_n, e_t, e_s, e_r)
    return _res("cluster_statics", "Statics of resting grains (sleep certificate)",
                "Stiffness of a grain on a wall from central differences of the production contact kernel vs the "
                "analytic JKR normal stiffness 1.2 E* a0 and Mindlin tangential k_t a0; static response to a 1 g "
                "sideways load: sliding force and rolling pseudo-force must both equal -m g.",
                f"normal {e_n:.1e}, tangential {e_t:.1e}, sliding force {e_s:.1e}, rolling {e_r:.1e}", "< 1e-6",
                worst < 1e-6)


def check_random_vibration():
    from . import vibration as V
    rng = np.random.default_rng(11)
    bp = V.GUIDE_RANDOM["thrust"]["breakpoints"]
    fs = 65536
    x, X, f = V.synthesize_random(bp, 20.0, fs, rng)
    r = V.welch_check(x, fs, bp, nperseg=8192)
    h = V.VibrationHistory(0.0, fs, np.vstack([0 * x, 0 * x, x]))
    ts = np.sort(rng.random(50) * 20)
    sp_err = float(np.abs(h.accel(ts)[2] / V.G0 - V.fourier_eval(X, f, x.size, ts)).max() / x.std())
    ok = (r["max_abs_dev_db"] < 1.5 and abs(r["grms_time"] / r["grms_spec"] - 1) < 0.02
          and abs(r["kurtosis"] - 3) < 0.05 and abs(r["skewness"]) < 0.05 and sp_err < 1e-4)
    return _res("random_vibration", "Random vibration synthesis (guide thrust axis)",
                "20 s of the RockSat User Guide 2026 thrust-axis test spectrum (flat 0.051 g^2/Hz, 20-2000 Hz, "
                "10.05 Grms) synthesised with Gaussian Fourier coefficients: Welch PSD (Hann, 8192 points) vs the spec "
                "over 40-1980 Hz, Grms, Gaussian moments, and the cubic-spline interpolant vs the exact band-limited "
                "signal between samples.",
                f"PSD within {r['max_abs_dev_db']:.2f} dB; Grms {r['grms_time']:.3f} vs {r['grms_spec']:.3f}; "
                f"kurtosis {r['kurtosis']:.3f}; skewness {r['skewness']:.3f}; spline error {sp_err:.1e} sigma",
                "< 1.5 dB (NASA-STD-7001C allows 3 dB); < 2%; 3 +/- 0.05; |s| < 0.05; < 1e-4", ok)


def check_time_step():
    """Halving the DEM step must not change a plastic rolling path or an impact outcome beyond 1e-4."""
    s, g = get_material("silica"), get_material("glass_borosilicate")
    R = 50e-6
    m = sphere_mass(s, R)
    wall = dict(E=g.E, nu=g.nu, gamma=g.gamma, e_dry=g.e_dry, p_y=g.yield_pressure)
    p = dem.make_params(R, 0.0, m, math.inf, s.E, s.nu, g.E, g.nu, 2 * math.sqrt(s.gamma * g.gamma), 0.9, math.inf,
                        0.4, 0.5, 100.0, 100.0, dem.DAMP_TSUJI_JKR, dem.BREAK_DISPLACEMENT, 1.0, 0.0)
    a_roll = p[dem.P_MUR] * 2 * p[dem.P_FC] / m
    z0 = R - p[dem.P_A0] ** 2 / (3 * R)
    T = 1e-3
    out = []
    for dt in (1.6e-8, 0.8e-8):
        S = dem.DEMSystem([[0.5, 0.5, z0]], [[0, 0, 0]], [[0, 0, 0]], R, s.rho, s.E, s.nu, s.gamma, s.e_dry,
                          s.yield_pressure, [1, 1, 1], wall, wall_faces=(4,))
        n = int(round(T / dt))
        S.run(dt, n, acc=np.array([2 * a_roll, 0, -9.80665]))
        out.append(S.x[0, 0] - 0.5)
    rel = abs(out[0] / out[1] - 1)
    return _res("time_step", "DEM time-step convergence",
                "Rolled distance of a 50 um silica grain on glass in 1 ms under twice its rolling limit, at the "
                "production step (~1/50 of the fastest contact period, 1.6e-8 s here) vs half of it.",
                f"{out[1] * 1e6:.4f} um; change {rel:.1e}", "< 1e-4", rel < 1e-4)


CHECKS = [check_dem_vs_1d, check_dem_conservation, check_rolling_on_wall, check_cluster_statics,
          check_random_vibration, check_time_step]


def run_all(progress=None):
    import time
    out = []
    for f in CHECKS:
        t0 = time.time()
        try:
            r = f()
        except Exception as exc:  # a crash is a failed check
            r = _res(f.__name__, f.__name__, "", f"error: {exc!r}", "", False)
        r["seconds"] = round(time.time() - t0, 1)
        out.append(r)
        if progress:
            progress(r)
    return out


def _wall_grain(mat_name="silica", R=50e-6):
    s, g = get_material(mat_name), get_material("glass_borosilicate")
    m = sphere_mass(s, R)
    wall = dict(E=g.E, nu=g.nu, gamma=g.gamma, e_dry=g.e_dry, p_y=g.yield_pressure)
    p = dem.make_params(R, 0.0, m, math.inf, s.E, s.nu, g.E, g.nu, 2 * math.sqrt(s.gamma * g.gamma),
                        math.sqrt(s.e_dry * g.e_dry), math.inf, 0.4, 0.5, 100.0, 100.0, dem.DAMP_TSUJI_JKR,
                        dem.BREAK_DISPLACEMENT, 1.0, 0.0)
    return s, g, m, wall, p


def check_roll1_vs_full():
    """The slaved-rolling model (roll1.py) against the full-stiffness model, for a grain rolling and rocking under a
    load that ramps past its rolling limit with 0.7, 2 and 3 kHz components."""
    from . import roll1
    worst = 0.0
    vals = []
    for name, R, T in (("silica", 50e-6, 2e-3), ("graphite", 25e-6, 1e-3)):
        s, g, m, wall, p = _wall_grain(name, R)
        I = 0.4 * m * R * R
        a_roll = p[dem.P_MUR] * 2 * p[dem.P_FC] / m
        d0 = p[dem.P_A0] ** 2 / (3 * R)

        def load(t):
            ax = a_roll * (2.0 * np.minimum(t / (0.5 * T), 1.0) + 0.5 * np.sin(2 * math.pi * 2000 * t))
            ay = 0.3 * a_roll * np.sin(2 * math.pi * 700 * t)
            az = -9.80665 * (1 + 0.5 * np.sin(2 * math.pi * 3000 * t))
            return np.column_stack([ax, ay, az])
        S = dem.DEMSystem([[0.5, 0.5, R - d0]], [[0, 0, 0]], [[0, 0, 0]], R, s.rho, s.E, s.nu, s.gamma, s.e_dry,
                          s.yield_pressure, [1, 1, 1], wall, mu_pw=0.4, h_roll=0.5, q_roll=100, q_twist=100,
                          wall_faces=(4,))
        OM = np.array([0.0, 0.0, 25.0])             # 25 rad/s of payload spin, the grain 0.7 m off the axis
        OMD = np.array([0.0, 0.0, 40.0])            # spinning up at 40 rad/s^2 (Euler force)
        S.off = np.array([0.0, 0.0, 0.0])
        S.gdrag[:] = 30.0                          # gas drag rate [1/s]
        dtA = 1e-8
        nA = int(round(T / dtA))
        tA = (np.arange(nA) + 1) * dtA
        S.run(dtA, nA, acc=load(tA), Om=OM[None, :] + OMD[None, :] * tA[:, None], Omd=OMD)
        Troll = 2 * math.pi * math.sqrt((I + m * R * R) / (p[dem.P_KR] * R * R))
        Ttw = 2 * math.pi * math.sqrt(I / p[dem.P_KTW])
        n = int(math.ceil(T / (min(Troll, Ttw) / 200)))
        dt = T / n
        x = np.array([[0.5, 0.5, R - d0]])
        v = np.zeros((1, 3))
        w = np.zeros((1, 3))
        h = np.zeros((1, dem.NH))
        h[0, dem.H_TOUCH] = 1
        h[0, dem.H_A] = p[dem.P_A0]
        flag = np.zeros(1, np.int64)
        Ts = np.arange(n + 1) * dt
        roll1.run(x, v, w, np.array([m]), np.array([I]), np.array([R]), np.array([4]), p[None, :].copy(), h,
                  np.ones(3), dt, n, load(Ts), OM[None, :] + OMD[None, :] * Ts[:, None], np.tile(OMD, (n + 1, 1)),
                  S.off, np.full(1, 30.0),
                  np.zeros((1, 3)), 0.02, flag, np.full(1, np.inf), np.zeros(1), np.zeros(1), np.full(1, -np.inf))
        path = math.hypot(S.x[0, 0] - 0.5, S.x[0, 1] - 0.5)
        err = float(np.linalg.norm(x[0, :2] - S.x[0, :2]) / path)
        worst = max(worst, err)
        vals.append(f"{name} r = {R * 1e6:.0f} um: path {path * 1e6:.3f} um, difference {err:.1e}")
    return _res("roll1_vs_full", "Slaved rolling vs full contact dynamics",
                "A grain on glass rolls past its limit and rocks under loads with 0.7, 2 and 3 kHz content (pressing "
                "load also oscillating), under 25 rad/s of payload spin spinning up at 40 rad/s^2 with the grain "
                "0.7 m off the axis (centrifugal, Coriolis and Euler forces) and gas drag. roll1.py (stiff normal and sliding modes held quasi-static, "
                "rolling and twisting resolved at 1/200 of their period) vs the full-stiffness DEM at 10 ns.",
                "; ".join(vals), "< 3e-5 of the rolled path", worst < 3e-5)


def check_sleep_exact():
    """A resting grain under a slowly rising sideways load, in the full model: while the engine's sleep certificate
    holds, the grain only deflects elastically, exactly by the quasi-static spring deflection that the engine applies
    when it wakes the grain; once the certificate fails the grain rolls."""
    s, g, m, wall, p = _wall_grain("silica", 50e-6)
    R = 50e-6
    a_roll = p[dem.P_MUR] * 2 * p[dem.P_FC] / m
    d0 = p[dem.P_A0] ** 2 / (3 * R)
    S = dem.DEMSystem([[0.5, 0.5, R - d0]], [[0, 0, 0]], [[0, 0, 0]], R, s.rho, s.E, s.nu, s.gamma, s.e_dry,
                      s.yield_pressure, [1, 1, 1], wall, mu_pw=0.4, h_roll=0.5, q_roll=100, q_twist=100,
                      wall_faces=(4,))
    T = 20e-3                                   # ramp 0 -> 1.5 a_roll in 20 ms (slow vs the 0.26 ms rocking period)
    dt = 1e-8
    n = int(round(T / dt))
    t = (np.arange(n) + 1) * dt
    acc = np.column_stack([1.5 * a_roll * t / T, np.zeros(n), np.full(n, -9.80665)])
    ks = p[dem.P_KT] * p[dem.P_A0]
    t_c = 0.98 / 1.5 * T                         # the certificate (load <= 98% of the limit) fails here
    err, held_max, xs_end = 0.0, 0.0, 0.0
    for k in range(0, n, 20000):
        S.run(dt, 20000, acc=acc[k:k + 20000])
        tk = (k + 20000) * dt
        if tk <= t_c:
            F_t = m * 1.5 * a_roll * tk / T
            x_qs = F_t / ks + F_t / p[dem.P_KR]          # sliding + rolling spring deflection under the static load
            err = max(err, abs((S.x[0, 0] - 0.5) - x_qs))
            held_max = max(held_max, abs(S.x[0, 0] - 0.5))
    xs_end = float(S.x[0, 0] - 0.5)
    xi_crit = 0.5 * p[dem.P_A0] / 12
    ok = err < 0.02 * xi_crit and xs_end > 100 * held_max
    return _res("sleep_exact", "Sleeping grains do not miss motion",
                "A 50 um silica grain on glass under a sideways load ramped from 0 to 1.5x its rolling limit in 20 ms, "
                "integrated in the full model. While the engine's sleep certificate would hold (load <= 98% of the "
                "limit), the grain's displacement must equal the static deflection of its sliding and rolling "
                "springs (what the engine applies on waking it); afterwards it must roll.",
                f"largest difference from the static deflection {err * 1e9:.3f} nm (deflection up to "
                f"{held_max * 1e9:.1f} nm, xi_crit {xi_crit * 1e9:.1f} nm); rolled {xs_end * 1e6:.2f} um by the end",
                "< 2% of xi_crit; rolls > 100x further afterwards", ok)


CHECKS += [check_roll1_vs_full, check_sleep_exact]


def check_engine_handoff():
    """Two free grains flown by the engine's exact free flight and handed to the DEM at their capture time must
    collide exactly as in a DEM run from the start (same kernel), and a grain released from a wall must leave as in
    the DEM."""
    from .engine import FullContactSim, FREE
    cfg = {"dt_s": 1e-3, "duration_s": 2e-3, "output_fps": 1000, "environment": {"preset": "ideal", "pressure_pa": 0.0},
           "contact": {"contact_charging": False}, "forces": {}}
    ch = {"name": "handoff", "size_m": [0.01, 0.01, 0.01], "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": 2, "radius_m": 50e-6, "velocity": {"speed_mean_m_s": 0.0},
                         "initial_state": {"type": "free"}}]}
    S = FullContactSim(cfg, ch, 0)
    R = 50e-6
    x0 = np.array([[0.004, 0.005, 0.005], [0.004 + 2 * R + 40e-6, 0.005 + 0.3 * R, 0.005]])
    v0 = np.array([[0.03, 0.0, 0.0], [-0.02, 0.0, 0.0]])
    w0 = np.array([[0.0, 0.0, 200.0], [0.0, 0.0, 0.0]])
    S.x[:], S.v[:], S.w[:] = x0, v0, w0
    S.regime[:] = FREE
    S.base.F_static[:] = 0.0
    S.run(progress=False)
    mat = get_material("silica")

    def ref(dt):
        D = dem.DEMSystem(x0, v0, w0, R, mat.rho, mat.E, mat.nu, mat.gamma, mat.e_dry, mat.yield_pressure,
                          [0.01, 0.01, 0.01], dict(E=1e11, nu=0.2, gamma=0.02, e_dry=0.8), walls=False,
                          mu_pp=S._mu(0, 1), h_roll=S.h_roll, q_roll=S.q_roll, q_twist=S.q_twist)
        D.run(dt, int(round(2e-3 / dt)))
        return D
    dt = min(S._dt_dem, S._hertz_dt(2 * float(np.abs(v0).max())))     # the step the engine's DEM used
    D1, D2 = ref(dt), ref(dt / 2)

    def diffs(A, B):
        return (float(np.abs(A.v - B.v).max() / np.abs(v0).max()), float(np.abs(A.w - B.w).max() / np.abs(w0).max()),
                float(np.abs(A.x - B.x).max() / R))
    e_eng = diffs(S, D2)
    e_dt = diffs(D1, D2)
    n_imp = sum(1 for c in S.collisions if c["kind"] == "pp")
    ok = n_imp == 1 and all(a <= 3 * b + 1e-7 for a, b in zip(e_eng, e_dt))
    return _res("engine_handoff", "Free flight to contact hand-off (engine)",
                "Two silica grains (one spinning at 200 rad/s) approach obliquely at 5 cm/s, 40 um apart. The engine flies "
                "them exactly, hands them to the DEM within the capture distance, resolves the impact and lets them "
                "fly apart. Compared after 2 ms with the same initial state run in the DEM throughout (at half the "
                "engine's step): the engine may differ by no more than 3x the DEM's own step error (DEM at the "
                "engine's step vs half of it), i.e. the hand-off adds nothing beyond time discretisation.",
                f"impacts {n_imp}; engine vs reference: velocity {e_eng[0]:.1e}, spin {e_eng[1]:.1e}, position "
                f"{e_eng[2]:.1e} of R; DEM step error: {e_dt[0]:.1e}, {e_dt[1]:.1e}, {e_dt[2]:.1e}",
                "1 impact; each <= 3x the step error", ok)


CHECKS += [check_engine_handoff]


def _regime_scene(all_dem=False, n_ppp=50, v_proj=0.06, nrow=3, gap=6e-6, v_land=0.015, dur=2e-3):
    """A row of resting 50 um silica grains on a glass floor under 1 g (6 um apart), struck end-on by a grain;
    a grain rolling at 4 mm/s beside the row; a grain landing on the row from above; spare grains resting apart."""
    from .engine import FullContactSim, ASLEEP, ROLL, FREE
    cfg = {"dt_s": 1e-3, "duration_s": dur, "output_fps": 1000,
           "environment": {"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": [0.0, 0.0, -1.0]},
           "contact": {"contact_charging": False, "all_dem": all_dem, "steps_per_period": n_ppp}, "forces": {}}
    ch = {"name": "regimes", "size_m": [0.003, 0.001, 0.001], "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": 11, "radius_m": 50e-6, "velocity": {"speed_mean_m_s": 0.0},
                         "initial_state": {"type": "free"}}]}
    S = FullContactSim(cfg, ch, 0)
    S.base.F_static[:] = 0.0
    R, g = 50e-6, gap
    T0 = float(S.env.t0_flight)
    S.v[:], S.w[:] = 0.0, 0.0
    S.regime[:] = FREE
    for k in range(8):
        S.x[k] = [0.6e-3 + k * (2 * R + g), 0.5e-3, R] if k < nrow else [1.9e-3 + (k - nrow) * 0.15e-3, 0.5e-3, R]
    S.x[9] = [S.x[min(3, nrow - 1), 0] - 20e-6, 0.5e-3 + 2 * R + 4e-6, R]
    for k in list(range(8)) + [9]:
        S._place_at_equilibrium(k, 4, S._row(k, -5), T0)
        S.regime[k] = ASLEEP
        S._sleep_face[k] = 4
    S._sleep_face.pop(9)
    S.v[9] = [4e-3, 0.0, 0.0]
    S.w[9] = np.cross([0.0, 0.0, 1.0], S.v[9]) / R
    S._roll_face[9] = 4
    S.regime[9] = ROLL
    assert S._try_roll(9, 4, T0, 1e-3)
    S.x[8], S.v[8] = [0.6e-3 - 2 * R - 30e-6, 0.5e-3, R + 1.5e-6], [v_proj, 0.0, 0.0]
    S.x[10], S.v[10] = [S.x[min(5, nrow - 1), 0] - 60e-6, 0.5e-3, 3.5 * R], [v_land, 0.0, -v_land]
    S.base.frozen[:] = S.regime != FREE
    S.run(progress=False)
    return S


def check_engine_regimes():
    """The engine's regimes (sleeping, slaved rolling, exact free flight, the verified step) against the same scene
    with every grain in the full DEM throughout: the regimes may add no more than the DEM's own step error."""
    from collections import Counter
    A = _regime_scene()
    B = _regime_scene(all_dem=True)
    C = _regime_scene(all_dem=True, n_ppp=100)
    R = 50e-6

    def ev(S):
        return Counter((c["kind"], c["i"], c["j"], c["outcome"]) for c in S.collisions)
    dx_ab = float(np.abs(A.x - B.x).max() / R)
    dx_bc = float(np.abs(B.x - C.x).max() / R)
    dv_ab = float(np.abs(A.v - B.v).max() / 0.06)
    dv_bc = float(np.abs(B.v - C.v).max() / 0.06)
    same = ev(A) == ev(B) == ev(C)
    moved = float(np.abs(B.x - A.x).max())
    ok = same and dx_ab <= 3 * dx_bc + 1e-6 and dv_ab <= 3 * dv_bc + 1e-6
    ne = A.n_events
    return _res("engine_regimes", "Engine regimes vs all-DEM",
                "A row of 3 resting 50 um silica grains on a glass floor under 1 g, 6 um apart, struck end-on by a "
                "grain at 6 cm/s (the blow passes down the row); a grain rolling at 4 mm/s beside the row; a grain "
                "landing on the row from above at 1.5 cm/s; 5 more grains resting apart. Run 2 ms by the engine (sleepers join the DEM when approached, the rolling "
                "grain on the slaved model, flight exact, every step verified) and with every grain in the full DEM "
                "throughout (at the engine's step and at half of it). The engine may differ from the all-DEM run by "
                "no more than 3x the DEM's own step error, with the same contacts and outcomes.",
                f"contacts: {sum(ev(A).values())} (same in all runs: {same}); engine vs all-DEM: position "
                f"{dx_ab:.1e} R, velocity {dv_ab:.1e} of 6 cm/s; DEM step error {dx_bc:.1e} R, {dv_bc:.1e}; "
                f"sleepers joined on approach {ne.get('dem_joins', 0)}, verified steps {ne.get('checked_steps', 0)}, "
                f"steps redone {ne.get('redos', 0)}",
                "same contacts; each <= 3x the step error", ok)


CHECKS += [check_engine_regimes]


def check_linres():
    """A stuck pair of grains still rocking on a floor: put to sleep with its residual motion tracked as exact
    linear response and woken 2 ms later, it must be where the full model puts it."""
    import copy
    from .engine import FullContactSim, ACTIVE
    cfg = {"dt_s": 1e-3, "duration_s": 4e-3, "output_fps": 1000,
           "environment": {"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": [0.0, 0.0, -1.0]},
           "contact": {"contact_charging": False, "all_dem": True}, "forces": {}}
    ch = {"name": "linres", "size_m": [0.001, 0.001, 0.001], "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": 2, "radius_m": 50e-6, "velocity": {"speed_mean_m_s": 0.0},
                         "initial_state": {"type": "free"}}]}
    S = FullContactSim(cfg, ch, 0)
    S.base.F_static[:] = 0.0
    R = 50e-6
    S.x[0] = [0.5e-3, 0.5e-3, R]
    S.x[1] = [0.5e-3 + 2 * R - 2e-9, 0.5e-3, R]
    S.v[:], S.w[:] = 0.0, 0.0
    S.v[1] = [0.0, 2e-4, 0.0]                      # a sideways nudge: rocking and twisting of the pair
    S.regime[:] = ACTIVE
    S.run(progress=False)                          # 4 ms in the full model: the stiff modes have died out
    dT = 2e-3
    A = copy.deepcopy(S)
    A.duration = S.t + dT
    A.run(progress=False)                          # the full model for 2 ms more
    T = S.env.t0_flight + S.t
    mem = np.array([0, 1])
    out = {}
    for keep in (True, False):
        B = copy.deepcopy(S)
        B.all_dem = False
        touch = B._touching()
        if not keep:
            B.v[:], B.w[:] = 0.0, 0.0              # comparison: the same sleep with the residual dropped
        slept = B._try_sleep_island(mem, touch, T, 1e-3)
        iid = B._island_of.get(0)
        if not slept or iid is None:
            out[keep] = None
            continue
        used = "res" in B._islands[iid]
        B._wake_island(iid, T + dT)
        out[keep] = (float(np.abs(B.x - A.x).max()), float(np.abs(B.w - A.w).max()), used)
    if out.get(True) is None or out.get(False) is None:
        return _res("linres", "Sleeping with the residual motion (linear response)", "", "did not sleep",
                    "sleeps", False)
    (ex, ew, used), (ex0, ew0, _) = out[True], out[False]
    ok = used and ex <= 0.05 * ex0 and ew <= 0.05 * ew0
    return _res("linres", "Sleeping with the residual motion (linear response)",
                "Two touching 50 um silica grains on a glass floor under 1 g, the second nudged sideways at 0.2 mm/s, "
                "integrated 4 ms in the full model (still rocking and twisting). From that state: (a) 2 ms more of "
                "the full model; (b) the engine puts the pair to sleep with its residual motion as exact linear "
                "response (stiffness and dashpots from the kernel's tangent) and wakes it 2 ms later. (b) must "
                "match (a) to 5% of the residual itself (what dropping the residual would miss).",
                f"engine vs full model: position {ex * 1e12:.2f} pm, spin {ew:.2e} rad/s; with the residual dropped: "
                f"{ex0 * 1e12:.2f} pm, {ew0:.2e} rad/s",
                "<= 5% of the dropped-residual error", ok)


CHECKS += [check_linres]


def check_slaved_vs_full():
    """The slaved cluster model (slaved.py) against the full contact model: a stuck pair on a glass floor, a grain
    stuck on top of another (leaning 20 deg) and a single grain, each driven past its rolling limit by a ramped
    sideways load with 0.7-3 kHz components, under 25 rad/s of spin, for 2 ms."""
    from . import slaved
    s, glass = get_material("silica"), get_material("glass_borosilicate")
    wall = dict(E=glass.E, nu=glass.nu, gamma=glass.gamma, e_dry=glass.e_dry, p_y=glass.yield_pressure)
    R, T, OM = 50e-6, 2e-3, np.array([0.0, 0.0, 25.0])

    def make(x0):
        D = dem.DEMSystem(x0, np.zeros_like(x0), np.zeros_like(x0), R, s.rho, s.E, s.nu, s.gamma, s.e_dry,
                          s.yield_pressure, [1, 1, 1], wall, mu_pw=0.4, mu_pp=0.6, h_roll=0.5, q_roll=100,
                          q_twist=100, wall_faces=(4,))
        D.off = -np.array([0.5, 0.5, 0.0])
        return D
    m = sphere_mass(s, R)
    pw = make(np.array([[0.5, 0.5, R]])).prm[0]
    pp = make(np.array([[0.5, 0.5, 0.5], [0.5 + 2 * R, 0.5, 0.5]])).prm[0]
    d0w = pw[dem.P_A0] ** 2 / (3 * pw[dem.P_REFF])
    d0p = pp[dem.P_A0] ** 2 / (3 * pp[dem.P_REFF])
    a_lim = pw[dem.P_MUR] * 2 * pw[dem.P_FC] / m
    th = math.radians(20)
    Dp = 2 * R - d0p
    cases = {
        "pair": (np.array([[0.5, 0.5, R - d0w], [0.5 + Dp, 0.5, R - d0w]]), lambda t: np.column_stack([
            0.3 * a_lim * np.sin(2 * math.pi * 700 * t),
            a_lim * (2.0 * np.minimum(t / (0.5 * T), 1.0) + 0.5 * np.sin(2 * math.pi * 2000 * t)),
            -9.80665 * (1 + 0.5 * np.sin(2 * math.pi * 3000 * t))])),
        "tower": (np.array([[0.5, 0.5, R - d0w], [0.5 + Dp * math.sin(th), 0.5, R - d0w + Dp * math.cos(th)]]),
                  lambda t: np.column_stack([
                      0.6 * a_lim * (np.minimum(t / (0.5 * T), 1.0) + 0.3 * np.sin(2 * math.pi * 2000 * t)),
                      0.3 * a_lim * np.sin(2 * math.pi * 700 * t),
                      -9.80665 * (1 + 0.5 * np.sin(2 * math.pi * 3000 * t))])),
        "single": (np.array([[0.5, 0.5, R - d0w]]), lambda t: np.column_stack([
            0.3 * a_lim * np.sin(2 * math.pi * 700 * t),
            a_lim * (2.0 * np.minimum(t / (0.5 * T), 1.0) + 0.5 * np.sin(2 * math.pi * 2000 * t)),
            -9.80665 * (1 + 0.5 * np.sin(2 * math.pi * 3000 * t))])),
    }
    vals, worst, ok_all = [], 0.0, True
    for name, (x0, load) in cases.items():
        def full(dt):
            D = make(x0)
            n = int(round(T / dt))
            D.run(dt, n, acc=load((np.arange(n) + 1) * dt), Om=OM)
            return D
        A, B = full(1e-8), full(5e-9)
        D = make(x0)
        dem.compute_forces(D.x, D.v, D.w, D.m, D.I, D.L, D.ci, D.cj, D.prm, D.hist, 0.0, np.zeros(3), OM, np.zeros(3),
                           D.off, D.gdrag, D.F, D.T, D.Fn, D.Fext, 0.0)
        on = D.hist[:, dem.H_TOUCH] > 0.5
        ci, cj = D.ci[on].copy(), D.cj[on].copy()
        prm, hist = np.ascontiguousarray(D.prm[on]), np.ascontiguousarray(D.hist[on])
        eta0 = slaved.slaving_indicator(x0, D.m, D.I, D.rad, ci, cj, prm, hist, D.L, 3000.0)
        x, v, w = x0.copy(), np.zeros_like(x0), np.zeros_like(x0)
        lam = slaved.enter(x, v, w, D.m, D.I, D.rad, ci, cj, prm, hist, D.L, D.off, load(np.zeros(1))[0], OM,
                           np.zeros(3), D.gdrag, D.Fext)[0]
        slaved.settle(x, D.m, D.rad, ci, cj, prm, hist, D.L, lam)
        n = int(math.ceil(T / (slaved.soft_period(D.m, D.I, D.rad, ci, cj, prm) / 400)))
        dt = T / n
        Ts = np.arange(n + 1) * dt
        N = len(x0)
        flag = np.zeros(2, np.int64)
        cface = np.zeros((N, 6), np.bool_)
        for k in range(len(ci)):
            if cj[k] < 0:
                cface[ci[k], -1 - cj[k]] = True
        slaved.run(x, v, w, D.m, D.I, D.rad, ci, cj, prm, hist, D.L, D.off, dt, n, load(Ts), np.tile(OM, (n + 1, 1)),
                   np.zeros((n + 1, 3)), D.gdrag, D.Fext, 0.02, 1e-7, lam, flag, np.full(N, np.inf), np.zeros(N),
                   np.zeros(N), np.full(N, -np.inf), np.zeros((0, 3)), np.zeros(0), cface, np.zeros(N), np.zeros(N), 1.0)
        path = float(np.linalg.norm(B.x - x0, axis=1).max())
        err = float(np.linalg.norm(x - B.x, axis=1).max()) / path
        e_dt = float(np.linalg.norm(A.x - B.x, axis=1).max()) / path
        ok_all = ok_all and flag[0] == 0 and err <= 3e-5
        if flag[0]:
            vals.append(f"{name}: stopped at step {int(flag[0])} (reason {int(flag[1])})")
        worst = max(worst, err)
        vals.append(f"{name}: path {path * 1e6:.2f} um, difference {err:.1e} (full model's own step error {e_dt:.0e}; "
                    f"slaving indicator {eta0[0]:.1e}, with the 3 kHz load {eta0[1]:.1e}: "
                    f"{'accepted' if eta0[0] <= slaved.ETA0_MAX and eta0[1] <= slaved.ETAF_MAX else 'refused'} by the "
                    f"engine)")
    return _res("slaved_vs_full", "Slaved clusters vs full contact dynamics",
                "A stuck pair of 50 um silica grains on a glass floor, a grain stuck on top of another (leaning 20 deg) "
                "and a single grain, each driven past its rolling limit by a ramped sideways load with 0.7-3 kHz "
                "components under 25 rad/s of spin, integrated 2 ms by the slaved model (stiff modes quasi-static, "
                "400 steps per softest period) and by the full model at 5 ns steps. Also reported: the slaving "
                "indicator the engine uses to accept a cluster (the shift of its soft frequencies; accepted at <= "
                "2e-4, and <= 1e-3 when weighted by the loads' frequencies), which is conservative: the leaning grain "
                "is refused though its error is small.",
                "; ".join(vals), "<= 3e-5 of the rolled path, no hand-back", ok_all)


CHECKS += [check_slaved_vs_full]


def _slaved_scene(all_dem=False, n_ppp=50, dur=6e-3):
    """A stuck pair of 50 um silica grains on a glass floor rolling sideways under 10 g (1 g pressing), and a single
    grain rolling after it, 0.4 mm behind and 10 cm/s faster, until it catches up and hits it."""
    from .engine import FullContactSim, ACTIVE, ASLEEP, ROLL, FREE
    cfg = {"dt_s": 1e-3, "duration_s": dur, "output_fps": 1000,
           "environment": {"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": [10.0, 0.0, -1.0]},
           "contact": {"contact_charging": False, "all_dem": all_dem, "steps_per_period": n_ppp}, "forces": {}}
    ch = {"name": "slaved", "size_m": [0.003, 0.001, 0.001], "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": 3, "radius_m": 50e-6, "velocity": {"speed_mean_m_s": 0.0},
                         "initial_state": {"type": "free"}}]}
    S = FullContactSim(cfg, ch, 0)
    S.base.F_static[:] = 0.0
    R = 50e-6
    T0 = float(S.env.t0_flight)
    S.v[:], S.w[:] = 0.0, 0.0
    S.x[0] = [1.0e-3, 0.5e-3, R]
    S.x[1] = [1.0e-3, 0.5e-3 + 2 * R - 2e-9, R]
    S.regime[0] = S.regime[1] = ACTIVE
    S.x[2] = [1.0e-3 - 2 * R - 400e-6, 0.5e-3 + R, R]
    S._place_at_equilibrium(2, 4, S._row(2, -5), T0)
    S.v[2] = [0.10, 0.0, 0.0]
    S.w[2] = np.cross([0.0, 0.0, 1.0], S.v[2]) / R
    if all_dem:
        S.regime[2] = ACTIVE
    else:
        S._roll_face[2] = 4
        S.regime[2] = ROLL
        assert S._try_roll(2, 4, T0, 1e-3)
    S.base.frozen[:] = S.regime != FREE
    S.run(progress=False)
    return S


def check_engine_slaved():
    """A slaved cluster among other moving grains, by the engine and with every grain in the full model."""
    from collections import Counter
    A = _slaved_scene()
    B = _slaved_scene(all_dem=True)
    C = _slaved_scene(all_dem=True, n_ppp=100)
    R = 50e-6

    def ev(S):
        return Counter((c["kind"], c["i"], c["j"], c["outcome"]) for c in S.collisions)
    x0 = np.array([[1.0e-3, 0.5e-3, R], [1.0e-3, 0.5e-3 + 2 * R, R], [1.0e-3 - 2 * R - 400e-6, 0.5e-3 + R, R]])
    d_ab = np.linalg.norm(A.x - B.x, axis=1)
    d_bc = np.linalg.norm(B.x - C.x, axis=1)
    paths = np.linalg.norm(B.x - x0, axis=1)
    allow = 3 * d_bc + 3e-5 * paths + 1e-12          # the full model's step error and the slaved models' accuracy
    same = ev(A) == ev(B)
    ne = A.n_events
    slv = ne.get("slaved_grain_steps", 0)
    ok = same and slv > 0 and bool((d_ab <= allow).all())
    dx_ab, dx_bc, path = float(d_ab.max() / R), float(d_bc.max() / R), float(paths[:2].max() / R)
    return _res("engine_slaved", "Slaved clusters in the engine vs all-DEM",
                "A stuck pair of 50 um silica grains on a glass floor rolls sideways under 10 g (1 g pressing) while a "
                "single grain rolls after it, 0.4 mm behind and 10 cm/s faster, and hits it. Run 6 ms by the engine "
                "(the pair on the slaved model, the single grain on the slaved rolling model, both captured into the "
                "full model before they meet) and with every grain in the full model throughout (at the engine's step "
                "and half of it). Same contacts and outcomes; each grain within 3x the full model's own step error "
                "plus 3e-5 of its path (the slaved models' validated accuracy); slaved steps taken.",
                f"contacts {sum(ev(A).values())} (same as all-DEM: {same}); slaved entries {ne.get('slaved_entries', 0)}, "
                f"slaved grain-steps {slv}; pair rolled {path:.2f} R; engine vs all-DEM {dx_ab:.1e} R "
                f"(allowed {float(allow.min() / R):.1e}-{float(allow.max() / R):.1e} R); DEM "
                f"step error {dx_bc:.1e} R; steps redone {ne.get('redos', 0)}",
                "same contacts; slaved steps taken; each grain <= 3x step error + 3e-5 of its path", ok)


CHECKS += [check_engine_slaved]

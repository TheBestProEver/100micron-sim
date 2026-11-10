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
    """Two free grains flown by the engine's exact free flight and handed to the DEM before they can touch must
    collide exactly as the DEM alone would on the same time grid; and the result must lie within the DEM's own
    time-discretisation spread (its step at different phases of the time grid) of a fine reference."""
    import math
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
    segs = []                    # the engine's DEM segments on the kept path: start, length, steps, state in and out
    dem_run = S._dem

    def state():
        row = S._key.get((0, 1))
        return S.x.copy(), S.v.copy(), S.w.copy(), (S._hist[row].copy() if row is not None else None)

    def traced(ai, t, H):
        vrel = 2 * float(np.linalg.norm(S.v[ai], axis=1).max()) if len(ai) else 0.0
        n = max(1, int(math.ceil(H / min(S._dt_dem, S._hertz_dt(vrel)))))
        segs[:] = [q for q in segs if q[0] < t - 1e-15]        # a redo restarts the step: drop the discarded path
        s_in = state()
        out = dem_run(ai, t, H)
        segs.append((t, H, n, sorted(int(k) for k in ai), s_in, state()))
        return out
    S._dem = traced
    S.run(progress=False)
    mat = get_material("silica")

    def system(x=x0, v=v0, w=w0):
        return dem.DEMSystem(x, v, w, R, mat.rho, mat.E, mat.nu, mat.gamma, mat.e_dry, mat.yield_pressure,
                             [0.01, 0.01, 0.01], dict(E=1e11, nu=0.2, gamma=0.02, e_dry=0.8), walls=False,
                             mu_pp=S._mu(0, 1), h_roll=S.h_roll, q_roll=S.q_roll, q_twist=S.q_twist)

    def fly(D, tau):                               # no contact force: what velocity Verlet gives, in closed form
        D.x += D.v * tau
        D.t += tau
        D.rebuild()
        D._first = True

    def ref(dt, phase=0.0):
        D = system()
        n = int(round(2e-3 / dt))
        if phase > 0:
            fly(D, phase * dt)
            D.run(dt, n - 1)
            fly(D, (1 - phase) * dt)
        else:
            D.run(dt, n)
        return D

    def diffs(A, B):
        return (float(np.abs(A.v - B.v).max() / np.abs(v0).max()), float(np.abs(A.w - B.w).max() / np.abs(w0).max()),
                float(np.abs(A.x - B.x).max() / R))
    # (a1) the free flight up to the first hand-off against straight lines; (a2) each DEM segment against the DEM
    # alone from the state the engine handed over, on the engine's grid (both grains in it, contact history copied)
    t1, (xa, va, wa, _) = segs[0][0], segs[0][4]
    e_fly = (float(np.abs(va - v0).max() / np.abs(v0).max()), float(np.abs(wa - w0).max() / np.abs(w0).max()),
             float(np.abs(xa - (x0 + v0 * t1)).max() / R))
    e_grid = (0.0, 0.0, 0.0)
    for t0, H, n, ai, (xi, vi, wi, hi), (xo, vo, wo, ho) in segs:
        if ai != [0, 1]:
            continue
        G = system(xi, vi, wi)
        if hi is not None:
            G.hist[0] = hi
        G.run(H / n, n)
        e_grid = tuple(max(a, b) for a, b in zip(e_grid, (
            float(np.abs(vo - G.v).max() / np.abs(v0).max()), float(np.abs(wo - G.w).max() / np.abs(w0).max()),
            float(np.abs(xo - G.x).max() / R))))
    # (b) against a fine reference (1/8 of the engine's step), beside the DEM's own spread over 8 grid phases
    dt = min(S._dt_dem, S._hertz_dt(2 * float(np.abs(v0).max())))
    F = ref(dt / 8)
    e_fine = diffs(S, F)
    spread = np.max([diffs(ref(dt, ph), F) for ph in np.arange(8) / 8], axis=0)
    n_imp = sum(1 for c in S.collisions if c["kind"] == "pp")
    ok = (n_imp == 1 and len(segs) > 0 and max(e_fly) <= 1e-9 and max(e_grid) <= 1e-8
          and all(a <= 1.5 * b + 1e-12 for a, b in zip(e_fine, spread)))
    return _res("engine_handoff", "Free flight to contact hand-off (engine)",
                "Two silica grains (one spinning at 200 rad/s) approach obliquely at 5 cm/s, 40 um apart. The engine flies "
                "them exactly, hands them to the DEM before they can come within the capture distance (leaving and "
                "rejoining it as its bounds allow), resolves the impact and lets them fly apart. Checked in parts: "
                "(a1) the free flight up to the first hand-off against straight lines (the only other forces are "
                "the ~1e-9 m/s^2 tidal and mutual-gravity terms); (a2) every DEM segment the engine ran against the "
                "DEM alone from the state the engine handed over, on the engine's own grid: the hand-off itself must "
                "add nothing beyond round-off (1e-8, the stiff contact amplifies round-off); (b) after 2 ms, against "
                "a fine DEM reference (1/8 of the engine's step) from the same start: within 1.5x the spread of the "
                "DEM at the engine's step over 8 phases of the time grid (where an impact starts between two steps "
                "is the DEM's own discretisation error, and an adhesive impact amplifies any difference in the state "
                "it starts from).",
                f"impacts {n_imp}; {len(segs)} DEM segments; (a1) free flight vs straight lines: velocity "
                f"{e_fly[0]:.1e}, spin {e_fly[1]:.1e}, position {e_fly[2]:.1e} of R; (a2) DEM segments vs the DEM "
                f"alone: {e_grid[0]:.1e}, {e_grid[1]:.1e}, {e_grid[2]:.1e}; (b) vs the fine reference: "
                f"{e_fine[0]:.1e}, {e_fine[1]:.1e}, {e_fine[2]:.1e}; DEM grid-phase spread {spread[0]:.1e}, "
                f"{spread[1]:.1e}, {spread[2]:.1e}",
                "1 impact; (a1) each <= 1e-9; (a2) each <= 1e-8; (b) each <= 1.5x the grid-phase spread", ok)


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


# ---------------------------------------------------------------------------- resting clusters at their contact limits
# Checks 14-19 compare the sleeping of resting clusters at their sliding, rolling and twisting limits (engine.py:
# _try_sleep_island, _island_advance, _wake_island) with the full model run on every grain throughout (all_dem), at
# the engine's DEM step and at half of it. An impact leaves a grain's rolling springs at 0.6-0.74 of their limit (it
# relaxes them as it stops), so the states at a limit are prepared by setting the springs by hand to (1 - 2^-40) of
# their limit, an admissible state of the contact law (the kernel holds a spring there while the load keeps it at the
# limit), and letting the full model settle the grains from there.
_CAP_PREP = {}


def _cap_sim(pos, a_g=(0.0, 0.0, 0.0), size=(1e-3, 1e-3, 1e-3), n_ppp=50, env=None, chamber=None, dt=1e-3):
    """Grains of 50 um silica at pos in a glass box, at rest, all in the full model (all_dem), under the uniform load
    a_g [g] (environment preset ideal, no gas, no charging) or the environment env (a flight profile)."""
    from .engine import FullContactSim, ACTIVE
    e = {"preset": "ideal", "pressure_pa": 0.0, "residual_accel_g": list(a_g)} if env is None else env
    cfg = {"dt_s": dt, "duration_s": 1e-3, "output_fps": 1000, "environment": e,
           "contact": {"contact_charging": False, "all_dem": True, "steps_per_period": n_ppp}, "forces": {}}
    ch = {"name": "cap", "size_m": list(size), "wall_material": "glass_borosilicate",
          "particles": [{"material": "silica", "count": len(pos), "radius_m": 50e-6,
                         "velocity": {"speed_mean_m_s": 0.0}, "initial_state": {"type": "free"}}]}
    ch.update(chamber or {})
    S = FullContactSim(cfg, ch, 0)
    S.base.F_static[:] = 0.0
    S.x[:] = np.asarray(pos, float)
    S.v[:], S.w[:] = 0.0, 0.0
    S.regime[:] = ACTIVE
    return S


def _cap_run(S, dur):
    S.duration = S.t + dur
    S.run(progress=False)
    return S


def _cap_touching(S):
    """Touching contacts as {(i, j): row}."""
    nr = len(S._ckeys)
    return {tuple(int(a) for a in S._ckeys[r]): int(r) for r in np.flatnonzero(S._hist[:nr, dem.H_TOUCH] > 0.5)}


def _cap_state(S, key):
    """One contact's ratios (pull-off, sliding, rolling, twisting), spring constants and limits (statics.state with
    the caps lifted: the springs as they are)."""
    i, j = key
    r = S._key[key]
    mem = np.array([i] if j < 0 else [i, j])
    ci, cj = np.array([0]), np.array([j if j < 0 else 1])
    prm, h = np.ascontiguousarray(S._prm[[r]]), np.ascontiguousarray(S._hist[[r]])
    lam, _, out = statics.state(np.ascontiguousarray(S.x[mem]), ci, cj, statics.lift_caps(prm), h, S.L)
    return (np.array([float(a[0]) for a in statics.ratios_by_kind(lam, prm, out)]), statics.kcomp(prm, h)[0],
            statics.caps(lam, prm, out)[0])


def _cap_set(S, keys, kind, d, rel=2 ** -40):
    """Set the rolling (kind 1, along the unit vector d) or twisting (kind 2, sign d) springs of the contacts keys to
    (1 - rel) of their limit: spring force -k h = (1 - rel) cap along d."""
    for key in keys:
        r = S._key[key]
        _, k, cp = _cap_state(S, key)
        if kind == 1:
            u = np.asarray(d, float) / np.linalg.norm(d)
            S._hist[r, dem.H_XR:dem.H_XR + 3] = -(1 - rel) * cp[1] / k[1] * u
        else:
            S._hist[r, dem.H_XTW] = -(1 - rel) * cp[2] / k[2] * float(np.sign(d))


def _cap_equilibrate(S, mem):
    """Put the grains mem (an anchored cluster) at the exact equilibrium of the current load, at rest, from their
    present state: the lifted-cap Newton solve the engine places islands with (statics.solve_equilibrium), its springs
    as they are. Returns the largest displacement it made."""
    from scipy.sparse.linalg import splu
    mem = np.asarray(mem, np.int64)
    T = float(S.env.t0_flight + S.t)
    ci, cj, rows = S._island_system(mem, list(_cap_touching(S)))
    x, prm, h = (np.ascontiguousarray(a) for a in (S.x[mem], S._prm[rows], S._hist[rows]))
    pe = statics.lift_caps(prm)
    K, _ = statics.jacobians(x, S.rad[mem], ci, cj, pe, h, S.L)
    lu = splu(K)
    lam, Fnet, out = statics.state(x, ci, cj, pe, h, S.L)
    tt, tr = S._island_tols(statics.caps(lam, prm, out), prm, h[:, dem.H_TOUCH] > 0.5)
    F = S._island_load(mem, T)
    E = statics.solve_equilibrium(x, ci, cj, pe, pe, h, S.L, lu, F, lu.solve(F + Fnet), tt, tr, 20)
    if not E["ok"]:
        raise RuntimeError(f"no equilibrium ({S._island_flags(E)})")
    S.x[mem], S._hist[rows] = E["xn"], E["hn"]
    S.v[mem], S.w[mem] = 0.0, 0.0
    return float(np.abs(E["xn"] - x).max())


def _cap_loads(S, fn, curv=None):
    """Replace the uniform load by fn(T) (n x 3, m/s^2) at flight times T: the profile term of loads.Loads, so the
    DEM, the certificates and the motion bounds all see it; curv(T0, T1) (3,) is added to Loads.chord_dev, the
    loads' exact deviation from their chord inside a step when fn is not linear in time."""
    L = S.loads
    L.profile_accel = lambda T: fn(np.atleast_1d(np.asarray(T, float)))
    if curv is not None:
        base = type(L).chord_dev.__get__(L)
        L.chord_dev = lambda T0, T1: base(T0, T1) + curv(T0, T1)


def _cap_engine(P, dur, setup=None, record=False):
    """Run the prepared all-DEM state P for dur with the engine (sleeping and waking clusters). Returns the run and
    the engine's reported bounds accumulated per grain (N x 3 kinds: sliding, rolling [m], twisting [rad]): the sum
    over every sleep of the island's flow_tol + ratchet + path + curv + snap (its bnd when woken, or at the end), and
    with record every commit as (iid, rows, histories before, after, the committed island dict with x, hist, the
    members' positions before, the island dict before)."""
    import copy
    S = copy.deepcopy(P)
    if setup:
        setup(S)
    S.all_dem = False
    acc = np.zeros((S.N, 3))
    acc_pc = np.zeros((S.N, 3))
    commits = []

    def add(isl):
        b = isl.get("bnd")
        if b is not None:
            acc[isl["members"]] += b["flow_tol_m"] + b["ratchet_m"] + b["path_m"] + b["curv_m"] + b["snap_m"]
            acc_pc[isl["members"]] += b["path_m"] + b["curv_m"]
    wake, commit = S._wake_island, S._island_commit

    def wake_rec(iid, T):
        add(S._islands[iid])
        return wake(iid, T)

    def commit_rec(iid, new):
        rows = np.array(new["rows"])
        before, xb, old = S._hist[rows].copy(), S.x[new["members"]].copy(), S._islands[iid]
        snap = dict(new, x=new["x"].copy(), hist=new["hist"].copy())
        commit(iid, new)
        commits.append((iid, rows, before, S._hist[rows].copy(), snap, xb, old))
    S._wake_island = wake_rec
    if record:
        S._island_commit = commit_rec
    _cap_run(S, dur)
    for isl in S._islands.values():
        add(isl)
    del S._wake_island
    S.__dict__.pop("_island_commit", None)
    S._acc_pc = acc_pc
    return S, acc, commits


def _cap_dem(P, dur, n_ppp, setup=None):
    """The same in the full model throughout, at steps_per_period n_ppp."""
    import copy
    S = copy.deepcopy(P)
    if setup:
        setup(S)
    S.n_ppp = n_ppp
    S._dt_dem = S._contact_dt()
    return _cap_run(S, dur)


def _cap_compare(E, A, B, acc, keys, acc_springs=None):
    """Engine E vs the full model A (and B at half its step): the largest differences of the positions and of the
    sliding, rolling and twisting springs of the contacts keys, A's own step error |A - B| for each, the engine's
    reported bound for each (positions: the larger of the sliding and rolling bounds of each grain), and whether
    every one is within 3x the step error plus the bound. Returns (ok, dict of (diff, step error, bound) by name)."""
    out = {"x": (float(np.abs(E.x - A.x).max()), float(np.abs(A.x - B.x).max()), float(acc[:, :2].max()))}
    sl = {"xt": slice(dem.H_XT, dem.H_XT + 3), "xr": slice(dem.H_XR, dem.H_XR + 3), "xtw": slice(dem.H_XTW, dem.H_XTW + 1)}
    for q, (nm, s) in enumerate(sl.items()):
        d = e = b = 0.0
        for key in keys:
            rE, rA, rB = E._key[key], A._key[key], B._key[key]
            d = max(d, float(np.abs(E._hist[rE, s] - A._hist[rA, s]).max()))
            e = max(e, float(np.abs(A._hist[rA, s] - B._hist[rB, s]).max()))
            g = [key[0]] + ([key[1]] if key[1] >= 0 else [])
            b = max(b, float((acc if acc_springs is None else acc_springs)[g, q].max()))
        out[nm] = (d, e, b)
    return all(d <= 3 * e + b for d, e, b in out.values()), out


def _cap_fmt(out):
    return "; ".join(f"{k} {d:.1e} (step error {e:.1e}, bound {b:.1e})" for k, (d, e, b) in out.items())


def _cap_corner_prep():
    """A grain in an edge (floor and -x wall) and a 3-grain cluster in the same edge (a grain in the edge, one beside
    it on the floor, one beside it along the edge), 50 um silica on glass, under 2 g pressing (-z) and 8 g along the
    floor into the wall (-x, the coast's centrifugal load): 8 ms in the full model from contact, every floor rolling
    spring set to (1 - 2^-40) of its limit along +x (the direction it rolled in), 19 ms more to settle. Cached."""
    if "corner" not in _CAP_PREP:
        R, d = 50e-6, 2 * 50e-6 - 2e-9
        S = _cap_sim([[R, 0.25e-3, R], [R, 0.7e-3, R], [R + d, 0.7e-3, R], [R, 0.7e-3 + d, R]], (-8.0, 0.0, -2.0))
        _cap_run(S, 8e-3)
        floor = [k for k in _cap_touching(S) if k[1] == -5]
        _cap_set(S, floor, 1, [1.0, 0.0, 0.0])
        _cap_run(S, 19e-3)
        _CAP_PREP["corner"] = (S, floor)
    return _CAP_PREP["corner"]


def check_cap_sleep_vs_dem():
    """A grain in an edge and a 3-grain corner cluster, their floor rolling springs at the limit, put to sleep by the
    engine (tier 1 static, tier 2 creep) under constant, shrinking and turning loads, against the full model."""
    import time
    t0 = time.time()
    G0 = 9.80665
    P, floor = _cap_corner_prep()
    keys = sorted(_cap_touching(P))
    r0 = min(float(_cap_state(P, k)[0][2]) for k in floor)
    T0, dur = float(P.env.t0_flight + P.t), 30e-3
    om = math.radians(30.0) / dur

    def press(T):
        a = np.zeros((len(T), 3))
        a[:, 0] = -8 * G0
        a[:, 2] = -(2.0 - 1.8 * np.clip((T - T0) / dur, 0.0, 1.0)) * G0
        return a

    def turn(T):
        th = om * np.clip(T - T0, 0.0, dur)
        return np.column_stack([-8 * G0 * np.cos(th), -8 * G0 * np.sin(th), np.full(len(T), -2 * G0)])
    cases = {"constant": None, "pressing 2 -> 0.2 g": (press, None),
             "turning 30 deg": (turn, lambda a, b: np.array([1.0, 1.0, 0.0]) * 8 * G0 * om * om * (b - a) ** 2 / 8)}
    vals, ok_all = [], r0 >= 0.999
    for name, ld in cases.items():
        setup = None if ld is None else (lambda S, ld=ld: _cap_loads(S, *ld))
        E, acc, commits = _cap_engine(P, dur, setup, record=ld is None)
        A, B = _cap_dem(P, dur, 50, setup), _cap_dem(P, dur, 100, setup)
        ne = E.n_events
        n_steps = int(round(dur / E.base.dt_base))
        asleep = (ne.get("island_static_steps", 0) + ne.get("island_creep_steps", 0)) / (2 * n_steps)
        same = sorted(_cap_touching(E)) == sorted(_cap_touching(A)) == sorted(_cap_touching(B))
        ok, out = _cap_compare(E, A, B, acc, keys)
        ok = ok and same and asleep >= 0.9
        txt = (f"{name}: asleep {asleep:.0%} of island-steps (static {ne.get('island_static_steps', 0)}, creep "
               f"{ne.get('island_creep_steps', 0)}, sleeps {ne.get('cluster_sleep', 0)}, wakes "
               f"{sum(v for k, v in ne.items() if k.startswith('island_wake_') and k != 'island_wake_transient_m')}); "
               f"same contacts {same}; engine vs full model: {_cap_fmt(out)}")
        if ld is None:
            placed = {}
            for iid, _, b, _, _, xb, old in commits:
                placed.setdefault(iid, (b, xb, old["F_ref"], old["F_lr_ref"]))
            n_same = n_diff = 0
            d_lr = (0.0, 0.0)
            for iid, _, _, a, c, _, _ in commits:
                hb, xb, Fb, Flb = placed[iid]
                moved = not (np.array_equal(a, hb) and np.array_equal(c["x"], xb))
                if np.array_equal(c["F_ref"], Fb) and np.array_equal(c["F_lr_ref"], Flb):
                    n_same += 1
                    n_diff += moved
                else:
                    d_lr = max(d_lr, (float(np.abs(c["F_lr_ref"] - Flb).max()), float(np.abs(c["x"] - xb).max())))
            ok = ok and n_same > 0 and n_diff == 0
            txt += (f"; of {n_same} commits at a load bit-identical to the placement's, {n_diff} not bit-identical to "
                    f"the placed state (histories and positions); {len(commits) - n_same} after the long-range "
                    f"forces were recomputed (change {d_lr[0]:.1e} N, positions moved {d_lr[1]:.1e} m)")
        ok_all = ok_all and ok
        vals.append(txt)
    sec = time.time() - t0
    return _res("cap_sleep_vs_dem", "Clusters asleep at their rolling limits vs all-DEM",
                "A 50 um silica grain in the edge of a glass floor and wall and a 3-grain cluster in the same edge, "
                "under 2 g pressing and 8 g along the floor into the wall; every floor rolling spring set to (1 - "
                "2^-40) of its limit and settled in the full model (smallest floor rolling ratio at the start "
                f"{r0:.6f}). Then 30 ms by the engine (the islands asleep at their limits: tier 1 static or tier 2 "
                "creep) and in the full model throughout at 50 and 100 steps per contact period, under (a) constant "
                "loads, (b) the pressing load falling linearly to 0.2 g (the rolling limits shrink: creep), (c) the "
                "sideways load turning by 30 degrees in the floor plane at constant size. The islands must be asleep "
                ">= 90% of the island-steps (the first step is in the full model before they are classified), the "
                "same contacts must touch at the end, and the positions and the sliding, rolling and twisting "
                "springs of every contact must match the full model within 3x its own step error (50 vs 100) plus "
                "the bound the engine reports for the island (flow tolerance + entry ratchet + path + load curvature "
                "+ snap, summed over every sleep of it); in (a) every state the engine commits (histories and "
                "positions) at a load bit-identical to the one it was placed at must be bit-identical to the placed "
                "state (this found the kernel's round-off at zero displacement, now skipped in "
                "statics.solve_equilibrium); the long-range forces on the grains (1e-18 N), frozen while steps are "
                "skipped, are recomputed after a skip at the positions the placement moved by 4e-14 m, a load change "
                "the engine follows exactly, so those commits are counted apart and their change reported. "
                f"Runtime {sec:.0f} s.",
                "; ".join(vals), "start ratio >= 0.999; asleep >= 90%; same contacts; each <= 3x step error + "
                "reported bound; (a) bit-identical", ok_all)


def check_cap_twist_microgravity(dur=0.5):
    """A grain in an edge held by locked-in contact forces with both twisting springs at 0.96 of their limit, in
    microgravity under a slowly turning residual load: it sleeps at once, stays asleep through skipped steps, and
    its twist is neither lost nor changed, as in the full model."""
    import time
    from .engine import ASLEEP
    t0 = time.time()
    G0, R = 9.80665, 50e-6
    S = _cap_sim([[R, 0.5e-3, R]], (-8.0, 0.0, -2.0))
    _cap_run(S, 8e-3)                               # pressed into the edge: contact forces locked in by the springs
    T0, f_spin = float(S.env.t0_flight + S.t), 1.0 / 60.0

    def resid(T):                                    # 1e-4 g turning with the payload's residual spin
        ph = 2 * math.pi * f_spin * (T - T0)
        return 1e-4 * G0 * np.column_stack([-np.cos(ph), -np.sin(ph), np.zeros(len(T))])

    def curv(a, b):
        return np.array([1.0, 1.0, 0.0]) * 1e-4 * G0 * (2 * math.pi * f_spin) ** 2 * (b - a) ** 2 / 8
    setup = lambda Q: _cap_loads(Q, resid, curv)
    setup(S)
    _cap_equilibrate(S, [0])
    keys = [(0, -5), (0, -1)]
    st = [_cap_state(S, k) for k in keys]
    M = 0.96 * min(c[2] for _, _, c in st)
    for key, (_, k, _) in zip(keys, st):            # equal twisting torques: what the edge's statics requires
        S._hist[S._key[key], dem.H_XTW] = -M / k[2]
    _cap_equilibrate(S, [0])
    st = [_cap_state(S, k) for k in keys]
    tw0 = np.array([S._hist[S._key[k], dem.H_XTW] for k in keys])
    ang = np.array([c[2] / k[2] for _, k, c in st])  # the limit angle of each twisting spring
    E, _, _ = _cap_engine(S, dur, setup)
    A, B = _cap_dem(S, dur, 50, setup), _cap_dem(S, dur, 100, setup)
    ne = E.n_events
    n = int(round(dur / E.base.dt_base))
    asleep = ne.get("island_static_steps", 0) + ne.get("island_creep_steps", 0)
    first = ne.get("cluster_sleep", 0) == 1 and asleep == n - 1 and E.regime[0] == ASLEEP
    skipped = float(ne.get("skipped_s", 0.0))
    dtw = {nm: np.abs(np.array([Q._hist[Q._key[k], dem.H_XTW] for k in keys]) - tw0) / ang for nm, Q in
           (("engine", E), ("full model", A))}
    keep = {nm: float((np.abs(np.array([Q._hist[Q._key[k], dem.H_XTW] for k in keys])) / np.abs(tw0)).min())
            for nm, Q in (("engine", E), ("full model", A))}
    dx, ex = float(np.abs(E.x - A.x).max()), float(np.abs(A.x - B.x).max())
    ok = (first and skipped > 0 and all(float(d.max()) <= 1e-6 for d in dtw.values())
          and all(k >= 0.9 for k in keep.values()) and dx <= 3 * ex + 1e-12)
    sec = time.time() - t0
    r0 = [float(r[3]) for r, _, _ in st]
    return _res("cap_twist_microgravity", "Locked-in twist in microgravity (sleeping)",
                "A 50 um silica grain pressed into the edge of a glass floor and wall (2 g and 8 g, 8 ms in the full "
                "model: the contact forces are locked in by the sliding springs, pull-off ratio "
                f"{float(st[0][0][0]):.3f}, sliding {float(st[0][0][1]):.3f}), both twisting springs then set to an "
                "equal torque of 0.96 of the smaller twisting limit (the edge's statics needs them equal) and placed "
                f"at the exact equilibrium (twisting ratios {r0[0]:.4f}, {r0[1]:.4f}), under 1e-4 g turning at the "
                f"payload's residual spin of 1/60 Hz. {dur:g} s by the engine (asleep, every step skipped while the "
                "certificate holds) and in the full model throughout at 50 and 100 steps per contact period. The "
                "grain must sleep at the first classification and stay asleep, steps must be skipped, each twisting "
                "spring must change by <= 1e-6 of its limit angle and keep >= 90% of its twist in both models (the "
                "kernel resets an over-limit twist at zero twist rate: it must not be lost), and the position must "
                f"match the full model within 3x its step error + 1e-12 m. Runtime {sec:.0f} s.",
                f"asleep from the first classification: {first} ({asleep} of {n} steps), skipped {skipped:.3f} s; "
                f"twist change / limit angle: engine {float(dtw['engine'].max()):.1e}, full model "
                f"{float(dtw['full model'].max()):.1e}; twist kept: engine {keep['engine']:.6f}, full model "
                f"{keep['full model']:.6f}; position engine vs full model {dx:.1e} m (step error {ex:.1e} m)",
                "first classification; skipped > 0; <= 1e-6; >= 0.9; <= 3x step error + 1e-12 m", ok)


def _cap_at_limit(S, keys, mem):
    """Set the rolling springs of the contacts keys back to (1 - 2^-40) of their limit along their present direction
    and place the grains mem at the exact equilibrium of the current load (_cap_equilibrate), three times: the full
    model's settling relaxes a spring set at its limit by a few 1e-4 of it, so this puts the state back at the limit
    as an exact equilibrium of the contact law."""
    for _ in range(3):
        for key in keys:
            _cap_set(S, [key], 1, -S._hist[S._key[key], dem.H_XR:dem.H_XR + 3])
        _cap_equilibrate(S, mem)


def _cap_pile_prep(name, a_g):
    """A 2-grain stack against a wall (one grain in the edge of the floor and the -x wall, one on top of it against
    the wall) and a 5-grain pile in the corner of the floor and the -x and -y walls (four on the floor in a square,
    one on top of them), 50 um silica on glass, under a_g [g]: 8 ms in the full model from contact, every floor
    rolling spring set to (1 - 2^-40) of its limit along +x, 19 ms more to settle, then set back to the limit at the
    exact equilibrium (_cap_at_limit). Cached by name."""
    if name not in _CAP_PREP:
        R, d = 50e-6, 2 * 50e-6 - 2e-9
        h = d / math.sqrt(2.0)
        pos = [[R, 0.6e-3, R], [R, 0.6e-3, R + d],
               [R, R, R], [R + d, R, R], [R, R + d, R], [R + d, R + d, R], [R + d / 2, R + d / 2, R + h]]
        S = _cap_sim(pos, a_g)
        _cap_run(S, 8e-3)
        floor = [k for k in _cap_touching(S) if k[1] == -5]
        _cap_set(S, floor, 1, [1.0, 0.0, 0.0])
        _cap_run(S, 19e-3)
        _cap_at_limit(S, [k for k in floor if k[0] < 2], [0, 1])
        _cap_at_limit(S, [k for k in floor if k[0] >= 2], [2, 3, 4, 5, 6])
        _CAP_PREP[name] = (S, floor)
    return _CAP_PREP[name]


def _count_solves(run):
    """run() with every statics.solve_equilibrium call recorded: (real kernel, iterations, converged) per call."""
    rec = []
    orig = statics.solve_equilibrium

    def wrapped(x, ci, cj, prm_apply, prm_el, *a, **k):
        E = orig(x, ci, cj, prm_apply, prm_el, *a, **k)
        rec.append((prm_apply is not prm_el, E["n_it"], bool(E["ok"])))
        return E
    statics.solve_equilibrium = wrapped
    try:
        out = run()
    finally:
        statics.solve_equilibrium = orig
    return out, rec


def check_creep_island_vs_dem(dur=30e-3):
    """Clusters creeping at their rolling limits (tier 2: the kernel's return map solved to equilibrium every step)
    under the burn's growing sideways load and the coast's shrinking pressing load, against the full model; and a
    true mechanism, which must not sleep."""
    import copy
    import time
    from .engine import ASLEEP
    t0 = time.time()
    G0 = 9.80665
    c30, s30 = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    vals, ok_all = [], True

    def burn(T, T0):
        a = (3.8 + 4.2 * np.clip((T - T0) / 0.2, 0.0, 1.0)) * G0
        return np.column_stack([-c30 * a, -s30 * a, np.full(len(T), -25 * G0)])

    def coast(T, T0):
        return np.column_stack([np.full(len(T), -8 * G0), np.zeros(len(T)),
                                -(2.0 - 1.8 * np.clip((T - T0) / dur, 0.0, 1.0)) * G0])
    for name, a_g, fn in (("burn", (-3.8 * c30, -3.8 * s30, -25.0), burn), ("coast", (-8.0, 0.0, -2.0), coast)):
        P, floor = _cap_pile_prep(name, a_g)
        keys = sorted(_cap_touching(P))
        r0 = min(float(_cap_state(P, k)[0][2]) for k in floor)
        xi = min(float(c[1] / k[1]) for _, k, c in (_cap_state(P, q) for q in keys))
        T0 = float(P.env.t0_flight + P.t)
        setup = lambda S, fn=fn, T0=T0: _cap_loads(S, lambda T: fn(T, T0))
        (E, acc, _), rec = _count_solves(lambda: _cap_engine(P, dur, setup))
        A, B = _cap_dem(P, dur, 50, setup), _cap_dem(P, dur, 100, setup)
        ne = E.n_events
        st, cr = ne.get("island_static_steps", 0), ne.get("island_creep_steps", 0)
        n_steps = int(round(dur / E.base.dt_base))
        creep = cr / max(st + cr, 1)
        real = [n for r, n, _ in rec if r]
        it_step = ne.get("island_newton_iters", 0) / max(cr, 1)
        same = sorted(_cap_touching(E)) == sorted(_cap_touching(A)) == sorted(_cap_touching(B))
        _, out = _cap_compare(E, A, B, acc, keys, E._acc_pc)
        dx, ex = out.pop("x")[:2]
        oks = all(d <= 3 * e + b for d, e, b in out.values())
        bxi = float(ne.get("island_bound_xi", 0.0))
        wakes = {k[12:]: v for k, v in ne.items() if k.startswith("island_wake_") and v and k != "island_wake_transient_m"}
        ok = (r0 >= 0.999 and cr > 0 and creep >= 0.9 and (st + cr) >= 0.9 * 2 * n_steps and it_step <= 10
              and same and oks and dx <= 3 * ex + 1e-3 * xi and bxi <= 1e-2)
        ok_all = ok_all and ok
        vals.append(f"{name}: start ratio {r0:.9f}; creep {cr} of {st + cr} sleeping island-steps ({creep:.1%}) of "
                    f"{2 * n_steps}; Newton iterations per creep step {it_step:.1f} (island_newton_iters "
                    f"{ne.get('island_newton_iters', 0)}; return-map solves {len(real)}, mean "
                    f"{float(np.mean(real)) if real else 0.0:.1f}, max {max(real, default=0)} iterations); wakes "
                    f"{wakes}; same contacts {same}; springs vs full model: {_cap_fmt(out)}; positions {dx:.1e} m (step "
                    f"error {ex:.1e}, 1e-3 xi_cap {1e-3 * xi:.1e}); island_bound_xi {bxi:.1e}, flow "
                    f"{float(ne.get('island_flow_m', 0.0)):.1e} m")
    # a true mechanism: two grains stacked on the floor away from the walls, pressed (25 g) until at rest, then the
    # burn's sideways load for 20 us in the full model (the springs take the direction it loads them in), both
    # rolling springs set to (1 - 2^-40) of their limit along it
    R, d = 50e-6, 2 * 50e-6 - 2e-9
    M = _cap_sim([[0.5e-3, 0.5e-3, R], [0.5e-3, 0.5e-3, R + d]], (0.0, 0.0, -25.0))
    _cap_run(M, 1e-3)
    T0m = float(M.env.t0_flight + M.t)
    setup = lambda S: _cap_loads(S, lambda T: burn(T, T0m))
    setup(M)
    _cap_run(M, 20e-6)
    mk = sorted(_cap_touching(M))
    for key in mk:
        _cap_set(M, [key], 1, -M._hist[M._key[key], dem.H_XR:dem.H_XR + 3])
    rm = [float(_cap_state(M, k)[0][2]) for k in mk]
    Q = copy.deepcopy(M)
    Q.all_dem = False
    entry = Q._try_sleep_island(np.array([0, 1]), list(_cap_touching(Q)), float(Q.env.t0_flight + Q.t), 1e-3)
    refused = {k[14:]: v for k, v in Q.n_events.items() if k.startswith("cluster_awake_") and v}
    Em, _, _ = _cap_engine(M, 10e-3, setup)
    Am = _cap_dem(M, 10e-3, 50, setup)
    rolled = float(np.abs(Am.x - M.x).max())
    nem = Em.n_events
    mech = nem.get("cluster_awake_mechanism", 0) + nem.get("island_wake_mechanism", 0)
    slept = nem.get("island_static_steps", 0) + nem.get("island_creep_steps", 0)
    okm = (len(mk) == 2 and min(rm) >= 0.999 and not entry and refused == {"mechanism": 1} and slept == 0
           and nem.get("cluster_sleep", 0) == 0 and not (Em.regime == ASLEEP).any() and rolled > 1e-7)
    ok_all = ok_all and okm
    rms = ", ".join(f"{r:.6f}" for r in rm)
    vals.append(f"mechanism: {len(mk)} contacts, rolling ratios {rms}; entry attempt at rest: slept {entry}, "
                f"refused as {refused}; engine run: cluster sleeps {nem.get('cluster_sleep', 0)}, island-steps asleep "
                f"{slept}, refused or woken as a mechanism {mech} times; the full model rolls it {rolled * 1e6:.2f} um "
                "in 10 ms")
    sec = time.time() - t0
    return _res("creep_island_vs_dem", "Creeping clusters at their rolling limits vs all-DEM",
                "A 2-grain stack against a glass wall (one grain in the edge of floor and wall, one on top of it "
                "against the wall) and a 5-grain pile in a corner (four on the floor, one on top), 50 um silica, every "
                "floor rolling spring set to (1 - 2^-40) of its limit and settled in the full model, then set back to "
                "the limit at the exact equilibrium. Loads: the burn's (25 g pressing; sideways 3.8 g growing at the "
                "rate of the design's verification list, 3.8 -> 8 g over 0.2 s, ASSUMED burn-like; run for its first "
                f"{dur * 1e3:.0f} ms for runtime, directed 30 degrees off the -x wall's normal towards -y so that it "
                "loads the springs along the wall: a sideways load straight into a wall is carried by its normal "
                "forces, and with it no step of the stack or pile crept in a first version of this check) and the "
                "coast's (8 g into the "
                f"wall; pressing 2 -> 0.2 g over {dur * 1e3:.0f} ms). The engine (islands creeping: tier 2) vs the "
                "full model at 50 and 100 steps per contact period. Required: creep steps >= 90% of the sleeping "
                "island-steps and the islands asleep >= 90% of the steps; island_newton_iters / island_creep_steps "
                "<= 10 (every solve of every sleeping step counted: the two lifted solves and the three return-map "
                "solves of a creep step, and the static steps' solves too); the sliding, rolling and twisting springs "
                "within 3x the full model's step error + the reported path_m + curv_m; positions within 3x the step "
                "error + 1e-3 of the smallest deflection at the limit; the same contacts; island_bound_xi <= 1e-2. "
                "And a true mechanism (two grains stacked on the floor away from the walls, both rolling springs at "
                "the limit along the burn's sideways load): an entry attempt on it at rest must be refused as a "
                "mechanism (cluster_awake_mechanism), and the engine must never put it to sleep while the full model "
                f"shows it rolling (> 0.1 um in 10 ms). Runtime {sec:.0f} s.",
                "; ".join(vals), "start ratio >= 0.999; creep >= 90%; asleep >= 90%; Newton iterations per creep step "
                "<= 10; springs <= 3x step error + path + curv; positions <= 3x step error + 1e-3 xi_cap; same "
                "contacts; bound <= 1e-2; mechanism refused as such, never asleep, rolling in the full model", ok_all)


def _cap_flight(T, pos, settle=5e-3):
    """Grains at pos in a 1 mm glass box 5 cm off the spin axis under the nominal flight profile (flight.py: thrust,
    drag, spin, de-spin, descent, splash), settled for settle s in the full model so that the run reaches flight time
    T, then put to sleep as one cluster (_try_sleep_island). Returns the run and the island's id."""
    env = {"preset": "ideal", "pressure_pa": 0.0, "start_time_s": T - settle, "flight_profile": {}}
    S = _cap_sim(pos, env=env, chamber={"center_from_spin_axis_m": [0.05, 0.0, 0.0],
                                         "center_from_cg_m": [0.05, 0.0, 0.0]}, dt=4e-3)
    _cap_run(S, settle)
    S.all_dem = False
    S.duration = S.t + 1e3                          # (the outer step is limited by the run's end)
    mem = np.arange(S.N)
    Tn = float(S.env.t0_flight + S.t)
    H = S._outer_dt(S.t)
    if not S._try_sleep_island(mem, list(_cap_touching(S)), Tn, H):
        raise RuntimeError(f"the cluster at T+{Tn:.3f} s did not sleep: "
                           f"{ {k: v for k, v in S.n_events.items() if k.startswith('cluster_awake') and v} }")
    return S, S._island_of[0]


def check_between_sample_bound(n_grid=20):
    """The sleeping-cluster certificate's worst case over a step (the loads' chord deviation, Newton residuals,
    drift dynamics and the path's measured curvature, _island_advance) against the exact equilibria on a fine grid
    inside every step, through the nominal flight's drag decay, splash, spin-up, a descent-table knot and the
    de-spin knot."""
    import time
    from .flight import FlightProfile
    t0 = time.time()
    prof = FlightProfile()
    R, d, L = 50e-6, 2 * 50e-6 - 2e-9, 1e-3

    def row(face_pos):
        return [face_pos(y) for y in (0.5e-3 - d, 0.5e-3, 0.5e-3 + d)]
    edge_xz = row(lambda y: [L - R, y, L - R])                  # in the edge of the +x wall and the +z face
    kn = [k for k in prof.kinks_in(400.0, 401.0)][0]           # a point of the descent table (a slope jump)
    windows = [  # name, start, cluster, step (None: the engine's outer step), steps, a knot to straddle
        ("drag decay after burnout", 29.5, edge_xz, None, 100, None),
        ("splash", prof.splash_s - 5e-3, row(lambda y: [0.5e-3, y, L - R]), 5e-4, 110, None),
        ("spin-up (coast between burns)", 15.0, edge_xz, None, 100, None),
        ("descent-table knot", kn - 0.2, row(lambda y: [R, y, 0.5e-3]), None, 100, kn),
        ("de-spin knot", prof.despin_s - 0.06, edge_xz, None, 180, prof.despin_s)]
    vals, viol_all, n_all, slack_min, ok_all = [], 0, 0, math.inf, True
    for name, T_s, pos, H_fix, n_steps, knot in windows:
        S, iid = _cap_flight(T_s, pos)
        T = float(S.env.t0_flight + S.t)
        rec = []
        ratios = S._ratios

        def rec_ratios(lam, D, prm, out, tm):
            r = ratios(lam, D, prm, out, tm)
            if D is not None:
                rec.append(r)
            return r
        S._ratios = rec_ratios
        n_v = n_g = steps = 0
        sl_min, sl_max, worst = math.inf, 0.0, 0.0
        straddled = False
        for q in range(n_steps):
            H = H_fix or S._outer_dt(T - S.env.t0_flight)
            if knot is not None and T < knot < T + H + 1e-12:
                H = max(2 * (knot - T), 1e-6)      # straddle the knot, with it at mid-step
                straddled = True
            isl = S._islands[iid]
            rec.clear()
            ok, new, why = S._island_advance(isl, T, T + H, count=False)
            if not ok:
                raise RuntimeError(f"{name}: the certificate failed at T+{T:.4f} s ({why})")
            (p0, s0), (p1, s1) = rec[0], rec[1]
            bp, bs = np.maximum(p0, p1), np.maximum(s0, s1)
            mem, ci, cj, prm, prm_el = isl["members"], isl["ci"], isl["cj"], isl["prm"], isl["prm_el"]
            x, hist = np.ascontiguousarray(S.x[mem]), np.ascontiguousarray(S._hist[isl["rows"]])
            tm = hist[:, dem.H_TOUCH] > 0.5
            tt, tr = S._island_tols(isl["capv"], prm, tm)
            lu = S._island_lu(isl)
            for Tg in T + H * (np.arange(1, n_grid + 1) / (n_grid + 1)):
                F = S._island_load(mem, Tg)
                Eg = statics.solve_equilibrium(x, ci, cj, prm_el, prm_el, hist, S.L, lu, F,
                                               lu.solve(F - isl["F_ref"]), 1e-3 * tt, 1e-3 * tr, 12)
                pg, sg = ratios(Eg["lam"], None, prm, Eg["out"], tm)
                gap = np.concatenate([bs.reshape(-1) - sg.reshape(-1), bp - pg])
                ex = np.concatenate([sg.reshape(-1), pg])
                n_v += int((gap < 0).sum())
                n_g += 1
                big = ex >= 0.1                    # the slack of the ratios that matter
                sl_min = min(sl_min, float(gap[big].min(initial=math.inf)))
                sl_max = max(sl_max, float(gap[big].max(initial=0.0)))
                worst = max(worst, float(sg.max()))
            S._island_commit(iid, new)
            T += H
            steps += 1
        S.__dict__.pop("_ratios", None)
        viol_all += n_v
        n_all += n_g
        slack_min = min(slack_min, sl_min)
        ok_all = ok_all and n_v == 0 and steps >= 100 and (knot is None or straddled)
        vals.append(f"{name} (T+{T_s:.3f} s, {steps} steps{', one straddling the knot' if straddled else ''}): "
                    f"violations {n_v} of {n_g} grid points; worst exact ratio {worst:.4f}; slack {sl_min:.1e} to "
                    f"{sl_max:.1e}")
    sec = time.time() - t0
    return _res("between_sample_bound", "The certificate's worst case between steps (flight loads)",
                "A resting row of three 50 um silica grains on a glass wall or in an edge, 5 cm off the spin axis, "
                "under the nominal flight profile (flight.py), put to sleep by the engine and advanced step by step by "
                "its certificate (_island_advance: the exact equilibria at the step's end and middle, the bound D on "
                "how far every contact quantity can stray from the secant between them). At "
                f"{n_grid} points inside every step the exact equilibrium (lifted-cap Newton to 1e-3 of the engine's "
                "tolerance) is solved and its pull-off, sliding, rolling and twisting ratios compared with the "
                "certificate's bounded worst ratios of that step: the drag decay after burnout (T+29.5 s; drag and "
                "4.8 Hz spin), the splash half-sine (0.5 ms steps so that 100 cover it), the spin-up in the coast "
                "between burns (T+15 s), a point of the descent table (a slope jump inside one step) and the de-spin "
                "knot (a jump of the Euler load and a kink of the spin, straddled by one step with the knot at "
                "mid-step: the engine never certifies across it, but the bound is stated for any interval). Steps are "
                "the engine's outer steps (4 ms, 0.66 ms while the payload spins at 4.8 Hz). A violation is a grid "
                "point where an exact ratio exceeds the bounded one. Runtime "
                f"{sec:.0f} s.",
                "; ".join(vals) + f"; total violations {viol_all} of {n_all} grid points, smallest slack "
                f"{slack_min:.1e}", "0 violations; >= 100 steps each; the knots straddled", ok_all)


def _cap_system(S, mem):
    """The engine's island system of the grains mem at their present state, as _try_sleep_island forms it: contact
    lists, parameters, histories, the lifted stiffness K and Jacobian J, the contact state and its caps."""
    mem = np.asarray(mem, np.int64)
    ci, cj, rows = S._island_system(mem, list(_cap_touching(S)))
    ci, cj, rows = S._island_near(mem, ci, cj, rows)
    x, prm, h = (np.ascontiguousarray(a) for a in (S.x[mem], S._prm[rows], S._hist[rows]))
    pe = statics.lift_caps(prm)
    K, J = statics.jacobians(x, S.rad[mem], ci, cj, pe, h, S.L)
    lam, Fnet, out = statics.state(x, ci, cj, pe, h, S.L)
    return dict(mem=mem, ci=ci, cj=cj, rows=rows, x=x, prm=prm, pe=pe, h=h, K=K, J=J, lam=lam, Fnet=Fnet, out=out,
                kc=statics.kcomp(prm, h), capv=statics.caps(lam, prm, out), tm=h[:, dem.H_TOUCH] > 0.5)


def check_ratchet_released_bound():
    """Left-over motion at a contact limit: the permanent slip it causes in the full model against the entry's
    released-system bound (_island_ratchet), and against the superseded energy-over-limit bound; and the oscillation
    a step's load ramp leaves in a cluster at its limit placed at rest, against the bounds on it."""
    import copy
    import time
    from scipy.linalg import eigh
    from scipy.sparse.linalg import splu
    t0 = time.time()
    G0, R = 9.80665, 50e-6
    # a grain in an edge under 8 g into the wall and 2 g pressing (as in check 14), its floor rolling spring set to
    # (1 - 2^-40) of its limit along +x
    S = _cap_sim([[R, 0.5e-3, R]], (-8.0, 0.0, -2.0))
    _cap_run(S, 8e-3)
    fl = (0, -5)
    _cap_set(S, [fl], 1, [1.0, 0.0, 0.0])
    _cap_run(S, 19e-3)                               # settled: the spring relaxes to 0.9996 of its limit
    for _ in range(3):                               # back to the limit, at the exact equilibrium
        _cap_set(S, [fl], 1, -S._hist[S._key[fl], dem.H_XR:dem.H_XR + 3])
        _cap_equilibrate(S, [0])
    r0 = float(_cap_state(S, fl)[0][2])
    Q = _cap_system(S, [0])
    kf = int(np.flatnonzero((Q["ci"] == 0) & (Q["cj"] == -5))[0])
    near = np.zeros((len(Q["prm"]), 3), bool)
    near[kf, 1] = True
    fh = S._fhat(Q["lam"])
    Krel = statics.released_stiffness(Q["K"], Q["J"], Q["kc"], near, fh)
    Md = np.array([S.m[0]] * 3 + [S.I[0]] * 3)
    w2, V = eigh((0.5 * (Krel + Krel.T)).toarray(), np.diag(Md))
    g = fh[kf, 1] @ Q["J"][8 * kf + 4:8 * kf + 7].toarray()          # the capped component's row along its force
    mode = int(np.argmax(np.abs(g @ V) / np.sqrt(np.maximum(w2, 1e-300))))
    xr0 = S._hist[S._key[fl], dem.H_XR:dem.H_XR + 3].copy()
    xi_cap = float(Q["capv"][kf, 1] / Q["kc"][kf, 1])
    T_run = 5e-3

    def slip_after(v6):
        """5 ms in the full model from the prepared state with the velocities v6, then the oscillation still
        left removed by placing the grain at its exact equilibrium (the springs' plastic history stays)."""
        D = copy.deepcopy(S)
        D.v[0], D.w[0] = v6[:3], v6[3:]
        _cap_run(D, T_run)
        _cap_equilibrate(D, [0])
        return D
    base = slip_after(np.zeros(6))
    xb = base._hist[base._key[fl], dem.H_XR:dem.H_XR + 3]
    rows1, held, old_fail = [], True, False
    for E in (1e-25, 1e-23, 1e-21, 1e-19, 1e-18):
        rb = S._island_ratchet(Q["K"], Q["J"], Q["prm"], Q["h"], Q["lam"], Q["out"], Q["kc"], Q["capv"], E, Q["mem"])
        bound = float(rb[1][kf, 1])
        meas = 0.0
        for sg in (1.0, -1.0):
            D = slip_after(sg * math.sqrt(2 * E) * V[:, mode])
            meas = max(meas, float(np.linalg.norm(D._hist[D._key[fl], dem.H_XR:dem.H_XR + 3] - xb)))
        old = E / float(Q["capv"][kf, 1])
        held = held and meas <= bound
        old_fail = old_fail or meas > old
        rows1.append(f"E {E:.0e} J: slip {meas:.2e} m, released bound {bound:.2e} m, E/cap {old:.2e} m")
    # (2) load ramps from rest at the limit, H = 1 ms the step: (a) 4 g more into the wall and 1 g less pressing over
    # one step, then held 4 ms; (b) a fifth of that per step, continued for the 5 ms
    H, n_H = 1e-3, 5
    T0 = float(S.env.t0_flight + S.t)
    a0 = np.array([-8.0, 0.0, -2.0]) * G0
    da = np.array([-4.0, 0.0, 1.0]) * G0
    lu = splu(Q["K"])
    tt, tr = S._island_tols(Q["capv"], Q["prm"], Q["tm"])
    isl = dict(K=Q["K"], J=Q["J"], prm=Q["prm"], lu=lu)
    w2el = float(eigh((0.5 * (Q["K"] + Q["K"].T)).toarray(), np.diag(Md), eigvals_only=True)[0])
    fac_el, fac_rel = min(1.0, 4 / (w2el * H * H)), min(1.0, 4 / (float(w2[0]) * H * H))

    def slip_bound(E):
        return float(S._island_ratchet(Q["K"], Q["J"], Q["prm"], Q["h"], Q["lam"], Q["out"], Q["kc"], Q["capv"], E,
                                       Q["mem"])[1][kf, 1])

    def ramp_slip(fn, a_end):
        """The capped spring after 5 ms of the load fn in the full model (the oscillation then left removed by the
        exact equilibrium) beyond the quasi-static answer at the final load a_end: the kernel's return map solved
        to equilibrium from the start state."""
        D = copy.deepcopy(S)
        _cap_loads(D, fn)
        _cap_run(D, n_H * H)
        _cap_equilibrate(D, [0])
        F1 = S._island_load(Q["mem"], T0) + np.concatenate([S.m[0] * (a_end - a0), np.zeros(3)])
        Cq = statics.solve_equilibrium(Q["x"], Q["ci"], Q["cj"], Q["prm"], Q["pe"], Q["h"], S.L, lu, F1,
                                       lu.solve(F1 + Q["Fnet"]), tt, tr, 40, tangent=S._island_tangent(isl, False))
        return float(np.linalg.norm(D._hist[D._key[fl], dem.H_XR:dem.H_XR + 3] - Cq["hn"][kf, dem.H_XR:dem.H_XR + 3]))
    b2 = {}
    for case, fn, a_end, dF in (
            ("(a) over one step, held", lambda T: a0[None, :] + da[None, :] * np.clip((T - T0) / H, 0.0, 1.0)[:, None],
             a0 + da, np.concatenate([S.m[0] * da, np.zeros(3)])),
            ("(b) a fifth per step, continued", lambda T: a0[None, :] + da[None, :] * np.clip(
                (T - T0) / (n_H * H), 0.0, None)[:, None], a0 + da, np.concatenate([S.m[0] * da / n_H, np.zeros(3)]))):
        meas = ramp_slip(fn, a_end)
        W = float(np.abs(dF) @ np.abs(lu.solve(dF)))
        du_el = splu((0.5 * (Q["K"] + Q["K"].T)).tocsc()).solve(dF)
        du_rel = splu((0.5 * (Krel + Krel.T)).tocsc()).solve(dF)
        E_modal = max(2 * float(du_el @ (Md * du_el)), 2 * float(du_rel @ (Md * du_rel))) / H ** 2
        E_drift = S._island_drift(Q["K"], Q["J"], Q["kc"], near, fh, Q["mem"], dF, 0.5 * dF, H)
        b2[case] = (meas, {"factor, elastic omega_min": slip_bound(W * fac_el),
                           "factor, released omega_min": slip_bound(W * fac_rel),
                           "modal 2 du^T M du / H^2": slip_bound(E_modal), "engine (_island_drift)": slip_bound(E_drift)})
    ma, ba = b2["(a) over one step, held"]
    mb, bb = b2["(b) a fifth per step, continued"]
    ok2 = ma <= ba["modal 2 du^T M du / H^2"] and mb <= bb["engine (_island_drift)"]
    ok = r0 >= 0.999 and held and old_fail and ok2
    sec = time.time() - t0
    return _res("ratchet_released_bound", "Left-over motion at a limit: the released-system slip bound",
                "A 50 um silica grain in the edge of a glass floor and wall (8 g into the wall, 2 g pressing), its "
                "floor rolling spring set to (1 - 2^-40) of its limit along the floor, settled 19 ms in the full "
                "model, set back to the limit and placed at the exact equilibrium (rolling ratio "
                f"{r0:.12f}; deflection at the limit "
                f"{xi_cap:.2e} m). (1) Left-over motion of prescribed energy E (1e-25 to 1e-18 J) given as a velocity "
                "along the mode of the released system (the capped spring removed, statics.released_stiffness) that "
                "moves the capped component most per unit energy, both signs, 5 ms in the full model: the permanent "
                "change of that spring against a run without it must stay within the engine's entry bound "
                "(_island_ratchet: the residual's displacement amplitude in the released system plus the dashpot "
                "shift) in every case, and must exceed the superseded bound E / limit in at least one (it treated "
                "the slip as the energy over the limit force; at a limit the load does the plastic work). (2) The "
                "same grain at rest at its limit under load ramps (H = 1 ms, the engine's step): (a) 4 g more into the "
                "wall and 1 g less pressing over one step, then held; (b) a fifth of that per step, continued; 5 ms "
                "in the full model each. The capped spring beyond the quasi-static answer (the kernel's return map "
                "solved to equilibrium at the final load) is the ramp's left-over slip; it is compared with the slip "
                "bounds (_island_ratchet) of the energy of the step's load change dF as charged by the implementer's "
                "former entry test, |dF| |K^-1 dF| min(1, 4 / (omega_min H)^2) with omega_min from the elastic and "
                "from the released stiffness (reported: whether the factor holds and whether the released omega_min "
                "is needed), by the modal sum 2 du^T M du / H^2 (du the shift of the equilibrium in the elastic or "
                "released system, the larger: the energy an undamped linear system keeps from the two kinks of a "
                "ramp over one step, derived here) and by the engine's present entry test (_island_drift: the "
                "kinetic energy relative to the drift of the equilibrium, du^T M du / (2 H^2) from rest, the larger "
                "of the elastic and released systems; a kink later in a sleep is the certificate's, through "
                "Loads.chord_dev). Required: (a) within the modal bound, (b) within the engine's bound. Runtime "
                f"{sec:.0f} s.",
                "(1) " + "; ".join(rows1) + "; (2) " + "; ".join(
                    f"{c}: slip beyond quasi-static {m:.2e} m, bounds " + ", ".join(f"{k} {v:.2e} m" for k, v in bd.items())
                    for c, (m, bd) in b2.items()) + f"; factor elastic {fac_el:.2e}, released {fac_rel:.2e}",
                "start ratio >= 0.999; (1) slip <= released bound in every case, > E/cap in at least one; (2a) slip <= "
                "the modal bound; (2b) slip <= the engine's bound", ok)


def check_island_across_breakpoint():
    """A cluster at its rolling limits asleep up to the de-spin knot (a jump of the Euler load): committed at the
    pre-jump equilibrium, woken there, the jump resolved by the full model; against the full model across the knot;
    and the permanent slip the jump's transient causes, against the released-system bound of its energy."""
    import copy
    import time
    from scipy.sparse.linalg import splu
    from .flight import FlightProfile
    t0 = time.time()
    Tk = FlightProfile().despin_s
    R, d, L = 50e-6, 2 * 50e-6 - 2e-9, 1e-3
    # three grains in the edge of the +x wall (4.6 g of centrifugal load at 4.8 Hz, 5 cm off the axis) and the +z
    # face (the coast's drag), as the cluster of check 14: one in the edge, one beside it on the wall, one beside it
    # along the edge
    pos = [[L - R, 0.5e-3, L - R], [L - R, 0.5e-3, L - R - d], [L - R, 0.5e-3 + d, L - R]]
    env = {"preset": "ideal", "pressure_pa": 0.0, "start_time_s": Tk - 0.03, "flight_profile": {}}
    P = _cap_sim(pos, env=env, chamber={"center_from_spin_axis_m": [0.05, 0.0, 0.0],
                                         "center_from_cg_m": [0.05, 0.0, 0.0]}, dt=4e-3)
    _cap_run(P, 10e-3)
    wall = [k for k in _cap_touching(P) if k[1] == -2]
    _cap_set(P, wall, 1, [0.0, 0.0, -1.0])
    _cap_run(P, 10e-3)
    _cap_at_limit(P, wall, [0, 1, 2])            # (the residual twisting of the settling dropped: see the text)
    r0 = min(float(_cap_state(P, k)[0][2]) for k in wall)
    keys = sorted(_cap_touching(P))
    dur = Tk + 0.01 - float(P.env.t0_flight + P.t)
    # (a) the engine across the knot, its commits recorded
    E, acc, commits = _cap_engine(P, dur, record=True)
    ne = E.n_events
    # the state committed by the step that ends at the knot: re-solved to the load just before and just after it
    rec = [c[4] for c in commits if abs(c[4]["T_ref"] - Tk) <= 1e-9]
    d_pre = d_post = math.nan
    S2 = E
    if rec:
        isl = rec[0]
        xk, hk = isl["x"], isl["hist"]
        mem, ci, cj, prm_el = isl["members"], isl["ci"], isl["cj"], isl["prm_el"]
        tm = hk[:, dem.H_TOUCH] > 0.5
        tt, tr = S2._island_tols(isl["capv"], isl["prm"], tm)
        lu = splu(statics.jacobians(xk, S2.rad[mem], ci, cj, prm_el, hk, S2.L)[0])
        out = []
        for side in ("-", "+"):
            F = S2._island_load(mem, Tk, side)
            Fnet = statics.state(xk, ci, cj, prm_el, hk, S2.L)[1]
            Eq = statics.solve_equilibrium(xk, ci, cj, prm_el, prm_el, hk, S2.L, lu, F, lu.solve(F + Fnet), tt, tr, 12)
            du = Eq["du"].reshape(-1, 6)            # translations, and rotations times the radius
            out.append(max(float(np.abs(du[:, :3]).max()), R * float(np.abs(du[:, 3:]).max())) if Eq["ok"] else math.inf)
        d_pre, d_post = out
    # (b) the full model across the knot at two steps
    A, B = _cap_dem(P, dur, 50), _cap_dem(P, dur, 100)
    same = sorted(_cap_touching(E)) == sorted(_cap_touching(A)) == sorted(_cap_touching(B))
    _, out_c = _cap_compare(E, A, B, acc, keys)
    ok_c = all(d <= 3 * e for d, e, _ in out_c.values())   # the step error alone (the bounds are reported)
    # (c) the jump's transient alone: from the pre-jump equilibrium at the knot, the full model with the loads held at
    # their post-jump values for 5 ms, then placed at the exact equilibrium (the oscillation left removed), against
    # the quasi-static answer (the kernel's return map solved to equilibrium at the post-jump load)
    slip = bound = E_j = jg = math.nan
    if rec:
        D = copy.deepcopy(P)
        mem, rows = isl["members"], isl["rows"]
        D.x[mem], D._hist[rows] = xk, hk
        D.v[mem], D.w[mem] = 0.0, 0.0
        D._islands, D._island_of = {}, {}
        D.regime[:] = 1
        D.all_dem = True
        D.t = Tk - float(D.env.t0_flight)
        a1 = D.loads.accel(np.array([Tk]), side="+")[0]
        Om1, Omd1 = D.loads.omega(Tk)[0], D.loads.omega_dot(Tk)[0]
        Ld = D.loads
        Ld.profile_accel = lambda T: np.repeat(a1[None, :], len(np.atleast_1d(T)), axis=0)
        Ld.omega = lambda T: np.repeat(Om1[None, :], len(np.atleast_1d(T)), axis=0)
        Ld.omega_dot = lambda T, side=None: np.repeat(Omd1[None, :], len(np.atleast_1d(T)), axis=0)
        Ld.breakpoints = np.zeros(0)
        Q = _cap_system(D, mem)
        F1 = D._island_load(mem, Tk)
        dF = F1 - S2._island_load(mem, Tk, "-")
        jg = float((np.linalg.norm(dF.reshape(-1, 6)[:, :3], axis=1) / D.m[mem]).max()) / 9.80665
        lu = splu(Q["K"])
        tt, tr = D._island_tols(Q["capv"], Q["prm"], Q["tm"])
        Cq = statics.solve_equilibrium(Q["x"], Q["ci"], Q["cj"], Q["prm"], Q["pe"], Q["h"], D.L, lu, F1,
                                       lu.solve(F1 + Q["Fnet"]), tt, tr, 40,
                                       tangent=D._island_tangent(dict(K=Q["K"], J=Q["J"], prm=Q["prm"], lu=lu), False))
        _cap_run(D, 5e-3)
        _cap_equilibrate(D, mem)
        hD = D._hist[Q["rows"]]
        kw = np.flatnonzero(Q["tm"] & (Q["cj"] == -2))
        dxr = np.linalg.norm(hD[kw, dem.H_XR:dem.H_XR + 3] - Cq["hn"][kw, dem.H_XR:dem.H_XR + 3], axis=1)
        slip = float(dxr.max())
        near = np.zeros((len(Q["prm"]), 3), bool)
        near[kw, 1] = True
        Kr = statics.released_stiffness(Q["K"], Q["J"], Q["kc"], near, D._fhat(Q["lam"]))
        E_j = 0.5 * float(dF @ splu((0.5 * (Kr + Kr.T)).tocsc()).solve(dF))
        rb = D._island_ratchet(Q["K"], Q["J"], Q["prm"], Q["h"], Q["lam"], Q["out"], Q["kc"], Q["capv"], E_j, mem)
        bound = float(rb[1][kw, 1].max()) if rb is not None else math.inf
    xi = min(float(c[1] / k[1]) for _, k, c in (_cap_state(P, q) for q in wall))
    slept = ne.get("island_static_steps", 0) + ne.get("island_creep_steps", 0)
    ok = (r0 >= 0.999 and bool(rec) and slept > 0 and ne.get("island_wake_fast", 0) >= 1 and d_pre <= 1e-15
          and d_post > 100 * max(d_pre, 1e-18) and same and ok_c and slip <= bound)
    sec = time.time() - t0
    return _res("island_across_breakpoint", "A cluster at its limits across a load jump (de-spin)",
                "Three 50 um silica grains in the edge of a glass wall and face, 5 cm off the spin axis under the "
                "nominal flight profile, pressed into the +x wall by the 4.8 Hz spin (4.6 g) and into the +z face by "
                "the coast's drag, every rolling spring on the wall set to (1 - 2^-40) of its limit (along the drag) "
                "and settled 10 ms in the full model, then set back to the limit at the exact equilibrium at rest "
                "(_cap_at_limit: the settling leaves a slowly decaying twisting oscillation of the pair contacts, 9e-8 "
                "rad at the knot, which the engine would refuse at entry (_try_sleep_island, E3, "
                "cluster_awake_ratchet: the code review's own run of this state without the reset) and which "
                "would otherwise dominate the comparison; smallest "
                f"ratio {r0:.9f}), from T+{Tk - 0.01:.2f} s across the "
                f"de-spin knot at T+{Tk:.1f} s, where the load on the grains jumps by {jg:.2f} g (the Euler load of the "
                "de-spin, flight.py). (a) The engine "
                "(asleep up to the knot, woken at it, the full model resolving the jump): the state committed by the "
                "step ending at the knot must be the equilibrium of the load just before it (a Newton solve from "
                "it moves nothing beyond 1e-15 m) and not of the load just after it (> 100x more); (b) the engine "
                "vs the full model across the knot at 50 and 100 steps per contact period: positions and sliding, "
                "rolling and twisting springs within 3x the full model's step error (the engine's reported bounds "
                "are listed, not added), the same contacts; (c) the "
                "jump's own transient: from that committed state, at rest, the loads held at their post-jump values "
                "for 5 ms in the full model; its permanent slip at the capped springs (against the kernel's "
                "return map solved to equilibrium at the post-jump load) must be within the released-system bound "
                "(_island_ratchet) of the jump's energy in the released system, dF^T K_rel^-1 dF / 2 (the critique's "
                f"dF^2 / (2 k_rel)). Runtime {sec:.0f} s.",
                f"asleep island-steps {slept}, woken at the knot {ne.get('island_wake_fast', 0)}; (a) committed at "
                f"the knot: Newton displacement to the pre-jump load {d_pre:.1e} m, to the post-jump load "
                f"{d_post:.1e} m; (b) {_cap_fmt(out_c)}; same contacts {same}; (c) jump energy {E_j:.1e} J, "
                f"permanent slip {slip:.2e} m vs bound {bound:.2e} m ({slip / xi:.1e} of the deflection at the "
                "limit)",
                "(a) <= 1e-15 m and > 100x that; (b) each <= 3x step error, same contacts; (c) slip <= bound", ok)


CHECKS += [check_cap_sleep_vs_dem, check_cap_twist_microgravity, check_creep_island_vs_dem,
           check_between_sample_bound, check_ratchet_released_bound, check_island_across_breakpoint]

"""Slaved model of a wall-anchored cluster of grains (the generalisation of roll1.py to several grains and walls).

For a set of grains resting on walls and on each other, every contact inside its pull-off and sliding limits: the
stiff modes (normal overlap and sliding at every contact) are held on their quasi-static values, and only the soft
modes are integrated: rolling and twisting at every contact, with the same sds springs, caps and dashpots as
dem.contact_force (rotate-rescale, increment, force, cap), at a fraction of the softest-mode period instead of the
stiff contact period. The stiff contact forces are Lagrange multipliers: with u = (v_1, w_1, ..., v_N, w_N), each
contact k gives three rows g_k = v_i - v_j + n_k x (r_i w_i + r_j w_j) = 0 (no normal approach, no slip; the slip
velocity of dem.contact_force), and the multipliers lam (the contact force on grain i) follow from

    G M^-1 W lam = -(G M^-1 f + Gdot u),        a = M^-1 (f + W lam),

f the body forces (uniform acceleration, centrifugal, Coriolis, Euler, drag, long-range) and the soft forces, W the
force map with the kernel's torque levers (r - delta/2 for grain pairs, as dem.contact_force). The caps of the soft
springs depend on lam (|F_ne + 2 F_c|): solved as a fixed point. Velocity Verlet in the order of dem.run_steps; after
each drift the normal overlaps are projected onto their quasi-static (JKR) values for the current contact forces, and
after each step the velocities onto G u = 0. Sliding springs are set to their quasi-static deflection -f_t / k_s, so
a cluster handed back to the full model starts on the slow manifold.

Only non-redundant clusters (the 3K contact rows independent, well conditioned): a grain on one wall, a stuck pair on
a wall, a grain on top of another, ... Redundant sets (a grain in an edge or corner, closed loops) stay in the full
model. A cluster is stopped (flag = step + 1, left at the start of that step) when any contact nears pull-off or its
sliding limit (within `margin`), the constraint rows lose conditioning, a contact would leave the elastic branch, or
two grains of the cluster that do not touch come within the capture distance. The error of slaving is of order
(soft / stiff frequency)^2, as for roll1 (verify_full.check_slaved_vs_full).

Provenance: numerical method written for DOLL-E (a multiplier / velocity-projection integrator, cf. RATTLE, Andersen
1983); the contact law is dem.contact_force's.
"""
from __future__ import annotations

import math

import numpy as np

from . import dem

try:
    from numba import njit
except ImportError:          # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

MAX_GRAINS = 8
MAX_CONTACTS = 16
COND_MIN = 0.05              # smallest / largest singular value of G M^-1/2 allowed (else: hand back)
ETA0_MAX = 2e-4             # slaving indicator: relative shift of the soft frequencies (single grain on a wall:
ETAF_MAX = 1e-3             # ~2e-5 for a measured path error of 1e-5), and weighted by the loads' frequencies


@njit(cache=True)
def jkr_overlap(p, h, Fne, a_guess):
    """Overlap and contact radius on the stable (or flattened) JKR branch for elastic normal force Fne."""
    w, E = p[dem.P_W], p[dem.P_ESTAR]
    if h[dem.H_YLD] > 0.5:
        R, off = h[dem.H_RP], h[dem.H_DP]
    else:
        R, off = p[dem.P_REFF], 0.0
    a_min = (9.0 * math.pi * w * R * R / (8.0 * E)) ** (1.0 / 3.0)     # dF/da = 0: end of the stable branch
    lo, hi = a_min, max(a_guess, a_min * 1.01)
    while dem.jkr_force(hi, R, w, E) < Fne:
        hi *= 1.5
    a = min(max(a_guess, lo), hi)
    for _ in range(60):                       # safeguarded Newton
        F = dem.jkr_force(a, R, w, E) - Fne
        if F > 0:
            hi = a
        else:
            lo = a
        dF = 4.0 * E * a * a / R - 1.5 * math.sqrt(8.0 * math.pi * w * E * a)
        an = a - F / dF if dF > 0 else 0.5 * (lo + hi)
        if not (lo < an < hi):
            an = 0.5 * (lo + hi)
        if abs(an - a) <= 1e-15 * a:
            a = an
            break
        a = an
    delta = a * a / R - math.sqrt(2.0 * math.pi * w * a / E) + off
    return delta, a


@njit(cache=True)
def _geom(x, rad, ci, cj, L, n, rho, dl):
    dxv = np.empty(3)
    for k in range(ci.shape[0]):
        i, j = ci[k], cj[k]
        if j >= 0:
            for d in range(3):
                dxv[d] = x[i, d] - x[j, d]
            r = math.sqrt(dxv[0] ** 2 + dxv[1] ** 2 + dxv[2] ** 2)
            dl[k] = rad[i] + rad[j] - r
        else:
            r = dem._wall_dx(x[i], L, -1 - j, dxv)
            dl[k] = rad[i] - r
        rho[k] = r
        for d in range(3):
            n[k, d] = dxv[d] / r


@njit(cache=True)
def _GW(rad, ci, cj, n, dl, G, W):
    """G (3K x 6N): contact velocity rows; W (6N x 3K): generalised force of a unit contact force (on grain i)."""
    G[:, :] = 0.0
    W[:, :] = 0.0
    for k in range(ci.shape[0]):
        i, j = ci[k], cj[k]
        n0, n1, n2 = n[k, 0], n[k, 1], n[k, 2]
        ri = rad[i]
        li = ri - 0.5 * dl[k] if j >= 0 else ri
        for e in range(3):
            G[3 * k + e, 6 * i + e] = 1.0
            W[6 * i + e, 3 * k + e] = 1.0
        # skew(n): S y = n x y
        S = ((0.0, -n2, n1), (n2, 0.0, -n0), (-n1, n0, 0.0))
        for a in range(3):
            for b in range(3):
                G[3 * k + a, 6 * i + 3 + b] = ri * S[a][b]
                W[6 * i + 3 + a, 3 * k + b] = -li * S[a][b]
        if j >= 0:
            rj = rad[j]
            lj = rj - 0.5 * dl[k]
            for e in range(3):
                G[3 * k + e, 6 * j + e] = -1.0
                W[6 * j + e, 3 * k + e] = -1.0
            for a in range(3):
                for b in range(3):
                    G[3 * k + a, 6 * j + 3 + b] = rj * S[a][b]
                    W[6 * j + 3 + a, 3 * k + b] = -lj * S[a][b]


@njit(cache=True)
def _gamma(v, w, rad, ci, cj, n, rho, g):
    """Gdot u = ndot x (r_i w_i + r_j w_j) for grain pairs (walls: 0)."""
    for k in range(ci.shape[0]):
        i, j = ci[k], cj[k]
        g[3 * k] = g[3 * k + 1] = g[3 * k + 2] = 0.0
        if j < 0:
            continue
        r0, r1, r2 = v[i, 0] - v[j, 0], v[i, 1] - v[j, 1], v[i, 2] - v[j, 2]
        vn = r0 * n[k, 0] + r1 * n[k, 1] + r2 * n[k, 2]
        d0 = (r0 - vn * n[k, 0]) / rho[k]
        d1 = (r1 - vn * n[k, 1]) / rho[k]
        d2 = (r2 - vn * n[k, 2]) / rho[k]
        O0 = rad[i] * w[i, 0] + rad[j] * w[j, 0]
        O1 = rad[i] * w[i, 1] + rad[j] * w[j, 1]
        O2 = rad[i] * w[i, 2] + rad[j] * w[j, 2]
        g[3 * k] = d1 * O2 - d2 * O1
        g[3 * k + 1] = d2 * O0 - d0 * O2
        g[3 * k + 2] = d0 * O1 - d1 * O0


@njit(cache=True)
def _body(x, v, w, m, I, off, acc, Om, Omd, gdrag, Fext, f):
    """Body forces and torques as dem.compute_forces (tubule frame)."""
    for i in range(x.shape[0]):
        rx, ry, rz = x[i, 0] + off[0], x[i, 1] + off[1], x[i, 2] + off[2]
        c0 = Om[1] * rz - Om[2] * ry
        c1 = Om[2] * rx - Om[0] * rz
        c2 = Om[0] * ry - Om[1] * rx
        cf0 = -(Om[1] * c2 - Om[2] * c1)
        cf1 = -(Om[2] * c0 - Om[0] * c2)
        cf2 = -(Om[0] * c1 - Om[1] * c0)
        co0 = -2.0 * (Om[1] * v[i, 2] - Om[2] * v[i, 1]) - (Omd[1] * rz - Omd[2] * ry)
        co1 = -2.0 * (Om[2] * v[i, 0] - Om[0] * v[i, 2]) - (Omd[2] * rx - Omd[0] * rz)
        co2 = -2.0 * (Om[0] * v[i, 1] - Om[1] * v[i, 0]) - (Omd[0] * ry - Omd[1] * rx)
        f[6 * i] = m[i] * (acc[0] + cf0 + co0 - gdrag[i] * v[i, 0]) + Fext[i, 0]
        f[6 * i + 1] = m[i] * (acc[1] + cf1 + co1 - gdrag[i] * v[i, 1]) + Fext[i, 1]
        f[6 * i + 2] = m[i] * (acc[2] + cf2 + co2 - gdrag[i] * v[i, 2]) + Fext[i, 2]
        f[6 * i + 3] = I[i] * (-(Om[1] * w[i, 2] - Om[2] * w[i, 1]) - Omd[0])
        f[6 * i + 4] = I[i] * (-(Om[2] * w[i, 0] - Om[0] * w[i, 2]) - Omd[1])
        f[6 * i + 5] = I[i] * (-(Om[0] * w[i, 1] - Om[1] * w[i, 0]) - Omd[2])


@njit(cache=True)
def _soft(w, n, prm, hist, ci, cj, dt, Fncrit, hout, f):
    """Rolling and twisting sds exactly as dem.contact_force (rotate-rescale, increment, force, cap), from the
    histories hist into hout; adds the torques to f."""
    for k in range(ci.shape[0]):
        i, j = ci[k], cj[k]
        p = prm[k]
        for q in range(hist.shape[1]):
            hout[k, q] = hist[k, q]
        h = hout[k]
        n0, n1, n2 = n[k, 0], n[k, 1], n[k, 2]
        rr0 = w[i, 0] - (w[j, 0] if j >= 0 else 0.0)
        rr1 = w[i, 1] - (w[j, 1] if j >= 0 else 0.0)
        rr2 = w[i, 2] - (w[j, 2] if j >= 0 else 0.0)
        if p[dem.P_KR] > 0.0:
            R = p[dem.P_REFF]
            vl0 = R * (rr1 * n2 - rr2 * n1)
            vl1 = R * (rr2 * n0 - rr0 * n2)
            vl2 = R * (rr0 * n1 - rr1 * n0)
            dem._rot_rescale(h, dem.H_XR, n0, n1, n2)
            h[dem.H_XR] += vl0 * dt
            h[dem.H_XR + 1] += vl1 * dt
            h[dem.H_XR + 2] += vl2 * dt
            kr, gr = p[dem.P_KR], p[dem.P_GR]
            fr0 = -kr * h[dem.H_XR] - gr * vl0
            fr1 = -kr * h[dem.H_XR + 1] - gr * vl1
            fr2 = -kr * h[dem.H_XR + 2] - gr * vl2
            Frc = p[dem.P_MUR] * Fncrit[k]
            mfr = math.sqrt(fr0 * fr0 + fr1 * fr1 + fr2 * fr2)
            if mfr > Frc:
                xm = math.sqrt(h[dem.H_XR] ** 2 + h[dem.H_XR + 1] ** 2 + h[dem.H_XR + 2] ** 2)
                if xm > 0.0:
                    s = Frc / mfr
                    h[dem.H_XR] = -(s * fr0 + gr * vl0) / kr
                    h[dem.H_XR + 1] = -(s * fr1 + gr * vl1) / kr
                    h[dem.H_XR + 2] = -(s * fr2 + gr * vl2) / kr
                    fr0 *= s
                    fr1 *= s
                    fr2 *= s
                else:
                    fr0 = fr1 = fr2 = 0.0
            t0 = R * (n1 * fr2 - n2 * fr1)
            t1 = R * (n2 * fr0 - n0 * fr2)
            t2 = R * (n0 * fr1 - n1 * fr0)
            f[6 * i + 3] += t0
            f[6 * i + 4] += t1
            f[6 * i + 5] += t2
            if j >= 0:
                f[6 * j + 3] -= t0
                f[6 * j + 4] -= t1
                f[6 * j + 5] -= t2
        if p[dem.P_KTW] > 0.0:
            mt = rr0 * n0 + rr1 * n1 + rr2 * n2
            h[dem.H_XTW] += mt * dt
            tq = -p[dem.P_KTW] * h[dem.H_XTW] - p[dem.P_GTW] * mt
            Mtc = p[dem.P_MUTW] * Fncrit[k]
            if abs(tq) > Mtc:
                sg = 1.0 if mt > 0 else (-1.0 if mt < 0 else 0.0)
                h[dem.H_XTW] = (Mtc * sg - p[dem.P_GTW] * mt) / p[dem.P_KTW]
                tq = -Mtc * sg
            f[6 * i + 3] += tq * n0
            f[6 * i + 4] += tq * n1
            f[6 * i + 5] += tq * n2
            if j >= 0:
                f[6 * j + 3] -= tq * n0
                f[6 * j + 4] -= tq * n1
                f[6 * j + 5] -= tq * n2


@njit(cache=True)
def _fncrit(lam, n, prm, hist, out):
    for k in range(n.shape[0]):
        Fne = lam[3 * k] * n[k, 0] + lam[3 * k + 1] * n[k, 1] + lam[3 * k + 2] * n[k, 2]
        Fp = hist[k, dem.H_FCP] if hist[k, dem.H_YLD] > 0.5 else prm[k, dem.P_FC]
        out[k] = abs(Fne + 2.0 * Fp)


@njit(cache=True)
def _accel(x, v, w, m, I, rad, ci, cj, prm, hist, L, off, dt, acc, Om, Omd, gdrag, Fext, Md, lam, hout, a_out,
           n, rho, dl, G, W, A):
    """Constrained accelerations at (x, v, w), soft springs advanced by dt from hist into hout; lam (in: guess,
    out: the multipliers). Returns the number of fixed-point iterations (-1 if it did not converge)."""
    _geom(x, rad, ci, cj, L, n, rho, dl)
    _GW(rad, ci, cj, n, dl, G, W)
    K3 = 3 * ci.shape[0]
    N6 = 6 * x.shape[0]
    g = np.empty(K3)
    _gamma(v, w, rad, ci, cj, n, rho, g)
    fb = np.empty(N6)
    _body(x, v, w, m, I, off, acc, Om, Omd, gdrag, Fext, fb)
    for a in range(K3):
        for b in range(K3):
            s = 0.0
            for c in range(N6):
                s += G[a, c] * W[c, b] / Md[c]
            A[a, b] = s
    Fnc = np.empty(ci.shape[0])
    f = np.empty(N6)
    b = np.empty(K3)
    it = 0
    for it in range(40):
        _fncrit(lam, n, prm, hist, Fnc)
        for c in range(N6):
            f[c] = fb[c]
        _soft(w, n, prm, hist, ci, cj, dt, Fnc, hout, f)
        for a in range(K3):
            s = 0.0
            for c in range(N6):
                s += G[a, c] * f[c] / Md[c]
            b[a] = -(s + g[a])
        lam_new = np.linalg.solve(A, b)
        dd = 0.0
        sc = 0.0
        for a in range(K3):
            dd = max(dd, abs(lam_new[a] - lam[a]))
            sc = max(sc, abs(lam_new[a]))
            lam[a] = lam_new[a]
        if dd <= 1e-13 * max(sc, 1e-30):
            break
    else:
        it = -1
    for c in range(N6):
        s = f[c]
        for a in range(K3):
            s += W[c, a] * lam[a]
        a_out[c] = s / Md[c]
    return it


@njit(cache=True)
def _project_overlaps(x, u, m, rad, ci, cj, prm, hist, L, lam, dt, n, rho, dl, ahint):
    """Move the centres so every normal overlap equals its quasi-static value for the current contact force
    (mass-weighted least change), and correct the velocities accordingly. Returns the largest residual."""
    K = ci.shape[0]
    N = x.shape[0]
    res = 0.0
    for _ in range(4):
        _geom(x, rad, ci, cj, L, n, rho, dl)
        tgt = np.empty(K)
        for k in range(K):
            Fne = lam[3 * k] * n[k, 0] + lam[3 * k + 1] * n[k, 1] + lam[3 * k + 2] * n[k, 2]
            tgt[k], ahint[k] = jkr_overlap(prm[k], hist[k], Fne, ahint[k])
        # d(dl_k)/dx_i = -n_k, d(dl_k)/dx_j = +n_k
        Jn = np.zeros((K, 3 * N))
        for k in range(K):
            i, j = ci[k], cj[k]
            for d in range(3):
                Jn[k, 3 * i + d] = -n[k, d]
                if j >= 0:
                    Jn[k, 3 * j + d] = n[k, d]
        Akk = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                s = 0.0
                for c in range(3 * N):
                    s += Jn[a, c] * Jn[b, c] / m[c // 3]
                Akk[a, b] = s
        r = tgt - dl
        res = 0.0
        for k in range(K):
            res = max(res, abs(r[k]))
        if res < 1e-15:
            break
        mu = np.linalg.solve(Akk, r)
        for c in range(3 * N):
            s = 0.0
            for k in range(K):
                s += Jn[k, c] * mu[k]
            dx = s / m[c // 3]
            x[c // 3, c % 3] += dx
            if dt > 0:
                u[6 * (c // 3) + c % 3] += dx / dt
    return res


@njit(cache=True)
def condition(x, m, I, rad, ci, cj, L):
    """Smallest / largest singular value of G M^-1/2 (0: redundant or degenerate rows)."""
    K, N = ci.shape[0], x.shape[0]
    n = np.empty((K, 3))
    rho = np.empty(K)
    dl = np.empty(K)
    G = np.empty((3 * K, 6 * N))
    W = np.empty((6 * N, 3 * K))
    _geom(x, rad, ci, cj, L, n, rho, dl)
    _GW(rad, ci, cj, n, dl, G, W)
    for c in range(6 * N):
        sc = 1.0 / math.sqrt(m[c // 6] if c % 6 < 3 else I[c // 6])
        for a in range(3 * K):
            G[a, c] *= sc
    if 3 * K > 6 * N:
        return 0.0
    s = np.linalg.svd(G)[1]
    return s.min() / s.max()


@njit(cache=True)
def run(x, v, w, m, I, rad, ci, cj, prm, hist, L, off, dt, nsteps, acc, Om, Omd, gdrag, Fext, margin, d_cap, lam,
        flag, bv, ba, boff, exc, xo, ro, cface, dmax, vmx, phi_max):
    """Advance the cluster by nsteps of dt. acc, Om, Omd: (nsteps + 1, 3) at the step boundaries. flag[0] = step + 1
    when it had to stop (state left at the start of that step), with flag[1] the reason (1 pull-off, 2 sliding,
    3 conditioning, 4 a non-touching pair of the cluster, a wall it does not touch (cface[i, f] False) or a static
    grain (centres xo, radii ro) within d_cap, 5 the fixed point did not converge, 6 plastic loading, 7 a contact
    normal turning faster than phi_max per step). exc: per grain, the most its distance from the start exceeds
    bv t + ba t^2/2 + boff. dmax, vmx: per grain, the largest distance from the start and the largest speed."""
    N = x.shape[0]
    K = ci.shape[0]
    N6, K3 = 6 * N, 3 * K
    Md = np.empty(N6)
    for i in range(N):
        for e in range(3):
            Md[6 * i + e] = m[i]
            Md[6 * i + 3 + e] = I[i]
    n = np.empty((K, 3))
    rho = np.empty(K)
    dl = np.empty(K)
    G = np.empty((K3, N6))
    W = np.empty((N6, K3))
    A = np.empty((K3, K3))
    a_c = np.empty(N6)
    hout = hist.copy()
    ahint = np.empty(K)
    for k in range(K):
        ahint[k] = hist[k, dem.H_A] if hist[k, dem.H_A] > 0 else prm[k, dem.P_A0]
    x00 = x.copy()
    xs = x.copy()
    vs = v.copy()
    ws = w.copy()
    hs = hist.copy()
    lams = lam.copy()
    u = np.empty(N6)
    # pairs of the cluster that do not touch
    touch = np.zeros((N, N), np.bool_)
    for k in range(K):
        if cj[k] >= 0:
            touch[ci[k], cj[k]] = True
            touch[cj[k], ci[k]] = True
    if _accel(x, v, w, m, I, rad, ci, cj, prm, hist, L, off, 0.0, acc[0], Om[0], Omd[0], gdrag, Fext, Md, lam, hout,
              a_c, n, rho, dl, G, W, A) < 0:
        flag[0], flag[1] = 1, 5
        return
    _geom(x, rad, ci, cj, L, n, rho, dl)
    nprev = n.copy()
    for s in range(nsteps):
        for k in range(K):
            for e in range(3):
                nprev[k, e] = n[k, e]
        xs[:, :] = x
        vs[:, :] = v
        ws[:, :] = w
        hs[:, :] = hist
        lams[:] = lam
        for i in range(N):
            for e in range(3):
                u[6 * i + e] = v[i, e] + 0.5 * dt * a_c[6 * i + e]
                u[6 * i + 3 + e] = w[i, e] + 0.5 * dt * a_c[6 * i + 3 + e]
                x[i, e] += dt * u[6 * i + e]
        _project_overlaps(x, u, m, rad, ci, cj, prm, hist, L, lam, dt, n, rho, dl, ahint)
        for i in range(N):
            for e in range(3):
                v[i, e] = u[6 * i + e]
                w[i, e] = u[6 * i + 3 + e]
        it = _accel(x, v, w, m, I, rad, ci, cj, prm, hist, L, off, dt, acc[s + 1], Om[s + 1], Omd[s + 1], gdrag,
                    Fext, Md, lam, hout, a_c, n, rho, dl, G, W, A)
        reason = 0
        if it < 0:
            reason = 5
        else:
            for c in range(N6):
                u[c] += 0.5 * dt * a_c[c]
            # velocity projection onto G u = 0 (with the same force map)
            Gu = np.empty(K3)
            for a in range(K3):
                sacc = 0.0
                for c in range(N6):
                    sacc += G[a, c] * u[c]
                Gu[a] = sacc
            y = np.linalg.solve(A, Gu)
            for c in range(N6):
                sacc = 0.0
                for a in range(K3):
                    sacc += W[c, a] * y[a]
                u[c] -= sacc / Md[c]
            for i in range(N):
                for e in range(3):
                    v[i, e] = u[6 * i + e]
                    w[i, e] = u[6 * i + 3 + e]
            # limits from the contact forces; sliding springs on their quasi-static deflection
            for k in range(K):
                p = prm[k]
                h = hout[k]
                Fne = lam[3 * k] * n[k, 0] + lam[3 * k + 1] * n[k, 1] + lam[3 * k + 2] * n[k, 2]
                ft0 = lam[3 * k] - Fne * n[k, 0]
                ft1 = lam[3 * k + 1] - Fne * n[k, 1]
                ft2 = lam[3 * k + 2] - Fne * n[k, 2]
                Fp = h[dem.H_FCP] if h[dem.H_YLD] > 0.5 else p[dem.P_FC]
                Fncrit = abs(Fne + 2.0 * Fp)
                if -Fne / Fp > 1.0 - margin:
                    reason = 1
                elif math.sqrt(ft0 * ft0 + ft1 * ft1 + ft2 * ft2) > (1.0 - margin) * p[dem.P_MUT] * Fncrit:
                    reason = 2
                if h[dem.H_YLD] > 0.5:
                    ks = p[dem.P_KT] * (4.5 * math.pi * p[dem.P_W] * h[dem.H_RP] ** 2 / p[dem.P_ESTAR]) ** (1.0 / 3.0)
                elif int(p[dem.P_TMODE]) == dem.TANG_MINDLIN:
                    ks = p[dem.P_KT] * h[dem.H_A]
                else:
                    ks = p[dem.P_KT] * p[dem.P_A0]
                h[dem.H_XT] = -ft0 / ks
                h[dem.H_XT + 1] = -ft1 / ks
                h[dem.H_XT + 2] = -ft2 / ks
                d_qs, a_qs = jkr_overlap(p, h, Fne, ahint[k])
                h[dem.H_A] = a_qs
                if d_qs > p[dem.P_DY] and d_qs >= h[dem.H_DMAX] * (1.0 - 1e-12):
                    reason = 6                         # plastic loading (dem.contact_force's rule): the full model
                if d_qs > h[dem.H_DMAX]:
                    h[dem.H_DMAX] = d_qs
                if a_qs > h[dem.H_AMAX]:
                    h[dem.H_AMAX] = a_qs
            # conditioning (every 25 steps) and gaps between grains of the cluster that do not touch
            if reason == 0 and s % 25 == 0:
                if condition(x, m, I, rad, ci, cj, L) < COND_MIN:
                    reason = 3
            if reason == 0:
                for i in range(N):
                    for j in range(i + 1, N):
                        if not touch[i, j]:
                            dd = math.sqrt((x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2 +
                                           (x[i, 2] - x[j, 2]) ** 2) - rad[i] - rad[j]
                            if dd < d_cap:
                                reason = 4
                    for f in range(6):
                        if not cface[i, f]:
                            ax = f // 2
                            gw = (x[i, ax] if f % 2 == 0 else L[ax] - x[i, ax]) - rad[i]
                            if gw < d_cap:
                                reason = 4
                    for q in range(xo.shape[0]):
                        dd = math.sqrt((x[i, 0] - xo[q, 0]) ** 2 + (x[i, 1] - xo[q, 1]) ** 2 +
                                       (x[i, 2] - xo[q, 2]) ** 2) - rad[i] - ro[q]
                        if dd < d_cap:
                            reason = 4
            if reason == 0:
                for k in range(K):                 # rotation of each contact normal during the step
                    c = n[k, 0] * nprev[k, 0] + n[k, 1] * nprev[k, 1] + n[k, 2] * nprev[k, 2]
                    if math.acos(min(max(c, -1.0), 1.0)) > phi_max:
                        reason = 7
        if reason != 0:
            x[:, :] = xs
            v[:, :] = vs
            w[:, :] = ws
            hist[:, :] = hs
            lam[:] = lams
            flag[0], flag[1] = s + 1, reason
            return
        for k in range(K):
            for q in range(hist.shape[1]):
                hist[k, q] = hout[k, q]
        tt = (s + 1) * dt
        for i in range(N):
            dd = math.sqrt((x[i, 0] - x00[i, 0]) ** 2 + (x[i, 1] - x00[i, 1]) ** 2 + (x[i, 2] - x00[i, 2]) ** 2)
            e = dd - (bv[i] * tt + 0.5 * ba[i] * tt * tt + boff[i])
            if e > exc[i]:
                exc[i] = e
            if dd > dmax[i]:
                dmax[i] = dd
            sp = math.sqrt(v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2)
            if sp > vmx[i]:
                vmx[i] = sp


def soft_period(m, I, rad, ci, cj, prm):
    """Shortest rocking/twisting period among the cluster's contacts (for the step: a fraction of it)."""
    T = math.inf
    for k in range(len(ci)):
        p = prm[k]
        i = ci[k]
        r = rad[i]
        Ic = I[i] + m[i] * r * r
        if p[dem.P_KR] > 0:
            T = min(T, 2 * math.pi * math.sqrt(min(Ic, I[i]) / (p[dem.P_KR] * p[dem.P_REFF] ** 2)))
        if p[dem.P_KTW] > 0:
            T = min(T, 2 * math.pi * math.sqrt(I[i] / p[dem.P_KTW]))
    return T


def enter(x, v, w, m, I, rad, ci, cj, prm, hist, L, off, acc, Om, Omd, gdrag, Fext):
    """Quasi-static contact forces at the current state and the force deviation of the stiff modes from them.
    Returns (lam, stiff force deviation per contact (normal, tangential), stiff-velocity norm |G u| per contact)."""
    K, N = len(ci), len(x)
    Md = np.empty(6 * N)
    for i in range(N):
        Md[6 * i:6 * i + 3] = m[i]
        Md[6 * i + 3:6 * i + 6] = I[i]
    n = np.empty((K, 3))
    rho = np.empty(K)
    dl = np.empty(K)
    G = np.empty((3 * K, 6 * N))
    W = np.empty((6 * N, 3 * K))
    A = np.empty((3 * K, 3 * K))
    a_c = np.empty(6 * N)
    hout = hist.copy()
    lam = np.zeros(3 * K)
    it = _accel(x, v, w, m, I, rad, ci, cj, prm, hist, L, off, 0.0, acc, Om, Omd, gdrag, Fext, Md, lam, hout, a_c,
                n, rho, dl, G, W, A)
    u = np.hstack([np.hstack([v[i], w[i]]) for i in range(N)])
    Gu = (G @ u).reshape(K, 3)
    return (lam if it >= 0 else None), n, dl, Gu


def settle(x, m, rad, ci, cj, prm, hist, L, lam):
    """Put the stiff state on the slow manifold for the contact forces lam: overlaps by projection (positions) and
    sliding springs at -f_t / k_s. Modifies x and hist."""
    K, N = len(ci), len(x)
    n = np.empty((K, 3))
    rho = np.empty(K)
    dl = np.empty(K)
    ahint = np.array([hist[k, dem.H_A] if hist[k, dem.H_A] > 0 else prm[k, dem.P_A0] for k in range(K)])
    u = np.zeros(6 * N)
    res = _project_overlaps(x, u, m, rad, ci, cj, prm, hist, L, lam, 0.0, n, rho, dl, ahint)
    for k in range(K):
        p = prm[k]
        lk = lam[3 * k:3 * k + 3]
        Fne = float(lk @ n[k])
        ft = lk - Fne * n[k]
        if hist[k, dem.H_YLD] > 0.5:
            ks = p[dem.P_KT] * (4.5 * math.pi * p[dem.P_W] * hist[k, dem.H_RP] ** 2 / p[dem.P_ESTAR]) ** (1 / 3)
        elif int(p[dem.P_TMODE]) == dem.TANG_MINDLIN:
            ks = p[dem.P_KT] * hist[k, dem.H_A]
        else:
            ks = p[dem.P_KT] * p[dem.P_A0]
        hist[k, dem.H_XT:dem.H_XT + 3] = -ft / ks
        hist[k, dem.H_A] = ahint[k]
    return res


def project_velocity(x, v, w, m, I, rad, ci, cj, L):
    """Remove the stiff (constraint-violating) part of the velocities: u -= M^-1 W A^-1 G u. Modifies v, w."""
    K, N = len(ci), len(x)
    n = np.empty((K, 3))
    rho = np.empty(K)
    dl = np.empty(K)
    G = np.empty((3 * K, 6 * N))
    W = np.empty((6 * N, 3 * K))
    _geom(x, rad, ci, cj, L, n, rho, dl)
    _GW(rad, ci, cj, n, dl, G, W)
    Md = np.repeat(np.column_stack([m, m, m, I, I, I]).reshape(-1), 1)
    A = G @ (W / Md[:, None])
    u = np.hstack([np.hstack([v[i], w[i]]) for i in range(N)])
    Gu = G @ u
    y = np.linalg.solve(A, Gu)
    u = u - (W @ y) / Md
    U = u.reshape(N, 6)
    v[:] = U[:, :3]
    w[:] = U[:, 3:]
    return Gu, A


def slaving_indicator(x, m, I, rad, ci, cj, prm, hist, L, f_load=0.0):
    """The slaving error indicator of a cluster (design panel's eta_0 / eta_F): the soft-mode frequencies of the full
    linearised cluster (stiffness from the kernel's own tangent with the rolling and twisting springs elastic, all
    masses) against those of the slaved one (the same stiffness on the admissible motions G u = 0):
    eta_0 = max_q |w_full,q^2 - w_slaved,q^2| / w_slaved,q^2, and eta_F the same weighted by
    max(1, (2 pi f_load / w_q)^2) for loads with content up to f_load. Returns (eta_0, eta_F, w_soft_max); eta = inf
    when the soft and stiff bands do not separate."""
    from scipy.linalg import eigh, null_space
    from . import statics
    K_ = len(ci)
    N = len(x)
    h = hist.copy()
    h[:, dem.H_XR:dem.H_XR + 3] = 0.0
    h[:, dem.H_XTW] = 0.0
    Kmat, _ = statics.jacobians(np.ascontiguousarray(x), rad, ci, cj, prm, h, L)
    Kd = Kmat.toarray()
    Kd = 0.5 * (Kd + Kd.T)
    Md = np.repeat(np.column_stack([m, m, m, I, I, I]).reshape(-1), 1)
    n = np.empty((K_, 3))
    rho = np.empty(K_)
    dl = np.empty(K_)
    G = np.empty((3 * K_, 6 * N))
    W = np.empty((6 * N, 3 * K_))
    _geom(np.ascontiguousarray(x), rad, ci, cj, L, n, rho, dl)
    _GW(rad, ci, cj, n, dl, G, W)
    Nb = null_space(G)
    d = Nb.shape[1]
    if d == 0:
        return 0.0, 0.0, 0.0
    try:
        wf = eigh(Kd, np.diag(Md), eigvals_only=True)
        ws = eigh(Nb.T @ Kd @ Nb, Nb.T @ np.diag(Md) @ Nb, eigvals_only=True)
    except Exception:
        return np.inf, np.inf, np.inf
    wf = np.sort(wf)[:d]
    ws = np.sort(ws)
    if (ws <= 0).any() or (wf <= 0).any():
        return np.inf, np.inf, np.inf
    eq = np.abs(wf - ws) / ws
    eta0 = float(eq.max())
    etaF = float((eq * np.maximum(1.0, (2 * math.pi * f_load) ** 2 / ws)).max())
    return eta0, etaF, float(math.sqrt(ws.max()))


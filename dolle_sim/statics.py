"""Linear statics of a resting cluster of grains, from the contact kernel itself (dem.contact_force).

For grains held by walls (a pile, an aggregate stuck to a wall, a grain in a corner) at static equilibrium, a small
change of the loads dF (forces and torques on every grain) moves them by du = K^-1 dF, and every contact's elastic
normal force, sliding force, rolling pseudo-force and twisting torque changes by J du. K = -dF_contact/du and J are
taken by central differences of the production kernel (so they are its exact tangent, including the Mindlin, rolling
and twisting springs, which in this formulation depend on the displacement increments): each displacement is applied
both as a position change and as velocity x time step, with a very long time step so that the dashpots add nothing.

This is what lets a resting cluster sleep exactly: while every contact stays inside its elastic limits (pull-off,
sliding, rolling, twisting), the plastic state cannot change, so the cluster is static; the test is evaluated for the
current loads every outer step.
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

EPS_X = 1e-13         # [m] translation step of the finite differences (relative to ~nm contact scales: 1e-4)
TAU = 1e3             # [s] fictitious step: dashpot forces ~ c eps / TAU are negligible
EPS_V = 1e-6          # [m/s] velocity step of the damping finite differences (dashpots are linear in velocity)
DT_V = 1e-24          # [s] fictitious step for them: the spring increments v dt are negligible


@njit(cache=True)
def _eval(x, ci, cj, prm, hist, L, dU, tau, out_all, ks):
    """Evaluate the contacts ks (indices) for displacement dU (n x 6: dx, dtheta). Histories are copied."""
    zero = np.zeros(3)
    dxv = np.empty(3)
    vi = np.empty(3)
    vj = np.empty(3)
    wi = np.empty(3)
    wj = np.empty(3)
    xi = np.empty(3)
    for q in range(ks.shape[0]):
        k = ks[q]
        i = ci[k]
        j = cj[k]
        h = hist[k].copy()
        for d in range(3):
            xi[d] = x[i, d] + dU[i, d]
            vi[d] = dU[i, d] / tau
            wi[d] = dU[i, 3 + d] / tau
        if j >= 0:
            for d in range(3):
                dxv[d] = xi[d] - (x[j, d] + dU[j, d])
                vj[d] = dU[j, d] / tau
                wj[d] = dU[j, 3 + d] / tau
            r = math.sqrt(dxv[0] ** 2 + dxv[1] ** 2 + dxv[2] ** 2)
            dem.contact_force(dxv, r, vi, vj, wi, wj, prm[k], h, tau, out_all[k], 0.0)
        else:
            r = dem._wall_dx(xi, L, -1 - j, dxv)
            dem.contact_force(dxv, r, vi, zero, wi, zero, prm[k], h, tau, out_all[k], 0.0)


@njit(cache=True)
def _eval_v(x, ci, cj, prm, hist, L, dV, dt, out_all, ks):
    """Evaluate the contacts ks at the current positions with velocities dV (n x 6: v, omega). Histories copied."""
    zero = np.zeros(3)
    dxv = np.empty(3)
    vi = np.empty(3)
    vj = np.empty(3)
    wi = np.empty(3)
    wj = np.empty(3)
    for q in range(ks.shape[0]):
        k = ks[q]
        i = ci[k]
        j = cj[k]
        h = hist[k].copy()
        for d in range(3):
            vi[d] = dV[i, d]
            wi[d] = dV[i, 3 + d]
        if j >= 0:
            for d in range(3):
                dxv[d] = x[i, d] - x[j, d]
                vj[d] = dV[j, d]
                wj[d] = dV[j, 3 + d]
            r = math.sqrt(dxv[0] ** 2 + dxv[1] ** 2 + dxv[2] ** 2)
            dem.contact_force(dxv, r, vi, vj, wi, wj, prm[k], h, dt, out_all[k], 0.0)
        else:
            r = dem._wall_dx(x[i], L, -1 - j, dxv)
            dem.contact_force(dxv, r, vi, zero, wi, zero, prm[k], h, dt, out_all[k], 0.0)


@njit(cache=True)
def _jac_coo(x, rad, ci, cj, prm, hist, L, vel=False):
    """K = -dF/du (6n x 6n) and J (8 components per contact x 6n) by central differences, as COO triplets. With
    vel: the same with respect to the velocities (C = -dF/dv, the dashpots, and J_v)."""
    n = x.shape[0]
    nc = ci.shape[0]
    outp = np.zeros((nc, dem.NOUT))
    outm = np.zeros((nc, dem.NOUT))
    dU = np.zeros((n, 6))
    cnt = np.zeros(n, np.int64)
    for k in range(nc):
        cnt[ci[k]] += 1
        if cj[k] >= 0:
            cnt[cj[k]] += 1
    start = np.zeros(n + 1, np.int64)
    for g in range(n):
        start[g + 1] = start[g] + cnt[g]
    lst = np.zeros(start[n], np.int64)
    fill = start[:n].copy()
    for k in range(nc):
        lst[fill[ci[k]]] = k
        fill[ci[k]] += 1
        if cj[k] >= 0:
            lst[fill[cj[k]]] = k
            fill[cj[k]] += 1
    nk = 6 * 12 * start[n] + 1
    nj = 6 * 8 * start[n] + 1
    Kr = np.zeros(nk, np.int64)
    Kc = np.zeros(nk, np.int64)
    Kv = np.zeros(nk)
    Jr = np.zeros(nj, np.int64)
    Jc = np.zeros(nj, np.int64)
    Jv = np.zeros(nj)
    pk = 0
    pj = 0
    for g in range(n):
        ks = lst[start[g]:start[g + 1]]
        if ks.shape[0] == 0:
            continue
        for d in range(6):
            if vel:
                eps = EPS_V if d < 3 else EPS_V / rad[g]
                dU[g, d] = eps
                _eval_v(x, ci, cj, prm, hist, L, dU, DT_V, outp, ks)
                dU[g, d] = -eps
                _eval_v(x, ci, cj, prm, hist, L, dU, DT_V, outm, ks)
            else:
                eps = EPS_X if d < 3 else EPS_X / rad[g]
                dU[g, d] = eps
                _eval(x, ci, cj, prm, hist, L, dU, TAU, outp, ks)
                dU[g, d] = -eps
                _eval(x, ci, cj, prm, hist, L, dU, TAU, outm, ks)
            dU[g, d] = 0.0
            col = 6 * g + d
            for q in range(ks.shape[0]):
                k = ks[q]
                i = ci[k]
                j = cj[k]
                for e in range(3):
                    dFi = (outp[k, e] - outm[k, e]) / (2 * eps)
                    dTi = (outp[k, 3 + e] - outm[k, 3 + e]) / (2 * eps)
                    Kr[pk], Kc[pk], Kv[pk] = 6 * i + e, col, -dFi
                    pk += 1
                    Kr[pk], Kc[pk], Kv[pk] = 6 * i + 3 + e, col, -dTi
                    pk += 1
                    if j >= 0:
                        dTj = (outp[k, 6 + e] - outm[k, 6 + e]) / (2 * eps)
                        Kr[pk], Kc[pk], Kv[pk] = 6 * j + e, col, dFi
                        pk += 1
                        Kr[pk], Kc[pk], Kv[pk] = 6 * j + 3 + e, col, -dTj
                        pk += 1
                vals = np.empty(8)
                vals[0] = (outp[k, 11] - outm[k, 11]) / (2 * eps)
                for e in range(3):
                    vals[1 + e] = (outp[k, 12 + e] - outm[k, 12 + e]) / (2 * eps)
                    vals[4 + e] = (outp[k, 15 + e] - outm[k, 15 + e]) / (2 * eps)
                vals[7] = (outp[k, 18] - outm[k, 18]) / (2 * eps)
                for e in range(8):
                    Jr[pj], Jc[pj], Jv[pj] = 8 * k + e, col, vals[e]
                    pj += 1
    return Kr[:pk], Kc[:pk], Kv[:pk], Jr[:pj], Jc[:pj], Jv[:pj]


def jacobians(x, rad, ci, cj, prm, hist, L, vel=False):
    """Sparse K (6n x 6n) and J (8 n_contacts x 6n); with vel, the damping matrix C = -dF/dv and J_v instead."""
    from scipy.sparse import coo_matrix
    n, nc = len(x), len(ci)
    Kr, Kc, Kv, Jr, Jc, Jv = _jac_coo(np.ascontiguousarray(x), rad, ci, cj, prm, hist, L, vel)
    K = coo_matrix((Kv, (Kr, Kc)), shape=(6 * n, 6 * n)).tocsc()
    J = coo_matrix((Jv, (Jr, Jc)), shape=(8 * nc, 6 * n)).tocsr()
    return K, J


@njit(cache=True)
def state(x, ci, cj, prm, hist, L):
    """Contact components (8 per contact), limits and the net contact force/torque on each grain at rest."""
    n = x.shape[0]
    nc = ci.shape[0]
    out = np.zeros((nc, dem.NOUT))
    dU = np.zeros((n, 6))
    ks = np.arange(nc)
    _eval(x, ci, cj, prm, hist, L, dU, TAU, out, ks)
    lam = np.zeros(8 * nc)
    Fnet = np.zeros(6 * n)
    for k in range(nc):
        lam[8 * k] = out[k, 11]
        for e in range(3):
            lam[8 * k + 1 + e] = out[k, 12 + e]
            lam[8 * k + 4 + e] = out[k, 15 + e]
        lam[8 * k + 7] = out[k, 18]
        i = ci[k]
        j = cj[k]
        for e in range(3):
            Fnet[6 * i + e] += out[k, e]
            Fnet[6 * i + 3 + e] += out[k, 3 + e]
            if j >= 0:
                Fnet[6 * j + e] -= out[k, e]
                Fnet[6 * j + 3 + e] += out[k, 6 + e]
    return lam, Fnet, out


def apply_displacement(x, ci, cj, prm, hist, L, dU):
    """Move the grains by dU (n x 6) quasi-statically: positions change and every spring history is advanced as the
    kernel would advance it for that relative motion. Returns new positions and histories."""
    out = np.zeros((len(ci), dem.NOUT))
    h = hist.copy()
    xn = x + dU[:, :3]
    _advance(x, ci, cj, prm, h, L, np.ascontiguousarray(dU), TAU, out)
    return xn, h


@njit(cache=True)
def _advance(x, ci, cj, prm, hist, L, dU, tau, out_all):
    zero = np.zeros(3)
    dxv = np.empty(3)
    vi = np.empty(3)
    vj = np.empty(3)
    wi = np.empty(3)
    wj = np.empty(3)
    xi = np.empty(3)
    for k in range(ci.shape[0]):
        i = ci[k]
        j = cj[k]
        for d in range(3):
            xi[d] = x[i, d] + dU[i, d]
            vi[d] = dU[i, d] / tau
            wi[d] = dU[i, 3 + d] / tau
        if j >= 0:
            for d in range(3):
                dxv[d] = xi[d] - (x[j, d] + dU[j, d])
                vj[d] = dU[j, d] / tau
                wj[d] = dU[j, 3 + d] / tau
            r = math.sqrt(dxv[0] ** 2 + dxv[1] ** 2 + dxv[2] ** 2)
            dem.contact_force(dxv, r, vi, vj, wi, wj, prm[k], hist[k], tau, out_all[k], 0.0)
        else:
            r = dem._wall_dx(xi, L, -1 - j, dxv)
            dem.contact_force(dxv, r, vi, zero, wi, zero, prm[k], hist[k], tau, out_all[k], 0.0)


def margins(lam, prm, out):
    """Largest ratio of each contact's load to its limit (pull-off, sliding, rolling, twisting)."""
    nc = len(prm)
    L8 = lam.reshape(nc, 8)
    Fne = L8[:, 0]
    Fpull = out[:, 19]
    Fncrit = np.abs(Fne + 2 * Fpull)
    fs = np.linalg.norm(L8[:, 1:4], axis=1)
    fr = np.linalg.norm(L8[:, 4:7], axis=1)
    tq = np.abs(L8[:, 7])
    r_pull = np.where(Fne < 0, -Fne / Fpull, 0.0)
    r_slide = fs / np.maximum(prm[:, dem.P_MUT] * Fncrit, 1e-300)
    r_roll = fr / np.maximum(prm[:, dem.P_MUR] * Fncrit, 1e-300)
    r_twist = tq / np.maximum(prm[:, dem.P_MUTW] * Fncrit, 1e-300)
    return np.maximum.reduce([r_pull, r_slide, r_roll, r_twist])


def margins_bounded(lam, dlam, prm, out):
    """margins() in the worst case when each contact quantity may deviate from lam by up to dlam (componentwise):
    normal load lowered by its bound, sliding and rolling force norms raised by the norm of theirs, twist raised."""
    nc = len(prm)
    L8 = lam.reshape(nc, 8)
    D8 = dlam.reshape(nc, 8)
    Fne = L8[:, 0] - D8[:, 0]
    Fpull = out[:, 19]
    Fncrit = np.maximum(Fne + 2 * Fpull, 0.0)
    fs = np.linalg.norm(L8[:, 1:4], axis=1) + np.linalg.norm(D8[:, 1:4], axis=1)
    fr = np.linalg.norm(L8[:, 4:7], axis=1) + np.linalg.norm(D8[:, 4:7], axis=1)
    tq = np.abs(L8[:, 7]) + D8[:, 7]
    r_pull = np.where(Fne < 0, -Fne / Fpull, 0.0)
    r_slide = fs / np.maximum(prm[:, dem.P_MUT] * Fncrit, 1e-300)
    r_roll = fr / np.maximum(prm[:, dem.P_MUR] * Fncrit, 1e-300)
    r_twist = tq / np.maximum(prm[:, dem.P_MUTW] * Fncrit, 1e-300)
    return np.maximum.reduce([r_pull, r_slide, r_roll, r_twist])


def energy_bounds(K, J, E, block=256):
    """The most each contact quantity lam_i = (J u)_i can deviate while the residual motion stays inside its energy
    ellipsoid u.Ks.u / 2 <= E (Ks the symmetric part of K; kinetic plus elastic energy, conserved or damped in the
    linear response): sqrt(2 E (J Ks^-1 J^T)_ii), exact for the ellipsoid. None if Ks is not positive definite."""
    from scipy.sparse.linalg import splu
    Ks = (0.5 * (K + K.T)).tocsc()
    try:
        lu = splu(Ks)
    except RuntimeError:
        return None
    Jt = J.T.tocsc()
    Jr = J.tocsr()
    d = np.empty(J.shape[0])
    for a in range(0, J.shape[0], block):
        b = min(a + block, J.shape[0])
        X = lu.solve(Jt[:, a:b].toarray())
        d[a:b] = np.einsum("ij,ji->i", Jr[a:b].toarray(), X)
    if (d < -1e-12 * np.abs(d).max()).any():
        return None
    return np.sqrt(2 * E * np.maximum(d, 0.0))



def residual_modes(K, C, Mdiag, G, u0, v0):
    """Exact linear response of a resting cluster to residual motion about its equilibrium:
    M u'' + (C + G) u' + K u = 0 (K = -dF/du and C = -dF/dv of the contacts from the kernel, G the rotating-frame and
    drag terms, M the masses and moments of inertia). Returns the eigenvalues lam and eigenvectors V of the first-order
    system, the modal amplitudes c of y0 = (u0, v0), and the reconstruction error |V c - y0| / |y0|."""
    from scipy.linalg import eig, solve
    n = len(Mdiag)
    Kd = K.toarray() if hasattr(K, "toarray") else np.asarray(K)
    Cd = (C.toarray() if hasattr(C, "toarray") else np.asarray(C)) + G
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Kd / Mdiag[:, None]
    A[n:, n:] = -Cd / Mdiag[:, None]
    lam, V = eig(A)
    y0 = np.concatenate([u0, v0])
    c = solve(V, y0.astype(complex))
    err = float(np.abs(V @ c - y0).max() / max(float(np.abs(y0).max()), 1e-300))
    return lam, V, c, err


def residual_state(lam, V, c, t):
    """Residual displacement u and velocity v at time t after the reference (real part of the modal sum)."""
    y = (V @ (c * np.exp(lam * t))).real
    n = len(y) // 2
    return y[:n], y[n:]


def residual_bound(V, c, J, Jv):
    """For every t >= 0 with every mode decaying: the most each contact quantity (J u + J_v v) and each coordinate of
    u can deviate, as the sum of the moduli of the modal contributions."""
    n = V.shape[0] // 2
    W = V * c[None, :]
    dl = np.abs(J @ W[:n] + Jv @ W[n:]).sum(1)
    ub = np.abs(W[:n]).sum(1)
    return dl, ub

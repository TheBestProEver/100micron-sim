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

Contacts at a cap (a sliding, rolling or twisting spring at its Coulomb-type limit mu |F_ne + 2 F_pull|): the friction
coefficients enter contact_force only through those three limits, so with them lifted (lift_caps: mu = 1e300) the
kernel is the elastic one and bit-identical to the real kernel wherever the real one takes no clip branch. The real
kernel's clip is a one-shot return map (xi = -(s f + g v)/k for a vector spring; the twist takes the sign of its
RATE, so at exactly zero rate an over-cap twist is reset to 0), and apply_displacement is exactly one application of
it to a finite displacement; its tangent at the cap is the elastic stiffness minus a rank-1 outward stiffness per
capped component (released_stiffness), which is both the consistent tangent of the return-mapped Newton iteration
(solve_equilibrium) and the stiffness in which a residual motion's permanent slip is bounded. Central differences
with the real parameters straddle the kink of a capped spring (they give the average of the two slopes), so every
tangent here is taken with the caps lifted; and a state evaluated by the real kernel at rest must sit strictly inside
its caps (snap_to_caps).
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


def apply_displacement(x, ci, cj, prm, hist, L, dU, with_out=False):
    """Move the grains by dU (n x 6) quasi-statically: positions change and every spring history is advanced as the
    kernel would advance it for that relative motion (one step of its return map: a spring over its cap is clipped
    once, by the excess). Returns new positions and histories, and with with_out the kernel output (n_contacts x
    NOUT) of that evaluation as well."""
    out = np.zeros((len(ci), dem.NOUT))
    h = hist.copy()
    xn = x + dU[:, :3]
    _advance(x, ci, cj, prm, h, L, np.ascontiguousarray(dU), TAU, out)
    return (xn, h, out) if with_out else (xn, h)


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


# ---------------------------------------------------------------------------- contacts at a cap
def lift_caps(prm):
    """Copy of the parameter array with the sliding, rolling and twisting limits lifted (mu = 1e300). The kernel with
    it is the elastic one, bit-identical to the real kernel whenever the real one takes no clip branch, since mu
    enters contact_force only through the three caps."""
    p = prm.copy()
    p[:, dem.P_MUT] = 1e300
    p[:, dem.P_MUR] = 1e300
    p[:, dem.P_MUTW] = 1e300
    return p


def overlaps(x, ci, cj, prm, L):
    """Overlap of every listed contact: rad_i + rad_j - |x_i - x_j| for a pair, rad_i minus the gap to the face
    -1 - cj for a wall (as dem._wall_dx measures it); negative for an open gap. The kernel yields a contact at an
    overlap above P_DY."""
    delta = np.empty(len(ci))
    pair = cj >= 0
    if pair.any():
        d = x[ci[pair]] - x[cj[pair]]
        delta[pair] = prm[pair, dem.P_RADI] + prm[pair, dem.P_RADJ] - np.linalg.norm(d, axis=1)
    wall = ~pair
    if wall.any():
        face = -1 - cj[wall]
        ax = face // 2
        xc = x[ci[wall], ax]
        delta[wall] = prm[wall, dem.P_RADI] - np.where(face % 2 == 0, np.abs(xc), np.abs(xc - L[ax]))
    return delta


def caps(lam, prm, out):
    """The limits of every contact, (n_contacts x 3) = [mu_t, mu_r, mu_tw] |F_ne + 2 F_pull| (sliding [N], rolling
    [N], twisting [N m]), with the elastic normal force F_ne = lam[8k] and the pull-off force out[:, 19]: the
    kernel's Fncrit times its three coefficients."""
    nc = len(prm)
    Fncrit = np.abs(lam.reshape(nc, 8)[:, 0] + 2 * out[:, 19])
    return prm[:, [dem.P_MUT, dem.P_MUR, dem.P_MUTW]] * Fncrit[:, None]


def kcomp(prm, hist):
    """Spring constants (n_contacts x 3) of the sliding, rolling and twisting springs exactly as contact_force forms
    them: k_s = P_KT a (Mindlin), P_KT times the a0 of the flattened contact (yielded) or P_KT a0; k_r = P_KR;
    k_tw = P_KTW."""
    kt = prm[:, dem.P_KT]
    a0p = (4.5 * math.pi * prm[:, dem.P_W] * hist[:, dem.H_RP] ** 2 / prm[:, dem.P_ESTAR]) ** (1.0 / 3.0)
    ks = np.where(prm[:, dem.P_TMODE].astype(np.int64) == dem.TANG_MINDLIN, kt * hist[:, dem.H_A],
                  np.where(hist[:, dem.H_YLD] > 0.5, kt * a0p, kt * prm[:, dem.P_A0]))
    return np.column_stack([ks, prm[:, dem.P_KR], prm[:, dem.P_KTW]])


def ratios_by_kind(lam, prm, out):
    """The four arrays margins() reduces, each (n_contacts,): pull-off (-F_ne / F_pull where the normal force is
    tensile), sliding, rolling and twisting (force or torque over its limit)."""
    nc = len(prm)
    L8 = lam.reshape(nc, 8)
    Fne = L8[:, 0]
    Fpull = out[:, 19]
    Fncrit = np.abs(Fne + 2 * Fpull)
    r_pull = np.where(Fne < 0, -Fne / Fpull, 0.0)
    r_slide = np.linalg.norm(L8[:, 1:4], axis=1) / np.maximum(prm[:, dem.P_MUT] * Fncrit, 1e-300)
    r_roll = np.linalg.norm(L8[:, 4:7], axis=1) / np.maximum(prm[:, dem.P_MUR] * Fncrit, 1e-300)
    r_twist = np.abs(L8[:, 7]) / np.maximum(prm[:, dem.P_MUTW] * Fncrit, 1e-300)
    return r_pull, r_slide, r_roll, r_twist


def ratios_by_kind_bounded(lam, dlam, prm, out):
    """The four arrays margins_bounded() reduces: the worst case when each contact quantity may deviate from lam by
    up to dlam (normal load lowered by its bound, sliding and rolling norms and the twist raised by theirs)."""
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
    return r_pull, r_slide, r_roll, r_twist


def snap_to_caps(hist, lam, prm, out, rel=2 ** -40):
    """Copy of the histories in which every sliding or rolling vector and every twist whose spring force at rest,
    -k h (k from kcomp, no dashpot term), exceeds (1 - rel) of its cap is scaled, direction and sign kept, so that
    the force is exactly (1 - rel) times the cap; returned with the largest relative force change made (about
    9.1e-13 for the default rel on a component that sat exactly at its cap, 0 if nothing was snapped). The caps are
    those of lam (normal force) and out (pull-off force); the ratio itself is formed from the histories, not from
    lam, because the real kernel evaluated at rest reads a clipped sliding or rolling spring as exactly its cap and an
    over-cap twist as 0 (contact_force: sg = sign(mt) = 0 at zero twist rate resets it). A state handed to the DEM,
    whose first force evaluation is at v = w = 0, or evaluated by state() with the real parameters, must therefore
    never sit over a cap, nor exactly on it (the clip tests are strict and round-off would decide)."""
    h = hist.copy()
    cp = caps(lam, prm, out)
    k = kcomp(prm, h)
    f = np.column_stack([k[:, 0] * np.linalg.norm(h[:, dem.H_XT:dem.H_XT + 3], axis=1),
                         k[:, 1] * np.linalg.norm(h[:, dem.H_XR:dem.H_XR + 3], axis=1),
                         k[:, 2] * np.abs(h[:, dem.H_XTW])])
    ratio = f / np.maximum(cp, 1e-300)
    over = ratio > 1 - rel
    s = np.where(over, (1 - rel) / np.maximum(ratio, 1e-300), 1.0)
    h[:, dem.H_XT:dem.H_XT + 3] *= s[:, 0:1]
    h[:, dem.H_XR:dem.H_XR + 3] *= s[:, 1:2]
    h[:, dem.H_XTW] *= s[:, 2]
    return h, (float(np.abs(1.0 - s).max()) if over.any() else 0.0)


def row_norms(lu, J, block=256):
    """||K^-T J_k^T||_2 for every component row k of J, lu the factorisation of K (block transposed solves, as
    energy_bounds). Then |(J K^-1 f)_k| <= rn_k ||f||_2 for every force vector f (Cauchy-Schwarz): the most a force
    of a given size can change each contact quantity through the linear response, without a dense inverse."""
    Jt = J.T.tocsc()
    rn = np.empty(J.shape[0])
    for a in range(0, J.shape[0], block):
        b = min(a + block, J.shape[0])
        X = lu.solve(Jt[:, a:b].toarray(), trans='T')
        rn[a:b] = np.linalg.norm(X, axis=0)
    return rn


def released_stiffness(K_el, J_el, kc, mask, fhat):
    """K_el minus the outward stiffness of every sliding, rolling or twisting component flagged in mask (n_contacts
    x 3, the kinds in that order): a component at its cap holds its force at the cap along its direction fhat
    (n_contacts x 3 x 3, the unit force direction per kind; the twist is a scalar, only its sign could matter and it
    drops out of the product), so the stiffness along that direction, the rank-1 matrix (1/k) (fhat^T J_kc)^T
    (fhat^T J_kc) with J_kc the component's rows of J_el (sliding 8k+1..8k+3, rolling 8k+4..8k+6, twist 8k+7) and
    k its spring constant from kcomp(), is removed; what a vector spring keeps is its stiffness against a rotation
    of its force. Sanity: one spring with kinematic row b has K = k b b^T and J_row = -k b^T, so (1/k) J^T J = K and
    a fully released spring contributes nothing. The result is the consistent elasto-plastic tangent of the
    return-mapped Newton iteration (solve_equilibrium with the real parameters) and the released system in which a
    residual motion's permanent slip is bounded. A cluster held against some motion only by released components has
    it singular (a mechanism) to the rounding of the finite differences in K_el and J_el, about 1e-11 of the spring
    stiffness: a factorisation may then fail or, as often, succeed on a round-off pivot and return a bound or a
    Newton step that is absurdly large; either is the mechanism's signature. Sparse csc of K_el's shape."""
    from scipy.sparse import coo_matrix
    nc = len(kc)
    rows, cols, vals = [], [], []
    q = 0
    for k, c in zip(*np.nonzero(mask)):
        if kc[k, c] <= 0.0:
            continue
        w = 1.0 / math.sqrt(kc[k, c])
        if c == 2:
            rows.append(q)
            cols.append(8 * k + 7)
            vals.append(w)
        else:
            for e in range(3):
                rows.append(q)
                cols.append(8 * k + 1 + 3 * c + e)
                vals.append(w * fhat[k, c, e])
        q += 1
    if q == 0:
        return K_el.tocsc()
    S = coo_matrix((vals, (rows, cols)), shape=(q, 8 * nc)).tocsr()
    V = S @ J_el
    return (K_el - V.T @ V).tocsc()


def solve_equilibrium(x, ci, cj, prm_apply, prm_el, hist, L, lu, Fext, du0, tol_trans, tol_rot, n_max,
                      tangent=None):
    """Static equilibrium of a cluster under the loads Fext (6n: force and torque on every grain) by Newton iteration
    on its displacement du (6n) from the reference state (x, hist). Every iterate is ONE application of the kernel to
    the reference state, apply_displacement(x, ..., prm_apply, hist, ..., du), never a step from the previous
    iterate (the kernel's clip is a one-shot return map), and the contact quantities of the state reached are read
    with the caps lifted, state(..., prm_el, ...): the springs as they are, no re-clip and no twist reset. Converged
    when the residual r = Fext + Fnet is within tol_trans on the translational and tol_rot on the rotational degrees
    of freedom (forces and torques have their own tolerances); the update is du += T^-1 r with T the factorised
    stiffness lu or, when given, tangent(xn, hn, lam, out) (an object with .solve: the released stiffness of the
    state reached, raising RuntimeError when singular). With prm_apply = prm_el this is Newton on the smooth elastic
    problem (1-3 iterations from a linear prediction du0); with the real parameters the kernel's return map is inside
    the loop and tangent supplies the consistent tangent. Stops at once, ok False, when the step breaks a contact
    (H_TOUCH on -> off), forms one (off -> on) or yields one (H_YLD off -> on, or H_DMAX above max(P_DY, its
    reference value): the kernel did the yield test itself); the flags broke, formed, yielded say which. mechanism
    (ok False) when the tangent is singular, its step is not finite, or the scaled residual max(|r_trans| /
    tol_trans, |r_rot| / tol_rot) fails to fall below 0.9 of its previous value on two consecutive iterations (the
    released components leave a motion no stiffness resists). Returns dict(ok, du, xn, hn, lam, out, Fnet, r, n_it,
    broke, formed, yielded, mechanism) for the last state evaluated (n_it kernel evaluations)."""
    du = np.array(du0, dtype=float).copy()
    touch0 = hist[:, dem.H_TOUCH] > 0.5
    yld0 = hist[:, dem.H_YLD] > 0.5
    dmax0 = np.maximum(prm_apply[:, dem.P_DY], hist[:, dem.H_DMAX]) * (1.0 + 1e-12)
    tt, tr = max(float(tol_trans), 1e-300), max(float(tol_rot), 1e-300)
    r_prev = math.inf
    stalls = 0
    res = None
    for it in range(n_max):
        xn, hn, _ = apply_displacement(x, ci, cj, prm_apply, hist, L, du.reshape(-1, 6), with_out=True)
        lam, Fnet, out = state(xn, ci, cj, prm_el, hn, L)
        r = Fext + Fnet
        touch1 = hn[:, dem.H_TOUCH] > 0.5
        res = dict(ok=False, du=du, xn=xn, hn=hn, lam=lam, out=out, Fnet=Fnet, r=r, n_it=it + 1,
                   broke=bool((touch0 & ~touch1).any()), formed=bool((~touch0 & touch1).any()),
                   yielded=bool((((hn[:, dem.H_YLD] > 0.5) & ~yld0) | (hn[:, dem.H_DMAX] > dmax0)).any()),
                   mechanism=False)
        if res["broke"] or res["formed"] or res["yielded"]:
            return res
        R6 = np.abs(r.reshape(-1, 6))
        rt, rr = float(R6[:, :3].max()), float(R6[:, 3:].max())
        if rt <= tt and rr <= tr:
            res["ok"] = True
            return res
        rs = max(rt / tt, rr / tr)
        stalls = stalls + 1 if rs > 0.9 * r_prev else 0
        if stalls >= 2:
            res["mechanism"] = True
            return res
        r_prev = rs
        try:
            fac = lu if tangent is None else tangent(xn, hn, lam, out)
            step = fac.solve(r)
        except RuntimeError:
            res["mechanism"] = True
            return res
        if not np.isfinite(step).all():
            res["mechanism"] = True
            return res
        du = du + step
    return res


def flow_increments(hn_el, hn_real):
    """Plastic increment (n_contacts x 3) of every sliding, rolling and twisting component between the elastic trial
    histories of a displacement (lifted caps) and the real kernel's histories of the same displacement: the norm of
    the sliding and of the rolling vector difference and the absolute twist difference; exactly zero where no clip
    branch was taken (the two kernels are bit-identical there), the clipped excess over the spring constant where
    one was."""
    d = hn_el - hn_real
    return np.column_stack([np.linalg.norm(d[:, dem.H_XT:dem.H_XT + 3], axis=1),
                            np.linalg.norm(d[:, dem.H_XR:dem.H_XR + 3], axis=1),
                            np.abs(d[:, dem.H_XTW])])


def twist_ambiguous(h_ref, hn_el, prm, caps_new):
    """Contacts whose elastic trial twist torque |k_tw h_el| exceeds the new twisting cap caps_new[:, 2] while the
    twist increment h_el - h_ref is exactly 0 or of the opposite sign to h_ref. There the kernel's clip takes the
    sign of the twist RATE (contact_force: sg = sign(mt)), so a round-off rate decides between keeping the twist at
    its cap, flipping it to the opposite cap and resetting it to 0: not resolved quasi-statically (the DEM takes
    such a contact)."""
    inc = hn_el[:, dem.H_XTW] - h_ref[:, dem.H_XTW]
    over = np.abs(prm[:, dem.P_KTW] * hn_el[:, dem.H_XTW]) > caps_new[:, 2]
    return over & ((inc == 0.0) | (inc * h_ref[:, dem.H_XTW] < 0.0))


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

"""Grains on one wall with the stiff contact modes slaved: rolling and twisting resolved, normal and sliding quasi-static.

A grain touching exactly one wall (and nothing else) has two kinds of contact motion. The normal and sliding springs
are stiff (periods 0.3-2.5 us); rolling (rocking, 40-260 us) and twisting (33-145 us) are soft. Every load on the grain
varies at <= 5 kHz, so the stiff modes follow the load quasi-statically: their dynamic deviation is of order
(f / f_stiff)^2 <= (25 kHz / 0.4 MHz)^2 ~ 4e-3 at the fastest rocking of the smallest grains, ~1e-4 typical (checked
against the full model in verify_full.check_roll1_vs_full). Holding them on their static solution (no normal motion,
no slip at the contact) leaves exact rigid-body equations for the soft modes (sphere on a plane, rolling without
slipping, about the contact point):

    (I + m r^2) dw_t/dt = [tau_ext + r n x F_ext + tau_roll]_t        I dw_n/dt = tau_ext.n + tau_twist
    v_t = r w x n,  F_t = m r (dw/dt x n) - F_ext,t (the contact's tangential force),  F_n,el = -F_ext.n

with the rolling and twisting spring-dashpot-sliders of dem.py (same formulas, same limits). The step is
min(rocking, twisting period) / steps_per_period. The grain goes back to the full model the moment its normal load
nears pull-off, its tangential force nears the sliding limit, or another body could reach it.
"""
from __future__ import annotations

import math

import numpy as np

from . import dem, slaved

try:
    from numba import njit
except ImportError:          # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _accel(g, x, v, w, m, I, rad, n, ax, p, h, Fc, dt_spring, acc_s, O, Od, off, gdrag, Fext, out):
    """Angular acceleration of grain g on its wall at the current state. The rolling and twisting histories are
    advanced by dt_spring with the current omega (LAMMPS ordering: increment, then force, then cap) and written to
    out[3:7]; out[0:3] = alpha; out[7] = pull-off ratio Fne/(-Fc) (> 0 means pulling); out[8] = |F_t| / limit;
    out[9:12] the quasi-static tangential contact force."""
    r = rad[g]
    Ic = I[g] + m[g] * r * r
    rx = x[g, 0] + off[0]
    ry = x[g, 1] + off[1]
    rz = x[g, 2] + off[2]
    c0 = O[1] * rz - O[2] * ry
    c1 = O[2] * rx - O[0] * rz
    c2 = O[0] * ry - O[1] * rx
    F0 = m[g] * (acc_s[0] - (O[1] * c2 - O[2] * c1) - 2.0 * (O[1] * v[g, 2] - O[2] * v[g, 1])
                 - (Od[1] * rz - Od[2] * ry) - gdrag[g] * v[g, 0]) + Fext[g, 0]
    F1 = m[g] * (acc_s[1] - (O[2] * c0 - O[0] * c2) - 2.0 * (O[2] * v[g, 0] - O[0] * v[g, 2])
                 - (Od[2] * rx - Od[0] * rz) - gdrag[g] * v[g, 1]) + Fext[g, 1]
    F2 = m[g] * (acc_s[2] - (O[0] * c1 - O[1] * c0) - 2.0 * (O[0] * v[g, 1] - O[1] * v[g, 0])
                 - (Od[0] * ry - Od[1] * rx) - gdrag[g] * v[g, 2]) + Fext[g, 2]
    ta0 = I[g] * (-(O[1] * w[g, 2] - O[2] * w[g, 1]) - Od[0])
    ta1 = I[g] * (-(O[2] * w[g, 0] - O[0] * w[g, 2]) - Od[1])
    ta2 = I[g] * (-(O[0] * w[g, 1] - O[1] * w[g, 0]) - Od[2])
    Fn_ext = F0 * n[0] + F1 * n[1] + F2 * n[2]
    Fne = -Fn_ext
    Fncrit = abs(Fne + 2.0 * Fc)
    kr, gr, mur = p[dem.P_KR], p[dem.P_GR], p[dem.P_MUR]
    ktw, gtw, mutw = p[dem.P_KTW], p[dem.P_GTW], p[dem.P_MUTW]
    vl0 = r * (w[g, 1] * n[2] - w[g, 2] * n[1])
    vl1 = r * (w[g, 2] * n[0] - w[g, 0] * n[2])
    vl2 = r * (w[g, 0] * n[1] - w[g, 1] * n[0])
    xr0 = h[dem.H_XR] + vl0 * dt_spring
    xr1 = h[dem.H_XR + 1] + vl1 * dt_spring
    xr2 = h[dem.H_XR + 2] + vl2 * dt_spring
    fr0 = -kr * xr0 - gr * vl0
    fr1 = -kr * xr1 - gr * vl1
    fr2 = -kr * xr2 - gr * vl2
    Frcrit = mur * Fncrit
    mfr = math.sqrt(fr0 * fr0 + fr1 * fr1 + fr2 * fr2)
    if mfr > Frcrit:
        if math.sqrt(xr0 * xr0 + xr1 * xr1 + xr2 * xr2) > 0.0:
            sc = Frcrit / mfr
            xr0 = -(sc * fr0 + gr * vl0) / kr
            xr1 = -(sc * fr1 + gr * vl1) / kr
            xr2 = -(sc * fr2 + gr * vl2) / kr
            fr0 *= sc
            fr1 *= sc
            fr2 *= sc
        else:                                    # as dem.contact_force
            fr0 = fr1 = fr2 = 0.0
    tr0 = r * (n[1] * fr2 - n[2] * fr1)
    tr1 = r * (n[2] * fr0 - n[0] * fr2)
    tr2 = r * (n[0] * fr1 - n[1] * fr0)
    mt = w[g, 0] * n[0] + w[g, 1] * n[1] + w[g, 2] * n[2]
    xtw = h[dem.H_XTW] + mt * dt_spring
    tq = -ktw * xtw - gtw * mt
    Mtcrit = mutw * Fncrit
    if abs(tq) > Mtcrit:
        sgt = 1.0 if mt > 0 else (-1.0 if mt < 0 else 0.0)
        xtw = (Mtcrit * sgt - gtw * mt) / ktw
        tq = -Mtcrit * sgt
    T0 = ta0 + r * (n[1] * F2 - n[2] * F1) + tr0 + tq * n[0]
    T1 = ta1 + r * (n[2] * F0 - n[0] * F2) + tr1 + tq * n[1]
    T2 = ta2 + r * (n[0] * F1 - n[1] * F0) + tr2 + tq * n[2]
    Tn = T0 * n[0] + T1 * n[1] + T2 * n[2]
    al0 = (T0 - Tn * n[0]) / Ic + Tn * n[0] / I[g]
    al1 = (T1 - Tn * n[1]) / Ic + Tn * n[1] / I[g]
    al2 = (T2 - Tn * n[2]) / Ic + Tn * n[2] / I[g]
    ft0 = m[g] * r * (al1 * n[2] - al2 * n[1]) - (F0 - Fn_ext * n[0])
    ft1 = m[g] * r * (al2 * n[0] - al0 * n[2]) - (F1 - Fn_ext * n[1])
    ft2 = m[g] * r * (al0 * n[1] - al1 * n[0]) - (F2 - Fn_ext * n[2])
    out[0], out[1], out[2] = al0, al1, al2
    out[3], out[4], out[5], out[6] = xr0, xr1, xr2, xtw
    out[7] = -Fne / Fc
    out[8] = math.sqrt(ft0 * ft0 + ft1 * ft1 + ft2 * ft2) / max(p[dem.P_MUT] * Fncrit, 1e-300)
    out[9], out[10], out[11] = ft0, ft1, ft2


@njit(cache=True)
def run(x, v, w, m, I, rad, face, prm, hist, L, dt, nsteps, acc, Om, Omd, off, gdrag, Fext, margin, flag,
        bv, ba, boff, exc):
    """Advance grains on one wall each (face 0..5) by nsteps of dt (velocity Verlet on the rolling and twisting
    rotation, as dem.run_steps). acc, Om, Omd: (nsteps + 1, 3), sampled at the step boundaries. A grain whose normal
    load nears pull-off or whose sliding force nears its limit (within `margin`) is stopped (flag = step + 1) and left
    at the start of that step. exc[g]: the most the grain's distance from its start exceeds the envelope
    bv t + ba t^2 / 2 + boff that other grains' capture times assumed for it (> 0: the assumption failed)."""
    M = x.shape[0]
    out = np.zeros(12)
    for g in range(M):
        ax = face[g] // 2
        n = np.zeros(3)
        n[ax] = 1.0 if face[g] % 2 == 0 else -1.0
        p = prm[g]
        h = hist[g]
        r = rad[g]
        Fc = h[dem.H_FCP] if h[dem.H_YLD] > 0.5 else p[dem.P_FC]
        if h[dem.H_YLD] > 0.5:               # flattened contact: a0 of radius R_p (dem.contact_force)
            ks = p[dem.P_KT] * (4.5 * math.pi * p[dem.P_W] * h[dem.H_RP] ** 2 / p[dem.P_ESTAR]) ** (1.0 / 3.0)
        elif int(p[dem.P_TMODE]) == dem.TANG_WADA:
            ks = p[dem.P_KT] * p[dem.P_A0]
        else:
            ks = p[dem.P_KT] * max(h[dem.H_A], 1e-30)
        _accel(g, x, v, w, m, I, rad, n, ax, p, h, Fc, 0.0, acc[0], Om[0], Omd[0], off, gdrag, Fext, out)
        a0, a1, a2 = out[0], out[1], out[2]
        x00, x01, x02 = x[g, 0], x[g, 1], x[g, 2]
        for s in range(nsteps):
            # state at the step start, kept as scalars for the rollback (h is only written after the check)
            xs0, xs1, xs2 = x[g, 0], x[g, 1], x[g, 2]
            vs0, vs1, vs2 = v[g, 0], v[g, 1], v[g, 2]
            ws0, ws1, ws2 = w[g, 0], w[g, 1], w[g, 2]
            # half kick, drift (no slip: v = r w x n)
            w[g, 0] += 0.5 * dt * a0
            w[g, 1] += 0.5 * dt * a1
            w[g, 2] += 0.5 * dt * a2
            v[g, 0] = r * (w[g, 1] * n[2] - w[g, 2] * n[1])
            v[g, 1] = r * (w[g, 2] * n[0] - w[g, 0] * n[2])
            v[g, 2] = r * (w[g, 0] * n[1] - w[g, 1] * n[0])
            for d in range(3):
                if d != ax:
                    x[g, d] += dt * v[g, d]
            tt = (s + 1) * dt
            dd = math.sqrt((x[g, 0] - x00) ** 2 + (x[g, 1] - x01) ** 2 + (x[g, 2] - x02) ** 2)
            e = dd - (bv[g] * tt + 0.5 * ba[g] * tt * tt + boff[g])
            if e > exc[g]:
                exc[g] = e
            # springs advanced with the half-step omega; forces at the new state
            _accel(g, x, v, w, m, I, rad, n, ax, p, h, Fc, dt, acc[s + 1], Om[s + 1], Omd[s + 1], off, gdrag,
                   Fext, out)
            if out[7] > 1.0 - margin or out[8] > 1.0 - margin:
                x[g, 0], x[g, 1], x[g, 2] = xs0, xs1, xs2
                v[g, 0], v[g, 1], v[g, 2] = vs0, vs1, vs2
                w[g, 0], w[g, 1], w[g, 2] = ws0, ws1, ws2
                flag[g] = s + 1
                break
            # the normal overlap on its quasi-static (JKR) value for the current normal load; plastic loading goes
            # to the full model (dem.contact_force's rule)
            Fne_now = -out[7] * Fc
            dq, aq = slaved.jkr_overlap(p, h, Fne_now, h[dem.H_A] if h[dem.H_A] > 0 else p[dem.P_A0])
            if dq > p[dem.P_DY] and dq >= h[dem.H_DMAX] * (1.0 - 1e-12):
                x[g, 0], x[g, 1], x[g, 2] = xs0, xs1, xs2
                v[g, 0], v[g, 1], v[g, 2] = vs0, vs1, vs2
                w[g, 0], w[g, 1], w[g, 2] = ws0, ws1, ws2
                flag[g] = s + 1
                break
            x[g, ax] = (r - dq) if face[g] % 2 == 0 else (L[ax] - r + dq)
            h[dem.H_A] = aq
            if dq > h[dem.H_DMAX]:
                h[dem.H_DMAX] = dq
            h[dem.H_XR], h[dem.H_XR + 1], h[dem.H_XR + 2], h[dem.H_XTW] = out[3], out[4], out[5], out[6]
            h[dem.H_XT] = -out[9] / ks
            h[dem.H_XT + 1] = -out[10] / ks
            h[dem.H_XT + 2] = -out[11] / ks
            a0, a1, a2 = out[0], out[1], out[2]
            w[g, 0] += 0.5 * dt * a0
            w[g, 1] += 0.5 * dt * a1
            w[g, 2] += 0.5 * dt * a2
            v[g, 0] = r * (w[g, 1] * n[2] - w[g, 2] * n[1])
            v[g, 1] = r * (w[g, 2] * n[0] - w[g, 0] * n[2])
            v[g, 2] = r * (w[g, 0] * n[1] - w[g, 1] * n[0])

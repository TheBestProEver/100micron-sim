"""Full-stiffness soft-sphere contact dynamics (DEM) for grains that touch each other or the walls.

Every grain is its own rigid sphere with 6 degrees of freedom. A contact carries the complete adhesive elastic
contact model used for dust aggregates (Wada et al. 2007, ApJ 661, 320; Dominik & Tielens 1995/1997; Krijt et al.
2014, J. Phys. D 47, 175302), written in the form of the LAMMPS GRANULAR package (pair_style granular jkr ...
tangential mindlin ... rolling sds ... twisting sds; src/GRANULAR/granular_model.cpp, gran_sub_mod_*.cpp), so that
the two codes can be compared term by term:

* normal: JKR, F(a) = 4 E* a^3/(3R) - sqrt(8 pi w E* a^3), delta(a) = a^2/R - sqrt(2 pi w a/E*); contact forms at
  delta = 0 and breaks at the end of the stable branch delta_c (Chokshi et al. 1993; Wada et al. 2007), or at the
  maximum tension as LAMMPS does (break="force", for the cross-check); optional Thornton & Ning (1998) plastic
  flattening; damping either DOLL-E's Tsuji-type law evaluated with the JKR radius or LAMMPS "viscoelastic";
* sliding: spring k_s = 8 G* a0 (Wada et al. 2007 eq. 12; a0 rather than the current a keeps the spring energy
  conservative; LAMMPS "tangential linear_history" with this k; the Mindlin variant 8 G* a, LAMMPS "mindlin", is
  available), history rotated into the tangent plane every step (Luding 2008, eq. 17),
  damping x_gt times the normal damping prefactor, Coulomb limit mu_t |F_ne + 2 F_c| (Thornton & Yin 1991; Marshall
  2009 eq. 43, as in LAMMPS);
* rolling: spring-dashpot-slider on the rolling displacement xi_r = int Reff (w_i - w_j) x n dt (Wang et al. 2015;
  LAMMPS "rolling sds"), k_r = 4 F_c / Reff (Wada et al. 2007 eq. 22, the Dominik & Tielens 1995 rolling spring),
  limit mu_r |F_ne + 2 F_c| with mu_r = 2 xi_crit / Reff so that the critical torque at zero load is 4 F_c xi_crit;
  xi_crit = h a0 / 12 (Krijt et al. 2014, eq. 16), h the adhesion hysteresis;
* twisting: spring-dashpot-slider on int (w_i - w_j).n dt, k_tw = (16/3) G a0^3 and a limit equal to
  G a0^3 / (3 pi) at zero load (Wada et al. 2007 eqs. 26 and 37);
* rolling and twisting damping from an assumed quality factor Q of the rocking and twisting modes.

The time step resolves the fastest contact mode (the sliding mode, period ~0.3-1.1 us for 25-75 um grains).
Integration is velocity Verlet for translation and rotation (as LAMMPS fix nve/sphere): half kick, drift, forces at
the new positions with the half-step velocities, half kick.

Provenance: equations from the papers above; the code is written for DOLL-E (no LAMMPS code is copied; the LAMMPS
sources were read to match definitions). See provenance.py.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange
except ImportError:          # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]
    prange = range

# ---------------------------------------------------------------------------- parameter vector layout
P_REFF, P_ESTAR, P_W, P_FC, P_A0, P_DBREAK, P_DAMPC, P_DAMPMODE, P_MEFF, P_KT, P_MUT, P_XGT, P_KR, P_GR, P_MUR, \
    P_KTW, P_GTW, P_MUTW, P_DY, P_PY, P_RADI, P_RADJ, P_BREAKMODE, P_GSHEAR, P_TMODE = range(25)
NP = 25
# sliding spring: k_s = 8 G* a0 (Wada et al. 2007 eq. 12; energy-conserving; LAMMPS "linear_history" with that k)
# or 8 G* a with the current contact radius (Mindlin; LAMMPS "mindlin"; not energy-conserving when a changes)
TANG_WADA, TANG_MINDLIN = 0, 1
# history layout
H_XT, H_XR, H_XTW, H_A, H_YLD, H_DMAX, H_RP, H_DP, H_FCP, H_TOUCH = 0, 3, 6, 7, 8, 9, 10, 11, 12, 13
NHP = 14                    # physics slots (cleared when a contact forms or breaks)
# event slots, kept across make/break so the driver can log impacts and separations
H_NFORM, H_TFORM, H_VIN, H_VTIN, H_NBREAK, H_TBREAK, H_VOUT, H_AMAX = 14, 15, 16, 17, 18, 19, 20, 21
NH = 22
NOUT = 20

DAMP_NONE, DAMP_TSUJI_JKR, DAMP_VISCOELASTIC = 0, 1, 2
BREAK_DISPLACEMENT, BREAK_FORCE = 0, 1


# ---------------------------------------------------------------------------- JKR algebra
@njit(cache=True)
def jkr_a(delta, R, w, E):
    """Stable-branch JKR contact radius for overlap delta (closed form, as LAMMPS calculate_contact_radius;
    cohesion = w/2). Returns 0 if delta is below the end of the stable branch."""
    g = 0.5 * w
    R2 = R * R
    dR = delta * R
    dR2 = dR * dR
    t0 = g * g * R2 * R2 * E
    t1 = 27.0 * math.pi * math.pi * t0
    t2 = 8.0 * dR * dR2 * E * E * E
    t3 = 4.0 * dR2 * E
    s1 = t0 * (t1 + 2.0 * t2)
    if s1 < 0.0:
        s1 = 0.0
    t4 = (t1 + t2 + 3.0 * math.sqrt(3.0) * math.pi * math.sqrt(s1)) ** (1.0 / 3.0)
    t5 = t3 / t4 + t4 / E
    s2 = 2.0 * dR + t5
    if s2 < 0.0:
        s2 = 0.0
    t6 = math.sqrt(s2)
    s3 = 4.0 * dR - t5 + 6.0 * math.sqrt(6.0) * g * math.pi * R2 / (E * t6) if t6 > 0 else 0.0
    if s3 < 0.0:
        s3 = 0.0
    return (t6 + math.sqrt(s3)) / math.sqrt(6.0)


@njit(cache=True)
def jkr_force(a, R, w, E):
    """JKR elastic + adhesive normal force (positive = repulsive)."""
    return 4.0 * E * a ** 3 / (3.0 * R) - math.sqrt(8.0 * math.pi * w * E * a ** 3)


@njit(cache=True)
def jkr_break_overlap(R, w, E, mode):
    """Overlap (negative) at which a contact of radius of curvature R breaks."""
    a0 = (4.5 * math.pi * w * R * R / E) ** (1.0 / 3.0)
    a = a0 / 6.0 ** (2.0 / 3.0) if mode == BREAK_DISPLACEMENT else a0 / 4.0 ** (1.0 / 3.0)
    return a * a / R - math.sqrt(2.0 * math.pi * w * a / E)


# ---------------------------------------------------------------------------- contact parameters
@njit(cache=True)
def make_params(radi, radj, mi, mj, Ei, nui, Ej, nuj, w, e_dry, p_y, mu_t, h_roll, q_roll, q_twist, damp_mode,
                break_mode, x_gt, eta_visc, tang_mode=TANG_WADA):
    """Parameter vector of one contact. radj = 0 / mj = inf for a wall. w: work of adhesion. h_roll: adhesion
    hysteresis of the Krijt et al. (2014) rolling law (xi_crit = h a0 / 12). q_roll, q_twist: quality factors.
    eta_visc: LAMMPS viscoelastic damping constant (only for damp_mode = viscoelastic)."""
    p = np.zeros(NP)
    wall = radj <= 0.0
    R = radi if wall else radi * radj / (radi + radj)
    meff = mi if wall else mi * mj / (mi + mj)
    E = 1.0 / ((1.0 - nui * nui) / Ei + (1.0 - nuj * nuj) / Ej)
    Gi = Ei / (2.0 * (1.0 + nui))
    Gj = Ej / (2.0 * (1.0 + nuj))
    Gst = 1.0 / ((2.0 - nui) / Gi + (2.0 - nuj) / Gj)      # sliding modulus G* (Wada 2007 eq. 13)
    Gtw = 1.0 / (1.0 / Gi + 1.0 / Gj)                      # twisting modulus G (Wada 2007 eq. 26)
    w = max(w, 1e-12)
    Fc = 1.5 * math.pi * w * R
    a0 = (4.5 * math.pi * w * R * R / E) ** (1.0 / 3.0)
    p[P_REFF], p[P_ESTAR], p[P_W], p[P_FC], p[P_A0] = R, E, w, Fc, a0
    p[P_DBREAK] = jkr_break_overlap(R, w, E, break_mode)
    p[P_BREAKMODE] = break_mode
    p[P_MEFF] = meff
    p[P_DAMPMODE] = damp_mode
    if damp_mode == DAMP_TSUJI_JKR:
        e = min(max(e_dry, 1e-4), 1.0)
        beta = math.log(e) / math.sqrt(math.log(e) ** 2 + math.pi ** 2) if e < 1.0 else 0.0
        p[P_DAMPC] = -2.0 * math.sqrt(5.0 / 6.0) * beta * math.sqrt(2.0 * E * meff)    # F_d = -c sqrt(a) v_n
    elif damp_mode == DAMP_VISCOELASTIC:
        p[P_DAMPC] = eta_visc * meff                                                 # F_d = -c a v_n
    p[P_KT] = 8.0 * Gst
    p[P_MUT] = mu_t
    p[P_XGT] = x_gt
    # rolling (Wada 2007 eq. 22; Krijt 2014 eq. 16); limit at zero load: k_r xi_crit = mu_r * 2 F_c
    kr = 4.0 * Fc / R
    xi_crit = h_roll * a0 / 12.0
    p[P_KR] = kr
    p[P_MUR] = 2.0 * xi_crit / R
    # rocking mode about the contact point: k_r R^2 / I_c, I_c = I + m r^2 = 1.4 m r^2 (wall); pair: counter-rotation
    if wall:
        Ic = 1.4 * mi * radi * radi
        lev = radi
    else:
        Ii = 0.4 * mi * radi * radi
        Ij = 0.4 * mj * radj * radj
        Ic = Ii * Ij / (Ii + Ij)
        lev = R
    om_r = math.sqrt(kr * R * R / Ic)
    p[P_GR] = Ic * om_r / (q_roll * lev * lev) if q_roll > 0 else 0.0
    ktw = 16.0 / 3.0 * Gtw * a0 ** 3
    p[P_KTW] = ktw
    p[P_MUTW] = Gtw * a0 ** 3 / (3.0 * math.pi) / (2.0 * Fc)
    It = 0.4 * mi * radi * radi if wall else (0.4 * mi * radi * radi) * (0.4 * mj * radj * radj) / (
        0.4 * mi * radi * radi + 0.4 * mj * radj * radj)
    om_t = math.sqrt(ktw / It)
    p[P_GTW] = It * om_t / q_twist if q_twist > 0 else 0.0
    p[P_PY] = p_y
    p[P_DY] = R * (math.pi * p_y / (2.0 * E)) ** 2 if p_y < 1e300 else 1e300
    p[P_RADI], p[P_RADJ] = radi, radj
    p[P_GSHEAR] = Gtw
    p[P_TMODE] = tang_mode
    return p


# ---------------------------------------------------------------------------- one contact
@njit(cache=True)
def _rot_rescale(v, o, n0, n1, n2):
    """Rotate history vector v[o:o+3] into the plane normal to n, keeping its length (LAMMPS rotate_rescale_vec)."""
    d = v[o] * n0 + v[o + 1] * n1 + v[o + 2] * n2
    if d == 0.0:
        return
    m0 = math.sqrt(v[o] * v[o] + v[o + 1] * v[o + 1] + v[o + 2] * v[o + 2])
    v[o] -= d * n0
    v[o + 1] -= d * n1
    v[o + 2] -= d * n2
    m1 = math.sqrt(v[o] * v[o] + v[o + 1] * v[o + 1] + v[o + 2] * v[o + 2])
    s = m0 / m1 if m1 > 0 else 0.0
    v[o] *= s
    v[o + 1] *= s
    v[o + 2] *= s


@njit(cache=True)
def _record_break(dx, r, vi, vj, h, tnow):
    n0, n1, n2 = dx[0] / r, dx[1] / r, dx[2] / r
    vn = (vi[0] - vj[0]) * n0 + (vi[1] - vj[1]) * n1 + (vi[2] - vj[2]) * n2
    for k in range(NHP):
        h[k] = 0.0
    h[H_NBREAK] += 1.0
    h[H_TBREAK] = tnow
    h[H_VOUT] = vn


@njit(cache=True)
def contact_force(dx, r, vi, vj, wi, wj, p, h, dt, out, tnow=0.0):
    """Force on grain i and torques on i and j from one contact.
    dx = x_i - x_j (for a wall: from the wall point to the centre), r = |dx|. h: history (updated in place).
    out[0:3] force on i, out[3:6] torque on i, out[6:9] torque on j, out[9] normal force, out[10] = 1 if still in
    contact, out[11] elastic normal force F_ne, out[12:15] sliding force, out[15:18] rolling pseudo-force,
    out[18] twisting torque, out[19] pull-off force of this contact (flattened if yielded). Writes out."""
    radi, radj = p[P_RADI], p[P_RADJ]
    radsum = radi + (radj if radj > 0 else 0.0)
    R, E, w, Fc = p[P_REFF], p[P_ESTAR], p[P_W], p[P_FC]
    delta = radsum - r
    for k in range(NOUT):
        out[k] = 0.0
    # touching? (contact forms at delta >= 0; once formed it holds until the break overlap)
    touching = h[H_TOUCH] > 0.5
    if not touching:
        if delta < 0.0:
            return
        for k in range(NHP):                 # a new contact: no sliding, rolling, twisting or plastic history
            h[k] = 0.0
        h[H_TOUCH] = 1.0
        # impact record: normal approach speed and tangential speed of the centres at first touch
        n0, n1, n2 = dx[0] / r, dx[1] / r, dx[2] / r
        vr0, vr1, vr2 = vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2]
        vn0 = vr0 * n0 + vr1 * n1 + vr2 * n2
        h[H_NFORM] += 1.0
        h[H_TFORM] = tnow
        h[H_VIN] = -vn0
        h[H_VTIN] = math.sqrt(max(vr0 * vr0 + vr1 * vr1 + vr2 * vr2 - vn0 * vn0, 0.0))
        h[H_AMAX] = 0.0
    # plastic (Thornton & Ning 1998) bookkeeping: loading beyond delta_y flattens the contact (radius R_p) and
    # leaves a permanent indentation d_p; unloading/reloading below d_max is elastic JKR of the flattened contact
    yielded = h[H_YLD] > 0.5
    if delta > p[P_DY] and delta >= h[H_DMAX] * (1.0 - 1e-12):
        kH = 4.0 / 3.0 * E * math.sqrt(R)
        Fy = kH * p[P_DY] ** 1.5
        Fmax = Fy + math.pi * p[P_PY] * R * (delta - p[P_DY])
        Rp = (4.0 * E / (3.0 * Fmax)) * ((2.0 * Fmax + Fy) / (2.0 * math.pi * p[P_PY])) ** 1.5
        kP = 4.0 / 3.0 * E * math.sqrt(Rp)
        h[H_RP] = Rp
        h[H_DP] = delta - (Fmax / kP) ** (2.0 / 3.0)
        h[H_FCP] = 1.5 * math.pi * w * Rp
        h[H_YLD] = 1.0
        yielded = True
        a = jkr_a(delta, R, w, E)
        Fne = jkr_force(a, R, w, E) + Fmax - kH * delta ** 1.5      # plastic loading law, adhesion with R
        Fpull = Fc
    elif yielded:
        Rp = h[H_RP]
        ds = delta - h[H_DP]
        if ds < jkr_break_overlap(Rp, w, E, int(p[P_BREAKMODE])):
            _record_break(dx, r, vi, vj, h, tnow)
            return
        a = jkr_a(ds, Rp, w, E)
        Fne = jkr_force(a, Rp, w, E)
        Fpull = h[H_FCP]
    else:
        if delta < p[P_DBREAK]:
            _record_break(dx, r, vi, vj, h, tnow)
            return
        a = jkr_a(delta, R, w, E)
        Fne = jkr_force(a, R, w, E)
        Fpull = Fc
    if delta > h[H_DMAX]:
        h[H_DMAX] = delta
    h[H_A] = a
    if a > h[H_AMAX]:
        h[H_AMAX] = a
    n0, n1, n2 = dx[0] / r, dx[1] / r, dx[2] / r
    vr0, vr1, vr2 = vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2]
    vn = vr0 * n0 + vr1 * n1 + vr2 * n2          # > 0: separating
    # normal damping
    mode = int(p[P_DAMPMODE])
    if mode == DAMP_TSUJI_JKR:
        dpre = p[P_DAMPC] * math.sqrt(a)
    elif mode == DAMP_VISCOELASTIC:
        dpre = p[P_DAMPC] * a
    else:
        dpre = 0.0
    Fn = Fne - dpre * vn
    Fncrit = abs(Fne + 2.0 * Fpull)
    # relative tangential velocity at the contact point: vt - (radi wi + radj wj) x n
    rj = radj if radj > 0 else 0.0
    wr0 = radi * wi[0] + rj * wj[0]
    wr1 = radi * wi[1] + rj * wj[1]
    wr2 = radi * wi[2] + rj * wj[2]
    vt0, vt1, vt2 = vr0 - vn * n0, vr1 - vn * n1, vr2 - vn * n2
    vtr0 = vt0 - (wr1 * n2 - wr2 * n1)
    vtr1 = vt1 - (wr2 * n0 - wr0 * n2)
    vtr2 = vt2 - (wr0 * n1 - wr1 * n0)
    # ---- sliding (Mindlin, LAMMPS ordering: rotate, increment, force, cap)
    xt = h[H_XT:H_XT + 3]
    _rot_rescale(h, H_XT, n0, n1, n2)
    xt[0] += vtr0 * dt
    xt[1] += vtr1 * dt
    xt[2] += vtr2 * dt
    if int(p[P_TMODE]) == TANG_MINDLIN:
        ks = p[P_KT] * a
    elif yielded:
        ks = p[P_KT] * (4.5 * math.pi * w * h[H_RP] ** 2 / E) ** (1.0 / 3.0)    # a0 of the flattened contact
    else:
        ks = p[P_KT] * p[P_A0]
    dt_damp = p[P_XGT] * dpre
    fs0 = -ks * xt[0] - dt_damp * vtr0
    fs1 = -ks * xt[1] - dt_damp * vtr1
    fs2 = -ks * xt[2] - dt_damp * vtr2
    Fscrit = p[P_MUT] * Fncrit
    mfs = math.sqrt(fs0 * fs0 + fs1 * fs1 + fs2 * fs2)
    if mfs > Fscrit:
        if math.sqrt(xt[0] ** 2 + xt[1] ** 2 + xt[2] ** 2) > 0.0 and ks > 0:
            s = Fscrit / mfs
            fs0 *= s
            fs1 *= s
            fs2 *= s
            xt[0] = -(fs0 + dt_damp * vtr0) / ks
            xt[1] = -(fs1 + dt_damp * vtr1) / ks
            xt[2] = -(fs2 + dt_damp * vtr2) / ks
        else:
            fs0 = fs1 = fs2 = 0.0
    out[0] = Fn * n0 + fs0
    out[1] = Fn * n1 + fs1
    out[2] = Fn * n2 + fs2
    out[9] = Fn
    out[11] = Fne
    out[12], out[13], out[14] = fs0, fs1, fs2
    out[19] = Fpull
    # torque of the tangential force: -(n x fs) * lever
    c0 = n1 * fs2 - n2 * fs1
    c1 = n2 * fs0 - n0 * fs2
    c2 = n0 * fs1 - n1 * fs0
    li = radi - 0.5 * delta if radj > 0 else radi
    lj = radj - 0.5 * delta if radj > 0 else 0.0
    out[3], out[4], out[5] = -c0 * li, -c1 * li, -c2 * li
    out[6], out[7], out[8] = -c0 * lj, -c1 * lj, -c2 * lj
    # ---- rolling (sds)
    rr0, rr1, rr2 = wi[0] - wj[0], wi[1] - wj[1], wi[2] - wj[2]
    if p[P_KR] > 0.0:
        vl0 = R * (rr1 * n2 - rr2 * n1)
        vl1 = R * (rr2 * n0 - rr0 * n2)
        vl2 = R * (rr0 * n1 - rr1 * n0)
        xr = h[H_XR:H_XR + 3]
        _rot_rescale(h, H_XR, n0, n1, n2)
        xr[0] += vl0 * dt
        xr[1] += vl1 * dt
        xr[2] += vl2 * dt
        kr, gr = p[P_KR], p[P_GR]
        fr0 = -kr * xr[0] - gr * vl0
        fr1 = -kr * xr[1] - gr * vl1
        fr2 = -kr * xr[2] - gr * vl2
        Frcrit = p[P_MUR] * Fncrit
        mfr = math.sqrt(fr0 * fr0 + fr1 * fr1 + fr2 * fr2)
        if mfr > Frcrit:
            if math.sqrt(xr[0] ** 2 + xr[1] ** 2 + xr[2] ** 2) > 0.0:
                s = Frcrit / mfr
                xr[0] = -(s * fr0 + gr * vl0) / kr
                xr[1] = -(s * fr1 + gr * vl1) / kr
                xr[2] = -(s * fr2 + gr * vl2) / kr
                fr0 *= s
                fr1 *= s
                fr2 *= s
            else:
                fr0 = fr1 = fr2 = 0.0
        out[15], out[16], out[17] = fr0, fr1, fr2
        t0 = R * (n1 * fr2 - n2 * fr1)
        t1 = R * (n2 * fr0 - n0 * fr2)
        t2 = R * (n0 * fr1 - n1 * fr0)
        out[3] += t0
        out[4] += t1
        out[5] += t2
        out[6] -= t0
        out[7] -= t1
        out[8] -= t2
    # ---- twisting (sds)
    if p[P_KTW] > 0.0:
        mt = rr0 * n0 + rr1 * n1 + rr2 * n2
        h[H_XTW] += mt * dt
        tq = -p[P_KTW] * h[H_XTW] - p[P_GTW] * mt
        Mtcrit = p[P_MUTW] * Fncrit
        if abs(tq) > Mtcrit:
            sg = 1.0 if mt > 0 else (-1.0 if mt < 0 else 0.0)
            h[H_XTW] = (Mtcrit * sg - p[P_GTW] * mt) / p[P_KTW]
            tq = -Mtcrit * sg
        out[18] = tq
        out[3] += tq * n0
        out[4] += tq * n1
        out[5] += tq * n2
        out[6] -= tq * n0
        out[7] -= tq * n1
        out[8] -= tq * n2
    out[10] = 1.0


# ---------------------------------------------------------------------------- forces on all grains
@njit(cache=True)
def _wall_dx(x, L, face, dx):
    """Vector from the wall point to the centre for box face 0..5 (-x, +x, -y, +y, -z, +z); returns its length."""
    ax = face // 2
    dx[0] = 0.0
    dx[1] = 0.0
    dx[2] = 0.0
    if face % 2 == 0:
        dx[ax] = x[ax]
    else:
        dx[ax] = x[ax] - L[ax]
    return abs(dx[ax])


@njit(cache=True)
def compute_forces(x, v, w, m, I, L, ci, cj, prm, hist, dt, acc, Om, Omd, off, gdrag, F, T, Fn_out, Fext, tnow,
                   OUT=np.empty((0, 20))):
    """Contact forces of all listed contacts plus body forces in the (spinning) tubule frame.
    cj < 0: wall face -1 - cj. acc: uniform apparent acceleration; Om, Omd: payload spin and its rate; off: tubule
    corner relative to the spin axis; gdrag: gas drag rate per grain [1/s]."""
    N = x.shape[0]
    for i in range(N):
        rx, ry, rz = x[i, 0] + off[0], x[i, 1] + off[1], x[i, 2] + off[2]
        # -Om x (Om x r)
        c0 = Om[1] * rz - Om[2] * ry
        c1 = Om[2] * rx - Om[0] * rz
        c2 = Om[0] * ry - Om[1] * rx
        cf0 = -(Om[1] * c2 - Om[2] * c1)
        cf1 = -(Om[2] * c0 - Om[0] * c2)
        cf2 = -(Om[0] * c1 - Om[1] * c0)
        # -2 Om x v and -Omd x r
        co0 = -2.0 * (Om[1] * v[i, 2] - Om[2] * v[i, 1]) - (Omd[1] * rz - Omd[2] * ry)
        co1 = -2.0 * (Om[2] * v[i, 0] - Om[0] * v[i, 2]) - (Omd[2] * rx - Omd[0] * rz)
        co2 = -2.0 * (Om[0] * v[i, 1] - Om[1] * v[i, 0]) - (Omd[0] * ry - Omd[1] * rx)
        F[i, 0] = m[i] * (acc[0] + cf0 + co0 - gdrag[i] * v[i, 0]) + Fext[i, 0]
        F[i, 1] = m[i] * (acc[1] + cf1 + co1 - gdrag[i] * v[i, 1]) + Fext[i, 1]
        F[i, 2] = m[i] * (acc[2] + cf2 + co2 - gdrag[i] * v[i, 2]) + Fext[i, 2]
        # spin relative to the tubule: dw/dt = T/I - Om x w - Omd  (sphere)
        T[i, 0] = I[i] * (-(Om[1] * w[i, 2] - Om[2] * w[i, 1]) - Omd[0])
        T[i, 1] = I[i] * (-(Om[2] * w[i, 0] - Om[0] * w[i, 2]) - Omd[1])
        T[i, 2] = I[i] * (-(Om[0] * w[i, 1] - Om[1] * w[i, 0]) - Omd[2])
    K = ci.shape[0]
    if OUT.shape[0] < K:
        OUT = np.empty((K, NOUT))
    if K >= PARALLEL_MIN_CONTACTS:
        _contact_loop(x, v, w, L, ci, cj, prm, hist, dt, OUT, tnow)
    else:
        _contact_loop_serial(x, v, w, L, ci, cj, prm, hist, dt, OUT, tnow)
    for k in range(K):
        Fn_out[k] = OUT[k, 9]
        if OUT[k, 10] == 0.0:
            continue
        i = ci[k]
        j = cj[k]
        F[i, 0] += OUT[k, 0]
        F[i, 1] += OUT[k, 1]
        F[i, 2] += OUT[k, 2]
        T[i, 0] += OUT[k, 3]
        T[i, 1] += OUT[k, 4]
        T[i, 2] += OUT[k, 5]
        if j >= 0:
            F[j, 0] -= OUT[k, 0]
            F[j, 1] -= OUT[k, 1]
            F[j, 2] -= OUT[k, 2]
            T[j, 0] += OUT[k, 6]
            T[j, 1] += OUT[k, 7]
            T[j, 2] += OUT[k, 8]


PARALLEL_MIN_CONTACTS = 3000          # below this, thread start-up costs more than it saves (measured)


@njit(cache=True)
def _contact_loop_serial(x, v, w, L, ci, cj, prm, hist, dt, OUT, tnow):
    zero = np.zeros(3)
    dxv = np.empty(3)
    for k in range(ci.shape[0]):
        i = ci[k]
        j = cj[k]
        if j >= 0:
            dxv[0] = x[i, 0] - x[j, 0]
            dxv[1] = x[i, 1] - x[j, 1]
            dxv[2] = x[i, 2] - x[j, 2]
            r = math.sqrt(dxv[0] ** 2 + dxv[1] ** 2 + dxv[2] ** 2)
            contact_force(dxv, r, v[i], v[j], w[i], w[j], prm[k], hist[k], dt, OUT[k], tnow)
        else:
            r = _wall_dx(x[i], L, -1 - j, dxv)
            contact_force(dxv, r, v[i], zero, w[i], zero, prm[k], hist[k], dt, OUT[k], tnow)


@njit(cache=True, parallel=True)
def _contact_loop(x, v, w, L, ci, cj, prm, hist, dt, OUT, tnow):
    """All contacts in parallel (each contact owns its history row; forces are summed afterwards in a fixed order,
    so results do not depend on the number of threads)."""
    zero = np.zeros(3)
    for k in prange(ci.shape[0]):
        i = ci[k]
        j = cj[k]
        dxv = np.empty(3)
        if j >= 0:
            dxv[0] = x[i, 0] - x[j, 0]
            dxv[1] = x[i, 1] - x[j, 1]
            dxv[2] = x[i, 2] - x[j, 2]
            r = math.sqrt(dxv[0] ** 2 + dxv[1] ** 2 + dxv[2] ** 2)
            contact_force(dxv, r, v[i], v[j], w[i], w[j], prm[k], hist[k], dt, OUT[k], tnow)
        else:
            r = _wall_dx(x[i], L, -1 - j, dxv)
            contact_force(dxv, r, v[i], zero, w[i], zero, prm[k], hist[k], dt, OUT[k], tnow)


@njit(cache=True)
def _event_time(x, v, F, m, L, ci, cj, prm, hist, h):
    """Earliest time in (0, h] at which some contact forms (gap reaches 0) or breaks (overlap reaches its break
    value), along this sub-step's drift x + tau (v + h F / 2m) (exact for the drift); h if none."""
    best = h
    for k in range(ci.shape[0]):
        i = ci[k]
        j = cj[k]
        p = prm[k]
        hk = hist[k]
        touching = hk[H_TOUCH] > 0.5
        radi, radj = p[P_RADI], p[P_RADJ]
        if touching:
            if hk[H_YLD] > 0.5:
                dbr = hk[H_DP] + jkr_break_overlap(hk[H_RP], p[P_W], p[P_ESTAR], int(p[P_BREAKMODE]))
            else:
                dbr = p[P_DBREAK]
            D = radi + (radj if radj > 0 else 0.0) - dbr
        else:
            D = radi + (radj if radj > 0 else 0.0)
        if j >= 0:
            d0 = x[i, 0] - x[j, 0]
            d1 = x[i, 1] - x[j, 1]
            d2 = x[i, 2] - x[j, 2]
            u0 = (v[i, 0] + 0.5 * h * F[i, 0] / m[i]) - (v[j, 0] + 0.5 * h * F[j, 0] / m[j])
            u1 = (v[i, 1] + 0.5 * h * F[i, 1] / m[i]) - (v[j, 1] + 0.5 * h * F[j, 1] / m[j])
            u2 = (v[i, 2] + 0.5 * h * F[i, 2] / m[i]) - (v[j, 2] + 0.5 * h * F[j, 2] / m[j])
            a2 = u0 * u0 + u1 * u1 + u2 * u2
            if a2 == 0.0:
                continue
            b = d0 * u0 + d1 * u1 + d2 * u2
            c = d0 * d0 + d1 * d1 + d2 * d2 - D * D
            disc = b * b - a2 * c
            if disc < 0.0:
                continue
            sq = math.sqrt(disc)
            if touching:
                tau = (-b + sq) / a2          # distance grows through D (separation)
                if c >= 0.0:
                    continue
            else:
                tau = (-b - sq) / a2          # distance shrinks through D (first touch)
                if c <= 0.0:
                    continue
        else:
            f = -1 - j
            ax = f // 2
            d = x[i, ax] if f % 2 == 0 else L[ax] - x[i, ax]
            u = v[i, ax] + 0.5 * h * F[i, ax] / m[i]
            if f % 2 == 1:
                u = -u
            if u == 0.0:
                continue
            tau = (D - d) / u
            if touching and not (d < D and u > 0.0):
                continue
            if not touching and not (d > D and u < 0.0):
                continue
        if 0.0 < tau < best:
            best = tau
    return best


@njit(cache=True)
def run_steps(x, v, w, m, I, L, ci, cj, prm, hist, dt, nsteps, acc, Om, Omd, off, gdrag, F, T, x_build, lim,
              Fn_out, Fext, t0, xs, dmax, vmax2, events=True):
    """Velocity-Verlet steps. acc, Om, Omd: (nsteps, 3) arrays sampled at each step's end. F, T must hold the
    forces at the start. A step in which some contact forms or breaks is split at that instant (event location;
    the force jumps there), so the discontinuity is integrated to second order. Stops early (returns the number of
    steps done) when some grain i has moved more than lim[i] since x_build (the caller then rebuilds the contact
    list). Records each grain's largest distance from xs in dmax, and the largest squared speed (the drift velocity
    and the end-of-step velocity) in vmax2[0]."""
    N = x.shape[0]
    OUT = np.empty((ci.shape[0], NOUT))
    for s in range(nsteps):
        rem = dt
        for _ in range(64):
            h = rem
            te = _event_time(x, v, F, m, L, ci, cj, prm, hist, h) if events else h
            if te < h * (1.0 - 1e-9) and te > 1e-4 * dt:
                h = te * (1.0 + 1e-9)
            for i in range(N):
                for d in range(3):
                    v[i, d] += 0.5 * h * F[i, d] / m[i]
                    w[i, d] += 0.5 * h * T[i, d] / I[i]
                    x[i, d] += h * v[i, d]
                sp = v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1] + v[i, 2] * v[i, 2]
                if sp > vmax2[0]:
                    vmax2[0] = sp
            rem -= h
            compute_forces(x, v, w, m, I, L, ci, cj, prm, hist, h, acc[s], Om[s], Omd[s], off, gdrag, F, T, Fn_out,
                           Fext, t0 + (s + 1) * dt - rem, OUT)
            for i in range(N):
                for d in range(3):
                    v[i, d] += 0.5 * h * F[i, d] / m[i]
                    w[i, d] += 0.5 * h * T[i, d] / I[i]
            if rem <= 1e-12 * dt:
                break
        stop = False
        for i in range(N):
            sp = v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1] + v[i, 2] * v[i, 2]
            if sp > vmax2[0]:
                vmax2[0] = sp
            dd = (x[i, 0] - x_build[i, 0]) ** 2 + (x[i, 1] - x_build[i, 1]) ** 2 + (x[i, 2] - x_build[i, 2]) ** 2
            if dd > lim[i] * lim[i]:
                stop = True
            de = math.sqrt((x[i, 0] - xs[i, 0]) ** 2 + (x[i, 1] - xs[i, 1]) ** 2 + (x[i, 2] - xs[i, 2]) ** 2)
            if de > dmax[i]:
                dmax[i] = de
        if stop:
            return s + 1
    return nsteps


# ---------------------------------------------------------------------------- python driver
class DEMSystem:
    """Grains in a box (faces at 0 and L) with full contact mechanics. Material inputs per grain:
    E, nu, gamma (surface energy), e_dry, p_y, mu (friction coefficient with the wall / other grains), and
    adhesion and rolling-hysteresis factors. The wall is one material (E_w, nu_w, gamma_w, e_dry_w, p_y_w)."""

    def __init__(self, x, v, w, rad, rho, E, nu, gamma, e_dry, p_y, L, wall, mu_pp=0.6, mu_pw=0.4, h_roll=0.5,
                 q_roll=30.0, q_twist=30.0, damp_mode=DAMP_TSUJI_JKR, break_mode=BREAK_DISPLACEMENT, x_gt=1.0,
                 eta_visc=0.0, adh=None, wall_adh=1.0, hfac=None, skin=None, walls=True, wall_faces=(0, 1, 2, 3, 4, 5),
                 events=True):
        self.x = np.ascontiguousarray(x, float).copy()
        self.v = np.ascontiguousarray(v, float).copy()
        self.w = np.ascontiguousarray(w, float).copy()
        N = self.N = len(self.x)
        b = lambda a: np.broadcast_to(np.asarray(a, float), (N,)).copy()
        self.rad, self.rho, self.E, self.nu, self.gamma = b(rad), b(rho), b(E), b(nu), b(gamma)
        self.e_dry, self.p_y = b(e_dry), b(p_y)
        self.m = 4.0 / 3.0 * math.pi * self.rad ** 3 * self.rho
        self.I = 0.4 * self.m * self.rad ** 2
        self.L = np.asarray(L, float)
        self.wall = dict(wall)
        self.mu_pp, self.mu_pw, self.h_roll = float(mu_pp), float(mu_pw), float(h_roll)
        self.q_roll, self.q_twist = float(q_roll), float(q_twist)
        self.damp_mode, self.break_mode, self.x_gt, self.eta_visc = int(damp_mode), int(break_mode), float(x_gt), \
            float(eta_visc)
        self.adh = b(1.0 if adh is None else adh)
        self.hfac = b(1.0 if hfac is None else hfac)
        self.wall_adh = float(wall_adh)
        self.walls = walls
        self.wall_faces = tuple(wall_faces)
        self.events = bool(events)
        self.skin = float(skin if skin is not None else 0.2 * self.rad.max())
        self.gdrag = np.zeros(N)
        self.Fext = np.zeros((N, 3))
        self.off = np.zeros(3)
        self.F = np.zeros((N, 3))
        self.T = np.zeros((N, 3))
        self.ci = np.zeros(0, np.int64)
        self.cj = np.zeros(0, np.int64)
        self.prm = np.zeros((0, NP))
        self.hist = np.zeros((0, NH))
        self.Fn = np.zeros(0)
        self._keys = {}
        self.t = 0.0
        self.rebuild()
        self._first = True

    def _params(self, i, j):
        if j >= 0:
            wk = 2.0 * math.sqrt(self.gamma[i] * self.gamma[j]) * math.sqrt(self.adh[i] * self.adh[j])
            return make_params(self.rad[i], self.rad[j], self.m[i], self.m[j], self.E[i], self.nu[i], self.E[j],
                               self.nu[j], wk, math.sqrt(self.e_dry[i] * self.e_dry[j]),
                               min(self.p_y[i], self.p_y[j]), self.mu_pp,
                               self.h_roll * math.sqrt(self.hfac[i] * self.hfac[j]), self.q_roll, self.q_twist,
                               self.damp_mode, self.break_mode, self.x_gt, self.eta_visc)
        W = self.wall
        wk = 2.0 * math.sqrt(self.gamma[i] * W["gamma"]) * self.adh[i] * self.wall_adh
        return make_params(self.rad[i], 0.0, self.m[i], math.inf, self.E[i], self.nu[i], W["E"], W["nu"], wk,
                           math.sqrt(self.e_dry[i] * W["e_dry"]), min(self.p_y[i], W.get("p_y", math.inf)),
                           self.mu_pw, self.h_roll * self.hfac[i], self.q_roll, self.q_twist, self.damp_mode,
                           self.break_mode, self.x_gt, self.eta_visc)

    def rebuild(self):
        """New contact candidate list (pairs and walls within the skin); histories of existing contacts are kept."""
        from scipy.spatial import cKDTree
        x, r = self.x, self.rad
        cand = []
        if self.N > 1:
            pairs = cKDTree(x).query_pairs(2 * r.max() + self.skin, output_type="ndarray")
            if len(pairs):
                d = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
                ok = d < r[pairs[:, 0]] + r[pairs[:, 1]] + self.skin
                cand += [(int(a), int(b)) for a, b in pairs[ok]]
        if self.walls:
            for f in self.wall_faces:
                ax = f // 2
                gap = x[:, ax] if f % 2 == 0 else self.L[ax] - x[:, ax]
                for i in np.flatnonzero(gap < r + self.skin):
                    cand.append((int(i), -1 - f))
        old_keys, old_p, old_h = self._keys, self.prm, self.hist
        K = len(cand)
        ci = np.empty(K, np.int64)
        cj = np.empty(K, np.int64)
        prm = np.empty((K, NP))
        hist = np.zeros((K, NH))
        keys = {}
        for k, (i, j) in enumerate(cand):
            ci[k], cj[k] = i, j
            keys[(i, j)] = k
            o = old_keys.get((i, j))
            if o is not None:
                prm[k] = old_p[o]
                hist[k] = old_h[o]
            else:
                prm[k] = self._params(i, j)
        self.ci, self.cj, self.prm, self.hist, self._keys = ci, cj, prm, hist, keys
        self.Fn = np.zeros(K)
        self.x_build = self.x.copy()

    def contacts(self):
        """(i, j, normal force) of the contacts that touch now (j < 0: wall face -1-j)."""
        on = self.hist[:, H_TOUCH] > 0.5
        return self.ci[on], self.cj[on], self.Fn[on]

    def run(self, dt, nsteps, acc=None, Om=None, Omd=None):
        """Advance nsteps of dt. acc/Om/Omd: None (zero), a 3-vector (constant) or an (nsteps, 3) array."""
        def series(a):
            if a is None:
                return np.zeros((nsteps, 3))
            a = np.asarray(a, float)
            return np.broadcast_to(a, (nsteps, 3)).copy() if a.ndim == 1 else np.ascontiguousarray(a)
        A, O, Od = series(acc), series(Om), series(Omd)
        if self._first:
            compute_forces(self.x, self.v, self.w, self.m, self.I, self.L, self.ci, self.cj, self.prm, self.hist,
                           0.0, A[0], O[0], Od[0], self.off, self.gdrag, self.F, self.T, self.Fn, self.Fext, self.t)
            self._first = False
        done = 0
        while done < nsteps:
            k = run_steps(self.x, self.v, self.w, self.m, self.I, self.L, self.ci, self.cj, self.prm, self.hist, dt,
                          nsteps - done, A[done:], O[done:], Od[done:], self.off, self.gdrag, self.F, self.T,
                          self.x_build, np.full(len(self.x), 0.5 * self.skin), self.Fn, self.Fext, self.t,
                          self.x_build.copy(), np.zeros(len(self.x)), np.zeros(1), self.events)
            done += k
            self.t += k * dt
            if done < nsteps:
                self.rebuild()
        return self

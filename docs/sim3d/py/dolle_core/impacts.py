"""One impact in 3-D: two rigid clumps (or a clump and a wall) touch at a point.  Stick or bounce?

The contact lasts about a microsecond (collide.py), far shorter than anything else in the tubule, so the impact is
treated as an instant: the 1-D contact model of collide.py decides the outcome along the line of centres, and the
two clumps' velocities jump.  Symbols: n the unit normal from grain i to grain j (or into the wall), cpt the
contact point, rA = cpt - X_A and rB = cpt - X_B the lever arms, v_rel = (V_A + omega_A x rA) - (V_B + omega_B x
rB) the relative velocity of the two surfaces at cpt, v_n = v_rel . n > 0 the approach speed.

1. Collision matrix (Stronge 2000, "Impact Mechanics", ch. 5).  An impulse P [N s] on clump B at lever arm rB
   (and -P on A) changes the contact-point relative velocity by -K P with
       K = K_A + K_B,   K_b = (1/M_b) 1 - [r_b]x Iinv_b [r_b]x       ([r]x: the cross-product matrix of r)
   A wall-bound clump does not move: K_b = 0.  The effective mass along the normal is m_eff = 1/(n . K n): for two
   single grains m1 m2/(m1 + m2) = m*; for an off-centre hit on a clump it is smaller (part of the push spins it).
2. Normal outcome: normal_impact runs collide.py's head-on integration for the RELATIVE motion with mass m_eff.
   That is exactly what the full simulator does (contact.py JKRContact.resolve(v_n, m_eff)), so stick/bounce and
   the normal restitution e_n are the 1-D model's (tests/test_dolle_core_sim3d.py checks it to 1e-12).
3. Stick: the clumps merge (clumps.merge) or the clump freezes on the wall.  Bounce: the impulse P is chosen so the
   normal relative velocity after the impact is -e_n v_n, and the tangential one -e_t v_t (e_t = 0: the surfaces
   stop slipping) if that stays inside the Coulomb friction cone |P_t| <= mu P_n; otherwise the surfaces slide
   and |P_t| = mu P_n.  If friction would ADD kinetic energy (possible for rigid bodies, "Kane's paradox";
   change of kinetic energy dKE = -P . v_rel + 1/2 P . K P), the frictionless impulse is used instead, which always
   loses energy.  (v0.5 dolle_sim/simulation.py _impulse, line for line.)
4. Fragmentation (Dominik & Tielens 1997, ApJ 480, 647): after a bounce, a clump hit with translational energy
   E_coll above 3 N_c E_break loses the struck grain (erosion), above 10 N_c E_break it shatters into single
   grains (N_c = N - 1 for a clump of N grains; E_break = 1.54 F_c |delta_c|, Chokshi et al. 1993, ApJ 407, 806).
   Those thresholds were derived for large aggregates in which rolling is easy; here they are EXTRAPOLATED to clumps
   of 2-3 grains, for which the rolling energy is 25x (graphite) to 61x (silica) E_break, the opposite regime.
   The broken bonds (the clump's real ones, Clumps.bonds) can only be paid from the energy the bounce removed,
   E_diss.  What happens to the rest, E_diss - cost, is not known: the 1-D contact model turned it into heat.
   ASSUMED (as in v0.5): a fraction Tubule.separation of it (default 1, all of it) comes back as the kinetic
   energy with which the fragments fly apart; separation = 0 is the opposite bound (it all stays heat, and the
   fragments keep the rigid-body velocities they had at the break).  notebook 06, section 7, gives the size of
   the difference.  (v0.5 _fragment, _separate.)

Momentum: A gets -P and B gets +P at the SAME point cpt, so the total linear momentum and the total angular
momentum about any point are unchanged (notebook 06, section 4).
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from .clumps import Clumps, angular_velocity, cross, detach, merge, freeze, n_bonds, split_disconnected
from .collide import CollisionResult, collide_pair
from .jkr import break_radius, pull_off_force
from .materials import friction_coefficient
from .pair import Pair, make_pair


def skew(r) -> np.ndarray:
    """[r]x, the matrix with [r]x a = r x a."""
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def collision_matrix(c: Clumps, b: int, r) -> np.ndarray:
    """K_b = (1/M_b) 1 - [r]x Iinv_b [r]x [1/kg]; zero for a wall-bound clump.  (v0.5 _K_body.)"""
    if c.frozen[b]:
        return np.zeros((3, 3))
    S = skew(r)
    return np.eye(3) / c.M[b] - S @ c.Iinv[b] @ S


def apply_impulse(c: Clumps, b: int, r, P) -> None:
    """Impulse P [N s] on clump b at lever arm r: V += P/M, L += r x P (nothing for a wall-bound clump)."""
    if c.frozen[b]:
        return
    c.V[b] += P / c.M[b]
    c.L[b] += cross(r, P)


def normal_impact(pair: Pair, v_n: float, m_eff: float, n_sub: int = 300) -> CollisionResult:
    """Head-on JKR impact of the relative motion: approach speed v_n [m/s], mass m_eff [kg], the contact laws of
    `pair` (R*, E*, w, e_dry, p_y).  Runs collide.py's collide_pair for a body of mass m_eff hitting an immovable
    partner (m2 = inf): its velocity IS the relative velocity, d(v)/dt = -F/m_eff, the equation contact.py
    _resolve_core(.., v_in, m_eff) integrates."""
    eff = replace(pair, m1=m_eff, m2=math.inf, m_star=m_eff)
    return collide_pair(eff, u1=v_n, u2=0.0, n_sub=n_sub, record=False)


def bounce_impulse(K, v_rel, n, e_n: float, mu: float, e_t: float = 0.0) -> tuple[np.ndarray, bool]:
    """Impulse P [N s] on the second body (the first gets -P) for a bounce, and whether the energy guard was used.
    Wanted: contact-point velocity after = -e_n v_n n - e_t v_t, so P = K^-1 (v_rel - target).  Outside the Coulomb
    cone: P = (1 + e_n) v_n / (n . K d) d with d = n + mu t_hat (sliding).  Energy guard: if dKE = -P.v_rel +
    1/2 P.K.P > 0, use the frictionless P = (1 + e_n) v_n n / (n . K n).  (v0.5 _impulse.)"""
    vn = float(v_rel @ n)
    vt = v_rel - vn * n
    target = -e_n * vn * n - e_t * vt
    try:
        P = np.linalg.solve(K, v_rel - target)
    except np.linalg.LinAlgError:
        P = np.linalg.lstsq(K, v_rel - target, rcond=None)[0]
    Pn = float(P @ n)
    Pt = P - Pn * n
    if not (np.linalg.norm(Pt) <= mu * Pn or float(np.linalg.norm(vt)) < 1e-15):
        tv = vt / np.linalg.norm(vt)
        d = n + mu * tv
        P = (1 + e_n) * vn / float(n @ K @ d) * d
    if float(-P @ v_rel + 0.5 * P @ K @ P) > 0:
        return (1 + e_n) * vn / float(n @ K @ n) * n, True
    return P, False


def break_energy(pair: Pair) -> float:
    """E_break = 1.54 F_c |delta_c| [J], the energy to break one contact (Chokshi, Tielens & Hollenbach 1993),
    with F_c = 1.5 pi w R* and delta_c the overlap at break.  Same arithmetic as dolle_sim materials.pair_props."""
    R, w, E = pair.R_star, pair.w, pair.E_star
    a_c = break_radius(R, w, E, "displacement")
    delta_c = a_c ** 2 / R - math.sqrt(2 * math.pi * w * a_c / E)
    return 1.54 * pull_off_force(R, w) * abs(delta_c)


def grain_pair(sim, i: int, j: int | None) -> Pair:
    """The Pair of grain i with grain j (j None: the wall), with the EXACT radii and adhesion factors:
    f = sqrt(f_i f_j) for two grains, f_i x f_wall on the wall (v0.5 _contact, without its rounding)."""
    key = (i, j)
    p = sim.pairs.get(key)
    if p is None:
        c = sim.clumps
        if j is None:
            p = make_pair(c.material[i], c.r[i], sim.tubule.wall, None, f_adh=c.adhesion[i] * sim.tubule.f_wall)
        else:
            p = make_pair(c.material[i], c.r[i], c.material[j], c.r[j], f_adh=math.sqrt(c.adhesion[i] * c.adhesion[j]))
        sim.pairs[key] = p
    return p


def friction(sim, i: int, j: int | None) -> float:
    """Sliding friction coefficient of the pair: Tubule.mu if set, else materials.py FRICTION."""
    if sim.tubule.mu is not None:
        return sim.tubule.mu
    c = sim.clumps
    return friction_coefficient(c.material[i], "wall" if j is None else c.material[j])


def _record(sim, kind, i, j, A, B, res, vn, vrel, n, m_eff, cpt, E_coll) -> dict:
    """The event record (v0.5's collision fields, SI units)."""
    c = sim.clumps
    return dict(t=sim.t_now, kind=kind, i=int(i), j=int(j), mat_i=c.material[i],
                mat_j=sim.tubule.wall if kind == "wall" else c.material[j], n_i=int(c.n[A]),
                n_j=0 if B is None else int(c.n[B]), frozen_j=True if B is None else bool(c.frozen[B]),
                outcome="stick" if res.stuck else "bounce", v_n=vn, v_t=float(np.linalg.norm(vrel - vn * n)),
                e_n=res.e, m_eff=m_eff, yielded=res.yielded, max_force=res.max_force,
                x=float(cpt[0]), y=float(cpt[1]), z=float(cpt[2]), E_coll=E_coll, E_diss=0.0, fragments=0,
                mode="", guard=False)


def grain_grain(sim, i: int, j: int) -> dict | None:
    """Resolve the impact of grain i (clump A) on grain j (clump B); None if they do not approach.
    (v0.5 _resolve_pp, without wall vibration, charging and the resting rule.)"""
    c = sim.clumps
    A, B = int(c.body[i]), int(c.body[j])
    rvec = c.pos[j] - c.pos[i]
    n = rvec / np.linalg.norm(rvec)
    cpt = c.pos[i] + n * c.r[i]
    rA, rB = cpt - c.X[A], cpt - c.X[B]
    fzA, fzB = bool(c.frozen[A]), bool(c.frozen[B])
    vA = np.zeros(3) if fzA else c.V[A] + cross(angular_velocity(c, A), rA)
    vB = np.zeros(3) if fzB else c.V[B] + cross(angular_velocity(c, B), rB)
    vrel = vA - vB
    vn = float(vrel @ n)
    if vn <= 0:
        return None
    K = collision_matrix(c, A, rA) + collision_matrix(c, B, rB)
    m_eff = 1.0 / float(n @ K @ n)
    res = normal_impact(grain_pair(sim, i, j), vn, m_eff, sim.tubule.n_sub)
    dV = (vA if fzA else c.V[A]) - (vB if fzB else c.V[B])          # translational energy of the two clumps
    mu_AB = c.M[B] if fzA else c.M[A] if fzB else c.M[A] * c.M[B] / (c.M[A] + c.M[B])
    rec = _record(sim, "pp", i, j, A, B, res, vn, vrel, n, m_eff, cpt, 0.5 * mu_AB * float(dV @ dV))
    rec["gap"] = float(np.linalg.norm(rvec)) - c.r[i] - c.r[j]       # ~0: how exactly the contact was found
    nA, nB = int(c.n[A]), int(c.n[B])
    if res.stuck:
        sim.counts["stick"] += 1
        if fzA or fzB:                                               # the free one hands its momentum over
            P, L = _momentum_of(c, B if fzA else A)
            sim.tubule_P += P
            sim.tubule_L += L
        merge(c, A, B, bond=(i, j))
        return rec
    sim.counts["bounce"] += 1
    P, rec["guard"] = bounce_impulse(K, vrel, n, res.e, friction(sim, i, j), sim.tubule.e_t)
    sim.counts["guard"] += rec["guard"]
    apply_impulse(c, A, rA, -P)
    apply_impulse(c, B, rB, P)
    for b, imp in ((A, -P), (B, P)):                                 # a wall-bound partner passes it to the tubule
        if c.frozen[b]:
            sim.tubule_P += imp
            sim.tubule_L += cross(cpt, imp)
    rec["E_diss"] = E_diss = float(P @ vrel - 0.5 * P @ K @ P)
    if sim.tubule.fragmentation and (nA > 1 or nB > 1):
        rec["fragments"], rec["mode"] = fragment(sim, [(A, i), (B, j)], rec["E_coll"], E_diss)
    return rec


def grain_wall(sim, g: int, axis: int, side: float) -> dict | None:
    """Resolve the impact of grain g (clump A) on the wall normal to `axis` at side -1 (x = 0) or +1 (x = size);
    None if it does not approach.  (v0.5 _resolve_wall, static walls.)"""
    c = sim.clumps
    A = int(c.body[g])
    n = np.zeros(3)
    n[axis] = side                                                   # from the grain into the wall
    cpt = c.pos[g] + n * c.r[g]
    rA = cpt - c.X[A]
    vA = c.V[A] + cross(angular_velocity(c, A), rA)
    vn = float(vA @ n)
    if vn <= 0:
        return None
    K = collision_matrix(c, A, rA)
    m_eff = 1.0 / float(n @ K @ n)
    res = normal_impact(grain_pair(sim, g, None), vn, m_eff, sim.tubule.n_sub)
    rec = _record(sim, "wall", g, -1, A, None, res, vn, vA, n, m_eff, cpt, 0.5 * c.M[A] * float(c.V[A] @ c.V[A]))
    x = c.pos[g, axis]
    rec["gap"] = float((x if side < 0 else sim.tubule.size[axis] - x) - c.r[g])
    if res.stuck:
        sim.counts["wall_stick"] += 1
        P, L = freeze(c, A, axis, side)
        sim.tubule_P += P
        sim.tubule_L += L
        return rec
    sim.counts["wall_bounce"] += 1
    P, rec["guard"] = bounce_impulse(K, vA, n, res.e, friction(sim, g, None), sim.tubule.e_t)
    sim.counts["guard"] += rec["guard"]
    apply_impulse(c, A, rA, -P)
    sim.tubule_P += P
    sim.tubule_L += cross(cpt, P)
    rec["E_diss"] = E_diss = float(P @ vA - 0.5 * P @ K @ P)
    if sim.tubule.fragmentation and c.n[A] > 1:
        rec["fragments"], rec["mode"] = fragment(sim, [(A, g)], rec["E_coll"], E_diss)
    return rec


def _momentum_of(c: Clumps, b: int) -> tuple[np.ndarray, np.ndarray]:
    """(M V, L + X x M V) of clump b."""
    P = c.M[b] * c.V[b]
    return P, c.L[b] + cross(c.X[b], P)


def _bond_energy(sim, b: int, k: int) -> float:
    """E_break of the bond holding grain k in clump b: its pair with the nearest grain it is bonded to
    (v0.5 _bond_energy)."""
    c = sim.clumps
    mem = [x for x in c.members[b] if x != k and (min(x, k), max(x, k)) in c.bonds[b]] or \
        [x for x in c.members[b] if x != k]
    d = np.linalg.norm(c.pos[mem] - c.pos[k], axis=1)
    return break_energy(grain_pair(sim, k, mem[int(np.argmin(d))]))


def fragment(sim, hits, E_coll: float, E_diss: float) -> tuple[int, str]:
    """Erosion or shattering of the free multi-grain clumps among hits [(clump, struck grain)] after a bounce
    (module docstring, point 4).  Returns (number of grains detached, "erosion" | "shatter" | "").  Energy budget:
    the first option whose bonds (n_contacts E_b) E_diss can pay is taken; the fragments separate with the
    fraction Tubule.separation of E_diss - cost (ASSUMED, default 1), so kinetic + bond energy never rises.
    (v0.5 _fragment.)"""
    c = sim.clumps
    agg = [(b, k) for b, k in hits if c.n[b] > 1 and not c.frozen[b]]
    if not agg:
        return 0, ""
    N_c = sum(int(c.n[b]) - 1 for b, _ in agg)
    E_b = float(np.mean([_bond_energy(sim, b, k) for b, k in agg]))
    options = []
    if E_coll > 10 * N_c * E_b:
        options.append(("shatter", {b: [[x] for x in c.members[b]][1:] for b, _ in agg},
                        sum(n_bonds(c, b) for b, _ in agg)))
    if E_coll > 3 * N_c * E_b:
        options.append(("erosion", {b: [[k]] for b, k in agg}, sum(n_bonds(c, b, k) for b, k in agg)))
    if not options:
        return 0, ""
    for mode, groups_of, n_contacts in options:
        cost = n_contacts * E_b
        if E_diss >= cost:
            break
    else:
        sim.counts["fragmentation_energy_limited"] += 1
        return 0, ""
    if mode != options[0][0]:
        sim.counts["fragmentation_energy_limited"] += 1
    sim.E_bonds_broken += cost
    out = 0
    for b, groups in groups_of.items():
        X0 = c.X[b].copy()
        new = detach(c, b, groups)
        new += split_disconnected(c, b)
        out += sum(len(g) for g in groups)
        separate(c, [b] + new, X0, sim.tubule.separation * (E_diss - cost) / len(groups_of))
    sim.counts[mode] += 1
    return out, mode


def separate(c: Clumps, parts, X0, E_sep: float) -> None:
    """Push the fragments `parts` apart radially from the old centre of mass X0 so that their kinetic energy rises
    by exactly E_sep [J]: V_p += s u_p with u_p the unit vector from X0 minus its mass-weighted mean (so sum m u = 0:
    momentum kept; the kicks are radial about the common centre of mass: angular momentum kept).  The parts already
    move, so dKE = s B + s^2 A with A = sum M u^2/2, B = sum M V.u, and s is the positive root.  (v0.5 _separate.)"""
    if E_sep <= 0 or len(parts) < 2:
        return
    m = c.M[parts]
    u = c.X[parts] - X0
    nrm = np.linalg.norm(u, axis=1)
    nrm[nrm == 0] = 1.0
    u = u / nrm[:, None]
    u -= (m[:, None] * u).sum(0) / m.sum()
    A = 0.5 * float((m * (u * u).sum(1)).sum())
    if A <= 0:
        return
    B = float((m * (c.V[parts] * u).sum(1)).sum())
    s = (-B + math.sqrt(B * B + 4 * A * E_sep)) / (2 * A)
    c.V[parts] += s * u

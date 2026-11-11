"""Rigid clumps: grains that have stuck together and now move as one rigid body.

Every grain k (a sphere: radius r_k [m], mass m_k [kg], own moment of inertia I0_k = (2/5) m_k r_k^2 [kg m^2])
belongs to exactly one clump b = body[k].  A clump is a single grain, or several grains that stuck.  Its state is

    M_b   mass [kg] = sum of its grains' masses
    X_b   centre of mass [m],  V_b its velocity [m/s]
    L_b   spin angular momentum about X_b [kg m^2/s]
    Q_b   orientation: the 3x3 rotation from the clump's principal axes ("body frame") to the tubule frame
    Ip_b  the three principal moments of inertia [kg m^2]; Iinv_b the inverse inertia tensor in the tubule frame
    omega_b = Iinv_b L_b   its angular velocity [rad/s]

A member grain sits at pos_k = X_b + off_k, with off_k = Q_b offb_k: its offset is FIXED in the body frame (offb),
which is what "rigid" means.  Its velocity is V_b + omega_b x off_k.

Clump ids.  The arrays have one slot per grain; a clump keeps the id (slot) of its larger part when two clumps
merge, and a detached part takes the lowest unused slot.  This is the rule of the full simulator's v0.5 chamber
engine (dolle_sim/simulation.py ChamberSim._merge, _detach), so ids, and therefore colours in the replays, agree.

Bonds.  Each clump keeps the list of its grain-grain bonds, bonds[b] = [(i, j), ...]: the pairs that stuck.  A
clump is held together only by them, so erosion pays for the bonds the eroded grain really has, and a clump splits
where its bonds are cut (bond_groups).  The bonds are stored rather than recomputed from distances because a grain
that sticks to a SPINNING clump is merged where the straight-line contact search put it, which can be a few
nanometres from touching (at most rho theta^2/2 = 1.25e-3 rho, rho the member's distance from the centre of mass;
notebook 06, section 3): a distance test with a tight tolerance would miss those bonds.

Momentum.  The total linear momentum of the free clumps is P = sum M_b V_b, the total angular momentum about the
tubule origin is L = sum (L_b + X_b x M_b V_b).  merge() and detach() conserve both exactly (to rounding error)
by construction: notebook 06, section 5, derives the parallel-axis identity they use.

Every function says which v0.5 function it mirrors; the arithmetic is the same, in the same order, so the two
engines agree to rounding error (tests/test_dolle_core_sim3d.py).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .materials import get_material, sphere_mass


def cross(a, b) -> np.ndarray:
    """Cross product of two 3-vectors (faster than np.cross for single vectors; v0.5 _cross3)."""
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


def cross_rows(a, b) -> np.ndarray:
    """Row-wise cross product of (N, 3) arrays (v0.5 _crossN)."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=-1)


@dataclass
class Clumps:
    """Grain arrays (index k): r, m, I0, material (names), adhesion (factor per grain), pos, off, offb, body.
    Clump arrays (index b, one slot per grain): X, V, L, M, Q, Ip, Iinv, members (lists of grain ids), bonds
    (lists of the bonded grain pairs (i, j), i < j), alive, frozen (stuck to a wall: does not move), wall_axis /
    wall_side (which wall: axis 0-2, side -1 or +1), n (number of grains).  Units SI."""
    r: np.ndarray
    m: np.ndarray
    I0: np.ndarray
    material: list
    adhesion: np.ndarray
    pos: np.ndarray
    off: np.ndarray
    offb: np.ndarray
    body: np.ndarray
    X: np.ndarray
    V: np.ndarray
    L: np.ndarray
    M: np.ndarray
    Q: np.ndarray
    Ip: np.ndarray
    Iinv: np.ndarray
    members: list
    bonds: list
    alive: np.ndarray
    frozen: np.ndarray
    wall_axis: np.ndarray
    wall_side: np.ndarray
    n: np.ndarray


def make_clumps(materials, radii, positions, velocities, spins=None, adhesion=None) -> Clumps:
    """One clump per grain.  materials: a name per grain (or one name for all); radii [m]; positions, velocities
    (N x 3) [m, m/s]; spins: angular velocities (N x 3) [rad/s], default 0; adhesion: factor per grain on the
    work of adhesion, default 1.  (v0.5 ChamberSim._init_particles, "bodies: initially one per monomer".)"""
    r = np.array(radii, dtype=float)
    N = len(r)
    mats = [materials] * N if isinstance(materials, str) else list(materials)
    m = np.array([sphere_mass(get_material(mats[k]).rho, r[k]) for k in range(N)])
    I0 = 0.4 * m * r ** 2
    pos = np.array(positions, dtype=float).reshape(N, 3)
    w = np.zeros((N, 3)) if spins is None else np.array(spins, dtype=float).reshape(N, 3)
    Iinv = np.zeros((N, 3, 3))
    Iinv[:, [0, 1, 2], [0, 1, 2]] = (1.0 / I0)[:, None]
    return Clumps(r=r, m=m, I0=I0, material=mats,
                  adhesion=np.ones(N) if adhesion is None else np.array(adhesion, dtype=float),
                  pos=pos, off=np.zeros((N, 3)), offb=np.zeros((N, 3)), body=np.arange(N),
                  X=pos.copy(), V=np.array(velocities, dtype=float).reshape(N, 3).copy(), L=w * I0[:, None],
                  M=m.copy(), Q=np.repeat(np.eye(3)[None], N, axis=0), Ip=np.repeat(I0[:, None], 3, axis=1),
                  Iinv=Iinv, members=[[k] for k in range(N)], bonds=[[] for _ in range(N)],
                  alive=np.ones(N, bool), frozen=np.zeros(N, bool),
                  wall_axis=np.full(N, -1), wall_side=np.zeros(N), n=np.ones(N, int))


def angular_velocity(c: Clumps, b: int) -> np.ndarray:
    """omega_b = Iinv_b L_b [rad/s] (v0.5 _w)."""
    return c.Iinv[b] @ c.L[b]


def grain_velocities(c: Clumps) -> np.ndarray:
    """Velocity of every grain's centre: V_b + omega_b x off_k for members of multi-grain clumps, V_b otherwise (a
    single sphere's spin does not move its centre).  (v0.5 _omega and _mono_vel.)"""
    vm = c.V[c.body]
    multi = np.flatnonzero(c.alive & ~c.frozen & (c.n > 1))
    if len(multi):
        W = np.zeros((len(c.r), 3))
        W[multi] = np.einsum("bij,bj->bi", c.Iinv[multi], c.L[multi])
        mm = np.flatnonzero(c.n[c.body] > 1)
        vm[mm] += cross_rows(W[c.body[mm]], c.off[mm])
    return vm


def inertia_tensor(c: Clumps, members, centre) -> np.ndarray:
    """I = sum_k [(I0_k + m_k |d_k|^2) 1 - m_k d_k d_k^T], d_k = pos_k - centre [kg m^2]: each sphere's own
    inertia plus the parallel-axis term (Steiner's theorem).  (v0.5 _inertia.)"""
    I = np.zeros((3, 3))
    for k in members:
        d = c.pos[k] - centre
        I += (c.I0[k] + c.m[k] * (d @ d)) * np.eye(3) - c.m[k] * np.outer(d, d)
    return I


def set_principal(c: Clumps, b: int, I) -> None:
    """Store the inertia tensor I (tubule frame) of clump b as principal moments Ip and axes Q (eigenvectors,
    made right-handed), and its members' body-frame offsets offb = Q^T off.  (v0.5 _set_principal.)"""
    vals, vecs = np.linalg.eigh(I)
    if np.linalg.det(vecs) < 0:
        vecs[:, 2] *= -1
    c.Q[b], c.Ip[b] = vecs, vals
    c.Iinv[b] = vecs @ np.diag(1.0 / vals) @ vecs.T
    mem = c.members[b]
    c.offb[mem] = c.off[mem] @ vecs


def rotate(c: Clumps, idx, h: float) -> None:
    """Free rotation of clumps idx over a time h [s] (no torque acts): the Dullweber-Leimkuhler-McLachlan splitting
    (1997, J. Chem. Phys. 107, 5840).  In the body frame the angular momentum Lb = Q^T L turns about each principal
    axis a in turn by the angle tau Lb_a / Ip_a, in the order x (h/2), y (h/2), z (h), y (h/2), x (h/2); the
    columns of Q turn with it.  Each sub-step is an exact rotation, so |L| and the world vector L = Q Lb are kept to
    rounding error, and the energy error stays bounded (notebook 06, section 6).  (v0.5 _rotate.)"""
    Q = c.Q[idx].copy()
    Ip = c.Ip[idx]
    Lb = np.einsum("kji,kj->ki", Q, c.L[idx])          # Q^T L
    for a, tau in ((0, 0.5 * h), (1, 0.5 * h), (2, h), (1, 0.5 * h), (0, 0.5 * h)):
        th = tau * Lb[:, a] / Ip[:, a]
        cs, sn = np.cos(th), np.sin(th)
        b1, b2 = (a + 1) % 3, (a + 2) % 3
        l1, l2 = Lb[:, b1].copy(), Lb[:, b2].copy()
        Lb[:, b1] = cs * l1 + sn * l2
        Lb[:, b2] = -sn * l1 + cs * l2
        q1, q2 = Q[:, :, b1].copy(), Q[:, :, b2].copy()
        Q[:, :, b1] = cs[:, None] * q1 + sn[:, None] * q2
        Q[:, :, b2] = -sn[:, None] * q1 + cs[:, None] * q2
    c.Q[idx] = Q
    c.L[idx] = np.einsum("kij,kj->ki", Q, Lb)
    c.Iinv[idx] = Q @ (np.eye(3)[None] / Ip[:, None, :]) @ np.transpose(Q, (0, 2, 1))
    local = np.full(len(c.r), -1)
    local[idx] = np.arange(len(idx))
    mm = np.flatnonzero(local[c.body] >= 0)
    c.off[mm] = np.einsum("kij,kj->ki", Q[local[c.body[mm]]], c.offb[mm])


def drift(c: Clumps, h: float) -> None:
    """Free flight over h [s]: every free clump moves in a straight line, X += V h (the acceleration is applied by
    the kicks, sim3d.py), and multi-grain clumps turn (rotate).  Then every grain is placed at X + off.
    (v0.5 _drift without the payload roll.)"""
    if h <= 0:
        return
    act = np.flatnonzero(c.alive & ~c.frozen)
    c.X[act] += c.V[act] * h
    multi = act[c.n[act] > 1]
    if len(multi):
        rotate(c, multi, h)
    c.pos = c.X[c.body] + c.off


def merge(c: Clumps, A: int, B: int, bond=None) -> int:
    """Two clumps stick: they become ONE rigid clump (returns its id, the larger part's).  bond = (i, j), the two
    grains that stuck (impacts.grain_grain); without it (clumps built by hand, from grains placed touching) the
    touching pairs between A and B become the bonds.  With M = M_A + M_B:

        X = (M_A X_A + M_B X_B)/M,   V = (M_A V_A + M_B V_B)/M                         (linear momentum kept)
        L = L_A + L_B + M_A (X_A - X) x (V_A - V) + M_B (X_B - X) x (V_B - V)          (angular momentum kept)
        I = sum over A, B of [I_b + M_b (|d_b|^2 1 - d_b d_b^T)],  d_b = X_b - X       (parallel axes)

    If either clump is stuck to a wall, the new clump is stuck there too: V = L = 0 (the wall takes the momentum;
    sim3d.py books it).  (v0.5 _merge.)"""
    if c.n[B] > c.n[A]:
        A, B = B, A
    MA, MB = c.M[A], c.M[B]
    M = MA + MB
    X = (MA * c.X[A] + MB * c.X[B]) / M
    V = (MA * c.V[A] + MB * c.V[B]) / M
    L = c.L[A] + c.L[B] + MA * cross(c.X[A] - X, c.V[A] - V) + MB * cross(c.X[B] - X, c.V[B] - V)
    I = np.zeros((3, 3))
    for b in (A, B):
        d = c.X[b] - X
        Ib = c.Q[b] @ np.diag(c.Ip[b]) @ c.Q[b].T
        I += Ib + c.M[b] * ((d @ d) * np.eye(3) - np.outer(d, d))
    fz = bool(c.frozen[A] or c.frozen[B])
    if fz:
        src = A if c.frozen[A] else B
        c.wall_axis[A], c.wall_side[A] = c.wall_axis[src], c.wall_side[src]
    if bond is None:
        new = [(min(a, k), max(a, k)) for a in c.members[A] for k in c.members[B] if touching(c, a, k)]
    else:
        new = [(min(bond), max(bond))]
    c.bonds[A], c.bonds[B] = c.bonds[A] + c.bonds[B] + new, []
    mem = c.members[A] + c.members[B]
    c.members[A], c.members[B] = mem, []
    c.body[mem] = A
    c.X[A], c.M[A] = X, M
    c.V[A], c.L[A] = (np.zeros(3), np.zeros(3)) if fz else (V, L)
    c.off[mem] = c.pos[mem] - X
    c.n[A] += c.n[B]
    set_principal(c, A, I)
    c.frozen[A] = fz
    c.alive[B] = False
    c.n[B] = 0
    c.wall_axis[B] = -1
    return int(A)


def freeze(c: Clumps, b: int, axis: int, side: float) -> tuple[np.ndarray, np.ndarray]:
    """Clump b sticks to the wall (axis, side): it stops, V = L = 0.  Returns the momentum (M V) and the angular
    momentum about the origin (L + X x M V) it hands to the tubule.  (v0.5 _resolve_wall, stick branch.)"""
    P = c.M[b] * c.V[b]
    Lo = c.L[b] + cross(c.X[b], P)
    c.frozen[b] = True
    c.wall_axis[b], c.wall_side[b] = axis, side
    c.V[b] = 0
    c.L[b] = 0
    return P, Lo


def touching(c: Clumps, a: int, k: int) -> bool:
    """Do grains a and k touch (centre distance < (r_a + r_k)(1 + 1e-6))?  Used only to find the bonds of clumps
    built by hand (merge without a bond); the simulation itself uses the stored bonds."""
    return bool(np.linalg.norm(c.pos[a] - c.pos[k]) < (c.r[a] + c.r[k]) * (1 + 1e-6))


def n_bonds(c: Clumps, b: int, k=None) -> int:
    """Number of bonds of clump b (or only those of its grain k).  (v0.5 _contacts_of.)"""
    return sum(1 for bd in c.bonds[b] if k is None or k in bd)


def bond_groups(c: Clumps, b: int) -> list[list[int]]:
    """Split the members of clump b into the groups its bonds hold together (connected parts).
    (v0.5 _components.)"""
    adj = set(c.bonds[b])
    left, comps = set(c.members[b]), []
    while left:                                  # (the same walk as touching_groups, so the parts come out in the
        stack = [left.pop()]                     #  same order when the bonds are the touching pairs)
        comp = list(stack)
        while stack:
            k = stack.pop()
            for j in list(left):
                if (min(j, k), max(j, k)) in adj:
                    left.remove(j)
                    stack.append(j)
                    comp.append(j)
        comps.append(comp)
    return comps


def touching_groups(c: Clumps, members) -> list[list[int]]:
    """Split grains into groups that hold together by geometric contact (distance test, touching).  Kept for checks
    of clumps built by hand; the simulation splits clumps by their bonds (bond_groups)."""
    left, comps = set(members), []
    while left:
        stack = [left.pop()]
        comp = list(stack)
        while stack:
            k = stack.pop()
            for j in list(left):
                if touching(c, j, k):
                    left.remove(j)
                    stack.append(j)
                    comp.append(j)
        comps.append(comp)
    return comps


def detach(c: Clumps, b: int, groups) -> list[int]:
    """Detach each group of member grains of clump b as its own rigid clump, moving with b's rigid-body motion at
    this instant (grain k at V_b + omega_b x off_k).  Each group takes its own share of momentum: mass-weighted
    mean velocity and position, and its spin about its own centre; the rest of b keeps what is left of the
    parent's P and L (by subtraction, so the totals are conserved exactly).  Returns the new clump ids.
    (v0.5 _detach.)"""
    w = angular_velocity(c, b)
    P = c.M[b] * c.V[b]
    Lo = c.L[b] + c.M[b] * cross(c.X[b], c.V[b])
    free = list(np.flatnonzero(~c.alive)[:len(groups)])
    new = []
    bonds = c.bonds[b]
    for g, nb in zip(groups, free):
        g = [int(k) for k in g]
        gs = set(g)
        c.bonds[nb] = [bd for bd in bonds if bd[0] in gs and bd[1] in gs]      # bonds across parts are broken
        mg = c.m[g]
        Mg = float(mg.sum())
        vg = c.V[b][None, :] + cross_rows(w[None, :], c.off[g])
        Xg = (mg[:, None] * c.pos[g]).sum(0) / Mg
        Vg = (mg[:, None] * vg).sum(0) / Mg
        Lg = np.zeros(3)
        for i, k in enumerate(g):
            Lg += c.I0[k] * w + c.m[k] * cross(c.pos[k] - Xg, vg[i] - Vg)
        P = P - Mg * Vg
        Lo = Lo - (Lg + Mg * cross(Xg, Vg))
        for k in g:
            c.members[b].remove(k)
            c.body[k] = nb
        c.members[nb] = g
        c.alive[nb], c.frozen[nb], c.n[nb], c.wall_axis[nb] = True, False, len(g), -1
        c.X[nb], c.V[nb], c.M[nb], c.L[nb] = Xg, Vg, Mg, Lg
        c.off[g] = c.pos[g] - Xg
        if len(g) == 1:
            k = g[0]
            c.Iinv[nb] = np.eye(3) / c.I0[k]
            c.Q[nb], c.Ip[nb] = np.eye(3), c.I0[k]
            c.offb[k] = 0.0
        else:
            set_principal(c, nb, inertia_tensor(c, g, Xg))
        new.append(int(nb))
    mem = c.members[b]
    rest = set(mem)
    c.bonds[b] = [bd for bd in bonds if bd[0] in rest and bd[1] in rest]
    M = c.m[mem].sum()
    X = (c.m[mem, None] * c.pos[mem]).sum(0) / M
    c.M[b], c.X[b], c.V[b] = M, X, P / M
    c.L[b] = Lo - M * cross(X, P / M)
    c.off[mem] = c.pos[mem] - X
    c.n[b] = len(mem)
    set_principal(c, b, inertia_tensor(c, mem, X))
    return new


def split_disconnected(c: Clumps, b: int) -> list[int]:
    """After grains were removed, any part of clump b no longer bonded to the rest becomes its own clump
    (v0.5 _split_disconnected)."""
    if c.n[b] < 2:
        return []
    comps = bond_groups(c, b)
    if len(comps) < 2:
        return []
    comps.sort(key=len, reverse=True)
    return detach(c, b, comps[1:])


def momentum(c: Clumps, bodies=None) -> tuple[np.ndarray, np.ndarray]:
    """(P, L): linear momentum [kg m/s] and angular momentum about the tubule origin [kg m^2/s] of the clumps
    `bodies` (default: every free clump), L = sum (L_b + X_b x M_b V_b).  A wall-bound clump has V = L = 0."""
    idx = np.flatnonzero(c.alive & ~c.frozen) if bodies is None else np.asarray(sorted(bodies), int)
    if not len(idx):
        return np.zeros(3), np.zeros(3)
    p = c.M[idx, None] * c.V[idx]
    return p.sum(0), (c.L[idx] + cross_rows(c.X[idx], p)).sum(0)


def kinetic_energy(c: Clumps) -> float:
    """Kinetic energy of the free clumps [J]: sum 1/2 M |V|^2 + 1/2 omega . L."""
    idx = np.flatnonzero(c.alive & ~c.frozen)
    if not len(idx):
        return 0.0
    W = np.einsum("bij,bj->bi", c.Iinv[idx], c.L[idx])
    return float(0.5 * (c.M[idx] * (c.V[idx] ** 2).sum(1)).sum() + 0.5 * (W * c.L[idx]).sum())


def spin_bound(c: Clumps) -> np.ndarray:
    """Upper bound on each clump's angular velocity [rad/s], |L| / I_min (free rotation keeps |L|; |omega| <=
    |L|/I_min for any orientation).  Zero for wall-bound clumps."""
    w = np.linalg.norm(c.L, axis=1) / c.Ip.min(1)
    w[~c.alive | c.frozen] = 0.0
    return w


def speed_bound(c: Clumps) -> np.ndarray:
    """Upper bound on each grain's speed while its clump flies freely [m/s]: |V_b| + |omega_b|max |off_k|."""
    b = c.body
    s = np.linalg.norm(c.V[b], axis=1) + spin_bound(c)[b] * np.linalg.norm(c.off, axis=1)
    s[c.frozen[b]] = 0.0
    return s

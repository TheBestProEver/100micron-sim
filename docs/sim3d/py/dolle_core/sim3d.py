"""The 3-D simulation: grains and rigid clumps flying in a tubule, colliding, sticking or bouncing.

This is the full simulator's v0.5 chamber engine (dolle_sim/simulation.py ChamberSim) rewritten to be read, with
only the physics that matters in DOLL-E's microgravity window (notebook 06 derives every step):

  * Grains are spheres; grains that stuck form a rigid clump (clumps.py).  Between impacts every free clump flies
    under the uniform residual acceleration a and spins freely.
  * One time step dt is velocity Verlet: half kick V += a dt/2, drift for dt resolving every impact on the way,
    half kick V += a dt/2.  With a constant a this is exact for the centre of mass, x = x0 + v0 t + a t^2/2.
  * During the drift the exact moment two grains of different clumps touch, |d + u t| = r_i + r_j, is found by
    solving a quadratic (next_contact); the walls by the time to reach each face.  Every clump is moved to that
    moment, the impact is resolved (impacts.py), and the search starts again.  So no impact is ever skipped.
  * Members of a spinning clump move on arcs, not lines.  As in v0.5, the drift is cut short (rotation_window) so
    that a clump turns by at most theta = 0.05 rad while anything is within its reach; then the straight-line
    search is off the true arc by at most rho theta^2/2 = 1.25e-3 rho (rho: the member's distance from the centre
    of mass; notebook 06, section 3).
  * Linear and angular momentum: every impact, merge and split conserves them to rounding error; the walls book
    what they receive (tubule_P, tubule_L) and the kicks what the residual acceleration adds (external_P,
    external_L).  Every event record carries its own conservation error (dP_rel, dL_rel).

What is left out, and how much it matters: notebook 06, section 9 (gas drag, the payload's slow roll, charging,
wall vibration and detachment, restructuring inside clumps).  The negligible forces of the full simulator are not
in this engine at all.

Differences from v0.5, all deliberate: candidate pairs by a brute-force distance test once per step (no k-d tree,
no scipy: it runs in a web browser under Pyodide); one event at a time; exact radii and adhesion in every contact
law (v0.5 rounds radii to 0.5 um and the adhesion factor to 2 % when it caches contact laws); per-pair friction
from materials.py FRICTION unless Tubule(mu=...) is given (v0.5 used one value, 0.3).  On identical starts the two
engines give the same events (tests/test_dolle_core_sim3d.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time as _time

import numpy as np

from .clumps import Clumps, cross_rows, drift, grain_velocities, make_clumps, momentum, speed_bound, spin_bound
from .impacts import grain_grain, grain_wall
from .materials import G0

INCH = 0.0254


@dataclass(frozen=True)
class Tubule:
    """The box the grains fly in, and the engine settings.  The tubule spans 0 <= x <= size[0] etc. [m]; z is the
    long axis (the rocket's axis when the tubule stands).  Defaults: DOLL-E's 0.5 x 0.5 x 1.5 in borosilicate
    tubule (configs/mission_quarter_deck.json: inner_size_in, wall_material, wall_adhesion_factor, dt_s)."""
    size: tuple = (0.5 * INCH, 0.5 * INCH, 1.5 * INCH)
    wall: str = "glass_borosilicate"
    f_wall: float = 1.0                 # factor on the grain-wall work of adhesion ("bare" wall)
    accel: tuple = (0.0, 0.0, 1e-4 * G0)  # m/s^2, residual acceleration toward the nose: ASSUMED order of magnitude
    dt: float = 4e-3                    # s, the mission time step
    mu: float | None = None             # None: per-pair friction (materials.FRICTION); a number overrides (v0.5: 0.3)
    e_t: float = 0.0                    # tangential restitution (v0.5 default, ASSUMED)
    fragmentation: bool = True          # erosion and shattering of clumps (Dominik & Tielens 1997)
    separation: float = 1.0             # fraction of a break-up's left-over E_diss given to the fragments (ASSUMED;
                                        # 1 = v0.5, 0 = the opposite bound: impacts.py, point 4)
    n_sub: int = 300                    # contact sub-steps per impact (v0.5 steps_per_contact)


COUNT_KEYS = ("stick", "bounce", "wall_stick", "wall_bounce", "erosion", "shatter", "guard",
              "fragmentation_energy_limited")


@dataclass
class Sim:
    """State of one run: the tubule, the clumps, the time t [s] (t_now: time of the event being resolved), the
    event records and frames, the momentum ledger [kg m/s, kg m^2/s about the tubule origin] and the counters."""
    tubule: Tubule
    clumps: Clumps
    t: float = 0.0
    t_now: float = 0.0
    steps: int = 0
    events: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    tubule_P: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tubule_L: np.ndarray = field(default_factory=lambda: np.zeros(3))
    external_P: np.ndarray = field(default_factory=lambda: np.zeros(3))
    external_L: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    L0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    counts: dict = field(default_factory=lambda: {k: 0 for k in COUNT_KEYS})
    pairs: dict = field(default_factory=dict)
    E_bonds_broken: float = 0.0
    contain_corrections: int = 0
    contain_max: float = 0.0
    run_time: float = 0.0
    next_frame: float | None = None     # time of the next regular frame [s] (kept here so that run() can be called
                                        # in pieces, as the page's sandbox does, with an even frame spacing)


def make_sim(tubule: Tubule, materials, radii, positions, velocities, spins=None, adhesion=None) -> Sim:
    """A Sim with one free clump per grain (clumps.make_clumps arguments; positions inside the tubule)."""
    sim = Sim(tubule, make_clumps(materials, radii, positions, velocities, spins, adhesion))
    sim.P0, sim.L0 = momentum(sim.clumps)
    return sim


def free_clumps(sim: Sim) -> np.ndarray:
    """Ids of the clumps that are flying (alive and not wall-bound)."""
    c = sim.clumps
    return np.flatnonzero(c.alive & ~c.frozen)


def kick(sim: Sim, h: float) -> None:
    """V += a h for every free clump (the residual acceleration acts at the centre of mass: no torque).  Books
    the momentum M a h and angular momentum X x M a h it adds (an external force).  (v0.5 _kick, no drag/roll.)"""
    act = free_clumps(sim)
    if not len(act):
        return
    c, a = sim.clumps, np.asarray(sim.tubule.accel, dtype=float)
    dp = c.M[act, None] * (a * h)[None, :]
    sim.external_P += dp.sum(0)
    sim.external_L += cross_rows(c.X[act], dp).sum(0)
    c.V[act] += a * h


def candidate_pairs(sim: Sim, h: float) -> tuple[np.ndarray, float]:
    """Pairs (i, j) of grains that could touch within h [s], and the speed v_lim [m/s] that list is valid for.
    No grain moves faster than v_lim (clumps.speed_bound, with the spin term doubled as a margin for the rotation
    splitting), so two grains close a gap at most 2 v_lim h: every pair with a gap below that is a candidate.
    Brute force over all pairs, at least one of them free (replaces v0.5's k-d tree neighbour list)."""
    c = sim.clumps
    N = len(c.r)
    s = speed_bound(c) + spin_bound(c)[c.body] * np.linalg.norm(c.off, axis=1)
    v_lim = float(s.max()) if N else 0.0
    if N < 2:
        return np.zeros((0, 2), int), v_lim
    I, J = np.triu_indices(N, 1)
    free = ~c.frozen[c.body]
    sel = free[I] | free[J]
    I, J = I[sel], J[sel]
    gap = np.sqrt(((c.pos[I] - c.pos[J]) ** 2).sum(1)) - c.r[I] - c.r[J]
    keep = gap < 2.0 * v_lim * h + 1e-9
    return np.stack((I[keep], J[keep]), axis=1), v_lim


def rotation_window(sim: Sim, vm, horizon: float, cand) -> float:
    """Longest drift (<= horizon) over which the straight-line contact search stays valid (v0.5 _rotation_window).
    A clump spinning at omega turns by omega h; if that exceeds 0.05 rad, h is cut to the time its members need
    to reach anything: gap/(s + v_lim) for grain gaps (s = |V| + omega |off| the member's top speed, v_lim the
    fastest grain) and gap/s for the walls, but never below 0.05/omega."""
    c = sim.clumps
    multi = np.flatnonzero(c.alive & ~c.frozen & (c.n > 1))
    if not len(multi) or horizon <= 0:
        return horizon
    w = np.linalg.norm(c.L[multi], axis=1) / c.Ip[multi].min(1)
    fast = w * horizon > 0.05
    if not fast.any():
        return horizon
    w_b = np.zeros(len(c.r))
    w_b[multi[fast]] = w[fast]
    m = np.flatnonzero(w_b[c.body] > 0)
    s_m = np.linalg.norm(c.V[c.body[m]], axis=1) + w_b[c.body[m]] * np.linalg.norm(c.off[m], axis=1)
    v_lim = max(float(np.linalg.norm(vm, axis=1).max()), float(s_m.max()))
    gap = np.full(len(c.r), np.inf)
    if len(cand):
        i, j = cand[:, 0], cand[:, 1]
        infast = w_b[c.body] > 0
        sel = (infast[i] | infast[j]) & (c.body[i] != c.body[j])
        if sel.any():
            i, j = i[sel], j[sel]
            d = np.linalg.norm(c.pos[i] - c.pos[j], axis=1) - c.r[i] - c.r[j]
            np.minimum.at(gap, i, d)
            np.minimum.at(gap, j, d)
    size = np.asarray(sim.tubule.size)
    p, r = c.pos[m], c.r[m][:, None]
    g_wall = np.minimum(p - r, size[None, :] - p - r).min(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.minimum(np.maximum(gap[m], 0) / (s_m + v_lim), np.maximum(g_wall, 0) / s_m)
    h = np.maximum(np.nan_to_num(h, nan=np.inf), 0.05 / w_b[c.body[m]])
    return min(horizon, float(h.min()))


def next_contact(sim: Sim, vm, horizon: float, cand, excl, wexcl) -> tuple | None:
    """The earliest contact within `horizon` [s] if every grain moves in a straight line at its velocity vm:
    (t, "pp", k) for candidate pair k, (t, "wall", grain, axis, side), or None.

    Two grains, relative position d = pos_j - pos_i and velocity u = vm_j - vm_i, touch when |d + u t| = S =
    r_i + r_j: a t^2 + 2 b t + c = 0 with a = u.u, b = d.u, c = d.d - S^2.  They approach if b < 0; the first root
    is t = (-b - sqrt(b^2 - a c))/a, or 0 if they already touch (c <= 0).  A grain reaches the wall x = 0 at
    t = (x - r)/(-v_x) if v_x < 0, the wall x = size at (size - r - x)/v_x if v_x > 0.  excl / wexcl mark pairs
    and walls that touch without approaching.  (v0.5 _events.)"""
    c = sim.clumps
    best = None
    if len(cand):
        i, j = cand[:, 0], cand[:, 1]
        bi, bj = c.body[i], c.body[j]
        idx = np.flatnonzero((bi != bj) & ~(c.frozen[bi] & c.frozen[bj]) & ~excl)
        if len(idx):
            i, j = i[idx], j[idx]
            d, u, S = c.pos[j] - c.pos[i], vm[j] - vm[i], c.r[i] + c.r[j]
            b = (d * u).sum(1)
            a = (u * u).sum(1)
            cc = (d * d).sum(1) - S * S
            disc = b * b - a * cc
            ok = np.flatnonzero((b < 0) & (disc >= 0))
            if len(ok):
                t = np.where(cc[ok] <= 0, 0.0, (-b[ok] - np.sqrt(disc[ok])) / np.maximum(a[ok], 1e-300))
                k = int(np.argmin(t))
                if t[k] <= horizon:
                    best = (float(t[k]), "pp", int(idx[ok[k]]))
    g = np.flatnonzero(~c.frozen[c.body])
    if len(g):
        size = np.asarray(sim.tubule.size)
        p, v, r = c.pos[g], vm[g], c.r[g][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = np.where(v < 0, np.maximum(p - r, 0) / -v, np.inf)
            t_hi = np.where(v > 0, np.maximum(size[None, :] - r - p, 0) / v, np.inf)
        tw = np.minimum(t_lo, t_hi)
        tw[wexcl[g]] = np.inf
        k, ax = np.unravel_index(int(np.argmin(tw)), tw.shape)
        if tw[k, ax] <= horizon and (best is None or tw[k, ax] < best[0]):
            side = -1.0 if t_lo[k, ax] <= t_hi[k, ax] else 1.0
            best = (float(tw[k, ax]), "wall", int(g[k]), int(ax), side)
    return best


def resolve(sim: Sim, ev, cand) -> dict | None:
    """Resolve event ev (from next_contact) and measure its conservation error: the change of the momentum and
    angular momentum of the clumps involved plus what the tubule received, relative to the momenta involved
    (dP_rel, dL_rel; rounding error is ~1e-16).  None if the grains turned out not to approach."""
    c = sim.clumps
    grains = (ev[2],) if ev[1] == "wall" else tuple(int(x) for x in cand[ev[2]])
    before = {int(c.body[g]) for g in grains}
    members = [k for b in before for k in c.members[b]]
    P0, L0 = momentum(c, before)
    scale_P = float(sum(np.linalg.norm(c.M[b] * c.V[b]) for b in before))
    scale_L = float(sum(np.linalg.norm(c.L[b]) + np.linalg.norm(np.cross(c.X[b], c.M[b] * c.V[b])) for b in before))
    keep_P, keep_L = sim.tubule_P, sim.tubule_L                 # book this event's wall share separately
    sim.tubule_P, sim.tubule_L = np.zeros(3), np.zeros(3)
    if ev[1] == "pp":
        rec = grain_grain(sim, *grains)
    else:
        rec = grain_wall(sim, ev[2], ev[3], ev[4])
    wall_P, wall_L = sim.tubule_P, sim.tubule_L
    sim.tubule_P, sim.tubule_L = keep_P + wall_P, keep_L + wall_L
    if rec is None:
        return None
    after = {int(c.body[k]) for k in members}
    P1, L1 = momentum(c, after)
    rec["dP_rel"] = float(np.linalg.norm(P1 + wall_P - P0)) / max(scale_P, 1e-300)
    rec["dL_rel"] = float(np.linalg.norm(L1 + wall_L - L0)) / max(scale_L, 1e-300)
    rec["clumps_after"] = sorted(after)
    sim.events.append(rec)
    return rec


def contain(sim: Sim) -> None:
    """Safety net (v0.5 _contain): no free grain may end a step outside the tubule; its clump is pushed back.
    It should never act (counted in contain_corrections; the tests require zero)."""
    c = sim.clumps
    size = np.asarray(sim.tubule.size)
    lo = c.r[:, None] - c.pos
    hi = c.pos + c.r[:, None] - size[None, :]
    pen = np.where(lo > 0, lo, 0.0) - np.where(hi > 0, hi, 0.0)
    pen[c.frozen[c.body]] = 0.0
    worst = float(np.abs(pen).max(initial=0.0))
    if worst > 0:
        sim.contain_corrections += 1
        sim.contain_max = max(sim.contain_max, worst)
        N = len(c.r)
        shift = np.zeros((N, 3))
        for k in range(3):
            np.maximum.at(shift[:, k], c.body, np.where(pen[:, k] > 0, pen[:, k], 0.0))
            neg = np.zeros(N)
            np.minimum.at(neg, c.body, np.where(pen[:, k] < 0, pen[:, k], 0.0))
            shift[:, k] += neg
        c.X += shift
        c.pos = c.X[c.body] + c.off


def step(sim: Sim) -> None:
    """One velocity-Verlet step of length dt: half kick; drift for dt, stopping at every contact to resolve it;
    half kick.  (v0.5 ChamberSim.step, events one at a time.)"""
    c, dt = sim.clumps, sim.tubule.dt
    t0 = sim.t
    kick(sim, 0.5 * dt)
    remaining = dt
    cand, v_lim = candidate_pairs(sim, dt)
    excl, wexcl = np.zeros(len(cand), bool), np.zeros((len(c.r), 3), bool)
    for _ in range(100000):
        if remaining <= 1e-15:
            break
        vm = grain_velocities(c)
        win = rotation_window(sim, vm, remaining, cand)
        ev = next_contact(sim, vm, win, cand, excl, wexcl)
        if ev is None:
            drift(c, win)
            remaining -= win
            continue
        drift(c, ev[0])
        remaining -= ev[0]
        sim.t_now = t0 + dt - remaining
        rec = resolve(sim, ev, cand)
        if rec is None:                                   # touching but not approaching: skip it for now
            if ev[1] == "pp":
                excl[ev[2]] = True
            else:
                wexcl[ev[2], ev[3]] = True
            continue
        changed = np.zeros(len(c.r), bool)                # their clumps changed velocity: search them again
        for b in rec["clumps_after"]:
            changed[c.members[b]] = True
        excl &= ~(changed[cand[:, 0]] | changed[cand[:, 1]])
        wexcl[changed] = False
        s = speed_bound(c) + spin_bound(c)[c.body] * np.linalg.norm(c.off, axis=1)
        if float(s[changed].max()) > v_lim:               # a grain is now faster than the list allows: rebuild
            cand, v_lim = candidate_pairs(sim, remaining)
            excl, wexcl = np.zeros(len(cand), bool), np.zeros((len(c.r), 3), bool)
    else:
        raise RuntimeError(f"the step at t = {t0:.6f} s needed more than 100000 events")
    contain(sim)
    sim.steps += 1
    sim.t = sim.steps * dt
    kick(sim, 0.5 * dt)


MATERIAL_COLOUR = {"silica": 0, "graphite": 1}
"""Colour index of a single grain by material (the page's palette: 0 silica, pale blue; 1 graphite, dark grey);
other materials use 0.  Clumps of 2 or more grains use 2 + (clump id mod 12)."""


def frame(sim: Sim) -> dict:
    """Snapshot for the replay: t [s], pos_um (N x 3 integer micrometres; 0.5 um rounding, 1 % of a radius),
    clump (clump id of each grain), wall (1 if its clump is wall-bound), colour (palette index)."""
    c = sim.clumps
    n_of = c.n[c.body]
    colour = np.where(n_of > 1, 2 + c.body % 12, [MATERIAL_COLOUR.get(m, 0) for m in c.material])
    return {"t": float(sim.t), "pos_um": np.rint(c.pos * 1e6).astype(int), "clump": c.body.copy(),
            "wall": c.frozen[c.body].astype(int), "colour": colour.astype(int)}


def run(sim: Sim, t_end: float, frame_dt: float = 1 / 30, slow_until: float = 0.5,
        stop_when_settled: bool = True, on_frame=None, final_frame: bool = True) -> Sim:
    """Run until t_end [s] (or until every grain is on a wall).  Frames: every step up to slow_until [s] (slow
    motion of the cloud phase), then every frame_dt, and one at t_end (final_frame; the sandbox, which calls run()
    in pieces, asks for it only at the end of the whole run).  on_frame(sim, frame) is called for each frame."""
    def record():
        f = frame(sim)
        sim.frames.append(f)
        if on_frame is not None:
            on_frame(sim, f)
    t_start = _time.perf_counter()
    if not sim.frames:
        record()
    if sim.next_frame is None:
        sim.next_frame = max(slow_until, sim.t) + frame_dt
    n_steps = int(round((t_end - sim.t) / sim.tubule.dt))
    for k in range(n_steps):
        step(sim)
        settled = not len(free_clumps(sim))
        last = final_frame and k == n_steps - 1
        if sim.t <= slow_until + 1e-9 or sim.t >= sim.next_frame - 1e-9 or (settled and stop_when_settled) or last:
            record()
            if sim.t > slow_until + 1e-9:
                while sim.next_frame <= sim.t + 1e-9:
                    sim.next_frame += frame_dt
        if settled and stop_when_settled:
            break
    sim.run_time += _time.perf_counter() - t_start
    return sim


def conservation_error(sim: Sim) -> tuple[float, float]:
    """Relative error of the momentum ledger now: |P_free + P_tubule - P_external - P0| / (sum of |M V| booked),
    and the same for the angular momentum about the origin.  Rounding error only (~1e-15)."""
    P, L = momentum(sim.clumps)
    dP = P + sim.tubule_P - sim.external_P - sim.P0
    dL = L + sim.tubule_L - sim.external_L - sim.L0
    c = sim.clumps
    sP = float(np.sum(c.m * np.linalg.norm(c.V[c.body], axis=1))) + np.linalg.norm(sim.P0) + \
        np.linalg.norm(sim.tubule_P) + np.linalg.norm(sim.external_P)
    sL = np.linalg.norm(sim.L0) + np.linalg.norm(sim.tubule_L) + np.linalg.norm(sim.external_L) + np.linalg.norm(L)
    return float(np.linalg.norm(dP) / max(sP, 1e-300)), float(np.linalg.norm(dL) / max(sL, 1e-300))


def summary(sim: Sim) -> dict:
    """Counts and key numbers of a run (for the replay and the report)."""
    c = sim.clumps
    pp = [e for e in sim.events if e["kind"] == "pp"]
    sizes = [int(c.n[b]) for b in np.flatnonzero(c.alive)]
    hist = {str(s): sizes.count(s) for s in sorted(set(sizes))}
    eP, eL = conservation_error(sim)
    return {"counts": dict(sim.counts), "grain_grain_impacts": len(pp), "events": len(sim.events),
            "clump_sizes_at_end": hist, "largest_clump": max(sizes) if sizes else 0,
            "free_clumps_at_end": int(len(free_clumps(sim))), "t_end_s": float(sim.t),
            "last_grain_grain_impact_s": max((e["t"] for e in pp), default=None),
            "median_grain_grain_v_n_m_s": float(np.median([e["v_n"] for e in pp])) if pp else None,
            "max_event_dP_rel": max((e["dP_rel"] for e in sim.events), default=0.0),
            "max_event_dL_rel": max((e["dL_rel"] for e in sim.events), default=0.0),
            "ledger_dP_rel": eP, "ledger_dL_rel": eL, "safety_net_corrections": sim.contain_corrections,
            "bond_energy_broken_J": sim.E_bonds_broken, "run_time_s": sim.run_time}

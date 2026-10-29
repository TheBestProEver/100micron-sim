"""DOLL-E v0.6 flight engine: every grain is its own body and every contact has the full contact mechanics.

Each grain is in one of three regimes, switched by exact or bounded tests:

* FREE: touching nothing and unable to touch anything during the next outer step H (its gap to every other grain and
  wall exceeds the most both can move in H). It flies on the exact free-flight integrator of simulation.py (velocity
  Verlet with drag and Coriolis integrated in closed form), so no contact can be missed or found late.
* ACTIVE: everything that touches something, or could touch something within H, is integrated by the full-stiffness
  soft-sphere DEM of dem.py (JKR + plastic yield + damping, Mindlin sliding, rolling and twisting springs with their
  limits; checked against LAMMPS to <= 2e-6), at a step of the fastest contact period / steps_per_period.
* ASLEEP: a grain resting on a wall whose contact is proven to stay inside its elastic limits: for the loads over the
  next step, the normal load stays above -F_c (pull-off), the tangential load below the sliding limit, and the rolling
  and twisting loads below theirs, each with a 2% margin. Inside these limits the contact mechanics is elastic, so no
  plastic variable (contact point, rolled distance, twist) can change and the grain cannot move (except by elastic
  deflections of < 1e-10 m). The test runs every outer step; the grain wakes (becomes ACTIVE) the moment it fails, at
  every load breakpoint (thrust on/off, de-spin, shocks), during release waveforms and vibration (kHz loads), and when
  another body could reach it. On waking, its springs are set to the exact static solution for the current load, so
  waking adds no transient. A woken grain goes back to sleep once its residual motion is too small to reach a limit.
Grains touching other grains (aggregates, piles) stay ACTIVE (v0.6: no sleep for multi-contact clusters yet).

Impacts are recorded when a contact forms (approach speed, tangential speed, normal, relative velocity), and their
outcome when it breaks (bounce, exit speed) or when it has held for 2 ms (stick).

Provenance: written for DOLL-E. Contact law: dem.py. Free flight, forces, setup: simulation.py. Loads: loads.py.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import dem, roll1, slaved, statics
from .environment import G0
from .loads import Loads

FREE, ACTIVE, ASLEEP, ROLL, SLAV = 0, 1, 2, 3, 4
# a run saved at a checkpoint may continue under another scenario only if a fresh run of that scenario differs from a
# fresh run of the saved one in these attributes alone (release schedule, flight-profile options) and its loads are
# identical up to the checkpoint (branch())
BRANCH_ALLOWED = ("cfg", "chamber", "base.cfg", "base.chamber", "base.release", "base.env.cfg", "base.env.profile.",
                  "loads.")
STICK_HOLD_S = 2e-3          # a contact that has held this long after forming is a stick
LINRES_MAX = 64              # clusters up to this size may sleep with their residual motion tracked exactly
FACE_NAMES = ["-x", "+x", "-y", "+y", "-z", "+z"]

# friction coefficients (rolling_params.md): silica on glass 0.4 (Meyer et al. 2014; Buckley 1973 in vacuum 0.5),
# silica-silica 0.6 (Liu et al. 2017), graphite dry/vacuum 0.5 (Buckley 1971). Other pairs: 0.4 (ASSUMED).
MU_DEFAULT = {("silica", "wall"): 0.4, ("silica", "silica"): 0.6, ("graphite", "wall"): 0.5,
              ("graphite", "graphite"): 0.5, ("graphite", "silica"): 0.5}


class FullContactSim:
    def __init__(self, cfg: dict, chamber: dict, seed: int = 0):
        from .simulation import ChamberSim
        b = self.base = ChamberSim(cfg, chamber, seed)
        self.cfg, self.chamber = cfg, chamber
        self.env, self.name, self.N = b.env, b.name, b.N
        self.rad, self.mat, self.mats, self.wall_mat = b.rad, b.mat, b.mats, b.wall_mat
        self.duration, self.size = b.duration, b.size
        self.dt = b.dt
        N = self.N
        cc = cfg.get("contact", {})
        self.n_ppp = int(cc.get("steps_per_period", 50))
        self.h_roll = float(cc.get("rolling_hysteresis", 0.5))
        self.q_roll = float(cc.get("rolling_q", 100.0))
        self.q_twist = float(cc.get("twisting_q", 100.0))
        self.x_gt = float(cc.get("tangential_damping_ratio", 1.0))
        self.margin = float(cc.get("sleep_margin", 0.02))
        self.n_ppp_roll = int(cc.get("roll_steps_per_period", 200))
        # a free or rolling grain joins the DEM before its gap to anything can fall below this (any positive value is
        # exact: the DEM resolves the approach from there); smaller means less DEM time
        self.d_cap = float(cc.get("capture_distance_m", 1e-7))
        self.use_roll = bool(cc.get("slaved_rolling", True))
        self.all_dem = bool(cc.get("all_dem", False))        # reference mode: every grain in the DEM throughout
        self.use_slaved = bool(cc.get("slaved_clusters", True))
        self.n_ppp_slv = int(cc.get("slaved_steps_per_period", 400))
        self.mu_tab = dict(MU_DEFAULT)
        for k, val in (cc.get("friction", {}) or {}).items():
            a, _, c = k.partition("-")
            self.mu_tab[(a, c)] = float(val)
        M = [b.mats[k] for k in b.mat]
        self.E = np.array([m.E for m in M])
        self.nu = np.array([m.nu for m in M])
        self.gam = np.array([m.gamma for m in M])
        self.edry = np.array([m.e_dry for m in M])
        self.py = np.array([m.yield_pressure for m in M])
        self.m = b.mass.copy()
        self.I = b.I0.copy()
        self.adh = b.adh.copy()
        self.gdrag = (b.dragc / b.mass) if b.env.P > 0 else np.zeros(N)
        self.x = b.pos.copy()
        self.v = b.V.copy()
        self.w = b.L / b.I0[:, None]
        self.loads = Loads(b.env, b.release, self._vibration(chamber.get("vibration"), seed))
        if b.env.jitter_rms > 0 or getattr(b.env, "_csv", None) is not None:
            raise ValueError(f"{self.name}: g-jitter (jitter_rms_g) and accel_csv are not represented in the "
                             "full-contact engine's loads, so its motion bounds could not cover them; set "
                             "jitter_rms_g = 0 and leave out accel_csv (the event-driven engine supports both)")
        self.off = b.axis_offset.copy()
        self.L = self.size.copy()
        self.regime = np.full(N, FREE, np.int8)
        # contact store
        self._key = {}
        self._prm = np.zeros((0, dem.NP))
        self._hist = np.zeros((0, dem.NH))
        self._seen_form = np.zeros(0)
        self._seen_break = np.zeros(0)
        self._ckeys = []
        self._kij = np.zeros((0, 2), np.int64)   # the same keys as an array
        self.collisions, self.detach_log, self.timeseries = [], [], []
        self.frames, self.frame_t, self.frame_bodies, self.frame_frozen, self.frame_spin = [], [], [], [], []
        self.n_events = {"stick": 0, "bounce": 0, "wall_stick": 0, "wall_bounce": 0, "detach": 0, "wake": 0,
                         "sleep": 0, "active_grain_steps": 0, "dem_steps": 0, "rebuilds": 0}
        self._pending = {}          # contact row -> collision record awaiting its outcome
        self._ever_freed = set()
        self.t = 0.0
        self._dt_dem = self._contact_dt()
        # dust resting on a face at lift-off: in adhesive equilibrium with the wall at zero load, asleep
        self._sleep_face = {}
        self._roll_face = {}        # grain -> face it rolls on (regime ROLL)
        self._last_face = -1
        self._islands = {}          # id -> sleeping cluster (members, contacts, factorised stiffness, reference)
        self._island_of = {}        # grain -> island id
        self._next_island = 0
        self._slv = {}              # id -> slaved cluster (regime SLAV): members, contact rows, local contact lists
        self._slv_of = {}           # grain -> slaved cluster id
        self._next_slv = 0
        self.max_island = int(cc.get("max_sleeping_cluster", 800))
        self._F_lr = np.zeros((N, 3))
        for k, ax, side in b._settled:
            face = 2 * ax + (1 if side > 0 else 0)
            row = self._row(k, -1 - face)
            self._place_at_equilibrium(k, face, row, float(self.env.t0_flight))
            self.regime[k] = ASLEEP
            self._sleep_face[int(k)] = face
            self.v[k] = 0.0
            self.w[k] = 0.0
        b.frozen[:] = self.regime != FREE
        self.checkpoints = []       # flight times [s] at which run() saves the whole state (save_checkpoint)
        self.checkpoint_dir = None
        self._next_out = None
        self._init_sig = state_signature(self)

    @staticmethod
    def _vibration(spec, seed):
        """Deck vibration histories from a scenario spec: {"windows_s": [[t0, t1], ...], "level": "guide" |
        "level_two", "seed": n}. 'guide': the RockSat User Guide 2026 test spectra on all three axes at once
        (thrust along z, lateral along x and y), an upper bound for flight (vibration.py)."""
        if not spec:
            return None
        from . import vibration as V
        rng = np.random.default_rng(int(spec.get("seed", 1000 + seed)))
        if spec.get("level", "guide") == "level_two":
            bp = {"x": V.LEVEL_TWO["breakpoints"], "y": V.LEVEL_TWO["breakpoints"], "z": V.LEVEL_TWO["breakpoints"]}
        else:
            bp = {"x": V.GUIDE_RANDOM["lateral"]["breakpoints"], "y": V.GUIDE_RANDOM["lateral"]["breakpoints"],
                  "z": V.GUIDE_RANDOM["thrust"]["breakpoints"]}
        fs = float(spec.get("sample_rate_hz", 65536))
        return [V.guide_random_history(float(a), bp, float(b2) - float(a), fs, rng) for a, b2 in spec["windows_s"]]

    # ------------------------------------------------------------------ contact store
    def _mu(self, i, j):
        a = self.mats[self.mat[i]].name
        c = "wall" if j < 0 else self.mats[self.mat[j]].name
        return self.mu_tab.get((a, c), self.mu_tab.get((c, a), 0.4))

    def _params(self, i, j):
        if j >= 0:
            wk = 2.0 * math.sqrt(self.gam[i] * self.gam[j]) * math.sqrt(self.adh[i] * self.adh[j])
            return dem.make_params(self.rad[i], self.rad[j], self.m[i], self.m[j], self.E[i], self.nu[i], self.E[j],
                                   self.nu[j], wk, math.sqrt(self.edry[i] * self.edry[j]),
                                   min(self.py[i], self.py[j]), self._mu(i, j), self.h_roll, self.q_roll,
                                   self.q_twist, dem.DAMP_TSUJI_JKR, dem.BREAK_DISPLACEMENT, self.x_gt, 0.0)
        W = self.wall_mat
        wk = 2.0 * math.sqrt(self.gam[i] * W.gamma) * self.adh[i] * self.base.wall_adh
        return dem.make_params(self.rad[i], 0.0, self.m[i], math.inf, self.E[i], self.nu[i], W.E, W.nu, wk,
                               math.sqrt(self.edry[i] * W.e_dry), min(self.py[i], W.yield_pressure), self._mu(i, -1),
                               self.h_roll, self.q_roll, self.q_twist, dem.DAMP_TSUJI_JKR, dem.BREAK_DISPLACEMENT,
                               self.x_gt, 0.0)

    def _row(self, i, j):
        key = (int(i), int(j))
        r = self._key.get(key)
        if r is None:
            r = len(self._ckeys)
            self._key[key] = r
            self._ckeys.append(key)
            if r >= len(self._prm):
                n = max(64, 2 * len(self._prm))
                self._prm = np.vstack([self._prm, np.zeros((n - len(self._prm), dem.NP))])
                self._hist = np.vstack([self._hist, np.zeros((n - len(self._hist), dem.NH))])
                self._seen_form = np.concatenate([self._seen_form, np.zeros(n - len(self._seen_form))])
                self._seen_break = np.concatenate([self._seen_break, np.zeros(n - len(self._seen_break))])
                self._kij = np.vstack([self._kij, np.zeros((n - len(self._kij), 2), np.int64)])
            self._prm[r] = self._params(i, j)
            self._kij[r] = (int(i), int(j))
        return r

    def _contact_dt(self):
        """DEM step: the fastest contact period (normal, k_n = 1.2 E* a0; sliding, 8 G* a0 with the rotational
        coupling) over every grain-wall and grain-grain pairing of the smallest grains, / steps_per_period."""
        Tmin = math.inf
        order = np.argsort(self.rad)
        cand = list(order[:3]) + [int(np.argmin(self.rad))]
        for i in cand:
            for j in [-1] + list(order[:2]):
                if j == i:
                    continue
                p = self._params(i, j) if j < 0 else self._params(i, int(j))
                meff = p[dem.P_MEFF]
                kn = 1.2 * p[dem.P_ESTAR] * p[dem.P_A0]
                ks = p[dem.P_KT] * p[dem.P_A0]
                Tn = 2 * math.pi * math.sqrt(meff / kn)
                coup = 3.5 / self.m[i] if j < 0 else 7.0 / self.m[i]
                Ts = 2 * math.pi / math.sqrt(ks * coup)
                Tmin = min(Tmin, Tn, Ts)
        return Tmin / self.n_ppp

    def _hertz_dt(self, vrel):
        """Inside fast impacts the Hertz contact time 2.87 (m*^2 / (R* E*^2 v))^(1/5) is shorter than the JKR periods."""
        if vrel <= 0:
            return math.inf
        i = int(np.argmin(self.rad))
        p = self._params(i, i if self.N == 1 else int(np.argsort(self.rad)[1]))
        th = 2.87 * (p[dem.P_MEFF] ** 2 / (p[dem.P_REFF] * p[dem.P_ESTAR] ** 2 * vrel)) ** 0.2
        return th / self.n_ppp

    # ------------------------------------------------------------------ statics of a grain on a wall
    @staticmethod
    def _wall_normal(face):
        n = np.zeros(3)
        ax = face // 2
        n[ax] = 1.0 if face % 2 == 0 else -1.0       # from the wall into the tubule
        return n

    def _body_force(self, idx, T):
        """Quasi-static force [N] and torque on resting grains idx at flight time T (v = 0, omega = 0)."""
        a = self.loads.accel(np.array([T]))[0]
        Om = self.loads.omega(T)[0]
        Omd = self.loads.omega_dot(T)[0]
        r = self.x[idx] + self.off[None, :]
        cen = -np.cross(Om[None, :], np.cross(Om[None, :], r))
        eul = -np.cross(Omd[None, :], r)
        F = self.m[idx, None] * (a[None, :] + cen + eul) + self._F_lr[idx]
        T_in = -self.I[idx, None] * Omd[None, :]            # spin held relative to the tubule
        return F, T_in

    def _margins(self, idx, faces, F, Tq):
        """For grains idx resting on faces (one wall contact each): the ratios of the static normal, sliding,
        rolling and twisting loads to their limits. > 1 - margin means the grain must be woken."""
        rows = np.array([self._key[(int(i), -1 - int(f))] for i, f in zip(idx, faces)])
        p = self._prm[rows]
        h = self._hist[rows]
        n = np.zeros((len(idx), 3))
        n[np.arange(len(idx)), faces // 2] = np.where(faces % 2 == 0, 1.0, -1.0)
        Fn = -(F * n).sum(1)                                 # pressing load (> 0 pushes the grain onto the wall)
        Ft = F + Fn[:, None] * n
        Fc = np.where(h[:, dem.H_YLD] > 0.5, h[:, dem.H_FCP], p[:, dem.P_FC])
        Fncrit = np.abs(Fn + 2 * Fc)
        ft = np.linalg.norm(Ft, axis=1)
        tw = np.abs((Tq * n).sum(1))
        tt = np.linalg.norm(Tq - (Tq * n).sum(1)[:, None] * n, axis=1)
        R = p[:, dem.P_REFF]
        r_pull = np.where(Fn < 0, -Fn / Fc, 0.0)
        r_slide = ft / (p[:, dem.P_MUT] * Fncrit)
        r_roll = (ft + tt / R) / (p[:, dem.P_MUR] * Fncrit)
        r_twist = tw / np.maximum(p[:, dem.P_MUTW] * Fncrit, 1e-300)
        return np.maximum.reduce([r_pull, r_slide, r_roll, r_twist]), rows, Fn, Ft, Tq, n

    def _place_at_equilibrium(self, k, face, row, T, F=None, Tq=None):
        """Put grain k at the exact static equilibrium of its one wall contact under the load at T: normal overlap
        from the JKR curve, sliding, rolling and twisting springs from the static force and torque."""
        if F is None:
            F, Tq = self._body_force(np.array([k]), T)
        F, Tq = F[0], Tq[0]
        n = self._wall_normal(face)
        p, h = self._prm[row], self._hist[row]
        R, w, E = p[dem.P_REFF], p[dem.P_W], p[dem.P_ESTAR]
        Fn = -float(F @ n)
        # JKR: the overlap on the stable branch for F_el = Fn (the flattened branch, offset d_p, if yielded)
        delta, a = slaved.jkr_overlap(p, h, Fn, h[dem.H_A] if h[dem.H_A] > 0 else p[dem.P_A0])
        ax = face // 2
        self.x[k, ax] = (self.rad[k] - delta) if face % 2 == 0 else (self.L[ax] - self.rad[k] + delta)
        Ft = F + Fn * n
        tw = float(Tq @ n)
        Tt = Tq - tw * n
        nf, tf, vin, vt, nb, tb, vo, am = (h[dem.H_NFORM], h[dem.H_TFORM], h[dem.H_VIN], h[dem.H_VTIN],
                                           h[dem.H_NBREAK], h[dem.H_TBREAK], h[dem.H_VOUT], h[dem.H_AMAX])
        # elastic tangential deflection held by the sliding and rolling springs before (centre moves with them:
        # v_t = d(xi_t)/dt + d(xi_r)/dt for a sphere on a wall)
        defl_old = (h[dem.H_XT:dem.H_XT + 3] + h[dem.H_XR:dem.H_XR + 3]).copy() if h[dem.H_TOUCH] > 0.5             else np.zeros(3)
        plastic = (h[dem.H_YLD], h[dem.H_RP], h[dem.H_DP], h[dem.H_FCP], h[dem.H_DMAX])
        h[:dem.NHP] = 0.0
        h[dem.H_YLD], h[dem.H_RP], h[dem.H_DP], h[dem.H_FCP], h[dem.H_DMAX] = plastic   # a flattening stays
        h[dem.H_TOUCH] = 1.0
        h[dem.H_A] = a
        h[dem.H_DMAX] = max(h[dem.H_DMAX], delta)
        # contact force on the grain balances the load: tangential force fs = -Ft (fs = -k_t a xi_t), the rolling
        # pseudo-force fr balances the torque about the centre: R n x (fr - fs) = -T_tangential
        if h[dem.H_YLD] > 0.5:                        # as dem.contact_force
            ks = p[dem.P_KT] * (4.5 * math.pi * w * h[dem.H_RP] ** 2 / E) ** (1.0 / 3.0)
        elif int(p[dem.P_TMODE]) == dem.TANG_MINDLIN:
            ks = p[dem.P_KT] * a
        else:
            ks = p[dem.P_KT] * p[dem.P_A0]
        h[dem.H_XT:dem.H_XT + 3] = Ft / ks
        fr = -Ft + np.cross(n, Tt) / R
        h[dem.H_XR:dem.H_XR + 3] = -fr / p[dem.P_KR]
        h[dem.H_XTW] = tw / p[dem.P_KTW]
        d = h[dem.H_XT:dem.H_XT + 3] + h[dem.H_XR:dem.H_XR + 3] - defl_old
        d -= (d @ n) * n
        self.x[k] += d
        h[dem.H_NFORM], h[dem.H_TFORM], h[dem.H_VIN], h[dem.H_VTIN] = nf, tf, vin, vt
        h[dem.H_NBREAK], h[dem.H_TBREAK], h[dem.H_VOUT], h[dem.H_AMAX] = nb, tb, vo, max(am, a)

    # ------------------------------------------------------------------ long-range and static forces
    def _update_long_range(self, t):
        b = self.base
        b.pos = self.x.copy()
        b.X = self.x.copy()
        b.V = self.v.copy()
        b.t = t
        b.env.update(t)
        if b._needs_rebuild(self.v):
            b._build_neighbors(self.v)
        b._pair_stale = b._charge_stale = b._sc_dirty = True
        F = b._forces_pos(self.v)
        if b._F_cf is not None:
            F = F - b._F_cf
        self._F_lr = F

    # ------------------------------------------------------------------ one outer step
    def _outer_dt(self, t):
        """Outer step: dt_s (4 ms by default); with free grains under launch loads, dt_launch_s scaled to the load
        (as in v0.5, so their free flight stays accurate); never more than 0.02 rad of roll; a capture shell of at
        >= 40 steps per cycle of any release
        waveform or vibration; and every step ends exactly at the next load breakpoint."""
        T = self.env.t0_flight + t
        prof = self.env.profile
        dt = self.base.dt_base
        free = self.regime == FREE
        if prof is not None and free.any() and (T < prof.ug_start_s or T >= prof.ug_end_s):
            a_g = max(float(np.linalg.norm(prof.apparent_accel(T))), 1e-6) / G0
            dt = min(dt, self.base.dt_launch * 2.0 ** math.floor(math.log2(math.sqrt(25.0 / a_g))))
        W = float(np.linalg.norm(self.loads.omega(np.array([T, T + dt])), axis=1).max())
        if W > 0:
            dt = min(dt, 0.02 / W)
        if self.loads.fast_window(T, T + dt):
            dt = min(dt, 1.0 / (40.0 * self.loads.max_freq(T, T + dt)))
        nb = self.loads.next_breakpoint(T)
        if T + dt > nb - 1e-12:
            dt = max(nb - T, 1e-9)
        if self.loads.breakpoint_in(T - 1e-9, T + 1e-9):
            # just after a load jump every grain on a wall is in the full model while its stiff modes ring (they
            # decay in ~5 us); a short step, after which the usual tests hand them back to the cheap models
            dt = min(dt, 5e-5)
        return min(dt, self.duration - t)

    def _motion_bound(self, H, T):
        """Most each grain can move in H: |v| H + (1/2) a_max H^2, a_max bounding every body force per unit mass
        over the step: the loads' bound over [T, T + H] (Loads.accel_bound), centrifugal and Euler terms out to the
        farthest point of the tubule, and the velocity-dependent Coriolis and drag terms c |v| (c = 2 Omega + g) with
        the speed they can reach: |a| <= (a0 + c |v0|) exp(c H) (Gronwall)."""
        a = self.loads.accel_bound(T, T + H)
        Om = float(np.linalg.norm(self.loads.omega(np.array([T, T + H])), axis=1).max())
        Omd = float(np.linalg.norm(self.loads.omega_dot(np.array([T, T + H - 1e-12])), axis=1).max())
        rmax = float(np.linalg.norm(np.abs(self.off) + self.L))
        vm = np.linalg.norm(self.v, axis=1)
        c = 2 * Om + self.gdrag
        amax = (np.linalg.norm(a) + Om * Om * rmax + Omd * rmax + np.linalg.norm(self._F_lr, axis=1) / self.m
                + c * vm) * np.exp(c * H)
        return vm * H + 0.5 * amax * H * H + 1e-9

    def _motion_bound_rest(self, H, T):
        v = self.v
        self.v = np.zeros_like(v)
        try:
            return self._motion_bound(H, T)
        finally:
            self.v = v

    def step(self, t, H):
        """One outer step, taken with the motion bounds below and then verified: every assumption one grain's
        treatment made about another is checked against what actually happened. The speed of every grain in the DEM
        against the speed bound the others used for it; the path of every rolling grain against the rolling envelope
        the others used for it; every grain that entered the DEM during the step (a sleeper a DEM grain came up to, a
        flying or rolling grain that was captured) against every grain outside the DEM after it entered. If a check fails the
        step is redone from its start, with those grains in the DEM from the start (or a larger speed bound), so no
        interaction is ever missed."""
        snap = self._snapshot()
        force, vfac = (set(range(self.N)) if self.all_dem else set()), 1.0
        noexit = set()
        why = {}
        for _ in range(64):
            bad, vf = self._step_once(t, H, force, vfac, noexit)
            if not bad and vf is None:
                break
            missed = set(self._chk["missed"])
            rolled = set(int(k) for k in self._chk["roll_bad"])
            key = "redos_missed_contact" if missed else ("redos_speed_bound" if vf is not None else
                                                         "redos_rolling_envelope" if rolled else "redos_outside_record")
            why[key] = why.get(key, 0) + 1
            self._restore(snap)
            if vf is None and bad <= force and (missed | rolled) <= noexit:
                force = noexit = set(range(self.N))   # nothing would change: every grain in the DEM throughout
            force |= bad                           # in the DEM from the start: the same check cannot fail again for them
            noexit |= missed | rolled              # these stay in it for the whole step: an exit could repeat the failure
            if vf is not None:
                vfac = vf
        else:
            raise RuntimeError(f"{self.name}: the step at t = {t} s could not be verified")
        for key, n in why.items():                 # every redo, by the check that asked for it
            self.n_events[key] = self.n_events.get(key, 0) + n
            self.n_events["redos"] = self.n_events.get("redos", 0) + n
        self.t = t + H
        self._log_events(t + H)
        self._classify(t + H)

    @staticmethod
    def _island_lu(isl):
        """The island's LU factor (refactorised, identically, after a checkpoint is loaded)."""
        if isl.get("lu") is None:
            from scipy.sparse.linalg import splu
            isl["lu"] = splu(isl["K"])
        return isl["lu"]

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_islands"] = {i: {k: v for k, v in isl.items() if k != "lu"} for i, isl in self._islands.items()}
        for k in ("_chk", "_vd_step", "_ag_step", "_noexit"):     # the last step's scratch state
            st.pop(k, None)
        return st

    def checkpoint_path(self, T, directory=None):
        return Path(directory or self.checkpoint_dir) / f"{self.name}_T{T:08.3f}.pkl"

    def save_checkpoint(self):
        """Save the whole state (pickle) at the current time, for branch()."""
        T = self.env.t0_flight + self.t
        path = self.checkpoint_path(T)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        self.n_events.setdefault("checkpoints_saved", []).append(round(T, 3))

    def _snapshot(self):
        nr = len(self._ckeys)
        return dict(x=self.x.copy(), v=self.v.copy(), w=self.w.copy(), regime=self.regime.copy(),
                    hist=self._hist[:nr].copy(), nr=nr, F_lr=self._F_lr.copy(), sleep_face=dict(self._sleep_face),
                    roll_face=dict(self._roll_face), islands=dict(self._islands), island_of=dict(self._island_of),
                    next_island=self._next_island, last_face=getattr(self, "_last_face", -1),
                    n_events=dict(self.n_events), rng=copy.deepcopy(self.base.rng.bit_generator.state),
                    slv=copy.deepcopy(self._slv), slv_of=dict(self._slv_of), next_slv=self._next_slv)

    def _restore(self, sn):
        self.x[:], self.v[:], self.w[:], self.regime[:] = sn["x"], sn["v"], sn["w"], sn["regime"]
        nr = sn["nr"]
        self._hist[:nr] = sn["hist"]
        self._hist[nr:] = 0.0                       # rows first made during the failed attempt
        self._F_lr = sn["F_lr"].copy()
        self._sleep_face, self._roll_face = dict(sn["sleep_face"]), dict(sn["roll_face"])
        self._islands, self._island_of = dict(sn["islands"]), dict(sn["island_of"])
        self._next_island, self._last_face, self.n_events = sn["next_island"], sn["last_face"], dict(sn["n_events"])
        self.base.rng.bit_generator.state = copy.deepcopy(sn["rng"])
        self._slv, self._slv_of, self._next_slv = copy.deepcopy(sn["slv"]), dict(sn["slv_of"]), sn["next_slv"]

    def _step_once(self, t, H, force, vfac, noexit=frozenset()):
        T = self.env.t0_flight + t
        self._update_long_range(t)
        v_start = self.v.copy()
        fast = self.loads.fast_window(T, T + H) or self.loads.breakpoint_in(T - 1e-9, T + 1e-9)
        # 1. asleep grains: certificate for the loads over the step (start, middle, end)
        asl = np.array([k for k in np.flatnonzero(self.regime == ASLEEP) if int(k) in self._sleep_face], np.int64)
        wake = np.zeros(self.N, bool)
        if len(asl):
            if fast:
                wake[asl] = True
            else:
                faces = np.array([self._sleep_face[k] for k in asl])
                worst = np.zeros(len(asl))
                for Ts in (T, T + 0.5 * H, T + H):
                    F, Tq = self._body_force(asl, Ts)
                    ratio = self._margins(asl, faces, F, Tq)[0]
                    worst = np.maximum(worst, ratio)
                wake[asl[worst > 1 - self.margin]] = True
        for iid in list(self._islands):
            isl = self._islands[iid]
            if fast:
                wake[isl["members"]] = True
                continue
            worst = 0.0
            for Ts in (T, T + 0.5 * H, T + H):
                worst = max(worst, float(self._island_ratio(isl, Ts).max()))
            if worst > 1 - self.margin:
                wake[isl["members"]] = True
        # 2. wake: a cluster wakes whole (into the DEM); a single grain goes to the slaved rolling model. Grains a
        # redo names go into the DEM from the start. Sleepers whose certificate holds stay asleep (static): one joins
        # the DEM when a DEM grain comes within 2 d_cap of it (see _dem), or a rolling or flying grain is captured near
        # it, and the verification below checks that nothing outside the DEM relied on it staying put.
        forced = np.zeros(self.N, bool)
        if force:
            forced[np.array(sorted(force), np.int64)] = True
        act = wake | forced
        for k in np.flatnonzero(act & (self.regime == ASLEEP)):
            if int(k) in self._island_of:
                act[self._islands[self._island_of[int(k)]]["members"]] = True
        newly = act & (self.regime == ASLEEP)
        woken = set()
        for k in np.flatnonzero(newly):
            k = int(k)
            if self.regime[k] != ASLEEP:          # already woken with its cluster
                continue
            if k in self._island_of:
                iid = self._island_of[k]
                if iid not in woken:
                    woken.add(iid)
                    self._wake_island(iid, T)
            else:
                self._wake(k, T)
                if self.use_roll and not forced[k]:
                    self._roll_face[k] = self._last_face
                    self.regime[k] = ROLL
                else:
                    self.regime[k] = ACTIVE
        for k in np.flatnonzero(forced & ((self.regime == ROLL) | (self.regime == FREE))):
            self._roll_face.pop(int(k), None)
            self.regime[k] = ACTIVE
        # a load jump (breakpoint) excites the stiff modes that the slaved models hold quasi-static: the full model
        # takes those grains for the step; and a redo that names a grain of a slaved cluster puts the cluster there
        jump = self.loads.breakpoint_in(T - 1e-9, T + 1e-9)
        for cid in list(self._slv):
            if jump or forced[self._slv[cid]["members"]].any():
                self._dissolve_slv(cid)
        if jump:
            for k in np.flatnonzero(self.regime == ROLL):
                self._roll_face.pop(int(k), None)
                self.regime[k] = ACTIVE
        # motion bounds: |v| H + a_max H^2 / 2 (body forces), the rolling envelope for rolling grains, 0 asleep
        s = self._motion_bound(H, T)
        s[self.regime == ASLEEP] = 0.0
        s_off = np.zeros(self.N)
        ro_all = np.flatnonzero(self.regime == ROLL)
        if len(ro_all):
            s[ro_all], s_off[ro_all] = self._roll_bound(ro_all, H, T)
        reg_seen = self.regime.copy()
        x_seen = self.x.copy()
        # slaved clusters: integrated over the whole step first (exact, stopping where anything static is within the
        # capture distance or a limit is near); the path they actually take is their envelope for everyone else
        slv_pre = {}
        for cid in list(self._slv):
            pre = slv_pre[cid] = self._slv_prerun(cid, t, H, T)
            c = self._slv[cid]
            mem, rows = c["members"], c["rows"]
            s[mem] = pre["s"]
            s_off[mem] = pre["s"]
            # the envelope is measured from the step start: back to the start state for the capture times, speed
            # bounds and step sizes; the end state is applied after them if nothing stops the cluster earlier
            pre["end"] = (self.x[mem].copy(), self.v[mem].copy(), self.w[mem].copy(), self._hist[rows].copy(),
                          c["lam"].copy())
            self._slv_restore(pre)
        self._vdem_fac = vfac
        self._noexit = noexit
        # dmax: each grain's largest distance in the step from x_ref, where it was when it entered the DEM (t_in)
        self._chk = dict(speed=[], vmax=[], outside=[], dmax=np.zeros(self.N), in_dem=np.zeros(self.N, bool),
                         t_in=np.full(self.N, np.inf), x_ref=x_seen.copy(), roll_bad=set(), x_seen=x_seen,
                         slv_of=dict(self._slv_of), exited={}, missed=set(), was_out=np.zeros(self.N, bool))
        vm0 = np.linalg.norm(self.v, axis=1)
        a_b = np.maximum((s - s_off - vm0 * H) * 2 / (H * H), 0.0)
        # 3. capture times: the earliest time each free or rolling grain could come within the capture distance of
        # anything (bounded motion of both sides). Up to it, a free grain flies exactly (velocity Verlet with
        # closed-form drag and Coriolis) and a rolling grain rolls on the slaved model; there it joins the DEM.
        joins = {}
        xjoin = []                                # (time, grains): slaved clusters joining off the dt_d grid
        dt_d = min(self._dt_dem, self._hertz_dt(2 * float(np.linalg.norm(self.v, axis=1).max())))
        self._join_dt = dt_d
        mob_idx = np.flatnonzero((self.regime == FREE) | (self.regime == ROLL) | (self.regime == SLAV))
        self._vd_step, self._ag_step = self._speed_bounds(H, a_b, self.regime != ASLEEP)
        if len(mob_idx):
            t_hit, used = self._capture_times(mob_idx, H, T, s, s_off)
            if used.any():
                ug = np.flatnonzero(used)
                self._chk["speed"].append((t, ug, self._vd_step[ug]))
            fr = mob_idx[(self.regime[mob_idx] == FREE)]
            ro = mob_idx[(self.regime[mob_idx] == ROLL)]
            th = dict(zip(mob_idx.tolist(), t_hit.tolist()))
            # slaved clusters: they join the DEM together at the earliest capture time of any member or where the
            # pre-run stopped; then they are run again from the step start to exactly that time
            for cid in list(self._slv):
                c, pre = self._slv[cid], slv_pre[cid]
                mem = c["members"]
                t_j = min(min(th[int(k)] for k in mem), pre["t_stop"])
                self._chk["outside"].append((mem.copy(), x_seen[mem].copy(), pre["s"].copy(), t, t + min(t_j, H),
                                             np.zeros((len(mem), 3)), pre["s"].copy()))
                self._chk["was_out"][mem] = True
                if t_j >= H:                      # it runs through: the pre-run's end state
                    xe, ve, we, he, le = pre["end"]
                    self.x[mem], self.v[mem], self.w[mem] = xe, ve, we
                    self._hist[c["rows"]] = he
                    c["lam"] = le.copy()
                    self._slv_count(pre["n_done"] * len(mem), pre.get("stop"))
                    continue
                # run again from the step start on the pre-run's own grid to the last slaved step before t_j: the
                # same path the envelope was built from; it joins the DEM at exactly that time
                dts = pre.get("dt", 0.0)
                msteps = int(math.floor(t_j / dts * (1 + 1e-12))) if dts > 0 else 0
                self._slv_restore(pre)
                if msteps > 0:
                    self._slv_advance(cid, t, msteps * dts, T, pre, dt_fix=dts, nsteps_fix=msteps)
                self._slv_count(msteps * len(mem), pre.get("stop") if pre["t_stop"] <= t_j else None)
                self._dissolve_slv(cid)
                if msteps > 0:
                    xjoin.append((t + msteps * dts, [int(k) for k in mem]))
            whole_f = np.array([k for k in fr if th[k] >= H], np.int64)
            if len(whole_f):
                self._outside(whole_f, s[whole_f], t, t + H, self.v[whole_f], s[whole_f] - vm0[whole_f] * H)
                self._free_flight(whole_f, t, H)
            whole_r = np.array([k for k in ro if th[k] >= H], np.int64)
            if len(whole_r):
                self._outside(whole_r, s[whole_r], t, t + H)
                failed = self._roll_step(whole_r, t, H, vm0[whole_r], a_b[whole_r], s_off[whole_r])
                self.regime[failed] = ACTIVE
            groups = {}
            for k in list(fr) + list(ro):
                if th[k] < H:
                    groups.setdefault(int(math.floor(th[k] / dt_d)), []).append(int(k))
            for kk in sorted(groups):
                g = groups[kk]
                flyers = np.array([k for k in g if self.regime[k] == FREE], np.int64)
                rollers = np.array([k for k in g if self.regime[k] == ROLL], np.int64)
                if kk > 0 and len(flyers):
                    self._outside(flyers, s[flyers], t, t + kk * dt_d, self.v[flyers], s[flyers] - vm0[flyers] * H)
                    self._free_flight(flyers, t, kk * dt_d)
                if kk > 0 and len(rollers):
                    self._outside(rollers, s[rollers], t, t + kk * dt_d)
                    failed = self._roll_step(rollers, t, kk * dt_d, vm0[rollers], a_b[rollers], s_off[rollers])
                    for k in failed:              # back at the step start: in the DEM for the whole step
                        self.regime[k] = ACTIVE
                    g = [k for k in g if k not in set(failed.tolist())]
                for k in g:
                    self._roll_face.pop(int(k), None)
                    self.regime[k] = ACTIVE
                joins.setdefault(kk, []).extend(g)
        # 4. active grains: full-stiffness DEM over [t, t + H] in chunks; captured grains join at their capture
        # times, sleepers when a DEM grain comes up to them; after each chunk, grains that touch nothing go back to
        # free flight, and grains resting on one wall to the slaved rolling model, up to the earliest time they could
        # come within the capture distance of anything (the rest of the step at most), and join the DEM again then
        ai = np.flatnonzero(self.regime == ACTIVE)
        late = {kk: v for kk, v in joins.items() if kk > 0}
        tj = {kk: t + kk * dt_d for kk in late}
        for q, (tq, g) in enumerate(xjoin):
            late[-1 - q] = g
            tj[-1 - q] = tq
        if len(ai):
            join_at = [(tj[q], q) for q in late]
            heapq.heapify(join_at)
            in_late = set(k for v in late.values() for k in v)
            cur = np.array([k for k in ai if int(k) not in in_late], np.int64)
            t_cur, t_end = t, t + H
            chunk = 20000 * self._dt_dem
            s_full = s.copy()
            s_full[self.regime == ASLEEP] = 0.0
            nkey = -10 ** 9                       # keys of the rejoins (captures are > 0, slaved clusters -1 - q)
            while t_cur < t_end - 1e-15:
                t_next = join_at[0][0] if join_at else t_end
                seg_end = min(t_next, t_cur + chunk, t_end)
                if seg_end > t_cur + 1e-15 and len(cur):
                    cur = self._dem(cur, t_cur, seg_end - t_cur)
                t_cur = seg_end
                while join_at and join_at[0][0] <= t_cur + 1e-15:
                    g = np.array(late.pop(heapq.heappop(join_at)[1]), np.int64)
                    for k in g:
                        self._roll_face.pop(int(k), None)
                    self.regime[g] = ACTIVE
                    cur = np.union1d(cur, g)
                if t_cur < t_end - 1e-15 and len(cur):
                    cur, rejoin = self._early_exit(cur, t_cur, t_end, x_seen, s_full, v_start, t, H)
                    for tq, g in rejoin:
                        late[nkey] = g
                        heapq.heappush(join_at, (tq, nkey))
                        nkey += 1
        return self._verify(s, reg_seen)

    def _outside(self, ks, S, t_from, t_back, V=None, C=None):
        """Record grains about to be advanced outside the DEM from t_from to t_back (the step's end, or when they
        join the DEM): position now, a bound S on their displacement, and for free flight its velocity V now and a
        bound C on the deviation from the straight line (|x(t) - x - V (t - t_from)| <= C; else V = 0, C = S)."""
        ks = np.asarray(ks, np.int64)
        if len(ks):
            S = np.asarray(S, float).copy()
            V = np.zeros((len(ks), 3)) if V is None else np.asarray(V, float).copy()
            C = S.copy() if C is None else np.minimum(np.asarray(C, float), S)
            self._chk["outside"].append((ks.copy(), self.x[ks].copy(), S, float(t_from), float(t_back), V, C))
            self._chk["was_out"][ks] = True

    def _verify(self, s, reg_seen):
        """The step's checks (see step). Returns (grains to put in the DEM from the start, new speed factor or
        None)."""
        chk = self._chk
        bad = set(int(k) for k in chk["roll_bad"])
        vf = None
        for tc, idx, vb in chk["speed"]:
            obs = np.zeros(len(idx))
            for ta, tb, vg in chk["vmax"]:
                if tb > tc + 1e-15:
                    obs = np.maximum(obs, vg[idx])
            over = obs > vb
            # a bound of ~0 (a grain taken to be at rest) that no factor can mend: that grain goes into the DEM
            # from the start instead, if it was not there already (then the bounds are scaled)
            zero = over & (vb <= 1e-6 * obs) & (reg_seen[idx] != ACTIVE)
            bad.update(int(k) for k in idx[zero])
            sc = over & ~zero
            if sc.any():
                vf = max(vf or 0.0, self._vdem_fac * 1.25 * float((obs[sc] / np.maximum(vb[sc], 1e-300)).max()))
        # a grain that entered the DEM during the step (not active at its start): up to then its capture time (or,
        # asleep, its static certificate) kept it clear of every other grain's bounded motion; from then on it is
        # within dmax of where it entered, and must be clear of every grain outside the DEM at the same time
        self._missed_end()
        bad |= chk["missed"]
        cand = np.flatnonzero(chk["in_dem"] & (reg_seen != ACTIVE))
        if len(cand) and chk["outside"]:
            K = np.concatenate([r[0] for r in chk["outside"]])
            cl = chk.get("slv_of", {})            # grains of one slaved cluster: their mutual contacts were resolved
            KC = np.array([cl.get(int(k), -1) for k in K], np.int64)       # by the slaved model, then the DEM
            X = np.vstack([r[1] for r in chk["outside"]])
            S = np.concatenate([r[2] for r in chk["outside"]])
            TA = np.concatenate([np.full(len(r[0]), r[3]) for r in chk["outside"]])
            TB = np.concatenate([np.full(len(r[0]), r[4]) for r in chk["outside"]])
            V = np.vstack([r[5] for r in chk["outside"]])
            C = np.concatenate([r[6] for r in chk["outside"]])
            xr, t_in = chk["x_ref"], chk["t_in"]
            Dg = chk["dmax"][cand]
            tree = cKDTree(X)
            rmax = float(self.rad.max())
            for q, g in enumerate(cand):
                for idx in tree.query_ball_point(xr[g], 2 * rmax + float(S.max()) + Dg[q] + self.d_cap + 1e-12):
                    k = int(K[idx])
                    if k == int(g) or (KC[idx] >= 0 and KC[idx] == cl.get(int(g), -2)) \
                            or TB[idx] <= t_in[g] + 1e-15 or TA[idx] > t_in[g] + 1e-15:
                        continue
                    # closest approach of the record's straight line to where g entered, while both are out of
                    # their reach of each other's treatment: [max(from, t_in), to]
                    d0 = xr[g] - X[idx]
                    v2 = float(V[idx] @ V[idx])
                    ta, tb = max(TA[idx], t_in[g]) - TA[idx], TB[idx] - TA[idx]
                    tau = min(max(float(d0 @ V[idx]) / v2, ta), max(ta, tb)) if v2 > 0 else 0.0
                    gap = float(np.linalg.norm(d0 - V[idx] * tau)) - self.rad[g] - self.rad[k]
                    if gap - C[idx] - Dg[q] <= self.d_cap:
                        bad.add(int(g))
                        break
        self.n_events["checked_steps"] = self.n_events.get("checked_steps", 0) + 1
        return bad, vf

    def _early_exit(self, cur, t_now, t_end, x_start, s_full, v_start, t_step, H):
        """Grains in the DEM that touch nothing (or rest on one wall only) leave it: exact free flight (or the slaved
        rolling model) up to a lower bound on the earliest time they could come within the capture distance of
        anything, where they join the DEM again (or to t_end if that is earlier). The bounds, all time-resolved:
        * walls along their normals: the gap is g0 + u tau - a tau^2 / 2 at the least (u the velocity away from the
          wall, a the grain's acceleration bound); for a rolling grain g0 - |v| tau - a tau^2 / 2 - offset (its
          rolling envelope bounds the distance, not the direction);
        * grains still in the DEM: current positions, their speed and acceleration bounds for the rest of the step
          (local, never below the step-start bounds; recorded and checked against the actual speeds);
        * grains that left the DEM earlier in this step: within S of where they left it until they rejoin, then
          their step-start speed bounds (recorded and checked);
        * free and sleeping grains outside the DEM all step: straight-line relative motion plus curvature bounds;
        * other grains outside the DEM (late joiners, slaved clusters): their step-start positions and whole-step
          motion bounds; once they join the DEM the step's verification checks them against every outside record.
        Rejoin times are rounded down to a grid (earlier is always safe) so that fewer DEM segments are cut.
        Returns the grains still in the DEM and [(time, [grain]), ...] for those coming back later in the step."""
        if self.all_dem:
            return cur, []
        in_cur = np.zeros(self.N, bool)
        in_cur[cur] = True
        touch = self._touching(in_cur)
        busy = set()
        nw = {}
        npr = set()
        for i, j in touch:
            busy.add(i)
            if j < 0:
                nw.setdefault(i, []).append(-1 - j)
            else:
                busy.add(j)
                npr.add(i)
                npr.add(j)
        rem = t_end - t_now
        T = self.env.t0_flight + t_now
        vm = np.linalg.norm(self.v, axis=1)
        s_rem = self._motion_bound(rem, T)
        a_k = np.maximum((s_rem - vm * rem) * 2 / (rem * rem), 0.0)
        d_cap = self.d_cap
        chk = self._chk
        exited = chk["exited"]
        vdc, agc = self._speed_bounds(rem, a_k, in_cur)
        vdc = np.maximum(vdc, self._vd_step)
        agc = np.maximum(agc, self._ag_step)
        ref = np.where(in_cur[:, None], self.x, x_start)
        off_any = self._off_any()
        s_ref = np.where(in_cur, vdc * rem + 0.5 * agc * rem * rem + off_any, s_full)
        for j, (te, xe, Se, tr, Ve, Ae) in exited.items():
            if not in_cur[j]:
                ref[j] = xe
                s_ref[j] = Se + (Ve * rem + 0.5 * Ae * rem * rem + off_any[j] if tr < t_end else 0.0)
        t_min = 1000 * self._dt_dem                # leaving for less than this is not worth it
        qg = 250 * self._dt_dem                    # grid of the rejoin times

        def t_iso(gap, v, a):
            g = gap - d_cap
            if g <= 0:
                return 0.0
            if a <= 0:
                return g / v if v > 0 else np.inf
            return 2 * g / (v + math.sqrt(v * v + 2 * a * g))

        def t_lin(d0, dv, c):
            q0 = float(d0 @ d0)
            if q0 <= c * c:
                return 0.0
            b = float(d0 @ dv)
            a2 = float(dv @ dv)
            if a2 == 0.0 or b >= 0.0:
                return np.inf
            disc = b * b - a2 * (q0 - c * c)
            if disc < 0:
                return np.inf
            return (-b - math.sqrt(disc)) / a2
        tree = None
        out, rollers = [], []
        tc_of, sk_of, off_of, vk_of = {}, {}, {}, {}
        used = {}                                  # grains outside whose step-start speed bound was relied on
        for k in cur:
            k = int(k)
            if k in self._noexit:
                continue
            own = -1
            if k in busy:
                if not self.use_roll or k in npr or len(nw.get(k, [])) != 1:
                    continue
                own = nw[k][0]
            off_k = self._roll_off(np.array([k]), np.array([own]))[0] if own >= 0 else 0.0
            vk = vm[k]
            if own >= 0:                          # it starts rolling at r w x n (_try_roll): the larger speed
                vk = max(vk, float(np.linalg.norm(self.rad[k] * np.cross(self.w[k], self._wall_normal(own)))))
            tc = np.inf
            for f in range(6):
                if f == own:
                    continue
                ax = f // 2
                g0 = (self.x[k, ax] if f % 2 == 0 else self.L[ax] - self.x[k, ax]) - self.rad[k] - d_cap - off_k
                u = self.v[k, ax] if f % 2 == 0 else -self.v[k, ax]
                if g0 <= 0:
                    tc = 0.0
                    break
                a = a_k[k]
                if own >= 0:                      # rolling: the envelope bounds the distance, not the direction
                    tw = 2 * g0 / (vk + math.sqrt(vk * vk + 2 * a * g0)) if vk > 0 or a > 0 else np.inf
                elif a > 0:
                    r = math.sqrt(u * u + 2 * a * g0)
                    tw = 2 * g0 / (r - u) if u <= 0 else (u + r) / a
                else:
                    tw = g0 / -u if u < 0 else np.inf
                tc = min(tc, tw)
            if tc < t_min:
                continue
            if tree is None:
                tree = cKDTree(ref)
            s_k = vk * rem + 0.5 * a_k[k] * rem * rem + off_k
            mine = {}
            for j in tree.query_ball_point(self.x[k], 2 * self.rad.max() + d_cap + s_k + float(s_ref.max())):
                if j == k:
                    continue
                rsum = self.rad[j] + self.rad[k]
                if in_cur[j]:
                    gap = float(np.linalg.norm(self.x[j] - self.x[k])) - rsum - off_k - off_any[j]
                    tc = min(tc, t_iso(gap, vk + vdc[j], a_k[k] + agc[j]))
                elif j in exited:
                    # within S of where it left until it rejoins; then in the DEM with the speed bound it had there
                    # when it left (recorded and checked against its speed after rejoining)
                    te, xe, Se, tr, Ve, Ae = exited[j]
                    gap = float(np.linalg.norm(xe - self.x[k])) - rsum - off_k - Se
                    t1 = t_iso(gap, vk, a_k[k])
                    if tr < t_end and t1 > tr - t_now:          # it may be back in the DEM by then
                        t2 = t_iso(gap - off_any[j], vk + Ve, a_k[k] + Ae)
                        t1 = max(tr - t_now, t2)
                        mine[j] = Ve
                    tc = min(tc, t1)
                elif self.regime[j] in (FREE, ASLEEP):
                    if self.regime[j] == FREE:
                        a_j = max((s_full[j] - float(np.linalg.norm(v_start[j])) * H) * 2 / (H * H), 0.0)
                        d0 = x_start[j] + v_start[j] * (t_now - t_step) - self.x[k]
                        dv = v_start[j] - self.v[k]
                        curv = 0.5 * a_j * H * H + 0.5 * a_k[k] * rem * rem + off_k
                    else:
                        d0 = self.x[j] - self.x[k]
                        dv = -self.v[k]
                        curv = 0.5 * a_k[k] * rem * rem + off_k
                    if own >= 0:                  # a roller may turn: no direction assumed
                        tc = min(tc, t_iso(float(np.linalg.norm(d0)) - rsum - curv + 0.5 * a_k[k] * rem * rem,
                                           vk + float(np.linalg.norm(dv + self.v[k])), a_k[k]))
                    else:
                        tc = min(tc, t_lin(d0, dv, rsum + d_cap + curv))
                else:
                    gap = float(np.linalg.norm(x_start[j] - self.x[k])) - rsum - off_k - s_full[j]
                    tc = min(tc, t_iso(gap, vk, a_k[k]))
                if tc < t_min:
                    break
            if tc < t_min:
                continue
            if tc < rem:
                tg = t_step + math.floor((t_now + tc - t_step) / qg) * qg - t_now
                if tg >= t_min:
                    tc = tg
            else:
                tc = rem
            used.update(mine)
            tc_of[k] = tc
            off_of[k] = off_k
            vk_of[k] = vk
            sk_of[k] = vk * tc + 0.5 * a_k[k] * tc * tc + off_k
            (rollers if own >= 0 else out).append(k)
        rejoin = []
        left = []
        xe_of = {k: self.x[k].copy() for k in out}
        for k in rollers:
            tc = tc_of[k]
            if self._try_roll(k, nw[k][0], T, tc):
                kk = np.array([k], np.int64)
                xe_of[k] = self.x[k].copy()
                self._outside(kk, [sk_of[k]], t_now, t_now + tc)
                failed = self._roll_step(kk, t_now, tc, np.array([vk_of[k]]), a_k[kk], np.array([off_of[k]]))
                if len(failed):
                    chk["outside"].pop()          # it stays in the DEM: no record
                    self.regime[k] = ACTIVE
                    continue
                left.append(k)
        n_roll = len(left)
        for k in out:
            kk = np.array([k], np.int64)
            self.regime[k] = FREE
            tq = tc_of[k]
            self._outside(kk, [sk_of[k]], t_now, t_now + tq, self.v[kk], [0.5 * a_k[k] * tq * tq + 1e-12])
            self._free_flight(kk, t_now, tq)
            left.append(k)
        if not left:
            return cur, []
        # where each one left and how far it may get: the record for the others' checks, and its largest distance
        # from where it entered the DEM (the step's verification uses it for grains not active at the start)
        xs = chk["x_ref"]
        for k in left:
            xe, Se = xe_of[k], sk_of[k]
            tr = t_now + tc_of[k] if tc_of[k] < rem else t_end
            exited[k] = (t_now, xe, Se, tr, vdc[k], agc[k])
            chk["dmax"][k] = max(chk["dmax"][k], float(np.linalg.norm(xe - xs[k])) + Se)
            if tr < t_end:
                rejoin.append((tr, [k]))
        # every grain the exits took as in the DEM, the leavers too: their speed after a rejoin is what the others
        # relied on (a leaver that stays out has no DEM speed to check)
        chk["speed"].append((t_now, cur.copy(), vdc[cur]))
        cur = np.setdiff1d(cur, np.array(left, np.int64))
        if used:
            ug = np.array(sorted(used), np.int64)
            chk["speed"].append((t_now, ug, np.array([used[j] for j in ug])))
        self.n_events["roll_reentries"] = self.n_events.get("roll_reentries", 0) + n_roll
        self.n_events["early_exits"] = self.n_events.get("early_exits", 0) + len(left) - n_roll
        self.n_events["early_exit_rejoins"] = self.n_events.get("early_exit_rejoins", 0) + len(rejoin)
        return cur, rejoin

    def _off_any(self):
        """The largest rolling offset each grain could have on any wall (rocking deflection + 1 nm)."""
        if getattr(self, "_off_any_c", None) is None:
            self._off_any_c = np.array([max(self._params(k, -1 - f)[dem.P_MUR] * self._params(k, -1 - f)[dem.P_REFF]
                                            for f in range(6)) + 1e-9 for k in range(self.N)])
        return self._off_any_c

    def _roll_off(self, ri, faces):
        """Constant part of the rolling envelope: the elastic rocking deflection (below xi_crit each way) and 1 nm."""
        rows = np.array([self._key[(int(k), -1 - int(f))] for k, f in zip(ri, faces)], np.int64)
        p = self._prm[rows]
        return p[:, dem.P_MUR] * p[:, dem.P_REFF] + 1e-9             # 2 xi_crit = mu_r R

    def _roll_bound(self, ri, H, T):
        """Most a grain rolling on its wall can move in H (returns the bound and its constant part): it stays on the
        wall (the slaved model holds its normal coordinate), and along it moves at most v t + (5/7) max(a_t - a_roll,
        0) t^2 / 2 (rigid rolling, a_t the largest load along the wall over the step with the certificate margin, a_roll
        its rolling limit), plus the elastic rocking deflection. Checked against the actual path every rolling step
        (roll1 exc)."""
        faces = np.array([self._roll_face[int(k)] for k in ri], np.int64)
        worst_at = np.zeros(len(ri))
        rows = np.array([self._key[(int(k), -1 - int(f))] for k, f in zip(ri, faces)], np.int64)
        p = self._prm[rows]
        n = np.zeros((len(ri), 3))
        n[np.arange(len(ri)), faces // 2] = np.where(faces % 2 == 0, 1.0, -1.0)
        a_roll = np.full(len(ri), np.inf)
        for Ts in (T, T + 0.5 * H, T + H):
            F, Tq = self._body_force(ri, Ts)
            Fn = -(F * n).sum(1)
            Ft = F + Fn[:, None] * n
            worst_at = np.maximum(worst_at, np.linalg.norm(Ft, axis=1) / self.m[ri])
            a_roll = np.minimum(a_roll, p[:, dem.P_MUR] * np.abs(Fn + 2 * p[:, dem.P_FC]) / self.m[ri])
        excess = np.maximum(worst_at * (1 + self.margin) - a_roll * (1 - self.margin), 0.0)
        off = self._roll_off(ri, faces)
        vm = np.linalg.norm(self.v[ri], axis=1)
        return vm * H + 0.5 * (5.0 / 7.0) * excess * H * H + off, off

    def _speed_bounds(self, H, a_b, mask=None):
        """Speed bound over the next H for each grain whose direction contacts can change (a grain in the DEM): the
        fastest grain it could meet during H (grains whose reaches can overlap, transitively, iterated until the
        groups stop growing) times sqrt(m_max / m_min) (the most a collision between two of them can give), plus the
        group's acceleration bound times H. So a slow cluster far from fast grains keeps a slow bound. Checked after
        the step against every grain's actual speed (_verify), and scaled up by the redo factor if one was exceeded.
        Returns the bounds and the group acceleration bounds (zero outside the mask)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        idx = np.arange(self.N) if mask is None else np.flatnonzero(mask)
        vb_all, ag_all = np.zeros(self.N), np.zeros(self.N)
        if not len(idx):
            return vb_all, ag_all
        u = np.linalg.norm(self.v[idx], axis=1)
        a = np.asarray(a_b)[idx]
        fac = math.sqrt(float(self.m.max() / self.m.min()))
        x, r = self.x[idx], self.rad[idx]
        n = len(idx)
        tree = cKDTree(x)
        vb, amax = u * fac + a * H, a.copy()
        for _ in range(12):
            reach = vb * H + 0.5 * amax * H * H
            pr = tree.query_pairs(2 * float(r.max()) + 2 * float(reach.max()) + self.d_cap + 1e-12,
                                  output_type="ndarray")
            if len(pr):
                d = np.linalg.norm(x[pr[:, 0]] - x[pr[:, 1]], axis=1) - r[pr[:, 0]] - r[pr[:, 1]]
                pr = pr[d <= reach[pr[:, 0]] + reach[pr[:, 1]] + self.d_cap]
            if len(pr):
                g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
                _, lab = connected_components(g, directed=False)
            else:
                lab = np.arange(n)
            um = np.zeros(lab.max() + 1)
            np.maximum.at(um, lab, u)
            am = np.zeros(lab.max() + 1)
            np.maximum.at(am, lab, a)
            nvb, namax = um[lab] * fac + am[lab] * H, am[lab]
            if np.array_equal(nvb, vb) and np.array_equal(namax, amax):
                break
            vb, amax = nvb, namax
        else:                                           # not settled: the global bound
            vb = np.full(n, float(u.max()) * fac + float(a.max()) * H)
            amax = np.full(n, float(a.max()))
        vb_all[idx], ag_all[idx] = vb * self._vdem_fac, amax
        return vb_all, ag_all

    def _capture_times(self, free, H, T, s, s_off):
        """For each free or rolling grain, a lower bound on the time at which it could first come within the capture
        distance d_cap of a wall or another grain during the step.
        * walls: for a free grain the gap along the wall normal is g0 + u tau - a tau^2/2 at the least (u its
          velocity away from the wall, a its acceleration bound); a rolling grain may turn, so g0 - |v| tau -
          a tau^2/2 - offset (its rolling envelope);
        * another free grain or a sleeping one: their relative motion is a straight line d0 + dv tau plus a curvature
          of at most (a_i + a_j) tau^2 / 2, so the distance is at least |d0 + dv tau| - (a_i + a_j) H^2 / 2;
        * a rolling grain: its envelope |v| tau + a tau^2 / 2 + offset (rolling bound);
        * a grain in the DEM: no direction is assumed, only its local speed bound (contacts can turn it).
        Sleeping grains do not move (their certificate holds for the step; one that joins the DEM is checked
        afterwards, see _verify). The speed bound of each grain in the DEM is local (_speed_bounds) and checked
        against its actual speed afterwards. Returns the times and which DEM grains' bounds were used."""
        d_cap = self.d_cap
        vm = np.linalg.norm(self.v, axis=1)
        a_bound = np.maximum((s - s_off - vm * H) * 2 / (H * H), 0.0)   # per-grain acceleration bound
        vd, ag = self._vd_step, self._ag_step
        off_any = self._off_any()
        out = np.full(len(free), np.inf)
        used = np.zeros(self.N, bool)
        act = self.regime == ACTIVE
        vd_act = float(vd[act].max()) if act.any() else 0.0

        def t_iso(gap, v, a):
            g = gap - d_cap
            if g <= 0:
                return 0.0
            if a <= 0:
                return g / v if v > 0 else np.inf
            return (-v + math.sqrt(v * v + 2 * a * g)) / a

        def t_lin(d0, dv, rsum, A):
            c = rsum + d_cap + 0.5 * A * H * H
            q0 = float(d0 @ d0)
            if q0 <= c * c:
                return 0.0
            b = float(d0 @ dv)
            a2 = float(dv @ dv)
            if a2 == 0.0 or b >= 0.0:
                return np.inf                                   # not approaching (the curvature is in c)
            disc = b * b - a2 * (q0 - c * c)
            if disc < 0:
                return np.inf
            return (-b - math.sqrt(disc)) / a2
        tree = cKDTree(self.x)
        reach = 2 * self.rad.max() + d_cap + 2 * max(float(s.max()), vd_act * H) + 1e-9
        slv_k = self.regime == SLAV
        vm_e = np.where(slv_k, 0.0, vm)
        for q, k in enumerate(free):
            best = np.inf
            k_roll = self.regime[k] == ROLL
            k_slv = bool(slv_k[k])
            own = self._roll_face.get(int(k), -1) if k_roll else -1
            for f in range(6):
                if f == own or k_slv:
                    continue
                ax = f // 2
                g0 = (self.x[k, ax] if f % 2 == 0 else self.L[ax] - self.x[k, ax]) - self.rad[k] - d_cap - s_off[k]
                u = self.v[k, ax] if f % 2 == 0 else -self.v[k, ax]
                if g0 <= 0:
                    best = 0.0
                    break
                a = a_bound[k]
                if k_roll:                        # the envelope bounds the distance rolled, not its direction
                    tw = 2 * g0 / (vm[k] + math.sqrt(vm[k] * vm[k] + 2 * a * g0)) if vm[k] > 0 or a > 0 else np.inf
                elif a > 0:
                    tw = (u + math.sqrt(u * u + 2 * a * g0)) / a
                else:
                    tw = g0 / -u if u < 0 else np.inf
                best = min(best, tw)
            for jj in tree.query_ball_point(self.x[k], reach):
                if jj == k or best == 0.0:
                    continue
                rsum = self.rad[jj] + self.rad[k]
                rj = self.regime[jj]
                if k_slv and (rj == ASLEEP or self._slv_of.get(int(jj)) == self._slv_of.get(int(k))):
                    continue                      # checked exactly by the pre-run (static, or the same cluster)
                if rj == ACTIVE:                  # (it may roll on a wall later: its rocking offset too)
                    gap = float(np.linalg.norm(self.x[jj] - self.x[k])) - rsum - s_off[k] - off_any[jj]
                    tp = t_iso(gap, vm_e[k] + vd[jj], a_bound[k] + ag[jj])
                    used[jj] = True
                elif k_roll or k_slv or rj == ROLL or rj == SLAV:
                    gap = float(np.linalg.norm(self.x[jj] - self.x[k])) - rsum - s_off[k] - s_off[jj]
                    vj = 0.0 if rj == ASLEEP else vm_e[jj]
                    tp = t_iso(gap, vm_e[k] + vj, a_bound[k] + a_bound[jj])
                elif rj == FREE:
                    tp = t_lin(self.x[jj] - self.x[k], self.v[jj] - self.v[k], rsum, a_bound[k] + a_bound[jj])
                else:
                    tp = t_lin(self.x[jj] - self.x[k], -self.v[k], rsum, a_bound[k])
                best = min(best, tp)
            out[q] = best
        return out, used

    def _skip_asleep(self, t, t_out=None):
        """Everything is asleep: advance time without stepping while every sleeper's certificate holds, checked at
        the same sample points the stepping would use (every dt_s), up to the next load breakpoint or fast load.
        Returns the new time (t if nothing can be skipped)."""
        T = self.env.t0_flight + t
        dts = self.base.dt_base
        t_lim = min(self.duration, self.loads.next_breakpoint(T) - self.env.t0_flight, t + 2.0,
                    t_out if t_out is not None and t_out > t else self.duration)
        if self.loads.fast_window(T, T + dts) or t_lim <= t + 2 * dts:
            return t
        n = int((t_lim - t) / dts)
        Ts = T + dts * np.arange(1, n + 1)
        asl = np.array([k for k in np.flatnonzero(self.regime == ASLEEP) if int(k) in self._sleep_face], np.int64)
        ok_until = n
        if len(asl):
            faces = np.array([self._sleep_face[k] for k in asl])
            for q, Tq in enumerate(Ts):
                F, Tqq = self._body_force(asl, Tq)
                if float(self._margins(asl, faces, F, Tqq)[0].max()) > 1 - self.margin:
                    ok_until = q
                    break
        for iid, isl in self._islands.items():
            for q in range(min(ok_until, n)):
                if float(self._island_ratio(isl, Ts[q]).max()) > 1 - self.margin:
                    ok_until = q
                    break
        if ok_until < 1:
            return t
        return t + ok_until * dts

    def _wake(self, k, T):
        face = self._sleep_face.pop(k)
        self._last_face = face
        row = self._key[(int(k), -1 - face)]
        self._place_at_equilibrium(k, face, row, T)
        self.v[k] = 0.0
        self.w[k] = 0.0
        self.n_events["wake"] += 1

    def _free_flight(self, free, t, H):
        b = self.base
        b.frozen[:] = True
        b.frozen[free] = False
        b.X[:], b.V[:] = self.x, self.v
        b.X[free], b.V[free] = self.x[free], self.v[free]
        b.L[free] = self.w[free] * self.I[free, None]
        b.pos = b.X[b.body] + b.off
        b.env.update(t)
        a_app = self.loads.accel(np.array([b.env.t0_flight + t]), side="+")[0]
        b._a_app_now = a_app
        if b.env.profile is not None:
            b._W_drift = 2 * math.pi * b.env.profile.spin_hz(b.env.t0_flight + t + 0.5 * H)
        F0 = b._forces_pos(b._mono_vel())
        act = b._kick(F0, a_app, 0.5 * H)
        b._drift(H, act)
        b.env.update(t + H)
        F1 = b._forces_pos(b._mono_vel())
        a1 = self.loads.accel(np.array([b.env.t0_flight + t + H]), side="-")[0]
        b._kick(F1, a1, 0.5 * H, brownian=True)
        self.x[free], self.v[free] = b.pos[free], b.V[free]
        self.w[free] = b.L[free] / self.I[free, None]

    # ------------------------------------------------------------------ DEM of the active grains
    def _missed(self, ai, x, ci, cj, rows):
        """Grains entering the DEM after being advanced outside it in this step that overlap a wall or grain they have
        no contact with: a contact formed outside the DEM, which no bound allows. The step's verification redoes the
        step with them in the DEM throughout. (Grains placed overlapping in an initial state form their contacts in
        the DEM's first step, as they should.)"""
        if not len(rows):
            return
        wo = self._chk["was_out"]
        pp = cj >= 0
        sel = wo[ai[ci]] | (pp & wo[ai[np.maximum(cj, 0)]])
        if not sel.any():
            return
        ci, cj, rows, pp = ci[sel], cj[sel], rows[sel], pp[sel]
        dl = np.empty(len(rows))
        if pp.any():
            dl[pp] = self.rad[ai[ci[pp]]] + self.rad[ai[cj[pp]]] - np.linalg.norm(x[ci[pp]] - x[cj[pp]], axis=1)
        wq = ~pp
        if wq.any():
            f = -1 - cj[wq]
            ax = f // 2
            xa = x[ci[wq], ax]
            dl[wq] = self.rad[ai[ci[wq]]] - np.where(f % 2 == 0, xa, self.L[ax] - xa)
        miss = (dl > 1e-12) & (self._hist[rows, dem.H_TOUCH] < 0.5)
        if miss.any():
            self._chk["missed"].update(int(k) for k in ai[ci[miss]])
            self._chk["missed"].update(int(k) for k in ai[cj[miss & pp]])

    def _missed_end(self):
        """The same check at the end of the step for the grains advanced outside the DEM and still outside it."""
        out = np.flatnonzero(self._chk["was_out"] & (self.regime != ACTIVE))
        if not len(out):
            return
        x, r = self.x, self.rad
        xo, ro = x[out], r[out][:, None]
        gaps = np.hstack([xo - ro, self.L[None, :] - xo - ro])        # columns: faces 0, 2, 4, 1, 3, 5
        for q, c in zip(*np.nonzero(gaps < -1e-12)):
            k, f = int(out[q]), (2 * c if c < 3 else 2 * (c - 3) + 1)
            row = self._key.get((k, -1 - f))
            if row is None or self._hist[row, dem.H_TOUCH] < 0.5:
                self._chk["missed"].add(k)
        lists = cKDTree(x).query_ball_point(xo, r[out] + float(r.max()))
        ii = np.repeat(out, [len(l) for l in lists])
        jj = np.fromiter((j for l in lists for j in l), np.int64, len(ii))
        ov = r[ii] + r[jj] - np.linalg.norm(x[jj] - x[ii], axis=1)
        for k, j in zip(ii[(ov > 1e-12) & (ii != jj)], jj[(ov > 1e-12) & (ii != jj)]):
            k, j = int(k), int(j)
            row = self._key.get((min(k, j), max(k, j)))
            if row is None or self._hist[row, dem.H_TOUCH] < 0.5:
                self._chk["missed"].update((k, j))

    def _candidates(self, ai, skin):
        loc = {int(g): k for k, g in enumerate(ai)}
        ci, cj, rows = [], [], []
        x = self.x[ai]
        if len(ai) > 1:
            pr = cKDTree(x).query_pairs(2 * self.rad[ai].max() + skin, output_type="ndarray")
            if len(pr):
                d = np.linalg.norm(x[pr[:, 0]] - x[pr[:, 1]], axis=1)
                ok = d < self.rad[ai][pr[:, 0]] + self.rad[ai][pr[:, 1]] + skin
                for a, c in pr[ok]:
                    gi, gj = int(ai[a]), int(ai[c])
                    if gi > gj:
                        gi, gj, a, c = gj, gi, c, a
                    rows.append(self._row(gi, gj))
                    ci.append(loc[gi])
                    cj.append(loc[gj])
        for f in range(6):
            axf = f // 2
            gap = x[:, axf] if f % 2 == 0 else self.L[axf] - x[:, axf]
            for k in np.flatnonzero(gap < self.rad[ai] + skin):
                rows.append(self._row(int(ai[k]), -1 - f))
                ci.append(int(k))
                cj.append(-1 - f)
        return np.array(ci, np.int64), np.array(cj, np.int64), np.array(rows, np.int64)

    def _dem(self, ai, t, H):
        """Full-stiffness DEM of the grains ai over [t, t + H]; returns the grains in it at the end. Sleeping grains are
        static obstacles: one joins (woken at its exact equilibrium, at rest; a cluster whole) as soon as a DEM grain
        comes within 2 d_cap of it, before any contact can form. Between contact-list builds each DEM grain may move at
        most min(half the skin, its clearance to the nearest sleeper - 2 d_cap), so none is missed. Records each
        chunk's largest speed, and each grain's entry time, entry position and largest distance from it (for the
        step's verification)."""
        T0 = self.env.t0_flight + t
        ai = np.asarray(ai, np.int64)
        vrel = 2 * float(np.linalg.norm(self.v[ai], axis=1).max()) if len(ai) else 0.0
        dt = min(self._dt_dem, self._hertz_dt(vrel))
        n = max(1, int(math.ceil(H / dt)))
        dt = H / n
        skin = 0.5 * float(self.rad.min())
        Ts_all = T0 + np.arange(1, n + 1) * dt
        acc_all = self.loads.accel(Ts_all, side="-")
        Om_all = self.loads.omega(Ts_all)
        Omd_all = self.loads.omega_dot(Ts_all)
        chk = self._chk
        vmax2 = np.zeros(1)
        vchunk = np.zeros(self.N)
        done = 0
        while done < n:
            Tn = T0 + done * dt
            ai, lim = self._watch(ai, Tn, skin)
            new = ai[~chk["in_dem"][ai]]
            if len(new):
                chk["in_dem"][new] = True
                chk["t_in"][new] = t + done * dt
                chk["x_ref"][new] = self.x[new]
                chk["dmax"][new] = 0.0
            x = np.ascontiguousarray(self.x[ai])
            v = np.ascontiguousarray(self.v[ai])
            w = np.ascontiguousarray(self.w[ai])
            m, I = self.m[ai], self.I[ai]
            gdrag = self.gdrag[ai]
            Fext = np.ascontiguousarray(self._F_lr[ai])
            xs = np.ascontiguousarray(chk["x_ref"][ai])
            dmax = chk["dmax"][ai].copy()
            vmaxg = np.zeros(len(ai))
            ci, cj, rows = self._candidates(ai, skin)
            self._missed(ai, x, ci, cj, rows)
            prm = np.ascontiguousarray(self._prm[rows])
            hist = np.ascontiguousarray(self._hist[rows])
            F = np.zeros_like(x)
            Tq = np.zeros_like(x)
            Fn = np.zeros(len(rows))
            Tn1 = np.array([Tn])
            dem.compute_forces(x, v, w, m, I, self.L, ci, cj, prm, hist, 0.0, self.loads.accel(Tn1, side="+")[0],
                               self.loads.omega(Tn1)[0], self.loads.omega_dot(Tn1)[0], self.off, gdrag, F, Tq, Fn,
                               Fext, Tn)
            xb = x.copy()
            kk = dem.run_steps(x, v, w, m, I, self.L, ci, cj, prm, hist, dt, n - done, acc_all[done:], Om_all[done:],
                               Omd_all[done:], self.off, gdrag, F, Tq, xb, lim, Fn, Fext, Tn, xs, dmax, vmax2,
                               vmaxg)
            vchunk[ai] = np.maximum(vchunk[ai], np.sqrt(vmaxg))
            self._hist[rows] = hist
            self.x[ai], self.v[ai], self.w[ai] = x, v, w
            chk["dmax"][ai] = dmax
            done += kk
            self.n_events["dem_steps"] += kk
            self.n_events["active_grain_steps"] += kk * len(ai)
            # where the full model's effort goes (counted at the start of each segment): grains touching other
            # grains, grains touching only walls, grains touching nothing
            cls = self._dem_classes(ai)
            for key, cnt in cls.items():
                self.n_events[key] = self.n_events.get(key, 0) + kk * cnt
            if done < n:
                self.n_events["rebuilds"] += 1
        chk["vmax"].append((t, t + H, vchunk))
        return ai

    def _dem_classes(self, ai):
        mask = np.zeros(self.N, bool)
        mask[ai] = True
        pair = np.zeros(self.N, bool)
        wall = np.zeros(self.N, bool)
        for i, j in self._touching(mask):
            if j >= 0:
                pair[i] = pair[j] = True
            else:
                wall[i] = True
        return {"dem_gs_in_clusters": int((pair & mask).sum()), "dem_gs_on_walls_only": int((wall & ~pair & mask).sum()),
                "dem_gs_touching_nothing": int((~wall & ~pair & mask).sum())}

    def _watch(self, ai, T, skin):
        """Sleepers within 2 d_cap of a DEM grain join it (repeated: a joined grain may be that close to another).
        Returns the DEM set and each DEM grain's movement allowance until the next list build."""
        d2 = 2 * self.d_cap
        rmax = float(self.rad.max())
        while True:
            lim = np.full(len(ai), 0.5 * skin)
            sl = np.flatnonzero(self.regime == ASLEEP)
            if not len(sl):
                return ai, lim
            xs = self.x[sl]
            c = getattr(self, "_sl_tree", None)        # sleepers do not move: reuse the tree while the set is the same
            if c is None or not (np.array_equal(c[0], sl) and np.array_equal(c[1], xs)):
                c = self._sl_tree = (sl, xs.copy(), cKDTree(xs))
            tree = c[2]
            join = set()
            for q, lst in enumerate(tree.query_ball_point(self.x[ai], 2 * rmax + skin)):
                if not lst:
                    continue
                jj = sl[np.array(lst, np.int64)]
                gap = np.linalg.norm(self.x[jj] - self.x[ai[q]], axis=1) - self.rad[jj] - self.rad[ai[q]]
                lim[q] = min(lim[q], float(gap.min()) - d2)
                join.update(int(j) for j in jj[gap <= d2])
            if not join:
                return ai, np.maximum(lim, 0.0)
            new = []
            for j in sorted(join):
                if self.regime[j] != ASLEEP:
                    continue
                if j in self._island_of:
                    mem = [int(g) for g in self._islands[self._island_of[j]]["members"]]
                    self._wake_island(self._island_of[j], T)
                    new += mem
                else:
                    if j in self._sleep_face:
                        self._wake(j, T)
                    self.v[j] = 0.0
                    self.w[j] = 0.0
                    self.regime[j] = ACTIVE
                    new.append(j)
            self.n_events["dem_joins"] = self.n_events.get("dem_joins", 0) + len(new)
            ai = np.union1d(ai, np.array(new, np.int64))

    # ------------------------------------------------------------------ regime changes after a step
    def _touching(self, mask=None):
        """Contacts that touch, as (i, j) keys; with a grain mask, only those involving a masked grain."""
        nr = len(self._ckeys)
        kij = self._kij[:nr][self._hist[:nr, dem.H_TOUCH] > 0.5]
        if mask is not None:
            kij = kij[mask[kij[:, 0]] | ((kij[:, 1] >= 0) & mask[np.maximum(kij[:, 1], 0)])]
        return [(int(a), int(b)) for a, b in kij.tolist()]

    def _classify(self, t):
        if self.all_dem:
            return
        T = self.env.t0_flight + t
        Hn = self._outer_dt(t) if t < self.duration else 0.0
        ro = np.flatnonzero(self.regime == ROLL)
        if len(ro) and Hn > 0 and not (self.loads.fast_window(T, T + Hn) or self.loads.breakpoint_in(T - 1e-9, T + Hn)):
            # the load test of _try_sleep for all rolling grains at once; only those that pass get the full test
            faces = np.array([self._roll_face[int(k)] for k in ro], np.int64)
            worst = np.zeros(len(ro))
            for Ts in (T, T + Hn):
                F, Tq = self._body_force(ro, Ts)
                worst = np.maximum(worst, self._margins(ro, faces, F, Tq)[0])
            for k in ro[worst <= 1 - 5 * self.margin]:
                f = self._roll_face[int(k)]
                if self._try_sleep(int(k), f, T, Hn):
                    self._roll_face.pop(int(k), None)
        if self._slv:
            touch_s = self._touching(self.regime == SLAV)
            for cid in list(self._slv):
                mem = self._slv[cid]["members"]
                self.regime[mem] = ACTIVE               # _try_sleep_island works on awake clusters
                if self._try_sleep_island(np.array(sorted(mem)), touch_s, T, Hn):
                    for g in mem:
                        self._slv_of.pop(int(g), None)
                    self._slv.pop(cid)
                else:
                    self.regime[mem] = SLAV
        ai = np.flatnonzero(self.regime == ACTIVE)
        if not len(ai):
            return
        touch = self._touching(self.regime == ACTIVE)
        nwall = np.zeros(self.N, int)
        npair = np.zeros(self.N, int)
        face_of = {}
        for i, j in touch:
            if j < 0:
                nwall[i] += 1
                face_of[i] = -1 - j
            else:
                npair[i] += 1
                npair[j] += 1
        H = self._outer_dt(t) if t < self.duration else 0.0
        s = self._motion_bound(max(H, 1e-6), T)
        gw = np.minimum(self.x - self.rad[:, None], self.L[None, :] - self.x - self.rad[:, None]).min(1)
        tree = cKDTree(self.x)
        # clusters of active grains joined by contacts: try to put each anchored one to sleep
        parent = {int(k): int(k) for k in ai}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i, j in touch:
            if j >= 0 and i in parent and j in parent:
                parent[find(i)] = find(j)
        groups = {}
        for k in ai:
            groups.setdefault(find(int(k)), []).append(int(k))
        for mem in groups.values():
            if len(mem) > 1 or nwall[mem[0]] > 1:
                if any(nwall[k] > 0 for k in mem) and len(mem) <= self.max_island:
                    mm = np.array(sorted(mem))
                    if not self._try_sleep_island(mm, touch, T, H) and self.use_slaved:
                        self._try_slaved(mm, touch, T, H)
        for k in ai:
            if self.regime[k] != ACTIVE:
                continue
            if nwall[k] == 0 and npair[k] == 0:
                # free once clear of everything by the capture distance (the next step's capture time takes over)
                d_cap = self.d_cap
                if gw[k] <= d_cap:
                    continue
                nb = tree.query_ball_point(self.x[k], 2 * self.rad.max() + d_cap)
                if all(j == k or np.linalg.norm(self.x[j] - self.x[k]) - self.rad[j] - self.rad[k] > d_cap
                       for j in nb):
                    self.regime[k] = FREE
            elif nwall[k] == 1 and npair[k] == 0:
                if not self._try_sleep(k, face_of[k], T, H) and self.use_roll:
                    self._try_roll(k, face_of[k], T, H)

    def _try_sleep(self, k, face, T, H):
        """Sleep a grain on one wall if its residual motion cannot carry any contact load past its limit: with
        E = kinetic energy + the energy of its deviation from static equilibrium, the largest extra spring force
        sqrt(2 k_max E) must be below 1% of the smallest remaining margin."""
        if H <= 0 or self.loads.fast_window(T, T + H) or self.loads.breakpoint_in(T - 1e-9, T + H):
            return False
        row = self._key[(int(k), -1 - face)]
        p, h = self._prm[row], self._hist[row]
        idx = np.array([k])
        worst = 0.0
        for Ts in (T, T + H):
            F, Tq = self._body_force(idx, Ts)
            ratio, _, Fn, _, _, _ = self._margins(idx, np.array([face]), F, Tq)
            worst = max(worst, float(ratio[0]))
        if worst > 1 - 5 * self.margin:
            return False
        Fc = p[dem.P_FC]
        Fncrit = abs(float(Fn[0]) + 2 * Fc)
        margin_force = (1 - worst) * min(p[dem.P_MUR], p[dem.P_MUT]) * Fncrit
        # deviation from equilibrium: kinetic energy and the energy of the unbalanced static load
        a = max(h[dem.H_A], 1e-12)
        kn = 1.2 * p[dem.P_ESTAR] * a
        kmax = max(kn, p[dem.P_KT] * a, p[dem.P_KR])
        Ek = 0.5 * self.m[k] * float(self.v[k] @ self.v[k]) + 0.5 * self.I[k] * float(self.w[k] @ self.w[k])
        F, Tq = self._body_force(idx, T)
        # rest position at the current load vs actual: normal and tangential residuals via the springs
        saved = (self.x[k].copy(), h.copy())
        self._place_at_equilibrium(k, face, row, T, F, Tq)
        x_eq, h_eq = self.x[k].copy(), h.copy()
        self.x[k], h[:] = saved
        dxn = float(np.abs(x_eq - self.x[k]).max())
        dxt = float(np.abs(h_eq[dem.H_XT:dem.H_XT + 3] - h[dem.H_XT:dem.H_XT + 3]).max())
        dxr = float(np.abs(h_eq[dem.H_XR:dem.H_XR + 3] - h[dem.H_XR:dem.H_XR + 3]).max())
        Ep = 0.5 * kn * dxn ** 2 + 0.5 * p[dem.P_KT] * a * dxt ** 2 + 0.5 * p[dem.P_KR] * dxr ** 2
        if math.sqrt(2 * kmax * (Ek + Ep)) > 0.01 * margin_force:
            return False
        # asleep: exact static equilibrium for the current load
        self.x[k], h[:] = x_eq, h_eq
        self.v[k] = 0.0
        self.w[k] = 0.0
        self.regime[k] = ASLEEP
        self._sleep_face[k] = face
        self.n_events["sleep"] += 1
        return True

    def _try_roll(self, k, face, T, H):
        """Hand a grain on one wall to the slaved-rolling model if the energy in its stiff modes (normal motion,
        slip at the contact, and the deviation of the normal and sliding springs from their quasi-static values) could
        add at most 1% of the remaining pull-off or sliding margin to the contact force."""
        row = self._key[(int(k), -1 - face)]
        p, h = self._prm[row], self._hist[row]
        n = self._wall_normal(face)
        r = self.rad[k]
        vn = float(self.v[k] @ n)
        slip = self.v[k] - vn * n - r * np.cross(self.w[k], n)
        a = max(h[dem.H_A], 1e-12)
        kn = 1.2 * p[dem.P_ESTAR] * a
        ks = p[dem.P_KT] * p[dem.P_A0]
        F, Tq = self._body_force(np.array([k]), T)
        Fne = -float(F[0] @ n)
        Fc = h[dem.H_FCP] if h[dem.H_YLD] > 0.5 else p[dem.P_FC]
        Fncrit = abs(Fne + 2 * Fc)
        m_pull = Fne + Fc
        m_slide = p[dem.P_MUT] * Fncrit - float(np.linalg.norm(F[0] - (F[0] @ n) * n))
        if m_pull <= 0 or m_slide <= 0:
            return False
        # hysteresis: roll1 hands back at 1 - margin of the pull-off or sliding limit, so enter only at 1 - 5 margin
        if -Fne / Fc > 1 - 5 * self.margin or \
                float(np.linalg.norm(F[0] - (F[0] @ n) * n)) > (1 - 5 * self.margin) * p[dem.P_MUT] * Fncrit:
            return False
        d_eq, a_eq = slaved.jkr_overlap(p, h, Fne, h[dem.H_A] if h[dem.H_A] > 0 else p[dem.P_A0])
        ax = face // 2
        d_now = (self.rad[k] - self.x[k, ax]) if face % 2 == 0 else (self.x[k, ax] - (self.L[ax] - self.rad[k]))
        Ft_s = F[0] - float(F[0] @ n) * n                      # static tangential load: sliding spring at Ft / ks
        Es = 0.5 * self.m[k] * (vn * vn + float(slip @ slip)) + 0.5 * kn * (d_now - d_eq) ** 2             + 0.5 * ks * float(((h[dem.H_XT:dem.H_XT + 3] - Ft_s / ks) ** 2).sum())
        if math.sqrt(2 * max(kn, ks) * Es) > 0.01 * min(m_pull, m_slide):
            return False
        # on the slow manifold: normal overlap at equilibrium, no slip
        self.x[k, ax] = (self.rad[k] - d_eq) if face % 2 == 0 else (self.L[ax] - self.rad[k] + d_eq)
        self.v[k] = r * np.cross(self.w[k], n)
        h[dem.H_A] = a_eq
        self._roll_face[k] = face
        self.regime[k] = ROLL
        self.n_events["roll_entries"] = self.n_events.get("roll_entries", 0) + 1
        return True

    def _jkr_radius(self, p, Fn):
        R, w, E = p[dem.P_REFF], p[dem.P_W], p[dem.P_ESTAR]
        a0 = p[dem.P_A0]
        lo, hi = a0 / 4 ** (1 / 3), a0
        while dem.jkr_force(hi, R, w, E) < Fn:
            hi *= 1.5
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if dem.jkr_force(mid, R, w, E) < Fn:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _roll_dt(self, ri):
        T = math.inf
        for k in ri:
            row = self._key[(int(k), -1 - self._roll_face[int(k)])]
            p = self._prm[row]
            r = self.rad[k]
            Ic = self.I[k] + self.m[k] * r * r
            T = min(T, 2 * math.pi * math.sqrt(Ic / (p[dem.P_KR] * r * r)), 2 * math.pi * math.sqrt(self.I[k] / p[dem.P_KTW]))
        return T / self.n_ppp_roll

    def _roll_step(self, ri, t, H, bv, ba, boff):
        """Advance rolling grains over [t, t+H]; returns those that must be redone in the full model. bv, ba, boff:
        the envelope other grains' capture times assumed for each (a grain whose path leaves it is reported to the
        step's verification)."""
        T0 = self.env.t0_flight + t
        dt = self._roll_dt(ri)
        n = max(1, int(math.ceil(H / dt)))
        dt = H / n
        Ts = T0 + np.arange(n + 1) * dt
        acc, Om, Omd = self.loads.accel_seg(Ts), self.loads.omega(Ts), self.loads.omega_dot(Ts)
        faces = np.array([self._roll_face[int(k)] for k in ri], np.int64)
        rows = np.array([self._key[(int(k), -1 - int(f))] for k, f in zip(ri, faces)], np.int64)
        x0, v0, w0, h0 = self.x[ri].copy(), self.v[ri].copy(), self.w[ri].copy(), self._hist[rows].copy()
        x, v, w = x0.copy(), v0.copy(), w0.copy()
        hist = h0.copy()
        flag = np.zeros(len(ri), np.int64)
        exc = np.full(len(ri), -np.inf)
        roll1.run(x, v, w, self.m[ri], self.I[ri], self.rad[ri], faces, np.ascontiguousarray(self._prm[rows]), hist,
                  self.L, dt, n, acc, Om, Omd, self.off, self.gdrag[ri], np.ascontiguousarray(self._F_lr[ri]),
                  self.margin, flag, np.ascontiguousarray(bv, dtype=float), np.ascontiguousarray(ba, dtype=float),
                  np.ascontiguousarray(boff, dtype=float), exc)
        ok = flag == 0
        self.x[ri[ok]], self.v[ri[ok]], self.w[ri[ok]] = x[ok], v[ok], w[ok]
        self._hist[rows[ok]] = hist[ok]
        self.n_events["roll_grain_steps"] = self.n_events.get("roll_grain_steps", 0) + n * len(ri)
        if hasattr(self, "_chk"):
            self._chk["roll_bad"].update(int(k) for k in ri[exc > 0])
        bad = ri[~ok]
        for k in bad:
            self._roll_face.pop(int(k), None)
        return bad

    # ------------------------------------------------------------------ sleeping clusters
    def _island_system(self, mem, touch):
        loc = {int(g): k for k, g in enumerate(mem)}
        ci, cj, rows = [], [], []
        for (i, j) in touch:
            if i in loc and (j < 0 or j in loc):
                ci.append(loc[i])
                cj.append(j if j < 0 else loc[j])
                rows.append(self._key[(i, j)])
        return np.array(ci, np.int64), np.array(cj, np.int64), np.array(rows, np.int64)

    def _island_load(self, mem, T):
        F, Tq = self._body_force(mem, T)
        return np.hstack([F, Tq]).reshape(-1)

    def _island_ratio(self, isl, T):
        dF = self._island_load(isl["members"], T) - isl["F_ref"]
        du = self._island_lu(isl).solve(dF)
        lam = isl["lam"] + isl["J"] @ du
        if "dl" in isl:                       # sleeping with a tracked residual motion: its largest effect included
            return statics.margins_bounded(lam, isl["dl"], isl["prm"], isl["out"])
        return statics.margins(lam, isl["prm"], isl["out"])

    def _linres(self, mem, x, ci, cj, prm, hist, K, J, du, lam_eq, lam_end, out, T):
        """The residual motion of a resting cluster about its equilibrium, as exact linear response (statics.
        residual_modes): stiffness and dashpots from the kernel's own tangent, the rotating-frame and drag terms, the
        masses. The cluster may sleep with it if every mode decays, the residual moves no grain by more than a tenth
        of the capture distance (so no approach can be missed), and every contact keeps 5x the sleep margin even with the largest the residual can add (for all later
        times); the residual is then added back exactly when the cluster wakes. None if not."""
        def no(why):
            self.n_events["cluster_awake_" + why] = self.n_events.get("cluster_awake_" + why, 0) + 1
        if len(mem) > LINRES_MAX:
            no("too_large")
            return None
        n = 6 * len(mem)
        C, Jv = statics.jacobians(x, self.rad[mem], ci, cj, prm, hist, self.L, vel=True)
        Om = self.loads.omega(np.array([T]))[0]
        W = np.array([[0.0, -Om[2], Om[1]], [Om[2], 0.0, -Om[0]], [-Om[1], Om[0], 0.0]])      # W v = Om x v
        Md = np.empty(n)
        G = np.zeros((n, n))
        for q, g in enumerate(mem):
            Md[6 * q:6 * q + 3] = self.m[g]
            Md[6 * q + 3:6 * q + 6] = self.I[g]
            G[6 * q:6 * q + 3, 6 * q:6 * q + 3] = 2 * self.m[g] * W + self.m[g] * self.gdrag[g] * np.eye(3)
            G[6 * q + 3:6 * q + 6, 6 * q + 3:6 * q + 6] = self.I[g] * W
        v0 = np.hstack([self.v[mem], self.w[mem]]).reshape(-1)
        lam, V, c, err = statics.residual_modes(K, C, Md, G, -du, v0)
        if err > 1e-6 or not (lam.real < 0).all():
            no("modes")
            return None
        dl, ub = statics.residual_bound(V, c, J, Jv)
        if float(np.linalg.norm(ub.reshape(-1, 6)[:, :3], axis=1).max()) > 0.1 * self.d_cap:
            no("residual_moves")
            return None
        worst = np.maximum(statics.margins_bounded(lam_eq, dl, prm, out),
                           statics.margins_bounded(lam_end, dl, prm, out))
        if not np.isfinite(worst).all() or (worst > 1 - 5 * self.margin).any():
            no("residual_loads")
            return None
        self.n_events["linres_sleep"] = self.n_events.get("linres_sleep", 0) + 1
        return dict(lam=lam, V=V, c=c, T=T, dl=dl)

    # ------------------------------------------------------------------ slaved clusters (slaved.py)
    def _rej(self, why):
        k = "slaved_rejected_" + why
        self.n_events[k] = self.n_events.get(k, 0) + 1
        return False

    def _try_slaved(self, mem, touch, T, H):
        """Hand a wall-anchored cluster to the slaved model (slaved.py) if (i) it is small and its contact rows are
        independent and well conditioned (no redundant contacts), (ii) no load jump is in the next step, (iii) at its
        quasi-static contact forces every contact keeps 5x the margin to pull-off and sliding and none would yield,
        and (iv) the energy in its stiff modes (normal and sliding deviation from those forces, and the velocity
        that violates no-slip) could add at most 1% of the smallest margin to any contact force (as _try_roll)."""
        if len(mem) > slaved.MAX_GRAINS or H <= 0 or self.loads.breakpoint_in(T - 1e-9, T + H):
            return self._rej("size_or_jump")
        ci, cj, rows = self._island_system(mem, touch)
        K, N = len(rows), len(mem)
        if not K or K > slaved.MAX_CONTACTS or 3 * K > 6 * N or not (cj < 0).any():
            return self._rej("contacts")
        x = np.ascontiguousarray(self.x[mem])
        v, w = np.ascontiguousarray(self.v[mem]), np.ascontiguousarray(self.w[mem])
        prm = np.ascontiguousarray(self._prm[rows])
        hist = np.ascontiguousarray(self._hist[rows])
        m, I, rad = self.m[mem], self.I[mem], self.rad[mem]
        if slaved.condition(x, m, I, rad, ci, cj, self.L) < 2 * slaved.COND_MIN:
            return self._rej("conditioning")
        f_load = self.loads.max_freq(T, T + H) if self.loads.fast_window(T, T + H) else 0.0
        eta0, etaF, w_soft = slaved.slaving_indicator(x, m, I, rad, ci, cj, prm, hist, self.L, f_load)
        if not (eta0 <= slaved.ETA0_MAX and etaF <= slaved.ETAF_MAX):
            return self._rej("slaving_error")
        Ta = np.array([T])
        lam, n, dl, Gu = slaved.enter(x, v, w, m, I, rad, ci, cj, prm, hist, self.L, self.off,
                                      self.loads.accel(Ta, side="+")[0],
                                      self.loads.omega(Ta)[0], self.loads.omega_dot(Ta)[0], self.gdrag[mem],
                                      np.ascontiguousarray(self._F_lr[mem]))
        if lam is None:
            return self._rej("solver")
        L3 = lam.reshape(K, 3)
        Fne = (L3 * n).sum(1)
        ft = L3 - Fne[:, None] * n
        Fp = np.where(hist[:, dem.H_YLD] > 0.5, hist[:, dem.H_FCP], prm[:, dem.P_FC])
        Fncrit = np.abs(Fne + 2 * Fp)
        r_pull = -Fne / Fp
        r_slide = np.linalg.norm(ft, axis=1) / (prm[:, dem.P_MUT] * Fncrit)
        rr = np.maximum(r_pull, r_slide)
        if not np.isfinite(rr).all() or float(rr.max()) > 1 - 5 * self.margin:
            return self._rej("near_pulloff_or_sliding")
        lamD, _, outD = statics.state(x, ci, cj, prm, hist, self.L)
        L8 = lamD.reshape(K, 8)
        dFn = L8[:, 0] - Fne
        dft = L8[:, 1:4] - ft
        a_c = np.maximum(hist[:, dem.H_A], 1e-12)
        kn = 1.2 * prm[:, dem.P_ESTAR] * a_c
        ks = prm[:, dem.P_KT] * prm[:, dem.P_A0]
        E = float((0.5 * dFn ** 2 / kn + 0.5 * (dft ** 2).sum(1) / ks).sum()
                  + 0.5 * float(m.max()) * float((Gu ** 2).sum()))
        mf = float(((1 - rr) * np.minimum(prm[:, dem.P_MUT], prm[:, dem.P_MUR]) * Fncrit).min())
        if math.sqrt(2 * float(max(kn.max(), ks.max())) * E) > 0.01 * mf:
            return self._rej("stiff_energy")
        if (np.array([slaved.jkr_overlap(prm[k], hist[k], Fne[k], a_c[k])[0] for k in range(K)])
                > prm[:, dem.P_DY]).any():
            return self._rej("plastic")
        # onto the slow manifold: overlaps and sliding springs for these forces, no-slip velocities
        slaved.settle(x, m, rad, ci, cj, prm, hist, self.L, lam)
        slaved.project_velocity(x, v, w, m, I, rad, ci, cj, self.L)
        self.x[mem], self.v[mem], self.w[mem] = x, v, w
        self._hist[rows] = hist
        cid = self._next_slv
        self._next_slv += 1
        self._slv[cid] = dict(members=np.array(mem, np.int64), rows=rows, ci=ci, cj=cj, lam=lam, w_soft=w_soft,
                              eta=eta0)
        for g in mem:
            self._slv_of[int(g)] = cid
        self.regime[mem] = SLAV
        self.n_events["slaved_entries"] = self.n_events.get("slaved_entries", 0) + 1
        return True

    def _dissolve_slv(self, cid):
        """A slaved cluster goes to the full model (its state is on the slow manifold)."""
        c = self._slv.pop(cid)
        for g in c["members"]:
            self._slv_of.pop(int(g), None)
        self.regime[c["members"]] = ACTIVE

    def _slv_obstacles(self, mem, reach):
        """Static grains (asleep) near the cluster: centres and radii, for the pre-run's approach checks."""
        sl = np.flatnonzero(self.regime == ASLEEP)
        if not len(sl):
            return np.zeros((0, 3)), np.zeros(0)
        d = np.linalg.norm(self.x[sl][:, None, :] - self.x[mem][None, :, :], axis=2).min(1)
        near = sl[d <= 2 * float(self.rad.max()) + self.d_cap + reach]
        return np.ascontiguousarray(self.x[near]), np.ascontiguousarray(self.rad[near])

    def _slv_prerun(self, cid, t, H, T):
        """Run a slaved cluster over [t, t + H] (or until it must stop) and return its envelope: each member's
        largest distance from its start plus the most it moves within one slaved step; t_stop (H if it ran through).
        The start state is kept so that the cluster can be run again to an earlier time."""
        c = self._slv[cid]
        mem, rows = c["members"], c["rows"]
        pre = dict(cid=cid, x=self.x[mem].copy(), v=self.v[mem].copy(), w=self.w[mem].copy(),
                   hist=self._hist[rows].copy(), lam=c["lam"].copy())
        f_load = self.loads.max_freq(T, T + H) if self.loads.fast_window(T, T + H) else 0.0
        eta0, etaF, w_soft = slaved.slaving_indicator(self.x[mem], self.m[mem], self.I[mem], self.rad[mem], c["ci"],
                                                      c["cj"], np.ascontiguousarray(self._prm[rows]),
                                                      np.ascontiguousarray(self._hist[rows]), self.L, f_load)
        if not (eta0 <= 1.5 * slaved.ETA0_MAX and etaF <= 1.5 * slaved.ETAF_MAX):
            pre.update(dict(t_stop=0.0, dmax=np.zeros(len(mem)), vmx=np.linalg.norm(pre["v"], axis=1), dt=0.0,
                            s=np.linalg.norm(pre["v"], axis=1) * 0.0 + 1e-12, n_done=0, stop=None))
            self.n_events["slaved_stops_indicator"] = self.n_events.get("slaved_stops_indicator", 0) + 1
            return pre
        c["w_soft"] = max(w_soft, 1e-30)
        a_max = float(np.linalg.norm(self.loads.accel_bound(T, T + H)))
        reach = float(np.linalg.norm(self.v[mem], axis=1).max()) * H + 0.5 * a_max * H * H
        for _ in range(3):
            xo, ro = self._slv_obstacles(mem, reach)
            self._slv_restore(pre)
            res = self._slv_advance(cid, t, H, T, pre, xo, ro)
            if float(res["dmax"].max()) <= reach:
                break
            reach = 2 * float(res["dmax"].max())        # it went further than the obstacles listed: list more
        else:
            res["t_stop"] = 0.0
        pre.update(res)
        pre["s"] = res["dmax"] + res["vmx"] * res["dt"] + 1e-12
        return pre

    def _slv_restore(self, pre):
        c = self._slv[pre["cid"]]
        mem, rows = c["members"], c["rows"]
        self.x[mem], self.v[mem], self.w[mem] = pre["x"], pre["v"], pre["w"]
        self._hist[rows] = pre["hist"]
        c["lam"] = pre["lam"].copy()

    def _slv_advance(self, cid, t, Hs, T, pre, xo=None, ro=None, relaxed=False, dt_fix=None, nsteps_fix=None):
        """Integrate a slaved cluster over [t, t + Hs] with slaved.run; stops early (state at the start of the slaved
        step where it had to stop). Returns dmax, vmx, t_stop and the slaved step."""
        c = self._slv[cid]
        mem, rows, ci, cj = c["members"], c["rows"], c["ci"], c["cj"]
        if xo is None:
            xo, ro = pre.get("xo", np.zeros((0, 3))), pre.get("ro", np.zeros(0))
        pre["xo"], pre["ro"] = xo, ro
        m, I, rad = self.m[mem], self.I[mem], self.rad[mem]
        prm = np.ascontiguousarray(self._prm[rows])
        hist = np.ascontiguousarray(self._hist[rows])
        x = np.ascontiguousarray(self.x[mem])
        v, w = np.ascontiguousarray(self.v[mem]), np.ascontiguousarray(self.w[mem])
        if dt_fix is not None:                   # a re-run on the pre-run's own grid (bit-identical path)
            dt, n = dt_fix, int(nsteps_fix)
        else:
            dt = min(slaved.soft_period(m, I, rad, ci, cj, prm), 2 * math.pi / c["w_soft"]) / self.n_ppp_slv
            n = max(1, int(math.ceil(Hs / dt)))
            dt = Hs / n
        if n == 0:
            return dict(dmax=np.zeros(len(mem)), vmx=np.linalg.norm(self.v[mem], axis=1), t_stop=0.0, dt=dt,
                        n_done=0, stop=None)
        T0 = self.env.t0_flight + t
        if dt_fix is not None and "acc" in pre:      # the pre-run's own load samples (bit-identical path)
            acc, Om, Omd = pre["acc"][:n + 1], pre["Om"][:n + 1], pre["Omd"][:n + 1]
        else:
            Ts = T0 + np.arange(n + 1) * dt
            acc, Om, Omd = self.loads.accel_seg(Ts), self.loads.omega(Ts), self.loads.omega_dot(Ts)
            pre["acc"], pre["Om"], pre["Omd"] = acc, Om, Omd
        cface = np.zeros((len(mem), 6), np.bool_)
        for k in range(len(ci)):
            if cj[k] < 0:
                cface[ci[k], -1 - cj[k]] = True
        flag = np.zeros(2, np.int64)
        Nm = len(mem)
        exc = np.full(Nm, -np.inf)
        dmax = np.zeros(Nm)
        vmx = np.zeros(Nm)
        lam = c["lam"]
        slaved.run(x, v, w, m, I, rad, ci, cj, prm, hist, self.L, self.off, dt, n, acc, Om, Omd, self.gdrag[mem],
                   np.ascontiguousarray(self._F_lr[mem]), (0.5 if relaxed else 1.0) * self.margin,
                   (0.5 if relaxed else 1.0) * self.d_cap, lam, flag, np.full(Nm, np.inf),
                   np.zeros(Nm), np.zeros(Nm), exc, np.ascontiguousarray(xo), np.ascontiguousarray(ro), cface, dmax,
                   vmx, 2e-4)
        self.x[mem], self.v[mem], self.w[mem] = x, v, w
        self._hist[rows] = hist
        t_stop = Hs if flag[0] == 0 else (int(flag[0]) - 1) * dt
        stop = None
        if flag[0]:
            stop = "slaved_stops_" + {1: "pulloff", 2: "sliding", 3: "conditioning", 4: "approach", 5: "solver",
                                      6: "plastic", 7: "normal_turn"}.get(int(flag[1]), "other")
        return dict(dmax=dmax, vmx=np.maximum(vmx, np.linalg.norm(pre["v"], axis=1)), t_stop=t_stop, dt=dt,
                    n_done=(int(flag[0]) - 1) if flag[0] else n, stop=stop)

    def _slv_count(self, grain_steps, stop):
        self.n_events["slaved_grain_steps"] = self.n_events.get("slaved_grain_steps", 0) + int(grain_steps)
        if stop:
            self.n_events[stop] = self.n_events.get(stop, 0) + 1

    def _try_sleep_island(self, mem, touch, T, H):
        """Sleep an anchored cluster if (i) no fast load or breakpoint is in the next step, (ii) at its exact static
        equilibrium every contact keeps 5x the sleep margin to its limits for the loads over the next step, and
        (iii) its residual motion (kinetic energy + elastic energy of its deviation from equilibrium) could add at
        most 1% of the smallest margin to any contact force."""
        if H <= 0 or self.loads.fast_window(T, T + H) or self.loads.breakpoint_in(T - 1e-9, T + H):
            return False
        from scipy.sparse.linalg import splu
        res = None
        tag = "cluster_awake_"                    # counts of why clusters stay in the full model (diagnostics)
        ci, cj, rows = self._island_system(mem, touch)
        if not len(rows):
            return False
        x = np.ascontiguousarray(self.x[mem])
        prm = np.ascontiguousarray(self._prm[rows])
        hist = np.ascontiguousarray(self._hist[rows])
        K, J = statics.jacobians(x, self.rad[mem], ci, cj, prm, hist, self.L)
        try:
            lu = splu(K)
        except RuntimeError:
            return False
        lam, Fnet, out = statics.state(x, ci, cj, prm, hist, self.L)
        Fext = self._island_load(mem, T)
        du = lu.solve(Fext + Fnet)
        lam_eq = lam + J @ du
        ratio = statics.margins(lam_eq, prm, out)
        ratio_end = statics.margins(lam_eq + J @ lu.solve(self._island_load(mem, T + H) - Fext), prm, out)
        rr = np.maximum(ratio, ratio_end)
        if not np.isfinite(rr).all() or float(rr.max()) > 1 - 5 * self.margin:
            k = tag + ("over_limit" if np.isfinite(rr).all() and float(rr.max()) > 1 else "near_limit")
            self.n_events[k] = self.n_events.get(k, 0) + 1
            return False
        Ek = 0.5 * float((self.m[mem] * (self.v[mem] ** 2).sum(1)).sum()
                         + (self.I[mem] * (self.w[mem] ** 2).sum(1)).sum())
        Ep = 0.5 * float(du @ (K @ du))
        kmax = float(np.abs(K.diagonal()[0::6]).max())
        L8 = lam_eq.reshape(-1, 8)
        Fncrit = np.abs(L8[:, 0] + 2 * out[:, 19])
        mf = float(((1 - rr) * np.minimum(prm[:, dem.P_MUT], prm[:, dem.P_MUR]) * Fncrit).min())
        if math.sqrt(2 * kmax * max(Ek + Ep, 0.0)) > 0.01 * mf:
            # the quick bound (stiffest spring times all the energy) fails: use the exact one, the most each contact
            # quantity can deviate inside the residual motion's energy ellipsoid; every contact's worst-case load
            # ratio must stay within 1% of its margin of the equilibrium value
            dl = statics.energy_bounds(K, J, max(Ek + Ep, 0.0))
            lam_end = lam_eq + J @ lu.solve(self._island_load(mem, T + H) - Fext)
            ok = dl is not None
            if ok:
                worst = np.maximum(statics.margins_bounded(lam_eq, dl, prm, out),
                                   statics.margins_bounded(lam_end, dl, prm, out))
                ok = np.isfinite(worst).all() and not ((worst - rr) > 0.01 * (1 - rr)).any()
            if not ok:
                # still moving too much to drop the motion: sleep with it tracked exactly (linear response)
                res = self._linres(mem, x, ci, cj, prm, hist, K, J, du, lam_eq, lam_end, out, T)
                if res is None:
                    return False                  # (_linres counts why)
        dU = du.reshape(-1, 6)
        xn, hn = statics.apply_displacement(x, ci, cj, prm, hist, self.L, dU)
        self.x[mem] = xn
        self._hist[rows] = hn
        self.v[mem] = 0.0
        self.w[mem] = 0.0
        # reference state for the certificate: the equilibrium just reached
        K, J = statics.jacobians(xn, self.rad[mem], ci, cj, prm, hn, self.L)
        lam, Fnet, out = statics.state(xn, ci, cj, prm, hn, self.L)
        iid = self._next_island
        self._next_island += 1
        self._islands[iid] = dict(members=mem, ci=ci, cj=cj, rows=rows, prm=prm, out=out, lam=lam, J=J, K=K,
                                  lu=splu(K), F_ref=self._island_load(mem, T), T_ref=T)
        if res is not None:
            self._islands[iid]["res"] = res
            self._islands[iid]["dl"] = res["dl"]
        for k in mem:
            self._island_of[int(k)] = iid
        self.regime[mem] = ASLEEP
        self.n_events["sleep"] += len(mem)
        self.n_events["cluster_sleep"] = self.n_events.get("cluster_sleep", 0) + 1
        return True

    def _wake_island(self, iid, T):
        """Wake a sleeping cluster: move it to the exact static equilibrium of the current load (linear response
        from the reference state; positions and spring histories), at rest."""
        isl = self._islands.pop(iid)
        mem = isl["members"]
        du = self._island_lu(isl).solve(self._island_load(mem, T) - isl["F_ref"]).reshape(-1, 6)
        vr = np.zeros((len(mem), 6))
        if "res" in isl:                      # the residual motion, evolved exactly since the cluster fell asleep
            r = isl["res"]
            ur, vrf = statics.residual_state(r["lam"], r["V"], r["c"], T - r["T"])
            du = du + ur.reshape(-1, 6)
            vr = vrf.reshape(-1, 6)
        x = np.ascontiguousarray(self.x[mem])
        hist = np.ascontiguousarray(self._hist[isl["rows"]])
        xn, hn = statics.apply_displacement(x, isl["ci"], isl["cj"], isl["prm"], hist, self.L, du)
        self.x[mem] = xn
        self._hist[isl["rows"]] = hn
        self.v[mem] = vr[:, :3]
        self.w[mem] = vr[:, 3:]
        for k in mem:
            self._island_of.pop(int(k), None)
        self.regime[mem] = ACTIVE
        self.n_events["wake"] += len(mem)

    # ------------------------------------------------------------------ impact and detachment records
    def _log_events(self, t):
        nrow = len(self._ckeys)
        h = self._hist[:nrow]
        T0 = self.env.t0_flight
        newf = np.flatnonzero(h[:, dem.H_NFORM] > self._seen_form[:nrow])
        for r in newf:
            i, j = self._ckeys[r]
            hr = h[r]
            rec = dict(t=float(hr[dem.H_TFORM]) - T0, kind="wall" if j < 0 else "pp", i=i, j=j,
                       mat_i=self.mats[self.mat[i]].name,
                       mat_j=self.wall_mat.name if j < 0 else self.mats[self.mat[j]].name,
                       n_i=1, n_j=0 if j < 0 else 1, frozen_i=False, frozen_j=j < 0, post_fragment=False,
                       v_n=float(hr[dem.H_VIN]), v_t=float(hr[dem.H_VTIN]),
                       v_rel=float(math.hypot(hr[dem.H_VIN], hr[dem.H_VTIN])), outcome="pending", e_n=0.0,
                       r_sum=float(self.rad[i] + (self.rad[j] if j >= 0 else 0.0)),
                       x=float(self.x[i, 0]), y=float(self.x[i, 1]), z=float(self.x[i, 2]),
                       n_x=0.0, n_y=0.0, n_z=0.0, vrel_x=0.0, vrel_y=0.0, vrel_z=0.0,
                       vcm_x=0.0, vcm_y=0.0, vcm_z=0.0, multiple=int(hr[dem.H_NFORM] - self._seen_form[r]) > 1)
            n_i, wall_i = self._cluster_before(i, j)
            rec["n_i"], rec["frozen_i"] = n_i, wall_i
            if j >= 0:
                d = self.x[i] - self.x[j]
                nv = d / np.linalg.norm(d)
                rec.update(n_x=float(nv[0]), n_y=float(nv[1]), n_z=float(nv[2]))
                rec["n_j"], rec["frozen_j"] = self._cluster_before(j, i)
            self.collisions.append(rec)
            self._pending[r] = rec
            self._seen_form[r] = hr[dem.H_NFORM]
            if self.base.charging:
                self.base.q = self.base.q
                self.base._charge(i, None if j < 0 else j, float(hr[dem.H_AMAX]))
        newb = np.flatnonzero(h[:, dem.H_NBREAK] > self._seen_break[:nrow])
        for r in newb:
            i, j = self._ckeys[r]
            rec = self._pending.pop(r, None)
            if rec is not None and rec["outcome"] == "pending":
                rec["outcome"] = "bounce"
                rec["e_n"] = float(abs(h[r, dem.H_VOUT]) / max(rec["v_n"], 1e-12))
                self.n_events["wall_bounce" if j < 0 else "bounce"] += 1
            elif j < 0:
                # a wall contact that had held: the grain was pulled or knocked off the wall
                self._ever_freed.add(int(i))
                self.n_events["detach"] += 1
                self.detach_log.append(dict(t=float(h[r, dem.H_TBREAK]), grain=int(i), n=1, how="contact",
                                            v=float(np.linalg.norm(self.v[i]))))
            self._seen_break[r] = h[r, dem.H_NBREAK]
        for r, rec in list(self._pending.items()):
            if t - rec["t"] >= STICK_HOLD_S and h[r, dem.H_TOUCH] > 0.5:
                rec["outcome"] = "stick"
                self.n_events["wall_stick" if rec["kind"] == "wall" else "stick"] += 1
                del self._pending[r]

    def _cluster_before(self, i, j):
        """Size of grain i's cluster and whether it touches a wall, not counting the new contact with j."""
        adj = {}
        wall = set()
        for a, c in self._touching():
            if c < 0:
                if (a, c) != (i, j):
                    wall.add(a)
            elif {a, c} != {i, j}:
                adj.setdefault(a, []).append(c)
                adj.setdefault(c, []).append(a)
        seen, todo = {i}, [i]
        while todo:
            k = todo.pop()
            for q in adj.get(k, ()):
                if q not in seen:
                    seen.add(q)
                    todo.append(q)
        return len(seen), bool(seen & wall)

    # ------------------------------------------------------------------ outputs compatible with simulation.py
    @property
    def body(self):
        return np.arange(self.N)

    @property
    def out_dt(self):
        return self.base.out_dt

    @property
    def nmono(self):
        """Grains per cluster, indexed by cluster id (a cluster's id is one of its grains), as in simulation.py."""
        return np.bincount(self.clusters(), minlength=self.N)

    @property
    def alive(self):
        return self.nmono > 0

    def _wall_contacts(self):
        face = np.full(self.N, -1)
        for i, j in self._touching():
            if j < 0:
                face[i] = -1 - j
        return face

    @property
    def frozen(self):
        return self._wall_contacts() >= 0

    @property
    def fw_ax(self):
        f = self._wall_contacts()
        return np.where(f >= 0, f // 2, -1)

    @property
    def fw_side(self):
        f = self._wall_contacts()
        return np.where(f >= 0, np.where(f % 2 == 0, -1.0, 1.0), 0.0)

    def clusters(self):
        parent = list(range(self.N))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i, j in self._touching():
            if j >= 0:
                parent[find(i)] = find(j)
        return np.array([find(k) for k in range(self.N)])

    def _record(self, t):
        cl = self.clusters()
        wall = self.frozen
        self.frame_t.append(t)
        self.frames.append(self.x.astype(np.float32).copy())
        self.frame_bodies.append(cl.astype(np.int32))
        self.frame_frozen.append(wall.copy())
        self.frame_spin.append((np.linalg.norm(self.w, axis=1) / (2 * math.pi)).astype(np.float32))
        free = ~wall
        sizes = np.bincount(cl)
        sizes = sizes[sizes > 0]
        sp = np.linalg.norm(self.v[free], axis=1)
        self.timeseries.append(dict(
            t=t, free_bodies=int(len(set(cl[free]))), free_monomers=int(free.sum()), wall_monomers=int(wall.sum()),
            mean_aggregate_size=float(sizes.mean()), max_aggregate_size=int(sizes.max()),
            mean_speed=float(sp.mean()) if len(sp) else 0.0,
            mean_spin_hz=float(self.frame_spin[-1][free].mean()) if free.any() else 0.0,
            max_spin_hz=float(self.frame_spin[-1][free].max()) if free.any() else 0.0,
            kinetic_energy=float(0.5 * (self.m[free] * sp ** 2).sum()),
            asleep=int((self.regime == ASLEEP).sum()), active=int((self.regime == ACTIVE).sum()),
            free_flight=int((self.regime == FREE).sum()), **{k: v for k, v in self.n_events.items()},
            mean_abs_charge_e=float(np.abs(self.base.q).mean() / 1.602176634e-19),
            net_charge_e=float(self.base.q.sum() / 1.602176634e-19)))

    def save_autosave(self, path, next_out, wall):
        """Save the whole state to `path` (atomically) so that a run killed later (out of memory, reboot) continues
        from here: the continued run is bit-for-bit the uninterrupted one (nothing here changes the state)."""
        self._next_out = next_out
        self._wall_prev = wall
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def run(self, progress=True, callback=None, callback_interval_s=0.5, autosave_path=None, autosave_s=1800.0):
        # the neighbour lists are built when a run starts; a run continued from a saved state keeps the saved lists
        # (rebuilt only when needed, exactly as in the uninterrupted run)
        if not getattr(self, "_run_started", False):
            self.base._build_neighbors()
            self._run_started = True
        t = self.t
        next_out = 0.0 if self._next_out is None else self._next_out
        out_dt = self.base.out_dt
        cps = sorted(c - self.env.t0_flight for c in self.checkpoints if c - self.env.t0_flight > t + 1e-9)
        t0 = last = last_cb = last_save = time.time()
        wall_prev = getattr(self, "_wall_prev", 0.0)
        while t < self.duration - 1e-12:
            if t >= next_out - 1e-12:
                self._record(t)
                next_out += out_dt
            while cps and t >= cps[0] - 1e-9:
                self._next_out = next_out
                self.save_checkpoint()
                cps.pop(0)
            if autosave_path is not None and time.time() - last_save >= autosave_s:
                self.save_autosave(autosave_path, next_out, wall_prev + time.time() - t0)
                last_save = time.time()
            t_cap = min(next_out, cps[0]) if cps else next_out
            if not (self.regime != ASLEEP).any():
                t_new = self._skip_asleep(t, t_cap)
                if t_new > t + 1e-12:
                    self.n_events["skipped_s"] = self.n_events.get("skipped_s", 0.0) + (t_new - t)
                    t = t_new
                    self.t = t
                    continue
            H = self._outer_dt(t)
            if cps and t + H > cps[0]:
                H = cps[0] - t
            self.step(t, H)
            t += H
            if callback is not None and time.time() - last_cb > callback_interval_s:
                last_cb = time.time()
                callback(t, self.duration, dict(self.n_events))
            if progress and time.time() - last > 5.0:
                last = time.time()
                print(f"  [{self.name}] T+{self.env.t0_flight + t:8.3f} s  active {int((self.regime == ACTIVE).sum())}"
                      f" rolling {int((self.regime == ROLL).sum())} slaved {int((self.regime == SLAV).sum())} "
                      f"free {int((self.regime == FREE).sum())} "
                      f"asleep {int((self.regime == ASLEEP).sum())}  "
                      f"impacts {len(self.collisions)}  ({wall_prev + time.time() - t0:.0f} s)", flush=True)
        for rec in self._pending.values():
            if rec["outcome"] == "pending":
                rec["outcome"] = "stick" if self.t - rec["t"] >= 0 else "pending"
        self._record(t)
        self.wall_time = wall_prev + time.time() - t0
        return self


# ---------------------------------------------------------------------------- checkpoints and branches
def state_signature(obj, pre="", out=None, depth=0, seen=None):
    """A hash of every attribute of a simulation (recursively into its sub-objects): two runs built from the same
    inputs have the same signature, so the attributes in which two scenarios differ can be listed exactly."""
    out = {} if out is None else out
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 4:
        return out
    seen.add(id(obj))
    for k, v in sorted(vars(obj).items()):
        if k == "_init_sig":
            continue
        key = pre + k
        if isinstance(v, np.ndarray):
            out[key] = hashlib.md5(v.tobytes() + str(v.shape).encode() + str(v.dtype).encode()).hexdigest()
        elif isinstance(v, (int, float, str, bool, type(None), tuple)):
            out[key] = repr(v)
        elif isinstance(v, (dict, list, set)):
            out[key] = hashlib.md5(repr(_stable(v)).encode()).hexdigest()
        elif isinstance(v, np.random.Generator):
            out[key] = repr(v.bit_generator.state)
        elif hasattr(v, "__dict__") and not callable(v):
            state_signature(v, key + ".", out, depth + 1, seen)
        else:
            out[key] = type(v).__name__
    return out


def _stable(v, depth=0):
    """A representation of v that is the same for two equal objects built separately (no memory addresses)."""
    if depth > 6:
        return type(v).__name__
    if isinstance(v, np.ndarray):
        return hashlib.md5(v.tobytes() + str(v.shape).encode() + str(v.dtype).encode()).hexdigest()
    if isinstance(v, dict):
        return sorted(((repr(k), _stable(x, depth + 1)) for k, x in v.items()), key=lambda kv: kv[0])
    if isinstance(v, set):
        return sorted(repr(_stable(x, depth + 1)) for x in v)
    if isinstance(v, (list, tuple)):
        return [_stable(x, depth + 1) for x in v]
    if isinstance(v, (int, float, str, bool, type(None), np.generic)):
        return repr(v)
    if hasattr(v, "__dict__") and not callable(v):
        return (type(v).__name__, _stable(vars(v), depth + 1))
    return type(v).__name__


def _allowed(key):
    return any(key == a or key.startswith(a + ".") or (a.endswith(".") and key.startswith(a)) for a in BRANCH_ALLOWED)


def branch(path, target: "FullContactSim") -> "FullContactSim":
    """Continue the run saved at `path` under the scenario of `target` (a freshly built simulation of the same tubule).
    Exact, and refused otherwise: (1) a fresh run of the target scenario may differ from a fresh run of the saved one
    only in BRANCH_ALLOWED (release schedule, flight-profile options, the configs that carry them); (2) every load
    (apparent acceleration, spin and its rate, releases, vibration) is identical up to the checkpoint, sampled every
    millisecond and at every breakpoint of either run, and so are the breakpoints and fast windows. The saved run then
    takes the target's release schedule and flight profile and continues."""
    with open(path, "rb") as f:
        C = pickle.load(f)
    diff = sorted(k for k in set(C._init_sig) | set(target._init_sig) if C._init_sig.get(k) != target._init_sig.get(k))
    bad = [k for k in diff if not _allowed(k)]
    if bad or C.name != target.name:
        raise ValueError(f"cannot branch {path.name}: the scenarios differ in {bad or ['tubule']}")
    T0 = float(C.env.t0_flight)
    Tc = T0 + C.t
    bps = [b for b in np.concatenate([C.loads.breakpoints, target.loads.breakpoints]) if T0 <= b <= Tc]
    Ts = np.unique(np.concatenate([np.arange(T0, Tc, 1e-3), bps, [Tc]]))
    for f in ("accel", "omega", "omega_dot", "release_accel", "vibration_accel"):
        a, b = getattr(C.loads, f)(Ts), getattr(target.loads, f)(Ts)
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"cannot branch {path.name}: '{f}' differs before T+{Tc:g} s "
                             f"(largest difference {np.abs(a - b).max():.3g})")
    if not np.array_equal(C.loads.breakpoints[C.loads.breakpoints <= Tc],
                          target.loads.breakpoints[target.loads.breakpoints <= Tc]):
        raise ValueError(f"cannot branch {path.name}: load breakpoints differ before T+{Tc:g} s")
    if C.loads.fast_window(T0, Tc) != target.loads.fast_window(T0, Tc):
        raise ValueError(f"cannot branch {path.name}: fast loads differ before T+{Tc:g} s")
    for t in Ts[::20]:
        if not np.array_equal(C.env.profile.apparent_accel(t), target.env.profile.apparent_accel(t)):
            raise ValueError(f"cannot branch {path.name}: the flight profile differs before T+{Tc:g} s")
    # take over the target's scenario-dependent parts (the saved environment keeps its own state and random stream)
    C.cfg, C.chamber = target.cfg, target.chamber
    C.base.cfg, C.base.chamber, C.base.release = target.base.cfg, target.base.chamber, target.base.release
    C.env.cfg, C.env.profile = target.env.cfg, target.env.profile
    C.loads = target.loads
    C.loads.env, C.loads.prof = C.env, C.env.profile
    C._init_sig = target._init_sig
    C.checkpoints, C.checkpoint_dir = list(target.checkpoints), target.checkpoint_dir
    C.n_events["branched_at_T"] = round(Tc, 3)
    C.n_events["branched_from"] = str(path.name)
    C.n_events["branch_differences"] = diff
    return C


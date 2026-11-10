"""DOLL-E v0.6 flight engine: every grain is its own body and every contact has the full contact mechanics.

Each grain is in one of three regimes, switched by exact or bounded tests:

* FREE: touching nothing and unable to touch anything during the next outer step H (its gap to every other grain and
  wall exceeds the most both can move in H). It flies on the exact free-flight integrator of simulation.py (velocity
  Verlet with drag and Coriolis integrated in closed form), so no contact can be missed or found late.
* ACTIVE: everything that touches something, or could touch something within H, is integrated by the full-stiffness
  soft-sphere DEM of dem.py (JKR + plastic yield + damping, Mindlin sliding, rolling and twisting springs with their
  limits; checked against LAMMPS to <= 2e-6), at a step of the fastest contact period / steps_per_period.
* ASLEEP: a grain resting on a wall, or a wall-anchored cluster of grains (a pile, an aggregate in a corner), whose
  contacts are proven to stay inside their limits for the loads over the next step. A single grain sleeps while the
  normal load stays above -F_c (pull-off) and the sliding, rolling and twisting loads below their limits, each with a
  2% margin (the loads at the step's samples plus their deviation from the chord between them, Loads.chord_dev; the
  same test at entry and at every step): inside these limits the contact mechanics is elastic, no plastic variable
  (contact point, rolled distance, twist) can change and the grain cannot move. A cluster sleeps as an island at its
  exact static equilibrium, which is recomputed every step from the committed state by a Newton solve with the
  sliding, rolling and twisting caps lifted (statics.solve_equilibrium), the caps being allowed to be reached: tier 1
  (static) when every bounded sliding, rolling and twisting ratio stays <= 1 + 1e-6 along the step, the bound
  covering the loads' deviation from their chord inside the step (Loads.chord_dev, spin_chord), the solve's residual,
  the dynamics of the quasi-static drift and the kernel's nonlinearity measured at the mid-chord state; the real
  kernel then takes no clip branch, so the plastic variables are exact and the committed state is the kernel's own
  equilibrium to 1e-6 of the caps. Tier 2 (creep, cfg contact.creep_islands) when a cap is exceeded: the kernel's
  own one-shot return map is iterated to equilibrium with its consistent tangent (statics.released_stiffness), the
  step's error is estimated by a half-step pair and every tolerated term is summed into a budget (island_budget_rel
  of the deflection at the cap) that wakes the island when spent. Loads at step ends are limits from the left, so a
  step ending at a load breakpoint is certified against the pre-jump load and the DEM resolves the jump from the
  committed pre-jump state.
  The tests run every outer step; a sleeper wakes (becomes ACTIVE) the moment its test fails, at every load
  breakpoint (thrust on/off, de-spin, shocks), during release waveforms and vibration (kHz loads), when another body
  could reach it (a grain in the DEM, or another sleeper when a drifting island could close the gap between them),
  and an island also at pull-off, break, yield, contact formation, a mechanism, excessive motion, flow or path error,
  and the budget (reasons counted as island_wake_*; every bound reported: island_bound_xi, island_flow_m,
  island_snap_rel, island_newton_rel, island_dyn_rel, island_wake_transient_m). On waking, a state at the exact
  static solution of the current load (at a breakpoint the load just before it), at rest, is handed to the DEM, so
  waking adds no transient the full model would not have; a cluster still moving may sleep with its residual motion
  tracked as exact linear response (statics.residual_modes).

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
# sleeping clusters at their sliding/rolling/twisting caps (_try_sleep_island, _island_advance): the cfg["contact"]
# keys with their defaults (island_gap_m None: 2 d_cap), the reasons an island wakes, the counters that always exist
ISLAND_DEFAULTS = dict(cap_sleep=True, creep_islands=True, island_tol_rel=1e-6, island_ratchet_rel=1e-3,
                       island_motion_m=1e-9, island_flow_max=0.05, island_path_tol=1e-3, island_budget_rel=1e-2,
                       island_gap_m=None, island_hyst_steps=20, island_newton_max=40, twist_continuity=False)
ISLAND_WAKE_REASONS = ("fast", "loading", "pulloff", "break", "yield", "newcontact", "mechanism", "motion", "flow",
                       "twist", "path", "budget", "newton", "approach", "neighbour")
ISLAND_COUNTERS = (("cluster_sleep_cap", "cluster_sleep_creep", "island_static_steps", "island_creep_steps",
                    "island_newton_iters", "island_refactor", "island_wake_fallback", "island_twist_continuity")
                   + tuple("island_wake_" + w for w in ISLAND_WAKE_REASONS)
                   + tuple("cluster_awake_" + w for w in ("mechanism", "over_limit", "ratchet", "certificate", "drift",
                                                          "newton", "transition", "neighbour")))
ISLAND_FLOATS = ("island_bound_xi", "island_flow_m", "island_snap_rel", "island_wake_transient_m", "island_newton_rel",
                 "island_dyn_rel")

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
        for k, v in ISLAND_DEFAULTS.items():       # sleeping clusters at their caps (_try_sleep_island, _island_advance)
            setattr(self, k, float(cc.get(k, 2 * self.d_cap)) if v is None else type(v)(cc.get(k, v)))
        for key in ISLAND_COUNTERS:
            self.n_events[key] = 0
        for key in ISLAND_FLOATS:
            self.n_events[key] = 0.0
        self._isl_pending = {}      # island id -> its state advanced over the step, committed once the step is verified
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

    def _body_force(self, idx, T, side=None):
        """Quasi-static force [N] and torque on resting grains idx at flight time T (v = 0, omega = 0); side "-"
        takes the loads as their limits from the left (Loads.accel, omega_dot), for step ends and skip samples: a
        step ending at a load breakpoint lies in the preceding piece."""
        a = self.loads.accel(np.array([T]), side=side)[0]
        Om = self.loads.omega(T)[0]
        Omd = self.loads.omega_dot(T, side=side)[0]
        r = self.x[idx] + self.off[None, :]
        cen = -np.cross(Om[None, :], np.cross(Om[None, :], r))
        eul = -np.cross(Omd[None, :], r)
        F = self.m[idx, None] * (a[None, :] + cen + eul) + self._F_lr[idx]
        T_in = -self.I[idx, None] * Omd[None, :]            # spin held relative to the tubule
        return F, T_in

    def _margins(self, idx, faces, F, Tq, dF=None):
        """For grains idx resting on faces (one wall contact each): the ratios of the static normal, sliding,
        rolling and twisting loads to their limits, > 1 - margin meaning the grain must be woken; with dF
        (len(idx),) [N] the worst case when the load may deviate from F, Tq by up to dF in force and dF R in torque
        (as statics.margins_bounded: the pressing load lowered by dF, the tangential force raised by dF, the
        twisting and the rolling torques by dF R; the loads' deviation from their chord inside a step,
        _chord_force). Returns the worst ratio per grain, the contact rows, the pressing load, the tangential load,
        the torque, the wall normals and the ratios by kind (4 x len(idx): pull-off, sliding, rolling, twisting)."""
        rows = np.array([self._key[(int(i), -1 - int(f))] for i, f in zip(idx, faces)])
        p = self._prm[rows]
        h = self._hist[rows]
        n = np.zeros((len(idx), 3))
        n[np.arange(len(idx)), faces // 2] = np.where(faces % 2 == 0, 1.0, -1.0)
        Fn = -(F * n).sum(1)                                 # pressing load (> 0 pushes the grain onto the wall)
        Ft = F + Fn[:, None] * n
        Fc = np.where(h[:, dem.H_YLD] > 0.5, h[:, dem.H_FCP], p[:, dem.P_FC])
        R = p[:, dem.P_REFF]
        d = np.zeros(len(idx)) if dF is None else np.asarray(dF, float)
        Fn_lo = Fn - d
        Fncrit = np.maximum(Fn_lo + 2 * Fc, 0.0)
        ft = np.linalg.norm(Ft, axis=1) + d
        tw = np.abs((Tq * n).sum(1)) + d * R
        tt = np.linalg.norm(Tq - (Tq * n).sum(1)[:, None] * n, axis=1) + d * R
        r_pull = np.where(Fn_lo < 0, -Fn_lo / Fc, 0.0)
        r_slide = ft / np.maximum(p[:, dem.P_MUT] * Fncrit, 1e-300)
        r_roll = (ft + tt / R) / np.maximum(p[:, dem.P_MUR] * Fncrit, 1e-300)
        r_twist = tw / np.maximum(p[:, dem.P_MUTW] * Fncrit, 1e-300)
        kinds = np.vstack([r_pull, r_slide, r_roll, r_twist])
        return kinds.max(0), rows, Fn, Ft, Tq, n, kinds

    def _chord_force(self, idx, T0, T1):
        """The most the quasi-static load on each resting grain idx can deviate, anywhere inside [T0, T1], from the
        chord joining its values at the ends [N]: m (||Loads.chord_dev|| + Loads.spin_chord |r_perp|), r_perp the
        distance from the spin axis as in the centrifugal term of _body_force; the Euler force -Omega_dot x r and
        torque -I Omega_dot are piecewise constant inside a step and deviate by nothing. 0 on constant and linear
        load pieces, where the single-grain tests are exact at their samples."""
        cd = float(np.linalg.norm(self.loads.chord_dev(T0, T1)))
        sc = float(self.loads.spin_chord(T0, T1))
        r = self.x[idx] + self.off[None, :]
        ax = np.asarray(self.loads.axis, float)
        rp = np.linalg.norm(r - (r @ ax)[:, None] * ax[None, :], axis=1)
        return self.m[idx] * (cd + sc * rp)

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
        # sleeping clusters whose certificate held become what it advanced them to (one woken during the step, by a
        # redo or a DEM grain's approach, is no longer an island and keeps nothing); the DEM's own measure of the
        # motion of the clusters woken in this step is recorded
        for iid, new in list(getattr(self, "_isl_pending", {}).items()):
            if iid in self._islands:
                self._island_commit(iid, new)
        self._isl_pending = {}
        woken = getattr(self, "_woken_mem", [])
        if woken:
            dm = float(self._chk["dmax"][np.concatenate(woken)].max())
            self.n_events["island_wake_transient_m"] = max(self.n_events.get("island_wake_transient_m", 0.0), dm)
        self._island_report()
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
        for k in ("_chk", "_vd_step", "_ag_step", "_noexit", "_isl_pending", "_s_isl", "_s_isl_max", "_woken_mem"):
            st.pop(k, None)                                           # the last step's scratch state
        return st

    def __setstate__(self, st):
        """A state saved by this or a previous code version: the cap-sleep settings missing from an older one take
        their defaults and its counters exist (island dicts of the older format are upgraded at first use)."""
        self.__dict__.update(st)
        for k, v in ISLAND_DEFAULTS.items():
            if k not in self.__dict__:
                setattr(self, k, 2 * self.d_cap if v is None else v)
        for key in ISLAND_COUNTERS:
            self.n_events.setdefault(key, 0)
        for key in ISLAND_FLOATS:
            self.n_events.setdefault(key, 0.0)
        self._isl_pending = {}

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
        self._isl_pending, self._woken_mem = {}, []  # the failed attempt's advanced islands and wakes are discarded

    def _step_once(self, t, H, force, vfac, noexit=frozenset()):
        T = self.env.t0_flight + t
        self._update_long_range(t)
        v_start = self.v.copy()
        fast = self.loads.fast_window(T, T + H) or self.loads.breakpoint_in(T - 1e-9, T + 1e-9)
        # 1. asleep grains: certificate for the loads over the step (start, middle, end; the end load as the limit
        # from the left, the loads' deviation from their chord inside the step added as in statics.margins_bounded;
        # the same test as the entry, _try_sleep)
        asl = np.array([k for k in np.flatnonzero(self.regime == ASLEEP) if int(k) in self._sleep_face], np.int64)
        wake = np.zeros(self.N, bool)
        if len(asl):
            if fast:
                wake[asl] = True
            else:
                faces = np.array([self._sleep_face[k] for k in asl])
                worst = np.zeros(len(asl))
                dF = self._chord_force(asl, T, T + H)
                for Ts, side in ((T, None), (T + 0.5 * H, None), (T + H, "-")):
                    F, Tq = self._body_force(asl, Ts, side)
                    ratio = self._margins(asl, faces, F, Tq, dF)[0]
                    worst = np.maximum(worst, ratio)
                wake[asl[worst > 1 - self.margin]] = True
        # sleeping clusters: a fast load or a breakpoint wakes them; a tier-0 island (the linear certificate of
        # v0.6: cap_sleep off, a tracked residual motion, or an island of the previous code version) is tested on its
        # linear response at the start, middle and end of the step (the end load as the limit from the left); the
        # others are advanced to the step's end by _island_advance, their end state pending until the step is
        # verified (_island_commit in step), and the most each member can move inside the step (s_isl) is its
        # displacement allowance in every motion bound below
        self._isl_pending = {}
        self._woken_mem = []
        s_isl = np.zeros(self.N)
        for iid in list(self._islands):
            isl = self._islands[iid]
            if fast:
                wake[isl["members"]] = True
                self._count("island_wake_fast")
                continue
            if self.cap_sleep and "tier" not in isl and "res" not in isl:
                isl = self._islands[iid] = self._island_upgrade(isl, T)
            if isl.get("tier", 0) == 0 or "res" in isl:
                worst = 0.0
                for Ts, side in ((T, None), (T + 0.5 * H, None), (T + H, "-")):
                    worst = max(worst, float(self._island_ratio(isl, Ts, side).max()))
                if worst > 1 - self.margin:
                    wake[isl["members"]] = True
                    self._count("island_wake_loading")
                continue
            ok, new, why = self._island_advance(isl, T, T + H)
            if ok:
                self._isl_pending[iid] = new
                s_isl[isl["members"]] = new["s_isl"]
            else:
                wake[isl["members"]] = True
                self._count("island_wake_" + why)
        # two sleepers of different groups (islands, single grains) that could touch within the step: both wake, and
        # the full model resolves the approach from the committed state (no DEM grain is near them to see it)
        if not fast:
            for g in self._sleeper_contacts(s_isl, wake):
                wake[g] = True
                self._count("island_wake_neighbour")
        self._s_isl = s_isl
        self._s_isl_max = float(s_isl.max()) if self.N else 0.0
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
        # motion bounds: |v| H + a_max H^2 / 2 (body forces), the rolling envelope for rolling grains, the drift of
        # a sleeping island (0 for a single sleeper and a tier-0 island)
        s = self._motion_bound(H, T)
        s[self.regime == ASLEEP] = s_isl[self.regime == ASLEEP]
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
        # (sleepers need no speed bound: a sleeping island's drift is its displacement allowance s_isl in s, and a
        # member woken inside the step gets local bounds where it joins, _early_exit)
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
            s_full[self.regime == ASLEEP] = self._s_isl[self.regime == ASLEEP]
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
        # within dmax of where it entered, and must be clear of every grain outside the DEM at the same time (a
        # member of a sleeping cluster woken by an approach too: placed within s_isl of its committed position,
        # which the others' bounds allowed for, its dmax is measured from where it was placed)
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
        * free and sleeping grains outside the DEM all step: straight-line relative motion plus curvature bounds (a
          sleeping cluster's drift allowance s_isl, its entry in s_full, added to the curvature);
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
                    else:                         # asleep: within s_full[j] (= s_isl) of its committed position
                        d0 = self.x[j] - self.x[k]
                        dv = -self.v[k]
                        curv = 0.5 * a_k[k] * rem * rem + off_k + s_full[j]
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

    def _drift(self):
        """Each sleeper's displacement allowance inside the current step (the s_isl of its island, set by _step_once;
        0 for a single sleeper, and for every grain before the first step) and the largest of them."""
        s = getattr(self, "_s_isl", None)
        return (s, self._s_isl_max) if s is not None else (np.zeros(self.N), 0.0)

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
        A single sleeping grain does not move; a sleeping cluster may drift by at most s_isl (its certificate's
        allowance, _step_once), by which its gaps are shrunk (their certificates hold for the step; one that joins
        the DEM is checked afterwards, see _verify). The speed bound of each grain in the DEM is local
        (_speed_bounds) and checked against its actual speed afterwards. Returns the times and which DEM grains'
        bounds were used."""
        d_cap = self.d_cap
        vm = np.linalg.norm(self.v, axis=1)
        a_bound = np.maximum((s - s_off - vm * H) * 2 / (H * H), 0.0)   # per-grain acceleration bound
        vd, ag = self._vd_step, self._ag_step
        s_isl = self._drift()[0]
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
                    continue                      # checked exactly by the pre-run (an obstacle grown by its drift
                                                  # allowance, _slv_obstacles, or the same cluster)
                if rj == ACTIVE:                  # (it may roll on a wall later: its rocking offset too)
                    gap = float(np.linalg.norm(self.x[jj] - self.x[k])) - rsum - s_off[k] - off_any[jj]
                    tp = t_iso(gap, vm_e[k] + vd[jj], a_bound[k] + ag[jj])
                    used[jj] = True
                elif k_roll or k_slv or rj == ROLL or rj == SLAV:
                    gap = float(np.linalg.norm(self.x[jj] - self.x[k])) - rsum - s_off[k] - s_off[jj]
                    vj = 0.0 if rj == ASLEEP else vm_e[jj]
                    if rj == ASLEEP:              # a sleeping cluster: anywhere within s_isl of its committed
                        gap -= s_isl[jj]          # position (its a_bound, 2 s_isl / H^2, allows for it once more)
                    tp = t_iso(gap, vm_e[k] + vj, a_bound[k] + a_bound[jj])
                elif rj == FREE:
                    tp = t_lin(self.x[jj] - self.x[k], self.v[jj] - self.v[k], rsum, a_bound[k] + a_bound[jj])
                else:                             # asleep: its drift allowance widens the capture shell
                    tp = t_lin(self.x[jj] - self.x[k], -self.v[k], rsum + s_isl[jj], a_bound[k])
                best = min(best, tp)
            out[q] = best
        return out, used

    def _skip_asleep(self, t, t_out=None):
        """Everything is asleep: advance time without stepping while every sleeper's certificate holds, checked at
        the same sample points the stepping would use (every dt_s), up to the next load breakpoint or fast load.
        Single sleepers are tested at every sample (the sample loads as limits from the left, the loads' chord
        deviation over the sample interval added, as in _step_once); sleeping clusters are advanced from sample to
        sample in lockstep (_island_advance, the sample loads as limits from the left, each island's state committed
        at once when its step holds; a tier-0 island keeps its linear test) and tested against each other and the
        single sleepers (_sleeper_contacts, with each island's allowance for the sample), all of them stopping at the
        first sample any one fails, so that every island's committed state is at the time returned (n_events is
        restored to what it was before that sample: the islands that passed it are not counted as having slept it,
        critique D2, found by a unit test, and the solves, refactorisations and twist settings of a sample the
        stepping redoes are not counted either). Nothing is skipped from
        a load breakpoint (a skip that started there would carry the committed pre-jump states across the jump as if
        the post-jump load had been reached quasi-statically, dropping the jump's transient: found by check 19,
        island_across_breakpoint); step() wakes the sleepers there instead. The long-range forces
        _F_lr are not updated during a skip: they stay those of the last step (their change is charged to the first
        step after it, _island_advance). Returns the new time (t if nothing can be skipped)."""
        T = self.env.t0_flight + t
        dts = self.base.dt_base
        t_lim = min(self.duration, self.loads.next_breakpoint(T) - self.env.t0_flight, t + 2.0,
                    t_out if t_out is not None and t_out > t else self.duration)
        if self.loads.fast_window(T, T + dts) or t_lim <= t + 2 * dts:
            return t
        if self.loads.breakpoint_in(T - 1e-9, T + 1e-9):
            return t                              # step() wakes every sleeper at a load jump
        n = int((t_lim - t) / dts)
        Ts = T + dts * np.arange(1, n + 1)
        asl = np.array([k for k in np.flatnonzero(self.regime == ASLEEP) if int(k) in self._sleep_face], np.int64)
        ok_until = n
        if len(asl):
            faces = np.array([self._sleep_face[k] for k in asl])
            for q, Tq in enumerate(Ts):
                F, Tqq = self._body_force(asl, Tq, "-")
                dF = self._chord_force(asl, T if q == 0 else Ts[q - 1], Tq)
                if float(self._margins(asl, faces, F, Tqq, dF)[0].max()) > 1 - self.margin:
                    ok_until = q
                    break
        for iid in list(self._islands):
            isl = self._islands[iid]
            if self.cap_sleep and "tier" not in isl and "res" not in isl:
                self._islands[iid] = self._island_upgrade(isl, T)
        for q in range(min(ok_until, n)):
            pend = {}
            ev0 = dict(self.n_events)
            s = np.zeros(self.N)
            for iid, isl in self._islands.items():
                if isl.get("tier", 0) == 0 or "res" in isl:
                    if float(self._island_ratio(isl, Ts[q], "-").max()) > 1 - self.margin:
                        pend = None
                        break
                    continue
                ok, new, why = self._island_advance(isl, T if q == 0 else Ts[q - 1], Ts[q])
                if not ok:
                    pend = None
                    break
                pend[iid] = new
                s[isl["members"]] = new["s_isl"]
            if pend is not None and self._sleeper_contacts(s):
                pend = None                        # two sleepers could touch: the stepping wakes them
            if pend is None:
                # the stepping redoes this sample: no island slept it, and none of its solves, refactorisations or
                # twist settings was made for a step taken (all counters as before the sample)
                self.n_events = ev0
                ok_until = q
                break
            for iid, new in pend.items():
                self._island_commit(iid, new)
        if ok_until < 1:
            return t
        self._island_report()
        return t + ok_until * dts

    def _jump_side(self, T, T_since):
        """The side of the load a sleeper woken at T is placed for: "-" (the limit from the left) when T is a load
        breakpoint and the sleeper was certified up to it from an earlier time T_since, so that the full model starts
        from the pre-jump equilibrium and resolves the jump itself (critique A2); None otherwise, and at the time a
        sleeper was placed (a run's start at a breakpoint begins after the jump: __init__ placed it there)."""
        return "-" if T > T_since + 1e-12 and self.loads.breakpoint_in(T - 1e-9, T + 1e-9) else None

    def _wake(self, k, T):
        """Wake a single sleeper at T: placed at the exact static equilibrium of its wall contact under the load at T,
        at rest; at a load breakpoint under the load just before it (_jump_side: the grain's certificate ran up to the
        jump, and a placement at the post-jump equilibrium would drop the jump's transient, whose overshoot can carry
        a spring to its limit and slip it, critique A2)."""
        face = self._sleep_face.pop(k)
        self._last_face = face
        row = self._key[(int(k), -1 - face)]
        F = Tq = None
        side = self._jump_side(T, float(self.env.t0_flight))
        if side:
            F, Tq = self._body_force(np.array([k]), T, side)
        self._place_at_equilibrium(k, face, row, T, F, Tq)
        self.v[k] = 0.0
        self.w[k] = 0.0
        self.n_events["wake"] += 1

    def _allowance(self, idx):
        """The displacement allowance of sleepers idx outside a step: the s_isl their island's last step committed, 0
        for a single sleeper and a tier-0 island (neither moves)."""
        s = np.zeros(len(idx))
        for q, k in enumerate(idx):
            isl = self._islands.get(self._island_of.get(int(k), -1))
            if isl is not None and isl.get("tier", 0) > 0 and "res" not in isl:
                s[q] = float(isl.get("s_isl", 0.0))
        return s

    def _sleeper_contacts(self, s, skip=None):
        """Pairs of sleepers of different groups (two islands, or an island and a single sleeper) whose gap could close
        within the step: |x_a - x_b| - r_a - r_b <= s_a + s_b at the committed positions, s (N,) each one's
        displacement allowance for the step (an island's s_isl, 0 for a single sleeper and a tier-0 island, which do
        not move). An island's own system lists only its member pairs and member-wall pairs (_island_near), _watch
        compares sleepers with DEM grains and _capture_times with free, rolling and slaved ones, so this is the only
        test between two sleepers (when every sleeper was static none was needed; a drifting island can close a gap
        to a sleeper next to it, measured: a tier-2 pair rising 3e-14 m per step closed a 1.5e-13 m gap to a grain
        asleep above it in 5 steps). Any positive gap left is exact: the full model resolves the approach from there.
        skip (N,) bool: grains being woken anyway (_watch covers them in the DEM). Returns the members of both sides,
        one array per pair of groups."""
        sl = np.flatnonzero(self.regime == ASLEEP)
        if skip is not None:
            sl = sl[~skip[sl]]
        if len(sl) < 2 or not (s[sl] > 0).any():
            return []
        pr = cKDTree(self.x[sl]).query_pairs(2 * float(self.rad[sl].max()) + 2 * float(s[sl].max()),
                                             output_type="ndarray")
        if not len(pr):
            return []
        a, b = sl[pr[:, 0]], sl[pr[:, 1]]
        grp = np.array([self._island_of.get(int(k), -1 - int(k)) for k in sl])
        ga, gb = grp[pr[:, 0]], grp[pr[:, 1]]
        gap = np.linalg.norm(self.x[a] - self.x[b], axis=1) - self.rad[a] - self.rad[b] - s[a] - s[b]
        out = {}
        sel = (ga != gb) & (s[a] + s[b] > 0) & (gap <= 0)
        for i, j in zip(ga[sel].tolist(), gb[sel].tolist()):
            out.setdefault((min(i, j), max(i, j)), np.concatenate(
                [self._islands[g]["members"] if g >= 0 else [-1 - g] for g in (i, j)]).astype(np.int64))
        return list(out.values())

    def _sleepers_near(self, mem, x, s):
        """Whether a sleeper outside mem lies within the reach of the grains mem about to be put to sleep at x with the
        allowance s: gap <= s + its own allowance (_allowance), the test _sleeper_contacts makes at every step, so
        that a cluster or a grain is not placed only to be woken at once."""
        sl = np.setdiff1d(np.flatnonzero(self.regime == ASLEEP), np.asarray(mem, np.int64))
        if not len(sl):
            return False
        so = self._allowance(sl)
        if s <= 0.0 and not (so > 0).any():
            return False                          # neither side moves
        x = np.atleast_2d(x)
        rm = self.rad[np.asarray(mem, np.int64)]
        for q, lst in enumerate(cKDTree(self.x[sl]).query_ball_point(x, float(rm.max() + self.rad[sl].max())
                                                                      + s + float(so.max()))):
            if lst:
                jj = np.array(lst, np.int64)
                if (np.linalg.norm(self.x[sl[jj]] - x[q], axis=1) - self.rad[sl[jj]] - rm[q] - s - so[jj] <= 0).any():
                    return True
        return False

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
        """Sleepers within 2 d_cap of a DEM grain join it (repeated: a joined grain may be that close to another); a
        sleeping cluster joins whole, at its state for the load now (_wake_island, counted island_wake_approach).
        A sleeping cluster may be anywhere within its drift allowance s_isl of its committed position (_step_once;
        0 for a single sleeper), so every gap to it is shrunk by s_isl: a DEM grain joins it when gap - s_isl <=
        2 d_cap and may move gap - s_isl - 2 d_cap until the next list build, so that no grain sits in a band where
        it may move by nothing yet does not join (the DEM would take one step per list build across it). Returns
        the DEM set and each DEM grain's movement allowance until the next list build."""
        d2 = 2 * self.d_cap
        rmax = float(self.rad.max())
        s_isl, s_max = self._drift()
        while True:
            lim = np.full(len(ai), 0.5 * skin)
            sl = np.flatnonzero(self.regime == ASLEEP)
            if not len(sl):
                return ai, lim
            xs = self.x[sl]
            # the committed positions change only at commits (step ends): the tree is reused while the set of
            # sleepers and their positions are the same
            c = getattr(self, "_sl_tree", None)
            if c is None or not (np.array_equal(c[0], sl) and np.array_equal(c[1], xs)):
                c = self._sl_tree = (sl, xs.copy(), cKDTree(xs))
            tree = c[2]
            join = set()
            for q, lst in enumerate(tree.query_ball_point(self.x[ai], 2 * rmax + skin + s_max)):
                if not lst:
                    continue
                jj = sl[np.array(lst, np.int64)]
                gap = (np.linalg.norm(self.x[jj] - self.x[ai[q]], axis=1) - self.rad[jj] - self.rad[ai[q]]
                       - s_isl[jj])
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
                    self._count("island_wake_approach")
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
            dF = self._chord_force(ro, T, T + Hn)
            for Ts, side in ((T, None), (T + Hn, "-")):
                F, Tq = self._body_force(ro, Ts, side)
                worst = np.maximum(worst, self._margins(ro, faces, F, Tq, dF)[0])
            for k in ro[worst <= 1 - self.margin]:
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
        """Sleep a grain on one wall. Entry is the stay test of _step_once: every load ratio (pull-off, sliding,
        rolling, twisting) <= 1 - margin at T and at T + H (the end load as the limit from the left) with the loads'
        deviation from their chord inside the step added (_margins with _chord_force: exact at the samples plus the
        deviation between them), so a grain that falls asleep holds its first step. Against chatter the ratios of
        each kind are extrapolated linearly from their values at T and T + H over island_hyst_steps steps (to the
        next breakpoint at most), and the grain stays awake, counted sleep_refused_drift, when that exceeds
        1 - margin. Its residual motion must not carry any contact load past its limit: with E = kinetic energy +
        the energy of its deviation from static equilibrium, the largest extra spring force sqrt(2 k_max E) must be
        below 1% of the smallest remaining margin. A single contact at its cap is a mechanism (nothing else holds
        the grain), so a grain above 1 - margin is left to _try_roll (ROLL). A grain within the drift allowance of a
        sleeping island stays awake (sleep_refused_neighbour, _sleepers_near): the next step would wake both."""
        if H <= 0 or self.loads.fast_window(T, T + H) or self.loads.breakpoint_in(T - 1e-9, T + H):
            return False
        row = self._key[(int(k), -1 - face)]
        p, h = self._prm[row], self._hist[row]
        idx, faces = np.array([k]), np.array([face])
        dF = self._chord_force(idx, T, T + H)
        F, Tq = self._body_force(idx, T)
        r0, _, Fn, _, _, _, k0 = self._margins(idx, faces, F, Tq, dF)
        F1, Tq1 = self._body_force(idx, T + H, "-")
        r1, _, _, _, _, _, k1 = self._margins(idx, faces, F1, Tq1, dF)
        worst = max(float(r0[0]), float(r1[0]))
        if worst > 1 - self.margin:
            return False
        Tx = min(T + self.island_hyst_steps * H, self.loads.next_breakpoint(T))
        if float((k0 + (k1 - k0) * ((Tx - T) / H)).max()) > 1 - self.margin:
            self._count("sleep_refused_drift")
            return False
        Fc = p[dem.P_FC]
        Fncrit = abs(float(Fn[0]) + 2 * Fc)
        margin_force = (1 - worst) * min(p[dem.P_MUR], p[dem.P_MUT]) * Fncrit
        # deviation from equilibrium: kinetic energy and the energy of the unbalanced static load
        a = max(h[dem.H_A], 1e-12)
        kn = 1.2 * p[dem.P_ESTAR] * a
        kmax = max(kn, p[dem.P_KT] * a, p[dem.P_KR])
        Ek = 0.5 * self.m[k] * float(self.v[k] @ self.v[k]) + 0.5 * self.I[k] * float(self.w[k] @ self.w[k])
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
        if self._sleepers_near([k], x_eq, 0.0):
            self._count("sleep_refused_neighbour")
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

    def _island_load(self, mem, T, side=None):
        """Load vector (6 per grain: force, torque) on the members at T; side "-" the limit from the left."""
        F, Tq = self._body_force(mem, T, side)
        return np.hstack([F, Tq]).reshape(-1)

    def _island_ratio(self, isl, T, side=None):
        """The linear certificate of a tier-0 island (cap_sleep off, a tracked residual motion, or an island of the
        previous code version): the worst load ratio of the linear response lam_ref + J K^-1 dF to the load at T."""
        dF = self._island_load(isl["members"], T, side) - isl["F_ref"]
        du = self._island_lu(isl).solve(dF)
        lam = isl["lam"] + isl["J"] @ du
        with np.errstate(divide="ignore", invalid="ignore"):      # an inactive near pair's 0/0 pull-off ratio (-> 0)
            if "dl" in isl:                   # sleeping with a tracked residual motion: its largest effect included
                return statics.margins_bounded(lam, isl["dl"], isl["prm"], isl["out"])
            return statics.margins(lam, isl["prm"], isl["out"])

    # ------------------------------------------------------------------ sleeping clusters at their caps
    def _count(self, key, n=1):
        self.n_events[key] = self.n_events.get(key, 0) + n

    def _awake(self, why):
        """A cluster stays in the full model; why it could not sleep is counted (cluster_awake_<why>)."""
        self._count("cluster_awake_" + why)
        return False

    @staticmethod
    def _island_bnd0():
        """An island's bounds at zero (the metre entries per kind: sliding, rolling [m], twisting [rad])."""
        return dict(newton_rel=0.0, dyn_rel=0.0, curv_rel=0.0, curv_m=np.zeros(3), flow_tol_m=np.zeros(3),
                    ratchet_m=np.zeros(3), path_m=np.zeros(3), flow_m=np.zeros(3), resid_m=np.zeros(3),
                    dyn_m=np.zeros(3), snap_rel=0.0, snap_m=np.zeros(3), total_rel=0.0)

    @staticmethod
    def _island_flags(E):
        """The wake reason a solve_equilibrium result carries; None if it converged with no contact transition."""
        if E["broke"]:
            return "break"
        if E["formed"]:
            return "newcontact"
        if E["yielded"]:
            return "yield"
        if E["mechanism"]:
            return "mechanism"
        return None if E["ok"] else "newton"

    @staticmethod
    def _ratios(lam, D, prm, out, tm):
        """statics.ratios_by_kind_bounded on the touching contacts tm (an inactive near pair has no cap and no
        pull-off force): the pull-off ratios (n_t,) and the sliding, rolling and twisting ratios (n_t x 3); D None
        for the plain ratios."""
        nc = len(prm)
        l8 = lam.reshape(nc, 8)[tm].reshape(-1)
        d8 = np.zeros_like(l8) if D is None else D.reshape(nc, 8)[tm].reshape(-1)
        rp, rs, rr, rt = statics.ratios_by_kind_bounded(l8, d8, prm[tm], out[tm])
        return rp, np.column_stack([rs, rr, rt])

    @staticmethod
    def _fhat(lam):
        """Unit directions (n_contacts x 3 x 3) of the sliding and rolling forces (the twist's drops out of
        statics.released_stiffness)."""
        L8 = lam.reshape(-1, 8)
        fh = np.zeros((L8.shape[0], 3, 3))
        fh[:, 0] = L8[:, 1:4] / np.maximum(np.linalg.norm(L8[:, 1:4], axis=1), 1e-300)[:, None]
        fh[:, 1] = L8[:, 4:7] / np.maximum(np.linalg.norm(L8[:, 4:7], axis=1), 1e-300)[:, None]
        return fh

    @staticmethod
    def _kind_norms(v8):
        """A per-slot vector (8 per contact) reduced per kind: the norms over the sliding and rolling slots and the
        absolute twist slot (n_contacts x 3)."""
        V = v8.reshape(-1, 8)
        return np.column_stack([np.linalg.norm(V[:, 1:4], axis=1), np.linalg.norm(V[:, 4:7], axis=1),
                                np.abs(V[:, 7])])

    @staticmethod
    def _island_row_norms(lu, J, block=256):
        """||K^-T J_k^T||_2 for every component row k of J over the translational and over the rotational degrees of
        freedom separately (8 n_contacts x 2), by block transposed solves as statics.row_norms; then |(J K^-1 f)_k|
        <= rn_t,k ||f_trans||_2 + rn_r,k ||f_rot||_2 (Cauchy-Schwarz on each part). The one norm over all degrees of
        freedom (statics.row_norms) is a valid bound too but mixes units: the rotational entries, ~1/R per N m, are
        1e4 times the translational ones per N, so a force residual of 1e-16 N would be charged as if it were a
        torque of 1e-16 N m (measured: 3.7e-4 of a rolling cap for the pair on a floor, ~1e4 times too loose)."""
        Jt = J.T.tocsc()
        rn = np.empty((J.shape[0], 2))
        rot = (np.arange(J.shape[1]) % 6) >= 3
        for a in range(0, J.shape[0], block):
            b = min(a + block, J.shape[0])
            X = lu.solve(Jt[:, a:b].toarray(), trans='T')
            rn[a:b, 0] = np.linalg.norm(X[~rot], axis=0)
            rn[a:b, 1] = np.linalg.norm(X[rot], axis=0)
        return rn

    @staticmethod
    def _island_disp_norm(lu, n, block=192):
        """max over the translational degrees of freedom i of n grains of ||(K^-1)_i, trans||_2, the row of K^-1 over
        the force components (lu the factor of K): |(K^-1 f)_i| <= this times ||f||_2 for every force vector f without
        torques (Cauchy-Schwarz), the most a load deviation of a given size can move any member, through every
        contact of its chain (block transposed solves, as _island_row_norms)."""
        tr = np.flatnonzero((np.arange(6 * n) % 6) < 3)
        best = 0.0
        for a in range(0, len(tr), block):
            c = tr[a:a + block]
            E = np.zeros((6 * n, len(c)))
            E[c, np.arange(len(c))] = 1.0
            X = lu.solve(E, trans='T')
            best = max(best, float(np.linalg.norm(X[tr], axis=0).max()))
        return best

    @staticmethod
    def _island_dashpots(prm, hist):
        """The kernel's dashpot coefficients per contact (n_contacts x 5): normal dpre (P_DAMPC sqrt(a) or P_DAMPC a
        by the damping mode), tangential x_gt dpre, rolling g_r (force per rolling speed), g_r R^2 (torque per
        angular speed) and the twisting g_tw."""
        a = np.maximum(hist[:, dem.H_A], 0.0)
        mode = prm[:, dem.P_DAMPMODE].astype(np.int64)
        dpre = np.abs(np.where(mode == dem.DAMP_TSUJI_JKR, prm[:, dem.P_DAMPC] * np.sqrt(a),
                               np.where(mode == dem.DAMP_VISCOELASTIC, prm[:, dem.P_DAMPC] * a, 0.0)))
        gr = prm[:, dem.P_GR]
        return np.column_stack([dpre, prm[:, dem.P_XGT] * dpre, gr, gr * prm[:, dem.P_REFF] ** 2, prm[:, dem.P_GTW]])

    def _island_tols(self, capv, prm, tm):
        """Per-DOF Newton tolerances: island_tol_rel of the smallest force cap (sliding, rolling) and of the smallest
        torque cap (twisting, or the rolling cap times the contact radius) over the touching contacts tm."""
        if not tm.any():
            return 0.0, 0.0
        c, tol = capv[tm], self.island_tol_rel
        return tol * float(c[:, :2].min()), tol * float(min(c[:, 2].min(), (prm[tm, dem.P_REFF] * c[:, 1]).min()))

    def _island_near(self, mem, ci, cj, rows):
        """The island's contact system: the touching contacts given plus every member pair and member-wall pair with a
        gap <= island_gap_m as an inactive contact (H_TOUCH = 0: the kernel gives it no force and no stiffness until
        the gap closes, when solve_equilibrium's formed flag reports it). A static island cannot close a gap, a
        creeping one can; the system is relisted at every refresh (_island_refresh), between which the drift stays
        far below island_gap_m, so no contact between members or with a wall can form unseen (one with a grain
        outside the island: _watch for the DEM's, _sleeper_contacts for other sleepers')."""
        have = set(zip(ci.tolist(), cj.tolist()))
        ci, cj, rows = list(ci), list(cj), list(rows)
        x, rad, gap = self.x[mem], self.rad[mem], self.island_gap_m
        if len(mem) > 1:
            pr = cKDTree(x).query_pairs(2 * float(rad.max()) + gap, output_type="ndarray")
            for a, c in pr.tolist():
                a, c = (a, c) if a < c else (c, a)          # members are sorted: a row's key is (smaller, larger)
                if (a, c) in have or float(np.linalg.norm(x[a] - x[c])) - rad[a] - rad[c] > gap:
                    continue
                ci.append(a)
                cj.append(c)
                rows.append(self._row(int(mem[a]), int(mem[c])))
        for f in range(6):
            ax = f // 2
            g = (x[:, ax] if f % 2 == 0 else self.L[ax] - x[:, ax]) - rad
            for k in np.flatnonzero(g <= gap).tolist():
                if (k, -1 - f) in have:
                    continue
                ci.append(k)
                cj.append(-1 - f)
                rows.append(self._row(int(mem[k]), -1 - f))
        return np.array(ci, np.int64), np.array(cj, np.int64), np.array(rows, np.int64)

    def _island_chord(self, mem, T0, T1):
        """||dF||_2 of the loads' chord deviation over [T0, T1] on the island's grains (_chord_force per grain; the
        rotational components deviate by nothing)."""
        return float(np.linalg.norm(self._chord_force(mem, T0, T1)))

    def _island_fdyn(self, isl, hist, du, l, T):
        """(||f_dyn,trans||_2, ||f_dyn,rot||_2): the forces and torques the drift du over l would need were it dynamic
        rather than quasi-static, per grain m |du| / l^2 plus the Coriolis and drag terms (2 m Omega + m g_drag) |du| /
        l and the dashpots' forces at the drift's mean rate, |C du| / l, C = -dF/dv the island's damping matrix with
        the caps lifted (statics.jacobians with vel, at its last refresh: the kernel's dashpots are linear in the
        relative velocities, and at a capped spring the clipped force carries no more than the lifted one); the
        rotations likewise with I and the gyroscopic I Omega. An island without C (committed before it existed)
        takes the previous bound (n_c g_max) |du| / l instead, g_max the largest translational dashpot coefficient
        of its contacts (normal, tangential, rolling; _island_dashpots) and n_c the grain's number of contacts,
        likewise g_r R^2 and g_tw for the rotations: valid but loose, since it applies the normal and sliding
        dashpots to every motion (1500x the rolling dashpot's force for a grain creeping by rolling, measured on
        check 14's corner grain)."""
        mem, ci, cj = isl["members"], isl["ci"], isl["cj"]
        tm = hist[:, dem.H_TOUCH] > 0.5
        if not tm.any():
            return 0.0, 0.0
        Om = float(np.linalg.norm(self.loads.omega(np.array([T]))[0]))
        m, I = self.m[mem], self.I[mem]
        dU = np.abs(np.asarray(du).reshape(-1, 6))
        if isl.get("Cd") is not None:
            fd = np.abs(isl["Cd"] @ np.asarray(du, float)).reshape(-1, 6) / l
            ft = (m / l ** 2 + (2 * m * Om + m * self.gdrag[mem]) / l)[:, None] * dU[:, :3] + fd[:, :3]
            fr = (I / l ** 2 + I * Om / l)[:, None] * dU[:, 3:] + fd[:, 3:]
            return float(np.linalg.norm(ft)), float(np.linalg.norm(fr))
        g = self._island_dashpots(isl["prm"][tm], hist[tm])
        gt, gr = float(g[:, :3].max()), float(g[:, 3:].max())
        nc = np.zeros(len(mem))
        np.add.at(nc, ci[tm], 1.0)
        pair = cj[tm] >= 0
        np.add.at(nc, cj[tm][pair], 1.0)
        ft = (m / l ** 2 + (nc * gt + 2 * m * Om + m * self.gdrag[mem]) / l)[:, None] * dU[:, :3]
        fr = (I / l ** 2 + (nc * gr + I * Om) / l)[:, None] * dU[:, 3:]
        return float(np.linalg.norm(ft)), float(np.linalg.norm(fr))

    def _island_tangent(self, isl, count):
        """A closure for solve_equilibrium's tangent argument (the creep solves): the released stiffness
        (statics.released_stiffness) of the island's lifted tangents at every sliding, rolling or twisting component
        whose ratio in the state reached is >= 1 - 1e-9 (the return-mapped components sit at their caps to rounding),
        factorised anew only when that set changes (counted island_refactor); the island's own factor when nothing is
        capped. A factorisation that fails raises RuntimeError, which solve_equilibrium reports as a mechanism (one
        that succeeds on a round-off pivot gives an absurd step, caught there as a break or a non-contraction)."""
        from scipy.sparse.linalg import splu
        K, J, prm, lu = isl["K"], isl["J"], isl["prm"], self._island_lu(isl)
        cache = {}

        def tangent(xn, hn, lam, out):
            tm = hn[:, dem.H_TOUCH] > 0.5
            mask = np.zeros((len(prm), 3), bool)
            mask[tm] = self._ratios(lam, None, prm, out, tm)[1] >= 1 - 1e-9
            if not mask.any():
                return lu
            key = mask.tobytes()
            if cache.get("key") != key:
                cache["fac"] = splu(statics.released_stiffness(K, J, statics.kcomp(prm, hn), mask, self._fhat(lam)))
                cache["key"] = key
                if count:
                    self._count("island_refactor")
            return cache["fac"]
        return tangent

    def _island_drift(self, K, J, kc, near, fhat, mem, dF, dFh, l):
        """The kinetic energy [J] of a cluster's motion relative to the quasi-static drift of its equilibrium, the part
        of its motion the engine drops when it puts the cluster to sleep (the drift itself is what the sleeping
        cluster's steps follow): (|v - u0|_M + |u_h - u_f|_M)^2 / 2, v the grains' velocities and spins (6 per
        grain), M their masses and moments of inertia, u_f = Ks^-1 dF / l and u_h = 2 Ks^-1 dFh / l the mean drift
        rates over the step and over its first half (dF, dFh the load changes over them, Ks the symmetric part of the
        stiffness), u0 = 2 u_h - u_f the rate at the step's start (exact for loads quadratic in time) and
        |u_h - u_f|_M its uncertainty; the larger of the values in the elastic system and in the released one
        (statics.released_stiffness over the components flagged in near, at their caps), since at a cap the drift
        may run through the released system instead. Under a constant load the drift is 0 and this is the kinetic
        energy itself. The previous entry test charged the whole kinetic energy plus |dF| |K^-1 dF| 4 / (omega_min
        H)^2 (the oscillation a cluster placed at rest would keep while its equilibrium moves), so a cluster that the
        full model carries along a slowly turning load at its caps, which moves with the drift and not about it, was
        refused at every step (check 14, turning load: 5e-22 J charged against 6e-28 J relative to the drift for a
        grain in an edge). Infinite when a stiffness does not factorise or a rate is not finite (refused as a
        mechanism)."""
        from scipy.sparse.linalg import splu
        Md = np.column_stack([self.m[mem]] * 3 + [self.I[mem]] * 3).reshape(-1)
        v = np.hstack([self.v[mem], self.w[mem]]).reshape(-1)

        def nm(a):
            return math.sqrt(max(float(a @ (Md * a)), 0.0))
        E = 0.0
        for Kx in ((K, statics.released_stiffness(K, J, kc, near, fhat)) if near.any() else (K,)):
            try:
                lu = splu((0.5 * (Kx + Kx.T)).tocsc())
            except RuntimeError:
                return math.inf
            uf = lu.solve(np.asarray(dF, float)) / l
            uh = lu.solve(np.asarray(dFh, float)) * (2.0 / l)
            if not (np.isfinite(uf).all() and np.isfinite(uh).all()):
                return math.inf
            E = max(E, 0.5 * (nm(v - (2.0 * uh - uf)) + nm(uh - uf)) ** 2)
        return E

    def _island_ratchet(self, K, J, prm, hist, lam, out, kc, capv, E_res, mem):
        """How far an entry residual of energy E_res (kinetic, elastic deviation from the equilibrium, and what the
        step's load change can pump in) can slip any sliding, rolling or twisting component near its cap, bounded in
        the RELEASED system (statics.released_stiffness over every such component): a capped spring holds its cap
        while the rest of the structure restores with the released stiffness, so the residual's displacement
        amplitude there, dl_rel = statics.energy_bounds(K_rel, J, E_res), is permanent slip (the load, not the
        residual's energy, does the plastic work), plus the dashpot's shift of a clipped history at the residual's
        speed, g v_E / k with v_E = sqrt(2 E_res / m_min) for sliding, R_max sqrt(2 E_res / I_min) for rolling and
        sqrt(2 E_res / I_min) for the twist rate. A component is near its cap when its ratio plus the residual's
        bound reaches 1 - sleep_margin (the margin standing in for the step's deviation bound, unknown at entry),
        first with the elastic bounds, then with the released ones, iterated until the released set is the near set:
        releasing a component softens the structure (released_stiffness subtracts positive semi-definite terms) and
        raises every bound, which may bring another component near, and a component near only by the larger bound
        must be released too (one pass under-bounded every slip: by 10x in a two-spring example, accepting a slip of
        5e-3 of the deflection at the cap); the others cannot flow. Returns
        (dl_rel (8 n_contacts), slip (n_contacts x 3, 0 where not near), near (n_contacts x 3)), or None when the
        released system is a mechanism: energy_bounds None, non-finite or above 1e3 times the elastic bounds
        (statics.released_stiffness: the singularity holds only to rounding)."""
        tm = hist[:, dem.H_TOUCH] > 0.5
        dl = statics.energy_bounds(K, J, E_res)
        if dl is None or not np.isfinite(dl).all():
            return None
        sds = self._ratios(lam, None, prm, out, tm)[1]

        def near_of(d):
            d3 = self._kind_norms(d)
            nr = np.zeros((len(prm), 3), bool)
            nr[tm] = sds + d3[tm] / np.maximum(capv[tm], 1e-300) >= 1 - self.margin
            return nr, d3
        near, d3 = near_of(dl)
        dlr = dl
        done = np.zeros_like(near)
        while (near & ~done).any():               # to a fixed point: released = near (at most 3 n_contacts passes)
            done = near
            dlr = statics.energy_bounds(statics.released_stiffness(K, J, kc, near, self._fhat(lam)), J, E_res)
            if dlr is None or not np.isfinite(dlr).all() or float(dlr.max()) > 1e3 * max(float(dl.max()), 1e-300):
                return None
            near2, d3 = near_of(dlr)
            near = near | near2
        g = self._island_dashpots(prm, hist)[:, [1, 2, 4]]
        m_min, I_min, R_max = float(self.m[mem].min()), float(self.I[mem].min()), float(self.rad[mem].max())
        vE = np.array([math.sqrt(2 * E_res / m_min), R_max * math.sqrt(2 * E_res / I_min),
                       math.sqrt(2 * E_res / I_min)])
        slip = (d3 + g * vE[None, :]) / np.maximum(kc, 1e-300)
        slip[~near] = 0.0
        return dlr, slip, near

    def _island_advance(self, isl, T0, T1, count=True):
        """Advance a sleeping cluster from its committed state (the exact equilibrium of F(T0): positions in
        self.x[members], histories in self._hist[rows], or the candidate arrays x, hist of a pending dict) to the
        load at T1 (its limit from the left) and certify the step. Pure: nothing of self is written; the end state
        and the updated bookkeeping come back as a new island dict (x, hist the state to commit, refresh whether the
        tangents are to be recomputed at the commit; _island_commit). Returns (ok, new_isl, why), why one of the
        island_wake_* reasons when not ok.
        Certificate (tier 1, static): a lifted-cap Newton solve (statics.solve_equilibrium) to F(T1) and one to
        F(T_mid) give the exact end and mid-chord equilibria. Along the step the contact quantities deviate from the
        secant between the exact ends (whose ratio to a cap is quasi-convex, so its largest value is at an end) by
        at most D_k = rn_k (||dF_chord|| + 2 (||r_1|| + ||r_0|| + ||r_mid||) + ||f_dyn||) + 2 |lam_mid - (lam_0 +
        lam_1)/2|_k: the loads' own deviation from their chord (_island_chord), the Newton residuals at the end, at the
        start (the committed residual plus the long-range forces' change since the commit) and at mid-chord, the
        forces the drift's dynamics would need (_island_fdyn), all through the row norms rn of J K^-1 taken over the
        translational and the rotational degrees of freedom separately (_island_row_norms, Cauchy-Schwarz on each
        part), and the path's deviation from the secant measured at the mid-chord state and doubled (a quadratic path
        deviates most at mid-chord). The residuals count twice: once for the secant's ends, once inside the doubled
        mid-chord deviation, which is measured from three states each exact only to its own residual (2 |e_mid| +
        |e_0| + |e_1|; without the mid state's own, a mid solve accepted on its first iterate with a residual near the
        tolerance can hide a real deviation of the opposite sign). The rn terms are doubled while K, J are stale
        (n_since_refresh > 0). If every bounded sliding,
        rolling and twisting ratio at both ends is <= 1 + island_tol_rel, the real kernel takes no clip branch
        anywhere in the step: the plastic variables are exact and the end state is the elastic one. The pull-off
        ratio must keep the sleep margin, no contact may break, form (the inactive near pairs of the system) or
        yield, and the cumulative bound (below) must stay within island_budget_rel of the deflection at the cap.
        Creep (tier 2, with creep_islands, when only sds caps are exceeded): the real kernel's one-shot return map
        iterated to equilibrium with the released stiffness as tangent (_island_tangent) to F(T1), and the half-step
        pair T0 -> T_mid -> T1 (the second half from the first half's state), which is the state committed; the
        motion |du|_inf <= island_motion_m (rotations times the smallest radius), every plastic increment
        (statics.flow_increments, the one-step solve) <= island_flow_max of the deflection at the cap, no twist
        increment whose sign the kernel would take from a round-off rate (statics.twist_ambiguous; with cfg
        twist_continuity such a twist is set to sign(h_ref) cap/k_tw instead, ASSUMED and counted
        island_twist_continuity), the path error 2 |h_1step - h_2step| <= island_path_tol of the deflection at the
        cap (Richardson, first order), pull-off, yield, break, formation and the budget as above. In tier 2 the
        error terms that tier 1 certifies through D are budgeted instead, through the row norms rnR of J K_rel^-1,
        K_rel the released stiffness of the committed creep state (the tangent's own factor): at a capped spring a
        force f is taken up by slip in the released structure, (J_k K_rel^-1 f) / k_k (critique A1), and a change
        dF_ne of its normal force moves its cap by mu_k |dF_ne|, which the kernel clips. So an in-step load excursion
        beyond the chord flows it by up to (rnR_k + mu_k rnR_ne) ||dF_chord|| / k_k (curv_m; the cap-change part was
        missing and is the larger one in a 5-grain pile at its rolling limits), and the Newton residuals of the
        committed, half-step and start states shift it by up to rnR_k (||r_2|| + ||r_mid|| + ||r_0||) / k_k
        (resid_m); both summed over the creep steps. The forces the drift's dynamics would need (_island_fdyn) make
        the full model lag the quasi-static creep by up to rnR_k ||f_dyn|| / k_k at a capped spring (dyn_m); a lag is
        not summed but kept as its running maximum: while a spring creeps its slip is its relative displacement less
        its fixed deflection at the cap, so the lag of the positions is the whole of its slip error, and it keeps at
        most that value when it stops (this project's own argument, from the linearised released dynamics with the
        set of capped springs fixed: ASSUMED beyond that). Each charge is taken only at components that it can take
        to their caps (ratio + all three charges / cap >= 1; the others change elastically), doubled while K, J are
        stale.
        Bounds (bnd; the metre entries per kind sliding, rolling [m], twisting [rad]): newton_rel, dyn_rel, curv_rel
        (largest rn ||r|| / cap, rn ||f_dyn|| / cap, rn ||dF_chord|| / cap: the certificate's inputs, newton_rel and
        dyn_rel in tier 2 with rnR; reported as island_newton_rel and island_dyn_rel), curv_m, resid_m and dyn_m
        (tier 2), flow_tol_m (the tolerated excess over 1 times the deflection at the cap, summed; 0
        when the prediction is <= 1), path_m (summed), flow_m (the plastic increments, summed: a record, not an
        error), snap_rel and snap_m (statics.snap_to_caps on the committed histories: the largest relative force
        change, and its deflection summed), ratchet_m (the entry residual's slip bound, _try_sleep_island), total_rel
        = max over kinds of (flow_tol + ratchet + path + curv + resid + dyn + snap) / the smallest deflection at the
        cap, the budgeted quantity. s_isl, the most any member is taken to move inside the step from its committed
        position (the motion bounds of every other grain), is max_i (|du_1|_i + 2 |du_mid - du_1/2|_i) (the path's
        end and its deviation from the secant measured at mid-step and doubled, as in D: a load that goes out and
        back within the step moves the members although du_1 = 0) + ru ||dF_chord|| + 1e-12, ru the largest row norm
        of K^-1 over a member's translational degrees of freedom (_island_disp_norm; K_rel's in tier 2; doubled while
        stale): a member at the top of a stack moves by the sum of the deflections down its chain, which the largest
        single contact deflection, used before, missed by 2-5x. Counters when count (the entry certificate passes
        False): island_static_steps, island_creep_steps, island_newton_iters, island_refactor."""
        mem, ci, cj, prm, prm_el, L = isl["members"], isl["ci"], isl["cj"], isl["prm"], isl["prm_el"], self.L
        x = np.ascontiguousarray(isl["x"] if "x" in isl else self.x[mem])
        hist = np.ascontiguousarray(isl["hist"] if "hist" in isl else self._hist[isl["rows"]])
        lu = self._island_lu(isl)
        lam0, out0, rn, kc, capv = isl["lam"], isl["out"], isl["rn"], isl["kc"], isl["capv"]
        tm = hist[:, dem.H_TOUCH] > 0.5
        its = [0]

        def ret(ok, new, why):
            if count:
                self._count("island_newton_iters", its[0])
                if ok:
                    self._count("island_creep_steps" if new["tier"] == 2 else "island_static_steps")
            return ok, new, why

        def solve(xs, hs, Fext, dus, real, tangent=None):
            E = statics.solve_equilibrium(xs, ci, cj, prm if real else prm_el, prm_el, hs, L, lu, Fext, dus, tt, tr,
                                          self.island_newton_max if real else 6, tangent=tangent)
            its[0] += E["n_it"]
            return E, self._island_flags(E)
        if not tm.any():
            return ret(False, None, "break")
        l = T1 - T0
        tt, tr = self._island_tols(capv, prm, tm)
        xi_cap = capv / np.maximum(kc, 1e-300)
        xi_min = xi_cap[tm].min(0)
        F1 = self._island_load(mem, T1, "-")
        Fm = self._island_load(mem, T0 + 0.5 * l, "-")
        dFlr = np.zeros((len(mem), 6))
        dFlr[:, :3] = self._F_lr[mem] - isl["F_lr_ref"]
        r0 = isl["r_ref"] + dFlr.reshape(-1)       # the committed state's residual against this step's start load
        du0 = lu.solve(F1 - isl["F_ref"])
        E1, why = solve(x, hist, F1, du0, False)
        if why:
            return ret(False, None, why)
        Em, why = solve(x, hist, Fm, 0.5 * du0, False)
        if why:
            return ret(False, None, why)
        lam1, out1 = E1["lam"], E1["out"]
        chord = self._island_chord(mem, T0, T1)
        ft, frot = self._island_fdyn(isl, hist, E1["du"], l, T0)

        def parts(r):
            r6 = r.reshape(-1, 6)
            return float(np.linalg.norm(r6[:, :3])), float(np.linalg.norm(r6[:, 3:]))
        (r1t, r1r), (r0t, r0r), (rmt, rmr) = parts(E1["r"]), parts(r0), parts(Em["r"])
        rres_t, rres_r = 2 * (r1t + r0t + rmt), 2 * (r1r + r0r + rmr)
        fac = 2.0 if isl["n_since_refresh"] > 0 else 1.0
        D = (fac * (rn[:, 0] * (chord + rres_t + ft) + rn[:, 1] * (rres_r + frot))
             + 2 * np.abs(Em["lam"] - 0.5 * (lam0 + lam1)))
        p0, s0 = self._ratios(lam0, D, prm, out0, tm)
        p1, s1 = self._ratios(lam1, D, prm, out1, tm)
        sds = np.maximum(s0, s1)
        b = self._island_bnd0()                   # (an island saved before an entry existed gets it at zero)
        b.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in isl["bnd"].items()})
        rn3t, rn3r = self._kind_norms(rn[:, 0])[tm], self._kind_norms(rn[:, 1])[tm]
        cap_t = np.maximum(capv[tm], 1e-300)
        kc_t = np.maximum(kc[tm], 1e-300)
        b["curv_rel"] = max(b["curv_rel"], float((rn3t * chord / cap_t).max()))
        if float(max(p0.max(), p1.max())) > 1 - self.margin:
            return ret(False, None, "pulloff")
        n_it = max(E1["n_it"], Em["n_it"])
        if (sds <= 1 + self.island_tol_rel).all():
            tier = 1
            xn, hn, lam_n, out_n, r_n = E1["xn"], E1["hn"], lam1, out1, E1["r"]
            du_tot, du_mid = E1["du"], Em["du"]
            b["flow_tol_m"] = b["flow_tol_m"] + (np.maximum(sds - 1.0, 0.0) * xi_cap[tm]).max(0)
            b["newton_rel"] = max(b["newton_rel"], float(((rn3t * rres_t + rn3r * rres_r) / cap_t).max()))
            b["dyn_rel"] = max(b["dyn_rel"], float(((rn3t * ft + rn3r * frot) / cap_t).max()))
            ru = isl.get("ru")
            if ru is None:                        # an island committed before ru existed
                ru = self._island_disp_norm(lu, len(mem))
            u_chord = fac * ru * chord
        else:
            if not self.creep_islands:
                return ret(False, None, "loading")
            tier = 2
            tangent = self._island_tangent(isl, count)
            C1, why = solve(x, hist, F1, E1["du"], True, tangent)
            if why:
                return ret(False, None, why)
            Cm, why = solve(x, hist, Fm, Em["du"], True, tangent)
            if why:
                return ret(False, None, why)
            C2, why = solve(Cm["xn"], Cm["hn"], F1, C1["du"] - Cm["du"], True, tangent)
            if why:
                return ret(False, None, why)
            n_it = max(n_it, C1["n_it"], Cm["n_it"], C2["n_it"])
            du_tot, du_mid = Cm["du"] + C2["du"], Cm["du"]
            dU = np.abs(du_tot.reshape(-1, 6))
            if max(float(dU[:, :3].max()), float(dU[:, 3:].max()) * float(self.rad[mem].min())) > self.island_motion_m:
                return ret(False, None, "motion")
            hn_el = statics.apply_displacement(x, ci, cj, prm_el, hist, L, C1["du"].reshape(-1, 6))[1]
            dp = statics.flow_increments(hn_el, C1["hn"])
            if (dp[tm] > self.island_flow_max * xi_cap[tm]).any():
                return ret(False, None, "flow")
            xn, hn, lam_n, out_n, r_n = C2["xn"], C2["hn"], C2["lam"], C2["out"], C2["r"]
            amb = statics.twist_ambiguous(hist, hn_el, prm, statics.caps(C1["lam"], prm, C1["out"])) & tm
            if amb.any():
                if not self.twist_continuity:
                    return ret(False, None, "twist")
                hn = hn.copy()
                hn[amb, dem.H_XTW] = np.sign(hist[amb, dem.H_XTW]) * statics.caps(lam_n, prm, out_n)[amb, 2] / kc[amb, 2]
                lam_n, Fnet_n, out_n = statics.state(xn, ci, cj, prm_el, hn, L)
                r_n = F1 + Fnet_n
                if count:
                    self._count("island_twist_continuity", int(amb.sum()))
            d = C1["hn"] - hn
            e = 2 * np.column_stack([np.linalg.norm(d[:, dem.H_XT:dem.H_XT + 3], axis=1),
                                     np.linalg.norm(d[:, dem.H_XR:dem.H_XR + 3], axis=1), np.abs(d[:, dem.H_XTW])])
            if (e[tm] > self.island_path_tol * xi_cap[tm]).any():
                return ret(False, None, "path")
            if float(self._ratios(lam_n, D, prm, out_n, tm)[0].max()) > 1 - self.margin:
                return ret(False, None, "pulloff")
            b["path_m"] = b["path_m"] + e[tm].max(0)
            b["flow_m"] = b["flow_m"] + dp[tm].max(0)
            # the released structure of the committed state: the excursion beyond the chord (own kind and the cap's
            # change with the normal force), the residuals and the drift's dynamics (see above)
            facR = tangent(C2["xn"], C2["hn"], C2["lam"], C2["out"])
            rnR = self._island_row_norms(facR, isl["J"])
            rR3t, rR3r = self._kind_norms(rnR[:, 0])[tm], self._kind_norms(rnR[:, 1])[tm]
            rRn = rnR[0::8, 0][tm][:, None]
            mu = prm[tm][:, [dem.P_MUT, dem.P_MUR, dem.P_MUTW]]
            ft2, frot2 = self._island_fdyn(isl, hist, du_tot, l, T0)
            (r2t, r2r), (rct, rcr) = parts(C2["r"]), parts(Cm["r"])
            f_curv = fac * (rR3t + mu * rRn) * chord
            f_res = fac * (rR3t * (r2t + rct + r0t) + rR3r * (r2r + rcr + r0r))
            f_dyn = fac * (rR3t * ft2 + rR3r * frot2)
            # only a component that the excursion, the residuals and the lag together can take to its cap can flow
            # (the others change elastically, which leaves no permanent error)
            reach = self._ratios(lam_n, None, prm, out_n, tm)[1] + (f_curv + f_res + f_dyn) / cap_t >= 1.0
            def worst(f):
                return np.where(reach, f / kc_t, 0.0).max(0)
            b["curv_m"] = b["curv_m"] + worst(f_curv)
            b["resid_m"] = b["resid_m"] + worst(f_res)
            b["dyn_m"] = np.maximum(b["dyn_m"], worst(f_dyn))
            b["newton_rel"] = max(b["newton_rel"], float((f_res / cap_t).max()))
            b["dyn_rel"] = max(b["dyn_rel"], float((f_dyn / cap_t).max()))
            u_chord = fac * self._island_disp_norm(facR, len(mem)) * chord
        hs, srel = statics.snap_to_caps(hn, lam_n, prm, out_n)
        if srel > 0.0:
            hn = hs
            lam_n, Fnet_n, out_n = statics.state(xn, ci, cj, prm_el, hn, L)
            r_n = F1 + Fnet_n
            b["snap_rel"] = max(b["snap_rel"], srel)
            b["snap_m"] = b["snap_m"] + srel * xi_cap[tm].max(0)
        u1, um = du_tot.reshape(-1, 6)[:, :3], du_mid.reshape(-1, 6)[:, :3]
        du_inf = float(np.abs(u1).max())
        s_isl = (float((np.linalg.norm(u1, axis=1) + 2 * np.linalg.norm(um - 0.5 * u1, axis=1)).max()) + u_chord
                 + 1e-12)
        if (~tm).any() and (-statics.overlaps(xn, ci[~tm], cj[~tm], prm[~tm], L) <= s_isl).any():
            return ret(False, None, "newcontact")
        tot = ((b["flow_tol_m"] + b["ratchet_m"] + b["path_m"] + b["curv_m"] + b["resid_m"] + b["dyn_m"] + b["snap_m"])
               / np.maximum(xi_min, 1e-300))
        b["total_rel"] = float(tot.max())
        if b["total_rel"] > self.island_budget_rel:
            return ret(False, None, "budget")
        refresh = isl["n_since_refresh"] + 1 >= 1000 or isl["du_since_refresh"] + du_inf > 1e-11 or n_it > 3
        new = dict(isl)
        new.update(x=xn, hist=hn, lam=lam_n, out=out_n, F_ref=F1, T_ref=T1, F_lr_ref=self._F_lr[mem].copy(),
                   r_ref=r_n, kc=statics.kcomp(prm, hn), capv=statics.caps(lam_n, prm, out_n), tier=tier,
                   du_max=du_inf, s_isl=s_isl, bnd=b, refresh=refresh)
        return ret(True, new, None)

    def _island_refresh(self, isl):
        """Recompute an island's contact system at its committed state (the touching contacts plus the inactive near
        pairs, _island_near) and there its lifted tangents K, J, their factor, the row norms, the damping matrix
        (_island_fdyn), the contact state, spring constants, caps and the residual against its reference load (in
        place: the dict is a new one, not yet
        shared with any snapshot)."""
        from scipy.sparse.linalg import splu
        mem, rows = isl["members"], isl["rows"]
        kij = self._kij[rows][self._hist[rows, dem.H_TOUCH] > 0.5]
        ci, cj, rows = self._island_system(mem, [(int(a), int(c)) for a, c in kij.tolist()])
        ci, cj, rows = self._island_near(mem, ci, cj, rows)
        x = np.ascontiguousarray(self.x[mem])
        prm = np.ascontiguousarray(self._prm[rows])
        hist = np.ascontiguousarray(self._hist[rows])
        prm_el = statics.lift_caps(prm)
        K, J = statics.jacobians(x, self.rad[mem], ci, cj, prm_el, hist, self.L)
        Cd = statics.jacobians(x, self.rad[mem], ci, cj, prm_el, hist, self.L, vel=True)[0]
        lu = splu(K)
        lam, Fnet, out = statics.state(x, ci, cj, prm_el, hist, self.L)
        isl.update(ci=ci, cj=cj, rows=rows, prm=prm, prm_el=prm_el, K=K, J=J, Cd=Cd, lu=lu,
                   rn=self._island_row_norms(lu, J),
                   ru=self._island_disp_norm(lu, len(mem)), lam=lam, out=out, kc=statics.kcomp(prm, hist),
                   capv=statics.caps(lam, prm, out), r_ref=isl["F_ref"] + Fnet)

    def _island_commit(self, iid, new):
        """Make a pending island state (_island_advance) the committed one: positions and histories written to
        self.x / self._hist, the dict replaced (never mutated: a redo's snapshot shares the old one); when the advance
        flagged a refresh (1000 steps, 1e-11 m of drift or more than 3 Newton iterations since the last), the lifted
        tangents, factor, row norms and the contact system's near pairs are recomputed at the committed state
        (_island_refresh, counted island_refactor)."""
        mem, rows = new["members"], new["rows"]
        self.x[mem] = new.pop("x")
        self._hist[rows] = new.pop("hist")
        if new.pop("refresh", False):
            self._island_refresh(new)
            self._count("island_refactor")
            new["n_since_refresh"], new["du_since_refresh"] = 0, 0.0
        else:
            new["n_since_refresh"] += 1
            new["du_since_refresh"] += new["du_max"]
        new["n_steps"] += 1
        self._islands[iid] = new

    def _island_upgrade(self, isl, T):
        """An island dict of the previous code version (no tier: its committed state is where one linear placement
        left it when it fell asleep, at T_ref, and the load at T may differ from F_ref) in this version's format, at
        T. Placed first as the previous version would wake it at T (the linear response to F(T) - F_ref through its
        own tangent, applied with the real kernel, as _wake_island's tier-0 placement), then solved from there to the
        exact equilibrium of F(T) (the lifted-cap Newton solve; the return map with the released tangent if a cap is
        exceeded, with creep_islands, as at entry), snapped to the caps, with the inactive near pairs in its system
        and the lifted tangents, row norms, spring constants and caps at that state (_island_refresh), F_ref = F(T),
        T_ref = T. The first step's certificate and a wake inside it then start from the exact equilibrium of the
        load now: keeping the old F_ref and T_ref made a wake in the first step hand the DEM the equilibrium of a load
        left long ago (1.7e-12 m from the equilibrium of the load now after a 1 g change; the previous version's own
        placement was 1.3e-16 m from it). Bounds zero but for the ratchet bound of the residual the previous version
        carried, the solve's displacement du from the linear placement, at rest: E_res = du^T K du / 2, its slip bound
        in the released system as at entry (infinite, so that the budget wakes the island, if that system is a
        mechanism). If the linear placement breaks or forms a contact, the solve flags anything (_island_flags), or a
        cap is exceeded with creep_islands off, the island stays as it was, tier 0 (the previous version's linear
        certificate and placement)."""
        mem, rows, L = isl["members"], isl["rows"], self.L
        F = self._island_load(mem, T)
        x0, h0 = np.ascontiguousarray(self.x[mem]), np.ascontiguousarray(self._hist[rows])
        du = self._island_lu(isl).solve(F - isl["F_ref"]).reshape(-1, 6)
        xl, hl = statics.apply_displacement(x0, isl["ci"], isl["cj"], isl["prm"], h0, L, du)
        if ((hl[:, dem.H_TOUCH] > 0.5) != (h0[:, dem.H_TOUCH] > 0.5)).any():
            return dict(isl, tier=0)
        self.x[mem], self._hist[rows] = xl, hl
        new = dict(isl)
        new.update(tier=1, n_steps=0, n_since_refresh=0, du_since_refresh=0.0, du_max=0.0, s_isl=1e-12, F_ref=F,
                   T_ref=T, F_lr_ref=self._F_lr[mem].copy(), bnd=self._island_bnd0())
        self._island_refresh(new)                 # the system at the linear placement, r_ref its residual at F(T)
        ci, cj, prm, prm_el, lu, K = new["ci"], new["cj"], new["prm"], new["prm_el"], new["lu"], new["K"]
        x, hist = np.ascontiguousarray(self.x[mem]), np.ascontiguousarray(self._hist[new["rows"]])
        tm = hist[:, dem.H_TOUCH] > 0.5
        tt, tr = self._island_tols(new["capv"], prm, tm)
        C = statics.solve_equilibrium(x, ci, cj, prm_el, prm_el, hist, L, lu, F, lu.solve(new["r_ref"]), tt, tr, 6)
        why = self._island_flags(C)
        if why is None and (self._ratios(C["lam"], None, prm, C["out"], tm)[1] > 1 + self.island_tol_rel).any():
            why = None if self.creep_islands else "loading"
            if why is None:
                new["tier"] = 2
                tan = self._island_tangent(dict(K=K, J=new["J"], prm=prm, lu=lu), False)
                C = statics.solve_equilibrium(x, ci, cj, prm, prm_el, hist, L, lu, F, C["du"], tt, tr,
                                              self.island_newton_max, tangent=tan)
                why = self._island_flags(C)
        if why is not None:
            self.x[mem], self._hist[rows] = x0, h0
            return dict(isl, tier=0)
        E_res = max(0.5 * float(C["du"] @ (K @ C["du"])), 0.0)
        self.x[mem] = C["xn"]
        self._hist[new["rows"]] = statics.snap_to_caps(C["hn"], C["lam"], prm, C["out"])[0]
        self._island_refresh(new)                 # tangents, caps and residual at the placed state
        hist = self._hist[new["rows"]]
        rb = self._island_ratchet(new["K"], new["J"], prm, hist, new["lam"], new["out"], new["kc"], new["capv"], E_res,
                                  mem)
        new["bnd"]["ratchet_m"] = np.full(3, np.inf) if rb is None else rb[1].max(0)
        return new

    def _island_report(self):
        """The sleeping clusters' bounds into n_events (running maxima): island_bound_xi, the largest cumulative bound
        relative to the deflection at the cap (what the budget limits), island_flow_m (the plastic increments of
        the creep steps, metres), island_snap_rel, and the per-step inputs island_newton_rel and island_dyn_rel (the
        largest Newton-residual and drift-dynamics terms relative to the cap; certified in tier 1, budgeted in tier
        2 as resid_m)."""
        ev = self.n_events
        for isl in self._islands.values():
            b = isl.get("bnd")
            if b is None:
                continue
            ev["island_bound_xi"] = max(ev.get("island_bound_xi", 0.0), float(b["total_rel"]))
            ev["island_flow_m"] = max(ev.get("island_flow_m", 0.0), float(b["flow_m"][:2].max()))
            ev["island_snap_rel"] = max(ev.get("island_snap_rel", 0.0), float(b["snap_rel"]))
            ev["island_newton_rel"] = max(ev.get("island_newton_rel", 0.0), float(b["newton_rel"]))
            ev["island_dyn_rel"] = max(ev.get("island_dyn_rel", 0.0), float(b["dyn_rel"]))

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
        """Static grains (asleep) near the cluster: centres and radii, for the pre-run's approach checks; a sleeping
        cluster may drift by up to s_isl inside the step (_step_once), so the radii carry the largest allowance."""
        sl = np.flatnonzero(self.regime == ASLEEP)
        if not len(sl):
            return np.zeros((0, 3)), np.zeros(0)
        s_max = self._drift()[1]
        d = np.linalg.norm(self.x[sl][:, None, :] - self.x[mem][None, :, :], axis=2).min(1)
        near = sl[d <= 2 * float(self.rad.max()) + self.d_cap + reach + s_max]
        return np.ascontiguousarray(self.x[near]), np.ascontiguousarray(self.rad[near] + s_max)

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
        """Sleep an anchored cluster, at its sliding, rolling and twisting caps as well, on candidate arrays (self.x,
        self._hist, self.v, self.w are written only on acceptance). (E0) No fast load or breakpoint in the next step;
        the island system = the touching contacts plus the near pairs (_island_near); the lifted stiffness factorises.
        (E1) The exact equilibrium of F(T) from the DEM state by the lifted-cap Newton solve (per-DOF tolerances,
        _island_tols): if every sliding, rolling and twisting ratio there is <= 1 + island_tol_rel it is the placed
        state (tier 1); else, with creep_islands, the real kernel's return map with the consistent tangent places it
        (tier 2; the DEM state was flowing); a mechanism or no convergence refuses. (E2) Pull-off <= 1 - margin now
        and at T + H (limit from the left, linear response) with the residual's released ellipsoid dl_rel included;
        the solve broke, formed and yielded nothing. (E3) The residual E_res = E_kv + du^T K du / 2 (E_kv the kinetic
        energy of the motion relative to the drift of the equilibrium, _island_drift, which is the kinetic energy
        itself under a constant load; the lifted K over-estimates the stored energy: an upper bound) may slip no
        component near its cap by more than island_ratchet_rel of its deflection at the cap (_island_ratchet, the
        released system; a mechanism there refuses); and, as before, it may perturb no contact's load by more than 1%
        of its remaining margin. When no component is near a cap and that fails, the cluster sleeps with the residual
        tracked as exact linear response (_linres, tier 0). When one is, the residual is dropped (the components near
        their caps are bounded by the slip test above; for them the 1% test cannot hold, since 1 - ratio is about 0
        at a cap), so every contact with no component near its cap must pass the 1% test, else the cluster stays
        awake (cluster_awake_ratchet): what is dropped then perturbs those contacts by at most 1% of their margins,
        as before, and slips the others by at most the bound reported in ratchet_m. (E4) The first
        certificate, _island_advance on the candidate over the next step, must pass (counted, not placed, otherwise);
        and a tier-1 candidate must keep every sds ratio <= 1 + island_tol_rel and the pull-off ratio <= 1 - margin
        on the linear extrapolation over island_hyst_steps steps (to the next breakpoint at most), else it is a tier-2
        candidate (refused with creep_islands off: a sleep that would not hold is not worth the placement); and no
        other sleeper may lie within reach of it in that step (_sleepers_near with the first step's s_isl; refused
        as cluster_awake_neighbour, since _sleeper_contacts would wake both at once). Then
        placed: snapped to (1 - 2^-40) of the caps, at rest, bounds zero but for the ratchet bound. Counters:
        cluster_sleep, cluster_sleep_cap (a component at or within the margin of its cap), cluster_sleep_creep,
        cluster_awake_{mechanism, newton, over_limit, ratchet, certificate, drift, transition, neighbour} and
        _linres's. With
        cfg contact.cap_sleep off the previous entry test runs unchanged (_try_sleep_island_linear)."""
        if H <= 0 or self.loads.fast_window(T, T + H) or self.loads.breakpoint_in(T - 1e-9, T + H):
            return False
        if not self.cap_sleep:
            return self._try_sleep_island_linear(mem, touch, T, H)
        from scipy.sparse.linalg import splu
        L = self.L
        ci, cj, rows = self._island_system(mem, touch)
        if not len(rows):
            return False
        ci, cj, rows = self._island_near(mem, ci, cj, rows)
        x = np.ascontiguousarray(self.x[mem])
        prm = np.ascontiguousarray(self._prm[rows])
        hist = np.ascontiguousarray(self._hist[rows])
        prm_el = statics.lift_caps(prm)
        tm = hist[:, dem.H_TOUCH] > 0.5
        K, J = statics.jacobians(x, self.rad[mem], ci, cj, prm_el, hist, L)
        try:
            lu = splu(K)
        except RuntimeError:
            return self._awake("mechanism")
        lam0, Fnet0, out0 = statics.state(x, ci, cj, prm_el, hist, L)
        tt, tr = self._island_tols(statics.caps(lam0, prm, out0), prm, tm)
        Fext = self._island_load(mem, T)
        C = statics.solve_equilibrium(x, ci, cj, prm_el, prm_el, hist, L, lu, Fext, lu.solve(Fext + Fnet0), tt, tr, 6)
        why = self._island_flags(C)
        if why:
            return self._awake(why if why in ("mechanism", "newton") else "transition")
        tier = 1
        if (self._ratios(C["lam"], None, prm, C["out"], tm)[1] > 1 + self.island_tol_rel).any():
            if not self.creep_islands:
                return self._awake("over_limit")
            tier = 2
            C = statics.solve_equilibrium(x, ci, cj, prm, prm_el, hist, L, lu, Fext, C["du"], tt, tr,
                                          self.island_newton_max,
                                          tangent=self._island_tangent(dict(K=K, J=J, prm=prm, lu=lu), False))
            why = self._island_flags(C)
            if why:
                return self._awake(why if why in ("mechanism", "newton") else "transition")
        du, xc, hc, lam_c, out_c = C["du"], C["xn"], C["hn"], C["lam"], C["out"]
        capv, kc = statics.caps(lam_c, prm, out_c), statics.kcomp(prm, hc)
        xi_cap = capv / np.maximum(kc, 1e-300)
        # (E3) the residual's energy and its released-system slip bound; (E2) pull-off with it, now and at the end
        dF = self._island_load(mem, T + H, "-") - Fext
        capd = np.zeros((len(prm), 3), bool)
        capd[tm] = self._ratios(lam_c, None, prm, out_c, tm)[1] >= 1 - self.margin
        E_kv = self._island_drift(K, J, kc, capd, self._fhat(lam_c), mem, dF,
                                  self._island_load(mem, T + 0.5 * H) - Fext, H)
        E_res = max(E_kv + 0.5 * float(du @ (K @ du)), 0.0)
        rb = self._island_ratchet(K, J, prm, hc, lam_c, out_c, kc, capv, E_res, mem)
        if rb is None:
            return self._awake("mechanism")
        dl_rel, slip, near = rb
        lam_end = lam_c + J @ lu.solve(dF)
        if max(float(self._ratios(lam_c, dl_rel, prm, out_c, tm)[0].max()),
               float(self._ratios(lam_end, dl_rel, prm, out_c, tm)[0].max())) > 1 - self.margin:
            return self._awake("over_limit")
        if (slip[near] > self.island_ratchet_rel * xi_cap[near]).any():
            return self._awake("ratchet")
        res = None
        pt, ot = prm[tm], out_c[tm]

        def sub(v):
            return v.reshape(len(prm), 8)[tm].reshape(-1)
        rr = np.maximum(statics.margins(sub(lam_c), pt, ot), statics.margins(sub(lam_end), pt, ot))
        worst = np.maximum(statics.margins_bounded(sub(lam_c), sub(dl_rel), pt, ot),
                           statics.margins_bounded(sub(lam_end), sub(dl_rel), pt, ot))
        far = ~near[tm].any(1)                    # contacts with no component near its cap
        bad = ~np.isfinite(worst) | ((worst - rr) > 0.01 * (1 - rr))
        if bad[far].any() or (not near.any() and bad.any()):
            if near.any():
                return self._awake("ratchet")     # the residual would be dropped: it must keep the 1% test
            with np.errstate(divide="ignore", invalid="ignore"):       # the near pairs' 0/0 pull-off ratios (-> 0)
                res = self._linres(mem, x, ci, cj, prm, hist, K, J, du, lam_c, lam_end, out_c, T)
            if res is None:
                return False                      # (_linres counts why)
        # (E4) the candidate: snapped, its tangents at the placed state, the first certificate, the hysteresis
        hc, srel = statics.snap_to_caps(hc, lam_c, prm, out_c)
        r_c = C["r"]
        if srel > 0.0:
            lam_c, Fnet_c, out_c = statics.state(xc, ci, cj, prm_el, hc, L)
            r_c = Fext + Fnet_c
        K2, J2 = statics.jacobians(xc, self.rad[mem], ci, cj, prm_el, hc, L)
        try:
            lu2 = splu(K2)
        except RuntimeError:
            return self._awake("mechanism")
        bnd = self._island_bnd0()
        bnd["ratchet_m"] = slip.max(0)
        Cd = statics.jacobians(xc, self.rad[mem], ci, cj, prm_el, hc, L, vel=True)[0]
        cand = dict(members=mem, ci=ci, cj=cj, rows=rows, prm=prm, prm_el=prm_el, K=K2, J=J2, Cd=Cd, lu=lu2, lam=lam_c,
                    out=out_c, F_ref=Fext, T_ref=T, F_lr_ref=self._F_lr[mem].copy(), r_ref=r_c,
                    rn=self._island_row_norms(lu2, J2), ru=self._island_disp_norm(lu2, len(mem)),
                    kc=statics.kcomp(prm, hc), capv=statics.caps(lam_c, prm, out_c),
                    tier=tier, n_steps=0, n_since_refresh=0, du_since_refresh=0.0, du_max=0.0, s_isl=1e-12, bnd=bnd,
                    x=xc, hist=hc)
        if res is not None:
            cand.update(res=res, dl=res["dl"], tier=0)
        else:
            ok, first, why = self._island_advance(cand, T, T + H, count=False)
            if not ok:
                return self._awake("certificate")
            if self._sleepers_near(mem, xc, first["s_isl"]):
                return self._awake("neighbour")
            if tier == 1:
                Tx = min(T + self.island_hyst_steps * H, self.loads.next_breakpoint(T))
                lam_x = lam_c + J2 @ lu2.solve(self._island_load(mem, Tx, "-") - Fext)
                px, sx = self._ratios(lam_x, None, prm, out_c, tm)
                if (sx > 1 + self.island_tol_rel).any() or float(px.max()) > 1 - self.margin:
                    if not self.creep_islands:
                        return self._awake("drift")
                    tier = cand["tier"] = 2
        self.x[mem] = cand.pop("x")
        self._hist[rows] = cand.pop("hist")
        self.v[mem] = 0.0
        self.w[mem] = 0.0
        iid = self._next_island
        self._next_island += 1
        self._islands[iid] = cand
        for k in mem:
            self._island_of[int(k)] = iid
        self.regime[mem] = ASLEEP
        self.n_events["sleep"] += len(mem)
        self._count("cluster_sleep")
        if (self._ratios(lam_c, None, prm, out_c, tm)[1] >= 1 - self.margin).any():
            self._count("cluster_sleep_cap")
        if tier == 2:
            self._count("cluster_sleep_creep")
        return True

    def _try_sleep_island_linear(self, mem, touch, T, H):
        """The entry test of v0.6 (cfg contact.cap_sleep off): sleep an anchored cluster if at its exact static
        equilibrium every contact keeps 5x the sleep margin to its limits for the loads over the next step, and its
        residual motion (kinetic energy + elastic energy of its deviation from equilibrium) could add at most 1% of
        the smallest margin to any contact force (else tracked exactly, _linres). Islands so made carry tier 0: the
        linear certificate (_island_ratio) and the linear-response placement at wake."""
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
                                  lu=splu(K), F_ref=self._island_load(mem, T), T_ref=T, tier=0)
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
        """Wake a sleeping cluster into the DEM at rest, at the exact static equilibrium of the load at T. A tier-0
        island (the linear certificate: cap_sleep off, a tracked residual motion, or an island of the previous code
        version): the linear response from its reference state to the load at T, at a load breakpoint the load just
        before it (_jump_side: the DEM resolves the jump, as for the other tiers), plus the residual motion evolved
        exactly since it fell asleep (as before). Otherwise, at the step start (T = T_ref) the committed state
        itself, which at a load breakpoint is the pre-jump equilibrium (the DEM resolves the jump); mid-step (a DEM
        grain approaching, _watch) the lifted Newton solve from the committed state to F(T) (tier 2: the return-map
        solve when a cap is exceeded), accepted only if it converges, breaks, forms and yields nothing and moves no
        grain by more than this step's s_isl (the allowance every bound of the step used for the island, _step_once;
        the committed dict's is the previous step's); otherwise the committed state itself is handed over
        at rest and island_wake_fallback counted: a history in which the kernel recorded a break is never committed
        (its time stamp would be wrong; the DEM resolves the break itself). island_wake_transient_m, the DEM's own
        measure of the woken members' motion over the rest of the step, is recorded by step()."""
        isl = self._islands.pop(iid)
        mem = isl["members"]
        for k in mem:
            self._island_of.pop(int(k), None)
        self.regime[mem] = ACTIVE
        self.n_events["wake"] += len(mem)
        if hasattr(self, "_woken_mem"):
            self._woken_mem.append(np.asarray(mem, np.int64))
        if isl.get("tier", 0) == 0 or "res" in isl:
            F = self._island_load(mem, T, self._jump_side(T, float(isl["T_ref"])))
            du = self._island_lu(isl).solve(F - isl["F_ref"]).reshape(-1, 6)
            vr = np.zeros((len(mem), 6))
            if "res" in isl:                      # the residual motion, evolved exactly since the cluster fell asleep
                r = isl["res"]
                ur, vrf = statics.residual_state(r["lam"], r["V"], r["c"], T - r["T"])
                du = du + ur.reshape(-1, 6)
                vr = vrf.reshape(-1, 6)
            x = np.ascontiguousarray(self.x[mem])
            hist = np.ascontiguousarray(self._hist[isl["rows"]])
            xn, hn = statics.apply_displacement(x, isl["ci"], isl["cj"], isl["prm"], hist, self.L, du)
            if ((hn[:, dem.H_TOUCH] > 0.5) != (hist[:, dem.H_TOUCH] > 0.5)).any():
                self._count("island_wake_fallback")   # a break or formation in the placement: the DEM resolves it
                self.v[mem] = 0.0
                self.w[mem] = 0.0
                return
            self.x[mem] = xn
            self._hist[isl["rows"]] = hn
            self.v[mem] = vr[:, :3]
            self.w[mem] = vr[:, 3:]
            return
        self.v[mem] = 0.0
        self.w[mem] = 0.0
        if abs(T - isl["T_ref"]) <= 1e-12:
            return                                # the committed state: snapped, at rest
        ci, cj, rows, prm, prm_el, L = isl["ci"], isl["cj"], isl["rows"], isl["prm"], isl["prm_el"], self.L
        x = np.ascontiguousarray(self.x[mem])
        hist = np.ascontiguousarray(self._hist[rows])
        tm = hist[:, dem.H_TOUCH] > 0.5
        lu = self._island_lu(isl)
        tt, tr = self._island_tols(isl["capv"], prm, tm)
        F = self._island_load(mem, T)
        E = statics.solve_equilibrium(x, ci, cj, prm_el, prm_el, hist, L, lu, F, lu.solve(F - isl["F_ref"]), tt, tr, 6)
        ok = self._island_flags(E) is None
        if ok and isl["tier"] == 2 and (self._ratios(E["lam"], None, prm, E["out"], tm)[1] > 1 + self.island_tol_rel).any():
            E = statics.solve_equilibrium(x, ci, cj, prm, prm_el, hist, L, lu, F, E["du"], tt, tr,
                                          self.island_newton_max, tangent=self._island_tangent(isl, False))
            ok = self._island_flags(E) is None
        s_now = float(self._drift()[0][mem].max())    # this step's allowance (_step_once), the one the bounds used
        if ok and float(np.abs(E["xn"] - x).max()) <= (s_now if s_now > 0 else isl["s_isl"]):
            self.x[mem] = E["xn"]
            self._hist[rows] = statics.snap_to_caps(E["hn"], E["lam"], prm, E["out"])[0]
        else:
            self._count("island_wake_fallback")

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


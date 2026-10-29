"""Multi-particle microgravity dust simulation for one experiment chamber.

Method (hybrid event-driven / DEM):
* Grains ("monomers") belong to rigid bodies. A body is a single grain or an
  aggregate of grains that have stuck together. Bodies translate and rotate;
  aggregates carry a principal-axis orientation and are rotated with the
  symplectic, energy- and angular-momentum-conserving splitting of
  Dullweber, Leimkuhler & McLachlan (1997).
* Each global step (~1-4 ms) is a velocity-Verlet (kick-drift-kick) step.
  The kicks integrate gas drag and Coriolis force exactly (a damped rotation
  of the velocity) together with the other forces held constant over the
  half-step: apparent (residual) acceleration, g-jitter, centrifugal force,
  gravity gradient, mutual gravity, Coulomb (short-range pairs + long-range
  space-charge mesh, electrostatics.py), charge-induced polarisation, image
  charges, external E field, Lorentz force, thermophoresis, photophoresis,
  radiation pressure and Brownian kicks. Position-dependent forces are
  re-evaluated at the end of every step (velocity Verlet).
* During the drift, continuous collision detection finds the exact time of
  every grain-grain and grain-wall contact. Impacts are processed in time
  order; after each one the changed bodies are re-predicted, so secondary
  collisions are never skipped. Each impact is resolved by the JKR + plastic
  contact integrator (contact.py) with the effective mass of the two rigid
  bodies at the contact point. Stick -> bodies merge rigidly (or freeze on
  the wall). Bounce -> impulse from the full 3x3 collision matrix with the
  DEM normal restitution and a Coulomb friction cone.
* Every grain and aggregate spins: spin is set at launch, exchanged with
  translation by friction and off-centre impacts, conserved through sticking,
  and damped by gas torque.
* Contact (tribo-)charging with saturation, grain-to-grain surface-energy
  scatter, anti-stick wall coatings and vibrating walls (impact phase drawn
  from the flux-weighted distribution, so the wall motion never lets a grain
  through), and Dominik & Tielens (1997) erosion / shattering of aggregates.

* Spins are stored relative to the chamber. During free flight each body keeps
  its absolute angular momentum fixed in inertial space, so the spinning chamber
  sees it precess at -Omega (this is the Coriolis torque on grain rotation).

Known simplifications (see README): grains are spheres; aggregates are rigid
between impacts (no rolling restructuring); no gas-mediated interaction
between grains (negligible at Kn >> 1); photophoresis without shadowing.

Provenance: written for DOLL-E; no REBOUND code is used here. Every method and
its source is listed in dolle_sim/provenance.py (and PROVENANCE.md).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial import cKDTree

from .materials import get_material, pair_props, sphere_mass
from .contact import JKRContact
from .environment import Environment, G0
from .electrostatics import SpaceChargeSolver, short_range_kernel
from .sizes import sample_diameters

K_E = 8.9875517923e9
G_N = 6.67430e-11
Q_E = 1.602176634e-19
EPS0 = 8.8541878128e-12
K_B = 1.380649e-23
C_LIGHT = 2.99792458e8
FRAG_TAG_S = 0.05            # impacts within this time of a fragmentation are tagged post_fragment


def _cross3(a, b):
    """Cross product of two 3-vectors (much cheaper than np.cross for single vectors)."""
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


def _crossN(a, b):
    """Row-wise cross product of (N,3) arrays (either may be a broadcastable (1,3) row)."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=-1)


def _skew(r):
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


class ChamberSim:
    def __init__(self, cfg: dict, chamber: dict, seed: int = 0):
        self.cfg = cfg
        self.chamber = chamber
        self.name = chamber.get("name", "chamber")
        self.rng = np.random.default_rng(seed)
        env_cfg = dict(cfg.get("environment", {}))
        env_cfg.update(chamber.get("environment", {}))
        self.env = Environment(env_cfg, self.rng)

        self.size = np.asarray(chamber.get("size_m", [0.05, 0.05, 0.05]), float)
        # position of the chamber's corner (origin) relative to the spin axis
        self.axis_offset = np.asarray(chamber.get("center_from_spin_axis_m", [0, 0, 0]), float) - self.size / 2
        ov = cfg.get("materials", {})
        self.wall_mat = get_material(chamber.get("wall_material", "aluminum_6061"), ov)

        cc = cfg.get("contact", {})
        self.mu_f = float(cc.get("friction_coefficient", 0.3))
        self.e_t = float(cc.get("tangential_restitution", 0.0))
        self.wall_mode = cc.get("wall_sticking", "jkr")          # jkr | never | always
        self.charging = bool(cc.get("contact_charging", False))
        self.z0 = float(cc.get("charging_cutoff_distance_m", 5e-10))
        self.fragmentation = bool(cc.get("fragmentation", True))
        self.skin = float(cc.get("neighbor_skin_m", 1e-3))
        self.contact_steps = int(cc.get("steps_per_contact", 300))
        # wall roughness / coating: scales the work of adhesion of grain-wall contacts
        self.wall_adh = float(chamber.get("wall_adhesion_factor", 1.0))
        vib = chamber.get("wall_vibration", {}) or {}
        self.vib_A = float(vib.get("amplitude_m", 0.0))
        self.vib_w = 2 * math.pi * float(vib.get("frequency_hz", 0.0))
        self.vib_axes = set(vib.get("axes", [0, 1, 2]))
        self.base_vib = (self.vib_A, self.vib_w, set(self.vib_axes))
        # release mechanisms: time windows (flight time, s) in which the whole tubule shakes along one axis.
        # A short burst (cycles = 1) models an impulsive kick. Wall-bound grains on the faces normal to the axis
        # are pulled off when the wall acceleration exceeds their pull-off force (see _check_detachment).
        self.release = []
        for r in chamber.get("release", []) or []:
            if r.get("type", "shake") not in ("shake", "kick"):
                raise ValueError(f"unknown release type '{r.get('type')}'")
            f = float(r["frequency_hz"])
            w = 2 * math.pi * f
            A = float(r["amplitude_m"]) if "amplitude_m" in r else float(r["peak_g"]) * G0 / (w * w)
            dur = float(r["duration_s"]) if "duration_s" in r else float(r.get("cycles", 1)) / f
            axr = r.get("axis", 2)
            axr = {"x": 0, "y": 1, "z": 2}.get(axr, axr)
            self.release.append(dict(start_s=float(r["start_s"]), end_s=float(r["start_s"]) + dur, amplitude_m=A,
                                     omega=w, axis=int(axr), peak_g=A * w * w / G0, type=r.get("type", "shake")))
        self.shake_axis = None

        fc = cfg.get("forces", {})
        self.f_coulomb = bool(fc.get("coulomb", True))
        self.f_polar = bool(fc.get("polarization", True))
        self.f_image = bool(fc.get("image_charges", True))
        self.f_selfgrav = bool(fc.get("self_gravity", True))
        self.f_lorentz = bool(fc.get("lorentz", True))
        self.f_efield = bool(fc.get("external_e_field", True))
        self.f_thermo = bool(fc.get("thermophoresis", True))
        self.f_photo = bool(fc.get("photophoresis", True))
        self.f_radp = bool(fc.get("radiation_pressure", True))
        self.f_brown = bool(fc.get("brownian", True))
        self.f_space = bool(fc.get("space_charge", True))
        self.photo_alpha = float(fc.get("photophoresis_accommodation", 1.0))
        self.coulomb_cut = float(fc.get("long_range_cutoff_m", cc.get("coulomb_cutoff_m", 3e-3)))
        self.lr_every = max(1, int(fc.get("long_range_update_every", 1)))
        self.sc_every = max(1, int(fc.get("space_charge_update_every", 25)))
        # chamber corner relative to the rocket centre of mass (for the gravity gradient)
        cg = chamber.get("center_from_cg_m", chamber.get("center_from_spin_axis_m", [0, 0, 0]))
        self.cg_offset = np.asarray(cg, float) - self.size / 2

        self.dt = float(cfg.get("dt_s", 1e-3))
        self.dt_base = self.dt
        # flight-profile runs: smaller steps while the payload is boosting / spinning fast
        self.dt_launch = float(cfg.get("dt_launch_s", 5e-4))
        Om = float(np.linalg.norm(self.env.omega))
        if Om > 0:
            self.dt = min(self.dt, 0.02 / Om)
        self.duration = float(cfg.get("duration_s", self.env.window_s))
        self.out_dt = 1.0 / float(cfg.get("output_fps", 20))
        self._step_count = 0
        self.t = 0.0

        self._contacts: dict = {}
        self._init_particles(chamber.get("particles", []), ov)
        self.collisions: list[dict] = []
        self.frames: list[np.ndarray] = []
        self.frame_bodies: list[np.ndarray] = []
        self.frame_t: list[float] = []
        self.frame_frozen: list[np.ndarray] = []
        self.frame_spin: list[np.ndarray] = []
        self.timeseries: list[dict] = []
        self.n_events = {"stick": 0, "bounce": 0, "wall_stick": 0, "wall_bounce": 0,
                         "erosion": 0, "shatter": 0, "friction_energy_guard": 0, "fragmentation_energy_limited": 0,
                         "midstep_neighbor_rebuild": 0, "rest": 0, "detach": 0, "detach_shake": 0}
        self.detach_log: list[dict] = []    # (flight time, grain, how) for every grain pulled off a wall
        self._a_app_now = np.zeros(3)
        self._W_drift = None                # spin rate used for the frame rotation during the current step
        self.E_bonds_broken = 0.0           # total bond energy paid by fragmentation [J] (energy bookkeeping)
        self._F_cf = None                   # centrifugal force per grain (excluded from body torques)
        self._nl_vlim = math.inf            # fastest grain speed the current neighbour list stays valid for
        self._topology_changed = False
        self.contain_corrections = 0        # safety-net position corrections (should stay ~0; see _contain)
        self.contain_max_m = 0.0
        self._F_pos = None
        self._F_pair = None
        # pair forces depend only on positions, charges and which grains share a body; all three can change only
        # while some body is free, so with every body wall-bound the last result is still exact (step() marks it)
        self._pair_stale = True
        # the same holds for the space-charge and image-charge forces; _sc_dirty marks motion since the last mesh solve
        self._charge_stale, self._sc_dirty, self._F_charge = True, True, None
        self._agg_strength = {}             # weakest bond (wall or grain-grain) of each wall-bound aggregate
        self.sc = None
        if self.f_space and self.f_coulomb and (self.q.any() or self.charging):
            self.sc = SpaceChargeSolver(self.size, self.coulomb_cut / 3.5)
            # short-range pair sum must reach ~3.5 sigma of the (possibly widened) Ewald split
            self.coulomb_cut = max(self.coulomb_cut, 3.5 * self.sc.sigma)

    # ------------------------------------------------------------ setup
    def _settle_on_face(self, r, face, ofs):
        """Random sequential placement of grains of radii r on one face, each touching it ("settled" bed).
        Grains do not overlap (centre distance >= 1.06 x the radius sum), also with grains settled earlier by other
        groups or on other faces, and stay clear of the side walls."""
        ax = {"x": 0, "y": 1, "z": 2}[face[1]]
        side = -1.0 if face[0] == "-" else 1.0
        tang = [k for k in range(3) if k != ax]
        P, Rr = self._settled_xyz, self._settled_r          # every grain settled so far (all groups, all faces)
        out = np.zeros((len(r), 3))
        for i, ri in enumerate(r):
            for _ in range(20000):
                c = np.empty(3)
                for k in tang:
                    c[k] = self.rng.uniform(ri, self.size[k] - ri)
                c[ax] = ri if side < 0 else self.size[ax] - ri
                if not len(Rr) or np.all(((P - c) ** 2).sum(1) >= (1.06 * (Rr + ri)) ** 2):
                    break
            else:
                raise ValueError(f"chamber '{self.name}': cannot fit {len(r)} grains as one layer on face {face}; "
                                 "multi-layer beds are not modelled")
            P = np.vstack([P, c])
            Rr = np.append(Rr, ri)
            out[i] = c
            self._settled.append((ofs + i, ax, side))
        self._settled_xyz, self._settled_r = P, Rr
        return out

    def _init_particles(self, groups, overrides):
        mats, mat_idx, rad, q, vel, pos, group_idx, spin, adh = [], [], [], [], [], [], [], [], []
        self.groups = []
        self._settled = []
        self._ever_freed = set()              # grains released from a wall at least once (detachment)
        self._settled_xyz, self._settled_r = np.zeros((0, 3)), np.zeros(0)
        n_ofs = 0
        for gi, g in enumerate(groups):
            m = get_material(g["material"], overrides)
            if m.name not in [x.name for x in mats]:
                mats.append(m)
            mi = [x.name for x in mats].index(m.name)
            n = int(g["count"])
            if "size_distribution" in g:
                # manufacturer / measured size distribution (sizes.py: number-weighted sampling)
                r = 0.5 * sample_diameters(g["size_distribution"], n, self.rng)
                r0 = float(r.max())
            else:
                r0 = float(g.get("radius_m", 50e-6))
                rs = float(g.get("radius_rel_sigma", 0.0))
                r = r0 * np.clip(1 + rs * self.rng.standard_normal(n), 0.5, 1.5)
            reg = g.get("region_m", {})
            margin = 2 * r0
            center = np.asarray(reg.get("center", self.size / 2), float)
            span = np.asarray(reg.get("size", self.size - 2 * margin), float)
            p = center + (self.rng.random((n, 3)) - 0.5) * span
            p = np.clip(p, margin, self.size - margin)
            init = g.get("initial_state", {}) or {}
            if init.get("type") == "settled":
                p = self._settle_on_face(r, init.get("face", "-z"), n_ofs)
            vcfg = g.get("velocity", {})
            vm = float(vcfg.get("speed_mean_m_s", 0.01))
            vs = float(vcfg.get("speed_sigma_m_s", 0.3 * vm))
            speed = np.abs(vm + vs * self.rng.standard_normal(n))
            mode = vcfg.get("mode", "isotropic")
            if mode == "jet":
                d0 = np.asarray(vcfg.get("direction", [1, 0, 0]), float)
                d0 /= np.linalg.norm(d0)
                cone = math.radians(float(vcfg.get("cone_deg", 10)))
                d = d0 + math.tan(cone) * self.rng.standard_normal((n, 3)) / math.sqrt(2)
            else:
                d = self.rng.standard_normal((n, 3))
            d /= np.linalg.norm(d, axis=1)[:, None]
            vel.append(d * speed[:, None])
            sc = g.get("spin", {})
            sm = float(sc.get("rate_hz_mean", 0.0))
            ss = float(sc.get("rate_hz_sigma", 0.5 * sm))
            ax = self.rng.standard_normal((n, 3))
            ax /= np.linalg.norm(ax, axis=1)[:, None]
            spin.append(ax * (2 * math.pi * np.abs(sm + ss * self.rng.standard_normal(n)))[:, None])
            gs = float(g.get("surface_energy_log_sigma", 0.0))
            adh.append(np.exp(gs * self.rng.standard_normal(n)))
            qm = float(g.get("charge_e_mean", 0.0))
            qs = float(g.get("charge_e_sigma", 0.0))
            q.append((qm + qs * self.rng.standard_normal(n)) * Q_E)
            mat_idx.append(np.full(n, mi))
            rad.append(r)
            pos.append(p)
            group_idx.append(np.full(n, gi))
            self.groups.append({**g, "material_obj": m})
            n_ofs += n
        self.mats = mats
        if not pos:
            raise ValueError(f"chamber '{self.name}' has no particles")
        self.pos = np.concatenate(pos)
        self.rad = np.concatenate(rad)
        self.mat = np.concatenate(mat_idx).astype(int)
        self.grp = np.concatenate(group_idx).astype(int)
        self.q = np.concatenate(q)
        N = self.N = len(self.pos)
        self.mass = np.array([sphere_mass(self.mats[self.mat[k]], self.rad[k]) for k in range(N)])
        self.I0 = 0.4 * self.mass * self.rad ** 2
        self.dragc = self.env.drag_coeff(self.rad)
        self.rotc = self.env.rot_drag_coeff(self.rad)
        self.polar = np.array([self.mats[k].polarizability for k in self.mat])
        self.adh = np.concatenate(adh)
        self.F_static = self._static_forces()
        self._remove_overlaps()

        # bodies: initially one per monomer
        self.body = np.arange(N)
        self.members = [[k] for k in range(N)]
        self.off = np.zeros((N, 3))          # monomer offset from its body COM, world frame
        self.offb = np.zeros((N, 3))         # same, body (principal-axis) frame
        self.X = self.pos.copy()
        self.V = np.concatenate(vel)
        self.M = self.mass.copy()
        self.L = np.concatenate(spin) * self.I0[:, None]
        self.Q = np.repeat(np.eye(3)[None], N, axis=0)     # body -> world rotation
        self.Ip = np.repeat(self.I0[:, None], 3, axis=1)    # principal moments
        self.Iinv = np.zeros((N, 3, 3))
        self.Iinv[:, [0, 1, 2], [0, 1, 2]] = (1.0 / self.I0)[:, None]
        self.Iiso = self.I0.copy()           # mean principal moment, for gas spin-down
        self.brot = self.rotc.copy()         # rotational drag coefficient of each body
        self.alive = np.ones(N, bool)
        self.frozen = np.zeros(N, bool)
        self.fw_ax = np.full(N, -1)          # wall a frozen body is stuck to (axis, side)
        self.fw_side = np.zeros(N)
        self.nmono = np.ones(N, int)
        self.bdrag = self.dragc.copy()
        self.last_split = np.full(N, -np.inf)
        for k, ax, side in self._settled:
            self.frozen[k] = True
            self.fw_ax[k], self.fw_side[k] = ax, side
            self.V[k] = 0
            self.L[k] = 0
        # JKR pull-off force of each grain from the wall, F_c = 1.5 pi w R (w with this grain's adhesion factor)
        self.Fc_wall = np.array([self._contact(k, None).p.F_c for k in range(N)])
        if self.wall_mode == "never":
            # idealised non-adhesive walls: a grain resting on one (under a load) leaves as soon as the load turns
            self.Fc_wall[:] = 0.0

    def _remove_overlaps(self):
        settled = np.zeros(len(self.pos), bool)
        settled[[k for k, _, _ in self._settled]] = True
        for _ in range(50):
            pairs = cKDTree(self.pos).query_pairs(2.05 * self.rad.max(), output_type="ndarray")
            if len(pairs) == 0:
                return
            d = np.linalg.norm(self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]], axis=1)
            close = d < 1.05 * (self.rad[pairs[:, 0]] + self.rad[pairs[:, 1]])
            # move a grain that is not part of a settled bed
            bad = np.where(settled[pairs[close, 1]], pairs[close, 0], pairs[close, 1])
            bad = bad[~settled[bad]]
            if len(bad) == 0:
                return
            m = 2 * self.rad.max()
            self.pos[bad] += (self.rng.random((len(bad), 3)) - 0.5) * 6 * self.rad.max()
            self.pos[bad] = np.clip(self.pos[bad], m, self.size - m)

    # ------------------------------------------------------------ contacts
    def _contact(self, i, j):
        """Contact model for grain i with grain j (j=None: wall). Cached on rounded parameters."""
        mi, ri = self.mat[i], self.rad[i]
        if j is None:
            f = self.adh[i] * self.wall_adh
            mj, rj = None, None
        else:
            f = math.sqrt(self.adh[i] * self.adh[j])
            mj, rj = self.mat[j], self.rad[j]
        key = (mi, round(ri * 2e6), mj, None if rj is None else round(rj * 2e6), round(math.log(f) * 50))
        c = self._contacts.get(key)
        if c is None:
            m2 = self.wall_mat if mj is None else self.mats[mj]
            f_r = math.exp(key[-1] / 50)
            c = self._contacts[key] = JKRContact(pair_props(self.mats[mi], ri, m2, rj, adhesion_factor=f_r),
                                                 steps=self.contact_steps)
        return c

    def _static_forces(self):
        """Position-independent forces on each grain: thermophoresis, photophoresis, radiation pressure."""
        env, F = self.env, np.zeros((self.N, 3))
        R = self.rad[:, None]
        if self.f_thermo and env.P > 0 and env.grad_T.any():
            # Waldmann free-molecular thermophoresis, F = -(32/15) R^2 kappa_tr grad T / v_th, with the
            # chamber-scale temperature-jump reduction of the gas heat flux (Kn_chamber = mfp / L)
            Kn_ch = env.mfp / self.size
            gT = env.grad_T / (1 + 3.9 * Kn_ch)
            F += -(32.0 / 15.0) * R ** 2 * env.kappa_tr / env.v_th * gT[None, :]
        if env.light_I > 0:
            d = env.light_dir
            absorb = np.array([self.mats[k].absorptivity for k in self.mat])[:, None]
            kp = np.array([self.mats[k].thermal_conductivity for k in self.mat])[:, None]
            if self.f_photo and env.P > 0:
                # free-molecular photophoresis of a conduction-limited sphere:
                # F = (pi/3) alpha J1 p I R^3 / (k_p T), J1 = absorptivity / 2; pushes away from the light
                F += (math.pi / 3) * self.photo_alpha * (0.5 * absorb) * env.P * env.light_I * R ** 3 \
                    / (env.T * kp) * d[None, :]
            if self.f_radp:
                # radiation pressure, Q_pr ~ 1 for black grains, ~0.3 for transparent (forward scattering)
                Q_pr = absorb + 0.3 * (1 - absorb)
                F += Q_pr * env.light_I * math.pi * R ** 2 / C_LIGHT * d[None, :]
        return F

    # ------------------------------------------------------------ rigid-body helpers
    def _w(self, b):
        """Angular velocity of body b."""
        return self.Iinv[b] @ self.L[b]

    def _omega(self):
        """Angular velocities of rotating aggregates (zeros for single grains: their spin
        does not move their centre; it is evaluated exactly at impacts via _w)."""
        W = np.zeros((self.N, 3))
        multi = np.flatnonzero(self.alive & ~self.frozen & (self.nmono > 1))
        if len(multi):
            W[multi] = np.einsum("bij,bj->bi", self.Iinv[multi], self.L[multi])
        return W

    def _mono_vel(self, W=None):
        """Centroid velocity of every grain (rigid-body motion of its body)."""
        if W is None:
            W = self._omega()
        vm = self.V[self.body]
        mm = np.flatnonzero(self.nmono[self.body] > 1)
        if len(mm):
            vm[mm] += _crossN(W[self.body[mm]], self.off[mm])
        return vm

    def _set_principal(self, b, I):
        """Store inertia tensor I (world frame) of body b as principal moments + orientation."""
        vals, vecs = np.linalg.eigh(I)
        if np.linalg.det(vecs) < 0:
            vecs[:, 2] *= -1
        self.Q[b], self.Ip[b] = vecs, vals
        self.Iinv[b] = vecs @ np.diag(1.0 / vals) @ vecs.T
        self.Iiso[b] = vals.mean()
        mem = self.members[b]
        self.offb[mem] = self.off[mem] @ vecs

    def _rotate(self, idx, h):
        """Free rigid-body rotation over h: DLM symplectic splitting about the principal axes."""
        Q = self.Q[idx].copy()
        Ip = self.Ip[idx]
        Lb = np.einsum("kji,kj->ki", Q, self.L[idx])          # Q^T L
        for a, tau in ((0, 0.5 * h), (1, 0.5 * h), (2, h), (1, 0.5 * h), (0, 0.5 * h)):
            th = tau * Lb[:, a] / Ip[:, a]
            c, s = np.cos(th), np.sin(th)
            b1, b2 = (a + 1) % 3, (a + 2) % 3
            l1, l2 = Lb[:, b1].copy(), Lb[:, b2].copy()
            Lb[:, b1] = c * l1 + s * l2
            Lb[:, b2] = -s * l1 + c * l2
            q1, q2 = Q[:, :, b1].copy(), Q[:, :, b2].copy()
            Q[:, :, b1] = c[:, None] * q1 + s[:, None] * q2
            Q[:, :, b2] = -s[:, None] * q1 + c[:, None] * q2
        self.Q[idx] = Q
        self.L[idx] = np.einsum("kij,kj->ki", Q, Lb)
        self.Iinv[idx] = Q @ (np.eye(3)[None] / Ip[:, None, :]) @ np.transpose(Q, (0, 2, 1))
        local = np.full(self.N, -1)
        local[idx] = np.arange(len(idx))
        mm = np.flatnonzero(local[self.body] >= 0)
        self.off[mm] = np.einsum("kij,kj->ki", Q[local[self.body[mm]]], self.offb[mm])

    def _frame_rotation(self, h):
        """Rotation matrix R(-Omega h): inertially fixed vectors seen from the chamber, which spins at Omega.
        With a time-dependent spin the rate at the middle of the step is used (exact for a piecewise-linear
        spin profile)."""
        # signed rate about spin_axis (a negative spin_rate_hz spins the other way)
        W = self._W_drift if self._W_drift is not None else float(self.env.omega @ self.env.spin_axis)
        if W == 0:
            return None
        k = self.env.spin_axis
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        th = -W * h
        return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)

    def _drift(self, h, act):
        """Free flight over h. Spin is stored relative to the chamber (L = I omega_rel); in the spinning
        chamber frame a torque-free body keeps its *absolute* angular momentum L + I Omega fixed in inertial
        space, which the chamber sees rotating at -Omega."""
        if h <= 0:
            return
        self.X[act] += self.V[act] * h
        Om = self.env.omega
        Rf = self._frame_rotation(h)
        multi = act[self.nmono[act] > 1]
        single = act[self.nmono[act] == 1]
        if Rf is not None and len(single):
            I0 = self.Iiso[single][:, None]
            La = self.L[single] + I0 * Om[None, :]
            self.L[single] = La @ Rf.T - I0 * Om[None, :]
        if len(multi):
            if Rf is not None:
                Iw = self.Q[multi] @ (self.Ip[multi][:, :, None] * np.transpose(self.Q[multi], (0, 2, 1)))
                self.L[multi] += Iw @ Om
            self._rotate(multi, h)
            if Rf is not None:
                self.Q[multi] = Rf[None] @ self.Q[multi]
                self.L[multi] = self.L[multi] @ Rf.T
                Q, Ip = self.Q[multi], self.Ip[multi]
                self.Iinv[multi] = Q @ (np.eye(3)[None] / Ip[:, None, :]) @ np.transpose(Q, (0, 2, 1))
                Iw = Q @ (Ip[:, :, None] * np.transpose(Q, (0, 2, 1)))
                self.L[multi] -= Iw @ Om
                local = np.full(self.N, -1)
                local[multi] = np.arange(len(multi))
                mm = np.flatnonzero(local[self.body] >= 0)
                self.off[mm] = np.einsum("kij,kj->ki", Q[local[self.body[mm]]], self.offb[mm])
        self.pos = self.X[self.body] + self.off

    def _rotation_window(self, vm, horizon):
        """Longest drift (<= horizon) over which collision prediction may move every grain in a straight line.

        Members of a spinning aggregate move on arcs, which the straight-line prediction follows only for
        small rotation angles (<= 0.05 rad: path error < 0.1% of the aggregate size). That matters only if
        the aggregate can touch something during the drift. A member moves at most (|V_b| + w_b r) h and a
        partner at most v_lim h, so if every gap to a neighbour or wall is larger than they can close in h,
        no contact is possible on either the true or the predicted path. Sets self._win_vlim, the partner
        speed assumed here: a collision that speeds a grain up beyond it forces a re-prediction."""
        self._win_vlim = None
        multi = np.flatnonzero(self.alive & ~self.frozen & (self.nmono > 1))
        if not len(multi) or horizon <= 0:
            return horizon
        Om = float(np.linalg.norm(self.env.omega))
        Ip = self.Ip[multi]
        # torque-free bound on the spin relative to the chamber: |w_abs| <= |L_abs| / I_min, w_rel = w_abs - Omega
        w = (np.linalg.norm(self.L[multi], axis=1) + Ip.max(1) * Om) / Ip.min(1) + Om
        fast = w * horizon > 0.05
        if not fast.any():
            return horizon
        w_b = np.zeros(len(self.alive))
        w_b[multi[fast]] = w[fast]
        m = np.flatnonzero(w_b[self.body] > 0)                 # members of fast-spinning aggregates
        s_m = np.linalg.norm(self.V[self.body[m]], axis=1) + w_b[self.body[m]] * np.linalg.norm(self.off[m], axis=1)
        v_lim = max(self._speed_bound(vm), float(s_m.max()))
        gap = np.full(self.N, np.inf)
        if len(self.nlist):
            infast = w_b[self.body] > 0
            i, j = self.nlist[:, 0], self.nlist[:, 1]
            sel = (infast[i] | infast[j]) & (self.body[i] != self.body[j])
            if sel.any():
                i, j = i[sel], j[sel]
                d = np.linalg.norm(self.pos[i] - self.pos[j], axis=1) - self.rad[i] - self.rad[j]
                np.minimum.at(gap, i, d)
                np.minimum.at(gap, j, d)
        p, r = self.pos[m], self.rad[m][:, None]
        g_wall = np.minimum(p - r, self.size[None, :] - p - r).min(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.minimum(np.maximum(gap[m], 0) / (s_m + v_lim), np.maximum(g_wall, 0) / s_m)
        h = np.maximum(np.nan_to_num(h, nan=np.inf), 0.05 / w_b[self.body[m]])
        self._win_vlim = v_lim
        return min(horizon, float(h.min()))

    def _K_body(self, b, r):
        """3x3 collision (inverse effective mass) matrix of body b at lever arm r."""
        if self.frozen[b]:
            return np.zeros((3, 3))
        S = _skew(r)
        return np.eye(3) / self.M[b] - S @ self.Iinv[b] @ S

    def _apply_impulse(self, b, r, P):
        if self.frozen[b]:
            return
        self.V[b] += P / self.M[b]
        self.L[b] += _cross3(r, P)

    def _impulse(self, K, vrel, n, e_n):
        """Impulse P (acting on the second body; the first gets -P) for a bounce.

        Normal: post-impact normal velocity = -e_n v_n exactly (full collision matrix).
        Tangential: stick at the contact point (tangential restitution e_t) if that lies
        inside the Coulomb cone, otherwise slide with |P_t| = mu P_n.
        """
        vn = float(vrel @ n)
        vt = vrel - vn * n
        target = -e_n * vn * n - self.e_t * vt
        try:
            P = np.linalg.solve(K, vrel - target)
        except np.linalg.LinAlgError:
            P = np.linalg.lstsq(K, vrel - target, rcond=None)[0]
        Pn = float(P @ n)
        Pt = P - Pn * n
        if not (np.linalg.norm(Pt) <= self.mu_f * Pn or float(np.linalg.norm(vt)) < 1e-15):
            tv = vt / np.linalg.norm(vt)
            d = n + self.mu_f * tv
            P = (1 + e_n) * vn / float(n @ K @ d) * d
        # Rigid-body friction laws can gain energy in eccentric impacts (Kane's paradox). The change in
        # kinetic energy of an impulse P is -P.vrel + 1/2 P.K.P; if positive, fall back to the frictionless
        # normal impulse, which always dissipates: dKE = -(1 - e_n) vn Pn / 2 <= 0.
        if float(-P @ vrel + 0.5 * P @ K @ P) > 0:
            P = (1 + e_n) * vn / float(n @ K @ n) * n
            self.n_events["friction_energy_guard"] += 1
        return P

    # ------------------------------------------------------------ walls
    def _wall_speed(self, ax):
        return self.vib_A * self.vib_w if (self.vib_A > 0 and ax in self.vib_axes) else 0.0

    def _sample_wall_phase(self, v_static, u, proj):
        """Inward wall velocity at impact, from the flux-weighted phase distribution.

        Wall amplitudes (~10 um) are far below the distance a grain travels between wall
        hits, so the oscillation phase at arrival is uniform and the impact rate is
        proportional to the approach speed v_static + u cos(phi) * proj.
        Returns the inward wall velocity, or None if the grain can never reach the wall.
        """
        vmax = v_static + u * abs(proj)
        if vmax <= 0:
            return None
        if u == 0:
            return 0.0
        for _ in range(1000):
            phi = self.rng.random() * 2 * math.pi
            w_in = u * math.cos(phi)
            if self.rng.random() * vmax < v_static + w_in * proj:
                return w_in
        return u * math.copysign(1.0, proj)

    def _body_wall_velocity(self, b, n_contact, v_other_n):
        """Velocity of a frozen (wall-bound) body: the vibration of its wall, flux-weighted."""
        ax = int(self.fw_ax[b])
        if ax < 0:
            return np.zeros(3)
        u = self._wall_speed(ax)
        if u == 0:
            return np.zeros(3)
        e_in = np.zeros(3)
        e_in[ax] = -self.fw_side[b]
        proj = -float(e_in @ n_contact)          # approach speed = v_other_n + w_in * proj
        w = self._sample_wall_phase(v_other_n, u, proj)
        return np.zeros(3) if w is None else w * e_in

    # ------------------------------------------------------------ forces
    def _long_range(self):
        charged = self.q.any() and (self.f_coulomb or self.f_polar)
        return charged or self.f_selfgrav

    def _forces_pos(self, vm):
        """Force on every grain [N] except the uniform apparent acceleration and the Coriolis force
        (which are applied exactly in _kick)."""
        m = self.mass[:, None]
        F = self.F_static.copy()
        Om = self.env.omega
        self._F_cf = None
        if Om.any() or self.env.omega_dot.any():
            r_ax = self.pos + self.axis_offset
            self._F_cf = -m * _crossN(Om[None, :], _crossN(Om[None, :], r_ax))  # centrifugal
            if self.env.omega_dot.any():
                self._F_cf -= m * _crossN(self.env.omega_dot[None, :], r_ax)     # Euler (spin-rate change)
            F += self._F_cf
        if self.env.gravity_gradient:
            F += m * self.env.tidal_accel(self.pos + self.cg_offset, self.t)
        charged = self.q.any()
        if charged:
            q = self.q[:, None]
            if self.f_efield and self.env.E_field.any():
                F += q * self.env.E_field[None, :]
            if self.f_lorentz:
                B = self.env.magnetic_field(self.t)
                if B.any():
                    F += q * _crossN(vm, B[None, :])
            solve_due = self.sc is not None and (self.sc.E is None or self._step_count % self.sc_every == 0)
            if self._F_charge is None or self._charge_stale or (solve_due and self._sc_dirty):
                self._F_charge = self._charge_forces(q, solve_due)
                self._charge_stale = False
            f_sc, wc, f_img = self._F_charge          # added in the original order (bit-identical results)
            if f_sc is not None:
                F += f_sc
            if f_img is not None:
                F[wc] += f_img
        if len(self.lrlist) and self._long_range():
            if self._F_pair is None or (self._step_count % self.lr_every == 0 and self._pair_stale):
                self._F_pair = self._pair_forces()
                self._pair_stale = False
            F += self._F_pair
        return F

    def _charge_forces(self, q, solve_due):
        """Space-charge force [N] on every grain, and short-range image-charge forces on the wall candidates
        (returned as (f_sc, wall_candidates, f_img)). They depend only on positions, charges and the neighbour list,
        so while every body is wall-bound _forces_pos reuses them (re-solving the mesh once if anything moved since
        its last solve): the result is the same as recomputing every step."""
        f_sc = f_img = None
        wc = self.wall_cand
        if self.f_coulomb and self.sc is not None:
            if solve_due:
                self.sc.solve(self.pos, self.q)
                self._sc_dirty = False
            f_sc = q * self.sc.field_at(self.pos)
        if self.f_image:
            # short-range image charge -beta q behind each nearby wall (the long-range part of the
            # images of a grounded chamber is already in the space-charge mesh)
            if len(wc):
                # with the space-charge mesh the walls are grounded conductors (Dirichlet phi = 0), so use
                # the conductor image factor 1 for the short-range part too (consistent split); without the
                # mesh, a dielectric wall's image factor (eps-1)/(eps+1)
                beta = 1.0 if self.sc is not None else self.wall_mat.image_factor
                p = self.pos[wc]
                h_lo = np.maximum(p, self.rad[wc][:, None])
                h_hi = np.maximum(self.size[None, :] - p, self.rad[wc][:, None])
                k = K_E * beta * self.q[wc] ** 2
                if self.sc is not None:
                    g_lo, g_hi = short_range_kernel(2 * h_lo, self.sc.sigma), short_range_kernel(2 * h_hi, self.sc.sigma)
                else:
                    g_lo, g_hi = 1 / (2 * h_lo) ** 2, 1 / (2 * h_hi) ** 2
                f_img = k[:, None] * (g_hi - g_lo)
        return f_sc, wc, f_img

    def _pair_forces(self):
        """Short-range Coulomb, polarisation and mutual-gravity forces between nearby grains. Each pair's force is
        kept and recomputed only when one of its grains has moved or changed charge (the same formula on the same
        numbers, summed in the same order: identical to recomputing every pair)."""
        F = np.zeros((self.N, 3))
        charged = bool(self.q.any())
        i, j = self.lrlist[:, 0], self.lrlist[:, 1]
        sel = self.body[i] != self.body[j]
        i, j = i[sel], j[sel]
        hold = self.dt * self.lr_every
        c = getattr(self, "_pf_cache", None)
        if c is not None and c["lr"] is self.lrlist and c["hold"] == hold and c["charged"] == charged \
                and np.array_equal(c["body"], self.body):
            ch = np.any(self.pos != c["pos"], axis=1) | (self.q != c["q"])
            upd = ch[i] | ch[j]
            f = c["f"]
            if upd.any():
                f = f.copy()
                f[upd] = self._pair_f(i[upd], j[upd], charged, hold)
        else:
            f = self._pair_f(i, j, charged, hold)
        self._pf_cache = dict(lr=self.lrlist, hold=hold, charged=charged, body=self.body.copy(), pos=self.pos.copy(),
                              q=self.q.copy(), f=f)
        for k in range(3):
            F[:, k] += np.bincount(i, f[:, k], self.N) - np.bincount(j, f[:, k], self.N)
        return F

    def _pair_f(self, i, j, charged, hold):
        """The pair forces on grain i of the pairs (i, j) [N] (minus that on j)."""
        r = self.pos[i] - self.pos[j]                 # from j to i
        d = np.maximum(np.linalg.norm(r, axis=1), self.rad[i] + self.rad[j])
        ok = d < self.coulomb_cut
        fmag = np.zeros(len(i))                       # > 0 repulsive (along r)
        if charged and self.f_coulomb:
            g = short_range_kernel(d, self.sc.sigma) if self.sc is not None else 1 / d ** 2
            fmag += K_E * self.q[i] * self.q[j] * g
        if charged and self.f_polar:
            # charge-induced dipole: F = -2 k q^2 beta R^3 / d^5 on the polarised partner
            fmag -= 2 * K_E * (self.q[i] ** 2 * self.polar[j] * self.rad[j] ** 3
                               + self.q[j] ** 2 * self.polar[i] * self.rad[i] ** 3) / d ** 5
        if self.f_selfgrav:
            fmag -= G_N * self.mass[i] * self.mass[j] / d ** 2
        # numerical limiter: over the (held) force interval a pair force may not move the grains by more
        # than ~1/4 of their separation (prevents overshoot for close, strongly-attracting pairs; the
        # true infall is then resolved by the contact model at impact)
        mu = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
        cap = mu * d / (2 * hold ** 2)
        fmag = np.clip(fmag, -cap, cap)
        return (fmag * ok / d)[:, None] * r

    # ------------------------------------------------------------ neighbour lists
    def _speed_bound(self, vm):
        """Upper bound on any grain speed during the next step: current maximum plus the most a vibrating
        wall can add in one bounce ((1 + e) A omega <= 2 A omega)."""
        vmax = float(np.linalg.norm(vm, axis=1).max()) if len(vm) else 0.0
        return vmax + 2 * self.vib_A * self.vib_w

    def _build_neighbors(self, vm=None):
        # the skin is widened when grains are fast, so no pair can close the gap before the next check
        if vm is None:
            vm = self._mono_vel()
        self.skin_eff = max(self.skin, 3.0 * self._speed_bound(vm) * self.dt)
        c_cut = 2 * self.rad.max() + self.skin_eff
        lr = self._long_range()
        cut = max(c_cut, self.coulomb_cut + self.skin_eff if lr else 0)
        pairs = cKDTree(self.pos).query_pairs(cut, output_type="ndarray")
        if len(pairs):
            d = np.linalg.norm(self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]], axis=1)
            self.nlist = pairs[d < c_cut]                 # contact candidates (collision detection)
            self.lrlist = pairs if lr else pairs[:0]      # long-range force pairs
        else:
            self.nlist = self.lrlist = pairs
        self.pos_at_build = self.pos.copy()
        self._charge_stale = True
        self._nl_vlim = 0.5 * self.skin_eff / self.dt     # no grain may move more than half the skin per step
        self._F_pair = None
        # grains close enough to a wall to possibly reach it before the next rebuild
        gap = np.minimum(self.pos, self.size[None, :] - self.pos).min(1) - self.rad
        self.wall_cand = np.flatnonzero(gap < self.skin_eff)

    def _needs_rebuild(self, vm):
        # no pair can close the skin gap before the next check: 2 (max displacement + v_bound dt) < skin.
        # _nl_vlim: the speed up to which the list stays valid over the next step (checked after every bounce)
        disp = np.linalg.norm(self.pos - self.pos_at_build, axis=1).max()
        self._nl_vlim = (0.5 * self.skin_eff - disp) / self.dt
        return 2 * (disp + self._speed_bound(vm) * self.dt) > self.skin_eff

    # ------------------------------------------------------------ events
    def _events(self, vm, remaining, excl, wexcl, subset=None):
        """Contact events predicted within `remaining`, sorted by time.
        subset: optional boolean mask of grains; only events involving them are returned."""
        evs = []
        if len(self.nlist):
            i, j = self.nlist[:, 0], self.nlist[:, 1]
            bi, bj = self.body[i], self.body[j]
            sel = (bi != bj) & ~(self.frozen[bi] & self.frozen[bj]) & ~excl
            if subset is not None:
                sel &= subset[i] | subset[j]
            if sel.any():
                idx = np.flatnonzero(sel)
                i, j = i[idx], j[idx]
                r = self.pos[j] - self.pos[i]
                u = vm[j] - vm[i]
                S = self.rad[i] + self.rad[j]
                b = (r * u).sum(1)
                a = (u * u).sum(1)
                c = (r * r).sum(1) - S * S
                disc = b * b - a * c
                ok = np.flatnonzero((b < 0) & (disc >= 0))
                if len(ok):
                    t = np.where(c[ok] <= 0, 0.0, (-b[ok] - np.sqrt(disc[ok])) / np.maximum(a[ok], 1e-300))
                    for kk in np.flatnonzero(t <= remaining):
                        evs.append((float(t[kk]), ("pp", int(idx[ok[kk]]))))
        wc = self.wall_cand
        keep = ~self.frozen[self.body[wc]]
        if subset is not None:
            keep &= subset[wc]
        wc = wc[keep]
        if len(wc):
            p, v, r = self.pos[wc], vm[wc], self.rad[wc][:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                t_lo = np.where(v < 0, np.maximum(p - r, 0) / -v, np.inf)
                t_hi = np.where(v > 0, np.maximum(self.size[None, :] - r - p, 0) / v, np.inf)
            tw = np.minimum(t_lo, t_hi)
            tw[wexcl[wc]] = np.inf
            for k, ax in zip(*np.nonzero(tw <= remaining)):
                side = -1.0 if t_lo[k, ax] <= t_hi[k, ax] else 1.0
                evs.append((float(tw[k, ax]), ("wall", int(wc[k]), int(ax), side)))
        evs.sort(key=lambda e: e[0])
        return evs

    # ------------------------------------------------------------ impacts
    def _point_velocity(self, b, r):
        return self.V[b] + _cross3(self._w(b), r)

    def _resolve_pp(self, k, t_now):
        i, j = (int(x) for x in self.nlist[k])
        A, B = self.body[i], self.body[j]
        rvec = self.pos[j] - self.pos[i]
        n = rvec / np.linalg.norm(rvec)
        cpt = self.pos[i] + n * self.rad[i]
        rA, rB = cpt - self.X[A], cpt - self.X[B]
        wA, wB = self._w(A), self._w(B)
        fzA, fzB = bool(self.frozen[A]), bool(self.frozen[B])
        vA = self.V[A] + _cross3(wA, rA) if not fzA else None
        vB = self.V[B] + _cross3(wB, rB) if not fzB else None
        if fzA:
            vA = self._body_wall_velocity(A, -n, float(-(vB @ n)))
        if fzB:
            vB = self._body_wall_velocity(B, n, float(vA @ n))
        vrel = vA - vB
        vn = float(vrel @ n)
        if vn <= 0:
            return False
        K = self._K_body(A, rA) + self._K_body(B, rB)
        m_eff = 1.0 / float(n @ K @ n)
        cm = self._contact(i, j)
        res = cm.resolve(vn, m_eff)
        vt = float(np.linalg.norm(vrel - vn * n))
        if self.charging:
            self._charge(i, j, res.max_contact_radius)
        # grain-centroid relative velocity (what a camera tracking centroids measures)
        cA = (self.V[A] + _cross3(wA, self.off[i])) if not fzA else vA
        cB = (self.V[B] + _cross3(wB, self.off[j])) if not fzB else vB
        vcm = cA - cB
        # translational collision energy of the two bodies (for fragmentation)
        VA = self.V[A] if not fzA else vA
        VB = self.V[B] if not fzB else vB
        dV = VA - VB
        if fzA or fzB:
            mu = self.M[B] if fzA else self.M[A]
        else:
            mu = self.M[A] * self.M[B] / (self.M[A] + self.M[B])
        E_coll = 0.5 * mu * float(dV @ dV)
        nA, nB = int(self.nmono[A]), int(self.nmono[B])
        post_frag = (t_now - max(self.last_split[i], self.last_split[j])) < FRAG_TAG_S
        rec = dict(t=t_now, kind="pp", i=i, j=j, mat_i=self.mats[self.mat[i]].name,
                   mat_j=self.mats[self.mat[j]].name, n_i=nA, n_j=nB, frozen_i=fzA, frozen_j=fzB,
                   post_fragment=bool(post_frag),
                   v_n=vn, v_t=vt, v_rel=float(np.linalg.norm(vrel)), m_eff=m_eff,
                   outcome="stick" if res.stuck else "bounce",
                   e_n=res.e_n, yielded=res.yielded, v_stick_est=cm.v_stick_analytic(m_eff),
                   adhesion_factor=math.sqrt(self.adh[i] * self.adh[j]),
                   max_force=res.max_force, q_i=self.q[i] / Q_E, q_j=self.q[j] / Q_E,
                   spin_i_hz=float(np.linalg.norm(wA)) / (2 * math.pi) if not fzA else 0.0,
                   spin_j_hz=float(np.linalg.norm(wB)) / (2 * math.pi) if not fzB else 0.0,
                   vrel_x=vrel[0], vrel_y=vrel[1], vrel_z=vrel[2],
                   vcm_x=vcm[0], vcm_y=vcm[1], vcm_z=vcm[2], n_x=n[0], n_y=n[1], n_z=n[2],
                   x=cpt[0], y=cpt[1], z=cpt[2], r_sum=self.rad[i] + self.rad[j], E_coll=E_coll, E_diss=0.0,
                   fragments=0)
        self.collisions.append(rec)
        resting = (not res.stuck) and (fzA != fzB) and self._comes_to_rest(B if fzA else A, n if fzB else -n,
                                                                            res.e_n * vn)
        if resting:
            rec["outcome"] = "rest"
        if res.stuck or resting:
            self.n_events["rest" if resting else "stick"] += 1
            self._merge(A, B)
        else:
            self.n_events["bounce"] += 1
            P = self._impulse(K, vrel, n, res.e_n)
            self._apply_impulse(A, rA, -P)
            self._apply_impulse(B, rB, P)
            # kinetic energy the impulse removed (exact for rigid bodies: dKE = -P.vrel + P.K.P/2)
            rec["E_diss"] = E_diss = float(P @ vrel - 0.5 * P @ K @ P)
            if self.fragmentation and (nA > 1 or nB > 1):
                rec["fragments"] = self._fragment([(A, i), (B, j)], E_coll, E_diss, t_now)
        return True

    def _resolve_wall(self, m, ax, side, t_now):
        A = self.body[m]
        n = np.zeros(3)
        n[ax] = side                                       # from the grain into the wall
        cpt = self.pos[m] + n * self.rad[m]
        rA = cpt - self.X[A]
        wA = self._w(A)
        vA = self.V[A] + _cross3(wA, rA)
        v_static = float(vA @ n)
        u = self._wall_speed(ax)
        w_in = self._sample_wall_phase(v_static, u, 1.0)   # inward wall velocity at impact
        if w_in is None or v_static + w_in <= 0:
            return False
        vn = v_static + w_in                               # approach speed relative to the wall
        v_wall = -w_in * n                                 # wall velocity vector (inward = -n)
        vrel = vA - v_wall
        K = self._K_body(A, rA)
        m_eff = 1.0 / float(n @ K @ n)
        cm = self._contact(m, None)
        res = cm.resolve(vn, m_eff)
        stuck = res.stuck if self.wall_mode == "jkr" else self.wall_mode == "always"
        e_n = res.e_n if not res.stuck else math.sqrt(self.mats[self.mat[m]].e_dry * self.wall_mat.e_dry)
        vt = float(np.linalg.norm(vrel - vn * n))
        if self.charging:
            self._charge(m, None, res.max_contact_radius)
        dV = self.V[A] - v_wall
        E_coll = 0.5 * self.M[A] * float(dV @ dV)
        rec = dict(t=t_now, kind="wall", i=m, j=-1, mat_i=self.mats[self.mat[m]].name,
                   mat_j=self.wall_mat.name, n_i=int(self.nmono[A]), n_j=0, frozen_i=False, frozen_j=True,
                   post_fragment=bool(t_now - self.last_split[m] < FRAG_TAG_S),
                   v_n=vn, v_t=vt, v_rel=float(np.linalg.norm(vrel)), m_eff=m_eff,
                   outcome="stick" if stuck else "bounce",
                   e_n=0.0 if stuck else e_n, yielded=res.yielded, v_stick_est=cm.v_stick_analytic(m_eff),
                   adhesion_factor=self.adh[m] * self.wall_adh,
                   max_force=res.max_force, q_i=self.q[m] / Q_E, q_j=0.0,
                   spin_i_hz=float(np.linalg.norm(wA)) / (2 * math.pi), spin_j_hz=0.0,
                   vrel_x=vrel[0], vrel_y=vrel[1], vrel_z=vrel[2],
                   vcm_x=dV[0], vcm_y=dV[1], vcm_z=dV[2], n_x=n[0], n_y=n[1], n_z=n[2],
                   x=cpt[0], y=cpt[1], z=cpt[2], r_sum=self.rad[m], E_coll=E_coll, E_diss=0.0, fragments=0)
        resting = (not stuck) and self._comes_to_rest(A, n, e_n * vn)
        if resting:
            rec["outcome"] = "rest"
        self.collisions.append(rec)
        if stuck or resting:
            self.n_events["rest" if resting else "wall_stick"] += 1
            self.frozen[A] = True
            self.fw_ax[A], self.fw_side[A] = ax, side
            self.V[A] = 0
            self.L[A] = 0
        else:
            self.n_events["wall_bounce"] += 1
            P = self._impulse(K, vrel, n, e_n)
            self._apply_impulse(A, rA, -P)
            # kinetic energy removed, in the wall's frame (the wall is infinitely massive)
            rec["E_diss"] = E_diss = float(P @ vrel - 0.5 * P @ K @ P)
            if self.fragmentation and self.nmono[A] > 1:
                rec["fragments"] = self._fragment([(A, m)], E_coll, E_diss, t_now)
        return True

    # ------------------------------------------------------------ fragmentation
    def _bond_energy(self, b, k):
        """Break energy of the grain-grain bond holding grain k in aggregate b (nearest member)."""
        mem = [x for x in self.members[b] if x != k]
        d = np.linalg.norm(self.pos[mem] - self.pos[k], axis=1)
        return self._contact(k, mem[int(np.argmin(d))]).p.E_break

    def _contacts_of(self, mem, k=None):
        """Number of touching grain pairs among mem (or only those involving grain k)."""
        mem = list(mem)
        n = 0
        for a_i, a in enumerate(mem):
            for b in mem[a_i + 1:]:
                if k is not None and k not in (a, b):
                    continue
                if np.linalg.norm(self.pos[a] - self.pos[b]) < (self.rad[a] + self.rad[b]) * (1 + 1e-6):
                    n += 1
        return n

    def _fragment(self, hits, E_coll, E_diss, t_now):
        """Dominik & Tielens (1997) thresholds: erosion above ~3 N_c E_break, shattering above ~10 N_c E_break,
        with E_coll the translational relative kinetic energy of the two bodies.

        Energy budget (rigid bodies): the only energy available is E_diss, the kinetic energy the bounce
        actually removed. Every broken contact costs E_break; if E_diss cannot pay for a shatter, the impact
        erodes the struck grain instead, and if it cannot pay for that either, nothing breaks. Fragments
        separate with what is left, so kinetic + bond energy never increases."""
        agg = [(b, k) for b, k in hits if self.nmono[b] > 1 and not self.frozen[b]]
        if not agg:
            return 0
        N_c = sum(int(self.nmono[b]) - 1 for b, _ in agg)
        E_b = float(np.mean([self._bond_energy(b, k) for b, k in agg]))
        options = []
        if E_coll > 10 * N_c * E_b:
            options.append(("shatter", {b: [[x] for x in self.members[b]][1:] for b, _ in agg},
                            sum(self._contacts_of(self.members[b]) for b, _ in agg)))
        if E_coll > 3 * N_c * E_b:
            options.append(("erosion", {b: [[k]] for b, k in agg},
                            sum(self._contacts_of(self.members[b], k) for b, k in agg)))
        if not options:
            return 0
        for mode, detach, n_contacts in options:
            cost = n_contacts * E_b
            if E_diss >= cost:
                break
        else:
            self.n_events["fragmentation_energy_limited"] += 1
            return 0
        if mode != options[0][0]:
            self.n_events["fragmentation_energy_limited"] += 1
        self.E_bonds_broken += cost
        E_sep = E_diss - cost
        out = 0
        for b, groups in detach.items():
            members_before = list(self.members[b])
            X0 = self.X[b].copy()
            new = self._detach(b, groups)
            new += self._split_disconnected(b)
            out += sum(len(g) for g in groups)
            self.last_split[members_before] = t_now
            self._separate([b] + new, X0, E_sep / len(detach))
        self.n_events[mode] += 1
        self._topology_changed = True
        return out

    def _separate(self, parts, X0, E_sep):
        """Give fragments radial separation velocities s*u from the old COM that raise their kinetic energy by
        exactly E_sep, conserving momentum (sum m u = 0) and angular momentum (radial kicks about the COM carry
        none). The fragments already move (and a spinning parent gives them different velocities), so
        dKE = s B + s^2 A with A = sum m u^2 / 2, B = sum m V.u; s is the positive root of A s^2 + B s = E_sep."""
        if E_sep <= 0 or len(parts) < 2:
            return
        m = self.M[parts]
        u = self.X[parts] - X0
        nrm = np.linalg.norm(u, axis=1)
        nrm[nrm == 0] = 1.0
        u = u / nrm[:, None]
        u -= (m[:, None] * u).sum(0) / m.sum()
        A = 0.5 * float((m * (u * u).sum(1)).sum())
        if A <= 0:
            return
        B = float((m * (self.V[parts] * u).sum(1)).sum())
        s = (-B + math.sqrt(B * B + 4 * A * E_sep)) / (2 * A)
        self.V[parts] += s * u

    def _inertia(self, mem, X):
        I = np.zeros((3, 3))
        for k in mem:
            r = self.pos[k] - X
            I += (self.I0[k] + self.mass[k] * (r @ r)) * np.eye(3) - self.mass[k] * np.outer(r, r)
        return I

    def _components(self, mem):
        """Groups of mutually touching grains (an aggregate is held together only by contacts)."""
        mem = list(mem)
        left, comps = set(mem), []
        while left:
            stack = [left.pop()]
            comp = list(stack)
            while stack:
                k = stack.pop()
                for j in list(left):
                    if np.linalg.norm(self.pos[j] - self.pos[k]) < (self.rad[j] + self.rad[k]) * (1 + 1e-6):
                        left.remove(j)
                        stack.append(j)
                        comp.append(j)
            comps.append(comp)
        return comps

    def _split_disconnected(self, b):
        """After grains were removed, any part of body b no longer in contact becomes its own body."""
        if self.nmono[b] < 2:
            return []
        comps = self._components(self.members[b])
        if len(comps) < 2:
            return []
        comps.sort(key=len, reverse=True)
        return self._detach(b, comps[1:])

    def _detach(self, b, groups):
        """Detach each group of member grains of body b as its own rigid body (moving with b's rigid-body
        motion at that instant). Linear and angular momentum are conserved. Returns the new body ids."""
        w = self._w(b)
        P = self.M[b] * self.V[b]
        Lo = self.L[b] + self.M[b] * _cross3(self.X[b], self.V[b])
        free = list(np.flatnonzero(~self.alive)[:len(groups)])
        new = []
        for g, nb in zip(groups, free):
            g = [int(k) for k in g]
            mg = self.mass[g]
            Mg = float(mg.sum())
            vg = self.V[b][None, :] + _crossN(w[None, :], self.off[g])
            Xg = (mg[:, None] * self.pos[g]).sum(0) / Mg
            Vg = (mg[:, None] * vg).sum(0) / Mg
            Lg = np.zeros(3)
            for i, k in enumerate(g):
                Lg += self.I0[k] * w + self.mass[k] * _cross3(self.pos[k] - Xg, vg[i] - Vg)
            P = P - Mg * Vg
            Lo = Lo - (Lg + Mg * _cross3(Xg, Vg))
            for k in g:
                self.members[b].remove(k)
                self.body[k] = nb
            self.members[nb] = g
            self.alive[nb], self.frozen[nb], self.nmono[nb] = True, False, len(g)
            self.fw_ax[nb] = -1
            self.X[nb], self.V[nb], self.M[nb], self.L[nb] = Xg, Vg, Mg, Lg
            self.off[g] = self.pos[g] - Xg
            if len(g) == 1:
                k = g[0]
                self.Iinv[nb] = np.eye(3) / self.I0[k]
                self.Q[nb], self.Ip[nb], self.Iiso[nb] = np.eye(3), self.I0[k], self.I0[k]
                self.offb[k] = 0.0
            else:
                self._set_principal(nb, self._inertia(g, Xg))
            self.bdrag[nb] = self.dragc[g].sum()
            self.brot[nb] = self._rot_drag(g)
            new.append(int(nb))
        mem = self.members[b]
        M = self.mass[mem].sum()
        X = (self.mass[mem, None] * self.pos[mem]).sum(0) / M
        self.M[b], self.X[b], self.V[b] = M, X, P / M
        self.L[b] = Lo - M * _cross3(X, P / M)
        self.off[mem] = self.pos[mem] - X
        self.nmono[b] = len(mem)
        self._set_principal(b, self._inertia(mem, X))
        self.bdrag[b] = self.dragc[mem].sum()
        self.brot[b] = self._rot_drag(mem)
        return new

    def _split(self, b, ks):
        """Detach grains ks from body b as single grains (then split off anything left disconnected)."""
        new = self._detach(b, [[k] for k in ks])
        return new + self._split_disconnected(b)

    def _rot_drag(self, mem):
        # gas torque coefficient: own spin drag + translational drag at the lever arm, averaged over
        # rotation axes (<r_perp^2> = 2/3 |r|^2), consistent with Iiso = trace(I)/3
        return float((self.rotc[mem] + (2.0 / 3.0) * self.dragc[mem] * (self.off[mem] ** 2).sum(1)).sum())

    def _charge(self, i, j, a):
        """Condenser model of contact charging (e.g. Lacks & Sankaran 2011).

        Charge flows through the contact area pi a^2 across a tunnelling gap z0,
        driven by the contact-potential difference minus the potential difference
        the grains already carry, so it saturates at |V_grain| ~ delta-phi:
            dq = (eps0 pi a^2 / z0) * [(phi_i - phi_j) + (V_i - V_j)],  q_i -= dq, q_j += dq
        (V = k q / R is the grain's surface potential; walls are grounded, V = 0).
        """
        phi_i = self.mats[self.mat[i]].work_function
        phi_j = self.wall_mat.work_function if j is None else self.mats[self.mat[j]].work_function
        V_i = K_E * self.q[i] / self.rad[i]
        V_j = 0.0 if j is None else K_E * self.q[j] / self.rad[j]
        C_c = EPS0 * math.pi * a * a / self.z0
        # cannot move more charge than brings the two potentials into balance
        C_i = self.rad[i] / K_E
        C_j = math.inf if j is None else self.rad[j] / K_E
        C_eq = 1.0 / (1.0 / C_i + 1.0 / C_j)
        drive = (phi_i - phi_j) + (V_i - V_j)          # electrons flow toward the higher work function
        dq = math.copysign(min(abs(C_c * drive), abs(C_eq * drive)), drive)
        self.q[i] -= dq
        if j is not None:
            self.q[j] += dq

    def _merge(self, A, B):
        if self.nmono[B] > self.nmono[A]:
            A, B = B, A
        MA, MB = self.M[A], self.M[B]
        M = MA + MB
        X = (MA * self.X[A] + MB * self.X[B]) / M
        V = (MA * self.V[A] + MB * self.V[B]) / M
        L = self.L[A] + self.L[B] + MA * _cross3(self.X[A] - X, self.V[A] - V) + MB * _cross3(self.X[B] - X, self.V[B] - V)
        I = np.zeros((3, 3))
        for b in (A, B):
            d = self.X[b] - X
            Ib = self.Q[b] @ np.diag(self.Ip[b]) @ self.Q[b].T
            I += Ib + self.M[b] * ((d @ d) * np.eye(3) - np.outer(d, d))
        fz = bool(self.frozen[A] or self.frozen[B])
        if fz:
            src = A if self.frozen[A] else B
            self.fw_ax[A], self.fw_side[A] = self.fw_ax[src], self.fw_side[src]
        mem = self.members[A] + self.members[B]
        self.members[A], self.members[B] = mem, []
        self.body[mem] = A
        self.X[A], self.M[A] = X, M
        self.V[A], self.L[A] = (np.zeros(3), np.zeros(3)) if fz else (V, L)
        self.off[mem] = self.pos[mem] - X
        self.nmono[A] += self.nmono[B]
        self._set_principal(A, I)
        self.brot[A] = self._rot_drag(mem)
        self.bdrag[A] += self.bdrag[B]
        self.frozen[A] = fz
        self.alive[B] = False
        self.nmono[B] = 0
        self.fw_ax[B] = -1

    # ------------------------------------------------------------ time integration
    def _kick(self, F, a_app, h, brownian=False):
        """Exact velocity update over h for all free bodies:
            dV/dt = a_app + F/M - V/tau - 2 Omega x V
        with F (sum of grain forces) held constant. Drag and Coriolis are integrated exactly
        (damped rotation about the spin axis); angular momentum gets the torque and gas spin-down."""
        act = np.flatnonzero(self.alive & ~self.frozen)
        if not len(act):
            return act
        Fb = np.empty((self.N, 3))
        for k in range(3):
            Fb[:, k] = np.bincount(self.body, F[:, k], self.N)
        Ma = self.M[act]
        acc = Fb[act] / Ma[:, None] + a_app[None, :]
        gam = self.bdrag[act] / Ma if self.env.P > 0 else np.zeros(len(act))
        V = self.V[act]
        Om = self.env.omega
        Wm = float(np.linalg.norm(Om))
        if Wm > 0:
            e = Om / Wm
            e1 = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(e, e1)
        else:
            e, e1, e2 = np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        # parallel component: exact damped update
        vp, ap = V @ e, acc @ e
        dec = np.exp(-gam * h)
        phi = np.where(gam > 0, -np.expm1(-gam * h) / np.where(gam > 0, gam, 1.0), h)
        vp = vp * dec + ap * phi
        # perpendicular: z = v.e1 + i v.e2, dz/dt = a_z - (gamma + 2 i Omega) z
        z = V @ e1 + 1j * (V @ e2)
        az = acc @ e1 + 1j * (acc @ e2)
        lam = gam + 2j * Wm
        ex = np.exp(-lam * h)
        small = np.abs(lam * h) < 1e-12
        fac = np.where(small, h, -np.expm1(-lam * h) / np.where(small, 1.0, lam))
        z = z * ex + az * fac
        self.V[act] = vp[:, None] * e + z.real[:, None] * e1 + z.imag[:, None] * e2
        Tb = None
        if (self.nmono[act] > 1).any():
            # torques from real forces only: the centrifugal (and Coriolis) torque of the spinning frame is
            # already in _drift, which keeps each body's absolute angular momentum fixed in inertial space
            T = _crossN(self.off, F if self._F_cf is None else F - self._F_cf)
            Tb = np.empty((self.N, 3))
            for k in range(3):
                Tb[:, k] = np.bincount(self.body, T[:, k], self.N)
        if self.env.P > 0:
            rdecay = np.exp(-h * self.brot[act] / self.Iiso[act])[:, None]
            self.L[act] = (self.L[act] + (Tb[act] * h if Tb is not None else 0)) * rdecay
            if brownian and self.f_brown:
                # Ornstein-Uhlenbeck thermal kicks consistent with the gas drag (variance over the full step)
                kT = K_B * self.env.T
                sv = np.sqrt(kT / Ma * (1 - np.exp(-4 * gam * h)))
                self.V[act] += sv[:, None] * self.rng.standard_normal((len(act), 3))
                Ii = self.Iiso[act]
                sl = Ii * np.sqrt(kT / Ii * (1 - rdecay[:, 0] ** 4))
                self.L[act] += sl[:, None] * self.rng.standard_normal((len(act), 3))
        elif Tb is not None:
            self.L[act] += Tb[act] * h
        return act

    def _comes_to_rest(self, b, n, v_out):
        """A bounce under a sustained load pressing body b onto its contact (unit normal n, from b into the
        contact) comes back within one step if v_out < a_n dt. Such a grain undergoes an unresolvable sequence
        of ever smaller bounces (inelastic collapse) and is at rest in adhesive contact: it is treated as
        stuck. Under the ~1e-4 g of microgravity this never applies (a_n dt ~ 1e-6 m/s)."""
        mem = self.members[b]
        a = self._a_app_now + (self._F_pos[mem].sum(0) / self.M[b] if self._F_pos is not None else 0.0)
        a_n = float(a @ n)
        return a_n > 0 and v_out < a_n * self.dt

    def _update_time(self, t):
        """Time-dependent spin and release shaking at simulation time t. Returns True if the wall vibration
        changed (the neighbour list must then be rebuilt)."""
        Om_old = self.env.omega.copy()
        self.env.update(t)
        dOm = Om_old - self.env.omega
        if dOm.any():
            # spins are stored relative to the chamber; a change of the payload's spin rate leaves each free
            # body's absolute angular momentum unchanged, so its relative spin changes by I (Omega_old - Omega_new)
            act = np.flatnonzero(self.alive & ~self.frozen)
            if len(act):
                Iw = self.Q[act] @ (self.Ip[act][:, :, None] * np.transpose(self.Q[act], (0, 2, 1)))
                self.L[act] += Iw @ dOm
        if not self.release:
            return False
        T = self.env.t0_flight + t
        A, w, axes = self.base_vib
        self.shake_axis = None
        for r in self.release:
            if r["start_s"] - 1e-12 <= T < r["end_s"] - 1e-12:     # same tolerance as _dt_at
                A, w, axes = r["amplitude_m"], r["omega"], {r["axis"]}
                self.shake_axis = r["axis"]
                break
        changed = (A, w) != (self.vib_A, self.vib_w)
        self.vib_A, self.vib_w, self.vib_axes = A, w, set(axes)
        return changed

    def _check_detachment(self, t_now):
        """Pull wall-bound grains off when the load away from the wall exceeds the JKR pull-off force.

        Load on a wall-bound body = its mass times the quasi-static apparent acceleration (uniform part,
        centrifugal, Euler, gravity gradient) plus its own forces (electrostatic, thermo-/photophoresis),
        plus, while the tubule shakes along the face normal, the wall's peak acceleration A omega^2. The
        pull-off force under a slowly varying load is F_c = 1.5 pi w R (JKR). Tangential loads (rolling,
        sliding) are not modelled. A grain freed by the shaking leaves with the wall's velocity at the moment
        the threshold is reached, A omega cos(phi) with sin(phi) = (F_c - F_qs) / (m A omega^2)."""
        fb = np.flatnonzero(self.alive & self.frozen)
        if not len(fb) or self._F_pos is None:
            return
        a_sh = self.vib_A * self.vib_w ** 2 if self.shake_axis is not None else 0.0
        single = fb[self.nmono[fb] == 1]
        if len(single):
            g = np.array([self.members[b][0] for b in single])
            u = np.zeros((len(single), 3))
            u[np.arange(len(single)), self.fw_ax[single]] = -self.fw_side[single]     # wall -> grain
            a_drive = self._a_app_now[None, :] + self._F_pos[g] / self.mass[g][:, None]
            F_qs = self.mass[g] * (a_drive * u).sum(1)
            F_sh = self.mass[g] * a_sh * (np.abs(u[:, self.shake_axis]) if self.shake_axis is not None else 0.0)
            Fc = self.Fc_wall[g]
            for k in np.flatnonzero(F_qs + F_sh > Fc):
                v = 0.0
                if F_qs[k] < Fc[k]:
                    sphi = (Fc[k] - F_qs[k]) / F_sh[k]
                    v = self.vib_A * self.vib_w * math.sqrt(max(0.0, 1.0 - sphi * sphi))
                self._free_body(int(single[k]), u[k] * v, t_now, int(g[k]), "shake" if F_qs[k] < Fc[k] else "load")
        for b in fb[self.nmono[fb] > 1]:
            if not self._aggregate_may_fail(int(b), a_sh):
                continue
            self._detach_aggregate(int(b), a_sh, t_now)

    def _aggregate_may_fail(self, b, a_sh):
        """Cheap test before the tree statics of _detach_aggregate. The load on any part of the aggregate along any
        bond is at most the sum of |m a| over all its grains (plus M a_sh while shaking). If that is below the weakest
        bond (wall or grain-grain), no bond can fail and the full check can be skipped: same result, much faster."""
        mem = self.members[b]
        key = (b, tuple(mem))
        if key not in self._agg_strength:
            fc = float(self.Fc_wall[mem].min())
            for i, k in enumerate(mem):
                for j in mem[i + 1:]:
                    if np.linalg.norm(self.pos[j] - self.pos[k]) < (self.rad[j] + self.rad[k]) * (1 + 1e-6):
                        fc = min(fc, float(self._contact(k, j).p.F_c))
            self._agg_strength[key] = fc
        a = self._a_app_now[None, :] + self._F_pos[mem] / self.mass[mem][:, None]
        load = float(np.linalg.norm(self.mass[mem][:, None] * a, axis=1).sum()) + float(self.M[b]) * a_sh
        return load >= self._agg_strength[key]

    def _free_body(self, b, v, t_now, grain, how):
        self._ever_freed.update(int(g) for g in self.members[b])
        self.frozen[b] = False
        self.fw_ax[b] = -1
        self.V[b] = v
        self.L[b] = 0.0
        self.n_events["detach"] += 1
        if how == "shake":
            self.n_events["detach_shake"] += 1
        self.detach_log.append(dict(t=self.env.t0_flight + t_now, grain=grain, n=int(self.nmono[b]), how=how,
                                    v=float(np.linalg.norm(v))))

    def _detach_aggregate(self, b, a_sh, t_now):
        """Tree statics for a wall-bound aggregate: the bond to the wall carries the load of the whole body,
        each grain-grain bond the load of the part beyond it. A bond fails in tension when its load along the
        line of centres exceeds the JKR pull-off force of that pair; the part beyond it comes off."""
        mem = list(self.members[b])
        ax, side = int(self.fw_ax[b]), float(self.fw_side[b])
        u_in = np.zeros(3)
        u_in[ax] = -side
        wall_pos = 0.0 if side < 0 else self.size[ax]
        gap = np.abs(self.pos[mem, ax] - wall_pos) - self.rad[mem]
        root = mem[int(np.argmin(gap))]
        # spanning tree of touching grains, from the grain on the wall
        parent, order, todo = {root: None}, [root], [root]
        while todo:
            k = todo.pop()
            for j in mem:
                if j not in parent and np.linalg.norm(self.pos[j] - self.pos[k]) < (self.rad[j] + self.rad[k]) * (1 + 1e-6):
                    parent[j] = k
                    order.append(j)
                    todo.append(j)
        a = self._a_app_now[None, :] + self._F_pos[mem] / self.mass[mem][:, None]
        idx = {k: i for i, k in enumerate(mem)}
        Msub = {k: self.mass[k] for k in mem}
        Fsub = {k: self.mass[k] * a[idx[k]] for k in mem}
        for k in reversed(order):
            if parent.get(k) is not None:
                Msub[parent[k]] += Msub[k]
                Fsub[parent[k]] = Fsub[parent[k]] + Fsub[k]

        def fails(F, M, u, Fc):
            F_qs = float(F @ u)
            F_sh = M * a_sh * (abs(u[self.shake_axis]) if self.shake_axis is not None else 0.0)
            if F_qs + F_sh <= Fc:
                return None
            if F_qs >= Fc:
                return 0.0, "load"
            sphi = (Fc - F_qs) / F_sh
            return self.vib_A * self.vib_w * math.sqrt(max(0.0, 1.0 - sphi * sphi)), "shake"
        res = fails(Fsub[root], Msub[root], u_in, self.Fc_wall[root])
        if res is not None:                                   # the whole aggregate comes off the wall
            self._free_body(b, u_in * res[0], t_now, root, res[1])
            return
        cut = []
        for k in order[1:]:
            par = parent[k]
            if any(c in cut for c in self._ancestors(k, parent)):
                continue
            u = self.pos[k] - self.pos[par]
            u /= np.linalg.norm(u)
            Fc = self._contact(k, par).p.F_c
            r = fails(Fsub[k], Msub[k], u, Fc)
            if r is not None:
                cut.append(k)
                sub = [j for j in order if j == k or k in self._ancestors(j, parent)]
                new = self._detach(b, [sub])
                # the aggregate moved rigidly with the wall, so the freed part leaves with the wall's velocity
                # (along the shaking axis, toward the side the bond was loaded); a quasi-static break gives it none
                v_free = np.zeros(3)
                if r[1] == "shake":
                    v_free[self.shake_axis] = math.copysign(r[0], u[self.shake_axis])
                for nb in new:
                    self.V[nb] = v_free
                    self._ever_freed.update(int(g) for g in self.members[nb])
                    self.n_events["detach"] += 1
                    if r[1] == "shake":
                        self.n_events["detach_shake"] += 1
                    self.detach_log.append(dict(t=self.env.t0_flight + t_now, grain=k, n=len(sub), how=r[1],
                                                v=float(r[0])))
        if cut:
            self._topology_changed = True

    @staticmethod
    def _ancestors(k, parent):
        out = []
        while parent.get(k) is not None:
            k = parent[k]
            out.append(k)
        return out

    def _contain(self):
        """Safety net: no free grain may end a step outside the chamber (pushes its body back)."""
        act = self.alive & ~self.frozen
        lo = self.rad[:, None] - self.pos
        hi = self.pos + self.rad[:, None] - self.size[None, :]
        pen = np.where(lo > 0, lo, 0.0) - np.where(hi > 0, hi, 0.0)
        pen[~act[self.body]] = 0.0
        worst = float(np.abs(pen).max(initial=0.0))
        if worst > 0:
            self.contain_corrections += 1
            self.contain_max_m = max(self.contain_max_m, worst)
            shift = np.zeros((self.N, 3))
            for k in range(3):
                # largest required push per body along each axis
                np.maximum.at(shift[:, k], self.body, np.where(pen[:, k] > 0, pen[:, k], 0.0))
                neg = np.zeros(self.N)
                np.minimum.at(neg, self.body, np.where(pen[:, k] < 0, pen[:, k], 0.0))
                shift[:, k] += neg
            self.X += shift
            self.pos = self.X[self.body] + self.off

    def step(self, t):
        """Velocity-Verlet step: half kick, collision-resolved drift, forces at the new positions, half kick."""
        dt = self.dt
        self.t = t
        self._step_count += 1
        if self._update_time(t):
            self._build_neighbors()
        a_app = self.env.apparent_accel(t, dt)
        self._a_app_now = a_app
        if self.env.profile is not None:
            self._W_drift = 2 * math.pi * self.env.profile.spin_hz(self.env.t0_flight + t + 0.5 * dt)
        if self._F_pos is None:
            self._F_pos = self._forces_pos(self._mono_vel())
        self._check_detachment(t)
        if (self.alive & ~self.frozen).any():
            self._pair_stale = True          # something can move, collide or charge during this step
            self._charge_stale = self._sc_dirty = True
        act = self._kick(self._F_pos, a_app, 0.5 * dt)

        remaining = dt
        excl = np.zeros(len(self.nlist), bool)
        wexcl = np.zeros((self.N, 3), bool)
        for _ in range(100000):
            if remaining <= 1e-15:
                break
            vm = self._mono_vel()
            win = self._rotation_window(vm, remaining)
            evs = self._events(vm, win, excl, wexcl)
            if not evs:
                self._drift(win, act)
                remaining -= win
                act = np.flatnonzero(self.alive & ~self.frozen)
                continue
            elapsed = 0.0
            touched = set()
            for kev, (t_ev, ev) in enumerate(evs):
                if ev[0] == "pp":
                    i, j = self.nlist[ev[1]]
                    bodies = (int(self.body[i]), int(self.body[j]))
                else:
                    bodies = (int(self.body[ev[1]]),)
                if touched.intersection(bodies):
                    break
                self._drift(t_ev - elapsed, act)
                elapsed = t_ev
                t_now = t + dt - remaining + elapsed
                if ev[0] == "pp":
                    hit = self._resolve_pp(ev[1], t_now)
                    if not hit:
                        excl[ev[1]] = True      # touching but not approaching: skip this pair for now
                else:
                    hit = self._resolve_wall(ev[1], ev[2], ev[3], t_now)
                    if not hit:
                        wexcl[ev[1], ev[2]] = True
                touched.update(int(self.body[g]) for g in ((ev[1],) if ev[0] == "wall" else self.nlist[ev[1]]))
                touched.update(bodies)
                if hit:
                    # bodies whose velocity just changed may now approach partners they were excluded from
                    changed = np.zeros(self.N, bool)
                    for b in touched:
                        changed[self.members[b]] = True
                    excl &= ~(changed[self.nlist[:, 0]] | changed[self.nlist[:, 1]])
                    wexcl[changed] = False
                if self._topology_changed or any(self.nmono[b] > 1 for b in bodies):
                    self._topology_changed = False
                    break   # an aggregate changed: recompute all predictions
                # re-predict the changed bodies: stop if they now hit something before the next listed event
                mask = np.zeros(self.N, bool)
                for b in touched:
                    mask[self.members[b]] = True
                vm_new = self._mono_vel()
                v_new = float(np.linalg.norm(vm_new[mask], axis=1).max())
                if v_new + 2 * self.vib_A * self.vib_w > self._nl_vlim:
                    # a bounce made a grain faster than the neighbour list allows: rebuild it now
                    self._build_neighbors(vm_new)
                    excl = np.zeros(len(self.nlist), bool)
                    wexcl = np.zeros((self.N, 3), bool)
                    self.n_events["midstep_neighbor_rebuild"] += 1
                    break
                if self._win_vlim is not None and v_new > self._win_vlim:
                    break   # faster than the drift window assumed: recompute the window and all predictions
                new = self._events(vm_new, win - elapsed, excl, wexcl, subset=mask)
                t_next = evs[kev + 1][0] - elapsed if kev + 1 < len(evs) else win - elapsed
                if new and new[0][0] < t_next:
                    break
            remaining -= elapsed
            act = np.flatnonzero(self.alive & ~self.frozen)
        else:
            raise RuntimeError(f"event loop did not finish the step at t = {t:.6f} s (100000 events)")
        self._contain()
        # forces at the new positions (velocity Verlet), then the second half kick. With a time-dependent spin
        # the end-of-step forces and the second half kick use Omega(t + dt) (keeps the scheme second order).
        self.t = t + dt
        if self.env.profile is not None and self._update_time(t + dt):
            self._build_neighbors()
        vm = self._mono_vel()
        if self._needs_rebuild(vm):
            self._build_neighbors(vm)
        self._F_pos = self._forces_pos(vm)
        self._kick(self._F_pos, a_app, 0.5 * dt, brownian=True)

    def _dt_at(self, t):
        """Variable time step (flight-profile runs, and any run with release windows). With a profile and free
        grains: outside the microgravity window a step scaled to the load (dt_launch_s at 25 g, longer for smaller
        loads, in powers of two), the base step inside it; never more than 0.02 rad of payload roll per step, and
        shorter steps for fast grains. Every release window or shock starts exactly at a step
        start and is resolved with >= 20 steps; the last step ends exactly at the end of the run."""
        T = self.env.t0_flight + t
        prof = self.env.profile
        dt = self.dt_base
        if prof is not None and (self.alive & ~self.frozen).any():
            # free grains: short steps while boosting / spinning fast (while every grain is wall-bound nothing
            # moves, and the base step is enough to follow the loads)
            if T < prof.ug_start_s or T >= prof.ug_end_s:     # boost, coast, re-entry, parachute, landing
                # between kicks a grain drifts on a straight segment; under a load a its true path bends away by
                # a dt^2 / 8. Keep that at the value dt_launch allows at the 25 g boost load (~8 um at 0.5 ms):
                # dt = dt_launch sqrt(25 g / a), e.g. 2.5 ms under the parachute (1 g), 0.9 ms at an 8 g re-entry.
                # (in powers of two of dt_launch, so the neighbour list is rebuilt only when the load changes 4x)
                a_g = max(float(np.linalg.norm(prof.apparent_accel(T))), 1e-6) / G0
                dt = min(dt, self.dt_launch * 2.0 ** math.floor(math.log2(math.sqrt(25.0 / a_g))))
            W = 2 * math.pi * max(abs(prof.spin_hz(T)), abs(prof.spin_hz(T + dt)))
            if W > 0:
                dt = min(dt, 0.02 / W)
            # fast grains (just after a kick): shorter steps keep the neighbour-list skin (3 v dt) at its base
            # size; otherwise nearly every pair in the tubule becomes a candidate. Collisions are found as events
            # either way; in microgravity this changes only the cost (under launch loads it also lowers the
            # resting-contact cut-off a_n dt, a numerical threshold). Powers of two of the base step, so the list
            # is rebuilt only when the speed changes by 2x.
            vb = self._speed_bound(self._mono_vel())
            if vb > 0 and 3 * vb * dt > self.skin:
                k = math.ceil(math.log2(3 * vb * self.dt_base / self.skin))
                dt = min(dt, max(self.dt_base / 2 ** k, 1e-5))
        windows = [(r["start_s"], r["end_s"]) for r in self.release]
        if prof is not None:
            windows += prof.event_windows()                 # impulses, parachute opening, splashdown
        eps = 1e-12
        for a, b in windows:
            if T < a - eps and T + dt > a:                  # never step over the start of a window
                dt = min(dt, a - T)
        for a, b in windows:
            if a - eps <= T < b - eps:                       # inside a window: >= 20 steps across it
                dt = min(dt, max((b - a) / 20, 1e-7))
        return min(dt, self.duration - t) if self.duration - t > 1e-12 else dt

    def _record(self, t):
        self.frame_t.append(t)
        self.frames.append(self.pos.astype(np.float32).copy())
        self.frame_bodies.append(self.body.astype(np.int32).copy())
        self.frame_frozen.append(self.frozen[self.body].copy())
        spin_b = np.linalg.norm(np.einsum("bij,bj->bi", self.Iinv, self.L), axis=1) / (2 * math.pi)
        self.frame_spin.append(spin_b[self.body].astype(np.float32))
        live = self.alive & ~self.frozen
        sizes = self.nmono[live]
        speeds = np.linalg.norm(self.V[live], axis=1)
        ke = 0.5 * (self.M[live] * speeds ** 2).sum()
        self.timeseries.append(dict(
            t=t, free_bodies=int(live.sum()), free_monomers=int(sizes.sum()),
            wall_monomers=int(self.nmono[self.alive & self.frozen].sum()),
            mean_aggregate_size=float(sizes.mean()) if len(sizes) else 0.0,
            max_aggregate_size=int(self.nmono[self.alive].max()),
            mean_speed=float(speeds.mean()) if len(speeds) else 0.0,
            mean_spin_hz=float(spin_b[live].mean()) if live.any() else 0.0,
            max_spin_hz=float(spin_b[live].max()) if live.any() else 0.0,
            kinetic_energy=float(ke), **dict(self.n_events),
            mean_abs_charge_e=float(np.abs(self.q).mean() / Q_E),
            net_charge_e=float(self.q.sum() / Q_E)))

    def run(self, progress=True, callback=None, callback_interval_s=0.5):
        """Run the whole window. callback(t, duration, n_events) is called every callback_interval_s of
        wall-clock time (used by the browser app for its progress bar)."""
        self._build_neighbors()
        t, next_out = 0.0, 0.0
        n_steps = int(round(self.duration / self.dt))
        prof = self.env.profile
        variable = prof is not None or bool(self.release)
        if variable:
            n_steps = 10 ** 9          # variable step: loop until the end of the window
        t0 = time.time()
        last = last_cb = t0
        for s in range(n_steps):
            if variable:
                if t >= self.duration - 1e-12:
                    break
                dt = self._dt_at(t)
                if abs(dt / self.dt - 1) > 1e-9:
                    self.dt = dt
                    self._build_neighbors()
            if t >= next_out - 1e-12:
                self._record(t)
                next_out += self.out_dt
            self.step(t)
            t = t + self.dt if variable else (s + 1) * self.dt
            if callback is not None and time.time() - last_cb > callback_interval_s:
                last_cb = time.time()
                callback(t, self.duration, dict(self.n_events))
            if progress and time.time() - last > 2.0:
                last = time.time()
                ev = self.n_events
                print(f"  [{self.name}] t={t:7.2f}/{self.duration:.0f}s  sticks={ev['stick']} "
                      f"bounces={ev['bounce']} wall={ev['wall_stick']}+{ev['wall_bounce']}  "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
        self._record(t)
        self.wall_time = time.time() - t0
        return self

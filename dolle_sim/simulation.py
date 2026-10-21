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

Known simplifications (see README): grains are spheres; aggregates are rigid
between impacts (no rolling restructuring); no gas-mediated interaction
between grains (negligible at Kn >> 1); photophoresis without shadowing;
Coriolis torque on the internal rotation of aggregates is neglected.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial import cKDTree

from .materials import get_material, pair_props, sphere_mass
from .contact import JKRContact
from .environment import Environment
from .electrostatics import SpaceChargeSolver, short_range_kernel

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
        Om = float(np.linalg.norm(self.env.omega))
        if Om > 0:
            self.dt = min(self.dt, 0.02 / Om)
        self.duration = float(cfg.get("duration_s", self.env.window_s))
        self.out_dt = 1.0 / float(cfg.get("output_fps", 20))
        self._step_count = 0
        self.t = 0.0

        self._init_particles(chamber.get("particles", []), ov)
        self._contacts: dict = {}
        self.collisions: list[dict] = []
        self.frames: list[np.ndarray] = []
        self.frame_bodies: list[np.ndarray] = []
        self.frame_t: list[float] = []
        self.frame_frozen: list[np.ndarray] = []
        self.frame_spin: list[np.ndarray] = []
        self.timeseries: list[dict] = []
        self.n_events = {"stick": 0, "bounce": 0, "wall_stick": 0, "wall_bounce": 0,
                         "erosion": 0, "shatter": 0}
        self._topology_changed = False
        self._F_pos = None
        self._F_pair = None
        self.sc = None
        if self.f_space and self.f_coulomb and (self.q.any() or self.charging):
            self.sc = SpaceChargeSolver(self.size, self.coulomb_cut / 3.5)
            # short-range pair sum must reach ~3.5 sigma of the (possibly widened) Ewald split
            self.coulomb_cut = max(self.coulomb_cut, 3.5 * self.sc.sigma)

    # ------------------------------------------------------------ setup
    def _init_particles(self, groups, overrides):
        mats, mat_idx, rad, q, vel, pos, group_idx, spin, adh = [], [], [], [], [], [], [], [], []
        self.groups = []
        for gi, g in enumerate(groups):
            m = get_material(g["material"], overrides)
            if m.name not in [x.name for x in mats]:
                mats.append(m)
            mi = [x.name for x in mats].index(m.name)
            n = int(g["count"])
            r0 = float(g.get("radius_m", 50e-6))
            rs = float(g.get("radius_rel_sigma", 0.0))
            r = r0 * np.clip(1 + rs * self.rng.standard_normal(n), 0.5, 1.5)
            reg = g.get("region_m", {})
            margin = 2 * r0
            center = np.asarray(reg.get("center", self.size / 2), float)
            span = np.asarray(reg.get("size", self.size - 2 * margin), float)
            p = center + (self.rng.random((n, 3)) - 0.5) * span
            p = np.clip(p, margin, self.size - margin)
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

    def _remove_overlaps(self):
        for _ in range(50):
            pairs = cKDTree(self.pos).query_pairs(2.05 * self.rad.max(), output_type="ndarray")
            if len(pairs) == 0:
                return
            d = np.linalg.norm(self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]], axis=1)
            bad = pairs[d < 1.05 * (self.rad[pairs[:, 0]] + self.rad[pairs[:, 1]]), 1]
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

    def _drift(self, h, act):
        if h <= 0:
            return
        self.X[act] += self.V[act] * h
        multi = act[self.nmono[act] > 1]
        if len(multi):
            self._rotate(multi, h)
        self.pos = self.X[self.body] + self.off

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
        if np.linalg.norm(Pt) <= self.mu_f * Pn or float(np.linalg.norm(vt)) < 1e-15:
            return P
        tv = vt / np.linalg.norm(vt)
        d = n + self.mu_f * tv
        Pn = (1 + e_n) * vn / float(n @ K @ d)
        return Pn * d

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
        if Om.any():
            r_ax = self.pos + self.axis_offset
            F -= m * _crossN(Om[None, :], _crossN(Om[None, :], r_ax))          # centrifugal
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
            if self.f_coulomb and self.sc is not None:
                if self.sc.E is None or self._step_count % self.sc_every == 0:
                    self.sc.solve(self.pos, self.q)
                F += q * self.sc.field_at(self.pos)
            if self.f_image:
                # short-range image charge -beta q behind each nearby wall (the long-range part of the
                # images of a grounded chamber is already in the space-charge mesh)
                wc = self.wall_cand
                if len(wc):
                    beta = self.wall_mat.image_factor
                    p = self.pos[wc]
                    h_lo = np.maximum(p, self.rad[wc][:, None])
                    h_hi = np.maximum(self.size[None, :] - p, self.rad[wc][:, None])
                    k = K_E * beta * self.q[wc] ** 2
                    if self.sc is not None:
                        g_lo, g_hi = short_range_kernel(2 * h_lo, self.sc.sigma), short_range_kernel(2 * h_hi, self.sc.sigma)
                    else:
                        g_lo, g_hi = 1 / (2 * h_lo) ** 2, 1 / (2 * h_hi) ** 2
                    F[wc] += k[:, None] * (g_hi - g_lo)
        if len(self.lrlist) and self._long_range():
            if self._F_pair is None or self._step_count % self.lr_every == 0:
                self._F_pair = self._pair_forces()
            F += self._F_pair
        return F

    def _pair_forces(self):
        """Short-range Coulomb, polarisation and mutual-gravity forces between nearby grains."""
        F = np.zeros((self.N, 3))
        charged = self.q.any()
        i, j = self.lrlist[:, 0], self.lrlist[:, 1]
        sel = self.body[i] != self.body[j]
        i, j = i[sel], j[sel]
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
        hold = self.dt * self.lr_every
        cap = mu * d / (2 * hold ** 2)
        fmag = np.clip(fmag, -cap, cap)
        f = (fmag * ok / d)[:, None] * r
        for k in range(3):
            F[:, k] += np.bincount(i, f[:, k], self.N) - np.bincount(j, f[:, k], self.N)
        return F

    # ------------------------------------------------------------ neighbour lists
    def _build_neighbors(self):
        c_cut = 2 * self.rad.max() + self.skin
        lr = self._long_range()
        cut = max(c_cut, self.coulomb_cut + self.skin if lr else 0)
        pairs = cKDTree(self.pos).query_pairs(cut, output_type="ndarray")
        if len(pairs):
            d = np.linalg.norm(self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]], axis=1)
            self.nlist = pairs[d < c_cut]                 # contact candidates (collision detection)
            self.lrlist = pairs if lr else pairs[:0]      # long-range force pairs
        else:
            self.nlist = self.lrlist = pairs
        self.pos_at_build = self.pos.copy()
        self._F_pair = None
        # grains close enough to a wall to possibly reach it before the next rebuild
        gap = np.minimum(self.pos, self.size[None, :] - self.pos).min(1) - self.rad
        self.wall_cand = np.flatnonzero(gap < self.skin)

    def _needs_rebuild(self, vm):
        # no pair can close the gap 'skin' before the next check: 2 (max displacement + v_max dt) < skin
        disp = np.linalg.norm(self.pos - self.pos_at_build, axis=1).max()
        vmax = np.linalg.norm(vm, axis=1).max() if len(vm) else 0.0
        return 2 * (disp + vmax * self.dt) > self.skin

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
                   x=cpt[0], y=cpt[1], z=cpt[2], r_sum=self.rad[i] + self.rad[j], E_coll=E_coll, fragments=0)
        self.collisions.append(rec)
        if res.stuck:
            self.n_events["stick"] += 1
            self._merge(A, B)
        else:
            self.n_events["bounce"] += 1
            P = self._impulse(K, vrel, n, res.e_n)
            self._apply_impulse(A, rA, -P)
            self._apply_impulse(B, rB, P)
            if self.fragmentation and (nA > 1 or nB > 1):
                rec["fragments"] = self._fragment([(A, i), (B, j)], E_coll, res.e_n, t_now)
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
                   x=cpt[0], y=cpt[1], z=cpt[2], r_sum=self.rad[m], E_coll=E_coll, fragments=0)
        self.collisions.append(rec)
        if stuck:
            self.n_events["wall_stick"] += 1
            self.frozen[A] = True
            self.fw_ax[A], self.fw_side[A] = ax, side
            self.V[A] = 0
            self.L[A] = 0
        else:
            self.n_events["wall_bounce"] += 1
            P = self._impulse(K, vrel, n, e_n)
            self._apply_impulse(A, rA, -P)
            if self.fragmentation and self.nmono[A] > 1:
                rec["fragments"] = self._fragment([(A, m)], E_coll, e_n, t_now)
        return True

    # ------------------------------------------------------------ fragmentation
    def _bond_energy(self, b, k):
        """Break energy of the grain-grain bond holding grain k in aggregate b (nearest member)."""
        mem = [x for x in self.members[b] if x != k]
        d = np.linalg.norm(self.pos[mem] - self.pos[k], axis=1)
        return self._contact(k, mem[int(np.argmin(d))]).p.E_break

    def _fragment(self, hits, E_coll, e_n, t_now):
        """Dominik & Tielens (1997): erosion above ~3 N_c E_break, shattering above ~10 N_c E_break."""
        agg = [(b, k) for b, k in hits if self.nmono[b] > 1 and not self.frozen[b]]
        if not agg:
            return 0
        N_c = sum(int(self.nmono[b]) - 1 for b, _ in agg)
        E_b = float(np.mean([self._bond_energy(b, k) for b, k in agg]))
        if E_coll > 10 * N_c * E_b:
            mode, self_detach = "shatter", {b: [x for x in self.members[b]][1:] for b, _ in agg}
        elif E_coll > 3 * N_c * E_b:
            mode, self_detach = "erosion", {b: [k] for b, k in agg}
        else:
            return 0
        n_broken = sum(len(v) for v in self_detach.values())
        # energy above the bond-breaking cost; the elastic fraction e_n^2 becomes separation motion
        E_sep = max(E_coll - n_broken * E_b, 0.0) * e_n ** 2
        out = 0
        for b, ks in self_detach.items():
            members_before = list(self.members[b])
            X0 = self.X[b].copy()
            new = self._split(b, ks)
            out += len(ks)
            self.last_split[members_before] = t_now
            parts = [b] + new
            self._separate(parts, X0, E_sep / len(self_detach))
        self.n_events[mode] += 1
        self._topology_changed = True
        return out

    def _separate(self, parts, X0, E_sep):
        """Give fragments radial separation velocities from the old COM, conserving momentum and
        angular momentum (radial kicks about the COM carry no angular momentum)."""
        if E_sep <= 0 or len(parts) < 2:
            return
        m = self.M[parts]
        u = self.X[parts] - X0
        nrm = np.linalg.norm(u, axis=1)
        nrm[nrm == 0] = 1.0
        u = u / nrm[:, None]
        u -= (m[:, None] * u).sum(0) / m.sum()
        ke = 0.5 * float((m * (u * u).sum(1)).sum())
        if ke <= 0:
            return
        s = math.sqrt(E_sep / ke)
        self.V[parts] += s * u

    def _split(self, b, ks):
        """Detach grains ks from body b as single grains; momentum and angular momentum are conserved.
        Returns the new body ids."""
        w = self._w(b)
        P = self.M[b] * self.V[b]
        Lo = self.L[b] + self.M[b] * _cross3(self.X[b], self.V[b])
        free = list(np.flatnonzero(~self.alive)[:len(ks)])
        for k, nb in zip(ks, free):
            I0 = self.I0[k]
            v = self.V[b] + _cross3(w, self.off[k])
            P = P - self.mass[k] * v
            Lo = Lo - (I0 * w + self.mass[k] * _cross3(self.pos[k], v))
            self.alive[nb], self.frozen[nb], self.nmono[nb] = True, False, 1
            self.fw_ax[nb] = -1
            self.members[nb] = [k]
            self.members[b].remove(k)
            self.body[k] = nb
            self.X[nb], self.V[nb], self.M[nb] = self.pos[k], v, self.mass[k]
            self.L[nb] = I0 * w
            self.Iinv[nb] = np.eye(3) / I0
            self.Q[nb], self.Ip[nb] = np.eye(3), I0
            self.Iiso[nb], self.brot[nb], self.bdrag[nb] = I0, self.rotc[k], self.dragc[k]
            self.off[k] = 0.0
            self.offb[k] = 0.0
        mem = self.members[b]
        M = self.mass[mem].sum()
        X = (self.mass[mem, None] * self.pos[mem]).sum(0) / M
        self.M[b], self.X[b], self.V[b] = M, X, P / M
        self.L[b] = Lo - M * _cross3(X, P / M)
        self.off[mem] = self.pos[mem] - X
        I = np.zeros((3, 3))
        for k in mem:
            r = self.off[k]
            I += (self.I0[k] + self.mass[k] * (r @ r)) * np.eye(3) - self.mass[k] * np.outer(r, r)
        self.nmono[b] = len(mem)
        self._set_principal(b, I)
        self.bdrag[b] = self.dragc[mem].sum()
        self.brot[b] = self._rot_drag(mem)
        return free[:len(ks)]

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
        Fb = np.empty((self.N, 3))
        for k in range(3):
            Fb[:, k] = np.bincount(self.body, F[:, k], self.N)
        act = np.flatnonzero(self.alive & ~self.frozen)
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
            T = _crossN(self.off, F)
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

    def _contain(self):
        """Safety net: no free grain may end a step outside the chamber (pushes its body back)."""
        act = self.alive & ~self.frozen
        lo = self.rad[:, None] - self.pos
        hi = self.pos + self.rad[:, None] - self.size[None, :]
        pen = np.where(lo > 0, lo, 0.0) - np.where(hi > 0, hi, 0.0)
        pen[~act[self.body]] = 0.0
        if np.abs(pen).max(initial=0.0) > 0:
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
        a_app = self.env.apparent_accel(t, dt)
        if self._F_pos is None:
            self._F_pos = self._forces_pos(self._mono_vel())
        act = self._kick(self._F_pos, a_app, 0.5 * dt)

        remaining = dt
        excl = np.zeros(len(self.nlist), bool)
        wexcl = np.zeros((self.N, 3), bool)
        for _ in range(100000):
            vm = self._mono_vel()
            evs = self._events(vm, remaining, excl, wexcl)
            if not evs:
                self._drift(remaining, act)
                break
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
                    if not self._resolve_pp(ev[1], t_now):
                        excl[ev[1]] = True      # grazing contact (contact point already separating)
                elif not self._resolve_wall(ev[1], ev[2], ev[3], t_now):
                    wexcl[ev[1], ev[2]] = True  # grazing contact with this wall
                touched.update(int(self.body[g]) for g in ((ev[1],) if ev[0] == "wall" else self.nlist[ev[1]]))
                touched.update(bodies)
                if self._topology_changed or any(self.nmono[b] > 1 for b in bodies):
                    self._topology_changed = False
                    break   # an aggregate changed: recompute all predictions
                # re-predict the changed bodies: stop if they now hit something before the next listed event
                mask = np.zeros(self.N, bool)
                for b in touched:
                    mask[self.members[b]] = True
                new = self._events(self._mono_vel(), remaining - elapsed, excl, wexcl, subset=mask)
                t_next = evs[kev + 1][0] - elapsed if kev + 1 < len(evs) else remaining - elapsed
                if new and new[0][0] < t_next:
                    break
            remaining -= elapsed
            act = np.flatnonzero(self.alive & ~self.frozen)
        self._contain()
        # forces at the new positions (velocity Verlet), then the second half kick
        self.t = t + dt
        vm = self._mono_vel()
        if self._needs_rebuild(vm):
            self._build_neighbors()
        self._F_pos = self._forces_pos(vm)
        self._kick(self._F_pos, a_app, 0.5 * dt, brownian=True)

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
        t0 = time.time()
        last = last_cb = t0
        for s in range(n_steps):
            if t >= next_out - 1e-12:
                self._record(t)
                next_out += self.out_dt
            self.step(t)
            t = (s + 1) * self.dt
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

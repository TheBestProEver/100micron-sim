"""Multi-particle microgravity dust simulation for one experiment chamber.

Method (hybrid event-driven / DEM):
* Grains ("monomers") belong to rigid bodies. A body is a single grain or an
  aggregate of grains that have stuck together; bodies translate and rotate
  (angular momentum is conserved, inertia tensor tracked in the world frame).
* Each global step (~1-4 ms) applies external forces exactly-integrated with gas
  drag: apparent (residual) acceleration of the rocket frame, g-jitter,
  centrifugal and Coriolis forces from payload spin, gravity gradient (tidal)
  and mutual gravity, Coulomb forces between charged grains, charge-induced
  dipole (polarisation) attraction, image-charge attraction to the walls, an
  external electric field and the Lorentz force q v x B.
* Every grain and aggregate spins: spin is set at launch, exchanged with
  translation by friction and off-centre impacts, conserved through sticking
  (aggregates inherit the angular momentum) and damped by gas torque.
* Bodies then drift ballistically. Continuous collision detection finds the
  exact time of every grain-grain and grain-wall contact inside the step; each
  impact is resolved by the JKR contact integrator (contact.py) using the
  correct effective mass of the two rigid bodies at the contact point.
  Outcome: stick (bodies merge rigidly, or freeze onto the wall) or bounce
  (normal impulse with the DEM restitution + Coulomb-limited friction).
* Optional contact (tribo-)charging: charge transferred on every contact,
  proportional to the work-function difference and the contact area.
* Gas-kinetic forces: thermophoresis from chamber temperature gradients,
  photophoresis and radiation pressure from the camera illumination, and
  Brownian (thermal) kicks consistent with the drag (fluctuation-dissipation).
* Grain-to-grain scatter of the effective surface energy (roughness), wall
  coatings/roughness (wall adhesion factor) and vibrating walls that put energy
  back into the dust so grains keep bouncing instead of settling on the walls.
* Aggregate erosion and catastrophic fragmentation using the energy thresholds
  of Dominik & Tielens (1997): loss of a grain above ~3 E_break, shattering
  above ~10 N_contacts E_break.

Known simplifications (see README): grains are spheres; aggregates are rigid
between impacts (no rolling restructuring); no gas-mediated interaction
between grains (negligible at Kn >> 1); one-way photophoresis (no shadowing).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial import cKDTree

from dataclasses import replace

from .materials import get_material, pair_props, sphere_mass
from .contact import JKRContact
from .environment import Environment

K_E = 8.9875517923e9
G_N = 6.67430e-11
Q_E = 1.602176634e-19
EPS0 = 8.8541878128e-12
K_B = 1.380649e-23
C_LIGHT = 2.99792458e8


def _cross3(a, b):
    """Cross product of two 3-vectors (much cheaper than np.cross for single vectors)."""
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


def _crossN(a, b):
    """Row-wise cross product of (N,3) arrays (either may be a broadcastable (1,3) row)."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=-1)


def _rodrigues(w, h):
    """Rotation matrices (k,3,3) for rotation vectors w*h."""
    th = np.linalg.norm(w, axis=1) * h
    k = np.zeros_like(w)
    nz = th > 0
    k[nz] = w[nz] / np.linalg.norm(w[nz], axis=1)[:, None]
    K = np.zeros((len(w), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


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
        # wall roughness / coating: scales the work of adhesion of grain-wall contacts
        self.wall_adh = float(chamber.get("wall_adhesion_factor", 1.0))
        vib = chamber.get("wall_vibration", {}) or {}
        self.vib_A = float(vib.get("amplitude_m", 0.0))
        self.vib_w = 2 * math.pi * float(vib.get("frequency_hz", 0.0))
        self.vib_axes = set(vib.get("axes", [0, 1, 2]))
        self.vib_phase = self.rng.random((3, 2)) * 2 * math.pi
        self.skin = float(cc.get("neighbor_skin_m", 1e-3))
        self.lr_every = int(cfg.get("forces", {}).get("long_range_update_every", 5))
        self._step_count = 0
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
        self.coulomb_cut = float(fc.get("long_range_cutoff_m", cc.get("coulomb_cutoff_m", 3e-3)))
        # chamber corner relative to the rocket centre of mass (for the gravity gradient)
        cg = chamber.get("center_from_cg_m", chamber.get("center_from_spin_axis_m", [0, 0, 0]))
        self.cg_offset = np.asarray(cg, float) - self.size / 2

        self.dt = float(cfg.get("dt_s", 1e-3))
        Om = float(np.linalg.norm(self.env.omega))
        if Om > 0:
            self.dt = min(self.dt, 0.02 / Om)
        self.duration = float(cfg.get("duration_s", self.env.window_s))
        self.out_dt = 1.0 / float(cfg.get("output_fps", 20))

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
        self.dragc = self.env.drag_coeff(self.rad)
        self.rotc = self.env.rot_drag_coeff(self.rad)
        self.polar = np.array([self.mats[k].polarizability for k in self.mat])
        self.adh = np.concatenate(adh)
        self.F_static = self._static_forces()
        self._remove_overlaps()

        # bodies: initially one per monomer
        self.body = np.arange(N)
        self.members = [[k] for k in range(N)]
        self.off = np.zeros((N, 3))
        self.X = self.pos.copy()
        self.V = np.concatenate(vel)
        self.M = self.mass.copy()
        I0 = 0.4 * self.mass * self.rad ** 2
        self.L = np.concatenate(spin) * I0[:, None]
        self.Iinv = np.zeros((N, 3, 3))
        self.Iinv[:, [0, 1, 2], [0, 1, 2]] = (1.0 / I0)[:, None]
        self.Iiso = I0.copy()               # mean principal moment, for gas spin-down
        self.brot = self.rotc.copy()        # rotational drag coefficient of each body
        self.alive = np.ones(N, bool)
        self.frozen = np.zeros(N, bool)
        self.nmono = np.ones(N, int)
        self.bdrag = self.dragc.copy()

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
            c = self._contacts[key] = JKRContact(pair_props(self.mats[mi], ri, m2, rj, adhesion_factor=f_r))
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
                # free-molecular photophoresis (Rohatschek 1995), J1 ~ absorptivity/2; pushes grains
                # away from the light (factor ~2 uncertain; assumes conduction-limited grains)
                F += (math.pi / 6) * (0.5 * absorb) * env.P * env.light_I * R ** 3 / (env.T * kp) * d[None, :]
            if self.f_radp:
                F += env.light_I * math.pi * R ** 2 / C_LIGHT * d[None, :]
        return F

    # ------------------------------------------------------------ helpers
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

    def _mono_vel(self, W):
        vm = self.V[self.body]
        mm = np.flatnonzero(self.nmono[self.body] > 1)
        if len(mm):
            vm[mm] += _crossN(W[self.body[mm]], self.off[mm])
        return vm

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
        self._F_lr = None
        # grains close enough to a wall to possibly reach it before the next rebuild
        gap = np.minimum(self.pos, self.size[None, :] - self.pos).min(1) - self.rad
        self.wall_cand = np.flatnonzero(gap < self.skin)

    def _drift(self, h, W, act):
        if h <= 0:
            return
        self.X[act] += self.V[act] * h
        multi = act[(self.nmono[act] > 1) & (np.abs(W[act]).sum(1) > 0)]
        if len(multi):
            R = _rodrigues(W[multi], h)
            local = np.full(self.N, -1)
            local[multi] = np.arange(len(multi))
            mm = np.flatnonzero(local[self.body] >= 0)
            self.off[mm] = np.einsum("kij,kj->ki", R[local[self.body[mm]]], self.off[mm])
            self.Iinv[multi] = R @ self.Iinv[multi] @ np.transpose(R, (0, 2, 1))
        self.pos = self.X[self.body] + self.off

    # ------------------------------------------------------------ forces
    def _long_range(self):
        charged = self.q.any() and (self.f_coulomb or self.f_polar)
        return charged or self.f_selfgrav

    def _forces(self, a_app, vm, reuse_pairs=False):
        """Force on every grain [N] (chamber frame)."""
        m = self.mass[:, None]
        F = m * a_app[None, :] + self.F_static
        Om = self.env.omega
        if Om.any():
            r_ax = self.pos + self.axis_offset
            F += m * (-2 * _crossN(Om[None, :], vm) - _crossN(Om[None, :], _crossN(Om[None, :], r_ax)))
        if self.env.gravity_gradient:
            F += m * self.env.tidal_accel(self.pos + self.cg_offset)
        charged = self.q.any()
        if charged:
            q = self.q[:, None]
            if self.f_efield and self.env.E_field.any():
                F += q * self.env.E_field[None, :]
            if self.f_lorentz and self.env.B_field.any():
                F += q * _crossN(vm, self.env.B_field[None, :])
            if self.f_image:
                # image charge -beta*q behind each nearby wall: attraction k beta q^2 / (2h)^2
                wc = self.wall_cand
                if len(wc):
                    beta = self.wall_mat.image_factor
                    p = self.pos[wc]
                    h_lo = np.maximum(p, self.rad[wc][:, None])
                    h_hi = np.maximum(self.size[None, :] - p, self.rad[wc][:, None])
                    k = K_E * beta * self.q[wc] ** 2
                    F[wc] += k[:, None] * (1 / (2 * h_hi) ** 2 - 1 / (2 * h_lo) ** 2)
        if len(self.lrlist) and self._long_range():
            if self._F_lr is None or (self._step_count % self.lr_every == 0 and not reuse_pairs):
                self._F_lr = self._pair_forces()
            F += self._F_lr
        return F

    def _pair_forces(self):
        """Coulomb, polarisation and mutual-gravity forces between nearby grains."""
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
            fmag += K_E * self.q[i] * self.q[j] / d ** 2
        if charged and self.f_polar:
            # charge-induced dipole: F = -2 k q^2 beta R^3 / d^5 on the polarised partner
            fmag -= 2 * K_E * (self.q[i] ** 2 * self.polar[j] * self.rad[j] ** 3
                               + self.q[j] ** 2 * self.polar[i] * self.rad[i] ** 3) / d ** 5
        if self.f_selfgrav:
            fmag -= G_N * self.mass[i] * self.mass[j] / d ** 2
        # numerical limiter: within one step a pair force may not move the grains by more
        # than ~1/4 of their separation (prevents overshoot for close, strongly-attracting
        # pairs; the true infall is then resolved by the contact model at impact)
        mu = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
        cap = mu * d / (2 * self.dt ** 2)
        fmag = np.clip(fmag, -cap, cap)
        f = (fmag * ok / d)[:, None] * r
        for k in range(3):
            F[:, k] += np.bincount(i, f[:, k], self.N) - np.bincount(j, f[:, k], self.N)
        return F

    # ------------------------------------------------------------ events
    def _events(self, vm, remaining, excl, wexcl):
        """All contact events predicted within `remaining`, sorted by time."""
        evs = []
        if len(self.nlist):
            i, j = self.nlist[:, 0], self.nlist[:, 1]
            bi, bj = self.body[i], self.body[j]
            sel = (bi != bj) & ~(self.frozen[bi] & self.frozen[bj]) & ~excl
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
        wc = wc[~self.frozen[self.body[wc]] & ~wexcl[wc]]
        if len(wc):
            p, v, r = self.pos[wc], vm[wc], self.rad[wc][:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                t_lo = np.where(v < 0, np.maximum(p - r, 0) / -v, np.inf)
                t_hi = np.where(v > 0, np.maximum(self.size[None, :] - r - p, 0) / v, np.inf)
            tw = np.minimum(t_lo, t_hi)
            for k, ax in zip(*np.nonzero(tw <= remaining)):
                side = -1.0 if t_lo[k, ax] <= t_hi[k, ax] else 1.0
                evs.append((float(tw[k, ax]), ("wall", int(wc[k]), int(ax), side)))
        evs.sort(key=lambda e: e[0])
        return evs

    def _k_body(self, b, r, n):
        """Inverse effective mass contribution of body b at lever arm r along direction n."""
        if self.frozen[b]:
            return 0.0
        rn = _cross3(r, n)
        return 1.0 / self.M[b] + float(n @ _cross3(self.Iinv[b] @ rn, r))

    def _apply_impulse(self, b, r, P):
        if self.frozen[b]:
            return
        self.V[b] += P / self.M[b]
        self.L[b] += _cross3(r, P)

    def _wall_velocity(self, ax, side, t):
        """Inward (into the chamber) velocity of a vibrating wall."""
        if self.vib_A <= 0 or ax not in self.vib_axes:
            return 0.0
        k = 0 if side < 0 else 1
        return self.vib_A * self.vib_w * math.cos(self.vib_w * t + self.vib_phase[ax, k])

    def _record_collision(self, **kw):
        self.collisions.append(kw)

    def _resolve_pp(self, k, t_now):
        i, j = (int(x) for x in self.nlist[k])
        A, B = self.body[i], self.body[j]
        rvec = self.pos[j] - self.pos[i]
        n = rvec / np.linalg.norm(rvec)
        cpt = self.pos[i] + n * self.rad[i]
        rA, rB = cpt - self.X[A], cpt - self.X[B]
        wA, wB = self._w(A), self._w(B)
        vA = self.V[A] + _cross3(wA, rA) if not self.frozen[A] else np.zeros(3)
        vB = self.V[B] + _cross3(wB, rB) if not self.frozen[B] else np.zeros(3)
        vrel = vA - vB
        vn = float(vrel @ n)
        if vn <= 0:
            return False
        kn = self._k_body(A, rA, n) + self._k_body(B, rB, n)
        m_eff = 1.0 / kn
        cm = self._contact(i, j)
        res = cm.resolve(vn, m_eff)
        vt_vec = vrel - vn * n
        vt = float(np.linalg.norm(vt_vec))
        if self.charging:
            self._charge(i, j, res.max_contact_radius)
        # translational collision energy of the two bodies (for fragmentation)
        dV = (self.V[A] if not self.frozen[A] else 0) - (self.V[B] if not self.frozen[B] else 0)
        if self.frozen[A] or self.frozen[B]:
            mu = self.M[B] if self.frozen[A] else self.M[A]
        else:
            mu = self.M[A] * self.M[B] / (self.M[A] + self.M[B])
        E_coll = 0.5 * mu * float(dV @ dV)
        nA, nB = int(self.nmono[A]), int(self.nmono[B])
        rec = dict(t=t_now, kind="pp", i=i, j=j, mat_i=self.mats[self.mat[i]].name,
                   mat_j=self.mats[self.mat[j]].name, n_i=nA, n_j=nB,
                   v_n=vn, v_t=vt, v_rel=float(np.linalg.norm(vrel)), m_eff=m_eff,
                   outcome="stick" if res.stuck else "bounce",
                   e_n=res.e_n, yielded=res.yielded, v_stick_est=cm.v_stick_analytic(m_eff),
                   adhesion_factor=math.sqrt(self.adh[i] * self.adh[j]),
                   max_force=res.max_force, q_i=self.q[i] / Q_E, q_j=self.q[j] / Q_E,
                   spin_i_hz=float(np.linalg.norm(wA)) / (2 * math.pi),
                   spin_j_hz=float(np.linalg.norm(wB)) / (2 * math.pi),
                   vrel_x=vrel[0], vrel_y=vrel[1], vrel_z=vrel[2], n_x=n[0], n_y=n[1], n_z=n[2],
                   x=cpt[0], y=cpt[1], z=cpt[2], E_coll=E_coll, fragments=0)
        self.collisions.append(rec)
        if res.stuck:
            self.n_events["stick"] += 1
            self._merge(A, B)
        else:
            self.n_events["bounce"] += 1
            Jn = (1 + res.e_n) * vn * m_eff
            P = Jn * n
            if vt > 1e-15:
                tv = vt_vec / vt
                kt = self._k_body(A, rA, tv) + self._k_body(B, rB, tv)
                Jt = min(self.mu_f * Jn, (1 + self.e_t) * vt / kt)
                P = P + Jt * tv
            self._apply_impulse(A, rA, -P)
            self._apply_impulse(B, rB, P)
            if self.fragmentation and (nA > 1 or nB > 1):
                rec["fragments"] = self._fragment([(A, i), (B, j)], E_coll, cm.p.E_break)
        return True

    def _fragment(self, hits, E_coll, E_break):
        """Dominik & Tielens (1997) energy thresholds for aggregates hit in a bouncing impact."""
        N_c = sum(int(self.nmono[b]) - 1 for b, _ in hits if not self.frozen[b])
        out = 0
        if N_c <= 0:
            return 0
        if E_coll > 10 * N_c * E_break:            # catastrophic disruption
            for b, _ in hits:
                if self.nmono[b] > 1 and not self.frozen[b]:
                    mem = list(self.members[b])
                    out += self._split(b, mem[1:])
            self.n_events["shatter"] += 1
        elif E_coll > 3 * E_break:                  # erosion: the struck grain is knocked off
            for b, k in hits:
                if self.nmono[b] > 1 and not self.frozen[b]:
                    out += self._split(b, [k])
            if out:
                self.n_events["erosion"] += 1
        if out:
            self._topology_changed = True
        return out

    def _split(self, b, ks):
        """Detach grains ks from body b; momentum and angular momentum are conserved."""
        w = self._w(b)
        P = self.M[b] * self.V[b]
        Lo = self.L[b] + self.M[b] * _cross3(self.X[b], self.V[b])
        free = list(np.flatnonzero(~self.alive)[:len(ks)])
        for k, nb in zip(ks, free):
            I0 = 0.4 * self.mass[k] * self.rad[k] ** 2
            v = self.V[b] + _cross3(w, self.off[k])
            P = P - self.mass[k] * v
            Lo = Lo - (I0 * w + self.mass[k] * _cross3(self.pos[k], v))
            self.alive[nb], self.frozen[nb], self.nmono[nb] = True, False, 1
            self.members[nb] = [k]
            self.members[b].remove(k)
            self.body[k] = nb
            self.X[nb], self.V[nb], self.M[nb] = self.pos[k], v, self.mass[k]
            self.L[nb] = I0 * w
            self.Iinv[nb] = np.eye(3) / I0
            self.Iiso[nb], self.brot[nb], self.bdrag[nb] = I0, self.rotc[k], self.dragc[k]
            self.off[k] = 0.0
        mem = self.members[b]
        M = self.mass[mem].sum()
        X = (self.mass[mem, None] * self.pos[mem]).sum(0) / M
        self.M[b], self.X[b], self.V[b] = M, X, P / M
        self.L[b] = Lo - M * _cross3(X, P / M)
        self.off[mem] = self.pos[mem] - X
        I = np.zeros((3, 3))
        for k in mem:
            r = self.off[k]
            I += (0.4 * self.mass[k] * self.rad[k] ** 2 + self.mass[k] * (r @ r)) * np.eye(3) - self.mass[k] * np.outer(r, r)
        self.Iinv[b] = np.linalg.inv(I)
        self.Iiso[b] = np.trace(I) / 3
        self.nmono[b] = len(mem)
        self.bdrag[b] = self.dragc[mem].sum()
        self.brot[b] = float((self.rotc[mem] + self.dragc[mem] * (self.off[mem] ** 2).sum(1)).sum())
        return len(ks)

    def _resolve_wall(self, m, ax, side, t_now):
        A = self.body[m]
        n = np.zeros(3)
        n[ax] = side
        cpt = self.pos[m] + n * self.rad[m]
        rA = cpt - self.X[A]
        wA = self._w(A)
        vA = self.V[A] + _cross3(wA, rA)
        v_wall = self._wall_velocity(ax, side, t_now)     # inward wall speed
        vn = float(vA @ n) + v_wall                        # approach speed relative to the wall
        if vn <= 0:
            return False
        m_eff = 1.0 / self._k_body(A, rA, n)
        cm = self._contact(m, None)
        res = cm.resolve(vn, m_eff)
        stuck = res.stuck if self.wall_mode == "jkr" else self.wall_mode == "always"
        e_n = res.e_n if not res.stuck else math.sqrt(self.mats[self.mat[m]].e_dry * self.wall_mat.e_dry)
        vt_vec = vA - float(vA @ n) * n
        vt = float(np.linalg.norm(vt_vec))
        if self.charging:
            self._charge(m, None, res.max_contact_radius)
        E_coll = 0.5 * self.M[A] * float(self.V[A] @ self.V[A])
        rec = dict(t=t_now, kind="wall", i=m, j=-1, mat_i=self.mats[self.mat[m]].name,
                   mat_j=self.wall_mat.name, n_i=int(self.nmono[A]), n_j=0, v_n=vn, v_t=vt,
                   v_rel=float(np.linalg.norm(vA)), m_eff=m_eff, outcome="stick" if stuck else "bounce",
                   e_n=0.0 if stuck else e_n, yielded=res.yielded, v_stick_est=cm.v_stick_analytic(m_eff),
                   adhesion_factor=self.adh[m] * self.wall_adh,
                   max_force=res.max_force, q_i=self.q[m] / Q_E, q_j=0.0,
                   spin_i_hz=float(np.linalg.norm(wA)) / (2 * math.pi), spin_j_hz=0.0,
                   vrel_x=vA[0], vrel_y=vA[1], vrel_z=vA[2], n_x=n[0], n_y=n[1], n_z=n[2],
                   x=cpt[0], y=cpt[1], z=cpt[2], E_coll=E_coll, fragments=0)
        self.collisions.append(rec)
        if stuck:
            self.n_events["wall_stick"] += 1
            self.frozen[A] = True
            self.V[A] = 0
            self.L[A] = 0
        else:
            self.n_events["wall_bounce"] += 1
            Jn = (1 + e_n) * vn * m_eff
            P = Jn * n
            if vt > 1e-15:
                tv = vt_vec / vt
                Jt = min(self.mu_f * Jn, (1 + self.e_t) * vt / self._k_body(A, rA, tv))
                P = P + Jt * tv
            self._apply_impulse(A, rA, -P)
            if self.fragmentation and self.nmono[A] > 1:
                rec["fragments"] = self._fragment([(A, m)], E_coll, cm.p.E_break)
        return True

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
            I += np.linalg.inv(self.Iinv[b]) + self.M[b] * ((d @ d) * np.eye(3) - np.outer(d, d))
        fz = self.frozen[A] or self.frozen[B]
        mem = self.members[A] + self.members[B]
        self.members[A], self.members[B] = mem, []
        self.body[mem] = A
        self.X[A], self.M[A] = X, M
        self.V[A], self.L[A] = (np.zeros(3), np.zeros(3)) if fz else (V, L)
        self.Iinv[A] = np.linalg.inv(I)
        self.Iiso[A] = np.trace(I) / 3
        self.off[mem] = self.pos[mem] - X
        self.brot[A] = float((self.rotc[mem] + self.dragc[mem] * (self.off[mem] ** 2).sum(1)).sum())
        self.nmono[A] += self.nmono[B]
        self.bdrag[A] += self.bdrag[B]
        self.frozen[A] = fz
        self.alive[B] = False
        self.nmono[B] = 0

    # ------------------------------------------------------------ main loop
    def _kick(self, F, h, brownian=False):
        """Exact drag + constant-force velocity update over h for all free bodies (half of a KDK step)."""
        Fb = np.empty((self.N, 3))
        for k in range(3):
            Fb[:, k] = np.bincount(self.body, F[:, k], self.N)
        act = np.flatnonzero(self.alive & ~self.frozen)
        Ma = self.M[act]
        acc = Fb[act] / Ma[:, None]
        Tb = None
        if (self.nmono[act] > 1).any():
            T = _crossN(self.off, F)
            Tb = np.empty((self.N, 3))
            for k in range(3):
                Tb[:, k] = np.bincount(self.body, T[:, k], self.N)
        if self.env.P > 0:
            tau = Ma / self.bdrag[act]
            decay = np.exp(-h / tau)[:, None]
            vterm = acc * tau[:, None]
            self.V[act] = vterm + (self.V[act] - vterm) * decay
            rdecay = np.exp(-h * self.brot[act] / self.Iiso[act])[:, None]
            self.L[act] = (self.L[act] + (Tb[act] * h if Tb is not None else 0)) * rdecay
            if brownian and self.f_brown:
                # Ornstein-Uhlenbeck thermal kicks consistent with the gas drag (over the full step)
                kT = K_B * self.env.T
                sv = np.sqrt(kT / Ma * (1 - decay[:, 0] ** 4))
                self.V[act] += sv[:, None] * self.rng.standard_normal((len(act), 3))
                Ii = self.Iiso[act]
                sl = Ii * np.sqrt(kT / Ii * (1 - rdecay[:, 0] ** 4))
                self.L[act] += sl[:, None] * self.rng.standard_normal((len(act), 3))
        else:
            self.V[act] += acc * h
            if Tb is not None:
                self.L[act] += Tb[act] * h
        return act

    def step(self, t):
        """Kick-drift-kick step: half kick, collision-resolved drift, half kick (2nd order)."""
        dt = self.dt
        self._step_count += 1
        a_app = self.env.apparent_accel(t, dt)
        W = self._omega()
        vm = self._mono_vel(W)
        F = self._forces(a_app, vm)
        act = self._kick(F, 0.5 * dt)

        remaining = dt
        excl = np.zeros(len(self.nlist), bool)
        wexcl = np.zeros(self.N, bool)
        for _ in range(10000):
            W = self._omega()
            vm = self._mono_vel(W)
            evs = self._events(vm, remaining, excl, wexcl)
            if not evs:
                self._drift(remaining, W, act)
                break
            # process events in time order until one involves a body already changed
            touched, elapsed = set(), 0.0
            for t_ev, ev in evs:
                if ev[0] == "pp":
                    i, j = self.nlist[ev[1]]
                    bodies = (int(self.body[i]), int(self.body[j]))
                else:
                    bodies = (int(self.body[ev[1]]),)
                if touched.intersection(bodies):
                    break
                self._drift(t_ev - elapsed, W, act)
                elapsed = t_ev
                t_now = t + dt - remaining + elapsed
                if ev[0] == "pp":
                    if not self._resolve_pp(ev[1], t_now):
                        excl[ev[1]] = True      # grazing contact (contact point already separating)
                elif not self._resolve_wall(ev[1], ev[2], ev[3], t_now):
                    wexcl[ev[1]] = True         # grazing contact
                touched.update(bodies)
                if self._topology_changed or any(self.nmono[b] > 1 for b in bodies):
                    self._topology_changed = False
                    break   # an aggregate changed: recompute predictions
            remaining -= elapsed
            act = np.flatnonzero(self.alive & ~self.frozen)
        # second half kick: velocity-dependent forces (Coriolis, Lorentz) re-evaluated at the new
        # velocities; position-dependent ones reuse this step's evaluation
        W = self._omega()
        F2 = self._forces(a_app, self._mono_vel(W), reuse_pairs=True)
        self._kick(F2, 0.5 * dt, brownian=True)
        disp = np.abs(self.pos - self.pos_at_build).max()
        if disp > 0.4 * self.skin:
            self._build_neighbors()

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
            mean_abs_charge_e=float(np.abs(self.q).mean() / Q_E)))

    def run(self, progress=True):
        self._build_neighbors()
        t, next_out = 0.0, 0.0
        n_steps = int(round(self.duration / self.dt))
        t0 = time.time()
        last = t0
        for s in range(n_steps):
            if t >= next_out - 1e-12:
                self._record(t)
                next_out += self.out_dt
            self.step(t)
            t = (s + 1) * self.dt
            if progress and time.time() - last > 2.0:
                last = time.time()
                ev = self.n_events
                print(f"  [{self.name}] t={t:7.2f}/{self.duration:.0f}s  sticks={ev['stick']} "
                      f"bounces={ev['bounce']} wall={ev['wall_stick']}+{ev['wall_bounce']}  "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
        self._record(t)
        self.wall_time = time.time() - t0
        return self

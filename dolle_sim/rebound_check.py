"""Optional cross-check of the DOLL-E engine against REBOUND (pip install rebound).

THIS IS THE ONLY FILE THAT USES REBOUND. The simulator itself never imports
it, and no REBOUND code is copied into DOLL-E. tests/test_provenance.py checks
that only this file imports REBOUND, that the simulator never imports this
file, and that the simulator runs with REBOUND unavailable.

REBOUND (Rein & Liu 2012, https://rebound.readthedocs.io) is a widely used,
well-tested N-body code. Here it re-runs a simplified version of the same
chamber, with the same initial conditions and external forces:
  * Part 1, independent integrator: over the first 0.2 s, REBOUND's IAS15
    integrator propagates every grain that has no collision, with the same
    equations of motion (we type the residual acceleration, centrifugal,
    Coriolis and drag terms into REBOUND; REBOUND computes mutual gravity).
    This checks our integration, not the equations or the drag law.
  * Part 2, partly independent: REBOUND's leapfrog, gravity and "line"
    collision detection, with OUR JKR contact model plugged in as REBOUND's
    collision resolver, our exact drag + Coriolis velocity step in REBOUND's
    pre/post-timestep hooks, and walls handled by our code after every REBOUND
    step. It checks collision detection and bookkeeping, NOT the contact
    physics, material values, spin, friction, charging, fragmentation, or any
    force switched off for the comparison (electrostatics, g-jitter, gravity
    gradient, thermo-/photophoresis, radiation pressure, Brownian motion, wall
    vibration).

How the physics is split in REBOUND:
  * leapfrog (DKD) for positions with the position-dependent forces
    (residual acceleration, centrifugal) in `additional_forces`;
  * gas drag + Coriolis applied as an exact velocity operator in
    pre/post-timestep hooks (Strang splitting), because leapfrog evaluates
    velocity-dependent forces explicitly and would pump energy in a spinning frame;
  * collisions: REBOUND reports a pair after the step; the resolver rewinds the
    pair along its straight-line relative motion to the actual contact time, so the
    contact normal and the normal speed are exact; it applies the impulse there and
    re-advances the pair to the end of the step.

What is compared (both engines run with the physics REBOUND can represent:
point-mass rotation-free grains, residual acceleration, gas drag, payload
spin, mutual gravity, JKR stick/bounce, static walls):
  * ballistic trajectories before any collision (against REBOUND IAS15),
  * collision statistics over the run, with Poisson uncertainties.

Differences by design: REBOUND merges stuck grains into one sphere of equal
volume, our engine keeps rigid aggregates of touching spheres (larger collision
cross-section); REBOUND grains do not spin.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np

from .simulation import ChamberSim
from .materials import pair_props
from .contact import JKRContact


def _simplify(cfg, chamber, n_max):
    """Strip the physics REBOUND cannot represent, from both engines' configs."""
    cfg = copy.deepcopy(cfg)
    ch = copy.deepcopy(chamber)
    for env in (cfg.setdefault("environment", {}), ch.setdefault("environment", {})):
        env["jitter_rms_g"] = 0.0
        env["gravity_gradient"] = False
        env["illumination"] = {"intensity_w_m2": 0.0}
        env["temperature_gradient_k_m"] = [0, 0, 0]
        env.pop("accel_csv", None)
    cfg["forces"] = {"coulomb": False, "polarization": False, "image_charges": False, "self_gravity": True,
                     "lorentz": False, "external_e_field": False, "thermophoresis": False,
                     "photophoresis": False, "radiation_pressure": False, "brownian": False,
                     "space_charge": False, "long_range_cutoff_m": 0.003}
    c = cfg.setdefault("contact", {})
    c.update({"contact_charging": False, "fragmentation": False, "friction_coefficient": 0.0})
    ch.pop("wall_vibration", None)
    total = sum(int(g["count"]) for g in ch["particles"])
    scale = min(1.0, n_max / max(total, 1))
    for g in ch["particles"]:
        g["count"] = max(1, int(round(int(g["count"]) * scale)))
        g["charge_e_mean"] = 0
        g["charge_e_sigma"] = 0
        g["spin"] = {"rate_hz_mean": 0.0, "rate_hz_sigma": 0.0}
    return cfg, ch


def _exact_velocity_operator(V, gam, Om, h):
    """Exact solution of dV/dt = -gam V - 2 Omega x V over h (rows of V, per-row gam)."""
    W = float(np.linalg.norm(Om))
    if W == 0:
        return V * np.exp(-gam * h)[:, None]
    e = Om / W
    e1 = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e, e1)
    vp = (V @ e) * np.exp(-gam * h)
    z = (V @ e1 + 1j * (V @ e2)) * np.exp(-(gam + 2j * W) * h)
    return vp[:, None] * e + z.real[:, None] * e1 + z.imag[:, None] * e2


def run_rebound(sim0: ChamberSim, duration: float):
    import rebound

    dt = sim0.dt
    env = sim0.env
    rs = rebound.Simulation()
    rs.G = 6.67430e-11
    rs.integrator = "leapfrog"
    rs.dt = dt
    rs.collision = "line"
    N = sim0.N
    for k in range(N):
        rs.add(m=sim0.mass[k], r=sim0.rad[k], x=sim0.pos[k, 0], y=sim0.pos[k, 1], z=sim0.pos[k, 2],
               vx=sim0.V[k, 0], vy=sim0.V[k, 1], vz=sim0.V[k, 2])
    # Per-particle bookkeeping keyed by REBOUND index. Particles are never removed (indices stay stable);
    # merged grains are "parked" far away with zero mass/radius; wall-stuck grains stay in place
    # (frozen, infinite mass) so they remain collision targets as in DOLL-E.
    parked = np.zeros(N, bool)
    frozen = np.zeros(N, bool)
    info = [dict(mat=int(sim0.mat[k]), rad=float(sim0.rad[k]), adh=float(sim0.adh[k]),
                 dragc=float(sim0.dragc[k]), n=1, m=float(sim0.mass[k])) for k in range(N)]
    contacts = {}
    log = []
    a_res, Om, off = env.a_res.copy(), env.omega.copy(), sim0.axis_offset.copy()

    def park(idx):
        p = rs.particles[idx]
        parked[idx] = True
        p.m, p.r = 0.0, 0.0
        p.x, p.y, p.z = 1e3 * (idx + 1), 1e3, 1e3
        p.vx = p.vy = p.vz = 0.0

    def contact(ki, kj):
        a = info[ki]
        b = info[kj] if kj is not None else None
        f = math.sqrt(a["adh"] * b["adh"]) if b else a["adh"] * sim0.wall_adh
        key = (a["mat"], round(a["rad"] * 2e6), None if b is None else b["mat"],
               None if b is None else round(b["rad"] * 2e6), round(math.log(f) * 50))
        c = contacts.get(key)
        if c is None:
            m2 = sim0.wall_mat if b is None else sim0.mats[b["mat"]]
            c = contacts[key] = JKRContact(pair_props(sim0.mats[a["mat"]], a["rad"], m2,
                                                      None if b is None else b["rad"],
                                                      adhesion_factor=math.exp(key[-1] / 50)))
        return c

    def velocity_operator(h):
        n = rs.N
        vel = np.zeros((n, 3))
        rs.serialize_particle_data(vxvyvz=vel)
        gam = np.array([info[k]["dragc"] / info[k]["m"] if env.P > 0 else 0.0 for k in range(n)])
        vel = _exact_velocity_operator(vel, gam, Om, h)
        vel[parked | frozen] = 0.0
        for k, p in enumerate(rs.particles):
            p.vx, p.vy, p.vz = vel[k]

    def pre(simp):
        velocity_operator(0.5 * dt)

    def post(simp):
        velocity_operator(0.5 * dt)

    def forces(simp):
        s = simp.contents
        n = s.N
        xyz = np.zeros((n, 3))
        s.serialize_particle_data(xyz=xyz)
        a = np.repeat(a_res[None, :], n, axis=0)
        if Om.any():
            a -= np.cross(Om[None, :], np.cross(Om[None, :], xyz + off))
        a[parked | frozen] = 0.0
        for k, p in enumerate(s.particles):
            p.ax += a[k, 0]
            p.ay += a[k, 1]
            p.az += a[k, 2]

    def resolve(simp, col):
        s = simp.contents
        i, j = col.p1, col.p2
        if parked[i] or parked[j] or (frozen[i] and frozen[j]):
            return 0
        ps = s.particles
        p1, p2 = ps[i], ps[j]
        x1 = np.array([p1.x, p1.y, p1.z]); x2 = np.array([p2.x, p2.y, p2.z])
        v1 = np.array([p1.vx, p1.vy, p1.vz]); v2 = np.array([p2.vx, p2.vy, p2.vz])
        S = p1.r + p2.r
        r_end, u = x2 - x1, v2 - v1
        # rewind along the relative straight-line motion to the contact time (largest root <= dt back)
        a2, b2, c2 = u @ u, -2 * (r_end @ u), r_end @ r_end - S * S
        disc = b2 * b2 - 4 * a2 * c2
        if a2 <= 0 or disc < 0:
            return 0
        sback = (-b2 + math.sqrt(disc)) / (2 * a2)        # time since contact
        if sback < 0 or sback > 1.5 * dt:
            return 0
        n = (r_end - u * sback) / S
        vn = float((v1 - v2) @ n)
        if vn <= 0:
            return 0                                      # was separating at contact
        if frozen[i] or frozen[j]:
            m_eff = p2.m if frozen[i] else p1.m
        else:
            m_eff = p1.m * p2.m / (p1.m + p2.m)
        res = contact(i, j).resolve(vn, m_eff)
        log.append(dict(t=s.t, kind="pp", n_i=info[i]["n"], n_j=info[j]["n"], v_n=vn,
                        outcome="stick" if res.stuck else "bounce", e_n=res.e_n,
                        frozen=bool(frozen[i] or frozen[j])))
        # move both back to contact
        x1c, x2c = x1 - v1 * sback, x2 - v2 * sback
        if res.stuck:
            if frozen[i] or frozen[j]:
                keep, gone = (i, j) if frozen[i] else (j, i)
                frozen[gone] = True                       # sticks to the wall-bound grain
                pg = ps[gone]
                xc = x2c if gone == j else x1c
                pg.x, pg.y, pg.z = xc
                pg.vx = pg.vy = pg.vz = 0.0
                return 0
            M = p1.m + p2.m
            vc = (p1.m * v1 + p2.m * v2) / M
            xc = (p1.m * x1c + p2.m * x2c) / M + vc * sback
            p1.x, p1.y, p1.z = xc
            p1.vx, p1.vy, p1.vz = vc
            p1.r = (p1.r ** 3 + p2.r ** 3) ** (1 / 3)
            p1.m = M
            info[i]["n"] += info[j]["n"]
            info[i]["dragc"] += info[j]["dragc"]
            info[i]["m"] = M
            park(j)
            return 0
        J = (1 + res.e_n) * vn * m_eff
        if not frozen[i]:
            v1 = v1 - J * n / p1.m
        if not frozen[j]:
            v2 = v2 + J * n / p2.m
        x1n, x2n = x1c + v1 * sback, x2c + v2 * sback
        if not frozen[i]:
            p1.x, p1.y, p1.z = x1n
            p1.vx, p1.vy, p1.vz = v1
        if not frozen[j]:
            p2.x, p2.y, p2.z = x2n
            p2.vx, p2.vy, p2.vz = v2
        return 0

    rs.additional_forces = forces
    rs.pre_timestep_modifications = pre
    rs.post_timestep_modifications = post
    rs.collision_resolve = resolve
    size = sim0.size
    wall_stuck = 0
    series = []
    t_out = 0.0
    n_steps = int(round(duration / dt))
    for _ in range(n_steps):
        rs.steps(1)
        for idx in np.flatnonzero(~parked & ~frozen):
            p = rs.particles[int(idx)]
            h = int(idx)
            for ax_i, attr, vattr in ((0, "x", "vx"), (1, "y", "vy"), (2, "z", "vz")):
                x, v = getattr(p, attr), getattr(p, vattr)
                lo, hi = p.r, size[ax_i] - p.r
                if (x < lo and v < 0) or (x > hi and v > 0):
                    res = contact(h, None).resolve(abs(v), p.m)
                    mode = sim0.wall_mode
                    stuck = res.stuck if mode == "jkr" else mode == "always"
                    plane = lo if x < lo else hi
                    if stuck:
                        wall_stuck += info[h]["n"]
                        frozen[h] = True
                        setattr(p, attr, plane)
                        p.vx = p.vy = p.vz = 0.0
                    else:
                        e = res.e_n if not res.stuck else math.sqrt(sim0.mats[info[h]["mat"]].e_dry * sim0.wall_mat.e_dry)
                        setattr(p, attr, plane + e * (plane - x))      # reflect the overshoot
                        setattr(p, vattr, -e * v)
                    log.append(dict(t=rs.t, kind="wall", n_i=info[h]["n"], n_j=0, v_n=abs(v),
                                    outcome="stick" if stuck else "bounce", e_n=res.e_n, frozen=False))
                    break
        if rs.t >= t_out - 1e-12:
            vel = np.zeros((rs.N, 3))
            rs.serialize_particle_data(vxvyvz=vel)
            live = ~parked & ~frozen
            series.append(dict(t=rs.t, free_bodies=int(live.sum()),
                               mean_speed=float(np.linalg.norm(vel[live], axis=1).mean()) if live.any() else 0.0,
                               wall_monomers=wall_stuck))
            t_out += 0.25
    return dict(log=log, series=series, wall_stuck=wall_stuck, final_N=int((~parked & ~frozen).sum()))


def _ballistic_rebound(sim0, t_end):
    """Integrate every grain with REBOUND IAS15 with collisions off and the same external forces."""
    import rebound
    rs = rebound.Simulation()
    rs.G = 6.67430e-11
    rs.integrator = "ias15"         # high-accuracy adaptive reference integrator
    env = sim0.env
    for k in range(sim0.N):
        rs.add(m=sim0.mass[k], r=sim0.rad[k], x=sim0.pos[k, 0], y=sim0.pos[k, 1], z=sim0.pos[k, 2],
               vx=sim0.V[k, 0], vy=sim0.V[k, 1], vz=sim0.V[k, 2])
    cm = sim0.dragc / sim0.mass if env.P > 0 else np.zeros(sim0.N)
    a_res, Om, off = env.a_res, env.omega, sim0.axis_offset

    def forces(simp):
        s = simp.contents
        N = s.N
        xyz = np.zeros((N, 3))
        vel = np.zeros((N, 3))
        s.serialize_particle_data(xyz=xyz, vxvyvz=vel)
        a = a_res[None, :] - cm[:, None] * vel
        if Om.any():
            a += -2 * np.cross(Om[None, :], vel) - np.cross(Om[None, :], np.cross(Om[None, :], xyz + off))
        for k, p in enumerate(s.particles):
            p.ax += a[k, 0]
            p.ay += a[k, 1]
            p.az += a[k, 2]

    rs.additional_forces = forces
    rs.force_is_velocity_dependent = 1
    rs.integrate(t_end, exact_finish_time=1)
    xyz = np.zeros((rs.N, 3))
    rs.serialize_particle_data(xyz=xyz)
    return xyz


def _stats(log):
    pp = [c for c in log if c["kind"] == "pp"]
    mono = [c for c in pp if c["n_i"] == 1 and c["n_j"] == 1 and not c.get("frozen", False)]
    n_mono = len(mono)
    sticks = sum(c["outcome"] == "stick" for c in mono)
    frac = sticks / n_mono if n_mono else None
    return {"grain_grain_impacts": len(pp), "grain_grain_impacts_err": math.sqrt(len(pp)),
            "grain_grain_sticks": sum(c["outcome"] == "stick" for c in pp),
            "free_monomer_impacts": n_mono,
            "free_monomer_stick_fraction": frac,
            "free_monomer_stick_fraction_err": (math.sqrt(frac * (1 - frac) / n_mono) if n_mono else None),
            "median_impact_speed_m_s": float(np.median([c["v_n"] for c in pp])) if pp else None,
            # single grains only: the two codes treat aggregates differently by design (sphere vs rigid cluster)
            "free_monomer_median_speed_m_s": float(np.median([c["v_n"] for c in mono])) if mono else None,
            "wall_impacts": sum(c["kind"] == "wall" for c in log),
            "wall_sticks": sum(c["kind"] == "wall" and c["outcome"] == "stick" for c in log)}


def compare(cfg: dict, chamber: dict, duration: float, out: Path, n_max: int = 400):
    cfg, chamber = _simplify(cfg, chamber, n_max)
    cfg["duration_s"] = duration
    out.mkdir(parents=True, exist_ok=True)

    # --- 1) ballistic trajectory check over a whole number of DOLL-E steps
    ours = ChamberSim(cfg, chamber, 1)
    ref0 = ChamberSim(cfg, chamber, 1)
    dt = ours.dt
    n_bal = max(1, int(round(min(0.2, duration) / dt)))
    t_bal = n_bal * dt
    ours._build_neighbors()
    for k in range(n_bal):
        ours.step(k * dt)
    rb = _ballistic_rebound(ref0, t_bal)
    touched = {c["i"] for c in ours.collisions} | {c["j"] for c in ours.collisions}
    untouched = [k for k in range(ours.N) if k not in touched and ours.nmono[ours.body[k]] == 1]
    dev = np.linalg.norm(ours.pos[untouched] - rb[untouched], axis=1) if untouched else np.array([np.nan])

    # --- 2) statistics over the full duration
    t0 = time.time()
    ours = ChamberSim(cfg, chamber, 1)
    ref_init = ChamberSim(cfg, chamber, 1)
    ours.run(progress=False)
    t_ours = time.time() - t0
    t0 = time.time()
    rbr = run_rebound(ref_init, duration)
    t_rb = time.time() - t0

    ours_log = [dict(c, frozen=bool(c.get("frozen_i") or c.get("frozen_j"))) if c["kind"] == "pp" else c
                for c in ours.collisions]
    rep = {
        "chamber": chamber.get("name"), "n_grains": ours.N, "duration_s": duration, "dt_s": dt,
        "ballistic_check": {"t_s": t_bal, "grains_compared": len(untouched),
                            "max_position_difference_m": float(np.nanmax(dev)),
                            "median_position_difference_m": float(np.nanmedian(dev))},
        "dolle_engine": {**_stats(ours_log), "run_time_s": t_ours},
        "rebound_engine": {**_stats(rbr["log"]), "run_time_s": t_rb,
                           "rebound_version": __import__("rebound").__version__},
    }
    (out / f"rebound_check_{chamber.get('name')}.json").write_text(json.dumps(rep, indent=2))
    _plot(ours, rbr, out / f"rebound_check_{chamber.get('name')}.png", chamber.get("name"))
    return rep


def _plot(ours, rb, path, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    t1 = [r["t"] for r in ours.timeseries]
    t2 = [r["t"] for r in rb["series"]]
    ax[0].plot(t1, [r["mean_speed"] * 1e3 for r in ours.timeseries], label="DOLL-E engine")
    ax[0].plot(t2, [r["mean_speed"] * 1e3 for r in rb["series"]], "--", label="REBOUND")
    ax[0].set(xlabel="t [s]", ylabel="mean speed [mm/s]", title="Speed decay")
    ax[1].plot(t1, [r["free_bodies"] for r in ours.timeseries], label="DOLL-E engine")
    ax[1].plot(t2, [r["free_bodies"] for r in rb["series"]], "--", label="REBOUND")
    ax[1].set(xlabel="t [s]", ylabel="free bodies", title="Free bodies (aggregation + wall loss)")
    v1 = [c["v_n"] for c in ours.collisions if c["kind"] == "pp"]
    v2 = [c["v_n"] for c in rb["log"] if c["kind"] == "pp"]
    if v1 or v2:
        allv = np.array(v1 + v2)
        bins = np.logspace(np.log10(max(allv.min(), 1e-6)), np.log10(allv.max()) + 1e-9, 25)
        ax[2].hist(v1, bins=bins, histtype="step", lw=2, label=f"DOLL-E ({len(v1)})")
        ax[2].hist(v2, bins=bins, histtype="step", lw=2, ls="--", label=f"REBOUND ({len(v2)})")
        ax[2].set_xscale("log")
    ax[2].set(xlabel="normal impact speed [m/s]", ylabel="count", title="Grain-grain impact speeds")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Engine cross-check vs REBOUND: {name}")
    fig.savefig(path, dpi=130)
    plt.close(fig)

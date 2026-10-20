"""Independent cross-check of the DOLL-E engine against REBOUND (pip install rebound).

REBOUND (Rein & Liu 2012, https://rebound.readthedocs.io) is a widely used,
well-tested N-body code. Here it provides an *independent* integrator,
mutual-gravity solver and collision-detection scheme ("line" detection) for
the same chamber, same initial conditions and same external forces, while our
JKR contact model is plugged in as REBOUND's collision resolver. Walls are
handled after every REBOUND step.

What is compared (both engines run with the physics REBOUND can represent:
point-mass rotation-free grains, residual acceleration, gas drag, payload
spin, mutual gravity, JKR stick/bounce, walls):
  * ballistic trajectories before the first collision (should agree to
    round-off/integrator error),
  * collision statistics over the run (chaotic, so statistical agreement:
    numbers of grain-grain impacts, sticking fraction, wall impacts, speed
    decay).

Differences by design: REBOUND merges stuck grains into one sphere of equal
volume, our engine keeps rigid aggregates of spheres; REBOUND has no grain
spin. So aggregate-related numbers can differ somewhat.
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


def _simplify(cfg, chamber):
    """Strip the physics REBOUND cannot represent, from both engines' configs."""
    cfg = copy.deepcopy(cfg)
    ch = copy.deepcopy(chamber)
    env = cfg.setdefault("environment", {})
    env["jitter_rms_g"] = 0.0
    env["gravity_gradient"] = False
    env["illumination"] = {"intensity_w_m2": 0.0}
    env["temperature_gradient_k_m"] = [0, 0, 0]
    cfg["forces"] = {"coulomb": False, "polarization": False, "image_charges": False, "self_gravity": True,
                     "lorentz": False, "external_e_field": False, "thermophoresis": False,
                     "photophoresis": False, "radiation_pressure": False, "brownian": False,
                     "long_range_cutoff_m": 0.003}
    c = cfg.setdefault("contact", {})
    c.update({"contact_charging": False, "fragmentation": False, "friction_coefficient": 0.0})
    ch.pop("wall_vibration", None)
    for g in ch["particles"]:
        g["charge_e_mean"] = 0
        g["charge_e_sigma"] = 0
        g["spin"] = {"rate_hz_mean": 0.0, "rate_hz_sigma": 0.0}
    return cfg, ch


def run_rebound(sim0: ChamberSim, duration: float, dt: float, gravity=True):
    import rebound

    env = sim0.env
    rs = rebound.Simulation()
    rs.G = 6.67430e-11 if gravity else 0.0
    rs.integrator = "leapfrog"
    rs.dt = dt
    rs.collision = "line"
    N = sim0.N
    mat = sim0.mat.copy()
    rad = sim0.rad.copy()
    adh = sim0.adh.copy()
    for k in range(N):
        rs.add(m=sim0.mass[k], r=rad[k], x=sim0.pos[k, 0], y=sim0.pos[k, 1], z=sim0.pos[k, 2],
               vx=sim0.V[k, 0], vy=sim0.V[k, 1], vz=sim0.V[k, 2])
    # per-particle bookkeeping keyed by REBOUND index. Particles are never removed (indices stay
    # stable inside a step); merged or wall-stuck grains are "parked" far away with zero mass/radius.
    parked = np.zeros(N, bool)

    def park(idx):
        p = rs.particles[idx]
        parked[idx] = True
        p.m, p.r = 0.0, 0.0
        p.x, p.y, p.z = 1e3 * (idx + 1), 1e3, 1e3
        p.vx = p.vy = p.vz = 0.0

    info = {k: dict(mat=int(mat[k]), rad=float(rad[k]), adh=float(adh[k]), dragc=float(sim0.dragc[k]), n=1,
                    m=float(sim0.mass[k])) for k in range(N)}
    contacts = {}
    log = []
    a_res = env.a_res.copy()
    Om = env.omega.copy()
    off = sim0.axis_offset.copy()

    def contact(ki, kj):
        a, b = info[ki], info[kj] if kj is not None else None
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

    def forces(simp):
        s = simp.contents
        n = s.N
        xyz, vel = np.zeros((n, 3)), np.zeros((n, 3))
        s.serialize_particle_data(xyz=xyz, vxvyvz=vel)
        c_m = np.array([info[k]["dragc"] / info[k]["m"] for k in range(n)])
        a = a_res[None, :] - c_m[:, None] * vel
        if Om.any():
            a += -2 * np.cross(Om[None, :], vel) - np.cross(Om[None, :], np.cross(Om[None, :], xyz + off))
        a[parked] = 0.0
        for k, p in enumerate(s.particles):
            p.ax += a[k, 0]
            p.ay += a[k, 1]
            p.az += a[k, 2]

    def resolve(simp, col):
        s = simp.contents
        ps = s.particles
        if parked[col.p1] or parked[col.p2]:
            return 0
        p1, p2 = ps[col.p1], ps[col.p2]
        dx = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        dv = np.array([p1.vx - p2.vx, p1.vy - p2.vy, p1.vz - p2.vz])
        d = np.linalg.norm(dx)
        n = dx / d
        vn = float(dv @ n)
        if vn <= 0:
            return 0                         # already separating
        m_eff = p1.m * p2.m / (p1.m + p2.m)
        h1, h2 = col.p1, col.p2
        res = contact(h1, h2).resolve(vn, m_eff)
        log.append(dict(t=s.t, kind="pp", n_i=info[h1]["n"], n_j=info[h2]["n"], v_n=vn,
                        outcome="stick" if res.stuck else "bounce", e_n=res.e_n))
        if res.stuck:
            M = p1.m + p2.m
            for attr in ("vx", "vy", "vz"):
                setattr(p1, attr, (p1.m * getattr(p1, attr) + p2.m * getattr(p2, attr)) / M)
            for attr in ("x", "y", "z"):
                setattr(p1, attr, (p1.m * getattr(p1, attr) + p2.m * getattr(p2, attr)) / M)
            p1.r = (p1.r ** 3 + p2.r ** 3) ** (1 / 3)
            p1.m = M
            info[h1]["n"] += info[h2]["n"]
            info[h1]["dragc"] += info[h2]["dragc"]
            info[h1]["m"] = M
            park(h2)
            return 0
        J = (1 + res.e_n) * vn * m_eff
        p1.vx -= J * n[0] / p1.m; p1.vy -= J * n[1] / p1.m; p1.vz -= J * n[2] / p1.m
        p2.vx += J * n[0] / p2.m; p2.vy += J * n[1] / p2.m; p2.vz += J * n[2] / p2.m
        return 0

    rs.additional_forces = forces
    rs.force_is_velocity_dependent = 1
    rs.collision_resolve = resolve
    size = sim0.size
    wall_stuck = 0
    wall_hits = 0
    series = []
    t_out = 0.0
    while rs.t < duration - 1e-12:
        rs.integrate(rs.t + dt, exact_finish_time=0)
        remove = []
        for idx in np.flatnonzero(~parked):
            p = rs.particles[int(idx)]
            h = int(idx)
            for ax_i, attr, vattr in ((0, "x", "vx"), (1, "y", "vy"), (2, "z", "vz")):
                x, v = getattr(p, attr), getattr(p, vattr)
                lo, hi = p.r, size[ax_i] - p.r
                if (x < lo and v < 0) or (x > hi and v > 0):
                    wall_hits += 1
                    res = contact(h, None).resolve(abs(v), p.m)
                    if res.stuck and sim0.wall_mode != "never":
                        wall_stuck += info[h]["n"]
                        remove.append(h)
                    else:
                        e = res.e_n if not res.stuck else math.sqrt(sim0.mats[info[h]["mat"]].e_dry * sim0.wall_mat.e_dry)
                        setattr(p, vattr, -e * v)
                    log.append(dict(t=rs.t, kind="wall", n_i=info[h]["n"], n_j=0, v_n=abs(v),
                                    outcome="stick" if (res.stuck and sim0.wall_mode != "never") else "bounce",
                                    e_n=res.e_n))
                    break
        for h in remove:
            park(h)
        if rs.t >= t_out - 1e-12:
            vel = np.zeros((rs.N, 3))
            rs.serialize_particle_data(vxvyvz=vel)
            live = ~parked
            series.append(dict(t=rs.t, free_bodies=int(live.sum()),
                               mean_speed=float(np.linalg.norm(vel[live], axis=1).mean()) if live.any() else 0.0,
                               wall_monomers=wall_stuck))
            t_out += 0.25
    xyz = np.zeros((rs.N, 3))
    rs.serialize_particle_data(xyz=xyz)
    return dict(log=log, series=series, wall_hits=wall_hits, wall_stuck=wall_stuck, final_N=int((~parked).sum()))


def compare(cfg: dict, chamber: dict, duration: float, out: Path, n_max: int = 400):
    cfg, chamber = _simplify(cfg, chamber)
    for g in chamber["particles"]:
        g["count"] = min(int(g["count"]), n_max)
    cfg["duration_s"] = duration
    dt = float(cfg.get("dt_s", 1e-3))
    out.mkdir(parents=True, exist_ok=True)

    # --- 1) ballistic trajectory check (no collisions: walls/neighbours irrelevant over a short time)
    ours = ChamberSim(cfg, chamber, 1)
    ref0 = ChamberSim(cfg, chamber, 1)
    t_bal = min(0.2, duration)
    ours.duration = t_bal
    ours.run(progress=False)
    rb = _ballistic_rebound(ref0, t_bal, dt)
    free = np.flatnonzero((ours.nmono[ours.body] == 1) & ~ours.frozen[ours.body])
    untouched = [k for k in free if not any(c["i"] == k or c["j"] == k for c in ours.collisions)]
    dev = np.linalg.norm(ours.pos[untouched] - rb[untouched], axis=1) if untouched else np.array([np.nan])

    # --- 2) statistics over the full duration
    t0 = time.time()
    ours = ChamberSim(cfg, chamber, 1)
    ref_init = ChamberSim(cfg, chamber, 1)
    ours.run(progress=False)
    t_ours = time.time() - t0
    t0 = time.time()
    rb = run_rebound(ref_init, duration, dt)
    t_rb = time.time() - t0

    def stats(log):
        pp = [c for c in log if c["kind"] == "pp"]
        mono = [c for c in pp if c["n_i"] == 1 and c["n_j"] == 1]
        return {"grain_grain_impacts": len(pp), "grain_grain_sticks": sum(c["outcome"] == "stick" for c in pp),
                "monomer_monomer_impacts": len(mono),
                "monomer_monomer_stick_fraction": (sum(c["outcome"] == "stick" for c in mono) / len(mono)) if mono else None,
                "median_impact_speed_m_s": float(np.median([c["v_n"] for c in pp])) if pp else None,
                "wall_impacts": sum(c["kind"] == "wall" for c in log),
                "wall_sticks": sum(c["kind"] == "wall" and c["outcome"] == "stick" for c in log)}

    rep = {
        "chamber": chamber.get("name"), "n_grains": ours.N, "duration_s": duration, "dt_s": dt,
        "ballistic_check": {"t_s": t_bal, "grains_compared": len(untouched),
                            "max_position_difference_m": float(np.nanmax(dev)),
                            "median_position_difference_m": float(np.nanmedian(dev))},
        "dolle_engine": {**stats(ours.collisions), "run_time_s": t_ours},
        "rebound_engine": {**stats(rb["log"]), "run_time_s": t_rb, "rebound_version": __import__("rebound").__version__},
    }
    (out / f"rebound_check_{chamber.get('name')}.json").write_text(json.dumps(rep, indent=2))
    _plot(ours, rb, out / f"rebound_check_{chamber.get('name')}.png", chamber.get("name"))
    return rep


def _ballistic_rebound(sim0, t_end, dt):
    """Integrate every grain with REBOUND with collisions off and the same external forces."""
    import rebound
    rs = rebound.Simulation()
    rs.G = 6.67430e-11
    rs.integrator = "ias15"         # high-accuracy adaptive reference integrator
    env = sim0.env
    for k in range(sim0.N):
        rs.add(m=sim0.mass[k], r=sim0.rad[k], x=sim0.pos[k, 0], y=sim0.pos[k, 1], z=sim0.pos[k, 2],
               vx=sim0.V[k, 0], vy=sim0.V[k, 1], vz=sim0.V[k, 2])
    cm = sim0.dragc / sim0.mass
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
    rs.integrate(t_end)
    xyz = np.zeros((rs.N, 3))
    rs.serialize_particle_data(xyz=xyz)
    return xyz


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

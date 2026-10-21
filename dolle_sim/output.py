"""Result writers: CSV logs, summary JSON, matplotlib plots and the HTML 3D viewer."""
from __future__ import annotations

import base64
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a6fdb", "#e0703a", "#2f9e66", "#b0479b", "#c9a227", "#5a5a5a"]


from .results import jsonable as _jsonable, collision_stats, viewer_payload, render_viewer, csv_text  # noqa: E402,F401


def _write_csv(path, rows):
    Path(path).write_text(csv_text(rows), newline="")


def write_chamber(sim, outdir: Path, preflight: dict, cam_cfg: dict | None = None):
    from .analysis import measurement_report
    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(outdir / "collisions.csv", sim.collisions)
    meas_rows, meas = measurement_report(sim, cam_cfg or {})
    _write_csv(outdir / "camera_measurements.csv", meas_rows)
    sim.measurement = (meas_rows, meas)
    _write_csv(outdir / "timeseries.csv", sim.timeseries)
    np.savez_compressed(outdir / "trajectory.npz", t=np.asarray(sim.frame_t),
                        pos=np.asarray(sim.frames), body=np.asarray(sim.frame_bodies),
                        radius=sim.rad, material=sim.mat,
                        material_names=np.array([m.name for m in sim.mats]))
    summary = {"chamber": sim.name, "n_grains": sim.N, "duration_s": sim.duration, "dt_s": sim.dt,
               "stats": collision_stats(sim), "measurement": meas, "preflight": preflight,
               "materials": [m.to_dict() for m in sim.mats], "wall_material": sim.wall_mat.to_dict()}
    (outdir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    return summary


# ------------------------------------------------------------------ plots
def plot_all(sims, outdir: Path, preflights):
    outdir.mkdir(parents=True, exist_ok=True)
    _plot_timeseries(sims, outdir / "timeseries.png")
    _plot_collisions(sims, outdir / "collision_velocities.png", preflights)
    _plot_sizes(sims, outdir / "aggregate_sizes.png")
    _plot_measurement(sims, outdir / "sticking_measurement.png", preflights)


def _plot_timeseries(sims, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for k, s in enumerate(sims):
        c = COLORS[k % len(COLORS)]
        t = [r["t"] for r in s.timeseries]
        g = lambda key: [r[key] for r in s.timeseries]  # noqa: E731
        ax[0, 0].plot(t, np.asarray(g("free_monomers")) / s.N, color=c, label=f"{s.name}: free")
        ax[0, 0].plot(t, np.asarray(g("wall_monomers")) / s.N, color=c, ls="--", label=f"{s.name}: on walls")
        ax[0, 1].plot(t, g("mean_aggregate_size"), color=c, label=f"{s.name} mean")
        ax[0, 1].plot(t, g("max_aggregate_size"), color=c, ls=":", label=f"{s.name} max")
        ax[1, 0].plot(t, g("stick"), color=c, label=f"{s.name}: sticks")
        ax[1, 0].plot(t, g("bounce"), color=c, ls="--", label=f"{s.name}: bounces")
        ax[1, 1].plot(t, np.asarray(g("mean_speed")) * 1e3, color=c, label=s.name)
    ax[0, 0].set(title="Grain fraction", xlabel="time [s]", ylabel="fraction")
    ax[0, 1].set(title="Aggregate size (free bodies)", xlabel="time [s]", ylabel="grains per body")
    ax[1, 0].set(title="Cumulative grain-grain collisions", xlabel="time [s]", ylabel="count")
    ax[1, 1].set(title="Mean speed of free bodies", xlabel="time [s]", ylabel="mm/s")
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.legend(fontsize=7, frameon=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_collisions(sims, path, preflights):
    fig, axes = plt.subplots(1, len(sims), figsize=(5.5 * len(sims), 4.2), squeeze=False, constrained_layout=True)
    for k, (s, pf) in enumerate(zip(sims, preflights)):
        ax = axes[0, k]
        pp = [c for c in s.collisions if c["kind"] == "pp"]
        st = np.array([c["v_n"] for c in pp if c["outcome"] == "stick"])
        bo = np.array([c["v_n"] for c in pp if c["outcome"] == "bounce"])
        allv = np.concatenate([st, bo]) if len(st) + len(bo) else np.array([1e-3, 1e-1])
        bins = np.logspace(np.log10(max(allv.min(), 1e-6)) - 0.1, np.log10(allv.max()) + 0.1, 30)
        ax.hist([st, bo], bins=bins, stacked=True, color=["#2f9e66", "#e0703a"], label=["stick", "bounce"])
        for g in pf["groups"]:
            ax.axvline(g["v_stick_pp_DEM_m_s"], color="k", ls="--", lw=1)
            ax.text(g["v_stick_pp_DEM_m_s"], ax.get_ylim()[1] * 0.95, f" v_stick {g['material']}",
                    fontsize=7, va="top")
        ax.set_xscale("log")
        ax.set(title=f"{s.name}: grain-grain impacts", xlabel="normal impact speed [m/s]", ylabel="count")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_measurement(sims, path, preflights):
    """Stick fraction vs speed for perfect 3D truth and for each camera strategy, with logistic fits."""
    fig, axes = plt.subplots(1, len(sims), figsize=(6.2 * len(sims), 4.6), squeeze=False, constrained_layout=True)
    for k, (s, pf) in enumerate(zip(sims, preflights)):
        ax = axes[0, k]
        rows, rep = getattr(s, "measurement", ([], {}))
        mono = [r for r in rows if r["n_i"] == 1 and r["n_j"] == 1 and not r.get("on_wall_partner")
                and not r.get("post_fragment")]
        series = [
            ("truth: 3D normal speed", mono, "v_n_true", rep.get("threshold_truth_normal_speed"), "#2a6fdb"),
            ("1 camera: 2D relative speed", [r for r in mono if r["observable"]], "v_rel_measured_2d",
             rep.get("threshold_2d_relative"), "#e0703a"),
            ("1 camera: 2D normal component", [r for r in mono if r["observable"]], "v_n_measured_2d",
             rep.get("threshold_2d_normal"), "#b0479b"),
            ("stereo: 3D normal speed", [r for r in mono if r.get("observable_stereo")], "v_n_measured_stereo",
             rep.get("threshold_stereo_normal"), "#2f9e66"),
        ]
        for label, rr, key, fit, col in series:
            if not rr:
                continue
            v = np.maximum(np.array([r[key] for r in rr]), 1e-5)
            y = np.array([r["outcome"] == "stick" for r in rr], float)
            bins = np.logspace(np.log10(v.min()), np.log10(v.max()) + 1e-9, 14)
            idx = np.digitize(v, bins)
            xc, fr = [], []
            for b in range(1, len(bins)):
                m = idx == b
                if m.sum() >= 3:
                    xc.append(np.sqrt(bins[b - 1] * bins[b]))
                    fr.append(y[m].mean())
            ax.plot(xc, fr, "o", color=col, ms=3.5, alpha=0.8)
            if fit:
                xx = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 200)
                ax.plot(xx, 1 / (1 + np.exp((np.log(xx) - np.log(fit["v50_m_s"])) / fit["log_width"])), color=col,
                        label=f"{label} (N={len(rr)}): {fit['v50_m_s'] * 1e3:.1f} "
                              f"[{fit['v50_lo_1sigma'] * 1e3:.1f}-{fit['v50_hi_1sigma'] * 1e3:.1f}] mm/s")
            else:
                ax.plot([], [], "o", color=col, label=f"{label} (N={len(rr)}): no fit (needs sticks and bounces)")
        for g in pf["groups"]:
            ax.axvline(g["v_stick_pp_DEM_m_s"], color="k", ls="--", lw=0.8)
        ax.set(xscale="log", ylim=(-0.05, 1.05), xlabel="impact speed [m/s]", ylabel="fraction of impacts that stick",
               title=f"{s.name}: sticking threshold (dashed = model v_stick)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_sizes(sims, path):
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    for k, s in enumerate(sims):
        sizes = s.nmono[s.alive]
        u, n = np.unique(sizes, return_counts=True)
        ax.plot(u, n, "o-", color=COLORS[k % len(COLORS)], label=s.name)
    ax.set(xscale="log", yscale="log", xlabel="grains per aggregate (incl. wall-bound)",
           ylabel="number of bodies", title="Final aggregate size distribution")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sweep(curves, velocities, thresholds, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for k, (label, p) in enumerate(curves.items()):
        ax[0].plot(velocities, p, color=COLORS[k % len(COLORS)], lw=2, label=label)
    ax[0].set(xscale="log", xlabel="impact speed [m/s]", ylabel="sticking probability",
              title="Monomer-monomer sticking probability", ylim=(-0.03, 1.03))
    for k, (label, rows) in enumerate(thresholds.items()):
        c = COLORS[k % len(COLORS)]
        ax[1].plot(rows[:, 0] * 2e6, rows[:, 1], color=c, lw=2, label=f"{label} (DEM)")
        ax[1].plot(rows[:, 0] * 2e6, rows[:, 2], color=c, lw=1, ls="--", label=f"{label} (Chokshi+93)")
    ax[1].set(xscale="log", yscale="log", xlabel="grain diameter [um]", ylabel="sticking velocity [m/s]",
              title="Sticking threshold vs grain size")
    for a in ax:
        a.grid(alpha=0.3, which="both")
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ viewer
def write_viewer(sims, path: Path, title: str, max_particles=1500, fps=15):
    path.write_text(render_viewer(viewer_payload(sims, max_particles, fps), title), encoding="utf-8")

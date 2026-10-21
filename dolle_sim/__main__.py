"""DOLL-E dust collision simulator -- command line interface.

    python -m dolle_sim run configs/rocksat_x_carbon_vs_silica.json
    python -m dolle_sim preflight configs/rocksat_x_carbon_vs_silica.json
    python -m dolle_sim sweep --pairs graphite:graphite silica:silica --radius 50e-6
    python -m dolle_sim invert --material graphite --radius 50e-6 --vstick 0.08
    python -m dolle_sim materials
    python -m dolle_sim analyze results/rocksat_x_carbon_vs_silica
    python -m dolle_sim rebound-check configs/rocksat_x_carbon_vs_silica.json --duration 10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .materials import MATERIALS
from .design import preflight, stick_probability_curve, threshold_vs_radius, infer_surface_energy
from .output import write_chamber, plot_all, write_viewer, plot_sweep, collision_stats, _jsonable


def _load(path):
    return json.loads(Path(path).read_text())


def _print_preflight(pf):
    print(f"\n=== Pre-flight: {pf['chamber']} ===")
    e = pf["environment"]
    print(f"  gas {e['pressure_pa']:g} Pa, mean free path {e['mean_free_path_m'] * 1e6:.3g} um, "
          f"payload spin {e['spin_rate_hz']:g} Hz about {np.round(e['spin_axis'], 2).tolist()}, centrifugal max {pf['centrifugal_accel_max_g']:.2e} g")
    for g in pf["groups"]:
        print(f"  [{g['material']}] N={g['count']} d={2e6 * g['radius_m']:.0f} um  Kn={g['knudsen_number']:.3g}  "
              f"t_stop={g['stopping_time_s']:.3g} s")
        print(f"      v_stick grain-grain: DEM {g['v_stick_pp_DEM_m_s'] * 1e3:.2f} mm/s "
              f"(Chokshi+93 {g['v_stick_pp_Chokshi_m_s'] * 1e3:.2f} mm/s), grain-wall {g['v_stick_wall_DEM_m_s'] * 1e3:.2f} mm/s")
        print(f"      launch {g['launch_speed_m_s'] * 1e3:.2f} mm/s, crossing time {g['chamber_crossing_time_s']:.2f} s, "
              f"~{g['expected_pp_collisions_total']:.0f} grain-grain collisions expected")
        print(f"      spin-down time {g['spin_down_time_s']:.3g} s, rms charge {g['rms_charge_e']:.3g} e, "
              f"Brownian speed {g['brownian_speed_m_s'] * 1e6:.2g} um/s")
        print(f"      wall: v_stick {g['v_stick_wall_DEM_m_s'] * 1e3:.2f} mm/s (adhesion factor {g['wall_adhesion_factor']:g}), "
              f"vibration {g['wall_vibration_speed_m_s'] * 1e3:.1f} mm/s -> {100 * g['wall_capture_probability_per_hit']:.1f}% of wall hits stick")
        print("      acceleration budget (m/s^2, and in g):")
        for k, v in sorted(g["acceleration_budget_m_s2"].items(), key=lambda kv: -kv[1]):
            print(f"        {k:34s} {v:10.3e}   {v / 9.80665:9.2e} g")
    for w in pf["warnings"]:
        print("  ! " + w)


def _print_measurement(m):
    print(f"  camera: {m['observable_monomer_impacts']} of {m['monomer_monomer_impacts']} monomer-monomer impacts "
          f"observable ({m['observable_monomer_impacts_stereo']} in stereo)")
    for key, label in (("threshold_truth_normal_speed", "truth (3D normal speed, all impacts)"),
                       ("threshold_2d_relative", "1 camera, 2D relative speed"),
                       ("threshold_2d_normal", "1 camera, 2D normal component"),
                       ("threshold_stereo_normal", "stereo cameras, 3D normal speed")):
        f = m.get(key)
        txt = (f"v_stick = {f['v50_m_s'] * 1e3:7.2f} mm/s  (1-sigma {f['v50_lo_1sigma'] * 1e3:.2f}"
               f"-{f['v50_hi_1sigma'] * 1e3:.2f}, N={f['n']})") if f else "no fit (needs sticks and bounces)"
        print(f"    {label:40s} {txt}")


def cmd_analyze(a):
    """Re-run the synthetic-camera analysis on saved results (no re-simulation)."""
    import csv
    from types import SimpleNamespace
    from .analysis import measurement_report
    from .output import _plot_measurement, _write_csv
    out = Path(a.results)
    cfg = _load(out / "config_used.json")
    top = _load(out / "summary.json")
    num = {"t", "v_n", "v_t", "v_rel", "vrel_x", "vrel_y", "vrel_z", "vcm_x", "vcm_y", "vcm_z",
           "n_x", "n_y", "n_z", "x", "y", "z", "r_sum"}
    sims, pfs = [], []
    for ch, summ in zip(cfg["chambers"], top["chambers"]):
        d = out / ch.get("name", "chamber")
        with open(d / "collisions.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        cols = []
        for r in rows:
            c = {k: (float(v) if k in num else v) for k, v in r.items()}
            for k in ("i", "j", "n_i", "n_j"):
                c[k] = int(c[k])
            for k in ("frozen_i", "frozen_j", "post_fragment", "yielded"):
                if k in c:
                    c[k] = c[k] in ("True", "true", "1")
            cols.append(c)
        cam = dict(cfg.get("camera", {}))
        cam.update(ch.get("camera", {}))
        if a.camera:
            cam.update(json.loads(a.camera))
        sim = SimpleNamespace(collisions=cols, size=np.asarray(ch.get("size_m", [0.05, 0.05, 0.05]), float),
                              duration=summ["duration_s"], name=summ["chamber"])
        mrows, rep = measurement_report(sim, cam)
        sim.measurement = (mrows, rep)
        _write_csv(d / "camera_measurements.csv", mrows)
        summ["measurement"] = rep
        (d / "summary.json").write_text(json.dumps(_jsonable(summ), indent=2))
        print(f"[{summ['chamber']}]")
        _print_measurement(rep)
        sims.append(sim)
        pfs.append(summ["preflight"])
    (out / "summary.json").write_text(json.dumps(_jsonable(top), indent=2))
    _plot_measurement(sims, out / "sticking_measurement.png", pfs)
    print(f"Updated {out / 'sticking_measurement.png'}")


def cmd_run(a):
    from .simulation import ChamberSim
    cfg = _load(a.config)
    if a.duration:
        cfg["duration_s"] = a.duration
    name = cfg.get("name", Path(a.config).stem)
    out = Path(a.out or Path("results") / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))
    sims, pfs, summaries = [], [], []
    seed = int(cfg.get("seed", 1))
    for k, ch in enumerate(cfg["chambers"]):
        pf = preflight(cfg, ch)
        _print_preflight(pf)
        sim = ChamberSim(cfg, ch, seed + k)
        print(f"\nRunning '{sim.name}': {sim.N} grains, {sim.duration:g} s at dt={sim.dt:g} s ...")
        sim.run(progress=not a.quiet)
        st = collision_stats(sim)
        print(f"  done in {sim.wall_time:.0f} s: {st['grain_grain_collisions']} grain-grain impacts "
              f"({st['grain_grain_sticks']} stuck), {st['wall_collisions']} wall impacts "
              f"({st['wall_sticks']} stuck), largest aggregate {st['max_aggregate_size']} grains")
        cam = dict(cfg.get("camera", {}))
        cam.update(ch.get("camera", {}))
        summaries.append(write_chamber(sim, out / sim.name, pf, cam))
        _print_measurement(summaries[-1]["measurement"])
        sims.append(sim)
        pfs.append(pf)
    plot_all(sims, out, pfs)
    write_viewer(sims, out / "viewer.html", name, max_particles=a.viewer_particles, fps=a.viewer_fps)
    (out / "summary.json").write_text(json.dumps(_jsonable({"name": name, "chambers": summaries}), indent=2))
    print(f"\nResults written to {out.resolve()}")
    print(f"  open {(out / 'viewer.html').resolve()} in a browser for the 3D replay")


def cmd_preflight(a):
    cfg = _load(a.config)
    for ch in cfg["chambers"]:
        _print_preflight(preflight(cfg, ch))


def cmd_sweep(a):
    pairs = [p.split(":") if ":" in p else (p, p) for p in a.pairs]
    v = np.logspace(math.log10(a.vmin), math.log10(a.vmax), a.nv)
    curves, thr = {}, {}
    radii = np.logspace(math.log10(a.rmin), math.log10(a.rmax), 12)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ma, mb in pairs:
        label = f"{ma}-{mb}"
        print(f"  {label} ...", flush=True)
        curves[label] = stick_probability_curve(ma, mb, a.radius, v, a.radius_sigma, a.gamma_sigma, a.samples)
        thr[label] = threshold_vs_radius(ma, mb, radii)
        p = curves[label]
        v50 = float(np.interp(0.5, p[::-1], v[::-1])) if p.max() >= 0.5 > p.min() else float("nan")
        print(f"     50% sticking speed at d={2e6 * a.radius:.0f} um: {v50 * 1e3:.2f} mm/s")
        for vi, pi in zip(v, p):
            rows.append(f"{label},{vi:.6g},{pi:.4f}")
    (out / "stick_probability.csv").write_text("pair,velocity_m_s,stick_probability\n" + "\n".join(rows))
    plot_sweep(curves, v, thr, out / "sticking_sweep.png",
               f"JKR-DEM sticking model, d = {2e6 * a.radius:.0f} um (gamma scatter {a.gamma_sigma}, size scatter {a.radius_sigma})")
    print(f"Wrote {out / 'sticking_sweep.png'} and stick_probability.csv")


def cmd_invert(a):
    g = infer_surface_energy(a.material, a.radius, a.vstick, a.partner)
    base = MATERIALS.get(a.material)
    print(f"Measured v_stick = {a.vstick * 1e3:.2f} mm/s for d = {2e6 * a.radius:.0f} um {a.material}"
          + (f" on {a.partner}" if a.partner else ""))
    print(f"  -> effective surface energy gamma = {g:.4g} J/m^2"
          + (f"  (database value {base.gamma:g}; ratio {g / base.gamma:.2f})" if base else ""))


def cmd_rebound_check(a):
    from .rebound_check import compare
    cfg = _load(a.config)
    out = Path(a.out)
    for ch in cfg["chambers"]:
        print(f"Cross-checking '{ch.get('name')}' against REBOUND ({a.duration:g} s, <= {a.max_grains} grains in total) ...", flush=True)
        r = compare(cfg, ch, a.duration, out, a.max_grains)
        b = r["ballistic_check"]
        print(f"  ballistic: {b['grains_compared']} grains, max position difference "
              f"{b['max_position_difference_m'] * 1e6:.3f} um after {b['t_s']:g} s")
        print(f"  {'':32s} {'DOLL-E':>12s} {'REBOUND':>12s}")
        for k in ("grain_grain_impacts", "grain_grain_sticks", "free_monomer_impacts",
                  "free_monomer_stick_fraction", "median_impact_speed_m_s", "wall_impacts", "wall_sticks",
                  "run_time_s"):
            x, y = r["dolle_engine"][k], r["rebound_engine"][k]
            f = (lambda z: "-" if z is None else (f"{z:.4g}" if isinstance(z, float) else str(z)))
            print(f"  {k:32s} {f(x):>12s} {f(y):>12s}")
    print(f"Wrote {out.resolve()}")


def cmd_verify(a):
    from .verification import run_all
    print("DOLL-E verification suite (analytic benchmarks and conservation laws)")
    n_ok = 0

    def show(r):
        nonlocal n_ok
        n_ok += r["passed"]
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['seconds']:6.1f}s  {r['name']}: {r['value']}  [{r['criterion']}]",
              flush=True)
    results = run_all(show)
    print(f"{n_ok}/{len(results)} checks passed")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))
    return 0 if n_ok == len(results) else 1


def cmd_materials(a):
    print(f"{'name':22s} {'rho':>7s} {'E [GPa]':>8s} {'nu':>5s} {'gamma':>7s} {'e_dry':>6s} {'phi[eV]':>7s}")
    for m in MATERIALS.values():
        print(f"{m.name:22s} {m.rho:7.0f} {m.E / 1e9:8.1f} {m.nu:5.2f} {m.gamma:7.3f} {m.e_dry:6.2f} {m.work_function:7.2f}")
        print(f"{'':22s} {m.note}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dolle_sim", description=f"DOLL-E dust simulator v{__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a full multi-particle simulation from a JSON config")
    r.add_argument("config")
    r.add_argument("--out")
    r.add_argument("--duration", type=float, help="override duration_s")
    r.add_argument("--viewer-particles", type=int, default=1500)
    r.add_argument("--viewer-fps", type=float, default=15)
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser("preflight", help="analytic design checks for a config (fast)")
    f.add_argument("config")
    f.set_defaults(fn=cmd_preflight)
    s = sub.add_parser("sweep", help="sticking probability vs impact speed for material pairs")
    s.add_argument("--pairs", nargs="+", default=["graphite:graphite", "amorphous_carbon:amorphous_carbon",
                                                   "silica:silica", "graphite:silica"])
    s.add_argument("--radius", type=float, default=50e-6)
    s.add_argument("--radius-sigma", type=float, default=0.05)
    s.add_argument("--gamma-sigma", type=float, default=0.3, help="log-normal scatter of surface energy")
    s.add_argument("--samples", type=int, default=150)
    s.add_argument("--vmin", type=float, default=1e-4)
    s.add_argument("--vmax", type=float, default=1.0)
    s.add_argument("--nv", type=int, default=60)
    s.add_argument("--rmin", type=float, default=0.5e-6)
    s.add_argument("--rmax", type=float, default=500e-6)
    s.add_argument("--out", default="results/sweep")
    s.set_defaults(fn=cmd_sweep)
    i = sub.add_parser("invert", help="infer effective surface energy from a measured sticking speed")
    i.add_argument("--material", required=True)
    i.add_argument("--partner")
    i.add_argument("--radius", type=float, default=50e-6)
    i.add_argument("--vstick", type=float, required=True, help="measured sticking velocity [m/s]")
    i.set_defaults(fn=cmd_invert)
    an = sub.add_parser("analyze", help="re-run the synthetic camera analysis on a results folder")
    an.add_argument("results")
    an.add_argument("--camera", help="JSON camera overrides, e.g. {\"fps\": 1000}")
    an.set_defaults(fn=cmd_analyze)
    rc = sub.add_parser("rebound-check", help="cross-check the engine against REBOUND on the same chamber")
    rc.add_argument("config")
    rc.add_argument("--duration", type=float, default=10.0)
    rc.add_argument("--max-grains", type=int, default=400)
    rc.add_argument("--out", default="results/rebound_check")
    rc.set_defaults(fn=cmd_rebound_check)
    v = sub.add_parser("verify", help="run the physics verification suite")
    v.add_argument("--json", help="also write the results to this JSON file")
    v.set_defaults(fn=cmd_verify)
    m = sub.add_parser("materials", help="list the material database")
    m.set_defaults(fn=cmd_materials)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

"""DOLL-E dust collision simulator -- command line interface.

    python -m dolle_sim run configs/flight_carbon_vs_silica.json
    python -m dolle_sim preflight configs/flight_carbon_vs_silica.json
    python -m dolle_sim sweep --pairs graphite:graphite silica:silica --radius 50e-6
    python -m dolle_sim invert --material graphite --radius 50e-6 --vstick 0.08
    python -m dolle_sim materials
    python -m dolle_sim analyze results/flight_carbon_vs_silica
    python -m dolle_sim verify         # the 26 model checks (verification.py)
    python -m dolle_sim sources        # where every part of the physics comes from
    python -m dolle_sim mission configs/mission_quarter_deck.json [--scenario kick] [--loads-only]
    python -m dolle_sim sizes --d10 90e-6 --d50 100e-6 --d90 112e-6 --weighting volume

The optional parts (the v0.6 full-contact engine and its checks, the REBOUND and LAMMPS cross-checks, grain sizes from
microscope images) have their own command line in extras/: python -m dolle_extras --help (extras/README.md).

Provenance: Command-line wrapper only (no physics).
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
              f"(energy estimate {g['v_stick_pp_energy_estimate_m_s'] * 1e3:.2f} mm/s), grain-wall {g['v_stick_wall_DEM_m_s'] * 1e3:.2f} mm/s")
        print(f"      launch {g['launch_speed_m_s'] * 1e3:.2f} mm/s, crossing time {g['chamber_crossing_time_s']:.2f} s, "
              f"~{g['expected_pp_collisions_total']:.0f} grain-grain collisions expected")
        print(f"      spin-down time {g['spin_down_time_s']:.3g} s, rms charge {g['rms_charge_e']:.3g} e, "
              f"Brownian speed {g['brownian_speed_m_s'] * 1e6:.2g} um/s"
              + (" (off)" if "brownian" in g.get("switched_off_forces", ()) else ""))
        print(f"      wall: v_stick {g['v_stick_wall_DEM_m_s'] * 1e3:.2f} mm/s (adhesion factor {g['wall_adhesion_factor']:g}), "
              f"vibration {g['wall_vibration_speed_m_s'] * 1e3:.1f} mm/s -> {100 * g['wall_capture_probability_per_hit']:.1f}% of wall hits stick")
        print("      acceleration budget (m/s^2, and in g; (off) = switched off in this config):")
        off = set(g.get("switched_off", ()))
        for k, v in sorted(g["acceleration_budget_m_s2"].items(), key=lambda kv: -kv[1]):
            print(f"        {k:34s} {v:10.3e}   {v / 9.80665:9.2e} g" + ("   (off)" if k in off else ""))
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


def cmd_verify(a):
    from .verification import run_all
    print("DOLL-E model checks (analytic benchmarks, conservation laws, independent models; also run in the browser)")
    n_ok = 0

    def show(r):
        nonlocal n_ok
        n_ok += r["passed"]
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['seconds']:6.1f}s  {r['name']}: {r['value']}  [{r['criterion']}]",
              flush=True)
    results = run_all(show)
    print(f"{n_ok}/{len(results)} checks passed")
    print("The optional v0.6 full-contact engine has its own checks: python -m dolle_extras verify (extras/README.md)")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))
    return 0 if n_ok == len(results) else 1


def cmd_materials(a):
    print(f"{'name':22s} {'rho':>7s} {'E [GPa]':>8s} {'nu':>5s} {'gamma':>7s} {'e_dry':>6s} {'phi[eV]':>7s}")
    for m in MATERIALS.values():
        print(f"{m.name:22s} {m.rho:7.0f} {m.E / 1e9:8.1f} {m.nu:5.2f} {m.gamma:7.3f} {m.e_dry:6.2f} {m.work_function:7.2f}")
        print(f"{'':22s} {m.note}")


def cmd_mission(a):
    return run_mission_cli(_load(a.config), a)


def run_mission_cli(mission: dict, a, runner=None):
    """The mission command for a loaded config. python -m dolle_extras mission calls it too, passing the v0.6
    engine's runner (dolle_sim itself only runs the v0.5 engine; mission.run_mission)."""
    from .mission import run_mission, launch_report, apply_scenario
    if a.grains:
        for t in mission["tubules"]:
            t["dust"]["count"] = int(a.grains)
    m = apply_scenario(mission, a.scenario)
    rep = launch_report(m)
    print(f"Launch loads ({m.get('scenario') or 'nominal'}): per-grain wall pull-off thresholds and the loads of each phase")
    for t in rep["tubules"]:
        th = t["threshold_g"]
        print(f"\n  {t['name']}: {t['dust']} on {t['wall']}, {t['orientation']}, r = {t['r_far_m'] * 100:.1f} cm; "
              f"pull-off thresholds p10 {th['p10']:.3g} g, median {th['median']:.3g} g, p90 {th['p90']:.3g} g")
        for ph in t["phases"]:
            print(f"    {ph['phase']:30s} T+{ph['t0']:6.1f}-{ph['t1']:6.1f} s  onto aft face up to {ph['max_press_onto_aft_g']:6.3g} g,"
                  f" off it up to {max(ph['max_pull_off_aft_g'], 0.0):7.3g} g"
                  f"  along it {ph['max_tangential_g']:7.3g} g  pulled off: {100 * ph['fraction_pulled_off_quasi_static']:5.1f}%"
                  f"  could slide: {100 * ph['fraction_could_slide']:5.1f}%")
        v = t["vibration_estimate"]
        print(f"    random-vibration estimate (Gaussian at the test level; see PROVENANCE.md): freed in the ground test "
              f"{100 * v['ground_random_test']:.1f}% (sine test {100 * v['ground_sine_test']:.1f}%), lift-off to burnout "
              f"{100 * v['flight_lift_off_to_burnout']:.1f}%, all {100 * v['all']:.1f}%")
    print("\n  'pulled off' and 'could slide' are this report's quasi-static estimate (pull-off normal to the aft face,"
          "\n  sliding along it). The v0.5 engine (the default) holds wall grains in place until they are pulled off;"
          "\n  the v0.6 engine in extras/ also resolves rolling, sliding and twisting: rolling starts at ~1 g of"
          "\n  sideways load for 100 um silica on clean glass (PROVENANCE.md), below the spin loads.")
    out = Path(a.out or Path("results") / "mission" / (a.scenario or "nominal"))
    if a.loads_only:
        out.mkdir(parents=True, exist_ok=True)
        (out / "launch_loads.json").write_text(json.dumps(_jsonable(rep), indent=2))
        print(f"\nWrote {out / 'launch_loads.json'}")
        return 0
    print(f"\nSimulating {len(m['tubules'])} tubules from T+{m.get('simulate_from_s', 0):g} s to the end of the run "
          f"(splashdown + 5 s unless simulate_to_s is set) ...")
    mission.setdefault("progress", True)
    if a.tubule is not None:
        from .mission import run_tubule
        path = run_tubule(mission, out, a.scenario, a.tubule, runner=runner)
        print(f"Tubule {a.tubule} done: final state in {path}")
        return 0
    if a.assemble:
        from .mission import assemble_mission
        summary, _ = assemble_mission(mission, out, a.scenario)
    else:
        summary, _ = run_mission(mission, out, a.scenario, jobs=a.jobs, runner=runner)
    for t in summary["tubules"]:
        print(f"  {t['tubule']}: {t['grains_released']} of {t['n_grains']} grains released at least once "
              f"({t['detachment_events']} release events: {t['detachments_by_phase']}); {t['free_at_power_off']} free at "
              f"power-off, {t['free_at_end']} at the end (T+{t['end_s']:.0f} s); "
              f"microgravity grain-grain impacts {t['ug_grain_grain_impacts']} ({t['ug_monomer_impacts']} single-grain, "
              f"{t['ug_monomer_sticks']} stuck)")
    print(f"\nResults written to {out.resolve()}")
    return 0


def cmd_sizes(a):
    from . import sizes
    if a.d10 and a.d50 and a.d90:
        f = sizes.lognormal_from_percentiles(a.d10, a.d50, a.d90, a.weighting)
        print(f"log-normal through {a.weighting}-weighted D10/D50/D90: count median {f['cmd_m']*1e6:.2f} um, "
              f"sigma_g {f['gsd']:.4f}, misfit (max |ln d - fit|) {f['lognormal_misfit_ln']:.3f}")
        mu, s_ = math.log(f["cmd_m"]), math.log(f["gsd"])
        print(f"  volume median {math.exp(mu + 3 * s_ * s_) * 1e6:.2f} um, Sauter d32 {math.exp(mu + 2.5 * s_ * s_) * 1e6:.2f} um")
        return 0
    print("give --d10 --d50 --d90 (grain sizes from images: python -m dolle_extras sizes-image)")
    return 1


def cmd_sources(a):
    from .provenance import as_text, as_markdown
    print(as_markdown() if a.markdown else as_text())


def add_mission_arguments(mi):
    """The arguments of the mission command (shared with python -m dolle_extras mission)."""
    mi.add_argument("config")
    mi.add_argument("--scenario")
    mi.add_argument("--out")
    mi.add_argument("--jobs", type=int, default=1, help="run tubules in parallel processes")
    mi.add_argument("--loads-only", action="store_true", help="only the analytic launch-load report")
    mi.add_argument("--grains", type=int, help="grains per tubule (overrides the config's example amount)")
    mi.add_argument("--tubule", type=int, help="run only this tubule (0-3) and save its final state (for a campaign "
                    "that schedules tubules one by one)")
    mi.add_argument("--assemble", action="store_true", help="write the outputs from the tubules' saved final states")
    return mi


MOVED_TO_EXTRAS = {"rebound-check": "rebound-check", "lammps-check": "lammps-check"}
"""Commands that moved to extras/ in v0.7, with their new names there (python -m dolle_extras <name>)."""
MOVED_MISSION_OPTIONS = ("--checkpoint-at", "--resume-from", "--resume-at", "--autosave-min")


def moved_command(argv) -> str | None:
    """The dolle_extras command that replaces a v0.6 command line, or None if the command line is a dolle_sim one."""
    if not argv:
        return None
    if argv[0] in MOVED_TO_EXTRAS:
        return MOVED_TO_EXTRAS[argv[0]]
    if argv[0] == "sizes" and any(x == "--image" or x.startswith("--image=") for x in argv[1:]):
        return "sizes-image"
    if argv[0] == "mission" and any(x.split("=")[0] in MOVED_MISSION_OPTIONS for x in argv[1:]):
        return "mission"
    return None


def main(argv=None):
    moved = moved_command(sys.argv[1:] if argv is None else list(argv))
    if moved:
        print(f"moved to extras in v0.7: python -m dolle_extras {moved} ... (or, without installing, python "
              f"extras/dolle_extras {moved} ...); see extras/README.md", file=sys.stderr)
        return 2
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
    v = sub.add_parser("verify", help="run the model checks (verification.py)")
    v.add_argument("--json", help="also write the results to this JSON file")
    v.set_defaults(fn=cmd_verify)
    m = sub.add_parser("materials", help="list the material database")
    m.set_defaults(fn=cmd_materials)
    mi = sub.add_parser("mission", help="all tubules through the whole flight on one clock (launch to splashdown)")
    add_mission_arguments(mi)
    mi.set_defaults(fn=cmd_mission)
    sz = sub.add_parser("sizes", help="size distributions: a log-normal through D10/D50/D90")
    sz.add_argument("--d10", type=float)
    sz.add_argument("--d50", type=float)
    sz.add_argument("--d90", type=float)
    sz.add_argument("--weighting", default="volume", choices=["number", "volume"])
    sz.set_defaults(fn=cmd_sizes)
    so = sub.add_parser("sources", help="where every part of the physics comes from, how it is checked, what is assumed")
    so.add_argument("--markdown", action="store_true")
    so.set_defaults(fn=cmd_sources)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

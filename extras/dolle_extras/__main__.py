"""Command line of the optional extras (the v0.6 engine, the cross-checks, grain sizes from images).

    python -m dolle_extras verify [--json F]                 the 19 checks of the v0.6 full-contact engine (~5 min)
    python -m dolle_extras mission CFG [mission options] [--checkpoint-at T ...] [--resume-from DIR --resume-at T]
                                   [--autosave-min M]        a mission with the v0.6 engine (engine "full")
    python -m dolle_extras rebound-check CFG [--duration S --max-grains N --out DIR]   needs pip install rebound
    python -m dolle_extras lammps-check [--out DIR]          needs pip install lammps
    python -m dolle_extras sizes-image IMG ... --um-per-px X [--dark-on-light | --light-on-dark] [--no-split] [--out F]
                                                             needs pip install scikit-image
    python -m dolle_extras sources [--markdown] [--write]    provenance of the extras (--write: extras/PROVENANCE.md)

Without installing the repository, run it from the extras/ folder (cd extras), or as python extras/dolle_extras ...
from the repository folder. The main simulator's commands are python -m dolle_sim --help.

Provenance: command-line wrapper only (no physics); the bodies of rebound-check, lammps-check and sizes-image moved
unchanged from dolle_sim/__main__.py (v0.6.2).
"""
from __future__ import annotations

if __package__ in (None, ""):          # run as `python extras/dolle_extras ...`: make the package importable
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == _here:
        sys.path.pop(0)                 # (the package folder itself must not be on the path)
    sys.path[:0] = [str(_here.parent), str(_here.parents[1])]
    from dolle_extras.__main__ import main
    sys.exit(main())

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load(path):
    return json.loads(Path(path).read_text())


def cmd_verify(a):
    from . import verify_full
    print("DOLL-E v0.6 full-contact engine checks (dem.py, statics.py, slaved.py, loads.py, vibration.py, engine.py)")
    n_ok = 0

    def show(r):
        nonlocal n_ok
        n_ok += r["passed"]
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['seconds']:6.1f}s  {r['name']}: {r['value']}  [{r['criterion']}]",
              flush=True)
    results = verify_full.run_all(show)
    print(f"{n_ok}/{len(results)} checks passed")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))
    return 0 if n_ok == len(results) else 1


def cmd_mission(a):
    from dolle_sim.__main__ import run_mission_cli
    from .full_mission import run_args
    mission = _load(a.config)
    mission.setdefault("engine", "full")
    if a.checkpoint_at:
        mission["checkpoints_s"] = list(a.checkpoint_at)
    if a.resume_from:
        if a.resume_at is None:
            raise SystemExit("--resume-from needs --resume-at")
        mission["resume_from"], mission["resume_at_s"] = a.resume_from, a.resume_at
        print(f"Continuing from the checkpoints at T+{a.resume_at:g} s in {a.resume_from} (verified identical up to then)")
    if a.autosave_min:
        mission["autosave_s"] = 60.0 * a.autosave_min
    return run_mission_cli(mission, a, runner=run_args if mission["engine"] == "full" else None)


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
                  "free_monomer_stick_fraction", "median_impact_speed_m_s", "free_monomer_median_speed_m_s",
                  "wall_impacts", "wall_sticks", "run_time_s"):
            x, y = r["dolle_engine"].get(k), r["rebound_engine"].get(k)
            f = (lambda z: "-" if z is None else (f"{z:.4g}" if isinstance(z, float) else str(z)))
            print(f"  {k:32s} {f(x):>12s} {f(y):>12s}")
    print(f"Wrote {out.resolve()}")


def cmd_lammps(a):
    from .lammps_check import compare
    import runpy
    tool = Path(__file__).resolve().parents[1] / "tools" / "lammps_contact_check.py"
    if tool.exists():
        print("Full contact model (JKR, sliding, rolling, twisting) vs LAMMPS GRANULAR, same coefficients and step:")
        mod = runpy.run_path(str(tool))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        mod["main"](str(Path(a.out) / "lammps_contact_check.json"))
    res = compare(Path(a.out))
    for r in res["collisions"]:
        print(f"{r['material']}: undamped sticking threshold LAMMPS {r['lammps_threshold_m_s']*1e3:.4f} mm/s vs DOLL-E "
              f"(same break criterion) {r['dolle_threshold_force_break_m_s']*1e3:.4f} mm/s "
              f"({100 * r['rel_diff_same_criterion']:+.2f}%); DOLL-E default criterion {r['dolle_threshold_default_m_s']*1e3:.4f} mm/s")
        for e in r["restitution"]:
            print(f"    v = {e['v_m_s']*1e3:.3f} mm/s: e LAMMPS {e['lammps_e']:.4f}, DOLL-E {e['dolle_e']:.4f}")
    for p_ in res["pulloff"]:
        print(f"pull-off {p_['grain']} on {p_['wall']}: LAMMPS {p_['lammps_detach_force_N']:.4e} N vs 1.5 pi w R "
              f"{p_['jkr_pulloff_N']:.4e} N ({100 * p_['rel_diff']:+.2f}%; checked every {100 * p_['detection_step_rel']:.4f}% of F_c)")
    print(f"Wrote {Path(a.out).resolve() / 'lammps_check.json'}")
    return 0


def cmd_sizes_image(a):
    from dolle_sim import sizes
    from .sizing import measure_image
    d_all = []
    for img in a.image:
        m = measure_image(img, a.um_per_px, dark_on_light=a.dark_on_light, split_touching=not a.no_split)
        d_all.append(m["diameters_m"])
        print(f"{img}: {m['n']} grains measured")
    d = np.concatenate(d_all) if d_all else np.zeros(0)
    if len(d):
        st = sizes.summary(d)
        print(f"all images: {len(d)} grains; number D10/D50/D90 = {st['number_d10_m']*1e6:.1f} / "
              f"{st['number_d50_m']*1e6:.1f} / {st['number_d90_m']*1e6:.1f} um; volume D10/D50/D90 = "
              f"{st['volume_d10_m']*1e6:.1f} / {st['volume_d50_m']*1e6:.1f} / {st['volume_d90_m']*1e6:.1f} um")
    if a.out:
        np.savetxt(a.out, d, header="area-equivalent diameter [m]")
        print(f"wrote {a.out} ({len(d)} grains from {len(a.image)} image(s); use as size_distribution "
              "{'type': 'measured', 'csv': ...})")
    return 0


def cmd_sources(a):
    from . import provenance
    if a.write:
        provenance.write()
        print("wrote extras/PROVENANCE.md")
        return 0
    print(provenance.as_markdown() if a.markdown else provenance.as_text())
    return 0


def main(argv=None):
    from dolle_sim import __version__
    from dolle_sim.__main__ import add_mission_arguments
    p = argparse.ArgumentParser(prog="dolle_extras", description=f"DOLL-E optional extras (dolle_sim v{__version__})")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="run the checks of the v0.6 full-contact engine (verify_full.py)")
    v.add_argument("--json", help="also write the results to this JSON file")
    v.set_defaults(fn=cmd_verify)
    mi = sub.add_parser("mission", help="a mission with the v0.6 full-contact engine (checkpoints, branches, autosave)")
    add_mission_arguments(mi)
    mi.add_argument("--checkpoint-at", type=float, nargs="*", help="flight times [s] at which to save each tubule's "
                    "state (to <out>/checkpoints) for other scenarios to continue from")
    mi.add_argument("--resume-from", help="a checkpoints directory of another scenario's run: continue from it "
                    "(verified: the scenarios must be identical up to --resume-at)")
    mi.add_argument("--resume-at", type=float, help="the checkpoint time to continue from [s]")
    mi.add_argument("--autosave-min", type=float, help="save each tubule's state every this many minutes of computing "
                    "(to <out>/autosave); a run started again continues from it, bit-for-bit")
    mi.set_defaults(fn=cmd_mission)
    rc = sub.add_parser("rebound-check", help="cross-check the engine against REBOUND on the same chamber")
    rc.add_argument("config")
    rc.add_argument("--duration", type=float, default=10.0)
    rc.add_argument("--max-grains", type=int, default=400)
    rc.add_argument("--out", default="results/rebound_check")
    rc.set_defaults(fn=cmd_rebound_check)
    lc = sub.add_parser("lammps-check", help="cross-check the JKR contact model against LAMMPS (pip install lammps)")
    lc.add_argument("--out", default="results/lammps_check")
    lc.set_defaults(fn=cmd_lammps)
    sz = sub.add_parser("sizes-image", help="grain diameters from microscope images (pip install scikit-image)")
    sz.add_argument("image", nargs="+")
    sz.add_argument("--um-per-px", type=float, required=True, help="the calibration, e.g. from a stage micrometer")
    sz.add_argument("--dark-on-light", dest="dark_on_light", action="store_true", default=None,
                    help="grains darker than the background (default: grains are the minority phase)")
    sz.add_argument("--light-on-dark", dest="dark_on_light", action="store_false",
                    help="grains brighter than the background (e.g. dark field)")
    sz.add_argument("--no-split", action="store_true", help="do not split touching grains (watershed)")
    sz.add_argument("--out", help="write the diameters [m] to this text file")
    sz.set_defaults(fn=cmd_sizes_image)
    so = sub.add_parser("sources", help="provenance of the extras (extras/PROVENANCE.md)")
    so.add_argument("--markdown", action="store_true")
    so.add_argument("--write", action="store_true", help="write extras/PROVENANCE.md")
    so.set_defaults(fn=cmd_sources)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

"""Build the GitHub Pages site in docs/.

    python tools/build_site.py            # copy the package + data into docs/
    python tools/build_site.py --replays  # also rebuild the web replays from results/
    python tools/build_site.py --figures  # also rebuild docs/figures from results/

The site is static: docs/app runs the Python package in the browser with Pyodide,
docs/replays holds down-sampled 3D replays of the desktop flight runs. (The 3-D page docs/sim3d is built by
tools/make_sim3d_replays.py, its replays, and tools/build_sim3d.py, the dolle_core copies of its live sandbox.)
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PKG = ROOT / "dolle_sim"
WEB_FILES = ["__init__.py", "analysis.py", "atmosphere.py", "contact.py", "design.py", "electrostatics.py", "environment.py",
             "flight.py", "materials.py", "mission.py", "results.py", "simulation.py", "sizes.py", "verification.py",
             "webapi.py", "viewer_template.html"]


def copy_package():
    dst = DOCS / "app" / "py" / "dolle_sim"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in WEB_FILES:
        shutil.copy2(PKG / f, dst / f)
    sys.path.insert(0, str(ROOT))
    from dolle_sim import __version__
    (DOCS / "app" / "py" / "manifest.json").write_text(json.dumps({"version": __version__, "files": WEB_FILES}, indent=1))
    print(f"copied {len(WEB_FILES)} package files (v{__version__})")


def copy_data():
    (DOCS / "data").mkdir(exist_ok=True)
    rb = {}
    for p in sorted((ROOT / "results" / "rebound_check").glob("rebound_check_*.json")):
        d = json.loads(p.read_text())
        rb[d["chamber"]] = d
    if rb:
        (DOCS / "data" / "rebound_check.json").write_text(json.dumps(rb, indent=1))
        print("wrote docs/data/rebound_check.json")
    shutil.copy2(ROOT / "configs" / "mission_quarter_deck.json", DOCS / "data" / "mission_config.json")
    lc = ROOT / "results" / "lammps_check" / "lammps_check.json"
    if lc.exists():
        shutil.copy2(lc, DOCS / "data" / "lammps_check.json")
        print("wrote docs/data/lammps_check.json")
    ms = {}
    for p in sorted((ROOT / "results" / "mission").glob("*/mission_summary.json")):
        d = json.loads(p.read_text())
        full = "dem_steps" in (d["tubules"][0].get("events") or {})
        ms[p.parent.name] = {"scenario": d.get("scenario") or "nominal", "gas": d.get("gas"),
                             "engine": "v0.6 full contact" if full else "v0.5 event-driven",
                             "tubules": d["tubules"], "launch_loads": d["launch_loads"]["tubules"]}
    if ms:
        (DOCS / "data" / "mission_results.json").write_text(json.dumps(ms, indent=1))
        print(f"wrote docs/data/mission_results.json ({len(ms)} scenarios)")
    (DOCS / ".nojekyll").write_text("")
    write_provenance()


def write_provenance():
    """PROVENANCE.md and the Sources tab data, both generated from dolle_sim/provenance.py."""
    sys.path.insert(0, str(ROOT))
    from dolle_sim import provenance
    counts = provenance.line_counts(PKG)
    (DOCS / "data" / "provenance.json").write_text(json.dumps(provenance.as_dict(counts), indent=1), encoding="utf-8")
    (ROOT / "PROVENANCE.md").write_text(provenance.as_markdown(counts), encoding="utf-8")
    print("wrote PROVENANCE.md and docs/data/provenance.json")
    # keep the number of verification checks quoted on the pages in step with verification.py and, for the
    # optional v0.6 engine whose checks "need the desktop" (the page says so), extras/dolle_extras/verify_full.py
    n_model = provenance.n_checks()
    vf = ROOT / "extras" / "dolle_extras" / "verify_full.py"
    n_engine = len(set(re.findall(r'_res\("(\w+)"', vf.read_text(encoding="utf-8")))) if vf.exists() else 0
    n = n_model + n_engine
    page = DOCS / "index.html"
    html = page.read_text(encoding="utf-8")
    html = re.sub(r"<!--checks-->.*?<!--/checks-->", f"<!--checks-->{n} checks<!--/checks-->", html)
    html = re.sub(r"<!--nmodel-->.*?<!--/nmodel-->", f"<!--nmodel-->{n_model}<!--/nmodel-->", html)
    page.write_text(html, encoding="utf-8")
    page = DOCS / "app" / "index.html"        # the browser runs the model checks only
    html = page.read_text(encoding="utf-8")
    page.write_text(re.sub(r"\b\d+ tests compare", f"{n_model} tests compare", html), encoding="utf-8")


def _decimate_viewer(src: Path, dst: Path, title: str, max_particles=500, frame_step=2, t_max=60.0):
    html = src.read_text(encoding="utf-8")
    m = re.search(r"const DATA = (\{.*?\});\nconst MAT_COLORS", html, re.S)
    data = json.loads(m.group(1))
    for ch in data["chambers"]:
        n, F = ch["n_shown"], ch["n_frames"]
        pos = np.frombuffer(base64.b64decode(ch["pos_b64"]), "<u2").reshape(F, n, 3)
        agg = np.frombuffer(base64.b64decode(ch["agg_b64"]), np.uint8).reshape(F, n)
        spin = np.frombuffer(base64.b64decode(ch["spin_b64"]), np.uint8).reshape(F, n)
        t = np.asarray(ch["t"])
        fsel = np.flatnonzero(t <= t_max)[::frame_step]
        # evenly spaced subset across all shown grains (grain order follows launch groups)
        sel = np.unique(np.linspace(0, n - 1, min(n, max_particles)).astype(int))
        k = len(sel)
        ch["pos_b64"] = base64.b64encode(np.ascontiguousarray(pos[np.ix_(fsel, sel)]).tobytes()).decode()
        ch["agg_b64"] = base64.b64encode(np.ascontiguousarray(agg[np.ix_(fsel, sel)]).tobytes()).decode()
        ch["spin_b64"] = base64.b64encode(np.ascontiguousarray(spin[np.ix_(fsel, sel)]).tobytes()).decode()
        ch["t"] = [ch["t"][i] for i in fsel]
        ch["n_frames"], ch["n_shown"] = len(fsel), k
        ch["radius_um"] = [ch["radius_um"][i] for i in sel]
        ch["material"] = [ch["material"][i] for i in sel]
        # keep every grain-grain impact and wall stick; drop the (very many) wall-bounce flashes
        ch["events"] = [e for e in ch["events"] if e[0] <= t_max and not (e[3] == 1 and e[2] == 0)]
    sys.path.insert(0, str(ROOT))
    from dolle_sim.results import render_viewer
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(render_viewer(data, title), encoding="utf-8")
    print(f"wrote {dst.relative_to(ROOT)} ({dst.stat().st_size / 1e6:.1f} MB)")


def build_replays():
    runs = [("flight_carbon_vs_silica", "mission.html", "Simulated flight: graphite vs silica (first 60 s)")]
    for run, name, title in runs:
        src = ROOT / "results" / run / "viewer.html"
        if src.exists():
            _decimate_viewer(src, DOCS / "replays" / name, title)
        else:
            print(f"skip {name}: {src} not found (run the simulation first)")


def build_figures():
    """README / site figures from the saved results (run the simulations, sweep and rebound-check first)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = DOCS / "figures"
    fig_dir.mkdir(exist_ok=True)
    R = ROOT / "results"
    for src, dst in [("flight_carbon_vs_silica/sticking_measurement.png", "sticking_measurement_isotropic.png"),
                     ("flight_carbon_vs_silica/timeseries.png", "timeseries_isotropic.png"),
                     ("sweep/sticking_sweep.png", "sticking_sweep.png"),
                     ("rebound_check/rebound_check_carbon.png", "rebound_check_carbon.png")]:
        if (R / src).exists():
            shutil.copy2(R / src, fig_dir / dst)
            print(f"copied {dst}")
        else:
            print(f"skip {dst}: results/{src} not found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", action="store_true", help="rebuild the web replays from results/")
    ap.add_argument("--figures", action="store_true", help="rebuild docs/figures from results/")
    a = ap.parse_args()
    copy_package()
    copy_data()
    if a.replays:
        build_replays()
    if a.figures:
        build_figures()

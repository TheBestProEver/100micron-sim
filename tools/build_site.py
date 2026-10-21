"""Build the GitHub Pages site in docs/.

    python tools/build_site.py            # copy the package + data into docs/
    python tools/build_site.py --replays  # also rebuild the web replays from results/

The site is static: docs/app runs the Python package in the browser with Pyodide,
docs/replays holds down-sampled 3D replays of the desktop flight runs.
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
WEB_FILES = ["__init__.py", "analysis.py", "contact.py", "design.py", "electrostatics.py", "environment.py",
             "materials.py", "results.py", "simulation.py", "verification.py", "webapi.py", "viewer_template.html"]


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
    (DOCS / ".nojekyll").write_text("")


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
    runs = [("rocksat_x_carbon_vs_silica", "mission.html", "Mission run: graphite vs silica (first 60 s)"),
            ("colliding_clouds", "clouds.html", "Colliding clouds: graphite vs silica (60 s)")]
    for run, name, title in runs:
        src = ROOT / "results" / run / "viewer.html"
        if src.exists():
            _decimate_viewer(src, DOCS / "replays" / name, title)
        else:
            print(f"skip {name}: {src} not found (run the simulation first)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", action="store_true")
    a = ap.parse_args()
    copy_package()
    copy_data()
    if a.replays:
        build_replays()

"""Copy the readable core's python files into the 3-D page, for its live sandbox (Pyodide in the browser).

    python tools/build_sim3d.py              copy dolle_core/*.py into docs/sim3d/py/dolle_core/ and write
                                             docs/sim3d/py/manifest.json (well under a second)
    python tools/build_sim3d.py --check      only check that the copies are identical to dolle_core/ (exit 1 if not)
    python tools/build_sim3d.py --replays    also rewrite the recorded replays (tools/make_sim3d_replays.py, ~15 s)

The page (docs/sim3d/index.html) has no build step: it loads three.js and Pyodide from a CDN, and its sandbox
fetches the files listed in the manifest and imports dolle_core from them, so the browser runs exactly the code in
this repository.  The notebook helpers dolle_core/nb_*.py are not copied: they need matplotlib and the full
simulator, and the sandbox does not use them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "dolle_core"
DEST = ROOT / "docs" / "sim3d" / "py"


def core_files() -> list[str]:
    """The python files of dolle_core the browser needs: every module except the notebook helpers nb_*.py."""
    return sorted(p.name for p in CORE.glob("*.py") if not p.name.startswith("nb_"))


def stale() -> list[str]:
    """Files whose copy in docs/sim3d/py/dolle_core/ is missing, different or not in the manifest (or left over)."""
    out = []
    man_path = DEST / "manifest.json"
    listed = json.loads(man_path.read_text(encoding="utf-8")).get("files", []) if man_path.exists() else []
    for f in core_files():
        dst = DEST / "dolle_core" / f
        if not dst.exists() or dst.read_bytes() != (CORE / f).read_bytes() or f not in listed:
            out.append(f)
    extra = {p.name for p in (DEST / "dolle_core").glob("*.py")} - set(core_files()) if DEST.exists() else set()
    return out + sorted(extra)


def copy_core() -> dict:
    """Copy the files (byte for byte) and write the manifest; returns the manifest."""
    pkg = DEST / "dolle_core"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    files = core_files()
    for f in files:
        shutil.copyfile(CORE / f, pkg / f)
    manifest = {"made_by": "python tools/build_sim3d.py", "package": "dolle_core", "files": files,
                "note": "byte-for-byte copies of dolle_core/*.py (without the notebook helpers nb_*.py)"}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only check that the copies are current")
    ap.add_argument("--replays", action="store_true", help="also run tools/make_sim3d_replays.py")
    a = ap.parse_args()
    if a.check:
        bad = stale()
        print("docs/sim3d/py is current" if not bad else f"stale copies: {bad} (run python tools/build_sim3d.py)")
        return 1 if bad else 0
    man = copy_core()
    size = sum((DEST / "dolle_core" / f).stat().st_size for f in man["files"])
    print(f"copied {len(man['files'])} files ({size / 1e3:.0f} kB) to {DEST.relative_to(ROOT).as_posix()}/dolle_core")
    if a.replays:
        return subprocess.call([sys.executable, str(ROOT / "tools" / "make_sim3d_replays.py")])
    return 0


if __name__ == "__main__":
    sys.exit(main())

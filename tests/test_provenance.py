"""The provenance table must stay true to the code; and the main package never needs the optional extras."""
import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from dolle_sim import provenance

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "dolle_sim"
EXTRAS = ROOT / "extras" / "dolle_extras"
MOVED = ("engine", "dem", "statics", "slaved", "roll1", "loads", "vibration", "verify_full", "sizing", "rebound_check",
         "lammps_check")                   # moved from dolle_sim to extras/dolle_extras in v0.7
needs_extras = pytest.mark.skipif(not EXTRAS.is_dir(), reason="the optional extras/ folder is not in this copy")
IMPORTS_REBOUND = re.compile(r"^\s*(import\s+rebound\b|from\s+rebound\b)|__import__\(\s*['\"]rebound['\"]|"
                             r"import_module\(\s*['\"]rebound['\"]", re.M)
IMPORTS_CHECK = re.compile(r"^\s*(from\s+\S*rebound_check\s+import|import\s+\S*rebound_check)", re.M)
IMPORTS_LAMMPS = re.compile(r"^\s*(import\s+lammps\b|from\s+lammps\b)", re.M)
IMPORTS_LCHECK = re.compile(r"^\s*(from\s+\S*lammps_check\s+import|import\s+\S*lammps_check)", re.M)


def _items():
    return [c for s in provenance.SECTIONS for c in s["items"]]


def _py_files():
    for d in (PKG, ROOT / "tools", ROOT / "tests", ROOT / "docs" / "app" / "py", ROOT / "extras"):
        yield from d.rglob("*.py")


@needs_extras
def test_rebound_is_only_imported_by_the_cross_check():
    users = sorted(str(f.relative_to(ROOT)).replace("\\", "/") for f in _py_files()
                   if IMPORTS_REBOUND.search(f.read_text(encoding="utf-8")))
    assert users == ["extras/dolle_extras/rebound_check.py"]
    # the simulator never imports the cross-check (only the extras' command-line "rebound-check" command does)
    importers = sorted(f.name for f in PKG.glob("*.py") if IMPORTS_CHECK.search(f.read_text(encoding="utf-8")))
    assert importers == []
    importers = sorted(f.name for f in EXTRAS.glob("*.py") if IMPORTS_CHECK.search(f.read_text(encoding="utf-8")))
    assert importers == ["__main__.py"]


@needs_extras
def test_lammps_is_only_imported_by_its_cross_check():
    users = sorted(str(f.relative_to(ROOT)).replace("\\", "/") for f in _py_files()
                   if IMPORTS_LAMMPS.search(f.read_text(encoding="utf-8")))
    # the cross-check, and the timing benchmark in tools/ (not part of the simulator)
    assert users == ["extras/dolle_extras/lammps_check.py", "extras/tools/lammps_benchmark.py",
                     "extras/tools/lammps_contact_check.py"]
    importers = sorted(f.name for f in PKG.glob("*.py") if IMPORTS_LCHECK.search(f.read_text(encoding="utf-8")))
    assert importers == []
    importers = sorted(f.name for f in EXTRAS.glob("*.py") if IMPORTS_LCHECK.search(f.read_text(encoding="utf-8")))
    assert importers == ["__main__.py"]


def test_simulator_runs_without_rebound():
    """Run a short simulation in a fresh interpreter in which `import rebound` and `import lammps` fail."""
    code = (
        "import sys; sys.modules['rebound'] = None; sys.modules['lammps'] = None\n"
        "from dolle_sim.simulation import ChamberSim\n"
        "from dolle_sim import webapi, verification, design, analysis\n"
        "cfg = {'dt_s': 4e-3, 'duration_s': 0.2, 'environment': {'preset': 'ideal'}}\n"
        "ch = {'name': 't', 'size_m': [0.01] * 3, 'particles': [{'material': 'silica', 'count': 50,"
        " 'radius_m': 5e-5, 'velocity': {'speed_mean_m_s': 0.01}}]}\n"
        "s = ChamberSim(cfg, ch, 0).run(progress=False)\n"
        "assert 'rebound' not in sys.modules or sys.modules['rebound'] is None\n"
        "print('ok', s.n_events['wall_bounce'])\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.startswith("ok"), r.stderr


def test_rebound_relations_are_as_stated():
    rel = [c["rebound"] for c in _items()]
    assert set(rel) <= set(provenance.REBOUND_RELATIONS)
    assert rel.count("uses REBOUND") == 1


def test_every_named_check_exists():
    ids = set(re.findall(r'_res\("(\w+)"', (PKG / "verification.py").read_text(encoding="utf-8")))
    assert len(ids) == 26 == provenance.n_checks()
    for c in _items():
        assert c["origin"] in provenance.ORIGINS, c["component"]
        for k in c["checks"]:
            assert k in ids, (c["component"], k)


def test_every_named_code_location_exists():
    for c in _items():
        for part in c["code"].split(";"):
            m = re.match(r"\s*([\w./]+\.py)(?::\s*(.*))?", part.split(" (")[0])
            assert m, c["code"]
            f = ROOT / m.group(1) if m.group(1).startswith("extras/") else PKG / m.group(1)
            if m.group(1).startswith("extras/") and not EXTRAS.is_dir():
                continue                                        # a copy without the optional extras/ folder
            assert f.exists(), c["code"]
            src = f.read_text(encoding="utf-8")
            for name in re.findall(r"\b_?[A-Za-z]\w*\b", m.group(2) or ""):
                if name in ("wall", "impacts", "PRESETS"):
                    continue
                assert re.search(rf"(def|class)\s+{name}\b", src), (c["code"], name)


def test_generated_files_are_up_to_date():
    """PROVENANCE.md and the website's copy must match dolle_sim/provenance.py (run tools/build_site.py)."""
    counts = provenance.line_counts(PKG)
    assert (ROOT / "PROVENANCE.md").read_text(encoding="utf-8") == provenance.as_markdown(counts)
    import json
    assert json.loads((ROOT / "docs" / "data" / "provenance.json").read_text(encoding="utf-8")) == \
        json.loads(json.dumps(provenance.as_dict(counts)))


def _extras_imports(path):
    """(line, enclosing function) of every import of dolle_extras in a file."""
    found = []

    def walk(node, func):
        for ch in ast.iter_child_nodes(node):
            f = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
            names = []
            if isinstance(ch, ast.Import):
                names = [a.name for a in ch.names]
            elif isinstance(ch, ast.ImportFrom):
                names = [ch.module or ""]
            if any(n.split(".")[0] == "dolle_extras" for n in names):
                found.append((ch.lineno, f))
            walk(ch, f)
    walk(ast.parse(path.read_text(encoding="utf-8")), None)
    return found


def test_dolle_sim_never_imports_extras():
    """Dependency rule: dolle_extras may use dolle_sim, never the reverse, not even lazily or by name; the main test
    suite imports dolle_extras only through pytest.importorskip (so it runs in a copy without extras/)."""
    for f in sorted(PKG.glob("*.py")):
        assert _extras_imports(f) == [], f.name
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"import_module\(|__import__\(|['\"]dolle_extras", text), f.name
    assert not (PKG / "_moved.py").exists()
    for m in MOVED:                                # the moved modules exist only in extras
        assert not (PKG / f"{m}.py").exists(), m
        assert (EXTRAS / f"{m}.py").exists() or not EXTRAS.is_dir(), m
    for f in sorted((ROOT / "tests").glob("*.py")):
        assert _extras_imports(f) == [], f.name    # (importorskip("dolle_extras...") is a call, not an import)
    for f in sorted((ROOT / "dolle_core").glob("*.py")):     # the readable core's physics never needs the simulators
        if not f.name.startswith("nb_"):
            text = f.read_text(encoding="utf-8")
            assert _extras_imports(f) == [] and not re.search(r"^\s*(from|import) dolle_sim", text, re.M), f.name


def test_main_package_runs_without_extras():
    """With dolle_extras and every moved module unavailable: a short simulation, a v0.5 mission and the command line
    run; engine "full" names the dolle_extras command, and so do the moved commands."""
    code = (
        "import sys, json, tempfile, pathlib\n"
        "sys.modules['dolle_extras'] = None\n"
        f"for m in {MOVED!r}: sys.modules['dolle_extras.' + m] = None\n"
        "from dolle_sim.simulation import ChamberSim\n"
        "from dolle_sim import webapi, verification, design, analysis, mission, __main__\n"
        "cfg = {'dt_s': 4e-3, 'duration_s': 0.2, 'environment': {'preset': 'ideal'}}\n"
        "ch = {'name': 't', 'size_m': [0.01] * 3, 'particles': [{'material': 'silica', 'count': 30,"
        " 'radius_m': 5e-5, 'velocity': {'speed_mean_m_s': 0.01}}]}\n"
        "ChamberSim(cfg, ch, 0).run(progress=False)\n"
        "m = json.loads(pathlib.Path('configs/mission_quarter_deck.json').read_text())\n"
        "m['tubules'] = m['tubules'][:1]; m['tubules'][0]['dust']['count'] = 10; m['simulate_to_s'] = 1.0\n"
        "out = pathlib.Path(tempfile.mkdtemp())\n"
        "summary, sims = mission.run_mission(m, out, write_outputs=False)\n"
        "assert type(sims[0]).__name__ == 'ChamberSim', type(sims[0])\n"
        "m['engine'] = 'full'\n"
        "try:\n"
        "    mission.run_mission(m, out, write_outputs=False)\n"
        "    raise SystemExit('engine full ran without extras')\n"
        "except ValueError as e:\n"
        "    assert 'python -m dolle_extras mission' in str(e), e\n"
        "assert __main__.main(['rebound-check', 'x.json']) == 2 and __main__.main(['sizes', '--image', 'a.png']) == 2\n"
        "try:\n"
        "    __main__.main(['verify', '--help'])\n"
        "except SystemExit as e:\n"
        "    assert e.code == 0\n"
        "assert not any(k.startswith('dolle_extras') and sys.modules[k] is not None for k in sys.modules)\n"
        "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-2000:] + r.stderr[-3000:]


@needs_extras
def test_pyproject_installs_the_extras_from_their_folder():
    """One pip install (also Colab's) gives dolle_sim, dolle_core and dolle_extras; plain pytest runs the main suite."""
    tomllib = pytest.importorskip("tomllib")       # python 3.11+ (pyproject.toml allows 3.10)
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    st = cfg["tool"]["setuptools"]
    assert set(st["packages"]) == {"dolle_sim", "dolle_core", "dolle_extras"}
    assert st["package-dir"] == {"dolle_extras": "extras/dolle_extras"} and (ROOT / "extras" / "dolle_extras").is_dir()
    assert cfg["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    import dolle_sim
    assert cfg["project"]["version"] == dolle_sim.__version__

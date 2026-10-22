"""The provenance table must stay true to the code."""
import re
import subprocess
import sys
from pathlib import Path

from dolle_sim import provenance

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "dolle_sim"
IMPORTS_REBOUND = re.compile(r"^\s*(import\s+rebound\b|from\s+rebound\b)|__import__\(\s*['\"]rebound['\"]|"
                             r"import_module\(\s*['\"]rebound['\"]", re.M)
IMPORTS_CHECK = re.compile(r"^\s*(from\s+\S*rebound_check\s+import|import\s+\S*rebound_check)", re.M)
IMPORTS_LAMMPS = re.compile(r"^\s*(import\s+lammps\b|from\s+lammps\b)", re.M)
IMPORTS_LCHECK = re.compile(r"^\s*(from\s+\S*lammps_check\s+import|import\s+\S*lammps_check)", re.M)


def _items():
    return [c for s in provenance.SECTIONS for c in s["items"]]


def _py_files():
    for d in (PKG, ROOT / "tools", ROOT / "tests", ROOT / "docs" / "app" / "py"):
        yield from d.rglob("*.py")


def test_rebound_is_only_imported_by_the_cross_check():
    users = sorted(str(f.relative_to(ROOT)).replace("\\", "/") for f in _py_files()
                   if IMPORTS_REBOUND.search(f.read_text(encoding="utf-8")))
    assert users == ["dolle_sim/rebound_check.py"]
    # the simulator never imports the cross-check (only the command-line "rebound-check" command does)
    importers = sorted(f.name for f in PKG.glob("*.py") if IMPORTS_CHECK.search(f.read_text(encoding="utf-8")))
    assert importers == ["__main__.py"]


def test_lammps_is_only_imported_by_its_cross_check():
    users = sorted(str(f.relative_to(ROOT)).replace("\\", "/") for f in _py_files()
                   if IMPORTS_LAMMPS.search(f.read_text(encoding="utf-8")))
    assert users == ["dolle_sim/lammps_check.py"]
    importers = sorted(f.name for f in PKG.glob("*.py") if IMPORTS_LCHECK.search(f.read_text(encoding="utf-8")))
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
    assert len(ids) == 25
    for c in _items():
        assert c["origin"] in provenance.ORIGINS, c["component"]
        for k in c["checks"]:
            assert k in ids, (c["component"], k)


def test_every_named_code_location_exists():
    for c in _items():
        for part in c["code"].split(";"):
            m = re.match(r"\s*([\w.]+\.py)(?::\s*(.*))?", part.split(" (")[0])
            assert m and (PKG / m.group(1)).exists(), c["code"]
            src = (PKG / m.group(1)).read_text(encoding="utf-8")
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

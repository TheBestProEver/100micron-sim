"""The provenance table of the extras (extras/dolle_extras/provenance.py) must stay true to the code, and the extras'
command line must start the three ways extras/README.md gives."""
import os
import re
import subprocess
import sys
from pathlib import Path

from dolle_extras import provenance

ROOT = Path(__file__).resolve().parents[2]
EXTRAS = ROOT / "extras" / "dolle_extras"


def _items():
    return [c for s in provenance.SECTIONS for c in s["items"]]


def test_every_named_code_location_exists():
    for c in _items():
        for part in c["code"].split(";"):
            m = re.match(r"\s*([\w.]+\.py)(?::\s*(.*))?", part.split(" (")[0])
            assert m and (EXTRAS / m.group(1)).exists(), c["code"]
            src = (EXTRAS / m.group(1)).read_text(encoding="utf-8")
            for name in re.findall(r"\b_?[A-Za-z]\w*\b", m.group(2) or ""):
                assert re.search(rf"(def|class)\s+{name}\b", src), (c["code"], name)


def test_every_named_check_exists():
    ids = set(re.findall(r'_res\("(\w+)"', (EXTRAS / "verify_full.py").read_text(encoding="utf-8")))
    assert len(ids) == 19 == provenance.n_checks()
    ids |= set(re.findall(r'_res\("(\w+)"', (ROOT / "dolle_sim" / "verification.py").read_text(encoding="utf-8")))
    for c in _items():
        for k in c["checks"]:
            assert k in ids, (c["component"], k)


def test_generated_file_is_up_to_date():
    """extras/PROVENANCE.md equals the table (python -m dolle_extras sources --write)."""
    assert (ROOT / "extras" / "PROVENANCE.md").read_text(encoding="utf-8") == provenance.as_markdown()


def test_command_line_starts_three_ways():
    """python -m dolle_extras (installed, or PYTHONPATH as conftest.py sets it), from inside extras/, and as
    python extras/dolle_extras from the repository folder."""
    bare = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}          # as in a fresh clone
    for cmd, cwd, env in (([sys.executable, "-m", "dolle_extras", "--help"], ROOT, os.environ),
                          ([sys.executable, "-m", "dolle_extras", "--help"], ROOT / "extras", bare),
                          ([sys.executable, str(Path("extras") / "dolle_extras"), "--help"], ROOT, bare)):
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and "sizes-image" in r.stdout, (cmd, cwd, r.stderr[-2000:])

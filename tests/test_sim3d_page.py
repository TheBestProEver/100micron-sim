"""The 3-D page docs/sim3d/: its python copies, its data, and the numbers its text quotes.

The page itself is JavaScript that only draws what python computed; these tests check what it is given:
  * docs/sim3d/py/dolle_core/ holds byte-for-byte copies of dolle_core (tools/build_sim3d.py), and that copy runs
    with numpy alone (no scipy, numba, matplotlib or dolle_sim), as it must under Pyodide in the browser;
  * the replay's momentum ledger (one row per frame) keeps the tubule's total momentum constant to rounding error;
  * the sandbox returns the momentum rows with its frames, and honours the grain-size band;
  * index.html loads only files that exist and libraries with pinned versions, and the numbers its text quotes
    agree with the data.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "sim3d"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import build_sim3d  # noqa: E402
from dolle_core import scenes  # noqa: E402


def test_python_copies_are_current():
    assert build_sim3d.stale() == [], "run python tools/build_sim3d.py"
    man = json.loads((PAGE / "py" / "manifest.json").read_text(encoding="utf-8"))
    assert {"__init__.py", "sim3d.py", "scenes.py", "clumps.py", "impacts.py", "collide.py"} <= set(man["files"])
    assert not [f for f in man["files"] if f.startswith("nb_")]


BLOCKED_RUN = r"""
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("scipy", "numba", "matplotlib", "dolle_sim", "dolle_extras"):
            raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
from dolle_core import scenes
assert scenes.__file__.replace("\\", "/").startswith(sys.argv[1].replace("\\", "/"))
st = scenes.sandbox_start(json.dumps({"n": 20, "t_end": 0.3, "seed": 4}))
out = json.loads(scenes.sandbox_advance(st, 0.3))
print(json.dumps([len(t["frames"]["t"]) for t in out["tubules"]]))
"""


def test_page_copy_runs_with_numpy_only():
    """The browser's copy imports and runs a sandbox step with scipy, numba, matplotlib and dolle_sim blocked."""
    r = subprocess.run([sys.executable, "-c", BLOCKED_RUN, str(PAGE / "py")], capture_output=True, text=True,
                       timeout=120, cwd=PAGE / "py")
    assert r.returncode == 0, r.stderr
    assert all(n > 10 for n in json.loads(r.stdout.strip().splitlines()[-1]))


def test_index_loads_existing_files_and_pinned_libraries():
    html = (PAGE / "index.html").read_text(encoding="utf-8")
    for ref in re.findall(r'(?:src|href)="([^"#:]+)"', html):
        target = (PAGE / ref).resolve()
        assert target.exists() or (target / "index.html").exists(), ref
    cdn = re.findall(r'"(https://cdn[^"]+)"', html)
    assert cdn and all(re.search(r"three@\d+\.\d+\.\d+/", u) for u in cdn)
    worker = (PAGE / "worker.js").read_text(encoding="utf-8")
    assert re.search(r'const PYODIDE = "https://cdn\.jsdelivr\.net/pyodide/v\d+\.\d+\.\d+/full/"', worker)
    js = (PAGE / "sim3d.js").read_text(encoding="utf-8")
    for f in ("data/two_tubules.json", "data/impact_closeup.json", "worker.js"):
        assert f in js and (PAGE / f).exists()


def load_replay() -> dict:
    path = PAGE / "data" / "two_tubules.json"
    if not path.exists():
        pytest.skip("run python tools/make_sim3d_replays.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_replay_momentum_ledger_is_constant():
    """One momentum row per frame; total = flying + walls - acceleration stays at its start value to rounding."""
    rep = load_replay()
    for tub in rep["tubules"]:
        M = np.array(tub["momentum"])
        assert tub["momentum_columns"] == list(scenes.MOMENTUM_COLUMNS)
        assert len(M) == len(tub["frames"]["t"]) and np.allclose(M[:, 0], tub["frames"]["t"])
        assert M[:, -1].max() < 1e-12
        total = M[:, 1:4] + M[:, 4:7] - M[:, 7:10]
        assert np.abs(total - total[0]).max() / tub["p_scale"] < 1e-6       # the stored 7-digit values
        assert np.linalg.norm(M[-1, 4:7]) > 0.05 * tub["p_scale"]           # the walls did take momentum


def test_sandbox_returns_momentum_and_honours_grain_size():
    st = scenes.sandbox_start(json.dumps({"n": 20, "t_end": 0.2, "materials": ["graphite"], "d_min_um": 125,
                                          "d_max_um": 150}))
    d = 2e6 * st["sims"][0].clumps.r
    assert 100 < np.median(d) < 175
    out = json.loads(scenes.sandbox_advance(st, 0.1))
    out2 = json.loads(scenes.sandbox_advance(st, 0.1))
    for o in (out, out2):
        t = o["tubules"][0]
        assert len(t["momentum"]) == len(t["frames"]["t"]) and t["p_scale"] > 0
        assert max(r[-1] for r in t["momentum"]) < 1e-12
    assert out["momentum_columns"] == list(scenes.MOMENTUM_COLUMNS)


def test_numbers_quoted_on_the_page_match_the_data():
    rep = load_replay()
    html = re.sub(r"\s+", " ", (PAGE / "index.html").read_text(encoding="utf-8"))
    by = {t["name"]: t for t in rep["tubules"]}
    v = {m: by[m]["sticking_speeds_mm_s"]["grain_grain"] for m in by}
    assert f"{v['silica']:.1f} mm/s for silica" in html and f"{v['graphite']:.0f} mm/s for graphite" in html

    def clumps(m):
        sizes = {int(k): n for k, n in by[m]["summary"]["clump_sizes_at_end"].items()}
        return sum(n for k, n in sizes.items() if k > 1), sum(k * n for k, n in sizes.items() if k > 1)
    (cg, gg), (cs, gs) = clumps("graphite"), clumps("silica")
    assert f"{cg} clumps holding {gg} grains against silica's {cs} holding {gs}" in html
    assert by["graphite"]["summary"]["counts"]["bounce"] == 0 and "they all stick" in html
    c = by["silica"]["summary"]["counts"]
    assert c["bounce"] > c["stick"] and "most impacts bounce" in html
    close = json.loads((PAGE / "data" / "impact_closeup.json").read_text(encoding="utf-8"))["cases"]
    squash = [cs["max_overlap_nm"] for cs in close]
    assert f"squash by {min(squash):.0f} to {max(squash):.0f} nanometres" in html
    worker = (PAGE / "worker.js").read_text(encoding="utf-8")
    assert "about 15 MB" in worker and "about 15 MB" in html

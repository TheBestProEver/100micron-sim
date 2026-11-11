"""Tests of the optional extras: python -m pytest extras/tests (from the repository folder; no install needed).

Puts the repository (dolle_sim, dolle_core) and extras/ (dolle_extras) on the path of the tests and of the python
processes they start."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "extras"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "extras")] +
                                           ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))

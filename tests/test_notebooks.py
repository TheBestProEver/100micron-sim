"""Every notebook in notebooks/ runs from start to finish without an error (nbclient, 180 s per notebook).

The notebooks are executed in memory from the notebooks/ folder (as `jupyter lab` would); the files on disk are not
changed.  They also must not print an absolute local path (their committed outputs are published).
"""
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def test_there_are_notebooks():
    assert len(NOTEBOOKS) >= 6


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_runs(path):
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=180, kernel_name="python3",
                                     resources={"metadata": {"path": str(path.parent)}})
    client.execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, errors
    text = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert str(Path.home()) not in text and ":\\Users\\" not in text

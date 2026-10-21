"""The verification suite (also run by `python -m dolle_sim verify` and in the browser app) must pass."""
import pytest

from dolle_sim.verification import list_checks, run_check


@pytest.mark.parametrize("name", list_checks())
def test_verification_check(name):
    r = run_check(name)
    assert r["passed"], f"{r['name']}: {r['value']} (criterion {r['criterion']}) {r.get('details', '')}"

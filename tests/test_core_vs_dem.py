"""dolle_core against the optional v0.6 engine's 3-D contact model (extras/dolle_extras/dem.py, engine.py).

These comparisons need the extras/ folder (a clone of the repository has it); without it they are skipped, and the
rest of the main suite still runs (tests/test_dolle_core.py compares dolle_core with dolle_sim itself).
"""
import math
import sys
from pathlib import Path

import pytest

import dolle_core as dc
from dolle_sim import materials as sim_materials

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "extras" / "dolle_extras").is_dir() and str(ROOT / "extras") not in sys.path:
    sys.path.append(str(ROOT / "extras"))          # a clone without `pip install -e .`
dem = pytest.importorskip("dolle_extras.dem", reason="the optional extras/ folder is not in this copy")
engine = pytest.importorskip("dolle_extras.engine", reason="the optional extras/ folder is not in this copy")
R = 50e-6


def test_friction_table_equals_engine():
    """dolle_core's FRICTION is the v0.6 engine's MU_DEFAULT (the sources are listed in dolle_core/materials.py)."""
    assert dc.FRICTION == engine.MU_DEFAULT


# ---------------------------------------------------------------------------- wall statics
def _make_params(grain, wall, f_adh=1.0, mu=0.4):
    s, g = sim_materials.get_material(grain), sim_materials.get_material(wall)
    m = sim_materials.sphere_mass(s, R)
    p = dem.make_params(R, 0.0, m, math.inf, s.E, s.nu, g.E, g.nu, 2 * math.sqrt(s.gamma * g.gamma) * f_adh, 0.9,
                        math.inf, mu, 0.5, 100.0, 100.0, dem.DAMP_TSUJI_JKR, dem.BREAK_DISPLACEMENT, 1.0, 0.0)
    return p, m


def test_rolling_limit_reproduces_check_rolling_on_wall():
    """verify_full.check_rolling_on_wall: a_roll = mu_r 2 F_c / m from dem.make_params, printed as 1.29 g."""
    p, m = _make_params("silica", "glass_borosilicate")
    a_roll = p[dem.P_MUR] * 2 * p[dem.P_FC] / m
    L = dc.wall_limits("silica", R, "glass_borosilicate")
    assert L.roll_g * dc.G0 == pytest.approx(a_roll, rel=1e-12)
    assert f"{L.roll_g:.2f}" == "1.29"
    assert dc.rolling_acceleration(3 * a_roll, a_roll) == pytest.approx(2 * a_roll / 1.4, rel=1e-15)


@pytest.mark.parametrize("grain,mu,f_adh", [("silica", 0.4, 1.0), ("graphite", 0.5, 1.0), ("silica", 0.4, 0.05)])
def test_wall_limits_equal_make_params(grain, mu, f_adh):
    p, m = _make_params(grain, "glass_borosilicate", f_adh, mu)
    L = dc.wall_limits(grain, R, "glass_borosilicate", f_adh=f_adh)
    assert L.mu == mu
    Fc, a0 = p[dem.P_FC], p[dem.P_A0]
    assert (L.F_c, L.a0) == (pytest.approx(Fc, rel=1e-14), pytest.approx(a0, rel=1e-14))
    assert L.pull_off_g == pytest.approx(Fc / m / dc.G0, rel=1e-14)
    assert L.slide_g == pytest.approx(mu * 2 * Fc / m / dc.G0, rel=1e-14)
    assert L.roll_g == pytest.approx(p[dem.P_MUR] * 2 * Fc / m / dc.G0, rel=1e-14)
    assert L.twist_torque == pytest.approx(p[dem.P_MUTW] * 2 * Fc, rel=1e-14)
    assert L.k_slide == pytest.approx(p[dem.P_KT] * a0, rel=1e-14)
    assert L.k_roll == pytest.approx(p[dem.P_KR], rel=1e-14)
    assert L.k_twist == pytest.approx(p[dem.P_KTW], rel=1e-14)

"""The negligible forces are off by default (v0.7), stay visible in the pre-flight budget, and can be switched on.

Off unless a config switches them on: mutual gravity, the Lorentz force, the external E field, photophoresis, radiation
pressure, Brownian motion (forces.<key>) and the gravity gradient (environment.gravity_gradient). For the flight demo
config each is at most 2.2% of the 1e-4 g residual acceleration (python -m dolle_sim preflight).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from dolle_sim.design import preflight
from dolle_sim.environment import Environment
from dolle_sim.mission import mission_cfg
from dolle_sim.simulation import FORCES_OFF_BY_DEFAULT, ChamberSim

ROOT = Path(__file__).resolve().parents[1]
CH = {"name": "t", "size_m": [0.01] * 3,
      "particles": [{"material": "silica", "count": 5, "radius_m": 5e-5, "velocity": {"speed_mean_m_s": 0.01}}]}
OFF = {"self_gravity": "f_selfgrav", "lorentz": "f_lorentz", "external_e_field": "f_efield",
       "photophoresis": "f_photo", "radiation_pressure": "f_radp", "brownian": "f_brown"}
ON = {"coulomb": "f_coulomb", "polarization": "f_polar", "image_charges": "f_image", "space_charge": "f_space",
      "thermophoresis": "f_thermo"}


def _demo():
    return json.loads((ROOT / "configs" / "flight_carbon_vs_silica.json").read_text())


def test_negligible_forces_are_off_by_default():
    assert set(FORCES_OFF_BY_DEFAULT) == set(OFF)
    s = ChamberSim({"dt_s": 4e-3, "duration_s": 0.1, "environment": {"preset": "ideal"}}, CH, 0)
    assert not any(getattr(s, a) for a in OFF.values())
    assert all(getattr(s, a) for a in ON.values())
    assert not s.env.gravity_gradient
    for preset in ("sounding_rocket", "ideal"):
        assert not Environment({"preset": preset}, np.random.default_rng(0)).gravity_gradient
    m = json.loads((ROOT / "configs" / "mission_quarter_deck.json").read_text())
    assert mission_cfg(m)["environment"]["gravity_gradient"] is False and mission_cfg(m)["engine"] == "v05"


def test_each_switch_turns_its_force_on():
    cfg = {"dt_s": 4e-3, "duration_s": 0.1, "forces": {k: True for k in OFF},
           "environment": {"preset": "ideal", "gravity_gradient": True}}
    s = ChamberSim(cfg, CH, 0)
    assert all(getattr(s, a) for a in OFF.values()) and s.env.gravity_gradient


def test_preflight_still_shows_switched_off_forces():
    """Every term of the budget is still computed (the website charts all of them); the switched-off ones are listed,
    and for the demo config none is large enough to warn about."""
    cfg = _demo()
    for ch in cfg["chambers"]:
        pf = preflight(cfg, ch)
        g = pf["groups"][0]
        b = g["acceleration_budget_m_s2"]
        assert len(b) == 17 and b["gravity_gradient_tidal"] > 0 and b["radiation_pressure"] > 0
        assert set(g["switched_off"]) == {"mutual_gravity_at_contact", "mutual_gravity_at_mean_spacing", "lorentz_qvB",
                                          "external_E_field", "photophoresis", "radiation_pressure",
                                          "gravity_gradient_tidal"}
        assert set(g["switched_off_forces"]) == set(OFF) | {"gravity_gradient"}
        assert not any("switched off but" in w for w in pf["warnings"])
        a_res = 1e-4 * 9.80665
        assert max(b[k] for k in g["switched_off"]) < 0.025 * a_res        # the 2.2% quoted everywhere
        assert g["brownian_speed_m_s"] < 1e-3 * g["v_stick_pp_DEM_m_s"]


def test_website_preflight_marks_switched_off_terms():
    """The website's chart draws every positive budget entry by name (docs/app/app.js, unchanged): webapi.preflight
    renames the switched-off ones "<name> (off)" and adds a note listing them; with everything on, nothing is marked."""
    from dolle_sim import webapi
    cfg = _demo()
    for pf in json.loads(webapi.preflight(json.dumps(cfg))):
        b = pf["groups"][0]["acceleration_budget_m_s2"]
        shown = [k for k, v in b.items() if v > 0]                  # the bars the chart draws
        assert {k[:-6] for k in b if k.endswith(" (off)")} == set(pf["groups"][0]["switched_off"])
        assert "radiation_pressure (off)" in shown and "residual_microgravity" in shown
        assert any(w.startswith("Note: switched off") and "Brownian motion" in w for w in pf["warnings"])
    cfg["forces"].update({k: True for k in OFF})
    cfg["environment"]["gravity_gradient"] = True
    for pf in json.loads(webapi.preflight(json.dumps(cfg))):
        assert not any(k.endswith("(off)") for k in pf["groups"][0]["acceleration_budget_m_s2"])
        assert not any(w.startswith("Note: switched off") for w in pf["warnings"])


def test_preflight_warns_when_a_switched_off_force_would_matter():
    cfg = _demo()
    cfg["environment"]["illumination"]["intensity_w_m2"] = 1e5          # 100x the demo lighting
    pf = preflight(cfg, cfg["chambers"][0])
    assert any("radiation_pressure is switched off but" in w for w in pf["warnings"])
    cfg["forces"]["radiation_pressure"] = True
    pf = preflight(cfg, cfg["chambers"][0])
    assert not any("radiation_pressure is switched off but" in w for w in pf["warnings"])


@pytest.mark.parametrize("name", ["flight_carbon_vs_silica.json", "mission_quarter_deck.json"])
def test_shipped_configs_follow_the_defaults(name):
    """No shipped config switches a negligible force on (the demo lists the switches, set to false)."""
    cfg = json.loads((ROOT / "configs" / name).read_text())
    forces = cfg.get("forces", {})
    assert not any(forces.get(k, False) for k in OFF)
    assert not cfg.get("environment", {}).get("gravity_gradient", False) and not cfg.get("gravity_gradient", False)

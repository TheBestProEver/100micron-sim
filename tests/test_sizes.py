"""Size distributions (sizes.py). Grain sizes from images: extras/tests/test_sizing_image.py."""
import math

import numpy as np
import pytest

from dolle_sim import sizes


def test_percentile_fit_round_trip_and_hatch_choate():
    cmd, gsd = 90e-6, 1.15
    s = math.log(gsd)
    mu_v = math.log(cmd) + 3 * s * s                   # Hatch-Choate: volume median = CMD exp(3 ln^2 sigma_g)
    d10, d50, d90 = (math.exp(mu_v + z * s) for z in (-sizes.Z90, 0.0, sizes.Z90))
    f = sizes.lognormal_from_percentiles(d10, d50, d90, "volume")
    assert f["cmd_m"] == pytest.approx(cmd, rel=1e-12)
    assert f["gsd"] == pytest.approx(gsd, rel=1e-12)
    assert f["lognormal_misfit_ln"] < 1e-12


def test_sampling_reproduces_volume_percentiles():
    spec = {"type": "percentiles", "d10_m": 80e-6, "d50_m": 95e-6, "d90_m": 114e-6, "weighting": "volume"}
    d = sizes.sample_diameters(spec, 200000, np.random.default_rng(1))
    st = sizes.summary(d)
    f = sizes.lognormal_from_percentiles(80e-6, 95e-6, 114e-6, "volume")
    mu, s = math.log(f["cmd_m"]), math.log(f["gsd"])
    for p, z in ((10, -sizes.Z90), (50, 0.0), (90, sizes.Z90)):
        assert st[f"volume_d{p}_m"] == pytest.approx(math.exp(mu + 3 * s * s + z * s), rel=5e-3)
    assert st["number_d50_m"] == pytest.approx(f["cmd_m"], rel=5e-3)


def test_sieve_fraction_in_range():
    lo, hi = 90e-6, 106e-6
    d = sizes.sample_diameters({"type": "sieve", "min_m": lo, "max_m": hi, "fraction_in_range": 0.9}, 200000,
                               np.random.default_rng(2))
    assert np.mean((d >= lo) & (d <= hi)) == pytest.approx(0.9, abs=0.005)
    assert sizes.describe({"type": "sieve", "min_m": lo, "max_m": hi, "fraction_in_range": 0.9})["gsd"] == \
        pytest.approx(1.051, abs=1e-3)
    u = sizes.sample_diameters({"type": "sieve", "min_m": lo, "max_m": hi}, 10000, np.random.default_rng(3))
    assert u.min() >= lo and u.max() <= hi


def test_truncation_and_measured_lists():
    spec = {"type": "lognormal", "median_m": 100e-6, "gsd": 1.5, "weighting": "number", "min_m": 80e-6, "max_m": 120e-6}
    d = sizes.sample_diameters(spec, 5000, np.random.default_rng(4))
    assert "truncated" in sizes.describe(spec)
    assert d.min() >= 80e-6 and d.max() <= 120e-6
    m = sizes.sample_diameters({"type": "measured", "diameters_m": [1e-4, 2e-4]}, 1000, np.random.default_rng(5))
    assert set(np.unique(m)) == {1e-4, 2e-4}


def test_invalid_lognormal_specs_are_rejected():
    with pytest.raises(ValueError):
        sizes.sample_diameters({"type": "lognormal", "median_m": 1e-4, "gsd": 1.1}, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        sizes.sample_diameters({"type": "lognormal", "median_m": 1e-4, "gsd": 0.1, "weighting": "number"}, 10,
                               np.random.default_rng(0))

"""Size distributions (sizes.py) and image sizing (sizing.py)."""
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


def test_image_sizing_recovers_known_diameters():
    pytest.importorskip("skimage")
    from dolle_sim.sizing import measure_image, synthetic_image
    rng = np.random.default_rng(6)
    true_px = np.exp(rng.normal(math.log(40), 0.12, 120))
    for touching in (0.0, 0.3):
        img, placed = synthetic_image(true_px, shape=(1300, 1300), seed=7, touching_fraction=touching)
        m = measure_image(img, um_per_px=2.0)
        assert abs(m["n"] - len(placed)) <= 2
        assert np.median(m["diameters_m"]) / 2e-6 == pytest.approx(np.median(placed), rel=0.01)


def test_image_sizing_splits_touching_grains_but_not_elongated_ones():
    pytest.importorskip("skimage")
    from skimage.draw import disk, ellipse, polygon, rectangle
    from dolle_sim.sizing import segment
    rng = np.random.default_rng(3)
    img = np.zeros((800, 800))
    for k in range(16):                       # isolated rods, ellipses and irregular flakes: one grain each
        cy, cx = 100 + 190 * (k // 4), 100 + 190 * (k % 4)
        if k % 3 == 0:
            rr, cc = ellipse(cy, cx, 18, 60, rotation=0.3 * k)
        elif k % 3 == 1:
            rr, cc = rectangle((cy - 15, cx - 55), extent=(30, 110))
        else:
            ang = np.sort(rng.uniform(0, 2 * np.pi, 9))
            rad = rng.uniform(25, 55, 9)
            rr, cc = polygon(cy + rad * np.sin(ang), cx + rad * np.cos(ang), img.shape)
        img[rr, cc] = 1
    assert segment(1 - img).max() == 16
    pairs = np.ones((700, 400))               # touching pairs, size ratio up to 2.5: two grains each
    for n, (r1, r2) in enumerate([(20, 20), (20, 14), (20, 10), (16, 16), (25, 12), (20, 17)]):
        rr, cc = disk((60 + 110 * n, 150), r1)
        pairs[rr, cc] = 0
        rr, cc = disk((60 + 110 * n, 150 + r1 + r2 - 1.0), r2)
        pairs[rr, cc] = 0
    assert segment(pairs).max() == 12


def test_image_sizing_at_20_px():
    pytest.importorskip("skimage")
    from dolle_sim.sizing import measure_image, synthetic_image
    true_px = np.exp(np.random.default_rng(9).normal(math.log(20), 0.12, 150))
    img, placed = synthetic_image(true_px, shape=(900, 900), seed=8, touching_fraction=0.2)
    m = measure_image(img, um_per_px=1.0)
    assert abs(m["n"] - len(placed)) <= 2
    assert np.median(m["diameters_m"]) / 1e-6 == pytest.approx(np.median(placed), rel=0.01)


def test_invalid_lognormal_specs_are_rejected():
    with pytest.raises(ValueError):
        sizes.sample_diameters({"type": "lognormal", "median_m": 1e-4, "gsd": 1.1}, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        sizes.sample_diameters({"type": "lognormal", "median_m": 1e-4, "gsd": 0.1, "weighting": "number"}, 10,
                               np.random.default_rng(0))

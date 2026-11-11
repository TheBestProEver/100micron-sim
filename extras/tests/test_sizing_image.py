"""Grain sizes from microscope images (extras/dolle_extras/sizing.py; needs scikit-image, skipped without it).

Moved from tests/test_sizes.py in v0.7, unchanged."""
import math

import numpy as np
import pytest


def test_image_sizing_recovers_known_diameters():
    pytest.importorskip("skimage")
    from dolle_extras.sizing import measure_image, synthetic_image
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
    from dolle_extras.sizing import segment
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
    from dolle_extras.sizing import measure_image, synthetic_image
    true_px = np.exp(np.random.default_rng(9).normal(math.log(20), 0.12, 150))
    img, placed = synthetic_image(true_px, shape=(900, 900), seed=8, touching_fraction=0.2)
    m = measure_image(img, um_per_px=1.0)
    assert abs(m["n"] - len(placed)) <= 2
    assert np.median(m["diameters_m"]) / 1e-6 == pytest.approx(np.median(placed), rel=0.01)

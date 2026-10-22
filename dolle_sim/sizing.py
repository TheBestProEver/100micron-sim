"""Measure grain diameters from microscope images (optional: needs scikit-image).

    python -m dolle_sim sizes --image photo.png --um-per-px 1.25 [--dark-on-light] [--out sizes.csv]

Pipeline (the standard scikit-image recipe for particle sizing):
  1. grey-scale image, global Otsu threshold (grains are assumed to be the minority phase unless the
     polarity is given);
  2. fill holes, remove specks smaller than min_area_px;
  3. drop grains cut by the image border (they would be measured too small);
  4. split touching grains (on by default; --no-split turns it off): Euclidean distance transform, one marker
     per extended maximum (h-maximum plateau; isolated elongated grains are not cut up), watershed;
  5. area-equivalent circle diameter of each region (the size measure recommended for static image
     analysis by ISO 13322-1), scaled by the calibration (um per pixel, e.g. from a stage micrometer).

Accuracy: with >= ~20 px across a grain the diameter of an isolated circle is recovered to ~1-2%
(tests/test_sizes.py checks this on synthetic images with known sizes). Imaging needs enough contrast,
grains in one focal plane, and few overlaps; count several hundred grains for a stable D50.

Provenance: scikit-image (van der Walt et al. 2014, PeerJ 2, e453) functions threshold_otsu (Otsu 1979,
IEEE Trans. SMC 9, 62), distance_transform_edt, h_maxima (Soille 2003, Morphological Image
Analysis), watershed and regionprops. No REBOUND
code.
"""
from __future__ import annotations

import numpy as np


def _require():
    try:
        import skimage  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("image sizing needs scikit-image: pip install scikit-image") from e


def segment(img: np.ndarray, dark_on_light: bool | None = None, min_area_px: int = 20,
            split_touching: bool = True, h_rel: float = 0.15) -> np.ndarray:
    """Label image of grains (0 = background).

    Touching grains are split by a watershed of the distance transform seeded with its extended maxima (h-maxima
    plateaus, h = h_rel x the median grain radius): a neck between two grains is a deep saddle and gets two seeds, while the flat ridge of
    an isolated elongated grain (rod, flake) is one plateau and one seed, so it is not cut up."""
    _require()
    from scipy import ndimage as ndi
    from skimage import filters, measure, morphology, segmentation
    img = np.asarray(img, float)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    t = filters.threshold_otsu(img)
    bright = img > t
    if dark_on_light is None:                 # grains = minority phase
        dark_on_light = bright.mean() > 0.5
    mask = ~bright if dark_on_light else bright
    mask = ndi.binary_fill_holes(mask)
    lab_all, _ = ndi.label(mask)                 # drop specks (version-independent small-object removal)
    area = np.bincount(lab_all.ravel())
    keep = area >= min_area_px
    keep[0] = False
    mask = keep[lab_all]
    if split_touching:
        dist = ndi.distance_transform_edt(mask)
        lab0 = measure.label(mask)
        med_r = np.median([r.equivalent_diameter_area for r in measure.regionprops(lab0)] or [10]) / 2
        # extended maxima: regional maxima of the reconstruction of (dist - h) under dist. Each peak whose height
        # above its saddle exceeds h becomes one connected plateau (one seed); shallower ripples merge.
        h = max(1.0, h_rel * med_r)
        rec = morphology.reconstruction(np.maximum(dist - h, 0.0), dist, method="dilation")
        markers = measure.label(morphology.local_maxima(rec) & mask)
        labels = segmentation.watershed(-dist, markers, mask=mask)
    else:
        labels = measure.label(mask)
    labels = segmentation.clear_border(labels)
    return labels


def measure_image(img, um_per_px: float, **kw) -> dict:
    """Area-equivalent diameters [m] (and shape descriptors) of the grains in one image."""
    _require()
    from skimage import io, measure
    if isinstance(img, (str, bytes)) or hasattr(img, "__fspath__"):
        img = io.imread(img)
    labels = segment(img, **kw)
    props = measure.regionprops(labels)
    d = np.array([p.equivalent_diameter_area for p in props]) * um_per_px * 1e-6
    def _ax(p, new, old):
        return getattr(p, new) if hasattr(p, new) else getattr(p, old)
    aspect = np.array([_ax(p, "axis_minor_length", "minor_axis_length") / _ax(p, "axis_major_length", "major_axis_length")
                       if _ax(p, "axis_major_length", "major_axis_length") > 0 else 1.0 for p in props])
    return {"diameters_m": d, "aspect_ratio": aspect, "n": int(len(d)), "labels": labels}


def synthetic_image(diameters_px, shape=(1200, 1200), seed=0, dark_on_light=True, noise=0.03, blur_px=1.0,
                    touching_fraction=0.0):
    """Test image of non-overlapping disks with known diameters (pixels). Returns (image, placed diameters)."""
    _require()
    from scipy import ndimage as ndi
    from skimage.draw import disk
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, float)
    placed, centres = [], []
    for d in diameters_px:
        r = d / 2
        for _ in range(500):
            if centres and rng.random() < touching_fraction:      # place in contact with a placed grain
                k = rng.integers(len(centres))
                ang = rng.uniform(0, 2 * np.pi)
                c = centres[k][0] + (centres[k][2] + r + 0.5) * np.array([np.cos(ang), np.sin(ang)])
            else:
                c = rng.uniform(r + 3, np.array(shape) - r - 3)
            if np.any(c < r + 3) or np.any(c > np.array(shape) - r - 3):
                continue
            if all(np.hypot(*(c - cc)) > r + rr + 0.4 for cc, _, rr in centres):
                centres.append((c, d, r))
                rr_, cc_ = disk(tuple(c), r, shape=shape)
                img[rr_, cc_] = 1.0
                placed.append(d)
                break
    img = ndi.gaussian_filter(img, blur_px) if blur_px > 0 else img
    img = img + noise * rng.standard_normal(shape)
    if dark_on_light:
        img = 1.0 - img
    return img, np.array(placed)

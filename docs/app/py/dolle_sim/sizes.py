"""Grain size distributions: manufacturer specifications, measured lists, and number/volume weighting.

A particle group in a run config can give, instead of radius_m (+ radius_rel_sigma),
    "size_distribution": {...}
with one of these forms (all diameters in metres):

    {"type": "lognormal", "median_m": 9.8e-5, "gsd": 1.08, "weighting": "number" | "volume"}   (weighting required)
    {"type": "percentiles", "d10_m": 9.0e-5, "d50_m": 1.0e-4, "d90_m": 1.12e-4, "weighting": "volume"}
    {"type": "sieve", "min_m": 9.0e-5, "max_m": 1.06e-4, "fraction_in_range": 0.9}
    {"type": "measured", "diameters_m": [...]}      or      {"type": "measured", "csv": "sizes.csv"}

Any of the first two may add "min_m" / "max_m" to truncate (e.g. at sieve cuts).

Weighting matters. Laser diffraction (ISO 13320) reports VOLUME-weighted percentiles (Dv10, Dv50,
Dv90); microscope counts and image analysis (ISO 13322) are NUMBER-weighted; the simulation needs
number-weighted grains. For a log-normal distribution the two have the same geometric standard
deviation sigma_g and their medians differ by the Hatch-Choate factor:
    volume (mass) median = count median * exp(3 ln^2 sigma_g),
and in general the median of the q-th moment distribution is CMD * exp(q ln^2 sigma_g); the Sauter
mean diameter is d32 = CMD * exp(2.5 ln^2 sigma_g) (Hatch & Choate 1929; Hinds 1999, ch. 4).

A log-normal is fitted to three percentiles by least squares in ln d against the standard-normal
quantiles z = -1.2816, 0, +1.2816; the residual tells you how far the data are from log-normal.

A sieve fraction alone (e.g. "90-106 um") does not say how sizes are distributed inside the band.
With "fraction_in_range" (a supplier's "> 90% in size range") the "sieve" form uses a log-normal centred on
sqrt(min max) whose sigma_g puts that fraction inside the band (2 Phi(ln(max/min) / (2 ln sigma_g)) - 1 =
fraction); without it, a distribution uniform in ln d inside the band. Both are ASSUMPTIONS (suppliers do not
say whether the fraction is by number or by mass), flagged in describe(). Prefer measured diameters or the
supplier's percentile data.

Provenance: log-normal statistics and the Hatch-Choate relations are standard (Hinds 1999); the
sampling uses numpy/scipy. Fitting and sampling are tested in tests/test_sizes.py. No REBOUND code.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

Z90 = 1.2815515655446004          # standard-normal quantile of 0.90
MOMENT = {"number": 0, "surface": 2, "volume": 3, "mass": 3}


def _q(weighting: str) -> int:
    try:
        return MOMENT[weighting]
    except KeyError:
        raise ValueError(f"weighting must be one of {sorted(MOMENT)}, not '{weighting}'") from None


def lognormal_from_percentiles(d10: float, d50: float, d90: float, weighting: str = "volume") -> dict:
    """Least-squares log-normal through three percentiles of a `weighting`-weighted distribution.

    Returns the NUMBER-weighted count median diameter (cmd_m), sigma_g, the median of the given
    weighting, and the largest misfit of the three points in ln d (0 for an exact log-normal)."""
    if not (0 < d10 < d50 < d90):
        raise ValueError("need 0 < d10 < d50 < d90")
    z = np.array([-Z90, 0.0, Z90])
    y = np.log([d10, d50, d90])
    s = float((z * (y - y.mean())).sum() / (z * z).sum())       # slope = ln sigma_g (z has zero mean)
    mu_w = float(y.mean())
    misfit = float(np.abs(y - (mu_w + s * z)).max())
    cmd = math.exp(mu_w - _q(weighting) * s * s)
    return {"cmd_m": cmd, "gsd": math.exp(s), "median_given_weighting_m": math.exp(mu_w),
            "weighting": weighting, "lognormal_misfit_ln": misfit}


def sieve_lognormal(lo: float, hi: float, fraction: float) -> tuple[float, float]:
    """(mu, s) of a log-normal centred on sqrt(lo hi) with the given fraction of grains inside [lo, hi]."""
    from scipy.stats import norm
    if not 0 < fraction < 1:
        raise ValueError("fraction_in_range must be between 0 and 1")
    s = math.log(hi / lo) / (2 * norm.ppf(0.5 * (1 + fraction)))
    return 0.5 * math.log(lo * hi), s


def lognormal_number_params(spec: dict) -> tuple[float, float]:
    """(mu, s) of ln d for the NUMBER-weighted distribution of a lognormal/percentiles spec."""
    t = spec["type"]
    if t == "lognormal":
        gsd, med = float(spec["gsd"]), float(spec["median_m"])
        if not (gsd >= 1 and med > 0):
            raise ValueError("a lognormal spec needs median_m > 0 and gsd >= 1 (gsd is sigma_g itself, not ln sigma_g)")
        if "weighting" not in spec:
            raise ValueError("a lognormal spec needs 'weighting': 'number' (microscopy) or 'volume' (laser "
                             "diffraction): the medians differ by exp(3 ln^2 gsd)")
        s = math.log(gsd)
        return math.log(med) - _q(spec["weighting"]) * s * s, s
    if t == "percentiles":
        f = lognormal_from_percentiles(float(spec["d10_m"]), float(spec["d50_m"]), float(spec["d90_m"]),
                                       spec.get("weighting", "volume"))
        return math.log(f["cmd_m"]), math.log(f["gsd"])
    raise ValueError(f"not a log-normal spec: {t}")


def _read_csv(path) -> np.ndarray:
    vals = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            for cell in row[:1]:
                try:
                    vals.append(float(cell))
                except ValueError:
                    pass                                         # header
    return np.asarray(vals)


def sample_diameters(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """n number-weighted diameters [m] drawn from a size_distribution spec."""
    t = spec.get("type")
    lo = float(spec.get("min_m", 0.0))
    hi = float(spec.get("max_m", math.inf))
    if t in ("lognormal", "percentiles"):
        mu, s = lognormal_number_params(spec)
        if s == 0:
            return np.full(n, math.exp(mu))
        from scipy.stats import truncnorm
        a = (math.log(lo) - mu) / s if lo > 0 else -np.inf
        b = (math.log(hi) - mu) / s if math.isfinite(hi) else np.inf
        return np.exp(mu + s * truncnorm.rvs(a, b, size=n, random_state=rng))
    if t == "sieve":
        if not (0 < lo < hi < math.inf):
            raise ValueError("a sieve spec needs 0 < min_m < max_m")
        if "fraction_in_range" in spec:
            mu, s = sieve_lognormal(lo, hi, float(spec["fraction_in_range"]))
            return np.exp(mu + s * rng.standard_normal(n))
        return np.exp(rng.uniform(math.log(lo), math.log(hi), n))
    if t == "measured":
        d = np.asarray(spec["diameters_m"], float) if "diameters_m" in spec else _read_csv(spec["csv"]) * float(
            spec.get("csv_scale_to_m", 1.0))
        d = d[(d > 0) & (d >= lo) & (d <= hi)]
        if len(d) == 0:
            raise ValueError("no measured diameters left")
        return rng.choice(d, size=n, replace=True)
    raise ValueError(f"unknown size_distribution type '{t}'")


def weighted_percentile(d: np.ndarray, p: float, weighting: str = "number") -> float:
    """p-th percentile (0-100) of the diameters, weighted by d^q (q = 0 number, 3 volume)."""
    d = np.sort(np.asarray(d, float))
    w = d ** _q(weighting)
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(p / 100.0 * w.sum(), c, d))


def summary(d) -> dict:
    """Number- and volume-weighted D10/D50/D90, mean, Sauter mean d32 [m] of a list of diameters."""
    d = np.asarray(d, float)
    out = {"n": int(len(d)), "mean_m": float(d.mean()), "sauter_d32_m": float((d ** 3).sum() / (d ** 2).sum())}
    for wname in ("number", "volume"):
        for p in (10, 50, 90):
            out[f"{wname}_d{p}_m"] = weighted_percentile(d, p, wname)
    if len(d) > 2:
        s = float(np.std(np.log(d), ddof=1))
        out["gsd_from_sample"] = math.exp(s)
    return out


def describe(spec: dict) -> dict:
    """Human-readable description of a spec, with its assumptions."""
    t = spec.get("type")
    out = {"type": t}
    if t in ("lognormal", "percentiles"):
        mu, s = lognormal_number_params(spec)
        out.update(count_median_m=math.exp(mu), gsd=math.exp(s), volume_median_m=math.exp(mu + 3 * s * s),
                   sauter_d32_m=math.exp(mu + 2.5 * s * s))
        if t == "percentiles":
            out["lognormal_misfit_ln"] = lognormal_from_percentiles(
                float(spec["d10_m"]), float(spec["d50_m"]), float(spec["d90_m"]),
                spec.get("weighting", "volume"))["lognormal_misfit_ln"]
    if t in ("lognormal", "percentiles") and ("min_m" in spec or "max_m" in spec):
        out["truncated"] = (f"sampled only between min_m and max_m: the sampled grains no longer follow the "
                            f"untruncated statistics above")
    if t == "sieve":
        if "fraction_in_range" in spec:
            mu, s = sieve_lognormal(float(spec["min_m"]), float(spec["max_m"]), float(spec["fraction_in_range"]))
            out.update(count_median_m=math.exp(mu), gsd=math.exp(s))
            out["assumption"] = ("log-normal centred on the band with the stated fraction inside it (shape and "
                                 "whether the fraction is by number or mass are not given by suppliers)")
        else:
            out["assumption"] = "uniform in ln(d) inside the sieve band (the band alone does not fix the shape)"
    return out

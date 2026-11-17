"""From flight data back to e_dry and the surface energy: predict each measured impact, score the prediction with
a likelihood, fit, make synthetic flight data, read and write impact lists (CSV).

What the flight gives.  Each filmed grain-grain impact gives an impact speed v (along the line of centres), whether
the grains stuck, and for a bounce the restitution e = (exit speed)/(impact speed).  The model has two numbers the
flight can test: the surface energy gamma (through the work of adhesion w = 2 gamma) and the dry restitution e_dry
(damping.py).  This module finds the (gamma, e_dry) that best explain a list of impacts, with error bars.

1. The model's restitution is ONE function of two numbers.
   In units of the JKR scales the head-on impact equation has a single free number, e_dry (scaling.py; notebook 03,
   section 5; the code's time step obeys the same scaling).  So the outcome of an impact at approach speed v depends
   only on
        x = v / v0     and     e_dry,     v0 = sqrt(2 |U0| / m*) = sqrt(2 k_U w^(5/3) R*^(4/3) E*^(-2/3) / m*),
   v0 [m/s] the undamped sticking speed of that pair (scaling.py; k_U = 7.0922, jkr.py).  The pair sticks if
   x < f(e_dry) (the damping factor, scaling.py) and otherwise bounces with e = h(x, e_dry).  The surface energy enters
   only through v0, which scales as w^(5/6): multiplying gamma by s multiplies v0 by s^(5/6).  Hence ONE table
   h(x, e_dry), computed once with collide.py's own loop on a reference pair (two 50 um radius silica grains, yield
   off, 300 steps per contact as in the simulator), gives the restitution of every pair at every surface energy.
   tests/test_dolle_core_inverse.py checks the table against collide.impact for graphite, other radii and a wall.

   The table (inverse_grid.json, rebuilt by build_grid()): f(e_dry) by bisection (relative tolerance 1e-6) and e^2 at
   u = ln(x / f(e_dry)) from 0 to ln 500, on e_dry = 0.30, 0.31, ..., 0.99.  Just above the threshold e rises like
   sqrt(u), so e^2 is close to linear in u there and is what is interpolated (linearly in u and in e_dry).

   Yield is left out of the table.  Below the yield speed v_y (plastic.py) a contact is elastic, so the table
   describes every impact slower than v_y (to its interpolation error, about 1e-3 in e); faster impacts are dropped
   from a fit, with a warning.  With the ASSUMED yield pressures v_y = 11 m/s for silica pairs and 0.84 m/s for
   graphite pairs (50 um radius).

2. The parameters.  gamma_scale s = w / w_cat, the factor on the work of adhesion of the CATALOGUE values
   (materials.py: gamma_cat = 0.025 J/m^2 for silica, 0.075 J/m^2 for graphite; materials.py quotes 0.014-0.25 J/m^2
   for silica in the literature, so s far from 1 is possible); for two grains of one material s = gamma / gamma_cat.
   And the pair's e_dry (for two materials the simulator's pair value is sqrt(e1 e2), pair.py).  ("The table" below
   always means the restitution grid h(x, e_dry) of section 1.)

3. The measurement model.  The speeds are the NORMAL components (along the line of centres) in 3-D, which needs two
   camera views (dolle_sim/analysis.py synthetic_camera, v_n_measured_stereo; PROVENANCE.md "Synthetic camera": one
   camera cannot recover it).  With ONE camera, an impact whose line of centres makes an angle theta with the view
   axis shows v sin(theta); for random orientations the mean of sin(theta) is pi/4, so measured speeds would be about
   21 % low on average and gamma (s scales as v_stick^(6/5)) about 25 % low.  Single-camera data must be corrected
   (projected centre separation at contact = (r1 + r2) sin(theta), which needs measured radii) or restricted to impacts
   close to the image plane.  The measured speed is v_m = v + eps with eps ~ N(0, sigma_v(v)) and
        sigma_v(v) = sqrt(sigma_abs^2 + (sigma_rel v)^2)    [m/s].
   sigma_abs comes from the camera's track fit, the formula of dolle_sim/analysis.py (synthetic_camera): a straight
   line fitted to k frames at fps frames/s with centroid error sigma_x gives a velocity error
   sigma_x fps sqrt(12/(k(k^2-1))) per grain and image axis, times sqrt(2) for a relative velocity; 0.85 mm/s for the
   ASSUMED design camera of configs/mission_quarter_deck.json (500 frames/s, 5 um, 6 frames).
   sigma_rel = 0.05 is ASSUMED.  Where it comes from: the line of centres is found from two centroids about 100 um
   apart, so its direction is off by d_theta = sqrt(2) x 5 um / 100 um = 0.07 rad.  Projecting the relative velocity
   (normal part v_n, tangential part v_t) on a direction turned by d_theta gives v_n cos(d_theta) + v_t sin(d_theta)
   = v_n + v_t d_theta to first order, so d(v_n)/v_n = d_theta tan(alpha), alpha the impact angle from head-on
   (tan(alpha) = v_t/v_n).  For random impact parameters the median tan(alpha) is 1 (7 %), with a heavy tail for
   grazing impacts; 0.05 corresponds to a typical v_t/v_n of 0.7 (ASSUMED).
   The exit speed is measured the same way, v_out_m = e v + eps_out (signed: noise can make it slightly negative),
   and the reported restitution is e_m = v_out_m/v_m.  Noise in both speeds can push e_m above 1; such values are
   expected and must not be clipped or dropped.
   Limits of this model (not tested by the synthetic data, which are made with the same assumptions): the direction
   error is common to the incoming and outgoing speed of one impact and scales with v_t, not v_n, but the model
   treats eps and eps_out as independent and proportional to v (probably conservative for e); the generator folds
   the measured speed (|v + eps|), the likelihood does not (matters only below about 2 sigma_abs = 1.7 mm/s); and if
   restitutions are measured preferentially for clear, fast bounces, that selection on e is not modelled.

4. The likelihood (how probable the observed impacts are for given s and e_dry).  Only v_m is known, so the true
   speed is integrated over:  P(v | v_m) ~ pi(v) N(v_m; v, sigma_v(v)), with pi(v) the distribution of true impact
   speeds, estimated from the measured speeds themselves (speed_prior_density).  Why pi is needed: if most impacts
   are slow, an impact measured at 6 mm/s is more likely a 5 mm/s impact measured high than a 7 mm/s impact measured
   low; a flat pi ignores that and biases the fit where the number of impacts changes quickly with speed (notebook
   07, section 5.3, compares the two).  With v_s = f(e_dry) v0 the sticking speed of the pair:
        stuck:                L = P(v < v_s | v_m)       (without pi: Phi((v_s - v_m)/sigma_v), Phi the standard
                                                          normal distribution: a Bernoulli outcome smoothed by the
                                                          speed error)
        bounced, e not known: L = P(v > v_s | v_m)
        bounced, e_m known:   L = integral over v > v_s of P(v | v_m) N(e_m; h(v/v0) v/v_m, s_e) dv,
                              s_e^2 = sigma_v(h v)^2 / v_m^2 + sigma_e_extra^2
   (given v and v_m, e_m = (h v + eps_out)/v_m is Gaussian).  sigma_e_extra (default 0) is an optional extra scatter
   of e that the model does not contain (spin, shape, charge).  The integral over v runs over N_CELLS + 2 = 130
   cells: 128 of width sigma_v/16 between -4 and +4 sigma_v around v_m, plus the two tails; the cell holding v_s is
   split at v_s using the normal probability inside it, so the likelihood is continuous in the parameters, and the
   restitution of its part above v_s is evaluated at the mean of the normal distribution over that part.  (With
   sigma_v/4 cells the steep rise of e just above v_s was sampled too coarsely when sigma_v << v_s, e.g. graphite with
   a gentle release, and -ln L became bumpy on a scale of 0.005 in ln s: the curvature error bars came out up to 2.7
   times too small.  With sigma_v/16 they agree with the profile intervals; tests/test_dolle_core_inverse.py.)
   ln L = sum of ln L_i.
   Optional scatter of the threshold from grain to grain: real powders have a spread of adhesion, ln w ~ N(0,
   sigma_ln_w), with sigma_ln_w the spread PER PAIR (per impact).  The simulator's flight runs give each GRAIN a factor
   exp(0.6 N(0,1)) (configs/mission_quarter_deck.json surface_energy_log_sigma, from the scatter of AFM adhesion on
   clean glass, Rush et al. 2018 as cited in PROVENANCE.md) and combine two grains as sqrt(a_i a_j)
   (dolle_sim/simulation.py, _contact), a grain and the wall as a_i x (wall factor).  So the per-pair spread is
   0.6/sqrt(2) = 0.42 for grain-grain impacts and 0.6 for grain-wall impacts.  (A grain that takes part in several
   impacts makes their outcomes correlated; the likelihood treats impacts as independent.)  An unmeasured radius
   varies within the sieve cut (size_scatter_ln_v0, about 0.03).  Both spread ln v0 (v0 scales as w^(5/6)):
   sigma_ln_v0^2 = (5/6 sigma_ln_w)^2 + sigma_size^2.  Then P(stick | v) = Phi((ln(f v0) - ln v)/sigma_ln_v0) exactly,
   and the restitution density is averaged over ln v0 (3 Gauss-Hermite nodes for sigma_ln_v0 <= 0.15, else 18 cells
   of the normal distribution that keep its tails).

5. The fit.  Maximum likelihood in (ln s, e_dry) (a scan of e_dry along the curve of equal sticking speed, then
   Nelder-Mead).  Error bars from the profile
   likelihood: the 1-sigma interval of e_dry is where min over s of -ln L rises by 1/2 above its minimum, and likewise
   for s (Wilks 1938; e.g. Cowan 1998, "Statistical Data Analysis", ch. 9: -2 ln L behaves like chi^2, and one
   parameter at 1 sigma is Delta chi^2 = 1, i.e. Delta(-ln L) = 1/2).  A covariance from the curvature matrix
   (Hessian: the second derivatives of -ln L; its inverse is the covariance) at the best fit is returned too.
   If at an end of the gamma interval the best e_dry sits on the e_dry search limit, that end is set by the search
   range, not by the data, and at_bound flags it.

Why stick/bounce alone cannot give both numbers.  The stick/bounce outcomes only see the threshold
v_s = f(e_dry) v0(s), i.e. the product f(e_dry) s^(5/6): any (s, e_dry) on that curve explains them equally well.
The restitution of bounces depends on e_dry separately (e approaches e_dry well above the threshold), so bounces with
measured e break the degeneracy.  notebooks/07_edry_the_energy_loss.ipynb shows both.

Units: SI unless a name says otherwise (CSV files use mm/s and um).  Plain python + numpy; scipy only for the
minimiser (scipy.optimize).
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass, field, replace
from functools import lru_cache
import json
import math
from pathlib import Path
import warnings

import numpy as np

from . import materials as _materials
from .collide import impact, sticking_threshold
from .materials import get_material
from .pair import make_pair
from .plastic import yield_speed
from .scaling import undamped_sticking_speed

GRID_FILE = Path(__file__).with_name("inverse_grid.json")
"""The cached table h(x, e_dry) (section 1 of the module docstring); rebuilt by build_grid()."""

E_GRID = tuple(round(0.30 + 0.01 * k, 2) for k in range(70))
"""e_dry values of the table: 0.30, 0.31, ..., 0.99."""

U_GRID = (0.0,) + tuple(float(u) for u in np.geomspace(1e-4, math.log(500.0), 59))
"""u = ln(x / f(e_dry)) values of the table: 0 (the threshold) and 59 values from 1e-4 to ln 500, dense near 0."""

N_SUB = 300
"""Steps per contact of the table (the simulator's steps_per_contact, collide.py)."""

D_SIEVE = (90e-6, 106e-6)
"""The flight's sieve cut, grain DIAMETERS [m] (the 3-D page's sizes, scenes.py START_SOURCES)."""

R_NOMINAL = 0.5 * math.sqrt(D_SIEVE[0] * D_SIEVE[1])
"""Radius used when a grain's radius was not measured: half the geometric mean of the sieve cut, 48.84 um."""


# ---------------------------------------------------------------------------- the table h(x, e_dry)
def _reference_pair():
    """The pair the table is computed on: two 50 um radius silica grains, yield off (scaling.py damping_factor)."""
    return make_pair("silica", 50e-6, "silica", 50e-6, plastic=False)


def build_grid(path: Path | str | None = GRID_FILE, n_sub: int = N_SUB) -> dict:
    """Compute the table with collide.py (about 15 s) and, if path is given, write it as JSON.  Returns
    {"e": e_dry values, "u": u values, "f": f(e_dry), "q": e^2 [len(e) x len(u)], "n_sub", "reference"}."""
    ref = _reference_pair()
    v0 = undamped_sticking_speed(ref)
    f_list, q_rows = [], []
    for e in E_GRID:
        p = replace(ref, e_dry=e)
        f = sticking_threshold(p, tol=1e-6, n_sub=n_sub) / v0
        row = [0.0] + [impact(p, f * math.exp(u) * v0, n_sub=n_sub).e ** 2 for u in U_GRID[1:]]
        f_list.append(f)
        q_rows.append(row)
    grid = {"e": list(E_GRID), "u": list(U_GRID), "f": f_list, "q": q_rows, "n_sub": n_sub,
            "reference": "two 50 um radius silica grains, yield off (dolle_core.inverse._reference_pair)"}
    if path is not None:
        Path(path).write_text(json.dumps(grid, separators=(",", ":")), encoding="utf-8")
    return grid


@dataclass(frozen=True)
class _Grid:
    e: np.ndarray
    u: np.ndarray
    ln_f: np.ndarray
    q: np.ndarray


_GRID: _Grid | None = None


def load_grid() -> _Grid:
    """The table, read once from inverse_grid.json (computed in memory by build_grid if the file is missing or was
    made with other grid values)."""
    global _GRID
    if _GRID is None:
        g = None
        if GRID_FILE.is_file():
            g = json.loads(GRID_FILE.read_text(encoding="utf-8"))
            if (tuple(g["e"]) != E_GRID or len(g["u"]) != len(U_GRID)
                    or not np.allclose(g["u"], U_GRID, rtol=1e-12) or g["n_sub"] != N_SUB):
                g = None
        if g is None:
            g = build_grid(path=None)
        _GRID = _Grid(np.array(g["e"]), np.array(g["u"]), np.log(np.array(g["f"])), np.array(g["q"]))
    return _GRID


def _check_e(e_dry: float) -> None:
    if not E_GRID[0] - 1e-12 <= e_dry <= E_GRID[-1] + 1e-12:
        raise ValueError(f"e_dry = {e_dry} is outside the table ({E_GRID[0]} to {E_GRID[-1]})")


def threshold_factor(e_dry):
    """f(e_dry) = v_stick / v0 (no unit), interpolated from the table (ln f linear in e_dry).  Same as
    scaling.damping_factor(e_dry, n_sub=300) to about 1e-4."""
    g = load_grid()
    e = np.asarray(e_dry, float)
    if np.any(e < g.e[0] - 1e-12) or np.any(e > g.e[-1] + 1e-12):
        raise ValueError(f"e_dry outside the table ({g.e[0]} to {g.e[-1]})")
    out = np.exp(np.interp(e, g.e, g.ln_f))
    return float(out) if out.ndim == 0 else out


def e_of_threshold_factor(f):
    """The e_dry whose threshold factor is f (inverse of threshold_factor; f decreases with e_dry), clipped to the
    table's range."""
    g = load_grid()
    return float(np.interp(math.log(f), g.ln_f[::-1], g.e[::-1]))


def restitution_of_x(x, e_dry: float) -> np.ndarray:
    """e = h(x, e_dry) (no unit) for x = v / v0 (array) and one e_dry; 0 where x <= f(e_dry) (the grains stick).
    Above u = ln(x/f) = ln 500 the last table value is kept (e is within 1 % of e_dry there)."""
    _check_e(e_dry)
    g = load_grid()
    i = min(max(int(np.searchsorted(g.e, e_dry)) - 1, 0), len(g.e) - 2)
    t = (e_dry - g.e[i]) / (g.e[i + 1] - g.e[i])
    q_row = (1 - t) * g.q[i] + t * g.q[i + 1]
    f = math.exp((1 - t) * g.ln_f[i] + t * g.ln_f[i + 1])
    x = np.asarray(x, float)
    u = np.log(np.maximum(x, 1e-300) / f)
    e = np.sqrt(np.maximum(np.interp(u, g.u, q_row), 0.0))
    return np.where(u > 0, e, 0.0)


# ---------------------------------------------------------------------------- the camera's speed error
def camera_speed_error(fps: float = 500.0, position_noise_m: float = 5e-6, track_frames: int = 6) -> float:
    """Error [m/s] of a relative velocity component from straight-line track fits: sqrt(2) sigma_x fps
    sqrt(12/(k(k^2-1))), the formula of dolle_sim/analysis.py synthetic_camera.  Defaults: the camera of
    configs/mission_quarter_deck.json (500 frames/s, 5 um centroid error, 6 frames): 0.845 mm/s."""
    k = track_frames
    return math.sqrt(2) * position_noise_m * fps * math.sqrt(12.0 / (k * (k * k - 1)))


@dataclass(frozen=True)
class SpeedError:
    """Measurement error of a speed: sigma_v(v) = sqrt(sigma_abs^2 + (sigma_rel v)^2) [m/s].  Defaults:
    sigma_abs from camera_speed_error() (dolle_sim/analysis.py), sigma_rel = 0.05 ASSUMED (module docstring, 3)."""
    sigma_abs: float = field(default_factory=camera_speed_error)
    sigma_rel: float = 0.05

    def sigma(self, v):
        """sigma_v(v) [m/s] for speeds v [m/s]."""
        v = np.asarray(v, float)
        return np.sqrt(self.sigma_abs ** 2 + (self.sigma_rel * v) ** 2)


# ---------------------------------------------------------------------------- a list of impacts
@dataclass
class Impacts:
    """Measured impacts of ONE material pair (grain-grain, or grain-wall when r2 is inf).  Arrays of equal length:
    speed [m/s] (measured impact speed along the line of centres), stuck (bool), restitution (measured e of a bounce;
    nan if stuck or not measured), r1, r2 [m] (nan if not measured; r2 = inf for a wall).  material1/material2 are
    names in materials.MATERIALS.  truth: for synthetic data, the true values (true speeds, e, radii, parameters)."""
    speed: np.ndarray
    stuck: np.ndarray
    restitution: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    material1: str
    material2: str
    truth: dict | None = None

    def __post_init__(self):
        n = len(self.speed)
        self.speed = np.asarray(self.speed, float)
        self.stuck = np.asarray(self.stuck, bool)
        self.restitution = np.asarray(self.restitution, float)
        self.r1 = np.broadcast_to(np.asarray(self.r1, float), (n,)).copy()
        self.r2 = np.broadcast_to(np.asarray(self.r2, float), (n,)).copy()
        get_material(self.material1), get_material(self.material2)
        if not (len(self.stuck) == len(self.restitution) == n):
            raise ValueError("speed, stuck and restitution must have the same length")
        if np.any(~(self.speed > 0)):
            raise ValueError("impact speeds must be > 0")

    def __len__(self) -> int:
        return len(self.speed)

    @property
    def pair(self) -> str:
        """'material1-material2', with '(wall)' for grain-wall impacts."""
        return f"{self.material1}-{self.material2}" + ("(wall)" if np.all(np.isinf(self.r2)) and len(self) else "")

    @property
    def has_restitution(self) -> np.ndarray:
        """True for bounces with a measured restitution."""
        return ~self.stuck & np.isfinite(self.restitution)

    def select(self, mask) -> "Impacts":
        """The impacts where mask is True (truth arrays of the same length are selected too)."""
        mask = np.asarray(mask, bool)
        truth = None
        if self.truth is not None:
            truth = {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape else v)
                     for k, v in self.truth.items()}
        return Impacts(self.speed[mask], self.stuck[mask], self.restitution[mask], self.r1[mask], self.r2[mask],
                       self.material1, self.material2, truth)

    def without_restitution(self) -> "Impacts":
        """The same impacts with every restitution blanked (stick/bounce only)."""
        out = self.select(np.ones(len(self), bool))
        out.restitution = np.full(len(self), np.nan)
        return out


# ---------------------------------------------------------------------------- per-impact constants
@lru_cache(maxsize=None)
def size_scatter_ln_v0(wall: bool = False, d_range: tuple = D_SIEVE, fraction: float = 0.9, n: int = 20000) -> float:
    """Standard deviation of ln v0 [no unit] when the radii are not measured: grains drawn from the sieve cut (a
    log-normal with `fraction` inside d_range, scenes.sieve_radii; the shape is ASSUMED there), v0 scaling as
    R*^(2/3) m*^(-1/2).  About 0.03 for two grains of the 90-106 um cut (fixed random seed 0)."""
    from .scenes import sieve_radii
    rng = np.random.default_rng(0)
    r1 = sieve_radii(n, d_range[0], d_range[1], fraction, rng)
    if wall:
        R, m = r1, r1 ** 3
    else:
        r2 = sieve_radii(n, d_range[0], d_range[1], fraction, rng)
        R, m = r1 * r2 / (r1 + r2), r1 ** 3 * r2 ** 3 / (r1 ** 3 + r2 ** 3)
    return float(np.std(np.log(R ** (2 / 3) * m ** -0.5)))


def pair_speeds(material1, material2, r1: float, r2: float | None, gamma_scale: float = 1.0) -> tuple:
    """(v0, v_y) [m/s] of one pair: the undamped sticking speed at gamma_scale x the catalogue work of adhesion
    (scaling.undamped_sticking_speed) and the yield speed (plastic.yield_speed).  r2 None or inf: a wall."""
    wall = r2 is None or math.isinf(r2)
    p = make_pair(material1, r1, material2, None if wall else r2, f_adh=gamma_scale)
    return undamped_sticking_speed(p), yield_speed(p.m_star, p.R_star, p.E_star, p.p_y)


# ---------------------------------------------------------------------------- normal distribution (numpy only)
def _erfc(x):
    """Complementary error function, fractional error < 1.2e-7 everywhere (Chebyshev fit 'erfcc' of Press, Teukolsky,
    Vetterling & Flannery, "Numerical Recipes", 2nd ed., section 6.2); accurate in the far tails too."""
    x = np.asarray(x, float)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    r = t * np.exp(poly)
    return np.where(x >= 0, r, 2.0 - r)


def normal_cdf(z):
    """Phi(z), the standard normal distribution function: 0.5 erfc(-z/sqrt 2)."""
    return 0.5 * _erfc(-np.asarray(z, float) / math.sqrt(2.0))


N_CELLS = 128
"""Finite cells of the true-speed integral (module docstring, section 4): width 8 sigma_v / N_CELLS = sigma_v/16."""

_Z_EDGES = np.concatenate([[-np.inf], np.linspace(-4.0, 4.0, N_CELLS + 1), [np.inf]])
"""Cells of the true-speed integral, in units of sigma_v around the measured speed: N_CELLS cells between -4 and 4,
plus the two tails."""
_PHI_LO, _PHI_HI = normal_cdf(_Z_EDGES[:-1]), normal_cdf(_Z_EDGES[1:])
_PHIC_LO, _PHIC_HI = normal_cdf(-_Z_EDGES[:-1]), normal_cdf(-_Z_EDGES[1:])        # 1 - Phi, without cancellation
_DPHI = _PHI_HI - _PHI_LO
_Z_MID = np.concatenate([[-4.2247], 0.5 * (_Z_EDGES[1:-2] + _Z_EDGES[2:-1]), [4.2247]])
"""Cell nodes: the middle of each finite cell; for the tails their mean, phi(4)/Phi(-4) = 4.2247."""


def _normal_pdf(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / math.sqrt(2 * math.pi)


def _split_node(zi: np.ndarray) -> np.ndarray:
    """Node of the part ABOVE zi of the cell that contains zi: the mean of the standard normal over [zi, upper edge],
    (phi(zi) - phi(b)) / (Phi(b) - Phi(zi)).  In the top tail (b infinite) this is phi(zi)/Phi(-zi), about
    zi + 1/zi, so the node stays above zi (a node below v_s would give the restitution of a stuck grain, 0).
    zi: array [n, 1]; returns [n, 1]."""
    k = np.clip(np.searchsorted(_Z_EDGES, zi, side="right") - 1, 0, len(_Z_EDGES) - 2)
    b = _Z_EDGES[k + 1]
    num = _normal_pdf(zi) - np.where(np.isfinite(b), _normal_pdf(np.where(np.isfinite(b), b, 0.0)), 0.0)
    den = normal_cdf(-zi) - normal_cdf(-b)
    ok = den > 1e-250
    z = np.where(ok, num / np.where(ok, den, 1.0), zi + 1.0 / np.maximum(np.abs(zi), 1.0))
    return np.clip(z, zi, np.where(np.isfinite(b), b, np.inf))


def speed_prior_density(v_measured, kind="data"):
    """The distribution of TRUE impact speeds assumed when going from a measured speed to the true one (module
    docstring, section 4).  Returns a function pi(v) (any normalisation; 0 for v <= 0).
      "data": estimated from the measured speeds themselves: a Gaussian kernel density in ln v (bandwidth
              1.06 sd(ln v) N^(-1/5), Silverman 1986, "Density Estimation", eq. 3.28), converted to a density in v.
              It is broader than the true distribution by the speed error, a small effect for its slope.
      "flat": pi = 1 for v > 0 (the plain "measured speed +- error" reading; biased where the number of impacts
              changes quickly with speed).
      or a function v -> density."""
    if callable(kind):
        return kind
    if kind == "flat":
        return lambda v: np.where(np.asarray(v) > 0, 1.0, 0.0)
    if kind != "data":
        raise ValueError("speed prior must be 'data', 'flat' or a function")
    lv = np.log(np.asarray(v_measured, float))
    h = max(1.06 * float(np.std(lv)) * len(lv) ** -0.2, 0.02)
    grid = np.linspace(lv.min() - 6 * h, lv.max() + 6 * h, 600)
    dens = np.zeros_like(grid)
    for chunk in np.array_split(lv, max(1, len(lv) // 500)):
        dens += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / h) ** 2).sum(axis=1)

    def pi(v):
        v = np.asarray(v, float)
        lnv = np.log(np.maximum(v, 1e-300))
        return np.where(v > 0, np.interp(lnv, grid, dens, left=0.0, right=0.0) / np.maximum(v, 1e-300), 0.0)
    return pi


@dataclass
class _Prepared:
    """Per-impact constants of a fit, computed once: v0 at gamma_scale = 1, yield speed, speed error at the measured
    speed, the spread of ln v0, the observations, and pw = the probability of each cell of true speed given the
    measured one (normal error times the speed prior, normalised)."""
    v_m: np.ndarray
    v0_1: np.ndarray
    v_y: np.ndarray
    sig_v: np.ndarray
    sig_ln: np.ndarray
    stuck: np.ndarray
    e_m: np.ndarray
    has_e: np.ndarray
    pw: np.ndarray
    ln_v_cells: np.ndarray
    error: SpeedError
    sigma_e_extra: float


def _prepare(imp: Impacts, error: SpeedError, sigma_ln_w: float, sigma_e_extra: float,
             speed_prior="data") -> _Prepared:
    n = len(imp)
    wall = np.isinf(imp.r2)
    known = np.isfinite(imp.r1) & (np.isfinite(imp.r2) | wall)
    cache, v0, vy = {}, np.empty(n), np.empty(n)
    for k in range(n):
        r1 = imp.r1[k] if np.isfinite(imp.r1[k]) else R_NOMINAL
        r2 = math.inf if wall[k] else (imp.r2[k] if np.isfinite(imp.r2[k]) else R_NOMINAL)
        key = (r1, r2)
        if key not in cache:
            cache[key] = pair_speeds(imp.material1, imp.material2, r1, r2)
        v0[k], vy[k] = cache[key]
    s_size = np.where(known, 0.0, np.where(wall, size_scatter_ln_v0(True), size_scatter_ln_v0(False)))
    sig_ln = np.sqrt((5.0 / 6.0 * sigma_ln_w) ** 2 + s_size ** 2)
    sig_v = error.sigma(imp.speed)
    # probability of each cell of true speed v given the measured v_m:  P(v | v_m) ~ pi(v) N(v_m; v, sigma_v(v)).
    # The cells are laid out with sigma_v(v_m); the ratio corrects for sigma_v changing with v.
    v = imp.speed[:, None] + _Z_MID[None, :] * sig_v[:, None]
    s_true = error.sigma(np.abs(v))
    z_true = (imp.speed[:, None] - v) / s_true
    ratio = np.exp(-0.5 * z_true ** 2 + 0.5 * _Z_MID[None, :] ** 2) * sig_v[:, None] / s_true
    pw = _DPHI[None, :] * ratio * speed_prior_density(imp.speed, speed_prior)(v)
    tot = pw.sum(axis=1, keepdims=True)
    pw = np.where(tot > 0, pw / np.where(tot > 0, tot, 1.0), _DPHI[None, :])
    return _Prepared(imp.speed, v0, vy, sig_v, sig_ln, imp.stuck.copy(),
                     np.where(imp.has_restitution, imp.restitution, 0.0), imp.has_restitution, pw,
                     np.log(np.maximum(v, 1e-300)), error, sigma_e_extra)


_T_EDGES = np.concatenate([[-np.inf], np.linspace(-4.0, 4.0, 17), [np.inf]])
_T_PROB = np.diff(normal_cdf(_T_EDGES))
_T_MID = np.concatenate([[-4.2247], 0.5 * (_T_EDGES[1:-2] + _T_EDGES[2:-1]), [4.2247]])
_GH3 = (math.sqrt(2.0) * np.polynomial.hermite.hermgauss(3)[0], np.polynomial.hermite.hermgauss(3)[1] / math.sqrt(math.pi))


def _scatter_nodes(sig_ln: np.ndarray) -> tuple:
    """Nodes t and weights for averaging over ln v0 ~ N(ln v0, sig_ln^2), i.e. v0 -> v0 exp(sig_ln t): one node
    without scatter; 3 Gauss-Hermite nodes for a small spread (<= 0.15, e.g. unmeasured radii); otherwise 18 cells of
    the standard normal (width 1/2 between -4 and 4, plus the tails), which keep the rare grains of the tails."""
    top = float(np.max(sig_ln)) if len(sig_ln) else 0.0
    if top == 0.0:
        return np.zeros(1), np.ones(1)
    if top <= 0.15:
        return _GH3
    return _T_MID, _T_PROB


# ---------------------------------------------------------------------------- the likelihood
def _terms(P: _Prepared, log_s: float, e_dry: float, moments: bool = False):
    """Per-impact P(stick), P(bounce) and, for impacts with a measured e, the likelihood density of e_m (if moments:
    instead the weight, mean and second moment of e_m over the bounce side).  Module docstring, section 4.
    Without a spread of v0, each cell of true speed is split at the sticking speed v_s: the part below sticks, the
    part above bounces (the split uses the normal probability inside the cell, so the result is continuous in the
    parameters).  With a spread (sig_ln > 0), P(stick | v) = P(f v0 > v) = Phi((ln(f v0) - ln v)/sig_ln) exactly, and
    the restitution density is averaged over the nodes of _scatter_nodes."""
    f = threshold_factor(e_dry)
    v0c = P.v0_1 * math.exp(5.0 / 6.0 * log_s)
    n = len(P.v_m)
    dens, m1, m2, wsum = (np.zeros(n) for _ in range(4))
    # 1. stick and bounce probabilities, integrated over the cells of true speed
    spread = P.sig_ln > 0
    below, above = np.empty_like(P.pw), np.empty_like(P.pw)
    if np.any(~spread):
        z_s = ((f * v0c[~spread] - P.v_m[~spread]) / P.sig_v[~spread])[:, None]   # v_stick, in sigma_v from v_m
        below[~spread] = np.clip((normal_cdf(z_s) - _PHI_LO) / _DPHI, 0.0, 1.0)
        above[~spread] = np.clip((normal_cdf(-z_s) - _PHIC_HI) / _DPHI, 0.0, 1.0)
    if np.any(spread):
        u = (np.log(f * v0c[spread])[:, None] - P.ln_v_cells[spread]) / P.sig_ln[spread][:, None]
        below[spread], above[spread] = normal_cdf(u), normal_cdf(-u)
    p_stick = np.sum(P.pw * below, axis=1)
    p_bounce = np.sum(P.pw * above, axis=1)
    # 2. the restitution of bounces
    idx = np.arange(n) if moments else np.flatnonzero(P.has_e)
    if len(idx):
        vm, sv, em = P.v_m[idx][:, None], P.sig_v[idx][:, None], P.e_m[idx][:, None]
        pw, sl = P.pw[idx], P.sig_ln[idx]
        for t, w in zip(*_scatter_nodes(sl)):
            v0 = v0c[idx] * np.exp(sl * t)
            zi = ((f * v0 - P.v_m[idx]) / P.sig_v[idx])[:, None]
            cw = pw * np.clip((normal_cdf(-zi) - _PHIC_HI) / _DPHI, 0.0, 1.0)
            inside = (zi > _Z_EDGES[:-1]) & (zi < _Z_EDGES[1:])
            z = np.where(inside, _split_node(zi), _Z_MID)               # node of the part above v_s
            v = np.maximum(vm + z * sv, 1e-12)
            e = restitution_of_x(v / v0[:, None], e_dry)
            mu = e * v / vm
            s2 = (P.error.sigma_abs ** 2 + (P.error.sigma_rel * e * v) ** 2) / vm ** 2 + P.sigma_e_extra ** 2
            if moments:
                m1[idx] += w * np.sum(cw * mu, axis=1)
                m2[idx] += w * np.sum(cw * (s2 + mu * mu), axis=1)
                wsum[idx] += w * np.sum(cw, axis=1)
            else:
                g = np.exp(-0.5 * (em - mu) ** 2 / s2) / np.sqrt(2 * math.pi * s2)
                dens[idx] += w * np.sum(cw * g, axis=1)
    if moments:
        return p_stick, p_bounce, m1, m2, wsum
    return p_stick, p_bounce, dens


def _nll(P: _Prepared, log_s: float, e_dry: float) -> float:
    """-ln L (module docstring, section 4)."""
    p_stick, p_bounce, dens = _terms(P, log_s, e_dry)
    li = np.where(P.stuck, p_stick, np.where(P.has_e, dens, p_bounce))
    li = np.maximum(li, 1e-300)
    return float(-np.sum(np.log(li)))


def log_likelihood(impacts: Impacts, gamma_scale: float, e_dry: float, error: SpeedError | None = None,
                   sigma_ln_w: float = 0.0, sigma_e_extra: float = 0.0, speed_prior="data") -> float:
    """ln L of the impacts for (gamma_scale, e_dry) (module docstring, section 4).  Impacts faster than the yield
    speed are not excluded here (fit() excludes them)."""
    P = _prepare(impacts, error or SpeedError(), sigma_ln_w, sigma_e_extra, speed_prior)
    return -_nll(P, math.log(gamma_scale), e_dry)


def nll_map(impacts: Impacts, gamma_scales, e_values, error: SpeedError | None = None, sigma_ln_w: float = 0.0,
            sigma_e_extra: float = 0.0, speed_prior="data") -> np.ndarray:
    """-ln L on a grid: array [len(e_values), len(gamma_scales)] (for maps of the likelihood)."""
    P = _prepare(impacts, error or SpeedError(), sigma_ln_w, sigma_e_extra, speed_prior)
    return np.array([[_nll(P, math.log(s), e) for s in gamma_scales] for e in e_values])


# ---------------------------------------------------------------------------- predictions
@dataclass
class Prediction:
    """The model's prediction for each impact at (gamma_scale, e_dry).  v0, v_stick, v_yield [m/s] of each pair;
    p_stick = P(stuck | measured speed), with the speed error and any scatter included; e_model = noise-free
    restitution at the measured speed (0 where it would stick); e_mean, e_sd = mean and spread of the MEASURED
    restitution if the impact bounces (nan where a bounce is practically impossible)."""
    v0: np.ndarray
    v_stick: np.ndarray
    v_yield: np.ndarray
    p_stick: np.ndarray
    e_model: np.ndarray
    e_mean: np.ndarray
    e_sd: np.ndarray


def predict(gamma_scale: float, e_dry: float, impacts: Impacts, error: SpeedError | None = None,
            sigma_ln_w: float = 0.0, sigma_e_extra: float = 0.0, speed_prior="data") -> Prediction:
    """Predicted stick probability and restitution of every impact in `impacts` (their measured speeds, radii and
    materials) for surface-energy factor gamma_scale and dry restitution e_dry.  Module docstring, sections 1-4."""
    P = _prepare(impacts, error or SpeedError(), sigma_ln_w, sigma_e_extra, speed_prior)
    log_s = math.log(gamma_scale)
    p_stick, _, m1, m2, wsum = _terms(P, log_s, e_dry, moments=True)
    v0 = P.v0_1 * gamma_scale ** (5.0 / 6.0)
    ok = wsum > 1e-12
    mean = np.where(ok, m1 / np.where(ok, wsum, 1.0), np.nan)
    sd = np.where(ok, np.sqrt(np.maximum(m2 / np.where(ok, wsum, 1.0) - mean ** 2, 0.0)), np.nan)
    return Prediction(v0, threshold_factor(e_dry) * v0, P.v_y, p_stick,
                      restitution_of_x(P.v_m / v0, e_dry), mean, sd)


def curves(gamma_scale: float, e_dry: float, material1="silica", material2=None, r1: float = R_NOMINAL,
           r2: float | None = R_NOMINAL, speeds=None, error: SpeedError | None = None) -> dict:
    """Noise-free model curves for one pair: {"v": speeds [m/s], "e": restitution (0 = stuck), "v_stick": [m/s],
    "p_stick": P(stuck | measured speed v) with the speed error, "v0", "v_yield"}.  r2 None: a wall."""
    material2 = material2 or material1
    v0, vy = pair_speeds(material1, material2, r1, r2, gamma_scale)
    vs = threshold_factor(e_dry) * v0
    v = np.geomspace(0.2 * vs, 30 * vs, 200) if speeds is None else np.asarray(speeds, float)
    err = error or SpeedError()
    return {"v": v, "e": restitution_of_x(v / v0, e_dry), "v_stick": vs, "v0": v0, "v_yield": vy,
            "p_stick": normal_cdf((vs - v) / err.sigma(v))}


# ---------------------------------------------------------------------------- the fit
@dataclass
class FitResult:
    """Best (gamma_scale, e_dry) and their 1-sigma profile-likelihood intervals.  gamma_scale = w / w_cat (catalogue value); w and
    gamma [J/m^2] are the fitted work of adhesion and, for one material, the surface energy w/2.  at_bound says
    whether an interval ran into the search limits (then the data do not constrain that side); for gamma it is also
    set when, at that end of the gamma interval, the best e_dry lies on the e_dry search limit (the end is then set by
    the e_dry search range, not by the data).  cov: covariance of (ln gamma_scale, e_dry) from the Hessian (curvature
    matrix) of -ln L.  v_stick: sticking speed of two grains of radius R_NOMINAL (diameter 97.7 um), or of one on the
    wall, at the best fit [m/s].  n_used, n_stick, n_restitution, n_dropped (faster than 0.9 v_y).  kept: boolean mask
    over the impacts passed to fit(), True for those used (use impacts.select(kept) for residuals)."""
    material1: str
    material2: str
    gamma_scale: float
    gamma_scale_interval: tuple
    e_dry: float
    e_dry_interval: tuple
    w: float
    gamma: float | None
    cov: np.ndarray
    at_bound: dict
    nll_min: float
    v_stick: float
    n_used: int
    n_stick: int
    n_restitution: int
    n_dropped: int
    settings: dict
    kept: np.ndarray | None = None

    @property
    def e_dry_halfwidth(self) -> float:
        """Half the width of the 1-sigma interval of e_dry."""
        return 0.5 * (self.e_dry_interval[1] - self.e_dry_interval[0])

    @property
    def gamma_rel_halfwidth(self) -> float:
        """Half the width of the 1-sigma interval of ln gamma_scale (= relative error of gamma for small errors)."""
        return 0.5 * math.log(self.gamma_scale_interval[1] / self.gamma_scale_interval[0])

    def covers(self, gamma_scale: float, e_dry: float) -> dict:
        """Does each 1-sigma interval contain the given (true) value?"""
        return {"gamma_scale": self.gamma_scale_interval[0] <= gamma_scale <= self.gamma_scale_interval[1],
                "e_dry": self.e_dry_interval[0] <= e_dry <= self.e_dry_interval[1]}

    def summary(self) -> str:
        """Plain-text summary of the fit."""
        lo, hi = self.gamma_scale_interval
        el, eh = self.e_dry_interval
        g = f", gamma = {self.gamma:.4f} J/m^2" if self.gamma is not None else ""
        flags = [k for k, b in self.at_bound.items() if b]
        e_lo, e_hi = self.settings.get("e_bounds", (E_GRID[0], E_GRID[-1]))
        wall = self.settings.get("wall", False)
        who = (f"one grain of {2e6 * R_NOMINAL:.1f} um diameter on the wall" if wall else
               f"two grains of {2e6 * R_NOMINAL:.1f} um diameter (radius {1e6 * R_NOMINAL:.1f} um)")
        note = ""
        if self.at_bound.get("e_dry low") and self.at_bound.get("e_dry high"):
            note = (f"\n  e_dry is not constrained: the best point is arbitrary, and the gamma interval only "
                    f"reflects e_dry in [{e_lo:.2f}, {e_hi:.2f}]")
        return (f"{self.material1}-{self.material2}: {self.n_used} impacts ({self.n_stick} stuck, "
                f"{self.n_restitution} bounces with e), {self.n_dropped} dropped (above 0.9 v_yield)\n"
                f"  gamma_scale = {self.gamma_scale:.3f} (1 sigma {lo:.3f} to {hi:.3f}){g}\n"
                f"  e_dry       = {self.e_dry:.3f} (1 sigma {el:.3f} to {eh:.3f})\n"
                f"  sticking speed of {who}: {self.v_stick * 1e3:.2f} mm/s"
                + (f"\n  interval reaches the search limit: {', '.join(flags)}" if flags else "") + note)


def _minimize_1d(fun, lo: float, hi: float, xatol: float = 1e-4) -> tuple:
    """Bounded scalar minimisation (scipy minimize_scalar, 'bounded' method: Brent's method); returns (x, f(x))."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(fun, bounds=(lo, hi), method="bounded", options={"xatol": xatol})
    return float(r.x), float(r.fun)


def _min_near(fun, guess: float, half: float, lo: float, hi: float, xatol: float, where: bool = False):
    """min of fun near guess: evaluate 7 points in [guess - half, guess + half] (and guess itself), then Brent's method
    between the neighbours of the best point.  (A narrow valley in a flat landscape defeats Brent's method alone.)
    Returns the minimum, or (minimum, its position) if where."""
    xs = np.unique(np.clip(np.concatenate([np.linspace(guess - half, guess + half, 7), [guess]]), lo, hi))
    vals = [fun(x) for x in xs]
    i = int(np.argmin(vals))
    best = (float(vals[i]), float(xs[i]))
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
    if b - a > xatol:
        x, fx = _minimize_1d(fun, a, b, xatol)
        if fx < best[0]:
            best = (fx, x)
    return best if where else best[0]


def _crossing(g, x_best: float, x_end: float, step: float, tol: float = 0.02) -> tuple:
    """Where g(x) = (profile - minimum - 1/2) changes sign between x_best (g = -1/2) and x_end: step out from x_best
    (doubling the step, starting from the curvature estimate `step`) until g > 0, then refine by false position
    (Illinois variant), finishing by bisection if the profile is far from a parabola.  Returns (x, reached_end)."""
    span = abs(x_end - x_best)
    if span == 0:
        return x_end, True
    sgn = 1.0 if x_end > x_best else -1.0
    a, ga = x_best, -0.5
    d = min(max(step if math.isfinite(step) else 0.1 * span, 1e-3 * span), span)
    while True:
        b = x_best + sgn * d
        gb = g(b)
        if gb >= 0:
            break
        if d >= span:
            return x_end, True
        a, ga = b, gb
        d = min(2 * d, span)
    side = 0
    for _ in range(6):
        c = b - gb * (b - a) / (gb - ga)
        gc = g(c)
        if abs(gc) < tol:
            return c, False
        if gc > 0:
            b, gb = c, gc
            if side == -1:
                ga *= 0.5
            side = -1
        else:
            a, ga = c, gc
            if side == 1:
                gb *= 0.5
            side = 1
    for _ in range(20):
        if abs(b - a) < 1e-3 * span:
            break
        c = 0.5 * (a + b)
        if g(c) > 0:
            b = c
        else:
            a = c
    return 0.5 * (a + b), False


def fit(impacts: Impacts, error: SpeedError | None = None, sigma_ln_w: float = 0.0, sigma_e_extra: float = 0.0,
        use_restitution: bool = True, e_bounds: tuple = (E_GRID[0], E_GRID[-1]),
        gamma_scale_bounds: tuple = (0.01, 100.0), profile: bool = True, speed_prior="data") -> FitResult:
    """Maximum-likelihood (gamma_scale, e_dry) for one material pair, with 1-sigma profile-likelihood intervals
    (module docstring, section 5).  use_restitution=False ignores the measured restitutions (stick/bounce only).
    Impacts faster than 0.9 x the yield speed are dropped (the table has no yield), with a warning.

    Search: (1) the best gamma_scale at e_dry = 0.8; (2) a scan of e_dry along the curve of equal sticking speed
    (f(e_dry) s^(5/6) fixed: the direction in which stick/bounce data cannot tell the two numbers apart);
    (3) Nelder-Mead in (ln s, e_dry) from the best point; (4) the profile intervals."""
    from scipy.optimize import minimize
    error = error or SpeedError()
    imp = impacts if use_restitution else impacts.without_restitution()
    P = _prepare(imp, error, sigma_ln_w, sigma_e_extra, speed_prior)
    keep = P.v_m < 0.9 * P.v_y
    kept = keep.copy()
    n_drop = int(np.sum(~keep))
    if n_drop:
        warnings.warn(f"{n_drop} impacts faster than 0.9 x the yield speed were dropped (the table has no yield)")
        imp = imp.select(keep)
        P = _prepare(imp, error, sigma_ln_w, sigma_e_extra, speed_prior)
    ls_lo, ls_hi = math.log(gamma_scale_bounds[0]), math.log(gamma_scale_bounds[1])
    e_lo, e_hi = e_bounds

    def clip_ls(ls):
        return min(max(ls, ls_lo), ls_hi)

    def nll(ls, e):
        return _nll(P, clip_ls(ls), min(max(e, e_lo), e_hi))

    def ridge(ls_ref, e_ref, e):
        """ln s that keeps the sticking speed f(e) s^(5/6) of (ls_ref, e_ref) at e_dry = e."""
        return ls_ref + 1.2 * math.log(threshold_factor(e_ref) / threshold_factor(e))

    e_mid = min(max(0.8, e_lo), e_hi)
    ls8, _ = _minimize_1d(lambda ls: nll(ls, e_mid), ls_lo, ls_hi)
    scan = [(e, clip_ls(ridge(ls8, e_mid, e))) for e in np.linspace(e_lo, e_hi, 15)]
    e0, ls0 = min(scan, key=lambda p: nll(p[1], p[0]))
    de = 0.02 if e0 < e_hi - 0.02 else -0.02
    res = minimize(lambda th: nll(th[0], th[1]) + 1e3 * (max(e_lo - th[1], 0) + max(th[1] - e_hi, 0)),
                   [ls0, e0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 300,
                            "initial_simplex": [[ls0, e0], [ls0 + 0.05, e0], [ls0, e0 + de]]})
    ls_b, e_b = clip_ls(float(res.x[0])), float(min(max(res.x[1], e_lo), e_hi))
    f_b = nll(ls_b, e_b)
    if f_b > nll(ls0, e0):
        ls_b, e_b, f_b = ls0, e0, nll(ls0, e0)

    # curvature (Hessian) of -ln L in (ln s, e_dry), by central differences
    h = np.array([0.02, 0.004])
    x0 = np.array([ls_b, min(max(e_b, e_lo + h[1]), e_hi - h[1])])
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h[i], np.eye(2)[j] * h[j]
            H[i, j] = (nll(*(x0 + ei + ej)) - nll(*(x0 + ei - ej)) - nll(*(x0 - ei + ej)) + nll(*(x0 - ei - ej))) \
                / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
    sd = [math.sqrt(cov[k, k]) if np.isfinite(cov[k, k]) and cov[k, k] > 0 else math.inf for k in range(2)]

    if profile:
        def prof_e(e):
            """min over ln s at fixed e_dry (searched near the curve of equal sticking speed), minus f_b + 1/2."""
            return _min_near(lambda ls: nll(ls, e), ridge(ls_b, e_b, e), 0.4, ls_lo, ls_hi, 1e-3) - f_b - 0.5

        def e_at(ls):
            """(min over e_dry at fixed ln s, the e_dry where it lies), searched near the curve of equal sticking
            speed."""
            g = e_of_threshold_factor(threshold_factor(e_b) * math.exp(5.0 / 6.0 * (ls_b - ls)))
            return _min_near(lambda e: nll(ls, e), g, 0.08, e_lo, e_hi, 1e-4, where=True)

        def prof_s(ls):
            """min over e_dry at fixed ln s, minus f_b + 1/2."""
            return e_at(ls)[0] - f_b - 0.5

        def e_limited(ls):
            """True if at this ln s the best e_dry lies on the e_dry search limit."""
            e_best = e_at(ls)[1]
            return e_best <= e_lo + 2e-3 or e_best >= e_hi - 2e-3

        e_l, b1 = _crossing(prof_e, e_b, e_lo, sd[1])
        e_h, b2 = _crossing(prof_e, e_b, e_hi, sd[1])
        s_l, b3 = _crossing(prof_s, ls_b, ls_lo, sd[0])
        s_h, b4 = _crossing(prof_s, ls_b, ls_hi, sd[0])
        b3 = b3 or e_limited(s_l)
        b4 = b4 or e_limited(s_h)
        at_bound = {"e_dry low": b1, "e_dry high": b2, "gamma low": b3, "gamma high": b4}
    else:
        e_l, e_h = max(e_b - sd[1], e_lo), min(e_b + sd[1], e_hi)
        s_l, s_h = max(ls_b - sd[0], ls_lo), min(ls_b + sd[0], ls_hi)
        at_bound = {"e_dry low": e_l <= e_lo, "e_dry high": e_h >= e_hi, "gamma low": s_l <= ls_lo,
                    "gamma high": s_h >= ls_hi}

    s_b = math.exp(ls_b)
    m1, m2 = get_material(imp.material1), get_material(imp.material2)
    w = 2 * math.sqrt(m1.gamma * m2.gamma) * s_b
    wall = len(imp) > 0 and bool(np.all(np.isinf(imp.r2)))
    v0, _ = pair_speeds(imp.material1, imp.material2, R_NOMINAL, None if wall else R_NOMINAL, s_b)
    return FitResult(imp.material1, imp.material2, s_b, (math.exp(s_l), math.exp(s_h)), e_b, (e_l, e_h), w,
                     w / 2 if m1.name == m2.name else None, cov, at_bound, f_b, threshold_factor(e_b) * v0,
                     len(imp), int(np.sum(imp.stuck)), int(np.sum(imp.has_restitution)), n_drop,
                     {"error": error, "sigma_ln_w": sigma_ln_w, "sigma_e_extra": sigma_e_extra,
                      "use_restitution": use_restitution, "speed_prior": speed_prior, "e_bounds": (e_lo, e_hi),
                      "wall": wall}, kept)


# ---------------------------------------------------------------------------- residuals: is the model right?
def residuals(impacts: Impacts, gamma_scale: float, e_dry: float, error: SpeedError | None = None,
              sigma_ln_w: float = 0.0, sigma_e_extra: float = 0.0, n_bins: int = 8, speed_prior="data") -> dict:
    """How well (gamma_scale, e_dry) describe the impacts.  Returns
    "e_pull": (e_m - e_mean)/e_sd for each bounce with a measured e (should scatter as N(0, 1));
    "x": measured speed / predicted sticking speed of each impact;
    "bins": per speed bin (in x, log-spaced) the observed and predicted stuck fraction and its binomial error, and
    the mean e pull with its error; "chi2_e": (chi^2, N) with chi^2 = sum of e_pull^2 over the N bounces with a
    measured e.  If the model is right, chi^2 = N +- sqrt(2N) (the mean and spread of a chi^2 distribution with N
    degrees of freedom; the two fitted parameters change this by little for N in the hundreds).
    Pass only the impacts the fit used (impacts.select(fit_result.kept))."""
    pr = predict(gamma_scale, e_dry, impacts, error, sigma_ln_w, sigma_e_extra, speed_prior)
    x = impacts.speed / pr.v_stick
    has = impacts.has_restitution & np.isfinite(pr.e_mean) & (pr.e_sd > 0)
    pull = np.full(len(impacts), np.nan)
    pull[has] = (impacts.restitution[has] - pr.e_mean[has]) / pr.e_sd[has]
    edges = np.geomspace(max(x.min(), 1e-3) * 0.999, x.max() * 1.001, n_bins + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        k = int(m.sum())
        if not k:
            continue
        p = float(pr.p_stick[m].mean())
        mp = m & has
        rows.append({"x_lo": a, "x_hi": b, "n": k, "stuck_observed": float(impacts.stuck[m].mean()),
                     "stuck_predicted": p, "stuck_error": math.sqrt(max(p * (1 - p), 1e-12) / k),
                     "n_e": int(mp.sum()), "e_pull_mean": float(np.mean(pull[mp])) if mp.any() else math.nan,
                     "e_pull_error": 1 / math.sqrt(mp.sum()) if mp.any() else math.nan})
    return {"x": x, "e_pull": pull, "bins": rows, "chi2_e": (float(np.nansum(pull ** 2)), int(has.sum())),
            "prediction": pr}


# ---------------------------------------------------------------------------- synthetic flight data
def make_synthetic_flight(n: int = 300, material: str = "silica", partner: str | None = None,
                          gamma_scale: float = 1.0, e_dry: float | None = None, speeds=None,
                          speed_range: tuple = (0.3, 10.0), d_range: tuple = D_SIEVE, wall: bool = False,
                          error: SpeedError | None = None, sigma_ln_w: float = 0.0, measure_radii: bool = False,
                          restitution_fraction: float = 1.0, exact: bool = False, e_loss=None,
                          seed: int = 0) -> Impacts:
    """Fake flight impacts from KNOWN (gamma_scale, e_dry).
    - grains: radii from the sieve cut d_range (scenes.sieve_radii: log-normal, 90 % inside; shape ASSUMED);
      partner None = same material; wall=True = grains hitting a wall of material `partner`.
    - true impact speeds: speeds(rng, n) [m/s] if given, else log-uniform from speed_range[0] to speed_range[1] times
      the TRUE sticking speed of two R_NOMINAL grains (or one on the wall) at the given gamma_scale and e_dry
      (ASSUMED: it spans the threshold and the plateau of e; the flight's speed distribution is not known; the 3-D
      page uses |N(15, 7.5)| mm/s).
    - each PAIR's adhesion is multiplied by exp(sigma_ln_w N(0,1)) (default 0: identical surfaces; the simulator's
      per-grain 0.6 is 0.42 per grain-grain pair and 0.6 per grain-wall pair, module docstring section 4).
    - outcome and true e from the table (exact=True: collide.impact on each actual pair, yield included).
    - e_loss (optional, to test a WRONG model): function (v/v_stick) -> factor multiplying the true e.
    - measured speed |v + N(0, sigma_v(v))| and signed exit speed e v + N(0, sigma_v(e v)); e_m = exit/measured
      speed (it can come out slightly negative or above 1 from noise, as real measurements can);
      a fraction restitution_fraction of the bounces has a measured e; radii are reported only if measure_radii.
    The true values are kept in Impacts.truth."""
    rng = np.random.default_rng(seed)
    from .scenes import sieve_radii
    partner = partner or material
    pair_e = make_pair(material, R_NOMINAL, partner, None if wall else R_NOMINAL).e_dry
    e_dry = pair_e if e_dry is None else e_dry
    error = error or SpeedError()
    r1 = sieve_radii(n, d_range[0], d_range[1], 0.9, rng)
    r2 = np.full(n, math.inf) if wall else sieve_radii(n, d_range[0], d_range[1], 0.9, rng)
    f_adh = gamma_scale * np.exp(sigma_ln_w * rng.standard_normal(n))
    if speeds is None:
        v0_ref, _ = pair_speeds(material, partner, R_NOMINAL, None if wall else R_NOMINAL, gamma_scale)
        v_ref = threshold_factor(e_dry) * v0_ref
        v = v_ref * np.exp(rng.uniform(math.log(speed_range[0]), math.log(speed_range[1]), n))
    else:
        v = np.asarray(speeds(rng, n), float)
    v0 = np.array([pair_speeds(material, partner, a, b, s)[0] for a, b, s in zip(r1, r2, f_adh)])
    if exact:
        e_true = np.empty(n)
        for k in range(n):
            p = make_pair(material, r1[k], partner, None if wall else r2[k], f_adh=f_adh[k])
            e_true[k] = impact(replace(p, e_dry=e_dry), v[k]).e
    else:
        e_true = np.array([float(restitution_of_x(v[k] / v0[k], e_dry)) for k in range(n)])
    if e_loss is not None:
        e_true = e_true * np.asarray(e_loss(v / (threshold_factor(e_dry) * v0)), float)
    stuck = e_true <= 0.0
    v_m = np.abs(v + error.sigma(v) * rng.standard_normal(n))
    v_out = e_true * v + error.sigma(e_true * v) * rng.standard_normal(n)
    measured = ~stuck & (rng.random(n) < restitution_fraction)
    e_m = np.where(measured, v_out / v_m, np.nan)
    truth = {"speed": v, "restitution": e_true, "r1": r1, "r2": r2, "f_adh": f_adh, "v0": v0,
             "gamma_scale": gamma_scale, "e_dry": e_dry, "sigma_ln_w": sigma_ln_w, "seed": seed}
    return Impacts(v_m, stuck, e_m, r1 if measure_radii else np.nan,
                   r2 if (measure_radii or wall) else np.nan, material, partner, truth)


# ---------------------------------------------------------------------------- CSV files
CSV_COLUMNS = ("speed_mm_s", "stuck", "restitution", "radius1_um", "radius2_um", "material1", "material2")
CSV_HELP = """\
# DOLL-E impact list (read by dolle_core.inverse.read_impacts_csv). One row per filmed impact. Columns:
#   speed_mm_s   impact speed along the line of centres (3-D normal approach speed from two camera views; a
#                single camera sees only its projection, about 21 % low on average), mm/s
#   stuck        1 if the grains stuck, 0 if they bounced
#   restitution  for a bounce: exit speed / impact speed (3-D normal components, two camera views); blank if
#                stuck or not measured.  Use the SIGNED exit component (noise can make it slightly negative) and keep
#                values above 1: both are expected from measurement noise and must not be clipped or dropped
#   radius1_um   radius of grain 1 in um if measured, else blank
#   radius2_um   radius of grain 2 in um if measured, else blank; the word wall for a grain hitting a wall
#   material1    material of grain 1, a name in dolle_core.materials.MATERIALS (e.g. silica, graphite)
#   material2    material of grain 2 (or of the wall)
# Lines starting with # are comments.
"""


def write_impacts_csv(path, impacts, comment: str = "") -> None:
    """Write one Impacts (or a list of them) as CSV (CSV_COLUMNS, CSV_HELP), preceded by `comment` lines (each
    prefixed with '# ')."""
    groups = [impacts] if isinstance(impacts, Impacts) else list(impacts)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for line in comment.splitlines():
            fh.write(f"# {line}\n")
        fh.write(CSV_HELP)
        wr = csv.writer(fh)
        wr.writerow(CSV_COLUMNS)
        num = lambda x: "" if not np.isfinite(x) else repr(float(x))          # noqa: E731
        for g in groups:
            for k in range(len(g)):
                r2 = "wall" if np.isinf(g.r2[k]) else num(g.r2[k] * 1e6)
                e = "" if g.stuck[k] or not np.isfinite(g.restitution[k]) else repr(float(g.restitution[k]))
                wr.writerow([repr(float(g.speed[k] * 1e3)), int(g.stuck[k]), e, num(g.r1[k] * 1e6), r2,
                             g.material1, g.material2])


def read_impacts_csv(path) -> dict:
    """Read an impact list written in the CSV_COLUMNS format.  Returns {pair name: Impacts}, one entry per
    (material1, material2, wall or not); for grain-grain rows the two materials are put in alphabetical order (with
    their radii swapped to match), so graphite,silica and silica,graphite rows are fitted together.  Raises
    ValueError with the line number for a malformed row."""
    rows: dict = {}
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [(i + 1, ln) for i, ln in enumerate(fh) if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path}: no data")
    header = next(csv.reader([lines[0][1]]))
    header = [h.strip() for h in header]
    missing = [c for c in CSV_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; expected {list(CSV_COLUMNS)}")
    col = {c: header.index(c) for c in CSV_COLUMNS}
    for lineno, ln in lines[1:]:
        cells = [c.strip() for c in next(csv.reader([ln]))]
        try:
            v = float(cells[col["speed_mm_s"]]) * 1e-3
            stuck = {"1": True, "0": False}[cells[col["stuck"]]]
            e = float(cells[col["restitution"]]) if cells[col["restitution"]] else math.nan
            r1 = float(cells[col["radius1_um"]]) * 1e-6 if cells[col["radius1_um"]] else math.nan
            c2 = cells[col["radius2_um"]].lower()
            r2 = math.inf if c2 == "wall" else (float(c2) * 1e-6 if c2 else math.nan)
            m1, m2 = cells[col["material1"]], cells[col["material2"]]
            get_material(m1), get_material(m2)
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError(f"{path}, line {lineno}: cannot read {ln.strip()!r} ({exc})") from None
        if not v > 0:
            raise ValueError(f"{path}, line {lineno}: speed must be > 0")
        if stuck and np.isfinite(e):
            raise ValueError(f"{path}, line {lineno}: a stuck impact has no restitution")
        if not math.isinf(r2) and m2 < m1:
            m1, m2, r1, r2 = m2, m1, r2, r1
        rows.setdefault((m1, m2, math.isinf(r2)), []).append((v, stuck, e, r1, r2))
    out = {}
    for (m1, m2, wall), rr in rows.items():
        a = np.array(rr, dtype=float)
        imp = Impacts(a[:, 0], a[:, 1].astype(bool), a[:, 2], a[:, 3], a[:, 4], m1, m2)
        out[imp.pair] = imp
    return out


# ---------------------------------------------------------------------------- other e_dry values in any model
@contextmanager
def override_material(name: str, **changes):
    """Temporarily change fields of a material in materials.MATERIALS (e.g. override_material("silica",
    e_dry=0.9)), so that code which looks materials up by name (the 3-D simulation, scenes.py) uses the new values.
    The table entry is restored on exit."""
    old = _materials.MATERIALS[name]
    _materials.MATERIALS[name] = replace(old, **changes)
    try:
        yield _materials.MATERIALS[name]
    finally:
        _materials.MATERIALS[name] = old

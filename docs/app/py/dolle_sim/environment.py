"""Flight environment: residual acceleration, g-jitter, payload spin and gas drag.

Everything is expressed in the *chamber frame* (axes fixed to the experiment,
which rotates with the rocket). In that frame a free particle feels
    a = a_app(t) - 2 Omega x v - Omega x (Omega x r_axis)
where a_app is minus the rocket's non-gravitational acceleration (aerodynamic
drag, thrust tails, ACS pulses), Omega the spin vector (rocket long axis = +z)
and r_axis the particle's position measured from the spin axis.

Residual gas co-rotates with the chamber, so drag acts on the velocity
relative to the chamber.

Also modelled (all optional, all in the chamber frame):
* Gravity gradient (tidal) acceleration of a free-falling payload,
      a_tid = (GM/r^3) [3 (x.e_up) e_up - x],
  x = position relative to the rocket's centre of mass; at most 2 GM/r^3 |x|
  (~3e-8 g per 10 cm at 130 km).
* Uniform external electric field E [V/m] (e.g. from charged/biased walls).
* Uniform magnetic field B [T] (Earth's field ~ 25-65 uT) -> Lorentz force q v x B.
  The local vertical and Earth's B field are fixed in inertial space, so in the
  spinning chamber frame they rotate as R(-Omega t); both are given at t = 0.
* Gas torque on spinning grains (free-molecular <-> continuum interpolation).
* Temperature gradient across the chamber [K/m] (thermophoresis) and the
  camera/LED illumination [W/m^2] (photophoresis, radiation pressure).

Presets (check the numbers against your launch provider's user guide):
* "sounding_rocket": a despun sounding rocket with attitude control above
  ~100 km, giving a ~3-4 minute window of low residual acceleration:
  1e-4 g quasi-steady + 5e-5 g rms jitter and a slow 0.0167 Hz (1 rpm) roll
  about the long axis.
* "spinning_rocket": a spin-stabilised rocket that is not despun. A few Hz of
  spin produces centrifugal accelerations of order 1 g at a few cm from the
  axis. This dominates over everything else and must be designed around
  (mount the chamber on-axis, or use a despun flight).
* "ideal": no residual acceleration; "drop_tower": a ~4.7 s drop.

Provenance: Standard rotating-frame mechanics and aerosol physics (drag:
Stokes with the Cunningham slip correction, Davies 1945 constants). The preset
values are representative assumptions, not vehicle data. No REBOUND code.
"""
from __future__ import annotations

import math
import csv

import numpy as np

G0 = 9.80665
K_B = 1.380649e-23
GM_EARTH = 3.986004418e14
R_EARTH = 6.371e6

GASES = {
    # molar mass [kg/mol], dynamic viscosity at 293 K [Pa s]
    "air": (0.02897, 1.81e-5),
    "nitrogen": (0.02801, 1.76e-5),
    "argon": (0.03995, 2.23e-5),
    "helium": (0.004003, 1.96e-5),
}

PRESETS = {
    "sounding_rocket": dict(residual_accel_g=[0.0, 0.0, 1e-4], jitter_rms_g=5e-5,
                            jitter_corr_time=0.05, spin_rate_hz=0.0167, window_s=210.0),
    "spinning_rocket": dict(residual_accel_g=[0.0, 0.0, 1e-4], jitter_rms_g=1e-4,
                            jitter_corr_time=0.05, spin_rate_hz=4.0, window_s=180.0),
    "ideal": dict(residual_accel_g=[0.0, 0.0, 0.0], jitter_rms_g=0.0,
                  jitter_corr_time=0.05, spin_rate_hz=0.0, window_s=210.0),
    "drop_tower": dict(residual_accel_g=[0.0, 0.0, 1e-5], jitter_rms_g=1e-6,
                       jitter_corr_time=0.05, spin_rate_hz=0.0, window_s=4.7),
}


class Environment:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        name = cfg.get("preset", "sounding_rocket")
        if name not in PRESETS:
            raise KeyError(f"unknown environment preset '{name}' (choose from {sorted(PRESETS)})")
        preset = dict(PRESETS[name])
        preset.update({k: v for k, v in cfg.items() if k != "preset"})
        c = self.cfg = preset
        self.rng = rng
        self.a_res = np.asarray(c.get("residual_accel_g", [0, 0, 0]), float) * G0
        self.jitter_rms = float(c.get("jitter_rms_g", 0.0)) * G0
        self.jitter_tau = float(c.get("jitter_corr_time", 0.05))
        axis = np.asarray(c.get("spin_axis", [0.0, 0.0, 1.0]), float)
        self.spin_hz = float(c.get("spin_rate_hz", 0.0))
        self.omega = 2 * math.pi * self.spin_hz * axis / np.linalg.norm(axis)
        self.window_s = float(c.get("window_s", 210.0))

        # gravity gradient of a free-falling payload
        self.gravity_gradient = bool(c.get("gravity_gradient", True))
        self.altitude = float(c.get("altitude_m", 130e3))
        up = np.asarray(c.get("local_vertical", [0.0, 0.0, 1.0]), float)
        self.e_up = up / np.linalg.norm(up)
        self.tidal_rate = GM_EARTH / (R_EARTH + self.altitude) ** 3     # [1/s^2]

        # electromagnetic fields (chamber frame)
        self.E_field = np.asarray(c.get("electric_field_v_m", [0.0, 0.0, 0.0]), float)
        self.B_field = np.asarray(c.get("magnetic_field_t", [0.0, 2.0e-5, -4.5e-5]), float)
        # start the coloured noise in its stationary distribution (not at zero)
        self._jitter = self.jitter_rms * rng.standard_normal(3) if self.jitter_rms > 0 else np.zeros(3)
        self._csv = None
        if c.get("accel_csv"):
            self._load_csv(c["accel_csv"])

        # gas
        self.P = float(c.get("pressure_pa", 1.0))
        self.T = float(c.get("temperature_k", 293.0))
        M, mu293 = GASES[c.get("gas", "air")]
        self.mu = mu293 * (self.T / 293.0) ** 0.7       # weakly T-dependent, P-independent
        m_mol = M / 6.02214076e23
        self.rho_gas = self.P * m_mol / (K_B * self.T)
        self.v_th = math.sqrt(8 * K_B * self.T / (math.pi * m_mol))
        self.mfp = (self.mu / self.P) * math.sqrt(math.pi * K_B * self.T / (2 * m_mol)) if self.P > 0 else math.inf
        # translational thermal conductivity of the gas (for thermophoresis)
        self.kappa_tr = 15.0 / 4.0 * (K_B / m_mol) * self.mu
        self.grad_T = np.asarray(c.get("temperature_gradient_k_m", [0.0, 0.0, 0.0]), float)
        light = c.get("illumination", {}) or {}
        self.light_I = float(light.get("intensity_w_m2", 0.0))
        d = np.asarray(light.get("direction", [1.0, 0.0, 0.0]), float)
        self.light_dir = d / np.linalg.norm(d)

    # ------------------------------------------------------------------ accel
    def _load_csv(self, path):
        """CSV with columns t, ax, ay, az in g (chamber frame), e.g. from flight IMU data."""
        rows = []
        with open(path, newline="") as f:
            for r in csv.reader(f):
                try:
                    rows.append([float(x) for x in r[:4]])
                except ValueError:
                    continue   # header
        arr = np.asarray(rows)
        self._csv = (arr[:, 0], arr[:, 1:4] * G0)

    def apparent_accel(self, t: float, dt: float) -> np.ndarray:
        """Uniform apparent acceleration in the chamber frame at time t."""
        if self._csv is not None:
            tt, aa = self._csv
            return np.array([np.interp(t, tt, aa[:, k]) for k in range(3)])
        if self.jitter_rms > 0:
            # Ornstein-Uhlenbeck (coloured) noise with the given rms and correlation time
            f = math.exp(-dt / self.jitter_tau)
            self._jitter = f * self._jitter + math.sqrt(1 - f * f) * self.jitter_rms * self.rng.standard_normal(3)
        return self.a_res + self._jitter

    # ------------------------------------------------------------------- drag
    def drag_coeff(self, radius) -> np.ndarray:
        """Linear drag coefficient c [kg/s] with F = -c v, valid from continuum to free-molecular.

        Stokes drag with the Cunningham slip correction; in the free-molecular
        (low pressure) limit this reduces to the Epstein law.
        """
        radius = np.asarray(radius, float)
        if self.P <= 0:
            return np.zeros_like(radius)
        Kn = self.mfp / radius
        Cc = 1 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))
        return 6 * math.pi * self.mu * radius / Cc

    def rot_drag_coeff(self, radius) -> np.ndarray:
        """Rotational drag coefficient c_rot [N m s] with torque = -c_rot * omega for a sphere.

        Continuum (Stokes): 8 pi mu R^3. Free-molecular, diffuse reflection:
        (2 pi / 3) rho_g v_th R^4. Written as c_rot = 0.4 c_trans R^2 / ratio where
        ratio = tau_rot / tau_trans goes from 0.3 (continuum) to 1.114 (free molecular).
        """
        radius = np.asarray(radius, float)
        if self.P <= 0:
            return np.zeros_like(radius)
        Kn = self.mfp / radius
        ratio = 0.3 + (1.114 - 0.3) * Kn / (1 + Kn)
        return 0.4 * self.drag_coeff(radius) * radius ** 2 / ratio

    def to_chamber(self, v_inertial: np.ndarray, t: float) -> np.ndarray:
        """An inertially fixed vector (given at t=0) expressed in the spinning chamber frame."""
        W = float(np.linalg.norm(self.omega))
        if W == 0:
            return v_inertial
        k = self.omega / W
        th = -W * t
        return (v_inertial * math.cos(th) + np.cross(k, v_inertial) * math.sin(th)
                + k * (k @ v_inertial) * (1 - math.cos(th)))

    def magnetic_field(self, t: float) -> np.ndarray:
        return self.to_chamber(self.B_field, t)

    def tidal_accel(self, x_cg: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Gravity-gradient acceleration at positions x_cg (relative to the rocket CG)."""
        if not self.gravity_gradient:
            return np.zeros_like(x_cg)
        up = self.to_chamber(self.e_up, t)
        return self.tidal_rate * (3 * (x_cg @ up)[:, None] * up[None, :] - x_cg)

    def describe(self) -> dict:
        return {
            "pressure_pa": self.P, "temperature_k": self.T,
            "gas_density_kg_m3": self.rho_gas, "mean_free_path_m": self.mfp,
            "residual_accel_m_s2": self.a_res.tolist(), "jitter_rms_m_s2": self.jitter_rms,
            "spin_rate_hz": self.spin_hz, "spin_axis": (self.omega / max(np.linalg.norm(self.omega), 1e-30)).tolist(),
            "window_s": self.window_s, "gravity_gradient": self.gravity_gradient,
            "tidal_accel_max_per_m_g": 2 * self.tidal_rate / G0,
            "electric_field_v_m": self.E_field.tolist(), "magnetic_field_t": self.B_field.tolist(),
            "temperature_gradient_k_m": self.grad_T.tolist(), "illumination_w_m2": self.light_I,
            "illumination_direction": self.light_dir.tolist(),
        }

"""Flight timeline: boost, coast, spin, de-spin and the microgravity window, as loads in the tubule frame.

The tubule (chamber) frame has +z along the rocket's long axis pointing to the nose; the aft end of a
standing tubule is its z = 0 face. The payload's axial acceleration a_ax(t) (+ = speeding up) appears to
a free grain as -a_ax z-hat, so thrust presses dust onto the aft face and drag deceleration pulls it
toward the nose end. The payload spins about +z at Omega(t), which adds centrifugal, Coriolis and
Euler (-dOmega/dt x r) accelerations.

Sources and their status (see provenance.py):
* RockSat User Guide 2026 (NASA Wallops Flight Facility, Rev-A, 4 Sept 2025), Table 3-1 and Sec. 3.3:
  5.2 s first-stage (Terrier) burn, 12.2 s coast, 11.7 s second-stage (Improved Malemute) burn;
  25 g sustained ascent load (design envelope) with ~50 g longitudinal impulses possible; spin ~4.8 Hz
  at second-stage burnout, de-spun to ~0-1 Hz shortly after burnout, ~0 Hz at apogee; apogee ~160 km;
  experiment power off T+332 s.  [published]
* NASA Sounding Rocket Program Office mission fact sheets for the same vehicle family (AWESOME 46.034/
  46.035, SEED 46.026/46.037, 2025): de-spin ~T+54 s, apogee ~T+197-200 s at ~158-161 km. From these
  the 100 km crossings are ~T+88 s and ~T+313 s.  [published analogs, used for defaults]
* Ettl & Pfaender (2009, 19th ESA PAC Symposium): yo-yo de-spin in 0.9 s; drag decreases from 1e-4 g at
  100 km to ~2e-6 g at apogee.  [published]
* DERIVED (order of magnitude, swept): the drag deceleration in the coast between the burns, ~3-5 g, from
  the fact-sheet trajectory points (AWESOME: Terrier burnout 5.2 s at 1.7 km, Malemute ignition 14.0 s at
  5.3 km; SEED: 6.2 s at 2.4 km, 13.0 s at 6.8 km) assuming constant accelerations; and after the second
  burn, drag ~ rho v^2 falling with an exponential atmosphere (scale height 7 km) along a ballistic
  vertical climb from burnout (25.7 s at 17.2 km, ~1.65 km/s, fact sheet + derived). Default 3 g at both.
* ASSUMED (no public data; swept in the mission runs): the shape of the thrust inside each burn (flat
  at the envelope by default), the spin-up shape (linear to 4.8 Hz), and the residual spin after de-spin.
* ASSUMED: the payload keeps its attitude (nose along the burnout velocity, no re-orientation), so drag, which
  opposes the velocity, decelerates it toward the nose on the way up and pushes it toward the aft face on the way
  down (the sign flips at apogee, T+~197 s). The quasi-steady residual acceleration (default 1e-4 g, an
  assumed order of magnitude; Ettl & Pfaender give drag alone falling to ~2e-6 g at apogee) is a floor on the
  drag magnitude with the same direction. Below ~100 km on the way down (after ug_end_s) the drag grows again.

Provenance: timeline values as listed above; the rotating-frame kinematics are standard. No REBOUND code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G0 = 9.80665

# documented residual spin options after de-spin [Hz] (see the module docstring and provenance.py)
RESIDUAL_SPIN_OPTIONS_HZ = {
    "guide_upper": 1.0,          # RockSat User Guide 2026: de-spun to ~0.0-1.0 Hz
    "guide_2025": 0.5,           # RockSat-X User's Guide 2025: ~0.5 Hz shortly after burnout
    "yoyo_tolerance": 0.25,      # NSROC yo-yo de-spin accuracy +/-0.25 Hz (NASA/TP-20230006855, 2023)
    "rexus_spec": 0.08,          # REXUS User Manual v7-15 (2017): yo-yo to at most 0.08 Hz
    "rexus12_measured": 0.19 / (2 * math.pi),   # 0.19 rad/s measured on REXUS 12 (Brisset et al. 2016)
    "rate_control": 0.2 / 360.0,  # 0.2 deg/s, NASA rate-control system (SRPO handbook 2005)
    "user_1rpm": 1.0 / 60.0,     # the DOLL-E team's earlier assumption (1 rpm)
}


@dataclass
class FlightProfile:
    """Event-timed load history. All times in s after lift-off (T+0), accelerations in g."""
    burn1: tuple = (0.0, 5.2)              # guide Table 3-1
    burn2: tuple = (17.4, 29.1)            # guide: 5.2 s burn + 12.2 s coast, then 11.7 s burn
    burn_g: float = 25.0                   # guide: 25 g sustained (design envelope; flat shape ASSUMED)
    coast_drag_g: float = 3.0              # DERIVED estimate, coast between burns (3-5 g from fact sheets; sweep)
    post_burn_drag_g: float = 3.0          # DERIVED estimate of the drag deceleration at second-stage burnout (sweep)
    burnout_altitude_m: float = 17.2e3     # fact sheet (AWESOME): Imp. Malemute burnout at 17.2 km
    burnout_speed_m_s: float = 1650.0      # derived from the fact-sheet apogee (1.63-1.67 km/s)
    scale_height_m: float = 7.0e3          # exponential atmosphere scale height (textbook, lower stratosphere)
    spin_burnout_hz: float = 4.8           # guide Table 3-1
    despin_s: float = 54.0                 # analog fact sheets (AWESOME, SEED): de-spin at T+54 s
    despin_duration_s: float = 0.9         # Ettl & Pfaender 2009 (yo-yo)
    residual_spin_hz: float = 1.0 / 60.0   # after de-spin; see RESIDUAL_SPIN_OPTIONS_HZ (sweep)
    residual_g: float = 1e-4               # quasi-steady residual acceleration in the coast (order 1e-4 g)
    ug_start_s: float = 88.0               # ~100 km crossing (derived from the analog apogee time/height)
    ug_end_s: float = 313.0                # ~100 km crossing on the way down (same analogs)
    power_off_s: float = 332.0             # guide Table 3-1
    lateral_g: float = 10.0                # guide Sec. 5.2.5: "about +/- 10 Gs" lateral during ascent (report only)
    impulses: list = field(default_factory=list)  # [{"t_s", "peak_g", "duration_s"}]: axial shocks (+ = forward)

    @classmethod
    def from_config(cls, c: dict | None) -> "FlightProfile":
        c = dict(c or {})
        if isinstance(c.get("residual_spin_hz"), str):
            c["residual_spin_hz"] = RESIDUAL_SPIN_OPTIONS_HZ[c["residual_spin_hz"]]
        for k in ("burn1", "burn2"):
            if k in c:
                c[k] = tuple(c[k])
        return cls(**c)

    # ------------------------------------------------------------------ axial acceleration of the payload
    def axial_g(self, t: float) -> float:
        """Payload acceleration along +z [g] (+ = toward the nose: thrust, or drag on the way down), excluding
        impulses. After burnout: drag (at least residual_g) opposing the ballistic velocity."""
        if self.burn1[0] <= t < self.burn1[1] or self.burn2[0] <= t < self.burn2[1]:
            return self.burn_g
        if self.burn1[1] <= t < self.burn2[0]:
            return -self.coast_drag_g
        if t >= self.burn2[1]:
            v = self.ballistic_speed(t)
            return -math.copysign(max(self.post_burn_drag(t), self.residual_g), v) if v != 0 else 0.0
        return 0.0

    @property
    def apogee_s(self) -> float:
        return self.burn2[1] + self.burnout_speed_m_s / G0

    def ballistic_speed(self, t: float) -> float:
        """Vertical speed [m/s] of a drag-free ballistic flight from burnout (+ = up)."""
        return self.burnout_speed_m_s - G0 * (t - self.burn2[1])

    def post_burn_drag(self, t: float) -> float:
        """Drag magnitude after burnout [g]: ~ rho(h) v^2 along a ballistic vertical flight (DERIVED estimate),
        rising again on the way down."""
        tau = t - self.burn2[1]
        v = self.ballistic_speed(t)
        h = self.burnout_altitude_m + self.burnout_speed_m_s * tau - 0.5 * G0 * tau * tau
        return (self.post_burn_drag_g * math.exp(-(h - self.burnout_altitude_m) / self.scale_height_m)
                * (v / self.burnout_speed_m_s) ** 2)

    def impulse_g(self, t: float) -> float:
        a = 0.0
        for p in self.impulses:
            t0, tau = float(p["t_s"]), float(p["duration_s"])
            if t0 <= t < t0 + tau:
                a += float(p["peak_g"]) * math.sin(math.pi * (t - t0) / tau)   # half-sine shock pulse
        return a

    def apparent_accel(self, t: float) -> np.ndarray:
        """Uniform apparent acceleration on a free grain in the tubule frame [m/s^2]."""
        return np.array([0.0, 0.0, -(self.axial_g(t) + self.impulse_g(t)) * G0])

    # ------------------------------------------------------------------ spin
    def spin_hz(self, t: float) -> float:
        """Roll rate [Hz]: linear spin-up to the burnout value (ASSUMED shape), yo-yo de-spin, residual."""
        if t < 0:
            return 0.0
        if t < self.burn2[1]:
            return self.spin_burnout_hz * t / self.burn2[1]
        if t < self.despin_s:
            return self.spin_burnout_hz
        if t < self.despin_s + self.despin_duration_s:
            f = (t - self.despin_s) / self.despin_duration_s
            return self.spin_burnout_hz + f * (self.residual_spin_hz - self.spin_burnout_hz)
        return self.residual_spin_hz

    def spin_angle(self, t: float, n: int = 200) -> float:
        """Roll angle theta(t) = integral of Omega dt [rad] (piecewise-linear spin, integrated exactly)."""
        knots = sorted({0.0, self.burn2[1], self.despin_s, self.despin_s + self.despin_duration_s})
        theta, t0 = 0.0, 0.0
        for t1 in knots[1:] + [math.inf]:
            hi = min(t, t1)
            if hi > t0:
                theta += 2 * math.pi * 0.5 * (self.spin_hz(t0) + self.spin_hz(hi - 1e-12)) * (hi - t0)
            if t <= t1:
                break
            t0 = t1
        return theta

    def spin_rate_dot(self, t: float, h: float = 1e-4) -> float:
        """dOmega/dt [rad/s^2]."""
        return 2 * math.pi * (self.spin_hz(t + h) - self.spin_hz(max(t - h, 0.0))) / (t + h - max(t - h, 0.0))

    # ------------------------------------------------------------------ summary
    def phases(self) -> list[dict]:
        """Flight phases with the range of the axial acceleration applied in each (axial_g_range, in g)."""
        d1 = self.despin_s + self.despin_duration_s
        p = [
            dict(name="first-stage burn", t0=self.burn1[0], t1=self.burn1[1], source="guide", powered=True),
            dict(name="coast between burns", t0=self.burn1[1], t1=self.burn2[0],
                 source="timing: guide; drag: derived estimate"),
            dict(name="second-stage burn", t0=self.burn2[0], t1=self.burn2[1], source="guide", powered=True),
            dict(name="coast in the upper atmosphere", t0=self.burn2[1], t1=self.despin_s,
                 source="derived drag estimate (decays with altitude)"),
            dict(name="de-spin", t0=self.despin_s, t1=d1,
                 source="time: analog fact sheets; duration: Ettl & Pfaender 2009"),
            dict(name="coast to ~100 km", t0=d1, t1=self.ug_start_s, source="derived"),
            dict(name="microgravity (above ~100 km)", t0=self.ug_start_s, t1=min(self.ug_end_s, self.power_off_s),
                 source="100 km crossings derived from analog fact sheets; residual assumed"),
        ]
        if self.ug_end_s < self.power_off_s:
            p.append(dict(name="descent below ~100 km", t0=self.ug_end_s, t1=self.power_off_s,
                          source="derived drag estimate; power off: guide"))
        for ph in p:
            ts = ph["t0"] + (np.arange(400) + 0.5) / 400 * (ph["t1"] - ph["t0"])
            a = [self.axial_g(t) for t in ts]
            ph["axial_g_range"] = [float(min(a)), float(max(a))]
        return p

    def loads_at(self, t: float, r_xy: np.ndarray) -> dict:
        """Apparent accelerations [g] at a point r_xy (m, from the spin axis, in the deck plane)."""
        w = 2 * math.pi * self.spin_hz(t)
        wd = self.spin_rate_dot(t)
        r = np.array([r_xy[0], r_xy[1], 0.0])
        cf = w * w * r                                        # centrifugal: +Omega^2 r (outward)
        eu = -np.cross([0, 0, wd], r)                          # Euler: -dOmega/dt x r
        ax = self.apparent_accel(t)
        return {"axial": ax / G0, "centrifugal": cf / G0, "euler": eu / G0, "total": (ax + cf + eu) / G0,
                "spin_hz": self.spin_hz(t)}

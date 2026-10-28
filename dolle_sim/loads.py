"""Every load on the grains as a function of flight time, in the frame of the (rigid) tubule.

The walls are fixed in this frame, so shaking, kicks and vibration of the tubule are exact uniform accelerations of
every grain (minus the tubule's acceleration), not moving walls. Sources:
* the quasi-static flight profile (flight.py): thrust, drag, parachute, splash, attitude; spin Omega(t) (piecewise
  linear, so dOmega/dt is exact and piecewise constant);
* release mechanisms, defined by the tubule's ACCELERATION a(tau) = -a_peak sin(2 pi f tau) along one axis for a whole
  number of cycles (a kick is one cycle). The tubule starts and ends at rest relative to the deck, so there is no
  velocity jump (the v0.5 model began the motion with a jump to A omega);
* random or sine vibration of the deck (vibration.py), when a scenario adds it.
Breakpoints (where a load or its slope jumps) are listed so that no step straddles one.
"""
from __future__ import annotations

import math

import numpy as np

from .flight import G0


class Loads:
    def __init__(self, env, release, vibration=None):
        self.env = env
        self.prof = env.profile
        self.t0 = float(env.t0_flight)
        self.axis = np.asarray(env.spin_axis, float)
        self.release = []
        for r in release or []:
            f = float(r["omega"]) / (2 * math.pi)
            n = max(1, round((r["end_s"] - r["start_s"]) * f))
            self.release.append(dict(start=float(r["start_s"]), end=float(r["start_s"]) + n / f, f=f,
                                     a=float(r["peak_g"]) * G0, axis=int(r["axis"]), type=r.get("type", "shake")))
        self.vib = list(vibration or [])
        p = self.prof
        if p is not None:
            knots = [0.0, p.burn2[1], p.despin_s, p.despin_s + p.despin_duration_s]
            self._sk = np.array(knots + [1e9])
            self._sv = np.array([p.spin_hz(k) for k in knots] + [p.residual_spin_hz]) * 2 * math.pi
            self._sv[1] = 2 * math.pi * p.spin_burnout_hz
            br = {0.0, p.burn1[0], p.burn1[1], p.burn2[0], p.burn2[1], p.apogee_s, p.despin_s,
                  p.despin_s + p.despin_duration_s, p.t_attitude, p.ug_start_s, p.ug_end_s, p.chute_s,
                  p.chute_s + p.chute_open_s, p.splash_s, p.splash_s + p.splash_stop_s}
            for a, b in p.event_windows():
                br.update((a, b))
        else:
            self._sk = np.array([0.0, 1e9])
            w = 2 * math.pi * float(env.omega @ env.spin_axis)
            self._sv = np.array([w, w])
            br = set()
        for r in self.release:
            br.update((r["start"], r["end"]))
        for h in self.vib:
            br.update((h.t0, h.t0 + h.T))
        self.breakpoints = np.array(sorted(br))

    # ------------------------------------------------------------------ spin
    def omega(self, T):
        """Spin vector(s) [rad/s] at flight time(s) T."""
        T = np.atleast_1d(np.asarray(T, float))
        return np.interp(T, self._sk, self._sv)[:, None] * self.axis[None, :]

    def omega_dot(self, T):
        T = np.atleast_1d(np.asarray(T, float))
        k = np.clip(np.searchsorted(self._sk, T, side="right") - 1, 0, len(self._sk) - 2)
        rate = (self._sv[k + 1] - self._sv[k]) / (self._sk[k + 1] - self._sk[k])
        return rate[:, None] * self.axis[None, :]

    # ------------------------------------------------------------------ acceleration on a grain
    def profile_accel(self, T):
        T = np.atleast_1d(np.asarray(T, float))
        if self.prof is None:
            return np.repeat(self.env.a_res[None, :], len(T), axis=0)
        return np.array([self.prof.apparent_accel(t) for t in T])

    def release_accel(self, T):
        T = np.atleast_1d(np.asarray(T, float))
        out = np.zeros((len(T), 3))
        for r in self.release:
            on = (T >= r["start"]) & (T < r["end"])
            if on.any():
                tau = T[on] - r["start"]
                # grains feel minus the tubule's acceleration -a sin(2 pi f tau)
                out[on, r["axis"]] += r["a"] * np.sin(2 * math.pi * r["f"] * tau)
        return out

    def vibration_accel(self, T):
        T = np.atleast_1d(np.asarray(T, float))
        out = np.zeros((len(T), 3))
        for h in self.vib:
            out -= h.accel(T).T
        return out

    def accel(self, T, coarse=1e-5):
        """Uniform apparent acceleration [m/s^2] at flight times T (sorted). The smooth flight profile is evaluated
        on a grid of `coarse` seconds and interpolated linearly (error <= coarse^2 max|a''|/8, ~1e-10 g for the
        profile's slowest-varying parts and ~1e-6 of the 0.05 s splash pulse); release waveforms and vibration
        are evaluated exactly at every T."""
        T = np.atleast_1d(np.asarray(T, float))
        if len(T) > 8 and self.prof is not None:
            n = max(2, int(math.ceil((T[-1] - T[0]) / coarse)) + 1)
            g = np.linspace(T[0], T[-1], n)
            ag = self.profile_accel(g)
            a = np.column_stack([np.interp(T, g, ag[:, k]) for k in range(3)])
        else:
            a = self.profile_accel(T)
        return a + self.release_accel(T) + self.vibration_accel(T)

    # ------------------------------------------------------------------ windows
    def fast_window(self, T0, T1):
        """True if a release waveform or vibration acts anywhere in [T0, T1] (loads with ~kHz content)."""
        for r in self.release:
            if r["start"] < T1 and r["end"] > T0:
                return True
        for h in self.vib:
            if h.t0 < T1 and h.t0 + h.T > T0:
                return True
        return False

    def max_freq(self, T0, T1):
        f = 0.0
        for r in self.release:
            if r["start"] < T1 and r["end"] > T0:
                f = max(f, r["f"])
        for h in self.vib:
            if h.t0 < T1 and h.t0 + h.T > T0:
                f = max(f, 2000.0)
        return f

    def next_breakpoint(self, T):
        k = np.searchsorted(self.breakpoints, T + 1e-12, side="right")
        return float(self.breakpoints[k]) if k < len(self.breakpoints) else math.inf

    def breakpoint_in(self, T0, T1):
        return bool(((self.breakpoints > T0 + 1e-12) & (self.breakpoints < T1 - 1e-12)).any())

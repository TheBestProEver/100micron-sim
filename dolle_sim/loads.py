"""Every load on the grains as a function of flight time, in the frame of the (rigid) tubule.

The walls are fixed in this frame, so shaking, kicks and vibration of the tubule are exact uniform accelerations of
every grain (minus the tubule's acceleration), not moving walls. Sources:
* the quasi-static flight profile (flight.py): thrust, drag, parachute, splash, attitude; spin Omega(t) (piecewise
  linear, so dOmega/dt is exact and piecewise constant);
* release mechanisms, defined by the tubule's ACCELERATION a(tau) = -a_peak sin(2 pi f tau) along one axis for a whole
  number of cycles (a kick is one cycle). The tubule starts and ends at rest relative to the deck, so there is no
  velocity jump (the v0.5 model began the motion with a jump to A omega);
* random or sine vibration of the deck (vibration.py), when a scenario adds it.
Breakpoints (where a load or its slope jumps) are listed so that no step straddles one. Every profile piece and
release window is half-open [start, end), so at a breakpoint the stored values are the post-jump ones; accel and
omega_dot with side="-" give the limits from the left, which the engine uses at step ends (a step ending at a
breakpoint lies in the preceding piece). chord_dev and spin_chord bound how far the loads move away, inside a step,
from the chord joining their step-end values (the certificates of resting grains and sleeping clusters).
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
            w = float(env.omega @ env.spin_axis)          # env.omega is in rad/s already
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

    def omega_dot(self, T, side=None):
        """dOmega/dt [rad/s^2], piecewise constant (omega is continuous). At a spin knot the value is the following
        piece's rate; side "-" gives the preceding piece's (the limit from the left)."""
        T = np.atleast_1d(np.asarray(T, float))
        k = np.clip(np.searchsorted(self._sk, T, side="left" if side == "-" else "right") - 1, 0, len(self._sk) - 2)
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

    def vibration_accel(self, T, side=None):
        T = np.atleast_1d(np.asarray(T, float))
        out = np.zeros((len(T), 3))
        for h in self.vib:
            out -= h.accel(T, side).T
        return out

    def accel_seg(self, Ts):
        """accel at the sample times of one integration segment: the first a step's start (limits from the right),
        the others step ends (from the left), so a load that switches on or off at a segment boundary is charged to
        the side it belongs to. The left limits apply to the flight profile too: a segment ending at a breakpoint is
        charged the pre-jump profile at its last sample (the step lies in the preceding piece)."""
        a = self.accel(Ts, side="-")
        a[0] = self.accel(Ts[:1], side="+")[0]
        return a

    def _left_times(self, T):
        """Times at which the limits from the left of the profile and release terms are taken: every T within 1e-9 s
        of a breakpoint moves to bp - 1e-12 s, strictly inside the preceding piece; the others are unchanged.
        Returns the times and the mask of the moved ones."""
        T = np.asarray(T, float)
        bp = self.breakpoints
        if not len(bp):
            return T, np.zeros(T.shape, bool)
        k = np.clip(np.searchsorted(bp, T), 1, len(bp) - 1)
        nearest = np.where(T - bp[k - 1] < bp[k] - T, bp[k - 1], bp[k])
        near = np.abs(nearest - T) <= 1e-9
        return np.where(near, nearest - 1e-12, T), near

    def accel(self, T, coarse=1e-5, side=None):
        """Uniform apparent acceleration [m/s^2] at flight times T (sorted). The smooth flight profile is evaluated
        on a grid of `coarse` seconds and interpolated linearly (error <= coarse^2 max|a''|/8, ~1e-10 g for the
        profile's slowest-varying parts and ~1e-6 of the 0.05 s splash pulse); release waveforms and vibration
        are evaluated exactly at every T.

        side None or "+": the stored values (profile pieces and release windows are half-open [start, end), so at a
        breakpoint they are the post-jump values, the limits from the right). side "-": the limits from the left,
        for step ends: every T within 1e-9 s of a breakpoint is evaluated for the profile and release terms at
        bp - 1e-12 s, strictly inside the preceding piece, where the loads are smooth, so the shift changes the
        value by at most 1e-12 s times the piece's slope: 2e-8 m/s^2 (2e-9 of 1 g) at the end of the splash pulse,
        the steepest profile piece, far less elsewhere, and 2 pi f a 1e-12 for a release waveform (3e-8 of its
        amplitude at 5 kHz); the other times are unchanged, and the vibration term takes side itself (vibration.py).
        On the interpolation grid only the last grid point moves (a grid point at the start of the range is the
        start of the piece the range lies in) and the moved samples are evaluated directly."""
        T = np.atleast_1d(np.asarray(T, float))
        Te, near = self._left_times(T) if side == "-" else (T, None)
        if len(T) > 8 and self.prof is not None:
            n = max(2, int(math.ceil((T[-1] - T[0]) / coarse)) + 1)
            g = np.linspace(T[0], T[-1], n)
            ag = self.profile_accel(g if near is None else np.concatenate([g[:-1], Te[-1:]]))
            a = np.column_stack([np.interp(T, g, ag[:, k]) for k in range(3)])
            if near is not None and near.any():
                a[near] = self.profile_accel(Te[near])
        else:
            a = self.profile_accel(Te)
        return a + self.release_accel(Te) + self.vibration_accel(T, side)

    def _pulse_accel(self, T):
        """The half-sine shocks' part of profile_accel at times T (FlightProfile.shock_pulses)."""
        T = np.atleast_1d(np.asarray(T, float))
        out = np.zeros((len(T), 3))
        for a, dur, peak, d in self.prof.shock_pulses():
            on = (T >= a) & (T < a + dur)
            if on.any():
                out[on] -= peak * np.sin(math.pi * (T[on] - a) / dur)[:, None] * d[None, :]
        return out

    def chord_dev(self, T0, T1):
        """Per-axis bound [m/s^2] on how far the loads can deviate, anywhere inside [T0, T1], from the chord joining
        their values at T0 (from the right) and T1 (from the left); l = T1 - T0. The terms, summed per axis:
        * the flight profile less its half-sine shocks: l^2/8 |a''| with |a''| the doubled second differences of 5
          samples, the heuristic accel_bound uses: a quadratic piece's a'' is exact (so twice its deviation), the
          smooth ascent drag and the tumbling load direction are estimated (check 17 measures it); constant and
          linear pieces give 0. The doubling also covers one slope jump s anywhere in the step that is not in
          kinks_in (the ascent drag meeting the residual floor): at fraction theta of a sub-interval h = l/4 it gives
          a second difference s theta h, so a term s theta l, against a deviation of at most s l theta / 4;
        * each half-sine shock acting inside (parachute opening, splash, user impulses): peak (pi/duration)^2 l^2/8
          along its direction, the exact curvature bound of a sine;
        * each slope jump of the profile inside (T0, T1) (the descent table's points, FlightProfile.kinks_in) and
          each spin knot inside: |Delta slope| l/4 (a kink at mid-step deviates most), Delta slope from one-sided
          differences over 1e-6 s on each side. The uniform acceleration has no kink at a spin knot (that is
          spin_chord's); the spin knots are breakpoints, included only so that the bound holds for any interval;
        * each breakpoint strictly inside: its jump |a(bp+) - a(bp-)| (the engine never certifies a step across one);
        * a release waveform acting inside: twice its amplitude on its axis; vibration acting inside: twice
          hv.abs_max(T0, T1): a signal within +-M deviates from a chord between two of its values by at most 2 M.
          Both are fast windows the engine never certifies across; the terms only keep the bound valid."""
        T0, T1 = float(T0), float(max(T1, T0))
        l = T1 - T0
        out = np.zeros(3)
        if l <= 0:
            return out
        if self.prof is not None:
            Ts = np.linspace(T0, T1, 5)
            Te = np.concatenate([Ts[:-1], self._left_times(Ts[-1:])[0]])
            ap = self.profile_accel(Te) - self._pulse_accel(Te)
            h = l / 4
            d2 = np.abs(ap[2:] - 2 * ap[1:-1] + ap[:-2]).max(0) / (h * h)
            out += 2 * d2 * l * l / 8
            for a, dur, peak, d in self.prof.shock_pulses():
                if a < T1 and a + dur > T0:
                    out += peak * (math.pi / dur) ** 2 * np.abs(d) * l * l / 8
            inside = (self._sk > T0) & (self._sk < T1)
            for k in self.prof.kinks_in(T0, T1) + self._sk[inside].tolist():
                e = 1e-6
                a3 = self.profile_accel(np.array([k - e, k, k + e]))
                out += np.abs(a3[2] - 2 * a3[1] + a3[0]) / e * l / 4
            for bp in self.breakpoints[(self.breakpoints > T0) & (self.breakpoints < T1)]:
                b = np.array([bp])
                out += np.abs(self.accel(b, side="+")[0] - self.accel(b, side="-")[0])
        for r in self.release:
            if r["start"] < T1 and r["end"] > T0:
                out[r["axis"]] += 2 * r["a"]
        for hv in self.vib:
            if hv.t0 <= T1 and hv.t0 + hv.T >= T0:
                out = out + 2 * hv.abs_max(T0, T1)
        return out

    def spin_chord(self, T0, T1):
        """Chord deviation of Omega^2 over [T0, T1] per unit radius [1/s^2], for the centrifugal load (the engine
        multiplies by m |r_perp|): Omega is piecewise linear, so inside a piece Omega^2 is quadratic with second
        derivative 2 Omega_dot^2 and deviates from its chord by 2 Omega_dot^2 l^2/8 = Omega_dot^2 l^2/4 at
        mid-step, exactly; the largest |Omega_dot| of the pieces the interval meets is used (the left limit at T1).
        A spin knot strictly inside (a breakpoint, which the engine never certifies across) adds the kink of Omega^2,
        2 Omega |Delta Omega_dot| l/4, and the jump |Delta Omega_dot| of the Euler term -Omega_dot x r, which is
        otherwise piecewise constant (no deviation inside a piece)."""
        T0, T1 = float(T0), float(max(T1, T0))
        l = T1 - T0
        rate = np.diff(self._sv) / np.diff(self._sk)
        k0 = int(np.clip(np.searchsorted(self._sk, T0, side="right") - 1, 0, len(rate) - 1))
        k1 = int(np.clip(np.searchsorted(self._sk, T1, side="left") - 1, k0, len(rate) - 1))
        out = float(np.abs(rate[k0:k1 + 1]).max()) ** 2 * l * l / 4
        for k in range(k0 + 1, k1 + 1):
            dw = abs(rate[k] - rate[k - 1])
            out += 2 * abs(self._sv[k]) * dw * l / 4 + dw
        return out

    def accel_bound(self, T0, T1):
        """A bound on |a| along each axis over [T0, T1]. The flight profile: at 5 points and at its own knots inside
        the interval (the descent table's points, the peaks of the half-sine shocks), which is exact for its
        constant, piecewise-linear and half-sine parts, plus the curvature margin h^2 |a''| / 8 (|a''| from the
        5 samples' second differences, doubled) for the smooth ascent drag. A release waveform by its amplitude. The
        vibration exactly (the largest |value| of each cubic piece of its spline over the interval, both ends
        included)."""
        T0, T1 = float(T0), float(max(T1, T0))
        Ts = np.linspace(T0, T1, 5)
        ap = self.profile_accel(Ts)
        out = np.abs(ap).max(0)
        if self.prof is not None:
            extra = self.prof.knots_in(T0, T1)
            if len(extra):
                out = np.maximum(out, np.abs(self.profile_accel(np.asarray(extra))).max(0))
        h = (T1 - T0) / 4
        if h > 0:
            d2 = np.abs(ap[2:] - 2 * ap[1:-1] + ap[:-2]).max(0) / (h * h)
            out = out + 2 * d2 * h * h / 8
        for r in self.release:
            if r["start"] < T1 and r["end"] > T0:
                out[r["axis"]] += r["a"]
        for hv in self.vib:
            if hv.t0 <= T1 and hv.t0 + hv.T >= T0:
                out = out + hv.abs_max(T0, T1)
        return out

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

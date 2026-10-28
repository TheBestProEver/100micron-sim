"""Random and sine vibration as time histories of the deck acceleration (tubule frame: grains feel -a_deck).

Specifications (RockSat User Guide 2026, Rev-A DRAFT, Appendix B; the same numbers are "Vehicle Level One" of the
NASA Sounding Rockets User Handbook 2015/2023, Table 6.3.4-1, an acceptance test of the flight unit):
* random, flat 20-2000 Hz: thrust axis 0.051 g^2/Hz (10.0 Grms), each lateral axis 0.029 g^2/Hz (7.6 Grms),
  20 s per axis at full level (the guide does not give the lower-level pre-runs);
* sine sweep, thrust axis only: 10-144 Hz at a constant peak velocity of 3 in/s, 144-2000 Hz at 7 g, 4 octaves per
  minute (amplitudes taken as 0-to-peak, the usual convention; the guide does not say);
* the handbook's "Vehicle Level Two" (12.7 Grms in all axes, 0.01 g^2/Hz at 20 Hz rising 1.8 dB/oct to 0.10 g^2/Hz at
  1000 Hz, flat to 2000 Hz, 10 s per axis) is the class the handbook gives for the Terrier-Improved Malemute and is
  kept as a sensitivity case.
No measured flight PSD of this vehicle is public; the only public flight measurement of a Terrier-boosted NASA
sounding rocket (Terrier-Black Brant, 1998; NTRS 20000114846 and 20030056584) is ~11-12 dB below these test levels.

Synthesis: the standard spectral representation (Shinozuka & Jan 1972, J. Sound Vib. 25, 111; Smallwood & Paez 1993,
Shock Vib. 1, 45): Gaussian Fourier coefficients X_k = N sqrt(G(f_k) df / 2) (a + i b)/sqrt(2), a, b ~ N(0, 1), with
the PSD interpolated log-log between breakpoints and zero outside them; x = irfft(X). The record is one period of
the inverse FFT, so it is a sample of a stationary Gaussian process with exactly the target one-sided PSD in
expectation (variance sum G df = Grms^2). Between samples the acceleration is a cubic spline, and velocity and
displacement are its exact antiderivatives (so a, v and x are mutually consistent to round-off); the spline error at
2000 Hz with fs = 65536 Hz is ~(2 pi f/fs)^4/384 = 3.5e-6 of the amplitude (checked in verification.py).

Checks (verification check "random_vibration", NASA-STD-7001C / MIL-STD-810H practice): Welch PSD (Hann, 50%
overlap) against the spec, Grms, Gaussian moments (skewness 0, kurtosis 3), and spline vs exact Fourier evaluation.
"""
from __future__ import annotations

import math

import numpy as np

G0 = 9.80665
IN = 0.0254

GUIDE_RANDOM = {
    "thrust": {"breakpoints": [(20.0, 0.051), (2000.0, 0.051)], "duration_s": 20.0},
    "lateral": {"breakpoints": [(20.0, 0.029), (2000.0, 0.029)], "duration_s": 20.0},
}
LEVEL_TWO = {"breakpoints": [(20.0, 0.01), (1000.0, 0.10), (2000.0, 0.10)], "duration_s": 10.0}
GUIDE_SINE = {"f1_hz": 10.0, "f2_hz": 2000.0, "oct_per_min": 4.0, "v_peak_m_s": 3 * IN, "a_peak_g": 7.0}


def psd_at(f, breakpoints):
    """One-sided acceleration PSD [g^2/Hz] at f, log-log interpolated between breakpoints, zero outside them."""
    f = np.asarray(f, float)
    bf = np.array([b[0] for b in breakpoints], float)
    bg = np.array([b[1] for b in breakpoints], float)
    out = np.zeros_like(f)
    ok = (f >= bf[0]) & (f <= bf[-1]) & (f > 0)
    out[ok] = np.exp(np.interp(np.log(f[ok]), np.log(bf), np.log(bg)))
    return out


def grms(breakpoints):
    """sqrt of the integral of the PSD [g], exact for log-log segments."""
    tot = 0.0
    for (f1, g1), (f2, g2) in zip(breakpoints[:-1], breakpoints[1:]):
        b = math.log(g2 / g1) / math.log(f2 / f1)
        tot += g1 * f1 / (b + 1) * ((f2 / f1) ** (b + 1) - 1) if abs(b + 1) > 1e-12 else g1 * f1 * math.log(f2 / f1)
    return math.sqrt(tot)


def synthesize_random(breakpoints, duration_s, fs, rng):
    """Samples [g] of a stationary Gaussian acceleration with the given one-sided PSD (period = duration_s)."""
    N = int(round(duration_s * fs))
    N += N % 2
    f = np.fft.rfftfreq(N, 1.0 / fs)
    df = fs / N
    G = psd_at(f, breakpoints)
    X = N * np.sqrt(G * df / 2) * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) / math.sqrt(2)
    X[0] = 0.0
    X[-1] = 0.0
    return np.fft.irfft(X, n=N), X, f


def fourier_eval(X, f, N, t):
    """Exact value of irfft(X, N) continued between samples (band-limited, periodic), at times t [s] (slow)."""
    t = np.atleast_1d(np.asarray(t, float))
    w = 2.0 * np.ones(f.size)
    w[0] = 1.0
    w[-1] = 1.0
    ph = np.exp(2j * np.pi * np.outer(t, f))
    return (ph @ (w * X)).real / N


def sine_sweep(f1, f2, oct_per_min, v_peak, a_peak_g, fs):
    """Logarithmic sine sweep f(t) = f1 2^(R t/60) [Hz], peak acceleration min(2 pi f v_peak, a_peak) [g]."""
    R = oct_per_min / 60.0
    T = math.log2(f2 / f1) / R
    t = np.arange(int(T * fs) + 1) / fs
    f = f1 * 2.0 ** (R * t)
    theta = 2 * math.pi * f1 / (R * math.log(2)) * (2.0 ** (R * t) - 1)
    A = np.minimum(2 * math.pi * f * v_peak, a_peak_g * G0) / G0
    return t, A * np.sin(theta)


class VibrationHistory:
    """Deck acceleration a(t) [m/s^2] along each axis as a cubic spline of samples, with velocity and displacement
    as its exact antiderivatives. Zero outside [t0, t0 + T]. The tubule-frame load on a grain is -a(t)."""

    def __init__(self, t0, fs, acc_g, start_v=None):
        from scipy.interpolate import CubicSpline
        acc = np.atleast_2d(np.asarray(acc_g, float)) * G0          # (3, N)
        if acc.shape[0] != 3:
            raise ValueError("acc_g must have 3 rows (x, y, z)")
        self.t0, self.fs = float(t0), float(fs)
        self.T = (acc.shape[1] - 1) / fs
        t = np.arange(acc.shape[1]) / fs
        self.a = CubicSpline(t, acc, axis=1, bc_type="natural")
        self.v = self.a.antiderivative(1)
        self.x = self.a.antiderivative(2)
        self.rms_g = np.sqrt((acc ** 2).mean(1)) / G0
        self.peak_g = np.abs(acc).max(1) / G0

    def _local(self, T):
        s = np.asarray(T, float) - self.t0
        return s, (s >= 0) & (s <= self.T)

    def accel(self, T):
        s, ok = self._local(T)
        return np.where(ok, self.a(np.clip(s, 0, self.T)), 0.0)

    def velocity(self, T):
        s, ok = self._local(T)
        return np.where(ok, self.v(np.clip(s, 0, self.T)), self.v(self.T) * (s > self.T))

    def displacement(self, T):
        s, ok = self._local(T)
        sc = np.clip(s, 0, self.T)
        x = self.x(sc)
        return np.where(s > self.T, x + self.v(self.T) * (s - self.T), np.where(ok, x, 0.0))


def guide_random_history(t0, axis_breakpoints, duration_s, fs, rng):
    """Independent random vibration on each axis (dict axis -> breakpoints; missing axes are quiet)."""
    acc = np.zeros((3, int(round(duration_s * fs)) + (int(round(duration_s * fs)) % 2)))
    for ax, bp in axis_breakpoints.items():
        k = {"x": 0, "y": 1, "z": 2}[ax] if isinstance(ax, str) else int(ax)
        x, _, _ = synthesize_random(bp, duration_s, fs, rng)
        acc[k] = x
    return VibrationHistory(t0, fs, acc)


def welch_check(x_g, fs, breakpoints, nperseg=2048, band=(40.0, 1980.0)):
    """Welch PSD of a synthesized record vs the spec: max |dB| deviation in the band, Grms, moments."""
    from scipy.signal import welch
    from scipy.stats import kurtosis, skew
    f, P = welch(x_g, fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2)
    sel = (f >= band[0]) & (f <= band[1])
    dev = 10 * np.log10(P[sel] / psd_at(f[sel], breakpoints))
    df = f[1] - f[0]
    inb = (f >= breakpoints[0][0]) & (f <= breakpoints[-1][0])
    n_avg = 2 * (len(x_g) // nperseg) - 1
    return dict(max_abs_dev_db=float(np.abs(dev).max()), mean_dev_db=float(dev.mean()),
                grms_welch=float(math.sqrt(P[inb].sum() * df)), grms_time=float(np.std(x_g)),
                grms_spec=grms(breakpoints), skewness=float(skew(x_g)), kurtosis=float(kurtosis(x_g, fisher=False)),
                averages=n_avg)

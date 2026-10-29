"""Flight timeline from lift-off to splashdown, as loads in the tubule frame.

The tubule (chamber) frame has +z along the rocket's long axis pointing to the nose; the aft end of a
standing tubule is its z = 0 face. The payload's specific force f(t) (what an accelerometer reads: thrust,
drag, parachute, water) appears to a free grain as -f, so thrust presses dust onto the aft face and drag
on the way up pulls it toward the nose end. The payload spins about +z at Omega(t), which adds centrifugal,
Coriolis and Euler (-dOmega/dt x r) accelerations.

Sources and their status (see provenance.py):
* RockSat User Guide 2026 (NASA Wallops Flight Facility, Rev-A, 4 Sept 2025, marked DRAFT), Table 3-1,
  Secs. 3.3 and 5.2.5: 5.2 s first-stage (Terrier) burn, 12.2 s coast, 11.7 s second-stage (Improved Malemute) burn;
  25 g sustained ascent load (design envelope) with ~50 g longitudinal impulses possible; spin ~4.8 Hz
  at second-stage burnout, de-spun to ~0-1 Hz shortly after burnout, ~0 Hz at apogee; apogee ~160 km;
  experiment power off T+332 s; chute deploy T+460 s; splash down T+882 s; designs must take 25 g
  quasi-static in all axes "during ascent and descent". Whether "chute deploy" is the main canopy or a
  drogue is not stated (the model treats it as the main).  [published]
* NASA Sounding Rocket Program Office mission fact sheets for the same vehicle family (AWESOME 46.034/
  46.035, SEED 46.026/46.037, 2025): de-spin ~T+54 s, apogee ~T+197-200 s at ~158-161 km. From these
  the 100 km crossings are ~T+88 s and ~T+313 s. RockOn 41.134 (2025, Wallops student flight on a
  Terrier-Improved Orion; trajectory plot, nominal or flown not stated): "Parachute deploy 474.7 sec at
  4.6 km"; NASA Sounding Rocket Program Handbook (810-HB-SRP, 2005, Sec. 5.16.1): parachutes "deployed in
  the 15,000 to 20,000 foot altitude region" (land recovery).  [published analogs, used for defaults]
* Ettl & Pfaender (2009, 19th ESA PAC Symposium): yo-yo de-spin in 0.9 s; drag decreases from 1e-4 g at
  100 km to ~2e-6 g at apogee.  [published]
* Air density: NRLMSIS 2.1 (atmosphere.py).  [published model]
* DERIVED (order of magnitude, swept): the drag deceleration in the coast between the burns, ~3-5 g, from
  the fact-sheet trajectory points (AWESOME: Terrier burnout 5.2 s at 1.7 km, Malemute ignition 14.0 s at
  5.3 km; SEED: 6.2 s at 2.4 km, 13.0 s at 6.8 km) assuming constant accelerations; after the second burn,
  drag rho v^2 / (2 beta_up) along a ballistic climb, beta_up set by 3 g at burnout (25.7 s at 17.2 km,
  ~1.65 km/s).
* DERIVED re-entry: a point-mass fall from apogee through NRLMSIS with drag rho v^2 / (2 beta); beta is
  solved so that the payload is at the chute altitude at T+460 s. It peaks at ~7.9 g near T+369 s at ~23 km.
  With gravity off and an exponential atmosphere the integrator reproduces the closed-form ballistic entry of
  Allen & Eggers (1958, NACA Report 1381) (verification check "reentry"). Comparable payloads: a DLR
  simulation of CUMA-2 gives 13.5 g at 25.8 km (Stamminger 2011, IAC-11-A2.5.9); the REXUS User Manual v7.16
  says "The typical deceleration during re-entry is below 20 g". Constant-coefficient point-mass models read
  5-20% low against measured TEXUS/MAXUS peaks (up to ~45% on spikes), so the real peak may be ~8-11 g.
* DERIVED parachute and splashdown: the opening decelerates the payload from its fall speed (~94 m/s) to the
  descent speed in a half-sine whose peak is 5 g, the "up to 5 g in any direction" of NASA-MSFC M-EH-75-2
  (1975, the typical sequence planned for Black Brant VC land recovery, pp. 35-36); under the canopy drag = weight, v(h) = v_sea sqrt(rho_0/rho(h)), with v_sea
  set by reaching the sea at T+882 s (~9.4 m/s; the same handbook gives landing at 30-40 ft/s). The water
  impact removes v_sea over splash_stop_s = 0.05 s (ASSUMED: no measured value was found; the impulse
  is fixed by momentum, the peak, ~31 g, scales with 1/splash_stop_s). The shocks act along the hanging axis by
  default (ASSUMED; the flight log below saturated laterally at splashdown): shock_direction tilts them.
* ATTITUDE (sourced where possible): above ~100 km nothing turns the payload, so it keeps its ascent attitude
  until it re-enters the atmosphere (attitude_change_s, default ug_end_s). In the fall, broadside: DLR flight
  data show payloads turning side-on to the flow before peak deceleration (Stamminger 2007, ESA PAC Visby);
  NASA-MSFC (1975): payloads tumble, loads up to +/-25 g along any axis at up to 2.5 Hz ("tumbling" option,
  which rotates only the load direction; the tumble's own centrifugal, Coriolis and Euler loads are not
  modelled). Under the canopy, nose down: the
  RockSat-C 2016 WVSU flight log reads -1 g along the rocket axis under the parachute (+1 g on the pad),
  and afloat the payload lies on its side (WV SPACE RockSat-C 2016 Final Report, Fig. 42). That was a
  different vehicle (Terrier-Improved Orion, spinning all flight, 119 km apogee), used here only for the
  orientation; RockSat-X data were not found. The scenarios descent_tumbling and hang_nose_up test the
  alternatives.
* ASSUMED (no public data; swept in the mission runs): the shape of the thrust inside each burn (flat
  at the envelope by default), the spin-up shape (linear to 4.8 Hz), the residual spin after de-spin, and
  the quasi-steady residual acceleration (1e-4 g, a floor on the drag magnitude with the drag's direction).

Provenance: timeline values as listed above; the rotating-frame kinematics are standard. No REBOUND code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import atmosphere

G0 = 9.80665
_DESCENT_CACHE: dict = {}       # solved descent trajectories, keyed by the parameters that determine them

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
    spin_burnout_hz: float = 4.8           # guide Table 3-1
    despin_s: float = 54.0                 # analog fact sheets (AWESOME, SEED): de-spin at T+54 s
    despin_duration_s: float = 0.9         # Ettl & Pfaender 2009 (yo-yo)
    residual_spin_hz: float = 1.0 / 60.0   # after de-spin; see RESIDUAL_SPIN_OPTIONS_HZ (sweep)
    residual_g: float = 1e-4               # quasi-steady residual acceleration in the coast (order 1e-4 g)
    ug_start_s: float = 88.0               # ~100 km crossing (derived from the analog apogee time/height)
    ug_end_s: float = 313.0                # ~100 km crossing on the way down (same analogs)
    power_off_s: float = 332.0             # guide Table 3-1 (an event; the simulation continues to splashdown)
    lateral_g: float = 10.0                # guide Sec. 5.2.5: "about +/- 10 Gs" lateral during ascent (report only)
    # ---- descent and landing (guide Table 3-1 times; see the module docstring for the other values)
    chute_s: float = 460.0                 # guide Table 3-1: chute deploy
    splash_s: float = 882.0                # guide Table 3-1: splash down
    float_s: float = 5.0                   # simulated time after splashdown
    chute_altitude_m: float = 4600.0       # NASA fact sheet RockOn 41.134 (2025): "Parachute deploy ... at 4.6 km"
    opening_peak_g: float = 5.0            # NASA-MSFC M-EH-75-2 (1975): each opening "up to 5 g" (sets the opening time)
    splash_stop_s: float = 0.05            # ASSUMED stopping time of the water impact (no measurement found)
    descent_beta_kg_m2: float | None = None   # ballistic coefficient m/(Cd A) in the fall; None = solve (see below)
    entry_horizontal_speed_m_s: float = 0.0   # horizontal speed at apogee (0 = vertical fall, the steepest entry)
    attitude_change_s: float | None = None  # when the fall attitude takes over (None = ug_end_s, re-entering the air)
    descent_attitude: str = "broadside"    # in the fall: broadside | tumbling | tail_first | nose_first (see docstring)
    tumble_hz: float = 1.0                 # ASSUMED tumbling rate (NASA-MSFC 1975: loads vary "at frequencies up to 2.5 Hz")
    hang_attitude: str = "nose_down"       # under the parachute: nose_down (WVSU RockSat-C 2016 flight data) | nose_up
    float_attitude: str = "side"           # afloat: side (WVSU RockSat-C 2016) | nose_up | nose_down
    shock_direction: str = "hang_axis"     # opening and splash shocks: hang_axis | lateral | oblique (ASSUMED)
    impulses: list = field(default_factory=list)  # [{"t_s", "peak_g", "duration_s"}]: axial shocks (+ = forward)

    def __post_init__(self):
        for name, ok in (("descent_attitude", ("broadside", "tumbling", "tail_first", "nose_first")),
                         ("hang_attitude", ("nose_up", "nose_down")),
                         ("float_attitude", ("side", "nose_up", "nose_down")),
                         ("shock_direction", ("hang_axis", "lateral", "oblique"))):
            if getattr(self, name) not in ok:
                raise ValueError(f"{name} must be one of {ok}")
        # ascent drag: rho(h) v^2 / (2 beta_up), with beta_up set by the drag at second-stage burnout
        rho_bo = atmosphere.density(self.burnout_altitude_m)
        self.ascent_beta_kg_m2 = (rho_bo * self.burnout_speed_m_s ** 2 / (2 * self.post_burn_drag_g * G0)
                                  if self.post_burn_drag_g > 0 else math.inf)
        self._descent_table()

    @classmethod
    def from_config(cls, c: dict | None) -> "FlightProfile":
        c = {k: v for k, v in dict(c or {}).items() if not k.startswith("_")}
        if isinstance(c.get("residual_spin_hz"), str):
            c["residual_spin_hz"] = RESIDUAL_SPIN_OPTIONS_HZ[c["residual_spin_hz"]]
        for k in ("burn1", "burn2"):
            if k in c:
                c[k] = tuple(c[k])
        return cls(**c)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(ascent_beta_kg_m2=(self.ascent_beta_kg_m2 if math.isfinite(self.ascent_beta_kg_m2) else None),
                 attitude_change_used_s=self.t_attitude, descent_beta_used_kg_m2=self.beta_down,
                 apogee_s=self.apogee_s, apogee_altitude_m=self.apogee_altitude_m,
                 chute_speed_m_s=self.chute_speed_m_s, chute_open_s=self.chute_open_s,
                 descent_speed_sea_level_m_s=self.v_sea, reentry_peak_g=self.reentry_peak_g,
                 reentry_peak_s=self.reentry_peak_s, end_s=self.end_s)
        return d

    # ------------------------------------------------------------------ trajectory
    @property
    def apogee_s(self) -> float:
        return self.burn2[1] + self.burnout_speed_m_s / G0

    @property
    def apogee_altitude_m(self) -> float:
        return self.burnout_altitude_m + self.burnout_speed_m_s ** 2 / (2 * G0)

    @property
    def end_s(self) -> float:
        return self.splash_s + self.float_s

    @property
    def t_attitude(self) -> float:
        """When the fall attitude (descent_attitude) takes over from the ascent attitude: by default when the payload
        re-enters the atmosphere (ug_end_s). Above ~100 km nothing turns it (no aerodynamic torque)."""
        return float(self.ug_end_s if self.attitude_change_s is None else self.attitude_change_s)

    def ballistic_speed(self, t: float) -> float:
        """Vertical speed [m/s] of the drag-free ascent from burnout (+ = up); used until apogee."""
        return self.burnout_speed_m_s - G0 * (t - self.burn2[1])

    def _fall(self, beta, t_end, dt=0.5, y0=None, t0=None, gravity=True, density=None):
        """Point-mass fall with drag rho v^2/(2 beta) and gravity g0 (R/(R+h))^2, RK4; by default from apogee at
        rest vertically, through the NRLMSIS density. y0 = (x, h, vx, vh). Returns arrays t, altitude, speed and
        the drag specific force [m/s^2]."""
        R_E = 6.371e6
        rho = density or atmosphere.density

        def rhs(y):
            x, h, vx, vh = y
            v = math.hypot(vx, vh)
            k = rho(h) * v / (2 * beta)
            g = G0 * (R_E / (R_E + h)) ** 2 if gravity else 0.0
            return np.array([vx, vh, -k * vx, -k * vh - g])
        t0 = self.apogee_s if t0 is None else t0
        y = (np.array([0.0, self.apogee_altitude_m, self.entry_horizontal_speed_m_s, 0.0]) if y0 is None
             else np.array(y0, float))
        n = max(int(math.ceil((t_end - t0) / dt)), 1)
        dt = (t_end - t0) / n                          # the last step ends exactly at t_end
        ts, hs, vs = [t0], [y[1]], [math.hypot(y[2], y[3])]
        fs = [rho(max(y[1], 0.0)) * vs[0] ** 2 / (2 * beta)]
        for k in range(n):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            v = math.hypot(y[2], y[3])
            ts.append(t0 + (k + 1) * dt)
            hs.append(y[1])
            vs.append(v)
            fs.append(rho(max(y[1], 0.0)) * v * v / (2 * beta))
            if y[1] <= 0:
                break
        return np.array(ts), np.array(hs), np.array(vs), np.array(fs)

    def _descent_table(self):
        """The fall from apogee to parachute deployment, then the parachute descent.

        Fall: if descent_beta_kg_m2 is None, the ballistic coefficient is solved so that the payload is at
        chute_altitude_m at chute_s (guide time, fact-sheet altitude). Opening: the speed drops from the fall speed
        to the descent speed in a half-sine deceleration whose peak is opening_peak_g, which sets the opening time
        (impulse = momentum change). Descent: speed v(h) = v_sea sqrt(rho_0 / rho(h)) (drag = weight), with v_sea
        set so that the payload reaches the sea at splash_s (guide)."""
        key = (self.apogee_s, self.apogee_altitude_m, self.chute_s, self.splash_s, self.chute_altitude_m,
               self.opening_peak_g, self.descent_beta_kg_m2, self.entry_horizontal_speed_m_s)
        if key in _DESCENT_CACHE:
            (self.beta_down, self._t, self._h, self._v, self._f, self.chute_speed_m_s, self.chute_open_s, self.v_sea,
             self.reentry_peak_g, self.reentry_peak_s) = _DESCENT_CACHE[key]
            return
        target = self.chute_altitude_m
        if self.descent_beta_kg_m2 is not None:
            beta = float(self.descent_beta_kg_m2)
        else:
            lo, hi = 1.0, 1e5               # h(chute_s) rises monotonically as beta falls (more drag, slower fall)
            for _ in range(30):             # beta to 1e-4 relative
                beta = math.sqrt(lo * hi)
                t, h, v, f = self._fall(beta, self.chute_s)
                if t[-1] < self.chute_s - 1e-9 or h[-1] < target:
                    hi = beta
                else:
                    lo = beta
            beta = math.sqrt(lo * hi)
        self.beta_down = beta
        self._t, self._h, self._v, self._f = self._fall(beta, self.chute_s)
        if self._t[-1] < self.chute_s - 1e-9:
            raise ValueError("the payload reaches the ground before the parachute opens: check descent_beta_kg_m2")
        self.chute_speed_m_s = float(self._v[-1])
        h_c = float(self._h[-1])
        rho0 = atmosphere.density(0.0)
        hs = np.linspace(0.0, h_c, 400)
        sq = np.sqrt([atmosphere.density(x) / rho0 for x in hs])
        v_sea = 10.0
        for _ in range(20):                 # opening time and descent speed depend on each other weakly
            v_d = v_sea * math.sqrt(rho0 / atmosphere.density(h_c))
            dv = max(self.chute_speed_m_s - v_d, 0.0)
            tau = 0.5 * math.pi * dv / max((self.opening_peak_g - 1.0) * G0, 1e-9) if dv > 0 else 0.0
            h1 = h_c - 0.5 * (self.chute_speed_m_s + v_d) * tau          # altitude lost while the canopy opens
            m = hs <= h1
            integral = float(np.sum(0.5 * (sq[m][1:] + sq[m][:-1]) * np.diff(hs[m]))) if m.sum() > 1 else 0.0
            v_sea = integral / max(self.splash_s - self.chute_s - tau, 1e-9)
        self.v_sea, self.chute_open_s = v_sea, tau
        k = int(np.argmax(self._f))
        self.reentry_peak_g = float(self._f[k] / G0)
        self.reentry_peak_s = float(self._t[k])
        _DESCENT_CACHE[key] = (self.beta_down, self._t, self._h, self._v, self._f, self.chute_speed_m_s,
                               self.chute_open_s, self.v_sea, self.reentry_peak_g, self.reentry_peak_s)

    # ------------------------------------------------------------------ specific force on the payload
    def axial_g(self, t: float) -> float:
        """Specific force on the payload [g] (what an accelerometer reads) along load_direction(t): thrust; drag
        (at least residual_g) opposing the motion; under the parachute 1 g plus the opening shock; the water
        impact; afloat 1 g. Excludes the user-supplied impulses."""
        if self.burn1[0] <= t < self.burn1[1] or self.burn2[0] <= t < self.burn2[1]:
            return self.burn_g
        if self.burn1[1] <= t < self.burn2[0]:
            return -self.coast_drag_g
        if self.burn2[1] <= t < self.apogee_s:
            v = self.ballistic_speed(t)
            tau = t - self.burn2[1]
            h = self.burnout_altitude_m + self.burnout_speed_m_s * tau - 0.5 * G0 * tau * tau
            drag = atmosphere.density(h) * v * v / (2 * self.ascent_beta_kg_m2) / G0
            return -max(drag, self.residual_g)
        if self.apogee_s <= t < self.chute_s:
            return max(float(np.interp(t, self._t, self._f)) / G0, self.residual_g)
        if self.chute_s <= t < self.chute_s + self.chute_open_s:      # opening: peak opening_peak_g
            return 1.0 + (self.opening_peak_g - 1.0) * math.sin(math.pi * (t - self.chute_s) / self.chute_open_s)
        if self.splash_s <= t < self.splash_s + self.splash_stop_s:   # water impact: v_sea -> 0 (momentum)
            return 1.0 + self.v_sea / self.splash_stop_s * 0.5 * math.pi * math.sin(
                math.pi * (t - self.splash_s) / self.splash_stop_s) / G0
        if t >= self.chute_s:
            return 1.0
        return 0.0

    def load_direction(self, t: float) -> np.ndarray:
        """Direction (tubule frame) along which axial_g acts. Until the payload re-enters the atmosphere (t_attitude,
        default T+313 s) it keeps its ascent attitude: +z (thrust and drag along the rocket axis; a negative axial_g
        points to the tail, so the residual reverses at apogee). Then: broadside (side-on to the flow, as measured on
        DLR payloads) +x; tumbling, rotating in the x-z plane at tumble_hz (the direction only: the tumble's own
        centrifugal, Coriolis and Euler loads are not modelled); tail first +z; nose first -z. Under the parachute the
        canopy pulls the upper end: nose down -z, nose up +z. Afloat: on its side +x, nose up +z, nose down -z. The
        opening and splash shocks act along the hanging axis, sideways (+x) or at 45 degrees (shock_direction)."""
        if t < self.t_attitude:
            return np.array([0.0, 0.0, 1.0])
        if t < self.chute_s:
            if self.descent_attitude == "tumbling":
                ph = 2 * math.pi * self.tumble_hz * (t - self.t_attitude)
                return np.array([math.sin(ph), 0.0, math.cos(ph)])
            return {"broadside": np.array([1.0, 0.0, 0.0]), "tail_first": np.array([0.0, 0.0, 1.0]),
                    "nose_first": np.array([0.0, 0.0, -1.0])}[self.descent_attitude]
        hang = np.array([0.0, 0.0, -1.0 if self.hang_attitude == "nose_down" else 1.0])
        shock = (self.chute_s <= t < self.chute_s + self.chute_open_s
                 or self.splash_s <= t < self.splash_s + self.splash_stop_s)
        if shock and self.shock_direction != "hang_axis":
            lat = np.array([1.0, 0.0, 0.0])
            return lat if self.shock_direction == "lateral" else (hang + lat) / math.sqrt(2.0)
        if t < self.splash_s + self.splash_stop_s:
            return hang
        return {"side": np.array([1.0, 0.0, 0.0]), "nose_up": np.array([0.0, 0.0, 1.0]),
                "nose_down": np.array([0.0, 0.0, -1.0])}[self.float_attitude]

    def impulse_g(self, t: float) -> float:
        a = 0.0
        for p in self.impulses:
            t0, tau = float(p["t_s"]), float(p["duration_s"])
            if t0 <= t < t0 + tau:
                a += float(p["peak_g"]) * math.sin(math.pi * (t - t0) / tau)   # half-sine shock pulse
        return a

    def knots_in(self, t0: float, t1: float) -> list:
        """Times inside (t0, t1) where the specific force can peak between samples: the descent table's points and
        the peaks of the half-sine shocks (parachute opening, splash, user impulses)."""
        out = []
        if t1 > self.apogee_s and t0 < self.chute_s:
            tt = np.asarray(self._t)
            out += tt[(tt > t0) & (tt < t1)].tolist()
        for a, b in self.event_windows():
            pk = 0.5 * (a + b)
            if t0 < pk < t1:
                out.append(pk)
        return out

    def event_windows(self) -> list:
        """Short load events the time step must resolve: the user impulses, the parachute opening, the splash."""
        w = [(float(i["t_s"]), float(i["t_s"]) + float(i["duration_s"])) for i in self.impulses]
        return w + [(self.chute_s, self.chute_s + self.chute_open_s),
                    (self.splash_s, self.splash_s + self.splash_stop_s)]

    def apparent_accel(self, t: float) -> np.ndarray:
        """Uniform apparent acceleration on a free grain in the tubule frame [m/s^2]: minus the specific force."""
        return -(self.axial_g(t) * self.load_direction(t) + self.impulse_g(t) * np.array([0.0, 0.0, 1.0])) * G0

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
        """Flight phases with the range of the specific force along the rocket axis (axial_g_range, + = toward the
        nose, in g) and its largest sideways part (lateral_g_max)."""
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
            dict(name="microgravity (above ~100 km)", t0=self.ug_start_s, t1=self.ug_end_s,
                 source="100 km crossings derived from analog fact sheets; residual assumed"),
            dict(name="re-entry", t0=self.ug_end_s, t1=self.chute_s,
                 source="ballistic fall through NRLMSIS 2.1 with a derived ballistic coefficient; chute time: guide"),
            dict(name="parachute opening", t0=self.chute_s, t1=self.chute_s + self.chute_open_s,
                 source="momentum change (fall speed to descent rate) over an assumed opening time"),
            dict(name="descent under the parachute", t0=self.chute_s + self.chute_open_s, t1=self.splash_s,
                 source="1 g (steady descent); splashdown time: guide"),
            dict(name="splashdown", t0=self.splash_s, t1=self.splash_s + self.splash_stop_s,
                 source="momentum change (descent rate to 0) over an assumed stopping time"),
            dict(name="afloat", t0=self.splash_s + self.splash_stop_s, t1=self.end_s, source="1 g"),
        ]
        for ph in p:
            ts = ph["t0"] + (np.arange(400) + 0.5) / 400 * (ph["t1"] - ph["t0"])
            f = np.array([-self.apparent_accel(t) / G0 for t in ts])     # specific force on the payload [g]
            ph["axial_g_range"] = [float(f[:, 2].min()), float(f[:, 2].max())]   # along +z (toward the nose)
            ph["lateral_g_max"] = float(np.hypot(f[:, 0], f[:, 1]).max())
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

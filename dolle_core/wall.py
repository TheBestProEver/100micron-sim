"""A grain resting on a wall: what it takes to pull it off, slide it, roll it or twist it.

A grain stuck on a wall sits at the JKR equilibrium (zero load: patch radius a0, jkr.py).  A load on the grain
(the rocket's acceleration, spin) can break the contact in four ways.  In each, a "spring" in the contact resists
until its force or torque reaches a limit.  Pull-off happens at F_c plus any pressing load; the sliding, rolling and
twisting limits are of the "Coulomb with adhesion" form, proportional to |F_ne + 2 F_c|, where F_ne [N] is the
elastic normal force (> 0 when the grain is pressed on the wall; zero load: F_ne = 0, the bracket is 2 F_c).  This form is Thornton & Yin (1991, Powder Technol. 65, 153) and Marshall (2009,
J. Comput. Phys. 228, 1541, eq. 43), as in LAMMPS; the simulator's code is dolle_sim/dem.py make_params and
contact_force.  Symbols: m grain mass [kg], R its radius [m] (R* = R on a wall), g0 = 9.80665 m/s^2.

  Pull-off (normal).  The contact holds a steady pull up to F_c = 1.5 pi w R (jkr.py).  A body acceleration a
  away from the wall pulls with m a, so it pulls off at a = F_c/m (plus any pressing load).

  Sliding.  Tangential spring k_s = 8 G* a0 (Wada et al. 2007, ApJ 661, 320, eqs. 12-13), with the sliding
  modulus 1/G* = (2 - nu1)/G1 + (2 - nu2)/G2, G = E/(2(1 + nu)).  It slips when its force reaches
  mu |F_ne + 2 F_c| (mu: materials.py FRICTION), so a sideways acceleration of 2 mu F_c / m (zero load) slides it.

  Rolling.  A sideways load also tries to roll the grain.  Rolling moves the patch: contact is made at the front
  edge and broken at the back, and because breaking a sticky contact costs more than making it returns, the
  pressure on the patch becomes lopsided and gives a torque against the roll (Dominik & Tielens 1995, Phil.
  Mag. A 72, 783).  Rolling spring k_r = 4 F_c / R (Wada et al. 2007 eq. 22); it holds until the rolling
  displacement reaches
  xi_crit = h a0 / 12 (Krijt, Dominik & Tielens 2014, J. Phys. D 47, 175302, eq. 16), h the adhesion hysteresis,
  0.5 (ASSUMED: fitted to ~1 um silica; the only measurement at these sizes, 70 um glass beads, gives 0.05-0.6).
  Limit torque M_r = (2 xi_crit / R) |F_ne + 2 F_c| R (LAMMPS "rolling sds" form, dem.py make_params), which is
  k_r xi_crit R = 4 F_c xi_crit at zero load; torque balance about the contact point (the load m a acts at the
  centre, height R) gives the rolling acceleration  a_roll = M_r / (m R) = 4 F_c xi_crit / (m R) at zero load.  This is by far
  the weakest limit: ~1.3 g for a 50 um radius silica grain on clean borosilicate, against ~1000 g to pull it off.
  Beyond it the grain rolls without slipping: (m + I/R^2) dv/dt = m (a - a_roll), I = (2/5) m R^2.

  Twisting.  Twisting spring k_tw = (16/3) G_tw a0^3 with the twisting modulus G_tw = 1/(1/G1 + 1/G2) (Wada et al.
  2007 eq. 26; not the sliding modulus G*); it holds up to the critical twist angle phi_crit = 1/(16 pi) (Wada et
  al. 2007), so the limit torque is k_tw phi_crit = G_tw a0^3 / (3 pi) at zero load (Wada et al. 2007 eq. 37),
  scaled by |F_ne + 2 F_c| / (2 F_c) under load as in dem.py.  A uniform acceleration exerts no torque about the wall
  normal, so twisting needs an angular acceleration of the wall: alpha = M_tw / I [rad/s^2].

The check: the simulator's verify_full.check_rolling_on_wall finds the rolling limit 1.29 g for a 50 um radius
silica grain on borosilicate and then rolling without slip (2.3e-8) at the speed of the formula above (3.2e-3).
(PROVENANCE.md's "1.3 g for 100 um" and "6.5 g for 50 um" refer to diameters.)
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .jkr import equilibrium_contact_radius, pull_off_force
from .materials import G0, get_material, friction_coefficient
from .pair import Pair, make_pair


def shear_modulus(E: float, nu: float) -> float:
    """G = E / (2 (1 + nu)) [Pa] (isotropic elasticity)."""
    return E / (2.0 * (1.0 + nu))


def sliding_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
    """G* = 1 / ((2 - nu1)/G1 + (2 - nu2)/G2) [Pa] (Wada et al. 2007 eq. 13; dem.py make_params Gst)."""
    return 1.0 / ((2.0 - nu1) / shear_modulus(E1, nu1) + (2.0 - nu2) / shear_modulus(E2, nu2))


def twisting_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
    """G = 1 / (1/G1 + 1/G2) [Pa] (Wada et al. 2007 eq. 26; dem.py make_params Gtw)."""
    return 1.0 / (1.0 / shear_modulus(E1, nu1) + 1.0 / shear_modulus(E2, nu2))


def critical_rolling_displacement(a0: float, h: float = 0.5) -> float:
    """xi_crit = h a0 / 12 [m] (Krijt et al. 2014 eq. 16); h = 0.5 ASSUMED (dem.py make_params)."""
    return h * a0 / 12.0


@dataclass(frozen=True)
class WallLimits:
    """Limits of a grain resting on a wall (SI; accelerations in g).  normal_bracket = |F_ne + 2 F_c| [N];
    k_slide [N/m], k_roll [N/m], k_twist [N m/rad] the spring stiffnesses; roll_torque, twist_torque [N m] the limit
    torques; *_g the body accelerations that reach each limit; twist_angular_acc [rad/s^2]."""
    pair: Pair
    mu: float
    h: float
    press_g: float
    a0: float
    F_c: float
    normal_bracket: float
    k_slide: float
    k_roll: float
    k_twist: float
    xi_crit: float
    roll_torque: float
    twist_torque: float
    pull_off_g: float
    slide_g: float
    roll_g: float
    twist_angular_acc: float


def wall_limits(grain, r: float, wall, f_adh: float = 1.0, mu: float | None = None, h: float = 0.5,
                press_g: float = 0.0) -> WallLimits:
    """Limits of a grain (material `grain`, radius r [m]) resting on a flat wall of material `wall`.  f_adh scales
    w (ASSUMED coating/roughness knob); mu defaults to FRICTION[(grain, "wall")]; h rolling hysteresis (ASSUMED
    0.5); press_g a load pressing the grain on the wall, in g (raises F_ne to m g0 press_g).  Formulas of the
    module docstring, as dolle_sim/dem.py make_params."""
    g, wl = get_material(grain), get_material(wall)
    p = make_pair(g, r, wl, None, f_adh=f_adh)
    mu = friction_coefficient(g.name, "wall") if mu is None else mu
    a0, F_c, m = equilibrium_contact_radius(p.R_star, p.w, p.E_star), pull_off_force(p.R_star, p.w), p.m1
    N = abs(m * G0 * press_g + 2 * F_c)
    xi = critical_rolling_displacement(a0, h)
    G_tw = twisting_modulus(g.E, g.nu, wl.E, wl.nu)
    M_tw = G_tw * a0 ** 3 / (3 * math.pi) / (2 * F_c) * N
    return WallLimits(pair=p, mu=mu, h=h, press_g=press_g, a0=a0, F_c=F_c, normal_bracket=N,
                      k_slide=8.0 * sliding_modulus(g.E, g.nu, wl.E, wl.nu) * a0, k_roll=4.0 * F_c / p.R_star,
                      k_twist=16.0 / 3.0 * G_tw * a0 ** 3, xi_crit=xi, roll_torque=2.0 * xi / p.R_star * N * r,
                      twist_torque=M_tw, pull_off_g=F_c / (m * G0) + press_g, slide_g=mu * N / (m * G0),
                      roll_g=2.0 * xi / p.R_star * N / (m * G0), twist_angular_acc=M_tw / (0.4 * m * r * r))


def rolling_acceleration(a_body: float, a_roll: float, v: float = 0.0, dt: float | None = None) -> float:
    """Acceleration of the grain's centre [m/s^2] along a sideways body acceleration a_body (>= 0) with rolling
    limit a_roll (both m/s^2), rolling without slip: (m + I/R^2) dv/dt = m (a_body - a_roll), I = (2/5) m R^2, so
    dv/dt = (a_body - a_roll)/1.4 (verify_full.check_rolling_on_wall).  From rest (v = 0) it is zero below the
    limit.  A grain already rolling (v > 0 [m/s], along a_body) keeps the limit torque against it and DECELERATES
    at (a_body - a_roll)/1.4 when the load falls below the limit; with dt [s] given, the deceleration is clamped
    so that v does not change sign within the step (the grain stops)."""
    if v <= 0.0:
        return max(a_body - a_roll, 0.0) / 1.4
    acc = (a_body - a_roll) / 1.4
    if dt is not None and acc < 0.0:
        acc = max(acc, -v / dt)
    return acc

"""dolle_core: the physics at the heart of the DOLL-E simulator, small enough to read.

Plain python + numpy (no numba, no classes beyond small dataclasses).  Every function states its equation, units
and source, and reproduces the full simulator (package dolle_sim) for what it covers: restitutions and sticking
speeds to rounding error (tests/test_dolle_core.py).

Read it in this order:

  materials.py   Material, MATERIALS, FRICTION, get_material, sphere_mass        the numbers, with sources
  pair.py        make_pair, reduced_modulus, reduced_radius, reduced_mass,        two bodies -> one effective
                 work_of_adhesion                                                 contact
  jkr.py         equilibrium_contact_radius, pull_off_force, overlap_of_radius,   sticky elastic contact (JKR)
                 force_of_radius, contact_radius, break_overlap, snap_on_radius,
                 hysteresis_energy, tabor_parameter
  damping.py     damping_beta, damping_coefficient                                energy lost to heat (e_dry)
  plastic.py     yield_overlap, yield_speed, flattened_radius, ...                denting (Thornton & Ning)
  collide.py     collide, collide_pair, impact, sticking_threshold                a head-on collision, step by
                                                                                  step; stick or bounce
  scaling.py     undamped_sticking_speed, scaling_law_speed, damping_factor       the threshold by hand
  wall.py        wall_limits, rolling_acceleration                                a grain resting on a wall
  many.py        line_collision                                                   grains on a line: momentum

The 3-D simulation (notebook 06; the page docs/sim3d/):

  clumps.py      Clumps, make_clumps, merge, detach, rotate, drift, momentum      rigid clumps of stuck grains
  impacts.py     collision_matrix, normal_impact, bounce_impulse, grain_grain,    one impact in 3-D: stick or
                 grain_wall, fragment                                             bounce, friction, break-up
  sim3d.py       Tubule, Sim, make_sim, step, run, next_contact, summary          the tubule and the event loop
  scenes.py      two_tubule_start, run_two_tubules, impact_closeup,               the page's scenes and the
                 sandbox_start, sandbox_advance                                   browser sandbox

Quick start:
    >>> from dolle_core import collide, make_pair, sticking_threshold
    >>> r = collide("silica", 50e-6, "silica", 50e-6, u1=0.01)        # 1 cm/s into a resting grain
    >>> r.outcome, r.e
    >>> sticking_threshold(make_pair("graphite", 50e-6, "graphite", 50e-6))   # m/s
    >>> from dolle_core import scenes, sim3d
    >>> sims = scenes.run_two_tubules(scenes.two_tubule_start())             # silica and graphite, ~5 s
    >>> sim3d.summary(sims["graphite"])["clump_sizes_at_end"]
"""
from .materials import G0, FRICTION, MATERIALS, Material, friction_coefficient, get_material, sphere_mass
from .pair import Pair, make_pair, reduced_mass, reduced_modulus, reduced_radius, work_of_adhesion
from .jkr import (break_overlap, break_radius, contact_radius, contact_work, equilibrium_contact_radius,
                  force_of_radius, hysteresis_energy, overlap_of_radius, pull_off_force, snap_on_radius,
                  tabor_parameter)
from .damping import damping_beta, damping_coefficient
from .plastic import (flattened_radius, hertz_force, permanent_indentation, plastic_loading_force,
                      thornton_ning_restitution, yield_overlap, yield_speed)
from .collide import (CollisionResult, TRACE_KEYS, centre_of_mass_velocities, collide, collide_pair, contact_law,
                      impact, sticking_threshold, time_step)
from .scaling import (chokshi_dominik_tielens_speed, damping_factor, equal_spheres_prefactor, hysteresis_constant,
                      scaling_law_speed, undamped_sticking_speed, wall_prefactor)
from .wall import WallLimits, rolling_acceleration, wall_limits
from .many import LineResult, line_collision
from .sim3d import Sim, Tubule, make_sim
from .scenes import impact_closeup, run_two_tubules, two_tubule_start

__all__ = [n for n in dir() if not n.startswith("_")]

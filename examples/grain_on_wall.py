"""A grain resting on a wall: the acceleration needed to pull it off, slide, roll or twist it.

Run:  python examples/grain_on_wall.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dolle_core as dc

R = 50e-6   # grain RADIUS [m]
for grain in ("silica", "graphite"):
    for wall, f_adh in (("glass_borosilicate", 1.0), ("glass_borosilicate", 0.05), ("glass_silanized", 1.0)):
        L = dc.wall_limits(grain, R, wall, f_adh=f_adh)
        print(f"{grain} (radius 50 um) on {wall}, adhesion factor {f_adh}: F_c = {L.F_c * 1e6:.2f} uN, "
              f"a0 = {L.a0 * 1e9:.1f} nm")
        print(f"   pull-off {L.pull_off_g:9.2f} g | sliding (mu {L.mu}) {L.slide_g:9.2f} g | rolling {L.roll_g:8.4f} g"
              f" | twisting: torque {L.twist_torque:.3e} N m = angular acceleration {L.twist_angular_acc:.2e} rad/s^2")
L = dc.wall_limits("silica", R, "glass_borosilicate", press_g=1.0)
print(f"\npressed on the wall by 1 g: silica rolls at {L.roll_g:.4f} g (instead of "
      f"{dc.wall_limits('silica', R, 'glass_borosilicate').roll_g:.4f} g)")
a_roll = dc.wall_limits("silica", R, "glass_borosilicate").roll_g * dc.G0
print(f"at 3x the rolling limit the grain's centre accelerates at {dc.rolling_acceleration(3 * a_roll, a_roll):.2f} "
      f"m/s^2 = (3 - 1) a_roll / 1.4  (rolling without slip, I = 2/5 m R^2)")

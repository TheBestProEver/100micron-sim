"""The sticking speed: from the integrated collisions and by hand.

Run:  python examples/sticking_threshold.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dolle_core as dc

R = 50e-6
cases = [("silica", R, "silica", R), ("graphite", R, "graphite", R),
         ("silica", R, "glass_borosilicate", None), ("graphite", R, "glass_borosilicate", None)]
print("sticking speed of 50 um radius grains [mm/s]   (integrated, with damping | without damping | f = ratio)")
for a, ra, b, rb in cases:
    p = dc.make_pair(a, ra, b, rb)
    v = dc.sticking_threshold(p)
    v0 = dc.undamped_sticking_speed(p)
    print(f"   {p.name:32s} {v * 1e3:9.4f} | {v0 * 1e3:8.4f} | {v / v0:.3f}   (e_dry of the pair {p.e_dry:.3f})")

print(f"\nby hand, equal spheres: v0 = {dc.equal_spheres_prefactor():.4f} gamma^(5/6) E*^(-1/3) R*^(-5/6) rho^(-1/2); "
      f"on a wall: {dc.wall_prefactor():.4f} w^(5/6) E*^(-1/3) R^(-5/6) rho^(-1/2)")
for name in ("silica", "graphite"):
    m, p = dc.get_material(name), dc.make_pair(name, R, name, R)
    print(f"   {name}: formula {dc.scaling_law_speed(m.gamma, p.E_star, p.R_star, m.rho) * 1e3:.4f} mm/s, "
          f"sqrt(2|U0|/m*) {dc.undamped_sticking_speed(p) * 1e3:.4f} mm/s, published Chokshi et al. form "
          f"{dc.chokshi_dominik_tielens_speed(m.gamma, p.E_star, p.R_star, m.rho) * 1e3:.4f} mm/s (prefactor 1.07: Chokshi et al. assume a different energy loss)")

print("\ndamping factor f(e_dry) = damped / undamped sticking speed (the same for every material and size):")
for e in (0.95, 0.85, 0.7, 0.5):
    print(f"   f({e}) = {dc.damping_factor(e, n_sub=1000, tol=1e-4):.3f}")

print("\nsize: silica pairs (radius -> threshold); expected ratio per halving 2^(5/6) = 1.782")
for r in (25e-6, 50e-6, 100e-6):
    print(f"   r = {r * 1e6:5.0f} um: {dc.sticking_threshold(dc.make_pair('silica', r, 'silica', r)) * 1e3:.3f} mm/s")

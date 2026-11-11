"""Two 50 um radius silica grains collide head-on; then four grains on a line.

Run:  python examples/collide_two_grains.py
Shows: stick or bounce, the restitution, where the energy goes, and that total momentum does not change.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import dolle_core as dc

R = 50e-6                                   # grain radius [m]
for u1 in (0.004, 0.02):                    # grain 1 at 4 mm/s and 20 mm/s into grain 2 at rest
    r = dc.collide("silica", R, "silica", R, u1=u1, u2=0.0)
    tr = r.trace
    E_in = 0.5 * r.pair.m_star * r.v_in ** 2
    dp = np.max(np.abs(tr["momentum"] - r.momentum_before)) / abs(r.momentum_before)
    print(f"silica-silica at {u1 * 1e3:.0f} mm/s: {r.outcome}, e = {r.e:.4f}, contact {r.contact_time * 1e6:.2f} us, "
          f"{len(tr['t'])} steps, max overlap {r.max_overlap * 1e9:.3f} nm, largest push {r.max_force * 1e6:.2f} uN")
    after = "set to the centre-of-mass velocity (bound pair)" if r.stuck else "measured at the break"
    print(f"   velocities after: v1 = {r.v1 * 1e3:.4f}, v2 = {r.v2 * 1e3:.4f} mm/s ({after}); "
          f"largest momentum change at any step, measured (relative): {dp:.1e}")
    i = len(tr["t"]) - 1
    print(f"   energy at the end, as a fraction of the incoming {E_in:.3e} J: kinetic (relative motion) "
          f"{tr['kinetic'][i] / E_in:.3f}, stored {tr['stored'][i] / E_in:.3f}, damping {tr['damping_loss'][i] / E_in:.3f}, "
          f"hysteresis {tr['hysteresis_loss'][i] / E_in:.3f}, integrator error {tr['energy_error'][i] / E_in:.1e}")
    if r.stuck:
        print(f"   bound: kinetic + stored = {(tr['kinetic'][i] + tr['stored'][i]) / E_in:.3f} of the incoming energy, "
              f"below the broken contact's 0, and damping only lowers it: the grains can never separate")

# four grains on a line, every contact force applied as an equal and opposite pair
mats, radii = ["silica", "graphite", "silica", "graphite"], [50e-6, 40e-6, 50e-6, 60e-6]
x0 = np.cumsum([0.0, 90.3e-6, 90.2e-6, 110.4e-6])
v0 = np.array([0.03, 0.0, -0.002, -0.01])
line = dc.line_collision(mats, radii, x0, v0, t_end=1e-4)
scale = np.sum(line.masses * np.abs(v0))
print(f"\nfour grains on a line, {len(line.momentum)} steps of {line.momentum_t[0] * 1e9:.2f} ns:")
print("   final velocities [mm/s]:", np.round(line.v[-1] * 1e3, 4), " still touching:", line.touching[-1])
print(f"   total momentum: start {np.sum(line.masses * v0):.6e}, end {line.momentum[-1]:.6e} kg m/s, "
      f"largest change {np.max(np.abs(line.momentum - np.sum(line.masses * v0))) / scale:.1e} (relative)")
print(f"   kinetic energy: start {line.kinetic[0]:.3e} J, end {line.kinetic[-1]:.3e} J (adhesion and damping took the rest)")

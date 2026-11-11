"""What changes from silica to carbon, and how the threshold would move for other carbons.

Run:  python examples/silica_vs_carbon.py

The carbon DOLL-E will fly is not decided.  The simulator's carbon is graphite (Chokshi, Tielens & Hollenbach
1993, Table 3, with e_dry and p_y ASSUMED).  The undamped sticking speed scales as
gamma^(5/6) E*^(-1/3) R*^(-5/6) rho^(-1/2) and damping multiplies it by f(e_dry), so for any candidate carbon only
gamma, E, nu, rho and e_dry matter (yield starts far above these speeds).
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dolle_core as dc

R = 50e-6
silica, graphite = dc.get_material("silica"), dc.get_material("graphite")
s, g = dc.make_pair(silica, R, silica, R), dc.make_pair(graphite, R, graphite, R)
v_s, v_g = dc.sticking_threshold(s, n_sub=1000, tol=1e-4), dc.sticking_threshold(g, n_sub=1000, tol=1e-4)
w_ratio = (graphite.gamma / silica.gamma) ** (5 / 6)
E_ratio = (g.E_star / s.E_star) ** (-1 / 3)
rho_ratio = (graphite.rho / silica.rho) ** -0.5
f_ratio = (v_g / dc.undamped_sticking_speed(g)) / (v_s / dc.undamped_sticking_speed(s))
print(f"silica {v_s * 1e3:.3f} mm/s -> graphite {v_g * 1e3:.3f} mm/s: x{v_g / v_s:.3f} =")
print(f"   gamma^(5/6) x{w_ratio:.3f}  *  E*^(-1/3) x{E_ratio:.3f}  *  rho^(-1/2) x{rho_ratio:.3f}"
      f"  (material values x{w_ratio * E_ratio * rho_ratio:.3f})  *  f(e_dry) x{f_ratio:.3f} (ASSUMED e_dry 0.7 vs 0.85)")

# Candidate carbons.  Values not marked are from the source named; everything ASSUMED is a placeholder to replace.
candidates = {
    "graphite (simulator)": graphite,
    "graphite, gamma 0.19 J/m^2 (cleavage energy, Wang et al. 2015)": replace(graphite, gamma=0.19),
    "graphite, e_dry 0.85 (= silica's; ASSUMED)": replace(graphite, e_dry=0.85),
    # glassy carbon: E 35 GPa and rho 1.42-1.54 g/cm^3 from the SIGRADUR glassy carbon data sheet; nu 0.17 as cited
    # for that material in the literature (Garion 2014, not checked here); gamma and e_dry ASSUMED = graphite's.
    "glassy carbon, rho 1420 (gamma, e_dry ASSUMED)": replace(graphite, name="glassy_carbon", E=35e9, nu=0.17,
                                                              rho=1420.0),
    "glassy carbon, rho 1540 (gamma, e_dry ASSUMED)": replace(graphite, name="glassy_carbon", E=35e9, nu=0.17,
                                                              rho=1540.0),
    "amorphous_carbon (simulator table, ALL ASSUMED)": dc.get_material("amorphous_carbon"),
}
print("\nsticking speed of two 50 um radius grains [mm/s] and ratio to silica:")
for label, m in candidates.items():
    v = dc.sticking_threshold(dc.make_pair(m, R, m, R), n_sub=1000, tol=1e-4)
    print(f"   {label:66s} {v * 1e3:8.3f}   x{v / v_s:.2f}")

print("\nsensitivity of the threshold (undamped part, exact exponents): +10% in a property changes it by")
for prop, expo in (("gamma", 5 / 6), ("E*", -1 / 3), ("rho", -1 / 2), ("radius", -5 / 6)):
    print(f"   {prop:7s} {100 * (1.1 ** expo - 1):+.1f}%")
for e in (0.6, 0.7, 0.8, 0.9):
    print(f"   e_dry = {e}: damping factor f = {dc.damping_factor(e, n_sub=1000, tol=1e-4):.3f}")

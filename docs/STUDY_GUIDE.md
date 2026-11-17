# DOLL-E simulator: a study guide in ten sessions

A path from "I have heard of it" to "I can explain and defend every part of it, and change any decision", in ten
sessions of one to two hours over two weeks, for someone who knows some physics and some coding. Each session has
a goal, what to read and run, exercises with the exact commands, a self-test with folded answers, and the questions
an official might ask with the points a good answer makes. The end of the guide has a 2-minute and a 10-minute
talk outline, a glossary and the numbers worth knowing by heart.

**How the numbers here were checked.** Every number in this guide is printed by a command or notebook cell named
next to it (all run on 2026-11-16), or is quoted from a generated file (`CHECKS.md`) or from
[DECISIONS.md](DECISIONS.md) with its entry ID. Anything assumed is marked **ASSUMED**, as in the rest of the
repository. Grain radius 50 um (diameter 100 um) unless stated.

**The documents you will use.**

| document | what it is for |
|---|---|
| [OVERVIEW.md](OVERVIEW.md) | the 15-minute overview: read it first and last |
| [notebooks/](../notebooks/README.md) 00-06 | the physics, derived step by step, every number printed by code |
| [examples/](../examples/) | the key calculations as plain scripts |
| `dolle_core/` | the readable physics (plain numpy), the same numbers as the full simulator |
| the 3-D page, https://thebestproever.github.io/100micron-sim/sim3d/ | the readable core at work in 3-D, in the browser |
| [DECISIONS.md](DECISIONS.md) | every decision: when, who, alternatives, evidence, effect, how to change it |
| [CHECKS.md](../CHECKS.md) | every check with its criterion and measured value |
| [MISSION_RUNS.md](MISSION_RUNS.md), [USER_GUIDE.md](USER_GUIDE.md), [HISTORY.md](HISTORY.md) | the long documents |

## Before session 1: set up once (15 minutes)

From the repository folder:

```bash
pip install -e .                          # numpy, scipy, matplotlib; makes dolle_core and dolle_sim importable
python examples/sticking_threshold.py     # should print the sticking speeds (session 1)
jupyter lab                               # then open notebooks/
```

The exercises below are short python snippets. Paste one into a new notebook cell, or save it as a file (for
example `try.py`) in the repository folder and run `python try.py`. If an exercise asks you to edit a file in the
repository, it also tells you how to undo the edit.

## The plan

| day | session | topic | decision log sections |
|---|---|---|---|
| 1 | 1 | the question, the experiment and the big picture | A, E1 |
| 2 | 2 | the sticky contact: JKR theory | B1, B2, B7 |
| 3 | 3 | one collision: energy loss, stick or bounce, the threshold | B3-B6, D1, D2 |
| 4 | 4 | momentum and energy: what the code guarantees | D3, D12, B13 |
| 5 | 5 | silica against carbon: the materials and their uncertainty | C1-C5, C8, G14, G15, O1, O2 |
| 6 | 6 | a grain on the wall | B8, B9, C6, C7, C9, C10, F20 |
| 7 | 7 | the flight, the forces and the mission runs | A3, A5, B18-B21, B23, F1-F21 |
| 8 | 8 | the 3-D simulation with rigid clumps | B12-B17, B22, D4, D5, E9, G12, G13, F13 |
| 9 | 9 | how we know it is right | D6-D14, E6-E8 |
| 10 | 10 | engines, history, team decisions and what is open | E1-E5, D15-D36 (skim), G, H, Open |
| 11-14 | rehearsal | the talks, a mock question round, the self-tests again | the whole index |

---

## Session 1. The question, the experiment and the big picture

**Goal.** Say in one minute what DOLL-E measures, why it must fly, what the simulator is for, and how the software
is organised.

**Read.**
- [OVERVIEW.md](OVERVIEW.md): "The 1-minute version", sections 2 and 3.
- [README.md](../README.md): "Start here", "The answer so far", "How realistic is it?".
- `notebooks/00_start_here.ipynb`: all of it (sections 1-6).
- [DECISIONS.md](DECISIONS.md): section 1 ("How to read it"), then A1-A8 and E1.

**Run.**

```bash
python examples/sticking_threshold.py
```

It prints the sticking speeds of 50 um radius grains: silica-silica 6.1752 mm/s (3.8266 without damping),
graphite-graphite 38.3586 (15.5758), silica on the borosilicate wall 6.6621 (3.8353), graphite on the wall 18.2333
(8.4819).

**Try it yourself.** Why microgravity, and why the dust has to start close together:

```python
import math
import dolle_core as dc
g0 = 9.80665
for a_g in (1.0, 1e-4):                       # how far a grain falls in 1 s
    print(f"{a_g:g} g: falls {0.5 * a_g * g0 * 1e3:.2f} mm in 1 s")
R = 50e-6                                     # mean free path of 100 grains in one tubule
volume = 12.7e-3 * 12.7e-3 * 38.1e-3          # m^3
n = 100 / volume                              # grains per m^3
sigma = math.pi * (2 * R) ** 2                # collision cross-section of two equal spheres
print(f"mean free path {1 / (n * sigma):.2f} m")
```

It prints `1 g: falls 4903.32 mm in 1 s`, `0.0001 g: falls 0.49 mm in 1 s` and `mean free path 1.96 m`. Change
`100` to `800` (the dust amount in the mission config) and see the mean free path shrink 8 times: still far longer
than the 12.7 mm tubule.

Then open the 3-D page (https://thebestproever.github.io/100micron-sim/sim3d/) and watch the replay of the two
tubules: which one forms clumps?

**Self-test.**

<details><summary>1. What single quantity does DOLL-E measure?</summary>

The sticking speed (threshold): the impact speed below which two ~100 um grains stick instead of bouncing, for
carbon and for silica (A1).
</details>

<details><summary>2. Why can it not be measured on the ground?</summary>

At 1 g a grain falls 4.9 m in one second; mm/s motions are swamped by gravity. In the 1e-4 g of the flight's quiet
window it falls 0.49 mm in a second (notebook 00 section 2).
</details>

<details><summary>3. What are the predicted sticking speeds for two 100 um grains?</summary>

Silica 6.18 mm/s, graphite 38.36 mm/s (`python examples/sticking_threshold.py`; about 6 times apart).
</details>

<details><summary>4. What are dolle_core, dolle_sim and extras?</summary>

`dolle_core`: the readable physics in plain numpy, giving the same numbers as the simulator. `dolle_sim`: the flight
simulator (the v0.5 event-driven engine, flight loads, cameras, 26 model checks). `extras/`: the optional v0.6
full-contact engine and the cross-checks against LAMMPS and REBOUND (OVERVIEW section 3).
</details>

<details><summary>5. Is the simulator's sticking speed "the answer"?</summary>

No: it is design guidance (A6). For 1.2 um silica on a flat target the model gives 0.276 m/s where Poppe, Blum &
Henning (2000) measured about 1.2 m/s, 4.3 times more (notebook 00, first code cell). The flight measures what the
model can only estimate.
</details>

<details><summary>6. Why did the project write its own simulator instead of using LAMMPS or REBOUND?</summary>

LAMMPS has the right contact physics but moves every grain with one global ~10 ns time step, which makes a
15-minute flight take weeks of computing per tubule (13-49 core-days to power-off, measured with
`extras/tools/lammps_benchmark.py`, `docs/HISTORY.md`); softening the grains to go faster breaks the wall pull-off
physics. REBOUND has no walls, grain spin or sticking. Both are used as outside checks instead (E1, D10, D11).
</details>

<details><summary>7. What does "ASSUMED" mean in this repository?</summary>

A value or choice with no published source or check behind it, stated openly. Every number is either sourced (a
paper, a data sheet, the code, a check) or marked ASSUMED (D9, PROVENANCE.md).
</details>

**Explain it out loud.**
- *"What is this project, in one minute?"* Key points: the question (sticking speed of 100 um carbon and silica
  grains); why it matters (how dust grows toward planets); why fly (gravity swamps mm/s motions); what the simulator
  does (predicts the speeds, checks that dust can be freed and filmed); why trust it (published physics, exact
  checks, outside codes, a readable core).
- *"Who decided what this simulator should do?"* Key points: the team defined the question (A1); the owner asked
  for the whole flight and for every number to be sourced (A3, D9); engineering choices were made while building and
  each is recorded with its evidence (DECISIONS.md, section 1, explains the "decided by" words).

---

## Session 2. The sticky contact: JKR theory

**Goal.** Explain where the sticky force comes from, what the JKR theory says, what the reduced quantities are and
what the Tabor parameter tells you.

**Read.**
- `notebooks/01_two_grains_colliding.ipynb` section 1 (the two grains and the reduced quantities, cells 3-7) and
  section 2 (the contact law, 2.1-2.4, cells 8-23).
- `dolle_core/pair.py` (how two bodies become one effective contact) and `dolle_core/jkr.py` (the JKR formulas).
  Each function states its equation and source.
- [DECISIONS.md](DECISIONS.md) B1, B2, B7.

**Try it yourself.**

```python
import dolle_core as dc
p = dc.make_pair("silica", 50e-6, "silica", 50e-6)
print(f"E* = {p.E_star/1e9:.2f} GPa, R* = {p.R_star*1e6:.1f} um, m* = {p.m_star:.3e} kg, w = {p.w:.3f} J/m^2")
a0 = dc.equilibrium_contact_radius(p.R_star, p.w, p.E_star)
Fc = dc.pull_off_force(p.R_star, p.w)
d_c = dc.break_overlap(p.R_star, p.w, p.E_star)
U0 = dc.hysteresis_energy(p.R_star, p.w, p.E_star)
print(f"a0 = {a0*1e9:.1f} nm, F_c = {Fc*1e6:.2f} uN, delta_c = {d_c*1e9:.3f} nm, |U0| = {U0:.3e} J")
print(f"undamped sticking speed sqrt(2|U0|/m*) = {(2*U0/p.m_star)**0.5*1e3:.3f} mm/s")
for m in ("silica", "graphite"):
    q = dc.make_pair(m, 50e-6, m, 50e-6)
    print(m, "Tabor parameter", round(dc.tabor_parameter(q.R_star, q.w, q.E_star), 2))
```

It prints:

```text
E* = 27.80 GPa, R* = 25.0 um, m* = 5.236e-10 kg, w = 0.050 J/m^2
a0 = 251.4 nm, F_c = 5.89 uN, delta_c = -0.696 nm, |U0| = 3.833e-15 J
undamped sticking speed sqrt(2|U0|/m*) = 3.827 mm/s
silica Tabor parameter 1.44
graphite Tabor parameter 8.76
```

Then change `"silica", 50e-6, "silica", 50e-6` to a grain on a wall, `"silica", 50e-6, "glass_borosilicate",
None` (a wall has infinite radius and mass), and see which numbers change and why: R* becomes the grain's 50 um and
m* the grain's mass, w = 0.045 J/m^2 (the measured bead-on-glass value, C6), F_c = 10.62 uN, and the undamped
sticking speed 3.835 mm/s.

**Self-test.**

<details><summary>1. Where does the sticky force come from?</summary>

Surface energy gamma (J/m^2): creating contact area releases energy 2 gamma per unit area for two like surfaces
(the work of adhesion w = 0.050 J/m^2 for silica). JKR balances that energy against the elastic energy of squashing
the spheres (Johnson, Kendall & Roberts 1971; notebook 01 section 2.2).
</details>

<details><summary>2. What are R*, m* and E*, and why use them?</summary>

The reduced radius, mass and modulus: a two-body contact behaves like one sphere of radius R* = R1 R2/(R1 + R2)
with mass m* against a flat, with 1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2. For two equal 50 um grains R* = 25 um
(notebook 01 section 1; `dolle_core/pair.py`).
</details>

<details><summary>3. What is the pull-off force, and how big is it here?</summary>

The largest tension the contact can hold, F_c = 1.5 pi w R*: 5.89 uN for two silica grains. For a single silica
grain that is 574 times its weight (notebook 03 section 2 table, column F_c/mg).
</details>

<details><summary>4. Why does the contact keep energy (hysteresis)?</summary>

The contact snaps on at zero overlap but only breaks after the neck has been stretched to the break overlap
delta_c = -0.696 nm. The work done stretching the neck, |U0| = 3.833e-15 J, is never returned (notebook 01
section 2.4).
</details>

<details><summary>5. Where does the contact break, and what was the alternative?</summary>

At the end of the stable branch (contact radius a0/6^(2/3)), the rule of Chokshi et al. (1993) and Dominik &
Tielens (1997). The alternative, breaking at the maximum tension (LAMMPS's rule), gives undamped thresholds 1.42297
times lower (B2; notebook 01 cell 19).
</details>

<details><summary>6. What does the Tabor parameter say, and why does it matter for silica?</summary>

It compares the elastic neck height with the range of the surface forces: large means JKR applies, small means the
DMT limit. Graphite pairs 8.76 (JKR fine); silica pairs 1.44 (between JKR and DMT), so JKR probably overstates
silica's hysteresis and its 6.2 mm/s is probably an upper estimate (B1; the range z0 = 0.3 nm is ASSUMED).
</details>

<details><summary>7. How are unlike materials combined?</summary>

Reduced modulus by the standard Hertz rule; work of adhesion as the geometric mean w = 2 sqrt(gamma1 gamma2),
dry restitution sqrt(e1 e2) and the smaller yield pressure: all three ASSUMED common practice (B7). They matter
mostly for grain-wall contacts.
</details>

**Explain it out loud.**
- *"Why JKR and not another adhesion model?"* Key points: it is the standard model of the published dust-collision
  work (Chokshi et al. 1993; Dominik & Tielens 1997; Wada et al. 2007) and of LAMMPS, which allows a term-by-term
  cross-check; DMT has no neck and no hysteresis; Maugis-Dugdale spans both but needs a force range nobody has
  measured for these grains; for silica JKR is near its edge (Tabor 1.44), stated as an open item (O4).
- *"Which numbers here are measured and which are assumed?"* Key points: silica's gamma, E and nu are the published
  quartz values; its density, e_dry and yield pressure are ASSUMED; the force range in the Tabor parameter is
  ASSUMED (C1, B1).

---

## Session 3. One collision: energy loss, stick or bounce, the threshold

**Goal.** Walk through one impact from touch to stick or bounce, name the two ways energy is lost, and explain how
the sticking speed is computed and how accurate it is.

**Read.**
- `notebooks/01_two_grains_colliding.ipynb` sections 3-12 (cells 24-57): damping, the integrator, one impact, the
  force-overlap loop, the energy budget, the stick-or-bounce tests, the threshold, restitution against speed,
  plastic yield, why a head-on impact is enough.
- `dolle_core/damping.py`, `dolle_core/collide.py` (`collide_pair` is the heart: about 40 lines),
  `dolle_core/plastic.py`.
- [DECISIONS.md](DECISIONS.md) B3-B6, D1, D2.

**Run.**

```bash
python examples/collide_two_grains.py
```

Silica at 4 mm/s sticks; at 20 mm/s it bounces with e = 0.7562, contact 0.76 us, 340 steps, maximum overlap
4.557 nm.

**Try it yourself.** Change the speed, the number of sub-steps and the break rule:

```python
import dolle_core as dc
for v in (0.004, 0.006, 0.0065, 0.010, 0.020, 0.100):
    r = dc.collide("silica", 50e-6, "silica", 50e-6, u1=v)
    print(f"{v*1e3:6.1f} mm/s: {r.outcome:6s} e = {r.e:.3f}")
p = dc.make_pair("silica", 50e-6, "silica", 50e-6)
print("threshold, 300 steps:", round(dc.sticking_threshold(p) * 1e3, 3), "mm/s")
print("threshold, 100 steps:", round(dc.sticking_threshold(p, n_sub=100) * 1e3, 3), "mm/s")
print("break at maximum tension:", round(dc.sticking_threshold(p, break_mode="force") * 1e3, 3), "mm/s")
print("yield starts at", round(dc.yield_speed(p.m_star, p.R_star, p.E_star, p.p_y), 2), "m/s")
```

It prints:

```text
   4.0 mm/s: stick  e = 0.000
   6.0 mm/s: stick  e = 0.000
   6.5 mm/s: bounce e = 0.229
  10.0 mm/s: bounce e = 0.603
  20.0 mm/s: bounce e = 0.756
 100.0 mm/s: bounce e = 0.829
threshold, 300 steps: 6.175 mm/s
threshold, 100 steps: 6.186 mm/s
break at maximum tension: 4.91 mm/s
yield starts at 11.13 m/s
```

Notice that the restitution climbs toward e_dry = 0.85 at high speed: far above the threshold the adhesion matters
less and only the damping is left.

**Self-test.**

<details><summary>1. Name the two ways an impact loses kinetic energy.</summary>

The adhesion hysteresis (the neck's stretching work, never returned) and the damping inside the material, set by
the dry restitution e_dry (OVERVIEW section 1, step 3). At 20 mm/s the damping takes 0.391 of the incoming energy
and the hysteresis 0.037 (`python examples/collide_two_grains.py`).
</details>

<details><summary>2. What is e_dry, and why is it "dry"?</summary>

The restitution a non-sticky impact of that material would have (no adhesion): the damping is calibrated so that
such an impact rebounds with exactly e_dry, at any speed (B3; notebook 01 section 3). "Dry" means without the
adhesive part. It is ASSUMED for every material (C3).
</details>

<details><summary>3. How does the code decide stick or bounce?</summary>

Four tests (B5): stick if the kinetic energy at touch is below the hysteresis energy, if the grains fall back into
contact, or if kinetic plus stored energy is negative; bounce once the overlap passes the break overlap. One speed
gives one outcome; scatter comes only from scatter in the grains' properties (C8).
</details>

<details><summary>4. How is the sticking speed found, and how accurate is it?</summary>

By bisection on the impact speed until the bracket is within 0.2% (D2). With 300 sub-steps per contact it is 3.8e-4
above the converged value (notebook 01 cell 32); with 100 steps (the browser) 6.186 instead of 6.175 mm/s.
</details>

<details><summary>5. Why velocity Verlet and 300 sub-steps?</summary>

One force evaluation per step, second order, and momentum exact because both grains get the same force in every half
kick (D1). RK4 costs four evaluations and the jumps at snap-on and break limit its accuracy anyway. 300 steps give
3.8e-4 accuracy, far below the material uncertainties.
</details>

<details><summary>6. Does plastic yield change the threshold?</summary>

No: yield starts at 11.13 m/s for silica pairs and 0.836 m/s for graphite pairs, far above the mm/s thresholds;
thresholds with and without yield agree to all digits (notebook 01 cell 53). It matters for release kicks
(~1.2-1.5 m/s, B4, F14).
</details>

<details><summary>7. Why is a head-on, one-dimensional impact enough?</summary>

Only the speed along the line of centres decides. The full 3-D contact model gives the same threshold to 1.6e-3
(time-step error: silica 6.186 against 6.177 mm/s), and off-centre hits at r/2 and r give normal sticking speeds of
6.1860 and 6.1861 mm/s, the same as head-on in 3-D (notebook 01 section 12, `check:dem_vs_1d`). Grazing and spinning impacts beyond those cases are ASSUMED to behave the same (B6).
</details>

<details><summary>8. What happens to the threshold if the break rule changes to "maximum tension"?</summary>

Silica drops from 6.175 to 4.91 mm/s (20.5% lower, B2): this is one of the decisions that would change a headline
number.
</details>

**Explain it out loud.**
- *"Walk me through what happens when two grains touch."* Key points: snap-on at zero overlap (force -(8/9) F_c),
  compression to a few nm, elastic rebound, the neck stretches past zero overlap, then either it breaks (bounce) or
  the grains cannot escape the adhesive well (stick); energy lost to damping and hysteresis (notebook 01 section 5).
- *"How do you know the integrator is accurate enough?"* Key points: convergence study (100 / 300 / 1000 / 3000
  sub-steps), 300 is 3.8e-4 from converged, bisection to 0.2%, and the undamped threshold matches the analytic
  energy balance (3.825 vs 3.827 mm/s, `check:threshold`).

---

## Session 4. Momentum and energy: what the code guarantees

**Goal.** Show why momentum is conserved exactly by the code, where kinetic energy goes, and what a violation
would look like.

**Read.**
- `notebooks/02_momentum_conservation.ipynb`: all (sections 1-7; section 6 is a deliberately broken version).
- The lines of `dolle_core/collide.py` that notebook 02 cell 8 prints (the half kick, drift, force, "equal and
  opposite").
- [DECISIONS.md](DECISIONS.md) D1 (the "Why"), D3, D12, B13.

**Try it yourself.** Two unlike grains, both moving:

```python
import dolle_core as dc
r = dc.collide("silica", 50e-6, "graphite", 50e-6, u1=0.030, u2=-0.010)   # unlike grains, both moving
p = r.pair
before = p.m1 * r.u1 + p.m2 * r.u2
after = p.m1 * r.v1 + p.m2 * r.v2
print(f"{r.outcome}, e = {r.e:.3f}; momentum before {before:.6e}, after {after:.6e} kg m/s")
ke_in = 0.5 * p.m1 * r.u1**2 + 0.5 * p.m2 * r.u2**2
ke_out = 0.5 * p.m1 * r.v1**2 + 0.5 * p.m2 * r.v2**2
print(f"kinetic energy kept: {ke_out / ke_in:.3f} of the incoming")
```

It prints `bounce, e = 0.617; momentum before 1.989675e-11, after 1.989675e-11 kg m/s` and `kinetic energy kept:
0.486 of the incoming`. Try `u1=0.010, u2=-0.005`: the pair approaches at 15 mm/s, below the silica-graphite
threshold (17.19 mm/s, notebook 03 section 2), so it sticks; for a stick the code SETS both velocities to the
centre-of-mass velocity (read the `CollisionResult` docstring in `dolle_core/collide.py` to see why).

Then run `python examples/collide_two_grains.py` again and read the four-grains-on-a-line part: total momentum is the
same at the start and the end, to a relative change of 3.4e-15.

**Self-test.**

<details><summary>1. Why is momentum conserved exactly in the code, not just approximately?</summary>

Every contact force is applied to both bodies as an equal and opposite pair in the same half kick (Newton's third
law written literally: `acc1, acc2 = -F_tot / pair.m1, F_tot / pair.m2`). The changes cancel in every step, so total
momentum changes only by floating-point round-off: 1.5e-15 relative over 340 steps (notebook 02 cells 5 and 12).
</details>

<details><summary>2. If momentum is conserved, why is kinetic energy not?</summary>

Momentum conservation fixes the centre-of-mass motion only. The relative motion loses energy to damping and
hysteresis; that loss is what makes sticking possible (notebook 02 section 4: of the relative kinetic energy at
20 mm/s, 0.3914 goes to damping, 0.0366 to hysteresis, 0.5719 = e^2 stays kinetic).
</details>

<details><summary>3. What would a violation look like?</summary>

Notebook 02 section 6 breaks the code on purpose: with the force applied to only one grain, momentum goes from
2.0944e-11 to -1.7659e-11 kg m/s in one impact. Such a bug would be caught at once by `check:conservation`.
</details>

<details><summary>4. How is it checked in 3-D, with spinning clumps?</summary>

`check:conservation` (momentum dP/P = 5.7e-17 and angular momentum dL/L = 1.6e-16 through sticking and
fragmentation) and `check:impulse` for the bounce impulse (CHECKS.md rows 10-11).
</details>

<details><summary>5. What does "the readable core equals the full code" mean here?</summary>

`tests/test_dolle_core.py` runs `dolle_core` and `dolle_sim` on the same cases: identical sticking speeds and
restitution within round-off (notebook 00 cell 10: silica 6.175177 mm/s in both, restitution difference 2.2e-16;
D12).
</details>

<details><summary>6. Why can a stuck pair's velocities not be read from the last time step?</summary>

The pair is still vibrating in its adhesive well when the code decides it is bound: at 5 mm/s the last step has
v1 = 0.6006 and v2 = 4.3994 mm/s, while the centre of mass moves at 2.5000 mm/s (notebook 02 cell 17). The result
reports the centre-of-mass velocity, which momentum conservation guarantees.
</details>

**Explain it out loud.**
- *"Is momentum conserved?"* Key points: exactly, to round-off (1e-15), because each force acts on both grains equal
  and opposite in the same step; shown in notebook 02, with a broken version for contrast, and checked in the full
  simulator by `check:conservation`.
- *"Then where does the energy go?"* Key points: into the material (damping, set by e_dry) and into the adhesion
  neck (hysteresis); for a stick, the rest stays as vibration of the stuck pair; that loss is the physics being
  measured.

---

## Session 5. Silica against carbon: the materials and their uncertainty

**Goal.** Explain why graphite sticks about 6 times more easily than silica, property by property, which inputs
are measured and which are assumed, and how much each uncertainty matters.

**Read.**
- `notebooks/03_silica_vs_carbon.ipynb`: all (sections 1-8, then "What this means for DOLL-E").
- `dolle_core/materials.py` (every value with its source) and `dolle_core/scaling.py` (the formula by hand and
  `damping_factor`).
- [OVERVIEW.md](OVERVIEW.md) section 2 ("Why graphite sticks ~6x more easily", "Which carbon is not decided").
- [DECISIONS.md](DECISIONS.md) C1-C5, C8, G14, G15, open decisions O1 and O2.

**Run.**

```bash
python examples/silica_vs_carbon.py
```

It splits the ratio: silica 6.177 mm/s to graphite 38.305 mm/s is x6.201 = gamma^(5/6) x2.498 times E*^(-1/3)
x1.709 times rho^(-1/2) x0.953 (material values x4.070) times f(e_dry) x1.523 (ASSUMED e_dry 0.70 against 0.85).
(This script uses 1,000 sub-steps and a 1e-4 bisection tolerance, so it prints 38.305 where the default settings of
`sticking_threshold.py`, 300 sub-steps and 0.2%, give 38.359 mm/s: the two agree within that 0.2%, D1 and D2.)

**Try it yourself.** Change a material value without editing any file:

```python
from dataclasses import replace
import dolle_core as dc
def v(mat):
    return dc.sticking_threshold(dc.make_pair(mat, 50e-6, mat, 50e-6)) * 1e3
si = dc.MATERIALS["silica"]
print("table silica         ", round(v(si), 2), "mm/s")
print("gamma 0.2 (vacuum)   ", round(v(replace(si, gamma=0.2)), 2), "mm/s")
print("e_dry 0.90           ", round(v(replace(si, e_dry=0.90)), 2), "mm/s")
print("e_dry 0.70           ", round(v(replace(si, e_dry=0.70)), 2), "mm/s")
print("graphite             ", round(v("graphite"), 2), "mm/s")
for e in (0.7, 0.85, 0.9):
    print(f"damping factor f({e}) = {dc.damping_factor(e):.3f}")
```

It prints silica 6.18, 34.96 with gamma = 0.2 J/m^2 (the vacuum value argued by Kimura et al. 2015), 5.32 with
e_dry = 0.90, 9.41 with e_dry = 0.70, graphite 38.36 mm/s, and f(0.7) = 2.459, f(0.85) = 1.614, f(0.9) = 1.392.
In a `dolle_sim` config the same change is `"materials": {"silica": {"gamma": 0.2}}` (DECISIONS.md section 1).

**Self-test.**

<details><summary>1. Write the sticking-speed formula without damping.</summary>

v0 = 1.6394 gamma^(5/6) E*^(-1/3) R*^(-5/6) rho^(-1/2) for two equal spheres (`python
examples/sticking_threshold.py`; notebook 03 section 4). With damping, multiply by f(e_dry).
</details>

<details><summary>2. Split the x6.2 between graphite and silica.</summary>

Surface energy x2.498, stiffness x1.709 (softer graphite makes a larger contact), density x0.953, together x4.070
from the material values, times x1.523 from the ASSUMED e_dry (`python examples/silica_vs_carbon.py`). So about a
quarter of the logarithm of the ratio (23%, notebook 03 cell 21) rests on a number nobody has measured.
</details>

<details><summary>3. What is f(e_dry), and why is it so important?</summary>

The factor by which damping raises the undamped threshold; it depends on e_dry alone, not on the material or size
(notebook 03 section 5 shows the same 1.61479 for every pair). f(0.85) = 1.61 (silica) and f(0.70) = 2.46
(graphite). A small change in e_dry moves the threshold a lot: 0.85 to 0.90 lowers silica from 6.18 to 5.32 mm/s.
It is the largest single uncertainty (C3, O1).
</details>

<details><summary>4. Which silica and which carbon does the simulator use, and are they decided?</summary>

Silica with the published quartz values and graphite with the values of Chokshi et al. (1993), both as placeholders:
the team has not chosen the powders (G14, G15). Depending on the carbon, it sticks 1.8 to 36 times more easily than
the table silica (notebook 03 cell 32: graphite 21.7-219.9, glassy carbon 20.2-126.5, amorphous carbon 11.1-152.6
mm/s; ranges mostly ASSUMED).
</details>

<details><summary>5. What would change the conclusion "carbon sticks more easily than silica"?</summary>

A silica surface energy as high as argued for dry silica in vacuum: gamma = 0.25 J/m^2 gives silica 42.10 mm/s,
above table graphite (notebook 03 cell 28). Kreuzig et al. (2024) find no such rise for water-repellent silica, so
this is open (C1).
</details>

<details><summary>6. How does grain size enter?</summary>

The threshold scales as radius^(-5/6): from 106 um to 45 um diameter silica goes from 5.89 to 12.02 mm/s, a factor
2.042 = (106/45)^(5/6) (notebook 03 section 7; A2).
</details>

<details><summary>7. Can the flight measure gamma directly?</summary>

Not by the sticking speed alone: it goes as gamma^(5/6) times f(e_dry), so a measured speed cannot separate the two.
`python -m dolle_sim invert --material silica --vstick 0.006` returns an EFFECTIVE surface energy (0.02413 J/m^2)
that absorbs every other model error (A7). The restitution of bouncing impacts pins e_dry (O1).
</details>

**Explain it out loud.**
- *"Why does carbon stick more easily?"* Key points: higher surface energy (x2.5) and a softer material that makes a
  bigger contact (x1.7) give x4; the assumed energy loss gives the remaining x1.5; the carbon and silica types are not
  chosen, so the range is 1.8x-36x.
- *"What is your biggest uncertainty, and what are you doing about it?"* Key points: e_dry (ASSUMED; a 5% change moves
  the threshold ~10-12%) and the materials; the flight measures both sticking and the restitution of bounces, which is
  how e_dry gets measured; a ground drop test could measure the wall adhesion (O1, O11).

---

## Session 6. A grain on the wall

**Goal.** Explain what holds a grain on the tubule wall, which limit breaks first, and why the dust stays on the
walls through launch but can roll.

**Read.**
- `notebooks/04_grain_on_a_wall.ipynb`: all (sections 1-8).
- `dolle_core/wall.py` (`wall_limits`).
- [OVERVIEW.md](OVERVIEW.md) section 2 ("Dust on the walls").
- [DECISIONS.md](DECISIONS.md) B8, B9, C6, C7, C9, C10, F20, and the open decisions O5, O11, O15.

**Run.**

```bash
python examples/grain_on_wall.py
```

Silica (50 um) on borosilicate: pull-off 1033.74 g, sliding (mu 0.4) 826.99 g, rolling 1.2935 g. Graphite: 1627.71,
1627.71 and 3.5897 g.

**Try it yourself.** Change the wall adhesion, the rolling hysteresis h, the friction and a pressing load:

```python
import dolle_core as dc
for kw in ({}, {"f_adh": 0.02}, {"h": 0.05}, {"h": 0.6}, {"mu": 0.3}, {"press_g": 25.0}):
    L = dc.wall_limits("silica", 50e-6, "glass_borosilicate", **kw)
    print(f"{str(kw):18s} pull-off {L.pull_off_g:8.1f} g  sliding {L.slide_g:8.1f} g  rolling {L.roll_g:6.3f} g")
L = dc.wall_limits("graphite", 50e-6, "glass_borosilicate")
print(f"graphite           pull-off {L.pull_off_g:8.1f} g  sliding {L.slide_g:8.1f} g  rolling {L.roll_g:6.3f} g")
```

It prints:

```text
{}                 pull-off   1033.7 g  sliding    827.0 g  rolling  1.294 g
{'f_adh': 0.02}    pull-off     20.7 g  sliding     16.5 g  rolling  0.007 g
{'h': 0.05}        pull-off   1033.7 g  sliding    827.0 g  rolling  0.129 g
{'h': 0.6}         pull-off   1033.7 g  sliding    827.0 g  rolling  1.552 g
{'mu': 0.3}        pull-off   1033.7 g  sliding    620.2 g  rolling  1.294 g
{'press_g': 25.0}  pull-off   1058.7 g  sliding    837.0 g  rolling  1.309 g
graphite           pull-off   1627.7 g  sliding   1627.7 g  rolling  3.590 g
```

Then run the launch-load report and its weakly bound variant:

```bash
python -m dolle_sim mission configs/mission_quarter_deck.json --loads-only
python -m dolle_sim mission configs/mission_quarter_deck.json --loads-only --scenario rough_grains
```

On clean glass every phase prints "pulled off: 0.0%". With `rough_grains` (adhesion x0.02) T1's median pull-off
threshold is 21.7 g, the 31.2 g splashdown would pull off 71.8% of grains taken one by one, and the ground vibration
test would free 87.8%.

**Self-test.**

<details><summary>1. Name the four limits of a grain on a wall.</summary>

Pull-off (normal), sliding, rolling and twisting (B9; notebook 04 sections 2-5).
</details>

<details><summary>2. Which limit breaks first, and by how much?</summary>

Rolling: 1.29 g for silica on borosilicate, against 1,034 g for pull-off and 827 g for sliding (the ratio of the
rolling to the pull-off limit is 1.251e-3, notebook 04 cell 15). Rolling is resisted only by a tiny lever arm, the
critical rolling displacement xi_crit = h a0/12 = 15.64 nm (notebook 04 section 4).
</details>

<details><summary>3. What does the launch do to the dust, then?</summary>

The largest loads (25 g pressing during the burns, ~10 g sideways from the spin, 31.2 g at splashdown) are far below
pull-off: the dust stays on the walls. The sideways spin load exceeds the rolling limit, so it can roll (OVERVIEW
section 2; notebook 04 section 7: T1 silica starts rolling at T+10.0 s).
</details>

<details><summary>4. Does the default engine show the rolling?</summary>

No. The v0.5 engine holds wall grains still until they are pulled off (B8). Only the optional v0.6 engine in
`extras/` resolves rolling and sliding along the walls, and its full-flight runs were stopped (G6, O16).
</details>

<details><summary>5. How uncertain is the rolling limit?</summary>

It is linear in the rolling hysteresis h (ASSUMED 0.5, fitted to ~1 um silica): h = 0.05 to 0.6 gives 0.129 to
1.552 g. Every value is below the ~10 g spin load, so the conclusion "can roll" holds; h sets how far (C10, O5).
</details>

<details><summary>6. Why is friction 0.4 here but 0.3 in the default engine?</summary>

`dolle_core` and the v0.6 engine use a sourced per-pair table (silica-glass 0.4 from Meyer et al. 2014); the v0.5
engine and the launch report use one unsourced 0.3 (C9). On the 3-D page it changes nothing; the launch report also
uses a different sliding formula, mu (F_c - pull) instead of mu |F_ne + 2 F_c| (F20, O15).
</details>

<details><summary>7. How do you free the dust for the experiment, then?</summary>

That is the main open design question (O3). A gentle 50 g shaker frees nothing; a 5,000 g kick at 5 kHz frees dust
but moves the wall at 1.56 m/s, 253 times silica's sticking speed (notebook 04 cell 33). Freeing 1,000 g-bound grains
at 1 cm/s would need ~156 kHz (ultrasonic).
</details>

<details><summary>8. What single wall measurement would help most?</summary>

The work of adhesion of the real powders on the real glass or coating, for example by the drop test of Zafar et al.
(2014): every wall limit scales with it (rolling as w^(4/3)) (C6, O11).
</details>

**Explain it out loud.**
- *"Will the dust survive launch?"* Key points: adhesion holds a 100 um grain about a thousand times harder than its
  weight on clean glass; nothing is pulled off at any phase; but it can roll sideways under the spin; with very weak
  adhesion (x0.02) splashdown and even the ground vibration test could free it (C7).
- *"How will you release it?"* Key points: undecided; compared as scenarios (F14); the problem is speed: anything
  strong enough to free 1,000 g-bound dust throws it at m/s, while sticking happens at mm/s.

---

## Session 7. The flight, the forces and the mission runs

**Goal.** Describe the flight as a sequence of loads on the dust, list the forces in the model and which ones
matter, and read the mission scenario table, knowing exactly which engine and code produced it.

**Read.**
- `notebooks/05_the_flight_in_brief.ipynb`: all (sections 1-6).
- [MISSION_RUNS.md](MISSION_RUNS.md): the header, the phase table, "What this means for the design" and the scenario
  table.
- [README.md](../README.md) "How realistic is it?" (the force table).
- [DECISIONS.md](DECISIONS.md) A3, A5, B18-B21, B23, F1-F21 (F3 and F14 in detail), G10.

**Run.**

```bash
python -m dolle_sim preflight configs/flight_carbon_vs_silica.json
```

It prints, for each material, the sticking speeds, the expected collisions and the acceleration budget, with the
switched-off forces marked "(off)". For graphite: residual 1.00e-04 g, g-jitter 5.00e-05 g, thermophoresis
3.21e-05 g, radiation pressure 2.16e-06 g (off).

**Try it yourself.** Switch a force on and make the chamber isothermal (this writes a new file, `my_flight.json`, in
the current folder; the shipped config is not touched):

```python
import json
cfg = json.load(open("configs/flight_carbon_vs_silica.json"))
cfg["forces"]["radiation_pressure"] = True                       # switch one force on
cfg["environment"]["temperature_gradient_k_m"] = [0.0, 0.0, 0.0]  # an isothermal chamber
json.dump(cfg, open("my_flight.json", "w"), indent=2)
print("wrote my_flight.json")
```

```bash
python -m dolle_sim preflight my_flight.json
```

Radiation pressure now shows without "(off)" (2.16e-06 g for graphite) and thermophoresis drops to 0. Then compare
the residual spin options in the launch-load report:

```bash
python -m dolle_sim mission configs/mission_quarter_deck.json --loads-only --scenario spin_1hz
```

In the microgravity phase the sideways load on T1 rises from 0.000116 g (1 rpm) to 0.415 g (1 Hz).

**Self-test.**

<details><summary>1. List the flight's phases with their loads on the dust.</summary>

Two 25 g burns (0-5.2 s and 17.4-29.1 s) with a ~3 g drag pull in the coast between them; spin up to 4.8 Hz (up to
9.56 g sideways at T1's far corner); yo-yo de-spin at T+54 s to 1 rpm; the microgravity window T+88-313 s (225 s,
residual 1e-4 g); re-entry peak 7.93 g sideways at T+368.6 s; parachute 5 g; splashdown 31.2 g at T+882 s; runs end
at T+887 s (notebook 05 cells 4 and 8).
</details>

<details><summary>2. Where do these numbers come from?</summary>

Burn times, 25 g, 4.8 Hz, the parachute and splashdown times from the launch provider's guide (RockSat User Guide
2026); de-spin and the 100 km crossings from mission fact sheets of the same vehicle; the re-entry from a point-mass
fall through an NRLMSIS 2.1 atmosphere; flat 25 g thrust, the 1e-4 g residual, the 1 rpm residual spin and the
0.05 s splash stop are ASSUMED (F3-F8).
</details>

<details><summary>3. Which small forces are switched off, and why is that safe?</summary>

Mutual gravity, the Lorentz force, an external field, photophoresis, radiation pressure, Brownian motion and the
gravity gradient: for the flight demo each is at most 2.2% of the 1e-4 g residual (radiation pressure on graphite,
2.16e-06 g in the pre-flight check). The pre-flight check still prints them and warns above 5% (threshold ASSUMED)
(B21).
</details>

<details><summary>4. Thermophoresis is on. Why does it matter?</summary>

A temperature gradient pushes grains (Waldmann's free-molecular law). With the demo's ASSUMED 3 K/m at 1 Pa it is
3.21e-05 g for graphite, about a third of the residual acceleration, so it is not negligible: the chamber should be
kept nearly isothermal (B23; `docs/USER_GUIDE.md`, design findings).
</details>

<details><summary>5. What does the mission scenario table say about releasing the dust?</summary>

The nominal 50 g shaker frees 0 of 3,200 grains; only kicks free dust (500-800 per tubule), at median 1.2-1.5 m/s,
far above the sticking speeds; few grain-grain impacts happen near the sticking speeds in any option, so a gentler
release is needed (F14; MISSION_RUNS.md).
</details>

<details><summary>6. Which engine and which code version produced that table?</summary>

The v0.5 event-driven engine, run on 2026-10-26 at commit 0731b2f, with the negligible forces on; it has not been
re-run since (G10). The flight demo result in the README was computed later, with v0.6.2, all forces on.
</details>

<details><summary>7. Why is the payload's spin modelled in a rotating frame?</summary>

In the tubule's own frame the walls are fixed, so wall contacts stay simple; the centrifugal, Euler and Coriolis
terms are added exactly, and inertially fixed vectors rotate in that frame (A5; `check:rotating_frame`,
`check:flight_frame`).
</details>

<details><summary>8. Why does the residual spin matter so much?</summary>

Centrifugal acceleration grows with the spin squared: at 1 Hz the sideways load on T1 is 0.415 g in the window,
thousands of times the 1e-4 g residual, and it decides where freed grains drift. The 1 rpm value is ASSUMED (F4).
</details>

**Explain it out loud.**
- *"How realistic is your flight model?"* Key points: published timeline from the launch provider's guide and fact
  sheets, a standard atmosphere model, every envelope assumption stated (F3); replace with accelerometer data when
  available; the stored scenario table is from the v0.5 engine (2026-10-26) and not re-run (G10, O13).
- *"Did you leave out any force?"* Key points: nothing identified was left out of the code (A4); seven tiny ones are
  switched off by default, each sized and printed by the pre-flight check (B21); thermophoresis, drag, charging and
  space charge stay on.

---

## Session 8. The 3-D simulation with rigid clumps

**Goal.** Explain how the 3-D page is computed (free flight, exact collision times, the 1-D contact model for each
impact, rigid clumps, break-up), what it leaves out, and how much the assumed start decides.

**Read.**
- `notebooks/06_the_3d_simulation.ipynb`: all (sections 1-10).
- `dolle_core/clumps.py`, `dolle_core/impacts.py`, `dolle_core/sim3d.py`, `dolle_core/scenes.py` (in that order;
  `START_SOURCES` in `scenes.py` gives the source of every input of the start).
- The 3-D page: the replay, "One impact up close" and "Try it: a live run in your browser".
- [DECISIONS.md](DECISIONS.md) B12-B17, B22, D4, D5, E9, G12, G13, F13.

**Try it yourself.** Change the start, the dust amount, the friction and the fragmentation (about half a minute in
all):

```python
from dolle_core import scenes, sim3d
from dataclasses import replace
def show(label, start):
    sims = scenes.run_two_tubules(start)
    out = []
    for m, s in sims.items():
        sm = sim3d.summary(s)
        out.append(f"{m} {sm['grain_grain_impacts']}/{s.counts['stick']}")
    print(f"{label:28s}", "  ".join(out))
show("baseline", scenes.two_tubule_start())
show("4 mm cloud", scenes.two_tubule_start(box_mm=4.0))
show("200 grains", scenes.two_tubule_start(n=200))
start = scenes.two_tubule_start()
show("friction 0.3 everywhere", {**start, "tubule": replace(start["tubule"], mu=0.3)})
show("no fragmentation", {**start, "tubule": replace(start["tubule"], fragmentation=False)})
```

It prints impacts / sticks:

```text
baseline                     silica 20/4  graphite 17/17
4 mm cloud                   silica 3/0  graphite 4/4
200 grains                   silica 46/11  graphite 42/41
friction 0.3 everywhere      silica 20/4  graphite 17/17
no fragmentation             silica 20/4  graphite 17/17
```

The start decides the outcome (a 4 mm cloud gives almost no impacts; twice the dust gives more than twice the
impacts), while friction and fragmentation change nothing here. In the page's sandbox, set "Start" to "spread" and
see that grains spread through the tubule practically never meet.

**Self-test.**

<details><summary>1. How is one 3-D impact decided?</summary>

By the same verified 1-D contact model, run with the effective mass of the contact point, 1/(n.K.n), and the normal
speed; for two single grains that is the reduced mass, and the outcomes and restitutions equal the 1-D ones exactly
(notebook 06 cell 14). A grain hitting the end of a dimer sees only 0.538 m, so part of the push turns the dimer
(cell 12).
</details>

<details><summary>2. What is a rigid clump, and what does it cost?</summary>

Grains that stick merge into one rigid body that keeps its momentum and angular momentum and can spin (B12). The
cost: no rolling, sliding or restructuring inside a clump (B17). An impact at 15 mm/s carries only 0.030 (silica) and
0.005 (graphite) of the restructuring energy 5 E_roll, but the rolling resistance is uncertain ~50x (notebook 06
cell 20).
</details>

<details><summary>3. How are collisions found without missing any?</summary>

The exact contact time of every candidate pair along the drift is the smaller root of a quadratic, |d + u t|^2 = S^2;
events are processed in time order and the bodies an impact changed are re-predicted (D5). The quadratic and a
bisection agree to 1.6e-17 s (notebook 06 cell 8).
</details>

<details><summary>4. When does a clump break?</summary>

By the Dominik & Tielens (1997) thresholds: erosion above 3 N_c E_break, shattering above 10 N_c E_break, paid only
from the energy the bounce really dissipated (B14). These were derived for aggregates of sub-micron grains, so for
2-3 grain clumps of 100 um grains they are an extrapolation.
</details>

<details><summary>5. What does the 3-D core leave out, and does it matter?</summary>

Gas drag (drag time ~1.3-1.5e4 s at 1e-3 Pa), payload roll and charging (B22). With the v0.5 engine, drag and
charging change no grain-grain outcome over 6 seeds x 100 grains; the 1 rpm roll moves where grains land and agrees
only within 2 sigma (silica sticks 21 to 27), so its effect is not shown to be nil (O6; notebook 06 section 9).
</details>

<details><summary>6. Why start from a compact cloud at 15 +/- 7.5 mm/s?</summary>

It is ASSUMED, because the release is undecided (G13, O3). 15 mm/s is the geometric mean of 6.2 and 38 mm/s, so the
same start is above silica's threshold and below graphite's. Spread through the tubule, grains essentially never meet
(mean free path ~2 m, session 1).
</details>

<details><summary>7. Does the 3-D core agree with the full simulator?</summary>

Yes: on identical starts the core and the v0.5 engine give the same events (silica 69 of 69, graphite 34 of 34
identical; notebook 06 cell 35), and momentum is kept to round-off through every bounce, merge and split.
</details>

**Explain it out loud.**
- *"What does the 3-D page show, and what does it not show?"* Key points: grains from a compact released cloud
  colliding in microgravity, silica mostly bouncing (20 impacts, 4 sticks) and graphite always sticking (17 of 17)
  and building clumps; it does not show how the dust is released, which is undecided; the start is ASSUMED and
  decides the counts.
- *"Why rigid clumps instead of full contact mechanics?"* Key points: exact momentum and spin bookkeeping, fast enough
  for a browser; the full model would need ~4e9 steps for 12 s (B12); the cost is no restructuring (open, O7).

---

## Session 9. How we know it is right

**Goal.** Explain the three layers of evidence (exact checks, established codes, a readable core that gives the same
numbers), what is tested automatically, and what is not verified.

**Read.**
- [OVERVIEW.md](OVERVIEW.md) section 4 and question 1 of section 7.
- [CHECKS.md](../CHECKS.md): Part 1 (26 model checks) in full; skim Part 2 and the LAMMPS and REBOUND sections.
- [PROVENANCE.md](../PROVENANCE.md): the first screen (how components are classed: published, standard, derived,
  assumed).
- [DECISIONS.md](DECISIONS.md) D6-D14, E6-E8.

**Run.**

```bash
python -m dolle_sim verify
```

About a minute; it ends with `26/26 checks passed`. Read each line: what is compared, the measured value, the
criterion in brackets.

**Try it yourself.** See the tests catch a change. Edit the silica surface energy in `dolle_core/materials.py`
(the `_m("silica", 2000.0, 54e9, 0.17, 0.025, ...)` line: change `0.025` to `0.030`) and run:

```bash
python -m pytest tests/test_dolle_core.py -q
```

Many tests fail, starting with `test_every_material_value_equals_dolle_sim`: the two material tables no longer agree.
Now make the same change in `dolle_sim/materials.py` (`gamma=0.025` in the `"silica"` entry) and run the test again:
only the tests that hold reference numbers on purpose fail (`test_threshold_reference_numbers`,
`test_wall_reference_numbers` and three others with hard-coded values). That is the decision "values exist in
several places, kept identical on purpose" (E7) and "reference numbers are hard-coded on purpose" (D12) at work.
Undo both edits with `git restore dolle_core/materials.py dolle_sim/materials.py`.

**Self-test.**

<details><summary>1. What are the three layers of evidence?</summary>

Exact solutions and conservation laws (26 model checks); agreement with established codes (the contact law in LAMMPS
term by term to 2.2e-6, flight trajectories against REBOUND to 3.2e-10 m); and a small readable version that gives
the same numbers (OVERVIEW section 4; CHECKS.md).
</details>

<details><summary>2. Give three of the model checks and their measured values.</summary>

For example: elastic limit e = 0.99990 (energy conserved without adhesion and damping); damping calibration
e = 0.8501 for e_dry = 0.85; momentum through sticking dP/P = 5.7e-17; no missed collisions, 0 overlap over 286
impacts; thermophoresis drift ratio 1.028 (CHECKS.md Part 1).
</details>

<details><summary>3. What exactly was compared with LAMMPS?</summary>

Undamped sticking speeds (silica 2.684 vs 2.691 mm/s, 0.26%; graphite 0.11%), restitution, pull-off (0.14%), and
seven set-ups of the v0.6 contact law term by term (to <= 2.2e-6). Damped collisions are not compared: LAMMPS refuses
its damping together with JKR (D10; CHECKS.md).
</details>

<details><summary>4. What does REBOUND check, and what not?</summary>

Trajectories (3.2e-10 m) and collision bookkeeping (impact counts and stick fractions within 1 sigma). It does NOT
check contact physics, and none of the physics comes from it (D11).
</details>

<details><summary>5. What is not verified?</summary>

The ASSUMED inputs (e_dry, the yield pressures, the rolling hysteresis, the residual acceleration and spin, the
release start); JKR's accuracy for silica (O4); the fragmentation thresholds at this grain size (B14); the threshold
fit's error bars, which covered the truth in only 9 of 20 cases (F19, O14); and the stored mission results, not
re-run since 2026-10-26 (G10).
</details>

<details><summary>6. Why can the decision log not silently go out of date?</summary>

`tests/test_decisions_doc.py` fails if any file path, `module.function`, `check:<id>`, test name or commit written in
code font in DECISIONS.md stops existing (D14). `extras/tests/test_engine_logic_doc.py` does the same for the v0.6
engine's decision map.
</details>

<details><summary>7. How is the provenance kept honest?</summary>

`dolle_sim/provenance.py` lists every component with its source, code location, checks and assumptions; it
generates PROVENANCE.md, and `tests/test_provenance.py` checks that it names only code and checks that exist and that
the generated files are up to date (D9).
</details>

**Explain it out loud.**
- *"How do you know the simulation is right?"* Key points: three layers (exact checks, outside codes, readable core);
  every check has a criterion and a measured value, in a generated table; and an honest list of what is not verified
  (the assumed inputs) and why the flight is needed.
- *"Could a student check this without trusting you?"* Key points: yes: `python -m dolle_sim verify` in a minute,
  `python -m pytest` for the tests, the notebooks derive every formula and print every number, and the provenance
  table traces every number to a paper, a data sheet, a check or the word ASSUMED.

---

## Session 10. Engines, history, team decisions and what is open

**Goal.** Tell the project's history in five sentences, explain why there are two engines and which one is the
default, who decided what in the strip-down, and list the open decisions with what each would change.

**Read.**
- [HISTORY.md](HISTORY.md): v0.7, "The two engines as described in v0.6.2", "Why our own engine".
- [extras/README.md](../extras/README.md) (what is optional and why) and `extras/docs/ENGINE_LOGIC.md` section 1 (the
  five regimes of the v0.6 engine: FREE, ACTIVE, ROLL, SLAV, ASLEEP).
- `extras/docs/design/at-limit-sleep/README.md` sections 1 and 2 (why the campaign was so slow, and how the fix was
  designed).
- [DECISIONS.md](DECISIONS.md) E1-E5, D15-D36 (skim: read only each "Decision" line), G1-G19, section H and the open
  decisions table.

**Run.**

```bash
git log --oneline
python -m dolle_extras verify
```

The first lists the project's commits, from the first simulator (2026-10-20) to the overview (2026-11-16). The second
runs the 19 checks of the v0.6 engine (about 5 minutes; it needs `pip install -e ".[extras]"` for numba, see
`extras/README.md`). It is optional.

**Try it yourself.** Pick three open decisions from DECISIONS.md (for example O1, O3 and O12) and, for each, write
down: what the team must decide, what each choice changes in the numbers, and the exact command or file to make the
change. Check your answers against the table. Then pick one superseded decision in section H and explain, in two
sentences, why it was replaced.

**Self-test.**

<details><summary>1. Tell the history in five sentences.</summary>

v0.1-v0.3 (2026-10-20/21): the simulator, the website, independent reviews that fixed the engine's bookkeeping, and
the provenance of every component. v0.4-v0.5 (2026-10-22/26): the whole flight on one clock, glass tubules, the
launch-load report, the LAMMPS and REBOUND checks, the flight to splashdown. v0.6 (2026-10-28): the full-contact
engine after the owner's "no shortcuts", with every step verified. The campaign of v0.6 runs was too slow, was fixed
(at-limit sleep, checks 14-19) and then stopped after the team's feedback (2026-11-10). v0.7 (2026-11-10/11): the
strip-down around the readable core, the notebooks and the 3-D page (HISTORY.md; DECISIONS.md section H).
</details>

<details><summary>2. Why are there two engines, and which is the default?</summary>

The v0.5 event-driven engine is cheap for free-flying clouds, at the price of rigid clumps, fragmentation rules and
frozen wall grains. The v0.6 full-contact engine removes those approximations (every contact rolls, slides and
twists) but is far slower. Since v0.7 the v0.5 engine is the default and the v0.6 engine is optional in `extras/`
(E2, E3, E5).
</details>

<details><summary>3. Who decided the strip-down, and how?</summary>

The Team reviewer asked on 2026-11-10 for code one can understand, with extra features stripped as far as possible
(G7). The owner then made each specific choice, agreed feature by feature with the coding sessions: the v0.6 engine
to `extras/`, tiny forces off, the cross-checks to `extras/`, the deletions, keeping the old site, a new 3-D page with
rigid clumps (E5, B21, G9, G12, E9).
</details>

<details><summary>4. Why was the v0.6 campaign stopped, and are there v0.6 results?</summary>

It was slow (static piles at their rolling limit were integrated at nanosecond steps), was fixed and restarted, and
then stopped for good after the team asked for understanding first. There are no v0.6 scenario results (G5, G6).
</details>

<details><summary>5. Was the campaign restarted before checks 14-19 passed?</summary>

Yes: on 2026-11-10 it was restarted at commit 726ec34 while the checks were still being written; the gate was on
using its results, not on running. After the checks and review fixes it was restarted from scratch at f133c3e (D35,
G6).
</details>

<details><summary>6. Name four open decisions and what each changes.</summary>

e_dry (thresholds x f(e_dry), O1); which silica and carbon (1.8x-36x, O2); the release mechanism, gas and
orientation (whether grains meet at mm/s at all, O3); the dust amount (impacts ~ N^2, O12). Others: JKR for silica
(O4), the rolling hysteresis (O5), re-running the stored results (O13), a licence (O18).
</details>

<details><summary>7. What naming rule does the public text follow?</summary>

Descriptive public text names no programme or institution, only "DOLL-E"; published sources may be cited by their
full reference (G1, owner's decision of 2026-10-21). The decision log uses titles only.
</details>

**Explain it out loud.**
- *"Why did you build a second engine and then set it aside?"* Key points: the owner wanted no shortcuts, so v0.6
  resolves every contact; it showed what v0.5 cannot (rolling along walls), but whole-flight runs took days to weeks;
  the team then asked for understanding first, so it moved to `extras/`, still runnable and checked (19 of 19).
- *"What is still open, and who decides?"* Key points: the team decides the hardware (release, gas, dust amount,
  materials, coating); the measurement decides e_dry and the adhesion; the owner decides process questions (re-run
  the stored results, history rewrite, licence). Each is in the open decisions table with the command to change it.

---

## How to change any decision

1. Find it in the index of [DECISIONS.md](DECISIONS.md) (section 2): the last column says where to change it.
2. Read the entry's "How to change it": the file, the function, the config key or the command.
3. Prefer a config override or a python `dataclasses.replace` to editing code (DECISIONS.md section 1).
4. Re-run what the table "What to re-run after a change" in DECISIONS.md section 1 lists, and update every document
   that quotes a number that changed.
5. Record the change as a new entry (or update the entry and add the old one to section H), then run
   `python -m pytest tests/test_decisions_doc.py -q`.

---

## Rehearsal (days 11-14)

- Day 11: give the 2-minute talk out loud three times, then the 10-minute talk once, with the OVERVIEW open.
- Day 12: answer every "Explain it out loud" question above without notes; check each against its key points.
- Day 13: redo every self-test; for each one you missed, re-read the decision log entry it cites.
- Day 14: a mock question round with a teammate who picks questions from OVERVIEW section 7 and from this guide.

## A 2-minute talk for officials

1. **The question (20 s).** DOLL-E asks below what impact speed ~100 um dust grains stick, and whether carbon and
   silica differ. It is a key input to how dust grows toward planets, and it cannot be measured on the ground because
   gravity swamps mm/s motions.
2. **The prediction (30 s).** From published contact physics (JKR adhesion), two 100 um silica grains stick below
   about 6 mm/s and graphite below about 38 mm/s. Of that factor of 6, a factor 1.5 rests on an assumed energy loss,
   which the flight will measure.
3. **The design finding (40 s).** Dust stays stuck to the tubule walls through launch (about a thousand times its
   weight) but can roll; a gentle shake frees nothing and a hard kick frees it far too fast. So the release mechanism
   is the key design question, and grains must start close together to meet at all.
4. **Why trust it (30 s).** Every component is checked against exact solutions (26 checks), the contact law agrees
   with LAMMPS to 2.2e-6, a readable core gives identical numbers, and every number is sourced or marked assumed.

## A 10-minute talk for officials

1. **The science question (1 min).** Dust growth in planet-forming discs; the sticking speed; why 100 um; why
   carbon against silica (OVERVIEW, 1-minute version).
2. **Why microgravity, and the experiment (1 min).** 1 g against 1e-4 g (0.49 mm fall in a second); glass tubules on
   a sounding rocket; cameras filming impacts in a 225 s window (notebooks 00 and 05).
3. **The physics of one impact (2 min).** JKR: surface energy against elasticity; snap-on, squash, neck, break;
   two losses (hysteresis and damping set by e_dry); momentum exact, energy lost; the sticking speed formula
   (notebooks 01 and 02).
4. **The results (2 min).** Silica 6.18 and graphite 38.36 mm/s; the x6.2 split (x4.07 materials, x1.52 assumed
   e_dry); the material choices are open (1.8x-36x); the comparison with the one published measurement (4.3x)
   shows why the experiment is needed (notebooks 03 and 00).
5. **The flight and the walls (2 min).** The loads phase by phase; pull-off ~1,000 g against loads up to 31 g;
   rolling at ~1.3 g under ~10 g spin; the release problem; collisions need a compact cloud (notebooks 04-06,
   the 3-D page).
6. **Why trust it (1 min).** Three layers of evidence; the generated checks table; what is not verified (CHECKS.md,
   OVERVIEW section 4).
7. **Decisions and what is open (1 min).** Own simulator instead of LAMMPS or REBOUND (time steps); JKR; assumed
   e_dry; head-on threshold; rigid clumps; strip-down for readability; open: e_dry, materials, release, dust amount
   (OVERVIEW section 5; DECISIONS.md open decisions).

Have ready, for questions: the 3-D page, `python examples/sticking_threshold.py` (runs in seconds) and the index of
DECISIONS.md.

---

## Glossary

| term | meaning |
|---|---|
| adhesion hysteresis, U0 | the energy a contact keeps because it forms at zero overlap but breaks only after the neck has been stretched; 3.833e-15 J for two 50 um silica grains |
| aggregate, clump | grains stuck together; in the default engine and the 3-D core a rigid body |
| apogee | the highest point of the flight, T+197.4 s at 156 km in the model (notebook 05) |
| ASSUMED | a value or choice with no published source or check behind it, stated openly |
| bisection | halving a speed interval until the stick/bounce boundary is found to 0.2% (D2) |
| borosilicate | the tubule glass in the examples (data-sheet values, C6) |
| break overlap, delta_c | the (negative) overlap at which a stretched contact breaks: -0.696 nm for two silica grains |
| check, `check:<id>` | an automated comparison with an exact solution, conservation law, limit or published data, with a criterion and a measured value (CHECKS.md) |
| Coriolis, centrifugal, Euler forces | the apparent forces in the frame spinning with the payload (A5) |
| contact charging | charge exchanged at contact, driven by the difference of work functions (B19) |
| Cunningham slip correction | the reduction of gas drag when the gas mean free path is not small against the grain (B18) |
| DEM | discrete element method: simulating each grain as a body with contact forces |
| de-spin (yo-yo) | weights on cables that slow the payload's spin from 4.8 Hz to the residual spin at T+54 s |
| DMT | the Derjaguin-Muller-Toporov adhesion model for stiff, weakly sticky contacts: no neck, no hysteresis |
| dry restitution, e_dry | the restitution a non-sticky impact of the material would have; sets the damping; ASSUMED (C3) |
| damping factor f(e_dry) | damped over undamped sticking speed; depends on e_dry alone (1.61 at 0.85, 2.46 at 0.70) |
| effective mass, m_eff | 1/(n.K.n): the mass the contact point of a rigid body offers along the normal (notebook 06 section 4) |
| energy guard | replacing a friction impulse that would create energy by the frictionless one (B13) |
| erosion, shattering | a clump losing the struck grain, or breaking into single grains (B14) |
| E_break, E_roll | the energy to break a bond and to roll one grain over another by the critical displacement |
| event-driven | advancing to the exact time of the next collision instead of testing for overlaps every step (D5) |
| extras | the optional folder with the v0.6 engine and the cross-checks (`extras/README.md`) |
| flight demo | the example config `configs/flight_carbon_vs_silica.json`: two 3 cm chambers, 2,000 grains each |
| g-jitter | small random accelerations of the payload; 5e-5 g rms in the demo, ASSUMED (F21) |
| island | in the v0.6 engine, a sleeping cluster of grains |
| JKR | the Johnson-Kendall-Roberts (1971) theory of an elastic contact with surface energy: the contact law of every impact (B1) |
| Knudsen number | gas mean free path over grain size; large (132 in the demo) means free-molecular gas |
| LAMMPS | an established molecular and granular dynamics code, used as the outside referee of the contact law (D10) |
| launch-load report | `python -m dolle_sim mission ... --loads-only`: loads against wall limits per phase (F10) |
| Maugis-Dugdale | an adhesion model spanning JKR and DMT; needs the range of the surface forces |
| mean free path | the average distance a grain flies before hitting another (1.96 m for 100 grains in a tubule) |
| microgravity window | T+88-313 s, above ~100 km, where impacts are counted (F5) |
| mission run | all four tubules simulated through the whole flight on one clock (F1) |
| mixing rule | how the properties of two different materials combine for a contact (B7) |
| Owner, Team, Team reviewer, Engineering, Review | the "decided by" words of the decision log (DECISIONS.md section 1) |
| Poisson ratio, nu | lateral contraction over axial strain of a material |
| preflight | `python -m dolle_sim preflight`: fast design checks and the force budget |
| provenance | where each component comes from: published, standard, derived or assumed (PROVENANCE.md) |
| pull-off force, F_c | the largest tension a JKR contact holds, 1.5 pi w R* |
| REBOUND | an N-body code for planets and rings, used only as an outside check of trajectories (D11) |
| reduced modulus, E* | 1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2 |
| reduced radius, R* | R1 R2/(R1 + R2); a flat wall has infinite radius, so R* is the grain's radius |
| reduced mass, m* | m1 m2/(m1 + m2); the mass of the relative motion |
| residual acceleration | the small steady acceleration left in the microgravity window, 1e-4 g ASSUMED (F6) |
| restitution, e | rebound speed over approach speed of a bounce (0 for a stick) |
| restructuring | grains rolling inside a clump, changing its shape; not modelled (B17) |
| rolling hysteresis, h | the factor in the critical rolling displacement xi_crit = h a0/12; 0.5 ASSUMED (C10) |
| scenario | a named change to the mission config, combined with "+" (F14) |
| silanised, fluorosilane glass | glass with a coating that lowers adhesion (C6, C7) |
| sleep certificate | in the v0.6 engine, a proof that a resting grain or cluster cannot change, so it need not be integrated (D18) |
| sticking speed, threshold | the impact speed below which two grains stick |
| sub-steps | the time steps inside one contact (300 per contact duration, D1) |
| surface energy, gamma | energy per unit area of a free surface (J/m^2); two like surfaces hold with w = 2 gamma |
| Tabor parameter | ratio of the elastic neck height to the range of surface forces: large means JKR, small means DMT |
| thermophoresis | the push on a grain by a gas temperature gradient (B23) |
| Thornton-Ning | the model of plastic yield in an impact used here (Thornton & Ning 1998, B4) |
| tubule | one of the small glass dust chambers, 12.7 x 12.7 x 38.1 mm inside in the example deck (F2) |
| velocity Verlet | the second-order time integrator (half kick, drift, half kick) used for every contact and flight |
| work of adhesion, w | the energy to separate unit area of contact; 2 sqrt(gamma1 gamma2) for unlike pairs |
| Young's modulus, E | the stiffness of a material (GPa) |
| v0.5, v0.6, v0.7 | the event-driven engine (default), the full-contact engine (optional, `extras/`), the strip-down release |

---

## The numbers worth knowing by heart

| number | what it is | where it comes from |
|---|---|---|
| 6.18 mm/s | silica-silica sticking speed, 50 um radius | `python examples/sticking_threshold.py` |
| 38.36 mm/s | graphite-graphite sticking speed | same |
| 6.66 / 18.23 mm/s | silica / graphite on the borosilicate wall | same |
| 3.83 / 15.58 mm/s | the same pairs without damping | same |
| x6.2 = x4.07 x x1.52 | graphite/silica ratio = material values x assumed e_dry | `python examples/silica_vs_carbon.py` |
| 2.5, 1.7, 0.95 | the ratio's surface-energy, stiffness and density parts | same |
| 1.61 / 2.46 | damping factor f(e_dry) at 0.85 (silica) and 0.70 (graphite) | `python examples/sticking_threshold.py` |
| 0.85 / 0.70 | e_dry of silica / graphite, ASSUMED | `dolle_core/materials.py` |
| 0.025 / 0.075 J/m^2 | surface energy of silica / graphite (published) | `dolle_core/materials.py` (notebook 03 section 1) |
| 54 / 10 GPa | Young's modulus of silica / graphite | same |
| 1.6394 | prefactor of the hand formula for equal spheres | `python examples/sticking_threshold.py` |
| 1.44 / 8.76 | Tabor parameter of silica / graphite pairs | session 2 snippet; notebook 03 section 2 |
| 0.756 | restitution of silica at 20 mm/s | `python examples/collide_two_grains.py` |
| 1e-15 | relative momentum error of one impact | notebook 02 cell 5 (1.5e-15) |
| 300 | sub-steps per contact (3.8e-4 from converged) | notebook 01 cell 32 |
| 1,034 / 827 / 1.29 g | silica on glass: pull-off, sliding, rolling | `python examples/grain_on_wall.py` |
| 1,628 / 3.59 g | graphite on glass: pull-off, rolling | same |
| 25 g, 4.8 Hz | burn load, spin at burnout | notebook 05 cell 11 (`burn_g`, `spin_burnout_hz`) |
| 9.56 g | largest centrifugal load (T1's far corner) | notebook 05 cell 8 |
| T+88-313 s, 225 s | the microgravity window | notebook 05 cell 4 |
| 1e-4 g, 1 rpm | residual acceleration and residual spin, both ASSUMED | notebook 05 cell 11 |
| 7.93 g, 5 g, 31.2 g | re-entry peak, parachute, splashdown | notebook 05 cells 4 and 8 |
| T+882 s / T+887 s | splashdown / end of a mission run | notebook 05 cells 4 and 11 |
| 0 of 3,200 | grains freed by the nominal 50 g shaker | MISSION_RUNS.md (v0.5 engine, 2026-10-26) |
| 1.56 m/s, x253 | wall speed of a 5,000 g / 5 kHz kick, against silica's sticking speed | notebook 04 cell 33 |
| 20 / 4 and 17 / 17 | 3-D page: silica and graphite impacts / sticks | session 8 snippet; notebook 06 cell 28 |
| 1.96 m | mean free path of 100 grains in a tubule | session 1 snippet |
| 26 / 19 | model checks / v0.6 engine checks, all passing | `python -m dolle_sim verify`; CHECKS.md |
| 2.2e-6, 0.14% | LAMMPS agreement: contact law term by term; pull-off | CHECKS.md |
| 3.2e-10 m | REBOUND trajectory agreement | CHECKS.md |
| 2.2% | the largest switched-off force against the residual (radiation pressure, graphite) | `python -m dolle_sim preflight configs/flight_carbon_vs_silica.json` |
| 0.276 vs ~1.2 m/s | model vs measured (Poppe et al. 2000) for 1.2 um silica on a flat target | notebook 00 cell 4 |
| 1.8x-36x | how much more easily carbon sticks than the table silica, by carbon type | notebook 03 cell 32 |

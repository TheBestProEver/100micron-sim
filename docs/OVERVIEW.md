# DOLL-E simulator: the overview (read this first; ~15 minutes)

Everything below is traceable: each number comes from the code in this repository (the file or notebook is named),
and each assumption is marked **ASSUMED**. Grain radius 50 um (diameter 100 um) unless stated.

## The 1-minute version

DOLL-E asks one question: **below what impact speed do ~100 um dust grains stick together, and is it different
for carbon and silicate (silica) dust?** That speed is a key input to how dust grows toward planets around young
stars, and it is hard to measure on the ground at mm/s speeds because gravity dominates such slow motions. So DOLL-E
flies glass tubules of dust on a sounding rocket and films grains colliding during the few minutes of
microgravity.

The simulator answers the questions before flight: what sticking speeds to expect (silica ~6 mm/s, graphite ~38
mm/s), whether the dust can be freed from the tubule walls and made to collide slowly enough, and whether the
cameras can measure it. It is built from published contact physics (JKR adhesion theory), every piece is checked
against exact solutions and against established codes (LAMMPS, REBOUND), and a small readable version of the core
physics (`dolle_core`, with notebooks) lets anyone follow every step.

## 1. The physics in five steps (notebooks 01 and 02)

1. **Two grains touch and snap together.** Surfaces attract (surface energy, gamma, in J/m^2). The moment two grains
   touch, adhesion pulls them into a small contact patch. This is the JKR theory (Johnson, Kendall & Roberts 1971):
   elastic squashing (stiffness from Young's modulus E) balanced against surface energy.
2. **They squash and push back.** The incoming motion compresses the contact (a few nanometres), elastic energy is
   stored, then released as they push apart.
3. **Energy is lost** in two ways: the adhesion hysteresis (the contact snaps on at zero overlap but only breaks
   after a neck has been stretched: that work is never returned) and internal damping (calibrated so that a
   non-sticky impact would rebound with the material's "dry restitution" e_dry, **ASSUMED** values).
4. **Stick or bounce.** If the energy left after the losses cannot break the neck, the grains stay stuck; if it
   can, they bounce. The speed where this flips is the **sticking speed** (threshold).
5. **Momentum is always conserved, energy is not.** Every contact force is applied to both grains as an equal and
   opposite pair (Newton's third law, written literally in the code), so total momentum is constant to round-off
   (1e-15 relative). Kinetic energy is lost to the two mechanisms above: that loss is what makes sticking possible.

Example (`python examples/collide_two_grains.py`): two silica grains at 4 mm/s **stick**; at 20 mm/s they
**bounce**, keeping 76% of their speed (restitution e = 0.756).

## 2. The key results

**Sticking speeds** (`python examples/sticking_threshold.py`):

| pair | sticks below | without damping |
|---|---|---|
| silica-silica | **6.18 mm/s** | 3.83 mm/s |
| graphite-graphite | **38.3 mm/s** | 15.6 mm/s |
| silica on the glass wall | 6.66 mm/s | 3.84 mm/s |
| graphite on the glass wall | 18.2 mm/s | 8.48 mm/s |

**A formula you can check by hand** (no damping): v0 = 1.6394 gamma^(5/6) E*^(-1/3) R*^(-5/6) rho^(-1/2)
(R* the reduced radius, E* the reduced modulus, rho the density). Damping multiplies it by a factor that depends only
on e_dry: 1.61 for silica's e_dry = 0.85, 2.46 for graphite's 0.70.

**Why graphite sticks ~6x more easily than silica** (notebook 03, `python examples/silica_vs_carbon.py`):
x6.2 = x4.07 from the material values (higher surface energy x2.5, softer x1.7, density x0.95) times x1.52 from the
**ASSUMED** energy loss (e_dry). So about a third of the difference rests on a number nobody has measured for these
grains: exactly the kind of thing the flight measures.

**Which carbon is not decided** (neither is which silica). Depending on the carbon and on literature values, carbon
sticks between **1.8x and 36x** more easily than silica (graphite 22-220 mm/s, glassy carbon 20-127, amorphous
carbon 11-153; ranges mostly **ASSUMED**). The notebooks show how to recompute for any choice.

**Dust on the walls** (notebook 04, `python examples/grain_on_wall.py`): a silica grain on the glass wall needs
**~1,000 g to pull off** (graphite ~1,600 g) but **starts rolling at ~1.3 g** (graphite ~3.6 g). Launch loads are
~25 g along the rocket axis and ~10 g sideways from the spin: **dust stays on the walls through launch (it cannot be
pulled off) but can roll along them**. Freeing dust for the experiment is therefore a design problem (the release
mechanism, still open): a gentle shake frees nothing, a hard kick frees dust but far too fast (m/s, while sticking
happens at mm/s).

**Collisions are rare unless the dust starts close together** (3-D page, notebook 06): 100 grains spread through a
12.7 x 12.7 x 38.1 mm tubule essentially never meet (the mean free path is ~2 m; the walls are ~6 mm away), so the
walls catch every grain within seconds. From a compact released cloud (**ASSUMED** start, speeds 15 +/- 7.5 mm/s)
silica mostly bounces (20 impacts, 4 sticks) and graphite always sticks (17 of 17) and builds clumps:
https://thebestproever.github.io/100micron-sim/sim3d/

## 3. How the software is organised

| part | what it is | size |
|---|---|---|
| `dolle_core/` | the readable core: one collision, materials, wall limits, a 3-D simulation with rigid clumps; plain numpy; identical results to the full code | ~1,300 lines of physics |
| `notebooks/00-06` | the explanations: derivations, the reason for each choice, every number printed by code; run locally or in Colab | 7 notebooks |
| `examples/` | the same calculations as plain Python scripts | 4 scripts |
| `dolle_sim/` | the flight simulator (v0.5 event-driven engine: many grains over a whole flight, flight loads, cameras) | ~6,700 lines |
| `extras/` | the v0.6 full-contact engine (every contact rolls, slides and twists) and the LAMMPS/REBOUND cross-checks; runnable, set aside | |
| `docs/sim3d/` | the 3-D page: replay, one impact up close, live sandbox in the browser | |

## 4. How we know it is right

* **Exact solutions and conservation laws**: 26 model checks (energy conserved with no damping, restitution
  equals the calibrated value, plastic restitution against Thornton & Ning 1998, momentum through sticking, no missed
  collisions...), all passing: [CHECKS.md](../CHECKS.md).
* **Established codes**: the same contact law in LAMMPS agrees term by term to 2.2e-6; flight trajectories against
  REBOUND to 3e-10 m.
* **The readable core equals the full code**: identical sticking speeds, restitution within 6e-16
  (`tests/test_dolle_core.py`).
* **Tests**: 195 in the main suite and 82 for the extras, all passing.

## 5. The decisions that matter most

| decision | why | where to change it |
|---|---|---|
| Write our own simulator instead of running LAMMPS or REBOUND | LAMMPS has the right contact physics but one global ~10 ns time step: 13-49 core-days per tubule for a whole flight; softening contacts to go faster breaks the pull-off physics. REBOUND has no walls, spin or sticking. So: our own engine, with LAMMPS's contact law term by term, cross-checked against both | - |
| JKR adhesion theory | the standard model for soft, sticky grains of this size; for silica it is near the edge of its validity (between JKR and the DMT model), noted as a limitation | `dolle_core/jkr.py` |
| Damping calibrated to an assumed e_dry | the energy lost inside the material is not measured for these grains; it changes the answer by x1.5-2.5; flagged as the largest uncertainty | `dolle_core/materials.py` (e_dry) |
| A head-on impact is enough for the threshold | the full 3-D contact model gives the same sticking speed to 0.16% (check `dem_vs_1d`) | - |
| 100 grains per tubule in simulations | the team has not decided the dust amount; 800 made runs take weeks; impact counts scale with grains squared | mission config |
| Stuck grains become rigid clumps (3-D core, v0.5) | simple, exact momentum and spin bookkeeping; the cost: clumps cannot roll or bend at their contacts (the v0.6 engine in extras can) | `dolle_core/clumps.py` |
| Strip-down (v0.7) | the team asked for something one can understand: the v0.6 engine and the cross-checks moved to `extras/`; forces 1000x too small to matter switched off by default (sizes listed in the README) | README, `extras/` |
| Materials | carbon and silica types not decided: graphite and the table silica as placeholders, with ranges | `dolle_core/materials.py` |

## 6. What is still open

* The energy loss per impact (e_dry) is **ASSUMED**: the biggest single uncertainty in the sticking speeds.
* Which silica and which carbon will fly.
* How to release the dust slowly enough (the release mechanism) and at what density, so that grains collide at
  mm/s-cm/s inside the camera's view.
* JKR's accuracy for silica (it sits between the JKR and DMT limits).

## 7. Questions you will be asked, with short answers

* **"How do you know the simulation is right?"** Three layers: exact solutions and conservation laws (26 checks),
  agreement with established codes (LAMMPS to 2e-6, REBOUND to 3e-10 m), and a small readable version that gives
  identical answers, so the physics can be inspected line by line.
* **"Why does carbon stick more easily?"** Higher surface energy (stickier surfaces) and a softer material (larger
  contact for the same impact) account for x4; the assumed energy loss accounts for the rest.
* **"What is the biggest uncertainty?"** The energy each impact loses inside the material, and the choice of
  materials: precisely what the flight data will pin down.
* **"Why not use existing software?"** Speed: LAMMPS would need weeks of computing per tubule; making it faster
  breaks the physics we care about. Our engine uses LAMMPS's contact law and is checked against it.
* **"Can the experiment actually measure the threshold?"** Only if the dust is released gently (mm/s) and close
  together: dust stays stuck to the walls under launch loads, a hard release throws it far too fast, and spread-out
  grains rarely meet. The release mechanism is the main design question the simulator is helping to answer.
* **"Is momentum conserved?"** Exactly (to 1e-15): every contact force acts on both grains, equal and opposite.
  Energy is not, and that loss is what lets grains stick (notebook 02).

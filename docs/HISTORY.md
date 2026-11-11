# History of the simulator

What changed between versions and why, with the text that described each version at the time. The current state is
in the [README](../README.md); this page keeps the record.

## v0.7 (10 Nov 2026): strip-down around the readable core

The team asked for code they can read and understand, with the extra features stripped out. v0.7 keeps the physics and
the values the same and changes what a reader meets first:

- **The readable core and the notebooks come first.** `dolle_core/` (plain numpy, the same contact physics and
  material values as the simulator, checked case by case by `tests/test_dolle_core.py`) and `notebooks/00-06`
  (derivations, the reason for every modelling choice, Colab badges) are the way in. The new 3-D page
  (https://thebestproever.github.io/100micron-sim/sim3d/) shows rigid clumps forming, built from them.
- **The v0.6 full-contact engine moved to `extras/`** (`extras/dolle_extras/`: engine, dem, statics, slaved, roll1,
  loads, vibration, verify_full; 7,815 lines), together with the REBOUND and LAMMPS cross-checks, the image-based
  grain sizing, the campaign tools (`run_campaign.py`, `mission_table.py`), the LAMMPS scripts, the atmosphere
  generator, the engine's decision map and design record, and their tests. All of it still runs: `python -m
  dolle_extras verify | mission | rebound-check | lammps-check | sizes-image | sources` and `python -m pytest
  extras/tests` ([extras/README.md](../extras/README.md)). `dolle_sim` went from 15,489 to 6,711 lines.
- **The default engine is the v0.5 event-driven engine again** (`dolle_sim/simulation.py`: stuck grains form rigid
  clumps). `python -m dolle_sim mission` used the v0.6 engine before; now `python -m dolle_extras mission` is the
  only way to run it (`dolle_sim` never imports the extras; a config with `"engine": "full"` given to `python -m
  dolle_sim mission` stops with a message naming that command, and so do the moved commands below).
- **The negligible forces are switched off by default**: mutual gravity, the Lorentz force and the external E field,
  photophoresis, radiation pressure, Brownian motion and the gravity gradient. The code stays; each is a switch
  (`forces.<key>`, `environment.gravity_gradient`). For the flight demo config each is at most 2.2% of the 1e-4 g
  residual acceleration (radiation pressure on graphite), and the Brownian speed (~2 µm/s) is 3e-4 of silica's
  sticking speed. `python -m dolle_sim preflight` still prints every term, marks the switched-off ones "(off)" and
  warns if one exceeds 5% of the residual acceleration or the Brownian speed exceeds 1% of the sticking speed (both
  thresholds ASSUMED). The 26 model checks give the same values as before, digit for digit (they set their forces
  explicitly). Stored results (the flight demo, the mission scenarios) were computed with the forces on and were not
  re-run.
- **`python -m dolle_sim verify` runs the 26 model checks**; the 19 checks of the v0.6 engine are
  `python -m dolle_extras verify`. CHECKS.md keeps both parts. The v0.6 engine checks set the negligible forces on
  explicitly (`verify_full.FORCES_V062`), as they were written; their values are identical to v0.6.2's.
- **Bonds are stored (v0.5 engine and the 3-D core alike).** A grain that sticks to a spinning clump is merged
  where the straight-line contact search put it, up to 1.25e-3 of its distance from the clump's centre of mass
  from touching (0.1-2 nm in the 3-D page's graphite trimers). The old distance test (1e-6 of the radius sum)
  missed such bonds, so an intact trimer could split without an impact and erosion could cost too few bonds. Each
  clump now keeps the pairs that stuck; the 3-D page's replay, the 26 checks and the core-vs-v0.5 comparisons are
  unchanged by it. Stored v0.5 results (flight demo, mission scenarios) predate the fix.
- **The separation energy after a break-up is a stated assumption** (`contact.fragment_separation`,
  `Tubule.separation`; default 1 as before, 0 the opposite bound): notebook 06, sections 7 and 9.
- **Deleted:** the colliding-clouds demo (its config, the replay `docs/replays/clouds.html` and two figures; it
  duplicated the flight demo) and `docs/data/verification_desktop.json` (read by nothing). The record of its result
  (README at commit 90a0058, v0.6.2, 60 s, two opposed dust clouds per chamber): true thresholds 38.2 [38.1–38.4] /
  6.26 [6.19–6.33] mm/s (graphite / silica), stereo 37.6 [37.0–38.2] / 6.33 [6.17–6.56] mm/s, one camera 24.6 /
  5.62 mm/s (1.6× / 1.1× too low, against 4.2× / 3.5× in the flight run's isotropic launch). `git show
  90a0058:configs/colliding_clouds_carbon_vs_silica.json` recovers the config.
- **The README was shortened** to the core, the notebooks and the 3-D page. Its other sections moved to
  [docs/USER_GUIDE.md](USER_GUIDE.md) (using the simulator, the physics, the results, the checks), to
  [docs/MISSION_RUNS.md](MISSION_RUNS.md) (the whole-flight runs) and here. Its comparison with Poppe, Blum & Henning
  (2000) was corrected: their 1.2 µm spheres hit a flat target, for which the model gives 0.27-0.28 m/s (4.3-4.5
  times below the measured ~1.2 m/s), not two free grains (0.25 m/s, 4.9 times); see notebook 00.

## The two engines as described in v0.6.2

The simulator was written from scratch for DOLL-E and had two engines (since v0.7 the first is optional, in
`extras/`, and the second is the default):

* **The full-contact engine (v0.6; every mission run).** Every grain is a sphere with six degrees of freedom and
  every grain–grain and grain–wall contact is resolved with the full adhesive contact law at full stiffness:
  JKR normal force with hysteresis, damping and plastic flattening, plus sliding, rolling and twisting springs
  with their limits (Wada et al. 2007; Dominik & Tielens 1995, 1997), written in the form of the LAMMPS GRANULAR
  package and identical to it term by term (differences ≤ 2.2×10⁻⁶ in 7 set-ups). There are no rigid
  aggregates, no fragmentation rules and no frozen wall grains: stuck grains are ordinary contacts that can
  roll, slide, twist and break. A full-stiffness run of the whole flight would take ~10⁴ core-days, so the
  engine uses the full model wherever something can happen and **exact or verified** treatments elsewhere:
  exact free flight up to the earliest time a grain could come within 0.1 µm of anything (and again after each
  contact, up to the next such time, so a bouncing grain is in the full model only near its bounces); resting grains and
  wall-anchored clusters held static while a certificate shows every contact stays inside its elastic limits;
  grains rolling on one wall, and stuck clusters rolling along a wall, on slaved models (the stiff contact modes held quasi-static, rolling and twisting integrated with the full springs) that match the full model to ≤ 3×10⁻⁵ of the rolled path; clusters that are still rocking sleep with that residual motion kept as exact linear response and added back when they wake. **Every step is checked
  after it is taken** and redone in the full model if any assumption one grain's treatment made about another
  failed; as a last safety net, a grain that reaches the full model overlapping something it has no contact with
  (which no bound allows) also makes the step be redone. See [CHECKS.md](../CHECKS.md), Part 2.
  **How it decides:** [extras/docs/ENGINE_LOGIC.md](../extras/docs/ENGINE_LOGIC.md) lists every decision the engine makes (which
  regime a grain is in, when it sleeps, wakes, rolls or enters the full model, how each step is verified), with the
  test as the code applies it, the function, the reason, whether it is exact or bounded, and the check that verifies
  it. The design of each newer part, stage by stage, is in [extras/docs/design/](../extras/docs/design/at-limit-sleep/README.md).
* **The event-driven engine (v0.5; long free-flying cloud runs, `run`, and the browser app).** Grains and rigid
  aggregates fly under the external forces in millisecond steps (velocity Verlet with exact drag and Coriolis;
  Dullweber–Leimkuhler–McLachlan rotation). Impacts are found by continuous collision detection, processed in
  time order, and each is resolved by integrating the same JKR + damping + plasticity law in nanosecond steps;
  stuck grains form rigid aggregates, which fragment by the Dominik & Tielens (1997) energy thresholds. This
  is what makes 15-minute runs of thousands of free-flying grains cheap; its two approximations (rigid
  aggregates, fragmentation rules) are the ones the full-contact engine removes.

## Why our own engine, and not REBOUND or LAMMPS as the base? (v0.6.2)

We checked both against their source code and documentation (the full notes are in PROVENANCE.md and
`extras/tools/lammps_benchmark.py`). What decides it is the time scales. A sticking contact between two 100 µm grains
lasts about a microsecond and must be followed in ~10 ns steps, but the flight lasts 15 minutes.

* **REBOUND** (v5.2) is an N-body code for planets and rings. It has no walls (its boundaries are open,
  periodic or shear), no grain spin or moment of inertia, and no aggregates (it can only merge stuck grains
  into one sphere). It also has no gas drag, no electrostatics and no adhesive contact. Its collision hook runs
  after a full step without the contact time. Building on it would mean writing all of DOLL-E's physics
  anyway, and it is GPL-3.0. We use it as an outside check of our trajectories and collision bookkeeping.
* **LAMMPS** (GRANULAR package) is the closest established code. It has real JKR adhesion with hysteresis,
  friction, rolling resistance, JKR walls that can vibrate, breakable bonds and rotating-frame forces through
  its standard fixes. But it moves every grain with one global time step. Measured on this laptop,
  `extras/tools/lammps_benchmark.py` runs 2.3×10⁷ grain-steps per second per core with the grains in flight (no
  contacts) and 6.1×10⁶ with the settled layer on the bottom (every grain in a JKR wall contact). At the
  ~10 ns step real glass and silica need, one tubule from lift-off to power-off takes **13–49 core-days**
  (35–130 to splashdown). DOLL-E takes 3 minutes to 10 hours for all four tubules from lift-off to
  splashdown on 4 cores (the 10 hours are the 500 g shaker, whose fast walls force short steps).
* **Softening the grains** is the usual DEM shortcut, but it does not work here. The JKR sticking speed scales
  as E^-1/3, so a 1000× softer grain sticks 10× faster. The standard correction, lowering the surface energy
  by (E_soft/E)^(2/5) (Hærvig et al. 2017; Chen et al. 2019; Washino et al. 2024), keeps the sticking
  speed but cuts the force needed to pull a grain off a wall 16×. That force is what decides launch and
  release.
* LAMMPS also lacks sticking grains that merge into rigid aggregates and contact charging (new C++ code).
  Free-molecular drag can be scripted with its variables but would need validating. A plugin could replace the
  integrator, but every existing granular pair style and fix assumes the one global step.

So DOLL-E does not move every grain with one global step. The full-contact engine (v0.6) runs LAMMPS's own
contact law at full stiffness, but only on the grains that can interact at that moment; everything else is
either flying exactly, resting under a certificate, or rolling on a slaved model, and each step is verified
afterwards. The event-driven engine (v0.5) is cheaper still for free-flying clouds, at the price of rigid
aggregates and fragmentation rules. LAMMPS is the independent referee of the contact law
(`extras/tools/lammps_contact_check.py`, `python -m dolle_extras lammps-check`).

## Commands that moved in v0.7

| before (v0.6.2) | since v0.7 |
|---|---|
| `python -m dolle_sim rebound-check CFG --duration 60 --max-grains 600` | `python -m dolle_extras rebound-check CFG --duration 60 --max-grains 600` |
| `python -m dolle_sim lammps-check` | `python -m dolle_extras lammps-check` |
| `python -m dolle_sim sizes --image micrograph.png --um-per-px 1.25` | `python -m dolle_extras sizes-image micrograph.png --um-per-px 1.25` |
| `python -m dolle_sim verify` (45 checks: 26 model + 19 engine) | `python -m dolle_sim verify` (26) and `python -m dolle_extras verify` (19) |
| `python -m dolle_sim mission ... --checkpoint-at / --resume-from / --resume-at / --autosave-min` | `python -m dolle_extras mission ...` with the same options |
| `python tools/lammps_contact_check.py`, `tools/run_campaign.py`, `tools/mission_table.py`, `tools/lammps_benchmark.py`, `tools/make_atmosphere.py` | `python extras/tools/<the same name>` |

(Without installing the repository, write `python extras/dolle_extras` for `python -m dolle_extras`.)

## Code reviews (v0.2 to v0.5)

The code was put through automated adversarial reviews by AI reviewers, one
per subsystem, with every finding re-checked by a second AI reviewer before it
was fixed. v0.3 adds a provenance audit of every component and of the public
text (68 confirmed findings), which is the basis of PROVENANCE.md.

The v0.5 additions (re-entry to splashdown, the atmosphere table, the
wall-bound shortcuts, the REBOUND/LAMMPS comparison) were reviewed the same
way: 23 findings from independent reviewers (about a dozen distinct issues),
all fixed. The main ones:

* **The fall attitude started at apogee.** The side-on fall attitude began at
  T+197 s, inside the microgravity window, so the 10⁻⁴ g residual turned
  sideways instead of reversing along the axis. Nothing turns the payload in
  vacuum: the attitude now changes at T+313 s, and every scenario was re-run.
* **The re-entry peak was presented as validated.** The Allen & Eggers check
  tests the integrator with gravity off; it does not validate the 7.9 g peak.
  Point-mass models of this kind read 5–20% low against measured flights, so
  the README (now docs/MISSION_RUNS.md) says ~8–11 g and quotes the measured check values instead of
  "< 10⁻⁶".
* **Shock directions.** The cited flight log shows the splash spike mostly
  sideways. The direction is now an option (`shock_oblique` scenario) and the
  default is marked assumed.
* Smaller: the tumbling option rotated only the load direction (now stated,
  and it starts at T+313 s); `np.trapezoid` needs NumPy 2; the summary JSON
  contained `Infinity`; end-of-run counts were labelled "at power-off"; the
  descent forced a neighbour-list rebuild on almost every step.

The v0.4 additions (flight timeline, launch report, release, glass, sizes,
LAMMPS check, web Flight tab) were reviewed in 7 areas. Each finding was
checked by two skeptics told to refute it: 41 findings, 3 refuted, the rest
fixed. The main ones:

* **The residual acceleration never reversed.** After apogee drag acts the
  other way, but the model kept a 10⁻⁴ g pull toward the nose for the whole
  descent, and it counted impacts down to T+332 s although the payload is
  below 100 km from ~T+313 s. Drag now follows the velocity, and the
  microgravity window ends at T+313 s (a separate descent phase follows).
* **The vibration "upper bound" was not one.** It used the 25 g thrust as
  holding the dust down and ignored the coast and the ground vibration test.
  It is now an estimate with the thrust ignored, covering lift-off to burnout
  plus the ground random and sine tests.
* Engine: fragments shaken off a wall-bound aggregate left along the bond
  instead of with the wall; settled beds of two dusts could overlap; a
  negative spin rate rotated the frame the wrong way; `wall_sticking: never`
  walls held grains that came to rest under a load; release windows were
  skipped in runs without a flight profile, and the "do not step over a
  window" branch could never run.
* Reports and docs: "released" counted events, not grains; the camera's
  field of view was transposed and did not follow the tubule orientation;
  sliding ignored the pull at the same instant; the LAMMPS pull-off "0.5%"
  was the detection grid (now 0.14%, checked 100× finer), and LAMMPS does
  allow some damping laws with JKR, just not DOLL-E's (Tsuji).
* Image sizing cut isolated elongated grains (flakes) into pieces. It now
  seeds the watershed with extended maxima and is tested on rods, ellipses,
  irregular flakes and unequal touching pairs.

Problems found and fixed in v0.3:

* **Fragmentation could create energy.** The separation energy was computed
  from the full translational collision energy, so oblique or eccentric
  impacts gave fragments more energy than the bounce dissipated (up to ~1/3 of
  the impact energy). It now uses the energy the impulse actually removed,
  E_diss = P·v_rel − ½PᵀKP, solves the separation kick exactly, and breaks
  bonds only when E_diss can pay for them. Eroding a "bridge" grain could
  leave two grains bonded without touching.
* **The centrifugal torque on aggregates was counted twice** (once explicitly
  and once inside the rotating-frame treatment of spin). Spins also used to
  ignore the rotation of the payload frame.
* Collision prediction for fast-spinning aggregates assumed straight-line
  motion over the whole step. The neighbour-list skin ignored speed gained from
  vibrating walls, and a bounce could make a grain faster than the list
  allowed (now adaptive, with an immediate rebuild). Friction impulses on
  eccentric aggregate impacts could create energy (now replaced by the
  frictionless impulse).
* Pair forces were held for 5 steps (20 ms) in the example configs; they are
  now recomputed every step.
* Smaller fixes: image-charge factor inconsistent between the short- and
  long-range parts, g-jitter starting at zero, double flux weighting in the
  pre-flight wall-capture estimate, polarisation budget ×2, a crash on
  `yield_pressure: null`, a mislabelled "Chokshi" estimate (it is a simple
  energy estimate), the ice density, and graphite's Poisson ratio (now Chokshi
  et al.'s 0.32).
* Many documentation claims were corrected: what is published versus assumed,
  what REBOUND does and does not check, how much the assumed damping drives the
  thresholds, the published spread of silica and graphite surface energies,
  and which components have no automated check.

Problems found and fixed in v0.2:

* Impacts onto wall-bound grains (twice the effective mass, so they stick at
  √2 lower speed) were mixed into the grain–grain statistics. This biased the
  "true" thresholds ~7–15% low. They are now flagged and kept separate.
* Vibrating walls could let grains through (wrong exclusion of grazing contacts).
* Aggregates were fragmented using the grain–*wall* bond energy, so they
  shattered on almost every wall hit. Fragments got no separation velocity and
  re-stuck immediately, producing spurious zero-speed "sticks".
* Post-yield damping acted on a contact that no longer existed, which made
  fast plastic impacts far too inelastic.
* The batched event loop could miss follow-on collisions; the neighbour-list
  rebuild test was too lax.
* Free rotation of aggregates pumped energy (stale angular velocity). Coriolis
  was only first order. The aggregate impulse ignored off-diagonal collision-matrix terms.
* No long-range space-charge field; photophoresis prefactor a factor 2 low;
  inertially fixed vectors not rotated in the spinning frame; aggregate spin-down 1.5× too fast.
* The camera used contact-point velocities (including invisible grain spin).
  The REBOUND check used end-of-step collision normals and a velocity-dependent
  leapfrog.

# The brief given to the design run

The design was produced in three stages by independent agents. Each designer got the same context and task (below) and its own angle, and none saw the others' work; the synthesis got the three designs and had to judge every claim against the code; the critique got the synthesis and had to find every way it could fail. Text reproduced verbatim except local paths (`<repository>`) and two wording changes ("the user" -> "the project", "this morning" -> the date). The measured numbers in the context come from loading the campaign's own autosaved states of 29 Oct 2026 and evaluating every refused cluster's contact ratios.

## Context (given to every agent)

```text
You are working on the DOLL-E dust simulator, a Python package at <repository> (package dolle_sim). READ-ONLY task: do not edit any
file in the repository, do not run long simulations (a campaign is running on this laptop and memory is short: do not
start Python processes that import dolle_sim unless essential, and never more than one at a time). You may write scratch
files only under your own temp dir if needed.

The engine (dolle_sim/engine.py, class FullContactSim) integrates every grain of a dust sample (100 grains, 45-106 um,
silica or graphite, in a 0.5x0.5x1.5 in glass tubule on a sounding rocket) with the full contact mechanics of
dolle_sim/dem.py (JKR normal force, Mindlin sliding spring, rolling and twisting spring-dashpot-sliders "sds" with caps
mu*Fncrit, Fncrit = |Fne + 2 Fpull|; see dem.contact_force, lines ~200-392). Regimes: FREE, ACTIVE (full DEM at the
stiff contact step dt_dem = fastest contact period/50 = 5-25 ns), ASLEEP, ROLL (roll1.py), SLAV (slaved.py). The standing
requirement of the project: NO shortcuts; every treatment exact or with an explicit, small, documented bound, and verified
against the full model with a published criterion (see CHECKS.md and dolle_sim/verify_full.py for the existing checks).

Sleeping clusters ("islands") are handled with exact linear statics (dolle_sim/statics.py: K, J by central differences
of the production kernel; margins(); energy_bounds(); linres residual-motion tracking):
- entry: FullContactSim._try_sleep_island (engine.py ~line 1905): equilibrium du = K^-1 (Fext + Fnet), lam_eq = lam + J du;
  ratio = statics.margins(lam_eq) at T and at T+H (loads at the step end); REQUIRES rr.max <= 1 - 5*margin (= 0.90,
  margin = 0.02); then a residual-motion test (kinetic + elastic deviation energy can add at most 1% of the smallest
  remaining margin; exact energy-ellipsoid bound statics.energy_bounds; else linres tracking via _linres); then the
  cluster is placed at its equilibrium at rest and the reference (lam, J, K, LU, F_ref) is stored.
- stay: every outer step in _step_once (~line 495-505) and in _skip_asleep (~1153): _island_ratio(isl, Ts) (linear
  response lam = lam_ref + J K^-1 (F(Ts) - F_ref)) at Ts = T, T+H/2, T+H must stay <= 1 - margin (0.98), else the
  island wakes (_wake_island: placed at the linear-response equilibrium of the current load, at rest, into the DEM).
  Any fast load window or load breakpoint in the step wakes everything.
- single grains on one wall: _try_sleep / ASLEEP with _margins (same 5*margin entry, margin stay).

MEASURED PROBLEM (29 Oct 2026, from autosaved states of the running campaign, nominal scenario, 100 grains/tubule):
- Almost all computing goes to clusters that are STATIC (speeds 1e-13 to 2e-9 m/s, a handful at 1e-4 m/s) but refused
  by the entry test, so they are integrated by the full DEM at 5-18 ns steps (4e7-1.2e8 DEM steps per flight-second):
  * boost/coast phases (T+20-39 s; axial load 0.2-25 g, payload spin 4.8 Hz, so ~8 g centripetal along the walls): the
    refused clusters (sizes 1, 2, 5, 12, 57 grains, piles in corners/edges with redundant contacts) have their worst
    contact AT the ROLLING cap: ratio >= 0.999 (most), some 0.98-0.999, some 0.90-0.98 (rolling or twisting).
  * microgravity (T+72 s, loads ~1e-4 g, spin 0.0167 Hz): a grain in an edge (two wall contacts, locked-in normal force
    0.66 uN at each, tangential 21% of the sliding limit) refused at TWIST ratio 0.962/0.956; a 3-grain bridge (two on
    the wall, one on both) refused at ROLL ratio 0.926. These stay in the DEM for the remaining ~800 s of flight.
- Consequence: silica tubules need ~6000-10000 s of computing per flight-second in the coasts, and microgravity would
  cost days per tubule; the 13-scenario campaign would take weeks. The fix must make these static clusters sleep.

PHYSICS NOTES: a contact whose sds spring sits at its cap with zero relative motion keeps exactly that state (no
increment, no clipping) as long as the cluster does not move; the DEM would change a plastic variable only if the load
pushes the contact further outward along its cap (plastic flow) or residual oscillations ratchet it. Loads on a
sleeping cluster vary smoothly between breakpoints (thrust tail, drag decaying with altitude, constant spin, Coriolis
zero at rest); Loads has interval bounds (dolle_sim/loads.py accel_bound) that could bound load changes over a step.
The twisting kernel has a quirk: at zero twist rate an over-cap twisting spring is reset to 0 (sg = 0), see dem.py.
```

## Task (given to the three designers)

```text
Produce a concrete, code-level design for letting such clusters sleep, with:
1. The entry criterion (what replaces/extends 'rr.max <= 0.90' for contacts at or near a cap), the stay criterion each
   outer step (and in _skip_asleep), and the wake rule, so that either (a) the full DEM provably would not change any
   plastic variable over the sleep interval, or (b) any change it would make is bounded by an explicit, small,
   computable quantity that the engine accumulates and reports (state the bound, its derivation, and its size for the
   numbers above). Cover: loads varying between the sampled times (use interval bounds, not samples, where it matters
   at zero margin), the linearisation error of the linear response, residual motion at capped contacts (ratcheting),
   the cap depending on the normal load (Fncrit), the dashpot term in the cap, and the twisting-kernel quirk.
2. Exactly which functions change (engine.py, statics.py; names, what each new piece computes), how the island
   reference stores per-contact state needed (e.g. entry ratios, which contacts are 'at limit'), checkpoint/pickle
   compatibility, and interaction with linres, slaved clusters, _verify and _skip_asleep.
3. The verification: new verify_full check(s) with criteria (e.g. sleeping vs all-DEM reference runs, all_dem=True mode
   exists, for a pile at its rolling cap under constant, decreasing and increasing loads; an edge grain with locked-in
   twist in microgravity) and unit tests; what each proves.
4. Expected speed-up for the measured cases and the risks / failure modes.
Read the actual code before designing (engine.py _try_sleep_island, _island_ratio, _step_once lines ~470-560,
_skip_asleep, _wake_island, _linres, _try_sleep, _margins; statics.py; dem.contact_force). Be precise; prefer the simplest
design that is rigorous. Return markdown in 'design' plus the structured fields.
```

## The three angles

* **unloading-certificate**: ANGLE: an elastic-unloading certificate. Contacts at or near a cap are allowed while the linear response (with interval load bounds) proves their load ratio cannot rise above its entry value / the cap over the step; bound the neglected ratcheting from residual motion via the energy ellipsoid; aim for the smallest change to the existing island machinery.
* **nonlinear-verification**: ANGLE: exact nonlinear verification. Instead of trusting the linear response near a cap, evaluate the production kernel (statics.state / apply_displacement) at the predicted configuration each step to certify equilibrium and limits (a Newton-type certificate), and design how to make this cheap enough; derive what is exact and what is bounded.
* **risk-first**: ANGLE: risk-first. Enumerate every way sleeping at a cap can silently differ from the full DEM (missed plastic flow between samples, ratcheting, cap shrinking with Fncrit, twist quirk, redundancy, wake transients, interaction with neighbours joining the DEM, checkpoints), then design the minimal rigorous criteria and the verification checks that would catch each failure.

## Task given to the synthesis (the three designs attached as JSON)

```text
Three independent designs for letting static at-limit clusters sleep follow (JSON). Read the relevant code yourself and
judge each claim against it (do not trust a design's description of the code). Then write the FINAL design: the
simplest rigorous one, grafting the best ideas from the others, with step-by-step implementation instructions precise
enough to code from (function signatures, formulas, thresholds and their justification, what is stored in the island
dict, the exact stay/wake tests, the error-bound bookkeeping in n_events, the verify_full check(s) with numeric criteria,
and unit tests). State clearly what is exact and what is bounded (with the bound). Flag anything in the designs that is
wrong about the code or not rigorous.
```

## Task given to the critique (the synthesis attached as JSON)

```text
Adversarially review this FINAL design for letting static at-limit clusters sleep. Read the code it touches. Find every
way it could (1) let the sleeping model differ from the full DEM by more than its stated bound, (2) wake too late (plastic
flow missed between samples or by linearisation error), (3) never sleep the measured cases (so no speed-up), (4) break
existing machinery (linres, slaved, _verify, _skip_asleep, checkpoints/pickling, branch(), counters), or (5) be untestable.
For each problem give the concrete scenario and the fix. Return markdown in 'design' (your review, with a corrected
summary of the criteria at the end) and fill the structured fields with the corrected version.
```

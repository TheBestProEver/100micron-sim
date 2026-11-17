# DOLL-E simulator: the decision log

Every decision made in this project from the first commit (2026-10-20) to 2026-11-16: what was decided, when, by
whom, what else was considered, why, how much the answer depends on it, and how to change it. Read
[OVERVIEW.md](OVERVIEW.md) first for the physics in 15 minutes; this file is the reference behind it.

Every number has a source: a paper (authors and year), a data sheet or published document (by title), the code (a
file or function), a check (`check:<id>`, its criterion and measured value in [CHECKS.md](../CHECKS.md)), or a
measurement made for this log with the repository's own functions (stated as "measured"); anything without one is
marked **ASSUMED**. Grain radius 50 um (diameter 100 um) unless stated.

## 1. How to read it

**IDs.** One letter for the section, one number: A the question and the scope; B physics models; C material values;
D numerical methods and verification; E engines and software architecture; F the flight and the experiment; G team
decisions and project process; H superseded decisions; O open decisions.

**Fields of each entry.**

| field | meaning |
|---|---|
| Decision | the decision in one sentence |
| Date | the commit date (from git), or "project notes" when the date comes from the owner's notes |
| Decided by | **Owner**: the student who owns the project; **Team**: the DOLL-E team (or the owner speaking for it); **Team reviewer**: the team member whose feedback of 2026-11-10 led to v0.7; **Engineering**: worked out while building the code (by the AI coding sessions working for the owner), usually with a check or measurement as evidence; **Review**: an automated adversarial review that found a problem (the fix is then Engineering) |
| Status | **in force**; **in force (extras)**: in force in the optional v0.6 engine in `extras/`; **superseded by X** (or "in part"); **open**: still to be decided; **done**: a one-time action completed; **rejected**: considered and not done |
| Alternatives considered | what else was possible or tried |
| Why | the reasons, with the evidence |
| Effect on results | how much the answers depend on it, with numbers |
| How to change it | the file, the `module.function`, the config key or the command, and what to re-run |
| References | commits (`commit:<hash>`), notebooks, checks, documents |

**Names written in code font are tested.** `tests/test_decisions_doc.py` fails if any file path, `module.function`,
`check:<id>`, test name or `commit:<hash>` written in this file stops existing, so the log cannot silently drift from
the code (paths under `results/` are output files that the named commands write, so they are not required to
exist). Config keys are written as in the JSON files (`contact.friction_coefficient` means the key
`friction_coefficient` inside `"contact"`).

**Changing a value without editing code.** In a `dolle_sim` config, `"materials": {"silica": {"gamma": 0.1,
"e_dry": 0.9}}` overrides any field of any material (`dolle_sim.materials.get_material`). In python with
`dolle_core`: `m = dataclasses.replace(dolle_core.MATERIALS["silica"], gamma=0.1)` and pass `m` instead of the name
to `dolle_core.pair.make_pair`, `dolle_core.collide.collide` or `dolle_core.wall.wall_limits`. In a config for the
v0.6 engine (`python -m dolle_extras mission`), the `contact` keys `rolling_hysteresis` (h, default 0.5),
`rolling_q` and `twisting_q` (Q, default 100) and `friction` (per pair, for example `{"silica-wall": 0.4}`) set the
rolling, twisting and friction parameters without editing code (read in `dolle_extras.engine.FullContactSim`).

**What to re-run after a change (the usual set).**

| after changing | run |
|---|---|
| a material value or pair rule | `python -m pytest tests/test_dolle_core.py -q` (two tests hold reference numbers on purpose: update them and every document quoting them), `python -m pytest tests/test_core_vs_dem.py -q` (the friction table and wall limits against the v0.6 contact model; skipped when `extras/` is absent), `python -m pytest tests/test_notebooks.py -q` (then re-read the numbers in the notebook text), `python -m dolle_sim verify` |
| the 3-D core or its start | `python -m pytest tests/test_dolle_core_sim3d.py tests/test_sim3d_page.py -q`, `python tools/make_sim3d_replays.py` |
| a default force or a shipped config | `python -m pytest tests/test_defaults.py -q` |
| the flight profile or the mission | `python -m pytest tests/test_flight.py -q`, `python -m dolle_sim mission configs/mission_quarter_deck.json --loads-only` |
| a check | `python -m dolle_sim verify --json results/verify_all.json` (a v0.6 check: `python -m dolle_extras verify --json results/verify_extras.json`; test counts: `python -m pytest tests --junitxml=results/pytest.xml`), then `python tools/checks_table.py` (regenerates `CHECKS.md`) |
| the provenance text | `python tools/build_site.py` (regenerates `PROVENANCE.md`) |
| anything in `extras/` | `python -m pytest extras/tests -q`, `python -m dolle_extras verify` |
| everything | `python -m pytest` |

## 2. Index

| ID | decision | status | where to change it |
|---|---|---|---|
| A1 | the question: sticking speed, carbon vs silica | in force | - |
| A2 | reference grain: 50 um radius, head-on | in force | radius arguments; `size_distribution` |
| A3 | scope: one impact to the whole flight, to splashdown | in force | `simulate_to_s` |
| A4 | every identifiable force in the code | superseded in part by B21 | force switches |
| A5 | payload spin, rotating frame | in force | `flight_profile` spin keys |
| A6 | predictions are design guidance | in force | `README.md` |
| A7 | pre-flight, sweep, inversion, synthetic camera | in force | `dolle_sim/design.py` |
| A8 | offline 3-D viewer that can hide wall-stuck grains | in force | `dolle_sim/viewer_template.html` |
| B1 | JKR adhesion theory | in force; silica open (O4) | `dolle_core/jkr.py` |
| B2 | break at the end of the stable branch | in force | `break_mode` |
| B3 | Tsuji damping with the JKR radius, set by e_dry | in force | `dolle_core/damping.py` |
| B4 | Thornton-Ning plastic yield | in force | `plastic=False`; `yield_pressure` |
| B5 | four stick/bounce tests, deterministic | in force | `dolle_core/collide.py` |
| B6 | only the normal speed decides; head-on threshold | in force | v0.6 engine only |
| B7 | mixing rules for unlike pairs (ASSUMED) | in force | `dolle_core/pair.py` + copies |
| B8 | default engine: wall grains held until pull-off | in force | `python -m dolle_extras mission` |
| B9 | wall limits: pull-off, sliding, rolling, twisting | in force | `dolle_core/wall.py` |
| B10 | vibrating walls: flux-weighted phase | in force | `wall_vibration` |
| B11 | impacts on wall-bound grains kept separate | in force | `dolle_sim/simulation.py` |
| B12 | stuck grains form rigid clumps | in force | `dolle_core/clumps.py` |
| B13 | bounce impulse, Coulomb cone, e_t = 0, energy guard | in force | `contact.tangential_restitution` |
| B14 | Dominik-Tielens fragmentation paid from E_diss | in force | `contact.fragmentation` |
| B15 | fragment separation energy (ASSUMED) | in force; open (O8) | `contact.fragment_separation` |
| B16 | stored bonds per clump | in force | `dolle_core/clumps.py` |
| B17 | clumps rigid, no restructuring | open (O7) | v0.6 contact model |
| B18 | gas drag: Stokes-Cunningham, co-rotating gas | in force | `gas.pressure_pa` |
| B19 | contact charging, condenser model | in force | `contact.contact_charging` |
| B20 | Ewald space charge, grounded walls | in force | `forces.space_charge` |
| B21 | negligible forces off by default | in force | `forces.*` switches |
| B22 | 3-D core leaves out drag, roll, charging | in force; roll open (O6) | `dolle_core/sim3d.py` |
| B23 | thermophoresis on by default; 3 K/m in the demo (ASSUMED) | in force | `forces.thermophoresis`; `environment.temperature_gradient_k_m` |
| C1 | silica values (gamma, E, nu published; rho ASSUMED) | in force (placeholder) | both material tables |
| C2 | graphite values (Chokshi et al. 1993) | in force (placeholder) | both material tables |
| C3 | e_dry ASSUMED for every material | in force; open (O1) | `e_dry` |
| C4 | yield pressures ASSUMED | in force | `yield_pressure` |
| C5 | other dust materials are placeholders | in force | both material tables |
| C6 | glass walls: data sheets + measured adhesion | in force | glass entries |
| C7 | anti-stick coating, wall adhesion factor | in force; open (O11) | `wall_adhesion_factor` |
| C8 | grain-to-grain scatter (ASSUMED amplitudes) | in force | `surface_energy_log_sigma` |
| C9 | friction: per-pair table vs global 0.3 | in force; open (O15) | `dolle_core/materials.py`; `contact.friction_coefficient` |
| C10 | rolling h = 0.5, Q = 100 (ASSUMED) | in force; open (O5) | `h` of the wall limits |
| C11 | non-mechanical properties, silent defaults | in force | `dolle_sim/materials.py` |
| D1 | impacts integrated, Verlet, 300 sub-steps | in force | `n_sub`; `contact.steps_per_contact` |
| D2 | threshold by bisection to 0.2% | in force | `tol` |
| D3 | Verlet with exact drag and Coriolis | in force | `dolle_sim/simulation.py` |
| D4 | DLM rigid-body rotation | in force | `dolle_core/clumps.py` |
| D5 | exact contact times, re-predict after impacts | in force | `dolle_core/sim3d.py` |
| D6 | time steps of the default engine | in force | `dt_s`, `dt_launch_s` |
| D7 | exact shortcuts while dust is wall-bound | in force | `dolle_sim/simulation.py` |
| D8 | checks with criteria; generated CHECKS.md | in force | `dolle_sim/verification.py` |
| D9 | generated provenance table; claim audit | in force | `dolle_sim/provenance.py` |
| D10 | LAMMPS as referee of the contact law | in force (extras) | `python -m dolle_extras lammps-check` |
| D11 | REBOUND only as an outside reference | in force (extras) | `python -m dolle_extras rebound-check` |
| D12 | readable core equals the full code | in force | `tests/test_dolle_core.py` |
| D13 | notebooks executed by tests, reviewed | in force | `tests/test_notebooks.py` |
| D14 | documents naming code are tested | in force | `tests/test_decisions_doc.py` |
| D15 | v0.6: exact or bounded, every step verified | in force (extras) | `extras/dolle_extras/engine.py` |
| D16 | v0.6 step 1/50 of the fastest contact period | in force (extras) | `contact.steps_per_period` |
| D17 | exact free flight to d_cap = 0.1 um | in force (extras) | `contact.capture_distance_m` |
| D18 | sleep certificate, 2% margin | in force (extras) | `contact.sleep_margin` |
| D19 | residual motion as linear response | in force (extras) | `extras/dolle_extras/engine.py` |
| D20 | slaved rolling and slaved clusters | in force (extras) | `extras/dolle_extras/roll1.py`, `extras/dolle_extras/slaved.py` |
| D21 | local speed bounds; short steps after jumps | in force (extras) | `extras/dolle_extras/engine.py` |
| D22 | branches from checkpoints only when exact | in force (extras) | `extras/dolle_extras/engine.py` |
| D23 | v0.6.1 exits, rejoin, safety net | in force (extras) | `extras/dolle_extras/engine.py` |
| D24 | rolling envelope assumes no direction | in force (extras) | `extras/dolle_extras/engine.py` |
| D25 | one Loads object; refuse unseen loads | in force (extras) | `extras/dolle_extras/loads.py` |
| D26 | >= 40 steps per load cycle (relaxing rejected) | in force (extras) | `extras/dolle_extras/engine.py` |
| D27 | at-limit sleep: tier 1 exact, tier 2 creep | in force (extras) | `contact.cap_sleep` |
| D28 | left limits at breakpoints; chord bound | in force (extras) | `extras/dolle_extras/loads.py` |
| D29 | ratchet bound in the released system | in force (extras) | `extras/dolle_extras/engine.py` |
| D30 | drift-relative entry energy (own derivation) | in force (extras); open (O9) | `extras/dolle_extras/engine.py` |
| D31 | tier-2 lag as running maximum (ASSUMED) | in force (extras) | `extras/dolle_extras/engine.py` |
| D32 | island tolerances | in force (extras) | `contact.island_*` keys |
| D33 | single grain: entry = stay at 0.98 | in force (extras) | `extras/dolle_extras/engine.py` |
| D34 | switches cap_sleep, twist_continuity | in force (extras) | `contact.cap_sleep` |
| D35 | checks 14-19; use of the results gated on them | done | `extras/dolle_extras/verify_full.py` |
| D36 | full-stiffness LAMMPS run after a release | open (O10) | `extras/tools/` |
| E1 | own simulator, not REBOUND or LAMMPS | in force | - |
| E2 | v0.5 event-driven engine | in force (default) | `dolle_sim/simulation.py` |
| E3 | v0.6 full-contact engine, "no shortcuts" | superseded by E5 (as default) | `extras/dolle_extras/engine.py` |
| E4 | v0.6 law in the LAMMPS form | in force (extras) | `extras/dolle_extras/dem.py` |
| E5 | v0.7: v0.6 to extras, v0.5 default | in force | `"engine"` key |
| E6 | `dolle_core`: copy of the physics, plain numpy | in force | `dolle_core/` |
| E7 | values and law in several places, kept equal | in force | all copies |
| E8 | `dolle_sim` never imports the extras | in force | `tests/test_provenance.py` |
| E9 | 3-D simulation in `dolle_core`, rigid clumps | in force | `dolle_core/sim3d.py` |
| E10 | pre-v0.7 state files no longer load | in force | - |
| F1 | whole flight on one clock | in force | `--scenario` |
| F2 | quarter deck, four tubules (example) | in force (example) | `tubules` |
| F3 | flight timeline from the guide + ASSUMED shapes | in force | `flight_profile` |
| F4 | residual spin 1 rpm (ASSUMED) | in force | `flight_profile.residual_spin_hz` |
| F5 | statistics only in T+88-313 s | in force | `flight_profile.ug_end_s` |
| F6 | residual 1e-4 g along the drag (ASSUMED) | in force | `flight_profile.residual_g` |
| F7 | re-entry: point mass, tabulated atmosphere | in force | `dolle_sim/atmosphere.py` |
| F8 | parachute 5 g, splash 0.05 s (ASSUMED) | in force | `flight_profile.splash_stop_s` |
| F9 | attitude at T+313 s; peak ~8-11 g | in force | `flight_profile.attitude_change_s` |
| F10 | launch-load report, Rice estimate | in force | `--loads-only` |
| F11 | v0.6 release waveforms; vibration upper bound | in force (extras) | `extras/dolle_extras/vibration.py` |
| F12 | dust starts as one layer (ASSUMED) | in force | `initial_state` |
| F13 | 800 grains in config, 100 in simulations | open (O12) | `dust.count`; `--grains` |
| F14 | release, gas, orientation as scenarios | open (O3) | `scenarios`; `release` |
| F15 | air at 1e-3 Pa (ASSUMED) | in force; open (O3) | `gas.pressure_pa` |
| F16 | flight demo: anti-stick and vibrating walls | in force (demo) | `configs/flight_carbon_vs_silica.json` |
| F17 | grain size distributions | in force | `size_distribution` |
| F18 | stereo cameras recommended | in force | `camera` |
| F19 | logistic threshold fit; coverage caveat | in force; open (O14) | `dolle_sim/analysis.py` |
| F20 | launch report's sliding flag mu (F_c - pull) | in force; open (O15) | `dolle_sim/mission.py` |
| F21 | environment presets; g-jitter 5e-5 g rms (ASSUMED) | in force | `environment.preset`; `jitter_rms_g` |
| G1 | public text names only "DOLL-E" | in force | - |
| G2 | git history not rewritten | in force; open (O17) | - |
| G3 | v0.6 campaign, one job per tubule | stopped (G6) | `extras/tools/run_campaign.py` |
| G4 | autosave and resilience | in force (extras) | `extras/tools/run_campaign.py` |
| G5 | measure first; stop and discard | done | - |
| G6 | campaign restarted four times, stopped for good | final | - |
| G7 | team feedback: understanding first | in force | - |
| G8 | notebooks with "why this choice" boxes | in force | `notebooks/` |
| G9 | deletions; short README | in force | `README.md` |
| G10 | stored results not re-run after v0.7 | open (O13) | `docs/MISSION_RUNS.md` |
| G11 | original website runs the real package | in force | `dolle_sim/webapi.py` |
| G12 | keep the site; new 3-D page | in force | `docs/sim3d/` |
| G13 | 3-D page start: compact cloud (ASSUMED) | in force | `dolle_core/scenes.py` |
| G14 | carbon not decided | open (O2) | material tables |
| G15 | silica not decided | open (O2) | material tables |
| G16 | three designs, synthesis, critique | method in force | - |
| G17 | decision map; branch shown on the front page | in force | `extras/docs/ENGINE_LOGIC.md` |
| G18 | 15-minute overview; this log; a study guide | in force | `docs/OVERVIEW.md` |
| G19 | no licence chosen | open (O18) | a `LICENSE` file |

Superseded decisions (H1-H26) and open decisions (O1-O18) have their own tables at the end.

## A. The question and the scope

### A1. The question: the sticking speed of ~100 um grains, carbon against silica
- **Decision:** the simulator serves one question: below what impact speed do ~100 um dust grains stick, is it
  different for carbon and silicate (silica) dust, and can the flight free the dust and film it well enough to
  measure that speed.
- **Date:** 2026-10-20 · **Decided by:** Team (the experiment's purpose) · **Status:** in force.
- **Alternatives considered:** none recorded.
- **Why:** the sticking speed is the quantity DOLL-E measures; on the ground, mm/s motions of 100 um grains are
  dominated by gravity, which is why the experiment flies (`notebooks/00_start_here.ipynb`).
- **Effect on results:** it defines every output. Today's answers: silica 6.18 mm/s, graphite 38.4 mm/s for two
  equal 50 um radius grains head-on (`examples/sticking_threshold.py`, `docs/OVERVIEW.md` section 2).
- **How to change it:** not a parameter.
- **References:** `commit:a3fffe8`, `commit:f0d7307`; `docs/OVERVIEW.md`.

### A2. The reference grain: 100 um diameter (50 um radius), two equal spheres head-on
- **Decision:** every headline number is for two equal grains of 50 um radius meeting head-on (or one grain on a
  flat wall); the mission examples use sieve bands 90-106 um and 45-53 um.
- **Date:** 2026-10-20 (reference); 2026-10-22 (sieve bands) · **Decided by:** Team (grain size), Engineering
  (the example bands) · **Status:** in force; the real powders are not chosen (G14, G15).
- **Alternatives considered:** any radius or size distribution (`dolle_sim/sizes.py`, F17).
- **Why:** the experiment's grains are ~100 um; a single reference size makes materials comparable.
- **Effect on results:** sticking speed scales as radius^(-5/6) (log-sensitivity -0.83, measured with
  `dolle_core.collide.sticking_threshold`). At 24.5 um radius: silica 11.2 mm/s, graphite 69.5 mm/s. Wall limits
  of the small grains: pull-off x4.2, rolling x5.3 (silica on borosilicate: 4,305 g and 6.83 g instead of 1,034 g
  and 1.29 g, `dolle_core.wall.wall_limits`).
- **How to change it:** radius arguments of `dolle_core.pair.make_pair` and `dolle_core.wall.wall_limits`; per
  tubule `size_distribution` in `configs/mission_quarter_deck.json`. Re-run `tests/test_dolle_core.py` only if the
  reference numbers themselves change.
- **References:** `commit:a3fffe8`, `commit:a866c4e`; `notebooks/03_silica_vs_carbon.ipynb`,
  `notebooks/04_grain_on_a_wall.ipynb`.

### A3. Scope: from one impact to the whole flight, lift-off to splashdown
- **Decision:** the simulator covers one impact, many grains in a tubule, and the whole flight on one clock from
  lift-off through the microgravity window, re-entry, parachute and splashdown (T+882 s); mission runs end at
  T+887 s, 5 s after splashdown (comment in `configs/mission_quarter_deck.json`).
- **Date:** 2026-10-22 (whole flight to power-off, v0.4); 2026-10-26 (to splashdown, v0.5) · **Decided by:** Owner
  · **Status:** in force.
- **Alternatives considered:** only the microgravity window; stopping at experiment power-off (T+332 s, the v0.4
  state).
- **Why:** the owner asked for the whole flight with verifiable sources; launch loads, re-entry and splashdown
  decide whether dust stays on the walls before and after the measurement.
- **Effect on results:** with clean or silanised glass nothing comes off the walls in any phase, the ~31 g
  splashdown included; with the `rough_grains` scenario (wall adhesion x0.02) 1-11 grains per tubule come off at
  splashdown, 6-72 if the payload hangs nose up (`docs/MISSION_RUNS.md`; computed on 2026-10-26 with the v0.5 code,
  `commit:0731b2f`, and never re-run since, see G10).
- **How to change it:** `simulate_from_s` and `simulate_to_s` in the mission config; `python -m dolle_sim mission
  configs/mission_quarter_deck.json --loads-only` for the loads alone.
- **References:** `commit:a866c4e`, `commit:0731b2f`; `docs/MISSION_RUNS.md`;
  `notebooks/05_the_flight_in_brief.ipynb`.

### A4. Realism first: every force that can be identified is in the code
- **Decision:** include residual gravity, g-jitter, payload spin, tidal and mutual gravity, gas drag and torque,
  Brownian motion, thermo- and photophoresis, radiation pressure, Coulomb and polarisation forces, image charges,
  external fields, the Lorentz force and contact charging (thermophoresis: B23; g-jitter: F21).
- **Date:** 2026-10-20 · **Decided by:** Owner · **Status:** superseded in part by B21 (since v0.7 the negligible
  forces are switched off by default; the code stays).
- **Alternatives considered:** only the forces that matter.
- **Why:** the owner's requirement of maximal realism (project notes, 2026-10-20).
- **Effect on results:** none on the sticking speeds; the largest of the switched-off forces is 2.2% of the
  residual acceleration (B21).
- **How to change it:** each force has a switch (B21).
- **References:** `commit:a3fffe8`, `commit:13dd8e8`.

### A5. The payload's spin is modelled, in a frame that spins with the payload
- **Decision:** grains are tracked in the frame rotating with the payload (centrifugal, Euler and exact Coriolis
  terms); inertially fixed vectors (the residual acceleration's direction, the magnetic field) rotate in that frame.
- **Date:** 2026-10-20 (frame); fixed in v0.2 (rotating fixed vectors) · **Decided by:** Owner (spin must be in),
  Engineering (method) · **Status:** in force. The spin values are F3 and F4.
- **Alternatives considered:** a non-rotating frame (would need rotating walls instead).
- **Why:** the payload spins at ~4.8 Hz during the burns and keeps a residual spin in the window; in the tubule's
  own frame the walls are fixed, so wall contacts stay simple.
- **Effect on results:** the spin is the largest sideways load on wall dust during the boost (~9.6 g at the far
  tubule corner at 4.8 Hz, `docs/MISSION_RUNS.md`), above every rolling limit (B9).
- **How to change it:** the spin entries of `flight_profile` (F3, F4); `spin_hz` in the environment for runs
  without a flight profile.
- **References:** `commit:a3fffe8`, `commit:8060592`; `check:rotating_frame`, `check:spin_rotating_frame`,
  `check:flight_frame`.

### A6. The predictions are design guidance, not the answer
- **Decision:** the documents present the predicted sticking speeds as design guidance, state that the material
  data are the weak point, and compare with the one published measurement at a similar geometry.
- **Date:** 2026-10-21 (claim audit); 2026-11-11 (comparison corrected) · **Decided by:** Owner (trust),
  Engineering (wording) · **Status:** in force.
- **Alternatives considered:** presenting the numbers as predictions of the flight result.
- **Why:** for 1.2 um silica spheres hitting a flat target the model gives 0.27-0.28 m/s, while Poppe, Blum &
  Henning (2000) measured ~1.2 m/s, 4.3-4.5x more (`notebooks/00_start_here.ipynb`). The earlier text compared with
  two free grains (0.25 m/s, 4.9x), which is not the experiment's geometry. Kimura et al. (2015) argue that dry
  silica in vacuum has a surface energy up to 10x the common value; Kreuzig et al. (2024) find no such rise for
  water-repellent silica.
- **Effect on results:** none on the numbers; it sets how they are presented.
- **How to change it:** `README.md`, `docs/OVERVIEW.md`.
- **References:** `commit:ed2f49d`, `commit:13dd8e8`; `docs/HISTORY.md` (v0.7).

### A7. Design and measurement tools from the start
- **Decision:** besides the simulation, ship a pre-flight design check with a force budget, sticking-probability
  sweeps, an inversion from a measured sticking speed to an effective surface energy, and a synthetic camera with a
  threshold fit.
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** a simulation only.
- **Why:** the team needs to know whether the cameras can measure the threshold and how to turn a measurement into
  a material value.
- **Effect on results:** the inversion returns an EFFECTIVE surface energy that absorbs every other model error.
  Because the sticking speed goes as gamma^(5/6) times f(e_dry) and e_dry is not measured (C3), a measured speed
  alone cannot separate gamma from e_dry.
- **How to change it:** `dolle_sim.design.preflight`, `dolle_sim.design.stick_probability_curve`,
  `dolle_sim.design.infer_surface_energy` (`python -m dolle_sim invert --material silica --vstick 0.006`).
- **References:** `commit:a3fffe8`, `commit:8060592`; `PROVENANCE.md`.

### A8. An offline 3-D viewer that can hide the grains stuck to the walls
- **Decision:** every `python -m dolle_sim run` and mission run writes viewer.html (in its output folder), a
  single-file 3-D replay that opens in a browser without a server, with a switch to hide grains stuck to the walls
  and a choice of colouring (for example by spin).
- **Date:** 2026-10-20 · **Decided by:** Owner (requirement, project notes; listed in the first commit message),
  Engineering (implementation) · **Status:** in force.
- **Alternatives considered:** plots only; the website's replays (added the same day, G11).
- **Why:** in a run with walls most grains end up on them, and they hide the free grains whose collisions are the
  data.
- **Effect on results:** none on numbers.
- **How to change it:** `dolle_sim/viewer_template.html` (the page), `dolle_sim.output.write_viewer` and
  `dolle_sim.results.viewer_payload` (what goes into it; `--viewer-particles` and `--viewer-fps` of
  `python -m dolle_sim run`).
- **References:** `commit:a3fffe8`.

## B. Physics models

### B1. JKR adhesion theory for every contact
- **Decision:** every contact (grain-grain and grain-wall) follows the JKR theory (Johnson, Kendall & Roberts 1971)
  in the collision form of Chokshi, Tielens & Hollenbach (1993).
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force; its accuracy for silica is open (O4).
- **Alternatives considered:** DMT (Derjaguin, Muller & Toporov 1975: pull-off 2 pi w R, no neck, no hysteresis);
  Maugis-Dugdale (Maugis 1992: spans both, but needs the range z0 of the surface forces, which is not measured).
- **Why:** the standard model of the published dust-collision work (Chokshi et al. 1993; Dominik & Tielens 1997;
  Wada et al. 2007) and of the LAMMPS GRANULAR package, which allows a term-by-term cross-check (D10).
- **Effect on results:** decides the hysteresis energy and therefore the threshold. Which theory applies is set
  by the Tabor parameter (`dolle_core.jkr.tabor_parameter`, z0 = 0.3 nm ASSUMED): silica pairs 1.44, silica on
  borosilicate 1.61, silica on silanised glass 1.12 (between the JKR and DMT limits: JKR overstates the hysteresis,
  so silica's 6.2 mm/s is probably an upper estimate); graphite pairs 8.76, graphite on borosilicate 4.99 (JKR
  applies). For pull-off JKR is the low end: the true value lies between 1 and 4/3 times JKR
  (`notebooks/04_grain_on_a_wall.ipynb`).
- **How to change it:** a new force law in `dolle_core/jkr.py` and `dolle_core/collide.py`, mirrored in
  `dolle_sim/contact.py` and `extras/dolle_extras/dem.py`; then re-run everything (`python -m pytest`,
  `python -m dolle_sim verify`, `python -m pytest extras/tests`).
- **References:** `commit:a3fffe8`; `notebooks/01_two_grains_colliding.ipynb`; `check:threshold` (undamped 3.825 vs
  3.827 mm/s), `check:elastic`.

### B2. The contact breaks at the end of the stable JKR branch
- **Decision:** a stretched contact breaks at the end of its stable branch (overlap delta_c, contact radius
  a0/6^(2/3)), the rule of Chokshi et al. (1993) and Dominik & Tielens (1997).
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** break at the maximum tension F = -F_c (the LAMMPS rule), kept as an option and used
  only for the LAMMPS cross-check.
- **Why:** beyond the maximum tension a contact under a prescribed displacement is still stable; it becomes unstable
  only at the end of the branch (`notebooks/01_two_grains_colliding.ipynb`).
- **Effect on results:** the force rule gives undamped sticking speeds lower by a factor 1.42297 for every material,
  and damped thresholds lower by 20.5% (silica, 6.18 to 4.91 mm/s) and 17.5% (graphite, 38.4 to 31.6 mm/s).
- **How to change it:** `break_mode="force"` in `dolle_core.collide.sticking_threshold`,
  `dolle_core.collide.impact` and `dolle_core.collide.collide`; `separation="force"` in
  `dolle_sim.contact.JKRContact`. Not exposed in configs. Re-run `tests/test_dolle_core.py`.
- **References:** `commit:a3fffe8`, `commit:a866c4e` (LAMMPS check with the matched rule).

### B3. Damping: the Tsuji form, evaluated with the JKR contact radius, calibrated to e_dry
- **Decision:** the damping force is -2 sqrt(5/6) beta sqrt(2 E* m*) sqrt(a) times the approach speed (Tsuji,
  Tanaka & Ishida 1992), with beta set from the pair's dry restitution e_dry; it uses the JKR contact radius a, so
  it also acts on the stretched neck (this project's own adaptation).
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** a constant dashpot (the restitution would depend on speed); the viscoelastic law of
  Brilliantov et al. (1996) (restitution falls with speed; needs a dissipation constant not known for these grains).
- **Why:** one number, e_dry, sets it; the restitution of a non-sticking impact is then independent of speed, so
  the damped threshold is the undamped one times a factor f(e_dry) that depends on e_dry alone.
- **Effect on results:** damping raises the thresholds x1.61 (silica, e_dry 0.85) and x2.46 (graphite, 0.70). The
  calibration reproduces e_dry within 6.1e-4 for e_dry 0.5-0.95. The e_dry values themselves are C3.
- **How to change it:** `dolle_core.damping.damping_coefficient` and `dolle_core.damping.damping_beta`, mirrored in
  `dolle_sim.contact._resolve_core` and `extras/dolle_extras/dem.py`. Re-run `tests/test_dolle_core.py`,
  `python -m dolle_sim verify`, `tests/test_notebooks.py`.
- **References:** `commit:a3fffe8`; `check:damping` (e = 0.8501 for e_dry = 0.85);
  `notebooks/01_two_grains_colliding.ipynb` section 3; `PROVENANCE.md` ("Viscoelastic damping").

### B4. Plastic yield (Thornton & Ning 1998), with unloading on the flattened contact
- **Decision:** a contact yields when the peak Hertz pressure reaches the yield pressure p_y; it then loads along
  the Thornton & Ning (1998) line, unloads elastically on a flattened radius with a permanent dent, and its later
  pull-off uses the flattened radius. Damping does not act on a separated contact.
- **Date:** 2026-10-20; post-yield unloading fixed 2026-10-20 (v0.2 review) · **Decided by:** Engineering; Review
  (the post-yield damping fault) · **Status:** in force.
- **Alternatives considered:** a purely elastic JKR contact.
- **Why:** fast impacts (a release kick) dent the grains and lose restitution. Combining yield with adhesion
  (F_el = F_JKR(a) + F_max(delta) - F_Hertz(delta) while loading beyond yield) is the simulator's own choice
  (`dolle_core/plastic.py` docstring).
- **Effect on results:** none on the thresholds: yield starts at 11.1 m/s (silica pairs), 0.84 m/s (graphite
  pairs), 19 m/s (silica on borosilicate), 0.57 m/s (graphite on borosilicate) (`dolle_core.plastic.yield_speed`);
  thresholds with and without yield agree to all digits. It matters for kicks at ~1.2-1.5 m/s. The raised pull-off
  after yield is not tested.
- **How to change it:** `make_pair(..., plastic=False)` (`dolle_core.pair.make_pair`); `"yield_pressure": null` in
  a config material override (never yields).
- **References:** `commit:a3fffe8`, `commit:8060592`; `check:thornton_ning`, `check:plastic_damping`.

### B5. Stick or bounce is decided by four tests, deterministically
- **Decision:** an impact sticks if the relative kinetic energy is below the hysteresis energy at touch, if the
  grains fall back into the contact, or if kinetic plus stored energy is negative; it bounces once the overlap
  passes the break overlap. One impact speed gives one outcome; probabilities come only from scatter in radius and
  surface energy.
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** a probabilistic sticking law (not used).
- **Why:** the contact law is deterministic; real scatter enters through the grains' properties (C8).
- **Effect on results:** the sticking curves of `python -m dolle_sim sweep` (defaults `--radius-sigma 0.05
  --gamma-sigma 0.3`, ASSUMED amplitudes) come entirely from that scatter. Two engine-specific rules: in the v0.5
  engine a rebound slower than load x time step under a sustained load counts as resting (never acts in microgravity,
  where it is ~4 um/s); in the v0.6 engine a contact held 2 ms is a stick.
- **How to change it:** `dolle_core.collide.collide_pair` (mirrors `dolle_sim.contact._resolve_core`);
  `dolle_sim.simulation.ChamberSim._comes_to_rest`; `dolle_extras.engine.STICK_HOLD_S`.
- **References:** `commit:a3fffe8`; `notebooks/01_two_grains_colliding.ipynb`.

### B6. Only the normal impact speed decides; a head-on 1-D impact gives the threshold
- **Decision:** stick or bounce depends only on the speed along the line of centres; the threshold is computed from
  a head-on one-dimensional impact. In 3-D the 1-D model is run with the rigid-body effective mass 1/(n.K.n) of the
  contact point, and impacts are instants.
- **Date:** 2026-10-20 (v0.5); 2026-11-11 (3-D core) · **Decided by:** Engineering · **Status:** in force (v0.5
  engine, `dolle_core`, 3-D core).
- **Alternatives considered:** resolving oblique impacts with sliding and rolling springs (the v0.6 contact model
  does).
- **Why:** checked against the 3-D contact model: head-on 3-D against 1-D agrees to 1.6e-3 (time-step error; 5e-4
  at half the step); oblique offsets of r/2 and r give the head-on sticking speed to 1e-4
  (`notebooks/01_two_grains_colliding.ipynb` section 12). The chance of a third grain arriving during a ~1 us
  contact is ~6e-6 in the densest 3-D cloud (`notebooks/06_the_3d_simulation.ipynb`).
- **Effect on results:** grazing and spinning impacts and other offsets are not checked: ASSUMED beyond the tested
  cases. An off-centre hit on a clump has a smaller effective mass and therefore a higher threshold.
- **How to change it:** only by moving to the v0.6 engine (`python -m dolle_extras mission`).
- **References:** `commit:a3fffe8`, `commit:75fa4e8`, `commit:784e16a`; `check:dem_vs_1d`.

### B7. Mixing rules for unlike pairs
- **Decision:** reduced modulus 1/E* = sum of (1 - nu^2)/E; reduced radius and mass as usual (a wall has infinite
  radius and mass); work of adhesion w = 2 sqrt(gamma1 gamma2) (geometric mean, ASSUMED); dry restitution
  e = sqrt(e1 e2) (ASSUMED); yield pressure the smaller of the two (ASSUMED).
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** arithmetic w = gamma1 + gamma2; Owens-Wendt with dispersive and polar parts; the
  smaller e_dry; measuring w directly for each pair.
- **Why:** common practice; not checked by simulation (every check uses like materials). Only the reduced modulus
  is standard (Hertz 1882).
- **Effect on results (measured):** they matter only for grain-wall contacts. Arithmetic instead of geometric w:
  silica on borosilicate 6.66 to 6.70 mm/s, but graphite on borosilicate 18.2 to 21.5 mm/s (w 0.078 to 0.095 J/m^2,
  pull-off +22%). The smaller e_dry instead of sqrt(e1 e2): +7% (silica on the wall), +14% (graphite). The yield rule
  has no effect on thresholds.
- **How to change it:** `dolle_core.pair.work_of_adhesion` and `dolle_core.pair.make_pair`, AND
  `dolle_sim.materials.pair_props`, AND, for the v0.6 engine, `dolle_extras.engine.FullContactSim._params` and
  `dolle_extras.dem.DEMSystem._params` (E7; `dolle_extras.dem.make_params` only receives the mixed w and e_dry as
  arguments). Re-run `tests/test_dolle_core.py` (reference numbers change on purpose), `tests/test_core_vs_dem.py`,
  `tests/test_dolle_core_sim3d.py`, `python -m dolle_sim verify`.
- **References:** `commit:a3fffe8`; `notebooks/03_silica_vs_carbon.ipynb`.

### B8. Default engine: wall grains are held still until pulled off
- **Decision:** in the v0.5 engine a grain on a wall is frozen until the load pulling it away exceeds the JKR
  pull-off force F_c = 1.5 pi w R; a wall-bound clump fails by tree statics, each bond in tension only. Rolling and
  sliding along the wall are not modelled (and stated as such).
- **Date:** 2026-10-20 (grains that stick to a wall are frozen, `commit:a3fffe8`); 2026-10-22 (released when the
  load exceeds F_c, `commit:a866c4e`) · **Decided by:** Engineering · **Status:** in force for the default engine;
  rolling and sliding are resolved by the v0.6 engine (E3) and by the wall limits of B9.
- **Alternatives considered:** rolling and sliding contacts (v0.6 engine).
- **Why:** cheap and exact for what it models: `check:detachment` (exact counts, release speeds to 2e-16 m/s);
  LAMMPS pull-off within 0.14% (D10).
- **Effect on results:** rolling is the weakest limit (1.29 g silica, 3.59 g graphite on borosilicate, B9), and the
  spin (~9.6 g), the lateral ascent load (~10 g) and re-entry (~7.9 g sideways) all exceed it. So in default runs
  wall dust stays put during launch by assumption; only the v0.6 engine shows the rolling.
- **How to change it:** `python -m dolle_extras mission configs/mission_quarter_deck.json --grains 100`; the rule
  itself is `dolle_sim.simulation.ChamberSim._check_detachment` and
  `dolle_sim.simulation.ChamberSim._detach_aggregate`; `contact.wall_sticking` (`"jkr"`, `"never"`, `"always"`).
- **References:** `commit:a3fffe8`, `commit:a866c4e`; `extras/README.md`; `docs/MISSION_RUNS.md`.

### B9. Wall limits: pull-off, sliding, rolling and twisting
- **Decision:** a grain on a wall has four limits: pull-off F_c = 1.5 pi w R (JKR); sliding at
  mu |F_ne + 2 F_c| with spring 8 G* a0 (Wada et al. 2007; Thornton & Yin 1991); rolling at the displacement
  xi_crit = h a0/12 with spring 4 F_c/R (Dominik & Tielens 1995; Krijt et al. 2014); twisting with spring
  (16/3) G a0^3 (Wada et al. 2007). These are the formulas of the v0.6 engine and of `dolle_core.wall.wall_limits`.
- **Date:** 2026-10-28 · **Decided by:** Engineering · **Status:** in force (v0.6 engine, `dolle_core`, notebook 04).
- **Alternatives considered:** friction mu F_ne without the adhesion term (zero friction at zero load, which
  contradicts Heim et al. 1999); the LAMMPS "twisting marshall" option (a unit error in its source: not used).
- **Why:** the published contact models for adhesive spheres, identical to LAMMPS (D10).
- **Effect on results (measured, 50 um radius):** silica on clean borosilicate: pull-off 1,034 g, sliding 827 g,
  rolling 1.29 g; on silanised glass 598, 479 and 0.62 g; graphite on borosilicate 1,628, 1,628 and 3.59 g.
  Pull-off and sliding scale with the adhesion factor, rolling as (factor)^(4/3) and linearly with h (0.13 to
  1.55 g for silica over h = 0.05 to 0.6); sliding scales with mu. A 25 g pressing load raises pull-off by 25 g and
  rolling by ~1%. The conclusion "dust cannot be pulled off at launch but can roll" rests on these numbers.
- **How to change it:** arguments `f_adh`, `mu`, `h`, `press_g` of `dolle_core.wall.wall_limits`;
  `dolle_core.wall.critical_rolling_displacement`. For the v0.6 engine, without editing code: `contact.friction`,
  `contact.rolling_hysteresis`, `contact.rolling_q`, `contact.twisting_q` (section 1); the limit formulas
  themselves (xi_crit = h a0/12, the sliding and twisting springs, the damping) are in `dolle_extras.dem.make_params`.
  Re-run `tests/test_dolle_core.py` (`test_wall_reference_numbers`), `tests/test_core_vs_dem.py`
  (`test_wall_limits_equal_make_params`, `test_rolling_limit_reproduces_check_rolling_on_wall`),
  `tests/test_notebooks.py`. The launch-load report uses a different sliding formula (F20).
- **References:** `commit:75fa4e8`, `commit:90a0058`; `check:rolling_on_wall` (onset 1.29 g, 3.2e-3 against the
  rigid-plastic solution); `notebooks/04_grain_on_a_wall.ipynb`.

### B10. Vibrating walls: the impact phase is drawn from the flux-weighted distribution
- **Decision:** a vibrating wall is modelled by its velocity only (not its displacement); the wall speed at an
  impact is drawn from the phase distribution weighted by the impact flux.
- **Date:** 2026-10-20 (v0.2) · **Decided by:** Review (grains passed through vibrating walls in v0.1), Engineering
  (fix) · **Status:** in force, for the flight demo's vibrating walls AND for every release window of the mission
  runs (shaker and kick scenarios): `dolle_sim.simulation.ChamberSim._update_time` sets the wall vibration from the
  release window and `dolle_sim.simulation.ChamberSim._sample_wall_phase` draws the wall speed for impacts on the
  shaking walls. The `rough_grains` and `kick` results in `docs/MISSION_RUNS.md` depend on it.
- **Alternatives considered:** deterministic moving walls (not compared).
- **Why:** the derived distribution of phases at which grains actually meet a moving wall; it never lets a grain
  through.
- **Effect on results:** `check:walls`: 0 penetrations over 501 bounces.
- **How to change it:** `wall_vibration` (amplitude, frequency, axes) in a chamber config;
  `dolle_sim.simulation.ChamberSim._sample_wall_phase`.
- **References:** `commit:8060592`.

### B11. Impacts onto wall-bound grains are kept out of the grain-grain statistics
- **Decision:** an impact onto a grain or clump held by a wall is flagged and excluded from the grain-grain
  statistics.
- **Date:** 2026-10-20 (v0.2) · **Decided by:** Review, Engineering · **Status:** in force.
- **Alternatives considered:** counting them as ordinary grain-grain impacts (the v0.1 behaviour).
- **Why:** a wall-held target acts like twice the effective mass, so such impacts stick at ~1/sqrt(2) of the
  grain-grain speed.
- **Effect on results:** mixing them in biased the "true" thresholds ~7-15% low (`docs/HISTORY.md`, v0.2).
- **How to change it:** not a switch; `dolle_sim.simulation.ChamberSim._resolve_pp`.
- **References:** `commit:8060592`.

### B12. Grains that stick become rigid clumps
- **Decision:** grains that stick merge into one rigid body that conserves momentum and angular momentum and can
  spin; a clump that sticks to a wall is frozen there.
- **Date:** 2026-10-20 (v0.5 engine); 2026-11-11 (3-D core) · **Decided by:** Engineering; reaffirmed by the Owner
  for the 3-D page (E9) · **Status:** in force for the v0.5 engine and the 3-D core; the v0.6 engine has
  no clumps (every contact stays a contact).
- **Alternatives considered:** full contact dynamics of every contact (v0.6 engine; for the 3-D page ~4e9 steps
  for 12 s, far too slow for a browser); merging into one sphere (what REBOUND offers); hard spheres with a fixed
  restitution (nothing could stick).
- **Why:** makes 15-minute runs of thousands of grains cheap, with exact momentum and spin bookkeeping.
- **Effect on results:** momentum to round-off (`check:conservation`, 1e-16). The cost: no rolling, sliding or
  restructuring inside a clump (B17).
- **How to change it:** `dolle_core.clumps.merge`, `dolle_sim.simulation.ChamberSim._merge`; for deformable
  aggregates use the v0.6 engine.
- **References:** `commit:a3fffe8`, `commit:784e16a`; `notebooks/06_the_3d_simulation.ipynb` section 5.

### B13. The bounce impulse: collision matrix, Coulomb cone, no tangential rebound, energy guard
- **Decision:** a bounce between rigid bodies uses the full 3x3 collision matrix at the contact point (Stronge
  2000), the normal restitution from the 1-D contact model, a Coulomb friction cone and tangential restitution
  e_t = 0 (ASSUMED); if friction would add kinetic energy (Kane's paradox on eccentric clump impacts), the
  frictionless impulse is used and counted.
- **Date:** 2026-10-20 (v0.2: full matrix); 2026-10-21 (v0.3: energy guard) · **Decided by:** Review (missing
  off-diagonal terms; energy created by friction), Engineering · **Status:** in force (v0.5 engine and 3-D core).
- **Alternatives considered:** a diagonal-only impulse (v0.1); e_t > 0.
- **Why:** couples spin and translation correctly; the guard stops friction creating energy.
- **Effect on results:** `check:impulse` (6.5e-16, inside the cone); `check:friction_energy` (2,000 eccentric
  impacts, largest change -2.8e-3, never positive). Friction changes only the tangential part of a bounce
  (coefficients: C9).
- **How to change it:** `contact.tangential_restitution` (v0.5); field `e_t` of `dolle_core.sim3d.Tubule`; code
  `dolle_core.impacts.bounce_impulse`, `dolle_sim.simulation.ChamberSim._impulse`.
- **References:** `commit:8060592`, `commit:ed2f49d`, `commit:784e16a`.

### B14. Clumps break by the Dominik & Tielens (1997) energy thresholds, paid from the energy really dissipated
- **Decision:** a clump is eroded (only the struck grain leaves) above 3 N_c E_break and shattered (every clump
  involved breaks into single grains) above 10 N_c E_break, with N_c = N - 1 and E_break = 1.54 F_c |delta_c|
  (Chokshi et al. 1993). Broken bonds are paid only from the energy the bounce actually removed,
  E_diss = P.v_rel - P.K.P/2: if it cannot pay for a shatter the struck grain is eroded, if not even that nothing
  breaks.
- **Date:** 2026-10-20 (thresholds); 2026-10-20 (v0.2: the clump's own bond energy); 2026-10-21 (v0.3: E_diss) ·
  **Decided by:** Engineering; Review (energy faults) · **Status:** in force (v0.5 engine, 3-D core).
- **Alternatives considered:** no fragmentation; the full translational collision energy (v0.2; gave fragments up
  to ~1/3 of the impact energy more than the bounce dissipated, see H4); full contact dynamics (v0.6).
- **Why:** published thresholds, but derived for aggregates of sub-micron grains that roll and restructure: for
  100 um clumps of 2-3 grains this is an EXTRAPOLATION (rolling costs 25x (graphite) to 61x (silica) E_break,
  `notebooks/06_the_3d_simulation.ipynb`). The thresholds themselves have no independent check.
- **Effect on results:** `check:fragmentation_energy` (kinetic plus bond energy never increases: largest 1e-13 over
  1,169 impacts with 95 fragmentations). In the 3-D page run, switching fragmentation off changes nothing
  (silica 20 impacts / 4 sticks, graphite 17/17; only 1-2 break-ups per run).
- **How to change it:** `contact.fragmentation` (v0.5 config); field `fragmentation` of `dolle_core.sim3d.Tubule`;
  code `dolle_core.impacts.fragment`, `dolle_sim.simulation.ChamberSim._fragment`. Re-run
  `tests/test_dolle_core_sim3d.py`, `python -m dolle_sim verify`.
- **References:** `commit:a3fffe8`, `commit:8060592`, `commit:ed2f49d`.

### B15. The energy given to fragments is a stated assumption
- **Decision:** after a break-up the fragments separate with the fraction `contact.fragment_separation` (default 1,
  all) of the energy left after paying the bonds; 0 (all of it heat) is the opposite bound and is available.
- **Date:** 2026-11-11 · **Decided by:** Engineering · **Status:** in force; ASSUMED (O8).
- **Alternatives considered:** any fraction between 0 and 1; how a real impact splits E_diss is unknown (the 1-D
  contact counts it all as heat).
- **Why:** keeps the v0.5 rule and makes the assumption visible, with both bounds runnable.
- **Effect on results:** on the 3-D start a fragment gets 1-6 E_break, 3-10 mm/s. Bound 0 instead of 1: 6 seeds x
  100 grains with the v0.5 engine change impacts/sticks from 97/21 to 98/22 (silica) and 87/82 to 88/83
  (graphite); the 3-D page run from 20/4 to 22/6 (silica) and 17/17 to 18/18 (graphite).
- **How to change it:** `contact.fragment_separation` (v0.5); field `separation` of `dolle_core.sim3d.Tubule`.
  Re-run `tests/test_dolle_core_sim3d.py` (`test_separation_bound_in_both_engines`).
- **References:** `commit:13dd8e8`, `commit:784e16a`; `notebooks/06_the_3d_simulation.ipynb` sections 7 and 9.

### B16. Each clump stores the pairs that stuck (no distance test)
- **Decision:** every clump keeps the list of grain pairs that stuck; erosion pays for the struck grain's real
  bonds, and a clump splits only where bonds are cut.
- **Date:** 2026-11-11 · **Decided by:** Engineering · **Status:** in force (v0.5 engine and 3-D core).
- **Alternatives considered:** the old distance test (a bond exists if the gap is below 1e-6 of the radius sum).
- **Why:** a grain sticking to a spinning clump is merged where the straight-line contact search put it, up to
  1.25e-3 of its distance from the clump's centre of mass from touching (0.1-2 nm gaps in the 3-D page's graphite
  trimers); the distance test missed such bonds, so an intact trimer could split without an impact.
- **Effect on results:** the 3-D page replay, the 26 model checks and the core-vs-v0.5 comparisons are unchanged;
  the stored v0.5 results predate the fix (G10).
- **How to change it:** `dolle_core.clumps.bond_groups`, `dolle_core.clumps.split_disconnected`. Re-run
  `tests/test_dolle_core_sim3d.py` (`test_clumps_are_held_by_their_stored_bonds`,
  `test_v05_bodies_are_held_by_their_stored_bonds`).
- **References:** `commit:13dd8e8`, `commit:784e16a`; `docs/HISTORY.md` (v0.7).

### B17. Clumps stay rigid: restructuring is not modelled
- **Decision:** no rolling or sliding inside a clump.
- **Date:** 2026-11-11 · **Decided by:** Engineering · **Status:** open (O7).
- **Alternatives considered:** test it with the v0.6 contact model, which resolves rolling at every contact.
- **Why:** a 15 mm/s impact carries 0.03 (silica) and 0.005 (graphite) of the 5 E_roll restructuring energy of
  Dominik & Tielens (1997), but the rolling resistance is uncertain by ~50x (C10), so for silica the margin could
  fall below 1.
- **Effect on results:** unknown until tested; it would change clump shapes, not the grain-grain sticking speeds.
- **How to change it:** run the scene with the v0.6 contact model (`extras/dolle_extras/dem.py`).
- **References:** `commit:784e16a`; `notebooks/06_the_3d_simulation.ipynb` section 9.

### B18. Gas drag: Stokes with the Cunningham slip correction; the gas co-rotates
- **Decision:** drag is Stokes drag with the Cunningham slip correction (constants 1.257, 0.4, 1.1 of Davies 1945),
  which bridges to the Epstein limit at low pressure; the gas co-rotates with the chamber; the gas torque on a
  spinning grain joins the two exact limits with this project's own interpolation; a clump's drag is the sum of
  its grains' (no shielding).
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force (`dolle_sim`; left out of the 3-D core,
  B22).
- **Alternatives considered:** specular reflection of gas molecules (28% less drag; the real surface interaction
  is unknown).
- **Why:** the standard drag law over the whole pressure range.
- **Effect on results:** at the mission's 1e-3 Pa the drag time is 1.3-1.5e4 s, so speeds change by less than 0.1%
  in 12 s (`notebooks/06_the_3d_simulation.ipynb` section 9). It matters in the `gas_100pa` and `gas_1atm`
  scenarios. The torque interpolation has no automated check.
- **How to change it:** `gas.pressure_pa` (0 switches drag off), `gas.gas`, `gas.temperature_k`;
  `dolle_sim.environment.Environment.drag_coeff`.
- **References:** `commit:a3fffe8`; `check:drag_limits` (Stokes exact, Epstein -2.5%).

### B19. Contact charging: a condenser model that saturates
- **Decision:** at each contact, charge flows across a ~0.5 nm gap driven by the work-function difference, capped
  when the potentials equalise (review by Matsusaka et al. 2010). On in both shipped configs.
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force (`dolle_sim`; left out of the 3-D core).
- **Alternatives considered:** no charging (the code default when the key is absent).
- **Why:** charging changes trajectories in a cloud with nothing to screen it.
- **Effect on results:** the work functions are ASSUMED (C11). Silica (5.0 eV) on borosilicate (5.0 eV) does not
  charge at all; graphite (4.6 eV) charges to ~0.4 V, and two such grains repel with ~1e-3 of the impact energy at
  graphite's sticking speed. With charging, the v0.5 engine changes no grain-grain outcome in 6 seeds x 100 grains
  (`notebooks/06_the_3d_simulation.ipynb` section 9). Same-material charging is not modelled; absolute charges are
  not validated.
- **How to change it:** `contact.contact_charging`, `contact.charging_cutoff_distance_m`; work functions in the
  material table (C11).
- **References:** `commit:a3fffe8`; `check:charging` (saturates to 0.1%).

### B20. Long-range electric fields: Ewald split with grounded walls
- **Decision:** Coulomb forces are an exact short-range pair sum (cut off at 3 mm) plus a particle-mesh Poisson
  solution with zero potential on the chamber walls; polarisation and image charges are included.
- **Date:** 2026-10-20 (v0.2) · **Decided by:** Review (the long-range field was missing), Engineering · **Status:**
  in force (`dolle_sim`).
- **Alternatives considered:** pair sum only (misses the space charge of a one-signed cloud).
- **Why:** contact charging leaves the cloud with one sign, and nothing screens it at low pressure.
- **Effect on results:** `check:space_charge` (1.3%; image factor 0.92). Zero potential on the walls matches only a
  grounded conducting inner surface: for bare glass this geometry is ASSUMED.
- **How to change it:** `forces.coulomb`, `forces.polarization`, `forces.image_charges`, `forces.space_charge`,
  `forces.long_range_cutoff_m`, `forces.space_charge_update_every`; `dolle_sim.electrostatics.SpaceChargeSolver`.
- **References:** `commit:8060592`.

### B21. Negligible forces are switched off by default
- **Decision:** mutual gravity, the Lorentz force, an external electric field, photophoresis, radiation pressure,
  Brownian motion and the gravity gradient stay in the code but are off by default; the pre-flight check still
  prints every term, marks the switched-off ones "(off)" and warns if one exceeds 5% of the residual acceleration or
  a Brownian speed exceeds 1% of the sticking speed (both thresholds ASSUMED). The v0.6 checks switch them on so
  their values are unchanged.
- **Date:** 2026-11-11 (v0.7) · **Decided by:** Owner (each specific choice, agreed feature by feature with the
  coding sessions), prompted by the Team reviewer's request of 2026-11-10 to strip extra features (G7) ·
  **Status:** in force; reverses A4 in part (H23). Thermophoresis is NOT in this list and stays on (B23).
- **Alternatives considered:** keep everything on (A4); delete the code.
- **Why:** readability; each is tiny. For the flight demo config against its 1e-4 g residual: radiation pressure
  2.2% (graphite) and 0.85% (silica), photophoresis 0.18% and 0.78%, gravity gradient 0.1%, mutual gravity ~7e-13 g,
  Lorentz ~5e-13 g; Brownian speed ~2 um/s, 3e-4 of silica's sticking speed (`README.md`, from
  `python -m dolle_sim preflight`).
- **Effect on results:** none on thresholds; the random stream changes (Brownian off), so a re-run of a stored
  result gives different numbers (G10).
- **How to change it:** `forces.self_gravity`, `forces.lorentz`, `forces.external_e_field`,
  `forces.photophoresis`, `forces.radiation_pressure`, `forces.brownian`, `environment.gravity_gradient`; the list
  is `dolle_sim.simulation.FORCES_OFF_BY_DEFAULT`; warnings `dolle_sim.design.OFF_WARN_FRACTION_OF_RESIDUAL` and
  `dolle_sim.design.OFF_WARN_BROWNIAN_FRACTION_OF_V_STICK`; v0.6 checks `dolle_extras.verify_full.FORCES_V062`.
  Re-run `tests/test_defaults.py` (`test_shipped_configs_follow_the_defaults`).
- **References:** `commit:13dd8e8`; `tests/test_defaults.py`. The free-molecular phoretic formulas warn below
  Knudsen number 10 (`commit:ed2f49d`).

### B22. What the 3-D core leaves out: gas drag, payload roll, charging
- **Decision:** the 3-D simulation of `dolle_core` has no gas drag, no payload roll (no Coriolis or centrifugal
  terms) and no charging.
- **Date:** 2026-11-11 · **Decided by:** Engineering · **Status:** in force; adding the roll is open (O6).
- **Alternatives considered:** include each (the v0.5 engine has all three).
- **Why:** each was sized: drag time ~1.4e4 s; charging repulsion ~1e-3 of the impact energy (B19). The v0.5 engine
  with and without drag or charging gives the same grain-grain outcomes on 6 seeds x 100 grains. The 1 rpm roll
  moves where grains land; its effect on grain-grain outcomes is within 2 sigma (silica impacts / sticks 97/21
  without it, 96/27 with it; graphite 87/82 and 91/86) and is not shown to be nil
  (`notebooks/06_the_3d_simulation.ipynb` section 9: "checked only to that level").
- **Effect on results:** grain-grain outcomes agree within the statistics above; for the roll that is a check to
  2 sigma, not a proof of no effect.
- **How to change it:** add the terms to `dolle_core.sim3d.kick`; compare with
  `tests/test_dolle_core_sim3d.py` (`test_left_out_effects_do_not_change_grain_grain_outcomes`).
- **References:** `commit:784e16a`; `notebooks/06_the_3d_simulation.ipynb` section 9.

### B23. Thermophoresis is on by default; the flight demo sets an ASSUMED gradient of 3 K/m
- **Decision:** the thermophoretic force (Waldmann's free-molecular law, F = -(32/15) R^2 kappa_tr grad T / v_th,
  code comment in `dolle_sim/simulation.py`) is on by default: `forces.thermophoresis` defaults to true and it is not
  in `dolle_sim.simulation.FORCES_OFF_BY_DEFAULT`. It acts only where a temperature gradient is set: the flight demo
  config sets `environment.temperature_gradient_k_m` = [0, 0, 3] K/m at 1 Pa (ASSUMED); the mission config sets
  none, so it is zero in mission runs.
- **Date:** 2026-10-20 (`commit:a3fffe8`, the same 3 K/m in the first demo config); kept on in v0.7 · **Decided
  by:** Owner (realism, A4), Engineering (formula and value) · **Status:** in force.
- **Alternatives considered:** switch it off with the negligible forces (B21; not done, because it is not
  negligible); a gradient of zero (an isothermal chamber).
- **Why:** at 1 Pa and 3 K/m it gives 3.2e-5 g (graphite) and 3.5e-5 g (silica), about a third of the 1e-4 g
  residual acceleration (`python -m dolle_sim preflight configs/flight_carbon_vs_silica.json`). The user guide lists
  it as a design finding: each kelvin across a 3 cm chamber at 1 Pa drives ~3.6e-4 g, 3.6x the residual, so the
  chamber should be isothermal to about 0.1 K (`docs/USER_GUIDE.md`, design findings). The owner's notes count
  temperature gradients among the main experimental systematics.
- **Effect on results:** `check:thermophoresis` (the drift against the standard free-molecular result: ratio 1.028).
  The gradient is ASSUMED; the free-molecular law holds only at large Knudsen number (the pre-flight check warns
  below 10, `commit:ed2f49d`), and its wall temperature-jump reduction is an uncited approximation
  (`docs/USER_GUIDE.md`).
- **How to change it:** `forces.thermophoresis` (true or false); `environment.temperature_gradient_k_m` (K/m, a
  vector) in a chamber config; the force is computed in `dolle_sim.simulation.ChamberSim`. Re-run
  `python -m dolle_sim preflight` for its size.
- **References:** `commit:a3fffe8`, `commit:13dd8e8`; `check:thermophoresis`; `docs/USER_GUIDE.md`.

## C. Material values

The values live in two tables kept identical on purpose, `dolle_core/materials.py` (readable) and
`dolle_sim/materials.py`; the v0.6 engine reads its material values from the `dolle_sim` chamber it is built on
(`dolle_sim.materials`) and mixes them for each pair in `dolle_extras.engine.FullContactSim._params` and
`dolle_extras.dem.DEMSystem._params` (E7). Units: density rho in kg/m^3,
Young's modulus E in GPa, surface energy gamma in J/m^2 per surface (two like surfaces hold with w = 2 gamma),
yield pressure p_y in GPa. All sensitivities below were measured on 2026-11-16 with the repository's functions
(`dolle_core.collide.sticking_threshold`, `dolle_core.wall.wall_limits`), 50 um radius: "log-sensitivity" is the
% change of the sticking speed per 1% change of the input. To repeat one:

```python
from dataclasses import replace
import dolle_core as dc
si = replace(dc.MATERIALS["silica"], gamma=0.2)            # any field of any material
print(dc.sticking_threshold(dc.make_pair(si, 50e-6, si, 50e-6)) * 1e3, "mm/s")   # 35.0
```

### C1. Silica: gamma, E and nu from the published quartz values; density ASSUMED
- **Decision:** silica gamma 0.025 J/m^2, E 54 GPa, nu 0.17 (the quartz values used by Chokshi et al. 1993 and
  Dominik & Tielens 1997); rho 2000 kg/m^3 ASSUMED.
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force as a placeholder: which silica flies is
  not decided (G15).
- **Alternatives considered:** measured in air 0.014-0.025 J/m^2; Kimura et al. (2015) argue 0.15-0.25 J/m^2 for
  dry silica in vacuum; Kreuzig et al. (2024) find no such rise for hydrophobic spheres; fused silica's E, nu, rho
  (lower the threshold 14%, to 5.33 mm/s, `notebooks/03_silica_vs_carbon.ipynb`).
- **Why:** the values of the published dust-collision work, so results compare with the literature.
- **Effect on results:** log-sensitivities: gamma +0.83, rho -0.50, E -0.33, nu -0.02. gamma = 0.2 J/m^2 (Kimura)
  gives 35.0 mm/s instead of 6.18 (x5.7), as sticky as graphite.
- **How to change it:** the `silica` entry of `dolle_core.materials.MATERIALS` AND `dolle_sim.materials.MATERIALS`
  (or a config override `"materials": {"silica": {"gamma": 0.1}}`); update the hard-coded numbers in
  `test_threshold_reference_numbers` and `test_wall_reference_numbers` and every document that quotes them.
- **References:** `commit:a3fffe8`, `commit:90a0058`; `PROVENANCE.md`.

### C2. Graphite: gamma, E, nu and density from Chokshi et al. (1993)
- **Decision:** graphite gamma 0.075 J/m^2 (basal plane), E 10 GPa, nu 0.32, rho 2200 kg/m^3 (Chokshi et al. 1993,
  Table 3). Graphite stands in for "carbon".
- **Date:** 2026-10-20; nu corrected to 0.32 on 2026-10-21 (claim audit) · **Decided by:** Engineering; Review (nu)
  · **Status:** in force as a placeholder: which carbon flies is not decided (G14).
- **Alternatives considered:** the cleavage energy of graphite (Wang et al. 2015) implies ~0.19 J/m^2; glassy and
  amorphous carbon (bands in `notebooks/03_silica_vs_carbon.ipynb`).
- **Why:** the published reference values for carbon grains.
- **Effect on results:** log-sensitivities: gamma +0.83, rho -0.50, E -0.33, nu -0.08. gamma = 0.19 J/m^2 gives
  83.2 mm/s instead of 38.4 (x2.2). Threshold bands: graphite 22-220 mm/s, glassy carbon 20-127, amorphous carbon
  11-153 (ranges mostly ASSUMED).
- **How to change it:** the `graphite` entry of both material tables (as C1).
- **References:** `commit:a3fffe8`, `commit:ed2f49d`.

### C3. Dry restitution e_dry: ASSUMED for every material
- **Decision:** e_dry = 0.85 (silica), 0.70 (graphite), 0.8 (every glass); restitution independent of speed.
- **Date:** 2026-10-20 · **Decided by:** Engineering (representative values) · **Status:** in force; the largest
  single uncertainty (O1).
- **Alternatives considered:** measure it from the restitution of non-sticking impacts (the flight or a ground
  test; `notebooks/03_silica_vs_carbon.ipynb`).
- **Why:** no measurement exists for these grains.
- **Effect on results:** the damping factor f(e_dry), by which the undamped threshold is multiplied (300 sub-steps,
  `dolle_core.scaling.damping_factor`):

  | e_dry | 0.3 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 | 0.99 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | f | 9.56 | 4.44 | 3.27 | 2.461 | 2.14 | 1.86 | 1.615 | 1.39 | 1.19 | 1.04 |

  Log-sensitivity -2.5 (silica at 0.85) and -2.0 (graphite at 0.70): a 5% change of e_dry moves the threshold by
  roughly 10-12%.
  0.85 to 0.90 moves silica from 6.18 to ~5.3 mm/s. Of the x6.2 graphite/silica ratio, x1.52 is f(0.70)/f(0.85);
  equal e_dry would give x4.1. Damping supplies ~62-83% of the threshold energy.
- **How to change it:** field `e_dry` of each material in both tables, or a config override. A material defined in
  a config without `e_dry` silently gets 0.8 (`dolle_sim.materials.Material` default, ASSUMED). Re-run
  `tests/test_dolle_core.py` (reference numbers), `tests/test_notebooks.py`, `python -m dolle_sim verify`.
- **References:** `commit:a3fffe8`, `commit:ed2f49d` (audit wording); `notebooks/03_silica_vs_carbon.ipynb`;
  `tests/test_dolle_core.py` (`test_damping_factor_depends_on_e_dry_only`).

### C4. Yield pressures: ASSUMED
- **Decision:** p_y = 3 GPa (silica), 0.3 GPa (graphite), 5 GPa (glasses); "~1.6x the uniaxial yield stress"
  (`dolle_sim/materials.py` docstring).
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** no yield (B4).
- **Why:** representative values; only fast impacts reach yield.
- **Effect on results:** none on thresholds (log-sensitivity 0.000); yield speeds in B4. It affects the rebound of
  kicked grains (graphite yields above 0.84 m/s; a kick releases at ~1.2-1.5 m/s).
- **How to change it:** field `yield_pressure` in both tables; `null` in a config override disables yield.
- **References:** `commit:a3fffe8`.

### C5. Other dust materials are placeholders
- **Decision:** amorphous carbon, a 50/50 carbon-silica mix, forsterite and alumina are in the table with ALL values
  ASSUMED; water ice uses gamma from Gundlach et al. (2011) and E, nu from Chokshi et al. (1993), for reference only.
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force (not used by any shipped config).
- **Alternatives considered:** leave them out.
- **Why:** lets a user try other dusts; the mix uses rough intermediate values, not a mixing rule.
- **Effect on results:** none on the shipped results.
- **How to change it:** both material tables.
- **References:** `commit:a3fffe8`.

### C6. Glass walls: elastic constants from data sheets, adhesion from measured bead-on-glass values
- **Decision:** borosilicate E 63 GPa, nu 0.20, rho 2230 (DURAN borosilicate 3.3 data sheet); fused silica from the
  HPFS 7980 data sheet; soda-lime from a compilation (lower confidence). The wall gamma is an EFFECTIVE value
  (0.0203 J/m^2) chosen so that, with the mixing rule, silica on clean glass has w = 0.045 J/m^2, measured for
  ~90 um beads on plasma-cleaned glass in room air (Sansao et al. 2021); silanised glass 0.0068 J/m^2 for
  w = 0.026 J/m^2 (Sansao et al. 2021; Zafar et al. 2014).
- **Date:** 2026-10-22 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** a gamma for glass from surface-energy tables (not at the right grain size).
- **Why:** measured values at the right grain size.
- **Effect on results:** sets every wall limit (B9) and the grain-wall sticking speeds (silica 6.66, graphite
  18.2 mm/s). Caveats: measured in room air (dry surfaces in vacuum may be several times stickier: Kimura et al.
  2015; Steinpilz et al. 2019); no graphite-on-glass measurement (the mixing rule gives w = 0.078 J/m^2, B7). A drop
  test (Zafar et al. 2014) on the real glass and powders would measure w directly (O11).
- **How to change it:** the glass entries of both material tables; per tubule `wall_material`.
- **References:** `commit:a866c4e`; `PROVENANCE.md` (glass walls).

### C7. Anti-stick coating and the wall adhesion factor
- **Decision:** a fluorosilane-coated glass (gamma 0.012 J/m^2, ALL values ASSUMED) and a per-chamber factor on the
  grain-wall work of adhesion (`wall_adhesion_factor`, default 1; 0.05 in the flight demo config; 0.02 in the
  `rough_grains` scenario) represent coatings and roughness.
- **Date:** 2026-10-20 (coating, factor); 2026-10-22 (`rough_grains`) · **Decided by:** Engineering (values), Owner
  (grains must bounce off walls, F16) · **Status:** in force; the real coating is not chosen or measured (O11).
- **Alternatives considered:** bare glass; soft PTFE (worse: large contact area, strong tribo-charging);
  `wall_sticking: "never"` (idealised, the pre-flight check warns).
- **Why:** a low-adhesion wall keeps grains bouncing in the demo; the factor is the one knob for unknown surfaces.
- **Effect on results:** a sticking speed scales as factor^(5/6) (the same factor 0.05 on a silica pair would give
  0.51 mm/s instead of 6.18); pull-off and sliding scale with the factor, rolling as factor^(4/3). With 0.02
  (`rough_grains`) the median pull-off falls to about 22 g (T1 silica) and 34 g (T2 graphite) (0.02 x the tubule
  medians of 1,087 g and 1,712 g in `docs/MISSION_RUNS.md`), for silica below the 31 g splashdown, and
  the ground vibration test alone frees an estimated 18-88% of grains. With factor 0.05 the wall contact is in the
  DMT regime where JKR probably overstates sticking (`PROVENANCE.md`).
- **How to change it:** `wall_adhesion_factor` per tubule or chamber; `f_adh` of `dolle_core.wall.wall_limits`;
  the `glass_fluorosilane` entry of both tables.
- **References:** `commit:a3fffe8`, `commit:a866c4e`; `docs/MISSION_RUNS.md`.

### C8. Grain-to-grain scatter (ASSUMED amplitudes)
- **Decision:** each grain's effective surface energy is multiplied by exp(sigma N(0,1)), sigma = 0.3 in the flight
  demo config and 0.6 in the mission config (the ~70% scatter of AFM adhesion measured on clean glass by Rush et al.
  2018); radii scatter 5% (Gaussian); a pair's factor is sqrt(f_i f_j), a grain-wall factor f_i times the wall
  factor.
- **Date:** 2026-10-20 (0.3, 5%); 2026-10-22 (0.6) · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** no scatter (one deterministic threshold).
- **Why:** real grains are rough and differ; the amplitudes are assumptions except the 0.6 (measured spread).
- **Effect on results:** turns the sharp threshold into a sticking curve. In the 3-D page run, no adhesion scatter
  gives silica 19 impacts / 5 sticks instead of 20/4 (graphite 17/17 both).
- **How to change it:** `dust.surface_energy_log_sigma` per tubule; `log_sigma_w` of
  `dolle_core.scenes.two_tubule_start`; `--gamma-sigma`, `--radius-sigma` of `python -m dolle_sim sweep`.
- **References:** `commit:a3fffe8`, `commit:a866c4e`.

### C9. Friction coefficients: a sourced per-pair table, and one global 0.3 in the default engine
- **Decision:** `dolle_core`, the 3-D core and the v0.6 engine use per-pair values: silica-glass 0.4 (Meyer et al.
  2014), silica-silica 0.6 (Liu et al. 2017, their uncoated-silica reference), graphite-graphite 0.5 (Buckley 1971,
  1973); graphite-glass and graphite-silica 0.5 ASSUMED; any other pair 0.4 ASSUMED. The v0.5 engine and the
  launch-load report use one value, 0.3, for every contact (no source: ASSUMED).
- **Date:** 2026-10-20 (0.3, v0.5); 2026-10-28 (table, v0.6); 2026-11-11 (table in the 3-D core) · **Decided by:**
  Engineering · **Status:** in force; the two engines differ (worth stating when comparing them).
- **Alternatives considered:** graphene-coated contacts reach ~0.003 (Liu et al. 2017): a carbon's mu is uncertain
  by more than 100x depending on its surface.
- **Why:** the table was assembled for the v0.6 engine's sliding limit; the v0.5 engine predates it and its 0.3 was
  never changed.
- **Effect on results:** in the 3-D page run mu = 0.3 instead of the table changes nothing (silica 20/4, graphite
  17/17 both ways): friction changes only the tangential part of a bounce. Wall sliding scales with mu (silica on
  borosilicate 207 / 620 / 1,241 g at 0.1 / 0.3 / 0.6), but sliding is never the weakest wall limit. The launch-load
  report differs by more than its mu: it also uses mu (F_c - pull) instead of mu |F_ne + 2 F_c|, so at zero load its
  sliding capacity is 0.3 F_c against 0.8 F_c (silica on glass) and 1.0 F_c (graphite) in `dolle_core` and the v0.6
  engine, 62.5% and 70% lower (F20). Note:
  `notebooks/06_the_3d_simulation.ipynb` calls the 3-D impulse the v0.5 impulse line for line; that is true of the
  formula, not of the default coefficient.
- **How to change it:** table `dolle_core.materials.FRICTION` AND `dolle_extras.engine.MU_DEFAULT`; v0.6 engine
  without editing code: `contact.friction` in its config (for example `{"silica-wall": 0.4}`); v0.5:
  `contact.friction_coefficient`; 3-D core: field `mu` of `dolle_core.sim3d.Tubule`; wall limits: `mu` of
  `dolle_core.wall.wall_limits`; launch report: F20. Re-run `tests/test_dolle_core.py` (`test_friction_lookup`),
  `tests/test_core_vs_dem.py` (`test_friction_table_equals_engine`), `tests/test_dolle_core_sim3d.py`.
- **References:** `commit:8060592`, `commit:75fa4e8`, `commit:784e16a`; `dolle_core.impacts.friction`.

### C10. Rolling and twisting parameters: h = 0.5 and Q = 100 ASSUMED
- **Decision:** rolling hysteresis h = 0.5 in xi_crit = h a0/12 (ASSUMED; fitted to ~1 um silica); rolling and
  twisting damping Q = 100 (ASSUMED); twisting limit angle 1/(16 pi) (Wada et al. 2007).
- **Date:** 2026-10-28 · **Decided by:** Engineering · **Status:** in force; h is open (O5).
- **Alternatives considered:** h = 0.05-0.6 measured for 70 um glass beads; models span ~50x.
- **Why:** no measurement for these grains.
- **Effect on results:** rolling limit scales linearly with h: silica on borosilicate 0.13 g (h = 0.05) to 1.55 g
  (0.6), ~12x through h alone; it also sets the restructuring margin of B17.
- **How to change it:** `h` of `dolle_core.wall.wall_limits` and `dolle_core.wall.critical_rolling_displacement`;
  in the v0.6 engine without editing code, `contact.rolling_hysteresis` (h, default 0.5), `contact.rolling_q` and
  `contact.twisting_q` (Q, default 100) in its config (passed to `dolle_extras.dem.make_params`, which computes
  xi_crit = h a0/12). Re-run `tests/test_dolle_core.py`, `tests/test_core_vs_dem.py`, `tests/test_notebooks.py`.
- **References:** `commit:75fa4e8`; `check:rolling_on_wall`.

### C11. Non-mechanical properties and config defaults: ASSUMED
- **Decision:** work function (contact charging), permittivity (polarisation, image charges), thermal conductivity
  and absorptivity (photophoresis, radiation pressure) are representative ASSUMED values. Missing fields of a
  material defined in a config take the defaults e_dry 0.8, work function 5.0 eV, permittivity 4.0, yield pressure
  infinite, thermal conductivity 1.0, absorptivity 0.5 (ASSUMED).
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** require every field (no silent defaults).
- **Why:** these only drive the small forces; the docstring of `dolle_sim/materials.py` states which values are
  published.
- **Effect on results:** the work functions decide whether charging exists at all (B19), not the sticking outcomes.
  The silent e_dry = 0.8 for a new material moves its threshold by f(0.8) = 1.86.
- **How to change it:** `dolle_sim.materials.Material` fields; config `"materials"` overrides.
- **References:** `commit:a3fffe8`.

## D. Numerical methods and verification standards

### D1. Every impact is integrated in time: velocity Verlet, 300 sub-steps per contact
- **Decision:** each impact is integrated at nanosecond steps with velocity Verlet, 300 steps per contact duration
  (the shorter of the Hertz time and an oscillation estimate 1.29x shorter than the true JKR period); the same force
  is applied to both bodies; no outcome is read from a table. The browser app uses 100 steps.
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** explicit Euler (first order); RK4 (four force evaluations, and the jumps at snap-on
  and break limit its accuracy anyway); lookup tables.
- **Why:** one force per step, second order, and momentum exact because both grains receive the same force in each
  half kick (`notebooks/02_momentum_conservation.ipynb`).
- **Effect on results:** silica threshold 6.184 / 6.179 / 6.177 / 6.177 mm/s at 100 / 300 / 1000 / 3000 steps
  (300 is 3.8e-4 above converged); the browser's 100 steps change results <0.4%. Momentum to 1e-15.
- **How to change it:** `n_sub` of `dolle_core.collide.sticking_threshold`; `contact.steps_per_contact` (v0.5);
  field `n_sub` of `dolle_core.sim3d.Tubule`; the browser value in `dolle_sim/webapi.py`. Safety cap 300,000 steps
  (`max_steps`), then reported as a stick.
- **References:** `commit:a3fffe8`, `commit:f5b27a8`; `notebooks/01_two_grains_colliding.ipynb` section 4.

### D2. The threshold is found by bisection to 0.2%
- **Decision:** geometric bisection between 1e-6 and 1 m/s (the upper end tripled while it still sticks), stopping
  at hi/lo - 1 < 2e-3.
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** a finer tolerance (slower).
- **Why:** 0.1% accuracy is far below the material uncertainties.
- **Effect on results:** ~0.1% on every quoted threshold.
- **How to change it:** `tol`, `lo`, `hi` of `dolle_core.collide.sticking_threshold`.
- **References:** `commit:a3fffe8`.

### D3. Free flight: velocity Verlet with exact drag and Coriolis
- **Decision:** kick-drift-kick, with dv/dt = -gamma v - 2 Omega x v solved in closed form in each kick; in the 3-D
  core (no drag, no roll) the kick is the exact parabola.
- **Date:** 2026-10-20 (v0.2) · **Decided by:** Review (v0.1 had first-order Coriolis), Engineering · **Status:**
  in force.
- **Alternatives considered:** first-order Coriolis (v0.1).
- **Why:** explicit methods have energy errors for velocity-dependent forces.
- **Effect on results:** `check:free_flight` (0.06 nm), `check:rotating_frame` (convergence ratio 4.00, i.e. second
  order); 3-D core line-vs-parabola error 2e-9 m.
- **How to change it:** `dolle_sim.simulation.ChamberSim._kick`, `dolle_core.sim3d.kick`.
- **References:** `commit:8060592`, `commit:784e16a`.

### D4. Rigid-body rotation: the symplectic splitting of Dullweber, Leimkuhler & McLachlan (1997)
- **Decision:** free rotation of clumps uses the DLM splitting, with absolute angular momentum fixed in inertial
  space.
- **Date:** 2026-10-20 (v0.2) · **Decided by:** Review (v0.1's stale angular velocity pumped energy), Engineering ·
  **Status:** in force (v0.5 engine and 3-D core).
- **Alternatives considered:** the v0.1 update.
- **Why:** symplectic: no energy drift over long runs.
- **Effect on results:** `check:rigid_body` (dL/L 5e-12, no energy drift), `check:spin_rotating_frame`.
- **How to change it:** `dolle_core.clumps.rotate`, `dolle_sim.simulation.ChamberSim._rotate`.
- **References:** `commit:8060592`, `commit:784e16a`; `notebooks/06_the_3d_simulation.ipynb` section 6.

### D5. Collision detection: exact contact times, re-predicted after every impact
- **Decision:** the exact contact time of every candidate pair along the drift is computed; impacts are processed in
  time order and the bodies an impact changed are re-predicted. The neighbour skin adapts to the fastest grain (with
  an immediate rebuild if a bounce makes a grain too fast), and spinning clumps near something are drifted in
  windows of at most 0.05 rad. The 3-D core solves the quadratic |d + u t|^2 = S^2 (smaller root) with a brute-force
  candidate list and one event at a time.
- **Date:** 2026-10-20 (v0.2); 2026-10-21 (v0.3: skin, windows); 2026-11-11 (3-D core) · **Decided by:** Review
  (v0.1's batched loop could miss follow-on collisions), Engineering · **Status:** in force.
- **Alternatives considered:** the v0.1 batched event loop; testing overlaps after each step (REBOUND's "line" mode);
  small time steps (~1e7 per second for 1 nm precision); a scipy k-d tree in the 3-D core (not available in the
  browser).
- **Why:** the event-driven method of Alder & Wainwright (1959): exact to round-off; for the 3-D core <=10 ms per
  step for <=300 grains, no scipy (runs in Pyodide), and the same events as the v0.5 engine.
- **Effect on results:** `check:no_overlap` (0 overlap over 286 impacts). Windows of 0.05 rad keep the arc error
  below 1.25e-3 of the radius and are 6x faster than applying them everywhere.
- **How to change it:** `dolle_sim.simulation.ChamberSim._events`, `dolle_sim.simulation.ChamberSim._rotation_window`,
  `contact.neighbor_skin_m`; `dolle_core.sim3d.next_contact`, `dolle_core.sim3d.candidate_pairs`,
  `dolle_core.sim3d.rotation_window`.
- **References:** `commit:8060592`, `commit:ed2f49d`, `commit:784e16a`;
  `tests/test_dolle_core_sim3d.py` (`test_contact_time_from_the_quadratic_matches_bisection`).

### D6. Time steps of the default engine
- **Decision:** base step 4 ms in the mission (1 ms fallback); 0.5 ms at 25 g during launch, scaled as
  sqrt(25 g / a) in powers of two outside the window; at most 0.02 rad of payload roll per step; every release window
  or shock starts on a step boundary and has at least 20 steps; pair forces recomputed every step; the space-charge
  mesh every 25 steps; contact laws cached with the radius rounded to 0.5 um and the adhesion factor to 2% (R* error
  up to 0.5%; the 3-D core uses exact values). While all dust is wall-bound outside the window the step is scaled
  with the load (added in v0.5), and after a kick the step is limited by the grains' speed. These two change the time
  step: they are accuracy choices, not bit-identical shortcuts (those are D7).
- **Date:** 2026-10-20 to 2026-10-22; every-step pair forces from 2026-10-21 (v0.3 review: they had been held 5
  steps, 20 ms) · **Decided by:** Engineering; Review (pair forces) · **Status:** in force.
- **Alternatives considered:** a fixed step everywhere (slower); holding pair forces for several steps.
- **Why:** keeps the path bending a dt^2/8 near ~8 um at launch and second-order accuracy elsewhere.
- **Effect on results:** second order is shown by `check:rotating_frame` and `check:flight_frame` (convergence ratio
  4.00 when the step is halved; `CHECKS.md` rows 8 and 23, `check:flight_frame` at dt = 0.5 ms);
  `check:free_flight` compares one run with the analytic solution (0.06 nm) and has no convergence test.
- **How to change it:** `dt_s`, `dt_launch_s`, `forces.long_range_update_every`, `forces.space_charge_update_every`
  in a config; `dolle_sim.simulation.ChamberSim._dt_at`, `dolle_sim.simulation.ChamberSim._contact`.
- **References:** `commit:a3fffe8`, `commit:ed2f49d`, `commit:a866c4e`, `commit:0731b2f`.

### D7. Exact shortcuts while all dust is wall-bound
- **Decision:** two exact shortcuts: while every grain is wall-bound the engine reuses the pair, charge and
  space-charge forces instead of recomputing them, and it skips the tree statics of a wall-bound clump while an upper
  bound on its load is below its weakest bond. (The load-scaled step and the speed-limited steps change the time step
  and belong to D6.)
- **Date:** 2026-10-22 (force reuse, `commit:a866c4e`); 2026-10-26 (the clump pre-check, `commit:0731b2f`) ·
  **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** the full calculation every step (~10x slower).
- **Why:** whole-flight runs ~10x faster (with the speed-limited steps of D6, `commit:a866c4e` message) and the
  shortcuts themselves change nothing.
- **Effect on results:** bit-identical: a 150-grain graphite run of the `kick+spin_1hz` scenario (charged grains,
  clumps on the walls) gives the same positions, velocities, charges and event counts with and without the two
  shortcuts (`tests/test_flight.py`, `test_wall_bound_shortcuts_are_exact`).
- **How to change it:** `dolle_sim.simulation.ChamberSim._aggregate_may_fail`, `dolle_sim.simulation.ChamberSim.step`.
- **References:** `commit:a866c4e`, `commit:0731b2f`.

### D8. Every check has a criterion and a measured value; CHECKS.md is generated
- **Decision:** each check states what it compares, its pass criterion and the measured value; `CHECKS.md` and
  `docs/data/checks.json` are generated by `tools/checks_table.py` from the result files (never hand-written). The
  main package has 26 model checks (`python -m dolle_sim verify`, also run in the browser); the v0.6 engine has 19
  (`python -m dolle_extras verify`).
- **Date:** 16 checks 2026-10-20; 25 on 2026-10-22; 26 on 2026-10-26; criteria and generated table 2026-10-28 ·
  **Decided by:** Owner (demanded "all checks with criteria"), Engineering · **Status:** in force.
- **Alternatives considered:** a hand-written list.
- **Why:** a table that cannot drift from the runs.
- **Effect on results:** today 64 of 64 checks and comparisons pass (26 model, 19 engine, LAMMPS, REBOUND).
- **How to change it:** add a check function returning `_res(...)` in `dolle_sim/verification.py` (list
  `dolle_sim.verification.CHECKS`) or `extras/dolle_extras/verify_full.py`; then `python -m dolle_sim verify --json
  results/verify_all.json` (for a v0.6 check `python -m dolle_extras verify --json results/verify_extras.json`; for
  the test counts `python -m pytest tests --junitxml=results/pytest.xml`) and `python tools/checks_table.py`, which
  also reads the LAMMPS and REBOUND result files (its docstring lists every input).
- **References:** `commit:f5b27a8`, `commit:a866c4e`, `commit:0731b2f`, `commit:75fa4e8`, `commit:f133c3e`;
  `CHECKS.md`.

### D9. Provenance of every component in one generated table; every public claim audited
- **Decision:** `dolle_sim/provenance.py` lists every component with its source, code location, checks and
  assumptions, marked published / standard / derived / assumed; it generates `PROVENANCE.md`, the website's Sources
  tab and `python -m dolle_sim sources`. All public text was audited claim by claim (68 confirmed corrections).
- **Date:** 2026-10-21 (v0.3) · **Decided by:** Owner ("trust is priority": every physics claim sourced or marked
  assumed), Review and Engineering · **Status:** in force; the extras have their own table
  (`extras/PROVENANCE.md`).
- **Alternatives considered:** sources scattered in docstrings.
- **Why:** an official can trace any number to a paper, a data sheet, a check or the word ASSUMED.
- **Effect on results:** 51 components today (12 published, 17 standard, 13 derived, 9 assumed).
- **How to change it:** edit `dolle_sim/provenance.py`, then `python tools/build_site.py`; `tests/test_provenance.py`
  (`test_generated_files_are_up_to_date`, `test_every_named_check_exists`) keeps it honest.
- **References:** `commit:ed2f49d`, `commit:13dd8e8`.

### D10. LAMMPS is the independent referee of the contact law
- **Decision:** the contact law is cross-checked against the LAMMPS GRANULAR package: for the 1-D law with the same
  break rule, and term by term for the v0.6 law (jkr normal, linear_history sliding, sds rolling and twisting, in
  LAMMPS's order: rotate, increment, force, cap).
- **Date:** 2026-10-22 (1-D check); 2026-10-28 (term by term) · **Decided by:** Engineering · **Status:** in force
  (in `extras/`).
- **Alternatives considered:** no outside check; building on LAMMPS (E1).
- **Why:** the closest established code with the same physics.
- **Effect on results:** undamped thresholds agree to 0.26% (silica) and 0.11% (graphite), restitution within
  0.011, pull-off 0.14% (the earlier "0.5%" was the detection grid, corrected in review); 7 set-ups of the v0.6 law
  agree to <=2.2e-6. Damped collisions are not compared: LAMMPS refuses its Tsuji damping together with JKR.
- **How to change it:** `python -m dolle_extras lammps-check` (needs `pip install lammps`);
  `extras/dolle_extras/lammps_check.py`, `extras/tools/lammps_contact_check.py`.
- **References:** `commit:a866c4e`, `commit:0731b2f`, `commit:75fa4e8`; `CHECKS.md`.

### D11. REBOUND only as an outside reference
- **Decision:** REBOUND is used only in an optional cross-check of trajectories and collision bookkeeping; none of
  the physics comes from it, and a test runs the simulator with REBOUND (and LAMMPS) blocked.
- **Date:** 2026-10-20; isolation test 2026-10-21 · **Decided by:** Owner (named REBOUND as a trusted reference),
  Engineering (scope) · **Status:** in force (moved to `extras/` in v0.7).
- **Alternatives considered:** build on REBOUND (E1).
- **Why:** an independent code for what it can check.
- **Effect on results:** trajectories agree to 3.2e-10 m; collision counts within 2 sigma. It does NOT check contact
  physics.
- **How to change it:** `python -m dolle_extras rebound-check configs/flight_carbon_vs_silica.json --duration 60
  --max-grains 600`; `extras/dolle_extras/rebound_check.py`; `tests/test_provenance.py`
  (`test_simulator_runs_without_rebound`).
- **References:** `commit:a3fffe8`, `commit:8060592`, `commit:ed2f49d`, `commit:13dd8e8`.

### D12. The readable core must equal the full code, number for number
- **Decision:** `tests/test_dolle_core.py` compares `dolle_core` with `dolle_sim` case by case (every material
  value, the pair properties, identical thresholds, restitution within 6e-16, momentum bookkeeping); reference
  numbers are hard-coded on purpose. The core-vs-v0.6 comparisons (friction table, wall limits, rolling onset) are
  in `tests/test_core_vs_dem.py` (skipped when `extras/` is absent).
- **Date:** 2026-11-10 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** test the core only against itself.
- **Why:** a reader of the core reads the real physics, not a teaching version (E6).
- **Effect on results:** changing any material value or pair rule breaks `test_threshold_reference_numbers`
  (6.17518, 38.35856, 6.66206, 18.23326 mm/s) and `test_wall_reference_numbers` (1033.74 / 1627.71 g,
  1.2935 / 3.5897 g) until they and every document quoting them are updated.
- **How to change it:** edit the numbers in `tests/test_dolle_core.py` together with the change; run
  `python -m pytest tests/test_dolle_core.py tests/test_core_vs_dem.py -q`.
- **References:** `commit:90a0058`; `notebooks/README.md`.

### D13. Notebooks are executed by the tests and were reviewed by two readers
- **Decision:** `tests/test_notebooks.py` runs every notebook start to finish; the notebooks were reviewed by a
  student reader and a physics checker (45 findings, applied).
- **Date:** 2026-11-10 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** notebooks outside the test suite.
- **Why:** an explanation that no longer runs is worse than none.
- **Effect on results:** the notebooks contain no asserts, so a changed number shows only when the text is re-read.
- **How to change it:** `python -m pytest tests/test_notebooks.py -q`.
- **References:** `commit:90a0058`, `commit:784e16a`.

### D14. Documents that name code are tested against the code
- **Decision:** the engine's decision map names functions as `module.function` and checks as `check:<id>`, and
  `extras/tests/test_engine_logic_doc.py` fails if any of them stops existing. This decision log follows the same
  rule (`tests/test_decisions_doc.py`: paths, `module.function` names, check ids, tests and commits).
- **Date:** 2026-11-10 (decision map); 2026-11-16 (this log) · **Decided by:** Owner (asked on 2026-10-31 to make
  every decision's logic visible and verifiable), Engineering · **Status:** in force (the decision map, this log
  `docs/DECISIONS.md` and its test `tests/test_decisions_doc.py`).
- **Alternatives considered:** untested documents.
- **Why:** documents that cannot silently go out of date.
- **Effect on results:** none on numbers.
- **How to change it:** run `python -m pytest tests/test_decisions_doc.py extras/tests/test_engine_logic_doc.py -q`
  after renaming anything.
- **References:** `commit:874976e`, `commit:f133c3e`; `extras/docs/ENGINE_LOGIC.md`.

### D15. The v0.6 standard: every reduced treatment is exact or bounded, and every step is verified
- **Decision:** in the v0.6 engine every treatment other than the full contact model is either exact (the full
  model would do the same) or bounded (the difference is computed and limited); every step is verified after it is
  taken (speeds against the bounds others assumed, rolling paths against envelopes, grains entering the full model
  against everything advanced outside it) and redone if any check fails (up to 64 attempts, then an error; redos
  counted by reason).
- **Date:** 2026-10-28 · **Decided by:** Engineering, under the Owner's "no shortcuts" demand (E3) · **Status:** in
  force (extras).
- **Alternatives considered:** unverified speed-ups; the full model everywhere (~1e4 core-days per flight).
- **Why:** makes speed-ups admissible under "no shortcuts".
- **Effect on results:** the v0.6 engine's answers equal the full model's within the stated bounds (19 checks).
- **How to change it:** `dolle_extras.engine.FullContactSim._verify`, `dolle_extras.engine.FullContactSim.step`.
- **References:** `commit:75fa4e8`, `commit:83af20d`; `extras/docs/ENGINE_LOGIC.md`.

### D16. v0.6 full-model time step: 1/50 of the fastest contact period
- **Decision:** the full model steps at 1/50 of the fastest contact period (5-25 ns), the Hertz time / 50 in fast
  impacts; the reduced models at 1/200 (single-grain rolling) and 1/400 (slaved clusters) of their soft periods.
- **Date:** 2026-10-28 · **Decided by:** Engineering · **Status:** in force (extras).
- **Alternatives considered:** softened contacts with longer steps (rejected, E1).
- **Why:** convergence.
- **Effect on results:** `check:time_step`: halving the step changes the rolled distance by 3.3e-7.
- **How to change it:** `contact.steps_per_period`, `contact.roll_steps_per_period`,
  `contact.slaved_steps_per_period`.
- **References:** `commit:75fa4e8`.

### D17. Exact free flight up to the capture distance d_cap = 0.1 um
- **Decision:** a grain touching nothing flies on the exact integrator until the earliest time it could come within
  d_cap = 0.1 um of anything; a sleeper joins the full model when a moving grain comes within 2 d_cap.
- **Date:** 2026-10-28 · **Decided by:** Engineering · **Status:** in force (extras).
- **Alternatives considered:** any other positive distance (all exact; smaller means less full-model time).
- **Why:** the full model resolves the whole approach, so the hand-off is exact.
- **Effect on results:** `check:engine_handoff`.
- **How to change it:** `contact.capture_distance_m`; `dolle_extras.engine.FullContactSim._capture_times`.
- **References:** `commit:75fa4e8`, `commit:83af20d`.

### D18. Sleep certificate for resting grains and clusters
- **Decision:** a resting grain or wall-anchored cluster is held static while every contact stays inside all its
  limits with a 2% margin (wake at 0.98; entry originally at 0.90); residual motion must not change any contact
  load by more than 1% of its margin (an exact energy-ellipsoid bound); on waking it is placed at the exact static
  equilibrium.
- **Date:** 2026-10-28 · **Decided by:** Engineering · **Status:** in force; extended to contacts AT their limits by
  D27 and D33.
- **Alternatives considered:** keep integrating resting grains (weeks of computing).
- **Why:** nothing can change in an elastic contact inside its limits.
- **Effect on results:** `check:sleep_exact` (0.047 nm of a 15 nm deflection), `check:cluster_statics`.
- **How to change it:** `contact.sleep_margin`; `dolle_extras.engine.FullContactSim._try_sleep`,
  `dolle_extras.engine.FullContactSim._try_sleep_island`.
- **References:** `commit:75fa4e8`.

### D19. A cluster still rocking may sleep with its residual motion kept as exact linear response
- **Decision:** the residual is kept as the exact linear response (modes of the stiffness and dashpots) and added
  back on waking; clusters up to 64 grains.
- **Date:** 2026-10-28 · **Decided by:** Engineering · **Status:** in force (extras).
- **Alternatives considered:** dropping the residual.
- **Why:** dropping it would miss motion.
- **Effect on results:** `check:linres`: 0.07 pm error against 937 pm if dropped.
- **How to change it:** `dolle_extras.engine.LINRES_MAX`; `dolle_extras.engine.FullContactSim._linres`.
- **References:** `commit:75fa4e8`, `commit:4242212`.

### D20. Slaved models for a grain rolling on a wall and for small anchored clusters
- **Decision:** a grain rolling on one wall holds its stiff normal and sliding modes quasi-static and integrates the
  soft modes at 1/200 of their period (enter at 0.90, hand back at 0.98, and on plastic loading); wall-anchored
  clusters of <=8 grains / 16 contacts, non-redundant and well conditioned, hold their stiff modes as Lagrange
  multipliers (soft springs at 1/400), accepted only if the soft-frequency shift indicator is <=2e-4 (<=1e-3
  load-weighted).
- **Date:** 2026-10-28 · **Decided by:** Engineering (the cluster model was designed by a panel of three independent
  designs and three judges; reviewed with two refuters per finding) · **Status:** in force (extras).
- **Alternatives considered:** keeping moving clusters in the full model (slow).
- **Why:** rolling along walls during the boost is the v0.6 engine's reason to exist; the full model is too slow.
- **Effect on results:** `check:roll1_vs_full` (<=3e-5 of the rolled path; measured 1.2e-5),
  `check:slaved_vs_full` (7e-6 to 1.7e-5), `check:engine_slaved`.
- **How to change it:** `extras/dolle_extras/roll1.py`, `extras/dolle_extras/slaved.py`;
  `dolle_extras.engine.FullContactSim._try_roll`, `dolle_extras.engine.FullContactSim._try_slaved`.
- **References:** `commit:75fa4e8`, `commit:4242212`.

### D21. Local speed bounds; short steps after load jumps
- **Decision:** each grain's speed bound comes from the grains that can reach it during the step (checked after the
  step); 50 us steps follow load jumps (stiff modes ring for ~5 us).
- **Date:** 2026-10-28 · **Decided by:** Engineering · **Status:** in force (extras).
- **Alternatives considered:** one global bound (a fast grain anywhere drags every roller into the full model).
- **Why:** speed.
- **Effect on results:** none on answers (verified steps).
- **How to change it:** `dolle_extras.engine.FullContactSim._speed_bounds`, `dolle_extras.engine.FullContactSim._outer_dt`.
- **References:** `commit:4242212`.

### D22. Scenario branches from checkpoints only when provably identical
- **Decision:** scenarios identical up to some time continue from a checkpoint only if fresh runs differ in nothing
  but the release schedule or flight-profile options (attributes hashed) and every load is identical up to the
  checkpoint (sampled each ms); otherwise they run from lift-off.
- **Date:** 2026-10-28 · **Decided by:** Engineering · **Status:** in force (extras).
- **Alternatives considered:** always run from lift-off (slower).
- **Why:** saves the shared part of the flight without changing a bit.
- **Effect on results:** bit-for-bit (`test_branch_from_checkpoint_is_exact`).
- **How to change it:** `dolle_extras.engine.branch`, `dolle_extras.engine.BRANCH_ALLOWED`.
- **References:** `commit:75fa4e8`.

### D23. v0.6.1: exits with rejoin, time-resolved verification, a missed-contact safety net
- **Decision:** a grain in the full model that touches nothing leaves it up to its next possible capture and rejoins
  then; grains entering the full model are verified time-resolved; a grain overlapping something it has no contact
  with forces a redo with it in the full model.
- **Date:** 2026-10-28 · **Decided by:** Engineering, Review · **Status:** in force (extras).
- **Alternatives considered:** keeping a bouncing grain in the full model throughout.
- **Why:** speed with the same answers; the safety net catches what no bound allows.
- **Effect on results:** launch to T+5 s from 76 to 42 s of computing; T+5.0-5.4 s from 184 to 60 s; the hand-off
  check was redesigned to test a real hand-off.
- **How to change it:** `dolle_extras.engine.FullContactSim._early_exit`, `dolle_extras.engine.FullContactSim._missed`.
- **References:** `commit:83af20d`.

### D24. A rolling grain's envelope assumes no direction
- **Decision:** a rolling grain's clearance uses its speed magnitude (Coriolis, rolling resistance and loads can turn
  it); acceleration bounds cover the whole step (spline maximum, waveform amplitude, flight profile at 5 points with
  a curvature margin, Gronwall factor exp((2 Omega + g) H)).
- **Date:** 2026-10-28 · **Decided by:** Review (found while profiling), Engineering · **Status:** in force
  (extras).
- **Alternatives considered:** the earlier bound assumed the velocity kept its direction (H13).
- **Why:** in the nominal flight a grain rolled 11.8 um into a wall it was "provably" clear of, and the overlap
  launched grains at 10-58 m/s.
- **Effect on results:** removes real missed contacts.
- **How to change it:** `dolle_extras.engine.FullContactSim._roll_bound`.
- **References:** `commit:83af20d`.

### D25. One Loads object for every model; the engine refuses inputs its bounds cannot see
- **Decision:** free flight uses the same loads as every other model and the bounds; the v0.6 engine refuses
  g-jitter and accelerometer files instead of silently using them.
- **Date:** 2026-10-28 (v0.6.2) · **Decided by:** Review (second independent review: 6 confirmed findings, 1
  refuted), Engineering · **Status:** in force (extras). No mission uses g-jitter or an accelerometer file.
- **Alternatives considered:** each model with its own loads (H14).
- **Why:** a bound that does not see a load is not a bound.
- **Effect on results:** pair-force cache added at the same time: bit-identical, 204 to 132 s.
- **How to change it:** `extras/dolle_extras/loads.py`.
- **References:** `commit:30840b8`.

### D26. At least 40 outer steps per cycle of the fastest load; relaxing it was rejected
- **Decision:** keep >= 40 outer steps per cycle of the highest load frequency, also in the `launch_vibration`
  scenario.
- **Date:** 2026-10-28 (project notes) · **Decided by:** Engineering · **Status:** in force; the relaxation was
  rejected.
- **Alternatives considered:** relaxing the rule for random vibration (would need signed interval load bounds in the
  rolling bound).
- **Why:** the bound would no longer be proven.
- **Effect on results:** none (the scenario is slow).
- **How to change it:** `dolle_extras.engine.FullContactSim._outer_dt`.
- **References:** project notes; `extras/docs/ENGINE_LOGIC.md`.

### D27. Clusters at their contact limits sleep: tier 1 exact, tier 2 bounded creep
- **Decision:** tier 1 lifts the contact limits, solves exact elastic equilibria at the step's end and middle and
  bounds every contact quantity between them; if every sliding, rolling and twisting ratio stays <= 1 + 1e-6 no
  plastic variable can change (exact). Otherwise tier 2 iterates the real law's return map to equilibrium (released
  stiffness as tangent, whole step and two half steps), with motion, flow, path error and a cumulative budget (1e-2
  of the deflection at the limit) enforced and reported. Pull-off keeps the 0.98 margin (a real instability).
- **Date:** 2026-10-31 · **Decided by:** Engineering (design process G16) · **Status:** in force (extras).
- **Alternatives considered:** keep refusing such clusters (weeks of computing); designs A, B and C of the design
  record.
- **Why:** static piles sat at their rolling limit (ratio >= 0.999), so the old 0.90 entry test never passed and
  they were integrated at 5-18 ns steps for the whole flight (G5).
- **Effect on results:** `check:cap_sleep_vs_dem` (asleep 95-97% of island-steps), `check:between_sample_bound`
  (0 violations in 11,800 grid points), `check:creep_island_vs_dem`; new engine ~14 s of computing per flight-second
  in the boost.
- **How to change it:** `dolle_extras.engine.ISLAND_DEFAULTS` (`cap_sleep`, `creep_islands`);
  `dolle_extras.engine.FullContactSim._island_advance`, `extras/dolle_extras/statics.py`.
- **References:** `commit:e7fafd6`, `commit:937a6dd`, `commit:fb4ccf1`;
  `extras/docs/design/at-limit-sleep/README.md`.

### D28. Left limits at load breakpoints, and the chord-deviation bound
- **Decision:** a step ending at a load breakpoint is certified against the pre-jump load; the chord-deviation bound
  limits how far loads stray from a straight line inside a step (curvature, exact half-sine, |slope jump| l/4 at
  kinks).
- **Date:** 2026-10-31 · **Decided by:** Review (adversarial critique of the synthesis), Engineering · **Status:** in
  force (extras).
- **Alternatives considered:** committing the post-jump equilibrium (dropped the jump's transient); the synthesis'
  kink term (off by a factor 2).
- **Why:** the critique's findings.
- **Effect on results:** `check:island_across_breakpoint` (1.8e-16 m from the pre-jump state).
- **How to change it:** `extras/dolle_extras/loads.py`; `extras/tests/test_loads_chord.py`.
- **References:** `commit:58b2ef3`.

### D29. The ratchet bound is computed in the released system
- **Decision:** the slip that left-over motion can cause at a component near its limit is bounded in the released
  system (springs at their limits removed, iterated to a fixed point) and must be <= 1e-3 of the deflection at the
  limit.
- **Date:** 2026-10-31; fixed point added 2026-11-10 · **Decided by:** Review (critique), Engineering · **Status:**
  in force (extras).
- **Alternatives considered:** the synthesis' bound E/cap (energy over the limit force), H18.
- **Why:** at a limit the external load does the plastic work; E/cap was low by 1e2-1e5.
- **Effect on results:** `check:ratchet_released_bound`: E/cap exceeded in 4 of 5 cases, the released bound always
  held (9x margin).
- **How to change it:** `dolle_extras.engine.FullContactSim._island_ratchet`; `island_ratchet_rel` in
  `dolle_extras.engine.ISLAND_DEFAULTS`.
- **References:** `commit:937a6dd`, `commit:fb4ccf1`.

### D30. Entry energy under a changing load: the motion relative to the drift of the equilibrium
- **Decision:** a cluster entering sleep is charged the kinetic energy of its motion RELATIVE to the drift of its
  equilibrium, the larger of the elastic and released systems.
- **Date:** 2026-10-31; revised 2026-11-10 · **Decided by:** Engineering · **Status:** in force (extras); verified
  only on the scenes of the checks (O9).
- **Alternatives considered:** the first version (whole kinetic energy plus an "adiabatic factor"), H19.
- **Why:** the old version refused a cluster carried along a 30-degree turning load at every attempt (58 of 58).
  Both versions are this project's own derivation with NO outside source.
- **Effect on results:** `check:ratchet_released_bound` finds the measured slips within both bounds.
- **How to change it:** `dolle_extras.engine.FullContactSim._island_drift`.
- **References:** `commit:937a6dd`, `commit:fb4ccf1`; `extras/docs/ENGINE_LOGIC.md` (open items).

### D31. Tier-2 dynamics are charged as a running maximum (ASSUMED)
- **Decision:** the full model's lag behind quasi-static creep is charged as its running maximum, not summed per
  step.
- **Date:** 2026-11-10 · **Decided by:** Engineering · **Status:** in force (extras); ASSUMED beyond the linearised
  argument.
- **Alternatives considered:** a sum per step (H20).
- **Why:** this project's own argument from the linearised released structure; the sum woke the turning-load case
  by budget after 1-2 steps.
- **Effect on results:** creeping clusters stay asleep where the full model shows they may.
- **How to change it:** `dolle_extras.engine.FullContactSim._island_advance` (where the lag is charged).
- **References:** `commit:fb4ccf1`; design record deviation 5 (`extras/docs/design/at-limit-sleep/README.md`).

### D32. Island tolerances
- **Decision:** limit tolerance 1e-6, ratchet 1e-3, motion 1e-9 m per step, flow 5%, half-step path 1e-3, budget
  1e-2, 20 hysteresis steps, 40 Newton iterations; row norms split into translational and rotational parts.
- **Date:** 2026-10-31 · **Decided by:** Engineering · **Status:** in force (extras).
- **Alternatives considered:** the synthesis' motion limit 1e-10 m (too tight for early creep); one combined row
  norm (overstated a 1e-16 N residual as 4e-4 of a rolling limit).
- **Why:** each value is the tightest that does not wake every capped cluster.
- **Effect on results:** set the fraction of island-steps asleep (D27).
- **How to change it:** `dolle_extras.engine.ISLAND_DEFAULTS` (also as `contact.*` config keys).
- **References:** `commit:937a6dd`.

### D33. A single grain at a limit enters sleep at the margin it stays at
- **Decision:** a single grain on one wall enters sleep at the same 0.98 it stays at, with the chord term and a
  20-step look-ahead against chatter.
- **Date:** 2026-10-31 · **Decided by:** Engineering · **Status:** in force (extras).
- **Alternatives considered:** entry at 0.90 (H26).
- **Why:** grains sitting at 0.96-0.999 of a limit never slept.
- **Effect on results:** an edge grain at twist ratio 0.962 would otherwise stay in the full model for 800 s.
- **How to change it:** `island_hyst_steps` in `dolle_extras.engine.ISLAND_DEFAULTS`;
  `dolle_extras.engine.FullContactSim._try_sleep`.
- **References:** `commit:937a6dd`.

### D34. Switches for the new sleep rules
- **Decision:** `contact.cap_sleep = false` restores the previous entry test; the `twist_continuity` option is off (a
  twist the law would resolve from a round-off rate sign wakes the island instead); old saved states are upgraded on
  load.
- **Date:** 2026-10-31 · **Decided by:** Engineering, Review · **Status:** in force (extras).
- **Alternatives considered:** no switch.
- **Why:** to compare with the old engine and to keep old states loadable within v0.6.
- **Effect on results:** none by default.
- **How to change it:** `dolle_extras.engine.ISLAND_DEFAULTS`; `dolle_extras.engine.FullContactSim._island_upgrade`.
- **References:** `commit:937a6dd`, `commit:fb4ccf1`.

### D35. Checks 14-19 against the full model; the campaign gated on them
- **Decision:** six checks against the full model at two step sizes (`check:cap_sleep_vs_dem`,
  `check:cap_twist_microgravity`, `check:creep_island_vs_dem`, `check:between_sample_bound`,
  `check:ratchet_released_bound`, `check:island_across_breakpoint`). The gate was on USING the campaign's results,
  not on running it: on 2026-11-10 the campaign was restarted at `commit:726ec34` while these checks were still being
  written (README at `commit:874976e`: its results would replace the table "once the checks of that change (14-19)
  pass"); after the checks passed and the review fixes were in, it was restarted from scratch at `commit:f133c3e` and
  the `commit:726ec34` run was set aside (G6).
- **Date:** 2026-10-31 (gate); 2026-11-10 (checks pass) · **Decided by:** Engineering, Review · **Status:** done.
- **Alternatives considered:** wait with the restart until the checks pass (not done: running in parallel saved
  time, and the gate on the results kept unchecked numbers out of the documents).
- **Why:** the new treatment must be checked before its results are used.
- **Effect on results:** writing them found 4 faults and the review 11 more, all fixed (e.g. a skip that started at
  a breakpoint carried pre-jump states across it; a failed skip sample counted as slept); 19 of 19 engine checks pass.
- **How to change it:** `extras/dolle_extras/verify_full.py`; `python -m dolle_extras verify`.
- **References:** `commit:fb4ccf1`, `commit:f133c3e`, `commit:874976e`; `CHECKS.md` Part 2.

### D36. A full-stiffness LAMMPS run of the seconds after a release: offered, not done
- **Decision:** a LAMMPS run at full stiffness of the first seconds after a release was noted as affordable (1-4 h
  of computing per tubule-second) and as the test of the default engine's two approximations (rigid clumps,
  fragmentation thresholds); it was not run.
- **Date:** 2026-10-26 · **Decided by:** Engineering (offered) · **Status:** open (O10).
- **Alternatives considered:** the v0.6 engine for the same purpose (its campaign was stopped).
- **Why:** would test B12 and B14 directly.
- **Effect on results:** unknown until run.
- **How to change it:** set up from `extras/tools/lammps_contact_check.py` and `extras/tools/lammps_benchmark.py`.
- **References:** `commit:0731b2f`; `PROVENANCE.md`.

## E. Engines and software architecture

### E1. Our own simulator, not built on REBOUND or LAMMPS
- **Decision:** write a dedicated python package for DOLL-E (`dolle_sim`) instead of building on an existing code;
  use the existing codes only as outside checks (D10, D11).
- **Date:** 2026-10-20 (written); 2026-10-26 (alternatives measured and documented) · **Decided by:** Owner (asked
  for a simulator, and asked why not use existing codes), Engineering (architecture and the answer) · **Status:** in
  force.
- **Alternatives considered:** REBOUND 5.2 (no walls, no grain spin or inertia, aggregates only by merging into one
  sphere, no gas drag, electrostatics or adhesion, collision hook after a full step, GPL-3.0); LAMMPS GRANULAR (the
  right contact physics, but one global ~10 ns step for every grain); softening the grains (the usual DEM shortcut).
- **Why:** time scales. A sticking contact lasts ~1 us and needs ~10 ns steps; the flight lasts ~15 minutes.
  Measured with `extras/tools/lammps_benchmark.py`: 2.3e7 grain-steps per second per core in flight, 6.1e6 with the
  settled layer, so one tubule takes 13-49 core-days to power-off (35-130 to splashdown); DOLL-E takes 3 minutes to
  10 hours for all four tubules on 4 cores. Softening fails here: the JKR sticking speed scales as E^(-1/3), and the
  standard correction gamma x (E_soft/E)^(2/5) (Haervig et al. 2017; Chen et al. 2019; Washino et al. 2024) keeps
  the sticking speed but cuts the wall pull-off 16x at E/1000, and pull-off decides launch and release.
- **Effect on results:** none directly; it is what makes whole-flight runs possible.
- **How to change it:** not a parameter.
- **References:** `commit:a3fffe8`, `commit:0731b2f`; `docs/HISTORY.md` ("Why our own engine").

### E2. The default engine (v0.5): event-driven free flight, impacts integrated in time
- **Decision:** grains and rigid clumps fly under the external forces in millisecond steps; impacts are found by
  continuous collision detection, processed in time order, and each is resolved by integrating the 1-D contact law
  in nanosecond steps; stuck grains form rigid clumps that fragment by energy thresholds.
- **Date:** 2026-10-20 (v0.1-v0.5) · **Decided by:** Engineering · **Status:** in force: the default again since
  v0.7 (E5).
- **Alternatives considered:** the v0.6 full-contact engine (E3).
- **Why:** cheap for free-flying clouds over 15 minutes, at the price of two stated approximations: rigid clumps (B12)
  and fragmentation rules (B14), plus frozen wall grains (B8).
- **Effect on results:** every stored scenario result (computed on 2026-10-26 with the v0.5 code, `commit:0731b2f`)
  and the flight demo result (computed with the v0.6.2 code, whose `run` command still used this engine) come from
  this engine (G10).
- **How to change it:** `dolle_sim/simulation.py` (`dolle_sim.simulation.ChamberSim`).
- **References:** `commit:a3fffe8` to `commit:0731b2f`; `docs/HISTORY.md`.

### E3. "No shortcuts": the v0.6 full-contact engine for mission runs
- **Decision:** a second engine in which every grain is a sphere with 6 degrees of freedom and every contact has the
  full adhesive law at full stiffness with sliding, rolling and twisting springs; no rigid clumps, no fragmentation
  rules, no frozen wall grains; exact or verified treatments where nothing can happen (D15-D35). It became the engine
  of `python -m dolle_sim mission`.
- **Date:** 2026-10-27 (demand, project notes); 2026-10-28 (v0.6, v0.6.1, v0.6.2) · **Decided by:** Owner ("no
  shortcuts", all checks with criteria), Engineering · **Status:** superseded as the default by E5 (v0.7); still
  runnable in `extras/`.
- **Alternatives considered:** keep v0.5 for missions (cheaper; approximations stated); LAMMPS everywhere (weeks).
- **Why:** removes the v0.5 engine's approximations; in particular grains can roll along the walls during the boost,
  which the default engine cannot show (B8).
- **Effect on results:** no scenario results exist from it: its campaign was stopped (G6). Its checks: 19 of 19 pass.
- **How to change it:** `python -m dolle_extras mission configs/mission_quarter_deck.json --grains 100`;
  `extras/dolle_extras/engine.py` (`dolle_extras.engine.FullContactSim`).
- **References:** `commit:75fa4e8`, `commit:4242212`, `commit:83af20d`, `commit:30840b8`;
  `extras/docs/ENGINE_LOGIC.md`.

### E4. The v0.6 contact law is written in the LAMMPS form, term by term
- **Decision:** the v0.6 law is jkr normal + linear_history sliding + sds rolling + sds twisting, in LAMMPS's order
  (rotate, increment, force, cap), with its reset of over-limit twist at zero rate.
- **Date:** 2026-10-28 · **Decided by:** Engineering · **Status:** in force (extras).
- **Alternatives considered:** an independent formulation (could not be checked term by term).
- **Why:** makes LAMMPS an exact referee (D10).
- **Effect on results:** 7 set-ups agree to <=2.2e-6.
- **How to change it:** `dolle_extras.dem.contact_force`, `dolle_extras.dem.make_params`; re-run
  `extras/tools/lammps_contact_check.py`.
- **References:** `commit:75fa4e8`.

### E5. v0.7: the v0.6 engine moves to extras/, the v0.5 engine is the default again
- **Decision:** the v0.6 engine and its satellites (7,815 lines), the LAMMPS and REBOUND cross-checks, image-based
  grain sizing, the campaign tools, the atmosphere generator, the decision map and the design record move to
  `extras/` (runnable as `python -m dolle_extras ...`, 82 tests); `python -m dolle_sim mission` uses the v0.5 engine
  again, and a config with `"engine": "full"` stops with a message naming `python -m dolle_extras mission`.
- **Date:** 2026-11-11 · **Decided by:** Owner (each specific choice, agreed feature by feature with the coding
  sessions), prompted by the Team reviewer's request of 2026-11-10 to strip extra features (G7) · **Status:** in
  force; reverses E3 as the default (H22).
- **Alternatives considered:** delete the v0.6 code; keep it in `dolle_sim`.
- **Why:** the team asked for a package one can read; the v0.6 campaign had been stopped; the published scenario
  table was from v0.5 anyway. Nothing is lost.
- **Effect on results:** `dolle_sim` went from 15,489 to 6,711 lines; the 26 model checks and the 19 engine checks
  give identical values.
- **How to change it:** `dolle_sim.mission.FULL_ENGINE_COMMAND`; the `"engine"` key of a mission config.
- **References:** `commit:13dd8e8`; `extras/README.md`; `docs/HISTORY.md` (v0.7, commands that moved).

### E6. dolle_core: a copy of the physics in plain numpy, not a simplification
- **Decision:** a small package (`dolle_core`: materials, jkr, damping, plastic, pair, collide, scaling, wall, many,
  and for the 3-D page clumps, impacts, sim3d, scenes) with the same equations, values and arithmetic order as
  `dolle_sim/contact.py`, in plain numpy without numba. `dolle_sim` keeps numba (optional) for speed.
- **Date:** 2026-11-10 · **Decided by:** Owner (asked for "all 100% realistic", not a teaching model), Engineering ·
  **Status:** in force.
- **Alternatives considered:** a simplified teaching model.
- **Why:** readable line by line, runs in a browser, and its numbers are the simulator's numbers (D12).
- **Effect on results:** thresholds identical, restitution within 6e-16 (`tests/test_dolle_core.py`).
- **How to change it:** any physics change must be made in both places (E7).
- **References:** `commit:90a0058`, `commit:784e16a`; `notebooks/README.md`.

### E7. Values and contact law exist in several places, kept identical on purpose
- **Decision:** material values exist in `dolle_core/materials.py` and `dolle_sim/materials.py` (the v0.6 engine
  reads the `dolle_sim` values); the pair mixing rules (geometric-mean w, sqrt(e1 e2), the smaller p_y) in
  `dolle_core/pair.py`, `dolle_sim.materials.pair_props`, `dolle_extras.engine.FullContactSim._params` and
  `dolle_extras.dem.DEMSystem._params`; the friction table in `dolle_core.materials.FRICTION` and
  `dolle_extras.engine.MU_DEFAULT`. `dolle_extras.dem.make_params` holds the reduced modulus, radius and mass, the
  sliding and twisting stiffnesses, xi_crit = h a0/12 and the damping, from w and e_dry passed in. The contact law
  in `dolle_core/collide.py` mirrors `dolle_sim.contact._resolve_core` line by line. Tests fail if one copy is
  changed alone.
- **Date:** 2026-11-10 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** one shared table imported by everything (would make `dolle_core` depend on the
  simulator, or the simulator on the readable core).
- **Why:** `dolle_core` must stand alone and stay readable; `dolle_sim` must not change behaviour for it.
- **Effect on results:** none; a safeguard.
- **How to change it:** edit every copy; `tests/test_dolle_core.py` (`test_every_material_value_equals_dolle_sim`,
  `test_pair_properties_equal_pair_props`) and `tests/test_core_vs_dem.py` (`test_friction_table_equals_engine`,
  `test_wall_limits_equal_make_params`) show what is out of step.
- **References:** `commit:90a0058`.

### E8. The dependency rule: dolle_sim never imports the extras
- **Decision:** `dolle_extras` may import `dolle_sim` and `dolle_core`; `dolle_sim` never imports `dolle_extras`,
  not even lazily; `dolle_core`'s physics imports neither simulator; moved commands print where they went.
- **Date:** 2026-11-11 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** optional imports inside `dolle_sim`.
- **Why:** the main package must run, and be understood, without the extras.
- **Effect on results:** none.
- **How to change it:** `tests/test_provenance.py` (`test_dolle_sim_never_imports_extras`,
  `test_main_package_runs_without_extras`) enforces it.
- **References:** `commit:13dd8e8`; `extras/README.md`.

### E9. A 3-D simulation in the readable core, with rigid clumps
- **Decision:** `dolle_core/clumps.py`, `dolle_core/impacts.py`, `dolle_core/sim3d.py` and `dolle_core/scenes.py`:
  grains fly under the residual acceleration, each impact is decided by the verified 1-D contact physics (B6), and
  sticking grains merge into rigid clumps (B12); exact contact times with a brute-force candidate list (D5); no
  scipy, so it runs in the browser.
- **Date:** 2026-11-11 · **Decided by:** Owner (a new 3-D page with rigid clumps, one of his strip-down choices,
  prompted by the Team reviewer's request of 2026-11-10), Engineering (method) · **Status:** in force.
- **Alternatives considered:** full DEM (~4e9 steps for 12 s); hard spheres with a fixed restitution (nothing could
  stick).
- **Why:** shows clumps forming, with physics a reader can follow in `notebooks/06_the_3d_simulation.ipynb`.
- **Effect on results:** linear and angular momentum to round-off through every bounce, merge and split; outcomes
  agree with the v0.5 engine on identical starts (`tests/test_dolle_core_sim3d.py`,
  `test_100_grain_ensemble_matches_dolle_sim`).
- **How to change it:** `dolle_core.sim3d.Tubule` fields (`dt`, `mu`, `e_t`, `fragmentation`, `separation`,
  `n_sub`); `dolle_core.sim3d.step`.
- **References:** `commit:784e16a`.

### E10. State files saved before v0.7 no longer load
- **Decision:** checkpoints and autosaves name the old module paths and do not load after the move; accepted.
- **Date:** 2026-11-11 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** a compatibility shim for the old module paths.
- **Why:** they could not continue bit for bit anyway (the default forces changed), and the campaign was stopped.
- **Effect on results:** v0.6 runs restart from lift-off.
- **How to change it:** not planned.
- **References:** `commit:13dd8e8`; `extras/docs/CAMPAIGN.md`.

## F. The flight and the experiment

### F1. Whole-flight mission runs on one clock, one chamber per tubule
- **Decision:** every tubule on the deck is simulated on one shared flight clock from lift-off, each tubule its own
  chamber; scenarios are named changes to the mission config and combine with "+".
- **Date:** 2026-10-22 (v0.4) · **Decided by:** Owner (the whole flight), Engineering · **Status:** in force.
- **Alternatives considered:** separate simulations per phase.
- **Why:** the loads of one phase set the state of the next.
- **Effect on results:** see A3 and F14.
- **How to change it:** `python -m dolle_sim mission configs/mission_quarter_deck.json --scenario kick+lying --jobs
  4`; `dolle_sim.mission.run_mission`, `dolle_sim.mission.apply_scenario`.
- **References:** `commit:a866c4e`; `docs/MISSION_RUNS.md`.

### F2. A quarter deck with four glass tubules (an example layout)
- **Decision:** four rectangular glass tubules of 0.5 x 0.5 x 1.5 in (12.7 x 12.7 x 38.1 mm, the given size used as
  the inside size) stand on a quarter deck (deck centre on the spin axis); example contents T1 silica 90-106 um on
  bare borosilicate, T2 graphite 90-106 um bare, T3 silica 45-53 um silanised, T4 graphite 45-53 um silanised.
- **Date:** 2026-10-22 · **Decided by:** Team (hardware concept: quarter deck, four glass tubules), Engineering
  (example contents and positions) · **Status:** in force as an example; positions, materials, amounts and
  orientation are placeholders ("replace them with the team's actual powders, glass and layout", the config's own
  comment).
- **Alternatives considered:** tubules lying (scenario `lying`).
- **Why:** deck dimensions from the RockSat User Guide 2026 (Secs. 1.5 and 5).
- **Effect on results:** positions set the centrifugal loads (the far corner sees ~9.6 g at 4.8 Hz); the small
  grains of T3/T4 have ~1.8x higher thresholds and, on the same wall, ~4x higher pull-off limits.
- **How to change it:** `tubules`, `inner_size_in`, `deck_position_m`, `orientation`, `wall_material` in
  `configs/mission_quarter_deck.json`.
- **References:** `commit:a866c4e`.

### F3. The flight timeline: published numbers, with stated envelope assumptions
- **Decision:** burns 0-5.2 s and 17.4-29.1 s (coast 12.2 s), 25 g, spin to 4.8 Hz at burnout, power-off T+332 s,
  parachute T+460 s, splashdown T+882 s (RockSat User Guide 2026, Table 3-1 and Sec. 3.3); de-spin at T+54 s and the
  100 km crossings from published mission fact sheets of the same vehicle; yo-yo de-spin 0.9 s (Ettl & Pfaender
  2009). ASSUMED: thrust flat at the 25 g envelope inside each burn, linear spin-up, the payload keeps its attitude
  above ~100 km, the guide's 50 g+ impulses not in the default timeline, the ~+/-10 g lateral ascent load (guide Sec.
  5.2.5) only in the load report.
- **Date:** 2026-10-22; extended 2026-10-26 · **Decided by:** Owner (cite the guide by its title), Engineering
  (assumptions) · **Status:** in force; replace with accelerometer data when available.
- **Alternatives considered:** a measured acceleration history (`environment.accel_csv`, none available).
- **Why:** the best published sources found.
- **Effect on results:** the 25 g thrust presses dust onto the aft face and is irrelevant to pull-off (thresholds
  ~500-8,700 g, `docs/MISSION_RUNS.md`); the spin is the main sideways load (A5).
- **How to change it:** any key of `flight_profile` in the mission config (`burn1`, `burn2`, `burn_g`,
  `spin_burnout_hz`, `despin_s`, ...); `dolle_sim.flight.FlightProfile`. Re-run `tests/test_flight.py`,
  `python -m dolle_sim verify` (`check:flight_frame`, `check:settling`), `tests/test_notebooks.py`.
- **References:** `commit:a866c4e`, `commit:0731b2f`; `PROVENANCE.md` (flight timeline).

### F4. Residual spin after de-spin: 1 rpm, ASSUMED and swept
- **Decision:** residual spin 1/60 Hz (1 rpm, the team's earlier assumption), with named alternatives from published
  sources: 1.0 Hz (guide upper value), 0.5 Hz (an earlier guide), 0.25 Hz (yo-yo tolerance), 0.08 Hz (another
  sounding-rocket programme's specification), 0.030 Hz (measured on one such flight), 5.6e-4 Hz (active rate
  control).
- **Date:** 2026-10-20 (1 rpm); 2026-10-22 (options) · **Decided by:** Team (1 rpm), Engineering (options) ·
  **Status:** in force; ASSUMED.
- **Alternatives considered:** the options above (scenario `spin_1hz`).
- **Why:** no measured value for this payload.
- **Effect on results:** centrifugal acceleration at 0.10 m from the axis: 1 rpm 1.1e-4 g, 0.25 Hz 0.025 g,
  0.5 Hz 0.10 g, 1 Hz 0.40 g. Anything above ~0.03 Hz dominates the 1e-4 g residual and decides where freed grains
  drift.
- **How to change it:** `flight_profile.residual_spin_hz` (a number or a key of
  `dolle_sim.flight.RESIDUAL_SPIN_OPTIONS_HZ`, e.g. `"guide_upper"`).
- **References:** `commit:a3fffe8`, `commit:a866c4e`.

### F5. Impacts are counted only inside the microgravity window, T+88-313 s
- **Decision:** impact statistics are counted between the ~100 km crossings (T+88 s and T+313 s).
- **Date:** 2026-10-22 · **Decided by:** Review (the v0.4 draft counted to T+332 s although the payload is below
  100 km from ~T+313 s), Engineering · **Status:** in force.
- **Alternatives considered:** counting to power-off.
- **Why:** outside the window the residual acceleration is not microgravity.
- **Effect on results:** defines the measurement window (225 s).
- **How to change it:** `flight_profile.ug_start_s`, `flight_profile.ug_end_s`.
- **References:** `commit:a866c4e`.

### F6. Residual acceleration: 1e-4 g, ASSUMED, in the direction of the drag
- **Decision:** a quasi-steady residual of 1e-4 g (ASSUMED order of magnitude, a floor on the drag), whose direction
  follows the drag and therefore reverses at apogee (T+~197 s).
- **Date:** 2026-10-20 (magnitude 1e-4 g, ASSUMED: in the environment preset of the first commit, `commit:a3fffe8`, now
  named `sounding_rocket` in `dolle_sim.environment.PRESETS`); 2026-10-22 (direction follows the drag,
  `commit:a866c4e`) · **Decided by:** Engineering (magnitude); Review (the v0.4 draft kept a pull toward the nose
  for the whole descent) and Engineering (direction) · **Status:** in force.
- **Alternatives considered:** drag alone (Ettl & Pfaender 2009 give ~2e-6 g at apogee); a fixed direction.
- **Why:** drag follows the velocity; the residual decides where freed grains drift.
- **Effect on results:** in the 3-D page run, 1e-3 g instead of 1e-4 g leaves grain-grain outcomes unchanged
  (silica 21/4, graphite 17/17) and sends more grains to the walls.
- **How to change it:** `flight_profile.residual_g`; field `accel` of `dolle_core.sim3d.Tubule`; `residual_g` of
  `dolle_core.scenes.two_tubule_start`.
- **References:** `commit:a3fffe8`, `commit:a866c4e`.

### F7. Re-entry: a point-mass fall through a tabulated atmosphere
- **Decision:** a 2-D point-mass fall from apogee through an NRLMSIS 2.1 density table (Emmert et al. 2021; launch
  site latitude 37.9 N, June, F10.7 = 150, Ap = 4; 1 km table); the ballistic coefficient (~320 kg/m^2) is solved so
  the payload is at 4.6 km at the guide's parachute time; ascent drag 3 g at burnout (derived from fact-sheet
  trajectory points).
- **Date:** 2026-10-26 · **Decided by:** Engineering · **Status:** in force.
- **Alternatives considered:** US Standard Atmosphere 1976 (differs ~3% at sea level, ~25% at 100 km).
- **Why:** density is needed only to order of magnitude.
- **Effect on results:** re-entry peak 7.9 g at T+369 s, sideways along the aft face (the payload falls side-on),
  above every rolling limit.
- **How to change it:** `flight_profile.descent_beta_kg_m2`, `flight_profile.descent_attitude`; the table
  `dolle_sim/atmosphere.py`, regenerated by `extras/tools/make_atmosphere.py`.
- **References:** `commit:0731b2f`; `check:reentry` (1.2e-6 against Allen & Eggers 1958, gravity off).

### F8. Parachute, descent and splashdown
- **Decision:** a 5 g half-sine parachute opening (the 1975 parachute design-criteria report M-EH-75-2: "up to
  5 g"); descent speed set so the payload reaches the sea at T+882 s (~9.4 m/s); splashdown removes that momentum in
  0.05 s (ASSUMED; ~31 g); the payload hangs nose down under the canopy and floats on its side (a 2016 student flight
  report on a related rocket); shocks along the hanging axis by default (ASSUMED).
- **Date:** 2026-10-26 · **Decided by:** Engineering; Review (the shock direction made an option) · **Status:** in
  force.
- **Alternatives considered:** `hang_nose_up`, `shock_oblique` scenarios; `descent_attitude: "tumbling"` (rotates
  the load direction only).
- **Why:** the impulse is fixed by momentum; the peak scales as 1/stopping time.
- **Effect on results:** ~31 g frees nothing on clean or silanised glass; the stopping time would have to be ~3 ms
  (~500 g) to start pulling off the weakest 10% of T1 silica grains. Hanging nose up decides whether the splash pulls
  dust off the aft face in `rough_grains`.
- **How to change it:** `flight_profile.opening_peak_g`, `flight_profile.splash_stop_s`,
  `flight_profile.hang_attitude`, `flight_profile.shock_direction`.
- **References:** `commit:0731b2f`; `docs/MISSION_RUNS.md`.

### F9. The fall attitude starts at re-entry; the re-entry peak is quoted as ~8-11 g
- **Decision:** the side-on fall attitude takes over at T+313 s, not at apogee; the 7.9 g model peak is quoted as
  ~8-11 g, because point-mass models read 5-20% low against measured flights, and the re-entry check is described as
  testing the integrator, not the peak.
- **Date:** 2026-10-26 · **Decided by:** Review, Engineering · **Status:** in force; every scenario was re-run.
- **Alternatives considered:** the draft switched attitude at apogee (T+197 s, inside the window, turning the
  residual sideways); the draft called the peak "validated".
- **Why:** nothing turns the payload in vacuum; the Allen & Eggers check runs with gravity off.
- **Effect on results:** the residual reverses along the axis in the window instead of turning sideways.
- **How to change it:** `flight_profile.attitude_change_s` (default: the end of the window).
- **References:** `commit:0731b2f`; `docs/HISTORY.md` (code reviews).

### F10. An analytic launch-load report with a random-vibration estimate
- **Decision:** per phase and tubule, the report compares pull-off thresholds with the loads, flags sliding
  (mu = 0.3, with its own formula, F20), and estimates random-vibration exceedance with Rice's up-crossing rate
  (Rice 1944) for the flight, the ground random test (10 Grms thrust axis, 7.6 Grms lateral, flat 20-2000 Hz) and the
  sine test (RockSat User Guide 2026, Appendix B).
- **Date:** 2026-10-22 · **Decided by:** Review (the v0.4 "upper bound" counted the 25 g thrust as holding dust down
  and ignored the coast and the ground test), Engineering · **Status:** in force; an ESTIMATE (Gaussian, flat
  spectrum; real, peakier vibration frees more).
- **Alternatives considered:** the v0.4 upper bound (H8).
- **Why:** dust must survive weeks of ground testing before flight.
- **Effect on results:** clean glass: 0.0% freed; `rough_grains`: the ground test alone frees ~18-88%. The report
  has no rolling check (B8).
- **How to change it:** `python -m dolle_sim mission configs/mission_quarter_deck.json --loads-only`;
  `dolle_sim.mission.launch_report`, `dolle_sim.mission.VIBRATION_TEST`, `vibration_test` in the mission config.
- **References:** `commit:a866c4e`.

### F11. v0.6 engine: release waveforms start and end at rest; random vibration only as an upper-bound scenario
- **Decision:** a shaker or kick moves the tubule with acceleration -a sin(2 pi f t) over whole cycles; the guide's
  test spectra (thrust 0.051 g^2/Hz, lateral 0.029 g^2/Hz, 20-2000 Hz) are synthesised as Gaussian deck
  accelerations (Shinozuka & Jan 1972; Smallwood & Paez 1993) only in the `launch_vibration` scenario; nominal runs
  carry no burn vibration.
- **Date:** 2026-10-28 · **Decided by:** Engineering · **Status:** in force (extras; the v0.5 engine keeps its
  waveform, which starts with a jump to A omega).
- **Alternatives considered:** a flight vibration spectrum (none of this vehicle is public); vibration in every run.
- **Why:** the guide's level is a ground acceptance test ~11-12 dB above the one public flight measurement on a
  related rocket.
- **Effect on results:** `check:random_vibration` (0.73 dB); a kick integrates to zero velocity.
- **How to change it:** `extras/dolle_extras/vibration.py`; `--scenario launch_vibration`.
- **References:** `commit:75fa4e8`.

### F12. The dust starts settled as one layer on the aft face (ASSUMED)
- **Decision:** at lift-off the dust rests on the aft face as one layer of non-touching grains (random sequential
  placement, centres >= 1.06 x the radius sum apart), each in adhesive contact with the wall.
- **Date:** 2026-10-22 · **Decided by:** Engineering · **Status:** in force; ASSUMED.
- **Alternatives considered:** a compacted multi-layer bed with pre-formed clusters (real powder; not modelled); a
  floating start (`"initial_state": {"type": "free"}`).
- **Why:** the simplest defined start.
- **Effect on results:** 800 grains of 90-106 um cover ~4% of the face; real powder would contain grain-grain bonds
  from the start.
- **How to change it:** `initial_state` in a tubule; `dolle_sim.simulation.ChamberSim._settle_on_face`.
- **References:** `commit:a866c4e`.

### F13. Dust amount: 800 grains per tubule in the config, 100 in the simulations
- **Decision:** the mission config carries 800 grains per tubule as an example; simulations (the stopped v0.6
  campaign, the 3-D page) use 100 per tubule in every scenario.
- **Date:** 2026-10-22 (800); 2026-10-29 (100) · **Decided by:** Engineering, under the Owner's instruction to finish
  the campaign; the amount itself is a Team decision not yet made · **Status:** in force as a convention; open
  (O12).
- **Alternatives considered:** 800 (with the v0.6 engine ~1e4 s of computing per flight-second in the coasts with
  ~60 grains in the full model: weeks); different amounts per scenario.
- **Why:** equal amounts keep scenarios comparable; impact counts scale with N^2.
- **Effect on results:** impact counts ~N^2: 800 grains give ~64x the impacts of 100 in the same volume.
- **How to change it:** `dust.count` per tubule, or `python -m dolle_sim mission ... --grains N`; GRAINS in
  `extras/tools/run_campaign.py`; `n` of `dolle_core.scenes.two_tubule_start`.
- **References:** `commit:a866c4e`, `commit:521ae1b`; `extras/docs/CAMPAIGN.md`.

### F14. Release mechanism, gas fill and orientation are compared as scenarios, not chosen
- **Decision:** the undecided hardware choices are scenarios: `no_release`; the nominal 50 g / 200 Hz shaker for
  10 s from T+90 s; `strong_shaker` (500 g, 1 kHz, 5 s); `kick` (three single 5,000 g / 5 kHz cycles at T+90, 150,
  210 s); `lying`; `spin_1hz`; `gas_100pa`, `gas_1atm`; `rough_grains`; `no_drag`; `descent_tumbling`;
  `hang_nose_up`; `shock_oblique`; `launch_vibration`.
- **Date:** 2026-10-22 (v0.4); extended 2026-10-26 · **Decided by:** Team (chose "compare") · **Status:** in force;
  the release mechanism is the main open design question (O3).
- **Alternatives considered:** pick one design now.
- **Why:** the team has not decided.
- **Effect on results (v0.5 engine, computed on 2026-10-26 at `commit:0731b2f`, `docs/MISSION_RUNS.md`):** the
  nominal shaker frees 0 of 3,200 grains; only
  kicks free dust (500-800 per tubule), at a median ~1.2-1.5 m/s, far above the sticking speeds; a gentler release
  (higher frequency, or slowing the grains) is needed.
- **How to change it:** `release` per tubule; `scenarios` in `configs/mission_quarter_deck.json`;
  `--scenario NAME` (combine with "+").
- **References:** `commit:a866c4e`, `commit:0731b2f`.

### F15. Gas: air at 1e-3 Pa in the mission (ASSUMED)
- **Decision:** the mission tubules hold air at 1e-3 Pa and 293 K; the flight demo uses 1 Pa; scenarios `gas_100pa`
  and `gas_1atm`.
- **Date:** 2026-10-22 · **Decided by:** Engineering · **Status:** in force; no source is given in the code:
  ASSUMED; the real fill is undecided (O3).
- **Alternatives considered:** the scenarios above; other gases (N2, Ar, He tables in the code).
- **Why:** a representative low pressure.
- **Effect on results:** at 1e-3 Pa drag is negligible over seconds (B18); at 100 Pa or 1 atm drag slows freed
  grains.
- **How to change it:** `gas.pressure_pa`, `gas.gas`, `gas.temperature_k`; `dolle_sim.environment.GASES`.
- **References:** `commit:a866c4e`.

### F16. Grains must keep bouncing off the walls: anti-stick walls and vibrating walls in the flight demo
- **Decision:** the flight demo config (two 3 cm cubes, 2,000 grains each of 50 um radius) uses fluorosilane-coated
  walls with adhesion factor 0.05 and walls vibrating at 200 Hz (25 um for graphite, 5 um for silica), launch speeds
  straddling each predicted sticking speed (graphite 35 +/- 20 mm/s, silica 7 +/- 4 mm/s), spin 5 +/- 3 Hz,
  charges sigma 20,000 e, 1 Pa. All demonstration values are ASSUMED.
- **Date:** 2026-10-20 · **Decided by:** Owner (grains must bounce off walls, because the data are grain-grain
  impacts), Engineering (values) · **Status:** in force for the demo; for the flight hardware undecided (O11).
- **Alternatives considered:** bare glass (captures grains within a couple of crossings; a 50 um radius silica
  grain sticks to bare borosilicate below 6.66 mm/s, `dolle_core.collide.sticking_threshold`, C6); PTFE;
  non-adhesive walls.
- **Why:** shows that the camera can measure thresholds when grains keep colliding.
- **Effect on results:** 0.02-0.09% of wall hits stick (`docs/USER_GUIDE.md`); this config's force budget is the
  one in the README.
- **How to change it:** `configs/flight_carbon_vs_silica.json`.
- **References:** `commit:a3fffe8`.

### F17. Grain size distributions
- **Decision:** sizes from sieve ranges, D10/D50/D90 (Hatch-Choate conversion between volume and number weighting;
  Hatch & Choate 1929), log-normal or measured lists; the simulation samples number-weighted. A sieve range with a
  fraction inside is a log-normal centred on sqrt(min max) (ASSUMED shape); without a fraction, uniform in ln d.
- **Date:** 2026-10-22 · **Decided by:** Engineering · **Status:** in force; image-based sizing moved to `extras/`,
  untested on real micrographs.
- **Alternatives considered:** a single radius.
- **Why:** suppliers give sieve bands or laser-diffraction percentiles (volume weighted); the simulation needs number
  weighting.
- **Effect on results:** thresholds scale as radius^(-5/6) (A2).
- **How to change it:** `size_distribution` in a config; `python -m dolle_sim sizes --d10 ... --d50 ... --d90 ...`;
  `dolle_sim.sizes.sieve_lognormal`; `dolle_core.scenes.sieve_radii`. Re-run `tests/test_sizes.py`.
- **References:** `commit:a866c4e`; `extras/dolle_extras/sizing.py`.

### F18. Two orthogonal cameras (stereo) are recommended
- **Decision:** recommend two orthogonal views (or one camera and a mirror), analysing free single-grain impacts
  along the line of centres at contact with ~1 um centroiding. The synthetic camera's settings (500 fps, 5 um
  position noise, 6 frames tracked, 1 cm depth of field) are ASSUMED.
- **Date:** 2026-10-20 (v0.2) · **Decided by:** Engineering (a design recommendation to the Team) · **Status:** in
  force.
- **Alternatives considered:** one camera with the 2-D relative speed or 2-D normal component.
- **Why:** in the flight demo one camera reads 3.5-4.2x too low; stereo reads -1.8% (graphite) and -5.5% (silica)
  with 5 um centroids, -0.3% / -1.7% with 1 um. The one-camera bias depends on geometry (1.1-1.6x in an opposed-cloud
  run), so it cannot be calibrated away; a faster camera with the same number of frames is worse (noise ~ fps /
  k^1.5).
- **Effect on results:** decides whether the threshold can be measured at all.
- **How to change it:** `camera` in a config (`fps`, `position_noise_m`, `track_frames`, `strategy`);
  `dolle_sim.analysis.synthetic_camera`.
- **References:** `commit:8060592`; `docs/USER_GUIDE.md` (results).

### F19. The threshold fit, with its coverage caveat stated
- **Decision:** the "measured" threshold is the 50% point of a maximum-likelihood logistic fit of stick against
  ln(speed), with 200 bootstrap resamples (Efron 1979) and a bounded fit so perfectly separated samples do not
  overflow; the classification of impacts is perfect (simulation truth) and impacts within 0.05 s of a
  fragmentation are excluded (ASSUMED).
- **Date:** 2026-10-20 · **Decided by:** Engineering · **Status:** in force; the error bars are too narrow (O14).
- **Alternatives considered:** an errors-in-variables fit (not done).
- **Why:** standard for binary outcomes.
- **Effect on results:** `check:threshold_fit`: error < 3.3%, but the 1-sigma bars covered the truth in only 9 of 20
  cases (reported, not hidden). Velocity noise (~0.85 mm/s per axis) biases the stereo threshold low by a few %.
- **How to change it:** `dolle_sim.analysis.fit_threshold`; `dolle_sim.simulation.FRAG_TAG_S`.
- **References:** `commit:a3fffe8`, `commit:f5b27a8`.

### F20. The launch-load report's own sliding flag: mu (F_c - pull)
- **Decision:** the launch-load report flags a grain as able to slide when the load along the aft face exceeds
  mu x (a_th - pull), with mu = `contact.friction_coefficient` (default 0.3), a_th = F_c/(m g) the grain's pull-off
  threshold and pull the load off the face at the same instant (`dolle_sim.mission.launch_report`). In force terms
  the friction capacity is mu (F_c - F_pull). This differs from the wall limit of B9, mu |F_ne + 2 F_c|, used by
  `dolle_core.wall.wall_limits` and the v0.6 engine.
- **Date:** 2026-10-22 · **Decided by:** Engineering · **Status:** in force; to be aligned with B9 (O15).
- **Alternatives considered:** B9's formula with the per-pair friction table (written six days later, for the v0.6
  engine, 2026-10-28).
- **Why:** written for v0.4, before the v0.6 contact model existed, as a flag only (the v0.5 engine does not model
  sliding; the code comment says "sliding (not modelled)").
- **Effect on results:** at zero external load the report's sliding capacity is 0.3 F_c; `dolle_core` and the v0.6
  engine give 0.4 x 2 F_c = 0.8 F_c for silica on glass (the report is 62.5% lower) and 0.5 x 2 F_c = 1.0 F_c for
  graphite (70% lower). For the example tubules on clean or silanised glass it changes nothing: even at 0.3 F_c the
  capacity (about 0.3 x 498 g = 150 g for the weakest 10% of T1 silica) is far above the largest sideways load,
  19.5 g, and the report prints "could slide: 0.0%" for every phase (`python -m dolle_sim mission
  configs/mission_quarter_deck.json --loads-only`). Rolling, the weakest limit, is not in the report at all (F10).
- **How to change it:** the line computing `cap` in `dolle_sim.mission.launch_report` (mu x |2 a_th - pull| would be
  B9's form, with pull counted positive away from the face) and
  `contact.friction_coefficient` in the mission config; re-run `python -m pytest tests/test_flight.py -q` and the
  `--loads-only` command above.
- **References:** `commit:a866c4e`; `notebooks/04_grain_on_a_wall.ipynb` section 3 (B9's formula).

### F21. Environment presets and the g-jitter (ASSUMED)
- **Decision:** a chamber's environment starts from a named preset in `dolle_sim.environment.PRESETS`, whose values a
  config can override: `sounding_rocket` (the default: residual 1e-4 g, g-jitter 5e-5 g rms with a 0.05 s
  correlation time, 1 rpm roll, 210 s window), `spinning_rocket`, `ideal` and `drop_tower`. The g-jitter is a random
  acceleration that relaxes exponentially with that correlation time (`dolle_sim.environment.Environment`). The
  flight demo uses `sounding_rocket`; mission runs use `ideal` plus the flight profile, with g-jitter off unless the
  mission config sets `jitter_rms_g` (`dolle_sim.mission.mission_cfg`); the v0.6 engine refuses g-jitter (D25).
- **Date:** 2026-10-20 (`commit:a3fffe8`) · **Decided by:** Engineering · **Status:** in force; every preset value is
  ASSUMED ("representative assumptions, not vehicle data", `dolle_sim/environment.py` docstring).
- **Alternatives considered:** a measured acceleration history (`environment.accel_csv`; none is available); no
  jitter.
- **Why:** a despun sounding rocket's quiet window needs a stated residual and jitter level until accelerometer data
  exist.
- **Effect on results:** g-jitter is the same for every grain in a chamber, so it changes no relative velocity
  between grains (`notebooks/06_the_3d_simulation.ipynb` section 9); it moves grains relative to the walls. In the
  flight demo's force budget it is 5e-5 g, half the residual (`python -m dolle_sim preflight
  configs/flight_carbon_vs_silica.json`).
- **How to change it:** `environment.preset`, `environment.jitter_rms_g`, `environment.jitter_corr_time`,
  `environment.residual_accel_g`, `environment.accel_csv` in a chamber config; `jitter_rms_g` at the top level of a
  mission config.
- **References:** `commit:a3fffe8`, `commit:30840b8` (the v0.6 engine refuses it); `PROVENANCE.md`.

## G. Team decisions and project process

### G1. Public text names only "DOLL-E"
- **Decision:** descriptive public text (headlines, website and interface text, README, configs, command-line
  output) names no programme, agency or institution, only "DOLL-E". Published sources may be cited by their full
  reference, including the issuing organisation, wherever their numbers are used (for example the RockSat User Guide
  2026).
- **Date:** 2026-10-21 (revised the same day to allow citations) · **Decided by:** Owner · **Status:** in force.
  Note: this log, and the brief it was written to, follow a stricter convention and cite documents by title only.
  The module docstring of `dolle_sim/flight.py`, the generated `PROVENANCE.md` (from `dolle_sim/provenance.py`) and
  the source column of `docs/MISSION_RUNS.md` name the issuing organisations inside citations of the timeline
  sources: that follows the owner's rule, though not this log's stricter convention.
- **Alternatives considered:** no rule.
- **Why:** the owner's choice for public material.
- **Effect on results:** none.
- **How to change it:** to apply the stricter convention everywhere, edit the texts named above (then
  `python tools/build_site.py` for `PROVENANCE.md`); `tests/test_decisions_doc.py` checks this file for e-mail
  addresses and for the guide being cited by title.
- **References:** `commit:ed2f49d`, `commit:a866c4e`.

### G2. The git history is not rewritten
- **Decision:** older commit messages contain names that the later naming rule forbids, and the author field of
  every commit (all 29 so far, the newest included) carries the owner's institutional e-mail address; new commits
  keep adding it while the local git `user.email` is unchanged. The history was not rewritten.
- **Date:** project notes · **Decided by:** Engineering (no history rewrite without the owner's explicit consent);
  the owner has not decided (O17) · **Status:** in force; open (O17).
- **Alternatives considered:** rewrite the messages AND the author e-mail (for example with a mailmap and
  git filter-repo), set a no-reply `user.email` for future commits, then force-push. This changes every commit hash
  and breaks every reference by hash, including this file's.
- **Why:** rewriting shared history needs the owner's explicit consent.
- **Effect on results:** none.
- **How to change it:** as in the alternatives (O17); then update every `commit:` reference in this file and run
  `python -m pytest tests/test_decisions_doc.py -q`.
- **References:** project notes.

### G3. The v0.6 campaign: every scenario to splashdown, scheduled tubule by tubule
- **Decision:** `extras/tools/run_campaign.py` runs every scenario to splashdown with the v0.6 engine; each
  (scenario, tubule) is one job; a pool takes the next job that can start; a branched tubule starts once its source
  saved the checkpoint.
- **Date:** 2026-10-28 · **Decided by:** Owner ("just finish the full simulation", project notes), Engineering ·
  **Status:** stopped (G6).
- **Alternatives considered:** whole scenarios two at a time (cores idle for hours: a scenario's tubules differ up to
  ~3x in run time), H15.
- **Why:** use every core.
- **Effect on results:** same configs and seeds give identical results (`test_tubule_by_tubule_run_is_the_mission_run`).
- **How to change it:** `extras/tools/run_campaign.py`, `extras/tools/mission_table.py`.
- **References:** `commit:75fa4e8`, `commit:2c231b4`, `commit:fbac976`; `extras/docs/CAMPAIGN.md`.

### G4. Autosave and resilience for multi-day runs
- **Decision:** whole-state autosave every 30 minutes of computing; a dead job resumes from it, bit for bit the same
  as an uninterrupted run WITHIN ONE CODE VERSION; out-of-memory deaths are retried once memory is free, other
  failures up to 3 times; a job starts only if its memory can be committed; the scheduler is restartable (adopts
  running jobs) and imports none of the simulator (it held ~0.5 GB).
- **Date:** 2026-10-29 · **Decided by:** Engineering · **Status:** in force (extras).
- **Alternatives considered:** restart failed jobs from lift-off.
- **Why:** multi-day runs on a ~16 GB laptop ran out of memory.
- **Effect on results:** none on answers (`test_killed_tubule_run_continues_from_its_autosave`).
- **How to change it:** AUTOSAVE_MIN, RETRIES and JOB_MEMORY_GB in `extras/tools/run_campaign.py`;
  `python -m dolle_extras mission ... --autosave-min`.
- **References:** `commit:521ae1b`, `commit:68b1982`, `commit:3d27407`, `commit:6089bb1`, `commit:e512b2d`,
  `commit:874976e`.

### G5. Measure before optimising; stop the campaign and discard its results
- **Decision:** load the campaign's saved states and examine every cluster the engine refused to put to sleep; then
  stop the campaign and discard its results; use the results of a restarted campaign only after the new treatment's
  checks pass. (The restart itself did not wait for the checks: see G6 and D35.)
- **Date:** 2026-10-29 (project notes) · **Decided by:** Engineering (owner informed) · **Status:** done.
- **Alternatives considered:** keep running; optimise without measuring.
- **Why:** static clusters (moving at 1e-13 to 2e-9 m/s) were integrated at 5-18 ns steps because a contact sat at
  its rolling limit (ratio >= 0.999); the coasts cost 1e3-1e4 s of computing per flight-second
  (`extras/docs/design/at-limit-sleep/01_brief.md`). Results of the old engine would not match the new one, and the
  bit-for-bit continuation holds only within one code version.
- **Effect on results:** led to D27-D35.
- **How to change it:** not a parameter.
- **References:** `commit:726ec34`, `commit:937a6dd`.

### G6. The campaign was restarted four times, then stopped for good
- **Decision:** the campaign was restarted on 2026-10-28 (the tubule-by-tubule scheduler, `commit:fbac976`), on
  2026-10-29 (100 grains per tubule, autosave, `commit:521ae1b`), on 2026-11-10 from a pinned copy at
  `commit:726ec34` while checks 14-19 were still being written (its results gated on them, D35), and again from
  scratch at `commit:f133c3e` after the checks and the review fixes (the `commit:726ec34` run set aside); the same
  day, after the team's feedback (G7), the owner stopped the flight campaign for good.
- **Date:** 2026-10-28 to 2026-11-10 (restarts); 2026-11-10 (stop, project notes) · **Decided by:** Engineering
  under the Owner's instruction to finish the campaign (restarts); Owner, following the Team reviewer's feedback
  (stop) · **Status:** final.
- **Alternatives considered:** finish the campaign.
- **Why:** understanding first (G7). The new engine ran the nominal boost at ~14 s of computing per flight-second
  (87-100 of 100 grains asleep); after 5.5 h the first restart had reached T+66 s in one tubule, T+12 s in another.
- **Effect on results:** there are NO v0.6 scenario results; the scenario table is from the v0.5 engine
  (`docs/MISSION_RUNS.md`, `notebooks/05_the_flight_in_brief.ipynb`).
- **How to change it:** `python extras/tools/run_campaign.py` (O16).
- **References:** `commit:fbac976`, `commit:521ae1b`, `commit:874976e`, `commit:13dd8e8`;
  `extras/docs/CAMPAIGN.md`.

### G7. Team feedback of 10 November: understanding first
- **Decision:** methods importable in Jupyter notebooks; a demonstration of momentum conservation; explanations of
  why the code does what it does and of what changes from silica to carbon; strip extra features as far as possible.
  The owner chose to build a small readable core first, then prune the repository feature by feature with the
  coding sessions, and to stop the campaign for good.
- **Date:** 2026-11-10 · **Decided by:** Team reviewer (feedback), Owner (choices) · **Status:** in force.
- **Alternatives considered:** keep the 15,000-line simulator as the front door; finish the campaign.
- **Why:** a simulator its owner cannot explain is of no use to the team; understanding beats completeness.
- **Effect on results:** led to E5, E6, B21, G8-G12.
- **How to change it:** not a parameter.
- **References:** `commit:90a0058`, `commit:13dd8e8`, `commit:784e16a`.

### G8. Notebooks with derivations and "why this choice" boxes; Colab, local or plain scripts
- **Decision:** notebooks 00-06 (start here; two grains colliding; momentum conservation, with a deliberately broken
  version; silica vs carbon one property at a time; a grain on a wall; the flight in brief; the 3-D simulation): full
  derivations from cited papers, every number printed by code, each modelling choice in a box with alternatives and
  evidence; they run locally, in Colab (badge) or as the scripts in `examples/`.
- **Date:** 2026-11-10 (00-05); 2026-11-11 (06) · **Decided by:** Owner and Team reviewer (topics; math "enough to
  explain everything and how it was derived and decided"), Engineering · **Status:** in force.
- **Alternatives considered:** documents only.
- **Why:** the owner must be able to explain every step.
- **Effect on results:** none on numbers; checked by D13.
- **How to change it:** `notebooks/`; `examples/sticking_threshold.py` and the other scripts.
- **References:** `commit:90a0058`, `commit:784e16a`; `notebooks/README.md`.

### G9. Strip-down details: deletions and a short README
- **Decision:** delete the colliding-clouds demo (its config, replay and two figures; it duplicated the flight demo)
  and a data file read by nothing, keeping their results in `docs/HISTORY.md` (recoverable from `commit:90a0058`);
  rewrite the README around the notebooks, `dolle_core` and the 3-D page, moving long sections to
  `docs/USER_GUIDE.md`, `docs/MISSION_RUNS.md` and `docs/HISTORY.md`.
- **Date:** 2026-11-11 · **Decided by:** Owner (each specific choice, agreed feature by feature with the coding
  sessions), prompted by the Team reviewer's request of 2026-11-10 to strip extra features (G7) · **Status:** in
  force.
- **Alternatives considered:** keep everything.
- **Why:** a short front door.
- **Effect on results:** none (the deleted run's result: true thresholds 38.2 / 6.26 mm/s, stereo 37.6 / 6.33).
- **How to change it:** `README.md`.
- **References:** `commit:13dd8e8`.

### G10. Stored results were not re-run after v0.7
- **Decision:** the stored results are kept and labelled, not re-run. The mission scenario table was computed on
  2026-10-26 with the v0.5 code (`commit:0731b2f`, "All twelve scenarios re-run to splashdown"), under the mission
  defaults of that time (negligible forces on), and has not been re-run since: its rows are identical in the README
  from `commit:0731b2f` to `commit:90a0058`, although v0.6 (`commit:75fa4e8`) had made the v0.6 engine the mission
  engine. The flight demo result was computed with the v0.6.2 code, all forces on (`docs/USER_GUIDE.md`, "Results").
  Both predate the stored-bond fix (B16). That the changes stay inside their error bars is ASSUMED, not checked.
- **Date:** 2026-11-11 · **Decided by:** Engineering, with the Owner · **Status:** in force; open (O13).
- **Alternatives considered:** re-run everything (hours to ~10 h per scenario).
- **Why:** the strip-down changed no physics of the default engine except the bond fix and the switched-off forces.
- **Effect on results:** a re-run gives different numbers (Brownian off changes the random stream).
- **How to change it:** `python -m dolle_sim run configs/flight_carbon_vs_silica.json`; `python -m dolle_sim mission
  configs/mission_quarter_deck.json --scenario NAME`; then update `docs/MISSION_RUNS.md` and `docs/USER_GUIDE.md`.
- **References:** `commit:0731b2f`, `commit:30840b8`, `commit:13dd8e8`; `docs/MISSION_RUNS.md` (header).

### G11. The original website runs the real package in the browser
- **Decision:** a GitHub Pages site (`docs/index.html`, the app in `docs/app/`) runs the real `dolle_sim` in the
  browser through Pyodide (preflight, short runs, the sticking calculator, the inversion, the 26 checks, the sources,
  replays), simplified on 2026-10-21 to a short explanation and a six-input quick setup with plain-language results.
- **Date:** 2026-10-20; simplified 2026-10-21 · **Decided by:** Engineering (no source records who asked for the
  site); the simplification is recorded without a requester either · **Status:** in force; unchanged since v0.7
  (G12).
- **Alternatives considered:** a separate JavaScript reimplementation (the browser runs the same code instead).
- **Why:** nothing to install; the same code as the desktop.
- **Effect on results:** the browser uses 100 contact sub-steps (<0.4%, D1).
- **How to change it:** `dolle_sim/webapi.py`; `python tools/build_site.py`.
- **References:** `commit:f5b27a8`, `commit:825984d`, `commit:ed2f49d`.

### G12. Keep the existing website; add a new 3-D page
- **Decision:** leave the existing site as it is (data files refreshed) and build a new page, `docs/sim3d/`: a
  recorded replay of two tubules side by side (silica and graphite, same start) with stick/bounce markers, counters
  and live momentum; one impact up close per material at a sticking and a bouncing speed; a live sandbox running
  `dolle_core` in the browser.
- **Date:** 2026-11-11 · **Decided by:** Owner (each specific choice, agreed feature by feature with the coding
  sessions), prompted by the Team reviewer's request of 2026-11-10 to strip extra features (G7); Engineering (page
  content, agreed with the Owner) · **Status:** in force.
- **Alternatives considered:** rebuild the existing site.
- **Why:** shows the readable core at work; notebook 06 derives each step.
- **Effect on results:** none.
- **How to change it:** `python tools/make_sim3d_replays.py` (replays), `python tools/build_sim3d.py` (copies
  `dolle_core` into `docs/sim3d/py/`); `tests/test_sim3d_page.py`.
- **References:** `commit:13dd8e8`, `commit:784e16a`.

### G13. The 3-D page's start: a compact released cloud (ASSUMED)
- **Decision:** two identical borosilicate tubules, silica and graphite, 100 grains each, the SAME compact cloud (a
  2 mm cube at the tubule centre), isotropic speeds |N(15, 7.5)| mm/s without spin, residual 1e-4 g along +z,
  12 s.
- **Date:** 2026-11-11 · **Decided by:** Engineering (page content agreed with the Owner) · **Status:** in force;
  ASSUMED (the release is undecided, O3).
- **Alternatives considered:** grains spread through the whole tubule (mean free path ~2 m against walls ~6 mm away:
  they essentially never meet; preset "spread" in the sandbox); a kick (nothing sticks).
- **Why:** 15 mm/s is the geometric mean of 6.2 and 38 mm/s, so the same start is fast for silica and slow for
  graphite.
- **Effect on results (measured, impacts / sticks):** baseline silica 20 / 4, graphite 17 / 17. The ASSUMED start
  decides the outcome: a 4 mm cloud gives 3 / 0 and 4 / 4; a 1.5 mm cloud 33 / 9 and 28 / 28;
  `speed_mm_s` = 7.5 with `sigma_mm_s` = 3.75 gives silica 20 / 11; `speed_mm_s` = 30 with `sigma_mm_s` = 15 gives
  22 / 3 and 17 / 14 (sigma half the mean, as in the baseline; with sigma held at 7.5 mm/s the two give silica
  11 / 5, graphite 12 / 12, and silica 12 / 0, graphite 10 / 9). Physics settings barely matter: friction 0.3,
  residual 1e-3 g, no adhesion scatter or no fragmentation give 19-21 / 4-5 and 17 / 17. Seeds 1 and 2: 15 / 4,
  14 / 14 and 18 / 5, 14 / 13.
- **How to change it:** arguments of `dolle_core.scenes.two_tubule_start` (`n`, `seed`, `box_mm`, `speed_mm_s`,
  `sigma_mm_s`, `log_sigma_w`, `residual_g`, `tubule`) and `dolle_core.scenes.run_two_tubules`; the sandbox's
  `dolle_core.scenes.SANDBOX_DEFAULTS`; sources of every input in `dolle_core.scenes.START_SOURCES`. Re-run
  `tests/test_dolle_core_sim3d.py`, `tests/test_sim3d_page.py`, `python tools/make_sim3d_replays.py`, and update
  `docs/OVERVIEW.md` section 2.
- **References:** `commit:784e16a`; `notebooks/06_the_3d_simulation.ipynb`.

### G14. Which carbon flies is not decided
- **Decision:** graphite stays the simulator's carbon as a placeholder; notebook 03 gives threshold bands for
  graphite (22-220 mm/s), glassy carbon (20-127) and amorphous carbon (11-153) from property ranges (mostly
  ASSUMED).
- **Date:** 2026-11-10 · **Decided by:** Team (undecided), Engineering (bands) · **Status:** in force; open (O2).
- **Alternatives considered:** pick a carbon in the simulator now.
- **Why:** the team has not chosen.
- **Effect on results:** "carbon sticks more easily than silica" holds over 1.8x-36x, unless silica's own surface
  energy is as high as argued for vacuum (C1).
- **How to change it:** add the chosen carbon to both material tables (C2).
- **References:** `commit:90a0058`; `notebooks/03_silica_vs_carbon.ipynb`.

### G15. Which silica flies is not decided
- **Decision:** keep the table silica (quartz values) and show the range: fused silica's E, nu and rho lower the
  threshold 14% (5.33 mm/s); whether Stober spheres, fused silica or soda-lime beads fly decides E and rho.
- **Date:** 2026-11-10 (project notes) · **Decided by:** Team (undecided) · **Status:** in force; open (O2).
- **Alternatives considered:** as G14.
- **Why:** the team has not chosen.
- **Effect on results:** -14% to x5.7 depending on the silica and its surface energy (C1).
- **How to change it:** both material tables (C1).
- **References:** `commit:90a0058`.

### G16. How the at-limit sleep was designed: three blind designs, a synthesis, an adversarial critique
- **Decision:** one brief for all; three designers who never saw each other's work (A: an elastic-unloading
  certificate, the smallest change; B: a Newton certificate evaluating the real law at the predicted state; C:
  risk-first, listing every failure mode first); a synthesis judged against the code; an adversarial critique whose
  corrected criteria were implemented; every stage kept verbatim.
- **Date:** 2026-10-29 to 2026-10-31 · **Decided by:** Engineering (multi-agent design) · **Status:** method in
  force.
- **Alternatives considered:** a single design.
- **Why:** independent designs expose each other's blind spots. Rejected parts: A's three samples per step and its
  heuristics; B's long analytic bounds (replaced by the measured half-step pair); C's budgeted flow in a frozen island
  and its active-set Newton (replaced by the strict 1 + 1e-6 test plus a creep step that resolves flow instead of
  bounding it).
- **Effect on results:** D27-D34.
- **How to change it:** not a parameter.
- **References:** `commit:726ec34`; `extras/docs/design/at-limit-sleep/README.md`,
  `extras/docs/design/at-limit-sleep/05_synthesis.md`, `extras/docs/design/at-limit-sleep/06_critique.md`.

### G17. A decision map of the engine, and work on a branch shown on the front page
- **Decision:** `extras/docs/ENGINE_LOGIC.md` lists every v0.6 engine decision with its test, function, reason,
  exactness and verifying check (tested, D14); engine work is committed on the branch `engine-sleep-at-limit`, and
  `main` is fast-forwarded to it so the front page shows the latest state.
- **Date:** 2026-10-31 (decision map requested; branch); 2026-11-10 (committed; front page asked for) · **Decided
  by:** decision map: Owner's request of 2026-10-31; the branch: Engineering (the coding sessions' rule to create a
  branch before committing to the default branch; no pull request was ever opened); fast-forwarding `main` to every
  push: Owner, 2026-11-10 ("merge it into main so it shows on the front page") · **Status:** in force
  (branch = main).
- **Alternatives considered:** work directly on `main`.
- **Why:** every decision's logic visible and verifiable; the front page current.
- **Effect on results:** none.
- **How to change it:** not a parameter.
- **References:** `commit:874976e`, `commit:f133c3e`.

### G18. A 15-minute overview, this decision log and a study guide
- **Decision:** `docs/OVERVIEW.md` (the question, the physics in five steps, key results, structure, verification,
  the main decisions, open items, likely questions), linked from the README; this file, the complete decision log,
  with its drift test; `docs/STUDY_GUIDE.md`, ten study sessions with exercises, self-tests, talk outlines, a
  glossary and the numbers worth knowing.
- **Date:** 2026-11-16 · **Decided by:** Owner · **Status:** in force: the overview (`commit:f0d7307`), this log,
  its test and the study guide (all committed on 2026-11-16).
- **Alternatives considered:** the long documents only.
- **Why:** the owner presents to the team and to officials and must be able to explain and change every decision.
- **Effect on results:** none.
- **How to change it:** edit the files; run `python -m pytest tests/test_decisions_doc.py -q`.
- **References:** `commit:f0d7307`.

### G19. The repository has no licence
- **Decision:** none yet: the repository has no licence file and `pyproject.toml` declares no licence.
- **Date:** state on 2026-11-16 · **Decided by:** not decided; the Owner's choice (O18) · **Status:** open (O18).
- **Alternatives considered:** a permissive licence (MIT, BSD-3-Clause, Apache-2.0: anyone may reuse the code, with
  attribution); a copyleft licence (GPL-3.0, the licence of REBOUND: derived code must stay open under the same
  terms); no licence (the current state).
- **Why:** not yet discussed. It matters for a public repository: without a licence the default copyright rules
  apply, so others may read the code but are granted no right to reuse or modify it (GitHub's documentation on
  licensing a repository). Officials may ask, all the more since E1 names REBOUND's GPL-3.0 as one reason not to
  build on it. The main package needs numpy, scipy and matplotlib (numba optional), all under permissive licences
  (each project's own licence file); nothing from REBOUND or LAMMPS is in the code (D10, D11).
- **Effect on results:** none.
- **How to change it:** add a `LICENSE` file at the top of the repository and a `license` entry in `pyproject.toml`.
- **References:** `pyproject.toml`; E1.

## H. Superseded decisions: what was tried, what replaced it, and why

Most reversals came from reviews (automated adversarial reviews, each finding re-checked by a second reviewer before
it was fixed, `docs/HISTORY.md`) or from a measurement. Three came from the owner: "no shortcuts" (H12), the stop
of the campaign after the team's feedback (H21) and the strip-down choices, prompted by the Team reviewer's request
(H22, H23). Read the table as a short history: v0.2 and v0.3 fixed
the engine's bookkeeping, v0.4 and v0.5 the flight, v0.6 the full-contact engine and its bounds, v0.7 the shape of
the project.

| ID | what was tried (date) | replaced by | why | decided by | commit |
|---|---|---|---|---|---|
| H1 | batched event loop, lax neighbour-list rebuild (2026-10-20) | re-predict after every impact (D5) | could miss follow-on collisions | Review, Engineering | `commit:8060592` |
| H2 | first-order Coriolis; rotation with a stale angular velocity (2026-10-20) | exact velocity operator (D3); DLM rotation (D4) | energy errors | Review, Engineering | `commit:8060592` |
| H3 | fragmentation with the grain-wall bond energy, fragments without separation (2026-10-20) | the clump's own bond energy, fragments separate (B14) | clumps shattered on almost every wall hit; spurious zero-speed sticks | Review, Engineering | `commit:8060592` |
| H4 | fragment energy from the full collision energy (2026-10-20) | the energy the bounce really dissipated, E_diss (B14) | created energy, up to ~1/3 of the impact energy | Review, Engineering | `commit:ed2f49d` |
| H5 | pair forces held for 5 steps, 20 ms (2026-10-20) | recomputed every step (D6) | accuracy | Review, Engineering | `commit:ed2f49d` |
| H6 | an earlier graphite Poisson ratio (2026-10-20) | 0.32 from Chokshi et al. 1993 (C2) | claim audit | Review | `commit:ed2f49d` |
| H7 | residual pull toward the nose for the whole descent; window counted to T+332 s (v0.4 draft) | drag-following residual (F6); window ends T+313 s (F5) | drag reverses at apogee; below 100 km from ~T+313 s | Review, Engineering | `commit:a866c4e` |
| H8 | vibration "upper bound" with the thrust holding dust down (v0.4 draft) | Rice estimate with the thrust ignored, ground tests included (F10) | it was not a bound | Review, Engineering | `commit:a866c4e` |
| H9 | LAMMPS pull-off agreement "0.5%" (v0.4 draft) | 0.14% (D10) | the 0.5% was the detection grid | Review | `commit:a866c4e` |
| H10 | fall attitude switched at apogee, T+197 s (v0.5 draft) | at T+313 s (F9); every scenario re-run | nothing turns the payload in vacuum | Review, Engineering | `commit:0731b2f` |
| H11 | re-entry peak presented as "validated" (v0.5 draft) | ~8-11 g with the caveat (F9) | the check tests only the integrator | Review | `commit:0731b2f` |
| H12 | v0.5 engine for mission runs (2026-10-22) | v0.6 full-contact engine (E3) | owner: "no shortcuts" | Owner | `commit:75fa4e8` |
| H13 | rolling envelope assuming the velocity keeps its direction (v0.6) | speed magnitude, Gronwall bound (D24) | real missed contacts, grains launched at 10-58 m/s | Review, Engineering | `commit:83af20d` |
| H14 | free flight with its own loads, g-jitter included (v0.6.1) | one Loads object; g-jitter and accelerometer files refused (D25) | the bounds never saw those loads | Review, Engineering | `commit:30840b8` |
| H15 | whole scenarios two at a time (2026-10-28) | one job per tubule (G3) | idle cores | Engineering | `commit:2c231b4` |
| H16 | 800 grains per tubule in the campaign (2026-10-28) | 100 (F13) | weeks of computing; out of memory | Engineering, under the Owner's instruction | `commit:521ae1b` |
| H17 | entry at 0.90 for resting clusters (v0.6) | at-limit sleep, tiers 1 and 2 (D27) | static piles at their limits never slept | Engineering | `commit:937a6dd` |
| H18 | ratchet bound E/cap (synthesis, 2026-10-31) | released-system bound (D29) | low by 1e2-1e5 | Review, Engineering | `commit:937a6dd` |
| H19 | entry energy with an "adiabatic factor" (2026-10-31) | drift-relative entry energy (D30) | refused every entry under a turning load | Engineering | `commit:fb4ccf1` |
| H20 | tier-2 dynamics summed per step (2026-10-31) | running maximum of the lag (D31) | woke clusters by budget in 1-2 steps | Engineering | `commit:fb4ccf1` |
| H21 | run the campaign to completion (2026-10-28) | stopped for good (G6) | team feedback: understanding first | Owner, Team reviewer | project notes |
| H22 | v0.6 engine as the default for missions (2026-10-28) | v0.5 default, v0.6 in `extras/` (E5) | readability; campaign stopped | Owner (prompted by the Team reviewer) | `commit:13dd8e8` |
| H23 | every force on (2026-10-20) | negligible forces off by default (B21) | readability; each <= 2.2% of the residual | Owner (prompted by the Team reviewer) | `commit:13dd8e8` |
| H24 | bonds found by a distance test (2026-10-20) | stored bonds (B16) | missed bonds across nm gaps | Engineering | `commit:13dd8e8` |
| H25 | Poppe comparison as two free grains, 4.9x (until 2026-11-10) | flat target, 4.3-4.5x (A6) | the experiment used a target | Engineering | `commit:13dd8e8` |
| H26 | single grains entered sleep at 0.90 (v0.6) | entry = stay at 0.98 with a 20-step look-ahead (D33) | grains near a limit never slept | Engineering | `commit:937a6dd` |

## Open decisions

What the team (or the owner) still has to decide, as of 2026-11-16, what each choice would change, and how to make
the change. "Re-run" lists what must be run after the change.

| ID | open decision | what each choice changes | how to make the change |
|---|---|---|---|
| O1 | **e_dry** of every material (C3); measure it (rebound of non-sticking impacts in flight or on the ground) | thresholds x f(e_dry): 0.70 to 0.85 is a factor 1.52; 0.85 to 0.90 lowers silica 6.18 to ~5.3 mm/s; it also blocks a clean inversion of gamma from the flight result (A7) | field `e_dry` in `dolle_core/materials.py` and `dolle_sim/materials.py`; re-run `tests/test_dolle_core.py` (update the reference numbers), `tests/test_notebooks.py`, `python -m dolle_sim verify`, and the numbers in `docs/OVERVIEW.md` |
| O2 | **which silica and which carbon** fly (G14, G15) | silica -14% (fused silica) to x5.7 (vacuum surface energy); carbon 11-220 mm/s; the carbon/silica ratio 1.8x-36x | add or edit the material in both tables (C1, C2); per tubule `dust.material` in `configs/mission_quarter_deck.json`; re-run as O1 |
| O3 | **release mechanism, gas fill, orientation** (F14, F15) | the nominal shaker frees nothing; kicks free dust at ~1.2-1.5 m/s, 30-250x the sticking speeds; whether grains meet at all depends on how compact the released cloud is (G13: a 2 mm cloud gives 20 silica impacts, a 4 mm cloud 3) | `release`, `orientation`, `gas` per tubule or as a new entry in `scenarios`; `python -m dolle_sim mission configs/mission_quarter_deck.json --scenario NAME`; for the 3-D page `box_mm` and `speed_mm_s` of `dolle_core.scenes.two_tubule_start` |
| O4 | **JKR's accuracy for silica** (B1): Tabor parameter 1.1-1.6 | Maugis-Dugdale would lower silica's hysteresis and threshold by an amount not yet computed; graphite unaffected | a new contact law in `dolle_core/jkr.py`, `dolle_sim/contact.py` and `extras/dolle_extras/dem.py`; re-run everything |
| O5 | **rolling hysteresis h** (C10), uncertain ~50x across models | rolling limit 0.13-1.55 g for silica over h = 0.05-0.6: below the ~10 g spin load for every h in that range, so dust can roll either way; h sets how far it rolls and the clump-restructuring margin (B17) | `h` of `dolle_core.wall.wall_limits`; v0.6 engine: `contact.rolling_hysteresis` in its config (no code edit); re-run `tests/test_dolle_core.py`, `tests/test_core_vs_dem.py`, `tests/test_notebooks.py`, `python -m pytest extras/tests` |
| O6 | **payload roll in the 3-D core** (B22) | moves where grains land; its effect on grain-grain outcomes is within 2 sigma (silica sticks 21 to 27 over 6 seeds x 100 grains in the v0.5 engine) and is not shown to be nil | add Coriolis and centrifugal terms to `dolle_core.sim3d.kick`; re-run `tests/test_dolle_core_sim3d.py` and `python tools/make_sim3d_replays.py` |
| O7 | **clump restructuring** (B17) | clump shapes; possibly silica clumps that roll apart instead of staying rigid | test with the v0.6 contact model (`extras/dolle_extras/dem.py`); if it matters, add restructuring to `dolle_core/impacts.py` |
| O8 | **fragment separation energy** (B15), ASSUMED 1 | 0 instead of 1 changes the 3-D run from 20/4 to 22/6 (silica), 17/17 to 18/18 (graphite) | `contact.fragment_separation`; field `separation` of `dolle_core.sim3d.Tubule` |
| O9 | **at-limit entry energy** (D30) verified only on the checks' own scenes; the stopped campaign's states were never re-examined with it | whether v0.6 clusters sleep correctly in real flight scenes | load saved states and compare with the full model, as `check:cap_sleep_vs_dem` does; `dolle_extras.engine.FullContactSim._island_drift` |
| O10 | **the full-stiffness LAMMPS run of the seconds after a release** (D36) | a direct test of rigid clumps and fragmentation thresholds (B12, B14); 1-4 h per tubule-second | build on `extras/tools/lammps_contact_check.py` |
| O11 | **coating and work of adhesion on the real hardware** (C6, C7, F16): drop tests (Zafar et al. 2014) of the real powders on the real glass, in vacuum if possible | every wall limit scales with w (rolling as w^(4/3)); `rough_grains`-like adhesion (x0.02) would let the 31 g splashdown and the ground vibration test free dust | the glass entries of both material tables, or `wall_adhesion_factor` per tubule; re-run `python -m dolle_sim mission ... --loads-only` and `tests/test_dolle_core.py` |
| O12 | **dust amount** per tubule (F13): 100 in simulations, 800 in the config | impact counts ~N^2; wall coverage ~4% at 800 | `dust.count` or `--grains N`; GRAINS in `extras/tools/run_campaign.py`; `n` of `dolle_core.scenes.two_tubule_start` |
| O13 | **re-run the stored results** after v0.7 (G10): the scenario table dates from 2026-10-26 (v0.5 code, `commit:0731b2f`), the flight demo from v0.6.2 | different numbers in `docs/MISSION_RUNS.md` and `docs/USER_GUIDE.md` (new random stream, stored bonds); conclusions expected, not checked, to hold | `python -m dolle_sim run configs/flight_carbon_vs_silica.json`; `python -m dolle_sim mission configs/mission_quarter_deck.json --scenario NAME` for each scenario; update the documents |
| O14 | **error bars of the threshold fit** (F19): 1-sigma bars cover the truth in 9 of 20 cases | honest uncertainties on the flight's measured threshold | an errors-in-variables fit in `dolle_sim.analysis.fit_threshold`; re-run `python -m dolle_sim verify` (`check:threshold_fit`) |
| O15 | **friction in the default engine and the launch report** (C9, F20): one unsourced 0.3 against the sourced per-pair table elsewhere, and the report's own sliding formula | no effect on the 3-D grain-grain outcomes; the report's sliding capacity at zero load is 0.3 F_c against 0.8 F_c (silica on glass) and 1.0 F_c (graphite), 62.5-70% lower (no grain is flagged either way on clean glass) | two fixes: (1) the per-pair mu: `contact.friction_coefficient` in both shipped configs, or a per-pair lookup in `dolle_sim.simulation.ChamberSim._impulse` and `dolle_sim.mission.launch_report`; (2) the 2 F_c term in `dolle_sim.mission.launch_report` (F20); re-run `tests/test_dolle_core_sim3d.py`, `tests/test_flight.py` |
| O16 | **whether to run the v0.6 scenarios** (G6) | the only way to see wall dust roll during the boost in a full flight | `python extras/tools/run_campaign.py` (days on a laptop) |
| O17 | **whether to rewrite the git history** (G2) | removes the forbidden names from old commit messages and the institutional e-mail address from every commit's author field; changes every commit hash | with the owner's explicit consent: rewrite the messages AND the author e-mail (for example a mailmap with git filter-repo), set a no-reply `user.email` for future commits, force-push; update every `commit:` reference in this file |
| O18 | **a licence** for the public repository (G19) | permissive (MIT, BSD-3-Clause, Apache-2.0) lets anyone reuse the code with attribution; GPL-3.0 keeps derived code open; none (today) grants no reuse rights | add a `LICENSE` file and a `license` entry in `pyproject.toml` |

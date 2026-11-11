# Mission runs: the whole flight in one sweep

Moved here from the README in v0.7 (10 Nov 2026), unchanged except where marked. The scenario table below was
computed with the **v0.5 event-driven engine** (`dolle_sim/simulation.py`: stuck grains form rigid clumps, wall grains
are held in place until they are pulled off), which is again the default engine since v0.7. It was computed with the
v0.6.2 code and the mission defaults of then, which had the negligible forces switched on. Since v0.7 they are off;
for these tubules the only non-zero ones are the gravity gradient (2-3e-8 g, 3e-4 of the 1e-4 g residual acceleration
in the microgravity window), Brownian motion (2-6 µm/s, at most 5e-4 of the sticking speeds) and mutual gravity
(< 1e-12 g), from the pre-flight budget of the mission tubules (`dolle_sim.design.preflight`). A re-run therefore gives slightly different
trajectories, not bit-identical ones; it has not been made. The optional v0.6 full-contact engine (in `extras/`,
see [extras/README.md](../extras/README.md)) also resolves rolling, sliding and twisting along the walls; its
re-runs of these scenarios were stopped before they finished ([extras/docs/CAMPAIGN.md](../extras/docs/CAMPAIGN.md)).
Notebook 05 (`notebooks/05_the_flight_in_brief.ipynb`) explains the flight's loads step by step and checks that its
copy of the table equals this one.

`python -m dolle_sim mission configs/mission_quarter_deck.json` simulates every tubule on the deck through the
whole flight on **one clock**, from lift-off (T+0, dust resting on the bottom of each tubule) through re-entry
and the parachute to 5 s after splashdown (T+887 s). The example config has four 0.5 × 0.5 × 1.5 in glass tubules on a quarter deck (half of
the ~12 in deck, 5.00 in high), with two dusts, two grain sizes and two glass surfaces. Materials, sizes, dust
amounts (800 grains each) and positions are **examples**: replace them with the real ones.

| phase | T+ (s) | what the dust feels | source |
|---|---|---|---|
| first-stage burn | 0–5.2 | 25 g pressing it onto the bottom; spin starts | RockSat User Guide 2026, Table 3-1 (flat 25 g envelope assumed) |
| coast between burns | 5.2–17.4 | ~3 g drag pulling it toward the top | derived from mission fact sheets (estimate) |
| second-stage burn | 17.4–29.1 | 25 g; spin rises to 4.8 Hz (up to ~9.5 g sideways at 10 cm from the axis) | guide (linear spin-up assumed); the guide's ~±10 g lateral ascent load enters the load report only |
| coast | 29.1–54 | drag falling from ~3 g to ~0.01 g; 4.8 Hz spin | derived estimate |
| de-spin | 54–54.9 | yo-yo de-spin to 1 rpm (Euler force) | fact sheets; Ettl & Pfaender (2009) |
| coast to ~100 km | 54.9–88 | ~0.01 → 10⁻⁴ g | derived |
| microgravity (above ~100 km) | 88–313 | 10⁻⁴ g residual along the rocket axis (toward the top until apogee at T+197 s, toward the bottom after), 1 rpm residual spin | 100 km crossings derived from fact sheets; residual assumed |
| re-entry | 313–460 | the payload turns side-on; drag rises to a **7.9 g** peak at T+369 s (~23 km); experiment power-off at T+332 s | point-mass fall through NRLMSIS 2.1 (integrator verified against Allen & Eggers with gravity off); such models read 5–20% low against measured flights, so ~8–11 g; side-on: published flight data of a related payload (PROVENANCE.md) |
| parachute opening | 460–463 | from 94 m/s to the descent speed, peak **5 g** | chute time: guide; 4.6 km: a mission fact sheet of the same vehicle; 5 g: parachute design criteria report M-EH-75-2 (1975) |
| under the parachute | 463–882 | 1 g toward the **nose** end (hanging nose down), 9.4 m/s at sea level | splash time: guide; nose down: a 2016 student flight log on a related rocket |
| splashdown | 882 | ~**31 g** toward the nose end for ~0.05 s | momentum (9.4 m/s); stopping time and direction assumed (that log saturated sideways) |
| afloat | 882–887 | 1 g sideways (floating on its side) | the same flight log |

`--loads-only` prints the **launch-load report** in seconds. For each tubule it gives the pull-off thresholds
of its dust: the JKR pull-off force 1.5πwR divided by the grain's weight, sampled over the size distribution
and the scatter in stickiness. It then compares them with the loads of every phase. For the example tubules
on clean or silanised glass:

| tubule | pull-off threshold: 10% / 50% / 90% of grains below | largest load pulling dust off the bottom: ascent / whole flight | sideways load: spin alone / with the guide's ±10 g lateral |
|---|---|---|---|
| T1 silica 90–106 µm, bare borosilicate | 498 / 1,087 / 2,386 g | 3 g (coast drag) / 31 g (splashdown) | 9.6 / 19.5 g |
| T2 graphite 90–106 µm, bare borosilicate | 784 / 1,712 / 3,757 g | 3 / 31 g | 7.4 / 17.4 g |
| T3 silica 45–53 µm, silanised | 1,153 / 2,517 / 5,524 g | 3 / 31 g | 7.4 / 17.4 g |
| T4 graphite 45–53 µm, silanised | 1,816 / 3,963 / 8,698 g | 3 / 31 g | 9.6 / 19.5 g |

Re-entry (7.9 g) presses sideways, along the bottom face, because the payload falls side-on. The 31 g
splashdown pulls the dust off the bottom only if the payload hangs nose down, as assumed.

**What this means for the design** (with the stated inputs):

1. **Adhesion, not the boost, holds the dust down.** A 100 µm grain sticks to clean glass about a thousand
   times harder than its own weight, so the 25 g boost is irrelevant: nothing comes off at any phase, the
   ~31 g splashdown included. This
   follows directly from the measured work of adhesion (w = 0.045 J/m² for ~90 µm glass beads on glass,
   Sansao et al. 2021).
2. **A gentle shaker frees nothing.** The nominal 50 g / 200 Hz shaker releases 0 of 3,200 grains (full
   simulation, below). Release needs accelerations above the thresholds (10–90% of grains: ~500–8,700 g for
   the examples), i.e. a sharp kick or ultrasonic excitation.
3. **A kick releases most grains at close to the wall's peak speed.** For a sinusoidal pulse that is
   a/(2πf): 5,000 g at 5 kHz gives ~1.5 m/s (the simulation finds a median of 1.2–1.5 m/s; only grains whose
   threshold is close to the peak leave slower). That is about 250 times the predicted sticking speed of
   100 µm silica. To free 1,000 g-bound grains at ~cm/s, the frequency must be ~100 kHz or more
   (ultrasonic), or the grains must be slowed afterwards (gas).
4. **Bare glass walls recapture the grains within seconds** (wall sticking speed ~8 mm/s for silica on clean
   glass), which limits the observing time after each release. This is why the earlier example configs
   assume an anti-stick coating and vibrating walls.
5. **Rolling (modelled only by the optional v0.6 engine in `extras/`).** The sideways spin load (~7–10 g for the
   example layout, up to ~14 g at the deck edge) is above the rolling threshold of most grains (~1.3 g for 100 µm
   silica on clean glass with the nominal rolling hysteresis; uncertain by ~50×, see PROVENANCE.md and notebook 04),
   so the full-contact engine rolls them across the bottom toward the outer wall, where they collide and pile up.
   The default v0.5 engine holds wall grains in place until they are pulled off, so it cannot show this. Treat the
   dust layout at release as uncertain until the rolling resistance of the real powder is measured.

The full simulations (`--scenario NAME`; combine with `+`, e.g. `kick+lying`; results go to
`results/mission/<scenario>/`) follow every grain of all four tubules from lift-off to splashdown. Each cell
gives **grains released · grain–grain impacts in the microgravity window** (single-grain impacts that stuck /
all single-grain impacts, and their median speed). Because the 50 g shaker frees nothing, the gas,
orientation and spin options are compared on top of the kick release:

| scenario | T1 silica 90–106 µm, bare | T2 graphite 90–106 µm, bare | T3 silica 45–53 µm, silanised | T4 graphite 45–53 µm, silanised |
|---|---|---|---|---|
| `nominal`: 50 g / 200 Hz shaker, 10 s at T+90 s (vacuum, standing) | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| `no_release`: no release mechanism | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| `no_drag`: nominal, no drag in the coasts | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| `rough_grains`: nominal, stickiness × 0.02 (rough grains or a coating) | 730 · 7051 (223/6189, 471 mm/s) | 588 · 3534 (117/2712, 449 mm/s) | 398 · 637 (39/368, 452 mm/s) | 179 · 124 (4/28, 421 mm/s) |
| `rough_grains+hang_nose_up`: hanging and floating nose up | the same in the window; after it 48 releases (12 at the parachute opening, 36 at splashdown) | the same; 72 (2, 70) | the same; 37 (3 in re-entry, 1, 33) | the same; 6 (0, 6) |
| `rough_grains+shock_oblique`: opening and splash shocks at 45° to the axis | the same; 1 (at splashdown) | the same; 13 | the same; 16 (3 in re-entry, 13) | the same; 8 |
| `strong_shaker`: 500 g / 1 kHz shaker, 5 s at T+90 s | 88 · 125 (0/39, 1071 mm/s) | 28 · 36 (0/2, 882 mm/s) | 6 · 0 | 0 · 0 |
| `kick`: three 5,000 g / 5 kHz single-cycle kicks (T+90, 150, 210 s) | 795 · 186 (3/66, 32 mm/s) | 772 · 239 (28/88, 60 mm/s) | 683 · 48 (0/20, 45 mm/s) | 502 · 28 (5/6, 43 mm/s) |
| `kick+lying`: kicks, tubules lying flat | 795 · 124 (6/30, 12 mm/s) | 772 · 102 (10/34, 117 mm/s) | 691 · 51 (1/14, 15 mm/s) | 510 · 25 (0/1, 160 mm/s) |
| `kick+spin_1hz`: kicks, 1 Hz residual spin (the guide's upper value) | 795 · 3740 (7/1737, 155 mm/s) | 770 · 1074 (21/551, 301 mm/s) | 683 · 778 (2/376, 224 mm/s) | 498 · 201 (4/94, 345 mm/s) |
| `kick+gas_100pa`: kicks, 100 Pa of air | 795 · 48 (0/4, 1045 mm/s) | 771 · 13 (2/7, 973 mm/s) | 683 · 4 | 499 · 1 |
| `kick+gas_1atm`: kicks, 1 atm of air | 795 · 57 (0/9, 548 mm/s) | 770 · 9 (1/7, 436 mm/s) | 683 · 1; 42 floating at power-off | 498 · 3; 20 floating at power-off |

Reading the table (800 grains per tubule, example materials; run times 3 min to 10 h per scenario on 4 cores):

* With dust on clean or silanised glass, **only the kicks free it** (500–800 grains per tubule). No release,
  the 50 g shaker and the 500 g shaker free 0–88.
* After kicks in vacuum there are 25–240 grain–grain impacts per tubule in the window, at median speeds of
  12–160 mm/s. That is mostly above the predicted sticking speeds (~6 mm/s for 100 µm silica, ~38 mm/s for
  graphite), so few stick: 0–6 of 14–66 single-grain silica impacts.
* **Lying tubules** give slower silica impacts (median 12–15 mm/s vs 32–45 mm/s standing).
* A **1 Hz residual spin** gives many more impacts (200–3,700), but its centrifugal push makes them fast
  (155–345 mm/s).
* **Gas** stops the kicked grains within a fraction of a second: few impacts (1–57, mostly right after a
  kick). At 1 atm, 20–42 of the smaller grains are still floating at power-off (T+332 s); re-entry drag
  (sideways, rising to ~0.02 g by then) has pushed the rest onto the side walls.
* **Weakly bound dust** (`rough_grains`, stickiness × 0.02, thresholds ~10–170 g) is freed by the 50 g shaker
  (178–729 grains), but the shaking walls keep it fast (median ~420–470 mm/s).
* **Re-entry, the parachute and splashdown** free nothing from clean or silanised glass: the largest load,
  31 g at splashdown, is far below the ~500 g thresholds. Weakly bound dust loses a few more grains: 3 in
  re-entry (one tubule) and 1–11 per tubule at splashdown, hanging nose down. The 50 g shaker has already
  freed the weakest grains on the bottom. Hanging nose up, the opening shock and splashdown pull on the top
  face instead, where grains stuck during the window, and free 6–72 per tubule; an oblique shock frees
  1–13. None of this changes the impacts in the window, but it changes where the dust is at recovery.
* None of the options gives many impacts near the sticking speeds. For a threshold measurement the
  release has to be gentler (higher frequency, or a peak just above the thresholds), or the grains slowed
  after release.

**Random vibration** is in the time-resolved run only in the `launch_vibration` scenario, and only with the v0.6
engine (`python -m dolle_extras mission ... --scenario launch_vibration`; the default engine prints a note and leaves
it out): the guide's
test spectra (thrust 0.051 g²/Hz, lateral 0.029 g²/Hz, 20–2000 Hz) synthesised as Gaussian deck accelerations on
all three axes through both burns, an upper bound because no measured flight spectrum is public. The
launch-load report also estimates it for the flight
(lift-off to burnout) and for the ground vibration test (20 s at 10 Grms on the thrust axis, 20–2000 Hz, the
guide's level; Rice's up-crossing rate for a Gaussian, with the thrust not counted as holding the dust down).
For the clean and silanised glass of the examples it is 0.0% (90% of the thresholds are above ~500 g). For weakly bound dust (`rough_grains`) the **ground
test alone** would free ~18–88% of the grains weeks before flight, and launch vibration ~21–90%.

**Measurements that would pin these numbers down:**

* the **work of adhesion** of the real powders on the real tubule glass and coating, in air and after a
  vacuum dry-out, with the drop-test method of Zafar et al. (2014): it needs only a drop tube and a camera;
* the **size distribution** of each powder (`python -m dolle_extras sizes-image ...`, validated on a certified
  glass-bead size standard) or the supplier's laser-diffraction D10/D50/D90 (`python -m dolle_sim sizes --d10 ...`);
* a **release test** on the ground: the fraction released and the release speeds of a candidate kicker,
  filmed at high speed.

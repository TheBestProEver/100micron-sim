# 100micron-sim — DOLL-E dust-collision simulator

**DOLL-E: Dust Observations in Low-gravity and Low-pressure Environments** (UChicago, NASA RockSat).

This simulator models ~100 µm dust grains colliding in microgravity inside a
low-pressure experiment chamber on a sounding rocket. It is built to answer one
mission question: *at what impact velocity do astrophysically relevant grains
(carbon vs. silicates) stick, and can our payload measure it?*

It is a hybrid of an N-body / granular code (in the spirit of REBOUND or
LIGGGHTS/EDEM) and a contact-mechanics solver (in the spirit of ANSYS explicit
dynamics, specialised to adhesive micro-contacts):

* **Flight scale (minutes):** every grain and aggregate is a rigid body moving
  under all external forces. The integrator is velocity Verlet: gas drag and
  Coriolis force are integrated exactly as a damped rotation of the velocity,
  and position-dependent forces are re-evaluated every step. Aggregates rotate
  with the energy- and angular-momentum-conserving Dullweber–Leimkuhler–McLachlan
  splitting.
* **Impact scale (~1 µs):** contacts are found by continuous collision
  detection. Impacts are processed in time order, and the bodies an impact
  changes are re-predicted before the next one, so follow-on collisions are
  not skipped. Each contact is resolved by integrating the JKR + plasticity
  contact law with nanosecond steps. Stick, bounce, restitution, contact
  charging and fragmentation all come out of that integration; nothing comes
  from a lookup table. The bounce impulse uses the full 3×3 collision matrix of
  the two rigid bodies, inside a Coulomb friction cone.

---

## Quick start

```bash
pip install -r requirements.txt
python -m dolle_sim preflight configs/rocksat_x_carbon_vs_silica.json   # seconds: design checks + force budget
python -m dolle_sim run configs/rocksat_x_carbon_vs_silica.json         # full simulation (~20-25 min on a laptop)
python -m dolle_sim analyze results/rocksat_x_carbon_vs_silica          # re-run the camera analysis on saved results
python -m dolle_sim sweep                                               # sticking probability vs speed curves
python -m dolle_sim invert --material graphite --vstick 0.08            # measured v_stick -> effective surface energy
python -m dolle_sim rebound-check configs/rocksat_x_carbon_vs_silica.json --duration 10   # cross-check vs REBOUND
python -m dolle_sim materials                                           # material database
python -m pytest -q                                                     # physics validation tests
```

`run` writes to `results/<name>/`:

| file | contents |
|---|---|
| `viewer.html` | interactive 3D replay (open in any browser, works offline): rotate/zoom, colour by material / aggregate size / **spin rate**, collision flashes, **hide grains stuck to the walls**, live impact-speed histogram |
| `sticking_measurement.png` | stick fraction vs impact speed, true 3D vs **what the camera would measure**, logistic fits with 1σ bootstrap errors |
| `collision_velocities.png`, `timeseries.png`, `aggregate_sizes.png` | impact-speed histograms, cloud evolution, aggregate growth |
| `<chamber>/collisions.csv` | every impact: time, grains, materials, normal/tangential/relative speed, effective mass, outcome, restitution, plastic yield flag, charges, spins, contact point, energy, fragments |
| `<chamber>/camera_measurements.csv` | per impact: in field of view / in focus / trackable, true and camera-measured speeds |
| `<chamber>/timeseries.csv`, `trajectory.npz`, `summary.json` | time series, full trajectories, all statistics + pre-flight report |

---

## Physics included

### Environment (chamber frame, rotating with the payload)
| effect | model |
|---|---|
| Residual microgravity | constant vector (default 1e-4 g, RockSat-X preset) or a **flight IMU CSV** (`accel_csv`: t, ax, ay, az in g) |
| g-jitter / vibration | Ornstein-Uhlenbeck coloured noise (rms, correlation time) |
| Payload spin (default **0.0167 Hz = 1 rpm**) | centrifugal + exact Coriolis, any spin axis, chamber offset from the axis |
| Gravity gradient (tidal) | (GM/r³)[3(x·ê_up)ê_up − x] about the rocket CG, at flight altitude; ê_up (inertially fixed) rotates in the spinning chamber frame |
| Mutual gravity | Newtonian, grain–grain (negligible for 100 µm grains, included for completeness) |
| Gas drag | Stokes + Cunningham slip correction (exact from continuum to free-molecular/Epstein); integrated exactly |
| Gas torque on spinning grains | free-molecular (2π/3)ρ v̄ R⁴ω ↔ continuum 8πμR³ω interpolation |
| Brownian motion | Ornstein-Uhlenbeck kicks consistent with the drag (translation + rotation) |
| Thermophoresis | Waldmann free-molecular law with chamber temperature-jump correction |
| Photophoresis | free-molecular law for a conduction-limited sphere, F = (π/3) α J₁ p I R³/(k_p T), J₁ = absorptivity/2 |
| Radiation pressure | from the camera illumination, Q_pr ≈ 1 (black) to 0.3 (transparent) |

### Electromagnetic
| effect | model |
|---|---|
| Coulomb | Ewald split: exact short-range pair sum (r < 3 mm) + **long-range space-charge field** from a particle-mesh Poisson solve with grounded walls (sine transform, P3M-deconvolved; ≤ 3% force error). Contact charging leaves the whole cloud with one sign, and nothing screens it at 1 Pa, so the long-range part matters |
| Polarisation | charge-induced dipole attraction between charged and neutral grains (Clausius-Mossotti) |
| Image charges | short range: dielectric walls, factor (ε−1)/(ε+1); long range: included in the grounded-wall mesh |
| External E-field | uniform, configurable (e.g. biased electrodes) |
| Lorentz force | q v×B, Earth's field by default (rotating in the spinning chamber frame) |
| Contact (tribo-)charging | condenser model driven by work-function difference, **saturating** at grain potential ≈ Δφ |

### Contacts and collisions
| effect | model |
|---|---|
| Adhesion | JKR with snap-on / pull-off hysteresis (Johnson-Kendall-Roberts 1971; Chokshi, Tielens & Hollenbach 1993) |
| Viscoelastic loss | Tsuji damping tuned to each material's dry restitution |
| Plastic yield | Thornton & Ning (1998) limiting-pressure loading; unloading on the flattened contact (radius R_p, overlap δ−δ_p), so the pull-off force rises to 1.5πwR_p. Restitution matches Thornton–Ning × e_dry (tested) |
| Friction / spin exchange | Coulomb friction cone at the contact point with the full collision matrix, which couples grain spin to translation |
| Grain-to-grain variability | log-normal scatter of effective surface energy per grain (roughness) and of radius |
| Aggregation | stuck grains become rigid aggregates conserving momentum and angular momentum (they spin) |
| Fragmentation | Dominik & Tielens (1997) thresholds on the aggregate's own grain–grain bond energy: erosion > 3 N_c E_break, shattering > 10 N_c E_break. Fragments separate radially with the elastically returned excess energy (momentum and angular momentum conserved) |
| Walls | JKR contact with the wall material × `wall_adhesion_factor` (coatings, roughness), optional **vibrating walls** that re-energise the dust. The impact phase is drawn from the flux-weighted distribution, so wall motion can never let a grain through. Grains stuck to a wall move with it, and impacts onto them are flagged and kept out of the grain–grain statistics |

### Measurement
A synthetic camera (frame rate, field of view, depth of field, centroid noise,
required free-flight frames) decides which impacts the payload would actually
record. It measures impact speeds with realistic error and fits the sticking
threshold with bootstrap error bars. Three analysis strategies are compared:

| strategy | what is measured | mission run, graphite / silica (truth 38.9 / 6.21 mm/s) |
|---|---|---|
| 1 camera, 2D relative speed | \|projected centroid v_rel\| | 10.2 / 1.5 mm/s: **wrong by 4×** |
| 1 camera, 2D normal component | v_rel projected on the imaged line of centres | 6.2 / 0.5 mm/s |
| **2 orthogonal cameras (stereo)** | 3D normal impact speed | **38.1 [37.7–38.6] / 5.71 [5.51–5.88] mm/s** |
| stereo, 1 µm centroiding | 3D normal impact speed | 38.9 / 6.17 mm/s |

Sticking is set by the *normal* impact speed. With one camera, oblique impacts
stick at high relative speed and the line-of-sight velocity is invisible. The
inferred threshold is then smeared and biased, by an amount that depends on
the launch geometry (in the colliding-clouds run it is 27 / 5.3 mm/s instead).
**Design recommendation: two orthogonal cameras (or one camera with a mirror
giving a second view), analysed on free monomer–monomer impacts using the
line of centres at contact.** The remaining 2–8% low bias of stereo comes from
centroid noise (5 µm here) and disappears at 1 µm. A faster camera with the
same number of tracked frames makes it *worse*: velocity noise grows as
fps / k^1.5, so what matters is centroid precision and the tracking baseline.
`python -m dolle_sim analyze <results>` re-runs this analysis (optionally with
`--camera '{"fps": 1000}'` overrides) without re-simulating.

---

## Keeping grains off the walls

The goal is grain–grain impacts, not wall sticking. In the model, walls stick
whenever a grain hits slower than the wall sticking velocity. Bare glass at
1–2 cm/s captures most grains within a couple of chamber crossings. The
mission configs keep grains bouncing *physically*:

1. **Low-adhesion walls**: a fluorinated (FDTS-type) coating on hard glass plus
   bead-blasting (`wall_material: glass_fluorosilane`,
   `wall_adhesion_factor: 0.05`). This drops wall sticking to ~0.4–1.2 mm/s.
   Soft PTFE is *worse*: it has low γ but a huge contact area, and it is a strong
   tribo-charger.
2. **Vibrating walls**: 200 Hz with 5–25 µm amplitude, so the wall speed is
   comparable to the target impact speed. This replaces the energy lost in
   inelastic bounces and keeps the grains at the speeds you want to probe.
3. `preflight` reports the predicted fraction of wall hits that stick. If you
   set `wall_sticking: "never"` (idealised walls), it warns that the hardware
   must actually achieve it.

The viewer's **"Hide grains stuck to walls"** checkbox removes the wall-bound
grains from the display.

---

## Results from the example configs

All numbers are for 100 µm grains with literature material parameters. They
predict what the flight would see *if* those parameters are right.

**Sticking thresholds (model).** Monomer–monomer, 50% sticking speed
(`python -m dolle_sim sweep`): graphite **39.2 mm/s**, amorphous carbon
**24.5 mm/s**, graphite–silica **17.7 mm/s**, silica **6.2 mm/s**. Carbon
grains are predicted to stick at 4–6× higher speeds than silica, mostly
because of their higher surface energy and lower stiffness.

![sticking sweep](docs/figures/sticking_sweep.png)

**Full mission run** (`configs/rocksat_x_carbon_vs_silica.json`: 2 × 2000
grains, 120 s, 1 Pa, 1e-4 g, 1 rpm roll, all forces on, anti-stick +
vibrating walls):

| | graphite chamber | silica chamber |
|---|---|---|
| grain–grain impacts | 20 010 (5 411 stuck) | 4 806 (783 stuck) |
| wall impacts / stuck | 497 088 / 109 (0.02%) | 133 128 / 116 (0.09%) |
| grains still free after 120 s | 89% | 94% |
| net cloud charge (contact charging) | +4.9×10⁷ e (≈ 2.4×10⁴ e per grain) | +2.1×10⁷ e |
| true threshold (3D normal speed, free monomers) | 38.9 [38.8–39.1] mm/s | 6.21 [6.17–6.26] mm/s |
| **stereo-camera measurement** | **38.1 [37.7–38.6] mm/s** | **5.71 [5.51–5.88] mm/s** |
| single camera, 2D relative speed | 10.2 mm/s | 1.5 mm/s |
| impacts onto wall-bound grains (separate class) | 469, threshold 30.0 mm/s (≈ 38.9/√2, as expected) | 91, threshold 4.4 mm/s |

![measurement](docs/figures/sticking_measurement_isotropic.png)

![time series](docs/figures/timeseries_isotropic.png)

**Colliding clouds** (`configs/colliding_clouds_carbon_vs_silica.json`, 60 s):
the true thresholds are 39.2 / 6.16 mm/s and stereo measures
38.4 [37.9–38.9] / 5.88 [5.68–6.08] mm/s. A single camera gives 27 / 5.3 mm/s
here, against 10 / 1.5 mm/s in the isotropic run. Its bias depends on the
geometry and cannot be calibrated away.

![clouds](docs/figures/clouds_snapshot.png)

**Design findings the simulator surfaced:**

1. **Use two orthogonal camera views and ~1 µm centroiding.** With one camera
   the inferred sticking speed is wrong by 1.2–6×, depending on the impact
   geometry. Stereo with 5 µm centroids reads 2–8% low; with 1 µm it matches.
2. **Anti-stick, vibrating walls work.** Fluorinated, roughened glass vibrated
   at 200 Hz (5–25 µm) catches only 0.02–0.09% of wall hits, about 90% of the
   grains stay free for the whole 120 s, and grain–grain impacts accumulate
   steadily (~170 per second for graphite).
3. **Thermophoresis**: at 1 Pa, each kelvin across a 3 cm chamber drives
   ~3.6e-4 g, 3.6× the residual microgravity. Keep the chamber isothermal to
   ≲ 0.1 K.
4. **Residual 1e-4 g sweeps a resting grain across 3 cm in ~8 s**, and the
   1 rpm roll adds Coriolis accelerations of ~7.5e-4 g at 3.5 cm/s. Both are in
   the simulation, so their effect on the measured impact speeds is included.
5. **Contact charging leaves the cloud one sign** (+2×10⁴ e per grain). The
   long-range space-charge field is therefore included; charge-induced
   polarisation dominates at contact. Lorentz force and mutual gravity are
   negligible (< 1e-12 g).

---

## Validation

`python -m pytest -q` runs 20 tests:

| test | checks |
|---|---|
| contact: elastic limit, damping, plasticity | energy conservation (e = 1.000), e_dry calibration, Thornton–Ning restitution (no damping and with e_dry = 0.7), γ = 0 limit |
| sticking thresholds | DEM threshold = JKR hysteresis estimate without damping; carbon > silica |
| free flight | drag + residual g exact to < 10 nm over 1 s; spinning frame (drag + Coriolis + centrifugal) vs a 10⁻¹² ODE solution: < 1 nm at 1 rpm, 2nd-order convergence at 30× the spin rate |
| rigid-body rotation | spinning trimer for 10 s: angular momentum to 10⁻¹⁰, energy bounded (no drift) |
| collisions | exact normal restitution through the 3×3 impulse matrix for random oblique aggregate impacts, friction cone respected; no interpenetration over 1 s of a dense 400-grain gas |
| walls | 200 grains, 2 s against 25 µm / 200 Hz vibrating walls: no grain ever outside the chamber |
| conservation | momentum and angular momentum through merging and splitting |
| electrostatics | space-charge solver vs Coulomb's law (≤ 6% at 1–8 mm, ~1% beyond); charge saturation |
| bookkeeping | impacts onto wall-bound grains are flagged |

`python -m dolle_sim rebound-check` runs the same chamber in
[REBOUND](https://rebound.readthedocs.io), an independent, widely used N-body
code with its own leapfrog/IAS15 integrators, gravity and "line" collision
detection. Our JKR model is plugged in as REBOUND's collision resolver,
rewound to the true contact time. Drag + Coriolis are applied there as an
exact operator in REBOUND's pre/post-timestep hooks. Ballistic trajectories
agree with IAS15 to < 1 nm. Collision statistics agree within Poisson noise
(see the figure and `results/rebound_check/*.json`).

![rebound check](docs/figures/rebound_check_carbon.png)

### Independent review (v0.2)

v0.1 was put through an adversarial review: five reviewers, one per
subsystem, and every finding checked by up to three independent skeptics.
Confirmed problems that are fixed in v0.2:

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

---

## Configuration

See `configs/rocksat_x_carbon_vs_silica.json` (two chambers: graphite vs
silica, isotropic launch) and `configs/colliding_clouds_carbon_vs_silica.json`
(two opposed dust clouds per chamber). Key blocks:

* `environment`: `preset` (`rocksat_x`, `rocksat_c`, `ideal`, `drop_tower`), `pressure_pa`, `gas`,
  `residual_accel_g`, `jitter_rms_g`, `spin_rate_hz`, `spin_axis`, `accel_csv`, `gravity_gradient`,
  `altitude_m`, `electric_field_v_m`, `magnetic_field_t`, `temperature_gradient_k_m`, `illumination`
* `forces`: switch any force on/off (`coulomb`, `space_charge`, `polarization`, `image_charges`, `self_gravity`,
  `lorentz`, `external_e_field`, `thermophoresis`, `photophoresis`, `radiation_pressure`, `brownian`);
  `long_range_cutoff_m` (Ewald split radius), `long_range_update_every` (pair forces, default every step),
  `space_charge_update_every` (mesh solve, default every 25 steps), `photophoresis_accommodation`
* `contact`: `friction_coefficient`, `wall_sticking` (`jkr`/`never`/`always`), `contact_charging`, `fragmentation`
* `camera`: `view_axis`, `fps`, `fov_m`, `depth_of_field_m`, `position_noise_m`, `track_frames`
* `materials`: override any property of any material, or define new ones
* `chambers[]`: `size_m`, `wall_material`, `wall_adhesion_factor`, `wall_vibration`, offsets from the spin
  axis/CG, and `particles[]` groups (`material`, `count`, `radius_m`, `radius_rel_sigma`,
  `surface_energy_log_sigma`, `charge_e_mean/sigma`, `spin`, `region_m`, `velocity` isotropic or jet)

## Materials

`silica`, `forsterite`, `alumina`, `graphite`, `amorphous_carbon`, `carbon_silica_50_50`, `water_ice`, and
the wall materials `glass_bk7`, `glass_fluorosilane`, `sapphire`, `aluminum_6061`, `ptfe`. Each has density,
modulus, Poisson ratio, surface energy, restitution, work function, permittivity, yield pressure, thermal
conductivity and absorptivity, with literature notes (`python -m dolle_sim materials`).

---

## How realistic is it?

It includes every effect we could identify that matters for 100 µm grains at
mm/s–cm/s speeds in a 0.1–10 Pa chamber, plus several that turn out
negligible (mutual gravity, Lorentz force, Brownian motion). The pre-flight
force budget shows which effects matter for *your* configuration.

What no simulation can supply is the **material parameters of your actual
powder**: surface energy, yield pressure and dry restitution. Real grains are
rough, irregular and possibly contaminated, so literature γ values can be off
by a factor of several. That is exactly why the experiment is worth flying.
A concrete example: for ~1.2 µm silica spheres this model (like all JKR
theory) predicts v_stick ≈ 0.25 m/s, but Poppe, Blum & Henning (2000) measured
≈ 1.2 m/s. Theory with literature parameters can be off by several times, so
the predicted numbers below are design guidance, not the answer.

The intended workflow:

1. Design with the literature values (this repo).
2. Calibrate γ and e_dry with ground tests (drop tower, or low-speed tabletop
   impacts on a substrate). `invert` turns a measured sticking velocity into
   γ_eff.
3. After flight, fit the measured sticking curve. Compare carbon vs silicate γ_eff
   and their trend with work function / electron affinity.

Remaining approximations: grains are spheres (irregular shapes would
need multi-sphere grains, which `region`/aggregate machinery could support).
Aggregates are rigid between impacts (no rolling restructuring). Photophoresis
ignores shadowing (the cloud is optically thin). Gas flow is not simulated
(free-molecular regime, Kn ≈ 100).

## Key references

Johnson, Kendall & Roberts (1971) Proc. R. Soc. A 324, 301 · Chokshi, Tielens & Hollenbach (1993) ApJ 407, 806 ·
Dominik & Tielens (1997) ApJ 480, 647 · Thornton & Ning (1998) Powder Technol. 99, 154 ·
Tsuji, Tanaka & Ishida (1992) Powder Technol. 71, 239 · Poppe, Blum & Henning (2000) ApJ 533, 454 ·
Blum & Wurm (2000) Icarus 143, 138 · Blum & Wurm (2008) ARA&A 46, 21 · Rohatschek (1995) J. Aerosol Sci. 26, 717 ·
Waldmann (1959) Z. Naturforsch. 14a, 589 · Lacks & Sankaran (2011) J. Phys. D 44, 453001 ·
Steinpilz et al. (2020) Nature Physics 16, 225 · Rein & Liu (2012) A&A 537, A128 (REBOUND).

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
  under all external forces, integrated with a 2nd-order kick-drift-kick
  scheme with exact gas-drag integration.
* **Impact scale (~1 µs):** every contact is detected exactly (continuous
  collision detection) and resolved by integrating the JKR + plasticity
  contact law with nanosecond steps. Stick, bounce, restitution, contact
  charging and fragmentation all come out of that integration. Nothing is
  assumed from a lookup table.

---

## Quick start

```bash
pip install -r requirements.txt
python -m dolle_sim preflight configs/rocksat_x_carbon_vs_silica.json   # seconds: design checks + force budget
python -m dolle_sim run configs/rocksat_x_carbon_vs_silica.json         # full simulation (~10-20 min)
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
| Payload spin (default **0.0167 Hz = 1 rpm**) | centrifugal + Coriolis, any spin axis, chamber offset from the axis |
| Gravity gradient (tidal) | (GM/r³)[3(x·ê_up)ê_up − x] about the rocket CG, at flight altitude |
| Mutual gravity | Newtonian, grain–grain (negligible for 100 µm grains, included for completeness) |
| Gas drag | Stokes + Cunningham slip correction (exact from continuum to free-molecular/Epstein); integrated exactly |
| Gas torque on spinning grains | free-molecular (2π/3)ρ v̄ R⁴ω ↔ continuum 8πμR³ω interpolation |
| Brownian motion | Ornstein-Uhlenbeck kicks consistent with the drag (translation + rotation) |
| Thermophoresis | Waldmann free-molecular law with chamber temperature-jump correction |
| Photophoresis | free-molecular (Rohatschek) law, uses grain absorptivity and thermal conductivity |
| Radiation pressure | from the camera illumination |

### Electromagnetic
| effect | model |
|---|---|
| Coulomb | pairwise, within a cutoff (default 3 mm) |
| Polarisation | charge-induced dipole attraction between charged and neutral grains (Clausius-Mossotti) |
| Image charges | attraction to dielectric/conducting walls, factor (ε−1)/(ε+1) |
| External E-field | uniform, configurable (e.g. biased electrodes) |
| Lorentz force | q v×B, Earth's field by default |
| Contact (tribo-)charging | condenser model driven by work-function difference, **saturating** at grain potential ≈ Δφ |

### Contacts and collisions
| effect | model |
|---|---|
| Adhesion | JKR with snap-on / pull-off hysteresis (Johnson-Kendall-Roberts 1971; Chokshi, Tielens & Hollenbach 1993) |
| Viscoelastic loss | Tsuji damping tuned to each material's dry restitution |
| Plastic yield | Thornton & Ning (1998) elastic-plastic loading/unloading combined with JKR adhesion (verified against the analytic restitution in the tests) |
| Friction / spin exchange | Coulomb-limited tangential impulse at the contact point, which couples grain spin to translation |
| Grain-to-grain variability | log-normal scatter of effective surface energy (roughness) and of radius |
| Aggregation | stuck grains become rigid aggregates conserving momentum and angular momentum (they spin) |
| Fragmentation | Dominik & Tielens (1997) energy thresholds: erosion > 3 E_break, shattering > 10 N_c E_break |
| Walls | JKR contact with the wall material × `wall_adhesion_factor` (coatings, roughness), optional **vibrating walls** that re-energise the dust so grains keep bouncing instead of settling |

### Measurement
A synthetic camera (frame rate, field of view, depth of field, centroid noise,
required free-flight frames) decides which impacts the payload would actually
record, measures their 2D relative speed with realistic error, and fits the
sticking threshold with bootstrap error bars. Use it to size cameras, lighting
and grain numbers *before* building hardware.

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

## Validation

* `python -m pytest -q`: elastic energy conservation, restitution calibration,
  Thornton–Ning plasticity (matches analytic to <0.5%), sticking-threshold
  limits, exact free flight with drag (<10 nm error over 1 s), momentum and
  angular momentum conservation through aggregation/fragmentation, charge
  saturation.
* `python -m dolle_sim rebound-check`: runs the same chamber in
  [REBOUND](https://rebound.readthedocs.io), an independent, widely used N-body
  integrator (leapfrog/IAS15, its own gravity and "line" collision
  detection). Our JKR model is plugged in as REBOUND's collision resolver.
  It compares ballistic trajectories (agree to ~0.05 µm over 0.2 s) and the
  collision statistics.

---

## Configuration

See `configs/rocksat_x_carbon_vs_silica.json` (two chambers: graphite vs
silica, isotropic launch) and `configs/colliding_clouds_carbon_vs_silica.json`
(two opposed dust clouds per chamber). Key blocks:

* `environment`: `preset` (`rocksat_x`, `rocksat_c`, `ideal`, `drop_tower`), `pressure_pa`, `gas`,
  `residual_accel_g`, `jitter_rms_g`, `spin_rate_hz`, `spin_axis`, `accel_csv`, `gravity_gradient`,
  `altitude_m`, `electric_field_v_m`, `magnetic_field_t`, `temperature_gradient_k_m`, `illumination`
* `forces`: switch any force on/off; `long_range_cutoff_m`, `long_range_update_every`
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

# extras/: the optional parts of the DOLL-E simulator

Nothing in this folder is needed to run the simulator, to follow the notebooks or to reproduce the main results. It
holds the parts that were moved out of the main package in v0.7 (10 Nov 2026), so that the code a reader meets first
stays small: [dolle_core/](../dolle_core/__init__.py) (the readable contact physics) and
[dolle_sim/](../dolle_sim/__init__.py) (the reference simulator, v0.5 engine). Everything here still runs and is
tested.

| what | files | lines | why it is optional |
|---|---|---|---|
| the v0.6 full-contact flight engine | `dolle_extras/engine.py`, `dem.py`, `statics.py`, `slaved.py`, `roll1.py`, `loads.py`, `vibration.py`, `verify_full.py` (its 19 checks) | 7,815 | A second, more detailed model of the flight: every grain is its own body and every contact has sliding, rolling and twisting springs (Wada et al. 2007; Dominik & Tielens 1995, 1997), identical to the LAMMPS granular contact law term by term. The default engine (`dolle_sim/simulation.py`) uses the same 1-D contact law for every impact but treats stuck grains as rigid clumps and holds wall grains until they are pulled off. The v0.6 engine matters for grains rolling along the walls during the spin ([docs/MISSION_RUNS.md](../docs/MISSION_RUNS.md), design point 5); it needs numba to be fast. |
| one tubule with the v0.6 engine | `dolle_extras/full_mission.py` | 69 | Checkpoints, scenario branches and autosave, used only by long v0.6 runs ([docs/CAMPAIGN.md](docs/CAMPAIGN.md)). |
| cross-checks against other codes | `dolle_extras/rebound_check.py`, `lammps_check.py`, `tools/lammps_contact_check.py`, `tools/lammps_benchmark.py` | 893 | Outside evidence for the contact law and the integrator (CHECKS.md, PROVENANCE.md); they need REBOUND or LAMMPS installed. Their results are stored in CHECKS.md and on the website. |
| grain sizes from microscope images | `dolle_extras/sizing.py` | 122 | Needs scikit-image; the main package fits size distributions from D10/D50/D90 (`python -m dolle_sim sizes`). Not yet tested on real micrographs. |
| campaign tools | `tools/run_campaign.py`, `tools/mission_table.py` | 366 | Schedule every mission scenario on the v0.6 engine and tabulate the results ([docs/CAMPAIGN.md](docs/CAMPAIGN.md)). |
| atmosphere table generator | `tools/make_atmosphere.py` | 78 | Regenerates `dolle_sim/atmosphere.py` from NRLMSIS 2.1 (needs the pymsis package); the table itself is in the main package. |
| documents | `docs/ENGINE_LOGIC.md`, `docs/design/at-limit-sleep/`, `docs/CAMPAIGN.md`, `PROVENANCE.md` | | How the v0.6 engine decides, the design record of its newest part, the campaign, and the provenance of these modules. |

## How to run it

After `pip install -e .` from the repository folder (or `pip install -e ".[extras]"`, which also installs numba and
psutil), `dolle_extras` is importable everywhere:

```bash
python -m dolle_extras verify                    # the 19 checks of the v0.6 engine (~5 min; CHECKS.md, Part 2)
python -m dolle_extras mission configs/mission_quarter_deck.json --loads-only     # same options as dolle_sim mission,
python -m dolle_extras mission configs/mission_quarter_deck.json --grains 100      # with the v0.6 engine, plus
                                                 # --checkpoint-at, --resume-from/--resume-at, --autosave-min
python -m dolle_extras rebound-check configs/flight_carbon_vs_silica.json --duration 60 --max-grains 600   # pip install rebound
python -m dolle_extras lammps-check              # pip install lammps; also runs tools/lammps_contact_check.py
python -m dolle_extras sizes-image photo.png --um-per-px 1.25                    # pip install scikit-image
python -m dolle_extras sources --write           # extras/PROVENANCE.md
python -m pytest extras/tests -q                 # the tests of everything here (~13 min)
```

Without installing, from a clone: write `python extras/dolle_extras` instead of `python -m dolle_extras` (from the
repository folder), or `cd extras` first. `python -m pytest extras/tests` needs no install (`extras/tests/conftest.py` puts
the repository on the path). `python -m dolle_extras mission` is the only way to run the v0.6 engine: `python -m
dolle_sim mission` runs the v0.5 engine and, for a config with `"engine": "full"`, stops with a message naming the
`dolle_extras` command. The commands that moved here (`rebound-check`, `lammps-check`, `sizes --image`, and the
mission options above) print the same pointer when given to `python -m dolle_sim`.

## The dependency rule

`dolle_extras` may import `dolle_sim` and `dolle_core`; `dolle_sim` never imports `dolle_extras`, not even lazily:
`python -m dolle_extras mission` hands its runner (`full_mission.run_args`) to `dolle_sim.mission.run_mission` and
`run_tubule`. `dolle_core`'s physics modules never import either simulator. Only two notebook helpers,
`dolle_core/nb_collisions.py dem_threshold` and `nb_flight.py dem_rolling_ramp`, import `dolle_extras` (inside the
function, for the notebooks' optional 3-D comparisons; without this folder the notebooks skip those cells), and the
main suite's comparisons with the 3-D contact model (`tests/test_core_vs_dem.py`) are skipped when this folder is
missing. `tests/test_provenance.py` checks the rule and runs the main package with every module of this folder blocked.

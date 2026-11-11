"""dolle_extras: the optional parts of the DOLL-E simulator (not needed to run or understand it).

What is here, and why it is optional (extras/README.md has the details):

  v0.6 full-contact flight engine   every grain its own body, every contact with sliding, rolling and twisting
    engine.py, dem.py, statics.py,  (about 7,800 lines, numba recommended). The default engine of dolle_sim (v0.5,
    slaved.py, roll1.py, loads.py,  simulation.py) treats stuck grains as rigid clumps and resolves every impact with
    vibration.py, verify_full.py    the same contact law (dolle_sim/contact.py); the v0.6 engine is kept as an
                                    independent, more detailed model of grains resting on the walls.
  cross-checks against other codes  rebound_check.py (REBOUND) and lammps_check.py (LAMMPS): outside evidence for the
                                    contact law and the integrator; they need those packages.
  image-based grain sizing          sizing.py (scikit-image): grain diameters from microscope images.
  full_mission.py                   runs one tubule with the v0.6 engine (checkpoints, branches, autosave); its run_args
                                    is the runner python -m dolle_extras mission hands to dolle_sim.mission.
  provenance.py                     the provenance table of these modules (extras/PROVENANCE.md).

Run:  python -m dolle_extras verify | mission | rebound-check | lammps-check | sizes-image | sources

Dependency rule (tests/test_provenance.py checks it): dolle_extras may import dolle_sim and dolle_core; dolle_sim never
imports dolle_extras. (dolle_core's notebook helpers nb_collisions.dem_threshold and nb_flight.dem_rolling_ramp import
it inside the function, for the notebooks' optional 3-D comparisons.)
"""
import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

# From a clone without `pip install -e .` (for example after `cd extras`), dolle_sim sits two folders up.
if _ilu.find_spec("dolle_sim") is None:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

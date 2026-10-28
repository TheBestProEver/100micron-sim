"""DOLL-E: Dust Observations in Low-gravity and Low-pressure Environments -- simulator.

Provenance: All physics in this package was written for DOLL-E. REBOUND is
imported only by rebound_check.py and LAMMPS only by lammps_check.py, both
optional cross-checks; the simulator never uses them. The source of every component is listed in
dolle_sim/provenance.py (also PROVENANCE.md and `python -m dolle_sim
sources`).
"""
__version__ = "0.6.0"

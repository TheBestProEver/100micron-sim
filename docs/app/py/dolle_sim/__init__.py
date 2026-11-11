"""DOLL-E: Dust Observations in Low-gravity and Low-pressure Environments -- simulator.

The reference simulator: materials, the 1-D contact law (contact.py), the event-driven engine in which stuck grains
form rigid clumps (simulation.py), the flight (flight.py, mission.py), the synthetic camera (analysis.py) and the
model checks (verification.py). The readable core of the contact physics is the package dolle_core; the optional
parts (the v0.6 full-contact engine, the REBOUND and LAMMPS cross-checks, grain sizes from images) are in extras/.

Provenance: All physics in this package was written for DOLL-E. REBOUND and LAMMPS are used only by the optional
cross-checks in extras/; the simulator never uses them. The source of every component is listed in
dolle_sim/provenance.py (also PROVENANCE.md and `python -m dolle_sim sources`).
"""
__version__ = "0.7.0"

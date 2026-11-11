# DOLL-E notebooks: the physics of the simulator, step by step

These notebooks explain the core of the DOLL-E dust simulator: why slow grains stick and fast ones bounce, what
changes between silica and carbon, and what holds dust on the tubule walls during the flight. Each derivation starts
from a cited paper and shows the algebra. Each "why this choice" box gives the alternatives and the evidence.

## Reading order

| notebook | what it answers |
|---|---|
| `00_start_here.ipynb` | What DOLL-E measures, why it needs microgravity, and how the full simulator `dolle_sim` and the readable core `dolle_core` are built |
| `01_two_grains_colliding.ipynb` | A head-on impact step by step: the JKR contact law, the adhesion hysteresis, damping, plastic yield, the sticking threshold, the time step |
| `02_momentum_conservation.ipynb` | Why the total momentum is conserved exactly (Newton's third law, and the line of code that does it) and where the kinetic energy goes |
| `03_silica_vs_carbon.ipynb` | Why graphite sticks at about 6 times silica's speed, property by property; the scaling law; other carbons and how uncertain the inputs are |
| `04_grain_on_a_wall.ipynb` | What holds a grain on the wall (pull-off, sliding, rolling, twisting) and which flight loads move it |
| `05_the_flight_in_brief.ipynb` | The flight from lift-off to splashdown as loads on the dust, and what can be measured in the microgravity window |

## How to open them

- **Locally:** from the repository folder run `jupyter lab`, then open `notebooks/`. The first code cell of each
  notebook puts the repository folder on the Python path. It works whether Jupyter starts in the repository folder
  or in `notebooks/`.
- **In the browser:** the "Open in Colab" badge at the top of each notebook opens it in Google Colab, and its first
  code cell installs the repository with pip. The badges point to the `main` branch, so they work only once
  `dolle_core/` and `notebooks/` are on `main`.
- **Without Jupyter:** the scripts in `examples/` print the key numbers, for example
  `python examples/sticking_threshold.py`.

A few cells run the full simulator's 3-D contact model, which needs numba: notebook 00 section 5, notebook 01
section 12, notebook 02 section 7 and notebook 04 section 4. Without numba those cells print a message instead. The
first such cell in a session compiles the 3-D engine, which takes about a minute.

## Can I trust what I read?

`dolle_core` performs the same calculations as `dolle_sim`, in the same order. `tests/test_dolle_core.py` checks
this case by case against `dolle_sim`:

- every material value;
- the pair properties;
- the sticking speeds, which come out identical;
- the restitutions, equal to rounding error, including dented (plastic) impacts;
- the simulator's own contact checks;
- momentum and energy bookkeeping;
- the wall limits.

`tests/test_notebooks.py` runs every notebook from start to finish.

Run both with `python -m pytest tests/test_dolle_core.py tests/test_notebooks.py -q`.

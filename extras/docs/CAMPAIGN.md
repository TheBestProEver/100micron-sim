# The v0.6 mission campaign: every scenario with the full-contact engine

This is optional. The mission results in [docs/MISSION_RUNS.md](../../docs/MISSION_RUNS.md) come from the default
v0.5 engine. This page describes how the optional v0.6 full-contact engine
([extras/README.md](../README.md)) re-runs the same scenarios, and how far that got. It was moved here from the README
in v0.7 (10 Nov 2026).

## Status

The full-contact re-runs of every scenario, launch to splashdown, were started and **stopped before they finished:
there are no v0.6 scenario results**. Notebook 05 says the same.

- **Dust amount: 100 grains per tubule in every scenario.** The amount is not decided; the configuration's 800 is an
  example. With the full contact model the dust that rolls along the walls in the coasts and under the spin (T+5 s to
  the de-spin at T+54 s) costs ~10^4 s of computing per second of flight at 800 grains per tubule (measured on the
  nominal run: ~60 grains in the full model at once), which is weeks for the campaign on a laptop. 100 grains in every
  scenario also keeps the scenarios comparable with one another. Impact counts scale with the amount (pairs ~ N^2 at
  the same density of free grains).
- **10 Nov 2026:** the campaign was restarted from scratch on the engine in which resting clusters also sleep at their
  contact limits. Static piles at their rolling limit had been kept in the full model, ~10^3-10^4 s of computing per
  flight-second in the coasts; see [the design record](design/at-limit-sleep/README.md). Its checks (14-19) pass
  (CHECKS.md, Part 2).
- **v0.7 changed two defaults** that these runs inherit: the negligible forces (mutual gravity, Lorentz, external
  field, photophoresis, radiation pressure, Brownian motion, gravity gradient) are off unless the mission config
  switches them on, and `python -m dolle_sim mission` uses the v0.5 engine. `python -m dolle_extras mission` (and
  therefore `run_campaign.py`) still uses the full-contact engine.

## How to run it

```bash
pip install -e ".[extras]"                      # numba and psutil; or run from a clone without installing:
python extras/tools/run_campaign.py --workers 8  # every scenario in extras/tools/run_campaign.py ORDER
python extras/tools/run_campaign.py kick         # or only some scenarios
python extras/tools/mission_table.py             # the Markdown tables of the results (docs/data/mission_tables.md)
```

Every tubule of every scenario is one job:
`python -m dolle_extras mission configs/mission_quarter_deck.json --scenario <s> --grains 100 --tubule <i> --autosave-min 30`,
logged to `results/mission_<s>_T<i>.log`. It has the same configuration and seed as a whole-scenario run, so the result
is the same (tested: `test_tubule_by_tubule_run_is_the_mission_run`). A pool of workers takes the next job that can
start. When a scenario's four tubules are done, its outputs are written (`--assemble`, logged to
`results/mission_<s>.log`), and a line `<s> exit=<code> <seconds> s` is appended to
`results/mission_scenarios_done.txt`. Scheduling tubules one by one keeps every core busy, because the tubules of
one scenario take very different times.

**Autosave.** Every job saves its whole state every 30 minutes of computing (`results/mission/<s>/autosave`). A job
that dies is started again and continues from its last autosave, bit for bit the uninterrupted run
(`test_killed_tubule_run_continues_from_its_autosave`). It is restarted as often as needed after running out of
memory, once the machine has the memory again, and up to three times after any other failure. A job starts only when
the machine can commit its memory. The script can be stopped and started again: finished tubules (their final states
in `results/mission/<s>/tubules`) are not run again, and jobs still running are adopted. Bit-for-bit continuation holds
within one code version only.

**Branches from checkpoints.** Scenarios that share the same flight up to some time continue from a checkpoint of one
run instead of repeating it (`--checkpoint-at`, `--resume-from`, `--resume-at`; `run_campaign.py` sets them up). The
branch is verified to be exact: the two scenarios must differ only in the release schedule or flight-profile options
and have identical loads up to the checkpoint, and a branched run is bit for bit the run from lift-off
(`test_branch_from_checkpoint_is_exact`); otherwise it runs from lift-off. The branches used:

| scenarios | continue from | at |
|---|---|---|
| `kick`, `no_release`, `strong_shaker` | `nominal` | T+89.9 s (the releases start at T+90 s) |
| `kick+spin_1hz` | `nominal` | T+54 s (the residual spin starts with the de-spin) |
| `rough_grains+hang_nose_up`, `rough_grains+shock_oblique` | `rough_grains` | T+459 s (they differ from the parachute on) |

**Old state files.** Checkpoints, autosaves and final states saved before v0.7 name the moved modules by their old
names (`dolle_sim.engine`, `dolle_sim.loads`, ...), and `dolle_sim.loads` no longer exists, so they do not load
after the move. They could not be continued bit for bit anyway, because v0.7 switched the negligible forces off.
Start such runs again from lift-off (delete `results/mission/<s>/autosave` and `checkpoints`).

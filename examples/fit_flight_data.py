"""Fit the surface energy and the dry restitution e_dry to a list of measured impacts.

Run:
    python examples/fit_flight_data.py                       # no file: makes a SYNTHETIC flight, writes it to
                                                             # synthetic_impacts.csv in the temp folder, fits it
    python examples/fit_flight_data.py examples/sample_impacts.csv
    python examples/fit_flight_data.py my_impacts.csv --plot fit.png --sigma-e-extra 0.03
    python examples/fit_flight_data.py --make-sample         # (re)write examples/sample_impacts.csv

The CSV format (one row per filmed impact; speeds in mm/s, radii in um) is printed by --help-csv and documented
in dolle_core/inverse.py (CSV_HELP).  Each material pair in the file is fitted separately.  What the fit does,
and why: dolle_core/inverse.py and notebooks/07_edry_the_energy_loss.ipynb.
"""
import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np                                    # noqa: E402

from dolle_core import inverse as inv                 # noqa: E402

SAMPLE = HERE / "sample_impacts.csv"


def make_sample(path: Path, seed: int = 2026) -> None:
    """A SYNTHETIC impact list: 200 silica-silica impacts (true gamma_scale 1.0, e_dry 0.90, radii not measured)
    and 200 graphite-graphite impacts (true gamma_scale 1.0, e_dry 0.70, radii measured)."""
    si = inv.make_synthetic_flight(200, "silica", gamma_scale=1.0, e_dry=0.90, seed=seed)
    gr = inv.make_synthetic_flight(200, "graphite", gamma_scale=1.0, e_dry=0.70, measure_radii=True, seed=seed + 1)
    err = inv.SpeedError()
    inv.write_impacts_csv(path, [si, gr], comment=(
        "SYNTHETIC DATA, NOT A MEASUREMENT. Made by examples/fit_flight_data.py (dolle_core.inverse."
        "make_synthetic_flight)\n"
        f"silica-silica: 200 impacts, true gamma_scale 1.0, e_dry 0.90, radii not reported (seed {seed})\n"
        f"graphite-graphite: 200 impacts, true gamma_scale 1.0, e_dry 0.70, radii reported (seed {seed + 1})\n"
        "true speeds log-uniform from 0.3 to 10 x the true sticking speed (ASSUMED); grains from the 90-106 um "
        "sieve cut\n"
        f"speed error sqrt({err.sigma_abs * 1e3:.3f}^2 + ({err.sigma_rel:.2f} v)^2) mm/s (dolle_core.inverse."
        "SpeedError)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="impact list (CSV); none = make and fit a synthetic one")
    ap.add_argument("--speed-error-mm-s", type=float, default=None,
                    help="absolute speed error in mm/s (default: the camera model, 0.845)")
    ap.add_argument("--rel-error", type=float, default=0.05, help="relative speed error (default 0.05, ASSUMED)")
    ap.add_argument("--sigma-e-extra", type=float, default=0.0, help="extra scatter of e not in the model")
    ap.add_argument("--sigma-ln-w", type=float, default=0.0,
                    help="spread of ln(adhesion) per impact (pair); the simulator's flight configs (0.6 per grain) "
                         "give 0.42 for grain-grain and 0.6 for grain-wall impacts")
    ap.add_argument("--stick-bounce-only", action="store_true", help="ignore the measured restitutions")
    ap.add_argument("--plot", help="save a figure of the data and the fitted model to this file")
    ap.add_argument("--make-sample", action="store_true", help="(re)write examples/sample_impacts.csv and stop")
    ap.add_argument("--help-csv", action="store_true", help="print the CSV format and stop")
    a = ap.parse_args()
    if a.help_csv:
        print(inv.CSV_HELP + ",".join(inv.CSV_COLUMNS))
        return
    if a.make_sample:
        make_sample(SAMPLE)
        print(f"wrote {SAMPLE.name}")
        return
    path = Path(a.csv) if a.csv else Path(tempfile.gettempdir()) / "synthetic_impacts.csv"
    if not a.csv:
        make_sample(path)
        print(f"no file given: wrote a SYNTHETIC example to {path.name} in the temp folder (true values in its "
              "header)\n")
    err = inv.SpeedError(sigma_abs=a.speed_error_mm_s * 1e-3, sigma_rel=a.rel_error) \
        if a.speed_error_mm_s is not None else inv.SpeedError(sigma_rel=a.rel_error)
    groups = inv.read_impacts_csv(path)
    fits = {}
    for name, imp in groups.items():
        r = inv.fit(imp, error=err, sigma_ln_w=a.sigma_ln_w, sigma_e_extra=a.sigma_e_extra,
                    use_restitution=not a.stick_bounce_only)
        fits[name] = r
        print(r.summary())
        res = inv.residuals(imp.select(r.kept), r.gamma_scale, r.e_dry, err, a.sigma_ln_w, a.sigma_e_extra)
        chi2, n = res["chi2_e"]
        print(f"  restitution residuals: chi^2 = sum of pulls^2 = {chi2:.1f} for {n} bounces (expected "
              f"{n} +- {np.sqrt(2 * n):.0f} if the model is right)")
        print("  how to read the table: a model problem shows as |mean pull| > 2x its error, or stuck observed -"
              " predicted > 2x its (+-), in several adjacent bins (notebook 07, section 5.7)")
        print("  speed / v_stick    n   stuck: observed  predicted (+-)   mean e pull (+-)")
        for b in res["bins"]:
            pull = f"{b['e_pull_mean']:+6.2f} ({b['e_pull_error']:.2f})" if b["n_e"] else "     -"
            print(f"  {b['x_lo']:5.2f} - {b['x_hi']:5.2f}  {b['n']:4d}          {b['stuck_observed']:5.2f}      "
                  f"{b['stuck_predicted']:5.2f} ({b['stuck_error']:.2f})   {pull}")
        print()
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.2), squeeze=False)
        for ax, (name, imp) in zip(axes[0], groups.items()):
            r = fits[name]
            c = inv.curves(r.gamma_scale, r.e_dry, imp.material1, imp.material2,
                           r2=None if np.all(np.isinf(imp.r2)) else inv.R_NOMINAL, error=err,
                           speeds=np.geomspace(imp.speed.min(), imp.speed.max(), 300))
            has = imp.has_restitution
            ax.semilogx(imp.speed[has] * 1e3, imp.restitution[has], ".", ms=4, label="bounces (measured e)")
            ax.semilogx(imp.speed[imp.stuck] * 1e3, np.zeros(imp.stuck.sum()), "|", ms=8, label="stuck")
            ax.semilogx(c["v"] * 1e3, c["e"], "k-", label=f"model: gamma x{r.gamma_scale:.2f}, e_dry {r.e_dry:.3f}")
            ax.axvline(r.v_stick * 1e3, color="0.5", ls=":",
                       label=f"fitted sticking speed ({2e6 * inv.R_NOMINAL:.1f} um diameter pair)")
            ax.set_xlabel("impact speed [mm/s]")
            ax.set_ylabel("restitution")
            ax.set_title(name)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print(f"figure: {a.plot}")


if __name__ == "__main__":
    main()

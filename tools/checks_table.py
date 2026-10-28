"""Write CHECKS.md: every check and comparison, what it compares, its pass criterion and the measured value.

Everything is read from the result files the checks themselves write, so the table cannot drift from the runs:
  results/verify_all.json                         python -m dolle_sim verify --json results/verify_all.json
  results/lammps_contact/lammps_contact_check.json python tools/lammps_contact_check.py   (needs pip install lammps)
  results/lammps_check/lammps_check.json           python -m dolle_sim lammps-check
  results/rebound_check/rebound_check_*.json       python -m dolle_sim rebound-check ...  (needs pip install rebound)
  results/pytest.xml                               python -m pytest tests --junitxml=results/pytest.xml
Run: python tools/checks_table.py   (also writes docs/data/checks.json: the same rows as data, kept in the repository)
"""
from __future__ import annotations

import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def _cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return out


def _ok(b):
    return "PASS" if b else "**FAIL**"


def suites():
    """Part 1 (verification.py) and part 2 (verify_full.py) from verify_all.json."""
    rows = json.loads((RES / "verify_all.json").read_text(encoding="utf-8"))
    full_ids = set(re.findall(r'_res\("(\w+)"', (ROOT / "dolle_sim" / "verify_full.py").read_text(encoding="utf-8")))
    p1 = [r for r in rows if r["id"] not in full_ids]
    p2 = [r for r in rows if r["id"] in full_ids]
    return p1, p2


def lammps_contact():
    f = RES / "lammps_contact" / "lammps_contact_check.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    crit = 1e-5
    rows = []
    for name, t in d["tests"].items():
        worst = max(float(t.get(k) or 0.0) for k in ("rel_diff_x", "rel_diff_v", "rel_diff_w"))
        extra = f"; outcome {t['outcome']}" if "outcome" in t else ""
        rows.append((name, f"position {t['rel_diff_x']:.1e}, velocity {t['rel_diff_v']:.1e}, spin "
                     f"{t['rel_diff_w']:.1e}{extra}", f"each <= {crit:.0e}", _ok(worst <= crit)))
    return rows


def lammps_threshold():
    f = RES / "lammps_check" / "lammps_check.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    rows = []
    for c in d["collisions"]:
        rel = abs(c["rel_diff_same_criterion"])
        rows.append((f"{c['material']} sticking threshold (r = {c['radius_m'] * 1e6:.0f} um, undamped)",
                     f"LAMMPS {c['lammps_threshold_m_s'] * 1e3:.3f} mm/s, DOLL-E {c['dolle_threshold_force_break_m_s'] * 1e3:.3f}"
                     f" mm/s ({rel:.2%})", "<= 1%", _ok(rel <= 0.01)))
        de = max(abs(r["lammps_e"] - r["dolle_e"]) for r in c["restitution"])
        rows.append((f"{c['material']} restitution at 1.2x, 2x, 4x the threshold",
                     ", ".join(f"{r['lammps_e']:.4f} vs {r['dolle_e']:.4f}" for r in c["restitution"]),
                     "|difference| <= 0.02", _ok(de <= 0.02)))
    for p in d["pulloff"]:
        rows.append((f"pull-off of {p['grain']} from {p['wall']}",
                     f"LAMMPS {p['lammps_detach_force_N']:.4e} N, JKR 1.5 pi w R {p['jkr_pulloff_N']:.4e} N "
                     f"({abs(p['rel_diff']):.2%})", "<= 0.5% (loading-rate lag)", _ok(abs(p["rel_diff"]) <= 0.005)))
    return rows


def rebound():
    rows = []
    for f in sorted((RES / "rebound_check").glob("rebound_check_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        b = d["ballistic_check"]
        rows.append((f"{d['chamber']}: ballistic flight, {b['grains_compared']} grains to {b['t_s']} s",
                     f"largest position difference {b['max_position_difference_m']:.1e} m", "<= 1e-9 m",
                     _ok(b["max_position_difference_m"] <= 1e-9)))
        a, r = d["dolle_engine"], d["rebound_engine"]
        for key, label in (("grain_grain_impacts", "grain-grain impacts"),
                           ("free_monomer_stick_fraction", "single-grain stick fraction")):
            ea, er = a.get(key + "_err", 0.0), r.get(key + "_err", 0.0)
            sig = math.hypot(ea, er)
            diff = abs(a[key] - r[key])
            fmt = "{:.3f}" if "fraction" in key else "{:.0f}"
            rows.append((f"{d['chamber']}: {label} in {d['duration_s']:.0f} s ({d['n_grains']} grains)",
                         f"DOLL-E {fmt.format(a[key])} +/- {fmt.format(ea)}, REBOUND {fmt.format(r[key])} +/- "
                         f"{fmt.format(er)} ({diff / sig if sig else 0:.1f} sigma)", "within 2 sigma (statistical)",
                         _ok(diff <= 2 * sig)))
    return rows


def pytest_counts():
    f = RES / "pytest.xml"
    if not f.exists():
        return None
    root = ET.parse(f).getroot()
    s = root if root.tag == "testsuite" else root.find("testsuite")
    n, fail, err, skip = (int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    return n, fail, err, skip


def main():
    p1, p2 = suites()
    out = ["# All checks and comparisons", "",
           "Generated by `tools/checks_table.py` from the result files the checks write (see the file for the "
           "commands). Every row states what is compared, the pass criterion and the measured value.", ""]
    total = passed = 0
    record = {"suites": {"part1": p1, "part2": p2}, "comparisons": {}}
    for title, rows, note in (
            ("Part 1: model checks (dolle_sim/verification.py; also run in the browser)", p1,
             "Analytic solutions, conservation laws, limits and published data for every model component."),
            ("Part 2: full-contact engine (dolle_sim/verify_full.py)", p2,
             "The contact dynamics, statics, slaved rolling, vibration synthesis and the regime machinery of the "
             "v0.6 engine against exact solutions, the full model and their own convergence.")):
        out += [f"## {title}", "", note, ""]
        out += _table(["#", "check", "what is compared", "criterion", "measured", "result"],
                      [(k + 1, r["name"], r["what"], r["criterion"], r["value"], _ok(r["passed"]))
                       for k, r in enumerate(rows)])
        out.append("")
        total += len(rows)
        passed += sum(r["passed"] for r in rows)
    for title, rows, note in (
            ("LAMMPS: the same contact law, term by term (tools/lammps_contact_check.py)", lammps_contact(),
             "pair_style granular jkr / tangential linear_history / rolling sds / twisting sds, fix wall/gran, "
             "nve/sphere, with the same coefficients and step as dem.py (contact breaking at maximum tension, "
             "LAMMPS's rule). Relative differences of the final state."),
            ("LAMMPS: sticking threshold, restitution, pull-off (dolle_sim/lammps_check.py)", lammps_threshold(),
             "LAMMPS's own JKR implementation against DOLL-E's 1-D contact model with the same break rule."),
            ("REBOUND: a chamber of flying grains (dolle_sim/rebound_check.py)", rebound(),
             "A simplified version of the same chamber in REBOUND: its IAS15 integrator for the ballistic part, "
             "its collision search with DOLL-E's contact model plugged in as the collision resolver for the "
             "statistics. Counted quantities agree within their Poisson errors.")):
        out += [f"## {title}", "", note, ""]
        if rows is None:
            out += ["(not run on this machine: needs the optional package)", ""]
            continue
        out += _table(["test", "measured", "criterion", "result"], rows)
        out.append("")
        record["comparisons"][title] = [dict(test=r[0], measured=r[1], criterion=r[2], result=r[3]) for r in rows]
        total += len(rows)
        passed += sum(r[3] == "PASS" for r in rows)
    pc = pytest_counts()
    out += ["## Unit and regression tests (tests/)", ""]
    if pc:
        n, fail, err, skip = pc
        out += [f"{n} tests: {n - fail - err - skip} passed, {fail} failed, {err} errors, {skip} skipped "
                "(`python -m pytest tests`).", ""]
    out += ["## Summary", "", f"{passed} of {total} checks and comparisons pass."
            + (f" Unit tests: {pc[0] - pc[1] - pc[2] - pc[3]}/{pc[0]}." if pc else ""), ""]
    (ROOT / "CHECKS.md").write_text("\n".join(out), encoding="utf-8")
    record["unit_tests"] = dict(zip(("tests", "failures", "errors", "skipped"), pc)) if pc else None
    record["summary"] = dict(passed=passed, total=total)
    (ROOT / "docs" / "data" / "checks.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote CHECKS.md: {passed}/{total} checks pass" + (f", {pc[0]} unit tests" if pc else ""))
    return passed, total


if __name__ == "__main__":
    main()

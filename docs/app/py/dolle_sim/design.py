"""Analytic pre-flight checks and sticking-threshold tools (no full simulation needed).

Provenance: Analytic estimates derived for DOLL-E from the same physics as the
engine. No REBOUND code.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .materials import get_material, pair_props, sphere_mass
from .contact import JKRContact
from .environment import Environment, G0

K_E = 8.9875517923e9
G_N = 6.67430e-11
Q_E = 1.602176634e-19


def preflight(cfg: dict, chamber: dict) -> dict:
    """Order-of-magnitude design numbers for one chamber, with warnings."""
    env_cfg = dict(cfg.get("environment", {}))
    env_cfg.update(chamber.get("environment", {}))
    env = Environment(env_cfg, np.random.default_rng(0))
    ov = cfg.get("materials", {})
    size = np.asarray(chamber.get("size_m", [0.05] * 3), float)
    vol = float(np.prod(size))
    wall = get_material(chamber.get("wall_material", "aluminum_6061"), ov)
    T = float(cfg.get("duration_s", env.window_s))
    groups = chamber.get("particles", [])
    n_tot = sum(int(g["count"]) for g in groups)
    out = {"chamber": chamber.get("name", "chamber"), "environment": env.describe(), "groups": [], "warnings": []}
    W = out["warnings"]

    a_res = float(np.linalg.norm(env.a_res))
    Om = env.omega
    spin = float(np.linalg.norm(Om))
    r_axis = np.asarray(chamber.get("center_from_spin_axis_m", [0, 0, 0]), float)
    corners = r_axis + (np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) - 0.5) * size
    e_ax = Om / spin if spin > 0 else np.array([0.0, 0, 1])
    r_perp = corners - (corners @ e_ax)[:, None] * e_ax
    r_far = float(np.linalg.norm(r_perp, axis=1).max())
    a_cf = spin ** 2 * r_far
    out["centrifugal_accel_max_g"] = a_cf / G0
    if a_cf > 10 * max(a_res, 1e-6):
        W.append(f"Spin ({spin / 2 / math.pi:.2f} Hz) gives up to {a_cf / G0:.2e} g centrifugal "
                 f"acceleration in the chamber -- this dominates the residual {a_res / G0:.1e} g. "
                 "Consider a despun flight or putting the chamber on the spin axis.")

    for g in groups:
        m = get_material(g["material"], ov)
        r = float(g.get("radius_m", 50e-6))
        if "size_distribution" in g:            # use the number-weighted median of the size distribution
            from .sizes import describe, sample_diameters
            r = 0.5 * float(np.median(sample_diameters(g["size_distribution"], 4000, np.random.default_rng(0))))
            desc = describe(g["size_distribution"])
            for key in ("assumption", "truncated"):
                if key in desc:
                    W.append(f"{g['material']} size distribution: {desc[key]}.")
        n = int(g["count"])
        v0 = float(g.get("velocity", {}).get("speed_mean_m_s", 0.01))
        at_rest = v0 <= 0 or (g.get("initial_state") or {}).get("type") == "settled"
        v0 = max(v0, 1e-6)                  # dust at rest: launch-speed estimates below are not meaningful
        mass = sphere_mass(m, r)
        c = float(env.drag_coeff(r))
        tau = mass / c if c > 0 else math.inf
        Kn = env.mfp / r
        pp = JKRContact(pair_props(m, r, m, r))
        wall_adh = float(chamber.get("wall_adhesion_factor", 1.0))
        pw = JKRContact(pair_props(m, r, wall, None, adhesion_factor=wall_adh))
        v_pp = pp.threshold(mass / 2)
        v_pw = pw.threshold(mass)
        # ---- force budget: typical accelerations acting on one grain [m/s^2]
        q_rms = math.hypot(float(g.get("charge_e_mean", 0.0)), float(g.get("charge_e_sigma", 0.0))) * Q_E
        sep = (vol / n_tot) ** (1 / 3)                       # mean grain spacing
        cg = np.asarray(chamber.get("center_from_cg_m", chamber.get("center_from_spin_axis_m", [0, 0, 0])), float)
        x_far = float(np.linalg.norm(np.abs(cg) + size / 2))
        Bmag = float(np.linalg.norm(env.B_field))
        Emag = float(np.linalg.norm(env.E_field))
        budget = {
            "residual_microgravity": a_res,
            "g_jitter_rms": env.jitter_rms,
            "centrifugal_payload_spin": a_cf,
            "coriolis_payload_spin": 2 * spin * v0,
            "gravity_gradient_tidal": 2 * env.tidal_rate * x_far if env.gravity_gradient else 0.0,
            "mutual_gravity_at_contact": G_N * mass / (2 * r) ** 2,
            "mutual_gravity_at_mean_spacing": G_N * mass / sep ** 2,
            "coulomb_at_mean_spacing": K_E * q_rms ** 2 / sep ** 2 / mass,
            "coulomb_at_1mm": K_E * q_rms ** 2 / 1e-3 ** 2 / mass,
            # both grains are charged and both are polarised by the other: two terms (as in the engine)
            "polarization_at_contact": 2 * 2 * K_E * q_rms ** 2 * m.polarizability * r ** 3 / (2 * r) ** 5 / mass,
            "image_charge_at_1mm_from_wall": K_E * wall.image_factor * q_rms ** 2 / (2e-3) ** 2 / mass,
            "lorentz_qvB": q_rms * v0 * Bmag / mass,
            "external_E_field": q_rms * Emag / mass,
            "gas_drag_at_launch_speed": v0 / tau if math.isfinite(tau) else 0.0,
        }
        # gas-kinetic forces (same formulas as the simulation)
        if env.P > 0 and env.grad_T.any():
            Kn_ch = env.mfp / size
            gT = np.linalg.norm(env.grad_T / (1 + 3.9 * Kn_ch))
            budget["thermophoresis"] = (32.0 / 15.0) * r ** 2 * env.kappa_tr / env.v_th * gT / mass
        else:
            budget["thermophoresis"] = 0.0
        thermo_per_K = (32.0 / 15.0) * r ** 2 * env.kappa_tr / env.v_th / (1 + 3.9 * env.mfp / float(size.min()))             / float(size.min()) / mass if env.P > 0 else 0.0     # per kelvin across the chamber
        alpha = float(cfg.get("forces", {}).get("photophoresis_accommodation", 1.0))
        budget["photophoresis"] = ((math.pi / 3) * alpha * 0.5 * m.absorptivity * env.P * env.light_I * r ** 3
                                   / (env.T * m.thermal_conductivity) / mass) if env.P > 0 else 0.0
        Q_pr = m.absorptivity + 0.3 * (1 - m.absorptivity)
        budget["radiation_pressure"] = Q_pr * env.light_I * math.pi * r ** 2 / 2.99792458e8 / mass
        v_brown = math.sqrt(1.380649e-23 * env.T / mass)
        # walls: vibration and the chance a wall hit ends in sticking
        vib = chamber.get("wall_vibration", {}) or {}
        v_vib = float(vib.get("amplitude_m", 0.0)) * 2 * math.pi * float(vib.get("frequency_hz", 0.0))
        # collisions: number density x cross-section x relative speed, decaying with drag
        nd = n_tot / vol
        sigma = math.pi * (2 * r) ** 2
        v_rel = 4.0 / 3.0 * v0           # mean relative speed of two equal-speed grains with isotropic directions
        dist = v0 * tau * (1 - math.exp(-T / tau)) if math.isfinite(tau) else v0 * T
        # walls capture grains slower than the wall-sticking speed: path ends after ~1/p_stick wall hits
        vs = float(g.get("velocity", {}).get("speed_sigma_m_s", 0.3 * v0))
        rng0 = np.random.default_rng(0)
        n_s = 20000
        speeds = np.abs(v0 + vs * rng0.standard_normal(n_s))
        if speeds.sum() <= 0:                # dust at rest (e.g. settled on a face): no wall hits to estimate
            speeds = np.full(n_s, 1e-12)
        # wall-hit rate is proportional to the speed: resample speeds flux-weighted
        speeds = rng0.choice(speeds, n_s, p=speeds / speeds.sum())
        # flux-weighted normal component of an isotropic velocity: v cos(theta), cos(theta) = sqrt(U)
        v_n = speeds * np.sqrt(rng0.random(n_s))
        # vibrating walls: phase uniform at arrival, hit rate weighted by the approach speed
        n_axes = len(set((chamber.get("wall_vibration") or {}).get("axes", [0, 1, 2]))) if v_vib > 0 else 0
        v_app = v_n + v_vib * np.cos(rng0.random(n_s) * 2 * math.pi)
        # v_n is already flux-weighted for a static wall; a moving wall changes the hit rate by (v_n + w) / v_n
        wgt = np.clip(v_app, 0, None) / np.maximum(v_n, 1e-30)
        p_vib = float((wgt * (v_app < v_pw)).sum() / wgt.sum()) if wgt.sum() > 0 else 1.0
        p_static = float(np.mean(v_n < v_pw))
        wall_mode = cfg.get("contact", {}).get("wall_sticking", "jkr")
        p_wall_true = (n_axes / 3) * p_vib + (1 - n_axes / 3) * p_static
        p_wall = max(p_wall_true, 1e-3) if wall_mode != "never" else 1e-3
        chord = 4 * vol / (2 * (size[0] * size[1] + size[1] * size[2] + size[0] * size[2]))
        dist = min(dist, chord / p_wall)
        mfp_coll = 1 / (nd * sigma)
        coll_per_grain = nd * sigma * v_rel / v0 * dist
        t_cross = float(size.min()) / max(v0, 1e-12)
        drift_v = a_res * tau if math.isfinite(tau) else math.inf
        d = {
            "material": m.name, "radius_m": r, "count": n, "mass_kg": mass,
            "knudsen_number": Kn, "stopping_time_s": tau,
            "v_stick_pp_DEM_m_s": v_pp, "v_stick_pp_energy_estimate_m_s": pp.v_stick_analytic(mass / 2),
            "v_stick_wall_DEM_m_s": v_pw, "pull_off_force_N": pp.p.F_c,
            "launch_speed_m_s": v0, "collision_mean_free_path_m": mfp_coll,
            "wall_capture_probability_per_hit": p_wall_true,
            "wall_adhesion_factor": wall_adh, "wall_vibration_speed_m_s": v_vib,
            "brownian_speed_m_s": v_brown,
            "thermophoresis_accel_per_kelvin_across_chamber_m_s2": thermo_per_K,
            "expected_pp_collisions_per_grain": coll_per_grain,
            "expected_pp_collisions_total": coll_per_grain * n / 2,
            "chamber_crossing_time_s": t_cross,
            "residual_drift_speed_m_s": drift_v,
            "rms_charge_e": q_rms / Q_E,
            "spin_down_time_s": tau * (0.3 + 0.814 * Kn / (1 + Kn)) if math.isfinite(tau) else math.inf,
            "acceleration_budget_m_s2": budget,
        }
        out["groups"].append(d)
        if tau < t_cross:
            W.append(f"[{m.name}] gas stopping time {tau:.2g} s < chamber crossing time {t_cross:.2g} s: "
                     "grains will stall in the gas before colliding. Lower the pressure "
                     f"(currently {env.P:g} Pa).")
        if at_rest:
            W.append(f"[{m.name}] dust starts at rest (settled on a face): the launch-speed estimates do not apply; "
                     "the collision rate depends on how the release mechanism frees the grains.")
        elif coll_per_grain < 0.1:
            W.append(f"[{m.name}] only ~{coll_per_grain:.2g} collisions per grain expected "
                     f"({d['expected_pp_collisions_total']:.0f} total). Increase number density, "
                     "use colliding jets/targets, or a longer window.")
        if v0 > 3 * v_pp:
            W.append(f"[{m.name}] launch speed {v0:.3g} m/s is well above the predicted sticking "
                     f"velocity {v_pp:.3g} m/s: most grain-grain impacts will bounce. Good for "
                     "bracketing the threshold only if you also sample lower speeds.")
        if wall_mode == "never":
            W.append(f"[{m.name}] wall_sticking='never' is an idealisation. With this wall "
                     f"({wall.name}, adhesion factor {wall_adh:g}) the model predicts {100 * p_wall_true:.0f}% of wall "
                     "hits would really stick -- the hardware must achieve the non-stick walls you assume.")
        elif p_wall_true > 0.05:
            W.append(f"[{m.name}] ~{100 * p_wall_true:.0f}% of wall hits are predicted to stick (wall v_stick "
                     f"{v_pw * 1e3:.2f} mm/s). To keep grains bouncing: anti-stick/rough walls "
                     "(lower 'wall_adhesion_factor', e.g. fluorinated + bead-blasted ~0.02-0.1) and/or vibrating "
                     "walls with speed amplitude well above the wall sticking speed.")
        if budget["thermophoresis"] > a_res:
            W.append(f"[{m.name}] thermophoresis ({budget['thermophoresis'] / G0:.1e} g) exceeds the residual "
                     f"microgravity: at {env.P:g} Pa each 1 K across the chamber gives {thermo_per_K / G0:.1e} g. "
                     "Keep the chamber isothermal (gradients well below ~0.1 K across it) or the drift will "
                     "dominate the relative grain motion.")
        if budget["photophoresis"] > 0.3 * a_res:
            W.append(f"[{m.name}] photophoresis from the illumination ({budget['photophoresis'] / G0:.1e} g) is "
                     "comparable to the residual gravity -- absorbing (carbon) grains drift away from the light. "
                     "Use pulsed/short illumination synced to the camera or lower intensity.")
        if env.P > 0 and Kn < 10 and (env.grad_T.any() or env.light_I > 0):
            W.append(f"[{m.name}] Knudsen number {Kn:.2g} < 10: the thermophoresis and photophoresis formulas are "
                     "free-molecular limits (Kn >> 1) and overestimate these forces at this pressure.")
        if math.isfinite(drift_v) and drift_v > 0.3 * v0:
            W.append(f"[{m.name}] residual acceleration drives a gas-drift of {drift_v:.2g} m/s "
                     "relative to the chamber, comparable to the launch speed.")
    if a_res > 0:
        L = float(size.min())
        t_sweep = math.sqrt(2 * L / a_res)   # free-molecular limit; drag only slows it
        out["residual_accel_sweep_time_s"] = t_sweep
        if t_sweep < T:
            W.append(f"Residual acceleration {a_res / G0:.1e} g sweeps a grain at rest across the "
                     f"{L * 100:.0f} cm chamber in ~{t_sweep:.0f} s (< {T:.0f} s run). Collisions must happen "
                     "early, or reduce the residual acceleration (ACS settings / chamber near the CG).")
    out["warnings"] = list(dict.fromkeys(W))
    return out


def stick_probability_curve(mat_a: str, mat_b: str, radius: float, velocities,
                            radius_rel_sigma=0.05, gamma_log_sigma=0.3, samples=200,
                            overrides=None, seed=0):
    """Monte-Carlo sticking probability vs impact speed for monomer-monomer impacts.

    Real powders are not identical spheres: sizes vary and the effective surface
    energy varies grain to grain (roughness, contamination). Each grain gets a
    log-normal surface-energy factor (sigma in natural-log units; the same model as
    the simulation's surface_energy_log_sigma, the pair uses sqrt(f_a f_b)) and a
    normally distributed radius, giving a smooth stick->bounce transition like the
    ones measured in experiments.
    """
    rng = np.random.default_rng(seed)
    ma0 = get_material(mat_a, overrides)
    mb0 = get_material(mat_b, overrides)
    probs = []
    contacts = []
    for _ in range(samples):
        ra = radius * max(0.5, 1 + radius_rel_sigma * rng.standard_normal())
        rb = radius * max(0.5, 1 + radius_rel_sigma * rng.standard_normal())
        f_pair = math.sqrt(math.exp(gamma_log_sigma * rng.standard_normal())
                           * math.exp(gamma_log_sigma * rng.standard_normal()))
        m_eff = 1 / (1 / sphere_mass(ma0, ra) + 1 / sphere_mass(mb0, rb))
        contacts.append((JKRContact(pair_props(ma0, ra, mb0, rb, adhesion_factor=f_pair)), m_eff))
    for v in velocities:
        probs.append(np.mean([c.resolve(v, me).stuck for c, me in contacts]))
    return np.asarray(probs)


def threshold_vs_radius(mat_a: str, mat_b: str, radii, overrides=None):
    ma, mb = get_material(mat_a, overrides), get_material(mat_b, overrides)
    rows = []
    for r in radii:
        c = JKRContact(pair_props(ma, r, mb, r))
        m_eff = 1 / (1 / sphere_mass(ma, r) + 1 / sphere_mass(mb, r))
        rows.append((r, c.threshold(m_eff), c.v_stick_analytic(m_eff)))
    return np.asarray(rows)


def infer_surface_energy(material: str, radius: float, v_stick_measured: float, partner: str | None = None,
                         overrides=None) -> float:
    """Invert the DEM model: which surface energy gamma reproduces a measured sticking speed?"""
    base = get_material(material, overrides)
    other = get_material(partner, overrides) if partner else None

    def vth(gamma):
        m = replace(base, gamma=gamma)
        # same-material impacts share the unknown gamma; a different partner keeps its own
        o = m if other is None or other.name == m.name else other
        c = JKRContact(pair_props(m, radius, o, radius))
        m_eff = 1 / (1 / sphere_mass(m, radius) + 1 / sphere_mass(o, radius))
        return c.threshold(m_eff)

    lo, hi = 1e-5, 50.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if vth(mid) < v_stick_measured:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.002:
            break
    return math.sqrt(lo * hi)

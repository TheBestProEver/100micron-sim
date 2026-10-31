# Design B: exact nonlinear verification (Newton certificate)

Verbatim output of one agent of the design run (29 Oct 2026). Only local file paths were replaced by `<repository>`; line numbers refer to the code as it was then (commit e512b2d). Statements about the code are the author's own and were checked by the next stage (see [README.md](README.md)); where a later stage found an error, the later stage is right.

**Title given by the author:** Quasi-static islands: sleeping clusters whose contacts sit at a cap (kernel return-map Newton certificate)

# Quasi-static islands: letting static clusters with capped contacts sleep

Read-only analysis of `dolle_sim/engine.py`, `statics.py`, `dem.py`, `loads.py`, `verify_full.py`, `tests/test_full_engine.py`. No file was edited, no simulator process started.

## 0. What the full model actually does at a cap (the basis of the design)

From `dem.contact_force` (sliding l. 306-325, rolling l. 341-374, twisting l. 376-390):

* every sds spring with history `xi`, stiffness `k`, cap `F_crit = mu * Fncrit`, `Fncrit = |Fne + 2 Fpull|`, is advanced as `xi <- rot(xi) + dxi; f = -k xi - g v; if |f| > F_crit: xi <- -(s f + g v)/k, s = F_crit/|f|` — a **radial return onto the cap evaluated at the new normal force**;
* with zero relative motion and constant load nothing changes; with zero motion and **falling Fncrit the spring is clipped every step** (plastic flow without motion); with rising Fncrit it unloads elastically from the cap;
* the twisting return takes its sign from the **rate**: `sg = sign(mt)`; at `mt == 0` an over-cap twist is reset to 0, at a tiny opposite-sign rate it flips. It is an ordinary radial return only when motion drives the spring outward;
* the dashpot term `g v` shifts the return point by `g|v|/k`.

Consequences. (i) A static cluster whose contacts are at a cap under slowly varying loads is *not* frozen in the DEM: the capped springs flow by the radial return while the configuration drifts elastically (~1e-13 m per outer step). (ii) The linear response (`lam_ref + J K^-1 dF`) is wrong at a cap by the released stiffness, which is why the current entry/stay tests (`rr <= 0.90`, `<= 0.98`) can never accept these clusters. (iii) `statics.apply_displacement` (`_advance`, `TAU = 1e3 s`) **is exactly one step of the kernel's own return map for a finite quasi-static displacement**: positions moved, histories rotated, incremented and clipped at the caps of the *new* normal force, dashpots scaled to nothing. So the production kernel already provides the exact one-shot return map; the only differences from the DEM are one-shot (per outer step) vs continuous (per 5-25 ns step) return and the dynamic lag, both bounded in section 2.

The right cheap model is therefore **rate-independent quasi-static elasto-plasticity with the kernel's return map, certified each step by evaluating the kernel at the predicted configuration (Newton)**. Sliding, rolling and twisting ratios reaching 1 stop being wake conditions; they are flow that the island executes exactly. The genuine instabilities remain wake conditions: pull-off (the JKR force maximum, beyond which no static branch exists), contact break/formation, yield, a mechanism (released stiffness leaves a soft direction), fast loads, and the twisting quirk.

## 1. Criteria

Notation: island state `S_T = (x, h)` for members `mem`, contacts `(ci, cj, rows)`, at rest, exact equilibrium of `F(T)` (`|F(T) + Fnet(S_T)|_inf <= tol_F`). `K, J` elastic tangents by `statics.jacobians` at the last refresh, `lu = splu(K)`, `C` the damping tangent (`jacobians(..., vel=True)`), `G, Md` the rotating-frame/drag and mass matrices as built in `_linres`. `cap_k` the cap force of each sds component, `rho_k = cap_k / k_k` its displacement, `tol_F = 1e-6 * min_k cap_k`.

### 1.1 Stay certificate (every outer step `[T, T+H]` in `_step_once`; every sample in `_skip_asleep`)

Newton on the exact kernel with the frozen elastic `K` as iteration matrix:

```
F1  = _island_load(mem, T+H)
du0 = lu.solve(F1 - F_ref)
for i in range(N_it = 40):
    xi, hi = apply_displacement(x, ci, cj, prm, h, L, du_i)      # one-shot return of the TOTAL increment from S_T
    snap_to_caps(hi, ...)                                        # 2.7: no zero-rate re-clipping, no twist reset
    lam_i, Fnet_i, out_i = state(xi, ci, cj, prm, hi, L)
    r_i = F1 + Fnet_i;   q_i = |r_i| / |r_{i-1}|
    if |r_i|_inf <= tol_F: break
    du_{i+1} = du_i + lu.solve(r_i)
```

The accumulated `du` is always applied to `S_T`, so the fixed point is the equilibrium of the one-shot return-mapped kernel (the map is continuous and piecewise linear; the iteration contracts with factor `q ~ released stiffness / total stiffness` in the released directions: ~1e-5 for pile grains with >= 2 contacts, ~0.4 for an edge grain held only by its twist springs, -> 1 for a mechanism).

Checks, any failure -> wake with the reason counted as `qs_wake_<reason>`:

* **S1 loads**: no fast window / breakpoint in `[T, T+H]` (existing rule, unchanged).
* **S2 newton**: converged within `N_it` with observed `q <= 0.9`. Non-convergence = the released caps leave a (near-)mechanism (e.g. a one-contact grain at its rolling cap, both rolling caps and the twist of an edge grain) -> the DEM must integrate it (it will roll).
* **S3 integrity**: `out_i[:,10] == 1` for every island contact (no break); no yield (`delta <= P_DY` or `<= H_DMAX`); no new contact possible: `2 * cum_motion < g_min` (`g_min` = smallest gap of non-listed member pairs and member-wall pairs, computed at entry).
* **S4 quasi-static**: `|du|_inf <= d_qs = 1e-10 m` (= 1e-3 `d_cap`); faster drift = a moving cluster.
* **S5 twist quirk**: for every contact whose trial twist `|k_tw (psi + dpsi)| > Mtcrit(Fne_new)`: require `dpsi != 0 and sign(dpsi) == sign(psi)`. Otherwise the kernel's answer (reset to 0 or flip) depends on the sign of a ~1e-20 rad/s rate: hand the cluster to the DEM, which is by definition the answer.
* **S6 pull-off**: `r_pull,k(S_{T+H}) + B_k / Fpull_k <= 1 - margin` for all contacts (`margins_bounded` with the accumulated bound `B` of section 2).
* **S7 flow**: plastic increment per contact and step `dp_k <= p_max * rho_k`, `p_max = 0.05` (keeps the one-shot return a fine approximation; faster flow belongs to the DEM).

On success the island carries a **pending** state `(x', h', lam', out', F_ref = F1, bnd', capped')` and `du_max = |du|_inf`. The DEM sees the island at `S_T` during the step (static obstacle) with `s[members] = du_max` instead of 0 in every motion bound; the pending state is **committed** after the DEM loop (before `_verify`) or discarded on a redo.

### 1.2 Entry (`_try_sleep_island`; replaces `rr.max <= 1 - 5 margin` for the sds components)

* **E0**: S1; `len(mem) <= max_island`; `splu(K)` succeeds; `energy_bounds` returns (symmetric `K` positive definite).
* **E1**: Newton (the loop above) to the exact equilibrium of `F(T)` from the DEM state (evaluated on a snapped copy); S2-S5 must hold. `Ek + Ep` with `Ep = 1/2 du.K du` is the residual energy that will be dropped.
* **E2 pull-off**: `r_pull <= 1 - 5 margin` (hysteresis kept for the only genuine instability).
* **E3 residual**: `dl = energy_bounds(K, J, Ek + Ep)` (exact ellipsoid bound, existing). `capped_k := ratio_k >= 1 - 1e-9` per sds component. At capped components require `2 dl_k <= eps_r * cap_k`, `eps_r = 1e-3` (shakedown, 2.6). At elastic components the existing rule (`margins_bounded(lam, dl) - ratio <= 1% of (1 - ratio)`); if it fails and no component is capped: `_linres` as now (a tracked residual is linear/elastic, so linres islands keep the present linear stay test and wake at `1 - margin` on any component).
* **E4**: the first stay certificate (to `T+H`) passes.

Then place at the Newton equilibrium (`apply_displacement`, as now), `v = w = 0`, store the island.

### 1.3 Wake

* S1 or a certificate failure at the step start: release `S_T` at rest (no displacement is needed: `S_T` is the exact equilibrium of `F(T)`, committed at the end of the previous step).
* Mid-step approach (`_watch`, a DEM grain within `2 d_cap`): state at `T_w` by the same Newton from `S_T` with `F(T_w)` (fallback `S_T`, counted `qs_wake_fallback`); release at rest.
* linres islands: as now (residual added back).
* Single grains on one wall (`_try_sleep`, `_margins`, `_wake`): unchanged; a one-contact grain at its rolling cap is a mechanism and rolls (ROLL / roll1.py), which is correct. The refused "size 1" clusters are edge/corner grains (`nwall > 1`) and go through the island path.

## 2. Error budget: what is exact, what is bounded

**Exact**: the equilibrium configuration at every sample time (to `tol_F`, with its effect bounded); `Fncrit` at the new normal force (the kernel evaluates it); the radial part of every return (scaling, identical one-shot or continuous); the flow amounts (the kernel's own rule); break/yield/approach detection; pull-off evaluation; the loads at the samples; the obstacle bookkeeping (`s = du_max`).

**Bounded** (per contact `k`, force units; `B_k` = sum; reported):

* **2.1 Residual tolerance** `bnd_tol = |J K^-1 r|` (exact first-order vector; x2 at capped components, see 2.10). `tol_F = 1e-6 cap` (forces ~1e-5 N, double round-off ~1e-20 N, so it is reachable). Not cumulative: each step converges anew. Size <= 1e-6 cap.
* **2.2 One-shot vs continuous return.** Spring at the cap, radius `rho0 = cap(T)/k`, direction `xhat`; trial increment `Delta` (relative sds motion of the step) with radial part `Delta_r = Delta.xhat`, tangential `Delta_t`; new cap `rho1`. One-shot: `xi' = rho1 (rho0 xhat + Delta)/|rho0 xhat + Delta|`, rotation `phi_1 = atan(|Delta_t|/(rho0 + Delta_r))`. Continuous (the DEM limit): rotation `phi_c = int |Delta_t| dt / rho(t) ~ |Delta_t| / rho_bar`. Both end at radius `rho1` exactly. `|phi_c - phi_1| <= |Delta_t| (|Delta_r| + |rho1 - rho0|) / rho^2 + |Delta_t|^3 / (3 rho^3)`, `rho = min(rho0, rho1)`; in displacement `e_ret,k = |Delta_t| (|Delta_r| + |Delta rho| + |Delta_t|) / rho`; zero when the trial stays inside the cap (linear springs: one-shot = continuous up to `rot_rescale`'s O(dtheta^2) ~ 1e-18). Cumulative (`bnd_ret`). Size: `Delta_t ~ R dtheta ~ 1.7e-13 m = 1e-5 rho`, `Delta rho <= 1e-4 rho` per step -> `<= 1e-9 rho` per step, `<= 5e-6 rho` over a 20 s coast.
* **2.3 Load path inside the step** (rate-independent plasticity is path dependent; the one-shot assumes the chord). `dF_path = m * Loads.accel_dev_bound(T, T+H)` (new: the curvature term of `accel_bound`, `h^2 |a''|/8` from 5 samples, doubled, plus exact evaluation at `knots_in`; spin: `Omega` piecewise linear so the chord is exact, `Omega^2` quadratic so its deviation is exactly `(Delta Omega)^2 r / 4`; release and vibration are fast windows and wake). Effect `d_path,k = 2 |J_k K^-1 dF_path|` (x2 at capped). An excursion beyond the chord flows outward and returns inside, so the plastic variable differs from the chord path's by at most `d_path/k`, **one-time** (shakedown): `bnd_path` is a running max, not a sum. Size: `a'' ~ 0.5 m/s^3` (drag decay) -> `da = 1e-6 m/s^2`, `dF ~ 1e-15 N/grain`, <= 3e-9 cap.
* **2.4 Dynamic lag** (the DEM integrates `M u'' + (C+G) u' + K u = F(t)`): `u = u_qs - K^-1 (C+G) u_qs' - K^-1 M u_qs'' + ...`. With `u_qs' = du/H`, `u_qs'' = K^-1 D2F/H^2` (`D2F` the doubled second difference of the three load samples): `bnd_dyn = |J K^-1 (C+G) du| / H + |J K^-1 M K^-1 D2F| / H^2`, computed as exact vectors with the island's `lu`, multiplied by `1/(1-q)` because at flowing contacts the effective stiffness is `K_ep >= (1-q) K` in the released directions. Not cumulative (a bounded lag at every instant; at a flowing contact the plastic variable follows the lagged cap by `mu dFne_dyn / k`). Size: lag time `2 zeta/omega <= 1/(Q omega_r) = 4.1e-7 s` (rocking, Q = 100, 2.4e4 rad/s is the slowest), rate <= 1e-4 cap / 4 ms -> <= 1e-8 cap.
* **2.5 Dashpot in the return**: the DEM returns onto `|s f + g v|`, i.e. `g|v|/k` off the cap; `|v| <= 2 |Delta xi| / H` (quasi-static rate, factor 2 for in-step variation): `bnd_dash = g_k * 2|Delta xi_k| / (H k_k)`. Size `3.5e-13 N s/m * 8.5e-11 m/s / 1.35e-8 N = 2e-15` cap.
* **2.6 Entry residual (ratchet / shakedown)**: the island is placed at rest, dropping `Ek + Ep`; in the DEM that energy decays as damped oscillation with largest excursion `dl_k = sqrt(2E (J Ks^-1 J^T)_kk)` (exact for the ellipsoid; conservative at capped components, where `Ks` from central differences is softer, `k/2`, and outward excursions dissipate). A decaying oscillation of amplitude `delta` at a capped perfectly plastic spring flows once by `delta` (first outward half-cycle), then sits `delta` inside the cap and every later cycle is elastic; superposed on monotonic flow the mean stays <= `delta` inside with peaks touching the cap. Hence the plastic variable differs from the residual-free path by <= `2 delta` at any time (2 for the 2-D circular cap: oblique excursions can rotate and shrink). `bnd_res,k = 2 dl_k`, one-time; entry requires `<= 1e-3 cap`. Size for a 57-grain pile at 2e-9 m/s: `E ~ 1.1e-25 J`, `dl ~ 4e-13 N = 3e-5 cap`, `bnd_res = 6e-5 cap`; at 1e-13 m/s ~1e-9; at 1e-7 m/s 1.5e-3 -> refused (correctly: moving).
* **2.7 Cap snapping**: after every return map and before every zero-rate kernel evaluation, sds components with ratio `> 1 - 2^-40` are scaled to exactly `(1 - 2^-40)` of the cap, so the kernel at zero rate never re-clips (and never hits the twist reset). Relative force change <= 9.1e-13 per clipping event, cumulative `bnd_snap` (<= 1e-7 over 1e5 flowing steps).
* **2.8 Twist quirk**: never approximated (S5 wakes).
* **2.9 Fncrit dependence**: exact in the return map; in the first-order bound operators it enters through `J`'s normal row times `mu`.
* **2.10 J at capped components**: central differences straddle the kink (`J_k = 1/2` of the elastic sensitivity for unloading, 0 for loading); every first-order bound term at capped components is doubled. The certificate never uses `J` (the kernel is evaluated). `K, J, C` are refreshed (section 3) so the bound operators stay within ~1%; a factor 2 safety is applied to all first-order terms.
* **2.11 Island motion vs the "static obstacle" assumption**: `s[members] = du_max <= 1e-10 m` enters `_watch`'s `lim`, `_early_exit`'s `s_full`, `_capture_times`' `vj` (as `du_max/H`); the committed positions change only between steps. Exact bookkeeping, no approximation.

**Totals for the measured cases** (silica 50 um on borosilicate: `Fc = 1.06e-5 N`, `a0 = 0.38 um`, `xi_crit = 15.7 nm`, `mu_r = 6.3e-4`, `k_r = 0.85 N/m`, `k_s = 2.0e4 N/m`, `k_tw = 3.5e-9 N m/rad`, `Mtcrit ~ 7e-11 N m`):

| term | boost/coast pile (bottom contact, 57 grains at 2e-9 m/s) | microgravity edge grain |
|---|---|---|
| cap force / displacement | `Frcrit = 1.4e-8 N`, `rho = 16 nm` | `Mtcrit = 7.1e-11 N m`, `psi = 0.02 rad` |
| cap change per step | <= 1e-4 rho | ~1e-9 |
| 2.1 tol | 1e-6 | 1e-6 |
| 2.2 return (cumulative, 5000 steps) | 5e-6 | ~0 |
| 2.3 path (max) | 3e-9 | ~0 |
| 2.4 dynamic lag | 1e-8 | ~0 |
| 2.5 dashpot | 2e-15 | ~0 |
| 2.6 entry ratchet (one-time) | 6e-5 | 3e-9 |
| 2.7 snap | 1e-12 | 1e-12 |
| **B / cap** | **<= 7e-5** (1.1e-12 m of a 16 nm rolling history) | **<= 1e-6** |

Reported in `n_events`: `qs_bound_max_rel` (max `B_k/cap_k` over the run), `qs_bound_cum_rel` (the cumulative parts), `qs_flow_total_m`, `qs_flow_contact_steps`, `qs_island_steps`, `qs_newton_iters`, `qs_refactor`, `qs_wake_<reason>`; they flow into the timeseries via `_record`.

## 3. Code changes

### statics.py (pure array functions, each unit-tested against the kernel)
* `sds_increments(x, ci, cj, prm, L, dU) -> (dxi_t (nc,3), dxi_r (nc,3), dpsi (nc,))`: the kernel's kinematic maps (`vtr`, `vl`, `mt` times dt with `dU` as displacement). Test: equals `h_new - rot(h_old)` from `apply_displacement` on an elastic contact to 1e-15.
* `caps(lam, prm, out) -> (Fscrit, Frcrit, Mtcrit)`; `capped_mask(lam, prm, out, tol=1e-9) -> (nc,3) bool`.
* `snap_to_caps(hist, prm, out, tol=2**-40)`: in place; returns the largest relative change (2.7).
* `twist_quirk(hist_old, dpsi, prm, Fncrit_new) -> bool mask` (S5).
* `flow_and_return_bound(hist_old_rot, incr, hist_new, caps_old, caps_new, k) -> (dp (nc,3), e_ret (nc,3))` (S7, 2.2).
* `qs_step(x, ci, cj, prm, hist, L, lu, F_target, tol, n_it, q_max) -> dict(du, xn, hn, lam, out, Fnet, r, iters, q, ok, broke, yielded)`: the Newton loop of 1.1.
* `first_order_effect(lu, J, vec, capped) -> |J K^-1 vec|` with the x2 at capped components (2.1, 2.3).
* `dynamic_lag(lu, J, C, G, Md, du, H, d2F, capped, q)` (2.4).
* `island_gaps(x, rad, ci, cj, L) -> g_min` (S3).
* Unchanged: `jacobians`, `state`, `apply_displacement`, `margins`, `margins_bounded`, `energy_bounds`, `residual_*` (callers pass snapped histories to `state`).

### loads.py
* `Loads.accel_dev_bound(T0, T1)`: deviation of `a(t)` from its chord over `[T0, T1]` (the curvature term of `accel_bound` plus exact evaluation at `knots_in`); the engine adds the exact `Omega^2` term.

### engine.py
* `_try_sleep_island(mem, touch, T, H)`: rewritten per E0-E4. Island dict: `members, ci, cj, rows, prm, K, J, C, G, Md, lu, lam, out, F_ref, T_ref, capped (nc,3), bnd {res, ret, path, dyn, dash, snap, tol: (nc,8)}, g_min, cum_motion, n_since_refresh, q_last` (+ `res, dl` for linres islands).
* `_island_advance(isl, T0, T1) -> (ok, isl_new, reason)`: `F1`, `dF_path`, `qs_step`, S2-S7, bound update; returns a **new** dict with `pending` and `du_max`; never mutates `isl` (copy-on-write, so `_snapshot`'s shallow `dict(self._islands)` and `_restore` stay exact).
* `_island_commit(iid)`: `self.x[mem], self._hist[rows] <- pending`; pending -> reference fields; `_island_refresh` (recompute `K, J, C, lu` at the committed state) when `n_since_refresh >= 50`, `|Delta lam_n| > 1% Fncrit`, cumulative `du` since refresh `> 1e-11 m`, or `iters > 6`.
* `_island_ratio`: kept only for linres islands (unchanged).
* `_wake_island(iid, T)`: per 1.3 (pending if `T` is the step end; else `qs_step` to `F(T)`; fallback counted).
* `_step_once` l. 495-505: per island `fast -> wake`; `"res" in isl -> old test`; else `_island_advance(isl, T, T+H)` -> wake or store `isl_new`, `s_isl[members] = du_max`. L. 550 `s[ASLEEP] = 0` -> `s_isl` (0 for single sleepers); l. 671 `s_full[ASLEEP] = 0` -> `s_isl`; `self._s_isl_max` stored; after the DEM loop and before `return self._verify(...)`: commit every pending island still asleep.
* `_watch`: `lim[q] = min(lim[q], gap.min() - d2 - self._s_isl_max)`; the sleeper tree cache already rebuilds on changed positions.
* `_capture_times` / `_early_exit` (l. 896, 1135-1143): ASLEEP => static becomes `s_isl`, `vj = du_max/H`.
* `_skip_asleep`: per sample `q`, advance every island from its committed state to `Ts[q]`; commit all if all pass, else stop at `q` (present `ok_until` logic); single sleepers tested as now.
* `__getstate__`: drops `lu` (as now); `pending` never exists between steps (asserted). `_island_upgrade(isl)` for pickles/autosaves of the present version: `capped` from `margins(lam) >= 1 - 1e-9`, `bnd` zeros, `C, G, Md` lazily at the first advance, `g_min`, `cum_motion = 0`. `branch()` / `BRANCH_ALLOWED` / `state_signature` unaffected (no islands at init).
* Config (`contact.*`): `qs_islands` (True), `qs_tol_rel` 1e-6, `qs_newton_max` 40, `qs_q_max` 0.9, `qs_motion_m` 1e-10, `qs_flow_max` 0.05, `qs_residual_cap_rel` 1e-3. `all_dem=True` untouched (reference mode bypasses `_classify`).
* Module docstring (ASLEEP paragraph), README status, CHECKS.md (regenerated).

Interactions: **linres** — elastic-only islands keep the present linear stay test and wake at a cap (a tracked residual could ratchet); **slaved** — `_classify` still tries `_try_sleep_island` before `_try_slaved` and for SLAV clusters each step; static clusters with redundant contacts (refused by `_try_slaved`'s `3K > 6N`) now sleep, moving ones stay slaved; **_verify** — island members are outside the DEM with `S = du_max`; mid-step wakes get `t_in`/`x_ref` at `T_w` as now; **_skip_asleep** — commits per sample; **checkpoints** — `lu` rebuilt by `_island_lu` as now, old islands upgraded.

## 4. Verification

New `verify_full` checks, each against all-DEM at the production step and at half of it (the DEM's own step error), same contacts, and against the engine's own reported bound `B` (so the bound itself is validated):

* **A `check_qs_capped_pile`** — three 50 um silica grains driven into a floor/-x-wall corner by a 2 g lateral load (all-DEM preparation, 10 ms) until static with the floor rolling springs at their caps (`statics.margins >= 0.999` asserted). Then 30 ms in three phases: constant load; pressing load 1 g -> 0.3 g (caps shrink: flow without motion); lateral load rising to 0.9 of the sliding limit (caps grow: elastic unloading, sliding approached). Criteria: engine asleep >= 95% of the time, `qs_flow_contact_steps > 0` in phase 2; positions `<= 3x DEM step error + 1e-12 m`; every contact's `xt, xr, xtw` `<= 3x DEM step error + B_k/k_k`; at the flowing contacts `|k_r xr| = Frcrit(final Fne)` within 1e-9 (the exact cap); reported `B <= 1e-3 cap`. Proves: flow under a shrinking cap and elastic unloading match the full model and the bound holds.
* **B `check_qs_edge_twist`** — one grain wedged in a floor/-x-wall edge (0.66 uN locked-in normal forces from a 1 g push and release), sliding springs ~20%, twist pre-loaded to 0.96 of its cap (Euler torque of a short spin-up in the all-DEM preparation), then 50 ms at 1e-4 g and 0.0167 Hz spin. Criteria: the engine sleeps at the first classification (the present engine refuses at 0.96) and stays asleep; positions `<= 3x step error + 1e-12 m`; histories unchanged in both runs to 1e-12 relative; `B <= 1e-6 cap`. Proves: entry at 0.96 is safe; the full model is static there.
* **C `check_qs_bridge`** — the 3-grain bridge (two on the floor, one on both), lateral load ramped so the bridge contacts' rolling ratio goes 0.93 -> 1 -> flowing -> 0.5 over 30 ms. Criteria as A, plus no wake during the cap crossing (asleep >= 95%). Proves: a cap can be reached and left while asleep.

Unit tests (`tests/test_full_engine.py`, new `tests/test_statics_qs.py`): `sds_increments` = kernel (1e-15); one-shot vs 1000-sub-step return within `e_ret` (random capped rolling trials, radial+tangential increments, changing cap); `snap_to_caps` -> `state()` reproduces `lam` to 1e-12 and never zeroes a twist; `twist_quirk` fires for (`dpsi = 0`, over cap) and (opposite sign), not for consistent flow; `qs_step` converges in <= 3 iterations on a corner pile (`q < 0.01`), reports `q ~ 0.4` on an edge grain held by twist only, fails on a one-contact grain at its cap (mechanism refused); the island sleeps, flows and commits (`|k_r xr|` = new cap exactly), `s[members] = du_max > 0`; each wake reason; redo via `_snapshot/_restore` leaves island state identical (copy-on-write); pickle round trip continues bit-identically (extends `test_checkpoint_with_sleeping_cluster_is_exact`); old-format island dict upgrades; `_skip_asleep` commits per sample and stops at a failing sample. Existing checks 4, 8, 10, 11, 13 and all 98 tests must still pass; CHECKS.md regenerated with `tools/checks_table.py`.

## 5. Expected speed-up and risks

Per island per outer step: 1 LU solve + 2-4 x (`apply_displacement` + `state`) (~nc kernel calls each, 0.1-0.3 ms for 150 contacts) + bounds (~0.2 ms) = 0.5-2 ms; 250 steps per flight-second -> 0.1-0.5 s per flight-second per island. Boost/coast with ~10 islands: <= 5 s/flight-second against 6000-10000 now if the few movers (1e-4 m/s) stay on the slaved models; if 3 grains remain in the DEM at 5-18 ns, ~450 s/flight-second (still 15-20x). Microgravity (800 s): <= 1 h per tubule instead of days; optional adaptive sample lengthening in `_skip_asleep` when `|dF|` per sample `< 1e-6 cap` brings it to minutes. Campaign: weeks -> ~1-3 days.

Risks and failure modes:
1. Newton with the frozen elastic `K` converges slowly or fails for soft clusters (edge grains held only by twist springs, `q ~ 0.4-0.9`): churn (wake -> DEM -> retry each step); cost is the status quo for those. Mitigation (phase 2): an elasto-plastic iteration matrix `K_ep = K - sum(released radial stiffness)` built from `sds_increments`, or Aitken acceleration; `qs_wake_newton` counts it.
2. The DEM reference at a cap is itself noisy (ratchet from its own residual, round-off clipping); verification differences will be dominated by the DEM's own step noise at the 1e-5 cap level; the criteria include the step error and `B`.
3. Twist quirk wakes: a capped twist with falling `Fncrit` lands in the DEM, whose outcome there is noise dependent (LAMMPS-conformant kernel behaviour). A kernel fix (sign from the spring at zero rate) is a physics change outside this design; flagged.
4. The load-path bound (2.3) uses the project's sampled-curvature practice (as `accel_bound`); a load oscillating inside a 4 ms step without being a fast window would evade it. A derivative bound from the profile's analytic form would make it rigorous.
5. Autosave resume: the continued campaign runs switch sleep rule mid-flight; both rules are exact-or-bounded, but "bit-for-bit continuation" holds only within one engine version; bounds are reported from the resume point.
6. Any place that assumes ASLEEP => `s = 0` that is missed (`_early_exit`, `_capture_times`, `_watch`) would be an unverified 1e-10 m assumption; the `_missed` safety net still catches contact formation.
7. Cumulative bounds over an 800 s microgravity phase (2e5 samples): `bnd_ret` stays ~2e-4 rho at worst; reported, not a wake criterion.
8. Yielded (plastic-flattened) contacts: the certificate wakes on any yield in the step; silica and graphite overlaps (3-9 nm) are far below `d_y` (0.1-1 um), so this is dormant.

## Entry criterion (summary)

Replace `rr.max <= 1 - 5*margin` on the sds components by a Newton certificate on the production kernel. E0: no fast window/breakpoint in [T, T+H]; size <= max_island; K factorises; sym(K) positive definite. E1: Newton from the DEM state (snapped copy) to the exact equilibrium of F(T) — du_{i+1} = du_i + K^-1 (F(T) + Fnet(apply_displacement(S, du_i))), the accumulated du always applied to the start state (one-shot return of the kernel), stop at |r|_inf <= tol_F = 1e-6 min cap force within 40 iterations with contraction q <= 0.9 (else a mechanism: refuse); no break/yield; |du|_inf <= 1e-10 m; twist-quirk test passes. E2: pull-off r_pull <= 1 - 5*margin (hysteresis kept only for the genuine instability). E3: residual energy Ek + Ep (Ep = 1/2 du.K du) -> dl = energy_bounds(K, J, E) (exact ellipsoid); capped components (ratio >= 1 - 1e-9): 2 dl_k <= 1e-3 cap_k (one-time shakedown ratchet); elastic components: the existing 1%-of-margin rule, or linres if nothing is capped. E4: the first stay certificate to T+H passes. Then place at the Newton equilibrium (apply_displacement), v = w = 0; store capped mask, bounds, g_min, C/G/Md.

## Stay and wake (summary)

Stay (each outer step in _step_once and each sample in _skip_asleep): from the committed state S_T (exact equilibrium of F(T), at rest), Newton with the frozen elastic K to the equilibrium of F(T+H) through the kernel's own return map (apply_displacement + snap_to_caps + state), then: S1 no fast window/breakpoint; S2 converged in <= 40 iterations with q <= 0.9; S3 no contact broke, no yield, 2*cum_motion < g_min; S4 |du|_inf <= 1e-10 m; S5 no twist over its cap with zero or opposite-sign increment; S6 r_pull + B/Fpull <= 1 - margin (margins_bounded with the accumulated bound); S7 plastic increment per contact <= 5% of its cap displacement. Sliding/rolling/twisting ratios reaching 1 are not wake conditions — they are flow that the return map executes exactly. On success the end state is pending, the DEM sees S_T as a static obstacle with s[members] = |du|_inf (not 0), and the state is committed after the DEM loop (discarded on a redo). Wake: any S-failure at the step start -> release S_T at rest (reason counted qs_wake_*); fast window/breakpoint -> release S_T; a DEM grain within 2 d_cap mid-step (_watch) -> Newton to F(T_w) from S_T (fallback S_T, counted), release at rest; linres islands keep the present linear test and wake at 1 - margin on any component.

## What is exact and what is bounded (summary)

Exact: equilibrium configuration at every sample (to tol), Fncrit at the new normal force, the radial part of every return, the flow amounts (the kernel's rule), break/yield/approach detection, pull-off evaluation, loads at the samples, the obstacle bookkeeping (s = du_max <= 1e-10 m). Bounded per contact (force units, B_k = sum, reported as qs_bound_max_rel / qs_bound_cum_rel): (1) residual tolerance |J K^-1 r| <= 1e-6 cap, not cumulative; (2) one-shot vs continuous radial return: e_ret = |Delta_t|(|Delta_r| + |Delta rho| + |Delta_t|)/rho per step (geometry of the return onto the cap circle; radial part exact; zero when elastic), cumulative, <= 1e-9 rho per step, <= 5e-6 rho over a 20 s coast; (3) load path inside the step (plasticity is path dependent): 2|J K^-1 m accel_dev_bound| (sampled curvature h^2|a''|/8 doubled + exact at knots, Omega^2 exact), one-time per excursion (shakedown) so a running max, <= 3e-9 cap; (4) dynamic lag |J K^-1 (C+G) du|/H + |J K^-1 M K^-1 D2F|/H^2 times 1/(1-q), from u = u_qs - K^-1 C u_qs' - K^-1 M u_qs'' (lag time 2 zeta/omega <= 1/(Q omega_r) = 4.1e-7 s), <= 1e-8 cap, not cumulative; (5) dashpot in the return g*2|Delta xi|/(H k) ~ 2e-15 cap; (6) entry residual dropped: 2 dl_k with dl = energy_bounds (exact ellipsoid), a decaying oscillation at a capped perfectly plastic spring ratchets once by its amplitude and then is elastic (2 for the 2-D cap), one-time, required <= 1e-3 cap; 6e-5 cap for a 57-grain pile at 2e-9 m/s, 1e-9 at 1e-13 m/s, 1.5e-3 at 1e-7 m/s (refused); (7) cap snapping to (1 - 2^-40) of the cap: 9.1e-13 per clipping event, cumulative <= 1e-7; (8) twist quirk never approximated (wake); (9) first-order bound terms doubled at capped components because central-difference J is half the unloading sensitivity there, plus a factor 2 for K staleness between refreshes. Totals for the measured cases: boost/coast pile B <= 7e-5 of the rolling cap force (1.1e-12 m of a 16 nm rolling history), dominated by the one-time entry term; microgravity edge grain B <= 1e-6 of its caps.

## Code changes proposed

- statics.py: add sds_increments(x, ci, cj, prm, L, dU) -> (dxi_t, dxi_r, dpsi), the kernel's kinematic maps (unit-tested = apply_displacement on an elastic contact to 1e-15)
- statics.py: add caps(lam, prm, out), capped_mask(lam, prm, out, tol=1e-9), snap_to_caps(hist, prm, out, tol=2**-40), twist_quirk(hist_old, dpsi, prm, Fncrit_new), flow_and_return_bound(hist_old_rot, incr, hist_new, caps_old, caps_new, k), island_gaps(x, rad, ci, cj, L)
- statics.py: add qs_step(x, ci, cj, prm, hist, L, lu, F_target, tol, n_it, q_max) -> dict(du, xn, hn, lam, out, Fnet, r, iters, q, ok, broke, yielded): the one-shot-return Newton loop; first_order_effect(lu, J, vec, capped) and dynamic_lag(lu, J, C, G, Md, du, H, d2F, capped, q); jacobians/state/apply_displacement/margins/margins_bounded/energy_bounds/residual_* unchanged
- loads.py: add Loads.accel_dev_bound(T0, T1): deviation of a(t) from its chord (curvature term of accel_bound + exact at knots_in)
- engine.py _try_sleep_island: rewrite per E0-E4 (Newton equilibrium at T, capped mask, residual rules 2 dl <= 1e-3 cap at capped / 1%-of-margin elsewhere / linres only if nothing capped, pull-off <= 1 - 5 margin, first stay certificate); island dict gains C, G, Md, capped, bnd{res, ret, path, dyn, dash, snap, tol}, g_min, cum_motion, n_since_refresh, q_last
- engine.py: new _island_advance(isl, T0, T1) -> (ok, isl_new, reason) implementing S1-S7 and the bound update, copy-on-write (never mutates isl; stores pending state and du_max)
- engine.py: new _island_commit(iid) (pending -> self.x, self._hist, reference fields) and _island_refresh (recompute K, J, C, lu when n_since_refresh >= 50, |Delta lam_n| > 1% Fncrit, cumulative du > 1e-11 m, or iters > 6)
- engine.py _island_ratio: kept only for linres islands (elastic, unchanged test, wake at 1 - margin on any component)
- engine.py _wake_island(iid, T): release pending if T is the step end, else qs_step from the committed state to F(T) (fallback committed state, counted qs_wake_fallback); residual added back for linres islands as now
- engine.py _step_once: island block (l. ~495-505) calls _island_advance or wakes; s[ASLEEP] = 0 (l. 550) and s_full[ASLEEP] = 0 (l. 671) become s_isl = du_max per island member; store self._s_isl_max; commit pending islands after the DEM loop before _verify
- engine.py _watch: lim[q] -= self._s_isl_max; _capture_times / _early_exit (l. 896, 1135-1143): ASLEEP static terms use s_isl and du_max/H
- engine.py _skip_asleep: advance every island per sample from its committed state, commit all when all pass, stop at the first failing sample (existing ok_until logic)
- engine.py __getstate__/_island_lu: lu dropped as now, pending asserted absent; new _island_upgrade(isl) for islands pickled by the present version (capped from margins, bnd zeros, C/G/Md lazily, g_min, cum_motion = 0); _snapshot/_restore unchanged (copy-on-write islands)
- engine.py: n_events keys qs_island_steps, qs_newton_iters, qs_refactor, qs_flow_contact_steps, qs_flow_total_m, qs_bound_max_rel, qs_bound_cum_rel, qs_wake_{newton,pulloff,approach,motion,twist_quirk,flow,break,yield,fallback}, cluster_awake_residual_capped; config contact.qs_islands (True), qs_tol_rel 1e-6, qs_newton_max 40, qs_q_max 0.9, qs_motion_m 1e-10, qs_flow_max 0.05, qs_residual_cap_rel 1e-3
- engine.py module docstring (ASLEEP paragraph), README status, CHECKS.md regenerated by tools/checks_table.py
- verify_full.py: check_qs_capped_pile, check_qs_edge_twist, check_qs_bridge (section 4) appended to CHECKS
- tests: new tests/test_statics_qs.py and additions to tests/test_full_engine.py (section 4)

## Verification proposed

- verify_full check_qs_capped_pile: 3 silica grains driven into a floor/wall corner (all-DEM prep) until the floor rolling springs are at their caps (margins >= 0.999); 30 ms in three phases (constant; pressing 1 g -> 0.3 g so caps shrink and flow without motion; lateral rising to 0.9 of sliding). Engine vs all-DEM(dt) vs all-DEM(dt/2): same contacts; asleep >= 95%, qs_flow_contact_steps > 0 in phase 2; positions <= 3x DEM step error + 1e-12 m; every contact's xt, xr, xtw <= 3x step error + B_k/k_k (validates the reported bound); |k_r xr| = Frcrit(final Fne) within 1e-9 at the flowing contacts; B <= 1e-3 cap. Proves flow under a shrinking cap and elastic unloading match the full model.
- verify_full check_qs_edge_twist: one grain wedged in an edge (0.66 uN locked-in normal forces), sliding ~20%, twist pre-loaded to 0.96 of its cap, 50 ms at 1e-4 g and 0.0167 Hz. Engine must sleep at the first classification (the present engine refuses) and stay asleep; positions <= 3x step error + 1e-12 m; histories unchanged in both runs to 1e-12 relative; B <= 1e-6 cap. Proves the entry at 0.96 is safe and the full model is static there.
- verify_full check_qs_bridge: 3-grain bridge, lateral ramp taking the bridge contacts' rolling ratio 0.93 -> 1 -> flowing -> 0.5 in 30 ms; criteria as the pile check plus no wake during the cap crossing (asleep >= 95%). Proves a cap can be reached and left while asleep.
- unit tests (tests/test_statics_qs.py): sds_increments equals the kernel's history increment on an elastic contact (1e-15); one-shot vs 1000-sub-step return within e_ret for random capped rolling trials with radial+tangential increments and a changing cap; snap_to_caps makes state() reproduce lam to 1e-12 and never zero a twist; twist_quirk fires for dpsi = 0 and opposite sign over the cap, not for consistent flow; qs_step converges in <= 3 iterations on a corner pile (q < 0.01), q ~ 0.4 on an edge grain held by twist only, fails (mechanism) for a one-contact grain at its rolling cap
- unit tests (tests/test_full_engine.py): island sleeps with capped flags, flows under a reduced pressing load (|k_r xr| equals the new cap exactly, |du| < 1e-10 m, bounds > 0 and < 1e-6), s[members] = du_max > 0; each wake reason triggers; _snapshot/_restore redo leaves island state identical (copy-on-write); pickle round trip continues bit-identically (extension of test_checkpoint_with_sleeping_cluster_is_exact); an old-format island dict is upgraded and advances; _skip_asleep commits per sample and stops at a failing sample
- regression: existing verify_full checks 4, 8, 10, 11, 13 and all 98 tests pass unchanged; CHECKS.md regenerated with tools/checks_table.py (new rows with criteria and measured values)

## Risks stated

- Newton with the frozen elastic K converges slowly (q 0.4-0.9) or fails for soft clusters (edge grains held only by twist springs, near-mechanisms): churn wake -> DEM -> retry each step at the status-quo cost for those clusters; mitigation phase 2: elasto-plastic iteration matrix K_ep from sds_increments or Aitken acceleration; tracked by qs_wake_newton
- The all-DEM reference at a cap is itself noisy (its own residual ratchets, round-off clipping): verification differences are dominated by the DEM's step noise at the ~1e-5 cap level; the criteria include the step error and the reported bound, so a failure would point at the bound, not at the DEM
- Twist quirk: a capped twist with falling Fncrit is handed to the DEM, whose outcome there depends on the sign of a ~1e-20 rad/s rate (LAMMPS-conformant kernel rule); a kernel fix (sign from the spring at zero rate) is a physics change outside this design and would need the LAMMPS term-by-term check re-run
- The load-path bound uses the project's sampled-curvature practice (as Loads.accel_bound); a load oscillating inside a 4 ms step that is not a fast window would evade it; a derivative bound from the profile's analytic form would make it rigorous
- Autosave resume: continued campaign tubules switch sleep rule mid-flight; both rules are exact-or-bounded but bit-for-bit continuation holds only within one engine version, and the bound accumulators start at the resume point
- Any missed place that assumes ASLEEP => s = 0 (_early_exit, _capture_times, _watch) would leave a 1e-10 m unverified assumption; the _missed safety net still catches contact formation
- Cumulative return-map bound over the 800 s microgravity phase (2e5 samples) stays ~2e-4 rho at worst; reported, not a wake criterion, so a faster-varying scenario would simply report a larger (still explicit) bound
- Speed-up depends on the few genuinely moving grains (1e-4 m/s) being on the slaved/rolling models; if 3 grains remain in the DEM at 5-18 ns steps the coast still costs ~450 s per flight-second (15-20x better, not 1000x)
- Microgravity cost at 4 ms samples is ~0.3 s per flight-second per island (<= 1 h per tubule); the optional adaptive sample lengthening in _skip_asleep needs its own bound scaling and test
- Yielded (plastic-flattened) contacts wake the island (as _try_slaved refuses them); dormant for silica/graphite at these loads (overlaps 3-9 nm vs d_y 0.1-1 um) but untested in flight

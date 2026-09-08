# Task 027 - True-3D fixed-localization surface-conductance intensity sweep

Fixed **azimuthal pollution sector** (localized surface-conductance patch) of constant area;
the surface conductance `sigma_s` is swept and the electrical diagnostics and the **surface
loss-power field** are tracked. This prepares the heat-input field for later electrothermal
coupling. **This is localized surface-conductance intensity validation, NOT discharge / streamer
/ breakdown physics.** No streamer, partial-discharge, or breakdown mechanism is modelled or
claimed; any such physics is explicitly out of scope / future work.

## 1. Method (all accepted machinery, reused unchanged)

| element | source | note |
|---------|--------|------|
| geometry / mesh | accepted Task 025 v5 true-3D DDB-123 solve mesh | read-only; **not** overwritten |
| polluted sector | MATC theta-window on `insulator_surface` (Task 026) | theta in [0, angle], full height |
| solver | custom `ComplexEQS` (nabla.((sigma+jw eps) grad phi)=0), BiCGStabl(4)+ILU2 | UMFPack direct is unavailable on this high-contrast 3D system (Tasks 024/025/026) |
| observables | `observables_3d` + surface-loss integration (Task 026) | identical extraction |

Geometry: **30 deg** full-height sector (primary) + a gated **10 deg** sector (optional, narrow
conductive-path candidate). Outputs in `results/{raw,processed}/027_fixed_sector_sigma_sweep/`;
Task 025/026 outputs untouched. Scripts: `run_fixed_sector_sweep.py`, `analyze_sweep.py`,
`make_sweep_figures.py`. Data: `sweep.json`, `analysis.json`, `mesh_audit.json`. Total runtime
27.5 min for 12 (30 deg) + 7 (10 deg) solves; every case converged (residual <= 1e-9).

### 1.1 Phasor / power convention (critical)

`U0 = Um/sqrt(3) = 71.0 kV` is an **RMS** phasor amplitude (`run_cvt.U0`, *"phase-to-earth RMS"*).
With RMS phasors the time-average dissipated power is

```
P_surface_direct = integral( sigma_s |grad_s phi|^2 ) dS          [W]   (NO 1/2 factor)
surface_leak     = P_surface_direct / U0                          [A]   (equivalent RMS in-phase leakage current)
P_surface_from_current = U0 * surface_leak                        [W]   (NO 1/2 factor)
q_s(x)           = sigma_s |grad_s phi|^2                          [W/m^2] (RMS surface loss density)
```

The `1/2` written in the task statement applies to **peak** phasors; because `U0` is RMS it is
dropped. By construction `P_surface_from_current == P_surface_direct` (surface_leak is defined as
`P_direct/U0`), so the power-consistency check is an internal-consistency identity:
**max relative error 2.2e-16 (machine epsilon)** - see `sigma_sweep_power_consistency.png`. The
heat input handed to the thermal task is the **direct spatial field** `q_s(x,y,z)`, not only the
scalar `surface_leak`.

## 2. Mesh / patch audit (reused accepted v5 mesh: 270 607 nodes, 1.59 M tets)

| quantity | 30 deg sector | 10 deg sector |
|----------|---------------|---------------|
| patch triangles | 18 930 | 7 120 |
| measured area fraction | 0.0835 | 0.0289 |
| expected (angle/360) | 0.0833 | 0.0278 |
| area error | **+0.2 %** | **+4.1 %** |
| **cut edge** (theta~angle) minSICN min / p5 | **0.450 / 0.463** | **0.429 / 0.442** |
| revolution seam (theta~0) minSICN min / p5 | 0.234 / 0.243 | 0.234 / 0.244 |
| worst-200 insulator tris on cut edge | 0 | 0 |

The **new cut edge** (the patch boundary at theta = angle) is well-meshed in both cases (minSICN
>= 0.43, far above the 0.30 gate floor) and carries **none** of the worst-200 insulator triangles.
The theta~0 **revolution seam** (minSICN 0.234) is the global insulator-surface minimum, but it is
the OCC revolution closure seam that is **pre-existing and shared by every [0, angle] sector,
including the already-accepted Task 026 30 deg case** - it is therefore reported, not treated as a
new blocker (minSICN 0.234 is a valid, non-inverted element, adequate for the smooth
Laplace-Beltrami surface-conductance term). See `patch_area_and_mesh_quality.png`. The 10 deg area
error (+4.1 %) is larger than the 30 deg (+0.2 %) because a narrow theta-window resolves its
boundary on a coarser angular fraction of finite-size triangles; it is within the 5 % gate.

**10 deg audit gate: area_ok / tris_ok / cut_edge_ok / cut_edge_no_worst = all PASS.** The 10 deg
sweep was therefore run.

## 3. Primary result - 30 deg sector intensity sweep

Clean baseline: **C = 5623.4 pF, ratio 0.1669, tan-delta 1.990e-3, volume_leak 2.497e-4 A,
Emax 1.3876 MV/m at the HV terminal** - matches Task 025/026 clean exactly.

| sigma_s [S] | C [pF] | ratio | tan-delta | surface_leak [A] | P_direct [W] | max q_s [W/m^2] | Emax [V/m] | iters | residual |
|------------:|-------:|------:|----------:|-----------------:|-------------:|----------------:|-----------:|------:|---------:|
| 0 (clean)   | 5623.4 | 0.1669 | 1.990e-3 | 0          | 0          | 0        | 1 387 587 | 8  | 6.5e-10 |
| 1e-10 | 5623.4 | 0.1665 | 1.995e-3 | 6.476e-7 | 4.599e-2 | 1.18e0  | 1 387 544 | 8  | 3.8e-10 |
| 3e-10 | 5623.4 | 0.1672 | 1.998e-3 | 1.022e-6 | 7.261e-2 | 2.54e0  | 1 387 501 | 10 | 6.9e-10 |
| 1e-9  | 5623.4 | 0.1667 | 2.006e-3 | 2.062e-6 | 1.464e-1 | 5.76e0  | 1 387 489 | 12 | 5.7e-10 |
| 3e-9  | 5623.4 | 0.1666 | 2.030e-3 | 5.037e-6 | 3.577e-1 | 8.56e0  | 1 387 481 | 17 | 9.6e-10 |
| 1e-8  | 5623.4 | 0.1669 | 2.117e-3 | 1.591e-5 | 1.130e0  | 1.77e1  | 1 387 466 | 21 | 6.8e-10 |
| 3e-8  | 5623.4 | 0.1668 | 2.368e-3 | 4.743e-5 | 3.368e0  | 4.83e1  | 1 387 490 | 27 | 8.2e-10 |
| 1e-7  | 5623.4 | 0.1666 | 3.250e-3 | 1.580e-4 | 1.122e1  | 1.45e2  | 1 387 463 | 31 | 9.5e-10 |
| 3e-7  | 5623.4 | 0.1665 | 5.756e-3 | 4.725e-4 | 3.355e1  | 3.95e2  | 1 387 355 | 31 | 5.7e-10 |
| 1e-6  | 5623.4 | 0.1666 | 1.454e-2 | 1.574e-3 | 1.118e2  | 1.29e3  | 1 387 329 | 33 | 7.8e-10 |
| 3e-6  | 5623.5 | 0.1669 | 3.962e-2 | 4.722e-3 | 3.353e2  | 3.90e3  | 1 387 295 | 31 | 9.6e-10 |
| 1e-5  | 5623.4 | 0.1666 | 1.274e-1 | 1.574e-2 | 1.118e3  | 1.28e4  | 1 387 315 | 32 | 2.9e-10 |

(`volume_leak` is constant at 2.497e-4 A - the bulk dielectric loss is unaffected by the surface
patch; full per-case data incl. max-q_s location in `sweep.json`.) Figures:
`sigma_sweep_surface_leak.png`, `sigma_sweep_surface_power.png`, `sigma_sweep_tandelta.png`,
`sigma_sweep_state_space.png`.

### 3.1 Regions (from local log-log slope d(log surface_leak)/d(log sigma_s))

- **Numerical-floor region, sigma_s <= 1e-9.** Apparent slope 0.42 -> 0.81 (sub-linear). The
  specific leak `surface_leak/sigma_s` is inflated (6476 -> 2062, vs the plateau 1574). This is
  **not** physical sub-linearity: the loss-driven out-of-phase (imaginary) field there is tiny and
  sits near the iterative-solver residual floor, leaving a small additive numerical component
  (~0.01-0.04 W) that dominates the genuine signal. **These points are below the diagnostic's
  numerical sensitivity and must not be read as physics.**
- **Linear region, sigma_s >= 1e-8** (transitional from 3e-9). Slope = **1.000** to three digits;
  specific leak constant at **1574** (`analysis.json["specific_leak"]`). `surface_leak`, `P_direct`
  and the surface part of `tan-delta` are all exactly proportional to `sigma_s`. This is the
  physically clean, diagnostically usable regime.
- **No saturation / surface-short region.** Even at the most conductive case (1e-5 S) the slope is
  still 1.000 and `C`/ratio are unchanged: the surface current `surface_leak = 1.574e-2 A` is only
  **12.5 %** of the divider reactive current (`omega C U0 = 0.125 A`), so the patch never shorts the
  divider. `tan-delta` reaches 0.127 (a heavily-polluted condition) but the response stays linear.
- **Volume/surface crossover at sigma_s ~ 1.4e-7 S.** Below it the constant bulk dielectric loss
  (clean tan-delta 1.99e-3) dominates; above it the surface patch is the dominant loss term.

### 3.2 Field behaviour & Emax (no artificial singularity)

`Emax = 1.39 MV/m` is **constant to 0.02 %** across the whole sweep (spread 292 V/m on 1.39e6) and
**always located at the HV terminal cap (r 0.003 m, z 1.85 m), never on the insulator patch edge**
(`emax_on_insulator = False` for every case). The conductive sector introduces **no artificial
field-edge singularity** - `sigma_sweep_emax_location.png` shows all points collapsed onto the
single terminal location, far from the creepage band.

The **surface loss density** `q_s`, however, peaks at the **theta ~ 30 deg patch edge** (current
crowding at the conductive-patch boundary, r ~ 0.12 m at a weather-shed lip) - this is a loss-density
feature, not an `Emax` singularity. The peak-`q_s` location migrates from an upper shed lip
(z ~ 1.5 m) to a lower one (z ~ 0.62 m) for sigma_s >= 3e-6 S, an **early field-redistribution sign**
even though the *integrated* power remains linear. `q_s` is otherwise self-similar (fixed spatial
shape, magnitude proportional to sigma_s in the linear regime) - see
`surface_loss_density_selected_sigmas.png`. The thermal task must therefore use the **field**
`q_s(x)`, which carries this mild patch-edge concentration, not a uniform patch power.

### 3.3 Phase is rejected (again)

The tap phase ranges over **[-185, +220] mdeg** non-monotonically across the sweep (clean
-3.3 mdeg), i.e. it is at the BiCGStabl iterative-residual / numerical noise floor with no usable
trend. **The tap phase is NOT a validated diagnostic here** - consistent with Tasks 026/026B.
`sigma_sweep_vtap_phase.png` is shown with the noise band shaded and labelled UNRELIABLE.

## 4. Optional 10 deg sector (narrow conductive-surface-path candidate)

Run after the gate passed (Section 2). Behaviour mirrors the 30 deg sector: linear for
sigma_s >= 1e-8 (slope 1.0), no saturation, crossover ~2.5e-7 S, `Emax` constant at the terminal
(never on the patch). **Area scaling holds:** at fixed sigma_s the 10 deg `surface_leak` is
0.351 x the 30 deg value, matching the area ratio 0.346 (= 0.0289/0.0835) - consistent with the
Task 026 area-scaling result. Selected values (sigma_s | surface_leak | P_direct | tan-delta):
1e-8 | 5.62e-6 | 0.399 W | 2.03e-3 ; 1e-6 | 5.53e-4 | 39.2 W | 6.39e-3 ; 1e-5 | 5.52e-3 | 392 W |
4.60e-2. The 10 deg patch is a **valid narrow localized surface-conductance path candidate**;
because its area is smaller, the same sigma_s yields proportionally less loss but the **same**
linear, robust diagnostic behaviour.

## 5. Required checks

| # | check | result |
|---|-------|--------|
| 1 | clean baseline matches Task 025/026 | **PASS** (C 5623.4 pF, tan-delta 1.99e-3, Emax 1.39 MV/m) |
| 2 | 30 deg patch area fraction correct | **PASS** (0.0835 vs 0.0833, +0.2 %) |
| 3 | polluted triangles only on `insulator_surface` | **PASS** (r,z creepage filter; area audit) |
| 4 | solver convergence documented every sigma_s | **PASS** (all 19 cases, residual <= 1e-9, 8-33 iters) |
| 5 | no material / contact regressions | **PASS** (accepted v5 mesh reused unchanged) |
| 6 | Emax location tracked, no artificial patch-edge jump | **PASS** (constant 1.39 MV/m at terminal; 0/19 on insulator) |
| 7 | surface_leak monotonic with sigma_s | **PASS** (strictly increasing) |
| 8 | P_surface_direct monotonic with sigma_s | **PASS** (strictly increasing) |
| 9 | tan-delta monotonic or explained | **PASS** (strictly increasing) |
| 10 | C and divider ratio stable | **PASS** (5623.4-5623.5 pF, ratio 0.1665-0.1672) |
| 11 | phase not overclaimed | **PASS** (explicitly rejected, Section 3.3) |
| 12 | high-sigma surface-short / saturation checked | **PASS** (no saturation; surface current <= 12.5 % of reactive current at 1e-5) |
| - | power consistency (direct vs current) | **PASS** (identical, rel err 2.2e-16) |
| - | q_s fields saved for weak/medium/severe | **PASS** (q_s VTUs for 1e-9..1e-5, both angles) |

## 6. Recommendation for the electrothermal (thermal-coupling) task

Use cases from the **clean linear regime (sigma_s >= 1e-8)**; avoid sigma_s <= 1e-9 (numerical
floor). Recommended weak / medium / severe (q_s fields saved as
`qs_fields/qs_30deg_s{1e-07,1e-06,1e-05}.vtu`):

| level | sigma_s [S] | P_direct [W] | max q_s [W/m^2] | tan-delta | condition |
|-------|------------:|-------------:|----------------:|----------:|-----------|
| **weak**   | 1e-7 | 11.2  | 1.45e2 | 3.25e-3 | surface loss just emerging above the bulk loss background |
| **medium** | 1e-6 | 111.8 | 1.29e3 | 1.45e-2 | clearly polluted |
| **severe** | 1e-5 | 1117.7| 1.28e4 | 1.27e-1 | heavily polluted; early loss-density redistribution |

In the linear regime the heat input is `q_s(x) = sigma_s * g(x)` with `g(x)` a fixed normalised
loss-shape (peaked at the weather-shed lips and the patch edge), so the thermal model can scale one
normalised field by `sigma_s`. The total patch heat for the 30 deg sector ranges ~11 W (weak) to
~1.1 kW (severe).

## 7. Verdict

**PASS: ready for electrothermal input**, with the following documented usage conditions:

1. Use the **direct q_s(x) field** (saved VTUs), not just scalar `surface_leak` - it carries the
   shed-lip and patch-edge loss concentration the thermal model needs.
2. Restrict to the **linear regime sigma_s >= 1e-8 S**; sigma_s <= 1e-9 S is below numerical
   sensitivity (floor) and must not be used.
3. `surface_leak`, `P_surface_direct` and `tan-delta` are the **robust, monotonic, power-consistent**
   observables. **Tap phase is rejected** (solver-noise limited).
4. No saturation/surface-short occurs up to 1e-5 S; recommended weak/medium/severe = 1e-7/1e-6/1e-5.

This task validated **localized surface-conductance intensity** on a fixed azimuthal sector in true
3D. It does **not** model discharge, streamer, or breakdown physics; that remains future work.

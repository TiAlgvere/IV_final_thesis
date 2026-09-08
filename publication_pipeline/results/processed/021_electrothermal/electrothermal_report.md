# Task 021 - One-Way Electrothermal Multiphysics (diagnosis -> risk)

## Purpose

Add the next layer above the diagnostic matrix: take the electrical loss from the
verified ComplexEQS solve, use it as a heat source, solve the steady temperature
rise, and decide **which fault signatures correspond to real thermal risk** (severity),
not just electrical detectability. One-way coupling only (EQS -> thermal).

## Method & model

- **Heat source** reuses the EQS loss integrands directly, so it is energy-consistent
  with the diagnostics: volumetric q''' = 0.5*sigma_eff*|E|^2 in the dielectric
  (p_vol = 0.5*volume_leak*U0) and a creepage SURFACE source
  0.5*sigma_s*|grad_s phi|^2 for pollution (p_surf = 0.5*surface_leak*U0).
- **Thermal solver**: self-contained axisymmetric P1 steady heat conduction
  (`thermal_solver.py`), -div(k grad T)=q''' on the SOLID sub-domain, with the air
  region replaced by a convective boundary. **Verified** against the analytic
  uniformly-heated cylinder (center dT 33.34 vs 33.33 K, 0.03 % error).
- (Numerical) capped-sigma metals carry no physical loss, so their volumetric heat is
  zeroed; only dielectric/surface losses heat the device.

## Documented thermal assumptions (revise with project data)

- ambient T_amb = 40 C (hot-day operating);
- natural convection h = 10 W/m2K on the solid-air surface;
- thermal conductivity k [W/mK]: metal=200.0, element=0.25, porcelain=1.5, oil=0.13;
- insulation hot-spot limit 105 C (oil/paper, IEC-class guidance).
- **The relative ranking of faults is robust; absolute dT scales with these assumptions** (especially h and the oil conductivity).

## Results (12 cases)

| case | family | p_vol [W] | p_surf [W] | dT_max [K] | T_abs [C] | hot-spot | risk |
|------|--------|-----------|------------|------------|-----------|----------|------|
| healthy | healthy | 8.9 | 0.0 | 4.9 | 44.9 | element_5 | minor |
| C1_loss_0.005 | C1_loss | 19.9 | 0.0 | 12.3 | 52.3 | element_5 | elevated |
| C1_loss_0.01 | C1_loss | 38.4 | 0.0 | 24.5 | 64.5 | element_5 | elevated |
| C1_loss_0.02 | C1_loss | 75.3 | 0.0 | 49.0 | 89.0 | element_5 | high |
| C1_loss_0.05 | C1_loss | 186.1 | 0.0 | 122.5 | 162.5 | element_5 | critical (>hot-spot limit) |
| C1_loss_0.1 | C1_loss | 370.6 | 0.0 | 245.0 | 285.0 | element_5 | critical (>hot-spot limit) |
| C2_loss_0.05 | C2_loss | 44.2 | 0.0 | 90.5 | 130.5 | element_11 | critical (>hot-spot limit) |
| pollution_1e-08 | pollution | 8.9 | 7.3 | 5.2 | 45.2 | element_4 | minor |
| pollution_1e-07 | pollution | 8.9 | 71.6 | 7.9 | 47.9 | element_4 | minor |
| pollution_1e-06 | pollution | 8.9 | 715.0 | 34.5 | 74.5 | element_6 | high |
| C1_cap_20 | C1_cap | 10.3 | 0.0 | 5.9 | 45.9 | element_11 | minor |
| disc_short | disc_short | 9.7 | 0.0 | 5.8 | 45.8 | element_9 | minor |

## Diagnosis -> thermal risk (the key finding)

The diagnostic families split cleanly by thermal risk:

- **Capacitance / disc-short (ratio faults): electrically loud, thermally SILENT.**
  They add ~no loss (dT ~ 4.9 K, ~healthy) - a metrology/ratio fault,
  not a thermal hazard. Detect and schedule, but no thermal urgency.
- **Internal dielectric loss: electrically + thermally CRITICAL.** dT_max scales ~linearly
  with tan-delta and crosses the insulation hot-spot limit by tan-delta ~0.05
  (T_abs > 105 C). Because real tan-delta itself rises with temperature, this is the
  one-way precursor of thermal runaway - the highest-severity family. The hot-spot
  localises to the faulted section (C1 -> mid-stack element_5; C2 -> element_11).
- **C2 loss is more severe per watt than C1 loss**: the C2 loss concentrates in only 2
  elements, so a smaller total power (44 W) still produces a higher local hot-spot
  (90 K) than a larger but more distributed C1 loss.
- **Surface pollution: thermally relevant only when heavy.** Light films (sigma_s <=1e-7)
  are an electrical signal but thermally minor; heavy wetting (sigma_s 1e-6, ~720 W)
  reaches 'high' risk (creepage ~74 C) - the dry-band / flashover precursor. The
  surface heat is deposited at the cooled boundary, so much convects away; a localised
  real dry band (not modelled) would concentrate it further.

## How this complements the diagnostic matrix

The Task 17-20 matrix answers *what* the fault is and *whether* it is detectable; this
layer answers *how urgent* it is. Combined rule:
- ratio fault (Vtap) -> identify + trend, low thermal urgency;
- dielectric loss (phase + tan-delta) -> thermal hazard, escalating -> prioritise;
- surface leakage (pollution) -> thermal hazard only once leakage/power is large.

## Caveats

- One-way coupling only: no temperature feedback on tan-delta/conductivity (so the
  real loss-fault case would run hotter - thermal runaway is under-estimated, not over).
- Steady-state only (no thermal time constant / transient).
- Uniform pollution + uniform convection; a localised dry band would give sharper local
  heating than the smeared surface source here.
- Absolute temperatures depend on the documented thermal properties and h; these are
  **model-based risk estimates for the representative DDB-123, not field-validated**.

Figures: thermal_field_maps.png, thermal_risk_curve.png, diagnosis_to_risk.png. Data: thermal_summary.csv, Tfield_*.npz.

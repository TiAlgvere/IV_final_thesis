# Task 019 - Internal Fault Severity Sweeps & Trajectory Refinement

## Purpose

Refine the Task 016/018 internal fault manifolds into smooth severity sweeps on the
representative rounded DDB-123 baseline, in the Task 018 observable state vector

    x = [ d|Vtap| %, d-theta mdeg, tan-delta, surface_leak A, volume_leak A ]

reported relative to the healthy baseline. Five families are swept: C1/C2 capacitance
increase, C1/C2 dielectric loss, and disc short (location + partial-short severity).

## Model assumptions

- Frequency-domain axisymmetric ComplexEQS (50 Hz); no multiphysics, no solver change.
- C1 = 10 homogenised stack elements, C2 = 2 elements; tap between them (foil 10).
- Capacitance fault = scale the C1- or C2-element `eps_r_eff` (loss held at baseline).
- Dielectric-loss fault = raise the element `tan-delta` (capacitance held).
- Disc short = one element keeps `eps_r_eff` but develops a leakage conductance
  `sigma`; `sigma=1 S/m` is effectively equipotential (full short), smaller `sigma`
  is a partial / leaky short.
- Floating metals use the capped `sigma=1 S/m` (no spurious LU loss).

## Healthy baseline

- |Vtap| = 11829.52 V   (tap ratio 0.1666, ideal n_c2/(n_c1+n_c2) = 0.1667)
- phase = -0.12 mdeg (numerical floor)
- terminal tan-delta = 2.000e-03
- surface leakage = 0.00e+00 A   (zero - clean)
- volume loss current = 2.505e-04 A

## Sweep definitions

| family | parameter | values |
|--------|-----------|--------|
| C1_cap / C2_cap | capacitance increase [%] | 0.1, 0.3, 1, 3, 10, 20 |
| C1_loss / C2_loss | element tan-delta | 0.002 (base), 0.005, 0.01, 0.02, 0.05, 0.10 |
| disc_short (location) | full short element | C1_HV, C1_mid, C1_tap, C2_tap, C2_gnd |
| disc_short (severity) | short conductance sigma [S/m] | 1e-4, 1e-3, 1e-2, 1e-1, 1 |

## Monotonicity checks

| check | monotonic? | end value |
|-------|-----------|-----------|
| C1_cap dvtap up | PASS | +16.13 |
| C2_cap dvtap down | PASS | -14.28 |
| C1_loss tand up | PASS | +0.0832 |
| C2_loss tand up | PASS | +0.01811 |
| C1_loss vol up | PASS | +0.01044 |
| C2_loss vol up | PASS | +0.002272 |
| C1_loss phase down | PASS | -4661 |
| C2_loss phase up | PASS | +4666 |

surface_leak == 0 for ALL internal faults: **PASS** (internal faults produce no creepage current - the external/internal split holds).

## Disc-short analytic consistency

Full short of one stack element -> ideal series-divider ratio shift
(`n_c2/(n_c1+n_c2)` with the shorted unit removed). C1 shorts: 10->9 units -> +9.09 %;
C2 shorts: 2->1 unit -> -45.45 %.

| location | FE d|Vtap| [%] | analytic [%] | |error| |
|----------|----------------|--------------|---------|
| C1_HV | +9.09 | +9.09 | 0.001 |
| C1_mid | +9.09 | +9.09 | 0.001 |
| C1_tap | +9.10 | +9.09 | 0.013 |
| C2_tap | -45.45 | -45.45 | 0.003 |
| C2_gnd | -45.44 | -45.45 | 0.008 |

FE matches the lumped model to <0.02 %-point, and is location-independent within C1
and within C2 - confirming the homogenised stack behaves as an ideal series divider
for ratio faults. (C2 shorts shift Vtap ~5x more than C1 shorts because C2 has only
2 series units, so removing one is a far larger fractional change.)

## Detection limits implied by the sweeps

Using the Task-018 trend-monitoring thresholds (d|Vtap| >= 0.05 %, d-theta >= 1 mdeg,
d tan-delta >= 1e-4, d volume-leak >= 1e-5 A). These are *assumed* monitoring
resolutions, **not** instrument or standard accuracy classes.

| family | min tested severity detected | via | note |
|--------|------------------------------|-----|------|
| C1_cap | 0.1 (cap_increase_pct) | dvtap | floor below grid; linear extrap d|Vtap|=0.05% at ~+0.06% cap |
| C2_cap | 0.1 (cap_increase_pct) | dvtap | floor below grid; linear extrap ~+0.07% cap |
| C1_loss | 0.005 (tan_delta) | dtheta, tand, vol_leak | model phase floor (1 mdeg) ~tan-delta 0.0020; practical 1-arc-min floor ~tan-delta 0.0024 |
| C2_loss | 0.005 (tan_delta) | dtheta, tand, vol_leak | phase-dominated, same as C1_loss with opposite sign |
| disc_short | 0.0001 (short_sigma) | dvtap, dtheta, tand, vol_leak | any conductive path is strongly detectable (Vtap + phase) |

All five internal families are detectable at (or below) the smallest tested step, so
the practical limit is set by the monitoring resolution, not by the physics:

- **Capacitance**: d|Vtap| is linear in capacitance change (~0.83 %Vtap per %cap for
  both C1 and C2), so a 0.05 % Vtap resolution implies a ~**+0.06 % capacitance** floor.
- **Dielectric loss**: the phase displacement is extraordinarily sensitive in the model
  (143 mdeg already at tan-delta 0.005), but real metering resolves ~arc-minutes
  (1 arc-min = 16.7 mdeg); at a ~10 arc-min practical floor, loss is detectable around
  **tan-delta ~ 0.005-0.01**. Terminal tan-delta and volume-leak corroborate.
- **Disc short**: a full short is a huge signature (+9 % / -45 % Vtap); even an incipient
  partial short (sigma 1e-4) already shows several-% Vtap and thousands of mdeg phase.

## Does the Task-018 classifier still separate the families?

Yes, with one refinement and one documented degeneracy:

1. **Ratio faults (C1_cap, C2_cap, full short)** - identified by |d|Vtap||, sign = side
   (C1 up / C2 down). surface_leak = 0, tan-delta ~ baseline. Cleanly separated.
2. **Dielectric loss (C1_loss, C2_loss)** - degree-scale phase with sign (C1 negative,
   C2 positive), terminal tan-delta and volume-leak up, surface_leak = 0, |d|Vtap|| ~ 0.
   Cleanly separated from pollution by **surface_leak = 0** (volume not surface current).
3. **External pollution** - the only family with surface_leak >> 0. Orthogonal to all
   internal families in the leakage-split panel.

**Refinement:** the classifier's hard `|d|Vtap|| > 3 %` gate is a *coarse* trigger - it
misses small capacitance drift (<3 %), which is nonetheless detectable by Vtap **trend**
monitoring down to ~0.06 %. For incipient capacitance ageing, use the trend, not the gate.

## Overlaps / blind spots

- **Partial short vs C1 dielectric loss (degeneracy).** A leaky/partial short at low
  conductance (sigma <= 1e-3) is signature-identical to a distributed C1 loss fault:
  large negative phase, raised tan-delta and volume-leak, small Vtap shift. Only as the
  short hardens (sigma -> 1) does its Vtap shift grow to the full +9.09 % and reveal it as
  a ratio fault. The disc-short trajectory therefore *bridges* the loss manifold and the
  ratio manifold - incipient shorts and dielectric loss are not separable from terminal
  observables alone until the short progresses.
- **Small capacitance drift (<3 %)** sits below the classifier gate (though above the
  trend-detection floor) - it looks 'nominal' to the rule unless trended.
- The C1/C2 *cap* and C1/C2 *loss* families are otherwise well separated (Vtap sign,
  phase sign, surface vs volume leakage).

## Caveat

These are **model-based diagnostic trajectories for the representative DDB-123 model**,
not field-validated operational thresholds. Absolute detection limits depend on the real
instrument/metering class and on temperature drift (a known confounder, not yet modelled).

Figures: `internal_sweep_vtap.png`, `internal_sweep_phase_loss.png`,
`internal_trajectory_matrix.png`, `diagnostic_matrix_refined.png`. Data:
`internal_fault_sweeps.csv`.

# Task 020 - Operational Noise & Detectability (temperature, frequency, burden)

## Purpose

Estimate the healthy operating-noise envelope of the representative DDB-123 and
determine which simulated fault signatures (Tasks 015/018/019) remain detectable
once normal temperature, grid-frequency and secondary-burden drift are included.
No new fault physics; no solver change. State vector x = [d|Vtap| %, d-theta mdeg, tan-delta, surface_leak A, volume_leak A].

## Assumptions (documented, revise with project data)

**Temperature** (T = -30..60 C, ref 20 C), applied *uniformly* to the stack:
- permittivity temp-coeff alpha_eps = 200 ppm/K (capacitor dielectric).
- loss-tangent factor f_tand(T) = exp(0.0173*(T-20)) (~doubles per 40 K rise).
- humidity/surface conductance kept ZERO in the healthy case (no wet pollution).

**Frequency** (f = 49.8..50.2 Hz): folded consistently into the SIF frequency and
the loss sigma_eff. Verified: |Vtap|, phase, tan-delta are invariant to 5 sig figs;
volume_leak scales linearly with f (50.2/50.0 -> +0.40 %). So frequency only moves
volume_leak.

**Secondary burden / measurement** - treated as an INSTRUMENTATION confounder, NOT
part of the EQS field solve:
- ratio (|Vtap|) error std 0.05 % (+-0.1 % ~ 2 sigma);
- phase error std 8.3 mdeg (+-1 arc-min ~ 2 sigma);
- tan-delta resolution std 1e-04;  surface-leak floor std 1e-07 A;
- volume_leak relative measurement noise 2 %.

## Key field result (why the envelope has this shape)

A capacitive divider ratio is first-order invariant to *uniform* temperature (C1 and
C2 scale together) and exactly invariant to frequency. The healthy sweep confirms it:
across -30..60 C, d|Vtap| stays within +-0.0006 % and phase within +-0.05 mdeg, while
tan-delta swings 8.4e-4 -> 4.0e-3 (x4.7) and volume_leak tracks it. **Therefore the
Vtap/phase healthy envelope is set by the burden/measurement noise, and the
tan-delta/volume_leak envelope by temperature.**

## Healthy noise envelope (Monte Carlo, N=20000)

| observable | mean | std | min | max |
|------------|------|-----|-----|-----|
| dvtap_pct | -0.000 | +0.050 | -0.190 | +0.192 |
| dtheta_mdeg | -0.11 | +8.30 | -31.08 | +30.59 |
| tan_delta | 2.043e-03 | 9.125e-04 | 4.923e-04 | 4.322e-03 |
| surface_leak | 7.99e-08 | 6.06e-08 | 5.53e-12 | 3.96e-07 |
| volume_leak | 2.564e-04 | 1.152e-04 | 9.801e-05 | 5.268e-04 |

Correlation matrix (healthy drift):

| | dvtap_pct | dtheta_mdeg | tan_delta | surface_leak | volume_leak |
|---|---|---|---|---|---|
| dvtap_pct | +1.00 | +0.01 | +0.01 | +0.00 | +0.01 |
| dtheta_mdeg | +0.01 | +1.00 | -0.01 | -0.00 | -0.00 |
| tan_delta | +0.01 | -0.01 | +1.00 | -0.01 | +0.99 |
| surface_leak | +0.00 | -0.00 | -0.01 | +1.00 | -0.01 |
| volume_leak | +0.01 | -0.00 | +0.99 | -0.01 | +1.00 |

The cloud is near-diagonal except a strong tan-delta / volume_leak correlation
(rho = +0.99) - both are temperature-driven. d|Vtap|, phase and surface_leak
are mutually independent instrument-noise dimensions. surface_leak has essentially
zero healthy spread (clean insulator), so ANY surface current is anomalous.

## Detectability: three levels

1. **model-detectable** - change above the numerical floor (Tasks 18/19).
2. **instrument-detectable** - the dominant single observable exceeds its measurement
   resolution (~2 sigma).
3. **operationally distinguishable** - the full 5-D state exits the 95 % healthy noise
   envelope (Mahalanobis distance d_M > 3.33, chi2 5 dof).

| family | model | instrument | operational (95% env.) | dominant |
|--------|-------|------------|------------------------|----------|
| C1 capacitance | +0.06 % | +0.12 % | +0.20 % | |Vtap| |
| C2 capacitance | +0.06 % | +0.12 % | +0.20 % | |Vtap| |
| C1 diel. loss | tand 0.0021 | tand 0.0023 | tand 0.0026 | phase |
| C2 diel. loss | tand 0.0021 | tand 0.0023 | tand 0.0026 | phase |
| full-creepage pollution | sigma_s ~1e-9 S | sigma_s ~1e-9 S | sigma_s <= 1e-09 S | surface_leak |
| localized pollution | patch-dependent | top/bottom only | top/bottom: yes; MID: no | surface_leak |
| disc short | any path | trivial (|Vtap|+phase) | sigma <= 1e-04 S/m | |Vtap| / phase |

First severity that exits the 95 % envelope (from the trajectories):
- C1_cap: +0.3 % ; C2_cap: +0.3 %
- C1_loss: tan-delta 0.005 ; C2_loss: tan-delta 0.005 (smallest tested step already exits, via phase)
- full-creepage pollution: sigma_s = 1e-09 S (smallest tested - exits at once)
- disc short (partial): sigma = 1e-04 S/m (smallest tested - exits at once)

## Localized pollution vs the envelope (Mahalanobis distance)

| patch | surface_leak [A] | d_M | exits 95% envelope? |
|-------|------------------|-----|---------------------|
| full | 2.01e-02 | 332093.29 | YES |
| top25 | 2.54e-06 | 40.52 | YES |
| mid25 | 2.40e-07 | 2.68 | no (inside healthy cloud) |
| bottom25 | 1.69e-06 | 26.88 | YES |
| streamer | 1.12e-01 | 1844933.14 | YES |

## Ambiguous cases / blind spots

- **Mid-creepage non-bridging patch (mid25): the operational blind spot.** Its surface
  leakage (~2.4e-7 A) is barely above the leakage noise floor and its phase shift is
  sub-mdeg, so d_M ~ 2.68 < 3.33 - it stays inside
  the healthy cloud. A localized dry-band mid the creepage is not operationally
  distinguishable from healthy drift here.
- **Small capacitance drift (< ~0.2 %)** is swamped by the +-0.1 % burden ratio noise -
  detectable only by long-term Vtap TREND, not a single reading.
- **Small dielectric loss seen via tan-delta is confounded by temperature** (the
  healthy tan-delta band 8e-4..4e-3 overlaps an incipient loss fault). The PHASE shift
  is the clean discriminator: uniform temperature produces ~0 phase, but a loss fault
  produces a sign-locked degree-scale phase (C1 negative, C2 positive).
- **Incipient (low-sigma) partial short still aliases C1 dielectric loss** (Task 019
  degeneracy) - both exit the envelope via phase, but neither Vtap nor the leakage split
  separates them until the short hardens.

## Recommendations - which measurements matter most

1. **Surface-leakage current monitor** - zero healthy variance makes it the definitive
   external-pollution detector (any reading >> noise floor = creepage film/bridge).
   Its blind spot is the isolated mid-creepage dry band.
2. **Phase (ratio-error angle) monitor** - immune to uniform temperature and frequency,
   so it cleanly flags internal dielectric loss (and its C1/C2 sign locates it). Needs
   ~arc-minute resolution.
3. **|Vtap| / ratio trend** - the ratio-fault (capacitance / short) detector; use TREND
   (not a single reading) to beat the burden noise down to ~+0.2 %.
4. **Terminal tan-delta alone is weak** - thermally confounded; use it only together
   with phase and with temperature compensation.

## Caveat

These are **model-based operational detectability estimates for the representative
DDB-123 model**, not field-validated thresholds. The temperature/burden coefficients are
documented assumptions; final thresholds require the actual instrument accuracy class,
real temperature-drift data, and field measurements. No field validation is claimed.

Figures: healthy_noise_envelope.png, detectability_overlay.png, detection_thresholds.png,
normalized_distance.png. Healthy data: healthy_field.json.

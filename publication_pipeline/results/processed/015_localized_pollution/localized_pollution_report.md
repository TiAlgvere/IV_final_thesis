# Task 015 - Localized Pollution / Streamer (surface-conductance windowing)

Base sigma_s = 1e-06 S (cases 1-4). Streamer = full-height ring with EFFECTIVE
sigma_s = 5.56e-06 S (local 1e-04 S x 20 deg / 360;
a true azimuthal streak needs the 3-D model). Clean tap phase = -0.12 mdeg; d-theta is relative to clean.

| case | sigma_s [S] | |Vtap| [V] | d-theta [mdeg] | tan(delta) | leakage [A] | Emax [V/m] | surf Emax [V/m] |
|------|-------------|-----------|----------------|------------|-------------|------------|------------------|
| full | 1.00e-06 | 11836.7 | -2.09 | 1.627e-01 | 2.039e-02 | 7.34e+05 | 2.67e+04 |
| top25 | 1.00e-06 | 11839.1 | -0.85 | 2.019e-03 | 2.532e-04 | 1.33e+06 | 8.10e+02 |
| mid25 | 1.00e-06 | 11832.1 | -0.12 | 2.002e-03 | 2.509e-04 | 7.44e+05 | 2.22e+02 |
| bottom25 | 1.00e-06 | 11805.0 | 2.46 | 2.013e-03 | 2.524e-04 | 7.47e+05 | 6.12e+02 |
| streamer | 5.56e-06 | 11836.8 | -0.33 | 8.946e-01 | 1.121e-01 | 7.34e+05 | 2.67e+04 |

**Worst case (max leakage): `streamer`** - leakage 1.12e-01 A, tan(delta) 8.946e-01, d-theta -0.33 mdeg.

## Read-out
- tan(delta) and leakage scale with the polluted AREA: full > each 25% window; a
  given 25% window gives ~1/4 of the full-creepage loss at the same sigma_s.
- Position matters for d-theta (phase): top/middle/bottom quarters couple to the
  internal tap differently even at equal sigma_s and equal area.
- Global Emax stays at the HV terminal (metal) - surface pollution does not move it;
  the surface field concentration (surf Emax) rises at the polluted window edges.
- In state space the localized cases sit on/near the full-creepage manifold but at
  reduced loss (smaller polluted area) - see `localized_state_space.png`.

## Artifacts
- `localized_state_space.png`, `localized_cases.csv`
- worst-case 3D: `worst_Imphi.png`, `worst_dE.png`, `worst_surface_loss.png`

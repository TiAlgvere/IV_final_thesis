# Task 008 - Pollution-Model Geometry Sanity Audit

Same complex-EQS pollution model as Task 006, no new physics and no new fault
types. Only the pollution-strip GEOMETRY is varied (thickness, coverage,
position). For each variant the conductivity sweep is re-run and the three
qualitative signatures are checked:

- **S1 phase shifts early**: |d-theta| reaches 1 deg at a sigma <= where
  |Vtap| has shifted 1 %.
- **S2 tan(delta) rises**: peak tan(delta) >= 1e-3 (clean state is exactly 0; CVT
  tan-delta monitoring resolves ~1e-4, so 1e-3 is comfortably measurable).
- **S3 large-sigma saturates/relaxes**: |d-theta| peaks at an interior sigma and
  |Vtap| saturates at the high-sigma end.

**Result: signatures hold for 8/8 geometry variants.**

Clean (sigma=0) |Vtap| is ~29117.6 V for every variant (the strip is dielectrically identical to porcelain at sigma=0), confirming the sigma=0 == clean property is geometry-independent.

| variant | group | phase peak [deg] @ sigma | tan(delta) peak | |Vtap| change [%] | sigma(phase>=1deg) | sigma(|dVtap|>=1%) | S1 | S2 | S3 | all |
|---|---|---|---|---|---|---|:--:|:--:|:--:|:--:|
| t=5mm cov=100% | thickness | 28.26 @ 5.9e-07 | 6.646e-02 | -65.3 | 1.4e-08 | 4.9e-08 | Y | Y | Y | Y |
| t=2mm cov=100% | thickness | 24.16 @ 2.0e-06 | 7.155e-02 | -61.5 | 4.9e-08 | 1.7e-07 | Y | Y | Y | Y |
| t=10mm cov=100% | thickness | 27.25 @ 5.9e-07 | 8.144e-02 | -69.0 | 4.1e-09 | 4.9e-08 | Y | Y | Y | Y |
| cov=25% bottom | coverage | 1.32 @ 1.7e-07 | 7.936e-03 | -4.6 | 1.7e-07 | 1.7e-07 | Y | Y | Y | Y |
| cov=50% bottom | coverage | 4.61 @ 1.7e-07 | 2.671e-02 | -16.1 | 4.9e-08 | 1.7e-07 | Y | Y | Y | Y |
| pos=bottom cov=40% | position | 3.18 @ 1.7e-07 | 1.869e-02 | -10.8 | 4.9e-08 | 1.7e-07 | Y | Y | Y | Y |
| pos=middle cov=40% | position | 1.78 @ 1.7e-07 | 1.038e-02 | -6.5 | 4.9e-08 | 4.9e-08 | Y | Y | Y | Y |
| pos=top cov=40% | position | 2.88 @ 1.7e-07 | 8.033e-03 | -11.1 | 1.4e-08 | 4.9e-08 | Y | Y | Y | Y |

## Interpretation

S1 (phase is the earliest indicator), S2 (loss rises) and S3 (interior
relaxation peak + |Vtap| saturation) are reported per variant above. Where all
three hold across thickness, coverage and position changes, the signatures are a
property of the conductive-pollution physics rather than the specific toy strip.
Absolute magnitudes (the |Vtap| change, the tan(delta) peak height, the transition
sigma) DO scale with geometry - smaller / thinner / less-covering strips perturb
the divider less - which is exactly the expected behaviour. The audit only claims
the qualitative SHAPE is robust: phase leads, tan(delta) rises to an interior peak,
and the high-sigma state relaxes while |Vtap| saturates. Any variant with an 'N' is
reported here, not hidden.

## Artifacts
- `audit_comparison.png` - 3x3 grid (rows: thickness / coverage / position; cols: |d-theta|, tan(delta), |Vtap| vs sigma)
- `sweep_<variant>.csv` - per-variant sweep data
- raw Elmer runs under `results/raw/008_pollution_geometry_audit/`

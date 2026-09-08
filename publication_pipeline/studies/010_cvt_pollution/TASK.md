# Task 010 - Pollution Model on the Representative DDB-123 CVT

## Goal
Apply the verified Tasks 006-008 pollution model to the validated Task 009
axisymmetric DDB-123, and test whether the diagnostic signature (phase
relaxation / state-space trajectory) survives in the realistic geometry.

## What was built
- `cvt_ddb123.py` gained an optional thin conductive `pollution_layer` skin on
  the porcelain-wall outer edge (eps_r = porcelain so sigma=0 == clean) - the
  same thin-conductive-volume-layer model verified in Tasks 006-008, now on the
  realistic creepage surface.
- `cvt_observables.py` extended to terminal tan(delta) + leakage (axisymmetric
  2*pi*r energy integrals, per-body sigma map).
- `run_cvt_pollution_sweep.py` (sigma sweep) + `analyze_cvt_pollution.py`
  (observables, state-space trajectory, comparison, report).

## Acceptance (all met)
1. Clean / sigma=0 terminal **C = 5624.0 pF** (= Task 009 clean). PASS
2. sigma=0 pollution-layer case **identical to clean** (same C, phase ~0). PASS
3. Small pollution sweep ran (16 sigma points). PASS
4. Trends vs Tasks 006-008: **tan(delta) monotonic rise** (same), **|d-theta|
   non-monotonic with an interior critical point** (same bend/relaxation).
   |Vtap| ~flat (~0.06%) and C ~flat - external pollution couples weakly to the
   internal divider, so the signal is millidegree-scale.
5. **State-space (log tan-delta, d-theta) trajectory SURVIVES**: dives to a
   critical corner (sigma~1e-5 S/m, d(d-theta)/d ln sigma = 0) then recovers -
   the ctfem Phase-9 signature, here reproduced by the verified Elmer solver on
   the validated geometry (and the negative-trough sign matches Phase-9's
   independent scikit-fem surface-sigma result).

## Thesis-grade chain now complete
- verified solver (Task 007 + axisymmetric coax re-verification),
- validated geometry (Task 009: 5624 pF, cross-checked vs scikit-fem to 5 sig figs),
- geometry-robust pollution signature (Task 008 + this realistic-geometry result),
- state-space trajectory (survives).

## Caveats / future
Realistic external-pollution signal is ~mdeg / sub-0.1% |Vtap| (weak coupling);
the phase extremum is a trough (negative) vs the toy's peak (positive) - shared
relaxation structure, geometry/coupling-dependent sign & scale. A true
surface-conductance term (vs volume-layer approximation) and shed-resolved
pollution are future refinements.

## How to run
```powershell
python publication_pipeline/scripts/run_cvt_pollution_sweep.py
python publication_pipeline/scripts/analyze_cvt_pollution.py
```
Outputs under `results/{raw,processed}/010_cvt_pollution/`.

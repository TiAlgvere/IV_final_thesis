# Task 010 - Pollution on the Representative DDB-123 CVT

The verified Tasks 006-008 pollution model (thin conductive volume layer,
eps_r = porcelain so sigma=0 == clean) applied to the validated Task 009
axisymmetric DDB-123, swept with the axisymmetric ComplexEQS solver. No new
physics. Pollution sits on the porcelain-wall creepage surface (external),
while the divider is the internal capacitor stack - so coupling is weak and
the diagnostic signal is at the millidegree scale (as in the ctfem Phase-9
scikit-fem study of the same device).

## Acceptance
1. Clean / sigma=0 terminal C = **5624.0 pF** (Task 009 clean 5624 pF). PASS
2. sigma=0 pollution-layer case == clean (identical C, phase -0.12 mdeg). PASS
3. Small pollution sweep ran (16 sigma points, 0 plus logspace 1e-10..1e-3). PASS

## Trends vs sigma (compare to Tasks 006-008)
- terminal C: ~flat at 5624 pF (external pollution barely changes terminal C).
- |Vtap|: 11829.2 -> 11835.9 V (~0.06% - weak, internal divider).
- **tan(delta): monotonic rise** 2.00e-03 -> 1.19e+00 (same as toy).
- **d-theta: NON-monotonic** - dives to -17.19 mdeg at sigma=1.0e-05 S/m then recovers
  (same bend/relaxation as the toy; here negative-going, matching Phase-9's
  scikit-fem surface-sigma result on this geometry).

## State-space trajectory (#5)
- tan(delta) monotonic: True
- d-theta has an interior critical point (max |d-theta|): True at sigma=1.0e-05 S/m
- **State-space (loss, phase) bend/relaxation SURVIVES: True**

The (log10 tan-delta, d-theta) trajectory shows the same characteristic bend with
a critical point (d(d-theta)/d ln sigma = 0) separating the capacitive-polluting
branch from the resistive-short branch - the Phase-9 signature, reproduced by the
verified Elmer solver on the validated geometry.

## Verdict
The four thesis-grade pillars are in place:
- verified solver (Task 007 + axisymmetric coax re-verification),
- validated geometry (Task 009, 5624 pF, cross-checked vs scikit-fem to 5 sig figs),
- geometry-robust pollution signature (Task 008 + this realistic-geometry result),
- state-space trajectory (survives here, mdeg scale).

Caveats (honest): the realistic external-pollution signal is ~mdeg / sub-0.1%
|Vtap| (weak coupling); the phase extremum is a trough (negative) vs the toy's
peak (positive) - the relaxation structure is shared, the sign/scale are geometry-
and coupling-dependent. A true surface-conductance term (vs the volume-layer
approximation) and shed-resolved pollution remain future refinements.

## Artifacts
- `observables_vs_sigma.png`, `state_space_trajectory.png`, `cvt_pollution_sweep.csv`

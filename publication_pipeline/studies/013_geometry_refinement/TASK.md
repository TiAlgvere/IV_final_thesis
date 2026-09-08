# Task 013 - Geometry Refinement + Surface-Pollution Model

## Goals (all met)
- round shed lips and metal/electrode corners (remove sharp-edge field singularities)
- pollution follows the creepage surface
- controlled mesh refinement at the rounded radii
- re-validate C ~ 5600 pF and creepage ~ 3075 mm

## What changed

### 1. Surface-conductance pollution model (replaces the straight volume skin)
`ComplexEQS.F90` gained a **surface (Laplace-Beltrami) leakage term**: a sheet
conductance `Surface Conductance = Real sigma_s` set on the `insulator_surface`
BC adds `INT_Gamma sigma_s grad_s(phi).grad_s(v) (2 pi r) ds`. This is the
physically-correct surface-pollution model and follows the creepage curve
EXACTLY (wall + shed tops/undersides/lips), with no pollution geometry needed.
`cvt_observables.py` integrates the same surface term so tan-delta / leakage
capture it.

**Verified** against the ctfem scikit-fem backend (which has the same term) on
the identical mesh across sigma_s = 1e-9 .. 1e-5 S: C, divider ratio, tap phase
(5 sig figs) and tan-delta (2.2e-3 -> 1.58) all agree
(`surface_conductance_verification.md`). The existing real/Cartesian
verification (Tasks 006-009, six tests) still passes - the surface loop no-ops
when no BC sets the keyword.

### 2. Corner rounding (fillet arcs)
`cvt_ddb123.py` now builds fillet arcs (analytic rounded polygons) at the
capacitor-disc / stack-terminal outer edges (r=0.0025 m), the HV head dome and
tank top corners (0.020 m), the terminal stub (0.008 m) and the weather-shed
tip lips (0.003 m). Default `rounded=True`; `rounded=False` reproduces the old
sharp geometry. Tangent lengths are clamped to the adjacent edges (safe).

### 3. Controlled mesh refinement at radii
Curvature-driven sizing (`Mesh.MeshSizeFromCurvature = 14`, floor = 0.5 x
smallest radius) resolves each fillet. Clean mesh 13.3k -> 19.5k nodes; the
extra nodes cluster at the shed lips and disc edges (see
`rounding_before_after.png`).

## Re-validation (rounded geometry)
- terminal **C = 5616 pF** (datasheet 5600; 0.3% - slightly closer than the
  sharp 5624 because the rounded electrodes shed a little edge capacitance)
- **creepage 3062 mm** (datasheet 3075; 0.4%)
- divider ratio 0.1666 (nominal 1/6); clean phase ~ 0
- cross-backend Elmer 5616 vs skfem 5610 pF (0.1%)
All checks pass (`validate_cvt.py`).

## Notes / scope
- The wall-shed root transitions are porcelain-porcelain (same material, no field
  interface) so they are intentionally left unfilleted.
- The straight volume pollution skin (`with_pollution_layer`) is retained for
  backward-compat but is superseded by the surface-conductance model.

## How to run
```powershell
python publication_pipeline/scripts/build_solver.py               # rebuild solver
python publication_pipeline/scripts/verify_complex_eqs.py         # real/axisym re-verify
python publication_pipeline/scripts/verify_surface_conductance.py # surface term vs skfem
python publication_pipeline/scripts/run_cvt.py                    # rounded clean CVT
python publication_pipeline/scripts/validate_cvt.py               # C / creepage / cross-backend
python publication_pipeline/scripts/viz_rounded_zoom.py           # sharp-vs-rounded figure
```
Artifacts under `results/processed/013_surface_conductance/`.

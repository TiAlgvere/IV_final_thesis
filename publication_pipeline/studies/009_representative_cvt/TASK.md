# Task 009 - Representative CVT Geometry (Arteche DDB-123)

## Goal
Transition the publication pipeline off the toy verification divider (Tasks
006-008, a Cartesian slab) onto a realistic, datasheet-anchored **axisymmetric**
CVT, so subsequent fault studies run on a geometry that actually represents the
device.

## What was built
- **Self-contained CVT geometry** `geometry/gmsh/cvt_ddb123.py` - an independent
  port of the validated ctfem DDB-123 model (no dependency on the legacy
  prototype). 12-element homogenized capacitor stack with floating interior
  divider discs (C1/C2, tap between them), porcelain housing with 27 weather
  sheds, oil fill, HV head dome, grounded EMU tank. Datasheet anchors: Um 123 kV,
  standard C 5600 pF, H 1830 mm, creepage 3075 mm.
- **Axisymmetric ComplexEQS** - the solver gained the radial metric weight
  (`integral ... (2 pi r) dr dz`), gated on `Coordinate System = Axi Symmetric`,
  leaving the verified Cartesian path (Tasks 006-008) untouched.
- **Run/observables/validation** scripts: `run_cvt.py`, `cvt_observables.py`,
  `validate_cvt.py`.

## Formulation choices (carried from the validated prototype)
- Floating metal electrodes via **capped conductivity sigma = 1 S/m** (NOT
  physical 3.5e7 S/m, which causes spurious LU-roundoff loss) - they float to
  their divider potentials with no Dirichlet on the interior discs.
- Dielectric loss **pre-folded into an effective conductivity**
  `sigma_eff = sigma + omega*eps0*eps_r*tan_delta` in the SIF renderer, so the
  solver needed no loss change - only the axisymmetric weight.
- Boundary conditions match ctfem: U0 on `hv_electrode`; **0 on both
  `ground_electrode` and `farfield`** (grounded enclosure); axis r=0 natural.
- U0 = 123 kV / sqrt(3) (phase-to-earth). C and the divider ratio are
  U0-independent.

## Acceptance criteria (all met)
- Solver re-verified: all Cartesian tests still pass AND a new **axisymmetric
  coaxial-capacitor** test matches `2 pi eps L / ln(b/a)` to ~2e-5
  (`verify_complex_eqs.py`).
- Geometry creepage 3062 mm vs datasheet 3075 mm (0.4%).
- **Terminal C = 5624 pF** vs datasheet 5600 pF (0.4%); **divider ratio 0.1666**
  vs nominal 2/12; phase ~ 0 (clean low-loss divider).
- **Cross-backend** (Elmer vs ctfem scikit-fem on the identical mesh): C agrees to
  5 significant figures (5624.0 vs 5623.9 pF); divider ratio identical. This is
  the strong V&V oracle and confirms the port is faithful.

## Out of scope (next tasks)
- Surface pollution on the sheds / insulator (re-applying the Task 006-008
  conductive-layer model to the realistic creepage surface).
- Capacitor-stack fault cases (shorted/aged element, water ingress in oil).
- Mesh-convergence (GCI) study of the CVT.
- Any multiphysics (thermal/moisture).

## How to run
```powershell
python publication_pipeline/scripts/build_solver.py          # rebuild solver if edited
python publication_pipeline/scripts/verify_complex_eqs.py     # solver re-verification (incl. axisym coax)
python publication_pipeline/scripts/run_cvt.py                # build + solve the DDB-123
python publication_pipeline/scripts/validate_cvt.py           # datasheet + cross-backend checks
```
Outputs under `results/{raw,processed}/009_representative_cvt/`.

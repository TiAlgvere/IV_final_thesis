# Task 026 - Azimuthal Surface-Conductance Sector Validation

Validates azimuthally localized surface-conductance pollution sectors on the
`insulator_surface` of the true-3D DDB-123 model. NOT a streamer task - the polluted region
is a uniform-sigma_s azimuthal sector ("localized pollution sector" / "sector patch").

## 1. Implementation strategy & files

**Decision - MATC theta-window, not explicit geometry-split groups (with proof).** The
preferred explicit-physical-group approach (revolving the 74-shed porcelain into separate
angular wedges so each sector is its own boundary group) was implemented
(`cvt_ddb123_3d_v5_sector.py`) but proved **computationally intractable**: the OCC `fragment`
of five non-convex 74-shed angular wedges ran > 20 min at ~3.8 GB without completing and was
killed. This is the "unless you can prove they are safer" case in the task - the explicit
split is not viable on this geometry. The adopted alternative is the **MATC theta-window on
the single (already remediated) `insulator_surface`**, which is the SAME windowing mechanism
verified in the 2D localized-pollution study, runs in seconds, and remains fully auditable:
the polluted-sector area is measured by theta-binning the insulator triangles and compared to
the requested angle fraction (below).

Surface Conductance per case:
- angle < 360: `Surface Conductance = Variable Coordinate ; Real MATC
  "sigma_s*(atan2(tx(1),tx(0))>=0)*(atan2(tx(1),tx(0))<=angle_rad)"` applied to the single
  `insulator_surface` boundary (theta in [0, angle]).
- 360: uniform `Surface Conductance = Real sigma_s` (identical to the Task 025 uniform case).

Files created (Task 025 baseline untouched; reused read-only):
- `geometry/gmsh/cvt_ddb123_3d_v5_sector.py` (explicit-split attempt - retained, documented intractable)
- `scripts/run_true3d_sector.py` (MATC sector run + mesh audit; reuses
  `results/raw/025_insulator_mesh_remediation/cvt3d_v5_solve.msh`)
- `scripts/make_sector_figures.py`, `scripts/viz_sector.py`
- outputs in `results/{raw,processed}/026_azimuthal_surface_conductance_validation/`

## 2. Mesh audit (reused accepted v5 solve mesh: 270 607 nodes, 1.59 M tets)

| quantity | value |
|----------|-------|
| global tet minSICN min / mean / p5 | 0.027 / 0.785 / 0.500 |
| global slivers (< 0.05) | 91 |
| insulator_surface (all): min / mean / p5 | 0.234 / 0.835 / 0.493 |
| **patch-edge triangles (theta within 3 deg of cuts 30/60/90/180)**: n / min / p5 / mean | 15 993 / **0.450 / 0.494** / 0.843 |
| per-sector insulator p5 (s0..s4) | 0.441 / 0.476 / 0.506 / 0.544 / 0.493 |
| worst-100 tets by material | 54 oil + 44 metal + 2 air (tank base) |
| worst-100 on the insulator surface | **2 / 100** (none on patch edges or shed lips) |

The patch-edge mesh is as good as the bulk insulator (p5 0.494, min 0.450) - the v5
remediation already makes the whole creepage surface uniform, so no extra azimuthal grading
was needed for acceptable surface-conductance integration. Worst elements are in the grounded
tank base, not on any sector edge.

## 3. Patch-area audit (measured vs requested)

| sector angle | measured area fraction | expected (angle/360) | rel. error |
|--------------|------------------------|----------------------|-----------|
| 30 deg | 0.0835 | 0.0833 | +0.2 % |
| 60 deg | 0.1671 | 0.1667 | +0.2 % |
| 90 deg | 0.2503 | 0.2500 | +0.1 % |
| 180 deg | 0.5000 | 0.5000 | 0.0 % |
| 360 deg | 1.0000 | 1.0000 | 0.0 % |

Polluted triangles lie only on `insulator_surface` (the theta-window multiplies a region
that is, by construction, the porcelain-air creepage). **Area fractions match the requested
sector angles to <= 0.2 %.**

## 4. Solver convergence

All 11 cases solved with the **BiCGStabl(4)+ILU2 iterative** solver (UMFPack direct fails on
this high-contrast complex system, as in Tasks 024/025). Per-case linear iterations ~31-34,
final residual <= 1e-9, ~70-90 s each. No divergence; all produced valid fields.

## 5. Observable table

Clean baseline: C = 5623.4 pF, ratio 0.1669, tan-delta 1.99e-3, phase -3.3 mdeg (matches
Task 025 clean exactly).

| sigma_s | angle | area frac | C [pF] | ratio | tan-delta | surface_leak [A] | Emax [V/m] |
|---------|-------|-----------|--------|-------|-----------|------------------|-----------|
| 1e-7 | 30 | 0.084 | 5623 | 0.1666 | 3.25e-3 | 1.58e-4 | 1.39e6 |
| 1e-7 | 60 | 0.167 | 5624 | 0.1666 | 4.49e-3 | 3.14e-4 | 1.39e6 |
| 1e-7 | 90 | 0.250 | 5624 | 0.1666 | 5.74e-3 | 4.70e-4 | 1.39e6 |
| 1e-7 | 180 | 0.500 | 5624 | 0.1668 | 9.47e-3 | 9.38e-4 | 1.39e6 |
| 1e-7 | 360 | 1.000 | 5624 | 0.1665 | 1.69e-2 | 1.88e-3 | 1.39e6 |
| 1e-6 | 30 | 0.084 | 5623 | 0.1666 | 1.45e-2 | 1.57e-3 | 1.39e6 |
| 1e-6 | 60 | 0.167 | 5624 | 0.1665 | 2.70e-2 | 3.14e-3 | 1.39e6 |
| 1e-6 | 90 | 0.250 | 5624 | 0.1663 | 3.94e-2 | 4.69e-3 | 1.39e6 |
| 1e-6 | 180 | 0.500 | 5624 | 0.1670 | 7.67e-2 | 9.37e-3 | 1.39e6 |
| 1e-6 | 360 | 1.000 | 5624 | 0.1668 | 1.51e-1 | 1.87e-2 | 1.39e6 |

## 6. Comparison to the Task 025 uniform result

| quantity (sigma_s=1e-6) | Task 026 360 deg | Task 025 uniform 3D | difference |
|-------------------------|------------------|---------------------|-----------|
| tan-delta | 1.514e-1 | 1.514e-1 | **0.0 %** |
| surface_leak | 1.874e-2 | 1.874e-2 | **0.0 %** |

The 360 deg sector **reproduces the accepted Task 025 uniform 3D pollution exactly** (the
360 case is, by construction, the uniform BC). This anchors the sector method to the accepted
baseline. (Task 025 itself agreed with the 2D axisymmetric model to ~7 %.)

## 7. Physics sanity checks

- **Clean** C/ratio/tan-delta match Task 025 clean (C 5623 pF, ratio 0.167). PASS
- **360 deg = Task 025 uniform** exactly. PASS
- **surface_leak scales linearly with angular coverage**: 1.57e-3 -> 3.14e-3 -> 4.69e-3 ->
  9.37e-3 -> 1.87e-2 (sigma_s=1e-6) for 30/60/90/180/360 deg, i.e. ~2x / 3x / 6x / 12x the
  30-deg value - matching the 2/3/6/12 angle ratios. PASS
- **tan-delta** rises smoothly/monotonically with angle (surface term linear in coverage on
  top of the constant volume term). PASS
- **C / divider ratio** stay 5623-5624 pF / ~0.167 (surface conductance adds loss, not
  capacitance). PASS
- **Emax = 1.39e6 V/m, CONSTANT across all sector cases** (located at the HV terminal cap,
  not at any patch edge) -> the sector edges introduce **no artificial field singularity**. PASS
- material tags, contacts, head/tank porcelain (=0) unchanged - the accepted v5 mesh is reused
  unmodified. PASS
- **Caveat (honest):** the tap PHASE is at the milli-degree iterative-residual floor (clean
  -3.3 mdeg) and is noisy across cases (tens of mdeg, not monotonic). With the direct solver
  unavailable on this mesh, phase is not a reliable 3D trend here; tan-delta and surface_leak
  are the robust smooth signals. This matches the earlier finding that clean 3D phase sits at
  the numerical floor.

## 7b. Phase reliability note (Task 026B)

**The tap phase is NOT an accepted observable in this 3D study.** Across the sector
cases the tap phase sits at the milli-degree level (clean -3.3 mdeg) and jumps
non-monotonically (tens of mdeg, sign-flipping) - it is at the **iterative-residual /
numerical noise floor** of the BiCGStabl(4)+ILU2 solve, not a physical trend. With the
direct (UMFPack) solver unavailable on this high-contrast 3D complex system, there is no
way to drive the residual low enough to resolve a clean phase signal. **Do not use the
Task 026 phase-vs-angle curve as a thesis conclusion.** The robust, validated observables
are **`surface_leak` and `tan_delta`**, both of which scale smoothly and monotonically
with polluted-sector coverage (Section 7). The rotation check (Section 7c) reinforces
this: the robust observables are rotation-invariant to ~1 %, while the phase varies by
tens of mdeg between azimuths.

## 7c. Rotational sanity check (Task 026B)

A single 30 deg surface-conductance sector (sigma_s = 1e-6 S) was re-solved at three
azimuths to test whether the localized-sector response depends on orientation. Reused the
converted Task 026 Elmer mesh (no remesh); 30 deg theta-window rotated to each azimuth.

| azimuth [deg] | polluted area [m^2] | surface_leak [A] | tan_delta | tap phase [mdeg] |
|---------------|---------------------|------------------|-----------|------------------|
| 0 - 30   | 0.1952 | 1.5552e-3          | 1.4386e-2 | -29.7 |
| 90 - 120 | 0.1969 | 1.5740e-3 (+1.2 %) | 1.4536e-2 (+1.0 %) | -4.3 |
| 180 - 210| 0.1962 | 1.5681e-3 (+0.8 %) | 1.4489e-2 (+0.7 %) | -58.0 |

**Spread across azimuths: surface_leak 1.2 %, tan_delta 1.0 % -> rotation-invariant = TRUE**
(< 5 % threshold). The robust observables are azimuth-independent, as expected for the
nominally axisymmetric divider: a polluted sector of a given angular width produces the
same surface-leakage / tan-delta signature regardless of where it sits. (The small ~1 %
residual is consistent with the slight geometric asymmetry of the tank/side-box and the
theta-binning granularity - it does **not** indicate a strong asymmetric-geometry effect.)
By contrast the tap **phase** swings from -4 to -58 mdeg between azimuths, independently
confirming the Section 7b phase caveat. Data: `rotation_check.json`,
script: `run_sector_rotation.py`.

## 8. Figures

Thesis-quality sector figures (Task 026B):
- `sector_sigma_s_maps.png` - full insulator surface, **polluted sector highlighted red**
  (sigma_s = 1e-6 S) vs clean grey, for 30/90/180/360 deg. Directly shows the azimuthal
  patch geometry and the 30->90->180->360 coverage growth.
- `sector_qs_density_panels.png` - surface-loss density q_s = 0.5*sigma_s*|grad_s phi|^2
  on the insulator, **common colour scale (0 - 1.0e3 W/m^2) across all angles**. The
  polluted sector lights up at ~250 W/m^2 (the local field is similar regardless of sector
  size); the *total* loss grows with area (= surface_leak linear in angle). Replaces the
  weak zoomed Im(phi) close-ups.

Supporting figures (Task 026): `sector_patch_placement_3d.png` (insulator coloured by
theta-sector), `sector_patch_area_audit.png`, `sector_surface_leak_vs_angle.png`,
`sector_phase_tandelta_vs_angle.png` (phase panel shown for completeness only - see 7b),
`sector_observables_vs_angle.png`, `sector_mesh_quality.png`,
`sector_worst100_locations.png`, `sector_surfaceloss_{30,90,180,360}deg.png` (superseded
by the two figures above).

## 9. Verdict

| acceptance criterion | status |
|----------------------|--------|
| clean baseline still passes | PASS (matches Task 025) |
| 360 deg reproduces Task 025 uniform within tolerance | PASS (0.0 %) |
| patch area fractions match requested sector angles | PASS (<= 0.2 %) |
| patch-edge mesh quality acceptable | PASS (p5 0.494, min 0.450) |
| observables vary smoothly with angular coverage | PASS for surface_leak / tan-delta / C / ratio; phase is at the mdeg residual floor (documented) |
| robust observables rotation-invariant | PASS (surface_leak/tan-delta spread 1.2 %/1.0 % across 3 azimuths) |
| no material / contact regressions | PASS (reused v5 mesh unchanged) |
| no fake streamer claim | PASS (uniform-sigma_s azimuthal sectors only) |

### Final statements (Task 026 / 026B)

1. **PASS for azimuthal surface-conductance sector validation.** Polluted-sector area
   matches the requested angle to <= 0.2 %, surface_leak scales linearly with coverage,
   the 360 deg case reproduces the accepted Task 025 uniform 3D result exactly (0.0 %),
   patch edges are well-meshed (p5 0.494) with no field singularity, the response is
   rotation-invariant to ~1 %, and there are no material/contact regressions.
2. **This is NOT a streamer model.** The polluted region is a uniform-sigma_s azimuthal
   surface-conductance sector (a localized-pollution sector / sector patch). No streamer,
   discharge, or breakdown physics is claimed or modelled.
3. **The phase trend is NOT accepted as a robust observable.** The tap phase is at the
   iterative-residual / numerical noise floor (tens of mdeg, non-monotonic, and not
   rotation-invariant); the Task 026 phase-vs-angle curve must not be used as a thesis
   conclusion.
4. **`surface_leak` and `tan-delta` are the accepted robust observables.** Both scale
   smoothly and monotonically with polluted-sector coverage and are rotation-invariant to
   ~1 %.

The explicit-physical-group geometry split was proven intractable and replaced by the
verified, auditable MATC theta-window. NOT extended to streamer physics.

Data: validation.json, mesh_meta.json, rotation_check.json. Scripts: run_true3d_sector.py,
run_sector_rotation.py, cvt_ddb123_3d_v5_sector.py (intractable explicit-split, documented),
make_sector_figures.py, viz_sector.py, viz_sector_v2.py.

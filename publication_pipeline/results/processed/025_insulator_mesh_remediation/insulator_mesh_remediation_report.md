# Task 025 - 3D Insulator-Surface Mesh Remediation

## Goal & result

Make the 74-shed 3D insulator surface suitable for quantitative surface-conductance /
streamer studies. The Task-024 blocker (insulator_surface p5 = 0.043, worst elements on the
shed-lip pollution path) is **resolved**: insulator_surface p5 0.043 -> ~0.49-0.54, global
slivers 15400 -> ~90, and the worst elements moved OFF the insulator surface into the
harmless grounded-tank base. Anatomy targets preserved; clean C/ratio still validate; uniform
3D surface conductance agrees with the axisymmetric model to within ~7 %.

## Remediation method (no macro-anatomy change)

The macro anatomy (74 alternating sheds, components, creepage, C, ratio) is unchanged. Two
mesh-quality changes, both "required for mesh quality" and both preserving the targets:

1. **Mesh-friendly shed micro-profile** (catalogue does not constrain these dimensions):
   - shed tip thickness 2.5 -> 4.5 mm (removes the sliver-prone thin tip),
   - droop 4.0 -> 2.0 mm (shallower inter-shed air wedge),
   - lip radius 1.5 -> 1.2 mm,
   - large/small overhangs **re-tuned analytically** (ov_large 20.0 mm, ov_small 8.65 mm) so
     creepage = **3075.0 mm** exactly.
2. **Mesh controls**: curvature-driven refinement OFF (it was forcing sub-mm lip slivers),
   controlled surface size on the sheds, Frontal-Delaunay (2D) + Delaunay (3D), MeshSizeMin
   = 2.2 mm, and multi-pass Netgen optimisation (OptimizeThreshold 0.35).

## Quality: global vs insulator_surface (reported separately)

Audit mesh (~2.8 M tets) / solve mesh (~1.6 M tets):

| metric | v4 (before) | v5 audit | v5 solve |
|--------|-------------|----------|----------|
| global min minSICN | 0.033 | 0.028 | 0.027 |
| global mean | 0.72 | 0.82 | 0.78 |
| global p1 / p5 | 0.053 / 0.138 | 0.475 / 0.592 | 0.399 / 0.500 |
| global slivers (< 0.05) | ~15 400 | **93** | **91** |
| **insulator_surface min** | **0.043** | **0.308** | **0.234** |
| **insulator_surface p5** | **0.043** | **0.535** | **0.493** |
| insulator_surface mean | 0.74 | 0.892 | 0.835 |
| insulator_surface p50 | - | 0.988 | 0.953 |

**Insulator-surface p5 improved ~12x (0.043 -> 0.49-0.54); its minimum is now 0.23-0.31.**
(`ddb123_v5_quality_comparison.png`, `ddb123_v5_global_quality_hist.png`.)

## Worst-100 element locations

Worst-100 (solve mesh): **52 oil + 44 metal + 4 air; ZERO porcelain, ZERO on the insulator
surface.** 96/100 cluster at r ~ 0.20 m, z ~ 0.10-0.30 m - the secondary-box / oil-fitting
junctions inside the grounded tank (low field, EQS-inert). The 4 air are at the wall-head /
wall-tank corners (r ~ 0.12, z = 0.55 / 1.61), not the shed lips. The low-quality elements
therefore affect only the tank-base solve robustness (handled by the iterative solver), NOT
the insulator surface, the capacitor, or any field hot-spot. (`ddb123_v5_worst100_locations.png`.)

## Regression checks (no regressions)

- Material tags: **head porcelain = 0.0, tank porcelain = 0.0**; material volumes consistent
  (porcelain 0.0224, oil 0.0687, element 0.0173, metal 0.0531, resin 1.9e-4 m^3).
- Floating features: secondary_box / oil_valve / oil_level share 74 / 80 / 88 nodes with the
  tank -> all ATTACHED.
- Creepage = 3075.0 mm; 74 alternating sheds preserved.

## Clean 3D validation (remediated mesh)

Cartesian 3D ComplexEQS (UMFPack direct fails on the high-contrast system -> BiCGStabl
iterative succeeds, 70 s):

| quantity | v5 3D | axisymmetric | delta | vs target |
|----------|-------|--------------|-------|-----------|
| capacitance | 5623 pF | 5616 pF | +0.12 % | **+0.42 % vs 5600** |
| divider ratio | 0.16695 | 0.16658 | +0.22 % | **+0.13 % vs 1/6** |
| tap phase | -3.3 mdeg | ~0 | iterative residual ~0 | near zero |

Clean C and ratio still pass (within 1 %).

## Uniform 3D surface conductance vs axisymmetric (requirement 6)

Uniform Surface Conductance sigma_s on insulator_surface; the 3D surface loss is integrated
over the porcelain-air creepage triangles (INT sigma_s |grad_s phi|^2 dS) and compared to the
verified 2D axisymmetric surface-conductance model at the same sigma_s.

| sigma_s [S] | tan-delta 3D | tan-delta 2D | delta | surface_leak 3D | surface_leak 2D | delta |
|-------------|--------------|--------------|-------|-----------------|-----------------|-------|
| 1e-7 | 1.694e-2 | 1.809e-2 | -6.4 % | 1.876e-3 | 2.017e-3 | -7.0 % |
| 1e-6 | 1.514e-1 | 1.627e-1 | -6.9 % | 1.874e-2 | 2.014e-2 | -6.9 % |

**Agreement ~7 %, documented tolerance.** The offset is essentially constant across a decade
of sigma_s (identical -7 % at both points), so the surface-conductance physics scales
correctly in 3D; the residual is a resolution/integration effect (coarse 1.6 M solve mesh +
triangle surface integration vs the fine 2D axisymmetric line integral). A finer 3D surface
mesh would reduce it; ~7 % is acceptable for this uniform cross-check.

## Acceptance checklist

| criterion | status |
|-----------|--------|
| no material-tag regressions | PASS (head/tank porcelain = 0) |
| no floating features | PASS (all side features attached) |
| clean C and ratio still pass | PASS (C +0.42 %, ratio +0.13 % vs target) |
| insulator_surface quality materially improves | PASS (p5 0.043 -> 0.49-0.54; min -> 0.23-0.31) |
| uniform 3D pollution agrees with axisymmetric | PASS (~7 % tolerance, consistent scaling) |

## Tradeoffs

Only the shed *micro-profile* (tip thickness, droop, lip radius) was adjusted, with overhangs
re-tuned so creepage stays 3075 mm. The macro anatomy, component set, material tags, C, ratio
and 74-shed count are unchanged. No tradeoff was required at the anatomy level. The remaining
~90 global slivers are in the tank-base fitting junctions (grounded, low-field, EQS-inert);
they can be cleaned later but do not affect any quantity of interest.

## Readiness verdict (updated from Task 024)

| aspect | status |
|--------|--------|
| External geometry / anatomy | PASS |
| Internal material tagging | PASS |
| Clean capacitance validation mesh | PASS |
| **Mesh for final 3D *uniform* surface pollution** | **PASS** (insulator p5 ~0.5, worst elements off the surface, uniform pollution validated to ~7 %) |
| Mesh for *azimuthally localized* streamer / hotspot | CONDITIONAL - the insulator surface is now uniformly high quality, but a localized patch/streamer still needs deliberate **azimuthal mesh grading at the patch edges** (Task 025 covers uniform only; localized refinement is the next step) |

Localized streamer was intentionally NOT run (per task). Next step before localized studies:
add azimuthal mesh refinement on a defined patch/sector and re-confirm.

Data: validation.json, solve_mesh_meta.json, cvt3d_v5.msh(.quality.npz). Scripts:
cvt_ddb123_3d_v5.py, run_true3d_v5.py, make_remediation_figures.py.
Figures: ddb123_v5_quality_comparison.png, ddb123_v5_global_quality_hist.png,
ddb123_v5_worst100_locations.png.

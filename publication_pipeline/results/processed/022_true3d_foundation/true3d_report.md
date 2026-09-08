# Task 022 - True 3D DDB-123 FEA Foundation

## Purpose

Establish the first **true 3D** finite-element path for the publication pipeline and
validate it against the verified 2D axisymmetric model on the clean baseline. The
axisymmetric model is exact for rotationally symmetric cases; true 3D is required for the
azimuthally-localized faults the thesis targets (stork streamers, one-sided pollution,
shed-sector wetting, asymmetric hot-spots). This task builds the 3D foundation - no new
fault physics.

## What is truly 3D here

- **Geometry**: a full-360 deg solid model (`geometry/gmsh/cvt_ddb123_3d.py`), built from
  OCC primitive cylinders, fragmented into a conformal partition and tagged via the OCC
  fragment map (robust for the non-convex air/oil volumes). Real 3D tetrahedral mesh.
- **Solver**: the SAME verified `ComplexEQS` solver, run in **Cartesian 3D**
  (`Coordinate System = Cartesian`). The axisymmetric `2*pi*r` metric is gated on
  `AxisSymmetric`/`CylindricSymmetric` and is therefore **OFF** here - confirmed by the
  formulation (`SUM(dBasisdx(q,1:3)*dBasisdx(p,1:3))` is a full 3D gradient dot-product)
  and by the validation below.
- **Fields & observables**: Re/Im potential on tetrahedra; C, |Vtap|, phase, stored
  energy and Emax are computed by **tetrahedral volume integration** (no 2*pi*r), and the
  figures are the genuine 3D field on a real axial slice - never a revolved 2D field.

## What remains approximate (documented)

- **Homogenized stack** (permitted by Task 022): the 12-element series stack is collapsed
  to a single dielectric column split by the tap into C1 (HV) and C2 (ground). With the
  validated `eps_r_eff` and the tap at the `n_c2/n` height this reproduces both the
  terminal capacitance and the divider ratio by construction (verified below). The
  individual interior foils are not resolved.
- **Plain porcelain cylinder - weather sheds deferred.** Sheds do not affect the clean
  baseline C / ratio / phase (verified - the validation passes without them). They ARE
  needed for creepage (3075 mm) and the surface-conductance pollution path, so they are
  the first item of the next refinement.
- **Finite air domain** (r = 0.8 m cylinder + far-field phi=0) instead of the 2D
  r = 5.49 m: the internal divider dominates C (eps_r_eff ~ 35000 vs 1), so the modest air
  box changes C negligibly (validation < 0.1 %).
- **Sharp primitive edges** (terminal stub) are not filleted as in the 2D rounded model,
  so Emax sits on the terminal corner and is corner/mesh-sensitive - reported cautiously.

## Mesh (coarse foundation)

| quantity | value |
|----------|-------|
| nodes | 23 527 |
| tetrahedra | 136 504 |
| min quality (minSICN, after Netgen optimise) | 0.318 |
| mean quality (minSICN) | 0.841 |
| element size: stack/electrodes | 0.010 - 0.030 m |
| element size: porcelain / oil | 0.030 / 0.040 m |
| element size: air (far) | 0.220 m |
| physical groups | 10 volumes + 4 surfaces (hv_electrode, ground_electrode, insulator_surface, farfield) |

## Solver & resources

- **Linear solver**: UMFPack **direct** succeeded on the complex 272 k-DOF system in
  **53 s** (single run, no instability). A BiCGStabl+ILU2 iterative fallback is wired in
  `run_true3d.render_sif_3d(direct=False)` for when the mesh outgrows direct memory.
- Memory: direct factorisation fit in available RAM at this size (~10^5 tets). For a
  full-resolution localized-fault mesh (sheds + azimuthal refinement, ~10^6+ tets), a
  direct solve will not fit - the iterative path becomes mandatory there.

## Validation vs axisymmetric (clean baseline)

Both models solved fresh in the same run (`run_true3d.py`); the 3D field is real, not
revolved.

| quantity | true 3D | axisymmetric | delta | acceptance | result |
|----------|---------|--------------|-------|------------|--------|
| terminal capacitance | 5622 pF | 5616 pF | **+0.09 %** | within 2-5 % | PASS (datasheet 5600 -> +0.4 %) |
| divider ratio | 0.1665 | 0.1666 | **-0.03 %** | within 1 % | PASS |
| \|Vtap\| | 11 830 V | 11 830 V | -0.03 % | - | PASS |
| tap phase | -0.012 mdeg | -0.122 mdeg | both ~0 | near zero | PASS (3D even closer to 0) |
| stored energy | 14.17 J | 14.16 J | +0.09 % | - | PASS |

Emax (3D) = 6.65e5 V/m at (r,z) = (0.035, 1.838) m - the HV terminal-stub edge (sharp
primitive corner; see caveat). The clean phase being at the numerical floor (sub-mdeg) in
both models confirms a stable, symmetric 3D solve.

**All acceptance criteria are met, with margin** (capacitance error 0.09 % vs the 2-5 %
bar). The true 3D path reproduces the validated axisymmetric physics.

## Figures (all from the true 3D mesh)

- `true3d_material.png` - material regions, exterior (tank / porcelain / head dome+terminal).
- `true3d_mesh.png` - the 3D tetrahedral surface mesh.
- `true3d_phi_re.png` - Re(phi) on an axial slice: 71 kV at the HV head decaying to the
  grounded tank / far field.
- `true3d_Emag.png` - log|E| on the slice: conductors field-free, field peaking at the
  terminal edge.
- `true3d_cutaway.png` - axial cutaway exposing the homogenized C1 / tap / C2 column, oil,
  porcelain wall, tank and head.
- `true3d_hotspot.png` - |E| on the device surface with the field hot-spot marked.

## Is this mesh sufficient for localized faults?

**Not yet - it is the symmetric foundation.** The clean baseline is validated, but to
model azimuthally-localized faults the next mesh must add:
1. **Weather sheds** (needed for creepage and the surface path), revolved in 3D.
2. **Azimuthal + surface refinement** on `insulator_surface` so a one-sided / sector
   pollution patch or a vertical streamer is resolved (the current tet mesh has no
   deliberate azimuthal grading).
3. **3D surface-conductance BC**: the `insulator_surface` group already exists; the
   Laplace-Beltrami `Surface Conductance` term (verified in 2D) must be exercised in 3D.

## Recommended next refinement (Task 023 direction)

1. Add the 27 weather sheds to the 3D geometry; re-validate uniform (axisymmetric)
   pollution in 3D against the 2D surface-conductance result (a symmetric cross-check).
2. Refine the porcelain/creepage surface and grade the mesh azimuthally; switch the linear
   solver to the iterative path as the element count crosses ~10^6.
3. Introduce the first **azimuthal** fault (e.g. a 30-90 deg pollution sector or a single
   vertical streamer) - the genuinely-3D regime that the axisymmetric model cannot
   represent - and compare its asymmetric phase / leakage signature to the symmetric case.
4. Optionally resolve the discrete stack foils if a 3D internal-fault study needs them.

## Caveats

- Model-based foundation for the representative DDB-123; **not** field-validated.
- Homogenized stack + plain porcelain + finite air + sharp edges as listed above; the
  validation confirms these do not affect the clean baseline targets, but Emax magnitude/
  location is corner-sensitive and is reported cautiously.
- One linear-solver run; no 3D mesh-convergence study yet (the coarse mesh already meets
  the C / ratio targets, so convergence is a refinement-stage task).

Data: `validation.json`. Scripts: `geometry/gmsh/cvt_ddb123_3d.py`, `scripts/run_true3d.py`,
`scripts/viz_true3d.py`.

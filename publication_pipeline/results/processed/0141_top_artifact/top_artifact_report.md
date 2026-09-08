# Task 014.1 - Top Insulator/Head-Transition Artifact

## Finding: RENDERING-INTERPOLATION ARTIFACT (not geometry, not material-ID)

The Task-014 3D material render coloured a **point-interpolated categorical**
material index. Averaging integer material IDs across an element/oil/metal
junction yields fractional indices that map to a DIFFERENT material's colour
(e.g. element id 3 averaged with oil id 1 = 2 = porcelain's colour) - this is the
blue/porcelain-looking patch at the head transition.

Evidence:
- `top_2d_material_and_mesh.png` - the TRUE 2D geometry at the transition is clean:
  topmost capacitor element + stack_top metal disc, thin oil, porcelain wall ending
  at z = stack_z_hi, solid metal head dome above. No stray porcelain/blue body.
- `top_3d_flat.png` - flat-shaded CELL material (cell_data carried through revolve
  is reported in the title); no blended patch.
- `top_3d_categorical.png` / `figureA_material_cutaway.png` - per-material solid
  colours (no scalar interpolation at all); the patch is gone.

Conclusion: **geometry and material IDs are correct; leave geometry unchanged.** The
thesis material figure uses categorical (per-material) colouring, not interpolated
scalars. Continuous fields (Im(phi), dE, surface loss) are genuinely smooth and are
rendered with interpolation as usual.

## Thesis figures
- Figure A `figureA_material_cutaway.png` - material cutaway (categorical).
- Figure B `figureB_Imphi.png` - Im(phi), polluted.
- Figure C `figureC_dE.png` - |E|_polluted - |E|_clean.
- Figure D `figureD_surface_loss.png` - surface pollution loss density on the
  porcelain creepage only (sigma_s = 1e-06 S).

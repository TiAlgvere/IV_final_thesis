# Task 014 - 3D Revolved Renders (DDB-123)

Axisymmetric solution revolved 270 deg (cutaway) with pyvista; device
cross-section only (surrounding air excluded). Clean vs polluted use the
surface-conductance model with sigma_s = 1e-06 S on insulator_surface.

## Figures
- `3d_material.png` - material regions (oil, capacitor elements, metal electrodes,
  porcelain wall + sheds); cut faces expose the C1/C2 stack.
- `3d_Emag.png` - log|E| (polluted): field stress, peaking at the electrode edges.
- `3d_Imphi.png` - Im(phi) (polluted): the quadrature potential driven by the
  surface-pollution leakage - concentrated along the porcelain creepage.
- `3d_loss.png` - volume dielectric loss density sigma|E|^2 (log).
- `3d_dE.png` - |E| change polluted-minus-clean: where the pollution redistributes
  the field.

These are visualization renders of the (verified) axisymmetric solution; all
quantitative results remain the 2-D/observable values from the earlier tasks.

# Task 021.5 - 3D Revolved Thermal Renders (DDB-123)

The verified one-way electrothermal T-fields from Task 021 revolved 270 deg
(cutaway) with pyvista; the solid CVT only (surrounding air excluded), T mapped in
degrees C (ambient T_amb = 40 C). Hot-spot marked with a sphere. Pure
visualization - no new physics, the thermal solver is unchanged.

## Figures
- `thermal_3d_healthy.png` - baseline dielectric loss only; the stack is mildly warm
  (peak ~45 C), well below any limit.
- `thermal_3d_C1_loss.png` - C1 dielectric loss (tan-delta 0.10): **internal dielectric
  loss heats the capacitor-stack VOLUME**; the hot-spot sits in the mid-stack
  (peak ~285 C, far above the 105 C insulation hot-spot limit).
- `thermal_3d_C2_loss.png` - C2 dielectric loss (tan-delta 0.05): the hot-spot localises
  to the C2 section (lower stack, near the tap), peak ~131 C - the same
  mechanism, but the C2 loss concentrates in only 2 elements, so a lower total power
  still gives a high local temperature (see Task 021).
- `thermal_3d_pollution.png` - heavy full-creepage pollution: **surface pollution heats
  the porcelain CREEPAGE path** (outer wall / sheds), peak ~74 C, while the
  core stays cooler - a distinctly different (surface) heating pattern.
- `thermal_3d_comparison.png` - the three side by side on a **common color scale**
  (40-105 C = ambient to insulation limit): healthy stays dark/cool,
  pollution warms the surface, and the C1-loss case saturates the scale (it is over
  the insulation limit throughout the stack).
- `thermal_3d_stack_closeup.png` - zoom on the capacitor-stack hot-spot (internal loss).
- `thermal_3d_creepage_closeup.png` - zoom on the porcelain creepage heating (pollution).

## Interpretation (consistent with Task 021)
- **Internal dielectric loss heats the capacitor stack volume** - the hot-spot is
  interior, in the lossy elements, and escalates with tan-delta.
- **Surface pollution heats the porcelain creepage path** - the heat is deposited on the
  outer surface (dry-band region), a different spatial signature from internal loss.
- **Ratio faults (capacitance change, disc short) are electrically visible but thermally
  quiet** - they add ~no loss, so they are not rendered here (they look like the healthy
  baseline); see the Task 021 summary table.

These are **one-way, steady-state** thermal results (electrical loss -> temperature, no
feedback): they show where each fault deposits heat and the resulting steady rise, NOT a
coupled thermal-runaway prediction. Absolute temperatures depend on the documented
thermal assumptions; the spatial pattern and relative ranking are the robust message.

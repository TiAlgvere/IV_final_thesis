# Task 012 - DDB-123 Visual Outputs

Publication figures for the representative axisymmetric DDB-123 CVT, clean
(sigma=0) and polluted (sigma=1e-5 S/m, the Task 010 critical point). The solve
is the r>=0 half-section; full-device plots are mirrored about r=0 for clarity.

## Figures
- **material_regions.png** - the meshed cross-section coloured by material:
  air, oil, capacitor elements (homogenized stack), metal electrodes/discs,
  porcelain wall + weather sheds, and the thin red pollution skin on the wall.
- **zoom_sheds_pollution.png** - close-up of the porcelain wall, a few weather
  sheds and the 3 mm conductive pollution skin (red), with mesh edges shown so
  the skin resolution is visible.
- **potential_Re_clean_vs_polluted.png** - in-phase potential Re(phi) with white
  equipotential lines. The capacitor stack carries the HV->ground gradient; the
  two cases look near-identical (external pollution barely moves the divider).
- **potential_Im_polluted.png** - quadrature potential Im(phi): essentially zero
  in the clean case, here it is the loss-current signature driven by the
  pollution conduction (the source of the tap phase displacement).
- **Efield_clean_vs_polluted.png** - |E| on a log scale; peaks at the sharp metal
  features (terminal/disc edges). See the hotspot table for the exact location.

## Hotspot table (max |E|)

| case | max \|E\| [V/m] | r [m] | z [m] | body | material |
|------|--------------|-------|-------|------|----------|
| clean | 7.272e+05 | 0.1860 | 1.7915 | air | air |
| polluted | 7.220e+05 | 0.1860 | 1.7915 | air | air |

The peak field sits in the dielectric adjacent to a sharp metal electrode edge
(the partial-discharge-risk location); pollution on the external wall does not
move it, consistent with the weak external->internal coupling seen in Task 010.

All PNGs are 200 dpi. Source: viz_cvt.py.

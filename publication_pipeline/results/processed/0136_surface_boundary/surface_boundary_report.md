# Task 013.6 - Surface-Conductance Boundary Visualization

The pollution model is a Surface Conductance BC on the `insulator_surface`
boundary (Task 013). These figures (no solve, no geometry change) highlight that
boundary in red on the rounded clean CVT.

## What the figures show
- `surface_conductance_boundary.png` - full mirrored device; the red curve is the
  entire air-facing insulator surface where sigma_s is applied.
- `surface_conductance_zoom.png` - several sheds; the red boundary hugs the wall
  between sheds, runs out along each shed top, around the rounded lip, back along
  the underside, and onto the wall again - i.e. it follows the true creepage path.

## Creepage path length
- meshed insulator_surface (summed meridional segment length): **3000 mm**
- analytic shed-profile creepage: 3062 mm
- datasheet creepage: 3075 mm
- insulator_surface line elements: 675

## No straight radial pollution skin
- dim-2 bodies present: ['air', 'base_tank', 'element_1', 'element_10', 'element_11', 'element_12', 'element_2', 'element_3', 'element_4', 'element_5', 'element_6', 'element_7', 'element_8', 'element_9', 'foil_1', 'foil_10', 'foil_11', 'foil_2', 'foil_3', 'foil_4', 'foil_5', 'foil_6', 'foil_7', 'foil_8', 'foil_9', 'head_housing', 'oil', 'porcelain', 'porcelain_shed', 'stack_bottom', 'stack_top']
- `pollution_layer` volume body present: **False** (should be False - pollution is now the surface BC, not a volume skin)

The boundary is one continuous creepage curve, not a straight radial band, confirming the surface-conductance model replaced the old volume skin.

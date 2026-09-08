# Task 013.5 - DDB-123 Geometry Audit (sharp vs rounded)

Geometry-realism check before the 3D renders. Clean CVT solved in both
geometries; figures show material regions, full mesh, and close-ups.

## Fillet radii used (rounded geometry)
| feature | radius [mm] |
|---------|-------------|
| weather-shed tip lip | 3.0 |
| HV compensator dome corner | 20.0 |
| tank top-outer corner | 20.0 |
| capacitor-disc / stack-terminal edge | 2.50 |
| HV terminal stub corner | 8.0 |
| (mesh curvature target) | 14 elem / 2*pi |

## Sharp vs rounded metrics
| metric | sharp | rounded | change |
|--------|-------|---------|--------|
| mesh nodes | 13252 | 19507 | +6255 |
| mesh elements | 26020 | 38360 | +12340 |
| terminal C [pF] | 5624.0 | 5616.4 | -0.14% |
| divider ratio | 0.1666 | 0.1666 | +0.00% |
| Emax [V/m] | 6.599e+05 | 7.439e+05 | +12.7% |
| Emax location (r,z) [m] | (0.182,1.797) air | (0.019,1.829) air | |

Datasheet anchors: C 5600 pF, creepage 3075 mm (analytic 3062 mm, unchanged).

## Read-out
- C and divider ratio are essentially unchanged by rounding (sub-percent): the
  fillets are small vs the electrode dimensions, so terminal behaviour is preserved.
- Emax is NOT a 'lower-is-better' comparison. At a SHARP corner |E| is a numerical
  singularity - finite on a given mesh but mesh-dependent (it grows without bound
  under refinement), so the sharp value is an under-resolved artifact. The rounded
  fillets give a BOUNDED, resolved peak: the trustworthy value. Here rounding the
  dome edge moved the peak from the dome corner (0.18, 1.80) to the terminal-stub
  tip (0.02, 1.83) - the genuine smallest-radius HV feature - and THAT resolved
  peak (~7.4e5 V/m) is what the |E| / PD-risk readouts should use.
- So the purpose of rounding is met: it turns Emax from a mesh-dependent singularity
  estimate into a finite, physically-meaningful peak at a real geometric feature.
- Node/element count rises from the curvature refinement at the fillets.

## Figures
- `1_full_geometry_sharp_vs_rounded.png`
- `2_full_mesh.png` (full mesh + shed-band zoom)
- `3_closeups.png` (HV terminal, top dome, one shed, capacitor discs, tank transition)

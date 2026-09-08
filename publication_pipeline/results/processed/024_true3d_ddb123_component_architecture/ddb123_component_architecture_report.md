# Task 024 - DDB-123 Component-Accurate 3D Architecture & Material Audit

Tooling: gmesh-MCP is not a Claude-Code MCP tool server (FastAPI app with its own embedded
LLM; not registered, no tools exposed). Direct gmsh-OCC was used with the user's approval -
deterministic and fully inspectable, which this audit requires.

## 1. Source selection (datasheet / catalogue / image)

- **Primary numerical reference = newer official Arteche DDB/DFK catalogue.** DDB-123:
  123 kV; **standard C = 5600 pF**; high C = 14000 pF; **creepage 3075 mm**; A = 450 mm;
  H = 1830 mm; 300 kg.
- Clean target = **5600 pF** (not 8800 pF). The older leaflet's 8800 pF / 24000 pF is an
  older/conflicting source; 14000 pF high-C option noted, not used.
- **Creepage 3075 mm = hard external-insulation validation target.**
- **74 alternating sheds = IMAGE-DERIVED** (counted off the reference picture; the catalogue
  gives no shed count; uncertainty +-several ribs). Large/small overhangs were TUNED so the
  analytic creepage of 74 alternating sheds = 3078 mm.

## 2. Architecture represented (full 360 deg)

primary terminal (rounded) -> oil-volume compensating HEAD (metal shell + internal oil) ->
porcelain insulator + 74 alternating sheds -> C1 element -> intermediate voltage TAP -> C2
element -> resin HF-terminal exit -> grounded TANK (shell) housing a simplified EMU
(inductive VT + series reactor + auxiliary) in oil -> secondary terminal box + oil valve +
oil-level fittings -> finite air (r = 0.9 m, far-field phi = 0). Electrical core =
homogenized C1/tap/C2 column (eps_r_eff, tap at n_c2/n).

## A. Numerical material-tag proof (bounding-box, from the mesh)

| region | materials present | unexpected | porcelain frac | verdict |
|--------|-------------------|------------|----------------|---------|
| top head bbox | air, metal, oil | none | **0.000** | PASS |
| grounded tank bbox | air, metal, oil, resin | none | **0.000** | PASS |
| secondary-box bbox | metal | none | 0.000 | PASS |
| porcelain/stack column bbox | porcelain, oil, element, metal, air | - | 0.215 | PASS |
| intermediate-tap bbox | metal, oil, (porcelain) | porcelain 0.9 % | 0.009 | PASS* |
| EMU bbox | metal, oil | none | 0.000 | PASS |
| resin HF-exit bbox | resin, metal, oil | none | 0.000 | PASS |

**Head and tank contain ZERO porcelain.** The porcelain-coloured triangles in the *earlier*
head/tank close-ups are a **rendering/framing artifact (type B)**, proven numerically: the old
head close-up window (z >= 1.55) caught porcelain at **z = 1.55-1.6098** (the insulator column
*below* the head; head starts z = 1.61); the old tank window (z <= 0.58) caught porcelain at
**z = 0.5501-0.58** (the column *above* the tank; tank ends z = 0.55). New categorical
close-ups use planar slices and show the head as metal-shell + oil only, the tank as
metal-shell + oil + EMU + resin only - no fragments inside metal.

\* The 0.9 % "porcelain" in the tap bbox is the axis-aligned-bbox corner effect (a square box
around the round r = 0.075 stack reaches r = 0.106 at corners, clipping the wall at r >= 0.095);
it is at the legitimate porcelain-oil interface, below the 2 % threshold, not inside the stack.

## B. Contact / floating-feature check (conformal mesh -> touching = shared nodes)

| feature | expected parent | touches | shared nodes | min dist | status |
|---------|-----------------|---------|--------------|----------|--------|
| primary terminal | head | yes | 28 | 0 | ATTACHED |
| terminal cap | terminal | yes | 105 | 0 | ATTACHED |
| secondary box | tank | yes | 74 | 0 | ATTACHED |
| oil valve | tank | yes | 100 | 0 | ATTACHED |
| **oil level** | tank | **yes (FIXED)** | **144** | **0** | **FIXED -> ATTACHED** |
| resin HF-exit | tank / stack_bottom / base_oil | yes | 114 / 102 / 74 | 0 | ATTACHED |
| EMU / reactor | base_oil | yes | 479 / 239 | 0 | ATTACHED |

The forensic found **oil_level floating (0 shared nodes, 16.9 mm gap)** - the side feature
seen in the external render. It was repositioned to penetrate the tank wall across its full
width and is now ATTACHED (144 shared nodes). All features attached; nothing floats.

## C/D. Cutaway + categorical figures (regenerated clean)

- **True central half-section** (`ddb123_true_half_section_material/_labeled/_closeup_stack/
  _closeup_tank.png`): the solid is clipped by the central y = 0 plane and the camera is placed
  on the *removed* side (auto-detected) so it looks INTO the cut face. Exposes terminal, oil
  head, porcelain + sheds, oil annulus, C1, tap, C2, resin HF-exit, tank, EMU/reactor,
  secondary box. No outside-only view.
- **Categorical material close-ups** (`ddb123_head_closeup_categorical/_tank_closeup_
  categorical/_stack_tap_categorical/_resin_hf_exit_categorical.png`): planar slices, cell-wise
  FLAT categorical colours (lighting off, no scalar interpolation, no smoothing), legend in a
  side panel outside the image, no mixed-material slivers.

## E. Capacitor-stack & component anatomy

| component | tag | z-range [m] | r-range [m] | physical meaning | basis |
|-----------|-----|-------------|-------------|------------------|-------|
| primary terminal | terminal(+cap) | 1.79-1.83 | 0-0.022 | HV terminal (rounded) | catalogue (rounded = assumption) |
| oil-compensating head | head_metal + head_oil | 1.61-1.79 | 0-0.18 | metal shell + internal oil volume | catalogue; internal split simplified |
| porcelain + 74 sheds | porcelain | 0.55-1.61 | 0.095-0.16 | external insulator shell + sheds | creepage official; 74 image-derived |
| oil annulus | oil | 0.55-1.61 | 0.075-0.095 | oil around the stack | schematic (homogenized) |
| **C1 equivalent** | element_c1 | **0.727-1.543** | 0-0.075 | upper capacitor (10/12) | homogenized eps_r_eff |
| **intermediate tap** | foil_tap | **0.720-0.725** | 0-0.074 | C1/C2 junction; Vtap measured here | catalogue (homogenized electrode) |
| **C2 equivalent** | element_c2 | **0.557-0.719** | 0-0.075 | lower capacitor (2/12) | homogenized eps_r_eff |
| HV electrode | stack_top | 1.545-1.549 | 0-0.074 | top stack terminal | schematic electrode |
| ground electrode | stack_bottom | 0.550-0.556 | 0-0.074 | bottom stack terminal | schematic electrode |
| resin HF-exit | resin | 0.521-0.549 | 0-0.044 | HF terminal exit (cap/ind separator) | catalogue; geometry approximate |
| lower oil-filled tank | tank_metal + base_oil | 0-0.55 | 0-0.225 | grounded enclosure + oil | catalogue; equipotential shell |
| inductive VT / EMU | base_emu | 0.10-0.38 | 0-0.12 | electromagnetic unit | SCHEMATIC, EQS-inactive |
| series reactor | reactor | 0.40-0.50 | 0-0.06 | compensating reactor | SCHEMATIC, EQS-inactive |
| auxiliary / ferroresonance | aux_component | 0.05-0.095 | 0-0.05 | aux / ferroresonance circuit | SCHEMATIC, EQS-inactive |
| secondary terminal box | secondary_box | 0.11-0.29 | side, +x | secondary terminals enclosure | SCHEMATIC, EQS-inactive |
| oil valve / level | oil_valve, oil_level | side, +x | - | monitoring accessories | SCHEMATIC, EQS-inactive |

The metal in/near the element layer is the **intermediate voltage tap (foil_tap)** plus the
HV/ground electrode terminals - intentional thin (~5 mm) electrodes at the C1/C2 boundaries.
They define, not corrupt, the C1/C2 regions. Not accidental, not the HF terminal (that is the
resin piece below the stack).

## F. Mesh-quality audit (separated from geometry)

Audit mesh (74 sheds): ~483 k nodes, ~2.7 M tets, ~445 k boundary tris.

| metric | value |
|--------|-------|
| global minSICN min / mean | 0.033 / 0.72 |
| p1 / p5 / p50 / p95 | 0.053 / 0.138 / 0.78 / 0.94 |
| slivers (q < 0.05) | ~15 400 |
| insulator_surface triangles | ~400 k; mean 0.74, **p5 = 0.043, min = 0.043** |

**Worst-100 tets (Python eta), with distances:** 36 air (ON the insulator surface, < 5 mm, at
the shed lips), 32 base_oil (in the tank, ~0.31-0.44 m from the insulator and ~1.3 m from HV),
32 secondary_box (in the external box). **Only 1 of the worst-100 is within 50 mm of an HV
conductor; none are in the capacitor element.**

Impact: the low-quality elements are (a) **on the insulator surface at the shed lips** - the
future surface-conductance / streamer path -> a real limitation; (b) thin oil/box slivers in
the grounded tank and external box - **harmless** (low field, EQS-inert). They do **not**
affect clean capacitance (validated; integral quantity) and do **not** sit at the field
hot-spot (terminal cap).

## Clean 3D electrical validation

Cartesian 3D ComplexEQS, no axisymmetric metric, real tetrahedral solve. UMFPack **direct
failed** on the sliver-y mesh (umf4num error); **BiCGStabl iterative succeeded** (63 s).

| quantity | v4 3D | axisym | delta | vs target |
|----------|-------|--------|-------|-----------|
| capacitance | 5616 pF | 5616 pF | -0.00 % | +0.29 % vs 5600 |
| divider ratio | 0.1665 | 0.1666 | -0.07 % | -0.12 % vs 1/6 |
| stored energy | 14.16 J | 14.16 J | -0.00 % | - |
| tap phase | ~6 mdeg | ~0 | iterative residual ~0 | near zero |

Creepage 3078 mm (+0.09 % vs 3075). Emax at the rounded terminal cap (cap-sensitive; cautious).

## Figures

`ddb123_external_reference_match`, `ddb123_true_half_section_material/_labeled/_closeup_stack/
_closeup_tank`, `ddb123_head_closeup_categorical`, `ddb123_tank_closeup_categorical`,
`ddb123_stack_tap_categorical`, `ddb123_resin_hf_exit_categorical`, `ddb123_material_axial_slice`,
`ddb123_component_cutaway_labeled`, `ddb123_horizontal_slices`, `ddb123_mesh_quality`,
`ddb123_mesh_shed_zoom`, `ddb123_clean_phi`, `ddb123_clean_Emag`. (The earlier clip_box
close-ups are superseded by the categorical/half-section figures.)

## Simplifications / schematic components

Homogenized C1/C2 (no discrete foils); EMU / reactor / auxiliary / secondary box / oil
valve+level = inactive schematic volumes for anatomy (not EM-active in the EQS solve); resin
HF-exit approximate; single insulator section (two-section + mid coupling not modelled);
finite air domain; sharp metal edges except the rounded terminal cap.

## Readiness verdict (separated)

| aspect | status | basis |
|--------|--------|-------|
| External geometry / anatomy | **PASS** | 74 alternating sheds; creepage 3078 mm; all components present; all side features ATTACHED (oil_level fixed) |
| Internal material tagging | **PASS** | bbox audit: zero porcelain in head/tank/box; closeup artifacts proven type-B; stack metal explained |
| Clean capacitance validation mesh | **PASS** | C 5616 pF / ratio 0.1665 validated; worst elements not in the capacitor; only 1/100 near HV |
| Mesh for final 3D surface pollution | **CONDITIONAL** | insulator-surface p5 = 0.043; 36/100 worst tets ON the insulator surface at the shed lips |
| Mesh for localized streamer / hotspot | **CONDITIONAL** | the streamer/pollution path is the shed-lip insulator surface, where the worst elements sit; main Emax (terminal) is clean |

**Verdict: external geometry and material tagging are PASS; the GEOMETRY-VALIDATION mesh is
PASS for clean capacitance; the FINAL PHYSICS mesh is NOT yet ready for quantitative surface
pollution / streamer / hotspot work.** Required before pollution studies:
1. Improve shed-lip insulator-surface mesh (thicker shed tips, reduced lip rounding, or a
   boundary-layer / local surface refinement) until insulator-surface p5 is acceptable.
2. Add deliberate azimuthal mesh grading for an azimuthally-localized patch/streamer.
3. Re-confirm the iterative solver on the finer azimuthally-refined mesh.
4. Implement/validate the 3D Surface Conductance BC vs the 2D result on a uniform case.

Data: material_audit.json, diagnose.json, mesh_quality.json, geometry_parameters.json,
validation.json. Scripts: cvt_ddb123_3d_v4.py, run_true3d_v4.py, audit_ddb123.py,
diagnose_ddb123.py, render_ddb123_clean.py.

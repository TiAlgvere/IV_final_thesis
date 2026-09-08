# Task 023 - True 3D DDB-123 Geometry Fidelity & Mesh-Quality Audit

## gmesh-MCP usage summary (REQUIRED disclosure)

**gmesh-MCP was not usable in this environment, so the work was done with direct
gmsh-OCC Python (user-approved fallback).** Findings before falling back:
- No gmesh tools are exposed to the agent (tool search returns none; the only MCP servers
  wired in are Gmail/Calendar/Drive; there is no `.mcp.json`, and `.claude/settings.local.json`
  holds only permissions).
- The project at `gmesh-mcp/` is **not a Model-Context-Protocol tool server**: it is a
  FastAPI app (`uvicorn`, `fastapi`) with its **own embedded LLM** (`src/llm/{openai,
  anthropic,ollama}`, `langchain`) that turns natural-language prompts into Gmsh scripts.
  Its `requirements.txt` has no `mcp`/`fastmcp`/JSON-RPC dependency.
- Even if connected, it would route geometry construction through a second opaque LLM,
  which is counter to a strict, deterministic geometry/material/mesh audit.

The user chose "Direct gmsh-OCC". All geometry/material/mesh operations below use the
verified gmsh-OCC path; figures use meshio + pyvista; quality stats use gmsh's own
`getElementQualities`.

## Geometry construction method

Full-360 deg solid, built from `geometry/gmsh/cvt_ddb123_3d_v2.py`:
- **Porcelain (outer shell + 27 weather sheds, rounded lips)**: the validated 2D porcelain
  profile - a wall rectangle (outer transitions filleted) plus 27 rounded-lip shed faces
  (the exact `CVTParams.shed_polygon` with `shed_lip_round` fillets) - is built in the x-z
  plane, **fused** into one face, and **revolved 360 deg about z**. So the sheds and lip
  rounding are physically faithful surfaces of revolution, not approximations.
- **Everything else** (homogenized C1/tap/C2 dielectric column, oil, HV head+terminal,
  grounded tank, air) are OCC primitive cylinders.
- A conformal `fragment` partitions the union; each output volume is labelled via the OCC
  fragment MAP + priority (robust for non-convex porcelain/air/oil where a centroid test
  fails). Material rule enforced by priority: stack/oil win inside r<=r_in, porcelain only
  outside - so the inner column can never be tagged porcelain.

## Material-region verification (PASS)

Confirmed visually (axial slice, horizontal slices, shed close-up) and numerically:
- **27 sheds present and tagged porcelain** (counted in the axial slice; creepage matches).
- **Inner column is element + oil, NOT porcelain** - the horizontal slice at z=1.00 m shows
  concentric element(centre) -> oil -> porcelain(outer); porcelain only appears as the
  outer shell and the shed/tip rings.
- **Oil surrounds the stack** (annulus r in [r_stack, r_in] + caps), metals tagged
  separately (tank/head/terminal/stack-end discs/tap), air does not overwrite solids.
- **All physical groups exported**: volumes = stack_bottom, element_c2, foil_tap,
  element_c1, stack_top, head_housing, base_tank, oil, porcelain, air; surfaces =
  hv_electrode, ground_electrode, insulator_surface, farfield.

### Material volume table (geometric, from OCC mass)

| material | volume [m^3] | note |
|----------|--------------|------|
| metal | 0.10616 | tank + head + terminal + stack-end discs + tap |
| element (dielectric) | 0.01735 | homogenized C1+C2 column |
| oil | 0.01238 | interior annulus around the stack |
| porcelain | 0.02885 | outer shell + 27 sheds (only the outer insulator) |
| air | 4.72105 | finite far-field domain |

No unexplained overlaps; interior (oil+element = 0.0297 m^3) and shell (porcelain) are
spatially distinct. No core region mislabeled as porcelain.

## Datasheet geometry checks

| quantity | value used | datasheet / target | provenance |
|----------|-----------|--------------------|------------|
| total height | 1.830 m | 1830 mm | datasheet |
| base width (tank diameter) | 0.450 m | 450 mm | datasheet |
| **creepage** | **3062 mm** | **3075 mm (-0.4 %)** | datasheet target - PASS (<1 %) |
| number of sheds | 27 | 27 | datasheet |
| shed pitch | 39.3 mm | - | derived (insulator height / 27) |
| shed overhang | 40 mm | - | inherited from validated axisymmetric model |
| shed lip round | 3 mm | - | inherited (Task 013) |
| head dome | r 0.18 / h 0.18 m | - | inherited / estimated |
| HV terminal | r 0.02 / h 0.04 m | - | inherited / estimated |
| tank | r 0.225 / h 0.55 m | base width datasheet; height estimated | mixed |
| capacitance (after solve) | 5612 pF | 5600 pF (+0.2 %) | datasheet target - compatible |
| eps_r_eff | 35146 | - | derived from rated C (homogenization) |

Datasheet-anchored: height, base width, creepage, 27 sheds, 5600 pF. Estimated / inherited
from the validated axisymmetric `CVTParams`: stack radius, disc/porcelain thicknesses, shed
profile, head/terminal/tank internal dimensions, eps_r_eff. No geometry feature was silently
dropped.

## Mesh-quality audit (PASS)

Audit mesh (controlled refinement: curvature-driven at the rounded lips; per-region sizing
finer on porcelain/sheds and stack interfaces):

| metric | value |
|--------|-------|
| nodes | 295 015 |
| tetrahedra | 1 700 804 |
| boundary triangles | 295 500 |
| min quality (minSICN) | **0.117** (no negative/invalid elements) |
| mean quality | 0.806 |
| percentiles p1 / p5 / p50 / p95 | 0.517 / 0.624 / 0.816 / 0.950 |
| slivers (q < 0.05) | **0** |
| insulator_surface triangles | 266 082; min 0.534, mean 0.934, p5 0.822 |
| mesh size: sheds / porcelain | ~0.016 m (+ curvature refinement at 3 mm lips) |
| mesh size: stack interfaces | 0.016-0.020 m |
| mesh size: terminal / head / tank | ~0.045-0.055 m |
| mesh size: far air | 0.25 m |

Worst 20 elements: all in **air at the porcelain-wall / metal junctions** (r~0.119 m,
z = 1.61 m top and z = 0.55 m bottom), quality 0.117-0.13 - i.e. benign slivers at re-entrant
air corners, **not** in the high-field dielectric/creepage regions. No severe slivers in
high-field regions. Surface mesh on the sheds is fine and high-quality (insulator p5 = 0.82),
so the mesh **is suitable for a later Surface Conductance BC**.

## Clean 3D electrical validation (PASS)

To keep a direct solve in memory, the SAME 27-shed geometry was meshed coarser for the solve
(mesh-refinement control; sheds barely affect the clean-baseline C). Cartesian 3D,
ComplexEQS, **no axisymmetric metric**, real tetrahedral solve.

- Solve mesh: 86 400 nodes / 482 426 tets, qmin 0.125, creepage 3062 mm.
- UMFPack **direct**, solved in **178 s**, no instability.

| quantity | v2 3D (sheds) | axisymmetric | delta | Task 022 3D |
|----------|---------------|--------------|-------|-------------|
| capacitance | 5612 pF | 5616 pF | **-0.08 %** | 5622 pF |
| divider ratio | 0.1665 | 0.1666 | **-0.03 %** | 0.1665 |
| \|Vtap\| | 11 826 V | 11 830 V | -0.03 % | 11 830 V |
| tap phase | -0.008 mdeg | -0.122 mdeg | both ~0 | -0.012 mdeg |
| stored energy | 14.15 J | 14.16 J | -0.08 % | 14.17 J |

Capacitance within 0.08 % of axisymmetric / +0.2 % of datasheet (target was 2-5 %, <1 %
preferred - PASS). Divider ratio within 0.03 % (target 1 % - PASS). Clean phase ~0 (PASS).
Emax = 7.6e5 V/m at (r,z) = (0.188, 1.80) m, the HV head-dome edge - corner-sensitive (the
head/terminal are not filleted in this homogenized 3D model), reported cautiously.

Figures (all from the real 3D mesh/solution, never revolved): `true3d_clean_phi.png`,
`true3d_clean_Emag.png`, `true3d_clean_cutaway.png`, `true3d_clean_hotspot.png`; geometry /
material / mesh figures as listed in the task.

## Runtime / memory notes

- Fine audit mesh build (1.7M tets): ~5 min. Coarse solve mesh build: ~3 min.
- Direct solve of 482 k tets: 178 s, fit in RAM. A direct solve of the full 1.7 M-tet audit
  mesh would not fit comfortably - it needs the iterative path (BiCGStabl+ILU, wired but not
  yet exercised since direct succeeded on the coarse mesh).

## What is now physically faithful

- 27 weather sheds as true surfaces of revolution, with rounded lips.
- Porcelain = outer shell + sheds only; oil and capacitor element fill the interior.
- Datasheet creepage (3062 vs 3075 mm), height, base width, shed count.
- Capacitance / divider ratio reproduced in true 3D.
- High-quality tet mesh with a fine, clean insulator surface.

## What remains approximate

- **Homogenized stack** (single C1/tap/C2 dielectric column; discrete interior foils not
  resolved) - permitted; reproduces C and ratio.
- **Finite air domain** (r = 0.8 m) instead of an open far field - negligible for C.
- **Sharp metal edges** on the head/terminal/tank (not filleted) -> Emax is corner-sensitive.
- Two mesh densities (fine audit / coarse solve); a single fine-mesh solve awaits the
  iterative path.

## Readiness

- **3D surface pollution (uniform/axisymmetric): READY.** `insulator_surface` exists with a
  fine, high-quality triangle mesh; the verified 2D Surface Conductance BC can be applied and
  cross-checked against the 2D result.
- **Azimuthally-localized streamer / sector pollution: GEOMETRY READY, MESH/SOLVER NOT YET.**
  The full-360 shed geometry can host an azimuthal patch, but a localized fault needs (a)
  deliberate **azimuthal mesh grading** at the patch edges (the current mesh is azimuthally
  uniform) and (b) a **working iterative solver** for the resulting fine asymmetric mesh
  (direct will not fit). These are the prerequisites before any localized-fault study.

## Verdict

Geometry, material assignment and mesh quality **pass**. The model is faithful enough for
uniform 3D surface-pollution work. Localized/azimuthal studies require azimuthal mesh control
and a validated iterative solve first - do not proceed to localized pollution until those are
in place.

Scripts: `geometry/gmsh/cvt_ddb123_3d_v2.py`, `scripts/run_true3d_v2.py`,
`scripts/audit_true3d.py`, `scripts/viz_clean_v2.py`. Data: `audit_mesh_meta.json`,
`solve_mesh_meta.json`, `validation_v2.json`.

# ct-fem — Parametric FEM framework for HV current-transformer condition monitoring

A reproducible 2-D **axisymmetric electro-quasistatic (EQS)** finite-element
framework for high-voltage **oil-paper insulated current transformers (CTs)**.
It models a hairpin / top-core HV CT (≈245 kV class), injects parametrized
insulation defects, and computes how **externally measurable electrical
quantities** change:

* terminal capacitance **C1**,
* dissipation factor **tan δ**,
* leakage-current magnitude / phase (complex terminal admittance **Y**),
* the **grading-foil potential ladder** and **per-gap field stress**.

The output is a tidy, provenance-stamped dataset of physics-grounded fault
signatures for predictive diagnostics and downstream ML.

> **Scope.** Only *electrically measurable* indicators are modelled. DGA /
> oil-chemistry diagnostics are out of scope. The **magnetic / metrological
> behaviour of the CT cores is intentionally NOT modelled here** — that is a
> separate lumped-circuit work package. External surface phenomena (porcelain
> sheds, pollution flashover) are also out of scope in v1 (a TODO hook is left
> in the geometry).

---

## 1. Physics model and its limits

### Formulation
We solve the complex scalar potential `φ` of the axisymmetric EQS problem

```
div( (σ + jω ε0 εr) ∇φ ) = 0                                   (1)
```

in the `(r, z)` half-plane, with the axisymmetric volume measure
`dV = 2π r dr dz`. Writing the per-material complex coefficient

```
κ = σ + jω ε0 εr (1 − j·tanδ)                                   (2)
```

the weak form is

```
a(φ, v) = ∫_Ω κ ∇φ·∇v (2π r) dr dz = 0.                         (3)
```

* Element: **Lagrange P2** on triangles, **complex** scalars (requires a complex
  PETSc build — see install).
* `ω = 2π·50 Hz` (default, parameter).
* BCs: `φ = U0 = Um/√3` (≈141 kV RMS for Um=245 kV) on the **HV electrode**
  (primary conductor + head); `φ = 0` on the **grounded** base/sleeve and the
  **far-field** boundary; **natural** (zero normal current) on the symmetry axis
  `r = 0` (the `2π r` weight makes this automatic — the axis is simply excluded
  from the Dirichlet sets).
* **Grading foils float.** Each foil is a thin domain with aluminium-like
  conductivity (σ=3.5e7 S/m); in the EQS formulation it settles at its natural
  potential with no constraint. This is deliberate — *not* replaced by multipoint
  constraints.

Reference for the EQS weak form: Haus & Melcher, *Electromagnetic Fields and
Energy*, Ch. 7, plus the standard axisymmetric Jacobian factor `2π r`.

### Observables
Terminal admittance is computed **two independent ways and cross-checked**
(`ctfem/observables.py`):

* **(A) energy/power functional** `Y = (1/U0²) ∫_Ω κ (∇φ·∇φ) dV` (non-conjugated
  product);
* **(B) reaction/flux** `I = Σ_{HV dofs} a(φ, basis_i)` — the discrete boundary
  flux `∮_HV κ ∇φ·n dΓ`.

These are algebraically identical (`φᵀAφ = U0·I`); the relative discrepancy is
reported as a numerical sanity check. Then `C1 = Im(Y)/ω`, `tanδ = Re(Y)/Im(Y)`.

### Known limitations / modelling choices
* The geometry is a **smooth, simplified** axisymmetric idealisation of the real
  device (single equivalent axial primary conductor instead of the true hairpin,
  smooth porcelain cone without sheds, thin foil annuli of fictitious 0.15 mm
  thickness — the spec explicitly says *do not* model true foil thickness).
* The **C1 ground reference** is a grounded vertical *sleeve* just outside the
  paper body (named `base_tank`), representing the real flange/last-foil ground.
  Without it the conductor-to-ground capacitance would be unphysically small;
  with it C1 lands in the realistic hundreds-of-pF range.
* Foil thickness is sub-element by design → thin sliver cells in the foils are
  expected and acceptable (the foils are equipotential conductors).
* Defects are injected at the **material level** (per-cell κ), so no remeshing is
  needed for any defect.

---

## 2. Install

Two interchangeable FEM backends implement the same physics and observables;
`detect_backend("auto")` picks whichever is available, so all scripts run
unmodified on a Windows laptop and on a Linux cluster.

### Windows-native (default on a laptop) — scikit-fem backend
```powershell
pip install numpy scipy gmsh scikit-fem meshio pandas pyarrow matplotlib pytest
```
Pure Python (scipy sparse LU, complex-native). The full pipeline — Phases 1–5,
all tests, plots — runs natively on Windows with this alone.

### Linux/WSL/HPC (optional, for large 3-D + MPI) — DOLFINx backend
DOLFINx + a **complex** PETSc build:
```bash
mamba env create -f environment.yml      # creates env "ctfem"
# or explicitly:
mamba create -n ctfem -c conda-forge python=3.11 fenics-dolfinx "petsc=*=complex*" \
    gmsh python-gmsh numpy scipy pandas pyarrow pyvista matplotlib pytest pyyaml mpi4py
```
The DOLFINx solver asserts a complex build at import and fails clearly on a
real-valued PETSc. Force a backend with `--backend skfem|dolfinx` on any script.

### Numerical note: metal conductivity cap
Physical aluminium (σ=3.5e7 S/m) against dielectric κ≈1e-8 gives a matrix
contrast of ~10¹⁵ — beyond double precision, producing *spurious* dissipation
(observed tan δ ≈ 6.5 instead of ≈0.002). The coefficient builder therefore caps
metal σ at `SIGMA_NUMERICAL_CAP = 1 S/m` (contrast ~10⁸): foils still float as
perfect equipotentials (relative error ~1e-8) but the matrix stays well-
conditioned. See `ctfem/materials.py`.

---

## 3. Run each phase

Each script writes to a timestamped folder under `results/` (gitignored).

```bash
# Phase 1 — parametric geometry (CT + degenerate coax), stats, optional PNG
python scripts/phase1_geometry.py --preset coarse --png

# Phase 2 — solver validation: coax analytic benchmark + CT mesh convergence
python scripts/phase2_validate.py

# Phase 3 — healthy baseline solve: C1, tanδ, foil ladder, field stress, plots
python scripts/phase3_baseline.py --preset coarse --png

# Phase 4 — defect studies: ΔC1, Δtanδ, ladder distortion, field redistribution
python scripts/phase4_defects.py --preset coarse

# Phase 5 — sweep -> dataset (>=200 rows), parallel + resumable
python scripts/phase5_sweep.py --out results/dataset.parquet --workers 8
python scripts/phase5_sweep.py --design lhs --n 256 --workers 8   # Latin-hypercube

# Phase 7 — Arteche DDB-123 CVT (110 kV Estonia): nameplate validation + defects
python scripts/phase7_cvt.py                      # 2-D healthy + defect set
python scripts/phase7_cvt.py --solve-3d --show    # + interactive 3-D window
```

Mesh presets: `coarse | medium | fine | ultrafine` (global refinement factor).
`coarse` keeps a full Phase-3 CT solve under ~2 min on a laptop (~30k triangles).

### Tests
```bash
pytest            # < 2 min; includes full solver validation via scikit-fem
```
The scikit-fem solver tests (analytic coax benchmark, admittance identity,
CT baseline smoke) run everywhere, including Windows. The DOLFINx variants
self-skip where a complex DOLFINx build is unavailable.

### Measured results (Windows, scikit-fem backend, coarse preset)
* Coax benchmark: C error **0.07 %**, tan δ error **0.05 %** (gate: 0.5 %);
  energy-vs-reaction admittance discrepancy ~1e-14 (exact identity).
* Healthy CT baseline: **C1 ≈ 342 pF**, tan δ ≈ 0.0018, near-uniform foil
  ladder (≈0.075 |φ|/U0 per step), peak gap stress ≈ 3 kV/mm at the conductor.
* Convergence: C1 changes 0.75 % between the two finest of three levels.
* Defect signatures (vs healthy): shorted foil → ΔC1 +3–4 % with Δtan δ ≈ 1e-4;
  moisture (sev 1.0) → tan δ ×6.5 and ΔC1 +6 %; aging (sev 1.0) → tan δ → 0.026;
  oil contamination → tan δ → 0.044 with ΔC1 ≈ 0.2 %.

---

## 4. Repository layout

```
ct-fem/
  environment.yml
  ctfem/
    config.py        # dataclasses: GeometryParams, OperatingParams, MaterialParams, DefectSpec, CaseConfig
    materials.py     # complex-permittivity material DB + (σ + jωε) helper
    geometry.py      # gmsh parametric builder -> .msh + name->tag sidecar; coax validator
    eqs_solver.py    # axisymmetric EQS solver (DOLFINx, complex PETSc)
    observables.py   # terminal admittance -> C1, tanδ; foil ladder; field stress
    defects.py       # material-level defect injection (per cell)
    sweep.py         # Cartesian/LHS sweep engine -> parquet/CSV with provenance
    validate.py      # coax analytic benchmark + mesh-convergence study
    viz.py           # optional PyVista/matplotlib plotting (headless)
    util.py          # results dirs, JSON dump
  scripts/           # phase1..5 CLI entry points
  tests/             # pytest (coax-only, fast)
  results/           # gitignored
```

---

## 5. Changing device dimensions (when real Elering data arrives)

All geometry lives in `GeometryParams` (`ctfem/config.py`) with defaults matching
the reference Arteche/VA TECH hairpin CT. Edit the field defaults, or pass a
populated `GeometryParams` to the builder / `CaseConfig`. Key fields:

| field | meaning |
|---|---|
| `total_height`, `base_height`, `porcelain_height`, `head_height`, `head_radius`, `base_radius` | overall envelope |
| `conductor_radius` | equivalent axial primary radius |
| `paper_inner_radius`, `paper_outer_radius_base/head`, `paper_z_bottom/top` | conical condenser body |
| `n_foils`, `foil_placement` (`equal_capacitance`/`equal_spacing`), `foil_thickness`, `foil_edge_margin` | grading foils |
| `ground_sleeve_radius`, `ground_sleeve_thickness` | the grounded C1 reference |
| `oil_outer_radius`, `porcelain_thickness`, `farfield_factor` | oil/porcelain/air |
| `lc_*`, `mesh_refinement`, `refine_dist_*` | meshing |

Nothing else needs to change — the builder, solver, observables and sweep all key
off the same dataclass. The mesh is reproducible from the config (deterministic
gmsh build); sampling (LHS) is seeded.

---

## 6. Adding a new `DefectSpec`

1. Add the kind to the `DefectKind` literal and any new fields to `DefectSpec`
   in `ctfem/config.py`; give it a branch in `DefectSpec.label()`.
2. Implement the per-cell injection in `ctfem/defects.apply_defect` — you receive
   cell centroids `(r, z)`, the region name per cell, the baseline κ array, the
   material DB, the operating point and the geometry; return a modified κ array.
   Use the helper `_kappa_from(eps_r, tanδ, σ, ω)` to build coefficients.
3. (Optional) add it to the Phase-4 list and the Phase-5 design.
4. Add a numpy-level unit test in `tests/test_defects.py` (no DOLFINx needed).

Implemented defects: `shorted_foil` (metal-σ bridge across a paper gap —
`index=k` bridges the gap on the inner side of foil *k*, i.e. between foil *k−1*
or the conductor and foil *k*), `moisture_ingress` (local wet-paper patch, εr &
tanδ up), `global_aging` (uniform paper tanδ rise + εr drift), `oil_contamination`
(oil σ up).

---

## 7. Dataset schema (Phase 5)

One row per case (`ctfem/sweep.solve_case`), written to parquet (CSV fallback):

* **provenance**: `case_id, name, defect_kind, severity, defect_index, z_center,
  extent, n_foils, foil_placement, mesh_refinement, frequency, um_kv,
  temperature_c, git_hash, n_triangles, n_nodes, solve_seconds`
* **observables**: `C1_pF, tan_delta, Y_real, Y_imag, admittance_discrepancy,
  peak_field_overall, foil{1..N}_frac, gap{1..N}_peakE`

The sweep is **resumable** (already-completed `case_id`s in the output file are
skipped) and **parallel** (`--workers` via multiprocessing; each worker builds
its own mesh and runs one serial solve — embarrassingly parallel, HPC-array-job
friendly). A temperature-dependent `tanδ(T)` hook lives in `Material.at_temperature`.

---

## 8. 3-D model (revolved)

A full 3-D device is generated by **revolving the same parametric (r, z) section
360°** about the vertical axis (`ctfem/geometry3d.py`, `build_ct_3d`). Because it
is geometrically identical to the 2-D section, under symmetric excitation the 3-D
solution **equals** the axisymmetric one — so the 2-D coax benchmark and CT
baseline also validate the 3-D model. Symmetry-breaking defects (a one-sided
moisture pocket, a localized delamination) can later be injected as azimuthally
localized material patches.

```powershell
# build + view on Windows (gmsh only; no DOLFINx needed)
python scripts/phase1_geometry3d.py --preset coarse --n-foils 6 --gui
python scripts/phase1_geometry3d.py --wedge 30          # cheap 30° test sector
```

### Interactive 3-D FEM (solve + explore) — runs on Windows
`scripts/phase6_interactive3d.py` solves the full 3-D EQS problem with the
scikit-fem backend (P1 tets, scipy direct solver) and opens an **interactive
PyVista window**: rotate/zoom with the mouse and **drag a clip plane** through
the device; the defect region is shown in red inside the potential (or |E|)
field. It also cross-checks the healthy 3-D C1/tan δ against the 2-D
axisymmetric reference, and writes a ParaView-compatible `.vtu`.

```powershell
# one-sided 60° moisture pocket, interactive window:
python scripts/phase6_interactive3d.py --kind moisture_ingress --severity 1.0 `
    --theta-deg 0 --defect-wedge-deg 60 --show
python scripts/phase6_interactive3d.py --field E --show    # |E| on the clip plane
```
Laptop guidance: default `--n-foils 4 --refine 0.2` is ~0.5 M tets and solves in
minutes; scale up gradually. The `.vtu` opens in ParaView for full
post-processing (isosurfaces, volume rendering, probes).

**Cost reality (measured).** The 0.15 mm 2-D foils are revolved as ~1.5 mm shells
(`foil_thickness_3d`; thickness is fictitious, C is set by foil *radius*). Even
so a full 360° device is **millions of tetrahedra**: a 45° / 3-foil / coarse
wedge is already ~0.5 M tets, i.e. a full 10-foil device is tens of millions.
The 3-D EQS solve (`ctfem/solver3d.py`) is therefore an **HPC / WSL job** — it
uses an iterative CG+hypre solver and cannot run on native Windows. Recommended
split: keep the fast, validated **2-D engine for the ML sweep dataset and
symmetric defects**, and use the **3-D model for realism plots and
symmetry-breaking studies**.

The 3-D solver/observables (`solver3d.solve_msh_3d`, `compute_observables_3d`)
mirror the 2-D math exactly but drop the `2πr` weight (true Cartesian volume) and
impose BCs on electrode *surfaces*. The defect engine is reused unchanged — the
solver feeds it each cell's cylindrical `(r=hypot(x,y), z)` centroid.

### Azimuthal (one-sided) defects — the reason 3-D earns its cost
`DefectSpec` has `theta_center` / `theta_extent` (radians). The default
(`theta_extent = 2π`) is a full ring = the rotationally-symmetric case the 2-D
model reproduces. Set `theta_extent < 2π` for a **localized** defect — a moisture
pocket or delamination on one side of the core — which axisymmetric 2-D
physically cannot represent. The 2-D solver ignores these fields; the 3-D solver
applies them via each cell's azimuth.

You can **see the localized defect in 3-D on Windows without any solve** — the
3-D builder overlays a per-cell indicator (reusing the exact `apply_defect` the
solver uses) as a gmsh view:

```powershell
# 60-degree moisture pocket on one side, mid-height, viewed in gmsh
python scripts/phase4_defect3d_view.py --kind moisture_ingress `
    --severity 1.0 --z 1.45 --theta-deg 0 --defect-wedge-deg 60 --n-foils 4 --gui
```
This writes `ct3d.msh` + `ct3d_defect.pos`; the `.pos` view highlights exactly
the cells the defect alters.

> Status: the 3-D **geometry** is built and tested on Windows (full-revolve
> regression test in `tests/test_geometry3d.py`). The 3-D **solver** is written
> to mirror the validated 2-D forms but must be run/validated under WSL/HPC (no
> Windows DOLFINx). Validate it by checking that an azimuthally-uniform case
> reproduces the 2-D C1/tan δ.

## 9. Expected qualitative behaviour (Phase 4)

* **ShortedFoil** → a step rise in C1 (the shorted gap drops out of the series
  capacitive ladder) and increased field stress in the neighbouring gaps; the
  foil ladder shows a flat (merged-potential) step.
* **MoistureIngress / GlobalAging** → tan δ rises with only a small ΔC1.
* **OilContamination** → loss rise concentrated in the oil path.

---

## 10. Second device: Arteche DDB-123 CVT (Phase 7)

`CVTParams` (config.py) + `build_cvt` / `build_cvt_3d` model the **capacitive
voltage transformer** used on the 110 kV side in Estonia (ARTECHE DDB/DFK
datasheet): a stack of series capacitor elements inside a porcelain insulator
on the grounded EMU tank, forming the C1/C2 divider with the intermediate
voltage tap.  Datasheet anchors: H = 1830 mm, base A = 450 mm, **rated standard
capacitance 5600 pF**, and **standard creepage distance 3075 mm** — hard
validation targets the CT never had.

**External profile detail.**  The insulator carries parametric **weather
sheds** (`n_sheds` drooping trapezoid ribs, region `porcelain_shed` with its
own finer `lc_shed`); `CVTParams.creepage_distance()` walks the revolved
surface profile and reproduces the datasheet 3075 mm within ~0.5 % at the
defaults (27 sheds, 40 mm overhang).  The HV head is the wide oil-compensator
**dome** (Ø360 mm — it shades the shed tips corona-ring fashion) plus the
primary terminal stub.  The **secondary terminal box** on the tank side breaks
rotational symmetry and therefore exists only in the 3-D build (extra-solids
hook in `build_ct_3d`; fused into the grounded tank by the fragment); the 2-D
axisymmetric model cannot represent it by definition.  Surface flashover /
pollution layers remain out of scope (EQS bulk model).

**Homogenized elements.** A real element is a *wound* paper-film capacitor; its
turns cannot be meshed.  Each element is a solid dielectric block between two
electrode discs with `eps_r_eff` chosen so the element carries its rated series
capacitance in its real volume (`CVTParams.element_epsr_eff()`).  This
preserves terminal C, tan δ, divider distribution and external fields; it does
not resolve fields inside a wound can (out of scope).  The interior discs are
named `foil_k` so the solver's potential-ladder observable **is** the divider
ladder; the tap is disc `tap_disc_index`.

**Measured (2-D, refine 1.0, ~23k tris):** healthy C = 5563.5 pF (**−0.65 %**
vs nameplate), tan δ = 0.0020 (= element dielectric), tap = 0.1664 vs nominal
2/12 (−0.18 %), i.e. 11.8 kV intermediate voltage at U0 = 71 kV.

CVT defect kinds (defects.py): **shorted_element** — the classic CVT failure:
C jumps by N/(N−1) (+9.1 % measured) wherever the short is, but the **ratio
error signs the section**: +9.2 % for a short in C1, −45.6 % for a short in the
(short, 2-element) C2 — and the disc ladder localizes the exact element.
**element_aging** — tan δ 0.002 → 0.02 at severity 1 with C unchanged.  Oil
contamination barely registers (the series path is inside the elements, unlike
the CT) — itself a discriminating feature between device architectures.

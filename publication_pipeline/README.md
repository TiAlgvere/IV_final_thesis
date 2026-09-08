# Publication Pipeline (ElmerFEM)

The publication-oriented ElmerFEM workflow for the thesis. The legacy scikit-fem
prototype in the repository root (`ctfem/`, `scripts/phase*.py`) remains unchanged and
is kept for development, exploration, and cross-validation. This pipeline is
intentionally isolated so the publication workflow can be stabilised without
rewriting the earlier prototype.

Workflow:

1. Gmsh geometry generation (`geometry/gmsh/`)
2. ElmerGrid conversion from `.msh` to Elmer mesh format
3. ElmerSolver execution from a SIF template (`elmer/sif_templates/`)
4. Python post-processing into a study record under `results/processed/`

---

## Prerequisites

ElmerFEM is external native software and is **not** installed through `pip`.

```powershell
# repo root
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dashboard,viz,sweep,dev]"

# point at the local Elmer install (see .env.example)
$env:ELMER_HOME = "C:\ElmerFEM-gui-nompi-Windows-AMD64"

# verify Elmer + Python deps resolve
python publication_pipeline/scripts/check_environment.py
```

The complex-EQS solver is a custom Elmer module and must be compiled once per
machine. The `elmerf90` shipped with the Windows build has hardcoded paths and does
not work; `build_solver.py` invokes the bundled gfortran directly instead:

```powershell
python publication_pipeline/scripts/build_solver.py
```

See `ENVIRONMENT.md` for the full environment notes.

---

## What is in version control, and what is not

| | Tracked | Regenerate with |
|---|---|---|
| Source, SIF templates, `solvers/ComplexEQS.F90` | yes | — |
| `results/processed/` — study reports, figures, observables | yes | the study's `run_*.py` |
| `results/raw/` — Elmer meshes, VTUs, solver logs (~3.2 GB) | **no** | the study's `run_*.py` |
| Compiled `.dll` solver modules | **no** | `scripts/build_solver.py` |
| `mesh/gmsh/clean_baseline.msh` (6 kB reference mesh) | yes | `scripts/build_mesh.py` |

A fresh clone therefore carries the complete **scientific record** (every report,
figure and observable) but none of the bulk solver output. Scripts that consume a
raw mesh — notably the sector studies, which read
`results/raw/025_insulator_mesh_remediation/cvt3d_v5_solve.msh` — need that study
re-run first:

```powershell
python publication_pipeline/scripts/run_true3d_v5.py     # regenerates the 025 solve mesh
python publication_pipeline/scripts/run_fixed_sector_sweep.py
```

---

## Study index

Each study writes a dated run directory under `results/raw/<study>/` and a curated
report plus figures under `results/processed/<study>/`. Reports are the primary
record — read those first.

| Study | Subject |
|---|---|
| `001_clean_baseline` | Clean mesh + solver handoff |
| `002_numerical_verification` | Mesh convergence (coarse → very fine) |
| `006_pollution_sweep` | Uniform surface-pollution σ sweep |
| `007_complex_eqs_verification` | Complex κ = σ + jωε solver verification |
| `008_pollution_geometry_audit` | Pollution-layer geometry audit |
| `009_representative_cvt` | Representative Arteche DDB-123 CVT |
| `010_cvt_pollution` | CVT pollution response |
| `011_cvt_mesh_audit` | CVT mesh quality audit |
| `012_cvt_visuals` | CVT field/potential visuals |
| `013_surface_conductance` | Surface-conductance formulation |
| `0135_geometry_audit` | Shed-profile geometry audit |
| `0136_surface_boundary` | Surface boundary-condition study |
| `014_3d_render` / `0141_top_artifact` | 3-D rendering, top-artifact resolution |
| `015_localized_pollution` | Localized (Type B) pollution patch |
| `016_stack_faults` | Capacitor-stack fault set |
| `017_diagnostic_matrix` | Fault → observable diagnostic matrix |
| `018_trajectories` | Degradation state-space trajectories |
| `019_internal_severity_sweeps` | Internal-fault severity sweeps |
| `020_noise_detectability` | Detectability under instrument noise |
| `021_electrothermal` / `021_5_thermal_3d` | Electrothermal coupling, 3-D thermal field |
| `022_true3d_foundation` | True-3D foundation model |
| `023_true3d_geometry_fidelity` | Shed-fidelity upgrade |
| `024_true3d_ddb123_component_architecture` | Component-accurate DDB-123 (v4) |
| `025_insulator_mesh_remediation` | Insulator mesh remediation → accepted v5 solve mesh |
| `026_azimuthal_surface_conductance_validation` | Azimuthal sector validation (MATC θ-window) |
| `027_fixed_sector_sigma_sweep` | Fixed-sector σ_s sweep |

`studies/<name>/TASK.md` holds the original task brief where one was written.

---

## Geometry lineage

`geometry/gmsh/` keeps the full 3-D model lineage rather than only the latest
revision, because several superseded variants are cited by the study reports:

- `clean_baseline.py` — 2-D axisymmetric baseline
- `cvt_ddb123.py` — validated `CVTParams` (ε_r_eff, shed profile, creepage); reused by every 3-D variant
- `cvt_ddb123_3d.py` → `_v2` → `_v3` → `_v4` → `_v5` — progressive 3-D fidelity; **v4** is the accepted architecture of study 024, **v5** produces the accepted solve mesh of study 025
- `cvt_ddb123_3d_v5_sector.py` — explicit angular-split attempt for study 026. **Retained deliberately as a documented negative result**: the OCC `fragment` on the split geometry proved computationally intractable, so study 026 adopted the MATC θ-window instead. See that study's report.

Do not prune these without re-reading the report that cites them.

---

## Related tooling

`gmesh-mcp/` is a third-party Gmsh MCP server, not part of this thesis and not
tracked here. Re-clone it beside the repo if needed:

```powershell
git clone https://github.com/rishabh10gpt/gmesh-mcp.git
```

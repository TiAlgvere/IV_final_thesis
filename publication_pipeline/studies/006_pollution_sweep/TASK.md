# Task 006 - Conductive-Pollution Conductivity Sweep (complex EQS)

## Goal
Introduce the first non-trivial physical perturbation - uniform conductive
pollution - and quantify how the clean diagnostic observables change with the
pollution conductivity sigma.

## Formulation
Time-harmonic electroquasistatic solve with a complex coefficient at power
frequency (50 Hz):

    div( (sigma + i*omega*eps0*eps_r) grad(phi) ) = 0

Pollution does nothing in a real electrostatic solve, so this is realised with a
**custom Elmer solver** (`elmer/solvers/ComplexEQS.F90`). The complex coefficient
kappa = sigma + i*omega*eps couples conduction and displacement, which is what
makes the pollution measurably perturb the capacitive divider.

## Pollution model (approximation, documented)
A thin **conductive volume strip** on the outer porcelain face
(`POLLUTION_LAYER` body, default thickness 5 mm), present in the mesh for the
whole sweep. Only its `Electric Conductivity` is varied; its relative
permittivity equals porcelain's, so at sigma = 0 the case is identical to clean.
This is a first-order stand-in for true surface conductivity, NOT exact surface
conductivity.

## Quantities tracked
- |Vtap| and relative shift vs clean
- phase displacement d-theta (HV phase = 0 reference)
- loss tangent tan(delta) = int(sigma|E|^2) / (omega int(eps|E|^2))
- leakage current: resistive/in-phase component and total terminal magnitude
- peak field Emax = max|E|

## Acceptance criteria (all met)
- ComplexEQS builds via `scripts/build_solver.py` and loads in ElmerSolver.
- sigma = 0 reproduces the analytical capacitive divider |Vtap| (rel err ~ 1e-11)
  with phase ~ 0, tan(delta) ~ 0 - i.e. the complex solver reduces to the
  validated real baseline (study 001).
- A measurable, monotone |Vtap| depression with sigma, plus an interior
  loss-relaxation peak in tan(delta) and phase. (`scripts/validate_complex_solver.py`
  passes all checks.)

## Key findings
- **Phase displacement and tan(delta) are the earliest indicators.** At
  sigma = 1e-8 S/m (very light pollution) |Vtap| has moved only ~0.15 %, yet
  d-theta ~ 1.3 deg and tan(delta) ~ 5e-3 are already clearly measurable.
- Both d-theta and tan(delta) **peak at an intermediate sigma** (~3e-7 .. 1e-6 S/m,
  where sigma ~ omega*eps - the Maxwell-Wagner / loss-relaxation transition) and
  fall off again at high sigma. |Vtap| instead saturates monotonically.
- Emax rises through the transition and saturates as the layer becomes
  conductive.

## Caveat on magnitudes
The verification geometry is intentionally simple: the pollution strip spans the
full porcelain height on the outer face and therefore couples the tap interface
toward ground quite directly, so the **absolute** |Vtap| change (~-65 % at high
sigma) is larger than a real CVT would show. The **qualitative signatures** (which
observable moves first, the relaxation peak in d-theta / tan(delta)) are the
deliverable; absolute magnitudes await a representative CVT geometry.

## Out of scope (per project philosophy)
- Thermal, moisture, streamer/partial-discharge, or any coupled multiphysics
- Non-uniform / patchy pollution
- Mesh-convergence (GCI) of the polluted case
- A representative full CVT geometry

## How to run
```powershell
# 1. build the solver (only needed once / after editing ComplexEQS.F90)
python publication_pipeline/scripts/build_solver.py

# 2. run the sweep (default: sigma = 0 plus logspace(1e-10 .. 1e-3), 16 points)
python publication_pipeline/scripts/run_pollution_sweep.py

# 3. plot + validate
python publication_pipeline/scripts/plot_sweep.py <results/raw/.../pollution_sweep.csv>
python publication_pipeline/scripts/validate_complex_solver.py
```
Outputs land under `results/{raw,processed}/006_pollution_sweep/run_<timestamp>/`.

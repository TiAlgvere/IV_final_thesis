# Task 011 - CVT Pollution Mesh-Convergence Sanity Audit

Pollution-layer DDB-123 at coarse / medium / fine refinement (mesh_refinement
0.5 / 1.0 / 1.8), two cases: clean and sigma=1e-5 S/m (the Task 010 critical
point). Sanity check: are the observables already stable at the medium mesh?

| level | refinement | nodes | tris |
|-------|-----------|-------|------|
| coarse | 0.5 | 5766 | 11281 |
| medium | 1.0 | 16388 | 32292 |
| fine | 1.8 | 49017 | 97186 |

## Case sigma = 0e+00 S/m

| observable | coarse | medium | fine | rel medium->fine |
|------------|--------|--------|------|------------------|
| C_total_pF | 5624.24 | 5624.01 | 5623.91 | 1.65e-05 |
| vtap_abs | 11829.2 | 11829.2 | 11829.2 | 1.18e-07 |
| vtap_phase_mdeg | -0.120477 | -0.12027 | -0.120854 | 4.84e-03 |
| tan_delta | 0.00199912 | 0.0019992 | 0.00199922 | 1.36e-05 |
| i_leak | 0.00025084 | 0.00025084 | 0.000250839 | 2.98e-06 |

## Case sigma = 1e-05 S/m

| observable | coarse | medium | fine | rel medium->fine |
|------------|--------|--------|------|------------------|
| C_total_pF | 5624.54 | 5624.31 | 5624.22 | 1.64e-05 |
| vtap_abs | 11833.3 | 11833.3 | 11833.3 | 1.48e-07 |
| vtap_phase_mdeg | -17.3144 | -17.3055 | -17.3046 | 5.60e-05 |
| tan_delta | 0.0139468 | 0.0139474 | 0.0139476 | 1.65e-05 |
| i_leak | 0.00175008 | 0.00175008 | 0.00175008 | 1.21e-07 |

## Verdict
- All key observables change < 2% from medium to fine (clean tap phase excluded,
  it sits at the numerical floor): **True**.
- The medium mesh is adequate for the pollution observables; the Task 010 result
  (C ~5624 pF, the phase trough, monotonic tan-delta) is not a mesh artifact.

## Artifacts
- `mesh_audit.csv`, `mesh_convergence.png`

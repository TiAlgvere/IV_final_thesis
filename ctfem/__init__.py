"""ctfem -- Parametric axisymmetric FEM framework for HV current-transformer
condition-monitoring research.

Scientific goal: model an oil-paper insulated HV current transformer (CT),
inject parametrized insulation defects, and compute how *externally measurable*
electrical quantities (capacitance C1, dissipation factor tan delta, leakage
current magnitude/phase) change -- producing physics-grounded fault signatures.

Out of scope (by design):
  * DGA / oil-chemistry diagnostics.
  * Magnetic / metrological behaviour of the CT cores (separate lumped-circuit
    work package).
  * External surface phenomena (porcelain sheds, pollution flashover).

The package is organised into the modules referenced by the thesis plan:
  config       -- dataclasses describing geometry, materials, operating point, defects
  materials    -- complex-permittivity material database + (sigma + j w eps) helper
  geometry     -- gmsh parametric builder (full CT + degenerate coaxial validator)
  eqs_solver   -- axisymmetric electro-quasistatic solver (DOLFINx, complex PETSc)
  observables  -- terminal admittance -> C1, tan delta; foil ladder; field stress
  defects      -- defect injection (material/geometry level)
  sweep        -- parametric sweep engine -> parquet/CSV dataset
  validate     -- analytic benchmarks + mesh convergence
"""
from __future__ import annotations

__version__ = "0.1.0"

# NOTE: we deliberately do *not* import dolfinx here. config/materials/geometry
# must be importable in a plain (gmsh + numpy) environment so that geometry
# generation and tests that do not touch the solver can run without a full
# FEniCSx/PETSc-complex stack. Solver modules perform their own complex-build
# assertion on import (see ctfem.eqs_solver.assert_complex_build).

from . import config as config  # noqa: E402,F401
from . import materials as materials  # noqa: E402,F401

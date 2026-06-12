"""Backend-independent helpers: region->material resolution, backend selection,
and the unified "solve one case" entry points used by validate/scripts/sweep.

Two FEM backends implement the same physics and observables:

  * ``dolfinx`` -- FEniCSx + complex PETSc (Linux/WSL/HPC; MPI-capable).
  * ``skfem``   -- scikit-fem + scipy sparse (pure Python; runs natively on
                   Windows).  Default wherever DOLFINx is unavailable.

``detect_backend("auto")`` prefers dolfinx when a complex build is importable
and falls back to skfem, so the same scripts run unmodified on a laptop and on
the cluster.
"""
from __future__ import annotations

from typing import Optional

from .config import (
    OperatingParams, MaterialParams, DefectSpec, GeometryParams,
)
from .materials import MaterialDB
from .obs_types import Observables


def region_material_name(region: str, mats: MaterialParams) -> str:
    """Resolve a physical-region name to a material name."""
    if region.startswith("foil_"):
        return mats.foil_material
    if region.startswith("element_"):
        # CVT homogenized capacitor elements (registered per-case via
        # CVTParams.material_db(); see the CVTParams docstring)
        return "element_dielectric"
    return mats.region_material.get(region, "air")


def detect_backend(prefer: str = "auto") -> str:
    """Return 'dolfinx' or 'skfem'.

    'auto' picks dolfinx only if it imports AND is a complex build; otherwise
    skfem (which always supports complex via numpy/scipy).
    """
    if prefer in ("dolfinx", "skfem"):
        return prefer
    if prefer != "auto":
        raise ValueError(f"unknown backend {prefer!r}")
    try:
        import numpy as np
        import dolfinx  # noqa: F401
        if np.issubdtype(dolfinx.default_scalar_type, np.complexfloating):
            return "dolfinx"
    except Exception:
        pass
    return "skfem"


def run_case_2d(
    msh_path: str,
    operating: OperatingParams,
    materials: Optional[MaterialParams] = None,
    matdb: Optional[MaterialDB] = None,
    defect: Optional[DefectSpec] = None,
    geometry: Optional[GeometryParams] = None,
    backend: str = "auto",
    return_solution: bool = False,
):
    """Solve a 2-D axisymmetric case on the chosen backend -> Observables.

    With ``return_solution=True`` returns (Observables, solution, backend_name)
    so callers (e.g. Phase-3 plotting) can post-process the field.
    """
    bk = detect_backend(backend)
    if bk == "dolfinx":
        from .eqs_solver import solve_msh
        from .observables import compute_observables
        sol = solve_msh(msh_path, operating, materials=materials, matdb=matdb,
                        defect=defect, geometry=geometry)
        obs = compute_observables(sol, geometry=geometry)
    else:
        from .skfem_solver import solve_msh_skfem, compute_observables_skfem
        sol = solve_msh_skfem(msh_path, operating, materials=materials,
                              matdb=matdb, defect=defect, geometry=geometry)
        obs = compute_observables_skfem(sol, geometry=geometry)
    if return_solution:
        return obs, sol, bk
    return obs


def run_case_3d(
    msh_path: str,
    operating: OperatingParams,
    materials: Optional[MaterialParams] = None,
    matdb: Optional[MaterialDB] = None,
    defect: Optional[DefectSpec] = None,
    geometry: Optional[GeometryParams] = None,
    backend: str = "auto",
    return_solution: bool = False,
):
    """Solve a 3-D Cartesian case on the chosen backend -> Observables."""
    bk = detect_backend(backend)
    if bk == "dolfinx":
        from .solver3d import solve_msh_3d, compute_observables_3d
        sol = solve_msh_3d(msh_path, operating, materials=materials,
                           matdb=matdb, defect=defect, geometry=geometry)
        row = compute_observables_3d(sol, geometry=geometry)
        obs = _row_to_observables(row, operating)
    else:
        from .skfem_solver import solve_msh_skfem, compute_observables_skfem
        sol = solve_msh_skfem(msh_path, operating, materials=materials,
                              matdb=matdb, defect=defect, geometry=geometry)
        obs = compute_observables_skfem(sol, geometry=geometry)
    if return_solution:
        return obs, sol, bk
    return obs


def _row_to_observables(row: dict, operating: OperatingParams) -> Observables:
    """Adapt the dolfinx-3D dict row to the shared Observables dataclass."""
    Y = complex(row["Y_real"], row["Y_imag"])
    fracs = [row[k] for k in sorted(row) if k.startswith("foil")]
    peaks = [row[k] for k in sorted(row) if k.startswith("gap")]
    return Observables(
        Y=Y, C1_pF=row["C1_pF"], tan_delta=row["tan_delta"],
        Y_reaction=Y, admittance_method_discrepancy=row.get(
            "admittance_discrepancy", 0.0),
        foil_potential_frac=fracs, peak_field_per_gap=peaks,
        peak_field_overall=row.get("peak_field_overall", 0.0),
    )

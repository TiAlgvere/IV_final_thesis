"""Analytic benchmarks and mesh-convergence studies (Phase 2).

The coaxial cylinder capacitor is the canonical analytic check for the
axisymmetric EQS solver:

    C_analytic = 2 pi eps0 eps_r L / ln(b / a)         (see materials.coax_capacitance)

and, for a uniformly lossy dielectric, the *computed* tan delta must equal the
prescribed material tan delta (it is a bulk property, independent of geometry).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace, field
from typing import Optional

import numpy as np

from .config import GeometryParams, OperatingParams, MaterialParams
from .materials import Material, MaterialDB, coax_capacitance


@dataclass
class CoaxValidation:
    r_inner: float
    r_outer: float
    length: float
    eps_r: float
    tan_delta: float
    C_analytic_pF: float
    C_fem_pF: float
    tan_delta_fem: float
    C_rel_error: float
    tan_delta_rel_error: float

    def passed(self, tol: float = 0.005) -> bool:
        return (self.C_rel_error < tol) and (self.tan_delta_rel_error < tol)

    def summary(self) -> str:
        return (
            f"coax a={self.r_inner} b={self.r_outer} L={self.length} "
            f"eps_r={self.eps_r} tand={self.tan_delta}\n"
            f"  C_analytic = {self.C_analytic_pF:.4f} pF\n"
            f"  C_fem      = {self.C_fem_pF:.4f} pF  (rel err {self.C_rel_error:.2e})\n"
            f"  tand_fem   = {self.tan_delta_fem:.5f}  (rel err "
            f"{self.tan_delta_rel_error:.2e})")


def validate_coax(
    r_inner: float = 0.01,
    r_outer: float = 0.05,
    length: float = 0.2,
    eps_r: float = 3.5,
    tan_delta: float = 0.01,
    *,
    lc: float = 0.0015,
    refinement: float = 1.0,
    msh_path: Optional[str] = None,
    backend: str = "auto",
) -> CoaxValidation:
    """Build a coax mesh, solve, and compare C and tan delta to analytics.

    Imports the solver lazily; `backend` is "auto" (prefer dolfinx, fall back
    to the Windows-native scikit-fem backend), "dolfinx", or "skfem".
    """
    from .geometry import build_coax
    from .common import run_case_2d

    tmp = None
    if msh_path is None:
        tmp = tempfile.TemporaryDirectory()
        msh_path = f"{tmp.name}/coax.msh"

    build_coax(r_inner, r_outer, length, msh_path, lc=lc,
               refinement=refinement, verbose=False)

    operating = OperatingParams()
    # single dielectric material with the requested eps_r / tan_delta
    matdb = MaterialDB.default()
    matdb.set("coax_diel", Material("coax_diel", eps_r=eps_r, tan_delta=tan_delta))
    materials = MaterialParams(region_material={"dielectric": "coax_diel"})

    obs = run_case_2d(msh_path, operating, materials=materials, matdb=matdb,
                      backend=backend)

    C_analytic = coax_capacitance(eps_r, r_inner, r_outer, length)
    C_fem = obs.C1_pF * 1e-12
    C_err = abs(C_fem - C_analytic) / C_analytic
    # lossless reference case: fall back to the absolute tan-delta error
    # (relative error is undefined at tan_delta = 0)
    if tan_delta > 0.0:
        tand_err = abs(obs.tan_delta - tan_delta) / tan_delta
    else:
        tand_err = abs(obs.tan_delta)

    if tmp is not None:
        tmp.cleanup()

    return CoaxValidation(
        r_inner, r_outer, length, eps_r, tan_delta,
        C_analytic * 1e12, obs.C1_pF, obs.tan_delta, C_err, tand_err)


@dataclass
class ConvergencePoint:
    refinement: float
    n_triangles: int
    C1_pF: float
    tan_delta: float


def convergence_study(
    geometry: GeometryParams,
    refinements: tuple[float, ...] = (0.5, 1.0, 1.8),
    workdir: Optional[str] = None,
    backend: str = "auto",
) -> list[ConvergencePoint]:
    """Mesh-convergence study on the full CT: C1 vs. refinement level.

    Reports >= 3 levels (spec).  The relative change between the two finest
    levels is the practical convergence estimate.
    """
    from .geometry import build_ct
    from .common import run_case_2d

    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory()
        workdir = tmp.name

    points: list[ConvergencePoint] = []
    for ref in refinements:
        g = replace(geometry, mesh_refinement=ref)
        path = f"{workdir}/ct_ref{ref}.msh"
        mres = build_ct(g, path, verbose=False)
        obs = run_case_2d(path, OperatingParams(), geometry=g, backend=backend)
        points.append(ConvergencePoint(ref, mres.n_triangles, obs.C1_pF, obs.tan_delta))

    if tmp is not None:
        tmp.cleanup()
    return points

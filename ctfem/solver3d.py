"""Full 3-D electro-quasistatic solver + observables (DOLFINx, complex PETSc).

This is the 3-D counterpart of :mod:`ctfem.eqs_solver`.  The physics is identical
(eq. 1 there); only the measure changes: a true 3-D Cartesian volume integral

    a(phi, v) = INT_Omega kappa grad(phi) . grad(v) dV = 0                  (3D)

i.e. WITHOUT the axisymmetric 2*pi*r weight.  Boundary conditions are imposed on
the electrode SURFACES (hv_electrode -> U0, ground_electrode + farfield -> 0);
the foils float as high-sigma volumes.

The defect engine is reused unchanged: defects are parametrized in cylindrical
(r, z), so the solver simply passes each cell's (r = hypot(x, y), z) centroid to
:func:`ctfem.defects.apply_defect`.  A defect that is azimuthally uniform
reproduces the axisymmetric result; a localized 3-D patch can be added later by
making the defect depend on the azimuth too.

!! Runs on Linux/WSL/HPC only (needs DOLFINx + complex PETSc).  It cannot be
   executed on native Windows.  The math mirrors the 2-D solver, which is
   validated against the analytic coax benchmark; validate this 3-D path in WSL
   before production use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem, assemble_vector
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc

# reuse the validated helpers from the 2-D solver
from .eqs_solver import assert_complex_build, _read_msh, region_material_name
from .config import OperatingParams, MaterialParams, DefectSpec, GeometryParams
from .materials import MaterialDB
from .geometry import load_tag_map
from . import defects as _defects

assert_complex_build()


@dataclass
class Solution3D:
    uh: fem.Function
    domain: dmesh.Mesh
    cell_tags: dmesh.MeshTags
    facet_tags: dmesh.MeshTags
    tag_map: dict[str, tuple[int, int]]
    kappa: fem.Function
    omega: float
    u0: float
    region_per_cell: np.ndarray
    centroids_rz: np.ndarray          # (n_cells, 2) cylindrical (r, z)


def _cell_centroids_xyz(domain: dmesh.Mesh) -> np.ndarray:
    tdim = domain.topology.dim
    im = domain.topology.index_map(tdim)
    n_cells = im.size_local + im.num_ghosts
    return dmesh.compute_midpoints(domain, tdim,
                                   np.arange(n_cells, dtype=np.int32))


def build_kappa_3d(domain, cell_tags, tag_map, operating, materials, matdb,
                   defect, geometry=None):
    """Per-cell complex coefficient kappa (DG0) with the defect applied.

    Identical bookkeeping to the 2-D builder, but cell centroids are converted
    from Cartesian (x, y, z) to cylindrical (r, z) before being handed to the
    (dimension-agnostic) defect engine.
    """
    V0 = fem.functionspace(domain, ("DG", 0))
    kappa_fn = fem.Function(V0, dtype=np.complex128)
    tdim = domain.topology.dim
    im = domain.topology.index_map(tdim)
    n_cells = im.size_local + im.num_ghosts

    kmap = matdb.kappa_map(operating.omega, operating.temperature_c)
    region_of_tag = {tag: name for name, (dim, tag) in tag_map.items() if dim == tdim}

    cell_tag_arr = np.full(n_cells, -1, dtype=np.int32)
    cell_tag_arr[cell_tags.indices] = cell_tags.values
    region_per_cell = np.empty(n_cells, dtype=object)
    kappa_arr = np.zeros(n_cells, dtype=np.complex128)
    for c in range(n_cells):
        region = region_of_tag.get(int(cell_tag_arr[c]), "air")
        region_per_cell[c] = region
        kappa_arr[c] = kmap[region_material_name(region, materials)]

    xyz = _cell_centroids_xyz(domain)
    rz = np.column_stack([np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2]])
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])    # cell azimuth for 3-D defects
    kappa_arr = _defects.apply_defect(defect, rz, region_per_cell, kappa_arr,
                                      matdb, operating, materials, geometry,
                                      theta=theta)
    kappa_fn.x.array[:] = kappa_arr
    kappa_fn.x.scatter_forward()
    return kappa_fn, region_per_cell, rz


def solve_msh_3d(msh_path, operating, materials=None, matdb=None, defect=None,
                 geometry=None, *, comm=MPI.COMM_WORLD, petsc_options=None):
    """Read a 3-D .msh, assemble and solve the EQS problem (no axisym weight)."""
    materials = materials or MaterialParams()
    matdb = matdb or MaterialDB.default()
    defect = defect or DefectSpec()

    tag_map = load_tag_map(msh_path)
    domain, cell_tags, facet_tags = _read_msh_3d(msh_path, comm)
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)

    V = fem.functionspace(domain, ("Lagrange", 2))
    kappa_fn, region_per_cell, rz = build_kappa_3d(
        domain, cell_tags, tag_map, operating, materials, matdb, defect, geometry)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa_fn * ufl.grad(u), ufl.grad(v)) * ufl.dx
    zero = fem.Constant(domain, np.complex128(0.0))
    L = ufl.inner(zero, v) * ufl.dx

    fdim = tdim - 1

    def _facets(name):
        if name not in tag_map:
            return None
        _, tag = tag_map[name]
        return facet_tags.indices[facet_tags.values == tag]

    bcs = []
    hv = _facets("hv_electrode")
    if hv is None or len(hv) == 0:
        raise RuntimeError("no hv_electrode facets in 3-D mesh")
    hv_dofs = fem.locate_dofs_topological(V, fdim, hv)
    bcs.append(fem.dirichletbc(fem.Constant(domain, np.complex128(operating.u0)),
                               hv_dofs, V))
    gnd = [f for f in (_facets("ground_electrode"), _facets("farfield"))
           if f is not None and len(f)]
    if not gnd:
        raise RuntimeError("no ground/farfield facets in 3-D mesh")
    gnd_dofs = fem.locate_dofs_topological(V, fdim, np.unique(np.concatenate(gnd)))
    bcs.append(fem.dirichletbc(fem.Constant(domain, np.complex128(0.0)),
                               gnd_dofs, V))

    opts = petsc_options or {  # iterative is wiser for large 3-D systems
        "ksp_type": "cg", "pc_type": "hypre",
        "ksp_rtol": 1e-9, "ksp_max_it": 2000,
    }
    problem = LinearProblem(a, L, bcs=bcs, petsc_options=opts)
    uh = problem.solve()
    uh.name = "phi"

    return Solution3D(uh, domain, cell_tags, facet_tags, tag_map, kappa_fn,
                      operating.omega, operating.u0, region_per_cell, rz)


def _read_msh_3d(msh_path, comm):
    out = gmshio.read_from_msh(msh_path, comm, gdim=3)
    if isinstance(out, tuple):
        return out[0], out[1], out[2]
    return out.mesh, getattr(out, "cell_tags", None), getattr(out, "facet_tags", None)


# --------------------------------------------------------------------------- #
# observables (3-D: plain volume measure, cylindrical-radius gap bands)
# --------------------------------------------------------------------------- #


def terminal_admittance_3d(sol: Solution3D):
    """(Y_energy, Y_reaction, discrepancy) -- same identity as the 2-D solver."""
    domain, uh, kappa = sol.domain, sol.uh, sol.kappa
    Yint = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(
            kappa * ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)), op=MPI.SUM)
    Y_energy = Yint / (sol.u0 ** 2)

    V = uh.function_space
    v = ufl.TestFunction(V)
    res = assemble_vector(fem.form(
        ufl.inner(kappa * ufl.grad(uh), ufl.grad(v)) * ufl.dx))
    res.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)
    fdim = domain.topology.dim - 1
    _, hv_tag = sol.tag_map["hv_electrode"]
    hv_facets = sol.facet_tags.indices[sol.facet_tags.values == hv_tag]
    hv_dofs = fem.locate_dofs_topological(V, fdim, hv_facets)
    nloc = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    I = domain.comm.allreduce(np.sum(res.array[hv_dofs[hv_dofs < nloc]]), op=MPI.SUM)
    Y_react = I / sol.u0
    disc = abs(Y_energy - Y_react) / max(abs(Y_energy), 1e-300)
    return Y_energy, Y_react, disc


def foil_ladder_3d(sol: Solution3D):
    domain, uh = sol.domain, sol.uh
    dx = ufl.Measure("dx", domain=domain, subdomain_data=sol.cell_tags)
    pots, fracs = [], []
    foil_names = sorted([n for n in sol.tag_map if n.startswith("foil_")],
                        key=lambda s: int(s.split("_")[1]))
    for name in foil_names:
        _, tag = sol.tag_map[name]
        num = domain.comm.allreduce(
            fem.assemble_scalar(fem.form(uh * dx(tag))), op=MPI.SUM)
        den = domain.comm.allreduce(
            fem.assemble_scalar(fem.form(1.0 * dx(tag))), op=MPI.SUM)
        phi = num / den if abs(den) > 0 else 0.0
        pots.append(phi)
        fracs.append(abs(phi) / sol.u0)
    return pots, fracs


def field_stress_3d(sol: Solution3D, geometry: GeometryParams):
    domain, uh = sol.domain, sol.uh
    V0 = fem.functionspace(domain, ("DG", 0))
    emag = fem.Function(V0)
    emag.interpolate(fem.Expression(
        ufl.sqrt(ufl.real(ufl.inner(ufl.grad(uh), ufl.grad(uh)))),
        V0.element.interpolation_points()))
    emag_arr = np.real(emag.x.array)
    radii = geometry.foil_radii()
    r = sol.centroids_rz[:, 0]
    is_paper = sol.region_per_cell == "paper_insulation"
    edges = np.concatenate(([geometry.conductor_radius], radii))
    peaks = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        sel = is_paper & (r >= min(lo, hi)) & (r <= max(lo, hi))
        lp = float(emag_arr[sel].max()) if np.any(sel) else 0.0
        peaks.append(domain.comm.allreduce(lp, op=MPI.MAX))
    overall = domain.comm.allreduce(
        float(emag_arr[is_paper].max()) if np.any(is_paper) else 0.0, op=MPI.MAX)
    return peaks, overall


def compute_observables_3d(sol: Solution3D, geometry: Optional[GeometryParams] = None):
    """Return a dict row mirroring the 2-D observables (C1_pF, tan_delta, ...)."""
    Y, Yr, disc = terminal_admittance_3d(sol)
    C1 = Y.imag / sol.omega
    tand = Y.real / Y.imag if abs(Y.imag) > 0 else float("nan")
    row = {"C1_pF": C1 * 1e12, "tan_delta": tand,
           "Y_real": Y.real, "Y_imag": Y.imag,
           "admittance_discrepancy": disc}
    if any(n.startswith("foil_") for n in sol.tag_map):
        _, fracs = foil_ladder_3d(sol)
        for i, f in enumerate(fracs):
            row[f"foil{i + 1}_frac"] = f
        if geometry is not None:
            peaks, overall = field_stress_3d(sol, geometry)
            row["peak_field_overall"] = overall
            for i, e in enumerate(peaks):
                row[f"gap{i + 1}_peakE"] = e
    return row

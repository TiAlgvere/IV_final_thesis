"""Axisymmetric electro-quasistatic (EQS) solver in DOLFINx (complex PETSc).

Governing equation
-------------------
We solve for the complex scalar potential phi:

    div( (sigma + j w eps0 eps_r) grad phi ) = 0                         (1)

in the axisymmetric (r, z) half-plane.  Writing kappa = sigma + j w eps0 eps_r
(see :mod:`ctfem.materials`), the axisymmetric weak form -- obtained by
multiplying (1) by a test function v and integrating over the *volume*
measure dV = 2 pi r dr dz -- is

    a(phi, v) = INT_Omega  kappa  grad(phi) . grad(v)  (2 pi r) dr dz = 0  (2)

with r = x[0].  (Axisymmetric EQS weak form; cf. Haus & Melcher, *EM Fields
and Energy*, Ch. 7, plus the standard axisymmetric Jacobian factor 2 pi r.)

Boundary conditions
--------------------
* phi = U0 on the HV electrode facets   (Dirichlet)
* phi = 0  on the grounded electrode and the far-field facets (Dirichlet)
* natural (zero normal current) on the symmetry axis r = 0 -- it is simply NOT
  in any Dirichlet set; the 2 pi r weight makes it a natural BC automatically.

Grading foils are NOT constrained: they are thin domains with metal-like
conductivity and float to their natural potentials (deliberate design choice).

Element: Lagrange P2 on triangles, complex-valued.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---- complex-build assertion (spec requirement) --------------------------- #
import dolfinx  # noqa: E402

import ufl  # noqa: E402
from dolfinx import fem, mesh as dmesh  # noqa: E402
from dolfinx.fem.petsc import LinearProblem  # noqa: E402
from dolfinx.io import gmshio  # noqa: E402
from mpi4py import MPI  # noqa: E402


def assert_complex_build() -> None:
    """Fail loudly if PETSc/DOLFINx was built with real scalars.

    The EQS formulation needs complex arithmetic (lossy permittivity).  Create
    the environment with a complex PETSc:
        mamba create -n ctfem -c conda-forge python=3.11 fenics-dolfinx \\
            "petsc=*=complex*" gmsh python-gmsh numpy scipy pandas pyvista \\
            matplotlib pytest pyyaml
    """
    if not np.issubdtype(dolfinx.default_scalar_type, np.complexfloating):
        raise RuntimeError(
            "DOLFINx/PETSc is a REAL build (default_scalar_type="
            f"{dolfinx.default_scalar_type}). The EQS solver requires a COMPLEX "
            "PETSc build. Recreate the conda env with \"petsc=*=complex*\"."
        )


assert_complex_build()

from .config import OperatingParams, MaterialParams, DefectSpec, GeometryParams  # noqa: E402
from .materials import MaterialDB  # noqa: E402
from .geometry import load_tag_map  # noqa: E402
from . import defects as _defects  # noqa: E402


@dataclass
class Solution:
    """Container for a solved EQS problem and everything observables need."""

    uh: fem.Function                 # complex P2 potential
    domain: dmesh.Mesh
    cell_tags: dmesh.MeshTags
    facet_tags: dmesh.MeshTags
    tag_map: dict[str, tuple[int, int]]    # name -> (dim, integer tag)
    kappa: fem.Function              # DG0 complex coefficient (per cell)
    omega: float
    u0: float
    region_per_cell: np.ndarray      # (n_cells,) object array of region names
    centroids: np.ndarray            # (n_cells, 2) cell (r, z) centroids


def _read_msh(msh_path: str, comm: MPI.Comm):
    """Read a gmsh .msh, returning (mesh, cell_tags, facet_tags).

    Handles both the dolfinx <=0.8 tuple return and the >=0.9 MeshData return.
    """
    out = gmshio.read_from_msh(msh_path, comm, gdim=2)
    if isinstance(out, tuple):
        return out[0], out[1], out[2]
    # >= 0.9: MeshData object with attributes
    cell_tags = getattr(out, "cell_tags", None)
    facet_tags = getattr(out, "facet_tags", None)
    return out.mesh, cell_tags, facet_tags


# moved to the backend-independent ctfem.common; re-exported for back-compat
from .common import region_material_name  # noqa: E402,F401


def _cell_centroids(domain: dmesh.Mesh) -> np.ndarray:
    """(N,2) array of (r, z) centroids for every local cell."""
    tdim = domain.topology.dim
    n_cells = domain.topology.index_map(tdim).size_local + \
        domain.topology.index_map(tdim).num_ghosts
    midpoints = dmesh.compute_midpoints(
        domain, tdim, np.arange(n_cells, dtype=np.int32))
    return midpoints[:, :2]  # (r, z)


def build_kappa(
    domain: dmesh.Mesh,
    cell_tags: dmesh.MeshTags,
    tag_map: dict[str, tuple[int, int]],
    operating: OperatingParams,
    materials: MaterialParams,
    matdb: MaterialDB,
    defect: DefectSpec,
    geometry: Optional[GeometryParams] = None,
) -> tuple[fem.Function, np.ndarray, np.ndarray]:
    """Assemble the per-cell complex coefficient kappa (DG0), with the defect.

    Builds the healthy baseline coefficient (one complex value per material
    region) then lets the defect mutate it cell-by-cell using cell centroids,
    so spatially-local defects (moisture patch, shorted-foil bridge) work
    without remeshing (material-level injection, spec requirement).
    """
    V0 = fem.functionspace(domain, ("DG", 0))
    kappa_fn = fem.Function(V0, dtype=np.complex128)

    tdim = domain.topology.dim
    n_cells = domain.topology.index_map(tdim).size_local + \
        domain.topology.index_map(tdim).num_ghosts

    # base material name per cell + base kappa per cell
    kmap = matdb.kappa_map(operating.omega, operating.temperature_c)
    region_of_tag = {tag: name for name, (dim, tag) in tag_map.items() if dim == tdim}

    region_per_cell = np.empty(n_cells, dtype=object)
    kappa_arr = np.zeros(n_cells, dtype=np.complex128)
    # cell_tags may not cover ghosts; default unknown cells to "air"
    cell_tag_arr = np.full(n_cells, -1, dtype=np.int32)
    cell_tag_arr[cell_tags.indices] = cell_tags.values
    for c in range(n_cells):
        region = region_of_tag.get(int(cell_tag_arr[c]), "air")
        region_per_cell[c] = region
        mname = region_material_name(region, materials)
        kappa_arr[c] = kmap[mname]

    # apply defect at the cell level (material-level injection)
    centroids = _cell_centroids(domain)
    kappa_arr = _defects.apply_defect(
        defect, centroids, region_per_cell, kappa_arr,
        matdb, operating, materials, geometry)

    # DG0: array index == cell index (identity dofmap). We filled ghosts too,
    # but scatter to be safe across ranks.
    kappa_fn.x.array[:] = kappa_arr
    kappa_fn.x.scatter_forward()
    return kappa_fn, region_per_cell, centroids


def solve_msh(
    msh_path: str,
    operating: OperatingParams,
    materials: Optional[MaterialParams] = None,
    matdb: Optional[MaterialDB] = None,
    defect: Optional[DefectSpec] = None,
    geometry: Optional[GeometryParams] = None,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
    petsc_options: Optional[dict] = None,
) -> Solution:
    """Read a .msh, assemble and solve the axisymmetric EQS problem.

    Dirichlet sets are taken from the facet physical groups: hv_electrode -> U0,
    ground_electrode + farfield -> 0.  (coax has no farfield group.)
    """
    materials = materials or MaterialParams()
    matdb = matdb or MaterialDB.default()
    defect = defect or DefectSpec()

    tag_map = load_tag_map(msh_path)
    domain, cell_tags, facet_tags = _read_msh(msh_path, comm)

    # ensure facet<->cell connectivity exists for locate_dofs_topological
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)

    # function space (P2 complex)
    V = fem.functionspace(domain, ("Lagrange", 2))

    # coefficient
    kappa_fn, region_per_cell, centroids = build_kappa(
        domain, cell_tags, tag_map, operating, materials, matdb, defect, geometry)

    # weak form, axisymmetric measure dV = 2 pi r dr dz
    r = ufl.SpatialCoordinate(domain)[0]
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    two_pi = 2.0 * np.pi
    a = ufl.inner(kappa_fn * ufl.grad(u), ufl.grad(v)) * (two_pi * r) * ufl.dx
    zero = fem.Constant(domain, np.complex128(0.0))
    L = ufl.inner(zero, v) * (two_pi * r) * ufl.dx

    # Dirichlet BCs from facet groups
    bcs = []
    fdim = domain.topology.dim - 1

    def _facets_for(name: str) -> Optional[np.ndarray]:
        if name not in tag_map:
            return None
        _, tag = tag_map[name]
        return facet_tags.indices[facet_tags.values == tag]

    u0 = operating.u0
    hv_facets = _facets_for("hv_electrode")
    if hv_facets is None or len(hv_facets) == 0:
        raise RuntimeError("no hv_electrode facets found in mesh")
    hv_dofs = fem.locate_dofs_topological(V, fdim, hv_facets)
    hv_value = fem.Constant(domain, np.complex128(u0))
    bcs.append(fem.dirichletbc(hv_value, hv_dofs, V))

    gnd_facets_list = []
    for nm in ("ground_electrode", "farfield"):
        f = _facets_for(nm)
        if f is not None and len(f):
            gnd_facets_list.append(f)
    if not gnd_facets_list:
        raise RuntimeError("no ground/farfield facets found in mesh")
    gnd_facets = np.unique(np.concatenate(gnd_facets_list))
    gnd_dofs = fem.locate_dofs_topological(V, fdim, gnd_facets)
    gnd_value = fem.Constant(domain, np.complex128(0.0))
    bcs.append(fem.dirichletbc(gnd_value, gnd_dofs, V))

    opts = petsc_options or {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(a, L, bcs=bcs, petsc_options=opts)
    uh = problem.solve()
    uh.name = "phi"

    return Solution(
        uh=uh, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags,
        tag_map=tag_map, kappa=kappa_fn, omega=operating.omega, u0=u0,
        region_per_cell=region_per_cell, centroids=centroids,
    )

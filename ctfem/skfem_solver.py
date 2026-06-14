"""Windows-native FEM backend: scikit-fem + scipy sparse (complex-native).

Implements the SAME physics as the DOLFINx solvers, on the same gmsh meshes,
producing the same Observables schema -- but in pure Python (numpy/scipy), so
the entire pipeline (Phases 2-5) runs natively on Windows with no PETSc/WSL.

2-D axisymmetric weak form (cf. ctfem.eqs_solver; Haus & Melcher Ch. 7):

    a(phi, v) = INT_Omega kappa grad(phi).grad(v) (2 pi r) dr dz = 0,
    kappa = sigma + j w eps0 eps_r (1 - j tan_delta)   (complex; materials.py)

3-D Cartesian weak form (cf. ctfem.solver3d): same without the 2*pi*r weight.

An optional surface-conduction term (sheet conductance sigma_s on the exterior
"insulator_surface" facet group) models the resistive leakage / pollution-layer
path for flashover studies; see ``solve_msh_skfem``.  It is off (negligible) by
default, so the baseline solve is unchanged.

Discretisation: Lagrange P2 triangles (2-D), P1 tetrahedra by default in 3-D
(P2 tets explode the direct-solver memory on a laptop; pass element="P2" when
on a bigger machine).  Linear solve: scipy SuperLU (complex LU).

Terminal admittance is computed with the same two cross-checked estimators as
the DOLFINx backend (see ctfem.observables docstring):

    (A) energy:    Y = (phi^T A phi) / U0^2          (A = full stiffness matrix,
                                                      non-conjugated product)
    (B) reaction:  Y = sum_{HV dofs} (A phi)_i / U0

which are algebraically identical (phi^T A phi = U0 * I); the discrepancy is a
numerical sanity check.

Field stress |E| is evaluated from the per-element P1 (linear-interpolant)
gradient of phi -- exact for P1, a centroid-level estimate for P2.  Peak-field
observables are used comparatively (defect vs healthy), where this estimator is
consistent; C1/tan-delta come from the full P2 stiffness matrix and carry the
benchmark accuracy.

Mesh I/O: skfem.Mesh.load reads the gmsh .msh via meshio and exposes physical
groups directly as named ``subdomains`` (cells) and ``boundaries`` (facets) --
no tag sidecar needed on this backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import skfem
from skfem import Basis, BilinearForm, FacetBasis, condense
from skfem import solve as skfem_solve
from skfem.helpers import dot, grad

from .config import (
    OperatingParams, MaterialParams, DefectSpec, GeometryParams,
)
from .materials import MaterialDB
from .obs_types import Observables
from .common import region_material_name
from . import defects as _defects


@dataclass
class SkfemSolution:
    """Solved EQS problem (scikit-fem backend), 2-D axisym or 3-D Cartesian."""

    phi: np.ndarray                  # complex dof vector (full, with BC values)
    mesh: "skfem.Mesh"
    basis: "Basis"
    A: "sp.spmatrix"                 # FULL (un-condensed) stiffness matrix
    hv_dofs: np.ndarray
    omega: float
    u0: float
    region_per_cell: np.ndarray      # (n_cells,) region names
    centroids_rz: np.ndarray         # (n_cells, 2) cylindrical (r, z)
    axisymmetric: bool
    surface_leakage_a: complex = 0.0  # insulator-surface leakage current [A]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _dofs_arr(d) -> np.ndarray:
    """Normalize skfem get_dofs() output to a flat int ndarray across versions."""
    for attr in ("flatten", "all"):
        if hasattr(d, attr):
            try:
                return np.unique(np.asarray(getattr(d, attr)(), dtype=np.int64))
            except TypeError:
                pass
    return np.unique(np.asarray(d, dtype=np.int64))


def _boundary_dofs(basis: Basis, mesh: "skfem.Mesh",
                   names: tuple[str, ...]) -> np.ndarray:
    """Union of dofs on the named boundaries that exist in the mesh."""
    have = [n for n in names if mesh.boundaries and n in mesh.boundaries]
    if not have:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate([_dofs_arr(basis.get_dofs(n)) for n in have]))


def _region_per_cell(mesh: "skfem.Mesh") -> np.ndarray:
    n = mesh.t.shape[1]
    rpc = np.full(n, "air", dtype=object)
    for name, idx in (mesh.subdomains or {}).items():
        if name.startswith("gmsh:"):
            # meshio/gmsh bookkeeping sets (e.g. "gmsh:bounding_entities"),
            # not physical regions -- labelling cells with them would corrupt
            # the material map and the device mask in the 3-D viewers
            continue
        rpc[np.asarray(idx, dtype=np.int64)] = name
    return rpc


def _p1_gradient(mesh: "skfem.Mesh", phi_nodes: np.ndarray) -> np.ndarray:
    """Per-element gradient of the linear interpolant of nodal values.

    Returns (dim, n_cells) complex.  Exact for P1; centroid estimate for P2.
    """
    p, t = mesh.p, mesh.t
    if p.shape[0] == 2:
        # triangle: grad lambda_i from the standard 2A formulas
        x1, y1 = p[0, t[0]], p[1, t[0]]
        x2, y2 = p[0, t[1]], p[1, t[1]]
        x3, y3 = p[0, t[2]], p[1, t[2]]
        det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)   # 2*signed area
        b = np.stack([y2 - y3, y3 - y1, y1 - y2]) / det        # d lambda / dx
        c = np.stack([x3 - x2, x1 - x3, x2 - x1]) / det        # d lambda / dy
        ph = phi_nodes[t]                                      # (3, nel)
        return np.stack([(b * ph).sum(axis=0), (c * ph).sum(axis=0)])
    # tetrahedron: grad phi = M^{-T} d, M = [p2-p1; p3-p1; p4-p1]
    e = np.stack([p[:, t[i]] - p[:, t[0]] for i in (1, 2, 3)], axis=0)  # (3,3,nel)
    M = np.transpose(e, (2, 0, 1))                       # (nel, 3, 3) rows=edges
    d = np.stack([phi_nodes[t[i]] - phi_nodes[t[0]] for i in (1, 2, 3)], axis=1)
    sol = np.linalg.solve(M.astype(np.float64), d[..., None].astype(np.complex128))
    return sol[..., 0].T                                  # (3, nel)


def _cell_measures(mesh: "skfem.Mesh") -> np.ndarray:
    """Element area (2-D) or volume (3-D), positive, shape (n_cells,)."""
    p, t = mesh.p, mesh.t
    if p.shape[0] == 2:
        x1, y1 = p[0, t[0]], p[1, t[0]]
        x2, y2 = p[0, t[1]], p[1, t[1]]
        x3, y3 = p[0, t[2]], p[1, t[2]]
        return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    e = np.stack([p[:, t[i]] - p[:, t[0]] for i in (1, 2, 3)], axis=0)
    M = np.transpose(e, (2, 0, 1))
    return np.abs(np.linalg.det(M)) / 6.0


# --------------------------------------------------------------------------- #
# solve
# --------------------------------------------------------------------------- #


def solve_msh_skfem(
    msh_path: str,
    operating: OperatingParams,
    materials: Optional[MaterialParams] = None,
    matdb: Optional[MaterialDB] = None,
    defect: Optional[DefectSpec] = None,
    geometry: Optional[GeometryParams] = None,
    *,
    element: Optional[str] = None,    # "P1" | "P2" | None (auto: P2 2-D, P1 3-D)
) -> SkfemSolution:
    """Read a gmsh .msh (2-D or 3-D) and solve the EQS problem."""
    materials = materials or MaterialParams()
    matdb = matdb or MaterialDB.default()
    defect = defect or DefectSpec()

    mesh = skfem.Mesh.load(msh_path)
    dim = mesh.p.shape[0]
    axisym = dim == 2

    if element is None:
        element = "P2" if axisym else "P1"
    if axisym:
        elem = skfem.ElementTriP2() if element == "P2" else skfem.ElementTriP1()
    else:
        elem = skfem.ElementTetP2() if element == "P2" else skfem.ElementTetP1()
    basis = Basis(mesh, elem)

    # per-cell complex coefficient with the defect applied (same engine as the
    # DOLFINx backends; in 3-D the cell azimuth enables localized defects)
    rpc = _region_per_cell(mesh)
    kmap = matdb.kappa_map(operating.omega, operating.temperature_c)
    kappa = np.array(
        [kmap[region_material_name(r, materials)] for r in rpc],
        dtype=np.complex128)
    cent = mesh.p[:, mesh.t].mean(axis=1).T        # (n_cells, dim)
    if axisym:
        rz = cent[:, :2].copy()
        theta = None
    else:
        rz = np.column_stack([np.hypot(cent[:, 0], cent[:, 1]), cent[:, 2]])
        theta = np.arctan2(cent[:, 1], cent[:, 0])
    kappa = _defects.apply_defect(defect, rz, rpc, kappa, matdb, operating,
                                  materials, geometry, theta=theta)

    # weak form; kappa broadcast to quadrature points (constant per cell)
    nqp = basis.X.shape[1]
    kap_qp = np.ascontiguousarray(
        np.broadcast_to(kappa[:, None], (kappa.size, nqp)))

    if axisym:
        @BilinearForm(dtype=np.complex128)
        def a_form(u, v, w):
            # axisymmetric EQS weak form, dV = 2 pi r dr dz  (r = w.x[0])
            return w["kap"] * dot(grad(u), grad(v)) * (2.0 * np.pi * w.x[0])
    else:
        @BilinearForm(dtype=np.complex128)
        def a_form(u, v, w):
            # 3-D Cartesian EQS weak form
            return w["kap"] * dot(grad(u), grad(v))

    A = a_form.assemble(basis, kap=kap_qp)

    # Optional surface-conduction (leakage) term on the exterior insulator
    # surface.  Adds a Laplace-Beltrami operator with sheet conductance sigma_s
    # to the volume EQS form:  a_s(phi, v) = INT_Gamma sigma_s grad_s(phi) .
    # grad_s(v) dS, where grad_s = (I - n n^T) grad is the tangential (surface)
    # gradient.  Because the volume coefficient already carries conduction
    # (Re kappa) AND displacement (Im kappa), the SAME complex solve spans both
    # regimes: at the clean default sigma_s ~ 1e-14 S the surface term is
    # negligible (pure electrostatic/capacitive limit), and raising sigma_s
    # transitions the surface toward a stationary-current (electrokinetic)
    # leakage path -- no separate solver needed.  Skipped when the mesh has no
    # "insulator_surface" group (2-D / shed-free meshes).
    A_surf = None              # kept separately to extract the leakage current
    sigma_s = getattr(materials, "surface_conductivity_s", 0.0)
    str_sigma = getattr(materials, "streamer_sigma_s", 0.0)
    if (sigma_s or str_sigma) and mesh.boundaries \
            and "insulator_surface" in mesh.boundaries:
        fbasis = FacetBasis(mesh, elem,
                            facets=mesh.boundaries["insulator_surface"])
        th_c = getattr(materials, "streamer_theta_center", 0.0)
        th_w = getattr(materials, "streamer_theta_extent", 0.0)
        z_rng = getattr(materials, "streamer_z_range", None)

        @BilinearForm(dtype=np.complex128)
        def surf_form(u, v, w):
            n = w.n
            gu, gv = grad(u), grad(v)
            gut = gu - dot(gu, n) * n      # tangential (surface) gradient
            gvt = gv - dot(gv, n) * n
            # per-quadrature-point sheet conductance: the uniform layer sigma_s
            # everywhere, raised to the streamer value inside its azimuthal+axial
            # window (3-D only; the 2-D ring has no azimuth to localize).
            ss = sigma_s
            if (not axisym) and str_sigma > 0.0:
                theta = np.arctan2(w.x[1], w.x[0])
                dth = (theta - th_c + np.pi) % (2.0 * np.pi) - np.pi
                in_win = np.abs(dth) <= 0.5 * th_w
                if z_rng is not None:
                    in_win = in_win & (w.x[2] >= z_rng[0]) & (w.x[2] <= z_rng[1])
                ss = np.where(in_win, str_sigma, sigma_s)
            integrand = ss * dot(gut, gvt)
            # axisymmetric: the creepage curve revolves into a surface, dS=2 pi r ds
            if axisym:
                integrand = integrand * (2.0 * np.pi * w.x[0])
            return integrand

        A_surf = surf_form.assemble(fbasis)
        A = A + A_surf

    # Dirichlet sets: hv_electrode -> U0 ; ground_electrode + farfield -> 0.
    # The symmetry axis r=0 is natural (the 2 pi r weight vanishes there).
    hv = _boundary_dofs(basis, mesh, ("hv_electrode",))
    if hv.size == 0:
        raise RuntimeError(f"no hv_electrode boundary in {msh_path}")
    gnd = _boundary_dofs(basis, mesh, ("ground_electrode", "farfield"))
    if gnd.size == 0:
        raise RuntimeError(f"no ground/farfield boundary in {msh_path}")

    u0 = operating.u0
    x = np.zeros(basis.N, dtype=np.complex128)
    x[hv] = u0
    b = np.zeros(basis.N, dtype=np.complex128)
    D = np.unique(np.concatenate([hv, gnd]))
    phi = skfem_solve(*condense(A, b, x=x, D=D))   # SuperLU (complex direct)

    # Surface leakage current to ground [A]: the surface admittance's share of
    # the terminal current, I_surf = (phi^T A_surf phi)/U0 (same non-conjugated
    # energy identity phi^T A phi = U0 I used for the terminal admittance).  In
    # 2-D the 2 pi r weight in A_surf makes this the full revolved-ring current.
    leak = complex(phi @ (A_surf @ phi)) / u0 if A_surf is not None else 0.0

    return SkfemSolution(
        phi=phi, mesh=mesh, basis=basis, A=A, hv_dofs=hv,
        omega=operating.omega, u0=u0, region_per_cell=rpc,
        centroids_rz=rz, axisymmetric=axisym, surface_leakage_a=leak)


# --------------------------------------------------------------------------- #
# observables (identical definitions to the DOLFINx backend)
# --------------------------------------------------------------------------- #


def compute_observables_skfem(
    sol: SkfemSolution, geometry: Optional[GeometryParams] = None
) -> Observables:
    A, phi, u0 = sol.A, sol.phi, sol.u0

    # (A) energy and (B) reaction admittance -- see module docstring
    Aphi = A @ phi
    Y_energy = (phi @ Aphi) / (u0 ** 2)
    Y_react = Aphi[sol.hv_dofs].sum() / u0
    disc = abs(Y_energy - Y_react) / max(abs(Y_energy), 1e-300)

    C1 = Y_energy.imag / sol.omega
    tand = Y_energy.real / Y_energy.imag if abs(Y_energy.imag) > 0 else float("nan")

    # nodal values of phi (vertex dofs; nodal-first numbering via nodal_dofs)
    phi_nodes = phi[sol.basis.nodal_dofs[0]]

    # foil ladder: measure-weighted mean of phi over each foil subdomain.
    # The foil is metal (equipotential), so the vertex-mean is exact.
    mesh = sol.mesh
    meas = _cell_measures(mesh)
    if sol.axisymmetric:
        meas = meas * (2.0 * np.pi * sol.centroids_rz[:, 0])
    pots: list[complex] = []
    fracs: list[float] = []
    foil_names = sorted(
        [n for n in (mesh.subdomains or {}) if n.startswith("foil_")],
        key=lambda s: int(s.split("_")[1]))
    for name in foil_names:
        idx = np.asarray(mesh.subdomains[name], dtype=np.int64)
        cellphi = phi_nodes[mesh.t[:, idx]].mean(axis=0)
        w = meas[idx]
        val = complex((cellphi * w).sum() / w.sum())
        pots.append(val)
        fracs.append(abs(val) / u0)

    # field stress per paper gap (P1-gradient estimator; comparative use)
    peaks: list[float] = []
    overall = 0.0
    if geometry is not None and foil_names:
        gradphi = _p1_gradient(mesh, phi_nodes)
        emag = np.sqrt((np.abs(gradphi) ** 2).sum(axis=0))
        radii = geometry.foil_radii()
        r = sol.centroids_rz[:, 0]
        is_paper = sol.region_per_cell == "paper_insulation"
        edges = np.concatenate(([geometry.conductor_radius], radii))
        for k in range(len(edges) - 1):
            lo, hi = sorted((edges[k], edges[k + 1]))
            selk = is_paper & (r >= lo) & (r <= hi)
            peaks.append(float(emag[selk].max()) if np.any(selk) else 0.0)
        overall = float(emag[is_paper].max()) if np.any(is_paper) else 0.0

    return Observables(
        Y=complex(Y_energy), C1_pF=C1 * 1e12, tan_delta=float(tand),
        Y_reaction=complex(Y_react), admittance_method_discrepancy=float(disc),
        foil_potentials=pots, foil_potential_frac=fracs,
        peak_field_per_gap=peaks, peak_field_overall=overall,
        surface_leakage_mA=abs(sol.surface_leakage_a) * 1e3)

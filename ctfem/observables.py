"""Post-processing: terminal admittance -> C1, tan delta; foil ladder; field stress.

Terminal admittance
--------------------
For a linear two-terminal device with applied potential U0 (phi = U0 on the HV
electrode, 0 on ground), the complex current into the HV electrode is I = Y U0.
Two independent evaluations are implemented and cross-checked:

  (A) Energy / power functional.  Using the variational identity with the test
      "function" w = phi / U0 (which is 1 on HV, 0 on ground):

          I = a(phi, phi/U0) = (1/U0) INT_Omega kappa grad(phi).grad(phi) dV
      =>  Y = I / U0 = (1/U0^2) INT_Omega kappa (grad phi . grad phi) dV,     (A)

      with dV = 2 pi r dr dz and a NON-conjugated dot product (ufl.dot).

  (B) Reaction / flux.  I = sum over HV dofs of a(phi, basis_i) = the discrete
      equivalent of the boundary flux INT_HV kappa grad(phi).n dGamma.        (B)

Both give the same Y (phi^T A phi = U0 * I); we report (A) and the relative
discrepancy with (B) as a numerical sanity check.

Then
      C1 = Im(Y) / w ,      tan delta = Re(Y) / Im(Y).

Note on floating foils: the energy integral runs over the WHOLE domain, including
the high-sigma foil/electrode metal.  There phi is (near) constant so grad(phi)~0
and the ohmic contribution sigma*|grad phi|^2 is physically negligible (good
conductors): the foils add no meaningful spurious loss to tan delta.  Keeping the
integral over all of Omega preserves the exact (A)==(B) identity.

(Standard capacitance/loss extraction; see e.g. any HV insulation text.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector
from mpi4py import MPI
from petsc4py import PETSc

from .eqs_solver import Solution
from .config import GeometryParams
# Observables moved to the backend-independent obs_types module so the
# scikit-fem backend shares the exact same result schema; re-exported here for
# backward compatibility.
from .obs_types import Observables  # noqa: F401


def _axisym(r):
    return 2.0 * np.pi * r


def terminal_admittance(sol: Solution) -> tuple[complex, complex, float]:
    """Return (Y_energy, Y_reaction, relative_discrepancy)."""
    domain = sol.domain
    uh = sol.uh
    kappa = sol.kappa
    r = ufl.SpatialCoordinate(domain)[0]

    # (A) energy functional -- ufl.dot is the NON-conjugated bilinear product
    energy_form = fem.form(
        kappa * ufl.dot(ufl.grad(uh), ufl.grad(uh)) * _axisym(r) * ufl.dx)
    integral = domain.comm.allreduce(
        fem.assemble_scalar(energy_form), op=MPI.SUM)
    Y_energy = integral / (sol.u0 ** 2)

    # (B) reaction: sum of a(phi, basis_i) over HV dofs.  Reuse uh's own space
    # so the assembled vector and the located dofs share the same dof ordering.
    V = uh.function_space
    v = ufl.TestFunction(V)
    a_lin = fem.form(
        ufl.inner(kappa * ufl.grad(uh), ufl.grad(v)) * _axisym(r) * ufl.dx)
    res = assemble_vector(a_lin)
    # accumulate ghost contributions onto owning ranks before summing owned dofs
    res.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)

    # locate HV dofs
    fdim = domain.topology.dim - 1
    _, hv_tag = sol.tag_map["hv_electrode"]
    hv_facets = sol.facet_tags.indices[sol.facet_tags.values == hv_tag]
    hv_dofs = fem.locate_dofs_topological(V, fdim, hv_facets)
    local_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    hv_local = hv_dofs[hv_dofs < local_size]
    I_react = domain.comm.allreduce(
        np.sum(res.array[hv_local]), op=MPI.SUM)
    Y_react = I_react / sol.u0

    disc = abs(Y_energy - Y_react) / max(abs(Y_energy), 1e-300)
    return Y_energy, Y_react, disc


def foil_ladder(sol: Solution) -> tuple[list[complex], list[float]]:
    """Volume-averaged potential of each foil domain (the grading ladder).

    Returns (complex potentials, |phi|/U0 fractions), ordered foil_1..foil_N.
    """
    domain = sol.domain
    uh = sol.uh
    r = ufl.SpatialCoordinate(domain)[0]
    dx = ufl.Measure("dx", domain=domain, subdomain_data=sol.cell_tags)

    pots: list[complex] = []
    fracs: list[float] = []
    foil_names = sorted(
        [n for n in sol.tag_map if n.startswith("foil_")],
        key=lambda s: int(s.split("_")[1]))
    for name in foil_names:
        _, tag = sol.tag_map[name]
        num = domain.comm.allreduce(
            fem.assemble_scalar(fem.form(uh * _axisym(r) * dx(tag))), op=MPI.SUM)
        den = domain.comm.allreduce(
            fem.assemble_scalar(fem.form(_axisym(r) * dx(tag))), op=MPI.SUM)
        phi = num / den if abs(den) > 0 else 0.0
        pots.append(phi)
        fracs.append(abs(phi) / sol.u0)
    return pots, fracs


def field_stress(
    sol: Solution, geometry: GeometryParams
) -> tuple[list[float], float]:
    """Peak |E| in each paper gap (between adjacent foils) and overall peak.

    |E| = |grad phi| is projected to DG0 (one cell-averaged value per cell);
    the per-gap peak is the max over paper cells whose centroid radius lies
    between consecutive foil radii.
    """
    domain = sol.domain
    uh = sol.uh
    V0 = fem.functionspace(domain, ("DG", 0))
    # |E|^2 = grad(phi) . conj(grad(phi))  (real, via ufl.inner)
    emag_expr = fem.Expression(
        ufl.sqrt(ufl.real(ufl.inner(ufl.grad(uh), ufl.grad(uh)))),
        V0.element.interpolation_points())
    emag = fem.Function(V0)
    emag.interpolate(emag_expr)
    emag_arr = np.real(emag.x.array)

    radii = geometry.foil_radii()
    r = sol.centroids[:, 0]
    is_paper = sol.region_per_cell == "paper_insulation"

    peaks: list[float] = []
    # gaps: conductor->foil1, foil1->foil2, ..., foil_{N-1}->foilN
    edges = np.concatenate(([geometry.conductor_radius], radii))
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        sel = is_paper & (r >= min(lo, hi)) & (r <= max(lo, hi))
        local_peak = float(emag_arr[sel].max()) if np.any(sel) else 0.0
        peaks.append(domain.comm.allreduce(local_peak, op=MPI.MAX))
    overall = domain.comm.allreduce(
        float(emag_arr[is_paper].max()) if np.any(is_paper) else 0.0, op=MPI.MAX)
    return peaks, overall


def compute_observables(
    sol: Solution, geometry: Optional[GeometryParams] = None
) -> Observables:
    """Compute the full observables bundle for a solved problem."""
    Y, Y_react, disc = terminal_admittance(sol)
    omega = sol.omega
    C1 = Y.imag / omega           # farads
    tand = Y.real / Y.imag if abs(Y.imag) > 0 else float("nan")

    pots: list[complex] = []
    fracs: list[float] = []
    peaks: list[float] = []
    overall = 0.0
    has_foils = any(n.startswith("foil_") for n in sol.tag_map)
    if has_foils:
        pots, fracs = foil_ladder(sol)
        if geometry is not None:
            peaks, overall = field_stress(sol, geometry)

    return Observables(
        Y=Y, C1_pF=C1 * 1e12, tan_delta=tand,
        Y_reaction=Y_react, admittance_method_discrepancy=disc,
        foil_potentials=pots, foil_potential_frac=fracs,
        peak_field_per_gap=peaks, peak_field_overall=overall,
    )


def coax_observables(sol: Solution) -> Observables:
    """Observables for the coax validation case (no foils/gaps)."""
    return compute_observables(sol, geometry=None)

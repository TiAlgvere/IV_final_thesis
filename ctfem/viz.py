"""Optional visualisation helpers (PyVista off-screen + matplotlib).

All plotting is *optional* and headless: nothing here is imported by the core
pipeline.  PyVista is used off-screen (no GUI) so it runs on HPC nodes; if it is
unavailable the functions raise a clear ImportError only when actually called.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def mesh_png(msh_path: str, png_path: str, *, show_edges: bool = True) -> str:
    """Render a coloured-by-region mesh image to PNG (PyVista, off-screen)."""
    import pyvista as pv
    from dolfinx.io import gmshio
    from dolfinx.plot import vtk_mesh
    from mpi4py import MPI

    domain, cell_tags, _ = gmshio.read_from_msh(msh_path, MPI.COMM_WORLD, gdim=2)
    topology, cell_types, geom = vtk_mesh(domain, domain.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geom)
    grid.cell_data["region"] = cell_tags.values
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(grid, scalars="region", show_edges=show_edges, cmap="tab20")
    pl.view_xy()
    pl.screenshot(png_path)
    pl.close()
    return png_path


def field_png(sol, png_path: str, *, field: str = "phi") -> str:
    """Render the potential (real part) over the mesh to PNG (PyVista)."""
    import pyvista as pv
    from dolfinx.plot import vtk_mesh

    uh = sol.uh
    V = uh.function_space
    topology, cell_types, geom = vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, geom)
    grid.point_data["phi_real"] = np.real(uh.x.array)
    grid.point_data["phi_abs"] = np.abs(uh.x.array)
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(grid, scalars="phi_real", cmap="turbo", show_edges=False)
    pl.view_xy()
    pl.screenshot(png_path)
    pl.close()
    return png_path


def write_xdmf(sol, path: str) -> str:
    """Write the potential to XDMF (real + imag) for ParaView."""
    from dolfinx.io import XDMFFile
    from mpi4py import MPI

    with XDMFFile(MPI.COMM_WORLD, path, "w") as xf:
        xf.write_mesh(sol.domain)
        sol.uh.name = "phi"
        xf.write_function(sol.uh)
    return path


def write_vtx(sol, path: str) -> str:
    """Write the potential to a VTX (.bp) folder for ParaView (complex-aware)."""
    from dolfinx.io import VTXWriter
    from mpi4py import MPI

    with VTXWriter(MPI.COMM_WORLD, path, [sol.uh], engine="BP4") as vtx:
        vtx.write(0.0)
    return path


def skfem_field_png(sol, png_path: str, *, zoom_paper: bool = True,
                    panels=None) -> str:
    """Potential plot for the scikit-fem backend (matplotlib tripcolor).

    Two panels: full device and a zoom on the active part, with equipotential
    contour lines (the classic grading picture).  Runs headless on Windows.
    `panels` overrides the view windows: list of (rmax, (zmin, zmax), title);
    the default is sized for the 245 kV CT.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    m = sol.mesh
    phi_nodes = np.real(sol.phi[sol.basis.nodal_dofs[0]]) / 1e3  # kV
    tri = mtri.Triangulation(m.p[0], m.p[1], m.t.T)

    if panels is None:
        panels = [(1.2, (-0.2, 3.6), "full device"),
                  (0.30, (0.2, 2.8), "condenser body (zoom)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 8))
    for ax, (rmax, zlim, title) in zip(axes, panels):
        tc = ax.tripcolor(tri, phi_nodes, shading="gouraud", cmap="turbo")
        ax.tricontour(tri, phi_nodes, levels=15, colors="k",
                      linewidths=0.4, alpha=0.6)
        ax.set_xlim(0, rmax)
        ax.set_ylim(*zlim)
        ax.set_aspect("equal")
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(title)
        fig.colorbar(tc, ax=ax, label=r"Re $\varphi$ [kV]", shrink=0.8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return png_path


def plot_foil_ladder(fracs: Sequence[float], png_path: str,
                     reference: Optional[Sequence[float]] = None) -> str:
    """Matplotlib plot of the foil-potential ladder (|phi|/U0 vs foil index)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(1, len(fracs) + 1)
    ax.plot(idx, fracs, "o-", label="computed")
    if reference is not None:
        ax.plot(np.arange(1, len(reference) + 1), reference, "x--",
                label="healthy ref", alpha=0.6)
    ax.set_xlabel("foil index (inner -> outer)")
    ax.set_ylabel(r"$|\varphi|/U_0$")
    ax.set_title("grading-foil potential ladder")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path

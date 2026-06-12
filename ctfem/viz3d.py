"""3-D post-processing shared by the device scripts (Windows-native).

Takes a solved :class:`ctfem.skfem_solver.SkfemSolution` (3-D tets) and
produces:

  * a ParaView/PyVista ``.vtu`` with the potential (points), |E|, a defect
    indicator, region ids and a "device" mask (cells) -- the mask lets viewers
    strip the huge far-field air ball with a single threshold;
  * an off-screen verification render (clipped device + defect cells in red);
  * an interactive PyVista window with a draggable clip plane.

Used by ``scripts/phase6_interactive3d.py`` (CT) and ``scripts/phase7_cvt.py``
(CVT) -- pass the case's own MaterialDB (e.g. ``CVTParams.material_db()``) so
the defect indicator resolves device-specific materials.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from .config import OperatingParams, MaterialParams, DefectSpec
from .materials import MaterialDB
from .common import region_material_name
from .defects import apply_defect


def defect_indicator(
    sol,
    spec: DefectSpec,
    geometry=None,
    operating: Optional[OperatingParams] = None,
    materials: Optional[MaterialParams] = None,
    matdb: Optional[MaterialDB] = None,
) -> np.ndarray:
    """Per-cell 0/1 indicator of where the defect changed the material."""
    op = operating or OperatingParams()
    mats = materials or MaterialParams()
    db = matdb or MaterialDB.default()
    kmap = db.kappa_map(op.omega, op.temperature_c)
    base = np.array(
        [kmap[region_material_name(r, mats)] for r in sol.region_per_cell],
        dtype=np.complex128)
    cent = sol.mesh.p[:, sol.mesh.t].mean(axis=1)
    theta = np.arctan2(cent[1], cent[0]) if not sol.axisymmetric else None
    after = apply_defect(spec, sol.centroids_rz, sol.region_per_cell,
                         base.copy(), db, op, mats, geometry, theta=theta)
    return (~np.isclose(after, base, atol=0.0)).astype(np.float32)


def export_vtu(sol, indicator: np.ndarray, vtu_path: str) -> None:
    """Write the 3-D solution to ParaView/PyVista .vtu (+ region-id legend)."""
    import meshio
    from .skfem_solver import _p1_gradient

    phi_nodes = sol.phi[sol.basis.nodal_dofs[0]]
    gradphi = _p1_gradient(sol.mesh, phi_nodes)
    emag = np.sqrt((np.abs(gradphi) ** 2).sum(axis=0))   # V/m per cell

    region_names = sorted(set(sol.region_per_cell))
    rid = {n: i for i, n in enumerate(region_names)}
    region_ids = np.array([rid[r] for r in sol.region_per_cell], dtype=np.int32)
    # device mask (everything except the far-field air ball) -- the viewers use
    # it to show the actual device instead of a huge dark air volume
    device = (sol.region_per_cell != "air").astype(np.float32)

    meshio.write(vtu_path, meshio.Mesh(
        points=sol.mesh.p.T,
        cells=[("tetra", sol.mesh.t.T)],
        point_data={"phi_kV": np.real(phi_nodes) / 1e3},
        cell_data={"E_kV_mm": [emag / 1e6],
                   "defect": [indicator],
                   "region": [region_ids],
                   "device": [device]},
    ))
    legend = os.path.splitext(vtu_path)[0] + "_regions.json"
    with open(legend, "w") as fh:
        json.dump(rid, fh, indent=2)


def device_grid(vtu_path: str):
    """Load the .vtu and strip the far-field air (keep the device)."""
    import pyvista as pv
    grid = pv.read(vtu_path)
    return grid.threshold(0.5, scalars="device"), grid


def _add_defect_actor(pl, dev, field: str, clim) -> bool:
    """Show defect cells with their TRUE field values + a red outline marker.

    Defects are volumetric material changes, not voltage sources -- painting
    them flat red used to suggest they sit at U0.  Rendering them with the same
    scalars/clim as the device shows the actual potential they float at; the
    red outline box only marks WHERE the defect is.
    """
    defect = dev.threshold(0.5, scalars="defect")
    if not defect.n_cells:
        return False
    pl.add_mesh(defect, scalars=field, cmap="turbo", clim=clim,
                show_scalar_bar=False)
    pl.add_mesh(defect.outline(), color="red", line_width=3,
                label="defect region")
    return True


def screenshot(vtu_path: str, png_path: str, field: str) -> None:
    """Off-screen verification render: clipped device + outlined defect."""
    import pyvista as pv
    dev, _ = device_grid(vtu_path)
    clim = dev.get_data_range(field)
    clipped = dev.clip(normal=[0.0, 1.0, 0.0])     # cut away y>0 half
    pl = pv.Plotter(off_screen=True, window_size=(900, 1100))
    pl.add_mesh(clipped, scalars=field, cmap="turbo", clim=clim)
    _add_defect_actor(pl, dev, field, clim)
    pl.add_axes()
    pl.view_isometric()
    pl.screenshot(png_path)
    pl.close()


def show_interactive(vtu_path: str, field: str) -> None:
    """Interactive window: draggable clip plane + defect at its true potential."""
    import pyvista as pv
    dev, _ = device_grid(vtu_path)
    clim = dev.get_data_range(field)
    pl = pv.Plotter(window_size=(1280, 960))
    if _add_defect_actor(pl, dev, field, clim):
        pl.add_legend()
    pl.add_mesh_clip_plane(dev, scalars=field, cmap="turbo", clim=clim,
                           assign_to_axis="y", invert=True,
                           show_edges=False)
    pl.add_axes()
    print("[viz3d] interactive window open: drag the plane through the device,"
          " rotate with the mouse, scroll to zoom. Close the window to exit.")
    pl.show()

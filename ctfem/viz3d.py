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


def cell_efield_kvmm(sol) -> np.ndarray:
    """Per-cell electric-field magnitude in kV/mm, from the complex P1 gradient.

    E = -grad(phi);  |E| = sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)  (complex norm, since
    phi is complex).  The skfem gradient is in V/m; divide by 1e6 for kV/mm.
    This is the dielectric-breakdown-relevant quantity (peak field vs the ~3
    kV/mm air strength), independent of the phase angle.
    """
    from .skfem_solver import _p1_gradient
    phi_nodes = sol.phi[sol.basis.nodal_dofs[0]]
    gradphi = _p1_gradient(sol.mesh, phi_nodes)
    return np.sqrt((np.abs(gradphi) ** 2).sum(axis=0)) / 1e6   # V/m -> kV/mm


def peak_efield_air(sol, r_max: float = 0.40, z_range=None):
    """Peak |E| [kV/mm] over AIR cells within radius `r_max` (and optional axial
    `z_range`) -- the air-gap field that governs flashover.  Excludes the far-
    field air ball via `r_max`.  Returns (peak_kVmm, (r, z) of the peak)."""
    e = cell_efield_kvmm(sol)
    r, z = sol.centroids_rz[:, 0], sol.centroids_rz[:, 1]
    sel = (sol.region_per_cell == "air") & (r <= r_max)
    if z_range is not None:
        sel = sel & (z >= z_range[0]) & (z <= z_range[1])
    if not np.any(sel):
        return 0.0, None
    i = int(np.argmax(np.where(sel, e, -np.inf)))
    return float(e[i]), (float(r[i]), float(z[i]))


def efield_screenshot(vtu_path: str, png_path: str, z_bot: float, z_top: float,
                      r_box: float = 0.35, clim=None) -> None:
    """Off-screen |E| render cropped to a column [|x|,|y| < r_box, z_bot..z_top]
    around the insulator -- keeps the near-shed AIR gap (the device mask would
    strip it).  `clim=None` auto-scales the colour map to the cropped field range
    so the field concentration is clearly visible (the absolute vs-3-kV/mm air-
    breakdown comparison is reported separately as the peak value); pass an
    explicit clim (e.g. (0, 3)) to put it on the breakdown scale instead."""
    import pyvista as pv
    grid = pv.read(vtu_path)
    box = grid.clip_box([-r_box, r_box, -r_box, r_box, z_bot, z_top], invert=False)
    cut = box.clip(normal=[0.0, 1.0, 0.0])      # cross-section to see inside
    if clim is None:
        clim = cut.get_data_range("E_kV_mm")
    pl = pv.Plotter(off_screen=True, window_size=(900, 1100))
    pl.add_mesh(cut, scalars="E_kV_mm", cmap="turbo", clim=clim,
                scalar_bar_args={"title": "|E| [kV/mm]"})
    pl.add_axes()
    pl.view_isometric()
    pl.screenshot(png_path)
    pl.close()


def export_vtu(sol, indicator: np.ndarray, vtu_path: str) -> None:
    """Write the 3-D solution to ParaView/PyVista .vtu (+ region-id legend)."""
    import meshio

    phi_nodes = sol.phi[sol.basis.nodal_dofs[0]]
    emag_kvmm = cell_efield_kvmm(sol)                     # |E| per cell [kV/mm]
    # local phase angle [milliradians] of the complex nodal potential vs the
    # (real) HV drive: theta = arctan2(phi_im, phi_re).  Shows WHERE the phase
    # vector twists during a lossy/polluted run (the relaxation-peak hot spots).
    phi_phase_mrad = np.arctan2(phi_nodes.imag, phi_nodes.real) * 1e3

    region_names = sorted(set(sol.region_per_cell))
    rid = {n: i for i, n in enumerate(region_names)}
    region_ids = np.array([rid[r] for r in sol.region_per_cell], dtype=np.int32)
    # device mask (everything except the far-field air ball) -- the viewers use
    # it to show the actual device instead of a huge dark air volume
    device = (sol.region_per_cell != "air").astype(np.float32)

    meshio.write(vtu_path, meshio.Mesh(
        points=sol.mesh.p.T,
        cells=[("tetra", sol.mesh.t.T)],
        point_data={"phi_kV": np.real(phi_nodes) / 1e3,
                    "phi_phase_mrad": phi_phase_mrad},
        cell_data={"E_kV_mm": [emag_kvmm],
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


def show_interactive(vtu_path: str, field: str = "phi_kV") -> None:
    """Interactive window: draggable clip plane + a key to toggle scalar fields.

    Press 't' to cycle the coloured field -- potential magnitude ``phi_kV`` <->
    phase angle ``phi_phase_mrad`` <-> field ``E_kV_mm`` -- so a polluted run
    shows both WHERE the potential sits and WHERE the phase vector twists hardest
    along the geometry.  `field` selects the one shown first.
    """
    import pyvista as pv
    dev, _ = device_grid(vtu_path)
    candidates = ["phi_kV", "phi_phase_mrad", "E_kV_mm"]
    fields = [f for f in candidates
              if f in dev.point_data or f in dev.cell_data] or [field]
    state = {"i": fields.index(field) if field in fields else 0}

    pl = pv.Plotter(window_size=(1280, 960))

    def _draw() -> None:
        pl.clear()
        f = fields[state["i"]]
        clim = dev.get_data_range(f)
        _add_defect_actor(pl, dev, f, clim)
        pl.add_mesh_clip_plane(dev, scalars=f, cmap="turbo", clim=clim,
                               assign_to_axis="y", invert=True, show_edges=False)
        pl.add_axes()
        pl.add_text(f"field: {f}   (press 't' to toggle)", name="_label",
                    font_size=10)
        pl.render()

    def _toggle() -> None:
        state["i"] = (state["i"] + 1) % len(fields)
        _draw()

    pl.add_key_event("t", _toggle)
    _draw()
    print("[viz3d] interactive window: 't' toggles field "
          f"({' / '.join(fields)}); drag the clip plane, rotate, scroll to zoom.")
    pl.show()

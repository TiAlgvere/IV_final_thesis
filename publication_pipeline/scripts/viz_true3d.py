"""Task 022 - TRUE 3D renders from the actual 3D mesh (NOT revolved 2D).

Loads the solved true-3D VTU (results/raw/022_true3d_foundation/run3d/.../case*.vtu) and
renders, with pyvista off-screen:
  true3d_material.png  - material regions (categorical), full device
  true3d_mesh.png      - surface tetrahedral mesh (edges)
  true3d_phi_re.png    - Re(phi) on an axial cut plane
  true3d_Emag.png      - |E| (log) on an axial cut plane
  true3d_cutaway.png   - half-section cutaway exposing the C1/tap/C2 stack
  true3d_hotspot.png   - |E| with the field hot-spot

Every figure is the genuine 3D field on tetrahedra; nothing is revolved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT.parent), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from run_elmer_case import _parse_mesh_names  # noqa: E402

pv.OFF_SCREEN = True
PROC = ROOT / "results" / "processed" / "022_true3d_foundation"
RAW = ROOT / "results" / "raw" / "022_true3d_foundation"

MAT_COLOR = {"air": "#e5e7eb", "oil": "#fde68a", "porcelain": "#cdab7e",
             "element": "#60a5fa", "metal": "#6b7280"}


def material_of(name):
    if name == "air":
        return "air"
    if name == "oil":
        return "oil"
    if name.startswith("porcelain"):
        return "porcelain"
    if name.startswith("element"):
        return "element"
    return "metal"


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    vtu = sorted((RAW / "run3d").glob("**/case*.vtu"))[-1]
    ids = _parse_mesh_names(RAW / "elmer" / "mesh" / "mesh.names")
    id2name = {bid: n for n, bid in ids["bodies"].items()}
    air_id = ids["bodies"]["air"]

    mesh = pv.read(str(vtu))
    gid = np.asarray(mesh.cell_data["GeometryIds"], int).reshape(-1)
    # categorical material index per cell
    mats = ["air", "oil", "porcelain", "element", "metal"]
    matidx = np.array([mats.index(material_of(id2name.get(g, "air"))) for g in gid])
    mesh.cell_data["material"] = matidx

    # |E| from grad(Re) and grad(Im)
    gr = mesh.compute_derivative(scalars="potential re", gradient="g_re")["g_re"]
    gi = mesh.compute_derivative(scalars="potential im", gradient="g_im")["g_im"]
    emag = np.sqrt((gr ** 2).sum(1) + (gi ** 2).sum(1))
    mesh.point_data["Emag"] = emag
    mesh.point_data["logE"] = np.log10(np.clip(emag, 1.0, None))

    # device = volume bodies only (gids 1..9); excludes air (10) and boundary faces (>=100).
    # threshold prunes orphan points so bounds/surface are correct (extract_cells does not).
    device = mesh.threshold([0.5, 9.5], scalars="GeometryIds")
    dsurf = device.extract_surface()
    b = dsurf.bounds  # true device bounds (orphan air points dropped by surface extract)
    ctr = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))

    def make_cam(focal=None, dist_factor=2.6, dir_vec=(1.0, -1.0, 0.35)):
        f = np.asarray(focal) if focal is not None else ctr
        dv = np.asarray(dir_vec, float); dv /= np.linalg.norm(dv)
        return [tuple(f + dist_factor * diag * dv), tuple(f), (0.0, 0.0, 1.0)]

    cam = make_cam()
    cmats = [MAT_COLOR[m] for m in mats]

    def plot(fn, body, scalars=None, cmap=None, clim=None, categorical=False,
             edges=False, title="", bar=True, camera=None):
        p = pv.Plotter(off_screen=True, window_size=(900, 1300)); p.set_background("white")
        if categorical:
            p.add_mesh(body, scalars=scalars, cmap=cmats, clim=[-0.5, len(mats) - 0.5],
                       n_colors=len(mats), show_scalar_bar=False, show_edges=False)
            p.add_legend([(mats[i], cmats[i]) for i in range(len(mats))],
                         bcolor="white", size=(0.16, 0.18), loc="upper right")
        else:
            p.add_mesh(body, scalars=scalars, cmap=cmap, clim=clim, show_edges=edges,
                       edge_color="#222222", line_width=0.3,
                       scalar_bar_args={"title": title, "vertical": True, "title_font_size": 22,
                                        "label_font_size": 16, "n_labels": 5,
                                        "position_x": 0.86, "position_y": 0.25} if bar else None,
                       show_scalar_bar=bar)
        if title:
            p.add_text(title, position="upper_left", font_size=13, color="black")
        p.camera_position = camera or cam
        p.screenshot(str(PROC / fn)); p.close()

    # 1. material regions (full device, exterior)
    plot("true3d_material.png", device, scalars="material", categorical=True,
         title="material regions (true 3D)")
    # 2. surface mesh
    plot("true3d_mesh.png", device.extract_surface(), scalars=None, cmap=["#9ca3af"],
         edges=True, bar=False, title="3D tetrahedral mesh (surface)")
    # 3 + 4 + 5: axial SLICE through the device (face-on shows the r-z cross-section)
    sl = mesh.slice(normal=(0, 1, 0), origin=tuple(ctr))          # full slice (with air/field)
    sl_dev = device.slice(normal=(0, 1, 0), origin=tuple(ctr))    # device-only slice
    cam_face = make_cam(dir_vec=(0.001, -1.0, 0.0))               # face-on to the y=0 plane
    plot("true3d_phi_re.png", sl, scalars="potential re", cmap="coolwarm",
         title="Re(phi) [V]  (axial slice)", camera=cam_face)
    plot("true3d_Emag.png", sl, scalars="logE", cmap="inferno",
         title="log10 |E| [V/m]  (axial slice)", camera=cam_face)
    plot("true3d_cutaway.png", sl_dev, scalars="material", categorical=True,
         title="cutaway: C1 / tap / C2 stack exposed", camera=cam_face)
    # 7. hotspot: |E| on the device surface, zoom to field max
    imax = int(np.argmax(emag))
    hp = mesh.points[imax]
    cam_hot = make_cam(focal=hp, dist_factor=0.9)
    p = pv.Plotter(off_screen=True, window_size=(900, 1100)); p.set_background("white")
    p.add_mesh(device.extract_surface(), scalars="logE", cmap="inferno",
               scalar_bar_args={"title": "log10 |E|", "vertical": True})
    p.add_mesh(pv.Sphere(0.02, center=hp), color="cyan")
    p.add_text(f"field hot-spot  |E|max={emag.max():.2e} V/m", position="upper_left",
               font_size=12, color="black")
    p.camera_position = cam_hot
    p.screenshot(str(PROC / "true3d_hotspot.png")); p.close()

    print("Wrote 7 true-3D renders in", PROC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

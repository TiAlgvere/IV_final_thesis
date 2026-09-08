"""Task 023 - clean-field 3D figures from the solved shed-fidelity VTU (NOT revolved 2D).

Reads the true-3D solved VTU from run_true3d_v2.py and renders Re(phi), |E|, a material
cutaway (with sheds) and the field hot-spot - all genuine 3D fields on tetrahedra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT.parent), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from run_elmer_case import _parse_mesh_names  # noqa: E402

pv.OFF_SCREEN = True
PROC = ROOT / "results" / "processed" / "023_true3d_geometry_fidelity"
RAW = ROOT / "results" / "raw" / "023_true3d_geometry_fidelity"
MATS = ["air", "oil", "porcelain", "element", "metal"]
MAT_COLOR = ["#e5e7eb", "#fde68a", "#cdab7e", "#60a5fa", "#6b7280"]


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
    mesh.cell_data["material"] = np.array([MATS.index(material_of(id2name.get(g, "air"))) for g in gid])
    gr = mesh.compute_derivative(scalars="potential re", gradient="gr")["gr"]
    gi = mesh.compute_derivative(scalars="potential im", gradient="gi")["gi"]
    emag = np.sqrt((gr ** 2).sum(1) + (gi ** 2).sum(1))
    mesh.point_data["logE"] = np.log10(np.clip(emag, 1.0, None))

    device = mesh.threshold([0.5, 9.5], scalars="GeometryIds")
    b = device.extract_surface().bounds
    ctr = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))

    def cam(dir_vec=(1, -1, 0.35), df=2.6, focal=None):
        f = np.asarray(focal) if focal is not None else ctr
        dv = np.asarray(dir_vec, float); dv /= np.linalg.norm(dv)
        return [tuple(f + df * diag * dv), tuple(f), (0, 0, 1)]

    face = cam(dir_vec=(0.001, -1, 0))
    sl = mesh.slice(normal=(0, 1, 0), origin=tuple(ctr))
    sl_dev = device.slice(normal=(0, 1, 0), origin=tuple(ctr))

    def plot(body, fn, title, scalars, cmap=None, categorical=False, camera=None, bar=True):
        p = pv.Plotter(off_screen=True, window_size=(950, 1350)); p.set_background("white")
        if categorical:
            p.add_mesh(body, scalars=scalars, cmap=MAT_COLOR, clim=[-0.5, 4.5], n_colors=5,
                       show_scalar_bar=False)
            p.add_legend([(MATS[i], MAT_COLOR[i]) for i in range(5)], bcolor="white",
                         size=(0.16, 0.18), loc="upper right")
        else:
            p.add_mesh(body, scalars=scalars, cmap=cmap, show_scalar_bar=bar,
                       scalar_bar_args={"title": title, "vertical": True, "title_font_size": 22,
                                        "label_font_size": 16, "n_labels": 5,
                                        "position_x": 0.86, "position_y": 0.25} if bar else None)
        p.add_text(title, position="upper_left", font_size=13, color="black")
        p.camera_position = camera or cam()
        p.screenshot(str(PROC / fn)); p.close()

    plot(sl, "true3d_clean_phi.png", "Re(phi) [V]  (axial slice)", "potential re", cmap="coolwarm", camera=face)
    plot(sl, "true3d_clean_Emag.png", "log10 |E| [V/m]  (axial slice)", "logE", cmap="inferno", camera=face)
    plot(sl_dev, "true3d_clean_cutaway.png", "clean cutaway (material + sheds)", "material",
         categorical=True, camera=face)

    imax = int(np.argmax(emag)); hp = mesh.points[imax]
    p = pv.Plotter(off_screen=True, window_size=(950, 1200)); p.set_background("white")
    p.add_mesh(device.extract_surface(), scalars="logE", cmap="inferno",
               scalar_bar_args={"title": "log10 |E|", "vertical": True})
    p.add_mesh(pv.Sphere(radius=0.03, center=hp), color="cyan")
    p.add_text(f"field hot-spot  |E|max={emag.max():.2e} V/m", position="upper_left",
               font_size=12, color="black")
    p.camera_position = cam(df=0.9, focal=hp)
    p.screenshot(str(PROC / "true3d_clean_hotspot.png")); p.close()
    print("Wrote 4 clean-field figures in", PROC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

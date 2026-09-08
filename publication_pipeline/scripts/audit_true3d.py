"""Task 023 - geometry / material / mesh-quality audit + figures for the shed-fidelity 3D.

Builds the FINE audit mesh (27 sheds), then from the .msh directly (no solve) renders the
geometry/material inspection figures and the mesh-quality figures, and writes an audit JSON
(material volumes, quality percentiles, worst elements, insulator-surface tri quality, all
physical-group tags). Clean-field figures come from the solved VTU (run_true3d_v2.py).

Reads the gmsh mesh with meshio -> builds a tetra-only pyvista grid tagged by material, so
material mislabels (e.g. core tagged porcelain) are visually and numerically catchable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT.parent), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from publication_pipeline.geometry.gmsh.cvt_ddb123_3d_v2 import MAT_OF, build_cvt_ddb123_3d_v2  # noqa: E402

pv.OFF_SCREEN = True
PROC = ROOT / "results" / "processed" / "023_true3d_geometry_fidelity"
RAW = ROOT / "results" / "raw" / "023_true3d_geometry_fidelity"
AUDIT_MSH = RAW / "cvt3d_v2_audit.msh"

MATS = ["air", "oil", "porcelain", "element", "metal"]
MAT_COLOR = ["#e5e7eb", "#fde68a", "#cdab7e", "#60a5fa", "#6b7280"]


def material_of_name(name):
    return MAT_OF.get(name, "air")


def load_tet_grid(msh):
    m = meshio.read(str(msh))
    pts = np.asarray(m.points, float)
    name3 = {}  # (dim,tag) -> name
    for nm, (tag, dim) in m.field_data.items():
        name3[(int(dim), int(tag))] = nm
    tets, phys = [], []
    for i, cb in enumerate(m.cells):
        if cb.type in ("tetra", "tetra10"):
            tets.append(np.asarray(cb.data, int)[:, :4])
            phys.append(np.asarray(m.cell_data["gmsh:physical"][i], int).reshape(-1))
    tets = np.vstack(tets); phys = np.concatenate(phys)
    matidx = np.array([MATS.index(material_of_name(name3.get((3, int(t)), "air"))) for t in phys])
    cells = np.hstack([np.full((len(tets), 1), 4), tets]).ravel()
    ctypes = np.full(len(tets), pv.CellType.TETRA, np.uint8)
    grid = pv.UnstructuredGrid(cells, ctypes, pts)
    grid.cell_data["material"] = matidx
    return grid


def cam_for(bounds, dist_factor=2.6, dir_vec=(1.0, -1.0, 0.35), focal=None):
    ctr = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
    diag = float(np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]))
    f = np.asarray(focal) if focal is not None else ctr
    dv = np.asarray(dir_vec, float); dv /= np.linalg.norm(dv)
    return [tuple(f + dist_factor * diag * dv), tuple(f), (0.0, 0.0, 1.0)]


def mat_plot(body, fname, title, cam, edges=False, win=(900, 1300)):
    p = pv.Plotter(off_screen=True, window_size=win); p.set_background("white")
    p.add_mesh(body, scalars="material", cmap=MAT_COLOR, clim=[-0.5, 4.5], n_colors=5,
               show_scalar_bar=False, show_edges=edges, edge_color="#333333", line_width=0.3)
    p.add_legend([(MATS[i], MAT_COLOR[i]) for i in range(5)], bcolor="white",
                 size=(0.16, 0.18), loc="upper right")
    p.add_text(title, position="upper_left", font_size=13, color="black")
    p.camera_position = cam
    p.screenshot(str(PROC / fname)); p.close()


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    if not AUDIT_MSH.is_file():
        print("Building FINE audit mesh (27 sheds) ...")
        meta = build_cvt_ddb123_3d_v2(AUDIT_MSH, n_sheds=27, curvature=7.0,
                                      lc={"shed": 0.016, "porcelain": 0.016, "oil": 0.028,
                                          "element_c1": 0.020, "element_c2": 0.016,
                                          "head_housing": 0.045, "base_tank": 0.055, "air": 0.250})
        (PROC / "audit_mesh_meta.json").write_text(
            json.dumps({k: v for k, v in meta.items() if k != "parameters"}, indent=2), encoding="utf-8")
        print(f"  nodes={meta['n_nodes']} tets={meta['n_tets']} qmin={meta['quality_minSICN']['min']:.3f}")
    meta = json.loads((PROC / "audit_mesh_meta.json").read_text(encoding="utf-8"))

    print("Loading mesh + building material grid ...")
    grid = load_tet_grid(AUDIT_MSH)
    matidx = grid.cell_data["material"]
    device = grid.extract_cells(matidx != MATS.index("air"))
    db = device.extract_surface().bounds
    full_cam = cam_for(db)

    # --- material / geometry figures ---
    print("Rendering material/geometry figures ...")
    mat_plot(device, "true3d_material_full.png", "material regions (27-shed 3D)", full_cam)

    half = grid.clip(normal=(0, 1, 0), origin=(0, 0, 0))                 # 270-ish cutaway
    half_dev = device.clip(normal=(0, 1, 0), origin=(0, 0, 0))
    mat_plot(half_dev, "true3d_material_cutaway.png", "axial cutaway (material)", full_cam)

    sl = device.slice(normal=(0, 1, 0))
    face_cam = cam_for(db, dir_vec=(0.001, -1.0, 0.0))
    mat_plot(sl, "true3d_material_axial_slice.png", "axial slice through the central axis", face_cam)

    # horizontal slices at bottom / mid / top
    zs = [0.30, 1.00, 1.50]
    p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1500, 620)); p.set_background("white")
    for k, z in enumerate(zs):
        hs = device.slice(normal=(0, 0, 1), origin=(0, 0, z))
        p.subplot(0, k)
        p.add_mesh(hs, scalars="material", cmap=MAT_COLOR, clim=[-0.5, 4.5], n_colors=5,
                   show_scalar_bar=False)
        p.add_text(f"z = {z:.2f} m", position="upper_edge", font_size=11, color="black")
        p.view_xy()
    p.screenshot(str(PROC / "true3d_material_horizontal_slices.png")); p.close()

    # shed close-up (several sheds): clip a z-band of the device, 3/4 view
    band = device.clip_box([0, db[1], db[2], db[3], 1.05, 1.35], invert=True)
    cam_shed = cam_for([0, 0.17, -0.17, 0.0, 1.05, 1.35], dist_factor=2.2)
    mat_plot(band, "true3d_shed_closeup.png", "weather-shed close-up (porcelain)", cam_shed, win=(1000, 900))

    # stack / tap cutaway close-up
    stack_sl = sl.clip_box([0, 0.13, -0.01, 0.01, 0.5, 1.6], invert=True)
    cam_stack = cam_for([0, 0.13, 0, 0, 0.5, 1.6], dir_vec=(0.001, -1, 0), dist_factor=2.2)
    mat_plot(stack_sl, "true3d_stack_cutaway.png", "capacitor stack / tap (axial slice)",
             cam_stack, win=(700, 1200))

    # head/tank transition close-ups (extra inspection)
    head_band = device.clip_box([0, db[1], db[2], db[3], 1.55, db[5]], invert=True)
    mat_plot(head_band, "true3d_head_closeup.png", "HV head / terminal transition",
             cam_for([0, 0.2, -0.2, 0, 1.55, db[5]], dist_factor=2.2), win=(1000, 900))
    tank_band = device.clip_box([0, db[1], db[2], db[3], db[4], 0.62], invert=True)
    mat_plot(tank_band, "true3d_tank_closeup.png", "grounded tank / porcelain transition",
             cam_for([0, 0.25, -0.25, 0, db[4], 0.62], dist_factor=2.2), win=(1000, 900))

    # --- mesh figures ---
    print("Rendering mesh figures ...")
    surf = device.extract_surface()
    p = pv.Plotter(off_screen=True, window_size=(900, 1300)); p.set_background("white")
    p.add_mesh(surf, color="#9fb3c8", show_edges=True, edge_color="#1f2937", line_width=0.3)
    p.add_text("3D tetrahedral surface mesh (27 sheds)", position="upper_left", font_size=13, color="black")
    p.camera_position = full_cam; p.screenshot(str(PROC / "true3d_mesh_full.png")); p.close()

    def mesh_zoom(box, focal, fname, title, df=2.0):
        sub = device.clip_box(box, invert=True).extract_surface()
        p = pv.Plotter(off_screen=True, window_size=(1000, 900)); p.set_background("white")
        p.add_mesh(sub, color="#cbd5e1", show_edges=True, edge_color="#111827", line_width=0.4)
        p.add_text(title, position="upper_left", font_size=12, color="black")
        p.camera_position = cam_for(box_to_bounds(box), dir_vec=(1, -1, 0.3), dist_factor=df, focal=focal)
        p.screenshot(str(PROC / fname)); p.close()

    def box_to_bounds(b):
        return [b[0], b[1], b[2], b[3], b[4], b[5]]

    mesh_zoom([0, 0.17, -0.17, 0.0, 1.05, 1.30], (0.12, -0.06, 1.17),
              "true3d_mesh_shed_zoom.png", "mesh near weather sheds")
    mesh_zoom([0, 0.20, -0.20, 0.0, 1.6, db[5]], (0.0, -0.05, 1.78),
              "true3d_mesh_terminal_zoom.png", "mesh near HV head / terminal")

    # quality histogram
    qnpz = Path(str(AUDIT_MSH) + ".quality.npz")
    if qnpz.is_file():
        q = np.load(qnpz)["tet_q"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(q, bins=80, color="#2563eb", alpha=0.8)
        for pc, c in [(1, "#b91c1c"), (5, "#ea580c"), (50, "#16a34a")]:
            v = np.percentile(q, pc); ax.axvline(v, color=c, ls="--", lw=1.5, label=f"p{pc}={v:.3f}")
        ax.axvline(0.05, color="k", ls=":", lw=1.2, label="sliver threshold 0.05")
        ax.set_xlabel("element quality (minSICN)"); ax.set_ylabel("count (log)"); ax.set_yscale("log")
        ax.set_title(f"Task 023 - tet quality (n={len(q)}, min={q.min():.3f}, mean={q.mean():.3f})")
        ax.legend(); fig.tight_layout(); fig.savefig(PROC / "true3d_mesh_quality_histogram.png", dpi=150)
        plt.close(fig)

    # worst-20 elements in r-z, colored by material
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap_m = {"air": "#9ca3af", "oil": "#eab308", "porcelain": "#b45309",
              "element": "#2563eb", "metal": "#374151"}
    for w in meta["worst20"]:
        ax.scatter(w["rz"][0], w["rz"][1], c=cmap_m.get(w["mat"], "k"), s=90, edgecolors="k")
        ax.annotate(f"{w['q']:.2f}", (w["rz"][0], w["rz"][1]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("r [m]"); ax.set_ylabel("z [m]"); ax.grid(alpha=0.3)
    ax.set_title("Task 023 - worst 20 elements (location + material, quality labelled)")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap_m[k],
               markeredgecolor="k", markersize=9, label=k) for k in cmap_m]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout(); fig.savefig(PROC / "true3d_worst_elements.png", dpi=150); plt.close(fig)

    print("Audit figures written to", PROC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

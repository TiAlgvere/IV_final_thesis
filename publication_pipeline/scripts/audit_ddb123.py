"""Task 024 - DDB-123 component-architecture audit: bounding-box material audit, figures,
mesh-quality stats. Reads the v4 mesh directly (meshio -> tet grid tagged by material) and
the solved VTU for clean fields. Builds a fine audit mesh if absent.

Outputs: material_audit.json, mesh_quality.json, geometry_parameters.json, and the figure
set required by Task 024.
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
from publication_pipeline.geometry.gmsh.cvt_ddb123_3d_v4 import (MAT_OF_V4, MATERIALS,  # noqa: E402
                                                                 build_cvt_ddb123_3d_v4)

pv.OFF_SCREEN = True
PROC = ROOT / "results" / "processed" / "024_true3d_ddb123_component_architecture"
RAW = ROOT / "results" / "raw" / "024_true3d_ddb123_component_architecture"
AUDIT_MSH = RAW / "cvt3d_v4_audit.msh"

MAT_COLOR = {"air": "#e5e7eb", "oil": "#fde68a", "porcelain": "#b9895a",
             "element": "#3b82f6", "metal": "#6b7280", "resin": "#16a34a"}
CMAP = [MAT_COLOR[m] for m in MATERIALS]

# bounding-box audit regions: name -> (expected materials)
EXPECTED = {
    "head": {"metal", "oil", "air"},
    "tank": {"metal", "oil", "resin", "air"},
    "secondary_box": {"metal", "air"},
    "porcelain_column": {"porcelain", "oil", "element", "metal", "air"},
    "C1": {"element", "metal", "oil"},
    "tap": {"metal", "element", "oil"},
    "C2": {"element", "metal", "oil"},
    "emu": {"metal", "oil"},
    "air_far": {"air"},
}


def load_grid(msh):
    m = meshio.read(str(msh))
    pts = np.asarray(m.points, float)
    name3 = {(int(dim), int(tag)): nm for nm, (tag, dim) in m.field_data.items()}
    tets, phys = [], []
    for i, cb in enumerate(m.cells):
        if cb.type in ("tetra", "tetra10"):
            tets.append(np.asarray(cb.data, int)[:, :4])
            phys.append(np.asarray(m.cell_data["gmsh:physical"][i], int).reshape(-1))
    tets = np.vstack(tets); phys = np.concatenate(phys)
    names = [name3.get((3, int(t)), "air") for t in phys]
    matidx = np.array([MATERIALS.index(MAT_OF_V4.get(n, "air")) for n in names])
    cells = np.hstack([np.full((len(tets), 1), 4), tets]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA, np.uint8), pts)
    grid.cell_data["material"] = matidx
    grid.cell_data["name"] = np.array(names)
    return grid, pts, tets, names


def tet_volumes(pts, tets):
    p0 = pts[tets[:, 0]]
    M = np.stack([pts[tets[:, 1]] - p0, pts[tets[:, 2]] - p0, pts[tets[:, 3]] - p0], axis=1)
    return np.abs(np.linalg.det(M)) / 6.0


def cam_for(b, df=2.6, dir_vec=(1, -1, 0.35), focal=None):
    ctr = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
    f = np.asarray(focal) if focal is not None else ctr
    dv = np.asarray(dir_vec, float); dv /= np.linalg.norm(dv)
    return [tuple(f + df * diag * dv), tuple(f), (0, 0, 1)]


def mat_plot(body, fname, title, cam, win=(950, 1350), transparent_porc=False):
    p = pv.Plotter(off_screen=True, window_size=win); p.set_background("white")
    if transparent_porc:
        for i, mname in enumerate(MATERIALS):
            sub = body.threshold([i - 0.5, i + 0.5], scalars="material")
            if sub.n_cells == 0:
                continue
            op = 0.12 if mname == "porcelain" else (0.0 if mname == "air" else 1.0)
            if op > 0:
                p.add_mesh(sub, color=MAT_COLOR[mname], opacity=op, show_scalar_bar=False)
    else:
        p.add_mesh(body, scalars="material", cmap=CMAP, clim=[-0.5, len(MATERIALS) - 0.5],
                   n_colors=len(MATERIALS), show_scalar_bar=False)
    p.add_legend([(m, MAT_COLOR[m]) for m in MATERIALS], bcolor="white", size=(0.15, 0.20),
                 loc="upper right")
    p.add_text(title, position="upper_left", font_size=13, color="black")
    p.camera_position = cam
    p.screenshot(str(PROC / fname)); p.close()


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    if not AUDIT_MSH.is_file():
        print("Building FINE audit mesh (74 sheds) ...")
        meta = build_cvt_ddb123_3d_v4(AUDIT_MSH, curvature=5.0,
                                      lc={"shed": 0.018, "porcelain": 0.016, "oil": 0.026,
                                          "element_c2": 0.014, "air": 0.260})
        (PROC / "audit_mesh_meta.json").write_text(
            json.dumps({k: v for k, v in meta.items() if k != "parameters"}, indent=2,
                       default=str), encoding="utf-8")
    meta = json.loads((PROC / "audit_mesh_meta.json").read_text(encoding="utf-8"))
    bbox = meta["component_bbox"]
    p = meta["parameters"] if "parameters" in meta else None

    print("Loading mesh ...")
    grid, pts, tets, names = load_grid(AUDIT_MSH)
    vols = tet_volumes(pts, tets)
    matidx = grid.cell_data["material"]
    centroids = pts[tets].mean(axis=1)
    rc = np.hypot(centroids[:, 0], centroids[:, 1]); zc = centroids[:, 2]

    # ---------- bounding-box material audit ----------
    def region_box(name):
        if name == "head":
            return bbox.get("head_metal")
        if name == "tank":
            return bbox.get("tank_metal")
        if name == "secondary_box":
            return bbox.get("secondary_box")
        if name == "porcelain_column":
            return bbox.get("porcelain")
        if name == "C1":
            return bbox.get("element_c1")
        if name == "tap":
            return bbox.get("foil_tap")
        if name == "C2":
            return bbox.get("element_c2")
        if name == "emu":
            return bbox.get("base_emu")
        if name == "air_far":
            return [0.5, 0.9, 0.5, 0.9, 0.5, 1.0]  # a far-air sample box
        return None

    audit_rows = []
    for region, exp in EXPECTED.items():
        bb = region_box(region)
        if bb is None:
            audit_rows.append({"region": region, "status": "MISSING bbox"}); continue
        pad = 1e-4
        inside = ((centroids[:, 0] >= bb[0] - pad) & (centroids[:, 0] <= bb[3] + pad) &
                  (centroids[:, 1] >= bb[1] - pad) & (centroids[:, 1] <= bb[4] + pad) &
                  (centroids[:, 2] >= bb[2] - pad) & (centroids[:, 2] <= bb[5] + pad))
        vol_in = {}
        for i, mname in enumerate(MATERIALS):
            v = float(vols[inside & (matidx == i)].sum())
            if v > 0:
                vol_in[mname] = v
        actual = set(vol_in)
        unexpected = sorted(actual - exp)
        # porcelain inside head/tank is the FAIL condition (unless legitimate interface)
        porc_frac = vol_in.get("porcelain", 0.0) / max(sum(vol_in.values()), 1e-30)
        fail = False
        if region in ("head", "tank", "secondary_box") and porc_frac > 0.02:
            fail = True
        audit_rows.append({
            "region": region, "expected": sorted(exp), "actual": sorted(actual),
            "unexpected": unexpected, "porcelain_frac": round(porc_frac, 4),
            "volumes_m3": {k: round(v, 6) for k, v in sorted(vol_in.items())},
            "pass": (not fail)})
    material_audit = {"regions": audit_rows,
                      "total_material_volumes_m3": meta["material_volumes_m3"],
                      "note": "porcelain in head/tank/box bbox > 2% volume => FAIL"}
    (PROC / "material_audit.json").write_text(json.dumps(material_audit, indent=2), encoding="utf-8")
    fails = [r["region"] for r in audit_rows if r.get("pass") is False]
    print("Material audit FAILS:", fails or "none")

    # ---------- figures ----------
    device = grid.threshold([0.5, len(MATERIALS) - 0.5], scalars="material")  # exclude air(0)
    db = device.extract_surface().bounds
    full = cam_for(db); face = cam_for(db, dir_vec=(0.001, -1, 0))

    # external (metal+porcelain visible)
    ext = grid.threshold([1.5, 5.5], scalars="material")  # porcelain..resin (exclude air,oil)
    mat_plot(ext, "ddb123_external_reference_match.png",
             "DDB-123 external (74 alternating sheds, head, tank, box)", full)
    # half-section (clip) of device, material
    half = device.clip(normal=(0, 1, 0), origin=(0, 0, 0))
    mat_plot(half, "ddb123_material_half_section.png", "half-section (material)", full)
    # axial slice
    sl = device.slice(normal=(0, 1, 0))
    mat_plot(sl, "ddb123_material_axial_slice.png", "material axial slice (through axis)", face)
    # transparent architecture (component cutaway base)
    mat_plot(device.clip(normal=(0, 1, 0), origin=(0, 0, 0)),
             "ddb123_transparent_architecture.png", "transparent porcelain, internals visible",
             full, transparent_porc=True)

    # component-labeled cutaway: axial slice + matplotlib labels at component centroids
    import matplotlib.image as mpimg
    mat_plot(sl, "_tmp_axial.png", "", face)
    img = mpimg.imread(str(PROC / "_tmp_axial.png"))
    fig, ax = plt.subplots(figsize=(11, 9)); ax.imshow(img); ax.axis("off")
    ax.set_title("DDB-123 component-labeled cutaway (axial slice)", fontsize=13)
    labels = [("primary terminal", 0.50, 0.07), ("oil-volume compensating head", 0.62, 0.16),
              ("porcelain insulator + 74 alternating sheds", 0.78, 0.34),
              ("C1 equivalent capacitor", 0.40, 0.40), ("intermediate voltage tap", 0.30, 0.66),
              ("C2 equivalent capacitor", 0.40, 0.70), ("resin HF-terminal exit", 0.30, 0.74),
              ("grounded tank", 0.66, 0.86), ("inductive VT / EMU", 0.42, 0.84),
              ("series reactor", 0.42, 0.80), ("secondary terminal box", 0.80, 0.88)]
    for txt, fx, fy in labels:
        ax.annotate(txt, xy=(fx * img.shape[1], fy * img.shape[0]),
                    xytext=(0.04 * img.shape[1] if fx < 0.5 else 0.80 * img.shape[1],
                            fy * img.shape[0]),
                    fontsize=8.5, color="black",
                    arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.0),
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
    fig.tight_layout(); fig.savefig(PROC / "ddb123_component_cutaway_labeled.png", dpi=150)
    plt.close(fig)
    (PROC / "_tmp_axial.png").unlink(missing_ok=True)

    # horizontal slices at head / C1 / tap / C2 / tank / box
    c1z = meta["c1_z"]; c2z = meta["c2_z"]; tapz = meta["tap_z"]
    levels = [("head", 1.70), ("C1", 0.5 * (c1z[0] + c1z[1])), ("tap", tapz),
              ("C2", 0.5 * (c2z[0] + c2z[1])), ("tank/EMU", 0.25), ("box", 0.20)]
    pl = pv.Plotter(off_screen=True, shape=(2, 3), window_size=(1500, 1100)); pl.set_background("white")
    for k, (lab, z) in enumerate(levels):
        hs = grid.slice(normal=(0, 0, 1), origin=(0, 0, z))
        pl.subplot(k // 3, k % 3)
        if hs.n_cells:
            pl.add_mesh(hs, scalars="material", cmap=CMAP, clim=[-0.5, len(MATERIALS) - 0.5],
                        n_colors=len(MATERIALS), show_scalar_bar=False)
        pl.add_text(f"{lab}  z={z:.2f}", position="upper_edge", font_size=10, color="black")
        pl.view_xy()
    pl.screenshot(str(PROC / "ddb123_horizontal_slices.png")); pl.close()

    # close-ups (clip z-bands, 3/4 view) + shed lip + shed pattern
    def closeup(zlo, zhi, fname, title, body=device):
        sub = body.clip_box([0, db[1], db[2], db[3], zlo, zhi], invert=True)
        mat_plot(sub, fname, title, cam_for([0, db[1], db[2], db[3], zlo, zhi], df=2.0),
                 win=(1000, 950))
    closeup(1.55, db[5], "ddb123_head_closeup.png", "top head / primary terminal")
    closeup(0.50, 0.95, "ddb123_stack_tap_cutaway.png", "C1 / tap / C2 stack region",
            body=device.clip(normal=(0, 1, 0), origin=(0, 0, 0)))
    closeup(db[4], 0.58, "ddb123_tank_emu_terminal_box_closeup.png", "tank / EMU / secondary box")
    closeup(0.95, 1.25, "ddb123_shed_pattern_closeup.png", "alternating shed pattern")
    # shed lip extreme close-up (axial slice zoom)
    lipsl = sl.clip_box([0.10, 0.18, -0.01, 0.01, 1.05, 1.18], invert=True)
    mat_plot(lipsl, "ddb123_shed_lip_closeup.png", "shed lip geometry (rounded)",
             cam_for([0.10, 0.18, 0, 0, 1.05, 1.18], dir_vec=(0.001, -1, 0), df=2.0), win=(900, 900))

    # mesh quality histogram + shed zoom
    qnpz = Path(str(AUDIT_MSH) + ".quality.npz")
    q = np.load(qnpz)["tet_q"] if qnpz.is_file() else np.array([meta["quality_minSICN"]["min"]])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(q, bins=80, color="#2563eb", alpha=0.8)
    for pc, c in [(1, "#b91c1c"), (5, "#ea580c"), (50, "#16a34a")]:
        ax.axvline(np.percentile(q, pc), color=c, ls="--", lw=1.4, label=f"p{pc}={np.percentile(q,pc):.3f}")
    ax.axvline(0.05, color="k", ls=":", label="sliver 0.05")
    ax.set_yscale("log"); ax.set_xlabel("minSICN"); ax.set_ylabel("count (log)")
    ax.set_title(f"DDB-123 mesh quality (n={len(q)}, min={q.min():.3f}, mean={q.mean():.3f})")
    ax.legend(); fig.tight_layout(); fig.savefig(PROC / "ddb123_mesh_quality.png", dpi=150)
    plt.close(fig)

    surf = device.clip_box([0, db[1], db[2], db[3], 0.95, 1.25], invert=True).extract_surface()
    pp = pv.Plotter(off_screen=True, window_size=(1000, 950)); pp.set_background("white")
    pp.add_mesh(surf, color="#cbd5e1", show_edges=True, edge_color="#111827", line_width=0.3)
    pp.add_text("mesh near alternating sheds", position="upper_left", font_size=12, color="black")
    pp.camera_position = cam_for([0, db[1], db[2], db[3], 0.95, 1.25], df=2.0)
    pp.screenshot(str(PROC / "ddb123_mesh_shed_zoom.png")); pp.close()

    # ---------- clean fields from solved VTU ----------
    try:
        from run_elmer_case import _parse_mesh_names
        vtu = sorted((RAW / "run3d").glob("**/case*.vtu"))[-1]
        ids = _parse_mesh_names(RAW / "elmer" / "mesh" / "mesh.names")
        id2name = {bid: n for n, bid in ids["bodies"].items()}
        air_id = ids["bodies"]["air"]
        m2 = pv.read(str(vtu))
        gid = np.asarray(m2.cell_data["GeometryIds"], int).reshape(-1)
        gr = m2.compute_derivative(scalars="potential re", gradient="gr")["gr"]
        gi = m2.compute_derivative(scalars="potential im", gradient="gi")["gi"]
        em = np.sqrt((gr ** 2).sum(1) + (gi ** 2).sum(1))
        m2.point_data["logE"] = np.log10(np.clip(em, 1.0, None))
        dev_b = [-0.23, 0.23, -0.23, 0.23, 0.0, 1.85]   # known device extent (avoids orphan bounds)
        ctr = np.array([0.0, 0.0, 0.9])
        facev = cam_for(dev_b, dir_vec=(0.001, -1, 0))
        slv = m2.slice(normal=(0, 1, 0), origin=tuple(ctr))
        for fn, sc, cm, ttl in [("ddb123_clean_phi.png", "potential re", "coolwarm", "Re(phi) [V]"),
                                ("ddb123_clean_Emag.png", "logE", "inferno", "log10 |E| [V/m]")]:
            pl = pv.Plotter(off_screen=True, window_size=(950, 1350)); pl.set_background("white")
            pl.add_mesh(slv, scalars=sc, cmap=cm, scalar_bar_args={"title": ttl, "vertical": True})
            pl.add_text(ttl + " (clean, axial slice)", position="upper_left", font_size=13, color="black")
            pl.camera_position = facev; pl.screenshot(str(PROC / fn)); pl.close()
        print("Clean-field figures done.")
    except Exception as e:
        print("Clean-field figures skipped (solve VTU not ready):", e)

    # geometry parameters + mesh quality JSON
    (PROC / "geometry_parameters.json").write_text(json.dumps({
        "total_height_mm": 1830, "A_base_mm": 450, "creepage_mm": meta["creepage_mm"],
        "n_sheds": meta["n_sheds"], "shed_overhang_large_mm": meta["shed_overhang_large_mm"],
        "shed_overhang_small_mm": meta["shed_overhang_small_mm"], "shed_pitch_mm": meta["shed_pitch_mm"],
        "shed_lip_mm": meta["shed_lip_mm"], "tap_z_m": meta["tap_z"], "c1_z_m": meta["c1_z"],
        "c2_z_m": meta["c2_z"], "component_bbox": bbox,
        "targets": {"C_pF": 5600, "C_high_pF": 14000, "creepage_mm": 3075, "ratio": 1 / 6}},
        indent=2, default=str), encoding="utf-8")
    (PROC / "mesh_quality.json").write_text(json.dumps({
        "n_nodes": meta["n_nodes"], "n_tets": meta["n_tets"], "n_boundary_tris": meta["n_boundary_tris"],
        "quality_minSICN": meta["quality_minSICN"], "n_slivers_lt_0.05": meta["n_slivers_lt_0.05"],
        "insulator_surface_tri_quality": meta["insulator_surface_tri_quality"],
        "worst20": meta["worst20"]}, indent=2, default=str), encoding="utf-8")
    print("Audit complete:", PROC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

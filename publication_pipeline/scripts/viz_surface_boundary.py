"""Task 013.6 - visualize the surface-conductance boundary (insulator_surface).

The pollution model is now a Surface Conductance BC on the `insulator_surface`
boundary (Task 013). This script (no solver/geometry change) builds the rounded
clean CVT mesh and highlights that boundary in red over the material map, showing
it traces the true creepage path (wall + shed tops + undersides + rounded lips)
and that NO straight radial pollution volume skin remains.

Outputs: surface_conductance_boundary.png, surface_conductance_zoom.png,
surface_boundary_report.md  (under results/processed/0136_surface_boundary/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from run_cvt import material_of_body  # noqa: E402

PROC = ROOT / "results" / "processed" / "0136_surface_boundary"
RAW = ROOT / "results" / "raw" / "0136_surface_boundary"

MATERIALS = ["air", "oil", "porcelain", "element", "metal", "pollution"]
MAT_COLORS = ["#eef2f7", "#fde9a8", "#d8c19a", "#b9d4f5", "#9aa3ad", "#dc2626"]
MAT_LABELS = {"air": "air", "oil": "oil", "porcelain": "porcelain", "element": "capacitor element",
              "metal": "metal", "pollution": "pollution"}
RED = "#dc2626"


def _load(msh: Path, meta: dict):
    m = meshio.read(str(msh))
    pts = np.asarray(m.points)[:, :2]
    phys = m.cell_data.get("gmsh:physical")
    tris, tri_tag, lines, line_tag = [], [], [], []
    for i, b in enumerate(m.cells):
        tag = np.asarray(phys[i]).reshape(-1) if phys is not None else None
        if b.type == "triangle":
            tris.append(np.asarray(b.data)); tri_tag.append(tag)
        elif b.type == "line":
            lines.append(np.asarray(b.data)); line_tag.append(tag)
    tris = np.vstack(tris); tri_tag = np.concatenate(tri_tag)
    lines = np.vstack(lines); line_tag = np.concatenate(line_tag)
    # tag -> name maps from the build metadata (per dimension)
    tag2name_2d = {v["tag"]: n for n, v in meta["physical_groups"].items() if v["dim"] == 2}
    ins_tag = meta["physical_groups"]["insulator_surface"]["tag"]
    return pts, tris, tri_tag, lines, line_tag, tag2name_2d, ins_tag


def _cell_mat(tri_tag, tag2name_2d):
    return np.array([MATERIALS.index(material_of_body(tag2name_2d[t])) for t in tri_tag])


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    p = CVTParams()
    msh = RAW / "cvt.msh"
    meta = build_cvt_ddb123(msh, p, rounded=True)  # default: NO volume pollution body

    has_pollution_body = "pollution_layer" in meta["physical_groups"]
    pts, tris, tri_tag, lines, line_tag, tag2name_2d, ins_tag = _load(msh, meta)
    ins = lines[line_tag == ins_tag]

    # discretized creepage = summed meridional length of the insulator_surface lines
    seg = pts[ins]
    creepage_mesh_mm = float(np.sum(np.hypot(seg[:, 1, 0] - seg[:, 0, 0], seg[:, 1, 1] - seg[:, 0, 1]))) * 1e3
    creepage_analytic_mm = meta["creepage_distance_mm"]

    cmap = ListedColormap(MAT_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(MATERIALS) + 0.5), cmap.N)
    cm = _cell_mat(tri_tag, tag2name_2d)

    def _draw(ax, mirror, lw):
        if mirror:
            P2 = np.vstack([pts, pts * np.array([-1.0, 1.0])])
            T2 = np.vstack([tris, tris + len(pts)])
            ax.tripcolor(mtri.Triangulation(P2[:, 0], P2[:, 1], T2),
                         facecolors=np.concatenate([cm, cm]), cmap=cmap, norm=norm, shading="flat")
            segs = [seg, seg * np.array([[-1.0, 1.0]])]
        else:
            ax.tripcolor(mtri.Triangulation(pts[:, 0], pts[:, 1], tris),
                         facecolors=cm, cmap=cmap, norm=norm, shading="flat")
            segs = [seg]
        for s in segs:
            ax.add_collection(LineCollection(s, colors=RED, linewidths=lw, zorder=5))

    # --- Fig 1: full device, insulator_surface in red (mirrored) -------------
    fig, ax = plt.subplots(figsize=(6.8, 9.5))
    _draw(ax, mirror=True, lw=1.4)
    ax.set_xlim(-0.30, 0.30); ax.set_ylim(-0.03, p.total_height + 0.03); ax.set_aspect("equal")
    ax.set_xlabel("r [m] (mirrored)"); ax.set_ylabel("z [m]")
    ax.set_title("DDB-123: insulator_surface (Surface Conductance BC) in red")
    handles = [Patch(facecolor=MAT_COLORS[i], edgecolor="k", label=MAT_LABELS[MATERIALS[i]]) for i in sorted(set(cm))]
    handles.append(Line2D([0], [0], color=RED, lw=2.5, label="insulator_surface (sigma_s)"))
    ax.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.92)
    ax.text(0.02, 0.02,
            f"creepage path = {creepage_mesh_mm:.0f} mm (meshed)\n"
            f"analytic {creepage_analytic_mm:.0f} mm / datasheet 3075 mm\n"
            f"{len(ins)} surface (line) elements",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
    fig.tight_layout()
    fig.savefig(PROC / "surface_conductance_boundary.png", dpi=200)
    plt.close(fig)

    # --- Fig 2: zoom on several sheds ---------------------------------------
    centers = p.shed_centers()
    zc = centers[len(centers) // 2]
    pitch = (p.stack_z_hi - p.tank_height) / p.n_sheds
    rtip = p.porcelain_inner_radius + p.porcelain_thickness + p.shed_overhang
    fig, ax = plt.subplots(figsize=(8.5, 8))
    _draw(ax, mirror=False, lw=2.6)
    ax.set_xlim(p.porcelain_inner_radius - 0.006, rtip + 0.012)
    ax.set_ylim(zc - 3.2 * pitch, zc + 3.2 * pitch); ax.set_aspect("equal")
    ax.set_xlabel("r [m]"); ax.set_ylabel("z [m]")
    ax.set_title("Zoom: red boundary follows wall -> shed top -> rounded lip -> underside -> wall")
    ax.legend(handles=[Line2D([0], [0], color=RED, lw=3, label="insulator_surface (Surface Conductance)"),
                       Patch(facecolor=MAT_COLORS[2], edgecolor="k", label="porcelain"),
                       Patch(facecolor=MAT_COLORS[1], edgecolor="k", label="oil"),
                       Patch(facecolor=MAT_COLORS[0], edgecolor="k", label="air")],
              loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PROC / "surface_conductance_zoom.png", dpi=200)
    plt.close(fig)

    # --- report -------------------------------------------------------------
    bodies = sorted(n for n, v in meta["physical_groups"].items() if v["dim"] == 2)
    lines_md = [
        "# Task 013.6 - Surface-Conductance Boundary Visualization",
        "",
        "The pollution model is a Surface Conductance BC on the `insulator_surface`",
        "boundary (Task 013). These figures (no solve, no geometry change) highlight that",
        "boundary in red on the rounded clean CVT.",
        "",
        "## What the figures show",
        "- `surface_conductance_boundary.png` - full mirrored device; the red curve is the",
        "  entire air-facing insulator surface where sigma_s is applied.",
        "- `surface_conductance_zoom.png` - several sheds; the red boundary hugs the wall",
        "  between sheds, runs out along each shed top, around the rounded lip, back along",
        "  the underside, and onto the wall again - i.e. it follows the true creepage path.",
        "",
        "## Creepage path length",
        f"- meshed insulator_surface (summed meridional segment length): **{creepage_mesh_mm:.0f} mm**",
        f"- analytic shed-profile creepage: {creepage_analytic_mm:.0f} mm",
        f"- datasheet creepage: 3075 mm",
        f"- insulator_surface line elements: {len(ins)}",
        "",
        "## No straight radial pollution skin",
        f"- dim-2 bodies present: {bodies}",
        f"- `pollution_layer` volume body present: **{has_pollution_body}** "
        f"(should be False - pollution is now the surface BC, not a volume skin)",
        "",
        "The boundary is one continuous creepage curve, not a straight radial band, "
        "confirming the surface-conductance model replaced the old volume skin.",
        "",
    ]
    (PROC / "surface_boundary_report.md").write_text("\n".join(lines_md), encoding="utf-8")

    print(f"insulator_surface: {len(ins)} line elements")
    print(f"creepage: mesh={creepage_mesh_mm:.0f} mm, analytic={creepage_analytic_mm:.0f} mm, datasheet=3075 mm")
    print(f"pollution_layer body present: {has_pollution_body} (expected False)")
    print(f"bodies: {bodies}")
    print(f"Figures + report in: {PROC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

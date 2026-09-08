"""Task 024 - clean categorical material renders + TRUE central half-section.

Fixes the cutaway ambiguity:
  - cell-wise FLAT categorical colours (lighting off, interpolate_before_map off, cell data)
  - interior close-ups use planar SLICES (no clip_box slivers)
  - true half-section: clip the solid by the central y=0 plane, then place the camera on the
    REMOVED side (auto-detected from the kept-half bounds) so it looks INTO the cut face
  - legend rendered OUTSIDE the image (matplotlib side panel)

Figures: ddb123_true_half_section_material/_labeled/_closeup_stack/_closeup_tank,
ddb123_head_closeup_categorical/_tank_closeup_categorical/_stack_tap_categorical/
_resin_hf_exit_categorical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT.parent), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from audit_ddb123 import load_grid, MATERIALS, MAT_COLOR  # noqa: E402

pv.OFF_SCREEN = True
PROC = ROOT / "results" / "processed" / "024_true3d_ddb123_component_architecture"
RAW = ROOT / "results" / "raw" / "024_true3d_ddb123_component_architecture"
AUDIT_MSH = RAW / "cvt3d_v4_audit.msh"
CMAP = [MAT_COLOR[m] for m in MATERIALS]
TMP = PROC / "_tmp_cat.png"


def render_cat(body, cam, win=(950, 1300)):
    p = pv.Plotter(off_screen=True, window_size=win); p.set_background("white")
    p.add_mesh(body, scalars="material", cmap=CMAP, clim=[-0.5, len(MATERIALS) - 0.5],
               n_colors=len(MATERIALS), interpolate_before_map=False, lighting=False,
               show_scalar_bar=False)
    p.camera_position = cam
    p.screenshot(str(TMP)); p.close()


def composite(out_name, title, labels=None):
    img = mpimg.imread(str(TMP))
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_axes([0.0, 0.0, 0.80, 0.94]); ax.imshow(img); ax.axis("off")
    ax.set_title(title, fontsize=12)
    lax = fig.add_axes([0.81, 0.35, 0.18, 0.32]); lax.axis("off")
    lax.legend(handles=[Patch(facecolor=MAT_COLOR[m], edgecolor="k", label=m) for m in MATERIALS],
               loc="center", frameon=True, title="material", fontsize=11)
    if labels:
        H, W = img.shape[0], img.shape[1]
        for txt, fx, fy, lx in labels:
            ax.annotate(txt, xy=(fx * W, fy * H), xytext=(lx * W, fy * H), fontsize=8.5,
                        color="black", arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.0),
                        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    fig.savefig(PROC / out_name, dpi=150); plt.close(fig)
    TMP.unlink(missing_ok=True)


def main() -> int:
    grid, pts, tets, names = load_grid(AUDIT_MSH)
    device = grid.threshold([0.5, len(MATERIALS) - 0.5], scalars="material")  # exclude air(0)

    # ---- TRUE central half-section: clip by y=0, camera on the REMOVED side ----
    half = device.clip(normal=(0, 1, 0), origin=(0, 0, 0))
    yb = half.bounds[2], half.bounds[3]
    cam_sign = +1.0 if abs(yb[1]) < abs(yb[0]) else -1.0   # removed side = larger |y|

    def hs_cam(zmid, dist):
        return [(0.0, cam_sign * dist, zmid), (0.0, 0.0, zmid), (0.0, 0.0, 1.0)]

    render_cat(half, hs_cam(0.9, 5.0))
    composite("ddb123_true_half_section_material.png",
              "TRUE central half-section (front 180 deg removed; camera looks into the cut face)")
    render_cat(half, hs_cam(0.9, 5.0))
    composite("ddb123_true_half_section_labeled.png", "TRUE central half-section - labeled",
              labels=[("primary terminal", 0.50, 0.09, 0.05), ("oil-volume head (metal shell + oil)", 0.55, 0.17, 0.05),
                      ("porcelain + 74 sheds", 0.66, 0.40, 0.84), ("oil annulus", 0.58, 0.45, 0.84),
                      ("C1 element", 0.50, 0.42, 0.05), ("intermediate tap", 0.50, 0.63, 0.05),
                      ("C2 element", 0.52, 0.67, 0.05), ("resin HF-exit", 0.52, 0.71, 0.05),
                      ("tank: EMU / reactor", 0.55, 0.85, 0.84), ("secondary box", 0.72, 0.88, 0.84)])
    render_cat(half, hs_cam(0.95, 2.0), win=(900, 1150))
    composite("ddb123_true_half_section_closeup_stack.png",
              "Half-section close-up: C1 / tap / C2 / oil annulus / porcelain")
    render_cat(half, hs_cam(0.28, 1.7), win=(1000, 950))
    composite("ddb123_true_half_section_closeup_tank.png",
              "Half-section close-up: tank / EMU / reactor / resin HF-exit / secondary box")

    # ---- clean categorical interior close-ups (planar slices) ----
    sl = device.slice(normal=(0, 1, 0))

    def zoom(zlo, zhi, rmax=0.24):
        return sl.clip_box([0, rmax, -0.02, 0.02, zlo, zhi], invert=True)

    def face_cam(zlo, zhi, rmax=0.24, df=2.3):
        zmid = 0.5 * (zlo + zhi); diag = np.hypot(2 * rmax, zhi - zlo)
        return [(0.0, df * diag, zmid), (0.0, 0.0, zmid), (0.0, 0.0, 1.0)]

    render_cat(zoom(1.55, 1.86, 0.21), face_cam(1.55, 1.86, 0.21), win=(1000, 900))
    composite("ddb123_head_closeup_categorical.png",
              "Top head = metal shell + oil cavity (categorical, flat).\n"
              "Brown porcelain BELOW z=1.61 is the adjacent insulator column, NOT inside the head.")
    render_cat(zoom(0.0, 0.58, 0.24), face_cam(0.0, 0.58, 0.24), win=(1000, 900))
    composite("ddb123_tank_closeup_categorical.png",
              "Grounded tank = metal shell + oil + EMU/reactor/aux (metal) + resin (green) (flat)")
    render_cat(zoom(0.50, 1.02, 0.14), face_cam(0.50, 1.02, 0.14), win=(820, 1100))
    composite("ddb123_stack_tap_categorical.png",
              "Stack: C1/C2 element (blue), intermediate tap (metal), oil (tan), porcelain (brown)")
    render_cat(zoom(0.50, 0.60, 0.10), face_cam(0.50, 0.60, 0.10, df=2.6), win=(1000, 820))
    composite("ddb123_resin_hf_exit_categorical.png",
              "Resin HF-terminal exit (green) at the stack base / tank top, with ground electrode")
    print("Wrote 8 clean categorical / half-section figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

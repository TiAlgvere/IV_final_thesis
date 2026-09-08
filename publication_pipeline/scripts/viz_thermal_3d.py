"""Task 021.5 - 3D revolved renders of the Task 021 temperature fields.

Pure visualization (no new physics, no solver change): loads the saved one-way
electrothermal T-fields (results/processed/021_electrothermal/Tfield_*.npz), revolves
the axisymmetric solid cross-section 270 deg (cutaway) with pyvista, maps T [C] onto
the solid CVT (air excluded), marks the hot-spot, and writes thesis PNGs:

  thermal_3d_healthy.png, thermal_3d_C1_loss.png, thermal_3d_pollution.png,
  thermal_3d_comparison.png (common clim), + a stack hot-spot close-up and a
  porcelain creepage close-up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "processed" / "021_electrothermal"
PROC = ROOT / "results" / "processed" / "021_5_thermal_3d"

pv.OFF_SCREEN = True
CFG = json.loads((SRC / "thermal_config.json").read_text(encoding="utf-8"))
T_AMB = CFG["T_amb_C"]; HOT = CFG["insul_hotspot_C"]
REV_ANGLE, REV_RES = 270.0, 120
CMAP = "inferno"


def build_solid(tag):
    """Revolved 3D solid with T [C] point data; returns (solid, hotspot_xyz, Tmax_C, zmid)."""
    d = np.load(SRC / f"Tfield_{tag}.npz")
    pts, tris, T = d["pts"], d["tris"], d["T"]
    used = np.unique(tris)
    remap = -np.ones(len(pts), int); remap[used] = np.arange(len(used))
    dpts, dtris = pts[used], remap[tris]
    Tc = T_AMB + T[used]

    P3 = np.column_stack([dpts[:, 0], np.zeros(len(used)), dpts[:, 1]])
    faces = np.hstack([np.full((len(dtris), 1), 3), dtris]).ravel()
    poly = pv.PolyData(P3, faces)
    poly.point_data["T"] = Tc
    solid = poly.extrude_rotate(resolution=REV_RES, angle=REV_ANGLE, capping=True)

    im = int(np.nanargmax(T))
    hot = (float(pts[im, 0]), 0.0, float(pts[im, 1]))   # on the theta=0 cut face
    zmid = 0.5 * (dpts[:, 1].min() + dpts[:, 1].max())
    return solid, hot, float(np.nanmax(Tc)), zmid


def _bar(title):
    return {"title": title, "title_font_size": 26, "label_font_size": 20, "n_labels": 6,
            "vertical": True, "position_x": 0.85, "position_y": 0.22, "height": 0.6, "width": 0.06}


def render(tag, fname, title, *, clim=None, cam=None, mark=True):
    solid, hot, tmax, zmid = build_solid(tag)
    if clim is None:
        clim = [T_AMB, max(tmax, T_AMB + 1)]
    p = pv.Plotter(off_screen=True, window_size=(950, 1350))
    p.set_background("white")
    p.add_mesh(solid, scalars="T", cmap=CMAP, clim=clim, smooth_shading=True,
               scalar_bar_args=_bar("T [C]"))
    if mark:
        p.add_mesh(pv.Sphere(radius=0.03, center=hot), color="cyan")
        p.add_mesh(pv.Sphere(radius=0.03, center=hot), color="black", style="wireframe", line_width=2)
    p.add_text(title, position="upper_left", font_size=15, color="black")
    p.camera_position = cam or [(2.7, -2.4, zmid + 0.95), (0.0, 0.0, zmid), (0.0, 0.0, 1.0)]
    p.screenshot(str(PROC / fname))
    p.close()
    return tmax, hot, zmid


def render_comparison(tags_titles, fname, clim):
    p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1800, 1300), border=False)
    p.set_background("white")
    for k, (tag, title) in enumerate(tags_titles):
        solid, hot, tmax, zmid = build_solid(tag)
        p.subplot(0, k)
        p.add_mesh(solid, scalars="T", cmap=CMAP, clim=clim, smooth_shading=True,
                   show_scalar_bar=(k == 2), scalar_bar_args=_bar("T [C]") if k == 2 else None)
        p.add_mesh(pv.Sphere(radius=0.03, center=hot), color="cyan")
        p.add_mesh(pv.Sphere(radius=0.03, center=hot), color="black", style="wireframe", line_width=2)
        p.add_text(title, position="upper_edge", font_size=12, color="black")
        p.camera_position = [(2.7, -2.4, zmid + 0.95), (0.0, 0.0, zmid), (0.0, 0.0, 1.0)]
    p.screenshot(str(PROC / fname))
    p.close()


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)

    tmax_h, *_ = render("healthy", "thermal_3d_healthy.png",
                        "Healthy baseline (dielectric loss only)")
    tmax_c, hot_c, zmid_c = render("C1_loss_0.1", "thermal_3d_C1_loss.png",
                                   "C1 dielectric loss  (tan-delta = 0.10)")
    tmax_c2, *_ = render("C2_loss_0.05", "thermal_3d_C2_loss.png",
                         "C2 dielectric loss  (tan-delta = 0.05)")
    tmax_p, hot_p, zmid_p = render("pollution_1e-06", "thermal_3d_pollution.png",
                                   "Heavy full-creepage pollution  (sigma_s = 1e-6 S)")

    # comparison: common clim from ambient to the insulation hot-spot limit
    render_comparison(
        [("healthy", "healthy"), ("C1_loss_0.1", "C1 dielectric loss (tand 0.10)"),
         ("pollution_1e-06", "pollution (sigma_s 1e-6)")],
        "thermal_3d_comparison.png", clim=[T_AMB, HOT])

    # close-up 1: capacitor-stack hot-spot (internal loss heats the stack volume)
    cam_stack = [(1.05, -0.95, hot_c[2] + 0.18), (0.0, 0.0, hot_c[2]), (0.0, 0.0, 1.0)]
    render("C1_loss_0.1", "thermal_3d_stack_closeup.png",
           "Close-up: capacitor-stack hot-spot (internal loss)", cam=cam_stack)

    # close-up 2: porcelain creepage heating (surface pollution heats the creepage path)
    cam_creep = [(1.15, -1.05, zmid_p + 0.25), (0.10, 0.0, zmid_p), (0.0, 0.0, 1.0)]
    render("pollution_1e-06", "thermal_3d_creepage_closeup.png",
           "Close-up: porcelain creepage heating (pollution)", cam=cam_creep)

    (PROC / "thermal_3d_report.md").write_text("\n".join([
        "# Task 021.5 - 3D Revolved Thermal Renders (DDB-123)",
        "",
        f"The verified one-way electrothermal T-fields from Task 021 revolved {REV_ANGLE:.0f} deg",
        "(cutaway) with pyvista; the solid CVT only (surrounding air excluded), T mapped in",
        f"degrees C (ambient T_amb = {T_AMB:.0f} C). Hot-spot marked with a sphere. Pure",
        "visualization - no new physics, the thermal solver is unchanged.",
        "",
        "## Figures",
        "- `thermal_3d_healthy.png` - baseline dielectric loss only; the stack is mildly warm",
        f"  (peak ~{tmax_h:.0f} C), well below any limit.",
        "- `thermal_3d_C1_loss.png` - C1 dielectric loss (tan-delta 0.10): **internal dielectric",
        f"  loss heats the capacitor-stack VOLUME**; the hot-spot sits in the mid-stack",
        f"  (peak ~{tmax_c:.0f} C, far above the {HOT:.0f} C insulation hot-spot limit).",
        "- `thermal_3d_C2_loss.png` - C2 dielectric loss (tan-delta 0.05): the hot-spot localises",
        f"  to the C2 section (lower stack, near the tap), peak ~{tmax_c2:.0f} C - the same",
        "  mechanism, but the C2 loss concentrates in only 2 elements, so a lower total power",
        "  still gives a high local temperature (see Task 021).",
        "- `thermal_3d_pollution.png` - heavy full-creepage pollution: **surface pollution heats",
        f"  the porcelain CREEPAGE path** (outer wall / sheds), peak ~{tmax_p:.0f} C, while the",
        "  core stays cooler - a distinctly different (surface) heating pattern.",
        "- `thermal_3d_comparison.png` - the three side by side on a **common color scale**",
        f"  ({T_AMB:.0f}-{HOT:.0f} C = ambient to insulation limit): healthy stays dark/cool,",
        "  pollution warms the surface, and the C1-loss case saturates the scale (it is over",
        "  the insulation limit throughout the stack).",
        "- `thermal_3d_stack_closeup.png` - zoom on the capacitor-stack hot-spot (internal loss).",
        "- `thermal_3d_creepage_closeup.png` - zoom on the porcelain creepage heating (pollution).",
        "",
        "## Interpretation (consistent with Task 021)",
        "- **Internal dielectric loss heats the capacitor stack volume** - the hot-spot is",
        "  interior, in the lossy elements, and escalates with tan-delta.",
        "- **Surface pollution heats the porcelain creepage path** - the heat is deposited on the",
        "  outer surface (dry-band region), a different spatial signature from internal loss.",
        "- **Ratio faults (capacitance change, disc short) are electrically visible but thermally",
        "  quiet** - they add ~no loss, so they are not rendered here (they look like the healthy",
        "  baseline); see the Task 021 summary table.",
        "",
        "These are **one-way, steady-state** thermal results (electrical loss -> temperature, no",
        "feedback): they show where each fault deposits heat and the resulting steady rise, NOT a",
        "coupled thermal-runaway prediction. Absolute temperatures depend on the documented",
        "thermal assumptions; the spatial pattern and relative ranking are the robust message.",
        "",
    ]), encoding="utf-8")
    print(f"Wrote 7 renders + thermal_3d_report.md in {PROC}")
    print(f"  peak T: healthy {tmax_h:.0f} C | C1_loss {tmax_c:.0f} C | C2_loss {tmax_c2:.0f} C | "
          f"pollution {tmax_p:.0f} C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

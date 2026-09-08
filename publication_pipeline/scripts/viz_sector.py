"""Task 026 - sector visualizations: azimuthal patch placement + surface-loss renders."""

from __future__ import annotations

import sys
from pathlib import Path

import meshio
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT.parent), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pv.OFF_SCREEN = True
PROC = ROOT / "results" / "processed" / "026_azimuthal_surface_conductance_validation"
RAW = ROOT / "results" / "raw" / "026_azimuthal_surface_conductance_validation"
MSH = ROOT / "results" / "raw" / "025_insulator_mesh_remediation" / "cvt3d_v5_solve.msh"
SEC_COLORS = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#a855f7"]  # theta-sectors
SECTOR_BOUNDS = [0, 30, 60, 90, 180, 360]


def _insulator_tris(msh):
    """Single insulator_surface group, triangles binned into theta-sectors (post-hoc)."""
    m = meshio.read(str(msh))
    pts = np.asarray(m.points, float)
    name3 = {(int(d), int(t)): nm for nm, (t, d) in m.field_data.items()}
    tris = []
    for i, cb in enumerate(m.cells):
        if not cb.type.startswith("triangle"):
            continue
        ph = np.asarray(m.cell_data["gmsh:physical"][i], int).reshape(-1)
        data = np.asarray(cb.data, int)[:, :3]
        for tg in np.unique(ph):
            if name3.get((2, int(tg)), "") == "insulator_surface":
                tris.append(data[ph == tg])
    tris = np.vstack(tris)
    cen = pts[tris].mean(1); th = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])) % 360.0
    out = {}
    for i in range(len(SECTOR_BOUNDS) - 1):
        m_ = (th >= SECTOR_BOUNDS[i]) & (th < SECTOR_BOUNDS[i + 1])
        out[i] = tris[m_]
    return pts, out


def _poly(pts, tris):
    used = np.unique(tris)
    remap = -np.ones(len(pts), int); remap[used] = np.arange(used.size)
    faces = np.hstack([np.full((len(tris), 1), 3), remap[tris]]).ravel()
    return pv.PolyData(pts[used], faces), used


def cam(focal=(0, 0, 1.0), df=3.0):
    return [(df, -df, focal[2] + 0.4 * df), tuple(focal), (0, 0, 1)]


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    pts, sec = _insulator_tris(MSH)

    # 1. patch placement: insulator surface colored by sector group
    p = pv.Plotter(off_screen=True, window_size=(1000, 1300)); p.set_background("white")
    for i in sorted(sec):
        poly, _ = _poly(pts, sec[i])
        p.add_mesh(poly, color=SEC_COLORS[i % len(SEC_COLORS)], show_scalar_bar=False)
    from matplotlib import cm  # noqa
    p.add_legend([(f"s{i} [{b}]", SEC_COLORS[i]) for i, b in
                  zip(range(5), ["0-30", "30-60", "60-90", "90-180", "180-360"])],
                 bcolor="white", size=(0.20, 0.20), loc="upper right")
    p.add_text("Insulator-surface azimuthal sector groups (s0..s4)", position="upper_left",
               font_size=13, color="black")
    p.camera_position = cam(); p.screenshot(str(PROC / "sector_patch_placement_3d.png")); p.close()

    # 2. surface-loss renders for representative angles (polluted sector highlighted on Im(phi))
    for angle in (30, 90, 180, 360):
        vtu = RAW / "run3d" / f"vtu_a{angle}.vtu"
        if not vtu.is_file():
            continue
        m2 = pv.read(str(vtu))
        gid = np.asarray(m2.cell_data["GeometryIds"], int).reshape(-1)
        # device surface = exclude air (highest-id volume); show Im(phi)
        dev = m2.threshold([0.5, 1e9], scalars="GeometryIds")  # all named bodies; air id is large boundary too
        # robust: extract by removing the biggest-volume body (air). Use point Im(phi).
        surf = m2.extract_surface()
        p = pv.Plotter(off_screen=True, window_size=(900, 1250)); p.set_background("white")
        p.add_mesh(surf, scalars="potential im", cmap="coolwarm",
                   scalar_bar_args={"title": "Im(phi) [V]", "vertical": True})
        p.add_text(f"{angle} deg polluted sector - Im(phi) (loss-current signature)",
                   position="upper_left", font_size=12, color="black")
        p.camera_position = cam()
        p.screenshot(str(PROC / f"sector_surfaceloss_{angle}deg.png")); p.close()
    print("Wrote sector placement + surface-loss figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

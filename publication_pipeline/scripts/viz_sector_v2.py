"""Task 026B - improved sector figures: full-insulator sigma_s map (polluted highlighted)
and surface-loss density q_s = 0.5*sigma_s*|grad_s phi|^2 with a COMMON colour scale across
angles. Replaces the weak zoomed Im(phi) close-ups.

Uses the saved solved VTUs (run3d/vtu_a{30,90,180,360}.vtu, sigma_s = 1e-6). The insulator
creepage triangles are extracted by r,z filter; sigma_s/q_s are mapped per triangle.
"""

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
ANGLES = [30, 90, 180, 360]
SIGMA = 1e-6
INS = (0.11, 0.17, 0.557, 1.605)   # r_lo,r_hi,z_lo,z_hi creepage filter


def insulator_data(vtu):
    """Return (PolyData of insulator creepage, theta[deg] per tri, q_s per tri) for sigma_s."""
    m = meshio.read(str(vtu))
    pts = np.asarray(m.points, float)
    pd = {k.lower(): np.asarray(v, float).reshape(-1) for k, v in m.point_data.items()}
    re, im = pd["potential re"], pd["potential im"]
    tris = next((np.asarray(cb.data, int)[:, :3] for cb in m.cells if cb.type.startswith("triangle")), None)
    p = pts[tris]; cen = p.mean(1)
    rc = np.hypot(cen[:, 0], cen[:, 1]); zc = cen[:, 2]
    on = (rc >= INS[0]) & (rc <= INS[1]) & (zc >= INS[2]) & (zc <= INS[3])
    tris = tris[on]; p = p[on]
    th = np.degrees(np.arctan2(p.mean(1)[:, 1], p.mean(1)[:, 0])) % 360.0
    e1 = p[:, 1]-p[:, 0]; e2 = p[:, 2]-p[:, 0]
    nrm = np.cross(e1, e2); nhat = nrm/np.linalg.norm(nrm, axis=1)[:, None]
    M = np.stack([e1, e2, nhat], 1); gsq = np.zeros(len(tris))
    for f in (re, im):
        b = np.stack([f[tris[:, 1]]-f[tris[:, 0]], f[tris[:, 2]]-f[tris[:, 0]], np.zeros(len(tris))], 1)[:, :, None]
        gsq += (np.linalg.solve(M, b)[:, :, 0]**2).sum(1)
    used = np.unique(tris); remap = -np.ones(pts.shape[0], int); remap[used] = np.arange(used.size)
    faces = np.hstack([np.full((len(tris), 1), 3), remap[tris]]).ravel()
    poly = pv.PolyData(pts[used], faces)
    return poly, th, gsq   # gsq = |grad_s phi|^2 per triangle


CAM = [(2.6, -2.6, 1.9), (0.0, 0.0, 1.05), (0.0, 0.0, 1.0)]


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    data = {}
    for a in ANGLES:
        vtu = RAW / "run3d" / f"vtu_a{a}.vtu"
        if vtu.is_file():
            data[a] = insulator_data(vtu)
    if not data:
        print("no saved VTUs found"); return 1

    def polluted_mask(a, th):
        return np.ones(len(th), bool) if a >= 360 else (th <= a + 1e-6)

    def qs_of(a, th, gsq):
        return 0.5 * SIGMA * gsq * polluted_mask(a, th)   # 0 outside the polluted sector

    qmax = max(float(qs_of(a, th, gsq).max()) for a, (poly, th, gsq) in data.items())

    # --- sigma_s map (polluted sector highlighted) ---
    p = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1400, 1600)); p.set_background("white")
    for k, a in enumerate(ANGLES):
        if a not in data:
            continue
        poly, th, gsq = data[a]
        sig = np.where(polluted_mask(a, th), SIGMA, 0.0)
        poly.cell_data["sigma_s"] = sig
        p.subplot(k // 2, k % 2)
        p.add_mesh(poly.copy(), scalars="sigma_s", cmap=["#d1d5db", "#dc2626"], clim=[0, SIGMA],
                   n_colors=2, show_scalar_bar=False, interpolate_before_map=False, lighting=False)
        p.add_text(f"{a} deg  (red=polluted, grey=clean)", position="upper_left",
                   font_size=11, color="black")
        p.camera_position = CAM
    p.screenshot(str(PROC / "sector_sigma_s_maps.png")); p.close()

    # --- surface-loss density q_s, COMMON colour scale ---
    p = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1400, 1600)); p.set_background("white")
    for k, a in enumerate(ANGLES):
        if a not in data:
            continue
        poly, th, gsq = data[a]
        poly.cell_data["q_s"] = qs_of(a, th, gsq)
        p.subplot(k // 2, k % 2)
        p.add_mesh(poly.copy(), scalars="q_s", cmap="inferno", clim=[0, qmax],
                   interpolate_before_map=False, lighting=False,
                   show_scalar_bar=(k == 3),
                   scalar_bar_args={"title": "q_s [W/m^2]", "vertical": True} if k == 3 else None)
        p.add_text(f"{a} deg  surface-loss density q_s", position="upper_edge", font_size=11, color="black")
        p.camera_position = CAM
    p.screenshot(str(PROC / "sector_qs_density_panels.png")); p.close()
    print(f"q_s common clim max = {qmax:.3e} W/m^2; wrote sector_sigma_s_maps.png + sector_qs_density_panels.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

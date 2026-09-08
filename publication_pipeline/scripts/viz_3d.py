"""Task 014 - 3D revolved rendering of the axisymmetric DDB-123 solution.

Solves the clean and polluted (surface-conductance) rounded CVT, then revolves the
device cross-section by 270 deg (a cutaway that exposes the interior) with pyvista
and renders publication PNGs of:
  * material regions
  * |E|            (polluted, log)
  * Im(phi)        (polluted - the loss-current / pollution signature)
  * loss density   sigma|E|^2  (volume dielectric dissipation, log)
  * dE = |E|_polluted - |E|_clean

The axisymmetric (r,z) solution is revolved purely for visualization; the air box
around the device is excluded so the cutaway shows the solid CVT.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from cvt_observables import EPS0, _grad_area, _read  # noqa: E402
from run_cvt import OMEGA, U0, build_material_table, ensure_solver, material_of_body, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

pv.OFF_SCREEN = True
PROC = ROOT / "results" / "processed" / "014_3d_render"
RAW = ROOT / "results" / "raw" / "014_3d_render"
SIGMA_S = 1.0e-6  # surface pollution for the polluted renders (transition region)
REV_ANGLE = 270.0
REV_RES = 96

MATERIALS = ["air", "oil", "porcelain", "element", "metal"]
MAT_COLORS = ["#e5e7eb", "#fde68a", "#cdab7e", "#60a5fa", "#6b7280"]


def solve(sigma_s: float, tag: str, mesh_src: Path, dll: Path, ids: dict, mat, env) -> Path:
    pdir = RAW / tag
    pdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(mesh_src, pdir / "mesh", dirs_exist_ok=True)
    shutil.copy2(dll, pdir / dll.name)
    mat_index = {m: i + 1 for i, m in enumerate(mat)}
    body_material = {n: material_of_body(n) for n in ids["bodies"]}
    bodies = [(bid, n) for n, bid in ids["bodies"].items()]
    sif = render_sif(mat, mat_index, body_material, bodies, ids["boundaries"]["hv_electrode"],
                     ids["boundaries"]["ground_electrode"], ids["boundaries"].get("farfield"),
                     ins_id=ids["boundaries"].get("insulator_surface"), sigma_s=sigma_s)
    (pdir / "case.sif").write_text(sif, encoding="utf-8")
    done = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                          capture_output=True, text=True, env=env)
    (pdir / "elmersolver.stdout.log").write_text(done.stdout, encoding="utf-8")
    if done.returncode != 0:
        print(done.stdout[-1500:]); raise RuntimeError(f"solve failed ({tag})")
    return sorted(pdir.glob("**/case*.vtu"))[-1]


def _emag(pts, tris, re, im):
    gr, _ = _grad_area(pts, tris, re)
    gi, _ = _grad_area(pts, tris, im)
    return np.sqrt(np.sum(gr**2, axis=1) + np.sum(gi**2, axis=1))


def _cell_to_point(tris, cellv, n_pts):
    acc = np.zeros(n_pts); cnt = np.zeros(n_pts)
    for k in range(3):
        np.add.at(acc, tris[:, k], cellv)
        np.add.at(cnt, tris[:, k], 1.0)
    cnt[cnt == 0] = 1.0
    return acc / cnt


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    params = CVTParams()

    # build rounded clean mesh (no volume pollution body) + convert
    msh = RAW / "cvt.msh"
    build_cvt_ddb123(msh, params, rounded=True)
    elmer_parent = RAW / "elmer"; elmer_parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(resolve_elmer_grid()), "14", "2", str(msh), "-out", "mesh"],
                   cwd=str(elmer_parent), capture_output=True, text=True, check=True)
    elmer_mesh = elmer_parent / "mesh"
    ids = _parse_mesh_names(elmer_mesh / "mesh.names")
    mat = build_material_table(params.element_epsr_eff())

    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    print("Solving clean + polluted ...")
    vtu_c = solve(0.0, "clean", elmer_mesh, dll, ids, mat, env)
    vtu_p = solve(SIGMA_S, "polluted", elmer_mesh, dll, ids, mat, env)
    pts, tris, gids, re_c, im_c = _read(vtu_c)
    _, _, _, re_p, im_p = _read(vtu_p)

    # per-cell fields
    e_c = _emag(pts, tris, re_c, im_c)
    e_p = _emag(pts, tris, re_p, im_p)
    e2_p = e_p**2
    body_sigma = {bid: sigma_eff(*mat[material_of_body(n)]) for n, bid in ids["bodies"].items()}
    sig = np.array([body_sigma[g] for g in gids])
    loss = 0.5 * sig * e2_p  # volume loss density [W/m^3]
    dE = e_p - e_c
    name_of = {bid: n for n, bid in ids["bodies"].items()}
    matidx_cell = np.array([MATERIALS.index(material_of_body(name_of[g])) if material_of_body(name_of[g]) in MATERIALS else 0 for g in gids])

    # device cross-section (exclude air): keep cells whose body != air
    air_id = ids["bodies"]["air"]
    dev = gids != air_id
    dtris = tris[dev]
    used = np.unique(dtris)
    remap = -np.ones(len(pts), int); remap[used] = np.arange(len(used))
    dpts = pts[used]
    dtris2 = remap[dtris]

    # point fields on device nodes
    npd = len(used)
    full_re_c, full_im_p = re_c, im_p
    pf = {
        "Im_phi": full_im_p[used],
        "logE": np.log10(np.clip(_cell_to_point(dtris2, e_p[dev], npd), 1.0, None)),
        "logLoss": np.log10(np.clip(_cell_to_point(dtris2, loss[dev], npd), 1e-6, None)),
        "dE": _cell_to_point(dtris2, dE[dev], npd),
        "material": _cell_to_point(dtris2, matidx_cell[dev].astype(float), npd),
    }

    # build 2D PolyData in the x-z plane (x=r, y=0, z=z), revolve about z
    P3 = np.column_stack([dpts[:, 0], np.zeros(npd), dpts[:, 1]])
    faces = np.hstack([np.full((len(dtris2), 1), 3), dtris2]).ravel()
    poly = pv.PolyData(P3, faces)
    for k, v in pf.items():
        poly.point_data[k] = v
    solid = poly.extrude_rotate(resolution=REV_RES, angle=REV_ANGLE, capping=True)

    H = params.total_height
    cam = [(2.6, -2.2, 1.9), (0.0, 0.0, 0.55 * H), (0.0, 0.0, 1.0)]

    def render(scalar, fname, title, cmap, clim=None, mat_mode=False):
        p = pv.Plotter(off_screen=True, window_size=(900, 1300))
        p.set_background("white")
        if mat_mode:
            p.add_mesh(solid, scalars=scalar, cmap=MAT_COLORS, clim=[-0.5, len(MATERIALS) - 0.5],
                       n_colors=len(MATERIALS), show_scalar_bar=False, smooth_shading=False)
            p.add_legend([(MATERIALS[i], MAT_COLORS[i]) for i in range(len(MATERIALS))],
                         bcolor="white", size=(0.18, 0.20), loc="upper right")
        else:
            p.add_mesh(solid, scalars=scalar, cmap=cmap, clim=clim, smooth_shading=True,
                       scalar_bar_args={"title": title, "title_font_size": 22, "label_font_size": 16,
                                        "n_labels": 5, "vertical": True, "position_x": 0.86, "position_y": 0.25})
        p.add_text(title, position="upper_left", font_size=14, color="black")
        p.camera_position = cam
        p.screenshot(str(PROC / fname))
        p.close()

    print("Rendering ...")
    render("material", "3d_material.png", "material regions", None, mat_mode=True)
    render("logE", "3d_Emag.png", "log10 |E| [V/m]  (polluted)", "inferno")
    render("Im_phi", "3d_Imphi.png", "Im(phi) [V]  (polluted)", "coolwarm",
           clim=[-float(np.percentile(np.abs(pf["Im_phi"]), 99)), float(np.percentile(np.abs(pf["Im_phi"]), 99))])
    render("logLoss", "3d_loss.png", "log10 loss density sigma|E|^2 [W/m^3]", "viridis")
    dlim = float(np.percentile(np.abs(pf["dE"]), 99))
    render("dE", "3d_dE.png", "dE = |E|_pol - |E|_clean [V/m]", "coolwarm", clim=[-dlim, dlim])

    (PROC / "render_3d_report.md").write_text("\n".join([
        "# Task 014 - 3D Revolved Renders (DDB-123)",
        "",
        f"Axisymmetric solution revolved {REV_ANGLE:.0f} deg (cutaway) with pyvista; device",
        "cross-section only (surrounding air excluded). Clean vs polluted use the",
        f"surface-conductance model with sigma_s = {SIGMA_S:.0e} S on insulator_surface.",
        "",
        "## Figures",
        "- `3d_material.png` - material regions (oil, capacitor elements, metal electrodes,",
        "  porcelain wall + sheds); cut faces expose the C1/C2 stack.",
        "- `3d_Emag.png` - log|E| (polluted): field stress, peaking at the electrode edges.",
        "- `3d_Imphi.png` - Im(phi) (polluted): the quadrature potential driven by the",
        "  surface-pollution leakage - concentrated along the porcelain creepage.",
        "- `3d_loss.png` - volume dielectric loss density sigma|E|^2 (log).",
        "- `3d_dE.png` - |E| change polluted-minus-clean: where the pollution redistributes",
        "  the field.",
        "",
        "These are visualization renders of the (verified) axisymmetric solution; all",
        "quantitative results remain the 2-D/observable values from the earlier tasks.",
        "",
    ]), encoding="utf-8")

    print(f"Wrote 5 renders + report in {PROC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

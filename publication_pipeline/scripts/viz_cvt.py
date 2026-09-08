"""Task 012 - publication visuals for the representative DDB-123 CVT.

Builds the pollution-layer mesh, solves clean (sigma=0) and polluted (sigma=1e-5),
and renders:
  * material-region map (full mirrored cross-section)
  * zoom on the porcelain sheds + pollution layer (half-section, mesh edges)
  * Re(phi) potential map, clean and polluted
  * Im(phi) map, polluted
  * |E| map, clean and polluted (log scale)
  * hotspot table (max |E| location + material), CSV + report

Axisymmetric (r,z): the field plots are mirrored about r=0 only for readability;
the solve is the r>=0 half-section. No new physics.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from cvt_observables import _grad_area, _read  # noqa: E402
from run_cvt import ensure_solver  # noqa: E402
from run_cvt_pollution_sweep import material_of_body, run_one  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

PROC = ROOT / "results" / "processed" / "012_cvt_visuals"
RAW = ROOT / "results" / "raw" / "012_cvt_visuals"
SIGMA_POL = 1.0e-5

MATERIALS = ["air", "oil", "porcelain", "element", "metal", "pollution"]
MAT_COLORS = ["#e5e7eb", "#fde68a", "#cdab7e", "#60a5fa", "#6b7280", "#dc2626"]
MAT_LABELS = {
    "air": "air", "oil": "oil", "porcelain": "porcelain", "element": "capacitor element",
    "metal": "metal (electrode)", "pollution": "pollution layer",
}


# --------------------------------------------------------------------------- #
def solve_case(sigma, run_root, elmer_mesh, dll, ids, eps_r_eff, env) -> Path:
    run_one(sigma, run_root, elmer_mesh, dll, ids, eps_r_eff, env)
    tag = "clean" if sigma == 0.0 else f"sigma_{sigma:.3e}"
    vtus = sorted((run_root / tag).glob("**/case*.vtu"))
    if not vtus:
        raise RuntimeError(f"no VTU for sigma={sigma}")
    return vtus[-1]


def _mirror_triang(pts, tris):
    P = pts[:, :2]
    P2 = np.vstack([P, P * np.array([-1.0, 1.0])])
    T2 = np.vstack([tris, tris + len(P)])
    return mtri.Triangulation(P2[:, 0], P2[:, 1], T2)


def _emag(pts, tris, re, im):
    gr, _ = _grad_area(pts, tris, re)
    gi, _ = _grad_area(pts, tris, im)
    return np.sqrt(np.sum(gr**2, axis=1) + np.sum(gi**2, axis=1))


def _device_xy(ax, params):
    ax.set_xlim(-0.30, 0.30)
    ax.set_ylim(-0.03, params.total_height + 0.03)
    ax.set_aspect("equal")
    ax.set_xlabel("r [m]  (mirrored)")
    ax.set_ylabel("z [m]")


# --------------------------------------------------------------------------- #
def plot_materials(pts, tris, gids, ids, params) -> None:
    matidx = {bid: MATERIALS.index(material_of_body(name)) for name, bid in ids["bodies"].items()}
    cellv = np.array([matidx[g] for g in gids])
    tri = _mirror_triang(pts, tris)
    cmap = ListedColormap(MAT_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(MATERIALS) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(6.5, 9))
    ax.tripcolor(tri, facecolors=np.concatenate([cellv, cellv]), cmap=cmap, norm=norm, shading="flat")
    _device_xy(ax, params)
    ax.set_title("DDB-123 material regions (axisymmetric, mirrored)")
    present = sorted(set(cellv))
    ax.legend(handles=[Patch(facecolor=MAT_COLORS[i], edgecolor="k", label=MAT_LABELS[MATERIALS[i]])
                       for i in present], loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(PROC / "material_regions.png", dpi=200)
    plt.close(fig)


def plot_zoom(pts, tris, gids, ids, params) -> None:
    # half-section zoom on a few sheds + the pollution skin
    matidx = {bid: MATERIALS.index(material_of_body(name)) for name, bid in ids["bodies"].items()}
    cellv = np.array([matidx[g] for g in gids])
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    cmap = ListedColormap(MAT_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(MATERIALS) + 0.5), cmap.N)

    r_out = params.porcelain_inner_radius + params.porcelain_thickness
    centers = params.shed_centers()
    zc = centers[len(centers) // 2]
    pitch = (params.stack_z_hi - params.tank_height) / params.n_sheds
    z0, z1 = zc - 2.2 * pitch, zc + 2.2 * pitch

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.tripcolor(tri, facecolors=cellv, cmap=cmap, norm=norm, shading="flat")
    ax.triplot(tri, color="k", lw=0.18, alpha=0.5)
    ax.axvspan(r_out - params.pollution_thickness, r_out, color="none")
    ax.set_xlim(params.porcelain_inner_radius - 0.005, r_out + params.shed_overhang + 0.01)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Zoom: porcelain wall + weather sheds + pollution skin (red)\n"
                 f"pollution layer at r in [{r_out - params.pollution_thickness:.3f}, {r_out:.3f}] m")
    ax.legend(handles=[Patch(facecolor=MAT_COLORS[MATERIALS.index(m)], edgecolor="k", label=MAT_LABELS[m])
                       for m in ("porcelain", "pollution", "oil", "air")], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PROC / "zoom_sheds_pollution.png", dpi=200)
    plt.close(fig)


def plot_re_pair(clean, polluted, params, u0) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (pts, tris, gids, re, im), title in zip(
        axes, [clean, polluted], ["clean (sigma=0)", f"polluted (sigma={SIGMA_POL:.0e})"]):
        tri = _mirror_triang(pts, tris)
        v = np.concatenate([re, re]) / 1e3
        tpc = ax.tripcolor(tri, v, cmap="viridis", shading="gouraud", vmin=0, vmax=u0 / 1e3)
        ax.tricontour(tri, v, levels=np.linspace(0, u0 / 1e3, 12), colors="w", linewidths=0.4, alpha=0.6)
        _device_xy(ax, params)
        ax.set_title(f"Re(phi)  {title}")
        fig.colorbar(tpc, ax=ax, label="Re(phi) [kV]", shrink=0.8)
    fig.suptitle("DDB-123 in-phase potential Re(phi) with equipotentials", fontsize=13)
    fig.savefig(PROC / "potential_Re_clean_vs_polluted.png", dpi=200)
    plt.close(fig)


def plot_im_polluted(polluted, params) -> None:
    pts, tris, gids, re, im = polluted
    tri = _mirror_triang(pts, tris)
    v = np.concatenate([im, im])
    lim = float(np.percentile(np.abs(im), 99.8)) or 1.0
    fig, ax = plt.subplots(figsize=(6.5, 9))
    tpc = ax.tripcolor(tri, v, cmap="coolwarm", shading="gouraud", vmin=-lim, vmax=lim)
    _device_xy(ax, params)
    ax.set_title(f"Im(phi)  polluted (sigma={SIGMA_POL:.0e})\n(quadrature potential = loss-current signature)")
    fig.colorbar(tpc, ax=ax, label="Im(phi) [V]", shrink=0.8)
    fig.tight_layout()
    fig.savefig(PROC / "potential_Im_polluted.png", dpi=200)
    plt.close(fig)


def plot_e_pair(clean, polluted, params) -> None:
    emags = []
    for (pts, tris, gids, re, im) in (clean, polluted):
        emags.append(_emag(pts, tris, re, im))
    vmax = max(float(np.percentile(e, 99.9)) for e in emags)
    vmin = vmax / 1e4
    fig, axes = plt.subplots(1, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (pts, tris, gids, re, im), e, title in zip(
        axes, [clean, polluted], emags, ["clean (sigma=0)", f"polluted (sigma={SIGMA_POL:.0e})"]):
        tri = _mirror_triang(pts, tris)
        c = np.clip(np.concatenate([e, e]), vmin, None)
        tpc = ax.tripcolor(tri, facecolors=c, cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax), shading="flat")
        _device_xy(ax, params)
        ax.set_title(f"|E|  {title}")
        fig.colorbar(tpc, ax=ax, label="|E| [V/m]", shrink=0.8)
    fig.suptitle("DDB-123 electric-field magnitude |E| (log scale)", fontsize=13)
    fig.savefig(PROC / "Efield_clean_vs_polluted.png", dpi=200)
    plt.close(fig)


def hotspots(clean, polluted, ids) -> list[dict]:
    name_of = {bid: name for name, bid in ids["bodies"].items()}
    rows = []
    for label, (pts, tris, gids, re, im) in [("clean", clean), ("polluted", polluted)]:
        e = _emag(pts, tris, re, im)
        i = int(np.argmax(e))
        cen = pts[tris[i]].mean(axis=0)
        body = name_of[gids[i]]
        rows.append({"case": label, "max_E_Vpm": float(e[i]), "r_m": float(cen[0]),
                     "z_m": float(cen[1]), "body": body, "material": material_of_body(body)})
    return rows


# --------------------------------------------------------------------------- #
def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    params = CVTParams()
    eps_r_eff = params.element_epsr_eff()

    import datetime as dt
    run_root = RAW / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    print("Building mesh + converting ...")
    msh = run_root / "mesh_gmsh" / "cvt.msh"
    build_cvt_ddb123(msh, params, with_pollution_layer=True)
    elmer_parent = run_root / "mesh_elmer"
    elmer_parent.mkdir(parents=True, exist_ok=True)
    eg = resolve_elmer_grid()
    subprocess.run([str(eg), "14", "2", str(msh), "-out", "mesh"], cwd=str(elmer_parent),
                   capture_output=True, text=True, check=True)
    elmer_mesh = elmer_parent / "mesh"
    ids = _parse_mesh_names(elmer_mesh / "mesh.names")

    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    from run_cvt import U0
    print("Solving clean + polluted ...")
    vtu_clean = solve_case(0.0, run_root, elmer_mesh, dll, ids, eps_r_eff, env)
    vtu_pol = solve_case(SIGMA_POL, run_root, elmer_mesh, dll, ids, eps_r_eff, env)
    clean = _read(vtu_clean)
    polluted = _read(vtu_pol)

    print("Rendering figures ...")
    plot_materials(*clean[:3], ids, params)
    plot_zoom(*clean[:3], ids, params)
    plot_re_pair(clean, polluted, params, U0)
    plot_im_polluted(polluted, params)
    plot_e_pair(clean, polluted, params)
    hot = hotspots(clean, polluted, ids)

    with (PROC / "hotspots.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "max_E_Vpm", "r_m", "z_m", "body", "material"])
        w.writeheader()
        for r in hot:
            w.writerow(r)

    report = [
        "# Task 012 - DDB-123 Visual Outputs",
        "",
        "Publication figures for the representative axisymmetric DDB-123 CVT, clean",
        "(sigma=0) and polluted (sigma=1e-5 S/m, the Task 010 critical point). The solve",
        "is the r>=0 half-section; full-device plots are mirrored about r=0 for clarity.",
        "",
        "## Figures",
        "- **material_regions.png** - the meshed cross-section coloured by material:",
        "  air, oil, capacitor elements (homogenized stack), metal electrodes/discs,",
        "  porcelain wall + weather sheds, and the thin red pollution skin on the wall.",
        "- **zoom_sheds_pollution.png** - close-up of the porcelain wall, a few weather",
        "  sheds and the 3 mm conductive pollution skin (red), with mesh edges shown so",
        "  the skin resolution is visible.",
        "- **potential_Re_clean_vs_polluted.png** - in-phase potential Re(phi) with white",
        "  equipotential lines. The capacitor stack carries the HV->ground gradient; the",
        "  two cases look near-identical (external pollution barely moves the divider).",
        "- **potential_Im_polluted.png** - quadrature potential Im(phi): essentially zero",
        "  in the clean case, here it is the loss-current signature driven by the",
        "  pollution conduction (the source of the tap phase displacement).",
        "- **Efield_clean_vs_polluted.png** - |E| on a log scale; peaks at the sharp metal",
        "  features (terminal/disc edges). See the hotspot table for the exact location.",
        "",
        "## Hotspot table (max |E|)",
        "",
        "| case | max \\|E\\| [V/m] | r [m] | z [m] | body | material |",
        "|------|--------------|-------|-------|------|----------|",
    ]
    for r in hot:
        report.append(f"| {r['case']} | {r['max_E_Vpm']:.3e} | {r['r_m']:.4f} | {r['z_m']:.4f} "
                      f"| {r['body']} | {r['material']} |")
    report += [
        "",
        "The peak field sits in the dielectric adjacent to a sharp metal electrode edge",
        "(the partial-discharge-risk location); pollution on the external wall does not",
        "move it, consistent with the weak external->internal coupling seen in Task 010.",
        "",
        "All PNGs are 200 dpi. Source: viz_cvt.py.",
        "",
    ]
    (PROC / "visuals_report.md").write_text("\n".join(report), encoding="utf-8")

    print("Hotspots:")
    for r in hot:
        print(f"  {r['case']}: |E|max={r['max_E_Vpm']:.3e} V/m at (r={r['r_m']:.4f}, z={r['z_m']:.4f}) "
              f"in {r['body']} ({r['material']})")
    print(f"Figures + report in: {PROC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

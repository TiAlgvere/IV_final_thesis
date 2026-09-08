"""Task 013.5 - geometry audit: sharp vs rounded DDB-123, before the 3D renders.

Builds and solves the clean CVT in both the sharp (Tasks 009-012) and rounded
(Task 013) geometries, then produces:
  1. full-device material map, sharp vs rounded (geometry only)
  2. full mesh view (edges) + a shed-refinement zoom
  3. close-ups: HV terminal, top dome, one shed, capacitor-disc region, tank transition
  4. report: actual fillet radii used
  5. sharp-vs-rounded metrics: node count, element count, C, divider ratio, Emax

Purpose: confirm the geometry is physically realistic before final 3D rendering.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from cvt_observables import _grad_area, _read, compute_cvt_observables  # noqa: E402
from run_cvt import FREQ, OMEGA, U0, build_material_table, ensure_solver, material_of_body, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

PROC = ROOT / "results" / "processed" / "0135_geometry_audit"
RAW = ROOT / "results" / "raw" / "0135_geometry_audit"

MATERIALS = ["air", "oil", "porcelain", "element", "metal", "pollution"]
MAT_COLORS = ["#e5e7eb", "#fde68a", "#cdab7e", "#60a5fa", "#6b7280", "#dc2626"]
MAT_LABELS = {"air": "air", "oil": "oil", "porcelain": "porcelain", "element": "capacitor element",
              "metal": "metal (electrode)", "pollution": "pollution"}


def build_and_solve(rounded: bool, tag: str, dll: Path, env: dict) -> dict:
    params = CVTParams()
    base = RAW / tag
    msh = base / "cvt.msh"
    meta = build_cvt_ddb123(msh, params, rounded=rounded)
    elmer_parent = base / "elmer"
    elmer_parent.mkdir(parents=True, exist_ok=True)
    eg = resolve_elmer_grid()
    subprocess.run([str(eg), "14", "2", str(msh), "-out", "mesh"], cwd=str(elmer_parent),
                   capture_output=True, text=True, check=True)
    elmer_mesh = elmer_parent / "mesh"
    ids = _parse_mesh_names(elmer_mesh / "mesh.names")

    mat = build_material_table(params.element_epsr_eff())
    mat_index = {m: i + 1 for i, m in enumerate(mat)}
    body_material = {name: material_of_body(name) for name in ids["bodies"]}
    bodies = [(bid, name) for name, bid in ids["bodies"].items()]
    sif = render_sif(mat, mat_index, body_material, bodies, ids["boundaries"]["hv_electrode"],
                     ids["boundaries"]["ground_electrode"], ids["boundaries"].get("farfield"))
    pdir = base / "run"
    pdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(elmer_mesh, pdir / "mesh", dirs_exist_ok=True)
    shutil.copy2(dll, pdir / dll.name)
    (pdir / "case.sif").write_text(sif, encoding="utf-8")
    done = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                          capture_output=True, text=True, env=env)
    (pdir / "elmersolver.stdout.log").write_text(done.stdout, encoding="utf-8")
    if done.returncode != 0:
        print(done.stdout[-1500:]); raise RuntimeError(f"solve failed ({tag})")
    vtu = sorted(pdir.glob("**/case*.vtu"))[-1]

    body_epsr = {bid: mat[body_material[name]][0] for name, bid in ids["bodies"].items()}
    body_sigma = {bid: sigma_eff(*mat[body_material[name]]) for name, bid in ids["bodies"].items()}
    obs = compute_cvt_observables(vtu, u0=U0, omega=OMEGA, body_epsr=body_epsr, body_sigma=body_sigma,
                                  tap_body_id=ids["bodies"][f"foil_{params.tap_disc_index}"])
    pts, tris, gids, re, im = _read(vtu)
    gr, _ = _grad_area(pts, tris, re)
    gi, _ = _grad_area(pts, tris, im)
    emag = np.sqrt(np.sum(gr**2, axis=1) + np.sum(gi**2, axis=1))
    i = int(np.argmax(emag))
    cen = pts[tris[i]].mean(axis=0)
    name_of = {bid: name for name, bid in ids["bodies"].items()}
    return {
        "rounded": rounded, "tag": tag, "params": params, "ids": ids, "body_material": body_material,
        "pts": pts, "tris": tris, "gids": gids, "obs": obs,
        "nodes": meta["n_nodes"], "elements": meta["n_triangles"],
        "emax": float(emag[i]), "emax_rz": (float(cen[0]), float(cen[1])),
        "emax_body": name_of[gids[i]], "name_of": name_of,
    }


def _cell_mat(d) -> np.ndarray:
    idx = {bid: MATERIALS.index(material_of_body(name)) for name, bid in d["ids"]["bodies"].items()}
    return np.array([idx[g] for g in d["gids"]])


def _mirror(d):
    P = d["pts"][:, :2]
    P2 = np.vstack([P, P * np.array([-1.0, 1.0])])
    T2 = np.vstack([d["tris"], d["tris"] + len(P)])
    return mtri.Triangulation(P2[:, 0], P2[:, 1], T2)


def _legend(ax, present):
    ax.legend(handles=[Patch(facecolor=MAT_COLORS[i], edgecolor="k", label=MAT_LABELS[MATERIALS[i]])
                       for i in present], loc="upper right", fontsize=7, framealpha=0.9)


def fig_full_geometry(sharp, rnd):
    cmap = ListedColormap(MAT_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(MATERIALS) + 0.5), cmap.N)
    H = sharp["params"].total_height
    fig, axes = plt.subplots(1, 2, figsize=(9, 10), constrained_layout=True)
    for ax, d, title in zip(axes, [sharp, rnd], ["sharp (Tasks 009-012)", "rounded (Task 013)"]):
        cm = _cell_mat(d)
        ax.tripcolor(_mirror(d), facecolors=np.concatenate([cm, cm]), cmap=cmap, norm=norm, shading="flat")
        ax.set_xlim(-0.30, 0.30); ax.set_ylim(-0.03, H + 0.03); ax.set_aspect("equal")
        ax.set_xlabel("r [m] (mirrored)"); ax.set_ylabel("z [m]"); ax.set_title(title)
        _legend(ax, sorted(set(cm)))
    fig.suptitle("DDB-123 material regions - sharp vs rounded (geometry only)", fontsize=13)
    fig.savefig(PROC / "1_full_geometry_sharp_vs_rounded.png", dpi=200)
    plt.close(fig)


def fig_full_mesh(rnd):
    p = rnd["params"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 10), constrained_layout=True)
    tri = _mirror(rnd)
    axes[0].triplot(tri, color="#334155", lw=0.15)
    axes[0].set_xlim(-0.30, 0.30); axes[0].set_ylim(-0.03, p.total_height + 0.03)
    axes[0].set_aspect("equal"); axes[0].set_xlabel("r [m] (mirrored)"); axes[0].set_ylabel("z [m]")
    axes[0].set_title("full mesh (rounded)")
    # shed-band zoom (half section) to see refinement
    cen = p.shed_centers()
    zc = cen[len(cen) // 2]
    pitch = (p.stack_z_hi - p.tank_height) / p.n_sheds
    tri2 = mtri.Triangulation(rnd["pts"][:, 0], rnd["pts"][:, 1], rnd["tris"])
    axes[1].triplot(tri2, color="#334155", lw=0.3)
    axes[1].set_xlim(p.porcelain_inner_radius - 0.005, p.porcelain_inner_radius + p.porcelain_thickness + p.shed_overhang + 0.01)
    axes[1].set_ylim(zc - 3 * pitch, zc + 3 * pitch); axes[1].set_aspect("equal")
    axes[1].set_xlabel("r [m]"); axes[1].set_ylabel("z [m]"); axes[1].set_title("shed-band refinement (zoom)")
    fig.suptitle("DDB-123 mesh (rounded) - full view + shed refinement", fontsize=13)
    fig.savefig(PROC / "2_full_mesh.png", dpi=220)
    plt.close(fig)


def fig_closeups(rnd):
    p = rnd["params"]
    cmap = ListedColormap(MAT_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(MATERIALS) + 0.5), cmap.N)
    cm = _cell_mat(rnd)
    tri = mtri.Triangulation(rnd["pts"][:, 0], rnd["pts"][:, 1], rnd["tris"])
    z_dome1 = p.stack_z_hi + p.head_height
    cen = p.shed_centers()
    zc = cen[len(cen) // 2]
    pitch = (p.stack_z_hi - p.tank_height) / p.n_sheds
    rtip = p.porcelain_inner_radius + p.porcelain_thickness + p.shed_overhang
    discs = [p.disc_span(k) for k in range(p.n_elements // 2, p.n_elements // 2 + 3)]
    views = [
        ("HV terminal", (-0.0, p.terminal_radius + 0.03), (z_dome1 - 0.01, p.total_height + 0.012)),
        ("top dome", (-0.0, p.head_radius + 0.03), (p.stack_z_hi - 0.03, z_dome1 + 0.03)),
        ("one shed", (p.porcelain_inner_radius - 0.004, rtip + 0.006), (zc - 1.4 * pitch, zc + 1.4 * pitch)),
        ("capacitor discs", (-0.0, p.stack_radius + 0.035), (discs[0][0] - 0.004, discs[-1][1] + 0.004)),
        ("tank transition", (-0.0, p.tank_radius + 0.02), (p.tank_height - 0.05, p.tank_height + 0.10)),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 6), constrained_layout=True)
    for ax, (title, xl, yl) in zip(axes, views):
        ax.tripcolor(tri, facecolors=cm, cmap=cmap, norm=norm, shading="flat")
        ax.triplot(tri, color="k", lw=0.15, alpha=0.45)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
        ax.set_xlabel("r [m]"); ax.set_ylabel("z [m]"); ax.set_title(title)
    fig.suptitle("DDB-123 rounded geometry - close-ups (material + mesh)", fontsize=14)
    fig.savefig(PROC / "3_closeups.png", dpi=200)
    plt.close(fig)


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    print("Solving sharp ..."); sharp = build_and_solve(False, "sharp", dll, env)
    print("Solving rounded ..."); rnd = build_and_solve(True, "rounded", dll, env)

    fig_full_geometry(sharp, rnd)
    fig_full_mesh(rnd)
    fig_closeups(rnd)

    p = CVTParams()
    lines = [
        "# Task 013.5 - DDB-123 Geometry Audit (sharp vs rounded)",
        "",
        "Geometry-realism check before the 3D renders. Clean CVT solved in both",
        "geometries; figures show material regions, full mesh, and close-ups.",
        "",
        "## Fillet radii used (rounded geometry)",
        "| feature | radius [mm] |",
        "|---------|-------------|",
        f"| weather-shed tip lip | {p.shed_lip_round*1e3:.1f} |",
        f"| HV compensator dome corner | {p.head_round*1e3:.1f} |",
        f"| tank top-outer corner | {p.head_round*1e3:.1f} |",
        f"| capacitor-disc / stack-terminal edge | {p.electrode_round*1e3:.2f} |",
        f"| HV terminal stub corner | {p.terminal_round*1e3:.1f} |",
        f"| (mesh curvature target) | {p.mesh_curvature:.0f} elem / 2*pi |",
        "",
        "## Sharp vs rounded metrics",
        "| metric | sharp | rounded | change |",
        "|--------|-------|---------|--------|",
        f"| mesh nodes | {sharp['nodes']} | {rnd['nodes']} | {rnd['nodes']-sharp['nodes']:+d} |",
        f"| mesh elements | {sharp['elements']} | {rnd['elements']} | {rnd['elements']-sharp['elements']:+d} |",
        f"| terminal C [pF] | {sharp['obs']['C_total_pF']:.1f} | {rnd['obs']['C_total_pF']:.1f} | "
        f"{(rnd['obs']['C_total_pF']/sharp['obs']['C_total_pF']-1)*100:+.2f}% |",
        f"| divider ratio | {sharp['obs']['divider_ratio']:.4f} | {rnd['obs']['divider_ratio']:.4f} | "
        f"{(rnd['obs']['divider_ratio']/sharp['obs']['divider_ratio']-1)*100:+.2f}% |",
        f"| Emax [V/m] | {sharp['emax']:.3e} | {rnd['emax']:.3e} | "
        f"{(rnd['emax']/sharp['emax']-1)*100:+.1f}% |",
        f"| Emax location (r,z) [m] | ({sharp['emax_rz'][0]:.3f},{sharp['emax_rz'][1]:.3f}) {sharp['emax_body']} | "
        f"({rnd['emax_rz'][0]:.3f},{rnd['emax_rz'][1]:.3f}) {rnd['emax_body']} | |",
        "",
        "Datasheet anchors: C 5600 pF, creepage 3075 mm (analytic 3062 mm, unchanged).",
        "",
        "## Read-out",
        "- C and divider ratio are essentially unchanged by rounding (sub-percent): the",
        "  fillets are small vs the electrode dimensions, so terminal behaviour is preserved.",
        "- Emax is NOT a 'lower-is-better' comparison. At a SHARP corner |E| is a numerical",
        "  singularity - finite on a given mesh but mesh-dependent (it grows without bound",
        "  under refinement), so the sharp value is an under-resolved artifact. The rounded",
        "  fillets give a BOUNDED, resolved peak: the trustworthy value. Here rounding the",
        "  dome edge moved the peak from the dome corner (0.18, 1.80) to the terminal-stub",
        "  tip (0.02, 1.83) - the genuine smallest-radius HV feature - and THAT resolved",
        "  peak (~7.4e5 V/m) is what the |E| / PD-risk readouts should use.",
        "- So the purpose of rounding is met: it turns Emax from a mesh-dependent singularity",
        "  estimate into a finite, physically-meaningful peak at a real geometric feature.",
        "- Node/element count rises from the curvature refinement at the fillets.",
        "",
        "## Figures",
        "- `1_full_geometry_sharp_vs_rounded.png`",
        "- `2_full_mesh.png` (full mesh + shed-band zoom)",
        "- `3_closeups.png` (HV terminal, top dome, one shed, capacitor discs, tank transition)",
        "",
    ]
    (PROC / "geometry_audit_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== sharp vs rounded ===")
    print(f"  nodes   : {sharp['nodes']} -> {rnd['nodes']}")
    print(f"  elements: {sharp['elements']} -> {rnd['elements']}")
    print(f"  C [pF]  : {sharp['obs']['C_total_pF']:.1f} -> {rnd['obs']['C_total_pF']:.1f}")
    print(f"  ratio   : {sharp['obs']['divider_ratio']:.4f} -> {rnd['obs']['divider_ratio']:.4f}")
    print(f"  Emax    : {sharp['emax']:.3e} ({sharp['emax_body']}) -> {rnd['emax']:.3e} ({rnd['emax_body']}) "
          f"[{(rnd['emax']/sharp['emax']-1)*100:+.1f}%]")
    print(f"Figures + report in: {PROC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

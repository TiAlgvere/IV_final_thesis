"""Task 025 - mesh-remediation figures: insulator-surface quality before/after,
global tet-quality histogram, worst-100 element locations by material."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
PROC = ROOT / "results" / "processed" / "025_insulator_mesh_remediation"
RAW = ROOT / "results" / "raw" / "025_insulator_mesh_remediation"

# v4 (pre-remediation) insulator-surface stats, from Task 024 audit
V4_INS = {"min": 0.043, "p5": 0.043, "mean": 0.744}


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    meta = json.loads((PROC / "solve_mesh_meta.json").read_text(encoding="utf-8"))
    ins = meta["insulator_surface_tri_quality"]

    # 1. insulator-surface quality before/after
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics = ["min", "p5", "mean"]
    x = np.arange(len(metrics)); w = 0.38
    ax.bar(x - w/2, [V4_INS[m] for m in metrics], w, label="v4 (before)", color="#dc2626")
    ax.bar(x + w/2, [ins[m] for m in metrics], w, label="v5 (remediated)", color="#16a34a")
    for i, m in enumerate(metrics):
        ax.text(i - w/2, V4_INS[m] + 0.01, f"{V4_INS[m]:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, ins[m] + 0.01, f"{ins[m]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylabel("insulator_surface minSICN")
    ax.set_title("Insulator-surface triangle quality: v4 -> v5 remediation"); ax.legend()
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PROC / "ddb123_v5_quality_comparison.png", dpi=150); plt.close(fig)

    # 2. global tet quality histogram (solve mesh)
    qnpz = Path(str(RAW / "cvt3d_v5_solve.msh") + ".quality.npz")
    if qnpz.is_file():
        q = np.load(qnpz)["tet_q"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(q, bins=80, color="#2563eb", alpha=0.8)
        for pc, c in [(1, "#b91c1c"), (5, "#ea580c"), (50, "#16a34a")]:
            ax.axvline(np.percentile(q, pc), color=c, ls="--", lw=1.4, label=f"p{pc}={np.percentile(q,pc):.3f}")
        ax.axvline(0.05, color="k", ls=":", label="sliver 0.05")
        ax.set_yscale("log"); ax.set_xlabel("global tet minSICN"); ax.set_ylabel("count (log)")
        ax.set_title(f"v5 global tet quality (n={len(q)}, min={q.min():.3f}, mean={q.mean():.3f}, "
                     f"slivers<0.05={int((q<0.05).sum())})")
        ax.legend(); fig.tight_layout()
        fig.savefig(PROC / "ddb123_v5_global_quality_hist.png", dpi=150); plt.close(fig)

    # 3. worst-100 locations by material (computed from the solve mesh)
    import meshio
    from publication_pipeline.geometry.gmsh.cvt_ddb123_3d_v4 import MAT_OF_V4
    mm = meshio.read(str(RAW / "cvt3d_v5_solve.msh"))
    pts = np.asarray(mm.points, float)
    name3 = {(int(d), int(t)): nm for nm, (t, d) in mm.field_data.items()}
    tets, phys = [], []
    for i, cb in enumerate(mm.cells):
        if cb.type.startswith("tetra"):
            tets.append(np.asarray(cb.data, int)[:, :4])
            phys.append(np.asarray(mm.cell_data["gmsh:physical"][i], int).reshape(-1))
    tets = np.vstack(tets); phys = np.concatenate(phys)
    mats = np.array([MAT_OF_V4.get(name3.get((3, int(t)), "air"), "air") for t in phys])
    P = pts[tets]; Cc = P.mean(1); rr = np.hypot(Cc[:, 0], Cc[:, 1]); zz = Cc[:, 2]
    p0 = P[:, 0]; M = np.stack([P[:, 1]-p0, P[:, 2]-p0, P[:, 3]-p0], 1)
    Vv = np.abs(np.linalg.det(M)) / 6.0
    e2 = np.zeros(len(tets))
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        e2 += ((P[:, a]-P[:, b])**2).sum(1)
    eta = 12.0 * (3.0*np.clip(Vv, 1e-30, None))**(2/3) / np.clip(e2, 1e-30, None)
    worst = np.argsort(eta)[:100]
    worst_list = [{"rz": [float(rr[i]), float(zz[i])], "mat": str(mats[i])} for i in worst]

    fig, ax = plt.subplots(figsize=(7, 8))
    cmap = {"air": "#9ca3af", "oil": "#eab308", "porcelain": "#b45309", "element": "#2563eb",
            "metal": "#374151", "resin": "#16a34a"}
    for wse in worst_list:
        ax.scatter(wse["rz"][0], wse["rz"][1], c=cmap.get(wse["mat"], "k"), s=55,
                   edgecolors="k", linewidths=0.3)
    ax.set_xlabel("r [m]"); ax.set_ylabel("z [m]"); ax.grid(alpha=0.3)
    ax.set_title("v5 worst-100 tets (location + material)")
    import collections
    bym = collections.Counter(w["mat"] for w in worst_list)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap.get(k, "k"),
               markeredgecolor="k", markersize=9, label=f"{k} ({bym[k]})") for k in bym]
    ax.legend(handles=handles, fontsize=8, title="worst-100 by material")
    fig.tight_layout(); fig.savefig(PROC / "ddb123_v5_worst100_locations.png", dpi=150); plt.close(fig)

    print("by material:", dict(bym))
    print("Wrote 3 remediation figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

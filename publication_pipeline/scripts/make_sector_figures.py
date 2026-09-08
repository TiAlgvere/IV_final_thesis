"""Task 026 - figures for the azimuthal surface-conductance sector validation (MATC method)."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT.parent), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
PROC = ROOT / "results" / "processed" / "026_azimuthal_surface_conductance_validation"


def main() -> int:
    v = json.loads((PROC / "validation.json").read_text(encoding="utf-8"))
    rows = v["cases"]; aud = v["mesh_audit"]
    clean = next(r for r in rows if r["case"] == "clean")

    def sub(ss):
        return sorted([r for r in rows if r.get("sigma_s") == ss and r["angle_deg"] > 0],
                      key=lambda r: r["angle_deg"])

    # 1. patch-area audit
    s6 = sub(1e-6)
    fig, ax = plt.subplots(figsize=(7, 5))
    ang = [r["angle_deg"] for r in s6]; meas = [r["area_fraction"] for r in s6]
    exp = [r["expected_fraction"] for r in s6]
    ax.plot(ang, exp, "k--s", label="expected = angle/360")
    ax.plot(ang, meas, "o-", color="#2563eb", label="measured area fraction")
    for a, m, e in zip(ang, meas, exp):
        ax.annotate(f"{(m/e-1)*100:+.1f}%", (a, m), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("sector angle [deg]"); ax.set_ylabel("polluted area fraction")
    ax.set_title("Patch area audit: measured vs requested sector fraction"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PROC / "sector_patch_area_audit.png", dpi=150); plt.close(fig)

    # 2. surface_leak vs angle
    fig, ax = plt.subplots(figsize=(7, 5))
    for ss, col in [(1e-7, "#16a34a"), (1e-6, "#dc2626")]:
        s = sub(ss)
        ax.plot([r["angle_deg"] for r in s], [r["surface_leak"] for r in s], "o-", color=col, label=f"sigma_s={ss:.0e}")
    ax.set_xlabel("sector angle [deg]"); ax.set_ylabel("surface_leak [A]"); ax.set_yscale("log")
    ax.set_title("Surface leakage vs angular coverage"); ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(PROC / "sector_surface_leak_vs_angle.png", dpi=150); plt.close(fig)

    # 3. phase + tanδ vs angle
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for ss, col in [(1e-7, "#16a34a"), (1e-6, "#dc2626")]:
        s = sub(ss)
        ax[0].plot([r["angle_deg"] for r in s], [r["tan_delta"] for r in s], "o-", color=col, label=f"sigma_s={ss:.0e}")
        ax[1].plot([r["angle_deg"] for r in s], [r["phase_mdeg"] for r in s], "o-", color=col, label=f"sigma_s={ss:.0e}")
    ax[0].axhline(clean["tan_delta"], color="0.5", ls=":", label="clean")
    ax[0].set_xlabel("angle [deg]"); ax[0].set_ylabel("tan-delta"); ax[0].set_yscale("log"); ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    ax[1].axhline(clean["phase_mdeg"], color="0.5", ls=":", label="clean")
    ax[1].set_xlabel("angle [deg]"); ax[1].set_ylabel("tap phase [mdeg]"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle("tan-delta and tap phase vs sector angle"); fig.tight_layout()
    fig.savefig(PROC / "sector_phase_tandelta_vs_angle.png", dpi=150); plt.close(fig)

    # 4. C, |Vtap| vs angle (sigma_s=1e-6)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5)); s = sub(1e-6)
    ax[0].plot([0]+[r["angle_deg"] for r in s], [clean["C_pF"]]+[r["C_pF"] for r in s], "o-", color="#2563eb")
    ax[0].axhline(5600, color="0.5", ls=":", label="5600 pF"); ax[0].set_xlabel("angle [deg]")
    ax[0].set_ylabel("C [pF]"); ax[0].set_title("C vs angle (sigma_s=1e-6)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot([0]+[r["angle_deg"] for r in s], [clean["vtap_abs"]]+[r["vtap_abs"] for r in s], "o-", color="#7c3aed")
    ax[1].set_xlabel("angle [deg]"); ax[1].set_ylabel("|Vtap| [V]"); ax[1].set_title("|Vtap| vs angle"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PROC / "sector_observables_vs_angle.png", dpi=150); plt.close(fig)

    # 5. mesh quality: per-theta-sector + patch-edge + global
    sec = aud["sector_tri"]; fig, ax = plt.subplots(figsize=(8.5, 5))
    names = list(sec.keys())
    mins = [sec[n]["min"] if sec[n]["min"] is not None else 0 for n in names]
    p5s = [sec[n]["p5"] if sec[n]["p5"] is not None else 0 for n in names]
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w/2, mins, w, label="min", color="#dc2626")
    ax.bar(x + w/2, p5s, w, label="p5", color="#16a34a")
    pe = aud["patch_edge_tri"]
    ax.axhline(pe["p5"], color="#7c3aed", ls="--", label=f"patch-edge p5={pe['p5']:.3f} (min={pe['min']:.3f})")
    ax.axhline(aud["insulator_surface"]["p5"], color="0.4", ls=":", label=f"all-insulator p5={aud['insulator_surface']['p5']:.3f}")
    ax.set_xticks(x); ax.set_xticklabels([n.replace("deg", "") for n in names], fontsize=8)
    ax.set_ylabel("insulator triangle minSICN"); ax.set_ylim(0, 1)
    ax.set_title(f"Per-sector insulator quality (global tet min={aud['global_tet']['min']:.3f}, "
                 f"slivers={aud['global_tet']['n_slivers_lt_0.05']})"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PROC / "sector_mesh_quality.png", dpi=150); plt.close(fig)

    # 6. worst-100 locations (r-z and theta-z)
    worst = aud["worst100"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    cmap = {"air": "#9ca3af", "oil": "#eab308", "porcelain": "#b45309", "element": "#2563eb",
            "metal": "#374151", "resin": "#16a34a"}
    for w_ in worst:
        ax[0].scatter(w_["rz"][0], w_["rz"][1], c=cmap.get(w_["mat"], "k"), s=42, edgecolors="k", linewidths=0.3)
        ax[1].scatter(w_["theta_deg"], w_["rz"][1], c=cmap.get(w_["mat"], "k"), s=42, edgecolors="k", linewidths=0.3)
    for a in (30, 60, 90, 180):
        ax[1].axvline(a, color="#b91c1c", ls=":", lw=1)
    ax[0].set_xlabel("r [m]"); ax[0].set_ylabel("z [m]"); ax[0].set_title("worst-100 (r,z)"); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("theta [deg]"); ax[1].set_ylabel("z [m]"); ax[1].set_title("worst-100 (theta,z); red=cut planes"); ax[1].grid(alpha=0.3)
    bym = collections.Counter(w_["mat"] for w_ in worst)
    ax[0].legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap.get(k, "k"),
                 markeredgecolor="k", markersize=9, label=f"{k} ({bym[k]})") for k in bym], fontsize=8)
    fig.tight_layout(); fig.savefig(PROC / "sector_worst100_locations.png", dpi=150); plt.close(fig)

    print("worst100 by material:", dict(bym))
    print("area fractions (1e-6):", [(r["angle_deg"], round(r["area_fraction"], 4)) for r in s6])
    print("Wrote sector figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

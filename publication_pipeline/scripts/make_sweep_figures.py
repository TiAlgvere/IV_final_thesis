"""Task 027 - figures for the fixed-sector surface-conductance intensity sweep.

Reads results/processed/027_fixed_sector_sigma_sweep/sweep.json (+ the saved q_s VTUs in
results/raw/027_fixed_sector_sigma_sweep/qs_fields/). All power/q_s values use the RMS phasor
convention documented in sweep.json["convention"] (no 1/2 factor).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "results" / "processed" / "027_fixed_sector_sigma_sweep"
RAW = ROOT / "results" / "raw" / "027_fixed_sector_sigma_sweep"
QDIR = RAW / "qs_fields"
QS_SELECT = [1e-9, 1e-7, 1e-5]   # weak / transition / severe panels


def _nz(rows):
    """sigma>0 rows sorted by sigma."""
    return sorted([r for r in rows if r["sigma_s"] > 0], key=lambda r: r["sigma_s"])


def _clean(rows):
    return next((r for r in rows if r["sigma_s"] == 0), None)


def fig_surface_leak(s30, s10):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for rows, lab, col in [(s30, "30 deg sector", "#dc2626"), (s10, "10 deg sector", "#2563eb")]:
        if not rows:
            continue
        x = [r["sigma_s"] for r in rows]; y = [r["surface_leak"] for r in rows]
        ax.loglog(x, y, "o-", color=col, label=lab)
    # reference linear slope through the lowest-sigma 30deg point
    if s30:
        x0, y0 = s30[0]["sigma_s"], s30[0]["surface_leak"]
        xs = np.array([s30[0]["sigma_s"], s30[-1]["sigma_s"]])
        ax.loglog(xs, y0 * xs / x0, "k--", lw=1, alpha=0.7, label="linear (slope 1) ref")
    ax.set_xlabel("surface conductance sigma_s [S]"); ax.set_ylabel("surface_leak [A]")
    ax.set_title("Surface leakage current vs sigma_s (fixed 30 deg sector)")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(PROC / "sigma_sweep_surface_leak.png", dpi=150); plt.close(fig)


def fig_surface_power(s30, s10):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for rows, lab, col in [(s30, "30 deg sector", "#dc2626"), (s10, "10 deg sector", "#2563eb")]:
        if not rows:
            continue
        x = [r["sigma_s"] for r in rows]; y = [r["P_surface_direct_W"] for r in rows]
        ax.loglog(x, y, "o-", color=col, label=lab)
    if s30:
        x0, y0 = s30[0]["sigma_s"], s30[0]["P_surface_direct_W"]
        xs = np.array([s30[0]["sigma_s"], s30[-1]["sigma_s"]])
        ax.loglog(xs, y0 * xs / x0, "k--", lw=1, alpha=0.7, label="linear (slope 1) ref")
    ax.set_xlabel("surface conductance sigma_s [S]")
    ax.set_ylabel("P_surface_direct [W]  (RMS, = integral sigma_s |grad_s phi|^2 dS)")
    ax.set_title("Direct surface loss power vs sigma_s (fixed 30 deg sector)")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(PROC / "sigma_sweep_surface_power.png", dpi=150); plt.close(fig)


def fig_tandelta(s30, s10, c30):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for rows, lab, col in [(s30, "30 deg sector", "#dc2626"), (s10, "10 deg sector", "#2563eb")]:
        if not rows:
            continue
        ax.loglog([r["sigma_s"] for r in rows], [r["tan_delta"] for r in rows], "o-", color=col, label=lab)
    if c30:
        ax.axhline(c30["tan_delta"], color="0.5", ls=":", label=f"clean tan-delta={c30['tan_delta']:.2e}")
    ax.set_xlabel("surface conductance sigma_s [S]"); ax.set_ylabel("tan-delta")
    ax.set_title("Loss tangent vs sigma_s (fixed 30 deg sector)")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(PROC / "sigma_sweep_tandelta.png", dpi=150); plt.close(fig)


def fig_phase(s30, c30):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = [r["sigma_s"] for r in s30]; y = [r["phase_mdeg"] for r in s30]
    ax.semilogx(x, y, "o-", color="#6b7280", label="tap phase (30 deg)")
    if c30 is not None:
        ax.axhline(c30["phase_mdeg"], color="0.5", ls=":", label=f"clean phase={c30['phase_mdeg']:.1f} mdeg")
    lo, hi = min(y + [c30["phase_mdeg"]]) if c30 else min(y), max(y + [c30["phase_mdeg"]]) if c30 else max(y)
    ax.axhspan(lo, hi, color="#fca5a5", alpha=0.25, label="iterative-residual / noise band")
    ax.set_xlabel("surface conductance sigma_s [S]"); ax.set_ylabel("tap phase [mdeg]")
    ax.set_title("Tap phase vs sigma_s - UNRELIABLE (near iterative solver floor; not a diagnostic)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PROC / "sigma_sweep_vtap_phase.png", dpi=150); plt.close(fig)


def fig_emax(s30, c30):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    rows = ([c30] if c30 else []) + s30
    x = [max(r["sigma_s"], 1e-11) for r in rows]; em = [r["emax"] for r in rows]
    spread = (max(em)/min(em) - 1)*100
    ax[0].semilogx(x, em, "o-", color="#b45309")
    ax[0].set_ylim(0, 1.5e6)
    ax[0].set_xlabel("sigma_s [S] (clean plotted at 1e-11)"); ax[0].set_ylabel("Emax [V/m]")
    ax[0].set_title(f"Emax magnitude vs sigma_s - constant at ~1.39 MV/m (spread {spread:.2f}%)")
    ax[0].grid(alpha=0.3, which="both")
    # Emax location (r,z) coloured by sigma; insulator-patch box overlaid
    r = [rr["emax_rz"][0] for rr in rows]; z = [rr["emax_rz"][1] for rr in rows]
    sc = ax[1].scatter(r, z, c=np.log10(x), cmap="viridis", s=70, edgecolors="k", linewidths=0.4)
    from matplotlib.patches import Rectangle
    ax[1].add_patch(Rectangle((0.11, 0.557), 0.06, 1.048, fill=False, edgecolor="#dc2626",
                              ls="--", lw=1.5, label="insulator creepage band"))
    ax[1].set_xlabel("r [m]"); ax[1].set_ylabel("z [m]"); ax[1].legend(fontsize=8)
    ax[1].set_title("Emax location (colour=log10 sigma_s); stays at HV terminal, not patch edge")
    fig.colorbar(sc, ax=ax[1], label="log10 sigma_s")
    fig.tight_layout(); fig.savefig(PROC / "sigma_sweep_emax_location.png", dpi=150); plt.close(fig)


def fig_state_space(s30, c30):
    """Diagnostic state-space trajectory: surface_leak vs (tan-delta - clean) parametric in sigma."""
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    td0 = c30["tan_delta"] if c30 else 0.0
    x = [r["surface_leak"] for r in s30]; y = [r["tan_delta"] - td0 for r in s30]
    c = np.log10([r["sigma_s"] for r in s30])
    ax.plot(x, y, "-", color="0.6", zorder=1)
    sc = ax.scatter(x, y, c=c, cmap="plasma", s=80, edgecolors="k", linewidths=0.4, zorder=2)
    for r in s30:
        ax.annotate(f"{r['sigma_s']:.0e}", (r["surface_leak"], r["tan_delta"] - td0),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("surface_leak [A]"); ax.set_ylabel("tan-delta - clean")
    ax.set_title("Diagnostic state-space trajectory (fixed 30 deg sector, colour=log10 sigma_s)")
    ax.grid(alpha=0.3, which="both"); fig.colorbar(sc, ax=ax, label="log10 sigma_s")
    fig.tight_layout(); fig.savefig(PROC / "sigma_sweep_state_space.png", dpi=150); plt.close(fig)


def fig_power_consistency(s30):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    pd_ = [r["P_surface_direct_W"] for r in s30]; pc = [r["P_surface_from_current_W"] for r in s30]
    ax.loglog(pd_, pc, "o", color="#2563eb")
    lo, hi = min(pd_ + pc), max(pd_ + pc)
    ax.loglog([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("P_surface_direct [W]"); ax.set_ylabel("P_surface_from_current [W]")
    ax.set_title("Power consistency: direct integral vs U0*surface_leak (RMS; identical)")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(PROC / "sigma_sweep_power_consistency.png", dpi=150); plt.close(fig)


def fig_area_mesh(meta):
    a30 = meta["audit_30deg"]; fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax[0].bar(["measured", "expected"], [a30["area_fraction"], a30["expected_fraction"]],
              color=["#2563eb", "#9ca3af"])
    ax[0].set_ylabel("polluted area fraction")
    ax[0].set_title(f"30 deg patch area fraction\n(measured {a30['area_fraction']:.4f} vs "
                    f"expected {a30['expected_fraction']:.4f}, "
                    f"{(a30['area_fraction']/a30['expected_fraction']-1)*100:+.1f}% ; "
                    f"n_tris={a30['n_tris_patch']})")
    ax[0].grid(alpha=0.3, axis="y")
    labels = ["cut-edge\n(theta~30)", "seam\n(theta~0)", "insulator\n(all)", "global tet"]
    mins = [a30["cut_edge"]["min"], a30["seam_edge"]["min"],
            meta["insulator_surface"]["min"], meta["global_tet"]["min"]]
    p5s = [a30["cut_edge"]["p5"], a30["seam_edge"]["p5"],
           meta["insulator_surface"]["p5"], meta["global_tet"]["5"]]
    x = np.arange(len(labels)); w = 0.38
    ax[1].bar(x - w/2, mins, w, label="min", color="#dc2626")
    ax[1].bar(x + w/2, p5s, w, label="p5", color="#16a34a")
    ax[1].axhline(0.30, color="0.4", ls="--", lw=1, label="gate floor 0.30")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8); ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("minSICN"); ax[1].set_title("Mesh quality: cut edge vs revolution seam vs bulk")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PROC / "patch_area_and_mesh_quality.png", dpi=150); plt.close(fig)


def fig_qs_density():
    import pyvista as pv
    pv.OFF_SCREEN = True
    cam = [(2.6, -2.6, 1.9), (0.0, 0.0, 1.05), (0.0, 0.0, 1.0)]
    avail = [(ss, QDIR / f"qs_30deg_s{ss:.0e}.vtu") for ss in QS_SELECT]
    avail = [(ss, p) for ss, p in avail if p.is_file()]
    if not avail:
        print("  (no q_s VTUs for selected sigmas yet - skipping q_s density figure)"); return
    p = pv.Plotter(off_screen=True, shape=(1, len(avail)), window_size=(560 * len(avail), 900))
    p.set_background("white")
    for k, (ss, path) in enumerate(avail):
        g = pv.read(str(path))
        qmax = float(np.asarray(g.cell_data["q_s_Wm2"]).max())
        p.subplot(0, k)
        # unique scalar-bar title per panel (pyvista merges bars that share a title -> one scale)
        bar = {"title": f"q_s [W/m^2]  s={ss:.0e}", "vertical": True, "fmt": "%.0f",
               "title_font_size": 16, "label_font_size": 14, "position_x": 0.78, "width": 0.16}
        p.add_mesh(g.copy(), scalars="q_s_Wm2", cmap="inferno", clim=[0, qmax if qmax > 0 else 1],
                   show_scalar_bar=True, lighting=False, scalar_bar_args=bar)
        tag = {1e-9: "weak", 1e-7: "transition / weak-diagnostic", 1e-5: "severe"}.get(ss, "")
        p.add_text(f"sigma_s={ss:.0e} ({tag})\nmax q_s={qmax:.0f} W/m^2", position="upper_left",
                   font_size=11, color="black")
        p.camera_position = cam
    p.screenshot(str(PROC / "surface_loss_density_selected_sigmas.png")); p.close()


def main() -> int:
    sj = json.loads((PROC / "sweep.json").read_text(encoding="utf-8"))
    s30 = _nz(sj["sweep_30deg"]); c30 = _clean(sj["sweep_30deg"])
    s10 = _nz(sj.get("sweep_10deg", []))
    fig_surface_leak(s30, s10)
    fig_surface_power(s30, s10)
    fig_tandelta(s30, s10, c30)
    fig_phase(s30, c30)
    fig_emax(s30, c30)
    fig_state_space(s30, c30)
    fig_power_consistency(s30)
    fig_area_mesh(sj["mesh"])
    fig_qs_density()
    print(f"Wrote figures to {PROC} ({len(s30)} 30deg + {len(s10)} 10deg cases).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task 018 (analysis) - degradation-vector / state-space trajectory formulation.

Reads trajectories.json and produces:
  - velocity (degradation) vectors: dx/d(log10 sigma) for pollution per decade;
    dx/ds finite-difference directions for internal-fault severity
  - detection limits (smallest severity crossing a trend-monitoring threshold)
  - figures: pollution trajectory with arrows; diagnostic matrix with arrows;
    simplified decision tree
  - a short mathematical report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "results" / "processed" / "018_trajectories"

COMP = ["dvtap_pct", "dtheta_mdeg", "tan_delta", "surface_leak", "volume_leak"]
# trend-monitoring detection thresholds (deltas vs baseline; sensitive differential)
DET = {"dvtap_pct": 0.05, "dtheta_mdeg": 1.0, "tan_delta": 1e-4, "surface_leak": 1e-6, "volume_leak": 1e-5}
FAMCOL = {"pollution": "#7c3aed", "C1_cap": "#2563eb", "C2_cap": "#16a34a",
          "C1_loss": "#f59e0b", "C2_loss": "#dc2626", "disc_short": "#0891b2"}


def vec(x):
    return np.array([x[c] for c in COMP], float)


def main() -> int:
    d = json.loads((PROC / "trajectories.json").read_text(encoding="utf-8"))
    base = d["baseline"]["x"]
    bvol, btand = base["volume_leak"], base["tan_delta"]

    poll = d["pollution"]
    ls = np.array([q["log10s"] for q in poll])
    tand = np.array([q["x"]["tan_delta"] for q in poll])
    dth = np.array([q["x"]["dtheta_mdeg"] for q in poll])
    surf = np.array([q["x"]["surface_leak"] for q in poll])

    # ---- velocity vectors ----
    # pollution per-decade velocity in the (log10 tand, d-theta) projection
    poll_vel = []
    for i in range(len(poll) - 1):
        dlog = ls[i + 1] - ls[i]
        poll_vel.append({"from_sigma": poll[i]["s"], "to_sigma": poll[i + 1]["s"],
                         "d_log10_tand": float((np.log10(tand[i + 1]) - np.log10(tand[i])) / dlog),
                         "d_dtheta_mdeg": float((dth[i + 1] - dth[i]) / dlog)})

    # internal direction vectors (unit) baseline -> max severity, and per-step
    internal_dir = {}
    for fault, seq in d["internal"].items():
        x0 = vec(base)
        xf = vec(seq[-1]["x"])
        dirv = xf - x0
        # normalise by per-component detection scale so the direction is comparable
        scale = np.array([DET[c] for c in COMP])
        nrm = dirv / scale
        unit = nrm / (np.linalg.norm(nrm) + 1e-300)
        internal_dir[fault] = {"raw": dirv.tolist(), "unit_detnorm": unit.tolist(),
                               "dominant": COMP[int(np.argmax(np.abs(unit)))]}

    # ---- detection limits: smallest severity crossing any threshold ----
    det_limits = {}
    # pollution
    for q in poll:
        x = q["x"]
        crossed = [c for c in COMP if abs(x[c] - base[c]) > DET[c]]
        if crossed:
            det_limits["pollution"] = {"sigma_s": q["s"], "first_obs": crossed}
            break
    for fault, seq in d["internal"].items():
        for q in seq[1:]:  # seq[0] is the per-fault baseline severity (no change)
            x = q["x"]
            crossed = [c for c in COMP if abs(x[c] - base[c]) > DET[c]]
            if crossed:
                det_limits[fault] = {"severity": q["s"], "first_obs": crossed}
                break

    # ============================ FIGURES ============================
    # A) pollution trajectory with arrows in (log10 tand, d-theta)
    fig, ax = plt.subplots(figsize=(8.2, 6))
    x = np.log10(tand); y = dth
    sc = ax.scatter(x, y, c=ls, cmap="viridis", s=80, zorder=3, edgecolors="k")
    ax.plot(x, y, "-", color="0.6", lw=1, zorder=2)
    for i in range(len(x) - 1):
        ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.8), zorder=4)
    icrit = int(np.argmin(dth))
    ax.scatter([x[icrit]], [y[icrit]], marker="*", s=300, color="gold", edgecolors="k", zorder=5)
    ax.annotate(f"critical (sigma_s={poll[icrit]['s']:.0e})", (x[icrit], y[icrit]),
                textcoords="offset points", xytext=(8, -14), fontsize=8)
    cb = fig.colorbar(sc, ax=ax); cb.set_label(r"$\log_{10}\sigma_s$")
    ax.set_xlabel(r"$\log_{10}\tan\delta$"); ax.set_ylabel(r"$\Delta\theta$ [mdeg]")
    ax.set_title("Pollution degradation trajectory (arrows = velocity per decade of $\\sigma_s$)")
    ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(PROC / "pollution_trajectory_arrows.png", dpi=170); plt.close(fig)

    # B) diagnostic matrix with degradation-direction arrows
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    def arrow(a, x0, y0, x1, y1, color, label):
        a.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                    color=color, lw=2.2, zorder=4))
        a.annotate(label, (x1, y1), textcoords="offset points", xytext=(5, 3), fontsize=8, color=color)

    # panel 1: d|Vtap| vs log10 tand
    a = ax[0]
    a.scatter(0, np.log10(btand), marker="*", s=200, color="k", zorder=5, label="healthy")
    for f, seq in d["internal"].items():
        xf = seq[-1]["x"]
        arrow(a, 0, np.log10(btand), xf["dvtap_pct"], np.log10(xf["tan_delta"]), FAMCOL[f], f)
    pf = poll[-2]["x"]  # a strong pollution point
    arrow(a, 0, np.log10(btand), pf["dvtap_pct"], np.log10(pf["tan_delta"]), FAMCOL["pollution"], "pollution")
    a.set_xlabel(r"$\Delta$|Vtap| [%]"); a.set_ylabel(r"$\log_{10}\tan\delta$")
    a.set_title("ratio faults -> |Vtap|;  loss/pollution -> tan-delta"); a.grid(True, alpha=0.3); a.axvline(0, color="0.85")

    # panel 2: surface_leak vs volume_leak (log, floored)
    a = ax[1]
    fl = 1e-7
    a.scatter(bvol, fl, marker="*", s=200, color="k", zorder=5, label="healthy")
    for f, seq in d["internal"].items():
        xf = seq[-1]["x"]
        arrow(a, bvol, fl, max(xf["volume_leak"], fl), max(xf["surface_leak"], fl), FAMCOL[f], f)
    arrow(a, bvol, fl, max(pf["volume_leak"], fl), max(pf["surface_leak"], fl), FAMCOL["pollution"], "pollution")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("volume loss current [A]"); a.set_ylabel("surface leakage [A] (floored 1e-7)")
    a.set_title("external (surface) vs internal (volume) leakage split"); a.grid(True, which="both", alpha=0.3)
    fig.suptitle("Diagnostic matrix with degradation vectors (baseline -> max severity)", fontsize=13)
    fig.savefig(PROC / "diagnostic_matrix_arrows.png", dpi=150); plt.close(fig)

    # C) simplified decision tree
    fig, ax = plt.subplots(figsize=(11, 8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(x0, y0, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.08", fc=fc, ec="k", lw=1.2))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)

    def link(x0, y0, x1, y1, lab=""):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color="k", lw=1.3))
        if lab:
            ax.text((x0 + x1) / 2 + 0.15, (y0 + y1) / 2, lab, fontsize=8, color="k")

    box(3.5, 9.0, 3.0, 0.8, "measure dx =\n[d|Vtap|, dθ, tanδ, surf_leak, vol_leak]", "#e5e7eb")
    box(3.7, 7.6, 2.6, 0.8, "|d|Vtap|| > 3% ?", "#fef3c7")
    box(6.9, 7.6, 3.0, 0.8, "capacitor ratio fault\n(+ = C1, − = C2, short)", "#bfdbfe")
    box(3.7, 6.2, 2.6, 0.8, "surface_leak ≫ 0 ?", "#fef3c7")
    box(6.9, 6.2, 3.0, 0.8, "external pollution\n(bridge: high leak; patch: phase only)", "#ddd6fe")
    box(3.7, 4.8, 2.6, 0.8, "tanδ↑ & |dθ| degree-scale ?", "#fef3c7")
    box(6.9, 4.8, 3.0, 0.8, "internal dielectric loss\n(dθ<0 = C1, dθ>0 = C2)", "#fed7aa")
    box(3.7, 3.4, 2.6, 0.8, "any obs > detection ?", "#fef3c7")
    box(6.9, 3.4, 3.0, 0.8, "weak / localized\n(below confident ID)", "#fee2e2")
    box(3.7, 2.0, 2.6, 0.8, "healthy / nominal", "#dcfce7")
    link(5.0, 9.0, 5.0, 8.4);
    link(5.0, 7.6, 5.0, 7.0, "no"); link(6.3, 8.0, 6.9, 8.0, "yes")
    link(5.0, 6.2, 5.0, 5.6, "no"); link(6.3, 6.6, 6.9, 6.6, "yes")
    link(5.0, 4.8, 5.0, 4.2, "no"); link(6.3, 5.2, 6.9, 5.2, "yes")
    link(5.0, 3.4, 5.0, 2.8, "no"); link(6.3, 3.8, 6.9, 3.8, "yes")
    ax.set_title("Simplified fault-classification decision tree", fontsize=13)
    fig.savefig(PROC / "decision_tree.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # ============================ REPORT ============================
    L = ["# Task 018 - State-Space Trajectory & Degradation-Vector Formulation", "",
         "## 1. State point",
         "Observable state vector (relative to the healthy/clean baseline x0):",
         "",
         "    x = [ d|Vtap| (%),  dθ (mdeg),  tanδ,  surface_leak (A),  volume_leak (A) ]",
         "",
         f"Baseline x0: tanδ0 = {btand:.3e}, volume_leak0 = {bvol:.3e} A, surface_leak0 = 0,",
         "d|Vtap| = dθ = 0. A fault moves the device to a point x in this 5-D space.",
         "",
         "## 2. Fault-family manifolds  x = M_f(s)",
         "Each family is a 1-parameter curve through x0, parameterised by severity s:",
         "- pollution  M_poll(log10 σ_s): full-creepage surface conductance (Task 10/15)",
         "- C1/C2 cap  M_cap(eps factor): capacitor permittivity (Task 16)",
         "- C1/C2 loss M_loss(tanδ): element dielectric loss (Task 16)",
         "- disc short M_short(n): n shorted C1 elements (Task 16)",
         "",
         "## 3. Pollution trajectory & velocity (per decade of σ_s)",
         "Velocity v = dx/d(log10 σ_s) in the (log10 tanδ, dθ) projection:",
         "",
         "| σ_s decade | d(log10 tanδ)/dec | d(dθ)/dec [mdeg] |",
         "|---|---|---|"]
    for v in poll_vel:
        L.append(f"| {v['from_sigma']:.0e}->{v['to_sigma']:.0e} | {v['d_log10_tand']:+.2f} | {v['d_dtheta_mdeg']:+.1f} |")
    L += ["",
          "tanδ and surface_leak rise ~one decade per decade of σ_s (slope ~ +1); dθ first",
          "dives (capacitive-polluting branch) then recovers (resistive-bridge branch) - the",
          "critical point dθ' = 0 is the relaxation corner. See pollution_trajectory_arrows.png.",
          "",
          "## 4. Internal-fault degradation vectors  v = dx/ds",
          "Direction (unit vector in detection-normalised coordinates; dominant observable):",
          "",
          "| fault | d|Vtap| | dθ | tanδ | surf | vol | dominant |",
          "|---|---|---|---|---|---|---|"]
    for f, info in internal_dir.items():
        u = info["unit_detnorm"]
        L.append(f"| {f} | {u[0]:+.2f} | {u[1]:+.2f} | {u[2]:+.2f} | {u[3]:+.2f} | {u[4]:+.2f} | {info['dominant']} |")
    L += ["",
          "Cap/short vectors point almost purely along d|Vtap|; loss vectors along (dθ, tanδ,",
          "volume_leak) with dθ sign = section; pollution points along (surface_leak, tanδ).",
          "The families are near-orthogonal in detection-normalised space -> separable.",
          "",
          "## 5. Classification rule",
          "Given measured dx, in order:",
          "1. |d|Vtap|| > 3%  -> capacitor ratio fault  (+ = C1 / short, − = C2)",
          "2. surface_leak ≫ baseline  -> external creepage pollution (bridge);",
          "   phase-only with ~0 leak -> localized pollution patch (sign = position)",
          "3. tanδ↑ + volume_leak↑ + degree-scale dθ  -> internal dielectric loss (dθ<0 C1, >0 C2)",
          "4. else -> healthy / below detection. See decision_tree.png.",
          "",
          "## 6. Detection limits (trend-monitoring thresholds)",
          f"Thresholds: d|Vtap| {DET['dvtap_pct']}%, dθ {DET['dtheta_mdeg']} mdeg, tanδ {DET['tan_delta']:.0e},",
          f"surface_leak {DET['surface_leak']:.0e} A, volume_leak {DET['volume_leak']:.0e} A (deltas vs baseline).",
          "Smallest severity that crosses any threshold:",
          ""]
    for fam, info in det_limits.items():
        key = "sigma_s" if "sigma_s" in info else "severity"
        L.append(f"- **{fam}**: {key} = {info[key]:.3g}  (first via {', '.join(info['first_obs'])})")
    L += ["",
          "Blind spot: a mid-creepage non-bridging pollution patch stays below all thresholds",
          "(sub-mdeg phase, baseline leak/tanδ) -> not identifiable from terminal observables.",
          "",
          "## Figures",
          "- pollution_trajectory_arrows.png, diagnostic_matrix_arrows.png, decision_tree.png",
          ""]
    (PROC / "trajectory_report.md").write_text("\n".join(L), encoding="utf-8")

    print("detection limits:", {k: (v.get("sigma_s") or v.get("severity")) for k, v in det_limits.items()})
    print("internal dominant:", {f: i["dominant"] for f, i in internal_dir.items()})
    print(f"figures + report in {PROC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

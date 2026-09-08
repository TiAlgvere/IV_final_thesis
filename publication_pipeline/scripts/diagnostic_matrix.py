"""Task 016.x - diagnostic matrix: external pollution family vs internal stack-fault
family in the split-observable space.

Reads the Task 015 (pollution) and Task 016 (internal) summaries (which now carry
the split leakage: surface_leak vs volume_leak) and produces:
  - a 4-panel separation figure:
      1. d|Vtap| vs tan-delta
      2. phase  vs tan-delta
      3. phase  vs surface leakage
      4. surface leakage vs volume leakage  (the external/internal leakage split)
  - a fault -> signature table with a rules-based diagnosis.

Axes use log / symlog so the mdeg-vs-degree phase scale and the decade-spanning
leakages are all legible (fixes the earlier flat/overlapping/clipped plot).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "results" / "processed" / "017_diagnostic_matrix"
POLL = ROOT / "results" / "processed" / "015_localized_pollution" / "summary.json"
INT = ROOT / "results" / "processed" / "016_stack_faults" / "summary.json"

COLORS = {
    "full": "#7c3aed", "top25": "#a855f7", "mid25": "#c084fc", "bottom25": "#6d28d9", "streamer": "#4c1d95",
    "healthy": "#111827", "C1_cap": "#2563eb", "C2_cap": "#16a34a",
    "C1_loss": "#f59e0b", "C2_loss": "#dc2626", "disc_short": "#0891b2",
}


def _get(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return 0.0


def diagnose(r, base_tand, base_vol):
    """Rules-based classifier from the split signature."""
    dv, dth, tand = r["dvtap_pct"], r["dtheta_mdeg"], r["tan_delta"]
    sl, vl = r["surface_leak"], r["volume_leak"]
    if abs(dv) > 3.0:
        sec = "C1" if dv > 0 else "C2"
        return f"capacitor ratio fault ({sec} cap up / disc short)"
    if sl > 5.0 * max(base_vol, 1e-12):
        return "external creepage pollution / bridge"
    if tand > 2.0 * base_tand and abs(dth) > 100.0 and vl > 2.0 * base_vol:
        return f"internal dielectric loss ({'C1' if dth < 0 else 'C2'})"
    if abs(dth) > 0.5 and tand < 2.0 * base_tand and sl < 5.0 * base_vol:
        return "localized pollution patch (position by phase sign)"
    return "healthy / nominal"


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    pol = json.loads(POLL.read_text(encoding="utf-8"))["cases"]
    intr = json.loads(INT.read_text(encoding="utf-8"))["rows"]

    base = next(r for r in intr if r["case"] == "healthy")
    base_tand, base_vol = base["tan_delta"], base["volume_leak"]

    pts = []
    for r in pol:
        pts.append({"family": "pollution", "marker": "^",
                    "case": r["case"], "dvtap_pct": _get(r, "dvtap_pct"),
                    "dtheta_mdeg": r["dtheta_mdeg"], "tan_delta": r["tan_delta"],
                    "surface_leak": r["surface_leak"], "volume_leak": r["volume_leak"]})
    for r in intr:
        pts.append({"family": "internal" if r["case"] != "healthy" else "baseline",
                    "marker": "*" if r["case"] == "healthy" else "o",
                    "case": r["case"], "dvtap_pct": r["dvtap_pct"],
                    "dtheta_mdeg": r["dtheta_mdeg"], "tan_delta": r["tan_delta"],
                    "surface_leak": r["surface_leak"], "volume_leak": r["volume_leak"]})

    for p in pts:
        p["diag"] = diagnose(p, base_tand, base_vol)

    def scatter(ax, xk, yk):
        for p in pts:
            ax.scatter(p[xk], p[yk], marker=p["marker"], s=170 if p["marker"] == "*" else 120,
                       color=COLORS.get(p["case"], "#888"), edgecolors="k", linewidths=0.6, zorder=4)

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 11), constrained_layout=True)
    LT_PHASE, LT_LEAK = 10.0, 1e-6  # symlog thresholds

    a = ax[0, 0]; scatter(a, "dvtap_pct", "tan_delta")
    a.set_yscale("log"); a.set_xlabel(r"$\Delta$|Vtap| [%]"); a.set_ylabel(r"terminal $\tan\delta$")
    a.set_title("1) ratio faults split off on |Vtap|"); a.axvline(0, color="0.8", lw=1); a.margins(x=0.15)

    a = ax[0, 1]; scatter(a, "tan_delta", "dtheta_mdeg")
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=LT_PHASE)
    a.set_xlabel(r"terminal $\tan\delta$"); a.set_ylabel(r"$\Delta\theta$ [mdeg] (symlog)")
    a.set_title("2) internal loss = degree-scale phase; sign = C1/C2"); a.axhline(0, color="0.8", lw=1)

    a = ax[1, 0]; scatter(a, "surface_leak", "dtheta_mdeg")
    a.set_xscale("symlog", linthresh=LT_LEAK); a.set_yscale("symlog", linthresh=LT_PHASE)
    a.set_xlabel("surface leakage [A] (symlog)"); a.set_ylabel(r"$\Delta\theta$ [mdeg] (symlog)")
    a.set_title("3) pollution bridge = high surface leakage"); a.axhline(0, color="0.8", lw=1)

    a = ax[1, 1]; scatter(a, "volume_leak", "surface_leak")
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=LT_LEAK)
    a.set_xlabel("volume loss current [A] (log)"); a.set_ylabel("surface leakage [A] (symlog)")
    a.set_title("4) the leakage split: external (y) vs internal (x)")

    handles = [Line2D([0], [0], marker=("*" if c == "healthy" else ("^" if c in ("full", "top25", "mid25", "bottom25", "streamer") else "o")),
                      color="w", markerfacecolor=COLORS[c], markeredgecolor="k", markersize=11, label=c)
               for c in COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Diagnostic matrix - external pollution (triangles) vs internal stack faults (circles)", fontsize=14)
    fig.savefig(PROC / "diagnostic_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- fault -> signature table ----
    def tag(v, lo, hi, unit=""):
        return f"{v:+.2g}{unit}"
    lines = [
        "# Diagnostic Matrix - Fault -> Signature",
        "",
        f"Baseline (healthy/clean): tan-delta {base_tand:.2e}, volume loss current {base_vol:.2e} A,",
        "surface leakage 0. d|Vtap| and d-theta are vs that baseline. Split leakage:",
        "surface_leak = creepage surface-conductance current; volume_leak = bulk loss current.",
        "",
        "| family | case | d|Vtap| [%] | d-theta [mdeg] | tan-delta | surface leak [A] | volume leak [A] | diagnosis |",
        "|--------|------|-------------|----------------|-----------|------------------|-----------------|-----------|",
    ]
    order = {"baseline": 0, "internal": 1, "pollution": 2}
    for p in sorted(pts, key=lambda x: (order[x["family"]], x["case"])):
        lines.append(f"| {p['family']} | {p['case']} | {p['dvtap_pct']:+.2f} | {p['dtheta_mdeg']:+.1f} | "
                     f"{p['tan_delta']:.2e} | {p['surface_leak']:.2e} | {p['volume_leak']:.2e} | {p['diag']} |")
    lines += [
        "",
        "## Classifier (from the split signature)",
        "1. **|d|Vtap|| > 3%**  -> capacitor ratio fault / disc short  (Vtap up = C1 side, down = C2).",
        "2. **surface leakage high** (>> baseline volume loss) -> external creepage pollution / bridge.",
        "3. **tan-delta up + volume loss up + degree-scale phase** -> internal dielectric loss",
        "   (phase sign: negative = C1 / above tap, positive = C2 / below tap).",
        "4. **phase shift with low tan-delta and ~zero surface leakage** -> localized pollution patch",
        "   (position from the phase sign).",
        "",
        "The split leakage is the key new axis: it cleanly separates the EXTERNAL path",
        "(surface leakage, pollution) from the INTERNAL path (volume loss, dielectric aging) -",
        "previously both collapsed into one 'leakage' number. See `diagnostic_matrix.png`.",
        "",
    ]
    (PROC / "diagnostic_matrix.md").write_text("\n".join(lines), encoding="utf-8")
    print("Families:", {f: sum(1 for p in pts if p["family"] == f) for f in order})
    print(f"Wrote {PROC / 'diagnostic_matrix.png'} and diagnostic_matrix.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

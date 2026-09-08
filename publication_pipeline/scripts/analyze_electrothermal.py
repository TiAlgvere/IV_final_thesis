"""Task 021 - analyse the one-way electrothermal results.

Reads results/processed/021_electrothermal/thermal_summary.csv (+ per-case T-field
npz + thermal_config.json) and produces:

  a) thermal_field_maps.png   T(r,z) for healthy / worst internal loss / heavy pollution
  b) thermal_risk_curve.png   dT_max vs severity (loss & pollution) with risk bands
  c) diagnosis_to_risk.png    the synthesis matrix: electrical signature -> thermal risk

and electrothermal_report.md. Model-based risk for the representative DDB-123 - the
relative ranking is robust; absolute dT depends on the documented thermal assumptions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "results" / "processed" / "021_electrothermal"
CSV = PROC / "thermal_summary.csv"
CFG = json.loads((PROC / "thermal_config.json").read_text(encoding="utf-8"))
T_AMB = CFG["T_amb_C"]; HOT = CFG["insul_hotspot_C"]
DT_LIMIT = HOT - T_AMB    # dT that reaches the insulation hot-spot limit

# risk bands by dT (K): (upper, label, color)
BANDS = [(1, "negligible", "#dcfce7"), (10, "minor", "#fef9c3"),
         (30, "elevated", "#fed7aa"), (DT_LIMIT, "high", "#fecaca"),
         (1e9, "critical", "#fca5a5")]
COLORS = {"healthy": "#111827", "C1_loss": "#f59e0b", "C2_loss": "#dc2626",
          "pollution": "#7c3aed", "C1_cap": "#2563eb", "disc_short": "#0891b2"}


def load_rows():
    rows = []
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("severity", "p_vol_W", "p_surf_W", "p_total_W", "dt_max_K", "T_abs_C",
                      "hot_r", "hot_z", "tan_delta", "surface_leak", "volume_leak"):
                if k in r:
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def field(tag):
    d = np.load(PROC / f"Tfield_{tag}.npz")
    return d["pts"], d["tris"], d["T"]


def fig_field_maps(rows):
    panels = [("healthy", "healthy (baseline dielectric loss)"),
              ("C1_loss_0.1", r"C1 dielectric loss ($\tan\delta$=0.10)"),
              ("pollution_1e-06", r"heavy pollution ($\sigma_s$=1e-6 S)")]
    fig, ax = plt.subplots(1, 3, figsize=(12, 7.4))
    for k, (tag, title) in enumerate(panels):
        pts, tris, T = field(tag)
        used = np.unique(tris)
        rmax = pts[used, 0].max(); zmin, zmax = pts[used, 1].min(), pts[used, 1].max()
        tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
        tc = ax[k].tripcolor(tri, T_AMB + T, shading="gouraud", cmap="inferno")
        im = int(np.nanargmax(T))
        ax[k].plot(pts[im, 0], pts[im, 1], "c*", ms=15, mec="k", zorder=5)
        ax[k].set_xlim(0, rmax * 1.05); ax[k].set_ylim(zmin - 0.03, zmax + 0.03)
        ax[k].set_aspect(0.32)   # radius exaggerated ~3x for visibility (tall thin device)
        ax[k].set_title(title, fontsize=10); ax[k].set_xlabel("r [m]")
        if k == 0:
            ax[k].set_ylabel("z [m]  (radius exaggerated ~3x)")
        cb = fig.colorbar(tc, ax=ax[k], fraction=0.09, pad=0.03)
        cb.set_label("T [C]")
    fig.suptitle("Task 021 - temperature field: internal loss heats the stack; "
                 "pollution heats the creepage surface", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PROC / "thermal_field_maps.png", dpi=150)
    plt.close(fig)


def _bands(ax):
    lo = 0.0
    for up, lab, col in BANDS:
        ax.axhspan(lo, up, color=col, alpha=0.6, zorder=0)
        lo = up
    ax.axhline(DT_LIMIT, color="#b91c1c", lw=1.4, ls="--",
               label=f"insulation hot-spot limit (+{DT_LIMIT:.0f} K -> {HOT:.0f} C)")


def fig_risk_curve(rows):
    loss = sorted([r for r in rows if r["family"] == "C1_loss"], key=lambda r: r["severity"])
    poll = sorted([r for r in rows if r["family"] == "pollution"], key=lambda r: r["severity"])
    base = next(r for r in rows if r["family"] == "healthy")["dt_max_K"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    a = ax[0]; _bands(a)
    a.plot([r["severity"] for r in loss], [r["dt_max_K"] for r in loss], "o-",
           color=COLORS["C1_loss"], lw=2, label="C1 dielectric loss")
    a.axhline(base, color="0.3", lw=1, ls=":", label=f"healthy ({base:.1f} K)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"element $\tan\delta$"); a.set_ylabel(r"max temperature rise $\Delta T$ [K]")
    a.set_title("internal dielectric loss -> volumetric heating"); a.legend(fontsize=8, loc="upper left")
    a.set_ylim(1, 400)

    a = ax[1]; _bands(a)
    a.plot([r["severity"] for r in poll], [r["dt_max_K"] for r in poll], "o-",
           color=COLORS["pollution"], lw=2, label="full-creepage pollution")
    a.axhline(base, color="0.3", lw=1, ls=":", label=f"healthy ({base:.1f} K)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"surface conductance $\sigma_s$ [S]"); a.set_ylabel(r"$\Delta T$ [K]")
    a.set_title("surface pollution -> creepage heating"); a.legend(fontsize=8, loc="upper left")
    a.set_ylim(1, 400)

    fig.suptitle("Task 021 - severity -> thermal risk (dT bands; absolute values depend on "
                 "thermal assumptions)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PROC / "thermal_risk_curve.png", dpi=150)
    plt.close(fig)


def fig_synthesis(rows):
    """Diagnosis -> risk matrix table."""
    spec = [
        ("C1/C2 capacitance", "|Vtap| (ratio)", "trend, +0.2%", "C1_cap"),
        ("disc short", "|Vtap| + phase", "trivial", "disc_short"),
        ("C1 dielectric loss", "phase + tan-delta", "tan-delta ~0.003", "C1_loss"),
        ("C2 dielectric loss", "phase + tan-delta", "tan-delta ~0.003", "C2_loss"),
        ("surface pollution", "surface leakage", "any film", "pollution"),
    ]
    def worst(fam):
        rs = [r for r in rows if r["family"] == fam]
        return max(rs, key=lambda r: r["dt_max_K"])
    cells = []
    rlabels = []
    for label, esig, edet, fam in spec:
        w = worst(fam)
        sev = (f"+{w['severity']:.0f}% / short" if fam in ("C1_cap", "disc_short")
               else (f"tand {w['severity']:g}" if "loss" in fam else f"sigma_s {w['severity']:g}"))
        interp = ("metrology - LOW risk" if w["dt_max_K"] < 10
                  else ("dry-band heating - RISK" if fam == "pollution"
                        else "thermal degradation - HIGH"))
        risk_short = "critical" if w["risk"].startswith("critical") else w["risk"]
        cells.append([label, esig, edet, sev, f"{w['dt_max_K']:.0f} K",
                      f"{w['T_abs_C']:.0f} C", risk_short, interp])
        rlabels.append(w["risk"])
    col = ["fault family", "electrical\nsignature", "operational\ndetectability",
           "worst case\nshown", "dT_max", "T_abs", "thermal\nrisk", "interpretation"]
    widths = [0.15, 0.13, 0.15, 0.13, 0.07, 0.07, 0.10, 0.20]
    fig, ax = plt.subplots(figsize=(15.5, 3.2)); ax.axis("off")
    t = ax.table(cellText=cells, colLabels=col, loc="center", cellLoc="center", colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 2.0)
    rmap = {"negligible": "#dcfce7", "minor": "#fef9c3", "elevated": "#fed7aa",
            "high": "#fecaca", "critical (>hot-spot limit)": "#fca5a5"}
    for (row, c), cell in t.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b"); cell.set_text_props(color="w", weight="bold")
        elif c == 6:
            cell.set_facecolor(rmap.get(rlabels[row-1], "#fff"))
        elif c == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Task 021 - diagnosis -> thermal risk: which fault signatures are real "
                 "thermal hazards", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(PROC / "diagnosis_to_risk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = load_rows()
    fig_field_maps(rows)
    fig_risk_curve(rows)
    fig_synthesis(rows)

    base = next(r for r in rows if r["family"] == "healthy")
    L = []
    L += ["# Task 021 - One-Way Electrothermal Multiphysics (diagnosis -> risk)", "",
          "## Purpose", "",
          "Add the next layer above the diagnostic matrix: take the electrical loss from the",
          "verified ComplexEQS solve, use it as a heat source, solve the steady temperature",
          "rise, and decide **which fault signatures correspond to real thermal risk** (severity),",
          "not just electrical detectability. One-way coupling only (EQS -> thermal).", ""]

    L += ["## Method & model", "",
          "- **Heat source** reuses the EQS loss integrands directly, so it is energy-consistent",
          "  with the diagnostics: volumetric q''' = 0.5*sigma_eff*|E|^2 in the dielectric",
          "  (p_vol = 0.5*volume_leak*U0) and a creepage SURFACE source",
          "  0.5*sigma_s*|grad_s phi|^2 for pollution (p_surf = 0.5*surface_leak*U0).",
          "- **Thermal solver**: self-contained axisymmetric P1 steady heat conduction",
          "  (`thermal_solver.py`), -div(k grad T)=q''' on the SOLID sub-domain, with the air",
          "  region replaced by a convective boundary. **Verified** against the analytic",
          "  uniformly-heated cylinder (center dT 33.34 vs 33.33 K, 0.03 % error).",
          "- (Numerical) capped-sigma metals carry no physical loss, so their volumetric heat is",
          "  zeroed; only dielectric/surface losses heat the device.", ""]

    L += ["## Documented thermal assumptions (revise with project data)", "",
          f"- ambient T_amb = {T_AMB:.0f} C (hot-day operating);",
          f"- natural convection h = {CFG['h_conv']:.0f} W/m2K on the solid-air surface;",
          "- thermal conductivity k [W/mK]: " +
          ", ".join(f"{k}={v}" for k, v in CFG["k_thermal"].items()) + ";",
          f"- insulation hot-spot limit {HOT:.0f} C (oil/paper, IEC-class guidance).",
          "- **The relative ranking of faults is robust; absolute dT scales with these "
          "assumptions** (especially h and the oil conductivity).", ""]

    L += ["## Results (12 cases)", "",
          "| case | family | p_vol [W] | p_surf [W] | dT_max [K] | T_abs [C] | hot-spot | risk |",
          "|------|--------|-----------|------------|------------|-----------|----------|------|"]
    for r in rows:
        L.append(f"| {r['case']} | {r['family']} | {r['p_vol_W']:.1f} | {r['p_surf_W']:.1f} | "
                 f"{r['dt_max_K']:.1f} | {r['T_abs_C']:.1f} | {r['hot_body']} | {r['risk']} |")
    L += [""]

    L += ["## Diagnosis -> thermal risk (the key finding)", "",
          "The diagnostic families split cleanly by thermal risk:", "",
          "- **Capacitance / disc-short (ratio faults): electrically loud, thermally SILENT.**",
          f"  They add ~no loss (dT ~ {base['dt_max_K']:.1f} K, ~healthy) - a metrology/ratio fault,",
          "  not a thermal hazard. Detect and schedule, but no thermal urgency.",
          "- **Internal dielectric loss: electrically + thermally CRITICAL.** dT_max scales ~linearly",
          "  with tan-delta and crosses the insulation hot-spot limit by tan-delta ~0.05",
          "  (T_abs > 105 C). Because real tan-delta itself rises with temperature, this is the",
          "  one-way precursor of thermal runaway - the highest-severity family. The hot-spot",
          "  localises to the faulted section (C1 -> mid-stack element_5; C2 -> element_11).",
          "- **C2 loss is more severe per watt than C1 loss**: the C2 loss concentrates in only 2",
          "  elements, so a smaller total power (44 W) still produces a higher local hot-spot",
          "  (90 K) than a larger but more distributed C1 loss.",
          "- **Surface pollution: thermally relevant only when heavy.** Light films (sigma_s <=1e-7)",
          "  are an electrical signal but thermally minor; heavy wetting (sigma_s 1e-6, ~720 W)",
          "  reaches 'high' risk (creepage ~74 C) - the dry-band / flashover precursor. The",
          "  surface heat is deposited at the cooled boundary, so much convects away; a localised",
          "  real dry band (not modelled) would concentrate it further.", ""]

    L += ["## How this complements the diagnostic matrix", "",
          "The Task 17-20 matrix answers *what* the fault is and *whether* it is detectable; this",
          "layer answers *how urgent* it is. Combined rule:",
          "- ratio fault (Vtap) -> identify + trend, low thermal urgency;",
          "- dielectric loss (phase + tan-delta) -> thermal hazard, escalating -> prioritise;",
          "- surface leakage (pollution) -> thermal hazard only once leakage/power is large.", ""]

    L += ["## Caveats", "",
          "- One-way coupling only: no temperature feedback on tan-delta/conductivity (so the",
          "  real loss-fault case would run hotter - thermal runaway is under-estimated, not over).",
          "- Steady-state only (no thermal time constant / transient).",
          "- Uniform pollution + uniform convection; a localised dry band would give sharper local",
          "  heating than the smeared surface source here.",
          "- Absolute temperatures depend on the documented thermal properties and h; these are",
          "  **model-based risk estimates for the representative DDB-123, not field-validated**.", "",
          "Figures: thermal_field_maps.png, thermal_risk_curve.png, diagnosis_to_risk.png. "
          "Data: thermal_summary.csv, Tfield_*.npz.", ""]

    (PROC / "electrothermal_report.md").write_text("\n".join(L), encoding="utf-8")
    print("Wrote 3 figures + electrothermal_report.md in", PROC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

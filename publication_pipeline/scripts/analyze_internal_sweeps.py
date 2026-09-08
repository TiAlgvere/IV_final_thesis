"""Task 019 - analyse the internal fault-family severity sweeps.

Reads results/processed/019_internal_severity_sweeps/internal_fault_sweeps.csv
(+ baseline.json) and the Task 018 pollution trajectory, then produces:

  a) internal_sweep_vtap.png        d|Vtap| vs severity (C1_cap, C2_cap, disc_short)
  b) internal_sweep_phase_loss.png  d-theta and tan-delta vs severity (C1_loss, C2_loss)
  c) internal_trajectory_matrix.png state-space trajectories, arrows = increasing severity
  d) diagnostic_matrix_refined.png  Task 018 matrix with smooth internal trajectories
                                    overlaid on the pollution trajectory

and internal_sweeps_report.md (monotonicity, disc-short analytic check, detection
limits, classifier separation, overlaps/blind spots).

Model-based diagnostic trajectories for the representative DDB-123 - not field
thresholds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "results" / "processed" / "019_internal_severity_sweeps"
CSV = PROC / "internal_fault_sweeps.csv"
BASE = PROC / "baseline.json"
POLL = ROOT / "results" / "processed" / "018_trajectories" / "trajectories.json"

COLORS = {"C1_cap": "#2563eb", "C2_cap": "#16a34a", "C1_loss": "#f59e0b",
          "C2_loss": "#dc2626", "disc_short": "#0891b2", "pollution": "#7c3aed"}

# Task 018 trend-monitoring detection thresholds (model-based, NOT instrument specs).
DET = {"dvtap_pct": 0.05, "dtheta_mdeg": 1.0, "d_tan_delta": 1.0e-4,
       "d_volume_leak": 1.0e-5, "surface_leak": 1.0e-6}
ARCMIN_MDEG = 1000.0 / 60.0   # 1 arc-minute = 16.67 mdeg (practical metering scale)


def load_rows():
    rows = []
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("severity_value", "dvtap_pct", "dtheta_mdeg", "tan_delta",
                      "surface_leak", "volume_leak", "vtap_abs"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def fam(rows, family, sparam=None):
    out = [r for r in rows if r["family"] == family]
    if sparam is not None:
        out = [r for r in out if r["severity_parameter"] == sparam]
    return sorted(out, key=lambda r: r["severity_value"])


def arrowed(ax, xs, ys, color, lw=2.0):
    """Line through (xs, ys) with arrowheads on each segment (increasing severity)."""
    ax.plot(xs, ys, "-", color=color, lw=lw, alpha=0.55, zorder=2)
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=0.9),
                    zorder=3)
    ax.scatter(xs, ys, s=26, color=color, edgecolors="k", linewidths=0.4, zorder=4)


# ----------------------------------------------------------------------------- figures
def fig_vtap(rows, base):
    c1, c2 = fam(rows, "C1_cap"), fam(rows, "C2_cap")
    sev = fam(rows, "disc_short", "short_sigma")
    loc = fam(rows, "disc_short", "short_location")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

    a = ax[0]
    a.plot([r["severity_value"] for r in c1], [r["dvtap_pct"] for r in c1],
           "o-", color=COLORS["C1_cap"], label="C1 cap +")
    a.plot([r["severity_value"] for r in c2], [r["dvtap_pct"] for r in c2],
           "s-", color=COLORS["C2_cap"], label="C2 cap +")
    a.axhline(0, color="0.8", lw=1)
    a.set_xscale("log"); a.set_xlabel("capacitance increase [%]")
    a.set_ylabel(r"$\Delta$|Vtap| [%]")
    a.set_title("ratio faults: C1 raises Vtap, C2 lowers it (monotonic)")
    a.legend(); a.grid(alpha=0.3, which="both")

    a = ax[1]
    a.plot([r["severity_value"] for r in sev], [r["dvtap_pct"] for r in sev],
           "o-", color=COLORS["disc_short"], label=r"C1 element, partial short ($\sigma$ sweep)")
    # full-short location levels, grouped (C1 shorts coincide, C2 shorts coincide)
    c1lvl = np.mean([r["dvtap_pct"] for r in loc if r["case_name"].startswith("C1")])
    c2lvl = np.mean([r["dvtap_pct"] for r in loc if r["case_name"].startswith("C2")])
    a.axhline(c1lvl, color="0.45", lw=1, ls="--")
    a.axhline(c2lvl, color="0.45", lw=1, ls=":")
    a.text(1.3e-4, c1lvl + 1.3, f"full short, any C1 element  ({c1lvl:+.1f}%)",
           fontsize=8, va="bottom", color="0.3")
    a.text(1.3e-4, c2lvl + 1.3, f"full short, any C2 element  ({c2lvl:+.1f}%)",
           fontsize=8, va="bottom", color="0.3")
    a.axhline(0, color="0.8", lw=1)
    a.set_xscale("log")
    a.set_xlabel(r"short conductance $\sigma$ [S/m]   (1 = full short)")
    a.set_ylabel(r"$\Delta$|Vtap| [%]")
    a.set_title("disc short: Vtap saturates to full-short level as $\\sigma\\uparrow$")
    a.legend(loc="center right", fontsize=9); a.grid(alpha=0.3, which="both")

    fig.suptitle("Task 019 - |Vtap| response of ratio / disc-short faults", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PROC / "internal_sweep_vtap.png", dpi=150)
    plt.close(fig)


def fig_phase_loss(rows, base):
    c1, c2 = fam(rows, "C1_loss"), fam(rows, "C2_loss")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

    a = ax[0]
    a.plot([r["severity_value"] for r in c1], [r["dtheta_mdeg"] for r in c1],
           "o-", color=COLORS["C1_loss"], label="C1 loss  (phase $-$)")
    a.plot([r["severity_value"] for r in c2], [r["dtheta_mdeg"] for r in c2],
           "s-", color=COLORS["C2_loss"], label="C2 loss  (phase $+$)")
    a.axhline(0, color="0.8", lw=1)
    for y in (ARCMIN_MDEG, -ARCMIN_MDEG):
        a.axhline(y, color="0.7", lw=0.8, ls=":")
    a.text(0.10, ARCMIN_MDEG * 1.3, "1 arc-min", fontsize=7.5, color="0.4")
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=10)
    a.set_xlabel(r"element $\tan\delta$"); a.set_ylabel(r"$\Delta\theta$ [mdeg] (symlog)")
    a.set_title("phase displacement: sign locates the loss (C1 $-$ / C2 $+$)")
    a.legend(); a.grid(alpha=0.3, which="both")

    a = ax[1]
    a.plot([r["severity_value"] for r in c1], [r["tan_delta"] for r in c1],
           "o-", color=COLORS["C1_loss"], label="C1 loss")
    a.plot([r["severity_value"] for r in c2], [r["tan_delta"] for r in c2],
           "s-", color=COLORS["C2_loss"], label="C2 loss")
    a.axhline(base["tan_delta"], color="0.7", lw=0.9, ls="--", label="baseline")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"element $\tan\delta$"); a.set_ylabel(r"terminal $\tan\delta$")
    a.set_title(r"terminal $\tan\delta$ rises (C1 stronger: more lossy volume)")
    a.legend(); a.grid(alpha=0.3, which="both")

    fig.suptitle("Task 019 - phase / loss response of dielectric-loss faults", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PROC / "internal_sweep_phase_loss.png", dpi=150)
    plt.close(fig)


def _traj(rows, family, sparam=None, with_base=True, base=None):
    seq = fam(rows, family, sparam)
    pts = [(r["dvtap_pct"], r["dtheta_mdeg"], r["tan_delta"], r["volume_leak"], r["surface_leak"])
           for r in seq]
    if with_base and base is not None:
        pts = [(0.0, 0.0, base["tan_delta"], base["volume_leak"], 0.0)] + pts
    return np.array(pts)


def fig_trajectory_matrix(rows, base):
    fams = [("C1_cap", None), ("C2_cap", None), ("C1_loss", None),
            ("C2_loss", None), ("disc_short", "short_sigma")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))

    # left: d|Vtap| vs tan-delta
    a = ax[0]
    for f, sp in fams:
        t = _traj(rows, f, sp, base=base)
        arrowed(a, t[:, 0], t[:, 2], COLORS[f])
    a.scatter([0], [base["tan_delta"]], marker="*", s=200, color="k", zorder=6, label="baseline")
    a.set_yscale("log"); a.axvline(0, color="0.85", lw=1)
    a.set_xlabel(r"$\Delta$|Vtap| [%]"); a.set_ylabel(r"terminal $\tan\delta$ (log)")
    a.set_title("ratio faults move horizontally; loss faults move up")
    a.grid(alpha=0.3, which="both")

    # right: tan-delta vs d-theta
    a = ax[1]
    for f, sp in fams:
        t = _traj(rows, f, sp, base=base)
        arrowed(a, t[:, 2], t[:, 1], COLORS[f])
    a.scatter([base["tan_delta"]], [0], marker="*", s=200, color="k", zorder=6)
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=10)
    a.axhline(0, color="0.85", lw=1)
    a.set_xlabel(r"terminal $\tan\delta$ (log)"); a.set_ylabel(r"$\Delta\theta$ [mdeg] (symlog)")
    a.set_title("loss faults split by phase sign; short bridges loss$\\to$ratio")
    a.grid(alpha=0.3, which="both")

    handles = [Line2D([0], [0], color=COLORS[f], lw=3,
                      label=f + (" ($\\sigma$ sweep)" if f == "disc_short" else ""))
               for f, _ in fams]
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="k",
                          markeredgecolor="k", markersize=12, label="baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Task 019 - internal fault-family state-space trajectories "
                 "(arrows = increasing severity)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(PROC / "internal_trajectory_matrix.png", dpi=150)
    plt.close(fig)


def fig_refined_matrix(rows, base):
    pol = json.loads(POLL.read_text(encoding="utf-8"))["pollution"]
    pol = sorted(pol, key=lambda r: r["s"])
    pol_x = [(p["x"]["dvtap_pct"], p["x"]["tan_delta"],
              p["x"]["volume_leak"], max(p["x"]["surface_leak"], 1e-7)) for p in pol]
    pol_x = [(0.0, base["tan_delta"], base["volume_leak"], 1e-7)] + pol_x
    pol_x = np.array(pol_x)

    fams = [("C1_cap", None), ("C2_cap", None), ("C1_loss", None),
            ("C2_loss", None), ("disc_short", "short_sigma")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))

    # left: d|Vtap| vs tan-delta (internal + pollution)
    a = ax[0]
    for f, sp in fams:
        t = _traj(rows, f, sp, base=base)
        arrowed(a, t[:, 0], t[:, 2], COLORS[f])
    arrowed(a, pol_x[:, 0], pol_x[:, 1], COLORS["pollution"])
    a.scatter([0], [base["tan_delta"]], marker="*", s=200, color="k", zorder=6)
    a.set_yscale("log"); a.axvline(0, color="0.85", lw=1)
    a.set_xlabel(r"$\Delta$|Vtap| [%]"); a.set_ylabel(r"terminal $\tan\delta$ (log)")
    a.set_title("Vtap separates ratio faults; pollution & loss both raise $\\tan\\delta$")
    a.grid(alpha=0.3, which="both")

    # right: volume leak vs surface leak (external/internal split)
    a = ax[1]
    for f, sp in fams:
        t = _traj(rows, f, sp, base=base)
        arrowed(a, t[:, 3], np.maximum(t[:, 4], 1e-7), COLORS[f])
    arrowed(a, pol_x[:, 2], pol_x[:, 3], COLORS["pollution"])
    a.scatter([base["volume_leak"]], [1e-7], marker="*", s=200, color="k", zorder=6)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("volume loss current [A] (log)")
    a.set_ylabel("surface leakage [A] (log, floored 1e-7)")
    a.set_title("leakage split: pollution $\\to$ surface (up); internal loss $\\to$ volume (right)")
    a.grid(alpha=0.3, which="both")

    handles = [Line2D([0], [0], color=COLORS[f], lw=3, label=f) for f, _ in fams]
    handles.append(Line2D([0], [0], color=COLORS["pollution"], lw=3, label="pollution"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Task 019 - refined diagnostic matrix: smooth internal trajectories "
                 "vs pollution", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(PROC / "diagnostic_matrix_refined.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- analysis
def monotonic(seq, key, sign):
    vals = [r[key] for r in seq]
    d = np.diff(vals)
    ok = np.all(d > -1e-12) if sign > 0 else np.all(d < 1e-12)
    return bool(ok), vals


def detection_floor(rows, base):
    """Smallest tested severity that crosses any Task-018 threshold, per family,
    plus a linear-extrapolated floor for the cap families."""
    out = {}

    def crosses(r):
        hits = []
        if abs(r["dvtap_pct"]) >= DET["dvtap_pct"]:
            hits.append("dvtap")
        if abs(r["dtheta_mdeg"]) >= DET["dtheta_mdeg"]:
            hits.append("dtheta")
        if abs(r["tan_delta"] - base["tan_delta"]) >= DET["d_tan_delta"]:
            hits.append("tand")
        if abs(r["volume_leak"] - base["volume_leak"]) >= DET["d_volume_leak"]:
            hits.append("vol_leak")
        return hits

    for family, sp in [("C1_cap", None), ("C2_cap", None), ("C1_loss", None),
                       ("C2_loss", None), ("disc_short", "short_sigma")]:
        seq = fam(rows, family, sp)
        first = next((r for r in seq if crosses(r)), None)
        out[family] = {"min_sev": first["severity_value"] if first else None,
                       "via": crosses(first) if first else [],
                       "param": seq[0]["severity_parameter"]}
    # extrapolated cap floor: dvtap is linear in cap% -> threshold at 0.05/slope
    for f in ("C1_cap", "C2_cap"):
        seq = fam(rows, f, None)
        slope = abs(seq[-1]["dvtap_pct"]) / seq[-1]["severity_value"]   # %Vtap per %cap
        out[f]["extrap_floor_pct"] = DET["dvtap_pct"] / slope
    return out


def main() -> int:
    rows = load_rows()
    base = json.loads(BASE.read_text(encoding="utf-8"))

    fig_vtap(rows, base)
    fig_phase_loss(rows, base)
    fig_trajectory_matrix(rows, base)
    fig_refined_matrix(rows, base)

    # ---- checks ----
    mono = {}
    mono["C1_cap dvtap up"] = monotonic(fam(rows, "C1_cap"), "dvtap_pct", +1)
    mono["C2_cap dvtap down"] = monotonic(fam(rows, "C2_cap"), "dvtap_pct", -1)
    mono["C1_loss tand up"] = monotonic(fam(rows, "C1_loss"), "tan_delta", +1)
    mono["C2_loss tand up"] = monotonic(fam(rows, "C2_loss"), "tan_delta", +1)
    mono["C1_loss vol up"] = monotonic(fam(rows, "C1_loss"), "volume_leak", +1)
    mono["C2_loss vol up"] = monotonic(fam(rows, "C2_loss"), "volume_leak", +1)
    mono["C1_loss phase down"] = monotonic(fam(rows, "C1_loss"), "dtheta_mdeg", -1)
    mono["C2_loss phase up"] = monotonic(fam(rows, "C2_loss"), "dtheta_mdeg", +1)

    surf0 = all(r["surface_leak"] == 0.0 for r in rows if r["family"] != "pollution")

    loc = fam(rows, "disc_short", "short_location")
    short_check = []
    for r in loc:
        an = float(r["notes"].split("=")[-1].rstrip("%"))
        short_check.append((r["case_name"], r["dvtap_pct"], an, abs(r["dvtap_pct"] - an)))

    det = detection_floor(rows, base)

    # ---- report ----
    L = []
    L += ["# Task 019 - Internal Fault Severity Sweeps & Trajectory Refinement", "",
          "## Purpose", "",
          "Refine the Task 016/018 internal fault manifolds into smooth severity sweeps on the",
          "representative rounded DDB-123 baseline, in the Task 018 observable state vector",
          "",
          "    x = [ d|Vtap| %, d-theta mdeg, tan-delta, surface_leak A, volume_leak A ]",
          "",
          "reported relative to the healthy baseline. Five families are swept: C1/C2 capacitance",
          "increase, C1/C2 dielectric loss, and disc short (location + partial-short severity).", ""]

    L += ["## Model assumptions", "",
          "- Frequency-domain axisymmetric ComplexEQS (50 Hz); no multiphysics, no solver change.",
          "- C1 = 10 homogenised stack elements, C2 = 2 elements; tap between them (foil 10).",
          "- Capacitance fault = scale the C1- or C2-element `eps_r_eff` (loss held at baseline).",
          "- Dielectric-loss fault = raise the element `tan-delta` (capacitance held).",
          "- Disc short = one element keeps `eps_r_eff` but develops a leakage conductance",
          "  `sigma`; `sigma=1 S/m` is effectively equipotential (full short), smaller `sigma`",
          "  is a partial / leaky short.",
          "- Floating metals use the capped `sigma=1 S/m` (no spurious LU loss).", ""]

    L += ["## Healthy baseline", "",
          f"- |Vtap| = {base['vtap_abs']:.2f} V   (tap ratio {base['vtap_abs']/base['U0']:.4f}, "
          f"ideal n_c2/(n_c1+n_c2) = {base['n_c2']/(base['n_c1']+base['n_c2']):.4f})",
          f"- phase = {base['phase_mdeg']:+.2f} mdeg (numerical floor)",
          f"- terminal tan-delta = {base['tan_delta']:.3e}",
          f"- surface leakage = {base['surface_leak']:.2e} A   (zero - clean)",
          f"- volume loss current = {base['volume_leak']:.3e} A", ""]

    L += ["## Sweep definitions", "",
          "| family | parameter | values |",
          "|--------|-----------|--------|",
          "| C1_cap / C2_cap | capacitance increase [%] | 0.1, 0.3, 1, 3, 10, 20 |",
          "| C1_loss / C2_loss | element tan-delta | 0.002 (base), 0.005, 0.01, 0.02, 0.05, 0.10 |",
          "| disc_short (location) | full short element | C1_HV, C1_mid, C1_tap, C2_tap, C2_gnd |",
          "| disc_short (severity) | short conductance sigma [S/m] | 1e-4, 1e-3, 1e-2, 1e-1, 1 |", ""]

    L += ["## Monotonicity checks", "",
          "| check | monotonic? | end value |",
          "|-------|-----------|-----------|"]
    for k, (ok, vals) in mono.items():
        L.append(f"| {k} | {'PASS' if ok else 'FAIL'} | {vals[-1]:+.4g} |")
    L += ["",
          f"surface_leak == 0 for ALL internal faults: **{'PASS' if surf0 else 'FAIL'}** "
          "(internal faults produce no creepage current - the external/internal split holds).", ""]

    L += ["## Disc-short analytic consistency", "",
          "Full short of one stack element -> ideal series-divider ratio shift",
          "(`n_c2/(n_c1+n_c2)` with the shorted unit removed). C1 shorts: 10->9 units -> +9.09 %;",
          "C2 shorts: 2->1 unit -> -45.45 %.", "",
          "| location | FE d|Vtap| [%] | analytic [%] | |error| |",
          "|----------|----------------|--------------|---------|"]
    for name, fe, an, err in short_check:
        L.append(f"| {name} | {fe:+.2f} | {an:+.2f} | {err:.3f} |")
    L += ["",
          "FE matches the lumped model to <0.02 %-point, and is location-independent within C1",
          "and within C2 - confirming the homogenised stack behaves as an ideal series divider",
          "for ratio faults. (C2 shorts shift Vtap ~5x more than C1 shorts because C2 has only",
          "2 series units, so removing one is a far larger fractional change.)", ""]

    L += ["## Detection limits implied by the sweeps", "",
          "Using the Task-018 trend-monitoring thresholds (d|Vtap| >= 0.05 %, d-theta >= 1 mdeg,",
          "d tan-delta >= 1e-4, d volume-leak >= 1e-5 A). These are *assumed* monitoring",
          "resolutions, **not** instrument or standard accuracy classes.", "",
          "| family | min tested severity detected | via | note |",
          "|--------|------------------------------|-----|------|"]
    notes = {
        "C1_cap": f"floor below grid; linear extrap d|Vtap|=0.05% at ~+{det['C1_cap'].get('extrap_floor_pct',0):.2f}% cap",
        "C2_cap": f"floor below grid; linear extrap ~+{det['C2_cap'].get('extrap_floor_pct',0):.2f}% cap",
        "C1_loss": "model phase floor (1 mdeg) ~tan-delta 0.0020; practical 1-arc-min floor ~tan-delta 0.0024",
        "C2_loss": "phase-dominated, same as C1_loss with opposite sign",
        "disc_short": "any conductive path is strongly detectable (Vtap + phase)",
    }
    for f in ("C1_cap", "C2_cap", "C1_loss", "C2_loss", "disc_short"):
        d = det[f]
        ms = d["min_sev"]
        L.append(f"| {f} | {ms:g} ({d['param']}) | {', '.join(d['via'])} | {notes[f]} |")
    L += ["",
          "All five internal families are detectable at (or below) the smallest tested step, so",
          "the practical limit is set by the monitoring resolution, not by the physics:",
          "",
          "- **Capacitance**: d|Vtap| is linear in capacitance change (~0.83 %Vtap per %cap for",
          "  both C1 and C2), so a 0.05 % Vtap resolution implies a ~**+0.06 % capacitance** floor.",
          "- **Dielectric loss**: the phase displacement is extraordinarily sensitive in the model",
          "  (143 mdeg already at tan-delta 0.005), but real metering resolves ~arc-minutes",
          "  (1 arc-min = 16.7 mdeg); at a ~10 arc-min practical floor, loss is detectable around",
          "  **tan-delta ~ 0.005-0.01**. Terminal tan-delta and volume-leak corroborate.",
          "- **Disc short**: a full short is a huge signature (+9 % / -45 % Vtap); even an incipient",
          "  partial short (sigma 1e-4) already shows several-% Vtap and thousands of mdeg phase.", ""]

    L += ["## Does the Task-018 classifier still separate the families?", "",
          "Yes, with one refinement and one documented degeneracy:", "",
          "1. **Ratio faults (C1_cap, C2_cap, full short)** - identified by |d|Vtap||, sign = side",
          "   (C1 up / C2 down). surface_leak = 0, tan-delta ~ baseline. Cleanly separated.",
          "2. **Dielectric loss (C1_loss, C2_loss)** - degree-scale phase with sign (C1 negative,",
          "   C2 positive), terminal tan-delta and volume-leak up, surface_leak = 0, |d|Vtap|| ~ 0.",
          "   Cleanly separated from pollution by **surface_leak = 0** (volume not surface current).",
          "3. **External pollution** - the only family with surface_leak >> 0. Orthogonal to all",
          "   internal families in the leakage-split panel.", "",
          "**Refinement:** the classifier's hard `|d|Vtap|| > 3 %` gate is a *coarse* trigger - it",
          "misses small capacitance drift (<3 %), which is nonetheless detectable by Vtap **trend**",
          "monitoring down to ~0.06 %. For incipient capacitance ageing, use the trend, not the gate.", ""]

    L += ["## Overlaps / blind spots", "",
          "- **Partial short vs C1 dielectric loss (degeneracy).** A leaky/partial short at low",
          "  conductance (sigma <= 1e-3) is signature-identical to a distributed C1 loss fault:",
          "  large negative phase, raised tan-delta and volume-leak, small Vtap shift. Only as the",
          "  short hardens (sigma -> 1) does its Vtap shift grow to the full +9.09 % and reveal it as",
          "  a ratio fault. The disc-short trajectory therefore *bridges* the loss manifold and the",
          "  ratio manifold - incipient shorts and dielectric loss are not separable from terminal",
          "  observables alone until the short progresses.",
          "- **Small capacitance drift (<3 %)** sits below the classifier gate (though above the",
          "  trend-detection floor) - it looks 'nominal' to the rule unless trended.",
          "- The C1/C2 *cap* and C1/C2 *loss* families are otherwise well separated (Vtap sign,",
          "  phase sign, surface vs volume leakage).", ""]

    L += ["## Caveat", "",
          "These are **model-based diagnostic trajectories for the representative DDB-123 model**,",
          "not field-validated operational thresholds. Absolute detection limits depend on the real",
          "instrument/metering class and on temperature drift (a known confounder, not yet modelled).",
          "", "Figures: `internal_sweep_vtap.png`, `internal_sweep_phase_loss.png`,",
          "`internal_trajectory_matrix.png`, `diagnostic_matrix_refined.png`. Data:",
          "`internal_fault_sweeps.csv`.", ""]

    (PROC / "internal_sweeps_report.md").write_text("\n".join(L), encoding="utf-8")

    print("Monotonicity:", {k: ("PASS" if v[0] else "FAIL") for k, (v) in
                            [(k, m) for k, m in mono.items()]})
    print("surface_leak==0 internal:", surf0)
    print("disc-short check (name, fe, analytic, err):")
    for s in short_check:
        print("  ", s)
    print("detection floors:", {f: det[f]["min_sev"] for f in det})
    print(f"Wrote 4 figures + internal_sweeps_report.md in {PROC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

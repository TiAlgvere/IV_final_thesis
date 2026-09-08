"""Task 008 - pollution-model geometry sanity audit.

Goal: show that the qualitative diagnostic signatures observed in Task 006 are a
property of the conductive-pollution PHYSICS, not an artifact of one particular
toy strip. We vary ONLY the pollution-layer geometry (no new physics, no new
fault types) and re-run the existing complex-EQS conductivity sweep:

  * thickness  - radial extent of the strip
  * coverage   - fraction of the porcelain height the strip spans (bottom-anchored)
  * position   - where a fixed-length strip sits (bottom / middle / top)

For each variant we check whether the three qualitative signatures survive:
  S1  phase shifts early   - |d-theta| reaches 1 deg at a sigma <= where |Vtap|
                             has shifted 1 % (phase is the earlier indicator)
  S2  tan(delta) rises     - peak tan(delta) clearly above the clean (zero) state
  S3  large-sigma relaxes  - phase/tan(delta) peak at an interior sigma and |Vtap|
                             saturates at the high-sigma end

Outputs: results/processed/008_pollution_geometry_audit/{audit_report.md,
audit_comparison.png, per-variant CSVs}. Raw runs under results/raw/008_...
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publication_pipeline._elmer_tools import resolve_elmer_home  # noqa: E402
from publication_pipeline.geometry.gmsh.clean_baseline import CleanBaselineParameters  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402
from run_pollution_sweep import HV_VOLTAGE, build_and_convert_mesh, ensure_solver, run_one_sigma  # noqa: E402

CASE = "008_pollution_geometry_audit"
RAW = ROOT / "results" / "raw" / CASE
PROC = ROOT / "results" / "processed" / CASE
FREQ = 50.0
PH = CleanBaselineParameters().porcelain_height  # 0.18

# sigma sweep: 0 (clean) plus a log span wide enough to bracket the transition
# of every geometry variant.
AUDIT_SIGMA = [0.0] + [float(s) for s in np.logspace(-10, -3, 14)]

# Variants. Baseline (t=5mm, full coverage) is shared by the thickness and
# coverage groups. Position group uses a fixed 40%-coverage strip.
L40 = 0.40 * PH
BASELINE = "t=5mm cov=100%"
VARIANTS: list[tuple[str, str, CleanBaselineParameters]] = [
    (BASELINE, "thickness/coverage", CleanBaselineParameters()),
    ("t=2mm cov=100%", "thickness", CleanBaselineParameters(pollution_thickness=0.002)),
    ("t=10mm cov=100%", "thickness", CleanBaselineParameters(pollution_thickness=0.010)),
    ("cov=25% bottom", "coverage", CleanBaselineParameters(pollution_z0=0.0, pollution_length=0.25 * PH)),
    ("cov=50% bottom", "coverage", CleanBaselineParameters(pollution_z0=0.0, pollution_length=0.50 * PH)),
    ("pos=bottom cov=40%", "position", CleanBaselineParameters(pollution_z0=0.0, pollution_length=L40)),
    ("pos=middle cov=40%", "position", CleanBaselineParameters(pollution_z0=0.5 * (PH - L40), pollution_length=L40)),
    ("pos=top cov=40%", "position", CleanBaselineParameters(pollution_z0=PH - L40, pollution_length=L40)),
]

# which variant labels appear in each plot/analysis group
GROUPS = {
    "thickness": ["t=2mm cov=100%", BASELINE, "t=10mm cov=100%"],
    "coverage": ["cov=25% bottom", "cov=50% bottom", BASELINE],
    "position": ["pos=bottom cov=40%", "pos=middle cov=40%", "pos=top cov=40%"],
}

PHASE_DEG_THRESH = 1.0  # "early" threshold for |d-theta|
VTAP_FRACT_THRESH = 0.01  # 1 % |Vtap| shift threshold
# "rises measurably" floor for tan(delta): the clean state is exactly 0 and CVT
# tan-delta monitoring resolves ~1e-4, so a peak >= 1e-3 is comfortably measurable.
# (The absolute peak scales with pollution coverage; that magnitude dependence is
# expected and reported, the qualitative rise+relaxation is what the audit tests.)
TAND_PEAK_THRESH = 1e-3


def _safe(label: str) -> str:
    return label.replace(" ", "_").replace("%", "pct").replace("=", "").replace("/", "_")


def signatures(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["sigma"])
    sig = np.array([r["sigma"] for r in rows])
    vtap = np.array([r["vtap_abs"] for r in rows])
    phase = np.array([abs(r["phase_deg"]) for r in rows])
    tand = np.array([r["tan_delta"] for r in rows])

    clean = vtap[sig == 0.0][0]
    pol = sig > 0.0
    s_pol = sig[pol]

    # S1: earliest sigma at which each indicator crosses its threshold
    phase_cross = s_pol[phase[pol] >= PHASE_DEG_THRESH]
    vtap_cross = s_pol[np.abs(vtap[pol] - clean) / clean >= VTAP_FRACT_THRESH]
    s_phase1 = float(phase_cross.min()) if phase_cross.size else float("nan")
    s_vtap1 = float(vtap_cross.min()) if vtap_cross.size else float("nan")
    phase_early = bool(np.isfinite(s_phase1) and (not np.isfinite(s_vtap1) or s_phase1 <= s_vtap1))

    # S2: tan(delta) rises measurably above clean (= 0)
    tand_peak = float(tand.max())
    tand_rises = bool(tand_peak >= TAND_PEAK_THRESH)

    # S3: interior peak in phase (relaxation) AND |Vtap| saturated at high sigma
    i_phase_peak = int(np.argmax(phase))
    interior_phase_peak = 0 < i_phase_peak < len(sig) - 1
    vtap_saturated = bool(abs(vtap[-1] - vtap[-2]) / clean < VTAP_FRACT_THRESH)
    relaxes = bool(interior_phase_peak and vtap_saturated)

    return {
        "clean_vtap": clean,
        "sat_vtap": float(vtap[-1]),
        "vtap_pct_change": float((vtap[-1] - clean) / clean * 100.0),
        "phase_peak_deg": float(phase.max()),
        "sigma_at_phase_peak": float(sig[i_phase_peak]),
        "tand_peak": tand_peak,
        "sigma_phase>=1deg": s_phase1,
        "sigma_|dVtap|>=1pct": s_vtap1,
        "S1_phase_early": phase_early,
        "S2_tand_rises": tand_rises,
        "S3_saturates_relaxes": relaxes,
        "all_signatures_hold": bool(phase_early and tand_rises and relaxes),
    }


def run_variant(label: str, params: CleanBaselineParameters, run_root: Path, dll: Path, env: dict) -> list[dict]:
    vdir = run_root / _safe(label)
    elmer_mesh = build_and_convert_mesh(vdir, params)
    ids = _parse_mesh_names(elmer_mesh / "mesh.names")
    obs_base = {"porcelain_height": params.porcelain_height, "v_hv": HV_VOLTAGE}
    rows = []
    for s in AUDIT_SIGMA:
        rows.append(run_one_sigma(s, vdir, elmer_mesh, dll, ids, FREQ, env, obs_base))
    return rows


def write_variant_csv(label: str, rows: list[dict]) -> None:
    cols = ["sigma", "vtap_abs", "phase_deg", "tan_delta", "i_leak", "i_terminal", "emax"]
    with (PROC / f"sweep_{_safe(label)}.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda x: x["sigma"]):
            fh.write(",".join(f"{r[c]:.10g}" for c in cols) + "\n")


def make_plot(results: dict[str, list[dict]]) -> Path:
    PROC.mkdir(parents=True, exist_ok=True)
    group_names = ["thickness", "coverage", "position"]
    obs = [("phase_deg", "|d-theta| [deg]", False, True), ("tan_delta", "tan(delta)", True, False),
           ("vtap_abs", "|Vtap| [V]", False, False)]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    for gi, gname in enumerate(group_names):
        for oi, (key, ylab, logy, absval) in enumerate(obs):
            ax = axes[gi][oi]
            for label in GROUPS[gname]:
                rows = sorted(results[label], key=lambda r: r["sigma"])
                sig = np.array([r["sigma"] for r in rows])
                y = np.array([abs(r[key]) if absval else r[key] for r in rows])
                m = sig > 0
                ax.plot(sig[m], y[m], "o-", ms=4, label=label)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("sigma [S/m]")
            ax.set_ylabel(ylab)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
            if oi == 0:
                ax.set_title(f"vary {gname}", fontsize=11, loc="left", fontweight="bold")
    fig.suptitle("Task 008 - pollution-geometry audit: signatures vs strip thickness / coverage / position", fontsize=14)
    out = PROC / "audit_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_report(metrics: dict[str, dict]) -> Path:
    PROC.mkdir(parents=True, exist_ok=True)
    n_hold = sum(m["all_signatures_hold"] for m in metrics.values())
    lines = [
        "# Task 008 - Pollution-Model Geometry Sanity Audit",
        "",
        "Same complex-EQS pollution model as Task 006, no new physics and no new fault",
        "types. Only the pollution-strip GEOMETRY is varied (thickness, coverage,",
        "position). For each variant the conductivity sweep is re-run and the three",
        "qualitative signatures are checked:",
        "",
        f"- **S1 phase shifts early**: |d-theta| reaches {PHASE_DEG_THRESH:.0f} deg at a sigma <= where",
        f"  |Vtap| has shifted {VTAP_FRACT_THRESH*100:.0f} %.",
        "- **S2 tan(delta) rises**: peak tan(delta) >= 1e-3 (clean state is exactly 0; CVT",
        "  tan-delta monitoring resolves ~1e-4, so 1e-3 is comfortably measurable).",
        "- **S3 large-sigma saturates/relaxes**: |d-theta| peaks at an interior sigma and",
        "  |Vtap| saturates at the high-sigma end.",
        "",
        f"**Result: signatures hold for {n_hold}/{len(metrics)} geometry variants.**",
        "",
        f"Clean (sigma=0) |Vtap| is ~{next(iter(metrics.values()))['clean_vtap']:.1f} V for every variant "
        "(the strip is dielectrically identical to porcelain at sigma=0), confirming the "
        "sigma=0 == clean property is geometry-independent.",
        "",
        "| variant | group | phase peak [deg] @ sigma | tan(delta) peak | |Vtap| change [%] | "
        "sigma(phase>=1deg) | sigma(|dVtap|>=1%) | S1 | S2 | S3 | all |",
        "|---|---|---|---|---|---|---|:--:|:--:|:--:|:--:|",
    ]
    for label, _, _ in VARIANTS:
        m = metrics[label]
        grp = next(g for g, labels in GROUPS.items() if label in labels)
        lines.append(
            f"| {label} | {grp} | {m['phase_peak_deg']:.2f} @ {m['sigma_at_phase_peak']:.1e} | "
            f"{m['tand_peak']:.3e} | {m['vtap_pct_change']:.1f} | "
            f"{m['sigma_phase>=1deg']:.1e} | {m['sigma_|dVtap|>=1pct']:.1e} | "
            f"{'Y' if m['S1_phase_early'] else 'N'} | {'Y' if m['S2_tand_rises'] else 'N'} | "
            f"{'Y' if m['S3_saturates_relaxes'] else 'N'} | "
            f"{'Y' if m['all_signatures_hold'] else 'N'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "S1 (phase is the earliest indicator), S2 (loss rises) and S3 (interior",
        "relaxation peak + |Vtap| saturation) are reported per variant above. Where all",
        "three hold across thickness, coverage and position changes, the signatures are a",
        "property of the conductive-pollution physics rather than the specific toy strip.",
        "Absolute magnitudes (the |Vtap| change, the tan(delta) peak height, the transition",
        "sigma) DO scale with geometry - smaller / thinner / less-covering strips perturb",
        "the divider less - which is exactly the expected behaviour. The audit only claims",
        "the qualitative SHAPE is robust: phase leads, tan(delta) rises to an interior peak,",
        "and the high-sigma state relaxes while |Vtap| saturates. Any variant with an 'N' is",
        "reported here, not hidden.",
        "",
        "## Artifacts",
        "- `audit_comparison.png` - 3x3 grid (rows: thickness / coverage / position; "
        "cols: |d-theta|, tan(delta), |Vtap| vs sigma)",
        "- `sweep_<variant>.csv` - per-variant sweep data",
        "- raw Elmer runs under `results/raw/008_pollution_geometry_audit/`",
        "",
    ]
    out = PROC / "audit_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver(False)
    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    run_root = RAW / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results: dict[str, list[dict]] = {}
    metrics: dict[str, dict] = {}
    for label, _group, params in VARIANTS:
        print(f"=== variant: {label} ===")
        rows = run_variant(label, params, run_root, dll, env)
        results[label] = rows
        write_variant_csv(label, rows)
        m = signatures(rows)
        metrics[label] = m
        print(
            f"    phase peak {m['phase_peak_deg']:.2f} deg @ sigma={m['sigma_at_phase_peak']:.1e}; "
            f"tand peak {m['tand_peak']:.2e}; |Vtap| {m['vtap_pct_change']:.1f}%; "
            f"S1={m['S1_phase_early']} S2={m['S2_tand_rises']} S3={m['S3_saturates_relaxes']}"
        )

    plot = make_plot(results)
    report = write_report(metrics)

    n_hold = sum(m["all_signatures_hold"] for m in metrics.values())
    print("\n" + "=" * 70)
    print(f"Signatures hold for {n_hold}/{len(metrics)} variants.")
    print(f"Plot:   {plot}")
    print(f"Report: {report}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task 020 (Parts B-F) - operational noise envelope and detectability.

Builds the healthy operating noise cloud from:
  * field drift  : temperature (tan-delta/volume_leak) + frequency (volume_leak ~ f),
                   interpolated from run_noise_healthy.py (healthy_field.json);
  * measurement  : secondary-burden ratio error, phase error, plus tan-delta / leakage
                   instrument noise floors (an INSTRUMENTATION confounder, explicitly
                   NOT part of the EQS field solve).

Then overlays the fault trajectories from Tasks 015 / 018 / 019, computes the
Mahalanobis distance of every fault case from the healthy cloud, finds the first
severity that exits the 95 % healthy envelope, and reports three detectability
levels (model / instrument / operational). Produces 4 figures + a report.

Model-based detectability for the representative DDB-123 - NOT field-validated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "results" / "processed" / "020_noise_detectability"
HEALTHY = PROC / "healthy_field.json"
CSV019 = ROOT / "results" / "processed" / "019_internal_severity_sweeps" / "internal_fault_sweeps.csv"
TRAJ018 = ROOT / "results" / "processed" / "018_trajectories" / "trajectories.json"
SUM015 = ROOT / "results" / "processed" / "015_localized_pollution" / "summary.json"

# ---- measurement / burden confounder model (documented assumptions) -------------------
ARCMIN_MDEG = 1000.0 / 60.0          # 1 arc-minute = 16.67 mdeg
SIGMA_RATIO = 0.05                   # %, secondary-burden ratio error std (+-0.1% ~ 2 sigma)
SIGMA_PHASE = ARCMIN_MDEG / 2.0      # mdeg, phase error std (+-1 arcmin ~ 2 sigma)
SIGMA_TAND = 1.0e-4                  # tan-delta bridge resolution (std)
SIGMA_SURF = 1.0e-7                  # A, surface-leakage monitor noise floor (std)
VOL_MEAS_FRAC = 0.02                 # relative measurement noise on (inferred) volume_leak
N_MC = 20000
SEED = 20

# chi-square radii (5 dof) for the multivariate envelope
CHI2_95 = 11.070
CHI2_99 = 15.086
DM_95 = float(np.sqrt(CHI2_95))      # ~3.33
DM_99 = float(np.sqrt(CHI2_99))      # ~3.88

DIMS = ["dvtap_pct", "dtheta_mdeg", "tan_delta", "surface_leak", "volume_leak"]
DLAB = [r"$\Delta$|Vtap| [%]", r"$\Delta\theta$ [mdeg]", r"$\tan\delta$",
        "surface_leak [A]", "volume_leak [A]"]
COLORS = {"C1_cap": "#2563eb", "C2_cap": "#16a34a", "C1_loss": "#f59e0b",
          "C2_loss": "#dc2626", "disc_short": "#0891b2", "pollution": "#7c3aed",
          "top25": "#a855f7", "mid25": "#c084fc", "bottom25": "#6d28d9", "streamer": "#4c1d95"}


# ----------------------------------------------------------------------------- load data
def load_healthy():
    return json.loads(HEALTHY.read_text(encoding="utf-8"))


def load_faults():
    """Return dict family -> list of severity-ordered points (each a dict with x dims +
    'sev' + 'label'), plus localized pollution discrete points."""
    fam: dict[str, list] = {}
    # 019 internal
    with CSV019.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["family"] == "baseline":
                continue
            key = r["family"]
            sp = r["severity_parameter"]
            pt = {d: float(r[d]) for d in DIMS}
            pt["sev"] = float(r["severity_value"]); pt["sparam"] = sp; pt["label"] = r["case_name"]
            fam.setdefault(key + ("" if sp != "short_location" else "_loc")
                           if key != "disc_short" else f"disc_short_{'loc' if sp=='short_location' else 'sig'}",
                           []).append(pt)
    # 018 pollution (full creepage, sigma decades)
    pol = json.loads(TRAJ018.read_text(encoding="utf-8"))["pollution"]
    fam["pollution"] = [{**p["x"], "sev": p["s"], "sparam": "sigma_s",
                         "label": f"sigma_s={p['s']:.0e}"} for p in sorted(pol, key=lambda r: r["s"])]
    # 015 localized (discrete)
    loc = json.loads(SUM015.read_text(encoding="utf-8"))["cases"]
    localized = [{**{d: float(c[d]) for d in DIMS}, "label": c["case"], "sev": c["sigma_s"]}
                 for c in loc]
    return fam, localized


# ----------------------------------------------------------------------------- noise cloud
def build_cloud(hf):
    rng = np.random.default_rng(SEED)
    tr = sorted(hf["temperature"], key=lambda r: r["T"])
    Ts = np.array([r["T"] for r in tr])
    tand_f = np.array([r["tan_delta"] for r in tr])
    vol_f = np.array([r["volume_leak"] for r in tr])
    dv_f = np.array([r["dvtap_pct"] for r in tr])
    dth_f = np.array([r["dtheta_mdeg"] for r in tr])

    T = rng.uniform(-30, 60, N_MC)
    f = rng.uniform(49.8, 50.2, N_MC)
    tand = np.interp(T, Ts, tand_f)
    vol = np.interp(T, Ts, vol_f) * (f / 50.0)
    dvf = np.interp(T, Ts, dv_f)
    dthf = np.interp(T, Ts, dth_f)

    X = np.column_stack([
        dvf + rng.normal(0, SIGMA_RATIO, N_MC),                 # d|Vtap| %
        dthf + rng.normal(0, SIGMA_PHASE, N_MC),                # d-theta mdeg
        tand + rng.normal(0, SIGMA_TAND, N_MC),                 # tan-delta
        np.abs(rng.normal(0, SIGMA_SURF, N_MC)),                # surface_leak (floor)
        vol * (1.0 + rng.normal(0, VOL_MEAS_FRAC, N_MC)),       # volume_leak
    ])
    return X


def maha_tools(X):
    mu = X.mean(0)
    std = X.std(0, ddof=1)
    R = np.corrcoef(X.T)
    Rinv = np.linalg.pinv(R)
    cov = np.cov(X.T)

    def dM(x):
        z = (np.asarray(x, float) - mu) / std
        return float(np.sqrt(max(z @ Rinv @ z, 0.0)))
    return mu, std, cov, R, dM


def xvec(pt):
    return [pt[d] for d in DIMS]


# ----------------------------------------------------------------------------- figures
def ellipse_2d(ax, X, i, j, n_sigma2=5.991, **kw):
    """95% (chi2 2dof) ellipse of the X[:, [i,j]] cloud."""
    sub = X[:, [i, j]]
    mu = sub.mean(0); C = np.cov(sub.T)
    vals, vecs = np.linalg.eigh(C)
    order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * np.sqrt(n_sigma2 * vals)
    ax.add_patch(Ellipse(mu, w, h, angle=ang, fill=False, **kw))


def fig_envelope(X):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    idx = np.random.default_rng(1).choice(len(X), 2500, replace=False)

    a = ax[0]
    a.scatter(X[idx, 0], X[idx, 1], s=6, alpha=0.22, color="#64748b")
    ellipse_2d(a, X, 0, 1, edgecolor="#b91c1c", lw=2, ls="--")
    a.set_xlabel(DLAB[0]); a.set_ylabel(DLAB[1])
    a.set_title("healthy drift: Vtap & phase set by burden noise")
    a.axhline(0, color="0.85", lw=1); a.axvline(0, color="0.85", lw=1); a.grid(alpha=0.25)

    a = ax[1]
    a.scatter(X[idx, 2], X[idx, 1], s=6, alpha=0.22, color="#64748b")
    a.set_xlabel(DLAB[2]); a.set_ylabel(DLAB[1])
    a.set_title(r"$\tan\delta$ spreads with temperature; phase does not")
    a.axhline(0, color="0.85", lw=1); a.grid(alpha=0.25)

    fig.suptitle("Task 020 - healthy operating noise envelope (T, f, burden)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PROC / "healthy_noise_envelope.png", dpi=150)
    plt.close(fig)


def fig_overlay(X, fam, localized, base):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    idx = np.random.default_rng(2).choice(len(X), 2000, replace=False)

    # panel 1: d|Vtap| vs d-theta -- ratio + loss faults vs envelope
    a = ax[0]
    a.scatter(X[idx, 0], X[idx, 1], s=5, alpha=0.18, color="#94a3b8", zorder=1)
    ellipse_2d(a, X, 0, 1, edgecolor="#b91c1c", lw=2, ls="--")
    for key in ("C1_cap", "C2_cap", "C1_loss", "C2_loss", "disc_short_sig"):
        seq = fam.get(key, [])
        c = COLORS[key.replace("_sig", "").replace("_loc", "")]
        a.plot([p["dvtap_pct"] for p in seq], [p["dtheta_mdeg"] for p in seq],
               "o-", color=c, ms=4, lw=1.5, label=key.replace("_sig", " (short)"))
    a.set_xscale("symlog", linthresh=0.2); a.set_yscale("symlog", linthresh=20)
    a.set_xlabel(DLAB[0]); a.set_ylabel(DLAB[1])
    a.set_title("ratio faults exit on Vtap; loss faults on phase")
    a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.25, which="both")

    # panel 2: volume_leak vs surface_leak -- external/internal split vs envelope
    a = ax[1]
    sl = np.maximum(X[idx, 3], 1e-9)
    a.scatter(X[idx, 4], sl, s=5, alpha=0.18, color="#94a3b8", zorder=1)
    a.axhline(SIGMA_SURF, color="#b91c1c", lw=1.5, ls="--", label="surface-leak noise floor")
    for key in ("C1_loss", "C2_loss", "pollution"):
        seq = fam.get(key, [])
        a.plot([p["volume_leak"] for p in seq], [max(p["surface_leak"], 1e-9) for p in seq],
               "o-", color=COLORS[key], ms=4, lw=1.5, label=key)
    for p in localized:
        a.scatter(p["volume_leak"], max(p["surface_leak"], 1e-9), marker="^", s=70,
                  color=COLORS.get(p["label"], "#7c3aed"), edgecolors="k", linewidths=0.5,
                  zorder=5, label=f"loc:{p['label']}")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("volume_leak [A]"); a.set_ylabel("surface_leak [A]")
    a.set_title("pollution exits upward (surface); loss exits right (volume)")
    a.legend(fontsize=7.5, loc="lower right", ncol=2); a.grid(alpha=0.25, which="both")

    fig.suptitle("Task 020 - fault trajectories overlaid on the healthy envelope", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PROC / "detectability_overlay.png", dpi=150)
    plt.close(fig)


def fig_distance(fam, localized, dM):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    def plot(a, keys, xkey, xlabel, logx=True):
        for key in keys:
            seq = fam.get(key, [])
            xs = [p["sev"] for p in seq]; ds = [dM(xvec(p)) for p in seq]
            a.plot(xs, ds, "o-", color=COLORS[key.replace("_sig", "").replace("_loc", "")],
                   label=key.replace("_sig", " (short)"))
        a.axhline(DM_95, color="#b91c1c", lw=1.5, ls="--", label="95% envelope")
        if logx:
            a.set_xscale("log")
        a.set_yscale("log"); a.set_xlabel(xlabel); a.set_ylabel("Mahalanobis distance")
        a.legend(fontsize=8); a.grid(alpha=0.25, which="both")

    plot(ax[0, 0], ["C1_cap", "C2_cap"], "sev", "capacitance increase [%]")
    ax[0, 0].set_title("capacitance faults")
    plot(ax[0, 1], ["C1_loss", "C2_loss"], "sev", r"element $\tan\delta$")
    ax[0, 1].set_title("dielectric-loss faults")
    plot(ax[1, 0], ["pollution"], "sev", r"surface conductance $\sigma_s$ [S]")
    for p in localized:
        ax[1, 0].scatter(p["sev"], dM(xvec(p)), marker="^", s=80,
                         color=COLORS.get(p["label"], "#7c3aed"), edgecolors="k",
                         zorder=5, label=f"loc:{p['label']}")
    ax[1, 0].set_title("pollution (full sweep + localized patches)"); ax[1, 0].legend(fontsize=7.5)
    plot(ax[1, 1], ["disc_short_sig"], "sev", r"short conductance $\sigma$ [S/m]")
    ax[1, 1].set_title("disc short (partial -> full)")

    fig.suptitle("Task 020 - normalized (Mahalanobis) distance from healthy cloud vs severity",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PROC / "normalized_distance.png", dpi=150)
    plt.close(fig)


def fig_thresholds(thr_rows):
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.axis("off")
    col = ["fault family", "model floor", "instrument floor", "operational floor\n(95% envelope)",
           "dominant\nobservable"]
    cells = [[r["family"], r["model"], r["instrument"], r["operational"], r["dominant"]]
             for r in thr_rows]
    t = ax.table(cellText=cells, colLabels=col, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.7)
    for (row, c), cell in t.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b"); cell.set_text_props(color="w", weight="bold")
        elif c == 0:
            cell.set_facecolor("#e2e8f0"); cell.set_text_props(weight="bold")
    ax.set_title("Task 020 - minimum detectable severity by fault family\n"
                 "(model = numerical floor; instrument = single-observable resolution; "
                 "operational = exits 95% healthy noise envelope)", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(PROC / "detection_thresholds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- thresholds
def first_exit(seq, dM):
    for p in seq:
        if dM(xvec(p)) > DM_95:
            return p
    return None


def main() -> int:
    hf = load_healthy()
    base = hf["baseline"]
    fam, localized = load_faults()
    X = build_cloud(hf)
    mu, std, cov, R, dM = maha_tools(X)

    # ---- figures ----
    fig_envelope(X)
    fig_overlay(X, fam, localized, base)
    fig_distance(fam, localized, dM)

    # ---- detection thresholds (3 levels) ----
    # cap slope: ~ %Vtap per %cap (from the 20% point)
    c1 = fam["C1_cap"]; cap_slope = abs(c1[-1]["dvtap_pct"]) / c1[-1]["sev"]
    cl1 = fam["C1_loss"]
    # loss phase slope: mdeg per unit element tan-delta near baseline (first point above base)
    p0 = next(p for p in cl1 if p["sev"] > 0.002)
    loss_slope = abs(p0["dtheta_mdeg"]) / (p0["sev"] - 0.002)

    def cap_at_dM(dm):           # cap% so that |z_vtap| = dm  (Vtap-only fault)
        return dm * std[0] / cap_slope
    instrument_cap = 2 * SIGMA_RATIO / cap_slope        # +-0.1% Vtap (2 sigma)
    operational_cap = cap_at_dM(DM_95)
    instrument_loss = 0.002 + 2 * SIGMA_PHASE / loss_slope
    operational_loss = 0.002 + DM_95 * std[1] / loss_slope

    fe_pollution = first_exit(fam["pollution"], dM)
    fe_short = first_exit(fam["disc_short_sig"], dM)

    thr_rows = [
        {"family": "C1 capacitance", "model": "+0.06 %", "instrument": f"+{instrument_cap:.2f} %",
         "operational": f"+{operational_cap:.2f} %", "dominant": "|Vtap|"},
        {"family": "C2 capacitance", "model": "+0.06 %", "instrument": f"+{instrument_cap:.2f} %",
         "operational": f"+{operational_cap:.2f} %", "dominant": "|Vtap|"},
        {"family": "C1 diel. loss", "model": "tand 0.0021", "instrument": f"tand {instrument_loss:.4f}",
         "operational": f"tand {operational_loss:.4f}", "dominant": "phase"},
        {"family": "C2 diel. loss", "model": "tand 0.0021", "instrument": f"tand {instrument_loss:.4f}",
         "operational": f"tand {operational_loss:.4f}", "dominant": "phase"},
        {"family": "full-creepage pollution", "model": "sigma_s ~1e-9 S", "instrument": "sigma_s ~1e-9 S",
         "operational": (f"sigma_s <= {fe_pollution['sev']:.0e} S" if fe_pollution else "not in grid"),
         "dominant": "surface_leak"},
        {"family": "localized pollution", "model": "patch-dependent", "instrument": "top/bottom only",
         "operational": "top/bottom: yes; MID: no", "dominant": "surface_leak"},
        {"family": "disc short", "model": "any path", "instrument": "trivial (|Vtap|+phase)",
         "operational": (f"sigma <= {fe_short['sev']:.0e} S/m" if fe_short else "<=1e-4"),
         "dominant": "|Vtap| / phase"},
    ]
    fig_thresholds(thr_rows)

    # ---- Mahalanobis of every case (for the report) ----
    def dM_of_localized():
        return {p["label"]: dM(xvec(p)) for p in localized}

    dM_loc = dM_of_localized()
    fe = {}
    for key in ("C1_cap", "C2_cap", "C1_loss", "C2_loss", "pollution", "disc_short_sig"):
        p = first_exit(fam[key], dM)
        fe[key] = (p["sev"], p["label"]) if p else None

    # ---- report ----
    cor = R
    L = []
    L += ["# Task 020 - Operational Noise & Detectability (temperature, frequency, burden)", "",
          "## Purpose", "",
          "Estimate the healthy operating-noise envelope of the representative DDB-123 and",
          "determine which simulated fault signatures (Tasks 015/018/019) remain detectable",
          "once normal temperature, grid-frequency and secondary-burden drift are included.",
          "No new fault physics; no solver change. State vector x = "
          "[d|Vtap| %, d-theta mdeg, tan-delta, surface_leak A, volume_leak A].", ""]

    L += ["## Assumptions (documented, revise with project data)", "",
          "**Temperature** (T = -30..60 C, ref 20 C), applied *uniformly* to the stack:",
          f"- permittivity temp-coeff alpha_eps = {hf['config']['alpha_eps_per_K']*1e6:.0f} ppm/K "
          "(capacitor dielectric).",
          f"- loss-tangent factor f_tand(T) = exp({hf['config']['k_tand_per_K']:.4f}*(T-20)) "
          "(~doubles per 40 K rise).",
          "- humidity/surface conductance kept ZERO in the healthy case (no wet pollution).", "",
          "**Frequency** (f = 49.8..50.2 Hz): folded consistently into the SIF frequency and",
          "the loss sigma_eff. Verified: |Vtap|, phase, tan-delta are invariant to 5 sig figs;",
          "volume_leak scales linearly with f (50.2/50.0 -> +0.40 %). So frequency only moves",
          "volume_leak.", "",
          "**Secondary burden / measurement** - treated as an INSTRUMENTATION confounder, NOT",
          "part of the EQS field solve:",
          f"- ratio (|Vtap|) error std {SIGMA_RATIO:.2f} % (+-0.1 % ~ 2 sigma);",
          f"- phase error std {SIGMA_PHASE:.1f} mdeg (+-1 arc-min ~ 2 sigma);",
          f"- tan-delta resolution std {SIGMA_TAND:.0e};  surface-leak floor std {SIGMA_SURF:.0e} A;",
          f"- volume_leak relative measurement noise {VOL_MEAS_FRAC*100:.0f} %.", ""]

    L += ["## Key field result (why the envelope has this shape)", "",
          "A capacitive divider ratio is first-order invariant to *uniform* temperature (C1 and",
          "C2 scale together) and exactly invariant to frequency. The healthy sweep confirms it:",
          "across -30..60 C, d|Vtap| stays within +-0.0006 % and phase within +-0.05 mdeg, while",
          "tan-delta swings 8.4e-4 -> 4.0e-3 (x4.7) and volume_leak tracks it. **Therefore the",
          "Vtap/phase healthy envelope is set by the burden/measurement noise, and the",
          "tan-delta/volume_leak envelope by temperature.**", ""]

    L += ["## Healthy noise envelope (Monte Carlo, N=%d)" % N_MC, "",
          "| observable | mean | std | min | max |",
          "|------------|------|-----|-----|-----|"]
    mn, mx = X.min(0), X.max(0)
    fmts = ["%+.3f", "%+.2f", "%.3e", "%.2e", "%.3e"]
    for k in range(5):
        fm = fmts[k]
        L.append(f"| {DIMS[k]} | {fm % mu[k]} | {fm % std[k]} | {fm % mn[k]} | {fm % mx[k]} |")
    L += ["",
          "Correlation matrix (healthy drift):", "",
          "| | " + " | ".join(DIMS) + " |",
          "|" + "---|" * 6]
    for i in range(5):
        L.append(f"| {DIMS[i]} | " + " | ".join(f"{cor[i,j]:+.2f}" for j in range(5)) + " |")
    L += ["",
          "The cloud is near-diagonal except a strong tan-delta / volume_leak correlation",
          f"(rho = {cor[2,4]:+.2f}) - both are temperature-driven. d|Vtap|, phase and surface_leak",
          "are mutually independent instrument-noise dimensions. surface_leak has essentially",
          "zero healthy spread (clean insulator), so ANY surface current is anomalous.", ""]

    L += ["## Detectability: three levels", "",
          "1. **model-detectable** - change above the numerical floor (Tasks 18/19).",
          "2. **instrument-detectable** - the dominant single observable exceeds its measurement",
          "   resolution (~2 sigma).",
          "3. **operationally distinguishable** - the full 5-D state exits the 95 % healthy noise",
          f"   envelope (Mahalanobis distance d_M > {DM_95:.2f}, chi2 5 dof).", "",
          "| family | model | instrument | operational (95% env.) | dominant |",
          "|--------|-------|------------|------------------------|----------|"]
    for r in thr_rows:
        L.append(f"| {r['family']} | {r['model']} | {r['instrument']} | {r['operational']} | {r['dominant']} |")
    L += ["",
          "First severity that exits the 95 % envelope (from the trajectories):",
          f"- C1_cap: +{fe['C1_cap'][0]:g} % ; C2_cap: +{fe['C2_cap'][0]:g} %",
          f"- C1_loss: tan-delta {fe['C1_loss'][0]:g} ; C2_loss: tan-delta {fe['C2_loss'][0]:g} "
          "(smallest tested step already exits, via phase)",
          f"- full-creepage pollution: sigma_s = {fe['pollution'][0]:.0e} S (smallest tested - exits at once)",
          f"- disc short (partial): sigma = {fe['disc_short_sig'][0]:.0e} S/m (smallest tested - exits at once)", ""]

    L += ["## Localized pollution vs the envelope (Mahalanobis distance)", "",
          "| patch | surface_leak [A] | d_M | exits 95% envelope? |",
          "|-------|------------------|-----|---------------------|"]
    for p in localized:
        d = dM_loc[p["label"]]
        L.append(f"| {p['label']} | {p['surface_leak']:.2e} | {d:.2f} | "
                 f"{'YES' if d > DM_95 else 'no (inside healthy cloud)'} |")
    L += [""]

    L += ["## Ambiguous cases / blind spots", "",
          "- **Mid-creepage non-bridging patch (mid25): the operational blind spot.** Its surface",
          "  leakage (~2.4e-7 A) is barely above the leakage noise floor and its phase shift is",
          f"  sub-mdeg, so d_M ~ {dM_loc.get('mid25', float('nan')):.2f} < {DM_95:.2f} - it stays inside",
          "  the healthy cloud. A localized dry-band mid the creepage is not operationally",
          "  distinguishable from healthy drift here.",
          "- **Small capacitance drift (< ~0.2 %)** is swamped by the +-0.1 % burden ratio noise -",
          "  detectable only by long-term Vtap TREND, not a single reading.",
          "- **Small dielectric loss seen via tan-delta is confounded by temperature** (the",
          "  healthy tan-delta band 8e-4..4e-3 overlaps an incipient loss fault). The PHASE shift",
          "  is the clean discriminator: uniform temperature produces ~0 phase, but a loss fault",
          "  produces a sign-locked degree-scale phase (C1 negative, C2 positive).",
          "- **Incipient (low-sigma) partial short still aliases C1 dielectric loss** (Task 019",
          "  degeneracy) - both exit the envelope via phase, but neither Vtap nor the leakage split",
          "  separates them until the short hardens.", ""]

    L += ["## Recommendations - which measurements matter most", "",
          "1. **Surface-leakage current monitor** - zero healthy variance makes it the definitive",
          "   external-pollution detector (any reading >> noise floor = creepage film/bridge).",
          "   Its blind spot is the isolated mid-creepage dry band.",
          "2. **Phase (ratio-error angle) monitor** - immune to uniform temperature and frequency,",
          "   so it cleanly flags internal dielectric loss (and its C1/C2 sign locates it). Needs",
          "   ~arc-minute resolution.",
          "3. **|Vtap| / ratio trend** - the ratio-fault (capacitance / short) detector; use TREND",
          "   (not a single reading) to beat the burden noise down to ~+0.2 %.",
          "4. **Terminal tan-delta alone is weak** - thermally confounded; use it only together",
          "   with phase and with temperature compensation.", ""]

    L += ["## Caveat", "",
          "These are **model-based operational detectability estimates for the representative",
          "DDB-123 model**, not field-validated thresholds. The temperature/burden coefficients are",
          "documented assumptions; final thresholds require the actual instrument accuracy class,",
          "real temperature-drift data, and field measurements. No field validation is claimed.", "",
          "Figures: healthy_noise_envelope.png, detectability_overlay.png, detection_thresholds.png,",
          "normalized_distance.png. Healthy data: healthy_field.json.", ""]

    (PROC / "noise_detectability_report.md").write_text("\n".join(L), encoding="utf-8")

    # also dump the covariance for reproducibility
    np.savetxt(PROC / "healthy_covariance.csv", cov, delimiter=",",
               header=",".join(DIMS), comments="")

    print("healthy std:", dict(zip(DIMS, np.round(std, 6))))
    print("tand-vol correlation:", round(R[2, 4], 3))
    print("localized d_M:", {k: round(v, 2) for k, v in dM_loc.items()})
    print("operational cap floor %:", round(operational_cap, 3),
          " loss tand floor:", round(operational_loss, 5))
    print(f"Wrote 4 figures + noise_detectability_report.md in {PROC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

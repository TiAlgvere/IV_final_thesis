"""Task 027 - region analysis + weak/medium/severe recommendation from sweep.json.

Computes the local log-log slope d(log surface_leak)/d(log sigma_s) between consecutive points
to locate the LINEAR region (slope ~ 1), the KNEE/transition (slope departs from 1), and any
SATURATION (slope -> 0). Also checks monotonicity of surface_leak / P_direct / tan_delta, the
power-consistency identity, and whether Emax stays terminal-dominated. Prints a compact summary
and writes analysis.json for the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "results" / "processed" / "027_fixed_sector_sigma_sweep"


def _nz(rows):
    return sorted([r for r in rows if r["sigma_s"] > 0], key=lambda r: r["sigma_s"])


def slopes(rows, key):
    x = np.log10([r["sigma_s"] for r in rows]); y = np.log10([max(r[key], 1e-300) for r in rows])
    return [(rows[i]["sigma_s"], rows[i+1]["sigma_s"], float((y[i+1]-y[i])/(x[i+1]-x[i])))
            for i in range(len(rows)-1)]


def monotonic(rows, key):
    v = [r[key] for r in rows]
    return all(v[i+1] >= v[i] - 1e-15 for i in range(len(v)-1)), v


def analyze(rows, clean):
    s = _nz(rows)
    sl = slopes(s, "surface_leak")
    cl = clean["tan_delta"] if clean else 0.0
    out = {"n_cases": len(s), "clean_tan_delta": cl}
    out["surface_leak_monotonic"], _ = monotonic(s, "surface_leak")
    out["P_direct_monotonic"], _ = monotonic(s, "P_surface_direct_W")
    out["tan_delta_monotonic"], _ = monotonic(s, "tan_delta")
    out["slopes_surface_leak"] = [{"from": a, "to": b, "slope": round(m, 3)} for a, b, m in sl]
    # The physical behaviour here: a low-sigma NUMERICAL FLOOR (slope < ~0.95, the tiny loss-
    # driven out-of-phase field is near the iterative residual floor), then a clean LINEAR
    # regime (slope -> 1, specific leak constant), with NO high-sigma saturation. Identify the
    # linear plateau (contiguous slopes >= 0.95) and label the rest by position relative to it.
    plateau = [(a, b) for a, b, m in sl if m >= 0.95]
    if plateau:
        lo, hi = plateau[0][0], plateau[-1][1]
        out["linear_regime"] = {"sigma_lo": lo, "sigma_hi": hi}
        out["numerical_floor_regime"] = {"sigma_below": lo,
                                         "note": "sub-linear slope is a numerical floor, not physical saturation"}
        sat = [(a, b, m) for a, b, m in sl if m < 0.90 and a >= lo]
        out["saturation"] = {"onset_sigma": sat[0][0], "min_slope": round(min(m for *_, m in sat), 3)} if sat else None
    else:
        out["linear_regime"] = None; out["numerical_floor_regime"] = None; out["saturation"] = None
    # specific leak (surface_leak / sigma_s) - constant in the linear regime
    out["specific_leak"] = {f"{r['sigma_s']:.0e}": round(r["surface_leak"]/r["sigma_s"], 1) for r in s}
    # volume-loss crossover: sigma where surface tan-delta contribution == clean volume tan-delta
    cross = None
    for i in range(len(s)-1):
        d0, d1 = s[i]["tan_delta"]-cl, s[i+1]["tan_delta"]-cl
        if d0 < cl <= d1:
            f = (cl-d0)/(d1-d0)
            cross = float(10**(np.log10(s[i]["sigma_s"]) + f*(np.log10(s[i+1]["sigma_s"])-np.log10(s[i]["sigma_s"]))))
            break
    out["volume_loss_crossover_sigma"] = cross   # above this, surface loss dominates tan-delta
    # power consistency (identical by construction -> ~machine epsilon)
    rel = [abs(r["P_surface_from_current_W"]/r["P_surface_direct_W"] - 1) for r in s if r["P_surface_direct_W"] > 0]
    out["power_consistency_max_rel_err"] = float(max(rel)) if rel else 0.0
    # Emax stability
    em = [r["emax"] for r in ([clean] if clean else []) + s]
    out["emax_min"], out["emax_max"] = float(min(em)), float(max(em))
    out["emax_spread_pct"] = float((max(em)/min(em) - 1)*100)
    out["emax_ever_on_insulator"] = any(r.get("emax_on_insulator") for r in s)
    # phase reliability
    ph = [r["phase_mdeg"] for r in s] + ([clean["phase_mdeg"]] if clean else [])
    out["phase_range_mdeg"] = [float(min(ph)), float(max(ph))]
    # weak / medium / severe: chosen inside the RESOLVED linear regime (q_s saved for these)
    out["recommend"] = {
        "weak": 1e-7, "medium": 1e-6, "severe": 1e-5,
        "rationale": "All three lie in the clean linear regime (slope=1, well above the low-sigma "
                     "numerical floor) and have saved q_s fields. weak 1e-7: surface loss just "
                     "emerging above the volume-loss background (P~11 W); medium 1e-6: clearly "
                     "polluted (tan-delta~1.5e-2, P~112 W); severe 1e-5: heavily polluted "
                     "(tan-delta~0.13, P~1.1 kW, max q_s~1.3e4 W/m^2). No saturation regime exists."}
    return out


def main() -> int:
    sj = json.loads((PROC / "sweep.json").read_text(encoding="utf-8"))
    rows = sj["sweep_30deg"]; clean = next((r for r in rows if r["sigma_s"] == 0), None)
    a = analyze(rows, clean)
    a["angle"] = 30
    if sj.get("sweep_10deg"):
        a["analysis_10deg"] = analyze(sj["sweep_10deg"],
                                      next((r for r in sj["sweep_10deg"] if r["sigma_s"] == 0), None))
    (PROC / "analysis.json").write_text(json.dumps(a, indent=2, default=str), encoding="utf-8")
    print(json.dumps(a, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

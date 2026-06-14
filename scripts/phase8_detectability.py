#!/usr/bin/env python
"""Phase 8 -- minimum detection thresholds vs the weather + load noise floor.

For each fault type, sweep severity from 0 upward and find the smallest value at
which the lumped-CVT secondary operating point first leaves the convex-hull noise
envelope (temperature -20..40 degC, grid frequency 49.8..50.2 Hz, secondary
burden 25 VA +/-5 %).  The fault is evaluated at the nominal operating point
(20 degC, 50 Hz, 25 VA); the envelope is the fault-free drift locus.

Internal faults (C1/C2 short, moisture) are instant circuit math; Type A surface
pollution uses 2-D FEA; Type B stork streamer uses 3-D FEA.  Writes
results/phase8_*/detectability_thresholds.csv + summary.json.

    python scripts/phase8_detectability.py
"""
from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ctfem.config import CVTParams, OperatingParams, MaterialParams
from ctfem.common import run_case_2d, run_case_3d
from ctfem.geometry import build_cvt
from ctfem.util import results_dir, dump_json
from ctfem.dashboard_data import (
    FaultPoint, operating_point, secondary_signature, noise_envelope,
    inside_envelope)

BURDEN = 25.0


def _fp(name, category, obs, tap):
    return FaultPoint(name, category, obs.C1_pF, obs.foil_potential_frac[tap - 1],
                      obs.tan_delta, obs.surface_leakage_mA)


def _sig(pt, healthy, u0):
    return secondary_signature(pt, healthy, BURDEN, u0, grid_freq=50.0,
                               ref_burden=BURDEN)


def main() -> None:
    out = results_dir("phase8")
    op = OperatingParams(um_kv=123.0)
    u0 = op.u0
    rows: list[dict] = []

    # --- 2-D baseline + noise floor ------------------------------------------
    cvt = CVTParams(mesh_refinement=1.0)
    matdb = cvt.material_db()
    tap = cvt.tap_disc_index
    msh = os.path.join(out, "cvt2d.msh")
    build_cvt(cvt, msh, verbose=False)
    healthy = _fp("healthy", "healthy", run_case_2d(msh, op, matdb=matdb,
                                                    backend="skfem"), tap)
    hull = noise_envelope(healthy, BURDEN, u0)
    pn = float(np.abs(hull[:, 1]).max())
    rn = float(np.abs(hull[:, 0]).max())
    print(f"[phase8] healthy C={healthy.C_pF:.1f} pF tap={healthy.tap_frac:.4f} "
          f"tand={healthy.tan_delta:.5f}")
    print(f"[phase8] noise floor (T,f,burden +/-5%): phase +/-{pn:.4f} arc-min, "
          f"ratio +/-{rn:.4f} %  ({len(hull)} hull vertices)")

    # --- internal faults (instant circuit) -----------------------------------
    for name, key in [("C1 element short", "c1_short"),
                      ("C2 element short", "c2_short"),
                      ("Internal moisture", "moisture")]:
        thr = None
        for s in np.linspace(0.0, 0.10, 401):
            cp = _sig(operating_point(healthy, **{key: float(s)}), healthy, u0)
            if not inside_envelope(hull, cp.ratio_err_pct, cp.phase_disp_min):
                thr = (s, cp)
                break
        unit = "% base C rise" if key == "moisture" else "% shorted"
        if thr is None:
            rows.append(dict(fault=name, threshold="none <10%", unit=unit,
                             ratio_pct="", phase_min="", note="below floor"))
            print(f"[phase8] {name:<18} not detectable up to 10 {unit}")
        else:
            s, cp = thr
            rows.append(dict(fault=name, threshold=round(s * 100, 3), unit=unit,
                             ratio_pct=round(cp.ratio_err_pct, 4),
                             phase_min=round(cp.phase_disp_min, 4), note=""))
            print(f"[phase8] {name:<18} threshold = {s * 100:5.2f} {unit:<13}"
                  f" (ratio {cp.ratio_err_pct:+.3f} %, phase {cp.phase_disp_min:+.3f} ')")

    # --- Type A surface pollution (2-D FEA) ----------------------------------
    thr = None
    for e in np.linspace(-9.0, -5.0, 25):
        ss = 10.0 ** e
        obs = run_case_2d(msh, op,
                          materials=MaterialParams(surface_conductivity_s=ss),
                          matdb=matdb, backend="skfem")
        cp = _sig(_fp("poll", "surface", obs, tap), healthy, u0)
        if not inside_envelope(hull, cp.ratio_err_pct, cp.phase_disp_min):
            thr = (e, cp, obs)
            break
    if thr is None:
        rows.append(dict(fault="Type A pollution", threshold="none", unit="sigma_s",
                         ratio_pct="", phase_min="", note="below floor"))
        print("[phase8] Type A pollution  not detectable in 1e-9..1e-5 S")
    else:
        e, cp, obs = thr
        rows.append(dict(fault="Type A pollution", threshold=f"1e{e:.2f}",
                         unit="sigma_s [S]", ratio_pct=round(cp.ratio_err_pct, 4),
                         phase_min=round(cp.phase_disp_min, 4),
                         note=f"leak {obs.surface_leakage_mA:.2f} mA; non-monotonic"))
        print(f"[phase8] Type A pollution  threshold = sigma_s 1e{e:.2f} S "
              f"(ratio {cp.ratio_err_pct:+.3f} %, phase {cp.phase_disp_min:+.3f} ', "
              f"leak {obs.surface_leakage_mA:.2f} mA)")

    # --- Type B stork streamer (3-D FEA; 3-D reference + envelope) ------------
    from ctfem.geometry3d import build_cvt_3d
    cvt3 = CVTParams(mesh_refinement=0.18)
    p3 = os.path.join(out, "cvt3d.msh")
    build_cvt_3d(cvt3, p3, verbose=False)
    tap3 = cvt3.tap_disc_index
    healthy3 = _fp("healthy3", "healthy",
                   run_case_3d(p3, op, matdb=matdb, backend="skfem"), tap3)
    hull3 = noise_envelope(healthy3, BURDEN, u0)
    centers = cvt3.shed_centers()
    pitch = (cvt3.stack_z_hi - cvt3.tank_height) / cvt3.n_sheds
    sheet = 0.5 * 0.2e-3
    thr = None
    for n in (4, 8, 12, 16, 20, 24):
        nn = min(n, len(centers))
        z_lo, z_hi = centers[-nn] - 0.6 * pitch, centers[-1] + 0.6 * pitch
        mats = MaterialParams(surface_conductivity_s=1e-14, streamer_sigma_s=sheet,
                              streamer_theta_extent=math.radians(15.0),
                              streamer_z_range=(z_lo, z_hi))
        obs = run_case_3d(p3, op, materials=mats, matdb=matdb, backend="skfem")
        cp = secondary_signature(_fp(f"s{n}", "user", obs, tap3), healthy3, BURDEN,
                                 u0, grid_freq=50.0, ref_burden=BURDEN)
        if not inside_envelope(hull3, cp.ratio_err_pct, cp.phase_disp_min):
            thr = (n, cp)
            break
    if thr is None:
        rows.append(dict(fault="Type B stork streamer", threshold="none <=24",
                         unit="sheds bridged", ratio_pct="", phase_min="",
                         note="terminally invisible; needs leakage/field sensing"))
        print("[phase8] Type B streamer  NOT detectable on secondary terminals "
              "up to 24 sheds (needs leakage-current / field sensing)")
    else:
        n, cp = thr
        rows.append(dict(fault="Type B stork streamer", threshold=n,
                         unit="sheds bridged", ratio_pct=round(cp.ratio_err_pct, 4),
                         phase_min=round(cp.phase_disp_min, 4), note=""))
        print(f"[phase8] Type B streamer  threshold = {n} sheds "
              f"(ratio {cp.ratio_err_pct:+.3f} %, phase {cp.phase_disp_min:+.3f} ')")

    # --- write CSV + JSON + print the table ----------------------------------
    keys = ["fault", "threshold", "unit", "ratio_pct", "phase_min", "note"]
    csvp = os.path.join(out, "detectability_thresholds.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    dump_json({"burden_va": BURDEN, "noise_phase_arcmin": pn,
               "noise_ratio_pct": rn, "thresholds": rows},
              os.path.join(out, "summary.json"))
    print(f"\n[phase8] {'fault':<24}{'threshold':>14} {'unit':<16}"
          f"{'ratio[%]':>10}{'phase[arcmin]':>15}")
    for r in rows:
        print(f"[phase8] {r['fault']:<24}{str(r['threshold']):>14} {r['unit']:<16}"
              f"{str(r['ratio_pct']):>10}{str(r['phase_min']):>15}")
    print(f"[phase8] wrote {csvp}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 8b -- detection threshold vs instrument resolution.

Sweeps the assumed PMU/metering trending resolution (the instrument-noise box
that is Minkowski-summed onto the weather + load convex-hull floor) and records
how the minimum detection threshold for each fault type moves.  Fault signatures
(including the FEA ones) do NOT depend on the instrument, so they are computed
ONCE; only the noise hull is recomputed per grade.  Writes
results/phase8b_*/threshold_vs_instrument.csv.

    python scripts/phase8_instrument_sweep.py
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
from ctfem.geometry3d import build_cvt_3d
from ctfem.util import results_dir, dump_json
from ctfem.dashboard_data import (
    FaultPoint, operating_point, secondary_signature, noise_envelope,
    inside_envelope)

BURDEN = 25.0
# instrument grades = scale x (0.05 % ratio, 0.5 arc-min phase) default
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
BASE_RATIO, BASE_PHASE = 0.05, 0.5


def _fp(name, cat, obs, tap):
    return FaultPoint(name, cat, obs.C1_pF, obs.foil_potential_frac[tap - 1],
                      obs.tan_delta, obs.surface_leakage_mA)


def _sig(pt, ref, u0):
    cp = secondary_signature(pt, ref, BURDEN, u0, grid_freq=50.0, ref_burden=BURDEN)
    return cp.ratio_err_pct, cp.phase_disp_min


def _threshold(traj, hull):
    """First severity in `traj` (list of (sev, ratio, phase)) outside `hull`."""
    for sev, r, p in traj:
        if not inside_envelope(hull, r, p):
            return sev
    return None


def main() -> None:
    out = results_dir("phase8b")
    op = OperatingParams(um_kv=123.0)
    u0 = op.u0

    # --- 2-D baseline + internal / pollution signature trajectories ----------
    cvt = CVTParams(mesh_refinement=1.0)
    matdb = cvt.material_db()
    tap = cvt.tap_disc_index
    msh = os.path.join(out, "cvt2d.msh")
    build_cvt(cvt, msh, verbose=False)
    healthy = _fp("healthy", "healthy",
                  run_case_2d(msh, op, matdb=matdb, backend="skfem"), tap)

    sevs = np.linspace(0.0, 0.10, 401)
    traj = {
        "C1_short_pct": [(float(s) * 100,
                          *_sig(operating_point(healthy, c1_short=float(s)),
                                healthy, u0)) for s in sevs],
        "C2_short_pct": [(float(s) * 100,
                          *_sig(operating_point(healthy, c2_short=float(s)),
                                healthy, u0)) for s in sevs],
        "moisture_pct": [(float(s) * 100,
                          *_sig(operating_point(healthy, moisture=float(s)),
                                healthy, u0)) for s in sevs],
    }
    print(f"[phase8b] healthy C={healthy.C_pF:.1f} pF; computed internal "
          f"trajectories")

    poll = []
    for e in np.arange(-9.0, -4.99, 0.25):
        obs = run_case_2d(msh, op, matdb=matdb, backend="skfem",
                          materials=MaterialParams(surface_conductivity_s=10.0 ** e))
        poll.append((float(e), *_sig(_fp("p", "surface", obs, tap), healthy, u0)))
    traj["pollution_log10sigma"] = poll
    print(f"[phase8b] pollution trajectory: {len(poll)} FEA points")

    # --- 3-D baseline + streamer trajectory ----------------------------------
    cvt3 = CVTParams(mesh_refinement=0.18)
    p3 = os.path.join(out, "cvt3d.msh")
    build_cvt_3d(cvt3, p3, verbose=False)
    tap3 = cvt3.tap_disc_index
    healthy3 = _fp("healthy3", "healthy",
                   run_case_3d(p3, op, matdb=matdb, backend="skfem"), tap3)
    centers = cvt3.shed_centers()
    pitch = (cvt3.stack_z_hi - cvt3.tank_height) / cvt3.n_sheds
    sheet = 0.5 * 0.2e-3
    streamer = []
    for n in (4, 8, 12, 16, 20, 24):
        nn = min(n, len(centers))
        z_lo, z_hi = centers[-nn] - 0.6 * pitch, centers[-1] + 0.6 * pitch
        mats = MaterialParams(surface_conductivity_s=1e-14, streamer_sigma_s=sheet,
                              streamer_theta_extent=math.radians(15.0),
                              streamer_z_range=(z_lo, z_hi))
        obs = run_case_3d(p3, op, materials=mats, matdb=matdb, backend="skfem")
        streamer.append((n, *_sig(_fp("s", "user", obs, tap3), healthy3, u0)))
    print(f"[phase8b] streamer trajectory: {len(streamer)} 3-D points")

    # --- sweep instrument grades (cheap; FEA already done) -------------------
    rows = []
    for k in SCALES:
        ir, ip = BASE_RATIO * k, BASE_PHASE * k
        hull2 = noise_envelope(healthy, BURDEN, u0, instr_ratio_pct=ir,
                               instr_phase_min=ip)
        hull3 = noise_envelope(healthy3, BURDEN, u0, instr_ratio_pct=ir,
                               instr_phase_min=ip)
        pn = float(np.abs(hull2[:, 1]).max())
        rn = float(np.abs(hull2[:, 0]).max())

        def _thr2(key):
            t = _threshold(traj[key], hull2)
            return ">range" if t is None else round(t, 4)

        s_thr = _threshold(streamer, hull3)
        row = {
            "scale_k": k,
            "instr_ratio_pct": ir,
            "instr_phase_min": ip,
            "noise_ratio_pct": round(rn, 4),
            "noise_phase_arcmin": round(pn, 4),
            "C1_short_%": _thr2("C1_short_pct"),
            "C2_short_%": _thr2("C2_short_pct"),
            "moisture_%": _thr2("moisture_pct"),
            "pollution_log10sigma": _thr2("pollution_log10sigma"),
            "streamer_sheds": ">24" if s_thr is None else s_thr,
        }
        rows.append(row)
        print(f"[phase8b] k={k:<4} instr(+/-{ir:.3f}%, +/-{ip:.2f}')  "
              f"C1={row['C1_short_%']}  moist={row['moisture_%']}%  "
              f"poll=1e{row['pollution_log10sigma']}  streamer={row['streamer_sheds']}")

    keys = ["scale_k", "instr_ratio_pct", "instr_phase_min", "noise_ratio_pct",
            "noise_phase_arcmin", "C1_short_%", "C2_short_%", "moisture_%",
            "pollution_log10sigma", "streamer_sheds"]
    csvp = os.path.join(out, "threshold_vs_instrument.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    dump_json({"burden_va": BURDEN, "base_instr": [BASE_RATIO, BASE_PHASE],
               "rows": rows}, os.path.join(out, "summary.json"))
    print(f"[phase8b] wrote {csvp}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Mesh convergence study for the DDB-123 CVT 3-D EQS model.

Holds the physical state FIXED (uniform surface pollution sigma_s = 1e-7 S -- the
phase-displacement relaxation peak, i.e. the most mesh-sensitive operating point)
and re-solves on >= 5 mesh densities (~50k .. ~500k tetrahedra), extracting the
terminal tan delta and the tap phase displacement for each.  Demonstrates that
both observables have physically stabilised (are mesh-independent) at the
standard density.  Writes mesh_convergence.csv + mesh_convergence.png.

    python scripts/mesh_convergence.py
    python scripts/mesh_convergence.py --refines 0.18 0.23 0.28 0.33 0.40
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ctfem.config import CVTParams, OperatingParams, MaterialParams
from ctfem.common import run_case_3d
from ctfem.geometry3d import build_cvt_3d
from ctfem.util import results_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refines", type=float, nargs="+",
                    default=[0.18, 0.23, 0.28, 0.33, 0.40],
                    help="mesh_refinement factors (-> ~50k..~500k tets)")
    ap.add_argument("--sigma-s", type=float, default=1e-7,
                    help="fixed uniform surface conductivity [S]")
    args = ap.parse_args()

    out = results_dir("mesh_convergence")
    op = OperatingParams(um_kv=123.0)
    csvp = os.path.join(out, "mesh_convergence.csv")
    rows: list[dict] = []
    print(f"[mesh] fixed state: sigma_s = {args.sigma_s:.0e} S; sweeping "
          f"{len(args.refines)} mesh densities")

    for ref in args.refines:
        cvt = CVTParams(mesh_refinement=ref)
        matdb = cvt.material_db()
        tap = cvt.tap_disc_index
        msh = os.path.join(out, f"cvt3d_r{ref}.msh")
        t0 = time.time()
        m = build_cvt_3d(cvt, msh, verbose=False)
        t_mesh = time.time() - t0
        try:
            t0 = time.time()
            obs = run_case_3d(
                msh, op, matdb=matdb, backend="skfem",
                materials=MaterialParams(surface_conductivity_s=args.sigma_s))
            t_solve = time.time() - t0
        except Exception as exc:           # largest mesh may exhaust the direct solver
            print(f"[mesh] refine={ref:.3f}  {m.n_triangles} tets -> "
                  f"{type(exc).__name__}; skipping")
            continue
        v = obs.foil_potentials[tap - 1]
        theta = math.degrees(math.atan2(v.imag, v.real)) * 1e3      # mdeg
        rows.append({"refine": ref, "n_tets": m.n_triangles, "n_dofs": m.n_nodes,
                     "tan_delta": round(obs.tan_delta, 6),
                     "tap_theta_mdeg": round(theta, 4),
                     "C_pF": round(obs.C1_pF, 2),
                     "mesh_s": round(t_mesh, 1), "solve_s": round(t_solve, 1)})
        print(f"[mesh] refine={ref:.3f}  {m.n_triangles:>7} tets  "
              f"{m.n_nodes:>6} DOFs  tand={obs.tan_delta:.5f}  "
              f"theta={theta:+.3f} mdeg  ({t_mesh:.0f}+{t_solve:.0f} s)")
        with open(csvp, "w", newline="") as fh:        # incremental, crash-safe
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    if len(rows) < 2:
        print("[mesh] need >= 2 successful solves to plot/conclude")
        return

    n = [r["n_tets"] for r in rows]
    td = [r["tan_delta"] for r in rows]
    th = [r["tap_theta_mdeg"] for r in rows]
    rel_td = abs(td[-1] - td[-2]) / abs(td[-2]) if td[-2] else float("nan")
    rel_th = abs(th[-1] - th[-2]) / abs(th[-2]) if th[-2] else float("nan")
    fig, ax1 = plt.subplots(figsize=(7.4, 4.8))
    l1, = ax1.plot(n, td, "o-", color="tab:red", label=r"tan $\delta$")
    ax1.set_xlabel("number of tetrahedra")
    ax1.set_ylabel(r"terminal  tan $\delta$", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    l2, = ax2.plot(n, th, "s--", color="tab:blue",
                   label=r"tap phase $\Delta\theta$")
    ax2.set_ylabel(r"tap phase displacement  [mdeg]", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title(rf"Mesh convergence ($\sigma_s$ = {args.sigma_s:.0e} S): "
                  r"tan $\delta$ and tap phase stabilise")
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles=[l1, l2], loc="center right", fontsize=9)
    ax1.text(0.03, 0.05, f"finest-pair change ({n[-2]//1000}k -> {n[-1]//1000}k "
             f"tets):\n  tan delta {rel_td:.2%},  tap phase {rel_th:.2%}",
             transform=ax1.transAxes, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mesh_convergence.png"), dpi=140)
    plt.close(fig)

    print(f"[mesh] relative change over the two finest meshes "
          f"({n[-2]} -> {n[-1]} tets): tan delta {rel_td:.2%}, "
          f"tap phase {rel_th:.2%}")
    print(f"[mesh] wrote mesh_convergence.csv + mesh_convergence.png in {out}")


if __name__ == "__main__":
    main()

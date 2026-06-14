#!/usr/bin/env python
"""Phase 9 -- pollution-degradation trajectory in observable state space.

Sweeps the surface conductivity sigma_s and tracks two observables jointly: the
terminal loss tan delta (monotonic) and the tap phase displacement d-theta
(NON-monotonic -- it dives to a trough then recovers).  Treating sigma_s as the
progression parameter gives a parametric (degradation-manifold) trajectory; its
tangent / velocity vector v = (d tan delta / d ln sigma_s, d d-theta / d ln
sigma_s) shows how fast and which way the fault moves.

The sign of d(d-theta)/d sigma_s splits the path into three regimes:
    < 0  capacitive insulation polluting (phase diverging)
    = 0  critical point  (maximum phase displacement -- the relaxation corner)
    > 0  resistive surface short-circuiting dominates (phase recovering)

NB: this is a PARAMETER-space trajectory of steady-state solves, not a dynamical
phase space -- the linear EQS model has no time-domain (ferroresonance/Duffing)
dynamics.  Writes results/phase9_*/statespace.csv + two figures.

    python scripts/phase9_statespace.py
"""
from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ctfem.config import CVTParams, OperatingParams, MaterialParams
from ctfem.common import run_case_2d
from ctfem.geometry import build_cvt
from ctfem.util import results_dir, dump_json
from ctfem.dashboard_data import FaultPoint, secondary_signature

BURDEN = 25.0


def _tap_phase_mdeg(obs, tap):
    v = obs.foil_potentials[tap - 1]
    return math.degrees(math.atan2(v.imag, v.real)) * 1e3


def main() -> None:
    out = results_dir("phase9")
    cvt = CVTParams(mesh_refinement=1.0)
    op = OperatingParams(um_kv=123.0)
    matdb = cvt.material_db()
    u0 = op.u0
    tap = cvt.tap_disc_index
    msh = os.path.join(out, "cvt2d.msh")
    build_cvt(cvt, msh, verbose=False)

    obs_h = run_case_2d(msh, op, matdb=matdb, backend="skfem")
    th0 = _tap_phase_mdeg(obs_h, tap)
    healthy = FaultPoint("healthy", "healthy", obs_h.C1_pF,
                         obs_h.foil_potential_frac[tap - 1], obs_h.tan_delta,
                         obs_h.surface_leakage_mA)

    sig = np.logspace(-9.0, -5.0, 31)
    tand, dth, leak, sratio, sphase = [], [], [], [], []
    for ss in sig:
        obs = run_case_2d(msh, op, matdb=matdb, backend="skfem",
                          materials=MaterialParams(surface_conductivity_s=ss))
        tand.append(obs.tan_delta)
        dth.append(_tap_phase_mdeg(obs, tap) - th0)
        leak.append(obs.surface_leakage_mA)
        fp = FaultPoint("p", "surface", obs.C1_pF,
                        obs.foil_potential_frac[tap - 1], obs.tan_delta,
                        obs.surface_leakage_mA)
        cp = secondary_signature(fp, healthy, BURDEN, u0, grid_freq=50.0,
                                 ref_burden=BURDEN)
        sratio.append(cp.ratio_err_pct)
        sphase.append(cp.phase_disp_min)
    tand = np.array(tand); dth = np.array(dth); leak = np.array(leak)
    sratio = np.array(sratio); sphase = np.array(sphase)

    # --- derivatives wrt ln(sigma_s) (sign == sign of d/d sigma_s) -----------
    lns = np.log(sig)
    d_dth = np.gradient(dth, lns)            # d(d-theta)/d ln sigma
    d_logtand = np.gradient(np.log10(tand), lns)
    speed = np.hypot(d_logtand, d_dth)       # velocity magnitude in state space
    i_crit = int(np.argmin(dth))             # max |phase|: d(d-theta)/dsigma = 0
    s_crit = sig[i_crit]
    print(f"[phase9] healthy tap phase {th0:+.3f} mdeg; swept {len(sig)} sigma_s")
    print(f"[phase9] critical point: sigma_s = {s_crit:.2e} S  "
          f"(d-theta = {dth[i_crit]:+.2f} mdeg, tan_delta = {tand[i_crit]:.4f})")

    # --- figure 1: parameter-space trajectory + velocity vectors -------------
    x = np.log10(tand)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    sc = ax.scatter(x, dth, c=np.log10(sig), cmap="viridis", s=45,
                    zorder=3, edgecolors="k", linewidths=0.4)
    ax.plot(x, dth, "-", color="0.5", lw=1.0, zorder=2)
    q = slice(0, len(sig), 2)
    ax.quiver(x[q], dth[q], d_logtand[q], d_dth[q], color="tab:red", zorder=4,
              angles="xy", width=0.004, alpha=0.8)
    ax.scatter([x[i_crit]], [dth[i_crit]], marker="*", s=360, color="crimson",
               edgecolors="k", zorder=5)
    ax.annotate(f"critical point\nsigma_s={s_crit:.0e} S\n(d(dtheta)/dsigma=0)",
                (x[i_crit], dth[i_crit]), textcoords="offset points",
                xytext=(22, 6), fontsize=8, color="crimson")
    ax.annotate("polluting\n(dtheta/dsigma<0)", (x[3], dth[3]),
                textcoords="offset points", xytext=(10, -26), fontsize=8)
    ax.annotate("resistive short\n(dtheta/dsigma>0)", (x[24], dth[24]),
                textcoords="offset points", xytext=(6, -34), fontsize=8)
    ax.text(0.02, 0.03, "red arrows = velocity (tangent, per ln sigma_s)",
            transform=ax.transAxes, fontsize=8, color="tab:red", va="bottom")
    ax.set_xlabel(r"terminal loss   $\log_{10}\tan\delta$")
    ax.set_ylabel(r"tap phase displacement   $\Delta\theta$  [mdeg]")
    ax.set_title("Pollution degradation: (loss, phase) parameter-space trajectory")
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\log_{10}\sigma_s$  [S]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "statespace_trajectory.png"), dpi=140)
    plt.close(fig)

    # --- figure 2: phase derivative + the three regimes ----------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhline(0.0, color="0.6", lw=0.9)
    ax.axvline(s_crit, color="crimson", ls="--", lw=1.0)
    ax.fill_between(sig, d_dth, 0, where=(sig < s_crit), color="tab:blue",
                    alpha=0.18, label="polluting (slope < 0)")
    ax.fill_between(sig, d_dth, 0, where=(sig >= s_crit), color="tab:orange",
                    alpha=0.18, label="resistive short (slope > 0)")
    ax.semilogx(sig, d_dth, "o-", color="tab:blue", ms=4)
    ax.annotate(f"critical\n{s_crit:.0e} S", (s_crit, 0.0),
                textcoords="offset points", xytext=(6, 10), fontsize=8,
                color="crimson")
    ax.set_xlabel(r"surface conductivity   $\sigma_s$  [S]")
    ax.set_ylabel(r"$\mathrm{d}\,\Delta\theta / \mathrm{d}\ln\sigma_s$  [mdeg]")
    ax.set_title("Phase-displacement velocity: sign splits the degradation regimes")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phase_derivative.png"), dpi=140)
    plt.close(fig)

    # --- CSV + JSON ----------------------------------------------------------
    csvp = os.path.join(out, "statespace.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["surface_sigma_S", "tan_delta", "tap_dtheta_mdeg",
                    "dtheta_dlnsigma", "dlog10tand_dlnsigma", "speed",
                    "sec_ratio_pct", "sec_phase_arcmin", "leakage_mA"])
        for i in range(len(sig)):
            w.writerow([f"{sig[i]:.4e}", f"{tand[i]:.6f}", f"{dth[i]:.4f}",
                        f"{d_dth[i]:.4f}", f"{d_logtand[i]:.4f}",
                        f"{speed[i]:.4f}", f"{sratio[i]:.4f}",
                        f"{sphase[i]:.4f}", f"{leak[i]:.4f}"])
    dump_json({"critical_sigma_s": float(s_crit),
               "critical_dtheta_mdeg": float(dth[i_crit]),
               "critical_tan_delta": float(tand[i_crit])},
              os.path.join(out, "summary.json"))
    print(f"[phase9] wrote statespace.csv + statespace_trajectory.png + "
          f"phase_derivative.png  in {out}")


if __name__ == "__main__":
    main()

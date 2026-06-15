#!/usr/bin/env python
"""Phase 7 -- Arteche DDB-123 capacitive voltage transformer (110 kV Estonia).

Models the DDB-123 CVT from the ARTECHE DDB/DFK datasheet: a stack of series
capacitor elements (homogenized wound paper-film cans between electrode discs)
inside a porcelain insulator on top of the grounded EMU tank.  Validation
anchors from the datasheet: H = 1830 mm, base A = 450 mm, and above all the
rated standard capacitance C = 5600 pF, which the FEM admittance must
reproduce.  The divider tap (C1/C2 joint) potential is an observable: the
interior electrode discs reuse the foil-ladder machinery, so
``foil_potential_frac`` IS the divider ladder.

Runs entirely on Windows (scikit-fem backend).  Outputs in results/phase7_*/:

  cvt2d.msh / cvt_field.png / cvt_ladder.png   -- mesh, potential, divider ladder
  defect_comparison.csv / summary.json          -- healthy vs defect signatures
  cvt3d.msh / cvt3d_result.vtu / view3d.png     -- with --solve-3d

Examples:
    python scripts/phase7_cvt.py                       # 2-D healthy + defect set
    python scripts/phase7_cvt.py --solve-3d --show     # + 3-D interactive window
    python scripts/phase7_cvt.py --kind element_aging --severity 0.5
    python scripts/phase7_cvt.py --kind water_ingress --z-center 0.75  # wet oil
    python scripts/phase7_cvt.py --phase-sweep         # tap phase vs pollution
    python scripts/phase7_cvt.py --circuit             # secondary ratio/phase err
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import asdict

from ctfem.config import CVTParams, OperatingParams, DefectSpec, MaterialParams
from ctfem.common import run_case_2d, run_case_3d
from ctfem.cvt_circuit import CVTCircuit, solve_from_fea
from ctfem.geometry import build_cvt
from ctfem.util import results_dir, dump_json


def _tap_phase_deg(obs, cvt: CVTParams, op: OperatingParams) -> float:
    """Phase angle [deg] of the complex tap potential vs the (real) HV drive.

    The C1/C2 intermediate-voltage tap sits at interior disc `tap_disc_index`.
    Its complex potential phi = phi_re + j phi_im picks up a phase lag as
    resistive leakage (surface pollution / internal defects) develops -- this is
    the CVT phase displacement tracked for online fault prediction.  Computed as
    theta = arctan2(phi_im, phi_re); the HV electrode is driven real (phase 0),
    so theta is the displacement of the tap relative to the primary.
    """
    tap = cvt.tap_disc_index
    if obs.foil_potentials:           # complex ladder (skfem backend)
        v = obs.foil_potentials[tap - 1]
    else:                             # backend without complex ladder -> phase 0
        v = complex(obs.foil_potential_frac[tap - 1] * op.u0)
    return float(np.degrees(np.arctan2(v.imag, v.real)))


def _report(tag: str, obs, cvt: CVTParams, op: OperatingParams) -> dict:
    """Print one solve and return its summary row."""
    tap = cvt.tap_disc_index
    tap_frac = obs.foil_potential_frac[tap - 1]
    ph_mdeg = _tap_phase_deg(obs, cvt, op) * 1e3      # milli-degrees
    row = {
        "case": tag,
        "C_pF": obs.C1_pF,
        "tan_delta": obs.tan_delta,
        "tap_frac": tap_frac,
        "tap_kV": tap_frac * op.u0 / 1e3,
        "tap_phase_mdeg": ph_mdeg,
        "leakage_mA": obs.surface_leakage_mA,
    }
    print(f"[phase7] {tag:<22} C={obs.C1_pF:8.1f} pF  tand={obs.tan_delta:.5f}"
          f"  tap={tap_frac:7.4f} ({row['tap_kV']:5.2f} kV)"
          f"  ph={ph_mdeg:+8.3f} mdeg")
    return row


# Surface-pollution sweep points (sheet conductance sigma_s [S]) with the
# Estonian / IEC 60815 pollution-class labels the user mapped them to.  Two
# points per decade so the (non-monotonic) tap-phase peak is resolved.
_POLLUTION_SWEEP = [
    (1e-9, "clean / rural inland"),
    (3e-9, ""),
    (1e-8, "light"),
    (3e-8, ""),
    (1e-7, "medium (Elering / IEC 'Medium')"),
    (3e-7, ""),
    (1e-6, "medium-heavy"),
    (3e-6, ""),
    (1e-5, "heavy (coastal / winter salt)"),
]


def _phase_displacement_sweep(msh: str, op: OperatingParams, matdb,
                              cvt: CVTParams, out: str) -> list[dict]:
    """Sweep the exterior surface conductivity and tabulate the tap phase
    displacement Delta-theta vs a clean baseline -- the CVT fault-prediction
    curve.  Writes phase_displacement.csv + .png and returns the rows.
    """
    import csv
    tap = cvt.tap_disc_index

    def _solve(sigma_s: float):
        return run_case_2d(
            msh, op, materials=MaterialParams(surface_conductivity_s=sigma_s),
            matdb=matdb, backend="skfem")

    base = _solve(1.0e-14)
    if not base.foil_potentials:
        raise SystemExit("[phase7] phase sweep needs the skfem backend "
                         "(complex tap potential)")
    th0 = _tap_phase_deg(base, cvt, op)
    print(f"[phase7] phase-displacement sweep at the C1/C2 tap (disc {tap}); "
          f"clean baseline theta = {th0 * 1e3:+.3f} mdeg")
    print(f"[phase7]   {'sigma_s[S]':>10} {'|Vtap|kV':>9} {'dtheta[mdeg]':>12} "
          f"{'tan_delta':>10} {'leak[mA]':>9}  class")
    rows: list[dict] = []
    for sigma_s, label in _POLLUTION_SWEEP:
        obs = _solve(sigma_s)
        v = obs.foil_potentials[tap - 1]
        th = _tap_phase_deg(obs, cvt, op)
        dth = th - th0
        rows.append({
            "surface_sigma_S": sigma_s, "tap_kV": abs(v) / 1e3,
            "theta_mdeg": th * 1e3, "dtheta_mdeg": dth * 1e3,
            "dtheta_urad": math.radians(dth) * 1e6,
            "tan_delta": obs.tan_delta, "leakage_mA": obs.surface_leakage_mA,
            "pollution_class": label,
        })
        print(f"[phase7]   {sigma_s:10.1e} {abs(v) / 1e3:9.3f} "
              f"{dth * 1e3:+12.3f} {obs.tan_delta:10.5f} "
              f"{obs.surface_leakage_mA:9.4f}  {label}")

    csvp = os.path.join(out, "phase_displacement.csv")
    with open(csvp, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wcsv.writeheader()
        wcsv.writerows(rows)
    _plot_phase_sweep(rows, os.path.join(out, "phase_displacement.png"))

    peak = max(rows, key=lambda r: abs(r["dtheta_mdeg"]))
    print(f"[phase7] wrote phase_displacement.csv + .png")
    print(f"[phase7] NOTE: |dtheta| peaks at sigma_s={peak['surface_sigma_S']:.0e} S "
          f"({peak['dtheta_mdeg']:+.3f} mdeg) then declines -- the tap phase shift "
          f"is NON-monotonic (RC relaxation).  tan_delta stays monotonic, so use "
          f"both to resolve severity past the peak.")
    return rows


def _plot_phase_sweep(rows: list[dict], png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ss = [r["surface_sigma_S"] for r in rows]
    dth = [r["dtheta_mdeg"] for r in rows]
    td = [r["tan_delta"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.semilogx(ss, dth, "o-", color="tab:blue")
    ax1.set_xlabel(r"surface conductivity  $\sigma_s$  [S]")
    ax1.set_ylabel(r"tap phase displacement  $\Delta\theta$  [mdeg]",
                   color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(0.0, color="0.7", lw=0.8)
    ax2 = ax1.twinx()
    ax2.loglog(ss, td, "s--", color="tab:red")
    ax2.set_ylabel(r"terminal  $\tan\delta$", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("CVT fault prediction: tap phase displacement vs surface pollution")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)


def _streamer_compare(args, op: OperatingParams, matdb, out: str) -> dict:
    """Type B 'bird streamer': a localized 3-D surface streak.  Solve clean,
    uniform (Type A) and localized-streamer (Type B) cases on the same 3-D mesh
    and compare the tap phase displacement, to test whether the localized
    biological transient is distinguishable from uniform weather pollution.

    Exports a VTU (with the phase-angle field) so the streak can be seen in 3-D.
    """
    from ctfem.geometry3d import build_cvt_3d
    from ctfem.viz3d import defect_indicator, export_vtu, screenshot, \
        show_interactive, efield_screenshot, peak_efield_air

    cvt3 = CVTParams(n_elements=args.n_elements, n_elements_c2=args.n_c2,
                     rated_capacitance_pF=args.rated_pf,
                     mesh_refinement=args.refine_3d)
    p3 = os.path.join(out, "cvt3d.msh")
    t0 = time.time()
    m3 = build_cvt_3d(cvt3, p3, verbose=False)
    print(f"[phase7] 3-D mesh: {m3.n_triangles} tets ({time.time() - t0:.0f} s)")
    tap = cvt3.tap_disc_index

    # streak geometry: a continuous vertical band over the upper N sheds, one side
    sheet = args.streamer_sigma * (args.streamer_film_mm * 1e-3)   # S/m * m -> S
    centers = cvt3.shed_centers()                                  # bottom -> top
    n = max(1, min(args.streamer_sheds, len(centers)))
    pitch = (cvt3.stack_z_hi - cvt3.tank_height) / cvt3.n_sheds
    z_lo, z_hi = centers[-n] - 0.6 * pitch, centers[-1] + 0.6 * pitch
    width = math.radians(args.streamer_width_deg)
    frac = args.streamer_width_deg / 360.0
    print(f"[phase7] streamer: sigma={args.streamer_sigma} S/m x "
          f"{args.streamer_film_mm} mm = {sheet:.2e} S sheet, "
          f"{args.streamer_width_deg:.0f} deg ({frac:.1%} of ring), "
          f"z=[{z_lo:.2f},{z_hi:.2f}] m (upper {n} sheds)")

    def _solve(mats):
        return run_case_3d(p3, op, materials=mats, matdb=matdb, backend="skfem",
                           return_solution=True)

    def _ph(obs):
        v = obs.foil_potentials[tap - 1]
        return math.degrees(math.atan2(v.imag, v.real)) * 1e3        # mdeg

    obs0, _, _ = _solve(MaterialParams(surface_conductivity_s=1e-14))
    obsA, _, _ = _solve(MaterialParams(surface_conductivity_s=sheet))
    matsB = MaterialParams(surface_conductivity_s=1e-14, streamer_sigma_s=sheet,
                           streamer_theta_center=0.0, streamer_theta_extent=width,
                           streamer_z_range=(z_lo, z_hi))
    obsB, solB, _ = _solve(matsB)

    th0 = _ph(obs0)
    print(f"[phase7]   {'case':<26}{'tan_delta':>11}{'dtheta[mdeg]':>13}"
          f"{'leak[mA]':>10}")
    cases = [("clean baseline", obs0),
             (f"uniform Type A ({sheet:.0e} S ring)", obsA),
             (f"streamer Type B ({args.streamer_width_deg:.0f} deg)", obsB)]
    for name, obs in cases:
        print(f"[phase7]   {name:<26}{obs.tan_delta:11.5f}"
              f"{_ph(obs) - th0:13.3f}{obs.surface_leakage_mA:10.4f}")
    print(f"[phase7] DISCRIMINATOR: same local sheet sigma, but the streak wets "
          f"only {frac:.1%} of the ring -> far less terminal tan_delta than the "
          f"uniform layer, yet a comparable LOCAL phase twist (see the VTU).")

    indicator = defect_indicator(solB, DefectSpec(kind="none"),
                                 matdb=matdb, operating=op)
    vtu = os.path.join(out, "cvt3d_streamer.vtu")
    export_vtu(solB, indicator, vtu)
    # default the streamer view to the phase field (where the streak shows up);
    # honour an explicit --field E/phi otherwise.  In the window 't' toggles.
    field_key = ("phi_phase_mrad" if args.field == "phi"
                 else {"phase": "phi_phase_mrad", "E": "E_kV_mm"}[args.field])
    screenshot(vtu, os.path.join(out, "view3d_streamer.png"), field_key)
    print(f"[phase7] wrote {vtu} + view3d_streamer.png "
          f"(view the 'phi_phase_mrad' field to see the streak)")

    # --- electric-field magnitude (dielectric-breakdown view) ----------------
    # |E| governs air breakdown (~3 kV/mm), not the phase angle.  Report the peak
    # air-gap field overall and specifically at the clean sheds BELOW the streak
    # (where the wet/dry triple junction concentrates the field), and render a
    # |E| view of that region.
    e_all, loc_all = peak_efield_air(solB, r_max=0.40)
    e_below, loc_below = peak_efield_air(solB, r_max=0.40,
                                         z_range=(cvt3.tank_height, z_lo))
    print(f"[phase7] peak |E| in air gap: {e_all:.3f} kV/mm overall"
          + (f" (r={loc_all[0]:.3f}, z={loc_all[1]:.3f} m)" if loc_all else "")
          + f";  {e_below:.3f} kV/mm at the clean sheds below the streak"
          + (f" (r={loc_below[0]:.3f}, z={loc_below[1]:.3f} m)" if loc_below else "")
          + f"  [air strength ~3 kV/mm]")
    efield_screenshot(vtu, os.path.join(out, "view3d_efield_streamer.png"),
                      z_bot=cvt3.tank_height, z_top=z_hi + 0.05)  # auto colour scale
    print(f"[phase7] wrote view3d_efield_streamer.png (|E| at the bottom sheds)")
    if args.show:
        show_interactive(vtu, field_key)

    return {"sheet_S": sheet, "width_deg": args.streamer_width_deg,
            "z_range": [z_lo, z_hi], "n_sheds": n, "theta_clean_mdeg": th0,
            "peak_E_air_kVmm": e_all, "peak_E_below_streak_kVmm": e_below,
            "uniform": {"C_pF": obsA.C1_pF, "tan_delta": obsA.tan_delta,
                        "dtheta_mdeg": _ph(obsA) - th0,
                        "leakage_mA": obsA.surface_leakage_mA},
            "streamer": {"C_pF": obsB.C1_pF, "tan_delta": obsB.tan_delta,
                         "dtheta_mdeg": _ph(obsB) - th0,
                         "leakage_mA": obsB.surface_leakage_mA}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-elements", type=int, default=12)
    ap.add_argument("--n-c2", type=int, default=2,
                    help="elements below the intermediate-voltage tap")
    ap.add_argument("--rated-pf", type=float, default=5600.0,
                    help="datasheet standard capacitance [pF]")
    ap.add_argument("--refine", type=float, default=1.0)
    ap.add_argument("--backend", default="skfem")
    # custom single defect (replaces the default demo set)
    ap.add_argument("--kind", default=None,
                    choices=["shorted_element", "element_aging",
                             "oil_contamination", "water_ingress"])
    ap.add_argument("--index", type=int, default=5)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--theta-deg", type=float, default=0.0)
    ap.add_argument("--defect-wedge-deg", type=float, default=360.0)
    # water_ingress pocket: location/size in the oil column (z in [0.55, 1.61] m
    # for the default DDB-123) and its endpoint material at severity 1.
    ap.add_argument("--z-center", type=float, default=0.75,
                    help="axial centre of the water pocket [m] (water_ingress)")
    ap.add_argument("--extent", type=float, default=0.15,
                    help="axial half-height of the water pocket [m]")
    ap.add_argument("--defect-eps-r", type=float, default=80.0,
                    help="defect relative permittivity at severity 1 (water=80)")
    ap.add_argument("--defect-sigma", type=float, default=1.0e-4,
                    help="defect conductivity [S/m] at severity 1 (water=1e-4)")
    # surface conductivity (pollution layer) on the exterior insulator surface;
    # sweep this for environmental-tracking / pollution-flashover studies.
    ap.add_argument("--surface-sigma", type=float, default=1.0e-14,
                    help="insulator surface sheet conductance sigma_s [S] "
                         "(clean~1e-14; wetted/polluted ~1e-9..1e-6)")
    ap.add_argument("--phase-sweep", action="store_true",
                    help="sweep --surface-sigma and tabulate the tap phase "
                         "displacement (CVT fault-prediction curve), then exit")
    # lumped CVT circuit (compensating reactor + IVT + burden): turns the FEA
    # divider parameters into the measurable secondary ratio error / phase disp.
    ap.add_argument("--circuit", action="store_true",
                    help="also report the lumped-CVT secondary ratio error and "
                         "phase displacement (reactor tuned to the healthy C_eq)")
    ap.add_argument("--burden-va", type=float, default=25.0,
                    help="secondary burden [VA] for the --circuit model")
    # 3-D
    ap.add_argument("--solve-3d", action="store_true",
                    help="also build+solve the full 3-D revolve")
    ap.add_argument("--refine-3d", type=float, default=0.35)
    ap.add_argument("--field", default="phi", choices=["phi", "phase", "E"],
                    help="3-D scalar to colour: phi (|V|), phase (phase angle), "
                         "E (field). The viewer's 't' key toggles between them.")
    # Type B 'bird streamer' (localized 3-D surface fault)
    ap.add_argument("--streamer", action="store_true",
                    help="localized 3-D surface streak; compares its tap phase "
                         "signature against a uniform (Type A) layer, then exits")
    ap.add_argument("--streamer-sigma", type=float, default=0.5,
                    help="streamer bulk conductivity [S/m]")
    ap.add_argument("--streamer-film-mm", type=float, default=0.2,
                    help="streamer wet-film thickness [mm]; sheet = sigma*film")
    ap.add_argument("--streamer-width-deg", type=float, default=15.0,
                    help="streamer azimuthal width [deg]")
    ap.add_argument("--streamer-sheds", type=int, default=4,
                    help="number of upper sheds the vertical streak bridges")
    ap.add_argument("--show", action="store_true",
                    help="interactive 3-D window (implies --solve-3d)")
    args = ap.parse_args()
    if args.show:
        args.solve_3d = True

    out = results_dir("phase7")
    cvt = CVTParams(n_elements=args.n_elements, n_elements_c2=args.n_c2,
                    rated_capacitance_pF=args.rated_pf,
                    mesh_refinement=args.refine)
    op = OperatingParams(um_kv=123.0)          # DDB-123: Um = 123 kV
    matdb = cvt.material_db()
    # physics-region assignment + the exterior-surface conductivity (pollution).
    mats = MaterialParams(surface_conductivity_s=args.surface_sigma)

    creep = cvt.creepage_distance() * 1e3
    print(f"[phase7] DDB-123 model: H={cvt.total_height} m, "
          f"{cvt.n_elements} elements ({cvt.n_elements - cvt.n_elements_c2}+"
          f"{cvt.n_elements_c2} C1/C2), eps_r_eff={cvt.element_epsr_eff():.0f}, "
          f"U0={op.u0 / 1e3:.1f} kV")
    print(f"[phase7]   {cvt.n_sheds} sheds -> creepage {creep:.0f} mm "
          f"(datasheet {cvt.rated_creepage_mm:.0f} mm, "
          f"{(creep - cvt.rated_creepage_mm) / cvt.rated_creepage_mm:+.1%})")
    if args.surface_sigma > 1e-12:
        print(f"[phase7]   pollution layer: surface sigma_s={args.surface_sigma:.1e} S "
              f"on insulator_surface")

    # --- 2-D mesh + healthy solve -------------------------------------------
    msh = os.path.join(out, "cvt2d.msh")
    mres = build_cvt(cvt, msh, verbose=False)
    print(f"[phase7] 2-D mesh: {mres.n_triangles} triangles")

    obs_h, sol_h, _ = run_case_2d(msh, op, materials=mats, matdb=matdb,
                                  backend=args.backend, return_solution=True)
    rows = [_report("healthy", obs_h, cvt, op)]

    c_rated = cvt.rated_capacitance_pF
    dev = (obs_h.C1_pF - c_rated) / c_rated
    k_nom = cvt.divider_ratio_nominal()
    tap_h = rows[0]["tap_frac"]
    print(f"[phase7]   vs nameplate:  C {obs_h.C1_pF:.1f} pF / rated "
          f"{c_rated:.0f} pF ({dev:+.2%});  tap {tap_h:.4f} / nominal "
          f"{k_nom:.4f} ({(tap_h - k_nom) / k_nom:+.2%})")

    # --- phase-displacement fault-prediction sweep (alternate mode) -----------
    if args.phase_sweep:
        sweep = _phase_displacement_sweep(msh, op, matdb, cvt, out)
        dump_json({"cvt": asdict(cvt), "mode": "phase_sweep",
                   "tap_disc_index": cvt.tap_disc_index, "sweep": sweep},
                  os.path.join(out, "summary.json"))
        print(f"[phase7] outputs in {out}")
        return

    # --- Type B 'bird streamer' localized 3-D fault (alternate mode) -----------
    if args.streamer:
        res = _streamer_compare(args, op, matdb, out)
        dump_json({"cvt": asdict(cvt), "mode": "streamer", "streamer": res},
                  os.path.join(out, "summary.json"))
        print(f"[phase7] outputs in {out}")
        return

    # --- optional lumped-CVT secondary response (reactor + burden) ------------
    circuit = h_resp = None
    if args.circuit:
        circuit = CVTCircuit(burden_va=args.burden_va)
        h_resp = solve_from_fea(circuit, obs_h.C1_pF * 1e-12, tap_h,
                                obs_h.tan_delta, op.u0, tune=True)
        rows[0]["sec_phase_min"] = h_resp.phase_min
        rows[0]["sec_dphase_min"] = 0.0
        rows[0]["sec_ratio_err_pct"] = 0.0
        print(f"[phase7]   CVT circuit: reactor L={circuit.reactor_l:.0f} H tuned "
              f"to healthy C_eq, {args.burden_va:.0f} VA burden; healthy secondary "
              f"phase {h_resp.phase_min:+.2f} arcmin")

    # --- defect cases ---------------------------------------------------------
    if args.kind:
        specs = [DefectSpec(kind=args.kind, severity=args.severity,
                            index=args.index,
                            z_center=args.z_center, extent=args.extent,
                            defect_eps_r=args.defect_eps_r,
                            defect_sigma=args.defect_sigma,
                            theta_center=math.radians(args.theta_deg),
                            theta_extent=math.radians(args.defect_wedge_deg))]
    else:
        n1 = cvt.n_elements - cvt.n_elements_c2
        specs = [
            DefectSpec(kind="shorted_element", severity=1.0, index=max(1, n1 // 2)),
            DefectSpec(kind="shorted_element", severity=1.0, index=n1 + 1),
            DefectSpec(kind="element_aging", severity=1.0, index=0),
            DefectSpec(kind="oil_contamination", severity=1.0),
            # water pocket low in the oil column (water sinks below the oil)
            DefectSpec(kind="water_ingress", severity=1.0, z_center=0.75,
                       extent=0.15),
        ]
    for spec in specs:
        obs_d = run_case_2d(msh, op, materials=mats, matdb=matdb, defect=spec,
                            backend=args.backend)
        row = _report(spec.label(), obs_d, cvt, op)
        row["dC_pct"] = 100.0 * (row["C_pF"] - obs_h.C1_pF) / obs_h.C1_pF
        row["ratio_err_pct"] = 100.0 * (row["tap_frac"] - tap_h) / tap_h
        row["dphase_mdeg"] = row["tap_phase_mdeg"] - rows[0]["tap_phase_mdeg"]
        print(f"[phase7]   signature:  dC={row['dC_pct']:+.2f}%  "
              f"ratio_err={row['ratio_err_pct']:+.2f}%  "
              f"d_tand={obs_d.tan_delta - obs_h.tan_delta:+.5f}  "
              f"dphase={row['dphase_mdeg']:+.3f} mdeg")
        if circuit is not None:
            resp = solve_from_fea(circuit, row["C_pF"] * 1e-12, row["tap_frac"],
                                  obs_d.tan_delta, op.u0, tune=False)
            row["sec_phase_min"] = resp.phase_min
            row["sec_dphase_min"] = resp.phase_min - h_resp.phase_min
            row["sec_ratio_err_pct"] = (abs(resp.u_int) / abs(h_resp.u_int)
                                        - 1.0) * 100.0
            print(f"[phase7]   secondary:  dphase={row['sec_dphase_min']:+.3f} "
                  f"arcmin  ratio_err={row['sec_ratio_err_pct']:+.3f}%")
        rows.append(row)

    # --- plots ----------------------------------------------------------------
    from ctfem.viz import skfem_field_png, plot_foil_ladder
    panels = [(0.7, (-0.1, 2.0), "full device"),
              (0.21, (0.45, 1.75), "stack + sheds (zoom)")]
    skfem_field_png(sol_h, os.path.join(out, "cvt_field.png"), panels=panels)
    nominal = [1.0 - k / cvt.n_elements for k in range(1, cvt.n_elements)]
    plot_foil_ladder(obs_h.foil_potential_frac,
                     os.path.join(out, "cvt_ladder.png"), reference=nominal)
    print(f"[phase7] wrote cvt_field.png + cvt_ladder.png")

    # --- CSV + JSON -----------------------------------------------------------
    import csv
    keys = ["case", "C_pF", "tan_delta", "tap_frac", "tap_kV", "tap_phase_mdeg",
            "leakage_mA", "dC_pct", "ratio_err_pct", "dphase_mdeg",
            "sec_phase_min", "sec_dphase_min", "sec_ratio_err_pct"]
    with open(os.path.join(out, "defect_comparison.csv"), "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=keys)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({k: r.get(k, "") for k in keys})

    summary = {"cvt": asdict(cvt), "rated_capacitance_pF": c_rated,
               "surface_sigma_S": args.surface_sigma,
               "C_pF_healthy": obs_h.C1_pF, "C_deviation": dev,
               "tan_delta_healthy": obs_h.tan_delta,
               "tap_frac_healthy": tap_h, "tap_frac_nominal": k_nom,
               "divider_ladder": obs_h.foil_potential_frac,
               "cases": rows}

    # --- optional 3-D ----------------------------------------------------------
    if args.solve_3d:
        from ctfem.geometry3d import build_cvt_3d
        from ctfem.viz3d import defect_indicator, export_vtu, screenshot, \
            show_interactive

        cvt3 = CVTParams(n_elements=args.n_elements, n_elements_c2=args.n_c2,
                         rated_capacitance_pF=args.rated_pf,
                         mesh_refinement=args.refine_3d)
        p3 = os.path.join(out, "cvt3d.msh")
        t0 = time.time()
        m3 = build_cvt_3d(cvt3, p3, verbose=False)
        print(f"[phase7] 3-D mesh: {m3.n_triangles} tets "
              f"({time.time() - t0:.0f} s)")

        spec3 = specs[0]
        t0 = time.time()
        obs3h = run_case_3d(p3, op, materials=mats, matdb=matdb,
                            backend=args.backend)
        print(f"[phase7] 3-D healthy:  C={obs3h.C1_pF:8.1f} pF  "
              f"tand={obs3h.tan_delta:.5f}   ({time.time() - t0:.0f} s)")
        t0 = time.time()
        obs3d, sol3, _ = run_case_3d(p3, op, materials=mats, matdb=matdb,
                                     defect=spec3, backend=args.backend,
                                     return_solution=True)
        print(f"[phase7] 3-D {spec3.label():<18} C={obs3d.C1_pF:8.1f} pF  "
              f"tand={obs3d.tan_delta:.5f}   ({time.time() - t0:.0f} s)")
        summary["C_pF_3d_healthy"] = obs3h.C1_pF
        summary["tand_3d_healthy"] = obs3h.tan_delta
        summary["C_pF_3d_defect"] = obs3d.C1_pF
        summary["tand_3d_defect"] = obs3d.tan_delta

        indicator = defect_indicator(sol3, spec3, matdb=matdb, operating=op)
        vtu = os.path.join(out, "cvt3d_result.vtu")
        export_vtu(sol3, indicator, vtu)
        field_key = {"phi": "phi_kV", "phase": "phi_phase_mrad",
                     "E": "E_kV_mm"}[args.field]
        screenshot(vtu, os.path.join(out, "view3d.png"), field_key)
        print(f"[phase7] wrote {vtu} ({int(indicator.sum())} defect cells) "
              f"+ view3d.png")

    dump_json(summary, os.path.join(out, "summary.json"))
    print(f"[phase7] outputs in {out}")

    if args.show:
        show_interactive(vtu, field_key)


if __name__ == "__main__":
    main()

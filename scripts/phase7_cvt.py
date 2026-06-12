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
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import asdict

from ctfem.config import CVTParams, OperatingParams, DefectSpec
from ctfem.common import run_case_2d, run_case_3d
from ctfem.geometry import build_cvt
from ctfem.util import results_dir, dump_json


def _report(tag: str, obs, cvt: CVTParams, op: OperatingParams) -> dict:
    """Print one solve and return its summary row."""
    tap = cvt.tap_disc_index
    tap_frac = obs.foil_potential_frac[tap - 1]
    row = {
        "case": tag,
        "C_pF": obs.C1_pF,
        "tan_delta": obs.tan_delta,
        "tap_frac": tap_frac,
        "tap_kV": tap_frac * op.u0 / 1e3,
    }
    print(f"[phase7] {tag:<22} C={obs.C1_pF:8.1f} pF  tand={obs.tan_delta:.5f}"
          f"  tap={tap_frac:7.4f} ({row['tap_kV']:5.2f} kV)")
    return row


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
                             "oil_contamination"])
    ap.add_argument("--index", type=int, default=5)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--theta-deg", type=float, default=0.0)
    ap.add_argument("--defect-wedge-deg", type=float, default=360.0)
    # 3-D
    ap.add_argument("--solve-3d", action="store_true",
                    help="also build+solve the full 3-D revolve")
    ap.add_argument("--refine-3d", type=float, default=0.35)
    ap.add_argument("--field", default="phi", choices=["phi", "E"])
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

    creep = cvt.creepage_distance() * 1e3
    print(f"[phase7] DDB-123 model: H={cvt.total_height} m, "
          f"{cvt.n_elements} elements ({cvt.n_elements - cvt.n_elements_c2}+"
          f"{cvt.n_elements_c2} C1/C2), eps_r_eff={cvt.element_epsr_eff():.0f}, "
          f"U0={op.u0 / 1e3:.1f} kV")
    print(f"[phase7]   {cvt.n_sheds} sheds -> creepage {creep:.0f} mm "
          f"(datasheet {cvt.rated_creepage_mm:.0f} mm, "
          f"{(creep - cvt.rated_creepage_mm) / cvt.rated_creepage_mm:+.1%})")

    # --- 2-D mesh + healthy solve -------------------------------------------
    msh = os.path.join(out, "cvt2d.msh")
    mres = build_cvt(cvt, msh, verbose=False)
    print(f"[phase7] 2-D mesh: {mres.n_triangles} triangles")

    obs_h, sol_h, _ = run_case_2d(msh, op, matdb=matdb, backend=args.backend,
                                  return_solution=True)
    rows = [_report("healthy", obs_h, cvt, op)]

    c_rated = cvt.rated_capacitance_pF
    dev = (obs_h.C1_pF - c_rated) / c_rated
    k_nom = cvt.divider_ratio_nominal()
    tap_h = rows[0]["tap_frac"]
    print(f"[phase7]   vs nameplate:  C {obs_h.C1_pF:.1f} pF / rated "
          f"{c_rated:.0f} pF ({dev:+.2%});  tap {tap_h:.4f} / nominal "
          f"{k_nom:.4f} ({(tap_h - k_nom) / k_nom:+.2%})")

    # --- defect cases ---------------------------------------------------------
    if args.kind:
        specs = [DefectSpec(kind=args.kind, severity=args.severity,
                            index=args.index,
                            theta_center=math.radians(args.theta_deg),
                            theta_extent=math.radians(args.defect_wedge_deg))]
    else:
        n1 = cvt.n_elements - cvt.n_elements_c2
        specs = [
            DefectSpec(kind="shorted_element", severity=1.0, index=max(1, n1 // 2)),
            DefectSpec(kind="shorted_element", severity=1.0, index=n1 + 1),
            DefectSpec(kind="element_aging", severity=1.0, index=0),
            DefectSpec(kind="oil_contamination", severity=1.0),
        ]
    for spec in specs:
        obs_d = run_case_2d(msh, op, matdb=matdb, defect=spec,
                            backend=args.backend)
        row = _report(spec.label(), obs_d, cvt, op)
        row["dC_pct"] = 100.0 * (row["C_pF"] - obs_h.C1_pF) / obs_h.C1_pF
        row["ratio_err_pct"] = 100.0 * (row["tap_frac"] - tap_h) / tap_h
        print(f"[phase7]   signature:  dC={row['dC_pct']:+.2f}%  "
              f"ratio_err={row['ratio_err_pct']:+.2f}%  "
              f"d_tand={obs_d.tan_delta - obs_h.tan_delta:+.5f}")
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
    keys = ["case", "C_pF", "tan_delta", "tap_frac", "tap_kV",
            "dC_pct", "ratio_err_pct"]
    with open(os.path.join(out, "defect_comparison.csv"), "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=keys)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({k: r.get(k, "") for k in keys})

    summary = {"cvt": asdict(cvt), "rated_capacitance_pF": c_rated,
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
        obs3h = run_case_3d(p3, op, matdb=matdb, backend=args.backend)
        print(f"[phase7] 3-D healthy:  C={obs3h.C1_pF:8.1f} pF  "
              f"tand={obs3h.tan_delta:.5f}   ({time.time() - t0:.0f} s)")
        t0 = time.time()
        obs3d, sol3, _ = run_case_3d(p3, op, matdb=matdb, defect=spec3,
                                     backend=args.backend,
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
        field_key = "phi_kV" if args.field == "phi" else "E_kV_mm"
        screenshot(vtu, os.path.join(out, "view3d.png"), field_key)
        print(f"[phase7] wrote {vtu} ({int(indicator.sum())} defect cells) "
              f"+ view3d.png")

    dump_json(summary, os.path.join(out, "summary.json"))
    print(f"[phase7] outputs in {out}")

    if args.show:
        show_interactive(vtu, field_key)


if __name__ == "__main__":
    main()

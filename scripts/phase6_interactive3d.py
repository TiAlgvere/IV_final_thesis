#!/usr/bin/env python
"""Phase 6 -- interactive 3-D FEM: solve the full 3-D CT and explore the result.

Runs ENTIRELY on Windows (scikit-fem backend, P1 tetrahedra, scipy direct
solver).  Solves three cases:

  1. a 2-D axisymmetric reference (same parameters) -- cross-validates the 3-D
     solver: for a healthy device the 3-D C1/tan-delta must match 2-D;
  2. the healthy 3-D device;
  3. the 3-D device with the requested (optionally azimuthally LOCALIZED)
     defect -- e.g. a one-sided moisture pocket, which 2-D cannot represent.

Outputs into results/phase6_*/:
  ct3d_result.vtu   -- ParaView/PyVista file: potential phi_kV (points),
                       field magnitude E_kV_mm, defect indicator, region id
                       (cells).  Open in ParaView for full post-processing.
  view3d.png        -- off-screen verification render (clipped field + defect).
  summary.json      -- C1/tan-delta of all three solves.

Interactive exploration (opens a window -- rotate with mouse, scroll to zoom,
DRAG THE PLANE to cut through the device; defect cells shown in red):

    python scripts/phase6_interactive3d.py --show
    python scripts/phase6_interactive3d.py --kind moisture_ingress --severity 1.0 \
        --theta-deg 0 --defect-wedge-deg 60 --show
    python scripts/phase6_interactive3d.py --field E --show     # |E| instead of phi

Mesh-size guidance (laptop): the default (--n-foils 4 --refine 0.2) is ~0.5 M
tets / ~0.1 M complex dofs and solves in minutes.  Increase gradually.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctfem.config import GeometryParams, OperatingParams, DefectSpec
from ctfem.util import results_dir, dump_json
from ctfem.viz3d import (defect_indicator as _defect_indicator,
                         export_vtu as _export_vtu,
                         screenshot as _screenshot,
                         show_interactive as _show_interactive)


def _build_defect(args) -> DefectSpec:
    if args.kind == "none":
        return DefectSpec()
    return DefectSpec(
        kind=args.kind, severity=args.severity, index=args.index,
        z_center=args.z, extent=args.extent,
        theta_center=math.radians(args.theta_deg),
        theta_extent=math.radians(args.defect_wedge_deg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-foils", type=int, default=4)
    ap.add_argument("--refine", type=float, default=0.2,
                    help="mesh refinement factor (0.2 ~ laptop, larger = finer)")
    ap.add_argument("--kind", default="moisture_ingress",
                    choices=["none", "moisture_ingress", "shorted_foil",
                             "global_aging", "oil_contamination"])
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--z", type=float, default=1.45)
    ap.add_argument("--extent", type=float, default=0.15)
    ap.add_argument("--theta-deg", type=float, default=0.0)
    ap.add_argument("--defect-wedge-deg", type=float, default=60.0,
                    help="azimuthal extent of the defect (360 = full ring)")
    ap.add_argument("--field", default="phi", choices=["phi", "E"],
                    help="scalar shown on the clip plane")
    ap.add_argument("--show", action="store_true",
                    help="open the interactive 3-D window at the end")
    ap.add_argument("--skip-2d", action="store_true")
    args = ap.parse_args()

    out = results_dir("phase6")
    field_key = "phi_kV" if args.field == "phi" else "E_kV_mm"
    g = GeometryParams(n_foils=args.n_foils, mesh_refinement=args.refine)
    op = OperatingParams()
    spec = _build_defect(args)
    summary: dict = {"defect": spec.label()}

    from ctfem.geometry import build_ct
    from ctfem.geometry3d import build_ct_3d
    from ctfem.common import run_case_2d, run_case_3d

    # --- 1. 2-D axisymmetric reference (fast; cross-validates the 3-D solve) --
    if not args.skip_2d:
        p2 = os.path.join(out, "ct2d.msh")
        build_ct(g, p2, verbose=False)
        obs2 = run_case_2d(p2, op, geometry=g, backend="skfem")
        summary["C1_pF_2d_healthy"] = obs2.C1_pF
        summary["tand_2d_healthy"] = obs2.tan_delta
        print(f"[phase6] 2-D reference:   C1={obs2.C1_pF:8.2f} pF  "
              f"tand={obs2.tan_delta:.5f}")

    # --- 2./3. 3-D solves ---------------------------------------------------
    p3 = os.path.join(out, "ct3d.msh")
    t0 = time.time()
    mres = build_ct_3d(g, p3, angle=2.0 * math.pi, verbose=False)
    print(f"[phase6] 3-D mesh: {mres.n_triangles} tets "
          f"({time.time() - t0:.0f} s)")

    t0 = time.time()
    obs3h = run_case_3d(p3, op, geometry=g, backend="skfem")
    print(f"[phase6] 3-D healthy:     C1={obs3h.C1_pF:8.2f} pF  "
          f"tand={obs3h.tan_delta:.5f}   ({time.time() - t0:.0f} s)")
    summary["C1_pF_3d_healthy"] = obs3h.C1_pF
    summary["tand_3d_healthy"] = obs3h.tan_delta

    if spec.kind != "none":
        t0 = time.time()
        obs3d, sol3, _ = run_case_3d(p3, op, defect=spec, geometry=g,
                                     backend="skfem", return_solution=True)
        print(f"[phase6] 3-D {spec.label():<22} C1={obs3d.C1_pF:8.2f} pF  "
              f"tand={obs3d.tan_delta:.5f}   ({time.time() - t0:.0f} s)")
        summary["C1_pF_3d_defect"] = obs3d.C1_pF
        summary["tand_3d_defect"] = obs3d.tan_delta
    else:
        _, sol3, _ = run_case_3d(p3, op, geometry=g, backend="skfem",
                                 return_solution=True)

    # --- export + render -----------------------------------------------------
    indicator = _defect_indicator(sol3, spec, g)
    vtu = os.path.join(out, "ct3d_result.vtu")
    _export_vtu(sol3, indicator, vtu)
    print(f"[phase6] wrote {vtu} ({int(indicator.sum())} defect cells)")

    png = os.path.join(out, "view3d.png")
    _screenshot(vtu, png, field_key)
    print(f"[phase6] wrote {png}")

    dump_json(summary, os.path.join(out, "summary.json"))
    print(f"[phase6] outputs in {out}")

    if args.show:
        _show_interactive(vtu, field_key)


if __name__ == "__main__":
    main()

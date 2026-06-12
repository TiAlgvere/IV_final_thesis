#!/usr/bin/env python
"""Phase 4 (3-D) -- visualize a localized defect in the 3-D mesh (NO solve).

Builds the revolved 3-D CT and overlays a per-cell indicator of where a defect
injects (reusing the exact same `apply_defect` the solver uses), so you can SEE
a one-sided moisture pocket / localized short in 3-D on Windows -- no DOLFINx.

    # 60-degree moisture pocket on one side, mid-height
    python scripts/phase4_defect3d_view.py --kind moisture_ingress \
        --severity 1.0 --z 1.45 --theta-deg 0 --wedge-deg 60 --gui

Open the resulting .pos view in gmsh (auto with --gui), or load ct3d.msh +
ct3d_defect.pos together.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctfem.config import GeometryParams, DefectSpec, apply_mesh_preset, MESH_PRESETS
from ctfem.geometry3d import build_ct_3d
from ctfem.util import results_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="coarse", choices=list(MESH_PRESETS))
    ap.add_argument("--n-foils", type=int, default=6)
    ap.add_argument("--kind", default="moisture_ingress",
                    choices=["moisture_ingress", "shorted_foil",
                             "global_aging", "oil_contamination"])
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--index", type=int, default=1, help="foil index (shorted_foil)")
    ap.add_argument("--z", type=float, default=1.45, help="moisture z-centre [m]")
    ap.add_argument("--extent", type=float, default=0.15)
    ap.add_argument("--theta-deg", type=float, default=0.0,
                    help="azimuthal centre of the defect [deg]")
    ap.add_argument("--defect-wedge-deg", type=float, default=60.0,
                    help="azimuthal extent of the defect [deg]; 360 = full ring")
    ap.add_argument("--revolve-deg", type=float, default=360.0,
                    help="device revolve angle [deg] (use <360 for a fast view)")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    out = results_dir("phase4_3d")
    g = apply_mesh_preset(GeometryParams(n_foils=args.n_foils), args.preset)
    defect = DefectSpec(
        kind=args.kind, severity=args.severity, index=args.index,
        z_center=args.z, extent=args.extent,
        theta_center=math.radians(args.theta_deg),
        theta_extent=math.radians(args.defect_wedge_deg))

    path = os.path.join(out, "ct3d.msh")
    res = build_ct_3d(g, path, angle=math.radians(args.revolve_deg),
                      defect=defect, verbose=True, gui=args.gui)
    print(res.summary())
    print(f"\n[phase4_3d] defect label: {defect.label()}")
    print(f"[phase4_3d] outputs in {out} (ct3d.msh + ct3d_defect.pos)")


if __name__ == "__main__":
    main()

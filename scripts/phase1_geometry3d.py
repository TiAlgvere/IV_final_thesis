#!/usr/bin/env python
"""Phase 1 (3-D) -- build and inspect the revolved 3-D CT mesh.

Runs on Windows (gmsh only; no DOLFINx needed).  Builds the full 360 deg device
by revolving the parametric r-z section, prints per-region element counts, and
optionally opens the gmsh GUI or writes a PNG.

    python scripts/phase1_geometry3d.py --preset coarse --n-foils 6 --gui
    python scripts/phase1_geometry3d.py --wedge 30        # cheap 30deg sector

WARNING: a full-resolution 3-D CT is millions of tetrahedra.  Use --n-foils and
a coarse --preset for laptop runs; refine on HPC.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctfem.config import GeometryParams, apply_mesh_preset, MESH_PRESETS
from ctfem.geometry3d import build_ct_3d
from ctfem.util import results_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="coarse", choices=list(MESH_PRESETS))
    ap.add_argument("--n-foils", type=int, default=6)
    ap.add_argument("--wedge", type=float, default=360.0,
                    help="revolve angle in degrees (default 360 = full device)")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    out = results_dir("phase1_3d")
    g = apply_mesh_preset(GeometryParams(n_foils=args.n_foils), args.preset)
    angle = math.radians(args.wedge)

    path = os.path.join(out, "ct3d.msh")
    res = build_ct_3d(g, path, angle=angle, verbose=True, gui=args.gui)
    print(res.summary())

    if args.png:
        try:
            from ctfem.viz import mesh_png
            print("[phase1_3d] wrote", mesh_png(path, os.path.join(out, "ct3d.png")))
        except Exception as e:  # pragma: no cover
            print(f"[phase1_3d] PNG skipped ({e})")

    print(f"\n[phase1_3d] outputs in {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 1 -- parametric geometry.

Builds the full CT mesh and the degenerate coaxial validation mesh using the
same builder code paths, prints mesh statistics, verifies every physical group
is non-empty (the builder raises otherwise), and -- if PyVista is available --
renders a coloured-by-region PNG.

    python scripts/phase1_geometry.py --preset coarse [--png] [--gui]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctfem.config import GeometryParams, apply_mesh_preset, MESH_PRESETS
from ctfem.geometry import build_ct, build_coax
from ctfem.util import results_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="coarse", choices=list(MESH_PRESETS))
    ap.add_argument("--n-foils", type=int, default=10)
    ap.add_argument("--placement", default="equal_capacitance",
                    choices=["equal_capacitance", "equal_spacing"])
    ap.add_argument("--png", action="store_true", help="render region PNG (PyVista)")
    ap.add_argument("--gui", action="store_true", help="open gmsh GUI (not headless)")
    args = ap.parse_args()

    out = results_dir("phase1")
    g = apply_mesh_preset(
        GeometryParams(n_foils=args.n_foils, foil_placement=args.placement),
        args.preset)

    print("=== full CT geometry ===")
    ct_path = os.path.join(out, "ct.msh")
    ct = build_ct(g, ct_path, verbose=True, gui=args.gui)
    print(ct.summary())

    print("\n=== degenerate coax validation geometry ===")
    coax_path = os.path.join(out, "coax.msh")
    cx = build_coax(0.01, 0.05, 0.2, coax_path, lc=0.002, verbose=True)
    print(cx.summary())

    if args.png:
        try:
            from ctfem.viz import mesh_png
            p = mesh_png(ct_path, os.path.join(out, "ct_mesh.png"))
            print(f"[phase1] wrote {p}")
        except Exception as e:  # pragma: no cover
            print(f"[phase1] PNG skipped ({e})")

    print(f"\n[phase1] outputs in {out}")


if __name__ == "__main__":
    main()

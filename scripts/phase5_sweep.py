#!/usr/bin/env python
"""Phase 5 -- parametric sweep -> dataset (parquet/CSV).

Produces the ML-ready dataset: one row per case with full provenance.  The demo
design covers ShortedFoil(k=1..9) x MoistureIngress severities (>= 200 rows when
combined with temperatures / z-locations), as required by the Definition of Done.

    python scripts/phase5_sweep.py --out results/dataset.parquet --workers 4
    python scripts/phase5_sweep.py --design lhs --n 256 --workers 8
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctfem.config import (CaseConfig, GeometryParams, DefectSpec,
                          apply_mesh_preset)
from ctfem.sweep import cartesian_design, lhs_design, run_sweep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/dataset.parquet")
    ap.add_argument("--design", default="cartesian", choices=["cartesian", "lhs"])
    ap.add_argument("--preset", default="coarse")
    ap.add_argument("--n", type=int, default=256, help="LHS sample count")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    base = CaseConfig(geometry=apply_mesh_preset(GeometryParams(), args.preset))

    if args.design == "cartesian":
        # default cartesian grid yields ~240 rows (>= 200, per Definition of
        # Done): ShortedFoil(k=1..9) x temps, MoistureIngress severities x z x
        # temps, plus global/oil severities x temps and healthy baselines.
        cases = cartesian_design(
            base,
            defect_kinds=["none", "shorted_foil", "moisture_ingress",
                          "global_aging", "oil_contamination"],
            severities=[round(0.1 * i, 1) for i in range(1, 11)],   # 0.1..1.0
            foil_indices=list(range(1, base.geometry.n_foils)),     # k=1..9
            z_centers=[0.9, 1.2, 1.45, 1.7, 2.0],
            temperatures=[20.0, 40.0, 60.0],
        )
    else:
        cases = lhs_design(
            base,
            defect_kinds=["shorted_foil", "moisture_ingress", "global_aging",
                          "oil_contamination"],
            n_samples=args.n)

    print(f"[phase5] design '{args.design}' -> {len(cases)} cases")
    df = run_sweep(cases, args.out, workers=args.workers,
                   resume=not args.no_resume)
    print(f"[phase5] dataset: {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 4 -- defect studies.

Applies each DefectSpec, reports delta-C1, delta-tan delta, foil-ladder
distortion and field-stress redistribution vs. the healthy baseline, and writes
a comparison table (CSV) + ladder plots.

    python scripts/phase4_defects.py --preset coarse
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ctfem.config import (GeometryParams, OperatingParams, MaterialParams,
                          DefectSpec, apply_mesh_preset, MESH_PRESETS)
from ctfem.geometry import build_ct
from ctfem.common import run_case_2d, detect_backend
from ctfem.util import results_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="coarse", choices=list(MESH_PRESETS))
    ap.add_argument("--backend", default="auto", choices=["auto", "dolfinx", "skfem"])
    args = ap.parse_args()
    out = results_dir("phase4")
    backend = detect_backend(args.backend)
    print(f"[phase4] backend: {backend}")

    g = apply_mesh_preset(GeometryParams(), args.preset)
    path = os.path.join(out, "ct.msh")
    build_ct(g, path, verbose=False)

    defects = [
        DefectSpec(kind="none"),
        DefectSpec(kind="shorted_foil", index=1),
        DefectSpec(kind="shorted_foil", index=5),
        DefectSpec(kind="moisture_ingress", severity=0.5, z_center=1.45, extent=0.15),
        DefectSpec(kind="moisture_ingress", severity=1.0, z_center=1.45, extent=0.15),
        DefectSpec(kind="global_aging", severity=0.5),
        DefectSpec(kind="global_aging", severity=1.0),
        DefectSpec(kind="oil_contamination", severity=1.0),
    ]

    rows = []
    healthy = None
    for d in defects:
        obs = run_case_2d(path, OperatingParams(), defect=d, geometry=g,
                          backend=backend)
        if d.kind == "none":
            healthy = obs
        rows.append({
            "defect": d.label(),
            "C1_pF": obs.C1_pF,
            "tan_delta": obs.tan_delta,
            "dC1_pF": obs.C1_pF - (healthy.C1_pF if healthy else obs.C1_pF),
            "dC1_pct": 100 * (obs.C1_pF - healthy.C1_pF) / healthy.C1_pF if healthy else 0.0,
            "dtand": obs.tan_delta - (healthy.tan_delta if healthy else obs.tan_delta),
            "peakE_MV_m": obs.peak_field_overall / 1e6,
        })
        print(f"  {d.label():<26} C1={obs.C1_pF:8.2f}pF tand={obs.tan_delta:.5f}")

    df = pd.DataFrame(rows)
    csv = os.path.join(out, "defect_comparison.csv")
    df.to_csv(csv, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n[phase4] wrote {csv}")
    print(f"[phase4] outputs in {out}")


if __name__ == "__main__":
    main()

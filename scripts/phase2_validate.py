#!/usr/bin/env python
"""Phase 2 -- solver validation: coax analytic benchmark + CT convergence.

    python scripts/phase2_validate.py [--preset coarse]

Checks:
  * computed coax C matches the analytic axisymmetric formula within 0.5%;
  * computed tan delta matches the prescribed material tan delta within 0.5%;
  * mesh convergence of C1 on the full CT over >= 3 refinement levels.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctfem.config import GeometryParams
from ctfem.validate import validate_coax, convergence_study
from ctfem.util import results_dir, dump_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refinements", type=float, nargs="+", default=[0.5, 1.0, 1.8])
    ap.add_argument("--backend", default="auto", choices=["auto", "dolfinx", "skfem"])
    args = ap.parse_args()
    out = results_dir("phase2")

    from ctfem.common import detect_backend
    print(f"[phase2] backend: {detect_backend(args.backend)}")

    print("=== coax analytic benchmark ===")
    v = validate_coax(eps_r=3.5, tan_delta=0.01, refinement=1.0,
                      backend=args.backend)
    print(v.summary())
    print("PASS" if v.passed() else "FAIL (tolerance 0.5%)")
    dump_json(v, os.path.join(out, "coax_validation.json"))

    print("\n=== CT mesh-convergence study ===")
    pts = convergence_study(GeometryParams(), refinements=tuple(args.refinements),
                            backend=args.backend)
    print(f"{'refine':>8} {'tris':>10} {'C1[pF]':>12} {'tan_delta':>12}")
    for p in pts:
        print(f"{p.refinement:8.2f} {p.n_triangles:10d} {p.C1_pF:12.3f} {p.tan_delta:12.5f}")
    if len(pts) >= 2:
        rel = abs(pts[-1].C1_pF - pts[-2].C1_pF) / pts[-1].C1_pF
        print(f"relative C1 change between two finest levels: {rel:.2%}")
    dump_json([p.__dict__ for p in pts], os.path.join(out, "convergence.json"))

    print(f"\n[phase2] outputs in {out}")
    if not v.passed():
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 3 -- healthy baseline full-CT solve.

Outputs: C1, tan delta, foil-potential ladder, peak field per gap, and
potential/field plots (PNG + XDMF/VTX).  Sanity-checks the ladder monotonicity
and the C1 magnitude.

    python scripts/phase3_baseline.py --preset coarse [--png]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctfem.config import GeometryParams, OperatingParams, apply_mesh_preset, MESH_PRESETS
from ctfem.geometry import build_ct
from ctfem.common import run_case_2d, detect_backend
from ctfem.util import results_dir, dump_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="coarse", choices=list(MESH_PRESETS))
    ap.add_argument("--backend", default="auto", choices=["auto", "dolfinx", "skfem"])
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    out = results_dir("phase3")

    g = apply_mesh_preset(GeometryParams(), args.preset)
    op = OperatingParams()
    path = os.path.join(out, "ct.msh")
    mres = build_ct(g, path, verbose=True)

    obs, sol, backend = run_case_2d(path, op, geometry=g, backend=args.backend,
                                    return_solution=True)
    print(f"[phase3] backend: {backend}")

    print(f"\nC1 = {obs.C1_pF:.2f} pF   tan_delta = {obs.tan_delta:.5f}")
    print(f"admittance method cross-check discrepancy: "
          f"{obs.admittance_method_discrepancy:.2e}")
    print("foil ladder (|phi|/U0):")
    for i, f in enumerate(obs.foil_potential_frac):
        print(f"  foil {i + 1:2d}: {f:.3f}")
    print("peak |E| per gap [kV/mm]:")
    for i, e in enumerate(obs.peak_field_per_gap):
        print(f"  gap {i + 1:2d}: {e / 1e6:.2f}")

    # sanity checks
    fr = obs.foil_potential_frac
    monotonic = all(fr[i] >= fr[i + 1] - 1e-3 for i in range(len(fr) - 1))
    print(f"\n[check] ladder monotonic decreasing: {monotonic}")
    plausible = 50.0 < obs.C1_pF < 5000.0
    print(f"[check] C1 in plausible range (50-5000 pF): {plausible}"
          + ("" if plausible else "  <-- FLAG: outside expected range"))

    dump_json(obs, os.path.join(out, "observables.json"))

    if args.png:
        from ctfem.viz import plot_foil_ladder
        plot_foil_ladder(obs.foil_potential_frac,
                         os.path.join(out, "ladder.png"))
        if backend == "skfem":
            from ctfem.viz import skfem_field_png
            skfem_field_png(sol, os.path.join(out, "phi.png"))
            print("[phase3] wrote ladder.png + phi.png")
        else:
            try:
                from ctfem.viz import field_png, write_xdmf
                field_png(sol, os.path.join(out, "phi.png"))
                write_xdmf(sol, os.path.join(out, "phi.xdmf"))
                print("[phase3] wrote PNG/XDMF plots")
            except Exception as e:  # pragma: no cover
                print(f"[phase3] dolfinx plotting skipped ({e})")

    print(f"\n[phase3] outputs in {out}")


if __name__ == "__main__":
    main()

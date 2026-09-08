"""Task 009 validation: representative CVT vs datasheet + cross-backend vs ctfem.

Two layers of evidence:
  1. Datasheet: terminal capacitance ~ 5600 pF and divider ratio ~ n_c2/n.
  2. Cross-backend: the SAME .msh solved by the legacy ctfem scikit-fem backend
     (independent solver, P2 elements) must agree on C and the tap divider ratio.
     This is the strong V&V oracle — it catches porting bugs in geometry/materials.

ctfem is imported here ONLY for validation; the publication pipeline itself does
not depend on it. Exits non-zero if any check fails; failures are reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams  # noqa: E402

PROC = ROOT / "results" / "processed" / "009_representative_cvt"

# tolerances (documented)
TOL_C_DATASHEET = 0.05   # 5% vs nameplate (homogenization + stray + mesh)
TOL_RATIO_NOMINAL = 0.02
TOL_CROSS = 0.03         # Elmer (P1) vs skfem (P2) on the same mesh


def main() -> int:
    summary_path = PROC / "latest_summary.json"
    if not summary_path.is_file():
        print(f"No CVT summary at {summary_path}; run run_cvt.py first.", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    obs = summary["observables"]
    c_elmer = obs["C_total_pF"]
    ratio_elmer = obs["divider_ratio"]
    rated = summary["rated_capacitance_pF"]
    nominal = summary["divider_ratio_nominal"]
    run_dir = Path(summary["run_dir"])
    msh = run_dir / "mesh_gmsh" / "cvt.msh"

    checks: list[tuple[str, bool, str]] = []

    rel_c = abs(c_elmer - rated) / rated
    checks.append(("Terminal C ~ datasheet 5600 pF",
                   rel_c < TOL_C_DATASHEET,
                   f"Elmer={c_elmer:.1f} pF, rated={rated:.0f} pF, rel={rel_c:.3f} (<{TOL_C_DATASHEET})"))

    rel_r = abs(ratio_elmer - nominal) / nominal
    checks.append(("Divider ratio ~ nominal n_c2/n",
                   rel_r < TOL_RATIO_NOMINAL,
                   f"Elmer={ratio_elmer:.4f}, nominal={nominal:.4f}, rel={rel_r:.3f} (<{TOL_RATIO_NOMINAL})"))

    # ---- cross-backend: same mesh through ctfem scikit-fem -------------------
    cross_detail = ""
    try:
        from ctfem.common import run_case_2d
        from ctfem.config import CVTParams as CtCVTParams, MaterialParams, OperatingParams

        op = OperatingParams(frequency=summary["frequency_hz"], um_kv=123.0)
        matdb = CtCVTParams().material_db()
        print("Running ctfem scikit-fem on the same mesh (cross-backend) ...")
        obs_sk = run_case_2d(str(msh), op, materials=MaterialParams(), matdb=matdb, backend="skfem")
        c_sk = obs_sk.C1_pF
        tap_idx = CVTParams().tap_disc_index  # foil_{tap_idx}; fracs are foil_1.. -> index tap_idx-1
        frac_tap = obs_sk.foil_potential_frac[tap_idx - 1]

        rel_cx = abs(c_elmer - c_sk) / c_sk
        checks.append(("C cross-backend (Elmer vs skfem, same mesh)",
                       rel_cx < TOL_CROSS,
                       f"Elmer={c_elmer:.1f} pF, skfem={c_sk:.1f} pF, rel={rel_cx:.3f} (<{TOL_CROSS})"))
        rel_rx = abs(ratio_elmer - frac_tap) / frac_tap
        checks.append(("Divider cross-backend (Elmer vs skfem)",
                       rel_rx < TOL_CROSS,
                       f"Elmer={ratio_elmer:.4f}, skfem={frac_tap:.4f}, rel={rel_rx:.3f} (<{TOL_CROSS})"))
        cross_detail = f"skfem C={c_sk:.1f} pF, tap frac={frac_tap:.4f}, tan_delta={obs_sk.tan_delta:.2e}"
    except Exception as exc:  # noqa: BLE001
        checks.append(("C cross-backend (Elmer vs skfem, same mesh)", False, f"ctfem cross-check failed: {exc}"))

    # ---- report -------------------------------------------------------------
    all_pass = all(ok for _, ok, _ in checks)
    print(f"Validation of: {summary_path}")
    print(f"Elmer C={c_elmer:.1f} pF, divider ratio={ratio_elmer:.4f}, |Vtap|={obs['vtap_abs']:.1f} V")
    if cross_detail:
        print(cross_detail)
    print("-" * 78)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")
    print("-" * 78)
    print("RESULT:", "ALL CHECKS PASS" if all_pass else "SOME CHECKS FAILED")

    lines = [
        "# Task 009 - Representative CVT Validation",
        "",
        "Axisymmetric ComplexEQS on the self-contained DDB-123 geometry, validated",
        "against the datasheet and cross-checked against the ctfem scikit-fem backend",
        "on the identical mesh.",
        "",
        f"- Elmer terminal C: **{c_elmer:.1f} pF** (datasheet 5600 pF)",
        f"- Divider ratio: **{ratio_elmer:.4f}** (nominal {nominal:.4f} = n_c2/n)",
        f"- |Vtap|: {obs['vtap_abs']:.1f} V at U0={obs['u0']:.1f} V; phase {obs['vtap_phase_deg']:.4f} deg",
        f"- Creepage (geometry): {summary['creepage_mm']:.0f} mm (datasheet 3075 mm)",
        "",
        "| check | status | detail |",
        "|-------|--------|--------|",
    ]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines += ["", f"**Result: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}.**", ""]
    (PROC / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

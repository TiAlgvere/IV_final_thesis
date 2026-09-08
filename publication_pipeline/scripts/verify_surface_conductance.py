"""Task 013 - verify the ComplexEQS surface-conductance term vs ctfem scikit-fem.

The new boundary term (Surface Conductance on insulator_surface) is the
surface-pollution model. ctfem's skfem backend has the same Laplace-Beltrami
surface term (surface_conductivity_s), so the strong check is: solve the SAME
clean CVT mesh with both backends across several sigma_s and compare the
field-derived observables (C, divider ratio, tap phase, tan-delta).

Imports ctfem only for validation. Exits non-zero on failure.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from cvt_observables import compute_cvt_observables  # noqa: E402
from run_cvt import FREQ, OMEGA, U0, build_material_table, ensure_solver, material_of_body, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

PROC = ROOT / "results" / "processed" / "013_surface_conductance"
RAW = ROOT / "results" / "raw" / "013_surface_conductance"
SIGMA_S = [1.0e-9, 1.0e-7, 1.0e-5]
BND_OFFSET = 100  # Elmer VtuOutputSolver boundary-entity GeometryIds offset


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    params = CVTParams()
    eps_r_eff = params.element_epsr_eff()

    msh = RAW / "mesh_gmsh" / "cvt.msh"
    build_cvt_ddb123(msh, params, with_pollution_layer=False)  # clean CVT; has insulator_surface
    elmer_parent = RAW / "mesh_elmer"
    elmer_parent.mkdir(parents=True, exist_ok=True)
    eg = resolve_elmer_grid()
    subprocess.run([str(eg), "14", "2", str(msh), "-out", "mesh"], cwd=str(elmer_parent),
                   capture_output=True, text=True, check=True)
    elmer_mesh = elmer_parent / "mesh"
    ids = _parse_mesh_names(elmer_mesh / "mesh.names")
    ins_id = ids["boundaries"]["insulator_surface"]
    tap_body_id = ids["bodies"][f"foil_{params.tap_disc_index}"]

    mat = build_material_table(eps_r_eff)
    mat_index = {m: i + 1 for i, m in enumerate(mat)}
    body_material = {name: material_of_body(name) for name in ids["bodies"]}
    bodies = [(bid, name) for name, bid in ids["bodies"].items()]
    body_epsr = {bid: mat[body_material[name]][0] for name, bid in ids["bodies"].items()}
    body_sigma = {bid: sigma_eff(*mat[body_material[name]]) for name, bid in ids["bodies"].items()}
    hv, gnd = ids["boundaries"]["hv_electrode"], ids["boundaries"]["ground_electrode"]
    ff = ids["boundaries"].get("farfield")

    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    solver = resolve_elmer_solver()

    # ctfem backend
    from ctfem.common import run_case_2d
    from ctfem.config import CVTParams as CtCVT, MaterialParams, OperatingParams
    op = OperatingParams(frequency=FREQ, um_kv=123.0)
    matdb = CtCVT().material_db()
    tap = params.tap_disc_index

    import shutil
    rows = []
    for ss in SIGMA_S:
        # --- ComplexEQS ---
        pdir = RAW / f"ss_{ss:.0e}"
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(elmer_mesh, pdir / "mesh", dirs_exist_ok=True)
        shutil.copy2(dll, pdir / dll.name)
        sif = render_sif(mat, mat_index, body_material, bodies, hv, gnd, ff, ins_id=ins_id, sigma_s=ss)
        (pdir / "case.sif").write_text(sif, encoding="utf-8")
        done = subprocess.run([str(solver), "case.sif"], cwd=str(pdir), capture_output=True, text=True, env=env)
        (pdir / "elmersolver.stdout.log").write_text(done.stdout, encoding="utf-8")
        if done.returncode != 0:
            print(done.stdout[-1500:]); return 1
        vtu = sorted(pdir.glob("**/case*.vtu"))[-1]
        e = compute_cvt_observables(vtu, u0=U0, omega=OMEGA, body_epsr=body_epsr, body_sigma=body_sigma,
                                    tap_body_id=tap_body_id, sigma_s=ss, ins_geom_id=BND_OFFSET + ins_id)

        # --- ctfem skfem ---
        obs = run_case_2d(str(msh), op, materials=MaterialParams(surface_conductivity_s=ss),
                          matdb=matdb, backend="skfem")
        v = obs.foil_potentials[tap - 1]
        sk_phase_deg = math.degrees(math.atan2(v.imag, v.real))

        rows.append({
            "sigma_s": ss,
            "C_elmer": e["C_total_pF"], "C_skfem": obs.C1_pF,
            "ratio_elmer": e["divider_ratio"], "ratio_skfem": obs.foil_potential_frac[tap - 1],
            "phase_elmer_deg": e["vtap_phase_deg"], "phase_skfem_deg": sk_phase_deg,
            "tand_elmer": e["tan_delta"], "tand_skfem": obs.tan_delta,
        })

    # ---- compare ----
    def rel(a, b):
        return abs(a - b) / (abs(b) if abs(b) > 1e-30 else 1.0)

    checks = []
    for r in rows:
        ss = r["sigma_s"]
        checks.append((f"C (sigma_s={ss:.0e})", rel(r["C_elmer"], r["C_skfem"]) < 0.01,
                       f"Elmer={r['C_elmer']:.1f} skfem={r['C_skfem']:.1f} pF"))
        checks.append((f"divider ratio (sigma_s={ss:.0e})", rel(r["ratio_elmer"], r["ratio_skfem"]) < 0.02,
                       f"Elmer={r['ratio_elmer']:.4f} skfem={r['ratio_skfem']:.4f}"))
        checks.append((f"tap phase (sigma_s={ss:.0e})", rel(r["phase_elmer_deg"], r["phase_skfem_deg"]) < 0.10,
                       f"Elmer={r['phase_elmer_deg']:.4e} skfem={r['phase_skfem_deg']:.4e} deg"))
        checks.append((f"tan_delta (sigma_s={ss:.0e})", rel(r["tand_elmer"], r["tand_skfem"]) < 0.10,
                       f"Elmer={r['tand_elmer']:.4e} skfem={r['tand_skfem']:.4e}"))
    all_pass = all(ok for _, ok, _ in checks)

    lines = ["# Task 013 - Surface-Conductance Term Verification (ComplexEQS vs ctfem skfem)", "",
             "Same clean CVT mesh, surface pollution sigma_s on insulator_surface, both backends.", "",
             "| sigma_s [S] | C Elmer/skfem [pF] | ratio E/sk | phase E/sk [deg] | tand E/sk |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['sigma_s']:.0e} | {r['C_elmer']:.1f} / {r['C_skfem']:.1f} | "
                     f"{r['ratio_elmer']:.4f} / {r['ratio_skfem']:.4f} | "
                     f"{r['phase_elmer_deg']:.3e} / {r['phase_skfem_deg']:.3e} | "
                     f"{r['tand_elmer']:.3e} / {r['tand_skfem']:.3e} |")
    lines += ["", f"**Result: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}** "
              "(C<1%, ratio<2%, phase<10%, tand<10%; phase/tand looser for P1-vs-P2 + small signal).", ""]
    (PROC / "surface_conductance_verification.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Verification: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    for name, ok, det in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {det}")
    print(f"Report: {PROC / 'surface_conductance_verification.md'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

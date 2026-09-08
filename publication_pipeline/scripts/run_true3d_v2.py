"""Task 023 - clean 3D ComplexEQS solve on the shed-fidelity geometry + validation.

Builds a COARSER full-360 mesh of the SAME 27-shed geometry (mesh-refinement control so a
direct solve fits - the audit uses the fine mesh; sheds barely affect the clean-baseline C)
-> Cartesian-3D ComplexEQS (no axisymmetric metric) -> true-3D observables -> compares to
the Task 022 result and a fresh axisymmetric baseline.

Reuses the verified 3D SIF renderer / 3D observables / 2D baseline from run_true3d.py.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123_3d_v2 import build_cvt_ddb123_3d_v2  # noqa: E402
from run_cvt import build_material_table, ensure_solver, material_of_body, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402
from run_true3d import observables_3d, render_sif_3d, run_2d_baseline  # noqa: E402

PROC = ROOT / "results" / "processed" / "023_true3d_geometry_fidelity"
RAW = ROOT / "results" / "raw" / "023_true3d_geometry_fidelity"
COARSE_LC = {"shed": 0.035, "porcelain": 0.028, "oil": 0.045, "element_c1": 0.030,
             "element_c2": 0.024, "foil_tap": 0.014, "stack_top": 0.014, "stack_bottom": 0.014,
             "head_housing": 0.055, "base_tank": 0.070, "air": 0.400}


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    print("Building COARSE 27-shed solve mesh (refinement control) ...")
    msh = RAW / "cvt3d_v2_solve.msh"
    meta = build_cvt_ddb123_3d_v2(msh, n_sheds=27, curvature=4.0, lc=COARSE_LC)
    print(f"  nodes={meta['n_nodes']} tets={meta['n_tets']} qmin={meta['quality_minSICN']['min']:.3f} "
          f"creepage={meta['creepage_mm']:.0f}mm")
    (PROC / "solve_mesh_meta.json").write_text(
        json.dumps({k: v for k, v in meta.items() if k not in ("parameters", "worst20")}, indent=2),
        encoding="utf-8")

    ep = RAW / "elmer"; ep.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(resolve_elmer_grid()), "14", "2", str(msh), "-out", "mesh"],
                   cwd=str(ep), capture_output=True, text=True, check=True)
    em = ep / "mesh"; ids = _parse_mesh_names(em / "mesh.names")
    bodies = [(bid, n) for n, bid in ids["bodies"].items()]
    tap_id = ids["bodies"]["foil_tap"]
    hv, gnd = ids["boundaries"]["hv_electrode"], ids["boundaries"]["ground_electrode"]
    ff = ids["boundaries"].get("farfield")

    eps = CVTParams().element_epsr_eff()
    mat = build_material_table(eps); mat_index = {m_: i + 1 for i, m_ in enumerate(mat)}
    bm = {n: material_of_body(n) for n in ids["bodies"]}
    be = {bid: mat[bm[n]][0] for n, bid in ids["bodies"].items()}
    bs = {bid: sigma_eff(*mat[bm[n]]) for n, bid in ids["bodies"].items()}

    pdir = RAW / "run3d"; pdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(em, pdir / "mesh", dirs_exist_ok=True); shutil.copy2(dll, pdir / dll.name)

    solved = None
    for direct in (True, False):
        sif = render_sif_3d(mat, mat_index, bm, bodies, hv, gnd, ff, direct=direct)
        (pdir / "case.sif").write_text(sif, encoding="utf-8")
        print(f"Solving 3D ({'UMFPack direct' if direct else 'BiCGStabl iterative'}) ...")
        t0 = time.time()
        d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                           capture_output=True, text=True, env=env)
        dt = time.time() - t0
        (pdir / "solver.log").write_text(d.stdout, encoding="utf-8")
        if d.returncode == 0 and sorted(pdir.glob("**/case*.vtu")):
            solved = ("direct" if direct else "iterative", dt)
            print(f"  solved in {dt:.1f}s ({solved[0]})"); break
        print(f"  FAILED ({'direct' if direct else 'iterative'}); tail:\n{d.stdout[-700:]}")
    if solved is None:
        print("3D solve failed with both linear solvers."); return 1

    vtu = sorted(pdir.glob("**/case*.vtu"))[-1]
    o3 = observables_3d(vtu, be, bs, tap_id)
    print("Running 2D axisymmetric baseline ...")
    o2 = run_2d_baseline(dll, env)
    t22 = None
    p22 = ROOT / "results" / "processed" / "022_true3d_foundation" / "validation.json"
    if p22.is_file():
        t22 = json.loads(p22.read_text(encoding="utf-8")).get("true3d")

    def cmp(a, b):
        return None if not b else (a / b - 1.0) * 100.0

    result = {
        "solver": solved[0], "solve_seconds": solved[1],
        "mesh": {"n_nodes": meta["n_nodes"], "n_tets": meta["n_tets"],
                 "n_boundary_tris": meta["n_boundary_tris"],
                 "qmin": meta["quality_minSICN"]["min"], "creepage_mm": meta["creepage_mm"]},
        "true3d_v2": o3, "axisymmetric": o2, "task022_true3d": t22,
        "delta_vs_axisym_pct": {k: cmp(o3[k], o2[k]) for k in
                                ("C_pF", "vtap_abs", "vtap_phase_mdeg", "divider_ratio", "stored_energy_J")},
        "delta_vs_task022_pct": ({k: cmp(o3[k], t22[k]) for k in
                                  ("C_pF", "vtap_abs", "divider_ratio", "stored_energy_J")} if t22 else None),
    }
    (PROC / "validation_v2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n               v2 3D        axisym        delta")
    for k, lab in [("C_pF", "C [pF]"), ("divider_ratio", "ratio"), ("vtap_abs", "|Vtap| [V]"),
                   ("vtap_phase_mdeg", "phase [mdeg]"), ("stored_energy_J", "energy [J]")]:
        dv = result["delta_vs_axisym_pct"][k]
        print(f"{lab:14s} {o3[k]:12.4g} {o2[k]:12.4g}  {('' if dv is None else f'{dv:+.2f}%')}")
    print(f"Emax = {o3['emax']:.3e} V/m at (r,z)=({o3['emax_rz'][0]:.3f},{o3['emax_rz'][1]:.3f})")
    print(f"\nWrote {PROC / 'validation_v2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

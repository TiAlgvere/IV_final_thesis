"""Task 024 - clean 3D ComplexEQS solve on the 74-shed component model + validation.

Coarser full-360 mesh of the v4 geometry (refinement control for a direct solve; sheds
barely affect clean-baseline C) -> Cartesian-3D ComplexEQS (no axisymmetric metric) ->
true-3D observables -> compares to axisymmetric + Task 022/023. Adds a `resin` dielectric
material. Writes validation.json + solve_mesh_meta.json.
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
for _p in (str(ROOT.parent), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123_3d_v4 import MAT_OF_V4, build_cvt_ddb123_3d_v4  # noqa: E402
from run_cvt import build_material_table, ensure_solver, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402
from run_true3d import observables_3d, render_sif_3d, run_2d_baseline  # noqa: E402

PROC = ROOT / "results" / "processed" / "024_true3d_ddb123_component_architecture"
RAW = ROOT / "results" / "raw" / "024_true3d_ddb123_component_architecture"
COARSE_LC = {"shed": 0.040, "porcelain": 0.028, "oil": 0.045, "element_c1": 0.030,
             "element_c2": 0.022, "foil_tap": 0.012, "stack_top": 0.012, "stack_bottom": 0.012,
             "head_metal": 0.055, "head_oil": 0.045, "tank_metal": 0.070, "base_emu": 0.045,
             "reactor": 0.030, "aux_component": 0.025, "base_oil": 0.050, "secondary_box": 0.035,
             "oil_valve": 0.018, "oil_level": 0.018, "resin": 0.015, "air": 0.420}


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    print("Building COARSE 74-shed solve mesh ...")
    msh = RAW / "cvt3d_v4_solve.msh"
    meta = build_cvt_ddb123_3d_v4(msh, curvature=4.0, lc=COARSE_LC)
    print(f"  nodes={meta['n_nodes']} tets={meta['n_tets']} qmin={meta['quality_minSICN']['min']:.3f} "
          f"slivers={meta['n_slivers_lt_0.05']} creepage={meta['creepage_mm']:.0f}mm")
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
    mat = build_material_table(eps)
    mat["resin"] = (4.5, 0.01, 0.0)                       # epoxy-resin HF-exit dielectric
    mat_index = {m_: i + 1 for i, m_ in enumerate(mat)}
    bm = {n: MAT_OF_V4.get(n, "air") for n in ids["bodies"]}
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

    def cmp(a, b):
        return None if not b else (a / b - 1.0) * 100.0
    result = {
        "solver": solved[0], "solve_seconds": solved[1],
        "mesh": {"n_nodes": meta["n_nodes"], "n_tets": meta["n_tets"],
                 "n_boundary_tris": meta["n_boundary_tris"], "qmin": meta["quality_minSICN"]["min"],
                 "creepage_mm": meta["creepage_mm"], "n_sheds": meta["n_sheds"]},
        "true3d_v4": o3, "axisymmetric": o2,
        "delta_vs_axisym_pct": {k: cmp(o3[k], o2[k]) for k in
                                ("C_pF", "vtap_abs", "vtap_phase_mdeg", "divider_ratio", "stored_energy_J")},
        "targets": {"C_pF": 5600, "divider_ratio": 1 / 6, "creepage_mm": 3075},
    }
    (PROC / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n               v4 3D        axisym        delta      vs 5600pF/(1/6)")
    for k, lab, tgt in [("C_pF", "C [pF]", 5600), ("divider_ratio", "ratio", 1 / 6),
                        ("vtap_abs", "|Vtap| [V]", None), ("vtap_phase_mdeg", "phase [mdeg]", None),
                        ("stored_energy_J", "energy [J]", None)]:
        dv = result["delta_vs_axisym_pct"][k]
        tg = "" if tgt is None else f"  ({(o3[k]/tgt-1)*100:+.2f}% vs target)"
        print(f"{lab:14s} {o3[k]:12.4g} {o2[k]:12.4g}  {('' if dv is None else f'{dv:+.2f}%'):>8}{tg}")
    print(f"Emax = {o3['emax']:.3e} V/m at (r,z)=({o3['emax_rz'][0]:.3f},{o3['emax_rz'][1]:.3f})")
    print(f"\nWrote {PROC / 'validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task 010 - surface-pollution conductivity sweep on the representative DDB-123 CVT.

Applies the verified Tasks 006-008 pollution model (a thin conductive volume layer,
eps_r = porcelain so sigma=0 == clean) to the validated Task 009 axisymmetric CVT,
on the porcelain-wall creepage surface. Only the pollution-layer conductivity is
swept; one fixed mesh. Reuses the run_cvt material/SIF machinery and the axisymmetric
ComplexEQS solver (no new physics).

Outputs CSV + summary under results/{raw,processed}/010_cvt_pollution/.
Acceptance #1/#2 (clean ~5624 pF; sigma=0 == clean) are checked here; trend/state-space
analysis is in analyze_cvt_pollution.py.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from cvt_observables import compute_cvt_observables  # noqa: E402
from run_cvt import FREQ, OMEGA, U0, ensure_solver, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

CASE = "010_cvt_pollution"
RAW = ROOT / "results" / "raw" / CASE
PROC = ROOT / "results" / "processed" / CASE
CLEAN_C_PF = 5624.0  # Task 009 clean terminal capacitance (acceptance reference)


def material_of_body(name: str) -> str:
    if name == "pollution_layer":
        return "pollution"
    if name == "air":
        return "air"
    if name == "oil":
        return "oil"
    if name in ("porcelain", "porcelain_shed"):
        return "porcelain"
    if name.startswith("element_"):
        return "element"
    if name.startswith("foil_") or name in ("stack_top", "stack_bottom", "head_housing", "base_tank"):
        return "metal"
    raise KeyError(f"no material rule for body {name!r}")


def material_table(eps_r_eff: float, sigma_pollution: float) -> dict[str, tuple[float, float, float]]:
    # pollution layer: porcelain dielectric (eps_r 6, tan_delta 0.005) PLUS the swept
    # conduction sigma. At sigma_pollution=0 it is identical to porcelain -> clean.
    return {
        "air": (1.0, 0.0, 0.0),
        "oil": (2.2, 0.001, 0.0),
        "porcelain": (6.0, 0.005, 0.0),
        "element": (eps_r_eff, 0.002, 0.0),
        "metal": (1.0, 0.0, 1.0),
        "pollution": (6.0, 0.005, sigma_pollution),
    }


def default_sigma_list() -> list[float]:
    return [0.0] + [float(s) for s in np.logspace(-10, -3, 15)]


def run_one(sigma, run_root, elmer_mesh, dll, ids, eps_r_eff, env) -> dict:
    tag = "clean" if sigma == 0.0 else f"sigma_{sigma:.3e}"
    pdir = run_root / tag
    pdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(elmer_mesh, pdir / "mesh", dirs_exist_ok=True)
    shutil.copy2(dll, pdir / dll.name)

    bodies_by_name = ids["bodies"]
    mat = material_table(eps_r_eff, sigma)
    mat_index = {m: i + 1 for i, m in enumerate(mat)}
    body_material = {name: material_of_body(name) for name in bodies_by_name}
    bodies = [(bid, name) for name, bid in bodies_by_name.items()]
    hv_id = ids["boundaries"]["hv_electrode"]
    gnd_id = ids["boundaries"]["ground_electrode"]
    ff_id = ids["boundaries"].get("farfield")

    sif = render_sif(mat, mat_index, body_material, bodies, hv_id, gnd_id, ff_id)
    (pdir / "case.sif").write_text(sif, encoding="utf-8")

    solver = resolve_elmer_solver()
    done = subprocess.run([str(solver), "case.sif"], cwd=str(pdir), capture_output=True, text=True, env=env)
    (pdir / "elmersolver.stdout.log").write_text(done.stdout, encoding="utf-8")
    if done.returncode != 0:
        print(done.stdout[-2000:])
        raise RuntimeError(f"ElmerSolver failed for sigma={sigma}")

    vtus = sorted(pdir.glob("**/case*.vtu"))
    if not vtus:
        raise RuntimeError(f"no VTU for sigma={sigma}")

    body_epsr = {bid: mat[body_material[name]][0] for name, bid in bodies_by_name.items()}
    body_sigma = {bid: sigma_eff(*mat[body_material[name]]) for name, bid in bodies_by_name.items()}
    tap_id = bodies_by_name[f"foil_{CVTParams().tap_disc_index}"]
    obs = compute_cvt_observables(vtus[-1], u0=U0, omega=OMEGA, body_epsr=body_epsr,
                                  body_sigma=body_sigma, tap_body_id=tap_id)
    obs["sigma"] = sigma
    obs["sigma_over_we"] = sigma / (OMEGA * 8.8541878128e-12 * 6.0)
    return obs


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    params = CVTParams()
    eps_r_eff = params.element_epsr_eff()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = RAW / f"run_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    print("Building pollution-layer CVT mesh ...")
    msh = run_root / "mesh_gmsh" / "cvt.msh"
    meta = build_cvt_ddb123(msh, params, with_pollution_layer=True)
    elmer_parent = run_root / "mesh_elmer"
    elmer_parent.mkdir(parents=True, exist_ok=True)
    eg = resolve_elmer_grid()
    done = subprocess.run([str(eg), "14", "2", str(msh), "-out", "mesh"],
                          cwd=str(elmer_parent), capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout); print(done.stderr, file=sys.stderr); return 1
    elmer_mesh = elmer_parent / "mesh"
    ids = _parse_mesh_names(elmer_mesh / "mesh.names")
    if "pollution_layer" not in ids["bodies"]:
        print("pollution_layer body missing from mesh", file=sys.stderr); return 1

    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    sigma_list = default_sigma_list()
    rows = []
    for sigma in sigma_list:
        print(f"  sigma = {sigma:.3e} S/m ...")
        row = run_one(sigma, run_root, elmer_mesh, dll, ids, eps_r_eff, env)
        rows.append(row)
        print(f"    C={row['C_total_pF']:.1f}pF  |Vtap|={row['vtap_abs']:.1f}V  "
              f"dtheta={row['vtap_phase_mdeg']:.2f}mdeg  tand={row['tan_delta']:.4e}  Ileak={row['i_leak']:.3e}")

    cols = ["sigma", "sigma_over_we", "C_total_pF", "vtap_abs", "vtap_phase_deg",
            "vtap_phase_mdeg", "tan_delta", "i_leak", "p_loss", "divider_ratio"]
    csv_path = run_root / "cvt_pollution_sweep.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.10g}" for c in cols) + "\n")
    shutil.copy2(csv_path, PROC / "cvt_pollution_sweep.csv")

    # ---- acceptance #1/#2 ---------------------------------------------------
    clean = next(r for r in rows if r["sigma"] == 0.0)
    c0 = clean["C_total_pF"]
    rel_clean = abs(c0 - CLEAN_C_PF) / CLEAN_C_PF
    acc = {
        "sigma0_C_pF": c0,
        "task009_clean_C_pF": CLEAN_C_PF,
        "rel_to_task009_clean": rel_clean,
        "acc1_clean_~5624pF": bool(rel_clean < 0.01),
        "acc2_sigma0_equals_clean": bool(rel_clean < 0.01),
        "sigma0_phase_mdeg": clean["vtap_phase_mdeg"],
        "sigma0_divider_ratio": clean["divider_ratio"],
    }

    summary = {
        "case": CASE, "timestamp": ts, "frequency_hz": FREQ, "u0_v": U0,
        "n_points": len(rows), "sigma_list": sigma_list,
        "pollution_thickness_m": params.pollution_thickness,
        "acceptance": acc, "csv": str(csv_path), "rows": rows,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PROC / "latest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== acceptance #1/#2 ===")
    print(f"  sigma=0 C = {c0:.1f} pF  vs Task009 clean {CLEAN_C_PF:.0f} pF  (rel {rel_clean:.4f})")
    print(f"  -> clean ~5624 pF: {acc['acc1_clean_~5624pF']}; sigma=0 == clean: {acc['acc2_sigma0_equals_clean']}")
    print(f"CSV: {csv_path}")
    print(f"Next: python publication_pipeline/scripts/analyze_cvt_pollution.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

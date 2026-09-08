"""Task 009 - solve the representative DDB-123 CVT with the axisymmetric ComplexEQS.

Builds the self-contained CVT mesh (geometry/gmsh/cvt_ddb123.py) -> ElmerGrid ->
renders an axisymmetric complex-EQS SIF (floating metals via capped sigma=1;
dielectric loss pre-folded into sigma_eff) -> ElmerSolver -> capacitance + divider
observables. The clean representative baseline; pollution is a later task.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
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
from build_solver import build_solver  # noqa: E402
from cvt_observables import compute_cvt_observables  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

CASE = "009_representative_cvt"
RAW = ROOT / "results" / "raw" / CASE
PROC = ROOT / "results" / "processed" / CASE
SOLVER_SOURCE = ROOT / "elmer" / "solvers" / "ComplexEQS.F90"
SOLVER_DLL = ROOT / "elmer" / "solvers" / "ComplexEQS.dll"

EPS0 = 8.8541878128e-12
FREQ = 50.0
OMEGA = 2.0 * math.pi * FREQ
UM_KV = 123.0
U0 = UM_KV * 1.0e3 / math.sqrt(3.0)  # phase-to-earth RMS amplitude

# Material table: name -> (eps_r, tan_delta, sigma_phys). element_dielectric eps_r
# is filled per-params (eps_r_eff). Metals use the capped sigma=1 S/m (ctfem
# SIGMA_NUMERICAL_CAP) so they float as equipotentials without spurious loss.
METAL_SIGMA_CAP = 1.0


def material_of_body(name: str) -> str:
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


def build_material_table(eps_r_eff: float) -> dict[str, tuple[float, float, float]]:
    return {
        "air": (1.0, 0.0, 0.0),
        "oil": (2.2, 0.001, 0.0),
        "porcelain": (6.0, 0.005, 0.0),
        "element": (eps_r_eff, 0.002, 0.0),
        "metal": (1.0, 0.0, METAL_SIGMA_CAP),
    }


def sigma_eff(eps_r: float, tan_delta: float, sigma_phys: float) -> float:
    """Pre-fold dielectric loss: sigma_eff = sigma + omega*eps0*eps_r*tan_delta."""
    return sigma_phys + OMEGA * EPS0 * eps_r * tan_delta


def render_sif(mat_table, mat_index, body_material, bodies, hv_id, gnd_id, ff_id,
               ins_id=None, sigma_s=0.0, surface_cond_str=None) -> str:
    """If ins_id is given with sigma_s>0 (or surface_cond_str), add a
    surface-conductance (pollution) BC on the insulator_surface creepage curve.
    surface_cond_str overrides the value, e.g. a windowed MATC expression:
      'Variable Coordinate 2\\n    Real MATC "(tx>=1.345)*1.0e-6"'."""
    parts = [
        'Header\n  CHECK KEYWORDS Warn\n  Mesh DB "." "mesh"\nEnd\n',
        'Simulation\n  Coordinate System = "Axi Symmetric"\n  Simulation Type = Steady State\n'
        f'  Steady State Max Iterations = 1\n  Output Intervals = 1\n  Frequency = {FREQ}\n'
        '  Output File = "case.result"\n  Post File = "case.vtu"\nEnd\n',
        "Constants\n  Permittivity Of Vacuum = 8.8541878128e-12\nEnd\n",
        'Equation 1\n  Name = "ComplexEQS"\n  Active Solvers(1) = 1\nEnd\n',
        'Solver 1\n  Equation = "ComplexEQS"\n  Procedure = "ComplexEQS" "ComplexEQSSolver"\n'
        "  Variable = Potential[Potential Re:1 Potential Im:1]\n  Linear System Complex = True\n"
        "  Linear System Solver = Direct\n  Linear System Direct Method = UMFPack\n"
        "  Steady State Convergence Tolerance = 1.0e-9\nEnd\n",
    ]
    for mname, idx in mat_index.items():
        eps, tand, sig = mat_table[mname]
        parts.append(
            f"Material {idx}\n  Name = \"{mname}\"\n  Relative Permittivity = {eps!r}\n"
            f"  Electric Conductivity = {sigma_eff(eps, tand, sig)!r}\nEnd\n"
        )
    for bid, name in bodies:
        parts.append(f"Body {bid}\n  Name = \"{name}\"\n  Equation = 1\n  Material = {mat_index[body_material[name]]}\nEnd\n")
    parts.append(
        f"Boundary Condition 1\n  Name = \"hv_electrode\"\n  Target Boundaries(1) = {hv_id}\n"
        f"  Potential Re = Real {U0!r}\n  Potential Im = Real 0.0\nEnd\n"
    )
    parts.append(
        f"Boundary Condition 2\n  Name = \"ground_electrode\"\n  Target Boundaries(1) = {gnd_id}\n"
        "  Potential Re = Real 0.0\n  Potential Im = Real 0.0\nEnd\n"
    )
    if ff_id is not None:
        parts.append(
            f"Boundary Condition 3\n  Name = \"farfield\"\n  Target Boundaries(1) = {ff_id}\n"
            "  Potential Re = Real 0.0\n  Potential Im = Real 0.0\nEnd\n"
        )
    if ins_id is not None and (sigma_s > 0.0 or surface_cond_str):
        val = surface_cond_str if surface_cond_str else f"Real {sigma_s!r}"
        parts.append(
            f"Boundary Condition 4\n  Name = \"insulator_surface\"\n  Target Boundaries(1) = {ins_id}\n"
            f"  Surface Conductance = {val}\nEnd\n"
        )
    return "\n".join(parts)


def ensure_solver() -> Path:
    if not SOLVER_DLL.is_file() or SOLVER_DLL.stat().st_mtime < SOLVER_SOURCE.stat().st_mtime:
        build_solver(SOLVER_SOURCE, SOLVER_DLL)
    return SOLVER_DLL


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    params = CVTParams()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RAW / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Building CVT mesh ...")
    msh = run_dir / "mesh_gmsh" / "cvt.msh"
    meta = build_cvt_ddb123(msh, params)
    (run_dir / "mesh_gmsh" / "cvt.mesh.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  nodes={meta['n_nodes']} tris={meta['n_triangles']} eps_r_eff={meta['element_epsr_eff']:.1f} "
          f"creepage={meta['creepage_distance_mm']:.0f}mm")

    elmergrid = resolve_elmer_grid()
    elmer_mesh_parent = run_dir / "mesh_elmer"
    elmer_mesh_parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([str(elmergrid), "14", "2", str(msh), "-out", "mesh"],
                          cwd=str(elmer_mesh_parent), capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout); print(done.stderr, file=sys.stderr)
        return 1
    elmer_mesh = elmer_mesh_parent / "mesh"

    ids = _parse_mesh_names(elmer_mesh / "mesh.names")
    bodies_by_name = ids["bodies"]
    boundaries = ids["boundaries"]

    eps_r_eff = params.element_epsr_eff()
    mat_table = build_material_table(eps_r_eff)
    mat_index = {m: i + 1 for i, m in enumerate(mat_table)}
    body_material = {name: material_of_body(name) for name in bodies_by_name}
    bodies = [(bid, name) for name, bid in bodies_by_name.items()]

    hv_id = boundaries["hv_electrode"]
    gnd_id = boundaries["ground_electrode"]
    ff_id = boundaries.get("farfield")

    sif = render_sif(mat_table, mat_index, body_material, bodies, hv_id, gnd_id, ff_id)
    (run_dir / "case.sif").write_text(sif, encoding="utf-8")
    shutil.copytree(elmer_mesh, run_dir / "mesh", dirs_exist_ok=True)
    shutil.copy2(dll, run_dir / dll.name)

    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    print("Solving (axisymmetric complex EQS) ...")
    solver = resolve_elmer_solver()
    done = subprocess.run([str(solver), "case.sif"], cwd=str(run_dir), capture_output=True, text=True, env=env)
    (run_dir / "elmersolver.stdout.log").write_text(done.stdout, encoding="utf-8")
    (run_dir / "elmersolver.stderr.log").write_text(done.stderr, encoding="utf-8")
    if done.returncode != 0:
        print(done.stdout[-2500:]); print(done.stderr, file=sys.stderr)
        return 1

    vtus = sorted(run_dir.glob("**/case*.vtu"))
    if not vtus:
        print("No VTU produced.", file=sys.stderr)
        return 1

    body_epsr = {bid: mat_table[body_material[name]][0] for name, bid in bodies_by_name.items()}
    body_sigma = {bid: sigma_eff(*mat_table[body_material[name]]) for name, bid in bodies_by_name.items()}
    tap_body_id = bodies_by_name[f"foil_{params.tap_disc_index}"]
    obs = compute_cvt_observables(vtus[-1], u0=U0, omega=OMEGA, body_epsr=body_epsr,
                                  body_sigma=body_sigma, tap_body_id=tap_body_id)

    summary = {
        "case": CASE, "timestamp": ts, "frequency_hz": FREQ, "u0_v": U0,
        "eps_r_eff": eps_r_eff, "tap_disc_index": params.tap_disc_index,
        "divider_ratio_nominal": params.divider_ratio_nominal(),
        "rated_capacitance_pF": params.rated_capacitance_pF,
        "creepage_mm": meta["creepage_distance_mm"],
        "observables": obs,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PROC / "latest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Representative DDB-123 CVT ===")
    print(f"  C_total      : {obs['C_total_pF']:.1f} pF   (datasheet 5600 pF)")
    print(f"  |Vtap|       : {obs['vtap_abs']:.1f} V   (U0 = {U0:.1f} V)")
    print(f"  divider ratio: {obs['divider_ratio']:.4f}   (nominal {params.divider_ratio_nominal():.4f})")
    print(f"  Vtap phase   : {obs['vtap_phase_deg']:.4f} deg")
    print(f"  summary      : {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

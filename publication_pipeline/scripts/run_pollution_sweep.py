"""Task 006 - conductive-pollution conductivity sweep (complex EQS, native Elmer).

Pipeline per sweep point:
  build one pollution mesh (gmsh) -> convert (ElmerGrid) -> for each sigma render
  the complex-EQS SIF, run ElmerSolver, extract diagnostic observables.

Only the pollution-layer conductivity changes across the sweep; the mesh is fixed.
At sigma = 0 the pollution body is dielectrically identical to porcelain, so that
point reproduces the clean baseline (the solver's validation anchor).

Outputs (under results/{raw,processed}/006_pollution_sweep/run_<ts>/):
  * pollution_sweep.csv   - one row per sigma with all observables
  * summary.json          - run metadata
  * pollution_sweep.png   - observable-vs-sigma curves (written by plot_sweep.py)
"""

from __future__ import annotations

import argparse
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
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_solver, resolve_elmer_home  # noqa: E402
from publication_pipeline.geometry.gmsh.clean_baseline import (  # noqa: E402
    CleanBaselineParameters,
    build_clean_baseline_mesh,
    write_mesh_log,
)
from build_solver import build_solver  # noqa: E402
from observables import ObservableConfig, compute_observables  # noqa: E402
from run_elmer_case import _parse_mesh_names, _render_sif  # noqa: E402

CASE_NAME = "006_pollution_sweep"
TEMPLATE_SIF = ROOT / "elmer" / "sif_templates" / "pollution_complex_eqs.sif"
SOLVER_SOURCE = ROOT / "elmer" / "solvers" / "ComplexEQS.F90"
SOLVER_DLL = ROOT / "elmer" / "solvers" / "ComplexEQS.dll"
RAW_RESULTS_DIR = ROOT / "results" / "raw" / CASE_NAME
PROCESSED_RESULTS_DIR = ROOT / "results" / "processed" / CASE_NAME

HV_VOLTAGE = 110000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Task 006 pollution-conductivity sweep.")
    parser.add_argument("--frequency", type=float, default=50.0, help="Power frequency [Hz].")
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="*",
        default=None,
        help="Explicit pollution conductivities [S/m]. Default: 0 plus logspace(1e-10..1e-3).",
    )
    parser.add_argument("--mesh-size-air", type=float, default=0.025)
    parser.add_argument("--pollution-thickness", type=float, default=0.005)
    parser.add_argument("--mesh-size-pollution", type=float, default=0.002)
    parser.add_argument("--rebuild-solver", action="store_true", help="Force recompiling ComplexEQS.dll.")
    return parser.parse_args()


def default_sigma_list() -> list[float]:
    # Span sigma/(omega*eps) from << 1 to >> 1 so the full diagnostic transition shows.
    return [0.0] + [float(s) for s in np.logspace(-10, -3, 15)]


def ensure_solver(rebuild: bool) -> Path:
    if rebuild or not SOLVER_DLL.is_file() or SOLVER_DLL.stat().st_mtime < SOLVER_SOURCE.stat().st_mtime:
        print("Building ComplexEQS solver ...")
        build_solver(SOLVER_SOURCE, SOLVER_DLL)
    else:
        print(f"Using existing solver: {SOLVER_DLL}")
    return SOLVER_DLL


def build_and_convert_mesh(run_raw_dir: Path, params: CleanBaselineParameters) -> Path:
    """Build the pollution gmsh mesh and convert it to Elmer format. Returns elmer mesh dir."""
    gmsh_dir = run_raw_dir / "mesh_gmsh"
    elmer_dir = run_raw_dir / "mesh_elmer"
    gmsh_dir.mkdir(parents=True, exist_ok=True)
    elmer_dir.mkdir(parents=True, exist_ok=True)

    msh_path = gmsh_dir / "pollution.msh"
    metadata = build_clean_baseline_mesh(msh_path, params, with_pollution_layer=True)
    write_mesh_log(gmsh_dir / "pollution.mesh.json", metadata)

    elmergrid = resolve_elmer_grid()
    if not elmergrid:
        raise RuntimeError("ElmerGrid not found (set ELMER_HOME).")
    completed = subprocess.run(
        [str(elmergrid), "14", "2", str(msh_path), "-out", "mesh"],
        cwd=str(elmer_dir),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError("ElmerGrid conversion failed.")
    return elmer_dir / "mesh"


def run_one_sigma(
    sigma: float,
    run_raw_dir: Path,
    elmer_mesh_dir: Path,
    solver_dll: Path,
    ids: dict[str, dict[str, int]],
    frequency: float,
    solver_env: dict[str, str],
    obs_config_base: dict,
) -> dict:
    """Solve one sweep point and return its observable row."""
    tag = "clean" if sigma == 0.0 else f"sigma_{sigma:.3e}"
    point_dir = run_raw_dir / tag
    point_dir.mkdir(parents=True, exist_ok=True)

    # Stage mesh + solver dll next to the SIF (Elmer loads the procedure from cwd).
    mesh_dst = point_dir / "mesh"
    if mesh_dst.exists():
        shutil.rmtree(mesh_dst)
    shutil.copytree(elmer_mesh_dir, mesh_dst)
    shutil.copy2(solver_dll, point_dir / solver_dll.name)

    sif_path = point_dir / "case.sif"
    _render_sif(
        TEMPLATE_SIF,
        sif_path,
        {
            "FREQUENCY": frequency,
            "POLLUTION_SIGMA": repr(sigma),
            "HV_VOLTAGE": repr(HV_VOLTAGE),
            "AIR_BODY_ID": ids["bodies"]["AIR"],
            "PORCELAIN_BODY_ID": ids["bodies"]["PORCELAIN"],
            "POLLUTION_BODY_ID": ids["bodies"]["POLLUTION_LAYER"],
            "HV_BOUNDARY_ID": ids["boundaries"]["HV_TERMINAL"],
            "GROUND_BOUNDARY_ID": ids["boundaries"]["GROUND"],
        },
    )

    solver = resolve_elmer_solver()
    completed = subprocess.run(
        [str(solver), "case.sif"],
        cwd=str(point_dir),
        capture_output=True,
        text=True,
        env=solver_env,
    )
    (point_dir / "elmersolver.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (point_dir / "elmersolver.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout[-2000:])
        raise RuntimeError(f"ElmerSolver failed for sigma={sigma}.")

    vtu_candidates = sorted(point_dir.glob("**/case*.vtu"))
    if not vtu_candidates:
        raise RuntimeError(f"No VTU produced for sigma={sigma}.")
    vtu_path = vtu_candidates[-1]

    omega = 2.0 * np.pi * frequency
    body_materials = {
        ids["bodies"]["AIR"]: (1.0, 0.0),
        ids["bodies"]["PORCELAIN"]: (5.0, 0.0),
        ids["bodies"]["POLLUTION_LAYER"]: (5.0, sigma),
    }
    config = ObservableConfig(
        porcelain_height=obs_config_base["porcelain_height"],
        v_hv=obs_config_base["v_hv"],
        omega=omega,
        body_materials=body_materials,
    )
    obs = compute_observables(vtu_path, config)
    obs["sigma"] = sigma
    obs["sigma_over_we"] = sigma / (omega * 8.8541878128e-12 * 5.0)
    obs["vtu_path"] = str(vtu_path)
    return obs


def main() -> int:
    args = parse_args()
    if not TEMPLATE_SIF.is_file():
        print(f"Missing SIF template: {TEMPLATE_SIF}", file=sys.stderr)
        return 1

    solver_dll = ensure_solver(args.rebuild_solver)

    params = CleanBaselineParameters(
        mesh_size_air=args.mesh_size_air,
        pollution_thickness=args.pollution_thickness,
        mesh_size_pollution=args.mesh_size_pollution,
    )
    sigma_list = args.sigma if args.sigma is not None else default_sigma_list()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_raw_dir = RAW_RESULTS_DIR / f"run_{timestamp}"
    run_processed_dir = PROCESSED_RESULTS_DIR / f"run_{timestamp}"
    run_raw_dir.mkdir(parents=True, exist_ok=True)
    run_processed_dir.mkdir(parents=True, exist_ok=True)

    print("Building + converting pollution mesh ...")
    elmer_mesh_dir = build_and_convert_mesh(run_raw_dir, params)
    ids = _parse_mesh_names(elmer_mesh_dir / "mesh.names")
    for body in ("AIR", "PORCELAIN", "POLLUTION_LAYER"):
        if body not in ids["bodies"]:
            raise RuntimeError(f"Body '{body}' missing from mesh.names: {ids}")

    # Environment for ElmerSolver: ensure its bin (umfpack etc.) is discoverable.
    solver_env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        solver_env["PATH"] = f"{home / 'bin'}{os.pathsep}{solver_env.get('PATH', '')}"

    obs_config_base = dict(
        porcelain_height=params.porcelain_height,
        v_hv=HV_VOLTAGE,
    )

    rows: list[dict] = []
    for sigma in sigma_list:
        print(f"  solving sigma = {sigma:.3e} S/m ...")
        row = run_one_sigma(
            sigma, run_raw_dir, elmer_mesh_dir, solver_dll, ids, args.frequency, solver_env, obs_config_base
        )
        rows.append(row)
        print(
            f"    |Vtap|={row['vtap_abs']:.2f} V  phase={row['phase_deg']:.4f} deg  "
            f"tand={row['tan_delta']:.4e}  Ileak={row['i_leak']:.3e}  Emax={row['emax']:.3e}"
        )

    # Write CSV.
    csv_path = run_raw_dir / "pollution_sweep.csv"
    columns = [
        "sigma", "sigma_over_we", "vtap_abs", "vtap_re", "vtap_im",
        "phase_deg", "tan_delta", "i_leak", "i_terminal", "p_loss", "emax",
    ]
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            fh.write(",".join(f"{row[c]:.10g}" for c in columns) + "\n")
    shutil.copy2(csv_path, run_processed_dir / "pollution_sweep.csv")

    summary = {
        "case": CASE_NAME,
        "timestamp": timestamp,
        "frequency_hz": args.frequency,
        "hv_voltage": HV_VOLTAGE,
        "n_points": len(rows),
        "sigma_list": sigma_list,
        "params": {
            "pollution_thickness": params.pollution_thickness,
            "mesh_size_air": params.mesh_size_air,
            "mesh_size_pollution": params.mesh_size_pollution,
        },
        "body_boundary_ids": ids,
        "csv": str(csv_path),
        "rows": rows,
    }
    (run_raw_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSweep complete: {len(rows)} points")
    print(f"CSV:     {csv_path}")
    print(f"Summary: {run_raw_dir / 'summary.json'}")
    print(f"Plot:    python publication_pipeline/scripts/plot_sweep.py \"{csv_path}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

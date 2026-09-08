from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_pipeline._elmer_tools import resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.clean_baseline import CleanBaselineParameters  # noqa: E402


CASE_NAME = "001_clean_baseline"
TEMPLATE_SIF = ROOT / "elmer" / "sif_templates" / "clean_baseline.sif"
ELMER_MESH_DIR = ROOT / "mesh" / "elmer" / "mesh"
RAW_RESULTS_DIR = ROOT / "results" / "raw" / CASE_NAME
PROCESSED_RESULTS_DIR = ROOT / "results" / "processed" / CASE_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first Elmer case for the publication pipeline.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the latest run directory if it exists.")
    return parser.parse_args()


def _parse_mesh_names(mesh_names_path: Path) -> dict[str, int]:
    boundary_ids: dict[str, int] = {}
    body_ids: dict[str, int] = {}
    section = None
    for line in mesh_names_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("!"):
            lower = stripped.lower()
            if "bodies" in lower:
                section = "body"
            elif "boundaries" in lower:
                section = "boundary"
            continue
        if not stripped.startswith("$") or section is None:
            continue
        match = re.match(r"\$\s*(.*?)\s*=\s*(\d+)$", stripped)
        if not match:
            continue
        name, identifier = match.groups()
        if section == "body":
            body_ids[name] = int(identifier)
        elif section == "boundary":
            boundary_ids[name] = int(identifier)
    return {"bodies": body_ids, "boundaries": boundary_ids}


def _render_sif(template_path: Path, output_path: Path, replacements: dict[str, int]) -> None:
    text = template_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    output_path.write_text(text, encoding="utf-8")


def _prepare_run_directories(overwrite: bool) -> tuple[Path, Path]:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_raw_dir = RAW_RESULTS_DIR / f"run_{timestamp}"
    run_processed_dir = PROCESSED_RESULTS_DIR / f"run_{timestamp}"
    if overwrite:
        shutil.rmtree(run_raw_dir, ignore_errors=True)
        shutil.rmtree(run_processed_dir, ignore_errors=True)
    run_raw_dir.mkdir(parents=True, exist_ok=True)
    run_processed_dir.mkdir(parents=True, exist_ok=True)
    return run_raw_dir, run_processed_dir


def _copy_mesh(run_raw_dir: Path) -> Path:
    target_mesh_dir = run_raw_dir / "mesh"
    if target_mesh_dir.exists():
        shutil.rmtree(target_mesh_dir)
    shutil.copytree(ELMER_MESH_DIR, target_mesh_dir)
    return target_mesh_dir


def _detect_potential_field(vtu: meshio.Mesh) -> tuple[str, np.ndarray]:
    for key, values in vtu.point_data.items():
        if key.lower() == "potential" or "potential" in key.lower():
            return key, np.asarray(values).reshape(-1)
    raise RuntimeError(f"No potential field found in VTU point data: {list(vtu.point_data)}")


def _tap_nodes_from_geometry(points: np.ndarray, tap_height: float, tol: float = 1e-8) -> np.ndarray:
    return np.where(np.abs(points[:, 1] - tap_height) <= tol)[0]


def _write_validation_plot(vtu_path: Path, processed_dir: Path, vtap_value: float, vtap_expected: float) -> Path:
    grid = meshio.read(vtu_path)
    field_name, potential = _detect_potential_field(grid)
    points = np.asarray(grid.points)
    triangles = None
    for cell_block in grid.cells:
        if cell_block.type in {"triangle", "triangle6"}:
            triangles = np.asarray(cell_block.data)
            break
    if triangles is None:
        raise RuntimeError("VTU does not contain triangle cells for plotting")

    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    fig, (ax_field, ax_cmp) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    contour = ax_field.tricontourf(triangulation, potential, levels=24, cmap="viridis")
    fig.colorbar(contour, ax=ax_field, label=f"{field_name} [V]")
    ax_field.set_title("Potential distribution")
    ax_field.set_xlabel("r")
    ax_field.set_ylabel("z")

    tap_nodes = _tap_nodes_from_geometry(points, CleanBaselineParameters().porcelain_height)
    if tap_nodes.size:
        ax_field.scatter(points[tap_nodes, 0], points[tap_nodes, 1], s=18, c="red", label="TAP_ELECTRODE")
        ax_field.legend(loc="best")

    labels = ["Elmer Vtap", "Analytical Vtap"]
    values = [vtap_value, vtap_expected]
    ax_cmp.bar(labels, values, color=["#3b82f6", "#f97316"])
    ax_cmp.set_ylabel("Voltage [V]")
    ax_cmp.set_title("Vtap check")
    ax_cmp.tick_params(axis="x", rotation=20)
    ax_cmp.text(0, vtap_value, f"{vtap_value:.1f}", ha="center", va="bottom")
    ax_cmp.text(1, vtap_expected, f"{vtap_expected:.1f}", ha="center", va="bottom")

    figure_path = processed_dir / "potential_distribution_and_vtap.png"
    fig.suptitle("Clean baseline Elmer validation", fontsize=13)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def main() -> int:
    args = parse_args()
    if not TEMPLATE_SIF.exists():
        print(f"Missing SIF template: {TEMPLATE_SIF}", file=sys.stderr)
        return 1
    if not ELMER_MESH_DIR.exists():
        print(f"Missing converted Elmer mesh: {ELMER_MESH_DIR}", file=sys.stderr)
        return 1

    solver = resolve_elmer_solver()
    if not solver:
        print("ElmerSolver executable not found.", file=sys.stderr)
        return 1

    run_raw_dir, run_processed_dir = _prepare_run_directories(args.overwrite)
    mesh_dir = _copy_mesh(run_raw_dir)
    mesh_names = _parse_mesh_names(mesh_dir / "mesh.names")

    sif_path = run_raw_dir / "case.sif"
    replacements = {
        "HV_BOUNDARY_ID": mesh_names["boundaries"]["HV_TERMINAL"],
        "GROUND_BOUNDARY_ID": mesh_names["boundaries"]["GROUND"],
        "TAP_BOUNDARY_ID": mesh_names["boundaries"]["TAP_ELECTRODE"],
    }
    _render_sif(TEMPLATE_SIF, sif_path, replacements)

    completed = subprocess.run(
        [str(solver), str(sif_path.name)],
        cwd=str(run_raw_dir),
        capture_output=True,
        text=True,
    )

    (run_raw_dir / "elmersolver.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_raw_dir / "elmersolver.stderr.log").write_text(completed.stderr, encoding="utf-8")

    vtu_candidates = sorted((run_raw_dir / "mesh").glob("case*.vtu"))
    if not vtu_candidates:
        vtu_candidates = sorted(run_raw_dir.glob("mesh/case*.vtu"))
    source_vtu_path = vtu_candidates[-1] if vtu_candidates else None
    vtu_path = run_raw_dir / "case.vtu"
    if source_vtu_path and source_vtu_path.exists():
        shutil.copy2(source_vtu_path, vtu_path)

    result = {
        "solver": str(solver),
        "returncode": completed.returncode,
        "success": completed.returncode == 0 and vtu_path.exists(),
        "run_raw_dir": str(run_raw_dir),
        "run_processed_dir": str(run_processed_dir),
        "vtu_source_path": str(source_vtu_path) if source_vtu_path else None,
        "vtu_path": str(vtu_path) if vtu_path.exists() else None,
        "boundary_ids": mesh_names["boundaries"],
    }

    if not result["success"]:
        print("ElmerSolver failed")
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        (run_raw_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return completed.returncode or 1

    vtu = meshio.read(vtu_path)
    field_name, potential = _detect_potential_field(vtu)
    points = np.asarray(vtu.points)
    tap_nodes = _tap_nodes_from_geometry(points, CleanBaselineParameters().porcelain_height)
    if tap_nodes.size == 0:
        print("Could not find TAP_ELECTRODE nodes in the VTU geometry.", file=sys.stderr)
        return 1
    vtap_value = float(np.mean(potential[tap_nodes]))

    params = CleanBaselineParameters()
    air_thickness = params.air_height - params.porcelain_height
    porcelain_thickness = params.porcelain_height
    epsilon_air = 1.0
    epsilon_porcelain = 5.0
    v_hv = 110000.0
    c_air = epsilon_air / air_thickness
    c_porcelain = epsilon_porcelain / porcelain_thickness
    vtap_expected = v_hv * (c_air / (c_air + c_porcelain))

    figure_path = _write_validation_plot(vtu_path, run_processed_dir, vtap_value, vtap_expected)
    summary = {
        **result,
        "potential_field": field_name,
        "vtap_value": vtap_value,
        "vtap_expected": vtap_expected,
        "vtap_error": vtap_value - vtap_expected,
        "validation_figure": str(figure_path),
    }
    (run_raw_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"ElmerSolver path: {solver}")
    print(f"Run directory: {run_raw_dir}")
    print(f"Potential field: {field_name}")
    print(f"Vtap extracted: {vtap_value:.6f} V")
    print(f"Vtap analytical: {vtap_expected:.6f} V")
    print(f"Validation figure: {figure_path}")
    print("ElmerSolver succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

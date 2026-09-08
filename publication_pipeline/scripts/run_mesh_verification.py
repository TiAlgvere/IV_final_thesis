from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
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

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.clean_baseline import (  # noqa: E402
    CleanBaselineParameters,
    build_clean_baseline_mesh,
    write_mesh_log,
)


CASE_NAME = "002_numerical_verification"
GMESH_DIR_NAME = "mesh_gmsh"
ELMER_DIR_NAME = "mesh_elmer"
LEVELS = [
    ("coarse", 1.8),
    ("medium", 1.3),
    ("fine", 1.0),
    ("very_fine", 0.7),
]
TEMPLATE_SIF = ROOT / "elmer" / "sif_templates" / "clean_baseline.sif"
RAW_ROOT = ROOT / "results" / "raw" / CASE_NAME
PROCESSED_ROOT = ROOT / "results" / "processed" / CASE_NAME
CSV_PATH = PROCESSED_ROOT / "mesh_verification.csv"
PLOT_PATH = PROCESSED_ROOT / "vtap_convergence.png"
SUMMARY_JSON_PATH = PROCESSED_ROOT / "richardson_gci.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mesh verification for the clean Elmer baseline.")
    parser.add_argument("--overwrite", action="store_true", help="Remove any existing run folder for this execution.")
    return parser.parse_args()


def _parse_mesh_names(mesh_names_path: Path) -> dict[str, dict[str, int]]:
    bodies: dict[str, int] = {}
    boundaries: dict[str, int] = {}
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
            bodies[name] = int(identifier)
        elif section == "boundary":
            boundaries[name] = int(identifier)
    return {"bodies": bodies, "boundaries": boundaries}


def _render_sif(template_path: Path, output_path: Path, replacements: dict[str, int]) -> None:
    text = template_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    output_path.write_text(text, encoding="utf-8")


def _create_run_root() -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = RAW_ROOT / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    return run_root


def _mesh_header_counts(mesh_header_path: Path) -> tuple[int, int, int]:
    lines = [line.strip() for line in mesh_header_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return 0, 0, 0
    counts = [int(value) for value in re.findall(r"\d+", lines[0])]
    if len(counts) < 3:
        return 0, 0, 0
    return counts[0], counts[1], counts[2]


def _copy_mesh(mesh_source_dir: Path, case_dir: Path) -> Path:
    target_mesh_dir = case_dir / "mesh"
    if target_mesh_dir.exists():
        shutil.rmtree(target_mesh_dir)
    shutil.copytree(mesh_source_dir, target_mesh_dir)
    return target_mesh_dir


def _detect_potential_field(vtu: meshio.Mesh) -> tuple[str, np.ndarray]:
    for key, values in vtu.point_data.items():
        if key.lower() == "potential" or "potential" in key.lower():
            return key, np.asarray(values).reshape(-1)
    raise RuntimeError(f"No potential field found in VTU point data: {list(vtu.point_data)}")


def _tap_nodes(points: np.ndarray, tap_height: float, tol: float = 1e-8) -> np.ndarray:
    return np.where(np.abs(points[:, 1] - tap_height) <= tol)[0]


def _triangle_emax(vtu: meshio.Mesh) -> float | None:
    field_name, potential = _detect_potential_field(vtu)
    _ = field_name
    points = np.asarray(vtu.points)
    triangles = None
    for cell_block in vtu.cells:
        if cell_block.type in {"triangle", "triangle6"}:
            triangles = np.asarray(cell_block.data)
            break
    if triangles is None or triangles.size == 0:
        return None

    emax = 0.0
    for tri in triangles:
        coords = points[np.asarray(tri), :2]
        values = potential[np.asarray(tri)]
        matrix = np.array([
            [coords[0, 0], coords[0, 1], 1.0],
            [coords[1, 0], coords[1, 1], 1.0],
            [coords[2, 0], coords[2, 1], 1.0],
        ])
        try:
            a, b, _ = np.linalg.solve(matrix, values)
        except np.linalg.LinAlgError:
            continue
        e_mag = float(np.hypot(a, b))
        if e_mag > emax:
            emax = e_mag
    return emax


def _compute_analytical_vtap() -> float:
    params = CleanBaselineParameters()
    v_hv = 110000.0
    air_thickness = params.air_height - params.porcelain_height
    porcelain_thickness = params.porcelain_height
    epsilon_air = 1.0
    epsilon_porcelain = 5.0
    c_air = epsilon_air / air_thickness
    c_porcelain = epsilon_porcelain / porcelain_thickness
    return v_hv * (c_air / (c_air + c_porcelain))


def _estimate_richardson_and_gci(valid_rows: list[dict[str, object]]) -> dict[str, object]:
    if len(valid_rows) < 3:
        return {"status": "not stable enough yet", "reason": "fewer than three valid mesh levels"}

    rows = sorted(valid_rows, key=lambda row: float(row["approx_h"]))
    fine, medium, coarse = rows[0], rows[1], rows[2]
    h_f = float(fine["approx_h"])
    h_m = float(medium["approx_h"])
    h_c = float(coarse["approx_h"])
    u_f = float(fine["vtap_fem"])
    u_m = float(medium["vtap_fem"])
    u_c = float(coarse["vtap_fem"])

    dif1 = u_c - u_m
    dif2 = u_m - u_f
    if abs(dif1) < 1e-12 or abs(dif2) < 1e-12:
        return {"status": "not stable enough yet", "reason": "mesh-level differences are too small"}

    r_fm = h_m / h_f
    r_mc = h_c / h_m
    if r_fm <= 1.0 or r_mc <= 1.0:
        return {"status": "not stable enough yet", "reason": "non-refining mesh sequence"}

    if abs(r_fm - r_mc) / max(r_fm, r_mc) > 0.25:
        return {"status": "not stable enough yet", "reason": "refinement ratios are too irregular"}

    ratio = abs(dif1 / dif2)
    if ratio <= 0.0 or math.isnan(ratio) or math.isinf(ratio):
        return {"status": "not stable enough yet", "reason": "invalid solution ratio"}

    r = (r_fm + r_mc) / 2.0
    p = math.log(ratio) / math.log(r)
    if not math.isfinite(p) or p <= 0.0:
        return {"status": "not stable enough yet", "reason": "observed order is not positive"}

    u_ext = u_f + (u_f - u_m) / (r**p - 1.0)
    gci_fine = 1.25 * abs((u_f - u_m) / u_f) / (r**p - 1.0) * 100.0
    if not (math.isfinite(u_ext) and math.isfinite(gci_fine)):
        return {"status": "not stable enough yet", "reason": "computed Richardson values are not finite"}

    return {
        "status": "computed",
        "estimated_asymptotic_vtap": u_ext,
        "observed_order_p": p,
        "gci_fine_percent": gci_fine,
    }


def _write_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    valid_rows = [row for row in rows if row["solver_success"]]
    if not valid_rows:
        return

    valid_rows = sorted(valid_rows, key=lambda row: float(row["approx_h"]))
    x_values = [float(row["approx_h"]) for row in valid_rows]
    y_values = [max(float(row["relative_vtap_error"]), 1e-16) for row in valid_rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.plot(x_values, y_values, marker="o", linewidth=1.8)
    for row in valid_rows:
        ax.annotate(row["mesh_level"], (float(row["approx_h"]), max(float(row["relative_vtap_error"]), 1e-16)), textcoords="offset points", xytext=(5, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Approximate h")
    ax.set_ylabel("Relative Vtap error")
    ax.set_title("Clean baseline mesh verification")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _run_single_level(
    level_name: str,
    lc_scale: float,
    run_root: Path,
    grid_exe: Path,
    solver_exe: Path,
) -> dict[str, object]:
    level_root = run_root / level_name
    gmsh_dir = level_root / GMESH_DIR_NAME
    elmer_dir = level_root / ELMER_DIR_NAME
    case_dir = level_root / "case"
    gmsh_dir.mkdir(parents=True, exist_ok=True)
    elmer_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = gmsh_dir / "clean_baseline.msh"
    mesh_log_path = gmsh_dir / "clean_baseline.mesh.json"
    params = CleanBaselineParameters(lc_scale=lc_scale)
    metadata = build_clean_baseline_mesh(mesh_path, params)
    write_mesh_log(mesh_log_path, metadata)

    conversion = subprocess.run(
        [str(grid_exe), "14", "2", str(mesh_path), "-out", "mesh"],
        cwd=str(elmer_dir),
        capture_output=True,
        text=True,
    )
    (level_root / "elmergrid.stdout.log").write_text(conversion.stdout, encoding="utf-8")
    (level_root / "elmergrid.stderr.log").write_text(conversion.stderr, encoding="utf-8")

    mesh_header = elmer_dir / "mesh" / "mesh.header"
    mesh_names = elmer_dir / "mesh" / "mesh.names"
    if not mesh_header.exists() or not mesh_names.exists():
        return {
            "mesh_level": level_name,
            "lc_scale": lc_scale,
            "solver_success": False,
            "error": "ElmerGrid did not produce a converted mesh",
            "raw_result_folder": str(case_dir),
            "node_count": None,
            "element_count": None,
            "boundary_element_count": None,
            "vtap_fem": None,
            "vtap_analytical": _compute_analytical_vtap(),
            "absolute_error": None,
            "relative_vtap_error": None,
            "emax": None,
            "approx_h": None,
        }

    node_count, element_count, boundary_count = _mesh_header_counts(mesh_header)
    names = _parse_mesh_names(mesh_names)
    _ = names

    _copy_mesh(elmer_dir / "mesh", case_dir)
    case_sif = case_dir / "case.sif"
    replacements = {
        "HV_BOUNDARY_ID": names["boundaries"]["HV_TERMINAL"],
        "GROUND_BOUNDARY_ID": names["boundaries"]["GROUND"],
        "TAP_BOUNDARY_ID": names["boundaries"]["TAP_ELECTRODE"],
    }
    _render_sif(TEMPLATE_SIF, case_sif, replacements)

    solver = subprocess.run(
        [str(solver_exe), str(case_sif.name)],
        cwd=str(case_dir),
        capture_output=True,
        text=True,
    )
    (case_dir / "elmersolver.stdout.log").write_text(solver.stdout, encoding="utf-8")
    (case_dir / "elmersolver.stderr.log").write_text(solver.stderr, encoding="utf-8")

    vtu_candidates = sorted((case_dir / "mesh").glob("case*.vtu"))
    source_vtu = vtu_candidates[-1] if vtu_candidates else None
    vtu_path = case_dir / "case.vtu"
    if source_vtu and source_vtu.exists():
        shutil.copy2(source_vtu, vtu_path)

    row: dict[str, object] = {
        "mesh_level": level_name,
        "lc_scale": lc_scale,
        "node_count": node_count,
        "element_count": element_count,
        "boundary_element_count": boundary_count,
        "vtap_fem": None,
        "vtap_analytical": _compute_analytical_vtap(),
        "absolute_error": None,
        "relative_vtap_error": None,
        "solver_success": solver.returncode == 0 and vtu_path.exists(),
        "raw_result_folder": str(case_dir),
        "approx_h": 1.0 / math.sqrt(element_count) if element_count else None,
        "emax": None,
        "elmergrid_returncode": conversion.returncode,
        "elmersolver_returncode": solver.returncode,
    }

    if not row["solver_success"]:
        row["error"] = "ElmerSolver failed or VTU missing"
        return row

    vtu = meshio.read(vtu_path)
    field_name, potential = _detect_potential_field(vtu)
    points = np.asarray(vtu.points)
    tap_nodes = _tap_nodes(points, CleanBaselineParameters().porcelain_height)
    if tap_nodes.size == 0:
        row["solver_success"] = False
        row["error"] = "Could not locate TAP_ELECTRODE nodes in VTU"
        return row

    vtap_fem = float(np.mean(potential[tap_nodes]))
    vtap_analytical = float(row["vtap_analytical"])
    abs_error = abs(vtap_fem - vtap_analytical)
    rel_error = abs_error / abs(vtap_analytical) if vtap_analytical else None
    row.update(
        {
            "vtap_fem": vtap_fem,
            "absolute_error": abs_error,
            "relative_vtap_error": rel_error,
            "emax": _triangle_emax(vtu),
            "potential_field": field_name,
        }
    )
    row["error"] = None

    summary = {
        **row,
        "solver_stdout": str(case_dir / "elmersolver.stdout.log"),
        "solver_stderr": str(case_dir / "elmersolver.stderr.log"),
        "vtu_path": str(vtu_path),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return row


def main() -> int:
    args = parse_args()
    if not TEMPLATE_SIF.exists():
        print(f"Missing SIF template: {TEMPLATE_SIF}", file=sys.stderr)
        return 1

    grid_exe = resolve_elmer_grid()
    solver_exe = resolve_elmer_solver()
    if not grid_exe or not solver_exe:
        print("Missing ElmerGrid or ElmerSolver executable. Set ELMER_HOME or add the binaries to PATH.", file=sys.stderr)
        return 1

    if args.overwrite:
        shutil.rmtree(RAW_ROOT, ignore_errors=True)
        shutil.rmtree(PROCESSED_ROOT, ignore_errors=True)

    run_root = _create_run_root()
    rows: list[dict[str, object]] = []
    for level_name, lc_scale in LEVELS:
        row = _run_single_level(level_name, lc_scale, run_root, grid_exe, solver_exe)
        rows.append(row)

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    valid_rows = [row for row in rows if row.get("solver_success")]
    for row in rows:
        row.setdefault("richardson_status", "not computed yet")
        row.setdefault("estimated_asymptotic_vtap", None)
        row.setdefault("observed_order_p", None)
        row.setdefault("gci_fine_percent", None)

    richardson = _estimate_richardson_and_gci(valid_rows)
    if richardson.get("status") == "computed":
        for row in rows:
            row["richardson_status"] = "computed"
            row["estimated_asymptotic_vtap"] = richardson["estimated_asymptotic_vtap"]
            row["observed_order_p"] = richardson["observed_order_p"]
            row["gci_fine_percent"] = richardson["gci_fine_percent"]
    else:
        for row in rows:
            row["richardson_status"] = richardson.get("status", "not stable enough yet")

    csv_fields = [
        "mesh_level",
        "lc_scale",
        "node_count",
        "element_count",
        "boundary_element_count",
        "approx_h",
        "vtap_fem",
        "vtap_analytical",
        "absolute_error",
        "relative_vtap_error",
        "emax",
        "solver_success",
        "elmergrid_returncode",
        "elmersolver_returncode",
        "raw_result_folder",
        "richardson_status",
        "estimated_asymptotic_vtap",
        "observed_order_p",
        "gci_fine_percent",
        "error",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in csv_fields})

    _write_plot(rows, PLOT_PATH)
    SUMMARY_JSON_PATH.write_text(json.dumps(richardson, indent=2), encoding="utf-8")

    print(f"CSV written to: {CSV_PATH}")
    print(f"Plot written to: {PLOT_PATH}")
    print(f"Richardson/GCI status: {richardson.get('status', 'not stable enough yet')}")
    for row in rows:
        print(
            f"{row['mesh_level']}: solver_success={row['solver_success']}, node_count={row['node_count']}, element_count={row['element_count']}, "
            f"vtap_fem={row['vtap_fem']}, relative_error={row['relative_vtap_error']}, raw_result_folder={row['raw_result_folder']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

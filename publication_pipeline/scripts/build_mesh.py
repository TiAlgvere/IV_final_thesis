from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from publication_pipeline.geometry.gmsh.clean_baseline import (  # noqa: E402
    CleanBaselineParameters,
    build_clean_baseline_mesh,
    write_mesh_log,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the clean baseline Gmsh mesh.")
    parser.add_argument(
        "--output-msh",
        type=Path,
        default=ROOT / "mesh" / "gmsh" / "clean_baseline.msh",
        help="Target .msh path inside publication_pipeline/mesh/gmsh/",
    )
    parser.add_argument(
        "--log-json",
        type=Path,
        default=ROOT / "mesh" / "gmsh" / "clean_baseline.mesh.json",
        help="Mesh parameter log written next to the mesh.",
    )
    parser.add_argument("--mesh-size-air", type=float, default=0.025)
    parser.add_argument("--mesh-size-porcelain", type=float, default=0.015)
    parser.add_argument("--mesh-size-electrode", type=float, default=0.01)
    parser.add_argument("--lc-scale", type=float, default=1.0, help="Uniform mesh-size scale factor.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = CleanBaselineParameters(
        lc_scale=args.lc_scale,
        mesh_size_air=args.mesh_size_air,
        mesh_size_porcelain=args.mesh_size_porcelain,
        mesh_size_electrode=args.mesh_size_electrode,
    )
    metadata = build_clean_baseline_mesh(args.output_msh, params)
    write_mesh_log(args.log_json, metadata)
    print(f"Wrote mesh: {args.output_msh}")
    print(f"Wrote mesh log: {args.log_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

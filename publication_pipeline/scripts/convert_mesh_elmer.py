from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_pipeline._elmer_tools import resolve_elmer_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Gmsh mesh to Elmer format.")
    parser.add_argument(
        "--input-msh",
        type=Path,
        default=ROOT / "mesh" / "gmsh" / "clean_baseline.msh",
        help="Input .msh file produced by build_mesh.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "mesh" / "elmer",
        help="Destination directory for the Elmer mesh.",
    )
    parser.add_argument(
        "--elmergrid",
        default=None,
        help="Optional explicit ElmerGrid executable name or path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_msh.exists():
        print(f"Missing input mesh: {args.input_msh}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.elmergrid:
        elmergrid = Path(args.elmergrid)
        if not elmergrid.exists():
            print(f"ElmerGrid executable not found: {elmergrid}", file=sys.stderr)
            return 1
    else:
        elmergrid = resolve_elmer_grid()

    if not elmergrid:
        print(
            "ElmerGrid executable not found. Set ELMER_HOME to the Elmer install root, "
            "install ElmerGrid on PATH, or pass --elmergrid with an explicit path.",
            file=sys.stderr,
        )
        return 1

    command = [str(elmergrid), "14", "2", str(args.input_msh), "-out", "mesh"]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(args.output_dir))
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        return completed.returncode

    print(f"Converted mesh written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

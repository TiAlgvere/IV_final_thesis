from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_pipeline._elmer_tools import (  # noqa: E402
    resolve_elmer_build_config,
    resolve_elmer_grid,
    resolve_elmer_solver,
)


def _module_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return "MISSING"

    version = getattr(module, "__version__", None)
    if version:
        return str(version)

    try:
        return importlib.metadata.version(module_name)
    except Exception:
        return "MISSING"


def _active_venv() -> str:
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return venv
    if sys.prefix != sys.base_prefix:
        return sys.prefix
    return "MISSING"


def _warn_if_elmer_home_is_incomplete(elmer_home: str) -> None:
    if not elmer_home or elmer_home == "MISSING":
        return

    grid_path = Path(elmer_home) / "bin" / "ElmerGrid.exe"
    if not grid_path.is_file():
        print(f"Warning: ELMER_HOME is set but {grid_path} is missing")


def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.splitlines()[0]}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Active venv: {_active_venv()}")
    print(f"gmsh: {_module_version('gmsh')}")
    print(f"meshio: {_module_version('meshio')}")

    elmer_home = os.environ.get("ELMER_HOME", "MISSING")
    print(f"ELMER_HOME: {elmer_home}")
    _warn_if_elmer_home_is_incomplete(elmer_home)

    elmer_grid = resolve_elmer_grid()
    elmer_solver = resolve_elmer_solver()
    print(f"ElmerGrid: {elmer_grid if elmer_grid else 'MISSING'}")
    print(f"ElmerSolver: {elmer_solver if elmer_solver else 'MISSING'}")

    # Custom-solver build toolchain (Task 006 ComplexEQS). NOTE: the shipped
    # elmerf90.exe is broken on this install; we build via the bundled gfortran.
    build = resolve_elmer_build_config()
    if build is None:
        print("Solver build toolchain: MISSING (cannot compile custom .F90 solvers)")
    else:
        print(f"Fortran (custom solvers): {build.fortran}")
        print(f"  Elmer include dir: {build.include_dir}")
        print(f"  Elmer lib dir:     {build.lib_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

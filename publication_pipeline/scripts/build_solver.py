"""Compile a custom Elmer solver module into a loadable shared library on Windows.

The Elmer Windows distribution ships an `elmerf90.exe` wrapper, but on this
install it is unusable: it has absolute paths baked in from the original build
machine (`D:/msys64/mingw64/bin/gfortran.exe`, `C:/Program Files (x86)/Elmer/...`)
that do not exist here, so it fails with "exec: No such file or directory".

This script bypasses the wrapper and invokes the gfortran bundled with Elmer
(`stripped_gfortran/bin/x86_64-w64-mingw32-gfortran.exe`, GNU Fortran 10.2.0,
ABI-matched to Elmer's compiled `.mod` files) directly, replicating the compile
flags that `elmerf90` itself uses but with the corrected local include/lib paths.

Usage:
    python scripts/build_solver.py                  # builds elmer/solvers/ComplexEQS.F90
    python scripts/build_solver.py path/to/My.F90   # builds an arbitrary source
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_pipeline._elmer_tools import ElmerBuildConfig, resolve_elmer_build_config  # noqa: E402

SOLVER_DIR = ROOT / "elmer" / "solvers"
DEFAULT_SOURCE = SOLVER_DIR / "ComplexEQS.F90"

# Flags mirror what `elmerf90` passes to gfortran (minus -DUSE_ARPACK, which our
# solver does not need). `-shared` produces a loadable DLL; `-lelmersolver` links
# against Elmer's import library so DefUtils & friends resolve.
GFORTRAN_FLAGS = [
    "-fallow-argument-mismatch",
    "-DCONTIG=",
    "-DMINGW32",
    "-DWIN32",
    "-DHAVE_EXECUTECOMMANDLINE",
    "-DUSE_ISO_C_BINDINGS",
    "-O3",
    "-DNDEBUG",
    "-shared",
]


def build_solver(source: Path, output: Path | None = None, config: ElmerBuildConfig | None = None) -> Path:
    """Compile ``source`` (.F90) into a shared library and return its path."""
    cfg = config or resolve_elmer_build_config()
    if cfg is None:
        raise RuntimeError(
            "Could not resolve the Elmer build toolchain (gfortran + include/lib). "
            "Set ELMER_HOME to the Elmer install root."
        )

    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Solver source not found: {source}")
    output = Path(output).resolve() if output is not None else source.with_suffix(".dll")

    env = os.environ.copy()
    # The bundled gfortran (and the linker stage) need their own bin dir AND
    # Elmer's bin dir on PATH to find backend/runtime DLLs (libgfortran etc.).
    env["PATH"] = f"{cfg.path_prefix}{os.pathsep}{env.get('PATH', '')}"

    cmd = [
        str(cfg.fortran),
        source.name,
        "-o",
        str(output),
        *GFORTRAN_FLAGS,
        f"-I{cfg.include_dir}",
        f"-L{cfg.lib_dir}",
        "-lelmersolver",
    ]

    print(f"Compiler: {cfg.fortran}")
    print(f"Include : {cfg.include_dir}")
    print(f"Lib     : {cfg.lib_dir}")
    print(f"Command : {' '.join(cmd)}")

    completed = subprocess.run(
        cmd,
        cwd=str(source.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)

    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"Solver build failed (exit {completed.returncode}).")

    print(f"Built: {output} ({output.stat().st_size} bytes)")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile a custom Elmer solver (.F90 -> .dll).")
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_SOURCE),
        help="Path to the .F90 solver source (default: elmer/solvers/ComplexEQS.F90).",
    )
    parser.add_argument("-o", "--output", default=None, help="Output shared-library path.")
    args = parser.parse_args()

    try:
        build_solver(Path(args.source), Path(args.output) if args.output else None)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the CLI
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

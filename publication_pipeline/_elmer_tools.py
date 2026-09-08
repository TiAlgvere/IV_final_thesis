from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_WINDOWS_INSTALL = Path(r"C:\ElmerFEM-gui-nompi-Windows-AMD64")
DEFAULT_LEGACY_ROOT = Path(r"C:\ElmerFEM")
DEFAULT_RELEASE_DIR = REPO_ROOT / "elmerfem-release-26.2"

# Name of the Fortran compiler bundled with the Windows Elmer distribution. The
# shipped `elmerf90.exe` wrapper is NOT usable on this install: it has absolute
# paths baked in from the build machine (D:/msys64/mingw64, C:/Program Files
# (x86)/Elmer) that do not exist here. We therefore call this bundled gfortran
# (GNU Fortran 10.2.0, ABI-matched to Elmer's .mod files) directly instead.
BUNDLED_FORTRAN_RELPATH = Path("stripped_gfortran") / "bin" / "x86_64-w64-mingw32-gfortran.exe"


def _exe_names(executable_name: str) -> list[str]:
    if executable_name.lower().endswith(".exe"):
        return [executable_name]
    return [executable_name, f"{executable_name}.exe"]


def _path_candidates(base: Path, executable_name: str) -> list[Path]:
    candidates: list[Path] = []
    for exe_name in _exe_names(executable_name):
        candidates.append(base / "bin" / exe_name)
    return candidates


def resolve_elmer_executable(executable_name: str) -> Path | None:
    search_roots: list[Path] = []
    elmer_home = os.environ.get("ELMER_HOME")
    if elmer_home:
        search_roots.append(Path(elmer_home))

    path = shutil.which(executable_name)
    if path:
        search_roots.append(Path(path).parent.parent)

    search_roots.append(DEFAULT_WINDOWS_INSTALL)
    search_roots.append(DEFAULT_LEGACY_ROOT)
    search_roots.append(DEFAULT_RELEASE_DIR)

    for root in search_roots:
        if root.is_file():
            return root
        for candidate in _path_candidates(root, executable_name):
            if candidate.is_file():
                return candidate

    return None


def resolve_elmer_grid() -> Path | None:
    return resolve_elmer_executable("ElmerGrid")


def resolve_elmer_solver() -> Path | None:
    return resolve_elmer_executable("ElmerSolver")


def resolve_elmer_home() -> Path | None:
    """Resolve the Elmer installation root directory (the dir holding bin/lib/share)."""
    elmer_home = os.environ.get("ELMER_HOME")
    if elmer_home and Path(elmer_home).is_dir():
        return Path(elmer_home)

    # Derive from a resolved executable: <root>/bin/ElmerSolver.exe -> <root>.
    for resolver in (resolve_elmer_solver, resolve_elmer_grid):
        exe = resolver()
        if exe is not None:
            return exe.parent.parent

    for root in (DEFAULT_WINDOWS_INSTALL, DEFAULT_LEGACY_ROOT, DEFAULT_RELEASE_DIR):
        if root.is_dir():
            return root
    return None


def resolve_elmer_fortran() -> Path | None:
    """Resolve a Fortran compiler usable for building custom Elmer solver modules.

    Prefers the gfortran bundled with the Windows Elmer distribution (its version
    matches Elmer's compiled .mod files). Falls back to a gfortran on PATH.
    """
    home = resolve_elmer_home()
    if home is not None:
        candidate = home / BUNDLED_FORTRAN_RELPATH
        if candidate.is_file():
            return candidate

    which = shutil.which("x86_64-w64-mingw32-gfortran") or shutil.which("gfortran")
    return Path(which) if which else None


@dataclass(frozen=True)
class ElmerBuildConfig:
    """Everything needed to compile a custom Elmer solver shared library."""

    fortran: Path
    include_dir: Path  # holds Elmer's .mod files (DefUtils etc.)
    lib_dir: Path  # holds libelmersolver import library
    compiler_bin_dir: Path  # bundled gfortran's bin dir
    elmer_bin_dir: Path  # Elmer's bin dir (libgfortran-5.dll, libelmersolver runtime, ...)

    @property
    def path_prefix(self) -> str:
        """PATH entries (os.pathsep-joined) the compiler and linker need to find DLLs."""
        return os.pathsep.join(str(p) for p in (self.compiler_bin_dir, self.elmer_bin_dir))


def resolve_elmer_build_config() -> ElmerBuildConfig | None:
    """Resolve the full toolchain for compiling solver modules, or None if incomplete."""
    home = resolve_elmer_home()
    fortran = resolve_elmer_fortran()
    if home is None or fortran is None:
        return None

    include_dir = home / "share" / "elmersolver" / "include"
    lib_dir = home / "lib" / "elmersolver"
    if not include_dir.is_dir() or not lib_dir.is_dir():
        return None

    return ElmerBuildConfig(
        fortran=fortran,
        include_dir=include_dir,
        lib_dir=lib_dir,
        compiler_bin_dir=fortran.parent,
        elmer_bin_dir=home / "bin",
    )

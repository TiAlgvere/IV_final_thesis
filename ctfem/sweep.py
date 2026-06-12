"""Parametric sweep engine -> tidy parquet/CSV dataset (Phase 5).

Generates a list of :class:`~ctfem.config.CaseConfig`, solves each independently
(optionally across processes), and writes one row per case with full provenance
(all parameters + git hash + mesh stats) so the dataset is reproducible and
self-describing.  Supports Latin-hypercube and Cartesian designs, parallel
workers (multiprocessing), and resume-after-interruption (already-completed case
ids are skipped).

Each worker builds its OWN mesh in a temp dir and runs a single serial solve, so
no MPI is needed at this level (per-solve MPI can be added later).  This keeps the
sweep embarrassingly parallel and HPC-array-job friendly.

Schema (documented; one row per case)
-------------------------------------
  case_id, name, defect_kind, severity, defect_index, z_center, extent,
  n_foils, foil_placement, mesh_refinement, frequency, um_kv, temperature_c,
  git_hash, n_triangles, n_nodes, solve_seconds,
  C1_pF, tan_delta, Y_real, Y_imag, admittance_discrepancy,
  peak_field_overall, foil{1..N}_frac, gap{1..N}_peakE
"""
from __future__ import annotations

import dataclasses
import itertools
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace, asdict
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    CaseConfig, GeometryParams, OperatingParams, MaterialParams, DefectSpec,
)


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def git_hash(default: str = "nogit") -> str:
    """Short git hash of the working tree, or `default` if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL)
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL)
        return out.decode().strip() + ("-dirty" if dirty else "")
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# design generation
# --------------------------------------------------------------------------- #


def _case_id(cfg: CaseConfig) -> str:
    d = cfg.defect
    return (f"{d.kind}__sev{d.severity:.3f}__idx{d.index}"
            f"__z{d.z_center:.3f}__N{cfg.geometry.n_foils}"
            f"__ref{cfg.geometry.mesh_refinement:.2f}"
            f"__T{cfg.operating.temperature_c:.1f}")


def cartesian_design(
    base: CaseConfig,
    defect_kinds: Sequence[str],
    severities: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    foil_indices: Sequence[int] = (1, 2, 3),
    z_centers: Sequence[float] = (1.45,),
    temperatures: Sequence[float] = (20.0,),
) -> list[CaseConfig]:
    """Cartesian product design over defect parameters."""
    cases: list[CaseConfig] = []
    for kind in defect_kinds:
        if kind == "shorted_foil":
            for idx, T in itertools.product(foil_indices, temperatures):
                cases.append(_mk(base, kind, 1.0, idx, base.defect.z_center, T))
        elif kind == "none":
            for T in temperatures:
                cases.append(_mk(base, "none", 0.0, 0, base.defect.z_center, T))
        elif kind == "moisture_ingress":
            # only moisture depends on axial location -> vary z here
            for sev, zc, T in itertools.product(severities, z_centers, temperatures):
                if sev == 0.0:
                    continue
                cases.append(_mk(base, kind, sev, 0, zc, T))
        else:  # global_aging, oil_contamination -- location-independent
            for sev, T in itertools.product(severities, temperatures):
                if sev == 0.0:
                    continue
                cases.append(_mk(base, kind, sev, 0, base.defect.z_center, T))
    return _dedup(cases)


def lhs_design(
    base: CaseConfig,
    defect_kinds: Sequence[str],
    n_samples: int,
    seed: int = 0,
    severity_range: tuple[float, float] = (0.05, 1.0),
    z_range: tuple[float, float] = (0.8, 2.2),
    temp_range: tuple[float, float] = (10.0, 80.0),
) -> list[CaseConfig]:
    """Latin-hypercube design (deterministic via `seed`)."""
    rng = np.random.default_rng(seed)
    cases: list[CaseConfig] = []
    n_per = max(1, n_samples // max(1, len(defect_kinds)))
    for kind in defect_kinds:
        # LHS over (severity, z, T) in unit cube
        cube = _lhs_unit(n_per, 3, rng)
        sev = severity_range[0] + cube[:, 0] * (severity_range[1] - severity_range[0])
        zc = z_range[0] + cube[:, 1] * (z_range[1] - z_range[0])
        temp = temp_range[0] + cube[:, 2] * (temp_range[1] - temp_range[0])
        for i in range(n_per):
            if kind == "shorted_foil":
                idx = int(rng.integers(1, base.geometry.n_foils))
                cases.append(_mk(base, kind, 1.0, idx, zc[i], temp[i]))
            elif kind == "none":
                cases.append(_mk(base, "none", 0.0, 0, base.defect.z_center, temp[i]))
            else:
                cases.append(_mk(base, kind, float(sev[i]), 0, float(zc[i]), float(temp[i])))
    return _dedup(cases)


def _lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-hypercube samples in the unit cube [0,1]^d."""
    out = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / n
    return out


def _mk(base: CaseConfig, kind: str, sev: float, idx: int, zc: float,
        T: float) -> CaseConfig:
    defect = replace(base.defect, kind=kind, severity=sev, index=idx, z_center=zc)
    operating = replace(base.operating, temperature_c=T)
    materials = replace(base.materials, temperature_c=T)
    cfg = replace(base, defect=defect, operating=operating, materials=materials)
    return replace(cfg, name=defect.label())


def _dedup(cases: Sequence[CaseConfig]) -> list[CaseConfig]:
    seen: dict[str, CaseConfig] = {}
    for c in cases:
        seen.setdefault(_case_id(c), c)
    return list(seen.values())


# --------------------------------------------------------------------------- #
# single-case solve (worker entry point -- must be top-level for pickling)
# --------------------------------------------------------------------------- #


def solve_case(cfg: CaseConfig) -> dict:
    """Build a mesh + solve one case; return a flat provenance+results row.

    Solver/observables are imported lazily; the backend is auto-detected
    (DOLFINx where available, otherwise the Windows-native scikit-fem backend),
    and recorded in the row for provenance.
    """
    from .geometry import build_ct
    from .common import run_case_2d, detect_backend

    backend = detect_backend("auto")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ct.msh")
        mres = build_ct(cfg.geometry, path, verbose=False)
        obs = run_case_2d(
            path, cfg.operating, materials=cfg.materials,
            defect=cfg.defect, geometry=cfg.geometry, backend=backend)
    dt = time.time() - t0

    row = {
        "case_id": _case_id(cfg),
        "name": cfg.name,
        "defect_kind": cfg.defect.kind,
        "severity": cfg.defect.severity,
        "defect_index": cfg.defect.index,
        "z_center": cfg.defect.z_center,
        "extent": cfg.defect.extent,
        "n_foils": cfg.geometry.n_foils,
        "foil_placement": cfg.geometry.foil_placement,
        "mesh_refinement": cfg.geometry.mesh_refinement,
        "frequency": cfg.operating.frequency,
        "um_kv": cfg.operating.um_kv,
        "temperature_c": cfg.operating.temperature_c,
        "git_hash": git_hash(),
        "backend": backend,
        "n_triangles": mres.n_triangles,
        "n_nodes": mres.n_nodes,
        "solve_seconds": dt,
    }
    row.update(obs.row())
    return row


# --------------------------------------------------------------------------- #
# sweep driver (parallel + resume)
# --------------------------------------------------------------------------- #


def run_sweep(
    cases: Sequence[CaseConfig],
    out_path: str,
    *,
    workers: int = 1,
    resume: bool = True,
    progress: bool = True,
) -> "pd.DataFrame":
    """Solve all cases and write a parquet (or CSV) dataset; supports resume.

    Already-completed case ids (present in an existing output file) are skipped
    when `resume` is True, so an interrupted sweep can be restarted.
    """
    existing_rows: list[dict] = []
    done_ids: set[str] = set()
    if resume and os.path.exists(out_path):
        prev = _read_any(out_path)
        existing_rows = prev.to_dict("records")
        done_ids = set(prev["case_id"].tolist())

    todo = [c for c in cases if _case_id(c) not in done_ids]
    if progress:
        print(f"[sweep] {len(cases)} cases, {len(done_ids)} done, "
              f"{len(todo)} to solve, workers={workers}")

    rows: list[dict] = list(existing_rows)

    def _flush() -> None:
        df = pd.DataFrame(rows)
        _write_any(df, out_path)

    if workers <= 1:
        for i, cfg in enumerate(todo):
            rows.append(solve_case(cfg))
            if progress:
                print(f"[sweep] {i + 1}/{len(todo)} {rows[-1]['case_id']} "
                      f"C1={rows[-1]['C1_pF']:.1f}pF tand={rows[-1]['tan_delta']:.4f}")
            _flush()  # checkpoint after every case -> resumable
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, row in enumerate(pool.imap_unordered(solve_case, todo)):
                rows.append(row)
                if progress:
                    print(f"[sweep] {i + 1}/{len(todo)} {row['case_id']} "
                          f"C1={row['C1_pF']:.1f}pF")
                _flush()

    df = pd.DataFrame(rows)
    _write_any(df, out_path)
    return df


def _read_any(path: str) -> "pd.DataFrame":
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_any(df: "pd.DataFrame", path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".parquet"):
        try:
            df.to_parquet(path, index=False)
            return
        except Exception:
            # fall back to CSV if no parquet engine is available
            path = path[: -len(".parquet")] + ".csv"
    df.to_csv(path, index=False)

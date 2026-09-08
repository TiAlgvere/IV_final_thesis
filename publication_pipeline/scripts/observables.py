"""Diagnostic-observable extraction from a complex-EQS Elmer solution.

Given a VTU written by the ComplexEQS solver (nodal fields ``potential re`` and
``potential im``), compute the CVT fault-diagnosis observables as functions of the
complex electric field E = -grad(phi), phi = phi_re + i*phi_im:

  * |Vtap|, phase displacement d-theta  - magnitude / phase of the tap potential
  * tan(delta)                          - global loss tangent = int(sigma|E|^2) / (w int(eps|E|^2))
  * leakage current I_leak / I_terminal - in-phase and total terminal current
  * P_loss                              - time-avg dissipated power (per unit depth)
  * Emax                                - peak field magnitude |E|

Per-element material properties (eps_r, sigma) are assigned from the per-cell
**body id** that Elmer writes into the VTU as ``GeometryIds``, via a caller-supplied
``body_materials`` map. This is exact for any geometry (any pollution strip shape,
coverage or position), unlike geometric centroid classification.

All integrals are per unit depth (the model is solved as Cartesian 2-D). The
factors of 1/2 distinguishing phasor-amplitude time-averages cancel in tan(delta);
they are kept explicitly in P_loss and the terminal currents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np

EPS0 = 8.8541878128e-12


@dataclass(frozen=True)
class ObservableConfig:
    """Geometry + material constants needed to interpret the field."""

    porcelain_height: float  # z of the air/porcelain (tap) interface [m]
    v_hv: float  # HV terminal voltage magnitude [V]
    omega: float  # angular frequency [rad/s]
    body_materials: dict[int, tuple[float, float]]  # body_id -> (eps_r, sigma [S/m])
    tap_tol: float = 1e-6


def _read_mesh(vtu_path: Path):
    """Return (points, triangles, triangle_body_ids, phi_re, phi_im)."""
    mesh = meshio.read(str(vtu_path))
    points = np.asarray(mesh.points, dtype=float)

    pd = {k.lower(): np.asarray(v, dtype=float).reshape(-1) for k, v in mesh.point_data.items()}
    phi_re = pd["potential re"]
    phi_im = pd["potential im"]

    tris = None
    gids = None
    geom = mesh.cell_data.get("GeometryIds")
    for i, block in enumerate(mesh.cells):
        if block.type in {"triangle", "triangle6"}:
            tris = np.asarray(block.data, dtype=int)[:, :3]
            if geom is not None:
                gids = np.asarray(geom[i], dtype=int).reshape(-1)
            break
    if tris is None:
        raise RuntimeError("VTU does not contain triangle cells")
    if gids is None:
        raise RuntimeError("VTU does not contain per-cell GeometryIds (body ids)")
    return points, tris, gids, phi_re, phi_im


def _triangle_gradients(points: np.ndarray, tris: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle constant gradient (Ntri,2) and area (Ntri,) for a P1 field."""
    p = points[:, :2]
    x1, x2, x3 = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    f1, f2, f3 = values[tris[:, 0]], values[tris[:, 1]], values[tris[:, 2]]

    detT = (x2[:, 0] - x1[:, 0]) * (x3[:, 1] - x1[:, 1]) - (x3[:, 0] - x1[:, 0]) * (x2[:, 1] - x1[:, 1])
    b = np.column_stack([x2[:, 1] - x3[:, 1], x3[:, 1] - x1[:, 1], x1[:, 1] - x2[:, 1]])
    c = np.column_stack([x3[:, 0] - x2[:, 0], x1[:, 0] - x3[:, 0], x2[:, 0] - x1[:, 0]])
    f = np.column_stack([f1, f2, f3])

    dfdx = np.sum(b * f, axis=1) / detT
    dfdy = np.sum(c * f, axis=1) / detT
    area = 0.5 * np.abs(detT)
    return np.column_stack([dfdx, dfdy]), area


def compute_observables(vtu_path: str | Path, config: ObservableConfig) -> dict[str, float]:
    points, tris, gids, phi_re, phi_im = _read_mesh(Path(vtu_path))

    grad_re, area = _triangle_gradients(points, tris, phi_re)
    grad_im, _ = _triangle_gradients(points, tris, phi_im)
    e2 = np.sum(grad_re**2, axis=1) + np.sum(grad_im**2, axis=1)  # |E|^2 = |grad re|^2 + |grad im|^2
    e_mag = np.sqrt(e2)

    missing = set(np.unique(gids)) - set(config.body_materials)
    if missing:
        raise KeyError(f"body ids {sorted(missing)} have no entry in body_materials {config.body_materials}")
    epsr = np.array([config.body_materials[g][0] for g in gids])
    sigma = np.array([config.body_materials[g][1] for g in gids])

    int_sigma_e2 = float(np.sum(sigma * e2 * area))
    int_eps_e2 = float(np.sum(EPS0 * epsr * e2 * area))

    tan_delta = int_sigma_e2 / (config.omega * int_eps_e2) if int_eps_e2 > 0 else 0.0
    p_loss = 0.5 * int_sigma_e2  # time-avg dissipated power per unit depth [W/m]
    emax = float(e_mag.max()) if e_mag.size else 0.0

    omega_eps_e2 = config.omega * int_eps_e2
    i_leak = int_sigma_e2 / config.v_hv if config.v_hv else 0.0  # in-phase current [A/m depth]
    i_terminal = float(np.hypot(int_sigma_e2, omega_eps_e2)) / config.v_hv if config.v_hv else 0.0

    tap = np.where(np.abs(points[:, 1] - config.porcelain_height) <= config.tap_tol)[0]
    if tap.size == 0:
        raise RuntimeError("no tap-electrode nodes found at the porcelain interface")
    vtap_re = float(np.mean(phi_re[tap]))
    vtap_im = float(np.mean(phi_im[tap]))
    vtap_abs = float(np.hypot(vtap_re, vtap_im))
    phase_deg = float(np.degrees(np.arctan2(vtap_im, vtap_re)))

    return {
        "vtap_re": vtap_re,
        "vtap_im": vtap_im,
        "vtap_abs": vtap_abs,
        "phase_deg": phase_deg,
        "tan_delta": tan_delta,
        "p_loss": p_loss,
        "i_leak": i_leak,
        "i_terminal": i_terminal,
        "emax": emax,
        "int_sigma_e2": int_sigma_e2,
        "int_eps_e2": int_eps_e2,
        "n_tap_nodes": int(tap.size),
    }

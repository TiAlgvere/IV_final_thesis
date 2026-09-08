"""Observables for the representative CVT solve (Task 009).

From a complex-EQS axisymmetric VTU (fields ``potential re``/``potential im`` and
per-cell ``GeometryIds`` body ids), compute:

  * terminal capacitance  C = (1/U0^2) INT eps0 eps_r |E|^2 (2 pi r) dr dz
    (energy method; only the real permittivity stores energy, so loss tanδ does
    not enter C). This is the HV-to-ground capacitance of the series stack.
  * intermediate-voltage tap: complex phi on the tap disc body -> |Vtap|, phase,
    and the divider ratio |Vtap|/U0.

Per-element eps_r comes from the body-id -> eps_r map (built from mesh.names), so
the classification is exact for the multi-region CVT.
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np

EPS0 = 8.8541878128e-12


def _read(vtu_path: Path):
    m = meshio.read(str(vtu_path))
    pts = np.asarray(m.points, dtype=float)
    pd = {k.lower(): np.asarray(v, dtype=float).reshape(-1) for k, v in m.point_data.items()}
    re, im = pd["potential re"], pd["potential im"]
    geom = m.cell_data.get("GeometryIds")
    tris = gids = None
    for i, b in enumerate(m.cells):
        if b.type in {"triangle", "triangle6"}:
            tris = np.asarray(b.data, dtype=int)[:, :3]
            gids = np.asarray(geom[i], dtype=int).reshape(-1) if geom is not None else None
            break
    if tris is None or gids is None:
        raise RuntimeError("VTU missing triangles or GeometryIds")
    return pts, tris, gids, re, im


def _grad_area(points: np.ndarray, tris: np.ndarray, values: np.ndarray):
    p = points[:, :2]
    x1, x2, x3 = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    f1, f2, f3 = values[tris[:, 0]], values[tris[:, 1]], values[tris[:, 2]]
    det = (x2[:, 0] - x1[:, 0]) * (x3[:, 1] - x1[:, 1]) - (x3[:, 0] - x1[:, 0]) * (x2[:, 1] - x1[:, 1])
    b = np.column_stack([x2[:, 1] - x3[:, 1], x3[:, 1] - x1[:, 1], x1[:, 1] - x2[:, 1]])
    c = np.column_stack([x3[:, 0] - x2[:, 0], x1[:, 0] - x3[:, 0], x2[:, 0] - x1[:, 0]])
    f = np.column_stack([f1, f2, f3])
    dfdx = np.sum(b * f, axis=1) / det
    dfdy = np.sum(c * f, axis=1) / det
    return np.column_stack([dfdx, dfdy]), 0.5 * np.abs(det)


def _surface_loss(vtu_path: Path, sigma_s: float, z_window: tuple[float, float] | None = None) -> float:
    """Surface-conductance real power INT sigma_s |grad_s phi|^2 (2 pi r) ds over
    the insulator_surface boundary line elements (axisymmetric). Mirrors the
    ComplexEQS surface term, so tan-delta/leakage capture the surface pollution.

    The insulator_surface is auto-detected as the boundary line group farthest
    from the symmetry axis (it is the only boundary that never touches r=0; the
    Elmer VTU re-numbers boundary GeometryIds by internal order, so we cannot
    rely on the mesh.names id).
    """
    m = meshio.read(str(vtu_path))
    pts = np.asarray(m.points, dtype=float)
    pd = {k.lower(): np.asarray(v, dtype=float).reshape(-1) for k, v in m.point_data.items()}
    re, im = pd["potential re"], pd["potential im"]
    geom = m.cell_data.get("GeometryIds")
    lines = lgid = None
    for i, b in enumerate(m.cells):
        if b.type == "line":
            lines = np.asarray(b.data, dtype=int)
            lgid = np.asarray(geom[i], dtype=int).reshape(-1) if geom is not None else None
            break
    if lines is None or lgid is None:
        return 0.0

    best_gid, best_minr = None, -1.0
    for g in np.unique(lgid):
        seg = lines[lgid == g]
        minr = float(pts[seg.reshape(-1), 0].min())
        if minr > best_minr:
            best_minr, best_gid = minr, g
    seg = lines[lgid == best_gid]
    if seg.size == 0:
        return 0.0
    p1, p2 = pts[seg[:, 0]], pts[seg[:, 1]]
    L = np.hypot(p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1])
    dre = re[seg[:, 1]] - re[seg[:, 0]]
    dim = im[seg[:, 1]] - im[seg[:, 0]]
    rmid = 0.5 * (p1[:, 0] + p2[:, 0])
    # per-node sigma_s windowing (matches the solver's nodal interpolation): a
    # node inside [z_lo, z_hi] carries sigma_s, outside carries 0; the segment uses
    # the node average. z_window=None => uniform (full creepage).
    if z_window is None:
        sig_seg = np.full(len(seg), sigma_s)
    else:
        zlo, zhi = z_window
        zn = sigma_s * ((pts[:, 1] >= zlo) & (pts[:, 1] <= zhi)).astype(float)
        sig_seg = 0.5 * (zn[seg[:, 0]] + zn[seg[:, 1]])
    return float(np.sum(sig_seg * (dre**2 + dim**2) / L * 2.0 * np.pi * rmid))


def compute_cvt_observables(
    vtu_path: str | Path,
    *,
    u0: float,
    omega: float,
    body_epsr: dict[int, float],
    body_sigma: dict[int, float],
    tap_body_id: int,
    sigma_s: float = 0.0,
    ins_geom_id: int | None = None,
    surface_z_window: tuple[float, float] | None = None,
) -> dict[str, float]:
    pts, tris, gids, re, im = _read(Path(vtu_path))

    grad_re, area = _grad_area(pts, tris, re)
    grad_im, _ = _grad_area(pts, tris, im)
    e2 = np.sum(grad_re**2, axis=1) + np.sum(grad_im**2, axis=1)  # |E|^2

    missing = set(np.unique(gids)) - set(body_epsr)
    if missing:
        raise KeyError(f"body ids {sorted(missing)} missing from body maps")
    epsr = np.array([body_epsr[g] for g in gids])
    sigma = np.array([body_sigma[g] for g in gids])  # sigma_eff (loss + pollution)
    r_cent = pts[tris].mean(axis=1)[:, 0]  # centroid radius
    dvol = area * 2.0 * np.pi * r_cent     # axisymmetric volume element

    # energy capacitance (real permittivity stores the energy)
    int_eps = float(np.sum(EPS0 * epsr * e2 * dvol))
    int_sig = float(np.sum(sigma * e2 * dvol))
    c_total = int_eps / u0**2

    # add surface-conductance (creepage pollution) real power if requested
    int_surf = _surface_loss(Path(vtu_path), sigma_s, surface_z_window) if sigma_s > 0 else 0.0
    int_real = int_sig + int_surf

    tan_delta = int_real / (omega * int_eps) if int_eps > 0 else 0.0
    p_loss = 0.5 * int_real                      # time-avg dissipated power [W]
    # split the in-phase terminal current by source:
    #   surface_leak = creepage surface-conductance current (external pollution path)
    #   volume_leak  = bulk dielectric/conduction loss current (internal path)
    #   i_leak       = total (kept for back-compat)
    surface_leak = int_surf / u0 if u0 else 0.0
    volume_leak = int_sig / u0 if u0 else 0.0
    i_leak = int_real / u0 if u0 else 0.0

    # tap potential: average complex phi over nodes of the tap-disc body
    tap_tris = tris[gids == tap_body_id]
    if tap_tris.size == 0:
        raise RuntimeError(f"no elements found for tap body id {tap_body_id}")
    tap_nodes = np.unique(tap_tris.reshape(-1))
    vtap_re = float(np.mean(re[tap_nodes]))
    vtap_im = float(np.mean(im[tap_nodes]))
    vtap_abs = float(np.hypot(vtap_re, vtap_im))

    return {
        "C_total_F": c_total,
        "C_total_pF": c_total * 1e12,
        "tan_delta": tan_delta,
        "p_loss": p_loss,
        "i_leak": i_leak,
        "surface_leak": surface_leak,
        "volume_leak": volume_leak,
        "vtap_re": vtap_re,
        "vtap_im": vtap_im,
        "vtap_abs": vtap_abs,
        "vtap_phase_deg": float(np.degrees(np.arctan2(vtap_im, vtap_re))),
        "vtap_phase_mdeg": float(np.degrees(np.arctan2(vtap_im, vtap_re)) * 1e3),
        "divider_ratio": vtap_abs / u0,
        "u0": u0,
        "n_tap_nodes": int(tap_nodes.size),
    }

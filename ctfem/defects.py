"""Defect injection (material-level, cell-by-cell, no remeshing).

Each :class:`~ctfem.config.DefectSpec` is applied by mutating the per-cell
complex coefficient kappa = sigma + j w eps0 eps_r produced for the healthy
baseline.  Working at the cell level (using cell centroids) lets us inject
*spatially local* defects -- a moisture patch over an axial window, a shorted-
foil conductive bridge in one gap -- without touching the mesh.

All four CT spec defects are implemented:
  shorted_foil       -- metal-like sigma bridge across the gap at foil `index`.
  moisture_ingress   -- local wet-paper patch: eps_r up, tan delta up.
  global_aging       -- uniform paper tan delta rise + slight eps_r drift.
  oil_contamination  -- oil conductivity increase.

CVT (capacitor-stack) defects:
  shorted_element    -- galvanic breakdown of capacitor element `index`
                        (1-based from the HV end): the element block becomes
                        conductive, so C jumps by ~N/(N-1) and the divider
                        ratio shifts (the classic CVT element failure).
  element_aging      -- element-dielectric tan delta rise (index 0 = all
                        elements, k >= 1 a single element).
  water_ingress      -- free-water pocket in the oil (eps_r/sigma from the
                        DefectSpec; defaults eps_r=80, sigma=1e-4 S/m).

None of these are boundary conditions: every defect is a cell-level change of
kappa, so defective regions FLOAT at whatever potential the field imposes.

The material parameters used here come from :mod:`ctfem.materials`; severity in
[0,1] interpolates between healthy and the worst-case values quoted in the spec.
"""
from __future__ import annotations

import numpy as np

from .config import DefectSpec, OperatingParams, MaterialParams, GeometryParams
from .materials import Material, MaterialDB, EPS0, SIGMA_NUMERICAL_CAP


# worst-case endpoints for severity interpolation (spec values)
_WET_PAPER_EPSR_MAX = 7.0      # eps_r 3.5 -> 7.0 at severity 1
_WET_PAPER_TAND_MAX = 0.10     # tan delta up to ~0.1
_AGING_TAND_MAX = 0.05         # global aged paper tan delta at severity 1
_AGING_EPSR_DRIFT = 0.3        # eps_r drift (3.5 -> 3.8) at severity 1
_OIL_SIGMA_MAX = 1.0e-9        # contaminated oil conductivity [S/m] at sev 1
_SHORT_SIGMA = 3.5e7           # metal-like bridge conductivity [S/m]
_ELEMENT_AGING_TAND_MAX = 0.02  # degraded paper-film element at severity 1


def _kappa_from(eps_r: float, tan_delta: float, sigma: float, omega: float) -> complex:
    """kappa = sigma + j w eps0 eps_r (1 - j tan_delta).  (see materials.py)"""
    eps_complex = eps_r * (1.0 - 1j * tan_delta)
    return sigma + 1j * omega * EPS0 * eps_complex


def _azimuth_mask(
    spec: DefectSpec, theta: np.ndarray | None, n: int
) -> np.ndarray:
    """Boolean mask selecting cells within the defect's azimuthal window.

    Full ring (default) or no azimuth available (2-D) -> all True.  Otherwise
    keep cells whose angle is within +/- theta_extent/2 of theta_center, with
    correct wrap-around at +/-pi.
    """
    if theta is None or spec.is_full_ring:
        return np.ones(n, dtype=bool)
    # smallest signed angular distance, wrapped to (-pi, pi]
    d = np.angle(np.exp(1j * (theta - spec.theta_center)))
    return np.abs(d) <= 0.5 * spec.theta_extent


def apply_defect(
    spec: DefectSpec,
    centroids: np.ndarray,          # (N, 2) cell (r, z)
    region_per_cell: np.ndarray,    # (N,) region names
    kappa: np.ndarray,              # (N,) complex baseline coefficient
    matdb: MaterialDB,
    operating: OperatingParams,
    materials: MaterialParams,
    geometry: GeometryParams | None = None,
    theta: np.ndarray | None = None,
) -> np.ndarray:
    """Return a copy of `kappa` with the defect applied at the cell level.

    `theta` (cell azimuth, radians) is optional and only used in 3-D for
    azimuthally-localized defects; in 2-D it is None and defects act as full
    rings (rotationally symmetric).
    """
    out = kappa.copy()
    azi = _azimuth_mask(spec, theta, kappa.shape[0])
    # severity-scaled kinds are a no-op at severity 0; the shorted kinds short
    # regardless (the kind itself implies the bridge exists)
    _severity_kinds = ("moisture_ingress", "global_aging", "oil_contamination",
                       "element_aging", "water_ingress")
    if spec.kind == "none" or (spec.severity == 0.0
                               and spec.kind in _severity_kinds):
        return out

    w = operating.omega
    z = centroids[:, 1]
    r = centroids[:, 0]
    is_paper = region_per_cell == "paper_insulation"
    is_oil = region_per_cell == "oil"

    if spec.kind == "shorted_foil":
        # Galvanically bridge foil `index` to the next outer foil (or to the
        # conductor for index == 0) by raising sigma of the paper annulus in the
        # radial gap, over a short axial window centred on the foil mid-height.
        geometry = geometry or GeometryParams(n_foils=materials_n_foils(region_per_cell))
        radii = geometry.foil_radii()
        k = int(spec.index)
        if k < 0 or k >= len(radii):
            raise ValueError(f"shorted_foil index {k} out of range (N={len(radii)})")
        # radial band between foil k-1 (or conductor) and foil k
        if k == 0:
            r_lo = geometry.conductor_radius
        else:
            r_lo = radii[k - 1]
        r_hi = radii[k]
        z0, z1 = geometry.foil_axial_span(k)
        zc = 0.5 * (z0 + z1)
        half = 0.5 * spec.short_axial_window
        sel = (is_paper & (r >= min(r_lo, r_hi)) & (r <= max(r_lo, r_hi))
               & (np.abs(z - zc) <= half) & azi)
        # severity scales how "hard" the short is (sigma magnitude); a full
        # short uses metal-like sigma. severity 0 still shorts (kind implies it).
        # The numerical cap keeps the matrix contrast benign (see materials.py);
        # the bridge stays >>1e8 x more conductive than paper, i.e. a hard short.
        sigma = min(_SHORT_SIGMA, SIGMA_NUMERICAL_CAP) * (
            spec.severity if spec.severity > 0 else 1.0)
        paper = matdb.get("paper")
        out[sel] = _kappa_from(paper.eps_r, paper.tan_delta, sigma, w)
        return out

    if spec.kind == "moisture_ingress":
        sel = is_paper & (np.abs(z - spec.z_center) <= spec.extent) & azi
        s = spec.severity
        paper = matdb.get("paper")
        eps_r = paper.eps_r + s * (_WET_PAPER_EPSR_MAX - paper.eps_r)
        tand = paper.tan_delta + s * (_WET_PAPER_TAND_MAX - paper.tan_delta)
        out[sel] = _kappa_from(eps_r, tand, 0.0, w)
        return out

    if spec.kind == "global_aging":
        s = spec.severity
        paper = matdb.get("paper")
        eps_r = paper.eps_r + s * _AGING_EPSR_DRIFT
        tand = paper.tan_delta + s * (_AGING_TAND_MAX - paper.tan_delta)
        out[is_paper & azi] = _kappa_from(eps_r, tand, 0.0, w)
        return out

    if spec.kind == "oil_contamination":
        s = spec.severity
        oil = matdb.get("oil")
        sigma = s * _OIL_SIGMA_MAX
        out[is_oil & azi] = _kappa_from(oil.eps_r, oil.tan_delta, sigma, w)
        return out

    if spec.kind == "shorted_element":
        # CVT: galvanically bridge capacitor element `index` (1-based, 1 = HV
        # end) -- the homogenized element block turns conductive.  Severity
        # scales the bridge sigma (a real element fails section-by-section);
        # severity 0 still shorts (the kind implies it).  Same numerical cap
        # rationale as shorted_foil (see materials.py).
        k = int(spec.index)
        name = f"element_{k}"
        sel = (region_per_cell == name) & azi
        if not np.any(region_per_cell == name):
            raise ValueError(
                f"shorted_element index {k}: region {name!r} not in mesh")
        el = matdb.get("element_dielectric")
        sigma = min(_SHORT_SIGMA, SIGMA_NUMERICAL_CAP) * (
            spec.severity if spec.severity > 0 else 1.0)
        out[sel] = _kappa_from(el.eps_r, el.tan_delta, sigma, w)
        return out

    if spec.kind == "water_ingress":
        # Free-water pocket in the oil: purely a volumetric kappa override
        # (eps_r/sigma from the spec; defaults are liquid water).  NO Dirichlet
        # is attached anywhere -- the pocket floats at the potential the
        # surrounding divider field imposes and the equipotentials wrap around
        # it.  At 50 Hz water's sigma/(w eps0 eps_r) >> 1, so the pocket acts
        # near-equipotential at its LOCAL level, not at U0.
        s = spec.severity
        oil = matdb.get("oil")
        eps_r = oil.eps_r + s * (spec.defect_eps_r - oil.eps_r)
        sigma = s * spec.defect_sigma
        sel = is_oil & (np.abs(z - spec.z_center) <= spec.extent) & azi
        out[sel] = _kappa_from(eps_r, oil.tan_delta, sigma, w)
        return out

    if spec.kind == "element_aging":
        # CVT: element-dielectric loss rise (degradation/partial-discharge
        # damage in the wound paper-film).  index 0 = all elements.
        s = spec.severity
        el = matdb.get("element_dielectric")
        tand = el.tan_delta + s * (_ELEMENT_AGING_TAND_MAX - el.tan_delta)
        is_el = np.char.startswith(region_per_cell.astype(str), "element_")
        if spec.index > 0:
            is_el &= region_per_cell == f"element_{int(spec.index)}"
        out[is_el & azi] = _kappa_from(el.eps_r, tand, 0.0, w)
        return out

    raise ValueError(f"unknown defect kind {spec.kind!r}")


def materials_n_foils(region_per_cell: np.ndarray) -> int:
    """Infer N foils from the region names present (fallback helper)."""
    foils = {name for name in np.unique(region_per_cell) if str(name).startswith("foil_")}
    return len(foils)

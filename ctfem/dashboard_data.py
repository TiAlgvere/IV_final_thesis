"""Compute layer for the Streamlit fault-explorer dashboard.

No ``streamlit`` import here, so every function is plain and unit-testable; the
UI (``dashboard.py``) adds the caching/widgets on top.  This wraps the Phase-7
pipeline: FEA observables (C, tap fraction, tan delta) -> lumped-CVT secondary
signatures (ratio error %, phase displacement arc-min) via :mod:`ctfem.cvt_circuit`.

Everything runs in 3-D (scikit-fem) on ONE cached mesh, so the baseline library
and the user's runs are dimensionally consistent and a phase-angle field is
always available to render.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .config import CVTParams, OperatingParams, MaterialParams, DefectSpec
from .common import run_case_3d
from .cvt_circuit import CVTCircuit, solve_from_fea, c1_c2_from_fea

# Capacitance temperature coefficient of oil-impregnated paper (representative
# literature value, ~ -0.04 %/degC): the oil permittivity falls with heat.  Used
# only by the instant circuit-level operating point, not the FEA solves.
TC_CAP_PER_C = -4.0e-4
# tan delta temperature coefficient (oil-impregnated paper loss climbs with heat;
# same linear form as ctfem.materials paper, ~ +2 %/degC of the base value):
# tan_delta(T) = base * (1 + TAND_TCOEF_PER_C * (T - 20)), clamped >= 0.  This is
# the summer-heat noise source for tan-delta-based pollution/moisture detection.
TAND_TCOEF_PER_C = 0.02


@dataclass
class FaultPoint:
    """FEA terminal observables for one case (what the lumped circuit needs)."""
    name: str
    category: str            # 'healthy' | 'internal' | 'surface' | 'user'
    C_pF: float
    tap_frac: float
    tan_delta: float
    leakage_mA: float = 0.0  # surface leakage current to ground


@dataclass
class CircuitPoint:
    """Lumped-CVT secondary signature for one FaultPoint at a given burden."""
    name: str
    category: str
    ratio_err_pct: float     # |U2|/|U2_healthy| - 1
    phase_disp_min: float    # phase displacement vs healthy [arc-minutes]
    phase_abs_min: float     # absolute secondary phase [arc-minutes]
    C_pF: float
    tan_delta: float
    leakage_mA: float = 0.0  # carried through for the scatter colour/size


def _point(name: str, category: str, obs, cvt: CVTParams) -> FaultPoint:
    tap = cvt.tap_disc_index
    return FaultPoint(name, category, obs.C1_pF,
                      obs.foil_potential_frac[tap - 1], obs.tan_delta,
                      obs.surface_leakage_mA)


# --------------------------------------------------------------------------- #
# FEA (one cached 3-D mesh, reused for every solve)
# --------------------------------------------------------------------------- #

def build_mesh_3d(cvt: CVTParams, msh_path: str) -> str:
    """Build the 3-D CVT mesh once (faults are material overrides on it).

    gmsh.initialize() installs a SIGINT handler via ``signal.signal``, which
    raises "signal only works in main thread" when the build runs in a worker
    thread (e.g. Streamlit's ScriptRunner).  Off the main thread we neutralise
    ``signal.signal`` for the duration of the build -- the dashboard does not
    need gmsh's Ctrl-C handling.  Only the mesh BUILD uses gmsh; the skfem solve
    loads the .msh via meshio and is unaffected.
    """
    import signal
    import threading
    from .geometry3d import build_cvt_3d
    if threading.current_thread() is threading.main_thread():
        build_cvt_3d(cvt, msh_path, verbose=False)
        return msh_path
    _orig = signal.signal
    signal.signal = lambda *a, **k: None      # no-op while gmsh (re)sets SIGINT
    try:
        build_cvt_3d(cvt, msh_path, verbose=False)
    finally:
        signal.signal = _orig
    return msh_path


def internal_defect_specs(cvt: CVTParams) -> list[tuple[str, str, object]]:
    """The baseline library: (name, category, DefectSpec or None)."""
    n1 = cvt.n_elements - cvt.n_elements_c2
    return [
        ("healthy", "healthy", None),
        ("C1 short", "internal",
         DefectSpec(kind="shorted_element", severity=1.0, index=max(1, n1 // 2))),
        ("C2 short", "internal",
         DefectSpec(kind="shorted_element", severity=1.0, index=n1 + 1)),
        ("element aging", "internal",
         DefectSpec(kind="element_aging", severity=1.0, index=0)),
        ("oil contamination", "internal",
         DefectSpec(kind="oil_contamination", severity=1.0)),
        ("water ingress", "internal",
         DefectSpec(kind="water_ingress", severity=1.0, z_center=0.75, extent=0.15)),
    ]


def baseline_library(msh_path: str, op: OperatingParams, matdb,
                     cvt: CVTParams) -> list[FaultPoint]:
    """Solve healthy + the internal-defect set -> FaultPoints (3-D FEA)."""
    pts = []
    for name, cat, spec in internal_defect_specs(cvt):
        obs = run_case_3d(msh_path, op, matdb=matdb, defect=spec, backend="skfem")
        pts.append(_point(name, cat, obs, cvt))
    return pts


def _streamer_materials(cvt: CVTParams, sigma_s: float, sheds: int,
                        sigma: float, film_mm: float, width_deg: float
                        ) -> MaterialParams:
    if sheds <= 0:
        return MaterialParams(surface_conductivity_s=sigma_s)
    centers = cvt.shed_centers()
    pitch = (cvt.stack_z_hi - cvt.tank_height) / cvt.n_sheds
    n = min(sheds, len(centers))
    z_lo, z_hi = centers[-n] - 0.6 * pitch, centers[-1] + 0.6 * pitch
    return MaterialParams(
        surface_conductivity_s=sigma_s, streamer_sigma_s=sigma * film_mm * 1e-3,
        streamer_theta_extent=math.radians(width_deg), streamer_z_range=(z_lo, z_hi))


def user_run(msh_path: str, op: OperatingParams, matdb, cvt: CVTParams,
             sigma_s: float, sheds: int, *, sigma: float = 0.5,
             film_mm: float = 0.2, width_deg: float = 15.0,
             png_path: str | None = None
             ) -> tuple[FaultPoint, str | None, str | None]:
    """Solve one user case (uniform pollution sigma_s + optional stork streamer).

    If `png_path` is given, also export the phase-angle field to a `.vtu` (for
    the in-app interactive 3-D view) and a `.png` snapshot fallback.  Returns
    (FaultPoint, png_path_or_None, vtu_path_or_None).
    """
    mats = _streamer_materials(cvt, sigma_s, sheds, sigma, film_mm, width_deg)
    obs, sol, _ = run_case_3d(msh_path, op, materials=mats, matdb=matdb,
                              backend="skfem", return_solution=True)
    label = f"sigma_s={sigma_s:.0e}" + (f" + streamer {sheds} sheds"
                                        if sheds > 0 else "")
    pt = _point(f"RUN: {label}", "user", obs, cvt)
    vtu = None
    if png_path is not None:
        try:
            from .viz3d import defect_indicator, export_vtu, screenshot
            ind = defect_indicator(sol, DefectSpec(kind="none"), matdb=matdb,
                                   operating=op)
            vtu = os.path.splitext(png_path)[0] + ".vtu"
            export_vtu(sol, ind, vtu)
            screenshot(vtu, png_path, "phi_phase_mrad")
        except Exception:           # rendering must never crash the dashboard
            png_path = vtu = None
    return pt, png_path, vtu


# --------------------------------------------------------------------------- #
# lumped circuit (instant; no FEA) -- this is what the burden slider drives
# --------------------------------------------------------------------------- #

def circuit_points(points: list[FaultPoint], healthy: FaultPoint,
                   burden_va: float, u0: float, *, frequency: float = 50.0
                   ) -> list[CircuitPoint]:
    """Map FaultPoints to secondary signatures with the reactor tuned on
    `healthy` and the given burden.  Pure phasor math -- runs instantly."""
    ck = CVTCircuit(burden_va=burden_va, frequency=frequency)
    h = solve_from_fea(ck, healthy.C_pF * 1e-12, healthy.tap_frac,
                       healthy.tan_delta, u0, tune=True)
    out = []
    for p in points:
        r = solve_from_fea(ck, p.C_pF * 1e-12, p.tap_frac, p.tan_delta, u0)
        out.append(CircuitPoint(
            p.name, p.category,
            (abs(r.u_int) / abs(h.u_int) - 1.0) * 100.0,
            r.phase_min - h.phase_min, r.phase_min, p.C_pF, p.tan_delta,
            p.leakage_mA))
    return out


# --------------------------------------------------------------------------- #
# instant operating point: internal faults + grid conditions (no FEA)
# --------------------------------------------------------------------------- #

def operating_point(healthy: FaultPoint, *, c1_short: float = 0.0,
                    c2_short: float = 0.0, moisture: float = 0.0,
                    temperature_c: float = 20.0,
                    name: str = "operating point") -> FaultPoint:
    """Healthy baseline perturbed by INSTANT (circuit-level) internal faults +
    temperature -- a synthetic FaultPoint, no FEA.

    Shorting a fraction f of a section's series elements raises that section's
    capacitance by 1/(1-f); moisture scales the base capacitance of both
    sections by (1+m); temperature scales it by (1 + TC*(T-20)).  Internal faults
    carry no surface leakage, so leakage_mA = 0.
    """
    c1, c2 = c1_c2_from_fea(healthy.C_pF * 1e-12, healthy.tap_frac)
    c1 /= max(1e-6, 1.0 - c1_short)
    c2 /= max(1e-6, 1.0 - c2_short)
    scale = (1.0 + moisture) * (1.0 + TC_CAP_PER_C * (temperature_c - 20.0))
    c1 *= scale
    c2 *= scale
    s = c1 + c2
    # tan delta climbs with temperature (oil-paper loss) -> a noise source for
    # the tan-delta channel itself, and it feeds the ratio/phase via G = wC tand.
    tand = max(0.0, healthy.tan_delta
               * (1.0 + TAND_TCOEF_PER_C * (temperature_c - 20.0)))
    return FaultPoint(name, "user", (c1 * c2 / s) * 1e12, c1 / s, tand, 0.0)


def secondary_signature(point: FaultPoint, healthy: FaultPoint, burden_va: float,
                        u0: float, *, grid_freq: float = 50.0,
                        ref_burden: float | None = None) -> CircuitPoint:
    """Secondary signature of `point`.  The reactor is tuned ONCE on healthy at
    the nominal 50 Hz / `ref_burden`, then the circuit is SOLVED at `grid_freq`
    and `burden_va` with that fixed reactor -- so off-nominal frequency AND a
    fluctuating load both detune it (noise sources).  Deltas are vs healthy at
    the nominal reference.  `ref_burden` defaults to `burden_va`.
    """
    ref_burden = burden_va if ref_burden is None else ref_burden
    ck = CVTCircuit(burden_va=ref_burden, frequency=50.0)
    h = solve_from_fea(ck, healthy.C_pF * 1e-12, healthy.tap_frac,
                       healthy.tan_delta, u0, tune=True)   # tune reactor (fixed)
    ck.frequency = grid_freq                                # grid runs off-nominal
    ck.burden_va = burden_va                                # actual fluctuating load
    c1, c2 = c1_c2_from_fea(point.C_pF * 1e-12, point.tap_frac)
    r = ck.solve(c1, c2, point.tan_delta, u0)
    return CircuitPoint(point.name, point.category,
                        (abs(r.u_int) / abs(h.u_int) - 1.0) * 100.0,
                        r.phase_min - h.phase_min, r.phase_min,
                        point.C_pF, point.tan_delta, point.leakage_mA)


def noise_envelope(healthy: FaultPoint, burden_va: float, u0: float, *,
                   temps=(-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0),
                   freqs=(49.8, 49.9, 50.0, 50.1, 50.2), burden_tol: float = 0.05,
                   instr_ratio_pct: float = 0.05, instr_phase_min: float = 0.5):
    """Convex hull (ordered (ratio, phase) vertices) of the total noise floor:
    the FAULT-FREE operating point swept over temperature x frequency x burden
    (+/-`burden_tol`), then Minkowski-summed with the instrument-resolution box
    (`instr_ratio_pct` %, `instr_phase_min` arc-min -- representative
    PMU/metering trending resolution).  The instrument box gives the (otherwise
    paper-thin) weather hull a realistic minimum thickness in every direction, so
    a fault must exceed BOTH drift AND what a meter can resolve.  Falls back to a
    box polygon if degenerate.
    """
    import numpy as np
    burdens = (burden_va * (1.0 - burden_tol), burden_va,
               burden_va * (1.0 + burden_tol))
    pts = []
    for t in temps:
        op = operating_point(healthy, temperature_c=t)
        for f in freqs:
            for b in burdens:
                cp = secondary_signature(op, healthy, b, u0, grid_freq=f,
                                         ref_burden=burden_va)
                pts.append((cp.ratio_err_pct, cp.phase_disp_min))
    pts = np.asarray(pts, dtype=float)
    # Minkowski-sum with the instrument-resolution box (min detectable change)
    box = np.array([[instr_ratio_pct, instr_phase_min],
                    [instr_ratio_pct, -instr_phase_min],
                    [-instr_ratio_pct, instr_phase_min],
                    [-instr_ratio_pct, -instr_phase_min]])
    pts = (pts[:, None, :] + box[None, :, :]).reshape(-1, 2)
    try:
        from scipy.spatial import ConvexHull
        return pts[ConvexHull(pts).vertices]
    except Exception:               # collinear/degenerate -> bounding-box polygon
        rmin, rmax = float(pts[:, 0].min()), float(pts[:, 0].max())
        pmin, pmax = float(pts[:, 1].min()), float(pts[:, 1].max())
        return np.array([[rmin, pmin], [rmax, pmin], [rmax, pmax], [rmin, pmax]])


def inside_envelope(vertices, x: float, y: float) -> bool:
    """Point-in-polygon (ray casting) for the convex-hull weather envelope."""
    v = vertices
    n = len(v)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = v[i][0], v[i][1]
        xj, yj = v[j][0], v[j][1]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi):
            inside = not inside
        j = i
    return inside

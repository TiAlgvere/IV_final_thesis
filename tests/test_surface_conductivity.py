"""Exterior-insulator surface-conductivity (leakage / pollution) tests.

Phase prep for environmental tracking + pollution-flashover studies: geometry3d
groups the air-facing porcelain/shed faces into an "insulator_surface" physical
surface, and the skfem solver attaches a surface-conduction (Laplace-Beltrami)
term with sheet conductance sigma_s on that group.  A clean surface
(sigma_s ~ 1e-14 S) is the pure-electrostatic baseline; a conductive pollution
layer opens a resistive HV->ground leakage path along the creepage surface,
raising the loss tangent while the (capacitive) C is essentially unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from ctfem.config import CVTParams, OperatingParams, MaterialParams, DefectSpec
from ctfem.common import run_case_3d
from ctfem.geometry3d import build_cvt_3d


@pytest.fixture(scope="module")
def cvt3d_mesh(tmp_path_factory):
    c = CVTParams(mesh_refinement=0.15)
    msh = str(tmp_path_factory.mktemp("cvt3d") / "cvt3d.msh")
    mres = build_cvt_3d(c, msh, verbose=False)
    return c, mres, msh


def test_insulator_surface_group_present(cvt3d_mesh):
    _, mres, _ = cvt3d_mesh
    # geometry3d must tag the air-facing porcelain + shed faces as one group.
    # NB: build_ct_3d reuses the generic MeshResult fields, so in 3-D the
    # *surface* groups are stored under `curve_groups` (volumes -> surface_groups).
    assert "insulator_surface" in mres.curve_groups
    assert mres.curve_groups["insulator_surface"] > 0


def test_clean_surface_is_negligible(cvt3d_mesh):
    # the default near-zero sigma_s must leave the baseline solve unchanged
    c, _, msh = cvt3d_mesh
    op = OperatingParams(um_kv=123.0)
    matdb = c.material_db()
    obs_off = run_case_3d(msh, op, matdb=matdb, backend="skfem",
                          materials=MaterialParams(surface_conductivity_s=0.0))
    obs_clean = run_case_3d(msh, op, matdb=matdb, backend="skfem",
                            materials=MaterialParams(surface_conductivity_s=1e-14))
    assert abs(obs_clean.tan_delta - obs_off.tan_delta) < 1e-6
    assert abs(obs_clean.C1_pF - obs_off.C1_pF) / obs_off.C1_pF < 1e-6


def test_pollution_layer_raises_leakage_loss(cvt3d_mesh):
    # a conductive surface layer adds a resistive leakage path: tan delta rises
    # sharply, C (capacitive) barely moves -- the discriminating signature.
    c, _, msh = cvt3d_mesh
    op = OperatingParams(um_kv=123.0)
    matdb = c.material_db()
    clean = run_case_3d(msh, op, matdb=matdb, backend="skfem",
                        materials=MaterialParams(surface_conductivity_s=1e-14))
    polluted = run_case_3d(msh, op, matdb=matdb, backend="skfem",
                           materials=MaterialParams(surface_conductivity_s=1e-6))
    assert polluted.tan_delta > 10.0 * clean.tan_delta      # strong leakage loss
    assert abs(polluted.C1_pF - clean.C1_pF) / clean.C1_pF < 0.02
    # monotonic: a lighter layer sits between clean and heavy pollution
    light = run_case_3d(msh, op, matdb=matdb, backend="skfem",
                        materials=MaterialParams(surface_conductivity_s=1e-9))
    assert clean.tan_delta < light.tan_delta < polluted.tan_delta
    # the physical surface leakage current to ground: ~0 when clean, growing
    # with pollution (this is the alternative diagnostic to phase displacement)
    assert clean.surface_leakage_mA < 0.01
    assert (polluted.surface_leakage_mA > light.surface_leakage_mA
            > clean.surface_leakage_mA)


def _streak_z_range(c, n_sheds):
    centers = c.shed_centers()
    pitch = (c.stack_z_hi - c.tank_height) / c.n_sheds
    n = min(n_sheds, len(centers))
    return centers[-n] - 0.6 * pitch, centers[-1] + 0.6 * pitch


def test_streamer_shifts_phase_without_raising_tand(cvt3d_mesh):
    # Type B 'bird streamer' fingerprint: a localized 15-deg conductive streak
    # rotates the tap phase (scaling with the creepage it bridges) but, wetting
    # only ~4% of the ring, leaves terminal tan_delta near clean -- UNLIKE a
    # uniform Type A layer of the same sheet conductance, which spikes tan_delta.
    c, _, msh = cvt3d_mesh
    op = OperatingParams(um_kv=123.0)
    matdb = c.material_db()
    tap = c.tap_disc_index

    def _solve(mats):
        return run_case_3d(msh, op, materials=mats, matdb=matdb, backend="skfem")

    def _phase_mdeg(obs):
        v = obs.foil_potentials[tap - 1]
        return np.degrees(np.arctan2(v.imag, v.real)) * 1e3

    clean = _solve(MaterialParams(surface_conductivity_s=1e-14))
    z_lo, z_hi = _streak_z_range(c, 20)          # long streak (bridges ~3/4 path)
    streak = _solve(MaterialParams(
        surface_conductivity_s=1e-14, streamer_sigma_s=1e-3,
        streamer_theta_extent=np.radians(15.0), streamer_z_range=(z_lo, z_hi)))
    uniform = _solve(MaterialParams(surface_conductivity_s=1e-3))

    assert abs(_phase_mdeg(streak) - _phase_mdeg(clean)) > 0.05   # phase moves
    assert streak.tan_delta < 1.5 * clean.tan_delta              # but tand ~ clean
    assert uniform.tan_delta > 50.0 * clean.tan_delta            # uniform spikes it


def test_streamer_localizes_leakage(cvt3d_mesh):
    # the same sheet conductance, full-ring vs a 15-deg streak: the streak's
    # terminal loss is far smaller (masking restricts it to ~4% of the surface).
    c, _, msh = cvt3d_mesh
    op = OperatingParams(um_kv=123.0)
    matdb = c.material_db()
    z_lo, z_hi = _streak_z_range(c, 4)
    uniform = run_case_3d(msh, op, matdb=matdb, backend="skfem",
                          materials=MaterialParams(surface_conductivity_s=1e-5))
    streak = run_case_3d(msh, op, matdb=matdb, backend="skfem",
                         materials=MaterialParams(
                             surface_conductivity_s=1e-14, streamer_sigma_s=1e-5,
                             streamer_theta_extent=np.radians(15.0),
                             streamer_z_range=(z_lo, z_hi)))
    assert streak.tan_delta < 0.1 * uniform.tan_delta


def test_phase_field_exported_to_vtu(cvt3d_mesh, tmp_path):
    # the 3-D VTU must carry the nodal phase-angle field (milliradians) so the
    # phase distribution can be visualized / toggled in PyVista or ParaView.
    import meshio
    from ctfem.viz3d import defect_indicator, export_vtu
    c, _, msh = cvt3d_mesh
    op = OperatingParams(um_kv=123.0)
    matdb = c.material_db()
    _, sol, _ = run_case_3d(msh, op, matdb=matdb, backend="skfem",
                            materials=MaterialParams(surface_conductivity_s=1e-6),
                            return_solution=True)
    ind = defect_indicator(sol, DefectSpec(kind="none"), matdb=matdb, operating=op)
    vtu = str(tmp_path / "out.vtu")
    export_vtu(sol, ind, vtu)
    m = meshio.read(vtu)
    assert "phi_phase_mrad" in m.point_data
    assert "phi_kV" in m.point_data
    assert "E_kV_mm" in m.cell_data        # field-magnitude (breakdown) channel


def test_peak_efield_air_is_physical(cvt3d_mesh):
    # |E| = complex norm of -grad(phi); the air-gap peak must be finite, positive,
    # and in a sane range for a 71 kV device (well within the ~3 kV/mm air limit
    # at this mild/coarse case), located inside the near-device window.
    from ctfem.viz3d import cell_efield_kvmm, peak_efield_air
    c, _, msh = cvt3d_mesh
    op = OperatingParams(um_kv=123.0)
    matdb = c.material_db()
    _, sol, _ = run_case_3d(msh, op, matdb=matdb, backend="skfem",
                            return_solution=True)
    e = cell_efield_kvmm(sol)
    assert np.all(np.isfinite(e)) and float(e.max()) > 0.0
    peak, loc = peak_efield_air(sol, r_max=0.40)
    assert loc is not None and 0.0 < peak < 10.0
    assert loc[0] <= 0.40                   # within the queried radius

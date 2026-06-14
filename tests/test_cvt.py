"""CVT (Arteche DDB-123) model tests: geometry, nameplate validation, defects.

The DDB-123 datasheet provides a hard validation anchor the CT never had: the
rated standard capacitance (5600 pF).  The healthy FEM admittance must
reproduce it (the homogenized elements are constructed to, so the residual
deviation measures only fringing/stray paths).  The divider ratio and the
shorted-element C-jump follow from elementary series-capacitor algebra:

    C_eq = C_e / N,    V_tap/U0 = N2/N,    short one element -> C_eq * N/(N-1).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from ctfem.config import CVTParams, OperatingParams, DefectSpec, MaterialParams
from ctfem.common import run_case_2d
from ctfem.geometry import build_cvt


@pytest.fixture(scope="module")
def cvt_case(tmp_path_factory):
    cvt = CVTParams(mesh_refinement=0.7)
    msh = str(tmp_path_factory.mktemp("cvt") / "cvt2d.msh")
    mres = build_cvt(cvt, msh, verbose=False)
    op = OperatingParams(um_kv=123.0)
    matdb = cvt.material_db()
    obs = run_case_2d(msh, op, matdb=matdb, backend="skfem")
    return cvt, mres, op, matdb, obs, msh


def test_cvt_mesh_groups(cvt_case):
    cvt, mres, *_ = cvt_case
    names = set(mres.surface_groups)
    # all stack pieces present: terminals, N-1 interior discs, N elements
    assert {"stack_top", "stack_bottom", "head_housing", "base_tank",
            "porcelain", "porcelain_shed", "oil", "air"} <= names
    assert {f"foil_{k}" for k in range(1, cvt.n_elements)} <= names
    assert {f"element_{k}" for k in range(1, cvt.n_elements + 1)} <= names
    assert {"hv_electrode", "ground_electrode", "farfield"} <= set(
        mres.curve_groups)


def test_cvt_creepage_matches_datasheet():
    # the shed profile must reproduce the DDB-123 standard creepage distance
    cvt = CVTParams()
    creep_mm = cvt.creepage_distance() * 1e3
    assert abs(creep_mm - cvt.rated_creepage_mm) / cvt.rated_creepage_mm < 0.05


def test_cvt_healthy_matches_nameplate(cvt_case):
    cvt, _, op, _, obs, _ = cvt_case
    # rated capacitance (fringing/strays allowed a few percent)
    rated = cvt.rated_capacitance_pF
    assert abs(obs.C1_pF - rated) / rated < 0.12
    # loss dominated by the element dielectric tan delta
    assert 0.001 < obs.tan_delta < 0.01
    # divider ladder: monotonically decreasing from HV to ground
    fr = np.array(obs.foil_potential_frac)
    assert len(fr) == cvt.n_elements - 1
    assert np.all(np.diff(fr) < 0.0)
    # tap voltage fraction ~ N2/N (equal elements)
    tap = fr[cvt.tap_disc_index - 1]
    assert abs(tap - cvt.divider_ratio_nominal()) / cvt.divider_ratio_nominal() < 0.08
    # both admittance estimators agree
    assert obs.admittance_method_discrepancy < 1e-8


def test_cvt_shorted_element_signature(cvt_case):
    cvt, _, op, matdb, obs_h, msh = cvt_case
    n = cvt.n_elements
    tap_h = obs_h.foil_potential_frac[cvt.tap_disc_index - 1]

    # short mid-C1 -> C jumps by ~N/(N-1), tap ratio goes UP
    obs_c1 = run_case_2d(msh, op, matdb=matdb, backend="skfem",
                         defect=DefectSpec(kind="shorted_element",
                                           severity=1.0, index=5))
    jump = obs_c1.C1_pF / obs_h.C1_pF
    assert abs(jump - n / (n - 1)) < 0.03
    tap_c1 = obs_c1.foil_potential_frac[cvt.tap_disc_index - 1]
    assert tap_c1 > tap_h * 1.05

    # short in C2 -> same C jump, but tap ratio goes DOWN
    obs_c2 = run_case_2d(msh, op, matdb=matdb, backend="skfem",
                         defect=DefectSpec(kind="shorted_element",
                                           severity=1.0, index=n))
    assert abs(obs_c2.C1_pF / obs_h.C1_pF - n / (n - 1)) < 0.03
    tap_c2 = obs_c2.foil_potential_frac[cvt.tap_disc_index - 1]
    assert tap_c2 < tap_h * 0.8


def test_cvt_element_aging_raises_tan_delta(cvt_case):
    cvt, _, op, matdb, obs_h, msh = cvt_case
    obs_a = run_case_2d(msh, op, matdb=matdb, backend="skfem",
                        defect=DefectSpec(kind="element_aging", severity=1.0,
                                          index=0))
    # all elements aged to tand 0.02 -> device tan delta rises ~10x, C barely
    assert obs_a.tan_delta > 5.0 * obs_h.tan_delta
    assert abs(obs_a.C1_pF - obs_h.C1_pF) / obs_h.C1_pF < 0.02


def test_cvt_2d_insulator_surface_group(cvt_case):
    _, mres, *_ = cvt_case
    # the 2-D axisymmetric builder must tag the air-facing porcelain + shed
    # boundary as the insulator_surface curve group (mirrors geometry3d), so the
    # same surface-conductivity term is available for rapid 2-D sweeps.
    assert "insulator_surface" in mres.curve_groups
    assert mres.curve_groups["insulator_surface"] > 0


def test_cvt_2d_pollution_raises_loss(cvt_case):
    cvt, _, op, matdb, obs_h, msh = cvt_case
    # obs_h was solved at the clean default (sigma_s ~ 1e-14); a conductive
    # pollution layer opens a resistive creepage-leakage path -> tan delta rises
    # sharply while C (capacitive) is essentially unchanged.
    polluted = run_case_2d(msh, op, matdb=matdb, backend="skfem",
                           materials=MaterialParams(surface_conductivity_s=1e-6))
    assert polluted.tan_delta > 5.0 * obs_h.tan_delta
    assert abs(polluted.C1_pF - obs_h.C1_pF) / obs_h.C1_pF < 0.02


def test_cvt_tap_phase_displacement_peaks(cvt_case):
    # Fault-prediction signal: the complex tap potential acquires a phase lag vs
    # the (real) HV drive as surface leakage develops.  It is a RELAXATION peak
    # -- maximal near the loss corner (~1e-7 S) and smaller on either side --
    # NOT monotonic, so phase alone underreports severity past the peak.
    cvt, _, op, matdb, _, msh = cvt_case
    tap = cvt.tap_disc_index

    def theta(ss):
        obs = run_case_2d(msh, op, matdb=matdb, backend="skfem",
                          materials=MaterialParams(surface_conductivity_s=ss))
        v = obs.foil_potentials[tap - 1]
        return np.degrees(np.arctan2(v.imag, v.real))

    th0 = theta(1e-14)
    assert abs(th0) < 1e-3                      # clean tap ~ in phase (<1 mdeg)
    low = abs(theta(1e-9) - th0)
    peak = abs(theta(1e-7) - th0)
    high = abs(theta(1e-5) - th0)
    assert peak > low                           # rises toward the loss corner
    assert peak > high                          # then relaxes back -> non-monotonic


def test_cvt_water_ingress_raises_loss(cvt_case):
    # A conductive water pocket (eps_r 80, sigma 1e-4 S/m) in the oil annulus is
    # a floating volumetric defect -- NOT a Dirichlet source.  It bridges
    # adjacent disc potentials locally, so the device tan delta rises while the
    # series C (set by the elements) barely moves.  The solver must stay
    # self-consistent (both admittance estimators still agree).
    cvt, _, op, matdb, obs_h, msh = cvt_case
    obs_w = run_case_2d(msh, op, matdb=matdb, backend="skfem",
                        defect=DefectSpec(kind="water_ingress", severity=1.0,
                                          z_center=0.75, extent=0.15))
    # headline signature: the near-equipotential conductive pocket adds a large
    # conduction loss (tan delta jumps ~60x) -- the discriminating feature.
    assert obs_w.tan_delta > 10.0 * obs_h.tan_delta
    # and it bridges adjacent disc levels, so C rises modestly (a few %), but
    # less than a full shorted-element jump N/(N-1) and in the OPPOSITE-of-zero
    # direction from oil_contamination (which barely moves C at all).
    dC = (obs_w.C1_pF - obs_h.C1_pF) / obs_h.C1_pF
    assert 0.0 < dC < 0.09
    assert obs_w.admittance_method_discrepancy < 1e-6   # no spurious BC

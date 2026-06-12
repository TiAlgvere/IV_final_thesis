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

from ctfem.config import CVTParams, OperatingParams, DefectSpec
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

"""Phase-2 acceptance tests on the Windows-native scikit-fem backend.

These mirror tests/test_solver_coax.py (the DOLFINx variant) but run anywhere
scipy + scikit-fem are installed -- i.e. on the Windows dev box too, so the
solver is exercised by every local pytest run, not just on HPC.
"""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("skfem")
pytest.importorskip("gmsh")

from ctfem.validate import validate_coax
from ctfem.config import GeometryParams, OperatingParams, DefectSpec, apply_mesh_preset


def test_coax_capacitance_within_half_percent():
    v = validate_coax(eps_r=3.5, tan_delta=0.0, refinement=1.0, backend="skfem")
    assert v.C_rel_error < 0.005, v.summary()


def test_coax_tan_delta_within_half_percent():
    v = validate_coax(eps_r=3.5, tan_delta=0.01, refinement=1.0, backend="skfem")
    assert v.C_rel_error < 0.005, v.summary()
    assert v.tan_delta_rel_error < 0.005, v.summary()


def test_admittance_methods_agree():
    from ctfem.geometry import build_coax
    from ctfem.skfem_solver import solve_msh_skfem, compute_observables_skfem
    from ctfem.config import MaterialParams
    from ctfem.materials import MaterialDB, Material

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "coax.msh")
        build_coax(0.01, 0.05, 0.2, p, lc=0.002, verbose=False)
        matdb = MaterialDB.default()
        matdb.set("cd", Material("cd", eps_r=3.5, tan_delta=0.01))
        sol = solve_msh_skfem(
            p, OperatingParams(),
            materials=MaterialParams(region_material={"dielectric": "cd"}),
            matdb=matdb)
        obs = compute_observables_skfem(sol)
        # phi^T A phi = U0 * I exactly -> machine-precision agreement
        assert obs.admittance_method_discrepancy < 1e-10


def test_ct_baseline_smoke():
    """Coarse full-CT solve: C1 plausible, ladder monotonic, tand near paper's."""
    from ctfem.geometry import build_ct
    from ctfem.common import run_case_2d

    g = GeometryParams(n_foils=6, mesh_refinement=0.35)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ct.msh")
        build_ct(g, p, verbose=False)
        obs = run_case_2d(p, OperatingParams(), geometry=g, backend="skfem")
    # plausible C1 for a 245 kV condenser CT: hundreds of pF
    assert 50.0 < obs.C1_pF < 5000.0, f"C1={obs.C1_pF} pF out of range"
    # foil ladder monotonic decreasing from HV to ground
    fr = obs.foil_potential_frac
    assert all(fr[i] >= fr[i + 1] - 1e-3 for i in range(len(fr) - 1)), fr
    assert fr[0] < 1.0 and fr[-1] > 0.0
    # loss must be positive and of the order of the paper tan delta
    assert 0.0 < obs.tan_delta < 0.05

"""Solver validation tests -- require a COMPLEX dolfinx build.

These are the Phase-2 acceptance checks.  They are skipped automatically where
DOLFINx/PETSc-complex is unavailable (e.g. a geometry-only dev box) so the rest
of the suite still runs fast.  They use only the small coax mesh (< 2 min).
"""
import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")

# require a complex build, otherwise skip (real build cannot run EQS)
if not np.issubdtype(dolfinx.default_scalar_type, np.complexfloating):
    pytest.skip("dolfinx is a real build; EQS solver needs complex PETSc",
                allow_module_level=True)

from ctfem.validate import validate_coax


def test_coax_capacitance_within_half_percent():
    v = validate_coax(eps_r=3.5, tan_delta=0.0, refinement=1.0)
    assert v.C_rel_error < 0.005, v.summary()


def test_coax_tan_delta_within_half_percent():
    v = validate_coax(eps_r=3.5, tan_delta=0.01, refinement=1.0)
    assert v.C_rel_error < 0.005, v.summary()
    assert v.tan_delta_rel_error < 0.005, v.summary()


def test_admittance_methods_agree():
    from ctfem.geometry import build_coax
    from ctfem.eqs_solver import solve_msh
    from ctfem.observables import compute_observables
    from ctfem.config import OperatingParams, MaterialParams
    from ctfem.materials import MaterialDB, Material
    import tempfile, os

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "coax.msh")
        build_coax(0.01, 0.05, 0.2, p, lc=0.0015, verbose=False)
        matdb = MaterialDB.default()
        matdb.set("cd", Material("cd", eps_r=3.5, tan_delta=0.01))
        sol = solve_msh(p, OperatingParams(),
                        materials=MaterialParams(region_material={"dielectric": "cd"}),
                        matdb=matdb)
        obs = compute_observables(sol)
        # energy and reaction admittance methods must agree closely
        assert obs.admittance_method_discrepancy < 1e-3

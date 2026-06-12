"""Unit tests for config + materials (no gmsh/dolfinx needed)."""
import math

import numpy as np
import pytest

from ctfem.config import GeometryParams, OperatingParams, DefectSpec
from ctfem.materials import Material, MaterialDB, EPS0, coax_capacitance


def test_equal_capacitance_geometric_progression():
    g = GeometryParams(n_foils=10, foil_placement="equal_capacitance")
    radii = g.foil_radii()
    assert len(radii) == 10
    # ln(radii) should be (near) equally spaced -> constant ratio
    ratios = radii[1:] / radii[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-6)


def test_equal_spacing_linear():
    g = GeometryParams(n_foils=8, foil_placement="equal_spacing")
    radii = g.foil_radii()
    diffs = np.diff(radii)
    assert np.allclose(diffs, diffs[0], rtol=1e-6)


def test_foil_axial_stagger():
    g = GeometryParams(n_foils=5, foil_edge_margin=0.05)
    spans = [g.foil_axial_span(k) for k in range(5)]
    # each successive foil shorter at both ends
    for k in range(4):
        z0a, z1a = spans[k]
        z0b, z1b = spans[k + 1]
        assert z0b > z0a and z1b < z1a


def test_operating_u0():
    op = OperatingParams(um_kv=245.0)
    assert op.u0 == pytest.approx(245e3 / math.sqrt(3), rel=1e-9)
    assert op.omega == pytest.approx(2 * math.pi * 50.0)


def test_material_kappa_formula():
    m = Material("x", eps_r=3.5, tan_delta=0.01, sigma=0.0)
    w = 2 * math.pi * 50
    k = m.kappa(w)
    # Re(kappa) = w eps0 eps_r tan_delta ; Im = w eps0 eps_r
    assert k.imag == pytest.approx(w * EPS0 * 3.5, rel=1e-12)
    assert k.real == pytest.approx(w * EPS0 * 3.5 * 0.01, rel=1e-9)
    # loss tangent recovered
    assert k.real / k.imag == pytest.approx(0.01, rel=1e-9)


def test_temperature_hook():
    m = Material("p", eps_r=3.5, tan_delta=0.003, tan_delta_tcoef=0.02)
    hot = m.at_temperature(70.0)  # +50 degC
    assert hot.tan_delta == pytest.approx(0.003 * (1 + 0.02 * 50))


def test_coax_capacitance_formula():
    C = coax_capacitance(eps_r=2.0, r_inner=0.01, r_outer=0.05, length=1.0)
    expected = 2 * math.pi * EPS0 * 2.0 * 1.0 / math.log(5.0)
    assert C == pytest.approx(expected, rel=1e-12)


def test_defect_severity_validation():
    with pytest.raises(ValueError):
        DefectSpec(kind="moisture_ingress", severity=1.5)

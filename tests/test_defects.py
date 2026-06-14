"""Defect injection unit tests (numpy-level, no dolfinx)."""
import numpy as np
import pytest

from ctfem.config import DefectSpec, OperatingParams, MaterialParams, GeometryParams
from ctfem.defects import apply_defect
from ctfem.materials import MaterialDB


def _toy_paper_cells(n=200):
    """Synthetic paper cells spanning the paper band radially and axially."""
    g = GeometryParams()
    radii = g.foil_radii()
    rng = np.random.default_rng(0)
    r = rng.uniform(g.conductor_radius, radii[-1], n)
    z = rng.uniform(g.paper_z_bottom, g.paper_z_top, n)
    centroids = np.column_stack([r, z])
    region = np.array(["paper_insulation"] * n, dtype=object)
    matdb = MaterialDB.default()
    op = OperatingParams()
    kmap = matdb.kappa_map(op.omega)
    kappa = np.full(n, kmap["paper"], dtype=np.complex128)
    return g, centroids, region, kappa, matdb, op


def test_moisture_raises_loss_and_permittivity():
    g, c, region, kappa, matdb, op = _toy_paper_cells()
    spec = DefectSpec(kind="moisture_ingress", severity=1.0,
                      z_center=1.45, extent=0.2)
    out = apply_defect(spec, c, region, kappa, matdb, op, MaterialParams(), g)
    # cells inside the patch must change; outside unchanged.
    # (atol=0: kappa magnitudes are ~1e-8, far below allclose's default atol.)
    inside = np.abs(c[:, 1] - 1.45) <= 0.2
    assert np.any(inside)
    assert not np.allclose(out[inside], kappa[inside], atol=0.0)
    assert np.allclose(out[~inside], kappa[~inside], atol=0.0)
    # wet paper: larger imaginary part (eps_r up) and larger real part (loss up)
    assert out[inside].imag.mean() > kappa[inside].imag.mean()
    assert out[inside].real.mean() > kappa[inside].real.mean()


def test_global_aging_increases_loss_everywhere():
    g, c, region, kappa, matdb, op = _toy_paper_cells()
    spec = DefectSpec(kind="global_aging", severity=1.0)
    out = apply_defect(spec, c, region, kappa, matdb, op, MaterialParams(), g)
    assert np.all(out.real >= kappa.real - 1e-30)
    assert out.real.mean() > kappa.real.mean()


def test_shorted_foil_creates_conductive_bridge():
    # build cells deterministically inside the gap that index=1 bridges, so the
    # test is not at the mercy of random sampling of a narrow radial band.
    g = GeometryParams()
    radii = g.foil_radii()
    z0, z1 = g.foil_axial_span(1)
    zc = 0.5 * (z0 + z1)
    rr = np.linspace(radii[0], radii[1], 20)
    zz = np.linspace(zc - 0.04, zc + 0.04, 10)
    R, Z = np.meshgrid(rr, zz)
    centroids = np.column_stack([R.ravel(), Z.ravel()])
    region = np.array(["paper_insulation"] * centroids.shape[0], dtype=object)
    matdb = MaterialDB.default()
    op = OperatingParams()
    kappa = np.full(centroids.shape[0], matdb.kappa_map(op.omega)["paper"],
                    dtype=np.complex128)
    spec = DefectSpec(kind="shorted_foil", index=1, short_axial_window=0.2)
    out = apply_defect(spec, centroids, region, kappa, matdb, op,
                       MaterialParams(), g)
    # bridged cells now have metal-like (huge) real part
    assert out.real.max() > 1e3 * kappa.real.max()


def test_azimuthal_moisture_is_localized():
    # build a ring of paper cells at fixed (r, z) but varying azimuth, then apply
    # a moisture defect localized to a 60-degree wedge centred at theta=0.
    g, _, _, _, matdb, op = _toy_paper_cells()
    n = 360
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    r = np.full(n, 0.5 * (g.conductor_radius + g.foil_radii()[-1]))
    z = np.full(n, 1.45)
    centroids = np.column_stack([r, z])
    region = np.array(["paper_insulation"] * n, dtype=object)
    kappa = np.full(n, matdb.kappa_map(op.omega)["paper"], dtype=np.complex128)

    spec = DefectSpec(kind="moisture_ingress", severity=1.0, z_center=1.45,
                      extent=0.2, theta_center=0.0,
                      theta_extent=np.deg2rad(60.0))
    out = apply_defect(spec, centroids, region, kappa, matdb, op,
                       MaterialParams(), g, theta=th)

    inside = np.abs(th) <= np.deg2rad(30.0)
    assert not np.allclose(out[inside], kappa[inside], atol=0.0)   # wedge changed
    assert np.allclose(out[~inside], kappa[~inside], atol=0.0)     # rest untouched
    # roughly the right fraction of the ring is affected (~60/360)
    changed = ~np.isclose(out, kappa, atol=0.0)
    assert abs(changed.mean() - 60.0 / 360.0) < 0.05


def test_full_ring_default_ignores_azimuth():
    # default theta_extent (2*pi) must behave as a full ring even with theta given
    g, c, region, kappa, matdb, op = _toy_paper_cells()
    th = np.random.default_rng(1).uniform(-np.pi, np.pi, kappa.shape[0])
    spec = DefectSpec(kind="global_aging", severity=1.0)
    a = apply_defect(spec, c, region, kappa, matdb, op, MaterialParams(), g, theta=th)
    b = apply_defect(spec, c, region, kappa, matdb, op, MaterialParams(), g, theta=None)
    assert np.allclose(a, b)


def _toy_oil_cells(n=200, z_lo=0.55, z_hi=1.61):
    """Synthetic oil cells spanning the CVT oil annulus axially."""
    rng = np.random.default_rng(0)
    r = rng.uniform(0.075, 0.095, n)        # stack_radius .. porcelain_inner
    z = rng.uniform(z_lo, z_hi, n)
    centroids = np.column_stack([r, z])
    region = np.array(["oil"] * n, dtype=object)
    matdb = MaterialDB.default()
    op = OperatingParams()
    kappa = np.full(n, matdb.kappa_map(op.omega)["oil"], dtype=np.complex128)
    return centroids, region, kappa, matdb, op


def test_water_ingress_is_volumetric_not_dirichlet():
    # A water pocket is a pure cell-level kappa override on the OIL volume: cells
    # inside the axial window change, the rest are untouched, and nothing is
    # pinned to a potential (apply_defect never returns a BC, only kappa).
    c, region, kappa, matdb, op = _toy_oil_cells()
    spec = DefectSpec(kind="water_ingress", severity=1.0,
                      z_center=0.75, extent=0.12)
    out = apply_defect(spec, c, region, kappa, matdb, op, MaterialParams())
    inside = np.abs(c[:, 1] - 0.75) <= 0.12
    assert np.any(inside)
    assert not np.allclose(out[inside], kappa[inside], atol=0.0)
    assert np.allclose(out[~inside], kappa[~inside], atol=0.0)
    # eps_r 2.2 -> 80: the displacement (imag) part jumps ~36x
    assert out[inside].imag.mean() > 30.0 * kappa[inside].imag.mean()
    # sigma 0 -> 1e-4 S/m dominates the real (loss) part: kappa.real ~ 1e-4,
    # vastly above the oil baseline (w*eps0*eps_r*tand ~ 1e-12)
    assert np.allclose(out[inside].real, spec.defect_sigma, rtol=1e-6)


def test_water_ingress_only_touches_oil():
    # the same axial window in PAPER must be ignored (the pocket lives in oil)
    c, _, kappa, matdb, op = _toy_oil_cells()
    region = np.array(["paper_insulation"] * c.shape[0], dtype=object)
    spec = DefectSpec(kind="water_ingress", severity=1.0,
                      z_center=0.75, extent=0.12)
    out = apply_defect(spec, c, region, kappa, matdb, op, MaterialParams())
    assert np.allclose(out, kappa, atol=0.0)


def test_water_ingress_zero_severity_noop():
    c, region, kappa, matdb, op = _toy_oil_cells()
    spec = DefectSpec(kind="water_ingress", severity=0.0,
                      z_center=0.75, extent=0.12)
    out = apply_defect(spec, c, region, kappa, matdb, op, MaterialParams())
    assert np.allclose(out, kappa, atol=0.0)


def test_none_is_noop():
    g, c, region, kappa, matdb, op = _toy_paper_cells()
    out = apply_defect(DefectSpec(kind="none"), c, region, kappa, matdb, op,
                       MaterialParams(), g)
    assert np.allclose(out, kappa)

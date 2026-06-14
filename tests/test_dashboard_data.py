"""Dashboard compute-layer tests (pure; no streamlit / no FEA solve)."""
import math

from ctfem.config import CVTParams
from ctfem.dashboard_data import (
    FaultPoint, circuit_points, internal_defect_specs,
    operating_point, secondary_signature, noise_envelope, inside_envelope)

U0 = 123e3 / math.sqrt(3)
H_PT = FaultPoint("healthy", "healthy", 5623.9, 0.1666, 0.00200)


def test_internal_defect_library_shape():
    specs = internal_defect_specs(CVTParams())
    names = [n for n, _, _ in specs]
    assert names[0] == "healthy"
    assert specs[0][2] is None                       # healthy carries no defect
    assert {"C1 short", "C2 short", "water ingress"} <= set(names)


def test_circuit_points_healthy_at_origin_faults_spread():
    pts = [
        FaultPoint("healthy", "healthy", 5623.9, 0.1666, 0.00200),
        FaultPoint("C2 short", "internal", 6132.5, 0.0909, 0.00200),
        FaultPoint("water ingress", "internal", 5940.8, 0.1552, 0.12289),
        FaultPoint("RUN: streamer", "user", 5573.5, 0.1666, 0.00200),
    ]
    cps = circuit_points(pts, pts[0], burden_va=25.0, u0=U0)
    by_name = {c.name: c for c in cps}
    # healthy sits at the origin of the map
    assert abs(by_name["healthy"].ratio_err_pct) < 1e-9
    assert abs(by_name["healthy"].phase_disp_min) < 1e-9
    # a shorted C2 element is the big mover (ratio error tens of %)
    assert abs(by_name["C2 short"].ratio_err_pct) > 10.0
    # water ingress gives a measurable (arc-minute) phase displacement
    assert abs(by_name["water ingress"].phase_disp_min) > 0.5
    # categories are carried through for the scatter colouring
    assert by_name["RUN: streamer"].category == "user"


def test_circuit_points_respond_to_burden():
    pts = [FaultPoint("healthy", "healthy", 5623.9, 0.1666, 0.00200),
           FaultPoint("water ingress", "internal", 5940.8, 0.1552, 0.12289)]
    light = circuit_points(pts, pts[0], 10.0, U0)[1]
    heavy = circuit_points(pts, pts[0], 100.0, U0)[1]
    assert abs(light.phase_disp_min - heavy.phase_disp_min) > 1e-6


def test_operating_point_internal_faults_shift_divider():
    # C1 short raises C1 -> tap fraction UP; C2 short -> tap fraction DOWN
    op_c1 = operating_point(H_PT, c1_short=0.10)
    op_c2 = operating_point(H_PT, c2_short=0.10)
    assert op_c1.tap_frac > H_PT.tap_frac > op_c2.tap_frac
    assert op_c1.C_pF > H_PT.C_pF and op_c2.C_pF > H_PT.C_pF
    assert op_c1.leakage_mA == 0.0          # internal faults: no surface leakage


def test_operating_point_moisture_scales_C_keeps_ratio():
    op = operating_point(H_PT, moisture=0.10)
    assert op.C_pF > H_PT.C_pF
    assert abs(op.tap_frac - H_PT.tap_frac) < 1e-9     # uniform -> ratio unchanged


def test_operating_point_temperature_coeff_sign():
    cold = operating_point(H_PT, temperature_c=-20.0)
    hot = operating_point(H_PT, temperature_c=40.0)
    # negative oil-paper temp. coefficient: colder -> higher capacitance
    assert cold.C_pF > H_PT.C_pF > hot.C_pF


def test_default_operating_point_is_healthy():
    op = operating_point(H_PT)                # all sliders at 0, T = 20 C
    assert abs(op.C_pF - H_PT.C_pF) < 1e-6
    assert abs(op.tap_frac - H_PT.tap_frac) < 1e-12


def test_grid_frequency_detunes_reactor():
    nominal = secondary_signature(H_PT, H_PT, 25.0, U0, grid_freq=50.0)
    off = secondary_signature(H_PT, H_PT, 25.0, U0, grid_freq=50.2)
    assert abs(nominal.phase_disp_min) < 1e-6         # healthy @ 50 Hz -> origin
    # off-nominal frequency detunes the (50 Hz-tuned) reactor -> nonzero signal
    assert abs(off.phase_disp_min) > 1e-4 or abs(off.ratio_err_pct) > 1e-4


def test_temperature_raises_tan_delta():
    cold = operating_point(H_PT, temperature_c=-20.0)
    hot = operating_point(H_PT, temperature_c=40.0)
    # oil-paper loss climbs with heat -> summer tan delta noise
    assert hot.tan_delta > H_PT.tan_delta > cold.tan_delta


def test_noise_envelope_is_convex_hull_polygon():
    env = noise_envelope(H_PT, 25.0, U0)
    assert env.ndim == 2 and env.shape[0] >= 3 and env.shape[1] == 2
    # the hull centroid is strictly inside the fault-free drift locus
    cx, cy = float(env[:, 0].mean()), float(env[:, 1].mean())
    assert inside_envelope(env, cx, cy)
    # a 10 % C1 short lands clearly OUTSIDE the weather-drift hull (detectable)
    op = operating_point(H_PT, c1_short=0.10)
    cp = secondary_signature(op, H_PT, 25.0, U0)
    assert not inside_envelope(env, cp.ratio_err_pct, cp.phase_disp_min)

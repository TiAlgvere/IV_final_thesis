"""Lumped CVT circuit model tests (pure phasor math, no FEA/mesh)."""
import math

from ctfem.cvt_circuit import CVTCircuit, c1_c2_from_fea, solve_from_fea

U1 = 123e3 / math.sqrt(3)            # DDB-123 phase-to-earth amplitude [V]
# representative healthy FEA observables (2-D, refine 1.0)
H = dict(c_term_f=5623.9e-12, tap_frac=0.1666, tan_delta=0.00200)


def test_c1_c2_roundtrip():
    c1, c2 = c1_c2_from_fea(H["c_term_f"], H["tap_frac"])
    assert abs(c1 * c2 / (c1 + c2) - H["c_term_f"]) / H["c_term_f"] < 1e-9
    assert abs(c1 / (c1 + c2) - H["tap_frac"]) < 1e-9


def test_reactor_tunes_to_resonance():
    ck = CVTCircuit()
    solve_from_fea(ck, H["c_term_f"], H["tap_frac"], H["tan_delta"], U1,
                   tune=True)
    c1, c2 = c1_c2_from_fea(H["c_term_f"], H["tap_frac"])
    w = ck.omega
    # series compensation: w^2 L (C1+C2) = 1
    assert abs(w * w * ck.reactor_l * (c1 + c2) - 1.0) < 1e-9


def test_healthy_phase_is_small():
    ck = CVTCircuit()
    h = solve_from_fea(ck, H["c_term_f"], H["tap_frac"], H["tan_delta"], U1,
                       tune=True)
    # a tuned CVT at rated burden has a small phase displacement (a few arcmin)
    assert abs(h.phase_min) < 10.0


def test_capacitance_fault_detunes_and_is_measurable():
    ck = CVTCircuit()
    h = solve_from_fea(ck, H["c_term_f"], H["tap_frac"], H["tan_delta"], U1,
                       tune=True)
    # water ingress: C_term up ~5.7%, tap down ~6.9%, big loss
    w = solve_from_fea(ck, 5944.7e-12, 0.1551, 0.12203, U1, tune=False)
    # detuning the (fixed) reactor produces a measurable secondary phase shift
    assert abs(w.phase_min - h.phase_min) > 0.5            # > 0.5 arc-minute
    # and a several-percent ratio error tracking the divider change
    assert abs(abs(w.u_int) / abs(h.u_int) - 1.0) > 0.02


def test_shorted_element_raises_ratio():
    ck = CVTCircuit()
    h = solve_from_fea(ck, H["c_term_f"], H["tap_frac"], H["tan_delta"], U1,
                       tune=True)
    # a shorted C1 element raises C_term and the tap fraction -> ratio rises
    s = solve_from_fea(ck, 6135.0e-12, 0.1818, 0.00200, U1, tune=False)
    assert abs(s.u_int) > abs(h.u_int)


def test_heavier_burden_shifts_phase():
    # the phase displacement depends on the burden (a sweepable design knob)
    light = CVTCircuit(burden_va=10.0)
    heavy = CVTCircuit(burden_va=100.0)
    rl = solve_from_fea(light, H["c_term_f"], H["tap_frac"], H["tan_delta"], U1,
                        tune=True)
    rh = solve_from_fea(heavy, H["c_term_f"], H["tap_frac"], H["tan_delta"], U1,
                        tune=True)
    assert abs(rl.phase_min - rh.phase_min) > 1e-6

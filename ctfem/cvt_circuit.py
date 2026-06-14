"""Lumped CVT circuit model: turn FEA divider parameters into the *measurable*
secondary ratio error and phase displacement.

A real capacitive voltage transformer is more than the C1/C2 divider the FEA
resolves.  The intermediate (tap) voltage feeds a series compensating REACTOR L,
tuned to resonate with the divider capacitance C_eq = C1 + C2 at line frequency,
then an intermediate voltage transformer (IVT) + burden.  At resonance the
reactor cancels the divider's source reactance, so the secondary tracks the
primary in magnitude and phase -- *as long as the capacitance stays at its tuned
value*.

This is why the lumped model matters for fault prediction: a fault that shifts
C1/C2 (water ingress, a shorted element) or adds loss (pollution) DETUNES the
reactor, producing a secondary phase displacement / ratio error far larger and
more measurable than the bare divider-node phase the FEA reports.  Metrology
defines these as the CVT ratio error and phase displacement (IEC 61869-5).

Phasor model (Thevenin at the tap; IVT taken ideal with the burden referred to
the intermediate side -- the IVT ratio then cancels from the phase angle):

    Y1 = G1 + j w C1,   Y2 = G2 + j w C2           (sections; G_i = w C_i tand)
    U_th = U1 Y1/(Y1+Y2),    Z_th = 1/(Y1+Y2)       (open-circuit tap + source Z)
    Z_L  = R_L + j w L,   L = 1/(w^2 C_eq_healthy),  R_L = w L / Q
    Z_b  = (V_int^2 / S_VA)(pf + j sqrt(1-pf^2))     (burden, referred)
    U_int = U_th Z_b / (Z_th + Z_L + Z_b)
    phase displacement = arg(U_int);  ratio from |U_int|.

The reactor is tuned ONCE on the healthy state (:meth:`CVTCircuit.tune`); faults
are then evaluated with that fixed L, so detuning shows up.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Optional


def c1_c2_from_fea(c_term_f: float, tap_frac: float) -> tuple[float, float]:
    """Recover the section capacitances C1, C2 [F] from the terminal (series)
    capacitance and the tap voltage fraction tap_frac = C1/(C1+C2).

    Two series caps give C_term = C1 C2/(C1+C2) and tap_frac = C1/(C1+C2), so
    C1+C2 = C_term/(tap_frac (1-tap_frac)).
    """
    if not (0.0 < tap_frac < 1.0):
        raise ValueError(f"tap_frac must be in (0,1), got {tap_frac}")
    s = c_term_f / (tap_frac * (1.0 - tap_frac))
    return tap_frac * s, (1.0 - tap_frac) * s


@dataclass
class CVTResponse:
    """One operating point of the lumped CVT."""
    c1_pF: float
    c2_pF: float
    u_int: complex                 # intermediate-side voltage phasor [V]
    phase_deg: float               # secondary phase displacement [deg]

    @property
    def phase_min(self) -> float:
        """Phase displacement in arc-minutes (the metrology unit)."""
        return self.phase_deg * 60.0


@dataclass
class CVTCircuit:
    """Lumped compensating-reactor + burden model around the FEA divider."""

    frequency: float = 50.0
    reactor_q: float = 80.0              # reactor quality factor w L / R_L
    burden_va: float = 25.0              # rated secondary burden [VA]
    burden_pf: float = 0.8               # burden power factor (lagging, inductive)
    reactor_l: Optional[float] = None    # H; None until tuned
    v_int: float = 0.0                   # intermediate voltage [V] (set on tune)

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.frequency

    def tune(self, c_eq_f: float, v_int: float) -> None:
        """Tune the reactor to resonate with C_eq = C1+C2 at line frequency."""
        self.reactor_l = 1.0 / (self.omega ** 2 * c_eq_f)
        self.v_int = v_int

    def _z_burden(self) -> complex:
        pf = self.burden_pf
        x = math.sqrt(max(0.0, 1.0 - pf * pf))     # inductive (lagging) burden
        return (self.v_int ** 2 / self.burden_va) * (pf + 1j * x)

    def solve(self, c1: float, c2: float, tan_delta: float,
              u1: float) -> CVTResponse:
        """Secondary response for a given divider state (reactor already tuned).

        `tan_delta` is the (terminal) divider loss tangent, applied uniformly to
        both sections; `u1` is the applied HV phase-to-earth amplitude [V].
        """
        if self.reactor_l is None:
            raise RuntimeError("tune() the reactor on the healthy state first")
        w = self.omega
        y1 = w * c1 * tan_delta + 1j * w * c1
        y2 = w * c2 * tan_delta + 1j * w * c2
        u_th = u1 * y1 / (y1 + y2)
        z_th = 1.0 / (y1 + y2)
        z_l = (w * self.reactor_l / self.reactor_q) + 1j * w * self.reactor_l
        z_b = self._z_burden()
        u_int = u_th * z_b / (z_th + z_l + z_b)
        return CVTResponse(c1 * 1e12, c2 * 1e12, u_int,
                           math.degrees(cmath.phase(u_int)))


def solve_from_fea(circuit: CVTCircuit, c_term_f: float, tap_frac: float,
                   tan_delta: float, u1: float, *, tune: bool = False
                   ) -> CVTResponse:
    """Convenience: recover C1/C2 from FEA observables and solve the circuit.

    Pass ``tune=True`` for the healthy baseline (sets the reactor L and the
    intermediate voltage); evaluate faults with ``tune=False`` so the fixed
    reactor detunes.
    """
    c1, c2 = c1_c2_from_fea(c_term_f, tap_frac)
    if tune or circuit.reactor_l is None:
        circuit.tune(c1 + c2, tap_frac * u1)
    return circuit.solve(c1, c2, tan_delta, u1)

"""Backend-independent result types (no dolfinx/skfem imports).

The Observables dataclass is shared by every solver backend (DOLFINx on
Linux/HPC, scikit-fem on Windows) so that scripts, tests and the sweep engine
see one schema regardless of where the solve ran.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observables:
    """Tidy per-solve result row (admittance, ladder, field stress)."""

    Y: complex
    C1_pF: float
    tan_delta: float
    Y_reaction: complex
    admittance_method_discrepancy: float    # |Y - Y_reaction| / |Y|
    foil_potentials: list[complex] = field(default_factory=list)
    foil_potential_frac: list[float] = field(default_factory=list)  # |phi|/U0
    peak_field_per_gap: list[float] = field(default_factory=list)   # V/m
    peak_field_overall: float = 0.0
    surface_leakage_mA: float = 0.0    # insulator-surface leakage current to gnd

    def row(self) -> dict:
        """Flat dict for DataFrame export."""
        d = {
            "C1_pF": self.C1_pF,
            "tan_delta": self.tan_delta,
            "Y_real": self.Y.real,
            "Y_imag": self.Y.imag,
            "admittance_discrepancy": self.admittance_method_discrepancy,
            "peak_field_overall": self.peak_field_overall,
        }
        for i, v in enumerate(self.foil_potential_frac):
            d[f"foil{i + 1}_frac"] = v
        for i, v in enumerate(self.peak_field_per_gap):
            d[f"gap{i + 1}_peakE"] = v
        return d

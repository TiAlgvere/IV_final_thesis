"""Material database and the complex-coefficient helper.

Electro-quasistatic (EQS) constitutive model
--------------------------------------------
Each linear, isotropic dielectric is described by a complex relative
permittivity and an optional DC conductivity.  Under harmonic excitation
exp(+j w t) the conduction + displacement current density is

    J_total = (sigma + j w eps0 eps_r_complex) E,                       (1)

and we fold the dielectric loss tangent into a complex permittivity:

    eps_r_complex = eps_r * (1 - j * tan_delta).                        (2)

So the scalar coefficient that multiplies grad(phi) in the weak form is

    kappa = sigma + j w eps0 eps_r * (1 - j tan_delta)
          = (sigma + w eps0 eps_r tan_delta)  +  j (w eps0 eps_r).      (3)

Reference: H.A. Haus & J.R. Melcher, *Electromagnetic Fields and Energy*,
Ch. 7 (quasistatics); standard condenser-bushing loss modelling.

Sign convention note
--------------------
With the exp(+j w t) convention a *lossy* dielectric has a permittivity with a
*negative* imaginary part (eq. 2).  The real part of `kappa` (eq. 3) is then
positive => positive dissipated power, as required.  The terminal admittance
Y = G + j w C therefore has Re(Y) = G > 0 and tan_delta = Re(Y)/Im(Y) > 0.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

import numpy as np

EPS0: float = 8.8541878128e-12   # vacuum permittivity [F/m]

# Numerical conductivity cap for metals in the EQS coefficient.
# Physical aluminium (sigma = 3.5e7 S/m) against dielectric kappa ~ 1e-8 S/m
# gives a matrix-entry contrast of ~3.5e15 -- at the edge of double precision,
# so LU round-off inside the metal regions produces SPURIOUS dissipation that
# can exceed the true dielectric loss by orders of magnitude (observed:
# tan delta ~ 6.5 instead of ~3e-3 on the full CT).  A floating conductor only
# needs |kappa_metal| >> |kappa_dielectric| to behave as an equipotential;
# sigma = 1 S/m gives a contrast of ~1e8 at 50 Hz -- physically equivalent
# grading behaviour (foil potential error ~1e-8 relative), numerically benign
# (spurious loss ~1e-6 relative).  Pass sigma_cap=None for the raw values.
SIGMA_NUMERICAL_CAP: float = 1.0   # S/m


@dataclass(frozen=True)
class Material:
    """A linear isotropic dielectric/conductor.

    Attributes
    ----------
    name : str
    eps_r : float
        Relative permittivity (real part).
    tan_delta : float
        Loss tangent (dimensionless).  Folded in via eq. (2).
    sigma : float
        DC conductivity [S/m].  For good conductors (foils, electrodes) this
        dominates; for dielectrics it is usually ~0 and loss comes from
        tan_delta.
    eps_r_tref : float
        Reference temperature [degC] at which eps_r/tan_delta are quoted.
    tan_delta_tcoef : float
        Linear temperature coefficient of tan_delta [1/degC]; simple hook for
        tan_delta(T) = tan_delta * (1 + tcoef * (T - tref)).  (Phase-5 sweeps.)
    """

    name: str
    eps_r: float
    tan_delta: float = 0.0
    sigma: float = 0.0
    eps_r_tref: float = 20.0
    tan_delta_tcoef: float = 0.0

    def at_temperature(self, temperature_c: float) -> "Material":
        """Return a copy with tan_delta adjusted for temperature (linear hook).

        Real oil-paper insulation loss is strongly temperature dependent; we
        expose a simple linear model so Phase-5 sweeps can vary T.  Replace with
        an Arrhenius fit when measured data is available.
        """
        if self.tan_delta_tcoef == 0.0:
            return self
        dt = temperature_c - self.eps_r_tref
        new_td = max(0.0, self.tan_delta * (1.0 + self.tan_delta_tcoef * dt))
        return replace(self, tan_delta=new_td)

    def kappa(self, omega: float, sigma_cap: float | None = None) -> complex:
        """Complex coefficient kappa = sigma + j w eps0 eps_r (1 - j tan_delta).

        See eq. (3).  `sigma_cap` limits the conductivity actually used in the
        FEM coefficient (see SIGMA_NUMERICAL_CAP) without altering the stored
        physical value.
        """
        sigma = self.sigma if sigma_cap is None else min(self.sigma, sigma_cap)
        eps_complex = self.eps_r * (1.0 - 1j * self.tan_delta)
        return sigma + 1j * omega * EPS0 * eps_complex


# --------------------------------------------------------------------------- #
# Default database
# --------------------------------------------------------------------------- #
# Values per the thesis spec.  Metals use a representative aluminium-like sigma
# (3.5e7 S/m) -- the foils float to their natural potentials in the EQS
# formulation purely because of this high conductivity (no Dirichlet BC).

_DEFAULTS: Dict[str, Material] = {
    "air": Material("air", eps_r=1.0, tan_delta=0.0, sigma=0.0),
    "oil": Material("oil", eps_r=2.2, tan_delta=0.001, sigma=0.0),
    "paper": Material(
        "paper", eps_r=3.5, tan_delta=0.003, sigma=0.0,
        # crude positive temperature dependence of loss for the T-hook
        tan_delta_tcoef=0.02,
    ),
    "porcelain": Material("porcelain", eps_r=6.0, tan_delta=0.005, sigma=0.0),
    "metal": Material("metal", eps_r=1.0, tan_delta=0.0, sigma=3.5e7),
}


class MaterialDB:
    """Mutable material database (a copy is made per case so defects are local).

    Use :meth:`get` to look up a material, :meth:`set` to override one (defects),
    and :meth:`kappa_map` to produce the {region_name -> complex kappa} mapping
    the solver assigns to mesh cells.
    """

    def __init__(self, materials: Dict[str, Material] | None = None) -> None:
        src = materials if materials is not None else _DEFAULTS
        self._mats: Dict[str, Material] = {k: v for k, v in src.items()}

    @classmethod
    def default(cls) -> "MaterialDB":
        return cls(_DEFAULTS)

    def copy(self) -> "MaterialDB":
        return MaterialDB(self._mats)

    def get(self, name: str) -> Material:
        if name not in self._mats:
            raise KeyError(f"unknown material {name!r}; have {sorted(self._mats)}")
        return self._mats[name]

    def set(self, name: str, material: Material) -> None:
        self._mats[name] = material

    def names(self) -> list[str]:
        return sorted(self._mats)

    def kappa_map(
        self,
        omega: float,
        temperature_c: float = 20.0,
        sigma_cap: float | None = SIGMA_NUMERICAL_CAP,
    ) -> Dict[str, complex]:
        """Map every material name to its complex EQS coefficient at (omega, T).

        Metal conductivities are capped at `sigma_cap` by default (see the
        SIGMA_NUMERICAL_CAP comment) -- pass None for raw physical values.
        """
        return {
            name: mat.at_temperature(temperature_c).kappa(omega, sigma_cap)
            for name, mat in self._mats.items()
        }


def coax_capacitance(eps_r: float, r_inner: float, r_outer: float, length: float) -> float:
    """Analytic capacitance of an ideal coaxial cylinder capacitor [F].

        C = 2 pi eps0 eps_r L / ln(r_outer / r_inner).

    Used by the Phase-2 analytic benchmark.
    """
    return 2.0 * np.pi * EPS0 * eps_r * length / np.log(r_outer / r_inner)

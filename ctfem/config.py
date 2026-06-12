"""Configuration dataclasses for the CT-FEM framework.

All tunable physical/geometric quantities live here (or in :mod:`ctfem.materials`)
so that no "magic numbers" are scattered through the solver/geometry code.

Defaults match the reference device: an Arteche / VA TECH hairpin (top-core)
HV current transformer, ~245 kV class.

    total height            ~= 3315 mm
    porcelain section       ~= 1900 mm
    base width              ~= 425 mm   (radius ~212.5 mm)
    head housing width      ~= 1020-1090 mm (radius ~510-545 mm)

The model is a 2-D **axisymmetric** (r, z) half-section.  All lengths are in
metres unless noted.  z increases upward; z = 0 is the bottom of the base.

When real Elering unit data arrives, edit the field defaults below (or pass a
populated :class:`GeometryParams` to the builder) -- nothing else needs to change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

FoilPlacement = Literal["equal_spacing", "equal_capacitance"]


@dataclass
class GeometryParams:
    """Parametric description of the axisymmetric CT half-section.

    The geometry is intentionally simplified (smooth porcelain cone, single
    equivalent axial primary conductor, thin foil annuli) -- see README for the
    list of modelling assumptions and their justification.
    """

    # -- overall envelope (metres) ----------------------------------------- #
    total_height: float = 3.315          # full device height
    base_height: float = 0.300           # grounded base/tank section height
    base_radius: float = 0.2125          # base outer radius (~425 mm wide)
    porcelain_height: float = 1.900      # porcelain insulator section height
    head_height: float = 0.700           # head housing height (top)
    head_radius: float = 0.525           # head outer radius (~1050 mm wide)

    # -- primary conductor -------------------------------------------------- #
    # The hairpin primary is modelled as a single equivalent axial conductor.
    conductor_radius: float = 0.030      # 30 mm default

    # -- paper insulation body (slightly conical) --------------------------- #
    # The condenser body tapers: larger radius at the grounded (base) end,
    # smaller near the head, mimicking a real graded condenser core.  It starts
    # ABOVE the grounded base (oil gap) so the HV conductor never touches ground.
    paper_inner_radius: float = 0.030    # hugs the conductor
    paper_outer_radius_base: float = 0.180   # radius at base end
    paper_outer_radius_head: float = 0.130   # radius at head end
    paper_z_bottom: float = 0.400        # axial start of the paper body (> base)
    paper_z_top: float = 2.550           # axial end of the paper body

    # -- grounded ground-sleeve (the C1 outer reference) -------------------- #
    # Real condenser cores ground the OUTERMOST foil via the mounting flange.
    # We model that ground reference as a thin vertical metal sleeve just
    # outside the paper body (named "base_tank": it is part of the grounded
    # metalwork). Without it the conductor-to-ground capacitance would be
    # unphysically small.  This is what sets C1 to the hundreds-of-pF range.
    ground_sleeve_radius: float = 0.190      # just outside paper_outer_radius_base
    ground_sleeve_thickness: float = 0.015

    # -- grading foils ------------------------------------------------------ #
    n_foils: int = 10                    # number of capacitive grading foils
    foil_thickness: float = 0.00015      # 0.15 mm modelled annulus thickness (2D)
    # In 3D, thin 0.15 mm revolved shells wreck the tet mesher and the foil
    # thickness is fictitious anyway (C is set by foil RADIUS), so the 3D builder
    # uses a thicker, mesh-friendly annulus by default.
    foil_thickness_3d: float = 0.0015    # 1.5 mm annulus for the 3D revolve
    foil_placement: FoilPlacement = "equal_capacitance"
    # staggered axial lengths: each successive (outer) foil is shorter at both
    # ends by `foil_edge_margin` * its index, as in real condenser bushings.
    foil_edge_margin: float = 0.060      # axial shortening step per foil (m)
    # radial band the foils occupy inside the paper (fractions of paper gap)
    foil_radial_inner_frac: float = 0.10
    foil_radial_outer_frac: float = 0.95

    # -- oil volume --------------------------------------------------------- #
    oil_outer_radius: float = 0.300      # oil out to porcelain inner wall

    # -- porcelain wall ----------------------------------------------------- #
    porcelain_thickness: float = 0.040   # smooth hollow cone wall thickness
    # NOTE(v1): no sheds -- external surface phenomena are out of scope.
    # TODO(sheds): add shed profile hook for future external-field studies.

    # -- far field ---------------------------------------------------------- #
    # Outer air boundary radius; spec requires >= 3x device height.
    farfield_factor: float = 3.0
    farfield_radius: Optional[float] = None  # computed if None

    # -- meshing ------------------------------------------------------------ #
    # Global refinement factor (>1 finer, <1 coarser) for convergence studies.
    mesh_refinement: float = 1.0
    # characteristic lengths (metres) BEFORE refinement factor is applied.
    # effective lc = lc / mesh_refinement (so refinement<1 => coarser/faster).
    lc_foil: float = 0.0030              # fine, near foil edges
    lc_paper: float = 0.012              # paper bulk between foils
    lc_oil: float = 0.060
    lc_air: float = 0.600                # coarse, far field
    # Distance/Threshold field band around the foil edges (only a thin shell is
    # forced to lc_foil; the paper bulk is governed by point-based lc_paper)
    refine_dist_min: float = 0.001
    refine_dist_max: float = 0.008

    def __post_init__(self) -> None:
        if self.farfield_radius is None:
            self.farfield_radius = self.farfield_factor * self.total_height
        if self.n_foils < 0:
            raise ValueError("n_foils must be >= 0")

    # -- derived helpers ---------------------------------------------------- #
    def foil_radii(self) -> np.ndarray:
        """Return the N foil mean radii according to the placement strategy.

        For a coaxial-like geometry the capacitance of the annular gap between
        radii r_a < r_b (per unit length) is

            C_gap = 2 pi eps / ln(r_b / r_a).            (axisymmetric coax)

        *equal_capacitance* grading makes every successive gap carry the same
        capacitance => the ln(r) values are equally spaced => radii follow a
        geometric progression between the conductor and the paper outer radius.
        This is the classic condenser-bushing grading.  *equal_spacing* spaces
        the radii linearly (a deliberately sub-optimal reference case).
        """
        n = self.n_foils
        if n == 0:
            return np.empty(0)
        r_inner = self.paper_inner_radius + self.foil_radial_inner_frac * (
            self.paper_outer_radius_base - self.paper_inner_radius
        )
        r_outer = self.paper_inner_radius + self.foil_radial_outer_frac * (
            self.paper_outer_radius_base - self.paper_inner_radius
        )
        # place N foils as interior dividers between conductor and ground shell
        if self.foil_placement == "equal_capacitance":
            # geometric progression: ln-spaced radii
            ln_vals = np.linspace(math.log(r_inner), math.log(r_outer), n)
            return np.exp(ln_vals)
        elif self.foil_placement == "equal_spacing":
            return np.linspace(r_inner, r_outer, n)
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown foil_placement {self.foil_placement!r}")

    def foil_axial_span(self, k: int) -> tuple[float, float]:
        """(z_bottom, z_top) of foil k (0-based, innermost first).

        Each successive foil is shortened symmetrically at both ends so that
        the outer foils are shorter -- this is what spreads the axial field and
        rounds the equipotential lines at the foil edges in real condenser cores.
        """
        margin = self.foil_edge_margin * k
        z0 = self.paper_z_bottom + margin
        z1 = self.paper_z_top - margin
        if z1 <= z0:
            raise ValueError(
                f"foil {k} has non-positive axial span; reduce foil_edge_margin "
                f"or n_foils (got z0={z0}, z1={z1})"
            )
        return z0, z1


# --------------------------------------------------------------------------- #
# CVT geometry (Arteche DDB/DFK capacitive voltage transformer)
# --------------------------------------------------------------------------- #


@dataclass
class CVTParams:
    """Parametric axisymmetric model of an Arteche DDB-123 capacitive voltage
    transformer (the 110 kV-side unit used in the Estonian grid).

    Datasheet anchors (ARTECHE DDB/DFK series, model DDB-123):
        highest voltage Um        123 kV
        standard capacitance      5600 pF      (the validation target)
        total height H            1830 mm
        base width A              450 mm
        BIL                       550 kVp

    Architecture (top to bottom): HV head with the oil-volume compensator, a
    hollow porcelain insulator containing the series capacitor stack immersed
    in oil, and the grounded EMU tank.  The stack of `n_elements` series
    capacitor elements forms the C1/C2 voltage divider; the intermediate
    voltage tap sits between the C1 section (`n_elements - n_elements_c2`
    elements) and the C2 section (`n_elements_c2` elements).

    Homogenized capacitor elements
    ------------------------------
    A real element is a WOUND paper-film capacitor (many m^2 of foil rolled
    into one can) -- its internal turns cannot be meshed.  Each element is
    homogenized as a solid dielectric cylinder between two metal electrode
    discs, with an effective permittivity chosen so the element carries its
    rated series capacitance in its real volume:

        eps_r_eff = C_element * h_element / (eps0 * pi * r_stack^2),
        C_element = n_elements * C_rated   (N equal elements in series).

    This preserves the terminal capacitance, the loss tangent, the divider
    voltage distribution and the field in the surrounding oil/porcelain.  It
    does NOT resolve the field inside a wound element (out of scope).
    """

    # -- nameplate / envelope (metres) -------------------------------------- #
    total_height: float = 1.830          # datasheet H
    tank_radius: float = 0.225           # datasheet A = 450 mm base width
    tank_height: float = 0.550           # grounded EMU tank section
    # HV head: the wide oil-volume-compensator dome (acts like a corona ring,
    # shaping the field pushing down over the sheds) + primary terminal stub.
    head_radius: float = 0.180           # compensator dome (~360 mm dia)
    head_height: float = 0.180           # dome height
    terminal_radius: float = 0.020       # primary terminal stub on the dome
    terminal_height: float = 0.040

    # -- porcelain insulator ------------------------------------------------- #
    porcelain_inner_radius: float = 0.095
    porcelain_thickness: float = 0.025
    # weather sheds (ribs): the datasheet creepage distance (3075 mm for the
    # DDB-123) is achieved by these folds; field concentrates at their tips.
    n_sheds: int = 27
    shed_overhang: float = 0.040         # radial reach beyond the wall
    shed_thickness_root: float = 0.018   # axial thickness at the wall
    shed_thickness_tip: float = 0.008    # axial thickness at the tip
    shed_droop: float = 0.010            # tip sits this much below the root
    rated_creepage_mm: float = 3075.0    # datasheet anchor (validation)

    # -- secondary terminal box on the tank side (3-D ONLY: it breaks the
    #    rotational symmetry, so the 2-D axisymmetric model cannot carry it) -- #
    secondary_box: bool = True
    secondary_box_depth: float = 0.15    # radial protrusion
    secondary_box_width: float = 0.20    # tangential width
    secondary_box_height: float = 0.25
    secondary_box_z0: float = 0.15       # bottom edge above ground

    # -- series capacitor stack --------------------------------------------- #
    n_elements: int = 12                 # total series elements (model count)
    n_elements_c2: int = 2               # elements below the tap (C2 section)
    stack_radius: float = 0.075          # element/disc radius
    disc_thickness: float = 0.006        # electrode disc thickness
    rated_capacitance_pF: float = 5600.0  # datasheet "standard capacitance"
    element_tan_delta: float = 0.002     # oil-impregnated paper-film element

    # -- far field ----------------------------------------------------------- #
    farfield_factor: float = 3.0
    farfield_radius: Optional[float] = None  # computed if None

    # -- meshing (same vocabulary as GeometryParams; duck-typed by the
    #    builders: lc_foil = discs, lc_paper = elements/solids) -------------- #
    mesh_refinement: float = 1.0
    lc_foil: float = 0.004               # electrode discs / disc-edge shell
    lc_paper: float = 0.015              # element dielectric, porcelain, metal
    lc_shed: float = 0.008               # shed profiles (field peaks at tips)
    lc_oil: float = 0.040
    lc_air: float = 0.500
    refine_dist_min: float = 0.002
    refine_dist_max: float = 0.012

    def __post_init__(self) -> None:
        if self.farfield_radius is None:
            self.farfield_radius = self.farfield_factor * self.total_height
        if self.n_elements < 2:
            raise ValueError("n_elements must be >= 2")
        if not (1 <= self.n_elements_c2 <= self.n_elements - 1):
            raise ValueError("n_elements_c2 must be in [1, n_elements-1]")
        if self.stack_radius >= self.porcelain_inner_radius:
            raise ValueError("stack_radius must be < porcelain_inner_radius")
        if self.element_height() <= 0:
            raise ValueError("stack does not fit: reduce n_elements or "
                             "disc_thickness")

    # -- derived stack layout (z, top-down indexing: element 1 at the HV end) #
    @property
    def stack_z_lo(self) -> float:
        """Bottom of the stack column (= tank top)."""
        return self.tank_height

    @property
    def stack_z_hi(self) -> float:
        """Top of the stack column (= compensator-dome bottom)."""
        return self.total_height - self.head_height - self.terminal_height

    def element_height(self) -> float:
        """Axial height of one homogenized element dielectric block."""
        interior = (self.stack_z_hi - self.disc_thickness) - (
            self.stack_z_lo + self.disc_thickness)
        return (interior - (self.n_elements - 1) * self.disc_thickness) \
            / self.n_elements

    def element_span(self, k: int) -> tuple[float, float]:
        """(z_bottom, z_top) of element k, 1-based, k=1 at the TOP (HV side)."""
        if not (1 <= k <= self.n_elements):
            raise ValueError(f"element index {k} out of [1, {self.n_elements}]")
        h = self.element_height()
        z_top = (self.stack_z_hi - self.disc_thickness) \
            - (k - 1) * (h + self.disc_thickness)
        return z_top - h, z_top

    def disc_span(self, k: int) -> tuple[float, float]:
        """(z_bottom, z_top) of interior disc k (between elements k and k+1)."""
        if not (1 <= k <= self.n_elements - 1):
            raise ValueError(f"disc index {k} out of [1, {self.n_elements - 1}]")
        z_bot_el, _ = self.element_span(k)
        return z_bot_el - self.disc_thickness, z_bot_el

    # -- shed profile (axisymmetric trapezoid, revolved into an annular rib) - #
    def shed_centers(self) -> list[float]:
        """Axial mid-height of each shed root, evenly spread over the wall."""
        h_ins = self.stack_z_hi - self.tank_height
        pitch = h_ins / self.n_sheds
        return [self.tank_height + (k - 0.5) * pitch
                for k in range(1, self.n_sheds + 1)]

    def shed_polygon(self, zc: float) -> list[tuple[float, float]]:
        """(r, z) trapezoid of one shed: root buried in the wall, drooping tip.

        The root starts at mid-wall (overlapping the porcelain rectangle --
        safe because both classify as porcelain) so OCC never leaves a sliver
        gap between shed and wall.
        """
        r_root = self.porcelain_inner_radius + 0.5 * self.porcelain_thickness
        r_tip = (self.porcelain_inner_radius + self.porcelain_thickness
                 + self.shed_overhang)
        tr, tt, d = (self.shed_thickness_root, self.shed_thickness_tip,
                     self.shed_droop)
        return [(r_root, zc - 0.5 * tr),
                (r_tip, zc - d - 0.5 * tt),
                (r_tip, zc - d + 0.5 * tt),
                (r_root, zc + 0.5 * tr)]

    def creepage_distance(self) -> float:
        """Surface path [m] along the outer insulator profile (tank to dome).

        Walks the revolved profile: the exposed wall column between sheds plus
        each shed's lower slant, tip face and upper slant, with the slants
        clipped exactly where they emerge from the wall face.  Validation
        target: the datasheet standard creepage distance (`rated_creepage_mm`).
        """
        h_ins = self.stack_z_hi - self.tank_height
        r_wall = self.porcelain_inner_radius + self.porcelain_thickness
        A, B, C, D = self.shed_polygon(0.0)

        def _wall_exit(p_in, p_out):
            t = (r_wall - p_in[0]) / (p_out[0] - p_in[0])
            return (r_wall, p_in[1] + t * (p_out[1] - p_in[1]))

        a = _wall_exit(A, B)    # bottom slant leaves the wall here
        d = _wall_exit(D, C)    # top slant leaves the wall here
        per_shed = (math.hypot(B[0] - a[0], B[1] - a[1])
                    + math.hypot(C[0] - B[0], C[1] - B[1])
                    + math.hypot(d[0] - C[0], d[1] - C[1]))
        covered = d[1] - a[1]   # wall span hidden behind each shed root
        return (h_ins - self.n_sheds * covered) + self.n_sheds * per_shed

    # -- derived electrical quantities --------------------------------------- #
    def element_epsr_eff(self) -> float:
        """Effective eps_r so each element carries its rated capacitance."""
        from .materials import EPS0
        c_elem = self.n_elements * self.rated_capacitance_pF * 1e-12
        area = math.pi * self.stack_radius ** 2
        return c_elem * self.element_height() / (EPS0 * area)

    @property
    def tap_disc_index(self) -> int:
        """Interior-disc index of the intermediate voltage tap (C1/C2 joint)."""
        return self.n_elements - self.n_elements_c2

    def divider_ratio_nominal(self) -> float:
        """Nominal tap voltage fraction V_tap/U0 = C1/(C1+C2) = N2/N."""
        return self.n_elements_c2 / self.n_elements

    def material_db(self):
        """MaterialDB with the homogenized element dielectric registered."""
        from .materials import Material, MaterialDB
        db = MaterialDB.default()
        db.set("element_dielectric", Material(
            "element_dielectric", eps_r=self.element_epsr_eff(),
            tan_delta=self.element_tan_delta))
        return db


# --------------------------------------------------------------------------- #
# Operating point
# --------------------------------------------------------------------------- #


@dataclass
class OperatingParams:
    """Excitation / operating point for the EQS solve."""

    frequency: float = 50.0              # Hz
    um_kv: float = 245.0                 # highest voltage for equipment (kV)
    # Applied phase-to-earth RMS voltage amplitude U0 = Um / sqrt(3).
    # Used as a real amplitude (the response is linear, so the absolute phase is
    # arbitrary; tan delta is a ratio and U0 cancels for C/tan-delta).
    temperature_c: float = 20.0          # ambient/operating temperature

    @property
    def omega(self) -> float:
        """Angular frequency w = 2 pi f."""
        return 2.0 * math.pi * self.frequency

    @property
    def u0(self) -> float:
        """Applied potential on the HV electrode (volts, RMS amplitude)."""
        return (self.um_kv * 1.0e3) / math.sqrt(3.0)


# --------------------------------------------------------------------------- #
# Materials (the numeric database lives in ctfem.materials; this records which
# named material fills each physical region, so defects can override entries.)
# --------------------------------------------------------------------------- #


@dataclass
class MaterialParams:
    """Assignment of named materials to physical regions, plus a temperature.

    The actual (eps_r, tan_delta, sigma) values are looked up in
    :mod:`ctfem.materials`.  Defects mutate a *copy* of the material database
    rather than this mapping.
    """

    region_material: dict[str, str] = field(
        default_factory=lambda: {
            "primary_conductor": "metal",
            "paper_insulation": "paper",
            "oil": "oil",
            "porcelain": "porcelain",
            "air": "air",
            "head_housing": "metal",
            "base_tank": "metal",
            # CVT stack terminals (the elements use the element_* prefix rule)
            "stack_top": "metal",
            "stack_bottom": "metal",
            "porcelain_shed": "porcelain",
            # foil_* regions handled separately (all "metal")
        }
    )
    foil_material: str = "metal"
    temperature_c: float = 20.0


# --------------------------------------------------------------------------- #
# Defects
# --------------------------------------------------------------------------- #

DefectKind = Literal[
    "none",
    "shorted_foil",
    "moisture_ingress",
    "global_aging",
    "oil_contamination",
    # CVT (capacitor-stack) defects:
    "shorted_element",
    "element_aging",
    "water_ingress",
]


@dataclass
class DefectSpec:
    """Parametrized insulation defect.

    A single dataclass with a `kind` discriminator keeps the sweep schema flat
    and serialisable.  Unused fields are ignored for a given kind.

    kinds:
      none               -- healthy baseline.
      shorted_foil       -- galvanically bridge foil `index` to foil index+1
                            (or to the conductor for index == 0).  Material-level
                            implementation: raise sigma of the paper annulus in
                            the gap to metal-like values over an axial window.
      moisture_ingress   -- local wet-paper patch centred at z=`z_center`,
                            axial half-`extent`, with `severity` in [0,1]
                            scaling eps_r 3.5->7 and tan delta up to ~0.1.
      global_aging       -- uniform paper tan delta rise + slight eps_r drift,
                            scaled by `severity`.
      oil_contamination  -- oil conductivity increase, scaled by `severity`.

    CVT kinds (capacitor stack of a DDB/DFK voltage transformer):
      shorted_element    -- galvanic breakdown of capacitor element `index`
                            (1-based, 1 = top/HV end).  THE classic CVT failure:
                            C jumps by N/(N-1) and the divider ratio shifts
                            (up for a short in C1, down for a short in C2).
      element_aging      -- element-dielectric tan delta rise, scaled by
                            `severity`; `index` = 0 ages ALL elements, k >= 1
                            ages only element k.
      water_ingress      -- free-water pocket in the OIL volume: a volumetric
                            material override (eps_r -> `defect_eps_r`, sigma ->
                            `defect_sigma`; defaults are liquid water, eps_r 80
                            and 1e-4 S/m) over the axial window z_center +/-
                            extent.  NOTE: like every defect here this is a
                            cell-level kappa change, NEVER a Dirichlet boundary
                            -- the pocket floats at whatever potential the
                            surrounding field imposes, and the equipotentials
                            distort around it.

    Azimuthal localization (3-D only)
    ---------------------------------
    `theta_center` / `theta_extent` (radians) make a defect azimuthally LOCAL,
    e.g. a moisture pocket on one side of the core -- something the axisymmetric
    2-D model physically cannot represent.  `theta_extent >= 2*pi` (the default)
    means a full ring, which is the rotationally-symmetric case that the 2-D
    solver reproduces.  The 2-D solver ignores these fields (no azimuth exists);
    the 3-D solver (`ctfem.solver3d`) applies them.
    """

    kind: DefectKind = "none"
    severity: float = 0.0                # in [0, 1] where meaningful
    index: int = 0                       # foil index for shorted_foil
    z_center: float = 1.45               # m, for moisture_ingress
    extent: float = 0.15                 # m half-height of moisture patch
    # axial window (m) over which a shorted-foil bridge is applied
    short_axial_window: float = 0.10
    # azimuthal localization (3-D); full ring by default -> backward compatible
    theta_center: float = 0.0            # rad
    theta_extent: float = 2.0 * math.pi  # rad (>= 2*pi means a full ring)
    # defect material (water_ingress): severity interpolates the host material
    # towards these endpoint values.  Defaults model free water at 50 Hz.
    defect_eps_r: float = 80.0
    defect_sigma: float = 1.0e-4         # S/m

    def __post_init__(self) -> None:
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(f"severity must be in [0,1], got {self.severity}")
        if self.theta_extent <= 0.0:
            raise ValueError(f"theta_extent must be > 0, got {self.theta_extent}")

    @property
    def is_full_ring(self) -> bool:
        """True if the defect is azimuthally symmetric (a full ring)."""
        return self.theta_extent >= 2.0 * math.pi - 1e-9

    def label(self) -> str:
        """Short human-readable label for plots/tables."""
        if self.kind == "none":
            return "healthy"
        if self.kind == "shorted_foil":
            return f"short_foil_{self.index}"
        if self.kind == "moisture_ingress":
            tag = f"moisture_z{self.z_center:.2f}_s{self.severity:.2f}"
            if not self.is_full_ring:
                tag += f"_th{self.theta_center:.2f}w{self.theta_extent:.2f}"
            return tag
        if self.kind == "global_aging":
            return f"aging_s{self.severity:.2f}"
        if self.kind == "oil_contamination":
            return f"oilcontam_s{self.severity:.2f}"
        if self.kind == "shorted_element":
            return f"short_elem_{self.index}"
        if self.kind == "element_aging":
            scope = "all" if self.index == 0 else f"elem{self.index}"
            return f"elem_aging_{scope}_s{self.severity:.2f}"
        if self.kind == "water_ingress":
            tag = f"water_z{self.z_center:.2f}_s{self.severity:.2f}"
            if not self.is_full_ring:
                tag += f"_th{self.theta_center:.2f}w{self.theta_extent:.2f}"
            return tag
        return self.kind


# --------------------------------------------------------------------------- #
# Top-level case bundle
# --------------------------------------------------------------------------- #


@dataclass
class CaseConfig:
    """Everything needed to define one simulation case."""

    geometry: GeometryParams = field(default_factory=GeometryParams)
    operating: OperatingParams = field(default_factory=OperatingParams)
    materials: MaterialParams = field(default_factory=MaterialParams)
    defect: DefectSpec = field(default_factory=DefectSpec)
    name: str = "case"

    def to_dict(self) -> dict:
        """Flat-ish nested dict for provenance/serialisation."""
        d = asdict(self)
        # keep temperature consistent between operating and materials
        return d


# --------------------------------------------------------------------------- #
# Mesh-size presets (so scripts share a vocabulary: "coarse"/"medium"/"fine")
# --------------------------------------------------------------------------- #

MESH_PRESETS: dict[str, float] = {
    "coarse": 0.5,   # for the < 2 min laptop Phase-3 requirement
    "medium": 1.0,
    "fine": 1.8,
    "ultrafine": 2.6,
}


def apply_mesh_preset(geom: GeometryParams, preset: str) -> GeometryParams:
    """Return a copy of `geom` with mesh_refinement set from a named preset."""
    if preset not in MESH_PRESETS:
        raise ValueError(f"unknown mesh preset {preset!r}; choose from {list(MESH_PRESETS)}")
    from dataclasses import replace
    return replace(geom, mesh_refinement=MESH_PRESETS[preset])

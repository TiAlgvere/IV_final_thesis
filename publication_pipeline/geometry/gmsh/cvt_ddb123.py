"""Self-contained representative CVT geometry — Arteche DDB-123 (Task 009).

This is the publication-pipeline port of the legacy ctfem prototype's validated
axisymmetric CVT (ctfem/config.py::CVTParams + ctfem/geometry.py::build_cvt). It
imports NOTHING from ctfem so the publication pipeline stays independently
reproducible, but it deliberately keeps the same physical-group NAMES so the same
.msh can be cross-checked through the ctfem scikit-fem backend.

Coordinate convention: x = r (radius >= 0), y = z (height, z=0 at base). 2-D
axisymmetric (r,z) half-section; the symmetry axis r=0 is a natural boundary and
is NOT tagged. Lengths in metres, MSH 4.1.

Datasheet anchors (ARTECHE DDB/DFK, model DDB-123, the 110 kV-side Estonian unit):
    highest voltage Um     123 kV
    standard capacitance   5600 pF   (validation target)
    total height H         1830 mm
    base width A           450 mm
    creepage               3075 mm

Architecture (top->bottom): HV head (oil-compensator dome + terminal stub), a
hollow porcelain insulator with weather sheds containing the series capacitor
stack immersed in oil, and the grounded EMU tank. The stack of `n_elements`
homogenized capacitor elements forms the C1/C2 divider; the intermediate-voltage
tap is the interior disc between the C1 and C2 sections.

Physical groups produced
------------------------
dim=2 bodies: element_1..N, foil_1..N-1 (interior divider discs), stack_top,
              stack_bottom, head_housing, base_tank, porcelain, porcelain_shed,
              oil, air
dim=1 curves: hv_electrode, ground_electrode, insulator_surface, farfield
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional, Sequence

import gmsh

EPS0 = 8.8541878128e-12


# --------------------------------------------------------------------------- #
# Parameters (faithful port of ctfem.config.CVTParams geometry + electrical)
# --------------------------------------------------------------------------- #
@dataclass
class CVTParams:
    """Parametric axisymmetric Arteche DDB-123 CVT (see module docstring)."""

    # nameplate / envelope (m)
    total_height: float = 1.830
    tank_radius: float = 0.225
    tank_height: float = 0.550
    head_radius: float = 0.180
    head_height: float = 0.180
    terminal_radius: float = 0.020
    terminal_height: float = 0.040

    # porcelain insulator + weather sheds
    porcelain_inner_radius: float = 0.095
    porcelain_thickness: float = 0.025
    n_sheds: int = 27
    shed_overhang: float = 0.040
    shed_thickness_root: float = 0.018
    shed_thickness_tip: float = 0.008
    shed_droop: float = 0.010
    rated_creepage_mm: float = 3075.0

    # series capacitor stack
    n_elements: int = 12
    n_elements_c2: int = 2
    stack_radius: float = 0.075
    disc_thickness: float = 0.006
    rated_capacitance_pF: float = 5600.0
    element_tan_delta: float = 0.002

    # Task 010 surface pollution: a thin conductive skin on the OUTER edge of the
    # porcelain wall (the verified Tasks 006-008 thin-conductive-volume-layer model
    # applied to the realistic creepage surface). Present only when
    # build_cvt_ddb123(with_pollution_layer=True); eps_r = porcelain so sigma=0 is
    # identical to clean. It forms a continuous axial leakage path along the wall.
    pollution_thickness: float = 0.003
    lc_pollution: float = 0.0015

    # Task 013 geometry refinement: corner-rounding radii [m] (fillet arcs) to
    # remove field singularities at sharp edges. Applied when build(rounded=True).
    electrode_round: float = 0.0025   # capacitor-disc / stack-terminal outer corners
    head_round: float = 0.020         # HV compensator dome corners
    terminal_round: float = 0.008     # primary terminal stub corners
    shed_lip_round: float = 0.003     # weather-shed tip lips
    mesh_curvature: float = 14.0      # elements per 2*pi of curvature at the fillets

    # far field
    farfield_factor: float = 3.0
    farfield_radius: Optional[float] = None

    # meshing (lc_foil = discs, lc_paper = elements/solids)
    mesh_refinement: float = 1.0
    lc_foil: float = 0.004
    lc_paper: float = 0.015
    lc_shed: float = 0.008
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
            raise ValueError("stack does not fit: reduce n_elements or disc_thickness")

    # -- derived stack layout (z, element 1 at the HV/top end) -------------- #
    @property
    def stack_z_lo(self) -> float:
        return self.tank_height

    @property
    def stack_z_hi(self) -> float:
        return self.total_height - self.head_height - self.terminal_height

    def element_height(self) -> float:
        interior = (self.stack_z_hi - self.disc_thickness) - (self.stack_z_lo + self.disc_thickness)
        return (interior - (self.n_elements - 1) * self.disc_thickness) / self.n_elements

    def element_span(self, k: int) -> tuple[float, float]:
        if not (1 <= k <= self.n_elements):
            raise ValueError(f"element index {k} out of [1, {self.n_elements}]")
        h = self.element_height()
        z_top = (self.stack_z_hi - self.disc_thickness) - (k - 1) * (h + self.disc_thickness)
        return z_top - h, z_top

    def disc_span(self, k: int) -> tuple[float, float]:
        if not (1 <= k <= self.n_elements - 1):
            raise ValueError(f"disc index {k} out of [1, {self.n_elements - 1}]")
        z_bot_el, _ = self.element_span(k)
        return z_bot_el - self.disc_thickness, z_bot_el

    # -- shed profile ------------------------------------------------------- #
    def shed_centers(self) -> list[float]:
        h_ins = self.stack_z_hi - self.tank_height
        pitch = h_ins / self.n_sheds
        return [self.tank_height + (k - 0.5) * pitch for k in range(1, self.n_sheds + 1)]

    def shed_polygon(self, zc: float) -> list[tuple[float, float]]:
        r_root = self.porcelain_inner_radius + 0.5 * self.porcelain_thickness
        r_tip = self.porcelain_inner_radius + self.porcelain_thickness + self.shed_overhang
        tr, tt, d = self.shed_thickness_root, self.shed_thickness_tip, self.shed_droop
        return [
            (r_root, zc - 0.5 * tr),
            (r_tip, zc - d - 0.5 * tt),
            (r_tip, zc - d + 0.5 * tt),
            (r_root, zc + 0.5 * tr),
        ]

    def creepage_distance(self) -> float:
        h_ins = self.stack_z_hi - self.tank_height
        r_wall = self.porcelain_inner_radius + self.porcelain_thickness
        A, B, C, D = self.shed_polygon(0.0)

        def _wall_exit(p_in, p_out):
            t = (r_wall - p_in[0]) / (p_out[0] - p_in[0])
            return (r_wall, p_in[1] + t * (p_out[1] - p_in[1]))

        a = _wall_exit(A, B)
        d = _wall_exit(D, C)
        per_shed = (
            math.hypot(B[0] - a[0], B[1] - a[1])
            + math.hypot(C[0] - B[0], C[1] - B[1])
            + math.hypot(d[0] - C[0], d[1] - C[1])
        )
        covered = d[1] - a[1]
        return (h_ins - self.n_sheds * covered) + self.n_sheds * per_shed

    # -- derived electrical ------------------------------------------------- #
    def element_epsr_eff(self) -> float:
        """Effective eps_r so each homogenized element carries its rated series C."""
        c_elem = self.n_elements * self.rated_capacitance_pF * 1e-12
        area = math.pi * self.stack_radius**2
        return c_elem * self.element_height() / (EPS0 * area)

    @property
    def tap_disc_index(self) -> int:
        """Interior-disc index of the intermediate voltage tap (C1/C2 joint)."""
        return self.n_elements - self.n_elements_c2

    def divider_ratio_nominal(self) -> float:
        """Nominal tap voltage fraction V_tap/U0 = C1/(C1+C2) = N2/N."""
        return self.n_elements_c2 / self.n_elements


# --------------------------------------------------------------------------- #
# classification polygons
# --------------------------------------------------------------------------- #
@dataclass
class _Region:
    name: str
    polygon: list[tuple[float, float]]
    priority: int
    is_hv_metal: bool = False
    is_ground_metal: bool = False
    radii: list[float] | None = None  # per-vertex fillet radius (None => sharp)


def _point_in_polygon(r: float, z: float, poly: Sequence[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        ri, zi = poly[i]
        rj, zj = poly[j]
        if ((zi > z) != (zj > z)) and (r < (rj - ri) * (z - zi) / (zj - zi + 1e-300) + ri):
            inside = not inside
        j = i
    return inside


def _cvt_regions(c: CVTParams, with_pollution: bool = False, rounded: bool = True) -> list[_Region]:
    regions: list[_Region] = []
    prio = 0
    rs = c.stack_radius
    # fillet radii (0 => sharp); polygons below are [axis-bottom, outer-bottom,
    # outer-top, axis-top], so radii [0, R, R, 0] round the two OUTER corners.
    er = c.electrode_round if rounded else 0.0
    hr = c.head_round if rounded else 0.0
    tr = c.terminal_round if rounded else 0.0
    lip = c.shed_lip_round if rounded else 0.0

    # Pollution skin: highest precedence (priority -1) so the thin band at the
    # wall outer edge is a continuous conductive shell over the porcelain height.
    if with_pollution:
        r_out = c.porcelain_inner_radius + c.porcelain_thickness
        r0 = r_out - c.pollution_thickness
        regions.append(_Region(
            "pollution_layer",
            [(r0, c.tank_height), (r_out, c.tank_height), (r_out, c.stack_z_hi), (r0, c.stack_z_hi)],
            -1))

    # interior electrode discs (floating divider ladder) - round the outer edges
    for k in range(1, c.n_elements):
        z0, z1 = c.disc_span(k)
        regions.append(_Region(f"foil_{k}", [(0.0, z0), (rs, z0), (rs, z1), (0.0, z1)], prio,
                               radii=[0.0, er, er, 0.0]))
        prio += 1

    # stack terminal discs (HV top, grounded bottom)
    regions.append(_Region(
        "stack_top",
        [(0.0, c.stack_z_hi - c.disc_thickness), (rs, c.stack_z_hi - c.disc_thickness),
         (rs, c.stack_z_hi), (0.0, c.stack_z_hi)],
        prio, is_hv_metal=True, radii=[0.0, er, er, 0.0]))
    prio += 1
    regions.append(_Region(
        "stack_bottom",
        [(0.0, c.stack_z_lo), (rs, c.stack_z_lo),
         (rs, c.stack_z_lo + c.disc_thickness), (0.0, c.stack_z_lo + c.disc_thickness)],
        prio, is_ground_metal=True, radii=[0.0, er, er, 0.0]))
    prio += 1

    # homogenized capacitor elements
    for k in range(1, c.n_elements + 1):
        z0, z1 = c.element_span(k)
        regions.append(_Region(f"element_{k}", [(0.0, z0), (rs, z0), (rs, z1), (0.0, z1)], prio))
        prio += 1

    # HV head: compensator dome + primary terminal stub
    z_dome1 = c.stack_z_hi + c.head_height
    regions.append(_Region(
        "head_housing",
        [(0.0, c.stack_z_hi), (c.head_radius, c.stack_z_hi),
         (c.head_radius, z_dome1), (0.0, z_dome1)],
        prio, is_hv_metal=True, radii=[0.0, hr, hr, 0.0]))
    prio += 1
    regions.append(_Region(
        "head_housing",
        [(0.0, z_dome1), (c.terminal_radius, z_dome1),
         (c.terminal_radius, c.total_height), (0.0, c.total_height)],
        prio, is_hv_metal=True, radii=[0.0, tr, tr, 0.0]))
    prio += 1

    # grounded EMU tank (round the air-facing top-outer corner)
    regions.append(_Region(
        "base_tank",
        [(0.0, 0.0), (c.tank_radius, 0.0), (c.tank_radius, c.tank_height), (0.0, c.tank_height)],
        prio, is_ground_metal=True, radii=[0.0, 0.0, hr, 0.0]))
    prio += 1

    # porcelain insulator: weather sheds (finer mesh) then the wall
    for zc in c.shed_centers():
        # shed_polygon = [root_bottom, tip_bottom, tip_top, root_top]; round the two
        # tip lips (vertices 1,2) so the field at the shed edge is not singular.
        regions.append(_Region("porcelain_shed", c.shed_polygon(zc), prio, radii=[0.0, lip, lip, 0.0]))
        prio += 1
    r_in = c.porcelain_inner_radius
    r_out = r_in + c.porcelain_thickness
    regions.append(_Region(
        "porcelain",
        [(r_in, c.tank_height), (r_out, c.tank_height), (r_out, c.stack_z_hi), (r_in, c.stack_z_hi)],
        prio))
    prio += 1

    # oil filling the insulator interior around the stack
    regions.append(_Region(
        "oil",
        [(0.0, c.tank_height), (r_in, c.tank_height), (r_in, c.stack_z_hi), (0.0, c.stack_z_hi)],
        prio))
    prio += 1

    # air box (lowest priority)
    Rf = float(c.farfield_radius)
    zc = 0.5 * c.total_height
    regions.append(_Region("air", [(0.0, zc - Rf), (Rf, zc - Rf), (Rf, zc + Rf), (0.0, zc + Rf)], prio))
    return regions


def _classify(cx: float, cz: float, regions: list[_Region]) -> Optional[_Region]:
    for reg in sorted(regions, key=lambda r: r.priority):
        if _point_in_polygon(cx, cz, reg.polygon):
            return reg
    return None


def _add_polygon_surface(poly: Sequence[tuple[float, float]]) -> int:
    pts = [gmsh.model.occ.addPoint(r, z, 0.0) for (r, z) in poly]
    lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    loop = gmsh.model.occ.addCurveLoop(lines)
    return gmsh.model.occ.addPlaneSurface([loop])


def _rounded_polygon_surface(poly: Sequence[tuple[float, float]], radii: Sequence[float]) -> int:
    """Planar surface from a polygon with circular fillets of the given per-vertex
    radii (0 = sharp). Tangent lengths are clamped to the adjacent edges, so an
    over-large radius is reduced rather than failing."""
    import math

    occ = gmsh.model.occ
    n = len(poly)
    V = [_np_xy(p) for p in poly]
    seg_start = [0] * n
    seg_end = [0] * n
    arc = [None] * n
    for i in range(n):
        P, A, B, r = V[i], V[(i - 1) % n], V[(i + 1) % n], radii[i]
        if r <= 0.0:
            p = occ.addPoint(P[0], P[1], 0.0)
            seg_start[i] = seg_end[i] = p
            continue
        u = _unit(A, P)
        w = _unit(B, P)
        ang = math.acos(max(-1.0, min(1.0, u[0] * w[0] + u[1] * w[1])))
        t = r / math.tan(ang / 2.0)
        t = min(t, 0.49 * _dist(A, P), 0.49 * _dist(B, P))
        r_eff = t * math.tan(ang / 2.0)
        d = r_eff / math.sin(ang / 2.0)
        Tin = (P[0] + t * u[0], P[1] + t * u[1])
        Tout = (P[0] + t * w[0], P[1] + t * w[1])
        bx, by = u[0] + w[0], u[1] + w[1]
        bn = math.hypot(bx, by)
        C = (P[0] + d * bx / bn, P[1] + d * by / bn)
        seg_end[i] = occ.addPoint(Tin[0], Tin[1], 0.0)
        seg_start[i] = occ.addPoint(Tout[0], Tout[1], 0.0)
        arc[i] = (seg_end[i], occ.addPoint(C[0], C[1], 0.0), seg_start[i])
    curves = []
    for i in range(n):
        curves.append(occ.addLine(seg_start[i], seg_end[(i + 1) % n]))
        a = arc[(i + 1) % n]
        if a is not None:
            curves.append(occ.addCircleArc(a[0], a[1], a[2]))
    return occ.addPlaneSurface([occ.addCurveLoop(curves)])


def _np_xy(p):
    return (float(p[0]), float(p[1]))


def _unit(a, b):
    import math
    dx, dy = a[0] - b[0], a[1] - b[1]
    L = math.hypot(dx, dy)
    return (dx / L, dy / L)


def _dist(a, b):
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _region_lc(name: str, c: CVTParams) -> float:
    if name == "pollution_layer":
        return c.lc_pollution
    if name.startswith("foil_") or name in ("stack_top", "stack_bottom"):
        return c.lc_foil
    if name == "porcelain_shed":
        return c.lc_shed
    if name.startswith("element_") or name in ("porcelain", "head_housing", "base_tank"):
        return c.lc_paper
    if name == "oil":
        return c.lc_oil
    return c.lc_air


def _set_mesh_fields(c: CVTParams, region_surfaces: dict[str, list[int]], disc_edge_curves: list[int],
                     rounded: bool = False) -> None:
    ref = max(c.mesh_refinement, 1e-6)
    lc_fine = c.lc_foil / ref
    lc_air = c.lc_air / ref
    # Controlled refinement at the fillet radii: curvature-driven sizing resolves
    # each rounded corner, with the floor set to a fraction of the smallest radius.
    if rounded and c.mesh_curvature > 0:
        smallest_r = min(c.electrode_round, c.terminal_round, c.shed_lip_round)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(lc_fine, 0.5 * smallest_r))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", c.mesh_curvature)
    else:
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc_fine)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_air)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)

    # coarse regions first so the finer size wins on shared points
    ordered = sorted(region_surfaces.items(), key=lambda kv: -_region_lc(kv[0], c))
    for name, tags in ordered:
        lc = _region_lc(name, c) / ref
        pts: set[int] = set()
        for t in tags:
            for (bdim, btag) in gmsh.model.getBoundary([(2, t)], recursive=True, oriented=False):
                if bdim == 0:
                    pts.add(abs(btag))
        if pts:
            gmsh.model.mesh.setSize([(0, p) for p in pts], lc)

    if not disc_edge_curves:
        return
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "CurvesList", disc_edge_curves)
    gmsh.model.mesh.field.setNumber(dist, "Sampling", 100)
    thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr, "InField", dist)
    gmsh.model.mesh.field.setNumber(thr, "SizeMin", lc_fine)
    gmsh.model.mesh.field.setNumber(thr, "SizeMax", lc_air)
    gmsh.model.mesh.field.setNumber(thr, "DistMin", c.refine_dist_min)
    gmsh.model.mesh.field.setNumber(thr, "DistMax", c.refine_dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)


# --------------------------------------------------------------------------- #
# build + tag
# --------------------------------------------------------------------------- #
def build_cvt_ddb123(
    out_msh: Path,
    params: CVTParams | None = None,
    *,
    with_pollution_layer: bool = False,
    rounded: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build + mesh the representative DDB-123 CVT; write MSH 4.1 to out_msh.

    ``rounded`` (default True, Task 013) applies fillet arcs at the capacitor-disc /
    stack-terminal / head / terminal / shed-lip corners and curvature-driven mesh
    refinement there, removing the sharp-corner field singularities. Pass
    rounded=False for the original sharp geometry (Tasks 009-012).

    ``with_pollution_layer`` (legacy) adds the straight volume skin; the preferred
    pollution model is now the Surface Conductance BC on insulator_surface, which
    needs no geometry change.
    """
    c = params or CVTParams()
    out_msh = Path(out_msh)
    out_msh.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.add("cvt_ddb123")

        regions = _cvt_regions(c, with_pollution=with_pollution_layer, rounded=rounded)

        # 1. add region surfaces (rounded where radii given), fragment to a partition
        src_tags = [
            _rounded_polygon_surface(reg.polygon, reg.radii) if reg.radii and any(reg.radii)
            else _add_polygon_surface(reg.polygon)
            for reg in regions
        ]
        gmsh.model.occ.synchronize()
        gmsh.model.occ.fragment([(2, t) for t in src_tags], [])
        gmsh.model.occ.synchronize()

        # 2. classify each surface by centroid
        region_surfaces: dict[str, list[int]] = {}
        surf_name_of_tag: dict[int, str] = {}
        for (dim, tag) in gmsh.model.getEntities(2):
            com = gmsh.model.occ.getCenterOfMass(2, tag)
            reg = _classify(com[0], com[1], regions)
            name = reg.name if reg is not None else "air"
            region_surfaces.setdefault(name, []).append(tag)
            surf_name_of_tag[tag] = name

        # 3. physical body groups
        for name, tags in region_surfaces.items():
            pg = gmsh.model.addPhysicalGroup(2, sorted(set(tags)))
            gmsh.model.setPhysicalName(2, pg, name)

        # 4. curve adjacency -> electrode / insulator / farfield tagging
        hv_names = {r.name for r in regions if r.is_hv_metal}
        gnd_names = {r.name for r in regions if r.is_ground_metal}
        foil_names = {r.name for r in regions if r.name.startswith("foil_")}
        insulator_names = {"porcelain", "porcelain_shed"}

        curve_to_surfaces: dict[int, list[str]] = {}
        for (dim, tag) in gmsh.model.getEntities(2):
            for (bdim, btag) in gmsh.model.getBoundary([(2, tag)], oriented=False):
                curve_to_surfaces.setdefault(abs(btag), []).append(surf_name_of_tag[tag])

        hv_c, gnd_c, ff_c, disc_c, ins_c = [], [], [], [], []
        Rf = float(c.farfield_radius)
        for ctag, neigh in curve_to_surfaces.items():
            com = gmsh.model.occ.getCenterOfMass(1, ctag)
            on_axis = com[0] < 1e-7
            ns = set(neigh)
            if not on_axis and (ns & hv_names) and (ns - hv_names):
                hv_c.append(ctag)
            if not on_axis and (ns & gnd_names) and (ns - gnd_names):
                gnd_c.append(ctag)
            if ns & foil_names:
                disc_c.append(ctag)
            if not on_axis and (ns & insulator_names) and "air" in ns:
                ins_c.append(ctag)
            if len(neigh) == 1 and neigh[0] == "air" and not on_axis:
                ff_c.append(ctag)

        def _curve_group(name: str, tags: list[int]) -> None:
            if tags:
                pg = gmsh.model.addPhysicalGroup(1, sorted(set(tags)))
                gmsh.model.setPhysicalName(1, pg, name)

        _curve_group("hv_electrode", hv_c)
        _curve_group("ground_electrode", gnd_c)
        _curve_group("insulator_surface", ins_c)
        _curve_group("farfield", ff_c)

        # 5. graded mesh, generate, write
        _set_mesh_fields(c, region_surfaces, disc_c, rounded=rounded)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(out_msh))

        physical_groups = {
            gmsh.model.getPhysicalName(dim, tag): {
                "dim": dim,
                "tag": tag,
                "n_entities": len(gmsh.model.getEntitiesForPhysicalGroup(dim, tag)),
            }
            for dim, tag in gmsh.model.getPhysicalGroups()
        }
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        _, tri_tags, _ = gmsh.model.mesh.getElements(2)
        metadata = {
            "geometry": "cvt_ddb123_axisymmetric",
            "output_msh": str(out_msh),
            "n_nodes": len(node_tags),
            "n_triangles": int(sum(len(t) for t in tri_tags)),
            "element_epsr_eff": c.element_epsr_eff(),
            "tap_disc_index": c.tap_disc_index,
            "divider_ratio_nominal": c.divider_ratio_nominal(),
            "creepage_distance_mm": c.creepage_distance() * 1000.0,
            "parameters": asdict(c),
            "physical_groups": physical_groups,
        }
        return metadata
    finally:
        gmsh.finalize()

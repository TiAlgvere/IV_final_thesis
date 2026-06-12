"""Parametric axisymmetric geometry + meshing with the gmsh Python API.

We build the (r, z) half-section as a set of overlapping polygons, fragment
them into a conformal partition with the OpenCASCADE kernel, then classify each
resulting surface into a named region by testing its centroid against the
*known* region polygons in a fixed priority order (most-specific region first).
This is far more robust than chasing boolean-output tags and handles the foils
nested inside the paper band cleanly.

Two builders share the same code paths (spec requirement):
  * :func:`build_ct`     -- the full CT condenser geometry.
  * :func:`build_coax`   -- a degenerate coaxial cylinder capacitor used as the
                            analytic validation case (Phase 2).

Conventions
-----------
* x-coordinate = r (radius, >= 0), y-coordinate = z (height).  z = 0 at base.
* Lengths in metres.  MSH written in version 4.1.
* The symmetry axis r = 0 is *not* tagged: it is a natural (zero normal
  current) boundary in the EQS weak form and must simply be excluded from the
  Dirichlet sets.

Physical groups produced
------------------------
dim=2 surfaces: primary_conductor, paper_insulation, foil_1..foil_N, oil,
                porcelain, air, head_housing, base_tank   (CT)
                dielectric                                  (coax)
dim=1 curves:   hv_electrode, ground_electrode, farfield   (CT)
                hv_electrode, ground_electrode              (coax)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import gmsh

from .config import GeometryParams, CVTParams

# Physical-group integer tags are assigned automatically by gmsh; we always
# refer to groups by *name* downstream (dolfinx exposes name->tag).

# region classification priority: most specific first
_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Small polygon helpers (point-in-polygon, used only for region classification)
# --------------------------------------------------------------------------- #


def _point_in_polygon(r: float, z: float, poly: Sequence[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (polygon as list of (r,z) vertices)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        ri, zi = poly[i]
        rj, zj = poly[j]
        if ((zi > z) != (zj > z)) and (
            r < (rj - ri) * (z - zi) / (zj - zi + 1e-300) + ri
        ):
            inside = not inside
        j = i
    return inside


@dataclass
class _Region:
    """A named classification polygon with a priority (lower = tested first)."""

    name: str
    polygon: list[tuple[float, float]]
    priority: int
    is_hv_metal: bool = False
    is_ground_metal: bool = False


# --------------------------------------------------------------------------- #
# Geometry-build result
# --------------------------------------------------------------------------- #


@dataclass
class MeshResult:
    """Outcome of a geometry+mesh build (for stats/reporting)."""

    path: str
    surface_groups: dict[str, int]          # name -> element count
    curve_groups: dict[str, int]            # name -> element count
    n_nodes: int
    n_triangles: int
    # name -> (dim, physical integer tag); written to a JSON sidecar so the
    # DOLFINx solver can map region names to the integer tags gmshio exposes.
    physical_tags: dict[str, tuple[int, int]] = field(default_factory=dict)

    def summary(self) -> str:
        # `surface_groups`/`curve_groups`/`n_triangles` are named for the 2-D
        # case; in 3-D they carry volume groups / surface groups / tet counts.
        # The labels below are written dimension-neutrally.
        lines = [f"mesh: {self.path}",
                 f"  nodes={self.n_nodes}  top-dim elements={self.n_triangles}",
                 "  region (cell) groups:"]
        for k, v in self.surface_groups.items():
            lines.append(f"    {k:<18} {v} elem")
        lines.append("  boundary (facet) groups:")
        for k, v in self.curve_groups.items():
            lines.append(f"    {k:<18} {v} elem")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Full CT geometry
# --------------------------------------------------------------------------- #


def _ct_regions(g: GeometryParams) -> list[_Region]:
    """Build the classification polygons for the full CT in priority order.

    Polygons are allowed to overlap (e.g. foils inside paper, paper inside the
    oil envelope); the *priority* ordering resolves nesting at classification
    time.
    """
    regions: list[_Region] = []
    prio = 0

    # -- foils (innermost, highest priority) ------------------------------- #
    radii = g.foil_radii()
    half_t = 0.5 * g.foil_thickness
    for k, rmid in enumerate(radii):
        z0, z1 = g.foil_axial_span(k)
        ra, rb = rmid - half_t, rmid + half_t
        poly = [(ra, z0), (rb, z0), (rb, z1), (ra, z1)]
        regions.append(_Region(f"foil_{k + 1}", poly, prio))
        prio += 1

    # -- primary conductor (axial cylinder, HV) ---------------------------- #
    # Starts at the paper body bottom (above the grounded base, with an oil gap)
    # and runs up to the head, so HV never touches ground.
    rc = g.conductor_radius
    z_cond0 = g.paper_z_bottom
    z_head0 = g.total_height - g.head_height
    z_cond1 = z_head0
    regions.append(_Region(
        "primary_conductor",
        [(0.0, z_cond0), (rc, z_cond0), (rc, z_cond1), (0.0, z_cond1)],
        prio, is_hv_metal=True))
    prio += 1

    # -- head housing (top, HV) -------------------------------------------- #
    z_head1 = g.total_height
    regions.append(_Region(
        "head_housing",
        [(0.0, z_head0), (g.head_radius, z_head0),
         (g.head_radius, z_head1), (0.0, z_head1)],
        prio, is_hv_metal=True))
    prio += 1

    # -- base tank + grounded ground-sleeve (both grounded "base_tank") ----- #
    # bottom disk:
    regions.append(_Region(
        "base_tank",
        [(0.0, 0.0), (g.base_radius, 0.0),
         (g.base_radius, g.base_height), (0.0, g.base_height)],
        prio, is_ground_metal=True))
    prio += 1
    # vertical ground sleeve just outside the paper body (the C1 ground ref):
    rs0 = g.ground_sleeve_radius
    rs1 = g.ground_sleeve_radius + g.ground_sleeve_thickness
    regions.append(_Region(
        "base_tank",
        [(rs0, g.base_height), (rs1, g.base_height),
         (rs1, g.paper_z_top), (rs0, g.paper_z_top)],
        prio, is_ground_metal=True))
    prio += 1

    # -- paper insulation (conical band around conductor) ------------------ #
    pin = g.paper_inner_radius
    regions.append(_Region(
        "paper_insulation",
        [(pin, g.paper_z_bottom),
         (g.paper_outer_radius_base, g.paper_z_bottom),
         (g.paper_outer_radius_head, g.paper_z_top),
         (pin, g.paper_z_top)],
        prio))
    prio += 1

    # -- porcelain wall (hollow cone; smooth, no sheds in v1) --------------- #
    # inner radius oil_outer_radius, wall thickness porcelain_thickness, over
    # the porcelain section height (sits above base).
    pz0 = g.base_height
    pz1 = g.base_height + g.porcelain_height
    r_in = g.oil_outer_radius
    r_out = g.oil_outer_radius + g.porcelain_thickness
    regions.append(_Region(
        "porcelain",
        [(r_in, pz0), (r_out, pz0), (r_out, pz1), (r_in, pz1)],
        prio))
    prio += 1

    # -- oil envelope (everything inside porcelain inner wall + head volume) #
    # Drawn as the full interior column; solids above take priority so the
    # leftover interior fragments classify as oil.
    oz0 = g.base_height
    oz1 = g.total_height
    regions.append(_Region(
        "oil",
        [(0.0, oz0), (r_in, oz0), (r_in, oz1), (0.0, oz1)],
        prio))
    prio += 1

    # -- air box (everything else, lowest priority) ------------------------ #
    Rf = float(g.farfield_radius)
    zc = 0.5 * g.total_height
    z_lo = zc - Rf
    z_hi = zc + Rf
    regions.append(_Region(
        "air",
        [(0.0, z_lo), (Rf, z_lo), (Rf, z_hi), (0.0, z_hi)],
        prio))

    return regions


def _classify(cx: float, cz: float, regions: list[_Region]) -> Optional[_Region]:
    """Return the highest-priority region whose polygon contains (cx, cz)."""
    for reg in sorted(regions, key=lambda r: r.priority):
        if _point_in_polygon(cx, cz, reg.polygon):
            return reg
    return None


def _add_polygon_surface(poly: Sequence[tuple[float, float]]) -> int:
    """Add an OCC planar surface from an (r,z) polygon; return its surface tag."""
    pts = [gmsh.model.occ.addPoint(r, z, 0.0) for (r, z) in poly]
    lines = []
    n = len(pts)
    for i in range(n):
        lines.append(gmsh.model.occ.addLine(pts[i], pts[(i + 1) % n]))
    loop = gmsh.model.occ.addCurveLoop(lines)
    return gmsh.model.occ.addPlaneSurface([loop])


def _region_lc(name: str, g) -> float:
    """Characteristic length for a region BEFORE the refinement factor.

    `g` is GeometryParams or CVTParams (duck-typed: both carry lc_*).
    """
    if name.startswith("foil_") or name in ("stack_top", "stack_bottom"):
        return g.lc_foil
    if name == "porcelain_shed":
        # |E| peaks at the shed tips; CVTParams carries a dedicated size
        return g.lc_shed
    if name.startswith("element_") or name in (
            "paper_insulation", "porcelain",
            "primary_conductor", "head_housing", "base_tank"):
        return g.lc_paper
    if name == "oil":
        return g.lc_oil
    return g.lc_air  # air and anything else


def _set_mesh_fields(
    g: GeometryParams,
    region_surfaces: dict[str, list[int]],
    foil_curve_tags: list[int],
) -> None:
    """Graded sizing: point-based per-region lc + a thin Threshold band on foils.

    gmsh computes the final element size as the MINIMUM of all enabled sources
    (size-from-points and the background field), so we combine:

      * point-based sizes set on each region's boundary points (coarse in air,
        medium in paper/oil, fine on the foils), interpolated into the interior
        via MeshSizeExtendFromBoundary;
      * a Distance/Threshold background field (spec requirement) that forces a
        thin shell around the foil edges down to lc_foil, then relaxes to lc_air
        beyond DistMax so it does NOT over-refine the paper bulk.

    Global refinement factor scales all lengths (effective lc = lc / refinement).
    """
    ref = max(g.mesh_refinement, 1e-6)
    lc_foil = g.lc_foil / ref
    lc_air = g.lc_air / ref

    gmsh.option.setNumber("Mesh.MeshSizeMin", lc_foil)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_air)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # point-based per-region sizing.  Shared boundary points get the LAST
    # assigned size, so process coarse regions first and fine regions last,
    # ensuring the finer size wins on any shared point.
    ordered = sorted(region_surfaces.items(),
                     key=lambda kv: -_region_lc(kv[0], g))
    for name, tags in ordered:
        lc = _region_lc(name, g) / ref
        pts: set[int] = set()
        for t in tags:
            bnd = gmsh.model.getBoundary([(2, t)], recursive=True, oriented=False)
            for (bdim, btag) in bnd:
                if bdim == 0:
                    pts.add(abs(btag))
        if pts:
            gmsh.model.mesh.setSize([(0, p) for p in pts], lc)

    if not foil_curve_tags:
        return

    # Distance/Threshold background field for the fine foil shell
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "CurvesList", foil_curve_tags)
    gmsh.model.mesh.field.setNumber(dist, "Sampling", 100)

    thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr, "InField", dist)
    gmsh.model.mesh.field.setNumber(thr, "SizeMin", lc_foil)
    gmsh.model.mesh.field.setNumber(thr, "SizeMax", lc_air)
    gmsh.model.mesh.field.setNumber(thr, "DistMin", g.refine_dist_min)
    gmsh.model.mesh.field.setNumber(thr, "DistMax", g.refine_dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)


def _build_and_tag(
    regions: list[_Region],
    g_for_fields: Optional[GeometryParams],
    out_path: str,
    *,
    tag_farfield: bool,
    verbose: bool,
    coax_radii: Optional[tuple[float, float]] = None,
) -> MeshResult:
    """Shared core: add polygons, fragment, classify, tag, mesh, verify, write."""
    # 1. add all region surfaces
    src_tags: list[int] = [_add_polygon_surface(reg.polygon) for reg in regions]
    gmsh.model.occ.synchronize()

    # 2. fragment everything into a conformal non-overlapping partition
    dimtags = [(2, t) for t in src_tags]
    out, _ = gmsh.model.occ.fragment(dimtags, [])
    gmsh.model.occ.synchronize()

    # 3. classify each resulting surface by centroid
    surfaces = gmsh.model.getEntities(2)
    region_surfaces: dict[str, list[int]] = {}
    for (dim, tag) in surfaces:
        com = gmsh.model.occ.getCenterOfMass(2, tag)
        reg = _classify(com[0], com[1], regions)
        if reg is None:
            # tiny sliver outside everything -> attach to air if present
            name = "air" if any(r.name == "air" for r in regions) else regions[-1].name
        else:
            name = reg.name
        region_surfaces.setdefault(name, []).append(tag)

    # 4. physical surface groups
    for name, tags in region_surfaces.items():
        pg = gmsh.model.addPhysicalGroup(2, tags)
        gmsh.model.setPhysicalName(2, pg, name)

    # 5. curve adjacency map for electrode/farfield tagging
    curve_to_surfaces: dict[int, list[str]] = {}
    surf_name_of_tag: dict[int, str] = {}
    for name, tags in region_surfaces.items():
        for t in tags:
            surf_name_of_tag[t] = name
    for (dim, tag) in surfaces:
        bnd = gmsh.model.getBoundary([(2, tag)], oriented=False)
        for (bdim, btag) in bnd:
            curve_to_surfaces.setdefault(abs(btag), []).append(surf_name_of_tag[tag])

    hv_names = {r.name for r in regions if r.is_hv_metal}
    gnd_names = {r.name for r in regions if r.is_ground_metal}

    hv_curves: list[int] = []
    gnd_curves: list[int] = []
    farfield_curves: list[int] = []
    foil_edge_curves: list[int] = []
    foil_names = {r.name for r in regions if r.name.startswith("foil_")}

    Rf = float(g_for_fields.farfield_radius) if (g_for_fields and tag_farfield) else None

    for ctag, neigh in curve_to_surfaces.items():
        com = gmsh.model.occ.getCenterOfMass(1, ctag)
        rmid = com[0]
        on_axis = rmid < 1e-7
        neigh_set = set(neigh)
        if coax_radii is not None:
            # coax: tag by radius of the (vertical) curve directly
            r_in, r_out = coax_radii
            tol = 1e-6 * (r_out - r_in)
            if abs(rmid - r_in) < tol:
                hv_curves.append(ctag)
            elif abs(rmid - r_out) < tol:
                gnd_curves.append(ctag)
        else:
            # electrode interface: between an HV/ground metal and a dielectric
            if not on_axis and (neigh_set & hv_names) and (neigh_set - hv_names):
                hv_curves.append(ctag)
            if not on_axis and (neigh_set & gnd_names) and (neigh_set - gnd_names):
                gnd_curves.append(ctag)
        # foil edges (for post-processing + as Distance-field sources)
        if neigh_set & foil_names:
            foil_edge_curves.append(ctag)
        # far field: a boundary curve with exactly one neighbour (air) on the
        # outer envelope (r = Rf or z extremes)
        if tag_farfield and len(neigh) == 1 and neigh[0] == "air":
            if not on_axis:
                farfield_curves.append(ctag)

    def _add_curve_group(name: str, tags: list[int]) -> None:
        if tags:
            pg = gmsh.model.addPhysicalGroup(1, sorted(set(tags)))
            gmsh.model.setPhysicalName(1, pg, name)

    _add_curve_group("hv_electrode", hv_curves)
    _add_curve_group("ground_electrode", gnd_curves)
    if tag_farfield:
        _add_curve_group("farfield", farfield_curves)
    _add_curve_group("foil_edges", foil_edge_curves)

    # 6. mesh size fields
    if g_for_fields is not None:
        _set_mesh_fields(g_for_fields, region_surfaces,
                         foil_edge_curves if foil_names else [])

    # 7. mesh
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.model.mesh.generate(2)

    # 8. verify every physical group is non-empty, gather stats + tag map
    surf_counts: dict[str, int] = {}
    curve_counts: dict[str, int] = {}
    physical_tags: dict[str, tuple[int, int]] = {}
    for (dim, pg) in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, pg)
        physical_tags[name] = (dim, pg)
        ents = gmsh.model.getEntitiesForPhysicalGroup(dim, pg)
        n_elem = 0
        for e in ents:
            etypes, etags, _ = gmsh.model.mesh.getElements(dim, e)
            n_elem += sum(len(t) for t in etags)
        if dim == 2:
            surf_counts[name] = n_elem
        elif dim == 1:
            curve_counts[name] = n_elem
        if n_elem == 0:
            raise RuntimeError(
                f"physical group {name!r} (dim {dim}) is EMPTY after meshing"
            )

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)
    tri_types, tri_tags, _ = gmsh.model.mesh.getElements(2)
    n_tri = sum(len(t) for t in tri_tags)

    gmsh.write(out_path)
    # JSON sidecar: name -> [dim, tag] so the solver can map names <-> tags
    sidecar = os.path.splitext(out_path)[0] + ".tags.json"
    with open(sidecar, "w") as fh:
        json.dump({k: list(v) for k, v in physical_tags.items()}, fh, indent=2)
    if verbose:
        print(f"[geometry] wrote {out_path}: {n_nodes} nodes, {n_tri} triangles")

    return MeshResult(out_path, surf_counts, curve_counts, n_nodes, n_tri,
                      physical_tags)


def load_tag_map(msh_path: str) -> dict[str, tuple[int, int]]:
    """Load the {name -> (dim, tag)} sidecar written next to a .msh file."""
    sidecar = os.path.splitext(msh_path)[0] + ".tags.json"
    with open(sidecar) as fh:
        raw = json.load(fh)
    return {k: (int(v[0]), int(v[1])) for k, v in raw.items()}


def build_ct(g: GeometryParams, out_path: str, *, verbose: bool = True,
             gui: bool = False) -> MeshResult:
    """Build and mesh the full parametric CT geometry; write MSH 4.1 to out_path."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("ct")
        regions = _ct_regions(g)
        result = _build_and_tag(regions, g, out_path, tag_farfield=True,
                                verbose=verbose)
        if gui:  # pragma: no cover - interactive only
            gmsh.fltk.run()
        return result
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------- #
# CVT geometry (Arteche DDB/DFK capacitive voltage transformer)
# --------------------------------------------------------------------------- #


def _cvt_regions(c: CVTParams) -> list[_Region]:
    """Classification polygons for the DDB-style CVT in priority order.

    Stack architecture (top-down): `stack_top` electrode disc bolted to the HV
    head, then alternating homogenized capacitor elements (`element_k`) and
    floating interior electrode discs (`foil_k` -- the name reuses the CT foil
    machinery so the solver's potential-ladder observable becomes the divider
    ladder for free), then `stack_bottom` bolted to the grounded EMU tank.
    The intermediate voltage tap is interior disc `c.tap_disc_index`.
    """
    regions: list[_Region] = []
    prio = 0
    rs = c.stack_radius

    # -- interior electrode discs (floating; the divider ladder) ------------ #
    for k in range(1, c.n_elements):
        z0, z1 = c.disc_span(k)
        regions.append(_Region(
            f"foil_{k}", [(0.0, z0), (rs, z0), (rs, z1), (0.0, z1)], prio))
        prio += 1

    # -- stack terminal discs (HV top, grounded bottom) ---------------------- #
    regions.append(_Region(
        "stack_top",
        [(0.0, c.stack_z_hi - c.disc_thickness), (rs, c.stack_z_hi - c.disc_thickness),
         (rs, c.stack_z_hi), (0.0, c.stack_z_hi)],
        prio, is_hv_metal=True))
    prio += 1
    regions.append(_Region(
        "stack_bottom",
        [(0.0, c.stack_z_lo), (rs, c.stack_z_lo),
         (rs, c.stack_z_lo + c.disc_thickness), (0.0, c.stack_z_lo + c.disc_thickness)],
        prio, is_ground_metal=True))
    prio += 1

    # -- homogenized capacitor elements -------------------------------------- #
    for k in range(1, c.n_elements + 1):
        z0, z1 = c.element_span(k)
        regions.append(_Region(
            f"element_{k}", [(0.0, z0), (rs, z0), (rs, z1), (0.0, z1)], prio))
        prio += 1

    # -- HV head: wide oil-compensator dome + primary terminal stub ----------- #
    # The dome's diameter (> the shed reach) is what shapes the field pushed
    # down over the insulator, corona-ring fashion.
    z_dome1 = c.stack_z_hi + c.head_height
    regions.append(_Region(
        "head_housing",
        [(0.0, c.stack_z_hi), (c.head_radius, c.stack_z_hi),
         (c.head_radius, z_dome1), (0.0, z_dome1)],
        prio, is_hv_metal=True))
    prio += 1
    regions.append(_Region(
        "head_housing",
        [(0.0, z_dome1), (c.terminal_radius, z_dome1),
         (c.terminal_radius, c.total_height), (0.0, c.total_height)],
        prio, is_hv_metal=True))
    prio += 1

    # -- grounded EMU tank (bottom) ------------------------------------------ #
    regions.append(_Region(
        "base_tank",
        [(0.0, 0.0), (c.tank_radius, 0.0),
         (c.tank_radius, c.tank_height), (0.0, c.tank_height)],
        prio, is_ground_metal=True))
    prio += 1

    # -- porcelain insulator: wall + weather sheds ----------------------------- #
    # Sheds are separate "porcelain_shed" regions (same material) so their
    # tips can get their own finer mesh size and queryable surface fields.
    # Their roots overlap the wall rectangle; classification priority resolves
    # the overlap and OCC fragmenting keeps everything conformal.
    for zc in c.shed_centers():
        regions.append(_Region("porcelain_shed", c.shed_polygon(zc), prio))
        prio += 1
    r_in = c.porcelain_inner_radius
    r_out = r_in + c.porcelain_thickness
    regions.append(_Region(
        "porcelain",
        [(r_in, c.tank_height), (r_out, c.tank_height),
         (r_out, c.stack_z_hi), (r_in, c.stack_z_hi)],
        prio))
    prio += 1

    # -- oil filling the insulator interior around the stack ------------------ #
    regions.append(_Region(
        "oil",
        [(0.0, c.tank_height), (r_in, c.tank_height),
         (r_in, c.stack_z_hi), (0.0, c.stack_z_hi)],
        prio))
    prio += 1

    # -- air box (lowest priority) -------------------------------------------- #
    Rf = float(c.farfield_radius)
    zc = 0.5 * c.total_height
    regions.append(_Region(
        "air",
        [(0.0, zc - Rf), (Rf, zc - Rf), (Rf, zc + Rf), (0.0, zc + Rf)],
        prio))

    return regions


def build_cvt(c: CVTParams, out_path: str, *, verbose: bool = True,
              gui: bool = False) -> MeshResult:
    """Build and mesh the parametric DDB-style CVT; write MSH 4.1 to out_path."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("cvt")
        regions = _cvt_regions(c)
        result = _build_and_tag(regions, c, out_path, tag_farfield=True,
                                verbose=verbose)
        if gui:  # pragma: no cover - interactive only
            gmsh.fltk.run()
        return result
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------- #
# Degenerate coaxial validation geometry (uses the same build/tag core)
# --------------------------------------------------------------------------- #


def build_coax(
    r_inner: float,
    r_outer: float,
    length: float,
    out_path: str,
    *,
    lc: float = 0.002,
    refinement: float = 1.0,
    verbose: bool = True,
    gui: bool = False,
) -> MeshResult:
    """Plain coaxial cylinder capacitor: one dielectric annulus a<r<b, 0<z<L.

    HV (inner, r=a) and ground (outer, r=b) Dirichlet curves; top/bottom edges
    are natural (zero normal current) so the field is purely radial and matches
    the analytic coax capacitance.  Uses the same fragment/classify/tag core as
    the full CT (spec: "using the same builder code paths").
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("coax")

        # single dielectric annulus; HV/ground curves tagged by radius.
        regions = [
            _Region("dielectric",
                    [(r_inner, 0.0), (r_outer, 0.0),
                     (r_outer, length), (r_inner, length)],
                    0),
        ]
        # uniform sizing
        lc_eff = lc / max(refinement, 1e-6)
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc_eff)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc_eff)

        result = _build_and_tag(regions, None, out_path, tag_farfield=False,
                                verbose=verbose, coax_radii=(r_inner, r_outer))
        if gui:  # pragma: no cover
            gmsh.fltk.run()
        return result
    finally:
        gmsh.finalize()

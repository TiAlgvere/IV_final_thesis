"""Clean baseline Gmsh geometry for the publication pipeline.

This is a minimal axisymmetric starter geometry that keeps the physical group
names stable for the Elmer pipeline while staying isolated from the legacy
scikit-fem prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

import gmsh


@dataclass(frozen=True)
class CleanBaselineParameters:
    inner_radius: float = 0.04
    porcelain_radius: float = 0.11
    outer_radius: float = 0.18
    air_height: float = 0.28
    porcelain_height: float = 0.18
    hv_tip_height: float = 0.03
    ground_base_height: float = 0.02
    lc_scale: float = 1.0
    mesh_size_air: float = 0.025
    mesh_size_porcelain: float = 0.015
    mesh_size_electrode: float = 0.01
    # Task 006: thin conductive-pollution layer on the outer porcelain face.
    # Modelled as a thin VOLUME strip (documented approximation of true surface
    # conductivity). Only added when build_clean_baseline_mesh(with_pollution_layer=True).
    pollution_thickness: float = 0.005  # radial extent of the strip [m]
    mesh_size_pollution: float = 0.002
    # Task 008 geometry audit: strip z-extent on the outer face. Defaults give the
    # full-height strip used in Task 006 (z0=0, length=porcelain_height).
    pollution_z0: float = 0.0  # lower z of the strip [m]
    pollution_length: float | None = None  # z-extent [m]; None => full porcelain_height


def build_clean_baseline_mesh(
    output_msh: Path,
    parameters: CleanBaselineParameters | None = None,
    with_pollution_layer: bool = False,
) -> dict[str, Any]:
    """Build a minimal 2-D axisymmetric baseline mesh and write it to disk.

    When ``with_pollution_layer`` is True a thin conductive-pollution strip is
    carved out of the outer porcelain face and tagged as the ``POLLUTION_LAYER``
    body. The strip is always present in that mesh so a conductivity sweep can
    vary only its material sigma without re-meshing; at sigma=0 (with the layer's
    relative permittivity set equal to porcelain) the case is identical to clean.
    The default (False) leaves the clean baseline mesh unchanged.
    """

    params = parameters or CleanBaselineParameters()
    output_msh = Path(output_msh)
    output_msh.parent.mkdir(parents=True, exist_ok=True)

    pollution_x0 = params.outer_radius - params.pollution_thickness
    pollution_z0 = params.pollution_z0
    pollution_length = params.pollution_length if params.pollution_length is not None else params.porcelain_height

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add("clean_baseline")

        mesh_size_air = params.mesh_size_air * params.lc_scale

        outer = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, params.outer_radius, params.air_height)
        porcelain = gmsh.model.occ.addRectangle(
            0.0,
            0.0,
            0.0,
            params.outer_radius,
            params.porcelain_height,
        )
        fragment_tools = [(2, porcelain)]
        if with_pollution_layer:
            pollution = gmsh.model.occ.addRectangle(
                pollution_x0,
                pollution_z0,
                0.0,
                params.pollution_thickness,
                pollution_length,
            )
            fragment_tools.append((2, pollution))
        gmsh.model.occ.synchronize()

        gmsh.model.occ.fragment([(2, outer)], fragment_tools)
        gmsh.model.occ.synchronize()

        for dim, tag in gmsh.model.getEntities(0):
            gmsh.model.mesh.setSize([(dim, tag)], mesh_size_air)

        air_surfaces: list[int] = []
        porcelain_surfaces: list[int] = []
        pollution_surfaces: list[int] = []
        boundary_lines: dict[str, list[int]] = {
            "HV_TERMINAL": [],
            "GROUND": [],
            "TAP_ELECTRODE": [],
            "POLLUTION_SURFACE": [],
        }

        def near(value: float, target: float, tol: float = 1e-6) -> bool:
            return abs(value - target) <= tol

        for dim, tag in gmsh.model.getEntities(2):
            bbox = gmsh.model.occ.getBoundingBox(dim, tag)
            is_air = (
                near(bbox[1], params.porcelain_height)
                and near(bbox[4], params.air_height)
            )
            # The strip is the only non-air surface whose left edge sits at
            # pollution_x0 (bulk porcelain starts at x=0). Avoid tight ymax
            # comparisons: gmsh pads bounding boxes by ~1e-7 after meshing.
            is_pollution = (
                with_pollution_layer
                and not is_air
                and near(bbox[0], pollution_x0)
            )
            if is_air:
                air_surfaces.append(tag)
            elif is_pollution:
                pollution_surfaces.append(tag)
            else:
                porcelain_surfaces.append(tag)

            for bdim, btag in gmsh.model.getBoundary([(2, tag)], oriented=False):
                if bdim != 1:
                    continue
                bxmin, bymin, _, bxmax, bymax, _ = gmsh.model.occ.getBoundingBox(1, btag)
                if near(bymin, 0.0) and near(bymax, 0.0):
                    boundary_lines["GROUND"].append(btag)
                elif near(bymin, params.air_height) and near(bymax, params.air_height):
                    boundary_lines["HV_TERMINAL"].append(btag)
                elif near(bymin, params.porcelain_height) and near(bymax, params.porcelain_height):
                    boundary_lines["TAP_ELECTRODE"].append(btag)
                elif near(bxmin, params.outer_radius) and bymax <= params.porcelain_height + 1e-6:
                    boundary_lines["POLLUTION_SURFACE"].append(btag)

        if not air_surfaces or not porcelain_surfaces:
            raise RuntimeError("baseline geometry did not produce distinct air/porcelain surfaces")
        if with_pollution_layer and not pollution_surfaces:
            raise RuntimeError("pollution layer requested but no pollution surface was produced")

        pg_air = gmsh.model.addPhysicalGroup(2, sorted(set(air_surfaces)))
        gmsh.model.setPhysicalName(2, pg_air, "AIR")

        pg_porcelain = gmsh.model.addPhysicalGroup(2, sorted(set(porcelain_surfaces)))
        gmsh.model.setPhysicalName(2, pg_porcelain, "PORCELAIN")

        if pollution_surfaces:
            pg_pollution = gmsh.model.addPhysicalGroup(2, sorted(set(pollution_surfaces)))
            gmsh.model.setPhysicalName(2, pg_pollution, "POLLUTION_LAYER")
            # Refine the thin strip so the conductive layer is resolved across its thickness.
            for tag in pollution_surfaces:
                for pdim, ptag in gmsh.model.getBoundary([(2, tag)], oriented=False, recursive=True):
                    gmsh.model.mesh.setSize([(pdim, ptag)], params.mesh_size_pollution)

        for name, tags in boundary_lines.items():
            if tags:
                pg = gmsh.model.addPhysicalGroup(1, sorted(set(tags)))
                gmsh.model.setPhysicalName(1, pg, name)

        gmsh.model.mesh.generate(2)
        gmsh.write(str(output_msh))

        physical_groups = {
            gmsh.model.getPhysicalName(dim, tag): {
                "dim": dim,
                "tag": tag,
                "entities": [int(entity) for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)],
            }
            for dim, tag in gmsh.model.getPhysicalGroups()
        }

        metadata = {
            "geometry": "clean_baseline_axisymmetric",
            "output_msh": str(output_msh),
            "with_pollution_layer": with_pollution_layer,
            "parameters": asdict(params),
            "physical_groups": physical_groups,
        }
        return metadata
    finally:
        gmsh.finalize()


def write_mesh_log(log_path: Path, metadata: dict[str, Any]) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

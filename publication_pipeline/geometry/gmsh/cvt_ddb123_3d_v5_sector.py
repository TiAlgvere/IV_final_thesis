"""Task 026 - sector-split insulator for azimuthal surface-conductance validation.

Same v5 anatomy (74 alternating sheds, components, creepage 3075, homogenized C1/tap/C2),
but the porcelain insulator is built as SEPARATE angular sectors so the insulator-surface
creepage is partitioned into EXPLICIT physical groups insulator_s0..s{N-1} (no MATC theta in
the BC). Cumulative sector bounds [0,30,60,90,180,360] deg give sectors that combine into the
requested polluted coverage:
  30 deg = s0 ; 60 = s0+s1 ; 90 = s0+s1+s2 ; 180 = +s3 ; 360 = all.

Each angular porcelain wedge is a copy of the fused (wall + 74 rounded-lip sheds) cross-section
rotated to the sector start and revolved by the sector width. All wedges are porcelain; their
air interfaces become insulator_s{i}. Full-360 components (stack/head/tank/EMU/box/oil) are
reused unchanged from v5. Azimuthal mesh grading is added near the radial cut planes.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gmsh

from .cvt_ddb123 import CVTParams
from .cvt_ddb123_3d_v2 import _face_xz
from .cvt_ddb123_3d_v4 import MAT_OF_V4, MATERIALS, HV_VOLS, GND_VOLS, INACTIVE  # noqa: F401
from .cvt_ddb123_3d_v5 import (OV_LARGE, LIP, _shed_polygon, _shed_data, _specs,  # noqa: F401
                               _tune_ov_small, _creepage_mm)

SECTOR_BOUNDS_DEG = [0.0, 30.0, 60.0, 90.0, 180.0, 360.0]


def _mat_of(name: str) -> str:
    if name.startswith("porcelain"):
        return "porcelain"
    return MAT_OF_V4.get(name, "air")


def build_sector_mesh(out_msh: Path, params: CVTParams | None = None, *,
                      sector_bounds=None, shed_size: float = 0.0045, edge_size: float = 0.0022,
                      edge_band_deg: float = 2.5, r_air: float = 0.9, z_pad: float = 0.35,
                      lc: dict | None = None, opt_passes: int = 3, verbose: bool = False) -> dict[str, Any]:
    c = params or CVTParams()
    c.n_sheds = 74
    bounds = [float(b) for b in (sector_bounds or SECTOR_BOUNDS_DEG)]
    nsec = len(bounds) - 1
    out_msh = Path(out_msh); out_msh.parent.mkdir(parents=True, exist_ok=True)
    ov_small = _tune_ov_small(c)
    LCd = {"foil_tap": 0.008, "stack_bottom": 0.010, "stack_top": 0.010,
           "element_c1": 0.022, "element_c2": 0.014, "oil": 0.024,
           "head_metal": 0.030, "head_oil": 0.030, "terminal": 0.010, "terminal_cap": 0.006,
           "tank_metal": 0.040, "base_emu": 0.028, "reactor": 0.018, "aux_component": 0.016,
           "base_oil": 0.032, "secondary_box": 0.026, "oil_valve": 0.014, "oil_level": 0.014,
           "resin": 0.012, "shed": shed_size, "air": 0.230}
    if lc:
        LCd.update(lc)
    specs, z_tap, z_c1, z_c2 = _specs(c)
    r_in = c.porcelain_inner_radius; r_out = r_in + c.porcelain_thickness
    centers, overhangs, pitch = _shed_data(c, ov_small)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0020)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.35)
        gmsh.model.add("cvt_ddb123_sector")
        occ = gmsh.model.occ
        inputs, prio_in, name_in = [], [], []

        def add(name, tag, prio):
            inputs.append((3, tag)); prio_in.append(prio); name_in.append(name)

        for (name, kind, p, prio) in specs:
            if kind == "cyl":
                R, z0, z1 = p
                add(name, occ.addCylinder(0, 0, z0, 0, 0, z1 - z0, R), prio)
            elif kind == "box":
                add(name, occ.addBox(*p), prio)
            elif kind == "sphere":
                x, y, z, rr = p
                add(name, occ.addSphere(x, y, z, rr), prio)

        # porcelain cross-section (wall + 74 sheds), fused once
        wall = _face_xz(occ, [(r_in, c.stack_z_lo), (r_out, c.stack_z_lo),
                              (r_out, c.stack_z_hi), (r_in, c.stack_z_hi)], radii=[0.0, LIP, LIP, 0.0])
        sheds = [_face_xz(occ, _shed_polygon(c, zc, ov), radii=[0.0, LIP, LIP, 0.0])
                 for zc, ov in zip(centers, overhangs)]
        occ.synchronize()
        fused, _ = occ.fuse([(2, wall)], [(2, s) for s in sheds])
        # copies: one face per sector (revolve consumes each)
        faces = [fused] + [occ.copy(fused) for _ in range(nsec - 1)]
        for i in range(nsec):
            a, b = bounds[i], bounds[i + 1]
            f = faces[i]
            if abs(a) > 1e-9:
                occ.rotate(f, 0, 0, 0, 0, 0, 1, math.radians(a))
            rev = occ.revolve(f, 0, 0, 0, 0, 0, 1, math.radians(b - a))
            for (d, t) in rev:
                if d == 3:
                    add(f"porcelain_s{i}", t, 6)

        add("air", occ.addCylinder(0, 0, -z_pad, 0, 0, c.total_height + 2 * z_pad, r_air), 100)

        out, fmap = occ.fragment(inputs, [])
        occ.synchronize()
        best: dict[int, tuple[float, str]] = {}
        for i, outs in enumerate(fmap):
            for (d, t) in outs:
                if d == 3 and (t not in best or prio_in[i] < best[t][0]):
                    best[t] = (prio_in[i], name_in[i])
        region_volumes: dict[str, list[int]] = {}
        name_of_vol: dict[int, str] = {}
        for t, (p, name) in best.items():
            region_volumes.setdefault(name, []).append(t); name_of_vol[t] = name
        # physical volume groups: merge porcelain_s* under their own names (keep distinct)
        for name, tags in region_volumes.items():
            pg = gmsh.model.addPhysicalGroup(3, sorted(set(tags)))
            gmsh.model.setPhysicalName(3, pg, name)

        surf_names: dict[int, set] = {}
        for (dim, vtag) in gmsh.model.getEntities(3):
            for (bdim, btag) in gmsh.model.getBoundary([(3, vtag)], oriented=False):
                if bdim == 2:
                    surf_names.setdefault(abs(btag), set()).add(name_of_vol[vtag])
        hv_s, gnd_s, ff_s = [], [], []
        ins_sec: dict[int, list[int]] = {i: [] for i in range(nsec)}
        for stag, ns in surf_names.items():
            if (ns & HV_VOLS) and (ns - HV_VOLS):
                hv_s.append(stag)
            if (ns & GND_VOLS) and (ns - GND_VOLS):
                gnd_s.append(stag)
            for i in range(nsec):
                if (f"porcelain_s{i}" in ns) and ("air" in ns):
                    ins_sec[i].append(stag)
            if ns == {"air"}:
                ff_s.append(stag)

        def _sg(name, tags):
            if tags:
                pg = gmsh.model.addPhysicalGroup(2, sorted(set(tags)))
                gmsh.model.setPhysicalName(2, pg, name)
        _sg("hv_electrode", hv_s); _sg("ground_electrode", gnd_s); _sg("farfield", ff_s)
        for i in range(nsec):
            _sg(f"insulator_s{i}", ins_sec[i])
        # also a combined insulator_surface group (all sectors) for convenience
        _sg("insulator_surface", [t for i in range(nsec) for t in ins_sec[i]])

        vol_by_mat: dict[str, float] = {}
        comp_bbox: dict[str, list] = {}
        for name, tags in region_volumes.items():
            mat = _mat_of(name)
            for t in tags:
                vol_by_mat[mat] = vol_by_mat.get(mat, 0.0) + occ.getMass(3, t)
                bb = occ.getBoundingBox(3, t)
                comp_bbox[name] = ([min(comp_bbox[name][k], bb[k]) for k in range(3)] +
                                   [max(comp_bbox[name][k+3], bb[k+3]) for k in range(3)]) if name in comp_bbox else list(bb)

        # ---- mesh sizing: per-region + azimuthal grading near the cut planes ----
        def _lc(name):
            if name.startswith("porcelain"):
                return LCd["shed"]
            return LCd.get(name, LCd["air"])
        gmsh.option.setNumber("Mesh.MeshSizeMax", LCd["air"])
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        cut_angles = bounds[:-1]  # radial cut planes (deg); patch edges
        for name, tags in sorted(region_volumes.items(), key=lambda kv: -_lc(kv[0])):
            size = _lc(name)
            pts = set()
            for t in tags:
                for (bd, bt) in gmsh.model.getBoundary([(3, t)], recursive=True, oriented=False):
                    if bd == 0:
                        pts.add(abs(bt))
            for pp in pts:
                xyz = gmsh.model.getValue(0, pp, [])
                s = size
                if name.startswith("porcelain"):
                    th = math.degrees(math.atan2(xyz[1], xyz[0])) % 360.0
                    dmin = min(min(abs(th - a), abs(th - a - 360), abs(th - a + 360)) for a in cut_angles)
                    if dmin <= edge_band_deg:
                        s = edge_size      # azimuthal grading at the patch edges
                gmsh.model.mesh.setSize([(0, pp)], s)

        gmsh.model.mesh.generate(3)
        for _ in range(opt_passes):
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception:
                pass
        gmsh.write(str(out_msh))

        # ---- stats ----
        import numpy as np
        ntags, _, _ = gmsh.model.mesh.getNodes()
        _, et3, _ = gmsh.model.mesh.getElements(3)
        _, et2, _ = gmsh.model.mesh.getElements(2)
        all_tets = np.array([t for tt in et3 for t in tt], dtype=np.int64)
        q = np.array(gmsh.model.mesh.getElementQualities(list(all_tets), "minSICN"))
        try:
            np.savez(str(out_msh) + ".quality.npz", tet_q=q)
        except Exception:
            pass
        pcts = {p: float(np.percentile(q, p)) for p in (1, 5, 50, 95)}

        # per-insulator-sector triangle quality + area
        def tri_group_stats(gname):
            for d, t in gmsh.model.getPhysicalGroups(2):
                if gmsh.model.getPhysicalName(d, t) == gname:
                    tri = []
                    for ent in gmsh.model.getEntitiesForPhysicalGroup(d, t):
                        _, ets, _ = gmsh.model.mesh.getElements(2, ent)
                        tri += [int(x) for x in (ets[0] if ets else [])]
                    if not tri:
                        return None
                    iq = np.array(gmsh.model.mesh.getElementQualities(tri, "minSICN"))
                    # area
                    area = 0.0
                    for tt in tri:
                        nds = gmsh.model.mesh.getElement(tt)[1]
                        pc = np.array([gmsh.model.mesh.getNode(int(nn))[0] for nn in nds])
                        area += 0.5 * np.linalg.norm(np.cross(pc[1] - pc[0], pc[2] - pc[0]))
                    return {"n": len(tri), "min": float(iq.min()), "mean": float(iq.mean()),
                            "p5": float(np.percentile(iq, 5)), "area_m2": float(area)}
            return None
        sector_stats = {f"insulator_s{i}": tri_group_stats(f"insulator_s{i}") for i in range(nsec)}
        ins_all = tri_group_stats("insulator_surface")

        # worst-100 with material
        tet_mat = {}
        for name, tg in region_volumes.items():
            mat = _mat_of(name)
            for ent in tg:
                _, ets, _ = gmsh.model.mesh.getElements(3, ent)
                for tt in (ets[0] if ets else []):
                    tet_mat[int(tt)] = mat
        order = np.argsort(q)[:100]; worst = []
        for idx in order:
            tt = int(all_tets[idx]); nds = gmsh.model.mesh.getElement(tt)[1]
            ctr = np.array([gmsh.model.mesh.getNode(int(nn))[0] for nn in nds]).mean(0)
            worst.append({"q": float(q[idx]), "mat": tet_mat.get(tt, "?"),
                          "rz": [float(math.hypot(ctr[0], ctr[1])), float(ctr[2])],
                          "theta_deg": float(math.degrees(math.atan2(ctr[1], ctr[0])) % 360)})

        pgroups = {gmsh.model.getPhysicalName(d, t): {"dim": d,
                   "n_entities": len(gmsh.model.getEntitiesForPhysicalGroup(d, t))}
                   for d, t in gmsh.model.getPhysicalGroups()}
        return {
            "geometry": "cvt_ddb123_3d_v5_sector", "output_msh": str(out_msh),
            "sector_bounds_deg": bounds, "n_sheds": 74, "ov_small_mm": ov_small * 1000,
            "n_nodes": len(ntags), "n_tets": int(all_tets.size),
            "n_boundary_tris": int(sum(len(t) for t in et2)),
            "quality_minSICN": {"min": float(q.min()), "mean": float(q.mean()), **pcts},
            "n_slivers_lt_0.05": int((q < 0.05).sum()), "worst100": worst,
            "insulator_surface_tri_quality": ins_all, "sector_tri_quality": sector_stats,
            "material_volumes_m3": vol_by_mat, "component_bbox": comp_bbox,
            "creepage_mm": _creepage_mm(c, ov_small),
            "tap_z": z_tap, "c1_z": z_c1, "c2_z": z_c2, "physical_groups": pgroups,
            "parameters": asdict(c),
        }
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    import json
    out = Path(__file__).resolve().parents[2] / "results" / "raw" / "026_azimuthal_surface_conductance_validation" / "test_sector.msh"
    m = build_sector_mesh(out, sector_bounds=[0, 30, 60, 90, 180, 360], shed_size=0.02,
                          lc={"air": 0.34}, opt_passes=1)
    tot = sum(s["area_m2"] for s in m["sector_tri_quality"].values() if s)
    print(json.dumps({"n_tets": m["n_tets"], "creepage_mm": round(m["creepage_mm"], 1),
                      "groups": [g for g in m["physical_groups"] if "insulator" in g],
                      "sector_areas": {k: (round(v["area_m2"], 5) if v else None)
                                       for k, v in m["sector_tri_quality"].items()},
                      "total_insulator_area": round(tot, 5),
                      "area_fracs": {k: (round(v["area_m2"]/tot, 4) if v else None)
                                     for k, v in m["sector_tri_quality"].items()}}, indent=2))

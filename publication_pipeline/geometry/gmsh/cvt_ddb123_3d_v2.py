"""Task 023 - geometry-fidelity TRUE 3D DDB-123 (full 360 deg) with 27 weather sheds.

Upgrade over Task 022's smooth-porcelain model. The porcelain insulator (outer shell +
27 rounded-lip weather sheds) is built by fusing the validated 2D porcelain+shed profile
(wall rectangle + 27 rounded shed faces, in the x-z plane) and revolving it 360 deg about
z. Everything else (homogenized C1/tap/C2 stack, oil, HV head+terminal, grounded tank,
air) stays as OCC primitive cylinders. A conformal `fragment` partitions the union and the
output is classified by the fragment MAP + priority (robust for the non-convex porcelain /
air / oil volumes - a centroid test fails there).

CRITICAL material rule: the inner column is oil + capacitor element; ONLY the outer shell
and sheds are porcelain. This is enforced by priority (oil/stack win over porcelain inside
r<=r_in) and verified by the material-volume table.

NO new physics. NO localized faults. Geometry/material/mesh fidelity only.
Reuses the validated CVTParams (eps_r_eff, shed profile, creepage) from cvt_ddb123.py.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gmsh

from .cvt_ddb123 import CVTParams


# ----------------------------------------------------------------- x-z plane face helpers
def _unit(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    L = math.hypot(dx, dy)
    return (dx / L, dy / L)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _face_xz(occ, poly, radii=None):
    """Planar face in the x-z plane (points at (r,0,z)) with optional per-vertex fillets."""
    n = len(poly)
    V = [(float(p[0]), float(p[1])) for p in poly]
    if not radii:
        pts = [occ.addPoint(r, 0.0, z) for (r, z) in V]
        lines = [occ.addLine(pts[i], pts[(i + 1) % n]) for i in range(n)]
        return occ.addPlaneSurface([occ.addCurveLoop(lines)])
    seg_start = [0] * n; seg_end = [0] * n; arc = [None] * n
    for i in range(n):
        P, A, B, r = V[i], V[(i - 1) % n], V[(i + 1) % n], radii[i]
        if r <= 0.0:
            p = occ.addPoint(P[0], 0.0, P[1]); seg_start[i] = seg_end[i] = p; continue
        u = _unit(A, P); w = _unit(B, P)
        ang = math.acos(max(-1.0, min(1.0, u[0] * w[0] + u[1] * w[1])))
        t = r / math.tan(ang / 2.0)
        t = min(t, 0.49 * _dist(A, P), 0.49 * _dist(B, P))
        r_eff = t * math.tan(ang / 2.0); d = r_eff / math.sin(ang / 2.0)
        Tin = (P[0] + t * u[0], P[1] + t * u[1]); Tout = (P[0] + t * w[0], P[1] + t * w[1])
        bx, by = u[0] + w[0], u[1] + w[1]; bn = math.hypot(bx, by)
        C = (P[0] + d * bx / bn, P[1] + d * by / bn)
        seg_end[i] = occ.addPoint(Tin[0], 0.0, Tin[1])
        seg_start[i] = occ.addPoint(Tout[0], 0.0, Tout[1])
        arc[i] = (seg_end[i], occ.addPoint(C[0], 0.0, C[1]), seg_start[i])
    curves = []
    for i in range(n):
        curves.append(occ.addLine(seg_start[i], seg_end[(i + 1) % n]))
        a = arc[(i + 1) % n]
        if a is not None:
            curves.append(occ.addCircleArc(a[0], a[1], a[2]))
    return occ.addPlaneSurface([occ.addCurveLoop(curves)])


def _cyl_specs(c: CVTParams):
    """(name, R, z0, z1, prio, hv, gnd) for the axisymmetric cylinder regions + tap z."""
    h = c.element_height(); rs = c.stack_radius; disc = c.disc_thickness
    z0 = c.stack_z_lo
    n_c1 = c.n_elements - c.n_elements_c2; n_c2 = c.n_elements_c2
    z_sb = (z0, z0 + disc); z_c2 = (z_sb[1], z_sb[1] + n_c2 * h)
    z_tp = (z_c2[1], z_c2[1] + disc); z_c1 = (z_tp[1], z_tp[1] + n_c1 * h)
    z_st = (z_c1[1], z_c1[1] + disc)
    z_dome = (c.stack_z_hi, c.stack_z_hi + c.head_height)
    z_term = (z_dome[1], z_dome[1] + c.terminal_height)
    return [
        ("stack_bottom", rs, *z_sb, 0, False, True),
        ("element_c2",   rs, *z_c2, 1, False, False),
        ("foil_tap",     rs, *z_tp, 2, False, False),
        ("element_c1",   rs, *z_c1, 3, False, False),
        ("stack_top",    rs, *z_st, 4, True, False),
        ("head_housing", c.head_radius, *z_dome, 5, True, False),
        ("head_housing", c.terminal_radius, *z_term, 6, True, False),
        ("base_tank",    c.tank_radius, 0.0, z0, 7, False, True),
        ("oil",          c.porcelain_inner_radius, z0, c.stack_z_hi, 8, False, False),
    ], (z_tp[0] + 0.5 * disc)


MAT_OF = {"air": "air", "oil": "oil", "porcelain": "porcelain",
          "element_c1": "element", "element_c2": "element",
          "stack_bottom": "metal", "stack_top": "metal", "foil_tap": "metal",
          "head_housing": "metal", "base_tank": "metal"}


def build_cvt_ddb123_3d_v2(out_msh: Path, params: CVTParams | None = None, *,
                           n_sheds: int | None = None, r_air: float = 0.8, z_pad: float = 0.30,
                           lc: dict | None = None, curvature: float = 12.0,
                           verbose: bool = False) -> dict[str, Any]:
    c = params or CVTParams()
    if n_sheds is not None:
        c.n_sheds = n_sheds
    out_msh = Path(out_msh); out_msh.parent.mkdir(parents=True, exist_ok=True)
    LC = {"foil_tap": 0.010, "stack_bottom": 0.010, "stack_top": 0.010,
          "element_c1": 0.020, "element_c2": 0.016, "porcelain": 0.012, "oil": 0.025,
          "head_housing": 0.040, "base_tank": 0.050, "air": 0.200, "shed": 0.008}
    if lc:
        LC.update(lc)

    cyls, z_tap = _cyl_specs(c)
    r_in = c.porcelain_inner_radius
    r_out = r_in + c.porcelain_thickness
    lip = c.shed_lip_round
    wall_round = c.shed_lip_round  # round the porcelain wall outer top/bottom transitions

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add("cvt_ddb123_3d_v2")
        occ = gmsh.model.occ

        # --- axisymmetric cylinder regions ---
        cyl_tags = [occ.addCylinder(0, 0, z0, 0, 0, z1 - z0, R)
                    for (name, R, z0, z1, prio, hv, gnd) in cyls]

        # --- porcelain: fuse wall + 27 rounded-lip sheds (x-z faces) then revolve 360 ---
        wall = _face_xz(occ, [(r_in, c.stack_z_lo), (r_out, c.stack_z_lo),
                              (r_out, c.stack_z_hi), (r_in, c.stack_z_hi)],
                        radii=[0.0, wall_round, wall_round, 0.0])
        shed_faces = [_face_xz(occ, c.shed_polygon(zc), radii=[0.0, lip, lip, 0.0])
                      for zc in c.shed_centers()]
        occ.synchronize()
        fused, _ = occ.fuse([(2, wall)], [(2, s) for s in shed_faces])
        rev = occ.revolve(fused, 0, 0, 0, 0, 0, 1, 2.0 * math.pi)
        porc_vols = [t for (d, t) in rev if d == 3]

        # --- air domain ---
        air = occ.addCylinder(0, 0, -z_pad, 0, 0, c.total_height + 2 * z_pad, r_air)

        # --- conformal partition ---
        inputs, prio_in, name_in = [], [], []
        for tag, (name, R, z0, z1, prio, hv, gnd) in zip(cyl_tags, cyls):
            inputs.append((3, tag)); prio_in.append(prio); name_in.append(name)
        for t in porc_vols:
            inputs.append((3, t)); prio_in.append(9); name_in.append("porcelain")
        inputs.append((3, air)); prio_in.append(100); name_in.append("air")

        out, fmap = occ.fragment(inputs, [])
        occ.synchronize()

        best: dict[int, tuple[int, str]] = {}
        for i, outs in enumerate(fmap):
            for (d, t) in outs:
                if d != 3:
                    continue
                if t not in best or prio_in[i] < best[t][0]:
                    best[t] = (prio_in[i], name_in[i])
        region_volumes: dict[str, list[int]] = {}
        name_of_vol: dict[int, str] = {}
        for t, (p, name) in best.items():
            region_volumes.setdefault(name, []).append(t)
            name_of_vol[t] = name

        for name, tags in region_volumes.items():
            pg = gmsh.model.addPhysicalGroup(3, sorted(set(tags)))
            gmsh.model.setPhysicalName(3, pg, name)

        # --- surface groups ---
        surf_names: dict[int, set] = {}
        for (dim, vtag) in gmsh.model.getEntities(3):
            for (bdim, btag) in gmsh.model.getBoundary([(3, vtag)], oriented=False):
                if bdim == 2:
                    surf_names.setdefault(abs(btag), set()).add(name_of_vol[vtag])
        hv_vols, gnd_vols = {"stack_top", "head_housing"}, {"base_tank", "stack_bottom"}
        hv_s, gnd_s, ins_s, ff_s = [], [], [], []
        for stag, ns in surf_names.items():
            if (ns & hv_vols) and (ns - hv_vols):
                hv_s.append(stag)
            if (ns & gnd_vols) and (ns - gnd_vols):
                gnd_s.append(stag)
            if {"porcelain", "air"} <= ns:
                ins_s.append(stag)
            if ns == {"air"}:
                ff_s.append(stag)

        def _sg(name, tags):
            if tags:
                pg = gmsh.model.addPhysicalGroup(2, sorted(set(tags)))
                gmsh.model.setPhysicalName(2, pg, name)
        _sg("hv_electrode", hv_s); _sg("ground_electrode", gnd_s)
        _sg("insulator_surface", ins_s); _sg("farfield", ff_s)

        # --- geometric material volumes (exact, from OCC mass) ---
        vol_by_mat: dict[str, float] = {}
        for name, tags in region_volumes.items():
            mat = MAT_OF.get(name, "air")
            for t in tags:
                vol_by_mat[mat] = vol_by_mat.get(mat, 0.0) + occ.getMass(3, t)

        # --- mesh sizing: curvature at lips + per-region size (coarse-first) ---
        def _lc(name):
            return LC.get(name, LC["air"])
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(0.004, 0.5 * lip))
        gmsh.option.setNumber("Mesh.MeshSizeMax", LC["air"])
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        # porcelain points get the shed size (sheds dominate porcelain refinement need)
        ordered = sorted(region_volumes.items(), key=lambda kv: -_lc(kv[0]))
        for name, tags in ordered:
            size = LC["shed"] if name == "porcelain" else _lc(name)
            pts = set()
            for t in tags:
                for (bd, bt) in gmsh.model.getBoundary([(3, t)], recursive=True, oriented=False):
                    if bd == 0:
                        pts.add(abs(bt))
            if pts:
                gmsh.model.mesh.setSize([(0, p) for p in pts], size)

        gmsh.model.mesh.generate(3)
        try:
            gmsh.model.mesh.optimize("Netgen")
        except Exception:
            pass
        gmsh.write(str(out_msh))

        # --- mesh statistics ---
        import numpy as np
        ntags, _, _ = gmsh.model.mesh.getNodes()
        _, etags3, _ = gmsh.model.mesh.getElements(3)
        _, etags2, _ = gmsh.model.mesh.getElements(2)
        all_tets = np.array([t for tt in etags3 for t in tt], dtype=np.int64)
        n_tet = int(all_tets.size); n_tri = int(sum(len(t) for t in etags2))
        q = np.array(gmsh.model.mesh.getElementQualities(list(all_tets), "minSICN"))
        pcts = {p: float(np.percentile(q, p)) for p in (1, 5, 50, 95)}
        try:
            np.savez(str(out_msh) + ".quality.npz", tet_q=q)
        except Exception:
            pass

        # tet -> material via physical volume entities
        tet_mat = {}
        for name, _tags in region_volumes.items():
            mat = MAT_OF.get(name, "air")
            for ent in _tags:
                _, ets, _ = gmsh.model.mesh.getElements(3, ent)
                for t in (ets[0] if ets else []):
                    tet_mat[int(t)] = mat
        order = np.argsort(q)
        worst = []
        for idx in order[:20]:
            tt = int(all_tets[idx])
            nds = gmsh.model.mesh.getElement(tt)[1]
            coords = np.array([gmsh.model.mesh.getNode(int(n))[0] for n in nds])
            ctr = coords.mean(0)
            worst.append({"q": float(q[idx]), "mat": tet_mat.get(tt, "?"),
                          "rz": [float(math.hypot(ctr[0], ctr[1])), float(ctr[2])]})
        n_sliver = int((q < 0.05).sum())

        # insulator_surface boundary-triangle quality
        ins_q = None
        for d, t in gmsh.model.getPhysicalGroups(2):
            if gmsh.model.getPhysicalName(d, t) == "insulator_surface":
                tris = []
                for ent in gmsh.model.getEntitiesForPhysicalGroup(d, t):
                    _, ets, _ = gmsh.model.mesh.getElements(2, ent)
                    tris += [int(x) for x in (ets[0] if ets else [])]
                if tris:
                    iq = np.array(gmsh.model.mesh.getElementQualities(tris, "minSICN"))
                    ins_q = {"n": len(tris), "min": float(iq.min()), "mean": float(iq.mean()),
                             "p5": float(np.percentile(iq, 5))}

        pgroups = {gmsh.model.getPhysicalName(d, t): {"dim": d,
                   "n_entities": len(gmsh.model.getEntitiesForPhysicalGroup(d, t))}
                   for d, t in gmsh.model.getPhysicalGroups()}

        return {
            "geometry": "cvt_ddb123_true3d_v2_sheds", "output_msh": str(out_msh),
            "n_sheds": c.n_sheds, "n_nodes": len(ntags), "n_tets": n_tet, "n_boundary_tris": n_tri,
            "quality_minSICN": {"min": float(q.min()), "mean": float(q.mean()), **pcts},
            "n_slivers_lt_0.05": n_sliver, "worst20": worst,
            "insulator_surface_tri_quality": ins_q,
            "material_volumes_m3": vol_by_mat,
            "element_epsr_eff": c.element_epsr_eff(),
            "divider_ratio_nominal": c.divider_ratio_nominal(),
            "creepage_mm": c.creepage_distance() * 1000.0,
            "shed_pitch_mm": (c.stack_z_hi - c.tank_height) / c.n_sheds * 1000.0,
            "shed_overhang_mm": c.shed_overhang * 1000.0, "shed_lip_round_mm": lip * 1000.0,
            "tap_z": z_tap, "lc": LC, "parameters": asdict(c), "physical_groups": pgroups,
        }
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    import json
    out = Path(__file__).resolve().parents[2] / "results" / "raw" / "023_true3d_geometry_fidelity" / "cvt3d_v2.msh"
    m = build_cvt_ddb123_3d_v2(out, verbose=False)
    show = {k: v for k, v in m.items() if k not in ("parameters", "lc", "worst20")}
    print(json.dumps(show, indent=2, default=str))
    print("worst5:", [(round(w["q"], 3), w["mat"]) for w in m["worst20"][:5]])

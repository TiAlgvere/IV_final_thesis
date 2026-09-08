"""Task 024 - component-accurate TRUE 3D DDB-123 (full 360 deg).

Upgrades the Task 023 shed-fidelity model toward the real Arteche DDB-123 architecture
(per the section diagram): primary terminal, oil-volume compensating HEAD (metal shell +
internal oil), porcelain insulator + 27 sheds, upper/lower capacitor stacks C1/C2, an
explicit intermediate-voltage TAP electrode, a hollow grounded TANK enclosing a distinct
base electromagnetic unit (inductive VT), a compensating-reactor volume, base oil, and an
external secondary terminal box. A mid oil-compensating coupling band marks the two-section
porcelain joint.

Electrical core unchanged (validated): homogenized C1/tap/C2 dielectric column with
eps_r_eff and the tap at n_c2/n height -> C ~ 5600 pF, ratio 1/6. The added components are
either shielded (head oil inside HV metal) or sit in/near the grounded tank (base unit,
reactor, box), so they are electrically inert for the divider. They are tagged
inactive/simplified; the mid coupling is a floating structural ring (its divider impact is
validated, not assumed).

Built from OCC primitives + the revolved fused porcelain (wall + rounded-lip sheds), then a
conformal `fragment` classified by the fragment MAP + priority (robust for non-convex
volumes). NO new physics, NO localized faults.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gmsh

from .cvt_ddb123 import CVTParams
from .cvt_ddb123_3d_v2 import _face_xz

# name -> material class (the 5 EQS materials)
MAT_OF_V3 = {
    "air": "air", "oil": "oil", "head_oil": "oil", "base_oil": "oil",
    "porcelain": "porcelain", "element_c1": "element", "element_c2": "element",
    "terminal": "metal", "terminal_cap": "metal", "head_metal": "metal",
    "stack_top": "metal", "stack_bottom": "metal", "foil_tap": "metal",
    "tank_metal": "metal", "base_emu": "metal", "reactor": "metal",
    "secondary_box": "metal", "mid_coupling": "metal",
}
# HV / ground electrodes (Dirichlet); everything else metal is floating
HV_VOLS = {"terminal", "terminal_cap", "head_metal", "stack_top"}
GND_VOLS = {"tank_metal", "stack_bottom"}
# physically present but not active EM components in the EQS model
INACTIVE = {"base_emu", "reactor", "secondary_box", "mid_coupling"}


def _specs(c: CVTParams, mid_coupling: bool):
    """List of (name, kind, params, prio). Lower prio wins on overlap.
    kind: 'cyl'(r,z0,z1) | 'box'(x,y,z,dx,dy,dz) | 'sphere'(x,y,z,r) | 'porcelain'."""
    h = c.element_height(); rs = c.stack_radius; disc = c.disc_thickness
    z0 = c.stack_z_lo
    n_c1 = c.n_elements - c.n_elements_c2; n_c2 = c.n_elements_c2
    z_sb = (z0, z0 + disc); z_c2 = (z_sb[1], z_sb[1] + n_c2 * h)
    z_tp = (z_c2[1], z_c2[1] + disc); z_c1 = (z_tp[1], z_tp[1] + n_c1 * h)
    z_st = (z_c1[1], z_c1[1] + disc)
    z_pt = c.stack_z_hi                                   # porcelain top / head bottom
    z_dome = (z_pt, z_pt + c.head_height)
    z_term = (z_dome[1], z_dome[1] + c.terminal_height)
    r_in = c.porcelain_inner_radius
    z_mid = 0.5 * (c.tank_height + c.stack_z_hi)
    specs = [
        # capacitor stack metal discs + dielectric columns
        ("stack_bottom", "cyl", (rs, *z_sb), 0),
        ("element_c2",   "cyl", (rs, *z_c2), 1),
        ("foil_tap",     "cyl", (rs, *z_tp), 0),
        ("element_c1",   "cyl", (rs, *z_c1), 1),
        ("stack_top",    "cyl", (rs, *z_st), 0),
        # oil-volume compensating head: metal shell + internal oil cavity
        ("head_oil",   "cyl", (0.14, z_dome[0] + 0.02, z_dome[1] - 0.01), 2),
        ("head_metal", "cyl", (c.head_radius, *z_dome), 3),
        ("terminal",     "cyl", (c.terminal_radius, z_term[0], z_term[1] - 0.005), 3),
        ("terminal_cap", "sphere", (0.0, 0.0, z_term[1] - 0.005, 0.022), 3),
        # grounded tank: metal shell enclosing base oil + base unit + reactor
        ("base_emu", "cyl", (0.12, 0.08, 0.40), 2),
        ("reactor",  "cyl", (0.06, 0.42, 0.50), 2),
        ("base_oil", "cyl", (0.20, 0.03, 0.52), 3),
        ("tank_metal", "cyl", (c.tank_radius, 0.0, z0), 4),
        # external secondary terminal box (on the tank side, +x)
        ("secondary_box", "box", (c.tank_radius - 0.005, -0.09, 0.10, 0.12, 0.18, 0.20), 1),
        # insulator interior oil around the stack
        ("oil", "cyl", (r_in, z0, c.stack_z_hi), 5),
    ]
    if mid_coupling:
        # mid oil-compensating coupling: external metal flange band (two-section joint).
        # prio 5.5 -> wins over porcelain (6) in r in [r_in, 0.17] but oil(5) keeps r<r_in.
        specs.append(("mid_coupling", "cyl", (0.17, z_mid - 0.03, z_mid + 0.03), 5.5))
    specs.append(("porcelain", "porcelain", None, 6))
    return specs, (z_tp[0] + 0.5 * disc), z_mid


def build_cvt_ddb123_3d_v3(out_msh: Path, params: CVTParams | None = None, *,
                           n_sheds: int | None = None, mid_coupling: bool = True,
                           r_air: float = 0.9, z_pad: float = 0.35,
                           lc: dict | None = None, curvature: float = 7.0,
                           verbose: bool = False) -> dict[str, Any]:
    c = params or CVTParams()
    if n_sheds is not None:
        c.n_sheds = n_sheds
    out_msh = Path(out_msh); out_msh.parent.mkdir(parents=True, exist_ok=True)
    LC = {"foil_tap": 0.008, "stack_bottom": 0.010, "stack_top": 0.010,
          "element_c1": 0.020, "element_c2": 0.014, "porcelain": 0.016, "oil": 0.026,
          "head_metal": 0.030, "head_oil": 0.030, "terminal": 0.010, "terminal_cap": 0.006,
          "tank_metal": 0.040, "base_emu": 0.030, "reactor": 0.020, "base_oil": 0.035,
          "secondary_box": 0.030, "mid_coupling": 0.020, "shed": 0.016, "air": 0.230}
    if lc:
        LC.update(lc)
    specs, z_tap, z_mid = _specs(c, mid_coupling)
    r_in = c.porcelain_inner_radius; r_out = r_in + c.porcelain_thickness
    lip = c.shed_lip_round

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add("cvt_ddb123_3d_v3")
        occ = gmsh.model.occ

        inputs, prio_in, name_in = [], [], []

        def add(name, tag, prio):
            inputs.append((3, tag)); prio_in.append(prio); name_in.append(name)

        for (name, kind, p, prio) in specs:
            if kind == "cyl":
                R, z0, z1 = p
                add(name, occ.addCylinder(0, 0, z0, 0, 0, z1 - z0, R), prio)
            elif kind == "box":
                x, y, z, dx, dy, dz = p
                add(name, occ.addBox(x, y, z, dx, dy, dz), prio)
            elif kind == "sphere":
                x, y, z, r = p
                add(name, occ.addSphere(x, y, z, r), prio)
            elif kind == "porcelain":
                wall = _face_xz(occ, [(r_in, c.stack_z_lo), (r_out, c.stack_z_lo),
                                      (r_out, c.stack_z_hi), (r_in, c.stack_z_hi)],
                                radii=[0.0, lip, lip, 0.0])
                sheds = [_face_xz(occ, c.shed_polygon(zc), radii=[0.0, lip, lip, 0.0])
                         for zc in c.shed_centers()]
                occ.synchronize()
                fused, _ = occ.fuse([(2, wall)], [(2, s) for s in sheds])
                rev = occ.revolve(fused, 0, 0, 0, 0, 0, 1, 2.0 * math.pi)
                for (d, t) in rev:
                    if d == 3:
                        add(name, t, prio)

        air = occ.addCylinder(0, 0, -z_pad, 0, 0, c.total_height + 2 * z_pad, r_air)
        add("air", air, 100)

        out, fmap = occ.fragment(inputs, [])
        occ.synchronize()

        best: dict[int, tuple[float, str]] = {}
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

        # surface groups
        surf_names: dict[int, set] = {}
        for (dim, vtag) in gmsh.model.getEntities(3):
            for (bdim, btag) in gmsh.model.getBoundary([(3, vtag)], oriented=False):
                if bdim == 2:
                    surf_names.setdefault(abs(btag), set()).add(name_of_vol[vtag])
        hv_s, gnd_s, ins_s, ff_s = [], [], [], []
        for stag, ns in surf_names.items():
            if (ns & HV_VOLS) and (ns - HV_VOLS):
                hv_s.append(stag)
            if (ns & GND_VOLS) and (ns - GND_VOLS):
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

        # geometric material volumes + per-component bounding boxes
        vol_by_mat: dict[str, float] = {}
        comp_bbox: dict[str, list] = {}
        for name, tags in region_volumes.items():
            mat = MAT_OF_V3.get(name, "air")
            for t in tags:
                vol_by_mat[mat] = vol_by_mat.get(mat, 0.0) + occ.getMass(3, t)
                bb = occ.getBoundingBox(3, t)
                if name in comp_bbox:
                    o = comp_bbox[name]
                    comp_bbox[name] = [min(o[0], bb[0]), min(o[1], bb[1]), min(o[2], bb[2]),
                                       max(o[3], bb[3]), max(o[4], bb[4]), max(o[5], bb[5])]
                else:
                    comp_bbox[name] = list(bb)

        # mesh sizing
        def _lc(name):
            return LC.get(name, LC["air"])
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(0.004, 0.5 * lip))
        gmsh.option.setNumber("Mesh.MeshSizeMax", LC["air"])
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        for name, tags in sorted(region_volumes.items(), key=lambda kv: -_lc(kv[0])):
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

        import numpy as np
        ntags, _, _ = gmsh.model.mesh.getNodes()
        _, etags3, _ = gmsh.model.mesh.getElements(3)
        _, etags2, _ = gmsh.model.mesh.getElements(2)
        all_tets = np.array([t for tt in etags3 for t in tt], dtype=np.int64)
        n_tet = int(all_tets.size); n_tri = int(sum(len(t) for t in etags2))
        q = np.array(gmsh.model.mesh.getElementQualities(list(all_tets), "minSICN"))
        try:
            np.savez(str(out_msh) + ".quality.npz", tet_q=q)
        except Exception:
            pass
        pcts = {p: float(np.percentile(q, p)) for p in (1, 5, 50, 95)}

        tet_mat = {}
        for name, _tags in region_volumes.items():
            mat = MAT_OF_V3.get(name, "air")
            for ent in _tags:
                _, ets, _ = gmsh.model.mesh.getElements(3, ent)
                for t in (ets[0] if ets else []):
                    tet_mat[int(t)] = mat
        order = np.argsort(q); worst = []
        for idx in order[:20]:
            tt = int(all_tets[idx]); nds = gmsh.model.mesh.getElement(tt)[1]
            ctr = np.array([gmsh.model.mesh.getNode(int(n))[0] for n in nds]).mean(0)
            worst.append({"q": float(q[idx]), "mat": tet_mat.get(tt, "?"),
                          "rz": [float(math.hypot(ctr[0], ctr[1])), float(ctr[2])]})
        n_sliver = int((q < 0.05).sum())

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
            "geometry": "cvt_ddb123_true3d_v3_components", "output_msh": str(out_msh),
            "n_sheds": c.n_sheds, "mid_coupling": mid_coupling,
            "n_nodes": len(ntags), "n_tets": n_tet, "n_boundary_tris": n_tri,
            "quality_minSICN": {"min": float(q.min()), "mean": float(q.mean()), **pcts},
            "n_slivers_lt_0.05": n_sliver, "worst20": worst,
            "insulator_surface_tri_quality": ins_q,
            "material_volumes_m3": vol_by_mat, "component_bbox": comp_bbox,
            "inactive_components": sorted(INACTIVE & set(region_volumes)),
            "element_epsr_eff": c.element_epsr_eff(),
            "divider_ratio_nominal": c.divider_ratio_nominal(),
            "creepage_mm": c.creepage_distance() * 1000.0,
            "tap_z": z_tap, "mid_z": z_mid, "lc": LC, "parameters": asdict(c),
            "physical_groups": pgroups,
        }
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    import json
    out = Path(__file__).resolve().parents[2] / "results" / "raw" / "024_true3d_component_architecture" / "cvt3d_v3.msh"
    m = build_cvt_ddb123_3d_v3(out, n_sheds=6, lc={"shed": 0.02, "air": 0.3}, mid_coupling=True)
    print(json.dumps({k: v for k, v in m.items() if k in
          ("n_nodes", "n_tets", "creepage_mm", "quality_minSICN", "material_volumes_m3",
           "inactive_components")}, indent=2, default=str))
    print("groups:", list(m["physical_groups"]))

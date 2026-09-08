"""Task 024 - component-accurate TRUE 3D DDB-123 with 74 ALTERNATING sheds.

Corrects the Task 023 model toward the real Arteche DDB-123 anatomy (reference photo +
catalogue): ~74 alternating large/small weather sheds (image-derived), a primary terminal,
an oil-volume compensating HEAD (metal shell + internal oil), the porcelain insulator, the
homogenized C1/tap/C2 capacitor column inside oil, a hollow grounded TANK housing a
simplified electromagnetic unit (inductive VT + series reactor + auxiliary/ferroresonance),
a resin high-frequency-terminal exit separating capacitive/inductive parts, a secondary
terminal box and simplified oil valve / level fittings.

Numerical targets: official Arteche DDB/DFK catalogue (5600 pF standard, creepage 3075 mm,
A 450 mm, H 1830 mm). Shed count 74 alternating is IMAGE-DERIVED (not tabulated); overhangs
(large 20.0 mm, small 8.7 mm) are tuned so the analytic creepage of 74 sheds ~ 3075 mm.

Electrical core unchanged/validated: homogenized C1/tap/C2 column (eps_r_eff, tap at n_c2/n).
Base components sit in/near the grounded tank (electrically inert for the divider) and are
tagged inactive/simplified. Built from OCC primitives + revolved fused porcelain; conformal
fragment classified by fragment MAP + priority. NO new physics, NO localized faults.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gmsh

from .cvt_ddb123 import CVTParams
from .cvt_ddb123_3d_v2 import _face_xz

# --- image-derived shed pattern (74 alternating) + tuned overhangs for 3075 mm creepage ---
N_SHEDS = 74
OV_LARGE = 0.0200      # m, large-shed overhang beyond the wall
OV_SMALL = 0.0087      # m, small-shed (intermediate rib) overhang
SHED_TH_ROOT = 0.006   # m
SHED_TH_TIP = 0.0025   # m
SHED_DROOP = 0.004     # m
SHED_LIP = 0.0015      # m, rounded lip radius

MAT_OF_V4 = {
    "air": "air", "oil": "oil", "head_oil": "oil", "base_oil": "oil",
    "porcelain": "porcelain", "element_c1": "element", "element_c2": "element",
    "resin": "resin",
    "terminal": "metal", "terminal_cap": "metal", "head_metal": "metal",
    "stack_top": "metal", "stack_bottom": "metal", "foil_tap": "metal",
    "tank_metal": "metal", "base_emu": "metal", "reactor": "metal",
    "aux_component": "metal", "secondary_box": "metal", "oil_valve": "metal",
    "oil_level": "metal",
}
HV_VOLS = {"terminal", "terminal_cap", "head_metal", "stack_top"}
GND_VOLS = {"tank_metal", "stack_bottom"}
INACTIVE = {"base_emu", "reactor", "aux_component", "secondary_box", "oil_valve",
            "oil_level", "resin"}
MATERIALS = ["air", "oil", "porcelain", "element", "metal", "resin"]


def _shed_data(c: CVTParams):
    h_ins = c.stack_z_hi - c.tank_height
    pitch = h_ins / N_SHEDS
    centers = [c.tank_height + (k + 0.5) * pitch for k in range(N_SHEDS)]
    overhangs = [OV_LARGE if k % 2 == 0 else OV_SMALL for k in range(N_SHEDS)]
    return centers, overhangs, pitch, h_ins


def _shed_polygon(c: CVTParams, zc: float, ov: float):
    r_root = c.porcelain_inner_radius + 0.5 * c.porcelain_thickness
    r_tip = c.porcelain_inner_radius + c.porcelain_thickness + ov
    return [(r_root, zc - 0.5 * SHED_TH_ROOT), (r_tip, zc - SHED_DROOP - 0.5 * SHED_TH_TIP),
            (r_tip, zc - SHED_DROOP + 0.5 * SHED_TH_TIP), (r_root, zc + 0.5 * SHED_TH_ROOT)]


def creepage_mm(c: CVTParams) -> float:
    centers, overhangs, pitch, h_ins = _shed_data(c)
    r_wall = c.porcelain_inner_radius + c.porcelain_thickness
    tot_per = tot_cov = 0.0
    for ov in overhangs:
        A, B, C, D = _shed_polygon(c, 0.0, ov)

        def we(pin, pout):
            t = (r_wall - pin[0]) / (pout[0] - pin[0])
            return (r_wall, pin[1] + t * (pout[1] - pin[1]))
        a = we(A, B); d = we(D, C)
        tot_per += (math.hypot(B[0] - a[0], B[1] - a[1]) + math.hypot(C[0] - B[0], C[1] - B[1])
                    + math.hypot(d[0] - C[0], d[1] - C[1]))
        tot_cov += d[1] - a[1]
    return ((h_ins - tot_cov) + tot_per) * 1000.0


def _specs(c: CVTParams):
    """(name, kind, params, prio). kind: cyl(r,z0,z1)|box(x,y,z,dx,dy,dz)|sphere(x,y,z,r)|porcelain."""
    h = c.element_height(); rs = c.stack_radius; disc = c.disc_thickness
    z0 = c.stack_z_lo
    n_c1 = c.n_elements - c.n_elements_c2; n_c2 = c.n_elements_c2
    z_sb = (z0, z0 + disc); z_c2 = (z_sb[1], z_sb[1] + n_c2 * h)
    z_tp = (z_c2[1], z_c2[1] + disc); z_c1 = (z_tp[1], z_tp[1] + n_c1 * h)
    z_st = (z_c1[1], z_c1[1] + disc)
    z_dome = (c.stack_z_hi, c.stack_z_hi + c.head_height)
    z_term = (z_dome[1], z_dome[1] + c.terminal_height)
    r_in = c.porcelain_inner_radius
    specs = [
        ("stack_bottom", "cyl", (rs, *z_sb), 0),
        ("element_c2",   "cyl", (rs, *z_c2), 1),
        ("foil_tap",     "cyl", (rs, *z_tp), 0),
        ("element_c1",   "cyl", (rs, *z_c1), 1),
        ("stack_top",    "cyl", (rs, *z_st), 0),
        ("head_oil",   "cyl", (0.14, z_dome[0] + 0.02, z_dome[1] - 0.01), 2),
        ("head_metal", "cyl", (c.head_radius, *z_dome), 3),
        ("terminal",     "cyl", (c.terminal_radius, z_term[0], z_term[1] - 0.006), 3),
        ("terminal_cap", "sphere", (0.0, 0.0, z_term[1] - 0.006, 0.022), 3),
        # base electromagnetic unit (simplified, inactive): IVT + reactor + auxiliary
        ("base_emu",      "cyl", (0.12, 0.10, 0.38), 2),
        ("reactor",       "cyl", (0.06, 0.40, 0.50), 2),
        ("aux_component", "cyl", (0.05, 0.05, 0.095), 2),
        ("base_oil",      "cyl", (0.20, 0.03, 0.52), 3),
        ("tank_metal",    "cyl", (c.tank_radius, 0.0, z0), 4),
        # resin HF-terminal exit (separates capacitive / inductive parts) at tank top
        ("resin", "cyl", (0.045, z0 - 0.03, z0), 1),
        # external secondary box + oil monitoring fittings on the tank side (+x).
        # inner x is set < tank-wall radius across the full y-width so each box PENETRATES
        # the tank cylinder (no floating gap); fittings are y-centred where the wall is widest.
        ("secondary_box", "box", (0.185, -0.075, 0.11, 0.155, 0.15, 0.18), 1),
        ("oil_valve", "box", (0.195, -0.03, 0.05, 0.075, 0.06, 0.05), 1),
        ("oil_level", "box", (0.195, -0.025, 0.34, 0.075, 0.05, 0.12), 1),
        ("oil", "cyl", (r_in, z0, c.stack_z_hi), 5),
        ("porcelain", "porcelain", None, 6),
    ]
    return specs, (z_tp[0] + 0.5 * disc), z_c1, z_c2


def build_cvt_ddb123_3d_v4(out_msh: Path, params: CVTParams | None = None, *,
                           r_air: float = 0.9, z_pad: float = 0.35,
                           lc: dict | None = None, curvature: float = 6.0,
                           verbose: bool = False) -> dict[str, Any]:
    c = params or CVTParams()
    c.n_sheds = N_SHEDS
    out_msh = Path(out_msh); out_msh.parent.mkdir(parents=True, exist_ok=True)
    LC = {"foil_tap": 0.008, "stack_bottom": 0.010, "stack_top": 0.010,
          "element_c1": 0.020, "element_c2": 0.014, "porcelain": 0.014, "oil": 0.024,
          "head_metal": 0.030, "head_oil": 0.030, "terminal": 0.010, "terminal_cap": 0.006,
          "tank_metal": 0.040, "base_emu": 0.028, "reactor": 0.018, "aux_component": 0.016,
          "base_oil": 0.032, "secondary_box": 0.026, "oil_valve": 0.012, "oil_level": 0.012,
          "resin": 0.012, "shed": 0.010, "air": 0.230}
    if lc:
        LC.update(lc)
    specs, z_tap, z_c1, z_c2 = _specs(c)
    r_in = c.porcelain_inner_radius; r_out = r_in + c.porcelain_thickness
    centers, overhangs, pitch, h_ins = _shed_data(c)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add("cvt_ddb123_3d_v4")
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
                x, y, z, r = p
                add(name, occ.addSphere(x, y, z, r), prio)
            elif kind == "porcelain":
                wall = _face_xz(occ, [(r_in, c.stack_z_lo), (r_out, c.stack_z_lo),
                                      (r_out, c.stack_z_hi), (r_in, c.stack_z_hi)],
                                radii=[0.0, SHED_LIP, SHED_LIP, 0.0])
                sheds = [_face_xz(occ, _shed_polygon(c, zc, ov),
                                  radii=[0.0, SHED_LIP, SHED_LIP, 0.0])
                         for zc, ov in zip(centers, overhangs)]
                occ.synchronize()
                fused, _ = occ.fuse([(2, wall)], [(2, s) for s in sheds])
                rev = occ.revolve(fused, 0, 0, 0, 0, 0, 1, 2.0 * math.pi)
                for (d, t) in rev:
                    if d == 3:
                        add(name, t, prio)

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
        for name, tags in region_volumes.items():
            pg = gmsh.model.addPhysicalGroup(3, sorted(set(tags)))
            gmsh.model.setPhysicalName(3, pg, name)

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

        vol_by_mat: dict[str, float] = {}
        comp_bbox: dict[str, list] = {}
        for name, tags in region_volumes.items():
            mat = MAT_OF_V4.get(name, "air")
            for t in tags:
                vol_by_mat[mat] = vol_by_mat.get(mat, 0.0) + occ.getMass(3, t)
                bb = occ.getBoundingBox(3, t)
                if name in comp_bbox:
                    o = comp_bbox[name]
                    comp_bbox[name] = [min(o[i], bb[i]) for i in range(3)] + \
                                      [max(o[i + 3], bb[i + 3]) for i in range(3)]
                else:
                    comp_bbox[name] = list(bb)

        def _lc(name):
            return LC.get(name, LC["air"])
        # do NOT over-refine the ~1.5 mm lips into sub-mm slivers (caused UMFPack to fail);
        # keep the floor near the lip radius so the thin sheds mesh cleanly.
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0020)
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
            mat = MAT_OF_V4.get(name, "air")
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
            "geometry": "cvt_ddb123_true3d_v4_74sheds", "output_msh": str(out_msh),
            "n_sheds": N_SHEDS, "shed_overhang_large_mm": OV_LARGE * 1000,
            "shed_overhang_small_mm": OV_SMALL * 1000, "shed_pitch_mm": pitch * 1000,
            "shed_lip_mm": SHED_LIP * 1000,
            "n_nodes": len(ntags), "n_tets": n_tet, "n_boundary_tris": n_tri,
            "quality_minSICN": {"min": float(q.min()), "mean": float(q.mean()), **pcts},
            "n_slivers_lt_0.05": int((q < 0.05).sum()), "worst20": worst,
            "insulator_surface_tri_quality": ins_q,
            "material_volumes_m3": vol_by_mat, "component_bbox": comp_bbox,
            "inactive_components": sorted(INACTIVE & set(region_volumes)),
            "element_epsr_eff": c.element_epsr_eff(),
            "divider_ratio_nominal": c.divider_ratio_nominal(),
            "creepage_mm": creepage_mm(c),
            "tap_z": z_tap, "c1_z": list(z_c1), "c2_z": list(z_c2),
            "lc": LC, "parameters": asdict(c), "physical_groups": pgroups,
        }
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    import json
    out = Path(__file__).resolve().parents[2] / "results" / "raw" / "024_true3d_ddb123_component_architecture" / "cvt3d_v4.msh"
    m = build_cvt_ddb123_3d_v4(out, lc={"shed": 0.02, "air": 0.32}, curvature=5.0)
    print(json.dumps({k: v for k, v in m.items() if k in
          ("n_nodes", "n_tets", "creepage_mm", "n_sheds", "quality_minSICN",
           "material_volumes_m3", "inactive_components")}, indent=2, default=str))
    print("groups:", list(m["physical_groups"]))

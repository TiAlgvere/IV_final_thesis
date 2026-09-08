"""Task 024 forensic - PROVE material/geometry status numerically (no guessing).

On the audit mesh (cvt3d_v4_audit.msh):
  1. material volume by tag inside head / tank / secondary-box bboxes; centroid + interface
     check for any unexpected material; also the head/tank CLOSE-UP windows to show where
     the porcelain-coloured pixels actually come from.
  4. floating-feature check: shared-node contact + min distance of each side feature to its
     parent (conformal mesh -> touching components share nodes).
  5. stack-metal z-ranges (tap vs electrode terminals).
  6. worst-50 tets by a Python quality metric, with material + location; insulator_surface.

Writes diagnose.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT.parent), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from publication_pipeline.geometry.gmsh.cvt_ddb123_3d_v4 import MAT_OF_V4  # noqa: E402

PROC = ROOT / "results" / "processed" / "024_true3d_ddb123_component_architecture"
RAW = ROOT / "results" / "raw" / "024_true3d_ddb123_component_architecture"
AUDIT_MSH = RAW / "cvt3d_v4_audit.msh"


def load():
    m = meshio.read(str(AUDIT_MSH))
    pts = np.asarray(m.points, float)
    name3 = {(int(dim), int(tag)): nm for nm, (tag, dim) in m.field_data.items()}
    tets, phys = [], []
    tri_nodes_by_name: dict[str, list] = {}
    for i, cb in enumerate(m.cells):
        ph = np.asarray(m.cell_data["gmsh:physical"][i], int).reshape(-1)
        if cb.type in ("tetra", "tetra10"):
            tets.append(np.asarray(cb.data, int)[:, :4]); phys.append(ph)
        elif cb.type in ("triangle", "triangle6"):
            data = np.asarray(cb.data, int)[:, :3]
            for tag in np.unique(ph):
                nm = name3.get((2, int(tag)))
                if nm:
                    tri_nodes_by_name.setdefault(nm, []).append(data[ph == tag])
    tets = np.vstack(tets); phys = np.concatenate(phys)
    names = np.array([name3.get((3, int(t)), "air") for t in phys])
    mats = np.array([MAT_OF_V4.get(n, "air") for n in names])
    surf_nodes = {k: np.unique(np.vstack(v)) for k, v in tri_nodes_by_name.items()}
    return pts, tets, names, mats, surf_nodes


def tet_vol_centroid(pts, tets):
    p = pts[tets]
    c = p.mean(1)
    p0 = p[:, 0]
    M = np.stack([p[:, 1] - p0, p[:, 2] - p0, p[:, 3] - p0], axis=1)
    V = np.abs(np.linalg.det(M)) / 6.0
    return V, c


def main() -> int:
    meta = json.loads((PROC / "audit_mesh_meta.json").read_text(encoding="utf-8"))
    bbox = meta["component_bbox"]
    pts, tets, names, mats, surf_nodes = load()
    V, C = tet_vol_centroid(pts, tets)
    r = np.hypot(C[:, 0], C[:, 1]); z = C[:, 2]
    out = {}

    # ---- 1. material by tag inside head / tank / box bboxes ----
    def in_box(bb, pad=0.0):
        return ((C[:, 0] >= bb[0] - pad) & (C[:, 0] <= bb[3] + pad) &
                (C[:, 1] >= bb[1] - pad) & (C[:, 1] <= bb[4] + pad) &
                (C[:, 2] >= bb[2] - pad) & (C[:, 2] <= bb[5] + pad))

    def region_report(bb, expected):
        m = in_box(bb)
        rep = {"bbox": [round(x, 4) for x in bb], "volumes_by_material": {}, "unexpected": []}
        for mat in sorted(set(mats[m])):
            vol = float(V[m & (mats == mat)].sum())
            rep["volumes_by_material"][mat] = round(vol, 8)
            if mat not in expected and vol > 0:
                sel = m & (mats == mat)
                cen = C[sel].mean(0)
                rep["unexpected"].append({
                    "material": mat, "volume_m3": round(vol, 8),
                    "centroid_rz": [round(float(np.hypot(cen[0], cen[1])), 4), round(float(cen[2]), 4)],
                    "frac": round(vol / float(V[m].sum()), 4),
                    "r_range": [round(float(r[sel].min()), 4), round(float(r[sel].max()), 4)]})
        return rep

    out["head_bbox"] = region_report(bbox["head_metal"], {"metal", "oil", "air"})
    out["tank_bbox"] = region_report(bbox["tank_metal"], {"metal", "oil", "resin", "air"})
    out["secondary_box_bbox"] = region_report(bbox["secondary_box"], {"metal", "air"})
    out["porcelain_stack_bbox"] = region_report(bbox["porcelain"],
                                                {"porcelain", "oil", "element", "metal", "air"})
    out["tap_bbox"] = region_report(bbox["foil_tap"], {"metal", "element", "oil"})
    out["emu_bbox"] = region_report(bbox["base_emu"], {"metal", "oil"})
    out["resin_bbox"] = region_report(bbox["resin"], {"resin", "metal", "oil", "element"})

    # head / tank CLOSE-UP windows (what the closeup figures actually showed)
    def window_porc(zlo, zhi, label):
        m = (z >= zlo) & (z <= zhi) & (mats == "porcelain")
        if not m.any():
            return {"window_z": [zlo, zhi], "porcelain_volume": 0.0}
        return {"window_z": [zlo, zhi], "porcelain_volume": round(float(V[m].sum()), 8),
                "porcelain_z_range": [round(float(z[m].min()), 4), round(float(z[m].max()), 4)],
                "note": "porcelain in this window is the adjacent insulator column, see z-range"}
    out["head_closeup_window"] = window_porc(1.55, 1.90, "head closeup z>=1.55")
    out["tank_closeup_window"] = window_porc(-0.35, 0.58, "tank closeup z<=0.58")

    # ---- 4. floating-feature check (conformal mesh: touching => shared nodes) ----
    def nodeset(nm):
        return set(np.unique(tets[names == nm]).tolist())
    feats = {"secondary_box": ["tank_metal"], "oil_valve": ["tank_metal"],
             "oil_level": ["tank_metal"],
             "resin": ["tank_metal", "stack_bottom", "base_oil", "element_c2"],
             "terminal": ["head_metal"], "terminal_cap": ["terminal", "head_metal"],
             "base_emu": ["base_oil"], "reactor": ["base_oil"]}
    out["floating_check"] = []
    for feat, parents in feats.items():
        fn = nodeset(feat)
        if not fn:
            out["floating_check"].append({"component": feat, "status": "ABSENT"}); continue
        touch = {}
        for par in parents:
            pn = nodeset(par)
            touch[par] = len(fn & pn)
        any_touch = any(v > 0 for v in touch.values())
        mind = 0.0
        if not any_touch:
            par_nodes = np.array(sorted(set().union(*[nodeset(p) for p in parents])))
            tree = cKDTree(pts[par_nodes]); d, _ = tree.query(pts[np.array(sorted(fn))])
            mind = float(d.min())
        out["floating_check"].append({
            "component": feat, "expected_parents": parents, "shared_nodes": touch,
            "touches_parent": bool(any_touch), "min_distance_m": round(mind, 5),
            "status": "ATTACHED" if any_touch else "FLOATING"})

    # ---- 5. stack metal z-ranges ----
    out["stack_metal"] = {}
    for nm in ("stack_bottom", "element_c2", "foil_tap", "element_c1", "stack_top", "resin"):
        sel = names == nm
        if sel.any():
            out["stack_metal"][nm] = {
                "material": MAT_OF_V4.get(nm), "z_range": [round(float(z[sel].min()), 4),
                round(float(z[sel].max()), 4)], "r_max": round(float(r[sel].max()), 4),
                "volume_m3": round(float(V[sel].sum()), 8)}
    out["stack_metal_note"] = ("foil_tap = intermediate voltage tap (between C1 and C2); "
                               "stack_top = HV electrode terminal; stack_bottom = ground electrode "
                               "terminal. All are intentional thin electrodes at the C1/C2 boundaries.")

    # ---- 6. worst-50 tets (python normalized quality eta = 12*(3V)^(2/3)/sum(edge^2)) ----
    p = pts[tets]
    e2 = np.zeros(len(tets))
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        e2 += ((p[:, a] - p[:, b]) ** 2).sum(1)
    eta = 12.0 * (3.0 * np.clip(V, 1e-30, None)) ** (2.0 / 3.0) / np.clip(e2, 1e-30, None)
    order = np.argsort(eta)[:100]
    # distance of each worst tet to the insulator surface and to the HV conductor
    ins_tree = cKDTree(pts[surf_nodes["insulator_surface"]]) if "insulator_surface" in surf_nodes else None
    hv_tree = cKDTree(pts[surf_nodes["hv_electrode"]]) if "hv_electrode" in surf_nodes else None
    out["worst100"] = []
    for i in order:
        d_ins = float(ins_tree.query(C[i])[0]) if ins_tree else None
        d_hv = float(hv_tree.query(C[i])[0]) if hv_tree else None
        out["worst100"].append({"eta": round(float(eta[i]), 4), "material": str(mats[i]),
                                "name": str(names[i]), "rz": [round(float(r[i]), 3), round(float(z[i]), 3)],
                                "dist_insulator_m": round(d_ins, 4) if d_ins is not None else None,
                                "dist_HV_m": round(d_hv, 4) if d_hv is not None else None})
    import collections
    out["worst100_by_material"] = dict(collections.Counter(mats[order].tolist()))
    out["worst100_by_name"] = dict(collections.Counter(names[order].tolist()))
    on_ins = sum(1 for w in out["worst100"] if w["dist_insulator_m"] is not None and w["dist_insulator_m"] < 0.005)
    near_hv = sum(1 for w in out["worst100"] if w["dist_HV_m"] is not None and w["dist_HV_m"] < 0.05)
    out["worst100_on_insulator_surface_lt5mm"] = on_ins
    out["worst100_near_HV_lt50mm"] = near_hv
    out["insulator_surface_tri_quality"] = meta["insulator_surface_tri_quality"]
    out["global_quality_minSICN"] = meta["quality_minSICN"]
    out["n_slivers_lt_0.05"] = meta["n_slivers_lt_0.05"]

    (PROC / "diagnose.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # console summary
    print("=== 1. MATERIAL IN HEAD/TANK/BOX BBOX ===")
    for k in ("head_bbox", "tank_bbox", "secondary_box_bbox"):
        rr = out[k]
        print(f"{k}: {rr['volumes_by_material']}  unexpected={rr['unexpected']}")
    print("head closeup window porcelain:", out["head_closeup_window"])
    print("tank closeup window porcelain:", out["tank_closeup_window"])
    print("\n=== 4. FLOATING CHECK ===")
    for f in out["floating_check"]:
        print(f)
    print("\n=== 5. STACK METAL ===")
    for k, v in out["stack_metal"].items():
        print(f"  {k}: {v}")
    print("\n=== 6. WORST-100 ===")
    print("by material:", out["worst100_by_material"], " by name:", out["worst100_by_name"])
    print("worst5:", out["worst100"][:5])
    print(f"worst100 ON insulator surface (<5mm): {out['worst100_on_insulator_surface_lt5mm']}; "
          f"near HV (<50mm): {out['worst100_near_HV_lt50mm']}")
    print("insulator_surface:", out["insulator_surface_tri_quality"])
    print(f"\nWrote {PROC / 'diagnose.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task 022 - TRUE 3D DDB-123 geometry (full 360 deg), homogenized stack.

First true-3D foundation for the publication pipeline. Built from gmsh OCC primitive
cylinders (a full revolution needs no periodic/symmetry BCs and no revolve seam), then
fragmented into a conformal partition. Each output volume is assigned to a region via the
OCC fragment MAP (which input cylinder each piece came from) with a priority rule for
overlaps - robust for the non-convex pieces (the air shell, the annular oil tube) where a
centroid test fails.

Homogenization (permitted by Task 022): the 12-element series stack is collapsed to a
single dielectric column split by the intermediate tap into C1 (HV side) and C2 (ground
side). With eps_r_eff (the validated 2D value) and the tap at the n_c2/n height, this
reproduces BOTH the terminal capacitance (~5600 pF) and the divider ratio (1/6):
  C_total = eps_r_eff*eps0*A / (10*h + 2*h),  Vtap/U0 = C1/(C1+C2) = 2/12.

Coarse foundation scope: plain porcelain cylinder (weather sheds deferred - they do not
affect the clean-baseline C/ratio/phase; they are the documented next refinement, needed
for creepage + surface-conductance). Air is a finite cylinder with a far-field phi=0
boundary; the internal divider dominates C so the modest air box is adequate.

Volume groups: stack_bottom, element_c2, foil_tap, element_c1, stack_top, head_housing,
base_tank, porcelain, oil, air.  Surface groups: hv_electrode, ground_electrode,
insulator_surface, farfield.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import gmsh

from .cvt_ddb123 import CVTParams  # reuse the validated parameter set / eps_r_eff


def _regions(c: CVTParams):
    """(name, R, z0, z1, prio, is_hv, is_gnd); lower prio wins on overlap. + tap z."""
    h = c.element_height()
    rs = c.stack_radius
    disc = c.disc_thickness
    z0 = c.stack_z_lo
    n_c1 = c.n_elements - c.n_elements_c2
    n_c2 = c.n_elements_c2
    z_sb = (z0, z0 + disc)
    z_c2 = (z_sb[1], z_sb[1] + n_c2 * h)
    z_tp = (z_c2[1], z_c2[1] + disc)
    z_c1 = (z_tp[1], z_tp[1] + n_c1 * h)
    z_st = (z_c1[1], z_c1[1] + disc)
    z_dome = (c.stack_z_hi, c.stack_z_hi + c.head_height)
    z_term = (z_dome[1], z_dome[1] + c.terminal_height)
    r_in = c.porcelain_inner_radius
    r_out = r_in + c.porcelain_thickness
    regs = [
        ("stack_bottom", rs, *z_sb, 0, False, True),
        ("element_c2",   rs, *z_c2, 1, False, False),
        ("foil_tap",     rs, *z_tp, 2, False, False),
        ("element_c1",   rs, *z_c1, 3, False, False),
        ("stack_top",    rs, *z_st, 4, True, False),
        ("head_housing", c.head_radius, *z_dome, 5, True, False),
        ("head_housing", c.terminal_radius, *z_term, 6, True, False),
        ("base_tank",    c.tank_radius, 0.0, z0, 7, False, True),
        ("oil",          r_in, z0, c.stack_z_hi, 8, False, False),
        ("porcelain",    r_out, z0, c.stack_z_hi, 9, False, False),
    ]
    return regs, (z_tp[0] + 0.5 * disc)


def build_cvt_ddb123_3d(out_msh: Path, params: CVTParams | None = None, *,
                        r_air: float = 0.8, z_pad: float = 0.30,
                        lc: dict | None = None, verbose: bool = False) -> dict[str, Any]:
    c = params or CVTParams()
    out_msh = Path(out_msh); out_msh.parent.mkdir(parents=True, exist_ok=True)
    LC = {"foil_tap": 0.010, "stack_bottom": 0.010, "stack_top": 0.010,
          "element_c1": 0.030, "element_c2": 0.022, "porcelain": 0.030, "oil": 0.040,
          "head_housing": 0.050, "base_tank": 0.060, "air": 0.220}
    if lc:
        LC.update(lc)

    regs, z_tap = _regions(c)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add("cvt_ddb123_3d")
        occ = gmsh.model.occ

        cyls = [occ.addCylinder(0, 0, z0, 0, 0, z1 - z0, R)
                for (name, R, z0, z1, prio, hv, gnd) in regs]
        air = occ.addCylinder(0, 0, -z_pad, 0, 0, c.total_height + 2 * z_pad, r_air)
        inputs = [(3, t) for t in cyls] + [(3, air)]
        prio_in = [r[4] for r in regs] + [100]
        name_in = [r[0] for r in regs] + ["air"]

        out, fmap = occ.fragment(inputs, [])
        occ.synchronize()

        # assign each output volume to the lowest-priority input it came from
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

        # surface -> adjacent volume-name set
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

        def _sgroup(name, tags):
            if tags:
                pg = gmsh.model.addPhysicalGroup(2, sorted(set(tags)))
                gmsh.model.setPhysicalName(2, pg, name)

        _sgroup("hv_electrode", hv_s)
        _sgroup("ground_electrode", gnd_s)
        _sgroup("insulator_surface", ins_s)
        _sgroup("farfield", ff_s)

        # graded mesh sizing (coarse-first so finer size wins on shared points)
        def _lc(name):
            return LC.get(name, LC["air"])

        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005)
        gmsh.option.setNumber("Mesh.MeshSizeMax", LC["air"])
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        for name, tags in sorted(region_volumes.items(), key=lambda kv: -_lc(kv[0])):
            pts = set()
            for t in tags:
                for (bd, bt) in gmsh.model.getBoundary([(3, t)], recursive=True, oriented=False):
                    if bd == 0:
                        pts.add(abs(bt))
            if pts:
                gmsh.model.mesh.setSize([(0, p) for p in pts], _lc(name))

        gmsh.model.mesh.generate(3)
        try:
            gmsh.model.mesh.optimize("Netgen")
        except Exception:
            pass
        gmsh.write(str(out_msh))

        ntags, _, _ = gmsh.model.mesh.getNodes()
        etypes, etags, _ = gmsh.model.mesh.getElements(3)
        n_tet = int(sum(len(t) for t in etags))
        qmin = qmean = None
        try:
            tet_tags = [t for tt in etags for t in tt]
            q = gmsh.model.mesh.getElementQualities(tet_tags, "minSICN")
            qmin, qmean = float(min(q)), float(sum(q) / len(q))
        except Exception:
            pass

        pgroups = {gmsh.model.getPhysicalName(d, t): {"dim": d,
                   "n": len(gmsh.model.getEntitiesForPhysicalGroup(d, t))}
                   for d, t in gmsh.model.getPhysicalGroups()}
        return {
            "geometry": "cvt_ddb123_true3d_homogenized", "output_msh": str(out_msh),
            "n_nodes": len(ntags), "n_tets": n_tet,
            "min_quality_minSICN": qmin, "mean_quality_minSICN": qmean,
            "element_epsr_eff": c.element_epsr_eff(),
            "divider_ratio_nominal": c.divider_ratio_nominal(),
            "tap_z": z_tap, "r_air": r_air,
            "lc": LC, "parameters": asdict(c), "physical_groups": pgroups,
        }
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    import json
    out = Path(__file__).resolve().parents[2] / "results" / "raw" / "022_true3d_foundation" / "cvt3d.msh"
    m = build_cvt_ddb123_3d(out, verbose=False)
    print(json.dumps({k: v for k, v in m.items() if k not in ("parameters", "lc")}, indent=2, default=str))

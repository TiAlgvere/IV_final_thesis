"""Task 026 - azimuthal surface-conductance sector validation (MATC theta-window method).

DECISION (documented): the explicit-physical-group geometry split (revolving the 74-shed
porcelain into 5 angular wedges) proved computationally intractable - the OCC fragment of
the non-convex shed wedges exceeded ~20 min / ~3.8 GB without completing. The verified,
tractable, and auditable alternative is the MATC theta-WINDOW on the single insulator_surface
(the same windowing the 2D localized-pollution study used). Auditability is preserved: the
polluted-sector area is measured by theta-binning the insulator triangles and compared to the
requested angle fraction.

Reuses the ACCEPTED Task 025 v5 solve mesh (single insulator_surface, remediated, p5~0.49) -
read-only, not overwritten. Cases: clean ; 30/60/90/180/360 deg x sigma_s {1e-7, 1e-6}.
Surface Conductance = sigma_s on theta in [0, angle] (uniform for 360).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT.parent), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gmsh  # noqa: E402
import meshio  # noqa: E402

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123_3d_v4 import MAT_OF_V4  # noqa: E402
from run_cvt import OMEGA, U0, FREQ, build_material_table, ensure_solver, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402
from run_true3d import observables_3d  # noqa: E402

PROC = ROOT / "results" / "processed" / "026_azimuthal_surface_conductance_validation"
RAW = ROOT / "results" / "raw" / "026_azimuthal_surface_conductance_validation"
SRC_MSH = ROOT / "results" / "raw" / "025_insulator_mesh_remediation" / "cvt3d_v5_solve.msh"
SIGMA_LIST = [1e-7, 1e-6]
ANGLES = [30, 60, 90, 180, 360]
SECTOR_BOUNDS = [0, 30, 60, 90, 180, 360]
INS_RZ = (0.11, 0.17, 0.557, 1.605)   # creepage triangle filter (r_lo,r_hi,z_lo,z_hi)


def mesh_audit(msh):
    """gmsh minSICN (vectorised): global tet + insulator-surface triangles binned by
    theta-sector + patch-edge band; worst-100 tets with material/theta."""
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(str(msh))
        name3 = {(d, t): gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()}
        nt, nc, _ = gmsh.model.mesh.getNodes()
        nt = nt.astype(np.int64); nc = nc.reshape(-1, 3)
        coord = np.zeros((int(nt.max()) + 1, 3)); coord[nt] = nc

        etypes, etags, enodes = gmsh.model.mesh.getElements(3)
        tets = etags[0].astype(np.int64)
        conn = enodes[0].reshape(-1, 4).astype(np.int64)
        q = np.array(gmsh.model.mesh.getElementQualities(list(tets), "minSICN"))
        glob = {"min": float(q.min()), "mean": float(q.mean()),
                **{str(p): float(np.percentile(q, p)) for p in (1, 5, 50)},
                "n_slivers_lt_0.05": int((q < 0.05).sum()), "n_tets": int(tets.size)}

        ins_tag = next((t for (d, t), nm in name3.items() if nm == "insulator_surface" and d == 2), None)
        sec_stats, edge, ins_all = {}, None, None
        if ins_tag is not None:
            tri_tags, tri_conn = [], []
            for ent in gmsh.model.getEntitiesForPhysicalGroup(2, ins_tag):
                ets, en = gmsh.model.mesh.getElements(2, ent)[1:]
                if ets:
                    tri_tags.append(ets[0].astype(np.int64)); tri_conn.append(en[0].reshape(-1, 3).astype(np.int64))
            tri_tags = np.concatenate(tri_tags); tri_conn = np.vstack(tri_conn)
            tq = np.array(gmsh.model.mesh.getElementQualities(list(tri_tags), "minSICN"))
            P = coord[tri_conn]; cen = P.mean(1)
            th = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])) % 360.0
            ar = 0.5 * np.linalg.norm(np.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0]), axis=1)
            ins_all = {"n": int(tri_tags.size), "min": float(tq.min()), "mean": float(tq.mean()),
                       "p5": float(np.percentile(tq, 5)), "area_m2": float(ar.sum())}
            for i in range(len(SECTOR_BOUNDS)-1):
                a, b = SECTOR_BOUNDS[i], SECTOR_BOUNDS[i+1]; m = (th >= a) & (th < b)
                sec_stats[f"s{i}_{a}-{b}deg"] = {"n": int(m.sum()),
                    "min": float(tq[m].min()) if m.any() else None,
                    "p5": float(np.percentile(tq[m], 5)) if m.any() else None,
                    "area_m2": float(ar[m].sum())}
            dmin = np.min(np.abs(th[:, None] - np.array([30, 60, 90, 180])[None, :]), axis=1)
            em = dmin <= 3.0
            edge = {"n": int(em.sum()), "min": float(tq[em].min()) if em.any() else None,
                    "p5": float(np.percentile(tq[em], 5)) if em.any() else None,
                    "mean": float(tq[em].mean()) if em.any() else None}

        id2name = {t: nm for (d, t), nm in name3.items() if d == 3}
        ent_tet = {}
        for d, t in gmsh.model.getPhysicalGroups(3):
            for ent in gmsh.model.getEntitiesForPhysicalGroup(3, t):
                ets = gmsh.model.mesh.getElements(3, ent)[1]
                for tt in (ets[0] if ets else []):
                    ent_tet[int(tt)] = id2name.get(t, "?")
        order = np.argsort(q)[:100]
        tcen = coord[conn].mean(1)
        worst = []
        for idx in order:
            tt = int(tets[idx]); c = tcen[idx]; nm = ent_tet.get(tt, "?")
            worst.append({"q": float(q[idx]), "name": nm,
                          "mat": ("porcelain" if nm.startswith("porcelain") else MAT_OF_V4.get(nm, "?")),
                          "rz": [float(math.hypot(c[0], c[1])), float(c[2])],
                          "theta_deg": float(math.degrees(math.atan2(c[1], c[0])) % 360)})
        return {"global_tet": glob, "insulator_surface": ins_all, "sector_tri": sec_stats,
                "patch_edge_tri": edge, "worst100": worst}
    finally:
        gmsh.finalize()


def render_sif(mat, mat_index, bm, bodies, hv, gnd, ff, ins, *, sigma_s=0.0, angle=0, direct=True):
    lin = ("  Linear System Solver = Direct\n  Linear System Direct Method = UMFPack\n" if direct
           else "  Linear System Solver = Iterative\n  Linear System Iterative Method = BiCGStabl\n"
                "  BiCGStabl polynomial degree = 4\n  Linear System Preconditioning = ILU2\n"
                "  Linear System Max Iterations = 8000\n  Linear System Residual Output = 100\n")
    parts = [
        'Header\n  CHECK KEYWORDS Warn\n  Mesh DB "." "mesh"\nEnd\n',
        'Simulation\n  Coordinate System = "Cartesian"\n  Simulation Type = Steady State\n'
        f'  Steady State Max Iterations = 1\n  Output Intervals = 1\n  Frequency = {FREQ}\n'
        '  Output File = "case.result"\n  Post File = "case.vtu"\nEnd\n',
        "Constants\n  Permittivity Of Vacuum = 8.8541878128e-12\nEnd\n",
        'Equation 1\n  Name = "ComplexEQS"\n  Active Solvers(1) = 1\nEnd\n',
        'Solver 1\n  Equation = "ComplexEQS"\n  Procedure = "ComplexEQS" "ComplexEQSSolver"\n'
        "  Variable = Potential[Potential Re:1 Potential Im:1]\n  Linear System Complex = True\n"
        f"{lin}  Linear System Convergence Tolerance = 1.0e-9\n"
        "  Steady State Convergence Tolerance = 1.0e-9\nEnd\n",
    ]
    for mn, idx in mat_index.items():
        eps, tand, sig = mat[mn]
        parts.append(f"Material {idx}\n  Name = \"{mn}\"\n  Relative Permittivity = {eps!r}\n"
                     f"  Electric Conductivity = {sigma_eff(eps, tand, sig)!r}\nEnd\n")
    for bid, name in bodies:
        parts.append(f"Body {bid}\n  Name = \"{name}\"\n  Equation = 1\n  Material = {mat_index[bm[name]]}\nEnd\n")
    parts.append(f"Boundary Condition 1\n  Name = \"hv_electrode\"\n  Target Boundaries(1) = {hv}\n"
                 f"  Potential Re = Real {U0!r}\n  Potential Im = Real 0.0\nEnd\n")
    parts.append(f"Boundary Condition 2\n  Name = \"ground_electrode\"\n  Target Boundaries(1) = {gnd}\n"
                 "  Potential Re = Real 0.0\n  Potential Im = Real 0.0\nEnd\n")
    if ff is not None:
        parts.append(f"Boundary Condition 3\n  Name = \"farfield\"\n  Target Boundaries(1) = {ff}\n"
                     "  Potential Re = Real 0.0\n  Potential Im = Real 0.0\nEnd\n")
    if sigma_s > 0.0 and ins is not None:
        if angle >= 360:
            sc = f"Real {float(sigma_s)!r}"
        else:
            ar = math.radians(angle)
            sc = ("Variable Coordinate\n"
                  f"    Real MATC \"{float(sigma_s)!r}*(atan2(tx(1),tx(0))>=0)*(atan2(tx(1),tx(0))<={ar!r})\"")
        parts.append(f"Boundary Condition 4\n  Name = \"polluted_sector\"\n  Target Boundaries(1) = {ins}\n"
                     f"  Surface Conductance = {sc}\nEnd\n")
    return "\n".join(parts)


def surface_loss(vtu, sigma_s, angle):
    m = meshio.read(str(vtu))
    pts = np.asarray(m.points, float)
    pd = {k.lower(): np.asarray(v, float).reshape(-1) for k, v in m.point_data.items()}
    re, im = pd["potential re"], pd["potential im"]
    tris = next((np.asarray(cb.data, int)[:, :3] for cb in m.cells if cb.type.startswith("triangle")), None)
    if tris is None:
        return None, None, None
    p = pts[tris]; cen = p.mean(1)
    rc = np.hypot(cen[:, 0], cen[:, 1]); zc = cen[:, 2]
    th = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])) % 360.0
    on = (rc >= INS_RZ[0]) & (rc <= INS_RZ[1]) & (zc >= INS_RZ[2]) & (zc <= INS_RZ[3])
    inwin = on & (th <= angle + 1e-6) if angle < 360 else on
    e1 = p[:, 1]-p[:, 0]; e2 = p[:, 2]-p[:, 0]
    nrm = np.cross(e1, e2); area = 0.5*np.linalg.norm(nrm, axis=1)
    nhat = nrm/np.linalg.norm(nrm, axis=1)[:, None]
    M = np.stack([e1, e2, nhat], 1); gsq = np.zeros(len(tris))
    for f in (re, im):
        b = np.stack([f[tris[:, 1]]-f[tris[:, 0]], f[tris[:, 2]]-f[tris[:, 0]], np.zeros(len(tris))], 1)[:, :, None]
        gsq += (np.linalg.solve(M, b)[:, :, 0]**2).sum(1)
    return (float(np.sum(sigma_s*gsq[inwin]*area[inwin])), float(area[inwin].sum()), float(area[on].sum()))


def parse_conv(out):
    it = re.findall(r"^\s*(\d+)\s+([0-9.Ee+-]+)\s*$", out, re.M)
    return {"iterations": int(it[-1][0]), "final_residual": float(it[-1][1])} if it else {"iterations": None, "final_residual": None}


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    print("Mesh audit (reusing accepted Task 025 v5 solve mesh) ...")
    aud = mesh_audit(SRC_MSH)
    (PROC / "mesh_meta.json").write_text(json.dumps(aud, indent=2, default=str), encoding="utf-8")
    print(f"  global tet min={aud['global_tet']['min']:.3f} slivers={aud['global_tet']['n_slivers_lt_0.05']}; "
          f"insulator p5={aud['insulator_surface']['p5']:.3f}; patch-edge p5={aud['patch_edge_tri']['p5']:.3f}")

    ep = RAW / "elmer"; ep.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(resolve_elmer_grid()), "14", "2", str(SRC_MSH), "-out", "mesh"],
                   cwd=str(ep), capture_output=True, text=True, check=True)
    em = ep / "mesh"; ids = _parse_mesh_names(em / "mesh.names")
    bodies = [(bid, n) for n, bid in ids["bodies"].items()]
    tap_id = ids["bodies"]["foil_tap"]
    hv, gnd = ids["boundaries"]["hv_electrode"], ids["boundaries"]["ground_electrode"]
    ff = ids["boundaries"].get("farfield"); ins = ids["boundaries"].get("insulator_surface")

    eps = CVTParams().element_epsr_eff()
    mat = build_material_table(eps); mat["resin"] = (4.5, 0.01, 0.0)
    mat_index = {m_: i + 1 for i, m_ in enumerate(mat)}
    bm = {n: ("porcelain" if n.startswith("porcelain") else MAT_OF_V4.get(n, "air")) for n in ids["bodies"]}
    be = {bid: mat[bm[n]][0] for n, bid in ids["bodies"].items()}
    bs = {bid: sigma_eff(*mat[bm[n]]) for n, bid in ids["bodies"].items()}

    pdir = RAW / "run3d"; pdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(em, pdir / "mesh", dirs_exist_ok=True); shutil.copy2(dll, pdir / dll.name)

    def solve(tag, sigma_s, angle):
        for direct in (True, False):
            sif = render_sif(mat, mat_index, bm, bodies, hv, gnd, ff, ins, sigma_s=sigma_s, angle=angle, direct=direct)
            (pdir / "case.sif").write_text(sif, encoding="utf-8")
            t0 = time.time()
            d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                               capture_output=True, text=True, env=env)
            if d.returncode == 0 and sorted(pdir.glob("**/case*.vtu")):
                return sorted(pdir.glob("**/case*.vtu"))[-1], ("direct" if direct else "iterative"), time.time()-t0, parse_conv(d.stdout)
        (pdir / f"FAIL_{tag}.txt").write_text(d.stdout[-4000:], encoding="utf-8"); raise RuntimeError(f"solve failed {tag}")

    rows = []

    def rec(tag, angle, ss, v, slv, dt, cv):
        o = observables_3d(v, be, bs, tap_id)
        ie = o["C_pF"]*1e-12*U0**2; isv = o["tan_delta"]*OMEGA*ie
        row = {"case": tag, "angle_deg": angle, "sigma_s": ss, "solver": slv, "solve_s": round(dt, 1), **cv,
               "C_pF": o["C_pF"], "vtap_abs": o["vtap_abs"], "phase_mdeg": o["vtap_phase_mdeg"],
               "divider_ratio": o["divider_ratio"], "emax": o["emax"], "emax_rz": o["emax_rz"],
               "volume_leak": isv/U0}
        if ss > 0 and angle:
            isf, parea, tarea = surface_loss(v, ss, angle)
            row.update({"tan_delta": (isv+isf)/(OMEGA*ie), "surface_leak": isf/U0,
                        "polluted_area_m2": parea, "total_insulator_area_m2": tarea,
                        "area_fraction": parea/tarea, "expected_fraction": min(angle, 360)/360.0})
        else:
            row.update({"tan_delta": o["tan_delta"], "surface_leak": 0.0})
        rows.append(row)
        extra = f" area_frac={row.get('area_fraction', 0):.3f}(exp {row.get('expected_fraction', 0):.3f})" if ss > 0 else ""
        print(f"  {tag}: C={o['C_pF']:.1f} ratio={o['divider_ratio']:.4f} tand={row['tan_delta']:.3e} "
              f"sleak={row['surface_leak']:.3e}{extra} ({slv},{dt:.0f}s)")

    print("clean ..."); v, s, dt, cv = solve("clean", 0.0, 0); rec("clean", 0, 0.0, v, s, dt, cv)
    for ss in SIGMA_LIST:
        for angle in ANGLES:
            tag = f"a{angle}_s{ss:.0e}"; print(f"{tag} ...")
            v, s, dt, cv = solve(tag, ss, angle); rec(tag, angle, ss, v, s, dt, cv)
            if abs(ss-1e-6) < 1e-12 and angle in (30, 90, 180, 360):
                shutil.copy2(v, RAW / "run3d" / f"vtu_a{angle}.vtu")

    (PROC / "validation.json").write_text(json.dumps({"mesh_audit": aud, "cases": rows}, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {PROC / 'validation.json'} ({len(rows)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

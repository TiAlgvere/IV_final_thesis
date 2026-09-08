"""Task 026B - rotational sanity check: a 30deg surface-conductance sector at three azimuths
([0-30], [90-120], [180-210]) at sigma_s=1e-6. If surface_leak and tan-delta are nearly
rotation-invariant, the localized-sector response is azimuth-independent (as expected for the
axisymmetric divider). Reuses the already-converted Task 026 Elmer mesh; no remesh.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT.parent), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meshio  # noqa: E402

from publication_pipeline._elmer_tools import resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123_3d_v4 import MAT_OF_V4  # noqa: E402
from run_cvt import OMEGA, U0, FREQ, build_material_table, ensure_solver, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402
from run_true3d import observables_3d  # noqa: E402

PROC = ROOT / "results" / "processed" / "026_azimuthal_surface_conductance_validation"
RAW = ROOT / "results" / "raw" / "026_azimuthal_surface_conductance_validation"
SIGMA = 1e-6
SECTORS = [(0, 30), (90, 120), (180, 210)]   # 30 deg sectors at three azimuths
INS = (0.11, 0.17, 0.557, 1.605)


def matc_window(a, b, ss):
    th = "(atan2(tx(1),tx(0))*180.0/pi + 360.0*(atan2(tx(1),tx(0))<0))"   # theta in [0,360)
    return (f"Variable Coordinate\n    Real MATC \"{float(ss)!r}*({th}>={float(a)!r})*({th}<={float(b)!r})\"")


def render_sif(mat, mat_index, bm, bodies, hv, gnd, ff, ins, a, b, ss):
    parts = [
        'Header\n  CHECK KEYWORDS Warn\n  Mesh DB "." "mesh"\nEnd\n',
        'Simulation\n  Coordinate System = "Cartesian"\n  Simulation Type = Steady State\n'
        f'  Steady State Max Iterations = 1\n  Output Intervals = 1\n  Frequency = {FREQ}\n'
        '  Output File = "case.result"\n  Post File = "case.vtu"\nEnd\n',
        "Constants\n  Permittivity Of Vacuum = 8.8541878128e-12\nEnd\n",
        'Equation 1\n  Name = "ComplexEQS"\n  Active Solvers(1) = 1\nEnd\n',
        'Solver 1\n  Equation = "ComplexEQS"\n  Procedure = "ComplexEQS" "ComplexEQSSolver"\n'
        "  Variable = Potential[Potential Re:1 Potential Im:1]\n  Linear System Complex = True\n"
        "  Linear System Solver = Iterative\n  Linear System Iterative Method = BiCGStabl\n"
        "  BiCGStabl polynomial degree = 4\n  Linear System Preconditioning = ILU2\n"
        "  Linear System Max Iterations = 8000\n  Linear System Residual Output = 200\n"
        "  Linear System Convergence Tolerance = 1.0e-9\n  Steady State Convergence Tolerance = 1.0e-9\nEnd\n",
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
    parts.append(f"Boundary Condition 4\n  Name = \"polluted_sector\"\n  Target Boundaries(1) = {ins}\n"
                 f"  Surface Conductance = {matc_window(a, b, ss)}\nEnd\n")
    return "\n".join(parts)


def surface_loss(vtu, ss, a, b):
    m = meshio.read(str(vtu)); pts = np.asarray(m.points, float)
    pd = {k.lower(): np.asarray(v, float).reshape(-1) for k, v in m.point_data.items()}
    re, im = pd["potential re"], pd["potential im"]
    tris = next((np.asarray(cb.data, int)[:, :3] for cb in m.cells if cb.type.startswith("triangle")), None)
    p = pts[tris]; cen = p.mean(1)
    rc = np.hypot(cen[:, 0], cen[:, 1]); zc = cen[:, 2]
    th = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])) % 360.0
    sel = (rc >= INS[0]) & (rc <= INS[1]) & (zc >= INS[2]) & (zc <= INS[3]) & (th >= a) & (th <= b)
    e1 = p[:, 1]-p[:, 0]; e2 = p[:, 2]-p[:, 0]
    nrm = np.cross(e1, e2); area = 0.5*np.linalg.norm(nrm, axis=1)
    nhat = nrm/np.linalg.norm(nrm, axis=1)[:, None]
    M = np.stack([e1, e2, nhat], 1); gsq = np.zeros(len(tris))
    for f in (re, im):
        bb = np.stack([f[tris[:, 1]]-f[tris[:, 0]], f[tris[:, 2]]-f[tris[:, 0]], np.zeros(len(tris))], 1)[:, :, None]
        gsq += (np.linalg.solve(M, bb)[:, :, 0]**2).sum(1)
    return float(np.sum(ss*gsq[sel]*area[sel])), float(area[sel].sum())


def main() -> int:
    dll = ensure_solver()
    env = os.environ.copy(); home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    pdir = RAW / "run3d"; em = RAW / "elmer" / "mesh"
    ids = _parse_mesh_names(em / "mesh.names")
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

    rows = []
    for (a, b) in SECTORS:
        sif = render_sif(mat, mat_index, bm, bodies, hv, gnd, ff, ins, a, b, SIGMA)
        (pdir / "case.sif").write_text(sif, encoding="utf-8")
        print(f"[{a}-{b}] solving ...")
        t0 = time.time()
        d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                           capture_output=True, text=True, env=env)
        if d.returncode != 0 or not sorted(pdir.glob("**/case*.vtu")):
            (pdir / f"FAIL_rot_{a}.txt").write_text(d.stdout[-3000:], encoding="utf-8")
            raise RuntimeError(f"rotation solve failed {a}-{b}")
        vtu = sorted(pdir.glob("**/case*.vtu"))[-1]
        o = observables_3d(vtu, be, bs, tap_id)
        ie = o["C_pF"]*1e-12*U0**2; isv = o["tan_delta"]*OMEGA*ie
        isf, parea = surface_loss(vtu, SIGMA, a, b)
        rows.append({"sector_deg": [a, b], "C_pF": o["C_pF"], "divider_ratio": o["divider_ratio"],
                     "tan_delta": (isv+isf)/(OMEGA*ie), "surface_leak": isf/U0,
                     "phase_mdeg": o["vtap_phase_mdeg"], "polluted_area_m2": parea,
                     "solve_s": round(time.time()-t0, 1)})
        print(f"  sleak={rows[-1]['surface_leak']:.4e} tand={rows[-1]['tan_delta']:.4e} "
              f"area={parea:.5f} ({rows[-1]['solve_s']}s)")

    base = rows[0]
    for r in rows:
        r["d_surface_leak_pct"] = (r["surface_leak"]/base["surface_leak"]-1)*100
        r["d_tan_delta_pct"] = (r["tan_delta"]/base["tan_delta"]-1)*100
    sl = [r["surface_leak"] for r in rows]; td = [r["tan_delta"] for r in rows]
    spread = {"surface_leak_spread_pct": (max(sl)/min(sl)-1)*100,
              "tan_delta_spread_pct": (max(td)/min(td)-1)*100}
    out = {"sigma_s": SIGMA, "sector_width_deg": 30, "cases": rows, "spread": spread,
           "rotation_invariant": spread["surface_leak_spread_pct"] < 5.0}
    (PROC / "rotation_check.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nspread: surface_leak {spread['surface_leak_spread_pct']:.1f}%  tand {spread['tan_delta_spread_pct']:.1f}%  "
          f"-> rotation-invariant={out['rotation_invariant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

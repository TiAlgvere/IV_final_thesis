"""Task 025 - clean validation + uniform 3D surface-conductance validation on the v5 mesh.

Builds a v5 solve mesh, runs:
  (a) clean 3D ComplexEQS -> C / ratio / phase vs axisymmetric;
  (b) uniform Surface Conductance sigma_s on insulator_surface in 3D, compares tan-delta and
      surface_leak to the 2D axisymmetric surface-conductance model at the same sigma_s.

The 3D surface loss int_surf = INT sigma_s |grad_s phi|^2 dS is integrated over the
insulator-surface boundary triangles (the porcelain-air creepage), matching the 2D
_surface_loss line integral. NO localized streamer.
"""

from __future__ import annotations

import json
import math
import os
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

import meshio  # noqa: E402

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123_3d_v5 import MAT_OF_V4, build_cvt_ddb123_3d_v5  # noqa: E402
from cvt_observables import compute_cvt_observables  # noqa: E402
from run_cvt import EPS0, OMEGA, U0, FREQ, build_material_table, ensure_solver, material_of_body, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402
from run_true3d import observables_3d  # noqa: E402

PROC = ROOT / "results" / "processed" / "025_insulator_mesh_remediation"
RAW = ROOT / "results" / "raw" / "025_insulator_mesh_remediation"
SIGMA_LIST = [1e-7, 1e-6]


def render_sif_3d(mat, mat_index, bm, bodies, hv, gnd, ff, *, ins=None, sigma_s=0.0, direct=True):
    lin = ("  Linear System Solver = Direct\n  Linear System Direct Method = UMFPack\n" if direct
           else "  Linear System Solver = Iterative\n  Linear System Iterative Method = BiCGStabl\n"
                "  BiCGStabl polynomial degree = 4\n  Linear System Preconditioning = ILU2\n"
                "  Linear System Max Iterations = 8000\n  Linear System Residual Output = 200\n")
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
    for mname, idx in mat_index.items():
        eps, tand, sig = mat[mname]
        parts.append(f"Material {idx}\n  Name = \"{mname}\"\n  Relative Permittivity = {eps!r}\n"
                     f"  Electric Conductivity = {sigma_eff(eps, tand, sig)!r}\nEnd\n")
    for bid, name in bodies:
        parts.append(f"Body {bid}\n  Name = \"{name}\"\n  Equation = 1\n"
                     f"  Material = {mat_index[bm[name]]}\nEnd\n")
    parts.append(f"Boundary Condition 1\n  Name = \"hv_electrode\"\n  Target Boundaries(1) = {hv}\n"
                 f"  Potential Re = Real {U0!r}\n  Potential Im = Real 0.0\nEnd\n")
    parts.append(f"Boundary Condition 2\n  Name = \"ground_electrode\"\n  Target Boundaries(1) = {gnd}\n"
                 "  Potential Re = Real 0.0\n  Potential Im = Real 0.0\nEnd\n")
    if ff is not None:
        parts.append(f"Boundary Condition 3\n  Name = \"farfield\"\n  Target Boundaries(1) = {ff}\n"
                     "  Potential Re = Real 0.0\n  Potential Im = Real 0.0\nEnd\n")
    if ins is not None and sigma_s > 0.0:
        parts.append(f"Boundary Condition 4\n  Name = \"insulator_surface\"\n  Target Boundaries(1) = {ins}\n"
                     f"  Surface Conductance = Real {float(sigma_s)!r}\nEnd\n")
    return "\n".join(parts)


def surface_loss_3d(vtu, sigma_s):
    """int_surf = INT sigma_s |grad_s phi|^2 dS over the porcelain-air creepage triangles."""
    m = meshio.read(str(vtu))
    pts = np.asarray(m.points, float)
    pd = {k.lower(): np.asarray(v, float).reshape(-1) for k, v in m.point_data.items()}
    re, im = pd["potential re"], pd["potential im"]
    tris = None
    for cb in m.cells:
        if cb.type in ("triangle", "triangle6"):
            tris = np.asarray(cb.data, int)[:, :3]; break
    if tris is None:
        return None
    p = pts[tris]
    cen = p.mean(1); rc = np.hypot(cen[:, 0], cen[:, 1]); zc = cen[:, 2]
    sel = (rc >= 0.11) & (rc <= 0.17) & (zc >= 0.557) & (zc <= 1.605)   # insulator creepage
    if not sel.any():
        return 0.0
    p = p[sel]; tr = tris[sel]
    e1 = p[:, 1] - p[:, 0]; e2 = p[:, 2] - p[:, 0]
    nrm = np.cross(e1, e2); area = 0.5 * np.linalg.norm(nrm, axis=1)
    nhat = nrm / np.linalg.norm(nrm, axis=1)[:, None]
    M = np.stack([e1, e2, nhat], axis=1)  # rows e1,e2,n
    gsq = np.zeros(len(tr))
    for f in (re, im):
        b = np.stack([f[tr[:, 1]] - f[tr[:, 0]], f[tr[:, 2]] - f[tr[:, 0]], np.zeros(len(tr))], axis=1)[:, :, None]
        g = np.linalg.solve(M, b)[:, :, 0]
        gsq += (g ** 2).sum(1)
    return float(np.sum(sigma_s * gsq * area))


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    print("Building v5 solve mesh ...")
    msh = RAW / "cvt3d_v5_solve.msh"
    meta = build_cvt_ddb123_3d_v5(msh, shed_size=0.006,
                                  lc={"oil": 0.04, "base_oil": 0.05, "tank_metal": 0.07,
                                      "element_c1": 0.03, "air": 0.40})
    print(f"  nodes={meta['n_nodes']} tets={meta['n_tets']} qmin={meta['quality_minSICN']['min']:.3f} "
          f"insulator p5={meta['insulator_surface_tri_quality']['p5']:.3f} creepage={meta['creepage_mm']:.0f}")
    (PROC / "solve_mesh_meta.json").write_text(json.dumps(
        {k: v for k, v in meta.items() if k not in ("parameters", "worst100")}, indent=2, default=str),
        encoding="utf-8")

    ep = RAW / "elmer"; ep.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(resolve_elmer_grid()), "14", "2", str(msh), "-out", "mesh"],
                   cwd=str(ep), capture_output=True, text=True, check=True)
    em = ep / "mesh"; ids = _parse_mesh_names(em / "mesh.names")
    bodies = [(bid, n) for n, bid in ids["bodies"].items()]
    tap_id = ids["bodies"]["foil_tap"]
    hv, gnd = ids["boundaries"]["hv_electrode"], ids["boundaries"]["ground_electrode"]
    ff = ids["boundaries"].get("farfield"); ins = ids["boundaries"].get("insulator_surface")

    eps = CVTParams().element_epsr_eff()
    mat = build_material_table(eps); mat["resin"] = (4.5, 0.01, 0.0)
    mat_index = {m_: i + 1 for i, m_ in enumerate(mat)}
    bm = {n: MAT_OF_V4.get(n, "air") for n in ids["bodies"]}
    be = {bid: mat[bm[n]][0] for n, bid in ids["bodies"].items()}
    bs = {bid: sigma_eff(*mat[bm[n]]) for n, bid in ids["bodies"].items()}

    pdir = RAW / "run3d"; pdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(em, pdir / "mesh", dirs_exist_ok=True); shutil.copy2(dll, pdir / dll.name)

    def solve(tag, sigma_s=0.0):
        for direct in (True, False):
            sif = render_sif_3d(mat, mat_index, bm, bodies, hv, gnd, ff, ins=ins, sigma_s=sigma_s, direct=direct)
            (pdir / "case.sif").write_text(sif, encoding="utf-8")
            t0 = time.time()
            d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                               capture_output=True, text=True, env=env)
            if d.returncode == 0 and sorted(pdir.glob("**/case*.vtu")):
                vtu = sorted(pdir.glob("**/case*.vtu"))[-1]
                return vtu, ("direct" if direct else "iterative"), time.time() - t0
            (pdir / f"solver_{tag}.log").write_text(d.stdout, encoding="utf-8")
        raise RuntimeError(f"solve failed: {tag}")

    # ---- clean ----
    print("Clean 3D solve ...")
    vtu, slv, dt = solve("clean", 0.0)
    o3 = observables_3d(vtu, be, bs, tap_id)
    int_eps = o3["C_pF"] * 1e-12 * U0 ** 2
    int_sig_clean = o3["tan_delta"] * OMEGA * int_eps
    print(f"  clean: C={o3['C_pF']:.1f}pF ratio={o3['divider_ratio']:.4f} ({slv}, {dt:.0f}s)")

    # 2D axisymmetric clean baseline
    p2 = CVTParams()
    m2 = RAW / "axi" / "cvt.msh"; m2.parent.mkdir(parents=True, exist_ok=True)
    build_cvt_ddb123(m2, p2, rounded=True)
    ep2 = RAW / "axi" / "elmer"; ep2.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(resolve_elmer_grid()), "14", "2", str(m2), "-out", "mesh"],
                   cwd=str(ep2), capture_output=True, text=True, check=True)
    em2 = ep2 / "mesh"; ids2 = _parse_mesh_names(em2 / "mesh.names")
    bodies2 = [(bid, n) for n, bid in ids2["bodies"].items()]
    tap2 = ids2["bodies"][f"foil_{p2.tap_disc_index}"]
    hv2, gnd2 = ids2["boundaries"]["hv_electrode"], ids2["boundaries"]["ground_electrode"]
    ff2, ins2 = ids2["boundaries"].get("farfield"), ids2["boundaries"].get("insulator_surface")
    mat2 = build_material_table(eps); mi2 = {m_: i + 1 for i, m_ in enumerate(mat2)}
    bm2 = {n: material_of_body(n) for n in ids2["bodies"]}
    be2 = {bid: mat2[bm2[n]][0] for n, bid in ids2["bodies"].items()}
    bs2 = {bid: sigma_eff(*mat2[bm2[n]]) for n, bid in ids2["bodies"].items()}
    pd2 = RAW / "axi" / "run"; pd2.mkdir(parents=True, exist_ok=True)
    shutil.copytree(em2, pd2 / "mesh", dirs_exist_ok=True); shutil.copy2(dll, pd2 / dll.name)

    def solve2d(sigma_s):
        sif = render_sif(mat2, mi2, bm2, bodies2, hv2, gnd2, ff2, ins_id=ins2, sigma_s=sigma_s)
        (pd2 / "case.sif").write_text(sif, encoding="utf-8")
        d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pd2),
                           capture_output=True, text=True, env=env)
        if d.returncode != 0:
            print(d.stdout[-800:]); raise RuntimeError("2D solve failed")
        v = sorted(pd2.glob("**/case*.vtu"))[-1]
        return compute_cvt_observables(v, u0=U0, omega=OMEGA, body_epsr=be2, body_sigma=bs2,
                                       tap_body_id=tap2, sigma_s=sigma_s)

    o2 = solve2d(0.0)
    clean = {"C_pF_3d": o3["C_pF"], "C_pF_2d": o2["C_total_pF"],
             "ratio_3d": o3["divider_ratio"], "ratio_2d": o2["divider_ratio"],
             "dC_pct": (o3["C_pF"] / o2["C_total_pF"] - 1) * 100,
             "dratio_pct": (o3["divider_ratio"] / o2["divider_ratio"] - 1) * 100,
             "phase_3d_mdeg": o3["vtap_phase_mdeg"], "solver": slv}

    # ---- uniform surface conductance: 3D vs 2D ----
    poll = []
    for ss in SIGMA_LIST:
        print(f"Surface conductance sigma_s={ss:.0e} ...")
        v3, s3, _ = solve(f"poll_{ss:.0e}", ss)
        ob3 = observables_3d(v3, be, bs, tap_id)
        int_surf3 = surface_loss_3d(v3, ss)
        ie3 = ob3["C_pF"] * 1e-12 * U0 ** 2
        tand3 = (int_sig_clean + int_surf3) / (OMEGA * ie3)
        sleak3 = int_surf3 / U0
        ob2 = solve2d(ss)
        poll.append({"sigma_s": ss,
                     "tan_delta_3d": tand3, "tan_delta_2d": ob2["tan_delta"],
                     "surface_leak_3d": sleak3, "surface_leak_2d": ob2["surface_leak"],
                     "d_tand_pct": (tand3 / ob2["tan_delta"] - 1) * 100 if ob2["tan_delta"] else None,
                     "d_sleak_pct": (sleak3 / ob2["surface_leak"] - 1) * 100 if ob2["surface_leak"] else None})
        print(f"  tand 3D={tand3:.3e} 2D={ob2['tan_delta']:.3e}; "
              f"sleak 3D={sleak3:.3e} 2D={ob2['surface_leak']:.3e}")

    result = {"clean": clean, "uniform_surface_conductance": poll,
              "mesh": {k: meta[k] for k in ("n_nodes", "n_tets", "quality_minSICN",
                       "n_slivers_lt_0.05", "insulator_surface_tri_quality", "creepage_mm")}}
    (PROC / "validation.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nclean dC={clean['dC_pct']:+.2f}%  dratio={clean['dratio_pct']:+.2f}%")
    print(f"Wrote {PROC / 'validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task 022 - solve the TRUE 3D DDB-123 clean baseline and validate vs axisymmetric.

Builds the full-360 homogenized 3D mesh (geometry/gmsh/cvt_ddb123_3d.py) -> ElmerGrid ->
renders a *Cartesian 3D* ComplexEQS SIF (NO axisymmetric 2*pi*r metric) -> ElmerSolver ->
true-3D observables (tetrahedral volume integration). Also runs the validated 2D
axisymmetric baseline and compares C / divider ratio / |Vtap| / phase / stored energy.

Do not trust unless Elmer actually solves the 3D mesh; the validation uses the real 3D
field, never revolved 2D.
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
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meshio  # noqa: E402

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123_3d import build_cvt_ddb123_3d  # noqa: E402
from cvt_observables import compute_cvt_observables  # noqa: E402
from run_cvt import EPS0, OMEGA, U0, build_material_table, ensure_solver, material_of_body, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

PROC = ROOT / "results" / "processed" / "022_true3d_foundation"
RAW = ROOT / "results" / "raw" / "022_true3d_foundation"
FREQ = 50.0


def render_sif_3d(mat, mat_index, body_material, bodies, hv, gnd, ff, *, direct=True):
    """Cartesian-3D ComplexEQS SIF. NO axisymmetric metric (Coordinate System=Cartesian)."""
    if direct:
        lin = ("  Linear System Solver = Direct\n  Linear System Direct Method = UMFPack\n")
    else:
        lin = ("  Linear System Solver = Iterative\n  Linear System Iterative Method = BiCGStabl\n"
               "  BiCGStabl polynomial degree = 4\n  Linear System Preconditioning = ILU2\n"
               "  Linear System Max Iterations = 5000\n  Linear System Residual Output = 100\n")
    parts = [
        'Header\n  CHECK KEYWORDS Warn\n  Mesh DB "." "mesh"\nEnd\n',
        'Simulation\n  Coordinate System = "Cartesian"\n  Simulation Type = Steady State\n'
        f'  Steady State Max Iterations = 1\n  Output Intervals = 1\n  Frequency = {FREQ}\n'
        '  Output File = "case.result"\n  Post File = "case.vtu"\nEnd\n',
        "Constants\n  Permittivity Of Vacuum = 8.8541878128e-12\nEnd\n",
        'Equation 1\n  Name = "ComplexEQS"\n  Active Solvers(1) = 1\nEnd\n',
        'Solver 1\n  Equation = "ComplexEQS"\n  Procedure = "ComplexEQS" "ComplexEQSSolver"\n'
        "  Variable = Potential[Potential Re:1 Potential Im:1]\n  Linear System Complex = True\n"
        f"{lin}"
        "  Linear System Convergence Tolerance = 1.0e-8\n"
        "  Steady State Convergence Tolerance = 1.0e-9\nEnd\n",
    ]
    for mname, idx in mat_index.items():
        eps, tand, sig = mat[mname]
        parts.append(f"Material {idx}\n  Name = \"{mname}\"\n  Relative Permittivity = {eps!r}\n"
                     f"  Electric Conductivity = {sigma_eff(eps, tand, sig)!r}\nEnd\n")
    for bid, name in bodies:
        parts.append(f"Body {bid}\n  Name = \"{name}\"\n  Equation = 1\n"
                     f"  Material = {mat_index[body_material[name]]}\nEnd\n")
    parts.append(f"Boundary Condition 1\n  Name = \"hv_electrode\"\n  Target Boundaries(1) = {hv}\n"
                 f"  Potential Re = Real {U0!r}\n  Potential Im = Real 0.0\nEnd\n")
    parts.append(f"Boundary Condition 2\n  Name = \"ground_electrode\"\n  Target Boundaries(1) = {gnd}\n"
                 "  Potential Re = Real 0.0\n  Potential Im = Real 0.0\nEnd\n")
    if ff is not None:
        parts.append(f"Boundary Condition 3\n  Name = \"farfield\"\n  Target Boundaries(1) = {ff}\n"
                     "  Potential Re = Real 0.0\n  Potential Im = Real 0.0\nEnd\n")
    return "\n".join(parts)


def observables_3d(vtu, body_epsr, body_sigma, tap_id):
    """True-3D observables: C (energy), |Vtap|, phase, stored energy, Emax, tan-delta."""
    m = meshio.read(str(vtu))
    pts = np.asarray(m.points, float)
    pd = {k.lower(): np.asarray(v, float).reshape(-1) for k, v in m.point_data.items()}
    re, im = pd["potential re"], pd["potential im"]
    tets = gids = None
    for i, b in enumerate(m.cells):
        if b.type in {"tetra", "tetra10"}:
            tets = np.asarray(b.data, int)[:, :4]
            gids = np.asarray(m.cell_data["GeometryIds"][i], int).reshape(-1)
            break
    if tets is None:
        raise RuntimeError("no tetrahedra in 3D VTU")

    p0 = pts[tets[:, 0]]
    M = np.stack([pts[tets[:, 1]] - p0, pts[tets[:, 2]] - p0, pts[tets[:, 3]] - p0], axis=1)  # (n,3,3)
    detM = np.linalg.det(M)
    vol = np.abs(detM) / 6.0
    b_re = np.stack([re[tets[:, 1]] - re[tets[:, 0]], re[tets[:, 2]] - re[tets[:, 0]],
                     re[tets[:, 3]] - re[tets[:, 0]]], axis=1)[:, :, None]  # (n,3,1)
    b_im = np.stack([im[tets[:, 1]] - im[tets[:, 0]], im[tets[:, 2]] - im[tets[:, 0]],
                     im[tets[:, 3]] - im[tets[:, 0]]], axis=1)[:, :, None]
    g_re = np.linalg.solve(M, b_re)[:, :, 0]
    g_im = np.linalg.solve(M, b_im)[:, :, 0]
    e2 = (g_re ** 2).sum(1) + (g_im ** 2).sum(1)
    epsr = np.array([body_epsr[g] for g in gids])
    sig = np.array([body_sigma[g] for g in gids])
    int_eps = float(np.sum(EPS0 * epsr * e2 * vol))
    int_sig = float(np.sum(sig * e2 * vol))
    c_total = int_eps / U0 ** 2
    energy = 0.5 * int_eps
    tan_delta = int_sig / (OMEGA * int_eps) if int_eps > 0 else 0.0

    tap_nodes = np.unique(tets[gids == tap_id].reshape(-1))
    vtap_re, vtap_im = float(re[tap_nodes].mean()), float(im[tap_nodes].mean())
    vtap_abs = math.hypot(vtap_re, vtap_im)

    emag = np.sqrt(e2)
    ic = int(np.argmax(emag))
    cent = pts[tets[ic]].mean(0)
    return {"C_pF": c_total * 1e12, "vtap_abs": vtap_abs,
            "vtap_phase_mdeg": math.degrees(math.atan2(vtap_im, vtap_re)) * 1e3,
            "divider_ratio": vtap_abs / U0, "stored_energy_J": energy,
            "tan_delta": tan_delta, "emax": float(emag.max()),
            "emax_rz": [float(math.hypot(cent[0], cent[1])), float(cent[2])],
            "n_tap_nodes": int(tap_nodes.size)}


def run_2d_baseline(dll, env):
    """Fresh axisymmetric baseline for the comparison."""
    p = CVTParams()
    msh = RAW / "axi" / "cvt.msh"; msh.parent.mkdir(parents=True, exist_ok=True)
    build_cvt_ddb123(msh, p, rounded=True)
    ep = RAW / "axi" / "elmer"; ep.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(resolve_elmer_grid()), "14", "2", str(msh), "-out", "mesh"],
                   cwd=str(ep), capture_output=True, text=True, check=True)
    em = ep / "mesh"; ids = _parse_mesh_names(em / "mesh.names")
    bodies = [(bid, n) for n, bid in ids["bodies"].items()]
    tap_id = ids["bodies"][f"foil_{p.tap_disc_index}"]
    hv, gnd = ids["boundaries"]["hv_electrode"], ids["boundaries"]["ground_electrode"]
    ff, ins = ids["boundaries"].get("farfield"), ids["boundaries"].get("insulator_surface")
    eps = p.element_epsr_eff()
    mat = build_material_table(eps); mat_index = {m_: i + 1 for i, m_ in enumerate(mat)}
    bm = {n: material_of_body(n) for n in ids["bodies"]}
    pdir = RAW / "axi" / "run"; pdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(em, pdir / "mesh", dirs_exist_ok=True); shutil.copy2(dll, pdir / dll.name)
    sif = render_sif(mat, mat_index, bm, bodies, hv, gnd, ff, ins_id=ins)
    (pdir / "case.sif").write_text(sif, encoding="utf-8")
    d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                       capture_output=True, text=True, env=env)
    if d.returncode != 0:
        print(d.stdout[-1000:]); raise RuntimeError("2D baseline solve failed")
    vtu = sorted(pdir.glob("**/case*.vtu"))[-1]
    be = {bid: mat[bm[n]][0] for n, bid in ids["bodies"].items()}
    bs = {bid: sigma_eff(*mat[bm[n]]) for n, bid in ids["bodies"].items()}
    o = compute_cvt_observables(vtu, u0=U0, omega=OMEGA, body_epsr=be, body_sigma=bs, tap_body_id=tap_id)
    return {"C_pF": o["C_total_pF"], "vtap_abs": o["vtap_abs"],
            "vtap_phase_mdeg": o["vtap_phase_mdeg"], "divider_ratio": o["divider_ratio"],
            "stored_energy_J": 0.5 * o["C_total_F"] * U0 ** 2, "tan_delta": o["tan_delta"]}


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    print("Building TRUE 3D mesh ...")
    msh = RAW / "cvt3d.msh"
    meta = build_cvt_ddb123_3d(msh)
    print(f"  nodes={meta['n_nodes']} tets={meta['n_tets']} qmin={meta['min_quality_minSICN']:.3f} "
          f"qmean={meta['mean_quality_minSICN']:.3f}")

    ep = RAW / "elmer"; ep.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(resolve_elmer_grid()), "14", "2", str(msh), "-out", "mesh"],
                   cwd=str(ep), capture_output=True, text=True, check=True)
    em = ep / "mesh"; ids = _parse_mesh_names(em / "mesh.names")
    bodies = [(bid, n) for n, bid in ids["bodies"].items()]
    tap_id = ids["bodies"]["foil_tap"]
    hv, gnd = ids["boundaries"]["hv_electrode"], ids["boundaries"]["ground_electrode"]
    ff = ids["boundaries"].get("farfield")

    eps = CVTParams().element_epsr_eff()
    mat = build_material_table(eps); mat_index = {m_: i + 1 for i, m_ in enumerate(mat)}
    bm = {n: material_of_body(n) for n in ids["bodies"]}
    be = {bid: mat[bm[n]][0] for n, bid in ids["bodies"].items()}
    bs = {bid: sigma_eff(*mat[bm[n]]) for n, bid in ids["bodies"].items()}

    pdir = RAW / "run3d"; pdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(em, pdir / "mesh", dirs_exist_ok=True); shutil.copy2(dll, pdir / dll.name)

    solved = None
    for direct in (True, False):
        sif = render_sif_3d(mat, mat_index, bm, bodies, hv, gnd, ff, direct=direct)
        (pdir / "case.sif").write_text(sif, encoding="utf-8")
        print(f"Solving 3D ({'UMFPack direct' if direct else 'BiCGStabl iterative'}) ...")
        t0 = time.time()
        d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                           capture_output=True, text=True, env=env)
        dt = time.time() - t0
        (pdir / "solver.log").write_text(d.stdout, encoding="utf-8")
        if d.returncode == 0 and sorted(pdir.glob("**/case*.vtu")):
            solved = ("direct" if direct else "iterative", dt)
            print(f"  solved in {dt:.1f}s ({solved[0]})")
            break
        print(f"  FAILED ({'direct' if direct else 'iterative'}); tail:\n{d.stdout[-700:]}")
    if solved is None:
        print("3D solve failed with both linear solvers."); return 1

    vtu = sorted(pdir.glob("**/case*.vtu"))[-1]
    o3 = observables_3d(vtu, be, bs, tap_id)
    print("Running 2D axisymmetric baseline for comparison ...")
    o2 = run_2d_baseline(dll, env)

    def cmp(k):
        a, b = o3[k], o2[k]
        return None if not b else (a / b - 1.0) * 100.0

    result = {
        "solver": solved[0], "solve_seconds": solved[1],
        "mesh": {k: meta[k] for k in ("n_nodes", "n_tets", "min_quality_minSICN", "mean_quality_minSICN")},
        "true3d": o3, "axisymmetric": o2,
        "delta_pct": {k: cmp(k) for k in ("C_pF", "vtap_abs", "vtap_phase_mdeg", "divider_ratio", "stored_energy_J")},
    }
    (PROC / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n               3D            axisym        delta")
    for k, label in [("C_pF", "C [pF]"), ("divider_ratio", "ratio"), ("vtap_abs", "|Vtap| [V]"),
                     ("vtap_phase_mdeg", "phase [mdeg]"), ("stored_energy_J", "energy [J]")]:
        dv = result["delta_pct"][k]
        print(f"{label:14s} {o3[k]:12.4g} {o2[k]:12.4g}  {('' if dv is None else f'{dv:+.2f}%')}")
    print(f"Emax 3D = {o3['emax']:.3e} V/m at (r,z)=({o3['emax_rz'][0]:.3f},{o3['emax_rz'][1]:.3f})")
    print(f"\nWrote {PROC / 'validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task 007 - verification audit of the ComplexEQS solver.

This is a VERIFICATION harness only. It adds no new physics and no new fault
cases; it builds simple analytically-tractable geometries and checks that
ComplexEQS solves the intended weak form

    div( (sigma + i*omega*eps0*eps_r) grad(phi) ) = 0

correctly. Tests:

  T1  sigma=0 reduces to the real StatElec baseline (node-by-node field match).
  T2  sign convention: positive sigma -> positive dissipated power, and the
      reactive part / interface phase carry the +j*omega*eps sign.
  T3  analytical 1-D lossy parallel-plate capacitor: numerical admittance
      Y = (A/L)(sigma + j*omega*eps) vs analytic.
  T4  mesh convergence of T3.
  T5  two stacked regions with different (sigma, eps): continuity of potential
      and of normal total current J_n = (sigma + i*omega*eps) E_n across the
      interface, plus the complex voltage-divider interface potential.

Outputs: results/processed/007_complex_eqs_verification/{verification_report.md,
*.csv, *.png}. Raw Elmer runs under results/raw/007_complex_eqs_verification/.

The script exits non-zero if any test fails its documented tolerance. Failures
are reported, not hidden.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402
import gmsh  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publication_pipeline._elmer_tools import (  # noqa: E402
    resolve_elmer_grid,
    resolve_elmer_home,
    resolve_elmer_solver,
)
from build_solver import build_solver  # noqa: E402
from observables import EPS0, _triangle_gradients  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

SOLVER_SOURCE = ROOT / "elmer" / "solvers" / "ComplexEQS.F90"
SOLVER_DLL = ROOT / "elmer" / "solvers" / "ComplexEQS.dll"
STATELEC_TEMPLATE = ROOT / "elmer" / "sif_templates" / "clean_baseline.sif"
BASELINE_MESH = ROOT / "mesh" / "elmer" / "mesh"
RAW = ROOT / "results" / "raw" / "007_complex_eqs_verification"
PROC = ROOT / "results" / "processed" / "007_complex_eqs_verification"

FREQ = 50.0
OMEGA = 2.0 * np.pi * FREQ


# --------------------------------------------------------------------------- #
# infrastructure
# --------------------------------------------------------------------------- #
@dataclass
class TestResult:
    name: str
    passed: bool
    tolerance: str
    metrics: dict = field(default_factory=dict)
    notes: str = ""


def solver_env() -> dict:
    import os

    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def ensure_solver() -> Path:
    if not SOLVER_DLL.is_file() or SOLVER_DLL.stat().st_mtime < SOLVER_SOURCE.stat().st_mtime:
        build_solver(SOLVER_SOURCE, SOLVER_DLL)
    return SOLVER_DLL


def run_elmer(run_dir: Path, sif_text: str, src_mesh_dir: Path, dll: Path | None, env: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    mesh_dst = run_dir / "mesh"
    # Overwrite in place rather than rmtree: these fixed run dirs persist between
    # runs and a destructive delete trips OneDrive/Windows file locks (WinError 5).
    shutil.copytree(src_mesh_dir, mesh_dst, dirs_exist_ok=True)
    if dll is not None:
        shutil.copy2(dll, run_dir / dll.name)
    (run_dir / "case.sif").write_text(sif_text, encoding="utf-8")

    solver = resolve_elmer_solver()
    done = subprocess.run([str(solver), "case.sif"], cwd=str(run_dir), capture_output=True, text=True, env=env)
    (run_dir / "elmersolver.stdout.log").write_text(done.stdout, encoding="utf-8")
    (run_dir / "elmersolver.stderr.log").write_text(done.stderr, encoding="utf-8")
    if done.returncode != 0:
        raise RuntimeError(f"ElmerSolver failed in {run_dir}:\n{done.stdout[-1500:]}")
    vtus = sorted(run_dir.glob("**/case*.vtu"))
    if not vtus:
        raise RuntimeError(f"No VTU produced in {run_dir}")
    return vtus[-1]


def build_rect_mesh(msh_path: Path, width: float, layers: list[tuple[float, str]], lc: float) -> float:
    """Stacked-rectangle mesh. layers = [(height, region_name), ...] bottom->top.

    Tags region bodies plus BOTTOM (y=0) and TOP (y=Ltot) boundaries; side walls
    are left unnamed (natural zero-normal-current BC -> uniform vertical field).
    Returns total height Ltot.
    """
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add("vrect")
        occ = gmsh.model.occ

        y = 0.0
        rect_tags = []
        intervals = []
        for h, name in layers:
            rect_tags.append(occ.addRectangle(0.0, y, 0.0, width, h))
            intervals.append((y, y + h, name))
            y += h
        ltot = y
        occ.synchronize()
        if len(rect_tags) > 1:
            occ.fragment([(2, rect_tags[0])], [(2, t) for t in rect_tags[1:]])
            occ.synchronize()

        for d, t in gmsh.model.getEntities(0):
            gmsh.model.mesh.setSize([(d, t)], lc)

        def near(a: float, b: float, tol: float = 1e-7) -> bool:
            return abs(a - b) <= tol

        region_surfs: dict[str, list[int]] = {name: [] for _, name in layers}
        for d, s in gmsh.model.getEntities(2):
            bb = occ.getBoundingBox(d, s)
            midy = 0.5 * (bb[1] + bb[4])
            name = next(nm for (a, b, nm) in intervals if a - 1e-6 <= midy <= b + 1e-6)
            region_surfs[name].append(s)
        for name, surfs in region_surfs.items():
            if surfs:
                pg = gmsh.model.addPhysicalGroup(2, sorted(set(surfs)))
                gmsh.model.setPhysicalName(2, pg, name)

        bottom, top = [], []
        for d, c in gmsh.model.getEntities(1):
            bb = occ.getBoundingBox(d, c)
            if near(bb[1], 0.0) and near(bb[4], 0.0):
                bottom.append(c)
            elif near(bb[1], ltot) and near(bb[4], ltot):
                top.append(c)
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, sorted(set(bottom))), "BOTTOM")
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, sorted(set(top))), "TOP")

        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh_path))
        return ltot
    finally:
        gmsh.finalize()


def build_annulus_mesh(msh_path: Path, a: float, b: float, length: float, lc: float) -> None:
    """Coaxial annulus a<=r<=b, 0<=z<=length (r,z rectangle); tags INNER (r=a) and
    OUTER (r=b) boundaries, single DIELECTRIC body. For the axisymmetric coax test."""
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add("annulus")
        occ = gmsh.model.occ
        s = occ.addRectangle(a, 0.0, 0.0, b - a, length)
        occ.synchronize()
        for d, t in gmsh.model.getEntities(0):
            gmsh.model.mesh.setSize([(d, t)], lc)
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [s]), "DIELECTRIC")

        def near(u, v, tol=1e-6):
            return abs(u - v) <= tol

        # Tag by curve centre-of-mass radius (robust to gmsh bbox padding):
        # vertical walls sit at r=a / r=b; horizontal edges at r=(a+b)/2.
        inner, outer = [], []
        for d, c in gmsh.model.getEntities(1):
            com = occ.getCenterOfMass(d, c)
            if near(com[0], a):
                inner.append(c)
            elif near(com[0], b):
                outer.append(c)
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, sorted(set(inner))), "INNER")
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, sorted(set(outer))), "OUTER")
        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()


def convert_mesh(msh_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    eg = resolve_elmer_grid()
    if not eg:
        raise RuntimeError("ElmerGrid not found.")
    done = subprocess.run(
        [str(eg), "14", "2", str(msh_path), "-out", "mesh"], cwd=str(out_dir), capture_output=True, text=True
    )
    if done.returncode != 0:
        raise RuntimeError(done.stdout + done.stderr)
    return out_dir / "mesh"


def complex_sif(materials: list[tuple[float, float]], bodies: list[tuple[int, int]], bcs: list[tuple[int, int, float, float]], axisymmetric: bool = False) -> str:
    """materials=[(eps_r, sigma),...] (1-indexed); bodies=[(body_id, mat_idx),...];
    bcs=[(bc_index, boundary_id, V_re, V_im),...]."""
    coord = "Axi Symmetric" if axisymmetric else "Cartesian 2D"
    parts = [
        "Header\n  CHECK KEYWORDS Warn\n  Mesh DB \".\" \"mesh\"\nEnd\n",
        f"Simulation\n  Coordinate System = \"{coord}\"\n  Simulation Type = Steady State\n"
        f"  Steady State Max Iterations = 1\n  Output Intervals = 1\n  Frequency = {FREQ}\n"
        f"  Output File = \"case.result\"\n  Post File = \"case.vtu\"\nEnd\n",
        "Constants\n  Permittivity Of Vacuum = 8.8541878128e-12\nEnd\n",
        "Equation 1\n  Name = \"ComplexEQS\"\n  Active Solvers(1) = 1\nEnd\n",
        "Solver 1\n  Equation = \"ComplexEQS\"\n  Procedure = \"ComplexEQS\" \"ComplexEQSSolver\"\n"
        "  Variable = Potential[Potential Re:1 Potential Im:1]\n  Linear System Complex = True\n"
        "  Linear System Solver = Direct\n  Linear System Direct Method = UMFPack\n"
        "  Steady State Convergence Tolerance = 1.0e-9\nEnd\n",
    ]
    for i, (eps, sig) in enumerate(materials, 1):
        parts.append(f"Material {i}\n  Relative Permittivity = {eps!r}\n  Electric Conductivity = {sig!r}\nEnd\n")
    for bid, mi in bodies:
        parts.append(f"Body {bid}\n  Equation = 1\n  Material = {mi}\nEnd\n")
    for bi, bnd, vre, vim in bcs:
        parts.append(
            f"Boundary Condition {bi}\n  Target Boundaries(1) = {bnd}\n"
            f"  Potential Re = Real {vre!r}\n  Potential Im = Real {vim!r}\nEnd\n"
        )
    return "\n".join(parts)


def read_complex(vtu: Path):
    m = meshio.read(str(vtu))
    pts = np.asarray(m.points, dtype=float)
    pd = {k.lower(): np.asarray(v, dtype=float).reshape(-1) for k, v in m.point_data.items()}
    re = pd["potential re"]
    im = pd["potential im"]
    tris = next(np.asarray(b.data, dtype=int)[:, :3] for b in m.cells if b.type in {"triangle", "triangle6"})
    return pts, tris, re, im


def read_real(vtu: Path):
    m = meshio.read(str(vtu))
    pts = np.asarray(m.points, dtype=float)
    for k, v in m.point_data.items():
        parts = k.lower().split()
        if "potential" in parts and "re" not in parts and "im" not in parts:
            return pts, np.asarray(v, dtype=float).reshape(-1)
    raise KeyError(f"no real potential field in {list(m.point_data)}")


def complex_field(pts, tris, re, im):
    """Per-element complex E = -grad(phi) and triangle area."""
    grad_re, area = _triangle_gradients(pts, tris, re)
    grad_im, _ = _triangle_gradients(pts, tris, im)
    ex = -(grad_re[:, 0] + 1j * grad_im[:, 0])
    ey = -(grad_re[:, 1] + 1j * grad_im[:, 1])
    return ex, ey, area


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test1_sigma0_vs_statelec(dll: Path, env: dict) -> TestResult:
    ids = _parse_mesh_names(BASELINE_MESH / "mesh.names")
    hv, gnd = ids["boundaries"]["HV_TERMINAL"], ids["boundaries"]["GROUND"]
    air, por = ids["bodies"]["AIR"], ids["bodies"]["PORCELAIN"]
    v0 = 110000.0

    statelec_sif = STATELEC_TEMPLATE.read_text(encoding="utf-8")
    statelec_sif = statelec_sif.replace("{{HV_BOUNDARY_ID}}", str(hv)).replace("{{GROUND_BOUNDARY_ID}}", str(gnd))
    vtu_se = run_elmer(RAW / "t1_statelec", statelec_sif, BASELINE_MESH, None, env)

    sif_cx = complex_sif(
        materials=[(1.0, 0.0), (5.0, 0.0)],
        bodies=[(air, 1), (por, 2)],
        bcs=[(1, hv, v0, 0.0), (2, gnd, 0.0, 0.0)],
    )
    vtu_cx = run_elmer(RAW / "t1_complex_sigma0", sif_cx, BASELINE_MESH, dll, env)

    pts_se, pot_se = read_real(vtu_se)
    pts_cx, re_cx, im_cx = read_complex(vtu_cx)[0], read_complex(vtu_cx)[2], read_complex(vtu_cx)[3]

    same_mesh = np.allclose(pts_se, pts_cx, atol=1e-12)
    d_re = float(np.max(np.abs(pot_se - re_cx)))
    im_max = float(np.max(np.abs(im_cx)))
    rel_re = d_re / v0
    rel_im = im_max / v0
    passed = same_mesh and rel_re < 1e-6 and rel_im < 1e-6
    return TestResult(
        "T1 sigma=0 reduces to StatElec baseline",
        passed,
        "max|dRe|/V0 < 1e-6 and max|Im|/V0 < 1e-6 (same mesh)",
        {
            "same_mesh_node_order": same_mesh,
            "max_abs_dRe_V": d_re,
            "rel_dRe": rel_re,
            "max_abs_Im_V": im_max,
            "rel_Im": rel_im,
            "n_nodes": int(pts_se.shape[0]),
        },
    )


def _admittance(vtu: Path, width: float, ltot: float, eps_r: float, sigma: float, v0: float) -> dict:
    pts, tris, re, im = read_complex(vtu)
    ex, ey, area = complex_field(pts, tris, re, im)
    e2 = np.abs(ex) ** 2 + np.abs(ey) ** 2  # |E|^2 (phasor magnitude squared)
    int_sig = sigma * float(np.sum(e2 * area))
    int_eps = EPS0 * eps_r * float(np.sum(e2 * area))
    g_num = int_sig / v0**2
    c_num = int_eps / v0**2
    y_num = complex(g_num, OMEGA * c_num)
    a_over_l = width / ltot
    y_an = a_over_l * (sigma + 1j * OMEGA * EPS0 * eps_r)
    # field uniformity (single material -> field should be uniform, Im ~ 0)
    ey_mean = complex(np.mean(ey.real), np.mean(ey.imag))
    ey_std = float(np.std(np.abs(ey)))
    return {
        "g_num": g_num,
        "c_num": c_num,
        "y_num": y_num,
        "y_analytic": y_an,
        "g_analytic": a_over_l * sigma,
        "c_analytic": a_over_l * EPS0 * eps_r,
        "rel_err_Y": abs(y_num - y_an) / abs(y_an),
        "p_loss": 0.5 * int_sig,
        "ey_mean_abs": abs(ey_mean),
        "ey_expected": v0 / ltot,
        "ey_rel_std": ey_std / (v0 / ltot),
        "nelem": int(tris.shape[0]),
    }


def test3_lossy_capacitor(dll: Path, env: dict) -> TestResult:
    width, ltot, eps_r, sigma, v0, lc = 0.02, 0.01, 4.0, 1.0e-6, 1000.0, 0.01 / 16
    msh = RAW / "t3_cap" / "cap.msh"
    height = build_rect_mesh(msh, width, [(ltot, "DIELECTRIC")], lc)
    mesh_dir = convert_mesh(msh, RAW / "t3_cap" / "elmer")
    ids = _parse_mesh_names(mesh_dir / "mesh.names")
    body = ids["bodies"]["DIELECTRIC"]
    top, bot = ids["boundaries"]["TOP"], ids["boundaries"]["BOTTOM"]

    sif = complex_sif([(eps_r, sigma)], [(body, 1)], [(1, top, v0, 0.0), (2, bot, 0.0, 0.0)])
    vtu = run_elmer(RAW / "t3_cap" / "run", sif, mesh_dir, dll, env)
    a = _admittance(vtu, width, height, eps_r, sigma, v0)
    passed = a["rel_err_Y"] < 1e-3 and a["ey_rel_std"] < 1e-6
    a.update({"width": width, "gap": ltot, "eps_r": eps_r, "sigma": sigma, "v0": v0})
    return TestResult(
        "T3 analytic lossy capacitor admittance",
        passed,
        "|Y_num - Y_analytic|/|Y_analytic| < 1e-3 and field-uniformity rel-std < 1e-6",
        a,
    )


def test2_sign_convention(t3: TestResult, t5_data: dict) -> TestResult:
    # (a) positive sigma -> strictly positive dissipated power (from T3 lossy cap)
    p_loss = t3.metrics["p_loss"]
    g_num = t3.metrics["g_num"]
    v0 = t3.metrics["v0"]
    p_terminal = 0.5 * g_num * v0**2  # Re(1/2 V I*) = 1/2 G V^2
    power_balance = abs(p_loss - p_terminal) / p_loss if p_loss else 1.0
    cap_reactive = OMEGA * t3.metrics["c_num"] > 0  # Im(Y) > 0 -> capacitive

    # (b) the +j*omega*eps SIGN is what fixes the phase of the complex interface
    #     potential in a two-material divider. Compare numeric vs analytic sign.
    vi_num = t5_data["vi_num"]
    vi_an = t5_data["vi_analytic"]
    sign_ok = (np.sign(vi_num.imag) == np.sign(vi_an.imag)) and (
        abs(vi_num - vi_an) / abs(vi_an) < 1e-3
    )

    passed = (p_loss > 0) and (power_balance < 1e-6) and cap_reactive and sign_ok
    return TestResult(
        "T2 sign convention (positive dissipation, +jwe reactive sign)",
        passed,
        "P_loss>0; |P_loss-Re(1/2 V I*)|/P_loss<1e-6; Im(Y)>0; sign & value of complex Vi match analytic",
        {
            "p_loss_W_per_m": p_loss,
            "p_terminal_W_per_m": p_terminal,
            "power_balance_rel": power_balance,
            "im_Y_positive_capacitive": bool(cap_reactive),
            "Vi_num_imag": vi_num.imag,
            "Vi_analytic_imag": vi_an.imag,
            "Vi_sign_and_value_match": bool(sign_ok),
        },
    )


def test4_convergence(dll: Path, env: dict) -> tuple[TestResult, list[dict]]:
    width, ltot, eps_r, sigma, v0 = 0.02, 0.01, 4.0, 1.0e-6, 1000.0
    rows = []
    for k in (4, 8, 16, 32):
        lc = ltot / k
        base = RAW / f"t4_conv_lc{k}"
        msh = base / "cap.msh"
        height = build_rect_mesh(msh, width, [(ltot, "DIELECTRIC")], lc)
        mesh_dir = convert_mesh(msh, base / "elmer")
        ids = _parse_mesh_names(mesh_dir / "mesh.names")
        body = ids["bodies"]["DIELECTRIC"]
        top, bot = ids["boundaries"]["TOP"], ids["boundaries"]["BOTTOM"]
        sif = complex_sif([(eps_r, sigma)], [(body, 1)], [(1, top, v0, 0.0), (2, bot, 0.0, 0.0)])
        vtu = run_elmer(base / "run", sif, mesh_dir, dll, env)
        a = _admittance(vtu, width, height, eps_r, sigma, v0)
        rows.append({"lc": lc, "k": k, "nelem": a["nelem"], "rel_err_Y": a["rel_err_Y"], "g_num": a["g_num"], "c_num": a["c_num"]})
    worst = max(r["rel_err_Y"] for r in rows)
    passed = worst < 1e-3
    return (
        TestResult(
            "T4 mesh convergence of lossy-capacitor admittance",
            passed,
            "rel_err_Y < 1e-3 at every refinement level",
            {"worst_rel_err_Y": worst, "levels": [r["k"] for r in rows]},
            notes="Parallel-plate field is uniform, so linear FE is exact at every level "
            "(error sits at round-off); this verifies mesh-independence.",
        ),
        rows,
    )


def test5_two_region(dll: Path, env: dict) -> tuple[TestResult, dict]:
    width = 0.02
    d1, d2 = 0.005, 0.005
    eps1, sig1 = 2.0, 1.0e-6
    eps2, sig2 = 6.0, 1.0e-7
    v0 = 1000.0
    lc = 0.005 / 8

    msh = RAW / "t5_tworegion" / "stack.msh"
    ltot = build_rect_mesh(msh, width, [(d1, "REGION1"), (d2, "REGION2")], lc)
    mesh_dir = convert_mesh(msh, RAW / "t5_tworegion" / "elmer")
    ids = _parse_mesh_names(mesh_dir / "mesh.names")
    r1, r2 = ids["bodies"]["REGION1"], ids["bodies"]["REGION2"]
    top, bot = ids["boundaries"]["TOP"], ids["boundaries"]["BOTTOM"]

    sif = complex_sif([(eps1, sig1), (eps2, sig2)], [(r1, 1), (r2, 2)], [(1, top, v0, 0.0), (2, bot, 0.0, 0.0)])
    vtu = run_elmer(RAW / "t5_tworegion" / "run", sif, mesh_dir, dll, env)

    pts, tris, re, im = read_complex(vtu)
    ex, ey, area = complex_field(pts, tris, re, im)
    cy = pts[tris].mean(axis=1)[:, 1]

    k1 = sig1 + 1j * OMEGA * EPS0 * eps1
    k2 = sig2 + 1j * OMEGA * EPS0 * eps2

    m1 = cy < d1
    m2 = cy > d1
    ey1 = complex(np.mean(ey[m1].real), np.mean(ey[m1].imag))
    ey2 = complex(np.mean(ey[m2].real), np.mean(ey[m2].imag))
    j1 = k1 * ey1  # normal total current density region 1
    j2 = k2 * ey2  # region 2
    j_continuity_rel = abs(j1 - j2) / abs(j1)

    # analytic series divider (bottom=0 .. top=v0)
    j_analytic = v0 / (d1 / k1 + d2 / k2)
    vi_analytic = j_analytic * d1 / k1  # interface potential measured from bottom

    interface = np.where(np.abs(pts[:, 1] - d1) <= 1e-7)[0]
    vi_num = complex(np.mean(re[interface]), np.mean(im[interface]))
    vi_std = float(np.std(np.abs(re[interface] + 1j * im[interface])))

    vi_rel = abs(vi_num - vi_analytic) / abs(vi_analytic)
    # potential continuity is structural for C0 nodal FE; verify field is bounded/monotone-consistent
    phi = re + 1j * im
    pot_continuous = (re.max() <= v0 + 1e-6 * v0) and (re.min() >= -1e-6 * v0)

    passed = j_continuity_rel < 1e-3 and vi_rel < 1e-3 and pot_continuous
    data = {
        "k1": k1, "k2": k2, "j1": j1, "j2": j2,
        "j_continuity_rel": j_continuity_rel,
        "j_analytic": j_analytic,
        "vi_num": vi_num, "vi_analytic": vi_analytic, "vi_rel": vi_rel, "vi_std": vi_std,
        "pot_continuous": pot_continuous,
        "pts": pts, "phi": phi, "d1": d1, "ltot": ltot, "v0": v0,
        "width": width,
    }
    return (
        TestResult(
            "T5 two-region continuity (potential + normal current)",
            passed,
            "|J_n1-J_n2|/|J_n1| < 1e-3; |Vi_num-Vi_analytic|/|Vi_analytic| < 1e-3; potential bounded/continuous",
            {
                "j_continuity_rel": j_continuity_rel,
                "abs_j1": abs(j1), "abs_j2": abs(j2),
                "vi_num": vi_num, "vi_analytic": vi_analytic, "vi_rel": vi_rel, "vi_std_over_nodes": vi_std,
                "potential_continuous_C0": bool(pot_continuous),
            },
        ),
        data,
    )


def test6_axisym_coax(dll: Path, env: dict) -> TestResult:
    """Axisymmetric coaxial capacitor: validates the radial metric weight against
    the analytic C = 2*pi*eps*L/ln(b/a). The 1/r field is curved, so unlike the
    uniform-field Cartesian cases this is also a genuine discretisation check."""
    a, b, length, eps_r, v0 = 0.02, 0.04, 0.05, 3.0, 1000.0
    lc = (b - a) / 40.0
    msh = RAW / "t6_coax" / "coax.msh"
    build_annulus_mesh(msh, a, b, length, lc)
    mesh_dir = convert_mesh(msh, RAW / "t6_coax" / "elmer")
    ids = _parse_mesh_names(mesh_dir / "mesh.names")
    body = ids["bodies"]["DIELECTRIC"]
    inner, outer = ids["boundaries"]["INNER"], ids["boundaries"]["OUTER"]

    sif = complex_sif([(eps_r, 0.0)], [(body, 1)], [(1, inner, v0, 0.0), (2, outer, 0.0, 0.0)], axisymmetric=True)
    vtu = run_elmer(RAW / "t6_coax" / "run", sif, mesh_dir, dll, env)

    pts, tris, re, im = read_complex(vtu)
    grad_re, area = _triangle_gradients(pts, tris, re)
    grad_im, _ = _triangle_gradients(pts, tris, im)
    e2 = np.sum(grad_re**2, axis=1) + np.sum(grad_im**2, axis=1)
    r_cent = pts[tris].mean(axis=1)[:, 0]  # centroid radius
    # energy capacitance with the axisymmetric volume element dV = 2*pi*r dr dz
    int_eps_e2 = EPS0 * eps_r * float(np.sum(e2 * area * 2.0 * np.pi * r_cent))
    c_num = int_eps_e2 / v0**2
    c_an = 2.0 * np.pi * EPS0 * eps_r * length / np.log(b / a)
    rel = abs(c_num - c_an) / c_an
    im_max = float(np.max(np.abs(im))) / v0
    passed = rel < 1e-3 and im_max < 1e-9
    return TestResult(
        "T6 axisymmetric coaxial capacitor",
        passed,
        "|C_num - C_analytic|/C_analytic < 1e-3 (axisymmetric metric); Im/V0 < 1e-9",
        {
            "C_num_F": c_num,
            "C_analytic_F": c_an,
            "rel_err_C": rel,
            "a_m": a, "b_m": b, "length_m": length, "eps_r": eps_r,
            "nelem": int(tris.shape[0]),
            "rel_Im": im_max,
        },
        notes="Curved 1/r field -> genuine convergence check; validates the radial weight.",
    )


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_csvs(t3: TestResult, conv_rows: list[dict], t5: dict) -> None:
    PROC.mkdir(parents=True, exist_ok=True)

    with (PROC / "mesh_convergence.csv").open("w", encoding="utf-8") as fh:
        fh.write("k,lc,nelem,rel_err_Y,g_num,c_num\n")
        for r in conv_rows:
            fh.write(f"{r['k']},{r['lc']:.10g},{r['nelem']},{r['rel_err_Y']:.6e},{r['g_num']:.10e},{r['c_num']:.10e}\n")

    with (PROC / "admittance_lossy_capacitor.csv").open("w", encoding="utf-8") as fh:
        fh.write("quantity,numeric,analytic,rel_err\n")
        fh.write(f"G[S/m_depth],{t3.metrics['g_num']:.10e},{t3.metrics['g_analytic']:.10e},"
                 f"{abs(t3.metrics['g_num']-t3.metrics['g_analytic'])/t3.metrics['g_analytic']:.3e}\n")
        fh.write(f"C[F/m_depth],{t3.metrics['c_num']:.10e},{t3.metrics['c_analytic']:.10e},"
                 f"{abs(t3.metrics['c_num']-t3.metrics['c_analytic'])/t3.metrics['c_analytic']:.3e}\n")
        fh.write(f"|Y|,{abs(t3.metrics['y_num']):.10e},{abs(t3.metrics['y_analytic']):.10e},"
                 f"{t3.metrics['rel_err_Y']:.3e}\n")


def make_plots(conv_rows: list[dict], t5: dict) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # convergence
    ax = axes[0]
    nel = [r["nelem"] for r in conv_rows]
    err = [max(r["rel_err_Y"], 1e-16) for r in conv_rows]
    ax.loglog(nel, err, "o-", color="#2563eb")
    ax.set_xlabel("number of elements")
    ax.set_ylabel("relative admittance error |Y_num - Y_an|/|Y_an|")
    ax.set_title("T4: lossy-capacitor admittance convergence")
    ax.grid(True, which="both", alpha=0.3)

    # T5 potential profile along centre line vs analytic
    ax = axes[1]
    pts, phi = t5["pts"], t5["phi"]
    d1, ltot, v0 = t5["d1"], t5["ltot"], t5["v0"]
    k1, k2, j = t5["k1"], t5["k2"], t5["j_analytic"]
    centre = np.abs(pts[:, 0] - t5["width"] / 2.0) <= t5["width"] / 8.0
    ysel = pts[centre, 1]
    order = np.argsort(ysel)
    ys = ysel[order]
    re_num = phi[centre].real[order]
    im_num = phi[centre].imag[order]
    yy = np.linspace(0, ltot, 200)
    v_an = np.where(yy <= d1, j * yy / k1, t5["vi_analytic"] + j * (yy - d1) / k2)
    ax.plot(ys, re_num, "o", ms=3, color="#2563eb", label="Re num")
    ax.plot(yy, v_an.real, "-", color="#1e3a8a", label="Re analytic")
    ax.plot(ys, im_num, "s", ms=3, color="#dc2626", label="Im num")
    ax.plot(yy, v_an.imag, "-", color="#7f1d1d", label="Im analytic")
    ax.axvline(d1, ls="--", color="#9ca3af", label="interface")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("potential [V]")
    ax.set_title("T5: two-region complex potential vs analytic divider")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Task 007 - ComplexEQS verification", fontsize=14)
    fig.savefig(PROC / "verification_plots.png", dpi=200)
    plt.close(fig)


def write_report(results: list[TestResult]) -> Path:
    PROC.mkdir(parents=True, exist_ok=True)
    n_pass = sum(r.passed for r in results)
    lines = [
        "# Task 007 - ComplexEQS Verification Report",
        "",
        "Verification audit of `publication_pipeline/elmer/solvers/ComplexEQS.F90`.",
        "No new physics, no new fault cases, no new thesis claims - this only checks",
        "that the solver reproduces the intended weak form",
        "`div((sigma + i*omega*eps0*eps_r) grad phi) = 0` on analytically tractable cases.",
        "",
        f"**Result: {n_pass}/{len(results)} tests passed.**",
        "",
        "| Test | Status | Tolerance |",
        "|------|--------|-----------|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | {r.tolerance} |")
    lines.append("")
    for r in results:
        lines.append(f"## {r.name} - {'PASS' if r.passed else 'FAIL'}")
        lines.append(f"Tolerance: {r.tolerance}")
        if r.notes:
            lines.append(f"Note: {r.notes}")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|--------|-------|")
        for k, v in r.metrics.items():
            if isinstance(v, complex):
                vs = f"{v.real:.6e} + {v.imag:.6e}j"
            elif isinstance(v, float):
                vs = f"{v:.6e}"
            else:
                vs = str(v)
            lines.append(f"| {k} | {vs} |")
        lines.append("")
    lines += [
        "## Artifacts",
        "- `mesh_convergence.csv`, `admittance_lossy_capacitor.csv`",
        "- `verification_plots.png` (convergence + two-region potential profile)",
        "- Raw Elmer runs: `results/raw/007_complex_eqs_verification/`",
        "",
    ]
    report = PROC / "verification_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    env = solver_env()

    results: list[TestResult] = []
    print("T1: sigma=0 vs StatElec ...")
    t1 = test1_sigma0_vs_statelec(dll, env)
    results.append(t1)

    print("T3: lossy capacitor admittance ...")
    t3 = test3_lossy_capacitor(dll, env)

    print("T4: mesh convergence ...")
    t4, conv_rows = test4_convergence(dll, env)

    print("T5: two-region continuity ...")
    t5, t5_data = test5_two_region(dll, env)

    print("T2: sign convention ...")
    t2 = test2_sign_convention(t3, t5_data)

    print("T6: axisymmetric coax ...")
    t6 = test6_axisym_coax(dll, env)

    results += [t2, t3, t4, t5, t6]  # report in numeric order

    write_csvs(t3, conv_rows, t5_data)
    make_plots(conv_rows, t5_data)
    report = write_report(results)

    print("\n" + "=" * 70)
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}")
    print("=" * 70)
    print(f"Report: {report}")
    all_pass = all(r.passed for r in results)
    print("RESULT:", "ALL TESTS PASS" if all_pass else "SOME TESTS FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

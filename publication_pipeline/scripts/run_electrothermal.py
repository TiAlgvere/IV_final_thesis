"""Task 021 - one-way electrothermal coupling on the representative DDB-123.

For each case: solve the axisymmetric ComplexEQS -> extract the electrical loss as a
heat source (volumetric dielectric loss q''' = 0.5*sigma_eff*|E|^2, and creepage
SURFACE loss 0.5*sigma_s*|grad_s phi|^2 for pollution) -> solve steady heat
conduction on the SOLID sub-domain (thermal_solver, air = convective BC) -> record
total loss power, max temperature rise, hot-spot location and a thermal-risk band.

The heat source reuses the exact EQS loss integrands, so the thermal layer is
energy-consistent with the diagnostic observables (volume_leak / surface_leak):
  p_vol  = 0.5 * volume_leak  * U0 ,  p_surf = 0.5 * surface_leak * U0 .

No solver / fault-physics change. Writes results/processed/021_electrothermal/
thermal_summary.csv (+ per-case T-field npz) for analyze_electrothermal.py.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
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
from cvt_observables import compute_cvt_observables  # noqa: E402
from run_cvt import OMEGA, U0, build_material_table, ensure_solver, material_of_body, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402
from thermal_solver import boundary_edges, solve_axisym_heat  # noqa: E402

PROC = ROOT / "results" / "processed" / "021_electrothermal"
RAW = ROOT / "results" / "raw" / "021_electrothermal"

# ---- documented thermal assumptions (expose; revise with project data) ----
T_AMB = 40.0                 # C, hot-day operating ambient
H_CONV = 10.0                # W/m2K, natural convection on the solid-air surface
K_THERMAL = {"metal": 200.0, "element": 0.25, "porcelain": 1.5, "oil": 0.13}  # W/mK
INSUL_HOTSPOT_C = 105.0      # C, oil/paper hot-spot limit (IEC-class guidance)

EPS0 = 8.8541878128e-12


def risk_band(dt_max: float, t_abs: float) -> str:
    if t_abs > INSUL_HOTSPOT_C:
        return "critical (>hot-spot limit)"
    if dt_max > 30.0:
        return "high"
    if dt_max > 10.0:
        return "elevated"
    if dt_max > 1.0:
        return "minor"
    return "negligible"


def _insulator_segments(m, pts):
    """The creepage line group (off-axis boundary with the largest min-radius)."""
    geom = m.cell_data.get("GeometryIds")
    lines = lgid = None
    for i, b in enumerate(m.cells):
        if b.type == "line":
            lines = np.asarray(b.data, dtype=int)
            lgid = np.asarray(geom[i], dtype=int).reshape(-1) if geom is not None else None
            break
    if lines is None or lgid is None:
        return np.empty((0, 2), int)
    best_g, best_minr = None, -1.0
    for g in np.unique(lgid):
        seg = lines[lgid == g]
        mr = float(pts[seg.reshape(-1), 0].min())
        if mr > best_minr:
            best_minr, best_g = mr, g
    return lines[lgid == best_g]


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    p = CVTParams()
    eps = p.element_epsr_eff()
    c1 = [f"element_{k}" for k in range(1, p.tap_disc_index + 1)]
    c2 = [f"element_{k}" for k in range(p.tap_disc_index + 1, p.n_elements + 1)]

    msh = RAW / "cvt.msh"
    build_cvt_ddb123(msh, p, rounded=True)
    ep = RAW / "elmer"; ep.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(resolve_elmer_grid()), "14", "2", str(msh), "-out", "mesh"],
                   cwd=str(ep), capture_output=True, text=True, check=True)
    elmer_mesh = ep / "mesh"
    ids = _parse_mesh_names(elmer_mesh / "mesh.names")
    bodies = [(bid, n) for n, bid in ids["bodies"].items()]
    tap_id = ids["bodies"][f"foil_{p.tap_disc_index}"]
    hv, gnd = ids["boundaries"]["hv_electrode"], ids["boundaries"]["ground_electrode"]
    ff, ins = ids["boundaries"].get("farfield"), ids["boundaries"].get("insulator_surface")

    # body id -> material class and thermal k (VTU GeometryIds use the body ids)
    id_to_class = {bid: material_of_body(n) for n, bid in ids["bodies"].items()}
    air_id = ids["bodies"]["air"]

    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    def base_materials():
        return build_material_table(eps), {n: material_of_body(n) for n in ids["bodies"]}

    def solve_eqs(tag, mat, bm, sigma_s=0.0):
        pdir = RAW / tag
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(elmer_mesh, pdir / "mesh", dirs_exist_ok=True)
        shutil.copy2(dll, pdir / dll.name)
        mat_index = {m_: i + 1 for i, m_ in enumerate(mat)}
        sif = render_sif(mat, mat_index, bm, bodies, hv, gnd, ff, ins_id=ins, sigma_s=sigma_s)
        (pdir / "case.sif").write_text(sif, encoding="utf-8")
        d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                           capture_output=True, text=True, env=env)
        if d.returncode != 0:
            print(d.stdout[-1200:]); raise RuntimeError(f"EQS solve failed {tag}")
        vtu = sorted(pdir.glob("**/case*.vtu"))[-1]
        be = {bid: mat[bm[n]][0] for n, bid in ids["bodies"].items()}
        bs = {bid: sigma_eff(*mat[bm[n]]) for n, bid in ids["bodies"].items()}
        obs = compute_cvt_observables(vtu, u0=U0, omega=OMEGA, body_epsr=be, body_sigma=bs,
                                      tap_body_id=tap_id, sigma_s=sigma_s)
        return vtu, bs, obs

    def thermal(vtu, body_sigma, sigma_s):
        """Build heat source from the EQS VTU and solve the steady T field."""
        m = meshio.read(str(vtu))
        pts = np.asarray(m.points, dtype=float)[:, :2]
        pd = {k.lower(): np.asarray(v, float).reshape(-1) for k, v in m.point_data.items()}
        re, im = pd["potential re"], pd["potential im"]
        tris = gids = None
        for i, b in enumerate(m.cells):
            if b.type in {"triangle", "triangle6"}:
                tris = np.asarray(b.data, int)[:, :3]
                gids = np.asarray(m.cell_data["GeometryIds"][i], int).reshape(-1)
                break

        # per-element |E|^2 (axisymmetric gradient in r,z)
        x1, x2, x3 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        det = (x2[:, 0]-x1[:, 0])*(x3[:, 1]-x1[:, 1]) - (x3[:, 0]-x1[:, 0])*(x2[:, 1]-x1[:, 1])
        bb = np.column_stack([x2[:, 1]-x3[:, 1], x3[:, 1]-x1[:, 1], x1[:, 1]-x2[:, 1]])
        cc = np.column_stack([x3[:, 0]-x2[:, 0], x1[:, 0]-x3[:, 0], x2[:, 0]-x1[:, 0]])
        gx_re = (bb*re[tris]).sum(1)/det; gy_re = (cc*re[tris]).sum(1)/det
        gx_im = (bb*im[tris]).sum(1)/det; gy_im = (cc*im[tris]).sum(1)/det
        e2 = gx_re**2 + gy_re**2 + gx_im**2 + gy_im**2

        cls = np.array([id_to_class[g] for g in gids])
        sig = np.array([body_sigma[g] for g in gids])
        # volumetric loss density 0.5*sigma_eff*|E|^2; zero in (numerical) metals & air
        q_tri = 0.5 * sig * e2
        q_tri[(cls == "metal") | (cls == "air")] = 0.0
        k_tri = np.array([K_THERMAL.get(cl, 0.0) for cl in cls])

        solid = cls != "air"
        st, sk, sq, sg = tris[solid], k_tri[solid], q_tri[solid], gids[solid]
        # solid boundary edges; drop the symmetry axis (r ~ 0), keep solid-air -> Robin
        be_solid = boundary_edges(st)
        rmid = 0.5 * (pts[be_solid[:, 0], 0] + pts[be_solid[:, 1], 0])
        robin = be_solid[rmid > 1e-6]

        # creepage surface heat source (pollution) on the insulator line group
        surf_edges = surf_pe = None
        p_surf = 0.0
        if sigma_s > 0.0:
            seg = _insulator_segments(m, pts)
            if len(seg):
                L = np.hypot(pts[seg[:, 1], 0]-pts[seg[:, 0], 0], pts[seg[:, 1], 1]-pts[seg[:, 0], 1])
                rm = 0.5 * (pts[seg[:, 0], 0] + pts[seg[:, 1], 0])
                dre = re[seg[:, 1]] - re[seg[:, 0]]; dim = im[seg[:, 1]] - im[seg[:, 0]]
                surf_pe = 0.5 * sigma_s * (dre**2 + dim**2) / L * 2.0 * np.pi * rm  # W per edge
                surf_edges = seg
                p_surf = float(surf_pe.sum())

        T = solve_axisym_heat(pts, st, sk, sq, h=H_CONV, robin_edges=robin,
                              surf_edges=surf_edges, surf_power_edge=surf_pe, T_amb=0.0)
        # totals
        r_c = (x1[:, 0]+x2[:, 0]+x3[:, 0])/3.0
        dvol = np.abs(det)/2.0 * 2.0*np.pi*r_c
        p_vol = float((q_tri*dvol)[solid].sum())
        node_body = np.zeros(pts.shape[0], int)
        node_body[st.reshape(-1)] = np.repeat(sg, 3)
        imax = int(np.nanargmax(T))
        return {"pts": pts, "tris": st, "gids": sg, "T": T,
                "dt_max": float(np.nanmax(T)), "hot_rz": pts[imax].tolist(),
                "hot_body": next(n for n, b in ids["bodies"].items() if b == node_body[imax]),
                "p_vol": p_vol, "p_surf": p_surf}

    # ---- case list ----
    def cap(fac, side):
        mat, bm = base_materials(); mat[f"element_{side}"] = (eps*fac, 0.002, 0.0)
        for b in (c1 if side == "c1" else c2):
            bm[b] = f"element_{side}"
        return mat, bm

    def loss(td, side):
        mat, bm = base_materials(); mat[f"element_{side}"] = (eps, td, 0.0)
        for b in (c1 if side == "c1" else c2):
            bm[b] = f"element_{side}"
        return mat, bm

    def short_full():
        mat, bm = base_materials(); mat["element_short"] = (eps, 0.0, 1.0)
        bm["element_5"] = "element_short"
        return mat, bm

    cases = [("healthy", "healthy", 0.0, *base_materials(), 0.0)]
    for td in (0.005, 0.01, 0.02, 0.05, 0.10):
        cases.append((f"C1_loss_{td:g}", "C1_loss", td, *loss(td, "c1"), 0.0))
    cases.append(("C2_loss_0.05", "C2_loss", 0.05, *loss(0.05, "c2"), 0.0))
    for ss in (1e-8, 1e-7, 1e-6):
        cases.append((f"pollution_{ss:g}", "pollution", ss, *base_materials(), ss))
    cases.append(("C1_cap_20", "C1_cap", 20.0, *cap(1.20, "c1"), 0.0))
    cases.append(("disc_short", "disc_short", 1.0, *short_full(), 0.0))

    rows = []
    for tag, family, sev, mat, bm, sigma_s in cases:
        vtu, bs, obs = solve_eqs(tag, mat, bm, sigma_s=sigma_s)
        th = thermal(vtu, bs, sigma_s)
        t_abs = T_AMB + th["dt_max"]
        rows.append({
            "case": tag, "family": family, "severity": sev,
            "p_vol_W": th["p_vol"], "p_surf_W": th["p_surf"],
            "p_total_W": th["p_vol"] + th["p_surf"],
            "dt_max_K": th["dt_max"], "T_abs_C": t_abs,
            "hot_r": th["hot_rz"][0], "hot_z": th["hot_rz"][1], "hot_body": th["hot_body"],
            "risk": risk_band(th["dt_max"], t_abs),
            "dvtap_pct": obs["vtap_abs"], "tan_delta": obs["tan_delta"],
            "surface_leak": obs["surface_leak"], "volume_leak": obs["volume_leak"]})
        np.savez(PROC / f"Tfield_{tag}.npz", pts=th["pts"], tris=th["tris"],
                 gids=th["gids"], T=th["T"])
        print(f"{tag:18s} p_tot={th['p_vol']+th['p_surf']:8.2f} W  dT_max={th['dt_max']:7.2f} K "
              f" T_abs={t_abs:6.1f} C  hot={th['hot_body']:14s} risk={rows[-1]['risk']}")

    cols = ["case", "family", "severity", "p_vol_W", "p_surf_W", "p_total_W", "dt_max_K",
            "T_abs_C", "hot_r", "hot_z", "hot_body", "risk", "tan_delta", "surface_leak",
            "volume_leak"]
    with (PROC / "thermal_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (PROC / "thermal_config.json").write_text(json.dumps(
        {"T_amb_C": T_AMB, "h_conv": H_CONV, "k_thermal": K_THERMAL,
         "insul_hotspot_C": INSUL_HOTSPOT_C, "U0": U0}, indent=2), encoding="utf-8")
    print(f"\nWrote {PROC / 'thermal_summary.csv'} ({len(rows)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

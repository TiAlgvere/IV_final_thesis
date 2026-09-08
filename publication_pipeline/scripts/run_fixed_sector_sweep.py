"""Task 027 - True-3D fixed-localization surface-conductance INTENSITY sweep.

Fixed azimuthal pollution sector (localized surface-conductance patch) of constant area;
sweep the surface conductance sigma_s and watch how the electrical diagnostics and the
surface loss-power FIELD evolve. This prepares the heat-input field for later electrothermal
coupling. This is NOT a streamer / discharge / breakdown task - it is localized
surface-conductance intensity validation only.

Method (all accepted from prior tasks, reused unchanged):
  - geometry/mesh : accepted Task 025 v5 true-3D DDB-123 solve mesh (read-only).
  - sector        : MATC theta-window on insulator_surface (Task 026), theta in [0, angle].
  - solver        : ComplexEQS (nabla.((sigma+jw eps) grad phi)=0), BiCGStabl(4)+ILU2
                    iterative (UMFPack direct is unavailable on this high-contrast 3D system,
                    Tasks 024/025/026).
  - observables   : observables_3d + surface-loss integration (Task 026).

PHASOR CONVENTION (critical for the power numbers):
  U0 = Um/sqrt(3) = 71.0 kV is an RMS phasor amplitude (run_cvt.U0, "phase-to-earth RMS").
  With RMS phasors the time-average dissipated power is  P = integral sigma_s |grad_s phi|^2 dS
  with NO 1/2 factor (the 1/2 in the task statement is for PEAK phasors). The equivalent
  leakage current is  surface_leak = P / U0 , so  P_from_current = U0 * surface_leak = P_direct
  identically. The surface loss-density field is  q_s = sigma_s |grad_s phi|^2  [W/m^2] (RMS).

Primary: 30 deg full-height sector, 12-value sigma_s sweep.
Optional: 10 deg sector, gated on a dedicated area/triangle-count/edge-quality audit.

Outputs (do NOT overwrite Task 025/026):
  results/raw/027_fixed_sector_sigma_sweep/        (mesh, run dirs, q_s VTUs, case copies)
  results/processed/027_fixed_sector_sigma_sweep/  (sweep.json, mesh_audit.json, report)
sweep.json is rewritten after EVERY case so progress is visible while running.
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
import pyvista as pv  # noqa: E402

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123_3d_v4 import MAT_OF_V4  # noqa: E402
from run_cvt import OMEGA, U0, build_material_table, ensure_solver, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402
from run_true3d import observables_3d  # noqa: E402
from run_true3d_sector import mesh_audit, parse_conv, render_sif  # noqa: E402

pv.OFF_SCREEN = True
PROC = ROOT / "results" / "processed" / "027_fixed_sector_sigma_sweep"
RAW = ROOT / "results" / "raw" / "027_fixed_sector_sigma_sweep"
SRC_MSH = ROOT / "results" / "raw" / "025_insulator_mesh_remediation" / "cvt3d_v5_solve.msh"
INS_RZ = (0.11, 0.17, 0.557, 1.605)   # creepage-surface filter (r_lo,r_hi,z_lo,z_hi)

SIGMA_FULL = [0.0, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5]
SIGMA_RED = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
ANGLE_PRIMARY = 30
ANGLE_OPT = 10
QS_SAVE_SIGMAS = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]   # weak..severe: save q_s field VTUs


def _insulator_tris(vtu):
    """Insulator-surface triangles from an Elmer case VTU: (points, tris, theta_deg, area,
    nhat, gsq) where gsq = |grad_s phi|^2 (Re^2+Im^2 of surface gradient)."""
    m = meshio.read(str(vtu))
    pts = np.asarray(m.points, float)
    pd = {k.lower(): np.asarray(v, float).reshape(-1) for k, v in m.point_data.items()}
    re, im = pd["potential re"], pd["potential im"]
    tris = next((np.asarray(cb.data, int)[:, :3] for cb in m.cells if cb.type.startswith("triangle")), None)
    p = pts[tris]; cen = p.mean(1)
    rc = np.hypot(cen[:, 0], cen[:, 1]); zc = cen[:, 2]
    th = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])) % 360.0
    on = (rc >= INS_RZ[0]) & (rc <= INS_RZ[1]) & (zc >= INS_RZ[2]) & (zc <= INS_RZ[3])
    tris, p, th, cen = tris[on], p[on], th[on], cen[on]
    e1 = p[:, 1]-p[:, 0]; e2 = p[:, 2]-p[:, 0]
    nrm = np.cross(e1, e2); area = 0.5*np.linalg.norm(nrm, axis=1)
    nhat = nrm/np.linalg.norm(nrm, axis=1)[:, None]
    M = np.stack([e1, e2, nhat], 1); gsq = np.zeros(len(tris))
    for f in (re, im):
        b = np.stack([f[tris[:, 1]]-f[tris[:, 0]], f[tris[:, 2]]-f[tris[:, 0]], np.zeros(len(tris))], 1)[:, :, None]
        gsq += (np.linalg.solve(M, b)[:, :, 0]**2).sum(1)
    return pts, tris, th, area, cen, gsq


def surface_field(vtu, sigma_s, angle, save_vtu=None):
    """RMS-convention surface loss. Returns P_direct, areas, max q_s + location, and (optional)
    saves a visualization-ready insulator-surface q_s VTU (q_s=0 outside the polluted patch)."""
    pts, tris, th, area, cen, gsq = _insulator_tris(vtu)
    inwin = th <= angle + 1e-6 if angle < 360 else np.ones(len(th), bool)
    qs = sigma_s * gsq                       # W/m^2 (RMS, no 1/2)
    qs_patch = np.where(inwin, qs, 0.0)
    p_direct = float(np.sum(qs[inwin] * area[inwin]))      # = integral q_s dS over patch
    parea = float(area[inwin].sum()); tarea = float(area.sum())
    out = {"P_surface_direct_W": p_direct, "polluted_area_m2": parea,
           "total_insulator_area_m2": tarea, "area_fraction": parea/tarea,
           "expected_fraction": angle/360.0}
    if inwin.any() and sigma_s > 0:
        i = int(np.argmax(qs_patch))
        c = cen[i]
        out["max_qs_Wm2"] = float(qs_patch[i])
        out["max_qs_loc_rzt"] = [float(math.hypot(c[0], c[1])), float(c[2]),
                                 float(math.degrees(math.atan2(c[1], c[0])) % 360)]
    else:
        out["max_qs_Wm2"] = 0.0; out["max_qs_loc_rzt"] = None
    if save_vtu is not None:
        used = np.unique(tris); remap = -np.ones(pts.shape[0], int); remap[used] = np.arange(used.size)
        faces = np.hstack([np.full((len(tris), 1), 3), remap[tris]]).ravel()
        poly = pv.PolyData(pts[used], faces)
        poly.cell_data["q_s_Wm2"] = qs_patch
        poly.cell_data["sigma_s"] = np.where(inwin, sigma_s, 0.0)
        poly.cast_to_unstructured_grid().save(str(save_vtu))
    return out


def audit_angle(msh, angle, edge_tol_deg=3.0):
    """Dedicated patch audit for a sector [0, angle]. Reports area fraction, triangle count,
    and TWO separate edge bands: the NEW cut edge (theta ~ angle) and the pre-existing
    revolution seam (theta ~ 0/360, shared by every [0,angle] sector incl. accepted Task 026
    30deg). The gate decision keys on the NEW cut edge; the seam is reported descriptively."""
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(str(msh))
        name3 = {(d, t): gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()}
        nt, nc, _ = gmsh.model.mesh.getNodes(); nt = nt.astype(np.int64)
        coord = np.zeros((int(nt.max())+1, 3)); coord[nt] = nc.reshape(-1, 3)
        ins_tag = next((t for (d, t), nm in name3.items() if nm == "insulator_surface" and d == 2), None)
        tri_tags, tri_conn = [], []
        for ent in gmsh.model.getEntitiesForPhysicalGroup(2, ins_tag):
            ets, en = gmsh.model.mesh.getElements(2, ent)[1:]
            if ets:
                tri_tags.append(ets[0].astype(np.int64)); tri_conn.append(en[0].reshape(-1, 3).astype(np.int64))
        tri_tags = np.concatenate(tri_tags); tri_conn = np.vstack(tri_conn)
        tq = np.array(gmsh.model.mesh.getElementQualities(list(tri_tags), "minSICN"))
        P = coord[tri_conn]; cen = P.mean(1)
        th = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])) % 360.0
        ar = 0.5*np.linalg.norm(np.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0]), axis=1)
        inwin = th <= angle + 1e-6

        def band(mask):
            return {"n": int(mask.sum()), "min": float(tq[mask].min()),
                    "p5": float(np.percentile(tq[mask], 5)), "mean": float(tq[mask].mean())}
        cut = np.abs(th - angle) <= edge_tol_deg
        seam = np.minimum(th, 360.0 - th) <= edge_tol_deg
        worst = np.argsort(tq)[:200]
        return {"angle_deg": angle, "n_tris_patch": int(inwin.sum()),
                "area_fraction": float(ar[inwin].sum()/ar.sum()), "expected_fraction": angle/360.0,
                "cut_edge": band(cut), "seam_edge": band(seam),
                "insulator_p5": float(np.percentile(tq, 5)),
                "worst200_on_cut_edge": int(cut[worst].sum()),
                "worst200_on_seam": int(seam[worst].sum())}
    finally:
        gmsh.finalize()


def build_runtime():
    """ElmerGrid-convert the v5 mesh into Task 027's own folder; build material/body maps."""
    dll = ensure_solver()
    env = os.environ.copy(); home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
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
    return dict(env=env, pdir=pdir, mat=mat, mat_index=mat_index, bm=bm, bodies=bodies,
                tap_id=tap_id, hv=hv, gnd=gnd, ff=ff, ins=ins, be=be, bs=bs)


def conv_from_stdout(out):
    """Robust BiCGStabl convergence: with 'Residual Output = 1' Elmer prints one '<it> <res>'
    line per iteration; take the last. Also capture the steady-state ComputeChange norm."""
    it = re.findall(r"^\s*(\d+)\s+([0-9.][0-9.Ee+-]*)\s*$", out, re.M)
    cc = re.findall(r"ComputeChange:\s*SS .*?:\s*\(\s*([0-9.Ee+-]+)\s+([0-9.Ee+-]+)", out)
    return {"iterations": int(it[-1][0]) if it else None,
            "final_residual": float(it[-1][1]) if it else None,
            "ss_norm": float(cc[-1][0]) if cc else None}


def solve(rt, tag, sigma_s, angle):
    """Iterative solve only (direct UMFPack unavailable on this 3D system). Residual Output is
    forced to 1 so every iteration's residual is printed and convergence is always recorded."""
    pdir = rt["pdir"]
    sif = render_sif(rt["mat"], rt["mat_index"], rt["bm"], rt["bodies"], rt["hv"], rt["gnd"],
                     rt["ff"], rt["ins"], sigma_s=sigma_s, angle=angle, direct=False)
    sif = sif.replace("Linear System Residual Output = 100", "Linear System Residual Output = 1")
    (pdir / "case.sif").write_text(sif, encoding="utf-8")
    t0 = time.time()
    d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                       capture_output=True, text=True, env=rt["env"])
    vtus = sorted(pdir.glob("**/case*.vtu"))
    if d.returncode != 0 or not vtus:
        (pdir / f"FAIL_{tag}.txt").write_text(d.stdout[-4000:], encoding="utf-8")
        raise RuntimeError(f"solve failed {tag}")
    return vtus[-1], time.time()-t0, conv_from_stdout(d.stdout)


def run_case(rt, tag, angle, ss, qdir):
    vtu, dt, cv = solve(rt, tag, ss, angle)
    o = observables_3d(vtu, rt["be"], rt["bs"], rt["tap_id"])
    ie = o["C_pF"]*1e-12*U0**2; isv = o["tan_delta"]*OMEGA*ie     # volume loss integral
    row = {"case": tag, "angle_deg": angle, "sigma_s": ss,
           "solver": "iterative BiCGStabl(4)+ILU2", "iterations": cv["iterations"],
           "final_residual": cv["final_residual"], "ss_norm": cv.get("ss_norm"),
           "converged": cv["final_residual"] is not None and cv["final_residual"] <= 1e-8,
           "solve_s": round(dt, 1),
           "C_pF": o["C_pF"], "vtap_abs": o["vtap_abs"], "divider_ratio": o["divider_ratio"],
           "phase_mdeg": o["vtap_phase_mdeg"], "phase_reliable": False,
           "emax": o["emax"], "emax_rz": o["emax_rz"],
           "emax_on_insulator": bool(INS_RZ[0] <= o["emax_rz"][0] <= INS_RZ[1]
                                     and INS_RZ[2] <= o["emax_rz"][1] <= INS_RZ[3]),
           "volume_leak": isv/U0}
    save = (qdir / f"qs_{angle}deg_s{ss:.0e}.vtu") if (ss in QS_SAVE_SIGMAS) else None
    sf = surface_field(vtu, ss, angle, save_vtu=save)
    p_dir = sf["P_surface_direct_W"]
    surf_leak = p_dir / U0
    row.update({"surface_leak": surf_leak,
                "P_surface_direct_W": p_dir,
                "P_surface_from_current_W": U0 * surf_leak,   # = p_dir (RMS, identical)
                "tan_delta": (isv + p_dir) / (OMEGA * ie),
                "max_qs_Wm2": sf["max_qs_Wm2"], "max_qs_loc_rzt": sf["max_qs_loc_rzt"],
                "polluted_area_m2": sf["polluted_area_m2"],
                "total_insulator_area_m2": sf["total_insulator_area_m2"],
                "area_fraction": sf["area_fraction"], "expected_fraction": sf["expected_fraction"],
                "qs_vtu": save.name if save else None})
    res_s = f"{cv['final_residual']:.1e}" if cv["final_residual"] is not None else "n/a"
    print(f"  {tag}: sig={ss:.0e} C={o['C_pF']:.1f} ratio={o['divider_ratio']:.4f} "
          f"tand={row['tan_delta']:.3e} sleak={surf_leak:.3e} Pdir={p_dir:.3e}W "
          f"maxqs={row['max_qs_Wm2']:.2e} Emax={o['emax']:.2e}@rz{[round(x,3) for x in o['emax_rz']]} "
          f"it={cv['iterations']} res={res_s} ({dt:.0f}s)", flush=True)
    return row


def dump(state):
    (PROC / "sweep.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    qdir = RAW / "qs_fields"; qdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("Mesh audit (reusing accepted Task 025 v5 solve mesh, read-only) ...", flush=True)
    aud = mesh_audit(SRC_MSH)
    (PROC / "mesh_audit.json").write_text(json.dumps(aud, indent=2, default=str), encoding="utf-8")
    a30 = audit_angle(SRC_MSH, ANGLE_PRIMARY)
    print(f"  30deg patch: n_tris={a30['n_tris_patch']} area_frac={a30['area_fraction']:.4f} "
          f"(exp {a30['expected_fraction']:.4f}) cut-edge p5={a30['cut_edge']['p5']:.3f} "
          f"min={a30['cut_edge']['min']:.3f}; seam min={a30['seam_edge']['min']:.3f} "
          f"(pre-existing, shared w/ accepted T026 30deg)", flush=True)

    state = {"convention": {
        "U0_V": U0, "U0_kind": "RMS phase-to-earth (Um/sqrt3)", "omega": OMEGA,
        "P_surface_direct": "integral sigma_s |grad_s phi|^2 dS  (RMS: NO 1/2 factor)",
        "P_surface_from_current": "U0 * surface_leak  (RMS: NO 1/2 factor)",
        "surface_leak": "P_surface_direct / U0  (equivalent RMS in-phase leakage current)",
        "q_s": "sigma_s |grad_s phi|^2  [W/m^2]  (RMS)",
        "note": "the 1/2 in the task statement applies to PEAK phasors; U0 is RMS so it is dropped. "
                "P_from_current == P_direct identically (surface_leak is defined as P_direct/U0)."},
        "mesh": {"global_tet": aud["global_tet"], "insulator_surface": aud["insulator_surface"],
                 "patch_edge_tri": aud["patch_edge_tri"], "audit_30deg": a30},
        "sweep_30deg": [], "audit_10deg": None, "sweep_10deg": []}
    dump(state)

    # --- primary 30 deg sweep (full sigma list) ---
    print(f"\n=== 30 deg sector sweep ({len(SIGMA_FULL)} sigma values) ===", flush=True)
    for ss in SIGMA_FULL:
        tag = "clean" if ss == 0 else f"a30_s{ss:.0e}"
        state["sweep_30deg"].append(run_case(rt, tag, ANGLE_PRIMARY, ss, qdir))
        dump(state)

    # --- optional 10 deg sector, gated on its own audit ---
    a10 = audit_angle(SRC_MSH, ANGLE_OPT)
    state["audit_10deg"] = a10; dump(state)
    af_err = abs(a10["area_fraction"]/a10["expected_fraction"] - 1)
    # Gate keys on the NEW cut edge (theta~10) + area + triangle count. The theta~0 revolution
    # seam is pre-existing and shared with the accepted Task 026 30deg sector, so it is reported
    # but not treated as a NEW blocker for the 10deg candidate.
    gate = {"area_ok": af_err <= 0.05, "tris_ok": a10["n_tris_patch"] >= 1500,
            "cut_edge_ok": a10["cut_edge"]["min"] >= 0.30,
            "cut_edge_no_worst": a10["worst200_on_cut_edge"] == 0}
    a10["gate"] = gate; a10["passed"] = all(gate.values())
    print(f"\n10deg audit: n_tris={a10['n_tris_patch']} area_frac={a10['area_fraction']:.4f} "
          f"(exp {a10['expected_fraction']:.4f}, err {af_err*100:.1f}%) "
          f"cut-edge min={a10['cut_edge']['min']:.3f} p5={a10['cut_edge']['p5']:.3f} "
          f"worst200_on_cut={a10['worst200_on_cut_edge']}; seam min={a10['seam_edge']['min']:.3f} "
          f"-> gate {gate} PASS={a10['passed']}", flush=True)
    dump(state)

    if a10["passed"]:
        print(f"\n=== 10 deg sector sweep (reduced, {len(SIGMA_RED)} sigma values) ===", flush=True)
        for ss in SIGMA_RED:
            tag = "clean10" if ss == 0 else f"a10_s{ss:.0e}"
            state["sweep_10deg"].append(run_case(rt, tag, ANGLE_OPT, ss, qdir))
            dump(state)
    else:
        print("10 deg patch did not pass the audit gate - skipped (documented).", flush=True)

    state["total_runtime_s"] = round(time.time() - t_start, 1)
    dump(state)
    print(f"\nDone. {len(state['sweep_30deg'])} (30deg) + {len(state['sweep_10deg'])} (10deg) cases "
          f"in {state['total_runtime_s']:.0f}s -> {PROC / 'sweep.json'}", flush=True)
    return 0


if __name__ == "__main__":
    rt = build_runtime()
    raise SystemExit(main())

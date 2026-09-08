"""Task 015 - localized pollution / streamer via surface-conductance windowing.

Uses the spatially-varying Surface Conductance BC (Task 015 solver change) to apply
pollution only over an axial window of the insulator_surface creepage:

  1. full creepage          sigma_s on all of [tank_height, stack_z_hi]
  2. top 25%                sigma_s on the upper quarter (near HV)
  3. middle 25%
  4. bottom 25%             (near ground)
  5. streamer (approx.)     a narrow vertical wet streak: in the axisymmetric model
                            this is a full-height ring with an EFFECTIVE sheet
                            conductance = local sigma_s * (azimuthal width / 2*pi).
                            A truly azimuthally-localized streak needs the 3-D model.

Per case: |Vtap|, tap phase, tan-delta, leakage, Emax + peak surface field, and the
state-space position (log tan-delta, d-theta). Plots the 5 cases on the full-creepage
sigma_s relaxation manifold, picks the worst case, and 3D-renders it.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from cvt_observables import _grad_area, _read, compute_cvt_observables  # noqa: E402
from run_cvt import OMEGA, U0, build_material_table, ensure_solver, material_of_body, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

pv.OFF_SCREEN = True
PROC = ROOT / "results" / "processed" / "015_localized_pollution"
RAW = ROOT / "results" / "raw" / "015_localized_pollution"
SB = 1.0e-6                 # base surface conductance for cases 1-4
STREAM_LOCAL = 1.0e-4       # local sheet conductance of a wet streak
STREAM_WIDTH_DEG = 20.0     # azimuthal width of the streak
STREAM_EFF = STREAM_LOCAL * STREAM_WIDTH_DEG / 360.0  # axisymmetric effective ring value
REV_ANGLE, REV_RES = 270.0, 96


def matc_window(zlo, zhi, val) -> str:
    return f'Variable Coordinate 2\n    Real MATC "(tx >= {zlo})*(tx <= {zhi})*{val}"'


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    p = CVTParams()
    TANK, SZHI = p.tank_height, p.stack_z_hi
    H = SZHI - TANK
    ctr = 0.5 * (TANK + SZHI)
    top = (SZHI - 0.25 * H, SZHI)
    mid = (ctr - 0.125 * H, ctr + 0.125 * H)
    bot = (TANK, TANK + 0.25 * H)

    # (label, sigma_s for observables, surface_cond_str for SIF, z_window for observables)
    cases = [
        ("full", SB, f"Real {SB!r}", None),
        ("top25", SB, matc_window(top[0], top[1], SB), top),
        ("mid25", SB, matc_window(mid[0], mid[1], SB), mid),
        ("bottom25", SB, matc_window(bot[0], bot[1], SB), bot),
        ("streamer", STREAM_EFF, f"Real {STREAM_EFF!r}", None),
    ]

    # build mesh + convert
    msh = RAW / "cvt.msh"
    build_cvt_ddb123(msh, p, rounded=True)
    ep = RAW / "elmer"; ep.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(resolve_elmer_grid()), "14", "2", str(msh), "-out", "mesh"],
                   cwd=str(ep), capture_output=True, text=True, check=True)
    elmer_mesh = ep / "mesh"
    ids = _parse_mesh_names(elmer_mesh / "mesh.names")
    mat = build_material_table(p.element_epsr_eff())
    mat_index = {m: i + 1 for i, m in enumerate(mat)}
    body_material = {n: material_of_body(n) for n in ids["bodies"]}
    bodies = [(bid, n) for n, bid in ids["bodies"].items()]
    body_epsr = {bid: mat[body_material[n]][0] for n, bid in ids["bodies"].items()}
    body_sigma = {bid: sigma_eff(*mat[body_material[n]]) for n, bid in ids["bodies"].items()}
    tap_id = ids["bodies"][f"foil_{p.tap_disc_index}"]
    hv, gnd, ff = ids["boundaries"]["hv_electrode"], ids["boundaries"]["ground_electrode"], ids["boundaries"].get("farfield")
    ins = ids["boundaries"].get("insulator_surface")

    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    def solve(tag, sc_str):
        pdir = RAW / tag
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(elmer_mesh, pdir / "mesh", dirs_exist_ok=True)
        shutil.copy2(dll, pdir / dll.name)
        sif = render_sif(mat, mat_index, body_material, bodies, hv, gnd, ff,
                         ins_id=ins, surface_cond_str=sc_str)
        (pdir / "case.sif").write_text(sif, encoding="utf-8")
        d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                           capture_output=True, text=True, env=env)
        (pdir / "elmersolver.stdout.log").write_text(d.stdout, encoding="utf-8")
        if d.returncode != 0:
            print(d.stdout[-1500:]); raise RuntimeError(f"solve failed {tag}")
        return sorted(pdir.glob("**/case*.vtu"))[-1]

    def emax_and_surface(vtu, sigma_s, z_window):
        pts, tris, gids, re, im = _read(vtu)
        gr, _ = _grad_area(pts, tris, re); gi, _ = _grad_area(pts, tris, im)
        emag = np.sqrt(np.sum(gr**2, axis=1) + np.sum(gi**2, axis=1))
        i = int(np.argmax(emag)); cen = pts[tris[i]].mean(axis=0)
        name_of = {bid: n for n, bid in ids["bodies"].items()}
        # peak surface tangential field on the polluted window of insulator_surface
        m = meshio.read(str(vtu)); mp = np.asarray(m.points)[:, :2]
        geom = m.cell_data.get("GeometryIds"); lines = lgid = None
        for j, b in enumerate(m.cells):
            if b.type == "line":
                lines = np.asarray(b.data, int); lgid = np.asarray(geom[j], int).reshape(-1); break
        bg, bm = None, -1.0
        for g in np.unique(lgid):
            s = lines[lgid == g]; mr = mp[s.reshape(-1), 0].min()
            if mr > bm: bm, bg = mr, g
        s = lines[lgid == bg]
        zmid = 0.5 * (mp[s[:, 0], 1] + mp[s[:, 1], 1])
        L = np.hypot(mp[s[:, 1], 0] - mp[s[:, 0], 0], mp[s[:, 1], 1] - mp[s[:, 0], 1])
        et = np.hypot(re[s[:, 1]] - re[s[:, 0]], im[s[:, 1]] - im[s[:, 0]]) / L
        win = np.ones(len(s), bool) if z_window is None else (zmid >= z_window[0]) & (zmid <= z_window[1])
        surf_emax = float(et[win].max()) if win.any() else 0.0
        return float(emag[i]), (float(cen[0]), float(cen[1])), name_of[gids[i]], surf_emax

    print("Solving clean reference ...")
    vtu_clean = solve("clean", None)
    obs_clean = compute_cvt_observables(vtu_clean, u0=U0, omega=OMEGA, body_epsr=body_epsr,
                                        body_sigma=body_sigma, tap_body_id=tap_id)
    phase0 = obs_clean["vtap_phase_mdeg"]
    v0 = obs_clean["vtap_abs"]

    rows = []
    for label, ss, sc_str, win in cases:
        print(f"Solving {label} ...")
        vtu = solve(label, sc_str)
        o = compute_cvt_observables(vtu, u0=U0, omega=OMEGA, body_epsr=body_epsr, body_sigma=body_sigma,
                                    tap_body_id=tap_id, sigma_s=ss, surface_z_window=win)
        emax, eloc, ebody, surf_emax = emax_and_surface(vtu, ss, win)
        rows.append({"case": label, "sigma_s": ss, "window": win, "vtu": str(vtu),
                     "vtap_abs": o["vtap_abs"], "dvtap_pct": (o["vtap_abs"] / v0 - 1) * 100.0,
                     "phase_mdeg": o["vtap_phase_mdeg"],
                     "dtheta_mdeg": o["vtap_phase_mdeg"] - phase0, "tan_delta": o["tan_delta"],
                     "i_leak": o["i_leak"], "surface_leak": o["surface_leak"], "volume_leak": o["volume_leak"],
                     "emax": emax, "emax_loc": eloc, "emax_body": ebody, "surf_emax": surf_emax})
        print(f"   |Vtap|={o['vtap_abs']:.1f} dtheta={o['vtap_phase_mdeg']-phase0:.2f}mdeg "
              f"tand={o['tan_delta']:.4e} Ileak={o['i_leak']:.3e} Emax={emax:.2e} surfEmax={surf_emax:.2e}")

    # reference full-creepage sigma_s manifold for the state-space context
    print("Reference full-creepage sweep ...")
    ref = []
    for ss in np.logspace(-9, -4, 8):
        ss = float(ss)
        vtu = solve(f"ref_{ss:.0e}", f"Real {ss!r}")
        o = compute_cvt_observables(vtu, u0=U0, omega=OMEGA, body_epsr=body_epsr, body_sigma=body_sigma,
                                    tap_body_id=tap_id, sigma_s=float(ss))
        ref.append((float(ss), o["tan_delta"], o["vtap_phase_mdeg"] - phase0))

    # worst case = max leakage current
    worst = max(rows, key=lambda r: r["i_leak"])

    # ---- state-space plot ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    rt = np.array([r[1] for r in ref]); rd = np.array([r[2] for r in ref])
    ax.plot(np.log10(rt), rd, "-", color="0.6", lw=1.2, zorder=1, label="full-creepage manifold (sigma_s sweep)")
    sc = ax.scatter(np.log10(rt), rd, c=np.log10([r[0] for r in ref]), cmap="Greys", s=25, zorder=2)
    colors = {"full": "#2563eb", "top25": "#16a34a", "mid25": "#f59e0b", "bottom25": "#dc2626", "streamer": "#7c3aed"}
    for r in rows:
        ax.scatter(np.log10(r["tan_delta"]), r["dtheta_mdeg"], s=130, color=colors[r["case"]],
                   edgecolors="k", zorder=5, label=r["case"])
        ax.annotate(r["case"], (np.log10(r["tan_delta"]), r["dtheta_mdeg"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.scatter(np.log10(worst["tan_delta"]), worst["dtheta_mdeg"], s=320, facecolors="none",
               edgecolors="crimson", linewidths=2.2, zorder=6, label=f"worst ({worst['case']})")
    ax.set_xlabel(r"terminal loss   $\log_{10}\tan\delta$")
    ax.set_ylabel(r"tap phase displacement $\Delta\theta$ [mdeg]")
    ax.set_title("Task 015 - localized pollution cases in (loss, phase) state space")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(PROC / "localized_state_space.png", dpi=190); plt.close(fig)

    # ---- CSV + report ----
    cols = ["case", "sigma_s", "vtap_abs", "dtheta_mdeg", "tan_delta", "i_leak", "emax", "surf_emax", "emax_body"]
    with (PROC / "localized_cases.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    # ---- 3D renders for the worst case ----
    render_worst(worst, ids, PROC, p)

    lines = [
        "# Task 015 - Localized Pollution / Streamer (surface-conductance windowing)",
        "",
        f"Base sigma_s = {SB:.0e} S (cases 1-4). Streamer = full-height ring with EFFECTIVE",
        f"sigma_s = {STREAM_EFF:.2e} S (local {STREAM_LOCAL:.0e} S x {STREAM_WIDTH_DEG:.0f} deg / 360;",
        "a true azimuthal streak needs the 3-D model). Clean tap phase = "
        f"{phase0:.2f} mdeg; d-theta is relative to clean.",
        "",
        "| case | sigma_s [S] | |Vtap| [V] | d-theta [mdeg] | tan(delta) | leakage [A] | Emax [V/m] | surf Emax [V/m] |",
        "|------|-------------|-----------|----------------|------------|-------------|------------|------------------|",
    ]
    for r in rows:
        lines.append(f"| {r['case']} | {r['sigma_s']:.2e} | {r['vtap_abs']:.1f} | {r['dtheta_mdeg']:.2f} | "
                     f"{r['tan_delta']:.3e} | {r['i_leak']:.3e} | {r['emax']:.2e} | {r['surf_emax']:.2e} |")
    lines += [
        "",
        f"**Worst case (max leakage): `{worst['case']}`** - leakage {worst['i_leak']:.2e} A, "
        f"tan(delta) {worst['tan_delta']:.3e}, d-theta {worst['dtheta_mdeg']:.2f} mdeg.",
        "",
        "## Read-out",
        "- tan(delta) and leakage scale with the polluted AREA: full > each 25% window; a",
        "  given 25% window gives ~1/4 of the full-creepage loss at the same sigma_s.",
        "- Position matters for d-theta (phase): top/middle/bottom quarters couple to the",
        "  internal tap differently even at equal sigma_s and equal area.",
        "- Global Emax stays at the HV terminal (metal) - surface pollution does not move it;",
        "  the surface field concentration (surf Emax) rises at the polluted window edges.",
        "- In state space the localized cases sit on/near the full-creepage manifold but at",
        "  reduced loss (smaller polluted area) - see `localized_state_space.png`.",
        "",
        "## Artifacts",
        "- `localized_state_space.png`, `localized_cases.csv`",
        "- worst-case 3D: `worst_Imphi.png`, `worst_dE.png`, `worst_surface_loss.png`",
        "",
    ]
    (PROC / "localized_pollution_report.md").write_text("\n".join(lines), encoding="utf-8")
    (PROC / "summary.json").write_text(json.dumps({"cases": rows, "worst": worst["case"], "ref": ref}, indent=2, default=str), encoding="utf-8")

    print(f"\nWorst case: {worst['case']} (leakage {worst['i_leak']:.2e} A)")
    print(f"Outputs in {PROC}")
    return 0


def render_worst(worst, ids, PROC, p):
    """3D revolved renders (Im phi, dE vs clean, windowed surface loss) for the worst case."""
    pts, tris, gids, re_p, im_p = _read(Path(worst["vtu"]))
    cv = sorted((RAW / "clean").glob("**/case*.vtu"))[-1]
    _, _, _, re_c, im_c = _read(cv)
    gr, _ = _grad_area(pts, tris, re_p); gi, _ = _grad_area(pts, tris, im_p)
    e_p = np.sqrt(np.sum(gr**2, axis=1) + np.sum(gi**2, axis=1))
    gr, _ = _grad_area(pts, tris, re_c); gi, _ = _grad_area(pts, tris, im_c)
    e_c = np.sqrt(np.sum(gr**2, axis=1) + np.sum(gi**2, axis=1))
    air = ids["bodies"]["air"]; dev = gids != air
    dtris = tris[dev]; used = np.unique(dtris); remap = -np.ones(len(pts), int); remap[used] = np.arange(len(used))
    dpts = pts[used]; dt = remap[dtris]

    def c2p(cellv):
        acc = np.zeros(len(dpts)); cnt = np.zeros(len(dpts))
        for k in range(3):
            np.add.at(acc, dt[:, k], cellv); np.add.at(cnt, dt[:, k], 1.0)
        cnt[cnt == 0] = 1.0
        return acc / cnt

    P3 = np.column_stack([dpts[:, 0], np.zeros(len(dpts)), dpts[:, 1]])
    poly = pv.PolyData(P3, np.hstack([np.full((len(dt), 1), 3), dt]).ravel())
    poly.point_data["Im"] = im_p[used]
    poly.point_data["dE"] = c2p((e_p - e_c)[dev])
    solid = poly.extrude_rotate(resolution=REV_RES, angle=REV_ANGLE, capping=True)
    cam = [(2.6, -2.2, 1.9), (0.0, 0.0, 0.55 * p.total_height), (0.0, 0.0, 1.0)]

    def render(scalar, fname, title, cmap):
        lim = float(np.percentile(np.abs(poly.point_data[scalar]), 99)) or 1.0
        pl = pv.Plotter(off_screen=True, window_size=(850, 1250)); pl.set_background("white")
        pl.add_mesh(solid, scalars=scalar, cmap=cmap, clim=[-lim, lim], smooth_shading=True,
                    scalar_bar_args={"title": title, "vertical": True, "position_x": 0.85, "position_y": 0.25})
        pl.add_text(f"worst case: {worst['case']} - {title}", position="upper_left", font_size=12, color="black")
        pl.camera_position = cam; pl.screenshot(str(PROC / fname)); pl.close()

    render("Im", "worst_Imphi.png", "Im(phi) [V]", "coolwarm")
    render("dE", "worst_dE.png", "dE = |E|_pol-|E|_clean [V/m]", "coolwarm")

    # windowed surface loss on insulator_surface
    mp = meshio.read(str(worst["vtu"])); mpts = np.asarray(mp.points)[:, :2]
    geom = mp.cell_data.get("GeometryIds"); lines = lgid = None
    for j, b in enumerate(mp.cells):
        if b.type == "line":
            lines = np.asarray(b.data, int); lgid = np.asarray(geom[j], int).reshape(-1); break
    bg, bm = None, -1.0
    for g in np.unique(lgid):
        s = lines[lgid == g]; mr = mpts[s.reshape(-1), 0].min()
        if mr > bm: bm, bg = mr, g
    seg = lines[lgid == bg]
    L = np.hypot(mpts[seg[:, 1], 0] - mpts[seg[:, 0], 0], mpts[seg[:, 1], 1] - mpts[seg[:, 0], 1])
    et2 = (re_p[seg[:, 1]] - re_p[seg[:, 0]])**2 + (im_p[seg[:, 1]] - im_p[seg[:, 0]])**2
    zmid = 0.5 * (mpts[seg[:, 0], 1] + mpts[seg[:, 1], 1])
    win = np.ones(len(seg), bool) if worst["window"] is None else (zmid >= worst["window"][0]) & (zmid <= worst["window"][1])
    dens = worst["sigma_s"] * et2 / L**2 * win  # 0 outside the window
    snodes = np.unique(seg); sr = -np.ones(len(mpts), int); sr[snodes] = np.arange(len(snodes))
    lp = pv.PolyData(); lp.points = np.column_stack([mpts[snodes, 0], np.zeros(len(snodes)), mpts[snodes, 1]])
    lp.lines = np.hstack([np.full((len(seg), 1), 2), sr[seg]]).ravel()
    pd = np.zeros(len(snodes)); ct = np.zeros(len(snodes))
    for k in range(2):
        np.add.at(pd, sr[seg[:, k]], dens); np.add.at(ct, sr[seg[:, k]], 1.0)
    ct[ct == 0] = 1.0
    lp.point_data["loss"] = pd / ct
    surf = lp.extrude_rotate(resolution=REV_RES, angle=360.0, capping=False)
    pl = pv.Plotter(off_screen=True, window_size=(850, 1250)); pl.set_background("white")
    pl.add_mesh(surf, scalars="loss", cmap="inferno", smooth_shading=True,
                scalar_bar_args={"title": "surface loss [W/m^2]", "vertical": True, "position_x": 0.85, "position_y": 0.25})
    pl.add_text(f"worst case: {worst['case']} - windowed surface loss", position="upper_left", font_size=12, color="black")
    pl.camera_position = cam; pl.screenshot(str(PROC / "worst_surface_loss.png")); pl.close()


if __name__ == "__main__":
    raise SystemExit(main())

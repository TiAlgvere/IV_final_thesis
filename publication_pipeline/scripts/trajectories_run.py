"""Task 018 (sweeps) - state-space trajectory data for the degradation-vector study.

Observable state vector x = [d|Vtap|%, d-theta mdeg, tan-delta, surface_leak A,
volume_leak A] relative to the healthy/clean baseline. We trace each fault family
as a parametric trajectory x(s):

  * pollution   : s = log10(sigma_s), full-creepage surface conductance, per-decade
  * C1/C2 cap   : s = permittivity increase factor
  * C1/C2 loss  : s = element tan-delta
  * disc short  : s = number of shorted C1 elements

Writes results/processed/018_trajectories/trajectories.json for trajectories_analyze.py.
No new physics; reuses the verified ComplexEQS solver.
"""

from __future__ import annotations

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

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from cvt_observables import compute_cvt_observables  # noqa: E402
from run_cvt import OMEGA, U0, build_material_table, ensure_solver, material_of_body, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

PROC = ROOT / "results" / "processed" / "018_trajectories"
RAW = ROOT / "results" / "raw" / "018_trajectories"

POLLUTION_SIGMA = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]   # decades
CAP_FACTORS = [1.0, 1.1, 1.2, 1.3]                        # +0..30 %
LOSS_TAND = [0.002, 0.01, 0.02, 0.05, 0.1]
SHORT_N = [0, 1, 2, 3]


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

    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    def solve(tag, mat, bm, surface_cond_str=None, sigma_s=0.0):
        pdir = RAW / tag
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(elmer_mesh, pdir / "mesh", dirs_exist_ok=True)
        shutil.copy2(dll, pdir / dll.name)
        mat_index = {m: i + 1 for i, m in enumerate(mat)}
        sif = render_sif(mat, mat_index, bm, bodies, hv, gnd, ff, ins_id=ins, surface_cond_str=surface_cond_str)
        (pdir / "case.sif").write_text(sif, encoding="utf-8")
        d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                           capture_output=True, text=True, env=env)
        if d.returncode != 0:
            print(d.stdout[-1200:]); raise RuntimeError(f"solve failed {tag}")
        vtu = sorted(pdir.glob("**/case*.vtu"))[-1]
        be = {bid: mat[bm[n]][0] for n, bid in ids["bodies"].items()}
        bs = {bid: sigma_eff(*mat[bm[n]]) for n, bid in ids["bodies"].items()}
        return compute_cvt_observables(vtu, u0=U0, omega=OMEGA, body_epsr=be, body_sigma=bs,
                                       tap_body_id=tap_id, sigma_s=sigma_s)

    def base_materials():
        return build_material_table(eps), {n: material_of_body(n) for n in ids["bodies"]}

    # baseline
    print("baseline ...")
    mh, bh = base_materials()
    o0 = solve("baseline", mh, bh)
    v0, ph0 = o0["vtap_abs"], o0["vtap_phase_mdeg"]

    def xvec(o):
        return {"dvtap_pct": (o["vtap_abs"] / v0 - 1) * 100.0,
                "dtheta_mdeg": o["vtap_phase_mdeg"] - ph0,
                "tan_delta": o["tan_delta"], "surface_leak": o["surface_leak"],
                "volume_leak": o["volume_leak"]}

    data = {"baseline": {"vtap": v0, "phase_mdeg": ph0,
                         "x": {"dvtap_pct": 0.0, "dtheta_mdeg": 0.0, "tan_delta": o0["tan_delta"],
                               "surface_leak": o0["surface_leak"], "volume_leak": o0["volume_leak"]}},
            "pollution": [], "internal": {"C1_cap": [], "C2_cap": [], "C1_loss": [], "C2_loss": [], "disc_short": []}}

    print("pollution sigma_s decades ...")
    for ss in POLLUTION_SIGMA:
        o = solve(f"poll_{ss:.0e}", mh, bh, surface_cond_str=f"Real {float(ss)!r}", sigma_s=float(ss))
        data["pollution"].append({"s": float(ss), "log10s": float(np.log10(ss)), "x": xvec(o)})
        print(f"   sigma_s={ss:.0e}: tand={o['tan_delta']:.3e} surf={o['surface_leak']:.3e}")

    def cap(fac, side):
        mat, bm = base_materials()
        name = f"element_{side}"
        mat[name] = (eps * fac, 0.002, 0.0)
        for b in (c1 if side == "c1" else c2):
            bm[b] = name
        return mat, bm

    def loss(td, side):
        mat, bm = base_materials()
        name = f"element_{side}"
        mat[name] = (eps, td, 0.0)
        for b in (c1 if side == "c1" else c2):
            bm[b] = name
        return mat, bm

    def short(n):
        mat, bm = base_materials()
        if n > 0:
            mat["element_short"] = (1.0, 0.0, 1.0)
            for k in range(1, n + 1):
                bm[f"element_{k}"] = "element_short"
        return mat, bm

    print("internal severity sweeps ...")
    for fac in CAP_FACTORS:
        data["internal"]["C1_cap"].append({"s": fac, "x": xvec(solve(f"c1cap_{fac}", *cap(fac, "c1")))})
        data["internal"]["C2_cap"].append({"s": fac, "x": xvec(solve(f"c2cap_{fac}", *cap(fac, "c2")))})
    for td in LOSS_TAND:
        data["internal"]["C1_loss"].append({"s": td, "x": xvec(solve(f"c1loss_{td}", *loss(td, "c1")))})
        data["internal"]["C2_loss"].append({"s": td, "x": xvec(solve(f"c2loss_{td}", *loss(td, "c2")))})
    for n in SHORT_N:
        data["internal"]["disc_short"].append({"s": n, "x": xvec(solve(f"short_{n}", *short(n)))})

    (PROC / "trajectories.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nWrote {PROC / 'trajectories.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

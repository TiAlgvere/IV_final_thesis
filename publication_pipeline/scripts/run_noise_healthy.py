"""Task 020 (Part A) - healthy operating-point sweep under temperature and frequency.

No fault is introduced. We perturb the *healthy* representative DDB-123 by:

  1. Temperature T = -30..60 C : capacitor-dielectric permittivity drifts with
     alpha_eps [ppm/K] and the dielectric loss tangent drifts with a (documented,
     parameterised) temperature factor f_tand(T). Applied uniformly to the stack.
  2. Grid frequency f = 49.8..50.2 Hz : folded consistently into both the SIF
     `Frequency` and the loss `sigma_eff = omega*eps0*eps_r*tan_delta`.

Two physics facts (verified here, used by analyze_noise.py):
  * The capacitive divider ratio is first-order invariant to *uniform* temperature
    (C1 and C2 scale together) and to frequency (kappa factors globally). So Vtap,
    phase and tan-delta barely move with T/f; tan-delta/volume_leak move with the
    material loss (temperature), and the volume loss CURRENT scales ~linearly with f.
  * Therefore the healthy Vtap/phase envelope is dominated by the secondary-burden /
    measurement confounder (handled analytically in analyze_noise.py), NOT the field.

No solver / fault-physics change. Writes
results/processed/020_noise_detectability/healthy_field.json.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_cvt  # noqa: E402  (mutate FREQ/OMEGA for the frequency check)
from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from cvt_observables import compute_cvt_observables  # noqa: E402
from run_cvt import EPS0, U0, build_material_table, ensure_solver, material_of_body, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

PROC = ROOT / "results" / "processed" / "020_noise_detectability"
RAW = ROOT / "results" / "raw" / "020_noise_detectability"

# ---- documented temperature-drift assumptions (exposed; revise with project data) ----
T_REF = 20.0                 # C, reference operating temperature
TEMPS = [-30, -20, -10, 0, 20, 40, 60]
ALPHA_EPS = 200e-6           # 1/K, capacitor dielectric permittivity temp-coeff (200 ppm/K)
# loss tangent temperature factor: mild exponential (~doubles per 40 K rise).
# f_tand(T) = exp(K_TAND*(T - T_REF)); K_TAND = ln(2)/40.
K_TAND = math.log(2.0) / 40.0
FREQS = [49.8, 50.0, 50.2]   # Hz, for the invariance check (full grid handled analytically)


def f_tand(T: float) -> float:
    return math.exp(K_TAND * (T - T_REF))


def temp_material_table(eps_ref: float, T: float) -> dict[str, tuple[float, float, float]]:
    """Healthy material table at temperature T. Permittivity temp-coeff on the
    capacitor dielectric; loss-tangent temp factor on all lossy dielectrics."""
    base = build_material_table(eps_ref)
    ft = f_tand(T)
    eps_T = eps_ref * (1.0 + ALPHA_EPS * (T - T_REF))
    base["element"] = (eps_T, 0.002 * ft, 0.0)
    base["oil"] = (2.2, 0.001 * ft, 0.0)
    base["porcelain"] = (6.0, 0.005 * ft, 0.0)
    return base


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    p = CVTParams()
    eps = p.element_epsr_eff()

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

    def solve(tag, mat, omega):
        bm = {n: material_of_body(n) for n in ids["bodies"]}
        pdir = RAW / tag
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(elmer_mesh, pdir / "mesh", dirs_exist_ok=True)
        shutil.copy2(dll, pdir / dll.name)
        mat_index = {m: i + 1 for i, m in enumerate(mat)}
        sif = render_sif(mat, mat_index, bm, bodies, hv, gnd, ff, ins_id=ins)
        (pdir / "case.sif").write_text(sif, encoding="utf-8")
        d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                           capture_output=True, text=True, env=env)
        if d.returncode != 0:
            print(d.stdout[-1200:]); raise RuntimeError(f"solve failed {tag}")
        vtu = sorted(pdir.glob("**/case*.vtu"))[-1]
        be = {bid: mat[bm[n]][0] for n, bid in ids["bodies"].items()}
        bs = {bid: sigma_eff(*mat[bm[n]]) for n, bid in ids["bodies"].items()}
        return compute_cvt_observables(vtu, u0=U0, omega=omega, body_epsr=be, body_sigma=bs,
                                       tap_body_id=tap_id)

    # ---- baseline (T_ref, 50 Hz) ----
    print("baseline (20 C, 50 Hz) ...")
    o0 = solve("baseline", temp_material_table(eps, T_REF), 2.0 * math.pi * 50.0)
    v0, ph0 = o0["vtap_abs"], o0["vtap_phase_mdeg"]
    print(f"   |Vtap|={v0:.3f} V  phase={ph0:+.3f} mdeg  tand={o0['tan_delta']:.4e}  "
          f"vol_leak={o0['volume_leak']:.4e}")

    def rec(o):
        return {"vtap_abs": o["vtap_abs"],
                "dvtap_pct": (o["vtap_abs"] / v0 - 1.0) * 100.0,
                "dtheta_mdeg": o["vtap_phase_mdeg"] - ph0,
                "tan_delta": o["tan_delta"], "surface_leak": o["surface_leak"],
                "volume_leak": o["volume_leak"]}

    # ---- temperature sweep at 50 Hz ----
    print("temperature sweep (50 Hz) ...")
    temp_rows = []
    for T in TEMPS:
        o = solve(f"T_{T}", temp_material_table(eps, T), 2.0 * math.pi * 50.0)
        r = {"T": T, **rec(o)}
        temp_rows.append(r)
        print(f"   T={T:+3d} C: dVtap={r['dvtap_pct']:+.4f}% dtheta={r['dtheta_mdeg']:+.3f}mdeg "
              f"tand={r['tan_delta']:.4e} vol={r['volume_leak']:.4e}")

    # ---- frequency invariance check (consistent omega folding) ----
    print("frequency check (20 C) ...")
    freq_rows = []
    saved = (run_cvt.FREQ, run_cvt.OMEGA)
    try:
        for f in FREQS:
            run_cvt.FREQ = f
            run_cvt.OMEGA = 2.0 * math.pi * f
            o = solve(f"f_{f}", temp_material_table(eps, T_REF), run_cvt.OMEGA)
            r = {"f": f, **rec(o)}
            freq_rows.append(r)
            print(f"   f={f} Hz: dVtap={r['dvtap_pct']:+.5f}% dtheta={r['dtheta_mdeg']:+.4f}mdeg "
                  f"tand={r['tan_delta']:.5e} vol={r['volume_leak']:.5e}")
    finally:
        run_cvt.FREQ, run_cvt.OMEGA = saved

    out = {
        "config": {"T_ref_C": T_REF, "alpha_eps_per_K": ALPHA_EPS, "k_tand_per_K": K_TAND,
                   "f_tand_model": "exp(k_tand*(T-T_ref)); ~doubles per 40 K",
                   "freq_ref_Hz": 50.0, "temps_C": TEMPS, "freqs_Hz": FREQS,
                   "note": "uniform temperature -> divider ratio invariant; volume_leak ~ linear in f"},
        "baseline": {"T": T_REF, "f": 50.0, "vtap_abs": v0, "phase_mdeg": ph0,
                     "tan_delta": o0["tan_delta"], "surface_leak": o0["surface_leak"],
                     "volume_leak": o0["volume_leak"], "U0": U0},
        "temperature": temp_rows,
        "frequency_check": freq_rows,
    }
    (PROC / "healthy_field.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {PROC / 'healthy_field.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

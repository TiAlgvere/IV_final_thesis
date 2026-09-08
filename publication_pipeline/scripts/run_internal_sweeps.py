"""Task 019 - internal fault-family severity sweeps (trajectory refinement).

Refines the Task 016/018 internal manifolds into smooth severity sweeps on the
representative rounded DDB-123 baseline. Same observable state vector as Task 018:

    x = [ d|Vtap| %, d-theta mdeg, tan-delta, surface_leak A, volume_leak A ]

reported relative to the healthy rounded DDB-123 baseline. Families:

  1. C1 capacitance increase : +0.1, 0.3, 1, 3, 10, 20 %   (eps_r of C1 elements)
  2. C2 capacitance increase : +0.1, 0.3, 1, 3, 10, 20 %
  3. C1 dielectric loss      : tan-delta 0.002, 0.005, 0.01, 0.02, 0.05, 0.10
  4. C2 dielectric loss      : tan-delta 0.002, 0.005, 0.01, 0.02, 0.05, 0.10
  5. disc short              : (a) full short at 5 locations (C1 HV/mid/tap, C2 tap/gnd)
                               (b) partial-short severity sigma = 1e-4..1 S/m on a C1 element

No solver / physics change - reuses the verified axisymmetric ComplexEQS solver and
the per-body material mechanism from run_cvt / trajectories_run.

Writes results/processed/019_internal_severity_sweeps/internal_fault_sweeps.csv
(+ baseline.json) for analyze_internal_sweeps.py.
"""

from __future__ import annotations

import csv
import json
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

from publication_pipeline._elmer_tools import resolve_elmer_grid, resolve_elmer_home, resolve_elmer_solver  # noqa: E402
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402
from cvt_observables import compute_cvt_observables  # noqa: E402
from run_cvt import OMEGA, U0, build_material_table, ensure_solver, material_of_body, render_sif, sigma_eff  # noqa: E402
from run_elmer_case import _parse_mesh_names  # noqa: E402

PROC = ROOT / "results" / "processed" / "019_internal_severity_sweeps"
RAW = ROOT / "results" / "raw" / "019_internal_severity_sweeps"

CAP_PCT = [0.1, 0.3, 1.0, 3.0, 10.0, 20.0]          # capacitance increase, %
LOSS_TAND = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10]  # element tan-delta (0.002 = baseline)
SHORT_LOCATIONS = [                                  # (element index, case label)
    (1, "C1_HV"), (5, "C1_mid"), (10, "C1_tap"), (11, "C2_tap"), (12, "C2_gnd")]
SHORT_SIGMA = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]          # partial-short conductance, S/m (1 = full)
SHORT_SIGMA_ELEM = 5                                 # C1 mid element used for the severity sweep

CSV_COLS = ["family", "severity_parameter", "severity_value", "case_name",
            "dvtap_pct", "dtheta_mdeg", "tan_delta", "surface_leak", "volume_leak",
            "vtap_abs", "notes"]


def _short_analytic_dvtap(elem_index: int, p: CVTParams) -> float:
    """Ideal series-divider |Vtap| shift for a full short of one stack element.

    Healthy ratio = n_c2 / (n_c1 + n_c2). Shorting one C1 element removes one C1
    series unit (n_c1 -> n_c1-1); shorting one C2 element removes one C2 unit.
    """
    n_c1 = p.tap_disc_index
    n_c2 = p.n_elements - p.tap_disc_index
    base = n_c2 / (n_c1 + n_c2)
    if elem_index <= p.tap_disc_index:      # C1 short
        n1, n2 = n_c1 - 1, n_c2
    else:                                   # C2 short
        n1, n2 = n_c1, n_c2 - 1
    new = n2 / (n1 + n2)
    return (new / base - 1.0) * 100.0


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

    def solve(tag, mat, bm):
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
        return compute_cvt_observables(vtu, u0=U0, omega=OMEGA, body_epsr=be, body_sigma=bs,
                                       tap_body_id=tap_id)

    def base_materials():
        return build_material_table(eps), {n: material_of_body(n) for n in ids["bodies"]}

    # ---- baseline ----
    print("baseline ...")
    mh, bh = base_materials()
    o0 = solve("baseline", mh, bh)
    v0, ph0 = o0["vtap_abs"], o0["vtap_phase_mdeg"]
    print(f"   baseline |Vtap|={v0:.4f} V  phase={ph0:+.2f} mdeg  tand={o0['tan_delta']:.3e}")

    rows: list[dict] = []

    def record(family, sparam, sval, case_name, o, notes=""):
        rows.append({
            "family": family, "severity_parameter": sparam, "severity_value": sval,
            "case_name": case_name,
            "dvtap_pct": (o["vtap_abs"] / v0 - 1.0) * 100.0,
            "dtheta_mdeg": o["vtap_phase_mdeg"] - ph0,
            "tan_delta": o["tan_delta"], "surface_leak": o["surface_leak"],
            "volume_leak": o["volume_leak"], "vtap_abs": o["vtap_abs"], "notes": notes})

    # baseline row (severity 0 / reference)
    record("baseline", "none", 0.0, "healthy", o0, "healthy rounded DDB-123 reference")

    # ---- helpers for fault material tables ----
    def cap(pct, side):
        mat, bm = base_materials()
        name = f"element_{side}"
        mat[name] = (eps * (1.0 + pct / 100.0), 0.002, 0.0)
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

    def short(elem_index, sigma):
        """Degrading element: keeps eps_r_eff, develops leakage conductance sigma.
        sigma=1 S/m -> equipotential (full short); smaller -> partial leaky path."""
        mat, bm = base_materials()
        mat["element_short"] = (eps, 0.0, float(sigma))
        bm[f"element_{elem_index}"] = "element_short"
        return mat, bm

    # ---- 1+2. capacitance sweeps ----
    print("C1/C2 capacitance sweeps ...")
    for pct in CAP_PCT:
        record("C1_cap", "cap_increase_pct", pct, f"C1_cap_{pct:g}pct",
               solve(f"c1cap_{pct:g}", *cap(pct, "c1")))
        record("C2_cap", "cap_increase_pct", pct, f"C2_cap_{pct:g}pct",
               solve(f"c2cap_{pct:g}", *cap(pct, "c2")))

    # ---- 3+4. dielectric-loss sweeps ----
    print("C1/C2 dielectric-loss sweeps ...")
    for td in LOSS_TAND:
        record("C1_loss", "tan_delta", td, f"C1_loss_{td:g}",
               solve(f"c1loss_{td:g}", *loss(td, "c1")))
        record("C2_loss", "tan_delta", td, f"C2_loss_{td:g}",
               solve(f"c2loss_{td:g}", *loss(td, "c2")))

    # ---- 5a. disc-short location (full short) ----
    print("disc-short location sweep (full short) ...")
    for idx, label in SHORT_LOCATIONS:
        an = _short_analytic_dvtap(idx, p)
        record("disc_short", "short_location", idx, label,
               solve(f"short_loc_{idx}", *short(idx, 1.0)),
               notes=f"full short; analytic d|Vtap|={an:+.2f}%")

    # ---- 5b. disc-short severity (partial short on one C1 element) ----
    print("disc-short severity sweep (partial short) ...")
    for sg in SHORT_SIGMA:
        record("disc_short", "short_sigma", sg, f"short_sigma_{sg:g}",
               solve(f"short_sig_{sg:g}", *short(SHORT_SIGMA_ELEM, sg)),
               notes=f"C1 element {SHORT_SIGMA_ELEM}; sigma={sg:g} S/m")

    # ---- write CSV + baseline.json ----
    with (PROC / "internal_fault_sweeps.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_COLS})

    baseline = {"vtap_abs": v0, "phase_mdeg": ph0, "tan_delta": o0["tan_delta"],
                "surface_leak": o0["surface_leak"], "volume_leak": o0["volume_leak"],
                "n_c1": p.tap_disc_index, "n_c2": p.n_elements - p.tap_disc_index,
                "U0": U0}
    (PROC / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    print(f"\nWrote {PROC / 'internal_fault_sweeps.csv'}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

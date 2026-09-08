"""Task 016 - capacitor-stack (internal) fault signatures.

Builds the INTERNAL fault family to contrast with the external pollution family,
so the two can later form a diagnostic matrix. Faults are volume material changes
on specific capacitor-element bodies of the rounded DDB-123 (no surface term):

  C1_cap   : C1 (HV section, elements 1..tap) permittivity +20%   -> C1 up  -> Vtap up
  C2_cap   : C2 (LV section, elements tap+1..N) permittivity +20% -> C2 up  -> Vtap down
  C1_loss  : C1 element loss tan-delta 0.002 -> 0.05
  C2_loss  : C2 element loss tan-delta 0.002 -> 0.05
  disc_short : one C1 element shorted (sigma = 1 S/m, equipotential) -> divider ratio jump

Per case: |Vtap|, dVtap%, tap phase, tan-delta, leakage, state-space (log tand, d-theta).
A state-space plot overlays the pollution family (Task 015) to preview the separation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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

PROC = ROOT / "results" / "processed" / "016_stack_faults"
RAW = ROOT / "results" / "raw" / "016_stack_faults"
CAP_FACTOR = 1.20      # +20% permittivity for capacitance faults
LOSS_TAND = 0.05       # raised element loss tangent
SHORT_ELEMENT = 5      # which C1 element to short (1-based)


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    dll = ensure_solver()
    p = CVTParams()
    eps = p.element_epsr_eff()
    c1 = [f"element_{k}" for k in range(1, p.tap_disc_index + 1)]          # 1..10 (HV)
    c2 = [f"element_{k}" for k in range(p.tap_disc_index + 1, p.n_elements + 1)]  # 11..12 (LV)

    msh = RAW / "cvt.msh"
    build_cvt_ddb123(msh, p, rounded=True)
    ep = RAW / "elmer"; ep.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(resolve_elmer_grid()), "14", "2", str(msh), "-out", "mesh"],
                   cwd=str(ep), capture_output=True, text=True, check=True)
    elmer_mesh = ep / "mesh"
    ids = _parse_mesh_names(elmer_mesh / "mesh.names")
    bodies = [(bid, n) for n, bid in ids["bodies"].items()]
    tap_id = ids["bodies"][f"foil_{p.tap_disc_index}"]
    hv, gnd, ff = ids["boundaries"]["hv_electrode"], ids["boundaries"]["ground_electrode"], ids["boundaries"].get("farfield")

    env = os.environ.copy()
    home = resolve_elmer_home()
    if home is not None:
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    def fault_spec(fault):
        """Return (mat_table, body_material) for a fault (per-body element overrides)."""
        mat = build_material_table(eps)
        bm = {n: material_of_body(n) for n in ids["bodies"]}
        if fault == "C1_cap":
            mat["element_c1"] = (eps * CAP_FACTOR, 0.002, 0.0)
            for b in c1: bm[b] = "element_c1"
        elif fault == "C2_cap":
            mat["element_c2"] = (eps * CAP_FACTOR, 0.002, 0.0)
            for b in c2: bm[b] = "element_c2"
        elif fault == "C1_loss":
            mat["element_c1"] = (eps, LOSS_TAND, 0.0)
            for b in c1: bm[b] = "element_c1"
        elif fault == "C2_loss":
            mat["element_c2"] = (eps, LOSS_TAND, 0.0)
            for b in c2: bm[b] = "element_c2"
        elif fault == "disc_short":
            mat["element_short"] = (1.0, 0.0, 1.0)  # shorted dielectric -> metal-like
            bm[f"element_{SHORT_ELEMENT}"] = "element_short"
        return mat, bm

    def solve(tag, mat, bm):
        pdir = RAW / tag
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(elmer_mesh, pdir / "mesh", dirs_exist_ok=True)
        shutil.copy2(dll, pdir / dll.name)
        mat_index = {m: i + 1 for i, m in enumerate(mat)}
        sif = render_sif(mat, mat_index, bm, bodies, hv, gnd, ff)
        (pdir / "case.sif").write_text(sif, encoding="utf-8")
        d = subprocess.run([str(resolve_elmer_solver()), "case.sif"], cwd=str(pdir),
                           capture_output=True, text=True, env=env)
        (pdir / "elmersolver.stdout.log").write_text(d.stdout, encoding="utf-8")
        if d.returncode != 0:
            print(d.stdout[-1500:]); raise RuntimeError(f"solve failed {tag}")
        vtu = sorted(pdir.glob("**/case*.vtu"))[-1]
        body_epsr = {bid: mat[bm[n]][0] for n, bid in ids["bodies"].items()}
        body_sigma = {bid: sigma_eff(*mat[bm[n]]) for n, bid in ids["bodies"].items()}
        return compute_cvt_observables(vtu, u0=U0, omega=OMEGA, body_epsr=body_epsr,
                                       body_sigma=body_sigma, tap_body_id=tap_id)

    print("Solving healthy ...")
    mh, bh = build_material_table(eps), {n: material_of_body(n) for n in ids["bodies"]}
    o0 = solve("healthy", mh, bh)
    v0, ph0 = o0["vtap_abs"], o0["vtap_phase_mdeg"]

    faults = ["C1_cap", "C2_cap", "C1_loss", "C2_loss", "disc_short"]
    rows = [{"case": "healthy", "vtap_abs": v0, "dvtap_pct": 0.0, "dtheta_mdeg": 0.0,
             "tan_delta": o0["tan_delta"], "i_leak": o0["i_leak"],
             "surface_leak": o0["surface_leak"], "volume_leak": o0["volume_leak"]}]
    for f in faults:
        print(f"Solving {f} ...")
        mat, bm = fault_spec(f)
        o = solve(f, mat, bm)
        rows.append({"case": f, "vtap_abs": o["vtap_abs"],
                     "dvtap_pct": (o["vtap_abs"] / v0 - 1) * 100.0,
                     "dtheta_mdeg": o["vtap_phase_mdeg"] - ph0,
                     "tan_delta": o["tan_delta"], "i_leak": o["i_leak"],
                     "surface_leak": o["surface_leak"], "volume_leak": o["volume_leak"]})
        print(f"   |Vtap|={o['vtap_abs']:.1f} ({rows[-1]['dvtap_pct']:+.2f}%) "
              f"dtheta={rows[-1]['dtheta_mdeg']:.2f}mdeg tand={o['tan_delta']:.4e} Ileak={o['i_leak']:.3e}")

    # CSV
    cols = ["case", "vtap_abs", "dvtap_pct", "dtheta_mdeg", "tan_delta", "i_leak"]
    with (PROC / "stack_faults.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if not isinstance(r[c], str) else r[c] for c in cols) + "\n")

    # ---- state-space: internal faults vs pollution family ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    # pollution family (Task 015 manifold + cases) if available
    poll = ROOT / "results" / "processed" / "015_localized_pollution" / "summary.json"
    if poll.is_file():
        ps = json.loads(poll.read_text(encoding="utf-8"))
        ref = ps.get("ref", [])
        if ref:
            rt = np.array([r[1] for r in ref]); rd = np.array([r[2] for r in ref])
            ax[0].plot(np.log10(rt), rd, "-", color="#d8b4fe", lw=2.0, zorder=1,
                       label="pollution family (creepage sigma_s)")
        for pc in ps.get("cases", []):
            ax[0].scatter(np.log10(pc["tan_delta"]), pc["dtheta_mdeg"], marker="^",
                          s=70, color="#a855f7", edgecolors="k", zorder=3)
            ax[1].scatter(pc["dvtap_pct"] if "dvtap_pct" in pc else
                          (pc["vtap_abs"] / 11829.5 - 1) * 100.0, np.log10(pc["tan_delta"]),
                          marker="^", s=70, color="#a855f7", edgecolors="k", zorder=3,
                          label="pollution" if pc["case"] == "full" else None)
    colors = {"healthy": "#111827", "C1_cap": "#2563eb", "C2_cap": "#16a34a",
              "C1_loss": "#f59e0b", "C2_loss": "#dc2626", "disc_short": "#0891b2"}
    for r in rows:
        ax[0].scatter(np.log10(r["tan_delta"]), r["dtheta_mdeg"], s=140, color=colors[r["case"]],
                      edgecolors="k", zorder=5, label=r["case"])
        ax[0].annotate(r["case"], (np.log10(r["tan_delta"]), r["dtheta_mdeg"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=7.5)
        ax[1].scatter(r["dvtap_pct"], np.log10(r["tan_delta"]), s=140, color=colors[r["case"]],
                      edgecolors="k", zorder=5)
        ax[1].annotate(r["case"], (r["dvtap_pct"], np.log10(r["tan_delta"])),
                       textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax[0].set_xlabel(r"terminal loss $\log_{10}\tan\delta$"); ax[0].set_ylabel(r"$\Delta\theta$ [mdeg]")
    ax[0].set_title("(loss, phase) state space"); ax[0].grid(True, alpha=0.3); ax[0].legend(fontsize=7, loc="best")
    ax[1].set_xlabel(r"$\Delta$|Vtap| [%]"); ax[1].set_ylabel(r"$\log_{10}\tan\delta$")
    ax[1].set_title("(|Vtap| shift, loss) - separates cap/short from loss/pollution")
    ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=7, loc="best")
    fig.suptitle("Task 016 - internal capacitor-stack faults vs external pollution family", fontsize=13)
    fig.savefig(PROC / "stack_faults_state_space.png", dpi=150)
    plt.close(fig)

    # ---- report ----
    lines = [
        "# Task 016 - Capacitor-Stack (Internal) Fault Signatures",
        "",
        f"Rounded DDB-123, healthy |Vtap| = {v0:.1f} V, tap phase {ph0:.2f} mdeg. C1 = elements",
        f"1..{p.tap_disc_index} (HV section), C2 = elements {p.tap_disc_index+1}..{p.n_elements} (LV).",
        f"Cap faults +{(CAP_FACTOR-1)*100:.0f}% eps_r; loss faults tan-delta -> {LOSS_TAND};",
        f"disc short = element {SHORT_ELEMENT} (C1) shorted (sigma=1 S/m). d-theta vs healthy.",
        "",
        "| case | |Vtap| [V] | dVtap [%] | d-theta [mdeg] | tan(delta) | leakage [A] |",
        "|------|-----------|-----------|----------------|------------|-------------|",
    ]
    for r in rows:
        lines.append(f"| {r['case']} | {r['vtap_abs']:.1f} | {r['dvtap_pct']:+.2f} | {r['dtheta_mdeg']:+.2f} "
                     f"| {r['tan_delta']:.3e} | {r['i_leak']:.3e} |")
    lines += [
        "",
        "## Fault separation (internal family vs pollution family)",
        "- **Capacitance faults (C1/C2 cap, disc short)** move |Vtap| (the divider ratio) with",
        "  little tan-delta or leakage: C1-cap / C1-short raise Vtap, C2-cap lowers it.",
        "- **Loss faults (C1/C2 loss)** raise tan-delta and shift phase with ~no Vtap change;",
        "  C1-loss vs C2-loss differ in phase (loss above vs below the tap).",
        "- **External pollution** (Task 015) raises tan-delta + leakage with little Vtap change;",
        "  it lives on its own creepage manifold in (loss, phase).",
        "- Key separators: **|Vtap| shift -> capacitive/ratio faults**; **leakage with tan-delta",
        "  -> external pollution**; **tan-delta with ~zero leakage and ~zero Vtap -> internal loss**.",
        "  See `stack_faults_state_space.png` (left: loss-phase; right: Vtap-loss).",
        "",
        "## Artifacts",
        "- `stack_faults.csv`, `stack_faults_state_space.png`",
        "",
    ]
    (PROC / "stack_faults_report.md").write_text("\n".join(lines), encoding="utf-8")
    (PROC / "summary.json").write_text(json.dumps({"healthy_vtap": v0, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nDone -> {PROC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

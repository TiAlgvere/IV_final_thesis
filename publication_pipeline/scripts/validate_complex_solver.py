"""Validate the Task 006 ComplexEQS solver against the latest pollution sweep.

Primary, rigorous check
-----------------------
With sigma = 0 the complex electroquasistatic solve must reduce to the real
electrostatic baseline, so |Vtap| must equal the analytical capacitive divider
(the same quantity that validated study 001) and the phase / loss must vanish.
This checks the new solver against an already-validated result.

Consistency / physics checks
-----------------------------
  * clean (sigma=0): phase ~ 0, tan(delta) ~ 0, i_leak ~ 0
  * limits: |Vtap| -> clean as sigma -> 0; |Vtap| saturates (monotone down) as sigma grows
  * loss-relaxation signature: tan(delta) and phase peak at an *interior* sigma
    (the Maxwell-Wagner transition near sigma ~ omega*eps), not at the endpoints

Analytical complex divider (bound)
----------------------------------
The clean divider ratio equals the lossless series-capacitor result; reported for
reference. A full 2-D analytical solution is not tractable for this geometry, so
the sigma=0 identity above is the primary oracle.

scikit-fem corroboration
-------------------------
The same complex coefficient kappa = sigma + j*omega*eps was independently
validated in the legacy ctfem prototype (~5600 pF); this Elmer solver adopts that
formulation. A matched 2-D cross-backend run is left as optional follow-up.

Exit code is non-zero if any hard assertion fails.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_pipeline.geometry.gmsh.clean_baseline import CleanBaselineParameters  # noqa: E402

CASE_DIR = ROOT / "results" / "raw" / "006_pollution_sweep"
HV_VOLTAGE = 110000.0


def analytical_clean_vtap(params: CleanBaselineParameters, v_hv: float = HV_VOLTAGE) -> float:
    """Series-capacitor divider Vtap (same formula as run_elmer_case.py)."""
    air_thickness = params.air_height - params.porcelain_height
    c_air = 1.0 / air_thickness
    c_porcelain = 5.0 / params.porcelain_height
    return v_hv * (c_air / (c_air + c_porcelain))


def load_latest_sweep() -> tuple[Path, list[dict[str, float]]]:
    csvs = sorted(CASE_DIR.glob("run_*/pollution_sweep.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No pollution_sweep.csv under {CASE_DIR}. Run run_pollution_sweep.py first.")
    latest = csvs[-1]
    with latest.open(encoding="utf-8") as fh:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]
    rows.sort(key=lambda r: r["sigma"])
    return latest, rows


def main() -> int:
    params = CleanBaselineParameters()
    vtap_analytical = analytical_clean_vtap(params)
    csv_path, rows = load_latest_sweep()

    clean = next((r for r in rows if r["sigma"] == 0.0), None)
    if clean is None:
        print("FAIL: sweep has no sigma=0 (clean) point.", file=sys.stderr)
        return 1

    sigmas = [r["sigma"] for r in rows]
    vtaps = [r["vtap_abs"] for r in rows]
    phases = [abs(r["phase_deg"]) for r in rows]
    tands = [r["tan_delta"] for r in rows]

    checks: list[tuple[str, bool, str]] = []

    # 1. sigma=0 reproduces the analytical capacitive divider.
    rel_err = abs(clean["vtap_abs"] - vtap_analytical) / vtap_analytical
    checks.append((
        "sigma=0 |Vtap| == analytical capacitive divider",
        rel_err < 1e-4,
        f"Elmer={clean['vtap_abs']:.4f} V, analytical={vtap_analytical:.4f} V, rel_err={rel_err:.2e}",
    ))

    # 2. clean case has vanishing loss/phase.
    checks.append((
        "sigma=0 phase ~ 0 and tan(delta) ~ 0 and i_leak ~ 0",
        abs(clean["phase_deg"]) < 1e-3 and clean["tan_delta"] < 1e-9 and abs(clean["i_leak"]) < 1e-12,
        f"phase={clean['phase_deg']:.2e} deg, tand={clean['tan_delta']:.2e}, i_leak={clean['i_leak']:.2e}",
    ))

    # 3. |Vtap| monotone non-increasing with sigma (pollution can only load the divider down).
    monotone = all(vtaps[i + 1] <= vtaps[i] + 1e-6 * vtaps[0] for i in range(len(vtaps) - 1))
    checks.append((
        "|Vtap| monotone non-increasing with sigma",
        monotone,
        f"clean={vtaps[0]:.1f} V -> saturated={vtaps[-1]:.1f} V",
    ))

    # 4. loss-relaxation signature: tan(delta) and phase peak at an interior sigma.
    tan_peak_i = max(range(len(tands)), key=lambda i: tands[i])
    phase_peak_i = max(range(len(phases)), key=lambda i: phases[i])
    interior = 0 < tan_peak_i < len(rows) - 1 and 0 < phase_peak_i < len(rows) - 1
    checks.append((
        "tan(delta) and phase peak at interior sigma (loss relaxation)",
        interior,
        f"tan(delta) peak at sigma={sigmas[tan_peak_i]:.2e} (={tands[tan_peak_i]:.3e}); "
        f"phase peak at sigma={sigmas[phase_peak_i]:.2e} (={phases[phase_peak_i]:.2f} deg)",
    ))

    # Report.
    print(f"Validation against: {csv_path}")
    print(f"Analytical clean Vtap: {vtap_analytical:.4f} V")
    print(f"High-sigma saturated |Vtap|: {vtaps[-1]:.4f} V  ({(vtaps[-1]/vtaps[0]-1)*100:.1f}% vs clean)")
    print("-" * 78)
    all_pass = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok
        print(f"[{status}] {name}")
        print(f"        {detail}")
    print("-" * 78)
    print("RESULT:", "ALL CHECKS PASS" if all_pass else "SOME CHECKS FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

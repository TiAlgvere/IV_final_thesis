"""Plot Task 006 pollution-sweep observables vs pollution conductivity.

Reads a ``pollution_sweep.csv`` produced by run_pollution_sweep.py and writes a
multi-panel figure of the diagnostic observables against sigma (log axis). The
clean state (sigma = 0) cannot sit on a log axis, so it is drawn as a dashed
reference line in each panel and the sweep points are the polluted cases.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(csv_path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.asarray(v) for k, v in cols.items()}


def plot_sweep(csv_path: str | Path, out_png: str | Path | None = None) -> Path:
    csv_path = Path(csv_path)
    data = _load(csv_path)
    sigma = data["sigma"]
    clean = sigma == 0.0
    pol = ~clean
    s = sigma[pol]

    panels = [
        ("vtap_abs", "|Vtap|  [V]", False),
        ("phase_deg", "phase displacement d-theta  [deg]", False),
        ("tan_delta", "loss tangent  tan(delta)", True),
        ("emax", "peak field  Emax  [V/m]", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (key, label, logy) in zip(axes, panels):
        y = data[key]
        ax.plot(s, y[pol], "o-", color="#2563eb", label="polluted")
        if clean.any():
            cval = float(y[clean][0])
            ax.axhline(cval, ls="--", color="#9ca3af", label=f"clean (sigma=0): {cval:.3g}")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("pollution conductivity sigma  [S/m]")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Leakage-current panel: resistive (in-phase, peaks at transition) vs total terminal.
    ax = axes[4]
    ax.plot(s, data["i_leak"][pol], "o-", color="#16a34a", label="resistive (in-phase)")
    if "i_terminal" in data:
        ax.plot(s, data["i_terminal"][pol], "s--", color="#7c3aed", label="total terminal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pollution conductivity sigma  [S/m]")
    ax.set_ylabel("terminal current  [A/m depth]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # 6th panel: relative |Vtap| shift vs the clean state (the headline diagnostic).
    ax = axes[5]
    if clean.any():
        v0 = float(data["vtap_abs"][clean][0])
        rel = (data["vtap_abs"][pol] - v0) / v0 * 100.0
        ax.plot(s, rel, "o-", color="#dc2626")
        ax.axhline(0.0, ls="--", color="#9ca3af")
        ax.set_xscale("log")
        ax.set_xlabel("pollution conductivity sigma  [S/m]")
        ax.set_ylabel("|Vtap| shift vs clean  [%]")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Task 006 - diagnostic observables vs conductive pollution (complex EQS)", fontsize=14)

    out_png = Path(out_png) if out_png else csv_path.with_name("pollution_sweep.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot pollution-sweep observables.")
    parser.add_argument("csv", type=Path, help="Path to pollution_sweep.csv")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path.")
    args = parser.parse_args()
    out = plot_sweep(args.csv, args.out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

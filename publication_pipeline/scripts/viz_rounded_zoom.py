"""Task 013 - before/after zoom showing the geometry rounding (sharp vs filleted).

Builds the clean CVT mesh twice (rounded=False and rounded=True) and plots a zoom
of a few weather sheds + the porcelain wall + a capacitor-disc edge, with mesh
edges, so the sharp-corner -> fillet change (and the curvature refinement) is
visible. No solve; geometry only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from publication_pipeline.geometry.gmsh.cvt_ddb123 import CVTParams, build_cvt_ddb123  # noqa: E402

PROC = ROOT / "results" / "processed" / "013_surface_conductance"
RAW = ROOT / "results" / "raw" / "013_surface_conductance"


def _tri(msh: Path):
    m = meshio.read(str(msh))
    pts = np.asarray(m.points)[:, :2]
    tris = np.vstack([np.asarray(b.data) for b in m.cells if b.type == "triangle"])
    return mtri.Triangulation(pts[:, 0], pts[:, 1], tris)


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    c = CVTParams()
    sharp = RAW / "geom_sharp.msh"
    rnd = RAW / "geom_rounded.msh"
    build_cvt_ddb123(sharp, c, rounded=False)
    build_cvt_ddb123(rnd, c, rounded=True)

    # zoom on a few mid-stack sheds + wall + the nearest disc edge
    centers = c.shed_centers()
    zc = centers[len(centers) // 2]
    pitch = (c.stack_z_hi - c.tank_height) / c.n_sheds
    z0, z1 = zc - 2.0 * pitch, zc + 2.0 * pitch
    r0, r1 = c.stack_radius - 0.01, c.porcelain_inner_radius + c.porcelain_thickness + c.shed_overhang + 0.005

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    for ax, msh, title in zip(axes, [sharp, rnd], ["sharp (Tasks 009-012)", "rounded (Task 013)"]):
        ax.triplot(_tri(msh), color="#334155", lw=0.3)
        ax.set_xlim(r0, r1)
        ax.set_ylim(z0, z1)
        ax.set_aspect("equal")
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(title)
    fig.suptitle("Task 013 - shed lips, wall & disc edges: sharp vs rounded (with curvature refinement)", fontsize=13)
    fig.savefig(PROC / "rounding_before_after.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {PROC / 'rounding_before_after.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

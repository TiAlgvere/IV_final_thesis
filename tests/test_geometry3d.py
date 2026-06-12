"""3-D geometry tests -- require gmsh (not dolfinx). Kept small/fast."""
import math
import os
import tempfile

import pytest

pytest.importorskip("gmsh")

from ctfem.config import GeometryParams
from ctfem.geometry3d import build_ct_3d
from ctfem.geometry import load_tag_map


def test_full_revolve_all_groups_nonempty():
    # very coarse + few foils so this stays well under the suite time budget
    g = GeometryParams(n_foils=3, mesh_refinement=0.18)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ct3d.msh")
        r = build_ct_3d(g, p, angle=2 * math.pi, verbose=False)
        # volume (region) groups
        for name in ["primary_conductor", "paper_insulation", "oil",
                     "porcelain", "air", "head_housing", "base_tank",
                     "foil_1", "foil_2", "foil_3"]:
            assert r.surface_groups.get(name, 0) > 0, f"{name} empty"
        # electrode/farfield SURFACE groups (the centroid-on-axis farfield bug
        # regression check)
        for name in ["hv_electrode", "ground_electrode", "farfield"]:
            assert r.curve_groups.get(name, 0) > 0, f"{name} empty"
        tags = load_tag_map(p)
        assert tags["paper_insulation"][0] == 3   # a volume group
        assert tags["hv_electrode"][0] == 2       # a surface group

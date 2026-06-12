"""Geometry tests -- require gmsh (not dolfinx). Fast (coax + tiny CT)."""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("gmsh")

from ctfem.config import GeometryParams
from ctfem.geometry import build_coax, build_ct, load_tag_map


def test_coax_groups_nonempty():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "coax.msh")
        r = build_coax(0.01, 0.05, 0.2, p, lc=0.005, verbose=False)
        assert r.surface_groups["dielectric"] > 0
        assert r.curve_groups["hv_electrode"] > 0
        assert r.curve_groups["ground_electrode"] > 0
        tags = load_tag_map(p)
        assert ("dielectric" in tags) and ("hv_electrode" in tags)


def test_ct_all_groups_nonempty():
    # coarse + few foils keeps this fast
    g = GeometryParams(n_foils=6, mesh_refinement=0.4)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ct.msh")
        r = build_ct(g, p, verbose=False)
        # every required surface region present and non-empty
        for name in ["primary_conductor", "paper_insulation", "oil",
                     "porcelain", "air", "head_housing", "base_tank"]:
            assert r.surface_groups.get(name, 0) > 0, f"{name} empty"
        for k in range(1, 7):
            assert r.surface_groups.get(f"foil_{k}", 0) > 0
        for name in ["hv_electrode", "ground_electrode", "farfield"]:
            assert r.curve_groups.get(name, 0) > 0, f"{name} empty"


def test_refinement_increases_element_count():
    g_coarse = GeometryParams(n_foils=4, mesh_refinement=0.3)
    g_fine = GeometryParams(n_foils=4, mesh_refinement=0.8)
    with tempfile.TemporaryDirectory() as d:
        c = build_ct(g_coarse, os.path.join(d, "c.msh"), verbose=False)
        f = build_ct(g_fine, os.path.join(d, "f.msh"), verbose=False)
        assert f.n_triangles > c.n_triangles

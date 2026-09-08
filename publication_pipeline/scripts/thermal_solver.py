"""Task 021 - self-contained axisymmetric steady-state heat-conduction solver.

One-way electrothermal coupling: the electrical loss density from the ComplexEQS
solve is the heat source for

    -div( k grad T ) = q'''        (volumetric, in the solid)
    -k dT/dn = h (T - T_amb)       (convection on the solid-air surface)
    + q'' surface source on the creepage (pollution) boundary

solved on the SOLID sub-domain (air is excluded and represented by the convective
boundary). Axisymmetric P1 triangles, integration weight 2*pi*r.

Self-contained (numpy/scipy) and reusing the EQS loss integrands directly, so the
heat source is energy-consistent with the diagnostic observables (volume_leak /
surface_leak). Verified against the analytic uniformly-heated cylinder.

API: solve_axisym_heat(...) -> nodal T (above T_amb if T_amb=0).
Run as a script to execute the verification benchmark.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def _tri_geom(points: np.ndarray, tris: np.ndarray):
    """Per-triangle gradient coefficients b,c, signed det (=2A), area, centroid r."""
    p = points[:, :2]
    x1, x2, x3 = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    b = np.column_stack([x2[:, 1] - x3[:, 1], x3[:, 1] - x1[:, 1], x1[:, 1] - x2[:, 1]])  # y_j-y_k
    c = np.column_stack([x3[:, 0] - x2[:, 0], x1[:, 0] - x3[:, 0], x2[:, 0] - x1[:, 0]])  # x_k-x_j
    det = (x2[:, 0] - x1[:, 0]) * (x3[:, 1] - x1[:, 1]) - (x3[:, 0] - x1[:, 0]) * (x2[:, 1] - x1[:, 1])
    area = 0.5 * np.abs(det)
    r_c = (x1[:, 0] + x2[:, 0] + x3[:, 0]) / 3.0
    return b, c, det, area, r_c


def boundary_edges(tris: np.ndarray) -> np.ndarray:
    """Edges that belong to exactly one triangle of `tris` (the sub-mesh boundary)."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    key = np.sort(e, axis=1)
    uniq, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return e[idx[cnt == 1]]  # keep original orientation of the single-owner edge


def solve_axisym_heat(points, tris, k_tri, q_tri, *, h, robin_edges,
                      surf_edges=None, surf_power_edge=None, T_amb=0.0):
    """Axisymmetric P1 steady heat conduction on the triangles `tris`.

    points (Np,2)=[r,z]; tris (M,3); k_tri (M,) W/mK; q_tri (M,) W/m3 volumetric.
    robin_edges (E,2): solid-air boundary node pairs (convection h, ambient T_amb).
    surf_edges (S,2)+surf_power_edge (S,): total W per edge deposited as a surface
    source (creepage pollution), split equally to the two edge nodes.
    Returns T (Np,) (= rise above T_amb when T_amb=0)."""
    used = np.unique(tris)
    g2l = -np.ones(points.shape[0], dtype=int); g2l[used] = np.arange(used.size)
    P = points[used]; tl = g2l[tris]
    n = used.size

    b, c, det, area, r_c = _tri_geom(P, tl)
    w = 2.0 * np.pi * r_c                      # axisymmetric weight at centroid
    # element stiffness K_e[i,j] = k * 2pi r_c (b_i b_j + c_i c_j) / (4A)
    coef = k_tri * w / (4.0 * area)            # (M,)
    rows = []; cols = []; vals = []
    for i in range(3):
        for j in range(3):
            rows.append(tl[:, i]); cols.append(tl[:, j])
            vals.append(coef * (b[:, i] * b[:, j] + c[:, i] * c[:, j]))

    f = np.zeros(n)
    # volumetric source (lumped): each node gets q*2pi r_c*A/3
    fe = q_tri * w * area / 3.0
    np.add.at(f, tl[:, 0], fe); np.add.at(f, tl[:, 1], fe); np.add.at(f, tl[:, 2], fe)

    # Robin convection on solid-air boundary edges (vectorised into COO)
    if robin_edges is not None and len(robin_edges):
        re = g2l[robin_edges]
        L = np.hypot(P[re[:, 1], 0] - P[re[:, 0], 0], P[re[:, 1], 1] - P[re[:, 0], 1])
        rm = 0.5 * (P[re[:, 0], 0] + P[re[:, 1], 0])
        wedge = 2.0 * np.pi * rm * L           # consistent edge mass M=[[1/3,1/6],[1/6,1/3]]*wedge
        for (a, bb, m) in ((0, 0, 1/3), (1, 1, 1/3), (0, 1, 1/6), (1, 0, 1/6)):
            rows.append(re[:, a]); cols.append(re[:, bb]); vals.append(h * wedge * m)
        if T_amb:
            np.add.at(f, re[:, 0], h * T_amb * wedge / 2.0)
            np.add.at(f, re[:, 1], h * T_amb * wedge / 2.0)

    K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()

    # creepage surface heat source (pollution), split per node
    if surf_edges is not None and len(surf_edges):
        se = g2l[surf_edges]
        np.add.at(f, se[:, 0], surf_power_edge / 2.0)
        np.add.at(f, se[:, 1], surf_power_edge / 2.0)

    T = spsolve(K, f)
    out = np.full(points.shape[0], np.nan)
    out[used] = T + (T_amb if not T_amb else 0.0)  # T_amb=0 -> rise; else already absolute
    return out


# --------------------------------------------------------------------------- verification
def _rect_mesh(R, H, nr, nz):
    r = np.linspace(0, R, nr); z = np.linspace(0, H, nz)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    pts = np.column_stack([rr.ravel(), zz.ravel()])
    idx = lambda i, j: i * nz + j
    tris = []
    for i in range(nr - 1):
        for j in range(nz - 1):
            tris.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            tris.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return pts, np.array(tris), r, z


def verify(tol=0.02) -> int:
    """Uniformly-heated solid cylinder, convective surface, insulated ends:
    analytic dT(r) = q(R^2 - r^2)/(4k) + qR/(2h); center dT = qR^2/(4k)+qR/(2h)."""
    R, H, k, h, q = 0.1, 0.4, 1.5, 10.0, 5.0e3
    pts, tris, r, z = _rect_mesh(R, H, 41, 21)
    k_tri = np.full(len(tris), k); q_tri = np.full(len(tris), q)
    # Robin only on r=R edges (outer); axis r=0 and z-ends insulated (natural)
    be = boundary_edges(tris)
    rmid = 0.5 * (pts[be[:, 0], 0] + pts[be[:, 1], 0])
    robin = be[rmid > R - 1e-9]
    T = solve_axisym_heat(pts, tris, k_tri, q_tri, h=h, robin_edges=robin, T_amb=0.0)

    def analytic(rr):
        return q * (R**2 - rr**2) / (4 * k) + q * R / (2 * h)

    # compare along the mid-height line
    jmid = len(z) // 2
    err = []
    for i in range(len(r)):
        node = i * len(z) + jmid
        err.append(abs(T[node] - analytic(r[i])))
    err = np.array(err); rel = err.max() / analytic(0.0)
    print(f"[verify] center dT num={T[0*len(z)+jmid]:.4f} K  analytic={analytic(0.0):.4f} K  "
          f"surface dT num={T[(len(r)-1)*len(z)+jmid]:.4f} analytic={analytic(R):.4f}")
    print(f"[verify] max |err|={err.max():.4e} K  relative={rel:.3%}  -> {'PASS' if rel < tol else 'FAIL'}")
    return 0 if rel < tol else 1


if __name__ == "__main__":
    raise SystemExit(verify())

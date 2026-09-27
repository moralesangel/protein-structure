"""Extra geometric node features for the binding-site graphs.

The base features describe a residue's own chemistry. A pocket, though, is a
property of the neighbourhood: a concave, partly buried patch. These add the
local geometry that mean-aggregated message passing cannot recover on its own.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

EXTRA_DIM = 6


def geometric(coords: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
    """Six features per residue, each scaled to roughly [-1, 1].

    0. contact density  - neighbours within the cutoff, a proxy for burial
    1. mid-shell density - neighbours in 8-12 A, distinguishing a groove from
       a flat surface
    2. concavity        - how far the residue sits from its neighbours' centre
       of mass, along the outward normal. Positive means it looks into a
       hollow, which is what a pocket is.
    3. radial position  - distance from the protein centre over the maximum,
       separating surface from core
    4. protrusion       - distance to the neighbourhood centroid
    5. planarity        - how flat the local patch is, from the spread of the
       neighbour cloud
    """
    n = len(coords)
    tree = cKDTree(coords)
    centre = coords.mean(axis=0)
    radial = np.linalg.norm(coords - centre, axis=1)
    max_radial = radial.max() or 1.0

    out = np.zeros((n, EXTRA_DIM), dtype=np.float32)

    near = tree.query_ball_point(coords, cutoff)
    mid = tree.query_ball_point(coords, 12.0)

    for i in range(n):
        nb = [j for j in near[i] if j != i]
        deg = len(nb)
        out[i, 0] = min(deg / 20.0, 1.5)
        out[i, 1] = min((len(mid[i]) - 1 - deg) / 30.0, 1.5)

        if deg >= 3:
            pts = coords[nb]
            com = pts.mean(axis=0)
            # Outward normal: away from the protein centre.
            outward = coords[i] - centre
            nrm = np.linalg.norm(outward)
            if nrm > 1e-6:
                outward = outward / nrm
                # Positive when the neighbours sit further out than the
                # residue itself, i.e. the residue lines a hollow.
                out[i, 2] = float(np.dot(com - coords[i], outward)) / cutoff

            d = coords[i] - com
            out[i, 4] = min(float(np.linalg.norm(d)) / cutoff, 1.5)

            # Planarity from the smallest eigenvalue share of the neighbour
            # cloud: a flat patch has one near-zero direction.
            cen = pts - com
            cov = cen.T @ cen / len(pts)
            ev = np.linalg.eigvalsh(cov)
            total = float(ev.sum())
            out[i, 5] = float(ev[0] / total) * 3.0 if total > 1e-6 else 0.0

        out[i, 3] = float(radial[i] / max_radial)

    return out

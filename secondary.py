"""Secondary structure assignment from Cα coordinates alone.

DSSP needs backbone N/O atoms to find hydrogen bonds; the demo only ships Cα
positions, so this uses the geometric criteria that P-SEA is built on: helices
and strands have characteristic Cα(i)-Cα(i+k) distances and Cα dihedral angles.
It agrees with DSSP on most residues, which is all the ribbon renderer needs.
"""

from __future__ import annotations

import numpy as np


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-9)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return np.degrees(np.arctan2(y, x))


def assign(coords: np.ndarray, chain_breaks: set[int] | None = None) -> str:
    """Return one character per residue: H (helix), E (strand), C (coil)."""
    n = len(coords)
    ss = ["C"] * n
    breaks = chain_breaks or set()

    def contiguous(i, k):
        """True when i..i+k stay inside one chain and are physically connected."""
        if i < 0 or i + k >= n:
            return False
        for j in range(i, i + k):
            if j + 1 in breaks:
                return False
            if np.linalg.norm(coords[j + 1] - coords[j]) > 4.5:
                return False
        return True

    # Alpha helix: Ca(i)-Ca(i+3) about 5.0-6.4 A, Ca(i)-Ca(i+4) about 5.8-7.2 A,
    # and the Ca dihedral near +50 degrees.
    for i in range(n):
        if not contiguous(i, 4):
            continue
        d3 = np.linalg.norm(coords[i + 3] - coords[i])
        d4 = np.linalg.norm(coords[i + 4] - coords[i])
        tau = _dihedral(coords[i], coords[i + 1], coords[i + 2], coords[i + 3])
        if 4.8 <= d3 <= 6.5 and 5.4 <= d4 <= 7.4 and 30 <= tau <= 80:
            for j in range(i, min(i + 4, n)):
                ss[j] = "H"

    # Beta strand: extended, so Ca(i)-Ca(i+2) about 6.4-7.2 A and the dihedral
    # is near 180 (or -180). Only assign where nothing was called a helix.
    for i in range(n):
        if not contiguous(i, 3):
            continue
        d2 = np.linalg.norm(coords[i + 2] - coords[i])
        d3 = np.linalg.norm(coords[i + 3] - coords[i])
        tau = abs(_dihedral(coords[i], coords[i + 1], coords[i + 2], coords[i + 3]))
        if 6.0 <= d2 <= 7.4 and 9.0 <= d3 <= 11.2 and tau >= 120:
            for j in range(i, min(i + 3, n)):
                if ss[j] == "C":
                    ss[j] = "E"

    # Drop runs too short to be real, which mostly removes isolated noise.
    out = list(ss)
    start = 0
    for i in range(1, n + 1):
        if i == n or out[i] != out[start]:
            run = i - start
            if out[start] == "H" and run < 4:
                for j in range(start, i):
                    out[j] = "C"
            elif out[start] == "E" and run < 3:
                for j in range(start, i):
                    out[j] = "C"
            start = i
    return "".join(out)

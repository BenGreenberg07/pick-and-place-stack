"""Orientation of a square, which the usual second-moment axis cannot give.

Principal-axis orientation comes from the covariance of an object's pixels, and
for a square that covariance is isotropic: the two eigenvalues are equal and
the eigenvector is whatever the noise happens to favour. Measured against
ground truth it is uniformly distributed, which is to say it carries no
information at all. That is easy to miss, because the number it returns always
looks plausible.

The fix is to use the symmetry the shape actually has. A square is invariant
under a quarter turn, so the right descriptor is the fourth-order orientational
order parameter borrowed from condensed-matter physics,

    psi4 = sum_j w_j * exp(4i * theta_j)

over the object's points, with theta_j the bearing of each point from the
centroid. Every term of a perfect square lands in phase, terms from a circular
or noisy blob cancel, and arg(psi4) / 4 recovers the square's axis. The
magnitude, normalised, doubles as a confidence: a round object scores near
zero and can be rejected rather than grasped at a made-up angle.
"""

from __future__ import annotations

import numpy as np


def square_orientation(points: np.ndarray, centroid=None) -> tuple[float, float]:
    """Return (angle, confidence) for a set of (N, 2) points forming a square.

    `angle` is in [-pi/4, pi/4) because a square grasp repeats every quarter
    turn. `confidence` is |psi4| normalised to [0, 1]; it is near 1 for a clean
    square and near 0 for a disc, and the caller should refuse to grasp below
    roughly 0.3.

    Points are weighted by r^4. Corners are what distinguish a square from a
    circle, they are the points furthest from the centroid, and the r^4 weight
    is what the continuous version of this integral carries anyway.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 8:
        return 0.0, 0.0
    c = pts.mean(axis=0) if centroid is None else np.asarray(centroid, dtype=float)
    d = pts - c
    r2 = (d**2).sum(axis=1)
    scale = r2.max()
    if scale < 1e-12:
        return 0.0, 0.0
    r2 = r2 / scale                      # keep the weights O(1) regardless of units
    theta = np.arctan2(d[:, 1], d[:, 0])
    w = r2**2
    psi = np.sum(w * np.exp(4j * theta))
    total = np.sum(w)
    if total < 1e-12:
        return 0.0, 0.0
    confidence = float(np.abs(psi) / total)
    # An r^4 weight is dominated by the corners, and a square's corners sit at
    # 45 degrees to its edges, so every corner term lands at phase pi rather
    # than 0 and psi4 points the opposite way to the edge direction. The
    # quarter turn puts the reported angle back on the edges, which is what a
    # parallel-jaw gripper has to line up with.
    angle = float(np.angle(psi) / 4.0) + np.pi / 4
    return (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4, confidence

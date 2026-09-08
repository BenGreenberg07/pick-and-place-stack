"""Planar homography: the map from camera pixels to millimetres on the table.

A camera looking at a flat table through a pinhole sees a projective transform
of that plane, so four point correspondences fully determine the mapping. This
is the piece that turns "a blob at pixel (412, 233)" into "an object 118 mm
right and 240 mm forward of the table origin", and it is the piece that is
silently wrong in most student pick-and-place demos, because a bad homography
still produces plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np


class DegenerateCorrespondences(ValueError):
    """Raised when the point correspondences do not determine a homography."""


def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centre at the origin, scale to mean distance sqrt(2).

    Skipping this is the classic way to get a homography that is numerically
    junk. Pixel coordinates run to ~1000 while the homogeneous coordinate is 1,
    so the design matrix has entries spanning six orders of magnitude and the
    SVD's small singular values drown in round-off.
    """
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    mean_dist = float(np.sqrt((centred**2).sum(axis=1)).mean())
    if mean_dist < 1e-12:
        raise DegenerateCorrespondences("all points are coincident")
    scale = np.sqrt(2.0) / mean_dist
    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )
    return centred * scale, T


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares homography H with dst ~ H @ src, by the normalised DLT.

    Each correspondence contributes two rows to a 2N x 9 system A h = 0; the
    solution is the right singular vector of A for the smallest singular value.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src and dst must both be (N, 2) arrays of the same length")
    if len(src) < 4:
        raise DegenerateCorrespondences(
            f"a homography needs at least 4 correspondences, got {len(src)}"
        )

    src_n, T_src = _normalise(src)
    dst_n, T_dst = _normalise(dst)

    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(rows)

    _, sv, Vt = np.linalg.svd(A)

    # A is 2N x 9 and a well-posed problem leaves it with rank 8: exactly one
    # null direction, which is H. `sv` only has min(2N, 9) entries, so for the
    # minimal N = 4 case the null direction is not represented in it at all.
    # Padding to length 9 makes both cases read the same way, and the second
    # smallest entry is then the one that must stay away from zero. If it does
    # not, there is a second null direction and H is not unique, which in
    # practice means three or more of the points are collinear.
    sv9 = np.concatenate([sv, np.zeros(9 - len(sv))])
    if sv9[-2] <= 1e-7 * sv9[0]:
        raise DegenerateCorrespondences(
            "correspondences do not determine a unique homography "
            "(three or more points collinear?); "
            f"conditioning {sv9[-2] / sv9[0]:.2e}"
        )
    H_n = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_n @ T_src
    if abs(H[2, 2]) < 1e-12:
        raise DegenerateCorrespondences("fitted homography is not normalisable")
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through H, dividing out the homogeneous coordinate."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    out = homo @ H.T
    w = out[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("point maps to the line at infinity under this homography")
    return out[:, :2] / w


def reprojection_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-correspondence residual, in the units of `dst`."""
    pred = apply_homography(H, src)
    return np.sqrt(((pred - np.asarray(dst, dtype=float)) ** 2).sum(axis=1))


# ------------------------------------------------------------------- robust


def fit_homography_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    threshold: float = 2.0,
    iterations: int = 400,
    seed: int = 0,
    refine: bool = True,
):
    """RANSAC homography. Returns (H, inlier_mask).

    Least squares has no defence against a single bad correspondence. One
    fiducial clicked on the wrong corner, or one auto-detected marker matched to
    the wrong ID, drags the whole fit: the residual is spread across every
    point, so the bad one is partly hidden and every good one is corrupted. The
    symptom is a calibration whose residuals are all mediocre and none obviously
    wrong.

    RANSAC inverts the logic. Fit from a minimal 4-point sample, count how many
    of the rest agree within `threshold`, keep the sample with the largest
    consensus, then refit on that consensus set. The outlier is not
    down-weighted, it is excluded, and it shows up plainly in the returned mask.

    `threshold` is in the units of `dst`, so millimetres for this package's
    pixel-to-table fits.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)
    if n < 4:
        raise DegenerateCorrespondences(f"RANSAC needs at least 4 points, got {n}")
    if n == 4:
        H = fit_homography(src, dst)
        return H, np.ones(4, dtype=bool)

    rng = np.random.default_rng(seed)
    best_H = None
    best_inliers = np.zeros(n, dtype=bool)

    for _ in range(iterations):
        idx = rng.choice(n, 4, replace=False)
        try:
            H = fit_homography(src[idx], dst[idx])
        except DegenerateCorrespondences:
            continue  # a collinear sample, which is common and not an error
        try:
            residual = reprojection_error(H, src, dst)
        except ValueError:
            continue  # the sample produced a homography that folds a point to infinity
        inliers = residual <= threshold
        if inliers.sum() > best_inliers.sum():
            best_H, best_inliers = H, inliers

    if best_H is None or best_inliers.sum() < 4:
        raise DegenerateCorrespondences(
            "RANSAC found no consensus set of at least 4 points; the "
            "correspondences are probably mismatched rather than merely noisy"
        )
    if refine:
        # Refit on the full consensus set: the winning 4-point sample defines
        # the inliers but is itself a minimal, noise-sensitive fit.
        best_H = fit_homography(src[best_inliers], dst[best_inliers])
        best_inliers = reprojection_error(best_H, src, dst) <= threshold
    return best_H, best_inliers

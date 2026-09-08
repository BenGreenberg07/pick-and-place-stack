"""Binary morphology and connected components, written out rather than called.

OpenCV has all of this. It is here in numpy because the pipeline has to run
without a camera stack installed, because the tests need a reference
implementation to check the OpenCV backend against, and because knowing what
`cv2.findContours` actually does is the difference between using a vision
library and understanding one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = padded.copy()
    for axis in (0, 1):
        stack = [out] + [
            np.roll(out, s * d, axis=axis) for d in range(1, radius + 1) for s in (+1, -1)
        ]
        out = np.all(np.stack(stack, axis=0), axis=0)
    return out[radius:-radius, radius:-radius]


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = padded.copy()
    for axis in (0, 1):
        stack = [out] + [
            np.roll(out, s * d, axis=axis) for d in range(1, radius + 1) for s in (+1, -1)
        ]
        out = np.any(np.stack(stack, axis=0), axis=0)
    return out[radius:-radius, radius:-radius]


def opening(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Erode then dilate: deletes specks smaller than the element, keeps sizes."""
    return dilate(erode(mask, radius), radius)


def closing(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilate then erode: fills pinholes and glare gaps inside an object."""
    return erode(dilate(mask, radius), radius)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = [0]

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]  # path halving
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def connected_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Two-pass union-find labelling. Returns (labels, count), background = 0.

    Pass one assigns provisional labels scanning top-to-bottom and records that
    two labels touch. Pass two collapses each equivalence class to its
    representative. 8-connectivity is the default because a block rotated 45
    degrees has staircase edges that 4-connectivity splits into fragments.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    uf = _UnionFind()

    neighbours = [(-1, 0), (0, -1)] if connectivity == 4 else [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    for y in range(h):
        row = mask[y]
        if not row.any():
            continue
        for x in np.flatnonzero(row):
            found = []
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx]:
                    found.append(labels[ny, nx])
            if not found:
                labels[y, x] = uf.add()
            else:
                lab = min(found)
                labels[y, x] = lab
                for other in found:
                    uf.union(lab, other)

    remap = np.zeros(len(uf.parent), dtype=np.int32)
    count = 0
    for i in range(1, len(uf.parent)):
        if uf.find(i) == i:
            count += 1
            remap[i] = count
    for i in range(1, len(uf.parent)):
        remap[i] = remap[uf.find(i)]
    return remap[labels], count


@dataclass
class Region:
    """One connected component, summarised by its image moments."""

    label: int
    area_px: int
    centroid_px: tuple[float, float]     # (u, v)
    angle_rad: float                     # principal axis, in [-pi/2, pi/2)
    eccentricity: float
    bbox: tuple[int, int, int, int]      # (u_min, v_min, u_max, v_max)
    # The region's own pixels, cropped to `bbox`. Kept because a downstream
    # occupancy grid modelling a wall as a circle at its centroid is wrong in
    # both directions at once, and the true footprint is right there already.
    mask: np.ndarray = field(default=None, repr=False)


def region_properties(labels: np.ndarray, count: int) -> list[Region]:
    """Second-moment summary of each labelled region.

    The orientation comes from the covariance of the pixel coordinates. Its
    eigenvector for the larger eigenvalue is the principal axis, which for a
    square block is ambiguous modulo 90 degrees; the caller has to resolve that
    against the gripper's own symmetry, which is why the angle is reported in a
    half-open half-turn rather than pretended to be unique.

    Everything is accumulated with one bincount pass per moment rather than a
    mask-per-label scan. The naive version is O(labels * pixels), which is
    unnoticeable on a clean mask and catastrophic on a noisy one that shatters
    into thousands of components.
    """
    if count == 0:
        return []
    vs, us = np.nonzero(labels)
    lab = labels[vs, us].astype(np.int64)
    us = us.astype(float)
    vs = vs.astype(float)
    n = count + 1

    area = np.bincount(lab, minlength=n).astype(float)
    su = np.bincount(lab, weights=us, minlength=n)
    sv = np.bincount(lab, weights=vs, minlength=n)
    suu = np.bincount(lab, weights=us * us, minlength=n)
    svv = np.bincount(lab, weights=vs * vs, minlength=n)
    suv = np.bincount(lab, weights=us * vs, minlength=n)

    umin = np.full(n, np.inf); umax = np.full(n, -np.inf)
    vmin = np.full(n, np.inf); vmax = np.full(n, -np.inf)
    np.minimum.at(umin, lab, us); np.maximum.at(umax, lab, us)
    np.minimum.at(vmin, lab, vs); np.maximum.at(vmax, lab, vs)

    out: list[Region] = []
    for i in range(1, n):
        a = area[i]
        if a == 0:
            continue
        cu, cv = su[i] / a, sv[i] / a
        muu = suu[i] / a - cu * cu
        mvv = svv[i] / a - cv * cv
        muv = suv[i] / a - cu * cv
        angle = 0.5 * float(np.arctan2(2 * muv, muu - mvv))
        tr, det = muu + mvv, muu * mvv - muv * muv
        disc = max(0.0, tr * tr / 4 - det)
        l1, l2 = tr / 2 + np.sqrt(disc), tr / 2 - np.sqrt(disc)
        bbox = (int(umin[i]), int(vmin[i]), int(umax[i]), int(vmax[i]))
        out.append(
            Region(
                label=i,
                area_px=int(a),
                centroid_px=(float(cu), float(cv)),
                angle_rad=float((angle + np.pi / 2) % np.pi - np.pi / 2),
                eccentricity=float(np.sqrt(1 - l2 / l1)) if l1 > 1e-12 else 0.0,
                bbox=bbox,
                mask=(labels[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1] == i),
            )
        )
    return out

"""The Hungarian algorithm, written out.

Data association is the step where tracking is actually won or lost. Greedy
nearest-neighbour matching is one line and is wrong in exactly the situation
that matters: two objects passing close to each other, where claiming the
locally best pair first forces the second object onto a worse match than the
globally optimal pairing would have given it. That is an identity swap, and
identity swaps are what a tracker exists to prevent.

This is the O(n^3) shortest-augmenting-path form with dual potentials. It is
here rather than imported from scipy because the whole package holds to numpy
only, and because the potentials are the part worth understanding: `u` and `v`
are the dual variables of the assignment LP, and maintaining them is what lets
each augmentation run against reduced costs instead of rescanning the matrix.
"""

from __future__ import annotations

import numpy as np

INF = float("inf")


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost perfect matching on rows. Returns (row, col) pairs.

    Accepts a rectangular (n, m) matrix. When there are more rows than columns
    the matrix is transposed internally, so every row is matched when n <= m and
    every column is matched otherwise. Infinite entries are allowed and are
    treated as forbidden pairs, which is how gating is expressed.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D matrix")
    if cost.size == 0:
        return []
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape

    # A finite stand-in for INF, large enough to never win but small enough to
    # keep the potentials finite. Real infinities poison the dual updates.
    finite = cost[np.isfinite(cost)]
    big = (float(finite.max() - finite.min()) + 1.0) * (n + m) + 1.0 if finite.size else 1.0
    work = np.where(np.isfinite(cost), cost, big)

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)     # p[j] = row assigned to column j
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = work[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    pairs = [(int(p[j] - 1), int(j - 1)) for j in range(1, m + 1) if p[j] > 0]
    # Drop pairs that only exist because a forbidden entry was padded.
    pairs = [(r, c) for r, c in pairs if np.isfinite(cost[r, c])]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


def assignment_cost(cost: np.ndarray, pairs) -> float:
    cost = np.asarray(cost, dtype=float)
    return float(sum(cost[r, c] for r, c in pairs))

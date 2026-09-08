"""String-pulling: turn a staircase grid path into a short polyline.

A* on a grid can only move in eight directions, so its output is a staircase
even in an empty room. Shortcutting removes every waypoint that the straight
line already passes, which typically drops a path from hundreds of cells to a
handful of corners and removes the visible stair-stepping from the arm's
motion.
"""

from __future__ import annotations

import numpy as np

from ..frames import Point2D
from .occupancy import OccupancyGrid

Cell = tuple[int, int]


def supercover_line(a: Cell, b: Cell) -> list[Cell]:
    """Every cell a segment from a to b touches, corners included.

    This is deliberately not Bresenham. Bresenham picks one cell per column and
    can hop diagonally between two blocked cells, so a line-of-sight test built
    on it will happily declare a route clear that passes through a wall.
    """
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0
    cells = [(r, c)]
    err = dr - dc
    while (r, c) != (r1, c1):
        e2 = 2 * err
        stepped_r = stepped_c = False
        if e2 > -dc:
            err -= dc
            r += sr
            stepped_r = True
        if e2 < dr:
            err += dr
            c += sc
            stepped_c = True
        if stepped_r and stepped_c:
            # A true diagonal move also grazes both orthogonal neighbours.
            cells.append((r, c - sc))
            cells.append((r - sr, c))
        cells.append((r, c))
    return cells


def line_of_sight(grid: OccupancyGrid, a: Cell, b: Cell) -> bool:
    return all(grid.is_free(cell) for cell in supercover_line(a, b))


def shortcut(grid: OccupancyGrid, path: list[Cell]) -> list[Cell]:
    """Greedy forward shortcutting: from each kept node, jump to the furthest
    later node still in line of sight."""
    if len(path) < 3:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_of_sight(grid, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def resample(points: list[Point2D], spacing_mm: float) -> list[Point2D]:
    """Even spacing along a polyline, so the trajectory has a uniform step."""
    if len(points) < 2:
        return list(points)
    frame = points[0].frame
    xy = np.array([[p.x, p.y] for p in points], dtype=float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-9:
        return [points[0], points[-1]]
    n = max(2, int(np.ceil(total / spacing_mm)) + 1)
    targets = np.linspace(0.0, total, n)
    out_x = np.interp(targets, s, xy[:, 0])
    out_y = np.interp(targets, s, xy[:, 1])
    return [Point2D(float(x), float(y), frame) for x, y in zip(out_x, out_y)]


def path_length_mm(points: list[Point2D]) -> float:
    if len(points) < 2:
        return 0.0
    xy = np.array([[p.x, p.y] for p in points], dtype=float)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())

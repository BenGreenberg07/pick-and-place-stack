"""A* on the occupancy grid, with the heuristic left as a knob.

The heuristic is a parameter rather than a constant so the repo can actually
measure the claim everyone repeats: that an admissible heuristic gives an
optimal path and a better-informed one expands fewer nodes. `benchmark.py`
runs the comparison. Manhattan on an 8-connected grid is included precisely
because it is inadmissible there, and the benchmark shows it returning shorter
search times and longer paths.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..frames import Point2D
from .occupancy import OccupancyGrid, OutsideGrid

Cell = tuple[int, int]
Heuristic = Callable[[Cell, Cell], float]

SQRT2 = math.sqrt(2.0)


class NoPathFound(RuntimeError):
    """Raised when the goal is unreachable from the start on this grid."""


# ------------------------------------------------------------------ heuristics


def h_zero(a: Cell, b: Cell) -> float:
    """Uninformed. A* with this is exactly Dijkstra, and is the baseline the
    others are measured against."""
    return 0.0


def h_manhattan(a: Cell, b: Cell) -> float:
    """|dr| + |dc|. Admissible on a 4-connected grid, NOT on an 8-connected one:
    a diagonal step costs sqrt(2) but Manhattan charges it 2, so the heuristic
    overestimates and A* can return a suboptimal path."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def h_euclidean(a: Cell, b: Cell) -> float:
    """Admissible everywhere, but loose on a grid: it allows motion in
    directions the grid does not have, so it under-informs the search."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def h_octile(a: Cell, b: Cell) -> float:
    """The exact cost of an obstacle-free 8-connected traverse: take as many
    diagonals as possible, then go straight. Admissible AND tight, which makes
    it the right default here."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)


HEURISTICS: dict[str, Heuristic] = {
    "zero": h_zero,
    "manhattan": h_manhattan,
    "euclidean": h_euclidean,
    "octile": h_octile,
}


# ----------------------------------------------------------------------- A*


@dataclass
class PlanResult:
    path_cells: list[Cell]
    cost: float
    expanded: int          # nodes popped from the open set
    generated: int         # nodes ever pushed
    heuristic: str
    _grid: OccupancyGrid = field(repr=False)

    def world_path(self) -> list[Point2D]:
        return [self._grid.to_world(c) for c in self.path_cells]

    @property
    def length_mm(self) -> float:
        return self.cost * self._grid.resolution_mm


_NEIGHBOURS_8 = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
]
_NEIGHBOURS_4 = _NEIGHBOURS_8[:4]


def astar(
    grid: OccupancyGrid,
    start: Cell,
    goal: Cell,
    *,
    heuristic: str = "octile",
    connectivity: int = 8,
    tie_breaker: float = 1e-3,
) -> PlanResult:
    """Shortest path from `start` to `goal`, or NoPathFound.

    `tie_breaker` scales the heuristic by (1 + eps). On an open grid, huge
    numbers of nodes share the same f value and A* explores all of them in an
    arbitrary order; nudging f to prefer nodes closer to the goal collapses
    that plateau and cuts expansions substantially. The path can be longer than
    optimal by at most a factor of (1 + eps), which at 1e-3 is nothing.
    """
    if heuristic not in HEURISTICS:
        raise ValueError(f"unknown heuristic {heuristic!r}; have {sorted(HEURISTICS)}")
    h = HEURISTICS[heuristic]
    moves = _NEIGHBOURS_8 if connectivity == 8 else _NEIGHBOURS_4

    if not grid.is_free(start):
        raise OutsideGrid(f"start cell {start} is blocked or off the grid")
    if not grid.is_free(goal):
        raise OutsideGrid(f"goal cell {goal} is blocked or off the grid")

    rows, cols = grid.shape
    g_score = np.full((rows, cols), np.inf)
    g_score[start] = 0.0
    came_from: dict[Cell, Cell] = {}
    closed = np.zeros((rows, cols), dtype=bool)

    open_heap: list[tuple[float, float, Cell]] = [(0.0, 0.0, start)]
    expanded = 0
    generated = 1

    while open_heap:
        f, g, cur = heapq.heappop(open_heap)
        if closed[cur]:
            continue  # a stale duplicate; the heap has no decrease-key
        closed[cur] = True
        expanded += 1

        if cur == goal:
            path = [cur]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return PlanResult(path, float(g), expanded, generated, heuristic, grid)

        for dr, dc, step in moves:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not grid.is_free(nxt) or closed[nxt]:
                continue
            # Refuse to cut a corner between two blocked cells; without this the
            # path squeezes through a diagonal gap the real robot cannot fit.
            if dr and dc and (
                not grid.is_free((cur[0] + dr, cur[1]))
                or not grid.is_free((cur[0], cur[1] + dc))
            ):
                continue
            tentative = g + step
            if tentative < g_score[nxt]:
                g_score[nxt] = tentative
                came_from[nxt] = cur
                heapq.heappush(
                    open_heap,
                    (tentative + (1.0 + tie_breaker) * h(nxt, goal), tentative, nxt),
                )
                generated += 1

    raise NoPathFound(
        f"no route from {start} to {goal}; the goal is walled off "
        f"after {expanded} expansions"
    )


def dijkstra_cost(grid: OccupancyGrid, start: Cell, goal: Cell, connectivity: int = 8) -> float:
    """Reference optimal cost, used by the tests to check A* optimality."""
    return astar(grid, start, goal, heuristic="zero", connectivity=connectivity,
                 tie_breaker=0.0).cost

"""Occupancy grids in table millimetres, with configuration-space inflation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import Point2D
from ..perception.morphology import dilate, erode
from ..transforms.calibration import WorkspaceBounds


class OutsideGrid(ValueError):
    """Raised when a requested cell lies outside the grid."""


@dataclass
class OccupancyGrid:
    """A boolean grid over the workspace. True means blocked.

    Indexing is grid[row, col] with row along table +y and col along table +x,
    so `to_world` and `to_cell` are the only two places the row/col versus x/y
    swap happens. Every off-by-one in a gridworld planner traces back to that
    swap being done in more than one place.
    """

    blocked: np.ndarray
    bounds: WorkspaceBounds
    resolution_mm: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.blocked.shape

    def to_cell(self, p: Point2D) -> tuple[int, int]:
        p.require("table")
        col = int((p.x - self.bounds.x_min) / self.resolution_mm)
        row = int((p.y - self.bounds.y_min) / self.resolution_mm)
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            raise OutsideGrid(
                f"({p.x:.1f}, {p.y:.1f}) mm is outside the workspace "
                f"x[{self.bounds.x_min:.0f}, {self.bounds.x_max:.0f}] "
                f"y[{self.bounds.y_min:.0f}, {self.bounds.y_max:.0f}]"
            )
        return row, col

    def to_world(self, cell: tuple[int, int]) -> Point2D:
        """Centre of the cell, so a path never sits on a cell boundary."""
        row, col = cell
        return Point2D(
            self.bounds.x_min + (col + 0.5) * self.resolution_mm,
            self.bounds.y_min + (row + 0.5) * self.resolution_mm,
            "table",
        )

    def is_free(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return (
            0 <= row < self.shape[0]
            and 0 <= col < self.shape[1]
            and not self.blocked[row, col]
        )

    def nearest_free(self, cell: tuple[int, int], max_radius: int = 12) -> tuple[int, int]:
        """The closest free cell to `cell`, for nudging a start or goal that
        landed inside an inflated obstacle. Raises if there is nothing close."""
        if self.is_free(cell):
            return cell
        for r in range(1, max_radius + 1):
            best, best_d = None, np.inf
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    if max(abs(dr), abs(dc)) != r:
                        continue
                    cand = (cell[0] + dr, cell[1] + dc)
                    if self.is_free(cand):
                        d = dr * dr + dc * dc
                        if d < best_d:
                            best, best_d = cand, d
            if best is not None:
                return best
        raise OutsideGrid(
            f"no free cell within {max_radius} cells of {cell}; the workspace is "
            "blocked here"
        )


def build_grid(
    bounds: WorkspaceBounds,
    obstacles: list[tuple[Point2D, float]] | None = None,
    *,
    footprints: list[np.ndarray] | None = None,
    resolution_mm: float = 5.0,
    clearance_mm: float = 18.0,
) -> OccupancyGrid:
    """Rasterise obstacles, then inflate by the robot's clearance radius.

    Inflating the obstacles instead of shrinking the robot is the standard
    configuration-space trick: after inflation the robot is a point, so the
    planner can search cell centres and never has to reason about the gripper's
    footprint again. Every path it returns is then automatically collision-free
    for the real, non-point robot.

    Obstacles come in two forms. `obstacles` is a list of (centre in the table
    frame, radius in mm), which is the right model for a compact object.
    `footprints` is a list of (N, 2) arrays of table-frame points, which is what
    the perception stage actually produces: the object's real outline, so a
    wall is rasterised as a wall rather than as a disc that is simultaneously
    too big across and too small along.
    """
    obstacles = obstacles or []
    rows = max(1, int(np.ceil(bounds.height / resolution_mm)))
    cols = max(1, int(np.ceil(bounds.width / resolution_mm)))
    grid = np.zeros((rows, cols), dtype=bool)

    ys = bounds.y_min + (np.arange(rows) + 0.5) * resolution_mm
    xs = bounds.x_min + (np.arange(cols) + 0.5) * resolution_mm
    XX, YY = np.meshgrid(xs, ys)

    for centre, radius in obstacles:
        centre.require("table")
        grid |= (XX - centre.x) ** 2 + (YY - centre.y) ** 2 <= radius**2

    for pts in footprints or []:
        pts = np.asarray(pts, dtype=float)
        if len(pts) == 0:
            continue
        cols_i = np.floor((pts[:, 0] - bounds.x_min) / resolution_mm).astype(int)
        rows_i = np.floor((pts[:, 1] - bounds.y_min) / resolution_mm).astype(int)
        keep = (rows_i >= 0) & (rows_i < rows) & (cols_i >= 0) & (cols_i < cols)
        grid[rows_i[keep], cols_i[keep]] = True
    # Subsampling the footprint leaves pin-holes between the sampled points, so
    # close them before inflating; an unfilled hole is a phantom doorway the
    # planner will happily route through.
    if footprints:
        grid = erode(dilate(grid, 1), 1)

    inflate_cells = int(np.ceil(clearance_mm / resolution_mm))
    if inflate_cells > 0 and grid.any():
        grid = dilate(grid, inflate_cells)
    return OccupancyGrid(grid, bounds, resolution_mm)

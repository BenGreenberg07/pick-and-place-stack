"""Collision checking for the whole arm, not just its fingertip.

Planning a collision-free path for the end effector is the easy half of the
problem, and the half that demos stop at. The arm is a chain of links sweeping
through the same workspace, and a tip path that threads a gap perfectly will
still drag the elbow straight through the obstacle beside it.

So this module checks the linkage. Every link is treated as a capsule (a
segment with a radius) in the table frame, and a configuration is in collision
if any capsule overlaps an obstacle. Because the tip clearance and the link
half-thickness are different numbers, the check re-inflates the grid from its
raw obstacles rather than reusing the tip's inflated copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import Point2D
from ..kinematics.arm import PlanarArm
from ..transforms.calibration import CameraCalibration
from .occupancy import OccupancyGrid


@dataclass
class CollisionReport:
    in_collision: bool
    link_index: int | None = None      # first offending link, base-most first
    point: Point2D | None = None       # roughly where, in the table frame

    def __bool__(self) -> bool:
        return self.in_collision


class ArmCollisionChecker:
    """Whole-linkage collision queries against an occupancy grid.

    `link_radius_mm` is the half-thickness of the arm's structure. It is
    normally smaller than the gripper clearance used for the tip, which is why
    this holds its own inflated grid.
    """

    def __init__(
        self,
        arm: PlanarArm,
        grid: OccupancyGrid,
        calibration: CameraCalibration,
        *,
        link_radius_mm: float = 12.0,
        sample_mm: float = 4.0,
    ) -> None:
        self.arm = arm
        self.calibration = calibration
        self.link_radius_mm = link_radius_mm
        self.sample_mm = sample_mm
        self.grid = grid.inflate(link_radius_mm)
        self._checks = 0

    @property
    def checks(self) -> int:
        """How many configurations have been tested. Collision checking is the
        inner loop of every sampling-based planner, so this is the number that
        actually explains an RRT's runtime."""
        return self._checks

    def link_points_table(self, q) -> np.ndarray:
        """Joint origins in the table frame, base to tip, as an (n+1, 2) array."""
        pts_base = self.arm.joint_positions(q)
        out = np.empty_like(pts_base)
        for i, (x, y) in enumerate(pts_base):
            p = self.calibration.base_to_table(Point2D(float(x), float(y), "base"))
            out[i] = (p.x, p.y)
        return out

    def check(self, q) -> CollisionReport:
        """Test one configuration. Reports the first offending link, if any."""
        self._checks += 1
        pts = self.link_points_table(q)
        for i in range(len(pts) - 1):
            hit = self._segment_hit(pts[i], pts[i + 1])
            if hit is not None:
                return CollisionReport(True, i, Point2D(float(hit[0]), float(hit[1]), "table"))
        return CollisionReport(False)

    def collides(self, q) -> bool:
        return self.check(q).in_collision

    def _segment_hit(self, a: np.ndarray, b: np.ndarray):
        """Walk a segment and return the first blocked point.

        Sampling finer than one grid cell is what stops a link from stepping
        over a thin obstacle between two samples.
        """
        length = float(np.hypot(*(b - a)))
        n = max(2, int(np.ceil(length / min(self.sample_mm, self.grid.resolution_mm))) + 1)
        for t in np.linspace(0.0, 1.0, n):
            p = a + (b - a) * t
            point = Point2D(float(p[0]), float(p[1]), "table")
            if not self.grid.bounds.contains(point):
                # Off the table is not an obstacle. The arm may swing outside
                # the camera's view; it simply cannot be verified there.
                continue
            if not self.grid.is_free(self.grid.to_cell(point)):
                return p
        return None

    def path_clear(self, q0, q1, *, max_step: float = 0.06) -> bool:
        """Test a straight line in joint space by subdividing it.

        The subdivision is bisection-ordered rather than sequential: the
        midpoint is checked first, then the quarters, and so on. A colliding
        edge usually fails somewhere in its middle, so this finds the failure in
        a handful of checks instead of marching all the way in from one end.
        """
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        steps = int(np.ceil(float(np.abs(q1 - q0).max()) / max_step))
        if steps <= 1:
            return not self.collides(q1)
        order: list[float] = []
        stack = [(0.0, 1.0, 0)]
        while stack:
            lo, hi, depth = stack.pop()
            mid = 0.5 * (lo + hi)
            order.append(mid)
            if (hi - lo) * steps > 2 and depth < 12:
                stack.append((lo, mid, depth + 1))
                stack.append((mid, hi, depth + 1))
        for t in [0.0, 1.0] + order:
            if self.collides(q0 + (q1 - q0) * t):
                return False
        return True


def first_clear_solution(solutions, checker: ArmCollisionChecker):
    """The first IK solution whose whole linkage is clear, or None.

    This is the join between inverse kinematics and collision checking, and it
    is why enumerating both elbow branches is worth the trouble: the elbow-up
    solution frequently puts the elbow inside the obstacle the tip just routed
    around, while the elbow-down solution for the very same target is fine.
    """
    for s in solutions:
        if not checker.collides(s.q):
            return s
    return None

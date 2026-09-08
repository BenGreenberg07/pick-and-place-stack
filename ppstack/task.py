"""Pick, carry, place, repeat: the layer that makes the name honest.

`pipeline.py` answers "where is the block and how do I reach it". This module
answers "there are five blocks and three bins, what should the arm do, and in
what order". Three things here are worth more than they look:

* **Sequencing.** Visiting the blocks in the order the detector happened to
  list them is arbitrary and usually bad. The order is a small travelling
  salesman problem and is solved as one.
* **The world changes as the task runs.** A block that has been picked up is no
  longer an obstacle. A block being carried makes the gripper effectively
  fatter, so the clearance the planner inflates by has to grow while it is held.
  Both are handled by rebuilding the grid between segments rather than planning
  once against a stale snapshot.
* **Carrying is a different problem from reaching.** The arm has to get the
  block to the bin without dragging it, or an elbow, through anything.

There are two collision problems here, not one, and conflating them is what
makes a naive version reject everything. The gripper travels in the block
plane, so it has to avoid every block on the table. The links travel above that
plane, so they pass harmlessly over a 20 mm block and only have to avoid things
tall enough to reach them. The planner therefore builds two grids from the same
detections: the tip is planned against everything, and the whole-arm collision
checker sees only the tall obstacles.

Everything is planar. There is no z axis, so grasping and releasing are
instantaneous state changes at a waypoint rather than a descend-close-lift
sequence. That is a real limitation, stated rather than hidden.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .frames import Point2D
from .kinematics.arm import PlanarArm
from .kinematics.ik import IKUnreachable, solve_collision_free
from .kinematics.trajectory import JointTrajectory, follow_path
from .pipeline import WorldObject
from .planning.astar import NoPathFound, astar
from .planning.collision import ArmCollisionChecker
from .planning.occupancy import OccupancyGrid, OutsideGrid, build_grid
from .planning.smoothing import path_length_mm, resample, shortcut
from .transforms.calibration import CameraCalibration


@dataclass(frozen=True)
class Bin:
    """A drop-off location, in the table frame."""

    name: str
    x: float
    y: float
    accepts: tuple[str, ...] = ()

    @property
    def position(self) -> Point2D:
        return Point2D(self.x, self.y, "table")

    def takes(self, colour: str) -> bool:
        return not self.accepts or colour in self.accepts


@dataclass
class Segment:
    """One motion, plus what the gripper is doing during it."""

    kind: str                      # "approach" or "carry"
    obj: WorldObject
    bin: Bin | None
    waypoints: list[Point2D]
    trajectory: JointTrajectory
    holding: bool
    grid: OccupancyGrid = field(repr=False, default=None)
    # The objects still on the table during this segment, kept so the animation
    # can show the world shrinking as blocks are removed from it.
    grid_objects: list = field(repr=False, default_factory=list)

    @property
    def length_mm(self) -> float:
        return path_length_mm(self.waypoints)


@dataclass
class TaskExecution:
    segments: list[Segment] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    tour_mm: float = 0.0             # planned tour, in the order chosen
    tour_mm_detection_order: float = 0.0   # the same jobs, in the order detected
    executed_mm: float = 0.0         # actual path length of the solved segments
    collision_checks: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.segments)

    @property
    def picks(self) -> int:
        return sum(1 for s in self.segments if s.kind == "carry")

    def summary(self) -> str:
        lines = [
            f"order: {' -> '.join(self.order) if self.order else '(nothing pickable)'}",
            f"{self.picks} block(s) placed in {len(self.segments)} segments, "
            f"{self.executed_mm:.0f} mm of solved path",
            f"planned tour {self.tour_mm:.0f} mm vs {self.tour_mm_detection_order:.0f} mm "
            f"in detection order "
            f"({100 * (1 - self.tour_mm / max(self.tour_mm_detection_order, 1e-9)):.0f}% saved "
            f"by sequencing)",
            f"{self.collision_checks} whole-arm collision checks",
        ]
        lines += [f"  skipped {name}: {why}" for name, why in self.skipped]
        return "\n".join(lines)


# ------------------------------------------------------------------ ordering


def _leg_cost(a: Point2D, b: Point2D) -> float:
    return a.distance_to(b)


def tour_cost(start: Point2D, jobs: list[tuple[WorldObject, Bin]]) -> float:
    """Total travel for a given order, counting both the empty and laden legs."""
    total = 0.0
    here = start
    for obj, dest in jobs:
        total += _leg_cost(here, obj.position) + _leg_cost(obj.position, dest.position)
        here = dest.position
    return total


def sequence_jobs(
    start: Point2D, jobs: list[tuple[WorldObject, Bin]]
) -> list[tuple[WorldObject, Bin]]:
    """Order the pick-and-place jobs to cut total travel.

    This is a travelling salesman problem with a twist: each stop has a fixed
    internal leg (block to its bin) whose cost the ordering cannot change, so
    only the empty legs between a bin and the next block are up for grabs.

    Nearest-neighbour builds a starting tour, then 2-opt improves it by
    reversing sub-tours while any reversal helps. 2-opt is the right amount of
    machinery here: it removes the crossed paths that greedy construction
    reliably produces, it is a few lines, and for the handful of blocks on a
    table an exact solver would be spending its time to save nothing. For six
    or fewer jobs every permutation is cheap, so it just enumerates them and
    the answer is exactly optimal.
    """
    if len(jobs) <= 1:
        return list(jobs)
    if len(jobs) <= 6:
        return min(
            (list(p) for p in itertools.permutations(jobs)),
            key=lambda order: tour_cost(start, order),
        )

    remaining = list(jobs)
    tour: list[tuple[WorldObject, Bin]] = []
    here = start
    while remaining:
        nxt = min(remaining, key=lambda j: _leg_cost(here, j[0].position))
        remaining.remove(nxt)
        tour.append(nxt)
        here = nxt[1].position

    best = tour_cost(start, tour)
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 2, len(tour) + 1):
                candidate = tour[:i] + tour[i:j][::-1] + tour[j:]
                cost = tour_cost(start, candidate)
                if cost < best - 1e-9:
                    tour, best, improved = candidate, cost, True
    return tour


# ----------------------------------------------------------------- execution


class TaskExecutor:
    """Runs a full multi-object pick-and-place cycle.

    `carry_clearance_mm` is added to the planner's inflation radius while a
    block is held, because the thing that has to fit through the gap is now the
    gripper plus its cargo.
    """

    def __init__(
        self,
        arm: PlanarArm,
        calibration: CameraCalibration,
        *,
        grid_resolution_mm: float = 5.0,
        clearance_mm: float = 17.0,
        carry_clearance_mm: float = 10.0,
        link_radius_mm: float = 10.0,
        waypoint_spacing_mm: float = 7.0,
        collision_aware: bool = True,
    ) -> None:
        self.arm = arm
        self.calibration = calibration
        self.grid_resolution_mm = grid_resolution_mm
        self.clearance_mm = clearance_mm
        self.carry_clearance_mm = carry_clearance_mm
        self.link_radius_mm = link_radius_mm
        self.waypoint_spacing_mm = waypoint_spacing_mm
        self.collision_aware = collision_aware

    # -------------------------------------------------------------- helpers

    def _grid(
        self, obstacles: list[WorldObject], extra_clearance: float, *, tall_only: bool = False
    ) -> OccupancyGrid:
        """Rasterise obstacles into a grid.

        `tall_only` keeps just the objects tall enough to reach the linkage,
        which is what the whole-arm collision checker must be built from. Using
        the tip's grid for the arm as well would have the elbow colliding with
        flat blocks it actually passes cleanly above, and the planner would
        report an impossible workspace.
        """
        selected = [o for o in obstacles if o.is_obstacle] if tall_only else obstacles
        footprints = [o.footprint_table for o in selected if len(o.footprint_table)]
        fallback = [
            (o.position, max(8.0, o.size_mm * 0.75))
            for o in selected
            if not len(o.footprint_table)
        ]
        return build_grid(
            self.calibration.bounds,
            fallback,
            footprints=footprints,
            resolution_mm=self.grid_resolution_mm,
            clearance_mm=self.clearance_mm + extra_clearance,
        )

    def _checker(self, obstacles: list[WorldObject]) -> ArmCollisionChecker | None:
        if not self.collision_aware:
            return None
        return ArmCollisionChecker(
            self.arm,
            self._grid(obstacles, 0.0, tall_only=True),
            self.calibration,
            link_radius_mm=self.link_radius_mm,
        )

    def _path(self, grid: OccupancyGrid, start: Point2D, goal: Point2D) -> list[Point2D]:
        start_cell = grid.nearest_free(grid.to_cell(start))
        goal_cell = grid.nearest_free(grid.to_cell(goal))
        plan = astar(grid, start_cell, goal_cell, heuristic="octile")
        corners = [grid.to_world(c) for c in shortcut(grid, plan.path_cells)]
        corners[0], corners[-1] = start, goal
        return resample(corners, self.waypoint_spacing_mm)

    def _trajectory(self, waypoints: list[Point2D], q_start, checker) -> JointTrajectory:
        """Solve the path, preferring collision-free postures at every waypoint.

        With collision awareness on, each waypoint's whole solution manifold is
        enumerated and the best clear posture chosen, seeded from the previous
        one so the arm does not reconfigure between neighbouring points. With it
        off, this is the tip-only behaviour, which is what the benchmark
        compares against.
        """
        base_pts = [self.calibration.table_to_base.apply(p) for p in waypoints]
        if not self.collision_aware or checker is None:
            return follow_path(self.arm, base_pts, q_start=q_start)

        qs, tips, flips, worst = [], [], 0, 0.0
        q_prev = np.asarray(q_start, dtype=float)
        for i, p in enumerate(base_pts):
            sol = solve_collision_free(self.arm, p, checker, q_seed=q_prev, n_orientations=48)
            if sol is None:
                raise IKUnreachable(
                    f"waypoint {i} of {len(base_pts)} at "
                    f"({waypoints[i].x:.0f}, {waypoints[i].y:.0f}) mm is reachable, but "
                    "every arm posture that reaches it puts a link inside an obstacle"
                )
            if i > 0 and float(np.abs(sol.q - q_prev).max()) > 0.6:
                flips += 1
            q_prev = sol.q
            qs.append(sol.q)
            tip = self.arm.forward(sol.q)
            tips.append((tip.x, tip.y))
            worst = max(worst, sol.position_error)
        return JointTrajectory(
            times=np.linspace(0.0, 2.0, len(qs)),
            q=np.vstack(qs),
            tip=np.array(tips),
            branch_flips=flips,
            max_position_error=worst,
        )

    # ------------------------------------------------------------------ run

    def run(
        self,
        objects: list[WorldObject],
        bins: list[Bin],
        *,
        q_start,
        pick_colors: tuple[str, ...] = ("red", "blue", "green", "yellow"),
    ) -> TaskExecution:
        """Plan and solve the whole cycle. Unpickable blocks are skipped, not fatal."""
        execution = TaskExecution()
        q = np.asarray(q_start, dtype=float)
        here = self.calibration.base_to_table(self.arm.fk_point(q))

        jobs: list[tuple[WorldObject, Bin]] = []
        for obj in objects:
            if obj.is_obstacle or obj.color not in pick_colors:
                continue
            dest = next((b for b in bins if b.takes(obj.color)), None)
            if dest is None:
                execution.skipped.append((obj.color, "no bin accepts this colour"))
                continue
            jobs.append((obj, dest))

        if not jobs:
            return execution

        ordered = sequence_jobs(here, jobs)
        execution.tour_mm = tour_cost(here, ordered)
        execution.tour_mm_detection_order = tour_cost(here, jobs)
        execution.order = [f"{o.color}->{b.name}" for o, b in ordered]

        # The rest pose itself has to be legal before anything else can be
        # planned from it. If it is not, look for another posture with the tip
        # in the same place, which is exactly what the redundancy is for.
        start_checker = self._checker(objects)
        if start_checker is not None and start_checker.collides(q):
            tip = self.arm.forward(q).point
            rescue = solve_collision_free(self.arm, tip, start_checker, q_seed=q)
            if rescue is None:
                execution.skipped.append(
                    ("(start)", "the arm's rest pose is in collision and no posture "
                                "with the same tip position is clear")
                )
                return execution
            q = rescue.q

        remaining = [o for o in objects]
        for obj, dest in ordered:
            others = [o for o in remaining if o is not obj]
            try:
                # Leg one: reach the block with an empty gripper.
                grid = self._grid(others, 0.0)
                checker = self._checker(others)
                wps = self._path(grid, here, obj.position)
                traj = self._trajectory(wps, q, checker)
                execution.segments.append(
                    Segment("approach", obj, dest, wps, traj, False, grid, list(remaining))
                )
                q, here = traj.q[-1], obj.position
                if checker is not None:
                    execution.collision_checks += checker.checks

                # Leg two: carry it to the bin. The block is no longer an
                # obstacle, and the gripper is now fatter by however big it is.
                grid = self._grid(others, self.carry_clearance_mm)
                checker = self._checker(others)
                wps = self._path(grid, here, dest.position)
                traj = self._trajectory(wps, q, checker)
                execution.segments.append(
                    Segment("carry", obj, dest, wps, traj, True, grid, list(others))
                )
                q, here = traj.q[-1], dest.position
                if checker is not None:
                    execution.collision_checks += checker.checks

                remaining = others  # it is in the bin now
            except (NoPathFound, OutsideGrid, IKUnreachable) as exc:
                execution.skipped.append((obj.color, str(exc)))
                continue

        execution.executed_mm = sum(s.length_mm for s in execution.segments)
        return execution

"""The stack, wired end to end, with every interface allowed to fail loudly.

Perception hands the planner numbers that are sometimes wrong, sometimes
missing, and sometimes physically impossible. A demo that assumes otherwise
looks fine on the author's desk and falls over on anyone else's. So each stage
here returns a `StageReport`, a failure is attributed to the stage that caused
it, and the pipeline returns a result object rather than raising into the
caller's lap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .frames import Point2D
from .kinematics.arm import PlanarArm
from .kinematics.ik import IKUnreachable, solve
from .kinematics.trajectory import JointTrajectory, follow_path
from .perception.detector import BlockDetector, Detection
from .perception.orientation import square_orientation
from .planning.astar import NoPathFound, PlanResult, astar
from .planning.occupancy import OccupancyGrid, OutsideGrid, build_grid
from .planning.smoothing import path_length_mm, resample, shortcut
from .transforms.calibration import CalibrationError, CameraCalibration

STAGES = ("perception", "transform", "planning", "kinematics")


@dataclass
class StageReport:
    stage: str
    ok: bool
    duration_ms: float
    detail: str = ""


@dataclass
class WorldObject:
    """A detection promoted into table millimetres."""

    color: str
    position: Point2D          # table frame
    grasp_angle_rad: float     # table frame, wrapped into [-pi/4, pi/4)
    grasp_confidence: float    # |psi4| in [0, 1]; below ~0.3 the angle is guesswork
    size_mm: float
    is_obstacle: bool
    source: Detection = field(repr=False)
    footprint_table: np.ndarray = field(default=None, repr=False)


@dataclass
class PipelineResult:
    reports: list[StageReport]
    objects: list[WorldObject] = field(default_factory=list)
    target: WorldObject | None = None
    grid: OccupancyGrid | None = None
    plan: PlanResult | None = None
    waypoints: list[Point2D] = field(default_factory=list)
    trajectory: JointTrajectory | None = None
    failure_stage: str | None = None
    failure: str = ""

    @property
    def ok(self) -> bool:
        return self.failure_stage is None

    @property
    def total_ms(self) -> float:
        return sum(r.duration_ms for r in self.reports)

    def summary(self) -> str:
        lines = [
            f"{r.stage:<12} {'ok  ' if r.ok else 'FAIL'} {r.duration_ms:7.1f} ms  {r.detail}"
            for r in self.reports
        ]
        if not self.ok:
            lines.append(f"stopped in {self.failure_stage}: {self.failure}")
        return "\n".join(lines)


class PickAndPlacePipeline:
    def __init__(
        self,
        arm: PlanarArm,
        calibration: CameraCalibration,
        detector: BlockDetector | None = None,
        *,
        grid_resolution_mm: float = 5.0,
        clearance_mm: float = 22.0,
        obstacle_radius_scale: float = 0.75,
        waypoint_spacing_mm: float = 6.0,
        min_grasp_confidence: float = 0.30,
    ) -> None:
        self.arm = arm
        self.calibration = calibration
        self.detector = detector or BlockDetector()
        self.grid_resolution_mm = grid_resolution_mm
        self.clearance_mm = clearance_mm
        self.obstacle_radius_scale = obstacle_radius_scale
        self.waypoint_spacing_mm = waypoint_spacing_mm
        self.min_grasp_confidence = min_grasp_confidence

    # ------------------------------------------------------------------ run

    def run(
        self,
        frame: np.ndarray,
        *,
        target_color: str = "red",
        q_start=None,
    ) -> PipelineResult:
        result = PipelineResult(reports=[])

        objects = self._stage(result, "perception", lambda: self._perceive(frame))
        if objects is None:
            return result
        result.objects = objects

        picked = self._stage(
            result, "transform", lambda: self._select_target(objects, target_color)
        )
        if picked is None:
            return result
        result.target = picked

        planned = self._stage(
            result, "planning", lambda: self._plan(objects, picked, q_start)
        )
        if planned is None:
            return result
        result.grid, result.plan, result.waypoints = planned

        traj = self._stage(
            result, "kinematics", lambda: self._solve(result.waypoints, q_start)
        )
        if traj is None:
            return result
        result.trajectory = traj
        return result

    def _stage(self, result: PipelineResult, name: str, fn):
        t0 = time.perf_counter()
        try:
            value = fn()
        except _StageFailure as exc:
            result.reports.append(
                StageReport(name, False, (time.perf_counter() - t0) * 1e3, str(exc))
            )
            result.failure_stage = name
            result.failure = str(exc)
            return None
        dt = (time.perf_counter() - t0) * 1e3
        value, detail = value
        result.reports.append(StageReport(name, True, dt, detail))
        return value

    # --------------------------------------------------------------- stages

    def _perceive(self, frame: np.ndarray):
        detections = self.detector.detect(frame)
        if not detections:
            raise _StageFailure("no objects detected; check lighting or the HSV bands")
        objects = [self._to_world(d) for d in detections]
        inside = [o for o in objects if self.calibration.bounds.contains(o.position)]
        dropped = len(objects) - len(inside)
        if not inside:
            raise _StageFailure(
                f"all {len(objects)} detections fell outside the calibrated workspace; "
                "the calibration is probably wrong"
            )
        detail = f"{len(inside)} objects" + (f" ({dropped} off-table, dropped)" if dropped else "")
        return inside, detail

    def _to_world(self, d: Detection) -> WorldObject:
        pos = self.calibration.pixel_to_table(d.u, d.v)
        mm_per_px = self.calibration.pixel_scale_mm(d.u, d.v)
        size = float(np.sqrt(d.area_px) * mm_per_px)
        footprint = (
            self.calibration.pixels_to_table(d.footprint_px)
            if d.footprint_px is not None and len(d.footprint_px)
            else np.empty((0, 2))
        )
        # The grasp angle is measured on the footprint after it has been mapped
        # into the table frame, never in pixels. Perspective shears a square
        # into a general quadrilateral, so an angle measured in the image is
        # wrong by however much the camera is tilted, and wrong by a different
        # amount at each corner of the table.
        angle, confidence = (
            square_orientation(footprint, centroid=(pos.x, pos.y))
            if len(footprint)
            else (0.0, 0.0)
        )
        return WorldObject(d.color, pos, angle, confidence, size, d.is_obstacle, d, footprint)

    def _select_target(self, objects: list[WorldObject], target_color: str):
        try:
            self.calibration.validate()
        except CalibrationError as exc:
            raise _StageFailure(str(exc)) from exc
        candidates = [o for o in objects if o.color == target_color and not o.is_obstacle]
        if not candidates:
            seen = sorted({o.color for o in objects})
            raise _StageFailure(
                f"no {target_color} object among the detections (saw: {', '.join(seen)})"
            )
        target = max(candidates, key=lambda o: o.size_mm)
        if target.grasp_confidence < self.min_grasp_confidence:
            raise _StageFailure(
                f"the {target_color} object's orientation is not determined "
                f"(psi4 = {target.grasp_confidence:.2f}, need "
                f"{self.min_grasp_confidence:.2f}); it is probably not a square block"
            )
        base = self.calibration.table_to_base.apply(target.position)
        r = float(np.hypot(base.x, base.y))
        if not (self.arm.min_reach <= r <= self.arm.max_reach):
            raise _StageFailure(
                f"{target_color} block is {r:.0f} mm from the arm base, outside its "
                f"reachable annulus [{self.arm.min_reach:.0f}, {self.arm.max_reach:.0f}] mm"
            )
        return target, f"{target_color} at ({target.position.x:.0f}, {target.position.y:.0f}) mm"

    def _plan(self, objects: list[WorldObject], target: WorldObject, q_start):
        # Everything that is not the target is something to avoid, including
        # other blocks of the same colour: knocking one over on the way to the
        # target is still a failure.
        blockers = [o for o in objects if o is not target]
        footprints = [o.footprint_table for o in blockers if len(o.footprint_table)]
        fallback = [
            (o.position, max(8.0, o.size_mm * self.obstacle_radius_scale))
            for o in blockers
            if not len(o.footprint_table)
        ]
        grid = build_grid(
            self.calibration.bounds,
            fallback,
            footprints=footprints,
            resolution_mm=self.grid_resolution_mm,
            clearance_mm=self.clearance_mm,
        )

        q0 = np.zeros(self.arm.n_joints) if q_start is None else np.asarray(q_start, float)
        tip_base = self.arm.fk_point(q0)
        start_table = self.calibration.base_to_table(tip_base)
        if not self.calibration.bounds.contains(start_table):
            raise _StageFailure(
                f"the arm currently rests at ({start_table.x:.0f}, {start_table.y:.0f}) mm, "
                "outside the calibrated workspace"
            )

        try:
            start_cell = grid.nearest_free(grid.to_cell(start_table))
            goal_cell = grid.nearest_free(grid.to_cell(target.position))
        except OutsideGrid as exc:
            raise _StageFailure(f"start or goal is buried in an obstacle: {exc}") from exc

        try:
            plan = astar(grid, start_cell, goal_cell, heuristic="octile")
        except NoPathFound as exc:
            raise _StageFailure(str(exc)) from exc

        cells = shortcut(grid, plan.path_cells)
        corners = [grid.to_world(c) for c in cells]
        # Snap the final waypoint back onto the block itself; the goal cell is
        # only the block's centre rounded to the grid.
        corners[-1] = target.position
        waypoints = resample(corners, self.waypoint_spacing_mm)
        detail = (
            f"{len(plan.path_cells)} cells -> {len(cells)} corners -> "
            f"{len(waypoints)} waypoints, {path_length_mm(waypoints):.0f} mm, "
            f"{plan.expanded} expansions"
        )
        return (grid, plan, waypoints), detail

    def _solve(self, waypoints: list[Point2D], q_start):
        base_waypoints = [self.calibration.table_to_base.apply(p) for p in waypoints]
        try:
            traj = follow_path(self.arm, base_waypoints, q_start=q_start)
        except IKUnreachable as exc:
            raise _StageFailure(str(exc)) from exc
        detail = (
            f"{len(traj)} poses, worst tip error {traj.max_position_error:.2e} mm, "
            f"{traj.branch_flips} branch flips"
        )
        return traj, detail


class _StageFailure(RuntimeError):
    """Internal marker so `_stage` can attribute a failure to its stage."""


def default_calibration(scene, *, table_to_base, bounds, noise_px: float = 0.6):
    """Calibrate against a synthetic scene's fiducials, with clicking noise."""
    pixel_pts, table_pts = scene.calibration_correspondences(noise_px=noise_px)
    return CameraCalibration.from_correspondences(
        pixel_pts, table_pts, table_to_base=table_to_base, bounds=bounds
    )

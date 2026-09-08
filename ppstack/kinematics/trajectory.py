"""Time-scaling and Cartesian path following.

A path is a sequence of positions. A trajectory is a path plus a schedule, and
the difference is where most "the arm jerked" bugs live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import Point2D
from .arm import PlanarArm
from .ik import IKSolution, IKUnreachable, solve


@dataclass
class JointTrajectory:
    times: np.ndarray      # (T,) seconds
    q: np.ndarray          # (T, n) joint angles
    tip: np.ndarray        # (T, 2) achieved tip positions, base frame
    branch_flips: int      # times the elbow branch changed mid-path
    max_position_error: float

    def __len__(self) -> int:
        return len(self.times)


def quintic_scaling(n: int) -> np.ndarray:
    """s(t) on [0, 1] with zero velocity AND zero acceleration at both ends.

    s(u) = 10u^3 - 15u^4 + 6u^5. The zero end-acceleration is the part that
    matters: a trapezoidal or cubic profile leaves a step in acceleration at
    the endpoints, which on real hardware is the jolt you can hear.
    """
    u = np.linspace(0.0, 1.0, n)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def interpolate_cartesian(start: Point2D, goal: Point2D, n: int) -> list[Point2D]:
    """Straight line in task space, quintic in time."""
    start.require(goal.frame)
    s = quintic_scaling(n)
    return [
        Point2D(
            start.x + (goal.x - start.x) * float(si),
            start.y + (goal.y - start.y) * float(si),
            start.frame,
        )
        for si in s
    ]


def follow_path(
    arm: PlanarArm,
    waypoints: list[Point2D],
    *,
    q_start=None,
    duration: float = 2.0,
    branch_flip_tolerance: float = 0.6,
) -> JointTrajectory:
    """Solve IK along a Cartesian path, seeding each solve from the last one.

    Seeding is the whole trick. Solved independently, adjacent waypoints can
    land on different elbow branches, and the arm snaps through a reconfiguration
    between two points a millimetre apart. Carrying the previous solution
    forward as the seed makes the branch choice sticky, and any remaining large
    jump is counted and reported rather than silently executed.
    """
    if len(waypoints) < 2:
        raise ValueError("a path needs at least two waypoints")

    q_prev = np.asarray(q_start, dtype=float) if q_start is not None else None
    qs: list[np.ndarray] = []
    tips: list[tuple[float, float]] = []
    flips = 0
    worst = 0.0

    for i, wp in enumerate(waypoints):
        try:
            sol: IKSolution = solve(arm, wp, q_seed=q_prev)
        except IKUnreachable as exc:
            raise IKUnreachable(
                f"waypoint {i} of {len(waypoints)} at ({wp.x:.1f}, {wp.y:.1f}) mm "
                f"is not solvable: {exc}"
            ) from exc
        # Only mid-path discontinuities count. The step from the rest pose
        # into the first waypoint is expected to be large.
        if i > 0 and float(np.abs(sol.q - q_prev).max()) > branch_flip_tolerance:
            flips += 1
        q_prev = sol.q
        qs.append(sol.q)
        tip = arm.forward(sol.q)
        tips.append((tip.x, tip.y))
        worst = max(worst, sol.position_error)

    return JointTrajectory(
        times=np.linspace(0.0, duration, len(waypoints)),
        q=np.vstack(qs),
        tip=np.array(tips),
        branch_flips=flips,
        max_position_error=worst,
    )


def joint_velocity_limits_ok(traj: JointTrajectory, max_rate: float) -> bool:
    """True when no joint exceeds `max_rate` rad/s anywhere on the trajectory."""
    if len(traj) < 2:
        return True
    dt = np.diff(traj.times)[:, None]
    return bool(np.all(np.abs(np.diff(traj.q, axis=0) / dt) <= max_rate + 1e-9))
